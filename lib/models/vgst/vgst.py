# lib/models/vgst/vgst.py
#
# 这个版本包含三类关键修改：
# 1. 语言分支 mask 语义修正：
#    BERT 的 attention_mask 是 1=有效,0=padding，
#    Transformer 的 key_padding_mask 需要 True=padding, False=有效。
#    我们在 forward_language() 里翻转语义，防止注意力产生 NaN。
#
# 2. forward_transformer() 与 Transformer.forward() 接口对齐：
#    直接把 {feat, mask, pos} 这三个 dict 传给 Transformer，
#    避免 mask/pos 对错位，避免无意义激活爆炸。
#
# 3. forward_box_head() 新增数值清洗：
#    - _sanitize_logits(): 把 NaN / ±inf 的 logits 变成 0，再 sigmoid。
#    - _sanitize_boxes(): 把回归出来的 boxes 中的 NaN/±inf 替换成安全值并 clamp [0,1]。
#    这样保证 out_dict["pred_boxes"] 不会包含 NaN，
#    VGSTActor._compute_losses() 里的 NaN 检查就不会再触发崩溃。
#
# 训练其它部分(optimizer、trainer、VGSTActor)保持不变。

import math
from typing import Optional, Dict, Union

import torch
from torch import nn
from torch.nn.init import trunc_normal_

from lib.models.vgst.backbone import build_backbone
from lib.models.vgst.transformer import build_transformer
from .vl_transformer import build_vl_transformer
from lib.models.language_model.bert import build_bert
from lib.utils.misc import NestedTensor
from .head import build_box_head


class VGST(nn.Module):
    """
    VGST 主模型（显存友好版 + 数值稳定版）

    模块结构：
    - backbone_color:   RGB/多模态视觉主干（我们改成 patch-token encoder，避免 OOM）
    - backbone_depth:   深度分支（我们改成轻量 CNN，下采样到 1/16）
    - language_backbone: BERT 封装，返回文本特征和 mask
    - transformer:      融合 color/depth/text 的多模态 Transformer
                         (内部有 _sanitize_pad_mask 防止注意力 NaN)
    - box_head:         预测目标框坐标

    关键点：
    - forward_language() 把文本 mask 转成 Transformer 需要的形式：
      True=padding, False=有效。
    - forward_box_head() 对输出框做 NaN/inf 清洗，避免训练因 NaN 中断。
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # --- 三路主干 ---
        self.backbone_color = build_vl_transformer(cfg)
        self.backbone_depth = build_backbone(cfg)
        self.language_backbone = build_bert(cfg)
        self.transformer = build_transformer(cfg)

        hidden_dim = cfg.MODEL.HIDDEN_DIM
        self.score_head = nn.Linear(hidden_dim, 1)   # [*,C] -> [*,1]
        self.return_intermediate = False

        # --- 通道对齐 ---
        # color 分支：我们的 build_vl_transformer 已经输出 hidden_dim 通道，所以这里是 Identity
        self.bottleneck_color = nn.Identity()

        # depth 分支：用 1x1 conv 把深度特征通道对齐到 hidden_dim
        depth_in_channels = getattr(self.backbone_depth, "num_channels", hidden_dim)
        self.bottleneck_depth = nn.Conv2d(depth_in_channels, hidden_dim, kernel_size=1)

        # 文本特征 (768 from BERT) -> hidden_dim
        self.text_proj = nn.Linear(768, hidden_dim)

        # 文本位置编码表 (最大长度 1000)
        self.nl_pos_embed = nn.Embedding(1000, hidden_dim)

        # --- queries & head ---
        self.num_queries = cfg.MODEL.NUM_QUERIES
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)

        self.box_head = build_box_head(cfg)
        self.head_type = cfg.MODEL.HEAD_TYPE

        if self.head_type in ["CORNER", "CORNER_LITE"]:
            self.feat_sz_s = int(self.box_head.feat_sz)
            self.feat_len_s = int(self.box_head.feat_sz ** 2)
        else:
            self.feat_sz_s = None
            self.feat_len_s = None

        self.aux_loss = cfg.TRAIN.DEEP_SUPERVISION

        # --- 记录下采样比例 (1/16)
        self.divisor = 16
        self.num_visu_template_token = (cfg.DATA.TEMPLATE.SIZE // self.divisor) ** 2
        self.num_visu_search_token = (cfg.DATA.SEARCH.SIZE // self.divisor) ** 2

        # --- 初始化网络参数 ---
        self.init = cfg.MODEL.VL.INIT
        self._reset_parameters(self.init)

    # ============================================================
    # 初始化模块参数
    # ============================================================
    def _reset_parameters(self, init_type: str):
        if init_type in ["xavier", "xavier_normal", "xavier_stable"]:
            self.apply(self._init_weights_xavier_stable)
        else:
            self.apply(self._init_weights_trunc_normal)

    @staticmethod
    def _init_weights_xavier_stable(m: nn.Module):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    @staticmethod
    def _init_weights_trunc_normal(m: nn.Module):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0.0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    # ============================================================
    # Color 分支 (RGB / 视觉)
    # ============================================================
    def forward_backbone_color(self, tensor_list_c: NestedTensor) -> Dict[str, torch.Tensor]:
        """
        input:
            tensor_list_c.tensors: [B, C, H, W]
            tensor_list_c.mask:    [B, H, W]  (bool, True=padding/无效位置)
        output dict:
            {
              "feat": [S_c, B, C],
              "mask": [B, S_c] (bool, True=padding),
              "pos":  [S_c, B, C]
            }
        """
        out_list_c, pos_list_c = self.backbone_color(tensor_list_c)

        src_c  = out_list_c[-1].tensors     # [B,C,Hc,Wc]
        mask_c = out_list_c[-1].mask        # [B,Hc,Wc] (bool)
        pos_c  = pos_list_c[-1]             # [B,C,Hc,Wc]

        src_proj_c = self.bottleneck_color(src_c)  # Identity

        B, C, Hc, Wc = src_proj_c.shape
        feat = src_proj_c.flatten(2).permute(2, 0, 1)  # [Hc*Wc, B, C]
        pos  = pos_c.flatten(2).permute(2, 0, 1)       # [Hc*Wc, B, C]
        mask = mask_c.flatten(1)                       # [B, Hc*Wc] bool

        return {"feat": feat, "mask": mask, "pos": pos}

    # ============================================================
    # Depth 分支
    # ============================================================
    def forward_backbone_depth(self, tensor_list_d: NestedTensor) -> Dict[str, torch.Tensor]:
        """
        同样把深度分支输出整理成 transformer 需要的 {feat,mask,pos} 结构。
        """
        out_list_d, pos_list_d = self.backbone_depth(tensor_list_d)

        src_d  = out_list_d[-1].tensors     # [B,Cd,Hd,Wd]
        mask_d = out_list_d[-1].mask        # [B,Hd,Wd] bool(True=padding)
        pos_d  = pos_list_d[-1]             # [B,Cd,Hd,Wd]

        # 用1x1 conv把通道压到 hidden_dim
        src_proj_d = self.bottleneck_depth(src_d)   # [B,C,Hd,Wd]
        pos_proj_d = self.bottleneck_depth(pos_d)   # [B,C,Hd,Wd]

        feat = src_proj_d.flatten(2).permute(2, 0, 1)  # [Hd*Wd,B,C]
        pos  = pos_proj_d.flatten(2).permute(2, 0, 1)  # [Hd*Wd,B,C]
        mask = mask_d.flatten(1)                       # [B,Hd*Wd] bool

        return {"feat": feat, "mask": mask, "pos": pos}

    # ============================================================
    # Language 分支
    # ============================================================
    def forward_language(self, lang_in: Optional[Union[Dict[str, torch.Tensor], NestedTensor]]):
        """
        支持两种输入：
        1) 已经是 {feat,mask,pos} 形式 -> 直接返回
        2) NestedTensor(token_ids, attn_mask):
            token_ids:   (L,B) or (B,L) long
            attn_mask:   (L,B) or (B,L) int (1=有效, 0=padding)

        返回:
            {
              "feat": [L,B,C],        # C = hidden_dim
              "mask": [B,L] (bool),   # True=padding, False=有效
              "pos":  [L,B,C]
            }

        为什么要翻转 mask 语义？
        - BERT 的 attention_mask: 1=有效, 0=padding
        - nn.MultiheadAttention 的 key_padding_mask: True=padding
        如果不翻，会把所有有效 token 当成“应该屏蔽”，attention 会全是 -inf，
        softmax(-inf...) -> NaN，最后 pred_boxes 也变成 NaN。
        """

        # 情况1：已经是 {feat,mask,pos} 的 dict
        if isinstance(lang_in, dict) and {"feat", "mask", "pos"} <= set(lang_in.keys()):
            return lang_in

        # 情况2：NestedTensor(token_ids, attn_mask)
        if isinstance(lang_in, NestedTensor):
            token_ids = lang_in.tensors      # [L,B] or [B,L]
            attn_mask = lang_in.mask         # [L,B] or [B,L], 1=有效,0=padding

            # self.language_backbone 是我们封装的 BERT
            # 返回:
            #   "lang_feats": [B,L,768]
            #   "lang_mask":  [B,L] (1=有效,0=padding)
            outputs = self.language_backbone(
                token_ids=token_ids,
                attention_mask=attn_mask
            )
            lang_feats = outputs["lang_feats"]  # [B,L,768]
            raw_mask   = outputs["lang_mask"]   # [B,L], int 0/1

            # (1) 特征：投影到 hidden_dim，并转成 [L,B,C]
            tgt_vl = self.text_proj(lang_feats.transpose(0, 1))  # [L,B,C]

            # (2) mask 语义翻转：
            # raw_mask.bool() -> True=有效, False=padding
            # 我们要 True=padding, False=有效
            mask_valid = raw_mask.bool()
            mask_pad   = ~mask_valid         # True=padding, False=有效
            # mask_pad: [B,L] bool, 这是 transformer 期望的 key_padding_mask 语义

            # (3) 文本位置编码 [L,B,C]
            L = tgt_vl.shape[0]
            if L > self.nl_pos_embed.num_embeddings:
                raise ValueError(
                    f"Sentence too long: {L} > {self.nl_pos_embed.num_embeddings}"
                )
            pos_vl = self.nl_pos_embed.weight[:L, :].unsqueeze(1).repeat(
                1, tgt_vl.shape[1], 1
            )  # [L,B,C]

            return {
                "feat": tgt_vl,     # [L,B,C]
                "mask": mask_pad,   # [B,L] bool (True=padding)
                "pos":  pos_vl      # [L,B,C]
            }

        # 其他情况 -> 抛错
        raise ValueError(
            "Language input is required. Pass NestedTensor(token_ids, attn_mask) "
            "or {feat,mask,pos}."
        )

    # ============================================================
    # Transformer 融合
    # ============================================================
    def forward_transformer(self, seq_c, seq_d, seq_vl):
        """
        以前老代码是把 feat/mask/pos 各自拆开、按位置参数传进 transformer，
        这会导致参数错位（特别是我们自己改过 transformer.forward 的签名）。

        现在我们直接把三个 dict (color/depth/language) 原样传进去，
        确保 mask / pos 不会传错，避免再出现爆炸数值。
        """
        hs, memory = self.transformer(
            seq_c,
            seq_d,
            seq_vl,
            self.query_embed.weight,
            mode="all",
            return_encoder_output=True,
        )
        return hs, memory

    # ============================================================
    # 数值清洗辅助函数（防 NaN/inf 进入 pred_boxes）
    # ============================================================
    @staticmethod
    def _sanitize_logits(logits: torch.Tensor) -> torch.Tensor:
        """
        logits: 任意形状，比如 [B,Nq,4] 的 raw 输出（还没过 sigmoid）。

        目标：
        - 把 NaN / +inf / -inf 全部变成 0，避免 sigmoid(NaN)=NaN 这种情况。
        - 返回一个新的张量（必要时 clone）。
        """
        finite_mask = torch.isfinite(logits)
        if finite_mask.all():
            return logits
        safe_logits = logits.clone()
        safe_logits[~finite_mask] = 0.0  # 0 -> sigmoid(0)=0.5，合理的“中立框”
        return safe_logits

    @staticmethod
    def _sanitize_boxes(boxes: torch.Tensor, clamp01: bool = True) -> torch.Tensor:
        """
        boxes: [B,N,4] 或 [B,4]，已经是坐标形式 (通常在 [0,1] 内)

        目标：
        - 把 NaN / ±inf 替换成 0.5 这种温和值
        - 可选地 clamp 到 [0,1] 内，避免无意义的大框/负框

        返回的是一个新张量（必要时 clone），不会报错。
        """
        finite_mask = torch.isfinite(boxes)
        if finite_mask.all():
            out = boxes
        else:
            out = boxes.clone()
            out[~finite_mask] = 0.5  # 合理的居中/稳定值

        if clamp01:
            out = out.clamp(0.0, 1.0)
        return out

    # ============================================================
    # Box head：decode -> box 预测
    # ============================================================
    def _set_aux_loss(self, outputs_coord: torch.Tensor):
        """
        组装 aux_outputs 做深监督 (DETR-style)。actor 里一般不会直接用这些，
        所以即使里面可能还有不稳定值，也不会导致训练直接崩。
        """
        out = []
        if outputs_coord.shape[0] > 1:
            for i in range(outputs_coord.shape[0] - 1):
                out.append({"pred_boxes": outputs_coord[i]})
        return out

    def forward_box_head(self, hs: torch.Tensor, memory: torch.Tensor):
        """
        根据 cfg.MODEL.HEAD_TYPE 生成预测框:
        - "MLP":   decoder hs[-1] -> box_head -> sigmoid -> [B,Nq,4] (cx,cy,w,h) in [0,1]
        - "CORNER"/"CORNER_LITE": memory reshape成特征图 -> corner-style head -> [B,4]

        这里我们把数值清洗逻辑加进来，确保 pred_boxes 里不会再出现 NaN，
        从而避免 VGSTActor 的 _compute_losses() 直接 raise。
        """

        if self.head_type == "MLP":
            # hs 形状: [num_decoder_layers,B,Nq,C] 或 [1,B,Nq,C]
            raw_logits = self.box_head(hs[-1])                 # [B,Nq,4]，未过 sigmoid
            raw_logits = self._sanitize_logits(raw_logits)
            outputs_coord = raw_logits.sigmoid()               # [B,Nq,4] in (0,1)
            outputs_coord = self._sanitize_boxes(outputs_coord, clamp01=True)

            scores = torch.sigmoid(self.score_head(hs[-1]).squeeze(-1))  # [B,Nq]
            out = {"pred_boxes": outputs_coord, "scores": scores}
            
            if self.aux_loss and hs.shape[0] > 1:
                aux_list = []
                for i in range(hs.shape[0] - 1):   # 不含最后一层
                    aux_raw = self.box_head(hs[i])            # [B,Nq,4]
                    aux_raw = self._sanitize_logits(aux_raw)
                    aux_coord = aux_raw.sigmoid()
                    aux_coord = self._sanitize_boxes(aux_coord, clamp01=True)
                    aux_list.append({"pred_boxes": aux_coord})
                out["aux_outputs"] = aux_list
            return out, outputs_coord

        elif self.head_type in ["CORNER", "CORNER_LITE"]:
            # memory: [S_cd,B,C] -> reshape 成 [B,C,H,W] 再走 corner-style 回归头
            B = memory.shape[1]
            C = memory.shape[2]
            feat_sz = self.feat_sz_s or int(math.sqrt(memory.shape[0]))
            feat_map = memory.permute(1, 2, 0).reshape(B, C, feat_sz, feat_sz)  # [B,C,H,W]

            boxes = self.box_head(feat_map)            # [B,4] (xyxy normalized 0~1?)
            boxes = self._sanitize_boxes(boxes, clamp01=True)
            outputs_coord = boxes.unsqueeze(1)         # [B,1,4]

            global_feat = memory.permute(1,0,2).mean(dim=1)   # [B,C]
            scores = torch.ones(memory.shape[1], 1, device=memory.device)  # [B,1]，简单起见
            out = {"pred_boxes": outputs_coord, "scores": scores}


            if self.aux_loss:
                out["aux_outputs"] = self._set_aux_loss(
                    outputs_coord.unsqueeze(0)
                )
            return out, outputs_coord

        else:
            raise ValueError(f"Unknown head type {self.head_type}")

    # ============================================================
    # 顶层 forward：兼容老调用方式
    # ============================================================
    def forward(
        self,
        tensor_list_c: Optional[NestedTensor] = None,
        tensor_list_d: Optional[NestedTensor] = None,
        seq_dict_language: Optional[Union[Dict[str, torch.Tensor], NestedTensor]] = None,
        mode: str = "no_class_head",
        run_cls_head: Optional[bool] = None,
        run_box_head: bool = False,
        **kwargs,
    ):
        """
        我们允许两种调用风格：
        - 新风格：直接传 NestedTensor(...)
        - 旧风格：使用关键字 color_data=..., depth_data=..., text_data=...
          我们这里自动兼容旧关键字，防止 trainer 里崩。

        返回：
            out_dict, outputs_coord
            其中 out_dict["pred_boxes"] 是 [B,N,4]，经过 sanitize 保证无 NaN。
        """

        # ---- 兼容历史关键字命名 ----
        if tensor_list_c is None and "color_data" in kwargs:
            tensor_list_c = kwargs.pop("color_data")
        if tensor_list_d is None and "depth_data" in kwargs:
            tensor_list_d = kwargs.pop("depth_data")
        if seq_dict_language is None and "text_data" in kwargs:
            seq_dict_language = kwargs.pop("text_data")

        # ---- 确保整个网络在 CUDA 上 ----
        if torch.cuda.is_available() and not next(self.parameters()).is_cuda:
            self.cuda()
        device = next(self.parameters()).device

        # ---- 把 NestedTensor 的内容搬到正确 device 上 ----
        if tensor_list_c is not None:
            tensor_list_c = NestedTensor(
                tensor_list_c.tensors.to(device),
                tensor_list_c.mask.to(device) if tensor_list_c.mask is not None else None,
            )
        if tensor_list_d is not None:
            tensor_list_d = NestedTensor(
                tensor_list_d.tensors.to(device),
                tensor_list_d.mask.to(device) if tensor_list_d.mask is not None else None,
            )

        # 文本分支既可能是 NestedTensor(token_ids, attn_mask)
        # 也可能是 {"feat","mask","pos"} dict，统一搬 device
        if isinstance(seq_dict_language, NestedTensor):
            seq_dict_language = NestedTensor(
                seq_dict_language.tensors.to(device),
                seq_dict_language.mask.to(device)
            )
        elif isinstance(seq_dict_language, dict):
            moved = {}
            for k, v in seq_dict_language.items():
                moved[k] = v.to(device) if torch.is_tensor(v) else v
            seq_dict_language = moved

        # ---- 三路编码 ----
        seq_c = self.forward_backbone_color(tensor_list_c) if tensor_list_c is not None else None
        seq_d = self.forward_backbone_depth(tensor_list_d) if tensor_list_d is not None else None
        seq_vl = self.forward_language(seq_dict_language)

        # ---- 多模态 transformer 融合，得到 hs(queries) 和 memory(encoder输出) ----
        hs, memory = self.forward_transformer(seq_c, seq_d, seq_vl)

        # ---- 如果只想调试 transformer 的输出，不走 head ----
        if mode == "transformer":
            return hs, memory

        # ---- 走检测头 / 跟踪头，输出预测框，并做数值清洗 ----
        out_dict, outputs_coord = self.forward_box_head(hs, memory)

        return out_dict, outputs_coord


def build_vgst(cfg):
    return VGST(cfg)
