import copy
import torch
from torch import nn, Tensor
from typing import Optional

###############################################################################
# 基础工具
###############################################################################


def _get_clones(module, N):
    """复制N份模块，用于堆叠多层encoder/decoder。"""
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def _get_activation_fn(activation):
    """与 DETR 一致的激活函数选择。"""
    if activation == "relu":
        return nn.ReLU(inplace=True)
    if activation == "gelu":
        return nn.GELU()
    if activation == "glu":
        return nn.GLU()
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")


###############################################################################
# Encoder / Decoder Layer 定义
###############################################################################


class TransformerEncoderLayer(nn.Module):
    """
    标准 Transformer Encoder Layer：
    - self-attn
    - FFN
    与 DETR 一致，接受显式 pos 编码
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="relu",
                 normalize_before=False):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # Feed Forward
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, src,
                     src_mask: Optional[Tensor] = None,
                     src_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None):
        # 自注意力
        q = k = self.with_pos_embed(src, pos)
        src2 = self.self_attn(
            q, k, value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # FFN
        src2 = self.linear2(
            self.dropout(self.activation(self.linear1(src)))
        )
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src

    def forward_pre(self, src,
                    src_mask: Optional[Tensor] = None,
                    src_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None):
        # LayerNorm 放前面
        src2 = self.norm1(src)
        q = k = self.with_pos_embed(src2, pos)
        src2 = self.self_attn(
            q, k, value=src2,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )[0]
        src = src + self.dropout1(src2)

        src2 = self.norm2(src)
        src2 = self.linear2(
            self.dropout(self.activation(self.linear1(src2)))
        )
        src = src + self.dropout2(src2)
        return src

    def forward(self, src,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(src, src_mask, src_key_padding_mask, pos)
        return self.forward_post(src, src_mask, src_key_padding_mask, pos)


class TransformerDecoderLayer(nn.Module):
    """
    标准 Transformer Decoder Layer：
    - self-attn (对queries)
    - cross-attn (queries attend to memory)
    - FFN
    """
    def __init__(self, d_model, nhead, dim_feedforward=2048,
                 dropout=0.1, activation="relu", normalize_before=False):
        super().__init__()

        # self-attn for target queries
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        # cross-attn: query 与 encoder memory 交互
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        # FFN
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        # Norm + Dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     tgt_mask: Optional[Tensor] = None,
                     memory_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        # 1) query 自注意力
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)

        # 2) cross-attention 到 encoder memory
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)

        # 3) FFN
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout3(tgt2)
        tgt = self.norm3(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    tgt_mask: Optional[Tensor] = None,
                    memory_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        # 与 forward_post 类似，只是 LayerNorm 前置
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(
            q, k, value=tgt2,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)

        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(
            query=self.with_pos_embed(tgt2, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask
        )[0]
        tgt = tgt + self.dropout2(tgt2)

        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)

        return tgt

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(
                tgt, memory, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask,
                pos, query_pos
            )
        return self.forward_post(
            tgt, memory, tgt_mask, memory_mask,
            tgt_key_padding_mask, memory_key_padding_mask,
            pos, query_pos
        )


class TransformerEncoder(nn.Module):
    """多层堆叠的 Encoder"""
    def __init__(self, encoder_layer, num_layers, norm=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(self, src,
                mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None):
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class TransformerDecoder(nn.Module):
    """多层堆叠的 Decoder，支持返回所有中间层输出"""
    def __init__(self, decoder_layer, num_layers, norm=None,
                 return_intermediate=False):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(self, tgt, memory,
                tgt_mask: Optional[Tensor] = None,
                memory_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        output = tgt
        intermediate = []

        for layer in self.layers:
            output = layer(
                output, memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)

        if self.return_intermediate:
            # [num_layers, B, Nq, C]
            return torch.stack(intermediate)

        # [1, B, Nq, C]
        return output.unsqueeze(0)


###############################################################################
# 总的 Transformer: 编码 color / depth / language + 融合 + 解码
###############################################################################


class Transformer(nn.Module):
    """
    多模态 Transformer
    - 对 color / depth / language 的序列分别编码
    - 融合 (fusion encoder)
    - decoder 用 object queries 预测目标

    重要: 这里新增 _sanitize_pad_mask()，保证不会把“整句全是padding(True)”
    的 mask 直接喂给 MultiheadAttention，否则 softmax(-inf,...)
    会变成 NaN，最后导致 pred_boxes 里出现 NaN 并触发
    VGSTActor 里的报错。:contentReference[oaicite:7]{index=7}
    """
    def __init__(self, d_model=256, dropout=0.1, nhead=8,
                 dim_feedforward=2048, num_encoder_layers=6,
                 num_fusion_layers=4, num_decoder_layers=6,
                 normalize_before=False, divide_norm=False,
                 return_intermediate_dec=False):
        super().__init__()

        # ===== 三个 encoder: color / depth / language =====
        encoder_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            "relu", normalize_before
        )
        encoder_norm = nn.LayerNorm(d_model) if normalize_before else None

        self.encoder_color = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )
        self.encoder_depth = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )
        self.encoder_vl = TransformerEncoder(
            encoder_layer, num_encoder_layers, encoder_norm
        )

        # ===== 融合阶段 =====
        # color+depth 或 language+cd_fused 拼接后通道是 2*d_model
        # 我们用 1x1 conv(neck_layer) 把通道压回 d_model
        self.neck_layer = nn.Conv1d(
            in_channels=2 * d_model,
            out_channels=d_model,
            kernel_size=1
        )

        fusion_layer = TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            "relu", normalize_before
        )
        fusion_norm = nn.LayerNorm(d_model) if normalize_before else None
        self.fusion = TransformerEncoder(
            fusion_layer, num_fusion_layers, fusion_norm
        )

        # ===== decoder =====
        decoder_layer = TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            "relu", normalize_before
        )
        decoder_norm = nn.LayerNorm(d_model)

        self.decoder = TransformerDecoder(
            decoder_layer,
            num_decoder_layers,
            decoder_norm,
            return_intermediate=return_intermediate_dec
        )

        self.d_model = d_model
        self.nhead = nhead
        self.divide_norm = divide_norm

        self._reset_parameters()

    def _reset_parameters(self):
        # Xavier init
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    @staticmethod
    def _sanitize_pad_mask(mask: Optional[Tensor]) -> Optional[Tensor]:
        """
        关键修复:
        对 key_padding_mask 做安全处理，避免 "全 True" 的情况导致
        MultiheadAttention NaN。

        输入:
            mask: [B,S] bool, True 表示该 token 是 padding/无效。
                  如果是 [B,H,W]，我们会展平成 [B,H*W]。
        返回:
            safe_mask: [B,S] bool，保证每个样本至少有一个 False。
        """
        if mask is None:
            return None

        m = mask.to(torch.bool)
        if m.dim() > 2:
            # e.g. [B,H,W] -> [B,H*W]
            m = m.flatten(1)

        # 找所有“整句全是 padding(True)”的样本
        all_pad = m.all(dim=1)  # [B]
        if all_pad.any():
            m = m.clone()
            # 对这些样本，强行把第0个token设成False，让注意力至少有1个合法位置
            m[all_pad, 0] = False

        return m

    def forward(
        self,
        seq_dict_color,
        seq_dict_depth,
        seq_dict_language,
        query_embed: Tensor,
        mode="all",
        return_encoder_output=False
    ):
        """
        Args:
            seq_dict_color/depth/language:
                {
                  "feat": [S,B,C],
                  "mask": [B,S] bool,
                  "pos" : [S,B,C]
                }
            query_embed: [Nq,C] 或 [Nq,B,C]

        Return:
            如果 mode == "encoder": 只返回融合后的 memory
            否则: 返回 decoder 输出的 hs 以及 (可选) encoder memory
        """

        # 0) 对 key_padding_mask 做安全修正，避免全 True => NaN
        color_mask = self._sanitize_pad_mask(seq_dict_color["mask"])
        depth_mask = self._sanitize_pad_mask(seq_dict_depth["mask"])
        lang_mask  = self._sanitize_pad_mask(seq_dict_language["mask"])

        # 1) 单模态编码
        enc_color = self.encoder_color(
            seq_dict_color["feat"],
            src_key_padding_mask=color_mask,
            pos=seq_dict_color["pos"]
        )  # [Sc,B,C]

        enc_depth = self.encoder_depth(
            seq_dict_depth["feat"],
            src_key_padding_mask=depth_mask,
            pos=seq_dict_depth["pos"]
        )  # [Sd,B,C]

        enc_lang = self.encoder_vl(
            seq_dict_language["feat"],
            src_key_padding_mask=lang_mask,
            pos=seq_dict_language["pos"]
        )  # [Sl,B,C]

        # 2) color & depth 融合
        # enc_color, enc_depth: [S,B,C] -> cat->[S,B,2C]
        assert enc_color.shape[0] == enc_depth.shape[0], \
            "color/depth token长度不一致，这里假设同尺寸特征图"
        cd_cat = torch.cat([enc_color, enc_depth], dim=2)  # [S,B,2C]

        # neck_layer 需要 [B,2C,S] -> [B,C,S]
        cd_cat_conv = self.neck_layer(cd_cat.permute(1, 2, 0))
        cd_fused = cd_cat_conv.permute(2, 0, 1)  # [S,B,C]

        cd_fused = self.fusion(
            cd_fused,
            src_key_padding_mask=color_mask,
            pos=seq_dict_color["pos"]
        )  # [S,B,C]

        # 3) language + (color+depth) 再融合一次
        S_cd = cd_fused.shape[0]
        lang_tail = enc_lang[-S_cd:]                 # 取语言最后 S_cd 个 token
        vdl_cat = torch.cat([lang_tail, cd_fused], dim=2)  # [S_cd,B,2C]

        vdl_cat_conv = self.neck_layer(vdl_cat.permute(1, 2, 0))
        memory = vdl_cat_conv.permute(2, 0, 1)        # [S_cd,B,C]

        memory = self.fusion(
            memory,
            src_key_padding_mask=color_mask,
            pos=seq_dict_color["pos"]
        )  # [S_cd,B,C]

        if mode == "encoder":
            # 只要encoder memory
            return memory

        # 4) decoder
        # query_embed: [Nq,C] 或 [Nq,B,C]
        if query_embed.dim() == 2:
            # 变成[Nq,B,C]
            B = memory.shape[1]
            query_embed = query_embed.unsqueeze(1).repeat(1, B, 1)

        # decoder 的初始 query 特征，全0
        tgt = torch.zeros_like(query_embed)

        hs = self.decoder(
            tgt, memory,
            memory_key_padding_mask=color_mask,
            pos=seq_dict_color["pos"],
            query_pos=query_embed
        )  # [num_layers,B,Nq,C] or [1,B,Nq,C]

        if return_encoder_output:
            return hs, memory

        return hs


def build_transformer(cfg):
    """
    根据 cfg 生成上述 Transformer。
    需要字段：
        cfg.MODEL.HIDDEN_DIM
        cfg.MODEL.TRANSFORMER.DROPOUT
        cfg.MODEL.TRANSFORMER.NHEADS
        cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD
        cfg.MODEL.TRANSFORMER.ENC_LAYERS
        cfg.MODEL.TRANSFORMER.FUS_LAYERS
        cfg.MODEL.TRANSFORMER.DEC_LAYERS
        cfg.MODEL.TRANSFORMER.PRE_NORM
        cfg.MODEL.TRANSFORMER.DIVIDE_NORM
        cfg.MODEL.TRANSFORMER.RETURN_INTERMEDIATE_DEC
    """
    return Transformer(
        d_model=cfg.MODEL.HIDDEN_DIM,
        dropout=cfg.MODEL.TRANSFORMER.DROPOUT,
        nhead=cfg.MODEL.TRANSFORMER.NHEADS,
        dim_feedforward=cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD,
        num_encoder_layers=cfg.MODEL.TRANSFORMER.ENC_LAYERS,
        num_fusion_layers=cfg.MODEL.TRANSFORMER.FUS_LAYERS,
        num_decoder_layers=cfg.MODEL.TRANSFORMER.DEC_LAYERS,
        normalize_before=cfg.MODEL.TRANSFORMER.PRE_NORM,
        divide_norm=cfg.MODEL.TRANSFORMER.DIVIDE_NORM,
        return_intermediate_dec=cfg.MODEL.TRANSFORMER.RETURN_INTERMEDIATE_DEC
    )
