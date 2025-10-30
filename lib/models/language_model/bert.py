import torch
import torch.nn as nn
from typing import Dict, Any

# 旧版 BERT 实现常用的库。你的环境里已经能 import pytorch_pretrained_bert，
# 因为之前训练跑到 batch 500+ 时说明它是可用的。
from pytorch_pretrained_bert import BertModel, BertTokenizer


class NestedTensor:
    """
    保留一个 NestedTensor 壳以保持兼容。
    视觉分支里 NestedTensor(tensors, mask)，mask=True 表示padding。
    文本这边不直接用这个类训练，但其他模块可能会 from bert import NestedTensor，
    所以别删它，避免 ImportError。
    """
    def __init__(self, tensors, mask):
        self.tensors = tensors
        self.mask = mask


class BERT(nn.Module):
    """
    我们自己的语言编码封装器。

    目标：
    - VGST.forward_language() 会像这样调用语言分支：
        outputs = self.language_backbone(
            token_ids=token_ids,
            attention_mask=attn_mask
        )

      所以这里的 forward() 必须显式支持关键字参数 token_ids / attention_mask。

    - dataloader 里给的文本张量是 (L, B) 形状，比如 (256, 1)：
        nl_token_ids:   (256, 1)
        nl_token_masks: (256, 1)
      而 BertModel 期望 (B, L)。
      我们需要自动把 (L,B) 转成 (B,L) 再喂进去。

    - 返回值需要包含：
        "lang_feats": [B, L, hidden_dim]  (hidden_dim=768 for bert-base)
        "lang_mask":  [B, L]
      这两个键在 VGST.forward_language() 里会被用来做 text_proj / 位置编码拼接。
    """

    def __init__(self, bert_model: BertModel, train_bert: bool, enc_num: int):
        super().__init__()
        self.bert_model = bert_model
        self.train_bert = train_bert
        self.enc_num = enc_num  # 我们保留这个字段（用于调试/日志）

        # 如果 config 说不要 finetune BERT，就冻结它的参数
        if not self.train_bert:
            for p in self.bert_model.parameters():
                p.requires_grad = False

    def _batch_first(self, x: torch.Tensor) -> torch.Tensor:
        """
        把 (L,B) 或 (B,L) 统一成 (B,L)。

        策略：
        - 输入必须是 2D: [A,B]
        - 如果 A > B，我们猜它是 (L,B)，就转置成 (B,L)
          例如 (256,1) -> (1,256)
        - 否则就保持不动，假定它已经是 (B,L)

        这样可以兼容你的数据格式 (256,1) 以及一些未来可能的 (B,L) 格式。
        """
        if x.dim() != 2:
            raise ValueError(
                f"BERT.forward expects 2D tensors [*, *], got {tuple(x.shape)}"
            )

        if x.shape[0] > x.shape[1]:
            # 例子: (256,1) -> (1,256)
            x = x.transpose(0, 1).contiguous()
        return x

    def forward(self,
                token_ids: torch.Tensor,
                attention_mask: torch.Tensor) -> Dict[str, Any]:
        """
        Args:
            token_ids:      Tensor (L,B) 或 (B,L), dtype=torch.long
            attention_mask: Tensor (L,B) 或 (B,L), same shape logic

            attention_mask 里通常是 1 表示“这个 token 有效 / 需要关注”，0 表示 padding。
            我们保持这个约定，不在这里取反，保持和上游一致。

        Returns:
            {
              "lang_feats": [B, L, hidden_dim],  # e.g. hidden_dim=768
              "lang_mask":  [B, L]
            }
        """

        # 1. 保证 batch 在第 0 维，即 [B,L]
        input_ids = self._batch_first(token_ids)
        attn_mask = self._batch_first(attention_mask)

        # 2. 调用底层 BertModel
        # pytorch_pretrained_bert.BertModel 的 forward 签名大致是：
        #   sequence_output, pooled_output = bert_model(
        #       input_ids,
        #       token_type_ids=None,
        #       attention_mask=attn_mask,
        #       output_all_encoded_layers=False
        #   )
        #
        # sequence_output: [B,L,hidden_dim]   (hidden_dim=768 for bert-base)
        # pooled_output:   [B,hidden_dim]     (CLS向量)
        sequence_output, pooled_output = self.bert_model(
            input_ids,
            token_type_ids=None,
            attention_mask=attn_mask,
            output_all_encoded_layers=False
        )

        # 3. 打包成 VGST 需要的 dict
        out = {
            "lang_feats": sequence_output,  # [B,L,hidden_dim]
            "lang_mask": attn_mask          # [B,L]
        }
        return out


def _cfg_get(ns, candidates, default=None):
    """
    小工具：从一个 SimpleNamespace/对象 ns 里尝试多个字段名，
    找到第一个存在的就返回，否则返回 default。

    例子：
        model_name = _cfg_get(bert_cfg,
                              ["TYPE", "MODEL", "NAME", "MODEL_NAME", "BERT_TYPE"],
                              "bert-base-uncased")
    """
    for key in candidates:
        if hasattr(ns, key):
            return getattr(ns, key)
    return default


def build_bert(cfg) -> BERT:
    """
    构造语言分支编码器。

    我们现在要做两件非常重要的事：
    1. 兼容你的 cfg 结构（它是 types.SimpleNamespace），但是字段名不一定叫 TYPE。
       有的仓库用 BERT.MODEL / BERT.NAME / BERT.TYPE…… 我们全都尝试。

    2. 返回的对象要满足：
       - .forward(token_ids=..., attention_mask=...) 可调用
       - 输出 dict 里有 "lang_feats" 和 "lang_mask"
       - 挂上一些属性（max_query_len, vocab, num_channels），主模型和上游代码可能会用

    这样主模型 VGST 里的这段就可以直接跑：
        outputs = self.language_backbone(token_ids=token_ids, attention_mask=attn_mask)
        tgt_vl = self.text_proj(outputs["lang_feats"].transpose(0,1))
        mask_vl = outputs["lang_mask"]
        ...
    """

    # 取到语言分支 config namespace
    bert_cfg = cfg.MODEL.LANGUAGE.BERT

    # 1) 读模型名字（优先从 TYPE / MODEL / NAME / MODEL_NAME / BERT_TYPE 这些字段获取）
    bert_type = _cfg_get(
        bert_cfg,
        ["TYPE", "MODEL", "NAME", "MODEL_NAME", "BERT_TYPE"],
        "bert-base-uncased"  # fallback：常用预训练 BERT
    )

    # 2) 是否 finetune BERT (冻结与否)
    train_bert = _cfg_get(
        bert_cfg,
        ["TRAIN_BERT", "FINETUNE", "FINE_TUNE", "TRAIN", "FINE_TUNE_BERT"],
        False  # fallback: 默认不finetune，省内存/显存
    )

    # 3) 记录用到的层数/层号 (有的仓库叫 ENC_NUM)
    enc_num = _cfg_get(
        bert_cfg,
        ["ENC_NUM", "NUM_LAYERS", "LAYERS"],
        12  # fallback: bert-base 有12层
    )

    # 4) 最大文本长度（dataloader 通常会用这个截断）
    max_query_len = _cfg_get(
        bert_cfg,
        ["MAX_QUERY_LEN", "MAX_LEN", "MAX_LENGTH", "MAX_TOKENS"],
        256  # fallback: 256，对应你日志里 nl_token_ids 长度=256
    )

    # 5) BERT hidden size，用来告诉主干后面线性层该接多少输入通道
    hidden_dim = _cfg_get(
        bert_cfg,
        ["HIDDEN_DIM", "DIM", "OUTPUT_DIM", "HIDDEN_SIZE"],
        768  # fallback: bert-base hidden dim
    )

    # === 真正构建 tokenizer / bert_model ===
    # 这会从本地 cache 里加载 'bert_type' 对应的预训练权重。
    # 你的环境之前已经成功跑到 batch 500+，说明 pytorch_pretrained_bert
    # 以及对应的权重是可用/可加载的，所以我们继续用这一套。
    tokenizer = BertTokenizer.from_pretrained(bert_type)
    bert_model = BertModel.from_pretrained(bert_type)

    # === 包装成我们上面定义的 BERT 类（带 forward(token_ids=..., attention_mask=...)）===
    bert_backbone = BERT(
        bert_model=bert_model,
        train_bert=train_bert,
        enc_num=enc_num
    )

    # === 给上游（VGST / 数据管线）可能用到的属性 ===
    # 这些字段名和之前版本保持一致，避免 AttributeError：
    bert_backbone.max_query_len = max_query_len      # dataloader 可用来截断
    bert_backbone.vocab = tokenizer.vocab            # 方便推理/调试
    bert_backbone.num_channels = hidden_dim          # 语言特征原始通道（通常=768）

    return bert_backbone
