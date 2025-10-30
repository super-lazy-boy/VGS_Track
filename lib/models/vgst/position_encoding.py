# position_encoding.py
# 为特征图生成2D位置编码：DETR风格的正弦位置编码(default)，
# 或可学习位置编码。
#
# cfg链接：
#   cfg.MODEL.POSITION_EMBEDDING   # "sine" / "learned" / "none"
#   cfg.MODEL.HIDDEN_DIM           # transformer通道维度C
#
# build_position_encoding(cfg) 会根据这些字段返回 PositionEmbeddingSine /
# PositionEmbeddingLearned / PositionEmbeddingNone。

import math
import torch
from torch import nn
import torch.nn.functional as F

from lib.utils.misc import NestedTensor


class PositionEmbeddingSine(nn.Module):
    """
    与 DETR 一致的 2D 正弦/余弦位置编码。
    输入: NestedTensor(tensor[B,C,H,W], mask[B,H,W])  mask=True表示padding
    输出: pos[B, 2*num_pos_feats, H, W]
    """

    def __init__(self, num_pos_feats=64, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats  # 每个坐标轴一半维度
        self.temperature = temperature
        self.normalize = normalize

        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = scale or 2 * math.pi

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None

        not_mask = ~mask  # 有效区域
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats, dtype=torch.float32, device=x.device
        )
        dim_t = self.temperature ** (
            2 * (dim_t // 2) / self.num_pos_feats
        )

        pos_x = x_embed[:, :, :, None] / dim_t  # (B,H,W,num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_t

        # 交替 sin/cos
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    可学习的位置编码。对(H,W)中每个坐标分配一个可学习向量。
    用于固定分辨率下的训练/推理。
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)

        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat(
            [
                x_emb.unsqueeze(0).repeat(h, 1, 1),
                y_emb.unsqueeze(1).repeat(1, w, 1),
            ],
            dim=-1,
        ).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


class PositionEmbeddingNone(nn.Module):
    """
    不使用位置信息，直接返回0张量。
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.num_pos_feats = num_pos_feats

    def forward(self, tensor_list: NestedTensor):
        x = tensor_list.tensors
        return torch.zeros(
            x.shape[0],
            self.num_pos_feats * 2,
            x.shape[-2],
            x.shape[-1],
            device=x.device,
            dtype=x.dtype,
        )


def build_position_encoding(cfg):
    """
    根据 cfg 构建位置编码模块。

    cfg.MODEL.HIDDEN_DIM:
        Transformer的隐藏通道数C。例如256。
    cfg.MODEL.POSITION_EMBEDDING:
        'sine'   -> DETR式正弦位置编码 (推荐，通用)
        'learned'-> 可学习位置编码
        'none'   -> 不加位置信息（主要调试用）

    返回:
        一个 nn.Module，forward(NestedTensor) -> pos(B, C, H, W)
    """
    num_pos_feats = cfg.MODEL.HIDDEN_DIM // 2

    if cfg.MODEL.POSITION_EMBEDDING in ("v2", "sine"):
        position_embedding = PositionEmbeddingSine(
            num_pos_feats, normalize=True
        )
    elif cfg.MODEL.POSITION_EMBEDDING in ("v3", "learned"):
        position_embedding = PositionEmbeddingLearned(num_pos_feats)
    elif cfg.MODEL.POSITION_EMBEDDING in ("none",):
        position_embedding = PositionEmbeddingNone(num_pos_feats)
    else:
        raise ValueError(
            f"Unsupported POSITION_EMBEDDING {cfg.MODEL.POSITION_EMBEDDING}"
        )

    return position_embedding
