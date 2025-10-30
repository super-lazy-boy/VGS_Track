import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils.misc import NestedTensor


class MLP(nn.Module):
    """简单的多层感知机，用于 box 回归 (cx, cy, w, h)。"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        layers = []
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        # x: [B, Nq, C] -> 输出 [B, Nq, 4]
        return self.mlp(x)


class MLPHead(nn.Module):
    """
    MLPHead:
    - 输入: decoder输出的query特征 [B, Nq, C]
    - 输出: 边框 [B, Nq, 4]
    """
    def __init__(self, hidden_dim, num_queries):
        super().__init__()
        self.bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        self.num_queries = num_queries
        self.feat_sz = None  # 对应CornerHead里的feat_sz, 在VGST里用来估计比例

    def forward(self, hs):
        """
        Args:
            hs: [num_decoder_layers, B, Nq, C] / 或 [B, Nq, C]
        Return:
            out: [B, Nq, 4]
        """
        if hs.dim() == 4:
            # 只取最后一层decoder输出
            hs_last = hs[-1]  # [B,Nq,C]
        else:
            hs_last = hs
        out = self.bbox_embed(hs_last)  # [B,Nq,4]
        return out


class Corner_Predictor_Lite(nn.Module):
    """
    卷积式 corner head 的精简版。
    (保留代码框架，方便后续如需 heatmap-based 预测。)
    """
    def __init__(self, inplanes=256, channel=256, feat_sz=16, stride=16):
        super().__init__()
        self.feat_sz = feat_sz
        self.stride = stride

        # 预测四个边界的 heatmap
        self.conv1 = nn.Conv2d(inplanes, channel, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU(inplace=True)

        self.conv_out = nn.Conv2d(channel, 4, kernel_size=1)

    def forward(self, x):
        """
        x: [B, C, H, W]
        return: 四个边界图 Bx4xH'xW'
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        out = self.conv_out(x)
        return out


def build_box_head(cfg):
    """
    根据 cfg.MODEL.HEAD_TYPE 构建不同的 box head。

    需要的cfg字段：
        cfg.MODEL.HEAD_TYPE           -> 'MLP' or 'CORNER'
        cfg.MODEL.HIDDEN_DIM          -> Transformer输出维度
        cfg.MODEL.NUM_QUERIES         -> 查询数量
        cfg.DATA.SEARCH.SIZE          -> 搜索区域输入大小(如256)
        cfg.MODEL.HEAD.STRIDE         -> backbone最终特征stride(如16)
        cfg.MODEL.HEAD.FEAT_SZ        -> 特征图分辨率(如16)

    返回:
        head_module (nn.Module)
    """

    head_type = cfg.MODEL.HEAD_TYPE

    if head_type.upper() == "MLP":
        # 最简单的 MLP 回归 head
        head_module = MLPHead(
            hidden_dim=cfg.MODEL.HIDDEN_DIM,
            num_queries=cfg.MODEL.NUM_QUERIES
        )
        # 给VGST用来记录特征图大小，方便后续计算尺度
        head_module.feat_sz = cfg.MODEL.HEAD.FEAT_SZ
        return head_module

    elif head_type.upper() == "CORNER":
        stride = cfg.MODEL.HEAD.STRIDE
        feat_sz = cfg.DATA.SEARCH.SIZE // stride

        head_module = Corner_Predictor_Lite(
            inplanes=cfg.MODEL.HEAD_DIM,
            channel=cfg.MODEL.HEAD_DIM,
            feat_sz=feat_sz,
            stride=stride,
        )
        head_module.feat_sz = feat_sz
        return head_module

    else:
        raise ValueError(f"Unknown HEAD_TYPE '{head_type}'")
