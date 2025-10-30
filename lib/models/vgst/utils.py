# utils.py
# 仅保留推理/训练主干真正使用的工具模块。
# - FrozenBatchNorm2d: 冻结版BN，在小batch或微调时稳定
# 其他调试类 / 杂项函数已剔除，保持最小依赖面。

import torch
from torch import nn


class FrozenBatchNorm2d(nn.Module):
    """
    冻结BN:
    - 不会更新running_mean / running_var
    - 也不依赖当前batch的统计量
    - 非常适合小batch或希望主干稳定的时候使用
    数学形式:
        y = (x - running_mean) / sqrt(running_var + eps) * weight + bias
    """

    def __init__(self, num_features: int):
        super().__init__()
        # 作为buffer而不是Parameter -> 不参与梯度
        self.register_buffer("weight", torch.ones(num_features))
        self.register_buffer("bias", torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """
        为了兼容 torchvision 预训练权重：
        torchvision 的 BN 里有 'num_batches_tracked'，但 FrozenBN 不需要。
        这里手动丢弃掉这个 key，避免加载时报错。
        """
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # reshape 成 [1,C,1,1] 方便广播
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)

        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias
