import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from lib.utils.misc import NestedTensor


class FrozenBatchNorm2d(nn.Module):
    """
    FrozenBatchNorm2d
    -----------------
    我们保留这个类是为了向后兼容。
    其他地方可能 `from backbone import FrozenBatchNorm2d`，如果拿掉会 ImportError。

    这个实现与 torchvision 的 FrozenBatchNorm2d 思路一致：
    - running_mean / running_var / weight / bias 全是 buffer，不参与梯度更新
    - forward 时用这些常量做仿真的 BN 归一化
    """
    def __init__(self, n: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W]
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        scale = w * (rv + self.eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                               missing_keys, unexpected_keys, error_msgs):
        # 忽略掉 BatchNorm2d 常见的 num_batches_tracked
        num_batches_tracked = prefix + "num_batches_tracked"
        if num_batches_tracked in state_dict:
            state_dict.pop(num_batches_tracked)
        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs
        )


class DepthBackbone(nn.Module):
    """
    DepthBackbone (延迟构建版 / lazy init)
    -------------------------------------

    目标：
    - 输入: [B, C_in, H, W]   (注意：C_in 可能是 1，也可能是 3。我们现在不能假设=1)
    - 输出: [B, hidden_dim, H/16, W/16]
      也就是经过总 stride=16 的下采样，与 VGST.divisor=16 对齐。

    结构：
    - 4 个卷积 block，每个 block stride=2，这样 H,W 依次 /2,/4,/8,/16
      block1: Conv(stride=2,k7) -> BN -> ReLU
      block2: Conv(stride=2,k3) -> BN -> ReLU
      block3: Conv(stride=2,k3) -> BN -> ReLU
      block4: Conv(stride=2,k3) -> BN -> ReLU

    难点：
    - 训练日志显示 depth 实际是 3 通道：
        expected input[1, 3, 192, 192] ...
      我们之前写死 in_channels=1，所以第一层卷积报错。
    - 解决方案：我们不在 __init__ 里固定 in_channels，而是延迟到第一次 forward。
      这样无论是 1 通道还是 3 通道都能适配。

    实现细节：
    - __init__ 里只保存 hidden_dim，并标记 self.initialized = False
    - forward(x) 第一次被调用时，查看 x.shape[1] 得到实际 in_channels
      然后动态构建 block1~block4 并把它们注册成 Module 属性
      (nn.Sequential(...) 会被正确注册为子模块)
    - 之后就正常 forward 那些 block

    好处：
    - 不需要提前知道深度图是 1 通道还是 3 通道
    - 不会再触发 RuntimeError: expected input[...] to have 1 channels
    """
    def __init__(self, hidden_dim: int = 256):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.initialized = False  # 是否已经根据真实输入通道数构建好了 block
        # 下面的占位属性用来让 IDE / 静态检查不报错
        self.block1 = None
        self.block2 = None
        self.block3 = None
        self.block4 = None

        # 这个属性会在 _init_layers 里被设定成最终输出通道数
        self.out_channels = hidden_dim

    def _init_layers(self, in_channels: int, device: torch.device):
        """
        根据真实的 in_channels 动态创建卷积层，然后放到正确的 device 上。
        """
        hidden_dim = self.hidden_dim

        # 我们用一个循序渐进的通道扩张，
        # 这样既不会太小（表示力不足），也不会太大（爆显存）
        mid1 = max(64, hidden_dim // 4)       # 例如 hidden_dim=256 -> mid1=64
        mid2 = max(128, hidden_dim // 2)      # -> 128
        mid3 = max(192, (hidden_dim * 3)//4)  # -> 192
        mid4 = hidden_dim                     # -> 256

        # block1: k7, stride=2 让分辨率直接 /2，类似轻量 ResNet 的第一层
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, mid1, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(mid1),
            nn.ReLU(inplace=True),
        )

        # block2: stride=2 -> /4
        self.block2 = nn.Sequential(
            nn.Conv2d(mid1, mid2, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(mid2),
            nn.ReLU(inplace=True),
        )

        # block3: stride=2 -> /8
        self.block3 = nn.Sequential(
            nn.Conv2d(mid2, mid3, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(mid3),
            nn.ReLU(inplace=True),
        )

        # block4: stride=2 -> /16, 输出到 hidden_dim
        self.block4 = nn.Sequential(
            nn.Conv2d(mid3, mid4, kernel_size=3, stride=2,
                      padding=1, bias=False),
            nn.BatchNorm2d(mid4),
            nn.ReLU(inplace=True),
        )

        # 把这些子模块都放到输入所在的 device（CUDA / CPU）
        self.block1.to(device)
        self.block2.to(device)
        self.block3.to(device)
        self.block4.to(device)

        # 记录一下已经初始化
        self.initialized = True
        self.out_channels = hidden_dim  # 方便外部读取

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, C_in, H, W], 其中 C_in 可以是 1，也可以是 3（甚至别的值）
        Returns:
            feat: [B, hidden_dim, H/16, W/16]
        """
        if not self.initialized:
            in_channels = x.shape[1]            # 动态感知真实通道数
            self._init_layers(in_channels, x.device)

        # 正常前向
        x = self.block1(x)   # /2
        x = self.block2(x)   # /4
        x = self.block3(x)   # /8
        x = self.block4(x)   # /16

        # x: [B, hidden_dim, H/16, W/16]
        return x


class DepthJoiner(nn.Module):
    """
    DepthJoiner
    -----------
    这个模块包装 DepthBackbone，使其输出与 VGST 对 depth 分支的期望保持一致：

    forward(NestedTensor(depth_img, depth_mask)) -> (out_list, pos_list)

    其中：
    - out_list[-1] 是 NestedTensor(feat, mask_ds)
        feat:    [B, hidden_dim, H', W']
        mask_ds: [B, H', W'] (bool)
    - pos_list[-1] 是 pos: [B, hidden_dim, H', W']
        我们这里直接用全 0 的张量当位置信息，占位用。
        后续 VGST 会对 feat 和 pos 分别做 flatten+permute，
        再过 transformer。因此我们只需保证形状对齐。

    同时我们暴露 num_channels，让 VGST 在 __init__ 里能拿到：
        depth_in_channels = getattr(self.backbone_depth, "num_channels", hidden_dim)
    这样 VGST 仍然会正确构造 self.bottleneck_depth = Conv2d(num_channels, hidden_dim, 1)
    """
    def __init__(self, backbone: DepthBackbone, hidden_dim: int):
        super().__init__()
        self.backbone = backbone
        self.num_channels = hidden_dim  # 供 VGST.__init__ 使用

    def forward(self, tensor_list: NestedTensor, mode: Optional[str] = None):
        assert isinstance(tensor_list, NestedTensor), \
            "DepthJoiner.forward expects a NestedTensor input."

        x = tensor_list.tensors        # [B, C_in, H, W]
        mask = tensor_list.mask        # [B, H, W] (bool) 或 None

        feat = self.backbone(x)        # [B, hidden_dim, H', W']
        B, C, Hp, Wp = feat.shape

        # 最近邻下采样 mask，让它跟特征空间对齐
        if mask is not None:
            mask_ds = F.interpolate(
                mask.float().unsqueeze(1),  # [B,1,H,W]
                size=(Hp, Wp),
                mode='nearest'
            ).to(torch.bool).squeeze(1)     # [B,Hp,Wp]
        else:
            mask_ds = None

        # 位置信息（pos）：我们用零张量占位，但形状必须匹配 feat
        pos = torch.zeros_like(feat)        # [B, C, Hp, Wp]

        out_list = [NestedTensor(feat, mask_ds)]
        pos_list = [pos]

        return out_list, pos_list


def build_backbone(cfg) -> DepthJoiner:
    """
    构造 depth 分支的 backbone，供 VGST 使用。

    和我们之前的版本相比，这里最大的变化是：
    - 不再假设深度输入一定是单通道 (1 channel)
    - 使用 DepthBackbone(hidden_dim=cfg.MODEL.HIDDEN_DIM) 并延迟初始化，
      在第一次 forward() 时根据真实输入 x.shape[1] 构建卷积层

    我们返回 DepthJoiner(backbone, hidden_dim)，
    让它暴露 num_channels = hidden_dim，保持 VGST.__init__ 的逻辑不变：
        depth_in_channels = getattr(self.backbone_depth, "num_channels", hidden_dim)
        self.bottleneck_depth = nn.Conv2d(depth_in_channels, hidden_dim, 1)

    这样：
    - VGST.forward_backbone_depth() 仍然可以直接调用
    - 后续 transformer 的多模态拼接格式不会变
    - 显存依旧友好（四个 stride=2 conv block，没有自注意力）
    """
    hidden_dim = cfg.MODEL.HIDDEN_DIM
    depth_backbone = DepthBackbone(hidden_dim=hidden_dim)

    model = DepthJoiner(depth_backbone, hidden_dim=hidden_dim)

    # 再显式一把，确保属性在 VGST 那边可见/可取
    model.num_channels = hidden_dim

    return model
