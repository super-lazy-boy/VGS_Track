# resnet.py
# ResNet 主干（用于深度分支），修复了 forward() 中未定义 _forward_impl 的问题。
#
# 注意：
#  - 结构与 torchvision.resnet 基本一致
#  - 在我们的 pipeline 里，真正用的是中间层的特征 (layer2/layer3)，
#    而不是最终 fc 分类输出
#  - build_backbone() 会用 IntermediateLayerGetter 取这些中间层
#
# 这些主干的超参（如是否使用dilation、哪种ResNet、是否加载预训练）统一由
# cfg.MODEL.BACKBONE.* 在 settings.py 中给出。

import torch
import torch.nn as nn
from torchvision.models.resnet import BasicBlock, Bottleneck, conv1x1
from torch.utils.model_zoo import load_url as load_state_dict_from_url

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]

# 官方 torchvision 的预训练权重 URL
model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


class ResNet(nn.Module):
    """
    标准 ResNet 主干:
    conv1/bn1/relu/maxpool -> layer1 -> layer2 -> layer3 -> layer4
    (avgpool+fc 主要用于分类任务；检测/跟踪时我们常从中间层截特征)
    """

    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super().__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # 例如 [False, dilation_flag, False]
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None or a 3-element list"
            )
        self.groups = groups
        self.base_width = width_per_group

        # stem
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 4 个 stage
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block,
            128,
            layers[1],
            stride=2,
            dilate=replace_stride_with_dilation[0],
        )
        self.layer3 = self._make_layer(
            block,
            256,
            layers[2],
            stride=2,
            dilate=replace_stride_with_dilation[1],
        )
        self.layer4 = self._make_layer(
            block,
            512,
            layers[3],
            stride=2,
            dilate=replace_stride_with_dilation[2],
        )

        # 分类头（检测里通常不用，但保留保证权重兼容）
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 参数初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu"
                )
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

        # 可选: 将残差分支最后BN权重设为0，初始化成近似恒等
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(
        self, block, planes, blocks, stride=1, dilate=False
    ):
        """
        构建一个ResNet stage (layerX)
        stride=2 表示下采样；dilate=True 则使用空洞卷积代替下采样。
        """
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            # 用空洞卷积保持空间分辨率更大
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            # 如果通道数或stride变了，需要下采样分支
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers_list = []
        layers_list.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers_list.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers_list)

    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        """
        标准 ResNet forward。
        在本项目里，通常我们不会用这个最终输出，而是通过
        IntermediateLayerGetter 在 backbone.py 里截取中间层特征图。
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 修复了原始实现中的命名不一致问题
        return self._forward_impl(x)


def _resnet(
    arch,
    block,
    layers,
    pretrained,
    progress,
    **kwargs,
):
    """
    通用工厂函数:
    - 根据 layers 配置构造ResNet
    - 如果 pretrained=True 则加载 torchvision 的预训练权重
    """
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(
            model_urls[arch], progress=progress
        )
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "resnet18",
        BasicBlock,
        [2, 2, 2, 2],
        pretrained,
        progress,
        **kwargs,
    )


def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "resnet34",
        BasicBlock,
        [3, 4, 6, 3],
        pretrained,
        progress,
        **kwargs,
    )


def resnet50(pretrained=False, progress=True, **kwargs):
    from torchvision.models.resnet import Bottleneck
    return _resnet(
        "resnet50",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        **kwargs,
    )


def resnet101(pretrained=False, progress=True, **kwargs):
    from torchvision.models.resnet import Bottleneck
    return _resnet(
        "resnet101",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        **kwargs,
    )


def resnet152(pretrained=False, progress=True, **kwargs):
    from torchvision.models.resnet import Bottleneck
    return _resnet(
        "resnet152",
        Bottleneck,
        [3, 8, 36, 3],
        pretrained,
        progress,
        **kwargs,
    )


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    from torchvision.models.resnet import Bottleneck
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "resnext50_32x4d",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        **kwargs,
    )


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    from torchvision.models.resnet import Bottleneck
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(
        "resnext101_32x8d",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        **kwargs,
    )


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    from torchvision.models.resnet import Bottleneck
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet50_2",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        **kwargs,
    )


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    from torchvision.models.resnet import Bottleneck
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet101_2",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        **kwargs,
    )
