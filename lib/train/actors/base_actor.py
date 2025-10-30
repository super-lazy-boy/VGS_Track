# lib/train/actors/base_actor.py
#
# BaseActor:
#   - 训练环节的“执行者”
#   - 封装模型 net 以及损失 objective
#   - 子类必须实现 __call__()：
#       接收一批 data -> 前向 -> 计算loss -> 返回loss和日志用的stats
#
# VGSTActor 会继承它并实现具体逻辑。

import torch
from lib.utils import TensorDict  # 这里保留类型注解用；实际运行中传普通 dict 也可以


class BaseActor:
    """
    BaseActor: 抽象父类。
    self.net      : 要训练的网络（例如 VGST）
    self.objective: 损失函数字典 (例如 {'giou': giou_loss, 'l1': l1_loss})
    """

    def __init__(self, net, objective):
        """
        Args:
            net        - 待训练的模型
            objective  - 用于计算损失的函数字典
        """
        self.net = net
        self.objective = objective

    def __call__(self, data: TensorDict):
        """
        子类负责实现:
          1. 把输入 batch 数据喂进模型
          2. 计算 loss
          3. 组织日志需要的 stats
        返回:
            loss  : torch.Tensor (可反向传播)
            stats : dict[str, float]，用于打印/记录
        """
        raise NotImplementedError

    def to(self, device: torch.device):
        """
        把整个网络搬到某个 device(GPU/CPU)
        """
        self.net.to(device)

    def train(self, mode: bool = True):
        """
        设置模型为训练 or 评估状态 (影响 Dropout/BN)
        """
        self.net.train(mode)

    def eval(self):
        """
        等价 self.train(False)
        """
        self.train(False)
