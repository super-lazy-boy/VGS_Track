# BaseTrainer:
#   - 负责最高层的 epoch 循环：for epoch in ...:
#   - 调用子类的 train_epoch()
#   - 负责 checkpoint 保存/恢复
#
# VGSTTrainer 继承它并实现 train_epoch() 以及统计/打印相关的辅助函数。

import os
import glob
import torch
import traceback

from torch.utils.data.distributed import DistributedSampler
from lib.train.admin import multigpu


class BaseTrainer:
    """基础 Trainer。子类(比如 VGSTTrainer) 需要实现 train_epoch()。"""

    def __init__(self, actor, loaders, optimizer, settings, lr_scheduler=None):
        """
        Args:
            actor        : Actor 实例（封装了模型 + loss 计算）
            loaders      : list[...]，通常包含训练集 DataLoader 和验证集 DataLoader
            optimizer    : torch.optim.Optimizer
            settings     : Settings() 实例，包含 env/save_dir/local_rank 等
            lr_scheduler : (可选) torch.optim.lr_scheduler.*
        """
        self.actor = actor
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.loaders = loaders
        self.settings = settings

        # 当前的 epoch 号（从1开始训练；也可以在恢复时被重写）
        self.epoch = 0

        # 统计信息（由子类负责填充/更新）
        self.stats = {}

        # ----------------- 设备选择 -----------------
        # run_training.py 通常会先把 settings.device 设成 "cuda:<rank>"。
        # 如果没设，这里兜底：有 GPU 就用 cuda:0，否则用 CPU。
        self.device = getattr(settings, 'device', None)
        if self.device is None:
            if torch.cuda.is_available() and getattr(settings, 'use_gpu', True):
                self.device = torch.device("cuda:0")
            else:
                self.device = torch.device("cpu")
            self.settings.device = self.device

        # 把 actor（里面包含的 net）搬到目标设备
        self.actor.to(self.device)

        # ----------------- checkpoint 目录 -----------------
        # 统一保存到 <save_dir>/checkpoints/<experiment_name>/xxx_ep0001.pth
        ckpt_root = os.path.join(self.settings.save_dir, 'checkpoints')
        os.makedirs(ckpt_root, exist_ok=True)
        self._checkpoint_dir = ckpt_root

    def train(self, max_epochs, load_latest=False, fail_safe=True,
              load_previous_ckpt=False, distill=False):
        """
        训练主循环。

        max_epochs (int): 要训练多少个 epoch （通常来自 cfg.TRAIN.EPOCH）

        大致流程：
          for epoch in 1..max_epochs:
              self.train_epoch()        # 子类实现，里头会跑训练+验证
              lr_scheduler.step()       # 如果有
              周期性地 save_checkpoint()

        我们还在循环结束后加了一个“兜底保存”的逻辑，保证最后一个 epoch 的
        权重一定落盘成 .pth，方便后续测试脚本直接加载最新权重。
        """
        start_epoch = self.epoch + 1

        # （可选）这里可以加 resume / distill 的逻辑，如果需要的话
        # if load_latest: self.load_checkpoint(...)
        # if load_previous_ckpt: ...
        # if distill: ...

        for cur_epoch in range(start_epoch, max_epochs + 1):
            self.epoch = cur_epoch

            try:
                # 1. 跑本 epoch（VGSTTrainer.train_epoch 会先跑训练loader，再跑验证loader）
                self.train_epoch()

                # 2. 学习率调度（常见 StepLR / MultiStepLR 每个 epoch 调一次）
                if self.lr_scheduler is not None:
                    self.lr_scheduler.step()

                # 3. 按策略定期保存 checkpoint
                save_every_epoch = getattr(self.settings, "save_every_epoch", False)
                if save_every_epoch or (self.epoch % 2 == 0):
                    # 只有主进程(rank==-1 或 rank==0)才负责真正写文件
                    if self.settings.local_rank in [-1, 0]:
                        self.save_checkpoint()

            except Exception as e:
                print(f"Training crashed at epoch {self.epoch}")
                print("Traceback:")
                print(traceback.format_exc())
                if fail_safe:
                    # 保持 fail-safe：如果你希望继续尝试，可以在这里吞掉异常。
                    # 这里我们选择抛出，防止静默错误。
                    raise e
                else:
                    raise e

        # 4. 训练循环全部结束后，兜底再保证保存一次最终的 .pth
        if self.settings.local_rank in [-1, 0]:
            net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net
            net_type = type(net).__name__
            directory = os.path.join(self._checkpoint_dir, self.settings.project_path)
            os.makedirs(directory, exist_ok=True)
            final_path = os.path.join(directory, f"{net_type}_ep{self.epoch:04d}.pth")

            # 如果最后一轮并没有在上面的循环里被保存（比如最后一轮是奇数且 save_every_epoch=False）
            # 那么现在主动保存一次，确保有最终模型权重。
            if not os.path.exists(final_path):
                self.save_checkpoint()

        print('Finished training!')

    def train_epoch(self):
        """
        子类必须实现。
        在 VGSTTrainer 里，这个方法被重写为：
        - 先跑训练集 DataLoader(反向+更新)
        - 再跑验证集 DataLoader(仅前向)
        - 记录统计并写 TensorBoard
        """
        raise NotImplementedError

    def save_checkpoint(self):
        """
        保存当前 epoch 的 checkpoint（基础版本）。
        VGSTTrainer 也实现了自己的 save_checkpoint()，带 scheduler 状态等。
        这里保留是为了兼容可能的其他 Trainer 子类。
        """
        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__

        state = {
            'epoch': self.epoch,
            'actor_type': actor_type,
            'net_type': net_type,
            'net': net.state_dict(),
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),
            'stats': self.stats,
            'settings': self.settings,
        }

        directory = os.path.join(self._checkpoint_dir, self.settings.project_path)
        os.makedirs(directory, exist_ok=True)

        tmp_file_path = os.path.join(directory, f'{net_type}_ep{self.epoch:04d}.tmp')
        torch.save(state, tmp_file_path)

        final_path = os.path.join(directory, f'{net_type}_ep{self.epoch:04d}.pth')
        os.rename(tmp_file_path, final_path)

        print(f"Checkpoint saved to {final_path}")

    def load_checkpoint(self, checkpoint=None, fields=None,
                        ignore_fields=None, load_constructor=False):
        """
        （保留原始恢复逻辑，方便以后需要 resume 的场景）
        """
        net = self.actor.net.module if multigpu.is_multi_gpu(self.actor.net) else self.actor.net

        actor_type = type(self.actor).__name__
        net_type = type(net).__name__

        # 解析 checkpoint 路径（支持 None/int/str）
        if checkpoint is None:
            glob_pattern = os.path.join(
                self._checkpoint_dir,
                self.settings.project_path,
                f'{net_type}_ep*.pth.tar'
            )
            checkpoint_list = sorted(glob.glob(glob_pattern))
            if checkpoint_list:
                checkpoint_path = checkpoint_list[-1]
            else:
                print('No matching checkpoint file found')
                return False
        elif isinstance(checkpoint, int):
            checkpoint_path = os.path.join(
                self._checkpoint_dir,
                self.settings.project_path,
                f'{net_type}_ep{checkpoint:04d}.pth.tar'
            )
        elif isinstance(checkpoint, str):
            if os.path.isdir(checkpoint):
                ckpt_list = sorted(glob.glob(os.path.join(checkpoint, '*_ep*.pth.tar')))
                if ckpt_list:
                    checkpoint_path = ckpt_list[-1]
                else:
                    raise Exception('No checkpoint found in directory {}'.format(checkpoint))
            else:
                checkpoint_path = os.path.expanduser(checkpoint)
        else:
            raise TypeError("checkpoint must be None / int / str")

        # 真正加载
        checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

        assert net_type == checkpoint_dict['net_type'], 'Network type mismatch.'

        if fields is None:
            fields = checkpoint_dict.keys()
        if ignore_fields is None:
            # 有些旧字段可能不兼容，默认忽略掉
            ignore_fields = [
                'settings', 'lr_scheduler', 'constructor', 'net_type',
                'actor_type', 'net_info'
            ]

        for key in fields:
            if key in ignore_fields:
                continue
            if key == 'net':
                net.load_state_dict(checkpoint_dict[key])
            elif key == 'optimizer':
                self.optimizer.load_state_dict(checkpoint_dict[key])
            else:
                setattr(self, key, checkpoint_dict[key])

        # 恢复 constructor / net_info
        if load_constructor and 'constructor' in checkpoint_dict and checkpoint_dict['constructor'] is not None:
            net.constructor = checkpoint_dict['constructor']
        if 'net_info' in checkpoint_dict and checkpoint_dict['net_info'] is not None:
            net.info = checkpoint_dict['net_info']

        # 如果恢复了 epoch，也同步 lr_scheduler 的 last_epoch
        if 'epoch' in fields and self.lr_scheduler is not None:
            self.lr_scheduler.last_epoch = self.epoch

        # 同步 DistributedSampler 的 epoch（如果存在）
        for loader in self.loaders:
            if hasattr(loader, 'sampler') and isinstance(loader.sampler, DistributedSampler):
                loader.sampler.set_epoch(self.epoch)

        return True
