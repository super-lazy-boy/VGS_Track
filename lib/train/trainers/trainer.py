# /remote-home/ai2005_11/VGST/lib/train/trainers/trainer.py
#
# 这个文件定义 VGSTTrainer，它负责一个完整训练流程里最关键的几件事：
#   - 把 DataLoader 取出的原始 batch 整理成模型可吃的格式
#   - 调用 actor 得到 loss，做反向传播和优化
#   - 跑验证集，只做前向不更新
#   - 统计/打印训练和验证指标
#   - 写 TensorBoard
#   - 按 epoch 保存 checkpoint (.pth)
#   - 额外保存验证最优的 best 模型 (<ModelName>_best.pth)
#
# 特别约定：
#   “一个 epoch = 先跑训练集(更新参数)，再跑验证集(评估性能)”
#
# 这样与经典写法保持一致，相当于你伪代码里的：
#   for epoch in range(E):
#       train(...)
#       val_loss = validate(...)
#       if val_loss < best_loss:
#           save best model
#
# 依赖：
#   - BaseTrainer: 提供 self.actor / self.optimizer / self.settings / self.device / self._checkpoint_dir 等基础
#   - AverageMeter, StatValue: 用来做loss等指标的平均统计
#   - TensorboardWriter: 写 tensorboard 曲线
#   - NestedTensor: (imgs, mask) 封装，方便下游模型吃可变分辨率的图像
#
# 重要字段：
#   self.best_val_loss         : 目前为止验证集最小的 Loss/total
#   self.best_ckpt_path        : 已经导出的 best 模型路径（方便打印）
#
#   self.stats[loader.name]    : 一个 OrderedDict，里面是若干 AverageMeter / StatValue
#                                例如：
#                                   self.stats["Train"]["Loss/total"].avg
#                                   self.stats["Val"]["IoU"].avg
#
#   loader.training == True    : 表示这个 DataLoader 是训练集
#   loader.training == False   : 表示这个 DataLoader 是验证集
#
#   cycle_dataset(loader)      : 遍历该 loader
#       - 如果是训练 loader：前向 -> loss.backward() -> optimizer.step()
#       - 如果是验证 loader：只前向，不反向不更新
#
#   train_epoch()              : 一个 epoch 的总体逻辑
#       1) 先跑所有训练 loader
#       2) 再跑所有验证 loader
#       3) 汇总统计 & 写TensorBoard
#       4) 根据验证集损失，看看要不要刷新 best 模型
#
# 训练过程中 BaseTrainer.train(max_epochs) 会一轮一轮调 train_epoch()。
#

import os
import time
import glob
import torch
import traceback
from collections import OrderedDict
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler

from lib.train.trainers import BaseTrainer
from lib.train.admin import AverageMeter, StatValue, TensorboardWriter
from lib.utils.misc import NestedTensor


class VGSTTrainer(BaseTrainer):
    """
    VGSTTrainer
    -----------

    - 把 dataloader 给的一批原始数据(batch) 整理成模型可吃的格式
      (搬到 GPU、封装成 NestedTensor、补充 meta 信息等)。
    - 调用 actor(batch) 得到 loss 和统计信息，并在需要时做反向传播+更新参数。
    - 记录/打印训练过程指标，写 TensorBoard。
    - 保存 checkpoint（.pth），供后续继续训练或离线调试。
    - 维护 "best 验证集模型"，导出 <ModelName>_best.pth，用于最终测试脚本加载。

    一个 epoch 的定义：
      1) 依次跑所有训练集 loader（会反向传播+更新参数）
      2) 依次跑所有验证集 loader（只前向、不更新）
      3) 做统计、写 TensorBoard
      4) 用验证集平均 loss 决定是否刷新 best 模型
    """

    def __init__(self, actor, loaders, optimizer, settings,
                 lr_scheduler=None, use_amp=False):
        """
        Args:
            actor : VGSTActor，负责前向+loss计算，返回 (loss, stats)
            loaders : list[DataLoader-like]，其中
                      loader.training == True  表示训练集
                      loader.training == False 表示验证集
            optimizer : torch.optim.Optimizer
            settings : Settings()，包含训练的全局配置(路径、batch大小、打印间隔等)
            lr_scheduler : 学习率调度器(可选)
            use_amp : 是否使用混合精度 (torch.cuda.amp)
        """
        super().__init__(actor, loaders, optimizer, settings, lr_scheduler)

        self.settings = settings
        self.use_amp = use_amp
        self.move_data_to_gpu = getattr(settings, 'move_data_to_gpu', True)

        # 给 settings 补默认字段（如果配置里没写）
        self._set_default_settings()
#无能的赵阿强
        # 每个 dataloader 维护一份独立的统计字典
        # 例如 self.stats["Train"]["Loss/total"] 是一个 AverageMeter
        self.stats = OrderedDict()
        for loader in self.loaders:
            self.stats[loader.name] = OrderedDict()

        # 只有主进程(rank == -1 或 0)负责写 TensorBoard
        if settings.local_rank in [-1, 0]:
            tensorboard_writer_dir = os.path.join(
                self.settings.env.tensorboard_dir,
                self.settings.project_path
            )
            os.makedirs(tensorboard_writer_dir, exist_ok=True)
            self.tensorboard_writer = TensorboardWriter(
                tensorboard_writer_dir,
                [l.name for l in loaders]
            )

        # AMP 混合精度支持（减少显存占用）
        if use_amp:
            self.scaler = GradScaler()

        # 追踪最佳验证集损失，用来导出 best.pth
        self.best_val_loss = float('inf')
        self.best_ckpt_path = None

        print(f"VGSTTrainer initialized with AMP={use_amp}, Device={self.device}")

    def _set_default_settings(self):
        """
        给 settings 填上 Trainer 运行时必须的字段。
        如果外部 cfg/settings 没配，就用这些默认值兜底。
        """
        defaults = {
            'print_interval': 10,   # 每隔多少 iter 打印一次统计
            'print_stats': None,    # 如果是列表，只打印其中名字的统计；为 None 打印全部
            'description': '',
            'grad_clip_norm': 0.0,  # >0 时会做梯度裁剪
            'save_every_epoch': False,
            'debug': False,         # True 时遇到异常会直接 raise，方便定位
        }
        for k, v in defaults.items():
            if getattr(self.settings, k, None) is None:
                setattr(self.settings, k, v)

    # ----------------------------------------------------------------------
    # 把 dataloader 的一批原始数据转成模型可用的 batch (搬到GPU，封装NestedTensor等)
    # ----------------------------------------------------------------------
    def process_multimodal_data(self, data: dict):
        """
        主要做三件事：
        1. (可选) 把所有 torch.Tensor / list[Tensor] / dict[...] 挪到当前设备 self.device
        2. 如果 RGB/Depth 还是裸 Tensor，则封装成 NestedTensor(tensor, mask)
           - NestedTensor.tensors: [B, C, H, W]
           - NestedTensor.mask   : [B, H, W] 的bool，True表示padding的无效区域
             我们目前简单地置全False，表示整张图都有效。
        3. 附加 meta 信息（当前 epoch、settings），有些 actor / loss 会用得到
        """

        # 1) 统一搬到 GPU/CPU (self.device)
        if self.move_data_to_gpu:
            def move_to_device(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.to(self.device, non_blocking=True)
                elif isinstance(obj, dict):
                    return {k: move_to_device(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [move_to_device(x) for x in obj]
                else:
                    return obj
            data = move_to_device(data)

        # 2) 如果出现 RGB / Depth 的裸 Tensor，把它们打包成 NestedTensor
        def to_nested(image_tensor: torch.Tensor) -> NestedTensor:
            # 这里默认整张图都有效 -> mask 全 False
            mask = torch.zeros(
                (image_tensor.shape[0], image_tensor.shape[2], image_tensor.shape[3]),
                dtype=torch.bool,
                device=image_tensor.device
            )
            return NestedTensor(image_tensor, mask)

        for key in ['template_color', 'search_color', 'template_depth', 'search_depth']:
            if key in data and isinstance(data[key], torch.Tensor):
                data[key] = to_nested(data[key])

        # 3) 附加 meta 信息，方便 actor/net 做一些依赖 epoch 的策略
        data['epoch'] = self.epoch
        data['settings'] = self.settings

        return data

    def _describe_batch(self, batch: dict) -> str:
        """
        出错时打印 batch 的“形状快照”，帮助 debug。
        """
        def shape_of(x):
            if isinstance(x, torch.Tensor):
                return tuple(x.shape) + (str(x.dtype),)
            if isinstance(x, NestedTensor):
                t = x.tensors
                m = x.mask
                return ("NestedTensor",
                        tuple(t.shape), str(t.dtype),
                        None if m is None else tuple(m.shape))
            if isinstance(x, dict):
                return {k: shape_of(v) for k, v in x.items()}
            if isinstance(x, list):
                return [shape_of(v) for v in x]
            return type(x).__name__

        preview = {k: shape_of(v) for k, v in list(batch.items())[:10]}
        return str(preview)

    # ----------------------------------------------------------------------
    # 遍历一个 loader
    #   - train loader: forward -> backward -> optimizer.step()
    #   - val loader  : forward (no backward/step)
    #   并累计统计指标到 self.stats[loader.name]
    # ----------------------------------------------------------------------
    def cycle_dataset(self, loader):
        """
        遍历一个 DataLoader:
        - 如果 loader.training == True：训练模式，会反向传播并更新参数。
        - 如果 loader.training == False：验证模式，只前向推理统计 loss/IoU 等。
        """
        self.actor.train(loader.training)          # 控制 Dropout/BN
        torch.set_grad_enabled(loader.training)    # 关闭验证阶段的梯度计算，节省显存

        self._init_timing()  # 初始化FPS统计用的计时器

        for i, batch in enumerate(loader, 1):
            processed = self.process_multimodal_data(batch)

            try:
                # ---------- 前向 ----------
                if not self.use_amp:
                    # 常规 FP32
                    loss, stats = self.actor(processed)
                else:
                    # 混合精度，显存占用更低
                    with autocast():
                        loss, stats = self.actor(processed)

                # ---------- 反向 & step (只在训练集做) ----------
                if loader.training:
                    self.optimizer.zero_grad()

                    if not self.use_amp:
                        # 纯 FP32 训练
                        loss.backward()

                        # 可选梯度裁剪
                        if self.settings.grad_clip_norm > 0:
                            torch.nn.utils.clip_grad_norm_(
                                self.actor.net.parameters(),
                                self.settings.grad_clip_norm
                            )

                        self.optimizer.step()

                    else:
                        # AMP 训练流程
                        self.scaler.scale(loss).backward()

                        if self.settings.grad_clip_norm > 0:
                            # unscale_ 把已经scale过的梯度还原回真实量级，再clip
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(
                                self.actor.net.parameters(),
                                self.settings.grad_clip_norm
                            )

                        self.scaler.step(self.optimizer)
                        self.scaler.update()

                    # 小技巧：释放没再用到的大块 CUDA 缓存，缓解显存碎片
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # ---------- 统计 / 打日志 ----------
                batch_size = self._get_batch_size(processed)
                self._update_stats(stats, batch_size, loader)
                self._print_stats(i, loader, batch_size)

            except RuntimeError as e:
                # RuntimeError 里最常见的是 CUDA OOM / shape mismatch
                print(f"[VGSTTrainer] Error in batch {i}: {repr(e)}")
                traceback.print_exc()

                # 对 OOM 简单自恢复：清空缓存然后跳过该 batch
                if 'CUDA out of memory' in str(e):
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue

                if getattr(self.settings, 'debug', False):
                    # debug 模式下直接抛出，方便快速定位
                    raise
                continue

            except Exception as e:
                # 其他逻辑错误，比如 batch 字段缺失
                print(f"[VGSTTrainer] Error in batch {i}: {repr(e)}")
                print("Batch preview:", self._describe_batch(processed))
                traceback.print_exc()
                if getattr(self.settings, 'debug', False):
                    raise
                continue

    def _get_batch_size(self, data: dict) -> int:
        """
        根据 batch 里的一个代表性张量来估计 batch_size，
        用于做加权平均统计 (AverageMeter.update(val, n=batch_size))
        """
        if 'template_color' in data and isinstance(data['template_color'], NestedTensor):
            return data['template_color'].tensors.shape[0]
        if 'search_color' in data and isinstance(data['search_color'], NestedTensor):
            return data['search_color'].tensors.shape[0]

        # fallback：找到第一个 Tensor / NestedTensor
        for v in data.values():
            if isinstance(v, torch.Tensor):
                return v.shape[0]
            if isinstance(v, NestedTensor):
                return v.tensors.shape[0]

        return 1  # 实在找不到就当作1

    # ----------------------------------------------------------------------
    # 一个 epoch = 训练(Train loaders) + 验证(Val loaders)
    # 并在 epoch 末尾根据验证集损失决定是否保存最优权重
    # ----------------------------------------------------------------------
    def train_epoch(self):
        """
        epoch 的完整流程：
          1) 先跑训练 loader（会更新模型）
          2) 再跑验证 loader（只前向）
          3) 计算本 epoch 的验证平均损失
          4) 如果这是目前最好的 val loss，则额外保存 best 模型
          5) 记录统计并写 TensorBoard
        """

        # --- 把 DataLoader 列成两组 ---
        train_loaders = [ld for ld in self.loaders if getattr(ld, "training", False)]
        val_loaders = [ld for ld in self.loaders if not getattr(ld, "training", False)]

        # --- (1) 训练阶段 ---
        for loader in train_loaders:
            # 分布式训练：要每个 epoch 设置不同随机种子以打乱数据
            if isinstance(loader.sampler, DistributedSampler):
                loader.sampler.set_epoch(self.epoch)

            print(f"[Epoch {self.epoch}] Training loader: {loader.name}")
            self.cycle_dataset(loader)  # 会前向+反向+step

        # --- (2) 验证阶段 ---
        for loader in val_loaders:
            if isinstance(loader.sampler, DistributedSampler):
                loader.sampler.set_epoch(self.epoch)

            print(f"[Epoch {self.epoch}] Validation loader: {loader.name}")
            self.cycle_dataset(loader)  # 只前向，不更新参数

        # --- (3) 计算这一轮的验证集平均损失 ---
        # 我们默认把验证集的指标名字叫 "Loss/total"
        # 这个名字来自 actor 返回的 stats，例如：
        #   stats = {
        #       'Loss/total': total_loss.item(),
        #       'Loss/giou' : giou_loss.item(),
        #       'Loss/l1'   : l1_loss.item(),
        #       'IoU'       : mean_iou.item(),
        #   }
        val_loss_epoch = self._compute_val_loss(val_loaders)

        # --- (4) 如果是目前最优，则保存 best 模型 (.pth，纯 state_dict) ---
        if val_loss_epoch is not None and val_loss_epoch < self.best_val_loss:
            self.best_val_loss = val_loss_epoch
            self._save_best_model()
            print(f"[Epoch {self.epoch}] New best model! val_loss={val_loss_epoch:.5f}")

        # --- (5) 统计收尾 + TensorBoard 写入 ---
        # 注意：_stats_new_epoch() 会调用每个 AverageMeter.new_epoch()，
        #       这会重置/滚动统计，因此必须在我们读取 val_loss 之后再调用它。
        self._stats_new_epoch()

        if self.settings.local_rank in [-1, 0]:
            self._write_tensorboard()

    def _compute_val_loss(self, val_loaders):
        """
        读取验证 loader(s) 的统计，计算本 epoch 的验证平均损失。
        逻辑：
          - 对每个验证 loader，取 self.stats[loader.name]['Loss/total'].avg
          - 把这些 loss 做简单平均（如果有多个验证 loader）
        返回 float 或 None（如果没验证集或没统计到 Loss/total）
        """
        losses = []
        for loader in val_loaders:
            loader_stats = self.stats.get(loader.name, {})
            meter = loader_stats.get('Loss/total', None)
            if meter is not None and hasattr(meter, 'avg'):
                losses.append(meter.avg)

        if len(losses) == 0:
            return None
        return sum(losses) / len(losses)

    # ----------------------------------------------------------------------
    # 下面是若干辅助方法：计时、统计、打印、写TensorBoard、保存ckpt
    # ----------------------------------------------------------------------
    def _init_timing(self):
        """初始化帧计数和时间戳，用于 FPS 统计。"""
        self.num_frames = 0
        self.start_time = time.time()
        self.prev_time = self.start_time

    def _update_stats(self, new_stats: OrderedDict, batch_size, loader):
        """
        把 actor 返回的 stats 合并进 self.stats[loader.name]。
        new_stats 是一个 dict，比如：
            {
              'Loss/total': 3.17,
              'Loss/giou' : 1.05,
              'Loss/l1'   : 0.21,
              'IoU'       : 0.10,
            }
        我们用 AverageMeter 来累计 (加权平均)，这样就能得到 epoch 均值。
        """
        if loader.name not in self.stats or self.stats[loader.name] is None:
            self.stats[loader.name] = OrderedDict()

        for name, val in new_stats.items():
            if name not in self.stats[loader.name]:
                self.stats[loader.name][name] = AverageMeter()
            # AverageMeter.update(val, n) 会把 val * n 加到 total，并把 count += n
            self.stats[loader.name][name].update(val, batch_size)

    def _print_stats(self, i, loader, batch_size):
        """
        定期打印训练/验证进度，包括：
        - loader 名 (Train / Val)
        - 当前 epoch 和 iter 进度
        - FPS(均值 / 当前batch)
        - 已累计的 loss / IoU 等指标的平均值
        也会（可选）写入到 settings.log_file
        """
        self.num_frames += batch_size
        current_time = time.time()
        batch_fps = batch_size / (current_time - self.prev_time)
        average_fps = self.num_frames / (current_time - self.start_time)
        self.prev_time = current_time

        if i % self.settings.print_interval == 0 or i == len(loader):
            line = f'[{loader.name}: {self.epoch}, {i}/{len(loader)}] '
            line += f'FPS: {average_fps:.1f} ({batch_fps:.1f}), '

            # 把 self.stats[loader.name] 里的所有 meter.avg 打出来
            if loader.name in self.stats and self.stats[loader.name] is not None:
                for name, meter in self.stats[loader.name].items():
                    if (self.settings.print_stats is None
                        or name in self.settings.print_stats):
                        if hasattr(meter, 'avg'):
                            line += f'{name}: {meter.avg:.5f}, '

            print(line.rstrip(', '))

            # 也写到日志文件里，方便回溯
            if getattr(self.settings, 'log_file', None) is not None:
                with open(self.settings.log_file, 'a') as f:
                    f.write(line.rstrip(', ') + '\n')

    def _stats_new_epoch(self):
        """
        每个 epoch 结束时做一些 housekeeping：
        1. 确保 self.stats[loader.name] 存在
        2. 记录当前学习率到 stats（LearningRate/groupX）
        3. 调用每个 meter.new_epoch()，让它“滚动/重置”为下个 epoch 做准备
        """
        # 确保所有 loader 都至少有 stats dict
        for loader in self.loaders:
            if loader.name not in self.stats or self.stats[loader.name] is None:
                self.stats[loader.name] = OrderedDict()

        # 把学习率写进训练集的 stats
        for loader in self.loaders:
            if loader.training and self.lr_scheduler is not None:
                # 有的 scheduler 有 get_lr()，有的需要 _get_lr(epoch)
                try:
                    lr_list = self.lr_scheduler.get_lr()
                except Exception:
                    lr_list = self.lr_scheduler._get_lr(self.epoch)

                for gi, lr in enumerate(lr_list):
                    var_name = f"LearningRate/group{gi}"
                    if var_name not in self.stats[loader.name]:
                        self.stats[loader.name][var_name] = StatValue()
                    self.stats[loader.name][var_name].update(lr)

        # 通知所有 meter 进入新 epoch
        for loader_stats in self.stats.values():
            if loader_stats is None:
                continue
            for stat in loader_stats.values():
                if hasattr(stat, 'new_epoch'):
                    stat.new_epoch()

    def _write_tensorboard(self):
        """
        把本 epoch 的统计曲线写入 TensorBoard。
        第 1 个 epoch 额外写 script_name 和 description，方便后续实验对照。
        """
        if self.epoch == 1:
            self.tensorboard_writer.write_info(
                self.settings.script_name,
                self.settings.description
            )
        self.tensorboard_writer.write_epoch(self.stats, self.epoch)

    # ----------------------------------------------------------------------
    # checkpoint 保存
    #   - save_checkpoint(): 带优化器、统计信息等完整训练状态，按 epoch 命名
    #   - _save_best_model(): 仅保存当前网络的 state_dict()，命名为 *_best.pth
    #     这个就是“表现最好的模型”，给测试脚本直接 load 来推理用
    # ----------------------------------------------------------------------
    def save_checkpoint(self):
        """
        保存完整 checkpoint（包含 optimizer/state/stats 等）。
        这个函数会被 BaseTrainer.train() 周期性调用 (比如每2个epoch)。
        输出文件名类似：
            checkpoints/<project_path>/<ModelName>_ep0004.pth
        """
        # 兼容 DataParallel / DDP: 真正的模型在 .module 下面
        net = self.actor.net.module if hasattr(self.actor.net, 'module') else self.actor.net

        state = {
            'epoch': self.epoch,
            'actor_type': type(self.actor).__name__,
            'net_type': type(net).__name__,
            'net': net.state_dict(),                 # 模型参数
            'net_info': getattr(net, 'info', None),
            'constructor': getattr(net, 'constructor', None),
            'optimizer': self.optimizer.state_dict(),# 优化器状态(继续训练要用)
            'stats': self.stats,                     # 训练/验证统计
            'settings': self.settings,               # 训练环境配置(方便复现)
            'lr_scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler else None
        }

        directory = os.path.join(self._checkpoint_dir, self.settings.project_path)
        os.makedirs(directory, exist_ok=True)

        # 先写 .tmp，成功后再 rename 成 .pth，避免中途崩溃留半截文件
        tmp_name = os.path.join(directory, f"{type(net).__name__}_ep{self.epoch:04d}.tmp")
        torch.save(state, tmp_name)

        final_name = os.path.join(directory, f"{type(net).__name__}_ep{self.epoch:04d}.pth")
        os.rename(tmp_name, final_name)

        print(f"[Epoch {self.epoch}] Checkpoint saved: {final_name}")
        self._cleanup_old_checkpoints(directory, type(net).__name__)

    def _save_best_model(self):
        """
        当本 epoch 的验证集损失优于历史最优时调用。
        只保存纯净的模型权重（没有优化器、没有统计），
        方便后续推理/测试脚本直接加载：
            model.load_state_dict(torch.load(...))
        """
        net = self.actor.net.module if hasattr(self.actor.net, 'module') else self.actor.net

        directory = os.path.join(self._checkpoint_dir, self.settings.project_path)
        os.makedirs(directory, exist_ok=True)

        best_path = os.path.join(directory, f"{type(net).__name__}_best.pth")
        torch.save(net.state_dict(), best_path)

        self.best_ckpt_path = best_path
        print(f"[Epoch {self.epoch}] Best model snapshot saved: {best_path}")

    def _cleanup_old_checkpoints(self, directory, net_type, keep_num=10):
        """
        为了不让磁盘爆掉，只保留最近 keep_num 个普通 checkpoint，
        把更老的 *_epXXXX.pth 删除。best.pth 不会被删。
        """
        ckpts = sorted(glob.glob(f"{directory}/{net_type}_ep*.pth"))
        if len(ckpts) > keep_num:
            for old_path in ckpts[:-keep_num]:
                os.remove(old_path)
                print(f"Removed old checkpoint: {old_path}")


def build_vgst_trainer(actor, loaders, optimizer, settings,
                       lr_scheduler=None, use_amp=False):
    """
    一个小工厂方法，外部 (train_script.run(settings)) 会用它来实例化 Trainer。
    """
    return VGSTTrainer(
        actor=actor,
        loaders=loaders,
        optimizer=optimizer,
        settings=settings,
        lr_scheduler=lr_scheduler,
        use_amp=use_amp
    )
