"""
lib/train/base_functions.py

本文件负责三件事：
  1. 把 cfg 里的关键训练/数据增强超参数同步到 settings（Trainer 会直接访问 settings）
  2. 构建 dataloader（包含训练集 + 验证集；并保证同一 epoch 内先训练后验证）
  3. 构建 optimizer 和 lr_scheduler

【本版的两个关键更新】
- 明确区分训练集(TrainSet)与验证集(ValidationSet)的根路径，杜绝把 test/ 误混进训练。
  cfg.DATA.TRAIN.ROOT 应指向 /remote-home/ai2005_11/VGST/data/TrainSet
  cfg.DATA.VAL.ROOT   应指向 /remote-home/ai2005_11/VGST/data/ValidationSet

- build_dataloaders() 现在返回 [loader_train, loader_val]，
  其中 loader_train.training=True, loader_val.training=False。
  VGSTTrainer.train_epoch() 会按这个顺序循环它拿到的 loaders，
  因此每个 epoch 内会“先训练再验证”，满足需求。

此外：
- 自动修正 DATASETS_RATIO 与 DATASETS_NAME 长度不一致的问题，避免
  random.choices() 报 "The number of weights does not match the population"。
"""

from typing import Tuple, List
import torch
from torch.utils.data.distributed import DistributedSampler

# ====== 多模态数据加载依赖 ======
# - CustomDataset:       逐序列读取 color / depth / nlp / bbox
# - VLTrackingSampler:   根据 template/search 帧、GT bbox、语言描述 组装训练样本
# - SPTProcessing:       做空间裁剪、随机抖动、归一化等图像预处理
# - LTRLoader:           我们的 DataLoader 封装 (支持 .training 标志、.epoch_interval 等)
# - transforms (tfm):    随机灰度化、水平翻转、颜色抖动、Normalize 等
from lib.train.dataset import CustomDataset
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm

from lib.utils.misc import is_main_process


def update_settings(settings, cfg):
    """
    将 cfg 里的关键训练/数据增强超参数，同步到 settings 中，方便 Trainer 使用。

    settings: 训练上下文（run_training.py 创建），Trainer/VGSTTrainer 会直接读它
    cfg:      通过 build_default_cfg() 构造出来的配置对象（SimpleNamespace 树）

    我们同步的字段包括：
      - 打印频率、日志控制
      - patch 裁剪/抖动/输出分辨率（SPTProcessing 需要）
      - batch size / grad clip
      - scheduler_type（只是记录字符串）
      - num_template / num_search（一次采样中取多少 template/search 帧）
    """

    # 日志打印频率
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.print_stats = None  # None = 打印所有可用统计项

    # patch 裁剪 & 抖动（SPTProcessing 会用到）
    settings.search_area_factor = {
        'template': cfg.DATA.TEMPLATE.FACTOR,
        'search': cfg.DATA.SEARCH.FACTOR,
    }
    settings.output_sz = {
        'template': cfg.DATA.TEMPLATE.SIZE,
        'search': cfg.DATA.SEARCH.SIZE,
    }
    settings.center_jitter_factor = {
        'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
        'search': cfg.DATA.SEARCH.CENTER_JITTER,
    }
    settings.scale_jitter_factor = {
        'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
        'search': cfg.DATA.SEARCH.SCALE_JITTER,
    }

    # 梯度裁剪阈值
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM

    # batch size
    settings.batchsize = cfg.TRAIN.BATCH_SIZE

    # 学习率调度器类型（字符串，便于日志打印）
    settings.scheduler_type = cfg.TRAIN.SCHEDULER  # 例如 "StepLR", "MultiStepLR"

    # 每个样本里用几帧 template/search（时序跟踪常见设置）
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)


def _names2datasets(name_list: List[str], root_dir: str, image_loader):
    """
    把一组数据集名字（如 ["TrainSet"] 或 ["ValidationSet"]）实例化成
    若干个 CustomDataset 对象。

    为什么我们要手动传 root_dir？
    ---------------------------------
    旧逻辑里 dataset 总是用 settings.env.data_dir 作为根目录，这会把
    /VGST/data/ 下面的所有子目录（包括 test/）都扫描进来，导致训练阶段
    读到没有标注的 test 序列。

    现在我们用:
        train_root = cfg.DATA.TRAIN.ROOT  -> 只指向 TrainSet/
        val_root   = cfg.DATA.VAL.ROOT    -> 只指向 ValidationSet/
    从而彻底隔离 train / val / test。

    Parameters
    ----------
    name_list : List[str]
        例如 ["TrainSet"] 或 ["ValidationSet"]。
        这个名字最终会写到 BaseVideoDataset.name 里，方便日志区分来源。
    root_dir : str
        这个 split 的数据根目录，比如:
        "/remote-home/ai2005_11/VGST/data/TrainSet"
    image_loader :
        传入我们工程里的 opencv_loader。

    Returns
    -------
    datasets : List[CustomDataset]
        每个 name 都会产出一个 CustomDataset(...) 实例。
    """
    datasets = []

    for name in name_list:
        ds = CustomDataset(
            root=root_dir,             # ⚠️ 关键：明确指定该 split 的根目录
            dtype='rgbcolormap',       # RGB + Depth colormap 融合成多通道
            image_loader=image_loader,
            split_file=None,           # 如果你有自定义 train/val 列表文件，可以填这里
            dataset_name=name,         # 仅用于日志/区分不同子数据集
        )
        datasets.append(ds)

    return datasets


def _normalize_ratios(ratio_list: List[float], n_datasets: int) -> List[float]:
    """
    保证权重列表 p_datasets 的长度 == 数据集个数。
    采样器内部会做 random.choices(self.datasets, self.p_datasets)，
    两者长度不一致会直接报 ValueError("The number of weights does not match the population").

    规则：
    - 如果只给了一个权重，比如 [1.0]，就复制成 n_datasets 长。
    - 如果给的比需要的多，就截断到前 n_datasets 个。
    - 如果给的比需要的少，就用最后一个值补齐到 n_datasets。
    """
    if len(ratio_list) == n_datasets:
        return list(map(float, ratio_list))

    if len(ratio_list) == 1:
        return [float(ratio_list[0]) for _ in range(n_datasets)]

    if len(ratio_list) > n_datasets:
        return list(map(float, ratio_list[:n_datasets]))

    # len(ratio_list) < n_datasets
    padded = list(map(float, ratio_list))
    last_val = float(ratio_list[-1]) if len(ratio_list) > 0 else 1.0
    while len(padded) < n_datasets:
        padded.append(last_val)
    return padded


def build_dataloaders(cfg, settings):
    """
    构建并返回 dataloader 列表。

    我们现在会返回两个 loader:
    - loader_train: 训练用，training=True，会做反向传播+优化。
    - loader_val:   验证用，training=False，只前向、统计loss/IoU，不更新权重。

    VGSTTrainer.train_epoch() 会按照 loaders 的顺序循环，
    所以每个 epoch 内都会先跑 Train 再跑 Val，从而实现
    “1 个 epoch = 训练 + 验证” 这个需求。
    （VGSTTrainer 会在 cycle_dataset() 里根据 loader.training 来决定
     是否做 backward 和 optimizer.step。）


    数据流回顾：
      CustomDataset (按 root_dir 扫描 TrainSet 或 ValidationSet 序列)
        ↓
      VLTrackingSampler (时序地抽 template/search 帧 + bbox + 文本描述)
        ↓
      SPTProcessing (裁剪、抖动、归一化，转成模型需要的tensor格式)
        ↓
      LTRLoader (PyTorch 风格的 DataLoader；带 training 标记)

    注：我们不再给 VLTrackingSampler 传 frame_sample_mode="causal"，
    因为 sampler 的实现不认识 "causal"，会直接 raise ValueError("Illegal frame sample mode")。
    现在让它走默认模式即可，避免崩溃。
    """

    # -------------------------------------------------
    # 1. 数据增强流水线：图像变换
    # -------------------------------------------------

    # transform_joint：成对帧(template/search)共享的几何增强（如水平翻转）
    transform_joint = tfm.Transform(
        tfm.ToGrayscale(probability=0.05),
        tfm.RandomHorizontalFlip(probability=0.5),
    )

    # transform_single：单帧增强（颜色抖动+Normalize）
    transform_single = tfm.Transform(
        tfm.ToTensorAndJitter(0.2),
        tfm.RandomHorizontalFlip_Norm(probability=0.5),
        tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
    )

    # -------------------------------------------------
    # 2. patch 裁剪 / 抖动参数包装成 SPTProcessing
    #    这些超参数在 update_settings() 里已经写进了 settings
    # -------------------------------------------------
    data_processing = processing.SPTProcessing(
        search_area_factor=settings.search_area_factor,
        output_sz=settings.output_sz,
        center_jitter_factor=settings.center_jitter_factor,
        scale_jitter_factor=settings.scale_jitter_factor,
        mode='sequence',              # 表示按时序抽帧
        transform=transform_single,
        joint_transform=transform_joint,
        settings=settings,
    )

    # 允许的最大帧间隔（template vs search 相隔多少帧）
    max_gap = getattr(cfg.DATA, "SAMPLER_MAX_INTERVAL",
                      getattr(cfg.DATA, "SAMPLER_INTERVAL", 200))

    train_cls_flag = getattr(cfg.TRAIN, "TRAIN_CLS", False)

    # =================================================
    # 3A. --------- 构建训练集 sampler / loader ---------
    # =================================================

    # 训练集根路径，比如 "/remote-home/.../VGST/data/TrainSet"
    train_root = getattr(cfg.DATA.TRAIN, "ROOT",
                         getattr(settings.env, "data_dir", None))

    # 训练集每个子数据集（通常就一个 "TrainSet"）
    train_datasets = _names2datasets(
        cfg.DATA.TRAIN.DATASETS_NAME,
        train_root,
        opencv_loader
    )

    # 对齐采样权重长度，避免 random.choices() 报错
    train_ratios = _normalize_ratios(
        cfg.DATA.TRAIN.DATASETS_RATIO,
        len(train_datasets)
    )

    # 训练采样器：会随机选数据集 -> 抽一段序列 -> 抽 template/search 帧
    sampler_train = sampler.VLTrackingSampler(
        datasets=train_datasets,
        p_datasets=train_ratios,
        samples_per_epoch=cfg.DATA.TRAIN.SAMPLE_PER_EPOCH,
        max_gap=max_gap,
        num_search_frames=settings.num_search,
        num_template_frames=settings.num_template,
        processing=data_processing,
        # 这里不要再传 frame_sample_mode="causal"，否则采样器报错
        train_cls=train_cls_flag,
        max_seq_len=cfg.DATA.MAX_SEQ_LENGTH,
        bert_model=cfg.MODEL.LANGUAGE.TYPE,
        bert_path=cfg.MODEL.LANGUAGE.VOCAB_PATH,
    )

    # 分布式下给每张卡一个 DistributedSampler，这样不会重复抽样
    if settings.local_rank != -1:
        train_dist_sampler = DistributedSampler(sampler_train)
        train_shuffle = False  # DDP 下通常由 DistributedSampler 控制随机性
    else:
        train_dist_sampler = None
        train_shuffle = True

    loader_train = LTRLoader(
        name='Train',
        dataset=sampler_train,
        training=True,                 # <- 训练阶段，会反向传播+更新权重
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=train_shuffle,
        num_workers=cfg.TRAIN.NUM_WORKER,
        drop_last=True,                # 丢掉最后不足 batch 的部分，保证张量形状一致
        stack_dim=1,                   # 把时间维堆在 dim=1，符合后续模型的期望
        sampler=train_dist_sampler,
        epoch_interval=1               # 每个 epoch 都跑
    )

    # =================================================
    # 3B. --------- 构建验证集 sampler / loader ---------
    # =================================================
    # 只有当 cfg.DATA 里提供 VAL 配置时才建；否则只返回训练 loader
    has_val_cfg = hasattr(cfg.DATA, "VAL")

    loaders = [loader_train]

    if has_val_cfg:
        # 验证集根路径，比如 "/remote-home/.../VGST/data/ValidationSet"
        val_root = getattr(cfg.DATA.VAL, "ROOT",
                           getattr(settings.env, "data_dir", None))

        val_datasets = _names2datasets(
            cfg.DATA.VAL.DATASETS_NAME,
            val_root,
            opencv_loader
        )

        val_ratios = _normalize_ratios(
            cfg.DATA.VAL.DATASETS_RATIO,
            len(val_datasets)
        )

        sampler_val = sampler.VLTrackingSampler(
            datasets=val_datasets,
            p_datasets=val_ratios,
            samples_per_epoch=getattr(
                cfg.DATA.VAL, "SAMPLE_PER_EPOCH",
                cfg.DATA.TRAIN.SAMPLE_PER_EPOCH  # fallback：如果没专门写就沿用训练的
            ),
            max_gap=max_gap,
            num_search_frames=settings.num_search,
            num_template_frames=settings.num_template,
            processing=data_processing,
            # 同样不传 frame_sample_mode
            train_cls=train_cls_flag,
            max_seq_len=cfg.DATA.MAX_SEQ_LENGTH,
            bert_model=cfg.MODEL.LANGUAGE.TYPE,
            bert_path=cfg.MODEL.LANGUAGE.VOCAB_PATH,
        )

        # 验证阶段一般不需要 shuffle
        if settings.local_rank != -1:
            val_dist_sampler = DistributedSampler(sampler_val, shuffle=False)
        else:
            val_dist_sampler = None

        loader_val = LTRLoader(
            name='Val',
            dataset=sampler_val,
            training=False,            # <- 验证阶段：不反向传播不step
            batch_size=cfg.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=cfg.TRAIN.NUM_WORKER,
            drop_last=False,           # 保留最后的不足 batch 的一撮，做完整评估
            stack_dim=1,
            sampler=val_dist_sampler,
            epoch_interval=1           # 每个 epoch 都验证一次
        )

        loaders.append(loader_val)

    # 返回顺序非常重要：Trainer 会按这个顺序在每个 epoch 内依次跑
    # [Train(loader.training=True), Val(loader.training=False)]
    return loaders


def get_optimizer_scheduler(
    net,
    cfg
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler._LRScheduler]:
    """
    构建优化器 optimizer + 学习率调度器 lr_scheduler。

    主要策略：
      - 把模型参数分组，给不同分支不同学习率：
            group0: 主干(Transformer、多模态融合、box_head等)
            group1: 视觉/深度 backbone，通常 lr 会乘一个缩放系数
            group2: 语言分支 (BERT / text_proj / nl_pos_embed)
        这样可以让大模型的不同部分用不同学习率，比较常见于 DETR / V-L 这种架构。

      - 根据 cfg.TRAIN.SCHEDULER ("StepLR" 或 "MultiStepLR") 来创建对应的调度器。

    依赖的 cfg 字段（需要你在 build_default_cfg() 里定义）：
      cfg.TRAIN.LR
      cfg.TRAIN.WEIGHT_DECAY
      cfg.TRAIN.BACKBONE_MULTIPLIER
      cfg.TRAIN.OPTIMIZER           # 目前只支持 "ADAMW"
      cfg.TRAIN.SCHEDULER           # "StepLR" / "MultiStepLR"
      cfg.TRAIN.LR_DROP_EPOCH
      cfg.TRAIN.MILESTONES
      cfg.TRAIN.GAMMA
      cfg.MODEL.LANGUAGE.BERT.LR
      cfg.TRAIN.TRAIN_CLS
    """

    base_lr = float(getattr(cfg.TRAIN, "LR", 1e-4))
    lang_lr = float(getattr(cfg.MODEL.LANGUAGE.BERT, "LR", 1e-5))
    train_cls_only = getattr(cfg.TRAIN, "TRAIN_CLS", False)

    # -------------------------
    # A. 按需要冻结/分组参数
    # -------------------------
    if train_cls_only:
        # 只训练分类头（其余参数全部冻结）
        print("Only training classification head. Freezing the rest.")
        params_cls = []
        for n, p in net.named_parameters():
            if "cls" in n and p.requires_grad:
                params_cls.append(p)
                print(f"[trainable cls param] {n}")
            else:
                p.requires_grad = False
        param_dicts = [{"params": params_cls}]
    else:
        group0_params = []  # 主干（transformer、多模态融合、box_head等）
        group1_params = []  # 视觉/深度 backbone
        group2_params = []  # 语言分支 (BERT/text_proj/nl_pos_embed)

        for n, p in net.named_parameters():
            if not p.requires_grad:
                continue

            # 语言支路
            if ("language_backbone" in n) or ("nl_pos_embed" in n) or ("text_proj" in n):
                group2_params.append(p)
                continue

            # 视觉/深度 backbone（但不包括 language_backbone）
            if ("backbone" in n) and ("language_backbone" not in n):
                group1_params.append(p)
                continue

            # 其他 -> group0
            group0_params.append(p)

        if is_main_process():
            print("Learnable parameter groups:")
            print(f"  group0 (main modules): {len(group0_params)} tensors")
            print(f"  group1 (vision/depth backbone): {len(group1_params)} tensors")
            print(f"  group2 (language branch): {len(group2_params)} tensors")

        param_dicts = [
            {"params": group0_params, "lr": base_lr},
            {
                "params": group1_params,
                "lr": base_lr * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
            {
                "params": group2_params,
                "lr": lang_lr * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
        ]

    # -------------------------
    # B. Optimizer
    # -------------------------
    if cfg.TRAIN.OPTIMIZER.upper() == "ADAMW":
        optimizer = torch.optim.AdamW(
            param_dicts,
            lr=base_lr,
            weight_decay=cfg.TRAIN.WEIGHT_DECAY,
        )
    else:
        raise ValueError(f"Unsupported optimizer {cfg.TRAIN.OPTIMIZER}")

    # -------------------------
    # C. LR Scheduler
    # -------------------------
    scheduler_name = cfg.TRAIN.SCHEDULER  # "StepLR" / "MultiStepLR"

    if scheduler_name == "StepLR":
        # 每隔 LR_DROP_EPOCH 个 epoch，lr *= GAMMA
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.TRAIN.LR_DROP_EPOCH,
            gamma=cfg.TRAIN.GAMMA,
        )

    elif scheduler_name == "MultiStepLR":
        # 在若干 milestone epoch 时刻，把 lr *= GAMMA
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=list(cfg.TRAIN.MILESTONES),
            gamma=cfg.TRAIN.GAMMA,
        )

    else:
        raise ValueError(
            f"Unsupported scheduler '{scheduler_name}'. "
            "Use 'StepLR' or 'MultiStepLR'."
        )

    return optimizer, lr_scheduler
