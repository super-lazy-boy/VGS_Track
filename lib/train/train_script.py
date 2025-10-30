import torch
import os

from lib.train.base_functions import (
    update_settings,
    get_optimizer_scheduler,
    build_dataloaders,
)
from lib.models.vgst.vgst import build_vgst
from lib.train.trainers.trainer import build_vgst_trainer
from lib.train.actors.vgst_actor import build_vgst_actor  # 你项目里的 actor
# 注意：vgst_actor.py 负责把数据组织好并调用 net(...) 计算loss


def run(settings):
    """
    真正的训练流程（student 模型）。
    这里假设 run_training.py 已经：
        - 构造了 Settings()
        - settings.cfg = build_default_cfg()
    所以我们只需要读取 settings.cfg，而不用再去解析 YAML 或其他外部配置文件。
    """

    # 一个简短的描述，Trainer 会写进 TensorBoard
    settings.description = 'Training script for multi-modal VGST tracking.'

    # 读取全局 cfg（所有模块共享的超参数）
    cfg = settings.cfg

    # 把 cfg 中的一些训练相关常量（例如打印间隔、grad clip）同步给 settings
    update_settings(settings, cfg)

    # --------------------------
    # 1. 构建模型
    # --------------------------
    print("=> Building VGST model...")
    net = build_vgst(cfg)  # 返回一个 VGST(nn.Module)

    # DDP / 多GPU 支持
    device = settings.device if hasattr(settings, 'device') else torch.device('cuda:0')
    net = net.to(device)

    # --------------------------
    # 2. 构建数据加载器
    # --------------------------
    print("=> Building data loaders...")
    loaders = build_dataloaders(cfg, settings)

    # --------------------------
    # 3. 构建优化器 & 学习率调度器
    # --------------------------
    print("=> Building optimizer and scheduler...")
    optimizer, lr_scheduler = get_optimizer_scheduler(net, cfg)

    # --------------------------
    # 4. 构建 Actor 和 Trainer
    # --------------------------
    # Actor 负责：
    #   - 前向传播调用 net(...)
    #   - 计算 loss
    #   - 返回 {loss总和, 各种分支loss统计}
    print("=> Building actor and trainer...")
    actor = build_vgst_actor(net, cfg)

    trainer = build_vgst_trainer(
        actor=actor,
        loaders=loaders,
        optimizer=optimizer,
        settings=settings,
        lr_scheduler=lr_scheduler,
        use_amp=cfg.TRAIN.AMP
    )

    # --------------------------
    # 5. 训练循环
    # --------------------------
    print("=> Start training loop.")
    # trainer.train() 里会跑多个 epoch
    trainer.train(cfg.TRAIN.EPOCH)
