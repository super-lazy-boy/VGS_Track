# lib/train/run_training.py
#
# 训练启动入口 (命令行)
# 步骤：
#   1. 解析命令行参数 (脚本名/实验名/保存目录/分布式rank/随机种子等)
#   2. 设定随机种子、cudnn
#   3. 初始化 Settings() + cfg = build_default_cfg()
#   4. 调用 train_script.run(settings) 开始训练
#
# 对比原始版本：
#   - 不再依赖 YAML/外部 config_module
#   - cfg 全部由 build_default_cfg() 生成，并保存到 settings.cfg
#   - 分布式训练(local_rank!=-1)时仍然使用 DDP 初始化
#
# 运行示例（单卡）:
#   python -m lib.train.run_training --script vgst --config exp1 --save_dir ./output --seed 42
#
# 运行示例（多卡，假设使用 torchrun/torch.distributed.launch 传入 local_rank）:
#   python -m lib.train.run_training --script vgst --config exp1 --local_rank 0 --save_dir ./output --seed 42

import os
import argparse
import random
import numpy as np
import torch
import torch.backends.cudnn
import torch.distributed as dist
import cv2 as cv

from lib.train.admin.settings import Settings, build_default_cfg
from lib.train import train_script


def init_seeds(seed):
    """
    为了可复现：固定 Python / NumPy / PyTorch 的随机性
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run_training(script_name,
                 config_name,
                 cudnn_benchmark=True,
                 local_rank=-1,
                 save_dir=None,
                 base_seed=42,
                 use_lmdb=False,
                 script_name_prv=None,
                 config_name_prv=None,
                 distill=False,
                 script_teacher=None,
                 config_teacher=None):
    """
    封装整个训练启动过程。
    Args:
        script_name(str): 实验脚本名（只作为标识和log/ckpt路径的一部分）
        config_name(str): 配置名（同上）
        cudnn_benchmark(bool): 是否启用cudnn benchmark
        local_rank(int): DDP本地rank；-1代表单卡
        save_dir(str): checkpoint和日志基础目录
        base_seed(int): 随机种子
        use_lmdb(bool): 数据是否LMDB（透传给Settings）
        script_name_prv, config_name_prv: 旧模型(蒸馏/finetune)可选
        distill(bool): 是否蒸馏
        script_teacher, config_teacher: 教师模型信息
    """

    if save_dir is None:
        print("save_dir not given. Default to ./output")
        save_dir = "./output"

    # 为了避免 OpenCV 在多线程下崩溃
    cv.setNumThreads(0)

    # 设置 cudnn 行为
    torch.backends.cudnn.benchmark = cudnn_benchmark

    # ========== 随机种子 ==========
    if base_seed is not None:
        if local_rank != -1:
            init_seeds(base_seed + local_rank)
        else:
            init_seeds(base_seed)

    # ========== 准备 Settings ==========
    settings = Settings()
    settings.script_name = script_name
    settings.config_name = config_name
    # 用 script_name/config_name 组合出一个 "实验标识路径"
    settings.project_path = f'train/{script_name}/{config_name}'

    # 对蒸馏/微调的支持（可选）
    if script_name_prv is not None and config_name_prv is not None:
        settings.project_path_prv = f'train/{script_name_prv}/{config_name_prv}'

    if distill:
        settings.distill = True
        settings.script_teacher = script_teacher
        settings.config_teacher = config_teacher
        if script_teacher is not None and config_teacher is not None:
            settings.project_path_teacher = f'train/{script_teacher}/{config_teacher}'

    settings.local_rank = local_rank
    settings.save_dir = os.path.abspath(save_dir)
    settings.use_lmdb = use_lmdb

    # ========== 构建 cfg 并挂到 settings 上 ==========
    # cfg 内含模型结构、训练超参、数据增强参数等
    cfg = build_default_cfg()
    settings.cfg = cfg

    # 其余诸如 tensorboard/workspace/checkpoints 路径会在
    # VGSTTrainer/BaseTrainer 内部按 settings.save_dir 拼出来

    # ========== 真正进入训练 ==========
    train_script.run(settings)


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='Run VGST training.')

    # 必填-ish
    parser.add_argument('--script', type=str, default='vgst',
                        help='Name of the train script (used for bookkeeping only).')
    parser.add_argument('--config', type=str, default='vgst',
                        help='Name of the experiment config (used for bookkeeping only).')

    # 训练配置
    parser.add_argument('--cudnn_benchmark', type=int, default=1,
                        help='Enable cudnn benchmark (1) or not (0).')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='local rank for DDP. -1 means single GPU.')
    parser.add_argument('--save_dir', type=str, default='./output',
                        help='Where to save checkpoints and logs.')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed.')
    parser.add_argument('--use_lmdb', type=int, choices=[0, 1], default=0,
                        help='Whether datasets are in LMDB format.')

    # 旧模型/蒸馏（可选）
    parser.add_argument('--script_prv', type=str, default=None,
                        help='Prev model script name (for finetune/distill).')
    parser.add_argument('--config_prv', type=str, default=None,
                        help='Prev model config name (for finetune/distill).')

    parser.add_argument('--distill', type=int, choices=[0, 1], default=0,
                        help='Use knowledge distillation?')
    parser.add_argument('--script_teacher', type=str, default=None,
                        help='Teacher script name.')
    parser.add_argument('--config_teacher', type=str, default=None,
                        help='Teacher config file name.')

    args = parser.parse_args()

    # ========== 分布式初始化 (如需要) ==========
    if args.local_rank != -1:
        # 使用 NCCL 后端初始化进程组
        dist.init_process_group(backend='nccl')
        torch.cuda.set_device(args.local_rank)
    else:
        # 单卡场景默认使用 GPU:0 如果可用
        if torch.cuda.is_available():
            torch.cuda.set_device(0)

    # ========== 启动训练主过程 ==========
    run_training(
        script_name=args.script,
        config_name=args.config,
        cudnn_benchmark=bool(args.cudnn_benchmark),
        local_rank=args.local_rank,
        save_dir=args.save_dir,
        base_seed=args.seed,
        use_lmdb=bool(args.use_lmdb),
        script_name_prv=args.script_prv,
        config_name_prv=args.config_prv,
        distill=bool(args.distill),
        script_teacher=args.script_teacher,
        config_teacher=args.config_teacher
    )


if __name__ == '__main__':
    main()
