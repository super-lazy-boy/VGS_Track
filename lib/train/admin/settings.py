import os
from types import SimpleNamespace
from lib.train.admin.environment import env_settings

###############################################################################
# Settings
# -----------------------------------------------------------------------------
# Settings() 负责“运行时环境”（日志路径、tensorboard 路径、保存目录等）
# build_default_cfg() 负责“训练/模型的所有超参数”，包括我们这次新增的
# 训练集/验证集路径。
###############################################################################


class Settings:
    """
    训练运行时的环境配置（不含模型结构超参数，超参数放 cfg 里）。

    重要字段：
        self.env                - 来自 env_settings()，包含 data_dir / save_dir /
                                  tensorboard_dir 等路径
        self.project_path       - 实验名（例如 'exp1'），run_training.py 会设置
        self.local_rank         - DDP 训练时的 GPU rank（-1 表示单卡/主进程）
        self.save_dir           - 输出 checkpoint / 日志的根目录
        self.cfg                - 训练/模型的所有详细超参数 (SimpleNamespace 树)
                                  -> run_training.py 会把 build_default_cfg() 的返回值
                                     赋给 settings.cfg
    """

    def __init__(self):
        # 1. 拿到环境默认目录配置
        env = env_settings()

        # 2. 做好 fallback，防止 env_settings() 里没给
        #    root_dir = 项目根目录 (大致推到 lib/../.. 的上一级)
        root_dir = os.path.realpath(
            os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../..")
        )
        default_data_dir = os.path.join(root_dir, "data")
        default_save_dir = os.path.join(root_dir, "output")
        default_tensorboard_dir = os.path.join(root_dir, "tensorboard")

        if not hasattr(env, "data_dir") or env.data_dir is None:
            env.data_dir = default_data_dir
        if not hasattr(env, "save_dir") or env.save_dir is None:
            env.save_dir = default_save_dir
        if not hasattr(env, "tensorboard_dir") or env.tensorboard_dir is None:
            env.tensorboard_dir = default_tensorboard_dir

        # 3. 确保目录存在
        os.makedirs(env.data_dir, exist_ok=True)
        os.makedirs(env.save_dir, exist_ok=True)
        os.makedirs(env.tensorboard_dir, exist_ok=True)

        # 4. 绑定到 settings
        self.env = env

        # 这些字段会在 run_training.py 里进一步覆盖
        self.project_path = "default_experiment"
        self.script_name = "vgst"
        self.config_name = "default_config"

        # 分布式 / 设备信息
        self.local_rank = -1
        self.num_gpus = 1

        # 日志 & 输出
        self.log_dir = env.save_dir
        self.log_file = None  # run_training 会根据 experiment name 生成
        self.save_dir = env.save_dir
        self.tensorboard_dir = env.tensorboard_dir

        # 打印等杂项
        self.description = ""
        self.print_interval = 10
        self.grad_clip_norm = 0.1

        # dataloader 出来的 batch 是否自动搬到 GPU
        self.move_data_to_gpu = True

        # AMP (自动混合精度) 的开关，Trainer 也会参考 cfg.TRAIN.AMP
        self.use_amp = False

        # cfg: 真正的训练/模型配置树。
        self.cfg = None


def build_default_cfg():
    """
    返回一个层级化的 SimpleNamespace `cfg`，包含模型结构、优化器、数据等超参数。
    之后所有组件都只读 cfg，而不是到处硬编码。

    我们在这里也加入了『训练集路径』『验证集路径』，方便 dataloader 构建函数使用。
    """

    cfg = SimpleNamespace()

    # ----------------------------------------------------------------------
    # MODEL 相关
    # ----------------------------------------------------------------------
    cfg.MODEL = SimpleNamespace()

    # 模型主维度
    cfg.MODEL.HIDDEN_DIM = 256
    cfg.MODEL.HEAD_DIM = 256

    # 位置编码
    cfg.MODEL.POSITION_EMBEDDING = "sine"

    # backbone 训练相关
    cfg.MODEL.TRAIN_BACKBONE = True
    cfg.MODEL.RETURN_INTERMEDIATE_LAYERS = False

    # decoder 查询向量个数（跟跟踪目标数量相关，通常=1）
    cfg.MODEL.NUM_QUERIES = 1
    cfg.MODEL.NUM_OBJECT_QUERIES = cfg.MODEL.NUM_QUERIES  # 兼容旧字段

    # 检测头类型
    cfg.MODEL.HEAD_TYPE = "MLP"

    # 参数初始化策略
    cfg.MODEL.INIT = "trunc_normal"

    # ------------------ 深度分支 backbone ------------------
    cfg.MODEL.BACKBONE = SimpleNamespace()
    cfg.MODEL.BACKBONE.NAME = "resnet18"
    cfg.MODEL.BACKBONE.DILATION = False
    cfg.MODEL.BACKBONE.PRETRAINED = ""

    # ------------------ RGB Transformer 分支 ------------------
    cfg.MODEL.VL = SimpleNamespace()
    cfg.MODEL.VL.HIDDEN_DIM = cfg.MODEL.HIDDEN_DIM
    cfg.MODEL.VL.DROPOUT = 0.1
    cfg.MODEL.VL.NHEAD = 8
    cfg.MODEL.VL.DIM_FEEDFORWARD = 2048
    cfg.MODEL.VL.ENC_LAYERS = 6
    cfg.MODEL.VL.NORM_BEFORE = False
    cfg.MODEL.VL.ACTIVATION = "relu"
    cfg.MODEL.VL.INIT = "trunc_normal"
    cfg.MODEL.VL.DIVIDE_NORM = False

    # ------------------ 多模态 Transformer (color/depth/text 融合+解码) ------------------
    cfg.MODEL.TRANSFORMER = SimpleNamespace()
    cfg.MODEL.TRANSFORMER.DROPOUT = 0.1
    cfg.MODEL.TRANSFORMER.NHEADS = 8
    cfg.MODEL.TRANSFORMER.DIM_FEEDFORWARD = 2048
    cfg.MODEL.TRANSFORMER.ENC_LAYERS = 6
    cfg.MODEL.TRANSFORMER.FUS_LAYERS = 4
    cfg.MODEL.TRANSFORMER.DEC_LAYERS = 6
    cfg.MODEL.TRANSFORMER.PRE_NORM = False
    cfg.MODEL.TRANSFORMER.DIVIDE_NORM = False
    cfg.MODEL.TRANSFORMER.RETURN_INTERMEDIATE_DEC = False

    # ------------------ 自然语言(BERT)分支 ------------------
    cfg.MODEL.LANGUAGE = SimpleNamespace()
    cfg.MODEL.LANGUAGE.TYPE = "bert-base-uncased"   # 直接走 huggingface/pytorch 预训练
    cfg.MODEL.LANGUAGE.IMPLEMENT = "pytorch"
    cfg.MODEL.LANGUAGE.PATH = ""
    cfg.MODEL.LANGUAGE.VOCAB_PATH = "./vocab.txt"

    cfg.MODEL.LANGUAGE.BERT = SimpleNamespace()
    cfg.MODEL.LANGUAGE.BERT.HIDDEN_DIM = 768              # bert-base-uncased 的隐藏维度
    cfg.MODEL.LANGUAGE.BERT.MAX_QUERY_LEN = 256           # 文本最大 token 数
    cfg.MODEL.LANGUAGE.BERT.ENC_NUM = 12                  # 取第几层特征
    cfg.MODEL.LANGUAGE.BERT.LR = 1e-5                     # >0 表示微调 BERT

    # ------------------ HEAD 派生信息 ------------------
    cfg.MODEL.HEAD = SimpleNamespace()
    cfg.MODEL.HEAD.STRIDE = 16      # backbone 输出 stride
    cfg.MODEL.HEAD.FEAT_SZ = 16     # 例如搜索区域192/stride16≈12，这里用16保持一致性

    # ----------------------------------------------------------------------
    # DATA 相关
    # ----------------------------------------------------------------------
    cfg.DATA = SimpleNamespace()

    # 标准化参数
    cfg.DATA.MEAN = [0.485, 0.456, 0.406]
    cfg.DATA.STD = [0.229, 0.224, 0.225]

    # 模板帧 (template) 裁剪/抖动
    cfg.DATA.TEMPLATE = SimpleNamespace()
    cfg.DATA.TEMPLATE.SIZE = 128
    cfg.DATA.TEMPLATE.FACTOR = 2.0
    cfg.DATA.TEMPLATE.CENTER_JITTER = 3.0
    cfg.DATA.TEMPLATE.SCALE_JITTER = 0.3

    # 搜索帧 (search) 裁剪/抖动
    cfg.DATA.SEARCH = SimpleNamespace()
    cfg.DATA.SEARCH.SIZE = 192
    cfg.DATA.SEARCH.FACTOR = 4.0
    cfg.DATA.SEARCH.CENTER_JITTER = 3.0
    cfg.DATA.SEARCH.SCALE_JITTER = 0.3

    # 文本序列上限
    cfg.DATA.MAX_SEQ_LENGTH = 256

    # 采样器设置
    cfg.DATA.SAMPLER_MODE = "default"
    cfg.DATA.SAMPLER_INTERVAL = 1
    cfg.DATA.SAMPLER_MAX_INTERVAL = [10]

    # ------------------ 训练集配置 ------------------
    cfg.DATA.TRAIN = SimpleNamespace()
    cfg.DATA.TRAIN.DATASETS_NAME = ["TrainSet"]           # 名字可用于区分不同子数据集
    cfg.DATA.TRAIN.DATASETS_RATIO = [1]                # 采样比例
    cfg.DATA.TRAIN.SAMPLE_PER_EPOCH = 1000                # 每个epoch想采多少样本
    # 新增：训练集的根路径（你的要求）
    cfg.DATA.TRAIN.ROOT = "/remote-home/ai2005_11/VGST/data/TrainSet"

    # ------------------ 验证集配置（新增） ------------------
    cfg.DATA.VAL = SimpleNamespace()
    cfg.DATA.VAL.DATASETS_NAME = ["ValidationSet"]
    cfg.DATA.VAL.DATASETS_RATIO = [1]
    cfg.DATA.VAL.SAMPLE_PER_EPOCH = 1000
    # 新增：验证集的根路径（你的要求）
    cfg.DATA.VAL.ROOT = "/remote-home/ai2005_11/VGST/data/ValidationSet"

    # ----------------------------------------------------------------------
    # TRAIN 相关 (优化器 / lr / 训练轮数 / AMP / grad clip ...)
    # ----------------------------------------------------------------------
    cfg.TRAIN = SimpleNamespace()

    # 打印频率
    cfg.TRAIN.PRINT_INTERVAL = 10

    # dataloader 基本参数
    cfg.TRAIN.BATCH_SIZE = 1
    cfg.TRAIN.NUM_WORKER = 8

    # AMP 自动混合精度
    cfg.TRAIN.AMP = True

    # 深监督、多层 loss
    cfg.TRAIN.DEEP_SUPERVISION = False

    # backbone 的学习率缩放倍数
    cfg.TRAIN.BACKBONE_MULTIPLIER = 0.1

    # 是否冻结 backbone BN
    cfg.TRAIN.FREEZE_BACKBONE_BN = False

    # 蒸馏相关
    cfg.TRAIN.DISTILL = False
    cfg.TRAIN.DISTILL_LOSS_TYPE = "KL"

    # 是否训练分类头
    cfg.TRAIN.TRAIN_CLS = False

    # 训练的总 epoch 数
    cfg.TRAIN.EPOCH = 50

    # 回归框 loss 权重
    cfg.TRAIN.GIOU_WEIGHT = 5.0
    cfg.TRAIN.L1_WEIGHT = 1.0

    # 基础优化器 / 学习率调度
    cfg.TRAIN.LR = 1e-4
    cfg.TRAIN.WEIGHT_DECAY = 1e-4
    cfg.TRAIN.OPTIMIZER = "AdamW"

    cfg.TRAIN.SCHEDULER = "StepLR"
    cfg.TRAIN.LR_DROP_EPOCH = 40
    cfg.TRAIN.GAMMA = 0.1
    cfg.TRAIN.MILESTONES = [30, 40]

    # 梯度裁剪阈值
    cfg.TRAIN.GRAD_CLIP_NORM = 0.1

    cfg.TRAIN.TOPK_QUERIES =1        # 方案A：Top-1；也可设 5
    cfg.TRAIN.SCORE_WEIGHT =0.2      # 方案B：score 蒸馏权重；0 代表不启用
    cfg.TRAIN.AUX_WEIGHT =0.5        # 方案D：aux 监督权重；0 代表不启用
    

    return cfg


def update_env_settings(settings: Settings):
    """
    如果需要根据外部传参动态重写 settings.env，可以在这里做统一修改。
    这里我们至少确保保存目录存在。
    """
    os.makedirs(settings.env.save_dir, exist_ok=True)
    return settings
