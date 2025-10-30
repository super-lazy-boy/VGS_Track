#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
/remote-home/ai2005_11/VGST/lib/test/test_inference.py

使用训练好的 VGST 模型 (.pth) 在测试集上做推理+评估。
评估指标和训练时打印的一致：总损失、GIoU损失、L1损失、IoU。

注意：
- 我们不再 import transformers / AutoTokenizer，
  所以不会再触发 libssl.so.10 的 ImportError。
- 我们直接复用训练同一条数据管线
  (CustomDataset -> VLTrackingSampler -> SPTProcessing -> LTRLoader)
  来自动生成 batch，包括文本 token 等。
"""

import os
import argparse
from collections import defaultdict

import torch
from types import SimpleNamespace

import random  # ### [MOD START] 为采样器猴子补丁里的 random.choice 使用
# ### [MOD END]


# ====== 这些模块都来自你的训练目录结构 ======
# settings: 包含 Settings 类、build_default_cfg()，负责全局配置和路径等
from lib.train.admin.settings import Settings, build_default_cfg, update_env_settings
# base_functions: 把 cfg 同步到 settings（例如裁剪参数、batchsize等）
from lib.train.base_functions import update_settings
# 数据流各环节：CustomDataset / VLTrackingSampler / SPTProcessing / LTRLoader / 以及图像增强 tfm
from lib.train.dataset import CustomDataset
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm

# 模型与训练执行器（Actor = "把一批数据丢进模型并算损失/指标"）
from lib.models.vgst.vgst import build_vgst        # 和训练时一样的构建函数
from lib.train.actors.vgst_actor import build_vgst_actor

# NestedTensor 是 VGST 前向期望的输入封装形式 (图像张量+mask)，
# 由 VGSTActor 负责组装，这里只需要 import 以便类型注解/安全
from lib.utils.misc import NestedTensor

# === 在 test_inference.py 顶部其它 import 后追加 ===
### [VIS MOD START]
import cv2
import numpy as np

# 仓库根目录（test_inference.py 位于 lib/test/ 下，两级回到根）
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
VIZ_DIR   = os.path.join(REPO_ROOT, "output", "viz_test")
os.makedirs(VIZ_DIR, exist_ok=True)
print(f"[VIZ] 将把可视化结果保存到: {VIZ_DIR}")

def _to_uint8_img(t):
    if t.dim() == 4:
        t = t[0]  # 取 batch 中第一张
    arr = t.detach().cpu().numpy()
    if arr.shape[0] >= 3:
        arr = arr[:3]
    else:
        arr = np.repeat(arr, 3, axis=0)
    arr = np.transpose(arr, (1, 2, 0))  # (H,W,C)
    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    # ★关键：保证是 C-contiguous，OpenCV 才能吃
    arr = np.ascontiguousarray(arr)
    return arr


def _xywh_to_xyxy(box):
    x, y, w, h = box
    return [x, y, x + w, y + h]

def _cxcywh_norm_to_xyxy_abs(box, W, H):
    """DETR风格 [cx,cy,w,h] 归一化坐标 -> 绝对像素xyxy"""
    cx, cy, w, h = box
    cx, cy, w, h = cx*W, cy*H, w*W, h*H
    x1, y1 = cx - w/2.0, cy - h/2.0
    x2, y2 = cx + w/2.0, cy + h/2.0
    return [x1, y1, x2, y2]

def _draw_box(img, box, color=(0,255,0), text=None):
    # ★兜底：确保传给 OpenCV 的是 C-contiguous
    img[:] = np.ascontiguousarray(img)
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    try:
        cv2.rectangle(img, (x1,y1), (x2,y2), color, 2)
        if text:
            cv2.putText(img, text, (x1, max(0, y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    except cv2.error as e:
        print("[VIZ] cv2.rectangle failed:", repr(e), "shape/dtype/flags=",
              img.shape, img.dtype, "C:", img.flags['C_CONTIGUOUS'])


def _blend_heatmap_on_image(img, heat):
    """heat: 2D 或 (1,Hh,Wh)。会resize到img大小并伪彩叠加"""
    if heat.ndim == 3:
        heat = heat[0]
    heat = heat.astype(np.float32)
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)
    H, W = img.shape[:2]
    heat = cv2.resize(heat, (W, H), interpolation=cv2.INTER_CUBIC)
    heat_color = cv2.applyColorMap((heat*255).astype(np.uint8), cv2.COLORMAP_JET)
    out = cv2.addWeighted(img, 0.6, heat_color, 0.4, 0)
    return out
### [VIS MOD END]

def build_test_loader(cfg, settings, test_root):
    """
    构建“测试集”的 DataLoader，基本复用训练的数据处理逻辑。

    数据流:
        CustomDataset(扫描 test_root 下的序列)
          -> VLTrackingSampler(按跟踪任务的方式采样 search帧/gt框/文本)
          -> SPTProcessing(裁剪、抖动、归一化成张量)
          -> LTRLoader(打成 batch)

    这里我们手动搭一份（而不是直接复用 build_dataloaders），
    这样就可以把 root 改成 test_root，并且不会碰到
    cfg.DATA.TRAIN.DATASETS_RATIO 和 DATASETS_NAME 长度不一致的问题。
    """
    print("=> [TestLoader] 开始构建测试 DataLoader ...")

    # ==== 1. 定义图像增强 / 归一化 ====
    # 注意：这基本照搬训练时的 transform 设置。
    # transform_joint: 对 template/search 成对做同步的几何增广
    transform_joint = tfm.Transform(
        tfm.ToGrayscale(probability=0.05),
        tfm.RandomHorizontalFlip(probability=0.5),
    )

    # transform_test: 对单帧做颜色抖动、翻转(概率)、归一化
    transform_test = tfm.Transform(
        tfm.ToTensorAndJitter(0.2),
        tfm.RandomHorizontalFlip_Norm(probability=0.5),
        tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD),
    )

    # ==== 2. 空间裁剪/抖动处理模块 ====
    # 这个模块在训练里叫 SPTProcessing，负责：
    #  - 根据 GT bbox 在原图上裁出 template/search 的 patch
    #  - jitter 中心/尺度
    #  - resize 到 cfg.DATA.TEMPLATE.SIZE / cfg.DATA.SEARCH.SIZE
    # 这些超参数我们已经在 update_settings() 里同步到 settings 了。
    data_processing_test = processing.SPTProcessing(
        search_area_factor=settings.search_area_factor,
        output_sz=settings.output_sz,
        center_jitter_factor=settings.center_jitter_factor,
        scale_jitter_factor=settings.scale_jitter_factor,
        mode='sequence',            # 和训练保持一致，表示按时序样本来处理
        transform=transform_test,
        joint_transform=transform_joint,
        settings=settings,
    )

    # ==== 3. 构建 CustomDataset ====
    # CustomDataset 会自动扫描 test_root 目录，读取每个序列：
    # - 彩色图 / 深度图
    # - 标注 bbox 序列
    # - 语言描述 (nlp)
    # 并在 __init__ 里统计 sequence_list / class_list。
    print(f"=> [TestLoader] 扫描测试集目录: {test_root}")
    test_dataset_list = [
        CustomDataset(
            root=test_root,
            dtype='rgbcolormap',      # 你的项目约定：RGB+深度colormap -> 6通道
            image_loader=opencv_loader,
            split_file=None,
            dataset_name="test",
        )
    ]

    # 我们给 p_datasets 一个和数据集列表等长的权重列表。
    # 训练里之前出过 "The number of weights does not match the population"
    # 就是因为长度不一致导致的。
    p_datasets = [1.0 for _ in test_dataset_list]

    # 迭代多少“样本”算完整一轮测试？
    # 为了简单，我们就复用 cfg.DATA.TRAIN.SAMPLE_PER_EPOCH
    #（相当于抽这么多条跟踪片段来评估平均效果）。
    samples_per_epoch = cfg.DATA.TRAIN.SAMPLE_PER_EPOCH

    # 采样器在训练里叫 VLTrackingSampler：
    # - 会从数据集中随机挑序列、挑帧，组装成一条 (search帧, gt bbox, 文本) 样本
    # - 会返回多模态字段：RGB+Depth张量、tokenized文本、归一化bbox等
    max_gap = getattr(cfg.DATA, "SAMPLER_MAX_INTERVAL",
                      getattr(cfg.DATA, "SAMPLER_INTERVAL", 200))
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)

    print("=> [TestLoader] 构建 VLTrackingSampler ...")
    test_sampler = sampler.VLTrackingSampler(
        datasets=test_dataset_list,
        p_datasets=[1.0 for _ in test_dataset_list],
        samples_per_epoch=samples_per_epoch,
        max_gap=max_gap,
        num_search_frames=settings.num_search,
        num_template_frames=settings.num_template,
        processing=data_processing_test,
        train_cls=train_cls,
        max_seq_len=cfg.DATA.MAX_SEQ_LENGTH,
        bert_model=cfg.MODEL.LANGUAGE.TYPE,
        bert_path=cfg.MODEL.LANGUAGE.VOCAB_PATH,
        frame_sample_mode='causal',  # 可以保留默认，也可以随便放个合法的
        mode="test",                 # 【新增: 告诉采样器我们是测试模式】
    )


    # ==== 4. 包装成 PyTorch 风格的 DataLoader (LTRLoader) ====
    # LTRLoader 是你项目里对 torch.utils.data.DataLoader 的轻封装，
    # 允许我们用 stack_dim=1 来把 batch 叠在第1维，
    # 于是 batch["search_images"] 形状可能是 (Ns, B, 6, H, W)，
    # VGSTActor 在前向时会自动把它 reshape 成 (B, 6, H, W) 再拆成RGB+Depth。:contentReference[oaicite:8]{index=8}
    print("=> [TestLoader] 构建 LTRLoader ...")
    test_loader = LTRLoader(
        name='Test',
        dataset=test_sampler,
        training=False,                       # 推理阶段，不打乱权重更新
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.TRAIN.NUM_WORKER,
        drop_last=False,
        stack_dim=1,                          # 和训练保持一致
        sampler=None,
    )

    return test_loader


def move_batch_to_device(batch, device):
    """
    把一个从 DataLoader 取出来的 batch 递归地搬到指定 device (cuda / cpu) 上。
    VGSTActor 需要的 key 包括:
      - "search_images" / "search_att"
      - "search_anno" (GT bbox)
      - "nl_token_ids" / "nl_token_masks"  (语言token)
    我们不去改这些字段的形状，只是简单 .to(device)。
    """
    out = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            out[k] = v.to(device, non_blocking=True)
        elif isinstance(v, dict):
            out[k] = {
                kk: (vv.to(device, non_blocking=True) if torch.is_tensor(vv) else vv)
                for kk, vv in v.items()
            }
        else:
            # 列表、字符串之类的meta信息（比如 seq 名字）就原样保留在CPU
            out[k] = v
    return out


def get_batch_size(batch):
    """
    估计当前 batch 的有效 batch_size，用于加权平均指标。
    我们优先用 'search_images' 这个字段来判断，因为训练/推理都会有它。

    训练里的 VGSTTrainer 也是类似按 search_images 的 (Ns,B,...) 或 (B,...) 来判断批大小，
    然后用这个 batch_size 去做平均/打印。:contentReference[oaicite:9]{index=9}
    """
    if "search_images" in batch and torch.is_tensor(batch["search_images"]):
        t = batch["search_images"]
        if t.dim() == 5:
            # 形如 (Ns, B, C, H, W)
            return t.shape[1]
        else:
            # 形如 (B, C, H, W)
            return t.shape[0]
    # 兜底：如果未来数据管线改名了
    if "nl_token_ids" in batch and torch.is_tensor(batch["nl_token_ids"]):
        return batch["nl_token_ids"].shape[0]
    raise RuntimeError("无法从 batch 推断 batch_size，请检查字段名。")


@torch.no_grad()
def evaluate_on_loader(actor, loader, device, print_interval=10):
    print("=> 开始在测试集上评估 ...")
    actor.eval()
    actor.to(device)

    total_weight = 0.0
    stat_accum = defaultdict(float)

    print("=> 1 [waiting for first batch from DataLoader]")
    for it, raw_batch in enumerate(loader, 1):
        print(f"[DATALOADER] got batch {it}")
        batch = move_batch_to_device(raw_batch, device)

        print("[ACTOR] forward start")
        loss, stats = actor(batch)

        # === 在 evaluate_on_loader() 的 for 循环里，紧跟 actor(batch) 之后 ===
        ### [VIS MOD START]
        try:
            viz = getattr(actor, "_last_viz", None)
            if isinstance(viz, dict) and len(viz) > 0:
                print(f"[VIZ] keys: {list(viz.keys())}")
                os.makedirs(VIZ_DIR, exist_ok=True)

                # 1) 取一张搜索图（先保证是 CPU Tensor -> NumPy）
                search = viz.get("search_images", None)
                if torch.is_tensor(search):
                    t = search
                    if t.dim() == 5:      # (Ns, B, C, H, W)
                        t = t[0]
                    # 这里只做粗略反归一化到 0-255
                    img = _to_uint8_img(t)     # 注意：_to_uint8_img 只收一个参数 t
                else:
                    img = None

                if img is not None:
                    H, W = img.shape[:2]

                    # 2) 画 GT（统一先转成 NumPy 列表再判断）
                    gt = viz.get("search_anno", None)
                    if torch.is_tensor(gt):
                        g = gt[0] if gt.dim() > 2 else gt
                        g = (g[0] if g.dim() > 1 else g).detach().cpu().numpy()
                        g_list = g.tolist()
                        # 检查 NaN
                        if not any([np.isnan(x) for x in g_list]):
                            gxyxy = _xywh_to_xyxy(g_list)
                            _draw_box(img, gxyxy, color=(255, 255, 255), text="GT")

                    # 3) 画 Pred（先转 NumPy 再进行数值范围判断）
                    pb = viz.get("pred_boxes", None)
                    if torch.is_tensor(pb):
                        p = pb[0]
                        p = p[0] if p.dim() > 1 else p
                        p = p.detach().cpu().numpy()
                        if p.shape[-1] == 4:
                            p_min = float(p.min())
                            p_max = float(p.max())
                            if (p_max <= 1.0001) and (p_min >= -0.0001):
                                pxyxy = _cxcywh_norm_to_xyxy_abs(p, W, H)
                            else:
                                pxyxy = _xywh_to_xyxy(p.tolist())
                            _draw_box(img, pxyxy, color=(0, 255, 0), text="Pred")

                    # 4) Heatmap（转为 NumPy 数组后再处理）
                    heat = viz.get("score_map", None)
                    if heat is not None:
                        h = heat
                        if torch.is_tensor(h):
                            h = h.detach().cpu().numpy()
                        h = np.array(h)
                        if h.ndim == 4:
                            h = h[0]
                        img_hm = _blend_heatmap_on_image(img.copy(), h)
                        heat_path = os.path.join(VIZ_DIR, f"batch{it:04d}_heat.png")
                        ok = cv2.imwrite(heat_path, img_hm)
                        print(f"[VIZ] save heat -> {heat_path} ({'OK' if ok else 'FAIL'})")

                    # 5) 保存结果图
                    box_path = os.path.join(VIZ_DIR, f"batch{it:04d}_box.png")
                    ok = cv2.imwrite(box_path, img)
                    print(f"[VIZ] save box  -> {box_path} ({'OK' if ok else 'FAIL'})")
                else:
                    print("[VIZ] no valid search image to draw.")
            else:
                print("[VIZ] _last_viz is empty")
        except Exception as _e:
            print("[VIZ] dump failed:", repr(_e))
        ### [VIS MOD END]



        print("[ACTOR] forward done")

        bs = get_batch_size(batch)
        total_weight += float(bs)

        for k, v in stats.items():
            stat_accum[k] += float(v) * float(bs)

        if it % print_interval == 0:
            avg_preview = {k: stat_accum[k] / max(total_weight, 1.0)
                           for k in stat_accum.keys()}
            print(f"[Test iter {it}] "
                  f"Loss/total: {avg_preview.get('Loss/total', 0):.4f}, "
                  f"GIoU: {avg_preview.get('Loss/giou', 0):.4f}, "
                  f"L1: {avg_preview.get('Loss/l1', 0):.4f}, "
                  f"IoU: {avg_preview.get('IoU', 0):.4f}")

    final_stats = {k: stat_accum[k] / max(total_weight, 1.0)
                   for k in stat_accum.keys()}
    return final_stats



def load_checkpoint_into_model(net, ckpt_path, device):
    """
    把 .pth checkpoint 的权重加载进模型。

    我们兼容两种常见保存格式：
      A) {'net': state_dict, 'epoch': N, ...}
      B) 直接就是 state_dict
    """
    print(f"=> Loading checkpoint from {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)

    if isinstance(ckpt, dict) and "net" in ckpt:
        net.load_state_dict(ckpt["net"], strict=False)
        epoch = ckpt.get("epoch", None)
    else:
        net.load_state_dict(ckpt, strict=False)
        epoch = None

    print("=> Checkpoint loaded.")
    return epoch


def main():
    parser = argparse.ArgumentParser(
        description="VGST 测试集推理与评估 (无 transformers 依赖版)"
    )
    parser.add_argument(
        "-ckpt", "--ckpt", required=True,
        help="训练完成后保存的 .pth (例如 VGST_best.pth)"
    )
    parser.add_argument(
        "--test_root", required=True,
        help="测试集根目录，例如 /remote-home/ai2005_11/VGST/data/test"
    )
    parser.add_argument(
        "--device", default="cuda:0",
        help="推理所用的 device，例如 cuda:0 或 cpu"
    )
    args = parser.parse_args()

    # -------------------------------------------------
    # 1) 初始化 settings / cfg
    # -------------------------------------------------
    # Settings: 训练时全局上下文，里面放了 env 路径、日志信息、batchsize、裁剪参数等。:contentReference[oaicite:11]{index=11}
    settings = Settings()

    # build_default_cfg(): 构造一个包含模型结构、训练超参、数据增强参数等的 cfg。:contentReference[oaicite:12]{index=12}
    cfg = build_default_cfg()
    # [VIZ PATCH] 确保存在 TEST 命名空间
    if not hasattr(cfg, "TEST") or cfg.TEST is None:
        cfg.TEST = SimpleNamespace()
    cfg.TEST.DUMP_VIZ = True  # 强制打开可视化

    # 我们把 cfg 存回 settings，保持和训练一致的接口习惯
    settings.cfg = cfg

    # 手动更新一下 env (尤其是 save_dir)，防止写日志时报错
    update_env_settings(settings)

    # 同步 cfg 里的关键信息（裁剪大小、jitter参数、batchsize、print_interval...）
    # 到 settings 里。VGSTTrainer / SPTProcessing / dataloader 都会用这些字段。:contentReference[oaicite:13]{index=13}
    update_settings(settings, cfg)

    # 我们的评估不用分布式，就直接用用户传进来的 device
    device = torch.device(args.device)
    settings.local_rank = -1
    settings.num_gpus = 1

    # -------------------------------------------------
    # 2) 构建 Model + Actor，并加载 checkpoint
    # -------------------------------------------------
    # build_vgst(cfg): 根据 cfg 构建 VGST 模型骨干、融合Transformer、检测头等。
    net = build_vgst(cfg)

    # VGSTActor: 训练/评估阶段的统一入口。
    # 它会：
    #   - 把 batch 拆成 RGB NestedTensor / Depth NestedTensor / 文本 NestedTensor
    #   - 调模型前向得到预测框
    #   - 用 GT 框算 GIoU + L1，返回 loss 和统计字典。:contentReference[oaicite:14]{index=14}
    print(f"=> Building VGSTActor for testing...")
    actor = build_vgst_actor(net, cfg)

    # 加载 checkpoint 权重
    start_epoch = load_checkpoint_into_model(net, args.ckpt, device)
    if start_epoch is not None:
        print(f"=> checkpoint epoch tag = {start_epoch}")

    # -------------------------------------------------
    # 3) 构建 test_loader
    # -------------------------------------------------
    test_loader = build_test_loader(cfg, settings, test_root=args.test_root)

    # -------------------------------------------------
    # 4) 评估循环：前向推理 + 统计指标
    # -------------------------------------------------
    final_stats = evaluate_on_loader(
        actor=actor,
        loader=test_loader,
        device=device,
        print_interval=settings.print_interval,   # 和训练打印频率一致
    )

    # -------------------------------------------------
    # 5) 输出最终平均结果
    # -------------------------------------------------
    print("======== Final Test Metrics ========")
    for k, v in final_stats.items():
        print(f"{k}: {v:.6f}")
    print("====================================")


if __name__ == "__main__":
    main()
