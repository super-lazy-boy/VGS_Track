# lib/train/actors/vgst_actor.py
import torch
import torch.nn.functional as F
from typing import Dict, Tuple
from types import SimpleNamespace

from .base_actor import BaseActor
from lib.utils.misc import NestedTensor


class VGSTActor(BaseActor):
    """
    VGSTActor
    ----------
    训练阶段由 Trainer 调用的“执行者 (Actor)”。它负责把 batch 组装成三路模态输入，
    调用 VGST 模型获得预测框，并计算损失 (GIoU + L1)。
    """

    def __init__(self, net: torch.nn.Module, cfg):
        super().__init__(net=net, objective=None)
        self.cfg = cfg
        self.giou_weight = cfg.TRAIN.GIOU_WEIGHT
        self.l1_weight = cfg.TRAIN.L1_WEIGHT

        # 可视化开关（更健壮：即使 cfg.TEST 不存在也不报错）
        test_ns = getattr(cfg, "TEST", SimpleNamespace(DUMP_VIZ=False))
        self.enable_viz_dump = getattr(test_ns, "DUMP_VIZ", False)
        self._last_viz = None

    # Trainer 会直接调用： loss, stats = actor(batch)
    def __call__(self, batch: Dict) -> Tuple[torch.Tensor, Dict[str, float]]:
        # 1) 视觉输入
        color_nt, depth_nt = self._extract_visual_inputs(batch)

        # 2) 文本输入 -> NestedTensor(token_ids, attn_mask)
        text_nt = self._extract_text_input(batch, device=color_nt.tensors.device)

        # 3) 调用模型（位置参数，与 VGST.forward 对齐）
        out = self.net(
            color_nt,          # RGB NestedTensor
            depth_nt,          # Depth NestedTensor
            text_nt,           # NestedTensor(token_ids, attn_mask)
            mode="train",      # 任意非 "transformer" 字符串 -> 走检测头
            run_box_head=True
        )

        # 对齐 VGST 返回： (out_dict, outputs_coord) 或 dict
        if isinstance(out, tuple):
            out_dict, _ = out
        elif isinstance(out, dict):
            out_dict = out
        else:
            raise RuntimeError("Unexpected VGST.forward output type.")

        # 4) 取 GT
        gt_bbox = self._extract_gt_bbox(batch, device=color_nt.tensors.device)

        # 5) 计算损失
        scores = out_dict.get("scores", None)
        loss, stats = self._compute_losses(out_dict["pred_boxes"], gt_bbox, scores=scores)

        aux_weight = float(getattr(self.cfg.TRAIN, "AUX_WEIGHT", 0.0))
        if aux_weight > 0 and "aux_outputs" in out_dict:
            aux_losses = []
            for aux in out_dict["aux_outputs"]:
                aux_boxes = aux["pred_boxes"]
                aux_loss, _ = self._compute_losses(aux_boxes, gt_bbox, scores=None)  # aux 不需要 scores
                aux_losses.append(aux_loss)
            if len(aux_losses) > 0:
                loss = loss + aux_weight * torch.stack(aux_losses).mean()

        # === 收集可视化材料（可选） ===
        if self.enable_viz_dump:
            viz = {}
            try:
                if isinstance(out_dict, dict) and "pred_boxes" in out_dict:
                    viz["pred_boxes"] = out_dict["pred_boxes"].detach().cpu()
                for k in ["scores", "pred_logits"]:
                    if isinstance(out_dict, dict) and k in out_dict:
                        viz["scores"] = out_dict[k].detach().cpu()
                        break
                for k in ["score_map", "response", "attn_map"]:
                    if isinstance(out_dict, dict) and k in out_dict:
                        viz["score_map"] = out_dict[k].detach().cpu()
                        break
                if "search_images" in batch and torch.is_tensor(batch["search_images"]):
                    viz["search_images"] = batch["search_images"].detach().cpu()
                if "search_anno" in batch and torch.is_tensor(batch["search_anno"]):
                    viz["search_anno"] = batch["search_anno"].detach().cpu()
                if "gt_bbox" in batch and torch.is_tensor(batch["gt_bbox"]):
                    viz["gt_bbox"] = batch["gt_bbox"].detach().cpu()
            except Exception:
                viz = {}
            self._last_viz = viz

        return loss, stats

    # ---------- 视觉模态组装 ----------
    def _extract_visual_inputs(self, batch: Dict) -> Tuple[NestedTensor, NestedTensor]:
        if "search_color" in batch and isinstance(batch["search_color"], NestedTensor):
            color_nt = batch["search_color"]
        else:
            if "search_images" not in batch:
                raise ValueError("Batch must provide 'search_color' or 'search_images' for RGB branch.")
            search_img = batch["search_images"]
            search_att = batch.get("search_att", None)
            if search_img.dim() == 5:   # (Ns,B,6,H,W) -> 取第0时刻
                search_img = search_img[0]
                if search_att is not None and search_att.dim() == 4:
                    search_att = search_att[0]
            rgb = search_img[:, :3]     # (B,3,H,W)
            if search_att is None:
                rgb_mask = torch.zeros(rgb.shape[0], rgb.shape[2], rgb.shape[3],
                                       dtype=torch.bool, device=rgb.device)
            else:
                rgb_mask = search_att.to(torch.bool)
            color_nt = NestedTensor(rgb, rgb_mask)

        if "search_depth" in batch and isinstance(batch["search_depth"], NestedTensor):
            depth_nt = batch["search_depth"]
        else:
            if "search_images" not in batch:
                raise ValueError("Batch must provide 'search_depth' or 'search_images' for depth branch.")
            search_img = batch["search_images"]
            search_att = batch.get("search_att", None)
            if search_img.dim() == 5:
                search_img = search_img[0]
                if search_att is not None and search_att.dim() == 4:
                    search_att = search_att[0]
            depth = search_img[:, 3:]   # (B,3,H,W)
            if search_att is None:
                depth_mask = torch.zeros(depth.shape[0], depth.shape[2], depth.shape[3],
                                         dtype=torch.bool, device=depth.device)
            else:
                depth_mask = search_att.to(torch.bool)
            depth_nt = NestedTensor(depth, depth_mask)

        return color_nt, depth_nt

    # ---------- 文本模态组装 ----------
    def _extract_text_input(self, batch: Dict, device: torch.device) -> NestedTensor:
        if "text_data" in batch and isinstance(batch["text_data"], NestedTensor):
            return batch["text_data"]

        token_ids = None
        attn_mask = None
        if "text_token_ids" in batch and "text_attention_mask" in batch:
            token_ids = batch["text_token_ids"]
            attn_mask = batch["text_attention_mask"]
        elif "nl_token_ids" in batch and "nl_token_masks" in batch:  # 兼容旧命名
            token_ids = batch["nl_token_ids"]
            attn_mask = batch["nl_token_masks"]
        if token_ids is None or attn_mask is None:
            raise ValueError("Missing text tokens. Expect 'text_token_ids'/'text_attention_mask' "
                             "or 'nl_token_ids'/'nl_token_masks', or pre-built 'text_data'.")

        return NestedTensor(token_ids.to(device).long(),
                            attn_mask.to(device).long())

    # ---------- 监督框 ----------
    def _extract_gt_bbox(self, batch: Dict, device: torch.device) -> torch.Tensor:
        if "gt_bbox" in batch:
            gt = batch["gt_bbox"]
        elif "search_anno" in batch:
            anno = batch["search_anno"]
            gt = anno[0] if anno.dim() == 3 else anno
        else:
            raise ValueError("Batch must contain 'gt_bbox' or 'search_anno' for box supervision.")
        return gt.to(device).float()

    # ---------- 损失 ----------
    @staticmethod
    def _box_cxcywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        cx, cy, w, h = boxes.unbind(dim=-1)
        x1 = cx - 0.5 * w
        y1 = cy - 0.5 * h
        x2 = cx + 0.5 * w
        y2 = cy + 0.5 * h
        return torch.stack([x1, y1, x2, y2], dim=-1)

    @staticmethod
    def _box_xywh_to_xyxy(boxes: torch.Tensor) -> torch.Tensor:
        x, y, w, h = boxes.unbind(dim=-1)
        x2 = x + w
        y2 = y + h
        return torch.stack([x, y, x2, y2], dim=-1)

    @staticmethod
    def _box_area(boxes_xyxy: torch.Tensor) -> torch.Tensor:
        return ((boxes_xyxy[:, 2] - boxes_xyxy[:, 0]).clamp(min=0) *
                (boxes_xyxy[:, 3] - boxes_xyxy[:, 1]).clamp(min=0))

    def _generalized_box_iou(self, boxes1_xyxy: torch.Tensor, boxes2_xyxy: torch.Tensor):
        x1 = torch.max(boxes1_xyxy[:, 0], boxes2_xyxy[:, 0])
        y1 = torch.max(boxes1_xyxy[:, 1], boxes2_xyxy[:, 1])
        x2 = torch.min(boxes1_xyxy[:, 2], boxes2_xyxy[:, 2])
        y2 = torch.min(boxes1_xyxy[:, 3], boxes2_xyxy[:, 3])
        inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
        area1 = self._box_area(boxes1_xyxy)
        area2 = self._box_area(boxes2_xyxy)
        union = area1 + area2 - inter
        iou = inter / union.clamp(min=1e-6)
        cx1 = torch.min(boxes1_xyxy[:, 0], boxes2_xyxy[:, 0])
        cy1 = torch.min(boxes1_xyxy[:, 1], boxes2_xyxy[:, 1])
        cx2 = torch.max(boxes1_xyxy[:, 2], boxes2_xyxy[:, 2])
        cy2 = torch.max(boxes1_xyxy[:, 3], boxes2_xyxy[:, 3])
        area_c = (cx2 - cx1).clamp(min=0) * (cy2 - cy1).clamp(min=0)
        giou = iou - (area_c - union) / area_c.clamp(min=1e-6)
        return giou, iou

    def _compute_losses(self, pred_boxes: torch.Tensor, gt_bbox_xywh: torch.Tensor, scores: torch.Tensor = None):
        """
        pred_boxes: [B, Nq, 4], 归一化 (cx, cy, w, h)
        gt_bbox_xywh: [B, 4],   归一化 (x,  y,  w, h)
        Top-k 负责：仅对 IoU 最高的 k 个 queries 计算回归损失（k=1 即 Top-1）
        """
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network output contains NaN. Stop Training.")

        B, Nq, _ = pred_boxes.shape

        # 预测: cxcywh -> xyxy, 并 reshape 成 [B,Nq,4]
        pred_xyxy_q = self._box_cxcywh_to_xyxy(pred_boxes).view(B, Nq, 4)

        # GT: xywh -> xyxy, 并广播到 [B,Nq,4]，再裁到 [0,1]
        gt_xyxy_1 = self._box_xywh_to_xyxy(gt_bbox_xywh)          # [B,4]
        gt_xyxy_q = gt_xyxy_1[:, None, :].repeat(1, Nq, 1).clamp(0.0, 1.0)  # [B,Nq,4]

        # 逐 query 的 IoU（展开到 [B*Nq,4] 计算后再还原）
        giou_all, iou_all = self._generalized_box_iou(
            pred_xyxy_q.reshape(-1, 4),
            gt_xyxy_q.reshape(-1, 4)
        )
        giou_all = giou_all.view(B, Nq)  # 仅为统计时可用
        iou_all  = iou_all.view(B, Nq)

        # 选择 Top-k 负责的 queries
        k = int(getattr(self.cfg.TRAIN, "TOPK_QUERIES", 1))
        k = max(1, min(k, Nq))
        topk_iou, topk_idx = torch.topk(iou_all, k=k, dim=1, largest=True, sorted=False)  # [B,k],[B,k]

        # 索引出 Top-k 的预测与 GT → [B,k,4]
        row_idx = torch.arange(B, device=pred_boxes.device).unsqueeze(1).expand(B, k)
        pred_topk = pred_xyxy_q[row_idx, topk_idx]    # [B,k,4]
        gt_topk   = gt_xyxy_q[row_idx, topk_idx]      # [B,k,4]

        # 在 Top-k 上计算回归损失（先展平到 [B*k,4]）
        giou_topk, _ = self._generalized_box_iou(
            pred_topk.reshape(-1, 4),
            gt_topk.reshape(-1, 4)
        )
        giou_loss = (1.0 - giou_topk).mean()          # Top-k 平均的 GIoU
        l1        = F.l1_loss(pred_topk, gt_topk, reduction="mean")
        total = self.giou_weight * giou_loss + self.l1_weight * l1

        score_loss = torch.tensor(0.0, device=pred_boxes.device)
        score_weight = float(getattr(self.cfg.TRAIN, "SCORE_WEIGHT", 0.0))  # 缺省不启用
        if score_weight > 0.0 and (scores is not None):
            # scores: [B,Nq]；IoU 目标用 iou_all（[B,Nq]），不回传 IoU 的梯度
            # 与 Top-k 不冲突：这里让“强者恒强”，高 IoU 的 query 得高分
            target = iou_all.detach()
            # 用 BCE 或 L2 都行；这里用 L2 更平滑
            score_loss = F.mse_loss(torch.clamp(scores, 0.0, 1.0), torch.clamp(target, 0.0, 1.0))
            total = total + score_weight * score_loss
            

        
        stats = {
            "Loss/total": total.item(),
            "Loss/giou": giou_loss.item(),
            "Loss/l1": l1.item(),
            "Loss/score": score_loss.item(),
            "IoU": iou_all.mean().item(),             # 仍统计所有 queries 的 IoU 均值便于对比
            "IoU/topk": topk_iou.mean().item(),       # 新增：Top-k 的 IoU（更贴近训练信号）
        }
        return total, stats


def build_vgst_actor(net: torch.nn.Module, cfg) -> VGSTActor:
    return VGSTActor(net=net, cfg=cfg)
