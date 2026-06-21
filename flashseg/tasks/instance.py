"""Instance segmentation task.

Provides instance-level mask prediction, per-instance ID assignment,
and Mask AP evaluation metric.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class InstancePrediction:
    """A single instance prediction."""

    instance_id: int
    class_id: int
    score: float
    mask: np.ndarray  # (H, W) binary mask
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2


def _mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute IoU between two binary masks."""
    intersection = (mask1 & mask2).sum()
    union = (mask1 | mask2).sum()
    return float(intersection) / max(float(union), 1e-6)


def _mask_iou_matrix(
    masks1: List[np.ndarray],
    masks2: List[np.ndarray],
) -> np.ndarray:
    """Compute pairwise mask IoU between two sets of masks."""
    M, N = len(masks1), len(masks2)
    iou_mat = np.zeros((M, N), dtype=np.float64)
    for i in range(M):
        for j in range(N):
            iou_mat[i, j] = _mask_iou(masks1[i], masks2[j])
    return iou_mat


def compute_mask_ap(
    pred_masks: List[np.ndarray],
    pred_scores: List[float],
    pred_classes: List[int],
    gt_masks: List[np.ndarray],
    gt_classes: List[int],
    iou_thresholds: Optional[List[float]] = None,
) -> Dict[str, float]:
    """Compute Mask AP at multiple IoU thresholds.

    Args:
        pred_masks: List of (H, W) predicted binary masks.
        pred_scores: Confidence scores per prediction.
        pred_classes: Class IDs per prediction.
        gt_masks: List of (H, W) ground-truth binary masks.
        gt_classes: Class IDs per GT mask.
        iou_thresholds: IoU thresholds (default: 0.5:0.05:0.95).

    Returns:
        Dict with ``'AP'`` (averaged), ``'AP50'``, ``'AP75'``, and per-threshold APs.
    """
    if iou_thresholds is None:
        iou_thresholds = [round(0.5 + 0.05 * i, 2) for i in range(10)]

    if len(pred_masks) == 0:
        return {"AP": 0.0, "AP50": 0.0, "AP75": 0.0}

    sorted_idx = np.argsort(pred_scores)[::-1]
    pred_masks = [pred_masks[i] for i in sorted_idx]
    pred_scores = [pred_scores[i] for i in sorted_idx]
    pred_classes = [pred_classes[i] for i in sorted_idx]

    all_classes = set(pred_classes) | set(gt_classes)
    aps_per_threshold = {}

    for iou_thresh in iou_thresholds:
        class_aps = []
        for cls in all_classes:
            cls_pred_idx = [i for i, c in enumerate(pred_classes) if c == cls]
            cls_gt_idx = [i for i, c in enumerate(gt_classes) if c == cls]

            if not cls_gt_idx:
                continue

            n_gt = len(cls_gt_idx)
            cls_pred_masks = [pred_masks[i] for i in cls_pred_idx]
            cls_gt_masks = [gt_masks[i] for i in cls_gt_idx]
            cls_scores = [pred_scores[i] for i in cls_pred_idx]

            if not cls_pred_masks:
                class_aps.append(0.0)
                continue

            iou_mat = _mask_iou_matrix(cls_pred_masks, cls_gt_masks)

            tp = np.zeros(len(cls_pred_masks))
            fp = np.zeros(len(cls_pred_masks))
            gt_matched = set()

            for p_idx in range(len(cls_pred_masks)):
                if iou_mat.shape[1] == 0:
                    fp[p_idx] = 1
                    continue

                best_gt = iou_mat[p_idx].argmax()
                best_iou = iou_mat[p_idx, best_gt]

                if best_iou >= iou_thresh and best_gt not in gt_matched:
                    tp[p_idx] = 1
                    gt_matched.add(best_gt)
                else:
                    fp[p_idx] = 1

            tp_cumsum = np.cumsum(tp)
            fp_cumsum = np.cumsum(fp)
            recalls = tp_cumsum / n_gt
            precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

            ap = _compute_ap(recalls, precisions)
            class_aps.append(ap)

        aps_per_threshold[iou_thresh] = np.mean(class_aps) if class_aps else 0.0

    ap = np.mean(list(aps_per_threshold.values()))
    ap50 = aps_per_threshold.get(0.5, 0.0)
    ap75 = aps_per_threshold.get(0.75, 0.0)

    result = {"AP": float(ap), "AP50": float(ap50), "AP75": float(ap75)}
    for thresh, val in aps_per_threshold.items():
        result[f"AP{int(thresh * 100)}"] = float(val)

    return result


def _compute_ap(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute AP using the 101-point interpolation."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([0.0], precisions, [0.0]))

    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    recall_points = np.linspace(0, 1, 101)
    ap = 0.0
    for r in recall_points:
        idx = np.where(mrec >= r)[0]
        if len(idx) > 0:
            ap += mpre[idx[0]]
    return ap / 101.0


def _mask_to_bbox(mask: np.ndarray) -> Tuple[int, int, int, int]:
    """Get bounding box from a binary mask."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return (0, 0, 0, 0)
    return (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))


class InstanceSegmentor:
    """Instance segmentation wrapper.

    Takes model predictions and produces per-instance masks with unique IDs.

    Args:
        num_classes: Number of semantic classes.
        score_threshold: Minimum confidence for keeping predictions.
        mask_threshold: Sigmoid threshold for binary mask.
        nms_iou_threshold: Mask IoU threshold for NMS.
        max_instances: Maximum number of instances per image.
    """

    def __init__(
        self,
        num_classes: int = 80,
        score_threshold: float = 0.5,
        mask_threshold: float = 0.5,
        nms_iou_threshold: float = 0.5,
        max_instances: int = 100,
    ):
        self.num_classes = num_classes
        self.score_threshold = score_threshold
        self.mask_threshold = mask_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_instances = max_instances
        self._next_id = 1

    def reset_ids(self):
        self._next_id = 1

    def process(
        self,
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> List[InstancePrediction]:
        """Convert model outputs to instance predictions.

        Args:
            pred_logits: (Q, num_classes+1) class logits per query.
            pred_masks: (Q, H', W') mask logits per query.
            image_size: (H, W) original image size.

        Returns:
            List of ``InstancePrediction`` objects.
        """
        scores, labels = pred_logits.softmax(-1)[:, :-1].max(-1)

        keep = scores > self.score_threshold
        scores = scores[keep]
        labels = labels[keep]
        masks_logits = pred_masks[keep]

        if len(scores) == 0:
            return []

        masks = F.interpolate(
            masks_logits.unsqueeze(1).float(),
            size=image_size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(1)

        binary_masks = (masks.sigmoid() > self.mask_threshold).cpu().numpy().astype(bool)
        scores_np = scores.cpu().numpy()
        labels_np = labels.cpu().numpy()

        keep_idx = self._mask_nms(binary_masks, scores_np)
        keep_idx = keep_idx[:self.max_instances]

        instances = []
        for idx in keep_idx:
            mask = binary_masks[idx]
            bbox = _mask_to_bbox(mask)
            inst = InstancePrediction(
                instance_id=self._next_id,
                class_id=int(labels_np[idx]),
                score=float(scores_np[idx]),
                mask=mask,
                bbox=bbox,
            )
            instances.append(inst)
            self._next_id += 1

        return instances

    def _mask_nms(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
    ) -> List[int]:
        """Per-mask NMS based on mask IoU."""
        order = scores.argsort()[::-1]
        keep = []

        suppressed = set()
        for i in order:
            if i in suppressed:
                continue
            keep.append(int(i))
            for j in order:
                if j in suppressed or j == i:
                    continue
                iou = _mask_iou(masks[i], masks[j])
                if iou > self.nms_iou_threshold:
                    suppressed.add(j)

        return keep

    def evaluate(
        self,
        predictions: List[List[InstancePrediction]],
        gt_masks_list: List[List[np.ndarray]],
        gt_classes_list: List[List[int]],
    ) -> Dict[str, float]:
        """Evaluate instance segmentation over a dataset.

        Args:
            predictions: Per-image instance predictions.
            gt_masks_list: Per-image GT masks.
            gt_classes_list: Per-image GT class labels.

        Returns:
            Dict with AP metrics.
        """
        all_pred_masks, all_pred_scores, all_pred_classes = [], [], []
        all_gt_masks, all_gt_classes = [], []

        for preds, gt_m, gt_c in zip(predictions, gt_masks_list, gt_classes_list):
            for p in preds:
                all_pred_masks.append(p.mask)
                all_pred_scores.append(p.score)
                all_pred_classes.append(p.class_id)
            all_gt_masks.extend(gt_m)
            all_gt_classes.extend(gt_c)

        return compute_mask_ap(
            all_pred_masks, all_pred_scores, all_pred_classes,
            all_gt_masks, all_gt_classes,
        )
