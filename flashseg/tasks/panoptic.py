"""Panoptic segmentation task.

Combines semantic and instance segmentation into a unified panoptic output
where every pixel is assigned both a semantic label and an instance ID.
Includes the Panoptic Quality (PQ) metric.
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


@dataclass
class PanopticSegment:
    """A single panoptic segment (stuff region or thing instance)."""

    segment_id: int
    class_id: int
    is_thing: bool
    area: int = 0
    bbox: Tuple[int, int, int, int] = (0, 0, 0, 0)
    score: float = 1.0


@dataclass
class PanopticResult:
    """Panoptic segmentation output for a single image."""

    panoptic_map: np.ndarray  # (H, W) — each pixel is segment_id
    segments: List[PanopticSegment]
    semantic_map: np.ndarray  # (H, W) — class labels

    @property
    def num_segments(self) -> int:
        return len(self.segments)

    def get_thing_segments(self) -> List[PanopticSegment]:
        return [s for s in self.segments if s.is_thing]

    def get_stuff_segments(self) -> List[PanopticSegment]:
        return [s for s in self.segments if not s.is_thing]


def compute_pq(
    pred_panoptic: np.ndarray,
    pred_segments: List[PanopticSegment],
    gt_panoptic: np.ndarray,
    gt_segments: List[PanopticSegment],
    thing_classes: Optional[Set[int]] = None,
    stuff_classes: Optional[Set[int]] = None,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute Panoptic Quality (PQ) and its decomposition.

    PQ = SQ * RQ, where:
      - SQ (Segmentation Quality) = avg IoU of matched segments
      - RQ (Recognition Quality) = F1 score of matching

    Args:
        pred_panoptic: (H, W) predicted panoptic map (segment IDs).
        pred_segments: List of predicted panoptic segments.
        gt_panoptic: (H, W) ground-truth panoptic map.
        gt_segments: List of GT panoptic segments.
        thing_classes: Set of "thing" class IDs.
        stuff_classes: Set of "stuff" class IDs.
        iou_threshold: IoU threshold for matching (default 0.5).

    Returns:
        Dict with ``'PQ'``, ``'SQ'``, ``'RQ'``, ``'PQ_th'``, ``'PQ_st'``, etc.
    """
    if thing_classes is None:
        thing_classes = {s.class_id for s in gt_segments if s.is_thing}
    if stuff_classes is None:
        stuff_classes = {s.class_id for s in gt_segments if not s.is_thing}

    all_classes = thing_classes | stuff_classes

    per_class_pq = {}

    for cls in all_classes:
        pred_segs_cls = [s for s in pred_segments if s.class_id == cls]
        gt_segs_cls = [s for s in gt_segments if s.class_id == cls]

        pred_ids = {s.segment_id for s in pred_segs_cls}
        gt_ids = {s.segment_id for s in gt_segs_cls}

        # Match predictions to GT
        matched_iou_sum = 0.0
        tp = 0
        matched_pred = set()
        matched_gt = set()

        for gt_seg in gt_segs_cls:
            gt_mask = gt_panoptic == gt_seg.segment_id
            best_iou = 0.0
            best_pred_id = -1

            for pred_seg in pred_segs_cls:
                if pred_seg.segment_id in matched_pred:
                    continue
                pred_mask = pred_panoptic == pred_seg.segment_id
                intersection = (gt_mask & pred_mask).sum()
                union = (gt_mask | pred_mask).sum()
                iou = float(intersection) / max(float(union), 1e-6)

                if iou > best_iou:
                    best_iou = iou
                    best_pred_id = pred_seg.segment_id

            if best_iou >= iou_threshold:
                tp += 1
                matched_iou_sum += best_iou
                matched_pred.add(best_pred_id)
                matched_gt.add(gt_seg.segment_id)

        fp = len(pred_ids - matched_pred)
        fn = len(gt_ids - matched_gt)

        sq = matched_iou_sum / max(tp, 1)
        rq = tp / max(tp + 0.5 * fp + 0.5 * fn, 1e-6)
        pq = sq * rq

        per_class_pq[cls] = {"PQ": pq, "SQ": sq, "RQ": rq, "TP": tp, "FP": fp, "FN": fn}

    thing_pqs = [v["PQ"] for c, v in per_class_pq.items() if c in thing_classes]
    stuff_pqs = [v["PQ"] for c, v in per_class_pq.items() if c in stuff_classes]
    all_pqs = [v["PQ"] for v in per_class_pq.values()]
    all_sqs = [v["SQ"] for v in per_class_pq.values()]
    all_rqs = [v["RQ"] for v in per_class_pq.values()]

    return {
        "PQ": float(np.mean(all_pqs)) if all_pqs else 0.0,
        "SQ": float(np.mean(all_sqs)) if all_sqs else 0.0,
        "RQ": float(np.mean(all_rqs)) if all_rqs else 0.0,
        "PQ_th": float(np.mean(thing_pqs)) if thing_pqs else 0.0,
        "PQ_st": float(np.mean(stuff_pqs)) if stuff_pqs else 0.0,
        "num_classes": len(per_class_pq),
        "per_class": per_class_pq,
    }


class PanopticSegmentor:
    """Panoptic segmentation: combines semantic (stuff) and instance (things).

    Merges class-level semantic predictions with instance-level mask predictions
    into a unified panoptic segmentation map.

    Args:
        thing_classes: Set of class IDs treated as "things" (countable objects).
        stuff_classes: Set of class IDs treated as "stuff" (amorphous regions).
        score_threshold: Minimum score for thing instances.
        overlap_threshold: Maximum overlap between instances before merging.
        stuff_area_threshold: Minimum pixel area for stuff segments.
    """

    def __init__(
        self,
        thing_classes: Set[int],
        stuff_classes: Set[int],
        score_threshold: float = 0.5,
        overlap_threshold: float = 0.5,
        stuff_area_threshold: int = 4096,
    ):
        self.thing_classes = thing_classes
        self.stuff_classes = stuff_classes
        self.score_threshold = score_threshold
        self.overlap_threshold = overlap_threshold
        self.stuff_area_threshold = stuff_area_threshold

    def merge(
        self,
        pred_logits: torch.Tensor,
        pred_masks: torch.Tensor,
        image_size: Tuple[int, int],
    ) -> PanopticResult:
        """Merge query predictions into a panoptic segmentation map.

        Args:
            pred_logits: (Q, num_classes+1) class logits.
            pred_masks: (Q, H', W') mask logits.
            image_size: (H, W) target size.

        Returns:
            ``PanopticResult`` with panoptic map and segments.
        """
        H, W = image_size

        masks = F.interpolate(
            pred_masks.unsqueeze(0).float(), size=(H, W),
            mode="bilinear", align_corners=False,
        ).squeeze(0)
        mask_probs = masks.sigmoid()  # (Q, H, W)

        cls_probs = pred_logits.softmax(-1)  # (Q, C+1)
        scores, labels = cls_probs[:, :-1].max(-1)

        panoptic_map = np.zeros((H, W), dtype=np.int32)
        semantic_map = np.full((H, W), 255, dtype=np.int32)
        segments: List[PanopticSegment] = []
        current_id = 1

        # Sort by score (descending)
        order = scores.argsort(descending=True)

        used_pixels = np.zeros((H, W), dtype=bool)

        for idx in order:
            cls_id = int(labels[idx].item())
            score = float(scores[idx].item())
            is_thing = cls_id in self.thing_classes

            if is_thing and score < self.score_threshold:
                continue

            binary = (mask_probs[idx].cpu().numpy() > 0.5)
            overlap = (binary & used_pixels).sum()
            area = binary.sum()

            if area == 0:
                continue

            if overlap / max(area, 1) > self.overlap_threshold:
                continue

            if not is_thing and area < self.stuff_area_threshold:
                continue

            valid = binary & ~used_pixels
            panoptic_map[valid] = current_id
            semantic_map[valid] = cls_id
            used_pixels |= valid

            ys, xs = np.where(valid)
            bbox = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())) if len(ys) > 0 else (0, 0, 0, 0)

            segments.append(PanopticSegment(
                segment_id=current_id,
                class_id=cls_id,
                is_thing=is_thing,
                area=int(valid.sum()),
                bbox=bbox,
                score=score,
            ))
            current_id += 1

        return PanopticResult(
            panoptic_map=panoptic_map,
            segments=segments,
            semantic_map=semantic_map,
        )

    def evaluate(
        self,
        predictions: List[PanopticResult],
        gt_maps: List[np.ndarray],
        gt_segments_list: List[List[PanopticSegment]],
    ) -> Dict[str, float]:
        """Evaluate panoptic segmentation over a dataset.

        Args:
            predictions: Per-image panoptic results.
            gt_maps: Per-image GT panoptic maps.
            gt_segments_list: Per-image GT segments.

        Returns:
            Averaged PQ metrics.
        """
        all_pq = []
        for pred, gt_map, gt_segs in zip(predictions, gt_maps, gt_segments_list):
            metrics = compute_pq(
                pred.panoptic_map, pred.segments,
                gt_map, gt_segs,
                thing_classes=self.thing_classes,
                stuff_classes=self.stuff_classes,
            )
            all_pq.append(metrics)

        if not all_pq:
            return {"PQ": 0.0, "SQ": 0.0, "RQ": 0.0, "PQ_th": 0.0, "PQ_st": 0.0}

        return {
            key: float(np.mean([m[key] for m in all_pq]))
            for key in ("PQ", "SQ", "RQ", "PQ_th", "PQ_st")
        }
