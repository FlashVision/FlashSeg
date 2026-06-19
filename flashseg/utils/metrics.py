"""Segmentation metrics."""

import torch
import numpy as np


def compute_miou(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255) -> float:
    """Compute mean Intersection over Union."""
    ious = []
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls

        valid = target != ignore_index
        pred_mask = pred_mask & valid
        target_mask = target_mask & valid

        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()

        if union == 0:
            continue
        ious.append(intersection / union)

    return np.mean(ious) if ious else 0.0


def compute_pixel_accuracy(pred: torch.Tensor, target: torch.Tensor, ignore_index: int = 255) -> float:
    """Compute pixel accuracy."""
    valid = target != ignore_index
    correct = ((pred == target) & valid).sum().item()
    total = valid.sum().item()
    return correct / total if total > 0 else 0.0


def compute_class_iou(pred: torch.Tensor, target: torch.Tensor, num_classes: int, ignore_index: int = 255) -> dict:
    """Compute per-class IoU."""
    class_ious = {}
    for cls in range(num_classes):
        pred_mask = pred == cls
        target_mask = target == cls

        valid = target != ignore_index
        pred_mask = pred_mask & valid
        target_mask = target_mask & valid

        intersection = (pred_mask & target_mask).sum().item()
        union = (pred_mask | target_mask).sum().item()

        if union > 0:
            class_ious[cls] = intersection / union

    return class_ious
