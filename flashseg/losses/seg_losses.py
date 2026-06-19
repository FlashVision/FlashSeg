"""Segmentation loss functions."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossEntropyLoss(nn.Module):
    """Standard cross-entropy loss with optional class weights and ignore index."""

    def __init__(self, weight=None, ignore_index: int = 255):
        super().__init__()
        self.loss_fn = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.loss_fn(pred, target)


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""

    def __init__(self, smooth: float = 1.0, ignore_index: int = 255):
        super().__init__()
        self.smooth = smooth
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        num_classes = pred.shape[1]
        pred_soft = F.softmax(pred, dim=1)

        mask = target != self.ignore_index
        target_masked = target.clone()
        target_masked[~mask] = 0

        target_onehot = F.one_hot(target_masked, num_classes).permute(0, 3, 1, 2).float()

        mask_expanded = mask.unsqueeze(1).expand_as(target_onehot)
        pred_masked = pred_soft * mask_expanded
        target_masked_oh = target_onehot * mask_expanded

        intersection = (pred_masked * target_masked_oh).sum(dim=(2, 3))
        union = pred_masked.sum(dim=(2, 3)) + target_masked_oh.sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (union + self.smooth)
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    """Focal loss for handling class imbalance."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, ignore_index: int = 255):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(pred, target, reduction="none", ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class CombinedLoss(nn.Module):
    """Combined CE + Dice loss."""

    def __init__(self, ce_weight: float = 0.5, dice_weight: float = 0.5, ignore_index: int = 255):
        super().__init__()
        self.ce = CrossEntropyLoss(ignore_index=ignore_index)
        self.dice = DiceLoss(ignore_index=ignore_index)
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(pred, target) + self.dice_weight * self.dice(pred, target)
