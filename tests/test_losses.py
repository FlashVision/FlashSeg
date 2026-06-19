"""Test loss functions."""

import torch
import pytest
from flashseg.losses import CrossEntropyLoss, DiceLoss, FocalLoss, CombinedLoss


def test_cross_entropy():
    loss_fn = CrossEntropyLoss(ignore_index=255)
    pred = torch.randn(2, 10, 64, 64)
    target = torch.randint(0, 10, (2, 64, 64))
    loss = loss_fn(pred, target)
    assert loss.item() > 0


def test_dice_loss():
    loss_fn = DiceLoss()
    pred = torch.randn(2, 10, 64, 64)
    target = torch.randint(0, 10, (2, 64, 64))
    loss = loss_fn(pred, target)
    assert 0 <= loss.item() <= 1.0


def test_focal_loss():
    loss_fn = FocalLoss()
    pred = torch.randn(2, 10, 64, 64)
    target = torch.randint(0, 10, (2, 64, 64))
    loss = loss_fn(pred, target)
    assert loss.item() > 0


def test_combined_loss():
    loss_fn = CombinedLoss()
    pred = torch.randn(2, 10, 64, 64)
    target = torch.randint(0, 10, (2, 64, 64))
    loss = loss_fn(pred, target)
    assert loss.item() > 0
