"""Segmentation head with multi-scale feature fusion."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class SegHead(nn.Module):
    """Lightweight segmentation head that fuses multi-scale FPN features."""

    def __init__(self, in_channels: int = 128, num_classes: int = 21, input_size: int = 512):
        super().__init__()
        self.input_size = input_size

        self.fuse_conv = nn.Sequential(
            nn.Conv2d(in_channels * 3, in_channels, 1, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Dropout2d(0.1),
            nn.Conv2d(in_channels // 2, num_classes, 1),
        )

    def forward(self, features: List[torch.Tensor]) -> torch.Tensor:
        target_size = features[0].shape[2:]

        upsampled = []
        for feat in features:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode="bilinear", align_corners=False)
            upsampled.append(feat)

        fused = torch.cat(upsampled, dim=1)
        fused = self.fuse_conv(fused)
        out = self.classifier(fused)

        out = F.interpolate(out, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        return out
