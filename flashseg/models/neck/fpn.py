"""Feature Pyramid Network for segmentation."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    """Lightweight Feature Pyramid Network."""

    def __init__(self, in_channels: List[int], out_channels: int = 128):
        super().__init__()

        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()

        for ch in in_channels:
            self.lateral_convs.append(
                nn.Sequential(
                    nn.Conv2d(ch, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )
            self.fpn_convs.append(
                nn.Sequential(
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, groups=out_channels, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.Conv2d(out_channels, out_channels, 1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                )
            )

    def forward(self, features: List[torch.Tensor]) -> List[torch.Tensor]:
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, features)]

        for i in range(len(laterals) - 1, 0, -1):
            upsampled = F.interpolate(laterals[i], size=laterals[i - 1].shape[2:], mode="bilinear", align_corners=False)
            laterals[i - 1] = laterals[i - 1] + upsampled

        outputs = [conv(lat) for conv, lat in zip(self.fpn_convs, laterals)]
        return outputs
