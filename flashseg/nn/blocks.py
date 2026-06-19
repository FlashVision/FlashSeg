"""Common neural network building blocks for segmentation."""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBnRelu(nn.Module):
    """Conv + BatchNorm + ReLU block."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1, groups: int = 1):
        super().__init__()
        padding = kernel // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class DepthwiseSeparableConv(nn.Module):
    """Depthwise separable convolution."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.depthwise = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel, stride, kernel // 2, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
        )
        self.pointwise = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pointwise(self.depthwise(x))


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling for multi-scale context."""

    def __init__(self, in_ch: int, out_ch: int, rates: List[int] = None):
        super().__init__()
        if rates is None:
            rates = [6, 12, 18]

        modules = [
            nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        ]

        for rate in rates:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=rate, dilation=rate, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )
            )

        modules.append(
            nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )
        )

        self.convs = nn.ModuleList(modules)
        self.project = nn.Sequential(
            nn.Conv2d(out_ch * (len(rates) + 2), out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        results = []
        for conv in self.convs[:-1]:
            results.append(conv(x))

        pool = self.convs[-1](x)
        pool = F.interpolate(pool, size=x.shape[2:], mode="bilinear", align_corners=False)
        results.append(pool)

        return self.project(torch.cat(results, dim=1))
