"""ShuffleNetV2 backbone for lightweight segmentation."""

from typing import List

import torch
import torch.nn as nn


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    """Channel shuffle operation."""
    batch, channels, height, width = x.size()
    channels_per_group = channels // groups
    x = x.view(batch, groups, channels_per_group, height, width)
    x = x.transpose(1, 2).contiguous()
    x = x.view(batch, channels, height, width)
    return x


class ShuffleUnit(nn.Module):
    """ShuffleNetV2 basic unit."""

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.stride = stride
        mid_channels = out_channels // 2

        if stride == 2:
            self.branch1 = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
                nn.BatchNorm2d(in_channels),
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )
            self.branch2 = nn.Sequential(
                nn.Conv2d(in_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )
        else:
            self.branch2 = nn.Sequential(
                nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(mid_channels, mid_channels, 3, 1, 1, groups=mid_channels, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.Conv2d(mid_channels, mid_channels, 1, bias=False),
                nn.BatchNorm2d(mid_channels),
                nn.ReLU(inplace=True),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.stride == 2:
            out = torch.cat([self.branch1(x), self.branch2(x)], dim=1)
        else:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat([x1, self.branch2(x2)], dim=1)
        return channel_shuffle(out, 2)


class ShuffleNetV2(nn.Module):
    """ShuffleNetV2 backbone with multi-scale feature output."""

    STAGE_CHANNELS = {
        0.25: [24, 24, 48, 96, 512],
        0.50: [24, 48, 96, 192, 1024],
        0.75: [24, 72, 144, 288, 1024],
        1.00: [24, 116, 232, 464, 1024],
        1.50: [24, 176, 352, 704, 1024],
        2.00: [24, 244, 488, 976, 2048],
    }

    STAGE_REPEATS = [4, 8, 4]

    def __init__(self, width_mult: float = 0.75):
        super().__init__()
        channels = self.STAGE_CHANNELS.get(width_mult, self.STAGE_CHANNELS[0.75])

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, channels[0], 3, 2, 1, bias=False),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(3, 2, 1)

        self.stage2 = self._make_stage(channels[0], channels[1], self.STAGE_REPEATS[0])
        self.stage3 = self._make_stage(channels[1], channels[2], self.STAGE_REPEATS[1])
        self.stage4 = self._make_stage(channels[2], channels[3], self.STAGE_REPEATS[2])

        self.out_channels = [channels[1], channels[2], channels[3]]

    def _make_stage(self, in_channels: int, out_channels: int, repeats: int) -> nn.Sequential:
        layers = [ShuffleUnit(in_channels, out_channels, stride=2)]
        for _ in range(repeats - 1):
            layers.append(ShuffleUnit(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        x = self.conv1(x)
        x = self.maxpool(x)

        c3 = self.stage2(x)
        c4 = self.stage3(c3)
        c5 = self.stage4(c4)

        return [c3, c4, c5]
