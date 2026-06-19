"""Model builder for FlashSeg."""

import torch
import torch.nn as nn

from flashseg.cfg.config import Config
from flashseg.models.backbone.shufflenetv2 import ShuffleNetV2
from flashseg.models.head.seg_head import SegHead
from flashseg.models.neck.fpn import FPN


def build_model(config: Config) -> nn.Module:
    """Build a FlashSeg model from config."""
    model = FlashSeg(config)
    return model


class FlashSeg(nn.Module):
    """FlashSeg: Lightweight segmentation network."""

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

        self.backbone = ShuffleNetV2(
            width_mult=config.width_mult,
        )

        backbone_channels = self.backbone.out_channels
        self.neck = FPN(
            in_channels=backbone_channels,
            out_channels=128,
        )

        self.head = SegHead(
            in_channels=128,
            num_classes=config.num_classes,
            input_size=config.input_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)
        features = self.neck(features)
        out = self.head(features)
        return out
