"""Layer-by-layer profiling."""

import logging

import torch

from flashseg.cfg.config import get_config
from flashseg.models.build import build_model

logger = logging.getLogger(__name__)


class Profiler:
    """Profile FlashSeg model layer-by-layer."""

    def __init__(self, model_path: str = None, model_size: str = "m", input_size: int = 512, num_classes: int = 21):
        config = get_config(model_size=model_size, input_size=input_size, num_classes=num_classes)
        self.model = build_model(config)

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location="cpu"))

        self.model.eval()
        self.input_size = input_size

    def run(self) -> dict:
        """Profile model and print per-module statistics."""
        torch.randn(1, 3, self.input_size, self.input_size)

        results = {}
        for name, module in self.model.named_children():
            params = sum(p.numel() for p in module.parameters())
            results[name] = {
                "params": params,
                "params_m": round(params / 1e6, 3),
            }
            print(f"  {name:20s} | {params:>10,} params | {params / 1e6:.3f}M")

        total = sum(p.numel() for p in self.model.parameters())
        print(f"  {'TOTAL':20s} | {total:>10,} params | {total / 1e6:.3f}M")
        results["total"] = {"params": total, "params_m": round(total / 1e6, 3)}
        return results
