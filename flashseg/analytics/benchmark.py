"""Model benchmarking for segmentation."""

import time
import logging

import torch

from flashseg.cfg.config import get_config
from flashseg.models.build import build_model

logger = logging.getLogger(__name__)


class Benchmark:
    """Benchmark FlashSeg model speed and efficiency."""

    def __init__(self, model_path: str = None, model_size: str = "m", input_size: int = 512, num_classes: int = 21, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        config = get_config(model_size=model_size, input_size=input_size, num_classes=num_classes)
        self.model = build_model(config).to(self.device)

        if model_path:
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        self.model.eval()
        self.input_size = input_size

    def run(self, warmup: int = 10, iterations: int = 100) -> dict:
        """Run benchmark and return timing results."""
        dummy = torch.randn(1, 3, self.input_size, self.input_size).to(self.device)

        # Warmup
        with torch.no_grad():
            for _ in range(warmup):
                self.model(dummy)

        if self.device.type == "cuda":
            torch.cuda.synchronize()

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(iterations):
                start = time.perf_counter()
                self.model(dummy)
                if self.device.type == "cuda":
                    torch.cuda.synchronize()
                times.append(time.perf_counter() - start)

        avg_ms = sum(times) / len(times) * 1000
        fps = 1000.0 / avg_ms
        params = sum(p.numel() for p in self.model.parameters())

        results = {
            "latency_ms": round(avg_ms, 2),
            "fps": round(fps, 1),
            "params": params,
            "params_m": round(params / 1e6, 2),
            "size_mb": round(params * 4 / 1024 / 1024, 2),
            "device": str(self.device),
            "input_size": self.input_size,
        }

        logger.info(f"Benchmark: {fps:.1f} FPS, {avg_ms:.2f}ms, {params / 1e6:.2f}M params")
        return results
