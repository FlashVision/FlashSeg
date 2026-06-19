"""FlashSeg ONNX exporter."""

import logging
from pathlib import Path

import torch

from flashseg.cfg.config import get_config
from flashseg.models.build import build_model

logger = logging.getLogger(__name__)


class Exporter:
    """Export FlashSeg models to ONNX format."""

    def __init__(
        self,
        model_path: str,
        model_size: str = "m",
        num_classes: int = 21,
        input_size: int = 512,
    ):
        self.model_path = model_path
        self.model_size = model_size
        self.num_classes = num_classes
        self.input_size = input_size

    def export(self, output: str = "model.onnx", simplify: bool = True, opset: int = 11) -> str:
        """Export model to ONNX."""
        return self.export_onnx(output, simplify, opset)

    def export_onnx(self, output: str = "model.onnx", simplify: bool = True, opset: int = 11) -> str:
        """Export to ONNX format."""
        config = get_config(model_size=self.model_size, input_size=self.input_size, num_classes=self.num_classes)
        model = build_model(config)
        model.load_state_dict(torch.load(self.model_path, map_location="cpu"))
        model.eval()

        dummy_input = torch.randn(1, 3, self.input_size, self.input_size)

        torch.onnx.export(
            model,
            dummy_input,
            output,
            opset_version=opset,
            input_names=["images"],
            output_names=["output"],
            dynamic_axes={"images": {0: "batch"}, "output": {0: "batch"}},
        )
        logger.info(f"Exported ONNX model to {output}")

        if simplify:
            try:
                import onnx
                from onnxsim import simplify as onnx_simplify

                model_onnx = onnx.load(output)
                model_simple, check = onnx_simplify(model_onnx)
                if check:
                    onnx.save(model_simple, output)
                    logger.info("ONNX model simplified")
            except ImportError:
                logger.warning("onnxsim not installed, skipping simplification")

        file_size = Path(output).stat().st_size / 1024 / 1024
        logger.info(f"ONNX model size: {file_size:.2f} MB")
        return output
