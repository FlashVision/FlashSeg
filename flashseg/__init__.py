"""FlashSeg - Ultra-lightweight real-time image segmentation."""

__version__ = "1.0.0"

from flashseg.cfg.config import get_config
from flashseg.engine.trainer import Trainer
from flashseg.engine.predictor import Predictor
from flashseg.engine.exporter import Exporter
from flashseg.engine.validator import Validator
from flashseg.models.build import build_model

__all__ = [
    "__version__",
    "get_config",
    "build_model",
    "Trainer",
    "Predictor",
    "Exporter",
    "Validator",
]
