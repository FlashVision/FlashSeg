"""Scene parsing solution — segment and label all objects in a scene."""

from typing import Dict, List

import numpy as np

from flashseg.engine.predictor import Predictor


class SceneParser:
    """Parse a scene into semantic regions with statistics."""

    def __init__(self, predictor: Predictor, class_names: List[str] = None):
        self.predictor = predictor
        self.class_names = class_names or [f"class_{i}" for i in range(predictor.num_classes)]

    def parse(self, image: np.ndarray) -> Dict[str, float]:
        """Segment image and return per-class area percentages."""
        mask = self.predictor.predict(image)
        total_pixels = mask.size
        results = {}

        for cls_id in range(self.predictor.num_classes):
            count = (mask == cls_id).sum()
            if count > 0:
                name = self.class_names[cls_id] if cls_id < len(self.class_names) else f"class_{cls_id}"
                results[name] = float(count) / total_pixels * 100.0

        return results

    def get_mask(self, image: np.ndarray, target_class: int) -> np.ndarray:
        """Get binary mask for a specific class."""
        mask = self.predictor.predict(image)
        return (mask == target_class).astype(np.uint8) * 255
