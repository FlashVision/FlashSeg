"""Area calculation from segmentation masks."""

from typing import Dict

import numpy as np

from flashseg.engine.predictor import Predictor


class AreaCalculator:
    """Calculate real-world areas from segmentation masks."""

    def __init__(self, predictor: Predictor, pixels_per_meter: float = 1.0):
        self.predictor = predictor
        self.pixels_per_meter = pixels_per_meter

    def calculate(self, image: np.ndarray) -> Dict[int, float]:
        """Calculate area in square meters for each class present."""
        mask = self.predictor.predict(image)
        pixel_area = 1.0 / (self.pixels_per_meter ** 2)

        areas = {}
        for cls_id in np.unique(mask):
            pixel_count = (mask == cls_id).sum()
            areas[int(cls_id)] = float(pixel_count) * pixel_area

        return areas

    def calculate_class(self, image: np.ndarray, class_id: int) -> float:
        """Calculate area for a specific class."""
        mask = self.predictor.predict(image)
        pixel_count = (mask == class_id).sum()
        pixel_area = 1.0 / (self.pixels_per_meter ** 2)
        return float(pixel_count) * pixel_area
