"""Background removal using segmentation."""

import numpy as np

from flashseg.engine.predictor import Predictor


class BackgroundRemover:
    """Remove or replace background using semantic segmentation."""

    def __init__(self, predictor: Predictor, foreground_classes: list = None):
        self.predictor = predictor
        self.foreground_classes = foreground_classes or [15]  # person class by default

    def remove(self, image: np.ndarray, background_color: tuple = (0, 0, 0)) -> np.ndarray:
        """Remove background, keeping only foreground classes."""
        mask = self.predictor.predict(image)

        fg_mask = np.zeros_like(mask, dtype=bool)
        for cls_id in self.foreground_classes:
            fg_mask |= (mask == cls_id)

        result = np.full_like(image, background_color, dtype=np.uint8)
        result[fg_mask] = image[fg_mask]
        return result

    def replace(self, image: np.ndarray, background: np.ndarray) -> np.ndarray:
        """Replace background with another image."""
        mask = self.predictor.predict(image)

        fg_mask = np.zeros_like(mask, dtype=bool)
        for cls_id in self.foreground_classes:
            fg_mask |= (mask == cls_id)

        import cv2
        bg_resized = cv2.resize(background, (image.shape[1], image.shape[0]))
        result = bg_resized.copy()
        result[fg_mask] = image[fg_mask]
        return result

    def get_alpha_matte(self, image: np.ndarray) -> np.ndarray:
        """Get alpha matte (0-255) for foreground."""
        mask = self.predictor.predict(image)
        alpha = np.zeros(mask.shape, dtype=np.uint8)
        for cls_id in self.foreground_classes:
            alpha[mask == cls_id] = 255
        return alpha
