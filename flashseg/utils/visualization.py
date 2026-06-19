"""Segmentation visualization utilities."""

import numpy as np
import cv2


def generate_colormap(num_classes: int) -> np.ndarray:
    """Generate a Pascal VOC-style colormap."""
    colormap = np.zeros((num_classes, 3), dtype=np.uint8)
    for i in range(num_classes):
        r, g, b = 0, 0, 0
        idx = i
        for j in range(8):
            r |= ((idx >> 0) & 1) << (7 - j)
            g |= ((idx >> 1) & 1) << (7 - j)
            b |= ((idx >> 2) & 1) << (7 - j)
            idx >>= 3
        colormap[i] = [r, g, b]
    return colormap


def colorize_mask(mask: np.ndarray, num_classes: int = 21) -> np.ndarray:
    """Convert a class-index mask to a colored RGB image."""
    colormap = generate_colormap(num_classes)
    colored = colormap[mask]
    return colored


def overlay_mask(image: np.ndarray, mask: np.ndarray, num_classes: int = 21, alpha: float = 0.5) -> np.ndarray:
    """Overlay colored segmentation mask on the original image."""
    colored = colorize_mask(mask, num_classes)
    overlay = cv2.addWeighted(image, 1 - alpha, colored.astype(np.uint8), alpha, 0)
    return overlay
