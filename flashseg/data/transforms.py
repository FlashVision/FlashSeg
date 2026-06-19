"""Data transforms for segmentation."""


import numpy as np
import torch


def get_train_transforms(input_size: int = 512):
    """Get training transforms."""

    def transform(image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image.copy())

    return transform


def get_val_transforms(input_size: int = 512):
    """Get validation transforms."""

    def transform(image: np.ndarray) -> torch.Tensor:
        image = image.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        image = (image - mean) / std
        image = np.transpose(image, (2, 0, 1))
        return torch.from_numpy(image.copy())

    return transform
