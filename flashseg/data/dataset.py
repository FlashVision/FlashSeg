"""Segmentation dataset classes."""

import logging
from pathlib import Path
from typing import Callable, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class SegmentationDataset(Dataset):
    """Dataset for semantic segmentation with image-mask pairs."""

    SUPPORTED_FORMATS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

    def __init__(
        self,
        images_dir: str,
        masks_dir: str,
        input_size: int = 512,
        num_classes: int = 21,
        transform: Optional[Callable] = None,
        augment: bool = False,
    ):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.input_size = input_size
        self.num_classes = num_classes
        self.transform = transform
        self.augment = augment

        self.image_files = sorted(
            [f for f in self.images_dir.iterdir() if f.suffix.lower() in self.SUPPORTED_FORMATS]
        )
        self.mask_files = sorted(
            [f for f in self.masks_dir.iterdir() if f.suffix.lower() in self.SUPPORTED_FORMATS]
        )

        assert len(self.image_files) == len(self.mask_files), (
            f"Mismatch: {len(self.image_files)} images vs {len(self.mask_files)} masks"
        )

        logger.info(f"Loaded {len(self.image_files)} image-mask pairs")

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image = cv2.imread(str(self.image_files[idx]))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.mask_files[idx]), cv2.IMREAD_GRAYSCALE)

        image = cv2.resize(image, (self.input_size, self.input_size))
        mask = cv2.resize(mask, (self.input_size, self.input_size), interpolation=cv2.INTER_NEAREST)

        if self.augment:
            image, mask = self._augment(image, mask)

        if self.transform:
            image = self.transform(image)
        else:
            image = image.astype(np.float32) / 255.0
            image = np.transpose(image, (2, 0, 1))
            image = torch.from_numpy(image)

        mask = torch.from_numpy(mask.astype(np.int64))
        return image, mask

    def _augment(self, image: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply random augmentations."""
        if np.random.random() > 0.5:
            image = np.fliplr(image).copy()
            mask = np.fliplr(mask).copy()

        if np.random.random() > 0.5:
            image = np.flipud(image).copy()
            mask = np.flipud(mask).copy()

        if np.random.random() > 0.5:
            k = np.random.randint(1, 4)
            image = np.rot90(image, k).copy()
            mask = np.rot90(mask, k).copy()

        # Color jitter (image only)
        if np.random.random() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            image = np.clip(image * factor, 0, 255).astype(np.uint8)

        return image, mask
