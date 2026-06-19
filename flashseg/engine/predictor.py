"""FlashSeg Predictor for inference."""

import logging
from pathlib import Path
from typing import Optional, Union

import cv2
import numpy as np
import torch

from flashseg.cfg.config import get_config
from flashseg.models.build import build_model

logger = logging.getLogger(__name__)


class Predictor:
    """FlashSeg inference engine."""

    def __init__(
        self,
        model_path: str,
        model_size: str = "m",
        num_classes: int = 21,
        input_size: int = 512,
        device: str = "cuda",
        conf_threshold: float = 0.5,
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.input_size = input_size
        self.num_classes = num_classes
        self.conf_threshold = conf_threshold

        config = get_config(model_size=model_size, input_size=input_size, num_classes=num_classes)
        self.model = build_model(config).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        logger.info(f"Loaded model from {model_path} on {self.device}")

    @torch.no_grad()
    def predict(self, source: Union[str, np.ndarray]) -> np.ndarray:
        """Predict segmentation mask.

        Args:
            source: Image path or numpy array (H, W, 3) in BGR.

        Returns:
            Segmentation mask (H, W) with class indices.
        """
        if isinstance(source, str):
            image = cv2.imread(source)
            if image is None:
                raise FileNotFoundError(f"Cannot read image: {source}")
        else:
            image = source

        orig_h, orig_w = image.shape[:2]
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (self.input_size, self.input_size))

        tensor = resized.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        tensor = (tensor - mean) / std
        tensor = np.transpose(tensor, (2, 0, 1))
        tensor = torch.from_numpy(tensor).unsqueeze(0).to(self.device)

        output = self.model(tensor)
        mask = output.argmax(dim=1).squeeze(0).cpu().numpy()

        mask = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST)
        return mask

    def predict_directory(self, directory: str, save_dir: Optional[str] = None) -> dict:
        """Predict on all images in a directory."""
        dir_path = Path(directory)
        results = {}
        extensions = (".jpg", ".jpeg", ".png", ".bmp")

        save_path = Path(save_dir) if save_dir else None
        if save_path:
            save_path.mkdir(parents=True, exist_ok=True)

        for img_file in sorted(dir_path.iterdir()):
            if img_file.suffix.lower() in extensions:
                mask = self.predict(str(img_file))
                results[img_file.name] = mask

                if save_path:
                    cv2.imwrite(str(save_path / f"{img_file.stem}_mask.png"), mask)

        logger.info(f"Predicted {len(results)} images from {directory}")
        return results

    def visualize(self, image: np.ndarray, mask: np.ndarray, alpha: float = 0.5) -> np.ndarray:
        """Overlay segmentation mask on image with color coding."""
        colormap = self._generate_colormap(self.num_classes)
        colored_mask = colormap[mask]
        overlay = cv2.addWeighted(image, 1 - alpha, colored_mask.astype(np.uint8), alpha, 0)
        return overlay

    @staticmethod
    def _generate_colormap(num_classes: int) -> np.ndarray:
        """Generate a colormap for visualization."""
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
