"""FlashSeg Validator."""

import logging

import torch
from torch.utils.data import DataLoader

from flashseg.cfg.config import get_config
from flashseg.data.dataset import SegmentationDataset
from flashseg.data.transforms import get_val_transforms
from flashseg.models.build import build_model
from flashseg.utils.metrics import compute_miou, compute_pixel_accuracy

logger = logging.getLogger(__name__)


class Validator:
    """FlashSeg validation engine."""

    def __init__(
        self,
        model_path: str,
        val_images: str,
        val_masks: str,
        model_size: str = "m",
        num_classes: int = 21,
        input_size: int = 512,
        batch_size: int = 16,
        device: str = "cuda",
    ):
        self.device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
        self.num_classes = num_classes

        config = get_config(model_size=model_size, input_size=input_size, num_classes=num_classes)
        self.model = build_model(config).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

        dataset = SegmentationDataset(
            val_images, val_masks, input_size, num_classes,
            transform=get_val_transforms(input_size),
        )
        self.dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    @torch.no_grad()
    def validate(self) -> dict:
        """Run validation and return metrics."""
        all_preds, all_targets = [], []

        for images, masks in self.dataloader:
            images = images.to(self.device)
            pred = self.model(images)
            pred_cls = pred.argmax(dim=1).cpu()
            all_preds.append(pred_cls)
            all_targets.append(masks)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)

        miou = compute_miou(all_preds, all_targets, self.num_classes)
        pixel_acc = compute_pixel_accuracy(all_preds, all_targets)

        results = {"mIoU": miou, "pixel_accuracy": pixel_acc}
        logger.info(f"Validation: mIoU={miou:.4f}, PixelAcc={pixel_acc:.4f}")
        return results
