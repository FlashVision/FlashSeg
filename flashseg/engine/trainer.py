"""FlashSeg Trainer."""

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

from flashseg.cfg.config import get_config, load_yaml_config
from flashseg.data.dataset import SegmentationDataset
from flashseg.data.transforms import get_train_transforms, get_val_transforms
from flashseg.losses import CombinedLoss
from flashseg.models.build import build_model
from flashseg.utils.metrics import compute_miou

logger = logging.getLogger(__name__)


class Trainer:
    """FlashSeg training engine."""

    def __init__(
        self,
        model_size: str = "m",
        train_images: str = "",
        train_masks: str = "",
        val_images: str = "",
        val_masks: str = "",
        num_classes: int = 21,
        input_size: int = 512,
        epochs: int = 100,
        batch_size: int = 16,
        lr: float = 0.01,
        device: str = "cuda",
        save_dir: str = "workspace",
        use_lora: bool = False,
        pretrained: bool = True,
        amp: bool = False,
        config_path: Optional[str] = None,
        **kwargs,
    ):
        if config_path:
            self.config = load_yaml_config(config_path)
        else:
            self.config = get_config(
                model_size=model_size,
                input_size=input_size,
                num_classes=num_classes,
                train_images=train_images,
                train_masks=train_masks,
                val_images=val_images,
                val_masks=val_masks,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                device=device,
                save_dir=save_dir,
                use_lora=use_lora,
                pretrained=pretrained,
                amp=amp,
                **kwargs,
            )

        self.device = torch.device(self.config.device if torch.cuda.is_available() or self.config.device == "cpu" else "cpu")
        self.save_dir = Path(self.config.save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.scaler = GradScaler() if self.config.amp else None
        self.callbacks = []

    def add_callback(self, callback):
        """Add a training callback."""
        self.callbacks.append(callback)

    def train(self):
        """Run full training loop."""
        cfg = self.config
        logger.info(f"FlashSeg training — model {cfg.model_size}, input {cfg.input_size}, device {cfg.device}")

        # Build model
        self.model = build_model(cfg).to(self.device)
        params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Params: {params:,} ({params * 4 / 1024 / 1024:.2f} MB)")

        # Data
        train_dataset = SegmentationDataset(
            cfg.train_images, cfg.train_masks, cfg.input_size, cfg.num_classes,
            transform=get_train_transforms(cfg.input_size), augment=cfg.augment,
        )
        val_dataset = SegmentationDataset(
            cfg.val_images, cfg.val_masks, cfg.input_size, cfg.num_classes,
            transform=get_val_transforms(cfg.input_size), augment=False,
        )

        train_loader = DataLoader(train_dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
        val_loader = DataLoader(val_dataset, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)

        # Loss, optimizer, scheduler
        criterion = CombinedLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=cfg.lr, momentum=cfg.momentum, weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.epochs)

        best_miou = 0.0
        for epoch in range(cfg.epochs):
            self.model.train()
            epoch_loss = 0.0

            pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{cfg.epochs}")
            for images, masks in pbar:
                images = images.to(self.device)
                masks = masks.to(self.device)

                self.optimizer.zero_grad()

                if self.config.amp:
                    with autocast():
                        pred = self.model(images)
                        loss = criterion(pred, masks)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    pred = self.model(images)
                    loss = criterion(pred, masks)
                    loss.backward()
                    self.optimizer.step()

                epoch_loss += loss.item()
                pbar.set_postfix(loss=f"{loss.item():.4f}")

            self.scheduler.step()
            avg_loss = epoch_loss / len(train_loader)

            # Validate
            val_miou = self._validate(val_loader)
            logger.info(f"Epoch {epoch + 1}: loss={avg_loss:.4f}, mIoU={val_miou:.4f}")

            if val_miou > best_miou:
                best_miou = val_miou
                torch.save(self.model.state_dict(), self.save_dir / "best.pth")
                logger.info(f"  Best model saved (mIoU={best_miou:.4f})")

            torch.save(self.model.state_dict(), self.save_dir / "last.pth")

            for cb in self.callbacks:
                cb.on_epoch_end(epoch, {"loss": avg_loss, "val_mIoU": val_miou})

        logger.info(f"Training complete. Best mIoU: {best_miou:.4f}")
        return best_miou

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> float:
        """Run validation and return mIoU."""
        self.model.eval()
        all_preds, all_targets = [], []

        for images, masks in val_loader:
            images = images.to(self.device)
            pred = self.model(images)
            pred_cls = pred.argmax(dim=1).cpu()
            all_preds.append(pred_cls)
            all_targets.append(masks)

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        return compute_miou(all_preds, all_targets, self.config.num_classes)
