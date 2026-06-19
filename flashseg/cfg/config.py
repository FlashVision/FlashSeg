"""Configuration management for FlashSeg."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


MODEL_SIZE_MAP = {
    "n": {"width_mult": 0.25, "depth_mult": 0.33},
    "s": {"width_mult": 0.50, "depth_mult": 0.33},
    "m": {"width_mult": 0.75, "depth_mult": 0.67},
    "l": {"width_mult": 1.00, "depth_mult": 1.00},
}


@dataclass
class Config:
    """FlashSeg configuration."""

    # Model
    model_size: str = "m"
    num_classes: int = 21
    input_size: int = 512
    width_mult: float = 0.75
    depth_mult: float = 0.67
    backbone: str = "shufflenetv2"
    neck: str = "fpn"
    head: str = "seg_head"

    # Training
    epochs: int = 100
    batch_size: int = 16
    lr: float = 0.01
    momentum: float = 0.9
    weight_decay: float = 5e-4
    warmup_epochs: int = 5
    scheduler: str = "cosine"
    amp: bool = False
    multi_gpu: bool = False

    # Data
    train_images: str = ""
    train_masks: str = ""
    val_images: str = ""
    val_masks: str = ""
    num_workers: int = 4
    augment: bool = True

    # LoRA
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_variant: str = "standard"

    # Knowledge Distillation
    use_kd: bool = False
    teacher_checkpoint: str = ""
    teacher_size: str = "l"
    kd_temperature: float = 4.0
    kd_alpha: float = 0.5

    # Pretrained
    pretrained: bool = True

    # Paths
    save_dir: str = "workspace"
    device: str = "cuda"

    # Extra
    extra: Dict[str, Any] = field(default_factory=dict)


def get_config(
    model_size: str = "m",
    input_size: int = 512,
    num_classes: int = 21,
    **overrides,
) -> Config:
    """Create a config with sensible defaults for the given model size."""
    size_params = MODEL_SIZE_MAP.get(model_size, MODEL_SIZE_MAP["m"])

    config = Config(
        model_size=model_size,
        input_size=input_size,
        num_classes=num_classes,
        width_mult=size_params["width_mult"],
        depth_mult=size_params["depth_mult"],
    )

    for key, value in overrides.items():
        if hasattr(config, key):
            setattr(config, key, value)

    return config


def load_yaml_config(path: str) -> Config:
    """Load configuration from a YAML file."""
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    model_size = data.pop("model_size", "m")
    input_size = data.pop("input_size", 512)
    num_classes = data.pop("num_classes", 21)

    return get_config(
        model_size=model_size,
        input_size=input_size,
        num_classes=num_classes,
        **data,
    )
