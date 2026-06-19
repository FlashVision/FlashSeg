# Models

## Available Variants

| Model | Params | FP16 Size | Input | Use Case |
|-------|--------|-----------|-------|----------|
| FlashSeg-n | 0.3M | ~0.7 MB | 256 | Ultra-edge |
| FlashSeg-s | 0.8M | ~1.6 MB | 256 | Edge devices |
| FlashSeg-m | 1.5M | ~3.0 MB | 512 | Balanced |
| FlashSeg-l | 3.2M | ~6.4 MB | 512 | Best accuracy |

## Architecture

- **Backbone**: ShuffleNetV2 (configurable width multiplier)
- **Neck**: Feature Pyramid Network (FPN)
- **Head**: Multi-scale fusion + classifier

## Supported Datasets

- Pascal VOC (21 classes)
- Cityscapes (19 classes)
- ADE20K (150 classes)
- Custom datasets (any number of classes)
