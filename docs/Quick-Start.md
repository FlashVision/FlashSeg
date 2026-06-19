# Quick Start

## Train

```python
from flashseg import Trainer

trainer = Trainer(
    model_size="m",
    train_images="data/images",
    train_masks="data/masks",
    val_images="data/val_images",
    val_masks="data/val_masks",
    num_classes=21,
    epochs=100,
    device="cuda",
)
trainer.train()
```

## Predict

```python
from flashseg import Predictor

predictor = Predictor(model_path="workspace/best.pth", num_classes=21, device="cuda")
mask = predictor.predict("photo.jpg")
overlay = predictor.visualize(image, mask)
```

## Export

```python
from flashseg import Exporter

exporter = Exporter(model_path="workspace/best.pth", num_classes=21)
exporter.export(output="model.onnx", simplify=True)
```

## CLI

```bash
flashseg train --model-size m --num-classes 21 --train-images data/img --train-masks data/mask --val-images data/val_img --val-masks data/val_mask
flashseg predict --model best.pth --source images/
flashseg export --model best.pth --output model.onnx --simplify
```
