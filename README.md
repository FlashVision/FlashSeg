<p align="center">
  <img src="assets/logo.png" width="200" alt="FlashSeg Logo">
</p>

<h1 align="center">FlashSeg</h1>

<p align="center">
  <a href="https://github.com/FlashVision/FlashSeg/actions"><img src="https://img.shields.io/github/actions/workflow/status/FlashVision/FlashSeg/ci.yml?logo=github" alt="CI"></a>
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Python-3.8+-3776ab?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/ONNX-Export-005CED?logo=onnx&logoColor=white" alt="ONNX">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
</p>

<p align="center">
  <b>Ultra-lightweight real-time semantic segmentation with LoRA fine-tuning & knowledge distillation</b>
</p>

<p align="center">
  <a href="#installation">Install</a> •
  <a href="#usage">Usage</a> •
  <a href="#models">Models</a> •
  <a href="#solutions">Solutions</a> •
  <a href="#training">Training</a> •
  <a href="#examples">Examples</a>
</p>

---

## What is FlashSeg?

FlashSeg is an ultra-lightweight semantic segmentation framework built for **speed and edge deployment**. Using a ShuffleNetV2 backbone with FPN neck, it delivers real-time pixel-level predictions with models as small as 0.3M parameters.

```bash
pip install -e .
flashseg train --model-size m --num-classes 21 --train-images data/images --train-masks data/masks --val-images data/val_img --val-masks data/val_mask
flashseg predict --model best.pth --source images/
```

---

## Installation

```bash
# From source
git clone https://github.com/FlashVision/FlashSeg.git
cd FlashSeg
pip install -e ".[all]"
```

### Optional extras

```bash
pip install -e ".[export]"      # ONNX export
pip install -e ".[analytics]"   # Benchmarking, plots
pip install -e ".[solutions]"   # Background removal, lane detection
pip install -e ".[all]"         # Everything
```

### Verify

```bash
flashseg check
flashseg settings
flashseg version
```

---

## Usage

### Python API

```python
from flashseg import Trainer, Predictor, Exporter

# Train
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

# Predict
predictor = Predictor(model_path="workspace/best.pth", num_classes=21, device="cuda")
mask = predictor.predict("photo.jpg")

# Export
exporter = Exporter(model_path="workspace/best.pth", num_classes=21)
exporter.export(output="model.onnx", simplify=True)
```

### CLI

```bash
flashseg train --model-size m --epochs 100 --device cuda \
  --train-images data/images --train-masks data/masks \
  --val-images data/val_img --val-masks data/val_mask --num-classes 21

flashseg predict --model best.pth --source images/ --save-dir output/

flashseg val --model best.pth --val-images data/val --val-masks data/val_mask

flashseg export --model best.pth --output model.onnx --simplify
```

---

## Models

| Model | Params | FP16 Size | Input | mIoU (VOC) |
|-------|--------|-----------|-------|-------------|
| **FlashSeg-n** | 0.3M | ~0.7 MB | 256 | — |
| **FlashSeg-s** | 0.8M | ~1.6 MB | 256 | — |
| **FlashSeg-m** | 1.5M | ~3.0 MB | 512 | — |
| **FlashSeg-l** | 3.2M | ~6.4 MB | 512 | — |

### Config-driven Training

```bash
flashseg train --config configs/flashseg_m_512_voc.yaml
flashseg train --config configs/flashseg_s_256_cityscapes.yaml
flashseg train --config configs/flashseg_m_512_lora.yaml
```

---

## Solutions

Built-in high-level applications:

```python
from flashseg import Predictor
from flashseg.solutions import BackgroundRemover, LaneDetector, SceneParser, AreaCalculator

predictor = Predictor(model_path="best.pth", num_classes=21)

# Remove background
remover = BackgroundRemover(predictor, foreground_classes=[15])
result = remover.remove(image)

# Detect lanes
lanes = LaneDetector(predictor, lane_class_id=1)

# Parse scenes
parser = SceneParser(predictor, class_names=["bg", "road", "building", ...])
stats = parser.parse(image)  # {"road": 35.2, "building": 12.1, ...}

# Calculate areas
calc = AreaCalculator(predictor, pixels_per_meter=10.0)
areas = calc.calculate(image)  # {class_id: area_m2, ...}
```

| Solution | Description |
|----------|-------------|
| **BackgroundRemover** | Remove/replace backgrounds, generate alpha mattes |
| **LaneDetector** | Detect road lanes from segmentation |
| **SceneParser** | Break scene into labeled regions with area stats |
| **AreaCalculator** | Measure real-world areas from masks |

---

## Training

### Standard

```bash
flashseg train --model-size m --epochs 100 --num-classes 21 --device cuda
```

### LoRA Fine-Tuning

```bash
flashseg train --model-size m --lora --config configs/flashseg_m_512_lora.yaml
```

### Mixed Precision

```bash
flashseg train --model-size m --amp --device cuda
```

---

## Examples

| Script | What it does |
|--------|--------------|
| `train_voc.py` | Train on Pascal VOC |
| `predict_image.py` | Segment a single image |
| `background_removal.py` | Remove image background |
| `export_onnx.py` | Export to ONNX |
| `benchmark_model.py` | Measure FPS and latency |

---

## Project Structure

```
FlashSeg/
├── flashseg/                  # Main package
│   ├── cfg/                   # Configuration + YAML loading
│   ├── data/                  # Datasets, transforms
│   ├── engine/                # Trainer, Predictor, Exporter, Validator
│   ├── models/                # ShuffleNetV2, FPN, SegHead
│   ├── losses/                # CE, Dice, Focal, Combined
│   ├── nn/                    # ConvBnRelu, ASPP, DepthwiseSeparable
│   ├── utils/                 # Metrics, visualization
│   ├── solutions/             # Background removal, lanes, scene parsing
│   └── analytics/             # Benchmark, profiler
├── configs/                   # YAML configs (pick & train)
├── examples/                  # Ready-to-run scripts
├── tests/                     # Unit tests
├── docker/                    # Dockerfile + compose
├── pyproject.toml             # Package config
└── LICENSE                    # MIT
```

---

## Docker

```bash
docker build -t flashseg -f docker/Dockerfile .
docker run --gpus all -v $(pwd)/data:/app/data flashseg predict --model best.pth --source data/
```

---

## Contributing

```bash
git clone https://github.com/FlashVision/FlashSeg.git
cd FlashSeg
pip install -e ".[dev,all]"
ruff check flashseg/
pytest tests/ -v
```

---

## License

MIT — see [LICENSE](LICENSE).

---

<p align="center">
  <a href="https://github.com/FlashVision"><b>FlashVision</b></a> — Open-source lightweight AI
</p>
