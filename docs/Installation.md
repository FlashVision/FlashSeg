# Installation

## From source

```bash
git clone https://github.com/FlashVision/FlashSeg.git
cd FlashSeg
pip install -e ".[all]"
```

## With extras

```bash
pip install -e ".[export]"      # ONNX export
pip install -e ".[analytics]"   # Benchmarking, plots
pip install -e ".[solutions]"   # Background removal, lanes
pip install -e ".[all]"         # Everything
```

## Verify

```bash
flashseg check
flashseg version
flashseg settings
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0
- OpenCV >= 4.5
