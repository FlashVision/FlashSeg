# FAQ

## What input format does FlashSeg need?

Images in any common format (JPG, PNG, BMP) and masks as grayscale images where each pixel value is the class index.

## How to prepare custom data?

```
data/
├── train/
│   ├── images/   (RGB images)
│   └── masks/    (grayscale, pixel value = class ID)
└── val/
    ├── images/
    └── masks/
```

## What loss function is used?

Combined Cross-Entropy + Dice loss by default. Focal loss also available for class imbalance.

## Can I use a custom backbone?

Yes, modify `flashseg/models/backbone/` and update the build function.

## How to export for mobile?

```bash
flashseg export --model best.pth --output model.onnx --simplify
```

Then convert ONNX to TFLite or CoreML for mobile deployment.
