"""FlashSeg CLI."""

import argparse
import logging
import sys

from flashseg import __version__


def main():
    """FlashSeg CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="flashseg",
        description="FlashSeg — Ultra-lightweight real-time image segmentation",
    )
    subparsers = parser.add_subparsers(dest="command")

    # Train
    train_parser = subparsers.add_parser("train", help="Train a segmentation model")
    train_parser.add_argument("--model-size", default="m", choices=["n", "s", "m", "l"])
    train_parser.add_argument("--train-images", required=True)
    train_parser.add_argument("--train-masks", required=True)
    train_parser.add_argument("--val-images", required=True)
    train_parser.add_argument("--val-masks", required=True)
    train_parser.add_argument("--num-classes", type=int, default=21)
    train_parser.add_argument("--input-size", type=int, default=512)
    train_parser.add_argument("--epochs", type=int, default=100)
    train_parser.add_argument("--batch-size", type=int, default=16)
    train_parser.add_argument("--lr", type=float, default=0.01)
    train_parser.add_argument("--device", default="cuda")
    train_parser.add_argument("--save-dir", default="workspace")
    train_parser.add_argument("--amp", action="store_true")
    train_parser.add_argument("--lora", action="store_true")
    train_parser.add_argument("--config", type=str, help="YAML config file")

    # Predict
    pred_parser = subparsers.add_parser("predict", help="Run segmentation inference")
    pred_parser.add_argument("--model", required=True)
    pred_parser.add_argument("--source", required=True)
    pred_parser.add_argument("--model-size", default="m")
    pred_parser.add_argument("--num-classes", type=int, default=21)
    pred_parser.add_argument("--input-size", type=int, default=512)
    pred_parser.add_argument("--device", default="cuda")
    pred_parser.add_argument("--save-dir", default="output")

    # Validate
    val_parser = subparsers.add_parser("val", help="Validate model")
    val_parser.add_argument("--model", required=True)
    val_parser.add_argument("--val-images", required=True)
    val_parser.add_argument("--val-masks", required=True)
    val_parser.add_argument("--model-size", default="m")
    val_parser.add_argument("--num-classes", type=int, default=21)
    val_parser.add_argument("--input-size", type=int, default=512)
    val_parser.add_argument("--device", default="cuda")

    # Export
    export_parser = subparsers.add_parser("export", help="Export to ONNX")
    export_parser.add_argument("--model", required=True)
    export_parser.add_argument("--output", default="model.onnx")
    export_parser.add_argument("--model-size", default="m")
    export_parser.add_argument("--num-classes", type=int, default=21)
    export_parser.add_argument("--input-size", type=int, default=512)
    export_parser.add_argument("--simplify", action="store_true")

    # Utility commands
    subparsers.add_parser("version", help="Print version")
    subparsers.add_parser("check", help="Run health check")
    subparsers.add_parser("settings", help="Show system info")

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    if args.command == "version":
        print(f"flashseg {__version__}")

    elif args.command == "check":
        _run_check()

    elif args.command == "settings":
        _show_settings()

    elif args.command == "train":
        from flashseg.engine.trainer import Trainer
        trainer = Trainer(
            model_size=args.model_size,
            train_images=args.train_images,
            train_masks=args.train_masks,
            val_images=args.val_images,
            val_masks=args.val_masks,
            num_classes=args.num_classes,
            input_size=args.input_size,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=args.device,
            save_dir=args.save_dir,
            amp=args.amp,
            use_lora=args.lora,
            config_path=args.config,
        )
        trainer.train()

    elif args.command == "predict":
        from flashseg.engine.predictor import Predictor
        predictor = Predictor(
            model_path=args.model,
            model_size=args.model_size,
            num_classes=args.num_classes,
            input_size=args.input_size,
            device=args.device,
        )
        predictor.predict_directory(args.source, save_dir=args.save_dir)

    elif args.command == "val":
        from flashseg.engine.validator import Validator
        validator = Validator(
            model_path=args.model,
            val_images=args.val_images,
            val_masks=args.val_masks,
            model_size=args.model_size,
            num_classes=args.num_classes,
            input_size=args.input_size,
            device=args.device,
        )
        validator.validate()

    elif args.command == "export":
        from flashseg.engine.exporter import Exporter
        exporter = Exporter(
            model_path=args.model,
            model_size=args.model_size,
            num_classes=args.num_classes,
            input_size=args.input_size,
        )
        exporter.export(output=args.output, simplify=args.simplify)

    else:
        parser.print_help()


def _run_check():
    """Run health check."""
    print("FlashSeg Health Check")
    print("=" * 40)
    checks = []

    try:
        import torch
        checks.append(("PyTorch", f"{torch.__version__}"))
        checks.append(("CUDA available", str(torch.cuda.is_available())))
        if torch.cuda.is_available():
            checks.append(("GPU", torch.cuda.get_device_name(0)))
    except ImportError:
        checks.append(("PyTorch", "NOT INSTALLED"))

    try:
        import cv2
        checks.append(("OpenCV", cv2.__version__))
    except ImportError:
        checks.append(("OpenCV", "NOT INSTALLED"))

    try:
        import flashseg
        checks.append(("FlashSeg", flashseg.__version__))
    except Exception as e:
        checks.append(("FlashSeg", f"ERROR: {e}"))

    try:
        from flashseg.models.build import build_model
        from flashseg.cfg.config import get_config
        config = get_config(model_size="m", input_size=512, num_classes=21)
        model = build_model(config)
        params = sum(p.numel() for p in model.parameters())
        checks.append(("Model build", f"OK ({params:,} params)"))
    except Exception as e:
        checks.append(("Model build", f"FAILED: {e}"))

    for name, status in checks:
        print(f"  {name:20s}: {status}")

    print("=" * 40)
    print("All checks passed!" if all("NOT INSTALLED" not in s and "FAILED" not in s for _, s in checks) else "Some checks failed.")


def _show_settings():
    """Show system settings."""
    import platform
    print("FlashSeg System Info")
    print("=" * 40)
    print(f"  Python:    {platform.python_version()}")
    print(f"  Platform:  {platform.platform()}")

    try:
        import torch
        print(f"  PyTorch:   {torch.__version__}")
        print(f"  CUDA:      {torch.version.cuda or 'N/A'}")
        if torch.cuda.is_available():
            print(f"  GPU:       {torch.cuda.get_device_name(0)}")
            mem = torch.cuda.get_device_properties(0).total_mem / 1024**3
            print(f"  GPU RAM:   {mem:.1f} GB")
        else:
            print("  GPU:       Not available")
    except ImportError:
        print("  PyTorch:   Not installed")

    try:
        import flashseg
        print(f"  FlashSeg:  {flashseg.__version__}")
    except ImportError:
        pass
    print("=" * 40)


if __name__ == "__main__":
    main()
