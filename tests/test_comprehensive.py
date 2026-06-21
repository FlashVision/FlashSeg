"""Comprehensive test suite for FlashSeg."""

import subprocess
import sys

import numpy as np
import pytest
import torch

B, C, H, W = 2, 3, 64, 64
NUM_CLASSES = 5


@pytest.fixture
def dummy_input():
    return torch.randn(B, C, H, W)


# ===================================================================
# 1. MODEL ARCHITECTURES
# ===================================================================


class TestFlashSeg:
    def test_forward(self, dummy_input):
        from flashseg.cfg.config import get_config
        from flashseg.models.build import build_model

        cfg = get_config(model_size="n", input_size=H, num_classes=NUM_CLASSES)
        model = build_model(cfg)
        model.eval()
        with torch.no_grad():
            out = model(dummy_input)
        assert out.shape[0] == B
        assert out.shape[1] == NUM_CLASSES

    def test_gradient_flow(self):
        from flashseg.cfg.config import get_config
        from flashseg.models.build import build_model

        cfg = get_config(model_size="n", input_size=64, num_classes=3)
        model = build_model(cfg)
        model.eval()
        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        out = model(x)
        out.sum().backward()
        assert x.grad is not None


class TestSAM:
    def test_forward(self):
        from flashseg.models.architectures.sam import SAM

        model = SAM(
            img_size=64,
            patch_size=8,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            decoder_embed_dim=32,
            num_mask_tokens=3,
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert "masks" in out
        assert "iou_scores" in out
        assert out["masks"].shape[0] == 1
        assert out["masks"].shape[1] == 3

    def test_with_point_prompt(self):
        from flashseg.models.architectures.sam import SAM

        model = SAM(
            img_size=64,
            patch_size=8,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            decoder_embed_dim=32,
            num_mask_tokens=3,
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        coords = torch.tensor([[[32.0, 32.0]]])
        labels = torch.tensor([[1]])
        with torch.no_grad():
            out = model(x, points=(coords, labels))
        assert out["masks"].shape[0] == 1

    def test_with_box_prompt(self):
        from flashseg.models.architectures.sam import SAM

        model = SAM(
            img_size=64,
            patch_size=8,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            decoder_embed_dim=32,
            num_mask_tokens=3,
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
        with torch.no_grad():
            out = model(x, boxes=boxes)
        assert "masks" in out

    def test_gradient(self):
        from flashseg.models.architectures.sam import SAM

        model = SAM(
            img_size=64,
            patch_size=8,
            embed_dim=32,
            encoder_depth=1,
            num_heads=4,
            decoder_embed_dim=16,
            num_mask_tokens=2,
        )
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        out = model(x)
        out["masks"].sum().backward()
        assert x.grad is not None


class TestSAM2:
    def test_forward(self):
        from flashseg.models.architectures.sam2 import SAM2

        model = SAM2(
            img_size=64,
            patch_size=8,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            decoder_embed_dim=32,
            num_mask_tokens=3,
            memory_size=4,
            memory_tokens=16,
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert "masks" in out
        assert "iou_scores" in out

    def test_track_video(self):
        from flashseg.models.architectures.sam2 import SAM2

        model = SAM2(
            img_size=64,
            patch_size=8,
            embed_dim=64,
            encoder_depth=2,
            num_heads=4,
            decoder_embed_dim=32,
            num_mask_tokens=2,
            memory_size=4,
            memory_tokens=16,
        )
        model.eval()
        frames = [torch.randn(1, 3, 64, 64) for _ in range(3)]
        with torch.no_grad():
            results = model.track_video(frames)
        assert len(results) == 3


class TestMask2Former:
    def test_forward(self):
        from flashseg.models.architectures.mask2former import Mask2Former

        model = Mask2Former(
            num_classes=NUM_CLASSES,
            embed_dim=32,
            num_queries=10,
            backbone_channels=[32, 64, 128],
            num_heads=4,
            num_layers=1,
        )
        model.eval()
        x = torch.randn(1, 3, 64, 64)
        with torch.no_grad():
            out = model(x)
        assert "pred_masks" in out
        assert "pred_logits" in out
        assert out["pred_logits"].shape[0] == 1
        assert out["pred_logits"].shape[1] == 10

    def test_gradient(self):
        from flashseg.models.architectures.mask2former import Mask2Former

        model = Mask2Former(
            num_classes=3, embed_dim=32, num_queries=5, backbone_channels=[32, 64, 128], num_heads=4, num_layers=1
        )
        x = torch.randn(1, 3, 64, 64, requires_grad=True)
        out = model(x)
        out["pred_logits"].sum().backward()
        assert x.grad is not None


# ===================================================================
# 2. LOSSES
# ===================================================================


class TestLosses:
    def test_cross_entropy_loss(self):
        from flashseg.losses import CrossEntropyLoss

        loss_fn = CrossEntropyLoss()
        pred = torch.randn(2, NUM_CLASSES, 32, 32)
        target = torch.randint(0, NUM_CLASSES, (2, 32, 32))
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_dice_loss(self):
        from flashseg.losses import DiceLoss

        loss_fn = DiceLoss()
        pred = torch.randn(2, NUM_CLASSES, 32, 32)
        target = torch.randint(0, NUM_CLASSES, (2, 32, 32))
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_focal_loss(self):
        from flashseg.losses import FocalLoss

        loss_fn = FocalLoss()
        pred = torch.randn(2, NUM_CLASSES, 32, 32)
        target = torch.randint(0, NUM_CLASSES, (2, 32, 32))
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_combined_loss(self):
        from flashseg.losses import CombinedLoss

        loss_fn = CombinedLoss()
        pred = torch.randn(2, NUM_CLASSES, 32, 32)
        target = torch.randint(0, NUM_CLASSES, (2, 32, 32))
        loss = loss_fn(pred, target)
        assert torch.isfinite(loss)

    def test_loss_gradient(self):
        from flashseg.losses import DiceLoss

        loss_fn = DiceLoss()
        pred = torch.randn(1, 3, 16, 16, requires_grad=True)
        target = torch.randint(0, 3, (1, 16, 16))
        loss = loss_fn(pred, target)
        loss.backward()
        assert pred.grad is not None


# ===================================================================
# 3. TASKS
# ===================================================================


class TestTasks:
    def test_instance_segmentor(self):
        from flashseg.tasks import InstanceSegmentor

        seg = InstanceSegmentor(num_classes=NUM_CLASSES)
        assert seg is not None

    def test_panoptic_segmentor(self):
        from flashseg.tasks import PanopticSegmentor

        seg = PanopticSegmentor(thing_classes={0, 1}, stuff_classes={2, 3, 4})
        assert seg is not None

    def test_interactive_segmentor_import(self):
        from flashseg.tasks import InteractiveSegmentor

        assert InteractiveSegmentor is not None

    def test_compute_mask_ap(self):
        from flashseg.tasks import compute_mask_ap

        pred_masks = [np.random.rand(64, 64) > 0.5 for _ in range(3)]
        pred_scores = [0.9, 0.8, 0.7]
        pred_classes = [0, 1, 0]
        gt_masks = [np.random.rand(64, 64) > 0.5 for _ in range(2)]
        gt_classes = [0, 1]
        result = compute_mask_ap(pred_masks, pred_scores, pred_classes, gt_masks, gt_classes)
        assert isinstance(result, dict)

    def test_compute_pq_import(self):
        from flashseg.tasks import compute_pq

        assert callable(compute_pq)


# ===================================================================
# 4. CLI
# ===================================================================


class TestCLI:
    def test_version(self):
        result = subprocess.run(
            [sys.executable, "-m", "flashseg.cli", "version"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0

    def test_no_command(self):
        result = subprocess.run(
            [sys.executable, "-m", "flashseg.cli"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0


# ===================================================================
# 5. ENGINE
# ===================================================================


class TestEngine:
    def test_imports(self):
        from flashseg.engine.exporter import Exporter
        from flashseg.engine.trainer import Trainer

        assert Trainer is not None
        assert Exporter is not None


# ===================================================================
# 6. DATA
# ===================================================================


class TestData:
    def test_transforms(self):
        from flashseg.data.transforms import get_train_transforms, get_val_transforms

        train_t = get_train_transforms(input_size=64)
        val_t = get_val_transforms(input_size=64)
        assert train_t is not None
        assert val_t is not None


# ===================================================================
# 7. UTILS
# ===================================================================


class TestUtils:
    def test_metrics(self):
        from flashseg.utils.metrics import compute_miou

        pred = torch.randint(0, 5, (10, 32, 32))
        target = torch.randint(0, 5, (10, 32, 32))
        miou = compute_miou(pred, target, num_classes=5)
        assert isinstance(miou, float)
        assert 0 <= miou <= 1

    def test_visualization(self):
        from flashseg.utils.visualization import colorize_mask

        mask = np.zeros((32, 32), dtype=np.int64)
        colored = colorize_mask(mask)
        assert colored.shape[:2] == (32, 32)


# ===================================================================
# 8. SOLUTIONS
# ===================================================================


class TestSolutions:
    def test_background_remover(self):
        from flashseg.solutions import BackgroundRemover

        assert BackgroundRemover is not None

    def test_lane_detector(self):
        from flashseg.solutions import LaneDetector

        assert LaneDetector is not None

    def test_area_calculator(self):
        from flashseg.solutions import AreaCalculator

        assert AreaCalculator is not None

    def test_scene_parser(self):
        from flashseg.solutions import SceneParser

        assert SceneParser is not None


# ===================================================================
# 9. CONFIG
# ===================================================================


class TestConfig:
    def test_get_config(self):
        from flashseg.cfg.config import get_config

        cfg = get_config(model_size="m", input_size=256, num_classes=10)
        assert cfg.num_classes == 10
        assert cfg.input_size == 256

    def test_config_variants(self):
        from flashseg.cfg.config import get_config

        for size in ["n", "s", "m", "l"]:
            cfg = get_config(model_size=size)
            assert cfg.width_mult > 0


# ===================================================================
# 10. BACKBONE, NECK, HEAD
# ===================================================================


class TestSubmodules:
    def test_shufflenet_backbone(self):
        from flashseg.models.backbone.shufflenetv2 import ShuffleNetV2

        bb = ShuffleNetV2(width_mult=0.25)
        x = torch.randn(1, 3, 64, 64)
        feats = bb(x)
        assert isinstance(feats, (list, tuple))

    def test_fpn_neck(self):
        from flashseg.models.neck.fpn import FPN

        backbone_ch = [24, 48, 96]
        fpn = FPN(in_channels=backbone_ch, out_channels=64)
        feats = [torch.randn(1, 24, 16, 16), torch.randn(1, 48, 8, 8), torch.randn(1, 96, 4, 4)]
        out = fpn(feats)
        assert out is not None

    def test_seg_head(self):
        from flashseg.models.head.seg_head import SegHead

        head = SegHead(in_channels=64, num_classes=5, input_size=32)
        feats = [torch.randn(1, 64, 16, 16), torch.randn(1, 64, 8, 8), torch.randn(1, 64, 4, 4)]
        out = head(feats)
        assert out.shape[1] == 5


# ===================================================================
# 11. EDGE CASES
# ===================================================================


class TestEdgeCases:
    def test_wrong_channels(self):
        from flashseg.cfg.config import get_config
        from flashseg.models.build import build_model

        cfg = get_config(model_size="n", input_size=64, num_classes=3)
        model = build_model(cfg)
        with pytest.raises(RuntimeError):
            model(torch.randn(1, 1, 64, 64))

    def test_single_class(self):
        from flashseg.cfg.config import get_config
        from flashseg.models.build import build_model

        cfg = get_config(model_size="n", input_size=64, num_classes=1)
        model = build_model(cfg)
        model.eval()
        with torch.no_grad():
            out = model(torch.randn(1, 3, 64, 64))
        assert out.shape[1] == 1


# ===================================================================
# 12. INTEGRATION
# ===================================================================


class TestIntegration:
    def test_full_pipeline(self):
        from flashseg.cfg.config import get_config
        from flashseg.losses import CombinedLoss
        from flashseg.models.build import build_model

        cfg = get_config(model_size="n", input_size=64, num_classes=3)
        model = build_model(cfg)
        loss_fn = CombinedLoss()

        model.train()
        x = torch.randn(2, 3, 64, 64)
        target = torch.randint(0, 3, (2, 64, 64))

        out = model(x)
        loss = loss_fn(out, target)
        assert torch.isfinite(loss)

        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            pred = model(x)
        assert pred.shape == (2, 3, 64, 64)


# ===================================================================
# 13. NN BLOCKS
# ===================================================================


class TestNNBlocks:
    def test_convbnrelu(self):
        from flashseg.nn.blocks import ConvBnRelu

        conv = ConvBnRelu(3, 16, 3)
        x = torch.randn(1, 3, 32, 32)
        out = conv(x)
        assert out.shape[1] == 16

    def test_depthwise_separable_conv(self):
        from flashseg.nn.blocks import DepthwiseSeparableConv

        block = DepthwiseSeparableConv(16, 32)
        x = torch.randn(1, 16, 32, 32)
        out = block(x)
        assert out.shape[1] == 32

    def test_aspp(self):
        from flashseg.nn.blocks import ASPP

        aspp = ASPP(in_ch=64, out_ch=32)
        aspp.eval()
        x = torch.randn(1, 64, 16, 16)
        with torch.no_grad():
            out = aspp(x)
        assert out.shape[1] == 32
