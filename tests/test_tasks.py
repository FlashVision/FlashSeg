"""Tests for instance, panoptic, and interactive segmentation tasks."""

import numpy as np
import torch


def test_mask_iou():
    from flashseg.tasks.instance import _mask_iou

    mask1 = np.zeros((10, 10), dtype=bool)
    mask1[2:8, 2:8] = True
    mask2 = np.zeros((10, 10), dtype=bool)
    mask2[4:10, 4:10] = True

    iou = _mask_iou(mask1, mask2)
    assert 0 < iou < 1


def test_compute_mask_ap():
    from flashseg.tasks.instance import compute_mask_ap

    H, W = 32, 32
    gt_mask = np.zeros((H, W), dtype=bool)
    gt_mask[5:20, 5:20] = True

    pred_mask = np.zeros((H, W), dtype=bool)
    pred_mask[6:21, 6:21] = True

    result = compute_mask_ap(
        pred_masks=[pred_mask],
        pred_scores=[0.9],
        pred_classes=[0],
        gt_masks=[gt_mask],
        gt_classes=[0],
    )
    assert "AP" in result
    assert "AP50" in result
    assert result["AP50"] > 0


def test_instance_segmentor_process():
    from flashseg.tasks.instance import InstanceSegmentor

    seg = InstanceSegmentor(num_classes=5, score_threshold=0.0)

    pred_logits = torch.randn(10, 6)
    pred_masks = torch.randn(10, 16, 16)
    instances = seg.process(pred_logits, pred_masks, image_size=(64, 64))
    assert isinstance(instances, list)
    for inst in instances:
        assert inst.instance_id > 0
        assert inst.mask.shape == (64, 64)


def test_panoptic_pq():
    from flashseg.tasks.panoptic import PanopticSegment, compute_pq

    H, W = 32, 32
    pred_map = np.zeros((H, W), dtype=np.int32)
    pred_map[0:16, 0:32] = 1
    pred_map[16:32, 0:32] = 2

    gt_map = np.zeros((H, W), dtype=np.int32)
    gt_map[0:16, 0:32] = 1
    gt_map[16:32, 0:32] = 2

    pred_segs = [
        PanopticSegment(segment_id=1, class_id=0, is_thing=False, area=512),
        PanopticSegment(segment_id=2, class_id=1, is_thing=True, area=512),
    ]
    gt_segs = [
        PanopticSegment(segment_id=1, class_id=0, is_thing=False, area=512),
        PanopticSegment(segment_id=2, class_id=1, is_thing=True, area=512),
    ]

    result = compute_pq(pred_map, pred_segs, gt_map, gt_segs)
    assert result["PQ"] == 1.0
    assert result["SQ"] == 1.0
    assert result["RQ"] == 1.0


def test_panoptic_segmentor_merge():
    from flashseg.tasks.panoptic import PanopticSegmentor

    seg = PanopticSegmentor(
        thing_classes={0, 1},
        stuff_classes={2, 3},
        score_threshold=0.0,
        stuff_area_threshold=0,
    )

    pred_logits = torch.randn(10, 5)  # 4 classes + no-object
    pred_masks = torch.randn(10, 16, 16)
    result = seg.merge(pred_logits, pred_masks, image_size=(64, 64))

    assert result.panoptic_map.shape == (64, 64)
    assert result.semantic_map.shape == (64, 64)


def test_interactive_session():
    from flashseg.tasks.interactive import InteractiveSession

    session = InteractiveSession(image_size=(256, 256))
    session.add_point(100, 100, is_foreground=True)
    session.add_point(50, 50, is_foreground=False)
    assert len(session.points) == 2

    session.undo()
    assert len(session.points) == 1

    session.clear()
    assert len(session.points) == 0
