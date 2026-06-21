"""Tests for SAM, SAM2, and Mask2Former architectures."""

import torch


def test_sam_image_encoder():
    from flashseg.models.architectures.sam import SAMImageEncoder

    encoder = SAMImageEncoder(
        img_size=64, patch_size=8, embed_dim=64,
        depth=2, num_heads=4, out_channels=32,
    )
    encoder.eval()
    x = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        out = encoder(x)
    assert out.shape[0] == 1
    assert out.shape[2] == 32


def test_sam_prompt_encoder_points():
    from flashseg.models.architectures.sam import SAMPromptEncoder

    enc = SAMPromptEncoder(embed_dim=32, img_size=64)
    coords = torch.tensor([[[10.0, 20.0], [30.0, 40.0]]])
    labels = torch.tensor([[1, 0]])
    sparse, dense = enc(points=(coords, labels))
    assert sparse.shape[0] == 1
    assert sparse.shape[2] == 32
    assert dense is None


def test_sam_prompt_encoder_box():
    from flashseg.models.architectures.sam import SAMPromptEncoder

    enc = SAMPromptEncoder(embed_dim=32, img_size=64)
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    sparse, dense = enc(boxes=boxes)
    assert sparse.shape[0] == 1


def test_sam_mask_decoder():
    from flashseg.models.architectures.sam import SAMMaskDecoder

    decoder = SAMMaskDecoder(embed_dim=32, num_heads=4, num_mask_tokens=3, depth=1)
    decoder.eval()

    img_embed = torch.randn(1, 64, 32)
    sparse = torch.randn(1, 2, 32)
    with torch.no_grad():
        masks, iou_scores = decoder(img_embed, sparse, image_size=(64, 64))
    assert masks.shape == (1, 3, 64, 64)
    assert iou_scores.shape == (1, 3)


def test_sam_full():
    from flashseg.models.architectures.sam import SAM

    model = SAM(
        img_size=64, patch_size=8, embed_dim=64,
        encoder_depth=2, num_heads=4,
        decoder_embed_dim=32, num_mask_tokens=3,
    )
    model.eval()

    x = torch.randn(1, 3, 64, 64)
    coords = torch.tensor([[[16.0, 16.0]]])
    labels = torch.tensor([[1]])
    with torch.no_grad():
        result = model(x, points=(coords, labels))
    assert "masks" in result
    assert "iou_scores" in result
    assert result["masks"].shape[1] == 3


def test_sam_predict():
    from flashseg.models.architectures.sam import SAM

    model = SAM(
        img_size=64, patch_size=8, embed_dim=64,
        encoder_depth=2, num_heads=4,
        decoder_embed_dim=32, num_mask_tokens=3,
    )
    x = torch.randn(1, 3, 64, 64)
    result = model.predict(x, multimask=False)
    assert result["masks"].shape[1] == 1


def test_sam2_forward():
    from flashseg.models.architectures.sam2 import SAM2

    model = SAM2(
        img_size=64, patch_size=8, embed_dim=64,
        encoder_depth=2, num_heads=4,
        decoder_embed_dim=32, num_mask_tokens=3,
        memory_size=3, memory_tokens=16,
    )
    model.eval()

    x = torch.randn(1, 3, 64, 64)
    coords = torch.tensor([[[16.0, 16.0]]])
    labels = torch.tensor([[1]])
    with torch.no_grad():
        result = model(x, points=(coords, labels), use_memory=False)
    assert "masks" in result


def test_sam2_memory():
    from flashseg.models.architectures.sam2 import SAM2

    model = SAM2(
        img_size=64, patch_size=8, embed_dim=64,
        encoder_depth=2, num_heads=4,
        decoder_embed_dim=32, num_mask_tokens=3,
        memory_size=3, memory_tokens=16,
    )
    model.eval()

    model.reset_memory()
    x = torch.randn(1, 3, 64, 64)
    coords = torch.tensor([[[16.0, 16.0]]])
    labels = torch.tensor([[1]])

    with torch.no_grad():
        r1 = model(x, points=(coords, labels), use_memory=True)
        r2 = model(x, use_memory=True)
    assert r2["masks"].shape[0] == 1


def test_memory_attention():
    from flashseg.models.architectures.sam2 import MemoryAttention

    attn = MemoryAttention(embed_dim=32, num_heads=4, num_layers=1)
    current = torch.randn(1, 16, 32)
    memory = torch.randn(1, 32, 32)
    out = attn(current, memory)
    assert out.shape == current.shape


def test_mask2former_forward():
    from flashseg.models.architectures.mask2former import Mask2Former

    model = Mask2Former(
        num_classes=10,
        backbone_channels=[32, 64, 128],
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        num_queries=20,
    )
    model.eval()

    x = torch.randn(1, 3, 128, 128)
    with torch.no_grad():
        out = model(x)
    assert "pred_logits" in out
    assert "pred_masks" in out
    assert out["pred_logits"].shape == (1, 20, 11)  # 10 classes + no-object


def test_mask2former_semantic():
    from flashseg.models.architectures.mask2former import Mask2Former

    model = Mask2Former(
        num_classes=5,
        backbone_channels=[32, 64, 128],
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        num_queries=10,
    )

    x = torch.randn(1, 3, 64, 64)
    sem_map = model.predict_semantic(x)
    assert sem_map.shape == (1, 64, 64)
    assert sem_map.dtype == torch.int64


def test_mask2former_instance():
    from flashseg.models.architectures.mask2former import Mask2Former

    model = Mask2Former(
        num_classes=5,
        backbone_channels=[32, 64, 128],
        embed_dim=64,
        num_heads=4,
        num_layers=2,
        num_queries=10,
    )

    x = torch.randn(1, 3, 64, 64)
    results = model.predict_instance(x, score_threshold=0.0)
    assert len(results) == 1
    assert "labels" in results[0]
    assert "masks" in results[0]
