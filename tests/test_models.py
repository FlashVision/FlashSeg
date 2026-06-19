"""Test FlashSeg models."""

import torch
import pytest
from flashseg.cfg.config import get_config
from flashseg.models.build import build_model


@pytest.mark.parametrize("model_size", ["n", "s", "m", "l"])
def test_model_forward(model_size):
    config = get_config(model_size=model_size, input_size=256, num_classes=10)
    model = build_model(config)
    model.eval()

    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        out = model(x)

    assert out.shape == (1, 10, 256, 256), f"Expected (1, 10, 256, 256), got {out.shape}"


def test_model_relative_sizes():
    sizes = ["n", "s", "m", "l"]
    param_counts = []
    for size in sizes:
        config = get_config(model_size=size, input_size=256, num_classes=10)
        model = build_model(config)
        params = sum(p.numel() for p in model.parameters())
        param_counts.append(params)

    for i in range(1, len(param_counts)):
        assert param_counts[i] > param_counts[i - 1], (
            f"Model {sizes[i]} should have more params than {sizes[i-1]}"
        )
