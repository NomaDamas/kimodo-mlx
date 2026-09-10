"""Encoder-layer parity against PyTorch's post-norm transformer primitives."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch.nn import functional as F

from kimodo_mlx.layers import as_mx, encoder_layer, gelu_erf


def test_gelu_erf_matches_pytorch() -> None:
    """Given a mixed-sign vector, MLX GELU matches torch.nn.GELU(exact)."""
    values = np.linspace(-3.0, 3.0, 17, dtype=np.float32)
    expected = F.gelu(torch.from_numpy(values), approximate="none").numpy()

    actual = np.array(gelu_erf(as_mx(values)))

    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)


def test_encoder_layer_matches_pytorch_post_norm() -> None:
    """Given shared weights, one MLX encoder layer matches PyTorch post-norm."""
    torch.manual_seed(0)
    batch, seq, width, heads = 2, 5, 16, 4
    x = torch.randn(batch, seq, width)
    layer = torch.nn.TransformerEncoderLayer(
        d_model=width,
        nhead=heads,
        dim_feedforward=32,
        dropout=0.0,
        activation="gelu",
        batch_first=True,
        norm_first=False,
    )
    layer.eval()
    with torch.no_grad():
        expected = layer(x).numpy()
        weights = {
            "self_attn.in_proj_weight": layer.self_attn.in_proj_weight.numpy(),
            "self_attn.in_proj_bias": layer.self_attn.in_proj_bias.numpy(),
            "self_attn.out_proj.weight": layer.self_attn.out_proj.weight.numpy(),
            "self_attn.out_proj.bias": layer.self_attn.out_proj.bias.numpy(),
            "linear1.weight": layer.linear1.weight.numpy(),
            "linear1.bias": layer.linear1.bias.numpy(),
            "linear2.weight": layer.linear2.weight.numpy(),
            "linear2.bias": layer.linear2.bias.numpy(),
            "norm1.weight": layer.norm1.weight.numpy(),
            "norm1.bias": layer.norm1.bias.numpy(),
            "norm2.weight": layer.norm2.weight.numpy(),
            "norm2.bias": layer.norm2.bias.numpy(),
        }
    mx_weights = {key: as_mx(value) for key, value in weights.items()}
    actual = np.array(encoder_layer(as_mx(x.numpy()), mx_weights, "", heads))

    np.testing.assert_allclose(actual, expected, rtol=1e-4, atol=1e-4)
    assert math.isfinite(float(np.max(np.abs(actual))))
