"""MLX primitives for the Kimodo motion transformer."""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
from numpy.typing import NDArray


def as_mx(values: NDArray[np.float32]) -> mx.array:
    """Upload an F32 numpy tensor to the default MLX device."""
    return mx.array(np.ascontiguousarray(values, dtype=np.float32))


def linear(x: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    """Apply a PyTorch-layout linear layer: y = x @ W.T + b."""
    return mx.matmul(x, mx.swapaxes(weight, -1, -2)) + bias


def layer_norm(x: mx.array, weight: mx.array, bias: mx.array) -> mx.array:
    """Apply LayerNorm with eps=1e-5 over the last axis."""
    mean = mx.mean(x, axis=-1, keepdims=True)
    variance = mx.mean((x - mean) * (x - mean), axis=-1, keepdims=True)
    normalized = (x - mean) * mx.rsqrt(variance + 1e-5)
    return normalized * weight + bias


def gelu_erf(x: mx.array) -> mx.array:
    """GELU with the erf formulation used by PyTorch and kimodo.cpp."""
    return 0.5 * x * (1.0 + mx.erf(x / math.sqrt(2.0)))


def sinusoidal_encoding(length: int, width: int) -> mx.array:
    """Return [length, width] sinusoidal positional encodings."""
    position = np.arange(length, dtype=np.float32)[:, None]
    div_term = np.power(10000.0, -np.arange(0, width, 2, dtype=np.float32) / width)
    encoded = np.zeros((length, width), dtype=np.float32)
    encoded[:, 0::2] = np.sin(position * div_term)
    encoded[:, 1::2] = np.cos(position * div_term)
    return as_mx(encoded)


def attention(x: mx.array, in_weight: mx.array, in_bias: mx.array, out_weight: mx.array, out_bias: mx.array, heads: int) -> mx.array:
    """Multi-head self-attention with concatenated QKV projection."""
    batch, seq, width = x.shape
    head_dim = width // heads
    qkv = linear(x, in_weight, in_bias)
    query, key, value = mx.split(qkv, 3, axis=-1)
    shape = (batch, seq, heads, head_dim)
    query = mx.transpose(mx.reshape(query, shape), (0, 2, 1, 3))
    key = mx.transpose(mx.reshape(key, shape), (0, 2, 1, 3))
    value = mx.transpose(mx.reshape(value, shape), (0, 2, 1, 3))
    scores = mx.matmul(query, mx.swapaxes(key, -1, -2)) / math.sqrt(head_dim)
    context = mx.matmul(mx.softmax(scores, axis=-1), value)
    joined = mx.reshape(mx.transpose(context, (0, 2, 1, 3)), (batch, seq, width))
    return linear(joined, out_weight, out_bias)


def encoder_layer(x: mx.array, weights: dict[str, mx.array], prefix: str, heads: int) -> mx.array:
    """Post-norm transformer encoder layer matching PyTorch/kimodo.cpp."""
    attended = attention(
        x,
        weights[prefix + "self_attn.in_proj_weight"],
        weights[prefix + "self_attn.in_proj_bias"],
        weights[prefix + "self_attn.out_proj.weight"],
        weights[prefix + "self_attn.out_proj.bias"],
        heads,
    )
    x = layer_norm(x + attended, weights[prefix + "norm1.weight"], weights[prefix + "norm1.bias"])
    hidden = linear(x, weights[prefix + "linear1.weight"], weights[prefix + "linear1.bias"])
    hidden = linear(gelu_erf(hidden), weights[prefix + "linear2.weight"], weights[prefix + "linear2.bias"])
    return layer_norm(x + hidden, weights[prefix + "norm2.weight"], weights[prefix + "norm2.bias"])
