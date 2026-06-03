"""Position encodings: Sinusoidal 2D and Rotary Position Embeddings for SAM3."""

import math
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


def compute_axial_cis(
    dim: int,
    end_x: int,
    end_y: int,
    scale: float = 1.0,
    theta: float = 10000.0,
) -> Tuple[mx.array, mx.array]:
    """Compute 2D axial rotary position embeddings matching HF Sam3ViTRotaryEmbedding.

    Returns:
        cos: (end_x*end_y, dim) cosine embeddings
        sin: (end_x*end_y, dim) sine embeddings
    """
    # Frequencies: step by 4 (not 2) because we split dim into 4 parts: x_pair, y_pair
    freqs = 1.0 / (theta ** (mx.arange(0, dim, 4).astype(mx.float32) / dim))

    # Grid positions (row-major: y changes with row, x changes with column)
    flat_idx = mx.arange(end_x * end_y)
    x_positions = (flat_idx % end_x).astype(mx.float32) * scale
    y_positions = (flat_idx // end_x).astype(mx.float32) * scale

    # Outer products: (N, dim//4) each
    freqs_x = x_positions[:, None] * freqs[None, :]
    freqs_y = y_positions[:, None] * freqs[None, :]

    # Concatenate x and y: (N, dim//2)
    inv_freq = mx.concatenate([freqs_x, freqs_y], axis=-1)

    # repeat_interleave(2): [f0, f0, f1, f1, ...] -> (N, dim)
    inv_freq = mx.stack([inv_freq, inv_freq], axis=-1).reshape(inv_freq.shape[0], -1)

    return mx.cos(inv_freq), mx.sin(inv_freq)


def rotate_pairwise(x: mx.array) -> mx.array:
    """Pairwise rotation: (x0,x1,x2,x3,...) -> (-x1,x0,-x3,x2,...)"""
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1 = x[..., 0]
    x2 = x[..., 1]
    rotated = mx.stack([-x2, x1], axis=-1)
    return rotated.reshape(*rotated.shape[:-2], -1)


def apply_rotary_enc(
    xq: mx.array,
    xk: mx.array,
    cos: mx.array,
    sin: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Apply 2D rotary position encoding matching HF implementation.

    Formula: q_out = q * cos + rotate_pairwise(q) * sin

    Args:
        xq: (B, H, N, D) queries (already transposed for SDPA)
        xk: (B, H, N, D) keys
        cos: (N, D) cosine embeddings
        sin: (N, D) sine embeddings
    Returns:
        xq_out, xk_out: rotated queries and keys
    """
    xq_out = xq * cos + rotate_pairwise(xq) * sin
    xk_out = xk * cos + rotate_pairwise(xk) * sin
    return xq_out, xk_out

