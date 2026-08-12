# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Reference implementation for RoPE (Rotary Position Embedding).

Golden reference using PyTorch, matching the AIE2P rope.cc kernel.
"""

import numpy as np
import torch
from ml_dtypes import bfloat16


def precompute_rope_cache(head_dim, max_seq_len, base=1000000.0):
    """Precompute cos/sin tables of shape (max_seq_len, head_dim)."""
    half = head_dim // 2
    inv_freq = 1.0 / (
        base ** (torch.arange(0, half, dtype=torch.float32) / half)
    )
    t = torch.arange(max_seq_len, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)  # (max_seq_len, half)
    emb = torch.cat([freqs, freqs], dim=-1)  # (max_seq_len, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x):
    """Rotate half of the last dimension: cat([-x[half:], x[:half]], dim=-1)."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope_ref(q, k, cos, sin, pos_offset=0):
    """Apply RoPE to q and k.

    Args:
        q: (H, S, d) bf16 tensor
        k: (KV_H, S, d) bf16 tensor
        cos, sin: (max_seq_len, d) float32 tables
        pos_offset: starting position index

    Returns:
        q_rot, k_rot: same shape/dtype as q, k
    """
    S = q.shape[1]
    cos_s = cos[pos_offset : pos_offset + S]  # (S, d)
    sin_s = sin[pos_offset : pos_offset + S]

    q_f = q.to(torch.float32)
    k_f = k.to(torch.float32)

    # broadcast (S, d) over (H, S, d)
    cos_b = cos_s.unsqueeze(0)  # (1, S, d)
    sin_b = sin_s.unsqueeze(0)

    q_rot = q_f * cos_b + rotate_half(q_f) * sin_b
    k_rot = k_f * cos_b + rotate_half(k_f) * sin_b

    return q_rot.to(q.dtype), k_rot.to(k.dtype)


def generate_golden_reference(heads=1, kv_heads=1, S_q=256, d=64,
                               pos_offset=0, seed=42):
    """Generate golden reference for RoPE test.

    Returns dict with:
        Q_in: original Q (heads, S, d) bf16
        K_in: original K (kv_heads, S, d) bf16
        V_in: original V (kv_heads, S, d) bf16 (untouched by RoPE)
        cos: cos table (S, d) float32
        sin: sin table (S, d) float32
        Q_rot: Q after RoPE (heads, S, d) bf16
        K_rot: K after RoPE (kv_heads, S, d) bf16
    """
    torch.manual_seed(seed)
    val_range = 4

    Q_in = (torch.rand(heads, S_q, d, dtype=torch.bfloat16) * val_range)
    K_in = (torch.rand(kv_heads, S_q, d, dtype=torch.bfloat16) * val_range)
    V_in = (torch.rand(kv_heads, S_q, d, dtype=torch.bfloat16) * val_range)

    cos, sin = precompute_rope_cache(d, S_q + pos_offset)
    cos_s = cos[pos_offset : pos_offset + S_q]
    sin_s = sin[pos_offset : pos_offset + S_q]

    Q_rot, K_rot = apply_rope_ref(Q_in, K_in, cos, sin, pos_offset=pos_offset)

    return {
        "Q_in": Q_in,
        "K_in": K_in,
        "V_in": V_in,
        "cos": cos_s.to(torch.bfloat16),
        "sin": sin_s.to(torch.bfloat16),
        "Q_rot": Q_rot,
        "K_rot": K_rot,
    }
