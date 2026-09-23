# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden model for the fused QK-norm + RoPE + V-copy decode operator.

Input/output layout (bf16, flat):
    qkv_in  = [Q (n_q_heads * head_dim) | K (n_kv_heads * head_dim) | V]
    scratch = [qk_gamma (n_qk_heads * head_dim) | cos (head_dim) | sin (head_dim)]

Q and K get a per-head RMSNorm (gamma table shared across Q and K heads) and
then an HF ``rotate_half`` RoPE using the *duplicated-half* cos/sin tables
(``cos[i + head_dim/2] == cos[i]``); V is copied through.
"""

import numpy as np
import torch


def reference(qkv, qk_gamma, cos, sin, n_q_heads, n_kv_heads, head_dim, eps=1e-6):
    q = qkv[: n_q_heads * head_dim].view(n_q_heads, head_dim).float()
    k = qkv[n_q_heads * head_dim: n_q_heads * head_dim + n_kv_heads * head_dim] \
        .view(n_kv_heads, head_dim).float()
    v = qkv[n_q_heads * head_dim + n_kv_heads * head_dim:].clone()
    gamma = qk_gamma.view(n_q_heads + n_kv_heads, head_dim).float()

    def rms_rope(x, g):
        inv = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)
        y = x * inv * g
        h = head_dim // 2
        y1, y2 = y[:, :h], y[:, h:]
        c1, c2 = cos[:h].float(), cos[h:].float()
        s1, s2 = sin[:h].float(), sin[h:].float()
        return torch.cat([y1 * c1 - y2 * s1, y2 * c2 + y1 * s2], dim=-1)

    q = rms_rope(q, gamma[:n_q_heads])
    k = rms_rope(k, gamma[n_q_heads:])
    return torch.cat([q.flatten(), k.flatten(), v]).to(torch.bfloat16)


def generate_golden_reference(n_q_heads=16, n_kv_heads=8, head_dim=128, seed=42):
    """Random qkv / gamma; cos-sin tables are REAL cos/sin of random angles with
    the duplicated-half convention the kernel documents."""
    rng = np.random.default_rng(seed)
    n_qk = n_q_heads + n_kv_heads
    qkv = torch.from_numpy(
        rng.standard_normal(n_qk * head_dim + n_kv_heads * head_dim).astype(np.float32)
    ).to(torch.bfloat16)
    gamma = (torch.from_numpy(rng.standard_normal(n_qk * head_dim).astype(np.float32)) * 0.2 + 1.0) \
        .to(torch.bfloat16)
    ang = torch.from_numpy(rng.uniform(0, 2 * np.pi, size=head_dim // 2).astype(np.float32))
    cos = torch.cat([ang.cos(), ang.cos()]).to(torch.bfloat16)
    sin = torch.cat([ang.sin(), ang.sin()]).to(torch.bfloat16)
    out = reference(qkv, gamma, cos, sin, n_q_heads, n_kv_heads, head_dim)
    return {
        "qkv_in": qkv, "gamma": gamma, "cos": cos, "sin": sin, "qkv_out": out,
    }
