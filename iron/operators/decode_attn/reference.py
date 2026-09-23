# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden model for the fused decode attention operator (S=1, GQA).

Layouts (bf16):
    Q   : (H, D)
    KV  : one combined arg; per head a region of 2*S*D = [K half | V half]
          (K occupies the first S*D, V the second) -- head-major, seq-major
          inside each half
    O   : (H, D)

Scores are raw Q·K dots scaled by ``scale = log2(e) / sqrt(D)`` -- the design
owns that value (design.py) and the kernel applies it to each finished block's
raw dot products before the streaming softmax, which uses base-2 exponentials
(hence the log2e term).  The causal boundary in the static arm is
``seq_pos = S_KV`` (every cached position valid).
"""

import math

import numpy as np
import torch


def reference(Q, K, V, seq_pos, head_dim, num_heads, num_kv_heads, causal=True):
    """Q: (H, D); K/V: (KV, S, D); seq_pos: int (inclusive causal boundary)."""
    scale = math.log2(math.e) / math.sqrt(head_dim)
    out = torch.zeros(num_heads, head_dim, dtype=torch.float32)
    for h in range(num_heads):
        kv = h * num_kv_heads // num_heads
        s = (Q[h].float() @ K[kv].float().T) * scale
        idx = torch.arange(s.shape[0])
        if causal:
            s = torch.where(idx <= seq_pos, s, torch.tensor(float("-inf")))
        p = torch.exp2(s - s.max())
        out[h] = (p @ V[kv].float()) / p.sum()
    return out.to(torch.bfloat16)


def generate_golden_reference(num_heads=32, num_kv_heads=8, head_dim=128,
                              seq_len_kv=512, seed=42):
    """Random Q and combined KV.  The combined arg packs, per head,
    [K half | V half]; the static arm's causal boundary is S_KV itself."""
    rng = np.random.default_rng(seed)
    Q = torch.from_numpy(rng.standard_normal((num_heads, head_dim)).astype(np.float32)) \
        .to(torch.bfloat16)
    K = (torch.from_numpy(rng.standard_normal((num_kv_heads, seq_len_kv, head_dim))
                          .astype(np.float32)) * 0.3).to(torch.bfloat16)
    V = (torch.from_numpy(rng.standard_normal((num_kv_heads, seq_len_kv, head_dim))
                                              .astype(np.float32))) * 0.3
    kv = torch.zeros(num_kv_heads, 2 * seq_len_kv, head_dim, dtype=torch.bfloat16)
    kv[:, :seq_len_kv] = K
    kv[:, seq_len_kv:] = V
    O = reference(Q, K, V, seq_len_kv - 1, head_dim, num_heads, num_kv_heads)
    return {
        "Q": Q,
        "kv_in": kv.reshape(num_kv_heads, 2 * seq_len_kv * head_dim),
        "O": O,
    }
