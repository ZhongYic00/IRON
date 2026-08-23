# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import torch
from ml_dtypes import bfloat16


def reference(x, gamma, head_dim, n_q_heads, n_k_heads, epsilon=1e-6):
    """CPU reference for QK-norm.

    x:    (n_q_heads*head_dim + n_k_heads*head_dim,) bf16 — flat [Q | K]
    gamma: (2*head_dim,) bf16 — [q_gamma | k_gamma]
    Returns per-head RMSNorm with shared per-group gamma (Qwen3 semantics).
    """
    n_qk_heads = n_q_heads + n_k_heads
    x = x.to(torch.float32).view(n_qk_heads, head_dim)
    q_gamma = gamma[:head_dim].to(torch.float32)
    k_gamma = gamma[head_dim:].to(torch.float32)

    out = torch.empty_like(x)
    for h in range(n_qk_heads):
        g = q_gamma if h < n_q_heads else k_gamma
        xh = x[h]
        rms = torch.sqrt(xh.pow(2).mean() + epsilon)
        out[h] = xh / rms * g
    return out.reshape(-1).to(torch.bfloat16)
