# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused Pre-Attention operator for decode (S=1).

Fuses: bare RMSNorm + QKV GEMV + RoPE into a single NPU dispatch.

Input: x (1024,) bf16 — pre-attention hidden state (after residual add)
Output: qkv (4096,) bf16 — QKV projection output (RoPE applied to Q+K only)

QKV split: Q = qkv[0:2048], K = qkv[2048:3072], V = qkv[3072:4096]
RoPE is applied to Q (2048 = 16 heads × 128) and K (1024 = 8 heads × 128)
but NOT V. This requires splitting QKV, applying RoPE to Q+K, then
concatenating back. Since OperatorSequence works on whole buffers,
we apply RoPE to the entire QKV (3072 elements = Q+K, V=1024 excluded).

For simplicity in this version, RoPE is applied to Q+K as a separate
step after the OperatorSequence (not fused yet). The runlist only
fuses RMSNorm + QKV GEMV.
"""

import aie.utils as aie_utils
import numpy as np

from iron.common.sequence import OperatorSequence
from iron.common.utils import get_shim_dma_limit
from iron.operators.gemv.op import GEMV
from iron.operators.rms_norm.op import RMSNorm
from iron.operators.rope.op import RoPE


class PreAttentionDecode(OperatorSequence):
    """Fused RMSNorm + QKV GEMV for single-token decode.

    Gamma is folded into QKV weight (W_qkv' = W_qkv * gamma).
    RoPE is applied separately (not fused — needs cos/sin table
    in scratch buffer, which OperatorSequence doesn't support yet
    for per-position-varying data).

    Runtime buffers:
    - input: "in" (1024 bf16)
    - output: "qkv_out" (4096 bf16)
    - scratch weights: "w_qkv" (per-layer, gamma-folded)
    """

    def __init__(self, embedding_dim, qkv_dim, head_dim, max_seq_len=512,
                 epsilon=1e-6, context=None):
        self.embedding_dim = embedding_dim  # 1024
        self.qkv_dim = qkv_dim  # 4096 (q+k+v)
        self.head_dim = head_dim  # 128
        self.max_seq_len = max_seq_len  # 512
        self.epsilon = epsilon

        dev = aie_utils.get_current_device()
        n_cols = get_shim_dma_limit(dev) // 2

        # Bare RMSNorm (gamma folded into QKV weight)
        rmsnorm = RMSNorm(
            size=embedding_dim,
            num_aie_columns=1,
            num_channels=1,
            tile_size=embedding_dim,
            weighted=False,
            epsilon=epsilon,
        )

        # QKV GEMV: W_qkv @ RMSNorm(x) → qkv (4096 elements)
        gemv_qkv = GEMV(
            M=qkv_dim,
            K=embedding_dim,
            num_aie_columns=n_cols,
            tile_size_input=4,
            tile_size_output=qkv_dim // n_cols,
        )

        runlist = [
            (rmsnorm, "in", "normed"),
            (gemv_qkv, "w_qkv", "normed", "qkv_out"),
        ]

        super().__init__(
            name=f"pre_attn_decode_e{embedding_dim}_q{qkv_dim}",
            runlist=runlist,
            input_args=["in"],
            output_args=["qkv_out"],
            context=context,
        )
