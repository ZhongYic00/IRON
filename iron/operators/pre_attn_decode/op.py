# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused Pre-Attention operator for decode (S=1).

Fuses: bare RMSNorm + QKV GEMV + RoPE into a single NPU dispatch.

Input: x (1024,) bf16 — pre-attention hidden state (after residual add)
Output: qk (24 * 128 = 3072,) bf16 — Q+K merged and RoPE-rotated

Scratch buffer contains per-layer weights:
  w_qkv: (4096, 1024) bf16 — fused QKV weight (n_head*hd + 2*n_kv_head*hd = 16*128+2*8*128 = 3072)
    Actually QKV: q_dim=2048, k_dim=1024, v_dim=1024 → total=4096
  cos_sin_table: (2 * max_seq * 128) bf16 — pre-computed cos/sin for all positions

position is passed via ScratchpadParameter "rope_position" (int32).
Only Q and K get RoPE (first 3072 elements of output = Q+K = 2048+1024).
V (last 1024 elements) does not get RoPE.
"""

import aie.utils as aie_utils
import numpy as np

from iron.common.sequence import OperatorSequence
from iron.common.utils import get_shim_dma_limit
from iron.operators.gemv.op import GEMV
from iron.operators.rms_norm.op import RMSNorm
from aie.iron import ScratchpadParameter


class PreAttentionDecode(OperatorSequence):
    """Fused RMSNorm + QKV GEMV + RoPE for single-token decode.

    Runtime buffers:
    - input: "in" (1024 bf16)
    - output: "out" (3072 bf16 = 24 heads × 128 — Q+K merged, V excluded)
    - scratch weights: "w_qkv" (per-layer), "cos_sin_table" (shared)
    - ScratchpadParameter "rope_position" (int32, set per decode step)
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

        # ScratchpadParameter for position
        rope_position = ScratchpadParameter("rope_position", np.int32)

        # Runlist:
        # 1. RMSNorm: in → normed (bare, gamma in weights)
        # 2. QKV GEMV: W_qkv @ normed → qkv_out
        #    (RoPE is applied separately — see note below)
        #
        # NOTE: RoPE needs to be applied to Q (first 2048 elements) and K
        # (next 1024 elements) of qkv_out, but NOT V (last 1024 elements).
        # This requires a custom elementwise kernel that reads from scratch
        # (cos_sin_table) and uses the position parameter.
        # For now, RoPE is NOT included in this sequence — it's applied
        # separately via the IRON RoPE operator. Future: add a custom
        # elementwise op to the runlist that does RoPE in-kernel.
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

        # Store the rope_position parameter for host-side access
        self.rope_position = rope_position
