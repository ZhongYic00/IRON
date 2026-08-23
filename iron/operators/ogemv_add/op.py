# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused O projection GEMV + residual add for decode (S=1).

Input:
  attn_out: (q_dim,) bf16 — attention output (Q heads × head_dim)
  x: (embedding_dim,) bf16 — residual hidden state
Output:
  out: (embedding_dim,) bf16 — x + W_o @ attn_out

Scratch weight: w_o (embedding_dim × q_dim) bf16 — O projection weight.
"""

import aie.utils as aie_utils

from iron.common.sequence import OperatorSequence
from iron.common.utils import get_shim_dma_limit
from iron.operators.gemv.op import GEMV
from iron.operators.elementwise_add.op import ElementwiseAdd


class OGemVAdd(OperatorSequence):
    """Fused O GEMV + residual add: out = x + W_o @ attn_out."""

    def __init__(self, embedding_dim, q_dim, context=None):
        self.embedding_dim = embedding_dim
        self.q_dim = q_dim

        dev = aie_utils.get_current_device()
        n_cols = get_shim_dma_limit(dev) // 2

        gemv_o = GEMV(
            M=embedding_dim,
            K=q_dim,
            num_aie_columns=n_cols,
            tile_size_input=4,
            tile_size_output=embedding_dim // n_cols,
        )

        add_op = ElementwiseAdd(
            size=embedding_dim,
            num_aie_columns=1,
            tile_size=embedding_dim,
        )

        runlist = [
            # O GEMV: W_o @ attn_out → o_proj
            (gemv_o, "w_o", "attn_out", "o_proj"),
            # Residual add: x + o_proj → out
            (add_op, "x", "o_proj", "out"),
        ]

        super().__init__(
            name=f"ogemv_add_e{embedding_dim}_q{q_dim}",
            runlist=runlist,
            input_args=["attn_out", "x"],
            output_args=["out"],
            context=context,
        )
