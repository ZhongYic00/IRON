# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Fused Post-Attention operator for decode (S=1).

Fuses: O GEMV + residual add + bare RMSNorm + SwiGLU MLP into a single NPU dispatch.

Input:
  attn_out: (q_dim,) bf16 — attention output (n_head × head_dim)
  x: (embedding_dim,) bf16 — residual hidden state
Output:
  out: (embedding_dim,) bf16 — x + W_o @ attn_out + MLP(RMSNorm(x + W_o @ attn_out))

Scratch weights (per-layer, gamma-folded):
  w_o: O projection weight (embedding_dim × q_dim)
  w_gate / w_up: gate/up weights with post_ln gamma folded in
  w_down: down projection weight
"""

import aie.utils as aie_utils

from iron.common.sequence import OperatorSequence
from iron.common.utils import get_shim_dma_limit
from iron.operators.gemv.op import GEMV
from iron.operators.silu.op import SiLU
from iron.operators.elementwise_mul.op import ElementwiseMul
from iron.operators.elementwise_add.op import ElementwiseAdd
from iron.operators.rms_norm.op import RMSNorm


class PostAttentionDecode(OperatorSequence):
    """Fused O GEMV + residual add + RMSNorm + SwiGLU for single-token decode.

    Replaces separate OGemVAdd + RMSNormSwiGLUDecode (2 dispatches → 1).
    """

    def __init__(self, embedding_dim, hidden_dim, q_dim, epsilon=1e-6, context=None):
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.q_dim = q_dim
        self.epsilon = epsilon

        dev = aie_utils.get_current_device()
        n_cols = get_shim_dma_limit(dev) // 2

        # O GEMV: W_o @ attn_out → o_proj
        gemv_o = GEMV(
            M=embedding_dim,
            K=q_dim,
            num_aie_columns=n_cols,
            tile_size_input=4,
            tile_size_output=embedding_dim // n_cols,
        )

        # Elementwise add: x + o_proj → residual
        add_op = ElementwiseAdd(
            size=embedding_dim,
            num_aie_columns=1,
            tile_size=embedding_dim,
        )

        # Bare RMSNorm (gamma folded into gate/up weights)
        rmsnorm = RMSNorm(
            size=embedding_dim,
            num_aie_columns=1,
            num_channels=1,
            tile_size=embedding_dim,
            weighted=False,
            epsilon=epsilon,
        )

        gemv_1 = GEMV(
            M=self.hidden_dim,
            K=self.embedding_dim,
            num_aie_columns=n_cols,
            tile_size_input=4,
            tile_size_output=self.hidden_dim // n_cols,
        )
        silu = SiLU(
            size=self.hidden_dim,
            num_aie_columns=n_cols,
            tile_size=self.hidden_dim // (n_cols * 2),
        )
        eltwise_mul = ElementwiseMul(
            size=self.hidden_dim,
            num_aie_columns=n_cols,
            tile_size=self.hidden_dim // n_cols,
        )
        gemv_2 = GEMV(
            M=self.embedding_dim,
            K=self.hidden_dim,
            num_aie_columns=n_cols,
            tile_size_input=1,
            tile_size_output=self.embedding_dim // n_cols,
        )

        # Final residual add: residual + mlp_out → out
        add_final = ElementwiseAdd(
            size=embedding_dim,
            num_aie_columns=1,
            tile_size=embedding_dim,
        )

        runlist = [
            # O GEMV: W_o @ attn_out → o_proj
            (gemv_o, "w_o", "attn_out", "o_proj"),
            # residual = x + o_proj
            (add_op, "x", "o_proj", "residual"),
            # Bare RMSNorm: residual → normed
            (rmsnorm, "residual", "normed"),
            # gate GEMV: W_gate' @ normed → left
            (gemv_1, "w_gate", "normed", "left"),
            # up GEMV: W_up' @ normed → right
            (gemv_1, "w_up", "normed", "right"),
            # SiLU(gate) → left_swished
            (silu, "left", "left_swished"),
            # silu(gate) * up → intermediate
            (eltwise_mul, "left_swished", "right", "intermediate"),
            # down GEMV: W_down @ intermediate → mlp_out
            (gemv_2, "w_down", "intermediate", "mlp_out"),
            # Final residual: residual + mlp_out → out
            (add_final, "residual", "mlp_out", "out"),
        ]

        super().__init__(
            name=f"post_attn_decode_e{embedding_dim}_h{hidden_dim}_q{q_dim}",
            runlist=runlist,
            input_args=["attn_out", "x"],
            output_args=["out"],
            context=context,
        )
