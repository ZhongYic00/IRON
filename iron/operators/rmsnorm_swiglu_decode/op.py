# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""RMSNorm + SwiGLU fused decode operator.

Fuses RMSNorm + gate GEMV + SiLU + up GEMV + mul + down GEMV into
a single NPU dispatch. Replaces separate RMSNorm dispatch + SwiGLUDecode.

Computes: W_down @ (SiLU(W_gate @ RMSNorm(x, gamma)) * (W_up @ RMSNorm(x, gamma)))
"""

import aie.utils as aie_utils

from iron.common.sequence import OperatorSequence
from iron.common.utils import get_shim_dma_limit
from iron.operators.gemv.op import GEMV
from iron.operators.silu.op import SiLU
from iron.operators.elementwise_mul.op import ElementwiseMul
from iron.operators.rms_norm.op import RMSNorm


class RMSNormSwiGLUDecode(OperatorSequence):
    """RMSNorm + SwiGLU feed-forward (single-token decode) as one dispatch.

    Runtime buffers (via ``get_callable().get_buffer(name)``):
    input ``in``; persistent weight scratch ``w_rmsnorm`` / ``w_gate`` /
    ``w_up`` / ``w_down``; output ``out``.
    """

    def __init__(self, embedding_dim, hidden_dim, epsilon=1e-6, context=None):
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.epsilon = epsilon

        dev = aie_utils.get_current_device()
        n_cols = get_shim_dma_limit(dev) // 2

        # RMSNorm: x → RMSNorm(x, gamma), size=embedding_dim
        # Use tile_size=embedding_dim so the entire vector fits in one tile.
        # This makes the input buffer size = embedding_dim (1024 bf16),
        # matching what GEMV expects as input.
        rmsnorm = RMSNorm(
            size=embedding_dim,
            num_aie_columns=1,
            num_channels=1,
            tile_size=embedding_dim,  # full vector in one tile
            weighted=True,  # with gamma weight
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

        runlist = [
            # RMSNorm: in → normed (with weight w_rmsnorm)
            (rmsnorm, "w_rmsnorm", "in", "normed"),
            # gate GEMV: W_gate @ normed → left
            (gemv_1, "w_gate", "normed", "left"),
            # up GEMV: W_up @ normed → right
            (gemv_1, "w_up", "normed", "right"),
            # SiLU(gate) → left_swished
            (silu, "left", "left_swished"),
            # silu(gate) * up → intermediate
            (eltwise_mul, "left_swished", "right", "intermediate"),
            # down GEMV: W_down @ intermediate → out
            (gemv_2, "w_down", "intermediate", "out"),
        ]

        super().__init__(
            name=f"rmsnorm_swiglu_decode_e{embedding_dim}_h{hidden_dim}",
            runlist=runlist,
            input_args=["in"],
            output_args=["out"],
            context=context,
        )
