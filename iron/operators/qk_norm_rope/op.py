# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar, Dict

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)
import aie.utils as aie_utils
from iron.common.device_utils import get_kernel_dir


@dataclass
class QKNormRoPE(MLIROperator):
    """Fused QK-norm + RoPE + V-copy for decode (S=1).

    Input: qkv_in (qkv_dim,) bf16 — flat QKV from GEMV.
    Output: qkv_out (qkv_dim,) bf16 — Q and K normalized + RoPE-rotated, V copied.

    Scratch args:
      scratch: (n_qk_heads * head_dim + 2 * head_dim,) bf16
        = [qk_gamma | cos(head_dim) | sin(head_dim)]
        Merged into one buffer to fit 2 MM2S DMA channels.
    """

    qkv_dim: int       # 4096 = 24*128 + 8*128
    head_dim: int       # 128
    n_qk_heads: int    # 24 (16 Q + 8 K)
    n_v_heads: int      # 8
    epsilon: float = 1e-6
    num_aie_columns: int = 1
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "epsilon": "eps",
    }

    def __post_init__(self):
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "my_qk_norm_rope",
                (
                    aie_utils.get_current_device(),
                    self.qkv_dim,
                    self.head_dim,
                    self.n_qk_heads,
                    self.n_v_heads,
                    self.epsilon,
                    0,  # trace_size
                ),
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        return [
            KernelObjectArtifact(
                "qk_norm_rope.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / arch_dir / "qk_norm_rope.cc"
                    )
                ],
            ),
        ]

    def get_arg_spec(self):
        gamma_sz = self.n_qk_heads * self.head_dim
        cos_sin_sz = 2 * self.head_dim
        scratch_sz = gamma_sz + cos_sin_sz
        return [
            AIERuntimeArgSpec("in", (self.qkv_dim,)),           # qkv_in
            AIERuntimeArgSpec("out", (self.qkv_dim,)),           # qkv_out
            AIERuntimeArgSpec("in", (scratch_sz,)),              # merged scratch
        ]
