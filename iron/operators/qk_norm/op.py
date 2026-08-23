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
class QKNorm(MLIROperator):
    """Qwen3 Q/K per-head RMSNorm (no RoPE, no V) for decode (S=1).

    Input:  qk_in  (n_q_heads * head_dim + n_k_heads * head_dim,) bf16 — flat [Q | K].
    Output: qk_out (same shape,) bf16 — per-head RMSNorm with shared per-group gamma.

    Scratch: [q_gamma(head_dim) | k_gamma(head_dim)] bf16.
      q_gamma is shared across all n_q_heads Q heads; k_gamma across n_k_heads K heads.
      The RMS statistic is computed per head.

    RoPE is applied separately via the standard iron RoPE operator; V is copied
    directly to the KV cache (no norm).  This keeps the operator faithful to the
    llama_npu.py blueprint (independent RoPE + per-token angle buffer) while
    adding only Qwen3's QK-norm.
    """

    head_dim: int          # 128
    n_q_heads: int         # 16
    n_k_heads: int         # 8
    epsilon: float = 1e-6
    num_aie_columns: int = 1
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "epsilon": "eps",
    }

    @property
    def n_qk_heads(self) -> int:
        return self.n_q_heads + self.n_k_heads

    @property
    def size(self) -> int:
        return self.n_qk_heads * self.head_dim

    def __post_init__(self):
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "qk_norm",
                (
                    aie_utils.get_current_device(),
                    self.head_dim,
                    self.n_q_heads,
                    self.n_k_heads,
                    self.epsilon,
                    0,  # trace_size
                ),
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        return [
            KernelObjectArtifact(
                "qk_norm.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / arch_dir / "qk_norm.cc"
                    )
                ],
            ),
        ]

    def get_arg_spec(self):
        gamma_sz = 2 * self.head_dim
        return [
            AIERuntimeArgSpec("in", (self.size,)),           # qk_in
            AIERuntimeArgSpec("out", (self.size,)),          # qk_out
            AIERuntimeArgSpec("in", (gamma_sz,)),            # merged [q_gamma | k_gamma]
        ]

    def reference(self, x, gamma):
        from iron.operators.qk_norm.reference import reference

        return reference(
            x, gamma, self.head_dim, self.n_q_heads, self.n_k_heads, self.epsilon
        )
