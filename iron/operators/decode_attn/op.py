# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar, Dict

import aie.utils as aie_utils

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
    KernelObjectArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)


@dataclass
class DecodeAttention(MLIROperator):
    """Fused decode-specialised attention (single query, M=1, GEMV semantics).

    Fuses scores-GEMV + softmax + context-GEMV for one query token into a single
    kernel per (query-head, column), keeping scores / softmax probabilities /
    the running context accumulator in L1 (no intermediate DRAM tensors).
    """

    num_heads: int          # H
    num_kv_heads: int       # KV
    head_dim: int           # D
    seq_len_kv: int         # S_KV (key cache length, the maximum)
    num_aie_columns: int = 8
    block_kv: int = 64      # B_KV (block size along S)
    # When True, the causal-mask boundary (seq_pos) is a runtime
    # ScratchpadParameter ("S_kv_eff") written by the host each dispatch,
    # instead of baking seq_len_kv into the kernel. Required for the
    # OperatorSequence (fused full-ELF) path.
    use_runtime_seq_len: bool = field(default=False, repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        "num_heads": "h",
        "num_kv_heads": "kv",
        "head_dim": "d",
        "seq_len_kv": "skv",
        "num_aie_columns": "col",
        "block_kv": "bkv",
    }

    def __post_init__(self):
        if self.seq_len_kv % 64 != 0:
            raise ValueError("seq_len_kv must be a multiple of 64")
        if self.block_kv % 16 != 0:
            raise ValueError("block_kv must be a multiple of 16")
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "decode_attn",
                (aie_utils.get_current_device(),),
                {
                    "cols": self.num_aie_columns,
                    "H": self.num_heads,
                    "KV": self.num_kv_heads,
                    "D": self.head_dim,
                    "S_KV": self.seq_len_kv,
                    "B_KV": self.block_kv,
                    "use_runtime_seq_len": self.use_runtime_seq_len,
                    "verbose": getattr(self.context, "mlir_verbose", False),
                },
            ),
        )

    def get_kernel_artifacts(self):
        return [
            KernelObjectArtifact(
                "decode_attn.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "aie2p" / "decode_attn.cc"
                    )
                ],
                extra_flags=[
                    f"-DD={self.head_dim}",
                    f"-DS_KV={self.seq_len_kv}",
                    f"-DB_KV={self.block_kv}",
                    f"-DVEC=64",
                ],
            )
        ]

    def get_arg_spec(self):
        H, KV, D, S = self.num_heads, self.num_kv_heads, self.head_dim, self.seq_len_kv
        return [
            AIERuntimeArgSpec("in", (H, D)),            # Q
            AIERuntimeArgSpec("in", (KV, 2 * S * D)),   # KV combined (K | V)
            AIERuntimeArgSpec("out", (H, D)),           # O
        ]
