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

    kv_direct=True (opt-in; `QKV_KV_DIRECT` in the 4B chain) adds a FOURTH
    runtime arg — the layer's whole KV cache — and has the producer write the
    4 KB K|V slice straight into it, replacing the chain's merged StridedCopy
    (`sc_kv`) runlist entry: one fewer op boundary / array reconfigure per
    layer.  The slice is written in the exact order sc_kv read it
    ([K head | K head | ... | V head | ...]) and the design drains it with
    sc_kv's own 3D tap plus the `k_cache_offset` scratchpad parameter, so the
    cache bytes — and therefore the layout decode_attn reads — do not change.
    """

    qkv_dim: int       # 4096 = 24*128 + 8*128
    head_dim: int       # 128
    n_qk_heads: int    # 24 (16 Q + 8 K)
    n_v_heads: int      # 8
    epsilon: float = 1e-6
    num_aie_columns: int = 1
    kv_direct: bool = False
    max_seq_len: int = 512
    kv_offset_parameter: str = field(default="k_cache_offset", repr=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "epsilon": "eps",
        "kv_direct": "kvdir",
        "max_seq_len": "seq",
    }

    def __post_init__(self):
        MLIROperator.__init__(self, context=self.context)

    @property
    def kv_cache_elems(self) -> int:
        """Elements in one layer's KV cache: head-major, per-token [K_t|V_t]."""
        return self.n_v_heads * 2 * self.max_seq_len * self.head_dim

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
                {
                    "kv_direct": self.kv_direct,
                    "max_seq_len": self.max_seq_len,
                    "kv_offset_parameter": self.kv_offset_parameter,
                },
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        source = "qk_norm_rope_kv.cc" if self.kv_direct else "qk_norm_rope.cc"
        return [
            KernelObjectArtifact(
                source.replace(".cc", ".o"),
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / arch_dir / source
                    )
                ],
            ),
        ]

    def get_arg_spec(self):
        gamma_sz = self.n_qk_heads * self.head_dim
        cos_sin_sz = 2 * self.head_dim
        scratch_sz = gamma_sz + cos_sin_sz
        spec = [
            AIERuntimeArgSpec("in", (self.qkv_dim,)),           # qkv_in
            AIERuntimeArgSpec("out", (self.qkv_dim,)),           # qkv_out
            AIERuntimeArgSpec("in", (scratch_sz,)),              # merged scratch
        ]
        if self.kv_direct:
            # Same buffer (and same length) decode_attn reads.
            spec.append(AIERuntimeArgSpec("inout", (self.kv_cache_elems,)))
        return spec
