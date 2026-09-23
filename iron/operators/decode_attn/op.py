# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar, Dict
import os

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
    # Batch (multi-query) mode for speculative verify: process `num_new_tokens`
    # new tokens per dispatch in groups of `batch_queries` queries, each group
    # streaming the KV cache once.  Kernels unchanged (flat row index
    # mh = m*heads_per_col + h).  Requires use_runtime_seq_len; the scratchpad
    # parameter becomes "S_kv_base" = cached-token count before the new block.
    # Q/O tensors are (num_new_tokens*H, D) token-major flat.
    batch_queries: int = 0
    num_new_tokens: int = 0
    causal: bool = True
    # Per-token [K_t|V_t] cache layout (DFlash verify): K at 2j*D, V at
    # (2j+1)*D within each head's region, so any append span has constant
    # strides.  Legacy block layout ([K_b|V_b] per B_KV tokens) when False.
    kv_token_interleaved: bool = True   # sc_k/sc_v write per-token [K_t|V_t] interleaved
    # O buffer rows in TOKENS (0 = num_new_tokens): set > num_new_tokens when
    # a downstream consumer (e.g. a GEMM A with M=32) needs a taller buffer —
    # the drains still write only num_new_tokens rows, the pad rows stay zero.
    out_buffer_tokens: int = 0
    # "combined": ONE KV arg, per-head [K half | V half] (StridedCopy append
    # layout).  "separate": TWO args (K cache, V cache), each per-head
    # head-major [KV, S_KV*D] — the xdna-engine qkv_head_dp drain layout. Same
    # inKV fifo, two fills; L1/kernel untouched. M=1 only.
    kv_layout: str = "combined"
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        "num_heads": "h",
        "num_kv_heads": "kv",
        "head_dim": "d",
        "seq_len_kv": "skv",
        "num_aie_columns": "col",
        "block_kv": "bkv",
        "batch_queries": "m",
        "num_new_tokens": "new",
        "kv_layout": "kvl",
    }

    def __post_init__(self):
        if self.seq_len_kv % 64 != 0:
            raise ValueError("seq_len_kv must be a multiple of 64")
        if self.block_kv % 16 != 0:
            raise ValueError("block_kv must be a multiple of 16")
        if self.batch_queries > 0:
            if self.num_new_tokens % self.batch_queries != 0:
                raise ValueError(
                    f"num_new_tokens ({self.num_new_tokens}) must be a multiple "
                    f"of batch_queries ({self.batch_queries})"
                )
            if not self.use_runtime_seq_len:
                raise ValueError("batch mode requires use_runtime_seq_len")
        MLIROperator.__init__(self, context=self.context)

    @property
    def kernel_source(self):
        """SOURCE file of the (single) kernel object this op links.

        Precedence: the context attribute (the selector shape gemv_int8/op.py
        uses) > $DECODE_ATTN_KERNEL_SOURCE (so a chain can A/B a variant without a
        constructor argument) > the shipped scalar file.  The object FILE stays
        `decode_attn.o` in every case, so the generated MLIR is byte-identical
        between arms and cannot be served stale from the fused-MLIR cache (that
        cache does NOT key on op kwargs -- 9/5 note); only the artifact's
        dependency changes, which is what the compile rule keys on.
        """
        return (getattr(self.context, "decode_attn_kernel_source", None)
                or os.environ.get("DECODE_ATTN_KERNEL_SOURCE", "decode_attn.cc"))

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
                    "batch_queries": self.batch_queries,
                    "num_new_tokens": self.num_new_tokens,
                    "causal": self.causal,
                    "out_buffer_tokens": self.out_buffer_tokens,
                    "kv_layout": self.kv_layout,
                    "verbose": getattr(self.context, "mlir_verbose", False),
                    "trace_size": int(os.environ.get("IRON_TRACE_SIZE", "0")),
                },
            ),
        )

    def get_kernel_artifacts(self):
        # decode_attn_kernel_source / $DECODE_ATTN_KERNEL_SOURCE: which SOURCE file
        # the one kernel object is built from (see kernel_source).  Variants live as
        # separate translation units -- no #if gymnastics, per the 2026-08-31
        # build-pollution incident -- while the object keeps the name the MLIR
        # links against.  Default = the shipped scalar file, so nothing changes
        # unless a caller opts in.
        src = self.kernel_source
        assert src.endswith(".cc"), src
        flags = [
            f"-DD={self.head_dim}",
            f"-DS_KV={self.seq_len_kv}",
            f"-DB_KV={self.block_kv}",
            f"-DVEC=64",
        ]
        if self.kv_token_interleaved:
            flags.append("-DKV_TOKEN_INTERLEAVED")
        return [
            KernelObjectArtifact(
                # fixed name: the MLIR's link_with is "decode_attn.o" regardless of
                # which source the selector picked (see kernel_source)
                "decode_attn.o",
                dependencies=[
                    SourceArtifact(
                        self.context.base_dir / "aie_kernels" / "aie2p" / src
                    )
                ],
                extra_flags=flags,
            )
        ]

    def get_arg_spec(self):
        H, KV, D, S = self.num_heads, self.num_kv_heads, self.head_dim, self.seq_len_kv
        # Batch mode: Q covers num_new_tokens token-major rows; O may be a
        # taller buffer (out_buffer_tokens) for downstream GEMM A shapes.
        q_rows = self.num_new_tokens * H if self.batch_queries else H
        o_tokens = max(self.out_buffer_tokens, self.num_new_tokens) \
            if self.batch_queries else 1
        if self.kv_layout == "separate":
            return [
                AIERuntimeArgSpec("in", (q_rows, D)),       # Q
                AIERuntimeArgSpec("in", (KV, S * D)),       # K cache (head-major)
                AIERuntimeArgSpec("in", (KV, S * D)),       # V cache (head-major)
                AIERuntimeArgSpec("out", (o_tokens * H, D)),  # O
            ]
        return [
            AIERuntimeArgSpec("in", (q_rows, D)),       # Q
            AIERuntimeArgSpec("in", (KV, 2 * S * D)),   # KV combined (K | V)
            AIERuntimeArgSpec("out", (o_tokens * H, D)),  # O
        ]
