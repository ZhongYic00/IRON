# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field
from typing import ClassVar, Dict

from iron.common import (
    MLIROperator,
    AIERuntimeArgSpec,
    KernelObjectArtifact,
    KernelArchiveArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)
import aie.utils as aie_utils
from iron.common.device_utils import get_kernel_dir


@dataclass
class QKVHeadDataParallel(MLIROperator):
    """Decode QKV head as ONE `aie.device`, data-parallel across `num_aie_columns` cores.

    Fuses RMSNorm(in) -> concatenated [Hq*HD + 2*Hkv*HD, D] QKV GEMV -> per-head qk-RMSNorm ->
    RoPE(q, k). Every core owns a contiguous row slice of the concatenated weight and runs every
    stage on it; see design.py for why that shape and not fuse/qkv-head's spatial one.

    Runtime interface: cur, n_in, Wqkv, n_qn, n_kn, ang -> qkv, where `qkv` is the concatenated
    [q | k | v] the caller then slices. The weight and the output are concatenated because the
    caller's graph already concatenates them (gen_llm_decode.py's FUSE_QKV_GEMV), not to save an
    argument.
    """

    D: int
    HD: int
    Hq: int
    Hkv: int
    max_seq: int
    num_aie_columns: int = 8
    epsilon: float = 1e-6
    tile_size_input: int = 4
    stack_size: int = 0xD00
    kv_offset_parameter: str | None = "kv_off"
    # Weight ObjectFifo depth; the L1 budget check in design.py follows it.
    weight_depth: int = field(default=2, repr=False)
    # True = one fill per HD object (the 0.6B arm's dataflow); False = one fill per
    # D-wide tensor (one BD, one host round trip -- see design_ours_kvlayout.py).
    misc_chunked: bool = field(default=True, repr=False)
    # Misc object width in elements; None = HD (the 0.6B arm's dataflow, and the
    # per-chunk fills that satisfy aiecc's 16-active-BD-per-tile rule at D=2560).
    # Setting it (the 4B chain uses D/5) widens the object so cur/n_in need far
    # fewer fills AND enables `misc_chunked=False`.
    misc_obj_elems: int | None = field(default=None, repr=False)
    # See design_ours_kvlayout.py: a constant per-head tile loop is fully unrolled
    # (~143 B per mv site), which is what overflows the 16 KB program memory at
    # tsi=8/D=2560.  True reads the bound from an L1 word instead.
    runtime_tile_loop: bool = field(default=False, repr=False)
    # INT8_SIGNED: which SOURCE the int8 matvec object is built from.  `None` = the shipped
    # biased-domain kernel (mv_int8.cc, payload = q+128, kernel does `unpack -> to_float -> add(-128)`).
    # "mv_int8_signed.cc" reads the payload as SIGNED int8 and drops the per-weight add -- the
    # packer must store the signed bytes together with it (the chain's `_set_int8_awq(..., signed=)`).
    # PER-INSTANCE on purpose (not a context attribute): the wire domain is a property of the weight
    # bytes this instance is fed, so it must never be settable for a different instance.
    kernel_source: str | None = field(default=None, repr=False)
    # M1 prerequisite: assign each column the head set decode_attn needs from it
    # (q_2c, q_2c+1, k_c, v_c) instead of a contiguous run of the concatenated [Q|K|V] weight, with
    # the host permuting Wqkv's rows to match (probe_m1_qkv_perm.py). Off = shipped behaviour.
    #
    # repr=True (i.e. part of `name`) is load-bearing, not cosmetic: the fused-MLIR/artifact cache
    # keys on the op NAME plus the generator source hash, so two arms of the same source that differ
    # only in a repr=False kwarg would silently share one another's artifacts -- the trap this tree
    # has already been bitten by ("new knobs must appear in the op name").
    attn_colmajor: bool = field(default=False)
    # M1: also run decode_attn's four kernels from THIS design (channel folding -- see
    # design_ours_kvlayout.py's "M1 channel folding" note).  Off = shipped behaviour.
    with_attn: bool = field(default=False)
    # M2: run decode_attn's four kernels on their OWN worker, one phase row below the qkv row, with
    # q handed over LIVE on the out fifo (design_m2_layer.py).  Same kernels as `with_attn`, unlike
    # it no channel folding: the constants keep riding the misc channel, so `cur` stays live.
    # repr=True on purpose -- see the `attn_colmajor` note above about the artifact cache key.
    attn_row3: bool = field(default=False)
    # M2: add the THIRD phase row -- swiglu_mlp_dp's fuse_o core (o-projection + SwiGLU MLP) as
    # its own worker on row 4 of every column, taking the attention's context and the layer's
    # residual and producing the next residual (design_m2_layer.py).  Needs `attn_row3` (row 4's
    # context IS row 3's output) and NOT `with_attn` (that arm folds the two phases onto one core,
    # which leaves no row 4).  repr=True on purpose -- see the `attn_colmajor` note above about
    # the artifact cache key.
    mlp_row4: bool = field(default=False)
    # Write the weighted RMSNorm over n_in instead of a third D-wide buffer (n_in is
    # dead after that step).  Saves 5 KB of core L1 -- what a D/2 misc object costs.
    inplace_norm: bool = field(default=False, repr=False)
    # Stage the misc channel through a MemTile: the shim side then works in D-wide
    # objects (one fill/one BD per tensor) while the core side keeps the MO-wide
    # objects.  See design_ours_kvlayout.py.
    misc_memtile: bool = field(default=False, repr=False)
    # Merge the per-head output drains into one drain per KIND per core (48 -> 10 tasks/layer at
    # 4B).  repr=True on purpose: MLIROperator.name skips repr=False fields, and the fused-MLIR
    # cache keys on the artifact NAME -- a repr=False knob would let two arms share one name.
    merge_drains: bool = field(default=False)
    context: object = field(default=None, repr=False)

    _name_aliases: ClassVar[Dict[str, str]] = {
        **MLIROperator._name_aliases,
        "num_aie_columns": "col",
        "epsilon": "eps",
        "tile_size_input": "tsi",
        "stack_size": "ss",
        "max_seq": "S",
        "kv_offset_parameter": "kvpar",
        "weight_depth": "wd",
        "misc_chunked": "mc",
        "misc_obj_elems": "mo",
        "runtime_tile_loop": "rtl",
        "attn_colmajor": "acm",
        "with_attn": "wa",
        "inplace_norm": "ipn",
        "misc_memtile": "mmt",
        "merge_drains": "md",
        "attn_row3": "ar3",
        "mlp_row4": "mr4",
    }

    def __post_init__(self):
        heads = self.Hq + 2 * self.Hkv
        if heads % self.num_aie_columns:
            raise ValueError(
                f"Hq + 2*Hkv ({heads}) must be divisible by num_aie_columns "
                f"({self.num_aie_columns}) -- every core owns a whole number of heads"
            )
        if self.HD % self.tile_size_input:
            raise ValueError(
                f"head_dim ({self.HD}) must be divisible by tile_size_input "
                f"({self.tile_size_input})"
            )
        if self.D % self.HD:
            raise ValueError(
                f"d_model ({self.D}) must be a whole number of head_dim ({self.HD}) chunks -- "
                "`cur` and `n_in` ride the HD-wide misc channel"
            )
        MLIROperator.__init__(self, context=self.context)

    def get_mlir_artifact(self):
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design.py",
                "qkv_head_dp",
                (aie_utils.get_current_device(), self.D, self.HD, self.Hq, self.Hkv,
                 self.max_seq),
                {
                    "epsilon": self.epsilon,
                    "kv_offset_parameter": self.kv_offset_parameter,
                    "weight_depth": self.weight_depth,
                    "tile_size_input": self.tile_size_input,
                    "stack_size": self.stack_size,
                    "n_aie_cols": self.num_aie_columns,
                },
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        kdir = self.context.base_dir / "aie_kernels"
        copy_obj = KernelObjectArtifact(
            "add.o", dependencies=[SourceArtifact(kdir / "generic" / "add.cc")]
        )
        rms_obj = KernelObjectArtifact(
            f"rms_norm_{self.D}.o",
            dependencies=[SourceArtifact(kdir / arch_dir / "rms_norm.cc")],
            extra_flags=[f"-DRMS_COLS={self.D}"],
        )
        # Same source, second symbol: this core calls weighted_rms_norm at D and at HD, and one
        # Kernel() binding fixes one signature per symbol. Prefixing a second object is what
        # swiglu_mlp_dp does for its two matvec DIM_Ks; the alternative -- a local copy of the
        # vendored kernel under two names -- is the duplication one-kernel-three-repos warns about.
        rms_hd_obj = KernelObjectArtifact(
            f"hd_rms_norm_{self.HD}.o",
            dependencies=[SourceArtifact(kdir / arch_dir / "rms_norm.cc")],
            extra_flags=[f"-DRMS_COLS={self.HD}"],
            prefix_symbols="hd_",
        )
        mv_obj = KernelObjectArtifact(
            f"gemv_{self.D}k_64vs.o",
            dependencies=[SourceArtifact(kdir / arch_dir / "mv_int8.cc")],
            extra_flags=[f"-DDIM_K={self.D}", "-DVEC_SIZE=128", "-DGROUP_SIZE=128"],
        )
        rope_obj = KernelObjectArtifact(
            "rope_0.o",
            dependencies=[SourceArtifact(kdir / "generic" / "rope.cc")],
            extra_flags=["-DTWO_HALVES"],
        )
        return [
            KernelArchiveArtifact(
                "qkv_head_dp_core.a",
                dependencies=[copy_obj, rms_obj, rms_hd_obj, mv_obj, rope_obj],
            )
        ]

    def get_arg_spec(self):
        QD, KVD = self.Hq * self.HD, self.Hkv * self.HD
        cache = self.Hkv * self.max_seq * self.HD
        # int8 wire: per row-block of tile_m rows,
        # [tile_m*D int8 | tile_m*(D/128)*2 B scales], column-major over M
        # (column c owns rows [c*M/cols, +M/cols)); viewed as bf16 elements.
        # (mv_int8.cc's [m*K u8 | m*G bf16 scales] tile, M = QD + 2*KVD.)
        tile_m = self.tile_size_input
        n_blocks = (QD + 2 * KVD) // tile_m
        wqkv_bytes = n_blocks * (tile_m * self.D + tile_m * (self.D // 128) * 2)
        return [
            AIERuntimeArgSpec("in", (self.D,)),                    # cur
            AIERuntimeArgSpec("in", (self.D,)),                    # n_in
            AIERuntimeArgSpec("in", (wqkv_bytes // 2,)),           # Wqkv int8 wire (bf16 view)
            # The three constants are MISC OBJECTS (misc_obj_elems or HD); the
            # kernels read their first HD elements, the host pads them.
            AIERuntimeArgSpec("in", ((self.misc_obj_elems or self.HD),)),  # n_qn
            AIERuntimeArgSpec("in", ((self.misc_obj_elems or self.HD),)),  # n_kn
            AIERuntimeArgSpec("in", ((self.misc_obj_elems or self.HD),)),  # ang
            AIERuntimeArgSpec("out", (QD,)),                       # q
            # ONE merged KV cache, matching design.py's drain: head-major
            # [K_t | V_t] layout — head hh owns [hh*2*max_seq*HD, +2*max_seq*HD),
            # token t's K at t*2*HD and V at t*2*HD + HD (relative to kv_off).
            AIERuntimeArgSpec("inout", (2 * cache,)),              # merged kv cache
        ]

    def reference(self, cur, n_in, wqkv, n_qn, n_kn, ang):
        """Returns the concatenated [q | k | v]; the caller places k and v itself. The device
        appends them to the caches directly, so there is no single output to compare against."""
        from iron.operators.qkv_head_dp.reference import reference

        return reference(cur, n_in, wqkv, n_qn, n_kn, ang,
                         self.D, self.HD, self.Hq, self.Hkv, self.epsilon)
