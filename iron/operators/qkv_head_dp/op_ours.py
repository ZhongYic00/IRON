# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OUR kv-layout arm of their QKVHeadDataParallel: same fused head (RMSNorm +
QKV GEMV + per-head qk-norm + RoPE), but K/V drain into ONE interleaved
per-token cache (the qwen3 chain's layout: token t's K at kv_off + h*HD, V at
kv_off + HD + h*HD, kv_off = t*2*HD) so the shipped decode_attn reads its
combined tap unchanged. design_ours_kvlayout.py; 8 args (one kv_cache).

Ported to Qwen3-4B decode shapes (D=2560, tsi=16, weight_depth=1) for the 4B
int8 chain's `QKV_HEAD_DP` gate; the 0.6B arm (D=1024, tsi=4, depth=2) is the
same code path.  See docs/notes/2026-09-13-4b-qkv-head-dp-port.md for why the 4B
shapes have to take tsi=16/depth=1 (the 16 KB program memory, not L1).
"""

import os
from dataclasses import dataclass, field
from typing import ClassVar, Dict

import numpy as np

import aie.utils as aie_utils

from iron.common import (AIERuntimeArgSpec, KernelObjectArtifact,
                         KernelArchiveArtifact, SourceArtifact,
                         PythonGeneratedMLIRArtifact, DesignGenerator)
from iron.common.device_utils import get_kernel_dir
from iron.operators.qkv_head_dp.op import QKVHeadDataParallel


@dataclass
class QKVHeadDataParallelOurs(QKVHeadDataParallel):
    """Same surface except k/v caches collapse into one interleaved buffer."""

    _name_aliases: ClassVar[Dict[str, str]] = {
        **QKVHeadDataParallel._name_aliases,
        "kv_layout_ours": "kvlo",
    }

    def get_mlir_artifact(self):
        if self.attn_row3 or self.mlp_row4:
            # Loud, not silent.  `attn_row3`/`mlp_row4` exist only in design_m2_layer.py (M2's
            # phase rows); forwarding them here is a TypeError from the shipped generator (and
            # would be worse than that if the generator took **kwargs: it would silently fall
            # back to a design with no attention half).  The M2 arms are `M2LayerOp` in
            # iron/applications/qwen3_0.6b/probe_m2_layer.py (row 2+3) and `M2Row4Op` in
            # probe_m2_row4.py (rows 2+3+4).
            raise ValueError(
                "attn_row3/mlp_row4 require the M2 layer design (design_m2_layer.py); the shipped "
                "design_ours_kvlayout.py does not implement them")
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design_ours_kvlayout.py",
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
                    "misc_chunked": self.misc_chunked,
                    "misc_obj_elems": self.misc_obj_elems,
                    "runtime_tile_loop": self.runtime_tile_loop,
                    "attn_colmajor": self.attn_colmajor,
                    "with_attn": self.with_attn,
                    "B_KV": 32,   # matches the chain's BLOCK_KV default
                    "inplace_norm": self.inplace_norm,
                    "misc_memtile": self.misc_memtile,
                    "merge_drains": self.merge_drains,
                },
            ),
        )

    def get_kernel_artifacts(self):
        # parent's artifact set with the bf16 mv.cc swapped for OUR int8
        # matvec (mode14 premul + VEC128); everything else identical.
        arch_dir = get_kernel_dir()
        kdir = self.context.base_dir / "aie_kernels"
        from iron.common import KernelObjectArtifact, KernelArchiveArtifact, SourceArtifact
        OPT = ["-Os"]  # program memory: 5-kernel core + the DIM_K=2560 mv loop
        # VEC_SIZE: the 0.6B arm was gated at 64, and 64 stays the DEFAULT here so
        # that arm's compile command is byte-identical to what was measured.  The
        # 4B chain sets QKV_DP_VEC=128 (the production setting elsewhere in this
        # tree: GEMVInt8 and swiglu_mlp_dp's int8_ours arm both compile mv_int8.cc
        # at DIM_K=2560 with VEC_SIZE=128).  NOTE the vector width is NOT in the
        # .o filename, so an override needs a fresh build dir.
        VEC = int(os.environ.get("QKV_DP_VEC", "64"))
        # PROBE: QKV_DP_NOOP=rms|rope swaps in an EMPTY kernel body so the op can be
        # timed without that compute (numbers from such a build are timing-only; the
        # design's self-test deliberately fails on them).
        _noop = os.environ.get("QKV_DP_NOOP", "")
        rms_src = "rms_norm_probe_noop.cc" if "rms" in _noop else "rms_norm.cc"
        rope_src = "rope_probe_noop.cc" if "rope" in _noop else "rope.cc"
        deps = [
            KernelObjectArtifact("add.o", dependencies=[SourceArtifact(kdir / "generic" / "add.cc")],
                                 extra_flags=OPT),
            KernelObjectArtifact(
                f"rms_norm_{self.D}.o",
                dependencies=[SourceArtifact(kdir / arch_dir / rms_src)],
                # RMS_COLS_ONLY: this core only ever calls the fixed-shape entry, so the
                # runtime-cols forms (and the __divsf3/__floatsisf they link) stay out.
                extra_flags=[f"-DRMS_COLS={self.D}", "-DRMS_COLS_ONLY"] + OPT,
            ),
            KernelObjectArtifact(
                f"hd_rms_norm_{self.HD}.o",
                dependencies=[SourceArtifact(kdir / arch_dir / rms_src)],
                extra_flags=[f"-DRMS_COLS={self.HD}", "-DRMS_COLS_ONLY"] + OPT,
                prefix_symbols="hd_",
            ),
            KernelObjectArtifact(
                f"gemv_{self.D}k_128vs_ours.o",
                dependencies=[SourceArtifact(kdir / arch_dir /
                                             (self.kernel_source or "mv_int8.cc"))],
                extra_flags=[f"-DDIM_K={self.D}", f"-DVEC_SIZE={VEC}",
                             "-DGROUP_SIZE=128"] + OPT,
            ),
            KernelObjectArtifact(
                "rope_0.o",
                dependencies=[SourceArtifact(kdir / "generic" / rope_src)],
                extra_flags=["-DTWO_HALVES"] + OPT,
            ),
        ]
        if self.with_attn or self.attn_row3:
            # M1: the merged op also carries decode_attn's four kernels (one object, four
            # functions), built with the SAME flags the attn op uses so the merged design's
            # symbols and shapes match what decode_attn/design.py declares.  Linking it here is
            # what makes the merged .text a MEASURED number instead of the sum of two designs'.
            attn_flags = [f"-DD={self.HD}", f"-DS_KV={self.max_seq}", "-DB_KV=32", "-DVEC=64",
                          "-DKV_TOKEN_INTERLEAVED"]
            # NOTE: returned as its OWN artifact, NOT inside the archive: the design's four
            # attention Kernels reference it by name, and an object that is both expanded from the
            # archive and linked standalone comes out as `duplicate symbol: attn_scores_block`.
            # decode_attn/op.py returns exactly this shape for the same reason.
            attn_obj = KernelObjectArtifact(
                "decode_attn.o",
                dependencies=[SourceArtifact(kdir / arch_dir / "decode_attn.cc")],
                extra_flags=attn_flags,
            )
            # ---- channel folding (see design_ours_kvlayout.py's "M1 channel folding" note):
            # three call sites need `copy_offset_bf16_vector` at shapes the archive's own binding
            # does not cover -- (D, WTILE), (HD, WTILE) for the constants that now arrive on the
            # weight channel, and (A_Q_ty, HD) for the q->L1 handoff.  A func.func symbol is keyed
            # by NAME, so each shape gets its OWN renamed object (prefix_symbols), exactly like
            # swiglu_mlp_dp's cx_copy_obj / oa_copy_obj.
            def _add_copy_obj(obj_name, prefix):
                return KernelObjectArtifact(
                    obj_name,
                    dependencies=[SourceArtifact(kdir / "generic" / "add.cc")],
                    extra_flags=OPT, prefix_symbols=prefix,
                )

            # wt_d/wt_hd are the CONSTANT FOLDING's call shapes: only the with_attn arm moves the
            # constants onto the weight channel.  q_copy is needed whenever the attn half runs --
            # under M2 it is how the live q object from row 2 lands in row 3's q buffer.
            copy_objs = [_add_copy_obj("q_copy.o", "qc_")]
            if self.with_attn:
                copy_objs += [_add_copy_obj("wt_d_copy.o", "wtd_"),
                              _add_copy_obj("wt_hd_copy.o", "wth_")]
            if self.mlp_row4:
                # ---- M2 ROW 4: swiglu_mlp_dp's fuse_o kernel set, recompiled under OWN symbol
                # prefixes.  A func.func symbol is keyed by NAME only, and the qkv row already
                # declares `matvec_vectorized_int8_bf16` and `copy_offset_bf16_vector` at other
                # signatures -- so every MLP call shape needs its own renamed object, the same
                # mechanism swiglu_mlp_dp's own archive uses for its three mv DIM_Ks (madd_/
                # mmul_/mcx_/moa_/mgu_/mdown_/mo_ are exactly its down_/o_/cx_/oa_ prefixes).
                # The gate/up mv is compiled at VEC_SIZE=128 (the production setting: the qkv row
                # runs 64), so it cannot share the qkv row's object either.
                g = self._mlp_row4_geom()
                mv_flags = ["-DVEC_SIZE=128", "-DGROUP_SIZE=128"] + OPT

                def _add_mlp_obj(obj_name, src, prefix, flags):
                    return KernelObjectArtifact(obj_name, dependencies=[SourceArtifact(src)],
                                                extra_flags=flags, prefix_symbols=prefix)

                # `weighted_rms_norm_fixed` and the (D, D) `copy_offset_bf16_vector` are NOT
                # rebuilt: the qkv archive already carries them at byte-identical signatures
                # (its rms_norm_D.o is -DRMS_COLS=D, the same fixed entry the MLP calls), and a
                # second copy would be a duplicate symbol.
                deps += [
                    _add_mlp_obj("mlp_add.o", kdir / "generic" / "add.cc", "madd_", OPT),
                    _add_mlp_obj("mlp_mul.o", kdir / "generic" / "mul.cc", "mmul_", OPT),
                    _add_mlp_obj("mlp_cx_copy.o", kdir / "generic" / "add.cc", "mcx_", OPT),
                    _add_mlp_obj("mlp_oa_copy.o", kdir / "generic" / "add.cc", "moa_", OPT),
                    _add_mlp_obj("mlp_silu.o", kdir / arch_dir / "silu.cc", None, OPT),
                    _add_mlp_obj("mlp_gu_gemv.o", kdir / arch_dir / "mv_int8.cc", "mgu_",
                                 [f"-DDIM_K={self.D}"] + mv_flags),
                    _add_mlp_obj("mlp_down_gemv.o", kdir / arch_dir / "mv_int8.cc", "mdown_",
                                 [f"-DDIM_K={g['ff']}"] + mv_flags),
                    _add_mlp_obj("mlp_o_gemv.o", kdir / arch_dir / "mv_int8.cc", "mo_",
                                 [f"-DDIM_K={self.Hq * self.HD}"] + mv_flags),
                ]
            return [KernelArchiveArtifact("qkv_head_dp_core.a", dependencies=deps), attn_obj,
                    *copy_objs]
        return [KernelArchiveArtifact("qkv_head_dp_core.a", dependencies=deps)]

    def _mlp_row4_geom(self):
        """ROW 4's geometry, from the ONE implementation of it (design_m2_layer.py's
        `mlp_row4_geometry`) -- the arg spec and the design's Runtime must size the same buffers,
        and they must agree on `ff` too (None there = 3*D, which this mirrors by passing None)."""
        from iron.operators.qkv_head_dp.design_m2_layer import mlp_row4_geometry
        assert self.mlp_row4 and self.attn_row3, "the row-4 surface needs attn_row3 as well"
        # n_aie_rows is always 1 for this design (the M2 layer is one row per phase); the qkv op
        # has no such field, unlike swiglu_mlp_dp's.
        return mlp_row4_geometry(self.D, self.Hq * self.HD, self.num_aie_columns, None)

    def get_arg_spec(self):
        if self.with_attn:
            # M1 surface: the five constants are declared at the WEIGHT OBJECT width (the tap's
            # bound check is against the declared tensor length, so a fill into a WTILE-wide fifo
            # object needs a WTILE-wide source), the q output stays in the surface but is unused
            # (q rides L1), and the attention half's context is appended -- the same trailing
            # "out" decode_attn itself declares.
            w = self.tile_size_input * (self.D + (self.D // 128) * 2) // 2   # WTILE_ELEMS
            wqkv = (self.Hq + 2 * self.Hkv) * self.HD * (self.D + (self.D // 128) * 2) // 2
            return [
                AIERuntimeArgSpec("in", (w,)),                     # x_pad (cur, zero-padded)
                AIERuntimeArgSpec("in", (w,)),                     # W_norm1 (padded)
                AIERuntimeArgSpec("in", (wqkv,)),                  # Wqkv int8 wire
                AIERuntimeArgSpec("in", (w,)),                     # W_qn (padded)
                AIERuntimeArgSpec("in", (w,)),                     # W_kn (padded)
                AIERuntimeArgSpec("in", (w,)),                     # rope_angles (padded)
                AIERuntimeArgSpec("out", (self.Hq * self.HD,)),    # q (unused under M1)
                AIERuntimeArgSpec("inout", (self.Hkv * self.max_seq * self.HD * 2,)),  # kv_cache
                AIERuntimeArgSpec("out", (self.Hq * self.HD,)),    # context
            ]
        g = self._mlp_row4_geom() if self.mlp_row4 else None
        return [
            AIERuntimeArgSpec("in", (self.D,)),                    # cur
            AIERuntimeArgSpec("in", (self.D,)),                    # n_in
            AIERuntimeArgSpec("in", ((self.Hq + 2 * self.Hkv) * self.HD * (self.D + (self.D // 128) * 2) // 2,)),  # Wqkv int8 wire (bf16-elem view)
            # The constants: three MISC OBJECTS (misc_obj_elems or HD wide), or ONE
            # packed D-wide object when a MemTile stages the shim side.  The kernels
            # read their first HD elements of each piece; the host pads.
            *([AIERuntimeArgSpec("in", (self.D,))] if self.misc_memtile else
              [AIERuntimeArgSpec("in", ((self.misc_obj_elems or self.HD),))] * 3),
            AIERuntimeArgSpec("out", (self.Hq * self.HD,)),        # q
            AIERuntimeArgSpec("inout", (self.Hkv * self.max_seq * self.HD * 2,)),  # kv_cache (K+V interleaved)
            # M1: the attention half's context output (all heads, head-major).
            # M2's row-3 attn half drains its context on the op's surface too (decode_attn's
            # trailing "out"); under with_attn it is the folded arm's context -- and under
            # mlp_row4 it is row 4's input as well (the shim refills it onto the misc channel).
            *([AIERuntimeArgSpec("out", (self.Hq * self.HD,))]
              if (self.with_attn or self.attn_row3) else []),
            # ---- M2 ROW 4's surface (fuse_o: o-projection + SwiGLU MLP), in the Runtime's own
            # order.  `a_scratch`/`gh_scratch` are the two all-gather round trips every core
            # re-reads (inout: the device writes them, the device reads them back, and the host
            # side's zero-fill at dispatch start is what makes the first round well-defined).
            *([
                AIERuntimeArgSpec("in", (self.D,)),                   # n_pf (MLP RMSNorm weight)
                AIERuntimeArgSpec("in", (g["wo_units"],)),            # Wo, tile-major windows
                AIERuntimeArgSpec("in", (g["wg_units"],)),            # Wg
                AIERuntimeArgSpec("in", (g["wg_units"],)),            # Wu
                AIERuntimeArgSpec("in", (g["wd_units"],)),            # Wd
                AIERuntimeArgSpec("inout", (self.D,)),               # a_scratch
                AIERuntimeArgSpec("inout", (g["ff"],)),              # gh_scratch
                AIERuntimeArgSpec("out", (self.D,)),                 # x_out (the new residual)
            ] if self.mlp_row4 else []),
        ]
