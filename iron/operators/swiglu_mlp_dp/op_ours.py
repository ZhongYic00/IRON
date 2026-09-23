# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""int8_ours arm of their SwiGLUMLPDataParallel: OUR mv_int8.cc (mode14 premul,
VEC_SIZE=128) on OUR wire (per (col,tile) [tile_m*K u8 | tile_m*(K/128)*2B bf16
scales], viewed as bf16 elements). Everything else (dataflow topology, arg
order, gh all-gather) is inherited unchanged from the upstream op.

Reference: dequant int8 weights (u8_biased-128)*scale then their MLP math.
"""

import os
from dataclasses import dataclass, field
from typing import ClassVar, Dict

import numpy as np

from iron.common import (
    AIERuntimeArgSpec,
    KernelObjectArtifact,
    KernelArchiveArtifact,
    SourceArtifact,
    PythonGeneratedMLIRArtifact,
    DesignGenerator,
)
import aie.utils as aie_utils
from iron.common.device_utils import get_kernel_dir
from iron.operators.swiglu_mlp_dp.op import SwiGLUMLPDataParallel


def _payload_to_q(payload_bytes: np.ndarray, kernel_source: str | None) -> np.ndarray:
    """Wire payload bytes -> the integer q the kernel's matvec multiplies, as float32.

    The two payload domains this tree ships carry the SAME integer in different bytes: the
    biased domain (`kernel_source is None`) stores q+128 in a uint8 and the kernel's
    `add(-128)` recovers q; the signed domain stores q as two's complement in an int8 and the
    kernel's `to_float<bfloat16>(int8)` IS q.  The packing class's `reference()` must follow
    whichever domain its `kernel_source` selected, or it silently dequantizes the signed wire
    into garbage.
    """
    if kernel_source is None:
        return payload_bytes.astype(np.float32) - 128.0
    return payload_bytes.view(np.int8).astype(np.float32)


@dataclass
class SwiGLUMLPDataParallelOurs(SwiGLUMLPDataParallel):
    """Same 6-in/1-out surface as the parent; weights are OUR packed int8.

    tile_rows_gu=12 / weight_depth=1 (vs the bf16 arm's 6/2): our wire row is
    K*65/64 bytes = 1040B at K=1024, so a 6-row tile is 6240B = 32 mod 64 —
    a load_v alignment footgun (every tile past the first lands mid-vector;
    see the 附21 pad lesson). 12 rows -> 12480B = 0 mod 64 ✓, TSI_D = 12/R = 4
    keeps the shared-tile identity (12*520 == 4*1560), and depth=1 holds the
    same L1 budget as 2-deep 6-row tiles."""

    tile_rows_gu: int = field(default=12, repr=False)
    weight_depth: int = field(default=1, repr=False)
    # OPTION B: stage the weight channel through the column's MemTile (a deep L2 buffer feeding a
    # shallow L1 fifo) -- see design_ours_int8.py.  Default False = the direct path.
    # repr=True (NOT the parent's repr=False): MLIROperator.name skips repr=False fields, so the
    # artifact name -- which is what the fused-MLIR cache keys on -- would be IDENTICAL for the
    # direct and the staged arm and a shared build dir would silently run the wrong ELF.  These two
    # are the only knobs that change the channel topology; tsi/weight_depth keep the parent's
    # repr=False (documented: wipe the build dir when changing op kwargs).
    weight_memtile: bool = field(default=False)
    weight_stage_depth: int = field(default=16)
    # Per-INSTANCE kernel sources for the two int8 matvec translation units.  Default None = the
    # production BIASED pair (mv_int8.cc for the gu/down0/o instantiations, mv_int8_acc.cc for the
    # down accumulator), which keeps every other chain's artifact names and MLIR byte-identical.
    # Set BOTH together (e.g. "mv_int8_signed.cc" / "mv_int8_acc_signed.cc") to read the payload as
    # signed int8, which drops the per-weight `add(-128)` from the hot loop.  The CALLER must pack
    # the three wires in the matching domain -- kernel source and packing domain are ONE switch
    # (the packer lives in the chain), exactly like GEMVInt8.kernel_source.
    # repr=False, like the parent's tile/weight knobs: these are not in `name`, so the kernel
    # variants are isolated by BUILD DIR (the tree's convention for kernel-source switches).  As an
    # extra guard the object FILENAMES get a "_sgn" tag below, so two arms sharing a build dir can
    # never silently reuse each other's cached .o.
    kernel_source: str | None = field(default=None, repr=False)
    acc_kernel_source: str | None = field(default=None, repr=False)

    # The two arms above MUST land in different artifact names: the fused-MLIR cache keys on the
    # NAME, not on the op kwargs, so two configs sharing a build dir would silently run each
    # other's ELF.  Same idiom as qkv_head_dp's mc/mo/rtl.
    _name_aliases: ClassVar[Dict[str, str]] = {
        **SwiGLUMLPDataParallel._name_aliases,
        "weight_memtile": "wmm",
        "weight_stage_depth": "wsd",
    }

    def __post_init__(self):
        super().__post_init__()
        if (self.FF % self.D) != 0:
            # 4B (chunked-down) arm: FF is padded up to a whole multiple of D (9728 -> 10240) by
            # the weight packer -- the pad rows carry ZERO weights, so the extra rows compute
            # exactly 0 and everything downstream (rows per core, emit rounds, K chunks) becomes
            # uniform.  A non-uniform shape (a short last round) moves a FRACTIONAL fifo object,
            # which is the fractional-object accounting failure the shim has:
            # accounting is per-object, so the drains read each other's data.  Cost: the pad rows'
            # own weight bytes, Wg/Wu +5.3%.
            self.ff_pad = -(-self.FF // self.D) * self.D
            # Tile rows x weight-fifo DEPTH: the weight stream is this op's dominant traffic, and
            # at depth=1 the shim cannot prefetch the next tile -- measured 29.2 GB/s against the
            # chain's own GEMVInt8 at 39.7 GB/s, whose weight fifo is depth=2
            # (operators/gemv_int8/design.py:88).  L1 decides the pair: 8 rows (20800 B, pad-free)
            # x depth 2 = 41,600 B of weight L1 does NOT fit the 64 KB budget, 4 rows (10400 B
            # + 32 B of tile pad) x depth 2 = 20,864 B does (53,120 B total).  Both are
            # env-overridable so the S2 bandwidth A/B can hold everything else fixed.
            n = self.num_aie_columns * self.num_aie_rows
            ff_pc = self.ff_pad // n
            self.tile_rows_gu = int(os.environ.get("MLP4B_TSI", "4"))
            self.weight_depth = int(os.environ.get("MLP4B_DEPTH", "2"))
            self.weight_memtile = os.environ.get("MLP4B_MEMTILE", "0") == "1"
            self.weight_stage_depth = int(os.environ.get("MLP4B_STAGE_DEPTH", "16"))
            assert ff_pc % self.tile_rows_gu == 0, f"{self.tile_rows_gu} must divide FF_PAD/N={ff_pc}"
            assert (self.D // n) % self.tile_rows_gu == 0, (
                f"{self.tile_rows_gu} must divide D/N={self.D // n} (TSI_D == TSI_GU here)")
        else:
            self.ff_pad = self.FF
            # 0.6B arm (FF divisible by D): the prefetch argument is shape-independent -- at
            # weight_depth=1 the shim cannot start the next weight tile until the core frees the
            # current one, so the stream stalls between tiles. The knobs above are 4B-only, so the
            # 0.6B arm has been running the dataclass default (depth=1) throughout. Env-gated for
            # A/B; L1 is the binding constraint (design_ours_int8's budget assert fails hard rather
            # than silently, so an over-budget pair shows up at compile time).
            self.tile_rows_gu = int(os.environ.get("MLP_TSI", self.tile_rows_gu))
            # depth default 2 (was the dataclass's 1 = no prefetch): A/B on 9/15 at TSI_GU=12
            # gave 45.0 -> 42.8 ms p50, gate cos 0.999373 unchanged. L1 at (12, 2) = 43,648 B of
            # 65,536 (depth 4 fails the assert at 68,608 B). TSI_GU=6 COMPILES but produces NaN
            # logits (chain gate), so 12 is the only valid tile width found — MLP_TSI stays a knob
            # but nothing else is known to work.
            self.weight_depth = int(os.environ.get("MLP_DEPTH", 2))

    def get_mlir_artifact(self):
        mlir_verbose = getattr(self.context, "mlir_verbose", False)
        return PythonGeneratedMLIRArtifact(
            f"{self.name}.mlir",
            DesignGenerator(
                self.operator_dir / "design_ours_int8.py",
                "my_swiglu_mlp_dp",
                (aie_utils.get_current_device(), self.D, self.ff_pad, self.epsilon),
                {
                    # 2 KB is the 0.6B arm's stack.  The chunked-down arm runs 4x the kernel calls
                    # per tile round (chunk-outer/tile-inner) out of the same core function, so it
                    # gets 4 KB -- the aie2p linker script places the stack flush against the
                    # objectFIFO buffers, so an overflow there is silent corruption, not a fault.
                    "stack_size": 0x1000 if (self.FF % self.D) else 0x800,
                    "n_aie_cols": self.num_aie_columns,
                    "n_aie_rows": self.num_aie_rows,
                    "QD": self.QD,
                    "fuse_o": self.fuse_o,
                    "weight_dtype": "int8_ours",
                    "group_size": self.group_size,
                    "weight_depth": self.weight_depth,
                    "tile_rows_gu": self.tile_rows_gu or None,
                    "chunk_down": (self.FF % self.D) != 0,
                    # The design gets the GATHER-padded FF (ff_pad) as FF; ff_real is the model's
                    # own hidden size, which bounds the gate/up WEIGHTS (Wd keeps the padded K).
                    "ff_real": self.FF,
                    "weight_memtile": self.weight_memtile,
                    "weight_stage_depth": self.weight_stage_depth,
                },
            ),
        )

    def get_kernel_artifacts(self):
        arch_dir = get_kernel_dir()
        kdir = self.context.base_dir / "aie_kernels"
        tag = "_int8_ours"

        add_obj = KernelObjectArtifact(
            f"add{tag}.o", dependencies=[SourceArtifact(kdir / "generic" / "add.cc")]
        )
        mul_obj = KernelObjectArtifact(
            f"mul{tag}.o", dependencies=[SourceArtifact(kdir / "generic" / "mul.cc")]
        )
        rms_norm_obj = KernelObjectArtifact(
            f"rms_norm_{self.D}{tag}.o",
            dependencies=[SourceArtifact(kdir / arch_dir / "rms_norm.cc")],
            extra_flags=[f"-DRMS_COLS={self.D}"],
        )
        silu_obj = KernelObjectArtifact(
            f"silu{tag}.o", dependencies=[SourceArtifact(kdir / arch_dir / "silu.cc")]
        )
        # OUR kernel, twice: gu (DIM_K=D) and down (DIM_K=FF, prefixed "down_").
        # VEC_SIZE=128 is the mode14 production setting (mv_int8.cc default in
        # this tree); GROUP_SIZE=128 matches the wire.
        # kernel_source / acc_kernel_source select the payload domain's translation units
        # (default None = the biased production pair).  `kern_tag` keeps the two domains' object
        # files distinct even inside one build dir.
        mv_src = self.kernel_source or "mv_int8.cc"
        acc_src = self.acc_kernel_source or "mv_int8_acc.cc"
        kern_tag = "" if self.kernel_source is None else "_sgn"
        common_flags = ["-DVEC_SIZE=128", "-DGROUP_SIZE=128"]
        mv_gu_obj = KernelObjectArtifact(
            f"gemv_{self.D}k_128vs_ours{kern_tag}.o",
            dependencies=[SourceArtifact(kdir / arch_dir / mv_src)],
            extra_flags=[f"-DDIM_K={self.D}"] + common_flags,
        )
        # 4B (chunked-down): the down matvec is CHUNKED along K -- every chunk is exactly D wide
        # (FF is pre-padded to a multiple of D), one chunk per misc object -- so the down weights
        # are chunk-major [chunk c][column g][tile j] and chunk 0 writes while the rest accumulate
        # (mv_int8_acc.cc, its own translation unit -- kernel variants never share a file's TU).
        # Uniform chunks mean ONE accumulate kernel: the old short-tail object (DIM_K = 2048) is
        # gone with the padding.
        if (self.FF % self.D) != 0:
            mv_d_obj = KernelObjectArtifact(
                f"down0_gemv_{self.D}k_128vs_ours{kern_tag}.o",
                dependencies=[SourceArtifact(kdir / arch_dir / mv_src)],
                extra_flags=[f"-DDIM_K={self.D}"] + common_flags,
                prefix_symbols="down0_",
            )
            mv_dacc_obj = KernelObjectArtifact(
                f"downacc_gemv_{self.D}k_128vs_ours{kern_tag}.o",
                dependencies=[SourceArtifact(kdir / arch_dir / acc_src)],
                extra_flags=[f"-DDIM_K={self.D}"] + common_flags,
                prefix_symbols="downacc_",
            )
            # The gate/up wire carries the model's real FF rows only, so the core zeroes the
            # unwritten tail of g_buf/u_buf once per dispatch (design_ours_int8.py's prologue).
            # Its own source file: a variant inside an existing kernel file risks one arm's build
            # flags leaking into another arm's cached object.
            zero_obj = KernelObjectArtifact(
                "zero_offset_bf16.o",
                dependencies=[SourceArtifact(kdir / arch_dir / "zero_offset_bf16.cc")],
            )
            deps = [add_obj, mul_obj, rms_norm_obj, silu_obj, mv_gu_obj,
                    mv_d_obj, mv_dacc_obj, zero_obj]
        else:
            mv_d_obj = KernelObjectArtifact(
                f"down_gemv_{self.FF}k_128vs_ours{kern_tag}.o",
                dependencies=[SourceArtifact(kdir / arch_dir / mv_src)],
                extra_flags=[f"-DDIM_K={self.FF}"] + common_flags,
                prefix_symbols="down_",
            )
            deps = [add_obj, mul_obj, rms_norm_obj, silu_obj, mv_gu_obj, mv_d_obj]
        if self.fuse_o:
            # fuse_o's three extra kernels, mirroring the bf16 arm's (op.py) set but built from
            # OUR int8 sources: Wo's own matvec (DIM_K = QD, a third DIM_K instantiation of
            # mv_int8.cc next to DIM_K=D and DIM_K=FF -- hence its own object + "o_" prefix so the
            # symbol is unique), plus the two renamed copy_offset objects the design's cx/a
            # reassembly calls. copy_offset_bf16_vector is generic over pointers but a func.func
            # symbol is keyed by NAME only, so each new call-site shape needs its own renamed
            # object (design_ours_int8.py's fuse_o Kernel declarations spell out the types).
            if (self.FF % self.D) != 0:
                # CHUNKED fuse_o: the o-projection is a K-chunked matvec, so its writer and its
                # accumulator are mv_int8_acc.cc's chunked entry points at THIS arm's o-chunk K
                # (both symbols live in one object, so one artifact + one `o_` prefix serves both
                # -- hence the two symbol names design_ours_int8.py declares).  The cx copy needs
                # the source-offset variant, i.e. copy_off2_bf16.cc.
                _n_cx, _k_o, _tsi_o, _n_o_tiles, _wtile = self._wo_tiling_chunked()
                mv_o_obj = KernelObjectArtifact(
                    f"o_gemv_{_k_o}k_128vs_ours{kern_tag}.o",
                    dependencies=[SourceArtifact(kdir / arch_dir / acc_src)],
                    extra_flags=[f"-DDIM_K={_k_o}"] + common_flags,
                    prefix_symbols="o_",
                )
                cxb_copy_obj = KernelObjectArtifact(
                    "copy_off2.o",
                    dependencies=[SourceArtifact(kdir / "generic" / "copy_off2_bf16.cc")],
                    prefix_symbols="cxb_",
                )
                deps += [mv_o_obj, cxb_copy_obj]
            else:
                mv_o_obj = KernelObjectArtifact(
                    f"o_gemv_{self.QD}k_128vs_ours{kern_tag}.o",
                    dependencies=[SourceArtifact(kdir / arch_dir / mv_src)],
                    extra_flags=[f"-DDIM_K={self.QD}"] + common_flags,
                    prefix_symbols="o_",
                )
                cx_copy_obj = KernelObjectArtifact(
                    "add_cxcopy.o", dependencies=[SourceArtifact(kdir / "generic" / "add.cc")],
                    prefix_symbols="cx_",
                )
                deps += [mv_o_obj, cx_copy_obj]
            # the a-slice all-gather's copy (a rename of add.cc's copy_offset_bf16_vector) is
            # common to both fuse_o arms
            oa_copy_obj = KernelObjectArtifact(
                "add_oacopy.o", dependencies=[SourceArtifact(kdir / "generic" / "add.cc")],
                prefix_symbols="oa_",
            )
            deps.append(oa_copy_obj)
        # archive name MUST match design_ours_int8.py's CORE_ARCHIVE
        # (f"swiglu_mlp_dp_core_{weight_dtype}g{group_size}.a") or the per-core
        # link can't find the object.
        core_archive = KernelArchiveArtifact(
            f"swiglu_mlp_dp_core_int8_oursg{self.group_size}.a", dependencies=deps)
        return [core_archive]

    def _wrow(self, K):
        """Wire units per weight ROW of width K, in BF16 ELEMENTS (bytes/2):
        row = K int8 payload + (K/128)*2B bf16 scales."""
        return (K + (K // 128) * 2) // 2

    def _wspec(self, n_units, comment_unused=None):
        # bf16 element units, dtype omitted (bf16) — the arena-friendly view.
        return AIERuntimeArgSpec("in", (n_units,))

    def _wg_units(self):
        """Wg/Wu's own arg length, in bf16 ELEMENT units -- the design's TILE-BLOCK length.

        design_ours_int8.py sizes these buffers by TILE BLOCKS, not rows:
        `Wg_L3_ty = np.ndarray[(N * N_GU_TILES * WTILE_UNITS,)]`, WTILE_UNITS =
        (TSI_GU * WROW_D * WUNIT + 63) // 64 * (64 // WUNIT).
        Non-chunked (0.6B) the tiles are exact (WTILE_UNITS == TSI_GU * WROW_D, no
        alignment pad), so that product IS ff_pad * _wrow(D); asserted below.
        Chunked (4B) every packed tile is rounded up to a 64 B multiple, and the
        gate/up tiles cover the model's REAL rows only (ff_real = self.FF): the
        64 pad elements of each core's gh slice are the zeroed L1 tail of
        g_buf/u_buf, not zero-weight wire rows (see design_ours_int8.py's
        `ff_real`).  So the tile count is (FF // N) // TSI_GU, and the row count
        (ff_pad * _wrow(D)) would be WRONG in both directions -- it misses the
        per-tile 64 B pad AND counts the 512 rows the wire no longer carries.
        """
        if (self.FF % self.D) == 0:
            n = self.num_aie_columns * self.num_aie_rows
            tsi = self.tile_rows_gu or 4
            tile_blocks = n * ((self.ff_pad // n) // tsi) * (tsi * self._wrow(self.D))
            rows = self.ff_pad * self._wrow(self.D)
            assert tile_blocks == rows, (
                f"tile-block Wg length {tile_blocks} != row-count length {rows} for the "
                f"non-chunked arm -- one of the two formulas is wrong for FF={self.FF}, D={self.D}"
            )
            return rows
        n = self.num_aie_columns * self.num_aie_rows
        tsi = self.tile_rows_gu or 4
        n_gu_tiles = (self.FF // n) // tsi
        wtile = (tsi * self._wrow(self.D) * 2 + 63) // 64 * 32    # bf16 units, 64 B-aligned
        return n * n_gu_tiles * wtile

    def _wd_units(self):
        """Wd's own arg length, in bf16 ELEMENT units.

        Non-chunked (0.6B): D rows of the FF-wide wire row -- unchanged.
        Chunked (4B): the down buffer is TILE BLOCKS in chunk-major order
        [chunk c][column g][tile j], so its length is N * N_CHUNKS * N_D_TILES * WTILE and the
        per-column stride is N_CHUNKS * N_D_TILES * WTILE (see design_ours_int8.py's Wd_L3_ty and
        the tg3 fills -- the three must agree or the DMA walks off the end).  Every chunk is D
        wide (FF pre-padded); the tile block itself is WTILE bytes, rounded up to a 64 B multiple
        with the pad inside the block (see design_ours_int8.py's WTILE_UNITS).
        """
        if (self.FF % self.D) == 0:
            return self.D * self._wrow(self.FF)
        n = self.num_aie_columns * self.num_aie_rows
        n_chunks = self.ff_pad // self.D
        tsi = self.tile_rows_gu or 4
        n_d_tiles = (self.D // n) // tsi
        wtile = (tsi * self._wrow(self.D) * 2 + 63) // 64 * 32    # bf16 units, 64 B-aligned
        return n * n_chunks * n_d_tiles * wtile

    def _gh_units(self):
        """gh_scratch length: the all-gather round-trips the whole FF vector, so N_CHUNKS * D --
        which is exactly the padded FF at 4B (every chunk is D wide) and FF at 0.6B.  Both the
        fills and the drains move whole objects of D / D_PER_CORE elements."""
        return self.ff_pad

    def _wo_tiling(self):
        """(N_O_TILES, TSI_O, WTILE_UNITS) -- the fuse_o tile arithmetic, mirroring
        design_ours_int8.py's FUSE_O section. WTILE_UNITS is in int8 BYTES here."""
        n = self.num_aie_columns * self.num_aie_rows
        tsi_gu = self.tile_rows_gu or 12
        wtile_units = tsi_gu * self._wrow(self.D) * 2          # _wrow is bf16 units -> bytes
        wrow_qd = self._wrow(self.QD) * 2                      # ditto, in bytes
        assert wtile_units % wrow_qd == 0, (
            f"fuse_o needs the shared weight tile ({wtile_units} B) to be a whole number of "
            f"Wo rows ({wrow_qd} B each); it isn't")
        tsi_o = wtile_units // wrow_qd
        n_o_tiles = -(-(self.D // n) // tsi_o)
        return n, n_o_tiles, tsi_o, wtile_units

    def _wo_tiling_chunked(self):
        """(N_CX, K_O, TSI_O, N_O_TILES, WTILE_UNITS) -- the CHUNKED arm's o-projection.

        Mirrors design_ours_int8.py's chunked fuse_o geometry block: the o-projection is split
        along K like `down` (K_O-wide chunk rows whose tile is byte-identical in shape to a
        TSI_GU-row D-wide tile, 64 B pad included), instead of being row-tiled onto the shared
        object as the 0.6B arm does -- at 4B the row-tiled form needs a 20800 B object, which does
        not fit L1 at weight_depth 2 and loses its prefetch at depth 1.  WTILE_UNITS is in bf16
        units, K_O in elements."""
        n = self.num_aie_columns * self.num_aie_rows
        d_per_core = self.D // n
        tsi_gu = self.tile_rows_gu
        assert tsi_gu, "the chunked arm always sets tile_rows_gu"
        wrow_d = self._wrow(self.D)                            # bf16 units per D-wide row
        wtile = (tsi_gu * wrow_d * 2 + 63) // 64 * 32          # bf16 units, 64 B-aligned
        assert self.QD % 128 == 0, self.QD
        for k in range(min(self.D, self.QD) // 128 * 128, 0, -128):
            if self.QD % k:
                continue
            wrow_o = (k + (k // 128) * 2) // 2
            if (tsi_gu * wrow_d) % wrow_o:
                continue
            tsi_o = (tsi_gu * wrow_d) // wrow_o
            if d_per_core % tsi_o:
                continue
            assert tsi_o * wrow_o == tsi_gu * wrow_d and 0 <= wtile - tsi_o * wrow_o < 32
            return self.QD // k, k, tsi_o, d_per_core // tsi_o, wtile
        raise ValueError(
            f"fuse_o (chunked): no K-chunk width divides QD={self.QD} into rows whose tile is "
            f"shape-identical to the {tsi_gu}-row D-wide tile ({tsi_gu * wrow_d} units)")

    def _wo_units(self):
        """Wo's arg length for the fuse_o arm, in the bf16 element units `_wspec` sizes with.

        TILE-major, not row-interleaved: the design's `Wo_L3_ty` is a tile-major buffer and the
        shim fills each core a CONTIGUOUS run of tiles -- [column g][tile j] on the non-chunked
        arm (block-major tiles `[TSI_O*QD payload | TSI_O*(QD/128)*2 scales]`, per-core windows
        that duplicate the ceil-rounding rows), and [K-chunk ci][column g][tile j] on the chunked
        one.  Row-interleaving has the same BYTE COUNT in the overlap case (132 rows == 22 tiles *
        6 rows), which is exactly why the wrong order passed every length check and still fed
        mv_int8.cc a transposed wire. See design_ours_int8.py's WO_L3_UNITS comment.
        """
        assert self.fuse_o, "_wo_units is the fuse_o arm's spec"
        n = self.num_aie_columns * self.num_aie_rows
        if (self.FF % self.D) != 0:
            n_cx, _k_o, _tsi_o, n_o_tiles, wtile = self._wo_tiling_chunked()
            # `wtile` is already in bf16 UNITS (the wire is addressed in bf16 elements, like
            # _wg_units/_wd_units) -- only the non-chunked `_wo_tiling` returns bytes.
            return n * n_cx * n_o_tiles * wtile
        _n, n_o_tiles, _tsi_o, wtile_units = self._wo_tiling()
        return (n * n_o_tiles * wtile_units) // 2              # bytes -> bf16 units

    def get_arg_spec(self):
        if self.fuse_o:
            # Order mirrors design_ours_int8.py's fuse_o `rt_args`:
            #   cur, cx, n_pf, Wo, Wg, Wu, Wd, gh_scratch, a_scratch, nxt
            # `a = Wo @ cx` is computed on-chip, so the standalone "a" input and the separate
            # gemv_output entry/design disappear from the runlist.
            return [
                AIERuntimeArgSpec("in", (self.D,)),                 # cur
                AIERuntimeArgSpec("in", (self.QD,)),                # cx (attention context)
                AIERuntimeArgSpec("in", (self.D,)),                 # n_pf
                self._wspec(self._wo_units()),                      # Wo, flat [D+pad, QD]
                self._wspec(self._wg_units()),                      # Wg
                self._wspec(self._wg_units()),                      # Wu
                self._wspec(self._wd_units()),                      # Wd
                AIERuntimeArgSpec("inout", (self._gh_units(),)),    # gh_scratch
                AIERuntimeArgSpec("inout", (self.D,)),              # a_scratch
                AIERuntimeArgSpec("out", (self.D,)),                # nxt
            ]
        # otherwise identical surface to the parent's non-fuse_o arm, weights resized to
        # our wire (bf16 element units)
        return [
            AIERuntimeArgSpec("in", (self.D,)),                 # cur
            AIERuntimeArgSpec("in", (self.D,)),                 # a
            AIERuntimeArgSpec("in", (self.D,)),                 # n_pf
            self._wspec(self._wg_units()),  # Wg (pad rows carry zero weights)
            self._wspec(self._wg_units()),  # Wu
            self._wspec(self._wd_units()),              # Wd
            AIERuntimeArgSpec("inout", (self._gh_units(),)),    # gh_scratch
            AIERuntimeArgSpec("out", (self.D,)),                # nxt
        ]

    def _dequant_flat(self, w_packed, rows, K):
        """Unpack fuse_o's Wo wire into a (D, K) float matrix.

        Wo is TILE-major: each core owns a contiguous run of `N_O_TILES` block-major tiles of
        `TSI_O` rows (`[TSI_O*K payload | TSI_O*(K/128)*2 scales]`), covering that core's
        ceil(D_PER_CORE/TSI_O)*TSI_O-row window; the windows duplicate the overlap rows, and only
        the first `rows` (= D) rows of the matrix are real. See `_wo_units`.
        """
        import ml_dtypes
        import torch

        if (self.FF % self.D) != 0:
            # CHUNKED wire: [K-chunk ci][column g][tile j], each tile `TSI_O` rows x `K_O` wide with
            # the block's own 64 B pad -- the order design_ours_int8.py's WO_L3_UNITS fills and the
            # core walks (chunk-outer, tile-inner).  `rows` is D and K is QD.
            n = self.num_aie_columns * self.num_aie_rows
            n_cx, k_o, tsi_o, n_o_tiles, wtile_u = self._wo_tiling_chunked()
            b = w_packed.contiguous().view(torch.uint8).numpy()
            tile_bytes = wtile_u * 2
            ng = k_o // 128
            rows_per_col = self.D // n
            W = np.empty((rows, K), dtype=np.float32)
            koff = 0
            for ci in range(n_cx):
                for g in range(n):
                    for j in range(n_o_tiles):
                        o = ((ci * n + g) * n_o_tiles + j) * tile_bytes
                        blk = b[o:o + tile_bytes]
                        w = _payload_to_q(blk[:tsi_o * k_o], self.kernel_source).reshape(
                            tsi_o, k_o)
                        sc = np.frombuffer(blk[tsi_o * k_o:].tobytes(),
                                           dtype=ml_dtypes.bfloat16).astype(np.float32)
                        sc = sc[:tsi_o * ng].reshape(tsi_o, ng)
                        r0 = g * rows_per_col + j * tsi_o
                        W[r0:r0 + tsi_o, koff:koff + k_o] = (
                            w.reshape(tsi_o, ng, 128) * sc[:, :, None]).reshape(tsi_o, k_o)
                koff += k_o
            return torch.from_numpy(W)

        n, n_o_tiles, tsi_o, _wu = self._wo_tiling()
        b = w_packed.contiguous().view(torch.uint8).numpy()
        n_groups = K // 128
        tile_bytes = tsi_o * (K + n_groups * 2)
        rows_per_col = self.D // n
        W = np.empty((rows, K), dtype=np.float32)
        for g in range(n):
            for j in range(n_o_tiles):
                o = (g * n_o_tiles + j) * tile_bytes
                blk = b[o:o + tile_bytes]
                w = _payload_to_q(blk[:tsi_o * K], self.kernel_source).reshape(tsi_o, K)
                sc = np.frombuffer(blk[tsi_o * K:].tobytes(),
                                   dtype=ml_dtypes.bfloat16).astype(np.float32)
                sc = sc.reshape(tsi_o, n_groups)
                r0 = g * rows_per_col + j * tsi_o
                keep = min(tsi_o, max(0, rows - r0))
                if keep <= 0:
                    continue
                rows_3d = w[:keep].reshape(keep, n_groups, K // n_groups)
                W[r0:r0 + keep] = (rows_3d * sc[:keep, :, None]).reshape(keep, K)
        return torch.from_numpy(W)

    def _dequant_tiles(self, w_packed, M, K, tile_m):
        """Unpack the TILE-major wire (per (col,tile) [tile_m*K u8 | tile_m*(K/128)*2 B scales])
        into an (M, K) float matrix. Used by Wg/Wu/Wd, whose per-column tiles cover their own
        contiguous rows (so this order is also their flat order); Wo under fuse_o is different --
        see `_dequant_flat`."""
        import ml_dtypes
        import torch

        cols = self.num_aie_columns
        b = w_packed.contiguous().view(torch.uint8).numpy()
        n_groups = K // 128
        tile_bytes = tile_m * (K + n_groups * 2)
        rows_per_col = M // cols
        tiles_per_col = rows_per_col // tile_m
        W = np.empty((M, K), dtype=np.float32)
        for c in range(cols):
            for t in range(tiles_per_col):
                o = (c * tiles_per_col + t) * tile_bytes
                blk = b[o:o + tile_bytes]
                w = _payload_to_q(blk[:tile_m * K], self.kernel_source).reshape(tile_m, K)
                sc = np.frombuffer(blk[tile_m * K:].tobytes(),
                                   dtype=ml_dtypes.bfloat16).astype(np.float32)
                sc = sc.reshape(tile_m, n_groups)
                rows = w.reshape(tile_m, n_groups, K // n_groups)
                W[c * rows_per_col + t * tile_m: c * rows_per_col + (t + 1) * tile_m] = (
                    (rows * sc[:, :, None]).reshape(tile_m, K))
        return torch.from_numpy(W)

    def reference(self, cur, a, n_pf, *rest):
        """int8 reference: unpack OUR wire (per (col,tile) [tile_m*K u8 |
        tile_m*(K/128)*2B bf16 s]), dequant (u8_biased-128)*scale, then the
        parent math: nxt = x1 + Wd @ (silu(Wg@hf) * (Wu@hf)), x1 = cur+a,
        hf = rmsnorm_w(x1, n_pf).  Under fuse_o the second argument is `cx` (QD wide) and Wo is
        the first weight, so the parent's reference_fused_o does the math."""
        import ml_dtypes
        import torch

        dequant = self._dequant_tiles

        tsi_gu = self.tile_rows_gu or 12
        tsi_d = tsi_gu // (self.FF // self.D)
        if self.fuse_o:
            from iron.operators.swiglu_mlp_dp.reference import reference_fused_o
            Wo_p, Wg_p, Wu_p, Wd_p = rest[:4]
            Wo = self._dequant_flat(Wo_p, self.D, self.QD)
            Wg = dequant(Wg_p, self.FF, self.D, tsi_gu)
            Wu = dequant(Wu_p, self.FF, self.D, tsi_gu)
            Wd = dequant(Wd_p, self.D, self.FF, tsi_d)
            return reference_fused_o(torch.as_tensor(cur).float(),
                                     torch.as_tensor(a).float(),
                                     torch.as_tensor(n_pf).float(),
                                     Wo, Wg, Wu, Wd, self.D, self.FF, self.QD, self.epsilon)

        Wg_p, Wu_p, Wd_p = rest[:3]
        Wg = dequant(Wg_p, self.FF, self.D, tsi_gu)
        Wu = dequant(Wu_p, self.FF, self.D, tsi_gu)
        Wd = dequant(Wd_p, self.D, self.FF, tsi_d)

        cur = torch.as_tensor(cur).float()
        a = torch.as_tensor(a).float()
        n_pf = torch.as_tensor(n_pf).float()
        x1 = cur + a
        rms = torch.sqrt(torch.mean(x1 * x1) + self.epsilon)
        hf = (x1 / rms) * n_pf
        g = Wg @ hf
        sig = torch.where(g >= 0, 1 / (1 + torch.exp(-g)), torch.exp(g) / (1 + torch.exp(g)))
        gh = g * sig * (Wu @ hf)
        return (x1 + Wd @ gh).to(torch.bfloat16)
