# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Data-parallel decode QKV head: ONE `aie.device` for

    hn  = weighted_RMSNorm(cur, n_in)                          D-wide, replicated on every core
    raw = Wqkv[head g] @ hn                                    core c's own heads
    out = RoPE(weighted_RMSNorm(raw, n_qn | n_kn), ang)        q and k heads
        = raw                                                  v heads

replacing four consecutive designs in the decode runlist (RMSNorm, the concatenated QKV GEMV, the
per-head qk-RMSNorm, the q+k RoPE) with one, so the group costs one `aiex.configure` per layer
instead of four.

The shape this is NOT is the point. `fuse/qkv-head` fused the same four ops and measured **+28.2%
SLOWER** on device, and the reason was never placement: it put cur, n_in AND every weight row on
ONE D-wide input ObjectFifo, which forces the matvec's `tile_size_input` to 1 (128 calls per head
instead of 32) and makes every weight fill hand the core 2 KB at a time. It also finished a
TaskGroup per column, which serialises the eight columns behind each other's drains. Both are
copied from that file's own record of what it gave up.

This one takes swiglu_mlp_dp's channel split instead, which measured -29.3% on the MLP block:

  MISC (1 input channel, BROADCAST to all N cores). HD-wide. Carries `cur` and `n_in` as D/HD
  chunks each -- reassembled into L1 by an explicit-offset copy, the same idiom swiglu_mlp_dp uses
  to rebuild its all-gathered `gh` -- then `n_qn`, `n_kn` and `ang`, which are acquired together
  and held for the rest of the body because each is read once per head. HD-wide rather than D-wide
  so every object is exactly one fill's worth: a D-wide tile would need a 128-of-1024 partial fill
  for the three small constants, which nothing in this codebase does.

  WEIGHT (1 input channel, per core). `tile_size_input` rows of D, filled ONCE per core as a single
  contiguous run over that core's slice of Wqkv -- one BD for 1 MB, chopped into tiles by the fifo
  rather than by the runtime.

  OUTPUT (1 of the 2 available). HD-wide, one drain per head.

Two in, one out per tile, against the hard 2-in/2-out of an AIE2P compute tile. Device-wide the
shim budget is 16 INPUT channels (`get_shim_dma_limit`), and this design spends misc(1) + weight(N)
= 9 at N=8 -- the same accounting that stops swiglu_mlp_dp at N=8 and not 16.

Heads are assigned to cores by ROW, not by kind: core c owns the contiguous rows
[c*ROWS_PER_CORE, (c+1)*ROWS_PER_CORE) of the concatenated [QD+2*KVD, D] weight, which is a whole
number of heads. Whether a head is q, k or v then decides only what happens to it AFTER the matvec,
and that is a Python-level branch -- each Worker traces its own body. Per-core weight bytes are
therefore identical whatever the q/k/v split is, which is the property the spatial version lost.
"""

import math

import aie.dialects.index as index
from aie.dialects.aie import T
from ml_dtypes import bfloat16
import numpy as np

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (Buffer, Kernel, ObjectFifo, Program, Runtime, ScratchpadParameter,
                      TaskGroup, Worker, sync_parameters)
from aie.iron.device import Tile

from iron.operators._trace import maybe_enable_trace

BF16 = bfloat16


def _tile_tap(total, n_rows, row_len, offset=0):
    """A 2-D tap over `n_rows` contiguous rows of `row_len` elements -- the same bytes as a flat
    tap of n_rows*row_len, but the BD's LENGTH stays one row and the row count rides the wrap
    dimension (<= 1023).  Used for the merged drains; see the drain loop."""
    return TensorAccessPattern((1, total), offset,
                               [1, 1, n_rows, row_len],
                               [0, 0, row_len, 1])


def _flat_tap(total, size, offset=0):
    """A contiguous [offset:offset+size) window of an L3 buffer of `total` elements. `total` is the
    FULL declared size -- TensorAccessPattern validates offset+extent against it, so a bare
    (size,) is correct only at offset 0."""
    return TensorAccessPattern((1, total), offset, [1, 1, 1, size], [0, 0, 0, 1])


def mlp_row4_geometry(D, QD, N, ff=None):
    """The row-4 (swiglu_mlp_dp fuse_o) tiling, in ONE place: this function is called by the
    design AND by op_ours.py, whose `get_arg_spec` has to size the very same buffers -- two
    copies of this arithmetic is how a wire and its arg spec drift apart.

    Mirrors design_ours_int8.py's NON-chunked arm (FF a whole multiple of D) at the shipped 0.6B
    knobs: tile_rows_gu=12, weight_depth=2.  All weight sizes are in bf16 ELEMENTS (bytes/2),
    which is the unit the fifo object itself is declared in (`[m*K u8 | m*(K/128)*2B scales]`
    blocks reinterpreted as bf16).  Every number is derived, none is hardcoded, because a
    hardcoded one is a bug waiting for the next shape ("hanging numbers are bugs").
    """
    FF = ff if ff else 3 * D
    assert FF % D == 0 and QD % D == 0, (FF, QD, D)
    R = FF // D                                  # 3 at 0.6B: gh chunks and down's K/D
    TSI_GU = 12
    assert TSI_GU % R == 0, f"tile_rows_gu ({TSI_GU}) must be a multiple of R ({R})"
    TSI_D = TSI_GU // R
    assert D % N == 0 and FF % N == 0, (D, FF, N)
    D_PER_CORE, FF_PER_CORE = D // N, FF // N
    assert FF_PER_CORE % 32 == 0, (
        f"silu_tile_bf16 walks its buffer 32 lanes at a time with no tail handling; FF/N "
        f"({FF_PER_CORE}) must be a multiple of 32")
    assert FF_PER_CORE % TSI_GU == 0 and D_PER_CORE % TSI_D == 0, (FF_PER_CORE, D_PER_CORE)
    assert FF_PER_CORE % D_PER_CORE == 0, (
        "every gh emit round must move a WHOLE out object (FF/N a multiple of D/N)")
    N_GH_ROUNDS = FF_PER_CORE // D_PER_CORE
    R_CX = QD // D                               # context chunks on the misc channel
    WROW_D = (D + (D // 128) * 2) // 2           # wire units per row of width D
    WROW_FF = (FF + (FF // 128) * 2) // 2
    WROW_QD = (QD + (QD // 128) * 2) // 2
    WTILE = TSI_GU * WROW_D                      # ONE fifo object serves Wg/Wu/Wd/Wo
    assert TSI_GU * WROW_D == TSI_D * WROW_FF, (
        "the shared weight tile identity broke (a Wg/Wu tile and a Wd tile must be the same bytes)")
    assert WTILE % WROW_QD == 0, (
        f"fuse_o needs the shared weight tile ({WTILE} units) to be a whole number of Wo rows "
        f"({WROW_QD} units each); it isn't, so Wo cannot share this channel")
    TSI_O = WTILE // WROW_QD
    N_O_TILES = -(-D_PER_CORE // TSI_O)          # ceil: every core reads FULL tiles
    O_WINDOW = N_O_TILES * TSI_O                 # rows actually read per core (>= D_PER_CORE)
    O_OVERLAP = O_WINDOW - D_PER_CORE            # extra rows read past this core's own slice
    assert 0 <= O_OVERLAP < TSI_O, (O_WINDOW, D_PER_CORE, TSI_O)
    N_GU_TILES = FF_PER_CORE // TSI_GU
    N_D_TILES = D_PER_CORE // TSI_D
    return {
        "ff": FF, "qd": QD, "r": R, "r_cx": R_CX, "tsi_gu": TSI_GU, "tsi_d": TSI_D,
        "d_per_core": D_PER_CORE, "ff_per_core": FF_PER_CORE, "n_gh_rounds": N_GH_ROUNDS,
        "wtile": WTILE, "tsi_o": TSI_O, "n_o_tiles": N_O_TILES, "o_window": O_WINDOW,
        "o_overlap": O_OVERLAP, "n_gu_tiles": N_GU_TILES, "n_d_tiles": N_D_TILES,
        # L3 buffers, in bf16 elements (tile-major, per-core windows -- see op_ours._wo_units)
        "wo_units": N * N_O_TILES * WTILE,
        "wg_units": N * N_GU_TILES * WTILE,
        "wd_units": N * N_D_TILES * WTILE,
        # the shipped 0.6B MLP weight-fifo depth (2 = the stream prefetches)
        "weight_depth": 2,
        # the MISC queue row 4 shares with row 2, in fill order:
        #   cur | n_in | n_qn | n_kn | ang | cx[R_CX] | n_pf | a_sc | gh[R]
        # `skip` is what row 2 has to WALK after its own five objects: the producer's empty lock
        # needs both consumers to release every object, so a consumer that stops early hangs the
        # next fill.
        "skip": R_CX + 1 + 1 + R,
    }


def m2_layer(
    dev,
    D,
    HD,
    Hq,
    Hkv,
    max_seq,
    epsilon=1e-6,
    tile_size_input=4,
    stack_size=0xD00,
    func_prefix="",
    n_aie_cols=8,
    kv_offset_parameter="kv_off",
    trace_size=0,
    weight_depth=2,
    misc_obj_elems=None,
    misc_chunked=True,
    runtime_tile_loop=False,
    inplace_norm=False,
    misc_memtile=False,
    merge_drains=False,
    attn_colmajor=False,
    with_attn=False,
    attn_row3=False,
    B_KV=32,
    mlp_row4=False,
    mlp_ff=None,
    mlp_epsilon=1e-5,
    mlp_noop=False,
    ctx_drain=True,
    mlp_dump="",
    mlp_depth=None,
):
    """`func_prefix` is not optional once this design is placed in an OperatorSequence -- see
    gemv/design.py's identical parameter. N = n_aie_cols, one core per column.

    `mlp_row4` adds the THIRD phase row: `swiglu_mlp_dp`'s fuse_o core (the o-projection folded
    into the SwiGLU MLP) as its own worker on row 4 of every column, taking the attention's
    context and the layer's residual and producing the next residual.  `mlp_ff` is the MLP's
    hidden width (None = 3*D, the 0.6B shape), `mlp_epsilon` its RMSNorm epsilon (1e-5, NOT this
    layer's qkv epsilon), `mlp_noop` swaps the row-4 core for a counter-only body (wiring probe
    for the M1 compile gate), `ctx_drain` keeps/removes row 3's context drain to DRAM (row 4
    REQUIRES it -- that drain is the context's only way to reach a broadcast)."""
    N = n_aie_cols
    tsi = tile_size_input
    QD, KVD = Hq * HD, Hkv * HD
    TOT = QD + 2 * KVD                      # rows of the concatenated Wqkv
    HEADS = Hq + 2 * Hkv
    assert TOT == HEADS * HD
    assert HEADS % N == 0, f"{HEADS} heads must divide across N={N} cores"
    HEADS_PER_CORE = HEADS // N
    ROWS_PER_CORE = HEADS_PER_CORE * HD
    assert HD % tsi == 0, f"HD ({HD}) must divide by tile_size_input ({tsi})"
    assert D % HD == 0, f"this design carries `cur`/`n_in` as D/HD chunks; D={D} HD={HD}"
    # MISC object width.  HD (the default) is the 0.6B arm's dataflow: cur/n_in arrive
    # as D/HD objects, and a fill can only cover a limited number of objects per shim
    # tile (aiecc: "Too many simultaneously active buffer descriptors on tile (0,0),
    # up to 16" -- the OBJECT count of a fill counts against it, which is why the 4B
    # run at D=2560 needs one fill per object in its own TaskGroup).  Widening the
    # object to D/MO cuts both the object count and the number of fills:
    #   MO=HD   D=2560 -> 20+20 chunk fills per layer (43 `@misc` DMA tasks)
    #   MO=D/5  D=2560 ->  5+5 objects, 2 fills + 3 constants = 5 DMA tasks
    # The three HD-wide constants then no longer fill an object, so they ride padded
    # objects and the kernels read their first HD elements (a memref of the object
    # width is passed where the C signature only touches HD elements -- the constants
    # are host-side padded, never delivered fractionally).
    MO = misc_obj_elems if misc_obj_elems else HD
    assert MO >= HD, f"misc object {MO} must hold one whole HD constant ({HD})"
    assert D % MO == 0, f"D ({D}) must be a whole number of misc objects ({MO})"
    N_MISC_CHUNKS = D // MO
    # M2 row 4 rides the SAME misc fifo (a second consumer = broadcast), so its object width is
    # pinned to one D vector: `cur`, `n_pf`, the context and both all-gather refills are then one
    # object each, and aiecc's 16-active-BD-per-tile rule -- which counts a fill's OBJECTS -- sees
    # 1+1+3 = 5 per layer instead of 8+8+3 = 19.  It also lets `x` reuse the existing D-wide copy
    # kernel.  Depth 6 (vs 3) is slack for the two consumers: row 2 holds the three constants
    # across its whole head loop while row 4 is still filling up.
    MISC_DEPTH = (4 if mlp_row4 else 3)
    if mlp_row4:
        assert MO == D, (
            f"mlp_row4 needs one D-wide misc object per refill (MO={MO}, D={D}) -- the context, "
            f"n_pf and the two all-gather scratch refills all ride that channel")
    N_W_TILES = HD // tsi                   # weight tiles per head
    # OUR int8 wire (mv_int8.cc): per matvec row [D int8 payload | (D/128)*2B
    # bf16 scales] = (D*65/64) bytes; the fifo object is the bf16-element view
    # (bytes/2) so the fused bf16 arena reinterprets cleanly. Alignment rule:
    # tsi * row_bytes must be a multiple of 64 (load_v<128> footgun) — tsi=4
    # at D=1024 (4160B), tsi=8 at D=2560 (20800B).
    ROW_BYTES = D + (D // 128) * 2
    assert (ROW_BYTES * tsi) % 64 == 0, (
        f"tile stride {ROW_BYTES * tsi}B breaks 64B load_v alignment; "
        f"raise tile_size_input (row {ROW_BYTES}B % 64 = {ROW_BYTES % 64})")
    WTILE_ELEMS = tsi * ROW_BYTES // 2

    # L1 budget (64 KB/core), computed rather than assumed -- the same check swiglu_mlp_dp carries.
    L1_BYTES = 65536
    misc_bytes = MISC_DEPTH * (MO * 2)      # depth 3: n_qn, n_kn and ang are held together
    weight_bytes = weight_depth * (WTILE_ELEMS * 2)
    out_bytes = 2 * (HD * 2)
    # cur, n_in, hn + raw, normed + the loop-bound word.  `inplace_norm` drops hn:
    # the weighted RMSNorm writes over n_in, which is dead after step 1 (the kernel
    # loads b[i] before storing c[i], so b == c is safe element-wise) -- worth 5 KB
    # of the 64 KB core, which is what a D/2 misc object costs.
    persistent_bytes = (3 if not inplace_norm else 2) * (D * 2) + 2 * (HD * 2) + 64
    total = misc_bytes + weight_bytes + out_bytes + persistent_bytes + stack_size
    assert total <= L1_BYTES, (
        f"estimated L1 use {total} B exceeds {L1_BYTES} B at tsi={tsi} "
        f"(misc={misc_bytes} weight={weight_bytes} out={out_bytes} "
        f"persistent={persistent_bytes} stack={stack_size})"
    )

    # ---- M2 ROW 4's geometry: swiglu_mlp_dp's fuse_o tiling, recomputed rather than guessed ----
    # Everything below mirrors design_ours_int8.py's arithmetic for the NON-chunked (FF = R*D)
    # arm at the shipped 0.6B knobs: tile_rows_gu=12, weight_depth=2.  The wire units are bf16
    # ELEMENTS (bytes/2) exactly as there, because the fifo object the kernel sees is the
    # [m*K u8 | m*(K/128)*2B scales] block reinterpreted as bf16.  `mlp_row4_geometry` is shared
    # with op_ours.py (the arg spec has to size the same buffers), so it cannot live inline here.
    MLP_RAIL = 0x1000                # the row-4 core's stack
    # 2 KB is design_ours_int8.py's 0.6B value, and it is TOO SMALL HERE.  The aie2p linker
    # script places the stack immediately below the first objectFIFO buffer -- on this tile that is
    # `wmlp_*_cons_buff_0`, the weight fifo's ring, at 0x70800 with the stack at [0x70000, 0x70800)
    # -- so an overflow does not fault, it CORRUPTS THE WEIGHT TILES.  MEASURED with 0x800
    # (probe_m2_row4.py, M2R4_RAMP=1): only 56 of each core's 768 weight rows arrive intact; the
    # rest are byte garbage, and the per-core g/u rows that survive are a single 256-row window.
    # The shipped 0.6B MLP gets away with 2 KB because its core is simpler (this one adds the
    # fused-o copy kernels and the unrolled mv loops); 4 KB is what the 4B chunked arm uses for
    # the same reason, and the L1 check below is what keeps it honest.
    if mlp_row4:
        assert attn_row3 and not with_attn, (
            "mlp_row4 is row 4 of the M2 phase split: it needs the attention on its own row "
            "(attn_row3) and cannot be combined with M1's folding (with_attn)")
        assert ctx_drain, (
            "row 4's `a = Wo @ cx` needs the WHOLE context, which only the DRAM round trip + the "
            "misc broadcast can deliver (the live attn_outO object is this column's Hq/N of Hq "
            "heads); dropping row 3's context drain removes row 4's input")
        assert MO == D, f"row 4's refills are D-wide misc objects (MO={MO}, D={D})"
        _G = mlp_row4_geometry(D, QD, N, mlp_ff)
        (FF, QD_L, R, R_CX, MLP_TSI_GU, MLP_TSI_D, D_PER_CORE, FF_PER_CORE, N_GH_ROUNDS,
         MLP_WTILE, TSI_O, N_O_TILES, O_WINDOW, O_OVERLAP, N_GU_TILES, N_D_TILES,
         WO_L3_UNITS, WG_L3_UNITS, WD_L3_UNITS, MLP_WD_DEPTH, MISC_ROW4_SKIP) = (
            _G["ff"], _G["qd"], _G["r"], _G["r_cx"], _G["tsi_gu"], _G["tsi_d"],
            _G["d_per_core"], _G["ff_per_core"], _G["n_gh_rounds"], _G["wtile"], _G["tsi_o"],
            _G["n_o_tiles"], _G["o_window"], _G["o_overlap"], _G["n_gu_tiles"], _G["n_d_tiles"],
            _G["wo_units"], _G["wg_units"], _G["wd_units"], _G["weight_depth"], _G["skip"])
        # The weight fifo's depth: 1, NOT the shipped 0.6B MLP's 2, and the reason is the DM budget
        # rather than taste.  At M2's 4 columns the weight OBJECT doubles (12,480 B) while this
        # tile's other buffers do not shrink; (misc 4) + (weight 2) + the buffers + the 4 KB stack
        # does not fit the 64 KB DM, and the aie2p linker script resolves that by laying the stack
        # over the weight fifo WITHOUT a diagnostic (see MLP_RAIL).  At depth 1 the tile is ~77%
        # full and the only cost is the fifo's prefetch -- which the shipped MLP itself traded away
        # for months before depth 2 was measured in.  `mlp_depth` is the probe knob for that A/B.
        MLP_WD_DEPTH = 1
        if mlp_depth:
            MLP_WD_DEPTH = mlp_depth
        # L1 (row 4's own tile -- a separate core from row 2's, so its own budget check).
        # The persistent list is the worker's OWN buffer list, counted: mx/x1/hf (3 D-wide),
        # npf_buf + a_buf (2 more D-wide -- see the row-4 body: npf and a_sc are COPIED out of
        # their fifo objects and released at once, so no object is ever held), cx (QD), the
        # o-projection window (O_WINDOW), gh (FF), g/u (2 x FF/N) and d (D/N).
        mlp_misc = MISC_DEPTH * (MO * 2)
        mlp_weight = MLP_WD_DEPTH * (MLP_WTILE * 2)
        mlp_out = 2 * (D_PER_CORE * 2)
        mlp_persistent = (5 * D + FF + 2 * FF_PER_CORE + D_PER_CORE + QD_L + O_WINDOW) * 2
        mlp_l1 = mlp_misc + mlp_weight + mlp_out + mlp_persistent + MLP_RAIL
        assert mlp_l1 <= L1_BYTES, (
            f"row 4's L1 estimate {mlp_l1} B exceeds {L1_BYTES} B "
            f"(misc={mlp_misc} weight={mlp_weight} out={mlp_out} "
            f"persistent={mlp_persistent} stack={MLP_RAIL})")
        # THIS NUMBER IS LOAD-BEARING, not a tidiness check: the tile's stack is placed
        # immediately below its first objectFIFO buffer, so a tile that does not fit does not fail
        # to link -- the stack is laid over the fifo's ring and the weight tiles arrive corrupted
        # (measured: 56 of 768 rows intact, everything else byte garbage).  Keep >= 8 KB of slack.
    else:
        FF = QD_L = R = R_CX = 0
        MLP_TSI_GU = MLP_TSI_D = MLP_WTILE = TSI_O = N_O_TILES = O_WINDOW = O_OVERLAP = 0
        N_GU_TILES = N_D_TILES = N_GH_ROUNDS = MLP_WD_DEPTH = MISC_ROW4_SKIP = 0
        WO_L3_UNITS = WG_L3_UNITS = WD_L3_UNITS = D_PER_CORE = FF_PER_CORE = 0

    # k and v are drained STRAIGHT into the KV caches at the token's own offset instead of into a
    # `qkv` buffer that a StridedCopy then re-reads and re-writes. The caches are the only consumer
    # of either (op_scores reads kc, TMatVec reads vc), so the intermediate never had a reader --
    # it existed because the append was a separate operator. Deletes two runs and one configure per
    # layer, and the k/v DDR round trip with them.
    kv_off_param = (ScratchpadParameter(kv_offset_parameter, np.int32)
                    if kv_offset_parameter is not None else None)

    D_ty = np.ndarray[(D,), np.dtype[BF16]]
    HD_ty = np.ndarray[(HD,), np.dtype[BF16]]
    MO_ty = np.ndarray[(MO,), np.dtype[BF16]]
    WTILE_ty = np.ndarray[(WTILE_ELEMS,), np.dtype[BF16]]
    W_L3_ty = np.ndarray[(TOT * ROW_BYTES // 2,), np.dtype[BF16]]
    Q_L3_ty = np.ndarray[(QD,), np.dtype[BF16]]
    # OUR kv layout: ONE interleaved per-token cache ([K_t|V_t] per token,
    # token t's K at kv_off + h*HD, V at kv_off + HD + h*HD, kv_off = t*2*HD
    # patched per token) — the qwen3 chain's cache, so the downstream
    # decode_attn reads its shipped combined tap unchanged.
    # OUR kv layout: interleaved per-token [K_t|V_t] — FULL K+V in one cache
    # (2x a single kind's extent).
    KV_L3_ty = np.ndarray[(Hkv * max_seq * HD * 2,), np.dtype[BF16]]

    # ---- kernels: one archive, every core plays every role ----
    CORE_ARCHIVE = f"{func_prefix}qkv_head_dp_core.a"
    if with_attn or attn_row3:
        HPC_LOCAL, NB_ATTN = Hq // N, max_seq // 32
        # ---- M1, measurement stage: the attention half's types/buffers/kernels, declared and
        # CALLED once each below so the merged core's .text is a real compile number rather than the
        # sum of the two designs'.  The arguments mirror decode_attn/design.py exactly (its
        # core_body is 25 lines and takes the same buffers), so the functional graft reuses these
        # declarations as-is.  attn's kernels live in one object (decode_attn.o, four functions);
        # op_ours adds it to this op's archive when with_attn is set.
        HPC = Hq // N                       # heads per column (attn's mapping: H/cols)
        assert Hq % N == 0 and Hkv % N == 0
        assert max_seq % B_KV == 0
        NB = max_seq // B_KV
        A_Q_ty = np.ndarray[(HPC, HD), np.dtype[BF16]]
        A_KV_ty = np.ndarray[(2 * B_KV, HD), np.dtype[BF16]]
        _f32, _i32 = np.float32, np.int32
        A_SC_ty = np.ndarray[(HPC, B_KV), np.dtype[_f32]]
        A_OA_ty = np.ndarray[(HPC, HD), np.dtype[_f32]]
        A_ML_ty = np.ndarray[(HPC,), np.dtype[_f32]]
        A_OBJ = f"{func_prefix}decode_attn.o"
        a_scores_k = Kernel(f"{func_prefix}attn_scores_block", A_OBJ,
                            [A_Q_ty, _i32, A_KV_ty, A_SC_ty, _i32, _i32, _f32])
        a_online_k = Kernel(f"{func_prefix}attn_online_block", A_OBJ,
                            [A_SC_ty, _i32, A_KV_ty, A_OA_ty, A_ML_ty, A_ML_ty,
                             _i32, _i32])
        a_fin_k = Kernel(f"{func_prefix}attn_finalize", A_OBJ,
                         [A_OA_ty, _i32, A_ML_ty, A_Q_ty])
        a_zero_k = Kernel(f"{func_prefix}attn_zero_state", A_OBJ,
                          [A_OA_ty, A_ML_ty, A_ML_ty, _i32])
        # Channel folding: two extra call-site shapes of the generic copy.  A func.func symbol is
        # keyed by NAME, so each shape needs its own renamed object (op_ours emits them with
        # prefix_symbols, the same mechanism swiglu_mlp_dp's cx_copy_obj/oa_copy_obj use):
        #   wtd_ : (MO_ty dst, WTILE_ty src) -- the five constants arriving on the weight channel
        #          (one object each, dst_offset 0, so no src_offset is needed anywhere);
        #   qc_  : (A_Q_ty dst, HD_ty src)   -- the q->L1 handoff of the RoPE output.
        copy_wt_k = Kernel(f"{func_prefix}wtd_copy_offset_bf16_vector", f"{func_prefix}wt_d_copy.o",
                           [D_ty, WTILE_ty, np.int32, np.int32])
        copy_wt_hd_k = Kernel(f"{func_prefix}wth_copy_offset_bf16_vector",
                              f"{func_prefix}wt_hd_copy.o", [MO_ty, WTILE_ty, np.int32, np.int32])
        copy_q_k = Kernel(f"{func_prefix}qc_copy_offset_bf16_vector", f"{func_prefix}q_copy.o",
                          [A_Q_ty, HD_ty, np.int32, np.int32])
        # FIFOs for the attention half.  KV blocks come from DRAM through decode_attn's own inKV
        # fifo + MemTile forward (col c, row 1) -- that forward is the deadlock fix from 8fd5cc2
        # and must not be simplified away.  q has NO fifo under M1: the qkv section writes it into
        # this column's q L1 buffer, which is what frees the second input channel for KV.
        A_OB_ty = np.ndarray[(HPC, HD), np.dtype[BF16]]
        attn_inKV_f = [ObjectFifo(A_KV_ty, name=f"attn_inKV_{c}", depth=1) for c in range(N)]
        attn_memKV = [
            attn_inKV_f[c].cons().forward(name=f"attn_memKV_{c}", tile=Tile(col=c, row=1),
                                          dims_to_stream=[(2 * B_KV, HD), (HD, 1)], depth=1)
            for c in range(N)
        ]
        attn_outO_f = [ObjectFifo(A_OB_ty, name=f"attn_outO_{c}", depth=2) for c in range(N)]
        # S_kv_eff: the attention mask boundary, the same scratchpad parameter decode_attn uses.
        attn_seq_param = ScratchpadParameter("S_kv_eff", np.int32)
        # Shim-side endpoints must be built EXPLICITLY with a tile, exactly as
        # decode_attn/design.py does (`rt_handles = [inQ[c].prod(tile=Tile(col=c, row=0)), ...]`):
        # calling .prod() lazily inside the sequence body is too late -- Runtime resolves the
        # endpoints before it traces the sequence and fails with "Prod endpoint not set".
        attn_handles = ([attn_inKV_f[c].prod(tile=Tile(col=c, row=0)) for c in range(N)]
                        + [attn_outO_f[c].cons(tile=Tile(col=c, row=0)) for c in range(N)])
        attn_scale = math.log2(math.e) / math.sqrt(HD)

    copy_kernel = Kernel(
        f"{func_prefix}copy_offset_bf16_vector", CORE_ARCHIVE, [D_ty, MO_ty, np.int32, np.int32]
    )
    # Two bindings of weighted_rms_norm at two widths. One Kernel() fixes ONE func.func signature
    # per symbol, so the HD-wide call site gets its own symbol from a prefixed object -- the same
    # mechanism swiglu_mlp_dp uses for its two matvec DIM_Ks, rather than a local copy of the
    # vendored kernel under a second name (which is what fuse/qkv-head did).
    wnorm_d_kernel = Kernel(
        f"{func_prefix}weighted_rms_norm_fixed", CORE_ARCHIVE,
        [D_ty, D_ty, D_ty, np.float32]
    )
    # The gamma/lut args are declared at the MISC OBJECT width (they are fifo objects,
    # so their type is what the object is): the C kernels read their first HD
    # elements, and the host pads each constant to a whole object.  Declaring them
    # HD would break the fused arena's exact-size assert instead.
    wnorm_hd_kernel = Kernel(
        f"{func_prefix}hd_weighted_rms_norm_fixed", CORE_ARCHIVE,
        [HD_ty, MO_ty, HD_ty, np.float32]
    )
    # OUR int8 matvec (mv_int8.cc, mode14 premul + VEC128): same (m, row_off,
    # a_tile, b, c) call shape as the bf16 mv, a_tile reinterpreted as the
    # [m*K u8 | m*(K/128)*2B scales] byte stream by the kernel.
    mv_kernel = Kernel(
        f"{func_prefix}matvec_vectorized_int8_bf16", CORE_ARCHIVE,
        [np.int32, np.int32, WTILE_ty, D_ty, HD_ty],
    )
    rope_kernel = Kernel(
        f"{func_prefix}rope", CORE_ARCHIVE, [HD_ty, MO_ty, HD_ty, np.int32]
    )

    # `depth=3` is what lets the core hold n_qn, n_kn and ang simultaneously (`.acquire(3)`); the
    # D/MO chunks before them stream through one at a time. Broadcast to N cores via N `.cons()`
    # handles -- the fan-out is in the stream-switch fabric, not the producer's own DMA.
    #
    # misc_memtile would put a MemTile between the shim and the cores, so the SHIM side
    # works in D-wide objects (one BD, one fill, per tensor) while the core side keeps
    # the MO-wide objects the L1 budget wants -- 3 fills per layer instead of 13.
    # MEASURED: THE PLACER REFUSES IT, and the reason is structural rather than a
    # tuning problem.  A MemTile sits in one column and is only connected to that
    # column's shim tile, so a fan-out has to be one shim fifo per column; giving the
    # broadcast ONE shim fifo whose 8 destinations are MemTiles makes aiecc co-locate
    # all 8 endpoints on a single MemTile, which has 6 input + 6 output DMA channels:
    #
    #   error: tile (0, 1) requires 8 input/8 output DMA channels, but only
    #          6 input/6 output available
    #
    # (pinning the consumer handles with `cons(tile=Tile(col=c, row=1))` as well does
    # not change it).  The routable alternative -- 8 per-column shim fifos, one fill
    # each -- costs 24 fills per layer, i.e. WORSE than the 13 this design already
    # issues.  So a MemTile cannot stage this channel; `misc_memtile` stays as the
    # opt-in record of that finding and is not usable.
    # `dims_to_stream=[(MO, 1)]` (pairs of (size, stride), highest dimension first) is
    # the 1-D form of decode_attn's inKV/memKV forward.
    #
    # What bounds the DIRECT path instead: aiecc counts a fill's OBJECTS against the
    # 16 simultaneously-active BDs a shim tile allows, and the misc tile already
    # carries the weight fill plus HEADS_PER_CORE drains (7), so the fills in the
    # shared TaskGroup may cover at most ~9 objects.  That is why D/2 objects work
    # (2+2+3 = 7), D/4 objects do not (4+4+3 = 11 -> the same error as above), and why
    # the shim cannot simply be handed a D-wide fill.
    if with_attn:
        # ---- M1 CHANNEL FOLDING (see docs/notes/2026-09-15-flm-parity-int8-limit.md §7) ----
        # A compute tile has exactly 2 input DMA channels, and the shipped QKV core already uses
        # both (misc + weight); attention adds inQ + inKV.  Merging the two phases therefore
        # requires FOLDING a stream, not just grafting a core body:
        #   * the five constants (cur, n_in, n_qn, n_kn, ang) ride the WEIGHT channel as its first
        #     five objects -- one fill each, issued before the weight fill in the same TaskGroup,
        #     from source BOs declared at the WEIGHT OBJECT width (the tap's bound check is
        #     against the declared tensor length, so the host pads them: the same "constants
        #     ride padded objects" trick this design already uses for the misc channel);
        #   * q never leaves the chip: the qkv section writes it into this column's attention-q
        #     L1 buffer (attn_colmajor guarantees the column owns the q heads attention needs);
        #   * KV keeps decode_attn's own inKV fifo + MemTile forward (the 8fd5cc2 deadlock fix)
        #     EXACTLY as it is -- that is the whole reason this folding beats the misc-time-mux
        #     alternative, which would have had to rebuild the KV path on a new channel.
        # Inputs per core become weight(1) + KV(1) = 2/2; outputs out(1) + context(1) = 2/2.
        assert MO == HD, (
            f"M1 folds the constants onto the weight channel, so the const L1 buffers are declared "
            f"MO_ty and must coincide with HD_ty (MO={MO}, HD={HD})")
        assert not misc_memtile and not merge_drains, (
            "M1 assumes the direct misc path is gone and per-head drains (merge_drains widens the "
            "out fifo's objects, which the q->L1 handoff would have to be taught about)")
        misc_cons = [None] * N
        misc_prod = None
    elif misc_memtile:
        misc_of = ObjectFifo(D_ty, name="misc_shim", depth=2)
        misc_mem = [
            misc_of.cons(tile=Tile(col=c, row=1)).forward(
                name=f"misc_mem_{c}", tile=Tile(col=c, row=1), obj_type=MO_ty,
                depth=3, dims_to_stream=[(MO, 1)],
            )
            for c in range(N)
        ]
        misc_cons = [f.cons() for f in misc_mem]
        misc_prod = misc_of.prod()
    else:
        misc_of = ObjectFifo(MO_ty, name="misc", depth=MISC_DEPTH)
        misc_cons = [misc_of.cons() for _ in range(N)]
        misc_prod = misc_of.prod()
    # M2 row 4's own consumer handles on the SAME fifo: a fifo with two consumers BROADCASTS every
    # object to both (measured in probe_m2_fanout.py), which is how the layer's residual `x`
    # reaches row 4 without a channel of its own -- and, with the row-4 refills also on this
    # fifo, why row 2 has to walk the tail objects too (a producer's empty lock needs EVERY
    # consumer to release, so a consumer that stops early hangs the next fill).
    misc_cons4 = [misc_of.cons(tile=Tile(col=c, row=4)) for c in range(N)] if mlp_row4 else None
    weight_ofs = [ObjectFifo(WTILE_ty, name=f"weight_{c}", depth=weight_depth)
                  for c in range(N)]
    out_ofs = [ObjectFifo(HD_ty, name=f"out_{c}", depth=2) for c in range(N)]

    # ---- M2 ROW 4's types and its two DRAM-facing fifos ----------------------------------------
    # The types are declared here because the fifos need them; the worker itself is built at the
    # end of this function_ (the `MLP` dict is the one handle it needs).
    # The weight stream is this row's OWN channel, not a `split()` of row 2's: a group object is
    # atomic and the two rows' tile counts differ (43+64+64+64 objects vs 8 heads * 32 tiles), so
    # a shared stream is lockstep by construction -- measured as a deadlock in
    # probe_m2_channels.py.  The out stream carries FIVE objects per column, not one: the a-slice,
    # the N_GH_ROUNDS gh rounds and the final residual all leave through it, exactly as the
    # shipped fuse_o op drains three different buffers through its own out fifo.
    MLP = None
    if mlp_row4:
        MTILE_ty = np.ndarray[(MLP_WTILE,), np.dtype[BF16]]
        DPC_ty = np.ndarray[(D_PER_CORE,), np.dtype[BF16]]
        MLP = {
            "D_ty": D_ty,
            "QD_ty": np.ndarray[(QD_L,), np.dtype[BF16]],
            "FF_ty": np.ndarray[(FF,), np.dtype[BF16]],
            "FFPC_ty": np.ndarray[(FF_PER_CORE,), np.dtype[BF16]],
            "DPC_ty": DPC_ty,
            "OWIN_ty": np.ndarray[(O_WINDOW,), np.dtype[BF16]],
            "MTILE_ty": MTILE_ty,
            "archive": CORE_ARCHIVE,
            "wmlp_ofs": [ObjectFifo(MTILE_ty, name=f"wmlp_{c}", depth=MLP_WD_DEPTH)
                         for c in range(N)],
            "xout_ofs": [ObjectFifo(DPC_ty, name=f"xout_{c}", depth=2) for c in range(N)],
        }
        MLP["wmlp_ps"] = [of.prod() for of in MLP["wmlp_ofs"]]
        MLP["xout_cs"] = [of.cons() for of in MLP["xout_ofs"]]

    def core_fn(misc_c, weight_c, out_p, cur_buf, nin_buf, hn_buf, raw_buf, nrm_buf,
                nwt_b, copy_k, wnorm_d_k, wnorm_hd_k, mv_k, rope_k, kinds, attn_bufs=None,
                a_scores_k=None, a_online_k=None, a_fin_k=None, a_zero_k=None, NB=0,
                attn_fifos=None, attn_seq_param=None, attn_scale=1.4427, oa_b=None, sc_b=None,
                m_b=None, l_b=None, consts_on_weight=False, q_to_l1=False, copy_wt_k=None,
                copy_wt_hd_k=None, copy_q_k=None, nqn_b=None, nkn_b=None, ang_b=None,
                attn_q_b=None, q_rows=None):
        # The per-head weight-tile loop bound.  A Python constant here is emitted as a
        # constant scf.for bound and LLVM unrolls it FULLY: ~143 B of .text per mv
        # call site, and there are HEADS_PER_CORE * N_W_TILES of them (96 at
        # tsi=8/6 heads for D=2560).  That is what put this five-kernel core at
        # .text 16896 B against the 16 KB program memory, forcing tsi=16/depth=1 --
        # i.e. giving up the weight fifo's prefetch.  Reading the bound from an L1
        # buffer (mha's runtime-parameter idiom) makes it an SSA value, so the loop
        # body is emitted once.  The buffer's value is baked in at compile time; the
        # load is a real memory access, so nothing can fold it back to a constant.
        nwt = nwt_b[0] if runtime_tile_loop else N_W_TILES

        if consts_on_weight:
            # M1 folding: the five constants are the FIRST FIVE objects of this core's weight
            # stream (op_ours' arg order: x_pad, W_norm1, W_qn, W_kn, rope_angles; the sequence
            # issues their fills before the weight fill, in the same TaskGroup).  Each is one
            # weight-object-wide blob holding its data at the head, copied with dst_offset 0 --
            # so the generic copy's missing src_offset never comes up.
            o = weight_c.acquire(1)
            copy_wt_k(cur_buf, o, D, 0)
            weight_c.release(1)
            o = weight_c.acquire(1)
            copy_wt_k(nin_buf, o, D, 0)
            weight_c.release(1)
            o = weight_c.acquire(1)
            copy_wt_hd_k(nqn_b, o, HD, 0)
            weight_c.release(1)
            o = weight_c.acquire(1)
            copy_wt_hd_k(nkn_b, o, HD, 0)
            weight_c.release(1)
            o = weight_c.acquire(1)
            copy_wt_hd_k(ang_b, o, HD, 0)
            weight_c.release(1)
            nqn_t, nkn_t, ang_t = nqn_b, nkn_b, ang_b
        else:
            # step 1: rebuild cur and n_in from D/MO chunks, then hn = weighted_RMSNorm(cur, n_in).
            # `hn` never reaches DDR -- it is what the four fused ops used to hand each other
            # through it.
            for i in range_(N_MISC_CHUNKS):
                ch = misc_c.acquire(1)
                copy_k(cur_buf, ch, MO, i * MO)
                misc_c.release(1)
            for i in range_(N_MISC_CHUNKS):
                ch = misc_c.acquire(1)
                copy_k(nin_buf, ch, MO, i * MO)
                misc_c.release(1)
            # step 2: n_qn, n_kn, ang -- read once per head, so acquired once and held to the end.
            # With the MemTile the three constants are the first three MO pieces of ONE
            # D-wide object (the host puts each at a piece boundary), and the object's
            # remaining pieces are drained after the head loop so the fifo accounting
            # stays whole.
            w3 = misc_c.acquire(3)
            nqn_t, nkn_t, ang_t = w3[0], w3[1], w3[2]
        wnorm_d_k(cur_buf, nin_buf, hn_buf, epsilon)

        # step 3: this core's heads, in weight-row order. `kinds` is a Python list, so the branch
        # is resolved while tracing and each core emits only the code its own heads need.
        # `q_to_l1` diverts the q heads on chip: the q still goes through the out fifo -- so it is
        # drained to DRAM exactly as before (one more object per column, and the q buffer stays
        # inspectable, which is what makes the handoff debuggable) -- and is ALSO copied into this
        # column's attention-q row (q_rows[si]; attn_colmajor makes the slot order the attention
        # kernel's h order), so the attention half needs no q fifo of its own.
        for si, kind in enumerate(kinds):
            out_t = out_p.acquire(1)
            dst = raw_buf if kind != "v" else out_t   # v heads land straight in the drain tile
            for j in range_(nwt):
                row_off = index.casts(T.i32(), j) * tsi
                wt = weight_c.acquire(1)
                mv_k(tsi, row_off, wt, hn_buf, dst)
                weight_c.release(1)
            if kind != "v":
                wnorm_hd_k(raw_buf, nqn_t if kind == "q" else nkn_t, nrm_buf, epsilon)
                rope_k(nrm_buf, ang_t, out_t, HD)
            if kind == "q" and q_to_l1:
                copy_q_k(attn_q_b, out_t, HD, q_rows[si] * HD)
            out_p.release(1)

        if with_attn:
            # ---- decode_attn's core_body (design.py:219), grafted onto the end of the QKV section:
            # the two halves are strictly sequential, so no data crosses columns.  KV blocks arrive
            # one at a time through the MemTile forward and the online-softmax state lives in this
            # column's L1; q is NOT a fifo under M1 -- the qkv section above already wrote it into
            # this column's attention-q buffer through the q->L1 handoff.
            kv_f, o_f = attn_fifos
            attn_seq = attn_seq_param.read() if attn_seq_param is not None else max_seq
            _o = o_f.acquire(1)
            for h in range_(HPC):
                a_zero_k(oa_b, m_b, l_b, h)
            for b in range_(NB):
                kvb = kv_f.acquire(1)
                for h in range_(HPC):
                    a_scores_k(attn_q_b, h, kvb, sc_b, b, attn_seq, attn_scale)
                    a_online_k(sc_b, h, kvb, oa_b, m_b, l_b, b, attn_seq)
                kv_f.release(1)
            for h in range_(HPC):
                a_fin_k(oa_b, h, l_b, _o)
            o_f.release(1)

        if not consts_on_weight:
            misc_c.release(3)
            if misc_memtile:
                # drain the padding pieces of the packed constants object
                for _ in range_(N_MISC_CHUNKS - 3):
                    pad = misc_c.acquire(1)
                    misc_c.release(1)
            # M2 row 4 shares this fifo, so its objects (the context, n_pf and the two
            # all-gather refills) arrive behind the five this core wants.  They are not this
            # core's, but the producer's empty lock needs BOTH consumers to release every
            # object: skipping them here is what keeps the next fill from hanging.  MEASURED
            # consequence in probe_m2_row4.py if this loop is dropped: the fills after object 5
            # never complete and the dispatch dies with ERT_CMD_STATE_TIMEOUT.
            for _ in range_(MISC_ROW4_SKIP):
                o = misc_c.acquire(1)
                misc_c.release(1)

    def head_kind(g):
        return "q" if g < Hq else ("k" if g < Hq + Hkv else "v")

    def col_slots(c):
        """(kind, q_head_index, kv_head_index) for each local slot of column c; exactly one of the
        two indices is set. DEFAULT = contiguous rows of the concatenated [Q|K|V] weight, which is
        what makes a column's weight slice contiguous (cols 0-3 all Q, 4-5 K, 6-7 V).
        `attn_colmajor` = the head set decode_attn needs from this column, [q_2c, q_2c+1, k_c, v_c],
        with the host permuting the Wqkv rows to match (probe_m1_qkv_perm.py). The merged design
        needs this: attn's per-column mapping is per-head, qkv's is per-kind, and only the
        per-head mapping lets one design do both the projections and its own attention."""
        if attn_colmajor:
            qpc, kvpc = Hq // N, Hkv // N
            out = [("q", c * qpc + i, None) for i in range(qpc)]
            out += [("k", None, c * kvpc + i) for i in range(kvpc)]
            out += [("v", None, c * kvpc + i) for i in range(kvpc)]
        else:
            out = []
            for h in range(HEADS_PER_CORE):
                g = c * HEADS_PER_CORE + h
                k = head_kind(g)
                out.append(("q", g, None) if k == "q" else
                           (("k", None, g - Hq) if k == "k" else ("v", None, g - Hq - Hkv)))
        assert len(out) == HEADS_PER_CORE, (len(out), HEADS_PER_CORE)
        return out

    workers = []
    for c in range(N):
        kinds = [k for (k, _q, _kv) in col_slots(c)]
        nin_b = Buffer(D_ty, name=f"nin_{c}")
        if with_attn:
            # The attention half's L1 working set is PER COLUMN (the placer requires every buffer
            # shared across Workers to carry an explicit tile; decode_attn/design.py builds its
            # buffers per column for the same reason).  ~18 KB/column, which is why it is declared
            # only when the graft is on.
            attn_bufs = (
                Buffer(A_Q_ty, name=f"attn_q_{c}"),
                Buffer(A_KV_ty, name=f"attn_kv_{c}"),
                Buffer(A_SC_ty, name=f"attn_scores_{c}",
                       initial_value=np.zeros((HPC, B_KV), dtype=np.float32)),
                Buffer(A_OA_ty, name=f"attn_out_acc_{c}",
                       initial_value=np.zeros((HPC, HD), dtype=np.float32)),
                Buffer(A_ML_ty, name=f"attn_m_{c}", initial_value=np.zeros((HPC,), dtype=np.float32)),
                Buffer(A_ML_ty, name=f"attn_l_{c}", initial_value=np.zeros((HPC,), dtype=np.float32)),
            )
            _q, _kv, _sc, _oa, _m, _l = attn_bufs
            # Channel folding's own L1 (declared MO_ty on purpose: at MO == HD the type is the
            # same as HD_ty, which is what lets the gamma args keep the existing kernel bindings).
            const_bufs = (
                Buffer(MO_ty, name=f"nqn_{c}"), Buffer(MO_ty, name=f"nkn_{c}"),
                Buffer(MO_ty, name=f"ang_{c}"),
            )
            _nqn, _nkn, _ang = const_bufs
            # q_rows[si] = which row of attn_q this column's si-th q slot lands in; under
            # attn_colmajor the slots are [q,q,k,v], so it is 0,1 (attn's h order).
            q_rows = []
            for (k, _qh, _kvh) in col_slots(c):
                q_rows.append(len(q_rows) if k == "q" else -1)
        workers.append(
            Worker(
                core_fn,
                [
                    misc_cons[c], weight_ofs[c].cons(), out_ofs[c].prod(),
                    Buffer(D_ty, name=f"cur_{c}"),
                    nin_b, (nin_b if inplace_norm else Buffer(D_ty, name=f"hn_{c}")),
                    Buffer(HD_ty, name=f"raw_{c}"), Buffer(HD_ty, name=f"nrm_{c}"),
                    Buffer(np.ndarray[(1,), np.dtype[np.int32]], name=f"nwt_{c}",
                           initial_value=np.array([N_W_TILES], dtype=np.int32)),
                    copy_kernel, wnorm_d_kernel, wnorm_hd_kernel, mv_kernel, rope_kernel,
                    kinds, attn_bufs if with_attn else None, a_scores_k if with_attn else None,
                    a_online_k if with_attn else None, a_fin_k if with_attn else None,
                    a_zero_k if with_attn else None, NB if with_attn else 0,
                    (attn_memKV[c].cons(), attn_outO_f[c].prod()) if with_attn else None,
                    # attn_scale, NOT the skeleton's placeholder 1.4427: the attention kernels
                    # expect the base-2 scale log2(e)/sqrt(HD) (decode_attn/design.py passes exactly
                    # that).  Passing log2(e) alone multiplies every score by sqrt(HD) = 11.31 --
                    # measured as a softmax that is 11x more peaked than the reference, which is
                    # invisible in the single-token case (a one-position softmax is 1 either way)
                    # and looks like a "some heads are fine" pattern on real data.
                    # `attn_scale` only exists in the with_attn block; the OFF arm keeps the
                    # placeholder it never uses (its attention half is not built at all).
                    attn_seq_param if with_attn else None,
                    (attn_scale if with_attn else 1.4427),
                    _oa if with_attn else None, _sc if with_attn else None,
                    _m if with_attn else None, _l if with_attn else None,
                    bool(with_attn), bool(with_attn), copy_wt_k if with_attn else None,
                    copy_wt_hd_k if with_attn else None, copy_q_k if with_attn else None,
                    _nqn if with_attn else None,
                    _nkn if with_attn else None, _ang if with_attn else None,
                    _q if with_attn else None, q_rows if with_attn else None,
                ],
                stack_size=stack_size,
                # PINNED, and that is load-bearing in the M2 phase design rather than tidy.
                # `attn_row3`/`mlp_row4` put a worker on every one of rows 2, 3 and 4 of each
                # column, and the shipped design leaves THIS one unpinned (a single-row design has
                # nothing to collide with).  With rows 3 and 4 pinned, an unpinned row 2 gets
                # placed wherever there is room -- MEASURED (M2Row4Op at 4 columns): the row-2
                # worker of column 1 landed on the SAME physical tile as the row-4 worker of
                # column 0.  One tile is one core: it then holds BOTH workers' L1 (which no
                # budget check covers) and sizes its single stack for ONE of them -- 0x800, while
                # row 2 asks for 0xD00.  The overflow lands in the weight fifo's buffer, which is
                # physically adjacent to the stack.  The fingerprint was exactly that: the
                # qkv/attention rows correct (their buffers are elsewhere), row 4's own `hf`
                # eventually correct, and the Wg/Wu tiles -- the ones streaming through the fifo
                # buffer the stack spilled into -- scrambled (gh cos 0.30).
                tile=Tile(col=c, row=2),
            )
        )

    def _misc_fills(misc_p, cur, nin, const_specs, tg):
        """Mode-specific misc fills; `const_specs` is the list of (buffer, tap) pairs for
        the constants (three MO-wide objects in the direct path, one packed D-wide
        object when the MemTile stages the shim side)."""
        if misc_memtile:
            # shim-side objects are D-wide: ONE fill per tensor, ONE BD
            misc_p.fill(cur, _flat_tap(D, D), wait=True, group=tg)
            misc_p.fill(nin, _flat_tap(D, D), wait=True, group=tg)
        elif misc_chunked:
            for i in range(N_MISC_CHUNKS):
                tg_i = TaskGroup()
                misc_p.fill(cur, _flat_tap(D, MO, i * MO), wait=True, group=tg_i)
                tg_i.finish()
            for i in range(N_MISC_CHUNKS):
                tg_i = TaskGroup()
                misc_p.fill(nin, _flat_tap(D, MO, i * MO), wait=True, group=tg_i)
                tg_i.finish()
        else:
            # cur/nin are D-wide, i.e. N_MISC_CHUNKS fifo objects each.  ONE fill hands
            # the core all of them (aiecc lowers it into one BD per destination object,
            # and they are all active at once -- that is what bounds this form: see the
            # misc_memtile comment above).
            misc_p.fill(cur, _flat_tap(D, D), wait=True, group=tg)
            misc_p.fill(nin, _flat_tap(D, D), wait=True, group=tg)
        for arg, tap in const_specs:
            misc_p.fill(arg, tap, wait=True, group=tg)

    def _tail(tg, misc_p, wqkv, q, kv_cache, weight_ps, out_cs):
        # ONE TaskGroup for the fills AND the drains, and that is load-bearing rather than tidy.
        # Runtime.finish_task_group awaits at group CLOSE, not at issue, so a group's BDs are all
        # programmed first -- but a group boundary is a hard barrier. Splitting fills and drains
        # DEADLOCKS this design: a core interleaves weight tiles with head outputs, so with the
        # out fifo at depth 2 it blocks after two heads, its weight fill can never complete, and
        # the fill group's await never returns to issue the drains. MEASURED as
        # ERT_CMD_STATE_TIMEOUT with `Fatal error type: 0x0` -- a hang, not a fault.
        #
        # `wait=True` on every task, not just the drains: a bare finish() with no waited task
        # lowers to dma_free_task, which recycles BD IDs at COMPILE time and emits no hardware
        # wait. BD pools are per shim TILE and shared across every objectFIFO mapped to it, so a
        # later fill can reprogram a descriptor whose transfer is still in flight and desync a
        # lock count -- that hung every N of swiglu_mlp_dp with its own TDR.
        #
        # Per-tile active BDs stay inside the 16 aiecc allows: each column carries 1 weight fill +
        # HEADS_PER_CORE drains, and the misc producer's 5 fills land on one tile.
        for c in range(N):
            # ONE fill per core for the whole slice; the fifo hands it to the core in WTILE_ELEMS
            # pieces. Filling per tile instead would be N_W_TILES*HEADS_PER_CORE BDs per core, and
            # small fills are half of why the spatial version lost.
            weight_ps[c].fill(
                wqkv, _flat_tap(TOT * ROW_BYTES // 2, ROWS_PER_CORE * ROW_BYTES // 2, c * ROWS_PER_CORE * ROW_BYTES // 2),
                wait=True, group=tg,
            )
            # OFF path (merge_drains=False) is byte-identical to what shipped: one drain per head.
            if merge_drains:
                # ONE drain per KIND per core instead of one per head: a core's heads are a
                # contiguous run of the concatenated [q|k|v] weight, so its n heads of one kind
                # are either n consecutive q rows (contiguous: one 2-D tap) or n consecutive KV
                # cache regions (stride 2*max_seq*HD: one 2-D tap with that row stride).  Both are
                # a single fill-sized task over the fifo's n objects, and the core is untouched --
                # it still acquires and releases one out object per head.  48 -> 10 tasks/layer.
                #
                # NOT the "one head's K and V halves are adjacent" merge: a core owns k heads
                # [Hq+hh] and v heads [Hq+Hkv+hh], i.e. its k and v heads are DIFFERENT cache
                # regions (they are 8 heads apart in the concatenated weight, and the contiguous
                # row split never puts a k head and its matching v head in the same core), so no
                # single 256-element tap covers both halves.  Merging by kind needs no object
                # widening either -- the out fifo stays HD-wide.
                # The merged drain must group by the head mapping that ACTUALLY decides which
                # head each slot computes -- `col_slots(c)` -- not by the contiguous-row mapping
                # `c*HEADS_PER_CORE + h`.  Under `attn_colmajor` a column owns [q_2c..q_2c+1, k_c,
                # v_c] while the contiguous mapping says [heads 8c..8c+7]; the two agree only for
                # column 0.  MEASURED before this fix (probe_m2_layer.py, 4 columns, acm+md):
                # q per-head cos = 1.000 for heads 0-3 (= column 0's, where the mappings coincide)
                # and garbage for 4-15 -- the column's q objects were being drained to the
                # contiguous head slots.  Identical groups when acm is off, so this is a no-op for
                # the shipped arm.
                groups = {}
                for (kind, _qh, _kvh) in col_slots(c):
                    g = (_qh if _qh is not None
                         else (Hq + _kvh if kind == "k" else Hq + Hkv + _kvh))
                    groups.setdefault(kind, []).append(g)
                for kind, gs in groups.items():
                    n_h = len(gs)
                    if kind == "q":
                        out_cs[c].drain(
                            q, _tile_tap(QD, n_h, HD, gs[0] * HD), wait=True, group=tg)
                    else:
                        hh0 = (gs[0] - Hq) if kind == "k" else (gs[0] - Hq - Hkv)
                        half = HD if kind == "v" else 0
                        out_cs[c].drain(
                            kv_cache,
                            TensorAccessPattern(
                                (1, Hkv * 2 * max_seq * HD),
                                hh0 * 2 * max_seq * HD + half,
                                [1, 1, n_h, HD],
                                [0, 0, 2 * max_seq * HD, 1],
                            ),
                            wait=True, group=tg, offset_parameter=kv_off_param,
                        )
            for (kind, _q_h, _kv_h) in (() if merge_drains else col_slots(c)):
                if kind == "q":
                    # q keeps a plain L3 buffer: the scores GEMV reads it whole, per token.
                    out_cs[c].drain(q, _flat_tap(QD, HD, _q_h * HD), wait=True, group=tg)
                else:
                    # OUR kv layout: the chain's head-major per-token-[K_t|V_t]
                    # cache. Head hh occupies a 2*max_seq*HD region (offset
                    # hh*2*max_seq*HD); within it token t's K sits at t*2*HD
                    # and V at t*2*HD + HD. The static head-segment base and
                    # the V half-offset ride the tap; the position term is
                    # kv_off = t*2*HD (the host's existing k_off value),
                    # patched per dispatch — identical for both kinds.
                    hh = _kv_h
                    seg = hh * 2 * max_seq * HD
                    half = HD if kind == "v" else 0
                    out_cs[c].drain(
                        kv_cache, _flat_tap(Hkv * 2 * max_seq * HD, HD, seg + half),
                        wait=True, group=tg, offset_parameter=kv_off_param,
                    )
        tg.finish()

    if with_attn:
        # M1's sequence: the constants ride the WEIGHT channel (no misc fifo exists), so the
        # signature has no misc endpoint, and -- because the K|V drains live inside `_tail`'s
        # TaskGroup while attn's KV fills read that same cache -- the second TaskGroup is what
        # gives the "this token's K|V landed before scoring" memory order.  Same-group BDs have no
        # ordering, and splitting `tg` itself deadlocks the design (see _tail's comment).
        def sequence(cur, nin, wqkv, nqn, nkn, ang, q, kv_cache, weight_ps, out_cs,
                     ctx_out, attn_h):
            if kv_off_param is not None:
                sync_parameters()
            tg = TaskGroup()
            # FIVE fills per column, in op_ours' arg order (x_pad, W_norm1, W_qn, W_kn, ang),
            # each one whole weight-object-wide blob whose data sits at the head of a source BO
            # declared at exactly that width.  Issued BEFORE the weight fill (below, inside
            # `_tail`) because a fifo hands its objects over in fill order -- and the core's first
            # five acquires are these constants.
            for c in range(N):
                for buf in (cur, nin, nqn, nkn, ang):
                    weight_ps[c].fill(buf, _flat_tap(WTILE_ELEMS, WTILE_ELEMS),
                                      wait=True, group=tg)
            _tail(tg, None, wqkv, q, kv_cache, weight_ps, out_cs)

            h_inKV = attn_h[:N]
            h_outO = attn_h[N:]
            tg2 = TaskGroup()
            for c in range(N):
                # ONE fill per column for the WHOLE interleaved head; the MemTile forward streams
                # it into per-block (2*B_KV, HD) tiles on the core.  Exactly decode_attn's own fill
                # (8fd5cc2): per-block fills would be NB*cols BDs, which is what deadlocked there.
                h_inKV[c].fill(kv_cache,
                               _flat_tap(Hkv * max_seq * HD * 2, max_seq * 2 * HD,
                                         c * max_seq * 2 * HD),
                               wait=True, group=tg2)
                h_outO[c].drain(ctx_out,
                                _flat_tap(QD, HPC_LOCAL * HD, c * HPC_LOCAL * HD),
                                wait=True, group=tg2)
            tg2.finish()

        const_tys = []
    elif misc_memtile:
        def sequence(cur, nin, wqkv, const, q, kv_cache, misc_p, weight_ps, out_cs):
            if kv_off_param is not None:
                sync_parameters()
            tg = TaskGroup()
            # One packed D-wide constants object: n_qn / n_kn / ang sit at the start of
            # its first three MO-wide pieces (which is what the core reads).
            _misc_fills(misc_p, cur, nin, [(const, _flat_tap(D, D))], tg)
            _tail(tg, misc_p, wqkv, q, kv_cache, weight_ps, out_cs)

        const_tys = [D_ty]
    else:
        # `ctx_out`/`attn_h` carry defaults so the OFF path (whose Runtime arg list has neither)
        # calls this signature unchanged; the ON path passes them.
        def sequence(cur, nin, wqkv, nqn, nkn, ang, q, kv_cache,
                     misc_p, weight_ps, out_cs, ctx_out=None, attn_h=None,
                     npf=None, wo=None, wg=None, wu=None, wd=None, a_sc=None, gh_sc=None,
                     xout=None, wmlp_ps=None, xout_cs=None):
            if kv_off_param is not None:
                sync_parameters()
            tg = TaskGroup()
            # The constants are MO-wide L3 buffers (the host pads them; the kernels read
            # their first HD elements).  One object each.
            _misc_fills(misc_p, cur, nin,
                        [(nqn, _flat_tap(MO, MO)), (nkn, _flat_tap(MO, MO)),
                         (ang, _flat_tap(MO, MO))], tg)
            _tail(tg, misc_p, wqkv, q, kv_cache, weight_ps, out_cs)
            # GROUP BOUNDARY = the memory ordering M1 needs: `_tail` contains the K|V drains (the
            # core finishes its whole QKV section inside tg), and attn's KV block fills below read
            # that same cache -- so they must be issued only after tg closes.  Same-group BDs have
            # no ordering, and splitting tg itself deadlocks this design (see _tail's comment).
            if attn_row3 and not with_attn:
                # M2 ROW 3's DRAM traffic: the KV cache in, the context out.  The KV fill is the
                # ONE-fill-per-column form that the MemTile forward then streams into per-block
                # tiles on the core (decode_attn's own, and the 8fd5cc2 deadlock fix); q has NO
                # fill here -- row 2 hands it over live on the out fifo.
                h_inKV = attn_h[:N]
                h_outO = attn_h[N:2 * N]
                tg2 = TaskGroup()
                for c in range(N):
                    # M2 at 4 columns: this column's attention half owns TWO kv heads (Hkv//N = 2)
                    # while decode_attn's 8-column geometry gave it exactly one -- so the fill must
                    # cover heads [2c, 2c+1].  Those two head-segments are CONTIGUOUS in the
                    # head-major cache, so it is still ONE fill: 2 * (2*max_seq*HD) elements at
                    # 2c * (2*max_seq*HD).  The MemTile forward then hands the core
                    # head 2c's NB blocks followed by head 2c+1's NB blocks, which is the order
                    # `attn_row_body`'s GQA loop consumes them in.
                    h_inKV[c].fill(kv_cache,
                                   _flat_tap(Hkv * max_seq * HD * 2, 2 * max_seq * 2 * HD,
                                             c * 2 * max_seq * 2 * HD),
                                   wait=True, group=tg2)
                    if ctx_drain:
                        h_outO[c].drain(ctx_out,
                                        _flat_tap(QD, HPC_LOCAL * HD, c * HPC_LOCAL * HD),
                                        wait=True, group=tg2)
                tg2.finish()
                # ---- M2 ROW 4's DRAM traffic (fuse_o: o-projection + SwiGLU MLP) ----
                # Four groups, each one barrier, in the row-4 core's own dependency order.  The
                # rule each of them has to satisfy (design_ours_int8.py's fuse_o post-mortem):
                # every task in group k must be REACHABLE by the core using only groups <= k.
                # That is why the Wo fills are here rather than in `tg` -- row 4 cannot start its
                # first matvec until the context has been through attention -> DRAM -> misc, so a
                # Wo fill in `tg` would wait on a consumer that a LATER group's drain still holds
                # back: the measured ERT_CMD_STATE_TIMEOUT of the shipped fuse_o arm.
                if mlp_row4:
                    # ---- WEIGHT FILLS: ONE CHUNK OF 16 OBJECTS PER COLUMN PER TASKGROUP --------
                    # NOT the fix for anything -- it was added while chasing the row-wise
                    # corruption below and measured NEUTRAL (identical device numbers before and
                    # after), which is part of what ruled the fill shape out.  It is kept because
                    # it bounds what each TaskGroup asks a shim tile to hold: 4 columns x 16
                    # objects is exactly the 16-simultaneously-active-BD budget aiecc's own error
                    # message names, where one 64-object fill per column is 4x over it.  The
                    # shipped MLP never exceeds 32 objects per fill (FF_PER_CORE/TSI_GU at 8
                    # columns); at M2's 4 columns that figure doubles.
                    # ONE TaskGroup PER CHUNK, all columns together, and the chunks are issued in
                    # the fifo's object order -- so the core can always consume in order, and the
                    # only cost of the boundary is the prefetch across a chunk edge (which is why
                    # the row-4 weight fifo is depth 1: see the L1 note above).
                    def _wchunk(ps, src, total_units, obj_units, n_obj, ch=16):
                        for k0 in range(0, n_obj, ch):
                            kc = min(ch, n_obj - k0)
                            tgc = TaskGroup()
                            for c in range(N):
                                ps[c].fill(
                                    src, _flat_tap(total_units, kc * obj_units,
                                                   c * n_obj * obj_units + k0 * obj_units),
                                    wait=True, group=tgc)
                            tgc.finish()

                    # tg3: the context (R_CX D-wide misc objects -- a BROADCAST, so every row-4
                    # core sees the whole QD vector), n_pf, then the o-projection's weights.
                    tg3 = TaskGroup()
                    for i in range(R_CX):
                        misc_p.fill(ctx_out, _flat_tap(QD, D, i * D), wait=True, group=tg3)
                    misc_p.fill(npf, _flat_tap(D, D), wait=True, group=tg3)
                    tg3.finish()
                    _wchunk(wmlp_ps, wo, WO_L3_UNITS, MLP_WTILE, N_O_TILES)
                    # the a-slice drain, in its own group: it can only complete once the core has
                    # consumed every Wo tile, which the chunk groups above already provided.
                    tg3b = TaskGroup()
                    for c in range(N):
                        xout_cs[c].drain(a_sc, _flat_tap(D, D_PER_CORE, c * D_PER_CORE),
                                         wait=True, group=tg3b)
                    tg3b.finish()
                    # tg4: the all-gathered `a` back to every core (ONE D-wide misc object), then
                    # the gate/up weights.
                    tg4 = TaskGroup()
                    misc_p.fill(a_sc, _flat_tap(D, D), wait=True, group=tg4)
                    tg4.finish()
                    _wchunk(wmlp_ps, wg, WG_L3_UNITS, MLP_WTILE, N_GU_TILES)
                    _wchunk(wmlp_ps, wu, WG_L3_UNITS, MLP_WTILE, N_GU_TILES)
                    # row 4's N_GH_ROUNDS gh objects, the drains the refill below must see through
                    # a barrier.
                    tg4b = TaskGroup()
                    for c in range(N):
                        for r in range(N_GH_ROUNDS):
                            xout_cs[c].drain(
                                gh_sc,
                                _flat_tap(FF, D_PER_CORE, c * FF_PER_CORE + r * D_PER_CORE),
                                wait=True, group=tg4b)
                    tg4b.finish()
                    # tg5: the all-gathered gh (N_GH_ROUNDS D-wide misc objects), then the down
                    # weights.
                    tg5 = TaskGroup()
                    for i in range(N_GH_ROUNDS):
                        misc_p.fill(gh_sc, _flat_tap(FF, D, i * D), wait=True, group=tg5)
                    tg5.finish()
                    _wchunk(wmlp_ps, wd, WD_L3_UNITS, MLP_WTILE, N_D_TILES)
                    # tg6: the new residual off the chip.
                    tg6 = TaskGroup()
                    for c in range(N):
                        xout_cs[c].drain(xout, _flat_tap(D, D_PER_CORE, c * D_PER_CORE),
                                         wait=True, group=tg6)
                    tg6.finish()
            elif with_attn:
                h_inQ = attn_h[:N]
                h_inKV = attn_h[N:2 * N]
                h_outO = attn_h[2 * N:]
                tg2 = TaskGroup()
                for c in range(N):
                    h_inQ[c].fill(q, _flat_tap(QD, HPC_LOCAL * HD, c * HPC_LOCAL * HD),
                                  wait=True, group=tg2)
                    for b in range(NB_ATTN):
                        h_inKV[c].fill(
                            kv_cache,
                            # total = the WHOLE cache (Hkv*max_seq*HD*2), not one head's segment:
                            # the tap's first argument is the tensor's length, its third the offset
                            # into it (passing the segment length trips "Offset too large").
                            _flat_tap(Hkv * max_seq * HD * 2, 2 * B_KV * HD,
                                      c * max_seq * 2 * HD + b * B_KV * 2 * HD),
                            wait=True, group=tg2)
                    h_outO[c].drain(ctx_out,
                                    _flat_tap(QD, HPC_LOCAL * HD, c * HPC_LOCAL * HD),
                                    wait=True, group=tg2)
                tg2.finish()

        const_tys = [MO_ty, MO_ty, MO_ty]

    if with_attn:
        # M1's arg list: the five constants are WTILE-wide (op_ours declares them so), and there is
        # no misc endpoint at all -- which is the whole point of the folding.
        # `context` is declared at the GLOBAL (Hq*HD,) width, not at the outO fifo's per-column
        # object width: it is the buffer the op's surface names, and each column's drain is a
        # 256-element WINDOW of it (the tap carries that), exactly as decode_attn does with L3_O_ty.
        assert Q_L3_ty.__args__[0][0] == Hq * HD, Q_L3_ty.__args__[0]
        rt_args = [WTILE_ty, WTILE_ty, W_L3_ty, WTILE_ty, WTILE_ty, WTILE_ty, Q_L3_ty, KV_L3_ty]
        rt = Runtime(
            sequence,
            [
                *rt_args,
                [of.prod() for of in weight_ofs], [of.cons() for of in out_ofs],
                Q_L3_ty, attn_handles,
            ],
        )
    else:
        _rt = [
            D_ty, D_ty, W_L3_ty, *const_tys, Q_L3_ty, KV_L3_ty,
            misc_prod,
            [of.prod() for of in weight_ofs], [of.cons() for of in out_ofs],
        ]
        if attn_row3:
            _rt += [Q_L3_ty, attn_handles]     # ctx_out, then the inKV/outO endpoints
        if mlp_row4:
            # Row 4's buffers, in the order the sequence's own parameters take them:
            #   n_pf, Wo, Wg, Wu, Wd, a_scratch, gh_scratch, x_out
            # (the two buffers the all-gathers pass through are `inout` on the op's surface --
            # they are device-written and device-read, and a host-side re-fill is what
            # zero-initializes them at dispatch start, exactly as in the shipped fuse_o op).
            _rt += [
                D_ty,                                   # n_pf
                np.ndarray[(WO_L3_UNITS,), np.dtype[BF16]],
                np.ndarray[(WG_L3_UNITS,), np.dtype[BF16]],
                np.ndarray[(WG_L3_UNITS,), np.dtype[BF16]],
                np.ndarray[(WD_L3_UNITS,), np.dtype[BF16]],
                D_ty,                                   # a_scratch
                np.ndarray[(FF,), np.dtype[BF16]],      # gh_scratch
                D_ty,                                   # x_out
                MLP["wmlp_ps"], MLP["xout_cs"],
            ]
        rt = Runtime(sequence, _rt)

    # ---- M2 ROW 3: decode_attn's four kernels, as their OWN worker.
    #
    # This is decode_attn/design.py's core_body, unchanged in its data flow, with two differences
    # that M2 forces:
    #   * q arrives LIVE on the out fifo from row 2 (a fifo with two consumers broadcasts every
    #     object to both -- probe_m2_fanout.py -- so the shim still drains the k/v and the q, while
    #     row 3 also sees each object).  The consumer therefore walks ALL HEADS_PER_CORE objects
    #     and copies only the q slots into its q buffer: under attn_colmajor the column's slots are
    #     [q,q,...,k,k,v,v], and `q_rows` (the same list the shipped design builds) says which of
    #     them are q and which attn-local head each becomes.
    #   * the context leaves on the outO fifo for row 4 (live), instead of a DRAM buffer.
    def attn_row_body(qf, kv_f, o_f, qb, oab, scb, mb, lb, q_rows, copy_q_k,
                      a_scores_k, a_online_k, a_fin_k, a_zero_k,
                      attn_seq_param, attn_scale, NB, HPC, GQA, GQA_Q):
        for s, qr in enumerate(q_rows):          # Python list -> unrolled, so `qr` is a constant
            o = qf.acquire(1)
            if qr >= 0:
                copy_q_k(qb, o, HD, qr * HD)
            qf.release(1)
        attn_seq = attn_seq_param.read() if attn_seq_param is not None else max_seq
        _o = o_f.acquire(1)
        for h in range_(HPC):
            a_zero_k(oab, mb, lb, h)
        # GQA: decode_attn's 8-column geometry gave every column exactly ONE kv head, so its body
        # could walk the q heads flat.  At M2's 4 columns a column owns GQA = Hkv//N kv heads and
        # GQA_Q = Hq//Hkv q heads per kv head, so the KV blocks come in per-head groups and each
        # group scores only ITS q heads -- the kernels are unchanged, `h` is still the column-local
        # q head slot (which is what `q_rows` wrote), and oa/m/l are per-q-head so groups do not
        # interfere.  Measured before this fix (probe_m2_layer.py): heads 0,1 right and 2-15 wrong,
        # exactly the fingerprint of scoring every q head against kv head c.
        for g in range(GQA):
            for b in range_(NB):
                kvb = kv_f.acquire(1)
                for i in range(GQA_Q):
                    h = g * GQA_Q + i
                    a_scores_k(qb, h, kvb, scb, b, attn_seq, attn_scale)
                    a_online_k(scb, h, kvb, oab, mb, lb, b, attn_seq)
                kv_f.release(1)
        for h in range_(HPC):
            a_fin_k(oab, h, lb, _o)
        o_f.release(1)

    if attn_row3:
        for c in range(N):
            q_rows = []
            for (k, _qh, _kvh) in col_slots(c):
                q_rows.append(len(q_rows) if k == "q" else -1)
            workers.append(Worker(
                attn_row_body,
                [out_ofs[c].cons(tile=Tile(col=c, row=3)),
                 attn_memKV[c].cons(tile=Tile(col=c, row=3)),
                 attn_outO_f[c].prod(tile=Tile(col=c, row=3)),
                 Buffer(A_Q_ty, name=f"aq_{c}"), Buffer(A_OA_ty, name=f"aoa_{c}"),
                 Buffer(A_SC_ty, name=f"asc_{c}"), Buffer(A_ML_ty, name=f"am_{c}"),
                 Buffer(A_ML_ty, name=f"al_{c}"),
                 q_rows, copy_q_k, a_scores_k, a_online_k, a_fin_k, a_zero_k,
                 attn_seq_param, attn_scale, NB, HPC,
                 Hkv // N, Hq // Hkv],
                tile=Tile(col=c, row=3)))

    # ---- M2 ROW 4's worker: swiglu_mlp_dp's fuse_o core, on the third phase row ----------------
    # Nested so it closes over this function's geometry; everything it owns is in `mlp`.
    def _mlp_row4(mlp, workers):
        BF16_M = np.dtype[BF16]
        D_ty_m = mlp["D_ty"]
        QD_ty = mlp["QD_ty"]
        FF_ty = mlp["FF_ty"]
        FFPC_ty = mlp["FFPC_ty"]
        DPC_ty = mlp["DPC_ty"]
        OWIN_ty = mlp["OWIN_ty"]
        MTILE_ty = mlp["MTILE_ty"]
        ARCH = mlp["archive"]

        # ---- kernels.  A func.func symbol is keyed by NAME only, so every call shape that differs
        # from the qkv row's own needs its own renamed object (op_ours' madd_/mmul_/mcx_/moa_/
        # mgu_/mdown_/mo_ artifacts).  Two symbols are REUSED rather than recompiled because the
        # qkv row already declares them at byte-identical signatures: `weighted_rms_norm_fixed`
        # (its rms_norm_D.o is built -DRMS_COLS=D, the same fixed entry the MLP calls) and
        # `copy_offset_bf16_vector` at (D_ty, MO_ty) -- with MO == D that is exactly the shape the
        # `x` reassembly needs.
        madd_k = Kernel(f"{func_prefix}madd_eltwise_add_bf16_vector", ARCH,
                        [D_ty_m, D_ty_m, D_ty_m, np.int32])
        madd_off_k = Kernel(f"{func_prefix}madd_eltwise_add_offset_a_bf16_vector", ARCH,
                            [D_ty_m, DPC_ty, DPC_ty, np.int32, np.int32])
        mmul_off_k = Kernel(f"{func_prefix}mmul_eltwise_mul_offset_ab_bf16_vector", ARCH,
                            [FFPC_ty, FFPC_ty, DPC_ty, np.int32, np.int32])
        mgh_copy_k = Kernel(f"{func_prefix}madd_copy_offset_bf16_vector", ARCH,
                            [FF_ty, D_ty_m, np.int32, np.int32])
        mcx_copy_k = Kernel(f"{func_prefix}mcx_copy_offset_bf16_vector", ARCH,
                            [QD_ty, D_ty_m, np.int32, np.int32])
        moa_copy_k = Kernel(f"{func_prefix}moa_copy_offset_bf16_vector", ARCH,
                            [DPC_ty, OWIN_ty, np.int32, np.int32])
        msilu_k = Kernel(f"{func_prefix}silu_tile_bf16", ARCH, [np.int32, FFPC_ty])
        mgu_k = Kernel(f"{func_prefix}mgu_matvec_vectorized_int8_bf16", ARCH,
                       [np.int32, np.int32, MTILE_ty, D_ty_m, FFPC_ty])
        mdown_k = Kernel(f"{func_prefix}mdown_matvec_vectorized_int8_bf16", ARCH,
                         [np.int32, np.int32, MTILE_ty, FF_ty, DPC_ty])
        mo_k = Kernel(f"{func_prefix}mo_matvec_vectorized_int8_bf16", ARCH,
                      [np.int32, np.int32, MTILE_ty, QD_ty, OWIN_ty])

        def mlp_row_body(w_c, misc_c, out_p, x_buf, x1_buf, hf_buf, npfb_buf, a_buf, cx_buf,
                         aslice_buf, gh_buf,
                         g_buf, u_buf, d_buf, madd_k, madd_off_k, mmul_off_k, mgh_copy_k,
                         mcx_copy_k, moa_copy_k, msilu_k, mgu_k, mdown_k, mo_k, mwnorm_k,
                         mxc_copy_k, core_id, tsi_gu, tsi_d, tsi_o, n_o_tiles, n_gu_tiles,
                         n_d_tiles, r_gh, r_cx, d_per_core, epsilon, noop):
            """design_ours_int8.py's fuse_o core_fn, verbatim in its DATA FLOW, with the channel
            mapping M2 forces:
              * the misc queue is the SHARED one, so the objects this core does not want (n_in and
                row 2's three constants) are consumed too -- dropping them hangs the producer's
                empty lock, which needs BOTH consumers to release every object;
              * `cur` is copied out of its object instead of being held as the (cur, a) pair the
                shipped op acquires, because in this queue `a` is 7 objects further down;
              * `cx` is reassembled from r_cx D-wide BROADCAST objects (the whole QD) -- see the
                row-4 geometry note: the live per-column context object is only HPC of the Hq
                heads, which cannot feed a Wo whose K is the full context;
              * the two all-gathers keep the shipped op's shape: drain on this row's own out fifo,
                re-read as D-wide misc objects, separated by a TaskGroup barrier each.
            """
            if noop:
                # WIRING PROBE (mlp_noop): the same acquire/release order with no kernels, so a
                # failure here is a channel/placement/TaskGroup fault and not a core-body one.
                o = misc_c.acquire(1)
                misc_c.release(1)
                o = misc_c.acquire(1)
                misc_c.release(1)
                c3 = misc_c.acquire(3)
                misc_c.release(3)
                for _ in range_(r_cx):
                    o = misc_c.acquire(1)
                    misc_c.release(1)
                o = misc_c.acquire(1)                    # n_pf
                misc_c.release(1)
                for _ in range_(n_o_tiles):
                    wt = w_c.acquire(1)
                    w_c.release(1)
                ot = out_p.acquire(1)                    # the a-slice round
                out_p.release(1)
                o = misc_c.acquire(1)                    # a_sc
                misc_c.release(1)
                for _ in range_(2 * n_gu_tiles):
                    wt = w_c.acquire(1)
                    w_c.release(1)
                for _ in range(r_gh):
                    ot = out_p.acquire(1)                # the gh rounds
                    out_p.release(1)
                for _ in range_(r_gh):
                    o = misc_c.acquire(1)                # the gh refill
                    misc_c.release(1)
                for _ in range_(n_d_tiles):
                    wt = w_c.acquire(1)
                    w_c.release(1)
                ot = out_p.acquire(1)                    # the residual round
                out_p.release(1)
                return

            # 1: the layer's residual `x` (= cur).  Copied into L1 and released AT ONCE: holding
            # the object would pin a fifo slot across the whole o-projection and stall row 2's
            # producer (the misc fifo is only MISC_DEPTH deep).
            o = misc_c.acquire(1)
            mxc_copy_k(x_buf, o, D, 0)
            misc_c.release(1)
            # 2: n_in (row 2's) and 3-5: row 2's three constants -- consumed, not used.
            o = misc_c.acquire(1)
            misc_c.release(1)
            misc_c.acquire(3)
            misc_c.release(3)
            # 6/7...: the context, R_CX D-wide broadcasts covering the whole QD.
            for i in range(r_cx):
                chunk = misc_c.acquire(1)
                mcx_copy_k(cx_buf, chunk, D, i * D)
                misc_c.release(1)
            # 8: n_pf -- acquired here because the queue order forces it (the producer hands objects
            # over strictly in order).  COPIED OUT AND RELEASED AT ONCE: this fifo has TWO
            # consumers and the ring is shallow, so holding an object across the o-projection
            # (which is where a later object would land) is how the data gets overwritten under
            # you.  MEASURED: holding it, `hf` reads back at cos 0.69 against the same input
            # (probe_m2_row4.py M2R4_DUMP=hf); copying it, hf is exact.
            npf_o = misc_c.acquire(1)
            mxc_copy_k(npfb_buf, npf_o, D, 0)
            misc_c.release(1)

            # step 0: this core's window of the o-projection: a_slice = Wo[c] @ cx.  A full
            # O_WINDOW-row window (>= D/N), so every fill is a whole TSI_O-row tile; only the
            # first D/N rows are real, the rest is this design's overlap+pad (see op_ours).
            for j in range_(n_o_tiles):
                j32 = index.casts(T.i32(), j)
                wt = w_c.acquire(1)
                mo_k(tsi_o, j32 * tsi_o, wt, cx_buf, aslice_buf)
                w_c.release(1)

            # step 0b: emit only this core's real D/N-wide a-slice (the overlap rows' contribution
            # is computed and dropped, exactly as in the shipped op).
            ot = out_p.acquire(1)
            moa_copy_k(ot, aslice_buf, d_per_core, 0)
            out_p.release(1)

            # step 0c: x1 = cur + a -- the all-gathered `a`, one D-wide misc object (every core
            # reads the SAME object: the misc channel is a broadcast).  Same copy-and-release
            # discipline as n_pf above.
            a_o = misc_c.acquire(1)
            mxc_copy_k(a_buf, a_o, D, 0)
            misc_c.release(1)
            madd_k(x_buf, a_buf, x1_buf, D)

            # step 2: hf = weighted_rms_norm(x1, n_pf) -- the MLP's own epsilon (1e-5), NOT this
            # layer's qkv epsilon.
            mwnorm_k(x1_buf, npfb_buf, hf_buf, epsilon)

            # step 3: g = Wg[this core's rows] @ hf, then u = Wu[same rows] @ hf, on the SAME
            # weight fifo, continued.
            for j in range_(n_gu_tiles):
                j32 = index.casts(T.i32(), j)
                wt = w_c.acquire(1)
                mgu_k(tsi_gu, j32 * tsi_gu, wt, hf_buf, g_buf)
                w_c.release(1)
            for j in range_(n_gu_tiles):
                j32 = index.casts(T.i32(), j)
                wt = w_c.acquire(1)
                mgu_k(tsi_gu, j32 * tsi_gu, wt, hf_buf, u_buf)
                w_c.release(1)

            # step 4: g = silu(g), in place over the whole FF/N slice.
            msilu_k(FF_PER_CORE, g_buf)

            # step 5: emit gh = silu(g)*u, straight out of g_buf/u_buf (no gh-sized scratch) --
            # N_GH_ROUNDS rounds of D/N, each a WHOLE out object.
            for r in range(N_GH_ROUNDS):
                ot = out_p.acquire(1)
                mmul_off_k(g_buf, u_buf, ot, d_per_core, r * d_per_core)
                out_p.release(1)

            # step 5b: the all-gather of gh (down's K is the full FF, so every core needs every
            # column's slice) -- N_GH_ROUNDS D-wide reads of the scratch the drains above filled.
            for i in range(r_gh):
                chunk = misc_c.acquire(1)
                mgh_copy_k(gh_buf, chunk, D, i * D)
                misc_c.release(1)

            # step 6: d = Wd[this core's rows] @ gh.
            for j in range_(n_d_tiles):
                j32 = index.casts(T.i32(), j)
                wt = w_c.acquire(1)
                mdown_k(tsi_d, j32 * tsi_d, wt, gh_buf, d_buf)
                w_c.release(1)

            # step 7: the new residual: x_out[this core's slice] = x1[same slice] + d.
            ot = out_p.acquire(1)
            if mlp_dump:
                # DEBUG ONLY (`mlp_dump`, the same trick the shipped MLP op carries as MLP4B_DUMP):
                # ship one INTERMEDIATE through the residual round instead of x1+d, so the host can
                # see what the core actually computed.  Valid only when d_buf == 0 (run it with a
                # zero Wd wire): with Wd = 0 the down matvec writes exact zeros, so the add below
                # is the identity.  `mlp_dump` is a design kwarg and the fused-MLIR cache does NOT
                # key on kwargs -- these arms need their own build directory.
                madd_off_k({"x": x_buf, "x1": x1_buf, "hf": hf_buf}[mlp_dump], d_buf, ot,
                           d_per_core, core_id * d_per_core)
            else:
                madd_off_k(x1_buf, d_buf, ot, d_per_core, core_id * d_per_core)
            out_p.release(1)

        for c in range(N):
            workers.append(Worker(
                mlp_row_body,
                [mlp["wmlp_ofs"][c].cons(tile=Tile(col=c, row=4)),
                 misc_cons4[c],
                 mlp["xout_ofs"][c].prod(tile=Tile(col=c, row=4)),
                 Buffer(D_ty_m, name=f"mx_{c}"), Buffer(D_ty_m, name=f"mx1_{c}"),
                 Buffer(D_ty_m, name=f"mhf_{c}"),
                 Buffer(D_ty_m, name=f"mnpf_{c}"), Buffer(D_ty_m, name=f"ma_{c}"),
                 Buffer(QD_ty, name=f"mcx_{c}"),
                 Buffer(OWIN_ty, name=f"maslice_{c}"), Buffer(FF_ty, name=f"mgh_{c}"),
                 Buffer(FFPC_ty, name=f"mg_{c}"), Buffer(FFPC_ty, name=f"mu_{c}"),
                 Buffer(DPC_ty, name=f"md_{c}"),
                 madd_k, madd_off_k, mmul_off_k, mgh_copy_k, mcx_copy_k, moa_copy_k,
                 msilu_k, mgu_k, mdown_k, mo_k,
                 # `weighted_rms_norm_fixed` / `copy_offset_bf16_vector` at the signatures row 2
                 # already declared -- the same Kernel objects, not new ones.
                 wnorm_d_kernel, copy_kernel,
                 c, MLP_TSI_GU, MLP_TSI_D, TSI_O, N_O_TILES, N_GU_TILES, N_D_TILES,
                 N_GH_ROUNDS, R_CX, D_PER_CORE, mlp_epsilon, mlp_noop],
                tile=Tile(col=c, row=4), stack_size=MLP_RAIL))

    if mlp_row4:
        _mlp_row4(MLP, workers)

    prog = Program(dev, rt, workers=workers)
    maybe_enable_trace(prog, trace_size, workers)
    return prog.resolve_program()
