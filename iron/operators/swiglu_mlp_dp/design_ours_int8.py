# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OUR-int8 arm of the data-parallel decode SwiGLU MLP (mv_int8.cc mode14, VEC128,\nour [tile_m*K u8 | tile_m*n_g bf16 s] wire; everything else identical to the\nupstream swiglu_mlp_dp design -- see that file for the full dataflow doc): every core runs every stage on its own 1/N slice, unlike
fuse/mlp-block's spatial 5-core PIPELINE (measured +45% slower -- one core streamed all 12.58 MB
of gate+up weights while 27 of 32 cores sat idle). Fusion stays TEMPORAL (one aie.device, one
aiex.configure); parallelism is SPATIAL, across N cores, at every stage:

    x1  = cur + a                             every core, full D (replicated -- cheap, 2 KB)
    hf  = RMSNorm_weighted(x1, n_pf)          every core, full D (replicated -- avoids a reduction)
    g   = Wg[c*FF/N:(c+1)*FF/N] @ hf          core c's own FF/N output rows
    u   = Wu[c*FF/N:(c+1)*FF/N] @ hf          core c's own FF/N output rows
    gh  = silu(g) * u                         core c's own FF/N slice
    -- ALL-GATHER gh (the one unavoidable exchange -- down's K is the full FF) --
    d   = Wd[c*D/N:(c+1)*D/N] @ gh            core c's own D/N output rows
    nxt[c*D/N:(c+1)*D/N] = x1[same] + d       core c's own D/N output rows

At the chunked (4B) arm two things change and both are visible in the runtime args below: the
gate/up weights carry only the model's REAL `ff_real` rows (the core zeroes the rest of its
g_buf/u_buf tail once per dispatch, zero_offset_bf16) and all three weight buffers are built in
`wire4b.wire_layout`'s ROW/COLUMN ORDER -- the gathered gh is core-major with ff_pad/N slots per
core, and Wd's scales are per 128 columns, so every 128-slot block of gh must carry one model
scale-group (which a natural contiguous assignment cannot do at 4B: ff/N = 1216 = 9.5 groups).

N compute tiles means N cores each with the SAME 2-input/2-output DMA-channel budget the
single-core fuse/mlp-block attempt tripped over. Three consolidations make N cores fit it:

  MISC (1 in): cur, a, n_pf are D-shaped, needed once each. The all-gathered gh is FF-shaped, but
  FF is a whole multiple of D (ratio R = FF/D), so it comes back as R separate D-sized reads on
  the SAME channel and is reassembled by an explicit-offset copy kernel. depth=2 lets the core
  hold cur+a simultaneously (`.acquire(2)`) for the first add; every other use is one at a time.
  Broadcast to all N cores via N `.cons()` handles on one producer -- fan-out is in the
  stream-switch fabric, not the source's own DMA (the mechanism fuse/mlp-block's P1 already uses
  to feed both P2 and P3 from one output port).

  WEIGHT (1 in): Wg, Wu (row width D) and Wd (row width FF) are three different L3 buffers, but
  the shared ObjectFifo tile is sized so ONE flat shape serves all three: TSI_GU rows of D
  elements == TSI_D rows of FF elements (TSI_D = TSI_GU // R). Reused sequentially for Wg's tiles,
  then Wu's, then Wd's -- the same "same fifo, several fill() calls" idiom fuse/mlp-block's
  gate/up A-tile stream already uses.

  OUTPUT (1 of the 2 available, one spare): gh's own FF/N slice is emitted in R D/N-sized chunks
  (an offset-mul kernel reading straight out of the g/u buffers, no separate gh-sized scratch) onto
  the SAME small ObjectFifo the final residual reuses for nxt -- R+1 sequential produce/drain
  rounds share one channel.

The gather has no native all-to-all primitive: ObjectFifoLink is many-to-one XOR one-to-many,
never both (confirmed against attn_core's identical wall in this same codebase). It round-trips
through an internal DRAM scratch buffer instead: each core drains its own gh slice in R chunks
(N*R small, disjoint, offset-addressed drains -- no join, no memtile fan-in, since at n_aie_rows=1
each core already owns a whole column's slice), then every core re-reads the FULL scratch buffer
back over the MISC channel. Cost is trivial (2*FF elements moved, ~0.03% of the layer's traffic);
correctness rests only on a TaskGroup barrier separating the drains from the refill.

MEASURED (device-free, aiecc placement): n_aie_rows=1 (N=n_aie_cols<=8) places with plain flat
per-core ObjectFifos -- no explicit MemTile step needed, the automatic placer inserts whatever
staging one column needs (it column-major-fills 4 rows before moving to the next column, so N=8
lands on physical columns 0-1, not 0-7, and that is fine -- the design never assumes a "logical
core c" is "physical column c"). Both N=16 and N=32 fail there: aiecc's error is explicit --
"no ShimNOCTile ... free: all 8 ShimNOCTile(s) are at 16/16 input... channels used" -- the
DEVICE-WIDE ShimDMA budget (16, matching get_shim_dma_limit()) is spent one channel per DISTINCT
shim-facing ObjectFifo, not "2 per tile" as the compute-tile figure might suggest; misc(1) +
weight(N) already exceeds it at N=16.

n_aie_rows>1 fixes this with the SAME two combinators fuse/mlp-block's own report cites as the
proven multi-row pattern in this codebase (whole_array_silu_iron.py's A-split / C-join): per
GROUP of n_aie_rows cores sharing one shim source,
  - WEIGHT: one group-level ObjectFifo (n_aie_rows*WTILE_UNITS per fill) is `.split()` into
    n_aie_rows row sub-fifos (WTILE_ty each) at a MemTile. One fill per weight-tile ROUND gathers
    all n_aie_rows rows' data for that round with a strided TAP (rows are NOT adjacent in Wg/Wd's
    own row-major layout -- consecutive rows in one group are FF_PER_CORE, resp. D_PER_CORE,
    elements apart), correct because ObjectFifoLink's own offsets place them contiguously in the
    fetched tile in row order, matching the `.split()` offsets below.
  - OUTPUT: one group-level ObjectFifo is `.join()` from n_aie_rows row sub-fifos (DPC_ty each);
    cores write into the row sub-fifo exactly as at n_aie_rows=1. Draining gh needs the same
    strided TAP (destination rows are FF_PER_CORE apart) since the join's own buffer is
    contiguous by row; draining the final nxt round does not (D_PER_CORE apart on both sides).
This only reduces the number of DISTINCT shim-facing ObjectFifos from N to n_aie_cols for both
weight and output -- misc is already 1 regardless of n_aie_rows (see the MISC paragraph above).

FUSE_O (fuse_o=True, n_aie_rows=1 only -- see below): folds the attention output projection
`a = Wo @ cx` into this design too, deleting a whole standalone GEMV design/configure/run from the
decode runlist. `a` was the ONLY external input this design didn't already compute on-chip; now
`cx` (QD-wide) and `Wo` ([D, QD]) arrive instead, and every core computes its own D/N slice of `a`
from its own row-slice of Wo, exactly like the existing Wd/d_buf step. Two wrinkles this adds:

  cx reassembly: QD is a whole multiple of D (R_CX = QD/D), so cx arrives as R_CX D-sized misc
  broadcasts and is reassembled by the SAME explicit-offset-copy idiom gh already uses -- no new
  channel, just more rounds through the existing one.

  Wo's row-tile CANNOT share Wg/Wu/Wd's byte-identical WTILE_ty tile cleanly: the shared tile size
  is forced to lcm(D, FF, QD) = 6*D (FF=3D, QD=2D here), which makes TSI_O = 6*D/QD a multiple of
  3, and D_PER_CORE = D/N a power of two for every N this codebase places -- a multiple of 3 can
  never divide a power of two, so a uniform TSI_O-row tiling of D_PER_CORE always leaves a
  remainder, for ANY N. Reusing the channel anyway with a "short" final fill was rejected: nothing
  in this codebase does a partial-tile fill into a fixed-shape ObjectFifo object (see
  qkv_head_dp/design.py's identical refusal, "a D-wide tile would need a 128-of-1024 partial fill
  ... which nothing in this codebase does"), and a dedicated second weight channel for Wo is a
  non-starter on ITS OWN merits: misc(1)+weight(N)+weight_o(N) is 17 input channels at N=8, one
  over the same 16-channel device-wide budget that already caps this design at N=8 (see above) --
  confirmed independently by attn_core's fusion attempt, which hit exactly this wall trying to
  fold op_o in elsewhere ("all 8 ShimNOCTile(s) are at 9/16 input, 16/16 output channels used").

  The fix is a 1-row OVERLAP, not a partial fill: every core reads ceil(D_PER_CORE/TSI_O) FULL
  TSI_O-row tiles (a window of N_O_TILES*TSI_O rows, >= D_PER_CORE), starting at its own
  c*D_PER_CORE offset in Wo. For every core but the last this window simply reads a few of the
  NEXT core's real rows too (harmless -- Wo is read-only, and the extra rows are computed but
  never drained). Only the LAST core's window would run past Wo's true D rows, so Wo is padded
  with O_OVERLAP (< TSI_O) zero rows at the very end -- a single, tiny, explicit append, not an
  assumption about stale buffer contents. Every fill is a full, byte-identical WTILE_ty tile,
  identical in shape to the existing Wg/Wu/Wd fills; only the LAST core's window ever touches a
  padding row, and that row's own output (computed, never drained) is exactly zero. The per-core
  matvec output lands in a plain (n_aie_rows=1-scoped) O_WINDOW-sized scratch buffer, not
  ObjectFifo-backed, so the overlap/pad tail costs nothing beyond that buffer's own bytes; only
  its first D_PER_CORE elements -- this core's real slice -- are drained.

  `a`'s own all-gather reuses gh's exact mechanism: each core drains its D_PER_CORE-wide real
  slice onto the shared OUTPUT channel (now a NEW, first round ahead of gh's own R rounds), then
  every core re-reads the full D-wide result over MISC once the drains are barriered -- structured
  as its own TaskGroup pair (tg_a_drain/tg_a_refill) ahead of the existing gh pair, because this
  core now produces its FIRST output (the a-slice drain) before consuming cur/n_pf, not after (see
  op.py's Runtime docstring for why a stale two-group split would deadlock here).

FUSE_O ON THE CHUNKED (4B) ARM -- Wo is a K-CHUNKED stream, not a row-tiled one.  The row-tiled
scheme above needs the shared object to hold a whole number of BOTH a D-wide row (WROW_D) and a Wo
row (WROW_QD).  At 0.6B that is free (QD = 2D); at 4B QD = 1.6D is not an integer multiple, and the
smallest object satisfying it -- lcm(WROW_D, WROW_QD) = 20800 B = 8 D-rows = 5 Wo rows, which is
ALSO the smallest 64 B-aligned multiple of a 2600 B row -- does not fit this design's L1 at
weight_depth 2 (73,216 B) while at depth 1 it loses its prefetch (29.2 vs 38.0 GB/s, more than the
one eliminated configure is worth).

So the chunked arm leaves the tile geometry ALONE and splits the o-projection along K exactly like
`down`: a chunk row is [K_O u8 | (K_O/128) bf16 scales] bytes long, and the shipped TSI_GU=4 tile
(WTILE_UNITS units, one in-block 64 B pad) holds TSI_O such rows whenever
TSI_O*WROW_O == TSI_GU*WROW_D -- the object is byte-identical in shape to the D-row tiles, only its
K-reading differs.  At 4B: K_O = 2048 (the widest 128-multiple divisor of QD=4096 that keeps
TSI_O = 5 an exact divisor of D_PER_CORE), so 2 chunks x 64 tiles x 5 rows cover each core's
D_PER_CORE rows with no overlap window, and the Wo wire is 10,682,368 B (+0.3% over the standalone
W_o's 10,649,600).  Two consequences worth knowing:

  * the o-matvec is mv_int8_acc.cc's writer/accumulator pair at DIM_K = K_O (one prefixed object
    serves both symbols), i.e. exactly the chunked down matvec's shape -- and it carries the same
    numerical signature: one bf16 rounding of the partial sum between chunks, where the standalone
    GEMV accumulates all of K in one f32 accumulator.  Per element that is a ~1-ultra-level
    difference (measured against a float Wo @ cx reference on the device: cos 0.999997 vs the
    standalone's 0.999998), but 36 layers of bf16 dynamics amplify it into a different trajectory:
    the chain's gate cosine moves while the generated token ids hold for 24 tokens;
  * cx arrives over MISC, whose object is D-wide (cur/a/npf/gh all need that), so the two fills sit
    at the offsets that keep both taps inside the QD-wide context buffer: [0, D) and [QD-D, QD).
    A K_O-wide chunk therefore starts MID-OBJECT (at 4B: the second one starts 512 elements in),
    which is why this arm needs the source-offset copy (copy_off2_bf16.cc) and one K_O-wide cx_buf
    reused per chunk -- a QD-wide buffer would cost 4096 B more of L1 for nothing.
"""

import os

from ml_dtypes import bfloat16
import numpy as np

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, TaskGroup, Worker
from aie.iron.device import Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib.tap import TensorAccessPattern

from iron.operators._trace import maybe_enable_trace

# Shared weight-tile row counts (see module docstring, WEIGHT channel). Fixed, not searched: this
# design is gated at Qwen3-0.6B's D=1024/FF=3072 (R=3) shape only, and 6/2 is verified below to
# fit L1 at every N in {8, 16, 32} this file is built against.
TSI_GU = 6
TSI_D = 2


def _tile_tap(total, n_tiles, tile_units, offset=0):
    """A 2-D tap over `n_tiles` contiguous weight tiles.  Same bytes as `_flat_tap`, but the BD's
    LENGTH stays one tile and the tile count rides the wrap dimension (<= 1023).

    The flat form puts the whole run in the length field, and the L3->L2 (MemTile) path then has
    to expand it: MEASURED with a flat tap at TSI=2 (608 tiles for Wg per column) aiecc fails with
    `'aie.dma_bd' op Allocator exhausted available BD IDs (maximum 24 available for channel 0)`.
    The direct L3->L1 fill tolerates the flat form; the staged one does not."""
    return TensorAccessPattern((1, total), offset,
                               [1, 1, n_tiles, tile_units],
                               [0, 0, tile_units, 1])


def _flat_tap(total, size, offset=0):
    """A contiguous [offset:offset+size) read/write into an L3 buffer of `total` elements.
    `total` is the FULL buffer's own declared size (TensorAccessPattern validates offset+extent
    against it), which is why a bare (size,) tensor_dims -- correct only at offset=0 -- silently
    rejects every sliced fill/drain this design needs once `total` differs from `size`."""
    return TensorAccessPattern((1, total), offset, [1, 1, 1, size], [0, 0, 0, 1])


def _group_tap(total, offset, n_rows, row_stride, run_hi, run_lo):
    """A group-of-`n_rows` gather/scatter: row r's `run_hi*run_lo`-element contiguous run sits
    `r*row_stride` elements apart in the L3 buffer, but CONTIGUOUS (row order) in the L2/L1 tile
    on the other end of the ObjectFifoLink -- exactly what `.split()`/`.join()`'s own `offsets=`
    (row r at r*run_hi*run_lo in the fetched/joined tile) assume. `run_hi*run_lo` splits a
    per-row run that exceeds the shim's 1023-element wrap cap into two dims (see _split_run) --
    at n_rows==1 this degenerates to _flat_tap's own [0,1,1,size] shape."""
    return TensorAccessPattern(
        (1, total), offset, [1, n_rows, run_hi, run_lo], [0, row_stride, run_lo, 1]
    )


def _split_run(total, lim=1023, gran=2):
    """Largest (hi, lo) with hi*lo == total, lo <= lim, lo a multiple of `gran` -- the shim BD
    4-dim wrap-size cap (mlir-aie's verifyStridesWraps), same constraint gemv/design.py's own
    split_run guards. Raises if no such split exists."""
    for lo in range(lim - (lim % gran), 0, -gran):
        if total % lo == 0:
            return total // lo, lo
    raise ValueError(f"{total} has no wrap-legal split (lim={lim}, gran={gran})")


def my_swiglu_mlp_dp(
    dev, D, FF, epsilon=1e-5, stack_size=0x800, func_prefix="", n_aie_cols=8, n_aie_rows=1,
    QD=None, fuse_o=False, trace_size=0, weight_dtype="bf16", group_size=0,
    weight_depth=2, tile_rows_gu=None, chunk_down=False, ff_real=None,
    weight_memtile=False, weight_stage_depth=16,
):
    """`func_prefix` is required (not optional) by iron.common.sequence.FusedDispatch the moment
    this design is placed in an OperatorSequence -- see gemv/design.py's identical parameter for
    the same reason. N = n_aie_cols * n_aie_rows; n_aie_rows=1 is the plain-ObjectFifo topology,
    n_aie_rows>1 uses the MemTile split/join topology -- see the module docstring for both.

    `fuse_o=True` folds `a = Wo @ cx` into this design (see module docstring's FUSE_O section);
    it needs `QD` (the attention context width) and is currently n_aie_rows==1 only -- the
    overlap/pad arithmetic below is derived for the plain per-core-direct-fill topology and has
    not been re-derived for the MemTile split/join one.

    `ff_real` is the MODEL's real FF when `FF` the caller passes is the GATHER-PADDED one (the
    chunked arm pads FF up to a whole multiple of D so the gh exchange moves whole objects).  It
    bounds the gate/up weights only: the wire stops at ff_real rows and the core zeroes the
    unwritten tail of g_buf/u_buf once per dispatch.  `FF` stays the gather/pad domain (gh rounds,
    Wd's K, the emit objects).  None (the 0.6B arm, and this file's other callers) means "no row
    padding", i.e. ff_real == FF.
    """
    # Local shadowing of the module defaults, so an arm can trade tile ROWS against fifo DEPTH at
    # constant L1: depth * TSI_GU * D * 2 bytes is what the budget below actually sees.
    TSI_GU = tile_rows_gu if tile_rows_gu else globals()["TSI_GU"]

    N = n_aie_cols * n_aie_rows
    # TWO SHAPES, one file.  When FF is a whole multiple of D (Qwen3-0.6B: 3072 = 3*1024) the down
    # projection's K = FF fits L1 as one vector and everything below is the shipped 0.6B structure,
    # byte-for-byte.  When it is NOT (Qwen3-4B: 9728 = 3.8*2560) the shared weight tile's minimum
    # equal-byte solution is 19 gate rows == 5 down rows == 49400 B of a 64 KB L1 -- unbuildable --
    # so the down matvec is CHUNKED along K instead: see CHUNKED below for the
    # geometry derivation.
    # `chunk_down=True` selects the 4B arm: the CALLER has already padded FF up to a whole
    # multiple of D (9728 -> 10240 = 4*2560), so every GATHER quantity below is UNIFORM -- FF/N
    # rows per core, (FF/N)/(D/N) = FF/D emit rounds per core, FF/D D-sized K-chunks.  The earlier
    # non-uniform shape emitted a short LAST round (256 of 320 elements into a 320-element out
    # object): a FRACTIONAL object transfer, exactly the hazard the FILL side already documents as
    # fatal -- the shim's accounting is per-object, so a short transfer desyncs which object the
    # core's next release lands in and the drains read each other's data.  Padding to whole
    # objects removes that class of bug entirely, and it costs NO weight bytes any more: the
    # gate/up wire stops at the model's real `ff_real` rows and the core zeroes the unwritten
    # tail of g_buf/u_buf once per dispatch (zero_offset_bf16).  Only Wd still carries the pad, in
    # its K dimension -- that is what makes the 64 pad elements of each core's gh slice harmless.
    CHUNKED = chunk_down or (FF % D) != 0 or os.environ.get("MLP4B_FORCE_CHUNKED") == "1"
    if not CHUNKED:
        TSI_D = TSI_GU // (FF // D)
        assert TSI_D >= 1 and TSI_GU % (FF // D) == 0, (
            f"tile_rows_gu={TSI_GU} must be a multiple of R=FF/D={FF // D}")
        R = FF // D  # =3 at Qwen3-0.6B's shape; also N_GH_CHUNKS (misc) and N_GH_ROUNDS (output)
    else:
        # ---- 4B geometry ----------------------------------------------------------------
        # Every wire row is one D-wide K-chunk: [k u8 | (k/BLOCK) bf16 scales] with k = D for
        # gate/up and for every down chunk.  Gate and down rows are therefore byte-identical, so
        # ONE shared weight tile still serves Wg/Wu/Wd with TSI_GU == TSI_D == tile_rows (8 at
        # D=2560: 8 rows * 2600 B = 20800 B = 325 cache lines, so the tile needs no pad at all --
        # and a padded tile would make the contiguous per-column fill TAPS drift by the pad).
        assert weight_dtype == "int8_ours", (
            "the chunked-down arm is derived for the int8_ours wire only")
        assert n_aie_rows == 1, "the chunked-down arm is derived for n_aie_rows=1"
        assert FF % D == 0, (
            f"the chunked arm needs FF pre-padded to a multiple of D (got FF={FF}, D={D}); "
            f"op_ours does that padding when the model's FF is not a whole multiple of D")
        TSI_D = TSI_GU
        R = None  # the 0.6B "R D-sized reads" arithmetic is replaced by the three below
    assert D % N == 0 and FF % N == 0, f"D={D}, FF={FF} must both be divisible by N={N}"
    D_PER_CORE = D // N
    FF_PER_CORE = FF // N
    assert FF_PER_CORE % 32 == 0, (
        f"silu_tile_bf16 walks its buffer 32 lanes at a time with no tail handling; "
        f"FF/N ({FF_PER_CORE}) must be a multiple of 32"
    )
    # Gate/up ROWS: the model's real FF (`ff_real`) vs the gather-padded FF.  The pad is a
    # GATHER-domain quantity (whole gh objects), no longer a weight-row one -- see the `ff_real`
    # docstring.  g_buf/u_buf stay FF_PER_CORE wide (the emit rounds and the L1 budget below are
    # written against that width); only the WEIGHT wire and the number of gate/up tiles shrink,
    # and GU_TAIL is the part of g_buf/u_buf that no tile writes any more.
    FF_PC_REAL = (ff_real if ff_real else FF) // N
    GU_TAIL = FF_PER_CORE - FF_PC_REAL
    if CHUNKED:
        # N_CHUNKS   : K-chunks the down matvec walks = FF/D (4 at 4B), ALL of them D-wide.  Each
        #              chunk arrives as ONE D-sized misc object and is consumed while resident, so
        #              no gh chunk buffer ever sits in L1 -- that is what buys back the 20480 B
        #              the whole-FF gh_buf would cost (which is what put the naive port over L1).
        # CHUNK_K    : each chunk's own K, i.e. its kernel's DIM_K.  Uniform, hence one accumulate
        #              kernel (mv_int8_acc.cc at DIM_K=D) serves every chunk but the first.
        # N_GH_ROUNDS/ROUND_SZ : emit and drain rounds per core -- FF/N divided by D/N, an exact
        #              integer because the caller padded FF to a multiple of D.  Every round is
        #              D_PER_CORE wide, so every out object moves a WHOLE object (see the
        #              fractional-transfer note at CHUNKED's definition above).
        assert FF % D == 0, f"the chunked arm needs the caller's padded FF (got FF={FF}, D={D})"
        N_CHUNKS = FF // D
        CHUNK_K = [D] * N_CHUNKS
        N_GH_ROUNDS = FF_PER_CORE // D_PER_CORE
        ROUND_SZ = [D_PER_CORE] * N_GH_ROUNDS
        assert N_CHUNKS >= 2, f"the chunked arm needs at least two chunks (FF={FF}, D={D})"
        assert all(k % 128 == 0 for k in CHUNK_K), f"chunk K must be a BLOCK multiple: {CHUNK_K}"
        assert sum(CHUNK_K) == FF and sum(ROUND_SZ) == FF_PER_CORE
        assert FF_PER_CORE % D_PER_CORE == 0, (
            f"FF/N ({FF_PER_CORE}) must be a whole number of D/N ({D_PER_CORE}) rounds -- "
            f"otherwise the last out object moves a FRACTION and the fifo's object accounting "
            f"desyncs")
        assert FF_PER_CORE % TSI_GU == 0 and D_PER_CORE % TSI_D == 0, (
            f"TSI_GU={TSI_GU} must divide FF/N={FF_PER_CORE} and TSI_D={TSI_D} D/N={D_PER_CORE}")
    # WEIGHT WIRE UNITS. bf16 weights are addressed in ELEMENTS; a group-quantized weight is a flat
    # byte row -- [n_groups x f32 scale][packed payload] -- so every weight size, offset and stride
    # below is in whatever unit the wire format uses. Activations (hf, gh, nxt, cx) are ALWAYS bf16
    # and keep their element units; mixing the two is exactly the bytes-vs-elements seam that has
    # no owner, so the weight quantities are named WROW_* and nothing else changes.
    if weight_dtype == "bf16":
        WDT, WUNIT = bfloat16, 2
        WROW_D, WROW_FF = D, FF
        WROW_QD = QD
    elif weight_dtype == "int8_ours":
        # OUR mv_int8.cc wire (per (col,tile) block [tile_m*K u8 | tile_m*(K/128)*2B
        # bf16 scales]), addressed in BF16 ELEMENTS (bytes/2) so the fused bf16
        # arena reinterprets cleanly (memref.reinterpret_cast refuses bf16->i8).
        # Row stride = K*65/64 bytes, affine in K with ratio R=FF/D preserved ->
        # the shared-tile identity below survives unchanged.
        assert group_size == 128, "int8_ours pins GROUP_SIZE=128 (mv_int8.cc default)"
        assert n_aie_rows == 1, (
            "int8_ours is only derived for the plain (n_aie_rows=1) topology"
        )
        WDT, WUNIT = bfloat16, 2
        WROW_D = (D + (D // 128) * 2) // 2      # bf16 elements per row
        WROW_FF = (FF + (FF // 128) * 2) // 2
        WROW_QD = ((QD + (QD // 128) * 2) // 2) if fuse_o else None
    else:
        from iron.operators.gemv.quant import row_stride_bytes
        assert weight_dtype in ("int4", "int8"), f"unknown weight_dtype {weight_dtype!r}"
        assert group_size > 0, "weight_dtype != 'bf16' needs an explicit group_size > 0"
        assert n_aie_rows == 1, (
            "quantized weights are only derived for the plain (n_aie_rows=1) topology -- the "
            "MemTile split/join TAPs below still carry element strides"
        )
        for name, K in (("D", D), ("FF", FF)) + ((("QD", QD),) if fuse_o else ()):
            assert K % group_size == 0, f"{name}={K} must be a whole number of groups ({group_size})"
        WDT, WUNIT = np.int8, 1
        WROW_D = row_stride_bytes(D, group_size, weight_dtype)
        WROW_FF = row_stride_bytes(FF, group_size, weight_dtype)
        WROW_QD = row_stride_bytes(QD, group_size, weight_dtype) if fuse_o else None

    # The shared-tile invariant is what lets Wg/Wu (row width D) and Wd (row width FF) ride ONE
    # ObjectFifo. It survives quantization because row_stride_bytes is affine in K with the same
    # group size, so the R = FF/D ratio is preserved: at int4 g128, 6*544 == 2*1632 == 3264 B.
    if not CHUNKED:
        assert TSI_GU * WROW_D == TSI_D * WROW_FF, (
            f"shared weight tile must be identical for gate/up and down: "
            f"{TSI_GU}*{WROW_D} != {TSI_D}*{WROW_FF} (weight_dtype={weight_dtype})"
        )
    else:
        # Every row is one D-wide chunk row -- [k u8 | (k/BLOCK) bf16 scales] with k = D for
        # gate/up AND for every down chunk (the caller padded FF to a multiple of D) -- so the
        # two row widths are equal BY CONSTRUCTION.
        WROW_FF = WROW_D
    assert FF_PER_CORE % TSI_GU == 0 and D_PER_CORE % TSI_D == 0, (
        f"N={N}: FF/N ({FF_PER_CORE}) must divide by TSI_GU ({TSI_GU}) and "
        f"D/N ({D_PER_CORE}) must divide by TSI_D ({TSI_D})"
    )
    if CHUNKED:
        # Gate/up tiles cover the REAL rows only; GU_TAIL elements of g_buf/u_buf are never
        # written by a tile and are zeroed once per dispatch instead (see CORE's prologue), which
        # is what the row padding of the wire used to provide.
        assert FF_PC_REAL % TSI_GU == 0, (
            f"real FF/N ({FF_PC_REAL}) must divide by TSI_GU ({TSI_GU}): a partial tile would be "
            f"a fractional fill -- the failure this arm's geometry exists to avoid"
        )
        assert GU_TAIL >= 0, f"ff_real ({FF_PC_REAL} rows/core) exceeds the padded FF/N"
        assert GU_TAIL % 64 == 0, (
            f"the zeroed gate/up tail ({GU_TAIL} elems) must be a whole number of 64-lane stores "
            f"(zero_offset_bf16 stores 64 bf16 per call)"
        )
    else:
        assert GU_TAIL == 0, (
            "the non-chunked arm has no gather padding, so ff_real must equal FF"
        )
    N_GU_TILES = (FF_PC_REAL if CHUNKED else FF_PER_CORE) // TSI_GU
    N_D_TILES = D_PER_CORE // TSI_D
    WTILE_UNITS = TSI_GU * WROW_D
    if CHUNKED:
        # The tile's own BYTE size must be a multiple of 64: a tile base landing mid-cache-line is
        # the load_v footgun (a tile base landing mid-cache-line).  8 rows is already exact (8*2600 =
        # 20800 = 325*64); 4 rows needs 32 B of pad (4*2600 = 10400 -> 10432).  The pad lives
        # INSIDE each packed tile block -- wire4b pads the same way -- so the runtime's per-column
        # fill is still ONE contiguous copy; it must just be counted in TILES
        # (N_*_TILES * WTILE_UNITS), never in rows: a row-counted fill stops short of the last
        # tile's content by exactly the pads (the 303.06-object fractional fill the fills below
        # already document).  The kernel reads only its [u8 | scales] prefix, so the pad is never
        # touched by compute -- it is only ever *transferred*.
        WTILE_UNITS = (TSI_GU * WROW_D * WUNIT + 63) // 64 * (64 // WUNIT)
        assert WTILE_UNITS >= TSI_GU * WROW_D
    # Down weight tiles per column: one round of N_D_TILES at 0.6B, N_CHUNKS rounds of them at 4B
    # (the down buffer is chunk-major: [chunk c][column g][tile j]).
    N_D_BLOCKS = N_CHUNKS * N_D_TILES if CHUNKED else N_D_TILES

    if fuse_o and CHUNKED:
        # ---- chunked fuse_o: Wo rides the weight channel as a K-CHUNKED stream, like `down` ----
        # The non-chunked arm below shares the weight object by making TSI_O = WTILE_UNITS //
        # WROW_QD a whole number of Wo rows, which requires the shared tile to be a whole number
        # of BOTH a D-wide row (WROW_D) and a Wo row (WROW_QD).  At 0.6B that is free (QD = 2D, so
        # the rows' ratio is 1/2); at 4B QD = 1.6D is not an integer multiple, and the smallest
        # object satisfying the identity is lcm(WROW_D, WROW_QD) = 8 D-rows = 20800 B (also the
        # smallest 64 B-aligned multiple of 2600 B, so it is forced TWICE over).  That object at
        # weight_depth 2 does not fit this design's L1 (73,216 B at TSI_GU=8), and at depth 1 the
        # weight stream loses its prefetch (29.2 vs 38.0 GB/s) -- more than the eliminated
        # configure is worth.  So this arm leaves the tile geometry ALONE and splits the
        # o-projection along K instead, exactly like `down`: a chunk row is
        # [K_O u8 | (K_O/128) bf16 scales] bytes long, and the shared TSI_GU=4 tile holds TSI_O
        # such rows with the SAME in-block 32 B pad whenever TSI_O*WROW_O == TSI_GU*WROW_D (the
        # tile is then byte-identical in shape to the D-row tiles, just a different (m, K)
        # reading -- the object is the channel's, the kernel's interpretation is per call).
        assert QD % 128 == 0, (
            f"fuse_o needs QD ({QD}) to be a whole number of 128-column scale groups")
        for _k in range(min(D, QD) // 128 * 128, 0, -128):
            if QD % _k:
                continue
            _wrow = (_k + (_k // 128) * 2) // 2      # bf16 units per o-chunk row
            if (TSI_GU * WROW_D) % _wrow:
                continue
            _tsi_o = (TSI_GU * WROW_D) // _wrow
            if D_PER_CORE % _tsi_o:
                continue
            K_O, WROW_O, TSI_O = _k, _wrow, _tsi_o
            break
        else:
            raise ValueError(
                f"fuse_o (chunked): no K-chunk width divides QD={QD} into rows whose tile is "
                f"shape-identical to the {TSI_GU}-row D-wide tile ({TSI_GU * WROW_D} units)")
        N_CX = QD // K_O                                 # K-chunks the o-projection walks
        N_O_TILES = D_PER_CORE // TSI_O
        O_WINDOW = D_PER_CORE                            # TSI_O divides D_PER_CORE: no overlap
        O_OVERLAP = 0
        assert TSI_O * WROW_O == TSI_GU * WROW_D, (TSI_O, WROW_O, TSI_GU, WROW_D)
        assert 0 <= WTILE_UNITS - TSI_O * WROW_O < 32, (
            f"the o-tile's own 64 B pad ({WTILE_UNITS - TSI_O * WROW_O} units) must be the same "
            f"slack the D-row tiles carry (< 32 units)")
        # WO_L3_UNITS: tile-major [K-chunk ci][column g][tile j] -- the order the core walks it
        # (chunk-outer, tile-inner) and the order the runtime's fills issue (see sequence()).
        WO_L3_UNITS = N * N_CX * N_O_TILES * WTILE_UNITS
        # cx delivery.  The misc channel's object type is D-wide (cur/a/npf/gh all need that), so
        # cx can only arrive as whole D-sized objects; the fills therefore sit at the two offsets
        # that keep both of them inside the QD-wide context buffer: [0, D) and [QD-D, QD).  Every
        # K_O-wide chunk lies inside one of them (K_O <= D), so each chunk is copied out of its
        # holder with a source offset -- `mv_int8_acc.cc`'s o-kernels read b[0:K_O], there is no
        # subview support end to end (see add.cc's copy_offset comment).
        assert K_O <= D, f"an o-chunk ({K_O}) must fit one D-wide misc object ({D})"
        CX_OFFS = sorted({0, QD - D} | {i * K_O for i in range(N_CX) if i * K_O <= QD - D})
        CX_SRC = []
        for ci in range(N_CX):
            lo = ci * K_O
            holders = [o for o in CX_OFFS if o <= lo and lo + K_O <= o + D]
            assert holders, (
                f"cx chunk {ci} ([{lo},{lo + K_O})) is not covered by any D-wide fill of "
                f"{CX_OFFS}")
            CX_SRC.append(lo - max(holders))
        # The core walks the cx objects in FIFO order -- ONE object per chunk, copied into the
        # single K_O-wide cx_buf and consumed before the next chunk's copy lands (see core_fn), so
        # the i-th fill must be the holder of chunk i.  True for every shape this arm is built at
        # (at 4B: fills at {0, QD-D} == one per chunk at src {0, 512}); a shape where one fill held
        # two chunks would need more objects than the fifo is filled with, or per-chunk buffers --
        # the L1 the single buffer exists to save -- so refuse it here instead of desyncing the
        # fifo at runtime.
        assert len(CX_OFFS) == N_CX and all(
            s == ci * K_O - CX_OFFS[ci] for ci, s in enumerate(CX_SRC)), (
            f"cx fills {CX_OFFS} (src {CX_SRC}) must be one per K-chunk ({N_CX}) at the chunk's "
            f"own offset")
    elif fuse_o:
        assert n_aie_rows == 1, "fuse_o is only derived for the plain (n_aie_rows=1) topology"
        assert QD is not None, "fuse_o needs QD (the attention context width)"
        assert QD % D == 0, f"fuse_o assumes QD ({QD}) is a whole multiple of D ({D})"
        R_CX = QD // D
        # The cx fills' offsets in the QD-wide context buffer. Non-chunked this is just one fill
        # per D-sized misc object (the core reassembles all R_CX of them into one cx_buf);
        # the chunked arm computes its own set above.
        CX_OFFS = [i * D for i in range(R_CX)]
        assert WTILE_UNITS % WROW_QD == 0, (
            f"fuse_o needs the shared weight tile ({WTILE_UNITS} units) to be a whole number of "
            f"Wo rows ({WROW_QD} units each); it isn't, so Wo can't share this channel"
        )
        TSI_O = WTILE_UNITS // WROW_QD
        N_O_TILES = -(-D_PER_CORE // TSI_O)          # ceil division
        O_WINDOW = N_O_TILES * TSI_O                 # rows actually read per core (>= D_PER_CORE)
        O_OVERLAP = O_WINDOW - D_PER_CORE             # extra rows read past this core's own slice
        assert O_OVERLAP < TSI_O                      # ceil() guarantees this; sanity check
        WO_ROWS_PADDED = D + O_OVERLAP                # rows read per core (overlap included)
        # Wo's arg length. TWICE-TOLD TRAP: `WO_ROWS_PADDED * WROW_QD` (the row-interleaved form)
        # equals `N * N_O_TILES * WTILE_UNITS` in BYTES only because the overlap duplicates rows --
        # 132 rows * WROW_QD == 22 tiles * 6 rows * WROW_QD. The ORDER differs, and mv_int8.cc takes
        # its weight argument as a TILE (payload block then scale block, `[m*K | m*S]`), so the
        # storage must be tile-major with per-core windows. The row-interleaved form silently fed
        # the kernel a transposed wire (measured: a_slice = 0/garbage, all-NaN logits) -- and the
        # bf16 arm cannot show it, because a bf16 wire row carries no scale bytes and the two
        # orders coincide there.
        WO_L3_UNITS = N * N_O_TILES * WTILE_UNITS     # tile-major, per-core windows disjoint

    # L1 budget check (64 KB/core) -- see module docstring's channel accounting for what each
    # buffer is. Computed, not guessed: this is exactly the "hanging numbers are bugs" rule.
    L1_BYTES = 65536
    # MEASURED, not 2: aiecc allocates depth+1 ring buffers for this fifo because its consumer
    # acquires a PAIR (`pair = misc_c.acquire(2)` for cur+a) -- the ELF's own symbol table shows
    # misc_<c>_cons_buff_0/1/2 at 5120 B each while the depth-2 weight and out fifos have exactly
    # two buffers each (read off build/swiglu_4b_n8's elfs_main_core_*/llvm-nm).  A depth-only
    # estimate under-counted this by one D-sized object and let the FUSE_O arm past this assert
    # straight into aiecc's own "allocated buffers exceeded available memory".
    misc_bytes = 3 * (D * 2)
    weight_bytes = weight_depth * (WTILE_UNITS * WUNIT)
    out_bytes = 2 * (D_PER_CORE * 2)  # depth=2
    persistent_bytes = 2 * (D * 2) + (FF * 2) + 2 * (FF_PER_CORE * 2) + (D_PER_CORE * 2)
    # x1_buf + hf_buf         gh_buf      g_buf + u_buf          d_buf
    if CHUNKED:
        # No gh_buf at all: each K-chunk of the all-gathered gh arrives as one D-sized misc object
        # and is consumed while resident, so the whole FF vector never sits in L1 -- the 20480 B
        # item that put the naive port over budget.  g/u keep the FF_PER_CORE width the emit
        # objects need; the FF_PC_REAL..FF_PER_CORE tail of each is zeroed by the core prologue
        # (its weights no longer exist -- see `ff_real`).
        persistent_bytes = 2 * (D * 2) + 2 * (FF_PER_CORE * 2) + (D_PER_CORE * 2)
    if fuse_o:
        # cx_buf + a_slice_buf.  The CHUNKED arm copies one K_O-wide chunk into the SAME buffer at
        # a time (see core_fn), so it costs K_O elements, not the whole QD-wide context.
        persistent_bytes += ((K_O if CHUNKED else QD) * 2) + (O_WINDOW * 2)
        # cx_buf                a_slice_buf
    total = misc_bytes + weight_bytes + out_bytes + persistent_bytes + stack_size
    assert total <= L1_BYTES, (
        f"N={N}: estimated L1 use {total} B exceeds {L1_BYTES} B "
        f"(misc={misc_bytes} weight={weight_bytes} out={out_bytes} "
        f"persistent={persistent_bytes} stack={stack_size})"
    )

    D_ty = np.ndarray[(D,), np.dtype[bfloat16]]
    FF_ty = np.ndarray[(FF,), np.dtype[bfloat16]]
    DPC_ty = np.ndarray[(D_PER_CORE,), np.dtype[bfloat16]]
    FFPC_ty = np.ndarray[(FF_PER_CORE,), np.dtype[bfloat16]]
    WTILE_ty = np.ndarray[(WTILE_UNITS,), np.dtype[WDT]]
    # Weight buffers are counted in TILE BLOCKS, not rows: the packer emits one
    # [TSI rows x k u8 | TSI rows x (k/BLOCK) scales] block per tile, which is what a fill copies
    # and what the kernel's own a_scl = a_tile + m*k/2 addressing assumes.  (At 0.6B
    # N*N_GU_TILES*WTILE_UNITS == FF*WROW_D and N*N_D_BLOCKS*WTILE_UNITS == D*WROW_FF exactly.)
    Wg_L3_ty = np.ndarray[(N * N_GU_TILES * WTILE_UNITS,), np.dtype[WDT]]
    Wd_L3_ty = np.ndarray[(N * N_D_BLOCKS * WTILE_UNITS,), np.dtype[WDT]]
    # CHUNKED: the scratch is exactly FF (== N_CHUNKS * D, since the caller padded FF), i.e. the
    # all-gather round-trips the FULL padded FF vector once per layer.  ObjectFifo transfers are
    # counted in OBJECTS: a fill or drain whose length differs from the object's own is a
    # FRACTIONAL delivery the fifo's accounting cannot represent -- measured as
    # ERT_CMD_STATE_TIMEOUT on the fill side, and as drains reading each other's objects on the
    # out side.  Every fill/drain in this arm now moves whole objects (that is the entire reason
    # FF is padded).
    GH_SCRATCH_ty = np.ndarray[(N_CHUNKS * D if CHUNKED else FF,), np.dtype[bfloat16]]
    if fuse_o:
        QD_ty = np.ndarray[(QD,), np.dtype[bfloat16]]
        OWIN_ty = np.ndarray[(O_WINDOW,), np.dtype[bfloat16]]
        Wo_L3_ty = np.ndarray[(WO_L3_UNITS,), np.dtype[WDT]]
        A_SCRATCH_ty = np.ndarray[(D,), np.dtype[bfloat16]]
        if CHUNKED:
            # One per K-chunk: the o-kernels take b[0:K_O], and a 4B chunk's data starts in the
            # MIDDLE of a D-wide misc object (see CX_SRC above), so each chunk gets its own small
            # buffer rather than one QD-wide one with sub-views (not expressible here).
            KO_ty = np.ndarray[(K_O,), np.dtype[bfloat16]]

    # ---- kernels (one archive per core -- every core plays every role) ----
    # The weight dtype is IN the archive name. Without it a bf16 build silently reuses a cached
    # int4 archive built earlier under the same name and dies at link with
    # "undefined symbol: <prefix>matvec_vectorized_bf16_bf16" -- the artifact-key collision this
    # tree already documents for the fused sequence name.
    _WTAG = "" if weight_dtype == "bf16" else f"_{weight_dtype}g{group_size}"
    CORE_ARCHIVE = f"{func_prefix}swiglu_mlp_dp_core{_WTAG}.a"
    # Two DIFFERENT bindings, not one reused: a Kernel() fixes ONE func.func signature for its
    # symbol, and the two call sites acquire differently-sized buffers (x1=cur+a is full-D; the
    # final residual is D/N-sized). The plain add costs nothing extra -- eltwise_add_bf16_vector
    # is already linked into every other design that touches add.cc.
    add_kernel = Kernel(
        f"{func_prefix}eltwise_add_bf16_vector", CORE_ARCHIVE, [D_ty, D_ty, D_ty, np.int32]
    )
    add_off_kernel = Kernel(
        f"{func_prefix}eltwise_add_offset_a_bf16_vector", CORE_ARCHIVE,
        [D_ty, DPC_ty, DPC_ty, np.int32, np.int32],
    )
    wnorm_kernel = Kernel(
        f"{func_prefix}weighted_rms_norm_fixed", CORE_ARCHIVE, [D_ty, D_ty, D_ty, np.float32]
    )
    _MVSYM = "int8" if weight_dtype == "int8_ours" else weight_dtype
    MV = f"matvec_vectorized_{_MVSYM}_bf16"
    mv_gu_kernel = Kernel(
        f"{func_prefix}{MV}", CORE_ARCHIVE,
        [np.int32, np.int32, WTILE_ty, D_ty, FFPC_ty],
    )
    # Down's own matvec: different DIM_K, same extern "C" name as mv_gu_kernel -- symbol
    # uniqueness is device-wide (one aie.device, one symbol table), so op.py compiles this one
    # from a prefixed object (see fuse/mlp-block's identical mv.cc reuse for the same reason).
    if CHUNKED:
        # Chunked down (4B): TWO bindings, both carrying the shared weight tile and ONE D-sized
        # misc object as b.  Chunk 0 writes d_buf (mv_int8.cc's own write symbol, prefixed); every
        # later chunk accumulates -- all of them at the SAME DIM_K = D, now that FF is padded to a
        # whole multiple of D (an unpadded FF would need a third, short-DIM_K tail kernel).
        mv_d_kernel = Kernel(
            f"{func_prefix}down0_{MV}", CORE_ARCHIVE,
            [np.int32, np.int32, WTILE_ty, D_ty, DPC_ty],
        )
        mv_d_acc_kernel = Kernel(
            f"{func_prefix}downacc_matvec_vectorized_{_MVSYM}_chunk_acc_bf16", CORE_ARCHIVE,
            [np.int32, np.int32, WTILE_ty, D_ty, DPC_ty],
        )
        # The gate/up wires carry the model's REAL FF rows only (see `ff_real`), so the last
        # GU_TAIL elements of g_buf/u_buf have no writer: this zeroes them once per dispatch.
        # Its own kernel file, own translation unit (a kernel variant sharing an existing file's
        # translation unit risks one arm's build flags leaking into another's object).
        zero_kernel = Kernel(
            f"{func_prefix}zero_offset_bf16", CORE_ARCHIVE, [FFPC_ty, np.int32, np.int32]
        )
    else:
        mv_d_kernel = Kernel(
            f"{func_prefix}down_{MV}", CORE_ARCHIVE,
            [np.int32, np.int32, WTILE_ty, FF_ty, DPC_ty],
        )
    silu_kernel = Kernel(f"{func_prefix}silu_tile_bf16", CORE_ARCHIVE, [np.int32, FFPC_ty])
    mul_off_kernel = Kernel(
        f"{func_prefix}eltwise_mul_offset_ab_bf16_vector", CORE_ARCHIVE,
        [FFPC_ty, FFPC_ty, DPC_ty, np.int32, np.int32],
    )
    copy_off_kernel = Kernel(
        f"{func_prefix}copy_offset_bf16_vector", CORE_ARCHIVE, [FF_ty, D_ty, np.int32, np.int32]
    )
    if fuse_o:
        if CHUNKED:
            # The o-projection is a K-chunked stream like `down` (see the geometry block's
            # docstring): the writer sets a_slice, every later chunk accumulates onto it.  Both
            # come from the acc source TU (mv_int8_acc.cc / its signed twin), which is where the
            # chunked entry points live -- hence the SAME `acc_kernel_source` switch the down
            # accumulator uses, and two more prefixed objects for symbol uniqueness.
            mv_o_kernel = Kernel(
                f"{func_prefix}o_matvec_vectorized_{_MVSYM}_chunk_bf16", CORE_ARCHIVE,
                [np.int32, np.int32, WTILE_ty, KO_ty, OWIN_ty],
            )
            mv_o_acc_kernel = Kernel(
                f"{func_prefix}o_matvec_vectorized_{_MVSYM}_chunk_acc_bf16", CORE_ARCHIVE,
                [np.int32, np.int32, WTILE_ty, KO_ty, OWIN_ty],
            )
            # cx's chunk copy: dst starts at 0, the SOURCE starts inside the D-wide misc object.
            copy_off_cxb_kernel = Kernel(
                f"{func_prefix}cxb_copy_offset_ab_bf16_vector", CORE_ARCHIVE,
                [KO_ty, D_ty, np.int32, np.int32, np.int32],
            )
        else:
            # o's own matvec: DIM_K=QD, distinct from mv_gu (DIM_K=D) and mv_d (DIM_K=FF) -- same
            # symbol-uniqueness reasoning as mv_d_kernel above.
            mv_o_kernel = Kernel(
                f"{func_prefix}o_{MV}", CORE_ARCHIVE,
                [np.int32, np.int32, WTILE_ty, QD_ty, OWIN_ty],
            )
            # copy_offset_bf16_vector is (dst, src, size, dst_offset) over raw pointers -- no
            # compile-time size baked in -- but a func.func symbol is keyed by NAME only, and MLIR's
            # verifier refuses two declarations of the same symbol with different memref types
            # ("redefinition of symbol"), so each new call-site shape needs its own renamed object
            # (op.py's cx_copy_obj/oa_copy_obj), exactly like mv_d_kernel's "down_" prefix below.
            copy_off_cx_kernel = Kernel(
                f"{func_prefix}cx_copy_offset_bf16_vector", CORE_ARCHIVE,
                [QD_ty, D_ty, np.int32, np.int32],
            )
        copy_off_a_kernel = Kernel(
            f"{func_prefix}oa_copy_offset_bf16_vector", CORE_ARCHIVE,
            [DPC_ty, OWIN_ty, np.int32, np.int32],
        )

    # ---- ObjectFifos: misc(1, always) + weight(n_aie_cols groups) + output(n_aie_cols groups).
    # n_aie_rows==1: plain per-core ObjectFifos, direct L3<->L1 (no MemTile step in this file --
    # the automatic placer inserts whatever staging one column needs). n_aie_rows>1: one
    # group-level ObjectFifo per column, split (weight) / joined (output) into n_aie_rows row
    # sub-fifos at a MemTile -- see module docstring. Either way `weight_ofs[c]`/`out_ofs[c]`
    # (c = g*n_aie_rows + r) end up as the per-core handles core_fn acquires/releases from; it
    # does not know or care which path built them. fuse_o adds no new shim-facing ObjectFifo: Wo
    # rides the SAME weight_ofs/gweight_ps channel as Wg/Wu/Wd, and `a`'s all-gather rides the
    # SAME out_ofs/gout_cs channel gh's all-gather already uses (see module docstring). ----
    misc_of = ObjectFifo(D_ty, name="misc", depth=2)
    weight_ofs = [None] * N
    out_ofs = [None] * N
    weight_prods = []
    if n_aie_rows == 1:
        for c in range(N):
            if weight_memtile:
                # OPTION B (MemTile staging): stage the
                # weight channel through the column's MemTile.  The shim keeps filling ONE big
                # tap per (matrix, column) into a DEEP MemTile buffer (weight_stage_depth tiles,
                # in L2 -- 512 KB per MemTile, so depth 16-32 costs nothing that matters), and
                # the MemTile forwards tile-by-tile into the core's L1 fifo at depth
                # `weight_depth`.  That is the whole point of the split: L1 caps the core-side
                # depth at ~3-4 tiles (a 5216 B tile at depth 4 is already 21 KB of the 64 KB),
                # while the MemTile can hold tens of tiles, so the shim's DMA never has to
                # restart at a phase boundary -- the bubbles the phase model blames are the
                # stream stopping and re-starting, not the bytes.
                # Same-object forward (no re-shaping): dims_to_stream is only for a forward that
                # CHANGES the object, which is what decode_attn's inKV/memKV does.
                wstage = ObjectFifo(WTILE_ty, name=f"wstage_{c}", depth=weight_stage_depth)
                weight_ofs[c] = wstage.cons().forward(
                    name=f"weight_{c}", tile=Tile(col=c, row=1), obj_type=WTILE_ty,
                    depth=weight_depth,
                )
                weight_prods.append(wstage.prod(tile=Tile(col=c, row=0)))
            else:
                weight_ofs[c] = ObjectFifo(WTILE_ty, name=f"weight_{c}", depth=weight_depth)
                weight_prods.append(weight_ofs[c].prod())
            # depth=2 stays the default: bumping it to N_GH_ROUNDS+1 (one object per emit round)
            # changed NOTHING about the residual-vs-gh scrambling,
            # so the extra L1 was pointless -- revert to the 0.6B-identical configuration.
            out_ofs[c] = ObjectFifo(DPC_ty, name=f"out_{c}", depth=2)
        group_weight_ofs = weight_ofs  # sequence() fills/drains these directly, one per "group"
        group_out_ofs = out_ofs
    else:
        RUN_HI, RUN_LO = _split_run(WTILE_UNITS)
        GROUP_WTILE_ty = np.ndarray[(n_aie_rows * WTILE_UNITS,), np.dtype[WDT]]
        GROUP_OTILE_ty = np.ndarray[(n_aie_rows * D_PER_CORE,), np.dtype[bfloat16]]
        group_weight_ofs = []
        group_out_ofs = []
        for g in range(n_aie_cols):
            gw = ObjectFifo(GROUP_WTILE_ty, name=f"weight_g{g}", depth=weight_depth)
            sub_w = gw.cons().split(
                [r * WTILE_UNITS for r in range(n_aie_rows)],
                obj_types=[WTILE_ty] * n_aie_rows,
                names=[f"weight_{g}_{r}" for r in range(n_aie_rows)],
                depths=[2] * n_aie_rows,
            )
            go = ObjectFifo(GROUP_OTILE_ty, name=f"out_g{g}", depth=2)
            sub_o = go.prod().join(
                [r * D_PER_CORE for r in range(n_aie_rows)],
                obj_types=[DPC_ty] * n_aie_rows,
                names=[f"out_{g}_{r}" for r in range(n_aie_rows)],
                depths=[2] * n_aie_rows,
            )
            for r in range(n_aie_rows):
                weight_ofs[g * n_aie_rows + r] = sub_w[r]
                out_ofs[g * n_aie_rows + r] = sub_o[r]
            group_weight_ofs.append(gw)
            group_out_ofs.append(go)
        weight_prods = [of.prod() for of in group_weight_ofs]

    def core_fn(misc_c, weight_c, out_p,
                x1_buf, hf_buf, gh_buf, g_buf, u_buf, d_buf,
                add_k, add_off_k, wnorm_k, mv_gu_k, mv_d_k, silu_k, mul_off_k, copy_off_k,
                core_id, *fo):
        if CHUNKED:
            # Chunked trailing args: the kernel every chunk but the first accumulates with, the
            # gate/up tail zeroer, and -- under fuse_o -- the o-projection's own writer/accumulator
            # pair plus the ONE cx buffer the chunks are copied into in turn (see below).
            (mv_d_acc_k, zero_k) = fo[:2]
            fo = fo[2:]
            if fuse_o:
                (mv_o_k, mv_o_acc_k, copy_off_cxb_k, copy_off_a_k,
                 a_slice_buf, cx_buf) = fo
            # Prologue: g_buf/u_buf are FF_PER_CORE wide because the emit rounds and the fifo
            # objects are, but the gate/up tiles only reach FF_PC_REAL rows.  Zero the unwritten
            # tails BEFORE anything reads them -- this is the job the zero-weight pad rows of the
            # old wire did, done once per dispatch instead of once per weight byte.  Nothing else
            # in this core function writes those elements, so one call here is stable for the
            # whole dispatch (silu(0)*0 = 0 keeps the tail of gh zero too).
            zero_k(g_buf, GU_TAIL, FF_PC_REAL)
            zero_k(u_buf, GU_TAIL, FF_PC_REAL)
        elif fuse_o:
            (cx_buf, a_slice_buf, mv_o_k, copy_off_cx_k, copy_off_a_k) = fo

        if fuse_o and CHUNKED:
            # step -1 (chunked): cx arrives as D-wide misc objects (the misc channel has ONE object
            # type, and cur/a/npf/gh all need it D-wide), at the offsets CX_OFFS that keep both
            # fills inside the QD-wide context buffer.  A K_O-wide chunk therefore starts in the
            # MIDDLE of its object (CX_SRC), so it is copied out and consumed before the next
            # chunk's copy lands -- ONE cx buffer serves every chunk (4096 B of L1 instead of one
            # per chunk; the chunks are sequential by construction, see step 0).
            for ci in range(N_CX):
                cob = misc_c.acquire(1)
                copy_off_cxb_k(cx_buf, cob, K_O, 0, CX_SRC[ci])
                misc_c.release(1)

                # step 0 (chunked): a_slice = Wo @ cx, chunk-outer / tile-inner, exactly the
                # chunked down matvec's shape -- chunk 0 writes (its tiles cover all D_PER_CORE
                # rows), every later chunk accumulates.
                o_k = mv_o_k if ci == 0 else mv_o_acc_k
                for j in range_(N_O_TILES):
                    j32 = index.casts(T.i32(), j)
                    wt = weight_c.acquire(1)
                    o_k(TSI_O, j32 * TSI_O, wt, cx_buf, a_slice_buf)
                    weight_c.release(1)

            # step 0b: drain this core's D_PER_CORE-wide a-slice onto the shared output channel --
            # the FIRST round through it, ahead of gh's own rounds.  The a-slice all-gather is the
            # gh one's exact mechanism (see the module docstring's FUSE_O section).
            ot = out_p.acquire(1)
            copy_off_a_k(ot, a_slice_buf, D_PER_CORE, 0)
            out_p.release(1)

            # step 0c: refill full `a` (barriered by the caller between 0b and here -- see
            # sequence()'s tg_a_drain/tg_a_refill split) and `cur`, adjacent in the misc queue by
            # construction, so one acquire(2) returns them as a pair exactly like the un-fused-o
            # arm below.
            pair = misc_c.acquire(2)
            add_k(pair[0], pair[1], x1_buf, D)
            misc_c.release(2)
        elif fuse_o:
            # step -1: reassemble cx (QD-wide) from R_CX D-sized misc broadcasts.
            for i in range(R_CX):
                chunk = misc_c.acquire(1)
                copy_off_cx_k(cx_buf, chunk, D, i * D)
                misc_c.release(1)

            # step 0: a_slice[0:O_WINDOW) = Wo[my window] @ cx -- a window of N_O_TILES full
            # TSI_O-row tiles, always >= D_PER_CORE rows (see module docstring's FUSE_O section).
            for j in range_(N_O_TILES):
                j32 = index.casts(T.i32(), j)
                row_off = j32 * TSI_O
                wt = weight_c.acquire(1)
                mv_o_k(TSI_O, row_off, wt, cx_buf, a_slice_buf)
                weight_c.release(1)

            # step 0b: drain only this core's real D_PER_CORE-wide prefix (discard the overlap
            # tail) onto the shared output channel -- the FIRST round through it now, ahead of
            # gh's own R rounds.
            ot = out_p.acquire(1)
            copy_off_a_k(ot, a_slice_buf, D_PER_CORE, 0)
            out_p.release(1)

            # step 0c: refill full `a` (barriered by the caller between 0b and here -- see
            # sequence()'s tg_a_drain/tg_a_refill split) and `cur`, adjacent in the misc queue by
            # construction (sequence() fills them as the last two items before this barrier and
            # the first item after it), so one acquire(2) still returns them as a pair exactly
            # like the non-fused-o arm below.
            pair = misc_c.acquire(2)
            add_k(pair[0], pair[1], x1_buf, D)
            misc_c.release(2)
        else:
            # step 1: x1 = cur + a, full D, replicated on every core.
            pair = misc_c.acquire(2)
            add_k(pair[0], pair[1], x1_buf, D)
            misc_c.release(2)

        # step 2: hf = weighted_rms_norm(x1, n_pf), full D, replicated.
        npf = misc_c.acquire(1)
        wnorm_k(x1_buf, npf, hf_buf, epsilon)
        misc_c.release(1)

        # step 3: g = Wg[my rows] @ hf, then u = Wu[my rows] @ hf -- same shared weight channel,
        # continued (Wg's N_GU_TILES tiles, then Wu's).
        for j in range_(N_GU_TILES):
            j32 = index.casts(T.i32(), j)
            row_off = j32 * TSI_GU
            wt = weight_c.acquire(1)
            mv_gu_k(TSI_GU, row_off, wt, hf_buf, g_buf)
            weight_c.release(1)
        for j in range_(N_GU_TILES):
            j32 = index.casts(T.i32(), j)
            row_off = j32 * TSI_GU
            wt = weight_c.acquire(1)
            mv_gu_k(TSI_GU, row_off, wt, hf_buf, u_buf)
            weight_c.release(1)

        # step 4: g = silu(g), in place over the whole FF/N slice.
        silu_k(FF_PER_CORE, g_buf)

        # step 5: emit gh = silu(g)*u, straight onto the shared output fifo (no separate
        # gh-slice buffer -- the offset read comes out of g_buf/u_buf directly).  At 0.6B that is
        # R rounds of D_PER_CORE; at 4B it is N_GH_ROUNDS = (FF/N)/(D/N) rounds, all D_PER_CORE
        # wide because the caller padded FF, so every round moves exactly one whole out object.
        for r in range(N_GH_ROUNDS if CHUNKED else R):
            ot = out_p.acquire(1)
            if os.environ.get("MLP4B_DUMP") == "cur":
                # DEBUG ONLY: ship the (by now recycled) first misc object's slot instead of gh.
                # Its content at this point is whatever the fifo last put there -- npf under the
                # identity probe.  Nonzero => the malloc-side fills DO deliver to L1; zero => they
                # do not (which is the "everything reads 0" symptom).
                add_off_k(pair[0], d_buf, ot,
                          ROUND_SZ[r] if CHUNKED else D_PER_CORE, r * D_PER_CORE)
            else:
                mul_off_k(g_buf, u_buf, ot,
                          ROUND_SZ[r] if CHUNKED else D_PER_CORE, r * D_PER_CORE)
            out_p.release(1)

        if CHUNKED:
            # step 6 (chunked down, 4B): walk gh's K chunks, each arriving as ONE misc object and
            # consumed while resident, accumulating every chunk's contribution into d_buf.  The
            # chunk loop is a static unroll so "chunk 0 writes, the rest accumulate" stays a
            # compile-time choice; every chunk is D wide, so one acc kernel covers them all.
            for ci in range(N_CHUNKS):
                ch = misc_c.acquire(1)
                dk = mv_d_k if ci == 0 else mv_d_acc_k
                for j in range_(N_D_TILES):
                    j32 = index.casts(T.i32(), j)
                    wt = weight_c.acquire(1)
                    if os.environ.get("MLP4B_SKIP") != "down":
                        dk(TSI_D, j32 * TSI_D, wt, ch, d_buf)
                    weight_c.release(1)
                misc_c.release(1)
        else:
            # step 5b: reassemble the all-gathered gh from R D-sized misc reads (the Runtime's
            # sequence issues these only AFTER every core's R output drains land in gh_scratch --
            # see the TaskGroup barrier in sequence()).
            for i in range(R):
                chunk = misc_c.acquire(1)
                copy_off_k(gh_buf, chunk, D, i * D)
                misc_c.release(1)

            # step 6: d = Wd[my rows] @ gh -- same shared weight channel, continued (Wd's N_D_TILES).
            for j in range_(N_D_TILES):
                j32 = index.casts(T.i32(), j)
                row_off = j32 * TSI_D
                wt = weight_c.acquire(1)
                mv_d_k(TSI_D, row_off, wt, gh_buf, d_buf)
                weight_c.release(1)

        # step 7: nxt[my rows] = x1[my rows] + d, onto the shared output fifo's (R+1)-th round.
        ot = out_p.acquire(1)
        if os.environ.get("MLP4B_DUMP") == "hf":
            # DEBUG ONLY (temporary): ship hf straight out (d_buf is 0 under the all-zero-weight
            # identity probe) so the host can see the x1->hf stage.  add_off_k is the only kernel
            # whose (D_ty -> DPC_ty) shape pair matches hf_buf/out here.
            add_off_k(hf_buf, d_buf, ot, D_PER_CORE, core_id * D_PER_CORE)
        elif os.environ.get("MLP4B_DUMP") == "x1":
            add_off_k(x1_buf, d_buf, ot, D_PER_CORE, core_id * D_PER_CORE)
        else:
            add_off_k(x1_buf, d_buf, ot, D_PER_CORE, core_id * D_PER_CORE)
        out_p.release(1)

        if os.environ.get("MLP4B_DUMMY") == "1":
            # DEBUG ONLY: one extra round AFTER the residual.  If nxt becomes correct with this,
            # the fault is "the LAST out round is lost", not the residual's content/offsets.
            ot = out_p.acquire(1)
            add_off_k(x1_buf, d_buf, ot, D_PER_CORE, core_id * D_PER_CORE)
            out_p.release(1)

    workers = []
    for c in range(N):
        x1_buf = Buffer(D_ty, name=f"x1_{c}")
        hf_buf = Buffer(D_ty, name=f"hf_{c}")
        # CHUNKED keeps no gh buffer in L1 at all (each K chunk rides the misc fifo), so the
        # positional gh_buf slot is never touched.  It is filled with the ALREADY-ALLOCATED
        # hf_buf rather than a tiny dummy: a 1-element buffer still costs a full aligned L1 slot
        # but its declared size is used by the allocator's byte accounting, so a 2-byte "hole"
        # in the middle of the buffer list is exactly the kind of mis-packing that silently
        # aliases two real buffers.  Unused arg, so sharing costs nothing.
        gh_buf = hf_buf if CHUNKED else Buffer(FF_ty, name=f"gh_{c}")
        g_buf = Buffer(FFPC_ty, name=f"g_{c}")
        u_buf = Buffer(FFPC_ty, name=f"u_{c}")
        d_buf = Buffer(DPC_ty, name=f"d_{c}")
        core_args = [
            misc_of.cons(), weight_ofs[c].cons(), out_ofs[c].prod(),
            x1_buf, hf_buf, gh_buf, g_buf, u_buf, d_buf,
            add_kernel, add_off_kernel, wnorm_kernel, mv_gu_kernel, mv_d_kernel,
            silu_kernel, mul_off_kernel, copy_off_kernel,
            c,
        ]
        if CHUNKED:
            core_args += [mv_d_acc_kernel, zero_kernel]
            if fuse_o:
                # (see core_fn's unpacking order) -- the o-projection's chunked writer/accumulator,
                # the two copies it calls, the a-slice it writes and the ONE cx buffer its chunks
                # are copied into in turn.
                a_slice_buf = Buffer(OWIN_ty, name=f"aslice_{c}")
                cx_buf = Buffer(KO_ty, name=f"cx_{c}")
                core_args += [mv_o_kernel, mv_o_acc_kernel, copy_off_cxb_kernel,
                              copy_off_a_kernel, a_slice_buf, cx_buf]
        elif fuse_o:
            cx_buf = Buffer(QD_ty, name=f"cx_{c}")
            a_slice_buf = Buffer(OWIN_ty, name=f"aslice_{c}")
            core_args += [cx_buf, a_slice_buf, mv_o_kernel, copy_off_cx_kernel, copy_off_a_kernel]
        # WORKER_PIN_ROW: see qkv_head_dp/design_ours_kvlayout.py -- default 2 pins each
        # worker to Tile(col=c, row=2); 0 restores the shipped column-major fold.
        _pin = int(os.environ.get("WORKER_PIN_ROW", "2"))
        workers.append(Worker(core_fn, core_args, stack_size=stack_size,
                              **({"tile": Tile(col=c, row=_pin)} if _pin else {})))

    def sequence(*args):
        if fuse_o:
            (cur, cx, npf, Wo, Wg, Wu, Wd, gh_scratch, a_scratch, nxt,
             misc_p, gweight_ps, gout_cs) = args
        else:
            (cur, a, npf, Wg, Wu, Wd, gh_scratch, nxt,
             misc_p, gweight_ps, gout_cs) = args
        # `wait=True` EVERYWHERE, not just on the drains: a plain TaskGroup.finish() with no
        # wait=True lowers to dma_free_task, which is compile-time BD-ID recycling ONLY -- no
        # hardware wait is emitted (AIEAssignRuntimeSequenceBDIDs.cpp; AIEDMATasksToNPU.cpp never
        # even sees dma_free_task). BD-ID pools are allocated PER SHIM TILE, shared across every
        # ObjectFifo mapped to that tile, so a later fill/drain on the SAME tile (misc, weight and
        # output are only 1-2 distinct shim tiles at N=8, since the placer fills 4 rows per column
        # before moving on) can get a recycled BD ID reprogrammed while the freed one's transfer
        # is still in flight -- a lock-count race, not a copy race, so it does not corrupt data,
        # it desyncs an ObjectFifo's acquire()/release() and hangs. Only wait=True lowers to a
        # real `dma_await_task`/NpuSyncOp barrier. MEASURED: without this, every N (including
        # N=8, which has no split/join to blame) hit a genuine device-side TDR
        # (aie2_tdr_detect, journalctl -k) and ERT_CMD_STATE_TIMEOUT, not just a host illusion.
        #
        # fuse_o inserts tg_a_drain/tg_a_refill AHEAD of this group (not merged into it): this
        # core now produces its FIRST output (the a-slice drain) before it has consumed cur/n_pf,
        # where the un-fused-o arm produces its first output (gh) only after consuming ALL of its
        # input. A TaskGroup boundary is a hard barrier (Runtime.finish_task_group awaits at group
        # close), so folding a's fills into tg1 unchanged while a's drain waits behind it would be
        # fine -- but folding the REFILL in with it would not: cur/n_pf are needed for step 1/2,
        # AFTER a is refilled, so their fill has to land in the group that FOLLOWS the a-slice
        # drain barrier, not the one that precedes it. Splitting fills vs drains across groups is
        # always safe; it is only unsafe to place a drain that depends on a not-yet-issued fill in
        # the SAME or an EARLIER group than that fill.
        tg1 = TaskGroup()
        if fuse_o:
            for o in range(len(CX_OFFS)):
                # chunked fuse_o: the fills sit at the two offsets that keep both D-wide taps
                # inside the QD-wide context buffer (see the geometry block's CX_OFFS)
                misc_p.fill(cx, _flat_tap(QD, D, CX_OFFS[o]), wait=True, group=tg1)
            # cur is filled here (fills-only group, before the a barrier) but not CONSUMED until
            # after a is refilled -- see core_fn's step 0c. It stays adjacent to a's own refill in
            # the misc queue only because nothing else is filled into misc between here and
            # tg_a_refill below (n_pf is deliberately deferred to that same later group).
            misc_p.fill(cur, _flat_tap(D, D), wait=True, group=tg1)
            if CHUNKED:
                # Wo is the weight stream's FOURTH matrix, walked chunk-outer / tile-inner exactly
                # like Wd (tg3) -- issued ci-outer so each column's fifo sees its chunks in the
                # order the core consumes them.  It can live in tg1 (unlike Wg/Wu below): the core
                # reaches these tiles before its a-slice drain, so the fills complete.
                for ci in range(N_CX):
                    for g in range(n_aie_cols):
                        base = (ci * n_aie_cols + g) * N_O_TILES * WTILE_UNITS
                        tap = (_tile_tap(WO_L3_UNITS, N_O_TILES, WTILE_UNITS, base)
                               if weight_memtile else
                               _flat_tap(WO_L3_UNITS, N_O_TILES * WTILE_UNITS, base))
                        gweight_ps[g].fill(Wo, tap, wait=True, group=tg1)
            else:
                for g in range(n_aie_cols):
                    gweight_ps[g].fill(
                        Wo, _flat_tap(WO_L3_UNITS, N_O_TILES * WTILE_UNITS,
                                      g * N_O_TILES * WTILE_UNITS),
                        wait=True, group=tg1,
                    )
        else:
            misc_p.fill(cur, _flat_tap(D, D), wait=True, group=tg1)
            misc_p.fill(a, _flat_tap(D, D), wait=True, group=tg1)
            misc_p.fill(npf, _flat_tap(D, D), wait=True, group=tg1)
        if n_aie_rows == 1:
            # Wg/Wu belong in tg1 ONLY when nothing barriers the core between its Wo reads and its
            # Wg reads. With fuse_o the core PRODUCES its a-slice in between, and that drain is in
            # tg_a_drain -- so a Wg fill here can never complete: the core cannot reach step 3 to
            # consume it until a drain that tg1.finish() is itself blocking gets issued.
            # MEASURED as ERT_CMD_STATE_TIMEOUT with `Fatal error type: 0x0`. They are issued
            # after tg_a_refill instead; see the invariant note there.
            if not fuse_o:
                if CHUNKED:
                    # Tile-block arithmetic, not per-row: every tile block is WTILE_UNITS wide and
                    # carries a 64B pad, so a per-row tap would deliver FF_PER_CORE*WROW_D/5216 =
                    # 303.06 objects per column -- a FRACTIONAL fill, which leaves the core's 304th
                    # acquire unsatisfied and deadlocks (measured: ERT_CMD_STATE_TIMEOUT).
                    _gu_tap = ((lambda total, off: _tile_tap(
                        total, N_GU_TILES, WTILE_UNITS, off)) if weight_memtile else
                               (lambda total, off: _flat_tap(
                                   total, N_GU_TILES * WTILE_UNITS, off)))
                    for g in range(n_aie_cols):
                        gweight_ps[g].fill(
                            Wg, _gu_tap(N * N_GU_TILES * WTILE_UNITS,
                                        g * N_GU_TILES * WTILE_UNITS),
                            wait=True, group=tg1,
                        )
                    for g in range(n_aie_cols):
                        gweight_ps[g].fill(
                            Wu, _gu_tap(N * N_GU_TILES * WTILE_UNITS,
                                        g * N_GU_TILES * WTILE_UNITS),
                            wait=True, group=tg1,
                        )
                else:
                    for g in range(n_aie_cols):
                        gweight_ps[g].fill(
                            Wg, _flat_tap(FF * WROW_D, FF_PER_CORE * WROW_D, g * FF_PER_CORE * WROW_D),
                            wait=True, group=tg1,
                        )
                    for g in range(n_aie_cols):
                        gweight_ps[g].fill(
                            Wu, _flat_tap(FF * WROW_D, FF_PER_CORE * WROW_D, g * FF_PER_CORE * WROW_D),
                            wait=True, group=tg1,
                        )
            tg1.finish()
        else:
            tg1.finish()
            # Round-major, group-minor, with a finish() per round: each round issues exactly
            # n_aie_cols fills (one per shim tile), so no tile ever has more than 1 in flight.
            # Batching all rounds into one TaskGroup instead hit aiecc's real per-tile BD queue
            # depth (16) -- shim (0,0) alone would have queued N_GU_TILES*2 (Wg+Wu) unfreed
            # descriptors for group 0. Gathers all n_aie_rows rows' data for one round with a
            # strided TAP (see _group_tap / module docstring).
            for i in range(N_GU_TILES):
                tgw = TaskGroup()
                for g in range(n_aie_cols):
                    base = g * n_aie_rows * FF_PER_CORE * D
                    gweight_ps[g].fill(
                        Wg,
                        _group_tap(FF * D, base + i * TSI_GU * D, n_aie_rows,
                                   FF_PER_CORE * D, RUN_HI, RUN_LO),
                        wait=True, group=tgw,
                    )
                tgw.finish()
            for i in range(N_GU_TILES):
                tgw = TaskGroup()
                for g in range(n_aie_cols):
                    base = g * n_aie_rows * FF_PER_CORE * D
                    gweight_ps[g].fill(
                        Wu,
                        _group_tap(FF * D, base + i * TSI_GU * D, n_aie_rows,
                                   FF_PER_CORE * D, RUN_HI, RUN_LO),
                        wait=True, group=tgw,
                    )
                tgw.finish()

        if fuse_o:
            # tg_a_drain: every core's real D_PER_CORE-wide a-slice, the FIRST round through the
            # shared output channel (gh's own R rounds and the final residual follow it).
            tg_a_drain = TaskGroup()
            for g in range(n_aie_cols):
                gout_cs[g].drain(
                    a_scratch, _flat_tap(D, D_PER_CORE, g * D_PER_CORE),
                    wait=True, group=tg_a_drain,
                )
            tg_a_drain.finish()

            # tg_a_refill: full `a` back to every core (misc), plus n_pf (deferred here so it
            # stays AFTER cur in the misc queue -- core_fn's pair-acquire needs cur and this fill
            # adjacent, and n_pf is consumed only after that pair, so its position here is fine).
            tg_a_refill = TaskGroup()
            misc_p.fill(a_scratch, _flat_tap(D, D), wait=True, group=tg_a_refill)
            misc_p.fill(npf, _flat_tap(D, D), wait=True, group=tg_a_refill)
            tg_a_refill.finish()

            # Wg/Wu, moved here from tg1. THE INVARIANT, stated one-directionally in TIME rather
            # than by task kind: every task in group k must be reachable by the core using only
            # groups <= k. A group is unsafe both when it holds a drain waiting on a later fill
            # AND -- the case that hung this design -- when it holds a fill the core cannot reach
            # until a later group's drain is issued.
            tg_gu = TaskGroup()
            if CHUNKED:
                # The CHUNKED gate/up wire is tile-blocked [column c][tile j] with the model's REAL
                # rows (no pad rows) and a 64 B pad inside every block, so the tap must count TILE
                # BLOCKS -- a per-row tap would deliver a FRACTIONAL number of objects per column
                # (measured: ERT_CMD_STATE_TIMEOUT). Identical form to the non-fuse_o chunked arm's
                # tg1 fills; only the task group differs (see the invariant note above).
                _gu_tap = ((lambda total, off: _tile_tap(
                    total, N_GU_TILES, WTILE_UNITS, off)) if weight_memtile else
                           (lambda total, off: _flat_tap(
                               total, N_GU_TILES * WTILE_UNITS, off)))
            for g in range(n_aie_cols):
                if CHUNKED:
                    gweight_ps[g].fill(
                        Wg, _gu_tap(N * N_GU_TILES * WTILE_UNITS,
                                    g * N_GU_TILES * WTILE_UNITS),
                        wait=True, group=tg_gu)
                else:
                    gweight_ps[g].fill(
                        Wg, _flat_tap(FF * WROW_D, FF_PER_CORE * WROW_D,
                                      g * FF_PER_CORE * WROW_D),
                        wait=True, group=tg_gu)
            for g in range(n_aie_cols):
                if CHUNKED:
                    gweight_ps[g].fill(
                        Wu, _gu_tap(N * N_GU_TILES * WTILE_UNITS,
                                    g * N_GU_TILES * WTILE_UNITS),
                        wait=True, group=tg_gu)
                else:
                    gweight_ps[g].fill(
                        Wu, _flat_tap(FF * WROW_D, FF_PER_CORE * WROW_D,
                                      g * FF_PER_CORE * WROW_D),
                        wait=True, group=tg_gu)
            tg_gu.finish()

        # Barrier: gh_scratch must be fully written before any core reads it back. Every core's
        # output-fifo drains for gh land at disjoint, contiguous offsets that together cover all
        # of gh_scratch exactly once, in the natural FF order Wd's rows expect.
        tg2 = TaskGroup()
        for g in range(n_aie_cols):
            off = 0
            for r in range(N_GH_ROUNDS if CHUNKED else R):
                size = ROUND_SZ[r] if CHUNKED else D_PER_CORE
                if n_aie_rows == 1:
                    tap = _flat_tap(FF, size, g * FF_PER_CORE + off)
                else:
                    tap = _group_tap(
                        FF, g * n_aie_rows * FF_PER_CORE + off,
                        n_aie_rows, FF_PER_CORE, 1, size,
                    )
                gout_cs[g].drain(gh_scratch, tap, wait=True, group=tg2)
                off += size
        tg2.finish()

        # gh_scratch refill AND Wd share this group: both are exactly what the core's down-matvec
        # step needs next, and neither has an ordering hazard against anything still pending.
        tg3 = TaskGroup()
        off = 0
        for i in range(N_CHUNKS if CHUNKED else R):
            # WHOLE objects only (see GH_SCRATCH_ty): each chunk is exactly D wide and each misc
            # object is exactly D wide, so this fill has no fractional remainder to worry about.
            misc_p.fill(
                gh_scratch,
                _flat_tap(N_CHUNKS * D if CHUNKED else FF, D, off),
                wait=True, group=tg3,
            )
            off += D
        if n_aie_rows == 1:
            if CHUNKED:
                # Chunk-major: [chunk c][column g][tile j], one fill per (g, c) covering that
                # column's N_D_TILES tiles of the chunk.  Issued c-outer/g-inner so each column's
                # fifo sees its chunks in the order the core consumes them.
                for c in range(N_CHUNKS):
                    for g in range(n_aie_cols):
                        base = (c * n_aie_cols + g) * N_D_TILES * WTILE_UNITS
                        tap = (_tile_tap(N * N_D_BLOCKS * WTILE_UNITS, N_D_TILES,
                                         WTILE_UNITS, base) if weight_memtile else
                               _flat_tap(N * N_D_BLOCKS * WTILE_UNITS,
                                         N_D_TILES * WTILE_UNITS, base))
                        gweight_ps[g].fill(Wd, tap, wait=True, group=tg3)
                tg3.finish()
            else:
                for g in range(n_aie_cols):
                    gweight_ps[g].fill(
                        Wd, _flat_tap(D * WROW_FF, D_PER_CORE * WROW_FF, g * D_PER_CORE * WROW_FF),
                        wait=True, group=tg3,
                    )
                tg3.finish()
        else:
            tg3.finish()
            for i in range(N_D_TILES):
                tgw = TaskGroup()
                for g in range(n_aie_cols):
                    base = g * n_aie_rows * D_PER_CORE * FF
                    gweight_ps[g].fill(
                        Wd,
                        _group_tap(D * FF, base + i * TSI_D * FF, n_aie_rows,
                                   D_PER_CORE * FF, RUN_HI, RUN_LO),
                        wait=True, group=tgw,
                    )
                tgw.finish()

        tg4 = TaskGroup()
        for g in range(n_aie_cols):
            # Final residual: joined-buffer row order and nxt's own indexing both step by
            # D_PER_CORE, so this drain -- unlike gh's -- is a plain contiguous run even at
            # n_aie_rows>1.
            gout_cs[g].drain(
                nxt, _flat_tap(D, n_aie_rows * D_PER_CORE, g * n_aie_rows * D_PER_CORE),
                wait=True, group=tg4,
            )
        if os.environ.get("MLP4B_DUMMY") == "1":
            # DEBUG ONLY: consume the extra round the core emits after the residual.
            for g in range(n_aie_cols):
                gout_cs[g].drain(
                    gh_scratch, _flat_tap(FF, D_PER_CORE, g * FF_PER_CORE),
                    wait=True, group=tg4,
                )
        tg4.finish()

    if fuse_o:
        rt_args = [
            D_ty, QD_ty, D_ty, Wo_L3_ty, Wg_L3_ty, Wg_L3_ty, Wd_L3_ty, GH_SCRATCH_ty, A_SCRATCH_ty,
            D_ty,
            misc_of.prod(),
            weight_prods, [of.cons() for of in group_out_ofs],
        ]
    else:
        rt_args = [
            D_ty, D_ty, D_ty, Wg_L3_ty, Wg_L3_ty, Wd_L3_ty, GH_SCRATCH_ty, D_ty,
            misc_of.prod(),
            weight_prods, [of.cons() for of in group_out_ofs],
        ]
    rt = Runtime(sequence, rt_args)

    prog = Program(dev, rt, workers=workers)
    maybe_enable_trace(prog, trace_size, workers)
    return prog.resolve_program()
