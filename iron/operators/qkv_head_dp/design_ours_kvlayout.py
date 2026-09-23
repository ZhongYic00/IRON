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
import os

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

# Placement knob: the SequentialPlacer hands unpinned workers the first N entries of
# get_compute_tiles(), which is COLUMN-major (col outer, row inner), so an 8-worker design
# lands on 2 physical columns x 4 rows.  The default WORKER_PIN_ROW=2 pins every worker to
# Tile(col=c, row=2) instead -- one physical column per logical column, one weight stream
# per shim tile.  MEASURED (2026-09-17, interleaved A/B, both chains, gates + ids
# bit-identical): 0.6B 10/10 pairs pin faster (median -0.55 ms/token, -1.4%), 4B 2/2
# (-1.3%); pinned never entered the folded arm's ~41 ms/slow mode (0/10 vs 2/10).  Set 0 to
# restore the folded placement (the pre-pin shipped shape).
_WORKER_PIN_ROW = int(os.environ.get("WORKER_PIN_ROW", "2"))

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


def qkv_head_dp(
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
    B_KV=32,
):
    """`func_prefix` is not optional once this design is placed in an OperatorSequence -- see
    gemv/design.py's identical parameter. N = n_aie_cols, one core per column."""
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
    misc_bytes = 3 * (MO * 2)               # depth 3: n_qn, n_kn and ang are held together
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
    if with_attn:
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
        # ---- M1 CHANNEL FOLDING ----
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
        misc_of = ObjectFifo(MO_ty, name="misc", depth=3)
        misc_cons = [misc_of.cons() for _ in range(N)]
        misc_prod = misc_of.prod()
    weight_ofs = [ObjectFifo(WTILE_ty, name=f"weight_{c}", depth=weight_depth)
                  for c in range(N)]
    out_ofs = [ObjectFifo(HD_ty, name=f"out_{c}", depth=2) for c in range(N)]

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
                **({"tile": Tile(col=c, row=_WORKER_PIN_ROW)} if _WORKER_PIN_ROW else {}),
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
                     misc_p, weight_ps, out_cs, ctx_out=None, attn_h=None):
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
            if with_attn:
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
        rt = Runtime(
            sequence,
            [
                D_ty, D_ty, W_L3_ty, *const_tys, Q_L3_ty, KV_L3_ty,
                misc_prod,
                [of.prod() for of in weight_ofs], [of.cons() for of in out_ofs],
            ],
        )

    prog = Program(dev, rt, workers=workers)
    maybe_enable_trace(prog, trace_size, workers)
    return prog.resolve_program()
