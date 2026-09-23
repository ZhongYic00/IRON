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

import aie.dialects.index as index
from aie.dialects.aie import T
from ml_dtypes import bfloat16
import numpy as np

from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (Buffer, Kernel, ObjectFifo, Program, Runtime, ScratchpadParameter,
                      TaskGroup, Worker, sync_parameters)

from iron.operators._trace import maybe_enable_trace

BF16 = bfloat16


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
    N_MISC_CHUNKS = D // HD
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
    misc_bytes = 3 * (HD * 2)               # depth 3: n_qn, n_kn and ang are held together
    weight_bytes = weight_depth * (WTILE_ELEMS * 2)
    out_bytes = 2 * (HD * 2)
    persistent_bytes = 3 * (D * 2) + 2 * (HD * 2)   # cur, n_in, hn + raw, normed
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
    copy_kernel = Kernel(
        f"{func_prefix}copy_offset_bf16_vector", CORE_ARCHIVE, [D_ty, HD_ty, np.int32, np.int32]
    )
    # Two bindings of weighted_rms_norm at two widths. One Kernel() fixes ONE func.func signature
    # per symbol, so the HD-wide call site gets its own symbol from a prefixed object -- the same
    # mechanism swiglu_mlp_dp uses for its two matvec DIM_Ks, rather than a local copy of the
    # vendored kernel under a second name (which is what fuse/qkv-head did).
    wnorm_d_kernel = Kernel(
        f"{func_prefix}weighted_rms_norm_fixed", CORE_ARCHIVE,
        [D_ty, D_ty, D_ty, np.float32]
    )
    wnorm_hd_kernel = Kernel(
        f"{func_prefix}hd_weighted_rms_norm_fixed", CORE_ARCHIVE,
        [HD_ty, HD_ty, HD_ty, np.float32]
    )
    # OUR int8 matvec (mv_int8.cc, mode14 premul + VEC128): same (m, row_off,
    # a_tile, b, c) call shape as the bf16 mv, a_tile reinterpreted as the
    # [m*K u8 | m*(K/128)*2B scales] byte stream by the kernel.
    mv_kernel = Kernel(
        f"{func_prefix}matvec_vectorized_int8_bf16", CORE_ARCHIVE,
        [np.int32, np.int32, WTILE_ty, D_ty, HD_ty],
    )
    rope_kernel = Kernel(
        f"{func_prefix}rope", CORE_ARCHIVE, [HD_ty, HD_ty, HD_ty, np.int32]
    )

    # `depth=3` is what lets the core hold n_qn, n_kn and ang simultaneously (`.acquire(3)`); the
    # D/HD chunks before them stream through one at a time. Broadcast to N cores via N `.cons()`
    # handles -- the fan-out is in the stream-switch fabric, not the producer's own DMA.
    misc_of = ObjectFifo(HD_ty, name="misc", depth=3)
    weight_ofs = [ObjectFifo(WTILE_ty, name=f"weight_{c}", depth=weight_depth)
                  for c in range(N)]
    out_ofs = [ObjectFifo(HD_ty, name=f"out_{c}", depth=2) for c in range(N)]

    def core_fn(misc_c, weight_c, out_p, cur_buf, nin_buf, hn_buf, raw_buf, nrm_buf,
                copy_k, wnorm_d_k, wnorm_hd_k, mv_k, rope_k, kinds):
        # step 1: rebuild cur and n_in from D/HD chunks, then hn = weighted_RMSNorm(cur, n_in).
        # `hn` never reaches DDR -- it is what the four fused ops used to hand each other through it.
        for i in range_(N_MISC_CHUNKS):
            ch = misc_c.acquire(1)
            copy_k(cur_buf, ch, HD, i * HD)
            misc_c.release(1)
        for i in range_(N_MISC_CHUNKS):
            ch = misc_c.acquire(1)
            copy_k(nin_buf, ch, HD, i * HD)
            misc_c.release(1)
        wnorm_d_k(cur_buf, nin_buf, hn_buf, epsilon)

        # step 2: n_qn, n_kn, ang -- read once per head, so acquired once and held to the end.
        w3 = misc_c.acquire(3)
        nqn_t, nkn_t, ang_t = w3[0], w3[1], w3[2]

        # step 3: this core's heads, in weight-row order. `kinds` is a Python list, so the branch
        # is resolved while tracing and each core emits only the code its own heads need.
        for kind in kinds:
            out_t = out_p.acquire(1)
            dst = raw_buf if kind != "v" else out_t   # v heads land straight in the drain tile
            for j in range_(N_W_TILES):
                row_off = index.casts(T.i32(), j) * tsi
                wt = weight_c.acquire(1)
                mv_k(tsi, row_off, wt, hn_buf, dst)
                weight_c.release(1)
            if kind != "v":
                wnorm_hd_k(raw_buf, nqn_t if kind == "q" else nkn_t, nrm_buf, epsilon)
                rope_k(nrm_buf, ang_t, out_t, HD)
            out_p.release(1)

        misc_c.release(3)

    def head_kind(g):
        return "q" if g < Hq else ("k" if g < Hq + Hkv else "v")

    workers = []
    for c in range(N):
        kinds = [head_kind(c * HEADS_PER_CORE + h) for h in range(HEADS_PER_CORE)]
        workers.append(
            Worker(
                core_fn,
                [
                    misc_of.cons(), weight_ofs[c].cons(), out_ofs[c].prod(),
                    Buffer(D_ty, name=f"cur_{c}"), Buffer(D_ty, name=f"nin_{c}"),
                    Buffer(D_ty, name=f"hn_{c}"),
                    Buffer(HD_ty, name=f"raw_{c}"), Buffer(HD_ty, name=f"nrm_{c}"),
                    copy_kernel, wnorm_d_kernel, wnorm_hd_kernel, mv_kernel, rope_kernel,
                    kinds,
                ],
                stack_size=stack_size,
            )
        )

    def sequence(cur, nin, wqkv, nqn, nkn, ang, q, kv_cache, misc_p, weight_ps, out_cs):
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
        if kv_off_param is not None:
            sync_parameters()
        tg = TaskGroup()
        # cur/nin are D-wide = N_MISC_CHUNKS fifo objects per fill; aiecc
        # expands each fill into one BD per object, so D=2560 (20 objects x2
        # fills) alone overflows the shim's 16-BD pool. Split into HD-chunk
        # fills with a finish() per chunk (swiglu round-major idiom).
        for i in range(N_MISC_CHUNKS):
            tg_i = TaskGroup()
            misc_p.fill(cur, _flat_tap(D, HD, i * HD), wait=True, group=tg_i)
            tg_i.finish()
        for i in range(N_MISC_CHUNKS):
            tg_i = TaskGroup()
            misc_p.fill(nin, _flat_tap(D, HD, i * HD), wait=True, group=tg_i)
            tg_i.finish()
        misc_p.fill(nqn, _flat_tap(HD, HD), wait=True, group=tg)
        misc_p.fill(nkn, _flat_tap(HD, HD), wait=True, group=tg)
        misc_p.fill(ang, _flat_tap(HD, HD), wait=True, group=tg)
        for c in range(N):
            # ONE fill per core for the whole slice; the fifo hands it to the core in WTILE_ELEMS
            # pieces. Filling per tile instead would be N_W_TILES*HEADS_PER_CORE BDs per core, and
            # small fills are half of why the spatial version lost.
            weight_ps[c].fill(
                wqkv, _flat_tap(TOT * ROW_BYTES // 2, ROWS_PER_CORE * ROW_BYTES // 2, c * ROWS_PER_CORE * ROW_BYTES // 2),
                wait=True, group=tg,
            )
            for h in range(HEADS_PER_CORE):
                g = c * HEADS_PER_CORE + h
                kind = head_kind(g)
                if kind == "q":
                    # q keeps a plain L3 buffer: the scores GEMV reads it whole, per token.
                    out_cs[c].drain(q, _flat_tap(QD, HD, g * HD), wait=True, group=tg)
                else:
                    # OUR kv layout: the chain's head-major per-token-[K_t|V_t]
                    # cache. Head hh occupies a 2*max_seq*HD region (offset
                    # hh*2*max_seq*HD); within it token t's K sits at t*2*HD
                    # and V at t*2*HD + HD. The static head-segment base and
                    # the V half-offset ride the tap; the position term is
                    # kv_off = t*2*HD (the host's existing k_off value),
                    # patched per dispatch — identical for both kinds.
                    hh = g - Hq if kind == "k" else g - Hq - Hkv
                    seg = hh * 2 * max_seq * HD
                    half = HD if kind == "v" else 0
                    out_cs[c].drain(
                        kv_cache, _flat_tap(Hkv * 2 * max_seq * HD, HD, seg + half),
                        wait=True, group=tg, offset_parameter=kv_off_param,
                    )
        tg.finish()

    rt = Runtime(
        sequence,
        [
            D_ty, D_ty, W_L3_ty, HD_ty, HD_ty, HD_ty, Q_L3_ty, KV_L3_ty,
            misc_of.prod(),
            [of.prod() for of in weight_ofs], [of.cons() for of in out_ofs],
        ],
    )

    prog = Program(dev, rt, workers=workers)
    maybe_enable_trace(prog, trace_size, workers)
    return prog.resolve_program()
