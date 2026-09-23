# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import math

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    Buffer,
    ScratchpadParameter,
    TaskGroup,
    WorkerRuntimeBarrier,
)
from aie.iron.device import Tile
from aie.iron.runtime.runtime import sync_parameters
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.helpers.dialects.scf import _for as range_


def decode_attn(
    dev,
    cols,
    H,
    KV,
    D,
    S_KV,
    B_KV,
    func_prefix="",
    kernel_object="decode_attn.o",
    use_runtime_seq_len=False,
    batch_queries=0,
    num_new_tokens=0,
    causal=True,
    out_buffer_tokens=0,
    kv_layout="combined",
    verbose=False,
    trace_size=0,
):
    """Fused decode attention (single query, M=1) in GEMV semantics.

    Each column owns `heads_per_col = H // cols` query heads sharing one KV head
    (GQA = H // KV).  K and V ride a single combined KV stream (2 input DMA
    channels total per core: Q + KV) to stay within the core's 2-in/2-out DMA
    budget.  Each (2*B_KV, D) block tile holds K_block (rows [0,B_KV)) and
    V_block (rows [B_KV,2*B_KV)) — the default block-interleaved cache layout;
    `kv_token_interleaved=True` (what this chain's applications use) keeps the same
    tiles but the model's cache stores per-token [K_t|V_t] pairs, selected in the
    kernel by its KV_TOKEN_INTERLEAVED row indexing.

    For each head the worker resets its running online-softmax state and then makes
    a SINGLE streaming pass over the KV blocks: per block it computes scores from
    the K half, folds them into the running max/sum (m, l) and rescales +
    accumulates the context from the V half (FlashAttention-2 online softmax,
    mirroring mha's partial_softmax/rescale_O), then finalizes once at the end.
    Nothing is re-streamed, and the full (heads_per_col, S_KV) score/probability
    matrices never materialize in L1 — that is what pinned the earlier TWO-PASS
    version at S_KV ~ 2048 against the 64 KB tile wall.  The per-block `scores`
    scratch is consumed immediately; out_acc accumulates in place.

    Batch mode (`batch_queries > 0`): processes `num_new_tokens` new tokens per
    dispatch (verify/speculative-decode path) in `PASS = NEW // M` groups of
    M queries each.  Per group the core acquires its (M*hpc, D) Q tile, streams
    the KV cache ONCE (the m-loop lives inside the KV-block loop, so KV traffic
    is 1x per group regardless of M — the batch AI-ratio win), finalizes that
    group's rows and releases Q/O; out_acc stays (M*hpc, D) group-local so 4B
    shapes (hpc=4) fit the 64KB L1 wall with block_kv=32.  The kernels are
    UNCHANGED: rows are addressed by the flat index mh = m*heads_per_col + h,
    and the per-query causal boundary is S_kv_base + g*M + m + 1.  `scores`
    stays a (hpc, B_KV) per-block scratch shared across queries (consumed
    immediately).  The scratchpad parameter is "S_kv_base" = number of cached
    tokens BEFORE the new block (legacy M=1 mode keeps "S_kv_eff" = boundary
    = pos+1).  `causal=False` (DFlash draft) masks nothing (INT32_MAX boundary).
    """
    heads_per_col = H // cols
    assert H % cols == 0, "H must divide evenly across columns"
    assert KV == cols, "GQA layout assumes one KV head per column"
    assert S_KV % B_KV == 0, "S_KV must be a multiple of B_KV"
    NB = S_KV // B_KV
    # kv_layout:
    #   "combined" (default): ONE KV arg, per-head region [K half | V half]
    #     (2*S_KV*D per head) — the qwen3 int8 chain's StridedCopy append layout.
    #   "separate": TWO args (K and V caches), each per-head head-major
    #     [head][seq][D] — the xdna-engine qkv_head_dp drain layout (their fused
    #     head writes k/v straight into two caches at kv_off). Same inKV fifo,
    #     two fills (K then V) preserve the [K all | V all] stream order the
    #     kernel and the MemTile forward() expect, so L1/kernel are untouched;
    #     only the L3 arg splits (DMA channel count unchanged). M=1 only:
    #     batch taps stay combined (dflash verify path).
    assert kv_layout in ("combined", "separate"), f"unknown kv_layout {kv_layout!r}"
    _separate = kv_layout == "separate"

    batch = batch_queries > 0
    if batch:
        assert not _separate, "batch taps are derived for combined only"
    if batch:
        M = batch_queries
        NEW = num_new_tokens if num_new_tokens > 0 else M
        assert NEW % M == 0, "num_new_tokens must be a multiple of batch_queries"
        PASS = NEW // M
        assert use_runtime_seq_len, "batch mode requires use_runtime_seq_len"
        if out_buffer_tokens:
            assert out_buffer_tokens % M == 0, \
                "out_buffer_tokens must be a multiple of batch_queries"
    OB = out_buffer_tokens if (batch and out_buffer_tokens) else \
        (NEW if batch else 1)
    q_rows = M * heads_per_col if batch else heads_per_col

    dtype = bfloat16
    f32 = np.float32
    i32 = np.int32

    # DRAM tensor types (KV combined: K in [0,S_KV*D), V in [S_KV*D,2*S_KV*D))
    # separate: K and V are TWO args, each per-head head-major [KV, S_KV*D]
    # (the xdna qkv_head_dp cache layout). Batch mode: Q/O are flat
    # (NEW*H, D) — token-major rows, per-column tap slices [c*hpc, c*hpc+hpc)
    # of each token's head block.
    if _separate:
        L3_K_ty = np.ndarray[(KV, S_KV * D), np.dtype[dtype]]
        L3_V_ty = np.ndarray[(KV, S_KV * D), np.dtype[dtype]]
    L3_Q_ty = np.ndarray[(NEW * H if batch else H, D), np.dtype[dtype]]
    L3_KV_ty = np.ndarray[(KV, 2 * S_KV * D), np.dtype[dtype]]
    L3_O_ty = np.ndarray[(OB * H if batch else H, D), np.dtype[dtype]]

    # L1 tile types (q/out rows grow with M in batch mode; scores stays
    # per-block — it is a scratch consumed immediately per (block, query)).
    L1_Q_ty = np.ndarray[(q_rows, D), np.dtype[dtype]]
    L1_KV_ty = np.ndarray[(2 * B_KV, D), np.dtype[dtype]]  # [K_block | V_block]
    L1_O_ty = np.ndarray[(q_rows, D), np.dtype[dtype]]

    # persistent L1 buffers (heads_per_col heads per column), block-local so they
    # do NOT grow with S_KV (this is what breaks the 64KB tile wall at S_KV>2048).
    # scores is f32 per-block scratch: raw q@k can be ~1e4, bf16 would destroy it.
    # Batch mode: the kernel indexes scores by the SAME flat row index mh as
    # q/out_acc, so the scratch must span ALL rows (M*hpc) — sizing it
    # (heads_per_col, B_KV) overflows into the neighbouring L1 buffers and
    # corrupts the online-softmax state (giant/zero alternating outputs).
    scores_ty = np.ndarray[(q_rows, B_KV), np.dtype[f32]]
    out_acc_ty = np.ndarray[(q_rows, D), np.dtype[f32]]
    # running online-softmax state per head: m = row max (log2e-scaled), l = sum.
    m_ty = np.ndarray[(q_rows,), np.dtype[f32]]
    l_ty = np.ndarray[(q_rows,), np.dtype[f32]]

    # kernels
    scores_kernel = Kernel(
        f"{func_prefix}attn_scores_block", f"{func_prefix}{kernel_object}",
        [L1_Q_ty, i32, L1_KV_ty, scores_ty, i32, i32, f32],
    )
    online_kernel = Kernel(
        f"{func_prefix}attn_online_block", f"{func_prefix}{kernel_object}",
        [scores_ty, i32, L1_KV_ty, out_acc_ty, m_ty, l_ty, i32, i32],
    )
    finalize_kernel = Kernel(
        f"{func_prefix}attn_finalize", f"{func_prefix}{kernel_object}",
        [out_acc_ty, i32, l_ty, L1_O_ty],
    )
    zero_kernel = Kernel(
        f"{func_prefix}attn_zero_state", f"{func_prefix}{kernel_object}",
        [out_acc_ty, m_ty, l_ty, i32],
    )

    # scale = log2e / sqrt(D)
    scale = math.log2(math.e) / math.sqrt(D)

    # ObjectFifos per column.  Batch mode drops Q/O fifo depth to 1 (L1 budget:
    # out_acc grows M*hpc*D*4B; depth-2 Q/O tiles would overflow the 64KB wall).
    inQ = [ObjectFifo(L1_Q_ty, name=f"inQ_{c}", depth=1 if batch else 2) for c in range(cols)]
    # KV rides a shim->MemTile->core forward chain: the shim producer fills the
    # WHOLE interleaved head (2*S_KV, D) once, the MemTile forward() streams it
    # into per-block (2*B_KV, D) tiles via dims_to_stream, matching mha's
    # inK.cons().forward() pattern. This keeps the shim BD count O(cols) instead
    # of O(NB*cols), which is what deadlocked at S_KV=256.
    inKV = [ObjectFifo(L1_KV_ty, name=f"inKV_{c}", depth=1) for c in range(cols)]
    memKV = [
        inKV[c].cons().forward(
            name=f"memKV_{c}",
            tile=Tile(col=c, row=1),
            dims_to_stream=[(2 * B_KV, D), (D, 1)],
            depth=1,
        )
        for c in range(cols)
    ]
    outO = [ObjectFifo(L1_O_ty, name=f"outO_{c}", depth=1 if batch else 2) for c in range(cols)]

    # persistent buffers per column (all block/head-local, independent of S_KV).
    scores_buf = [
        Buffer(type=scores_ty, name=f"scores_{c}",
               initial_value=np.zeros((heads_per_col, B_KV), dtype=f32))
        for c in range(cols)
    ]
    out_acc_buf = [
        Buffer(type=out_acc_ty, name=f"out_acc_{c}",
               initial_value=np.zeros((q_rows, D), dtype=f32))
        for c in range(cols)
    ]
    m_buf = [
        Buffer(type=m_ty, name=f"m_{c}",
               initial_value=np.zeros((q_rows,), dtype=f32))
        for c in range(cols)
    ]
    l_buf = [
        Buffer(type=l_ty, name=f"l_{c}",
               initial_value=np.zeros((q_rows,), dtype=f32))
        for c in range(cols)
    ]

    def core_body(q_fifo, kv_fifo, o_fifo, scores, out_acc, m, l,
                  scores_k, online_k, fin_k, zero_k, seq_kv_eff_param=None,
                  barrier=None):
        # Barrier + scratchpad read live OUTSIDE the while loop (once per
        # dispatch), matching mha's batched_matmul_qk and softmax: the barrier
        # lock and the scratchpad sync lock are each acquired once per token,
        # then the fixed-size KV work loop runs to completion.  The enclosing
        # scf.for sys.maxsize is added by Worker(while_true=True), with the
        # per-token Q acquire as the cross-dispatch blocking sync point.
        if barrier is not None:
            barrier.wait_for_value(1)
        seq_pos = seq_kv_eff_param.read() if seq_kv_eff_param is not None else S_KV

        q = q_fifo.acquire(1)
        o = o_fifo.acquire(1)
        # reset online state for both heads, then a SINGLE streaming pass: for
        # each KV block compute block scores and immediately fold them into the
        # running max/sum + context accumulator (online softmax, FlashAttention-2).
        for h in range_(heads_per_col):
            zero_k(out_acc, m, l, h)
        for b in range_(NB):
            kv = kv_fifo.acquire(1)
            for h in range_(heads_per_col):
                scores_k(q, h, kv, scores, b, seq_pos, scale)
                online_k(scores, h, kv, out_acc, m, l, b, seq_pos)
            kv_fifo.release(1)
        for h in range_(heads_per_col):
            fin_k(out_acc, h, l, o)
        q_fifo.release(1)
        o_fifo.release(1)

    def core_body_batch(q_fifo, kv_fifo, o_fifo, scores, out_acc, m, l,
                        scores_k, online_k, fin_k, zero_k, seq_kv_eff_param=None,
                        barrier=None):
        # Batch variant: PASS groups of M queries per dispatch.  Per group the
        # worker acquires its (M*hpc, D) Q tile, zeroes/accumulates/finalizes
        # ONLY that group's rows (out_acc stays (M*hpc, D) — group-local), and
        # releases Q/O so the next group can proceed.  The KV cache streams
        # ONCE per group.  Per-query causal boundary = S_kv_base + g*M + m + 1;
        # causal=False passes INT32_MAX (no mask, DFlash draft).
        if barrier is not None:
            barrier.wait_for_value(1)
        seq_pos = seq_kv_eff_param.read() if seq_kv_eff_param is not None else S_KV

        for g in range_(PASS):
            q = q_fifo.acquire(1)
            o = o_fifo.acquire(1)
            for mh in range_(q_rows):
                zero_k(out_acc, m, l, mh)
            for b in range_(NB):
                kv = kv_fifo.acquire(1)
                for mi in range_(M):
                    if causal:
                        seq_pos_m = seq_pos + g * M + mi + 1
                    else:
                        seq_pos_m = 2147483647
                    for h in range_(heads_per_col):
                        mh = mi * heads_per_col + h
                        scores_k(q, mh, kv, scores, b, seq_pos_m, scale)
                        online_k(scores, mh, kv, out_acc, m, l, b, seq_pos_m)
                kv_fifo.release(1)
            for mh in range_(q_rows):
                fin_k(out_acc, mh, l, o)
            q_fifo.release(1)
            o_fifo.release(1)

    # runtime seq_pos scratchpad parameter (decode full-ELF path)
    # Batch mode reads "S_kv_base" (cached-token count before the new block);
    # legacy M=1 keeps "S_kv_eff" (= causal boundary, pos+1).
    seq_kv_eff_param = (
        ScratchpadParameter("S_kv_base" if batch else "S_kv_eff", np.int32)
        if use_runtime_seq_len
        else None
    )
    # per-column barrier: registers an aie.lock on each core that the scratchpad
    # parameter sync releases (set_lock) after the host writes S_kv_eff. Mirrors
    # softmax's vector_size_parameter + WorkerRuntimeBarrier pairing.
    worker_barriers = [WorkerRuntimeBarrier(initial_value=0) for c in range(cols)]

    # WORKER_PIN_ROW: see design_ours_kvlayout.py -- default 2 pins this column's worker to
    # Tile(col=c, row=2); WORKER_PIN_ROW=0 restores the SequentialPlacer's column-major fold.
    import os as _pin_os
    _WORKER_PIN_ROW = int(_pin_os.environ.get("WORKER_PIN_ROW", "2"))
    workers = [
        Worker(
            core_body_batch if batch else core_body,
            [
                inQ[c].cons(), memKV[c].cons(), outO[c].prod(),
                scores_buf[c], out_acc_buf[c], m_buf[c], l_buf[c],
                scores_kernel, online_kernel, finalize_kernel, zero_kernel,
            ]
            + ([seq_kv_eff_param] if seq_kv_eff_param is not None else [])
            + ([worker_barriers[c]] if use_runtime_seq_len else []),
            while_true=True,
            **({"tile": Tile(col=c, row=_WORKER_PIN_ROW)} if _WORKER_PIN_ROW else {}),
        )
        for c in range(cols)
    ]

    # Taps: Q and O are per-column 2-head slices; KV is streamed block-by-block,
    # once per group in batch mode (PASS fills), once total in legacy mode.
    if batch:
        # Q: per-group 2-D taps (the worker acquires group g's (M*hpc, D) tile;
        # PASS fills per dispatch).  O: ONE full-OB tap per column over flat
        # (OB*H, D) — the memtile aggregates the group tiles into the shim
        # drain (token-major, per-column head slice, gap-carrying L3 tap).
        Q_tap = [
            [
                TensorAccessPattern(
                    tensor_dims=L3_Q_ty.__args__[0],
                    offset=g * M * H * D + c * heads_per_col * D,
                    sizes=[1, 1, M, heads_per_col * D],
                    strides=[0, 0, H * D, 1],
                )
                for c in range(cols)
            ]
            for g in range(PASS)
        ]
        O_tap = [
            [
                TensorAccessPattern(
                    tensor_dims=L3_O_ty.__args__[0],
                    offset=g * M * H * D + c * heads_per_col * D,
                    sizes=[1, 1, M, heads_per_col * D],
                    strides=[0, 0, H * D, 1],
                )
                for c in range(cols)
            ]
            for g in range(PASS)
        ]
    else:
        Q_tap = [
            TensorAccessPattern(
                tensor_dims=L3_Q_ty.__args__[0],
                offset=c * heads_per_col * D,
                sizes=[1, 1, 1, heads_per_col * D],
                strides=[0, 0, 0, 1],
            )
            for c in range(cols)
        ]
        O_tap = [
            TensorAccessPattern(
                tensor_dims=L3_O_ty.__args__[0],
                offset=c * heads_per_col * D,
                sizes=[1, 1, 1, heads_per_col * D],
                strides=[0, 0, 0, 1],
            )
            for c in range(cols)
        ]
    # ONE tap per KV head covering the whole interleaved [K|V] head
    # (2*S_KV*D elements at offset c*(2*S_KV*D)). The MemTile forward() then
    # streams it into per-block (2*B_KV, D) tiles. This keeps the shim BD count
    # O(cols) instead of O(NB*cols).
    #
    # The shim's DMA BD encodes the leading non-unit dimension (here 2*S_KV) as a
    # 10-bit wrap field (max 1023), so S_KV > 511 overflows it and the BD writes
    # out of bounds (clobbering the input buffer -> all-NaN). Mirror mha's
    # legalize_tas: when 2*S_KV exceeds 1023, collapse the tap to a contiguous
    # 1-D descriptor ([1,1,1,2*S_KV*D], stride 1). The total element count and the
    # interleaved [K|V] layout are unchanged, and the shim buffer_length field is
    # 32-bit so the collapsed length fits, so the MemTile forward(dims_to_stream)
    # still re-tiles it into per-block (2*B_KV, D) chunks correctly.
    #
    # separate layout: same per-head tap geometry, but TWO fills — K_tap[c]
    # (offset c*S_KV*D in the K arg) then V_tap[c] (offset c*S_KV*D in the V
    # arg) — onto the SAME inKV fifo, preserving the [K all | V all] stream
    # order. S_KV=512 -> per-head outer 512 <= 1023, so the 2-D wrap form is
    # legal without the collapse.
    _kv_outer = 2 * S_KV
    _kv_sizes = [1, 1, _kv_outer, D]
    _kv_strides = [0, 0, D, 1]
    if _kv_outer > 1023:
        # Collapse to 1-D: the interleaved [K|V] head is contiguous (stride D ==
        # inner size D), so a flattened stride-1 descriptor is valid.
        _kv_sizes = [1, 1, 1, _kv_outer * D]
        _kv_strides = [0, 0, 0, 1]
    KV_tap = [
        TensorAccessPattern(
            tensor_dims=L3_KV_ty.__args__[0],
            offset=c * (2 * S_KV * D),
            sizes=list(_kv_sizes),
            strides=list(_kv_strides),
        )
        for c in range(cols)
    ]
    if _separate:
        # per-BLOCK taps: the kernel consumes [K_block | V_block] tiles in block
        # order, so the K and V fills must ALTERNATE per block (round-major with
        # finish() per round — the swiglu weight-fill idiom). A single K-then-V
        # fill pair streams [K all][V all] and the kernel reads K's tokens
        # B_KV.. as V_b0 (the stage6 per-head cos collapse).
        # Per-round BD load: 2 fills per column (K+V), then finish() frees.
        _sep_sizes_b = [1, 1, B_KV, D]
        _sep_strides_b = [0, 0, D, 1]
        K_tap = [
            [
                TensorAccessPattern(
                    tensor_dims=L3_K_ty.__args__[0],
                    offset=c * (S_KV * D) + b * (B_KV * D),
                    sizes=list(_sep_sizes_b),
                    strides=list(_sep_strides_b),
                )
                for c in range(cols)
            ]
            for b in range(NB)
        ]
        V_tap = [
            [
                TensorAccessPattern(
                    tensor_dims=L3_V_ty.__args__[0],
                    offset=c * (S_KV * D) + b * (B_KV * D),
                    sizes=list(_sep_sizes_b),
                    strides=list(_sep_strides_b),
                )
                for c in range(cols)
            ]
            for b in range(NB)
        ]

    # Runtime sequence body — runs at resolve time inside the runtime_sequence op.
    # New API: Runtime(seq_fn, fn_args); fn_args type entries become RuntimeData
    # (fill/drain targets), other objects pass through to the body unchanged.
    # Bindings are POSITIONAL, so the two layouts get two distinct seq_fn's.
    rt_handles = [inQ[c].prod(tile=Tile(col=c, row=0)) for c in range(cols)] + [
        inKV[c].prod(tile=Tile(col=c, row=0)) for c in range(cols)
    ] + [
        outO[c].cons(tile=Tile(col=c, row=0)) for c in range(cols)
    ]

    if _separate:
        def seq_fn(Q, K, V, O, handles):
            h_inQ = handles[0:cols]
            h_inKV = handles[cols:2 * cols]
            h_outO = handles[2 * cols:3 * cols]
            if use_runtime_seq_len:
                sync_parameters()
                for c in range(cols):
                    worker_barriers[c].set(1)
            tg = TaskGroup()
            # M=1 only: Q fill, then the KV fills in NB rounds of K/V
            # ALTERNATION (round-major, finish() per round — shim BD pool is
            # per-shim-tile; 8 cols x 2 fills = 16 in flight then freed; the
            # swiglu weight-fill idiom). The kernel consumes [K_block|V_block]
            # tiles in block order, so a single K-then-V fill pair would stream
            # [K all][V all] and the kernel would read K's tokens B_KV.. as
            # V_b0 (the stage6 per-head cos collapse).
            for c in range(cols):
                h_inQ[c].fill(Q, tap=Q_tap[c], group=tg)
            for b in range(NB):
                tgw = TaskGroup()
                for c in range(cols):
                    h_inKV[c].fill(K, tap=K_tap[b][c], group=tgw)
                for c in range(cols):
                    h_inKV[c].fill(V, tap=V_tap[b][c], group=tgw)
                tgw.finish()
            for c in range(cols):
                h_outO[c].drain(O, tap=O_tap[c], wait=True, group=tg)
            tg.finish()

        rt = Runtime(seq_fn, [L3_Q_ty, L3_K_ty, L3_V_ty, L3_O_ty, rt_handles])
    else:
        def seq_fn(Q, KV, O, handles):
            h_inQ = handles[0:cols]
            h_inKV = handles[cols:2 * cols]
            h_outO = handles[2 * cols:3 * cols]
            if use_runtime_seq_len:
                sync_parameters()
                for c in range(cols):
                    worker_barriers[c].set(1)
            tg = TaskGroup()
            # Batch mode: per-group Q fills + KV fills + O drains (the core holds
            # only group g's Q/O tiles in L1 at a time; fifo depth-1 backpressure
            # sequences the groups).  Legacy: single Q/KV fill + single O drain.
            if batch:
                for g in range(PASS):
                    for c in range(cols):
                        h_inQ[c].fill(Q, tap=Q_tap[g][c], group=tg)
                    for c in range(cols):
                        h_inKV[c].fill(KV, tap=KV_tap[c], group=tg)
                for g in range(PASS):
                    for c in range(cols):
                        h_outO[c].drain(O, tap=O_tap[g][c], wait=True, group=tg)
            else:
                for c in range(cols):
                    h_inQ[c].fill(Q, tap=Q_tap[c], group=tg)
                # Single streaming pass: fill each head's WHOLE KV once (the
                # MemTile forward() streams it into NB blocks on the core, and
                # the online softmax folds each block into the running max/sum +
                # context accumulator as it arrives, so no second pass is
                # needed).
                for c in range(cols):
                    h_inKV[c].fill(KV, tap=KV_tap[c], group=tg)
                for c in range(cols):
                    h_outO[c].drain(O, tap=O_tap[c], wait=True, group=tg)
            tg.finish()

        rt = Runtime(seq_fn, [L3_Q_ty, L3_KV_ty, L3_O_ty, rt_handles])

    my_program = Program(dev, rt, workers=workers)
    # Hardware-trace hook (IRON_TRACE_SIZE / IRON_TRACE_NTILES or trace_size).
    # Route trace egress through a shim column the design does not use for DMA
    # (avoids masterset same-destination conflicts).
    import os as _os
    from iron.operators._trace import resolve_trace_size
    _ts = resolve_trace_size(trace_size)
    if _ts > 0:
        ntiles = max(0, int(_os.environ.get("IRON_TRACE_NTILES", "1")))
        egress = int(_os.environ.get("IRON_TRACE_EGRESS_COL", "7"))
        my_program.enable_trace(
            _ts, workers=list(workers)[:ntiles], egress_shim_col=egress
        )
    return my_program.resolve_program()
