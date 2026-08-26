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
    verbose=False,
):
    """Fused decode attention (single query, M=1) in GEMV semantics.

    Each column owns `heads_per_col = H // cols` query heads sharing one KV head
    (GQA = H // KV).  K and V ride a single combined KV stream (2 input DMA
    channels total per core: Q + KV) to stay within the core's 2-in/2-out DMA
    budget.  Each (2*B_KV, D) block tile holds K_block (rows [0,B_KV)) and
    V_block (rows [B_KV,2*B_KV)).

    For each head the worker (1) streams KV blocks and computes scores from the
    K half, (2) two-pass softmax in L1, (3) re-streams KV blocks and accumulates
    the context GEMV from the V half over columns, (4) normalizes.  scores/p/
    out_acc live in small L1 Buffers; only q/KV/out traverse DRAM.
    """
    heads_per_col = H // cols
    assert H % cols == 0, "H must divide evenly across columns"
    assert KV == cols, "GQA layout assumes one KV head per column"
    assert S_KV % B_KV == 0, "S_KV must be a multiple of B_KV"
    NB = S_KV // B_KV

    dtype = bfloat16
    f32 = np.float32
    i32 = np.int32

    # DRAM tensor types (KV combined: K in [0,S_KV*D), V in [S_KV*D,2*S_KV*D))
    L3_Q_ty = np.ndarray[(H, D), np.dtype[dtype]]
    L3_KV_ty = np.ndarray[(KV, 2 * S_KV * D), np.dtype[dtype]]
    L3_O_ty = np.ndarray[(H, D), np.dtype[dtype]]

    # L1 tile types
    L1_Q_ty = np.ndarray[(heads_per_col, D), np.dtype[dtype]]
    L1_KV_ty = np.ndarray[(2 * B_KV, D), np.dtype[dtype]]  # [K_block | V_block]
    L1_O_ty = np.ndarray[(heads_per_col, D), np.dtype[dtype]]

    # persistent L1 buffers (heads_per_col heads per column).
    # scores is f32: raw q@k can be ~1e4, bf16 storage would destroy softmax.
    scores_ty = np.ndarray[(heads_per_col, S_KV), np.dtype[f32]]
    p_ty = np.ndarray[(heads_per_col, S_KV), np.dtype[dtype]]
    out_acc_ty = np.ndarray[(heads_per_col, D), np.dtype[f32]]
    sum_ty = np.ndarray[(heads_per_col,), np.dtype[f32]]

    # kernels
    scores_kernel = Kernel(
        f"{func_prefix}attn_scores_block", f"{func_prefix}{kernel_object}",
        [L1_Q_ty, i32, L1_KV_ty, scores_ty, i32, i32, f32],
    )
    softmax_kernel = Kernel(
        f"{func_prefix}attn_softmax_head", f"{func_prefix}{kernel_object}",
        [scores_ty, i32, p_ty, sum_ty, i32],
    )
    context_kernel = Kernel(
        f"{func_prefix}attn_context_block", f"{func_prefix}{kernel_object}",
        [p_ty, i32, L1_KV_ty, out_acc_ty, i32],
    )
    finalize_kernel = Kernel(
        f"{func_prefix}attn_finalize", f"{func_prefix}{kernel_object}",
        [out_acc_ty, i32, sum_ty, L1_O_ty],
    )
    zero_kernel = Kernel(
        f"{func_prefix}attn_zero_outacc", f"{func_prefix}{kernel_object}",
        [out_acc_ty, i32],
    )

    # scale = log2e / sqrt(D)
    scale = math.log2(math.e) / math.sqrt(D)

    # ObjectFifos per column.
    inQ = [ObjectFifo(L1_Q_ty, name=f"inQ_{c}", depth=2) for c in range(cols)]
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
    outO = [ObjectFifo(L1_O_ty, name=f"outO_{c}", depth=2) for c in range(cols)]

    # persistent buffers per column
    scores_buf = [
        Buffer(type=scores_ty, name=f"scores_{c}",
               initial_value=np.zeros((heads_per_col, S_KV), dtype=f32))
        for c in range(cols)
    ]
    p_buf = [
        Buffer(type=p_ty, name=f"p_{c}",
               initial_value=np.zeros((heads_per_col, S_KV), dtype=dtype))
        for c in range(cols)
    ]
    out_acc_buf = [
        Buffer(type=out_acc_ty, name=f"out_acc_{c}",
               initial_value=np.zeros((heads_per_col, D), dtype=f32))
        for c in range(cols)
    ]
    sum_buf = [
        Buffer(type=sum_ty, name=f"sum_{c}",
               initial_value=np.zeros((heads_per_col,), dtype=f32))
        for c in range(cols)
    ]

    def core_body(q_fifo, kv_fifo, o_fifo, scores, p, out_acc, sum_,
                  scores_k, sm_k, ctx_k, fin_k, zero_k, seq_kv_eff_param=None,
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
        # pass 1: scores for BOTH heads over the K half of each KV block
        for b in range_(NB):
            kv = kv_fifo.acquire(1)
            for h in range_(heads_per_col):
                scores_k(q, h, kv, scores, b, seq_pos, scale)
            kv_fifo.release(1)
        # softmax + reset accumulator for both heads (L1, no DRAM)
        for h in range_(heads_per_col):
            sm_k(scores, h, p, sum_, seq_pos)
            zero_k(out_acc, h)
        # pass 2: context for BOTH heads over the V half of each KV block
        for b in range_(NB):
            kv = kv_fifo.acquire(1)
            for h in range_(heads_per_col):
                ctx_k(p, h, kv, out_acc, b)
            kv_fifo.release(1)
        for h in range_(heads_per_col):
            fin_k(out_acc, h, sum_, o)
        q_fifo.release(1)
        o_fifo.release(1)

    # runtime seq_pos scratchpad parameter (decode full-ELF path)
    seq_kv_eff_param = (
        ScratchpadParameter("S_kv_eff", np.int32) if use_runtime_seq_len else None
    )
    # per-column barrier: registers an aie.lock on each core that the scratchpad
    # parameter sync releases (set_lock) after the host writes S_kv_eff. Mirrors
    # softmax's vector_size_parameter + WorkerRuntimeBarrier pairing.
    worker_barriers = [WorkerRuntimeBarrier(initial_value=0) for c in range(cols)]

    workers = [
        Worker(
            core_body,
            [
                inQ[c].cons(), memKV[c].cons(), outO[c].prod(),
                scores_buf[c], p_buf[c], out_acc_buf[c], sum_buf[c],
                scores_kernel, softmax_kernel, context_kernel, finalize_kernel,
                zero_kernel,
            ]
            + ([seq_kv_eff_param] if seq_kv_eff_param is not None else [])
            + ([worker_barriers[c]] if use_runtime_seq_len else []),
            while_true=True,
        )
        for c in range(cols)
    ]

    # Taps: Q and O are per-column 2-head slices; KV is streamed block-by-block,
    # once for the scores pass and once for the context pass.
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
    KV_tap = [
        TensorAccessPattern(
            tensor_dims=L3_KV_ty.__args__[0],
            offset=c * (2 * S_KV * D),
            sizes=[1, 1, 2 * S_KV, D],
            strides=[0, 0, D, 1],
        )
        for c in range(cols)
    ]

    # Runtime sequence body — runs at resolve time inside the runtime_sequence op.
    # New API: Runtime(seq_fn, fn_args); fn_args type entries become RuntimeData
    # (fill/drain targets), other objects pass through to the body unchanged.
    rt_handles = [inQ[c].prod(tile=Tile(col=c, row=0)) for c in range(cols)] + [
        inKV[c].prod(tile=Tile(col=c, row=0)) for c in range(cols)
    ] + [outO[c].cons(tile=Tile(col=c, row=0)) for c in range(cols)]

    def seq_fn(Q, KV, O, handles):
        h_inQ = handles[0:cols]
        h_inKV = handles[cols:2 * cols]
        h_outO = handles[2 * cols:3 * cols]

        # Sync the runtime seq_pos scratchpad (S_kv_eff) written by the host via
        # ParameterScratchpad, then release each column's barrier lock so the
        # persistent core reads the new seq_pos. The set(1) here is what the
        # scratchpad lowering pairs with the core's wait_for_value(1).
        if use_runtime_seq_len:
            sync_parameters()
            for c in range(cols):
                worker_barriers[c].set(1)

        tg = TaskGroup()
        for c in range(cols):
            h_inQ[c].fill(Q, tap=Q_tap[c], group=tg)
        # scores pass then context pass: re-fill each head's WHOLE KV once per
        # pass (the MemTile forward() streams it into NB blocks on the core).
        for _pass in range(2):
            for c in range(cols):
                h_inKV[c].fill(KV, tap=KV_tap[c], group=tg)
        for c in range(cols):
            h_outO[c].drain(O, tap=O_tap[c], wait=True, group=tg)
        tg.finish()

    rt = Runtime(seq_fn, [L3_Q_ty, L3_KV_ty, L3_O_ty, rt_handles])

    my_program = Program(dev, rt, workers=workers)
    return my_program.resolve_program()
