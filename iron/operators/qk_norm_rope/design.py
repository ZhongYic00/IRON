# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    ScratchpadParameter,
    TaskGroup,
    Worker,
    sync_parameters,
)
from aie.iron.device import NPU2
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib.tap import TensorAccessPattern
from ml_dtypes import bfloat16


def my_qk_norm_rope(
    dev,
    qkv_dim,
    head_dim,
    n_qk_heads,
    n_v_heads,
    epsilon,
    trace_size=0,
    kv_direct=False,
    max_seq_len=0,
    kv_offset_parameter="k_cache_offset",
    func_prefix="",
):
    dtype = bfloat16

    tensor_ty = np.ndarray[(qkv_dim,), np.dtype[dtype]]
    # Merge gamma + cos_sin into one ObjectFifo to stay within 2 MM2S channels
    gamma_sz = n_qk_heads * head_dim       # 24 * 128 = 3072
    cos_sin_sz = 2 * head_dim              # 2 * 128 = 256
    scratch_sz = gamma_sz + cos_sin_sz     # 3328
    scratch_ty = np.ndarray[(scratch_sz,), np.dtype[dtype]]

    # depth 1 everywhere: at the 4B shape (qkv 6144, scratch 5376) depth-2
    # in/out FIFOs overflow the 64KB tile L1 (~60KB+ of buffers + padding).
    # The op runs once per dispatch, so double buffering gains nothing.
    of_in = ObjectFifo(tensor_ty, name="in", depth=1)
    of_out = ObjectFifo(tensor_ty, name="out", depth=1)
    of_scratch = ObjectFifo(scratch_ty, name="scratch", depth=1)

    # kv_direct: the producer writes the KV-cache slice itself and one drain
    # scatters it, replacing the chain's separate merged-StridedCopy (`sc_kv`)
    # entry — one fewer op boundary / configure point per layer for a copy that
    # moves 4 KB.
    #
    # The fifo OBJECT is one whole (2, n_v_heads, head_dim) K|V slice (2048 bf16
    # at the 4B shape) laid out as sc_kv's input was — [K head | ... | V head |
    # ...], head_dim contiguous each — and it is drained with sc_kv's own tap:
    # sizes [1, 2, n_v_heads, head_dim] (4D-padded exactly as StridedCopy pads),
    # strides [0, head_dim, 2*max_seq_len*head_dim, 1], offset 0, plus the
    # `k_cache_offset` scratchpad parameter.  The parameter is an ELEMENT offset
    # (AIEX.td: "If used as an address offset on a BD, the parameter is a
    # multiple of the BD's element size") and the host writes
    # seq_pos*2*head_dim into it — the same value the chain wrote for sc_kv, so
    # the cache bytes are byte-identical to what the removed op produced.
    kv_ty = None
    of_kv = None
    kv_cache_ty = None
    kv_tap = None
    kv_off_param = None
    if kv_direct:
        assert max_seq_len > 0, "kv_direct needs the cache's max_seq_len for its tap"
        kv_obj_elems = 2 * n_v_heads * head_dim
        kv_ty = np.ndarray[(kv_obj_elems,), np.dtype[dtype]]
        kv_cache_ty = np.ndarray[
            (n_v_heads * 2 * max_seq_len * head_dim,), np.dtype[dtype]
        ]
        of_kv = ObjectFifo(kv_ty, name="kv", depth=1)
        kv_tap = TensorAccessPattern(
            (n_v_heads * 2 * max_seq_len * head_dim,),
            0,
            [1, 2, n_v_heads, head_dim],
            [0, head_dim, 2 * max_seq_len * head_dim, 1],
        )
        kv_off_param = ScratchpadParameter(kv_offset_parameter, np.int32)

    kernel_args = [tensor_ty, tensor_ty, scratch_ty]
    if kv_direct:
        kernel_args.append(kv_ty)
    kernel_args += [np.float32, np.int32, np.int32, np.int32]
    kernel = Kernel(
        f"{func_prefix}qk_norm_rope_kv_vcopy" if kv_direct
        else f"{func_prefix}qk_norm_rope_vcopy",
        f"{func_prefix}{'qk_norm_rope_kv' if kv_direct else 'qk_norm_rope'}.o",
        kernel_args,
    )

    n_iters = 1

    def core_body(of_in, of_out, of_scratch, kernel):
        for _ in range_(n_iters):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            elem_scratch = of_scratch.acquire(1)
            kernel(elem_out, elem_in, elem_scratch, epsilon, head_dim,
                   n_qk_heads, n_v_heads)
            of_in.release(1)
            of_out.release(1)
            of_scratch.release(1)

    def core_body_kv(of_in, of_out, of_scratch, of_kv, kernel):
        for _ in range_(n_iters):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            elem_scratch = of_scratch.acquire(1)
            elem_kv = of_kv.acquire(1)
            kernel(elem_out, elem_in, elem_scratch, elem_kv, epsilon, head_dim,
                   n_qk_heads, n_v_heads)
            of_in.release(1)
            of_out.release(1)
            of_scratch.release(1)
            of_kv.release(1)

    if kv_direct:
        my_worker = Worker(
            core_body_kv,
            [
                of_in.cons(),
                of_out.prod(),
                of_scratch.cons(),
                of_kv.prod(),
                kernel,
            ],
            stack_size=0x2000,
        )
    else:
        my_worker = Worker(
            core_body,
            [
                of_in.cons(),
                of_out.prod(),
                of_scratch.cons(),
                kernel,
            ],
            stack_size=0x2000,
        )

    qkv_tap = TensorAccessPattern((1, qkv_dim), 0, [1, 1, 1, qkv_dim], [0, 0, 0, 1])
    scratch_tap = TensorAccessPattern((1, scratch_sz), 0, [1, 1, 1, scratch_sz], [0, 0, 0, 1])

    def sequence(A, B, C, in_prod, out_cons, scratch_prod):
        tg = TaskGroup()
        in_prod.fill(A, qkv_tap, group=tg)
        scratch_prod.fill(C, scratch_tap, group=tg)
        out_cons.drain(B, qkv_tap, wait=True, group=tg)
        tg.finish()

    def sequence_kv(A, B, C, D, in_prod, out_cons, scratch_prod, kv_prod):
        # The offset parameter must be latched into the cores'/DMA view of the
        # scratchpad before the drain's BD is issued (StridedCopy idiom).
        sync_parameters()
        tg = TaskGroup()
        in_prod.fill(A, qkv_tap, group=tg)
        scratch_prod.fill(C, scratch_tap, group=tg)
        out_cons.drain(B, qkv_tap, wait=True, group=tg)
        kv_prod.drain(D, kv_tap, wait=True, group=tg,
                      offset_parameter=kv_off_param)
        tg.finish()

    if kv_direct:
        rt = Runtime(
            sequence_kv,
            [tensor_ty, tensor_ty, scratch_ty, kv_cache_ty,
             of_in.prod(), of_out.cons(), of_scratch.prod(), of_kv.prod()],
        )
    else:
        # Unchanged from before kv_direct existed: same fifos, same taps, same
        # runtime sequence shape => byte-identical generated MLIR.
        rt = Runtime(
            sequence,
            [tensor_ty, tensor_ty, scratch_ty,
             of_in.prod(), of_out.cons(), of_scratch.prod()],
        )
    return Program(dev, rt, workers=[my_worker]).resolve_program()
