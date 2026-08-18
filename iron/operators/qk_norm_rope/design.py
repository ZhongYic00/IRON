# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
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
    func_prefix="",
):
    dtype = bfloat16

    tensor_ty = np.ndarray[(qkv_dim,), np.dtype[dtype]]
    # Merge gamma + cos_sin into one ObjectFifo to stay within 2 MM2S channels
    gamma_sz = n_qk_heads * head_dim       # 24 * 128 = 3072
    cos_sin_sz = 2 * head_dim              # 2 * 128 = 256
    scratch_sz = gamma_sz + cos_sin_sz     # 3328
    scratch_ty = np.ndarray[(scratch_sz,), np.dtype[dtype]]

    of_in = ObjectFifo(tensor_ty, name="in", depth=2)
    of_out = ObjectFifo(tensor_ty, name="out", depth=2)
    of_scratch = ObjectFifo(scratch_ty, name="scratch", depth=1)

    kernel = Kernel(
        f"{func_prefix}qk_norm_rope_vcopy",
        f"{func_prefix}qk_norm_rope.o",
        [tensor_ty, tensor_ty, scratch_ty, np.float32, np.int32],
    )

    n_iters = 1

    def core_body(of_in, of_out, of_scratch, kernel):
        for _ in range_(n_iters):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            elem_scratch = of_scratch.acquire(1)
            kernel(elem_out, elem_in, elem_scratch, epsilon, head_dim)
            of_in.release(1)
            of_out.release(1)
            of_scratch.release(1)

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

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, scratch_ty) as (A, B, C):
        rt.start(my_worker)
        tg = rt.task_group()
        rt.fill(of_in.prod(), A, qkv_tap, task_group=tg)
        rt.fill(of_scratch.prod(), C, scratch_tap, task_group=tg)
        rt.drain(of_out.cons(), B, qkv_tap, wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program()
