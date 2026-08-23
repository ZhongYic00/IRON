# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib.tap import TensorAccessPattern
from ml_dtypes import bfloat16


def qk_norm(
    dev,
    head_dim,
    n_q_heads,
    n_k_heads,
    epsilon,
    trace_size=0,
    func_prefix="",
):
    dtype = bfloat16
    n_qk_heads = n_q_heads + n_k_heads
    size = n_qk_heads * head_dim
    gamma_sz = 2 * head_dim  # [q_gamma | k_gamma]

    tensor_ty = np.ndarray[(size,), np.dtype[dtype]]
    gamma_ty = np.ndarray[(gamma_sz,), np.dtype[dtype]]

    of_in = ObjectFifo(tensor_ty, name="in", depth=2)
    of_out = ObjectFifo(tensor_ty, name="out", depth=2)
    of_gamma = ObjectFifo(gamma_ty, name="gamma", depth=1)

    kernel = Kernel(
        f"{func_prefix}qk_norm_bf16",
        f"{func_prefix}qk_norm.o",
        [tensor_ty, tensor_ty, gamma_ty, np.float32, np.int32, np.int32, np.int32],
    )

    n_iters = 1

    def core_body(of_in, of_out, of_gamma, kernel):
        for _ in range_(n_iters):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            elem_gamma = of_gamma.acquire(1)
            kernel(elem_out, elem_in, elem_gamma, epsilon, head_dim, n_q_heads, n_k_heads)
            of_in.release(1)
            of_out.release(1)
            of_gamma.release(1)

    my_worker = Worker(
        core_body,
        [
            of_in.cons(),
            of_out.prod(),
            of_gamma.cons(),
            kernel,
        ],
        stack_size=0x2000,
    )

    qk_tap = TensorAccessPattern((1, size), 0, [1, 1, 1, size], [0, 0, 0, 1])
    gamma_tap = TensorAccessPattern((1, gamma_sz), 0, [1, 1, 1, gamma_sz], [0, 0, 0, 1])

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, gamma_ty) as (A, B, C):
        rt.start(my_worker)
        tg = rt.task_group()
        rt.fill(of_in.prod(), A, qk_tap, task_group=tg)
        rt.fill(of_gamma.prod(), C, gamma_tap, task_group=tg)
        rt.drain(of_out.cons(), B, qk_tap, wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program()
