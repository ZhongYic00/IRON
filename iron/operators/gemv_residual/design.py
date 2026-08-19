# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker


def my_matvec_residual(
    dev,
    cols,
    M,
    K,
    m_input,
    m_output=None,
    kernel_object="gemv_residual.o",
    verbose=False,
    func_prefix="",
):
    if m_output is None:
        m_output = m_input

    assert M % cols == 0
    assert (M // cols) % m_output == 0
    assert m_output % m_input == 0

    dtype_in = np.dtype[bfloat16]
    L1_A_ty = np.ndarray[(m_input, K), dtype_in]
    # B contains residual(M) + vec(K) — all cols share the same B
    L1_B_ty = np.ndarray[(M + K,), dtype_in]
    L1_C_ty = np.ndarray[(m_output,), dtype_in]
    L3_A_ty = np.ndarray[(M * K,), dtype_in]
    # B DDR layout: [residual(M) | vec(K)]
    L3_B_ty = np.ndarray[(M + K,), dtype_in]
    L3_C_ty = np.ndarray[(M,), dtype_in]

    matvec = Kernel(
        f"{func_prefix}matvec_residual_vectorized_bf16_bf16",
        f"{func_prefix}{kernel_object}",
        [np.int32, np.int32, np.int32, L1_A_ty, L1_B_ty, L1_C_ty],
    )

    A_L3L1_fifos = [ObjectFifo(L1_A_ty, name=f"A_{i}", depth=2) for i in range(cols)]
    B_L3L1_fifos = [ObjectFifo(L1_B_ty, name=f"B_{i}", depth=1) for i in range(cols)]
    C_L1L3_fifos = [ObjectFifo(L1_C_ty, name=f"C_{i}", depth=2) for i in range(cols)]

    def core_body(col_idx, A_fifo, B_fifo, C_fifo, matvec):
        one_idx = index.constant(1)
        col_offset_val = index.constant(col_idx * (M // cols))
        m_input_idx = index.constant(m_input)
        for _ in range_(0xFFFFFFFF):
            b = B_fifo.acquire(1)
            for i_idx in range_(M // m_output // cols):
                c = C_fifo.acquire(1)
                for j_idx in range_(m_output // m_input):
                    j_mul_m = index.mul(j_idx, m_input_idx)  # index type
                    local_offset = index.casts(T.i32(), j_mul_m)
                    global_offset = index.casts(T.i32(), index.add(col_offset_val, j_mul_m))
                    a = A_fifo.acquire(1)
                    matvec(m_input, local_offset, global_offset, a, b, c)
                    A_fifo.release(1)
                C_fifo.release(1)
            B_fifo.release(1)

    workers = [
        Worker(
            core_body,
            [i, A_L3L1_fifos[i].cons(), B_L3L1_fifos[i].cons(), C_L1L3_fifos[i].prod(), matvec],
        )
        for i in range(cols)
    ]

    A_taps = [
        TensorAccessPattern(
            tensor_dims=L3_A_ty.__args__[0],
            offset=col * (M // cols) * K,
            sizes=[1, 1, 1, (M // cols) * K],
            strides=[0, 0, 0, 1],
        )
        for col in range(cols)
    ]
    # All cols share the same B (residual+vec)
    B_tap = TensorAccessPattern(
        tensor_dims=L3_B_ty.__args__[0],
        offset=0,
        sizes=[1, 1, 1, M + K],
        strides=[0, 0, 0, 1],
    )
    C_taps = [
        TensorAccessPattern(
            tensor_dims=L3_C_ty.__args__[0],
            offset=col * (M // cols),
            sizes=[1, 1, 1, M // cols],
            strides=[0, 0, 0, 1],
        )
        for col in range(cols)
    ]

    rt = Runtime()
    with rt.sequence(L3_A_ty, L3_B_ty, L3_C_ty) as (A, B, C):
        rt.start(*workers)
        tg = rt.task_group()
        for col in range(cols):
            rt.fill(B_L3L1_fifos[col].prod(), B, B_tap, task_group=tg)
        for col in range(cols):
            rt.fill(A_L3L1_fifos[col].prod(), A, A_taps[col], task_group=tg)
        for col in range(cols):
            rt.drain(C_L1L3_fifos[col].cons(), C, C_taps[col], wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program()
