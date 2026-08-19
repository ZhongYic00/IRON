# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker


def my_matvec_argmax(
    dev,
    cols,
    M,
    K,
    m_input,
    m_output=None,
    kernel_object="gemv_argmax.o",
    verbose=False,
    func_prefix="",
):
    if m_output is None:
        m_output = m_input

    assert M % cols == 0
    assert (M // cols) % m_output == 0
    assert m_output % m_input == 0

    dtype_in = np.dtype[np.float32]
    L1_A_ty = np.ndarray[(m_input, K), dtype_in]
    L1_B_ty = np.ndarray[(K,), dtype_in]
    # Output: 2 floats per col (max_val, argmax_idx)
    L1_C_ty = np.ndarray[(2,), dtype_in]
    L3_A_ty = np.ndarray[(M * K,), dtype_in]
    L3_B_ty = np.ndarray[(K,), dtype_in]
    L3_C_ty = np.ndarray[(cols * 2,), dtype_in]  # 2 per col

    matvec = Kernel(
        f"{func_prefix}matvec_argmax_f32_f32",
        f"{func_prefix}{kernel_object}",
        [np.int32, np.int32, np.int32, L1_A_ty, L1_B_ty, L1_C_ty],
    )

    A_L3L1_fifos = [ObjectFifo(L1_A_ty, name=f"A_{i}", depth=2) for i in range(cols)]
    B_L3L1_fifos = [ObjectFifo(L1_B_ty, name=f"B_{i}", depth=1) for i in range(cols)]
    # C depth=1: all sub-tile calls use the same buffer (no rotation)
    C_L1L3_fifos = [ObjectFifo(L1_C_ty, name=f"C_{i}", depth=1) for i in range(cols)]

    def core_body(col_idx, A_fifo, B_fifo, C_fifo, matvec):
        col_base = col_idx * (M // cols)
        col_base_idx = index.constant(col_base)
        m_input_idx = index.constant(m_input)
        for _ in range_(0xFFFFFFFF):
            b = B_fifo.acquire(1)
            for i_idx in range_(M // m_output // cols):
                c = C_fifo.acquire(1)
                # Initialize running max: c[0] = -inf, c[1] = 0
                # (ObjectFifo depth=2, first acquire gives fresh buffer — need init)
                # Actually we can't easily init here. The kernel reads c_out[0] and c_out[1]
                # on first call. We need c to start as [-inf, 0].
                # Workaround: use a large negative number. Set c via a write.
                # But we can't write to c before kernel. Let kernel handle init:
                # if col_offset == col_base (first sub-tile), init to -inf.
                for j_idx in range_(m_output // m_input):
                    j_mul_m = index.mul(j_idx, m_input_idx)
                    global_offset = index.add(col_base_idx, j_mul_m)
                    global_offset_i32 = index.casts(T.i32(), global_offset)
                    local_offset_i32 = index.casts(T.i32(), j_mul_m)
                    a = A_fifo.acquire(1)
                    matvec(m_input, global_offset_i32, local_offset_i32, a, b, c)
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
    B_tap = TensorAccessPattern(
        tensor_dims=L3_B_ty.__args__[0],
        offset=0,
        sizes=[1, 1, 1, K],
        strides=[0, 0, 0, 1],
    )
    # Each col writes 2 floats to its slot in the output
    C_taps = [
        TensorAccessPattern(
            tensor_dims=L3_C_ty.__args__[0],
            offset=col * 2,
            sizes=[1, 1, 1, 2],
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
