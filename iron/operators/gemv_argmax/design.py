# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np

import aie.dialects.index as index
from aie.dialects.aie import T
from aie.helpers.dialects.scf import _for as range_
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.buffer import Buffer


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
    assert M % cols == 0
    m_per_col = M // cols
    if m_output is None:
        m_output = m_input
    assert (m_per_col) % m_output == 0
    assert m_output % m_input == 0

    dtype_in = np.dtype[np.float32]
    L1_A_ty = np.ndarray[(m_input, K), dtype_in]
    L1_B_ty = np.ndarray[(K,), dtype_in]
    L1_C_ty = np.ndarray[(2,), dtype_in]
    # Running max buffer: persistent L1 memory, not ObjectFifo
    RunMax_ty = np.ndarray[(2,), dtype_in]  # [max_val, argmax_idx]
    L3_A_ty = np.ndarray[(M * K,), dtype_in]
    L3_B_ty = np.ndarray[(K,), dtype_in]
    L3_C_ty = np.ndarray[(cols * 2,), dtype_in]

    matvec = Kernel(
        f"{func_prefix}matvec_argmax_f32_f32",
        f"{func_prefix}{kernel_object}",
        [np.int32, np.int32, np.int32, L1_A_ty, L1_B_ty, RunMax_ty],
    )

    A_L3L1_fifos = [ObjectFifo(L1_A_ty, name=f"A_{i}", depth=2) for i in range(cols)]
    B_L3L1_fifos = [ObjectFifo(L1_B_ty, name=f"B_{i}", depth=1) for i in range(cols)]
    C_L1L3_fifos = [ObjectFifo(L1_C_ty, name=f"C_{i}", depth=1) for i in range(cols)]

    # Persistent L1 buffers for running max — one per column
    init_val = np.array([-1e30, 0.0], dtype=np.float32)
    running_max_bufs = [
        Buffer(RunMax_ty, initial_value=init_val, name=f"rmax_{i}")
        for i in range(cols)
    ]

    def core_body(col_idx, A_fifo, B_fifo, C_fifo, rmax_buf, matvec):
        col_base = col_idx * m_per_col
        col_base_idx = index.constant(col_base)
        m_input_idx = index.constant(m_input)
        m_output_idx = index.constant(m_output)
        for _ in range_(0xFFFFFFFF):
            b = B_fifo.acquire(1)
            for i_idx in range_(m_per_col // m_output):
                c = C_fifo.acquire(1)
                # i_idx * m_output = base offset for this output tile
                i_mul_mout = index.mul(i_idx, m_output_idx)
                for j_idx in range_(m_output // m_input):
                    j_mul_m = index.mul(j_idx, m_input_idx)
                    # Global offset: col_base + i_idx*m_output + j_idx*m_input
                    global_offset = index.add(index.add(col_base_idx, i_mul_mout), j_mul_m)
                    global_offset_i32 = index.casts(T.i32(), global_offset)
                    # local_offset: 0 only on very first sub-tile call (i=0, j=0)
                    # used by kernel to init running max
                    local_offset = index.add(index.mul(i_idx, index.constant(m_output // m_input)), j_idx)
                    local_offset_i32 = index.casts(T.i32(), local_offset)
                    a = A_fifo.acquire(1)
                    matvec(m_input, global_offset_i32, local_offset_i32, a, b, rmax_buf)
                    A_fifo.release(1)
                # Copy final running max to output buffer
                c[0] = rmax_buf[0]
                c[1] = rmax_buf[1]
                C_fifo.release(1)
            B_fifo.release(1)

    workers = [
        Worker(
            core_body,
            [
                i,
                A_L3L1_fifos[i].cons(),
                B_L3L1_fifos[i].cons(),
                C_L1L3_fifos[i].prod(),
                running_max_bufs[i],
                matvec,
            ],
        )
        for i in range(cols)
    ]

    # Each col gets its full matrix block via DMA — ObjectFifo streams it
    # in chunks of m_input×K (the ObjectFifo element size).
    A_taps = [
        TensorAccessPattern(
            tensor_dims=L3_A_ty.__args__[0],
            offset=col * m_per_col * K,
            sizes=[1, 1, 1, m_per_col * K],
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
