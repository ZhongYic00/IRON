# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import numpy as np
from ml_dtypes import bfloat16

from aie.iron import (
    Kernel,
    ObjectFifo,
    ScratchpadParameter,
    Program,
    Runtime,
    Worker,
    Buffer,
    WorkerRuntimeBarrier,
)
from aie.helpers.taplib.tap import TensorAccessPattern
from aie.helpers.dialects.scf import _for as range_
from iron.operators._trace import maybe_enable_trace


def scale_param_design(
    dev,
    size,
    tile_size,
    trace_size,
    param_name,
    func_prefix="",
    kernel_obj_file="scale_param.o",
):
    dtype = bfloat16
    n_tiles = size // tile_size

    tile_ty = np.ndarray[(tile_size,), np.dtype[dtype]]
    tensor_ty = np.ndarray[(size,), np.dtype[dtype]]

    of_in = ObjectFifo(tile_ty, name="in_fifo")
    of_out = ObjectFifo(tile_ty, name="out_fifo")

    kernel = Kernel(
        f"{func_prefix}scale_by_param",
        f"{func_prefix}{kernel_obj_file}",
        [tile_ty, tile_ty, np.int32, np.int32],
    )

    # Runtime scalar read from the scratchpad (host-written per dispatch).
    factor_param = ScratchpadParameter(param_name, np.int32)

    def core_body(of_in, of_out, kernel, factor_src, barrier):
        barrier.wait_for_value(1)
        factor = factor_src.read()
        for _ in range_(n_tiles):
            elem_in = of_in.acquire(1)
            elem_out = of_out.acquire(1)
            kernel(elem_in, elem_out, factor, tile_size)
            of_in.release(1)
            of_out.release(1)

    barrier = WorkerRuntimeBarrier()
    worker = Worker(
        core_body,
        fn_args=[of_in.cons(), of_out.prod(), kernel, factor_param, barrier],
    )

    taps = [
        TensorAccessPattern(
            (1, size), i * tile_size, [1, 1, 1, tile_size], [0, 0, 0, 1]
        )
        for i in range(n_tiles)
    ]

    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty) as (A, C):
        maybe_enable_trace(rt, trace_size, [worker])
        rt.start(worker)
        rt.sync_parameters()
        rt.set_barrier(barrier, 1)
        tg = rt.task_group()
        for i in range(n_tiles):
            rt.fill(of_in.prod(), A, taps[i], task_group=tg)
            rt.drain(of_out.cons(), C, taps[i], wait=True, task_group=tg)
        rt.finish_task_group(tg)

    return Program(dev, rt).resolve_program()
