#!/usr/bin/env python3
"""End-to-end ScratchpadParameter verification.

Wraps ScaleByParam in a single-step OperatorSequence and verifies the host can
write an int32 parameter per dispatch and that the kernel actually reads it.

Usage:
    cd /home/zyc/Github/iron
    PYTHONPATH="/home/zyc/Github/iron:/opt/xilinx/xrt/python" \
        python3 iron/operators/scale_param/test_end_to_end.py
"""

import os
import sys

sys.path.insert(0, "/home/zyc/Github")

import numpy as np
import torch

import triton
from triton.backends.amd_triton_npu.driver import NPUDriver
triton.runtime.driver.set_active(NPUDriver())

from aie.iron.device import NPU2
import aie.utils as aie_utils
aie_utils.set_current_device(NPU2())

from iron.common.sequence import OperatorSequence
from iron.operators.scale_param.op import ScaleByParam


def main():
    size = 64
    tile_size = 64

    op = ScaleByParam(size=size, tile_size=tile_size, param_name="scale_param")
    seq = OperatorSequence(
        name="scale_param_seq",
        runlist=[(op, "in", "out")],
        input_args=["in"],
        output_args=["out"],
        dispatch="fused",
    )
    seq.compile()

    call = seq.get_callable()

    # The callable exposes .params (ParameterScratchpad) once the ELF is built.
    params = call.params
    assert params is not None, "expected params.txt to be generated"

    in_buf = call.get_buffer("in")
    out_buf = call.get_buffer("out")
    in_buf.to("cpu")

    for factor in [2, 7, -3]:
        x = np.arange(size, dtype=np.float32).reshape(-1)
        # write input (bf16 view so it lands in the buffer in the right dtype)
        in_buf.torch_view()[:] = torch.from_numpy(x).to(torch.bfloat16).reshape(-1)
        in_buf.to("npu")

        # write the runtime parameter
        params.write("scale_param", factor)
        params.sync()

        call()

        out_buf.to("cpu")
        y = out_buf.torch_view().float().numpy().reshape(-1)
        expected = (x * factor).astype(np.float32)
        err = float(np.abs(y - expected).max())
        print(f"factor={factor:3d}: max_abs_err={err:.4f}  (y[:4]={y[:4].tolist()}, exp={expected[:4].tolist()})")

        if err > 0.5:
            print("  !! MISMATCH")
            return 1
    print("OK: scratchpad parameter reached the core and changed the output.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
