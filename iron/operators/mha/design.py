# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import os
import sys
import math
import copy
from pathlib import Path

from ml_dtypes import bfloat16
import numpy as np

from aie.iron import (
    Kernel,
    ObjectFifo,
    Program,
    Runtime,
    Worker,
    Buffer,
    WorkerRuntimeBarrier,
    TaskGroup,
)
from aie.iron.device import NPU2, Tile
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorTiler2D, TensorAccessSequence, TensorAccessPattern
from aie.helpers.dialects.scf import if_, else_
from iron.operators._trace import resolve_trace_size

# ---------------------------------------------------------------------------
# mlir-aie 1.4.0 aiecc compatibility shim
#
# IRON's compilation rules emit flags (--no-compile-host,
# --aie-generate-xclbin, --aie-generate-npu-insts) that the 1.4.0 wheel's
# aiecc does not recognise. The 1.4.0 driver uses a graph-cut model:
# request outputs with --get-xclbin / --get-npu-insts and supply
# --xclbin-name / --npu-insts-name for the filenames.
#
# This shim patches ShellCompilationCommand.run to fix the command list
# right before the aiecc subprocess is launched. It is a no-op on newer
# mlir-aie versions (the stripped flags are simply absent; --get-* are
# supported by the newer driver too).
# ---------------------------------------------------------------------------
_aiecc_compat_done = False


def _patch_aiecc_for_1_4_0():
    global _aiecc_compat_done
    if _aiecc_compat_done:
        return
    _aiecc_compat_done = True

    import iron.common.compilation.base as _base

    _REMOVE_FLAGS = {
        "--no-compile-host",
        "--aie-generate-xclbin",
        "--aie-generate-npu-insts",
        "--no-compile",
    }

    def _fix_cmd(cmd_list):
        """Strip 1.4.0-incompatible flags and add --get-* shorthands."""
        fixed = [c for c in cmd_list if c not in _REMOVE_FLAGS]
        has_xclbin = any(c.startswith("--xclbin-name=") for c in fixed)
        has_insts = any(c.startswith("--npu-insts-name=") for c in fixed)
        if has_xclbin and "--get-xclbin" not in fixed:
            idx = next(
                i for i, c in enumerate(fixed) if c.startswith("--xclbin-name=")
            )
            fixed.insert(idx, "--get-xclbin")
        if has_insts and "--get-npu-insts" not in fixed:
            idx = next(
                i for i, c in enumerate(fixed) if c.startswith("--npu-insts-name=")
            )
            fixed.insert(idx, "--get-npu-insts")
        return fixed

    # Patch ShellCompilationCommand.run — this is called right before the
    # aiecc subprocess is launched, so the command list is fixed here
    # regardless of which CompilationRule created it.
    _orig_run = _base.ShellCompilationCommand.run

    def _patched_run(self):
        if hasattr(self, "command") and isinstance(self.command, list):
            # Only fix aiecc commands
            if self.command and "aiecc" in self.command[0]:
                self.command = _fix_cmd(self.command)
                # Ensure xclbinutil is on PATH (needed for --get-xclbin)
                xrt_bin = "/opt/xilinx/xrt/bin"
                if hasattr(self, "env") and isinstance(self.env, dict):
                    path_val = self.env.get("PATH", "")
                    if xrt_bin not in path_val:
                        self.env["PATH"] = xrt_bin + ":" + path_val
        return _orig_run(self)

    _base.ShellCompilationCommand.run = _patched_run


_patch_aiecc_for_1_4_0()

dtype_map = {
    "bf16": bfloat16,
    "f32": np.float32,
}

microkernel_mac_dim_map = {
    "npu": {
        "bf16": (4, 8, 4),
    },
    "npu1": {
        "bf16": (4, 8, 4),
    },
    "npu2": {
        "bf16": {
            # emulate_bf16_mmul_with_bfp16
            True: (8, 8, 8),
            False: (4, 8, 8),
        },
    },
}


def main():
    argparser = argparse.ArgumentParser(
        prog="AIE Matrix Multiplication MLIR Design (Single Core)",
        description="Emits MLIR code for a matrix multiplication design of the given input size",
    )
    argparser.add_argument("--heads", type=int, default=1)
    argparser.add_argument("--S_q", type=int, default=256)
    argparser.add_argument("--S_kv", type=int, default=256)
    argparser.add_argument("-d", type=int, default=64)
    argparser.add_argument("--B_q", type=int, default=64)
    argparser.add_argument("--B_kv", type=int, default=64)
    argparser.add_argument(
        "--num_KV_heads",
        type=int,
        default=2,
        help="Number of heads for Key-Value pairs",
    )
    argparser.add_argument("--number-of-pipeline", type=int, default=1)
    argparser.add_argument("--emulate-bf16-mmul-with-bfp16", type=bool, default=False)
    argparser.add_argument("--trace_size", type=int, default=0)
    argparser.add_argument(
        "--output-file-path",
        "-o",
        type=str,
        default="my_mha.mlir",
        help="Output file path for the generated MLIR module",
    )
    argparser.add_argument(
        "--verbose", action="store_true", help="Enable verbose output"
    )

    args = argparser.parse_args()
    dev = NPU2()

    maybe_module = fused_mha(
        dev=dev,
        heads=args.heads,
        S_q=args.S_q,
        S_kv=args.S_kv,
        d=args.d,
        B_q=args.B_q,
        B_kv=args.B_kv,
        number_of_pipelines=args.number_of_pipeline,
        num_KV_heads=args.num_KV_heads,
        emulate_bf16_mmul_with_bfp16=args.emulate_bf16_mmul_with_bfp16,
        trace_size=args.trace_size,
        verbose=args.verbose,
    )

    output_file_path = Path(args.output_file_path)

    with open(output_file_path, "w") as f:
        f.write(str(maybe_module))

    if args.verbose:
        print(f"MLIR module written to {output_file_path}")


def fused_mha(
    dev,
    heads: int,
    S_q: int,
    S_kv: int,
    d: int,
    B_q: int,
    B_kv: int,
    number_of_pipelines: int,
    num_KV_heads: int,
    emulate_bf16_mmul_with_bfp16: bool,
    trace_size: int = 0,
    verbose: bool = False,
    func_prefix: str = "",
):

    of_depth = 2
    # For d > 64, the V and O tiles are (d, B_kv) and (B_q, d) respectively,
    # doubling in size vs d=64. With depth=2 this overflows AIE L1 (64 KB).
    # Use depth=1 for the large PV-side fifos to stay within memory budget.
    pv_depth = 1 if d > 64 else of_depth
    vectorized = True
    enable_tracing = resolve_trace_size(trace_size) > 0
    dtype_str = "bf16"

    if number_of_pipelines > 6:
        number_of_pipelines_join_distribute = number_of_pipelines // 2
    else:
        number_of_pipelines_join_distribute = number_of_pipelines

    S_q_eff = S_q
    S_kv_eff = S_kv
    S_q_pad = (
        (S_q_eff + (B_q * number_of_pipelines - 1)) // (B_q * number_of_pipelines)
    ) * (B_q * number_of_pipelines)
    S_kv_pad = (
        (S_kv_eff + (B_kv * number_of_pipelines - 1)) // (B_kv * number_of_pipelines)
    ) * (B_kv * number_of_pipelines)
    num_q_blocks = S_q_pad // B_q
    num_kv_blocks = S_kv_pad // B_kv
    num_q_block_per_pipeline = num_q_blocks // number_of_pipelines

    # VJUNG: When the number of KV heads is 0, treat it as regular MHA (num_KV_heads == heads).
    # Otherwise, num_KV_heads < heads indicates GQA.
    if num_KV_heads == 0:
        num_KV_heads = heads

    assert (
        emulate_bf16_mmul_with_bfp16
    ), "Only emulate_bf16_mmul_with_bfp16=True is supported"

    # r, s, t are the dimensions required by the microkernel MAC instructions.
    mac_dims = microkernel_mac_dim_map["npu2"][dtype_str]
    r, s, t = mac_dims[emulate_bf16_mmul_with_bfp16]

    if verbose:
        print(f"Device: {dev}")
        print(f"Number of heads: {heads}")
        print(f"MHA Dimensions: S_q={S_q}, S_kv={S_kv}, d={d}, B_q={B_q}, B_kv={B_kv}")
        print(f"Padded Dimensions: S_q_pad={S_q_pad}, S_kv_pad={S_kv_pad}")
        print(f"Data type: {dtype_str}")
        print(f"Microkernel MAC dimensions: r={r}, s={s}, t={t}")
        print(f"Vectorized: {vectorized}")
        print(f"Enable tracing: {enable_tracing}")

    assert num_KV_heads > 0, "Number of KV heads must be greater than 0"
    assert heads > 0, "Number of heads must be greater than 0"
    assert (
        num_KV_heads <= heads
    ), "Number of KV heads must be less than or equal to number of heads"
    assert (
        heads % num_KV_heads == 0
    ), f"Number of heads ({heads}) must be divisible by number of KV heads ({num_KV_heads})"

    assert B_q % r == 0, f"B_q must be divisible by r ({B_q} % {r} != 0)"
    assert B_kv % t == 0, f"B_kv must be divisible by t ({B_kv} % {t} != 0)"
    assert d % s == 0, f"d must be divisible by s ({d} % {s} != 0)"

    assert S_q_pad % B_q == 0, "Padded S_q must be divisible by B_q"
    assert S_kv_pad % B_kv == 0, "Padded S_kv must be divisible by B_kv"

    dtype = dtype_map[dtype_str]

    inv_scale = (
        1 / np.sqrt(d)
    ) * 1.4453125  # 1.4453125 ≈ log2(e), converts softmax base

    # Tensors living in DRAM
    Q_ty = np.ndarray[
        (
            heads,
            S_q_pad,
            d,
        ),
        np.dtype[dtype],
    ]
    KV_ty = np.ndarray[
        (num_KV_heads, S_kv_pad * d),
        np.dtype[dtype],
    ]

    # Tensors living on the AIE-array
    q_ty = np.ndarray[(B_q, d), np.dtype[dtype]]
    k_ty = np.ndarray[(d, B_kv), np.dtype[dtype]]
    qk_ty = np.ndarray[(B_q, B_kv), np.dtype[dtype]]
    s_ty = np.ndarray[(4 * B_q,), np.dtype[dtype]]

    # AIE kernel declarations
    func_type = "" if vectorized else "_scalar"
    mha_obj = f"{func_prefix}mha.o"
    pt_obj = f"{func_prefix}mha_passThrough.o"
    zero_kernel = Kernel(f"{func_prefix}zero_{dtype_str}", mha_obj, [qk_ty])
    zero_kernel_pv = Kernel(f"{func_prefix}zero_{dtype_str}_pv", mha_obj, [q_ty])

    memcopy_kernel_scale = Kernel(
        f"{func_prefix}passThroughLine", pt_obj, [s_ty, s_ty, np.int32]
    )

    scale_buffer_init_kernel = Kernel(f"{func_prefix}init_scale_buffer", mha_obj, [s_ty, np.int32])

    partial_softmax_kernel = Kernel(
        f"{func_prefix}partial_softmax",
        mha_obj,
        [
            qk_ty,
            qk_ty,
            s_ty,
            np.ndarray[(2,), np.dtype[np.int32]],
            dtype,
            np.int32,
            np.int32,
            np.int32,
            np.int32,
        ],
    )

    matmul_QK = Kernel(
        f"{func_prefix}matmul_bf16_bf16_wrapper{func_type}",
        mha_obj,
        [q_ty, k_ty, qk_ty, np.ndarray[(2,), np.dtype[np.int32]]],
    )

    matmul_PV = Kernel(
        f"{func_prefix}matmul_PV",
        mha_obj,
        [
            qk_ty,
            k_ty,
            q_ty,
            s_ty,
            np.int32,
            np.int32,
            np.ndarray[(2,), np.dtype[np.int32]],
        ],
    )

    rescale_O = Kernel(
        f"{func_prefix}rescale_O",
        mha_obj,
        [q_ty, s_ty, np.int32, np.ndarray[(2,), np.dtype[np.int32]]],
    )

    # RoPE kernel — applies rotary position embedding to Q/K tiles in-place.
    # With identity cos/sin (cos=1, sin=0) this is a no-op; the kernel's
    # fast path detects identity and returns immediately.
    rope_kernel = Kernel(
        f"{func_prefix}rope_qk_bf16",
        mha_obj,
        [q_ty, k_ty, np.ndarray[(d,), np.dtype[dtype]], np.ndarray[(d,), np.dtype[dtype]], np.int32, np.int32],
    )

    # AIE-array data movement with object fifos
    q_dims = None
    if vectorized:
        q_dims = [(B_q // r, r * d), (d // s, s), (r, d), (s, 1)]

    inQ = ObjectFifo(
        np.ndarray[(number_of_pipelines_join_distribute * B_q, d), np.dtype[dtype]],
        name="inQ",
    )
    memQ = inQ.cons().split(
        offsets=[B_q * d * i for i in range(number_of_pipelines_join_distribute)],
        obj_types=[q_ty] * number_of_pipelines_join_distribute,
        names=[f"memQ{i}" for i in range(number_of_pipelines_join_distribute)],
        dims_to_stream=[q_dims] * number_of_pipelines_join_distribute,
        depths=[pv_depth] * number_of_pipelines_join_distribute,
        tile=Tile(col=6, row=1),
    )  # Split between N pipelines
    if number_of_pipelines > 6:
        inQ2 = ObjectFifo(
            np.ndarray[(number_of_pipelines_join_distribute * B_q, d), np.dtype[dtype]],
            name="inQ2",
        )
        memQ += inQ2.cons().split(
            offsets=[B_q * d * i for i in range(number_of_pipelines_join_distribute)],
            obj_types=[q_ty] * number_of_pipelines_join_distribute,
            names=[f"memQ2{i}" for i in range(number_of_pipelines_join_distribute)],
            dims_to_stream=[q_dims] * number_of_pipelines_join_distribute,
            depths=[pv_depth] * number_of_pipelines_join_distribute,
            tile=Tile(col=7, row=1),
        )  # Split between N pipelines

    # VJUNG: The SequentialPlacer will place all of these on the same MemTile if Placement is specified. We would need a list of placement in case of one-many or many-one.
    # I think the Sequential Placer will fail if we do a split/join with more than 6 I/Os cuz it tries to place them all on the same tile.

    # K is stored in column-major order
    k_dims = None
    if vectorized:
        k_dims = [(B_kv // t, t * d), (d // s, s), (t, d), (s, 1)]
    if number_of_pipelines == 1:
        # Decode: direct shim→core via core tile forward (no mem tile)
        # depth=2 on inK to hold cos/sin + K data for time-multiplexed RoPE
        # forward on core tile does dims_to_stream reshape
        inK = ObjectFifo(k_ty, name="inK", depth=2)
        memK = inK.cons().forward(
            name="memK",
            dims_to_stream=k_dims,
            depth=1,  # depth 1 on core tile (small L1)
        )
    else:
        # Prefill: shim→mem→core broadcast (original path)
        inK = ObjectFifo(
            k_ty,
            name="inK",
            depth=of_depth,
        )
        memK = inK.cons().forward(
            name="memK",
            dims_to_stream=k_dims,
            tile=Tile(col=3, row=1),
            depth=pv_depth,
        )  # Broadcast, give this handle to N pipelines

    v_dims = None
    if vectorized:
        # V is stored as (d, B_kv). Stream it transposed as (B_kv, d) for the
        # PV matmul which uses b_row_maj=true with DIM_K_PV=B_kv, DIM_N_PV=d.
        v_dims = [(B_kv // t, t * d), (d // s, s), (t, d), (s, 1)]

    if number_of_pipelines == 1:
        inV = ObjectFifo(k_ty, name="inV", depth=of_depth)
        memV = inV.cons().forward(
            name="memV",
            dims_to_stream=v_dims,
            depth=1,
        )
    else:
        inV = ObjectFifo(
            k_ty,
            name="inV",
            depth=of_depth,
        )
        memV = inV.cons().forward(
            name="memV",
            dims_to_stream=v_dims,
            tile=Tile(col=4, row=1),
            depth=pv_depth,
        )  # Broadcast, give this handle to N pipelines

    a_dims = None
    if vectorized:
        a_dims = [(B_q // r, r * B_kv), (r, t), (B_kv // t, r * t), (t, 1)]
    memA = []
    outA = []
    for i in range(number_of_pipelines):
        memA.append(ObjectFifo(qk_ty, depth=of_depth, name=f"memA{i}"))
        outA.append(
            memA[i]
            .cons()
            .forward(
                name=f"outA{i}",
                dims_to_stream=a_dims,
                depth=of_depth,
                # tile=Tile(col=i, row=1))
            )
        )  # Local to 1 pipeline

    memP = []
    outP = []
    for i in range(number_of_pipelines):
        memP.append(ObjectFifo(qk_ty, depth=of_depth, name=f"memP{i}"))
        outP.append(
            memP[i]
            .cons()
            .forward(
                name=f"outP{i}",
                dims_to_stream=a_dims,
                depth=of_depth,
                # tile=Tile(col=i, row=1)
            )
        )  # Local to 1 pipeline

    # Scale buffer for partial softmax
    scaleOF = []
    for i in range(number_of_pipelines):
        scaleOF.append(
            ObjectFifo(s_ty, depth=of_depth, name=f"scaleOF{i}")
        )  # Local to 1 pipeline

    o_dims = None
    if vectorized:
        o_dims = [(B_q // r, r * d), (r, t), (d // t, r * t), (t, 1)]
    memO = ObjectFifo(
        np.ndarray[(number_of_pipelines_join_distribute * B_q, d), np.dtype[dtype]],
        name="memO",
        dims_to_stream=o_dims,
    )
    outO = memO.prod().join(
        offsets=[B_q * d * i for i in range(number_of_pipelines_join_distribute)],
        obj_types=[q_ty] * number_of_pipelines_join_distribute,
        names=[f"outO{i}" for i in range(number_of_pipelines_join_distribute)],
        depths=[pv_depth] * number_of_pipelines_join_distribute,
        tile=Tile(col=6, row=1),
    )  # Join onto the output OF
    if number_of_pipelines > 6:
        memO2 = ObjectFifo(
            np.ndarray[(number_of_pipelines_join_distribute * B_q, d), np.dtype[dtype]],
            name="memO2",
            dims_to_stream=o_dims,
        )
        outO += memO2.prod().join(
            offsets=[B_q * d * i for i in range(number_of_pipelines_join_distribute)],
            obj_types=[q_ty] * number_of_pipelines_join_distribute,
            names=[f"outO2{i}" for i in range(number_of_pipelines_join_distribute)],
            depths=[pv_depth] * number_of_pipelines_join_distribute,
            tile=Tile(col=7, row=1),
        )

    def batched_matmul_qk(
        of_q,
        of_k,
        of_a_out,
        zero,
        matmul_QK,
        rope,
        cos_buf,
        sin_buf,
        q_block_bias,
        mha_rtps,
        barrier,
        idx_buffer,
    ):

        barrier.wait_for_value(1)

        loop_idx_q = mha_rtps[0]
        loop_idx_kv = mha_rtps[1]

        for _ in range_(sys.maxsize):

            idx_buffer[0] = 0
            idx_buffer[1] = q_block_bias

            for _ in range_(loop_idx_q):

                elem_in_q = of_q.acquire(1)

                for _ in range_(loop_idx_kv):

                    elem_in_k = of_k.acquire(1)
                    elem_a_out = of_a_out.acquire(1)

                    # Apply RoPE to Q/K tiles before matmul (no-op with identity cos/sin).
                    rope(elem_in_q, elem_in_k, cos_buf, sin_buf, B_q, d)

                    zero(elem_a_out)
                    matmul_QK(elem_in_q, elem_in_k, elem_a_out, idx_buffer)

                    of_k.release(1)
                    of_a_out.release(1)

                    idx_buffer[0] += 1
                idx_buffer[0] = 0
                idx_buffer[1] += number_of_pipelines

                of_q.release(1)

    def softmax(
        of_in_a,
        of_out_p,
        of_out_scale,
        partial_softmax,
        init_scale_buffer,
        memcopy_kernel_scale,
        q_block_bias,
        mha_rtps,
        barrier,
        idx_buffer,
        scale_buffer,
    ):

        # VJUNG: The index buffer count how many Q and KV block this worker has processed
        # From this info we can infer the position in A and P

        barrier.wait_for_value(1)

        loop_idx_q = mha_rtps[0]
        loop_idx_kv = mha_rtps[1]

        S_q_effective = mha_rtps[2]
        S_kv_effective = mha_rtps[3]

        for _ in range_(sys.maxsize):

            # VJUNG: Required otherwise the buffer is maintained when doing warmup!
            idx_buffer[0] = 0
            idx_buffer[1] = q_block_bias

            for _ in range_(loop_idx_q):

                init_scale_buffer(scale_buffer, B_q)

                for _ in range_(loop_idx_kv):

                    elt_of_out_p = of_out_p.acquire(1)
                    elt_of_in_a = of_in_a.acquire(1)
                    elt_of_out_scale = of_out_scale.acquire(1)

                    partial_softmax(
                        elt_of_in_a,
                        elt_of_out_p,
                        scale_buffer,
                        idx_buffer,
                        inv_scale,
                        B_q,
                        B_kv,
                        S_q_effective,
                        S_kv_effective,
                    )
                    memcopy_kernel_scale(scale_buffer, elt_of_out_scale, 4 * B_q)

                    of_in_a.release(1)
                    of_out_p.release(1)
                    of_out_scale.release(1)

                    idx_buffer[0] += 1
                idx_buffer[0] = 0
                idx_buffer[1] += number_of_pipelines

    def batched_matmul_pv(
        of_p,
        of_v,
        of_scale,
        of_o_out,
        zero,
        matmul_PV,
        rescale_O,
        q_block_bias,
        mha_rtps,
        barrier,
        idx_buffer,
    ):

        barrier.wait_for_value(1)

        loop_idx_q = mha_rtps[0]
        loop_idx_kv = mha_rtps[1]

        for _ in range_(sys.maxsize):

            # VJUNG: Required otherwise the buffer is maintained when doing warmup!
            idx_buffer[0] = 0
            idx_buffer[1] = q_block_bias

            for _ in range_(loop_idx_q):

                elem_o_out = of_o_out.acquire(1)

                zero(elem_o_out)

                ### First iteration, don't rescale O_{i-1}
                elem_in_p = of_p.acquire(1)
                elem_in_v = of_v.acquire(1)
                elt_of_out_scale = of_scale.acquire(1)

                matmul_PV(
                    elem_in_p,
                    elem_in_v,
                    elem_o_out,
                    elt_of_out_scale,
                    B_q,
                    0,
                    idx_buffer,
                )

                of_p.release(1)
                of_v.release(1)
                of_scale.release(1)

                idx_buffer[0] += 1
                ###

                with if_(loop_idx_kv > 2) as if_op:
                    for _ in range_(loop_idx_kv - 2):
                        elem_in_p = of_p.acquire(1)
                        elem_in_v = of_v.acquire(1)
                        elt_of_out_scale2 = of_scale.acquire(1)

                        matmul_PV(
                            elem_in_p,
                            elem_in_v,
                            elem_o_out,
                            elt_of_out_scale2,
                            B_q,
                            1,
                            idx_buffer,
                        )

                        of_p.release(1)
                        of_v.release(1)
                        of_scale.release(1)

                        idx_buffer[0] += 1

                ### Last iteration, final rescaling
                with if_(loop_idx_kv > 1) as if_op:
                    elem_in_p = of_p.acquire(1)
                    elem_in_v = of_v.acquire(1)
                    elt_of_out_scale3 = of_scale.acquire(1)

                    matmul_PV(
                        elem_in_p,
                        elem_in_v,
                        elem_o_out,
                        elt_of_out_scale3,
                        B_q,
                        1,
                        idx_buffer,
                    )
                    rescale_O(elem_o_out, elt_of_out_scale3, B_q, idx_buffer)

                    of_p.release(1)
                    of_v.release(1)
                    of_scale.release(1)

                    idx_buffer[0] += 1
                # else:
                with else_(if_op):
                    rescale_O(elem_o_out, elt_of_out_scale, B_q, idx_buffer)
                    idx_buffer[0] += 1
                ###

                idx_buffer[0] = 0
                idx_buffer[1] += number_of_pipelines

                of_o_out.release(1)

    # Runtime parameter for workers loop index
    # VJUNG: We need one Buffer per worker since they need to be placed
    mha_rtps_list = [
        [
            Buffer(
                np.ndarray[(4,), np.dtype[np.int32]],
                name=f"mha_rtpss_{i}_stage{j}",
                initial_value=None,
                use_write_rtp=True,
            )
            for i in range(number_of_pipelines)
        ]
        for j in range(3)
    ]

    worker_barrier_list = [
        [WorkerRuntimeBarrier(initial_value=0) for i in range(number_of_pipelines)]
        for j in range(3)
    ]

    # Local L1 Buffers for cos/sin on each matmul worker tile.
    # Identity values (cos=1, sin=0) make rope_kernel a no-op.
    cos_bufs = [
        Buffer(type=np.ndarray[(d,), np.dtype[dtype]], name=f"cos_l1_{i}",
               initial_value=np.ones(d, dtype=dtype),
               tile=Tile(col=i, row=2))
        for i in range(number_of_pipelines)
    ]
    sin_bufs = [
        Buffer(type=np.ndarray[(d,), np.dtype[dtype]], name=f"sin_l1_{i}",
               initial_value=np.zeros(d, dtype=dtype),
               tile=Tile(col=i, row=2))
        for i in range(number_of_pipelines)
    ]

    # Create worker from task
    matmul_workers = []
    softmax_workers = []
    matmul_pv_workers = []

    for i in range(number_of_pipelines):
        idx_buffer_qk = Buffer(
            initial_value=np.zeros(shape=(2,), dtype=np.int32),
            name=f"idx_buffer_qk_{i}",
        )
        matmul_workers.append(
            Worker(
                batched_matmul_qk,
                fn_args=[
                    memQ[i].cons(),
                    memK.cons(),
                    memA[i].prod(),
                    zero_kernel,
                    matmul_QK,
                    rope_kernel,
                    cos_bufs[i],
                    sin_bufs[i],
                    i,
                    mha_rtps_list[0][i],
                    worker_barrier_list[0][i],
                    idx_buffer_qk,
                ],
                stack_size=0xD00,
                tile=Tile(col=i, row=2),
                while_true=False,
            )
        )
        idx_buffer_softmax = Buffer(
            initial_value=np.zeros(shape=(2,), dtype=np.int32),
            name=f"idx_buffer_softmax_{i}",
        )
        scale_buffer_softmax = Buffer(
            initial_value=np.zeros(shape=(4 * B_q,), dtype=dtype),
            name=f"scale_buffer_softmax_{i}",
        )
        softmax_workers.append(
            Worker(
                softmax,
                fn_args=[
                    outA[i].cons(),
                    memP[i].prod(),
                    scaleOF[i].prod(),
                    partial_softmax_kernel,
                    scale_buffer_init_kernel,
                    memcopy_kernel_scale,
                    i,
                    mha_rtps_list[1][i],
                    worker_barrier_list[1][i],
                    idx_buffer_softmax,
                    scale_buffer_softmax,
                ],
                stack_size=0xD00,
                tile=Tile(col=i, row=3),
                while_true=False,
            )
        )
        idx_buffer_pv = Buffer(
            initial_value=np.zeros(shape=(2,), dtype=np.int32),
            name=f"idx_buffer_pv_{i}",
        )
        matmul_pv_workers.append(
            Worker(
                batched_matmul_pv,
                fn_args=[
                    outP[i].cons(),
                    memV.cons(),
                    scaleOF[i].cons(),
                    outO[i].prod(),
                    zero_kernel_pv,
                    matmul_PV,
                    rescale_O,
                    i,
                    mha_rtps_list[2][i],
                    worker_barrier_list[2][i],
                    idx_buffer_pv,
                ],
                stack_size=0xD00,
                tile=Tile(col=i, row=4),
                while_true=False,
            )
        )

    # Define tensor access patterns for inputs/outputs
    # A and B are tiled across M and N respectively, while C is tiled across M and N
    Q_tiles = TensorTiler2D.group_tiler(
        (heads * S_q_pad, d), (number_of_pipelines_join_distribute * B_q, d), (1, 1)
    )

    K_tiles = TensorTiler2D.group_tiler(
        (num_KV_heads * S_kv_pad, d), (S_kv_pad, d), (1, 1)
    )

    V_tiles = TensorTiler2D.group_tiler(
        (num_KV_heads * S_kv_pad, d), (S_kv_pad, d), (1, 1)
    )

    O_tiles = TensorTiler2D.group_tiler(
        (heads * S_q_pad, d), (number_of_pipelines_join_distribute * B_q, d), (1, 1)
    )

    def print_tap_seq_info(tap_seq, name):
        for idx, tap in enumerate(tap_seq):
            print(f"{name} tile {idx}:")
            print(f"  Offset: {tap.offset}")
            print(f"  Sizes: {tap.sizes}")
            print(f"  Strides: {tap.strides}")

    def legalize_tap(tap: TensorAccessPattern, max_dim_size: int):

        sizes = copy.deepcopy(tap._sizes)

        # Skip is no need to legalize
        if all(size <= max_dim_size for size in sizes):
            return tap

        # Check that the transfer is continuous
        for idx, stride in enumerate(tap._strides[:-1]):
            if stride != 0 and stride != tap._sizes[idx + 1]:
                raise ValueError(f"Cannot legalize DMA non-contiguous DMA transfer")
        assert tap._strides[-1] == 1, f"Cannot legalize DMA non-contiguous DMA transfer"

        tap._sizes = [1, 1, 1, math.prod(sizes)]
        tap._strides = [0, 0, 0, 1]

        return tap

    def legalize_tas(tas: TensorAccessSequence):

        max_dim_size = 1023  # Max DMA dimension size for memTile DMA on NPU2

        for tap in tas:
            tap = legalize_tap(tap, max_dim_size)

    legalize_tas(K_tiles)
    legalize_tas(V_tiles)

    if verbose:
        print(f"DMA Transfer Configuration: DRAM <-> Mem tile")
        # print_tap_seq_info(Q_tiles, "Q")
        print_tap_seq_info(K_tiles, "K")
        print_tap_seq_info(V_tiles, "V")
        # print_tap_seq_info(O_tiles, "O")

    # Pre-create ObjectFifo handles with shim tiles for runtime DMA
    # In mlir-aie 1.4.0, the shim tile is set on the handle via prod(tile=)/cons(tile=),
    # not on fill()/drain() as in the newer API.
    inQ_prod = inQ.prod(tile=Tile(col=4, row=0))
    inK_prod = inK.prod(tile=Tile(col=5, row=0))
    inV_prod = inV.prod(tile=Tile(col=6, row=0))
    memO_cons = memO.cons(tile=Tile(col=7, row=0))
    if number_of_pipelines > 6:
        inQ2_prod = inQ2.prod(tile=Tile(col=4, row=0))
        memO2_cons = memO2.cons(tile=Tile(col=7, row=0))

    # Collect runtime DMA handles so they get registered with the Runtime.
    # flatten_fn_args in Runtime._register_fn_args will find the ObjectFifoHandles
    # inside this nested list and bind their shim endpoints eagerly.
    rt_handles = [inQ_prod, inK_prod, inV_prod, memO_cons]
    if number_of_pipelines > 6:
        rt_handles += [inQ2_prod, memO2_cons]

    # Runtime sequence body — runs at resolve time inside the runtime_sequence op.
    # In mlir-aie 1.4.0, Runtime takes (seq_fn, fn_args) where fn_args entries
    # that are types become RuntimeData (fill/drain targets) and other objects
    # pass through to the body unchanged.
    def seq_fn(Q, K, V, O, handles):
        h_inQ_prod, h_inK_prod, h_inV_prod, h_memO_cons = handles[:4]
        if number_of_pipelines > 6:
            h_inQ2_prod, h_memO2_cons = handles[4], handles[5]

        # Set runtime parameters on worker buffers (was rt.inline_ops)
        for j in range(3):
            for i in range(number_of_pipelines):
                mha_rtps_list[j][i][0] = num_q_block_per_pipeline
                mha_rtps_list[j][i][1] = num_kv_blocks
                mha_rtps_list[j][i][2] = S_q_eff
                mha_rtps_list[j][i][3] = S_kv_eff

        # Set barriers to release workers (was rt.set_barrier)
        for j in range(3):
            for i in range(number_of_pipelines):
                worker_barrier_list[j][i].set(1)

        # Workers start automatically — no rt.start() needed in 1.4.0.
        # Trace is configured on Program, not Runtime, after construction.

        for head_idx in range(heads):

            kv_head_idx = head_idx // (heads // num_KV_heads)

            for q_block_idx in range(num_q_block_per_pipeline):

                # Initialize a group for parallel drain tasks, with fill
                # resources free'd when drains complete.
                tg = TaskGroup()

                if number_of_pipelines > 6:
                    h_inQ_prod.fill(
                        Q,
                        tap=Q_tiles[
                            2 * head_idx * num_q_block_per_pipeline + q_block_idx * 2
                        ],
                        group=tg,
                    )
                    h_inQ2_prod.fill(
                        Q,
                        tap=Q_tiles[
                            2 * head_idx * num_q_block_per_pipeline
                            + q_block_idx * 2
                            + 1
                        ],
                        group=tg,
                    )
                else:
                    h_inQ_prod.fill(
                        Q,
                        tap=Q_tiles[head_idx * num_q_block_per_pipeline + q_block_idx],
                        group=tg,
                    )

                # Throw on bd containing the full K and V in the object fifo,
                # then does it transfer chunks of inKV size at the time?
                h_inK_prod.fill(
                    K,
                    tap=K_tiles[kv_head_idx],
                    group=tg,
                )
                h_inV_prod.fill(
                    V,
                    tap=V_tiles[kv_head_idx],
                    group=tg,
                )

                if number_of_pipelines > 6:
                    h_memO_cons.drain(
                        O,
                        tap=O_tiles[
                            2 * head_idx * num_q_block_per_pipeline + q_block_idx * 2
                        ],
                        wait=True,
                        group=tg,
                    )
                    h_memO2_cons.drain(
                        O,
                        tap=O_tiles[
                            2 * head_idx * num_q_block_per_pipeline
                            + q_block_idx * 2
                            + 1
                        ],
                        wait=True,
                        group=tg,
                    )
                else:
                    h_memO_cons.drain(
                        O,
                        tap=O_tiles[head_idx * num_q_block_per_pipeline + q_block_idx],
                        wait=True,
                        group=tg,
                    )

                tg.finish()

    # Create the runtime with the sequence function and fn_args.
    # fn_args: type entries (Q_ty, KV_ty) become RuntimeData;
    # the rt_handles list passes through and its ObjectFifoHandles are registered.
    rt = Runtime(seq_fn, [Q_ty, KV_ty, KV_ty, Q_ty, rt_handles])

    # Create the program from the device type, runtime, and workers.
    # In 1.4.0, workers are passed to Program (not started from Runtime).
    dev_ty = NPU2()
    all_workers = matmul_workers + softmax_workers + matmul_pv_workers
    my_program = Program(dev_ty, rt, workers=all_workers)

    # Enable trace on Program (not Runtime) if requested.
    # In 1.4.0, enable_trace lives on Program and configures both the traced
    # workers' tiles and the Runtime's trace-buffer sequencing.
    ts = resolve_trace_size(trace_size)
    if ts > 0:
        ntiles = max(0, int(os.environ.get("IRON_TRACE_NTILES", "1")))
        my_program.enable_trace(
            ts,
            workers=list(all_workers)[:ntiles],
        )

    # Place components (assign them resources on the device) and generate an MLIR module
    module = my_program.resolve_program()
    return module
