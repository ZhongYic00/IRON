#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Device tests for the data-parallel decode SwiGLU MLP (0.6B form, FF % D == 0).

Two arms ship and both are tested:
  * ``SwiGLUMLPDataParallel`` -- bf16 weights, one flat row-major buffer per matrix.
  * ``SwiGLUMLPDataParallelOurs`` -- int8 payload + per-(row, group-of-128) bf16 scales on the
    per-(column, tile) wire ``[tile_m*K u8 | tile_m*(K/128)*2 B scales]``, dequantized in
    registers by mv_int8.cc.

The golden mirrors the KERNEL chain's arithmetic, not the ideal math: the residual add and the
silu/sigmoid epilogue run in the bf16 vector domain (the silu is the tanh-based sigmoid the
kernel implements), the weighted RMSNorm accumulates in f32 and stores bf16, and every matvec
accumulates in f32 and rounds its result to bf16 at the store.  For the int8 arm the kernel also
folds the group scale into the activations BEFORE the mac (``xs = bf16(x * s)``); the golden
mirrors that premul rounding through ``matvec_int8`` -- comparing against an f32-exact dequantize
blames the kernel for the premul's own rounding on cancellation-heavy rows.

The chunked-down arm (FF not a multiple of D, wire4b.py) is intentionally NOT exercised here --
it exists for the Qwen3-4B chain and is covered by that chain's gate.  Likewise fuse_o.
"""

import numpy as np
import pytest
import torch
import aie.utils as aie_utils

from iron.common.context import AIEContext
from iron.common.test_utils import run_test
from iron.operators.swiglu_mlp_dp.op import SwiGLUMLPDataParallel
from iron.operators.swiglu_mlp_dp.op_ours import SwiGLUMLPDataParallelOurs

GROUP = 128


def get_params():
    """(D, FF, num_aie_columns, domain).

    The 0.6B decode shape (D=1024, FF=3072, R=FF/D=3) and a small shape; the small shape
    exercises the bf16 arm only because the int8 wire's 12-row weight tile is only 64-byte
    aligned at D=1024 (the tile stride is 12*(D + D/64) bytes).
    """
    return [
        pytest.param(1024, 3072, 8, "bf16", id="qwen3-0.6b-bf16"),
        pytest.param(512, 1536, 2, "bf16", id="small-bf16"),
        pytest.param(1024, 3072, 8, "int8_ours", id="qwen3-0.6b-int8-ours"),
        pytest.param(1024, 3072, 4, "int8_ours", id="qwen3-0.6b-int8-ours-c4"),
    ]


def matvec_bf16(w, x):
    """bf16 matvec as generic/mv.cc computes it: f32 accumulator, bf16 store."""
    return (w.to(torch.float32) @ x.to(torch.float32)).to(torch.bfloat16)


def matvec_int8(q, s_bits, x):
    """int8 matvec as mv_int8.cc computes it: the group scale is premultiplied into the
    activations (``xs = bf16(x * s)``), the sum accumulates in f32.

    ``q`` is (M, K) integer payload as f32; ``s_bits`` is the (M, K//128) scale table as uint16
    bf16 bit patterns."""
    s = torch.from_numpy(s_bits).view(torch.bfloat16).float()
    s_exp = s.repeat_interleave(x.numel() // s.shape[1], dim=1)
    xs = (x.to(torch.float32)[None, :] * s_exp).to(torch.bfloat16).to(torch.float32)
    return (q.to(torch.float32) * xs).sum(dim=1)


def silu_bf16_inplace(g):
    """silu_tile_bf16: sigmoid via tanh, every vector op rounding to bf16."""
    half = g.to(torch.float32) * 0.5
    tanh_half = torch.tanh(half).to(torch.bfloat16)
    one_plus = (tanh_half.to(torch.float32) + 1.0).to(torch.bfloat16)
    sig = (one_plus.to(torch.float32) * 0.5).to(torch.bfloat16)
    return (g.to(torch.float32) * sig.to(torch.float32)).to(torch.bfloat16)


def reference(cur, a, n_pf, matvec_g, matvec_u, matvec_d, eps=1e-5):
    """Golden for one dispatch, mirroring the device roundings listed in the module docstring.

    ``matvec_*`` take the bf16 activation and return the (FF,) resp. (D,) matvec result already
    rounded to bf16, so each arm's own scale/premul handling stays inside its matvec."""
    x1 = (cur.to(torch.float32) + a.to(torch.float32)).to(torch.bfloat16)
    rms = torch.sqrt((x1.to(torch.float32) ** 2).mean() + eps)
    hf = (x1.to(torch.float32) / rms * n_pf.to(torch.float32)).to(torch.bfloat16)
    g = matvec_g(hf)
    g_silu = silu_bf16_inplace(g)
    gh = (g_silu.to(torch.float32) * matvec_u(hf).to(torch.float32)).to(torch.bfloat16)
    d = matvec_d(gh)
    return (x1.to(torch.float32) + d.to(torch.float32)).to(torch.bfloat16)


def pack_int8_wire(w_u8, s_bits, cols, tile_m, k):
    """Host-side wire packing: per (column, tile) block
    ``[tile_m*K u8 payload | tile_m*(K/128)*2 B bf16 scale bits]``, column-major over M.

    ``tile_m`` must divide M//cols; at the 0.6B form the tiles are exact (no 64-byte pad)."""
    m = w_u8.shape[0]
    n_groups = k // GROUP
    tile_bytes = tile_m * (k + n_groups * 2)
    rows_per_col = m // cols
    tiles_per_col = rows_per_col // tile_m
    out = np.zeros(cols * tiles_per_col * tile_bytes, np.uint8)
    for c in range(cols):
        for t in range(tiles_per_col):
            r0 = c * rows_per_col + t * tile_m
            blk = out[(c * tiles_per_col + t) * tile_bytes:][:tile_bytes]
            blk[:tile_m * k] = w_u8[r0:r0 + tile_m].reshape(-1)
            s = s_bits[r0:r0 + tile_m].reshape(-1)
            blk[tile_m * k:tile_m * k + s.size * 2] = s.view(np.uint8)
    return out


def quantize_weights(w):
    """Symmetric per-(row, group-of-128) int8 quantization of an (M, K) bf16 weight matrix.

    Returns (u8 payload in the BIASED domain (q+128), scale bits as (M, K//128) uint16) -- the
    payload domain the production packers write and mv_int8.cc's biased default reads."""
    m, k = w.shape
    n_groups = k // GROUP
    groups = w.to(torch.float32).reshape(m, n_groups, GROUP)
    scale = (groups.abs().amax(dim=2) / 127.0).clamp(min=1e-8).to(torch.bfloat16)
    q = torch.round(w.to(torch.float32).reshape(m, n_groups, GROUP)
                    / scale.to(torch.float32)[:, :, None]).clamp(-127, 127).to(torch.int8)
    # biased payload domain: byte = q + 128 (the kernel's `add(-128)` recovers q)
    return ((q.reshape(m, k).numpy().astype(np.int16) + 128).astype(np.uint8),
            scale.contiguous().view(torch.uint16).numpy())


@pytest.mark.supported_devices("npu2")
@pytest.mark.parametrize("D,FF,num_aie_columns,domain", get_params())
def test_swiglu_mlp_dp(D, FF, num_aie_columns, domain, aie_context):
    torch.manual_seed(42)

    if domain == "bf16":
        wg = torch.randn(FF, D, dtype=torch.bfloat16) * 0.1
        wu = torch.randn(FF, D) * 0.1
        wd = torch.randn(D, FF) * 0.1
        operator = SwiGLUMLPDataParallel(D=D, FF=FF, num_aie_columns=num_aie_columns,
                                         num_aie_rows=1, context=AIEContext(
                                             build_dir=f"build/test_swiglu_mlp_dp/"
                                                        f"bf16_d{D}f{FF}c{num_aie_columns}"))
        wire = {"Wg": wg.reshape(-1), "Wu": wu.reshape(-1), "Wd": wd.reshape(-1)}
        matvec_g = lambda x: matvec_bf16(wg, x)
        matvec_u = lambda x: matvec_bf16(wu, x)
        matvec_d = lambda x: matvec_bf16(wd, x)
    else:
        wg_q, wg_s = quantize_weights(torch.randn(FF, D) * 0.1)
        wu_q, wu_s = quantize_weights(torch.randn(FF, D))
        wd_q, wd_s = quantize_weights(torch.randn(D, FF))
        operator = SwiGLUMLPDataParallelOurs(
            D=D, FF=FF, num_aie_columns=num_aie_columns, num_aie_rows=1,
            group_size=GROUP,
            context=AIEContext(build_dir=f"build/test_swiglu_mlp_dp/int8_ours_"
                                        f"d{D}f{FF}c{num_aie_columns}"))
        tsi_gu = operator.tile_rows_gu or 12
        tsi_d = tsi_gu // (FF // D)
        wire = {
            "Wg": torch.from_numpy(pack_int8_wire(wg_q, wg_s, num_aie_columns, tsi_gu, D)),
            "Wu": torch.from_numpy(pack_int8_wire(wu_q, wu_s, num_aie_columns, tsi_gu, D)),
            "Wd": torch.from_numpy(pack_int8_wire(wd_q, wd_s, num_aie_columns, tsi_d, FF)),
        }
        qg = torch.from_numpy(wg_q).float()
        qu = torch.from_numpy(wu_q).float()
        qd = torch.from_numpy(wd_q).float()
        matvec_g = lambda x: matvec_int8(qg, wg_s, x).to(torch.bfloat16)
        matvec_u = lambda x: matvec_int8(qu, wu_s, x).to(torch.bfloat16)
        matvec_d = lambda x: matvec_int8(qd, wd_s, x).to(torch.bfloat16)

    cur = torch.randn(D).to(torch.bfloat16) * 0.2
    a = torch.randn(D).to(torch.bfloat16) * 0.2
    n_pf = torch.rand(D) + 0.5

    golden_nxt = reference(cur, a, n_pf, matvec_g, matvec_u, matvec_d, eps=operator.epsilon)

    spec = operator.get_arg_spec()
    assert wire["Wg"].view(torch.bfloat16).numel() == spec[3].shape[0], (
        spec[3].shape, wire["Wg"].numel())

    input_buffers = {
        "cur": cur, "a": a, "n_pf": n_pf,
        "Wg": wire["Wg"].view(torch.bfloat16),
        "Wu": wire["Wu"].view(torch.bfloat16),
        "Wd": wire["Wd"].view(torch.bfloat16),
        "gh_scratch": torch.zeros(FF, dtype=torch.bfloat16),
    }
    output_buffers = {"nxt": golden_nxt}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.05, abs_tol=0.05
    )

    print(f"\nLatency: {latency_us:.1f} us")
    print(f"Effective Bandwidth: {bandwidth_gbps:.6e} GB/s\n")
    assert not errors, f"Test failed with errors: {errors}"
