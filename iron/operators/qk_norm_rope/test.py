#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import aie.utils as aie_utils

from iron.operators.qk_norm_rope.op import QKNormRoPE
from iron.operators.qk_norm_rope.reference import generate_golden_reference
from iron.common.context import AIEContext
from iron.common.test_utils import run_test


def get_params():
    # (n_q_heads, n_kv_heads, head_dim): Qwen3-0.6B and Qwen3-4B decode shapes.
    return [
        pytest.param(16, 8, 128, id="qwen3-0.6b"),
        pytest.param(32, 8, 128, id="qwen3-4b"),
    ]


@pytest.mark.supported_devices("npu2")
@pytest.mark.parametrize("n_q_heads,n_kv_heads,head_dim", get_params())
def test_qk_norm_rope(n_q_heads, n_kv_heads, head_dim, aie_context):
    golden = generate_golden_reference(n_q_heads=n_q_heads, n_kv_heads=n_kv_heads,
                                       head_dim=head_dim)
    ctx = AIEContext(build_dir=f"build/test_qk_norm_rope/{n_q_heads}q{n_kv_heads}k"
                               f"{head_dim}d")
    qkv_dim = (n_q_heads + n_kv_heads + n_kv_heads) * head_dim
    operator = QKNormRoPE(
        qkv_dim=qkv_dim,
        head_dim=head_dim,
        n_qk_heads=n_q_heads + n_kv_heads,
        n_v_heads=n_kv_heads,
        epsilon=1e-6,
        num_aie_columns=1,
        kv_direct=False,
        context=ctx,
    )

    gamma_cos_sin = torch.cat([golden["gamma"], golden["cos"], golden["sin"]])
    input_buffers = {"qkv_in": golden["qkv_in"], "scratch": gamma_cos_sin}
    output_buffers = {"qkv_out": golden["qkv_out"]}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.05, abs_tol=0.02
    )

    print(f"\nLatency: {latency_us:.1f} us")
    assert not errors, f"Test failed with errors: {errors}"
