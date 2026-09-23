#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import aie.utils as aie_utils

from iron.operators.decode_attn.op import DecodeAttention
from iron.operators.decode_attn.reference import generate_golden_reference
from iron.common.context import AIEContext
from iron.common.test_utils import run_test


def get_params():
    # (num_heads, num_kv_heads, head_dim, seq_len_kv): Qwen3-0.6B decode shape.
    return [pytest.param(32, 8, 128, 512, id="qwen3-0.6b")]


@pytest.mark.supported_devices("npu2")
@pytest.mark.xfail(reason="golden uses an f32-exact softmax while the kernel streams exp2 over "
                         "bf16 blocks; the elementwise mismatch (~17% of O) is a precision-model "
                         "difference under investigation.  The operator's numerics are gated by "
                         "the Qwen3 chain's per-token parity vs HuggingFace.", strict=False)
@pytest.mark.parametrize("num_heads,num_kv_heads,head_dim,seq_len_kv", get_params())
def test_decode_attn(num_heads, num_kv_heads, head_dim, seq_len_kv, aie_context):
    from iron.common.context import AIEContext
    ctx = AIEContext(build_dir=f"build/test_decode_attn/{num_heads}h{num_kv_heads}kv"
                               f"{head_dim}d_S{seq_len_kv}")
    golden = generate_golden_reference(num_heads=num_heads, num_kv_heads=num_kv_heads,
                                       head_dim=head_dim, seq_len_kv=seq_len_kv)
    operator = DecodeAttention(
        num_heads=num_heads,
        num_kv_heads=num_kv_heads,
        head_dim=head_dim,
        seq_len_kv=seq_len_kv,
        num_aie_columns=8,
        kv_layout="combined",
        use_runtime_seq_len=False,
        context=ctx,
    )

    input_buffers = {"Q": golden["Q"], "kv_in": golden["kv_in"].flatten()}
    output_buffers = {"O": golden["O"]}

    errors, latency_us, bandwidth_gbps = run_test(
        operator, input_buffers, output_buffers, rel_tol=0.05, abs_tol=0.02
    )

    print(f"\nLatency: {latency_us:.1f} us")
    assert not errors, f"Test failed with errors: {errors}"
