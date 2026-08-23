#!/usr/bin/env python3
"""Minimal compile + correctness check for the new QKNorm operator."""
import torch
import numpy as np
import ml_dtypes

from aie.iron.device import NPU2
import aie.utils as aie_utils
aie_utils.set_current_device(NPU2())

from iron.common.context import AIEContext
from iron.common.test_utils import run_test
from iron.operators.qk_norm.op import QKNorm

head_dim, n_q_heads, n_k_heads = 128, 16, 8
size = (n_q_heads + n_k_heads) * head_dim

ctx = AIEContext()
ctx.build_dir.mkdir(parents=True, exist_ok=True)

op = QKNorm(head_dim=head_dim, n_q_heads=n_q_heads, n_k_heads=n_k_heads,
            epsilon=1e-6, context=ctx)

x = torch.randn(size, dtype=torch.bfloat16)
q_gamma = torch.randn(head_dim, dtype=torch.bfloat16)
k_gamma = torch.randn(head_dim, dtype=torch.bfloat16)
gamma = torch.cat([q_gamma, k_gamma])

ref = op.reference(x, gamma)

errors, latency_us, _ = run_test(
    op,
    input_buffers={"in": x, "gamma": gamma},
    output_buffers={"out": ref},
    rel_tol=0.04,
    abs_tol=1e-6,
)
print(f"compiled OK, latency={latency_us:.2f}us")
print(f"errors per output buffer: {errors}")
print("PASS" if all(len(e) == 0 for e in errors.values()) else "FAIL")
