#!/usr/bin/env python3
"""Measure per-token decode latency (TPOT) of the whole-model Qwen3 NPU sequence.

Reuses the cached ELF (no recompile). Warms up, then times N decode steps.
"""
import time
import torch
import sys
sys.path.insert(0, "/home/zyc/Github/iron/iron/applications/qwen3_0.6b")
from qwen3_npu import Qwen3NPU, emb_dim

N = int(sys.argv[1]) if len(sys.argv) > 1 else 50

print("loading model (reusing cached ELF)...", flush=True)
t0 = time.perf_counter()
model = Qwen3NPU("/home/zyc/Packages/NPU/Qwen3-0.6B/model.safetensors")
print(f"load done in {(time.perf_counter()-t0)*1000:.0f} ms", flush=True)

# warmup
for _ in range(3):
    x = torch.randn(emb_dim, dtype=torch.bfloat16)
    model(x, seq_pos=10)

# timed loop at a fixed decode position (steady state)
seq_pos = 100
times = []
for _ in range(N):
    x = torch.randn(emb_dim, dtype=torch.bfloat16)
    t0 = time.perf_counter()
    model(x, seq_pos=seq_pos)
    times.append((time.perf_counter() - t0) * 1000)  # ms
    seq_pos += 1

times = torch.tensor(times)
print(f"N={N}, mean TPOT = {times.mean().item():.2f} ms, "
      f"min = {times.min().item():.2f}, max = {times.max().item():.2f}, "
      f"p50 = {torch.median(times).item():.2f}")
print(f"tokens/s = {1000/times.mean().item():.2f}")
