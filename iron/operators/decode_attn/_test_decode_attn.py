#!/usr/bin/env python3
"""Standalone correctness + latency test for DecodeAttention (fused decode attn).

Builds the operator through OperatorSequence, runs it on the MHA fixture
(q/K/V/seq_pos), and compares against a CPU reference (cosine > 0.99, no NaN).

The operator's KV input must be INTERLEAVED per block:
    [K_block0 | V_block0 | K_block1 | V_block1 | ...]
(one block = block_kv keys).  This test packs the fixture K/V into that layout.
"""
import os, sys, time, math
sys.path.insert(0, "/home/zyc/Github")
import torch
import numpy as np

from aie.iron.device import NPU2
import aie.utils as aie_utils
aie_utils.set_current_device(NPU2())

from iron.common.context import AIEContext
from iron.common.sequence import OperatorSequence
from iron.operators.decode_attn.op import DecodeAttention


def cpu_reference(q, K, V, S_kv, head_dim=128, GQA=2):
    scale = 1.0 / math.sqrt(head_dim)
    H = q.shape[0]
    out = torch.zeros(H, head_dim)
    for h in range(H):
        kv_h = h // GQA
        s = q[h].float() @ K[kv_h, :S_kv, :].float().t() * scale
        out[h] = torch.softmax(s, dim=0) @ V[kv_h, :S_kv, :].float()
    return out


def pack_kv_interleaved(K, V, S_kv, block_kv):
    """Pack (KV_H, S_kv, D) K/V into interleaved [K_b0|V_b0|K_b1|V_b1...] per head."""
    KV_H = K.shape[0]
    D = K.shape[2]
    NB = S_kv // block_kv
    blocks = []
    for b in range(NB):
        kb = K[:, b * block_kv:(b + 1) * block_kv, :].reshape(KV_H, block_kv * D)
        vb = V[:, b * block_kv:(b + 1) * block_kv, :].reshape(KV_H, block_kv * D)
        blocks += [kb, vb]
    return torch.cat(blocks, dim=1)


def main():
    fix = torch.load("/home/zyc/Github/triton-xdna/examples/qwen3_0.6b/tests_mha/mha_fixture.pt")
    q = fix["q"]; K = fix["K"]; V = fix["V"]; seq_pos = int(fix["seq_pos"])
    H, d = q.shape
    KV_H, S_full, _ = K.shape
    GQA = H // KV_H
    S_kv = seq_pos          # bake seq_len_kv = seq_pos (truncate K/V to valid prefix)
    block_kv = 64

    ref = cpu_reference(q, K, V, S_kv, head_dim=d, GQA=GQA)

    ctx = AIEContext(build_dir="/tmp/bd_final")
    op = DecodeAttention(num_heads=H, num_kv_heads=KV_H, head_dim=d,
                         seq_len_kv=S_kv, num_aie_columns=8, block_kv=block_kv,
                         context=ctx)
    seq = OperatorSequence(
        name="decode_attn_final",
        runlist=[(op, "q", "kv", "o")],
        input_args=["q", "kv"],
        output_args=["o"],
        context=ctx,
    )
    seq.compile()
    fc = seq.get_callable()
    print("compiled OK")

    fc.get_buffer("q").torch_view()[:] = q.bfloat16().flatten()
    kv = pack_kv_interleaved(K[:, :S_kv, :].to(torch.bfloat16),
                             V[:, :S_kv, :].to(torch.bfloat16), S_kv, block_kv)
    fc.get_buffer("kv").torch_view()[:] = kv.flatten()
    fc.get_buffer("q").to("npu")
    fc.get_buffer("kv").to("npu")

    # correctness
    fc()
    fc.get_buffer("o").to("cpu")
    o = fc.get_buffer("o").torch_view().reshape(H, d).float()
    n_nan = torch.isnan(o).sum().item()
    cos = torch.nn.functional.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item()
    hd = [(o[h] - ref[h]).abs().max().item() for h in range(H)]
    print(f"correctness: nan={n_nan} cosine={cos:.6f}")
    print(f"  per-head maxdiff: {[f'{x:.3f}' for x in hd]}")

    # latency
    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        fc()
    t1 = time.perf_counter()
    per = (t1 - t0) / iters * 1e6
    print(f"latency: {per:.1f} us/step")

    ok = (n_nan == 0) and (cos > 0.99)
    print("RESULT:", "PASS" if ok else "FAIL")


if __name__ == "__main__":
    main()
