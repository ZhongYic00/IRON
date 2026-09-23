#!/usr/bin/env python3
"""Runtime seq_pos (causal mask) test for DecodeAttention.

Builds DecodeAttention once with seq_len_kv=S_MAX (the maximum), with
use_runtime_seq_len=True, then dispatches it several times with different
S_kv_eff values written via fc.params. Verifies each dispatch masks KV rows
[seq_pos, S_MAX) correctly against a CPU reference (cosine > 0.99, no NaN).

The KV input must be INTERLEAVED per block:
    [K_block0 | V_block0 | K_block1 | V_block1 | ...]
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


def pack_kv_interleaved(K, V, S_max, block_kv):
    """Pack (KV_H, S_max, D) K/V into interleaved [K_b0|V_b0|K_b1|V_b1...]."""
    KV_H = K.shape[0]
    D = K.shape[2]
    NB = S_max // block_kv
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
    block_kv = 64
    S_max = 256          # compile-time maximum; seq_pos rides runtime param
                          # (512 hits the per-tile 16-BD cap: 2 pass x NB=8 = 16 KV BDs + Q/O)
    assert S_full >= S_max, f"fixture has S={S_full} < S_max={S_max}"

    ctx = AIEContext(build_dir="/tmp/bd_runtime_seq")
    op = DecodeAttention(num_heads=H, num_kv_heads=KV_H, head_dim=d,
                         seq_len_kv=S_max, num_aie_columns=8, block_kv=block_kv,
                         use_runtime_seq_len=True, context=ctx)
    seq = OperatorSequence(
        name="decode_attn_runtime_seq",
        runlist=[(op, "q", "kv", "o")],
        input_args=["q", "kv"],
        output_args=["o"],
        context=ctx,
    )
    seq.compile()
    fc = seq.get_callable()
    print("compiled OK")

    assert fc.params is not None, "params.txt not generated (use_runtime_seq_len=True expected it)"

    # persistent Q / KV: KV covers all S_max rows with real data in [0, S_max)
    # (the tail beyond a given seq_pos is masked by the kernel at runtime).
    fc.get_buffer("q").torch_view()[:] = q.bfloat16().flatten()
    kv = pack_kv_interleaved(K[:, :S_max, :].to(torch.bfloat16),
                             V[:, :S_max, :].to(torch.bfloat16), S_max, block_kv)
    fc.get_buffer("kv").torch_view()[:] = kv.flatten()
    fc.get_buffer("q").to("npu")
    fc.get_buffer("kv").to("npu")

    all_ok = True
    for sp in [64, 128, 192, 256]:
        ref = cpu_reference(q, K, V, sp, head_dim=d, GQA=GQA)

        fc.params.write("S_kv_eff", sp)
        fc.params.sync()

        fc()
        fc.get_buffer("o").to("cpu")
        o = fc.get_buffer("o").torch_view().reshape(H, d).float()
        n_nan = torch.isnan(o).sum().item()
        cos = torch.nn.functional.cosine_similarity(
            o.flatten(), ref.flatten(), dim=0).item()
        md = (o - ref).abs().max().item()
        ok = (n_nan == 0) and (cos > 0.99)
        all_ok = all_ok and ok
        print(f"seq_pos={sp:3d}: nan={n_nan} cosine={cos:.6f} maxdiff={md:.3f} "
              f"{'OK' if ok else 'FAIL'}")
        sys.stdout.flush()

    # latency (seq_pos=512)
    fc.params.write("S_kv_eff", S_max)
    fc.params.sync()
    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        fc()
    t1 = time.perf_counter()
    print(f"latency (S_kv_eff={S_max}): {(t1 - t0) / iters * 1e6:.1f} us/step")

    print("RESULT:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
