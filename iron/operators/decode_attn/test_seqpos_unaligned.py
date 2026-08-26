#!/usr/bin/env python3
"""Non-block-aligned seq_pos causal-mask correctness test for DecodeAttention.

Covers seq_pos values that are NOT multiples of block_kv (64), and values
smaller than one block (seq_pos < block_kv), which exercise the softmax tail /
short-mask path that a plain `for j + VEC <= seq_pos` loop would leave
un-normalized (denominator == 0 -> NaN).

Builds DecodeAttention once with seq_len_kv=256 and use_runtime_seq_len=True,
then dispatches it with many S_kv_eff values and checks each against a CPU
causal-attention reference (cosine > 0.99, no NaN).

The KV input must be INTERLEAVED per block:
    [K_block0 | V_block0 | K_block1 | V_block1 | ...]
"""
import os, sys, math
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
    torch.manual_seed(0)
    H, KV_H, d, block_kv, S_max = 16, 8, 128, 64, 256
    GQA = H // KV_H
    q = torch.randn(H, d)
    K = torch.randn(KV_H, S_max, d)
    V = torch.randn(KV_H, S_max, d)

    ctx = AIEContext(build_dir="/tmp/bd_seqpos_unaligned")
    op = DecodeAttention(num_heads=H, num_kv_heads=KV_H, head_dim=d,
                         seq_len_kv=S_max, num_aie_columns=8, block_kv=block_kv,
                         use_runtime_seq_len=True, context=ctx)
    seq = OperatorSequence(name="da_unaligned", runlist=[(op, "q", "kv", "o")],
                           input_args=["q", "kv"], output_args=["o"], context=ctx)
    seq.compile()
    fc = seq.get_callable()
    print("compiled OK (runtime seq_len, S_KV=256)", flush=True)

    fc.get_buffer("q").torch_view()[:] = q.bfloat16().flatten()
    kv = pack_kv_interleaved(K.to(torch.bfloat16), V.to(torch.bfloat16), S_max, block_kv)
    fc.get_buffer("kv").torch_view()[:] = kv.flatten()
    fc.get_buffer("q").to("npu")
    fc.get_buffer("kv").to("npu")

    all_ok = True
    # block-aligned, non-aligned, short (< block), and boundary values
    for sp in [1, 33, 64, 65, 100, 127, 128, 130, 191, 192, 255, 256]:
        ref = cpu_reference(q, K, V, sp, head_dim=d, GQA=GQA)
        fc.params.write("S_kv_eff", sp)
        fc.params.sync()
        fc()
        fc.get_buffer("o").to("cpu")
        o = fc.get_buffer("o").torch_view().reshape(H, d).float()
        n_nan = torch.isnan(o).sum().item()
        cos = torch.nn.functional.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item()
        ok = (n_nan == 0) and (cos > 0.99)
        all_ok = all_ok and ok
        print(f"seq_pos={sp:3d}: nan={n_nan} cosine={cos:.6f} {'OK' if ok else 'FAIL'}",
              flush=True)

    print("RESULT:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
