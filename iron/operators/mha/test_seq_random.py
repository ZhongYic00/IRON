#!/usr/bin/env python3
"""Regression check: does MHA-as-sequence + scratchpad still produce correct
results on random (small-norm) data, like the pre-sequence-single-op path did
(which hit cosine 0.987)?  Distinguishes a sequence/scratchpad regression from
a fixture-data numerics issue."""
import os, sys, torch
sys.path.insert(0, "/home/zyc/Github")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import triton
from triton.backends.amd_triton_npu.driver import NPUDriver
triton.runtime.driver.set_active(NPUDriver())
from aie.iron.device import NPU2
import aie.utils as aie_utils
aie_utils.set_current_device(NPU2())

from iron.common.sequence import OperatorSequence
from iron.operators.mha.op import MHA


def run(seq_len, q, K, V, seq_pos):
    H, d = q.shape
    KV_H, S_kv, _ = K.shape
    mha = MHA(num_heads=H, seq_len=seq_len, d=d, num_KV_heads=KV_H,
              num_of_pipelines=1, seq_len_kv=S_kv)
    seq = OperatorSequence(
        name=f"mha_seq_{seq_len}", runlist=[(mha, "q", "k", "v", "o")],
        input_args=["q", "k", "v"], output_args=["o"], dispatch="fused")
    seq.compile()
    fc = seq.get_callable()
    fc.params.write("S_q_eff", seq_pos + 1)
    fc.params.write("S_kv_eff", seq_pos)
    fc.params.sync()
    sp = seq_len
    qb = torch.full((H, sp, d), -1e8, dtype=torch.bfloat16)
    qb[:, seq_pos, :] = q.bfloat16()
    fc.get_buffer("q").torch_view()[:] = qb.flatten()
    fc.get_buffer("k").torch_view()[:] = K.flatten().to(torch.bfloat16)
    fc.get_buffer("v").torch_view()[:] = V.flatten().to(torch.bfloat16)
    for n in ("q", "k", "v"):
        fc.get_buffer(n).to("npu")
    fc()
    fc.get_buffer("o").to("cpu")
    return fc.get_buffer("o").torch_view().reshape(H, sp, d)[:, seq_pos, :].float()


def ref(q, K, V, seq_pos, GQA):
    H, d = q.shape
    out = torch.zeros(H, d)
    for h in range(H):
        kv_h = h // GQA
        k = K[kv_h, :seq_pos, :].float()
        v = V[kv_h, :seq_pos, :].float()
        s = q[h].float() @ k.t() * (1 / d**0.5)
        out[h] = torch.softmax(s, dim=0) @ v
    return out


def main():
    torch.manual_seed(0)
    H, d, KV_H, S_kv, seq_pos = 16, 128, 8, 512, 128
    GQA = H // KV_H
    q = torch.randn(H, d)
    K = torch.randn(KV_H, S_kv, d)
    V = torch.randn(KV_H, S_kv, d)
    r = ref(q, K, V, seq_pos, GQA)
    for seq_len in [256, 512]:
        o = run(seq_len, q, K, V, seq_pos)
        n_nan = torch.isnan(o).sum().item()
        cos = torch.nn.functional.cosine_similarity(o.flatten(), r.flatten(), dim=0).item()
        print(f"RANDOM seq_len={seq_len}: nan={n_nan}, cosine={cos:.6f}")


if __name__ == "__main__":
    main()
