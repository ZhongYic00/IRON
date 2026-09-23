#!/usr/bin/env python3
"""Diagnostics for batch DecodeAttention failures.

Isolates:
  A. M=1 reduction (batch_queries=1, num_new=1, non-causal): must match legacy
     single-query full attention. If this fails, the batch plumbing (tap/types)
     is broken, not the m-loop.
  B. M=8, non-causal, per-row comparison: are all 16 output rows identical
     (=> Q rows not delivered per-token) or row-scrambled (=> index bug)?
"""
import sys, math
sys.path.insert(0, "/home/zyc/Github")
import torch

from aie.iron.device import NPU2
import aie.utils as aie_utils
aie_utils.set_current_device(NPU2())

from iron.common.context import AIEContext
from iron.common.sequence import OperatorSequence
from iron.operators.decode_attn.op import DecodeAttention


def pack_kv_interleaved(K, V, S_max, block_kv):
    KV_H, D = K.shape[0], K.shape[2]
    NB = S_max // block_kv
    blocks = []
    for b in range(NB):
        kb = K[:, b * block_kv:(b + 1) * block_kv, :].reshape(KV_H, block_kv * D)
        vb = V[:, b * block_kv:(b + 1) * block_kv, :].reshape(KV_H, block_kv * D)
        blocks += [kb, vb]
    return torch.cat(blocks, dim=1)


def ref_full(Q, K, V, S_kv, GQA, scale):
    """Q (n, H, d), full attention over [0, S_kv) for every query."""
    n, H, d = Q.shape
    out = torch.zeros(n, H, d)
    for m in range(n):
        for h in range(H):
            kv_h = h // GQA
            s = Q[m, h].float() @ K[kv_h, :S_kv, :].float().t() * scale
            out[m, h] = torch.softmax(s, dim=0) @ V[kv_h, :S_kv, :].float()
    return out


def run_case(name, M, num_new, S_kv_base, S_max=256, block_kv=32, iters=1):
    fix = torch.load("/home/zyc/Github/triton-xdna/examples/qwen3_0.6b/tests_mha/mha_fixture.pt")
    q1 = fix["q"]; K = fix["K"]; V = fix["V"]
    H, d = q1.shape
    GQA = H // K.shape[0]
    scale = 1.0 / math.sqrt(d)

    ctx = AIEContext(build_dir=f"/tmp/bd_diag_{name}")
    op = DecodeAttention(num_heads=H, num_kv_heads=K.shape[0], head_dim=d,
                         seq_len_kv=S_max, num_aie_columns=8, block_kv=block_kv,
                         use_runtime_seq_len=True,
                         batch_queries=M, num_new_tokens=num_new,
                         causal=False, context=ctx)
    seq = OperatorSequence(
        name=f"diag_{name}", runlist=[(op, "q", "kv", "o")],
        input_args=["q", "kv"], output_args=["o"], context=ctx)
    seq.compile()
    fc = seq.get_callable()
    print(f"[{name}] compiled OK")

    torch.manual_seed(0)
    Q = (q1.bfloat16().float() + 0.5 * torch.randn(num_new, H, d)).bfloat16().contiguous()
    kv = pack_kv_interleaved(K[:, :S_max, :].to(torch.bfloat16),
                             V[:, :S_max, :].to(torch.bfloat16), S_max, block_kv)
    fc.get_buffer("q").torch_view()[:] = Q.flatten()
    fc.get_buffer("kv").torch_view()[:] = kv.flatten()
    fc.get_buffer("q").to("npu")
    fc.get_buffer("kv").to("npu")
    fc.params.write("S_kv_base", S_kv_base)
    fc.params.sync()
    fc()
    fc.get_buffer("o").to("cpu")
    o = fc.get_buffer("o").torch_view().reshape(num_new, H, d).float()

    ref = ref_full(Q, K, V, S_kv_base + num_new, GQA, scale) \
        if False else ref_full(Q, K, V, S_max, GQA, scale)
    print(f"[{name}] per-row cosine vs own ref (full attn over {S_max}):")
    for m in range(min(num_new, 16)):
        c = torch.nn.functional.cosine_similarity(
            o[m].flatten(), ref[m].flatten(), dim=0).item()
        print(f"  row {m:2d}: cos={c:.4f}")
    # row-to-row self-similarity (are outputs identical across queries?)
    if num_new > 1:
        c00 = torch.nn.functional.cosine_similarity(
            o[0].flatten(), o[1].flatten(), dim=0).item()
        print(f"[{name}] cos(out[0], out[1]) = {c00:.4f} "
              f"({'IDENTICAL ROWS -> Q delivery bug' if c00 > 0.999 else 'rows differ'})")
    # per-head diag on row 0: which heads are right?
    heads_cos = [torch.nn.functional.cosine_similarity(
        o[0, h], ref[0, h], dim=0).item() for h in range(H)]
    print(f"[{name}] row0 per-head cos: {[f'{c:.2f}' for c in heads_cos]}")


if __name__ == "__main__":
    run_case("m1", M=1, num_new=1, S_kv_base=192)
    run_case("m8", M=8, num_new=16, S_kv_base=192)
