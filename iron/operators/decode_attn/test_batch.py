#!/usr/bin/env python3
"""Batch (multi-query) test for DecodeAttention — the DFlash verify path.

Builds DecodeAttention with batch_queries=8, num_new_tokens=16 (2 groups of 8),
S_max=256, block_kv=32 (batch L1 budget), and dispatches it with
S_kv_base=192 (new tokens at absolute positions 192..207).

Checks (vs CPU reference):
  1. causal=True:  query m at position base+m attends exactly [0, base+m]
     (per-query causal boundary S_kv_base + g*M + m + 1).
  2. causal=False: every query attends the FULL cache (DFlash draft semantics).
  3. The two outputs differ from each other (the mask actually does something).

Latency: batch dispatch vs a reference M=1 dispatch on the same fixture.
KV input is INTERLEAVED per block: [K_b0 | V_b0 | K_b1 | V_b1 | ...].
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


def cpu_reference_batch(Q, K, V, S_kv_base, num_new, S_max, causal, head_dim=128, GQA=2):
    """Reference: num_new tokens at positions S_kv_base..S_kv_base+num_new-1.

    causal: query m attends [0, S_kv_base+m]  (p+1 keys)
    non-causal: every query attends [0, S_max)
    """
    H = Q.shape[1]
    scale = 1.0 / math.sqrt(head_dim)
    out = torch.zeros(num_new, H, head_dim)
    for m in range(num_new):
        n_keys = S_max if not causal else S_kv_base + m + 1
        for h in range(H):
            kv_h = h // GQA
            s = Q[m, h].float() @ K[kv_h, :n_keys, :].float().t() * scale
            out[m, h] = torch.softmax(s, dim=0) @ V[kv_h, :n_keys, :].float()
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


def build_and_run(causal, S_kv_base, num_new=16, M=8, S_max=256, block_kv=32, iters=50):
    fix = torch.load("/home/zyc/Github/triton-xdna/examples/qwen3_0.6b/tests_mha/mha_fixture.pt")
    q1 = fix["q"]; K = fix["K"]; V = fix["V"]
    if os.environ.get("T_RANDDATA"):
        torch.manual_seed(7)
        q1 = torch.randn_like(q1) * 0.5
        K = torch.randn_like(K) * 0.3
        V = torch.randn_like(V) * 0.3
    H, d = q1.shape
    KV_H = K.shape[0]
    GQA = H // KV_H

    tag = "causal" if causal else "noncausal"
    ctx = AIEContext(build_dir=f"/tmp/bd_batch_{tag}")
    op = DecodeAttention(num_heads=H, num_kv_heads=KV_H, head_dim=d,
                         seq_len_kv=S_max, num_aie_columns=8, block_kv=block_kv,
                         use_runtime_seq_len=True,
                         batch_queries=M, num_new_tokens=num_new,
                         causal=causal, context=ctx)
    seq = OperatorSequence(
        name=f"decode_attn_batch_{tag}",
        runlist=[(op, "q", "kv", "o")],
        input_args=["q", "kv"],
        output_args=["o"],
        context=ctx,
    )
    seq.compile()
    fc = seq.get_callable()
    print(f"[{tag}] compiled OK")
    assert fc.params is not None

    # 16 distinct queries: fixture q for token 0, small random offsets otherwise
    torch.manual_seed(0)
    Q = q1.bfloat16().unsqueeze(0).repeat(num_new, 1, 1).contiguous()
    Q[1:] = (q1.bfloat16().float() + 0.5 * torch.randn(num_new - 1, H, d)).bfloat16()

    # Sanitize the fixture tail: rows >= 192 have V==0 exactly and huge K
    # (degenerate: softmax collapses onto V=0 rows, making causal vs
    # non-causal vacuous). Replace with O(1) random data.
    K = K.clone()
    V = V.clone()
    K[:, 192:, :] = 0.5 * torch.randn_like(K[:, 192:, :])
    V[:, 192:, :] = 0.5 * torch.randn_like(V[:, 192:, :])
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

    ref = cpu_reference_batch(Q, K, V, S_kv_base, num_new, S_max, causal,
                              head_dim=d, GQA=GQA)
    n_nan = torch.isnan(o).sum().item()
    cos = torch.nn.functional.cosine_similarity(
        o.flatten(), ref.flatten(), dim=0).item()
    md = (o - ref).abs().max().item()
    ok = (n_nan == 0) and (cos > 0.99)
    print(f"[{tag}] base={S_kv_base} NEW={num_new} M={M}: nan={n_nan} "
          f"cosine={cos:.6f} maxdiff={md:.4f} {'OK' if ok else 'FAIL'}")

    # latency
    t0 = time.perf_counter()
    for _ in range(iters):
        fc()
    t1 = time.perf_counter()
    print(f"[{tag}] latency: {(t1 - t0) / iters * 1e6:.1f} us/dispatch")
    return ok, o


def main():
    ok_c, o_c = build_and_run(causal=True, S_kv_base=192)
    ok_n, o_n = build_and_run(causal=False, S_kv_base=192)
    # masks must actually differ
    d_masks = (o_c - o_n).abs().max().item()
    print(f"causal vs non-causal maxdiff = {d_masks:.4f} "
          f"({'OK' if d_masks > 1e-2 else 'SUSPICIOUS: identical outputs'})")
    all_ok = ok_c and ok_n and d_masks > 1e-2
    print("RESULT:", "PASS" if all_ok else "FAIL")


if __name__ == "__main__":
    main()
