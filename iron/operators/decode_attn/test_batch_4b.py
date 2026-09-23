#!/usr/bin/env python3
"""4B-dims batch DecodeAttention numeric test (test_batch.py style).
H=32, KV=8, d=128, S_max=512, block_kv=32, M=8x2 groups, base=480."""
import sys, math
sys.path.insert(0, "/home/zyc/Github")
import torch

from aie.iron.device import NPU2
import aie.utils as aie_utils
aie_utils.set_current_device(NPU2())

from iron.common.context import AIEContext
from iron.common.sequence import OperatorSequence
from iron.operators.decode_attn.op import DecodeAttention

import os
H = int(os.environ.get('T_H', '32'))
KV_H, D = 8, 128
S_MAX = int(os.environ.get('T_SMAX', '512'))
BLOCK, M, NEW = 32, 8, 16
BASE = int(os.environ.get('T_BASE', '480'))


def ref(Q, K, V, base, causal, GQA=4):
    scale = 1.0 / math.sqrt(D)
    out = torch.zeros(NEW, H, D)
    for m in range(NEW):
        nk = S_MAX if not causal else base + m + 1
        for h in range(H):
            s = Q[m, h].float() @ K[h // GQA, :nk, :].float().t() * scale
            out[m, h] = torch.softmax(s, dim=0) @ V[h // GQA, :nk, :].float()
    return out


def run(causal):
    tag = "causal" if causal else "noncausal"
    ctx = AIEContext(build_dir=f"/tmp/bd_attn4b_{tag}")
    op = DecodeAttention(num_heads=H, num_kv_heads=KV_H, head_dim=D,
                         seq_len_kv=S_MAX, num_aie_columns=8, block_kv=BLOCK,
                         use_runtime_seq_len=True, batch_queries=M,
                         num_new_tokens=NEW, causal=causal, context=ctx)
    seq = OperatorSequence(f"attn4b_{tag}", [(op, "q", "kv", "o")],
                           input_args=["q", "kv"], output_args=["o"],
                           context=ctx)
    seq.compile()
    fc = seq.get_callable()
    torch.manual_seed(3)
    Q = torch.randn(NEW, H, D).bfloat16() * 0.5
    K = torch.randn(KV_H, S_MAX, D).bfloat16() * 0.3
    V = torch.randn(KV_H, S_MAX, D).bfloat16() * 0.3
    # interleaved [K_b0|V_b0|...]
    NB = S_MAX // BLOCK
    blocks = []
    for b in range(NB):
        blocks += [K[:, b*BLOCK:(b+1)*BLOCK, :].reshape(KV_H, BLOCK*D),
                   V[:, b*BLOCK:(b+1)*BLOCK, :].reshape(KV_H, BLOCK*D)]
    kv = torch.cat(blocks, dim=1)
    fc.get_buffer("q").torch_view()[:] = Q.flatten()
    fc.get_buffer("kv").torch_view()[:] = kv.flatten()
    fc.get_buffer("q").to("npu"); fc.get_buffer("kv").to("npu")
    fc.params.write("S_kv_base", BASE)
    fc.params.sync()
    fc()
    fc.get_buffer("o").to("cpu")
    o = fc.get_buffer("o").torch_view().reshape(NEW, H, D).float()
    r = ref(Q, K, V, BASE, causal)
    n_nan = torch.isnan(o).sum().item()
    cos = torch.nn.functional.cosine_similarity(
        o.flatten(), r.flatten(), dim=0).item()
    print(f"[{tag}] nan={n_nan} cos={cos:.6f} "
          f"{'OK' if n_nan == 0 and cos > 0.99 else 'FAIL'}", flush=True)
    # where did each token's data land? best-match ref token per output token
    for m in [0, 1, 8, 15]:
        sims = [torch.nn.functional.cosine_similarity(
            o[m].flatten(), r[mm].flatten(), dim=0).item() for mm in range(NEW)]
        best = int(torch.tensor(sims).argmax().item())
        print(f"  out[{m:2d}] best-match ref[{best:2d}] cos={sims[best]:.3f} "
              f"outnorm={o[m].norm().item():.3f} refnorm={r[m].norm().item():.3f}", flush=True)
    for m in [0, 1, 8, 15]:
        c = torch.nn.functional.cosine_similarity(
            o[m].flatten(), r[m].flatten(), dim=0).item()
        heads = [torch.nn.functional.cosine_similarity(
            o[m, h], r[m, h], dim=0).item() for h in range(0, H, 4)]
        print(f"  tok {m:2d}: cos={c:.4f} heads[0::4]="
              + " ".join(f"{h:.2f}" for h in heads), flush=True)
    return o


if __name__ == "__main__":
    run(True)
    run(False)
