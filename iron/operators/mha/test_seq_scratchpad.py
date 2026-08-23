#!/usr/bin/env python3
"""M1 feasibility gate: wrap MHA in a single-step OperatorSequence and drive
S_kv_eff / S_q_eff via fused.params (ParameterScratchpad).

Verifies:
1. MHA-as-sequence compiles and produces params.txt.
2. fused.params is not None and can write S_q_eff / S_kv_eff.
3. seq_len=256/512 output matches CPU reference (cosine > 0.99), no NaN.
"""
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


def cpu_reference(q, K, V, seq_pos, head_dim=128, GQA=2):
    scale = 1.0 / head_dim ** 0.5
    H = q.shape[0]
    out = torch.zeros(H, head_dim)
    for h in range(H):
        kv_h = h // GQA
        k = K[kv_h, :seq_pos, :].float()
        v = V[kv_h, :seq_pos, :].float()
        s = q[h].float() @ k.t() * scale
        out[h] = torch.softmax(s, dim=0) @ v
    return out


def main():
    fix = torch.load("/home/zyc/Github/triton-xdna/examples/qwen3_0.6b/tests_mha/mha_fixture.pt")
    q = fix["q"]; K = fix["K"]; V = fix["V"]; seq_pos = fix["seq_pos"]
    H, d = q.shape
    KV_H, S_kv, _ = K.shape
    GQA = H // KV_H

    ref = cpu_reference(q, K, V, seq_pos)

    for seq_len in [256, 512]:
        sp = seq_len
        mha = MHA(num_heads=H, seq_len=seq_len, d=d, num_KV_heads=KV_H,
                  num_of_pipelines=1, seq_len_kv=S_kv)
        seq = OperatorSequence(
            name=f"mha_seq_{seq_len}",
            runlist=[(mha, "q", "k", "v", "o")],
            input_args=["q", "k", "v"],
            output_args=["o"],
            dispatch="fused",
        )
        seq.compile()
        fc = seq.get_callable()

        # Write runtime params: decode uses S_q_eff=1?  No — for seq_len=256/512
        # the query is at seq_pos; valid query rows = S_q_eff. For decode-style
        # single query it's 1 valid row, but the kernel computes per Q block.
        # Here we emulate full-seq attention with causal mask over seq_pos.
        assert fc.params is not None, "params.txt not generated!"
        # S_q_eff = seq_pos+1: query occupies rows [0, seq_pos], only row
        # seq_pos is real (others -1e8).  S_kv_eff = seq_pos: KV rows
        # [seq_pos, S_kv) are -1e8 padding (current K not yet written).
        fc.params.write("S_q_eff", seq_pos + 1)
        fc.params.write("S_kv_eff", seq_pos)
        fc.params.sync()

        # Q buffer: (H, sp, d), query at seq_pos, rest pad
        q_buf = torch.full((H, sp, d), -1e8, dtype=torch.bfloat16)
        q_buf[:, seq_pos, :] = q.bfloat16()
        fc.get_buffer("q").torch_view()[:] = q_buf.flatten()
        fc.get_buffer("q").to("npu")

        # K/V buffers in head-major (KV_H, S_kv*d)
        fc.get_buffer("k").torch_view()[:] = K.flatten().to(torch.bfloat16)
        fc.get_buffer("v").torch_view()[:] = V.flatten().to(torch.bfloat16)
        fc.get_buffer("k").to("npu")
        fc.get_buffer("v").to("npu")

        fc()

        fc.get_buffer("o").to("cpu")
        o = fc.get_buffer("o").torch_view().reshape(H, sp, d)[:, seq_pos, :].float()
        n_nan = torch.isnan(o).sum().item()
        cos = torch.nn.functional.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item()
        hd = [(o[h] - ref[h]).abs().max().item() for h in range(H)]
        print(f"seq_len={seq_len}: nan={n_nan}, cosine={cos:.6f}")
        print(f"  per-head maxdiff: {[f'{x:.2f}' for x in hd]}")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
