#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-0.6B decode on NPU — whole-model single OperatorSequence (pure iron).

Faithful port of llama_npu.py's whole-model single-sequence structure, with the
only Qwen3-specific addition being QK-norm (q_norm / k_norm per-head RMSNorm).
RoPE uses the standard iron RoPE operator with a per-token variable angle buffer
(rope_angles), exactly like llama.

Model (Qwen3-0.6B):
  emb_dim=1024, hidden_dim=3072, n_heads=16, n_kv_heads=8 (GQA=2),
  head_dim=128, vocab=151936, n_layers=28, rope_theta=1e6, eps=1e-6.

Decode path (single token, S=1), one layer:
  RMSNorm(x) -> QKV GEMV -> QK-norm(Q,K) -> RoPE(Q,K) -> StridedCopy(K,V into cache)
  -> Repeat (GQA broadcast) -> GEMV scores -> scale -> Softmax -> per-head Transpose(V)
  -> GEMV context -> O GEMV -> residual add -> RMSNorm -> SwiGLU MLP -> residual add.
After 28 layers: final RMSNorm -> GEMV(lm_head, f32) -> logits.
"""

import torch
import math
import numpy as np
import ml_dtypes
from pathlib import Path
from safetensors import safe_open

from aie.iron.device import NPU2
import aie.utils as aie_utils
aie_utils.set_current_device(NPU2())

from iron.common.context import AIEContext
from iron.common.sequence import OperatorSequence
from iron.operators import (
    RMSNorm, GEMV, StridedCopy, DecodeAttention,
    ElementwiseMul, ElementwiseAdd, SiLU, RoPE, QKNorm,
)

# ---------------- model config ----------------
emb_dim = 1024
hidden_dim = 3072
n_heads = 16
n_kv_heads = 8
head_dim = 128
q_dim = n_heads * head_dim        # 2048
kv_dim = n_kv_heads * head_dim    # 1024
qkv_dim = q_dim + 2 * kv_dim      # 4096
vocab_size = 151936
n_layers = 28
max_seq_len = 2048
block_kv = 64                     # decode_attn KV block size along S
eps = 1e-6
rope_theta = 1e6

# byte offsets within qkv_out (bf16, 2 bytes/elem) and qk_normed
q_end = q_dim * 2                    # 4096
k_end = (q_dim + kv_dim) * 2         # 6144
v_start = k_end
v_end = qkv_dim * 2                  # 8192
vps = max_seq_len * head_dim * 2     # per-head values size in bytes


class Qwen3NPU:
    def __init__(self, weights_path):
        self.context = AIEContext()
        self.context.build_dir.mkdir(parents=True, exist_ok=True)
        self.weights = safe_open(weights_path, framework="pt")

        # ---- weights ----
        self.weight_cache = {}
        for i in range(n_layers):
            L = f"model.layers.{i}"
            F = self.weights.get_tensor
            self.weight_cache[i] = {
                "input_norm": F(f"{L}.input_layernorm.weight").to(torch.bfloat16),
                "q": F(f"{L}.self_attn.q_proj.weight").to(torch.bfloat16),
                "k": F(f"{L}.self_attn.k_proj.weight").to(torch.bfloat16),
                "v": F(f"{L}.self_attn.v_proj.weight").to(torch.bfloat16),
                "o": F(f"{L}.self_attn.o_proj.weight").to(torch.bfloat16),
                "q_norm": F(f"{L}.self_attn.q_norm.weight").to(torch.bfloat16),
                "k_norm": F(f"{L}.self_attn.k_norm.weight").to(torch.bfloat16),
                "norm2": F(f"{L}.post_attention_layernorm.weight").to(torch.bfloat16),
                "gate": F(f"{L}.mlp.gate_proj.weight").to(torch.bfloat16),
                "up": F(f"{L}.mlp.up_proj.weight").to(torch.bfloat16),
                "down": F(f"{L}.mlp.down_proj.weight").to(torch.bfloat16),
            }
        self.final_norm = self.weights.get_tensor("model.norm.weight").to(torch.bfloat16)
        # lm_head shares embed_tokens (tied embeddings); bf16 weights, f32 acc argmax
        self.out_head = self.weights.get_tensor("model.embed_tokens.weight").to(torch.bfloat16)

        self._build_operators()
        self._build_sequence()
        self._load_weights()

    # ---------------- operators ----------------
    def _build_operators(self):
        c = self.context
        self.rms = RMSNorm(size=emb_dim, num_aie_columns=1, num_channels=1,
                           tile_size=emb_dim, weighted=True, epsilon=eps, context=c)
        self.gemv_qkv = GEMV(M=qkv_dim, K=emb_dim, num_aie_columns=8,
                             tile_size_input=4, tile_size_output=qkv_dim // 8, context=c)
        self.qk_norm = QKNorm(head_dim=head_dim, n_q_heads=n_heads, n_k_heads=n_kv_heads,
                              epsilon=eps, context=c)
        self.rope_q = RoPE(rows=n_heads, cols=head_dim, angle_rows=1, context=c)
        self.rope_k = RoPE(rows=n_kv_heads, cols=head_dim, angle_rows=1, context=c)
        # K/V cache now lives in a single INTERLEAVED buffer per layer:
        #   [K_b0 | V_b0 | K_b1 | V_b1 | ...]  (block_kv tokens per block),
        # matching decode_attn's KV input layout. Each head occupies a full
        # 2*max_seq_len*head_dim element block; K goes to the front half of each
        # block, V to the back half. The per-token write position is a runtime
        # offset parameter computed on the host (see __call__).
        self.sc_k = StridedCopy(input_sizes=(n_kv_heads, head_dim), input_strides=(head_dim, 1),
                                input_offset=0, output_sizes=(n_kv_heads, head_dim),
                                output_strides=(2 * max_seq_len * head_dim, 1), output_offset=0,
                                input_buffer_size=kv_dim,
                                output_buffer_size=n_kv_heads * 2 * max_seq_len * head_dim,
                                num_aie_channels=1, output_offset_parameter="k_cache_offset", context=c)
        self.sc_v = StridedCopy(input_sizes=(n_kv_heads, head_dim), input_strides=(head_dim, 1),
                                input_offset=0, output_sizes=(n_kv_heads, head_dim),
                                output_strides=(2 * max_seq_len * head_dim, 1), output_offset=0,
                                input_buffer_size=kv_dim,
                                output_buffer_size=n_kv_heads * 2 * max_seq_len * head_dim,
                                num_aie_channels=1, output_offset_parameter="v_cache_offset", context=c)
        # Fused decode attention (single query, M=1): scores-GEMV + softmax +
        # context-GEMV in one kernel per (head, column). Replaces the prefill-style
        # gemv_scores + attn_scale + softmax + transpose_v + gemv_context chain,
        # and subsumes the GQA Repeat (one KV head per column).
        self.decode_attn = DecodeAttention(
            num_heads=n_heads, num_kv_heads=n_kv_heads, head_dim=head_dim,
            seq_len_kv=max_seq_len, num_aie_columns=8, block_kv=block_kv,
            use_runtime_seq_len=True, context=c)
        self.gemv_output = GEMV(M=emb_dim, K=q_dim, num_aie_columns=8,
                                tile_size_input=4, tile_size_output=emb_dim // 8, context=c)
        self.residual_add = ElementwiseAdd(size=emb_dim, tile_size=emb_dim // 8, context=c)
        self.gemv_ffn = GEMV(M=hidden_dim, K=emb_dim, num_aie_columns=8,
                             tile_size_input=4, tile_size_output=hidden_dim // 8, context=c)
        self.silu = SiLU(size=hidden_dim, tile_size=hidden_dim // 8, num_aie_columns=8, context=c)
        self.mul = ElementwiseMul(size=hidden_dim, tile_size=hidden_dim // 8,
                                  num_aie_columns=8, context=c)
        self.gemv_ffn_down = GEMV(M=emb_dim, K=hidden_dim, num_aie_columns=8,
                                  tile_size_input=1, tile_size_output=emb_dim // 8, context=c)
        # lm_head: bf16 GEMV computing logits (f32 accumulate, bf16 store), fused
        # INTO the sequence exactly like llama — the final RMSNorm feeds a plain
        # GEMV(W_out_head) that emits "logits" as the sequence output. argmax runs
        # on the host over the bf16 logits (llama does the same; bf16 logits have
        # ~0.4% rel error, far below any margin that would flip the argmax).
        # tile_size_output must divide M/num_aie_columns = 151936/8 = 18992.
        # 18992 = 16 * 1187 (1187 prime), so the only viable m_input=4-compatible
        # tile outputs are 4, 8, 16. llama's 32 works only because its vocab
        # 128256/8=16032 is 32-divisible; Qwen3 vocab is not. 16 is the largest
        # valid tile and keeps L1 pressure low.
        self.lmhead = GEMV(M=vocab_size, K=emb_dim, num_aie_columns=8,
                           tile_size_input=4, tile_size_output=16,
                           context=c)

    # ---------------- sequence ----------------
    def _build_sequence(self):
        runlist = []
        for i in range(n_layers):
            runlist.extend([
                (self.rms, "x", f"W_norm1_{i}", "x_norm"),
                (self.gemv_qkv, f"W_qkv_{i}", "x_norm", "qkv_out"),
                (self.qk_norm, f"qkv_out[0:{k_end}]", "qk_normed", f"W_qk_gamma_{i}"),
                (self.rope_q, f"qk_normed[0:{q_end}]", "rope_angles", "rope_q"),
                (self.rope_k, f"qk_normed[{q_end}:{k_end}]", "rope_angles", "rope_k"),
                (self.sc_k, "rope_k", f"kv_cache_{i}"),
                (self.sc_v, f"qkv_out[{v_start}:{v_end}]", f"kv_cache_{i}"),
                (self.decode_attn, "rope_q", f"kv_cache_{i}", "context"),
                (self.gemv_output, f"W_o_{i}", "context", "attn_output"),
                (self.residual_add, "x", "attn_output", "x"),
                (self.rms, "x", f"W_norm2_{i}", "x_norm"),
                (self.gemv_ffn, f"W_gate_{i}", "x_norm", "ffn_gate"),
                (self.gemv_ffn, f"W_up_{i}", "x_norm", "ffn_up"),
                (self.silu, "ffn_gate", "ffn_gate"),
                (self.mul, "ffn_gate", "ffn_up", "ffn_hidden"),
                (self.gemv_ffn_down, f"W_down_{i}", "ffn_hidden", "ffn_output"),
                (self.residual_add, "x", "ffn_output", "x"),
            ])
        # Final RMSNorm + lm_head GEMV both live in the SAME sequence (llama-style),
        # so decode is a single dispatch producing "logits". The final RMSNorm emits
        # an independent buffer "x_final" (avoids in-out x colliding in
        # subbuffer_layout), which the lm_head GEMV reads.
        runlist += [
            (self.rms, "x", "W_final_norm", "x_final"),
            (self.lmhead, "W_out_head", "x_final", "logits"),
        ]

        # interleaved KV cache: K and V share one buffer per layer
        # ([K_b0|V_b0|K_b1|V_b1|...]), sized 2x the old separated k/v cache.
        kv_cache_size = n_kv_heads * 2 * max_seq_len * head_dim * 2
        self.seq = OperatorSequence(
            "qwen3_0.6b_decode",
            runlist,
            input_args=["x", "rope_angles"],
            output_args=["logits"],
            buffer_sizes={
                **{f"kv_cache_{i}": kv_cache_size for i in range(n_layers)},
                "rope_angles": head_dim * 2,
                "logits": vocab_size * 2,
            },
            context=self.context,
        )
        self.seq.compile()
        self.fc = self.seq.get_callable()
        print(f"compiled OK; buffer_sizes: {self.seq.buffer_sizes}")
        print(f"n buffers: {len(self.seq.subbuffer_layout)}")

    # ---------------- weight load ----------------
    def _load_weights(self):
        for i in range(n_layers):
            w = self.weight_cache[i]
            self._set(f"W_norm1_{i}", w["input_norm"])
            self._set(f"W_qkv_{i}", torch.cat([w["q"], w["k"], w["v"]], dim=0))
            self._set(f"W_qk_gamma_{i}", torch.cat([w["q_norm"], w["k_norm"]], dim=0))
            self._set(f"W_o_{i}", w["o"])
            self._set(f"W_norm2_{i}", w["norm2"])
            self._set(f"W_gate_{i}", w["gate"])
            self._set(f"W_up_{i}", w["up"])
            self._set(f"W_down_{i}", w["down"])
        self._set("W_final_norm", self.final_norm)
        self._set("W_out_head", self.out_head)

        # RoPE angle table (interleaved cos/sin, half dim)
        self._precompute_rope_angles()

        # sync static buffers to device
        self.fc.input_buffer.to("npu")
        self.fc.scratch_buffer.to("npu")

    def _set(self, name, t):
        b = self.fc.get_buffer(name).torch_view()
        b[:] = t.flatten() if t.dim() > 1 else t

    def _precompute_rope_angles(self):
        half = head_dim // 2
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)              # (max_seq_len, half)
        emb = torch.cat([freqs, freqs], dim=-1)       # (max_seq_len, head_dim)
        cos = emb.cos().to(torch.bfloat16)            # (max_seq_len, head_dim)
        sin = emb.sin().to(torch.bfloat16)
        cos_half = cos[:, :half]
        sin_half = sin[:, :half]
        # interleaved [cos0, sin0, cos1, sin1, ...] -> (max_seq_len, head_dim)
        self.angle_table = torch.stack([cos_half, sin_half], dim=-1).reshape(max_seq_len, head_dim)

    # ---------------- forward (single token) ----------------
    def __call__(self, x, seq_pos):
        """x: (emb_dim,) bf16 hidden state for the new token; seq_pos: position index."""
        # interleaved KV cache write positions for this token (element offsets,
        # one head occupies a 2*max_seq_len*head_dim block with [K_b|V_b] layout).
        b = seq_pos // block_kv
        j = seq_pos % block_kv
        blk = 2 * block_kv * head_dim
        k_off = b * blk + j * head_dim                 # K[b][j]
        v_off = b * blk + (block_kv + j) * head_dim    # V[b][j]

        # update per-token rope_angles
        self.fc.get_buffer("rope_angles").torch_view()[:] = self.angle_table[seq_pos]
        # write x
        self.fc.get_buffer("x").torch_view()[:] = x

        self.fc.params.write("k_cache_offset", np.int32(k_off))
        self.fc.params.write("v_cache_offset", np.int32(v_off))
        self.fc.params.write("S_kv_eff", np.int32(seq_pos + 1))
        self.fc.params.sync()

        self.fc()  # runs 28 layers + final RMSNorm + lm_head GEMV -> logits

        # logits already synced to CPU by SequenceCallable.__call__ (_sync_outputs)
        logits = self.fc.get_buffer("logits").torch_view()
        return torch.argmax(logits).item()


if __name__ == "__main__":
    # Smoke test: load weights, run one decode step, report argmax + no NaN.
    model = Qwen3NPU("/home/zyc/Packages/NPU/Qwen3-0.6B/model.safetensors")
    x = torch.randn(emb_dim, dtype=torch.bfloat16)
    token_id = model(x, seq_pos=10)
    print(f"argmax token_id = {token_id}")
