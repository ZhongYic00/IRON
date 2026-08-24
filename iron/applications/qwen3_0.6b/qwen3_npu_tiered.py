#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-0.6B decode on NPU — tiered sequence-length dispatch (pure iron).

Same whole-model single-OperatorSequence structure as qwen3_npu.py, but the
attention compute width (gemv_scores M / softmax cols / transpose M /
gemv_context K) is TIERED: a 256-length tier for short sequences (prompt +
first ~256 tokens) and a 512-length tier for the tail. Short decode avoids
scoring against a full 512-length KV cache, saving ~9.4ms/token.

The K/V cache lives in per-tier scratch buffers (256 vs 512 wide), so on the
single tier switch (seq_pos reaching 256) the K/V is migrated from the 256 tier
into the wider 512 tier (shape (8,256,128) -> (8,512,128) stride repack,
~28MB one-time copy).

Model (Qwen3-0.6B): emb=1024, hidden=3072, n_head=16, n_kv=8 (GQA=2),
head_dim=128, vocab=151936, n_layer=28, rope_theta=1e6, eps=1e-6.
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
    RMSNorm, GEMV, StridedCopy, Repeat, Softmax,
    ElementwiseMul, ElementwiseAdd, SiLU, Transpose, RoPE, QKNorm,
)

# ---------------- model config ----------------
n_heads = 16
n_kv_heads = 8
head_dim = 128
q_dim = n_heads * head_dim        # 2048
kv_dim = n_kv_heads * head_dim    # 1024
qkv_dim = q_dim + 2 * kv_dim      # 4096
emb_dim = 1024
hidden_dim = 3072
vocab_size = 151936
n_layers = 28
eps = 1e-6
rope_theta = 1e6

# Tiered attention widths, in increasing order (each must be a multiple of
# Transpose's m=256, and cover the full max_seq_len at the top tier).
MAX_SEQ_LEN = 512
ATTN_LENS = (256, 512)

# byte offsets within qkv_out (bf16, 2 bytes/elem) and qk_normed
q_end = q_dim * 2                    # 4096
k_end = (q_dim + kv_dim) * 2         # 6144
v_start = k_end
v_end = qkv_dim * 2                  # 8192


class Qwen3NPUTiered:
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
        self.out_head = self.weights.get_tensor("model.embed_tokens.weight").to(torch.bfloat16)

        self._build_operators()
        self._build_sequences()
        self._load_weights()
        self._cur_tier = 0  # index into ATTN_LENS

    # ---------------- operators (shared across tiers) ----------------
    def _build_operators(self):
        c = self.context
        # shared: emb->hidden / hidden->emb / norm / output / mlp / lm_head
        self.rms = RMSNorm(size=emb_dim, num_aie_columns=1, num_channels=1,
                           tile_size=emb_dim, weighted=True, epsilon=eps, context=c)
        self.gemv_qkv = GEMV(M=qkv_dim, K=emb_dim, num_aie_columns=8,
                             tile_size_input=4, tile_size_output=qkv_dim // 8, context=c)
        self.qk_norm = QKNorm(head_dim=head_dim, n_q_heads=n_heads, n_k_heads=n_kv_heads,
                              epsilon=eps, context=c)
        self.rope_q = RoPE(rows=n_heads, cols=head_dim, angle_rows=1, context=c)
        self.rope_k = RoPE(rows=n_kv_heads, cols=head_dim, angle_rows=1, context=c)
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
        # lm_head: bf16 GEMV + host argmax, fused into the sequence (llama-style).
        self.lmhead = GEMV(M=vocab_size, K=emb_dim, num_aie_columns=8,
                           tile_size_input=4, tile_size_output=16, context=c)

        # per-tier attention operators
        self.attn = {}
        for L in ATTN_LENS:
            vps = L * head_dim * 2  # per-head values size in bytes (this tier)
            self.attn[L] = {
                "sc_k": StridedCopy(
                    input_sizes=(n_kv_heads, head_dim), input_strides=(head_dim, 1),
                    input_offset=0, output_sizes=(n_kv_heads, head_dim),
                    output_strides=(L * head_dim, 1), output_offset=0,
                    input_buffer_size=kv_dim,
                    output_buffer_size=n_kv_heads * L * head_dim,
                    num_aie_channels=1, output_offset_parameter="cache_offset", context=c),
                "sc_v": StridedCopy(
                    input_sizes=(n_kv_heads, head_dim), input_strides=(head_dim, 1),
                    input_offset=0, output_sizes=(n_kv_heads, head_dim),
                    output_strides=(L * head_dim, 1), output_offset=0,
                    input_buffer_size=kv_dim,
                    output_buffer_size=n_kv_heads * L * head_dim,
                    num_aie_channels=1, output_offset_parameter="cache_offset", context=c),
                "gemv_scores": GEMV(M=L, K=head_dim, num_aie_columns=8,
                                    tile_size_input=4, tile_size_output=L // 8,
                                    num_batches=n_heads, context=c),
                "attn_scale": ElementwiseMul(size=n_heads * L, tile_size=L // 8,
                                             num_aie_columns=8, context=c),
                "softmax": Softmax(rows=n_heads, cols=L, num_aie_columns=1, num_channels=1,
                                   rtp_vector_size=L,
                                   vector_size_parameter="softmax_vector_size", context=c),
                "transpose_v": Transpose(M=L, N=head_dim, num_aie_columns=2, num_channels=1,
                                         m=256, n=32, s=8, context=c),
                "gemv_context": GEMV(M=head_dim, K=L, num_aie_columns=8,
                                     tile_size_input=4, tile_size_output=4,
                                     num_batches=n_heads, context=c),
                "repeat": Repeat(rows=n_kv_heads, cols=L * head_dim,
                                 repeat=n_heads // n_kv_heads, transfer_size=head_dim, context=c),
                "vps": vps,
            }

    # ---------------- sequences ----------------
    def _build_sequences(self):
        self.seqs = {}
        self.fcs = {}
        for ti, L in enumerate(ATTN_LENS):
            a = self.attn[L]
            vps = a["vps"]
            runlist = []
            for i in range(n_layers):
                runlist.extend([
                    (self.rms, "x", f"W_norm1_{i}", "x_norm"),
                    (self.gemv_qkv, f"W_qkv_{i}", "x_norm", "qkv_out"),
                    (self.qk_norm, f"qkv_out[0:{k_end}]", "qk_normed", f"W_qk_gamma_{i}"),
                    (self.rope_q, f"qk_normed[0:{q_end}]", "rope_angles", "rope_q"),
                    (self.rope_k, f"qk_normed[{q_end}:{k_end}]", "rope_angles", "rope_k"),
                    (a["sc_k"], "rope_k", f"k_cache_{i}"),
                    (a["sc_v"], f"qkv_out[{v_start}:{v_end}]", f"v_cache_{i}"),
                    (a["repeat"], f"k_cache_{i}", "scores_keys"),
                    (a["repeat"], f"v_cache_{i}", "scores_values"),
                    (a["gemv_scores"], "scores_keys", "rope_q", "scores"),
                    (a["attn_scale"], "scores", "attn_scale", "scores"),
                    (a["softmax"], "scores", "weights"),
                ] + [
                    (a["transpose_v"],
                     f"scores_values[{h * vps}:{(h + 1) * vps}]",
                     f"scores_values_T[{h * vps}:{(h + 1) * vps}]")
                    for h in range(n_heads)
                ] + [
                    (a["gemv_context"], "scores_values_T", "weights", "context"),
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
            runlist += [
                (self.rms, "x", "W_final_norm", "x_final"),
                (self.lmhead, "W_out_head", "x_final", "logits"),
            ]

            cache_size = n_kv_heads * L * head_dim * 2
            seq = OperatorSequence(
                f"qwen3_0.6b_decode_L{L}",
                runlist,
                input_args=["x", "rope_angles"],
                output_args=["logits"],
                buffer_sizes={
                    **{f"k_cache_{i}": cache_size for i in range(n_layers)},
                    **{f"v_cache_{i}": cache_size for i in range(n_layers)},
                    "scores_values": n_heads * L * head_dim * 2,
                    "scores_values_T": n_heads * L * head_dim * 2,
                    "rope_angles": head_dim * 2,
                    "logits": vocab_size * 2,
                },
                context=self.context,
            )
            seq.compile()
            self.seqs[L] = seq
            self.fcs[L] = seq.get_callable()
            print(f"tier L={L}: compiled OK; n buffers = {len(seq.subbuffer_layout)}")

    # ---------------- weight load ----------------
    def _load_weights(self):
        scale = 1.0 / math.sqrt(head_dim)
        # weights are identical across tiers; write them into the highest tier's
        # fc (buffers are per-fc, so write into every tier).
        for L in ATTN_LENS:
            fc = self.fcs[L]
            for i in range(n_layers):
                w = self.weight_cache[i]
                self._set(fc, f"W_norm1_{i}", w["input_norm"])
                self._set(fc, f"W_qkv_{i}", torch.cat([w["q"], w["k"], w["v"]], dim=0))
                self._set(fc, f"W_qk_gamma_{i}", torch.cat([w["q_norm"], w["k_norm"]], dim=0))
                self._set(fc, f"W_o_{i}", w["o"])
                self._set(fc, f"W_norm2_{i}", w["norm2"])
                self._set(fc, f"W_gate_{i}", w["gate"])
                self._set(fc, f"W_up_{i}", w["up"])
                self._set(fc, f"W_down_{i}", w["down"])
            self._set(fc, "W_final_norm", self.final_norm)
            self._set(fc, "W_out_head", self.out_head)
            attn_scale_buf = fc.get_buffer("attn_scale").torch_view()
            attn_scale_buf[:] = torch.full(attn_scale_buf.shape, scale, dtype=torch.bfloat16)
            fc.input_buffer.to("npu")
            fc.scratch_buffer.to("npu")

        self._precompute_rope_angles()

    def _set(self, fc, name, t):
        b = fc.get_buffer(name).torch_view()
        b[:] = t.flatten() if t.dim() > 1 else t

    def _precompute_rope_angles(self):
        half = head_dim // 2
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        t = torch.arange(MAX_SEQ_LEN, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        cos = emb.cos().to(torch.bfloat16)
        sin = emb.sin().to(torch.bfloat16)
        cos_half = cos[:, :half]
        sin_half = sin[:, :half]
        self.angle_table = torch.stack([cos_half, sin_half], dim=-1).reshape(MAX_SEQ_LEN, head_dim)

    # ---------------- tier management ----------------
    def _tier_for(self, seq_pos):
        """Return the ATTN_LENS index and value covering seq_pos (the KV half-open
        length is seq_pos+1; the tier's width must be >= that)."""
        half_open = seq_pos + 1
        for ti, L in enumerate(ATTN_LENS):
            if half_open <= L:
                return ti, L
        return len(ATTN_LENS) - 1, ATTN_LENS[-1]

    def _migrate_kv(self, from_L, to_L):
        """Copy K/V from the from_L tier into the to_L tier (shape
        (n_kv_heads, from_L, head_dim) -> (n_kv_heads, to_L, head_dim) via a
        stride repack). Per-layer, per-K/V.

        The from-tier's KV lives on-device (updated by each fc() dispatch),
        so we must first sync it back to host before reading; otherwise we'd
        copy stale init-zeros. The to-tier's KV is then written on host and
        pushed back to device.
        """
        fc_from = self.fcs[from_L]
        fc_to = self.fcs[to_L]
        # Pull the from-tier's whole scratch (incl. K/V) device -> host so the
        # host-side torch_view() reflects the latest KV.
        fc_from.scratch_buffer.device = "npu"
        fc_from.scratch_buffer.to("cpu")
        for i in range(n_layers):
            for kind in ("k_cache", "v_cache"):
                src = fc_from.get_buffer(f"{kind}_{i}").torch_view()  # (n_kv, from_L*head)
                dst = fc_to.get_buffer(f"{kind}_{i}").torch_view()    # (n_kv, to_L*head)
                s = src.reshape(n_kv_heads, from_L, head_dim)
                d = dst.reshape(n_kv_heads, to_L, head_dim)
                d[:, :from_L, :] = s
        fc_to.scratch_buffer.to("npu")  # push migrated KV back to device for to-tier

    # ---------------- forward (single token) ----------------
    def __call__(self, x, seq_pos):
        ti, L = self._tier_for(seq_pos)
        if ti > self._cur_tier:
            # switch upward: migrate KV from the current tier to the new one
            self._migrate_kv(ATTN_LENS[self._cur_tier], L)
            self._cur_tier = ti

        fc = self.fcs[L]
        cache_offset = seq_pos * head_dim

        fc.get_buffer("rope_angles").torch_view()[:] = self.angle_table[seq_pos]
        fc.get_buffer("x").torch_view()[:] = x
        fc.params.write("cache_offset", np.int32(cache_offset))
        fc.params.write("softmax_vector_size", np.int32(seq_pos + 1))
        fc.params.sync()

        fc()
        logits = fc.get_buffer("logits").torch_view()
        return torch.argmax(logits).item()


if __name__ == "__main__":
    model = Qwen3NPUTiered("/home/zyc/Packages/NPU/Qwen3-0.6B/model.safetensors")
    x = torch.randn(emb_dim, dtype=torch.bfloat16)
    print(f"argmax@10 = {model(x, seq_pos=10)}")
    print(f"argmax@300 = {model(x, seq_pos=300)}")
