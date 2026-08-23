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
    RMSNorm, GEMV, StridedCopy, Repeat, Softmax,
    ElementwiseMul, ElementwiseAdd, SiLU, Transpose, RoPE, QKNorm,
)
from iron.operators.gemv_argmax.op import GEMVArgmax
from iron.operators.gemv_argmax_bf16.op import GEMVArgmaxBF16
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
import pyxrt as xrt

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
max_seq_len = 512
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
        self.sc_k = StridedCopy(input_sizes=(n_kv_heads, head_dim), input_strides=(head_dim, 1),
                                input_offset=0, output_sizes=(n_kv_heads, head_dim),
                                output_strides=(max_seq_len * head_dim, 1), output_offset=0,
                                input_buffer_size=kv_dim,
                                output_buffer_size=n_kv_heads * max_seq_len * head_dim,
                                num_aie_channels=1, output_offset_parameter="cache_offset", context=c)
        self.sc_v = StridedCopy(input_sizes=(n_kv_heads, head_dim), input_strides=(head_dim, 1),
                                input_offset=0, output_sizes=(n_kv_heads, head_dim),
                                output_strides=(max_seq_len * head_dim, 1), output_offset=0,
                                input_buffer_size=kv_dim,
                                output_buffer_size=n_kv_heads * max_seq_len * head_dim,
                                num_aie_channels=1, output_offset_parameter="cache_offset", context=c)
        self.gemv_scores = GEMV(M=max_seq_len, K=head_dim, num_aie_columns=8,
                                tile_size_input=4, tile_size_output=max_seq_len // 8,
                                num_batches=n_heads, context=c)
        self.attn_scale = ElementwiseMul(size=n_heads * max_seq_len,
                                         tile_size=max_seq_len // 8, num_aie_columns=8, context=c)
        self.softmax = Softmax(rows=n_heads, cols=max_seq_len, num_aie_columns=1,
                               num_channels=1, rtp_vector_size=max_seq_len,
                               vector_size_parameter="softmax_vector_size", context=c)
        self.transpose_v = Transpose(M=max_seq_len, N=head_dim, num_aie_columns=2,
                                     num_channels=1, m=256, n=32, s=8, context=c)
        self.gemv_context = GEMV(M=head_dim, K=max_seq_len, num_aie_columns=8,
                                 tile_size_input=4, tile_size_output=4,
                                 num_batches=n_heads, context=c)
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
        self.repeat = Repeat(rows=n_kv_heads, cols=max_seq_len * head_dim,
                             repeat=n_heads // n_kv_heads, transfer_size=head_dim, context=c)
        # lm_head: bf16 GEMV + f32 argmax epilogue, INDEPENDENT dispatch. bf16
        # weights halve the DDR read (622MB f32 -> 311MB bf16) vs GEMVArgmax;
        # f32 accumulation keeps argmax precision. Verified against model.py.
        self.lmhead_op = GEMVArgmaxBF16(M=vocab_size, K=emb_dim, num_aie_columns=8,
                                        tile_size_input=4, tile_size_output=vocab_size // 8,
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
                (self.sc_k, "rope_k", f"k_cache_{i}"),
                (self.sc_v, f"qkv_out[{v_start}:{v_end}]", f"v_cache_{i}"),
                (self.repeat, f"k_cache_{i}", "scores_keys"),
                (self.repeat, f"v_cache_{i}", "scores_values"),
                (self.gemv_scores, "scores_keys", "rope_q", "scores"),
                (self.attn_scale, "scores", "attn_scale", "scores"),
                (self.softmax, "scores", "weights"),
            ] + [
                (self.transpose_v,
                 f"scores_values[{h * vps}:{(h + 1) * vps}]",
                 f"scores_values_T[{h * vps}:{(h + 1) * vps}]")
                for h in range(n_heads)
            ] + [
                (self.gemv_context, "scores_values_T", "weights", "context"),
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
        # Final RMSNorm stays in the sequence (bf16 weighted, matches llama). It emits
        # an independent output buffer "x_final" so "x" is not an in-out arg (which
        # would collide in subbuffer_layout). lm_head (f32 argmax) runs as an
        # independent dispatch on x_final.
        runlist += [(self.rms, "x", "W_final_norm", "x_final")]

        cache_size = n_kv_heads * max_seq_len * head_dim * 2
        self.seq = OperatorSequence(
            "qwen3_0.6b_decode",
            runlist,
            input_args=["x", "rope_angles"],
            output_args=["x_final"],
            buffer_sizes={
                **{f"k_cache_{i}": cache_size for i in range(n_layers)},
                **{f"v_cache_{i}": cache_size for i in range(n_layers)},
                "scores_values": n_heads * max_seq_len * head_dim * 2,
                "scores_values_T": n_heads * max_seq_len * head_dim * 2,
                "rope_angles": head_dim * 2,
            },
            context=self.context,
        )
        self.seq.compile()
        self.fc = self.seq.get_callable()
        print(f"compiled OK; buffer_sizes: {self.seq.buffer_sizes}")
        print(f"n buffers: {len(self.seq.subbuffer_layout)}")

    # ---------------- weight load ----------------
    def _load_weights(self):
        scale = 1.0 / math.sqrt(head_dim)
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
        attn_scale_buf = self.fc.get_buffer("attn_scale").torch_view()
        attn_scale_buf[:] = torch.full(attn_scale_buf.shape, scale, dtype=torch.bfloat16)

        # RoPE angle table (interleaved cos/sin, half dim)
        self._precompute_rope_angles()

        # sync static buffers to device
        self.fc.input_buffer.to("npu")
        self.fc.scratch_buffer.to("npu")

        # ---- independent lm_head (bf16 GEMV + f32 argmax) ----
        self.lmhead_op.compile()
        self.lmhead_call = self.lmhead_op.get_callable()
        self.lmhead_mat = XRTTensor.from_torch(self.out_head.flatten().contiguous())  # bf16 (151936*1024,)
        self.lmhead_vec = XRTTensor.from_torch(torch.zeros(emb_dim, dtype=torch.bfloat16))
        self.lmhead_out = XRTTensor.from_torch(torch.zeros(16, dtype=torch.float32))
        self.lmhead_mat.buffer_object().sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)

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
        cache_offset = seq_pos * head_dim

        # update per-token rope_angles
        self.fc.get_buffer("rope_angles").torch_view()[:] = self.angle_table[seq_pos]
        # write x
        self.fc.get_buffer("x").torch_view()[:] = x

        self.fc.params.write("cache_offset", np.int32(cache_offset))
        self.fc.params.write("softmax_vector_size", np.int32(seq_pos + 1))
        self.fc.params.sync()

        self.fc()  # runs 28 layers + final RMSNorm, leaves x_final in output buffer

        # x_final is already synced to CPU by SequenceCallable.__call__ (_sync_outputs)
        x_final = self.fc.get_buffer("x_final").torch_view()
        self.lmhead_vec.to_torch().copy_(x_final)  # bf16 -> bf16, no cast
        self.lmhead_vec.buffer_object().sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_TO_DEVICE)
        self.lmhead_call(self.lmhead_mat, self.lmhead_vec, self.lmhead_out)
        self.lmhead_out.buffer_object().sync(xrt.xclBOSyncDirection.XCL_BO_SYNC_BO_FROM_DEVICE)
        out = self.lmhead_out.to_torch()
        vals = out[0::2]
        idxs = out[1::2].to(torch.int32)
        best = torch.argmax(vals).item()
        return idxs[best].item()


if __name__ == "__main__":
    # Smoke test: load weights, run one decode step, report argmax + no NaN.
    model = Qwen3NPU("/home/zyc/Packages/NPU/Qwen3-0.6B/model.safetensors")
    x = torch.randn(emb_dim, dtype=torch.bfloat16)
    token_id = model(x, seq_pos=10)
    print(f"argmax token_id = {token_id}")
