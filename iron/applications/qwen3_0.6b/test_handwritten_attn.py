#!/usr/bin/env python3
"""Minimal pure-iron handwritten decode attention (one layer), to verify that
KV cache living in scratch + StridedCopy incremental write + per-head GEMV
attention does NOT regress like the built-in MHA reading from scratch.

QKNormRoPE is used for QK-norm + RoPE + V-copy (Qwen3 has QK-norm).  Attention
scores/context are handwritten with GEMV (num_batches=n_heads) + Softmax +
Transpose, like llama_npu.py.
"""
import torch
import numpy as np
import ml_dtypes

from aie.iron.device import NPU2
import aie.utils as aie_utils
aie_utils.set_current_device(NPU2())

from iron.common.context import AIEContext
from iron.common.sequence import OperatorSequence
from iron.operators import (
    RMSNorm, GEMV, StridedCopy, Repeat, Softmax,
    ElementwiseMul, ElementwiseAdd, SiLU, Transpose, RoPE, QKNorm,
)
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor
import pyxrt as xrt

emb_dim, hidden_dim = 1024, 3072
n_heads, n_kv_heads, head_dim = 16, 8, 128
q_dim = n_heads * head_dim
kv_dim = n_kv_heads * head_dim
qkv_dim = q_dim + 2 * kv_dim
prompt_len = 512  # max context (S_kv)

ctx = AIEContext()
ctx.build_dir.mkdir(parents=True, exist_ok=True)

# --- operators ---
rms_norm = RMSNorm(size=emb_dim, num_aie_columns=1, num_channels=1,
                   tile_size=emb_dim, weighted=True, context=ctx)
gemv_qkv = GEMV(M=qkv_dim, K=emb_dim, num_aie_columns=8,
                tile_size_input=4, tile_size_output=qkv_dim // 8, context=ctx)
qk_norm = QKNorm(head_dim=head_dim, n_q_heads=n_heads, n_k_heads=n_kv_heads,
                 epsilon=1e-6, context=ctx)
rope_q = RoPE(rows=n_heads, cols=head_dim, angle_rows=1, context=ctx)
rope_k = RoPE(rows=n_kv_heads, cols=head_dim, angle_rows=1, context=ctx)
sc_k = StridedCopy(input_sizes=(n_kv_heads, head_dim), input_strides=(head_dim, 1),
                   input_offset=0, output_sizes=(n_kv_heads, head_dim),
                   output_strides=(prompt_len * head_dim, 1), output_offset=0,
                   input_buffer_size=kv_dim,
                   output_buffer_size=n_kv_heads * prompt_len * head_dim,
                   num_aie_channels=1, output_offset_parameter="cache_offset", context=ctx)
sc_v = StridedCopy(input_sizes=(n_kv_heads, head_dim), input_strides=(head_dim, 1),
                   input_offset=0, output_sizes=(n_kv_heads, head_dim),
                   output_strides=(prompt_len * head_dim, 1), output_offset=0,
                   input_buffer_size=kv_dim,
                   output_buffer_size=n_kv_heads * prompt_len * head_dim,
                   num_aie_channels=1, output_offset_parameter="cache_offset", context=ctx)
gemv_scores = GEMV(M=prompt_len, K=head_dim, num_aie_columns=8, tile_size_input=4,
                   tile_size_output=prompt_len // 8, num_batches=n_heads, context=ctx)
attn_scale = ElementwiseMul(size=n_heads * prompt_len, tile_size=prompt_len // 8,
                            num_aie_columns=8, context=ctx)
softmax = Softmax(rows=n_heads, cols=prompt_len, num_aie_columns=1, num_channels=1,
                  rtp_vector_size=prompt_len, vector_size_parameter="softmax_vector_size",
                  context=ctx)
transpose_v = Transpose(M=prompt_len, N=head_dim, num_aie_columns=2, num_channels=1,
                        m=256, n=32, s=8, context=ctx)
gemv_context = GEMV(M=head_dim, K=prompt_len, num_aie_columns=8, tile_size_input=4,
                    tile_size_output=4, num_batches=n_heads, context=ctx)
gemv_output = GEMV(M=emb_dim, K=q_dim, num_aie_columns=8, tile_size_input=4,
                   tile_size_output=emb_dim // 8, context=ctx)
residual_add = ElementwiseAdd(size=emb_dim, tile_size=emb_dim // 8, context=ctx)
gemv_ffn_up = GEMV(M=hidden_dim, K=emb_dim, num_aie_columns=8, tile_size_input=4,
                   tile_size_output=hidden_dim // 8, context=ctx)
silu = SiLU(size=hidden_dim, tile_size=hidden_dim // 8, num_aie_columns=8, context=ctx)
mul = ElementwiseMul(size=hidden_dim, tile_size=hidden_dim // 8, num_aie_columns=8, context=ctx)
gemv_ffn_down = GEMV(M=emb_dim, K=hidden_dim, num_aie_columns=8, tile_size_input=1,
                     tile_size_output=emb_dim // 8, context=ctx)
repeat = Repeat(rows=n_kv_heads, cols=prompt_len * head_dim,
                repeat=n_heads // n_kv_heads, transfer_size=head_dim, context=ctx)

# qkv_rope layout: [Q(q_dim) | K(kv_dim) | V(kv_dim)].  Slice byte offsets.
q_end = q_dim * 2
k_start = q_dim * 2
k_end = (q_dim + kv_dim) * 2
v_start = k_end
v_end = qkv_dim * 2

# per-head values size in bytes (for the per-head transpose slices)
vps = prompt_len * head_dim * 2

runlist = [
    (rms_norm, "x", "W_norm", "x_norm"),
    (gemv_qkv, "W_qkv", "x_norm", "qkv_out"),
    # QK-norm: per-head RMSNorm on [Q|K] segment of qkv_out (V untouched)
    (qk_norm, f"qkv_out[0:{k_end}]", "qk_normed", "W_qk_gamma"),
    # RoPE: rotate Q (16 heads) and K (8 heads) independently; angles are a
    # per-token variable buffer (like llama's rope_angles)
    (rope_q, f"qk_normed[0:{q_end}]", "rope_angles", "rope_q"),
    (rope_k, f"qk_normed[{q_end}:{k_end}]", "rope_angles", "rope_k"),
    # write K and V into per-layer cache
    (sc_k, "rope_k", "k_cache"),
    (sc_v, f"qkv_out[{v_start}:{v_end}]", "v_cache"),
    (repeat, "k_cache", "scores_keys"),
    (repeat, "v_cache", "scores_values"),
    (gemv_scores, "scores_keys", "rope_q", "scores"),
    (attn_scale, "scores", "attn_scale", "scores"),
    (softmax, "scores", "weights"),
] + [
    (transpose_v, f"scores_values[{h * vps}:{(h + 1) * vps}]",
     f"scores_values_T[{h * vps}:{(h + 1) * vps}]")
    for h in range(n_heads)
] + [
    (gemv_context, "scores_values_T", "weights", "context"),
    (gemv_output, "W_o", "context", "attn_output"),
    (residual_add, "x", "attn_output", "x"),
    # SwiGLU MLP (post-attn norm + FFN + residual)
    (rms_norm, "x", "W_norm2", "x_norm2"),
    (gemv_ffn_up, "W_gate", "x_norm2", "ffn_gate"),
    (gemv_ffn_up, "W_up", "x_norm2", "ffn_up"),
    (silu, "ffn_gate", "ffn_gate"),
    (mul, "ffn_gate", "ffn_up", "ffn_hidden"),
    (gemv_ffn_down, "W_down", "ffn_hidden", "ffn_output"),
    (residual_add, "x", "ffn_output", "x"),
]

seq = OperatorSequence(
    "qwen3_attn_decode_1layer",
    runlist,
    input_args=["x", "rope_angles"],
    output_args=["x"],
    buffer_sizes={
        "k_cache": n_kv_heads * prompt_len * head_dim * 2,
        "v_cache": n_kv_heads * prompt_len * head_dim * 2,
        "scores_values": n_heads * prompt_len * head_dim * 2,
        "scores_values_T": n_heads * prompt_len * head_dim * 2,
        "rope_angles": head_dim * 2,  # (1, head_dim) interleaved cos/sin
    },
    context=ctx,
)
seq.compile()
fc = seq.get_callable()
print("compiled OK")
print("buffer_sizes:", seq.buffer_sizes)
print("n buffers:", len(seq.subbuffer_layout))

# weight init (random, just to verify compile + no NaN)
for name in seq.subbuffer_layout:
    if name in ("x", "attn_output", "scores", "weights", "context", "x_norm", "qkv_out", "qk_normed", "rope_q", "rope_k", "scores_keys", "scores_values", "scores_values_T"):
        continue
    try:
        fc.get_buffer(name).torch_view()[:] = torch.randn(fc.get_buffer(name).torch_view().shape, dtype=torch.bfloat16).flatten()
    except Exception as e:
        pass

x = torch.randn(emb_dim, dtype=torch.bfloat16)
fc.get_buffer("x").torch_view()[:] = x
fc.get_buffer("x").to("npu")
fc.scratch_buffer.to("npu")

fc.params.write("cache_offset", np.int32(10 * head_dim))
fc.params.write("softmax_vector_size", np.int32(11))
fc.params.sync()

fc()
fc.get_buffer("x").to("cpu")
out = fc.get_buffer("x").torch_view()
print(f"output shape {out.shape}, nan={torch.isnan(out).any().item()}, absmax={out.abs().max().item():.4f}")
