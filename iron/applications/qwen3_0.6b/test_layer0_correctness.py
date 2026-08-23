#!/usr/bin/env python3
"""Single-layer (layer 0) decode correctness check against a torch reference.

Loads REAL Qwen3-0.6B layer-0 weights, runs the handwritten-attention iron
OperatorSequence on the NPU, and compares against a pure-torch reference.
Verifies (before expanding to 28 layers) that QK-norm + RoPE + GQA attention +
MLP numerics are correct (cosine ~ 1.0, no NaN).
"""
import torch
import numpy as np
import math
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
from aie.utils.hostruntime.xrtruntime.tensor import XRTTensor

# Qwen3-0.6B config
emb_dim, hidden_dim = 1024, 3072
n_heads, n_kv_heads, head_dim = 16, 8, 128
q_dim = n_heads * head_dim          # 2048
kv_dim = n_kv_heads * head_dim      # 1024
qkv_dim = q_dim + 2 * kv_dim        # 4096
prompt_len = 512                     # max context (S_kv)
eps = 1e-6
rope_theta = 1e6
seq_pos = 10                         # decode position to verify at

# ---------------- load real layer-0 weights ----------------
weights_path = "/home/zyc/Packages/NPU/Qwen3-0.6B/model.safetensors"
sf = safe_open(weights_path, framework="pt")
L = "model.layers.0"
def W(name):
    return sf.get_tensor(f"{L}.{name}").to(torch.bfloat16)

w_input_norm = W("input_layernorm.weight")                  # (1024,)
w_q = W("self_attn.q_proj.weight")                          # (2048,1024)
w_k = W("self_attn.k_proj.weight")                          # (1024,1024)
w_v = W("self_attn.v_proj.weight")                          # (1024,1024)
w_o = W("self_attn.o_proj.weight")                          # (1024,2048)
w_q_norm = W("self_attn.q_norm.weight")                     # (128,)
w_k_norm = W("self_attn.k_norm.weight")                     # (128,)
w_norm2 = W("post_attention_layernorm.weight")              # (1024,)
w_gate = W("mlp.gate_proj.weight")                          # (3072,1024)
w_up = W("mlp.up_proj.weight")                              # (3072,1024)
w_down = W("mlp.down_proj.weight")                          # (1024,3072)

w_qkv = torch.cat([w_q, w_k, w_v], dim=0).contiguous()      # (4096,1024)
w_qk_gamma = torch.cat([w_q_norm, w_k_norm], dim=0)         # (256,)

# ---------------- torch reference (single decode token) ----------------
torch.manual_seed(0)
x_in = torch.randn(emb_dim, dtype=torch.bfloat16)

# RoPE angle table (interleaved cos/sin, half dim, method_type=0)
half = head_dim // 2
inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half))
freqs = inv_freq * seq_pos                       # (half,) at this position
emb = torch.cat([freqs, freqs], dim=-1)          # (head_dim,) duplicated
cos_full = emb.cos().to(torch.float32)           # (head_dim,)
sin_full = emb.sin().to(torch.float32)           # (head_dim,)
# interleaved [cos0, sin0, cos1, sin1, ...] over half dim (128 total elements)
cos_half = cos_full[:half]
sin_half = sin_full[:half]
rope_angles = torch.stack([cos_half, sin_half], dim=-1).reshape(head_dim).to(torch.bfloat16)  # (128,)

def rms_norm(x, w):
    xf = x.to(torch.float32)
    var = xf.pow(2).mean()
    return (xf * torch.rsqrt(var + eps) * w.to(torch.float32)).to(torch.bfloat16)

def qk_norm(x, w):
    # x: (n_heads, head_dim) or (n_kv_heads, head_dim); w: (head_dim,) shared
    xf = x.to(torch.float32)
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * w.to(torch.float32)).to(torch.bfloat16)

def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

def rope(xr, cos, sin):
    xf = xr.to(torch.float32)
    return (xf * cos + rotate_half(xf) * sin).to(torch.bfloat16)

def softmax_ref(x):
    xf = x.to(torch.float32)
    m = xf.max()
    e = torch.exp(xf - m)
    return e / e.sum()

# --- reference forward ---
x_norm = rms_norm(x_in, w_input_norm)                    # (1024,)
qkv = (w_qkv.to(torch.float32) @ x_norm.to(torch.float32)).to(torch.bfloat16)  # (4096,)
q = qkv[:q_dim].reshape(n_heads, head_dim)               # (16,128)
k = qkv[q_dim:q_dim+kv_dim].reshape(n_kv_heads, head_dim)# (8,128)
v = qkv[q_dim+kv_dim:].reshape(n_kv_heads, head_dim)     # (8,128)
q = qk_norm(q, w_q_norm)                                  # QK-norm on Q
k = qk_norm(k, w_k_norm)                                  # QK-norm on K
q = rope(q, cos_full, sin_full)                           # RoPE Q
k = rope(k, cos_full, sin_full)                           # RoPE K

# KV cache: prior keys/values (positions [0, seq_pos] valid; tail zeroed).
# For a real decode, only positions <= seq_pos are populated; tail is 0.
torch.manual_seed(1)
k_cache = torch.zeros(n_kv_heads, prompt_len, head_dim, dtype=torch.bfloat16)
v_cache = torch.zeros(n_kv_heads, prompt_len, head_dim, dtype=torch.bfloat16)
k_cache[:, :seq_pos] = torch.randn(n_kv_heads, seq_pos, head_dim, dtype=torch.bfloat16)
v_cache[:, :seq_pos] = torch.randn(n_kv_heads, seq_pos, head_dim, dtype=torch.bfloat16)
k_cache[:, seq_pos, :] = k
v_cache[:, seq_pos, :] = v

# GQA: repeat k/v to 16 heads
k_all = k_cache.repeat_interleave(n_heads // n_kv_heads, dim=0)   # (16, 512, 128)
v_all = v_cache.repeat_interleave(n_heads // n_kv_heads, dim=0)   # (16, 512, 128)

# scores: (16, 512) = q @ k^T / sqrt(head_dim)
scale = 1.0 / math.sqrt(head_dim)
scores_full = torch.einsum('hd,hpd->hp', q.to(torch.float32), k_all.to(torch.float32)) * scale
# causal mask: only positions [0, seq_pos] participate; tail -> 0 weight (softmax of -inf)
weights_ref = torch.zeros(n_heads, prompt_len, dtype=torch.float32)
for h in range(n_heads):
    s = scores_full[h, :seq_pos + 1]
    m = s.max()
    e = torch.exp(s - m)
    weights_ref[h, :seq_pos + 1] = e / e.sum()
context = torch.einsum('hp,hpd->hd', weights_ref, v_all.to(torch.float32))  # (16,128)
attn_out = (w_o.to(torch.float32) @ context.reshape(-1).to(torch.float32)).to(torch.bfloat16)  # (1024,)
x_res = (x_in.to(torch.float32) + attn_out.to(torch.float32)).to(torch.bfloat16)

# MLP
x_norm2 = rms_norm(x_res, w_norm2)
gate = (w_gate.to(torch.float32) @ x_norm2.to(torch.float32)).to(torch.bfloat16)
up = (w_up.to(torch.float32) @ x_norm2.to(torch.float32)).to(torch.bfloat16)
gate = torch.nn.functional.silu(gate.to(torch.float32)).to(torch.bfloat16)
ffn_hidden = (gate.to(torch.float32) * up.to(torch.float32)).to(torch.bfloat16)
ffn_out = (w_down.to(torch.float32) @ ffn_hidden.to(torch.float32)).to(torch.bfloat16)
x_ref = (x_res.to(torch.float32) + ffn_out.to(torch.float32)).to(torch.bfloat16)

print("torch reference done, x_ref shape", x_ref.shape)

# ---------------- NPU single-layer sequence ----------------
ctx = AIEContext()
ctx.build_dir.mkdir(parents=True, exist_ok=True)

rms_norm_op = RMSNorm(size=emb_dim, num_aie_columns=1, num_channels=1,
                      tile_size=emb_dim, weighted=True, epsilon=eps, context=ctx)
gemv_qkv_op = GEMV(M=qkv_dim, K=emb_dim, num_aie_columns=8,
                   tile_size_input=4, tile_size_output=qkv_dim // 8, context=ctx)
qk_norm_op = QKNorm(head_dim=head_dim, n_q_heads=n_heads, n_k_heads=n_kv_heads,
                    epsilon=eps, context=ctx)
rope_q_op = RoPE(rows=n_heads, cols=head_dim, angle_rows=1, context=ctx)
rope_k_op = RoPE(rows=n_kv_heads, cols=head_dim, angle_rows=1, context=ctx)
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
softmax_op = Softmax(rows=n_heads, cols=prompt_len, num_aie_columns=1, num_channels=1,
                     rtp_vector_size=prompt_len, vector_size_parameter="softmax_vector_size",
                     context=ctx)
transpose_v = Transpose(M=prompt_len, N=head_dim, num_aie_columns=2, num_channels=1,
                        m=256, n=32, s=8, context=ctx)
gemv_context = GEMV(M=head_dim, K=prompt_len, num_aie_columns=8, tile_size_input=4,
                    tile_size_output=4, num_batches=n_heads, context=ctx)
gemv_output = GEMV(M=emb_dim, K=q_dim, num_aie_columns=8, tile_size_input=4,
                   tile_size_output=emb_dim // 8, context=ctx)
residual_add = ElementwiseAdd(size=emb_dim, tile_size=emb_dim // 8, context=ctx)
gemv_ffn = GEMV(M=hidden_dim, K=emb_dim, num_aie_columns=8, tile_size_input=4,
                tile_size_output=hidden_dim // 8, context=ctx)
silu_op = SiLU(size=hidden_dim, tile_size=hidden_dim // 8, num_aie_columns=8, context=ctx)
mul = ElementwiseMul(size=hidden_dim, tile_size=hidden_dim // 8, num_aie_columns=8, context=ctx)
gemv_ffn_down = GEMV(M=emb_dim, K=hidden_dim, num_aie_columns=8, tile_size_input=1,
                     tile_size_output=emb_dim // 8, context=ctx)
repeat = Repeat(rows=n_kv_heads, cols=prompt_len * head_dim,
                repeat=n_heads // n_kv_heads, transfer_size=head_dim, context=ctx)

q_end = q_dim * 2
k_end = (q_dim + kv_dim) * 2
v_start = k_end
v_end = qkv_dim * 2
vps = prompt_len * head_dim * 2

runlist = [
    (rms_norm_op, "x", "W_norm", "x_norm"),
    (gemv_qkv_op, "W_qkv", "x_norm", "qkv_out"),
    (qk_norm_op, f"qkv_out[0:{k_end}]", "qk_normed", "W_qk_gamma"),
    (rope_q_op, f"qk_normed[0:{q_end}]", "rope_angles", "rope_q"),
    (rope_k_op, f"qk_normed[{q_end}:{k_end}]", "rope_angles", "rope_k"),
    (sc_k, "rope_k", "k_cache"),
    (sc_v, f"qkv_out[{v_start}:{v_end}]", "v_cache"),
    (repeat, "k_cache", "scores_keys"),
    (repeat, "v_cache", "scores_values"),
    (gemv_scores, "scores_keys", "rope_q", "scores"),
    (attn_scale, "scores", "attn_scale", "scores"),
    (softmax_op, "scores", "weights"),
] + [
    (transpose_v, f"scores_values[{h * vps}:{(h + 1) * vps}]",
     f"scores_values_T[{h * vps}:{(h + 1) * vps}]")
    for h in range(n_heads)
] + [
    (gemv_context, "scores_values_T", "weights", "context"),
    (gemv_output, "W_o", "context", "attn_output"),
    (residual_add, "x", "attn_output", "x"),
    (rms_norm_op, "x", "W_norm2", "x_norm2"),
    (gemv_ffn, "W_gate", "x_norm2", "ffn_gate"),
    (gemv_ffn, "W_up", "x_norm2", "ffn_up"),
    (silu_op, "ffn_gate", "ffn_gate"),
    (mul, "ffn_gate", "ffn_up", "ffn_hidden"),
    (gemv_ffn_down, "W_down", "ffn_hidden", "ffn_output"),
    (residual_add, "x", "ffn_output", "x"),
]

seq = OperatorSequence(
    "qwen3_layer0_correctness",
    runlist,
    input_args=["x", "rope_angles"],
    output_args=["x"],
    buffer_sizes={
        "k_cache": n_kv_heads * prompt_len * head_dim * 2,
        "v_cache": n_kv_heads * prompt_len * head_dim * 2,
        "scores_values": n_heads * prompt_len * head_dim * 2,
        "scores_values_T": n_heads * prompt_len * head_dim * 2,
        "rope_angles": head_dim * 2,
    },
    context=ctx,
)
seq.compile()
fc = seq.get_callable()
print("compiled OK; buffers:", len(seq.subbuffer_layout))

# ---- load real weights into static buffers ----
def set_buf(name, t):
    fc.get_buffer(name).torch_view()[:] = t.flatten() if t.dim() > 1 else t

set_buf("W_norm", w_input_norm)
set_buf("W_qkv", w_qkv)
set_buf("W_qk_gamma", w_qk_gamma)
set_buf("W_o", w_o)
set_buf("W_norm2", w_norm2)
set_buf("W_gate", w_gate)
set_buf("W_up", w_up)
set_buf("W_down", w_down)
fc.get_buffer("attn_scale").torch_view()[:] = torch.tensor([scale], dtype=torch.bfloat16).expand(
    fc.get_buffer("attn_scale").torch_view().numel())

# k_cache / v_cache: write prior cache contents (flatten as (n_kv_heads, prompt_len, head_dim))
fc.get_buffer("k_cache").torch_view()[:] = k_cache.reshape(-1)
fc.get_buffer("v_cache").torch_view()[:] = v_cache.reshape(-1)

# input
fc.get_buffer("x").torch_view()[:] = x_in
fc.get_buffer("rope_angles").torch_view()[:] = rope_angles
fc.get_buffer("x").to("npu")
fc.get_buffer("rope_angles").to("npu")
fc.scratch_buffer.to("npu")

fc.params.write("cache_offset", np.int32(seq_pos * head_dim))
fc.params.write("softmax_vector_size", np.int32(seq_pos + 1))
fc.params.sync()

fc()
fc.get_buffer("x").to("cpu")
x_npu = fc.get_buffer("x").torch_view().to(torch.float32)

x_ref_f = x_ref.to(torch.float32)
cos = torch.nn.functional.cosine_similarity(x_npu.reshape(-1), x_ref_f.reshape(-1), dim=0)
maxerr = (x_npu - x_ref_f).abs().max()
print(f"x_npu[0:6]  = {x_npu[:6].tolist()}")
print(f"x_ref[0:6]  = {x_ref_f[:6].tolist()}")
print(f"cosine = {cos.item():.6f}")
print(f"maxerr = {maxerr.item():.6f}")
print(f"nan_npu = {torch.isnan(x_npu).any().item()}")
print("PASS" if cos.item() > 0.99 and not torch.isnan(x_npu).any() else "FAIL")
