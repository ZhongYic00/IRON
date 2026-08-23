#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""torch-only 28-layer reference vs HF, per-layer cosine, to locate the layer
where NPU/bfloat16 accumulation first diverges (also validates my layer math)."""
import sys, time
import torch
import math
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/zyc/Packages/NPU/Qwen3-0.6B/"

emb_dim, hidden_dim = 1024, 3072
n_heads, n_kv_heads, head_dim = 16, 8, 128
n_layers = 28
eps = 1e-6
rope_theta = 1e6

def rms_norm(x, w):
    xf = x.to(torch.float32)
    return (xf * torch.rsqrt(xf.pow(2).mean() + eps) * w.to(torch.float32)).to(torch.bfloat16)

def qk_norm(x, w):
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return (xf * torch.rsqrt(var + eps) * w.float()).to(torch.bfloat16)

def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

def rope(xr, cos, sin):
    xf = xr.float()
    return (xf * cos + rotate_half(xf) * sin).to(torch.bfloat16)

def main():
    sf = safe_open(MODEL_PATH + "model.safetensors", framework="pt")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    hf = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
    hf.eval()

    msgs = [{"role": "user", "content": "你好，请介绍一下你自己"}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    L = input_ids.shape[1]

    emb = sf.get_tensor("model.embed_tokens.weight").to(torch.bfloat16)

    # HF forward, capture per-layer hidden
    with torch.no_grad():
        out = hf(input_ids, output_hidden_states=True)
    hf_hiddens = out.hidden_states  # tuple len n_layers+1; [0]=embed, [i]=after layer i

    # my torch reference, per-layer vs HF
    x = emb[input_ids[0].tolist()]  # (L, 1024)
    cos_per_layer = []
    for i in range(n_layers):
        Lw = f"model.layers.{i}"
        w_in_ln = sf.get_tensor(f"{Lw}.input_layernorm.weight").to(torch.bfloat16)
        w_q = sf.get_tensor(f"{Lw}.self_attn.q_proj.weight").to(torch.bfloat16)
        w_k = sf.get_tensor(f"{Lw}.self_attn.k_proj.weight").to(torch.bfloat16)
        w_v = sf.get_tensor(f"{Lw}.self_attn.v_proj.weight").to(torch.bfloat16)
        w_o = sf.get_tensor(f"{Lw}.self_attn.o_proj.weight").to(torch.bfloat16)
        w_qn = sf.get_tensor(f"{Lw}.self_attn.q_norm.weight").to(torch.bfloat16)
        w_kn = sf.get_tensor(f"{Lw}.self_attn.k_norm.weight").to(torch.bfloat16)
        w_ln2 = sf.get_tensor(f"{Lw}.post_attention_layernorm.weight").to(torch.bfloat16)
        w_g = sf.get_tensor(f"{Lw}.mlp.gate_proj.weight").to(torch.bfloat16)
        w_u = sf.get_tensor(f"{Lw}.mlp.up_proj.weight").to(torch.bfloat16)
        w_d = sf.get_tensor(f"{Lw}.mlp.down_proj.weight").to(torch.bfloat16)

        xn = rms_norm(x, w_in_ln)  # (L, 1024)
        qkv = (torch.cat([w_q, w_k, w_v], 0).float() @ xn.float().T).T.to(torch.bfloat16)  # (L, 4096)
        q = qkv[:, :2048].reshape(L, n_heads, head_dim)
        k = qkv[:, 2048:3072].reshape(L, n_kv_heads, head_dim)
        v = qkv[:, 3072:].reshape(L, n_kv_heads, head_dim)
        q = qk_norm(q, w_qn); k = qk_norm(k, w_kn)

        # rope
        half = head_dim // 2
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        freqs = torch.outer(torch.arange(L, dtype=torch.float32), inv_freq)
        emb_rot = torch.cat([freqs, freqs], dim=-1)
        cos = emb_rot.cos().to(torch.float32).unsqueeze(1)  # (L, 1, 128)
        sin = emb_rot.sin().to(torch.float32).unsqueeze(1)
        q = rope(q, cos, sin); k = rope(k, cos, sin)

        # GQA: repeat k/v to n_heads FIRST, then causal attention
        k_all = k.repeat_interleave(n_heads // n_kv_heads, dim=1)  # (L, 16, 128)
        v_all = v.repeat_interleave(n_heads // n_kv_heads, dim=1)  # (L, 16, 128)
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.einsum('bhd,mhd->bhm', q.float(), k_all.float()) * scale  # (L,16,L)
        causal = torch.tril(torch.ones(L, L, dtype=torch.bool))
        scores = scores.masked_fill(~causal.unsqueeze(1), float('-inf'))  # (L,1,L) broadcast
        attn = torch.softmax(scores, dim=-1).to(torch.bfloat16)  # (L,16,L)
        ctx = torch.einsum('bhm,mhd->bhd', attn.float(), v_all.float()).reshape(L, 2048).to(torch.bfloat16)
        attn_out = (w_o.float() @ ctx.float().T).T.to(torch.bfloat16)  # (L,1024)
        x = (x.float() + attn_out.float()).to(torch.bfloat16)

        xn2 = rms_norm(x, w_ln2)
        gate = (w_g.float() @ xn2.float().T).T.to(torch.bfloat16)
        up = (w_u.float() @ xn2.float().T).T.to(torch.bfloat16)
        gate = torch.nn.functional.silu(gate.float()).to(torch.bfloat16)
        ffn = (gate.float() * up.float()).to(torch.bfloat16)
        ffn_out = (w_d.float() @ ffn.float().T).T.to(torch.bfloat16)
        x = (x.float() + ffn_out.float()).to(torch.bfloat16)

        # compare last position hidden vs HF layer i hidden (final-norm NOT applied here; HF hidden_states[i] is post-residual pre-norm)
        hf_h = out.hidden_states[i + 1][0, -1, :].to(torch.float32)  # after layer i
        my_h = x[-1].to(torch.float32)
        cos = torch.nn.functional.cosine_similarity(my_h, hf_h, dim=0)
        cos_per_layer.append(cos.item())
        print(f"layer {i}: cosine = {cos:.6f}")

    # final norm cosine
    w_fin = sf.get_tensor("model.norm.weight").to(torch.bfloat16)
    x_final = rms_norm(x, w_fin)[-1].to(torch.float32)
    hf_final = hf.model.norm(out.hidden_states[-1])[0, -1, :].to(torch.float32)
    cos_fin = torch.nn.functional.cosine_similarity(x_final, hf_final, dim=0)
    print(f"final norm: cosine = {cos_fin:.6f}")
    print("MIN layer cosine:", min(cos_per_layer))

if __name__ == "__main__":
    main()
