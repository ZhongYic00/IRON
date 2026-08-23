#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinpoint which sub-step of layer 27 diverges: RMSNorm / attention / MLP,
by calling HF's module-level forward and comparing each intermediate."""
import torch, math
from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/zyc/Packages/NPU/Qwen3-0.6B/"
emb_dim, hidden_dim = 1024, 3072
n_heads, n_kv_heads, head_dim = 16, 8, 128
eps = 1e-6
rope_theta = 1e6

def rms_norm(x, w):
    xf = x.to(torch.float32)
    return (xf * torch.rsqrt(xf.pow(2).mean() + eps) * w.to(torch.float32))

def qk_norm(x, w):
    xf = x.float()
    var = xf.pow(2).mean(dim=-1, keepdim=True)
    return xf * torch.rsqrt(var + eps) * w.float()

def rotate_half(x):
    h = x.shape[-1] // 2
    return torch.cat([-x[..., h:], x[..., :h]], dim=-1)

def main():
    sf = safe_open(MODEL_PATH + "model.safetensors", framework="pt")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    hf = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
    hf.eval()

    msgs = [{"role": "user", "content": "你好"}]
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"][:, :1]  # L=1

    emb = sf.get_tensor("model.embed_tokens.weight").to(torch.bfloat16)
    x = emb[input_ids[0].tolist()]  # (1, 1024)

    # HF 逐层, 拿到 layer 26 后的 hidden
    with torch.no_grad():
        h = hf.model.embed_tokens(input_ids)
        for i in range(27):
            h = hf.model.layers[i](h)[0] if isinstance(hf.model.layers[i](h), tuple) else hf.model.layers[i](h)
        x_in = h  # (1, 1, 1024) layer 27 输入

    # 我的 layer 27 各子步骤
    L = "model.layers.27"
    w_in = sf.get_tensor(f"{L}.input_layernorm.weight").to(torch.bfloat16)
    w_q = sf.get_tensor(f"{L}.self_attn.q_proj.weight").to(torch.bfloat16)
    w_k = sf.get_tensor(f"{L}.self_attn.k_proj.weight").to(torch.bfloat16)
    w_v = sf.get_tensor(f"{L}.self_attn.v_proj.weight").to(torch.bfloat16)
    w_o = sf.get_tensor(f"{L}.self_attn.o_proj.weight").to(torch.bfloat16)
    w_qn = sf.get_tensor(f"{L}.self_attn.q_norm.weight").to(torch.bfloat16)
    w_kn = sf.get_tensor(f"{L}.self_attn.k_norm.weight").to(torch.bfloat16)
    w_ln2 = sf.get_tensor(f"{L}.post_attention_layernorm.weight").to(torch.bfloat16)
    w_g = sf.get_tensor(f"{L}.mlp.gate_proj.weight").to(torch.bfloat16)
    w_u = sf.get_tensor(f"{L}.mlp.up_proj.weight").to(torch.bfloat16)
    w_d = sf.get_tensor(f"{L}.mlp.down_proj.weight").to(torch.bfloat16)

    xv = x_in[0, -1].to(torch.float32)  # (1024,)
    xn = rms_norm(xv, w_in)
    # HF input_layernorm
    with torch.no_grad():
        hf_xn = hf.model.layers[27].input_layernorm(x_in)
    cos = torch.nn.functional.cosine_similarity(xn, hf_xn[0,-1].float(), dim=0)
    print(f"input_layernorm: cosine={cos:.6f}  mine[{xn[:3]}] hf[{hf_xn[0,-1,:3]}]")

    # post_attention_layernorm (用同一个 x)  —— 直接对比 norm weight 大的那个
    with torch.no_grad():
        hf_ln2 = hf.model.layers[27].post_attention_layernorm(x_in)
    xn2 = rms_norm(xv, w_ln2)
    cos2 = torch.nn.functional.cosine_similarity(xn2, hf_ln2[0,-1].float(), dim=0)
    print(f"post_attention_layernorm: cosine={cos2:.6f}")

    # q_norm / k_norm (per head)
    q = (w_q.float() @ xn).reshape(n_heads, head_dim)
    q_mine = qk_norm(q, w_qn)
    with torch.no_grad():
        hf_q = hf.model.layers[27].self_attn.q_norm(xn.reshape(1,1,n_heads,head_dim).transpose(1,2))
    cosq = torch.nn.functional.cosine_similarity(q_mine.reshape(-1), hf_q[0,:,0,:].reshape(-1).float(), dim=0)
    print(f"q_norm: cosine={cosq:.6f}")

if __name__ == "__main__":
    main()
