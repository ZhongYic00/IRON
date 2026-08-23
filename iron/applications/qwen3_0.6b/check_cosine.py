#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HF cosine check: prefill the SAME prompt through HF (f32) and the pure-iron
NPU (bf16), then compare the final-norm hidden state (the vector feeding
lm_head) cosine. Avoids greedy-decoding divergence (Qwen3 thinking mode).

cosine > 0.99 on the final hidden proves the 28-layer accumulation is correct.
"""
import sys, time
import torch
import numpy as np

sys.path.insert(0, "/home/zyc/Github/iron/iron/applications/qwen3_0.6b")
from qwen3_npu import Qwen3NPU, emb_dim

from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/zyc/Packages/NPU/Qwen3-0.6B/"

def main():
    prompt = sys.argv[1] if len(sys.argv) > 1 else "你好，请介绍一下你自己"
    print("loading NPU...", flush=True)
    npu = Qwen3NPU(MODEL_PATH + "model.safetensors")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    emb = safe_open(MODEL_PATH + "model.safetensors", framework="pt").get_tensor(
        "model.embed_tokens.weight").to(torch.bfloat16)

    print("loading HF...", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
    hf.eval()

    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    assert input_ids.shape[0] == 1
    prompt_ids = input_ids[0].tolist()
    L = len(prompt_ids)
    print(f"prompt: {L} tokens", flush=True)

    # ---- NPU prefill (token by token) ----
    for pos, tid in enumerate(prompt_ids):
        x = emb[tid]
        npu(x, seq_pos=pos)
    x_final_npu = npu.fc.get_buffer("x_final").torch_view().to(torch.float32)  # (1024,)

    # ---- HF prefill (single forward) ----
    with torch.no_grad():
        out = hf(input_ids, output_hidden_states=True)
    last_hidden = out.hidden_states[-1]                      # (1, L, 1024)
    hf_final_norm = hf.model.norm(last_hidden)               # (1, L, 1024) after final RMSNorm
    finalvec_hf = hf_final_norm[0, -1, :].to(torch.float32)  # last position

    # ---- cosine ----
    cos = torch.nn.functional.cosine_similarity(x_final_npu, finalvec_hf, dim=0)
    maxerr = (x_final_npu - finalvec_hf).abs().max()
    nan = torch.isnan(x_final_npu).any().item()
    print(f"\nNPU x_final[0:5] = {x_final_npu[:5].tolist()}")
    print(f"HF  final[0:5]   = {finalvec_hf[:5].tolist()}")
    print(f"cosine = {cos.item():.6f}")
    print(f"maxerr = {maxerr.item():.6f}")
    print(f"nan = {nan}")
    print("PASS" if cos.item() > 0.99 and not nan else "CHECK")

if __name__ == "__main__":
    main()
