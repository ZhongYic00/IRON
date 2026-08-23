#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Bisect the 28-layer accumulation error: prefill N prompt tokens and compare
the final-norm hidden at each token position against HF. Reveals whether the
error is in KV-cache accumulation (multi-token) or a per-layer issue."""
import sys, time
import torch
import numpy as np

sys.path.insert(0, "/home/zyc/Github/iron/iron/applications/qwen3_0.6b")
from qwen3_npu import Qwen3NPU, emb_dim

from safetensors import safe_open
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_PATH = "/home/zyc/Packages/NPU/Qwen3-0.6B/"

def main():
    n_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    print("loading NPU...", flush=True)
    npu = Qwen3NPU(MODEL_PATH + "model.safetensors")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    emb = safe_open(MODEL_PATH + "model.safetensors", framework="pt").get_tensor(
        "model.embed_tokens.weight").to(torch.bfloat16)

    print("loading HF...", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
    hf.eval()

    messages = [{"role": "user", "content": "你好，请介绍一下你自己"}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(text, return_tensors="pt")["input_ids"]
    prompt_ids = ids[0][:n_tokens].tolist()
    if n_tokens < ids.shape[1]:
        print(f"truncated to first {n_tokens} of {ids.shape[1]} prompt tokens")

    # ---- NPU prefill token by token, dump each position's x_final ----
    npu_vectors = []
    for pos, tid in enumerate(prompt_ids):
        x = emb[tid]
        npu(x, seq_pos=pos)
        npu_vectors.append(npu.fc.get_buffer("x_final").torch_view().to(torch.float32).clone())

    # ---- HF prefill, final-norm hidden per position ----
    in_ids = torch.tensor([prompt_ids])
    with torch.no_grad():
        out = hf(in_ids, output_hidden_states=True)
    last_hidden = out.hidden_states[-1]
    hf_norm = hf.model.norm(last_hidden)  # (1, L, 1024)

    for pos in range(len(prompt_ids)):
        cos = torch.nn.functional.cosine_similarity(npu_vectors[pos], hf_norm[0, pos, :].to(torch.float32), dim=0)
        print(f"pos {pos}: cosine = {cos.item():.6f}")

if __name__ == "__main__":
    main()
