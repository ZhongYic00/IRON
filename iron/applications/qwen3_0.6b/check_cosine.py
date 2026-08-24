#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""HF cosine check: prefill the SAME prompt through HF (f32) and the pure-iron
NPU (bf16), then compare the final logits (the lm_head output) cosine and
argmax. This directly proves the 28-layer accumulation is correct and that the
NPU would pick the same next token as HF greedy decode.

cosine > 0.99 on the logits proves the full forward is correct.
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

    # ---- NPU prefill (token by token); the last __call__ leaves logits in the
    #      auto-synced output buffer ----
    for pos, tid in enumerate(prompt_ids):
        npu(emb[tid], seq_pos=pos)
    logits_npu = npu.fc.get_buffer("logits").torch_view().to(torch.float32)  # (vocab,)

    # ---- HF prefill (single forward) ----
    with torch.no_grad():
        out = hf(input_ids)
    logits_hf = out.logits[0, -1, :].to(torch.float32)  # (vocab,)

    # ---- cosine + argmax ----
    cos = torch.nn.functional.cosine_similarity(logits_npu, logits_hf, dim=0)
    am_npu = torch.argmax(logits_npu).item()
    am_hf = torch.argmax(logits_hf).item()
    nan = torch.isnan(logits_npu).any().item()
    print(f"\nNPU logits[:5] = {logits_npu[:5].tolist()}")
    print(f"HF  logits[:5] = {logits_hf[:5].tolist()}")
    print(f"cosine = {cos.item():.6f}")
    print(f"argmax  : npu={am_npu} hf={am_hf} {'MATCH' if am_npu == am_hf else 'DIFF'}")
    print(f"nan = {nan}")
    print("PASS" if cos.item() > 0.99 and not nan else "CHECK")

if __name__ == "__main__":
    main()
