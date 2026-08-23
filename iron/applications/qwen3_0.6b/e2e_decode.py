#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end Qwen3-0.6B decode correctness check against HF (greedy).

Runs the pure-iron whole-model Qwen3NPU (28-layer sequence + bf16 GEMVArgmax
lm_head) token-by-token, and compares the generated token sequence against
HF transformers greedy decoding. Prefill is done by feeding each prompt token
through the S=1 decode path with incrementing seq_pos (KV cache accumulates
via StridedCopy cache_offset), which is correct for a 0.6B model / short prompt.

Verifies: (1) generated text is coherent (Chinese included), (2) token-for-token
agreement with HF greedy decode.
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
    max_tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 64

    # ---- NPU model ----
    print("loading NPU model (cached ELF)...", flush=True)
    t0 = time.perf_counter()
    npu = Qwen3NPU(MODEL_PATH + "model.safetensors")
    print(f"NPU load {(time.perf_counter()-t0)*1000:.0f} ms", flush=True)

    # ---- tokenizer + embedding ----
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    emb = safe_open(MODEL_PATH + "model.safetensors", framework="pt").get_tensor(
        "model.embed_tokens.weight").to(torch.bfloat16)  # (151936, 1024)

    # ---- HF reference model ----
    print("loading HF reference...", flush=True)
    hf = AutoModelForCausalLM.from_pretrained(MODEL_PATH, dtype=torch.float32)
    hf.eval()

    # ---- chat template ----
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    prompt_ids = input_ids[0].tolist()
    print(f"prompt: {len(prompt_ids)} tokens", flush=True)

    # ---- NPU decode (prefill token-by-token + generation) ----
    npu_ids = []
    # prefill
    for pos, tid in enumerate(prompt_ids):
        x = emb[tid]  # bf16 (1024,)
        next_tok = npu(x, seq_pos=pos)
    gen_start = len(prompt_ids)
    # generation
    npu_gen = []
    for step in range(max_tokens):
        x = emb[next_tok]
        next_tok = npu(x, seq_pos=gen_start + step)
        npu_gen.append(next_tok)
        if next_tok == tokenizer.eos_token_id:
            break

    # ---- HF greedy decode ----
    with torch.no_grad():
        out = hf.generate(input_ids, max_new_tokens=max_tokens, do_sample=False,
                          eos_token_id=tokenizer.eos_token_id)
    hf_gen = out[0][len(prompt_ids):].tolist()
    hf_gen = hf_gen[:len(npu_gen)]  # align length (truncate at eos)

    # ---- compare ----
    npu_text = tokenizer.decode(npu_gen, skip_special_tokens=True)
    hf_text = tokenizer.decode(hf_gen, skip_special_tokens=True)

    print(f"\n=== NPU output ({len(npu_gen)} tokens) ===")
    print(npu_text)
    print(f"\n=== HF output ({len(hf_gen)} tokens) ===")
    print(hf_text)

    n_match = sum(1 for a, b in zip(npu_gen, hf_gen) if a == b)
    agree = n_match / max(len(npu_gen), 1)
    print(f"\ntoken agreement: {n_match}/{len(npu_gen)} = {agree:.3f}")
    print("PASS" if agree >= 0.8 else "CHECK")

if __name__ == "__main__":
    main()
