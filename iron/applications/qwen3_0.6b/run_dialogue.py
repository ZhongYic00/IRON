#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Stream a chat with Qwen3-0.6B on NPU using the fused-decode-attn model.

Pure-iron whole-model Qwen3NPU (qwen3_npu.py, decode_attn fused). Prefill +
decode are both token-by-token on the NPU with a persistent interleaved KV
cache; decode streams to stdout and prints TPOT/TPS at the end.

Usage:
    python3 run_dialogue.py --prompt "你好，请介绍一下MoE" --max-tokens 1024
"""

import argparse
import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen3_npu import Qwen3NPU, emb_dim

from safetensors import safe_open
from transformers import AutoTokenizer

DEFAULT_MODEL_DIR = os.environ.get("QWEN3_MODEL_DIR", "/srv/qwen3-0.6b")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", type=str, default="你好，请介绍一下MoE")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--system", type=str, default=None)
    ap.add_argument("--model-path", type=str, default=DEFAULT_MODEL_DIR,
                    help="dir with model.safetensors + tokenizer files")
    args = ap.parse_args()

    print("loading model...", flush=True)
    t0 = time.perf_counter()
    model = Qwen3NPU(args.model_path + "/model.safetensors")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    emb = safe_open(args.model_path + "/model.safetensors", framework="pt").get_tensor(
        "model.embed_tokens.weight").to(torch.bfloat16)
    print(f"model loaded in {(time.perf_counter() - t0) * 1000:.0f} ms", flush=True)

    messages = []
    if args.system:
        messages.append({"role": "system", "content": args.system})
    messages.append({"role": "user", "content": args.prompt})
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    prompt_ids = input_ids[0].tolist()
    print(f"prompt: {len(prompt_ids)} tokens", flush=True)

    eos = tokenizer.eos_token_id

    # ---- prefill (token by token) ----
    for pos, tid in enumerate(prompt_ids):
        next_tok = model(emb[tid], seq_pos=pos)

    # ---- decode loop (stream) ----
    print("\n=== generating ===\n", flush=True)
    decode_times = []
    gen_start = len(prompt_ids)
    gen = []
    for step in range(args.max_tokens):
        t0 = time.perf_counter()
        next_tok = model(emb[next_tok], seq_pos=gen_start + step)
        decode_times.append((time.perf_counter() - t0) * 1000)
        gen.append(next_tok)
        piece = tokenizer.decode([next_tok], skip_special_tokens=True)
        print(piece, end="", flush=True)
        if next_tok == eos:
            break
    print("\n", flush=True)

    # ---- summary ----
    if len(decode_times) > 2:
        steady = decode_times[2:]
    else:
        steady = decode_times
    tpot = sum(steady) / max(len(steady), 1)
    n = len(gen)
    total_s = sum(decode_times) / 1000
    tps = n / total_s if total_s > 0 else 0
    print(f"\n{'=' * 50}")
    print(f"tokens generated: {n}")
    print(f"steady TPOT: {tpot:.1f} ms")
    print(f"TPS: {tps:.1f} tokens/s")
    print(f"{'=' * 50}", flush=True)


if __name__ == "__main__":
    main()
