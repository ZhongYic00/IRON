# Qwen3-0.6B decode on AMD Ryzen AI NPU (pure IRON)

Whole-model decode of [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) on
NPU2 (AIE2P, Strix Point), compiled as a **single OperatorSequence ELF**: all 28
transformer layers + final RMSNorm + lm_head GEMV run in one NPU dispatch per
token. Attention is the fused [`DecodeAttention`](../../operators/decode_attn/)
operator (scores GEMV + online-softmax streaming + context GEMV in one kernel,
runtime `seq_pos`, KV cache in DRAM, supported `S_KV` up to 4096).

Qwen3-specific handling on top of the
[Llama 3.2 1B](../llama_3.2_1b/) reference app: QK-norm (per-head RMSNorm on
Q/K, `QKNorm` operator) and GQA (16 Q heads / 8 KV heads, folded into the fused
attention kernel — no explicit repeat needed).

## Validated results (Ryzen AI 9 HX 370, Strix Point)

| Check | Command | Expected |
|---|---|---|
| Prefill logits vs HuggingFace (f32) | `python3 check_cosine.py` | cosine ≈ 0.9985, argmax MATCH, `PASS` |
| 64-token generation vs HF greedy | `python3 e2e_decode.py` | token agreement ≥ 0.8, `PASS` |
| Decode latency | `python3 bench_tpot.py 50` | mean TPOT ≈ 68 ms (≈14.5 tok/s) |
| Streaming chat | `python3 run_dialogue.py` | coherent Chinese/English text |

## Requirements

**Hardware / driver stack**

- NPU2 (AIE2P) SoC — e.g. Ryzen AI 300 "Strix" / "Krackan"
- XRT ≥ 2.26 with the NPU plugin (`source /opt/xilinx/xrt/setup.sh`)
- `amdxdna` kernel driver 2.26 (DKMS build from xdna-driver; the in-tree
  driver is too old for full-ELF dispatch)
- `memlock unlimited` for the user running the app

**Python packages** (from the repo root)

```bash
pip install -r requirements.txt          # toolchain + iron (see note below)
pip install -r iron/applications/qwen3_0.6b/requirements_qwen3.txt
```

> Toolchain note: this branch compiles with the **mlir-aie 1.4.0** aiecc
> selectors (`--get-full-elf` / `--get-scratchpad-parameters`), so the
> `requirements.txt` pins `mlir_aie_no_rtti==1.4.0` +
> `llvm-aie==21.0.0.2026080601` — the exact combination validated on NPU2.
> Both are permanent GitHub release assets. If pip cannot resolve a
> transitive dependency of the no-rtti wheel, it is not needed by iron —
> install that wheel with `--no-deps`.

**Model weights** (≈1.2 GB)

```bash
huggingface-cli download Qwen/Qwen3-0.6B --local-dir /srv/qwen3-0.6b
```

Any directory works; point the scripts at it with `--model-path`
(`run_dialogue*.py`) or `QWEN3_MODEL_DIR` (everything else). Default:
`/srv/qwen3-0.6b`.

## Quick start (run from this directory)

```bash
export QWEN3_MODEL_DIR=/srv/qwen3-0.6b   # or pass --model-path

# 1. Smoke test — compiles the whole-model ELF (first run only, several
#    minutes), then runs one decode step:
python3 qwen3_npu.py
# -> "argmax token_id = ..." (any id is fine; it must not be NaN)

# 2. Correctness: NPU prefill vs HF f32 prefill, compare final logits
python3 check_cosine.py "你好，请介绍一下你自己"
# -> cosine = 0.998xxx   argmax: npu=... hf=... MATCH   PASS

# 3. End-to-end: 64 generated tokens vs HF greedy
python3 e2e_decode.py "你好，请介绍一下你自己" 64
# -> token agreement: N/64 >= 0.8   PASS

# 4. Interactive streaming chat
python3 run_dialogue.py --prompt "你好" --max-tokens 256

# 5. TPOT benchmark (50 steps at steady state)
python3 bench_tpot.py 50
```

## Scripts

| File | Purpose |
|---|---|
| `qwen3_npu.py` | `Qwen3NPU` — whole-model single OperatorSequence (primary model class) |
| `qwen3_npu_tiered.py` | `Qwen3NPUTiered` — 256/512 seq-length tiered variant (saves ~9 ms/token on short sequences) |
| `run_dialogue.py` / `run_dialogue_tiered.py` | streaming chat (argparse: `--prompt --max-tokens --system --model-path`) |
| `check_cosine.py` | prefill logits cosine + argmax vs HF |
| `e2e_decode.py` | N-token generation vs HF greedy decode |
| `bench_tpot.py` | decode latency (TPOT) benchmark |

## Notes

- First run compiles ~40 operators into one ELF; artifacts are cached under
  `build/` (repo-root default, configurable via `AIEContext(build_dir=...)`).
  `rm -rf build/` whenever kernel or operator code changes — same-named
  operators with different parameters otherwise reuse stale artifacts.
- Prefill is token-by-token through the S=1 decode path (KV cache accumulates
  via `StridedCopy`); for a 0.6B model and typical prompts this is fine and
  keeps the model to a single compiled variant.
- The tiered variant switches from the 256-wide to the 512-wide KV tier once
  `seq_pos` reaches 256 (one-time K/V repack, ~28 MB).
