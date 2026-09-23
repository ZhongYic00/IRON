# Qwen3-0.6B decode on AMD Ryzen AI NPU (pure IRON, int8)

Whole-model decode of [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) on
NPU2 (AIE2P, Strix Point), compiled as a **single OperatorSequence ELF**: all 28
transformer layers + final RMSNorm + lm_head GEMV run in one NPU dispatch per
token.

The default arm set keeps weights int8 (W8A16) in DRAM with per-(row,
group-of-128) bf16 scales — the group scale is folded into the activations
(premul) so the GEMV hot loop reads int8 bytes and accumulates in f32 — and
fuses the per-layer decode steps into three operators:

- fused QKV projection head ([`QKVHeadDataParallelOurs`](../../operators/qkv_head_dp/)):
  weighted input RMSNorm + the concatenated Q/K/V GEMV + per-head QK-norm +
  RoPE in one design, with the K|V slice written straight into the layer's
  interleaved KV cache;
- fused decode attention ([`DecodeAttention`](../../operators/decode_attn/)):
  scores GEMV + online-softmax streaming + context GEMV in one kernel (runtime
  `seq_pos`, KV cache in DRAM, supported `S_KV` up to 4096, GQA folded into the
  kernel — 16 Q heads / 8 KV heads, no explicit repeat);
- fused MLP block ([`SwiGLUMLPDataParallelOurs`](../../operators/swiglu_mlp_dp/)):
  post-attention RMSNorm + gate/up GEMV + SiLU·mul + down GEMV + residual add
  in one design, with the attention output projection folded in (`fuse_o`).

lm_head is an RTN int8 [`GEMVInt8`](../../operators/gemv_int8/) over the tied
embedding table.  Qwen3's QK-norm is also available standalone as the
[`QKNorm`](../../operators/qk_norm/) operator for the dormant fine-grained arm
(`QKV_HEAD_DP=0`).

## Validated results (Ryzen AI 9 H 365, Strix Point)

| Check | Command | Expected |
|---|---|---|
| Prefill logits vs HuggingFace (f32) | `python3 check_cosine_int8.py` | cosine ≈ 0.9995, argmax MATCH, `PASS` |
| Decode latency (default arm set) | see the flag block in `qwen3_npu_int8.py` | ≈33 ms/token (≈30 tok/s) |
## Optimization history

Every number below is a measured p50 from the same-shell A/B rounds archived
with each change (dates are 2026).  The whole 0.6B path went from the bf16
fine-grained chain to the shipped fused config:

```text
68.7 ms ──● bf16 fine-grained chain (decode_attn lands)              8/26
          │
70.67 ────● int8 fine-grained (GEMVInt8: premul dequant, VEC128,      8/29–9/10
          │         mode14 — standalone 16.7 → 39.7 GB/s)
60.0 ─────● + fused MLP (swiglu_mlp_dp, bf16 shell)
59.0 ─────●   int8 MLP shell
55.3 ─────● + fused QKV head (qkv_head_dp, bf16)
52.4 ─────●   qkv int8
50.4 ─────● + int8 lm_head (RTN from the tied embed table)
42.9 ─────● independent-buffer arena + weight-fifo depth (3 cuts)     9/15
37.2 ─────● fuse_o: Wo folded into the fused MLP design              9/15
33.1 ─────● signed int8 domain (byte-identical outputs, −11%)         9/17
32.0–32.4●  this tree: upstream toolchain re-measured (9/23–24)
```

Byte-identical logits were verified at every rung above (`check_cosine_int8`
cosine 0.9994–0.9996, argmax MATCH throughout).

## Performance breakdown and roofline

Measured machine constants (roofline ledger): DDR pure read **49.7 GB/s**,
NPU large-copy **53–58 GB/s**; the big streams' realized *marginal* rate is
**38.6–40.4 GB/s** (78–81% of pure read, flow-length independent).

Per-token DRAM accounting (0.6B, S_kv = 512):

| stage | DRAM read / token | roofline @ 49.7 GB/s | measured |
|---|---|---|---|
| fused MLP ×28 | ~11.7 MB/layer → 327 MB | 6.6 ms | ~13–15 ms (40.5 GB/s marginal ≈ 80% of pure read) |
| fused QKV head ×28 | ~4.3 MB/layer → 119 MB | 2.4 ms | ~9.5–10.5 ms (27 GB/s — shell-bound: 13 misc fills + 48 per-head drains/layer) |
| fused attention ×28 | KV 2.1 MB/layer → 58.7 MB | 1.2 ms | ~5 ms (**92% fixed cost**, 8% bytes — KV blocking measured as a dead end) |
| lm_head (int8) | 158 MB | 3.2 ms | ~4.3 ms (36.5 GB/s via its own BO; 53–54 GB/s standalone peak) |
| dispatch + per-entry | — | 0.22 ms | one 0.22 ms dispatch + ~10 µs/entry in-chain |
| **total** | **~610 MB + KV** | **13.4 ms** | **32.0–32.4 ms (≈41% of the pure-read roofline)** |

Two structural conclusions from the leave-one-out attribution:

- the largest term is the op streams themselves, and they already run at the
  measured marginal rate (~40 GB/s = 80% of pure read) — the remaining gap to
  the 19–22 ms "aligned-with-FLM-structure" theoretical limit (46–53 tok/s)
  is the qkv head's shell overhead and per-entry floors;
- the byte-halving escape hatch is int4 (303 MB → 6.1 ms floor), a separate
  branch — the shipped chain is int8 by design.

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

> Toolchain note: this application compiles with the repo-level
> `requirements.txt`, which pins the validated mlir-aie/llvm-aie pair
> (`mlir_aie==1.4.4.dev18`, `llvm-aie==22.0.0.2026091701` at time of writing).
> The packages are not on official PyPI and resolve through the
> release-asset `--find-links` entries already listed there.

**Model weights**

```bash
huggingface-cli download Qwen/Qwen3-0.6B-GPTQ-Int8 --local-dir /srv/qwen3-0.6b-gptq-int8
huggingface-cli download Qwen/Qwen3-0.6B --local-dir /srv/qwen3-0.6b            # f32 logit reference
```

`check_cosine_int8.py` reads the GPTQ-Int8 weights from `$QWEN3_INT8_MODEL_DIR`
(default `/srv/qwen3-0.6b-gptq-int8`) and the bf16 reference of the SAME model
from `$QWEN3_MODEL_DIR` (default `/srv/qwen3-0.6b`).

## Quick start (run from this directory)

```bash
# 1. Correctness gate: NPU prefill vs HF f32 prefill, compare final logits.
#    Compiles the whole-model ELF on first run (several minutes).
python3 check_cosine_int8.py "你好，请介绍一下你自己"
# -> cosine = 0.9995xx   argmax: npu=... hf=... MATCH   PASS

# 2. One decode step from Python (the library entry point):
#    from qwen3_npu_int8 import Qwen3NPU
#    model = Qwen3NPU(".../model.safetensors"); token = model(x, seq_pos=0)
#    (`python3 qwen3_npu_int8.py` runs the same thing as a smoke test.)
```

## Arm set

The fused arms are env-gated at the top of `qwen3_npu_int8.py` (`MLP_XDNA`,
`MLP_FUSE_O`, `QKV_HEAD_DP`, `LMHEAD_INT8`, `WEIGHTS_INDEP`, `KV_INDEP`,
`INT8_SIGNED`, ...); each defaults to its measured-best value and `=0` restores
the fine-grained int8 runlist.  `INT8_06B_BUILD_DIR` moves the compile cache —
**each distinct arm set needs its own build dir**, because the fused-MLIR/ELF
cache does not key on the runlist and two arms sharing a directory silently
run each other's binary.

## Notes

- First run compiles the whole-model ELF (several minutes); artifacts are
  cached under `<this app dir>/build_int8` (configurable via
  `INT8_06B_BUILD_DIR`).  `rm -rf` that directory whenever kernel or operator
  code changes — same-named operators with different parameters otherwise
  reuse stale artifacts, which surfaces as undefined symbols at link time.
- Prefill is token-by-token through the S=1 decode path (KV cache accumulates
  via the fused head's cache writes); for a 0.6B model and typical prompts
  this is fine and keeps the model to a single compiled variant.
