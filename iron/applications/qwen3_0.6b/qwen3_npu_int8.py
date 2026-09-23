#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3-0.6B int8 (W8A16 in-kernel dequant) decode on NPU — whole-model single OperatorSequence.

Faithful port of llama_npu.py's whole-model single-sequence structure.  The
per-layer decode steps run as three fused operators (see below); Qwen3's
QK-norm and GQA are folded into the fused operators rather than chained as
standalone entries.

Model (Qwen3-0.6B):
  emb_dim=1024, hidden_dim=3072, n_heads=16, n_kv_heads=8 (GQA=2),
  head_dim=128, vocab=151936, n_layers=28, rope_theta=1e6, eps=1e-6.

Per layer, one entry each:
  QKVHeadDataParallelOurs  weighted RMSNorm(x) + the concatenated Q/K/V int8
                           GEMV + per-head QK-norm + RoPE, with the K|V slice
                           written straight into the layer's interleaved KV
                           cache (one array reconfigure per layer);
  DecodeAttention          scores GEMV + online softmax + context GEMV in one
                           kernel, GQA folded in, runtime seq_pos;
  SwiGLUMLPDataParallelOurs  post-attention RMSNorm + gate/up int8 GEMV +
                           SiLU*mul + down GEMV + residual add in one design,
                           with the attention output projection folded in
                           (fuse_o).
After 28 layers: final RMSNorm -> int8 GEMV(lm_head, RTN from the tied
embed_tokens) -> logits.

Weights: GPTQ int8 payload + per-(row, group-of-128) bf16 scales, read as a
SIGNED domain (mv_int8_signed.cc) with the group scale premultiplied into
the activations; all wires and kernel sources below are pinned to that pair.
"""

import torch
import os
import numpy as np
import ml_dtypes
from pathlib import Path
from safetensors import safe_open

from aie.iron.device import NPU2
import aie.utils as aie_utils
aie_utils.set_current_device(NPU2())

from iron.common.context import AIEContext
from iron.common.sequence import OperatorSequence
from iron.operators import RMSNorm, DecodeAttention
from iron.operators.gemv_int8.op import GEMVInt8
from iron.operators.swiglu_mlp_dp.op_ours import SwiGLUMLPDataParallelOurs
from iron.operators.qkv_head_dp.op_ours import QKVHeadDataParallelOurs

# ---------------- chain configuration ----------------
# The chain runs the measured-best fused arm set (six-arm p50 ladder: baseline
# 70.67 / bf16 shell 60.04 / int8 shell 59.00 / +qkv bf16 55.29 / +qkv int8
# 52.38 / +lm_head int8 50.42 ms/token; the shipped config with the fused MLP
# and signed domain measures 32-33 ms/token p50).
# int8 lm_head (RTN-quantized from the tied bf16 embed_tokens; this GPTQ checkpoint has no
# lm_head wire).  Halves the chain's largest single DRAM stream.
# Weight-tile rows for the int8 lm_head: a 0.6B wire row is 1024 + (1024/128)*2 = 1040 B and
# 4 * 1040 = 4160 is a 64 B multiple (the wire's wide-vector alignment rule) ⇒ tsi=4.
LMHEAD_TSI = int(os.environ.get("LMHEAD_TSI", "4"))
# lm_head's output tile rows (M per tile). 16 is the largest value that keeps the vocab tile grid
# legal at 151936/8, and it is the shipped setting; exposed for the tile-size A/B (the 4B lesson was
# that tile SIZE can matter more than depth, and this is the chain's largest single stream).
LMHEAD_TSO = int(os.environ.get("LMHEAD_TSO", "16"))
# GEMVInt8's weight-fifo depth (see op.py): 2 = double buffering (the historical value). The
# chain's largest stream is lm_head's 158 MB/token at 36.5 GB/s against a 49.7 GB/s pure-read
# bound, so this is the knob that could close that gap -- the same one that gave swiglu_mlp_dp
# 29.2 -> 38 GB/s.
LMHEAD_WDEPTH = int(os.environ.get("LMHEAD_WDEPTH", "2"))
# E2b: a weight read through
# the shared scratch arena's sub-buffer tap runs at 25-27 GB/s, while the SAME op and bytes with
# the weight in its own BO run at 36-37 GB/s (probe_ops06b_lmhead_mech.py arm F: 6.246 -> 4.330 ms
# for lm_head; F-A = +0.07 ms vs the standalone path, i.e. the arena tax is the whole gap).  This
# inlines the tail weight as an `independent_buffer_args` BO to recover it.
LMHEAD_INDEP = True
# Same mechanism, applied to the per-layer weight arrays (Wo/Wg/Wu/Wd) and the interleaved KV
# caches — measured -3.5..-4.0 ms (weights) and -1.9% (KV), gate cos unchanged.
WEIGHTS_INDEP = True
KV_INDEP = True
# Fused decode-QKV head (`iron/operators/qkv_head_dp/`, OUR-kv-layout int8 arm): ONE design
# replaces [rms1, gemv_qkv, qk_norm, rope_q, rope_k, sc_k, sc_v] — weighted RMSNorm(norm1) +
# the concatenated QKV GEMV + per-head qk-norm + RoPE, with the K|V slice drained straight
# into the chain's own interleaved per-token cache at `k_cache_offset`.  So the QKV group
# costs ONE array reconfigure per layer instead of seven, with no x_norm/qkv_out DDR round
# trips between them, and decode_attn's tap is unchanged (the fused head writes that exact
# layout).  Op defaults carry the 0.6B behaviour (VEC=64, misc_obj_elems=HD, no runtime tile
# loop), and QKV_DP_* stay overridable for the 4B-style knobs.
# Merge the fused head's per-head output drains into one per kind per core (48 -> 10 tasks/layer
# at 4B shapes, 32 -> 8 at 0.6B).  Opt-in; the off path is unchanged.
QKV_DP_MERGE = os.environ.get("QKV_DP_MERGE", "0") == "1"
QKV_DP_TSI = int(os.environ.get("QKV_DP_TSI", "4"))
# qkv's tile loop: the static form materializes every per-head drain as its own task (larger
# .text, measured 16896 B before the 9/13 trim), the runtime form emits the loop with a runtime
# bound (7472 B).
QKV_RUNTIME_TILE = os.environ.get("QKV_RUNTIME_TILE", "1") == "1"

# ---------------- model config ----------------
emb_dim = 1024
hidden_dim = 3072
n_heads = 16
n_kv_heads = 8
head_dim = 128
q_dim = n_heads * head_dim        # 2048
kv_dim = n_kv_heads * head_dim    # 1024
qkv_dim = q_dim + 2 * kv_dim      # 4096
vocab_size = 151936
n_layers = 28
max_seq_len = 512
# The int8 payload is read as a SIGNED domain (mv_int8_signed.cc over the same
# bytes the biased form would read): the hot loop is 2 vector ops per block
# instead of 3, measured -11% ms/token with byte-identical outputs (both
# domains feed `mac` the same integer q — exact in bf16's 8-bit significand).
# Kernel source and packing domain move together, per wire, never one
# without the other.
SIGNED_GEMV_KERNEL = "mv_int8_signed.cc"
SIGNED_ACC_KERNEL = "mv_int8_acc_signed.cc"
# decode_attn's KV block size along S. Pure READ-side blocking: the cache itself is per-token
# interleaved [K_t | V_t] head-major (qkv_head_dp's drain tap defines that layout), so changing
# this does not touch the layout contract — hence env-overridable for the S/block A/B.
# 32 (was 64): A/B 42.12 -> 41.67/41.58 ms p50, gate cos 0.999373 unchanged. Larger values do not
# even compile (128/256 exceed the design's L1 budget); 32 evidently halves the L1 tile pressure.
block_kv = int(os.environ.get("BLOCK_KV", "32"))
eps = 1e-6
rope_theta = 1e6


class Qwen3NPU:
    @staticmethod
    def context_root():
        """Isolated build dir (default: <this app dir>/build) — the fused-MLIR/ELF
        cache does not key on the runlist or the arm flags, so two arms (or two
        sessions) sharing one directory silently run each other's binary.

        INT8_06B_BUILD_DIR overrides it; give each ARM its own cache when A/B-ing.
        """
        root = Path(os.environ.get(
            "INT8_06B_BUILD_DIR",
            str(Path(__file__).resolve().parent / "build_int8")))
        root.mkdir(parents=True, exist_ok=True)
        return root

    def __init__(self, weights_path):
        self.context = AIEContext(build_dir=self.context_root())
        self.context.build_dir.mkdir(parents=True, exist_ok=True)
        self.weights = safe_open(weights_path, framework="pt")

        # ---- weights: GPTQ int8 (qweight int32-packed + fp16 scales), the
        # bf16 layers (norms / qk gamma / embed) ride alongside in the same file.
        self.weight_cache = {}
        for i in range(n_layers):
            L = f"model.layers.{i}"
            F = self.weights.get_tensor
            self.weight_cache[i] = {
                "input_norm": F(f"{L}.input_layernorm.weight").to(torch.bfloat16),
                "q_qw": F(f"{L}.self_attn.q_proj.qweight"),
                "q_sc": F(f"{L}.self_attn.q_proj.scales"),
                "k_qw": F(f"{L}.self_attn.k_proj.qweight"),
                "k_sc": F(f"{L}.self_attn.k_proj.scales"),
                "v_qw": F(f"{L}.self_attn.v_proj.qweight"),
                "v_scl": F(f"{L}.self_attn.v_proj.scales"),
                "o": F(f"{L}.self_attn.o_proj.qweight"),
                "o_scl": F(f"{L}.self_attn.o_proj.scales"),
                "q_norm": F(f"{L}.self_attn.q_norm.weight").to(torch.bfloat16),
                "k_norm": F(f"{L}.self_attn.k_norm.weight").to(torch.bfloat16),
                "norm2": F(f"{L}.post_attention_layernorm.weight").to(torch.bfloat16),
                "gate": F(f"{L}.mlp.gate_proj.qweight"),
                "gate_scl": F(f"{L}.mlp.gate_proj.scales"),
                "up": F(f"{L}.mlp.up_proj.qweight"),
                "up_scl": F(f"{L}.mlp.up_proj.scales"),
                "down": F(f"{L}.mlp.down_proj.qweight"),
                "down_scl": F(f"{L}.mlp.down_proj.scales"),
            }
        self.final_norm = self.weights.get_tensor("model.norm.weight").to(torch.bfloat16)
        # lm_head / embed_tokens stay bf16 (GPTQ lm_head: false)
        self.out_head = self.weights.get_tensor("model.embed_tokens.weight").to(torch.bfloat16)

        self._build_operators()
        self._build_sequence()
        self._load_weights()

    # ---------------- operators ----------------
    def _build_operators(self):
        c = self.context
        # Final RMSNorm (the only standalone norm entry; the per-layer norms are
        # folded into the fused QKV head and the fused MLP).
        self.rms = RMSNorm(size=emb_dim, num_aie_columns=1, num_channels=1,
                           tile_size=emb_dim, weighted=True, epsilon=eps, context=c)
        # Fused decode attention (single query, M=1): scores-GEMV + softmax +
        # context-GEMV in one kernel per (head, column). Replaces the prefill-style
        # gemv_scores + attn_scale + softmax + transpose_v + gemv_context chain,
        # and subsumes the GQA Repeat (one KV head per column).
        self.decode_attn = DecodeAttention(
            num_heads=n_heads, num_kv_heads=n_kv_heads, head_dim=head_dim,
            seq_len_kv=max_seq_len, num_aie_columns=8, block_kv=block_kv,
            use_runtime_seq_len=True, context=c)
        # Whole MLP block in ONE design: x1 = cur + a, hf = RMSNorm_w(x1, n_pf),
        # nxt = x1 + Wo@cx + Wd@(silu(Wg@hf) * (Wu@hf)) with fuse_o (the o
        # projection computed inside the design, so there is no standalone
        # gemv_output entry and no attn_output round trip).  int8_ours wire: OUR
        # mv_int8.cc (mode14/VEC128) over OUR [tile_m*K u8 | tile_m*(K/128) bf16 s]
        # tile stream, TSI_GU=12 for gate/up (12 rows x 1040 B = 12480 B, 64
        # B-aligned) and TSI_D=4 for down (4 x 3120 B = 12480 B — the shared-tile
        # identity the design asserts).
        self.mlp_xdna = SwiGLUMLPDataParallelOurs(
            D=emb_dim, FF=hidden_dim, num_aie_columns=8, group_size=128,
            epsilon=eps, fuse_o=True, QD=q_dim,
            # The MLP's int8 matvec translation units (gu/down, plus fuse_o's own
            # o-projection instantiation) read the signed domain, paired with the
            # signed wires packed in `_load_weights`.  acc_kernel_source covers
            # the chunked-down arm (not taken at 0.6B shapes, where FF % D == 0) —
            # set it anyway so the op is never half-switched.
            kernel_source=SIGNED_GEMV_KERNEL,
            acc_kernel_source=SIGNED_ACC_KERNEL,
            context=c)
        # Fused QKV head: weighted RMSNorm(x, W_norm1) + the concatenated Wqkv
        # GEMV + per-head qk-norm + RoPE(q,k), with the K|V slice drained straight
        # into the chain's own interleaved per-token cache at `k_cache_offset`.
        # So the QKV group costs ONE array reconfigure per layer instead of
        # seven, with no x_norm/qkv_out DDR round trips between them, and
        # decode_attn's tap is unchanged (the fused head writes that exact
        # layout).
        self.qkv_dp = QKVHeadDataParallelOurs(
            D=emb_dim, HD=head_dim, Hq=n_heads, Hkv=n_kv_heads,
            max_seq=max_seq_len, num_aie_columns=8,
            tile_size_input=QKV_DP_TSI, epsilon=eps,
            merge_drains=QKV_DP_MERGE,
            runtime_tile_loop=QKV_RUNTIME_TILE,
            # The fused head's own Wqkv matvec object is built from this source
            # (op_ours.py's `gemv_{D}k_128vs_ours.o`), paired with the signed
            # W_qkv wire packed in `_load_weights`.
            kernel_source=SIGNED_GEMV_KERNEL,
            kv_offset_parameter="k_cache_offset", context=c)
        # lm_head as the chain's own int8 GEMV family (mode14/VEC128): this is
        # the chain's single largest DRAM stream, and it is bytes-bound.
        # tile_size_output must divide M/num_aie_columns = 151936/8 = 18992.
        # 18992 = 16 * 1187 (1187 prime), so the only viable m_input=4-compatible
        # tile outputs are 4, 8, 16. llama's 32 works only because its vocab
        # 128256/8=16032 is 32-divisible; Qwen3 vocab is not. 16 is the largest
        # valid tile and keeps L1 pressure low.
        self.lmhead = GEMVInt8(M=vocab_size, K=emb_dim, num_aie_columns=8,
                               tile_size_input=LMHEAD_TSI, tile_size_output=LMHEAD_TSO,
                               weight_depth=LMHEAD_WDEPTH,
                               kernel_source=SIGNED_GEMV_KERNEL,
                               context=c)

    # ---------------- sequence ----------------
    def _build_sequence(self):
        runlist = []
        for i in range(n_layers):
            # ONE entry for the whole QKV group: weighted RMSNorm(x, W_norm1) +
            # the concatenated Wqkv GEMV + per-head qk-norm + RoPE(q,k), with the
            # K|V slice written straight into the interleaved per-token cache
            # (keyed by the same `k_cache_offset`).  `rope_angles` is the op's
            # `ang` object — the same 128-element interleaved [cos,sin] row the
            # standalone RoPE ops read (see __call__).
            runlist.extend([
                (self.qkv_dp, "x", f"W_norm1_{i}", f"W_qkv_{i}",
                 f"W_qn_{i}", f"W_kn_{i}", "rope_angles", "rope_q",
                 f"kv_cache_{i}"),
                (self.decode_attn, "rope_q", f"kv_cache_{i}", "context"),
            ])
            # The fused MLP computes x1 = cur + a internally, so passing the
            # attention residual as `a` DROPS the standalone residual_add here
            # (keeping it would double-add attn_output); nxt lands back in "x".
            # fuse_o: `a = Wo @ cx` is computed by this design from the attention
            # context directly (cx = decode_attn's output), so there is no
            # gemv_output entry and no attn_output buffer; `a_scratch` is the
            # design's a-slice all-gather round-trip (see op_ours).
            runlist.append((self.mlp_xdna, "x", "context", f"W_norm2_{i}",
                            f"Wo_{i}", f"Wg_{i}", f"Wu_{i}", f"Wd_{i}",
                            "gh_scratch", "a_scratch", "x"))
        # Final RMSNorm + lm_head GEMV both live in the SAME sequence (llama-style),
        # so decode is a single dispatch producing "logits". The final RMSNorm emits
        # an independent buffer "x_final" (avoids in-out x colliding in
        # subbuffer_layout), which the lm_head GEMV reads.
        runlist += [
            (self.rms, "x", "W_final_norm", "x_final"),
            (self.lmhead, "W_out_head", "x_final", "logits"),
        ]

        # interleaved KV cache: K and V share one buffer per layer
        # ([K_b0|V_b0|K_b1|V_b1|...]), sized 2x the old separated k/v cache.
        kv_cache_size = n_kv_heads * 2 * max_seq_len * head_dim * 2
        # The fused MLP's gh all-gather round-trip buffer (ff_pad bf16 elements;
        # ff_pad == FF at these shapes) — an explicit entry so the arena length is
        # pinned independently of the op's own arg spec — plus fuse_o's a-slice
        # round-trip (D bf16 elements).
        mlp_bufs = {"gh_scratch": self.mlp_xdna._gh_units() * 2,
                    "a_scratch": emb_dim * 2}
        # NOTE: the *other* weights still live in the scratch buffer: fused-dispatch's main
        # sequence passes only the three global BOs (input/output/scratch) to each operator, so a
        # weight in an independent set_arg slot was historically invisible to its operator.
        # `W_out_head` is the one exception (LMHEAD_INDEP, E2b §1.1c): it is ~158 MB and the arena
        # sub-buffer tap costs it ~30% of its stream rate, which the independent BO recovers.
        # WEIGHTS_INDEP extends the same move to the per-layer weight arrays (see its note above);
        # the list is derived from the runlist so it tracks whichever arm built it.
        indep = ["W_out_head"]
        indep += list(dict.fromkeys(
            arg for entry in runlist for arg in entry[1:]
            if arg.startswith(("W_qkv_", "W_o_", "Wo_", "Wg_", "Wu_", "Wd_"))))
        indep += list(dict.fromkeys(
            arg for entry in runlist for arg in entry[1:]
            if arg.startswith("kv_cache_")))
        self.seq = OperatorSequence(
            "qwen3_0_6b_decode",
            runlist,
            input_args=["x", "rope_angles"],
            output_args=["logits"],
            buffer_sizes={
                **{f"kv_cache_{i}": kv_cache_size for i in range(n_layers)},
                "logits": vocab_size * 2,
                "rope_angles": head_dim * 2,
                **mlp_bufs,
            },
            independent_buffer_args=indep,
            context=self.context,
        )
        self.seq.compile()
        self.fc = self.seq.get_callable()

        # WORKAROUND: rebuild the sequence with a distinct name (handmade
        # style).  The original-name sequence produces NaN through the whole
        # chain even with identical runlist/params — root cause TBD in the
        # fused-dispatch name/ELF path.
        # WORKAROUND part 2: build in an INDEPENDENT context (the first
        # sequence's compile poisons shared state in the same context).
        self.context2 = AIEContext(build_dir=self.context_root() / "seq2")
        self.seq = OperatorSequence(
            "qwen3_0_6b_decode_v2",
            runlist,
            input_args=["x", "rope_angles"],
            output_args=["logits"],
            buffer_sizes={
                **{f"kv_cache_{i}": kv_cache_size for i in range(n_layers)},
                "logits": vocab_size * 2,
                "rope_angles": head_dim * 2,
                **mlp_bufs,
            },
            independent_buffer_args=indep,
            context=self.context2,
        )
        self.seq.compile()
        self.fc = self.seq.get_callable()
        print(f"compiled OK; buffer_sizes: {self.seq.buffer_sizes}")
        print(f"n buffers: {len(self.seq.subbuffer_layout)}")

    # ---------------- weight load ----------------
    def _gptq_rows(self, qweight, scales, signed=False):
        """(K/4,M) u32 + (G,M) fp16 -> ((M,K) uint8 payload bytes, (M,G) bf16).

        This checkpoint stores GPTQ sym u8 with the true weight = (u8 - 128) * scale
        (zero-point 128, NOT 127 — verified against the bf16 reference in
        probe_int8_signed_domain_06b.py: residual 8.3e-3 vs 2.8e-2 for 127), so the
        raw u8 IS q + 128 and the biased kernel's `add(-128)` recovers q.

        `signed=True` writes the SAME integer q as two's complement instead —
        byte = (u8 - 128) mod 256 — which `to_float<bfloat16>(int8)` reads directly,
        dropping the per-weight `add(-128)` (mv_int8_signed.cc).  Both domains feed
        `mac` the same integer vector, so this is exact arithmetic, not a
        re-quantization; the returned tensor stays uint8 BYTES in both domains (only
        the values move), which is why `_pack_int8` needs no domain argument.
        """
        qw = qweight.numpy().astype(np.uint32)
        sc = scales.numpy().astype(np.float32)
        K4, M = qw.shape
        K = K4 * 4
        u = ((qw[:, :, None] >> np.array([0, 8, 16, 24], dtype=np.uint32))
             & np.uint32(0xFF)).astype(np.int16)        # u8_gptq = q + 128
        if signed:
            # q as two's complement; q in [-128,127] is exact in int8 and in bf16
            w = ((u - 128).astype(np.int8).view(np.uint8)
                 .transpose(0, 2, 1).reshape(K, M))
        else:
            w = u.transpose(0, 2, 1).reshape(K, M).astype(np.uint8)  # (K, M) biased u8
        w_mk = torch.from_numpy(np.ascontiguousarray(w.T))  # (M, K)
        sc_bf = torch.from_numpy(np.ascontiguousarray(sc)).to(torch.bfloat16)
        return w_mk, sc_bf.T.contiguous()               # (M, G)

    def _pack_int8(self, w_q, sc_mg, M, K, cols, m_input):
        """Per (M//cols)/m_input sub-tile: [m*K payload bytes | m*G bf16 scale] bytes,
        viewed as the bf16 element stream the design's L3 tap expects.

        Domain-agnostic by construction: `w_q` carries uint8 BYTES in both payload
        domains (the domain changes the VALUES, never the container — see `_gptq_rows`),
        so this only concatenates.  It has no padding to flip either: every tile is
        exactly m_input*(K + (K/128)*2) bytes and each caller's M // cols divides by
        m_input (asserted where the design asserts the shared-tile identity).
        """
        n_groups = K // 128  # GROUP_SIZE (matches mv_int8.cc -DGROUP_SIZE=128)
        # Byte view on purpose: an int8 payload tensor would be promoted by the
        # concatenate below and re-wrapped by `astype(np.uint8)` (value wrap, not a
        # reinterpret); this makes both domains take the identical byte path.
        w_np = np.ascontiguousarray(w_q.numpy()).view(np.uint8)
        s_bytes = sc_mg.contiguous().view(torch.uint8).numpy().reshape(M, n_groups * 2)
        per_col = M // cols
        chunks = []
        for c in range(cols):
            for r0 in range(c * per_col, (c + 1) * per_col, m_input):
                chunks.append(np.concatenate([
                    w_np[r0:r0 + m_input].reshape(-1),
                    s_bytes[r0:r0 + m_input].reshape(-1)]))
        ab = np.concatenate(chunks).astype(np.uint8)
        return torch.from_numpy(ab).view(torch.bfloat16)

    def _set_wo_tiled(self, name, w_q, sc_mg, M, K, cols, tsi_o, n_o_tiles):
        """fuse_o's Wo in the design's per-core TILE-major wire: each core gets its own contiguous
        run of `n_o_tiles` block-major tiles (payload block, then scale block), covering the core's
        ceil(D_PER_CORE/TSI_O) TSI_O-row window — rows past the matrix end are zero (the design's
        overlap/pad), and the windows' duplicated rows are stored per core, not shared.
        Mirrors design_ours_int8.py's WO_L3_UNITS fill and op_ours._wo_tiling.

        PAD ROWS (the per-core overlap tail, `pad` rows past the matrix end): payload byte 0 and
        scale byte 0 in BOTH domains.  That is domain-symmetric on its own — the row's
        contribution is (byte - 128)*0 == byte*0 == 0 — so, unlike a pad sitting under a NONZERO
        scale (the 4B's Wd gather-pad columns), this fill needs no domain flip.  The
        assert below makes that a checked invariant instead of a hope.
        """
        n_groups = K // 128
        rows_per_col = M // cols
        pad = n_o_tiles * tsi_o - rows_per_col
        assert pad >= 0, (n_o_tiles, tsi_o, rows_per_col)
        W = np.zeros((M + pad, K), np.uint8)
        S = np.zeros((M + pad, n_groups * 2), np.uint8)
        W[:M] = w_q.numpy().astype(np.uint8)
        S[:M] = sc_mg.contiguous().view(torch.uint8).numpy().reshape(M, n_groups * 2)
        assert not W[M:].any() and not S[M:].any(), \
            "Wo pad rows must be zero in both payload domains (their contribution is scale 0)"
        chunks = []
        for g in range(cols):
            for j in range(n_o_tiles):
                r0 = g * rows_per_col + j * tsi_o
                chunks.append(np.concatenate([W[r0:r0 + tsi_o].reshape(-1),
                                              S[r0:r0 + tsi_o].reshape(-1)]))
        self._set(name, torch.from_numpy(np.concatenate(chunks)).view(torch.bfloat16))

    def _set_int8(self, name, qw_list, sc_list, M, K, cols, m_input, signed=False):
        """Concatenate GPTQ-packed projections along M, then pack into the
        [int8 | scales] tile stream and write the buffer.

        `signed` is the payload domain of the resulting wire (`_gptq_rows`).  Callers must pass it
        only for wires whose producing op carries the matching `kernel_source`: Wqkv under the
        fused QKV head, and Wg/Wu/Wd (and fuse_o's Wo) under the fused MLP.
        """
        ws, scs = [], []
        m0 = 0
        for qw, sc in zip(qw_list, sc_list):
            w_q, s_mg = self._gptq_rows(qw, sc, signed=signed)
            ws.append(w_q)
            scs.append(s_mg)
            m0 += w_q.shape[0]
        w_all = torch.cat(ws, 0)
        sc_all = torch.cat(scs, 0)
        self._set(name, self._pack_int8(w_all, sc_all, m0, K, cols, m_input))

    def _set_int8_rtn(self, name, w_bf16_mk, M, K, cols, m_input, signed=False):
        """bf16 (M,K) weight -> symmetric per-128-group int8 -> packed tile stream.

        Needed because this GPTQ checkpoint's lm_head is TIED to bf16 embed_tokens
        (`lm_head: false`), so no prequantized wire exists.  The biased kernel's dequant is
        `w = (byte - 128) * scale` and the signed one's is `w = int8(byte) * scale`, so per
        128-element group along K the quantizer stores `s = max|w| / 127` and the same integer
        `q = round(w / s_wire)` clamped to [-127,127], written as `q + 128` (`signed=False`) or as
        two's complement (`signed=True`).  The rounding is done against the BF16 scale the wire
        will actually carry, so the residual is the rounding of q alone,
        |w - deq| <= s/2 = max|w|/254 (~0.4% of the group peak) — the same order as bf16's own
        rounding.  f32 math, chunked over M so the f32 working set stays small (vocab x emb f32
        would be ~0.6 GB).  No padding: M // cols divides by m_input and K by 128, so the wire is
        exactly as long as the op's arg spec.
        """
        w = w_bf16_mk
        assert tuple(w.shape) == (M, K), (tuple(w.shape), (M, K))
        assert K % 128 == 0, (K, 128)
        n_groups = K // 128
        payload = torch.empty((M, K), dtype=torch.uint8)
        sc = torch.empty((M, n_groups), dtype=torch.bfloat16)
        chunk = 16384
        for r0 in range(0, M, chunk):
            blk = w[r0:r0 + chunk].float().reshape(-1, n_groups, 128)
            s_ = blk.abs().amax(dim=2) / 127.0
            s_ = torch.where(s_ > 0, s_, torch.ones_like(s_))
            s_wire = s_.to(torch.bfloat16)
            q = torch.round(blk / s_wire.float()[:, :, None]).clamp_(-127, 127)
            if signed:
                # the same integer q as two's complement; the kernel's to_float<int8> IS q
                payload[r0:r0 + chunk] = q.to(torch.int8).view(torch.uint8).reshape(-1, K)
            else:
                payload[r0:r0 + chunk] = (q + 128).clamp_(1, 255).to(torch.uint8).reshape(-1, K)
            sc[r0:r0 + chunk] = s_wire
        self._set(name, self._pack_int8(payload, sc, M, K, cols, m_input))

    def _load_weights(self):
        for i in range(n_layers):
            w = self.weight_cache[i]
            self._set(f"W_norm1_{i}", w["input_norm"])
            # int8_ours wire: OUR mv_int8 (mode14/VEC128) over OUR
            # [tile_m*K u8 | tile_m*(K/128) bf16 s] tile stream, q/k/v
            # concatenated along M, m_input = the op's tile_size_input (4
            # rows x 1040 B = 4160 B, a 64 B multiple).  Per-head qk-norm
            # weights ride as two separate 128-element bf16 args.  The wire is
            # packed signed, paired with the fused head's per-instance
            # `kernel_source=` (a signed wire under a biased kernel, or the
            # reverse, reads q+128).
            self._set_int8(f"W_qkv_{i}",
                           [w["q_qw"], w["k_qw"], w["v_qw"]],
                           [w["q_sc"], w["k_sc"], w["v_scl"]],
                           qkv_dim, emb_dim, 8, QKV_DP_TSI,
                           signed=True)
            self._set(f"W_qn_{i}", w["q_norm"])
            self._set(f"W_kn_{i}", w["k_norm"])
            self._set(f"W_norm2_{i}", w["norm2"])
            # Fused whole-block MLP wire: the SAME per-(col,tile) packer the
            # GEMVInt8 arms use, with m_input = the design's tile rows —
            # TSI_GU (12 by default) for gate/up and TSI_D = TSI_GU/R for
            # down; TSI_GU*WROW_D == TSI_D*WROW_FF is the design's shared-tile
            # identity.  Three separate args, gate/up/down.
            # The values are READ OFF THE OP, not hardcoded: MLP_TSI/MLP_DEPTH
            # (op_ours.__post_init__) are compile-time knobs, and a packer that
            # ignored them would silently feed the design a wire of the wrong
            # tile shape (no error, wrong numbers — caught only by the gate).
            R = hidden_dim // emb_dim
            tsi_gu = self.mlp_xdna.tile_rows_gu
            tsi_d = tsi_gu // R
            assert tsi_gu * (emb_dim + (emb_dim // 128) * 2) == \
                tsi_d * (hidden_dim + (hidden_dim // 128) * 2), (
                    f"MLP weight tile identity broken: TSI_GU={tsi_gu} TSI_D={tsi_d}")
            # The three MLP wires ride the domain the fused MLP op was built
            # with (`kernel_source=`/`acc_kernel_source=` on `self.mlp_xdna`).
            self._set_int8(f"Wg_{i}", [w["gate"]], [w["gate_scl"]],
                           hidden_dim, emb_dim, self.mlp_xdna.num_aie_columns, tsi_gu,
                           signed=True)
            self._set_int8(f"Wu_{i}", [w["up"]], [w["up_scl"]],
                           hidden_dim, emb_dim, self.mlp_xdna.num_aie_columns, tsi_gu,
                           signed=True)
            self._set_int8(f"Wd_{i}", [w["down"]], [w["down_scl"]],
                           emb_dim, hidden_dim, self.mlp_xdna.num_aie_columns, tsi_d,
                           signed=True)
            # fuse_o's Wo rides the SAME weight channel, in the design's per-core
            # TILE-major wire (block-major tiles) — see op_ours._wo_tiling and
            # design_ours_int8.py's WO_L3_UNITS comment for why the row-interleaved form
            # (which has the same byte count) is wrong.
            # fuse_o's Wo is read by the MLP's own o-projection kernel
            # (`o_gemv_{QD}k_128vs_ours.o`, built from the MLP op's `kernel_source`), so it
            # moves with Wg/Wu/Wd, not with a gemv_output of its own.
            o_q, o_s = self._gptq_rows(w["o"], w["o_scl"], signed=True)
            _n, n_o_tiles, tsi_o, _wu = self.mlp_xdna._wo_tiling()
            assert self.mlp_xdna._wo_units() * 2 == \
                _n * n_o_tiles * _wu, "Wo arg length drifted from the design's tiling"
            self._set_wo_tiled(f"Wo_{i}", o_q, o_s, emb_dim, q_dim,
                               self.mlp_xdna.num_aie_columns, tsi_o, n_o_tiles)
            del self.weight_cache[i]
        self._set("W_final_norm", self.final_norm)
        # This checkpoint's lm_head is TIED to bf16 embed_tokens, so quantize
        # it here into the same [u8 | bf16 scales] tile stream the chain's
        # other int8 GEMVs use (cols=8, tsi=LMHEAD_TSI=4).
        self._set_int8_rtn("W_out_head", self.out_head, vocab_size, emb_dim, 8,
                           LMHEAD_TSI, signed=True)
        del self.final_norm, self.out_head

        # RoPE angle table (interleaved cos/sin, half dim)
        self._precompute_rope_angles()

        # sync static buffers to device
        # sync_independent_buffers() pushes dedicated weight BOs.  The packed
        # int8 weights live in the scratch buffer and were written via
        # torch_view(), which does NOT mark the parent dirty (only .data does),
        # so the residency-guarded push inside _sync_inputs would no-op and the
        # device would run on the zero-initialized scratch.  Force the push.
        self.fc.sync_independent_buffers()
        self.fc.input_buffer.device = "cpu"
        self.fc.input_buffer.to("npu")
        self.fc.scratch_buffer.device = "cpu"
        self.fc.scratch_buffer.to("npu")

    def _set(self, name, t):
        b = self.fc.get_buffer(name).torch_view()
        b[:] = t.flatten() if t.dim() > 1 else t

    def _precompute_rope_angles(self):
        half = head_dim // 2
        inv_freq = 1.0 / (rope_theta ** (torch.arange(0, half, dtype=torch.float32) / half))
        t = torch.arange(max_seq_len, dtype=torch.float32)
        freqs = torch.outer(t, inv_freq)              # (max_seq_len, half)
        emb = torch.cat([freqs, freqs], dim=-1)       # (max_seq_len, head_dim)
        cos = emb.cos().to(torch.bfloat16)            # (max_seq_len, head_dim)
        sin = emb.sin().to(torch.bfloat16)
        cos_half = cos[:, :half]
        sin_half = sin[:, :half]
        # interleaved [cos0, sin0, cos1, sin1, ...] -> (max_seq_len, head_dim)
        self.angle_table = torch.stack([cos_half, sin_half], dim=-1).reshape(max_seq_len, head_dim)

    # ---------------- forward (single token) ----------------
    def __call__(self, x, seq_pos):
        """x: (emb_dim,) bf16 hidden state for the new token; seq_pos: position index."""
        # interleaved KV cache write position for this token (element offset):
        # interleaved per-token layout (matches decode_attn's KV_tap), each
        # token occupies 2*head_dim elements: K at t*2*hd, V at t*2*hd+hd (the
        # fused head keys the V half off k_cache_offset's static +head_dim term).
        k_off = seq_pos * (2 * head_dim)                  # K[t]

        # update per-token rope_angles
        _ang = self.angle_table[seq_pos]
        self.fc.get_buffer("rope_angles").torch_view()[:] = _ang
        # write x; _sync_inputs pushes input+scratch (residency-guarded)
        self.fc.get_buffer("x").torch_view()[:] = x

        self.fc.params.write("k_cache_offset", np.int32(k_off))
        self.fc.params.write("S_kv_eff", np.int32(seq_pos + 1))
        self.fc.params.sync()

        self.fc()  # runs 28 layers + final RMSNorm + lm_head GEMV -> logits

        # logits already synced to CPU by SequenceCallable.__call__ (_sync_outputs).
        # Host argmax over vocab bf16 takes a slow CPU path (0.41 ms of a 39 ms token); routing it
        # through f32 numpy is 8x faster and returns the same index (measured 0.048 vs 0.413 ms,
        # agreement True), i.e. it is free time the NPU does not have to wait for.
        logits = self.fc.get_buffer("logits").torch_view()
        if os.environ.get("ARGMAX_FAST", "1") == "1":
            return int(np.argmax(logits.float().numpy()))
        return torch.argmax(logits).item()


if __name__ == "__main__":
    # Smoke test: load weights, run one decode step, report argmax + no NaN.
    # Weights dir (the GPTQ-Int8 checkpoint's model.safetensors) via
    # $QWEN3_INT8_MODEL_DIR (default /srv/qwen3-0.6b-gptq-int8).
    weights = Path(os.environ.get(
        "QWEN3_INT8_MODEL_DIR", "/srv/qwen3-0.6b-gptq-int8")) / "model.safetensors"
    model = Qwen3NPU(weights)
    x = torch.randn(emb_dim, dtype=torch.bfloat16)
    token_id = model(x, seq_pos=10)
    print(f"argmax token_id = {token_id}")
