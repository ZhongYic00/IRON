// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Fused QK-norm + RoPE + V-copy + KV-DIRECT kernel for AIE2P.
//
// This is qk_norm_rope.cc plus ONE extra output: the K|V slice that the chain's
// merged StridedCopy (`sc_kv`) used to scatter into the KV cache.  The producer
// writes it itself, so the scatter op (a whole runlist entry / configure point
// per layer, moving 4 KB) disappears.  The math is byte-for-byte the same
// normalization + rotate_half RoPE as the base kernel; only the destination of
// K and V is duplicated.
//
//   Input:  qkv (qkv_dim,) bf16, [Q(n_q*HD) | K(n_v*HD) | V(n_v*HD)]
//   Output: qkv_out (qkv_dim,) bf16 — Q and K normalized + RoPE-rotated, V
//           copied (UNCHANGED from qk_norm_rope.cc; the chain's decode_attn only
//           reads qkv_out[0:q_dim] once the scatter op is gone, but keeping the
//           tail written costs two L1 stores and removes a whole class of
//           "which buffer is stale" doubt while the arm is being validated).
//   Output: kv (2*n_v_heads*head_dim,) bf16 — the KV-cache slice, laid out
//           EXACTLY like the source slice sc_kv read:
//             [K head 0 .. KVH-1 | V head 0 .. KVH-1], head_dim contiguous each
//           The design drains this one 4 KB object with sc_kv's own tap
//           (sizes [1,2,KVH,HD], strides [0,HD,2*S_KV*HD,1], plus the
//           `k_cache_offset` scratchpad parameter = t*2*head_dim elements), so
//           the bytes landing in the cache are the bytes sc_kv used to write —
//           the layout contract stays defined by the READER (decode_attn).
//
// Scratch (merged gamma + cos_sin to fit 2 MM2S channels):
//   scratch: [qk_gamma(n_qk_heads*head_dim) | cos(head_dim) | sin(head_dim)]
//     cos/sin are HF rotate_half tables: halves duplicated.
//
// n_qk_heads / n_v_heads are runtime args so one .o serves both
// Qwen3-0.6B (24/8) and Qwen3-4B (40/8) shapes.
//
// K head <-> KV head mapping: the QK loop walks [Q heads | K heads], so the K
// half of `kv` is indexed h - (n_qk_heads - n_v_heads), i.e. this assumes
// n_k_heads == n_v_heads (GQA with equal K/V head counts) — true for every
// Qwen3 shape here (8 K heads, 8 V heads).  The kernel cannot check it: it
// never sees n_q_heads on its own.

#include <aie_api/aie.hpp>
#include <stdint.h>

#define VLEN 64  // AIE2P bf16 vector length
#define FLEN 32  // AIE2P float vector length

using namespace aie;

extern "C" {

void qk_norm_rope_kv_vcopy(bfloat16 *restrict qkv_out,
                           const bfloat16 *restrict qkv_in,
                           const bfloat16 *restrict scratch,
                           bfloat16 *restrict kv,
                           const float eps,
                           const int32_t head_dim,
                           const int32_t n_qk_heads,
                           const int32_t n_v_heads)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    const bfloat16 *qk_gamma = scratch;                          // [0 .. n_qk_heads*head_dim)
    const bfloat16 *cos = scratch + n_qk_heads * head_dim;       // [gamma_sz .. gamma_sz+head_dim)
    const bfloat16 *sin = cos + head_dim;                        // [gamma_sz+head_dim .. +head_dim)

    const int32_t n_q_heads = n_qk_heads - n_v_heads;            // K heads == n_v_heads

    // Process Q and K heads (norm + RoPE)
    for (int h = 0; h < n_qk_heads; h++) {
        const bfloat16 *x = qkv_in + h * head_dim;
        bfloat16 *out = qkv_out + h * head_dim;
        const bfloat16 *gamma = qk_gamma + h * head_dim;

        // --- RMSNorm (head_dim elements) ---
        // Pass 1: compute sum(x^2) using mul_square (bf32→f32, reduce)
        float sum_sq = 0.0f;
        for (int i = 0; i < 4; i++) {  // 4 × 32 = 128
            aie::vector<bfloat16, FLEN> xv = aie::load_v<FLEN>((bfloat16 *)(x + i * FLEN));
            aie::vector<float, FLEN> sq = aie::mul_square(xv);
            sum_sq += aie::reduce_add(sq);
        }
        float rms = sum_sq / (float)head_dim + eps;
        float inv_rms = aie::invsqrt(rms);

        // Pass 2: normalize: out = x * inv_rms * gamma, written straight to the
        // output memref. NOTE: no kernel-allocated intermediate array here —
        // aie::store_v into local (stack/BSS) arrays is broken on this
        // toolchain (wild stores + core fault); memref stores are fine.
        aie::vector<float, FLEN> inv_v = aie::broadcast<float, FLEN>(inv_rms);

        for (int i = 0; i < 4; i++) {  // 4 × 32 = 128
            int off = i * FLEN;
            aie::accum<accfloat, FLEN> acc;
            acc.from_vector(aie::load_v<FLEN>((bfloat16 *)(x + off)), 0);
            acc = aie::mul(acc.to_vector<float>(), inv_v);
            aie::accum<accfloat, FLEN> gacc;
            gacc.from_vector(aie::load_v<FLEN>((bfloat16 *)(gamma + off)), 0);
            acc = aie::mul(acc.to_vector<float>(), gacc.to_vector<float>());
            aie::store_v((bfloat16 *)(out + off), acc.to_vector<bfloat16>());
        }

        // --- RoPE: rotate_half, in-place on out ---
        aie::vector<bfloat16, VLEN> xn0 = aie::load_v<VLEN>((bfloat16 *)out);
        aie::vector<bfloat16, VLEN> xn1 = aie::load_v<VLEN>((bfloat16 *)(out + VLEN));

        aie::vector<bfloat16, VLEN> cos0 = aie::load_v<VLEN>((bfloat16 *)cos);
        aie::vector<bfloat16, VLEN> cos1 = aie::load_v<VLEN>((bfloat16 *)(cos + VLEN));
        aie::vector<bfloat16, VLEN> sin0 = aie::load_v<VLEN>((bfloat16 *)sin);
        aie::vector<bfloat16, VLEN> sin1 = aie::load_v<VLEN>((bfloat16 *)(sin + VLEN));

        // out[0:64] = xn0 * cos0 + (-xn1) * sin0
        aie::vector<bfloat16, VLEN> zeros = aie::broadcast<bfloat16, VLEN>((bfloat16)0.0);
        aie::vector<bfloat16, VLEN> neg_xn1 = aie::sub(zeros, xn1);

        aie::accum<accfloat, VLEN> acc0;
        acc0 = aie::mul(xn0, cos0);
        acc0 = aie::add(acc0, aie::mul(neg_xn1, sin0));
        aie::store_v((bfloat16 *)out, acc0.to_vector<bfloat16>());

        // out[64:128] = xn1 * cos1 + xn0 * sin1
        aie::accum<accfloat, VLEN> acc1;
        acc1 = aie::mul(xn1, cos1);
        acc1 = aie::add(acc1, aie::mul(xn0, sin1));
        aie::store_v((bfloat16 *)(out + VLEN), acc1.to_vector<bfloat16>());

        // K heads additionally land in the KV slice's K half; the rotated
        // values are already in registers, so this is two extra L1 stores.
        if (h >= n_q_heads) {
            bfloat16 *kdst = kv + (h - n_q_heads) * head_dim;
            aie::store_v((bfloat16 *)kdst, acc0.to_vector<bfloat16>());
            aie::store_v((bfloat16 *)(kdst + VLEN), acc1.to_vector<bfloat16>());
        }
    }

    // Copy V heads (no norm, no RoPE) into both destinations.
    const bfloat16 *v_in = qkv_in + n_qk_heads * head_dim;
    bfloat16 *v_out = qkv_out + n_qk_heads * head_dim;
    bfloat16 *v_kv = kv + n_v_heads * head_dim;                  // V half of the KV slice
    int32_t v_elems = n_v_heads * head_dim;
    for (int i = 0; i < v_elems; i += VLEN) {
        aie::vector<bfloat16, VLEN> v = aie::load_v<VLEN>((bfloat16 *)(v_in + i));
        aie::store_v((bfloat16 *)(v_out + i), v);
        aie::store_v((bfloat16 *)(v_kv + i), v);
    }
}

} // extern "C"
