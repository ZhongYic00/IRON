// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Fused QK-norm + RoPE + V-copy kernel for AIE2P.
//
// Input: qkv (qkv_dim,) bf16 — flat QKV vector from GEMV.
//   Layout: [Q(16 heads × 128) | K(8 heads × 128) | V(8 heads × 128)]
// Output: qkv_out (qkv_dim,) bf16 — Q and K normalized + RoPE-rotated, V copied.
//
// Scratch (merged gamma + cos_sin to fit 2 MM2S channels):
//   scratch: [qk_gamma(3072) | cos(128) | sin(128)]
//     qk_gamma: (n_qk_heads * head_dim,) bf16 — per-head QK norm gamma
//     cos: (head_dim,) bf16 — cos table for current position
//     sin: (head_dim,) bf16 — sin table for current position

#include <aie_api/aie.hpp>
#include <stdint.h>

#define VLEN 64  // AIE2P bf16 vector length
#define FLEN 32  // AIE2P float vector length

using namespace aie;

extern "C" {

void qk_norm_rope_vcopy(bfloat16 *restrict qkv_out,
                         const bfloat16 *restrict qkv_in,
                         const bfloat16 *restrict scratch,
                         const float eps,
                         const int32_t head_dim)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    const int32_t n_qk_heads = 24;  // 16 Q + 8 K
    const int32_t n_v_heads = 8;
    const bfloat16 *qk_gamma = scratch;                          // [0 .. 3071]
    const bfloat16 *cos = scratch + n_qk_heads * head_dim;       // [3072 .. 3199]
    const bfloat16 *sin = cos + head_dim;                        // [3200 .. 3327]

    // Process Q and K heads (norm + RoPE)
    for (int h = 0; h < n_qk_heads; h++) {
        const bfloat16 *x = qkv_in + h * head_dim;
        bfloat16 *out = qkv_out + h * head_dim;
        const bfloat16 *gamma = qk_gamma + h * head_dim;

        // --- RMSNorm (128 elements) ---
        // Pass 1: compute sum(x^2) using mul_square (bf32→f32, reduce)
        float sum_sq = 0.0f;
        for (int i = 0; i < 4; i++) {  // 4 × 32 = 128
            aie::vector<bfloat16, FLEN> xv = aie::load_v<FLEN>((bfloat16 *)(x + i * FLEN));
            aie::vector<float, FLEN> sq = aie::mul_square(xv);
            sum_sq += aie::reduce_add(sq);
        }
        float rms = sum_sq / 128.0f + eps;
        float inv_rms = aie::invsqrt(rms);

        // Pass 2: normalize: out = x * inv_rms * gamma
        // Process in 4 chunks of 32 bf16 (matching float vector length)
        bfloat16 xn_buf[128];

        aie::accum<accfloat, FLEN> inv_rms_acc;
        inv_rms_acc.from_vector(aie::broadcast<float, FLEN>(inv_rms), 0);

        for (int i = 0; i < 4; i++) {  // 4 × 32 = 128
            int off = i * FLEN;
            aie::accum<accfloat, FLEN> acc;
            acc.from_vector(aie::load_v<FLEN>((bfloat16 *)(x + off)), 0);
            acc = aie::mul(acc.to_vector<float>(), inv_rms_acc.to_vector<float>());
            aie::accum<accfloat, FLEN> gacc;
            gacc.from_vector(aie::load_v<FLEN>((bfloat16 *)(gamma + off)), 0);
            acc = aie::mul(acc.to_vector<float>(), gacc.to_vector<float>());
            aie::store_v((bfloat16 *)(xn_buf + off), acc.to_vector<bfloat16>());
        }

        // --- RoPE: rotate_half on 128 elements ---
        aie::vector<bfloat16, VLEN> xn0 = aie::load_v<VLEN>((bfloat16 *)xn_buf);
        aie::vector<bfloat16, VLEN> xn1 = aie::load_v<VLEN>((bfloat16 *)(xn_buf + VLEN));

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
    }

    // Copy V heads (no norm, no RoPE)
    const bfloat16 *v_in = qkv_in + n_qk_heads * head_dim;
    bfloat16 *v_out = qkv_out + n_qk_heads * head_dim;
    int32_t v_elems = n_v_heads * head_dim;  // 1024
    for (int i = 0; i < v_elems; i += VLEN) {
        aie::vector<bfloat16, VLEN> v = aie::load_v<VLEN>((bfloat16 *)(v_in + i));
        aie::store_v((bfloat16 *)(v_out + i), v);
    }
}

} // extern "C"
