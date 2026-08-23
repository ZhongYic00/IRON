// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// QK-norm kernel for AIE2P (Qwen3 Q/K per-head RMSNorm, no RoPE, no V).
//
// Input: qk_in (qkv_dim_without_v,) bf16 — flat [Q(n_q_heads × head_dim) | K(n_k_heads × head_dim)]
// Output: qk_out (same shape) bf16 — per-head RMSNorm with per-group shared gamma.
//
// Qwen3 semantics: q_norm.weight and k_norm.weight are each a single (head_dim,)
// vector shared across ALL heads of that group (16 Q heads share q_gamma;
// 8 K heads share k_gamma).  The RMS statistic is computed PER HEAD, but the
// gamma is shared, so the scratch only holds [q_gamma(head_dim) | k_gamma(head_dim)].
//
// Scratch: [q_gamma(128) | k_gamma(128)] — 2 × head_dim elements.

#include <aie_api/aie.hpp>
#include <stdint.h>

#define VLEN 64  // AIE2P bf16 vector length
#define FLEN 32  // AIE2P float vector length

using namespace aie;

extern "C" {

void qk_norm_bf16(bfloat16 *restrict qk_out,
                  const bfloat16 *restrict qk_in,
                  const bfloat16 *restrict gamma,
                  const float eps,
                  const int32_t head_dim,
                  const int32_t n_q_heads,
                  const int32_t n_k_heads)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    const bfloat16 *q_gamma = gamma;                      // [0 .. head_dim)
    const bfloat16 *k_gamma = gamma + head_dim;           // [head_dim .. 2*head_dim)

    const int32_t n_qk_heads = n_q_heads + n_k_heads;

    for (int h = 0; h < n_qk_heads; h++) {
        const bfloat16 *x = qk_in + h * head_dim;
        bfloat16 *out = qk_out + h * head_dim;
        // Q heads use q_gamma, K heads use k_gamma.
        const bfloat16 *g = (h < n_q_heads) ? q_gamma : k_gamma;

        // --- RMSNorm per head (head_dim elements) ---
        // Pass 1: sum(x^2) using mul_square (bf16 -> f32, reduce)
        float sum_sq = 0.0f;
        const int32_t n_loops = head_dim / FLEN;  // head_dim=128 -> 4
        for (int i = 0; i < n_loops; i++) {
            aie::vector<bfloat16, FLEN> xv = aie::load_v<FLEN>((bfloat16 *)(x + i * FLEN));
            aie::vector<float, FLEN> sq = aie::mul_square(xv);
            sum_sq += aie::reduce_add(sq);
        }
        float rms = sum_sq / (float)head_dim + eps;
        float inv_rms = aie::invsqrt(rms);

        // Pass 2: out = x * inv_rms * gamma
        aie::accum<accfloat, FLEN> inv_rms_acc;
        inv_rms_acc.from_vector(aie::broadcast<float, FLEN>(inv_rms), 0);

        for (int i = 0; i < n_loops; i++) {
            int off = i * FLEN;
            aie::accum<accfloat, FLEN> acc;
            acc.from_vector(aie::load_v<FLEN>((bfloat16 *)(x + off)), 0);
            acc = aie::mul(acc.to_vector<float>(), inv_rms_acc.to_vector<float>());
            aie::accum<accfloat, FLEN> gacc;
            gacc.from_vector(aie::load_v<FLEN>((bfloat16 *)(g + off)), 0);
            acc = aie::mul(acc.to_vector<float>(), gacc.to_vector<float>());
            aie::store_v((bfloat16 *)(out + off), acc.to_vector<bfloat16>());
        }
    }
}

} // extern "C"
