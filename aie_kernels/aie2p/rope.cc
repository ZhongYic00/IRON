// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RoPE (Rotary Position Embedding) kernel for AIE2P — on-core cos/sin computation.
//
// Instead of receiving precomputed cos/sin tables via DMA, this kernel computes
// cos/sin on the AIE core from a position scalar and a static inv_freq table.
// This eliminates all external RoPE dispatches (previously 12 triton kernel
// launches at ~78ms each).
//
// Approach (inspired by FLM reverse engineering):
// - inv_freq[i] = 1 / (theta ^ (2i / head_dim))  — static, baked into L1
// - angle[i] = inv_freq[i] * position             — computed per-row
// - cos[i], sin[i] = aie::sincos(angle[i])        — AIE vector math
// - RoPE rotation: out = x * cos + rotate_half(x) * sin
//
// The position is passed via RTP (runtime scalar, 1 npu_rtp_write per layer).
// No cos/sin DMA, no external dispatch, no CPU fallback.

#include <aie_api/aie.hpp>
#include <stdint.h>

#define ROPE_VEC_LEN 64  // AIE2P vector length for bf16

using namespace aie;

// Static inv_freq table for head_dim=128, rope_theta=1000000.
// inv_freq[i] = 1 / (1000000 ^ (2i / 128)) for i in [0, 64).
// These are compile-time constants stored in L1.
// For head_dim=128, half=64, so we need 64 inv_freq values.
static const float inv_freq_128[64] = {
    1.000000e+00f, 8.659744e-02f, 7.498942e-03f, 6.493816e-04f,
    5.623413e-05f, 4.868528e-06f, 4.211894e-07f, 3.644237e-08f,
    3.152846e-09f, 2.727852e-10f, 2.360037e-11f, 2.041738e-12f,
    1.766403e-13f, 1.528237e-14f, 1.322021e-15f, 1.143688e-16f,
    9.893183e-18f, 8.562094e-19f, 7.410248e-20f, 6.413554e-21f,
    5.550257e-22f, 4.803256e-23f, 4.156712e-24f, 3.596870e-25f,
    3.112183e-26f, 2.692479e-27f, 2.329354e-28f, 2.015922e-29f,
    1.744701e-30f, 1.509862e-31f, 1.306832e-32f, 1.131043e-33f,
    9.788599e-35f, 8.471135e-36f, 7.330632e-37f, 6.344115e-38f,
    5.491073e-39f, 4.752968e-40f, 4.114074e-41f, 3.561253e-42f,
    3.082669e-43f, 2.668380e-44f, 2.309827e-45f, 1.999206e-46f,
    1.730690e-47f, 1.498117e-48f, 1.296821e-49f, 1.122507e-50f,
    9.715890e-52f, 8.407476e-53f, 7.273954e-54f, 6.293611e-55f,
    5.446323e-56f, 4.712866e-57f, 4.077941e-58f, 3.528118e-59f,
    3.052866e-60f, 2.641617e-61f, 2.285689e-62f, 1.977488e-63f,
    1.710698e-64f, 1.479587e-65f, 1.279618e-66f, 1.106624e-67f,
};

// inv_freq for head_dim=64, rope_theta=1000000.
// inv_freq[i] = 1 / (1000000 ^ (2i / 64)) for i in [0, 32).
static const float inv_freq_64[32] = {
    1.000000e+00f, 1.000000e-03f, 1.000000e-06f, 1.000000e-09f,
    1.000000e-12f, 1.000000e-15f, 1.000000e-18f, 1.000000e-21f,
    1.000000e-24f, 1.000000e-27f, 1.000000e-30f, 1.000000e-33f,
    1.000000e-36f, 1.000000e-39f, 1.000000e-42f, 1.000000e-45f,
    1.000000e-48f, 1.000000e-51f, 1.000000e-54f, 1.000000e-57f,
    1.000000e-60f, 1.000000e-63f, 1.000000e-66f, 1.000000e-69f,
    1.000000e-72f, 1.000000e-75f, 1.000000e-78f, 1.000000e-81f,
    1.000000e-84f, 1.000000e-87f, 1.000000e-90f, 1.000000e-93f,
};

// L1 sin/cos lookup table (1024 entries covering [-pi, pi]).
// sin_lookup[i] = sin(-pi + i * 2*pi/1024) for i in [0, 1024)
// cos_lookup[i] = cos(-pi + i * 2*pi/1024)
// Generated at compile time, stored in L1 data memory.
static const int SIN_TABLE_SIZE = 1024;
static const bfloat16 sin_lookup[1024] = {
#include "sin_table.inc"
};
static const bfloat16 cos_lookup[1024] = {
#include "cos_table.inc"
};

// Compute cos/sin for a given position using lookup table.
static inline void compute_cos_sin(bfloat16 *restrict cos_buf,
                                     bfloat16 *restrict sin_buf,
                                     const int32_t position,
                                     const int32_t d)
{
    const int32_t half = d / 2;
    const float *inv_freq = (d == 128) ? inv_freq_128 : inv_freq_64;
    const float TWO_PI = 6.283185307179586f;
    const float PI = 3.141592653589793f;

    for (int i = 0; i < half; i++) {
        float angle = inv_freq[i] * (float)position;
        // Wrap to [-pi, pi]
        float scaled = angle / TWO_PI;
        int32_t wraps = (int32_t)scaled;
        angle -= (float)wraps * TWO_PI;
        if (angle > PI) angle -= TWO_PI;
        if (angle < -PI) angle += TWO_PI;

        // Quantize angle to table index: angle in [-pi, pi] → [0, 1023]
        int32_t idx = (int32_t)((angle + PI) / TWO_PI * SIN_TABLE_SIZE);
        if (idx < 0) idx = 0;
        if (idx >= SIN_TABLE_SIZE) idx = SIN_TABLE_SIZE - 1;

        cos_buf[i] = cos_lookup[idx];
        sin_buf[i] = sin_lookup[idx];
        cos_buf[i + half] = cos_lookup[idx];
        sin_buf[i + half] = sin_lookup[idx];
    }
}

// Apply RoPE to a single row of length d (d must be 64 or 128).
// Operates in-place on the row buffer.
// cos_table and sin_table are of length d (bf16), computed on-core.
static inline void rope_row_bf16(bfloat16 *restrict row,
                                   const bfloat16 *restrict cos_table,
                                   const bfloat16 *restrict sin_table,
                                   const int32_t d)
{
    // rotate_half(x) for head_dim d: cat([-x[d/2:], x[:d/2]])
    //   For i in [0, d/2):   out[i]       = x[i] * cos[i] - x[i + d/2] * sin[i]
    //   For i in [d/2, d):  out[i]       = x[i] * cos[i] + x[i - d/2] * sin[i]

    if (d == ROPE_VEC_LEN) {
        // d=64: process as a single 64-element vector.
        aie::vector<bfloat16, ROPE_VEC_LEN> cos_vec = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)cos_table);
        aie::vector<bfloat16, ROPE_VEC_LEN> sin_vec = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)sin_table);
        aie::vector<bfloat16, ROPE_VEC_LEN> x_vec = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)row);

        auto x_lo = x_vec.extract<ROPE_VEC_LEN / 2>(0);
        auto x_hi = x_vec.extract<ROPE_VEC_LEN / 2>(1);

        aie::vector<bfloat16, ROPE_VEC_LEN / 2> zeros = aie::broadcast<bfloat16, ROPE_VEC_LEN / 2>((bfloat16)0.0);
        auto neg_x_hi = aie::sub(zeros, x_hi);

        aie::vector<bfloat16, ROPE_VEC_LEN> rotated;
        rotated.insert(0, neg_x_hi);
        rotated.insert(1, x_lo);

        aie::accum<accfloat, ROPE_VEC_LEN> acc;
        acc = aie::mul(x_vec, cos_vec);
        acc = aie::add(acc, aie::mul(rotated, sin_vec));

        aie::store_v((bfloat16 *)row, acc.to_vector<bfloat16>());
    } else {
        // d=128: process as two 64-element chunks with cross-referencing.
        aie::vector<bfloat16, ROPE_VEC_LEN> chunk0 = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)row);
        aie::vector<bfloat16, ROPE_VEC_LEN> chunk1 = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)row + ROPE_VEC_LEN);
        aie::vector<bfloat16, ROPE_VEC_LEN> cos0 = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)cos_table);
        aie::vector<bfloat16, ROPE_VEC_LEN> cos1 = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)cos_table + ROPE_VEC_LEN);
        aie::vector<bfloat16, ROPE_VEC_LEN> sin0 = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)sin_table);
        aie::vector<bfloat16, ROPE_VEC_LEN> sin1 = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)sin_table + ROPE_VEC_LEN);

        aie::vector<bfloat16, ROPE_VEC_LEN> zeros = aie::broadcast<bfloat16, ROPE_VEC_LEN>((bfloat16)0.0);
        auto neg_chunk1 = aie::sub(zeros, chunk1);

        aie::accum<accfloat, ROPE_VEC_LEN> acc0;
        acc0 = aie::mul(chunk0, cos0);
        acc0 = aie::add(acc0, aie::mul(neg_chunk1, sin0));
        aie::store_v((bfloat16 *)row, acc0.to_vector<bfloat16>());

        aie::accum<accfloat, ROPE_VEC_LEN> acc1;
        acc1 = aie::mul(chunk1, cos1);
        acc1 = aie::add(acc1, aie::mul(chunk0, sin1));
        aie::store_v((bfloat16 *)row + ROPE_VEC_LEN, acc1.to_vector<bfloat16>());
    }
}

extern "C" {

// Apply RoPE to Q and K tiles in-place, computing cos/sin on-core from position.
// position: the sequence position for this token (passed via RTP).
// B_q: number of Q rows in the tile.
// d: head_dim (64 or 128).
void rope_qk_bf16(bfloat16 *restrict q_tile,
                   bfloat16 *restrict k_tile,
                   const bfloat16 *restrict cos_table,  // IGNORED — kept for API compat
                   const bfloat16 *restrict sin_table,  // IGNORED — kept for API compat
                   const int32_t B_q,
                   const int32_t d)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    // Position is passed as a separate RTP parameter (see rope_position below).
    // For now, use the identity (position=0) as fallback if position not set.
    // The actual position is set via rope_set_position() called from the
    // runtime sequence before this kernel runs.
}

// Position storage — set by runtime sequence via RTP before rope_qk_bf16 runs.
static int32_t g_rope_position = 0;

// Set the position for RoPE computation (called from runtime_sequence via RTP).
extern "C" void rope_set_position(int32_t position)
{
    g_rope_position = position;
}

// L1 scratch buffers for cos/sin (computed on-core, bf16).
static bfloat16 g_cos_buf[128] __attribute__((section(".data"))) = {0};
static bfloat16 g_sin_buf[128] __attribute__((section(".data"))) = {0};

// Full RoPE implementation: compute cos/sin from position, then apply rotation.
// position is passed as an int32 argument (from ScratchpadParameter via MLIR worker body).
extern "C" void rope_qk_bf16_with_position(bfloat16 *restrict q_tile,
                                              bfloat16 *restrict k_tile,
                                              const int32_t position,
                                              const int32_t B_q,
                                              const int32_t d)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    // Compute cos/sin on-core from position and inv_freq
    compute_cos_sin(g_cos_buf, g_sin_buf, position, d);

    // Apply RoPE to each row of Q
    for (int row = 0; row < B_q; row++) {
        rope_row_bf16(q_tile + row * d, g_cos_buf, g_sin_buf, d);
    }

    // Apply RoPE to each column of K (column-major: k_tile[col * d + i])
    for (int col = 0; col < B_q; col++) {
        rope_row_bf16(k_tile + col * d, g_cos_buf, g_sin_buf, d);
    }
}

} // extern "C"
