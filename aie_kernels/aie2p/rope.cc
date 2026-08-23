// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RoPE (Rotary Position Embedding) kernel for AIE2P.
//
// Applies rotary position embedding to Q and K tiles in-place (L1 memory):
//   q_out[i] = q[i] * cos[i] - rotate_half(q)[i] * sin[i]
//   k_out[j] = k[j] * cos[j] - rotate_half(k)[j] * sin[j]
//
// rotate_half(x) for head_dim d: cat([-x[d/2:], x[:d/2]])
// This is equivalent to:
//   For i in [0, d/2):   out[i]       = x[i] * cos[i] - x[i + d/2] * sin[i]
//   For i in [d/2, d):  out[i]       = x[i] * cos[i] + x[i - d/2] * sin[i]
//
// cos/sin are precomputed tables of shape (d,) — one row's worth.
// The caller passes the correct cos/sin slice for the current position.
//
// This kernel operates on tiles already in L1. It is designed to be called
// from the QK matmul worker, right before matmul_QK, so Q/K tiles don't
// leave the AIE core between RoPE and matmul.

#include <aie_api/aie.hpp>
#include <stdint.h>

#define ROPE_VEC_LEN 64  // AIE2P vector length for bf16

using namespace aie;

// Apply RoPE to a single row of length d (d must be 64 or 128).
// Operates in-place on the row buffer.
// cos_table and sin_table are of length d.
static inline void rope_row_bf16(bfloat16 *restrict row,
                                   const bfloat16 *restrict cos_table,
                                   const bfloat16 *restrict sin_table,
                                   const int32_t d)
{
    // rotate_half(x) for head_dim d: cat([-x[d/2:], x[:d/2]])
    //   For i in [0, d/2):   out[i]       = x[i] * cos[i] - x[i + d/2] * sin[i]
    //   For i in [d/2, d):  out[i]       = x[i] * cos[i] + x[i - d/2] * sin[i]

    // Fast path: identity rotation (cos=1, sin=0) — skip entirely to avoid
    // accumulator rounding that introduces spurious bf16 errors.
    if (cos_table[0] == (bfloat16)1.0 && sin_table[0] == (bfloat16)0.0) {
        return;  // no-op
    }

    if (d == ROPE_VEC_LEN) {
        // d=64: process as a single 64-element vector.
        // x = [a0..a31 | b0..b31]
        // rotate_half(x) = [-b0..-b31 | a0..a31]

        aie::vector<bfloat16, ROPE_VEC_LEN> cos_vec = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)cos_table);
        aie::vector<bfloat16, ROPE_VEC_LEN> sin_vec = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)sin_table);
        aie::vector<bfloat16, ROPE_VEC_LEN> x_vec = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)row);

        auto x_lo = x_vec.extract<ROPE_VEC_LEN / 2>(0);  // first half: a0..a31
        auto x_hi = x_vec.extract<ROPE_VEC_LEN / 2>(1);  // second half: b0..b31

        aie::vector<bfloat16, ROPE_VEC_LEN / 2> zeros = aie::broadcast<bfloat16, ROPE_VEC_LEN / 2>((bfloat16)0.0);
        auto neg_x_hi = aie::sub(zeros, x_hi);

        aie::vector<bfloat16, ROPE_VEC_LEN> rotated;
        rotated.insert(0, neg_x_hi);  // first half = -b
        rotated.insert(1, x_lo);      // second half = a

        aie::accum<accfloat, ROPE_VEC_LEN> acc;
        acc = aie::mul(x_vec, cos_vec);
        acc = aie::add(acc, aie::mul(rotated, sin_vec));

        aie::store_v((bfloat16 *)row, acc.to_vector<bfloat16>());
    } else {
        // d=128: process as two 64-element chunks with cross-referencing.
        // x = [chunk0 | chunk1] where chunk0 = x[0:64], chunk1 = x[64:128]
        // rotate_half(x) = [-chunk1 | chunk0]
        //   rotated_chunk0 = -chunk1  (negate second half → first half of rotated)
        //   rotated_chunk1 = chunk0   (first half → second half of rotated)

        aie::vector<bfloat16, ROPE_VEC_LEN> chunk0 = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)row);
        aie::vector<bfloat16, ROPE_VEC_LEN> chunk1 = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)row + ROPE_VEC_LEN);
        aie::vector<bfloat16, ROPE_VEC_LEN> cos0 = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)cos_table);
        aie::vector<bfloat16, ROPE_VEC_LEN> cos1 = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)cos_table + ROPE_VEC_LEN);
        aie::vector<bfloat16, ROPE_VEC_LEN> sin0 = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)sin_table);
        aie::vector<bfloat16, ROPE_VEC_LEN> sin1 = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)sin_table + ROPE_VEC_LEN);

        // Negate chunk1 for the first half of rotated
        aie::vector<bfloat16, ROPE_VEC_LEN> zeros = aie::broadcast<bfloat16, ROPE_VEC_LEN>((bfloat16)0.0);
        auto neg_chunk1 = aie::sub(zeros, chunk1);

        // out[0:64] = chunk0 * cos0 + (-chunk1) * sin0
        aie::accum<accfloat, ROPE_VEC_LEN> acc0;
        acc0 = aie::mul(chunk0, cos0);
        acc0 = aie::add(acc0, aie::mul(neg_chunk1, sin0));
        aie::store_v((bfloat16 *)row, acc0.to_vector<bfloat16>());

        // out[64:128] = chunk1 * cos1 + chunk0 * sin1
        aie::accum<accfloat, ROPE_VEC_LEN> acc1;
        acc1 = aie::mul(chunk1, cos1);
        acc1 = aie::add(acc1, aie::mul(chunk0, sin1));
        aie::store_v((bfloat16 *)row + ROPE_VEC_LEN, acc1.to_vector<bfloat16>());
    }
}

extern "C" {

// Apply RoPE to a Q tile of shape (B_q, d) in-place.
// cos_table and sin_table are of shape (d,) — the position-specific slice.
void rope_qk_bf16(bfloat16 *restrict q_tile,
                   bfloat16 *restrict k_tile,
                   const bfloat16 *restrict cos_table,
                   const bfloat16 *restrict sin_table,
                   const int32_t B_q,
                   const int32_t d)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    // Apply RoPE to each row of Q (row-major, B_q rows of d elements)
    for (int row = 0; row < B_q; row++) {
        rope_row_bf16(q_tile + row * d, cos_table, sin_table, d);
    }

    // K tile is (d, B_kv) in column-major order.
    // Each column of K needs RoPE applied to its d elements.
    // k_tile[col * d + i] is the i-th element of column col.
    // We need to apply RoPE to each column independently.
    // But all columns at the same sequence position share the same cos/sin.
    // For the QK matmul, K is laid out as (d, B_kv) column-major,
    // so k_tile[i + col * d] = K[col, i] (element i of column col).
    //
    // Since cos/sin don't change across columns (same position), we can
    // apply RoPE column by column. For B_kv columns:
    for (int col = 0; col < B_q; col++) {  // B_q == B_kv in current design
        // K column at offset col * d
        rope_row_bf16(k_tile + col * d, cos_table, sin_table, d);
    }
}

} // extern "C"
