// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// RoPE kernel for OperatorSequence — reads position from scratchpad parameter.
// Uses a pre-loaded cos/sin lookup table in scratch buffer.
//
// Layout in scratch buffer (set by OperatorSequence):
//   cos_sin_table: (2 * max_seq_len * head_dim) bf16 — [cos_table | sin_table]
//     cos_table[pos * head_dim + i] = cos(inv_freq[i] * pos)
//     sin_table[pos * head_dim + i] = sin(inv_freq[i] * pos)
//
// position is read via ScratchpadParameter "rope_position" (int32).

#include <aie_api/aie.hpp>
#include <stdint.h>

#define ROPE_VEC_LEN 64  // AIE2P vector length for bf16

using namespace aie;

extern "C" {

// Apply RoPE to a Q/K vector of length d=128 in-place.
// cos_sin_table: [cos_table (max_seq * d) | sin_table (max_seq * d)]
// position: current sequence position (from ScratchpadParameter)
// max_seq: max sequence length (for table indexing)
// d: head_dim (128)
void rope_qk_scratchpad(bfloat16 *restrict qk_vec,
                        const bfloat16 *restrict cos_sin_table,
                        const int32_t position,
                        const int32_t max_seq,
                        const int32_t d)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);

    const int32_t half = d / 2;
    const bfloat16 *cos_row = cos_sin_table + position * d;
    const bfloat16 *sin_row = cos_sin_table + max_seq * d + position * d;

    // d=128: process as two 64-element chunks
    // x = [chunk0 | chunk1] where chunk0 = x[0:64], chunk1 = x[64:128]
    // rotate_half(x) = [-chunk1 | chunk0]
    //   out[0:64] = chunk0 * cos[0:64] - chunk1 * sin[0:64]
    //   out[64:128] = chunk1 * cos[64:128] + chunk0 * sin[64:128]

    aie::vector<bfloat16, ROPE_VEC_LEN> chunk0 = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)qk_vec);
    aie::vector<bfloat16, ROPE_VEC_LEN> chunk1 = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)qk_vec + ROPE_VEC_LEN);
    aie::vector<bfloat16, ROPE_VEC_LEN> cos0 = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)cos_row);
    aie::vector<bfloat16, ROPE_VEC_LEN> cos1 = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)cos_row + ROPE_VEC_LEN);
    aie::vector<bfloat16, ROPE_VEC_LEN> sin0 = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)sin_row);
    aie::vector<bfloat16, ROPE_VEC_LEN> sin1 = aie::load_v<ROPE_VEC_LEN>((bfloat16 *)sin_row + ROPE_VEC_LEN);

    aie::vector<bfloat16, ROPE_VEC_LEN> zeros = aie::broadcast<bfloat16, ROPE_VEC_LEN>((bfloat16)0.0);
    auto neg_chunk1 = aie::sub(zeros, chunk1);

    aie::accum<accfloat, ROPE_VEC_LEN> acc0;
    acc0 = aie::mul(chunk0, cos0);
    acc0 = aie::add(acc0, aie::mul(neg_chunk1, sin0));
    aie::store_v((bfloat16 *)qk_vec, acc0.to_vector<bfloat16>());

    aie::accum<accfloat, ROPE_VEC_LEN> acc1;
    acc1 = aie::mul(chunk1, cos1);
    acc1 = aie::add(acc1, aie::mul(chunk0, sin1));
    aie::store_v((bfloat16 *)qk_vec + ROPE_VEC_LEN, acc1.to_vector<bfloat16>());
}

} // extern "C"
