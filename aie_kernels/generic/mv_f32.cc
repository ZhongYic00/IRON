// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// f32 (float) version of mv.cc for precision-sensitive operations (e.g. lm_head).
// AIE2P float vector size = 32 (vs bf16's 64).

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

#define REL_WRITE 0
#define REL_READ 1

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>

#ifndef VEC_SIZE
#define VEC_SIZE 32  // f32 vector size on AIE2P
#endif

template <uint32_t r, uint32_t k>
void matvec_vectorized_f32(uint32_t m, const float *__restrict a, const float *__restrict b, float *__restrict c)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    float *c_end = c + m;
    const float *b_end = b + k;
    for (; c < c_end; c++) {
        aie::accum<accfloat, r> acc = aie::zeros<accfloat, r>();
        AIE_LOOP_MIN_ITERATION_COUNT(k / VEC_SIZE)
        for (const float *__restrict b_cur = b; b_cur < b_end; b_cur += r, a += r) {
            aie::vector<float, r> a_vec = aie::load_v<r>(a);
            aie::vector<float, r> b_vec = aie::load_v<r>(b_cur);
            acc = aie::mac(acc, a_vec, b_vec);
        }
        *c = aie::reduce_add(acc.template to_vector<float>());
    }
}

extern "C" {

void matvec_vectorized_f32_f32(uint32_t m,
                               uint32_t row_offset,
                               const float *__restrict a_in,
                               const float *__restrict b_in,
                               float *__restrict c_out)
{
    c_out += row_offset;
    matvec_vectorized_f32<VEC_SIZE, DIM_K>(m, a_in, b_in, c_out);
}

} // extern "C"
