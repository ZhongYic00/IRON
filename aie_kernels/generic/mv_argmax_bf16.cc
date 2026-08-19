// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// bf16 GEMV with fused argmax — f32 accumulation for precision, VEC_SIZE=64 for speed.
// Inputs are bf16, but dot products use accfloat accumulation (same as standard GEMV).
// argmax compares f32 values — as precise as CPU f32.

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>
#include <math.h>

#define REL_WRITE 0
#define REL_READ 1

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>

#ifndef VEC_SIZE
#define VEC_SIZE 64  // bf16 vector size on AIE2P
#endif

template <uint32_t r, uint32_t k>
void matvec_argmax_vec(uint32_t m,
                       uint32_t col_offset,
                       uint32_t row_offset,
                       const bfloat16 *__restrict a,
                       const bfloat16 *__restrict b,
                       float *__restrict rmax)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    const bfloat16 *b_end = b + k;

    volatile float *rmax_v = rmax;
    float local_max = rmax_v[0];
    int32_t local_argmax = (int32_t)rmax_v[1];

    if (row_offset == 0) {
        local_max = -1e30f;
        local_argmax = 0;
    }

    for (uint32_t row = 0; row < m; row++) {
        aie::accum<accfloat, r> acc = aie::zeros<accfloat, r>();
        AIE_LOOP_MIN_ITERATION_COUNT(k / VEC_SIZE)
        for (const bfloat16 *__restrict b_cur = b; b_cur < b_end; b_cur += r, a += r) {
            aie::vector<bfloat16, r> a_vec = aie::load_v<r>(a);
            aie::vector<bfloat16, r> b_vec = aie::load_v<r>(b_cur);
            acc = aie::mac(acc, a_vec, b_vec);
        }
        float val = aie::reduce_add(acc.template to_vector<float>());
        if (val > local_max) {
            local_max = val;
            local_argmax = col_offset + (int32_t)row;
        }
    }

    rmax_v[0] = local_max;
    rmax_v[1] = (float)local_argmax;
}

extern "C" {

void matvec_argmax_bf16_bf16(uint32_t m,
                             uint32_t col_offset,
                             uint32_t row_offset,
                             const bfloat16 *__restrict a_in,
                             const bfloat16 *__restrict b_in,
                             float *__restrict rmax_out)
{
    matvec_argmax_vec<VEC_SIZE, DIM_K>(m, col_offset, row_offset, a_in, b_in, rmax_out);
}

} // extern "C"
