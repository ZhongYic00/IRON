// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// f32 GEMV with fused argmax: computes W @ x, outputs argmax index per column.
// Each kernel call processes m_input rows and updates the running max in c_out.
// c_out[0] = running max value, c_out[1] = running argmax index (as float).
// On first call (row_offset == 0), c_out is initialized to [-inf, 0].
// C ObjectFifo depth=1 ensures same buffer is reused across sub-tile calls.

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
#define VEC_SIZE 32
#endif

template <uint32_t r, uint32_t k>
void matvec_argmax_vec(uint32_t m,
                       uint32_t col_offset,
                       uint32_t row_offset,
                       const float *__restrict a,
                       const float *__restrict b,
                       float *__restrict out)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    const float *b_end = b + k;

    // Read running max from output buffer (or init on first call)
    float local_max;
    int32_t local_argmax;
    if (row_offset == 0) {
        local_max = -1e30f;
        local_argmax = 0;
    } else {
        local_max = out[0];
        local_argmax = (int32_t)out[1];
    }

    for (uint32_t row = 0; row < m; row++) {
        aie::accum<accfloat, r> acc = aie::zeros<accfloat, r>();
        AIE_LOOP_MIN_ITERATION_COUNT(k / VEC_SIZE)
        for (const float *__restrict b_cur = b; b_cur < b_end; b_cur += r, a += r) {
            aie::vector<float, r> a_vec = aie::load_v<r>(a);
            aie::vector<float, r> b_vec = aie::load_v<r>(b_cur);
            acc = aie::mac(acc, a_vec, b_vec);
        }
        float val = aie::reduce_add(acc.template to_vector<float>());
        if (val > local_max) {
            local_max = val;
            local_argmax = col_offset + (int32_t)row;
        }
    }

    // Write back updated running max
    out[0] = local_max;
    out[1] = (float)local_argmax;
}

extern "C" {

void matvec_argmax_f32_f32(uint32_t m,
                           uint32_t col_offset,
                           uint32_t row_offset,
                           const float *__restrict a_in,
                           const float *__restrict b_in,
                           float *__restrict c_out)
{
    matvec_argmax_vec<VEC_SIZE, DIM_K>(m, col_offset, row_offset, a_in, b_in, c_out);
}

} // extern "C"
