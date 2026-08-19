// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// f32 GEMV with fused argmax using persistent L1 Buffer for running max.
// Each kernel call processes m_input rows, reads running max from rmax_buf,
// updates it, and writes back. The design copies rmax_buf to output ObjectFifo
// after all sub-tiles are processed.

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
                       float *__restrict rmax)  // persistent L1 buffer [max_val, argmax_idx]
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    const float *b_end = b + k;

    // Read running max from persistent buffer
    volatile float *rmax_v = rmax;
    float local_max = rmax_v[0];
    int32_t local_argmax = (int32_t)rmax_v[1];

    // On first sub-tile (row_offset == 0), re-initialize
    if (row_offset == 0) {
        local_max = -1e30f;
        local_argmax = 0;
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

    // Write back to persistent buffer
    rmax_v[0] = local_max;
    rmax_v[1] = (float)local_argmax;
}

extern "C" {

void matvec_argmax_f32_f32(uint32_t m,
                           uint32_t col_offset,
                           uint32_t row_offset,
                           const float *__restrict a_in,
                           const float *__restrict b_in,
                           float *__restrict rmax_out)
{
    matvec_argmax_vec<VEC_SIZE, DIM_K>(m, col_offset, row_offset, a_in, b_in, rmax_out);
}

} // extern "C"
