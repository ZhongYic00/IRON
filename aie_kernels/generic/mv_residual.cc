// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Macroe Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Matrix-vector multiplication with residual add: out = W @ x + residual
// b_in layout: [residual(M) | vec(K)]
// All cols receive the full b_in. row_offset is the LOCAL offset within
// this col's output buffer (0..m_output). global_offset is passed as a
// separate parameter for indexing into the residual array.

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
#define VEC_SIZE 64
#endif

template <uint32_t r, uint32_t k>
void matvec_residual_vec(uint32_t m,
                         uint32_t row_offset,      // local offset in c_out
                         uint32_t global_offset,    // global offset for residual
                         const bfloat16 *__restrict a,
                         const bfloat16 *__restrict b,
                         bfloat16 *__restrict c)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    // b layout: [residual(M) | vec(K)]
    const bfloat16 *__restrict residual = b + global_offset;
    const bfloat16 *__restrict vec = b + DIM_M;  // vec starts after all M residual elements
    const bfloat16 *vec_end = vec + k;

    c += row_offset;
    bfloat16 *c_end = c + m;
    for (; c < c_end; c++, residual++) {
        aie::accum acc = aie::zeros<accfloat, r>();
        AIE_LOOP_MIN_ITERATION_COUNT(k / VEC_SIZE)
        for (const bfloat16 *__restrict v_cur = vec; v_cur < vec_end; v_cur += r, a += r) {
            aie::vector<bfloat16, r> a_vec = aie::load_v<r>(a);
            aie::vector<bfloat16, r> v_vec = aie::load_v<r>(v_cur);
            acc = aie::mac(acc, a_vec, v_vec);
        }
        *c = static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>())) + *residual;
    }
}

extern "C" {

void matvec_residual_vectorized_bf16_bf16(uint32_t m,
                                           uint32_t row_offset,
                                           uint32_t global_offset,
                                           const bfloat16 *__restrict a_in,
                                           const bfloat16 *__restrict b_in,
                                           bfloat16 *__restrict c_out)
{
    matvec_residual_vec<VEC_SIZE, DIM_K>(m, row_offset, global_offset, a_in, b_in, c_out);
}

} // extern "C"
