// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// int8-weight GEMV chunk, in-kernel dequant (the shipped premul / VEC128 body
// of mv_int8.cc) WITH an accumulating entry point.
//
// Why this file exists: swiglu_mlp_dp's down projection has K = FF, and at
// Qwen3-4B's shape (D=2560, FF=9728) the all-gathered gh cannot stay in L1 as
// one K-vector -- FF*2 = 19456 B out of a 64 KB L1 is what pushes the whole
// design over budget.  The down matvec is therefore split along K into
// ceil(FF/D) chunks whose K is one D-sized gh chunk, and every chunk after the
// first must ADD into the output row instead of overwriting it.
//
// The chunked wire keeps each chunk's own [m*K u8 | m*(K/128) scales] block
// contiguous, so the body below is mv_int8.cc's byte-for-byte; only the final
// store differs.
//
// Separate TRANSLATION UNIT, not a DEQUANT_MODE branch inside mv_int8.cc: the
// 2026-08-31 build-pollution incident (broken #if nesting that silently
// corrupted the default branch while the source "looked" untouched) is why
// kernel variants live in their own files.
//
//  m:       number of output rows for THIS tile (m_input of the L1 W-tile)
//  a_tile:  chunk payload [m*k int8 weights][m*(k/BLOCK) bf16 scales]
//  b:       (k,) bf16 input vector = this chunk's slice of gh
//  c:       (m,) bf16 output row slice

#define NOCPP

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>

#ifndef VEC_SIZE
#define VEC_SIZE 128
#endif

#ifndef GROUP_SIZE
#define GROUP_SIZE 128
#endif

#ifndef DIM_K
#error "DIM_K (this chunk's own K) must be supplied by the kernel artifact flags"
#endif

constexpr int BLOCK = GROUP_SIZE;

template <bool ACC>
static void matvec_int8_chunk(uint32_t m,
                              uint32_t k,
                              const bfloat16 *__restrict a_tile,
                              const bfloat16 *__restrict b,
                              bfloat16 *__restrict c)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    const int n_groups = k / BLOCK;
    // SIGNED payload (INT8_SIGNED): the +128 bias and its per-weight vector add are gone.
    const int8_t *__restrict a_q = (const int8_t *)a_tile;            // m*k bytes
    const bfloat16 *__restrict a_scl = a_tile + (size_t)m * k / 2;      // m*(k/BLOCK) bf16

    const aie::vector<bfloat16, VEC_SIZE> bias128 =
        aie::broadcast<bfloat16, VEC_SIZE>((bfloat16)-128.0f);

    for (uint32_t row = 0; row < m; row++) {
        const int8_t *__restrict arow = a_q + (size_t)row * k;
        const bfloat16 *__restrict srow = a_scl + (size_t)row * n_groups;
        aie::accum acc = aie::zeros<accfloat, VEC_SIZE>();
        for (int g = 0; g < n_groups; g++) {
            aie::vector<bfloat16, VEC_SIZE> s_vec =
                aie::broadcast<bfloat16, VEC_SIZE>(srow[g]);
            const int8_t *__restrict ablk = arow + (size_t)g * BLOCK;
            const bfloat16 *__restrict bblk = b + (size_t)g * BLOCK;
            for (int i = 0; i < BLOCK; i += VEC_SIZE) {
                aie::vector<bfloat16, VEC_SIZE> b_vec =
                    aie::load_v<VEC_SIZE>(bblk + i);
                aie::vector<bfloat16, VEC_SIZE> xs =
                    aie::mul(b_vec, s_vec).template to_vector<bfloat16>();
                // SIGNED payload: 2 ops per block instead of 3 (to_float -> mac)
                aie::vector<int8_t, VEC_SIZE> q = aie::load_v<VEC_SIZE>(ablk + i);
                aie::vector<bfloat16, VEC_SIZE> w = aie::to_float<bfloat16>(q);
                acc = aie::mac(acc, w, xs);
            }
        }
        float part = aie::reduce_add(acc.template to_vector<float>());
        c[row] = ACC ? static_cast<bfloat16>(static_cast<float>(c[row]) + part)
                     : static_cast<bfloat16>(part);
    }
}

extern "C" {

// c = A @ b (the first K-chunk -- writes the accumulator)
void matvec_vectorized_int8_chunk_bf16(uint32_t m,
                                       uint32_t row_offset,
                                       const bfloat16 *__restrict a_tile,
                                       const bfloat16 *__restrict b_in,
                                       bfloat16 *__restrict c_out)
{
    c_out += row_offset;
    matvec_int8_chunk<false>(m, DIM_K, a_tile, b_in, c_out);
}

// c += A @ b (every later K-chunk)
void matvec_vectorized_int8_chunk_acc_bf16(uint32_t m,
                                           uint32_t row_offset,
                                           const bfloat16 *__restrict a_tile,
                                           const bfloat16 *__restrict b_in,
                                           bfloat16 *__restrict c_out)
{
    c_out += row_offset;
    matvec_int8_chunk<true>(m, DIM_K, a_tile, b_in, c_out);
}

} // extern "C"
