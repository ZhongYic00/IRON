// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

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

void matvec_scalar(uint32_t m,
                   uint32_t k,
                   const bfloat16 *__restrict a,
                   const bfloat16 *__restrict b,
                   bfloat16 *__restrict c)
{
    for (uint32_t row = 0; row < m; row++) {
        float acc = 0;
        for (uint32_t i = 0; i < k; i++) {
            acc += a[row * k + i] * b[i];
        }
        c[row] = static_cast<bfloat16>(acc);
    }
}

/*
Matrix-vector multiplication kernel

 - m: Number of output rows == number of rows in the input matrix
 - k: Number of columns in the input matrix == length of the input vector
 - a: Pointer to the input matrix, stored in row-major order
 - b: Pointer to the input vector
 - c: Pointer to the output vector
 - r: Vector size; data from the matrix and vector will be loaded in and processed in chunks of this size
*/
template <uint32_t r, uint32_t k>
void matvec_vectorized(uint32_t m, const bfloat16 *__restrict a, const bfloat16 *__restrict b, bfloat16 *__restrict c)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    bfloat16 *c_end = c + m;
    const bfloat16 *b_end = b + k;
    for (; c < c_end; c++) {
        aie::accum acc = aie::zeros<accfloat, r>();
        // The following two pragmas enable pipelining the zero-overhead loop, but they do assume that there are at
        // least two iterations of the loop, i.e. k >= 2*r. This pragma will break the code if that is not the case!
        AIE_LOOP_MIN_ITERATION_COUNT(k / VEC_SIZE)
        for (const bfloat16 *__restrict b_cur = b; b_cur < b_end; b_cur += r, a += r) {
            aie::vector<bfloat16, r> a_vec = aie::load_v<r>(a);
            aie::vector<bfloat16, r> b_vec = aie::load_v<r>(b_cur);
            acc = aie::mac(acc, a_vec, b_vec);
        }
        *c = static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
    }
}

/*
Mixed int8(matrix) x bf16(vector) matvec. The matrix A is int8 (e.g. a quantized resident K/V cache,
halving its LPDDR re-read); the per-TENSOR dequant scale factors out of the dot product, so it is folded
host-side (into the bf16 vector B or the consuming op's weights) and NOT applied here. Each int8 A-chunk is
widened to bf16 (aie::to_float) then MAC'd with the bf16 B-chunk — identical accumulation to the bf16 path.
*/
template <uint32_t r, uint32_t k>
void matvec_vectorized_i8(uint32_t m, const int8 *__restrict a, const bfloat16 *__restrict b, bfloat16 *__restrict c)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    bfloat16 *c_end = c + m;
    const bfloat16 *b_end = b + k;
    for (; c < c_end; c++) {
        aie::accum acc = aie::zeros<accfloat, r>();
        AIE_LOOP_MIN_ITERATION_COUNT(k / VEC_SIZE)
        for (const bfloat16 *__restrict b_cur = b; b_cur < b_end; b_cur += r, a += r) {
            aie::vector<int8, r> a_i8 = aie::load_v<r>(a);
            aie::vector<bfloat16, r> a_vec = aie::to_float<bfloat16>(a_i8, 0);
            aie::vector<bfloat16, r> b_vec = aie::load_v<r>(b_cur);
            acc = aie::mac(acc, a_vec, b_vec);
        }
        *c = static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
    }
}

// GELU (tanh approx) — same math as aie_kernels/aie2p/gelu.cc. In-place over n bf16 elements (n a multiple
// of 16). Called as a GEMV epilogue over the FULL m_output C-tile (NOT the per-call m_input matvec tile —
// m_input can be < 16, which would overrun a 16-wide vector).
static inline void gelu_inplace_bf16(bfloat16 *__restrict v, int32_t n)
{
    const bfloat16 k0_5 = 0.5f, k1 = 1.0f, sqrt_2_over_pi = 0.79788456f, kBeta = 0.044715f;
    auto v05 = aie::broadcast<bfloat16, 16>(k0_5);
    auto v1 = aie::broadcast<bfloat16, 16>(k1);
    auto vs2opi = aie::broadcast<bfloat16, 16>(sqrt_2_over_pi);
    auto vBeta = aie::broadcast<bfloat16, 16>(kBeta);
    auto it = aie::begin_restrict_vector<16>(v);
    for (int i = 0; i < n; i += 16) {
        aie::vector<bfloat16, 16> x = *it;
        aie::vector<bfloat16, 16> x2 = aie::mul(x, x);
        aie::vector<bfloat16, 16> x3 = aie::mul(x, x2);
        aie::vector<bfloat16, 16> x3_beta = aie::mul(x3, vBeta);
        aie::vector<bfloat16, 16> inner = aie::add(x, x3_beta);
        auto inner1 = aie::mul(inner, vs2opi);
        auto tanh_out = aie::tanh<bfloat16>(inner1.to_vector<float>());
        aie::vector<bfloat16, 16> one_plus_tanh = aie::add(tanh_out, v1);
        aie::vector<bfloat16, 16> mul_v05 = aie::mul(v05, one_plus_tanh);
        auto result = aie::mul(x, mul_v05);
        *it++ = result.to_vector<bfloat16>();
    }
}

extern "C" {

// GEMV GELU epilogue: gelu(c) in-place over n elements (the full m_output C-tile). Called once per C-tile in
// the core_body after the matvec inner-loop. n is passed at runtime (= m_output, a multiple of 16).
void gelu_tile_bf16(uint32_t n, bfloat16 *__restrict c)
{
    gelu_inplace_bf16(c, (int32_t)n);
}

/* The row offset parameter in the functions below is a workaround. The output will be written to c + row_offset * m.
 * This is simpler than to do pointer arithmetic in the calling MLIR code, but that's all this is for -- an offset into
 * `c`.  */

void matvec_scalar_bf16_bf16(uint32_t m,
                             uint32_t row_offset,
                             const bfloat16 *__restrict a_in,
                             const bfloat16 *__restrict b_in,
                             bfloat16 *__restrict c_out)
{
    c_out += row_offset;
    matvec_scalar(m, DIM_K, a_in, b_in, c_out);
}

void matvec_vectorized_bf16_bf16(uint32_t m,
                                 uint32_t row_offset,
                                 const bfloat16 *__restrict a_in,
                                 const bfloat16 *__restrict b_in,
                                 bfloat16 *__restrict c_out)
{
    c_out += row_offset;
    matvec_vectorized<VEC_SIZE, DIM_K>(m, a_in, b_in, c_out);
}

// int8 matrix A x bf16 vector B -> bf16 C (per-tensor dequant scale folded host-side; see template above).
// A is passed as a bf16* (the MLIR/fusion arena is bf16-typed) but the BYTES are int8 (2 int8 per bf16
// slot) -> we reinterpret to int8* here. This keeps the MLIR all-bf16 (no element-type-changing
// reinterpret_cast, which the verifier rejects; no i8-arena fusion change). DIM_K = the int8 contract length.
void matvec_vectorized_i8_bf16(uint32_t m,
                               uint32_t row_offset,
                               const bfloat16 *__restrict a_packed,
                               const bfloat16 *__restrict b_in,
                               bfloat16 *__restrict c_out)
{
    c_out += row_offset;
    const int8 *__restrict a_in = reinterpret_cast<const int8 *>(a_packed);
    matvec_vectorized_i8<VEC_SIZE, DIM_K>(m, a_in, b_in, c_out);
}

} // extern "C"