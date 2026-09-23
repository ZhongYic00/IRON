// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

template <typename T, int N>
void rms_norm_general(const T *restrict input,
                      const T *restrict input2,
                      T *restrict output,
                      int32_t cols,
                      float epsilon)
{
    event0();
    ::aie::vector<float, N> add_res = ::aie::zeros<float, N>();

    int vector_chunks = cols / N;
    for (int i = 0; i < vector_chunks; i++) {
        ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
        ::aie::vector<float, N> square_v = ::aie::mul_square(reg_a);
        add_res = ::aie::add(add_res, square_v);
    }
    float sum_sq = ::aie::reduce_add(add_res);

    int remaining = cols % N;
    if (remaining > 0) {
        int start_idx = vector_chunks * N;
        for (int i = 0; i < remaining; i++) {
            T val = input[start_idx + i];
            float square = static_cast<float>(val) * static_cast<float>(val);
            sum_sq += square;
        }
    }

    float rms = sum_sq / cols + epsilon;
    float inv_rms = aie::invsqrt(rms);
    // Normalize in f32 and round once. Mirrors layer_norm.cc's accfloat path.
    ::aie::accum<accfloat, N> inv_rms_v;
    inv_rms_v.from_vector(::aie::broadcast<float, N>(inv_rms), 0);

    for (int i = 0; i < vector_chunks; i++) {
        ::aie::accum<accfloat, N> reg_a;
        reg_a.from_vector(::aie::load_v<N>(input + i * N), 0);
        reg_a = ::aie::mul(reg_a.template to_vector<float>(), inv_rms_v.template to_vector<float>());
        if (input2) {
            ::aie::accum<accfloat, N> reg_b;
            reg_b.from_vector(::aie::load_v<N>(input2 + i * N), 0);
            reg_a = ::aie::mul(reg_a.template to_vector<float>(), reg_b.template to_vector<float>());
        }
        ::aie::store_v(output + i * N, reg_a.template to_vector<T>());
    }

    if (remaining > 0) {
        int start_idx = vector_chunks * N;
        for (int i = 0; i < remaining; i++) {
            T val = input[start_idx + i];
            T norm_val = static_cast<T>(static_cast<float>(val) * inv_rms);
            if (input2) {
                T mul_val = input2[start_idx + i];
                output[start_idx + i] = static_cast<T>(static_cast<float>(norm_val) * static_cast<float>(mul_val));
            } else {
                output[start_idx + i] = norm_val;
            }
        }
    }
    event1();
}

#ifdef RMS_COLS
// Shape-specialized entry point. `cols` is a COMPILE-TIME constant here, which is what removes the
// two scalar-float constructs the runtime-cols form above cannot avoid:
//   * `sum_sq / cols` is a runtime float divide plus an int->float, and those are the ONLY reason
//     __divsf3 (1168 B) and __floatsisf link into a core. Measured on qkv_head_dp: dropping the
//     divide alone took .text 10416 -> 9248 B.
//   * the `cols % N` remainder loops are dead whenever N divides cols -- true for every shape this
//     decode uses (D=1024, HD=128, N=32) -- but they still link, and they carry scalar float math.
//     Removing them took .text 9248 -> 8448 B. Together: -1968 B, 18.9% of that core's program
//     memory, against a 16384 B region.
// Left OPT-IN behind RMS_COLS so the runtime-cols entry point above keeps its exact semantics for
// callers with a non-multiple-of-N size. Same idiom as mv.cc's -DDIM_K.
template <typename T, int N, int COLS>
void rms_norm_fixed(const T *restrict input,
                    const T *restrict input2,
                    T *restrict output,
                    float epsilon)
{
    static_assert(COLS % N == 0, "RMS_COLS must be a multiple of the vector width");
    event0();
    ::aie::vector<float, N> add_res = ::aie::zeros<float, N>();
    for (int i = 0; i < COLS / N; i++) {
        ::aie::vector<T, N> reg_a = ::aie::load_v<N>(input + i * N);
        ::aie::vector<float, N> square_v = ::aie::mul_square(reg_a);
        add_res = ::aie::add(add_res, square_v);
    }
    // COLS is constexpr, so the reciprocal is folded at compile time -- and it is EXACT for the
    // powers of two this rail uses, so this is not a precision change against `sum_sq / cols`.
    constexpr float inv_cols = 1.0f / static_cast<float>(COLS);
    // `* inv_cols + epsilon` runs on the VECTOR unit, not in scalar f32: AIE2P has no scalar FPU,
    // so every scalar float op is a libcall with register spills (62 ns measured for one __mulsf3
    // in decode_attn's inner loop).  A broadcast multiply + add is one VLIW instruction each and
    // rounds identically -- both are IEEE round-to-nearest-even under the conv_even mode set at
    // entry -- so this is bit-identical rather than a precision change.
    ::aie::vector<float, N> m_v =
        ::aie::mul(::aie::broadcast<float, N>(::aie::reduce_add(add_res)),
                   ::aie::broadcast<float, N>(inv_cols)).template to_vector<float>();
    m_v = ::aie::add(m_v, ::aie::broadcast<float, N>(epsilon));
    float inv_rms = aie::invsqrt(m_v[0]);
    ::aie::accum<accfloat, N> inv_rms_v;
    inv_rms_v.from_vector(::aie::broadcast<float, N>(inv_rms), 0);
    for (int i = 0; i < COLS / N; i++) {
        ::aie::accum<accfloat, N> reg_a;
        reg_a.from_vector(::aie::load_v<N>(input + i * N), 0);
        reg_a = ::aie::mul(reg_a.template to_vector<float>(), inv_rms_v.template to_vector<float>());
        if (input2) {
            ::aie::accum<accfloat, N> reg_b;
            reg_b.from_vector(::aie::load_v<N>(input2 + i * N), 0);
            reg_a = ::aie::mul(reg_a.template to_vector<float>(), reg_b.template to_vector<float>());
        }
        ::aie::store_v(output + i * N, reg_a.template to_vector<T>());
    }
    event1();
}
#endif

extern "C" {
// RMS_COLS_ONLY: an object built for a caller that only ever uses the fixed-shape entry
// point (the qkv_head_dp arm) leaves the runtime-cols forms out entirely.  They are what
// drags __divsf3 (1168 B of .text) and __floatsisf into the core -- the divide is
// `sum_sq / cols` with a RUNTIME cols -- and the qkv core has no use for either.
#ifndef RMS_COLS_ONLY
void rms_norm_bf16_vector(bfloat16 *input, bfloat16 *output, int32_t size, float epsilon)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even); // round-to-nearest-even; do not inherit a floor rounding mode
                                                        // from a prior kernel
    rms_norm_general<bfloat16, 32>(input, nullptr, output, size, epsilon);
}
#endif

#ifdef RMS_COLS
// No `size` argument: the shape is RMS_COLS, so there is nothing for a caller to get wrong.
void weighted_rms_norm_fixed(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, float epsilon)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even); // round-to-nearest-even; do not inherit a floor rounding mode
    rms_norm_fixed<bfloat16, 32, RMS_COLS>(a_in, b_in, c_out, epsilon);
}
#endif
#ifndef RMS_COLS_ONLY
void weighted_rms_norm(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, int32_t size, float epsilon)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even); // round-to-nearest-even; do not inherit a floor rounding mode
                                                        // from a prior kernel
    rms_norm_general<bfloat16, 32>(a_in, b_in, c_out, size, epsilon);
}
#endif
}
