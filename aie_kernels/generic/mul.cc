// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

template <typename T_in, typename T_out> void eltwise_mul(T_in *a, T_in *b, T_out *c, int size)
{
    for (int i = 0; i < size; i++) {
        c[i] = a[i] * b[i];
    }
}

template <typename T_in, typename T_out> void eltwise_vmul(T_in *a, T_in *b, T_out *c, int size)
{

    constexpr int vec_factor = 32;
    event0();
    // `size` is a per-core tile the CALLER picks and nothing upstream requires it to divide
    // vec_factor. The bound must be the last FULL vector, not `size`: `i < size` admits a final
    // iteration with fewer than vec_factor elements left, and load_v/store_v are full-width
    // regardless -- so it reads and WRITES up to vec_factor-1 elements PAST the buffer. Silent,
    // and in L1 the memory it lands on is another objectFIFO buffer or the stack.
    //
    // Sibling defect to add.cc's, and the opposite direction: add.cc used a truncating `size/32`
    // bound and silently left the tail UNWRITTEN. Both were invisible because every tile in the
    // Qwen3-0.6B graph divides 32 (d_model/8 = 128, ffn/8 = 384, S/8 = 256); Gemma-3-270M's
    // residual tile is 640/8 = 80 and is what exposed the add.cc half.
    const int F = size / vec_factor;
    for (int i = 0; i < F * vec_factor; i += vec_factor) {
        auto A = aie::load_v<vec_factor>(a + i);
        auto B = aie::load_v<vec_factor>(b + i);
        auto C = aie::mul(A, B).template to_vector<T_out>();
        aie::store_v(c + i, C);
    }
    for (int i = F * vec_factor; i < size; i++) {
        c[i] = a[i] * b[i];
    }
    event1();
}

extern "C" {

void eltwise_mul_bf16_scalar(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, int size)
{
    eltwise_mul<bfloat16, bfloat16>(a_in, b_in, c_out, size);
}
void eltwise_mul_bf16_vector(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, int size)
{
    eltwise_vmul<bfloat16, bfloat16>(a_in, b_in, c_out, size);
}

// Same op, but `a_in`/`b_in` are read starting `ab_offset` elements in. swiglu_mlp_dp calls this
// to emit gh = silu(g)*u in D/N-sized chunks straight out of the core's full FF/N-sized g/u
// buffers, so the chunk can be written directly into the shared output ObjectFifo tile (see this
// design's core_fn) instead of needing a separate gh-sized scratch buffer.
void eltwise_mul_offset_ab_bf16_vector(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, int size,
                                        int ab_offset)
{
    eltwise_vmul<bfloat16, bfloat16>(a_in + ab_offset, b_in + ab_offset, c_out, size);
}
} // extern "C"
