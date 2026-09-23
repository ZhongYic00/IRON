// SPDX-FileCopyrightText: Copyright (C) 2025 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#define NOCPP

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <type_traits>

template <typename T_in, typename T_out> void eltwise_add(T_in *a, T_in *b, T_out *c, int size)
{
    for (int i = 0; i < size; i++) {
        c[i] = a[i] + b[i];
    }
}

template <typename T_in, typename T_out> void eltwise_vadd(T_in *a, T_in *b, T_out *c, int size)
{

    constexpr int vec_factor = 32;
    event0();
    T_in *__restrict pA1 = a;
    T_in *__restrict pB1 = b;
    T_out *__restrict pC1 = c;
    const int F = size / vec_factor;
    AIE_PREPARE_FOR_PIPELINING
    AIE_LOOP_MIN_ITERATION_COUNT(16)
    for (int i = 0; i < F; i++) {
        aie::vector<T_in, vec_factor> A0 = aie::load_v<vec_factor>(pA1);
        pA1 += vec_factor;
        aie::vector<T_in, vec_factor> B0 = aie::load_v<vec_factor>(pB1);
        pB1 += vec_factor;
        aie::vector<T_out, vec_factor> cout = aie::add(A0, B0);
        aie::store_v(pC1, cout);
        pC1 += vec_factor;
    }
    // Tail. `size` is a per-core tile size chosen by the caller and nothing upstream requires it to
    // be a multiple of vec_factor, so without this the last `size % 32` outputs are NEVER WRITTEN
    // and the consumer reads whatever the buffer held. Silent, not a crash, and invisible to any
    // model whose tile happens to divide 32: Qwen3-0.6B's residual tile is 1024/8 = 128 = 32*4 and
    // is correct, while Gemma-3-270M's is 640/8 = 80 = 32*2 + 16, dropping 20% of every residual
    // add and turning full-depth decode into garbage tokens.
    const int tail = size - F * vec_factor;   // pA1/pB1/pC1 already point past the vector body
    for (int i = 0; i < tail; i++) {
        pC1[i] = pA1[i] + pB1[i];
    }
    event1();
}

extern "C" {

void eltwise_add_bf16_scalar(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, int size)
{
    eltwise_add<bfloat16, bfloat16>(a_in, b_in, c_out, size);
}

void eltwise_add_bf16_vector(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, int size)
{
    eltwise_vadd<bfloat16, bfloat16>(a_in, b_in, c_out, size);
}

// Same op, but `a_in` is read starting `a_offset` elements in -- the swiglu_mlp_dp core's own
// slice of a REPLICATED full-D buffer (every core holds the whole x1, but the final residual only
// needs its own D/N rows of it). Mirrors mv.cc's row_offset: the offset is a kernel argument, not
// caller-side pointer arithmetic on an acquired ObjectFifo tile (no subview support end to end --
// see the design.py this is called from).
void eltwise_add_offset_a_bf16_vector(bfloat16 *a_in, bfloat16 *b_in, bfloat16 *c_out, int size,
                                       int a_offset)
{
    eltwise_vadd<bfloat16, bfloat16>(a_in + a_offset, b_in, c_out, size);
}

// Plain element copy, `size` elements, writing `dst` starting at `dst_offset`. Used once per
// swiglu_mlp_dp core to reassemble the all-gathered gh vector from misc-fifo chunks smaller than
// FF. Not vectorized -- called 3x per token (D-sized chunks), nowhere near this design's
// bottleneck (the weight stream), so the extra code size of a vector loop isn't worth it.
void copy_offset_bf16_vector(bfloat16 *dst, bfloat16 *src, int size, int dst_offset)
{
    dst += dst_offset;
    for (int i = 0; i < size; i++) {
        dst[i] = src[i];
    }
}

} // extern "C"
