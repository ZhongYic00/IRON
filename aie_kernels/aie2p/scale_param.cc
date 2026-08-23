// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Minimal kernel to verify the ScratchpadParameter end-to-end path:
// reads an int32 scalar "scale_param" from the scratchpad (written by the host
// via ParameterScratchpad each dispatch) and computes out = in * factor, where
// factor = scale_param (broadcast as bf16).  Output therefore changes with the
// host-written value, giving an unambiguous check that the parameter reached
// the core.

#include <aie_api/aie.hpp>
#include <stdint.h>

using namespace aie;

#define VEC_LEN 64

extern "C" {

void scale_by_param(bfloat16 *restrict in,
                    bfloat16 *restrict out,
                    const int32_t factor,
                    const int32_t n_elems)
{
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    bfloat16 f = (bfloat16)(float)factor;
    aie::vector<bfloat16, VEC_LEN> fv = aie::broadcast<bfloat16, VEC_LEN>(f);

    int32_t i = 0;
    for (; i + VEC_LEN <= n_elems; i += VEC_LEN) {
        aie::vector<bfloat16, VEC_LEN> x = aie::load_v<VEC_LEN>(in + i);
        aie::accum<accfloat, VEC_LEN> acc = aie::mul(x, fv);
        aie::store_v(out + i, acc.to_vector<bfloat16>());
    }
    for (; i < n_elems; i++) {
        out[i] = in[i] * f;
    }
}

} // extern "C"
