// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Both-offset element copy: `size` bf16 elements from src[src_offset:] to dst[dst_offset:].
//
// add.cc's copy_offset_bf16_vector carries only the DESTINATION offset, which is enough for the
// swiglu_mlp_dp core's gh-reassembly calls (a chunk always arrives at the START of a misc object,
// and only its destination position in the wider buffer varies).  The chunked fuse_o arm needs the
// mirror image: its cx K-chunks come out of the MIDDLE of a D-wide misc object (the two fills that
// keep both taps inside the QD-wide context buffer are [0,D) and [QD-D,QD), so at 4B's
// QD=4096/D=2560 the second chunk's 2048 elements start 512 elements into the second object),
// while the destination is a K_O-wide buffer that always starts at 0.
//
// Translation unit of its own, NOT a second function inside add.cc: kernel variants that share a
// file are the 2026-08-31 build-pollution hazard this tree already documents (a broken #if nesting
// in one variant silently poisoned the default branch of another).  Same body shape as
// copy_offset_bf16_vector -- deliberately not vectorized: it runs twice per token, nowhere near
// this design's weight-bound critical path.

#define NOCPP

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

extern "C" {

void copy_offset_ab_bf16_vector(bfloat16 *dst, bfloat16 *src, int size, int dst_offset,
                                int src_offset)
{
    dst += dst_offset;
    src += src_offset;
    for (int i = 0; i < size; i++) {
        dst[i] = src[i];
    }
}

} // extern "C"
