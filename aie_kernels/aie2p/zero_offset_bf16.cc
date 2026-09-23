// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Zero a TAIL of a bf16 L1 buffer: buf[offset : offset + size).
//
// Why this file exists: swiglu_mlp_dp's chunked (4B) arm keeps the all-gathered gh
// object FF_PER_CORE wide so that every emit/drain round moves a WHOLE fifo object
// (the fractional-object bug of 2026-09-11), but the gate/up weights only carry the
// model's real FF rows.  The last FF_PER_CORE - FF/N elements of g_buf/u_buf are
// therefore never written by a matvec tile, and they must READ AS ZERO -- which is
// exactly what the row padding of the wire used to provide.  The core zeroes those
// tails once per dispatch, before anything consumes them.
//
// Writes through a BUFFER ARGUMENT, never a kernel-local array: aie::store_v into a
// kernel-local BSS/data array hangs the core in this toolchain (2026-08-29 note); a
// passed-in L1 buffer is the normal pattern this tree uses (cf. fused_swiglu.cc's
// swiglu_zero_out, which is this same loop with a fixed size).
//
// Separate TRANSLATION UNIT rather than a branch in an existing kernel file: the
// 2026-08-31 build-pollution incident (#if nesting that silently corrupted the
// default branch while the source "looked" untouched) is why kernel variants live
// in their own files.

#define NOCPP

#include <stdint.h>

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>

extern "C" {

// size must be a multiple of 64 (one 1024-bit bf16 vector = the widest store_v).
void zero_offset_bf16(bfloat16 *__restrict buf, int32_t size, int32_t offset)
{
    for (int i = 0; i < size; i += 64)
        aie::store_v(buf + offset + i, aie::zeros<bfloat16, 64>());
}

} // extern "C"
