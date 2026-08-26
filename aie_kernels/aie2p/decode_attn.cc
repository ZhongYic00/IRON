// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Decode-specialised fused attention kernel (single query, M=1).
// Fuses scores-GEMV + softmax + context-GEMV for one query head in L1.
//
// KV layout: each KV head is stored INTERLEAVED per block —
//   [K_block0 | V_block0 | K_block1 | V_block1 | ...]
// where each block covers B_KV keys.  Each streamed tile is (2*B_KV, D) with
// K_block in rows [0,B_KV) and V_block in rows [B_KV,2*B_KV).

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <limits>

using namespace aie;

extern "C" {

// scores[h*S_KV + b*B_KV + j] = dot(q[h,:], K_block[j,:]) * scale.
void attn_scores_block(const bfloat16 *__restrict q,
                       int32_t h,
                       const bfloat16 *__restrict kv,
                       float *__restrict scores,
                       int32_t b,
                       int32_t seq_pos,
                       const float scale)
{
    set_rounding(rounding_mode::conv_even);
    const int D_V = D / VEC;
    const bfloat16 *qrow = q + h * D;
    float *s_base = scores + h * S_KV + b * B_KV;
    const bfloat16 *krow_base = kv;   // K half is rows [0, B_KV)

    for (int32_t j = 0; j < B_KV; j++) {
        int32_t key_idx = b * B_KV + j;
        if (key_idx >= seq_pos) {
            s_base[j] = -std::numeric_limits<float>::infinity();
            continue;
        }
        accum<accfloat, VEC> acc = zeros<accfloat, VEC>();
        const bfloat16 *krow = krow_base + j * D;
        for (int v = 0; v < D_V; v++) {
            vector<bfloat16, VEC> qv = load_v<VEC>(qrow + v * VEC);
            vector<bfloat16, VEC> kvv = load_v<VEC>(krow + v * VEC);
            acc = mac(acc, qv, kvv);
        }
        float s = reduce_add(acc.template to_vector<float>());
        s_base[j] = s * scale;
    }
}

// softmax over pre-scaled scores: p[j] = exp2(s[j] - max); sum = denominator.
// Handles arbitrary seq_pos (not necessarily a multiple of VEC).  Scores at
// [seq_pos, S_KV) are already masked to -inf by attn_scores_block, so we can
// safely round the exp loop up to the next VEC multiple: the -inf lanes
// contribute exp2(-inf) == 0 and keep the whole thing vectorized (the
// bare-metal environment has no scalar exp2f/libm to link against).
void attn_softmax_head(const float *__restrict scores,
                       int32_t h,
                       bfloat16 *__restrict p,
                       float *__restrict sum,
                       int32_t seq_pos)
{
    set_rounding(rounding_mode::conv_even);
    const float *srow = scores + h * S_KV;
    bfloat16 *prow = p + h * S_KV;

    float max_val = -std::numeric_limits<float>::infinity();
    for (int32_t j = 0; j < seq_pos; j++) {
        if (srow[j] > max_val)
            max_val = srow[j];
    }

    // number of lanes actually holding a real (unmasked) score, rounded up to
    // a full VEC chunk so the exp pass stays vectorized.
    int32_t n_lanes = (seq_pos + VEC - 1) / VEC * VEC;
    if (n_lanes == 0)
        n_lanes = VEC;

    vector<float, VEC> maxv = broadcast<float, VEC>(max_val);
    accum<accfloat, VEC> exp_accum = zeros<accfloat, VEC>();
    for (int32_t j = 0; j < n_lanes; j += VEC) {
        vector<float, VEC> sv = load_v<VEC>(srow + j);
        accum<accfloat, VEC> sc = zeros<accfloat, VEC>();
        sc = add(sc, sv);
        accum<accfloat, VEC> e = sub(sc, maxv);
        vector<bfloat16, VEC> ex = exp2<bfloat16>(e.to_vector<float>());
        store_v(prow + j, ex);
        exp_accum = add(exp_accum, ex);
    }
    // mask positions [seq_pos, S_KV) to zero (defensive; context pass only
    // reads [0, seq_pos)).
    for (int32_t m = seq_pos; m < S_KV; m++)
        prow[m] = (bfloat16)0.0f;
    sum[h] = reduce_add(exp_accum.to_vector<float>());
}

// out_acc[h*D + i] += sum_{j in block} p[h][j] * V_block[j,i]
void attn_context_block(const bfloat16 *__restrict p,
                        int32_t h,
                        const bfloat16 *__restrict kv,
                        float *__restrict out_acc,
                        int32_t b)
{
    set_rounding(rounding_mode::conv_even);
    const int D_V = D / VEC;
    const bfloat16 *prow = p + h * S_KV + b * B_KV;
    float *orow = out_acc + h * D;
    const bfloat16 *vrow_base = kv + B_KV * D;   // V half

    for (int32_t j = 0; j < B_KV; j++) {
        bfloat16 pj = prow[j];
        if ((float)pj == 0.0f)
            continue;
        vector<bfloat16, VEC> pv = broadcast<bfloat16, VEC>(pj);
        const bfloat16 *vrow = vrow_base + j * D;
        for (int v = 0; v < D_V; v++) {
            vector<bfloat16, VEC> vv = load_v<VEC>(vrow + v * VEC);
            accum<accfloat, VEC> r = mul(pv, vv);
            vector<float, VEC> acc_old = load_v<VEC>(orow + v * VEC);
            acc_old = add(acc_old, r.to_vector<float>());
            store_v(orow + v * VEC, acc_old);
        }
    }
}

void attn_zero_outacc(float *__restrict out_acc, int32_t h)
{
    set_rounding(rounding_mode::conv_even);
    const int D_V = D / VEC;
    float *orow = out_acc + h * D;
    vector<float, VEC> zv = zeros<float, VEC>();
    for (int v = 0; v < D_V; v++)
        store_v(orow + v * VEC, zv);
}

void attn_finalize(const float *__restrict out_acc,
                   int32_t h,
                   const float *__restrict sum,
                   bfloat16 *__restrict out)
{
    set_rounding(rounding_mode::conv_even);
    const float *orow = out_acc + h * D;
    bfloat16 *out_row = out + h * D;
    float inv = 1.0f / sum[h];
    for (int i = 0; i < D; i++)
        out_row[i] = (bfloat16)(orow[i] * inv);
}

} // extern "C"
