// SPDX-FileCopyrightText: Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Decode-specialised fused attention kernel (single query, M=1) with ONLINE
// softmax streaming.  Unlike the earlier two-pass design (which materialised
// the full (heads_per_col, S_KV) scores and softmax probabilities in L1 and so
// hit the 64KB tile wall past S_KV ~ 2048), this version keeps only a
// per-block scratch and streams KV blocks once, maintaining running max/sum
// (m, l) and a running context accumulator (out_acc) — the FlashAttention-2
// online-softmax recurrence, mirroring mha's partial_softmax/rescale_O.
//
// KV layout: each KV head is stored INTERLEAVED per block —
//   [K_block0 | V_block0 | K_block1 | V_block1 | ...]
// where each block covers B_KV keys.  Each streamed tile is (2*B_KV, D) with
// K_block in rows [0,B_KV) and V_block in rows [B_KV,2*B_KV).

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <limits>

using namespace aie;

// Bare-metal scalar exp2 (no libm): 2^x via the AIE exponent unit on a small
// broadcast vector, since the online recurrence needs a scalar exponential.
// aie2p (AIE2PS) has no f32-precision exp2 (the exp2_bf20 variant requires
// AIE-ML); use aie::exp2<bfloat16> like mha's partial_softmax, which carries the
// running max/sum in bf16 and is validated correct at S_kv=512. bf16 mantissa
// is adequate for the reweight factor m_old - m_new.
static inline float exp2_scalar(float x)
{
    vector<float, 8> vx = broadcast<float, 8>(x);
    vector<bfloat16, 8> e = exp2<bfloat16>(vx);
    return (float)e[0];
}

extern "C" {

// scores[h*B_KV + j] = dot(q[h,:], K_block[j,:]) * scale for this block, with
// positions key_idx = b*B_KV + j >= seq_pos masked to -inf (causal).
// `scores` is a per-block scratch of (heads_per_col, B_KV), NOT the whole row.
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
    // scores is now block-local: h * B_KV
    float *s_base = scores + h * B_KV;
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

// Online-softmax update + context accumulation for ONE KV block of ONE head.
//   m[h], l[h], out_acc[h*D..] are the running state (m=row max in log2e-scaled
//   space, l=sum of exp2(s - m), out_acc=context GEMV numerator).
// Per block:
//   m_new = max(m, max_j scores[j])
//   scale_new = exp2(m_old - m_new)          (reweight factor for old state)
//   out_acc = out_acc * scale_new + sum_j exp2(scores[j]-m_new) * V_block[j,:]
//   l = l * scale_new + sum_j exp2(scores[j]-m_new)
// On the FIRST block m_old is -inf and l_old is 0; the reweight is a no-op.
void attn_online_block(const float *__restrict scores,
                       int32_t h,
                       const bfloat16 *__restrict kv,
                       float *__restrict out_acc,
                       float *__restrict m,
                       float *__restrict l,
                       int32_t b,
                       int32_t seq_pos)
{
    set_rounding(rounding_mode::conv_even);
    const int D_V = D / VEC;
    const float *srow = scores + h * B_KV;
    float *orow = out_acc + h * D;
    const bfloat16 *vrow_base = kv + B_KV * D;   // V half is rows [B_KV, 2*B_KV)

    int32_t valid = seq_pos - b * B_KV;          // valid lanes in this block
    if (valid < 0) valid = 0;
    if (valid > B_KV) valid = B_KV;

    // Block max over valid lanes.
    float m_block = -std::numeric_limits<float>::infinity();
    for (int32_t j = 0; j < valid; j++)
        if (srow[j] > m_block)
            m_block = srow[j];

    float m_old = m[h];
    float m_new = (m_block > m_old) ? m_block : m_old;

    // reweight factor rw = exp2(m_old - m_new);  == 1.0 when m_old == -inf
    // (first block) or m_new == m_old.
    float rw = exp2_scalar(m_old - m_new);

    // Compute this block's exp2(s - m_new) and its sum, while also accumulating
    // the context GEMV numerator with the reweighted old accumulator.
    float block_sum = 0.0f;
    // Vectorized context accumulation: for each row j, pj = exp2(scores[j]-m_new),
    // then out_acc[i] += pj * V_block[j,i].  We first reweight out_acc in place,
    // then accumulate.
    if (rw != 1.0f) {
        for (int i = 0; i < D; i++)
            orow[i] *= rw;
    }

    for (int32_t j = 0; j < valid; j++) {
        float e = exp2_scalar(srow[j] - m_new);
        block_sum += e;
        bfloat16 pj = (bfloat16)e;
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

    // Update running l and m.
    l[h] = l[h] * rw + block_sum;
    m[h] = m_new;
}

// Bare-metal scalar exp2 (no libm): 2^x via the AIE exponent unit on a small
// broadcast vector, since the online recurrence needs a scalar exponential.
// aie2p (AIE2PS) has no f32-precision exp2 (the exp2_bf20 variant requires
// Zero the running accumulator / state for one head before the block loop.
void attn_zero_state(float *__restrict out_acc,
                     float *__restrict m,
                     float *__restrict l,
                     int32_t h)
{
    set_rounding(rounding_mode::conv_even);
    const int D_V = D / VEC;
    float *orow = out_acc + h * D;
    vector<float, VEC> zv = zeros<float, VEC>();
    for (int v = 0; v < D_V; v++)
        store_v(orow + v * VEC, zv);
    m[h] = -std::numeric_limits<float>::infinity();
    l[h] = 0.0f;
}

// out[h][i] = out_acc[h][i] / l[h].
void attn_finalize(const float *__restrict out_acc,
                   int32_t h,
                   const float *__restrict l,
                   bfloat16 *__restrict out)
{
    set_rounding(rounding_mode::conv_even);
    const float *orow = out_acc + h * D;
    bfloat16 *out_row = out + h * D;
    float inv = (l[h] > 0.0f) ? (1.0f / l[h]) : 0.0f;
    for (int i = 0; i < D; i++)
        out_row[i] = (bfloat16)(orow[i] * inv);
}

} // extern "C"
