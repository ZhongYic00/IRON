// FOUR ROWS PER ITERATION (ILP-only) variant of the shipped decode_attn.cc.
//
// SHIPPED as of 2026-09-14 (slice 12).  It replaced the one-row generation,
// kept verbatim as decode_attn_1row.cc (DECODE_ATTN_KERNEL_SOURCE escape hatch).
//
// Why: after the 2026-09-13 soft-float fix the remaining per-(KV row, Q head)
// cost is ~126 ns and the hardware trace says the kernel is instruction/
// dependency bound (INSTR_VECTOR 99.7% at long context).  K = head_dim = 128 is
// only 2 x v64 chunks, so the cross-lane reduce_add (a 6-stage shuffle tree over
// 64 lanes) retires exactly ONE scalar per row and cannot amortise - unlike
// mv_int8's 76-chunk rows.  This variant keeps the reduce shape but keeps FOUR
// independent accumulators / reduce trees in flight, and inherits the packed
// -inf mask fill from decode_attn_nofp.cc (no per-row mask test, no per-row -inf
// store).  Measured: standalone slope -33% (h32) / -46% (h16) per KV block,
// -33% per valid row, op @S_KV=512 -9.2%; in-chain 0.6B A/B/A -0.94% at
// seq_pos=463, -0.83% on gen, -0.37% at the short-context control; BIT-IDENTICAL
// output (maxAbsDiff 0 over S_kv_eff 1..512).
//
// SIBLING VARIANTS (all opt-in via DECODE_ATTN_KERNEL_SOURCE):
//   decode_attn_1row.cc     the previous shipped generation (one row/iteration,
//                           post-softfix) - bit-identical, slower.
//   decode_attn_softfp.cc   pre-2026-09-13 (one software __mulsf3/__gtsf2 per
//                           KV row).
//   decode_attn_nofp.cc     packed exp2 over 32 rows + packed mask fill; NOT
//                           adopted (won at neither context length).
//   decode_attn_4row_int.cc 4-row-INTERLEAVED-K layout (one within-16-lane tree
//                           retires 4 scores); the read-side contract works but
//                           it ties this file within 1-2% while requiring the
//                           block cache layout - see the note §12.3.
//   decode_attn_two_row.cc / decode_attn_vec.cc  older opt-ins (see §4/§9).
//
// Decode-specialised fused attention kernel// Decode-specialised fused attention kernel (single query, M=1) with ONLINE
// softmax streaming.  SHIPPED implementation as of 2026-09-13 (it replaced the
// soft-float-per-row version, which is kept as decode_attn_softfp.cc).
//
// SIBLING VARIANTS (all opt-in, all selected with DECODE_ATTN_KERNEL_SOURCE):
//   decode_attn_softfp.cc   the pre-2026-09-13 shipped form (one software
//                           __mulsf3/__gtsf2 per KV row); escape hatch.
//   decode_attn_two_row.cc  two KV rows per iteration (better ILP, ~25% faster
//                           slope) — reverted in-chain on 9/13 because it pays
//                           ~4% MORE at the short contexts this chain runs in;
//                           revive at max_seq_len ~1024.  NOTE: it was written
//                           against the pre-9/13 one-row kernel and has NOT been
//                           re-derived from this file.
//   decode_attn_vec.cc      slice-2 experiment (register-resident out_acc),
//                           neutral; superseded.
//
// NO SCALAR FP IN THE INNER LOOPS (2026-09-13 attribution, docs/notes/2026-09-13-attn-vec-inner-loop.md):
// AIE2P has no scalar FPU, so every scalar f32 multiply/compare in the shipped
// decode_attn.cc becomes a libgcc soft-float CALL emitted *inside a per-row
// loop*.  `llvm-objdump -r` on the pre-9/13 object counts them:
//   attn_scores_block : 1 __mulsf3   (`s_base[j] = s * scale`, once per KV row)
//   attn_online_block : 2 __gtsf2    (`if (srow[j] > m_block)`, once per row)
//                      + 5 __mulsf3  (D-wide `orow[i] *= rw` + the l/h update)
//   attn_finalize     : 6            (`__divsf3` + D-wide `orow[i] * inv`)
// Differential timing of the shipped kernel against single-change probes put the
// per-row price of those two inner-loop calls at 62 ns (__mulsf3) + 15 ns
// (__gtsf2) out of a 203 ns total per (KV row, Q head) — 38% of the inner loop.
//
// This variant removes BOTH inner-loop calls while keeping the arithmetic in the
// vector domain, where f32 multiply/max are real hardware ops:
//   * scores: store the raw dot product and apply `scale` to the finished block
//     row with B_KV/32 packed f32 multiplies (still inside attn_scores_block,
//     which owns `scale`; plain L1 memory, no MAC/accumulator context).
//   * scores: hoist the q chunks out of the row loop (they do not depend on j).
//   * online: packed block max over the whole B_KV row (lanes [valid,B_KV) hold
//     -inf from the scores pass, so they cannot win the max) and a packed
//     D/32-wise reweight.
// Verified BIT-IDENTICAL to the shipped kernel's output (maxAbsDiff = 0) at
// S_kv_eff = 1, 8, 12, 36, 64, 192, 512 on the 4B shape, full-ELF path.
//
// DEAD ENDS measured in the same session (do not repeat):
//   * packing `attn_finalize`'s D-wide `orow[i] * inv` + `(bfloat16)` cast into
//     vector ops CORRUPTS the output (element shift + 1e38 garbage + NaN) —
//     the bf16 cast/store in the vector domain is miscompiled by the aie2p
//     backend, same family as the 2026-08-31 GEMV-epilogue wall.  Leave
//     attn_finalize's scalar loop alone.
//   * folding `scale` into `q` in bf16 (a vector multiply BEFORE the MAC) is
//     correct but not exact: cos 0.999998 / 0.5% relative vs the shipped kernel.
//
// KV layout: each KV head is stored INTERLEAVED per block —
//   [K_block0 | V_block0 | K_block1 | V_block1 | ...]
// where each block covers B_KV keys.  Each streamed tile is (2*B_KV, D) with
// K_block in rows [0,B_KV) and V_block in rows [B_KV,2*B_KV).

#include <aie_api/aie.hpp>
#include <stdint.h>
#include <limits>

using namespace aie;

// KV row layout selector (compile-time) — unchanged from the shipped kernel.
#ifdef KV_TOKEN_INTERLEAVED
#define K_ROW(j) (((size_t)(j)) * 2 * D)
#define V_ROW(j) (((size_t)(j)) * 2 * D + D)
#else
#define K_ROW(j) (((size_t)(j)) * D)
#define V_ROW(j) ((size_t)(B_KV + (j)) * D)
#endif

// Packed f32 element count (1024-bit AIE vector limit); D = 128 -> 4 chunks.
#define FP32_VEC 32

// Bare-metal scalar exp2 (no libm) — unchanged from the shipped kernel: this is
// a hardware vector op, not a libcall.
static inline float exp2_scalar(float x)
{
    vector<float, 8> vx = broadcast<float, 8>(x);
    vector<bfloat16, 8> e = exp2<bfloat16>(vx);
    return (float)e[0];
}

extern "C" {

// scores[h*B_KV + j] = dot(q[h,:], K_block[j,:]) * scale for this block, with
// positions key_idx = b*B_KV + j >= seq_pos masked to -inf (causal).
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
    float *s_base = scores + h * B_KV;
    const bfloat16 *krow_base = kv;

    // q is invariant across the row loop: load its D_V chunks once.
    vector<bfloat16, VEC> qv[D / VEC];
#pragma unroll
    for (int v = 0; v < D_V; v++)
        qv[v] = load_v<VEC>(qrow + v * VEC);

    // Valid rows of this block (the same clamp attn_online_block uses); masked
    // rows are exactly [nv, B_KV) and are filled with -inf by packed stores, so
    // the row loops below need no mask test and no per-row -inf store.
    int32_t nv = seq_pos - b * B_KV;
    if (nv < 0) nv = 0;
    if (nv > B_KV) nv = B_KV;
    {
        const vector<float, FP32_VEC> ninf =
            broadcast<float, FP32_VEC>(-std::numeric_limits<float>::infinity());
#pragma unroll
        for (int i = 0; i < B_KV / FP32_VEC; i++)
            store_v(s_base + i * FP32_VEC, ninf);
    }

    // FOUR ROWS PER ITERATION, four independent accumulators.  Each row keeps its
    // own 2-chunk mac sequence and its own 64-lane reduce_add, so the per-row
    // result is bit-identical to the one-row kernel; what changes is that four
    // independent 6-stage reduce trees are in flight at once (with one row in
    // flight the tree's latency is fully exposed).
    int32_t j = 0;
    for (; j + 4 <= nv; j += 4) {
        const bfloat16 *k0 = krow_base + K_ROW(j);
        const bfloat16 *k1 = krow_base + K_ROW(j + 1);
        const bfloat16 *k2 = krow_base + K_ROW(j + 2);
        const bfloat16 *k3 = krow_base + K_ROW(j + 3);
        accum<accfloat, VEC> a0 = zeros<accfloat, VEC>();
        accum<accfloat, VEC> a1 = zeros<accfloat, VEC>();
        accum<accfloat, VEC> a2 = zeros<accfloat, VEC>();
        accum<accfloat, VEC> a3 = zeros<accfloat, VEC>();
#pragma unroll
        for (int v = 0; v < D_V; v++) {
            vector<bfloat16, VEC> kv0 = load_v<VEC>(k0 + v * VEC);
            vector<bfloat16, VEC> kv1 = load_v<VEC>(k1 + v * VEC);
            vector<bfloat16, VEC> kv2 = load_v<VEC>(k2 + v * VEC);
            vector<bfloat16, VEC> kv3 = load_v<VEC>(k3 + v * VEC);
            a0 = mac(a0, qv[v], kv0);
            a1 = mac(a1, qv[v], kv1);
            a2 = mac(a2, qv[v], kv2);
            a3 = mac(a3, qv[v], kv3);
        }
        s_base[j] = reduce_add(a0.template to_vector<float>());
        s_base[j + 1] = reduce_add(a1.template to_vector<float>());
        s_base[j + 2] = reduce_add(a2.template to_vector<float>());
        s_base[j + 3] = reduce_add(a3.template to_vector<float>());
    }
    for (; j < nv; j++) {                       // < 4-row tail
        accum<accfloat, VEC> acc = zeros<accfloat, VEC>();
        const bfloat16 *krow = krow_base + K_ROW(j);
#pragma unroll
        for (int v = 0; v < D_V; v++) {
            vector<bfloat16, VEC> kvv = load_v<VEC>(krow + v * VEC);
            acc = mac(acc, qv[v], kvv);
        }
        s_base[j] = reduce_add(acc.template to_vector<float>());
    }

    // Apply `scale` to the finished row with packed f32 multiplies: one soft-float
    // __mulsf3 per row (the shipped form) becomes B_KV/32 vector multiplies per
    // (block, head).  Masked lanes hold -inf and stay -inf.
    {
        const vector<float, FP32_VEC> scv = broadcast<float, FP32_VEC>(scale);
#pragma unroll
        for (int i = 0; i < B_KV / FP32_VEC; i++) {
            vector<float, FP32_VEC> x = load_v<FP32_VEC>(s_base + i * FP32_VEC);
            store_v(s_base + i * FP32_VEC,
                    mul(x, scv).template to_vector<float>());
        }
    }
}

// Online-softmax update + context accumulation for ONE KV block of ONE head.
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
    const bfloat16 *vrow_base = kv;

    int32_t valid = seq_pos - b * B_KV;          // valid lanes in this block
    if (valid < 0) valid = 0;
    if (valid > B_KV) valid = B_KV;

    // Block max over the whole B_KV row: lanes [valid, B_KV) were written as
    // -inf by attn_scores_block (and -inf * scale == -inf), so they cannot win
    // the max; an all-masked block still yields -inf.  One packed max chain
    // replaces the `valid` scalar __gtsf2 calls.
    float m_block;
    {
        vector<float, FP32_VEC> mx = load_v<FP32_VEC>(srow);
#pragma unroll
        for (int i = 1; i < B_KV / FP32_VEC; i++)
            mx = max(mx, load_v<FP32_VEC>(srow + i * FP32_VEC));
        m_block = reduce_max(mx);
    }

    float m_old = m[h];
    float m_new = (m_block > m_old) ? m_block : m_old;

    // reweight factor rw = exp2(m_old - m_new);  == 1.0 when m_old == -inf
    // (first block) or m_new == m_old.
    float rw = exp2_scalar(m_old - m_new);

    float block_sum = 0.0f;
    if (rw != 1.0f) {
        // Packed reweight: the D-wide scalar loop is D soft-float __mulsf3 calls
        // in the shipped kernel (it runs on the first block of every token, and
        // whenever the running max grows).
        const vector<float, FP32_VEC> rv = broadcast<float, FP32_VEC>(rw);
#pragma unroll
        for (int i = 0; i < D / FP32_VEC; i++) {
            vector<float, FP32_VEC> x = load_v<FP32_VEC>(orow + i * FP32_VEC);
            store_v(orow + i * FP32_VEC,
                    mul(x, rv).template to_vector<float>());
        }
    }

    for (int32_t j = 0; j < valid; j++) {
        float e = exp2_scalar(srow[j] - m_new);
        block_sum += e;
        bfloat16 pj = (bfloat16)e;
        vector<bfloat16, VEC> pv = broadcast<bfloat16, VEC>(pj);
        const bfloat16 *vrow = vrow_base + V_ROW(j);
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

// out[h][i] = out_acc[h][i] / l[h].  UNCHANGED from the shipped kernel on
// purpose: packing this loop's multiply + bf16 cast into vector ops corrupts the
// output on aie2p (measured, see the header note).
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
