//===- mv_gateup_silu.cc --------------------------------------------------===//
//
// Fused MLP first stage (aie2p): paired gate/up weight stream.
//   mlp_gateup(m_half, k, row_off, gu_stride, a_pair, x, gu):
//     gu[row_off : row_off+m_half]        = gate_tile @ x  (premul int8)
//     gu[gu_stride + row_off : +m_half]   = up_tile   @ x
//   mlp_silu_half(gu, gh, m_half):
//     gh[i] = silu(gu[i]) * gu[m_half + i]
//
// matvec_half = the UNCHANGED premul mv_int8 body.  The pair tile =
// [gate w (m*k) | gate s (m*k/G*2) | pad | up w | up s | pad]:
// each half is padded up to the next 64-byte boundary so the up half
// stays 64-byte aligned (aie2p load_v truncates misaligned addresses
// silently — see the 2026-08-31 archive footgun).
//===----------------------------------------------------------------------===//

#define NOCPP

#include <stdint.h>
#include <stdlib.h>

#include "../aie_kernel_utils.h"

#include <aie_api/aie.hpp>

#ifndef VEC_SIZE
#define VEC_SIZE 64
#endif
#ifndef GROUP_SIZE
#define GROUP_SIZE 128
#endif

constexpr int BLOCK = GROUP_SIZE;

static void matvec_half(uint32_t m, uint32_t k, const uint8_t *__restrict a_q,
                        const bfloat16 *__restrict x, bfloat16 *__restrict c_out) {
    const int n_groups = k / BLOCK;
    const bfloat16 *__restrict a_scl =
        (const bfloat16 *)(a_q + (size_t)m * k);

    const aie::vector<bfloat16, VEC_SIZE> bias128 =
        aie::broadcast<bfloat16, VEC_SIZE>((bfloat16)-128.0f);

    for (uint32_t row = 0; row < m; row++) {
        const uint8_t *__restrict arow = a_q + (size_t)row * k;
        const bfloat16 *__restrict srow = a_scl + (size_t)row * n_groups;
        aie::accum acc = aie::zeros<accfloat, VEC_SIZE>();
        for (int g = 0; g < n_groups; g++) {
            aie::vector<bfloat16, VEC_SIZE> s_vec =
                aie::broadcast<bfloat16, VEC_SIZE>(srow[g]);
            const uint8_t *__restrict ablk = arow + (size_t)g * BLOCK;
            const bfloat16 *__restrict bblk = x + (size_t)g * BLOCK;
            for (int i = 0; i < BLOCK; i += VEC_SIZE) {
                aie::vector<bfloat16, VEC_SIZE> b_vec =
                    aie::load_v<VEC_SIZE>(bblk + i);
                aie::vector<bfloat16, VEC_SIZE> xs =
                    aie::mul(b_vec, s_vec).template to_vector<bfloat16>();
                aie::vector<uint8_t, VEC_SIZE> q =
                    aie::load_v<VEC_SIZE>(ablk + i);
                aie::vector<bfloat16, VEC_SIZE> w =
                    aie::to_float<bfloat16>(aie::unpack(q), 0);
                w = aie::add(w, bias128);
                acc = aie::mac(acc, w, xs);
            }
        }
        c_out[row] =
            static_cast<bfloat16>(aie::reduce_add(acc.template to_vector<float>()));
    }
}

extern "C" void mlp_gateup(uint32_t m_half, uint32_t k, uint32_t row_off,
                           uint32_t gu_stride,
                           const bfloat16 *__restrict a_pair,
                           const bfloat16 *__restrict x,
                           bfloat16 *__restrict gu) {
    ::aie::set_rounding(aie::rounding_mode::conv_even);
    // pair tile: [gate w (m*k B) | gate s (m*k/G*2 B) | pad | up w | up s | pad]
    // half span = round_up(m*k + m*k/G*2, 64) bytes — pad to the next 64B
    // boundary so the up half (and every later tile) stays 64B-aligned
    // (aie2p load_v truncates misaligned addresses silently — see the
    // 2026-08-31 archive footgun).  Generic in (m, k): a fixed +32B pad
    // only aligned 0.6B (k=1024); k=2560 landed mid-line at 48.
    const size_t half_bytes = (size_t)m_half * k + (size_t)m_half * (k / GROUP_SIZE) * 2u;
    const size_t half_bf = (half_bytes + 63u) / 64u * 32u;
    matvec_half(m_half, k, (const uint8_t *)a_pair, x, gu + row_off);
    matvec_half(m_half, k, (const uint8_t *)(a_pair + half_bf), x,
                gu + gu_stride + row_off);
}

extern "C" void mlp_silu_half(const bfloat16 *__restrict gu,
                              bfloat16 *__restrict gh, uint32_t m_half) {
    auto it_g = aie::begin_restrict_vector<32>(gu);
    auto it_u = aie::begin_restrict_vector<32>(gu + m_half);
    auto it_o = aie::begin_restrict_vector<32>(gh);

    aie::vector<bfloat16, 16> reg_05_half = aie::broadcast<bfloat16, 16>(0.5f);
    aie::vector<bfloat16, 32> reg_1 = aie::broadcast<bfloat16, 32>(1.0f);
    aie::vector<bfloat16, 32> reg_05_wide = aie::broadcast<bfloat16, 32>(0.5f);

    for (uint32_t i = 0; i < m_half; i += 32) {
        auto A = *it_g++;
        auto B = *it_u++;

        auto half_lo = aie::mul(A.extract<16>(0), reg_05_half);
        auto half_hi = aie::mul(A.extract<16>(1), reg_05_half);
        auto tanh_lo = aie::tanh<bfloat16>(half_lo.to_vector<float>());
        auto tanh_hi = aie::tanh<bfloat16>(half_hi.to_vector<float>());
        auto tanh_half = aie::concat(tanh_lo, tanh_hi);
        auto one_plus = aie::add(tanh_half, reg_1);
        auto sigmoid = aie::mul(one_plus, reg_05_wide)
                           .template to_vector<bfloat16>();
        auto silu_a = aie::mul(A, sigmoid).template to_vector<bfloat16>();

        auto C = aie::mul(silu_a, B).template to_vector<bfloat16>();
        *it_o++ = C;
    }
}
