/*
MIT License

Copyright (c) 2023-2024 The ggml authors (https://github.com/ggml-org/ggml/blob/master/AUTHORS)
Copyright (c) 2023-2024 The llama.cpp authors (https://github.com/ggml-org/llama.cpp/blob/master/AUTHORS)
Copyright (c) 2024-2025 The ik_llama.cpp authors (https://github.com/ikawrakow/ik_llama.cpp/blob/main/AUTHORS)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Extracted/adapted from ik_llama.cpp commit 5bf8f0fe (2026-10-02).
Additional IQ quantization code Copyright (C) 2024 Iwan Kawrakow.
Source ranges: ggml-common.h 401-411, 663-667, 742-750, 2273-2281;
iqk_quantize.cpp 37-43, 1995-2007, 2323-2540, 2564-2597,
2933-2949, 4413-4574, 4601-4622; ggml-quants.c 3974-4011;
iqk_common.h 365-458, 513-558; iqk_gemm_iqk_quants.cpp
924-962, 1096-1140, 1223-1254, 1430-1560.
Quantizers are ik reference implementations with no imatrix/user_data;
QHelper reduces to direct invocation in this case. Default fudge factors
are IQ3_K=1.01 and IQ4_KS=1.0 (ggml.c 2086, 2114).
Original LUT native Q8 loops specialize ik AVX2 GEMM; i16 uses vpmaddwd
and per128 scales. R4 also vendors IQ3_S/IQ4_XS/Q3_K/Q4_K/Q6_K/Q8_0
layouts, default reference quantizers and dequantizers from ggml-quants.c;
source ranges are recorded beside each extraction. Their Q8_K activation
dots adapt the native AVX2 unpack/sign/integer-dot algorithms, sharing each
decoded group across tokens. IQ3_S default quantizer fudge is1.033; the
other R4 defaults are1.0. Its512-grid initialization preserves the exact
source grid map, distance/index tie order and three neighbor shells.
Q4_K affine dots explicitly round both coefficient products separately
before subtraction, preserving exact zero when both components coincide;
this precision safeguard differs from a two-FMA native accumulation.
*/
#include "kernels.h"
#include <immintrin.h>
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <type_traits>
#include <array>
#include <cfloat>
#include <cstdio>
#include <utility>
#if !defined(__AVX2__) || !defined(__FMA__) || !defined(__F16C__)
#error "Compile with -mavx2 -mfma -mf16c -march=broadwell"
#endif
#if defined(__AVX512F__) || defined(__AVX512VNNI__) || defined(__AVXVNNI__)
#error "AVX-512 and VNNI are forbidden in this Broadwell benchmark"
#endif
namespace lutbench {
namespace {
constexpr int QK_K = 256;
using ggml_half = uint16_t;
inline float half_to_float(ggml_half h) { return _cvtsh_ss(h); }
inline ggml_half float_to_half(float f) { return _cvtss_sh(f, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC); }
#define GGML_FP16_TO_FP32 half_to_float
#define GGML_FP32_TO_FP16 float_to_half
#define GGML_ASSERT assert
#define MAX(a,b) std::max(a,b)
#define MIN(a,b) std::min(a,b)
#define MM256_SET1_M128I(a) _mm256_broadcastsi128_si256(a)
#define MM256_SET_M128I(hi,lo) _mm256_inserti128_si256(_mm256_castsi128_si256(lo),hi,1)
struct block_iq3_k {
    ggml_half d; uint16_t extra, scales_h; uint8_t scales_l[8], qs[64], qh[32];
};
struct block_iq4_ks { uint8_t scales[8], qs[128]; };

// R4 ggml IQ formats, ik commit5bf8f0fe ggml-common.h:507-515,646-653.
struct block_iq3_s { ggml_half d; uint8_t qs[64],qh[8],signs[32],scales[4]; };
struct block_iq4_xs { ggml_half d; uint16_t scales_h; uint8_t scales_l[4],qs[128]; };
static_assert(sizeof(block_iq3_s)==110 && sizeof(block_iq4_xs)==136,"ggml IQ layouts");

struct block_q6_K { uint8_t ql[128],qh[64]; int8_t scales[16]; ggml_half d; };
static_assert(sizeof(block_q6_K)==210,"q6_K layout");
struct block_q3_K { uint8_t hmask[32],qs[64],scales[12]; ggml_half d; };
struct block_q4_K { ggml_half d,dmin; uint8_t scales[12],qs[128]; };
static_assert(sizeof(block_q3_K)==110 && sizeof(block_q4_K)==144,"q3_K/q4_K layouts");
struct block_q8_0 { ggml_half d; int8_t qs[32]; };
static_assert(sizeof(block_q8_0)==34,"q8_0 layout");
struct block_q8_K { float d, sum; int8_t qs[256]; int16_t bsums[16]; };
static_assert(sizeof(block_iq3_k)==110 && sizeof(block_iq4_ks)==136 && sizeof(block_q8_K)==296, "ik layouts");
constexpr int8_t iq3nl_values[16] = {-63,-40,-23,-10,1,13,28,47,-59,-36,-19,-6,5,17,32,51};
constexpr int8_t iq4k_values[32] = {-127,-104,-83,-65,-49,-35,-22,-10,1,13,25,38,53,69,89,113,-123,-100,-79,-61,-45,-31,-18,-6,5,17,29,42,57,73,93,117};

inline int nearest_int(float fval) {
    assert(fval <= 4194303.f);
    float val = fval + 12582912.f;
    int i; memcpy(&i, &val, sizeof(int));
    return (i & 0x007fffff) - 0x00400000;
}


const int8_t iq3nl_index[111] = {
  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  8,  8,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  9,
  9,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 10, 10,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3, 11, 11,  4,  4,  4,  4,
  4,  4,  4,  4,  4,  4, 12,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5, 13, 13,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,
  6,  6,  6,  6, 14, 14,  7,  7,  7,  7,  7,  7,  7,  7, 7
};
inline int best_index_iq3nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 111) return ix < 0 ? 0 : 7;
    ix = iq3nl_index[ix];
    return ix < 8 ? ix : x - values[ix-8] < values[ix-7] - x ? ix-8 : ix-7;
}

const int8_t iq4nl_index[241] = {
     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0, 16, 16,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,  1,
     1, 17, 17,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2,  2, 18,  3,  3,  3,  3,  3,  3,  3,  3,  3,  3,
     3,  3,  3,  3,  3,  3, 19,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4,  4, 20,  5,  5,  5,  5,  5,  5,  5,  5,  5,  5,
     5,  5, 21, 21,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6,  6, 22,  7,  7,  7,  7,  7,  7,  7,  7,  7,  7, 23, 23,  8,  8,  8,  8,
     8,  8,  8,  8,  8,  8, 24,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9,  9, 25, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 10, 26, 26,
    11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 11, 27, 27, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 28, 13, 13, 13,
    13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 13, 29, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14, 14,
    14, 14, 14, 14, 30, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15, 15
};
inline int best_index_iq4nl(const int8_t * values, float x) {
    int ix = (int)x - values[0];
    if (ix < 0 || ix >= 241) return ix < 0 ? 0 : 15;
    ix = iq4nl_index[ix];
    return ix < 16 ? ix : x - values[ix-16] < values[ix-15] - x ? ix-16 : ix-15;
}


static void quantize_row_iq3_k_impl(const float * x, void * vy, int n_per_row, const float * quant_weights) {

    constexpr int ntry = 3;

    block_iq3_k * y = (block_iq3_k *)vy;

    float scales[QK_K/16];
    float weight[16];
    uint8_t L[16];

    const int8_t * shifted_values = iq3nl_values + 8;

    const float fudge = 1.01f;

    for (int ibl = 0; ibl < n_per_row/QK_K; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq3_k));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        const float * xbl = x + ibl*QK_K;
        float sumx2 = 0;
        for (int j = 0; j < QK_K; ++j) sumx2 += xbl[j]*xbl[j];
        const float sigma2 = 1.5f*sumx2/QK_K;

        uint16_t extra = 0;

        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {
            const float * xb = xbl + 16*ib;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*QK_K + ib*16;
                for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < 16; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/iq3nl_values[0] : max/iq3nl_values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            float best = 0;
            for (int j = 0; j < 16; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq3nl(iq3nl_values, al);
                float q = iq3nl_values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq3nl(iq3nl_values, -al);
                q = iq3nl_values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            if (sumq2_p > 0) {
                d = sumqx_p/sumq2_p;
                best = d*sumqx_p;
            }
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            bool is_shifted = false;
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (2*itry + iq3nl_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(iq3nl_values, al);
                    float q = iq3nl_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(iq3nl_values, -al);
                    q = iq3nl_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
                }
                id = (2*itry + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq3nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq3nl(shifted_values, -al);
                    q = shifted_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
                }
            }
            if (!d) {
                scales[ib] = 0; continue;
            }

            const int8_t * block_values = is_shifted ? shifted_values : iq3nl_values;
            float sumqx = 0, sumq2 = 0;
            id = 1/d;
            for (int j = 0; j < 16; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq3nl(block_values, al);
                L[j] = l;
                float q = block_values[l];
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
            if (sumq2 > 0) d = sumqx/sumq2;

            float best_d = d;
            for (int iter = 0; iter < 128; ++iter) {
                float gmax = 0;
                int best_j = -1, dir = 0;
                for (int j = 0; j < 16; ++j) {
                    float w = weight[j];
                    float g = d * w * (xb[j] - d*block_values[L[j]]);
                    if (g > 0 && L[j] < 7) {
                        if (g > gmax) {
                            gmax = g; best_j = j; dir = 1;
                        }
                    }
                    else if (g < 0 && L[j] > 0) {
                        if (-g > gmax) {
                            gmax = -g; best_j = j; dir = -1;
                        }
                    }
                }
                if (best_j < 0) break;

                float w = weight[best_j];
                sumqx += w*xb[best_j]*(block_values[L[best_j]+dir] - block_values[L[best_j]]);
                sumq2 += w*(block_values[L[best_j]+dir]*block_values[L[best_j]+dir] - block_values[L[best_j]]*block_values[L[best_j]]);
                L[best_j] += dir;
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    best_d = sumqx/sumq2; best = best_d*sumqx;
                }
                else if (iter > 8) break;

            }

            scales[ib] = d;

            if (is_shifted) extra |= (1 << ib);

            float abs_scale = fabsf(scales[ib]);
            max_abs_scale = MAX(max_abs_scale, abs_scale);
        }

        if (!max_abs_scale) continue;

        float d = max_abs_scale/31;
        y[ibl].extra = extra;
        float id = 1/d;

        float sumqx = 0, sumq2 = 0;
        for (int ib = 0; ib < QK_K/16; ++ib) {
            int ls = nearest_int(0.5f*(id*fabsf(scales[ib])-1));
            ls = MAX(0, MIN(15, ls));
            y[ibl].scales_l[ib/2] |= (ls << 4*(ib%2));
            if (scales[ib] < 0) y[ibl].scales_h |= (1 << ib);
            ls = (2*ls + 1) * (scales[ib] < 0 ? -1 : 1);
            float dl = d * ls;
            if (dl) {
                const int8_t * block_values = y[ibl].extra & (1 << ib) ? shifted_values : iq3nl_values;
                const float * xb = xbl + 16*ib;
                if (quant_weights) {
                    const float * qw = quant_weights + ibl*QK_K + ib*16;
                    for (int j = 0; j < 16; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
                } else {
                    for (int j = 0; j < 16; ++j) weight[j] = 0.25f*sigma2 + xb[j]*xb[j];
                }
                float idl = 1/dl;
                int ib32 = ib/2;
                int offset = 16*(ib%2);
                uint8_t * qs = y[ibl].qs + 32*(ib32/4) + offset;
                uint8_t * qh = y[ibl].qh + 32*(ib32/8) + offset;
                for (int j = 0; j < 16; ++j) {
                    const float al = idl*xb[j];
                    int ibest = best_index_iq3nl(block_values, al);
                    qs[j] |= ((ibest &  3) << 2*(ib32%4));
                    qh[j] |= ((ibest >> 2) << (ib32%8));
                    float w = weight[j];
                    float q = block_values[ibest]*ls;
                    sumqx += w*q*xb[j];
                    sumq2 += w*q*q;
                }
            }
        }
        y[ibl].d = GGML_FP32_TO_FP16(fudge*(sumq2 > 0 ? sumqx/sumq2 : d));

    }
}

static void quantize_row_iq4_k_impl_bs128(const int super_block_size, const int block_size,
        int n_per_row, const float * x, char * cy,
        float * all_scales, float * weight,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    //GGML_ASSERT(super_block_size == 256 && block_size == 128);

    float * dptr = (float *)cy;
    block_iq4_ks * y = (block_iq4_ks *)(dptr + 1);

    const int8_t * shifted_values = values + 16;

    float amax_scale = 0;

    const float fudge = 1.0f;

    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        memset(&y[ibl], 0, sizeof(block_iq4_ks));
        const float * xbl = x + ibl*super_block_size;
        auto scales = all_scales + ibl*(super_block_size/block_size);
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            float amax = 0, max = 0;
            for (int j = 0; j < block_size; ++j) {
                float ax = fabsf(xb[j]);
                if (ax > amax) {
                    amax = ax; max = xb[j];
                }
            }
            if (amax < 1e-16f) {
                scales[ib] = 0;
                continue;
            }
            float d = ntry > 0 ? -max/values[0] : max/values[0];
            float id = 1/d;
            float sumqx_p = 0, sumq2_p = 0;
            float sumqx_m = 0, sumq2_m = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float al = id*xb[j];
                int l = best_index_iq4nl(values, al);
                float q = values[l];
                sumqx_p += w*q*xb[j];
                sumq2_p += w*q*q;
                l = best_index_iq4nl(values, -al);
                q = values[l];
                sumqx_m += w*q*xb[j];
                sumq2_m += w*q*q;
            }
            d = sumqx_p/sumq2_p;
            bool is_shifted = false;
            float best = d*sumqx_p;
            if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                d = sumqx_m/sumq2_m; best = d*sumqx_m;
            }
            for (int itry = -ntry; itry <= ntry; ++itry) {
                id = (itry + values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq4nl(values, al);
                    float q = values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq4nl(values, -al);
                    q = values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = false;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = false;
                }
                id = (itry + shifted_values[0])/max;
                sumqx_p = sumq2_p = 0;
                sumqx_m = sumq2_m = 0;
                for (int j = 0; j < block_size; ++j) {
                    float w = weight[j];
                    float al = id*xb[j];
                    int l = best_index_iq4nl(shifted_values, al);
                    float q = shifted_values[l];
                    sumqx_p += w*q*xb[j];
                    sumq2_p += w*q*q;
                    l = best_index_iq4nl(shifted_values, -al);
                    q = shifted_values[l];
                    sumqx_m += w*q*xb[j];
                    sumq2_m += w*q*q;
                }
                if (sumq2_p > 0 && sumqx_p*sumqx_p > best*sumq2_p) {
                    d = sumqx_p/sumq2_p; best = d * sumqx_p; is_shifted = true;
                }
                if (sumq2_m > 0 && sumqx_m*sumqx_m > best*sumq2_m) {
                    d = sumqx_m/sumq2_m; best = d * sumqx_m; is_shifted = true;
                }
            }
            if (is_shifted) y[ibl].scales[ib] = 0x01;
            scales[ib] = d;
            amax_scale = std::max(amax_scale, std::abs(d));
        }
    }
    float d = amax_scale/127;
    *dptr = fudge*d;
    if (!d) return;
    float id = d ? 1/d : 0.f;
    float sumqx = 0, sumq2 = 0;
    //float mse = 0;
    for (int ibl = 0; ibl < n_per_row/super_block_size; ++ibl) {
        const float * xbl = x + ibl*super_block_size;
        float sigma2 = 0;
        for (int j = 0; j < super_block_size; ++j) sigma2 += xbl[j]*xbl[j];
        sigma2 *= 2.f/super_block_size;
        auto scales = all_scales + (super_block_size/block_size)*ibl;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            const int8_t * block_values = y[ibl].scales[ib] & 0x01 ? shifted_values : values;
            int l = nearest_int(0.5f*(id*scales[ib]+127.f));
            l = std::max(0, std::min(127, l)) << 1;
            //printf("d = %g, id = %g, scales = %g, l = %d, dl = %g\n", d, id, scales[ib], l, d*(l - 127));
            y[ibl].scales[ib] |= l;
            l -= 127;
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            const float * xb = xbl + ib*block_size;
            if (quant_weights) {
                const float * qw = quant_weights + ibl*super_block_size + ib*block_size;
                for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
            } else {
                for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
            }
            auto qs = y[ibl].qs + ib*(block_size/2);
            for (int j = 0; j < block_size/2; ++j) {
                uint8_t i1 = best_index_iq4nl(block_values, idl*xb[j]);
                uint8_t i2 = best_index_iq4nl(block_values, idl*xb[j+block_size/2]);
                qs[j] = i1 | (i2 << 4);
                float w1 = weight[j];
                float w2 = weight[j+block_size/2];
                float q1 = block_values[i1]*l;
                float q2 = block_values[i2]*l;
                sumqx += w1*q1*xb[j] + w2*q2*xb[j+block_size/2];
                sumq2 += w1*q1*q1 + w2*q2*q2;
                //float diff = xb[j] - d*q1; mse += diff*diff;
                //diff = xb[j+block_size/2] - d*q2; mse += diff*diff;
            }
        }
    }
    //printf("rmse = %g\n", sqrt(mse/n_per_row));
    if (sumq2 > 0) *dptr = fudge*sumqx/sumq2;
}

void dequantize_row_iq3_k(const block_iq3_k * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;

        uint16_t sh = x[i].scales_h;
        uint16_t extra = x[i].extra;

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            float dl1 = d * ((2*(x[i].scales_l[ib32] & 0xf) + 1) * ((sh & 1) ? -1 : 1));
            float dl2 = d * ((2*(x[i].scales_l[ib32] >>  4) + 1) * ((sh & 2) ? -1 : 1));
            sh >>= 2;
            const int8_t * values1 = extra & 1 ? iq3nl_values + 8 : iq3nl_values;
            const int8_t * values2 = extra & 2 ? iq3nl_values + 8 : iq3nl_values;
            extra >>= 2;
            int shift_l = 2*(ib32%4);
            int shift_h = ib32%8;
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl1 * values1[((qs[j+ 0] >> shift_l) & 3) | (((qh[j+ 0] >> shift_h) & 1) << 2)];
                y[j+16] = dl2 * values2[((qs[j+16] >> shift_l) & 3) | (((qh[j+16] >> shift_h) & 1) << 2)];
            }
            y += 32;
            if (shift_l == 6) qs += 32;
        }

    }
}

void dequantize_row_iq4_ks(const block_iq4_ks * x, float * y, int64_t k) {
    constexpr int kBlockSize = 32; //128;
    GGML_ASSERT(k%QK_K == 0);
    const float * dptr = (const float *)x;
    float d = *dptr;
    x = (const block_iq4_ks *)(dptr + 1);
    int nblock = k/QK_K;
    for (int ibl = 0; ibl < nblock; ++ibl) {
        auto qs = x[ibl].qs;
        for (int ib = 0; ib < QK_K/kBlockSize; ++ib) {
            float dl = d * ((int)(x[ibl].scales[ib] & 254) - 127);
            const int8_t * values = iq4k_values + ((x[ibl].scales[ib] & 1) << 4);
            for (int j = 0; j < kBlockSize/2; ++j) {
                y[j             ] = dl * values[qs[j] & 0xf];
                y[j+kBlockSize/2] = dl * values[qs[j] >>  4];
            }
            y  += kBlockSize;
            qs += kBlockSize/2;
        }
    }
}

void quantize_row_q8_K_ref(const float * x, block_q8_K * y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        float max = 0;
        float amax = 0;
        for (int j = 0; j < QK_K; ++j) {
            float ax = fabsf(x[j]);
            if (ax > amax) {
                amax = ax; max = x[j];
            }
        }
        if (!amax) {
            y[i].d = 0;
            memset(y[i].qs, 0, QK_K);
            x += QK_K;
            continue;
        }
        //const float iscale = -128.f/max;
        // We need this change for IQ2_XXS, else the AVX implementation becomes very awkward
        const float iscale = -127.f/max;
        for (int j = 0; j < QK_K; ++j) {
            int v = nearest_int(iscale*x[j]);
            y[i].qs[j] = MIN(127, v);
        }
        for (int j = 0; j < QK_K/16; ++j) {
            int sum = 0;
            for (int ii = 0; ii < 16; ++ii) {
                sum += y[i].qs[j*16 + ii];
            }
            y[i].bsums[j] = sum;
        }
        y[i].d = 1/iscale;
        x += QK_K;
    }
}


template <typename Block, bool per_row_scale = false, bool is_f16 = false>
struct BaseDequantizer {
    BaseDequantizer(const void * vx, size_t bx) : vx(vx), bx(bx) {}
    inline void new_row(int ix) {
        if constexpr (per_row_scale) {
            if constexpr (is_f16) {
                const ggml_half * dptr = (const ggml_half *)((const char *)vx + bx*ix);
                d = GGML_FP16_TO_FP32(*dptr);
                x = (const Block *)(dptr + 1);
            } else {
                const float * dptr = (const float *)((const char *)vx + bx*ix);
                d = *dptr;
                x = (const Block *)(dptr + 1);
            }
        } else {
            x = (const Block *)((const char *)vx + bx*ix);
        }
    }

    const void *  vx;
    const size_t  bx;
    const Block * x;

    float d;
};

template <typename Q8, typename Bits>
static inline void multiply_add(const Bits& bits, const __m256i * scales, int j, int i, const Q8& q8, __m256i * sumi) {
    if (j == 0) {
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            const __m256i p1 = _mm256_madd_epi16(scales[0], _mm256_maddubs_epi16(bits.values[0], q8.load_quants(iy, i, 0)));
            const __m256i p2 = _mm256_madd_epi16(scales[1], _mm256_maddubs_epi16(bits.values[1], q8.load_quants(iy, i, 1)));
            const __m256i p3 = _mm256_madd_epi16(scales[2], _mm256_maddubs_epi16(bits.values[2], q8.load_quants(iy, i, 2)));
            const __m256i p4 = _mm256_madd_epi16(scales[3], _mm256_maddubs_epi16(bits.values[3], q8.load_quants(iy, i, 3)));
            sumi[iy] = _mm256_add_epi32(_mm256_add_epi32(p1, p3), _mm256_add_epi32(p2, p4));
        }
    } else {
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            const __m256i p1 = _mm256_madd_epi16(scales[0], _mm256_maddubs_epi16(bits.values[0], q8.load_quants(iy, i, 4)));
            const __m256i p2 = _mm256_madd_epi16(scales[1], _mm256_maddubs_epi16(bits.values[1], q8.load_quants(iy, i, 5)));
            const __m256i p3 = _mm256_madd_epi16(scales[2], _mm256_maddubs_epi16(bits.values[2], q8.load_quants(iy, i, 6)));
            const __m256i p4 = _mm256_madd_epi16(scales[3], _mm256_maddubs_epi16(bits.values[3], q8.load_quants(iy, i, 7)));
            sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p1, p3));
            sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p2, p4));
        }
    }
}

template <typename Q8, typename Bits>
static inline void multiply_add_avx2(const Bits& bits, const __m256i * scales, int j, int i, const Q8& q8, __m256i * sumi) {
    __m256i p[4];
    if (j == 0) {
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            for (int k = 0; k < 4; ++k) {
                auto s = _mm256_sign_epi8(bits.values[k], bits.values[k]);
                p[k] = _mm256_madd_epi16(scales[k], _mm256_maddubs_epi16(s, _mm256_sign_epi8(q8.load_quants(iy, i, k), bits.values[k])));
            }
            sumi[iy] = _mm256_add_epi32(_mm256_add_epi32(p[0], p[1]), _mm256_add_epi32(p[2], p[3]));
        }
    } else {
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            for (int k = 0; k < 4; ++k) {
                auto s = _mm256_sign_epi8(bits.values[k], bits.values[k]);
                p[k] = _mm256_madd_epi16(scales[k], _mm256_maddubs_epi16(s, _mm256_sign_epi8(q8.load_quants(iy, i, 4+k), bits.values[k])));
            }
            sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p[0], p[2]));
            sumi[iy] = _mm256_add_epi32(sumi[iy], _mm256_add_epi32(p[1], p[3]));
        }
    }
}

struct Q2Bits {
    inline void prepare(const uint8_t * q2, int j) {
        auto q2bits = _mm256_loadu_si256((const __m256i *)q2 + j);
        values[0] = _mm256_and_si256(q2bits, ml);
        values[1] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 2), ml);
        values[2] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 4), ml);
        values[3] = _mm256_and_si256(_mm256_srli_epi16(q2bits, 6), ml);
    }
    __m256i values[4];
    const __m256i ml = _mm256_set1_epi8(0x03);
};

struct Q4Bits {
    inline void prepare(const uint8_t * q4, int j) {
        auto q4bits = _mm256_loadu_si256((const __m256i*)q4 + 2*j+0);
        values[0] = _mm256_and_si256(q4bits, ml);
        values[1] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), ml);
        q4bits = _mm256_loadu_si256((const __m256i*)q4 + 2*j+1);
        values[2] = _mm256_and_si256(q4bits, ml);
        values[3] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), ml);
    }
    inline void prepare64(const uint8_t * q4, int j) {
        auto q4bits = _mm256_loadu_si256((const __m256i*)q4 + 2*j+0);
        values[0] = _mm256_and_si256(q4bits, ml);
        values[2] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), ml);
        q4bits = _mm256_loadu_si256((const __m256i*)q4 + 2*j+1);
        values[1] = _mm256_and_si256(q4bits, ml);
        values[3] = _mm256_and_si256(_mm256_srli_epi16(q4bits, 4), ml);
    }
    inline void prepare16(const uint8_t * q4, int j) {
        values[0] = dequant16(q4 + 64*j +  0);
        values[1] = dequant16(q4 + 64*j + 16);
        values[2] = dequant16(q4 + 64*j + 32);
        values[3] = dequant16(q4 + 64*j + 48);
    }
    inline __m256i dequant16(const uint8_t * qs) const {
        // Broadcast both lanes, then select the high nibbles in the upper lane.
        const __m256i packed = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i *)qs));
        return _mm256_and_si256(ml, _mm256_blend_epi32(packed, _mm256_srli_epi16(packed, 4), 0xF0));
    }
    __m256i values[4];
    const __m256i ml = _mm256_set1_epi8(0xf);
};

inline void prepare_scales_16(const __m256i& all_scales, __m256i * scales) {
    const __m128i l_scales = _mm256_extracti128_si256(all_scales, 0);
    const __m128i h_scales = _mm256_extracti128_si256(all_scales, 1);
    scales[0] = MM256_SET1_M128I(l_scales);
    scales[1] = MM256_SET1_M128I(h_scales);
}

struct IQXKScales {
    IQXKScales(int8_t shift, int8_t min_val) : min(_mm256_set1_epi16(min_val)), eshift(_mm_set1_epi8(shift)) {}
    template <typename Q8>
    inline void process(int i, float d, uint16_t extra, __m128i scales8, const Q8& q8, __m256 * accm, __m256i * scales) const {
        auto scales16 = _mm256_cvtepi8_epi16(_mm_shuffle_epi8(scales8, hshuff));
        process(i, d, extra, scales16, q8, accm, scales);
    }
    template <typename Q8>
    inline void process(int i, float d, uint16_t extra, __m256i scales16, const Q8& q8, __m256 * accm, __m256i * scales) const {
        auto extra128 = _mm_set1_epi16(extra);
        extra128 = _mm_cmpeq_epi8(_mm_and_si128(extra128, emask), emask);
        extra128 = _mm_and_si128(extra128, eshift);
        extra128 = _mm_shuffle_epi8(extra128, eshuffle);
        auto scales_s = _mm256_mullo_epi16(scales16, _mm256_add_epi16(min, _mm256_cvtepi8_epi16(extra128)));
        for (int iy = 0; iy < Q8::nrc_y; ++iy) {
            const __m256i prod  = _mm256_madd_epi16(scales_s, q8.load_bsums(iy, i));
            accm[iy] = _mm256_fmadd_ps(_mm256_set1_ps(d * q8.scale(iy, i)), _mm256_cvtepi32_ps(prod), accm[iy]);
        }
        prepare_scales_16(scales16, scales);
    }

    const __m256i min;
    const __m128i eshift;
    const __m128i hshuff   = _mm_set_epi32(0x0f070e06, 0x0d050c04, 0x0b030a02, 0x09010800);
    const __m128i emask    = _mm_set_epi32(0x80804040, 0x20201010, 0x08080404, 0x02020101);
    const __m128i eshuffle = _mm_set_epi32(0x0f0d0b09, 0x07050301, 0x0e0c0a08, 0x06040200);
};

struct DequantizerIQ3K final : public BaseDequantizer<block_iq3_k> {
    DequantizerIQ3K(const void * vx, size_t bx) : BaseDequantizer(vx, bx), iqxk(4, -64), values(load_values()) {}
    template <typename Q8>
    inline void new_block(int i, const Q8& q8, __m256 * accm, __m256i * scales) {
        d = GGML_FP16_TO_FP32(x[i].d);
        iqxk.process(i, d, x[i].extra, make_scales(x[i].scales_h, x[i].scales_l), q8, accm, scales);
        hbits = _mm256_loadu_si256((const __m256i *)x[i].qh);
    }
    inline void prepare(int i, int j) {
        bits.prepare(x[i].qs, j);
        auto h256 = j == 0 ? hbits : _mm256_srli_epi16(hbits, 4);
        bits.values[0] = _mm256_or_si256(bits.values[0], _mm256_and_si256(_mm256_slli_epi16(h256, 2), hmask));
        bits.values[1] = _mm256_or_si256(bits.values[1], _mm256_and_si256(_mm256_slli_epi16(h256, 1), hmask));
        bits.values[2] = _mm256_or_si256(bits.values[2], _mm256_and_si256(h256, hmask));
        bits.values[3] = _mm256_or_si256(bits.values[3], _mm256_and_si256(_mm256_srli_epi16(h256, 1), hmask));
        bits.values[0] = _mm256_shuffle_epi8(values, bits.values[0]);
        bits.values[1] = _mm256_shuffle_epi8(values, bits.values[1]);
        bits.values[2] = _mm256_shuffle_epi8(values, bits.values[2]);
        bits.values[3] = _mm256_shuffle_epi8(values, bits.values[3]);
    }
    static inline __m256i load_values() {
        static const uint8_t kvalues_iq3nl[16] = {1, 24, 41, 54, 65, 77, 92, 111, 5, 28, 45, 58, 69, 81, 96, 115};
        auto val128 = _mm_loadu_si128((const __m128i *)kvalues_iq3nl);
        return MM256_SET1_M128I(val128);
    }
    inline __m128i make_scales(uint16_t signs, const uint8_t * scales_l) const {
        uint64_t aux64; std::memcpy(&aux64, scales_l, 8);
        auto scl = _mm_and_si128(_mm_set_epi64x(aux64 >> 4, aux64), _mm_set1_epi8(0xf));
        scl = _mm_add_epi8(_mm_slli_epi16(scl, 1), m1);
        const __m128i sc_signs = _mm_cmpeq_epi8(_mm_and_si128(_mm_set1_epi16(signs), sign_mask), sign_mask);
        const __m128i sch = _mm_shuffle_epi8(_mm_or_si128(sc_signs, _mm_set1_epi8(1)), hshuff);
        return _mm_sign_epi8(scl, sch);
    }

    Q2Bits bits;
    const IQXKScales iqxk;
    const __m256i values;
    __m256i hbits;
    const __m256i hmask  = _mm256_set1_epi8(4);
    const __m128i m1 = _mm_set1_epi8(1);
    const __m128i sign_mask = _mm_set_epi64x(0x8080404020201010, 0x0808040402020101);
    const __m128i hshuff = _mm_loadu_si128((const __m128i*)k_shuff);
    constexpr static uint8_t k_shuff[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};
};

struct DequantizerIQ4KS final : public BaseDequantizer<block_iq4_ks, true> {
    DequantizerIQ4KS(const void * vx, size_t bx) : BaseDequantizer(vx, bx) { load_values(); }
    template <typename Q8>
    inline __m256i new_block(int i, [[maybe_unused]] const Q8& q8, [[maybe_unused]] __m256 * accd) {
        auto scales128 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i *)x[i].scales));
        scales128 = _mm_add_epi16(_mm_and_si128(scales128, mask), m127);
        return MM256_SET1_M128I(scales128);
    }
    inline void prepare(int i, int j) {
        bits.prepare16(x[i].qs, j);
        bits.values[0] = _mm256_shuffle_epi8(values[x[i].scales[4*j+0] & 1], bits.values[0]);
        bits.values[1] = _mm256_shuffle_epi8(values[x[i].scales[4*j+1] & 1], bits.values[1]);
        bits.values[2] = _mm256_shuffle_epi8(values[x[i].scales[4*j+2] & 1], bits.values[2]);
        bits.values[3] = _mm256_shuffle_epi8(values[x[i].scales[4*j+3] & 1], bits.values[3]);
    }
    void load_values() {
        auto v1 = _mm_loadu_si128((const __m128i *)iq4k_values+0);
        auto v2 = _mm_loadu_si128((const __m128i *)iq4k_values+1);
        values[0] = MM256_SET1_M128I(v1);
        values[1] = MM256_SET1_M128I(v2);
    }


    Q4Bits bits;
    __m256i values[2];
    const __m128i mask     = _mm_set1_epi16(254);
    const __m128i m127     = _mm_set1_epi16(-127);
};

inline __m256i get_scale_shuffle_16(int i) {
    static const uint8_t k_shuffle[128] = {
         0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,     2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3, 2, 3,
         4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5, 4, 5,     6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7, 6, 7,
         8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9, 8, 9,    10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11,
        12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,    14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15,
    };
    return _mm256_loadu_si256((const __m256i*)k_shuffle + i);
}

inline void set_scales_16(const __m256i& all_scales, __m256i * scales) {
    scales[0] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_16(0));
    scales[1] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_16(1));
    scales[2] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_16(2));
    scales[3] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_16(3));
}

inline __m256i get_scale_shuffle_8(int i) {
    return _mm256_set1_epi16((2*i) | ((2*i+1) << 8));
}

inline void set_scales_8(const __m256i& all_scales, int j, __m256i * scales) {
    scales[0] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_8(4*j+0));
    scales[1] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_8(4*j+1));
    scales[2] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_8(4*j+2));
    scales[3] = _mm256_shuffle_epi8(all_scales, get_scale_shuffle_8(4*j+3));
}

struct NativeQ8 {
    constexpr static int nrc_y=1;
    const block_q8_K *y;
    explicit NativeQ8(const Activation &a) : y(reinterpret_cast<const block_q8_K *>(a.q8.data())) {}
    inline __m256i load_quants(int, int i, int j) const { return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[i].qs)+j); }
    inline __m256i load_bsums(int, int i) const { return _mm256_loadu_si256(reinterpret_cast<const __m256i *>(y[i].bsums)); }
    inline float scale(int,int i) const { return y[i].d; }
};
inline float horizontal(__m256 x) {
    __m128 t=_mm_add_ps(_mm256_castps256_ps128(x),_mm256_extractf128_ps(x,1));
    t=_mm_add_ps(t,_mm_movehl_ps(t,t));
    return _mm_cvtss_f32(_mm_add_ss(t,_mm_shuffle_ps(t,t,1)));
}

// R4 grid/quantizer/reference extraction: ggml-common.h:1417ff;
// ggml-quants.c:3811-3851,3949-3971,13668-13691,13933-14114,14793-14978.
alignas(64) constexpr uint32_t iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

// Adapted from ggml-quants.c:13502-13656, retaining the same grid map,
// distance/index tie order and the nearest3 distinct distance shells.
// The source computes neighbor counts in a first pass; vectors let us build
// the same serialized table in one pass. Initialization is thread-safe.
struct IQ3SQuantTables {
    std::array<int,4096> map;
    std::vector<uint16_t> neighbours;
    IQ3SQuantTables() {
        map.fill(-1);
        for(int g=0;g<512;++g) {
            const auto *v=reinterpret_cast<const uint8_t *>(iq3s_grid+g);
            unsigned index=0;
            for(int j=0;j<4;++j) index|=unsigned((v[j]-1)/2)<<(3*j);
            map[index]=g;
        }
        std::array<std::pair<int,int>,512> distances;
        for(int point=0;point<4096;++point) {
            if(map[point]>=0) continue;
            for(int g=0;g<512;++g) {
                const auto *v=reinterpret_cast<const uint8_t *>(iq3s_grid+g);
                int d2=0;
                for(int j=0;j<4;++j) {
                    const int delta=int(v[j])-(2*((point>>(3*j))&7)+1);
                    d2+=delta*delta;
                }
                distances[g]={d2,g};
            }
            std::sort(distances.begin(),distances.end());
            const size_t offset=neighbours.size();
            map[point]=-int(offset+1); neighbours.push_back(0);
            int distance=distances[0].first,shells=1;
            for(const auto &entry:distances) {
                if(entry.first>distance) {
                    if(shells==3) break;
                    distance=entry.first; ++shells;
                }
                neighbours.push_back(uint16_t(entry.second));
                ++neighbours[offset];
            }
        }
    }
};
const IQ3SQuantTables &iq3_s_quant_tables() {
    static const IQ3SQuantTables tables;
    return tables;
}
static int iq3_find_best_neighbour(const uint16_t * neighbours, const uint32_t * grid,
        const float * xval, const float * weight, float scale, int8_t * L) {
    int num_neighbors = neighbours[0];
    GGML_ASSERT(num_neighbors > 0);
    float best_d2 = FLT_MAX;
    int grid_index = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *)(grid + neighbours[j]);
        float d2 = 0;
        for (int i = 0; i < 4; ++i) {
            float q = pg[i];
            float diff = scale*q - xval[i];
            d2 += weight[i]*diff*diff;
        }
        if (d2 < best_d2) {
            best_d2 = d2; grid_index = neighbours[j];
        }
    }
    GGML_ASSERT(grid_index >= 0);
    const int8_t * pg = (const int8_t *)(grid + grid_index);
    for (int i = 0; i < 4; ++i) L[i] = (pg[i] - 1)/2;
    return grid_index;
}
static void quantize_row_iq3_s_impl(int block_size, const float * x, void * vy, int n,
        const float * quant_weights,
        float   * scales,
        float   * weight,
        float   * xval,
        int8_t  * L,
        int8_t  * Laux,
        float   * waux,
        bool    * is_on_grid,
        bool    * is_on_grid_aux,
        uint8_t * block_signs) {

    const auto &tables=iq3_s_quant_tables();
    const uint32_t *kgrid_q3xs=iq3s_grid;
    const int *kmap_q3xs=tables.map.data();
    const uint16_t *kneighbors_q3xs=tables.neighbours.data();

    //GGML_ASSERT(quant_weights   && "missing quantization weights");
    GGML_ASSERT(kgrid_q3xs      && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kmap_q3xs       && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(kneighbors_q3xs && "forgot to call ggml_quantize_init()?");
    GGML_ASSERT(n%QK_K == 0);

    const int kMaxQ = 8;

    const int64_t nbl = n/QK_K;

    block_iq3_s * y = static_cast<block_iq3_s *>(vy);

    const int bs4 = block_size/4;
    const int bs8 = block_size/8;

    const float fudge = 1.033f;

    for (int ibl = 0; ibl < nbl; ++ibl) {

        memset(&y[ibl], 0, sizeof(block_iq3_s));
        y[ibl].d = GGML_FP32_TO_FP16(0.f);

        uint8_t * qs = y[ibl].qs;
        uint8_t * qh = y[ibl].qh;
        uint8_t * signs = y[ibl].signs;

        float max_scale = 0;

        const float * xbl = x + QK_K*ibl;
        float sumx2 = 0;
        for (int i = 0; i < QK_K; ++i) sumx2 += xbl[i]*xbl[i];
        float sigma2 = 2*sumx2/QK_K;

        for (int ib = 0; ib < QK_K/block_size; ++ib) {
            const float * xb = xbl + block_size*ib;
            if (quant_weights) {
                const float * qw = quant_weights + QK_K*ibl + block_size*ib;
                for (int i = 0; i < block_size; ++i) weight[i] = qw[i] * sqrtf(sigma2 + xb[i]*xb[i]);
            } else {
                for (int i = 0; i < block_size; ++i) weight[i] = xb[i]*xb[i];
            }
            for (int i = 0; i < block_size; ++i) waux[i] = sqrtf(weight[i]);
            for (int k = 0; k < bs8; ++k) {
                uint8_t s = 0;
                for (int i = 0; i < 8; ++i) {
                    if (xb[8*k + i] >= 0) xval[8*k + i] = xb[8*k + i];
                    else {
                        xval[8*k + i] = -xb[8*k + i]; s |= (1 << i);
                    }
                }
                block_signs[k] = s;
            }
            float max = xval[0];
            for (int i = 1; i < block_size; ++i) max = MAX(max, xval[i]);
            if (!max) {
                scales[ib] = 0;
                continue;
            }
            float best = 0;
            float scale = max/(2*kMaxQ-1);
            for (int k = 0; k < bs4; ++k) is_on_grid[k] = false;
            for (int is = -9; is <= 9; ++is) {
                float id = (2*kMaxQ-1+is*0.2f)/max;
                float this_scale = 1/id;
                for (int k = 0; k < bs4; ++k) {
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        Laux[4*k+i] = MAX(0, MIN(kMaxQ-1, l));
                    }
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) u |= (Laux[4*k+i] << 3*i);
                    int grid_index = kmap_q3xs[u];
                    is_on_grid_aux[k] = true;
                    if (grid_index < 0) {
                        is_on_grid_aux[k] = false;
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, this_scale, Laux + 4*k);
                    }
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < block_size; ++i) {
                    float w = weight[i];
                    float q = 2*Laux[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                    scale = sumqx/sumq2; best = scale*sumqx;
                    for (int i = 0; i < block_size; ++i) L[i] = Laux[i];
                    for (int k = 0; k < bs4; ++k) is_on_grid[k] = is_on_grid_aux[k];
                }
            }
            int n_not_ongrid = 0;
            for (int k = 0; k < bs4; ++k) if (!is_on_grid[k]) ++n_not_ongrid;
            if (n_not_ongrid > 0 && scale > 0) {
                float id = 1/scale;
                for (int k = 0; k < bs4; ++k) {
                    //if (is_on_grid[k]) continue;
                    uint16_t u = 0;
                    for (int i = 0; i < 4; ++i) {
                        int l = nearest_int(0.5f*(id*xval[4*k+i]-1));
                        l = MAX(0, MIN(kMaxQ-1, l));
                        u |= (l << 3*i);
                    }
                    int grid_index = kmap_q3xs[u];
                    if (grid_index < 0) {
                        const uint16_t * neighbours = kneighbors_q3xs - kmap_q3xs[u] - 1;
                        grid_index = iq3_find_best_neighbour(neighbours, kgrid_q3xs, xval + 4*k, waux + 4*k, scale, L + 4*k);
                    }
                    const int8_t * pg = (const int8_t *)(kgrid_q3xs + grid_index);
                    for (int i = 0; i < 4; ++i) L[4*k+i] = (pg[i] - 1)/2;
                }
                float sumqx = 0, sumq2 = 0;
                for (int i = 0; i < block_size; ++i) {
                    float w = weight[i];
                    float q = 2*L[i] + 1;
                    sumqx += w*xval[i]*q;
                    sumq2 += w*q*q;
                }
                if (sumq2 > 0) scale = sumqx/sumq2;
            }
            if (scale < 0) {
                // This should never happen, but just in case, flip scale so that it is positive (we use uint's to encode the scale)
                // and correspondingly flip quant signs.
                scale = -scale;
                for (int k = 0; k < bs8; ++k) block_signs[k] = ~block_signs[k];
            }
            for (int k = 0; k < bs4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) u |= (L[4*k+i] << 3*i);
                int grid_index = kmap_q3xs[u];
                if (grid_index < 0) {
                    printf("Oops: found point %u not on grid:", u);
                    for (int i = 0; i < 4; ++i) printf(" %d", L[4*k+i]);
                    printf("\n");
                    throw std::runtime_error("reference quantizer produced an off-grid index");
                }
                qs[k] = grid_index & 255;
                qh[(ib*bs4+k)/8] |= ((grid_index >> 8) << ((ib*bs4+k)%8));
            }
            qs += bs4;
            for (int k = 0; k < bs8; ++k) signs[k] = block_signs[k];
            signs += bs8;
            GGML_ASSERT(scale >= 0);
            scales[ib] = scale;
            max_scale = MAX(max_scale, scale);
        }

        if (!max_scale) {
            continue;
        }

        float d = max_scale/31;
        y[ibl].d = GGML_FP32_TO_FP16(d * fudge);
        float id = 1/d;
        for (int ib = 0; ib < QK_K/block_size; ib += 2) {
            int l1 = nearest_int(0.5f*(id*scales[ib+0]-1));
            l1 = MAX(0, MIN(15, l1));
            int l2 = nearest_int(0.5f*(id*scales[ib+1]-1));
            l2 = MAX(0, MIN(15, l2));
            y[ibl].scales[ib/2] = l1 | (l2 << 4);
        }

    }
}
static void quantize_row_iq4_nl_impl(const int super_block_size, const int block_size, const float * x,
        ggml_half * dh, uint8_t * q4, uint16_t * scales_h, uint8_t * scales_l,
        float * scales, float * weight, uint8_t * L,
        const int8_t * values,
        const float * quant_weights,
        const int ntry) {

    float sigma2 = 0;
    for (int j = 0; j < super_block_size; ++j) sigma2 += x[j]*x[j];
    sigma2 *= 2.f/super_block_size;

    memset(q4, 0, super_block_size/2);
    dh[0] = GGML_FP32_TO_FP16(0.f);

    const float fudge = 1.0f;

    float max_scale = 0, amax_scale = 0;
    for (int ib = 0; ib < super_block_size/block_size; ++ib) {
        const float * xb = x + ib*block_size;
        uint8_t * Lb = L + ib*block_size;
        if (quant_weights) {
            const float * qw = quant_weights + ib*block_size;
            for (int j = 0; j < block_size; ++j) weight[j] = qw[j] * sqrtf(sigma2 + xb[j]*xb[j]);
        } else {
            for (int j = 0; j < block_size; ++j) weight[j] = xb[j]*xb[j];
        }
        float amax = 0, max = 0;
        for (int j = 0; j < block_size; ++j) {
            float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax; max = xb[j];
            }
        }
        if (amax < 1e-15f) {
            scales[ib] = 0;
            continue;
        }
        float d = ntry > 0 ? -max/values[0] : max/values[0];
        float id = 1/d;
        float sumqx = 0, sumq2 = 0;
        for (int j = 0; j < block_size; ++j) {
            float al = id*xb[j];
            int l = best_index_iq4nl(values, al);
            Lb[j] = l;
            float q = values[l];
            float w = weight[j];
            sumqx += w*q*xb[j];
            sumq2 += w*q*q;
        }
        d = sumqx/sumq2;
        float best = d*sumqx;
        float best_sumqx = sumqx, best_sumq2 = sumq2;
        for (int itry = -ntry; itry <= ntry; ++itry) {
            id = (itry + values[0])/max;
            sumqx = sumq2 = 0;
            for (int j = 0; j < block_size; ++j) {
                float al = id*xb[j];
                int l = best_index_iq4nl(values, al);
                float q = values[l];
                float w = weight[j];
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
            if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                d = sumqx/sumq2; best = d * sumqx;
                best_sumqx = sumqx; best_sumq2 = sumq2;
                for (int j = 0; j < block_size; ++j) {
                    float al = id*xb[j];
                    Lb[j] = best_index_iq4nl(values, al);
                }
            }
            id = (itry + values[15])/max;
            sumqx = sumq2 = 0;
            for (int j = 0; j < block_size; ++j) {
                float al = id*xb[j];
                int l = best_index_iq4nl(values, al);
                float q = values[l];
                float w = weight[j];
                sumqx += w*q*xb[j];
                sumq2 += w*q*q;
            }
            if (sumq2 > 0 && sumqx*sumqx > best*sumq2) {
                d = sumqx/sumq2; best = d * sumqx;
                best_sumqx = sumqx; best_sumq2 = sumq2;
                for (int j = 0; j < block_size; ++j) {
                    float al = id*xb[j];
                    Lb[j] = best_index_iq4nl(values, al);
                }
            }
        }
        sumqx = best_sumqx; sumq2 = best_sumq2;
        for (int iter = 0; iter < 32*block_size; ++iter) {
            float min_step = INFINITY;
            int best_j = -1; int dir = 0;
            for (int j = 0; j < block_size; ++j) {
                float w = weight[j];
                float g = d * w * (xb[j] - d*values[Lb[j]]);
                if (g > 0 && Lb[j] < 15) {
                    float step = (values[Lb[j]+1] - values[Lb[j]])/g;
                    if (step < min_step) {
                        min_step = step; best_j = j; dir = 1;
                    }
                }
                else if (g < 0 && Lb[j] > 0) {
                    float step = (values[Lb[j]-1] - values[Lb[j]])/g;
                    if (step < min_step) {
                        min_step = step; best_j = j; dir = -1;
                    }
                }
            }
            if (best_j < 0) break;

            float new_sumqx = sumqx, new_sumq2 = sumq2;
            float w = weight[best_j];
            new_sumqx += w*xb[best_j]*(values[Lb[best_j]+dir] - values[Lb[best_j]]);
            new_sumq2 += w*(values[Lb[best_j]+dir]*values[Lb[best_j]+dir] - values[Lb[best_j]]*values[Lb[best_j]]);
            if (new_sumq2 > 0 && new_sumqx*new_sumqx > best*new_sumq2) {
                sumqx = new_sumqx; sumq2 = new_sumq2;
                d = sumqx/sumq2; best = d*sumqx;
                Lb[best_j] += dir;
            }
            else {
                break;
            }
        }

        scales[ib] = d;
        float abs_d = fabsf(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d; max_scale = d;
        }
    }

    if (super_block_size/block_size > 1) {
        int nb = super_block_size/block_size;
        memset(scales_h, 0, ((nb+7)/8)*sizeof(uint16_t));
        float d = -max_scale/32;
        dh[0] = GGML_FP32_TO_FP16(d*fudge);
        float id = d ? 1/d : 0.f;
        for (int ib = 0; ib < super_block_size/block_size; ++ib) {
            int l = nearest_int(id*scales[ib]);
            l = MAX(-32, MIN(31, l));
            float dl = d * l;
            float idl = dl ? 1/dl : 0.f;
            uint8_t * Lb = L + ib*block_size;
            const float * xb = x + ib*block_size;
            for (int j = 0; j < block_size; ++j) {
                Lb[j] = best_index_iq4nl(values, idl*xb[j]);
            }
            l += 32;
            uint8_t l_l = l & 0xf;
            uint8_t l_h = l >>  4;
            if (ib%2 == 0) scales_l[ib/2] = l_l;
            else scales_l[ib/2] |= (l_l << 4);
            scales_h[ib/8] |= (l_h << 2*(ib%8));
        }
    } else {
        dh[0] = GGML_FP32_TO_FP16(scales[0] * fudge);
        if (ntry > 0) {
            float id = scales[0] ? 1/scales[0] : 0;
            for (int j = 0; j < super_block_size; ++j) {
                L[j] = best_index_iq4nl(values, id*x[j]);
            }
        }
    }

    for (int i = 0; i < super_block_size/32; ++i) {
        for (int j = 0; j < 16; ++j) {
            q4[16*i + j] = L[32*i + j] | (L[32*i + 16 + j] << 4);
        }
    }
}
void dequantize_row_iq3_s(const block_iq3_s * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const uint8_t * qs = x[i].qs;
        const uint8_t * qh = x[i].qh;
        const uint8_t * signs = x[i].signs;

        for (int ib32 = 0; ib32 < QK_K/32; ib32 += 2) {
            const float db1 = d * (1 + 2*(x[i].scales[ib32/2] & 0xf));
            const float db2 = d * (1 + 2*(x[i].scales[ib32/2] >>  4));
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[0] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[0] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db1 * grid1[j] * (signs[l] & (1u<<(j+0)) ? -1.f : 1.f);
                    y[j+4] = db1 * grid2[j] * (signs[l] & (1u<<(j+4)) ? -1.f : 1.f);
                }
                y += 8;
            }
            qs += 8;
            signs += 4;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid1 = (const uint8_t *)(iq3s_grid + (qs[2*l+0] | ((qh[1] << (8-2*l)) & 256)));
                const uint8_t * grid2 = (const uint8_t *)(iq3s_grid + (qs[2*l+1] | ((qh[1] << (7-2*l)) & 256)));
                for (int j = 0; j < 4; ++j) {
                    y[j+0] = db2 * grid1[j] * (signs[l] & (1u<<(j+0)) ? -1.f : 1.f);
                    y[j+4] = db2 * grid2[j] * (signs[l] & (1u<<(j+4)) ? -1.f : 1.f);
                }
                y += 8;
            }
            qh += 2;
            qs += 8;
            signs += 4;
        }
    }
}
void dequantize_row_iq4_xs(const block_iq4_xs * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const uint8_t * qs = x[i].qs;

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int ib = 0; ib < QK_K/32; ++ib) {
            const int ls = ((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4);
            const float dl = d * (ls - 32);
            for (int j = 0; j < 16; ++j) {
                y[j+ 0] = dl * iq4k_values[qs[j] & 0xf];
                y[j+16] = dl * iq4k_values[qs[j] >>  4];
            }
            y  += 32;
            qs += 16;
        }
    }
}


// Q6_K reference extraction ggml-quants.c:1733-1861,3171-3273.
static float make_qx_quants(int n, int nmax, const float * x, int8_t * L, int rmse_type,
        const float * qw) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < 1e-15f) { // all zero
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
        }
        return 1/iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
        float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
        sumlx += w*x[i]*l;
        suml2 += w*l*l;
    }
    float scale = suml2 ? sumlx/suml2 : 0.0f;
    if (return_early) return suml2 > 0 ? 0.5f*(scale + 1/iscale) : 1/iscale;
    float best = scale * sumlx;
    float best_sumlx = sumlx, best_suml2 = suml2;
    for (int is = -9; is <= 9; ++is) {
        iscale = -(nmax + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
            best_sumlx = sumlx; best_suml2 = suml2;
        }
        iscale = (nmax-1 + 0.1f*is) / max;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            float w = qw ? qw[i] : rmse_type == 1 ? x[i] * x[i] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[i]) : sqrtf(fabsf(x[i]));
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        if (suml2 > 0 && sumlx*sumlx > best*suml2) {
            for (int i = 0; i < n; ++i) {
                int l = nearest_int(iscale * x[i]);
                L[i] = nmax + MAX(-nmax, MIN(nmax-1, l));
            }
            scale = sumlx/suml2; best = scale*sumlx;
            best_sumlx = sumlx; best_suml2 = suml2;
        }
    }

    sumlx = best_sumlx; suml2 = best_suml2;
    for (int iter = 0; iter < n*(2*nmax-1); ++iter) {
        float abs_gmax = 0, gmax = 0;
        int best_j = -1;
        for (int j = 0; j < n; ++j) {
            float w = qw ? qw[j] : rmse_type == 1 ? x[j] * x[j] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[j]) : sqrtf(fabsf(x[j]));
            int l = L[j] - nmax;
            float g = scale * w * (x[j] - scale*l);
            if ((g > 0 && l < nmax-1) || (g < 0 && l > -nmax)) {
                float ag = fabsf(g);
                if (ag > abs_gmax) {
                    abs_gmax = ag; gmax = g; best_j = j;
                }
            }
        }
        if (best_j < 0) break;

        float new_sumlx = sumlx, new_suml2 = suml2;
        float w = qw ? qw[best_j] : rmse_type == 1 ? x[best_j] * x[best_j] : rmse_type == 2 ? 1 : rmse_type == 3 ? fabsf(x[best_j]) : sqrtf(fabsf(x[best_j]));
        int l = L[best_j] - nmax;
        if (gmax > 0) {
            new_sumlx += w*x[best_j];
            new_suml2 += w*(2*l + 1);
            l += 1;
        } else {
            new_sumlx -= w*x[best_j];
            new_suml2 -= w*(2*l - 1);
            l -= 1;
        }
        if (new_suml2 > 0 && new_sumlx*new_sumlx > best*new_suml2) {
            sumlx = new_sumlx; suml2 = new_suml2;
            scale = sumlx/suml2; best = scale*sumlx;
            L[best_j] = l + nmax;
            GGML_ASSERT(L[best_j] >= 0 && L[best_j] <= 2*nmax-1);
        }
        else {
            break;
        }

    }
    return scale;
}

void quantize_row_q6_K_ref(const float * x, block_q6_K * y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    int8_t L[QK_K];
    float   scales[QK_K/16];

    const float fudge = 1.0f;

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float max_abs_scale = 0;

        for (int ib = 0; ib < QK_K/16; ++ib) {

            const float scale = make_qx_quants(16, 32, x + 16*ib, L + 16*ib, 1, NULL);
            scales[ib] = scale;

            const float abs_scale = fabsf(scale);
            if (abs_scale > max_abs_scale) {
                max_abs_scale = abs_scale;
                max_scale = scale;
            }

        }

        if (max_abs_scale < 1e-15f) {
            memset(&y[i], 0, sizeof(block_q6_K));
            y[i].d = GGML_FP32_TO_FP16(0.f);
            x += QK_K;
            continue;
        }

        float iscale = -128.f/max_scale;
        y[i].d = GGML_FP32_TO_FP16(fudge/iscale);
        for (int ib = 0; ib < QK_K/16; ++ib) {
            y[i].scales[ib] = MIN(127, nearest_int(iscale*scales[ib]));
        }

        for (int j = 0; j < QK_K/16; ++j) {
            float d = GGML_FP16_TO_FP32(y[i].d) * y[i].scales[j];
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-32, MIN(31, l));
                L[16*j + ii] = l + 32;
            }
        }

        uint8_t * ql = y[i].ql;
        uint8_t * qh = y[i].qh;
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                const uint8_t q1 = L[j + l +  0] & 0xF;
                const uint8_t q2 = L[j + l + 32] & 0xF;
                const uint8_t q3 = L[j + l + 64] & 0xF;
                const uint8_t q4 = L[j + l + 96] & 0xF;
                ql[l+ 0] = q1 | (q3 << 4);
                ql[l+32] = q2 | (q4 << 4);
                qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) | ((L[j + l + 96] >> 4) << 6);
            }
            ql += 64;
            qh += 32;
        }

        x += QK_K;
    }
}
void dequantize_row_q6_K(const block_q6_K * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * ql = x[i].ql;
        const uint8_t * qh = x[i].qh;
        const int8_t  * sc = x[i].scales;

        for (int n = 0; n < QK_K; n += 128) {
            for (int l = 0; l < 32; ++l) {
                int is = l/16;
                const int8_t q1 = (int8_t)((ql[l +  0] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                const int8_t q2 = (int8_t)((ql[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                const int8_t q3 = (int8_t)((ql[l +  0]  >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                const int8_t q4 = (int8_t)((ql[l + 32]  >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                y[l +  0] = d * sc[is + 0] * q1;
                y[l + 32] = d * sc[is + 2] * q2;
                y[l + 64] = d * sc[is + 4] * q3;
                y[l + 96] = d * sc[is + 6] * q4;
            }
            y  += 128;
            ql += 64;
            qh += 32;
            sc += 8;
        }
    }
}


// Q3_K/Q4_K references ggml-quants.c:1862-1923,1961-2050,2493-2618,2731-2830.
static float make_q3_quants(int n, int nmax, const float * x, int8_t * L, bool do_rmse) {
    float max = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        float ax = fabsf(x[i]);
        if (ax > amax) { amax = ax; max = x[i]; }
    }
    if (amax < 1e-15f) { // all zero
        for (int i = 0; i < n; ++i) { L[i] = 0; }
        return 0.f;
    }
    float iscale = -nmax / max;
    if (do_rmse) {
        float sumlx = 0;
        float suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale * x[i]);
            l = MAX(-nmax, MIN(nmax-1, l));
            L[i] = l;
            float w = x[i]*x[i];
            sumlx += w*x[i]*l;
            suml2 += w*l*l;
        }
        for (int itry = 0; itry < 5; ++itry) {
            int n_changed = 0;
            for (int i = 0; i < n; ++i) {
                float w = x[i]*x[i];
                float slx = sumlx - w*x[i]*L[i];
                if (slx > 0) {
                    float sl2 = suml2 - w*L[i]*L[i];
                    int new_l = nearest_int(x[i] * sl2 / slx);
                    new_l = MAX(-nmax, MIN(nmax-1, new_l));
                    if (new_l != L[i]) {
                        slx += w*x[i]*new_l;
                        sl2 += w*new_l*new_l;
                        if (sl2 > 0 && slx*slx*suml2 > sumlx*sumlx*sl2) {
                            L[i] = new_l; sumlx = slx; suml2 = sl2;
                            ++n_changed;
                        }
                    }
                }
            }
            if (!n_changed) {
                break;
            }
        }
        for (int i = 0; i < n; ++i) {
            L[i] += nmax;
        }
        return sumlx / suml2;
    }
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale * x[i]);
        l = MAX(-nmax, MIN(nmax-1, l));
        L[i] = l + nmax;
    }
    return 1/iscale;
}
static float make_qkx2_quants(int n, int nmax, const float * x, const float * weights,
        uint8_t * L, float * the_min, uint8_t * Laux,
        float rmin, float rdelta, int nstep, bool use_mad) {
    float min = x[0];
    float max = x[0];
    float sum_w = weights[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {

        if (x[i] < min) min = x[i];
        if (x[i] > max) max = x[i];
        float w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (min > 0) min = 0;
    if (max - min < 1e-10f) {
        for (int i = 0; i < n; ++i) L[i] = 0;
        *the_min = -min;
        return 0.f;
    }
    float iscale = nmax/(max - min);
    float scale = 1/iscale;
    float best_mad = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int(iscale*(x[i] - min));
        L[i] = MAX(0, MIN(nmax, l));
        float diff = scale * L[i] + min - x[i];
        diff = use_mad ? fabsf(diff) : diff * diff;
        float w = weights[i];
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -min;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta*is + nmax)/(max - min);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int(iscale*(x[i] - min));
            l = MAX(0, MIN(nmax, l));
            Laux[i] = l;
            float w = weights[i];
            sum_l += w*l;
            sum_l2 += w*l*l;
            sum_xl += w*l*x[i];
        }
        float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l)/D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl)/D;
            if (this_min > 0) {
                this_min = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff = use_mad ? fabsf(diff) : diff * diff;
                float w = weights[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    *the_min = -min;
    return scale;
}
static inline void get_scale_min_k4(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63; *m = q[j + 4] & 63;
    } else {
        *d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        *m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
void quantize_row_q3_K_ref(const float * x, block_q3_K * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    int8_t L[QK_K];
    float scales[QK_K / 16];

    const float fudge = 1.0f;

    for (int i = 0; i < nb; i++) {

        float max_scale = 0;
        float amax = 0;
        for (int j = 0; j < QK_K/16; ++j) {
            scales[j] = make_q3_quants(16, 4, x + 16*j, L + 16*j, true);
            float scale = fabsf(scales[j]);
            if (scale > amax) {
                amax = scale; max_scale = scales[j];
            }
        }

        memset(y[i].scales, 0, 12);
        if (max_scale) {
            float iscale = -32.f/max_scale;
            for (int j = 0; j < QK_K/16; ++j) {
                int l = nearest_int(iscale*scales[j]);
                l = MAX(-32, MIN(31, l)) + 32;
                if (j < 8) {
                    y[i].scales[j] = l & 0xF;
                } else {
                    y[i].scales[j-8] |= ((l & 0xF) << 4);
                }
                l >>= 4;
                y[i].scales[j%4 + 8] |= (l << (2*(j/4)));
            }
            y[i].d = GGML_FP32_TO_FP16(fudge/iscale);
        } else {
            y[i].d = GGML_FP32_TO_FP16(0.f);
        }

        int8_t sc;
        for (int j = 0; j < QK_K/16; ++j) {
            sc = j < 8 ? y[i].scales[j] & 0xF : y[i].scales[j-8] >> 4;
            sc = (sc | (((y[i].scales[8 + j%4] >> (2*(j/4))) & 3) << 4)) - 32;
            float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) {
                continue;
            }
            for (int ii = 0; ii < 16; ++ii) {
                int l = nearest_int(x[16*j + ii]/d);
                l = MAX(-4, MIN(3, l));
                L[16*j + ii] = l + 4;
            }
        }

        memset(y[i].hmask, 0, QK_K/8);
        // We put the high-bit for the 1st 8 quants into bit 0, the next 8 into bit 1, etc.
        int m = 0;
        uint8_t hm = 1;
        for (int j = 0; j < QK_K; ++j) {
            if (L[j] > 3) {
                y[i].hmask[m] |= hm;
                L[j] -= 4;
            }
            if (++m == QK_K/8) {
                m = 0; hm <<= 1;
            }
        }
        for (int j = 0; j < QK_K; j += 128) {
            for (int l = 0; l < 32; ++l) {
                y[i].qs[j/4 + l] = L[j + l] | (L[j + l + 32] << 2) | (L[j + l + 64] << 4) | (L[j + l + 96] << 6);
            }
        }

        x += QK_K;
    }
}
void dequantize_row_q3_K(const block_q3_K * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    const uint32_t kmask1 = 0x03030303;
    const uint32_t kmask2 = 0x0f0f0f0f;

    uint32_t aux[4];
    const int8_t * scales = (const int8_t*)aux;

    for (int i = 0; i < nb; i++) {

        const float d_all = GGML_FP16_TO_FP32(x[i].d);

        const uint8_t * q = x[i].qs;
        const uint8_t * hm = x[i].hmask;
        uint8_t m = 1;

        memcpy(aux, x[i].scales, 12);
        uint32_t tmp = aux[2];
        aux[2] = ((aux[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
        aux[3] = ((aux[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
        aux[0] = (aux[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
        aux[1] = (aux[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

        int is = 0;
        float dl;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+ 0] >> shift) & 3) - ((hm[l+ 0] & m) ? 0 : 4));
                }

                dl = d_all * (scales[is++] - 32);
                for (int l = 0; l < 16; ++l) {
                    *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3) - ((hm[l+16] & m) ? 0 : 4));
                }

                shift += 2;
                m <<= 1;
            }
            q += 32;
        }

    }
}
void quantize_row_q4_K_ref(const float * x, block_q4_K * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    float   weights[32];
    float mins[QK_K/32];
    float scales[QK_K/32];

    for (int i = 0; i < nb; i++) {
        float max_scale = 0; // as we are deducting the min, scales are always positive
        float max_min = 0;
        for (int j = 0; j < QK_K/32; ++j) {
            //scales[j] = make_qkx1_quants(32, 15, x + 32*j, L + 32*j, &mins[j], 9, 0.5f);
            float sum_x2 = 0;
            for (int l = 0; l < 32; ++l) sum_x2 += x[32*j + l] * x[32*j + l];
            float av_x = sqrtf(sum_x2/32);
            for (int l = 0; l < 32; ++l) weights[l] = av_x + fabsf(x[32*j + l]);
            scales[j] = make_qkx2_quants(32, 15, x + 32*j, weights, L + 32*j, &mins[j], Laux, -1.f, 0.1f, 20, false);
            if (std::isnan(scales[j])) {
                printf("Oops: NaN scale\n");
                throw std::runtime_error("reference quantizer produced NaN scale");
            }
            float scale = scales[j];
            if (scale > max_scale) {
                max_scale = scale;
            }
            float min = mins[j];
            if (min > max_min) {
                max_min = min;
            }
        }

        float inv_scale = max_scale > 0 ? 63.f/max_scale : 0.f;
        float inv_min   = max_min   > 0 ? 63.f/max_min   : 0.f;
        for (int j = 0; j < QK_K/32; ++j) {
            uint8_t ls = nearest_int(inv_scale*scales[j]);
            uint8_t lm = nearest_int(inv_min*mins[j]);
            ls = std::min<int>(63,ls);
            lm = std::min<int>(63,lm);
            if (j < 4) {
                y[i].scales[j] = ls;
                y[i].scales[j+4] = lm;
            } else {
                y[i].scales[j+4] = (ls & 0xF) | ((lm & 0xF) << 4);
                y[i].scales[j-4] |= ((ls >> 4) << 6);
                y[i].scales[j-0] |= ((lm >> 4) << 6);
            }
        }
        y[i].d = GGML_FP32_TO_FP16(max_scale/63.f);
        y[i].dmin = GGML_FP32_TO_FP16(max_min/63.f);

        uint8_t sc, m;
        for (int j = 0; j < QK_K/32; ++j) {
            get_scale_min_k4(j, y[i].scales, &sc, &m);
            const float d = GGML_FP16_TO_FP32(y[i].d) * sc;
            if (!d) continue;
            const float dm = GGML_FP16_TO_FP32(y[i].dmin) * m;
            for (int ii = 0; ii < 32; ++ii) {
                int l = nearest_int((x[32*j + ii] + dm)/d);
                l = MAX(0, MIN(15, l));
                L[32*j + ii] = l;
            }
        }

        uint8_t * q = y[i].qs;
        for (int j = 0; j < QK_K; j += 64) {
            for (int l = 0; l < 32; ++l) q[l] = L[j + l] | (L[j + l + 32] << 4);
            q += 32;
        }

        x += QK_K;
    }
}
void dequantize_row_q4_K(const block_q4_K * x, float * y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {
        const uint8_t * q = x[i].qs;

        const float d   = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        int is = 0;
        uint8_t sc, m;
        for (int j = 0; j < QK_K; j += 64) {
            get_scale_min_k4(is + 0, x[i].scales, &sc, &m);
            const float d1 = d * sc; const float m1 = min * m;
            get_scale_min_k4(is + 1, x[i].scales, &sc, &m);
            const float d2 = d * sc; const float m2 = min * m;
            for (int l = 0; l < 32; ++l) *y++ = d1 * (q[l] & 0xF) - m1;
            for (int l = 0; l < 32; ++l) *y++ = d2 * (q[l]  >> 4) - m2;
            q += 32; is += 2;
        }
    }
}


// Q8_0 reference ggml-quants.c:911-936,1703-1717; default fudge1.0.
void quantize_row_q8_0_ref(const float * x, block_q8_0 * y, int64_t k) {
    assert(k % 32 == 0);
    const int nb = k / 32;

    const float fudge = 1.0f;

    for (int i = 0; i < nb; i++) {
        float amax = 0.0f; // absolute max

        for (int j = 0; j < 32; j++) {
            const float v = x[i*32 + j];
            amax = MAX(amax, fabsf(v));
        }

        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f/d : 0.0f;

        y[i].d = GGML_FP32_TO_FP16(fudge * d);

        for (int j = 0; j < 32; ++j) {
            const float x0 = x[i*32 + j]*id;

            y[i].qs[j] = roundf(x0);
        }
    }
}
void dequantize_row_q8_0(const block_q8_0 * x, float * y, int64_t k) {
    static const int qk = 32;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

inline void require_n(int n) { if(n<=0 || n%256) throw std::invalid_argument("n must be a positive multiple of 256"); }
} // anonymous namespace
namespace {
void dot_legacy_formats(Format format,const void *row,const Activation *inputs,int n,int m,float *outputs,Kernel kernel);
void dot_q6_dispatch(const void *row,const Activation *inputs,int n,int m,float *outputs,Kernel kernel);
void dot_q8_0_dispatch(const void *row,const Activation *inputs,int n,int m,float *outputs,Kernel kernel);
void dot_q34_dispatch(Format format,const void *row,const Activation *inputs,int n,int m,float *outputs,Kernel kernel);
}
const char *format_name(Format format) {
    switch(format) {
        case Format::IQ3_K: return "IQ3_K";
        case Format::IQ4_KS: return "IQ4_KS";
        case Format::IQ3_S: return "IQ3_S";
        case Format::IQ4_XS: return "IQ4_XS";
        case Format::Q6_K: return "Q6_K";
        case Format::Q3_K: return "Q3_K";
        case Format::Q4_K: return "Q4_K";
        case Format::Q8_0: return "Q8_0";
    }
    throw std::invalid_argument("unsupported packed row format");
}
size_t row_bytes(Format format, int n) {
    require_n(n);
    switch(format) {
        case Format::IQ3_K: return size_t(n/256)*sizeof(block_iq3_k);
        case Format::IQ4_KS: return sizeof(float)+size_t(n/256)*sizeof(block_iq4_ks);
        case Format::IQ3_S: return size_t(n/256)*sizeof(block_iq3_s);
        case Format::IQ4_XS: return size_t(n/256)*sizeof(block_iq4_xs);
        case Format::Q6_K: return size_t(n/256)*sizeof(block_q6_K);
        case Format::Q3_K: return size_t(n/256)*sizeof(block_q3_K);
        case Format::Q4_K: return size_t(n/256)*sizeof(block_q4_K);
        case Format::Q8_0: return size_t(n/32)*sizeof(block_q8_0);
    }
    throw std::invalid_argument("unsupported packed row format");
}
void slice_packed_row(Format format, const void *src, int n,
                      int begin, int count, void *out) {
    require_n(n);
    (void)format_name(format);
    if(!src || !out || begin<0 || count<=0 || begin%256 || count%256 || begin>n || count>n-begin)
        throw std::invalid_argument("invalid packed row slice");
    const size_t src_bytes=row_bytes(format,n),out_bytes=row_bytes(format,count);
    const uintptr_t src_address=reinterpret_cast<uintptr_t>(src);
    const uintptr_t out_address=reinterpret_cast<uintptr_t>(out);
    // Subtractions avoid address+length overflow while testing both orders.
    const bool overlap=src_address<=out_address ? out_address-src_address<src_bytes
                                               : src_address-out_address<out_bytes;
    if(overlap) throw std::invalid_argument("packed row slice buffers must not overlap");
    const auto *source=static_cast<const uint8_t *>(src);
    auto *destination=static_cast<uint8_t *>(out);
    if(format!=Format::IQ4_KS) {
        const size_t block_bytes=row_bytes(format,256);
        std::memcpy(destination,source+size_t(begin/256)*block_bytes,out_bytes);
    } else {
        // This scale belongs to the original complete row, not the slice.
        // Copy its exact bits; recomputing it would change represented weights.
        std::memcpy(destination,source,sizeof(float));
        std::memcpy(destination+sizeof(float),source+sizeof(float)+size_t(begin/256)*sizeof(block_iq4_ks),
                    out_bytes-sizeof(float));
    }
}
void quantize(Format format, const float *x, void *out, int n) {
    require_n(n);
    if(format==Format::IQ3_K) quantize_row_iq3_k_impl(x,out,n,nullptr);
    else if(format==Format::Q8_0) quantize_row_q8_0_ref(x,static_cast<block_q8_0 *>(out),n);
    else if(format==Format::Q6_K) quantize_row_q6_K_ref(x,static_cast<block_q6_K *>(out),n);
    else if(format==Format::Q3_K) quantize_row_q3_K_ref(x,static_cast<block_q3_K *>(out),n);
    else if(format==Format::Q4_K) quantize_row_q4_K_ref(x,static_cast<block_q4_K *>(out),n);
    else if(format==Format::IQ3_S) {
        float scales[8],weight[32],xval[32],waux[32];
        int8_t l[32],laux[32]; bool ongrid[8],ongridaux[8]; uint8_t signs[4];
        quantize_row_iq3_s_impl(32,x,out,n,nullptr,scales,weight,xval,l,laux,waux,ongrid,ongridaux,signs);
    } else if(format==Format::IQ4_XS) {
        uint8_t l[256]; float weight[32],scales[8];
        auto *blocks=static_cast<block_iq4_xs *>(out);
        for(int b=0;b<n/256;++b)
            quantize_row_iq4_nl_impl(256,32,x+256*b,&blocks[b].d,blocks[b].qs,&blocks[b].scales_h,blocks[b].scales_l,
                                     scales,weight,l,iq4k_values,nullptr,7);
    } else if(format==Format::IQ4_KS) {
        std::vector<float> scales(n/32);
        float weight[32];
        quantize_row_iq4_k_impl_bs128(256,32,n,x,static_cast<char *>(out),scales.data(),weight,iq4k_values,nullptr,7);
    } else throw std::invalid_argument("unsupported packed row format");
}
void dequantize(Format format, const void *x, float *out, int n) {
    require_n(n);
    if(format==Format::IQ3_K) dequantize_row_iq3_k(static_cast<const block_iq3_k *>(x),out,n);
    else if(format==Format::IQ4_KS) dequantize_row_iq4_ks(static_cast<const block_iq4_ks *>(x),out,n);
    else if(format==Format::IQ3_S) dequantize_row_iq3_s(static_cast<const block_iq3_s *>(x),out,n);
    else if(format==Format::IQ4_XS) dequantize_row_iq4_xs(static_cast<const block_iq4_xs *>(x),out,n);
    else if(format==Format::Q8_0) dequantize_row_q8_0(static_cast<const block_q8_0 *>(x),out,n);
    else if(format==Format::Q6_K) dequantize_row_q6_K(static_cast<const block_q6_K *>(x),out,n);
    else if(format==Format::Q3_K) dequantize_row_q3_K(static_cast<const block_q3_K *>(x),out,n);
    else if(format==Format::Q4_K) dequantize_row_q4_K(static_cast<const block_q4_K *>(x),out,n);
    else throw std::invalid_argument("unsupported packed row format");
}
void prepare_storage(Path path, int n, Activation &out, Kernel kernel) {
    require_n(n); out.path=path; out.n=n; out.kernel=kernel;
    if(path==Path::Q8) out.q8.resize(size_t(n/256)*sizeof(block_q8_K));
    else {
        out.i16.resize(n); out.scales.resize(n/128);
        if(kernel==Kernel::Optimized) {
            out.i16_swizzled.resize(n); out.i16_lo.resize(n); out.i16_hi.resize(n);
            out.i16_lo_bsums.resize(n/16); out.i16_hi_bsums.resize(n/16);
        }
    }
}
void prepare_range(Path path, const float *x, int n, int begin, int end, Activation &out) {
    const int block=path==Path::Q8 ? 256 : 128;
    if(out.n!=n || out.path!=path || begin<0 || end>n || begin>end || begin%block || end%block)
        throw std::invalid_argument("invalid activation preparation range");
    if(path==Path::Q8) {
        auto *dst=reinterpret_cast<block_q8_K *>(out.q8.data())+begin/256;
        std::memset(dst,0,size_t((end-begin)/256)*sizeof(block_q8_K));
        quantize_row_q8_K_ref(x+begin,dst,end-begin);
    } else {
        for(int b=begin/128;b<end/128;++b) {
            float amax=0;
            for(int j=0;j<128;++j) amax=std::max(amax,std::abs(x[128*b+j]));
            const float scale=amax/16383.f;
            out.scales[b]=scale;
            const float inv=amax ? 16383.f/amax : 0.f;
            for(int j=0;j<128;++j)
                out.i16[128*b+j]=int16_t(std::max(-16383,std::min(16383,nearest_int(x[128*b+j]*inv))));
            if(out.kernel==Kernel::Baseline) continue;
            for(int j=0;j<128;++j) {
                const int a=out.i16[128*b+j];
                out.i16_hi[128*b+j]=int8_t(a/128);
                out.i16_lo[128*b+j]=int8_t(a%128);
            }
            for(int j=0;j<128;j+=16) {
                int lo=0,hi=0;
                for(int k=0;k<16;++k) { lo+=out.i16_lo[128*b+j+k]; hi+=out.i16_hi[128*b+j+k]; }
                out.i16_lo_bsums[8*b+j/16]=int16_t(lo); out.i16_hi_bsums[8*b+j/16]=int16_t(hi);
            }
            for(int j=0;j<128;j+=32) {
                const auto lo=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(out.i16.data()+128*b+j));
                const auto hi=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(out.i16.data()+128*b+j+16));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(out.i16_swizzled.data()+128*b+j),_mm256_permute2x128_si256(lo,hi,0x20));
                _mm256_storeu_si256(reinterpret_cast<__m256i *>(out.i16_swizzled.data()+128*b+j+16),_mm256_permute2x128_si256(lo,hi,0x31));
            }
        }
    }
}
void prepare(Path path, const float *x, int n, Activation &out, Kernel kernel) {
    prepare_storage(path,n,out,kernel);
    prepare_range(path,x,n,0,n,out);
}
void activation_dequantize(const Activation &x, float *out, int n) {
    require_n(n);
    if(x.n!=n) throw std::invalid_argument("activation shape mismatch");
    if(x.path==Path::Q8) {
        const auto *p=reinterpret_cast<const block_q8_K *>(x.q8.data());
        for(int b=0;b<n/256;++b) for(int j=0;j<256;++j) out[256*b+j]=p[b].d*p[b].qs[j];
    } else for(int j=0;j<n;++j) out[j]=x.scales[j/128]*x.i16[j];
}
#ifdef LUTBENCH_MCA
#define MCA_BEGIN(name) asm volatile("# LLVM-MCA-BEGIN " name)
#define MCA_END(name) asm volatile("# LLVM-MCA-END " name)
#else
#define MCA_BEGIN(name) do {} while(0)
#define MCA_END(name) do {} while(0)
#endif

// Fixed allocation for Broadwell: 0=float accumulator, 1=block multiplier,
// 2=signed scales, 3=high bits, 4=low bits, 5/6/7=temporaries,
// 8=half-block scales, 9=integer sum, 10..15=unpack tables/masks.
// All loop memory operands are input data or read-only constants. In particular,
// the asm owns the back edge, so GCC (including LTO) cannot insert spills in it.
#ifndef FT_IQ3_Q8_ASM
#define FT_IQ3_Q8_ASM 1
#endif
#if FT_IQ3_Q8_ASM && defined(__x86_64__) && defined(__GNUC__)
struct alignas(32) IQ3Q8AsmConstants { uint8_t v[15][32]; };
static constexpr IQ3Q8AsmConstants iq3_q8_asm_constants = {{
        {1,24,41,54,65,77,92,111,1,24,41,54,65,77,92,111,
         1,24,41,54,65,77,92,111,1,24,41,54,65,77,92,111},
        {1,1,65,65,24,24,77,77,41,41,92,92,54,54,111,111,
         1,1,65,65,24,24,77,77,41,41,92,92,54,54,111,111},
        {3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3},
        {12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12,12},
        {4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4},
        {2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2,2},
        {15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15,15},
        {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1},
        {2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1},
        {1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128,1,2,4,8,16,32,64,128},
        {192,255,192,255,192,255,192,255,192,255,192,255,192,255,192,255,192,255,192,255,192,255,192,255,192,255,192,255,192,255,192,255},
        {0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,2,3,2,3,2,3,2,3,2,3,2,3,2,3,2,3},
        {4,5,4,5,4,5,4,5,4,5,4,5,4,5,4,5,6,7,6,7,6,7,6,7,6,7,6,7,6,7,6,7},
        {8,9,8,9,8,9,8,9,8,9,8,9,8,9,8,9,10,11,10,11,10,11,10,11,10,11,10,11,10,11,10,11},
        {12,13,12,13,12,13,12,13,12,13,12,13,12,13,12,13,14,15,14,15,14,15,14,15,14,15,14,15,14,15,14,15},
    }};

static float dot_iq3_q8_asm(const void *row, const Activation &x, int n) {
    const auto *w = static_cast<const uint8_t *>(row);
    const auto *a = x.q8.data();
    int blocks = n / 256;
    register float result asm("xmm0");
    asm volatile(
        "vxorps %%xmm0, %%xmm0, %%xmm0\n\t"
        "test %[blocks], %[blocks]\n\t"
        "jle 2f\n\t"
        "vmovdqa 0(%[k]), %%ymm10\n\t"
        "vmovdqa 32(%[k]), %%ymm11\n\t"
        "vmovdqa 64(%[k]), %%ymm12\n\t"
        "vmovdqa 96(%[k]), %%ymm13\n\t"
        "vmovdqa 128(%[k]), %%ymm14\n\t"
        "vmovdqa 160(%[k]), %%ymm15\n\t"
        ".p2align 5\n\t"
        "1:\n\t"
        "vmovdqu 14(%[w]), %%ymm7\n\t"
        "vmovdqu 78(%[w]), %%ymm3\n\t"
        "vmovq 6(%[w]), %%xmm4\n\t"
        "vpsrlw $4, %%xmm4, %%xmm5\n\t"
        "vpunpcklbw %%xmm5, %%xmm4, %%xmm4\n\t"
        "vpand 192(%[k]), %%xmm4, %%xmm4\n\t"
        "vpaddb %%xmm4, %%xmm4, %%xmm4\n\t"
        "vpaddb 224(%[k]), %%xmm4, %%xmm4\n\t"
        "vpbroadcastq 2(%[w]), %%ymm5\n\t"
        "vpshufb 256(%[k]), %%ymm5, %%ymm5\n\t"
        "vpand 288(%[k]), %%ymm5, %%ymm5\n\t"
        "vpcmpeqb 288(%[k]), %%ymm5, %%ymm5\n\t"
        "vpor 224(%[k]), %%xmm5, %%xmm6\n\t"
        "vpsignb %%xmm6, %%xmm4, %%xmm4\n\t"
        "vpmovsxbw %%xmm4, %%ymm2\n\t"
        "vextracti128 $1, %%ymm5, %%xmm5\n\t"
        "vpand %%xmm14, %%xmm5, %%xmm5\n\t"
        "vpmovsxbw %%xmm5, %%ymm5\n\t"
        "vpaddw 320(%[k]), %%ymm5, %%ymm5\n\t"
        "vpmullw %%ymm2, %%ymm5, %%ymm5\n\t"
        "vpmaddwd 264(%[a]), %%ymm5, %%ymm5\n\t"
        "vcvtdq2ps %%ymm5, %%ymm5\n\t"
        "vpbroadcastw (%[w]), %%xmm1\n\t"
        "vcvtph2ps %%xmm1, %%xmm1\n\t"
        "vmulss (%[a]), %%xmm1, %%xmm1\n\t"
        "vbroadcastss %%xmm1, %%ymm1\n\t"
        "vfmadd132ps %%ymm1, %%ymm0, %%ymm5\n\t"
        "vmovaps %%ymm5, %%ymm0\n\t"
        "vpermq $68, %%ymm2, %%ymm8\n\t"
        "vmovdqa %%ymm7, %%ymm4\n\t"
        "vpsllw $2, %%ymm3, %%ymm6\n\t"

        "vpand %%ymm14, %%ymm6, %%ymm6\n\t"
        "vpand %%ymm12, %%ymm4, %%ymm5\n\t"
        "vpor %%ymm6, %%ymm5, %%ymm5\n\t"
        "vpshufb %%ymm5, %%ymm10, %%ymm5\n\t"
        "vpmaddubsw 8(%[a]), %%ymm5, %%ymm5\n\t"
        "vpshufb 352(%[k]), %%ymm8, %%ymm7\n\t"
        "vpmaddwd %%ymm7, %%ymm5, %%ymm5\n\t"
        "vmovdqa %%ymm5, %%ymm9\n\t"
        "vpand %%ymm15, %%ymm3, %%ymm6\n\t"
        "vpand %%ymm13, %%ymm4, %%ymm5\n\t"
        "vpor %%ymm6, %%ymm5, %%ymm5\n\t"
        "vpshufb %%ymm5, %%ymm11, %%ymm5\n\t"
        "vpmaddubsw 40(%[a]), %%ymm5, %%ymm5\n\t"
        "vpshufb 384(%[k]), %%ymm8, %%ymm7\n\t"
        "vpmaddwd %%ymm7, %%ymm5, %%ymm5\n\t"
        "vpaddd %%ymm5, %%ymm9, %%ymm9\n\t"
        "vpsrlw $4, %%ymm4, %%ymm4\n\t"
        "vpand %%ymm14, %%ymm3, %%ymm6\n\t"
        "vpand %%ymm12, %%ymm4, %%ymm5\n\t"
        "vpor %%ymm6, %%ymm5, %%ymm5\n\t"
        "vpshufb %%ymm5, %%ymm10, %%ymm5\n\t"
        "vpmaddubsw 72(%[a]), %%ymm5, %%ymm5\n\t"
        "vpshufb 416(%[k]), %%ymm8, %%ymm7\n\t"
        "vpmaddwd %%ymm7, %%ymm5, %%ymm5\n\t"
        "vpaddd %%ymm5, %%ymm9, %%ymm9\n\t"
        "vpsrlw $2, %%ymm3, %%ymm3\n\t"
        "vpand %%ymm15, %%ymm3, %%ymm6\n\t"
        "vpand %%ymm13, %%ymm4, %%ymm5\n\t"
        "vpor %%ymm6, %%ymm5, %%ymm5\n\t"
        "vpshufb %%ymm5, %%ymm11, %%ymm5\n\t"
        "vpmaddubsw 104(%[a]), %%ymm5, %%ymm5\n\t"
        "vpshufb 448(%[k]), %%ymm8, %%ymm7\n\t"
        "vpmaddwd %%ymm7, %%ymm5, %%ymm5\n\t"
        "vpaddd %%ymm5, %%ymm9, %%ymm9\n\t"
        "vpermq $238, %%ymm2, %%ymm8\n\t"
        "vmovdqu 46(%[w]), %%ymm4\n\t"
        "vpand %%ymm14, %%ymm3, %%ymm6\n\t"
        "vpand %%ymm12, %%ymm4, %%ymm5\n\t"
        "vpor %%ymm6, %%ymm5, %%ymm5\n\t"
        "vpshufb %%ymm5, %%ymm10, %%ymm5\n\t"
        "vpmaddubsw 136(%[a]), %%ymm5, %%ymm5\n\t"
        "vpshufb 352(%[k]), %%ymm8, %%ymm7\n\t"
        "vpmaddwd %%ymm7, %%ymm5, %%ymm5\n\t"
        "vpaddd %%ymm5, %%ymm9, %%ymm9\n\t"
        "vpsrlw $2, %%ymm3, %%ymm3\n\t"
        "vpand %%ymm15, %%ymm3, %%ymm6\n\t"
        "vpand %%ymm13, %%ymm4, %%ymm5\n\t"
        "vpor %%ymm6, %%ymm5, %%ymm5\n\t"
        "vpshufb %%ymm5, %%ymm11, %%ymm5\n\t"
        "vpmaddubsw 168(%[a]), %%ymm5, %%ymm5\n\t"
        "vpshufb 384(%[k]), %%ymm8, %%ymm7\n\t"
        "vpmaddwd %%ymm7, %%ymm5, %%ymm5\n\t"
        "vpaddd %%ymm5, %%ymm9, %%ymm9\n\t"
        "vpsrlw $4, %%ymm4, %%ymm4\n\t"
        "vpand %%ymm14, %%ymm3, %%ymm6\n\t"
        "vpand %%ymm12, %%ymm4, %%ymm5\n\t"
        "vpor %%ymm6, %%ymm5, %%ymm5\n\t"
        "vpshufb %%ymm5, %%ymm10, %%ymm5\n\t"
        "vpmaddubsw 200(%[a]), %%ymm5, %%ymm5\n\t"
        "vpshufb 416(%[k]), %%ymm8, %%ymm7\n\t"
        "vpmaddwd %%ymm7, %%ymm5, %%ymm5\n\t"
        "vpaddd %%ymm5, %%ymm9, %%ymm9\n\t"
        "vpsrlw $2, %%ymm3, %%ymm3\n\t"
        "vpand %%ymm15, %%ymm3, %%ymm6\n\t"
        "vpand %%ymm13, %%ymm4, %%ymm5\n\t"
        "vpor %%ymm6, %%ymm5, %%ymm5\n\t"
        "vpshufb %%ymm5, %%ymm11, %%ymm5\n\t"
        "vpmaddubsw 232(%[a]), %%ymm5, %%ymm5\n\t"
        "vpshufb 448(%[k]), %%ymm8, %%ymm7\n\t"
        "vpmaddwd %%ymm7, %%ymm5, %%ymm5\n\t"
        "vpaddd %%ymm5, %%ymm9, %%ymm9\n\t"
        "vcvtdq2ps %%ymm9, %%ymm9\n\t"
        "vfmadd213ps %%ymm0, %%ymm1, %%ymm9\n\t"
        "vmovaps %%ymm9, %%ymm0\n\t"
        "add $110, %[w]\n\t"
        "add $296, %[a]\n\t"
        "sub $1, %[blocks]\n\t"
        "jg 1b\n\t"
        "2:\n\t"
        "vextractf128 $1, %%ymm0, %%xmm1\n\t"
        "vaddps %%xmm0, %%xmm1, %%xmm0\n\t"
        "vmovhlps %%xmm0, %%xmm0, %%xmm1\n\t"
        "vaddps %%xmm1, %%xmm0, %%xmm0\n\t"
        "vpermilps $1, %%xmm0, %%xmm1\n\t"
        "vaddss %%xmm1, %%xmm0, %%xmm0\n\t"
        : "=x"(result), [w] "+&r"(w), [a] "+&r"(a), [blocks] "+&r"(blocks)
        : [k] "r"(&iq3_q8_asm_constants)
        : "cc", "memory", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
          "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15");
    return result;
}
// M2: 0/1=float accumulators, 8/9=integer sums, 6=correction,
// 2=scales, 15=half scales, 3/4=packed bits, 5/7/14=scratch, 10..13=tables/masks.
static void dot_iq3_q8_m2_asm(const void *row, const Activation *inputs, int n, float *out) {
    const auto *w = static_cast<const uint8_t *>(row);
    const auto *a = inputs[0].q8.data();
    const auto *a1 = inputs[1].q8.data();
    int blocks = n / 256;
    asm volatile(
        "vxorps %%xmm0, %%xmm0, %%xmm0\n\t"
        "vxorps %%xmm1, %%xmm1, %%xmm1\n\t"
        "test %[blocks], %[blocks]\n\t"
        "jle 2f\n\t"
        "vmovdqa 0(%[k]), %%ymm10\n\t"
        "vmovdqa 32(%[k]), %%ymm11\n\t"
        "vmovdqa 64(%[k]), %%ymm12\n\t"
        "vmovdqa 96(%[k]), %%ymm13\n\t"
        ".p2align 5\n\t"
        "1:\n\t"
        "vmovq 6(%[w]), %%xmm4\n\t"
        "vpsrlw $4, %%xmm4, %%xmm5\n\t"
        "vpunpcklbw %%xmm5, %%xmm4, %%xmm4\n\t"
        "vpand 192(%[k]), %%xmm4, %%xmm4\n\t"
        "vpaddb %%xmm4, %%xmm4, %%xmm4\n\t"
        "vpaddb 224(%[k]), %%xmm4, %%xmm4\n\t"
        "vpbroadcastq 2(%[w]), %%ymm5\n\t"
        "vpshufb 256(%[k]), %%ymm5, %%ymm5\n\t"
        "vpand 288(%[k]), %%ymm5, %%ymm5\n\t"
        "vpcmpeqb 288(%[k]), %%ymm5, %%ymm5\n\t"
        "vpor 224(%[k]), %%xmm5, %%xmm6\n\t"
        "vpsignb %%xmm6, %%xmm4, %%xmm4\n\t"
        "vpmovsxbw %%xmm4, %%ymm2\n\t"
        "vextracti128 $1, %%ymm5, %%xmm5\n\t"
        "vpand 128(%[k]), %%xmm5, %%xmm5\n\t"
        "vpmovsxbw %%xmm5, %%ymm5\n\t"
        "vpaddw 320(%[k]), %%ymm5, %%ymm5\n\t"
        "vpmullw %%ymm2, %%ymm5, %%ymm6\n\t"
        "vmovdqu 78(%[w]), %%ymm3\n\t"
        "vpermq $68, %%ymm2, %%ymm15\n\t"
        "vmovdqu 14(%[w]), %%ymm4\n\t"
        "vpsllw $2, %%ymm3, %%ymm7\n\t"

        "vpand 128(%[k]), %%ymm7, %%ymm7\n\t"
        "vpand %%ymm12, %%ymm4, %%ymm5\n\t"
        "vpor %%ymm7, %%ymm5, %%ymm5\n\t"
        "vpshufb %%ymm5, %%ymm10, %%ymm5\n\t"
        "vpshufb 352(%[k]), %%ymm15, %%ymm7\n\t"
        "vpmaddubsw 8(%[a]), %%ymm5, %%ymm14\n\t"
        "vpmaddwd %%ymm7, %%ymm14, %%ymm14\n\t"
        "vmovdqa %%ymm14, %%ymm9\n\t"
        "vpmaddubsw 8(%[a1]), %%ymm5, %%ymm14\n\t"
        "vpmaddwd %%ymm7, %%ymm14, %%ymm14\n\t"
        "vmovdqa %%ymm14, %%ymm8\n\t"
        "vpand 160(%[k]), %%ymm3, %%ymm7\n\t"
        "vpand %%ymm13, %%ymm4, %%ymm5\n\t"
        "vpor %%ymm7, %%ymm5, %%ymm5\n\t"
        "vpshufb %%ymm5, %%ymm11, %%ymm5\n\t"
        "vpshufb 384(%[k]), %%ymm15, %%ymm7\n\t"
        "vpmaddubsw 40(%[a]), %%ymm5, %%ymm14\n\t"
        "vpmaddwd %%ymm7, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm9, %%ymm9\n\t"
        "vpmaddubsw 40(%[a1]), %%ymm5, %%ymm14\n\t"
        "vpmaddwd %%ymm7, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm8, %%ymm8\n\t"
        "vpsrlw $4, %%ymm4, %%ymm4\n\t"
        "vpand 128(%[k]), %%ymm3, %%ymm7\n\t"
        "vpand %%ymm12, %%ymm4, %%ymm5\n\t"
        "vpor %%ymm7, %%ymm5, %%ymm5\n\t"
        "vpshufb %%ymm5, %%ymm10, %%ymm5\n\t"
        "vpshufb 416(%[k]), %%ymm15, %%ymm7\n\t"
        "vpmaddubsw 72(%[a]), %%ymm5, %%ymm14\n\t"
        "vpmaddwd %%ymm7, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm9, %%ymm9\n\t"
        "vpmaddubsw 72(%[a1]), %%ymm5, %%ymm14\n\t"
        "vpmaddwd %%ymm7, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm8, %%ymm8\n\t"
        "vpsrlw $2, %%ymm3, %%ymm3\n\t"
        "vpand 160(%[k]), %%ymm3, %%ymm7\n\t"
        "vpand %%ymm13, %%ymm4, %%ymm5\n\t"
        "vpor %%ymm7, %%ymm5, %%ymm5\n\t"
        "vpshufb %%ymm5, %%ymm11, %%ymm5\n\t"
        "vpshufb 448(%[k]), %%ymm15, %%ymm7\n\t"
        "vpmaddubsw 104(%[a]), %%ymm5, %%ymm14\n\t"
        "vpmaddwd %%ymm7, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm9, %%ymm9\n\t"
        "vpmaddubsw 104(%[a1]), %%ymm5, %%ymm14\n\t"
        "vpmaddwd %%ymm7, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm8, %%ymm8\n\t"
        "vpermq $238, %%ymm2, %%ymm15\n\t"
        "vmovdqu 46(%[w]), %%ymm4\n\t"
        "vpand 128(%[k]), %%ymm3, %%ymm7\n\t"
        "vpand %%ymm12, %%ymm4, %%ymm5\n\t"
        "vpor %%ymm7, %%ymm5, %%ymm5\n\t"
        "vpshufb %%ymm5, %%ymm10, %%ymm5\n\t"
        "vpshufb 352(%[k]), %%ymm15, %%ymm7\n\t"
        "vpmaddubsw 136(%[a]), %%ymm5, %%ymm14\n\t"
        "vpmaddwd %%ymm7, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm9, %%ymm9\n\t"
        "vpmaddubsw 136(%[a1]), %%ymm5, %%ymm14\n\t"
        "vpmaddwd %%ymm7, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm8, %%ymm8\n\t"
        "vpsrlw $2, %%ymm3, %%ymm3\n\t"
        "vpand 160(%[k]), %%ymm3, %%ymm7\n\t"
        "vpand %%ymm13, %%ymm4, %%ymm5\n\t"
        "vpor %%ymm7, %%ymm5, %%ymm5\n\t"
        "vpshufb %%ymm5, %%ymm11, %%ymm5\n\t"
        "vpshufb 384(%[k]), %%ymm15, %%ymm7\n\t"
        "vpmaddubsw 168(%[a]), %%ymm5, %%ymm14\n\t"
        "vpmaddwd %%ymm7, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm9, %%ymm9\n\t"
        "vpmaddubsw 168(%[a1]), %%ymm5, %%ymm14\n\t"
        "vpmaddwd %%ymm7, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm8, %%ymm8\n\t"
        "vpsrlw $4, %%ymm4, %%ymm4\n\t"
        "vpand 128(%[k]), %%ymm3, %%ymm7\n\t"
        "vpand %%ymm12, %%ymm4, %%ymm5\n\t"
        "vpor %%ymm7, %%ymm5, %%ymm5\n\t"
        "vpshufb %%ymm5, %%ymm10, %%ymm5\n\t"
        "vpshufb 416(%[k]), %%ymm15, %%ymm7\n\t"
        "vpmaddubsw 200(%[a]), %%ymm5, %%ymm14\n\t"
        "vpmaddwd %%ymm7, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm9, %%ymm9\n\t"
        "vpmaddubsw 200(%[a1]), %%ymm5, %%ymm14\n\t"
        "vpmaddwd %%ymm7, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm8, %%ymm8\n\t"
        "vpsrlw $2, %%ymm3, %%ymm3\n\t"
        "vpand 160(%[k]), %%ymm3, %%ymm7\n\t"
        "vpand %%ymm13, %%ymm4, %%ymm5\n\t"
        "vpor %%ymm7, %%ymm5, %%ymm5\n\t"
        "vpshufb %%ymm5, %%ymm11, %%ymm5\n\t"
        "vpshufb 448(%[k]), %%ymm15, %%ymm7\n\t"
        "vpmaddubsw 232(%[a]), %%ymm5, %%ymm14\n\t"
        "vpmaddwd %%ymm7, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm9, %%ymm9\n\t"
        "vpmaddubsw 232(%[a1]), %%ymm5, %%ymm14\n\t"
        "vpmaddwd %%ymm7, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm8, %%ymm8\n\t"
        "vpbroadcastw (%[w]), %%xmm3\n\t"
        "vcvtph2ps %%xmm3, %%xmm3\n\t"
        "vpmaddwd 264(%[a]), %%ymm6, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm9, %%ymm9\n\t"
        "vcvtdq2ps %%ymm9, %%ymm9\n\t"
        "vmulss (%[a]), %%xmm3, %%xmm5\n\t"
        "vbroadcastss %%xmm5, %%ymm5\n\t"
        "vfmadd231ps %%ymm9, %%ymm5, %%ymm0\n\t"
        "vpmaddwd 264(%[a1]), %%ymm6, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm8, %%ymm8\n\t"
        "vcvtdq2ps %%ymm8, %%ymm8\n\t"
        "vmulss (%[a1]), %%xmm3, %%xmm5\n\t"
        "vbroadcastss %%xmm5, %%ymm5\n\t"
        "vfmadd231ps %%ymm8, %%ymm5, %%ymm1\n\t"
        "add $110, %[w]\n\t"
        "add $296, %[a]\n\t"
        "add $296, %[a1]\n\t"
        "sub $1, %[blocks]\n\t"
        "jg 1b\n\t"
        "2:\n\t"
        "vextractf128 $1, %%ymm0, %%xmm14\n\t"
        "vaddps %%xmm0, %%xmm14, %%xmm0\n\t"
        "vmovhlps %%xmm0, %%xmm0, %%xmm14\n\t"
        "vaddps %%xmm14, %%xmm0, %%xmm0\n\t"
        "vpermilps $1, %%xmm0, %%xmm14\n\t"
        "vaddss %%xmm14, %%xmm0, %%xmm0\n\t"
        "vmovss %%xmm0, 0(%[out])\n\t"
        "vextractf128 $1, %%ymm1, %%xmm14\n\t"
        "vaddps %%xmm1, %%xmm14, %%xmm1\n\t"
        "vmovhlps %%xmm1, %%xmm1, %%xmm14\n\t"
        "vaddps %%xmm14, %%xmm1, %%xmm1\n\t"
        "vpermilps $1, %%xmm1, %%xmm14\n\t"
        "vaddss %%xmm14, %%xmm1, %%xmm1\n\t"
        "vmovss %%xmm1, 4(%[out])\n\t"
        : [w] "+&r"(w), [a] "+&r"(a), [a1] "+&r"(a1), [blocks] "+&r"(blocks)
        : [k] "r"(&iq3_q8_asm_constants), [out] "r"(out)
        : "cc", "memory", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6",
          "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15");
}
// M4: 0..3=float accumulators, 4..7=integer sums, 8/9=scales,
// 10/11=packed bits, 12=decoded weights, 13=group scale, 14=LUT/product, 15=mask.
// Seed integer sums with the correction (integer addition is associative here).
// The LUT dies after each shuffle: reload read-only data, then reuse 14 for all
// four tokens' multiply chains. This fits 16 YMM without a stack workspace.
static void dot_iq3_q8_m4_asm(const void *row, const Activation *inputs, int n, float *out) {
    const auto *w = static_cast<const uint8_t *>(row);
    const auto *a0 = inputs[0].q8.data();
    const auto *a1 = inputs[1].q8.data();
    const auto *a2 = inputs[2].q8.data();
    const auto *a3 = inputs[3].q8.data();
    int blocks = n / 256;
    asm volatile(
        "vxorps %%xmm0, %%xmm0, %%xmm0\n\t"
        "vxorps %%xmm1, %%xmm1, %%xmm1\n\t"
        "vxorps %%xmm2, %%xmm2, %%xmm2\n\t"
        "vxorps %%xmm3, %%xmm3, %%xmm3\n\t"
        "test %[blocks], %[blocks]\n\t"
        "jle 2f\n\t"
        "vmovdqa 64(%[k]), %%ymm15\n\t"
        ".p2align 5\n\t"
        "1:\n\t"
        "vmovq 6(%[w]), %%xmm11\n\t"
        "vpsrlw $4, %%xmm11, %%xmm12\n\t"
        "vpunpcklbw %%xmm12, %%xmm11, %%xmm11\n\t"
        "vpand 192(%[k]), %%xmm11, %%xmm11\n\t"
        "vpaddb %%xmm11, %%xmm11, %%xmm11\n\t"
        "vpaddb 224(%[k]), %%xmm11, %%xmm11\n\t"
        "vpbroadcastq 2(%[w]), %%ymm12\n\t"
        "vpshufb 256(%[k]), %%ymm12, %%ymm12\n\t"
        "vpand 288(%[k]), %%ymm12, %%ymm12\n\t"
        "vpcmpeqb 288(%[k]), %%ymm12, %%ymm12\n\t"
        "vpor 224(%[k]), %%xmm12, %%xmm13\n\t"
        "vpsignb %%xmm13, %%xmm11, %%xmm11\n\t"
        "vpmovsxbw %%xmm11, %%ymm8\n\t"
        "vextracti128 $1, %%ymm12, %%xmm12\n\t"
        "vpand 128(%[k]), %%xmm12, %%xmm12\n\t"
        "vpmovsxbw %%xmm12, %%ymm12\n\t"
        "vpaddw 320(%[k]), %%ymm12, %%ymm12\n\t"
        "vpmullw %%ymm8, %%ymm12, %%ymm13\n\t"
        "vpmaddwd 264(%[a0]), %%ymm13, %%ymm4\n\t"
        "vpmaddwd 264(%[a1]), %%ymm13, %%ymm5\n\t"
        "vpmaddwd 264(%[a2]), %%ymm13, %%ymm6\n\t"
        "vpmaddwd 264(%[a3]), %%ymm13, %%ymm7\n\t"
        "vmovdqu 78(%[w]), %%ymm10\n\t"
        "vpermq $68, %%ymm8, %%ymm9\n\t"
        "vmovdqu 14(%[w]), %%ymm11\n\t"
        "vpsllw $2, %%ymm10, %%ymm13\n\t"
        "vpand 128(%[k]), %%ymm13, %%ymm13\n\t"
        "vpand %%ymm15, %%ymm11, %%ymm12\n\t"
        "vpor %%ymm13, %%ymm12, %%ymm12\n\t"
        "vmovdqa 0(%[k]), %%ymm14\n\t"
        "vpshufb %%ymm12, %%ymm14, %%ymm12\n\t"
        "vpshufb 352(%[k]), %%ymm9, %%ymm13\n\t"
        "vpmaddubsw 8(%[a0]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm4, %%ymm4\n\t"
        "vpmaddubsw 8(%[a1]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm5, %%ymm5\n\t"
        "vpmaddubsw 8(%[a2]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm6, %%ymm6\n\t"
        "vpmaddubsw 8(%[a3]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm7, %%ymm7\n\t"
        "vpand 160(%[k]), %%ymm10, %%ymm13\n\t"
        "vpand 96(%[k]), %%ymm11, %%ymm12\n\t"
        "vpor %%ymm13, %%ymm12, %%ymm12\n\t"
        "vmovdqa 32(%[k]), %%ymm14\n\t"
        "vpshufb %%ymm12, %%ymm14, %%ymm12\n\t"
        "vpshufb 384(%[k]), %%ymm9, %%ymm13\n\t"
        "vpmaddubsw 40(%[a0]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm4, %%ymm4\n\t"
        "vpmaddubsw 40(%[a1]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm5, %%ymm5\n\t"
        "vpmaddubsw 40(%[a2]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm6, %%ymm6\n\t"
        "vpmaddubsw 40(%[a3]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm7, %%ymm7\n\t"
        "vpsrlw $4, %%ymm11, %%ymm11\n\t"
        "vpand 128(%[k]), %%ymm10, %%ymm13\n\t"
        "vpand %%ymm15, %%ymm11, %%ymm12\n\t"
        "vpor %%ymm13, %%ymm12, %%ymm12\n\t"
        "vmovdqa 0(%[k]), %%ymm14\n\t"
        "vpshufb %%ymm12, %%ymm14, %%ymm12\n\t"
        "vpshufb 416(%[k]), %%ymm9, %%ymm13\n\t"
        "vpmaddubsw 72(%[a0]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm4, %%ymm4\n\t"
        "vpmaddubsw 72(%[a1]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm5, %%ymm5\n\t"
        "vpmaddubsw 72(%[a2]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm6, %%ymm6\n\t"
        "vpmaddubsw 72(%[a3]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm7, %%ymm7\n\t"
        "vpsrlw $2, %%ymm10, %%ymm10\n\t"
        "vpand 160(%[k]), %%ymm10, %%ymm13\n\t"
        "vpand 96(%[k]), %%ymm11, %%ymm12\n\t"
        "vpor %%ymm13, %%ymm12, %%ymm12\n\t"
        "vmovdqa 32(%[k]), %%ymm14\n\t"
        "vpshufb %%ymm12, %%ymm14, %%ymm12\n\t"
        "vpshufb 448(%[k]), %%ymm9, %%ymm13\n\t"
        "vpmaddubsw 104(%[a0]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm4, %%ymm4\n\t"
        "vpmaddubsw 104(%[a1]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm5, %%ymm5\n\t"
        "vpmaddubsw 104(%[a2]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm6, %%ymm6\n\t"
        "vpmaddubsw 104(%[a3]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm7, %%ymm7\n\t"
        "vpermq $238, %%ymm8, %%ymm9\n\t"
        "vmovdqu 46(%[w]), %%ymm11\n\t"
        "vpand 128(%[k]), %%ymm10, %%ymm13\n\t"
        "vpand %%ymm15, %%ymm11, %%ymm12\n\t"
        "vpor %%ymm13, %%ymm12, %%ymm12\n\t"
        "vmovdqa 0(%[k]), %%ymm14\n\t"
        "vpshufb %%ymm12, %%ymm14, %%ymm12\n\t"
        "vpshufb 352(%[k]), %%ymm9, %%ymm13\n\t"
        "vpmaddubsw 136(%[a0]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm4, %%ymm4\n\t"
        "vpmaddubsw 136(%[a1]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm5, %%ymm5\n\t"
        "vpmaddubsw 136(%[a2]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm6, %%ymm6\n\t"
        "vpmaddubsw 136(%[a3]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm7, %%ymm7\n\t"
        "vpsrlw $2, %%ymm10, %%ymm10\n\t"
        "vpand 160(%[k]), %%ymm10, %%ymm13\n\t"
        "vpand 96(%[k]), %%ymm11, %%ymm12\n\t"
        "vpor %%ymm13, %%ymm12, %%ymm12\n\t"
        "vmovdqa 32(%[k]), %%ymm14\n\t"
        "vpshufb %%ymm12, %%ymm14, %%ymm12\n\t"
        "vpshufb 384(%[k]), %%ymm9, %%ymm13\n\t"
        "vpmaddubsw 168(%[a0]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm4, %%ymm4\n\t"
        "vpmaddubsw 168(%[a1]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm5, %%ymm5\n\t"
        "vpmaddubsw 168(%[a2]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm6, %%ymm6\n\t"
        "vpmaddubsw 168(%[a3]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm7, %%ymm7\n\t"
        "vpsrlw $4, %%ymm11, %%ymm11\n\t"
        "vpand 128(%[k]), %%ymm10, %%ymm13\n\t"
        "vpand %%ymm15, %%ymm11, %%ymm12\n\t"
        "vpor %%ymm13, %%ymm12, %%ymm12\n\t"
        "vmovdqa 0(%[k]), %%ymm14\n\t"
        "vpshufb %%ymm12, %%ymm14, %%ymm12\n\t"
        "vpshufb 416(%[k]), %%ymm9, %%ymm13\n\t"
        "vpmaddubsw 200(%[a0]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm4, %%ymm4\n\t"
        "vpmaddubsw 200(%[a1]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm5, %%ymm5\n\t"
        "vpmaddubsw 200(%[a2]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm6, %%ymm6\n\t"
        "vpmaddubsw 200(%[a3]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm7, %%ymm7\n\t"
        "vpsrlw $2, %%ymm10, %%ymm10\n\t"
        "vpand 160(%[k]), %%ymm10, %%ymm13\n\t"
        "vpand 96(%[k]), %%ymm11, %%ymm12\n\t"
        "vpor %%ymm13, %%ymm12, %%ymm12\n\t"
        "vmovdqa 32(%[k]), %%ymm14\n\t"
        "vpshufb %%ymm12, %%ymm14, %%ymm12\n\t"
        "vpshufb 448(%[k]), %%ymm9, %%ymm13\n\t"
        "vpmaddubsw 232(%[a0]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm4, %%ymm4\n\t"
        "vpmaddubsw 232(%[a1]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm5, %%ymm5\n\t"
        "vpmaddubsw 232(%[a2]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm6, %%ymm6\n\t"
        "vpmaddubsw 232(%[a3]), %%ymm12, %%ymm14\n\t"
        "vpmaddwd %%ymm13, %%ymm14, %%ymm14\n\t"
        "vpaddd %%ymm14, %%ymm7, %%ymm7\n\t"
        "vpbroadcastw (%[w]), %%xmm14\n\t"
        "vcvtph2ps %%xmm14, %%xmm14\n\t"
        "vcvtdq2ps %%ymm4, %%ymm4\n\t"
        "vmulss (%[a0]), %%xmm14, %%xmm13\n\t"
        "vbroadcastss %%xmm13, %%ymm13\n\t"
        "vfmadd231ps %%ymm4, %%ymm13, %%ymm0\n\t"
        "vcvtdq2ps %%ymm5, %%ymm5\n\t"
        "vmulss (%[a1]), %%xmm14, %%xmm13\n\t"
        "vbroadcastss %%xmm13, %%ymm13\n\t"
        "vfmadd231ps %%ymm5, %%ymm13, %%ymm1\n\t"
        "vcvtdq2ps %%ymm6, %%ymm6\n\t"
        "vmulss (%[a2]), %%xmm14, %%xmm13\n\t"
        "vbroadcastss %%xmm13, %%ymm13\n\t"
        "vfmadd231ps %%ymm6, %%ymm13, %%ymm2\n\t"
        "vcvtdq2ps %%ymm7, %%ymm7\n\t"
        "vmulss (%[a3]), %%xmm14, %%xmm13\n\t"
        "vbroadcastss %%xmm13, %%ymm13\n\t"
        "vfmadd231ps %%ymm7, %%ymm13, %%ymm3\n\t"
        "add $110, %[w]\n\t"
        "add $296, %[a0]\n\t"
        "add $296, %[a1]\n\t"
        "add $296, %[a2]\n\t"
        "add $296, %[a3]\n\t"
        "sub $1, %[blocks]\n\t"
        "jg 1b\n\t"
        "2:\n\t"
        "vextractf128 $1, %%ymm0, %%xmm14\n\t"
        "vaddps %%xmm0, %%xmm14, %%xmm0\n\t"
        "vmovhlps %%xmm0, %%xmm0, %%xmm14\n\t"
        "vaddps %%xmm14, %%xmm0, %%xmm0\n\t"
        "vpermilps $1, %%xmm0, %%xmm14\n\t"
        "vaddss %%xmm14, %%xmm0, %%xmm0\n\t"
        "vmovss %%xmm0, 0(%[out])\n\t"
        "vextractf128 $1, %%ymm1, %%xmm14\n\t"
        "vaddps %%xmm1, %%xmm14, %%xmm1\n\t"
        "vmovhlps %%xmm1, %%xmm1, %%xmm14\n\t"
        "vaddps %%xmm14, %%xmm1, %%xmm1\n\t"
        "vpermilps $1, %%xmm1, %%xmm14\n\t"
        "vaddss %%xmm14, %%xmm1, %%xmm1\n\t"
        "vmovss %%xmm1, 4(%[out])\n\t"
        "vextractf128 $1, %%ymm2, %%xmm14\n\t"
        "vaddps %%xmm2, %%xmm14, %%xmm2\n\t"
        "vmovhlps %%xmm2, %%xmm2, %%xmm14\n\t"
        "vaddps %%xmm14, %%xmm2, %%xmm2\n\t"
        "vpermilps $1, %%xmm2, %%xmm14\n\t"
        "vaddss %%xmm14, %%xmm2, %%xmm2\n\t"
        "vmovss %%xmm2, 8(%[out])\n\t"
        "vextractf128 $1, %%ymm3, %%xmm14\n\t"
        "vaddps %%xmm3, %%xmm14, %%xmm3\n\t"
        "vmovhlps %%xmm3, %%xmm3, %%xmm14\n\t"
        "vaddps %%xmm14, %%xmm3, %%xmm3\n\t"
        "vpermilps $1, %%xmm3, %%xmm14\n\t"
        "vaddss %%xmm14, %%xmm3, %%xmm3\n\t"
        "vmovss %%xmm3, 12(%[out])\n\t"
        : [w] "+&r"(w), [blocks] "+&r"(blocks), [a0] "+&r"(a0), [a1] "+&r"(a1), [a2] "+&r"(a2), [a3] "+&r"(a3)
        : [k] "r"(&iq3_q8_asm_constants), [out] "r"(out)
        : "cc", "memory", "xmm0", "xmm1", "xmm2", "xmm3", "xmm4", "xmm5", "xmm6", "xmm7", "xmm8", "xmm9", "xmm10", "xmm11", "xmm12", "xmm13", "xmm14", "xmm15");
}

#endif

// Exact AVX2 nrc_x=nrc_y=1 specialization of ik mul_mat_qY_K_q8_K_T.
__attribute__((noinline)) float dot_iq3_q8(const void *row, const Activation &x, int n) {
#if FT_IQ3_Q8_ASM && defined(__x86_64__) && defined(__GNUC__)
    return dot_iq3_q8_asm(row, x, n);
#else
    NativeQ8 q8(x); DequantizerIQ3K deq(row,0); deq.new_row(0);
    __m256 acc[1]={_mm256_setzero_ps()};
    for(int i=0;i<n/256;++i) {
        MCA_BEGIN("iq3_q8");
        __m256i all_scales[2],scales[4],sumi[1];
        deq.new_block(i,q8,acc,all_scales);
        #pragma GCC unroll 2
        for(int j=0;j<2;++j) {
            deq.prepare(i,j); set_scales_16(all_scales[j],scales);
            multiply_add(deq.bits,scales,j,i,q8,sumi);
        }
        acc[0]=_mm256_fmadd_ps(_mm256_set1_ps(deq.d*q8.scale(0,i)),_mm256_cvtepi32_ps(sumi[0]),acc[0]);
        MCA_END("iq3_q8");
    }
    return horizontal(acc[0]);
#endif
}
// Exact AVX2 nrc_x=nrc_y=1 specialization of ik mul_mat_qX_K_q8_K_T.
__attribute__((noinline)) float dot_iq4_q8(const void *row, const Activation &x, int n) {
    NativeQ8 q8(x); DequantizerIQ4KS deq(row,0); deq.new_row(0);
    __m256 acc[1]={_mm256_setzero_ps()};
    for(int i=0;i<n/256;++i) {
        MCA_BEGIN("iq4_q8");
        auto all_scales=deq.new_block(i,q8,acc);
        __m256i scales[4],sumi[1];
        #pragma GCC unroll 2
        for(int j=0;j<2;++j) {
            deq.prepare(i,j); set_scales_8(all_scales,j,scales);
            multiply_add_avx2(deq.bits,scales,j,i,q8,sumi);
        }
        acc[0]=_mm256_fmadd_ps(_mm256_set1_ps(deq.d*q8.scale(0,i)),_mm256_cvtepi32_ps(sumi[0]),acc[0]);
        MCA_END("iq4_q8");
    }
    return horizontal(acc[0]);
}
namespace {
template<bool iq3> __attribute__((always_inline)) inline float dot_i16_impl(const void *row,const Activation &x,int n) {
    using Deq=std::conditional_t<iq3,DequantizerIQ3K,DequantizerIQ4KS>;
    Deq deq(row,0); deq.new_row(0);
    __m256 acc=_mm256_setzero_ps();
    for(int i=0;i<n/256;++i) {
        if constexpr(iq3) { MCA_BEGIN("iq3_i16"); } else { MCA_BEGIN("iq4_i16"); }
        __m256i all_scales[2];
        if constexpr(iq3) {
            deq.d=half_to_float(deq.x[i].d);
            auto sc=_mm256_cvtepi8_epi16(_mm_shuffle_epi8(deq.make_scales(deq.x[i].scales_h,deq.x[i].scales_l),deq.iqxk.hshuff));
            prepare_scales_16(sc,all_scales);
            deq.hbits=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(deq.x[i].qh));
        } else {
            auto sc=_mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(deq.x[i].scales)));
            sc=_mm_add_epi16(_mm_and_si128(sc,_mm_set1_epi16(254)),_mm_set1_epi16(-127));
            all_scales[0]=all_scales[1]=MM256_SET1_M128I(sc);
        }
        #pragma GCC unroll 2
        for(int j=0;j<2;++j) {
            deq.prepare(i,j);
            __m256i scales[4];
            if constexpr(iq3) set_scales_16(all_scales[j],scales);
            else set_scales_8(all_scales[0],j,scales);
            // IQ4_KS worst case: 127 LUT * 127 scale * 16383 activation.
            // Two i32 vectors each see <=8 products/lane per128 inputs:
            // 8*127*127*16383 = 2,113,931,256 < INT32_MAX.
            // Convert each vector separately before their sum; summing them
            // as i32 could overflow for adversarial all-positive inputs.
            __m256i sums[2]={_mm256_setzero_si256(),_mm256_setzero_si256()};
            #pragma GCC unroll 4
            for(int k=0;k<4;++k) {
                __m256i v=deq.bits.values[k];
                if constexpr(iq3) {
                    unsigned extra=deq.x[i].extra >> (8*j+2*k);
                    auto off=MM256_SET_M128I(_mm_set1_epi8(-64+4*((extra>>1)&1)),_mm_set1_epi8(-64+4*(extra&1)));
                    v=_mm256_add_epi8(v,off);
                }
                auto wl=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
                auto wh=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(v,1));
                wl=_mm256_mullo_epi16(wl,MM256_SET1_M128I(_mm256_castsi256_si128(scales[k])));
                wh=_mm256_mullo_epi16(wh,MM256_SET1_M128I(_mm256_extracti128_si256(scales[k],1)));
                const auto *a=x.i16.data()+256*i+128*j+32*k;
                auto lo=_mm256_madd_epi16(wl,_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a)));
                auto hi=_mm256_madd_epi16(wh,_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a+16)));
                sums[k&1]=_mm256_add_epi32(sums[k&1],_mm256_add_epi32(lo,hi));
            }
            auto sf=_mm256_add_ps(_mm256_cvtepi32_ps(sums[0]),_mm256_cvtepi32_ps(sums[1]));
            acc=_mm256_fmadd_ps(_mm256_set1_ps(deq.d*x.scales[2*i+j]),sf,acc);
        }
        if constexpr(iq3) { MCA_END("iq3_i16"); } else { MCA_END("iq4_i16"); }
    }
    return horizontal(acc);
}
}
__attribute__((noinline)) float dot_iq3_i16(const void *row,const Activation &x,int n) { return dot_i16_impl<true>(row,x,n); }
__attribute__((noinline)) float dot_iq4_i16(const void *row,const Activation &x,int n) { return dot_i16_impl<false>(row,x,n); }
float dot(Format format,const void *row,const Activation &x,int n) {
    require_n(n);
    if(x.n!=n) throw std::invalid_argument("activation shape mismatch");
    if(format==Format::Q3_K || format==Format::Q4_K) {
        float out; dot_q34_dispatch(format,row,&x,n,1,&out,Kernel::Baseline); return out;
    }
    if(format==Format::Q8_0) {
        float out; dot_q8_0_dispatch(row,&x,n,1,&out,Kernel::Baseline); return out;
    }
    if(format==Format::Q6_K) {
        float out; dot_q6_dispatch(row,&x,n,1,&out,Kernel::Baseline); return out;
    }
    if(format==Format::IQ3_S || format==Format::IQ4_XS) {
        float out; dot_legacy_formats(format,row,&x,n,1,&out,Kernel::Baseline); return out;
    }
    if(format==Format::IQ3_K) return x.path==Path::Q8 ? dot_iq3_q8(row,x,n):dot_iq3_i16(row,x,n);
    return x.path==Path::Q8 ? dot_iq4_q8(row,x,n):dot_iq4_i16(row,x,n);
}

// R2 reuses each decoded32-weight group across all tokens. IQ3/I16/M1
// decomposes each original14-bit activation exactly as lo+128*hi, where
// lo=a%128 and hi=a/128 both fit signed7 bits. Two native unsigned-LUT dots
// and their precomputed sums supply the exact integer result; the original
// per128 activation scales are unchanged. M2/M4 and IQ4 use lane-local
// i16 unpacking with a prepared permutation, reusing each weight scale.
// No path changes the packed weight layout or activation quantization.
namespace {
template<bool IQ3, bool I16, int M>
__attribute__((always_inline)) inline void optimized_impl(const void *row,const Activation *inputs,int n,float *outputs) {
    constexpr bool Split=IQ3 && I16 && M==1;
    using Deq=std::conditional_t<IQ3,DequantizerIQ3K,DequantizerIQ4KS>;
    Deq deq(row,0); deq.new_row(0);
    const block_q8_K *q8[M];
    const int8_t *i16_lo[M], *i16_hi[M];
    const int16_t *i16[M];
    const float *activation_scales[M];
    __m256 acc[M];
    #pragma GCC unroll 4
    for(int m=0;m<M;++m) {
        q8[m]=reinterpret_cast<const block_q8_K *>(inputs[m].q8.data());
        i16[m]=inputs[m].i16_swizzled.data();
        i16_lo[m]=inputs[m].i16_lo.data(); i16_hi[m]=inputs[m].i16_hi.data();
        activation_scales[m]=inputs[m].scales.data();
        acc[m]=_mm256_setzero_ps();
    }
    const __m256i mask3=_mm256_set1_epi8(3),mask1=_mm256_set1_epi8(1);
    const __m256i zero=_mm256_setzero_si256();
    const __m256i iq3table=MM256_SET1_M128I(_mm_loadu_si128(reinterpret_cast<const __m128i *>(iq3nl_values)));
    const __m256i iq3unsigned=MM256_SET1_M128I(_mm_setr_epi8(1,24,41,54,65,77,92,111,5,28,45,58,69,81,96,115));
    const __m128i extra_mask=_mm_set_epi32(0x80804040,0x20201010,0x08080404,0x02020101);
    const __m128i extra_order=_mm_setr_epi8(0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15);
    for(int b=0;b<n/256;++b) {
        __m256i all_scales[2],correction,extra_flags;
        if constexpr(IQ3) {
            deq.d=half_to_float(deq.x[b].d);
            auto s8=deq.make_scales(deq.x[b].scales_h,deq.x[b].scales_l);
            auto s16=_mm256_cvtepi8_epi16(_mm_shuffle_epi8(s8,deq.iqxk.hshuff));
            prepare_scales_16(s16,all_scales);
            auto e=_mm_set1_epi16(deq.x[b].extra);
            e=_mm_cmpeq_epi8(_mm_and_si128(e,extra_mask),extra_mask);
            e=_mm_shuffle_epi8(e,extra_order);
            if constexpr(I16 && !Split) extra_flags=MM256_SET1_M128I(_mm_and_si128(e,_mm_set1_epi8(8)));
            else correction=_mm256_mullo_epi16(s16,_mm256_add_epi16(_mm256_set1_epi16(-64),
                                      _mm256_cvtepi8_epi16(_mm_and_si128(e,_mm_set1_epi8(4)))));
        } else {
            auto s16=_mm_cvtepu8_epi16(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(deq.x[b].scales)));
            s16=_mm_add_epi16(_mm_and_si128(s16,_mm_set1_epi16(254)),_mm_set1_epi16(-127));
            all_scales[0]=all_scales[1]=MM256_SET1_M128I(s16);
        }
        __m256i q8sum[M];
        if constexpr(!I16) {
            #pragma GCC unroll 4
            for(int m=0;m<M;++m) q8sum[m]=zero;
        }
        __m256i extra_shuffle=MM256_SET_M128I(_mm_set1_epi8(1),_mm_set1_epi8(0));
        #pragma GCC unroll 2
        for(int h=0;h<2;++h) {
            __m256i lo,hi;
            if constexpr(IQ3) {
                lo=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(deq.x[b].qs)+h);
                hi=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(deq.x[b].qh));
                if(h) hi=_mm256_srli_epi16(hi,4);
            }
            __m256i scale_shuffle;
            if constexpr(IQ3) scale_shuffle=MM256_SET_M128I(_mm_set1_epi16(0x0302),_mm_set1_epi16(0x0100));
            else scale_shuffle=_mm256_set1_epi16((8*h)|((8*h+1)<<8));
            __m256i sum0[M],sum1[M];
            if constexpr(I16) {
                #pragma GCC unroll 4
                for(int m=0;m<M;++m) sum0[m]=sum1[m]=zero;
            }
            #pragma GCC unroll 4
            for(int k=0;k<4;++k) {
                __m256i v;
                if constexpr(IQ3) {
                    auto idx=_mm256_or_si256(_mm256_and_si256(lo,mask3),
                                            _mm256_slli_epi16(_mm256_and_si256(hi,mask1),2));
                    lo=_mm256_srli_epi16(lo,2); hi=_mm256_srli_epi16(hi,1);
                    if constexpr(I16 && !Split) {
                        idx=_mm256_or_si256(idx,_mm256_shuffle_epi8(extra_flags,extra_shuffle));
                        v=_mm256_shuffle_epi8(iq3table,idx);
                        extra_shuffle=_mm256_add_epi8(extra_shuffle,_mm256_set1_epi8(2));
                    } else v=_mm256_shuffle_epi8(iq3unsigned,idx);
                } else {
                    const int g=4*h+k;
                    auto packed=_mm256_broadcastsi128_si256(_mm_loadu_si128(reinterpret_cast<const __m128i *>(deq.x[b].qs+16*g)));
                    auto idx=_mm256_and_si256(_mm256_blend_epi32(packed,_mm256_srli_epi16(packed,4),0xF0),_mm256_set1_epi8(15));
                    v=_mm256_shuffle_epi8(deq.values[deq.x[b].scales[g]&1],idx);
                }
                auto scales=_mm256_shuffle_epi8(all_scales[IQ3 ? h : 0],scale_shuffle);
                scale_shuffle=_mm256_add_epi8(scale_shuffle,_mm256_set1_epi8(IQ3 ? 4 : 2));
                if constexpr(I16) {
                    if constexpr(Split) {
                        // a=lo+128*hi is exact, with both signed parts in
                        // [-127,127]. Native unsigned IQ3 LUT values<=111
                        // keep every maddubs pair below signed16 saturation.
                        #pragma GCC unroll 4
                        for(int m=0;m<M;++m) {
                            const int offset=256*b+128*h+32*k;
                            auto al=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(i16_lo[m]+offset));
                            auto ah=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(i16_hi[m]+offset));
                            sum0[m]=_mm256_add_epi32(sum0[m],_mm256_madd_epi16(scales,_mm256_maddubs_epi16(v,al)));
                            sum1[m]=_mm256_add_epi32(sum1[m],_mm256_madd_epi16(scales,_mm256_maddubs_epi16(v,ah)));
                        }
                    } else {
                        auto sign=_mm256_cmpgt_epi8(zero,v);
                        auto wl=_mm256_mullo_epi16(_mm256_unpacklo_epi8(v,sign),scales);
                        auto wh=_mm256_mullo_epi16(_mm256_unpackhi_epi8(v,sign),scales);
                        #pragma GCC unroll 4
                        for(int m=0;m<M;++m) {
                            const auto *a=i16[m]+256*b+128*h+32*k;
                            auto p=_mm256_add_epi32(_mm256_madd_epi16(wl,_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a))),
                                                   _mm256_madd_epi16(wh,_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a+16))));
                            if constexpr(IQ3) sum0[m]=_mm256_add_epi32(sum0[m],p);
                            else if(k&1) sum1[m]=_mm256_add_epi32(sum1[m],p);
                            else sum0[m]=_mm256_add_epi32(sum0[m],p);
                        }
                    }
                } else {
                    __m256i magnitude;
                    if constexpr(!IQ3) magnitude=_mm256_sign_epi8(v,v);
                    #pragma GCC unroll 4
                    for(int m=0;m<M;++m) {
                        auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8[m][b].qs+128*h+32*k));
                        __m256i pairs;
                        if constexpr(IQ3) pairs=_mm256_maddubs_epi16(v,a);
                        else pairs=_mm256_maddubs_epi16(magnitude,_mm256_sign_epi8(a,v));
                        q8sum[m]=_mm256_add_epi32(q8sum[m],_mm256_madd_epi16(scales,pairs));
                        if constexpr(!IQ3 && M>=2) asm("" : "+x"(q8sum[m]));
                    }
                }
            }
            if constexpr(I16) {
                #pragma GCC unroll 4
                for(int m=0;m<M;++m) {
                    if constexpr(Split) {
                        const auto c=h ? _mm256_extracti128_si256(correction,1) : _mm256_castsi256_si128(correction);
                        auto sl=_mm_loadu_si128(reinterpret_cast<const __m128i *>(inputs[m].i16_lo_bsums.data()+16*b+8*h));
                        auto sh=_mm_loadu_si128(reinterpret_cast<const __m128i *>(inputs[m].i16_hi_bsums.data()+16*b+8*h));
                        sum0[m]=_mm256_add_epi32(sum0[m],_mm256_inserti128_si256(zero,_mm_madd_epi16(c,sl),0));
                        sum1[m]=_mm256_add_epi32(sum1[m],_mm256_inserti128_si256(zero,_mm_madd_epi16(c,sh),0));
                        // Even after assigning correction sums to the
                        // low128 lanes, the conservative full14-bit bound
                        // is(16*111+32*64)*31*16383=1,942,106,352<i32max.
                        sum0[m]=_mm256_add_epi32(sum0[m],_mm256_slli_epi32(sum1[m],7));
                    }
                    auto f=_mm256_cvtepi32_ps(sum0[m]);
                    if constexpr(!IQ3) f=_mm256_add_ps(f,_mm256_cvtepi32_ps(sum1[m]));
                    acc[m]=_mm256_fmadd_ps(_mm256_set1_ps(deq.d*activation_scales[m][2*b+h]),f,acc[m]);
                }
            }
        }
        if constexpr(!I16) {
            #pragma GCC unroll 4
            for(int m=0;m<M;++m) {
                if constexpr(IQ3) q8sum[m]=_mm256_add_epi32(q8sum[m],_mm256_madd_epi16(correction,
                                  _mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8[m][b].bsums))));
                acc[m]=_mm256_fmadd_ps(_mm256_set1_ps(deq.d*q8[m][b].d),_mm256_cvtepi32_ps(q8sum[m]),acc[m]);
            }
        }
    }
    #pragma GCC unroll 4
    for(int m=0;m<M;++m) outputs[m]=horizontal(acc[m]);
}
} // anonymous namespace
#define DEFINE_OPTIMIZED(name,iq3,i16,m) \
__attribute__((noinline)) void name(const void *row,const Activation *inputs,int n,float *outputs) { optimized_impl<iq3,i16,m>(row,inputs,n,outputs); }
DEFINE_OPTIMIZED(dot_iq3_i16_optimized_m1,true,true,1)
DEFINE_OPTIMIZED(dot_iq3_i16_optimized_m2,true,true,2)
DEFINE_OPTIMIZED(dot_iq3_i16_optimized_m4,true,true,4)
DEFINE_OPTIMIZED(dot_iq4_i16_optimized_m1,false,true,1)
DEFINE_OPTIMIZED(dot_iq4_i16_optimized_m2,false,true,2)
DEFINE_OPTIMIZED(dot_iq4_i16_optimized_m4,false,true,4)
DEFINE_OPTIMIZED(dot_iq3_q8_optimized_m1,true,false,1)
#if FT_IQ3_Q8_ASM && defined(__x86_64__) && defined(__GNUC__)
__attribute__((noinline)) void dot_iq3_q8_optimized_m2(const void *row, const Activation *inputs, int n, float *out) { dot_iq3_q8_m2_asm(row, inputs, n, out); }
#else
DEFINE_OPTIMIZED(dot_iq3_q8_optimized_m2,true,false,2)
#endif
#if FT_IQ3_Q8_ASM && defined(__x86_64__) && defined(__GNUC__)
__attribute__((noinline)) void dot_iq3_q8_optimized_m4(const void *row, const Activation *inputs, int n, float *out) { dot_iq3_q8_m4_asm(row, inputs, n, out); }
#else
DEFINE_OPTIMIZED(dot_iq3_q8_optimized_m4,true,false,4)
#endif
DEFINE_OPTIMIZED(dot_iq4_q8_optimized_m1,false,false,1)
DEFINE_OPTIMIZED(dot_iq4_q8_optimized_m2,false,false,2)
DEFINE_OPTIMIZED(dot_iq4_q8_optimized_m4,false,false,4)
#undef DEFINE_OPTIMIZED
void dot_multi(Format format,const void *row,const Activation *inputs,int n,int m,float *outputs,Kernel kernel) {
    require_n(n);
    if(m!=1 && m!=2 && m!=4) throw std::invalid_argument("dot_multi supports M=1,2,4");
    for(int t=0;t<m;++t) if(inputs[t].n!=n || inputs[t].path!=inputs[0].path)
        throw std::invalid_argument("activation shape or path mismatch");
    if(kernel==Kernel::Baseline) { for(int t=0;t<m;++t) outputs[t]=dot(format,row,inputs[t],n); return; }
    const bool i16=inputs[0].path==Path::I16;
    // IQ4 M1 has the same integer lanes and FMA order in the shared loop.
    if(!i16 && m==1 && format!=Format::IQ4_KS) { outputs[0]=dot(format,row,inputs[0],n); return; }
    if(i16) for(int t=0;t<m;++t) if(inputs[t].kernel!=Kernel::Optimized)
        throw std::invalid_argument("optimized i16 kernel requires optimized activation preparation");
    if(format==Format::Q3_K || format==Format::Q4_K) {
        dot_q34_dispatch(format,row,inputs,n,m,outputs,Kernel::Optimized); return;
    }
    if(format==Format::Q8_0) {
        dot_q8_0_dispatch(row,inputs,n,m,outputs,Kernel::Optimized); return;
    }
    if(format==Format::Q6_K) {
        dot_q6_dispatch(row,inputs,n,m,outputs,Kernel::Optimized); return;
    }
    if(format==Format::IQ3_S || format==Format::IQ4_XS) {
        dot_legacy_formats(format,row,inputs,n,m,outputs,Kernel::Optimized); return;
    }
#define RUN_OPT(prefix) do { if(m==1) prefix##_m1(row,inputs,n,outputs); else if(m==2) prefix##_m2(row,inputs,n,outputs); else prefix##_m4(row,inputs,n,outputs); } while(0)
    if(format==Format::IQ3_K) { if(i16) RUN_OPT(dot_iq3_i16_optimized); else RUN_OPT(dot_iq3_q8_optimized); }
    else { if(i16) RUN_OPT(dot_iq4_i16_optimized); else RUN_OPT(dot_iq4_q8_optimized); }
#undef RUN_OPT
}
float dot(Format format,const void *row,const Activation &x,int n,Kernel kernel) {
    if(kernel==Kernel::Baseline) return dot(format,row,x,n);
    float out; dot_multi(format,row,&x,n,1,&out,kernel); return out;
}

// R4 native IQ3_S/IQ4_XS AVX2 dots. Grid-index expansion, scalar indexed
// 512-grid loads, sign expansion, maddubs/maddwd and scaling follow ik
// ggml-quants.c:10954-11040 and12291-12326; IQ3 index helper corresponds to
// iqk_gemm_iquants.cpp:601-621. Each decoded group serves every token.
namespace {
__attribute__((always_inline)) inline __m256i iq3_s_grid32(const uint8_t *qs,uint8_t qh) {
    union { __m256i vector; uint32_t index[8]; } indices;
    auto low=_mm256_cvtepu8_epi32(_mm_loadl_epi64(reinterpret_cast<const __m128i *>(qs)));
    auto high=_mm256_and_si256(_mm256_sllv_epi32(_mm256_set1_epi32(qh),_mm256_set_epi32(1,2,3,4,5,6,7,8)),_mm256_set1_epi32(256));
    indices.vector=_mm256_or_si256(low,high);
    return _mm256_set_epi32(iq3s_grid[indices.index[7]],iq3s_grid[indices.index[6]],iq3s_grid[indices.index[5]],iq3s_grid[indices.index[4]],
                           iq3s_grid[indices.index[3]],iq3s_grid[indices.index[2]],iq3s_grid[indices.index[1]],iq3s_grid[indices.index[0]]);
}
template<bool IQ3S,bool I16,int M,bool Swizzled=true>
__attribute__((always_inline)) inline void legacy_iq_impl(const void *row,const Activation *inputs,int n,float *outputs) {
    const auto *iq3=static_cast<const block_iq3_s *>(row);
    const auto *iq4=static_cast<const block_iq4_xs *>(row);
    const block_q8_K *q8[M]; const int16_t *i16[M]; const float *scales[M];
    __m256 acc[M];
    #pragma GCC unroll 4
    for(int m=0;m<M;++m) {
        q8[m]=reinterpret_cast<const block_q8_K *>(inputs[m].q8.data());
        i16[m]=Swizzled ? inputs[m].i16_swizzled.data() : inputs[m].i16.data();
        scales[m]=inputs[m].scales.data(); acc[m]=_mm256_setzero_ps();
    }
    const auto zero=_mm256_setzero_si256();
    const auto values=MM256_SET1_M128I(_mm_loadu_si128(reinterpret_cast<const __m128i *>(iq4k_values)));
    const auto sign_shuffle=MM256_SET_M128I(_mm_setr_epi8(2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3),
                                          _mm_setr_epi8(0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1));
    const auto sign_bits=_mm256_set1_epi64x(INT64_C(0x8040201008040201));
    for(int b=0;b<n/256;++b) {
        const float d=half_to_float(IQ3S ? iq3[b].d : iq4[b].d);
        __m256i block_sums[M];
        if constexpr(!I16) {
            #pragma GCC unroll 4
            for(int m=0;m<M;++m) block_sums[m]=zero;
        }
        #pragma GCC unroll 2
        for(int h=0;h<2;++h) {
            __m256i half_sums[M];
            if constexpr(I16) {
                #pragma GCC unroll 4
                for(int m=0;m<M;++m) half_sums[m]=zero;
            }
            #pragma GCC unroll 4
            for(int k=0;k<4;++k) {
                const int g=4*h+k;
                __m256i v,negative,weight_scale;
                if constexpr(IQ3S) {
                    v=iq3_s_grid32(iq3[b].qs+8*g,iq3[b].qh[g]);
                    uint32_t signs; std::memcpy(&signs,iq3[b].signs+4*g,sizeof(signs));
                    auto sign_bytes=_mm256_shuffle_epi8(_mm256_set1_epi32(int32_t(signs)),sign_shuffle);
                    negative=_mm256_cmpeq_epi8(_mm256_and_si256(sign_bytes,sign_bits),sign_bits);
                    const int scale=2*((iq3[b].scales[g/2]>>(4*(g&1)))&15)+1;
                    weight_scale=_mm256_set1_epi16(int16_t(scale));
                } else {
                    auto packed=_mm_loadu_si128(reinterpret_cast<const __m128i *>(iq4[b].qs+16*g));
                    auto indices=_mm256_and_si256(MM256_SET_M128I(_mm_srli_epi16(packed,4),packed),_mm256_set1_epi8(15));
                    v=_mm256_shuffle_epi8(values,indices);
                    negative=_mm256_cmpgt_epi8(zero,v);
                    const int scale=(((iq4[b].scales_l[g/2]>>(4*(g&1)))&15)|(((iq4[b].scales_h>>(2*g))&3)<<4))-32;
                    weight_scale=_mm256_set1_epi16(int16_t(scale));
                }
                if constexpr(I16) {
                    if constexpr(IQ3S) v=_mm256_sub_epi8(_mm256_xor_si256(v,negative),negative);
                    __m256i wl,wh;
                    if constexpr(Swizzled) {
                        wl=_mm256_mullo_epi16(_mm256_unpacklo_epi8(v,negative),weight_scale);
                        wh=_mm256_mullo_epi16(_mm256_unpackhi_epi8(v,negative),weight_scale);
                    } else {
                        wl=_mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(v)),weight_scale);
                        wh=_mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(v,1)),weight_scale);
                    }
                    // IQ3: |w|<=15*31=465; IQ4_XS: |w|<=127*32=4064.
                    // <=16 products/lane per128 inputs means worst case
                    // 16*4064*16383=1,065,288,192<INT32_MAX.
                    #pragma GCC unroll 4
                    for(int m=0;m<M;++m) {
                        const auto *a=i16[m]+256*b+32*g;
                        auto p=_mm256_add_epi32(_mm256_madd_epi16(wl,_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a))),
                                               _mm256_madd_epi16(wh,_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a+16))));
                        half_sums[m]=_mm256_add_epi32(half_sums[m],p);
                    }
                } else {
                    __m256i magnitude;
                    if constexpr(!IQ3S) magnitude=_mm256_sign_epi8(v,v);
                    #pragma GCC unroll 4
                    for(int m=0;m<M;++m) {
                        auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8[m][b].qs+32*g));
                        __m256i pairs;
                        if constexpr(IQ3S) pairs=_mm256_maddubs_epi16(v,_mm256_sub_epi8(_mm256_xor_si256(a,negative),negative));
                        else pairs=_mm256_maddubs_epi16(magnitude,_mm256_sign_epi8(a,v));
                        block_sums[m]=_mm256_add_epi32(block_sums[m],_mm256_madd_epi16(pairs,weight_scale));
                    }
                }
            }
            if constexpr(I16) {
                #pragma GCC unroll 4
                for(int m=0;m<M;++m)
                    acc[m]=_mm256_fmadd_ps(_mm256_set1_ps(d*scales[m][2*b+h]),_mm256_cvtepi32_ps(half_sums[m]),acc[m]);
            }
        }
        if constexpr(!I16) {
            #pragma GCC unroll 4
            for(int m=0;m<M;++m)
                acc[m]=_mm256_fmadd_ps(_mm256_set1_ps(d*q8[m][b].d),_mm256_cvtepi32_ps(block_sums[m]),acc[m]);
        }
    }
    #pragma GCC unroll 4
    for(int m=0;m<M;++m) outputs[m]=horizontal(acc[m]);
}
} // anonymous namespace
#define DEFINE_LEGACY_IQ(name,iq3,i16,m) \
__attribute__((noinline)) void name(const void *row,const Activation *inputs,int n,float *outputs) { legacy_iq_impl<iq3,i16,m>(row,inputs,n,outputs); }
DEFINE_LEGACY_IQ(dot_iq3_s_q8_m1,true,false,1)
DEFINE_LEGACY_IQ(dot_iq3_s_q8_m2,true,false,2)
DEFINE_LEGACY_IQ(dot_iq3_s_q8_m4,true,false,4)
DEFINE_LEGACY_IQ(dot_iq3_s_i16_m1,true,true,1)
DEFINE_LEGACY_IQ(dot_iq3_s_i16_m2,true,true,2)
DEFINE_LEGACY_IQ(dot_iq3_s_i16_m4,true,true,4)
DEFINE_LEGACY_IQ(dot_iq4_xs_q8_m1,false,false,1)
DEFINE_LEGACY_IQ(dot_iq4_xs_q8_m2,false,false,2)
DEFINE_LEGACY_IQ(dot_iq4_xs_q8_m4,false,false,4)
DEFINE_LEGACY_IQ(dot_iq4_xs_i16_m1,false,true,1)
DEFINE_LEGACY_IQ(dot_iq4_xs_i16_m2,false,true,2)
DEFINE_LEGACY_IQ(dot_iq4_xs_i16_m4,false,true,4)
#undef DEFINE_LEGACY_IQ
__attribute__((noinline)) void dot_iq3_s_i16_baseline_m1(const void *row,const Activation *inputs,int n,float *outputs) { legacy_iq_impl<true,true,1,false>(row,inputs,n,outputs); }
__attribute__((noinline)) void dot_iq4_xs_i16_baseline_m1(const void *row,const Activation *inputs,int n,float *outputs) { legacy_iq_impl<false,true,1,false>(row,inputs,n,outputs); }
namespace {
void dot_legacy_formats(Format format,const void *row,const Activation *inputs,int n,int m,float *outputs,Kernel kernel) {
    const bool i16=inputs[0].path==Path::I16;
    if(i16 && kernel==Kernel::Baseline) {
        if(format==Format::IQ3_S) dot_iq3_s_i16_baseline_m1(row,inputs,n,outputs);
        else dot_iq4_xs_i16_baseline_m1(row,inputs,n,outputs);
        return;
    }
#define RUN_LEGACY(prefix) do { if(m==1) prefix##_m1(row,inputs,n,outputs); else if(m==2) prefix##_m2(row,inputs,n,outputs); else prefix##_m4(row,inputs,n,outputs); } while(0)
    if(format==Format::IQ3_S) { if(i16) RUN_LEGACY(dot_iq3_s_i16); else RUN_LEGACY(dot_iq3_s_q8); }
    else { if(i16) RUN_LEGACY(dot_iq4_xs_i16); else RUN_LEGACY(dot_iq4_xs_q8); }
#undef RUN_LEGACY
}
} // anonymous namespace
// Q6_K AVX2 unpack and unsigned-dot/min correction follow ik's native
// ggml-quants.c ggml_vec_dot_q6_K_q8_K AVX2 branch. Exact I16 uses the same
// unpacked signed values with full14-bit activations and per128 scales.
namespace {
template<bool I16,int M,bool Swizzled=true>
__attribute__((always_inline)) inline void q6_impl(const void *row,const Activation *inputs,int n,float *outputs) {
    const auto *weights=static_cast<const block_q6_K *>(row);
    const block_q8_K *q8[M]; const int16_t *i16[M]; const float *scales[M]; __m256 acc[M];
    #pragma GCC unroll 4
    for(int m=0;m<M;++m) {
        q8[m]=reinterpret_cast<const block_q8_K *>(inputs[m].q8.data());
        i16[m]=Swizzled ? inputs[m].i16_swizzled.data() : inputs[m].i16.data();
        scales[m]=inputs[m].scales.data(); acc[m]=_mm256_setzero_ps();
    }
    const auto zero=_mm256_setzero_si256(),m4=_mm256_set1_epi8(15),m3=_mm256_set1_epi8(3),m32=_mm256_set1_epi8(32);
    for(int b=0;b<n/256;++b) {
        const float d=half_to_float(weights[b].d);
        auto all_scales=_mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i *>(weights[b].scales)));
        __m256i scale_halves[2]; prepare_scales_16(all_scales,scale_halves);
        __m256i block_sums[M];
        if constexpr(!I16) {
            #pragma GCC unroll 4
            for(int m=0;m<M;++m) block_sums[m]=zero;
        }
        #pragma GCC unroll 2
        for(int h=0;h<2;++h) {
            auto high_bits=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(weights[b].qh+32*h));
            __m256i half_sums[M];
            if constexpr(I16) {
                #pragma GCC unroll 4
                for(int m=0;m<M;++m) half_sums[m]=zero;
            }
            #pragma GCC unroll 4
            for(int k=0;k<4;++k) {
                const int g=4*h+k;
                auto low=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(weights[b].ql+64*h+32*(k&1)));
                if(k>=2) low=_mm256_srli_epi16(low,4);
                auto upper=_mm256_and_si256(_mm256_srli_epi16(high_bits,2*k),m3);
                auto code=_mm256_or_si256(_mm256_and_si256(low,m4),_mm256_slli_epi16(upper,4));
                auto weight_scale=_mm256_shuffle_epi8(scale_halves[h],get_scale_shuffle_16(k));
                if constexpr(I16) {
                    auto v=_mm256_sub_epi8(code,m32);
                    auto sign=_mm256_cmpgt_epi8(zero,v);
                    __m256i wl,wh;
                    if constexpr(Swizzled) {
                        wl=_mm256_mullo_epi16(_mm256_unpacklo_epi8(v,sign),weight_scale);
                        wh=_mm256_mullo_epi16(_mm256_unpackhi_epi8(v,sign),weight_scale);
                    } else {
                        auto sl=MM256_SET1_M128I(_mm256_castsi256_si128(weight_scale));
                        auto sh=MM256_SET1_M128I(_mm256_extracti128_si256(weight_scale,1));
                        wl=_mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(v)),sl);
                        wh=_mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(v,1)),sh);
                    }
                    //16*32*128*16383=1,073,676,288<i32max per128 inputs.
                    #pragma GCC unroll 4
                    for(int m=0;m<M;++m) {
                        const auto *a=i16[m]+256*b+32*g;
                        auto p=_mm256_add_epi32(_mm256_madd_epi16(wl,_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a))),
                                               _mm256_madd_epi16(wh,_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a+16))));
                        half_sums[m]=_mm256_add_epi32(half_sums[m],p);
                    }
                } else {
                    #pragma GCC unroll 4
                    for(int m=0;m<M;++m) {
                        auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8[m][b].qs+32*g));
                        auto pairs=_mm256_sub_epi16(_mm256_maddubs_epi16(code,a),_mm256_maddubs_epi16(m32,a));
                        block_sums[m]=_mm256_add_epi32(block_sums[m],_mm256_madd_epi16(pairs,weight_scale));
                    }
                }
            }
            if constexpr(I16) {
                #pragma GCC unroll 4
                for(int m=0;m<M;++m) acc[m]=_mm256_fmadd_ps(_mm256_set1_ps(d*scales[m][2*b+h]),_mm256_cvtepi32_ps(half_sums[m]),acc[m]);
            }
        }
        if constexpr(!I16) {
            #pragma GCC unroll 4
            for(int m=0;m<M;++m) acc[m]=_mm256_fmadd_ps(_mm256_set1_ps(d*q8[m][b].d),_mm256_cvtepi32_ps(block_sums[m]),acc[m]);
        }
    }
    #pragma GCC unroll 4
    for(int m=0;m<M;++m) outputs[m]=horizontal(acc[m]);
}
} // anonymous namespace
#define DEFINE_Q6(name,i16,m) \
__attribute__((noinline)) void name(const void *row,const Activation *inputs,int n,float *outputs) { q6_impl<i16,m>(row,inputs,n,outputs); }
DEFINE_Q6(dot_q6_k_q8_m1,false,1)
DEFINE_Q6(dot_q6_k_q8_m2,false,2)
DEFINE_Q6(dot_q6_k_q8_m4,false,4)
DEFINE_Q6(dot_q6_k_i16_m1,true,1)
DEFINE_Q6(dot_q6_k_i16_m2,true,2)
DEFINE_Q6(dot_q6_k_i16_m4,true,4)
#undef DEFINE_Q6
__attribute__((noinline)) void dot_q6_k_i16_baseline_m1(const void *row,const Activation *inputs,int n,float *outputs) { q6_impl<true,1,false>(row,inputs,n,outputs); }
namespace {
void dot_q6_dispatch(const void *row,const Activation *inputs,int n,int m,float *outputs,Kernel kernel) {
    const bool i16=inputs[0].path==Path::I16;
    if(i16 && kernel==Kernel::Baseline) { dot_q6_k_i16_baseline_m1(row,inputs,n,outputs); return; }
    if(i16) {
        if(m==1) dot_q6_k_i16_m1(row,inputs,n,outputs);
        else if(m==2) dot_q6_k_i16_m2(row,inputs,n,outputs);
        else dot_q6_k_i16_m4(row,inputs,n,outputs);
    } else {
        if(m==1) dot_q6_k_q8_m1(row,inputs,n,outputs);
        else if(m==2) dot_q6_k_q8_m2(row,inputs,n,outputs);
        else dot_q6_k_q8_m4(row,inputs,n,outputs);
    }
}
} // anonymous namespace
// Q3_K/Q4_K native AVX2 unpack/scales and dot recipes from ggml-quants.c.
// Q4_K keeps the affine minimum as a separate integer dot, preserving its
// two original fp16 block coefficients. M2/M4 share every decoded weight.
// fp-contract=off on these wrappers prevents reassociating the independently
// rounded affine products into two FMAs, which can leave a nonzero residual
// when a valid packed block represents only zero weights.
namespace {
__attribute__((always_inline)) inline __m256i q3_k_scales(const uint8_t *packed) {
    uint32_t aux[4]; std::memcpy(aux,packed,12); const uint32_t high=aux[2];
    constexpr uint32_t m2=0x03030303,m4=0x0f0f0f0f;
    aux[2]=((aux[0]>>4)&m4)|(((high>>4)&m2)<<4);
    aux[3]=((aux[1]>>4)&m4)|(((high>>6)&m2)<<4);
    aux[0]=(aux[0]&m4)|(((high>>0)&m2)<<4);
    aux[1]=(aux[1]&m4)|(((high>>2)&m2)<<4);
    return _mm256_cvtepi8_epi16(_mm_add_epi8(_mm_loadu_si128(reinterpret_cast<const __m128i *>(aux)),_mm_set1_epi8(-32)));
}
template<bool Q3,bool I16,int M,bool Swizzled=true>
__attribute__((always_inline)) inline void q34_impl(const void *row,const Activation *inputs,int n,float *outputs) {
    const auto *q3=static_cast<const block_q3_K *>(row); const auto *q4=static_cast<const block_q4_K *>(row);
    const block_q8_K *q8[M]; const int16_t *i16[M]; const float *act_scales[M]; __m256 acc[M];
    #pragma GCC unroll 4
    for(int m=0;m<M;++m) {
        q8[m]=reinterpret_cast<const block_q8_K *>(inputs[m].q8.data());
        i16[m]=Swizzled ? inputs[m].i16_swizzled.data() : inputs[m].i16.data();
        act_scales[m]=inputs[m].scales.data(); acc[m]=_mm256_setzero_ps();
    }
    const auto zero=_mm256_setzero_si256();
    for(int b=0;b<n/256;++b) {
        const float d=half_to_float(Q3 ? q3[b].d : q4[b].d);
        float dmin=0;
        if constexpr(!Q3) dmin=half_to_float(q4[b].dmin);
        __m256i scale_halves[2];
        if constexpr(Q3) prepare_scales_16(q3_k_scales(q3[b].scales),scale_halves);
        __m256i block_sums[M],block_mins[M];
        if constexpr(!I16) {
            #pragma GCC unroll 4
            for(int m=0;m<M;++m) block_sums[m]=block_mins[m]=zero;
        }
        #pragma GCC unroll 2
        for(int h=0;h<2;++h) {
            __m256i half_sums[M],half_mins[M];
            if constexpr(I16) {
                #pragma GCC unroll 4
                for(int m=0;m<M;++m) half_sums[m]=half_mins[m]=zero;
            }
            #pragma GCC unroll 4
            for(int k=0;k<4;++k) {
                const int g=4*h+k;
                __m256i code,negative_code,weight_scale,min_scale16,min_scale8;
                if constexpr(Q3) {
                    auto low=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(q3[b].qs+32*h));
                    code=_mm256_and_si256(_mm256_srli_epi16(low,2*k),_mm256_set1_epi8(3));
                    auto high=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(q3[b].hmask));
                    negative_code=_mm256_and_si256(_mm256_cmpeq_epi8(_mm256_and_si256(high,_mm256_set1_epi8(int8_t(1u<<g))),zero),_mm256_set1_epi8(4));
                    weight_scale=_mm256_shuffle_epi8(scale_halves[h],get_scale_shuffle_16(k));
                } else {
                    auto packed=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(q4[b].qs+32*(g/2)));
                    code=_mm256_and_si256(_mm256_srli_epi16(packed,4*(g&1)),_mm256_set1_epi8(15));
                    uint8_t scale,min; get_scale_min_k4(g,q4[b].scales,&scale,&min);
                    weight_scale=_mm256_set1_epi16(scale);
                    min_scale16=_mm256_set1_epi16(min); min_scale8=_mm256_set1_epi8(min);
                }
                if constexpr(I16) {
                    auto v=code;
                    if constexpr(Q3) v=_mm256_sub_epi8(v,negative_code);
                    auto sign=_mm256_cmpgt_epi8(zero,v);
                    __m256i wl,wh;
                    if constexpr(Swizzled) {
                        wl=_mm256_mullo_epi16(_mm256_unpacklo_epi8(v,sign),weight_scale);
                        wh=_mm256_mullo_epi16(_mm256_unpackhi_epi8(v,sign),weight_scale);
                    } else {
                        auto sl=MM256_SET1_M128I(_mm256_castsi256_si128(weight_scale));
                        auto sh=MM256_SET1_M128I(_mm256_extracti128_si256(weight_scale,1));
                        wl=_mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_castsi256_si128(v)),sl);
                        wh=_mm256_mullo_epi16(_mm256_cvtepi8_epi16(_mm256_extracti128_si256(v,1)),sh);
                    }
                    // Q3 maxintegerweight128; Q4 max945 andmin63. All
                    // i32 partials flush per128 inputs, <=16 terms/lane.
                    #pragma GCC unroll 4
                    for(int m=0;m<M;++m) {
                        const auto *a=i16[m]+256*b+32*g;
                        auto al=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a));
                        auto ah=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a+16));
                        auto p=_mm256_add_epi32(_mm256_madd_epi16(wl,al),_mm256_madd_epi16(wh,ah));
                        half_sums[m]=_mm256_add_epi32(half_sums[m],p);
                        if constexpr(!Q3) half_mins[m]=_mm256_add_epi32(half_mins[m],_mm256_add_epi32(_mm256_madd_epi16(min_scale16,al),_mm256_madd_epi16(min_scale16,ah)));
                    }
                } else {
                    #pragma GCC unroll 4
                    for(int m=0;m<M;++m) {
                        auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8[m][b].qs+32*g));
                        auto pairs=_mm256_maddubs_epi16(code,a);
                        if constexpr(Q3) pairs=_mm256_sub_epi16(pairs,_mm256_maddubs_epi16(negative_code,a));
                        block_sums[m]=_mm256_add_epi32(block_sums[m],_mm256_madd_epi16(pairs,weight_scale));
                        if constexpr(!Q3) block_mins[m]=_mm256_add_epi32(block_mins[m],_mm256_madd_epi16(_mm256_maddubs_epi16(min_scale8,a),_mm256_set1_epi16(1)));
                    }
                }
            }
            if constexpr(I16) {
                #pragma GCC unroll 4
                for(int m=0;m<M;++m) {
                    if constexpr(Q3) acc[m]=_mm256_fmadd_ps(_mm256_set1_ps(d*act_scales[m][2*b+h]),_mm256_cvtepi32_ps(half_sums[m]),acc[m]);
                    else {
                        auto product=_mm256_mul_ps(_mm256_set1_ps(d*act_scales[m][2*b+h]),_mm256_cvtepi32_ps(half_sums[m]));
                        auto minimum=_mm256_mul_ps(_mm256_set1_ps(dmin*act_scales[m][2*b+h]),_mm256_cvtepi32_ps(half_mins[m]));
                        acc[m]=_mm256_add_ps(acc[m],_mm256_sub_ps(product,minimum));
                    }
                }
            }
        }
        if constexpr(!I16) {
            #pragma GCC unroll 4
            for(int m=0;m<M;++m) {
                if constexpr(Q3) acc[m]=_mm256_fmadd_ps(_mm256_set1_ps(d*q8[m][b].d),_mm256_cvtepi32_ps(block_sums[m]),acc[m]);
                else {
                    auto product=_mm256_mul_ps(_mm256_set1_ps(d*q8[m][b].d),_mm256_cvtepi32_ps(block_sums[m]));
                    auto minimum=_mm256_mul_ps(_mm256_set1_ps(dmin*q8[m][b].d),_mm256_cvtepi32_ps(block_mins[m]));
                    acc[m]=_mm256_add_ps(acc[m],_mm256_sub_ps(product,minimum));
                }
            }
        }
    }
    #pragma GCC unroll 4
    for(int m=0;m<M;++m) outputs[m]=horizontal(acc[m]);
}
} // anonymous namespace
#define DEFINE_Q34(name,q3,i16,m) \
__attribute__((noinline,optimize("fp-contract=off"))) void name(const void *row,const Activation *inputs,int n,float *outputs) { q34_impl<q3,i16,m>(row,inputs,n,outputs); }
DEFINE_Q34(dot_q3_k_q8_m1,true,false,1)
DEFINE_Q34(dot_q3_k_q8_m2,true,false,2)
DEFINE_Q34(dot_q3_k_q8_m4,true,false,4)
DEFINE_Q34(dot_q3_k_i16_m1,true,true,1)
DEFINE_Q34(dot_q3_k_i16_m2,true,true,2)
DEFINE_Q34(dot_q3_k_i16_m4,true,true,4)
DEFINE_Q34(dot_q4_k_q8_m1,false,false,1)
DEFINE_Q34(dot_q4_k_q8_m2,false,false,2)
DEFINE_Q34(dot_q4_k_q8_m4,false,false,4)
DEFINE_Q34(dot_q4_k_i16_m1,false,true,1)
DEFINE_Q34(dot_q4_k_i16_m2,false,true,2)
DEFINE_Q34(dot_q4_k_i16_m4,false,true,4)
#undef DEFINE_Q34
__attribute__((noinline,optimize("fp-contract=off"))) void dot_q3_k_i16_baseline_m1(const void *row,const Activation *inputs,int n,float *outputs) { q34_impl<true,true,1,false>(row,inputs,n,outputs); }
__attribute__((noinline,optimize("fp-contract=off"))) void dot_q4_k_i16_baseline_m1(const void *row,const Activation *inputs,int n,float *outputs) { q34_impl<false,true,1,false>(row,inputs,n,outputs); }
namespace {
void dot_q34_dispatch(Format format,const void *row,const Activation *inputs,int n,int m,float *outputs,Kernel kernel) {
    const bool i16=inputs[0].path==Path::I16;
    if(i16 && kernel==Kernel::Baseline) {
        if(format==Format::Q3_K) dot_q3_k_i16_baseline_m1(row,inputs,n,outputs);
        else dot_q4_k_i16_baseline_m1(row,inputs,n,outputs);
        return;
    }
#define RUN_Q34(prefix) do { if(m==1) prefix##_m1(row,inputs,n,outputs); else if(m==2) prefix##_m2(row,inputs,n,outputs); else prefix##_m4(row,inputs,n,outputs); } while(0)
    if(format==Format::Q3_K) { if(i16) RUN_Q34(dot_q3_k_i16); else RUN_Q34(dot_q3_k_q8); }
    else { if(i16) RUN_Q34(dot_q4_k_i16); else RUN_Q34(dot_q4_k_q8); }
#undef RUN_Q34
}
} // anonymous namespace
// Q8_0 has one fp16 coefficient per32 signed bytes. Native Q8_K
// activations exclude-128, so abs(weight) as unsigned8 can contain128
// safely: maddubs pairs are bounded by2*128*127=32512<INT16_MAX.
namespace {
template<bool I16,int M,bool Swizzled=true>
__attribute__((always_inline)) inline void q8_0_impl(const void *row,const Activation *inputs,int n,float *outputs) {
    const auto *weights=static_cast<const block_q8_0 *>(row);
    const block_q8_K *q8[M]; const int16_t *i16[M]; const float *act_scales[M]; __m256 acc[M];
    #pragma GCC unroll 4
    for(int m=0;m<M;++m) {
        q8[m]=reinterpret_cast<const block_q8_K *>(inputs[m].q8.data());
        i16[m]=Swizzled ? inputs[m].i16_swizzled.data() : inputs[m].i16.data();
        act_scales[m]=inputs[m].scales.data(); acc[m]=_mm256_setzero_ps();
    }
    for(int b=0;b<n/256;++b) {
        #pragma GCC unroll 8
        for(int g=0;g<8;++g) {
            const auto &block=weights[8*b+g]; const float d=half_to_float(block.d);
            const auto v=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(block.qs));
            if constexpr(I16) {
                __m256i wl,wh;
                if constexpr(Swizzled) {
                    const auto sign=_mm256_cmpgt_epi8(_mm256_setzero_si256(),v);
                    wl=_mm256_unpacklo_epi8(v,sign); wh=_mm256_unpackhi_epi8(v,sign);
                } else {
                    wl=_mm256_cvtepi8_epi16(_mm256_castsi256_si128(v));
                    wh=_mm256_cvtepi8_epi16(_mm256_extracti128_si256(v,1));
                }
                #pragma GCC unroll 4
                for(int m=0;m<M;++m) {
                    const auto *a=i16[m]+256*b+32*g;
                    const auto p=_mm256_add_epi32(_mm256_madd_epi16(wl,_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a))),
                                                 _mm256_madd_epi16(wh,_mm256_loadu_si256(reinterpret_cast<const __m256i *>(a+16))));
                    acc[m]=_mm256_fmadd_ps(_mm256_set1_ps(d*act_scales[m][2*b+g/4]),_mm256_cvtepi32_ps(p),acc[m]);
                }
            } else {
                const auto magnitude=_mm256_sign_epi8(v,v);
                #pragma GCC unroll 4
                for(int m=0;m<M;++m) {
                    const auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i *>(q8[m][b].qs+32*g));
                    const auto p=_mm256_madd_epi16(_mm256_maddubs_epi16(magnitude,_mm256_sign_epi8(a,v)),_mm256_set1_epi16(1));
                    acc[m]=_mm256_fmadd_ps(_mm256_set1_ps(d*q8[m][b].d),_mm256_cvtepi32_ps(p),acc[m]);
                }
            }
        }
    }
    #pragma GCC unroll 4
    for(int m=0;m<M;++m) outputs[m]=horizontal(acc[m]);
}
} // anonymous namespace
#define DEFINE_Q80(name,i16,m) \
__attribute__((noinline)) void name(const void *row,const Activation *inputs,int n,float *outputs) { q8_0_impl<i16,m>(row,inputs,n,outputs); }
DEFINE_Q80(dot_q8_0_q8_m1,false,1)
DEFINE_Q80(dot_q8_0_q8_m2,false,2)
DEFINE_Q80(dot_q8_0_q8_m4,false,4)
DEFINE_Q80(dot_q8_0_i16_m1,true,1)
DEFINE_Q80(dot_q8_0_i16_m2,true,2)
DEFINE_Q80(dot_q8_0_i16_m4,true,4)
#undef DEFINE_Q80
__attribute__((noinline)) void dot_q8_0_i16_baseline_m1(const void *row,const Activation *inputs,int n,float *outputs) { q8_0_impl<true,1,false>(row,inputs,n,outputs); }
namespace {
void dot_q8_0_dispatch(const void *row,const Activation *inputs,int n,int m,float *outputs,Kernel kernel) {
    const bool i16=inputs[0].path==Path::I16;
    if(i16 && kernel==Kernel::Baseline) { dot_q8_0_i16_baseline_m1(row,inputs,n,outputs); return; }
    if(i16) {
        if(m==1) dot_q8_0_i16_m1(row,inputs,n,outputs);
        else if(m==2) dot_q8_0_i16_m2(row,inputs,n,outputs);
        else dot_q8_0_i16_m4(row,inputs,n,outputs);
    } else {
        if(m==1) dot_q8_0_q8_m1(row,inputs,n,outputs);
        else if(m==2) dot_q8_0_q8_m2(row,inputs,n,outputs);
        else dot_q8_0_q8_m4(row,inputs,n,outputs);
    }
}
} // anonymous namespace
} // namespace lutbench
