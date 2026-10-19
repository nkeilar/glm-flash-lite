// Independent production-fallback vs asm check; raw packed patterns supplement
// the audit's quantized-data differential gates. No reference calls the candidate.
#include "lut/kernels.h"
#include <dlfcn.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>
using Dot = float (*)(const void *, const lutbench::Activation &, int);
using Multi = void (*)(const void *, const lutbench::Activation *, int, float *);
struct Lib { Dot m1; Multi m2, m4; };
struct Weight { uint16_t d, extra, signs; uint8_t scales[8], qs[64], qh[32]; };
struct Q8 { float d, sum; int8_t qs[256]; int16_t bsums[16]; };
static_assert(sizeof(Weight) == 110 && sizeof(Q8) == 296);
static uint32_t bits(float x) { uint32_t u; std::memcpy(&u, &x, 4); return u; }
static Lib open(const char *path) {
    void *h = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!h) { std::fprintf(stderr, "%s\n", dlerror()); std::exit(2); }
    auto f = reinterpret_cast<Dot>(dlsym(h, "_ZN8lutbench10dot_iq3_q8EPKvRKNS_10ActivationEi"));
    if (!f) { std::fprintf(stderr, "%s\n", dlerror()); std::exit(2); }
    auto m2 = reinterpret_cast<Multi>(dlsym(h, "_ZN8lutbench23dot_iq3_q8_optimized_m2EPKvPKNS_10ActivationEiPf"));
    auto m4 = reinterpret_cast<Multi>(dlsym(h, "_ZN8lutbench23dot_iq3_q8_optimized_m4EPKvPKNS_10ActivationEiPf"));
    if (!m2 || !m4) { std::fprintf(stderr, "%s\n", dlerror()); std::exit(2); }
    return {f, m2, m4};
}
int main(int argc, char **argv) {
    if (argc < 3) return 2;
    auto ref = open(argv[1]);
    std::vector<Lib> variants;
    for (int i = 2; i < argc; ++i) variants.push_back(open(argv[i]));
    std::mt19937 rng(20261020);
    std::vector<Weight> w(16);
    std::vector<lutbench::Activation> acts(4);
    for (auto &a : acts) a.q8.resize(16 * sizeof(Q8));
    size_t checked = 0;
    for (unsigned trial = 0; trial < 100000; ++trial) {
        for (auto &v : w) {
            auto *bytes = reinterpret_cast<uint8_t *>(&v);
            for (size_t j = 0; j < sizeof(v); ++j) bytes[j] = uint8_t(rng());
            // Exhaust all fp16 encodings, including subnormals, signed zero,
            // infinity and NaNs; later blocks provide competing NaN payloads.
            v.d = uint16_t(trial < 65536 ? trial : rng());
        }
        for (auto &a : acts) for (int block = 0; block < 16; ++block) {
            Q8 q{};
            uint32_t u = rng();
            if (trial % 4) u = (u & 0x807fffffU) | ((100 + rng() % 50) << 23);
            std::memcpy(&q.d, &u, 4);
            for (auto &b : q.qs) b = trial % 7 == 0 ? int8_t(-128) : trial % 7 == 1 ? int8_t(127) : int8_t(rng());
            for (int g = 0; g < 16; ++g) {
                int s = 0; for (int j = 0; j < 16; ++j) s += q.qs[16*g+j];
                q.bsums[g] = int16_t(s);
            }
            std::memcpy(a.q8.data() + block * sizeof(Q8), &q, sizeof(q));
        }
        for (int n : {0, 256, 1024, 4096}) for (int m : {1, 2, 4}) {
            float expected[4];
            auto run = [&](const Lib &lib, float *out) {
                if (m == 1) out[0] = lib.m1(w.data(), acts[0], n);
                else (m == 2 ? lib.m2 : lib.m4)(w.data(), acts.data(), n, out);
            };
            run(ref, expected);
            for (size_t i = 0; i < variants.size(); ++i) {
                float actual[4]; run(variants[i], actual);
                for (int t = 0; t < m; ++t) {
                    if (bits(actual[t]) != bits(expected[t])) {
                        std::fprintf(stderr, "MISMATCH trial=%u n=%d m=%d token=%d lib=%s ref=%08x actual=%08x\n", trial, n, m, t, argv[i+2], bits(expected[t]), bits(actual[t]));
                        return 1;
                    }
                    ++checked;
                }
            }
        }
    }
    std::printf("BIT-EXACT raw-pattern outputs=%zu (all fp16 encodings, random float scales, byte extrema, M=1/2/4, 0/1/4/16 blocks)\n", checked);
}
