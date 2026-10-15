// Bitwise A/B of moe_forward on a synthetic EXL3 layer (random tiles — any 96 bytes decode to valid trellis weights):
// stock schedule vs GLM53_CT_SPLIT1=1 (or any env-switched kernel variant). Dumps raw output floats per case.
//   g++ -O3 -std=c++17 -mavx2 -mfma -mf16c -ffp-contract=off -pthread -DHAVE_SPLIT1 -I<ft_core dir> kernel-ab-harness.cpp -o h
//   GLM53_CT_SPLIT1=0 ./h a.bin ; GLM53_CT_SPLIT1=1 ./h b.bin ; cmp a.bin b.bin
// -ffp-contract=off matters: with GCC default contraction, ANY edit to ft_core.h can move I16 last bits (FPCONTRACT-01).
// To prove a change bit-exact FOR THE SERVER, also build with the server's flags (no -ffp-contract=off, see GLM53_CT_CFLAGS).
// Env: HARNESS_THREADS (8) on cpus HARNESS_CPUS ("0-7"; the server: "2-43"), HARNESS_SWZ=1 (server layout), HARNESS_E (8).
// Decode-shaped cases (1 token x K experts, K in HARNESS_KS, default "1,2,3,4,6") follow the multi-token cases.
// [E6] -DHAVE_E6: GLM53_CT_FUSE=1 (+ GLM53_CT_FUSE_MAXN) fused 2-run schedule, GLM53_CT_PF_LEAN=1 lean GEMV prefetch.
// HARNESS_TIME=1: timing instead of dumping — decode calls (1 token x K experts, cycling a DRAM-cold pool of HARNESS_E
// experts), HARNESS_ITERS per K, mode I16; with HAVE_E6 the variants in HARNESS_VARIANTS ("base,lean,lean+fuse") alternate
// call by call (drift cancels) and each is reported; a "pfN" token sets PF_DIST (default 6), e.g. "lean+fuse+pf4".
// HARNESS_CPUS takes ranges, e.g. "2-17,22-37" (16 + 16 threads balanced over both sockets).
// [bdw-opt] -DHAVE_SERIAL_AB: HARNESS_VARIANTS token "serial" toggles GLM53_CT_SERIAL_SUM per call.
// -DHAVE_LIGHT_AB: token "static" toggles GLM53_CT_LIGHT_STATIC for short phases only.
// HARNESS_PRIME_MS (0): untimed initial decode work to stabilize frequency/pool startup before K=1.
// HARNESS_ROTATE=1 rotates variant order on each common route; HARNESS_ACT_LIMIT (0) selects the live clamp.
// // Timing variants can include a registry kernel name; HARNESS_KERNEL_VERIFY=1 checks every I16 block against base.
#include "ft_core.h"
#include <array>

int main(int argc, char** argv)
{
    init_perm(); init_tables();
    if (const char* v = std::getenv("HARNESS_ACT_LIMIT")) ACT_LIMIT = float(std::atof(v));
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> byte(0, 255);
    std::normal_distribution<float> nd(0.f, 1.f);
    auto mk = [&](int k, int n) {
        Mat m; m.k = k; m.n = n; m.swz = 0;
        const size_t nb = size_t(k / 16) * (n / 16) * TILE_BYTES;
        uint8_t* tr = static_cast<uint8_t*>(big_alloc(nb)); for (size_t i = 0; i < nb; ++i) tr[i] = uint8_t(byte(rng));
        uint16_t* suh = new uint16_t[k]; uint16_t* svh = new uint16_t[n];
        for (int i = 0; i < k; ++i) suh[i] = f2h((byte(rng) & 1 ? 1.f : -1.f) * (0.5f + 0.01f * (byte(rng) % 50)));
        for (int i = 0; i < n; ++i) svh[i] = f2h((byte(rng) & 1 ? 1.f : -1.f) * (0.5f + 0.01f * (byte(rng) % 50)));
        m.tr = tr; m.suh = suh; m.svh = svh; return m;
    };
    Layer L; L.H = 4096; L.I = 2048;
    auto envi = [](const char* k, int d) { const char* v = std::getenv(k); return v ? std::atoi(v) : d; };
    const int E = envi("HARNESS_E", 8), threads = envi("HARNESS_THREADS", 8);
    std::vector<int> ks; { std::string v = std::getenv("HARNESS_KS") ? std::getenv("HARNESS_KS") : "1,2,3,4,6"; std::stringstream ss(v); std::string t; while (std::getline(ss, t, ',')) ks.push_back(std::stoi(t)); }
    for (int e = 0; e < E; ++e) { Expert X; X.g = mk(L.H, L.I); X.u = mk(L.H, L.I); X.d = mk(L.I, L.H); L.ex.push_back(X); }
    // HARNESS_SWZ=1: test the band-contiguous layout the server runs with (GLM53_CT_SWZ=1) via ft_core's own swizzle
    if (const char* v = std::getenv("HARNESS_SWZ"); v && std::atoi(v) == 1) swizzle_layer(L);
    std::vector<int> cpus;
    { std::string v = std::getenv("HARNESS_CPUS") ? std::getenv("HARNESS_CPUS") : "0-7"; std::stringstream ss(v); std::string r;
      while (std::getline(ss, r, ','))   // "2-43" or "2-17,22-37"
      { const size_t d = r.find('-'); const int a = std::stoi(r.substr(0, d)), b = d == std::string::npos ? a : std::stoi(r.substr(d + 1));
        for (int c = a; c <= b; ++c) cpus.push_back(c); } }
    Pool pool; pool.start(threads, cpus);
    if (envi("HARNESS_TIME", 0))
    {
        const int iters = envi("HARNESS_ITERS", 300);
        const bool rotate = envi("HARNESS_ROTATE", 0) != 0;
        std::mt19937 r2(4242);
        std::vector<float> x(L.H); for (auto& v : x) v = r16(0.35f * std::normal_distribution<float>(0.f, 1.f)(r2));
        std::vector<float> out(L.H);
        std::vector<std::string> vars;
        { std::string v = std::getenv("HARNESS_VARIANTS") ? std::getenv("HARNESS_VARIANTS") : "base"; std::stringstream ss(v); std::string t; while (std::getline(ss, t, ',')) vars.push_back(t); }
        const double prime_end = now() + envi("HARNESS_PRIME_MS", 0) * 0.001;
        for (int it = 0; now() < prime_end; ++it)
        {
            std::vector<std::vector<std::pair<int, float>>> route(1);
            const int K = std::min(4, E);
            for (int q = 0; q < K; ++q) route[0].push_back({ (it * K + q) % E, 1.0f / K });
            moe_forward(pool, L, x.data(), 1, route, out.data(), 2);
        }
        for (int K : ks)
        {
            const int nvar = int(vars.size());
            std::vector<std::vector<double>> tot(nvar); std::vector<std::array<double, 5>> ph(nvar, std::array<double, 5>{});
            for (int it = 0; it < nvar * iters; ++it)
            {
                const int var = (it % nvar + (rotate ? (it / nvar) % nvar : 0)) % nvar;
#ifdef HAVE_SERIAL_AB
                g_serial_sum().store(vars[var].find("serial") != std::string::npos, std::memory_order_relaxed);
#endif
#ifdef HAVE_LIGHT_AB
                g_light_static().store(vars[var].find("static") != std::string::npos, std::memory_order_relaxed);
#endif
#ifdef HAVE_E6
                g_fuse().store(vars[var].find("fuse") != std::string::npos);
                pf_lean().store(vars[var].find("lean") != std::string::npos);
                { const size_t q = vars[var].find("pf"); PF_DIST = q == std::string::npos ? 6 : std::atoi(vars[var].c_str() + q + 2); }   // e.g. lean+pf4
#endif
                std::vector<std::vector<std::pair<int, float>>> route(1);
                for (int q = 0; q < K; ++q) route[0].push_back({ ((it / nvar) * K + q) % E, 1.0f / K });
                PhaseTimes pt;
                const double t0 = now();
                moe_forward(pool, L, x.data(), 1, route, out.data(), 2, &pt);
                const double dt = now() - t0;
                if (it / nvar >= 20) { tot[var].push_back(dt * 1e6); for (int k = 0; k < 5; ++k) ph[var][k] += pt.t[k] * 1e6; }   // skip 20 complete common routes
            }
            for (int var = 0; var < nvar; ++var)
            {
                auto& v = tot[var]; std::sort(v.begin(), v.end());
                double mean = 0; for (double d : v) mean += d; mean /= v.size();
                std::printf("K=%d %-6s calls %zu  mean %7.1f us  p50 %7.1f  p10 %7.1f  p90 %7.1f | phases mean us: %6.1f %6.1f %6.1f %6.1f %6.1f\n",
                            K, vars[var].c_str(), v.size(), mean, v[v.size() / 2], v[v.size() / 10], v[v.size() * 9 / 10],
                            ph[var][0] / v.size(), ph[var][1] / v.size(), ph[var][2] / v.size(), ph[var][3] / v.size(), ph[var][4] / v.size());
            }
        }
        pool.stop();
        return 0;
    }
    FILE* fo = std::fopen(argv[1], "wb");
    const int ntoks[3] = { 1, 3, 4 };
    for (int mode : { 1, 0, 2 })
        for (int nt : ntoks)
            for (int rep = 0; rep < 3; ++rep)        // 3 reps: also checks run-to-run determinism within a schedule
            {
                std::mt19937 r2(777 + nt);            // same inputs for every mode/rep
                std::vector<float> x(size_t(nt) * L.H);
                for (auto& v : x) v = r16(0.35f * std::normal_distribution<float>(0.f, 1.f)(r2));
                std::vector<std::vector<std::pair<int, float>>> route(nt);
                for (int t = 0; t < nt; ++t) for (int k = 0; k < 4; ++k) route[t].push_back({ (t + 2 * k) % E, 0.25f });
                std::vector<float> out(size_t(nt) * L.H);
                moe_forward(pool, L, x.data(), nt, route, out.data(), mode);
                std::fwrite(out.data(), 4, out.size(), fo);
                double s = 0; for (float v : out) s += std::fabs(v);
                std::printf("mode %d ntok %d rep %d  sum|out| %.9g\n", mode, nt, rep, s);
            }
    // decode-shaped: 1 token x K experts (the CPU tier's real call shape), modes I16 / EXACT, distinct expert sets
    for (int mode : { 2, 1 })
        for (int K : ks)
            for (int rep = 0; rep < 2; ++rep)
            {
                std::mt19937 r2(991 + K);
                std::vector<float> x(L.H);
                for (auto& v : x) v = r16(0.35f * std::normal_distribution<float>(0.f, 1.f)(r2));
                std::vector<std::vector<std::pair<int, float>>> route(1);
                for (int q = 0; q < K; ++q) route[0].push_back({ (3 * q + K) % E, 0.9f / (q + 1) });
                std::vector<float> out(L.H);
                moe_forward(pool, L, x.data(), 1, route, out.data(), mode);
                std::fwrite(out.data(), 4, out.size(), fo);
                double s = 0; for (float v : out) s += std::fabs(v);
                std::printf("decode mode %d K %d rep %d  sum|out| %.9g\n", mode, K, rep, s);
            }
    std::fclose(fo);
    pool.stop();
#ifdef HAVE_SPLIT1
    std::printf("split1=%d\n", int(split1_enabled()));
#endif
}
