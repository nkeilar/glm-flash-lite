// Standalone correctness and failure tests; workstation timing is not NUMA evidence.
#include "../ft_lut.h"
#include "../lut/kernels.h"
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <sched.h>
#include <set>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <unistd.h>

namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F fn) { bool rejected = false; try { fn(); } catch (const std::exception&) { rejected = true; } check(rejected, "invalid operation accepted"); }
float fp16(float x) { return _cvtsh_ss(_cvtss_sh(x, _MM_FROUND_TO_NEAREST_INT)); }
lutbench::Format fmt(int type) {
    return type == 138 ? lutbench::Format::IQ3_K : type == 144 ? lutbench::Format::IQ4_KS : lutbench::Format::Q6_K;
}
std::vector<int> cpus() {
    cpu_set_t allowed; CPU_ZERO(&allowed); check(sched_getaffinity(0, sizeof(allowed), &allowed) == 0, "affinity unavailable");
    std::set<std::pair<int, int>> cores; std::vector<int> result;
    for (int c = 0; c < CPU_SETSIZE && result.size() < 4; ++c) if (CPU_ISSET(c, &allowed)) {
        int pkg = -1, core = -1;
        std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/";
        std::ifstream(base + "physical_package_id") >> pkg; std::ifstream(base + "core_id") >> core;
        if (pkg >= 0 && core >= 0 && cores.insert({pkg, core}).second) result.push_back(c);
    }
    check(result.size() >= 4, "tests need four distinct allowed physical cores"); return result;
}
struct Temporary {
    std::string path;
    std::vector<std::string> files;
    Temporary() { char name[] = "/tmp/lut-backend-test-XXXXXX"; char* p = mkdtemp(name); check(p, "mkdtemp failed"); path = p; }
    ~Temporary() { for (const auto& f : files) unlink(f.c_str()); rmdir(path.c_str()); }
    std::string name(const std::string& value) { std::string result = path + "/" + value; files.push_back(result); return result; }
};
struct Fixture {
    ft_lut::LayerInput input;
    std::array<std::vector<uint8_t>, 3> packed;
    std::array<size_t, 3> stride{};
    int h = 256, i = 512, experts;
    Fixture(Temporary& temp, int layer, int count) : experts(count) {
        input.model_layer = layer; input.internal_li = layer - 3;
        input.path = temp.name("weights-" + std::to_string(layer));
        std::array<int, 3> types{138, 138, 144};
        if (layer == 11) types = {144, 144, 14};
        else if (layer == 12 || layer == 44) types[2] = 14;
        uint64_t offset = 64;
        for (int m = 0; m < 3; ++m) {
            int width = m == 2 ? i : h, rows = m == 2 ? h : i;
            stride[m] = lutbench::row_bytes(fmt(types[m]), width);
            packed[m].resize(stride[m] * rows * experts);
            input.tensors[m] = {offset, packed[m].size(), types[m], width, rows, experts};
            std::vector<float> row(width);
            for (int e = 0; e < experts; ++e) for (int r = 0; r < rows; ++r) {
                float amplitude = (m < 2 && r % 127 == 0) ? 1.3f : 0.06f;
                for (int c = 0; c < width; ++c) row[c] = amplitude * (std::sin((e + 1) * (r + 3) * (c + 1) * .003f + m) + .4f * std::cos(c * .07f + e));
                lutbench::quantize(fmt(types[m]), row.data(), packed[m].data() + (size_t(e) * rows + r) * stride[m], width);
            }
            offset = (offset + packed[m].size() + 31) / 32 * 32;
        }
        std::ofstream file(input.path, std::ios::binary);
        for (int m = 0; m < 3; ++m) { file.seekp(std::streamoff(input.tensors[m].offset)); file.write(reinterpret_cast<const char*>(packed[m].data()), std::streamsize(packed[m].size())); }
        file.close(); input.identity = ft_lut::file_identity(input.path);
    }
    const uint8_t* row(int m, int expert, int r) const {
        return packed[m].data() + (size_t(expert) * (m == 2 ? h : i) + r) * stride[m];
    }
};
void dot_m(lutbench::Format f, const void* row, const lutbench::Activation* a, int n, int m, float* y) {
    if (m == 3) { lutbench::dot_multi(f, row, a, n, 2, y); lutbench::dot_multi(f, row, a + 2, n, 1, y + 2); }
    else lutbench::dot_multi(f, row, a, n, m, y);
}
std::vector<float> reference(const Fixture& fixture, const std::vector<float>& x, const ft_lut::Routes& routes, lutbench::Path path) {
    int m = int(routes.size()); std::vector<float> output(size_t(m) * fixture.h, 0);
    for (int expert = 0; expert < fixture.experts; ++expert) {
        int count = 0, token[4]; float weight[4];
        for (int t = 0; t < m; ++t) for (auto pick : routes[t]) if (pick.first == expert) { token[count] = t; weight[count++] = pick.second; }
        if (!count) continue;
        std::array<lutbench::Activation, 4> input, hidden_q;
        std::vector<float> hidden(size_t(count) * fixture.i);
        for (int t = 0; t < count; ++t) lutbench::prepare(path, x.data() + size_t(token[t]) * fixture.h, fixture.h, input[t]);
        for (int r = 0; r < fixture.i; ++r) {
            float g[4], u[4];
            dot_m(fmt(fixture.input.tensors[0].type_id), fixture.row(0, expert, r), input.data(), fixture.h, count, g);
            dot_m(fmt(fixture.input.tensors[1].type_id), fixture.row(1, expert, r), input.data(), fixture.h, count, u);
            for (int t = 0; t < count; ++t) hidden[size_t(t) * fixture.i + r] = std::min(g[t] / (1.f + std::exp(-g[t])), 10.f) * std::max(-10.f, std::min(10.f, u[t]));
        }
        for (int t = 0; t < count; ++t) lutbench::prepare(path, hidden.data() + size_t(t) * fixture.i, fixture.i, hidden_q[t]);
        for (int r = 0; r < fixture.h; ++r) {
            float y[4]; dot_m(fmt(fixture.input.tensors[2].type_id), fixture.row(2, expert, r), hidden_q.data(), fixture.i, count, y);
            for (int t = 0; t < count; ++t) output[size_t(token[t]) * fixture.h + r] += weight[t] * y[t];
        }
    }
    return output;
}
double relative(const std::vector<float>& got, const std::vector<float>& expected) {
    double error = 0, energy = 0;
    for (size_t j = 0; j < got.size(); ++j) { check(std::isfinite(got[j]), "nonfinite result"); double diff = double(got[j]) - expected[j]; error += diff * diff; energy += double(expected[j]) * expected[j]; }
    return energy ? std::sqrt(error / energy) : std::sqrt(error);
}
void layout(ft_lut::Backend& backend, const Fixture& f) {
    for (int group = 0; group < 2; ++group) for (int expert : {0, f.experts - 1}) for (int m = 0; m < 3; ++m) {
        auto packed = backend.fixture_slice(f.input.internal_li, group, expert, m);
        int width = m == 2 ? f.i / 2 : f.h, rows = m == 2 ? f.h : f.i / 2;
        size_t stride = lutbench::row_bytes(fmt(f.input.tensors[m].type_id), width);
        check(packed.size() == stride * rows, "split arena size differs from layout ledger");
        std::vector<float> original(m == 2 ? f.i : f.h), half(width);
        for (int r : {0, rows / 2, rows - 1}) {
            const uint8_t* src = f.row(m, expert, m == 2 ? r : group * rows + r);
            const uint8_t* dst = packed.data() + size_t(r) * stride;
            lutbench::dequantize(fmt(f.input.tensors[m].type_id), src, original.data(), m == 2 ? f.i : f.h);
            lutbench::dequantize(fmt(f.input.tensors[m].type_id), dst, half.data(), width);
            check(std::equal(half.begin(), half.end(), original.begin() + (m == 2 ? group * width : 0)), "split changed represented weight values");
            if (m == 2 && f.input.tensors[m].type_id == 144) check(std::memcmp(src, dst, 4) == 0, "IQ4_KS row scale was not duplicated exactly");
        }
    }
}
void numeric(ft_lut::Backend& backend, const Fixture& f, lutbench::Path path, int& cases, double& worst) {
    for (int m = 1; m <= 4; ++m) for (int k = 0; k <= f.experts; ++k) {
        std::vector<float> x(size_t(m) * f.h);
        for (size_t n = 0; n < x.size(); ++n) x[n] = fp16(.7f * std::sin(float(n + 1) * .071f) + .17f * std::cos(float(n) * .021f));
        ft_lut::Routes routes(m);
        for (int t = 0; t < m; ++t) for (int j = 0; j < k; ++j) {
            // Different ordering, weights and per-token subsets, including a
            // three-occurrence expert and nonadjacent token pairs.
            if (j && (j + 2 * t) % 5 == 0) continue;
            int expert = (j + t) % f.experts;
            routes[t].push_back({expert, .07f + .013f * float(t + j + 1)});
        }
        auto expected = reference(f, x, routes, path);
        std::vector<float> got(expected.size(), 123.f);
        backend.forward(f.input.internal_li, x.data(), m, routes, got.data());
        double error = relative(got, expected); worst = std::max(worst, error);
        check(error <= 1e-5, "TP2 output differs from canonical packed-weight reference"); ++cases;
    }
}
} // namespace

int main() {
    try {
        // Full-bank planning is structural only; these descriptors deliberately
        // name no input files and cannot be treated as verified payload evidence.
        std::vector<ft_lut::LayerInput> bank;
        for (int layer = 3; layer <= 44; ++layer) {
            ft_lut::LayerInput in; in.model_layer = layer; in.internal_li = layer - 3;
            std::array<int, 3> types{138, 138, 144};
            if (layer == 11) types = {144, 144, 14}; else if (layer == 12 || layer == 44) types[2] = 14;
            uint64_t offset = 4096;
            for (int m = 0; m < 3; ++m) {
                int width = m == 2 ? 2048 : 4096, rows = m == 2 ? 4096 : 2048;
                uint64_t bytes = lutbench::row_bytes(fmt(types[m]), width) * uint64_t(rows) * 288;
                // Every production expert projection is page aligned, including
                // IQ4_KS gate/up's four-byte global row scale. The streaming
                // admission ledger subtracts these exact touched pages.
                int split_width = m == 2 ? width / 2 : width;
                int split_rows = m == 2 ? rows : rows / 2;
                uint64_t expert_bytes = lutbench::row_bytes(fmt(types[m]), split_width) * uint64_t(split_rows);
                uint64_t page = uint64_t(sysconf(_SC_PAGESIZE));
                check(page && expert_bytes % page == 0, "production expert projection is not page aligned");
                uint64_t remaining = expert_bytes * 288;
                for (int e = 0; e < 288; ++e) {
                    check(remaining >= expert_bytes, "production touched-page ledger underflow");
                    remaining -= expert_bytes;
                }
                check(remaining == 0, "production touched-page ledger did not drain");
                in.tensors[m] = {offset, bytes, types[m], width, rows, 288}; offset += bytes;
            }
            in.identity.size = offset; bank.push_back(in);
        }
        auto plan = ft_lut::bank_plan(bank);
        check(plan.canonical_payload_bytes == 143879307264ULL && plan.tp2_payload_bytes == 144063332352ULL &&
              plan.mapped_bytes[0] == 72031666176ULL && plan.mapped_bytes[1] == 72031666176ULL, "full-bank layout byte ledger mismatch");
        auto broken = bank; broken[1] = broken[0]; rejects([&] { ft_lut::bank_plan(broken); });
        broken = bank; broken[0].tensors[0].experts = 1; rejects([&] { ft_lut::bank_plan(broken); });
        broken = bank; broken[8].tensors[0].type_id = 138; rejects([&] { ft_lut::bank_plan(broken); });
        broken = bank; broken[0].tensors[1].offset = broken[0].tensors[0].offset; rejects([&] { ft_lut::bank_plan(broken); });
        uint64_t host_minimum = (101ULL << 30) + plan.tp2_payload_bytes;
        std::array<uint64_t, 2> free{plan.mapped_bytes[0] + (8ULL << 30) + (512ULL << 20), plan.mapped_bytes[1] + (8ULL << 30) + (512ULL << 20)};
        ft_lut::check_startup_headroom(host_minimum, free, plan.mapped_bytes);
        rejects([&] { ft_lut::check_startup_headroom(host_minimum - 1, free, plan.mapped_bytes); });
        --free[1]; rejects([&] { ft_lut::check_startup_headroom(host_minimum, free, plan.mapped_bytes); });
        rejects([&] { ft_lut::check_startup_headroom(UINT64_MAX, {UINT64_MAX, UINT64_MAX}, {UINT64_MAX, 1}); });
        ft_lut::check_memory_limit(200, 100, 100);
        rejects([&] { ft_lut::check_memory_limit(200, 101, 100); });
        rejects([&] { ft_lut::check_memory_limit(200, 201, 0); });
        Temporary temp; auto workers = cpus();
        Fixture ordinary(temp, 3, 8), upgrade(temp, 11, 8), down6(temp, 12, 8);
        int cases = 0; double worst = 0;
        for (const std::string activation : {"i16", "q8"}) {
            ft_lut::Options options; options.cpus = workers; options.virtual_tp2 = true; options.activation = activation;
            options.initial_generation = std::numeric_limits<uint64_t>::max() - 1;
            auto backend = ft_lut::Backend::offline_fixture(options);
            for (const Fixture* fixture : {&ordinary, &upgrade, &down6}) {
                backend->add_layer(fixture->input); layout(*backend, *fixture);
                numeric(*backend, *fixture, activation == "i16" ? lutbench::Path::I16 : lutbench::Path::Q8, cases, worst);
            }
            auto receipt = backend->receipt();
            check(receipt.offline_fixture && receipt.virtual_tp2 && !receipt.placement_verified, "virtual geometry claimed NUMA placement");
            check(receipt.regions.size() == 18 && receipt.payload_bytes[0] == receipt.payload_bytes[1] && receipt.staging_peak_bytes <= 64 * 1024 * 1024,
                  "loader receipt geometry or bounded staging is incorrect");
            for (const auto& region : receipt.regions) check(region.sampled_pages == 32, "placement did not sample every expert projection");
            std::vector<float> x(4 * ordinary.h, .25f), out(x.size(), 123.f);
            ft_lut::Routes route(4); route[0] = {{0, .4f}, {1, .6f}}; route[2] = {{0, .25f}}; route[3] = {{1, .9f}};
            for (int phase = 0; phase < 4; ++phase) for (int group = 0; group < 2; ++group) {
                std::fill(out.begin(), out.end(), 123.f); backend->inject_failure(phase, group);
                rejects([&] { backend->forward(0, x.data(), 4, route, out.data()); });
                check(std::all_of(out.begin(), out.end(), [](float v) { return v == 123.f; }), "worker failure published a partial caller result");
                backend->inject_failure(-1); backend->forward(0, x.data(), 4, route, out.data());
                check(relative(out, reference(ordinary, x, route, activation == "i16" ? lutbench::Path::I16 : lutbench::Path::Q8)) <= 1e-5,
                      "pool failed to drain/reuse after a worker exception");
            }
            auto duplicate = route; duplicate[0].push_back({0, .5f});
            rejects([&] { backend->forward(0, x.data(), 4, duplicate, out.data()); });
            auto invalid = route; invalid[0][0].first = ordinary.experts;
            rejects([&] { backend->forward(0, x.data(), 4, invalid, out.data()); });
            invalid = route; invalid[0][0].second = std::numeric_limits<float>::quiet_NaN();
            rejects([&] { backend->forward(0, x.data(), 4, invalid, out.data()); });
            for (bool overflow_weight : {false, true}) {
                std::fill(out.begin(), out.end(), 123.f);
                std::fill(x.begin(), x.end(), overflow_weight ? 25.f : std::numeric_limits<float>::max());
                auto overflow_route = route;
                if (overflow_weight) for (auto& row : overflow_route) for (auto& pick : row) pick.second = std::numeric_limits<float>::max();
                rejects([&] { backend->forward(0, x.data(), 4, overflow_route, out.data()); });
                check(std::all_of(out.begin(), out.end(), [](float v) { return v == 123.f; }), "overflow published a partial caller result");
                std::fill(x.begin(), x.end(), .25f);
                backend->forward(0, x.data(), 4, route, out.data());
            }
            x[0] = std::numeric_limits<float>::infinity();
            rejects([&] { backend->forward(0, x.data(), 4, route, out.data()); });
            rejects([&] { backend->forward(0, x.data(), 5, route, out.data()); });
            rejects([&] { backend->add_layer(ordinary.input); });
        }
        ft_lut::Options options; options.cpus = workers; options.virtual_tp2 = true;
        auto backend = ft_lut::Backend::offline_fixture(options);
        auto bad = ordinary.input; bad.path = temp.name("missing"); rejects([&] { backend->add_layer(bad); });
        bad = ordinary.input; ++bad.identity.mtime_ns; rejects([&] { backend->add_layer(bad); });
        bad = ordinary.input; bad.tensors[0].offset = std::numeric_limits<uint64_t>::max(); rejects([&] { backend->add_layer(bad); });
        bad = ordinary.input; bad.tensors[0].type_id = 144; rejects([&] { backend->add_layer(bad); });
        bad = ordinary.input; bad.tensors[1].offset = bad.tensors[0].offset; rejects([&] { backend->add_layer(bad); });
        bad = ordinary.input; bad.path = temp.name("symlink"); check(symlink(ordinary.input.path.c_str(), bad.path.c_str()) == 0, "symlink fixture failed");
        rejects([&] { backend->add_layer(bad); });
        // A corrupted packed row scale must fail numerically without publishing
        // a partial output, even for the explicitly untrusted offline fixture API.
        auto corrupt = ordinary.input; corrupt.path = temp.name("corrupt-scale");
        { std::ifstream source(ordinary.input.path, std::ios::binary); std::ofstream target(corrupt.path, std::ios::binary); target << source.rdbuf(); }
        { std::fstream target(corrupt.path, std::ios::in | std::ios::out | std::ios::binary); uint32_t nan = 0x7fc00000; target.seekp(std::streamoff(corrupt.tensors[2].offset)); target.write(reinterpret_cast<const char*>(&nan), 4); }
        corrupt.identity = ft_lut::file_identity(corrupt.path);
        auto corrupt_backend = ft_lut::Backend::offline_fixture(options); corrupt_backend->add_layer(corrupt);
        std::vector<float> x(ordinary.h, .25f), output(ordinary.h, 123.f); ft_lut::Routes routes{{{0, 1.f}}};
        rejects([&] { corrupt_backend->forward(0, x.data(), 1, routes, output.data()); });
        check(std::all_of(output.begin(), output.end(), [](float v) { return v == 123.f; }), "corrupt scale published a partial result");
        options.require_verified_placement = true; auto strict = ft_lut::Backend::offline_fixture(options);
        rejects([&] { strict->add_layer(ordinary.input); });
        check(!strict->receipt().placement_verified && strict->receipt().regions.empty(), "rejected placement published a loaded layer");
        options.cpus = {workers[0], workers[0]}; rejects([&] { ft_lut::Backend::offline_fixture(options); });
        std::cout << "PASS cases=" << cases << " worst_relative_l2=" << worst
                  << " paths=i16,q8 M=1,2,3,4 K=0..8 profiles=ordinary,upgrade,down6"
                  << " worker_failures=16 placement_claim=none generation_wrap=covered\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
