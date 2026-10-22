// Standalone correctness and failure tests; workstation timing is not NUMA evidence.
#include "../ft_lut.h"
#include "../lut/kernels.h"
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cerrno>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <linux/mempolicy.h>
#include <limits>
#include <map>
#include <sched.h>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <sys/stat.h>
#include <sys/mman.h>
#include <unistd.h>

namespace {
void check(bool ok, const char* message) { if (!ok) throw std::runtime_error(message); }
template<class F> void rejects(F fn) { bool rejected = false; try { fn(); } catch (const std::exception&) { rejected = true; } check(rejected, "invalid operation accepted"); }
template<class F> void rejects_with(F fn, const char* message) {
    bool rejected = false;
    try { fn(); } catch (const std::exception& e) { rejected = std::string(e.what()).find(message) != std::string::npos; }
    check(rejected, "operation did not fail at its required admission check");
}
float fp16(float x) { return _cvtsh_ss(_cvtss_sh(x, _MM_FROUND_TO_NEAREST_INT)); }
lutbench::Format fmt(int type) {
    return type == 138 ? lutbench::Format::IQ3_K : type == 144 ? lutbench::Format::IQ4_KS : lutbench::Format::Q6_K;
}
std::vector<int> cpus() {
    cpu_set_t allowed; CPU_ZERO(&allowed); check(sched_getaffinity(0, sizeof(allowed), &allowed) == 0, "affinity unavailable");
    std::set<std::pair<int, int>> cores; std::vector<int> result;
    for (int c = 0; c < CPU_SETSIZE && result.size() < 5; ++c) if (CPU_ISSET(c, &allowed)) {
        int pkg = -1, core = -1;
        std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/";
        std::ifstream(base + "physical_package_id") >> pkg; std::ifstream(base + "core_id") >> core;
        if (pkg >= 0 && core >= 0 && cores.insert({pkg, core}).second) result.push_back(c);
    }
    check(result.size() >= 5, "tests need five distinct allowed physical cores for asymmetric 2/3 groups"); return result;
}
// Timing only (LUT_TIMING_ALL_CORES=1): every allowed distinct physical core, e.g. production's 42 under taskset.
std::vector<int> timing_cpus() {
    const char* all = std::getenv("LUT_TIMING_ALL_CORES");
    if (!all || std::string(all) != "1") return cpus();
    cpu_set_t allowed; CPU_ZERO(&allowed); check(sched_getaffinity(0, sizeof(allowed), &allowed) == 0, "affinity unavailable");
    std::set<std::pair<int, int>> cores; std::vector<int> result;
    for (int c = 0; c < CPU_SETSIZE; ++c) if (CPU_ISSET(c, &allowed)) {
        int pkg = -1, core = -1;
        std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/";
        std::ifstream(base + "physical_package_id") >> pkg; std::ifstream(base + "core_id") >> core;
        if (pkg >= 0 && core >= 0 && cores.insert({pkg, core}).second) result.push_back(c);
    }
    check(result.size() >= 2, "timing needs two allowed physical cores"); return result;
}
std::vector<int> allowed_siblings() {
    cpu_set_t allowed; CPU_ZERO(&allowed); check(sched_getaffinity(0, sizeof(allowed), &allowed) == 0, "affinity unavailable");
    std::map<std::pair<int, int>, int> cores;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) if (CPU_ISSET(cpu, &allowed)) {
        int package = -1, core = -1;
        const auto base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
        std::ifstream(base + "physical_package_id") >> package; std::ifstream(base + "core_id") >> core;
        if (package < 0 || core < 0) continue;
        auto found = cores.find({package, core});
        if (found != cores.end()) return {found->second, cpu};
        cores[{package, core}] = cpu;
    }
    return {};
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
    Fixture(Temporary& temp, int layer, int count, int width = 256, int intermediate = 512) : h(width), i(intermediate), experts(count) {
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
    // Test/probe reader only. Production descriptors come from the independent
    // SHA/GGUF validator; this helper never grants artifact or quality admission.
    Fixture(const std::string& path, int layer) : h(4096), i(2048), experts(2) {
        input.model_layer = layer; input.internal_li = layer - 3; input.path = path;
        input.identity = ft_lut::file_identity(path);
        std::ifstream file(path, std::ios::binary); check(bool(file), "cannot read probe GGUF");
        auto take = [&](void* p, size_t n) {
            check(uint64_t(file.tellg()) <= (64ULL << 20) && n <= (64ULL << 20) - uint64_t(file.tellg()), "probe header exceeds 64 MiB");
            file.read(static_cast<char*>(p), std::streamsize(n)); check(bool(file), "truncated probe GGUF header");
        };
        auto u32 = [&] { uint32_t n; take(&n, 4); return n; };
        auto u64 = [&] { uint64_t n; take(&n, 8); return n; };
        auto str = [&] { uint64_t n = u64(); check(n <= 1024, "probe GGUF string too long"); std::string s(size_t(n), '\0'); if (n) take(s.data(), size_t(n)); return s; };
        auto skip = [&](uint64_t n) { check(n <= (64ULL << 20) && uint64_t(file.tellg()) <= (64ULL << 20) - n, "probe metadata too large"); file.seekg(std::streamoff(n), std::ios::cur); check(bool(file), "invalid probe metadata"); };
        auto scalar_size = [](uint32_t type) -> unsigned { switch(type) { case 0: case 1: case 7: return 1; case 2: case 3: return 2; case 4: case 5: case 6: return 4; case 10: case 11: case 12: return 8; default: return 0; } };
        char magic[4]; take(magic, 4); check(std::memcmp(magic, "GGUF", 4) == 0 && u32() == 3, "probe needs little-endian GGUF v3");
        uint64_t nt = u64(), nk = u64(); check(nt == 3 && nk <= 4096, "invalid probe GGUF descriptor counts");
        uint32_t alignment = 32; bool architecture = false; std::set<std::string> metadata;
        for (uint64_t k = 0; k < nk; ++k) {
            std::string name = str(); uint32_t type = u32(); check(metadata.insert(name).second, "duplicate probe metadata");
            if (name == "general.architecture") { check(type == 8, "invalid probe architecture type"); architecture = str() == "glm5-next"; }
            else if (name == "general.alignment") { check(type == 4, "invalid probe alignment type"); alignment = u32(); }
            else if (scalar_size(type)) skip(scalar_size(type));
            else if (type == 8) skip(u64());
            else if (type == 9) { uint32_t item = u32(); uint64_t count = u64(); check(count <= 1000000 && item != 9, "invalid probe metadata array");
                if (scalar_size(item)) skip(count * scalar_size(item));
                else { check(item == 8, "unknown probe metadata array type"); for (uint64_t q = 0; q < count; ++q) skip(u64()); }
            } else throw std::runtime_error("unknown probe metadata type");
        }
        check(architecture && alignment && alignment <= 65536 && !(alignment & (alignment - 1)), "probe architecture/alignment mismatch");
        std::set<int> matrices;
        for (int t = 0; t < 3; ++t) {
            std::string name = str(); int m = -1;
            for (int p = 0; p < 3; ++p) if (name == "blk." + std::to_string(layer) + ".ffn_" + std::string(p == 0 ? "gate" : p == 1 ? "up" : "down") + "_exps.weight") m = p;
            check(m >= 0 && matrices.insert(m).second && u32() == 3, "invalid probe tensor name/rank");
            uint64_t width = u64(), rows = u64(), count = u64(); uint32_t type = u32(); uint64_t offset = u64();
            check(width == uint64_t(m == 2 ? i : h) && rows == uint64_t(m == 2 ? h : i) && count == 2, "probe tensor is not production-width E2");
            check(type == uint32_t(m == 2 ? (layer == 11 ? 14 : 144) : (layer == 11 ? 144 : 138)), "probe tensor recipe mismatch");
            stride[m] = lutbench::row_bytes(fmt(int(type)), int(width));
            input.tensors[m] = {offset, stride[m] * rows * count, int(type), int(width), int(rows), int(count)};
        }
        uint64_t data_start = (uint64_t(file.tellg()) + alignment - 1) / alignment * alignment;
        for (int m = 0; m < 3; ++m) {
            auto& tensor = input.tensors[m]; check(tensor.offset % alignment == 0 && data_start <= input.identity.size && tensor.offset <= input.identity.size - data_start, "probe tensor offset exceeds file");
            tensor.offset += data_start; check(tensor.nbytes <= input.identity.size - tensor.offset && tensor.nbytes <= (64ULL << 20), "probe tensor payload exceeds bound");
            packed[m].resize(size_t(tensor.nbytes)); file.seekg(std::streamoff(tensor.offset)); file.read(reinterpret_cast<char*>(packed[m].data()), std::streamsize(packed[m].size())); check(bool(file), "short probe tensor payload");
        }
        check(ft_lut::file_identity(path) == input.identity, "probe input changed while reading");
    }
    const uint8_t* row(int m, int expert, int r) const {
        return packed[m].data() + (size_t(expert) * (m == 2 ? h : i) + r) * stride[m];
    }
};
void dot_m(lutbench::Format f, const void* row, const lutbench::Activation* a, int n, int m, float* y) {
    if (m == 3) { lutbench::dot_multi(f, row, a, n, 2, y); lutbench::dot_multi(f, row, a + 2, n, 1, y + 2); }
    else lutbench::dot_multi(f, row, a, n, m, y);
}
std::vector<float> reference(const Fixture& fixture, const std::vector<float>& x, const ft_lut::Routes& routes, lutbench::Path path, float limit = 10.f) {
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
            for (int t = 0; t < count; ++t) hidden[size_t(t) * fixture.i + r] = std::min(g[t] / (1.f + std::exp(-g[t])), limit) * std::max(-limit, std::min(limit, u[t]));
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
void numeric(ft_lut::Backend& backend, const Fixture& f, lutbench::Path path, int& cases, double& worst, float limit = 10.f) {
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
        auto expected = reference(f, x, routes, path, limit);
        std::vector<float> got(expected.size(), 123.f);
        backend.forward(f.input.internal_li, x.data(), m, routes, got.data());
        std::vector<float> prechecked(got.size(), 123.f);
        backend.forward_finite_input(f.input.internal_li, x.data(), m, routes, prechecked.data());
        check(std::memcmp(got.data(), prechecked.data(), got.size() * sizeof(float)) == 0,
              "prechecked ingress changed backend output bits");
        double error = relative(got, expected); worst = std::max(worst, error);
        check(error <= 1e-5, "TP2 output differs from canonical packed-weight reference"); ++cases;
    }
}
struct GroupedRows {
    std::vector<int32_t> experts, offsets{0}, token;
    std::vector<float> weight;
    void append(int expert, const std::vector<int>& tokens) {
        experts.push_back(expert);
        for (int t : tokens) { token.push_back(t); weight.push_back(fp16(.035f + .003f * float((expert + 3 * t) % 23))); }
        offsets.push_back(int32_t(token.size()));
    }
};
uint32_t random_word(uint32_t& state) { state = state * 1664525U + 1013904223U; return state; }
std::vector<uint16_t> grouped_input(int ntok, int h, uint32_t seed) {
    std::vector<uint16_t> result(size_t(ntok) * h);
    for (auto& value : result) value = _cvtss_sh(float(int(random_word(seed) >> 8) - 8388608) / 8388608.f, _MM_FROUND_TO_NEAREST_INT);
    return result;
}
std::vector<float> expanded(const std::vector<uint16_t>& input) {
    std::vector<float> result(input.size());
    for (size_t n = 0; n < result.size(); ++n) result[n] = _cvtsh_ss(input[n]);
    return result;
}
bool grouped(ft_lut::Backend& backend, int li, const std::vector<uint16_t>& x, int ntok,
             const GroupedRows& rows, std::vector<float>& output, std::string& why,
             double budget = 30000, int* chunks = nullptr) {
    return backend.forward_grouped(li, x.data(), ntok, int(rows.experts.size()), rows.experts.data(),
                                   rows.offsets.data(), rows.token.data(), rows.weight.data(), output.data(), budget, &why, chunks);
}
void same_bits(const std::vector<float>& a, const std::vector<float>& b, const char* message) {
    check(a.size() == b.size(), message);
    if (std::memcmp(a.data(), b.data(), a.size() * sizeof(float)) != 0) {
        for (size_t n = 0; n < a.size(); ++n) if (std::memcmp(&a[n], &b[n], sizeof(float)) != 0) {
            uint32_t got, expected; std::memcpy(&got, &a[n], sizeof(got)); std::memcpy(&expected, &b[n], sizeof(expected));
            std::cerr << message << " index=" << n << " got_bits=0x" << std::hex << got << " expected_bits=0x" << expected << std::dec << '\n';
            break;
        }
        check(false, message);
    }
}
// Independent TP2 oracle: packed slices come from the loader, but no grouped
// worklist, scratch, projection routine or coordinator is used here.
std::vector<float> grouped_reference(ft_lut::Backend& backend, const Fixture& f,
                                     const std::vector<uint16_t>& x, int ntok, const GroupedRows& rows) {
    const auto input = expanded(x);
    const int half = f.i / 2;
    std::array<std::vector<float>, 2> partial;
    for (int group = 0; group < 2; ++group) {
        partial[group].resize(size_t(ntok) * f.h, 0.f);
        for (size_t j = 0; j < rows.experts.size(); ++j) {
            const int start = rows.offsets[j], count = rows.offsets[j + 1] - start;
            std::array<std::vector<uint8_t>, 3> packed;
            std::array<size_t, 3> stride;
            for (int matrix = 0; matrix < 3; ++matrix) {
                packed[matrix] = backend.fixture_slice(f.input.internal_li, group, rows.experts[j], matrix);
                stride[matrix] = lutbench::row_bytes(fmt(f.input.tensors[matrix].type_id), matrix == 2 ? half : f.h);
            }
            std::vector<lutbench::Activation> activation(count), hidden_q(count);
            std::vector<float> hidden(size_t(count) * half);
            for (int t = 0; t < count; ++t)
                lutbench::prepare(lutbench::Path::Q8, input.data() + size_t(rows.token[start + t]) * f.h, f.h, activation[t]);
            for (int r = 0; r < half; ++r) for (int tile = 0; tile < count; tile += 4) {
                const int m = std::min(4, count - tile); float g[4], u[4];
                dot_m(fmt(f.input.tensors[0].type_id), packed[0].data() + size_t(r) * stride[0], activation.data() + tile, f.h, m, g);
                dot_m(fmt(f.input.tensors[1].type_id), packed[1].data() + size_t(r) * stride[1], activation.data() + tile, f.h, m, u);
                for (int t = 0; t < m; ++t) {
                    check(std::isfinite(g[t]) && std::isfinite(u[t]), "oracle nonfinite gate/up projection");
                    float gate = std::min(g[t] / (1.0f + std::exp(-g[t])), 10.f);
                    float up = std::max(-10.f, std::min(10.f, u[t]));
                    float value = gate * up;
                    check(std::isfinite(value), "oracle nonfinite hidden activation");
                    hidden[size_t(tile + t) * half + r] = value;
                }
            }
            for (int t = 0; t < count; ++t)
                lutbench::prepare(lutbench::Path::Q8, hidden.data() + size_t(t) * half, half, hidden_q[t]);
            for (int r = 0; r < f.h; ++r) for (int tile = 0; tile < count; tile += 4) {
                const int m = std::min(4, count - tile); float y[4];
                dot_m(fmt(f.input.tensors[2].type_id), packed[2].data() + size_t(r) * stride[2], hidden_q.data() + tile, half, m, y);
                for (int t = 0; t < m; ++t) {
                    float contribution = rows.weight[start + tile + t] * y[t];
                    float& value = partial[group][size_t(rows.token[start + tile + t]) * f.h + r];
                    float updated = value + contribution;
                    // Decode validates the separately rounded product. Omitting
                    // this contract lets production GCC fuse the multiply/add.
                    check(std::isfinite(y[t]) && std::isfinite(contribution) && std::isfinite(updated), "oracle nonfinite weighted down output");
                    value = updated;
                }
            }
        }
    }
    for (size_t n = 0; n < partial[0].size(); ++n) {
        partial[0][n] = partial[0][n] + partial[1][n];
        check(std::isfinite(partial[0][n]), "oracle nonfinite merged result");
    }
    return partial[0];
}
void grouped_checks(const std::vector<int>& workers, const Fixture& ordinary, const Fixture& upgrade) {
    ft_lut::Options options; options.cpus = workers; options.virtual_tp2 = true; options.activation = "q8";
    auto backend = ft_lut::Backend::offline_fixture(options);
    backend->add_layer(ordinary.input); backend->add_layer(upgrade.input);
    check(backend->grouped_ready() && !backend->grouped_runtime_failed(), "loaded grouped backend is not ready");
    int parity_cases = 0, oracle_cases = 0, chunk_cases = 0;
    for (const Fixture* f : {&ordinary, &upgrade}) {
        for (int ntok = 1; ntok <= 4; ++ntok) for (uint32_t seed = 1; seed <= 8; ++seed) {
            const auto x = grouped_input(ntok, f->h, seed + uint32_t(f->input.model_layer * 31));
            const auto xf = expanded(x); GroupedRows rows; ft_lut::Routes routes(ntok);
            for (int e = 0; e < 8; ++e) {
                std::vector<int> tokens;
                const unsigned mask = e == 0 ? (1U << ntok) - 1 : unsigned(e + seed) % (1U << ntok);
                for (int t = 0; t < ntok; ++t) if (mask & (1U << t)) tokens.push_back(t);
                if (tokens.empty()) continue;
                rows.append(e, tokens);
                for (int r = rows.offsets[rows.offsets.size() - 2]; r < rows.offsets.back(); ++r)
                    routes[rows.token[r]].push_back({e, rows.weight[r]});
            }
            std::vector<float> decode(x.size(), 123.f), actual(x.size(), 123.f); std::string why;
            backend->forward(f->input.internal_li, xf.data(), ntok, routes, decode.data());
            check(grouped(*backend, f->input.internal_li, x, ntok, rows, actual, why), "valid small grouped call failed");
            same_bits(actual, decode, "grouped/decode bit parity failed for random M=1..4 or gapped pair");
            ++parity_cases;
        }
        // Every count5..64 occurs for multiple overlapping experts. Gapped
        // tokens leave unused outputs and exercise every final tile shape.
        for (int count = 5; count <= 64; ++count) {
            const int ntok = 132; const auto x = grouped_input(ntok, f->h, uint32_t(count + f->input.model_layer));
            GroupedRows rows;
            for (int e : {0, 2, 7}) {
                std::vector<int> tokens;
                for (int t = 0; t < count - (e == 2 && count > 5 ? 1 : 0); ++t) tokens.push_back(2 * t + e % 2);
                rows.append(e, tokens);
            }
            if (count == 17) rows.weight[0] = -.25f; // grouped admits any finite routing weight.
            auto expected = grouped_reference(*backend, *f, x, ntok, rows);
            std::vector<float> actual(x.size(), 123.f); std::string why; int chunks = -1;
            backend->fixture_set_grouped_chunk_rows(512);
            check(grouped(*backend, f->input.internal_li, x, ntok, rows, actual, why, 30000, &chunks) && chunks == 1,
                  "valid large grouped single chunk failed");
            const std::string context = "grouped output differs from independent TP2 dot_multi oracle layer=" + std::to_string(f->input.model_layer) + " rows=" + std::to_string(count);
            same_bits(actual, expected, context.c_str()); ++oracle_cases;
            if (count == 5 || count == 17 || count == 63 || count == 64) for (int cap : {1, count}) {
                backend->fixture_set_grouped_chunk_rows(cap);
                std::vector<float> chunked(actual.size(), 123.f);
                check(grouped(*backend, f->input.internal_li, x, ntok, rows, chunked, why, 30000, &chunks) && chunks > 1,
                      "forced grouped chunks were not published");
                same_bits(actual, chunked, "chunk boundary changed grouped output bits"); ++chunk_cases;
            }
        }
    }
    backend->fixture_set_grouped_chunk_rows(512);
    auto x = grouped_input(4, ordinary.h, 81); GroupedRows valid;
    valid.append(0, {0, 2}); valid.append(2, {1, 3}); valid.append(7, {0, 1, 2, 3});
    auto reject = [&](const GroupedRows& rows, const std::vector<uint16_t>& input, int ntok) {
        const auto generation = backend->fixture_generations(); const auto saved = input;
        std::vector<float> output(size_t(std::max(1, ntok)) * ordinary.h, 123.f); std::string why; int chunks = -1;
        check(!grouped(*backend, 0, input, ntok, rows, output, why, 30000, &chunks) && !why.empty(), "invalid grouped request accepted");
        check(std::all_of(output.begin(), output.end(), [](float v) { return v == 123.f; }), "grouped rejection published caller output");
        const auto after = backend->fixture_generations();
        for (int g = 0; g < 2; ++g) check(generation[g].epoch == after[g].epoch, "grouped validation published worker generation");
        check(input == saved && backend->grouped_ready() && !backend->grouped_runtime_failed(), "grouped admission rejection changed backend/input state");
    };
    auto bad = valid; std::swap(bad.experts[0], bad.experts[1]); reject(bad, x, 4);
    bad = valid; bad.experts[1] = bad.experts[0]; reject(bad, x, 4);
    bad = valid; bad.experts[0] = -1; reject(bad, x, 4);
    bad = valid; bad.experts.back() = ordinary.experts; reject(bad, x, 4);
    bad = valid; bad.token[1] = bad.token[0]; reject(bad, x, 4);
    bad = valid; std::swap(bad.token[0], bad.token[1]); reject(bad, x, 4);
    bad = valid; bad.token[0] = -1; reject(bad, x, 4);
    bad = valid; bad.token[0] = 4; reject(bad, x, 4);
    bad = valid; bad.offsets[0] = 1; reject(bad, x, 4);
    bad = valid; bad.offsets[1] = -1; reject(bad, x, 4);
    bad = valid; bad.offsets[2] = bad.offsets[1] - 1; reject(bad, x, 4);
    for (float value : {INFINITY, -INFINITY, std::numeric_limits<float>::quiet_NaN()}) {
        bad = valid; bad.weight[0] = value; reject(bad, x, 4);
    }
    for (uint16_t value : {uint16_t(0x7c00), uint16_t(0xfc00), uint16_t(0x7e00)})
        for (size_t position : {size_t(0), size_t(7), size_t(8), x.size() - 1}) {
            auto invalid_x = x; invalid_x[position] = value; reject(valid, invalid_x, 4);
        }
    GroupedRows too_many; std::vector<int> tokens(65); for (int t = 0; t < 65; ++t) tokens[t] = t;
    too_many.append(0, tokens); reject(too_many, grouped_input(65, ordinary.h, 3), 65);
    reject(valid, x, 0); reject(GroupedRows{}, grouped_input(1025, ordinary.h, 3), 1025);
    // At the exact token bound, an empty route set writes all zero outputs.
    auto boundary_x = grouped_input(1024, ordinary.h, 9); std::vector<float> boundary(boundary_x.size(), 123.f); std::string why;
    check(grouped(*backend, 0, boundary_x, 1024, GroupedRows{}, boundary, why), "1024-token empty grouped request rejected");
    check(std::all_of(boundary.begin(), boundary.end(), [](float v) { return v == 0.f; }), "unrouted grouped tokens were not zeroed");
    // A decode generation owns the same lock used for grouped admission.
    backend->inject_delay(0, 1, 100); const auto before = backend->fixture_generations();
    auto xf = expanded(x); std::vector<float> decode(x.size()); ft_lut::Routes route(4); route[0] = {{0, 1.f}};
    std::atomic<bool> done{false}; std::exception_ptr error;
    std::thread in_flight([&] { try { backend->forward(0, xf.data(), 4, route, decode.data()); } catch (...) { error = std::current_exception(); } done.store(true, std::memory_order_release); });
    const auto witness_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2); bool active = false;
    while (!done.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < witness_deadline) {
        const auto generation = backend->fixture_generations();
        if (generation[0].epoch != before[0].epoch) { active = true; break; }
        std::this_thread::yield();
    }
    std::vector<float> busy_output(x.size(), 123.f);
    const bool ready_while_busy = backend->grouped_ready();
    const bool busy_ok = grouped(*backend, 0, x, 4, valid, busy_output, why);
    in_flight.join(); if (error) std::rethrow_exception(error);
    check(active && !ready_while_busy && !busy_ok && why == "busy", "grouped call did not reject concurrent decode as busy");
    check(std::all_of(busy_output.begin(), busy_output.end(), [](float v) { return v == 123.f; }), "busy grouped request published output");
    backend->inject_delay(-1, 1, 0); check(backend->grouped_ready(), "busy rejection became sticky");
    options.activation = "i16"; auto i16 = ft_lut::Backend::offline_fixture(options); i16->add_layer(ordinary.input);
    std::fill(busy_output.begin(), busy_output.end(), 123.f);
    check(!grouped(*i16, 0, x, 4, valid, busy_output, why) && (why.find("Q8") != std::string::npos || why.find("q8") != std::string::npos), "I16 grouped mode lacks clear Q8 rejection");
    check(std::all_of(busy_output.begin(), busy_output.end(), [](float v) { return v == 123.f; }), "I16 rejection published output");
    options.activation = "q8";
    // Each individual chunk is below the budget; the whole eight-chunk call
    // exceeds it. Resetting a relative timeout at each chunk would pass wrongly.
    auto timed = ft_lut::Backend::offline_fixture(options); timed->add_layer(ordinary.input);
    timed->fixture_set_grouped_chunk_rows(1); timed->inject_delay(0, 1, 30);
    GroupedRows many; for (int e = 0; e < 8; ++e) many.append(e, {0});
    int chunks = -1; std::fill(busy_output.begin(), busy_output.end(), 123.f);
    check(!grouped(*timed, 0, x, 4, many, busy_output, why, 120, &chunks), "grouped chunks reset their deadline");
    check(chunks >= 2 && chunks < 8 && timed->receipt().timed_out_jobs == 1, "grouped absolute deadline was not exercised across chunks");
    check(std::all_of(busy_output.begin(), busy_output.end(), [](float v) { return v == 123.f; }), "grouped timeout published partial output");
    timed->inject_delay(-1, 1, 0);
    check(timed->grouped_runtime_failed() && !timed->grouped_ready(), "grouped timeout is not sticky");
    check(!grouped(*timed, 0, x, 4, valid, busy_output, why), "sticky grouped timeout accepted a later call");
    for (int phase = 0; phase < 4; ++phase) {
        auto failed = ft_lut::Backend::offline_fixture(options); failed->add_layer(ordinary.input); failed->inject_failure(phase, 1);
        std::fill(busy_output.begin(), busy_output.end(), 123.f);
        check(!grouped(*failed, 0, x, 4, valid, busy_output, why) && failed->grouped_runtime_failed() && !failed->grouped_ready(), "grouped worker failure is not sticky");
        check(std::all_of(busy_output.begin(), busy_output.end(), [](float v) { return v == 123.f; }), "grouped worker failure published output");
    }
    std::cout << "GROUPED_PASS decode_bit_cases=" << parity_cases << " independent_oracle_cases=" << oracle_cases
              << " rows_per_expert=5..64 forced_chunk_cases=" << chunk_cases
              << " gapped_pairs=0,2;1,3 rejection_no_worker_publication=covered token_bound=1024 busy=covered sticky_failures=covered absolute_chunk_deadline=covered\n";
}
void grouped_boundary_checks(const std::vector<int>& workers) {
    Temporary temp; Fixture f(temp, 3, 9);
    ft_lut::Options options; options.cpus = workers; options.virtual_tp2 = true; options.activation = "q8";
    auto ordinary = ft_lut::Backend::offline_fixture(options);
    rejects([&] { ordinary->add_layer(f.input); }); // The original eight-expert admission remains strict.
    auto backend = ft_lut::Backend::grouped_benchmark_fixture(options); backend->add_layer(f.input);
    const auto x = grouped_input(1024, f.h, 911); GroupedRows rows;
    std::vector<int> tokens(64); for (int t = 0; t < 64; ++t) tokens[t] = 2 * t;
    for (int e = 0; e < 8; ++e) rows.append(e, tokens);
    std::vector<float> output(x.size(), 123.f); std::string why; int chunks = -1;
    check(grouped(*backend, 0, x, 1024, rows, output, why, 30000, &chunks) && chunks == 1, "512 grouped rows did not fit exactly one chunk");
    rows.append(8, {1023});
    auto expected = grouped_reference(*backend, f, x, 1024, rows);
    check(grouped(*backend, 0, x, 1024, rows, output, why, 30000, &chunks) && chunks == 2, "513 grouped rows did not use two default chunks");
    same_bits(output, expected, "default 512-row chunk boundary or token1023 changed output bits");
    check(std::any_of(output.begin() + size_t(1023) * f.h, output.end(), [](float value) { return value != 0.f; }), "last admitted token did not receive its expert row");
    backend->fixture_set_grouped_chunk_rows(64);
    check(grouped(*backend, 0, x, 1024, rows, output, why, 30000, &chunks) && chunks == 9, "64-row forced chunks did not preserve whole experts");
    same_bits(output, expected, "nine forced chunks changed grouped output bits");
    const auto receipt = backend->receipt();
    check(receipt.payload_bytes[0] + receipt.payload_bytes[1] <= (4ULL << 30) && receipt.offline_fixture && !receipt.bank_complete && !receipt.placement_verified,
          "benchmark fixture exceeded its bound or claimed serving/NUMA admission");
    std::cout << "GROUPED_BOUNDARY_PASS rows=512,513 chunks=1,2,9 token1023=covered ordinary_fixture_bound=8\n";
}
void grouped_numeric_failure_checks(const std::vector<int>& workers) {
    Temporary temp; Fixture f(temp, 3, 1);
    {
        std::fstream file(f.input.path, std::ios::in | std::ios::out | std::ios::binary);
        for (int matrix = 0; matrix < 3; ++matrix) {
            const auto& tensor = f.input.tensors[matrix]; std::vector<float> row(tensor.width, .01f);
            for (int r = 0; r < tensor.rows; ++r)
                lutbench::quantize(fmt(tensor.type_id), row.data(), f.packed[matrix].data() + size_t(r) * f.stride[matrix], tensor.width);
            file.seekp(std::streamoff(tensor.offset));
            file.write(reinterpret_cast<const char*>(f.packed[matrix].data()), std::streamsize(tensor.nbytes));
        }
        check(bool(file), "cannot write grouped numerical-failure fixture");
    }
    f.input.identity = ft_lut::file_identity(f.input.path);
    ft_lut::Options options; options.cpus = workers; options.virtual_tp2 = true; options.activation = "q8";
    GroupedRows rows; rows.append(0, {0}); rows.weight[0] = std::numeric_limits<float>::max();
    for (float value : {.25f, 25.f}) {
        auto backend = ft_lut::Backend::offline_fixture(options); backend->add_layer(f.input);
        std::vector<uint16_t> x(f.h, _cvtss_sh(value, _MM_FROUND_TO_NEAREST_INT)); std::vector<float> output(f.h, 123.f); std::string why;
        check(!grouped(*backend, 0, x, 1, rows, output, why), "grouped finite weighted/merge overflow accepted");
        check(why.find(value == .25f ? "merged" : "weighted") != std::string::npos, "grouped overflow failed in the wrong numerical phase");
        check(backend->grouped_runtime_failed() && !backend->grouped_ready(), "grouped numerical overflow is not sticky");
        check(std::all_of(output.begin(), output.end(), [](float v) { return v == 123.f; }), "grouped numerical overflow published output");
    }
    // IQ3_K begins each256-weight block with an fp16 scale. Corrupt only that
    // scale, retaining a structurally admitted file and otherwise finite rows.
    {
        std::fstream file(f.input.path, std::ios::in | std::ios::out | std::ios::binary);
        const uint16_t inf = 0x7c00; file.seekp(std::streamoff(f.input.tensors[0].offset)); file.write(reinterpret_cast<const char*>(&inf), sizeof(inf));
        check(bool(file), "cannot corrupt grouped projection scale");
    }
    f.input.identity = ft_lut::file_identity(f.input.path);
    auto backend = ft_lut::Backend::offline_fixture(options); backend->add_layer(f.input);
    rows.weight[0] = 1.f; std::vector<uint16_t> x(f.h, _cvtss_sh(.25f, _MM_FROUND_TO_NEAREST_INT));
    std::vector<float> output(f.h, 123.f); std::string why;
    check(!grouped(*backend, 0, x, 1, rows, output, why) && why.find("projection") != std::string::npos, "corrupted grouped scale did not fail its projection");
    check(backend->grouped_runtime_failed() && !backend->grouped_ready(), "grouped nonfinite projection is not sticky");
    check(std::all_of(output.begin(), output.end(), [](float v) { return v == 123.f; }), "grouped nonfinite projection published output");
    std::cout << "GROUPED_NUMERIC_FAILURE_PASS nonfinite_projection=covered finite_weight_overflow=covered merge_overflow=covered sticky=covered\n";
}
int grouped_timing(int argc, char** argv) {
    using Clock = std::chrono::steady_clock;
    check(argc == 2 || argc == 7, "usage: --grouped-timing [EXPERTS ROWS H I REPETITIONS]");
    const int nexp = argc == 7 ? std::stoi(argv[2]) : 250, nrow = argc == 7 ? std::stoi(argv[3]) : 10;
    const int h = argc == 7 ? std::stoi(argv[4]) : 4096, intermediate = argc == 7 ? std::stoi(argv[5]) : 2048;
    const int repetitions = argc == 7 ? std::stoi(argv[6]) : 3;
    check(nexp > 0 && nexp <= 288 && nrow > 0 && nrow <= 64 && h >= 256 && h <= 4096 && h % 256 == 0 &&
          intermediate >= 512 && intermediate <= 2048 && intermediate % 512 == 0 && repetitions > 0 && repetitions <= 20,
          "grouped timing dimensions exceed test-only bounds");
    Temporary temp; ft_lut::LayerInput layer; layer.model_layer = 3; layer.internal_li = 0; layer.path = temp.name("random-iq3-iq4");
    uint64_t extent = 64, payload = 0;
    for (int matrix = 0; matrix < 3; ++matrix) {
        const int width = matrix == 2 ? intermediate : h, rows = matrix == 2 ? h : intermediate, type = matrix == 2 ? 144 : 138;
        const uint64_t bytes = uint64_t(lutbench::row_bytes(fmt(type), width)) * rows * nexp;
        layer.tensors[matrix] = {extent, bytes, type, width, rows, nexp}; payload += bytes; extent = (extent + bytes + 31) / 32 * 32;
    }
    check(payload <= (4ULL << 30), "synthetic packed weights exceed4GiB");
    {
        std::ofstream file(layer.path, std::ios::binary); uint32_t seed = 0xc001cafeU;
        // All bit patterns are valid IQ3_K/IQ4_KS codes and integer scales.
        // Randomize them directly, then give every floating scale a small,
        // finite value. One expert buffer bounds generation memory and avoids
        // retaining a second multi-GiB float or packed copy beside the backend.
        for (int matrix = 0; matrix < 3; ++matrix) {
            const auto& tensor = layer.tensors[matrix]; const size_t stride = lutbench::row_bytes(fmt(tensor.type_id), tensor.width);
            check(stride == (matrix == 2 ? 4 + size_t(tensor.width / 256) * 136 : size_t(tensor.width / 256) * 110), "random packed generator layout mismatch");
            std::vector<uint8_t> expert(stride * tensor.rows); file.seekp(std::streamoff(tensor.offset));
            for (int e = 0; e < nexp; ++e) {
                for (size_t p = 0; p < expert.size(); p += sizeof(uint32_t)) {
                    uint32_t word = random_word(seed); std::memcpy(expert.data() + p, &word, std::min(sizeof(word), expert.size() - p));
                }
                for (int r = 0; r < tensor.rows; ++r) {
                    auto* row = expert.data() + size_t(r) * stride;
                    if (matrix == 2) {
                        const float scale = .000005f + float(random_word(seed) >> 28) * .0000002f; std::memcpy(row, &scale, sizeof(scale));
                    } else for (int b = 0; b < tensor.width / 256; ++b) {
                        const uint16_t scale = _cvtss_sh(.00005f + float(random_word(seed) >> 28) * .000002f, _MM_FROUND_TO_NEAREST_INT);
                        std::memcpy(row + size_t(b) * 110, &scale, sizeof(scale));
                    }
                }
                file.write(reinterpret_cast<const char*>(expert.data()), std::streamsize(expert.size()));
            }
        }
        check(bool(file), "cannot write random packed timing fixture");
    }
    layer.identity = ft_lut::file_identity(layer.path);
    ft_lut::Options options; options.cpus = timing_cpus(); options.virtual_tp2 = true; options.activation = "q8"; options.job_timeout_ms = 3600000;
    std::printf("GROUPED_TIMING_WORKERS %zu\n", options.cpus.size());
    auto backend = ft_lut::Backend::grouped_benchmark_fixture(options); backend->add_layer(layer);
    const auto receipt = backend->receipt();
    check(receipt.payload_bytes[0] + receipt.payload_bytes[1] <= (4ULL << 30), "owned synthetic weights exceed4GiB");
    const int ntok = 128; const auto x = grouped_input(ntok, h, 0x12345678U); const auto xf = expanded(x);
    GroupedRows rows;
    for (int e = 0; e < nexp; ++e) {
        std::vector<int> tokens; const int first = (e * 17) % (ntok - nrow + 1);
        for (int t = 0; t < nrow; ++t) tokens.push_back(first + t);
        rows.append(e, tokens);
    }
    struct DecodeCall { int m; std::vector<float> x, out; ft_lut::Routes routes; };
    std::vector<DecodeCall> calls;
    for (int e = 0; e < nexp; ++e) for (int tile = 0; tile < nrow; tile += 4) {
        const int m = std::min(4, nrow - tile); DecodeCall call; call.m = m; call.routes.resize(m); call.x.resize(size_t(m) * h); call.out.resize(call.x.size());
        for (int t = 0; t < m; ++t) {
            const int row = rows.offsets[e] + tile + t;
            std::copy_n(xf.data() + size_t(rows.token[row]) * h, h, call.x.data() + size_t(t) * h);
            call.routes[t].push_back({e, rows.weight[row]});
        }
        calls.push_back(std::move(call));
    }
    std::vector<float> output(x.size()); std::string why; int chunks = -1;
    auto run_grouped = [&] {
        check(grouped(*backend, 0, x, ntok, rows, output, why, 3600000, &chunks), "grouped timing run failed");
    };
    auto run_decode = [&] { for (auto& call : calls) backend->forward(0, call.x.data(), call.m, call.routes, call.out.data()); };
    run_grouped(); run_decode();
    // Compare one expert at a time: its4/4/remainder decode tiles have exactly
    // the grouped tile arithmetic. Summing already-merged outputs of different
    // experts would introduce a different TP2 rounding order.
    for (int e : {0, nexp - 1}) {
        GroupedRows probe; probe.experts = {e}; probe.offsets = {0, nrow};
        probe.token.assign(rows.token.begin() + rows.offsets[e], rows.token.begin() + rows.offsets[e + 1]);
        probe.weight.assign(rows.weight.begin() + rows.offsets[e], rows.weight.begin() + rows.offsets[e + 1]);
        check(grouped(*backend, 0, x, ntok, probe, output, why, 3600000), "timing fixture single-expert check failed");
        for (int tile = 0; tile < nrow; tile += 4) {
            const auto& call = calls[size_t(e) * ((nrow + 3) / 4) + tile / 4];
            for (int t = 0; t < call.m; ++t)
                check(std::memcmp(output.data() + size_t(probe.token[tile + t]) * h, call.out.data() + size_t(t) * h, size_t(h) * sizeof(float)) == 0,
                      "random packed timing weights changed grouped/decode tile bits");
        }
    }
    std::vector<double> grouped_ms, decode_ms;
    for (int repeat = 0; repeat < repetitions; ++repeat) {
        auto begin = Clock::now(); run_grouped(); grouped_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - begin).count());
        begin = Clock::now(); run_decode(); decode_ms.push_back(std::chrono::duration<double, std::milli>(Clock::now() - begin).count());
    }
    std::sort(grouped_ms.begin(), grouped_ms.end()); std::sort(decode_ms.begin(), decode_ms.end());
    double checksum = 0; for (float value : output) { check(std::isfinite(value), "nonfinite timing output"); checksum += value; }
    std::cout << "GROUPED_TIMING scope=workstation_synthetic placement_claim=none experts=" << nexp << " rows_per_expert=" << nrow
              << " H=" << h << " I=" << intermediate << " tokens=" << ntok << " repetitions=" << repetitions
              << " grouped_p50_ms=" << grouped_ms[grouped_ms.size() / 2] << " repeated_decode_p50_ms=" << decode_ms[decode_ms.size() / 2]
              << " decode_calls=" << calls.size() << " grouped_chunks=" << chunks << " packed_bytes=" << payload
              << " owned_payload_bytes=" << receipt.payload_bytes[0] + receipt.payload_bytes[1]
              << " scratch_mapped_bytes_per_group=" << receipt.scratch_mapped_bytes[0] << " tile_bits=verified_first_last_expert checksum=" << checksum << '\n';
    return 0;
}
void merged_overflow_checks(const std::vector<int>& workers) {
    Temporary temp;
    Fixture fixture(temp, 3, 1);
    // Uniform positive projections make each half's weighted result finite,
    // while their final sum overflows. Exercise the merge admission itself.
    {
        std::fstream file(fixture.input.path, std::ios::in | std::ios::out | std::ios::binary);
        for (int matrix = 0; matrix < 3; ++matrix) {
            const auto& tensor = fixture.input.tensors[matrix];
            std::vector<float> row(tensor.width, .01f);
            for (int r = 0; r < tensor.rows; ++r)
                lutbench::quantize(fmt(tensor.type_id), row.data(), fixture.packed[matrix].data() + size_t(r) * fixture.stride[matrix], tensor.width);
            file.seekp(std::streamoff(tensor.offset));
            file.write(reinterpret_cast<const char*>(fixture.packed[matrix].data()), std::streamsize(tensor.nbytes));
        }
        check(bool(file), "cannot write merge-overflow fixture");
    }
    fixture.input.identity = ft_lut::file_identity(fixture.input.path);
    for (const std::string activation : {"i16", "q8"}) {
        ft_lut::Options options; options.cpus = workers; options.virtual_tp2 = true; options.activation = activation;
        auto backend = ft_lut::Backend::offline_fixture(options); backend->add_layer(fixture.input);
        std::vector<float> input(fixture.h, .25f), output(fixture.h);
        for (bool finite_input : {false, true}) {
            std::fill(output.begin(), output.end(), 123.f);
            ft_lut::Routes routes{{{0, std::numeric_limits<float>::max()}}};
            rejects_with([&] {
                if (finite_input) backend->forward_finite_input(0, input.data(), 1, routes, output.data());
                else backend->forward(0, input.data(), 1, routes, output.data());
            }, "nonfinite merged LUT result");
            check(std::all_of(output.begin(), output.end(), [](float v) { return v == 123.f; }), "merge overflow published caller output");
            routes[0][0].second = 1.f;
            backend->forward(0, input.data(), 1, routes, output.data());
            check(std::all_of(output.begin(), output.end(), [](float v) { return std::isfinite(v) && v > 0; }), "backend did not recover after merge overflow");
        }
    }
}
void mapping_allocator_checks() {
    lutbench::AlignedAllocator<float> allocator;
    rejects([&] { allocator.allocate(std::numeric_limits<size_t>::max()); });
    rejects([&] { lutbench::aligned_mapping_bytes(std::numeric_limits<size_t>::max()); });
    const size_t page = size_t(sysconf(_SC_PAGESIZE));
    for (size_t count : {size_t(1), page / sizeof(float), page / sizeof(float) + 1}) {
        void* address;
        {
            lutbench::AlignedVector<float> values(count, 1.25f);
            address = values.data();
            check(reinterpret_cast<uintptr_t>(address) % page == 0, "scratch allocation is not page aligned");
            check(values.front() == 1.25f && values.back() == 1.25f, "mapped vector initialization failed");
            unsigned char resident = 0;
            check(mincore(address, page, &resident) == 0 && (resident & 1), "mapped scratch was not first touched");
        }
        // No intervening allocation: release must unmap, never retain resident
        // libc pages that a worker on the other NUMA node can later inherit.
        unsigned char resident = 0; errno = 0;
        check(mincore(address, page, &resident) == -1 && errno == ENOMEM, "scratch release retained a resident mapping");
    }
}
void verify_policy_receipts(const ft_lut::Receipt& receipt) {
    for (const auto& row : receipt.scratch) {
        check(row.payload_bytes == receipt.scratch_bytes[row.group] && row.mapped_bytes == receipt.scratch_mapped_bytes[row.group], "scratch receipt byte ledger mismatch");
        check(row.mapped_bytes >= row.payload_bytes && row.mapped_bytes <= ((receipt.activation == "q8" ? 48ULL : 32ULL) << 20), "scratch mapped allocation bound incorrect");
        if (row.preferred_policy)
            check(row.policy_mode == MPOL_PREFERRED && row.policy_nodes == std::vector<int>{row.node} && !row.policy_failures && !row.policy_errno, "scratch policy was not read back exactly");
    }
    for (const auto& row : receipt.regions) if (row.preferred_policy)
        check(row.policy_mode == MPOL_PREFERRED && row.policy_nodes == std::vector<int>{row.node}, "weight policy was not read back exactly");
    for (const auto& row : receipt.controls) {
        if (row.preferred_policy) check(row.policy_mode == MPOL_PREFERRED && row.policy_nodes == std::vector<int>{row.node}, "control policy was not read back exactly");
        for (const auto& policy : row.thread_policies) if (policy.verified)
            check(policy.mode == MPOL_PREFERRED && policy.nodes == std::vector<int>{policy.node}, "thread policy was not read back exactly");
    }
}
void idle_measure(ft_lut::Backend& backend, const Fixture& f) {
    using Clock = std::chrono::steady_clock;
    for (int tokens : {1, 4}) {
        std::vector<float> x(size_t(tokens) * f.h, .25f), out(x.size());
        ft_lut::Routes route(tokens); for (auto& token : route) token = {{0, .7f}};
        std::vector<double> warm, after_idle; auto before = backend.receipt().pool_naps;
        for (int n = 0; n < 8; ++n) {
            auto begin = Clock::now(); backend.forward(f.input.internal_li, x.data(), tokens, route, out.data());
            warm.push_back(std::chrono::duration<double, std::micro>(Clock::now() - begin).count());
            std::this_thread::sleep_for(std::chrono::milliseconds(60));   // > the 50 ms production spin, so idle workers must back off
            begin = Clock::now(); backend.forward(f.input.internal_li, x.data(), tokens, route, out.data());
            after_idle.push_back(std::chrono::duration<double, std::micro>(Clock::now() - begin).count());
        }
        auto naps = backend.receipt().pool_naps;
        check(naps[0] > before[0] && naps[1] > before[1], "idle workers did not enter bounded backoff");
        std::sort(warm.begin(), warm.end()); std::sort(after_idle.begin(), after_idle.end());
        std::cout << "IDLE_MEASUREMENT scope=real_loader_smoke H=" << f.h << " I=" << f.i
                  << " M=" << tokens << " selected_experts=1 samples=8 warm_p50_us=" << warm[4] << " after_60ms_p50_us=" << after_idle[4]
                  << " warm_max_us=" << warm.back() << " after_60ms_max_us=" << after_idle.back() << '\n';
    }
}
void integers(const std::vector<int>& values) { std::cout << '['; for (size_t n = 0; n < values.size(); ++n) { if (n) std::cout << ','; std::cout << values[n]; } std::cout << ']'; }
int numa_probe(const std::string& cpu_list, const std::string& directory, const std::string& activation) {
    ft_lut::Options options; options.activation = activation; options.virtual_tp2 = false;
    size_t start = 0;
    while (start < cpu_list.size()) { size_t end = cpu_list.find(',', start); std::string item = cpu_list.substr(start, end - start); size_t used = 0; int cpu = std::stoi(item, &used); check(used == item.size(), "invalid probe CPU list"); options.cpus.push_back(cpu); if (end == std::string::npos) break; start = end + 1; }
    Fixture ordinary(directory + "/blk.03.experts-0-1.lut.gguf", 3), upgrade(directory + "/blk.11.experts-0-1.lut.gguf", 11);
    auto backend = ft_lut::Backend::offline_fixture(options);
    backend->add_layer(ordinary.input); backend->add_layer(upgrade.input);
    auto receipt = backend->receipt(); verify_policy_receipts(receipt);
    int cases = 0; double worst = 0;
    for (const Fixture* f : {&ordinary, &upgrade}) { layout(*backend, *f); numeric(*backend, *f, activation == "i16" ? lutbench::Path::I16 : lutbench::Path::Q8, cases, worst); }
    std::cout << "{\"scope\":\"physical NUMA small-fixture loader probe; no full bank or real quality admission\",\"placement_verified\":"
              << (receipt.placement_verified ? "true" : "false") << ",\"cases\":" << cases << ",\"worst_relative_l2\":" << worst << ",\"controls\":[";
    for (size_t i = 0; i < receipt.controls.size(); ++i) { if (i) std::cout << ','; const auto& c = receipt.controls[i];
        std::cout << "{\"node\":" << c.node << ",\"verified\":" << (c.preferred_policy ? "true" : "false") << ",\"mode\":" << c.policy_mode << ",\"mask_nodes\":"; integers(c.policy_nodes);
        std::cout << ",\"policy_errno\":" << c.policy_errno << ",\"query_errno\":" << c.query_errno << ",\"observed_node\":" << c.observed_node << ",\"cpus\":"; integers(c.cpus);
        std::cout << ",\"threads\":[";
        for (size_t n = 0; n < c.thread_policies.size(); ++n) { if (n) std::cout << ','; const auto& p = c.thread_policies[n]; std::cout << "{\"cpu\":" << p.cpu << ",\"mode\":" << p.mode << ",\"verified\":" << (p.verified ? "true" : "false") << ",\"errno\":" << p.error << ",\"mask_nodes\":"; integers(p.nodes); std::cout << '}'; }
        std::cout << "]}";
    }
    std::cout << "],\"regions\":[";
    for (size_t n = 0; n < receipt.regions.size(); ++n) { if (n) std::cout << ','; const auto& r = receipt.regions[n];
        std::cout << "{\"layer\":" << r.model_layer << ",\"group\":" << r.group << ",\"matrix\":" << r.matrix << ",\"node\":" << r.node << ",\"mode\":" << r.policy_mode << ",\"mask_nodes\":"; integers(r.policy_nodes);
        std::cout << ",\"verified\":" << (r.preferred_policy ? "true" : "false") << ",\"policy_errno\":" << r.policy_errno << ",\"query_errno\":" << r.query_errno << ",\"sampled_pages\":" << r.sampled_pages << ",\"off_node_pages\":" << r.off_node_pages << ",\"page_errors\":" << r.page_errors << '}';
    }
    std::cout << "],\"scratch\":[";
    for (size_t n = 0; n < receipt.scratch.size(); ++n) { if (n) std::cout << ','; const auto& r = receipt.scratch[n]; std::cout << "{\"node\":" << r.node << ",\"query_errno\":" << r.query_errno << ",\"sampled_pages\":" << r.sampled_pages << ",\"off_node_pages\":" << r.off_node_pages << ",\"page_errors\":" << r.page_errors << '}'; }
    std::cout << "]}\n";
    idle_measure(*backend, ordinary);
    return receipt.placement_verified ? 0 : 2;
}
} // namespace

int main(int argc, char** argv) {
    try {
        if (argc > 1 && std::string(argv[1]) == "--numa-probe") {
            check(argc == 5, "usage: --numa-probe CPU_CSV FIXTURE_DIRECTORY i16|q8"); return numa_probe(argv[2], argv[3], argv[4]);
        }
        if (argc > 1 && std::string(argv[1]) == "--grouped-timing") return grouped_timing(argc, argv);
        mapping_allocator_checks();
        std::string real_directory;
        if (argc > 1) { check(argc == 3 && std::string(argv[1]) == "--fixtures", "usage: [--fixtures FIXTURE_DIRECTORY]"); real_directory = argv[2]; }
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
        merged_overflow_checks(workers);
        ft_lut::Options rejected_bank; rejected_bank.cpus = workers; rejected_bank.virtual_tp2 = true;
        rejects([&] { ft_lut::Backend::staged_bank(rejected_bank, bank); });
        rejects([&] { ft_lut::Backend::staged_bank(rejected_bank, broken); });
        Fixture ordinary(temp, 3, 8), upgrade(temp, 11, 8), down6(temp, 12, 8);
        grouped_checks(workers, ordinary, upgrade);
        grouped_boundary_checks(workers);
        grouped_numeric_failure_checks(workers);
        int cases = 0; double worst = 0;
        // The diagnostics cannot enter production, even before a bank exists.
        // Verify the specific early rejection instead of relying on empty-bank
        // or workstation topology errors to hide an admitted diagnostic mode.
        for (const std::string pages : {"base", "eager"}) {
            ft_lut::Options option; option.fixture_weight_pages = pages;
            rejects_with([&] { ft_lut::Backend::staged_bank(option, {}); }, "production LUT rejects offline diagnostic options");
        }
        for (int spin : {-1, 0, 1, 1000000}) {
            ft_lut::Options option; option.fixture_spin_us = spin;
            rejects_with([&] { ft_lut::Backend::staged_bank(option, {}); }, "production LUT rejects offline diagnostic options");
        }
        { ft_lut::Options option; option.fixture_wake = "sleep";
          rejects_with([&] { ft_lut::Backend::staged_bank(option, {}); }, "production LUT rejects offline diagnostic options"); }
        { ft_lut::Options option; option.fixture_nap_us = 20;
          rejects_with([&] { ft_lut::Backend::staged_bank(option, {}); }, "production LUT rejects offline diagnostic options"); }
        { ft_lut::Options option; option.fixture_idle_after_drain = false;
          rejects_with([&] { ft_lut::Backend::staged_bank(option, {}); }, "production LUT rejects offline diagnostic options"); }
#ifdef FT_LUT_DIAG_TRACE
        rejects_with([&] { ft_lut::Backend::staged_bank(ft_lut::Options{}, {}); }, "instrumented LUT builds are offline only");
#endif
        ft_lut::Options diagnostic; diagnostic.cpus = workers; diagnostic.virtual_tp2 = true;
        diagnostic.fixture_allow_smt = true;
        rejects_with([&] { ft_lut::Backend::staged_bank(diagnostic, {}); }, "SMT diagnostics require an offline virtual TP2 fixture");
        diagnostic.virtual_tp2 = false;
        rejects_with([&] { ft_lut::Backend::offline_fixture(diagnostic); }, "SMT diagnostics require an offline virtual TP2 fixture");
        diagnostic.virtual_tp2 = true; diagnostic.fixture_allow_smt = false;
        diagnostic.fixture_weight_pages = "invalid";
        rejects_with([&] { ft_lut::Backend::offline_fixture(diagnostic); }, "invalid offline LUT weight page policy");
        diagnostic.fixture_weight_pages = "deferred";
        diagnostic.fixture_wake = "invalid";
        rejects_with([&] { ft_lut::Backend::offline_fixture(diagnostic); }, "invalid offline LUT wake policy");
        diagnostic.fixture_wake = "sleep";
        for (int nap : {0, 1000001}) {
            diagnostic.fixture_nap_us = nap;
            rejects_with([&] { ft_lut::Backend::offline_fixture(diagnostic); }, "offline LUT nap duration");
        }
        diagnostic.fixture_nap_us = 20;
        for (int spin : {-2, 1000001}) {
            diagnostic.fixture_spin_us = spin;
            rejects_with([&] { ft_lut::Backend::offline_fixture(diagnostic); }, "offline LUT spin duration");
        }
        const auto siblings = allowed_siblings();
        if (!siblings.empty()) {
            diagnostic.fixture_spin_us = 2000; diagnostic.cpus = siblings;
            rejects_with([&] { ft_lut::Backend::offline_fixture(diagnostic); }, "distinct physical cores");
            diagnostic.fixture_allow_smt = true;
            auto smt = ft_lut::Backend::offline_fixture(diagnostic); smt->add_layer(ordinary.input);
            check(smt->receipt().virtual_tp2 && !smt->receipt().placement_verified, "SMT diagnostic claimed physical placement");
            layout(*smt, ordinary);
            diagnostic.cpus = {siblings[0], siblings[0]};
            rejects_with([&] { ft_lut::Backend::offline_fixture(diagnostic); }, "CPU is repeated");
        }
        // Page and wake policies must leave packed slices and arithmetic exact.
        // No performance assertion is made here: THP advice is not a guarantee.
        for (const std::string activation : {"i16", "q8"}) {
            std::vector<float> reference, input(ordinary.h, .25f), output(ordinary.h);
            const ft_lut::Routes route{{{0, .25f}, {1, .75f}, {2, .5f}}};
            for (const std::string pages : {"deferred", "base", "eager"}) for (int spin : {2000, -1, 0}) for (const std::string wake : {"sleep", "futex"}) {
                ft_lut::Options option; option.cpus = workers; option.virtual_tp2 = true; option.activation = activation;
                option.fixture_weight_pages = pages; option.fixture_spin_us = spin;
                option.fixture_wake = wake; option.initial_generation = uint64_t(UINT32_MAX) - 26;
                option.fixture_idle_after_drain = wake == "futex";
                if (wake == "futex" && spin == 0) option.fixture_nap_us = 1000;
                auto probe = ft_lut::Backend::offline_fixture(option);
                check(probe->fixture_arenas().empty(), "empty fixture has a weight arena");
                probe->add_layer(ordinary.input); layout(*probe, ordinary); verify_policy_receipts(probe->receipt());
                const auto arenas = probe->fixture_arenas(); check(arenas.size() == 6, "fixture arena diagnostic omitted a projection");
                uint64_t payload = 0; const auto page = uintptr_t(sysconf(_SC_PAGESIZE));
                for (const auto& arena : arenas) {
                    check(arena.address && arena.address % page == 0 && arena.mapped_bytes % page == 0 &&
                          arena.payload_bytes <= arena.mapped_bytes && arena.mapped_bytes - arena.payload_bytes < page,
                          "fixture arena address or extent is invalid");
                    check(arena.model_layer == 3 && arena.group >= 0 && arena.group < 2 && arena.matrix >= 0 && arena.matrix < 3 &&
                          arena.page_policy == pages, "fixture arena labels are invalid");
                    payload += arena.payload_bytes;
                }
                const auto receipt = probe->receipt();
                check(payload == receipt.payload_bytes[0] + receipt.payload_bytes[1], "fixture arena diagnostic payload ledger mismatch");
                if (wake == "futex" && spin >= 0) {
                    const auto before = probe->receipt().pool_naps;
                    const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
                    std::this_thread::sleep_for(std::chrono::milliseconds(3));
                    while (true) {
                        const auto after = probe->receipt().pool_naps;
                        if (after[0] > before[0] && after[1] > before[1]) break;
                        check(std::chrono::steady_clock::now() < deadline, "futex test did not exercise parked workers");
                        std::this_thread::sleep_for(std::chrono::milliseconds(1));
                    }
                }
                probe->forward(0, input.data(), 1, route, output.data());
#ifdef FT_LUT_DIAG_TRACE
                const auto traces = probe->fixture_trace(); check(traces.size() == workers.size(), "worker trace inventory mismatch");
                for (const auto& trace : traces) {
                    check(trace.publication_ns && trace.wake_ns >= trace.publication_ns && trace.completion_ns >= trace.wake_ns,
                          "worker trace timestamps are not ordered");
                    uint64_t elapsed = 0;
                    for (auto duration : trace.phase_ns) elapsed += duration;
                    for (auto duration : trace.barrier_ns) elapsed += duration;
                    check(elapsed <= trace.completion_ns - trace.wake_ns, "worker phase trace exceeds whole job interval");
                }
#endif
                if (reference.empty()) reference = output;
                else check(std::memcmp(reference.data(), output.data(), output.size() * sizeof(float)) == 0, "offline diagnostic changed arithmetic bits");
                if (spin == -1) check(probe->receipt().pool_naps == std::array<uint64_t, 2>{0, 0}, "spin-only fixture entered nap backoff");
                if (wake == "futex") {
                    // Loading publishes25 generations, so these calls cross
                    // the separate32-bit futex wrap without reusing a64-bit job.
                    for (int repeat = 0; repeat < 3; ++repeat) {
                        probe->forward(0, input.data(), 1, route, output.data());
                        check(std::memcmp(reference.data(), output.data(), output.size() * sizeof(float)) == 0, "futex generation wrap changed output");
                    }
                    if (pages == "base" && spin == 0) {
                        probe->inject_failure(1, 1);
                        rejects([&] { probe->forward(0, input.data(), 1, route, output.data()); });
                        probe->inject_failure(-1); probe->fixture_set_timeout(2); probe->inject_delay(0, 1, 10);
                        rejects([&] { probe->forward(0, input.data(), 1, route, output.data()); });
                        probe->fixture_set_timeout(30000); probe->inject_delay(-1, 1, 0);
                        probe->forward(0, input.data(), 1, route, output.data());
                        check(std::memcmp(reference.data(), output.data(), output.size() * sizeof(float)) == 0, "futex pool failed recovery after drain");
                        probe->request_abort();
                        rejects([&] { probe->forward(0, input.data(), 1, route, output.data()); });
                        probe->shutdown();
                    }
                }
            }
        }
        {
            ft_lut::Options option; option.cpus = workers; option.virtual_tp2 = true;
            option.fixture_wake = "futex"; option.fixture_spin_us = 0; option.fixture_idle_after_drain = true;
            option.initial_generation = UINT64_MAX - 26;
            auto probe = ft_lut::Backend::offline_fixture(option); probe->add_layer(ordinary.input);
            probe->inject_delay(3, 1, 500);
            std::vector<float> input(ordinary.h, .25f), output(ordinary.h);
            const ft_lut::Routes route{{{0, 1.f}}};
            std::atomic<bool> returned{false}; std::exception_ptr failure;
            std::thread forward([&] {
                try { probe->forward(0, input.data(), 1, route, output.data()); } catch (...) { failure = std::current_exception(); }
                returned.store(true, std::memory_order_release);
            });
            bool witnessed = false; uint64_t delayed_epoch = 0;
            const auto witness_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
            while (!returned.load(std::memory_order_acquire) && std::chrono::steady_clock::now() < witness_deadline) {
                const auto generations = probe->fixture_generations();
                if (generations[0].epoch == generations[1].epoch &&
                    generations[0].completed == generations[0].epoch &&
                    generations[1].completed != generations[1].epoch) {
                    witnessed = true; delayed_epoch = generations[0].epoch; break;
                }
                std::this_thread::yield();
            }
            const auto before = probe->receipt().pool_naps;
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            const auto after = probe->receipt().pool_naps;
            const auto generations = probe->fixture_generations();
            const bool active = generations[0].epoch == delayed_epoch && generations[1].epoch == delayed_epoch &&
                                generations[0].completed == delayed_epoch && generations[1].completed != delayed_epoch;
            forward.join();
            if (failure) std::rethrow_exception(failure);
            check(witnessed && active && before[0] == after[0], "completed group entered nap before whole generation retired");
            probe->inject_delay(-1, 1, 0);
            probe->forward(0, input.data(), 1, route, output.data()); //64-bit generation wraps after delayed retirement.
            probe->shutdown(); probe->shutdown();
        }
        // Recreate the cost-table lifetime pattern: independent successive
        // backends must own and release their scratch mappings for every recipe.
        for (const std::string activation : {"i16", "q8"}) {
            std::array<uint64_t, 2> mapped{};
            for (int repeat = 0; repeat < 2; ++repeat) for (const Fixture* fixture : {&ordinary, &upgrade, &down6}) {
                ft_lut::Options options; options.cpus = workers; options.virtual_tp2 = true; options.activation = activation;
                auto fresh = ft_lut::Backend::offline_fixture(options); fresh->add_layer(fixture->input);
                const auto receipt = fresh->receipt(); verify_policy_receipts(receipt);
                if (!mapped[0]) mapped = receipt.scratch_mapped_bytes;
                check(receipt.scratch_mapped_bytes == mapped, "scratch footprint changed across recipe/backend resets");
                numeric(*fresh, *fixture, activation == "i16" ? lutbench::Path::I16 : lutbench::Path::Q8, cases, worst);
            }
        }
        for (const std::string activation : {"i16", "q8"}) {
            ft_lut::Options options; options.cpus = workers; options.virtual_tp2 = true; options.activation = activation;
            options.initial_generation = std::numeric_limits<uint64_t>::max() - 1;
            auto backend = ft_lut::Backend::offline_fixture(options);
            for (const Fixture* fixture : {&ordinary, &upgrade, &down6}) {
                backend->add_layer(fixture->input); layout(*backend, *fixture);
                numeric(*backend, *fixture, activation == "i16" ? lutbench::Path::I16 : lutbench::Path::Q8, cases, worst);
            }
            auto receipt = backend->receipt();
            verify_policy_receipts(receipt);
            check(receipt.workers == std::array<int, 2>{2, 3}, "asymmetric 2/3 groups were not exercised");
            check(receipt.offline_fixture && receipt.virtual_tp2 && !receipt.placement_verified, "virtual geometry claimed NUMA placement");
            check(receipt.regions.size() == 18 && receipt.payload_bytes[0] == receipt.payload_bytes[1] && receipt.staging_peak_bytes <= 64 * 1024 * 1024,
                  "loader receipt geometry or bounded staging is incorrect");
            for (const auto& region : receipt.regions) check(region.sampled_pages == 32, "placement did not sample every expert projection");
            std::vector<float> x(4 * ordinary.h, .25f), out(x.size(), 123.f);
            ft_lut::Routes route(4); route[0] = {{0, .4f}, {1, .6f}}; route[2] = {{0, .25f}}; route[3] = {{1, .9f}};
            backend->forward(0, x.data(), 4, route, out.data());
            const auto before_flush = out;
            check(backend->fixture_flush_weights() == receipt.payload_bytes[0] + receipt.payload_bytes[1], "weight flush extent differs from owned payload");
            backend->forward(0, x.data(), 4, route, out.data());
            check(out == before_flush, "owned-weight flushing changed numerical output");
            // A flush owns the same mutex as forward; a concurrent replay must
            // either execute with exclusive access or reject without touching it.
            std::exception_ptr flush_error, forward_error;
            std::thread flusher([&] { try { for (int n = 0; n < 20; ++n) backend->fixture_flush_weights(); } catch (...) { flush_error = std::current_exception(); } });
            try {
                for (int n = 0; n < 20; ++n) {
                    try { backend->forward(0, x.data(), 4, route, out.data()); check(out == before_flush, "concurrent flush changed replay output"); }
                    catch (const std::runtime_error& error) {
                        if (std::string(error.what()).find("one in-flight coordinator") == std::string::npos) throw;
                    }
                }
            } catch (...) { forward_error = std::current_exception(); }
            flusher.join();
            if (flush_error) std::rethrow_exception(flush_error);
            if (forward_error) std::rethrow_exception(forward_error);
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
            for (float invalid_value : {std::numeric_limits<float>::infinity(), -std::numeric_limits<float>::infinity(),
                                        std::numeric_limits<float>::quiet_NaN()}) {
                for (size_t position : {size_t(0), size_t(7), size_t(8), x.size() - 1}) {
                    std::fill(x.begin(), x.end(), .25f); std::fill(out.begin(), out.end(), 123.f);
                    x[position] = invalid_value;
                    auto invalid_route = route; invalid_route[0][0].first = ordinary.experts;
                    rejects_with([&] { backend->forward(0, x.data(), 4, invalid_route, out.data()); }, "nonfinite LUT input");
                    check(std::all_of(out.begin(), out.end(), [](float v) { return v == 123.f; }), "invalid input published caller output");
                }
            }
            rejects([&] { backend->forward(0, x.data(), 5, route, out.data()); });
            rejects([&] { backend->add_layer(ordinary.input); });
            std::fill(x.begin(), x.end(), .25f); std::fill(out.begin(), out.end(), 123.f);
            backend->fixture_set_timeout(5); backend->inject_delay(0, 1, 30);
            rejects([&] { backend->forward(0, x.data(), 4, route, out.data()); });
            check(std::all_of(out.begin(), out.end(), [](float v) { return v == 123.f; }), "timeout published a partial result");
            check(backend->receipt().timed_out_jobs == 1, "timeout was not recorded");
            backend->inject_delay(-1, 1, 0); backend->fixture_set_timeout(30000);
            backend->forward(0, x.data(), 4, route, out.data());
            check(relative(out, reference(ordinary, x, route, activation == "i16" ? lutbench::Path::I16 : lutbench::Path::Q8)) <= 1e-5, "pool did not drain and recover after timeout");
            // A late fallback receives the original generation's remaining
            // budget, rather than resetting a full per-call timeout.
            std::fill(out.begin(), out.end(), 123.f); backend->inject_delay(0, 1, 30);
            rejects([&] { backend->forward(0, x.data(), 4, route, out.data(), 5); });
            check(backend->receipt().timed_out_jobs == 2, "remaining-budget deadline was not recorded");
            check(std::all_of(out.begin(), out.end(), [](float value) { return value == 123.f; }), "remaining-budget deadline published output");
            backend->inject_delay(-1, 1, 0); backend->forward(0, x.data(), 4, route, out.data());
            std::atomic<bool> stop_stats{false}; std::thread stats([&] { while (!stop_stats.load()) (void)backend->receipt(); });
            try { for (int n = 0; n < 100; ++n) backend->forward(0, x.data(), 4, route, out.data()); }
            catch (...) { stop_stats.store(true); stats.join(); throw; }
            stop_stats.store(true); stats.join();
        }
        // Cancellation is sticky for serving: drain both asymmetric groups,
        // leave the caller result untouched and refuse subsequent generations.
        for (const std::string wake : {"sleep", "futex"}) for (int cancellation : {0, 1, 2}) {
            int64_t abort_word = 0;
            ft_lut::Options cancel_options; cancel_options.cpus = workers; cancel_options.virtual_tp2 = true;
            cancel_options.fixture_wake = wake;
            if (wake == "futex") cancel_options.fixture_spin_us = 0;
            if (cancellation == 1) cancel_options.abort_flag = &abort_word;
            auto cancelled = ft_lut::Backend::offline_fixture(cancel_options); cancelled->add_layer(ordinary.input);
            cancelled->inject_delay(0, 1, 40);
            std::vector<float> x(ordinary.h, .25f), out(ordinary.h, 123.f);
            ft_lut::Routes routes(1); routes[0] = {{0, 1.f}};
            std::this_thread::sleep_for(std::chrono::milliseconds(3));
            std::thread aborter([&] {
                std::this_thread::sleep_for(std::chrono::milliseconds(8));
                if (cancellation == 1) __atomic_store_n(&abort_word, int64_t(1), __ATOMIC_RELEASE);
                else if (cancellation == 2) cancelled->shutdown();
                else cancelled->request_abort();
            });
            const auto started = std::chrono::steady_clock::now();
            rejects([&] { cancelled->forward(0, x.data(), 1, routes, out.data()); });
            aborter.join();
            check(std::chrono::steady_clock::now() - started >= std::chrono::milliseconds(30), "cancellation returned before the delayed pool drained");
            check(std::all_of(out.begin(), out.end(), [](float value) { return value == 123.f; }), "cancellation published partial output");
            rejects([&] { cancelled->forward(0, x.data(), 1, routes, out.data()); });
            cancelled->shutdown(); cancelled->shutdown();
        }
        Temporary production_temp;
        std::unique_ptr<Fixture> production = real_directory.empty() ? std::make_unique<Fixture>(production_temp, 3, 1, 4096, 2048) : std::make_unique<Fixture>(real_directory + "/blk.03.experts-0-1.lut.gguf", 3);
        std::unique_ptr<Fixture> real_upgrade;
        if (!real_directory.empty()) real_upgrade = std::make_unique<Fixture>(real_directory + "/blk.11.experts-0-1.lut.gguf", 11);
        // Production-width loading copies every arena before deferred THP
        // advice. Packed bytes must remain identical; advice proves neither
        // huge-page promotion nor locality (the latter has separate receipts).
        for (const std::string activation : {"i16", "q8"}) {
            ft_lut::Options production_options; production_options.cpus = workers; production_options.virtual_tp2 = true; production_options.activation = activation; production_options.act_limit = 1.5f;
            auto full_width = ft_lut::Backend::offline_fixture(production_options); full_width->add_layer(production->input);
            layout(*full_width, *production); numeric(*full_width, *production, activation == "i16" ? lutbench::Path::I16 : lutbench::Path::Q8, cases, worst, 1.5f);
            const auto full_receipt = full_width->receipt(); verify_policy_receipts(full_receipt);
            check(full_receipt.act_limit == 1.5f, "model activation limit not recorded");
            if (sysconf(_SC_PAGESIZE) == 4096) {
                const uint64_t expected = activation == "i16" ? 4849664 : 26411008;
                check(full_receipt.scratch_mapped_bytes == std::array<uint64_t, 2>{expected, expected}, "production scratch mapped ledger differs from page-rounded recipe");
                if (activation == "q8")
                    check(full_receipt.scratch_bytes == std::array<uint64_t, 2>{22722560, 22722560}, "production Q8 scratch payload ledger differs from recipe");
            }
            if (real_upgrade) { full_width->add_layer(real_upgrade->input); layout(*full_width, *real_upgrade); numeric(*full_width, *real_upgrade, activation == "i16" ? lutbench::Path::I16 : lutbench::Path::Q8, cases, worst, 1.5f); }
            if (activation == "i16") idle_measure(*full_width, *production);
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
        options.cpus = workers;
        for (float limit : {0.f, -1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()}) { options.act_limit = limit; rejects([&] { ft_lut::Backend::offline_fixture(options); }); }
        std::cout << "PASS cases=" << cases << " worst_relative_l2=" << worst
                  << " paths=i16,q8 M=1,2,3,4 K=0..8 profiles=ordinary,upgrade,down6"
                  << " worker_failures=16 timeouts=4 cancellations=6 owned_weight_flush=covered stats_concurrency=200 groups=2/3 production_width=4096/2048 scratch_unmap=covered repeated_profile_resets=12 placement_claim=none generation_wrap=covered offline_page_spin_wake_bits=36 diagnostic_admission=covered futex_wrap_drain_abort=covered\n";
        return 0;
    } catch (const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
