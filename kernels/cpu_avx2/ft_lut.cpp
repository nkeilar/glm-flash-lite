// NUMA TP=2 expert backend; adapted from the separately attributed LUT benchmark.
#include "ft_lut.h"
#include "lut/kernels.h"
#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <cmath>
#include <cstring>
#include <exception>
#include <fcntl.h>
#include <filesystem>
#include <fstream>
#include <limits>
#include <linux/mempolicy.h>
#include <linux/futex.h>
#include <map>
#include <mutex>
#include <new>
#include <pthread.h>
#include <sched.h>
#include <set>
#include <sstream>
#include <stdexcept>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/syscall.h>
#include <thread>
#include <type_traits>
#include <unistd.h>

namespace ft_lut {
namespace {
using lutbench::Activation;
using lutbench::Format;
using lutbench::Path;
constexpr uint64_t FIXTURE_LIMIT = 256ULL << 20;
constexpr uint64_t STAGING_LIMIT = 64ULL << 20;
constexpr int MAX_TOKENS = 4, MAX_PICKS = 32;
constexpr uint64_t HOST_FLOOR = 100ULL << 30, NODE_FLOOR = 8ULL << 30;
constexpr uint64_t HOST_OVERHEAD = 1ULL << 30, NODE_OVERHEAD = 512ULL << 20;
void require(bool valid, const char* message) { if (!valid) throw std::runtime_error(message); }
size_t mul(size_t a, size_t b) {
    require(!b || a <= std::numeric_limits<size_t>::max() / b, "LUT size overflow");
    return a * b;
}
FileIdentity identity(int fd) {
    struct stat st{};
    require(fstat(fd, &st) == 0 && S_ISREG(st.st_mode), "LUT input must be a regular file");
    require(st.st_size >= 0, "LUT input has negative size");
    return {uint64_t(st.st_dev), uint64_t(st.st_ino), uint64_t(st.st_size),
            int64_t(st.st_mtim.tv_sec) * 1000000000 + st.st_mtim.tv_nsec,
            int64_t(st.st_ctim.tv_sec) * 1000000000 + st.st_ctim.tv_nsec};
}
struct File {
    int fd;
    explicit File(const std::string& path) : fd(open(path.c_str(), O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK)) {
        require(fd >= 0, "cannot open regular LUT input without following symlinks");
        try { (void)identity(fd); } catch (...) { close(fd); throw; }
    }
    ~File() { close(fd); }
    void read(uint64_t offset, void* target, size_t bytes) {
        require(offset <= uint64_t(std::numeric_limits<off_t>::max()) &&
                bytes <= uint64_t(std::numeric_limits<off_t>::max()) - offset, "LUT file offset overflow");
        auto* out = static_cast<uint8_t*>(target);
        while (bytes) {
            ssize_t count = pread(fd, out, bytes, off_t(offset));
            if (count < 0 && errno == EINTR) continue;
            require(count > 0, "short LUT input while streaming rows");
            out += count; offset += size_t(count); bytes -= size_t(count);
        }
    }
};
Format format(int type) {
    if (type == 138) return Format::IQ3_K;
    if (type == 144) return Format::IQ4_KS;
    if (type == 14) return Format::Q6_K;
    throw std::runtime_error("unsupported LUT routed weight type");
}
int cpu_node(int cpu) {
    for (int node = 0; node < 1024; ++node) {
        std::string name = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/node" + std::to_string(node);
        if (access(name.c_str(), F_OK) == 0) return node;
    }
    throw std::runtime_error("cannot discover worker NUMA node");
}
uint64_t host_available() {
    std::ifstream file("/proc/meminfo"); std::string key, line; uint64_t kib = 0;
    while (std::getline(file, line)) { std::istringstream row(line); row >> key; if (key == "MemAvailable:") { require(bool(row >> kib) && kib <= std::numeric_limits<uint64_t>::max() / 1024, "invalid host memory telemetry"); return kib * 1024; } }
    throw std::runtime_error("host MemAvailable telemetry missing");
}
uint64_t node_free(int node) {
    std::ifstream file("/sys/devices/system/node/node" + std::to_string(node) + "/meminfo"); std::string line;
    while (std::getline(file, line)) {
        std::istringstream row(line); std::string label, key; int observed; uint64_t kib;
        if (row >> label >> observed >> key >> kib && label == "Node" && observed == node && key == "MemFree:") {
            require(kib <= std::numeric_limits<uint64_t>::max() / 1024, "invalid node memory telemetry"); return kib * 1024;
        }
    }
    throw std::runtime_error("per-node MemFree telemetry missing");
}
uint64_t unsigned_value(const std::string& value) {
    require(!value.empty() && value.find_first_not_of("0123456789") == std::string::npos, "invalid memory-limit telemetry");
    size_t consumed = 0; unsigned long long result = std::stoull(value, &consumed);
    require(consumed == value.size(), "invalid memory-limit telemetry"); return uint64_t(result);
}
void cgroup_headroom(uint64_t additional) {
    std::ifstream membership("/proc/self/cgroup"); std::string line, relative;
    while (std::getline(membership, line)) if (line.rfind("0::", 0) == 0) relative = line.substr(3);
    require(!relative.empty() && relative[0] == '/' && relative.find("..") == std::string::npos,
            "full LUT loader requires identifiable cgroup-v2 memory limits");
    const std::filesystem::path root("/sys/fs/cgroup");
    std::filesystem::path current = relative == "/" ? root : root / relative.substr(1);
    require(std::filesystem::is_directory(current), "cannot identify current cgroup-v2 memory controller");
    while (true) {
        std::ifstream limit_file(current / "memory.max"); std::string value;
        if (limit_file >> value) {
            if (value != "max") {
                std::ifstream used_file(current / "memory.current"); std::string used;
                require(bool(used_file >> used), "finite cgroup memory usage is unavailable");
                check_memory_limit(unsigned_value(value), unsigned_value(used), additional);
            }
        } else require(current == root, "cgroup memory-limit telemetry is unavailable");
        if (current == root) break;
        current = current.parent_path();
        require(!current.empty() && current.string().rfind(root.string(), 0) == 0, "cgroup hierarchy escapes controller root");
    }
}
std::string physical_core(int cpu) {
    std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/topology/";
    std::string package, core;
    require(bool(std::ifstream(base + "physical_package_id") >> package) &&
            bool(std::ifstream(base + "core_id") >> core), "cannot discover physical worker core");
    return package + ":" + core;
}
constexpr size_t POLICY_NODES = 1024, WORD_BITS = 8 * sizeof(unsigned long);
// Linux get_nodes() decrements maxnode before reading the mask. Include the
// sentinel bit and storage for either syscall's rounded mask length.
constexpr unsigned long POLICY_MAXNODE = POLICY_NODES + 1;
using NodeMask = std::array<unsigned long, (POLICY_MAXNODE + WORD_BITS - 1) / WORD_BITS>;
struct Policy {
    bool verified = false;
    int mode = -1, error = 0;
    NodeMask mask{};
};
std::vector<int> policy_nodes(const Policy& policy) {
    std::vector<int> nodes;
    for (size_t n = 0; n < POLICY_MAXNODE; ++n)
        if (policy.mask[n / WORD_BITS] & (1UL << (n % WORD_BITS))) nodes.push_back(int(n));
    return nodes;
}
Policy read_policy(int node, void* address, bool by_address) {
    Policy policy;
    if (syscall(SYS_get_mempolicy, &policy.mode, policy.mask.data(), POLICY_MAXNODE,
                address, by_address ? MPOL_F_ADDR : 0) < 0) { policy.error = errno; return policy; }
    NodeMask wanted{}; wanted[size_t(node) / WORD_BITS] = 1UL << (node % WORD_BITS);
    policy.verified = policy.mode == MPOL_PREFERRED && policy.mask == wanted;
    if (!policy.verified) policy.error = EPROTO;
    return policy;
}
Policy preferred(void* ptr, size_t bytes, int node) {
    require(node >= 0 && node < 1024, "NUMA node exceeds bounded mask");
    NodeMask mask{}; mask[size_t(node) / WORD_BITS] = 1UL << (node % WORD_BITS);
    if (syscall(SYS_mbind, ptr, bytes, MPOL_PREFERRED, mask.data(), POLICY_MAXNODE, 0) < 0) {
        Policy policy; policy.error = errno; return policy;
    }
    return read_policy(node, ptr, true);
}
Policy thread_preferred(int node) {
    require(node >= 0 && node < 1024, "NUMA node exceeds bounded mask");
    NodeMask mask{}; mask[size_t(node) / WORD_BITS] = 1UL << (node % WORD_BITS);
    // A preferred policy may fall back. Weight/control placement is measured,
    // and failure is reported rather than converted to strict binding.
    if (syscall(SYS_set_mempolicy, MPOL_PREFERRED, mask.data(), POLICY_MAXNODE) < 0) {
        Policy policy; policy.error = errno; return policy;
    }
    return read_policy(node, nullptr, false);
}
struct Arena {
    uint8_t* data = nullptr;
    size_t bytes = 0, mapped = 0;
    Policy policy;
    std::string page_policy;
    Arena(size_t n, int node, std::string pages = "base") : bytes(n), page_policy(std::move(pages)) {
        const size_t page = size_t(sysconf(_SC_PAGESIZE));
        require(n && n <= std::numeric_limits<size_t>::max() - page, "invalid LUT arena size");
        mapped = (n + page - 1) / page * page;
        void* pointer = mmap(nullptr, mapped, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        require(pointer != MAP_FAILED, "LUT arena mmap failed");
        data = static_cast<uint8_t*>(pointer);
        // Default first touch uses ordinary pages: preferred THP allocation can
        // fall back to a remote node when local high-order blocks are scarce.
        // Do not change any global THP setting or impose a strict memory bind.
        // Eager THP is an offline weight-only experiment. Establish and read
        // back the VMA policy before any owner-worker first touch. The hint
        // guarantees neither huge pages nor locality; receipts still measure
        // placement and offline smaps observations measure actual page size.
        if (page_policy == "eager") policy = preferred(data, mapped, node);
        if (madvise(data, mapped, page_policy == "eager" ? MADV_HUGEPAGE : MADV_NOHUGEPAGE) != 0) {
            munmap(data, mapped); data = nullptr;
            throw std::runtime_error("cannot establish LUT first-touch page advice");
        }
        if (page_policy != "eager") policy = preferred(data, mapped, node);
    }
    void hint_after_touch(int node) {
        // All payload pages have already been touched by their owning workers.
        // This is only a deferred promotion hint, not a page-size or throughput
        // promise. No forced collapse/migration is requested. Linux khugepaged
        // selects collapse nodes from the existing pages; residency is still
        // measured below rather than inferred from advice or policy success.
        if (page_policy == "deferred") (void)madvise(data, mapped, MADV_HUGEPAGE);
        policy = read_policy(node, data, true);
    }
    ~Arena() { if (data) munmap(data, mapped); }
    Arena(const Arena&) = delete;
};
const Options& checked_options(const Options& options, bool fixture) {
    require(options.fixture_weight_pages == "deferred" || options.fixture_weight_pages == "base" || options.fixture_weight_pages == "eager",
            "invalid offline LUT weight page policy");
    require(options.fixture_spin_us >= -1 && options.fixture_spin_us <= 1000000, "offline LUT spin duration must be -1..1000000 microseconds");
    require(options.fixture_wake == "sleep" || options.fixture_wake == "futex", "invalid offline LUT wake policy");
    require(options.fixture_nap_us >= 1 && options.fixture_nap_us <= 1000000, "offline LUT nap duration must be1..1000000 microseconds");
    require(!options.fixture_allow_smt || (fixture && options.virtual_tp2), "SMT diagnostics require an offline virtual TP2 fixture");
    require(fixture || (!options.fixture_allow_smt && options.fixture_weight_pages == "deferred" && options.fixture_spin_us == 50000 && options.fixture_wake == "futex" && options.fixture_nap_us == 1000 && options.fixture_idle_after_drain),
            "production LUT rejects offline diagnostic options");
#ifdef FT_LUT_DIAG_TRACE
    require(fixture, "instrumented LUT builds are offline only");
#endif
    return options;
}
#ifdef FT_LUT_DIAG_TRACE
uint64_t trace_clock_ns() {
    return uint64_t(std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count());
}
#endif
struct Barrier {
    std::atomic<unsigned> count{0}, epoch{0};
    unsigned n;
    explicit Barrier(unsigned size) : n(size) {}
    void wait() {
        unsigned seen = epoch.load(std::memory_order_acquire);
        if (count.fetch_add(1, std::memory_order_acq_rel) + 1 == n) {
            count.store(0, std::memory_order_relaxed);
            epoch.fetch_add(1, std::memory_order_release);
        } else while (epoch.load(std::memory_order_acquire) == seen) _mm_pause();
    }
};
struct alignas(64) Control {
    std::atomic<uint64_t> epoch;
    alignas(64) std::atomic<uint64_t> completed;
    alignas(64) std::atomic<unsigned> remaining{0};
    alignas(64) Barrier barrier;
    alignas(64) std::atomic<uint64_t> naps{0};
    // A dedicated32-bit futex word avoids aliasing the64-bit publication epoch.
    alignas(64) std::atomic<uint32_t> wake_generation;
    alignas(64) std::atomic<uint64_t> retired;
    Control(unsigned n, uint64_t initial) : epoch(initial), completed(initial), barrier(n), wake_generation(uint32_t(initial)), retired(initial) {}
};
static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t) && std::atomic<uint32_t>::is_always_lock_free,
              "LUT futex requires an aligned native32-bit atomic");
class Workers {
    struct Group {
        int node = -1, count = 0;
        std::unique_ptr<Arena> arena;
        Control* control = nullptr;
        std::atomic<bool> ready{false};
#ifdef FT_LUT_DIAG_TRACE
        uint64_t publication_ns = 0;
#endif
    };
    std::array<Group, 2> groups_;
    std::vector<std::thread> threads_;
    std::vector<int> group_, local_, cpus_;
    std::vector<Policy> thread_policies_;
    std::atomic<unsigned> ready_{0};
    std::atomic<bool> stop_{false};
    std::atomic<bool> abort_{false};
    std::atomic<bool> cancelled_{false};
    std::atomic<uint64_t> timeouts_{0};
    std::mutex error_mutex_;
    std::exception_ptr error_;
    void (*task_)(void*, int, int, int) = nullptr;
    void* context_ = nullptr;
    uint64_t submitted_;
    uint64_t timeout_ms_;
    const int64_t* external_abort_ = nullptr;
    bool futex_wake_ = false;
    bool idle_after_drain_ = false;
    std::atomic<int> wake_error_{0}, last_wake_errno_{0};
    std::atomic<uint64_t> wake_failures_{0};
#ifdef FT_LUT_DIAG_TRACE
    struct alignas(128) TraceSlot { FixtureWorkerTrace value; };
    std::array<std::array<TraceSlot, 128>, 2> traces_{};
#endif
public:
    bool virtual_tp2;
    explicit Workers(const Options& options) : submitted_(options.initial_generation), timeout_ms_(options.job_timeout_ms), external_abort_(options.abort_flag), futex_wake_(options.fixture_wake == "futex"), idle_after_drain_(options.fixture_idle_after_drain), virtual_tp2(options.virtual_tp2) {
        set_timeout(options.job_timeout_ms);
        require(options.cpus.size() >= 2 && options.cpus.size() <= 128, "LUT needs at least two bounded worker CPUs");
        cpu_set_t allowed; CPU_ZERO(&allowed);
        require(sched_getaffinity(0, sizeof(allowed), &allowed) == 0, "cannot read allowed worker CPUs");
        std::set<int> unique;
        std::set<std::string> cores;
        std::map<int, int> labels;
        for (size_t i = 0; i < options.cpus.size(); ++i) {
            int cpu = options.cpus[i];
            require(cpu >= 0 && cpu < CPU_SETSIZE && CPU_ISSET(cpu, &allowed) && unique.insert(cpu).second,
                    "LUT worker CPU is repeated or outside allowed affinity");
            const bool distinct_core = cores.insert(physical_core(cpu)).second;
            require(distinct_core || options.fixture_allow_smt, "LUT workers must use distinct physical cores");
            int node = cpu_node(cpu);
            int label = options.virtual_tp2 ? int(i >= options.cpus.size() / 2) : node;
            if (!labels.count(label)) {
                require(labels.size() < 2, "LUT supports exactly two NUMA groups");
                int g = int(labels.size()); labels[label] = g; groups_[g].node = node;
            }
            int g = labels.at(label);
            require(groups_[g].node == node, "a worker group spans physical NUMA nodes");
            group_.push_back(g); local_.push_back(groups_[g].count++);
        }
        require(labels.size() == 2 && groups_[0].count && groups_[1].count, "LUT requires exactly two nonempty worker groups");
        cpus_ = options.cpus; thread_policies_.resize(cpus_.size());
        for (auto& group : groups_) group.arena = std::make_unique<Arena>(sizeof(Control), group.node);
        try {
            for (size_t i = 0; i < options.cpus.size(); ++i) threads_.emplace_back([this, i, cpu = options.cpus[i], initial = options.initial_generation, spin_us = options.fixture_spin_us, nap_us = options.fixture_nap_us] {
                Group& group = groups_[group_[i]];
                try {
                    cpu_set_t affinity; CPU_ZERO(&affinity); CPU_SET(cpu, &affinity);
                    require(pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity) == 0, "LUT worker affinity failed");
                    thread_policies_[i] = thread_preferred(group.node);
                } catch (...) { capture_error(); }
                if (local_[i] == 0) {
                    group.control = new(group.arena->data) Control(unsigned(group.count), initial);
                    group.ready.store(true, std::memory_order_release);
                } else while (!group.ready.load(std::memory_order_acquire) && !stop_.load(std::memory_order_relaxed)) _mm_pause();
                ready_.fetch_add(1, std::memory_order_release);
                if (stop_.load(std::memory_order_relaxed)) return;
                uint64_t seen = initial;
                while (true) {
                    // Finishing this worker's rows is not whole-job idleness:
                    // another group may still be on the critical path. The
                    // coordinator retires both groups only after both release
                    // completions were acquired. New work/stop always bypasses
                    // this observation; it is not an additional pool barrier.
                    if (idle_after_drain_ && spin_us >= 0)
                        while (group.control->epoch.load(std::memory_order_acquire) == seen &&
                               group.control->retired.load(std::memory_order_acquire) != seen &&
                               !stop_.load(std::memory_order_relaxed)) _mm_pause();
                    unsigned polls = 0;
                    const auto spin_end = std::chrono::steady_clock::now() + std::chrono::microseconds(spin_us < 0 ? 0 : spin_us);
                    bool napping = false;
                    while (group.control->epoch.load(std::memory_order_acquire) == seen && !stop_.load(std::memory_order_relaxed)) {
                        if (!napping) {
                            _mm_pause();
                            if ((++polls & 255U) == 0 && spin_us >= 0) napping = std::chrono::steady_clock::now() >= spin_end;
                        } else {
                            group.control->naps.fetch_add(1, std::memory_order_relaxed);
                            if (futex_wake_) {
                                const uint32_t expected = group.control->wake_generation.load(std::memory_order_acquire);
                                // Publication between either load and WAIT changes the
                                // futex value, so the kernel returns EAGAIN instead of
                                // parking after a missed wake. The64-bit epoch remains
                                // authoritative, including across32-bit futex wrap.
                                if (group.control->epoch.load(std::memory_order_acquire) != seen || stop_.load(std::memory_order_relaxed)) continue;
                                const timespec nap{nap_us / 1000000, (nap_us % 1000000) * 1000};
                                const long rc = syscall(SYS_futex, &group.control->wake_generation, FUTEX_WAIT_PRIVATE, expected, &nap, nullptr, 0);
                                if (rc < 0 && errno != EAGAIN && errno != EINTR && errno != ETIMEDOUT) {
                                    // Preserve the worker and its drain responsibility.
                                    // Report the failed primitive after a complete job;
                                    // bounded fallback also lets constructor/stop join.
                                    try { throw std::runtime_error("LUT futex wait failed"); } catch (...) { capture_error(); }
                                    std::this_thread::sleep_for(std::chrono::microseconds(nap_us));
                                }
                            } else std::this_thread::sleep_for(std::chrono::microseconds(nap_us));
                        }
                    }
                    if (stop_.load(std::memory_order_relaxed)) return;
                    seen = group.control->epoch.load(std::memory_order_relaxed);
#ifdef FT_LUT_DIAG_TRACE
                    auto& trace = traces_[group_[i]][local_[i]].value;
                    trace.wake_ns = trace_clock_ns();
                    trace.publication_ns = group.publication_ns;
#endif
                    try { task_(context_, group_[i], local_[i], group.count); } catch (...) { capture_error(); }
#ifdef FT_LUT_DIAG_TRACE
                    trace.completion_ns = trace_clock_ns();
#endif
                    if (group.control->remaining.fetch_sub(1, std::memory_order_acq_rel) == 1)
                        group.control->completed.store(seen, std::memory_order_release);
                }
            });
            while (ready_.load(std::memory_order_acquire) != options.cpus.size()) std::this_thread::yield();
            rethrow();
        } catch (...) { stop(); throw; }
    }
    ~Workers() { stop(); for (auto& g : groups_) if (g.control) g.control->~Control(); }
    void capture_error() { std::lock_guard<std::mutex> lock(error_mutex_); if (!error_) error_ = std::current_exception(); }
    void rethrow() {
        std::exception_ptr error;
        { std::lock_guard<std::mutex> lock(error_mutex_); error = error_; error_ = nullptr; }
        if (error) std::rethrow_exception(error);
        if (wake_error_.exchange(0, std::memory_order_acq_rel)) throw std::runtime_error("LUT futex wake failed; pools drained");
    }
    void run(void (*fn)(void*, int, int, int), void* context, uint64_t remaining_ms = 0) {
        require(!cancelled_.load(std::memory_order_acquire), "LUT backend was cancelled");
        abort_.store(false, std::memory_order_release);
        task_ = fn; context_ = context; ++submitted_;
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(remaining_ms ? std::min(remaining_ms, timeout_ms_) : timeout_ms_);
        for (auto& g : groups_) {
#ifdef FT_LUT_DIAG_TRACE
            const int group = int(&g - groups_.data());
            for (size_t worker = 0; worker < cpus_.size(); ++worker) if (group_[worker] == group) {
                auto& trace = traces_[group][local_[worker]].value;
                trace = {}; trace.cpu = cpus_[worker]; trace.group = group; trace.local = local_[worker];
            }
#endif
            g.control->remaining.store(unsigned(g.count), std::memory_order_relaxed);
#ifdef FT_LUT_DIAG_TRACE
            g.publication_ns = trace_clock_ns();
#endif
            g.control->epoch.store(submitted_, std::memory_order_release);
            notify(g);
        }
        bool expired = false, deadline_expired = false; unsigned polls = 0;
        for (auto& g : groups_) while (g.control->completed.load(std::memory_order_acquire) != submitted_) {
            if (!expired && (++polls & 255U) == 0 && (stop_.load(std::memory_order_relaxed) || cancelled_.load(std::memory_order_acquire) || (external_abort_ && __atomic_load_n(external_abort_, __ATOMIC_ACQUIRE)) || std::chrono::steady_clock::now() >= deadline)) {
                expired = true; deadline_expired = std::chrono::steady_clock::now() >= deadline;
                abort_.store(true, std::memory_order_release);
            }
            _mm_pause();
        }
        // Publish retirement even when the completed generation will report an
        // error. Equality remains valid across64-bit epoch wrap and load jobs.
        if (idle_after_drain_) for (auto& group : groups_)
            group.control->retired.store(submitted_, std::memory_order_release);
        // Contexts and buffers remain owned by the caller until BOTH groups have
        // drained. Never detach a worker or return while it can reference them.
        if (expired) { if (deadline_expired) timeouts_.fetch_add(1, std::memory_order_relaxed); rethrow();
            throw std::runtime_error(deadline_expired ? "LUT job deadline expired; both pools drained" : "LUT job cancelled; both pools drained"); }
        rethrow();
        require(!aborted(), "LUT job cancelled; both pools drained");
    }
    void set_timeout(uint64_t milliseconds) { require(milliseconds >= 1 && milliseconds <= 3600000, "LUT job timeout must be 1..3600000 milliseconds"); timeout_ms_ = milliseconds; }
    void request_abort() {
        cancelled_.store(true, std::memory_order_release); abort_.store(true, std::memory_order_release);
        for (auto& group : groups_) notify(group);
    }
    bool aborted() const { return abort_.load(std::memory_order_acquire) || cancelled_.load(std::memory_order_acquire) || (external_abort_ && __atomic_load_n(external_abort_, __ATOMIC_ACQUIRE)); }
    uint64_t timeouts() const { return timeouts_.load(std::memory_order_relaxed); }
    uint64_t wake_failures() const { return wake_failures_.load(std::memory_order_relaxed); }
    int wake_errno() const { return last_wake_errno_.load(std::memory_order_relaxed); }
    std::array<uint64_t, 2> naps() const { return {groups_[0].control->naps.load(std::memory_order_relaxed), groups_[1].control->naps.load(std::memory_order_relaxed)}; }
    std::array<FixtureGeneration, 2> generations() const {
        std::array<FixtureGeneration, 2> result;
        for (int group = 0; group < 2; ++group) {
            const auto& control = *groups_[group].control;
            result[group] = {control.epoch.load(std::memory_order_acquire), control.completed.load(std::memory_order_acquire), control.retired.load(std::memory_order_acquire)};
        }
        return result;
    }
    void barrier(int g) { groups_[g].control->barrier.wait(); }
#ifdef FT_LUT_DIAG_TRACE
    void trace_phase(int group, int local, int phase, uint64_t nanoseconds) { traces_[group][local].value.phase_ns[size_t(phase)] = nanoseconds; }
    void trace_barrier(int group, int local, int barrier, uint64_t nanoseconds) { traces_[group][local].value.barrier_ns[size_t(barrier)] = nanoseconds; }
    std::vector<FixtureWorkerTrace> trace() const {
        std::vector<FixtureWorkerTrace> result;
        for (size_t worker = 0; worker < cpus_.size(); ++worker) result.push_back(traces_[group_[worker]][local_[worker]].value);
        return result;
    }
#endif
    int node(int g) const { return groups_[g].node; }
    int count(int g) const { return groups_[g].count; }
    std::vector<ControlReceipt> controls() const {
        std::vector<ControlReceipt> result;
        for (int index = 0; index < 2; ++index) {
            const auto& g = groups_[index]; ControlReceipt receipt;
            receipt.group = index; receipt.node = g.node; receipt.preferred_policy = g.arena->policy.verified; receipt.policy_errno = g.arena->policy.error;
            receipt.policy_mode = g.arena->policy.mode; receipt.policy_nodes = policy_nodes(g.arena->policy);
            for (size_t id = 0; id < cpus_.size(); ++id) if (group_[id] == index) {
                receipt.cpus.push_back(cpus_[id]);
                const auto& policy = thread_policies_[id];
                receipt.worker_policy_failures += !policy.verified;
                receipt.thread_policies.push_back({cpus_[id], g.node, policy.mode, policy.error, policy.verified, policy_nodes(policy)});
                if (!local_[id]) receipt.owner_cpu = cpus_[id];
            }
            void* address = g.arena->data;
            if (syscall(SYS_move_pages, 0, 1, &address, nullptr, &receipt.observed_node, 0) < 0) receipt.query_errno = errno;
            result.push_back(std::move(receipt));
        }
        return result;
    }
    bool controls_verified() const {
        for (const auto& receipt : controls())
            if (!receipt.preferred_policy || receipt.worker_policy_failures || receipt.query_errno || receipt.observed_node != receipt.node) return false;
        return !virtual_tp2;
    }
    void notify(Group& group) {
        if (!futex_wake_ || !group.ready.load(std::memory_order_acquire)) return;
        group.control->wake_generation.fetch_add(1, std::memory_order_release);
        // Always wake every parked worker. Avoid a separate waiter-count fast
        // path whose ordering could race the final epoch check before WAIT.
        if (syscall(SYS_futex, &group.control->wake_generation, FUTEX_WAKE_PRIVATE, INT32_MAX, nullptr, nullptr, 0) < 0) {
            const int error = errno;
            wake_error_.store(error, std::memory_order_release);
            last_wake_errno_.store(error, std::memory_order_relaxed);
            wake_failures_.fetch_add(1, std::memory_order_relaxed);
        }
    }
    void stop() {
        stop_.store(true);
        for (auto& group : groups_) notify(group);
        for (auto& t : threads_) if (t.joinable()) t.join();
    }
};
struct NodeWeights {
    std::array<std::unique_ptr<Arena>, 3> arenas;
    std::array<size_t, 3> strides{}, expert_bytes{};
    const uint8_t* row(int matrix, int expert, int r) const {
        return arenas[matrix]->data + size_t(expert) * expert_bytes[matrix] + size_t(r) * strides[matrix];
    }
};
struct Layer {
    int model_layer, li, h, intermediate, experts;
    std::array<Format, 3> formats;
    std::array<NodeWeights, 2> nodes;
};
struct Job { int expert = 0, m = 0; int tokens[4]{}; float weights[4]{}; };
struct Scratch {
    std::array<Activation, 4> input;
    std::array<std::array<Activation, 2>, 6> pairs;
    std::array<std::array<Activation, 4>, MAX_PICKS> hidden_q;
    lutbench::AlignedVector<float> hidden, partial;
    int h = 0, half = 0;
    std::vector<void*> sample_addresses() const {
        std::set<uintptr_t> pages;
        const uintptr_t page = uintptr_t(sysconf(_SC_PAGESIZE));
        auto sample = [&](const auto& values) {
            if (values.empty()) return;
            uintptr_t begin = reinterpret_cast<uintptr_t>(values.data());
            size_t size = values.size() * sizeof(typename std::decay_t<decltype(values)>::value_type);
            pages.insert(begin / page * page); pages.insert((begin + size - 1) / page * page);
        };
        auto activation = [&](const Activation& a) {
            sample(a.scales); sample(a.i16); sample(a.i16_lo_bsums); sample(a.i16_hi_bsums);
            sample(a.i16_swizzled); sample(a.i16_lo); sample(a.i16_hi); sample(a.q8);
        };
        for (const auto& a : input) activation(a);
        for (const auto& pair : pairs) for (const auto& a : pair) activation(a);
        for (const auto& job : hidden_q) for (const auto& a : job) activation(a);
        sample(hidden); sample(partial);
        std::vector<void*> result; for (auto address : pages) result.push_back(reinterpret_cast<void*>(address)); return result;
    }
    size_t bytes() const {
        auto activation_bytes = [](const Activation& a) {
            return a.scales.capacity() * sizeof(float) + (a.i16.capacity() + a.i16_lo_bsums.capacity() + a.i16_hi_bsums.capacity() + a.i16_swizzled.capacity()) * sizeof(int16_t)
                + a.i16_lo.capacity() + a.i16_hi.capacity() + a.q8.capacity();
        };
        size_t size = (hidden.capacity() + partial.capacity()) * sizeof(float);
        for (const auto& a : input) size += activation_bytes(a);
        for (const auto& pair : pairs) for (const auto& a : pair) size += activation_bytes(a);
        for (const auto& job : hidden_q) for (const auto& a : job) size += activation_bytes(a);
        return size;
    }
    size_t mapped_bytes() const {
        size_t total = 0;
        auto count = [&](const auto& values) {
            if (values.capacity()) total += lutbench::aligned_mapping_bytes(values.capacity() * sizeof(typename std::decay_t<decltype(values)>::value_type));
        };
        auto activation = [&](const Activation& a) {
            count(a.scales); count(a.i16); count(a.i16_lo_bsums); count(a.i16_hi_bsums);
            count(a.i16_swizzled); count(a.i16_lo); count(a.i16_hi); count(a.q8);
        };
        for (const auto& a : input) activation(a);
        for (const auto& pair : pairs) for (const auto& a : pair) activation(a);
        for (const auto& job : hidden_q) for (const auto& a : job) activation(a);
        count(hidden); count(partial); return total;
    }
    void init(Path path, int width, int ihalf) {
        for (auto& a : input) lutbench::prepare_storage(path, width, a);
        for (auto& pair : pairs) for (auto& a : pair) lutbench::prepare_storage(path, width, a);
        for (auto& job : hidden_q) for (auto& a : job) lutbench::prepare_storage(path, ihalf, a);
        hidden.resize(size_t(MAX_PICKS) * 4 * ihalf);
        partial.resize(size_t(4) * width);
        h = width; half = ihalf;
    }
};
int pair_index(int a, int b) {
    int index = 0;
    for (int x = 0; x < 4; ++x) for (int y = x + 1; y < 4; ++y, ++index) if (x == a && y == b) return index;
    throw std::runtime_error("invalid token pair");
}
void dots(Format fmt, const void* row, const Activation* inputs, int n, int m, float* out) {
    if (m == 3) {
        lutbench::dot_multi(fmt, row, inputs, n, 2, out);
        lutbench::dot_multi(fmt, row, inputs + 2, n, 1, out + 2);
    } else lutbench::dot_multi(fmt, row, inputs, n, m, out);
}
} // namespace

bool FileIdentity::operator==(const FileIdentity& other) const {
    return dev == other.dev && ino == other.ino && size == other.size && mtime_ns == other.mtime_ns && ctime_ns == other.ctime_ns;
}
FileIdentity file_identity(const std::string& path) { File f(path); return identity(f.fd); }

struct Backend::Impl {
    Options options;
    Path path;
    mutable std::mutex mutex, report_mutex;
    std::map<int, std::unique_ptr<Layer>> layers;
    std::array<Scratch, 2> scratch;
    Receipt report;
    Workers workers;
    std::atomic<bool> failed{false};
    std::mutex failure_mutex;
    std::exception_ptr failure;
    int injected_phase = -1, injected_group = -1;
    int delayed_phase = -1, delayed_group = -1;
    unsigned delay_ms = 0;
    Layer* active = nullptr;
    const float* x = nullptr;
    int ntok = 0, njobs = 0;
    std::array<Job, MAX_PICKS> jobs;
    std::array<uint64_t, 2> pending_mapped{};
    explicit Impl(const Options& o, bool fixture) : options(o), path(o.activation == "i16" ? Path::I16 : Path::Q8), workers(checked_options(o, fixture)) {
        require(o.activation == "i16" || o.activation == "q8", "LUT activation must be i16 or q8");
        require(std::isfinite(o.act_limit) && o.act_limit > 0, "LUT SwiGLU limit must be finite and positive");
        report.activation = o.activation; report.act_limit = o.act_limit; report.virtual_tp2 = o.virtual_tp2;
        report.offline_fixture = fixture;
        report.controls = workers.controls();
        for (int g = 0; g < 2; ++g) { report.nodes[g] = workers.node(g); report.workers[g] = workers.count(g); }
    }
    ~Impl() { workers.stop(); }
    void headroom() const {
        check_startup_headroom(host_available(), {node_free(workers.node(0)), node_free(workers.node(1))}, pending_mapped);
        cgroup_headroom(pending_mapped[0] + pending_mapped[1] + HOST_OVERHEAD);
    }
    template<class F> void phase(int number, int group, int local, F fn) noexcept {
#ifdef FT_LUT_DIAG_TRACE
        const uint64_t phase_start = trace_clock_ns();
#endif
        try {
            if (number == delayed_phase && group == delayed_group && local == 0)
                std::this_thread::sleep_for(std::chrono::milliseconds(delay_ms));
            if (number == injected_phase && group == injected_group && local == 0) throw std::runtime_error("injected offline LUT worker failure");
            if (!failed.load(std::memory_order_acquire) && !workers.aborted()) fn();
        } catch (...) {
            std::lock_guard<std::mutex> lock(failure_mutex);
            if (!failure) failure = std::current_exception();
            failed.store(true, std::memory_order_release);
        }
#ifdef FT_LUT_DIAG_TRACE
        workers.trace_phase(group, local, number, trace_clock_ns() - phase_start);
#endif
    }
    static void compute(void* opaque, int group, int local, int nworkers) {
        auto& self = *static_cast<Impl*>(opaque);
        Layer& l = *self.active; Scratch& s = self.scratch[group];
        const int half = l.intermediate / 2;
        const int block = self.path == Path::I16 ? 128 : 256;
        self.phase(0, group, local, [&] {
            struct Input { const float* source; Activation* destination; } inputs[16];
            int count = 0; bool needed[6]{};
            for (int t = 0; t < self.ntok; ++t) inputs[count++] = {self.x + size_t(t) * l.h, &s.input[t]};
            for (int j = 0; j < self.njobs; ++j) {
                const Job& job = self.jobs[j];
                if (job.m >= 2 && job.tokens[1] != job.tokens[0] + 1) needed[pair_index(job.tokens[0], job.tokens[1])] = true;
            }
            int pair = 0;
            for (int a = 0; a < 4; ++a) for (int b = a + 1; b < 4; ++b, ++pair) if (needed[pair]) {
                inputs[count++] = {self.x + size_t(a) * l.h, &s.pairs[pair][0]};
                inputs[count++] = {self.x + size_t(b) * l.h, &s.pairs[pair][1]};
            }
            const int blocks = l.h / block;
            for (int task = local; task < count * blocks; task += nworkers) {
                if (self.workers.aborted()) break;
                auto in = inputs[task / blocks]; int begin = (task % blocks) * block;
                lutbench::prepare_range(self.path, in.source, l.h, begin, begin + block, *in.destination);
            }
        });
#ifdef FT_LUT_DIAG_TRACE
        const uint64_t barrier0_start = trace_clock_ns();
#endif
        self.workers.barrier(group);
#ifdef FT_LUT_DIAG_TRACE
        self.workers.trace_barrier(group, local, 0, trace_clock_ns() - barrier0_start);
#endif
        self.phase(1, group, local, [&] {
            // Match the codec runner's contiguous row ownership: adjacent
            // hidden values stay on one worker's cache lines, and each worker
            // streams through one expert's gate/up rows before changing expert.
            const int begin = half * local / nworkers, end = half * (local + 1) / nworkers;
            for (int j = 0; j < self.njobs; ++j) {
                const Job& job = self.jobs[j];
                const Activation* input = &s.input[job.tokens[0]];
                if (job.m >= 2 && job.tokens[1] != job.tokens[0] + 1) input = s.pairs[pair_index(job.tokens[0], job.tokens[1])].data();
                for (int r = begin; r < end; ++r) {
                    if (self.workers.aborted()) break;
                    float g[4], u[4]; int first = job.m == 3 ? 2 : job.m;
                    dots(l.formats[0], l.nodes[group].row(0, job.expert, r), input, l.h, first, g);
                    dots(l.formats[1], l.nodes[group].row(1, job.expert, r), input, l.h, first, u);
                    if (job.m == 3) {
                        dots(l.formats[0], l.nodes[group].row(0, job.expert, r), &s.input[job.tokens[2]], l.h, 1, g + 2);
                        dots(l.formats[1], l.nodes[group].row(1, job.expert, r), &s.input[job.tokens[2]], l.h, 1, u + 2);
                    }
                    for (int t = 0; t < job.m; ++t) {
                        require(std::isfinite(g[t]) && std::isfinite(u[t]), "nonfinite LUT gate/up projection");
                        float gate = std::min(g[t] / (1.0f + std::exp(-g[t])), self.options.act_limit);
                        float up = std::max(-self.options.act_limit, std::min(self.options.act_limit, u[t]));
                        float hidden = gate * up;
                        require(std::isfinite(hidden), "nonfinite LUT hidden activation");
                        s.hidden[(size_t(j) * 4 + t) * half + r] = hidden;
                    }
                }
            }
        });
#ifdef FT_LUT_DIAG_TRACE
        const uint64_t barrier1_start = trace_clock_ns();
#endif
        self.workers.barrier(group);
#ifdef FT_LUT_DIAG_TRACE
        self.workers.trace_barrier(group, local, 1, trace_clock_ns() - barrier1_start);
#endif
        self.phase(2, group, local, [&] {
            const int blocks = half / block;
            for (int task = local; task < self.njobs * 4 * blocks; task += nworkers) {
                if (self.workers.aborted()) break;
                int b = task % blocks, slot = task / blocks, j = slot / 4, t = slot % 4;
                if (t >= self.jobs[j].m) continue;
                lutbench::prepare_range(self.path, s.hidden.data() + size_t(slot) * half, half, b * block, (b + 1) * block, s.hidden_q[j][t]);
            }
        });
#ifdef FT_LUT_DIAG_TRACE
        const uint64_t barrier2_start = trace_clock_ns();
#endif
        self.workers.barrier(group);
#ifdef FT_LUT_DIAG_TRACE
        self.workers.trace_barrier(group, local, 2, trace_clock_ns() - barrier2_start);
#endif
        self.phase(3, group, local, [&] {
            const int begin = l.h * local / nworkers, end = l.h * (local + 1) / nworkers;
            for (int r = begin; r < end; ++r)
                for (int t = 0; t < self.ntok; ++t) s.partial[size_t(t) * l.h + r] = 0;
            // Expert-major streaming preserves the same ascending expert
            // addition order for every output while avoiding interleaved rows.
            for (int j = 0; j < self.njobs; ++j) {
                const Job& job = self.jobs[j];
                for (int r = begin; r < end; ++r) {
                    if (self.workers.aborted()) break;
                    float values[4];
                    dots(l.formats[2], l.nodes[group].row(2, job.expert, r), s.hidden_q[j].data(), half, job.m, values);
                    for (int t = 0; t < job.m; ++t) {
                        float contribution = job.weights[t] * values[t];
                        float& partial = s.partial[size_t(job.tokens[t]) * l.h + r];
                        float updated = partial + contribution;
                        require(std::isfinite(values[t]) && std::isfinite(contribution) && std::isfinite(updated), "nonfinite LUT weighted down output");
                        partial = updated;
                    }
                }
            }
        });
    }
};

BankPlan bank_plan(const std::vector<LayerInput>& inputs) {
    require(inputs.size() == 42, "full LUT bank requires exactly 42 routed layers");
    BankPlan plan; std::set<int> seen;
    const uint64_t page = uint64_t(sysconf(_SC_PAGESIZE));
    for (const auto& in : inputs) {
        require(in.model_layer >= 3 && in.model_layer <= 44 && in.internal_li == in.model_layer - 3 && seen.insert(in.model_layer).second,
                "full LUT bank layer inventory is incomplete or duplicated");
        std::array<int, 3> types{138, 138, 144};
        if (in.model_layer == 11) types = {144, 144, 14};
        else if (in.model_layer == 12 || in.model_layer == 44) types[2] = 14;
        std::vector<std::pair<uint64_t, uint64_t>> ranges;
        for (int m = 0; m < 3; ++m) {
            const auto& t = in.tensors[m]; int width = m == 2 ? 2048 : 4096, rows = m == 2 ? 4096 : 2048;
            require(t.type_id == types[m] && t.width == width && t.rows == rows && t.experts == 288, "full LUT bank tensor recipe/geometry mismatch");
            size_t canonical = mul(mul(lutbench::row_bytes(format(t.type_id), width), size_t(rows)), 288);
            require(t.nbytes == canonical && t.offset <= in.identity.size && t.nbytes <= in.identity.size - t.offset,
                    "full LUT bank tensor extent mismatch");
            require(canonical / 288 <= STAGING_LIMIT, "full LUT bank staging bound exceeded");
            ranges.emplace_back(t.offset, t.offset + t.nbytes); plan.canonical_payload_bytes += canonical;
            size_t split = mul(mul(lutbench::row_bytes(format(t.type_id), m == 2 ? 1024 : 4096), size_t(m == 2 ? 4096 : 1024)), 288);
            uint64_t mapped = (split + page - 1) / page * page;
            for (auto& node : plan.mapped_bytes) node += mapped;
            plan.tp2_payload_bytes += 2 * split;
        }
        std::sort(ranges.begin(), ranges.end());
        for (size_t n = 1; n < ranges.size(); ++n) require(ranges[n - 1].second <= ranges[n].first, "full LUT bank tensor extents overlap");
    }
    require(plan.canonical_payload_bytes == 143879307264ULL && plan.tp2_payload_bytes == 144063332352ULL &&
            plan.mapped_bytes[0] == 72031666176ULL && plan.mapped_bytes[1] == 72031666176ULL, "full LUT bank exact byte ledger mismatch");
    return plan;
}
void check_startup_headroom(uint64_t available, const std::array<uint64_t, 2>& free,
                            const std::array<uint64_t, 2>& remaining) {
    require(remaining[0] <= std::numeric_limits<uint64_t>::max() - remaining[1], "LUT headroom size overflow");
    require(available >= HOST_FLOOR + HOST_OVERHEAD && remaining[0] + remaining[1] <= available - HOST_FLOOR - HOST_OVERHEAD,
            "LUT startup would violate host 100 GiB reserve plus declared peak overhead");
    for (int group = 0; group < 2; ++group)
        require(free[group] >= NODE_FLOOR + NODE_OVERHEAD && remaining[group] <= free[group] - NODE_FLOOR - NODE_OVERHEAD,
                "LUT startup would violate per-node 8 GiB reserve plus declared overhead");
}
void check_memory_limit(uint64_t limit, uint64_t current, uint64_t additional) {
    require(current <= limit && additional <= limit - current, "LUT startup exceeds a finite cgroup memory limit");
}
Backend::Backend(const Options& options, bool fixture) : impl_(std::make_unique<Impl>(options, fixture)) {}
std::unique_ptr<Backend> Backend::offline_fixture(const Options& options) { return std::unique_ptr<Backend>(new Backend(options)); }
std::unique_ptr<Backend> Backend::staged_bank(const Options& options, const std::vector<LayerInput>& inputs) {
    checked_options(options, false);
    const auto plan = bank_plan(inputs);
    require(!options.virtual_tp2 && options.cpus.size() == 42, "full LUT bank requires 42 physical workers on two actual NUMA nodes");
    std::vector<int> nodes;
    for (int cpu : options.cpus) { int node = cpu_node(cpu); if (std::find(nodes.begin(), nodes.end(), node) == nodes.end()) nodes.push_back(node); }
    require(nodes.size() == 2, "full LUT bank requires two actual NUMA nodes");
    check_startup_headroom(host_available(), {node_free(nodes[0]), node_free(nodes[1])}, plan.mapped_bytes);
    cgroup_headroom(plan.mapped_bytes[0] + plan.mapped_bytes[1] + HOST_OVERHEAD);
    // No arena allocation until all file identities and all 126 extents passed.
    for (const auto& input : inputs) { File file(input.path); require(identity(file.fd) == input.identity, "full LUT bank source fingerprint mismatch"); }
    Options strict = options; strict.require_verified_placement = true;
    auto backend = std::unique_ptr<Backend>(new Backend(strict, false));
    backend->impl_->pending_mapped = plan.mapped_bytes;
    for (const auto& input : inputs) backend->add_layer(input);
    for (const auto& input : inputs) require(file_identity(input.path) == input.identity, "full LUT bank source changed before readiness");
    require(backend->impl_->layers.size() == 42 && backend->impl_->pending_mapped == std::array<uint64_t, 2>{0, 0}, "full LUT bank publication is incomplete");
    backend->impl_->report.bank_complete = true;
    return backend;
}
Backend::~Backend() = default;

void Backend::add_layer(const LayerInput& in) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    std::lock_guard<std::mutex> report_lock(impl_->report_mutex);
    require(in.model_layer >= 3 && in.model_layer <= 44 && in.internal_li == in.model_layer - 3, "LUT model/internal layer mapping mismatch");
    require(!impl_->layers.count(in.internal_li), "LUT layer already loaded");
    int h = in.tensors[0].width, i = in.tensors[0].rows, e = in.tensors[0].experts;
    if (impl_->report.offline_fixture)
        require(h >= 256 && h <= 4096 && h % 256 == 0 && i >= 512 && i <= 2048 && i % 512 == 0 && e >= 1 && e <= 8,
                "offline LUT fixture dimensions exceed bounded geometry");
    else {
        require(!impl_->report.bank_complete && h == 4096 && i == 2048 && e == 288, "full LUT bank geometry/readiness mismatch");
        impl_->headroom();
    }
    if (!impl_->layers.empty()) {
        const auto& first = *impl_->layers.begin()->second;
        require(first.h == h && first.intermediate == i, "LUT layers must share input/intermediate geometry");
    }
    std::array<int, 3> types{138, 138, 144};
    if (in.model_layer == 11) types = {144, 144, 14};
    else if (in.model_layer == 12 || in.model_layer == 44) types[2] = 14;
    std::array<size_t, 3> source_stride{}, projection_bytes{};
    uint64_t additional = 0;
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    for (int m = 0; m < 3; ++m) {
        const auto& t = in.tensors[m];
        require(t.type_id == types[m] && t.width == (m == 2 ? i : h) && t.rows == (m == 2 ? h : i) && t.experts == e,
                "LUT tensor recipe or dimensions mismatch");
        source_stride[m] = lutbench::row_bytes(format(t.type_id), t.width);
        projection_bytes[m] = mul(source_stride[m], size_t(t.rows));
        require(projection_bytes[m] <= STAGING_LIMIT && t.nbytes == mul(projection_bytes[m], size_t(e)) &&
                t.offset <= in.identity.size && t.nbytes <= in.identity.size - t.offset, "LUT tensor bounds/staging mismatch");
        ranges.emplace_back(t.offset, t.offset + t.nbytes);
        size_t final_stride = lutbench::row_bytes(format(t.type_id), m == 2 ? i / 2 : h);
        additional += 2 * mul(mul(final_stride, size_t(m == 2 ? h : i / 2)), size_t(e));
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t n = 1; n < ranges.size(); ++n) require(ranges[n - 1].second <= ranges[n].first, "overlapping LUT tensor ranges");
    if (impl_->report.offline_fixture)
        require(additional <= FIXTURE_LIMIT && impl_->report.payload_bytes[0] + impl_->report.payload_bytes[1] <= FIXTURE_LIMIT - additional,
                "offline LUT fixture exceeds aggregate allocation bound");
    File file(in.path); require(identity(file.fd) == in.identity, "LUT source fingerprint changed before load");
    auto layer = std::make_unique<Layer>();
    layer->model_layer = in.model_layer; layer->li = in.internal_li; layer->h = h; layer->intermediate = i; layer->experts = e;
    std::vector<RegionReceipt> receipts;
    for (int m = 0; m < 3; ++m) {
        layer->formats[m] = format(types[m]);
        for (int g = 0; g < 2; ++g) {
            auto& node = layer->nodes[g]; node.strides[m] = lutbench::row_bytes(layer->formats[m], m == 2 ? i / 2 : h);
            node.expert_bytes[m] = mul(node.strides[m], size_t(m == 2 ? h : i / 2));
            node.arenas[m] = std::make_unique<Arena>(mul(node.expert_bytes[m], size_t(e)), impl_->workers.node(g), impl_->options.fixture_weight_pages);
            require(impl_->report.offline_fixture || node.arenas[m]->policy.verified, "full LUT bank preferred placement policy unavailable or read-back mismatch");
        }
    }
    struct Copy {
        Layer* layer; int matrix, expert; const uint8_t* data; size_t source_stride;
        static void run(void* opaque, int group, int local, int nworkers) {
            auto& job = *static_cast<Copy*>(opaque); auto& l = *job.layer;
            const int rows = job.matrix == 2 ? l.h : l.intermediate / 2;
            auto& node = l.nodes[group];
            for (int r = local; r < rows; r += nworkers) {
                uint8_t* destination = node.arenas[job.matrix]->data + size_t(job.expert) * node.expert_bytes[job.matrix] + size_t(r) * node.strides[job.matrix];
                if (job.matrix == 2)
                    lutbench::slice_packed_row(l.formats[2], job.data + size_t(r) * job.source_stride, l.intermediate, group * (l.intermediate / 2), l.intermediate / 2, destination);
                else std::memcpy(destination, job.data + size_t(group * (l.intermediate / 2) + r) * job.source_stride, node.strides[job.matrix]);
            }
        }
    };
    for (int m = 0; m < 3; ++m) {
        std::vector<uint8_t> staging(projection_bytes[m]);
        impl_->report.staging_peak_bytes = std::max(impl_->report.staging_peak_bytes, uint64_t(staging.size()));
        for (int expert = 0; expert < e; ++expert) {
            if (!impl_->report.offline_fixture && expert % 16 == 0) impl_->headroom();
            file.read(in.tensors[m].offset + size_t(expert) * staging.size(), staging.data(), staging.size());
            Copy job{layer.get(), m, expert, staging.data(), source_stride[m]};
            impl_->workers.run(&Copy::run, &job);
            if (!impl_->report.offline_fixture) for (int group = 0; group < 2; ++group) {
                size_t touched = layer->nodes[group].expert_bytes[m];
                require(touched % size_t(sysconf(_SC_PAGESIZE)) == 0 && impl_->pending_mapped[group] >= touched, "LUT remaining-touch ledger mismatch");
                impl_->pending_mapped[group] -= touched;
            }
        }
    }
    require(identity(file.fd) == in.identity && file_identity(in.path) == in.identity, "LUT source/path changed while streaming rows");
    for (int g = 0; g < 2; ++g) for (int m = 0; m < 3; ++m)
        layer->nodes[g].arenas[m]->hint_after_touch(impl_->workers.node(g));
    // Policy read-back and move_pages sampling occur after the hint. A preferred
    // policy permits fallback; only the observed samples can verify locality.
    for (int g = 0; g < 2; ++g) for (int m = 0; m < 3; ++m) {
        const auto& node = layer->nodes[g]; const auto& arena = *node.arenas[m];
        RegionReceipt receipt;
        receipt.model_layer = in.model_layer; receipt.group = g; receipt.node = impl_->workers.node(g); receipt.matrix = m;
        receipt.payload_bytes = arena.bytes; receipt.mapped_bytes = arena.mapped;
        receipt.preferred_policy = arena.policy.verified; receipt.policy_errno = arena.policy.error;
        receipt.policy_mode = arena.policy.mode; receipt.policy_nodes = policy_nodes(arena.policy);
        std::vector<void*> addresses;
        const size_t page = size_t(sysconf(_SC_PAGESIZE));
        for (int expert = 0; expert < e; ++expert) for (int sample = 0; sample < 4; ++sample) {
            size_t byte = size_t(expert) * node.expert_bytes[m] + (node.expert_bytes[m] - 1) * size_t(sample) / 3;
            addresses.push_back(arena.data + byte / page * page);
        }
        std::vector<int> status(addresses.size(), -1);
        if (syscall(SYS_move_pages, 0, addresses.size(), addresses.data(), nullptr, status.data(), 0) < 0) receipt.query_errno = errno;
        else for (int observed : status) { if (observed < 0) ++receipt.page_errors; else if (observed != receipt.node) ++receipt.off_node_pages; }
        receipt.sampled_pages = addresses.size(); receipts.push_back(receipt);
    }
    struct Init {
        Impl* self; int h, half;
        static void run(void* opaque, int group, int local, int) {
            auto& init = *static_cast<Init*>(opaque);
            if (!local && init.self->scratch[group].h == 0) init.self->scratch[group].init(init.self->path, init.h, init.half);
        }
    } init{impl_.get(), h, i / 2};
    impl_->workers.run(&Init::run, &init);
    for (int group = 0; group < 2; ++group) {
        impl_->report.scratch_bytes[group] = impl_->scratch[group].bytes();
        impl_->report.scratch_mapped_bytes[group] = impl_->scratch[group].mapped_bytes();
        require(impl_->report.scratch_mapped_bytes[group] <= 32ULL * 1024 * 1024, "LUT per-node mapped scratch exceeds declared bound");
    }
    bool verified = impl_->workers.controls_verified();
    for (const auto& receipt : receipts) verified = verified && receipt.preferred_policy && !receipt.query_errno && !receipt.page_errors && !receipt.off_node_pages;
    std::vector<ScratchReceipt> scratch_receipts;
    for (int group = 0; group < 2; ++group) {
        auto addresses = impl_->scratch[group].sample_addresses(); std::vector<int> status(addresses.size(), -1);
        ScratchReceipt receipt; receipt.group = group; receipt.node = impl_->workers.node(group); receipt.sampled_pages = addresses.size();
        receipt.payload_bytes = impl_->report.scratch_bytes[group]; receipt.mapped_bytes = impl_->report.scratch_mapped_bytes[group];
        for (void* address : addresses) {
            const auto policy = read_policy(receipt.node, address, true);
            if (receipt.policy_mode == -1) { receipt.policy_mode = policy.mode; receipt.policy_nodes = policy_nodes(policy); }
            if (!policy.verified) { ++receipt.policy_failures; if (!receipt.policy_errno) receipt.policy_errno = policy.error; }
        }
        receipt.preferred_policy = !addresses.empty() && receipt.policy_failures == 0;
        if (syscall(SYS_move_pages, 0, addresses.size(), addresses.data(), nullptr, status.data(), 0) < 0) receipt.query_errno = errno;
        else for (int observed : status) { if (observed < 0) ++receipt.page_errors; else if (observed != receipt.node) ++receipt.off_node_pages; }
        verified = verified && receipt.preferred_policy && !receipt.query_errno && !receipt.page_errors && !receipt.off_node_pages;
        scratch_receipts.push_back(receipt);
    }
    require(!impl_->options.require_verified_placement || verified, "LUT placement could not be verified");
    Receipt published = impl_->report;
    for (const auto& receipt : receipts) {
        published.payload_bytes[receipt.group] += receipt.payload_bytes;
        published.mapped_bytes[receipt.group] += receipt.mapped_bytes;
        published.regions.push_back(receipt);
    }
    published.placement_verified = impl_->layers.empty() ? verified : published.placement_verified && verified;
    published.scratch = std::move(scratch_receipts);
    require(identity(file.fd) == in.identity && file_identity(in.path) == in.identity, "LUT source/path changed before publication");
    impl_->layers.emplace(in.internal_li, std::move(layer));
    impl_->report = std::move(published);
}

void Backend::forward(int li, const float* x, int ntok, const Routes& routes, float* output, uint64_t remaining_ms) {
    std::unique_lock<std::mutex> lock(impl_->mutex, std::try_to_lock);
    require(lock.owns_lock(), "LUT supports one in-flight coordinator job");
    require(remaining_ms <= 3600000, "LUT remaining deadline budget exceeds one hour");
    require(x && output && ntok >= 1 && ntok <= MAX_TOKENS && routes.size() == size_t(ntok), "LUT supports only M=1..4 with matching routes");
    auto found = impl_->layers.find(li); require(found != impl_->layers.end(), "LUT layer is not loaded");
    Layer& layer = *found->second;
    require(impl_->report.offline_fixture || impl_->report.bank_complete, "full LUT bank is not ready");
    for (size_t n = 0; n < size_t(ntok) * layer.h; ++n) require(std::isfinite(x[n]), "nonfinite LUT input");
    int picks = 0;
    for (const auto& row : routes) {
        require(row.size() <= 8, "LUT supports at most eight routes per token");
        for (size_t p = 0; p < row.size(); ++p) {
            const auto& pick = row[p];
            bool duplicate = false;
            for (size_t previous = 0; previous < p; ++previous) duplicate = duplicate || row[previous].first == pick.first;
            require(pick.first >= 0 && pick.first < layer.experts && std::isfinite(pick.second) && pick.second >= 0 && !duplicate,
                    "invalid, duplicate or nonfinite LUT route");
            ++picks;
        }
    }
    require(picks <= MAX_PICKS, "LUT route worklist exceeds bound");
    impl_->active = &layer; impl_->x = x; impl_->ntok = ntok; impl_->njobs = 0;
    // Expert-major order is deterministic and weights are applied once per half.
    for (int expert = 0; expert < layer.experts; ++expert) {
        Job job; job.expert = expert;
        for (int t = 0; t < ntok; ++t) for (const auto& pick : routes[t]) if (pick.first == expert) {
            int slot = job.m++; job.tokens[slot] = t; job.weights[slot] = pick.second;
        }
        if (job.m) impl_->jobs[impl_->njobs++] = job;
    }
    impl_->failed.store(false, std::memory_order_release); impl_->failure = nullptr;
    impl_->workers.run(&Impl::compute, impl_.get(), remaining_ms);
    if (impl_->failed.load(std::memory_order_acquire)) std::rethrow_exception(impl_->failure);
    // Both release completions have been acquired before any caller output write.
    for (int n = 0; n < ntok * layer.h; ++n)
        require(std::isfinite(impl_->scratch[0].partial[n] + impl_->scratch[1].partial[n]), "nonfinite merged LUT result");
    for (int n = 0; n < ntok * layer.h; ++n) output[n] = impl_->scratch[0].partial[n] + impl_->scratch[1].partial[n];
}

void Backend::request_abort() { impl_->workers.request_abort(); }
void Backend::shutdown() {
    impl_->workers.request_abort();
    std::lock_guard<std::mutex> guard(impl_->mutex);
    impl_->workers.stop();
}

Receipt Backend::receipt() const {
    std::lock_guard<std::mutex> lock(impl_->report_mutex);
    Receipt report = impl_->report; report.pool_naps = impl_->workers.naps(); report.timed_out_jobs = impl_->workers.timeouts();
    report.futex_wake_failures = impl_->workers.wake_failures(); report.futex_wake_errno = impl_->workers.wake_errno(); return report;
}
void Backend::inject_failure(int phase, int group) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    require(impl_->report.offline_fixture && phase >= -1 && phase <= 3 && group >= 0 && group < 2, "invalid fixture failure phase");
    impl_->injected_phase = phase; impl_->injected_group = group;
}
void Backend::inject_delay(int phase, int group, unsigned milliseconds) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    require(impl_->report.offline_fixture && phase >= -1 && phase <= 3 && group >= 0 && group < 2 && milliseconds <= 10000,
            "invalid fixture delay");
    impl_->delayed_phase = phase; impl_->delayed_group = group; impl_->delay_ms = milliseconds;
}
void Backend::fixture_set_timeout(uint64_t milliseconds) {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    require(impl_->report.offline_fixture, "timeout override is an offline fixture API only");
    impl_->workers.set_timeout(milliseconds);
}
uint64_t Backend::fixture_flush_weights() {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    require(impl_->report.offline_fixture && !impl_->layers.empty(), "weight flushing requires a loaded offline LUT fixture");
    uint64_t total = 0;
    // Validate the entire owned extent before issuing any cache operation.
    for (const auto& entry : impl_->layers) for (const auto& node : entry.second->nodes) for (const auto& arena : node.arenas) {
        require(arena && arena->data && arena->bytes && arena->bytes <= FIXTURE_LIMIT - total,
                "fixture weight flush exceeds its owned allocation bound");
        total += arena->bytes;
    }
    _mm_mfence();
    for (const auto& entry : impl_->layers) for (const auto& node : entry.second->nodes) for (const auto& arena : node.arenas)
        for (size_t offset = 0; offset < arena->bytes; offset += 64) _mm_clflush(arena->data + offset);
    _mm_mfence();
    return total;
}
std::vector<uint8_t> Backend::fixture_slice(int li, int group, int expert, int matrix) const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    require(impl_->report.offline_fixture, "packed slice copies are an offline fixture API only");
    auto found = impl_->layers.find(li); require(found != impl_->layers.end(), "fixture layer missing");
    const auto& layer = *found->second;
    require(group >= 0 && group < 2 && expert >= 0 && expert < layer.experts && matrix >= 0 && matrix < 3, "fixture slice index out of range");
    const auto& node = layer.nodes[group]; const uint8_t* first = node.row(matrix, expert, 0);
    return {first, first + node.expert_bytes[matrix]};
}
std::vector<FixtureArena> Backend::fixture_arenas() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    require(impl_->report.offline_fixture, "arena addresses are an offline fixture API only");
    std::vector<FixtureArena> result;
    for (const auto& entry : impl_->layers) for (int group = 0; group < 2; ++group) for (int matrix = 0; matrix < 3; ++matrix) {
        const auto& arena = *entry.second->nodes[group].arenas[matrix];
        result.push_back({entry.second->model_layer, group, impl_->workers.node(group), matrix,
                          reinterpret_cast<uintptr_t>(arena.data), arena.bytes, arena.mapped, arena.page_policy});
    }
    return result;
}
std::array<FixtureGeneration, 2> Backend::fixture_generations() const {
    // offline_fixture is immutable after construction. Do not take the forward
    // mutex: observing an in-flight delayed generation is the purpose of this API.
    require(impl_->report.offline_fixture, "generation snapshots are an offline fixture API only");
    return impl_->workers.generations();
}
#ifdef FT_LUT_DIAG_TRACE
std::vector<FixtureWorkerTrace> Backend::fixture_trace() const {
    std::lock_guard<std::mutex> lock(impl_->mutex);
    require(impl_->report.offline_fixture, "worker traces are an offline fixture API only");
    return impl_->workers.trace();
}
#endif
} // namespace ft_lut
