// NUMA TP=2 expert backend. Codec sources and MIT attribution are in lut/.
#pragma once
#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

namespace ft_lut {

struct FileIdentity {
    uint64_t dev = 0, ino = 0, size = 0;
    int64_t mtime_ns = 0, ctime_ns = 0;
    bool operator==(const FileIdentity& other) const;
};
FileIdentity file_identity(const std::string& path);

// Offsets are absolute file offsets, including the GGUF data_start.
struct TensorInput {
    uint64_t offset = 0, nbytes = 0;
    int type_id = 0, width = 0, rows = 0, experts = 0;
};
struct LayerInput {
    int model_layer = 0, internal_li = 0;
    std::string path;
    FileIdentity identity;
    std::array<TensorInput, 3> tensors; // gate, up, down
};

struct RegionReceipt {
    int model_layer = -1, group = -1, node = -1, matrix = -1;
    uint64_t payload_bytes = 0, mapped_bytes = 0;
    bool preferred_policy = false;
    int policy_errno = 0, query_errno = 0;
    int policy_mode = -1;
    std::vector<int> policy_nodes;
    uint64_t sampled_pages = 0, off_node_pages = 0, page_errors = 0;
};
struct ThreadPolicyReceipt {
    int cpu = -1, node = -1, mode = -1, error = 0;
    bool verified = false;
    std::vector<int> nodes;
};
struct ControlReceipt {
    int group = -1, node = -1, owner_cpu = -1, query_errno = 0, observed_node = -1;
    bool preferred_policy = false;
    int policy_errno = 0, worker_policy_failures = 0;
    int policy_mode = -1;
    std::vector<int> policy_nodes;
    std::vector<int> cpus;
    std::vector<ThreadPolicyReceipt> thread_policies;
};
struct ScratchReceipt {
    int group = -1, node = -1, query_errno = 0;
    uint64_t sampled_pages = 0, off_node_pages = 0, page_errors = 0;
};
struct Receipt {
    bool offline_fixture = true, virtual_tp2 = false, placement_verified = false;
    bool bank_complete = false;
    std::string activation;
    float act_limit = 10.0f;
    std::array<int, 2> nodes{}, workers{};
    std::array<uint64_t, 2> payload_bytes{}, mapped_bytes{};
    std::array<uint64_t, 2> scratch_bytes{};
    uint64_t staging_peak_bytes = 0;
    std::array<uint64_t, 2> pool_naps{};
    uint64_t timed_out_jobs = 0;
    std::vector<ControlReceipt> controls;
    std::vector<ScratchReceipt> scratch;
    std::vector<RegionReceipt> regions;
};
struct Options {
    std::vector<int> cpus;
    std::string activation = "i16";
    bool virtual_tp2 = false; // explicit offline geometry test, never NUMA evidence
    bool require_verified_placement = false;
    float act_limit = 10.0f;
    // Cooperative deadline: abort further phase work, then drain both pools
    // before returning an error. A permanently stuck thread requires external
    // process supervision; its referenced storage must never be freed early.
    uint64_t job_timeout_ms = 30000;
    // Optional mapped ingress cancellation word, owned until backend destruction.
    const int64_t* abort_flag = nullptr;
    uint64_t initial_generation = 0; // offline wrap/reuse tests
};
struct BankPlan {
    std::array<uint64_t, 2> mapped_bytes{};
    uint64_t canonical_payload_bytes = 0, tp2_payload_bytes = 0;
    uint64_t host_working_overhead_bytes = 1ULL << 30;
    uint64_t per_node_working_overhead_bytes = 512ULL << 20;
};
// Layout planning performs no I/O and is not source verification.
BankPlan bank_plan(const std::vector<LayerInput>& inputs);
void check_startup_headroom(uint64_t host_available, const std::array<uint64_t, 2>& node_free,
                            const std::array<uint64_t, 2>& remaining_mapped);
void check_memory_limit(uint64_t limit, uint64_t current, uint64_t additional);
using Routes = std::vector<std::vector<std::pair<int, float>>>;

class Backend {
public:
    // Offline fixtures have at most eight experts and 256 MiB aggregate payload.
    // This API never starts a GPU ingress coordinator or authorizes serving.
    static std::unique_ptr<Backend> offline_fixture(const Options& options);
    // Descriptors must come from independently SHA-verified manifest/GGUF bytes.
    // The native loader rechecks open-file/path identities and every extent.
    // This factory does not admit real quality or start a GPU coordinator.
    static std::unique_ptr<Backend> staged_bank(const Options& options, const std::vector<LayerInput>& inputs);
    // Callers must quiesce API calls before destroying the object. All worker
    // threads are joined before any weight or scratch allocation is released.
    ~Backend();
    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
    void add_layer(const LayerInput& input);
    void forward(int internal_li, const float* x, int ntok, const Routes& routes, float* output, uint64_t remaining_ms = 0);
    Receipt receipt() const;
    // Sticky cooperative cancellation; current generations still drain before
    // forward returns, and later jobs are rejected. Safe from another thread.
    void request_abort();
    // After the coordinator drains, join both pools while retaining allocations.
    void shutdown();
    // Fixture-only controls: error propagation and exact packed-layout checks.
    void inject_failure(int phase, int group = 1);
    void inject_delay(int phase, int group, unsigned milliseconds);
    void fixture_set_timeout(uint64_t milliseconds);
    // Flush only owned loaded fixture weight cache lines (<=256 MiB total).
    // Serialized with forward/load/shutdown; returns flushed payload bytes.
    uint64_t fixture_flush_weights();
    std::vector<uint8_t> fixture_slice(int internal_li, int group, int expert, int matrix) const;
private:
    struct Impl;
    explicit Backend(const Options& options, bool fixture = true);
    std::unique_ptr<Impl> impl_;
};

} // namespace ft_lut
