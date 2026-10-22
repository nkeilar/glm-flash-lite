// FreeToken CPU tier: host side (torch CPU extension). Owns the ft_mul1 pool and a worker thread that
// serves one MoE job at a time through pinned, GPU-mapped buffers written by the GPU split kernel
// (ft_tier_cu.cu):
//   ctrl[0] job seq (GPU writes last)   ctrl[1] done seq (worker writes last)
//   ctrl[2] layer   ctrl[3] ntok   ctrl[4] npicks
//   hx    [ntok, H] fp16 expert input      picks [npicks][3] = (token, expert, weight f32 bits)
//   hout  [ntok, H] fp32 weighted routed partial of the CPU picks (the GPU adds it into the MoE output)
// Weights are read in place from the native pinned trellis (the same home copy the GPU cache /
// zero-copy / staging paths use); suh/svh are CPU copies of the GPU's tensors.
//
// [rt] Remote expert tier (a second machine's GPUs over TCP): when tier_remote() has been called, picks whose
// expert is resident on the remote expert server (a per-layer bitmap) are sent there over one persistent TCP_NODELAY
// connection BEFORE the local CPU work starts; the reply (fp32 [ntok, H] weighted partial) is added into hout after
// the local experts. Any failure (send/recv error, bad reply, timeout GLM53_RT_TIMEOUT_US) closes the connection and
// computes those picks locally, so a contribution is never dropped; a background thread reconnects. Wire format:
// ops/remote-tier/PROTOCOL.md.
#include <torch/extension.h>
#include "ft_core.h"
#ifdef FT_ENABLE_LUT
#include "ft_lut.h"
#include <limits>
#endif
#include <arpa/inet.h>
#include <netdb.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <poll.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <cerrno>
#include <mutex>

namespace ft {

struct Stats { std::atomic<long> jobs{0}, experts{0}, tokens{0}, empty{0}; std::atomic<long long> busy_ns{0}, lat_ns{0}; };

// [rt] remote expert tier client state (worker thread uses the socket; the reconnect thread only creates it)
struct RtHdr { char magic[4]; uint32_t seq; int32_t layer, ntok, npicks, flags; };
static_assert(sizeof(RtHdr) == 24, "RtHdr");
// [E5] GRT2 (dynamic server, T-18 "E5 build spec"): request adds the client's applied residency generation + demand
// hints; the reply adds the server generation, residency deltas since client_gen and a per-pick refused mask
struct RtHdr2 { char magic[4]; uint32_t seq; int32_t layer, ntok, npicks, flags; uint32_t client_gen; uint16_t n_hint, pad; };
static_assert(sizeof(RtHdr2) == 32, "RtHdr2");
struct RtRep2 { char magic[4]; uint32_t seq; int32_t layer, ntok, status, flags; uint32_t server_gen; uint16_t n_delta, delta_full; uint64_t refused_mask; };
static_assert(sizeof(RtRep2) == 40, "RtRep2");
struct Remote
{
    std::string host; int port = 0;
    std::atomic<int> fd{-1};
    std::vector<std::vector<uint8_t>> res;   // [li][e] 1 = resident on the server
    std::vector<int> mlayer;                 // li -> model layer index (what the server keys on)
    long long timeout_ns = 20000000;
    std::atomic<long> check{0};              // GLM53_RT_CHECK=N: every Nth remote call also computes its picks here (rel RMS diff)
    uint32_t seq = 0;
    std::vector<int32_t> rpicks;
    std::vector<float> rbuf, tmp;
    std::thread th; std::atomic<bool> quit{false};
    std::atomic<long> calls{0}, picks{0}, fallbacks{0}, fallback_picks{0}, connects{0}, errors{0}, timeouts{0}, refused{0};
    std::atomic<long long> rtt_ns{0}, wait_ns{0}, rtt_max_ns{0};
#ifdef FT_ENABLE_LUT
    std::mutex chk_mu;
#endif
    double chk_d2 = 0, chk_r2 = 0; long chk_n = 0;   // worker-thread only: sum |remote-local|^2, sum |local|^2, calls
    std::atomic<bool> enabled{true};          // runtime switch (tier_remote_set): false = every pick computed here
    // [E5] GRT2. version is set by whoever opened the current connection (GRTC v2 probe); the worker re-syncs the full
    // residency map after every (re)connect by sending client_gen = ~0. res is written only by the worker (deltas) and
    // read by the worker without the lock; res_mu orders it against tier_remote_res() (Python mask sync).
    std::atomic<int> version{1};
    std::atomic<bool> resync{true};
    uint32_t client_gen = 0xFFFFFFFFu;
    std::vector<int16_t> hints;                    // (model_layer, expert) pairs computed here because not resident
    std::vector<int> li_of_ml;                     // model layer -> li (-1 = none)
    std::vector<int16_t> dbuf;
    std::mutex res_mu;
    std::atomic<long> deltas{0}, full_maps{0}, partial{0}, hints_sent{0};
};

// [rt] per-job histogram by number of LOCAL picks (bucket 16 = 16+): jobs, wall ns of run_job, ns of local compute,
// ns waiting for the remote reply after local compute (T-18 hypotheses H3/H4). Written by the worker thread only.
struct JobHist { std::atomic<long long> n[17], busy[17], local[17], wait[17]; };

struct Tier
{
    Pool pool;
#ifdef FT_ENABLE_LUT
    std::unique_ptr<ft_lut::Backend> lut;
    bool lut_fixture = false, lut_test_ingress = false;
    uint64_t lut_timeout_ms = 30000;
    double lut_deadline = 0;
    int64_t lut_generation = 0;
    std::atomic<bool> lut_active{false};
    std::atomic<bool> lut_sticky{false};
    // Serializes a whole coordinator job (including remote work) with grouped
    // prefill; the backend's own try-lock also protects its compute scratch.
    std::mutex lut_job_mutex;
    std::atomic<unsigned> lut_waiters{0};
    std::atomic<long> lut_failed{0}, lut_completed{0};
    std::mutex lut_error_mutex;
    std::string lut_error;
#endif
    std::vector<Layer> layers;
    std::vector<torch::Tensor> keep;
    volatile int64_t* ctrl = nullptr;
    const uint16_t* hx = nullptr;
    const int32_t* picks = nullptr;
    float* hout = nullptr;
    int mode = -1, H = 4096, I = 2048, swz = 1;
    size_t swz_bytes = 0;
    std::thread th;
    std::atomic<bool> quit{false}, running{false};
    Stats st;
    // [bdw-opt] per-phase wall ns of moe_forward + prep (fp16->fp32 conversion + route build), summed over jobs.
    // Atomic: the worker thread adds, /stats (Python thread) reads concurrently — plain doubles here were a data race
    // (Codex audit 2026-10-03). Relaxed is enough: each counter is an independent monotone sum.
    std::atomic<int64_t> pt_ns[5] = { 0, 0, 0, 0, 0 };
    std::atomic<int64_t> prep_ns{0};
    std::vector<float> xf;
    Remote* rt = nullptr;   // [rt]
    JobHist jh;             // [rt]
};

Tier* T = nullptr;
#ifdef FT_ENABLE_LUT
static std::mutex lut_lifecycle_mutex;
// A caller may arrive with or without the GIL (remote/stop release it). Never
// wait on the lifecycle mutex while retaining the GIL: full-bank ingest releases
// and reacquires it while this mutex protects unpublished runtime ownership.
struct LutLifecycleLock {
    std::unique_lock<std::mutex> lock{lut_lifecycle_mutex, std::defer_lock};
    LutLifecycleLock() {
        if (PyGILState_Check()) { pybind11::gil_scoped_release release; lock.lock(); }
        else lock.lock();
    }
};
#endif

void tier_init(int64_t threads, std::vector<int64_t> cpus, torch::Tensor ctrl, torch::Tensor hx, torch::Tensor picks, torch::Tensor hout,
               int64_t mode, int64_t H, int64_t I, int64_t nlayers, int64_t swz)
{
#ifdef FT_ENABLE_LUT
    LutLifecycleLock lifecycle;
#endif
    TORCH_CHECK(T == nullptr, "tier already initialised");
    init_perm(); init_tables();
    T = new Tier();
    std::vector<int> c(cpus.begin(), cpus.end());
    cpu_set_t saved; sched_getaffinity(0, sizeof(saved), &saved);
    T->pool.start(int(threads), c);          // pins the calling thread to c[0]; the worker thread takes that role
    sched_setaffinity(0, sizeof(saved), &saved);   // give the Python thread its affinity back
    T->ctrl = reinterpret_cast<volatile int64_t*>(ctrl.data_ptr<int64_t>());
    T->hx = reinterpret_cast<const uint16_t*>(hx.data_ptr());
    T->picks = picks.data_ptr<int32_t>();
    T->hout = hout.data_ptr<float>();
    T->keep = { ctrl, hx, picks, hout };
    T->mode = int(mode); T->H = int(H); T->I = int(I); T->swz = int(swz);
    T->layers.resize(nlayers);
}

// ptrs: int64 [E, 3] host addresses of gate/up/down trellis (native layout); scales fp16 CPU [E, k] / [E, n]
void tier_add_layer(int64_t li, torch::Tensor ptrs, torch::Tensor suh_g, torch::Tensor svh_g, torch::Tensor suh_u, torch::Tensor svh_u,
                    torch::Tensor suh_d, torch::Tensor svh_d)
{
#ifdef FT_ENABLE_LUT
    LutLifecycleLock lifecycle;
#endif
    TORCH_CHECK(T, "init first");
#ifdef FT_ENABLE_LUT
    TORCH_CHECK(!T->lut, "LUT startup cannot register a second trellis CPU bank");
#endif
    const int E = int(ptrs.size(0)), H = T->H, I = T->I;
    for (auto& t : { suh_g, svh_g, suh_u, svh_u, suh_d, svh_d })
        TORCH_CHECK(t.device().is_cpu() && t.scalar_type() == at::kHalf && t.is_contiguous() && t.size(0) == E, "scales: CPU fp16 contiguous [E, .]");
    TORCH_CHECK(suh_g.size(1) == H && svh_g.size(1) == I && suh_d.size(1) == I && svh_d.size(1) == H, "scale shapes");
    auto p = ptrs.accessor<int64_t, 2>();
    Layer& L = T->layers[li];
    L.H = H; L.I = I; L.ex.resize(E);
    auto u16 = [](const torch::Tensor& t, int e) { return reinterpret_cast<const uint16_t*>(t.data_ptr()) + size_t(e) * t.size(1); };
    for (int e = 0; e < E; ++e)
    {
        L.ex[e].g = { reinterpret_cast<const uint8_t*>(p[e][0]), u16(suh_g, e), u16(svh_g, e), H, I };
        L.ex[e].u = { reinterpret_cast<const uint8_t*>(p[e][1]), u16(suh_u, e), u16(svh_u, e), H, I };
        L.ex[e].d = { reinterpret_cast<const uint8_t*>(p[e][2]), u16(suh_d, e), u16(svh_d, e), I, H };
    }
    for (auto& t : { ptrs, suh_g, svh_g, suh_u, svh_u, suh_d, svh_d }) T->keep.push_back(t);
    if (T->swz)
    {
        // CPU-side block-contiguous copy ([n/128][k/16][8 tiles]): every 128-output work unit streams one
        // contiguous run (C004: 0.114-0.120 vs 0.165-0.172 ms/expert on the native order). Plain (unpinned)
        // huge-page memory; the pinned native home copy stays the GPU's.
        const size_t mb = size_t(H / 16) * (I / 16) * TILE_BYTES;           // bytes per matrix (all three equal)
        uint8_t* buf = static_cast<uint8_t*>(big_alloc(mb * 3 * E));
        T->swz_bytes += mb * 3 * E;
        std::vector<std::thread> ts;
        const int nt = 16;
        for (int w = 0; w < nt; ++w)
            ts.emplace_back([&, w] {
                for (int e = w; e < E; e += nt)
                    for (int k = 0; k < 3; ++k)
                    {
                        Mat& m = k == 0 ? L.ex[e].g : k == 1 ? L.ex[e].u : L.ex[e].d;
                        uint8_t* dst = buf + (size_t(e) * 3 + k) * mb;
                        const int tk_n = m.k / 16, tn = m.n / 16;
                        for (int b = 0; b < tn / 8; ++b)
                            for (int tk = 0; tk < tk_n; ++tk)
                                std::memcpy(dst + (size_t(b) * tk_n + tk) * 768, m.tr + (size_t(tk) * tn + b * 8) * TILE_BYTES, 768);
                        m.tr = dst; m.swz = 1;
                    }
            });
        for (auto& t : ts) t.join();
    }
}

#ifdef FT_ENABLE_LUT
// Mapped ingress extension: [5] sticky abort, [6] sticky error code,
// [7] failed generation, [8] drained generation. [0..4] retain their ABI.
static bool lut_aborted()
{
    return T && T->lut && (T->lut_sticky.load(std::memory_order_acquire) ||
        (T->ctrl && (__atomic_load_n(const_cast<int64_t*>(T->ctrl + 5), __ATOMIC_ACQUIRE) ||
                     __atomic_load_n(const_cast<int64_t*>(T->ctrl + 6), __ATOMIC_ACQUIRE))));
}
static void lut_fail(int64_t generation, int code, const std::string& message)
{
    { std::lock_guard<std::mutex> guard(T->lut_error_mutex); if (T->lut_error.empty()) T->lut_error = message; }
    T->lut_sticky.store(true, std::memory_order_release);
    if (T->ctrl)
    {
        __atomic_store_n(const_cast<int64_t*>(T->ctrl + 5), int64_t(1), __ATOMIC_RELEASE);
        __atomic_store_n(const_cast<int64_t*>(T->ctrl + 7), generation, __ATOMIC_RELEASE);
        __atomic_store_n(const_cast<int64_t*>(T->ctrl + 6), int64_t(code), __ATOMIC_RELEASE);
    }
    T->lut->request_abort();
}
static void lut_checkpoint()
{
    TORCH_CHECK(!lut_aborted() && !T->quit.load(std::memory_order_relaxed), "LUT request cancelled; ingress remains blocked");
    TORCH_CHECK(now() < T->lut_deadline, "LUT coordinator job deadline expired");
}
#endif

// [rt] socket helpers: full write (MSG_NOSIGNAL) and a busy-polled full read bounded by an absolute deadline
static bool rt_writev(int fd, struct iovec* iov, int n)
{
    while (n > 0)
    {
#ifdef FT_ENABLE_LUT
        if (T && T->lut && (lut_aborted() || T->quit.load(std::memory_order_relaxed) || now() >= T->lut_deadline)) return false;
#endif
        struct msghdr mh{}; mh.msg_iov = iov; mh.msg_iovlen = n;
        ssize_t w = sendmsg(fd, &mh, MSG_NOSIGNAL);
        if (w <= 0) return false;
        while (n > 0 && size_t(w) >= iov->iov_len) { w -= iov->iov_len; ++iov; --n; }
        if (n > 0) { iov->iov_base = static_cast<char*>(iov->iov_base) + w; iov->iov_len -= w; }
    }
    return true;
}
static int rt_read(int fd, void* buf, size_t len, double deadline)   // 1 ok, 0 timeout, -1 error
{
    char* p = static_cast<char*>(buf);
    while (len)
    {
#ifdef FT_ENABLE_LUT
        if (T && T->lut && (lut_aborted() || T->quit.load(std::memory_order_relaxed) || now() >= std::min(deadline, T->lut_deadline))) return 0;
#endif
        ssize_t r = recv(fd, p, len, MSG_DONTWAIT);
        if (r > 0) { p += r; len -= r; continue; }
        if (r == 0 || (errno != EAGAIN && errno != EWOULDBLOCK)) return -1;
        if (now() > deadline) return 0;
        _mm_pause();
    }
    int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_QUICKACK, &one, sizeof one);
    return 1;
}
static void rt_drop(Remote* R) { int f = R->fd.exchange(-1); if (f >= 0) close(f); }

static int rt_connect(const std::string& host, int port, int* version = nullptr)
{
#ifdef FT_ENABLE_LUT
    const bool bounded_lut = T && T->lut;
    const double handshake_deadline = now() + 2.0;
#endif
    struct addrinfo hints{}, *ai = nullptr; hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
    if (getaddrinfo(host.c_str(), std::to_string(port).c_str(), &hints, &ai) || !ai) return -1;
    int fd = socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK, 0);
    int rc = fd < 0 ? -1 : connect(fd, ai->ai_addr, ai->ai_addrlen);
    freeaddrinfo(ai);
    if (fd < 0) return -1;
    if (rc < 0 && errno != EINPROGRESS) { close(fd); return -1; }
    struct pollfd pf{ fd, POLLOUT, 0 };
    int err = 0; socklen_t el = sizeof err;
    if (poll(&pf, 1, 500) != 1 || getsockopt(fd, SOL_SOCKET, SO_ERROR, &err, &el) || err) { close(fd); return -1; }
    int one = 1; setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof one);
    int buf = 1 << 20; setsockopt(fd, SOL_SOCKET, SO_SNDBUF, &buf, sizeof buf); setsockopt(fd, SOL_SOCKET, SO_RCVBUF, &buf, sizeof buf);
    // handshake: a GRTP ping must come back before the connection is used — through the relay a TCP connect succeeds
    // even when the expert server is down (the relay then closes), which would cost one failed job per reconnect
    RtHdr ph{ { 'G', 'R', 'T', 'P' }, 0, 0, 0, 0, 0 }, pr{};
    if (send(fd, &ph, sizeof ph, MSG_NOSIGNAL) != (ssize_t) sizeof ph) { close(fd); return -1; }
    size_t got = 0;
    while (got < sizeof pr)
    {
#ifdef FT_ENABLE_LUT
        if (bounded_lut && (now() >= handshake_deadline || T->quit.load(std::memory_order_relaxed))) { close(fd); return -1; }
#endif
        struct pollfd rf{ fd, POLLIN, 0 };
        if (poll(&rf, 1, 300) != 1) { close(fd); return -1; }
        ssize_t r = recv(fd, reinterpret_cast<char*>(&pr) + got, sizeof pr - got, MSG_DONTWAIT);
        if (r <= 0) { if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) continue; close(fd); return -1; }
        got += size_t(r);
    }
    if (std::memcmp(pr.magic, "GRTP", 4)) { close(fd); return -1; }
    if (version)
    {
        // [E5] which protocol does this server speak? GRTC returns "GRTC" + u32 len + JSON; version 2 = dynamic (GRT2)
        *version = 1;
        RtHdr ch{ { 'G', 'R', 'T', 'C' }, 0, 0, 0, 0, 0 };
        if (send(fd, &ch, sizeof ch, MSG_NOSIGNAL) != (ssize_t) sizeof ch) { close(fd); return -1; }
        std::string buf; size_t need = 8;
        while (buf.size() < need)
        {
#ifdef FT_ENABLE_LUT
            if (bounded_lut && (now() >= handshake_deadline || T->quit.load(std::memory_order_relaxed))) { close(fd); return -1; }
#endif
            struct pollfd rf{ fd, POLLIN, 0 };
            if (poll(&rf, 1, 2000) != 1) { close(fd); return -1; }
            char tmp[65536];
            ssize_t r = recv(fd, tmp, std::min(sizeof tmp, need - buf.size()), MSG_DONTWAIT);
            if (r <= 0) { if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) continue; close(fd); return -1; }
            buf.append(tmp, size_t(r));
            if (buf.size() == 8 && need == 8)
            {
                if (buf.compare(0, 4, "GRTC")) { close(fd); return -1; }
#ifdef FT_ENABLE_LUT
                uint32_t len; std::memcpy(&len, buf.data() + 4, 4);
                if (bounded_lut && len > (1u << 20)) { close(fd); return -1; }
                need = 8 + size_t(len);
#else
                uint32_t len; std::memcpy(&len, buf.data() + 4, 4); need = 8 + size_t(len);
#endif
            }
        }
        if (buf.find("\"version\": 2") != std::string::npos || buf.find("\"version\":2") != std::string::npos) *version = 2;
    }
    return fd;
}

#ifdef FT_ENABLE_LUT
static bool lut_convert_finite(const uint16_t* hx, float* xf, size_t n)
{
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 inf = _mm256_set1_ps(INFINITY);
    __m256 bad = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
    {
        const __m256 value = _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i*>(hx + i)));
        _mm256_storeu_ps(xf + i, value);
        bad = _mm256_or_ps(bad, _mm256_cmp_ps(_mm256_and_ps(value, absmask), inf, _CMP_NLT_UQ));
    }
    bool ok = _mm256_movemask_ps(bad) == 0;
    for (; i < n; ++i) { xf[i] = h2f(hx[i]); ok &= std::isfinite(xf[i]); }
    return ok;
}
static bool lut_all_finite(const float* x, size_t n)
{
    const __m256 absmask = _mm256_castsi256_ps(_mm256_set1_epi32(0x7fffffff));
    const __m256 inf = _mm256_set1_ps(INFINITY);
    __m256 bad = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 8 <= n; i += 8)
        bad = _mm256_or_ps(bad, _mm256_cmp_ps(_mm256_and_ps(_mm256_loadu_ps(x + i), absmask), inf, _CMP_NLT_UQ));
    bool ok = _mm256_movemask_ps(bad) == 0;
    for (; i < n; ++i) ok &= std::isfinite(x[i]);
    return ok;
}
#endif

// Compute `route` on the CPU into out (zeroing it when the route is empty).
// finite_input is used only after run_job's checked ingress conversion.
static void local_forward(int li, int ntok, const std::vector<std::vector<std::pair<int, float>>>& route, int nr, float* out, PhaseTimes* lp, bool finite_input = false)
{
#ifdef FT_ENABLE_LUT
    if (T->lut)
    {
        const bool active = T->lut_active.load(std::memory_order_acquire);
        if (active) lut_checkpoint();
        const uint64_t remaining_ms = active ? uint64_t(std::max(1.0, std::ceil((T->lut_deadline - now()) * 1000.0))) : 0;
        if (finite_input) T->lut->forward_finite_input(li, T->xf.data(), ntok, route, out, remaining_ms);
        else T->lut->forward(li, T->xf.data(), ntok, route, out, remaining_ms);
        if (active) lut_checkpoint();
        return;
    }
#endif
    if (nr == 0) { std::memset(out, 0, size_t(ntok) * T->H * 4); return; }
    int maxm = 0;
    { std::vector<int> cnt(T->layers[li].ex.size(), 0); for (auto& r : route) for (auto& pr : r) maxm = std::max(maxm, ++cnt[pr.first]); }
    const int mode = T->mode >= 0 ? T->mode : (maxm <= 2 ? 2 : 0);
    moe_forward(T->pool, T->layers[li], T->xf.data(), ntok, route, out, mode, lp);
}

void run_job(int li, int ntok, int np)
{
    const double tp0 = now();
    const int H = T->H;
#ifdef FT_ENABLE_LUT
    if (T->lut)
    {
        lut_checkpoint();
        TORCH_CHECK(li >= 0 && li < int(T->layers.size()) && ntok >= 1 && ntok <= 4 && np >= 0 && np <= ntok * 8,
                    "LUT ingress layer/token/pick bounds violated");
        std::array<int, 4> count{};
        std::array<std::array<bool, 288>, 4> seen{};
        const int experts = int(T->layers[size_t(li)].ex.size());
        for (int k = 0; k < np; ++k)
        {
            const int t = T->picks[k * 3], e = T->picks[k * 3 + 1]; float weight;
            std::memcpy(&weight, &T->picks[k * 3 + 2], 4);
            TORCH_CHECK(t >= 0 && t < ntok && e >= 0 && e < experts && std::isfinite(weight) && weight >= 0,
                        "LUT ingress contains an invalid route");
            TORCH_CHECK(++count[size_t(t)] <= 8 && !seen[size_t(t)][size_t(e)], "LUT ingress duplicate route or too many picks per token");
            seen[size_t(t)][size_t(e)] = true;
        }
    }
#endif
    T->xf.resize(size_t(std::max(ntok, 1)) * H);
#ifdef FT_ENABLE_LUT
    if (T->lut)
    {
        TORCH_CHECK(lut_convert_finite(T->hx, T->xf.data(), size_t(ntok) * H), "LUT ingress contains nonfinite activation");
    }
    else
#endif
    for (size_t i = 0; i < size_t(ntok) * H; ++i) T->xf[i] = h2f(T->hx[i]);
    if (np == 0)
    {
        std::memset(T->hout, 0, size_t(ntok) * H * 4); T->st.empty++;
        T->prep_ns.fetch_add(int64_t((now() - tp0) * 1e9), std::memory_order_relaxed);   // [bdw-opt] count prep for empty jobs too
        return;
    }
    // [rt] split the picks: resident on the remote server -> rroute (sent first), the rest -> route (CPU here)
    Remote* R = T->rt;
    const bool use_rt = R && R->enabled.load(std::memory_order_relaxed) && ntok <= 8 && R->fd.load(std::memory_order_relaxed) >= 0
                        && li < int(R->res.size()) && !R->res[li].empty();
    std::vector<std::vector<std::pair<int, float>>> route(ntok), rroute;
    int nl = 0, nr = 0;
    if (use_rt) { rroute.resize(ntok); R->rpicks.clear(); }
    for (int k = 0; k < np; ++k)
    {
        const int t = T->picks[k * 3], e = T->picks[k * 3 + 1]; float w; std::memcpy(&w, &T->picks[k * 3 + 2], 4);
        if (use_rt && e >= 0 && e < int(R->res[li].size()) && R->res[li][e])
        {
            rroute[t].push_back({ e, w }); ++nr;
            R->rpicks.insert(R->rpicks.end(), { T->picks[k * 3], T->picks[k * 3 + 1], T->picks[k * 3 + 2] });
        }
        else { route[t].push_back({ e, w }); ++nl; }
    }
    // [E5] GRT2: every pick computed here is a demand hint for the dynamic server (sent with the next request)
    const bool v2 = R && R->enabled.load(std::memory_order_relaxed) && ntok <= 8 && R->fd.load(std::memory_order_relaxed) >= 0
                    && R->version.load() == 2;            // (a layer with no resident expert yet still sends hints)
    if (v2 && nl)
    {
        for (int t = 0; t < ntok; ++t)
            for (auto& pr : route[t]) { R->hints.push_back(int16_t(R->mlayer[li])); R->hints.push_back(int16_t(pr.first)); }
        if (R->hints.size() > 2 * 512) R->hints.erase(R->hints.begin(), R->hints.end() - 2 * 512);   // keep the newest
    }
    double t_send = 0; bool sent = false;
    if (nr)
    {
        const int fd = R->fd.load(std::memory_order_relaxed);
        if (v2)
        {
#ifdef FT_ENABLE_LUT
            std::lock_guard<std::mutex> generation_lock(R->chk_mu);
#endif
            if (R->resync.exchange(false)) R->client_gen = 0xFFFFFFFFu;   // after a (re)connect: ask for the full map
            const uint16_t nh = uint16_t(std::min<size_t>(R->hints.size() / 2, 256));
            RtHdr2 h{ { 'G', 'R', 'T', '2' }, ++R->seq, R->mlayer[li], ntok, nr, 0, R->client_gen, nh, 0 };
            struct iovec iov[4] = { { &h, sizeof h }, { const_cast<uint16_t*>(T->hx), size_t(ntok) * H * 2 },
                                    { R->rpicks.data(), R->rpicks.size() * 4 }, { R->hints.data(), size_t(nh) * 4 } };
            t_send = now();
            sent = fd >= 0 && rt_writev(fd, iov, 4);
            if (sent && nh) { R->hints.erase(R->hints.begin(), R->hints.begin() + 2 * nh); R->hints_sent += nh; }
        }
        else
        {
            RtHdr h{ { 'G', 'R', 'T', '1' }, ++R->seq, R->mlayer[li], ntok, nr, 0 };
            struct iovec iov[3] = { { &h, sizeof h }, { const_cast<uint16_t*>(T->hx), size_t(ntok) * H * 2 },
                                    { R->rpicks.data(), R->rpicks.size() * 4 } };
            t_send = now();
            sent = fd >= 0 && rt_writev(fd, iov, 3);
        }
        if (!sent) { R->errors++; rt_drop(R); }
    }
    T->prep_ns.fetch_add(int64_t((now() - tp0) * 1e9), std::memory_order_relaxed);
    PhaseTimes lp;   // [bdw-opt] job-local, then published atomically
    const double tl0 = now();
    local_forward(li, ntok, route, nl, T->hout, &lp, true);
    const double tl1 = now();
    double twait = 0;
    if (nr)
    {
        bool ok = false;                 // remote output received and added (possibly minus refused picks)
        uint64_t rmask = ~0ull;          // picks still to compute here (bit i = rpicks[i]); all of them unless ok
        if (sent)
        {
            const double tw = now();
            const int fd = R->fd.load(std::memory_order_relaxed);
            R->rbuf.resize(size_t(ntok) * H);
            const double dl = t_send + R->timeout_ns * 1e-9;
            int rc;
            bool hdr_ok, refused;
            if (v2)
            {
                RtRep2 rh{};
                rc = fd >= 0 ? rt_read(fd, &rh, sizeof rh, dl) : -1;
                hdr_ok = rc == 1 && !std::memcmp(rh.magic, "GRT2", 4) && rh.seq == R->seq && rh.layer == R->mlayer[li] && rh.ntok == ntok;
                refused = hdr_ok && rh.status != 0;           // whole call refused: header (+ deltas) only, stream in sync
                if (hdr_ok && !refused) rc = rt_read(fd, R->rbuf.data(), size_t(ntok) * H * 4, dl);
                if (hdr_ok && rc == 1 && rh.n_delta) { R->dbuf.resize(size_t(rh.n_delta) * 3); rc = rt_read(fd, R->dbuf.data(), size_t(rh.n_delta) * 6, dl); }
#ifdef FT_ENABLE_LUT
                if (T->lut && hdr_ok && rc == 1 && !refused)
                {
                    const uint64_t valid_mask = nr == 64 ? ~0ull : ((1ull << nr) - 1);
                    if ((rh.refused_mask & ~valid_mask) || !std::all_of(R->rbuf.begin(), R->rbuf.end(), [](float x) { return std::isfinite(x); }))
                    { hdr_ok = false; rc = -1; }
                }
#endif
                if (hdr_ok && rc == 1)
                {
                    if (rh.n_delta || rh.delta_full)
                    {
                        std::lock_guard<std::mutex> g(R->res_mu);
                        if (rh.delta_full) for (auto& row : R->res) std::fill(row.begin(), row.end(), uint8_t(0));
                        for (size_t d = 0; d < rh.n_delta; ++d)
                        {
                            const int ml = R->dbuf[d * 3], e = R->dbuf[d * 3 + 1], op = R->dbuf[d * 3 + 2];
                            const int l = ml >= 0 && ml < int(R->li_of_ml.size()) ? R->li_of_ml[size_t(ml)] : -1;
                            if (l < 0 || e < 0) continue;
                            if (R->res[size_t(l)].empty()) R->res[size_t(l)].assign(T->layers[size_t(l)].ex.size(), 0);
                            if (e < int(R->res[size_t(l)].size())) R->res[size_t(l)][size_t(e)] = op > 0 ? 1 : 0;   // retiring: stop sending it
                        }
                        R->deltas += rh.n_delta; if (rh.delta_full) R->full_maps++;
                    }
#ifdef FT_ENABLE_LUT
                    { std::lock_guard<std::mutex> generation_lock(R->chk_mu); R->client_gen = rh.server_gen; }
#else
                    R->client_gen = rh.server_gen;
#endif
                    if (!refused) { ok = true; rmask = rh.refused_mask; }
                }
                if (rc != 1) refused = false;                 // a short read after the header: stream out of sync -> drop below
            }
            else
            {
                RtHdr rh{};
                rc = fd >= 0 ? rt_read(fd, &rh, sizeof rh, dl) : -1;
                hdr_ok = rc == 1 && !std::memcmp(rh.magic, "GRT1", 4) && rh.seq == R->seq && rh.layer == R->mlayer[li] && rh.ntok == ntok;
                // a refusal is header-only (flags != 0): the stream stays in sync, keep the connection, compute here
                refused = hdr_ok && rh.flags != 0;
                if (hdr_ok && !refused) rc = rt_read(fd, R->rbuf.data(), size_t(ntok) * H * 4, dl);
#ifdef FT_ENABLE_LUT
                if (T->lut && hdr_ok && !refused && rc == 1 &&
                    !std::all_of(R->rbuf.begin(), R->rbuf.end(), [](float x) { return std::isfinite(x); })) { hdr_ok = false; rc = -1; }
#endif
                if (hdr_ok && !refused && rc == 1) { ok = true; rmask = 0; }
            }
            if (refused) R->refused++;
            if (ok)
            {
                const long chk = R->check.load(std::memory_order_relaxed);
                if (chk > 0 && rmask == 0 && R->calls.load(std::memory_order_relaxed) % chk == 0)
                {
                    R->tmp.resize(size_t(ntok) * H);
                    local_forward(li, ntok, rroute, nr, R->tmp.data(), nullptr, true);
#ifdef FT_ENABLE_LUT
                    std::lock_guard<std::mutex> diagnostic_lock(R->chk_mu);
#endif
                    for (size_t i = 0; i < size_t(ntok) * H; ++i)
                    { const double d = double(R->rbuf[i]) - R->tmp[i]; R->chk_d2 += d * d; R->chk_r2 += double(R->tmp[i]) * R->tmp[i]; }
                    R->chk_n++;
                }
                for (size_t i = 0; i < size_t(ntok) * H; ++i) T->hout[i] += R->rbuf[i];
                const double t1 = now();
                twait = t1 - tw;
                const long long rtt = (long long) ((t1 - t_send) * 1e9);
                R->calls++; R->picks += nr; R->rtt_ns += rtt; R->wait_ns += (long long) ((t1 - tw) * 1e9);
                long long m = R->rtt_max_ns.load(); while (rtt > m && !R->rtt_max_ns.compare_exchange_weak(m, rtt)) {}
            }
            else if (!refused) { if (rc == 0) R->timeouts++; else R->errors++; rt_drop(R); }   // close: a late reply can never be misread
        }
        if (rmask)
        {
            // never drop a contribution: compute here every pick the server did not (all of them unless ok; with GRT2
            // partial refusal only the masked ones — rpicks order = the order of the picks we sent)
            std::vector<std::vector<std::pair<int, float>>> fr(static_cast<size_t>(ntok));
            int nf = 0;
            for (int k = 0; k < nr; ++k)
                if (k >= 64 || (rmask >> k) & 1ull)
                {
                    float w; std::memcpy(&w, &R->rpicks[size_t(k) * 3 + 2], 4);
                    fr[size_t(R->rpicks[size_t(k) * 3])].push_back({ R->rpicks[size_t(k) * 3 + 1], w }); ++nf;
                }
            if (nf)
            {
                R->tmp.resize(size_t(ntok) * H);
                local_forward(li, ntok, fr, nf, R->tmp.data(), nullptr, true);
                for (size_t i = 0; i < size_t(ntok) * H; ++i) T->hout[i] += R->tmp[i];
                if (ok) R->partial++; else { R->fallbacks++; R->fallback_picks += nf; }
            }
        }
    }
#ifdef FT_ENABLE_LUT
    if (T->lut)
    {
        lut_checkpoint();
        // The backend already validated a local-only result. Remote/fallback
        // additions need their own check because the final add can overflow.
        if (nr) TORCH_CHECK(lut_all_finite(T->hout, size_t(ntok) * H), "nonfinite LUT routed result");
    }
#endif
    for (int k = 0; k < 5; ++k) T->pt_ns[k].fetch_add(int64_t(lp.t[k] * 1e9), std::memory_order_relaxed);
    T->st.experts += np; T->st.tokens += ntok;
    const int b = std::min(nl, 16);   // [rt] job histogram
    T->jh.n[b].fetch_add(1, std::memory_order_relaxed);
    T->jh.busy[b].fetch_add((long long) ((now() - tp0) * 1e9), std::memory_order_relaxed);
    T->jh.local[b].fetch_add((long long) ((tl1 - tl0) * 1e9), std::memory_order_relaxed);
    T->jh.wait[b].fetch_add((long long) (twait * 1e9), std::memory_order_relaxed);
}

// [rt T-27] CPU-tier SwiGLU clamp (exllamav3 semantics); 0 = off. Call before tier_start.
#ifdef FT_ENABLE_LUT
void tier_set_act_limit(double limit) {
    TORCH_CHECK(!T || !T->lut, "LUT activation limit is fixed by model-bound startup admission");
    ACT_LIMIT = float(limit); }
#else
void tier_set_act_limit(double limit) { ACT_LIMIT = float(limit); }
#endif

// [E5] current residency as uint8 [layers, E] (rows of layers without residency stay 0) — for the device masks (E4 split,
// E3b skip) that cpu_tier refreshes when the dynamic server's deltas change it
torch::Tensor tier_remote_res(int64_t E)
{
    TORCH_CHECK(T && T->rt, "tier_remote_res: remote tier not enabled");
    Remote* R = T->rt;
    auto out = torch::zeros({ int64_t(R->res.size()), E }, torch::kUInt8);
    std::lock_guard<std::mutex> g(R->res_mu);
    for (size_t li = 0; li < R->res.size(); ++li)
        std::memcpy(out.data_ptr<uint8_t>() + li * size_t(E), R->res[li].data(), std::min<size_t>(R->res[li].size(), size_t(E)));
    return out;
}

// [rt] runtime controls: remote on/off and numerics-check period (no restart; T-18 verification)
void tier_remote_set(bool enabled, int64_t check_every)
{
    TORCH_CHECK(T && T->rt, "tier_remote_set: remote tier not enabled");
    T->rt->enabled = enabled;
    if (check_every >= 0) T->rt->check = long(check_every);
}

std::vector<int64_t> tier_job_hist()
{
    std::vector<int64_t> v;
    if (!T) return v;
    for (int b = 0; b < 17; ++b)
        v.insert(v.end(), { T->jh.n[b].load(), T->jh.busy[b].load(), T->jh.local[b].load(), T->jh.wait[b].load() });
    return v;
}

// [rt] enable the remote tier. res[li]: uint8 [E] residency (empty tensor = none for that layer); mlayer[li]: model layer
void tier_remote(std::string host, int64_t port, std::vector<torch::Tensor> res, std::vector<int64_t> mlayer, int64_t timeout_us)
{
#ifdef FT_ENABLE_LUT
    LutLifecycleLock lifecycle;
#endif
    TORCH_CHECK(T && !T->rt, "tier_remote: init first / already enabled");
    TORCH_CHECK(res.size() == T->layers.size() && mlayer.size() == T->layers.size(), "tier_remote: one entry per layer");
#ifdef FT_ENABLE_LUT
    if (T->lut)
    {
        TORCH_CHECK(!T->running && timeout_us >= 1 && timeout_us <= 3600000000LL && port >= 1 && port <= 65535,
                    "LUT remote setup requires a stopped tier and bounded port/timeout");
        for (size_t li = 0; li < res.size(); ++li)
            TORCH_CHECK(res[li].device().is_cpu() && res[li].scalar_type() == at::kByte && res[li].is_contiguous() &&
                        (res[li].numel() == 0 || res[li].numel() == int64_t(T->layers[li].ex.size())) && mlayer[li] == int64_t(li) + 3,
                        "LUT remote residency/model-layer geometry mismatch");
        // Resolve once at startup. Reconnect and teardown never depend on a new
        // potentially slow DNS transaction; the worker socket uses this address.
        struct addrinfo hints{}, *address = nullptr; hints.ai_family = AF_INET; hints.ai_socktype = SOCK_STREAM;
        TORCH_CHECK(getaddrinfo(host.c_str(), nullptr, &hints, &address) == 0 && address, "LUT remote address resolution failed");
        char numeric[INET_ADDRSTRLEN]; const auto* ipv4 = reinterpret_cast<const sockaddr_in*>(address->ai_addr);
        const char* converted = inet_ntop(AF_INET, &ipv4->sin_addr, numeric, sizeof numeric); freeaddrinfo(address);
        TORCH_CHECK(converted, "LUT remote address conversion failed"); host = numeric;
    }
#endif
    Remote* R = new Remote();
    R->host = host; R->port = int(port); R->timeout_ns = timeout_us * 1000;
    { const char* v = std::getenv("GLM53_RT_CHECK"); R->check = v ? std::max(0, std::atoi(v)) : 0; }
    R->res.resize(res.size());
    for (size_t li = 0; li < res.size(); ++li)
    {
        TORCH_CHECK(res[li].scalar_type() == at::kByte && res[li].is_contiguous(), "residency: uint8 contiguous");
        const uint8_t* p = res[li].data_ptr<uint8_t>();
        R->res[li].assign(p, p + res[li].numel());
        R->mlayer.push_back(int(mlayer[li]));
    }
    for (int ml : R->mlayer) { if (ml >= int(R->li_of_ml.size())) R->li_of_ml.resize(size_t(ml) + 1, -1); }
    for (size_t li = 0; li < R->mlayer.size(); ++li) R->li_of_ml[size_t(R->mlayer[li])] = int(li);
    int ver = 1;
    int fd = rt_connect(host, int(port), &ver);
    if (fd >= 0) { R->version = ver; R->resync = true; R->fd = fd; R->connects++; }
    R->th = std::thread([R] {
        while (!R->quit.load())
        {
            for (int i = 0; i < 20 && !R->quit.load(); ++i) std::this_thread::sleep_for(std::chrono::milliseconds(100));
            if (R->fd.load() < 0)
            {
                int ver = 1, f = rt_connect(R->host, R->port, &ver);
                if (f >= 0) { R->version = ver; R->resync = true; int x = -1; if (!R->fd.compare_exchange_strong(x, f)) close(f); else R->connects++; }
            }
        }
    });
    T->rt = R;
}

std::vector<int64_t> tier_remote_stats()
{
    if (!T || !T->rt) return {};
    Remote* R = T->rt;
#ifdef FT_ENABLE_LUT
    std::lock_guard<std::mutex> diagnostic_lock(R->chk_mu);
    const double metric = R->chk_r2 > 0 ? 1e9 * std::sqrt(R->chk_d2 / R->chk_r2) : -1;
    const int64_t bounded_metric = metric >= double(std::numeric_limits<int64_t>::max()) ? std::numeric_limits<int64_t>::max() :
                                   (metric >= 0 && std::isfinite(metric) ? int64_t(metric) : -1);
#endif
    return { R->fd.load() >= 0, R->calls.load(), R->picks.load(), R->fallbacks.load(), R->fallback_picks.load(), R->connects.load(),
             R->errors.load(), R->timeouts.load(), R->rtt_ns.load(), R->wait_ns.load(), R->rtt_max_ns.load(),
#ifdef FT_ENABLE_LUT
             R->chk_n, bounded_metric, R->refused.load(),
#else
             R->chk_n, (int64_t) (R->chk_r2 > 0 ? 1e9 * std::sqrt(R->chk_d2 / R->chk_r2) : -1), R->refused.load(),
#endif
             R->version.load(), R->deltas.load(), R->full_maps.load(), R->partial.load(), R->hints_sent.load(), int64_t(R->client_gen) };   // [E5] 14-19
}

#ifdef FT_ENABLE_LUT
static void lut_worker_main(int cpu0)
{
    Pool::pin(cpu0);
    int64_t last = __atomic_load_n(const_cast<int64_t*>(T->ctrl + 8), __ATOMIC_ACQUIRE);
    while (!T->quit.load(std::memory_order_acquire))
    {
        const int64_t generation = __atomic_load_n(const_cast<int64_t*>(T->ctrl), __ATOMIC_ACQUIRE);
        if (generation == last || lut_aborted()) { _mm_pause(); continue; }
        std::unique_lock<std::mutex> job(T->lut_job_mutex, std::try_to_lock);
        if (!job.owns_lock() || lut_aborted() || T->quit.load(std::memory_order_acquire)) { _mm_pause(); continue; }
        const double started = now();
        T->lut_active.store(true, std::memory_order_release);
        T->lut_generation = generation; T->lut_deadline = started + T->lut_timeout_ms * 1e-3;
        try
        {
            TORCH_CHECK(generation > 0 && generation > last, "LUT ingress generation reuse or wrap is unsupported");
            run_job(int(T->ctrl[2]), int(T->ctrl[3]), int(T->ctrl[4]));
            lut_checkpoint();
            // Publish only complete results. Draining and result completion are
            // distinct: failures publish [8] but never the success word [1].
            __atomic_store_n(const_cast<int64_t*>(T->ctrl + 8), generation, __ATOMIC_RELEASE);
            __atomic_store_n(const_cast<int64_t*>(T->ctrl + 1), generation, __ATOMIC_RELEASE);
            T->lut_completed.fetch_add(1, std::memory_order_relaxed);
        }
        catch (const std::exception& error) { T->lut_failed++; lut_fail(generation, 1, error.what()); }
        catch (...) { T->lut_failed++; lut_fail(generation, 1, "unknown LUT coordinator failure"); }
        // run_job/backend always drain before unwinding. Keep input and output
        // pinned references alive even after this thread stops.
        __atomic_store_n(const_cast<int64_t*>(T->ctrl + 8), generation, __ATOMIC_RELEASE);
        T->lut_active.store(false, std::memory_order_release);
        T->st.busy_ns += (long long)((now() - started) * 1e9); T->st.jobs++;
        last = generation;
    }
}
#endif

void worker_main(int cpu0)
{
    Pool::pin(cpu0);
    int64_t last = T->ctrl[0];
    long idle = 0;
    while (!T->quit.load(std::memory_order_relaxed))
    {
        const int64_t s = __atomic_load_n(const_cast<int64_t*>(T->ctrl), __ATOMIC_ACQUIRE);
        if (s == last)
        {
            if (++idle < 400000) _mm_pause(); else std::this_thread::sleep_for(std::chrono::microseconds(20));
            continue;
        }
        idle = 0;
        const double t0 = now();
        run_job(int(T->ctrl[2]), int(T->ctrl[3]), int(T->ctrl[4]));
        T->st.busy_ns += (long long) ((now() - t0) * 1e9);
        T->st.jobs++;
        last = s;
        __atomic_store_n(const_cast<int64_t*>(T->ctrl + 1), s, __ATOMIC_RELEASE);
    }
}

// [E6] runtime CPU-job schedule switches (ft_core.h): fused 2-run schedule for jobs of <= fuse_maxn experts, lean GEMV
// prefetch. Negative = unchanged. Returns {fuse, fuse_maxn, pf_lean}. Read by the worker at every job (relaxed atomics).
std::vector<int64_t> tier_set_e6(int64_t fuse, int64_t fuse_maxn, int64_t lean)
{
    if (fuse >= 0) g_fuse().store(int(fuse));
    if (fuse_maxn >= 0) g_fuse_maxn().store(int(fuse_maxn));
    if (lean >= 0) pf_lean().store(int(lean));
    return { g_fuse().load(), g_fuse_maxn().load(), pf_lean().load() };
}

void tier_start(int64_t cpu0)
{
#ifdef FT_ENABLE_LUT
    LutLifecycleLock lifecycle;
#endif
    TORCH_CHECK(T && !T->running, "init first / already running");
#ifdef FT_ENABLE_LUT
    if (T->lut)
    {
        TORCH_CHECK(!T->lut_fixture, "offline LUT fixture cannot start a production coordinator");
        const auto receipt = T->lut->receipt();
        TORCH_CHECK(receipt.bank_complete && receipt.placement_verified && T->ctrl, "LUT production bank/ingress is not ready");
        TORCH_CHECK(!lut_aborted() && !T->quit.load(), "LUT coordinator is stopped or failed; restart required");
        cpu_set_t allowed; CPU_ZERO(&allowed);
        TORCH_CHECK(cpu0 >= 0 && cpu0 < CPU_SETSIZE && sched_getaffinity(0, sizeof allowed, &allowed) == 0 && CPU_ISSET(int(cpu0), &allowed),
                    "LUT coordinator CPU is outside allowed affinity");
        T->th = std::thread(lut_worker_main, int(cpu0)); T->running = true; return;
    }
#endif
    T->running = true;
    T->th = std::thread(worker_main, int(cpu0));
}

void tier_stop()
{
#ifdef FT_ENABLE_LUT
    LutLifecycleLock lifecycle;
#endif
    if (!T) return;
#ifdef FT_ENABLE_LUT
    if (T->lut)
    {
        if (T->ctrl) lut_fail(__atomic_load_n(const_cast<int64_t*>(T->ctrl), __ATOMIC_ACQUIRE), 3, "LUT coordinator stopped");
        T->quit.store(true, std::memory_order_release);
        if (T->th.joinable()) T->th.join();
        if (T->rt) { T->rt->quit = true; if (T->rt->th.joinable()) T->rt->th.join(); rt_drop(T->rt); }
        T->lut->shutdown();
        if (T->ctrl) __atomic_store_n(const_cast<int64_t*>(T->ctrl + 8), __atomic_load_n(const_cast<int64_t*>(T->ctrl), __ATOMIC_ACQUIRE), __ATOMIC_RELEASE);
        T->running = false;
        // Retain backend + ingress ownership until process teardown/reset. A
        // device timeout may have returned before the CPU stopped reading.
        return;
    }
#endif
    T->quit = true;
    if (T->th.joinable()) T->th.join();
    if (T->rt) { T->rt->quit = true; if (T->rt->th.joinable()) T->rt->th.join(); rt_drop(T->rt); }   // [rt]
    T->pool.stop();
}

// Synchronous forward for tests (worker must not be running): x fp32 [m, H], sel int32 [m, k], w fp32 [m, k]
torch::Tensor tier_forward(int64_t li, torch::Tensor x, torch::Tensor sel, torch::Tensor w, int64_t mode)
{
    TORCH_CHECK(T && !T->running, "tier_forward: worker running");
#ifdef FT_ENABLE_LUT
    if (T->lut)
    {
        TORCH_CHECK(T->lut_fixture && mode == -1, "LUT fixture activation is fixed at init; use mode=-1");
        TORCH_CHECK(li >= 0 && li < int64_t(T->layers.size()), "LUT fixture layer index out of range");
        TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kFloat && x.is_contiguous() && x.dim() == 2 && x.size(1) == T->H && x.size(0) >= 1 && x.size(0) <= 4,
                    "LUT fixture x must be contiguous CPU float32 [M,H], M=1..4");
        TORCH_CHECK(sel.device().is_cpu() && sel.scalar_type() == at::kInt && sel.is_contiguous() && sel.dim() == 2 && sel.size(0) == x.size(0) && sel.size(1) <= 8,
                    "LUT fixture sel must be contiguous CPU int32 [M,K], K<=8");
        TORCH_CHECK(w.device().is_cpu() && w.scalar_type() == at::kFloat && w.is_contiguous() && w.sizes() == sel.sizes(), "LUT fixture weights must match sel as contiguous CPU float32");
        const int m = int(x.size(0)), k = int(sel.size(1));
        auto out = torch::zeros({m, T->H}, torch::kFloat);
        ft_lut::Routes route(m); int nr = 0;
        auto sa = sel.accessor<int32_t, 2>(); auto wa = w.accessor<float, 2>();
        for (int t = 0; t < m; ++t) for (int j = 0; j < k; ++j)
        {
            TORCH_CHECK(sa[t][j] >= -1 && std::isfinite(wa[t][j]) && wa[t][j] >= 0,
                        "LUT fixture route IDs/weights must be finite and valid");
            if (sa[t][j] == -1) { TORCH_CHECK(wa[t][j] == 0, "LUT fixture masked picks require zero weight"); }
            else { route[t].push_back({sa[t][j], wa[t][j]}); ++nr; }
        }
        T->xf.assign(x.data_ptr<float>(), x.data_ptr<float>() + x.numel());
        local_forward(int(li), m, route, nr, out.data_ptr<float>(), nullptr);
        return out;
    }
#endif
    const int m = int(x.size(0)), k = int(sel.size(1));
    auto out = torch::zeros({ m, T->H }, torch::kFloat);
    std::vector<std::vector<std::pair<int, float>>> route(m);
    auto sa = sel.accessor<int32_t, 2>(); auto wa = w.accessor<float, 2>();
    for (int t = 0; t < m; ++t) for (int j = 0; j < k; ++j) if (sa[t][j] >= 0) route[t].push_back({ sa[t][j], wa[t][j] });
    int maxm = 0;
    { std::vector<int> cnt(T->layers[li].ex.size(), 0); for (auto& r : route) for (auto& pr : r) maxm = std::max(maxm, ++cnt[pr.first]); }
    const int md = mode >= 0 ? int(mode) : (maxm <= 2 ? 2 : 0);
    moe_forward(T->pool, T->layers[li], x.data_ptr<float>(), m, route, out.data_ptr<float>(), md);
    return out;
}

std::vector<int64_t> tier_stats()
{
    if (!T) return {};
    // [bdw-opt] appended (indices 6..10): phase/step 0-4 wall ns (stock path fills 0-3; split1 fills 0-4), 11: prep ns
    return { T->st.jobs.load(), T->st.experts.load(), T->st.tokens.load(), T->st.empty.load(), (int64_t) T->st.busy_ns.load(), (int64_t) T->swz_bytes,
             T->pt_ns[0].load(std::memory_order_relaxed), T->pt_ns[1].load(std::memory_order_relaxed),
             T->pt_ns[2].load(std::memory_order_relaxed), T->pt_ns[3].load(std::memory_order_relaxed),
             T->pt_ns[4].load(std::memory_order_relaxed), T->prep_ns.load(std::memory_order_relaxed) };
}

#ifdef FT_ENABLE_LUT
static ft_lut::LayerInput lut_descriptor(pybind11::dict descriptor)
{
    ft_lut::LayerInput input;
    input.model_layer = descriptor["model_layer"].cast<int>(); input.internal_li = descriptor["internal_li"].cast<int>(); input.path = descriptor["path"].cast<std::string>();
    auto identity = descriptor["identity"].cast<pybind11::dict>();
    input.identity = {identity["dev"].cast<uint64_t>(), identity["ino"].cast<uint64_t>(), identity["size"].cast<uint64_t>(),
                      identity["mtime_ns"].cast<int64_t>(), identity["ctime_ns"].cast<int64_t>()};
    auto tensors = descriptor["tensors"].cast<pybind11::list>(); TORCH_CHECK(tensors.size() == 3, "LUT needs gate/up/down descriptors");
    for (int m = 0; m < 3; ++m)
    {
        auto tensor = tensors[m].cast<pybind11::dict>();
        input.tensors[m] = {tensor["offset"].cast<uint64_t>(), tensor["nbytes"].cast<uint64_t>(), tensor["type_id"].cast<int>(),
                           tensor["width"].cast<int>(), tensor["rows"].cast<int>(), tensor["experts"].cast<int>()};
    }
    return input;
}
static void lut_bind_ingress(Tier& tier, torch::Tensor ctrl, torch::Tensor hx, torch::Tensor picks, torch::Tensor hout, bool fixture)
{
    for (const auto& tensor : {ctrl, hx, picks, hout})
    {
        TORCH_CHECK(tensor.device().is_cpu() && tensor.is_contiguous(), "LUT ingress must be contiguous CPU tensors");
        TORCH_CHECK(fixture ? !tensor.is_pinned() : tensor.is_pinned(),
                    fixture ? "offline coordinator tests require unpinned CPU buffers" : "production LUT ingress requires pinned GPU-mapped buffers");
    }
    TORCH_CHECK(ctrl.scalar_type() == at::kLong && ctrl.numel() >= 16 && hx.scalar_type() == at::kHalf && hx.numel() >= 4 * tier.H &&
                picks.scalar_type() == at::kInt && picks.numel() >= 4 * 8 * 3 && hout.scalar_type() == at::kFloat && hout.numel() >= 4 * tier.H,
                "LUT ingress buffer dtype/capacity mismatch");
    for (int i = 0; i < 16; ++i) TORCH_CHECK(ctrl.data_ptr<int64_t>()[i] == 0, "LUT ingress control must start at zero");
    tier.ctrl = reinterpret_cast<volatile int64_t*>(ctrl.data_ptr<int64_t>());
    tier.hx = reinterpret_cast<const uint16_t*>(hx.data_ptr()); tier.picks = picks.data_ptr<int32_t>(); tier.hout = hout.data_ptr<float>();
    tier.keep = {ctrl, hx, picks, hout};
}
// Python supplies independently SHA-verified complete descriptors and admits
// the model-bound real quality report before entering this allocation boundary.
void tier_init_lut(std::vector<int64_t> cpus, std::string activation, pybind11::list descriptors,
                   torch::Tensor ctrl, torch::Tensor hx, torch::Tensor picks, torch::Tensor hout,
                   double act_limit, int64_t job_timeout_ms)
{
    LutLifecycleLock lifecycle;
    TORCH_CHECK(!T, "tier already initialised");
    TORCH_CHECK(std::isfinite(act_limit) && act_limit > 0 && act_limit <= std::numeric_limits<float>::max(), "LUT activation limit must be finite and positive");
    TORCH_CHECK(job_timeout_ms >= 1 && job_timeout_ms <= 3600000, "LUT coordinator timeout must be 1..3600000 milliseconds");
    std::vector<ft_lut::LayerInput> inputs;
    for (auto descriptor : descriptors) inputs.push_back(lut_descriptor(descriptor.cast<pybind11::dict>()));
    // Complete geometry is checked before pools, pinning or bank allocation.
    ft_lut::bank_plan(inputs);
    auto tier = std::make_unique<Tier>(); tier->H = 4096; tier->I = 2048; tier->swz = 0; tier->lut_timeout_ms = uint64_t(job_timeout_ms);
    lut_bind_ingress(*tier, ctrl, hx, picks, hout, false);
    ft_lut::Options options; options.cpus.assign(cpus.begin(), cpus.end()); options.activation = activation;
    options.act_limit = float(act_limit); options.job_timeout_ms = uint64_t(job_timeout_ms); options.abort_flag = ctrl.data_ptr<int64_t>() + 5;
    // Release the GIL during full-bank ingest; no global runtime is visible until
    // both the native loader and strict placement checks have completed.
    { pybind11::gil_scoped_release release; tier->lut = ft_lut::Backend::staged_bank(options, inputs); }
    tier->layers.resize(42);
    for (auto& layer : tier->layers) { layer.H = 4096; layer.I = 2048; layer.ex.resize(288); }
    T = tier.release();
}
void tier_lut_check()
{
    TORCH_CHECK(T && T->lut, "LUT backend is not initialized");
    if (lut_aborted())
    {
        std::lock_guard<std::mutex> guard(T->lut_error_mutex);
        const int64_t code = T->ctrl ? __atomic_load_n(const_cast<int64_t*>(T->ctrl + 6), __ATOMIC_ACQUIRE) : 1;
        const int64_t generation = T->ctrl ? __atomic_load_n(const_cast<int64_t*>(T->ctrl + 7), __ATOMIC_ACQUIRE) : 0;
        TORCH_CHECK(false, "LUT request aborted (error ", code, ", generation ", generation, "): ",
                    T->lut_error.empty() ? "device combine timeout or ingress failure; restart required" : T->lut_error);
    }
}

bool tier_lut_grouped_ready()
{
    // Readiness is an observation, not a reservation. Both locks are try-locks:
    // Python uses this before masking GPU picks and must get an immediate answer.
    std::unique_lock<std::mutex> lifecycle(lut_lifecycle_mutex, std::try_to_lock);
    if (!lifecycle.owns_lock() || !T || !T->lut || lut_aborted() || T->quit.load(std::memory_order_acquire)) return false;
    std::unique_lock<std::mutex> job(T->lut_job_mutex, std::try_to_lock);
    return job.owns_lock() && T->lut->grouped_ready();
}

std::tuple<bool, std::string, double, int> tier_lut_forward_grouped(
    int64_t layer, torch::Tensor x_f16, torch::Tensor experts, torch::Tensor offsets,
    torch::Tensor row_token, torch::Tensor row_weight, torch::Tensor out, double timeout_ms)
{
    const double started = now();
    auto result = [&](bool ok, const std::string& why, int chunks = 0) {
        return std::make_tuple(ok, why, (now() - started) * 1000.0, chunks);
    };
    // Lifetime ownership must cover the GIL release. A concurrent stop/reset or
    // grouped call is busy, rather than a reason to wait with the GIL held.
    std::unique_lock<std::mutex> lifecycle(lut_lifecycle_mutex, std::try_to_lock);
    if (!lifecycle.owns_lock()) return result(false, "busy");
    if (!T || !T->lut) return result(false, "LUT backend is not initialized");
    if (lut_aborted() || T->quit.load(std::memory_order_acquire)) return result(false, "LUT request aborted; restart required");
    std::unique_lock<std::mutex> job(T->lut_job_mutex, std::try_to_lock);
    if (!job.owns_lock()) return result(false, "busy");
    if (layer < 0 || layer > std::numeric_limits<int>::max()) return result(false, "LUT layer index out of range");
    if (!x_f16.device().is_cpu() || x_f16.scalar_type() != at::kHalf || !x_f16.is_contiguous() ||
        x_f16.dim() != 2 || x_f16.size(1) != T->H || x_f16.size(0) < 1 || x_f16.size(0) > ft_lut::kGroupedMaxTokens)
        return result(false, "grouped x must be contiguous CPU fp16 [T,H], T=1..1024");
    if (!out.device().is_cpu() || out.scalar_type() != at::kFloat || !out.is_contiguous() || out.sizes() != x_f16.sizes())
        return result(false, "grouped out must be contiguous CPU float32 [T,H] matching x");
    auto index_vector = [](const torch::Tensor& value) {
        return value.device().is_cpu() && value.scalar_type() == at::kInt && value.is_contiguous() && value.dim() == 1;
    };
    if (!index_vector(experts) || !index_vector(offsets) || !index_vector(row_token) ||
        !row_weight.device().is_cpu() || row_weight.scalar_type() != at::kFloat || !row_weight.is_contiguous() || row_weight.dim() != 1)
        return result(false, "grouped rows require contiguous CPU int32 vectors and float32 row_weight");
    if (experts.numel() >= std::numeric_limits<int>::max() || row_token.numel() > std::numeric_limits<int32_t>::max() ||
        offsets.numel() != experts.numel() + 1 || row_weight.numel() != row_token.numel() ||
        offsets.data_ptr<int32_t>()[experts.numel()] != row_token.numel())
        return result(false, "grouped offsets and row tensor lengths do not match");
    if (!std::isfinite(timeout_ms) || timeout_ms <= 0 || timeout_ms > 3600000)
        return result(false, "grouped timeout must be finite and in (0,3600000] milliseconds");
    bool ok = false; int chunks = 0; std::string why;
    {
        pybind11::gil_scoped_release release;
        ok = T->lut->forward_grouped(int(layer), reinterpret_cast<const uint16_t*>(x_f16.data_ptr()), int(x_f16.size(0)),
                                    int(experts.numel()), experts.data_ptr<int32_t>(), offsets.data_ptr<int32_t>(),
                                    row_token.data_ptr<int32_t>(), row_weight.data_ptr<float>(), out.data_ptr<float>(), timeout_ms, &why, &chunks);
        if (!ok && T->lut->grouped_runtime_failed())
        {
            T->lut_failed.fetch_add(1, std::memory_order_relaxed);
            const int64_t generation = T->ctrl ? __atomic_load_n(const_cast<int64_t*>(T->ctrl), __ATOMIC_ACQUIRE) : 0;
            lut_fail(generation, 1, why);
        }
    }
    return result(ok, why, chunks);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> tier_lut_build_grouped_rows(
    torch::Tensor sel, torch::Tensor weights, torch::Tensor take, int64_t E)
{
    TORCH_CHECK(E >= 1 && E < std::numeric_limits<int32_t>::max(), "grouped E must fit a positive int32 expert count");
    TORCH_CHECK(sel.device().is_cpu() && sel.scalar_type() == at::kLong && sel.is_contiguous() && sel.dim() == 2 &&
                sel.size(0) <= std::numeric_limits<int32_t>::max() && sel.numel() <= std::numeric_limits<int32_t>::max(),
                "grouped sel must be contiguous CPU int64 [T,K] with int32-sized rows");
    TORCH_CHECK(weights.device().is_cpu() && (weights.scalar_type() == at::kHalf || weights.scalar_type() == at::kFloat) &&
                weights.is_contiguous() && weights.sizes() == sel.sizes(), "grouped weights must match sel as contiguous CPU fp16 or float32");
    TORCH_CHECK(take.device().is_cpu() && take.scalar_type() == at::kByte && take.is_contiguous() && take.dim() == 1 && take.numel() == E,
                "grouped take must be contiguous CPU uint8 [E]");
    const auto* selected = sel.data_ptr<int64_t>(); const auto* mask = take.data_ptr<uint8_t>();
    const bool half = weights.scalar_type() == at::kHalf;
    const auto* wh = half ? reinterpret_cast<const uint16_t*>(weights.data_ptr()) : nullptr;
    const auto* wf = half ? nullptr : weights.data_ptr<float>();
    auto weight_at = [&](int64_t i) { return half ? h2f(wh[i]) : wf[i]; };
    std::vector<int32_t> count(size_t(E), 0), last_token(size_t(E), -1);
    for (int64_t e = 0; e < E; ++e) TORCH_CHECK(mask[e] <= 1, "grouped take entries must be zero or one");
    const int64_t K = sel.size(1);
    for (int64_t t = 0; t < sel.size(0); ++t) for (int64_t k = 0; k < K; ++k)
    {
        const int64_t at = t * K + k, e = selected[at]; const float w = weight_at(at);
        TORCH_CHECK(std::isfinite(w), "grouped routing weight must be finite");
        TORCH_CHECK(e >= -1 && e < E, "grouped expert index out of range");
        if (e == -1) { TORCH_CHECK(w == 0, "grouped masked picks require zero weight"); continue; }
        if (!mask[e]) continue;
        TORCH_CHECK(last_token[size_t(e)] != t, "grouped duplicate expert for token");
        last_token[size_t(e)] = int32_t(t); ++count[size_t(e)];
    }
    int32_t nexp = 0, rows = 0;
    for (int32_t n : count) if (n) { ++nexp; rows += n; }
    const auto indices = torch::TensorOptions().device(torch::kCPU).dtype(torch::kInt);
    auto experts = torch::empty({nexp}, indices), offsets = torch::empty({int64_t(nexp) + 1}, indices);
    auto row_token = torch::empty({rows}, indices), row_weight = torch::empty({rows}, indices.dtype(torch::kFloat));
    auto* ep = experts.data_ptr<int32_t>(); auto* op = offsets.data_ptr<int32_t>();
    std::vector<int32_t> cursor(size_t(E), 0); int32_t next = 0; op[0] = 0;
    for (int64_t e = 0; e < E; ++e) if (count[size_t(e)])
    {
        ep[next] = int32_t(e); cursor[size_t(e)] = op[next]; op[next + 1] = op[next] + count[size_t(e)]; ++next;
    }
    auto* tp = row_token.data_ptr<int32_t>(); auto* wp = row_weight.data_ptr<float>();
    // Token-major scatter into expert prefix sums is stable: each expert's
    // tokens ascend without a sort, in O(T*K + E) time.
    for (int64_t t = 0; t < sel.size(0); ++t) for (int64_t k = 0; k < K; ++k)
    {
        const int64_t at = t * K + k, e = selected[at];
        if (e < 0 || !mask[e]) continue;
        const int32_t row = cursor[size_t(e)]++; tp[row] = int32_t(t); wp[row] = weight_at(at);
    }
    return std::make_tuple(experts, offsets, row_token, row_weight);
}

void tier_lut_abort()
{
    TORCH_CHECK(T && T->lut && T->ctrl, "LUT coordinator ingress is not initialized");
    lut_fail(__atomic_load_n(const_cast<int64_t*>(T->ctrl), __ATOMIC_ACQUIRE), 3, "LUT request explicitly cancelled");
}
// CPU-only fault tests bind ordinary, deliberately non-mapped buffers. This
// cannot start the production path or satisfy bank/quality admission.
void tier_lut_fixture_coordinator_start(int64_t cpu, torch::Tensor ctrl, torch::Tensor hx, torch::Tensor picks, torch::Tensor hout, int64_t timeout_ms)
{
    LutLifecycleLock lifecycle;
    TORCH_CHECK(T && T->lut && T->lut_fixture && !T->running && !T->ctrl, "fresh offline fixture required");
    TORCH_CHECK(timeout_ms >= 1 && timeout_ms <= 3600000, "LUT coordinator timeout out of range");
    lut_bind_ingress(*T, ctrl, hx, picks, hout, true); T->lut_test_ingress = true; T->lut_timeout_ms = uint64_t(timeout_ms);
    T->lut->fixture_set_timeout(uint64_t(timeout_ms));
    cpu_set_t allowed; CPU_ZERO(&allowed);
    TORCH_CHECK(cpu >= 0 && cpu < CPU_SETSIZE && sched_getaffinity(0, sizeof allowed, &allowed) == 0 && CPU_ISSET(int(cpu), &allowed), "offline coordinator CPU is outside allowed affinity");
    T->th = std::thread(lut_worker_main, int(cpu)); T->running = true;
}
// Emulate the device's publication/wait protocol without CUDA or mapped pages.
// Waiting has a combine-style deadline and leaves storage owned on failure.
torch::Tensor tier_lut_fixture_job(int64_t li, torch::Tensor x, torch::Tensor sel, torch::Tensor weights, int64_t sequence, int64_t timeout_ms)
{
    TORCH_CHECK(T && T->lut_fixture && T->lut_test_ingress && T->running, "offline coordinator test is not running");
    unsigned no_waiter = 0;
    TORCH_CHECK(T->lut_waiters.compare_exchange_strong(no_waiter, 1), "only one offline ingress waiter is supported");
    struct Waiter { Tier* tier; ~Waiter() { tier->lut_waiters.store(0, std::memory_order_release); } } waiter{T};
    TORCH_CHECK(x.device().is_cpu() && x.scalar_type() == at::kHalf && x.is_contiguous() && x.dim() == 2 &&
                x.size(0) >= 1 && x.size(0) <= 4 && x.size(1) == T->H, "offline ingress x must be CPU half [M,H], M=1..4");
    TORCH_CHECK(sel.device().is_cpu() && sel.scalar_type() == at::kInt && sel.is_contiguous() && sel.dim() == 2 &&
                sel.size(0) == x.size(0) && sel.size(1) <= 8 && weights.device().is_cpu() && weights.scalar_type() == at::kFloat &&
                weights.is_contiguous() && weights.sizes() == sel.sizes(), "offline ingress route tensors must be CPU int32/float32 [M,K<=8]");
    TORCH_CHECK(sequence > 0 && sequence < std::numeric_limits<int64_t>::max() && timeout_ms >= 1 && timeout_ms <= 3600000,
                "offline ingress generation/deadline out of bounds");
    tier_lut_check();
    const int64_t previous = __atomic_load_n(const_cast<int64_t*>(T->ctrl), __ATOMIC_ACQUIRE);
    TORCH_CHECK(sequence > previous && previous == __atomic_load_n(const_cast<int64_t*>(T->ctrl + 8), __ATOMIC_ACQUIRE),
                "offline ingress generation reused or previous job is still draining");
    std::memcpy(const_cast<uint16_t*>(T->hx), x.data_ptr(), size_t(x.numel()) * 2);
    int np = 0;
    for (int token = 0; token < x.size(0); ++token) for (int slot = 0; slot < sel.size(1); ++slot)
    {
        const int at = token * int(sel.size(1)) + slot;
        if (sel.data_ptr<int32_t>()[at] == -1 && weights.data_ptr<float>()[at] == 0) continue;
        auto* pick = const_cast<int32_t*>(T->picks) + size_t(np++) * 3;
        pick[0] = token; pick[1] = sel.data_ptr<int32_t>()[at]; std::memcpy(pick + 2, weights.data_ptr<float>() + at, 4);
    }
    T->ctrl[2] = li; T->ctrl[3] = x.size(0); T->ctrl[4] = np;
    __atomic_store_n(const_cast<int64_t*>(T->ctrl), sequence, __ATOMIC_RELEASE);
    {
        pybind11::gil_scoped_release release;
        const double deadline = now() + timeout_ms * 1e-3;
        while (__atomic_load_n(const_cast<int64_t*>(T->ctrl + 1), __ATOMIC_ACQUIRE) != sequence && !lut_aborted())
        {
            if (now() >= deadline) { lut_fail(sequence, 2, "offline combine deadline expired"); break; }
            _mm_pause();
        }
    }
    tier_lut_check();
    auto output = torch::empty({x.size(0), int64_t(T->H)}, torch::kFloat);
    std::memcpy(output.data_ptr<float>(), T->hout, size_t(output.numel()) * 4);
    return output;
}

void tier_lut_fixture_fault(int phase, int group, int delay_ms)
{
    TORCH_CHECK(T && T->lut_fixture && !T->running, "fault injection requires a stopped offline fixture");
    if (delay_ms < 0) T->lut->inject_failure(phase, group);
    else T->lut->inject_delay(phase, group, unsigned(delay_ms));
}

// Explicit bounded offline API: it owns neither ingress buffers nor a GPU
// coordinator, and cannot be selected to bypass production startup admission.
void tier_init_lut_fixture(std::vector<int64_t> cpus, std::string activation, int64_t H, int64_t I, int64_t E, bool virtual_tp2, double act_limit)
{
#ifdef FT_ENABLE_LUT
    LutLifecycleLock lifecycle;
#endif
    TORCH_CHECK(T == nullptr, "tier already initialised");
    TORCH_CHECK(H >= 256 && H <= 4096 && H % 256 == 0 && I >= 512 && I <= 2048 && I % 512 == 0 && E >= 1 && E <= 8,
                "LUT offline fixture exceeds bounded geometry");
    ft_lut::Options options; options.cpus.assign(cpus.begin(), cpus.end()); options.activation = activation; options.virtual_tp2 = virtual_tp2;
    TORCH_CHECK(std::isfinite(act_limit) && act_limit > 0.0 && act_limit <= std::numeric_limits<float>::max(),
                "LUT activation limit must be finite and positive");
    options.act_limit = float(act_limit);
    auto runtime = ft_lut::Backend::offline_fixture(options);
    auto tier = std::make_unique<Tier>();
    tier->lut = std::move(runtime); tier->lut_fixture = true; tier->H = int(H); tier->I = int(I); tier->swz = 0;
    tier->layers.resize(42);
    for (auto& layer : tier->layers) { layer.H = int(H); layer.I = int(I); layer.ex.resize(size_t(E)); }
    T = tier.release();
}
void tier_lut_fixture_add_layer(pybind11::dict descriptor)
{
    LutLifecycleLock lifecycle;
    TORCH_CHECK(T && T->lut_fixture && T->lut && !T->running, "offline LUT fixture init required");
    ft_lut::LayerInput input;
    input.model_layer = descriptor["model_layer"].cast<int>(); input.internal_li = descriptor["internal_li"].cast<int>(); input.path = descriptor["path"].cast<std::string>();
    auto identity = descriptor["identity"].cast<pybind11::dict>();
    input.identity = {identity["dev"].cast<uint64_t>(), identity["ino"].cast<uint64_t>(), identity["size"].cast<uint64_t>(),
                      identity["mtime_ns"].cast<int64_t>(), identity["ctime_ns"].cast<int64_t>()};
    auto tensors = descriptor["tensors"].cast<pybind11::list>(); TORCH_CHECK(tensors.size() == 3, "offline LUT fixture needs gate/up/down descriptors");
    for (int m = 0; m < 3; ++m)
    {
        auto tensor = tensors[m].cast<pybind11::dict>();
        input.tensors[m] = {tensor["offset"].cast<uint64_t>(), tensor["nbytes"].cast<uint64_t>(), tensor["type_id"].cast<int>(),
                           tensor["width"].cast<int>(), tensor["rows"].cast<int>(), tensor["experts"].cast<int>()};
        TORCH_CHECK(input.tensors[m].width == (m == 2 ? T->I : T->H) && input.tensors[m].rows == (m == 2 ? T->H : T->I) &&
                    input.tensors[m].experts == int(T->layers[0].ex.size()), "offline LUT fixture disagrees with init dimensions");
    }
    T->lut->add_layer(input);
}
pybind11::dict tier_lut_fixture_stats()
{
    TORCH_CHECK(T && T->lut, "LUT init required");
    const auto receipt = T->lut->receipt(); pybind11::dict result;
    result["scope"] = T->lut_fixture ? "offline fixture only; no serving or promotion admission" : "production LUT bank";
    result["bank_complete"] = receipt.bank_complete;
    result["production_ready"] = !T->lut_fixture && receipt.bank_complete && receipt.placement_verified;
    result["coordinator_running"] = T->running.load(); result["coordinator_active"] = T->lut_active.load();
    result["completed_jobs"] = T->lut_completed.load(); result["failed_jobs"] = T->lut_failed.load();
    result["job_timeout_ms"] = T->lut_timeout_ms;
    result["aborted"] = lut_aborted();
    if (T->ctrl) { result["error_code"] = __atomic_load_n(const_cast<int64_t*>(T->ctrl + 6), __ATOMIC_ACQUIRE);
                   result["failed_generation"] = __atomic_load_n(const_cast<int64_t*>(T->ctrl + 7), __ATOMIC_ACQUIRE);
                   result["drained_generation"] = __atomic_load_n(const_cast<int64_t*>(T->ctrl + 8), __ATOMIC_ACQUIRE); }
    result["format"] = "lut"; result["activation"] = receipt.activation; result["act_limit"] = receipt.act_limit; result["swz_bytes"] = T->swz_bytes;
    result["placement_verified"] = receipt.placement_verified; result["virtual_tp2"] = receipt.virtual_tp2;
    result["payload_bytes"] = receipt.payload_bytes; result["mapped_bytes"] = receipt.mapped_bytes; result["scratch_bytes"] = receipt.scratch_bytes;
    result["scratch_mapped_bytes"] = receipt.scratch_mapped_bytes;
    result["nodes"] = receipt.nodes; result["workers"] = receipt.workers; result["staging_peak_bytes"] = receipt.staging_peak_bytes;
    result["pool_naps"] = receipt.pool_naps; result["timed_out_jobs"] = receipt.timed_out_jobs;
    result["futex_wake_failures"] = receipt.futex_wake_failures; result["futex_wake_errno"] = receipt.futex_wake_errno;
    result["placement_scope"] = "sampled weight slices, pool controls and activation/scratch payloads; metadata not placed";
    pybind11::list regions, controls, scratch;
    for (const auto& row : receipt.regions) {
        pybind11::dict record;
        record["model_layer"] = row.model_layer; record["group"] = row.group; record["node"] = row.node; record["matrix"] = row.matrix;
        record["payload_bytes"] = row.payload_bytes; record["mapped_bytes"] = row.mapped_bytes; record["preferred_policy"] = row.preferred_policy;
        record["policy_errno"] = row.policy_errno; record["policy_mode"] = row.policy_mode; record["policy_nodes"] = row.policy_nodes;
        record["query_errno"] = row.query_errno; record["sampled_pages"] = row.sampled_pages;
        record["off_node_pages"] = row.off_node_pages; record["page_errors"] = row.page_errors; regions.append(record);
    }
    for (const auto& row : receipt.controls) {
        pybind11::dict record;
        record["group"] = row.group; record["node"] = row.node; record["owner_cpu"] = row.owner_cpu; record["cpus"] = row.cpus;
        record["preferred_policy"] = row.preferred_policy; record["policy_errno"] = row.policy_errno;
        record["policy_mode"] = row.policy_mode; record["policy_nodes"] = row.policy_nodes;
        pybind11::list policies;
        for (const auto& policy : row.thread_policies) {
            pybind11::dict item; item["cpu"] = policy.cpu; item["node"] = policy.node; item["mode"] = policy.mode;
            item["error"] = policy.error; item["verified"] = policy.verified; item["nodes"] = policy.nodes; policies.append(item);
        }
        record["thread_policies"] = policies;
        record["worker_policy_failures"] = row.worker_policy_failures; record["query_errno"] = row.query_errno; record["observed_node"] = row.observed_node;
        controls.append(record);
    }
    for (const auto& row : receipt.scratch) {
        pybind11::dict record;
        record["group"] = row.group; record["node"] = row.node; record["query_errno"] = row.query_errno;
        record["sampled_pages"] = row.sampled_pages; record["off_node_pages"] = row.off_node_pages; record["page_errors"] = row.page_errors;
        record["preferred_policy"] = row.preferred_policy; record["policy_mode"] = row.policy_mode; record["policy_nodes"] = row.policy_nodes;
        record["policy_errno"] = row.policy_errno; record["policy_failures"] = row.policy_failures;
        record["payload_bytes"] = row.payload_bytes; record["mapped_bytes"] = row.mapped_bytes;
        scratch.append(record);
    }
    result["regions"] = regions; result["controls"] = controls; result["scratch"] = scratch; return result;
}
uint64_t tier_lut_fixture_flush_weights()
{
    LutLifecycleLock lifecycle;
    TORCH_CHECK(T && T->lut && T->lut_fixture && !T->running && T->lut_waiters.load(std::memory_order_acquire) == 0,
                "weight flushing requires a stopped offline LUT fixture with no API waiter");
    // Lifecycle ownership outlives the GIL release; reset/stop cannot free or
    // shut down this backend until the bounded flush has completed.
    pybind11::gil_scoped_release release;
    return T->lut->fixture_flush_weights();
}
void tier_lut_fixture_reset()
{
    LutLifecycleLock lifecycle;
    TORCH_CHECK(T && T->lut_fixture && !T->running && T->lut_waiters.load(std::memory_order_acquire) == 0,
                "only a stopped offline LUT fixture with no API waiter can reset");
    if (T->rt) { T->rt->quit = true; if (T->rt->th.joinable()) T->rt->th.join(); rt_drop(T->rt); delete T->rt; T->rt = nullptr; }
    T->lut.reset(); delete T; T = nullptr;
}
#endif

}  // namespace ft

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("tier_init", &ft::tier_init);
    m.def("tier_add_layer", &ft::tier_add_layer);
    m.def("tier_start", &ft::tier_start);
#ifdef FT_ENABLE_LUT
    m.def("tier_stop", &ft::tier_stop, pybind11::call_guard<pybind11::gil_scoped_release>());
#else
    m.def("tier_stop", &ft::tier_stop);
#endif
    m.def("tier_forward", &ft::tier_forward);
    m.def("tier_stats", &ft::tier_stats);
#ifdef FT_ENABLE_LUT
    m.def("tier_init_lut", &ft::tier_init_lut, pybind11::arg("cpus"), pybind11::arg("activation"), pybind11::arg("descriptors"),
          pybind11::arg("ctrl"), pybind11::arg("hx"), pybind11::arg("picks"), pybind11::arg("hout"), pybind11::arg("act_limit"), pybind11::arg("job_timeout_ms") = 30000);
    m.def("tier_lut_stats", &ft::tier_lut_fixture_stats);
    m.def("tier_lut_check", &ft::tier_lut_check);
    m.def("tier_lut_grouped_ready", &ft::tier_lut_grouped_ready);
    m.def("tier_lut_forward_grouped", &ft::tier_lut_forward_grouped,
          pybind11::arg("layer"), pybind11::arg("x_f16"), pybind11::arg("experts"), pybind11::arg("offsets"),
          pybind11::arg("row_token"), pybind11::arg("row_weight"), pybind11::arg("out"), pybind11::arg("timeout_ms"));
    m.def("tier_lut_build_grouped_rows", &ft::tier_lut_build_grouped_rows,
          pybind11::arg("sel"), pybind11::arg("w"), pybind11::arg("take"), pybind11::arg("E"));
    m.def("tier_lut_abort", &ft::tier_lut_abort);
    m.def("tier_lut_fixture_coordinator_start", &ft::tier_lut_fixture_coordinator_start);
    m.def("tier_lut_fixture_fault", &ft::tier_lut_fixture_fault);
    m.def("tier_lut_fixture_job", &ft::tier_lut_fixture_job);
    m.def("tier_init_lut_fixture", &ft::tier_init_lut_fixture,
          pybind11::arg("cpus"), pybind11::arg("activation"), pybind11::arg("H"), pybind11::arg("I"),
          pybind11::arg("E"), pybind11::arg("virtual_tp2"), pybind11::arg("act_limit") = 10.0);
    m.def("tier_lut_fixture_add_layer", &ft::tier_lut_fixture_add_layer);
    m.def("tier_lut_fixture_stats", &ft::tier_lut_fixture_stats);
    m.def("tier_lut_fixture_reset", &ft::tier_lut_fixture_reset);
    m.def("tier_lut_fixture_flush_weights", &ft::tier_lut_fixture_flush_weights);
#endif
    m.def("tier_set_e6", &ft::tier_set_e6);
#ifdef FT_ENABLE_LUT
    m.def("tier_remote", &ft::tier_remote, pybind11::call_guard<pybind11::gil_scoped_release>());
#else
    m.def("tier_remote", &ft::tier_remote);              // [rt]
#endif
    m.def("tier_remote_stats", &ft::tier_remote_stats);  // [rt]
    m.def("tier_remote_set", &ft::tier_remote_set);      // [rt]
    m.def("tier_remote_res", &ft::tier_remote_res);      // [E5]
    m.def("tier_job_hist", &ft::tier_job_hist);          // [rt]
    m.def("tier_set_act_limit", &ft::tier_set_act_limit); // [rt T-27]
}
