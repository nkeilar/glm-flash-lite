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
    bool lut_fixture = false;
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

void tier_init(int64_t threads, std::vector<int64_t> cpus, torch::Tensor ctrl, torch::Tensor hx, torch::Tensor picks, torch::Tensor hout,
               int64_t mode, int64_t H, int64_t I, int64_t nlayers, int64_t swz)
{
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
    TORCH_CHECK(T, "init first");
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

// [rt] socket helpers: full write (MSG_NOSIGNAL) and a busy-polled full read bounded by an absolute deadline
static bool rt_writev(int fd, struct iovec* iov, int n)
{
    while (n > 0)
    {
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
            struct pollfd rf{ fd, POLLIN, 0 };
            if (poll(&rf, 1, 2000) != 1) { close(fd); return -1; }
            char tmp[65536];
            ssize_t r = recv(fd, tmp, std::min(sizeof tmp, need - buf.size()), MSG_DONTWAIT);
            if (r <= 0) { if (r < 0 && (errno == EAGAIN || errno == EWOULDBLOCK)) continue; close(fd); return -1; }
            buf.append(tmp, size_t(r));
            if (buf.size() == 8 && need == 8)
            {
                if (buf.compare(0, 4, "GRTC")) { close(fd); return -1; }
                uint32_t len; std::memcpy(&len, buf.data() + 4, 4); need = 8 + size_t(len);
            }
        }
        if (buf.find("\"version\": 2") != std::string::npos || buf.find("\"version\":2") != std::string::npos) *version = 2;
    }
    return fd;
}

// Compute `route` on the CPU into out (zeroing it when the route is empty)
static void local_forward(int li, int ntok, const std::vector<std::vector<std::pair<int, float>>>& route, int nr, float* out, PhaseTimes* lp)
{
#ifdef FT_ENABLE_LUT
    if (T->lut)
    {
        T->lut->forward(li, T->xf.data(), ntok, route, out);
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
    T->xf.resize(size_t(std::max(ntok, 1)) * H);
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
    local_forward(li, ntok, route, nl, T->hout, &lp);
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
                    R->client_gen = rh.server_gen;
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
                if (hdr_ok && !refused && rc == 1) { ok = true; rmask = 0; }
            }
            if (refused) R->refused++;
            if (ok)
            {
                const long chk = R->check.load(std::memory_order_relaxed);
                if (chk > 0 && rmask == 0 && R->calls.load(std::memory_order_relaxed) % chk == 0)
                {
                    R->tmp.resize(size_t(ntok) * H);
                    local_forward(li, ntok, rroute, nr, R->tmp.data(), nullptr);
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
                local_forward(li, ntok, fr, nf, R->tmp.data(), nullptr);
                for (size_t i = 0; i < size_t(ntok) * H; ++i) T->hout[i] += R->tmp[i];
                if (ok) R->partial++; else { R->fallbacks++; R->fallback_picks += nf; }
            }
        }
    }
    for (int k = 0; k < 5; ++k) T->pt_ns[k].fetch_add(int64_t(lp.t[k] * 1e9), std::memory_order_relaxed);
    T->st.experts += np; T->st.tokens += ntok;
    const int b = std::min(nl, 16);   // [rt] job histogram
    T->jh.n[b].fetch_add(1, std::memory_order_relaxed);
    T->jh.busy[b].fetch_add((long long) ((now() - tp0) * 1e9), std::memory_order_relaxed);
    T->jh.local[b].fetch_add((long long) ((tl1 - tl0) * 1e9), std::memory_order_relaxed);
    T->jh.wait[b].fetch_add((long long) (twait * 1e9), std::memory_order_relaxed);
}

// [rt T-27] CPU-tier SwiGLU clamp (exllamav3 semantics); 0 = off. Call before tier_start.
void tier_set_act_limit(double limit) { ACT_LIMIT = float(limit); }

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
    TORCH_CHECK(T && !T->rt, "tier_remote: init first / already enabled");
    TORCH_CHECK(res.size() == T->layers.size() && mlayer.size() == T->layers.size(), "tier_remote: one entry per layer");
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
    return { R->fd.load() >= 0, R->calls.load(), R->picks.load(), R->fallbacks.load(), R->fallback_picks.load(), R->connects.load(),
             R->errors.load(), R->timeouts.load(), R->rtt_ns.load(), R->wait_ns.load(), R->rtt_max_ns.load(),
             R->chk_n, (int64_t) (R->chk_r2 > 0 ? 1e9 * std::sqrt(R->chk_d2 / R->chk_r2) : -1), R->refused.load(),
             R->version.load(), R->deltas.load(), R->full_maps.load(), R->partial.load(), R->hints_sent.load(), int64_t(R->client_gen) };   // [E5] 14-19
}

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
    TORCH_CHECK(T && !T->running, "init first / already running");
#ifdef FT_ENABLE_LUT
    TORCH_CHECK(!T->lut, "LUT GPU coordinator startup is unavailable: reviewed real-activation admission and request-error handling are required; offline API only");
#endif
    T->running = true;
    T->th = std::thread(worker_main, int(cpu0));
}

void tier_stop()
{
    if (!T) return;
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
// Explicit bounded offline API: it owns neither ingress buffers nor a GPU
// coordinator, and cannot be selected to bypass production startup admission.
void tier_init_lut_fixture(std::vector<int64_t> cpus, std::string activation, int64_t H, int64_t I, int64_t E, bool virtual_tp2)
{
    TORCH_CHECK(T == nullptr, "tier already initialised");
    TORCH_CHECK(H >= 256 && H <= 4096 && H % 256 == 0 && I >= 512 && I <= 2048 && I % 512 == 0 && E >= 1 && E <= 8,
                "LUT offline fixture exceeds bounded geometry");
    ft_lut::Options options; options.cpus.assign(cpus.begin(), cpus.end()); options.activation = activation; options.virtual_tp2 = virtual_tp2;
    auto runtime = ft_lut::Backend::offline_fixture(options);
    auto tier = std::make_unique<Tier>();
    tier->lut = std::move(runtime); tier->lut_fixture = true; tier->H = int(H); tier->I = int(I); tier->swz = 0;
    tier->layers.resize(42);
    for (auto& layer : tier->layers) { layer.H = int(H); layer.I = int(I); layer.ex.resize(size_t(E)); }
    T = tier.release();
}
void tier_lut_fixture_add_layer(pybind11::dict descriptor)
{
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
    TORCH_CHECK(T && T->lut_fixture && T->lut, "offline LUT fixture init required");
    const auto receipt = T->lut->receipt(); pybind11::dict result;
    result["scope"] = "offline fixture only; no serving or promotion admission";
    result["format"] = "lut"; result["activation"] = receipt.activation; result["swz_bytes"] = T->swz_bytes;
    result["placement_verified"] = receipt.placement_verified; result["virtual_tp2"] = receipt.virtual_tp2;
    result["payload_bytes"] = receipt.payload_bytes; result["mapped_bytes"] = receipt.mapped_bytes; result["scratch_bytes"] = receipt.scratch_bytes;
    result["nodes"] = receipt.nodes; result["workers"] = receipt.workers; result["staging_peak_bytes"] = receipt.staging_peak_bytes;
    result["placement_scope"] = "sampled weight slices, pool controls and activation/scratch payloads; metadata not placed";
    pybind11::list regions, controls, scratch;
    for (const auto& row : receipt.regions) {
        pybind11::dict record;
        record["model_layer"] = row.model_layer; record["group"] = row.group; record["node"] = row.node; record["matrix"] = row.matrix;
        record["payload_bytes"] = row.payload_bytes; record["mapped_bytes"] = row.mapped_bytes; record["preferred_policy"] = row.preferred_policy;
        record["policy_errno"] = row.policy_errno; record["query_errno"] = row.query_errno; record["sampled_pages"] = row.sampled_pages;
        record["off_node_pages"] = row.off_node_pages; record["page_errors"] = row.page_errors; regions.append(record);
    }
    for (const auto& row : receipt.controls) {
        pybind11::dict record;
        record["group"] = row.group; record["node"] = row.node; record["owner_cpu"] = row.owner_cpu; record["cpus"] = row.cpus;
        record["preferred_policy"] = row.preferred_policy; record["policy_errno"] = row.policy_errno;
        record["worker_policy_failures"] = row.worker_policy_failures; record["query_errno"] = row.query_errno; record["observed_node"] = row.observed_node;
        controls.append(record);
    }
    for (const auto& row : receipt.scratch) {
        pybind11::dict record;
        record["group"] = row.group; record["node"] = row.node; record["query_errno"] = row.query_errno;
        record["sampled_pages"] = row.sampled_pages; record["off_node_pages"] = row.off_node_pages; record["page_errors"] = row.page_errors;
        scratch.append(record);
    }
    result["regions"] = regions; result["controls"] = controls; result["scratch"] = scratch; return result;
}
void tier_lut_fixture_reset()
{
    TORCH_CHECK(T && T->lut_fixture && !T->running && !T->rt, "only a stopped offline LUT fixture can reset");
    T->lut.reset(); delete T; T = nullptr;
}
#endif

}  // namespace ft

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("tier_init", &ft::tier_init);
    m.def("tier_add_layer", &ft::tier_add_layer);
    m.def("tier_start", &ft::tier_start);
    m.def("tier_stop", &ft::tier_stop);
    m.def("tier_forward", &ft::tier_forward);
    m.def("tier_stats", &ft::tier_stats);
#ifdef FT_ENABLE_LUT
    m.def("tier_init_lut_fixture", &ft::tier_init_lut_fixture);
    m.def("tier_lut_fixture_add_layer", &ft::tier_lut_fixture_add_layer);
    m.def("tier_lut_fixture_stats", &ft::tier_lut_fixture_stats);
    m.def("tier_lut_fixture_reset", &ft::tier_lut_fixture_reset);
#endif
    m.def("tier_set_e6", &ft::tier_set_e6);
    m.def("tier_remote", &ft::tier_remote);              // [rt]
    m.def("tier_remote_stats", &ft::tier_remote_stats);  // [rt]
    m.def("tier_remote_set", &ft::tier_remote_set);      // [rt]
    m.def("tier_remote_res", &ft::tier_remote_res);      // [E5]
    m.def("tier_job_hist", &ft::tier_job_hist);          // [rt]
    m.def("tier_set_act_limit", &ft::tier_set_act_limit); // [rt T-27]
}
