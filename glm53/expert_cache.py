"""Dynamic GPU expert cache for stock exllamav3 1.5.1 (GLM-5.3-Flash; bit-exact by construction).

Layout: every routed expert's trellis lives in pinned host memory (exl3_tiers.py with GLM53_ZC_VRAM=0, the home copy,
always valid). Each GPU gets one slot arena (all free VRAM after load minus a reserve) shared by all MoE layers on that
GPU. Per layer and forward step, after routing and before the expert kernels:

  ec_step  (1 block, device-side, no host sync): dedup the picks; hits -> CLOCK ref bit + pin; if admission is on
           (picks <= GLM53_EC_ADMIT_MAX, i.e. decode / small batches), each miss takes a CLOCK victim slot (never one
           pinned by this layer's current picks), the victim's pointer-table rows go back to its home (host) address,
           the miss's rows point at the slot, and a copy job is queued
  ec_copy  (grid-stride zero-copy gather): queued experts host -> slot (the same PCIe bytes the kernel would have read)
  exllamav3's own MoE kernels then run unchanged and read each expert through the live pointer tables
  (multi_gate / multi_up / multi_down .ptrs_trellis and BC_BlockSparseMLP's gu_trellis_ptr), so the arithmetic and the
  bytes are identical to an all-VRAM run: only the address differs. Pointer copies exllamav3 takes at load time
  (host-side *_cpu tables, per-expert Linear BCs) keep the home address, which stays valid.
Prefill (large picks) runs with admission off: hits read VRAM, misses read zero-copy, the decode working set survives.

Prefill staging (GLM53_EC_STAGE_GB > 0, per buffer; two buffers per GPU): for a forward with >= GLM53_EC_STAGE_MIN
picks per layer, each layer's NON-cached experts are copied host -> a VRAM staging buffer on a copy stream (copy engine)
one layer ahead, while the previous layer computes; the layer's tables point at the buffer for its MoE and go back to
home afterwards. Buffer overflow -> the rest stays zero-copy. Cached set read once per forward (no admission in prefill).

Env: GLM53_EC=1 (serve.py), GLM53_EC_RESERVE_GB (per GPU, default 2.5), GLM53_EC_ADMIT_MAX (default 64 picks),
     GLM53_EC_WARM (optional JSON {layer_key: [score x E]}: initial fill by score), GLM53_EC_MAX_SLOTS (per GPU cap)
[E3] Background admission (T-19, see the comment above ec_step_k): GLM53_EC_BG=1 (loader thread + ring), GLM53_EC_BG_ON (1),
     GLM53_EC_BG_QMAX (2 experts copying at once), GLM53_EC_BG_PER_CALL (1 per layer call), GLM53_EC_BG_GATE (1: the
     candidate's decayed pick count must beat the victim's), GLM53_EC_BG_DECAY (0.99 per call of the layer),
     GLM53_EC_BG_RING (1024), GLM53_EC_BG_CPU (-1 = unpinned). Candidates are CPU-lane picks (cpu_tier sets m._ct_orig).
     GLM53_EC_BG_SKIP_REMOTE (1, E3b): while the 3090 tier is attached, skip the experts it holds (cpu_tier: m._rt_res).
"""
import json, os, time
import torch

_SRC = r"""
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <atomic>
#include <chrono>
#include <thread>
#include <vector>
#include <pthread.h>
#include <sched.h>

// [E3] background admission (T-19). A CPU-lane miss (picked by the router, masked out of `sel` by ft_split) can be
// admitted OFF the critical path: ec_step takes a CLOCK victim (rows -> home, exactly as a normal admission), marks the
// slot LOADING (pin = -2: never a victim, never a hit; the key keeps its home rows and slotof -1, so ft_split keeps
// routing it to the CPU tier / zero-copy) and logs {key, slot, gen} to a mapped host ring. A host thread (BgLoader)
// copies the expert home -> slot on a low-priority copy stream (copy engine) and then marks landed[slot] = gen. A later
// ec_step of the same layer commits it (rows -> slot, slotof = slot): the row flip is the only publish, and it happens
// only after the bytes have landed. Stream order on the one compute stream guarantees no earlier kernel still reads the
// victim slot when the DMA starts (the host sees the log only after ec_step ran; later kernels read the victim from home).
// bgp: 0 on, 1 qmax (DMAs in flight), 2 per_call, 3 gate (cand freq must beat the victim's), 4 ring entries,
//      5 skip (E3b: while the 3090 tier is attached, keys it holds (bgskip) are never candidates and count as freq 0
//      when they are the victim -- their picks are hidden under the local CPU work, so caching them buys nothing)
// bgf: 0 freq decay per call of the layer. bgc: 0 admitted (ever), 1 landed (ever; land kernel). hc: 0 head (device), 1 tail (host)
// stats: 0 hits 1 misses 2 admitted 3 no_victim 4 evictions | 5 bg_admit 6 - 7 bg_commit 8 bg_gate_reject 9 bg_ring_full
//        10 bg_no_victim 11 bg_qmax 12 bg_abort 13 bg_cands 14 loading_picks
__device__ __forceinline__ void ec_rows(const int64_t* tabs, int l, int e, int64_t g, int64_t u, int64_t d)
{
    int64_t* pg = (int64_t*) tabs[l * 4 + 0]; int64_t* pu = (int64_t*) tabs[l * 4 + 1];
    int64_t* pd = (int64_t*) tabs[l * 4 + 2]; int64_t* gu = (int64_t*) tabs[l * 4 + 3];
    pg[e] = g; pu[e] = u; pd[e] = d;
    if (gu) { gu[2 * e] = g; gu[2 * e + 1] = u; }
}

__global__ void ec_step_k(const int64_t* __restrict__ sel, int n_picks, int first, int li, int E, int S, int admit,
    int* slotof_all, const int64_t* home_all, const int64_t* tabs, int* owner, int* refb, int* pin, int* ctl,
    int* jobs, int maxj, unsigned long long* stats, const long long* __restrict__ cbase, int spc, long long rec,
    long long off_u, long long off_d,
    const int64_t* __restrict__ osel, int bg, int* pend_all, int* lgen, const int* landed, float* freq_all,
    const int* __restrict__ bgp, const float* __restrict__ bgf, unsigned long long* bgc, int* ring, long long* hc, int* bgctl,
    const int* __restrict__ bgskip)
{
    __shared__ unsigned mask[32], omask[32];
    __shared__ int misses[1024];
    __shared__ int cand[64];
    __shared__ float candf[64];
    __shared__ int nmiss, nhit, epoch, ncand, nload;
    int t = threadIdx.x;
    if (t < 32) { mask[t] = 0; omask[t] = 0; }
    if (t == 0) { nmiss = 0; nhit = 0; ncand = 0; nload = 0; epoch = ctl[1] + 1; ctl[1] = epoch; ctl[2] = 0; }
    __syncthreads();
    for (int i = t; i < n_picks; i += blockDim.x) {
        long long e = sel[i] - first;
        if (e >= 0 && e < E) atomicOr(&mask[e >> 5], 1u << (e & 31));
        if (bg) { long long o = osel[i] - first; if (o >= 0 && o < E) atomicOr(&omask[o >> 5], 1u << (o & 31)); }
    }
    __syncthreads();
    int* slotof = slotof_all + (long long) li * E;
    int* pend = pend_all + (long long) li * E;
    if (bg) {   // [E3] commit pass: this layer's landed background loads become resident before the hit pass
        for (int e = t; e < E; e += blockDim.x) {
            int s = pend[e];
            if (s >= 0 && ((volatile const int*) landed)[s] == lgen[s]) {
                long long a = cbase[s / spc] + (long long) (s % spc) * rec;
                ec_rows(tabs, li, e, a, a + off_u, a + off_d);
                slotof[e] = s; pin[s] = 0; refb[s] = 1; pend[e] = -1;
                atomicAdd(&stats[7], 1ull);
            }
        }
        __syncthreads();
    }
    float decay = bg ? bgf[0] : 1.0f;
    int skip = bg ? bgp[5] : 0;
    for (int e = t; e < E; e += blockDim.x) {
        bool in = mask[e >> 5] & (1u << (e & 31));
        if (in) {
            int s = slotof[e];
            if (s >= 0) { refb[s] = 1; pin[s] = epoch; atomicAdd(&nhit, 1); }
            else if (bg && pend[e] >= 0) atomicAdd(&nload, 1);   // loading: stays zero-copy, never admitted twice
            else { int k = atomicAdd(&nmiss, 1); misses[k] = e; }
        }
        if (bg) {   // [E3] decayed pick frequency (all router picks, CPU lane included) + background candidates
            bool picked = omask[e >> 5] & (1u << (e & 31));
            float f = freq_all[(long long) li * E + e] * decay + (picked ? 1.0f : 0.0f);
            freq_all[(long long) li * E + e] = f;
            if (picked && !in && slotof[e] < 0 && pend[e] < 0 && !(skip && bgskip[(long long) li * E + e])) {
                int k = atomicAdd(&ncand, 1);
                if (k < 64) { cand[k] = e; candf[k] = f; }
            }
        }
    }
    __syncthreads();
    if (t != 0) return;
    stats[0] += nhit; stats[1] += nmiss + nload; stats[14] += nload; stats[13] += ncand;
    if (!admit || S == 0) return;
    int hand = ctl[0], nj = 0;
    int64_t* pg = (int64_t*) tabs[li * 4 + 0]; int64_t* pu = (int64_t*) tabs[li * 4 + 1];
    int64_t* pd = (int64_t*) tabs[li * 4 + 2]; int64_t* gu = (int64_t*) tabs[li * 4 + 3];
    for (int k = 0; k < nmiss && nj < maxj; k++) {
        int e = misses[k], found = -1;
        for (int step = 0; step < 2 * S; step++) {
            int s = hand; hand = (hand + 1 == S) ? 0 : hand + 1;
            if (pin[s] == epoch || pin[s] < 0) continue;   // pinned by this step, or disabled (elastic)
            if (refb[s]) { refb[s] = 0; continue; }
            found = s; break;
        }
        if (found < 0) { stats[3]++; break; }
        int s = found, old = owner[s];
        if (old >= 0) {
            int ol = old / E, oe = old % E;
            slotof_all[(long long) ol * E + oe] = -1;
            const int64_t* h = home_all + ((long long) ol * E + oe) * 3;
            int64_t* og = (int64_t*) tabs[ol * 4 + 0]; int64_t* ou = (int64_t*) tabs[ol * 4 + 1];
            int64_t* od = (int64_t*) tabs[ol * 4 + 2]; int64_t* ogu = (int64_t*) tabs[ol * 4 + 3];
            og[oe] = h[0]; ou[oe] = h[1]; od[oe] = h[2];
            if (ogu) { ogu[2 * oe] = h[0]; ogu[2 * oe + 1] = h[1]; }
            stats[4]++;
        }
        owner[s] = li * E + e; slotof[e] = s; refb[s] = 1; pin[s] = epoch;
        long long a = cbase[s / spc] + (long long) (s % spc) * rec;
        pg[e] = a; pu[e] = a + off_u; pd[e] = a + off_d;
        if (gu) { gu[2 * e] = a; gu[2 * e + 1] = a + off_u; }
        jobs[2 * nj] = e; jobs[2 * nj + 1] = s; nj++;
        stats[2]++;
    }
    // [E3] background admission of the warmest CPU-lane misses (never blocks: any limit -> skip, counted)
    int nc = ncand < 64 ? ncand : 64;
    if (bg && bgp[0] && nc > 0) {
        int qmax = bgp[1], per = bgp[2], gate = bgp[3], R = bgp[4], done = 0;
        long long head = hc[0];
        while (done < per) {
            int bi = -1; float bf = -1.0f;
            for (int k = 0; k < nc; k++) if (cand[k] >= 0 && candf[k] > bf) { bf = candf[k]; bi = k; }
            if (bi < 0) break;
            int e = cand[bi]; cand[bi] = -1;
            if ((long long) (bgc[0] - ((volatile unsigned long long*) bgc)[1]) >= qmax) { stats[11]++; break; }
            if (head - ((volatile long long*) hc)[1] >= R) { stats[9]++; break; }
            int found = -1;
            for (int step = 0; step < 2 * S; step++) {
                int s = hand; hand = (hand + 1 == S) ? 0 : hand + 1;
                if (pin[s] == epoch || pin[s] < 0) continue;
                if (refb[s]) { refb[s] = 0; continue; }
                found = s; break;
            }
            if (found < 0) { stats[10]++; break; }
            int s = found, old = owner[s];
            if (gate && old >= 0 && !(skip && bgskip[old]) && freq_all[old] >= bf) { stats[8]++; break; }
            if (old >= 0) {
                int ol = old / E, oe = old % E;
                slotof_all[(long long) ol * E + oe] = -1;
                const int64_t* h = home_all + ((long long) ol * E + oe) * 3;
                ec_rows(tabs, ol, oe, h[0], h[1], h[2]);
                stats[4]++;
            }
            int g = ++bgctl[0];
            owner[s] = li * E + e; pin[s] = -2; refb[s] = 1; lgen[s] = g; pend[e] = s;
            int* r = ring + (head % R) * 4;
            r[0] = li * E + e; r[1] = s; r[2] = g; r[3] = 0;
            head++; done++; bgc[0]++; stats[5]++;
        }
        if (done) { __threadfence_system(); ((volatile long long*) hc)[0] = head; }
    }
    ctl[0] = hand; ctl[2] = nj;
}

__global__ void ec_land_k(int* landed, int s, int gen, unsigned long long* bgc)
{
    // [E3] stream-ordered after the slot's 3 copies on the background stream
    __threadfence();
    ((volatile int*) landed)[s] = gen;
    atomicAdd(&bgc[1], 1ull);
}

__global__ void ec_bg_commit_k(int L, int E, int* pend_all, const int* lgen, const int* landed, int* slotof_all,
    const int64_t* tabs, int* owner, int* pin, int* refb, const long long* __restrict__ cbase, int spc, long long rec,
    long long off_u, long long off_d, unsigned long long* stats)
{
    // [E3] quiesce (no copy outstanding): landed loads -> resident, anything else aborted (rows already home)
    long long g = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= (long long) L * E) return;
    int s = pend_all[g];
    if (s < 0) return;
    int l = g / E, e = g % E;
    if (((volatile const int*) landed)[s] == lgen[s]) {
        long long a = cbase[s / spc] + (long long) (s % spc) * rec;
        ec_rows(tabs, l, e, a, a + off_u, a + off_d);
        slotof_all[g] = s; pin[s] = 0; refb[s] = 1;
        atomicAdd(&stats[7], 1ull);
    } else {
        owner[s] = -1; pin[s] = 0; refb[s] = 0;
        atomicAdd(&stats[12], 1ull);
    }
    pend_all[g] = -1;
}

__global__ void ec_copy_k(const int* __restrict__ ctl, const int* __restrict__ jobs, int li, int E,
    const int64_t* __restrict__ home_all, const long long* __restrict__ cbase, int spc, long long rec, long long vg,
    long long vu, long long vd, long long off_u, long long off_d)
{
    long long nj = ctl[2];
    long long vpj = vg + vu + vd;
    long long total = nj * vpj;
    long long stride = (long long) gridDim.x * blockDim.x;
    for (long long base = (long long) blockIdx.x * blockDim.x + threadIdx.x; base < total; base += 4 * stride) {
        int4 v[4]; int4* dst[4]; int n = 0;
        #pragma unroll
        for (int u = 0; u < 4; u++) {
            long long idx = base + u * stride;
            if (idx >= total) break;
            long long j = idx / vpj, r = idx - j * vpj;
            int e = jobs[2 * j], s = jobs[2 * j + 1];
            const int64_t* h = home_all + ((long long) li * E + e) * 3;
            const int4* src; long long doff;
            if (r < vg) { src = (const int4*) h[0] + r; doff = r * 16; }
            else if (r < vg + vu) { src = (const int4*) h[1] + (r - vg); doff = off_u + (r - vg) * 16; }
            else { src = (const int4*) h[2] + (r - vg - vu); doff = off_d + (r - vg - vu) * 16; }
            v[u] = *src;
            dst[u] = (int4*) (cbase[s / spc] + (long long) (s % spc) * rec + doff);
            n++;
        }
        for (int u = 0; u < n; u++) *dst[u] = v[u];
    }
}

void ec_step(at::Tensor sel, int64_t first, int64_t li, int64_t E, int64_t S, int64_t admit, at::Tensor slotof, at::Tensor home,
             at::Tensor tabs, at::Tensor owner, at::Tensor refb, at::Tensor pin, at::Tensor ctl, at::Tensor jobs,
             at::Tensor stats, int64_t arena, int64_t spc, int64_t rec, int64_t off_u, int64_t off_d,
             at::Tensor osel, int64_t bg, at::Tensor pend, at::Tensor lgen, at::Tensor landed, at::Tensor freq,
             at::Tensor bgp, at::Tensor bgf, at::Tensor bgc, int64_t ring, int64_t hc, at::Tensor bgctl, at::Tensor bgskip)
{
    c10::cuda::CUDAGuard g(sel.device());
    auto st = at::cuda::getCurrentCUDAStream(sel.device().index());
    TORCH_CHECK(sel.dtype() == at::kLong && sel.is_contiguous(), "sel int64 contiguous");
    TORCH_CHECK(osel.dtype() == at::kLong && osel.is_contiguous() && osel.numel() == sel.numel(), "osel int64, same picks");
    TORCH_CHECK(E <= 1024, "E <= 1024");
    TORCH_CHECK(stats.numel() >= 16, "stats >= 16");
    ec_step_k<<<1, 512, 0, st>>>(sel.data_ptr<int64_t>(), (int) sel.numel(), (int) first, (int) li, (int) E, (int) S, (int) admit,
        slotof.data_ptr<int>(), home.data_ptr<int64_t>(), tabs.data_ptr<int64_t>(), owner.data_ptr<int>(),
        refb.data_ptr<int>(), pin.data_ptr<int>(), ctl.data_ptr<int>(), jobs.data_ptr<int>(), (int) (jobs.numel() / 2),
        (unsigned long long*) stats.data_ptr<int64_t>(), (const long long*) arena, (int) spc, rec, off_u, off_d,
        osel.data_ptr<int64_t>(), (int) bg, pend.data_ptr<int>(), lgen.data_ptr<int>(), landed.data_ptr<int>(),
        freq.data_ptr<float>(), bgp.data_ptr<int>(), bgf.data_ptr<float>(), (unsigned long long*) bgc.data_ptr<int64_t>(),
        (int*) ring, (long long*) hc, bgctl.data_ptr<int>(), bgskip.data_ptr<int>());
}

void ec_bg_commit(int64_t L, int64_t E, at::Tensor pend, at::Tensor lgen, at::Tensor landed, at::Tensor slotof, at::Tensor tabs,
                  at::Tensor owner, at::Tensor pin, at::Tensor refb, int64_t arena, int64_t spc, int64_t rec, int64_t off_u,
                  int64_t off_d, at::Tensor stats)
{
    c10::cuda::CUDAGuard g(pend.device());
    auto st = at::cuda::getCurrentCUDAStream(pend.device().index());
    long long n = L * E;
    ec_bg_commit_k<<<(int) ((n + 255) / 256), 256, 0, st>>>((int) L, (int) E, pend.data_ptr<int>(), lgen.data_ptr<int>(),
        landed.data_ptr<int>(), slotof.data_ptr<int>(), tabs.data_ptr<int64_t>(), owner.data_ptr<int>(), pin.data_ptr<int>(),
        refb.data_ptr<int>(), (const long long*) arena, (int) spc, rec, off_u, off_d, (unsigned long long*) stats.data_ptr<int64_t>());
}

// [E3] BgLoader: one host thread per pool. Polls the mapped ring (head written by ec_step), issues each logged expert's
// 3 copies home -> slot on its own non-blocking lowest-priority stream, then ec_land_k. Only cudaMemcpyAsync + kernel
// launches: no allocation, no sync, so the GPU's ft_combine spin-wait can never deadlock against it. If it stops,
// LOADING keys simply never commit: qmax fills and background admission stops (never a wrong row).
struct BgLoader {
    int dev = 0, R = 0, E = 0, spc = 0, cpu = -1;
    long long rec = 0, off_u = 0, off_d = 0, sg = 0, su = 0, sd = 0;
    volatile int* ring = nullptr; volatile long long* hc = nullptr;
    const int64_t* home = nullptr; at::Tensor home_keep;
    std::vector<long long> cbase;
    int* landed = nullptr; unsigned long long* bgc = nullptr;
    cudaStream_t st = nullptr;
    std::atomic<long long> issued{0}, bytes{0}, wakes{0};
    std::atomic<int> err{0};
    std::thread th;

    void run() {
        cudaSetDevice(dev);
        if (cpu >= 0) { cpu_set_t cs; CPU_ZERO(&cs); CPU_SET(cpu, &cs); pthread_setaffinity_np(pthread_self(), sizeof(cs), &cs); }
        long long tail = hc[1];
        for (;;) {
            long long head = hc[0];
            if (head == tail || err.load()) { std::this_thread::sleep_for(std::chrono::microseconds(50)); continue; }
            std::atomic_thread_fence(std::memory_order_acquire);
            wakes++;
            for (; tail < head; tail++) {
                volatile int* r = ring + (tail % R) * 4;
                int key = r[0], s = r[1], gen = r[2];
                const int64_t* h = home + (long long) key * 3;
                char* d = (char*) (cbase[s / spc] + (long long) (s % spc) * rec);
                cudaMemcpyAsync(d, (const void*) h[0], (size_t) sg, cudaMemcpyDefault, st);
                cudaMemcpyAsync(d + off_u, (const void*) h[1], (size_t) su, cudaMemcpyDefault, st);
                cudaMemcpyAsync(d + off_d, (const void*) h[2], (size_t) sd, cudaMemcpyDefault, st);
                ec_land_k<<<1, 1, 0, st>>>(landed, s, gen, bgc);
                cudaError_t e = cudaGetLastError();
                if (e != cudaSuccess) { err = (int) e; fprintf(stderr, " !! expert_cache bg: %s (background admission stops)\n", cudaGetErrorString(e)); break; }
                issued++; bytes += sg + su + sd;
            }
            std::atomic_thread_fence(std::memory_order_release);
            hc[1] = tail;
        }
    }
};
static std::vector<BgLoader*> BG;   // never freed: the detached threads outlive static destruction at exit

int64_t bg_start(int64_t ring, int64_t hc, int64_t R, at::Tensor home_cpu, std::vector<int64_t> cbase, int64_t spc, int64_t rec,
                 int64_t off_u, int64_t off_d, int64_t sg, int64_t su, int64_t sd, int64_t E, at::Tensor landed, at::Tensor bgc,
                 int64_t device, int64_t cpu)
{
    TORCH_CHECK(home_cpu.device().is_cpu() && home_cpu.dtype() == at::kLong && home_cpu.is_contiguous(), "home_cpu int64 CPU");
    auto* b = new BgLoader();
    b->dev = (int) device; b->R = (int) R; b->E = (int) E; b->spc = (int) spc; b->cpu = (int) cpu;
    b->rec = rec; b->off_u = off_u; b->off_d = off_d; b->sg = sg; b->su = su; b->sd = sd;
    b->ring = (volatile int*) ring; b->hc = (volatile long long*) hc;
    b->home_keep = home_cpu; b->home = home_cpu.data_ptr<int64_t>();
    b->cbase.assign(cbase.begin(), cbase.end());
    b->landed = landed.data_ptr<int>(); b->bgc = (unsigned long long*) bgc.data_ptr<int64_t>();
    c10::cuda::CUDAGuard g((c10::DeviceIndex) device);
    int lo = 0, hi = 0;
    cudaDeviceGetStreamPriorityRange(&lo, &hi);
    TORCH_CHECK(cudaStreamCreateWithPriority(&b->st, cudaStreamNonBlocking, lo) == cudaSuccess, "bg stream");
    b->th = std::thread([b] { b->run(); });
    b->th.detach();
    BG.push_back(b);
    return (int64_t) BG.size() - 1;
}

bool bg_drain(int64_t h, int64_t timeout_ms)
{
    // caller has synchronized the compute stream (every ec_step that could log has run); wait until the thread has
    // issued everything logged, then for the background stream
    auto* b = BG.at(h);
    auto t0 = std::chrono::steady_clock::now();
    while (b->hc[1] != b->hc[0]) {
        if (b->err.load() || std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(timeout_ms)) return false;
        std::this_thread::sleep_for(std::chrono::microseconds(50));
    }
    return cudaStreamSynchronize(b->st) == cudaSuccess;
}

void bg_set_cbase(int64_t h, std::vector<int64_t> cbase)
{
    // only while drained and paused (elastic exit re-points flex chunks)
    auto* b = BG.at(h);
    b->cbase.assign(cbase.begin(), cbase.end());
}

std::vector<int64_t> bg_stats(int64_t h)
{
    auto* b = BG.at(h);
    return {b->issued.load(), b->bytes.load(), b->wakes.load(), (int64_t) b->err.load(), b->hc[0], b->hc[1]};
}

void ec_copy(at::Tensor ctl, at::Tensor jobs, int64_t li, int64_t E, at::Tensor home, int64_t arena, int64_t spc, int64_t rec,
             int64_t sg, int64_t su, int64_t sd, int64_t off_u, int64_t off_d, int64_t grid)
{
    c10::cuda::CUDAGuard g(ctl.device());
    auto st = at::cuda::getCurrentCUDAStream(ctl.device().index());
    ec_copy_k<<<(int) grid, 256, 0, st>>>(ctl.data_ptr<int>(), jobs.data_ptr<int>(), (int) li, (int) E,
        home.data_ptr<int64_t>(), (const long long*) arena, (int) spc, rec, sg / 16, su / 16, sd / 16, off_u, off_d);
}


__global__ void ec_verify_k(const int* __restrict__ owner, const int* __restrict__ slotof_all, const int64_t* __restrict__ home_all,
    const int64_t* __restrict__ tabs, const long long* __restrict__ cbase, int spc, long long rec, long long sg, long long su, long long sd,
    long long off_u, long long off_d, int S, int E, int L, unsigned long long* out)
{
    // out[0] = slots whose bytes differ from home, out[1] = table entries inconsistent with slotof, out[2] = owned slots
    int s = blockIdx.x;
    if (s < S) {
        int o = owner[s];
        if (o >= 0) {
            int l = o / E, e = o % E;
            if (threadIdx.x == 0) atomicAdd(&out[2], 1ull);
            const int64_t* h = home_all + ((long long) l * E + e) * 3;
            const int4* src[3] = {(const int4*) h[0], (const int4*) h[1], (const int4*) h[2]};
            long long off[3] = {0, off_u, off_d}; long long n[3] = {sg / 16, su / 16, sd / 16};
            __shared__ int bad;
            if (threadIdx.x == 0) bad = 0;
            __syncthreads();
            for (int f = 0; f < 3; f++) {
                const int4* d = (const int4*) (cbase[s / spc] + (long long) (s % spc) * rec + off[f]);
                for (long long i = threadIdx.x; i < n[f]; i += blockDim.x) {
                    int4 a = src[f][i], b = d[i];
                    if (a.x != b.x || a.y != b.y || a.z != b.z || a.w != b.w) bad = 1;
                }
            }
            __syncthreads();
            if (threadIdx.x == 0 && bad) atomicAdd(&out[0], 1ull);
            if (threadIdx.x == 0 && slotof_all[(long long) l * E + e] != s) atomicAdd(&out[1], 1ull);
        }
    }
    // table check: one thread per (layer, expert)
    long long gid = (long long) blockIdx.x * blockDim.x + threadIdx.x;
    if (gid < (long long) L * E) {
        int l = gid / E, e = gid % E;
        int sl = slotof_all[gid];
        const int64_t* pg = (const int64_t*) tabs[l * 4 + 0]; const int64_t* pu = (const int64_t*) tabs[l * 4 + 1];
        const int64_t* pd = (const int64_t*) tabs[l * 4 + 2]; const int64_t* gu = (const int64_t*) tabs[l * 4 + 3];
        const int64_t* h = home_all + gid * 3;
        int64_t eg, eu, ed;
        if (sl >= 0) { long long a = cbase[sl / spc] + (long long) (sl % spc) * rec; eg = a; eu = a + off_u; ed = a + off_d; }
        else { eg = h[0]; eu = h[1]; ed = h[2]; }
        bool ok = pg[e] == eg && pu[e] == eu && pd[e] == ed;
        if (gu) ok = ok && gu[2 * e] == eg && gu[2 * e + 1] == eu;
        if (!ok) atomicAdd(&out[1], 1ull);
    }
}

at::Tensor ec_verify(at::Tensor owner, at::Tensor slotof, at::Tensor home, at::Tensor tabs, int64_t arena, int64_t spc, int64_t rec,
                     int64_t sg, int64_t su, int64_t sd, int64_t off_u, int64_t off_d, int64_t S, int64_t E, int64_t L)
{
    c10::cuda::CUDAGuard g(owner.device());
    auto st = at::cuda::getCurrentCUDAStream(owner.device().index());
    auto out = torch::zeros({4}, owner.options().dtype(at::kLong));
    long long nb = std::max((long long) S, ((long long) L * E + 255) / 256);
    ec_verify_k<<<(int) nb, 256, 0, st>>>(owner.data_ptr<int>(), slotof.data_ptr<int>(), home.data_ptr<int64_t>(),
        tabs.data_ptr<int64_t>(), (const long long*) arena, (int) spc, rec, sg, su, sd, off_u, off_d, (int) S, (int) E, (int) L,
        (unsigned long long*) out.data_ptr<int64_t>());
    return out;
}

__global__ void ec_evict_k(int s0, int s1, int disable, int E, int* owner, int* slotof_all, const int64_t* home_all,
    const int64_t* tabs, int* refb, int* pin)
{
    int s = s0 + blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= s1) return;
    int o = owner[s];
    if (o >= 0) {
        int l = o / E, e = o % E;
        slotof_all[(long long) l * E + e] = -1;
        const int64_t* h = home_all + ((long long) l * E + e) * 3;
        int64_t* pg = (int64_t*) tabs[l * 4 + 0]; int64_t* pu = (int64_t*) tabs[l * 4 + 1];
        int64_t* pd = (int64_t*) tabs[l * 4 + 2]; int64_t* gu = (int64_t*) tabs[l * 4 + 3];
        pg[e] = h[0]; pu[e] = h[1]; pd[e] = h[2];
        if (gu) { gu[2 * e] = h[0]; gu[2 * e + 1] = h[1]; }
        owner[s] = -1;
    }
    refb[s] = 0;
    pin[s] = disable ? -1 : 0;
}

void ec_evict(int64_t s0, int64_t s1, int64_t disable, int64_t E, at::Tensor owner, at::Tensor slotof, at::Tensor home,
              at::Tensor tabs, at::Tensor refb, at::Tensor pin)
{
    c10::cuda::CUDAGuard g(owner.device());
    auto st = at::cuda::getCurrentCUDAStream(owner.device().index());
    int n = (int) (s1 - s0);
    if (n <= 0) return;
    ec_evict_k<<<(n + 255) / 256, 256, 0, st>>>((int) s0, (int) s1, (int) disable, (int) E, owner.data_ptr<int>(),
        slotof.data_ptr<int>(), home.data_ptr<int64_t>(), tabs.data_ptr<int64_t>(), refb.data_ptr<int>(), pin.data_ptr<int>());
}

void memcpy_runs(at::Tensor runs, int64_t stream, int64_t device)
{
    // runs: CPU int64 [n, 3] = (dst, src, bytes); cudaMemcpyDefault (src may be a UVA host alias)
    c10::cuda::CUDAGuard g((c10::DeviceIndex) device);
    auto r = runs.accessor<int64_t, 2>();
    for (int64_t i = 0; i < runs.size(0); i++)
        cudaMemcpyAsync((void*) r[i][0], (const void*) r[i][1], (size_t) r[i][2], cudaMemcpyDefault, (cudaStream_t) stream);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("ec_bg_commit", &ec_bg_commit);
    m.def("bg_start", &bg_start);
    m.def("bg_drain", &bg_drain);
    m.def("bg_set_cbase", &bg_set_cbase);
    m.def("bg_stats", &bg_stats);
    m.def("ec_step", &ec_step);
    m.def("ec_copy", &ec_copy);
    m.def("memcpy_runs", &memcpy_runs);
    m.def("ec_verify", &ec_verify);
    m.def("ec_evict", &ec_evict);
}
"""

_EXT = None
_BC_ARGS = {}      # id(multi_gate.ptrs_trellis) -> gu_trellis_ptr tensor captured from BC_BlockSparseMLP(...)
POOLS = []


def _ext():
    global _EXT
    if _EXT is None:
        from torch.utils.cpp_extension import load
        d = os.environ.get("GLM53_EC_BUILD", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "ec"))
        os.makedirs(d, exist_ok=True)
        src = os.path.join(d, "ec.cu")
        if not os.path.exists(src) or open(src).read() != _SRC:
            open(src, "w").write(_SRC)
        _EXT = load(name="glm53_ec", sources=[src], build_directory=d, extra_cuda_cflags=["-O3"], verbose=False)
    return _EXT


def install():
    """Before model load: capture gu_trellis_ptr (a local of BlockSparseMLP.load_local) from the BC constructor."""
    from exllamav3.ext import exllamav3_ext as ext
    import exllamav3.modules.block_sparse_mlp as bsm
    real = ext.BC_BlockSparseMLP

    def wrapped(*a, **k):
        obj = real(*a, **k)
        _BC_ARGS[id(a[16])] = (a[16], a[22], a[43])   # gate ptrs (identity key), up ptrs, gu_trellis_ptr
        return obj
    ext.BC_BlockSparseMLP = wrapped
    bsm.ext.BC_BlockSparseMLP = wrapped
    _ext()   # build now (fail fast, before the long load)
    print(" -- expert_cache installed", flush=True)


class Pool:
    def __init__(self, device, mods, reserve, max_slots, admit_max, stage_bytes=0, stage_min=8192):
        self.device = device
        self.stage_min = stage_min
        self.mods = mods
        self.admit_max = admit_max
        m0 = mods[0]
        self.E = m0.num_local_experts or m0.num_experts
        self.first = int(m0.routing_first or 0)
        assert all((m.num_local_experts or m.num_experts) == self.E and int(m.routing_first or 0) == self.first for m in mods)
        sizes = [m0.multi_gate.linears[0].inner.trellis.numel() * m0.multi_gate.linears[0].inner.trellis.element_size(),
                 m0.multi_up.linears[0].inner.trellis.numel() * m0.multi_up.linears[0].inner.trellis.element_size(),
                 m0.multi_down.linears[0].inner.trellis.numel() * m0.multi_down.linears[0].inner.trellis.element_size()]
        al = lambda x: (x + 255) // 256 * 256
        self.sg, self.su, self.sd = sizes
        self.off_u = al(self.sg); self.off_d = self.off_u + al(self.su); self.rec = self.off_d + al(self.sd)
        for m in mods:
            for mm, s in ((m.multi_gate, self.sg), (m.multi_up, self.su), (m.multi_down, self.sd)):
                for l in mm.linears:
                    assert l.inner.trellis.numel() * l.inner.trellis.element_size() == s and s % 16 == 0
        torch.cuda.synchronize(device); torch.cuda.empty_cache()
        free, _ = torch.cuda.mem_get_info(device)
        nst = int(stage_bytes // self.rec) if stage_bytes > 0 else 0
        self.stage_n = nst
        self.elastic = float(os.environ.get("GLM53_EC_ELASTIC_GB", "0")) * 1024 ** 3
        self.stage = None
        if nst and not self.elastic:
            self.stage = [torch.empty((nst, self.rec), dtype=torch.uint8, device=device) for _ in range(2)]
            free -= 2 * nst * self.rec
        S = max(0, int((free - reserve) // self.rec))
        if max_slots is not None:
            S = min(S, max_slots)
        # arena in chunks of spc slots: flex chunks (the last ones) are handed back to torch during prefill (elastic)
        self.spc = int(os.environ.get("GLM53_EC_CHUNK_SLOTS", "32"))
        nch = (S + self.spc - 1) // self.spc
        S = nch * self.spc if S else 0
        self.chunks = []
        for c in range(nch):
            try:
                self.chunks.append(torch.empty((self.spc, self.rec), dtype=torch.uint8, device=device))
            except torch.OutOfMemoryError:
                break
        S = len(self.chunks) * self.spc
        self.S = S
        self.cbase = torch.tensor([t.data_ptr() for t in self.chunks] or [0], dtype=torch.int64, device=device)
        self.flex = []
        if self.elastic and self.chunks:
            nflex = min(len(self.chunks), int(-(-self.elastic // (self.spc * self.rec))))
            self.flex = list(range(len(self.chunks) - nflex, len(self.chunks)))
        self.in_prefill = False
        self.elastic_stats = {"enter": 0, "exit": 0, "realloc_fail": 0}
        L = len(mods)
        i32 = dict(dtype=torch.int32, device=device)
        self.slotof = torch.full((L, self.E), -1, **i32)
        self.owner = torch.full((max(S, 1),), -1, **i32)
        self.refb = torch.zeros((max(S, 1),), **i32)
        self.pin = torch.zeros((max(S, 1),), **i32)
        self.ctl = torch.zeros((4,), **i32)
        self.jobs = torch.zeros((2 * 1024,), **i32)
        self.stats = torch.zeros((16,), dtype=torch.int64, device=device)
        home, tabs = [], []
        self.keep = []
        for m in mods:
            pg, pu, pd = m.multi_gate.ptrs_trellis, m.multi_up.ptrs_trellis, m.multi_down.ptrs_trellis
            cap = _BC_ARGS.get(id(pg))
            gu = None
            if cap is not None:
                assert cap[0] is pg and cap[1] is pu
                gu = cap[2]
                assert gu.shape == (self.E, 2) and gu.dtype == torch.long and gu.device == pg.device
                assert torch.equal(gu[:, 0], pg) and torch.equal(gu[:, 1], pu), m.key
            else:
                print(f" !! expert_cache: no gu table for {m.key} (fused bszN path would read home)", flush=True)
            for t in (pg, pu, pd):
                assert t.dtype == torch.long and t.is_contiguous() and t.device == torch.device(device)
            home.append(torch.stack([pg, pu, pd], dim=1))
            tabs.append([pg.data_ptr(), pu.data_ptr(), pd.data_ptr(), gu.data_ptr() if gu is not None else 0])
            self.keep.append((pg, pu, pd, gu))
        self.home = torch.stack(home).contiguous()                    # [L, E, 3] host (UVA) addresses
        self.tabs = torch.tensor(tabs, dtype=torch.int64, device=device)
        # GLM53_EC_COPY_GRID blocks for ec_copy (16 saturates the link and leaves the SMs to concurrent work)
        self.grid = int(os.environ.get("GLM53_EC_COPY_GRID", "0")) or torch.cuda.get_device_properties(device).multi_processor_count * 8
        self.home_cpu = self.home.cpu()
        self.size_hist = {}
        if nst:
            self.cstream = torch.cuda.Stream(device)
            self.timing = bool(int(os.environ.get("GLM53_EC_TIMING", "0")))
            self.ev_ready = [torch.cuda.Event(enable_timing=self.timing) for _ in range(2)]
            self.tev = {}   # li -> [copy_start, copy_end, hook(compute stream)] events of the current forward
            self.staged = {}          # li -> (buffer idx, expert idx tensor on device)
            self.last_li = None
            self.slot_cpu = None
            self.stage_stats = {"forwards": 0, "layers": 0, "experts": 0, "overflow": 0}
        for li, m in enumerate(mods):
            m._ec = (self, li)
        self._bg_init()

    # ---- [E3] background admission (T-19) --------------------------------------------------------------------
    def _bg_init(self):
        """Device state is always allocated (small) so ec_step has one signature; the loader thread only with GLM53_EC_BG=1."""
        L, E, dev = len(self.mods), self.E, self.device
        i32 = dict(dtype=torch.int32, device=dev)
        self.pend = torch.full((L, E), -1, **i32)
        self.lgen = torch.zeros((max(self.S, 1),), **i32)
        self.landed = torch.zeros((max(self.S, 1),), **i32)
        self.freq = torch.zeros((L, E), dtype=torch.float32, device=dev)
        self.bgctl = torch.zeros((4,), **i32)
        self.bgc = torch.zeros((4,), dtype=torch.int64, device=dev)
        self.bgp = torch.zeros((8,), **i32)
        self.bgf = torch.ones((2,), dtype=torch.float32, device=dev)
        self.bgskip = torch.zeros((L, E), **i32)        # E3b: 1 = held by the remote 3090 tier (cpu_tier sets m._rt_res)
        self.rt_attached = False; self.bg_dirty = False
        self.bg = None
        self.bg_paused = False
        self.bg_state = {"pauses": 0, "drain_fail": 0, "broken": False}
        self.ring_ptr = self.hc_ptr = 0
        if os.environ.get("GLM53_EC_BG") != "1" or self.S == 0:
            return
        self.bg_R = int(os.environ.get("GLM53_EC_BG_RING", "1024"))
        self.ring = torch.zeros((self.bg_R * 4,), dtype=torch.int32).pin_memory()
        self.hc = torch.zeros((4,), dtype=torch.int64).pin_memory()
        self.ring_ptr, self.hc_ptr = self.ring.data_ptr(), self.hc.data_ptr()
        self.bg_cfg = {"on": int(os.environ.get("GLM53_EC_BG_ON", "1")), "qmax": int(os.environ.get("GLM53_EC_BG_QMAX", "2")),
                       "per_call": int(os.environ.get("GLM53_EC_BG_PER_CALL", "1")), "gate": int(os.environ.get("GLM53_EC_BG_GATE", "1")),
                       "decay": float(os.environ.get("GLM53_EC_BG_DECAY", "0.99")),
                       "skip_remote": int(os.environ.get("GLM53_EC_BG_SKIP_REMOTE", "1"))}
        self._bg_write(0)                                   # off until attach_modules has warmed the cache
        self.bg = _ext().bg_start(self.ring_ptr, self.hc_ptr, self.bg_R, self.home_cpu.contiguous(), self._cbase_list(),
                                  self.spc, self.rec, self.off_u, self.off_d, self.sg, self.su, self.sd, self.E, self.landed,
                                  self.bgc, self.device.index, int(os.environ.get("GLM53_EC_BG_CPU", "-1")))

    def _cbase_list(self):
        return [t.data_ptr() if t is not None else 0 for t in self.chunks] or [0]

    def _bg_write(self, on):
        c = self.bg_cfg
        skip = int(bool(c["skip_remote"]) and self.rt_attached)
        self.bgp.copy_(torch.tensor([on, c["qmax"], c["per_call"], c["gate"], self.bg_R, skip, 0, 0], dtype=torch.int32))
        self.bgf.copy_(torch.tensor([c["decay"], 0.0], dtype=torch.float32))

    def bg_set(self, **kw):
        if self.bg is None:
            return None
        for k, v in kw.items():
            if v is not None:
                self.bg_cfg[k] = type(self.bg_cfg[k])(v)
        if not self.bg_paused:
            self._bg_write(self.bg_cfg["on"])
        return dict(self.bg_cfg)

    def set_rt(self, attached=None):
        """E3b: rebuild the remote-residency mask from the modules (main thread) and/or note the attach state (any
        thread: the device write happens in the next step() on the main thread)."""
        if attached is None:
            for li, m in enumerate(self.mods):
                r = getattr(m, "_rt_res", None)
                if r is not None:
                    self.bgskip[li].copy_(r.to(torch.int32))
        elif bool(attached) != self.rt_attached:
            self.rt_attached = bool(attached); self.bg_dirty = True

    def _bg_pause(self):
        """Quiesce: no new admissions, every logged copy issued and finished, landed -> resident, rest aborted. After it
        no slot is LOADING, so elastic eviction, staging snapshots and ec_verify see a plain cache."""
        if self.bg is None or self.bg_paused:
            return not self.bg_state["broken"]
        self._bg_write(0)                                   # stream-ordered: later ec_steps admit nothing
        torch.cuda.synchronize(self.device)                 # every ec_step that could have logged has run
        self.bg_paused = True
        if not _ext().bg_drain(self.bg, 5000):
            self.bg_state["drain_fail"] += 1; self.bg_state["broken"] = True
            print(" !! expert_cache bg: drain failed -- background admission stays off (LOADING slots stay pinned)", flush=True)
            return False
        _ext().ec_bg_commit(len(self.mods), self.E, self.pend, self.lgen, self.landed, self.slotof, self.tabs, self.owner,
                            self.pin, self.refb, self.cbase.data_ptr(), self.spc, self.rec, self.off_u, self.off_d, self.stats)
        torch.cuda.synchronize(self.device)
        self.bg_state["pauses"] += 1
        return True

    def _bg_resume(self):
        if self.bg is None or not self.bg_paused or self.bg_state["broken"]:
            return
        _ext().bg_set_cbase(self.bg, self._cbase_list())
        self._bg_write(self.bg_cfg["on"])
        self.bg_paused = False

    # ---- prefill staging -------------------------------------------------------------------------------------
    def _restore(self, li):
        b, idx = self.staged.pop(li)
        pg, pu, pd, gu = self.keep[li]
        h = self.home[li].index_select(0, idx)
        pg.index_copy_(0, idx, h[:, 0].contiguous()); pu.index_copy_(0, idx, h[:, 1].contiguous())
        pd.index_copy_(0, idx, h[:, 2].contiguous())
        if gu is not None:
            gu.index_copy_(0, idx, h[:, :2].contiguous())

    def _issue(self, li, b):
        """Enqueue on the copy stream: layer li's non-cached experts -> staging buffer b (after its previous user)."""
        if li >= len(self.mods):
            return
        miss = (self.slot_cpu[li] < 0).nonzero().flatten()
        n = int(miss.numel())
        take = miss[: self.stage_n]
        self.stage_stats["overflow"] += n - int(take.numel())
        runs = []
        base = self.stage[b].data_ptr()
        hc = self.home_cpu[li]
        for j, e in enumerate(take.tolist()):
            d = base + j * self.rec
            runs.append((d, int(hc[e, 0]), self.sg)); runs.append((d + self.off_u, int(hc[e, 1]), self.su))
            runs.append((d + self.off_d, int(hc[e, 2]), self.sd))
        with torch.cuda.stream(self.cstream):
            self.cstream.wait_stream(torch.cuda.current_stream(self.device))   # previous user of buffer b is done
            if self.timing:
                e0 = torch.cuda.Event(enable_timing=True); e0.record(self.cstream)
                self.tev.setdefault(li, [None, None, None])[0] = e0
            if runs:
                _ext().memcpy_runs(torch.tensor(runs, dtype=torch.int64), self.cstream.cuda_stream, self.device.index)
            self.ev_ready[b].record(self.cstream)
            if self.timing:
                e1 = torch.cuda.Event(enable_timing=True); e1.record(self.cstream)
                self.tev.setdefault(li, [None, None, None])[1] = e1
        self.pending = (li, b, take)
        self.stage_stats["layers"] += 1; self.stage_stats["experts"] += int(take.numel())

    def _stage_hook(self, li, n_picks):
        prefill = n_picks >= self.stage_min
        for l in [k for k in self.staged if k != li]:
            self._restore(l)                          # stream order: after that layer's MoE
        if not prefill:
            self.last_li = None
            if self.in_prefill:
                self._elastic_exit()
            return
        if self.elastic and self.bg_state["broken"]:
            return                                      # [bdw-opt] reclaim was skipped: no staging buffers, prefill reads zero-copy
        cur = torch.cuda.current_stream(self.device)
        if self.last_li is None or li <= self.last_li:  # new forward pass on this device
            self._bg_pause()                            # [E3] before the slotof snapshot / elastic eviction
            if self.timing:
                self._timing_collect()
            if self.elastic and not self.in_prefill:
                self._elastic_enter()
            self.slot_cpu = self.slotof.cpu()           # stable during prefill (no admission)
            self.stage_stats["forwards"] += 1
            self.pending = None
            self._issue(li, li % 2)
        self.last_li = li
        if self.pending is None or self.pending[0] != li:
            self._issue(li, li % 2)
        _, b, take = self.pending
        if self.timing:
            eh = torch.cuda.Event(enable_timing=True); eh.record(cur)
            self.tev.setdefault(li, [None, None, None])[2] = eh
        cur.wait_event(self.ev_ready[b])
        if take.numel():
            idx = take.to(self.device, non_blocking=True)
            base = self.stage[b].data_ptr()
            a = base + torch.arange(take.numel(), dtype=torch.int64, device=self.device) * self.rec
            pg, pu, pd, gu = self.keep[li]
            pg.index_copy_(0, idx, a); pu.index_copy_(0, idx, a + self.off_u); pd.index_copy_(0, idx, a + self.off_d)
            if gu is not None:
                gu.index_copy_(0, idx, torch.stack([a, a + self.off_u], 1))
            self.staged[li] = (b, idx)
        self._issue(li + 1, (li + 1) % 2)              # next layer, other buffer, overlaps this layer's compute

    def _elastic_enter(self):
        """Prefill: evict + disable the flex chunks' slots, hand their memory back to torch (activations of big
        chunks), allocate the staging buffers from it."""
        if not self._bg_pause():                        # [E3] no copy may land in a chunk handed back to torch
            # [bdw-opt] a copy may still be in flight into a flex chunk: never hand it back (prefill runs without it)
            print(" !! expert_cache: background copies not drained -- elastic reclaim skipped", flush=True)
            return
        e = _ext()
        for c in self.flex:
            e.ec_evict(c * self.spc, (c + 1) * self.spc, 1, self.E, self.owner, self.slotof, self.home, self.tabs,
                       self.refb, self.pin)
        torch.cuda.synchronize(self.device)
        for c in self.flex:
            self.chunks[c] = None
        self.stage = [torch.empty((self.stage_n, self.rec), dtype=torch.uint8, device=self.device) for _ in range(2)]
        self.in_prefill = True
        self.elastic_stats["enter"] += 1

    def _elastic_exit(self):
        """Back to decode: free staging, reallocate the flex chunks, re-enable their slots (empty)."""
        torch.cuda.synchronize(self.device)
        self.stage = None
        torch.cuda.empty_cache()
        e = _ext()
        for c in self.flex:
            try:
                t = torch.empty((self.spc, self.rec), dtype=torch.uint8, device=self.device)
            except torch.OutOfMemoryError:
                self.elastic_stats["realloc_fail"] += 1
                continue
            self.chunks[c] = t
            self.cbase[c] = t.data_ptr()
            e.ec_evict(c * self.spc, (c + 1) * self.spc, 0, self.E, self.owner, self.slotof, self.home, self.tabs,
                       self.refb, self.pin)
        self.in_prefill = False
        self.elastic_stats["exit"] += 1

    def _timing_collect(self):
        """Per-layer copy (staging) vs compute ms of the last prefill forward: copy = copy-stream span of the layer's
        staging; compute = gap between consecutive layer hooks on the compute stream (MoE + attention of a layer)."""
        if not self.tev:
            return
        torch.cuda.synchronize(self.device)
        ls = sorted(self.tev)
        copy = [self.tev[l][0].elapsed_time(self.tev[l][1]) for l in ls if self.tev[l][0] and self.tev[l][1]]
        hooks = [self.tev[l][2] for l in ls if self.tev[l][2] is not None]
        comp = [hooks[i].elapsed_time(hooks[i + 1]) for i in range(len(hooks) - 1)]
        self.last_timing = {"layers": len(ls), "copy_ms_mean": round(sum(copy) / max(1, len(copy)), 2),
                            "compute_ms_mean": round(sum(comp) / max(1, len(comp)), 2),
                            "copy_ms_total": round(sum(copy), 1), "compute_ms_total": round(sum(comp), 1)}
        self.tev = {}

    def step(self, li, sel, admit):
        n = sel.numel()
        b = 1 << max(0, (n - 1).bit_length())
        self.size_hist[b] = self.size_hist.get(b, 0) + 1
        if self.stage_n:
            self._stage_hook(li, sel.numel())
        if admit and self.bg_paused:
            self._bg_resume()
        elif self.bg_dirty and not self.bg_paused and self.bg is not None:
            self.bg_dirty = False
            self._bg_write(self.bg_cfg["on"])
        e = _ext()
        sel = sel.reshape(-1)
        if sel.dtype != torch.long or not sel.is_contiguous():
            sel = sel.long().contiguous()
        osel = sel
        if self.bg is not None:                         # [E3] the router's picks before ft_split masked the CPU lane
            m = self.mods[li]
            o = getattr(m, "_ct_orig", None)
            pending = getattr(m, "_ct_pending", False) or getattr(m, "_pf", None) is not None   # [bdw-prefill] grouped prefill too
            if o is not None and pending and o.numel() == sel.numel():
                osel = o.reshape(-1)
                if osel.dtype != torch.long or not osel.is_contiguous():
                    osel = osel.long().contiguous()
        e.ec_step(sel, self.first, li, self.E, self.S, int(admit), self.slotof, self.home, self.tabs, self.owner, self.refb, self.pin,
                  self.ctl, self.jobs, self.stats, self.cbase.data_ptr(), self.spc, self.rec, self.off_u, self.off_d,
                  osel, int(self.bg is not None), self.pend, self.lgen, self.landed, self.freq, self.bgp, self.bgf, self.bgc,
                  self.ring_ptr, self.hc_ptr, self.bgctl, self.bgskip)
        if admit:
            e.ec_copy(self.ctl, self.jobs, li, self.E, self.home, self.cbase.data_ptr(), self.spc, self.rec, self.sg, self.su,
                      self.sd, self.off_u, self.off_d, self.grid)

    def warm(self, scores):
        """Initial fill: top S/L experts per layer by score (admission path, so tables stay consistent)."""
        k = self.S // max(1, len(self.mods))
        if k == 0:
            return
        with torch.cuda.device(self.device):
            for li, m in enumerate(self.mods):
                sc = scores.get(m.key)
                if sc is None:
                    continue
                if getattr(m, "cpu_split_first", None) is not None:
                    top = list(range(min(k, self.E)))       # static CPU split: local ids are already hot -> cold
                else:
                    loc = range(self.first, self.first + self.E)
                    top = sorted(loc, key=lambda x: -sc[x])[:k]
                for c in range(0, len(top), 256):
                    sel = torch.tensor(top[c:c + 256], dtype=torch.long, device=self.device)
                    self.step(li, sel, True)
            torch.cuda.synchronize(self.device)
        self.stats.zero_()

    def verify(self):
        """Full consistency check (reads every owned slot and its home copy over PCIe): bytes and pointer tables."""
        was = self.bg is not None and not self.bg_paused
        self._bg_pause()                                # [E3] commit/abort every background load: the check is exact
        torch.cuda.synchronize(self.device)
        o = _ext().ec_verify(self.owner, self.slotof, self.home, self.tabs, self.cbase.data_ptr(), self.spc, self.rec, self.sg,
                             self.su, self.sd, self.off_u, self.off_d, self.S, self.E, len(self.mods)).tolist()
        r = {"device": str(self.device), "bad_slots": o[0], "bad_tables": o[1], "owned": o[2]}
        if self.bg is not None:
            own = self.owner[self.owner >= 0]
            r.update(loading=int((self.pin == -2).sum()), pending=int((self.pend >= 0).sum()),
                     dup_owner=int(own.numel() - torch.unique(own).numel()))
            if was:
                self._bg_resume()
        return r

    def summary(self):
        s = self.stats.tolist()
        return {"device": str(self.device), "layers": len(self.mods), "slots": self.S, "slot_bytes": self.rec,
                "arena_gb": round(self.S * self.rec / 1e9, 2),
                "elastic": dict(self.elastic_stats, flex_chunks=len(self.flex), flex_gb=round(len(self.flex) * self.spc * self.rec / 1e9, 2), in_prefill=self.in_prefill) if self.elastic else None,
                "mem_alloc_gb": round(torch.cuda.memory_allocated(self.device) / 1e9, 2),
                "mem_peak_gb": round(torch.cuda.max_memory_allocated(self.device) / 1e9, 2), "hits": s[0], "misses": s[1], "admitted": s[2],
                "no_victim": s[3], "evictions": s[4],
                "hit_rate": round(s[0] / max(1, s[0] + s[1]), 4),
                "stage": dict(self.stage_stats, per_buffer=self.stage_n) if self.stage_n else None,
                "bg": self._bg_summary(s),
                "picks_hist": dict(sorted(self.size_hist.items())),
                "stage_timing_last_forward": getattr(self, "last_timing", None)}


    def _bg_summary(self, s):
        if self.bg is None:
            return None
        c = self.bgc.tolist(); h = _ext().bg_stats(self.bg)
        return dict(self.bg_cfg, paused=self.bg_paused, rt_attached=self.rt_attached, skip_keys=int(self.bgskip.sum()), **self.bg_state, admitted=s[5], committed=s[7], gate_reject=s[8],
                    ring_full=s[9], no_victim=s[10], qmax_block=s[11], aborted=s[12], candidates=s[13], loading_picks=s[14],
                    in_flight=c[0] - c[1], landed=c[1], issued=h[0], gb=round(h[1] / 1e9, 2), wakes=h[2], err=h[3])


def find_moe(modules):
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    seen = []

    def walk(mm):
        if isinstance(mm, BlockSparseMLP):
            seen.append(mm)
        for c in getattr(mm, "modules", []) or []:
            walk(c)
    for top in modules:
        walk(top)
    return [m for m in seen if getattr(m, "multi_up", None) is not None and m.device is not None
            and (m.num_local_experts is None or m.num_local_experts > 0)]


def attach(model):
    pools = attach_modules(find_moe(model.modules))
    _hook_prefill(type(model))
    return pools


def _hook_prefill(cls):
    """Elastic mode: hand the flex chunks back BEFORE the first layer of a big prefill chunk runs (dense + KDA layers
    ahead of the first MoE layer already need the activation memory)."""
    if getattr(cls, "_glm53_ec_prefill_hooked", False):
        return
    for name in ("prefill", "forward"):
        orig = getattr(cls, name)

        def wrapped(self, input_ids, params=None, _orig=orig):
            try:
                n = int(input_ids.shape[-1]) * int(input_ids.shape[0]) if hasattr(input_ids, "shape") else 0
            except Exception:
                n = 0
            for p in POOLS:
                if p.elastic and p.stage_n and not p.in_prefill and n * 8 >= p.stage_min:
                    p._elastic_enter()
            return _orig(self, input_ids, params)
        setattr(cls, name, wrapped)
    cls._glm53_ec_prefill_hooked = True


def attach_modules(mods):
    by_dev = {}
    for m in mods:
        by_dev.setdefault(str(m.device), []).append(m)
    reserve = float(os.environ.get("GLM53_EC_RESERVE_GB", "2.5")) * 1024 ** 3
    mx = os.environ.get("GLM53_EC_MAX_SLOTS")
    admit_max = int(os.environ.get("GLM53_EC_ADMIT_MAX", "64"))
    stage_bytes = float(os.environ.get("GLM53_EC_STAGE_GB", "0")) * 1024 ** 3
    stage_min = int(os.environ.get("GLM53_EC_STAGE_MIN", "8192"))
    warm = os.environ.get("GLM53_EC_WARM")
    scores = json.load(open(warm)) if warm else None
    for dev, ms in by_dev.items():
        p = Pool(torch.device(dev), ms, reserve, int(mx) if mx else None, admit_max, stage_bytes, stage_min)
        POOLS.append(p)
        if scores:
            p.warm(scores)
        if p.bg is not None:                            # [E3] background admission on after the warm fill
            p.freq.zero_()
            p._bg_write(p.bg_cfg["on"])
        for m in ms:
            orig = m.routing_fn

            def routed(bsz, cfg, z, params, _orig=orig, _m=m):
                sel, w = _orig(bsz, cfg, z, params)
                pool, li = _m._ec
                if not params.get("autosplit_measure") and not params.get("tp_warmup"):
                    pool.step(li, sel, sel.numel() <= pool.admit_max)
                return sel, w
            m.routing_fn = routed
        print(f" -- expert_cache: {dev}: {len(ms)} MoE layers (experts {p.first}..{p.first + p.E - 1}), {p.S} slots x "
              f"{p.rec / 1e6:.2f} MB = {p.S * p.rec / 1e9:.2f} GB ({p.S / max(1, len(ms)):.1f}/layer), admit <= "
              f"{admit_max} picks, stage 2 x {p.stage_n} experts (>= {stage_min} picks)"
              f"{', warm ' + warm if warm else ''}", flush=True)
    return POOLS


def summary():
    return [p.summary() for p in POOLS]


def verify():
    return [p.verify() for p in POOLS]


def bg_set(**kw):
    """[E3] runtime knobs: on, qmax, per_call, gate, decay, skip_remote (None = unchanged)."""
    return [p.bg_set(**kw) for p in POOLS]


def set_rt(attached=None):
    """[E3b] cpu_tier: attached=None after setting m._rt_res (rebuild the mask), True/False on attach/detach."""
    for p in POOLS:
        p.set_rt(attached)
