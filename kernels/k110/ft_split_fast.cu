// K110: drop-in faster ft_split (CPU-tier handoff, kernels/cpu_avx2/ft_tier_cu.cu) with identical outputs.
// Stock: thread 0 scans all E experts, insertion-sorts misses through a local array, and walks the picks serially
// from global memory (16 us at bsz 1 without a job, 22 us with one, 38-40 us at bsz 4 on the 3090).
// Here: picks staged in shared memory by all threads; unique experts found from the picks (n <= 64) instead of
// scanning E; each miss's position in the (score asc, expert id asc) order computed in parallel (= the stock stable
// insertion sort, strict >); the tiny cost-model loop stays serial with the same float expressions; the published
// pick list is built in pick order by a warp ballot prefix (same order as the stock serial loop).
// [E4] Three-target split (p[8] on while the remote 3090 tier is attached): misses the 3090s hold (rres) are ranked in
// their own class and costed on their own lane, exactly as the stock kernel (ft_tier_cu.cu) does; outputs identical
// to the stock kernel for both models (tests/ft_split_e4_test.py compares fast vs stock vs a float32 reference).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

#define MAXP 64

__global__ void ft_split_fast_k(const int64_t* __restrict__ sel, const __half* __restrict__ w, const __half* __restrict__ z,
    int n, int bsz, int topk, int H, int E, int first, const int* __restrict__ slotof, const float* __restrict__ score,
    const float* __restrict__ p, volatile long long* ctrl, __half* hx, int* hpicks, long long seq, int li, int keep_ids,
    int64_t* sel_out, __half* w_out, long long* dflag, long long* stats, const int* __restrict__ rres)
{
    __shared__ int cnt[1024];
    __shared__ unsigned char cpu[1024];
    __shared__ long long s_e[MAXP];          // sel[s] (global id)
    __shared__ float s_w[MAXP];               // weight as float
    __shared__ int u_e[MAXP], nu;             // unique local experts (first occurrence order)
    __shared__ int u_hit[MAXP];
    __shared__ float u_sc[MAXP];
    __shared__ int u_cls[MAXP];
    __shared__ int miss[MAXP], rmiss[MAXP], nm_s, nr_s, nh_s, best_s, publish, npk;
    __shared__ float pol[20];
    const int t = threadIdx.x;
#ifdef FT_ENABLE_LUT
    // Same ingress guard as the stock ft_split_k (kernels/cpu_avx2/ft_tier_cu.cu): one ingress generation owns
    // hx/picks/hout until its CPU drain word is published, and a sticky failure refuses every later handoff
    // (all picks stay on the GPU, dflag = -1) until restart.
    __shared__ int ingress_failed;
    if (t == 0)
    {
        ingress_failed = ctrl[5] || ctrl[6] || seq < 0 || (ctrl[0] != ctrl[8]) || (seq > 0 && seq <= ctrl[0]);
        if (ingress_failed && !ctrl[6])
        {
            ctrl[5] = 1; ctrl[7] = seq; __threadfence_system(); ctrl[6] = 4; __threadfence_system();
        }
    }
    __syncthreads();
    if (ingress_failed)
    {
        for (int s = t; s < n; s += blockDim.x) { sel_out[s] = sel[s]; w_out[s] = w[s]; }
        if (t == 0) dflag[0] = -1;
        return;
    }
#endif
    for (int e = t; e < E; e += blockDim.x) { cnt[e] = 0; cpu[e] = 0; }
    if (t < 20) pol[t] = p[t];
    if (t == 0) nu = 0;
    for (int s = t; s < n; s += blockDim.x) { s_e[s] = sel[s]; s_w[s] = __half2float(w[s]); }
    __syncthreads();
    for (int s = t; s < n; s += blockDim.x)
    {
        const long long e = s_e[s] - first;
        if (s_w[s] != 0.0f && e >= 0 && e < E)
        {
            if (atomicAdd(&cnt[e], 1) == 0)
            {
                int k = atomicAdd(&nu, 1);
                u_e[k] = (int) e;
            }
        }
    }
    __syncthreads();
    const int U = nu;
    const bool tri = pol[8] > 0.5f && rres != nullptr && (int) pol[6] < 0;
    const bool lut = pol[12] > 0.0f;   // [bdw-opt] measured CPU-lane cost table (see ft_tier_cu.cu)
    if (t < U)
    {
        const int e = u_e[t];
        u_hit[t] = slotof[e] >= 0;
        u_sc[t] = score[e];
        u_cls[t] = (tri && rres[e]) ? 1 : 0;      // [E4] 1 = held by the 3090 tier: its own (coldest-first) list
    }
    __syncthreads();
    if (t < U && !u_hit[t])
    {
        // rank in the stock order within the miss's class: ascending score, ties by ascending expert id
        const int e = u_e[t];
        const float sc = u_sc[t];
        const int cl = u_cls[t];
        int r = 0;
        for (int j = 0; j < U; ++j)
        {
            if (u_hit[j] || u_cls[j] != cl) continue;
            const int e2 = u_e[j];
            const float s2 = u_sc[j];
            if (s2 < sc || (s2 == sc && e2 < e)) ++r;
        }
        if (cl) rmiss[r] = e; else miss[r] = e;
    }
    if (t == 0)
    {
        int nh = 0, nr = 0;
        for (int j = 0; j < U; ++j) { nh += u_hit[j]; nr += !u_hit[j] && u_cls[j]; }
        nh_s = nh; nr_s = nr; nm_s = U - nh - nr;
    }
    __syncthreads();
    if (t == 0)
    {
        const int nm = nm_s, nh = nh_s, nr = nr_s;
        int best = 0, bestr = 0;
        const int force = (int) pol[6], maxc = min((int) pol[5], nm);
        if (force >= 0) best = min(force, nm);
        else if (tri)
        {
            // [E4] same search and float expressions as ft_tier_cu.cu's three-target branch
            float bt = 1e30f, cr = 0.0f;
            for (int kr = 0; kr <= nr; ++kr)
            {
                if (kr) cr += pol[10] * (1.0f + pol[4] * (cnt[rmiss[kr - 1]] - 1));
                const float rem = kr ? pol[9] + cr : 0.0f;
                float c = 0.0f, cx = 0.0f;
                for (int k = 0; k <= maxc; ++k)
                {
                    if (k) { c += pol[3] * (1.0f + pol[4] * (cnt[miss[k - 1]] - 1)); cx += pol[3] * pol[4] * (cnt[miss[k - 1]] - 1); }
                    const int zz = nm - k + nr - kr;
                    const float gpu = pol[1] * (nh + zz) + pol[0] * zz;
                    const float loc = k ? (lut ? pol[11 + (k < 8 ? k : 8)] + (k > 8 ? pol[3] * (k - 8) : 0.0f) + cx : pol[2] + c) : 0.0f;
                    const float tt = fmaxf(gpu, fmaxf(loc, rem));
                    if (tt < bt - 1e-6f) { bt = tt; best = k; bestr = kr; }
                }
            }
            for (int k = 0; k < bestr; ++k) cpu[rmiss[k]] = 1;
        }
        else
        {
            float bt = 1e30f, c = 0.0f, cx = 0.0f;
            for (int k = 0; k <= maxc; ++k)
            {
                if (k) { c += pol[3] * (1.0f + pol[4] * (cnt[miss[k - 1]] - 1)); cx += pol[3] * pol[4] * (cnt[miss[k - 1]] - 1); }
                const float gpu = pol[1] * (nh + nm - k) + pol[0] * (nm - k);
                const float loc = k ? (lut ? pol[11 + (k < 8 ? k : 8)] + (k > 8 ? pol[3] * (k - 8) : 0.0f) + cx : pol[2] + c) : 0.0f;
                const float tt = fmaxf(gpu, loc);
                if (tt < bt - 1e-6f) { bt = tt; best = k; }
            }
        }
        for (int k = 0; k < best; ++k) cpu[miss[k]] = 1;
        best_s = best + bestr;
        publish = best + bestr > 0 || pol[7] > 0.5f;
        stats[0] += nh; stats[1] += nm + nr; stats[2] += best + bestr; stats[3] += 1;
    }
    __syncthreads();
    // published picks in pick order: warp 0 ballot prefix over s
    if (publish && t < 32)
    {
        int base = 0;
        for (int s0 = 0; s0 < n; s0 += 32)
        {
            const int s = s0 + t;
            bool take = false;
            long long e = 0;
            if (s < n)
            {
                e = s_e[s] - first;
                take = e >= 0 && e < E && cpu[e] && s_w[s] != 0.0f;
            }
            const unsigned m = __ballot_sync(0xffffffffu, take);
            if (take)
            {
                const int k = base + __popc(m & ((1u << t) - 1u));
                hpicks[k * 3] = s / topk; hpicks[k * 3 + 1] = (int) e; hpicks[k * 3 + 2] = __float_as_int(s_w[s]);
            }
            base += __popc(m);
        }
        if (t == 0) npk = base;
    }
    for (int s = t; s < n; s += blockDim.x)
    {
        const long long e = s_e[s];
        const bool c = e - first >= 0 && e - first < E && cpu[e - first];
        sel_out[s] = (c && !keep_ids) ? -1 : e;
        w_out[s] = c ? __float2half(0.0f) : w[s];
    }
    if (publish)
    {
        const int nx = bsz * H / 8;
        for (int i = t; i < nx; i += blockDim.x) reinterpret_cast<int4*>(hx)[i] = reinterpret_cast<const int4*>(z)[i];
        __threadfence_system();
        __syncthreads();
        if (t == 0)
        {
            ctrl[2] = li; ctrl[3] = bsz; ctrl[4] = npk;
            __threadfence_system();
            ctrl[0] = seq;
            dflag[0] = seq;                        // device flag, read by ft_combine later on the same stream
        }
    }
    else if (t == 0) dflag[0] = 0;
}

void ft_split(torch::Tensor sel, torch::Tensor w, torch::Tensor z, int64_t E, int64_t first, torch::Tensor slotof, torch::Tensor score,
              torch::Tensor pol, int64_t ctrl, int64_t hx, int64_t hpicks, int64_t seq, int64_t li, int64_t keep_ids,
              torch::Tensor sel_out, torch::Tensor w_out, torch::Tensor dflag, torch::Tensor stats, torch::Tensor rres)
{
    c10::cuda::CUDAGuard g(sel.device());
    TORCH_CHECK(pol.numel() >= 20, "ft_split_fast: policy needs 20 entries");
    TORCH_CHECK(rres.numel() == 0 || (rres.dtype() == at::kInt && rres.numel() >= E), "ft_split_fast: rres int32 [E] or empty");
    auto st = at::cuda::getCurrentCUDAStream(sel.device().index());
    TORCH_CHECK(sel.dtype() == at::kLong && w.dtype() == at::kHalf && z.dtype() == at::kHalf && z.is_contiguous(), "dtypes");
#ifdef FT_ENABLE_LUT
    // Same buffer/shape/generation admission as the stock LUT ft_split (kernels/cpu_avx2/ft_tier_cu.cu).
    for (const auto& tensor : {w, z, slotof, score, pol, sel_out, w_out, dflag, stats})
        TORCH_CHECK(tensor.device() == sel.device() && tensor.is_contiguous(), "LUT split buffers require one CUDA device and contiguous storage");
    TORCH_CHECK(rres.numel() == 0 || (rres.device() == sel.device() && rres.is_contiguous()), "LUT remote residency device/layout mismatch");
    TORCH_CHECK(z.dim() == 2 && z.size(0) >= 1 && z.size(0) <= 4 && z.size(1) == 4096 && sel.numel() == w.numel() && sel.numel() <= z.size(0) * 8,
                "LUT split input shape mismatch");
    TORCH_CHECK(sel.is_contiguous() && w.is_contiguous() && slotof.dtype() == at::kInt && slotof.numel() >= E &&
                score.dtype() == at::kFloat && score.numel() >= E && pol.dtype() == at::kFloat &&
                sel_out.dtype() == at::kLong && sel_out.numel() >= sel.numel() && w_out.dtype() == at::kHalf && w_out.numel() >= w.numel() &&
                dflag.dtype() == at::kLong && dflag.numel() >= 1 && stats.dtype() == at::kLong && stats.numel() >= 8,
                "LUT split buffer dtype/capacity mismatch");
#endif
    const int bsz = (int) z.size(0), H = (int) z.size(1), n = (int) sel.numel(), topk = n / bsz;
    TORCH_CHECK(E <= 1024 && H % 8 == 0, "ft_split: E <= 1024");
    TORCH_CHECK(n <= MAXP, "ft_split_fast: <= 64 picks (the stock kernel serves larger calls)");
#ifdef FT_ENABLE_LUT
    TORCH_CHECK(bsz >= 1 && bsz <= 4 && topk >= 1 && topk <= 8 && n == bsz * topk && E >= 1 && E <= 288 && H == 4096,
                "LUT split requires M=1..4, topk<=8 and model geometry");
    TORCH_CHECK(seq >= 0 && seq < 0x7fffffffffffffffLL, "LUT job generation must be positive without wrap");
#endif
    ft_split_fast_k<<<1, 256, 0, st>>>(sel.data_ptr<int64_t>(), (const __half*) w.data_ptr(), (const __half*) z.data_ptr(), n, bsz, topk, H,
        (int) E, (int) first, slotof.data_ptr<int>(), score.data_ptr<float>(), pol.data_ptr<float>(), (volatile long long*) ctrl,
        (__half*) hx, (int*) hpicks, seq, (int) li, (int) keep_ids, sel_out.data_ptr<int64_t>(), (__half*) w_out.data_ptr(),
        (long long*) dflag.data_ptr<int64_t>(), (long long*) stats.data_ptr<int64_t>(),
        rres.numel() ? rres.data_ptr<int>() : nullptr);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ft_split", &ft_split);
}
