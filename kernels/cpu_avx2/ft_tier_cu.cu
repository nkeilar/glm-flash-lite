// FreeToken CPU tier: GPU side. Two kernels per MoE layer in decode, both on the compute stream, no host sync:
//  ft_split  (after the router, before the expert cache step): pick the coldest cache misses for the CPU
//            (greedy on a cost model), publish the job to pinned host memory (x rows, picks, then the seq flag),
//            and hand exllamav3 a selection with those picks removed (id -1, weight 0: the cache step skips
//            them, the fused MoE kernel treats weight 0 as inactive)
//  ft_combine (after the GPU experts): wait for the worker's done flag, add its fp32 partial into the MoE output
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_fp16.h>

// policy p[]: 0 t_zc (ms per zero-copy miss incl. its GPU compute), 1 t_hit (ms per GPU-resident expert),
// 2 cpu_a (ms fixed per CPU job incl. handoff), 3 cpu_b (ms per CPU expert), 4 cpu_tok (extra fraction per extra
// token of an expert), 5 max_cpu, 6 force_n (-1 = cost model), 7 handshake (1 = publish even empty jobs),
// [E4] 8 three-target on (1 while the remote 3090 tier is attached and GLM53_CT_E4=1), 9 r_a (ms: 3090 round trip of
// a job), 10 r_b (ms per remote pick). With 8 on, misses the 3090s hold (rres) are costed on their own lane, which runs
// in parallel with the local CPU work: CPU lane = max(a + b * local, r_a + r_b * remote); the split searches both counts
// (coldest first in each class). The client sends resident CPU-lane picks to the 3090s exactly as before.
// [bdw-opt] 12-19 measured CPU-lane cost (ms) for 1..8 local experts (job time incl. handoff; > 8: + cpu_b per expert),
// replacing cpu_a + cpu_b * k when p[12] > 0 (the multi-token extra cpu_b * cpu_tok * (cnt - 1) still applies).
__global__ void ft_split_k(const int64_t* __restrict__ sel, const __half* __restrict__ w, const __half* __restrict__ z,
    int n, int bsz, int topk, int H, int E, int first, const int* __restrict__ slotof, const float* __restrict__ score,
    const float* __restrict__ p, volatile long long* ctrl, __half* hx, int* hpicks, long long seq, int li, int keep_ids,
    int64_t* sel_out, __half* w_out, long long* dflag, long long* stats, const int* __restrict__ rres)
{
    // E <= 1024. cnt[e] = tokens that picked local expert e; cpu[e] = 1 if the CPU takes e
    __shared__ int cnt[1024];
    __shared__ unsigned char cpu[1024];
    __shared__ int npk, publish;
    __shared__ int s_miss[1024];
    __shared__ int s_rmiss[1024];
    const int t = threadIdx.x;
#ifdef FT_ENABLE_LUT
    // One ingress generation owns hx/picks/hout until its CPU drain word is
    // published. A sticky failure refuses every later handoff until restart.
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
    __syncthreads();
    for (int s = t; s < n; s += blockDim.x)
    {
        const long long e = sel[s] - first;
        if (__half2float(w[s]) != 0.0f && e >= 0 && e < E) atomicAdd(&cnt[e], 1);
    }
    __syncthreads();
    if (t == 0)
    {
        // misses sorted coldest first (static routing-frequency score); decode has <= 8 x bsz of them
        // miss list in SHARED memory: a 4 KB per-thread local array made the runtime grow the device-wide
        // local-memory reservation (~0.5 GB) and the next launch hit OOM once the elastic cache had re-taken all
        // free VRAM
        int nm = 0, nh = 0, nr = 0;
        int* miss = s_miss;
        int* rmiss = s_rmiss;
        const bool tri = p[8] > 0.5f && rres != nullptr && (int) p[6] < 0;
        const bool lut = p[12] > 0.0f;   // [bdw-opt] measured CPU-lane cost table p[12..19] for 1..8 local experts
        for (int e = 0; e < E; ++e)
        {
            if (!cnt[e]) continue;
            if (slotof[e] >= 0) { ++nh; continue; }
            int* l = (tri && rres[e]) ? rmiss : miss;
            int b = (l == rmiss) ? nr++ : nm++;
            while (b > 0 && score[l[b - 1]] > score[e]) { l[b] = l[b - 1]; --b; }
            l[b] = e;
        }
        int best = 0, bestr = 0;
        const int force = (int) p[6], maxc = min((int) p[5], nm);
        if (force >= 0) best = min(force, nm);
        else if (tri)
        {
            // [E4] three targets: kr coldest 3090-held misses -> 3090 lane, k coldest others -> local CPU, rest zero-copy
            float bt = 1e30f, cr = 0.0f;
            for (int kr = 0; kr <= nr; ++kr)
            {
                if (kr) cr += p[10] * (1.0f + p[4] * (cnt[rmiss[kr - 1]] - 1));
                const float rem = kr ? p[9] + cr : 0.0f;
                float c = 0.0f, cx = 0.0f;
                for (int k = 0; k <= maxc; ++k)
                {
                    if (k) { c += p[3] * (1.0f + p[4] * (cnt[miss[k - 1]] - 1)); cx += p[3] * p[4] * (cnt[miss[k - 1]] - 1); }
                    const int z = nm - k + nr - kr;
                    const float gpu = p[1] * (nh + z) + p[0] * z;
                    const float loc = k ? (lut ? p[11 + (k < 8 ? k : 8)] + (k > 8 ? p[3] * (k - 8) : 0.0f) + cx : p[2] + c) : 0.0f;
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
                if (k) { c += p[3] * (1.0f + p[4] * (cnt[miss[k - 1]] - 1)); cx += p[3] * p[4] * (cnt[miss[k - 1]] - 1); }
                const float gpu = p[1] * (nh + nm - k) + p[0] * (nm - k);
                const float loc = k ? (lut ? p[11 + (k < 8 ? k : 8)] + (k > 8 ? p[3] * (k - 8) : 0.0f) + cx : p[2] + c) : 0.0f;
                const float tt = fmaxf(gpu, loc);
                if (tt < bt - 1e-6f) { bt = tt; best = k; }
            }
        }
        for (int k = 0; k < best; ++k) cpu[miss[k]] = 1;
        publish = best + bestr > 0 || p[7] > 0.5f;   // [E4] a job of only 3090-held picks must still be published
        stats[0] += nh; stats[1] += nm + nr; stats[2] += best + bestr; stats[3] += 1;
        npk = 0;
        if (publish)
            for (int s = 0; s < n; ++s)
            {
                const long long e = sel[s] - first;
                if (e < 0 || e >= E || !cpu[e] || __half2float(w[s]) == 0.0f) continue;
                hpicks[npk * 3] = s / topk; hpicks[npk * 3 + 1] = (int) e; hpicks[npk * 3 + 2] = __float_as_int(__half2float(w[s]));
                ++npk;
            }
    }
    __syncthreads();
    // CPU picks leave the GPU selection: id -1 + weight 0 in the fused decode path (skipped entirely, and the
    // cache step ignores them); keep_ids (bsz > MAX_BSZN paths, which index by expert id) keeps the id with
    // weight 0 -> exact zero contribution
    for (int s = t; s < n; s += blockDim.x)
    {
        const long long e = sel[s];
        const bool c = e - first >= 0 && e - first < E && cpu[e - first];
        sel_out[s] = (c && !keep_ids) ? -1 : e;
        w_out[s] = c ? __float2half(0.0f) : w[s];
    }
    if (publish)
    {
        const int nx = bsz * H / 8;   // int4 = 8 halves
        for (int i = t; i < nx; i += blockDim.x) reinterpret_cast<int4*>(hx)[i] = reinterpret_cast<const int4*>(z)[i];
        __threadfence_system();
        __syncthreads();
        if (t == 0)
        {
            ctrl[2] = li; ctrl[3] = bsz; ctrl[4] = npk;
            __threadfence_system();
            ctrl[0] = seq;
            __threadfence_system();
            dflag[0] = seq;
        }
    }
    else if (t == 0) dflag[0] = 0;
}

__global__ void ft_combine_k(void* out, int is_half, int n, const long long* __restrict__ dflag, volatile long long* ctrl,
    const float* hout, long long* stats, long long timeout_ns)
{
#ifdef FT_ENABLE_LUT
    // A single block makes the timeout decision once for every output lane.
    // No block can add a successful partial while another block aborts it.
    __shared__ int go;
    if (threadIdx.x == 0)
    {
        const long long sequence = dflag[0];
        go = sequence == 0 ? 0 : 1;
        if (sequence != 0)
        {
            unsigned long long started, elapsed;
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(started));
            elapsed = 0;
            if (sequence > 0 && ctrl[1] == sequence) atomicAdd((unsigned long long*)&stats[7], 1ull);
            while (sequence > 0 && !ctrl[5] && !ctrl[6] && ctrl[1] != sequence && (long long)elapsed <= timeout_ns)
            {
                unsigned long long current;
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(current));
                elapsed = current - started;
            }
            if (sequence < 0 || ctrl[5] || ctrl[6] || ctrl[1] != sequence)
            {
                go = -1;
                ctrl[5] = 1; ctrl[7] = sequence > 0 ? sequence : ctrl[0];
                __threadfence_system();
                if (!ctrl[6]) ctrl[6] = 2;
                __threadfence_system();
                atomicAdd((unsigned long long*)&stats[6], 1ull);
            }
            atomicAdd((unsigned long long*)&stats[4], elapsed);
            atomicAdd((unsigned long long*)&stats[5], 1ull);
        }
    }
    __syncthreads();
    if (!go) return;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        // Poison is secondary protection. The Python LUT hook synchronizes this
        // stream and calls tier_lut_check before a token can leave the request.
        const float value = go > 0 ? __ldcv(hout + i) : __int_as_float(0x7fffffff);
        if (is_half) { __half* target = (__half*)out; target[i] = go > 0 ? __float2half(__half2float(target[i]) + value) : __float2half(value); }
        else ((float*)out)[i] = go > 0 ? ((float*)out)[i] + value : value;
    }
#else
    __shared__ int go;
    if (threadIdx.x == 0)
    {
        const long long s = dflag[0];
        go = s != 0;
        if (go)
        {
            unsigned long long t0, t1;
            // [rt] stats[7]: combines where the CPU partial was already done when the GPU arrived (= the GPU path was
            // the slower runner of this layer); T-18 hypothesis H1
            if (ctrl[1] >= s && blockIdx.x == 0) atomicAdd((unsigned long long*) &stats[7], 1ull);
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
            t1 = t0;
            while (ctrl[1] < s)
            {
                asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t1));
                if ((long long) (t1 - t0) > timeout_ns) { go = 0; if (blockIdx.x == 0) atomicAdd((unsigned long long*) &stats[6], 1ull); break; }
            }
            if (blockIdx.x == 0) { atomicAdd((unsigned long long*) &stats[4], (unsigned long long) (t1 - t0)); atomicAdd((unsigned long long*) &stats[5], 1ull); }
        }
    }
    __syncthreads();
    if (!go) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
    {
        const float v = __ldcv(hout + i);
        if (is_half) { __half* o = (__half*) out; o[i] = __float2half(__half2float(o[i]) + v); }
        else ((float*) out)[i] += v;
    }
#endif
}

void ft_split(torch::Tensor sel, torch::Tensor w, torch::Tensor z, int64_t E, int64_t first, torch::Tensor slotof, torch::Tensor score,
              torch::Tensor pol, int64_t ctrl, int64_t hx, int64_t hpicks, int64_t seq, int64_t li, int64_t keep_ids,
              torch::Tensor sel_out, torch::Tensor w_out, torch::Tensor dflag, torch::Tensor stats, torch::Tensor rres)
{
    c10::cuda::CUDAGuard g(sel.device());
    TORCH_CHECK(pol.numel() >= 20, "ft_split: policy needs 20 entries");
    TORCH_CHECK(rres.numel() == 0 || (rres.dtype() == at::kInt && rres.numel() >= E), "ft_split: rres int32 [E] or empty");
    auto st = at::cuda::getCurrentCUDAStream(sel.device().index());
    TORCH_CHECK(sel.dtype() == at::kLong && w.dtype() == at::kHalf && z.dtype() == at::kHalf && z.is_contiguous(), "dtypes");
#ifdef FT_ENABLE_LUT
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
#ifdef FT_ENABLE_LUT
    TORCH_CHECK(bsz >= 1 && bsz <= 4 && topk >= 1 && topk <= 8 && n == bsz * topk && E >= 1 && E <= 288 && H == 4096,
                "LUT split requires M=1..4, topk<=8 and model geometry");
    TORCH_CHECK(seq >= 0 && seq < 0x7fffffffffffffffLL, "LUT job generation must be positive without wrap");
#endif
    ft_split_k<<<1, 256, 0, st>>>(sel.data_ptr<int64_t>(), (const __half*) w.data_ptr(), (const __half*) z.data_ptr(), n, bsz, topk, H,
        (int) E, (int) first, slotof.data_ptr<int>(), score.data_ptr<float>(), pol.data_ptr<float>(), (volatile long long*) ctrl,
        (__half*) hx, (int*) hpicks, seq, (int) li, (int) keep_ids, sel_out.data_ptr<int64_t>(), (__half*) w_out.data_ptr(),
        (long long*) dflag.data_ptr<int64_t>(), (long long*) stats.data_ptr<int64_t>(),
        rres.numel() ? rres.data_ptr<int>() : nullptr);
}

void ft_combine(torch::Tensor out, torch::Tensor dflag, int64_t ctrl, int64_t hout, torch::Tensor stats, int64_t timeout_ns)
{
    c10::cuda::CUDAGuard g(out.device());
    auto st = at::cuda::getCurrentCUDAStream(out.device().index());
    TORCH_CHECK(out.is_contiguous() && (out.dtype() == at::kFloat || out.dtype() == at::kHalf), "out fp32/fp16 contiguous");
#ifdef FT_ENABLE_LUT
    TORCH_CHECK(out.numel() >= 1 && out.numel() <= 4 * 4096 && dflag.device() == out.device() && stats.device() == out.device(),
                "LUT combine buffers require one CUDA device and bounded output");
#endif
    const int n = (int) out.numel();
#ifdef FT_ENABLE_LUT
    TORCH_CHECK(dflag.dtype() == at::kLong && dflag.is_contiguous() && dflag.numel() >= 1 &&
                stats.dtype() == at::kLong && stats.is_contiguous() && stats.numel() >= 8,
                "LUT combine flag/stat buffer mismatch");
    TORCH_CHECK(n >= 1 && n <= 4 * 4096 && timeout_ns > 0 && timeout_ns <= 3600000000000LL, "LUT combine shape/deadline out of bounds");
    ft_combine_k<<<1, 256, 0, st>>>(out.data_ptr(), out.dtype() == at::kHalf, n, (const long long*) dflag.data_ptr<int64_t>(),
        (volatile long long*) ctrl, (const float*) hout, (long long*) stats.data_ptr<int64_t>(), timeout_ns);
#else
    ft_combine_k<<<16, 256, 0, st>>>(out.data_ptr(), out.dtype() == at::kHalf, n, (const long long*) dflag.data_ptr<int64_t>(),
        (volatile long long*) ctrl, (const float*) hout, (long long*) stats.data_ptr<int64_t>(), timeout_ns);
#endif
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("ft_split", &ft_split);
    m.def("ft_combine", &ft_combine);
}
