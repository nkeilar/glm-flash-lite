"""CPU tier for GLM-5.3-Flash on 1 GPU (kernels/cpu_avx2/ft_core.h, ft_tier_ext.cpp, ft_tier_cu.cu).

Per MoE layer in decode (bsz <= GLM53_CT_MAXBSZ), right after the router and BEFORE expert_cache's step:
  ft_split (GPU, 1 block): hits / misses from the cache's slotof; misses sorted coldest first by a static routing-frequency
      score; n_cpu from a cost model (GPU: t_hit per resident expert + t_zc per zero-copy miss; CPU: a + b per expert);
      the n_cpu coldest misses become a CPU job published to pinned host memory (x rows, picks, seq flag); exllamav3 gets
      the selection with those picks removed (id -1 / weight 0) -> the cache does not admit them (warm misses still go
      zero-copy + admit), the fused MoE kernel skips them
  CPU worker (C++ thread + 21 pinned pool threads) computes the job with ft_mul1 (I16 for <= 2 tokens/expert, AFFINE
      otherwise) straight from the native pinned home copy (exl3_tiers arenas), writes an fp32 partial
  ft_combine (GPU, in BlockSparseMLP.cpu_split_combine, after the GPU experts): waits on the done flag, adds the partial
Prefill / big batches: tier off (GPU staging path unchanged).

Env: GLM53_CPU_TIER=1, GLM53_CT_THREADS (default: number of CPUs in GLM53_CT_CPUS), GLM53_CT_CPUS ("auto" = one
     logical CPU per physical core minus the first 2, e.g. "2-23" on a 24-core part; or an explicit list "2-23"), GLM53_CT_MODE (-1 auto | 0 AFFINE | 1 EXACT | 2 I16),
     GLM53_CT_MAXBSZ (4), GLM53_CT_STATS (score JSON; default GLM53_EC_WARM / GLM53_ZC_STATS),
     GLM53_CT_TZC (0.40), GLM53_CT_THIT (0.023), GLM53_CT_A (0.11), GLM53_CT_B (0.104), GLM53_CT_TOK (0.2),
     GLM53_CT_MAXN (32), GLM53_CT_FORCE_N (-1), GLM53_CT_SWZ (1: CPU-side block-contiguous copy, +114.6 GB RAM),
     GLM53_CT_VALIDATE (0; 1 = panel through the CPU kernel: all misses of every forward <= GLM53_CT_MAXBSZ tokens), GLM53_CT_HANDSHAKE (0), GLM53_CT_SELFTEST (1)
Hook (serve.py): cpu_tier.wrap(model) before expert_cache.attach(model); cpu_tier.start() after it; /stats: summary().
[rt] Remote expert tier (T-18): GLM53_RT=host:port (expert server, ops/remote-tier/expert_server.py), GLM53_RT_TIMEOUT_US
     (20000), GLM53_RT_PLAN (plan file used when the server is not up
     at start), GLM53_RT_CHECK (0; N = every Nth remote call also computes its picks locally and reports the rel RMS difference). At start() the server's plan (which experts it holds) is fetched over the same port; the CPU job then sends
     resident picks there and computes the rest here; any failure computes them here (never dropped).
"""
import json, os, time
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_KDIR = os.path.join(os.path.dirname(_HERE), "kernels", "cpu_avx2")
_H = None
_CU = None
S = {"mods": [], "dev": None, "seq": 0, "started": False, "selftest": None}
# [bdw-opt trace] GLM53_TRACE_ROUTES=<path>: append one JSON line per decode MoE step: [layer, orig picks..., post-split
# picks...] (post-split: picks sent to the CPU tier are -1). Forces two device syncs per layer: TRACE RUNS ONLY.
TR = {"path": os.environ.get("GLM53_TRACE_ROUTES"), "buf": []}


def set_trace(path=None):
    """[E5 prep] runtime routing trace (no restart): path = start appending rows there, None = flush and stop."""
    _trace_flush()
    TR["path"] = path or None
    return {"trace": TR["path"]}


def _trace_flush():
    if TR["path"] and TR["buf"]:
        with open(TR["path"], "a") as f:
            for row in TR["buf"]:
                f.write(json.dumps(row) + "\n")
        TR["buf"].clear()


def _build():
    global _H, _CU
    if _H is not None:
        return
    from torch.utils.cpp_extension import load
    d = os.environ.get("GLM53_CT_BUILD", os.path.join(os.path.dirname(_HERE), "build", "ct"))
    os.makedirs(d, exist_ok=True)
    # Portable ISA flags (AVX2 + FMA + F16C is all the kernel uses); the campaign build used -march=znver3.
    isa = os.environ.get("GLM53_CT_CFLAGS", "-mavx2 -mfma -mf16c -mtune=znver3").split()
    _H = load(name="ft_tier_host", sources=[os.path.join(_KDIR, "ft_tier_ext.cpp")], build_directory=d,
              extra_cflags=["-O3", *isa, "-std=c++17", "-I" + _KDIR], extra_ldflags=["-lpthread"], verbose=False)
    _CU = load(name="ft_tier_cu", sources=[os.path.join(_KDIR, "ft_tier_cu.cu")], build_directory=d,
               extra_cuda_cflags=["-O3"], verbose=False)


def preflight():
    """Validate the format before any CPU-tier allocation or extension build."""
    from lut_manifest import preflight as format_preflight
    return format_preflight()


def auto_cpus(skip=2):
    """One logical CPU per physical core (first SMT sibling) among the CPUs this process may run on, minus the first
    `skip` cores (left to the GPU-feeding Python thread and the OS). On a 24-core EPYC 7443P with SMT: 2-23 (22)."""
    allowed = sorted(os.sched_getaffinity(0))
    seen, firsts = set(), []
    for c in allowed:
        try:
            core = open(f"/sys/devices/system/cpu/cpu{c}/topology/core_id").read().strip()
            pkg = open(f"/sys/devices/system/cpu/cpu{c}/topology/physical_package_id").read().strip()
        except OSError:
            core, pkg = str(c), "0"
        if (pkg, core) not in seen:
            seen.add((pkg, core)); firsts.append(c)
    return firsts[skip:] if len(firsts) > skip + 1 else firsts[-1:]


def _cpus():
    spec = os.environ.get("GLM53_CT_CPUS", "auto")
    if spec == "auto":
        return auto_cpus()
    out = []
    for part in spec.split(","):
        a, _, b = part.partition("-")
        out += list(range(int(a), int(b or a) + 1))
    return out


def _lut(spec):
    """[bdw-opt] "0.45,0.74,..." -> 8 floats (CPU-lane ms for 1..8 local experts; missing entries extrapolate the last step);
    empty -> 8 zeros = off (cost model cpu_a + cpu_b * k)."""
    v = [float(x) for x in spec.split(",") if x.strip()] if spec else []
    while v and len(v) < 8:
        v.append(v[-1] + (v[-1] - v[-2] if len(v) > 1 else 0.25))
    return v[:8] if v else [0.0] * 8


def _policy(dev):
    g = lambda k, d: float(os.environ.get(k, d))
    return torch.tensor([g("GLM53_CT_TZC", 0.40), g("GLM53_CT_THIT", 0.023), g("GLM53_CT_A", 0.11), g("GLM53_CT_B", 0.104),
                         g("GLM53_CT_TOK", 0.2), g("GLM53_CT_MAXN", 32), g("GLM53_CT_FORCE_N", -1), g("GLM53_CT_HANDSHAKE", 0),
                         0.0, g("GLM53_CT_RA", 0.37), g("GLM53_CT_RB", 0.03), 0.0,    # [E4] 8 on (auto thread), 9 r_a, 10 r_b
                         *_lut(os.environ.get("GLM53_CT_LUT", ""))],   # [bdw-opt] 12-19 measured CPU-lane ms for 1..8 local experts (0 = off)
                        dtype=torch.float32, device=dev)


def wrap(model):
    """Before expert_cache.attach: wrap every MoE router (so the cache sees the CPU picks removed) and the combine hook."""
    preflight()
    _build()
    import expert_cache
    mods = expert_cache.find_moe(model.modules)
    assert mods, "no MoE modules"
    dev = mods[0].device
    assert all(m.device == dev for m in mods), "CPU tier: single GPU only"
    S["dev"] = torch.device(dev)
    H = mods[0].hidden_size
    # GLM53_CT_VALIDATE=1: every forward up to GLM53_CT_MAXBSZ (default 4096) tokens goes through the tier with ALL
    # cache misses on the CPU (policy force), so the teacher-forced panel measures the CPU kernel's numerics
    validate = os.environ.get("GLM53_CT_VALIDATE") == "1"
    maxbsz = int(os.environ.get("GLM53_CT_MAXBSZ", "4096" if validate else "4"))
    if validate:
        os.environ.setdefault("GLM53_CT_FORCE_N", "100000"); os.environ.setdefault("GLM53_CT_TIMEOUT_S", "120")
    S.update(maxbsz=maxbsz, H=H, validate=validate, maxpicks=max(64, 8 * maxbsz))
    S["ctrl"] = torch.zeros(16, dtype=torch.int64).pin_memory()
    S["hx"] = torch.zeros(max(8, maxbsz) * H, dtype=torch.half).pin_memory()
    S["picks"] = torch.zeros(S["maxpicks"] * 3, dtype=torch.int32).pin_memory()
    S["hout"] = torch.zeros(max(8, maxbsz) * H, dtype=torch.float32).pin_memory()
    S["dflag"] = torch.zeros(1, dtype=torch.int64, device=dev)
    S["stats"] = torch.zeros(8, dtype=torch.int64, device=dev)
    sp = os.environ.get("GLM53_CT_STATS") or os.environ.get("GLM53_EC_WARM") or os.environ.get("GLM53_ZC_STATS")
    scores = json.load(open(sp)) if sp else {}
    S["pol"] = _policy(dev)
    S["timeout_ns"] = int(float(os.environ.get("GLM53_CT_TIMEOUT_S", "2")) * 1e9)
    for m in mods:
        E = m.num_local_experts or m.num_experts
        first = int(m.routing_first or 0)
        sc = scores.get(m.key)
        m._ct_score = torch.tensor(sc[first:first + E] if sc else [0.0] * E, dtype=torch.float32, device=dev)
        m._ct_sel = torch.empty(S["maxpicks"], dtype=torch.long, device=dev)
        m._ct_rres = torch.zeros(E, dtype=torch.int32, device=dev)   # [E4] 1 = held by the 3090 tier (_rt_enable fills it)
        m._ct_w = torch.empty(S["maxpicks"], dtype=torch.half, device=dev)
        m._ct_pending = False
        orig = m.routing_fn

        def routed(bsz, cfg, z, params, _orig=orig, _m=m):
            sel, w = _orig(bsz, cfg, z, params)
            _m._ct_pending = False
            if (not S["started"] or bsz > S["maxbsz"] or params.get("autosplit_measure") or params.get("tp_warmup")
                    or getattr(_m, "_ec", None) is None or sel.numel() > S["maxpicks"]):
                return sel, w
            pool, li = _m._ec
            S["seq"] += 1
            n = sel.numel()
            so, wo = _m._ct_sel[:n], _m._ct_w[:n]
            zz = z.view(bsz, -1)
            if zz.dtype != torch.half or not zz.is_contiguous():
                zz = zz.half().contiguous()
            _m._ct_orig = sel                     # [E3] expert_cache's background admission reads the CPU lane from it
            _CU.ft_split(sel.contiguous(), w.contiguous(), zz, pool.E, pool.first, pool.slotof[li], _m._ct_score, S["pol"],
                         S["ctrl"].data_ptr(), S["hx"].data_ptr(), S["picks"].data_ptr(), S["seq"], li, int(bsz > 8), so, wo, S["dflag"], S["stats"], _m._ct_rres)
            _m._ct_pending = True
            if TR["path"] and bsz == 1:                  # [E5 prep] + hit flags (the local GPU cache) per original pick
                hit = (pool.slotof[li].index_select(0, (sel.view(-1) - pool.first).clamp(0, pool.E - 1)) >= 0).int()
                TR["buf"].append([li] + sel.view(-1).tolist() + so.view(-1).tolist() + hit.tolist())
                if len(TR["buf"]) >= 4200:
                    _trace_flush()
            return so.view(sel.shape), wo.view(w.shape)
        m.routing_fn = routed

        orig_comb = m.cpu_split_combine

        def comb(fhs, cpu_partial, cpu_pending, shape, _orig=orig_comb, _m=m):
            if _m._ct_pending:
                _m._ct_pending = False
                f = fhs if fhs.is_contiguous() else fhs.contiguous()
                _CU.ft_combine(f, S["dflag"], S["ctrl"].data_ptr(), S["hout"].data_ptr(), S["stats"], S["timeout_ns"])
                fhs = f
            return _orig(fhs, cpu_partial, cpu_pending, shape)
        m.cpu_split_combine = comb
    S["mods"] = mods
    _prelaunch(dev, H, mods[0])
    print(f" -- cpu_tier: wrapped {len(mods)} MoE layers on {dev} (max bsz {maxbsz}, scores {sp})", flush=True)


def _prelaunch(dev, H, m):
    """Launch both kernels once BEFORE expert_cache sizes its arena, so any per-kernel launch resources (local-memory
    reservation, module load) are taken while VRAM is still free (otherwise: launch OOM once the elastic arena
    holds all free VRAM)."""
    E = m.num_local_experts or m.num_experts
    sel = torch.zeros(1, 8, dtype=torch.long, device=dev); w = torch.zeros(1, 8, dtype=torch.half, device=dev)
    z = torch.zeros(1, H, dtype=torch.half, device=dev)
    slot = torch.full((E,), -1, dtype=torch.int32, device=dev)
    pol = torch.tensor([0.4, 0.023, 0.11, 0.104, 0.2, 0, 0, 0, 0, 0, 0, 0] + [0.0] * 8, dtype=torch.float32, device=dev)   # force 0, no publish
    st = torch.zeros(8, dtype=torch.int64, device=dev)
    _CU.ft_split(sel, w, z, E, int(m.routing_first or 0), slot, m._ct_score, pol, S["ctrl"].data_ptr(), S["hx"].data_ptr(),
                 S["picks"].data_ptr(), 0, 0, 0, m._ct_sel[:8], m._ct_w[:8], S["dflag"], st, m._ct_rres)
    out = torch.zeros(1, H, dtype=torch.float32, device=dev)
    _CU.ft_combine(out, S["dflag"], S["ctrl"].data_ptr(), S["hout"].data_ptr(), st, S["timeout_ns"])
    torch.cuda.synchronize(dev)


def start():
    """After expert_cache.attach: register the native pinned home copies with the host kernel, self-test, start."""
    preflight()
    mods = S["mods"]
    m0 = mods[0]
    H, I = m0.hidden_size, m0.intermediate_size
    cpus = _cpus()
    threads = int(os.environ.get("GLM53_CT_THREADS", "0")) or len(cpus)
    assert 1 <= threads <= len(cpus), f"GLM53_CT_THREADS={threads} needs at least that many CPUs in GLM53_CT_CPUS ({cpus})"
    mode = int(os.environ.get("GLM53_CT_MODE", "-1"))
    swz = int(os.environ.get("GLM53_CT_SWZ", "1"))
    _H.tier_init(threads, cpus, S["ctrl"], S["hx"], S["picks"], S["hout"], mode, H, I, len(mods), swz)
    import exl3_tiers
    arenas = [(a.data_ptr(), a.data_ptr() + a.numel()) for a in exl3_tiers._ARENAS]
    t0 = time.time()
    for m in mods:
        pool, li = m._ec
        ptrs = pool.home_cpu[li].contiguous()
        lo, hi = int(ptrs.min()), int(ptrs.max())
        assert any(a <= lo and hi < b for a, b in arenas) or all(any(a <= int(p) < b for a, b in arenas) for p in ptrs.flatten().tolist()), \
            f"{m.key}: home pointers outside the pinned arenas"
        sc = lambda lins, attr: torch.stack([getattr(l.inner, attr).view(-1) for l in lins]).half().cpu().contiguous()
        g, u, d = m.multi_gate.linears, m.multi_up.linears, m.multi_down.linears
        _H.tier_add_layer(li, ptrs, sc(g, "suh"), sc(g, "svh"), sc(u, "suh"), sc(u, "svh"), sc(d, "suh"), sc(d, "svh"))
    print(f" -- cpu_tier: registered {len(mods)} layers in {time.time() - t0:.1f} s (CPU block-contiguous copy "
          f"{_H.tier_stats()[5] / 1e9:.1f} GB); threads {threads} on cpus {cpus[:threads]}", flush=True)
    if os.environ.get("GLM53_CT_SELFTEST", "1") == "1":
        S["selftest"] = selftest()
        print(f" -- cpu_tier selftest: {S['selftest']}", flush=True)
    lim = float(getattr(m0, "act_limit", 0.0) or 0.0)
    if os.environ.get("GLM53_CT_SWIGLU_CLAMP") == "1" and lim:     # [rt T-27] match the GPU path's clamp
        _H.tier_set_act_limit(lim); S["act_limit"] = lim
        print(f" -- cpu_tier: SwiGLU clamp ON (limit {lim}, exllamav3 semantics)", flush=True)
    if os.environ.get("GLM53_RT"):
        _rt_enable(mods)
    _H.tier_start(cpus[0])
    if S.get("rt"):
        _rt_auto_b_start()
    S["started"] = True


def _rt_plan(host, port, timeout=10.0):
    """[rt] Ask the expert server which experts it holds: CTRL request -> {"layers": {model_layer: [experts]}, ...}."""
    import socket, struct
    with socket.create_connection((host, port), timeout=timeout) as s:
        s.sendall(b"GRTC" + bytes(20))
        hdr = b""
        while len(hdr) < 8:
            c = s.recv(8 - len(hdr)); assert c, "expert server closed"; hdr += c
        assert hdr[:4] == b"GRTC", hdr
        n = struct.unpack("<I", hdr[4:])[0]; body = b""
        while len(body) < n:
            c = s.recv(n - len(body)); assert c, "expert server closed"; body += c
    return json.loads(body)


def _rt_enable(mods):
    """[rt] Fetch the plan and hand per-layer residency bitmaps to the host tier. No server = tier stays local-only."""
    host, port = os.environ["GLM53_RT"].rsplit(":", 1)
    try:
        plan = _rt_plan(host, int(port))
    except Exception as ex:
        # GLM53_RT_PLAN: the plan file the (leased, not yet running) server will load — enable anyway; the client
        # reconnects every 2 s and the server refuses any pick it does not hold (computed here instead)
        pf = os.environ.get("GLM53_RT_PLAN")
        if not pf:
            print(f" -- rt: expert server {host}:{port} unreachable at start ({ex!r}) and no GLM53_RT_PLAN; remote tier OFF", flush=True)
            return
        plan = json.load(open(pf)); plan.setdefault("plan_name", os.path.basename(pf)); plan.setdefault("backend", "pending")
        plan["layers"] = {k: (v["experts"] if isinstance(v, dict) else v) for k, v in plan["layers"].items()}
        print(f" -- rt: expert server {host}:{port} not up yet ({ex!r}); using plan file {pf}", flush=True)
    res, ml, n = [], [], 0
    for m in mods:
        E = m.num_local_experts or m.num_experts
        first = int(m.routing_first or 0)
        N = int(m.key.split(".layers.")[1].split(".")[0])
        b = torch.zeros(E, dtype=torch.uint8)
        for e in plan["layers"].get(str(N), []):
            if first <= e < first + E:
                b[e - first] = 1; n += 1
        res.append(b); ml.append(N)
        m._rt_res = b                             # [E3b] expert_cache skips these as background candidates while attached
        m._ct_rres.copy_(b.to(torch.int32))       # [E4] ft_split's three-target cost model
    _H.tier_remote(host, int(port), res, ml, int(os.environ.get("GLM53_RT_TIMEOUT_US", "20000")))
    try:
        import expert_cache
        expert_cache.set_rt(None)
    except Exception as ex:
        print(f" -- rt: expert_cache skip mask not set ({ex!r})", flush=True)
    S["rt"] = {"server": f"{host}:{port}", "experts": n, "backend": plan.get("backend"), "plan": plan.get("plan_name")}
    S["rt_on"] = True
    print(f" -- rt: remote expert tier ON: {host}:{port} holds {n} experts ({plan.get('backend')}, {plan.get('plan_name')})", flush=True)


@torch.inference_mode()
def selftest(n_layers=3, n_exp=4, mtok=(1, 3)):
    """ft_mul1 vs fp64 truth from exllamav3's own dequantised weights (LinearEXL3.get_weight_tensor = GPU decode +
    Hadamard + suh/svh), computed on the CPU: rel RMS of the weighted routed sum per mode.
    """
    out = {}
    mods = S["mods"]
    g = torch.Generator(device="cpu").manual_seed(0)
    cases = []
    for li in list(range(0, len(mods), max(1, len(mods) // n_layers)))[:n_layers]:
        m = mods[li]
        E = m.num_local_experts or m.num_experts
        for mt in mtok:
            x = (torch.randn(mt, m.hidden_size, generator=g) * 0.35).half()
            exps = torch.randperm(E, generator=g)[:n_exp].tolist()
            ref = torch.zeros(mt, m.hidden_size, dtype=torch.float64)
            xd = x.double()
            for e in exps:
                Wg = m.gates[e].inner.get_weight_tensor().cpu().double(); Wu = m.ups[e].inner.get_weight_tensor().cpu().double()
                a = (torch.nn.functional.silu(xd @ Wg) * (xd @ Wu)).half().double()
                del Wg, Wu
                Wd = m.downs[e].inner.get_weight_tensor().cpu().double()
                ref += 0.25 * (a @ Wd)
                del Wd
            cases.append((li, x, exps, ref))
    torch.cuda.empty_cache()
    for mode in (1, 0, 2):
        errs = []
        for li, x, exps, ref in cases:
            mt = x.shape[0]
            sel = torch.tensor([exps] * mt, dtype=torch.int32)
            w = torch.full((mt, len(exps)), 0.25)
            cpu = _H.tier_forward(li, x.float(), sel, w, mode).double()
            errs.append(float((cpu - ref).norm() / ref.norm()))
        out[{1: "EXACT", 0: "AFFINE", 2: "I16"}[mode]] = round(max(errs), 6)
    return out


def summary():
    _trace_flush()   # [bdw-opt trace] /stats also flushes pending trace rows
    if not S["mods"]:
        return None
    st = S["stats"].tolist()
    h = _H.tier_stats() if _H else []
    calls = max(1, st[3])
    return {"calls": st[3], "unique_hits": st[0], "unique_misses": st[1], "cpu_experts": st[2],
            "cpu_experts_per_call": round(st[2] / calls, 3), "misses_per_call": round(st[1] / calls, 3),
            "hit_rate_unique": round(st[0] / max(1, st[0] + st[1]), 4),
            "gpu_wait_ms_per_job": round(st[4] / 1e6 / max(1, st[5]), 4), "gpu_waits": st[5], "timeouts": st[6],
            "host_jobs": h[0] if h else None, "host_busy_ms_per_job": round(h[4] / 1e6 / max(1, h[0]), 4) if h else None,
            "host_empty_jobs": h[3] if h else None, "cpu_copy_gb": round(h[5] / 1e9, 1) if h else None, "selftest": S["selftest"],
            "policy": S["pol"].tolist(),
            # [bdw-opt] per-job wall ms: prep (fp16->fp32 + route), then moe_forward steps (each incl. its pool barrier).
            # stock: [prep, p0 in, p1 gate+up+act, p2 down, p3 sum, 0]; split1: [prep, p0 in, gate|up, act, down, sum]
            "split1": os.environ.get("GLM53_CT_SPLIT1") == "1",
            "phase_ms_per_job": ([round(h[11] / 1e6 / max(1, h[0]), 4)] + [round(v / 1e6 / max(1, h[0]), 4) for v in h[6:11]])
                                if len(h) > 11 else None,
            "remote": _rt_summary(),
            # [rt] combines where the CPU partial was already done when the GPU arrived (GPU path was the slower runner)
            "gpu_late": st[7], "gpu_late_frac": round(st[7] / max(1, st[5]), 4),
            "job_hist": _job_hist()}


def _job_hist():
    """[rt] per local-pick-count bucket: jobs, mean job ms, mean local-compute ms, mean remote-wait ms."""
    if not _H or not hasattr(_H, "tier_job_hist"):
        return None
    h = _H.tier_job_hist(); out = {}
    for b in range(17):
        n, busy, loc, wait = h[4 * b:4 * b + 4]
        if n:
            out[str(b) if b < 16 else "16+"] = [n, round(busy / n / 1e6, 4), round(loc / n / 1e6, 4), round(wait / n / 1e6, 4)]
    return out


_POL = {"tzc": 0, "thit": 1, "a": 2, "b": 3, "tok": 4, "maxn": 5, "force_n": 6, "ra": 9, "rb": 10}


def set_policy(rt=None, check=None, b_rt=None, auto_b=None, e4=None, fuse=None, fuse_maxn=None, pf_lean=None, lut=None,
               ovl_after=None, **kw):
    """[rt] runtime knobs (T-18 verification, no restart): ft_split cost model values in S["pol"] (read by the GPU split
    kernel every call) and the remote tier on/off + numerics-check period. Returns the current values."""
    if b_rt is not None:
        S["b_rt"] = float(b_rt)                              # B used while the 3090s are attached (auto mode)
    if auto_b is not None:
        S["b_auto"] = bool(auto_b)
    if e4 is not None:
        S["e4"] = bool(e4)                                   # [E4] three-target split while attached (auto thread)
    if lut is not None:                                      # [bdw-opt] "" or "0" turns the table off
        S["pol"][12:20] = torch.tensor(_lut("" if lut in ("", "0") else lut), dtype=torch.float32)
    try:
        import k_overlap
        after = k_overlap.set_after(None if ovl_after is None else bool(ovl_after))   # [bdw-opt]
    except Exception:
        after = None
    e6 = _H.tier_set_e6(-1 if fuse is None else int(fuse), -1 if fuse_maxn is None else int(fuse_maxn),
                        -1 if pf_lean is None else int(pf_lean))   # [E6] CPU-job schedule switches
    if kw.get("b") is not None and S.get("b_auto"):
        S["b_auto"] = False                                  # a manual B overrides the auto-switch (auto_b=1 re-enables)
    for k, v in kw.items():
        if v is not None:
            S["pol"][_POL[k]] = float(v)
    if (rt is not None or check is not None) and S.get("rt"):
        _H.tier_remote_set(bool(S["rt_on"] if rt is None else rt), -1 if check is None else int(check))
        if rt is not None:
            S["rt_on"] = bool(rt)
    p = S["pol"].tolist()
    return {"policy": {k: round(p[i], 4) for k, i in _POL.items()}, "e4": S.get("e4"), "e4_active": p[8] > 0.5,
            "e6": {"fuse": e6[0], "fuse_maxn": e6[1], "pf_lean": e6[2]}, "lut": [round(x, 3) for x in p[12:20]],
            "ovl_after": after,
            "remote_on": S.get("rt_on"), "auto_b": S.get("b_auto"),
            "b_rt": S.get("b_rt"), "b_base": S.get("b_base"), "remote": _rt_summary()}


def _rt_sync_masks():
    """[E5] copy the C++ client's live residency (GRT2 deltas) into the device masks of the E4 three-target split
    (m._ct_rres) and the E3b background-admission skip (expert_cache bgskip, rebuilt from m._rt_res). Called from the
    0.5 s auto-B thread; only rows that changed are written. Lag is safe: a retiring expert is still served, a new one
    only waits."""
    mods = S["mods"]
    res = _H.tier_remote_res(int(mods[0]._ct_rres.numel()))
    last = S.get("rt_res_last")
    if last is not None and torch.equal(res, last):
        return
    changed = 0
    for li, m in enumerate(mods):
        if last is None or not torch.equal(res[li], last[li]):
            m._rt_res = res[li].clone()
            m._ct_rres.copy_(res[li].to(torch.int32))
            changed += 1
    S["rt_res_last"] = res
    S["rt_resident"] = int(res.sum())
    S["rt_mask_syncs"] = S.get("rt_mask_syncs", 0) + 1
    try:
        import expert_cache
        expert_cache.set_rt(None)
    except Exception as ex:
        print(f" -- rt: bgskip sync failed ({ex!r})", flush=True)


def _rt_auto_b_start():
    """[rt E2] ft_split's CPU price B: GLM53_RT_B (0.15) while the expert server is connected and enabled — most CPU-tier
    picks then go to the 3090s, so a CPU-tier expert is cheaper — and GLM53_CT_B (0.31) otherwise. Sweep 2026-10-11:
    B 0.12-0.20 was ~+7% over 0.31 with the 3090s attached; lower starves the local GPU cache. Writes only on a state change."""
    import threading
    S["b_base"] = float(os.environ.get("GLM53_CT_B", "0.104"))
    S["b_rt"] = float(os.environ.get("GLM53_RT_B", "0.15"))
    S["b_auto"] = os.environ.get("GLM53_RT_AUTO_B", "1") == "1"
    S["e4"] = os.environ.get("GLM53_CT_E4", "0") == "1"

    def loop():
        cur = None
        cur4 = None
        while True:
            try:
                r = _H.tier_remote_stats()
                want = S["b_rt"] if (r and r[0] and S.get("rt_on", True)) else S["b_base"]
                try:
                    import expert_cache
                    expert_cache.set_rt(bool(r and r[0] and S.get("rt_on", True)))   # [E3b] applied by step()
                except Exception:
                    pass
                att = bool(r and r[0] and S.get("rt_on", True))
                if att and len(r) > 19 and r[14] == 2:
                    _rt_sync_masks()                             # [E5] dynamic server: residency moves at runtime
                w4 = 1.0 if (S["e4"] and att) else 0.0
                if w4 != cur4:
                    S["pol"][8] = w4; cur4 = w4                  # [E4] three-target split only while the 3090s serve
                if not S["b_auto"]:
                    cur = None                                   # re-applies when auto mode is re-enabled
                elif want != cur:
                    S["pol"][3] = want; cur = want
                    print(f" -- rt: remote {'attached' if want == S['b_rt'] else 'detached'} -> ft_split B = {want}", flush=True)
            except Exception as ex:
                print(f" -- rt: auto-B error {ex!r}", flush=True)
            time.sleep(0.5)
    threading.Thread(target=loop, daemon=True, name="rt-auto-b").start()


def _rt_summary():
    """[rt] remote expert tier counters (None when GLM53_RT is off)."""
    if not S.get("rt") or not _H:
        return None
    r = _H.tier_remote_stats()
    c = max(1, r[1])
    return dict(S["rt"], connected=bool(r[0]), calls=r[1], picks=r[2], picks_per_call=round(r[2] / c, 3), fallbacks=r[3],
                fallback_picks=r[4], connects=r[5], errors=r[6], timeouts=r[7], rtt_ms_mean=round(r[8] / 1e6 / c, 4),
                wait_ms_mean=round(r[9] / 1e6 / c, 4), rtt_ms_max=round(r[10] / 1e6, 3),
                check_calls=r[11], check_rel_rms=(r[12] / 1e9 if r[12] >= 0 else None),
                refused=r[13] if len(r) > 13 else None,
                **(dict(protocol=r[14], deltas=r[15], full_maps=r[16], partial_refusals=r[17], hints_sent=r[18],
                        client_gen=r[19], resident=S.get("rt_resident"), mask_syncs=S.get("rt_mask_syncs", 0))
                   if len(r) > 19 else {}))


def reset_stats():
    S["stats"].zero_()
