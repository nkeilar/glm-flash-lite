"""Zero-copy expert tier for stock exllamav3 1.5.1 (monkeypatch; no source or kernel edits).

Three tiers per MoE layer, by a per-expert ranking (e.g. routing frequency):
  rank [0, V)          VRAM-resident (normal exllamav3 load)
  rank [V, first)      zero-copy: the expert's .trellis lives in pinned host memory; exllamav3's own MoE kernels read it
                       through their per-expert pointer tables over PCIe (ext.pinned_cuda_view = UVA alias, the same
                       mechanism exllamav3 uses for pinned vision towers). suh/svh (tiny) stay in VRAM.
  rank [first, 288)    CPU worker (stock -mcs tail; optional), placement from EXL3_MOE_CPU_SPLIT_STATS
Autosplit accounting uses torch.cuda.memory_allocated, which excludes pinned host memory, so zero-copy experts do not
consume the per-device VRAM budget.

Env:
  GLM53_ZC_VRAM   int V: experts per layer kept in VRAM (the rest of the GPU slice becomes zero-copy);
                  or a JSON file {layer_key: V}
  GLM53_ZC_STATS  JSON {layer_key: [score x E]} ranking (default: EXL3_MOE_CPU_SPLIT_STATS); required
One pinned arena per layer (exact size; torch's per-tensor pinned allocations round up to powers of two).
"""
import json, os, re
import torch

_ACTIVE = {"key": None}
_ARENAS = []          # keep pinned stores alive
_STATS = {"zc_experts": 0, "zc_bytes": 0, "layers": 0}


def _load_json(p):
    with open(p) as f:
        return json.load(f)


_THP_KEEP = []   # [bdw-opt zc-thp] mmap objects backing THP arenas (must outlive their tensors)


_NODE1_ARENAS = [int(os.environ.get("GLM53_ZC_NODE1_ARENAS", "0") or 0)]


def _thp_host_tensor(nbytes):
    """[bdw-opt zc-thp] 2 MB-aligned anonymous mapping with madvise(MADV_HUGEPAGE) BEFORE first touch, so the faults
    cudaHostRegister triggers allocate transparent huge pages (host THP mode 'madvise'). Process-local; same bytes.
    Microbenchmark on the test box (2x Xeon E5-2699 v4, RTX 3090, PCIe 3.0): GPU zero-copy reads from 4 KB-backed registered memory drop to
    2.6-5.7 GB/s at modest kernel parallelism vs ~9.6-9.9 GB/s THP-backed (PCIe 3.0 x16)."""
    import ctypes, mmap
    al = 2 << 20
    mm = mmap.mmap(-1, nbytes + al, flags=mmap.MAP_PRIVATE | mmap.MAP_ANONYMOUS)
    base = ctypes.addressof(ctypes.c_char.from_buffer(mm))
    off = ((base + al - 1) & ~(al - 1)) - base
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    libc.madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    rc = libc.madvise(base + off, nbytes, 14)   # MADV_HUGEPAGE
    if rc != 0:
        print(f" -- zc-thp: madvise failed errno={ctypes.get_errno()} (falling back to 4 KB pages)", flush=True)
    if _NODE1_ARENAS[0] > 0:
        # [T-61 E1] GLM53_ZC_NODE1_ARENAS=N: the first N layer arenas prefer node 1. The GPU (and so this whole arena)
        # sits on node 0, which E1's half of the LUT bank also needs; GPU reads of these N layers cross the socket link.
        _NODE1_ARENAS[0] -= 1
        mask = (ctypes.c_ulong * 2)(1 << 1, 0)
        rc = libc.syscall(237, ctypes.c_void_p(base + off), ctypes.c_size_t(nbytes), 1, mask, ctypes.c_ulong(65), 0)  # mbind MPOL_PREFERRED
        if rc != 0:
            print(f" -- zc-thp: mbind node 1 failed errno={ctypes.get_errno()} (default placement)", flush=True)
    ctypes.memset(base + off, 0, nbytes)   # first touch here (the verified bench path), not inside cudaHostRegister
    t = torch.frombuffer(mm, dtype=torch.uint8, count=nbytes, offset=off)
    _THP_KEEP.append(mm)
    return t


def pinned_exact(nbytes, dev_idx):
    """Exact-size pinned host buffer (torch's pinned allocator rounds each block up to a power of two: a 2.72 GB layer
    arena would cost 4 GB). Plain host tensor + cudaHostRegister under the GPU's context (pinned_cuda_view checks the
    pointer's device). [bdw-opt] GLM53_ZC_THP=1: THP-backed host memory instead of torch.empty (see _thp_host_tensor)."""
    if os.environ.get("GLM53_ZC_THP") == "1":
        t = _thp_host_tensor(nbytes)
    else:
        t = torch.empty(nbytes, dtype=torch.uint8, device="cpu")
    with torch.cuda.device(dev_idx):
        r = torch.cuda.cudart().cudaHostRegister(t.data_ptr(), nbytes, 2)   # cudaHostRegisterMapped
    assert int(r) == 0, f"cudaHostRegister failed: {r}"
    assert t.is_pinned()
    return t


def install():
    from exllamav3.ext import exllamav3_ext as ext
    from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
    from exllamav3.modules.linear import Linear
    from exllamav3.loader.safetensors import SafetensorsCollection

    vr = os.environ["GLM53_ZC_VRAM"]
    per_layer_v = _load_json(vr) if not vr.lstrip("-").isdigit() else None
    v_default = int(vr) if per_layer_v is None else None
    stats_path = os.environ.get("GLM53_ZC_STATS") or os.environ.get("EXL3_MOE_CPU_SPLIT_STATS")
    assert stats_path, "GLM53_ZC_STATS (or EXL3_MOE_CPU_SPLIT_STATS) required"
    stats = _load_json(stats_path)

    orig_split = BlockSparseMLP.cpu_maybe_split_load

    def split_then_mark(self, device, **kwargs):
        orig_split(self, device, **kwargs)
        if device is None or torch.device(device).type != "cuda" or self.key not in stats:
            return
        score = stats[self.key]
        order = sorted(range(len(score)), key=lambda e: -score[e])
        rank = {e: r for r, e in enumerate(order)}
        V = per_layer_v.get(self.key, 0) if per_layer_v is not None else v_default
        lins = []
        for group in ([self.gates] if self.gated else []) + [self.ups, self.downs]:
            for l in group:
                e = int(re.search(r"experts\.(\d+)\.", l.key).group(1))
                if rank[e] >= V:
                    lins.append(l)
        if not lins:
            return
        stc = self.config.stc
        sizes = [stc.get_tensor_size(l.key + ".trellis") for l in lins]
        total = sum(sizes)
        dev_idx = torch.device(device).index or 0
        arena = pinned_exact(total, dev_idx)
        _ARENAS.append(arena)
        off = 0
        for l, sz in zip(lins, sizes):
            l._zc_slot = (arena, off, sz, dev_idx)
            off += sz
        n_exp = len({re.search(r"experts\.(\d+)\.", l.key).group(1) for l in lins})
        _STATS["zc_experts"] += n_exp; _STATS["zc_bytes"] += total; _STATS["layers"] += 1
        print(f" -- zero-copy tier: {self.key} {n_exp} experts ({total / 1e9:.2f} GB pinned) V={V}", flush=True)

    BlockSparseMLP.cpu_maybe_split_load = split_then_mark

    orig_lin_load = Linear.load

    def lin_load(self, device, **kwargs):
        if getattr(self, "_zc_slot", None) is None:
            return orig_lin_load(self, device, **kwargs)
        _ACTIVE["key"] = self.key + ".trellis"
        _ACTIVE["lin"] = self
        try:
            return orig_lin_load(self, device, **kwargs)
        finally:
            _ACTIVE["key"] = None; _ACTIVE["lin"] = None

    Linear.load = lin_load

    orig_get = SafetensorsCollection.get_tensor

    def get_tensor(self, key, device=None, *args, **kwargs):
        if key != _ACTIVE["key"]:
            return orig_get(self, key, device, *args, **kwargs)
        arena, off, sz, dev_idx = _ACTIVE["lin"]._zc_slot
        kwargs.pop("no_defer", None)
        t = orig_get(self, key, "cpu", *args, no_defer=True, **kwargs)
        assert t.numel() * t.element_size() == sz, (key, t.shape, sz)
        p = arena[off: off + sz].view(t.dtype).view(t.shape)
        p.copy_(t)
        return ext.pinned_cuda_view(p, dev_idx)

    SafetensorsCollection.get_tensor = get_tensor
    print(f" -- exl3_tiers installed: V={vr} stats={stats_path}", flush=True)


def summary():
    return dict(_STATS)
