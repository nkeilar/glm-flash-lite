"""Faster CPU-tier handoff kernel (kernels/k110/ft_split_fast.cu), bit-identical outputs to the stock
ft_split (kernels/cpu_avx2/ft_tier_cu.cu): 3000 random decode cases, 0 mismatches. Stock 24 / 39 / 73 us at bsz
1 / 4 / 8 with a job -> 8.6 / 11.5 / 19 us (the kernel sits on the critical path before the CPU job can start).
Env: GLM53_K_FTSPLIT=1 (serve.py, after cpu_tier.start): cpu_tier._CU.ft_split -> fast kernel for <= 64 picks.
No change to cpu_tier.py: its module-level _CU handle is swapped for a namespace with the fast ft_split.
LUT format: the guarded precompiled module (FT_ENABLE_LUT) from the reviewed receipt, never a JIT build.
"""
import os, types
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(os.path.dirname(_HERE), "kernels", "k110", "ft_split_fast.cu")
STATS = {"fast": 0, "stock": 0}


def _load_lut(settings):
    """LUT serving never JIT-builds: import the guarded (FT_ENABLE_LUT) fast split module that preflight
    admitted from the reviewed precompile receipt, with the same identity checks as cpu_tier._build_lut."""
    import importlib.util, sys
    from lut_manifest import SPLIT_MODULE, identity, no_symlinks, require
    require(settings.precompiled is not None and SPLIT_MODULE in settings.precompiled["modules"],
            "GLM53_K_FTSPLIT with LUT requires the guarded fast split module in the reviewed precompile")
    entry = settings.precompiled["modules"][SPLIT_MODULE]
    path = no_symlinks(entry["path"])
    require(identity(path.stat(follow_symlinks=False)) == entry["identity"], "precompiled fast split module changed before import")
    existing = sys.modules.get("ft_split_fast")
    require(existing is None or getattr(existing, "__file__", None) == str(path), "another fast split module is already loaded")
    spec = importlib.util.spec_from_file_location("ft_split_fast", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    require(no_symlinks(module.__file__) == path and identity(path.stat(follow_symlinks=False)) == entry["identity"],
            "loaded fast split module identity differs from admission")
    return module


def install():
    import cpu_tier
    settings = cpu_tier.S.get("settings")
    if settings is not None and settings.format == "lut":
        fast = _load_lut(settings)
    else:
        from torch.utils.cpp_extension import load
        d = os.environ.get("GLM53_K110_BUILD", os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "build", "k110f"))
        os.makedirs(d, exist_ok=True)
        fast = load(name="ft_split_fast", sources=[_SRC], build_directory=d, extra_cuda_cflags=["-O3"], verbose=False)
    stock = cpu_tier._CU

    def ft_split(sel, *a):
        if sel.numel() <= 64:
            STATS["fast"] += 1
            return fast.ft_split(sel, *a)
        STATS["stock"] += 1
        return stock.ft_split(sel, *a)
    cpu_tier._CU = types.SimpleNamespace(ft_split=ft_split, ft_combine=stock.ft_combine)
    print(" -- k_ftsplit (K110): CPU-tier ft_split -> fast kernel (<= 64 picks)", flush=True)
