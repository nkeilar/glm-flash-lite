#!/usr/bin/env python3
"""Compile serving extensions without creating a CUDA context.

Native sources and resulting modules are SHA-bound to the serving checkout.
No model is imported, installed packages are never changed, and every build
uses a fresh directory. Run a new process for each build variant.
"""
import argparse
import ast
import re
import contextlib
import hashlib
import inspect
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import stat
import sys
import time
from unittest import mock

sys.dont_write_bytecode = True
from gguf import check_no_symlinks
from quantize_s1 import atomic_json

REPOSITORY = Path(__file__).absolute().parents[2]
SPLIT_SOURCE = 'kernels/k110/ft_split_fast.cu'
# Tuned for the Broadwell test box. -mtune changes speed only, never results (verified bit-exact); set it to your
# CPU. LTO + -fno-semantic-interposition remove the per-row PLT dispatch. Never add -march=native, -DNDEBUG or
# -ffp-contract: GCC would contract the Q8 activation quantiser into FMAs and change results.
FLAGS = ['-O3', '-mavx2', '-mfma', '-mf16c', '-mtune=broadwell', '-fno-semantic-interposition', '-flto', '-std=c++17',
         '-DFT_OPT_BSUM_SHUF=1', '-DFT_OPT_FLOAT_EXTRACT=1',
         '-DFT_OPT_EXP_EXTRACT=1', '-DFT_OPT_BSUM_MADD_N=32', '-DFT_OPT_CTRL_REUSE=1']
LDFLAGS = ['-lpthread', '-flto=auto', '-O3', '-mavx2', '-mfma', '-mf16c', '-mtune=broadwell', '-fno-semantic-interposition']


def digest(path):
    check_no_symlinks(path)
    with Path(path).open('rb') as stream:
        before = os.fstat(stream.fileno())
        if not stat.S_ISREG(before.st_mode):
            raise ValueError('build inputs and outputs must be regular files')
        value = hashlib.sha256()
        while data := stream.read(4 * 2**20):
            value.update(data)
        after = os.fstat(stream.fileno())
    fields = ('st_dev', 'st_ino', 'st_size', 'st_mtime_ns', 'st_ctime_ns')
    if any(getattr(before, field) != getattr(after, field) for field in fields):
        raise ValueError('build input or output changed during hashing')
    return value.hexdigest()


def torch_versions(torch_module):
    """Record distinct distribution metadata and exact imported runtime versions."""
    runtime = str(torch_module.__version__)
    metadata = importlib.metadata.version('torch')
    if metadata != runtime.split('+', 1)[0]:
        raise ValueError('Torch metadata differs from the imported runtime public version')
    return dict(torch_version=runtime, torch_runtime_version=runtime,
                torch_metadata_version=metadata)


def plan(source, build_root, variant, revision, *, cuda_home=None, cuda_arch='8.6', jobs=2):
    source, build_root = Path(source).absolute(), Path(build_root).absolute()
    for path in (source, build_root):
        if '..' in path.parts:
            raise ValueError('source and build paths must not traverse parents')
        check_no_symlinks(path)
    if source == build_root or source.is_relative_to(build_root):
        raise ValueError('use a separate build directory')
    if build_root.is_relative_to(source) and build_root.relative_to(source).parts[0] in ('kernels', 'glm53'):
        raise ValueError('build directory must not modify native or Python source directories')
    if not isinstance(revision, str) or not re.fullmatch('[a-f0-9]{40}', revision):
        raise ValueError('pin the serving source full revision')
    if not re.fullmatch(r'[1-9][0-9]?\.[0-9](\+PTX)?', cuda_arch):
        raise ValueError('select one explicit CUDA architecture, for example 8.6')
    if type(jobs) is not int or not 1 <= jobs <= 128:
        raise ValueError('compiler jobs must be 1..128')
    if variant not in ('lut', 'capture'):
        raise ValueError('unknown build variant')
    head = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if head != revision or len(head) != 40:
        raise ValueError('serving source revision mismatch')
    if subprocess.check_output(['git', '-C', str(source), 'status', '--porcelain', '--untracked-files=no'], text=True):
        raise ValueError('serving source must have no tracked changes')
    kernel = source / 'kernels/cpu_avx2'
    directory = build_root / variant
    check_no_symlinks(directory)
    if directory.exists():
        raise ValueError('variant build directory must be new; use a fresh build root after any failure')
    sources = [kernel / 'ft_tier_ext.cpp']
    flags = FLAGS + ['-I' + str(kernel)]
    if variant == 'lut':
        sources += [kernel / 'ft_lut.cpp', kernel / 'lut/kernels.cpp']
        flags += ['-DFT_ENABLE_LUT=1']
    bound = {str(p.relative_to(source)): digest(p) for p in sorted(kernel.rglob('*'))
             if p.is_file() and p.suffix in ('.cpp', '.h', '.cu', '.cuh')}
    split = source / SPLIT_SOURCE
    if variant == 'lut':
        check_no_symlinks(split)
        if not split.is_file():
            raise ValueError('LUT build requires the guarded k110 fast split source')
        bound[SPLIT_SOURCE] = digest(split)
    k101 = {}
    if variant == 'capture':
        ec_source = source / 'glm53/expert_cache.py'
        check_no_symlinks(ec_source)
        bound['glm53/expert_cache.py'] = digest(ec_source)
        sources, flags = [], []
        hc_source = source / 'glm53/k_hcfuse.py'
        if hc_source.is_file():
            check_no_symlinks(hc_source)
            bound['glm53/k_hcfuse.py'] = digest(hc_source)
            k101 = dict(k101_source=str(directory / 'k101.cu'), k101_name='glm53_k101',
                        k101_cuda_flags=['-lineinfo', '-O3', '--use_fast_math'])
    return dict(schema='e1-precompile/v1', source=str(source), source_revision=head,
                variant=variant, build_directory=str(directory), sources=[str(p) for p in sources],
                cuda_source=str(directory / 'ec.cu' if variant == 'capture' else kernel / 'ft_tier_cu.cu'),
                cuda_name='glm53_ec' if variant == 'capture' else 'ft_tier_cu',
                cuda_flags=['-O3', '-DFT_ENABLE_LUT=1'] if variant == 'lut' else ['-O3'],
                flags=flags, source_sha256=bound,
                cuda_arch=cuda_arch, compiler_jobs=jobs, cuda_home=str(Path(cuda_home).absolute()) if cuda_home else None, production_ready=False, **k101,
                **(dict(split_source=str(split), split_name='ft_split_fast', split_cuda_flags=['-O3', '-DFT_ENABLE_LUT=1'])
                   if variant == 'lut' else {}))


def compile_plan(spec):
    runner_sha256 = digest(__file__)
    directory = Path(spec['build_directory'])
    check_no_symlinks(directory)
    directory.mkdir(parents=True, exist_ok=False)
    # No runtime cache can reach the live checkout or the serving build.
    for key, value in {'TORCH_EXTENSIONS_DIR': directory / 'torch', 'CUDA_CACHE_PATH': directory / 'cuda',
                       'TRITON_CACHE_DIR': directory / 'triton', 'XDG_CACHE_HOME': directory / 'cache',
                       'TMPDIR': directory / 'tmp'}.items():
        check_no_symlinks(value)
        value.mkdir(exist_ok=False)
        os.environ[key] = str(value)
    os.environ.update(MAX_JOBS=str(spec['compiler_jobs']), CUDA_VISIBLE_DEVICES='', TORCH_CUDA_ARCH_LIST=spec['cuda_arch'],
                      PYTHONDONTWRITEBYTECODE='1')
    started = time.monotonic()
    import torch
    versions = torch_versions(torch)
    if vars(torch.cuda).get('_initialized'):
        raise RuntimeError('CUDA was already initialized')
    with contextlib.ExitStack() as guard:
        guard.enter_context(mock.patch.object(torch.cuda, '_is_compiled', return_value=False))
        for name in ('is_available', 'init', '_lazy_init', 'device_count', 'set_device', 'synchronize', 'manual_seed_all'):
            guard.enter_context(mock.patch.object(torch.cuda, name, side_effect=RuntimeError('precompile must not access CUDA')))
        from torch.utils import cpp_extension
        options = dict(name='ft_tier_host', sources=spec['sources'], build_directory=str(directory),
                       extra_cflags=spec['flags'], extra_ldflags=LDFLAGS, with_cuda=False, verbose=True)
        if 'with_sycl' in inspect.signature(cpp_extension.load).parameters:
            options['with_sycl'] = False
        if spec['variant'] != 'capture':
            cpp_extension.load(**options)
        cuda_home_raw = spec.get('cuda_home') or os.environ.get('CUDA_HOME') or cpp_extension.CUDA_HOME
        if not cuda_home_raw:
            raise RuntimeError('set --cuda-home or CUDA_HOME to an installed CUDA toolkit')
        cuda_home = Path(cuda_home_raw)
        check_no_symlinks(cuda_home / 'bin/nvcc')
        if not (cuda_home / 'bin/nvcc').is_file():
            raise RuntimeError('existing CUDA compiler required; package installation is forbidden')
        cpp_extension.CUDA_HOME = str(cuda_home)
        if spec['variant'] == 'capture':
            tree = ast.parse((Path(spec['source']) / 'glm53/expert_cache.py').read_text())
            values = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == '_SRC' for target in node.targets)]
            if len(values) != 1 or not isinstance(values[0], str):
                raise ValueError('expert_cache._SRC must be a single reviewed literal')
            generated = directory / 'ec.cu'
            generated.write_text(values[0])
            cuda_source, cuda_name = str(generated), 'glm53_ec'
        else:
            cuda_source, cuda_name = spec['cuda_source'], 'ft_tier_cu'
        if (cuda_source, cuda_name) != (spec['cuda_source'], spec['cuda_name']):
            raise RuntimeError('compiled CUDA source differs from the prepared recipe')
        options.update(name=cuda_name, sources=[cuda_source], extra_cflags=[], extra_ldflags=[],
                       extra_cuda_cflags=spec['cuda_flags'], with_cuda=True)
        cpp_extension.load(**options)
        if spec.get('k101_name'):
            tree = ast.parse((Path(spec['source']) / 'glm53/k_hcfuse.py').read_text())
            values = [ast.literal_eval(node.value) for node in tree.body if isinstance(node, ast.Assign)
                      and any(isinstance(target, ast.Name) and target.id == '_SRC' for target in node.targets)]
            if len(values) != 1 or not isinstance(values[0], str):
                raise ValueError('k_hcfuse._SRC must be a single reviewed literal')
            Path(spec['k101_source']).write_text(values[0])
            options.update(name=spec['k101_name'], sources=[spec['k101_source']], extra_cuda_cflags=spec['k101_cuda_flags'])
            cpp_extension.load(**options)
        if spec['variant'] == 'lut':
            options.update(name=spec['split_name'], sources=[spec['split_source']], extra_cuda_cflags=spec['split_cuda_flags'])
            cpp_extension.load(**options)
    if vars(torch.cuda).get('_initialized'):
        raise RuntimeError('precompile unexpectedly initialized CUDA')
    # Fingerprint after compilation as well, rejecting source races.
    for rel, expected in spec['source_sha256'].items():
        if digest(Path(spec['source']) / rel) != expected:
            raise RuntimeError('source changed while compiling')
    if digest(__file__) != runner_sha256:
        raise RuntimeError('precompile tool changed during build')
    receipt = dict(spec, runner_sha256=runner_sha256,
                   generated_source_sha256=({cuda_source: digest(cuda_source),
                                             **({spec['k101_source']: digest(spec['k101_source'])} if spec.get('k101_name') else {})}
                                            if spec['variant']=='capture' else {}),
                   state='COMPILED_NO_GPU_CONTEXT', **versions,
                   elapsed_seconds=time.monotonic() - started, cuda_initialized=False,
                   outputs={p.name: digest(p) for p in directory.glob('*.so')})
    expected = {'glm53_ec.so'} if spec['variant'] == 'capture' else {'ft_tier_host.so', 'ft_tier_cu.so'}
    if spec['variant'] == 'lut':
        expected |= {'ft_split_fast.so'}
    if spec.get('k101_name'):
        expected |= {'glm53_k101.so'}
    if set(receipt['outputs']) != expected:
        raise RuntimeError('unexpected module names: use a fresh process for each variant')
    atomic_json(directory / 'reviewed-precompile.json', receipt)
    return receipt


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=REPOSITORY,
                        help='serving checkout (default: repository containing this tool)')
    parser.add_argument('--revision', required=True, help='reviewed full serving checkout commit')
    parser.add_argument('--build-root', type=Path, default=Path('./e1-work/precompiled'))
    parser.add_argument('--variant', choices=('lut', 'capture'), default='lut')
    parser.add_argument('--cuda-home', type=Path, help='installed CUDA toolkit; otherwise CUDA_HOME/discovery')
    parser.add_argument('--cuda-arch', default='8.6', help='one explicit architecture; 8.6 for RTX 3090')
    parser.add_argument('--jobs', type=int, default=2, help='maximum parallel compiler jobs')
    parser.add_argument('--nice', type=int, default=0, help='increase compiler niceness by 0..19')
    parser.add_argument('--plan', action='store_true', help='validate/print without importing Torch or compiling')
    args = parser.parse_args(argv)
    if not 0 <= args.nice <= 19:
        raise ValueError('--nice must be 0..19')
    spec = plan(args.source, args.build_root, args.variant, args.revision,
                cuda_home=args.cuda_home, cuda_arch=args.cuda_arch, jobs=args.jobs)
    if args.nice and not args.plan:
        os.nice(args.nice)
    result = spec if args.plan else compile_plan(spec)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
