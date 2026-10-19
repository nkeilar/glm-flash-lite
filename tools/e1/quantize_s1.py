#!/usr/bin/env python3
"""Exact E1 recipe, per-expert imatrix adapter and safe GGUF publication.

The command records CPU-only quantizer provenance. Full conversion is exposed
by quantize_s2.py. Model I/O is streaming and uses only the Python standard library.
"""
from __future__ import annotations

import argparse
import array
import contextlib
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import struct
import subprocess
import sys
import tempfile
import time

from gguf import (GGUF, TYPES, check_no_symlinks, read_gguf, sha256_file,
                  tensor_bytes, write_subset)

REVISION = 'a38483c8cd5df544f53d70fb281afe97369d5ab6'
IK_REVISION = '5bf8f0fe4db98e9560de0f495232f8d3a733c678'
MATRICES = ('gate', 'up', 'down')
METADATA_SHARD = 'Q8_0/GLM-5.3-Flash-Q8_0-00001-of-00008.gguf'
PINNED = {'Q8_0/GLM-5.3-Flash-Q8_0-00001-of-00008.gguf': (9429984, '5dc239a59cbd2d3b9d90b647cf697b80c0f4101bfaa2280b3102156e94a24ccf'), 'Q8_0/GLM-5.3-Flash-Q8_0-00002-of-00008.gguf': (49665857344, '8aabbbc0a92c551894f62a0737b433c9fad60a94baaa61e669f96329bb1d88a6'), 'Q8_0/GLM-5.3-Flash-Q8_0-00003-of-00008.gguf': (49822821280, '7b50941b404fed3413be4e0ca52af43e79113e4b3e2c940d1f4cdf74e95ebb39'), 'Q8_0/GLM-5.3-Flash-Q8_0-00004-of-00008.gguf': (49849652064, 'ff27fb4536d524cae687c4fb88dfa1b43a8ee0b4256e55304f95dc1aa38e8b81'), 'Q8_0/GLM-5.3-Flash-Q8_0-00005-of-00008.gguf': (49974608352, 'c088636854c5da312e44a7be4195cd765dd9eb6339d7b014d1da4192851a9b5b'), 'Q8_0/GLM-5.3-Flash-Q8_0-00006-of-00008.gguf': (49840638624, '4bece55cb9e98720f3ef2e1e9683983aeced77434dbcd54584c0b7c80a2e95cb'), 'Q8_0/GLM-5.3-Flash-Q8_0-00007-of-00008.gguf': (49831834720, 'a9f792204b0b58155e6092f9a4160a0659fcdc98a1c126e563ecc3cd02ad68ad'), 'Q8_0/GLM-5.3-Flash-Q8_0-00008-of-00008.gguf': (41987123840, 'a6cb6887c0047653f08c0ab6e564972451d9511322202970f7052e36052aefd6'), 'imatrix_unsloth.gguf_file': (512687616, '27d2ef6b6be0e314b373080f96322a1e4668b09b192664f3edc231c3b7c2d749')}

def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def sync_directory(path):
    descriptor = os.open(path, os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_json(path: Path, value: dict, *, replace=False):
    """Replace private progress, or exclusively publish an immutable artifact."""
    path = Path(path)
    check_no_symlinks(path)
    payload = (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode()
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + '.', suffix='.pending', dir=path.parent)
    temporary = Path(temporary)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if replace:
            os.replace(temporary, path)
        else:
            os.link(temporary, path)
            temporary.unlink()
        sync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def fingerprint(path: Path):
    check_no_symlinks(path)
    st = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(st.st_mode):
        raise ValueError(f'input must be a regular file: {path}')
    return dict(device=st.st_dev, inode=st.st_ino, size=st.st_size,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)


def verify_source(path: Path, expected: tuple[int, str], receipts: Path,
                  tick=lambda: None) -> dict:
    """Hash all bytes; reuse only an exact unchanged inode's own SHA receipt."""
    size, digest = expected
    before = fingerprint(path)
    if before['size'] != size:
        raise ValueError(f'input incomplete: {path}: {before["size"]} != {size}')
    key = hashlib.sha256(str(path.absolute()).encode()).hexdigest()
    receipt = receipts / (key + '.json')
    check_no_symlinks(receipt)
    if receipt.exists():
        previous = json.loads(receipt.read_text())
        if previous.get('stat') == before and previous.get('sha256') == digest:
            return previous
    actual = sha256_file(path, tick)
    if actual != digest or fingerprint(path) != before:
        raise ValueError(f'input hash mismatch or input changed during verification: {path}')
    result = dict(path=str(path.absolute()), sha256=digest, stat=before, verified_utc=now())
    atomic_json(receipt, result, replace=True)
    return result


def recipe(layer: int, matrix: str) -> str:
    if matrix not in MATRICES or not 3 <= layer <= 44:
        raise ValueError('recipe applies only to main-decoder routed expert triplets')
    if matrix == 'down':
        return 'Q6_K' if layer in (11, 12, 44) else 'IQ4_KS'
    return 'IQ4_KS' if layer == 11 else 'IQ3_K'


def tensor_names(layer: int) -> list[str]:
    return [f'blk.{layer}.ffn_{m}_exps.weight' for m in MATRICES]


def custom_rules(layers: list[int]) -> str:
    # All names explicitly pinned: no COPY mode, broad regex, or base mixture.
    # ik's custom parser uses strcmp against ggml names (unlike ftype parsing).
    spellings = {'IQ3_K': 'iq3_k', 'IQ4_KS': 'iq4_ks', 'Q6_K': 'q6_K'}
    return ','.join('^' + re.escape(name) + '$=' + spellings[recipe(layer, matrix)]
                    for layer in layers for matrix, name in zip(MATRICES, tensor_names(layer)))


def validate_triplet(tensors: list[dict], layer: int, production=True):
    by_name = {t['name']: t for t in tensors}
    names = tensor_names(layer)
    if set(by_name) != set(names) or len(tensors) != 3:
        raise ValueError(f'layer {layer}: need exactly routed gate/up/down, got {list(by_name)}')
    gate, up, down = [by_name[n] for n in names]
    if len(gate['dims']) != 3 or gate['dims'] != up['dims']:
        raise ValueError('gate/up shapes must be identical [hidden,intermediate,experts]')
    hidden, inter, experts = gate['dims']
    if down['dims'] != [inter, hidden, experts]:
        raise ValueError('down must be [intermediate,hidden,experts]')
    if hidden % 256 or inter % 256:
        raise ValueError('expert row length must be divisible by 256')
    if production and gate['dims'] != [4096, 2048, 288]:
        raise ValueError(f'unexpected GLM5.3 expert geometry: {gate["dims"]}')


def floats(data: bytes):
    values = array.array('f')
    values.frombytes(data)
    if sys.byteorder != 'little':
        values.byteswap()
    return values


def imatrix_entry(imatrix: GGUF, tensor: dict) -> tuple[array.array, dict]:
    """Normalize each expert independently; output has ik's ne[0]*ne[2].

    The pinned quantizer's supplied converter divides every expert by counts[0],
    which is incorrect
    for routed experts. We emit normalized float32 values with ncall=1.
    """
    name = tensor['name']
    candidates = [name]
    if '.ffn_gate_exps.' in name:
        candidates += [name.replace('ffn_gate_exps', 'ffn_gate_up_exps'),
                       name.replace('ffn_gate_exps', 'ffn_up_exps')]
    elif '.ffn_up_exps.' in name:
        candidates += [name.replace('ffn_up_exps', 'ffn_gate_up_exps')]
    names = {t['name'] for t in imatrix.tensors}
    chosen = next(((base, suffix) for base in candidates
                   for suffix in ('.in_sum2', '.sums')
                   if base + suffix in names and base + '.counts' in names), None)
    if chosen is None:
        raise ValueError(f'missing imatrix sums/counts for {name}')
    base, suffix = chosen
    sums_t, counts_t = imatrix.tensor(base + suffix), imatrix.tensor(base + '.counts')
    n0, _, experts = tensor['dims']
    # Modern imatrix [input_channels, nmat] + [1,nmat]. nmat may be 1
    # when gate/up see a shared input: broadcast must be explicit and recorded.
    if sums_t['type_id'] != 0 or counts_t['type_id'] != 0:
        raise ValueError('imatrix sums and counts must be F32')
    dims = sums_t['dims']
    if dims not in ([n0, experts], [n0, 1], [n0]):
        raise ValueError(f'imatrix shape mismatch for {name}: {dims}')
    nmat = dims[1] if len(dims) == 2 else 1
    if counts_t['dims'] not in ([1, nmat], [nmat]):
        raise ValueError(f'imatrix count shape mismatch for {name}: {counts_t["dims"]}')
    sums, counts = floats(imatrix.read_tensor(base + suffix)), floats(imatrix.read_tensor(base + '.counts'))
    if any(not math.isfinite(v) or v < 0 for v in sums):
        raise ValueError(f'negative or nonfinite imatrix sum for {name}')
    if any(not math.isfinite(v) or v <= 0 for v in counts):
        raise ValueError(f'uncalibrated expert (zero/invalid imatrix count) for {name}')
    values = array.array('f', (v / counts[i // n0] for i, v in enumerate(sums)))
    if nmat == 1:
        values *= experts
    if len(values) != n0 * experts or not any(values):
        raise ValueError(f'empty or wrong-size imatrix for {name}')
    return values, dict(tensor=name, source=base, sums_suffix=suffix,
                        sums_tensor=base + suffix, count_tensor=base + '.counts',
                        sums_shape=dims, counts_shape=counts_t['dims'],
                        broadcast_experts=nmat == 1, output_values=len(values),
                        calibrated_experts=experts, min_count=min(counts), max_count=max(counts),
                        normalization='per-expert sum/count; legacy ncall=1')


def write_imatrix(imatrix: GGUF, tensors: list[dict], output: Path,
                  tick=lambda: None) -> list[dict]:
    if imatrix.metadata.get('general.type') != 'imatrix':
        raise ValueError('expected GGUF general.type=imatrix')
    check_no_symlinks(output)
    mappings = []
    with output.open('xb') as f:
        f.write(struct.pack('<i', len(tensors)))
        for tensor in tensors:
            tick()
            values, mapping = imatrix_entry(imatrix, tensor)
            data = values.tobytes()
            if sys.byteorder != 'little':
                values.byteswap()
                data = values.tobytes()
            name = tensor['name'].encode()
            f.write(struct.pack('<i', len(name)) + name)
            f.write(struct.pack('<ii', 1, len(values)) + data)
            mappings.append(mapping)
        dataset = json.dumps(imatrix.metadata.get('imatrix.datasets', []), separators=(',', ':')).encode()
        f.write(struct.pack('<ii', int(imatrix.metadata.get('imatrix.chunk_count', 0)), len(dataset)) + dataset)
        f.flush()
        os.fsync(f.fileno())
    return mappings



def dependency_paths(ldd_output: str) -> list[Path]:
    # CPU-only admission covers linkage AND the CMake cache. Check
    # every ldd line, including unresolved dependencies, before recording parts.
    gpu = re.compile(r'(cuda|cublas|cudart|nvidia|vulkan|sycl|hipblas|rocblas|amdhip|opencl|ggml-metal)', re.I)
    if gpu.search(ldd_output):
        raise ValueError('GPU-linked quantizer is forbidden; use CPU-only build')
    paths = []
    for line in ldd_output.splitlines():
        if 'not found' in line:
            raise ValueError(f'missing quantizer dependency: {line.strip()}')
        match = re.search(r'=> (/\S+)', line)
        if match:
            library = Path(match.group(1))
            if re.match(r'lib(?:llama|ggml)(?:[.-])', library.name):
                paths.append(library)
    return paths


def quantizer_components(binary: Path, tick=lambda: None, timeout_seconds=15):
    components = {str(binary.resolve()): sha256_file(binary, tick)}
    result = subprocess.run(['ldd', str(binary)], text=True, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, timeout=timeout_seconds)
    if result.returncode != 0:
        if 'not a dynamic executable' in result.stdout or 'statically linked' in result.stdout:
            return components
        raise ValueError('cannot establish quantizer shared-library provenance')
    for library in dependency_paths(result.stdout):
        components[str(library.resolve())] = sha256_file(library, tick)
    return components


GPU_OPTIONS = ('GGML_CUDA', 'GGML_HIPBLAS', 'GGML_METAL', 'GGML_VULKAN',
               'GGML_SYCL', 'GGML_KOMPUTE', 'GGML_OPENCL', 'GGML_RPC', 'GGML_BACKEND_DL')


def build_receipt(binary: Path, build_dir: Path) -> dict:
    cache_path = build_dir / 'CMakeCache.txt'
    cache = {}
    for line in cache_path.read_text().splitlines():
        match = re.match(r'([^/#][^:]*):[^=]+=(.*)', line)
        if match:
            cache[match.group(1)] = match.group(2)
    if cache.get('GGML_CUDA') != 'OFF' or any(cache.get(key, 'OFF') != 'OFF' for key in GPU_OPTIONS):
        raise ValueError('build receipt requires CUDA and every optional GPU/RPC/dynamic backend OFF')
    if not binary.resolve().is_relative_to(build_dir.resolve()):
        raise ValueError('quantizer binary is not in the declared CMake build directory')
    source = Path(cache['CMAKE_HOME_DIRECTORY'])
    revision = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != IK_REVISION:
        raise ValueError('build source revision is not pinned ik_llama.cpp')
    if subprocess.run(['git', '-C', str(source), 'diff', '--quiet', 'HEAD', '--']).returncode:
        raise ValueError('build source has tracked modifications')
    components = quantizer_components(binary)
    return dict(schema='e1-quantizer-build/v1', source_revision=revision,
                tracked_source_clean=True, cmake_cache_sha256=sha256_file(cache_path),
                disabled_backends={key: cache.get(key, 'OFF') for key in GPU_OPTIONS},
                components={Path(path).name: digest for path, digest in components.items()},
                recorded_utc=now())


def validate_build_receipt(receipt: dict, components: dict):
    if (receipt.get('schema') != 'e1-quantizer-build/v1' or receipt.get('source_revision') != IK_REVISION
            or receipt.get('tracked_source_clean') is not True
            or receipt.get('disabled_backends') != {key: 'OFF' for key in GPU_OPTIONS}
            or not re.fullmatch('[a-f0-9]{64}', receipt.get('cmake_cache_sha256', ''))
            or receipt.get('components') != {Path(path).name: digest for path, digest in components.items()}):
        raise ValueError('quantizer CPU-only/source/component build receipt mismatch')


def validate_expected(source: Path):
    found = {}
    for line in (source / 'EXPECTED').read_text().splitlines():
        name, size, digest = line.split()
        if name in found:
            raise ValueError('duplicate source manifest entry')
        found[name] = (int(size), digest)
    if found == PINNED:
        kind = 'canonical-publisher-lfs'
    else:
        raise ValueError('fetch EXPECTED does not match pinned a38483c source')
    return dict(kind=kind, path=str(source / 'EXPECTED'),
                sha256=sha256_file(source / 'EXPECTED'),
                byte_verification='canonical publisher LFS SHA256 only')


def publish(pending: Path, final: Path, expected_tensors: list[dict],
            context: dict, tick=lambda: None, quantizer_metrics=None):
    """Validate and atomically publish without a replace-existing race."""
    for path in (pending, final):
        check_no_symlinks(path)
    before = fingerprint(pending)
    actual = read_gguf(pending)
    expected = {t['name']: (t['dims'], t['target_type']) for t in expected_tensors}
    found = {t['name']: (t['dims'], t['type']) for t in actual.tensors}
    if expected != found or len(actual.tensors) != 3 or any(k.startswith('split.') for k in actual.metadata):
        raise ValueError('output census mismatch; pending GGUF was not published')
    with pending.open('rb') as stream:
        os.fsync(stream.fileno())
    receipt = dict(path=str(final.absolute()), bytes=pending.stat().st_size,
                   sha256=sha256_file(pending, tick), context=context, verified_utc=now())
    if fingerprint(pending) != before:
        raise ValueError('pending GGUF changed during publication verification')
    if quantizer_metrics is not None:
        receipt['quantizer_metrics'] = quantizer_metrics
    intent = final.with_suffix(final.suffix + '.publishing.json')
    verified = final.with_suffix(final.suffix + '.verified.json')
    if any(p.exists() for p in (final, intent, verified)):
        raise FileExistsError(f'will not overwrite existing publication: {final}')
    atomic_json(intent, receipt, replace=False)
    # link is an atomic no-replace publication; both files are in the same directory.
    os.link(pending, final)
    pending.unlink()
    sync_directory(final.parent)
    atomic_json(verified, receipt, replace=False)
    intent.unlink()
    sync_directory(final.parent)
    return receipt


def resume(final: Path, context: dict, tick=lambda: None):
    receipt = final.with_suffix(final.suffix + '.verified.json')
    intent = final.with_suffix(final.suffix + '.publishing.json')
    for path in (final, receipt, intent):
        check_no_symlinks(path)
    if not any(path.exists() for path in (final, receipt, intent)):
        return None
    if not final.exists() or (not receipt.exists() and not intent.exists()):
        raise ValueError(f'orphan publication requires review (never auto-overwrite): {final}')
    result = json.loads((receipt if receipt.exists() else intent).read_text())
    before = fingerprint(final)
    if (result.get('context') != context or sha256_file(final, tick) != result.get('sha256')
            or before['size'] != result.get('bytes') or fingerprint(final) != before):
        raise ValueError(f'resume context/hash mismatch for {final}')
    if not receipt.exists():
        atomic_json(receipt, result, replace=False)
    if intent.exists():
        if json.loads(intent.read_text()) != result:
            raise ValueError(f'publication intent disagrees with receipt: {final}')
        intent.unlink()
        sync_directory(final.parent)
    return result


def conversion_cpus(text=None, threads=None, allowed=None):
    allowed = set(os.sched_getaffinity(0)) if allowed is None else set(allowed)
    chosen = set()
    if text is None:
        chosen = allowed
    else:
        for item in text.split(','):
            if not re.fullmatch(r'\d+(?:-\d+)?', item):
                raise ValueError('--cpus must contain CPU numbers or inclusive ranges')
            lo, _, hi = item.partition('-')
            if int(hi or lo) < int(lo):
                raise ValueError('reversed CPU range')
            chosen.update(range(int(lo), int(hi or lo) + 1))
    if not chosen or not chosen <= allowed:
        raise ValueError('selected CPUs must be in the allowed affinity set')
    if threads is not None and not 1 <= threads <= len(chosen):
        raise ValueError('--threads must be between 1 and the selected CPU count')
    return sorted(chosen), threads or len(chosen)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('build-receipt',))
    parser.add_argument('--quantizer', type=Path, required=True)
    parser.add_argument('--quantizer-build-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='immutable receipt; otherwise print JSON')
    args = parser.parse_args(argv)
    value = build_receipt(args.quantizer, args.quantizer_build_dir)
    if args.output:
        atomic_json(args.output, value, replace=False)
    else:
        print(json.dumps(value, indent=2, sort_keys=True))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        print(f'quantizer provenance failed: {exc}', file=sys.stderr)
        sys.exit(1)
