#!/usr/bin/env python3
"""Build all 42 routed E1 layers from independently SHA-verified Q8_0 shards.

No network, server inspection, deployment or quality claim is performed here.
The public v1 manifest retains historical recipe-approval fields required by
serving preflight; real routed quality reports remain separate required inputs.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import time

import quantize_s1 as core
from gguf import check_no_symlinks, read_gguf, sha256_file, tensor_bytes, write_subset

LAYERS = tuple(range(3, 45))
CANONICAL_BYTES = 143879307264
TP2_BYTES = 144063332352
# Historical conversion approval, not generated reports or current quality proof.
# These fixed digests also appear in reference-bank.json. v1 requires these
# legacy fields even for fresh builds; real_pass remains false throughout.
HISTORICAL_PROXY_SHA256 = [
    '58bc916f3162fe5ef41e72a8d8cc0d997683c526a1ed5b015febe0898d99df9a',
    '15fdf90f839fb7107cf098d35c806bbf8aa419cd19df981c1f71a72f7d73cc23',
]
OFFSET_CONVENTION = 'tensor offset is relative to artifact.data_start; row_stride includes IQ4_KS scale'


def declared_inventory():
    result = []
    ids = {value[0]: key for key, value in core.TYPES.items()}
    for layer in LAYERS:
        for matrix, name in zip(core.MATRICES, core.tensor_names(layer)):
            dims = [2048, 4096, 288] if matrix == 'down' else [4096, 2048, 288]
            typ = core.recipe(layer, matrix)
            result.append(dict(layer=layer, name=name, dims=dims, type=typ,
                               type_id=ids[typ], nbytes=tensor_bytes(dims, ids[typ])))
    if len(result) != 126 or sum(t['nbytes'] for t in result) != CANONICAL_BYTES:
        raise ValueError('declared full-bank recipe byte ledger is inconsistent')
    return result


def receipt_json(path, limit=16 * 2**20):
    check_no_symlinks(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW), 'rb') as stream:
        if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
            raise ValueError('receipt must be a regular file')
        payload = stream.read(limit + 1)
    if len(payload) > limit:
        raise ValueError('receipt exceeds bounded JSON size')
    return json.loads(payload), hashlib.sha256(payload).hexdigest()


def fetch_done(source):
    path = source / 'FETCH-DONE'
    before = core.fingerprint(path)
    with path.open('rb') as stream:
        content = stream.read(7)
    if content not in (b'DONE', b'DONE\n') or core.fingerprint(path) != before:
        raise ValueError('conversion requires FETCH-DONE exactly DONE')
    return dict(sha256=hashlib.sha256(content).hexdigest(), state='DONE')


def all_sources_unchanged(verified, source):
    for name, receipt in verified.items():
        if core.fingerprint(source / name) != receipt['stat']:
            raise ValueError(f'verified source changed: {name}')


def inventory(shards):
    by_name = {}
    for shard in shards:
        for tensor in shard.tensors:
            if tensor['name'] in by_name:
                raise ValueError(f'duplicate source tensor {tensor["name"]}')
            by_name[tensor['name']] = (shard, tensor)
    for layer in LAYERS:
        try:
            triplet = [by_name[name][1] for name in core.tensor_names(layer)]
        except KeyError as exc:
            raise ValueError(f'missing routed tensor: {exc}') from exc
        core.validate_triplet(triplet, layer)
        if any(t['type'] != 'Q8_0' for t in triplet):
            raise ValueError('full-bank source routed tensors must be original Q8_0')
    return by_name


def check_census(path, expected, q8=False):
    actual = read_gguf(path)
    want = {t['name']: (t['dims'], 'Q8_0' if q8 else t['target_type']) for t in expected}
    got = {t['name']: (t['dims'], t['type']) for t in actual.tensors}
    if got != want or len(actual.tensors) != 3 or any(k.startswith('split.') for k in actual.metadata):
        raise ValueError('completed artifact has wrong tensor census')
    return actual


def prepare_imatrix(imatrix, tensors, scratch, layer):
    """Keep a stable basename in GGUF metadata, independent of local paths."""
    path = scratch / f'blk.{layer:02d}.imatrix.dat'
    pending = path.with_suffix('.dat.pending')
    for item in (path, pending):
        check_no_symlinks(item)
    # Only this deterministic, unpublished scratch filename may be restarted.
    pending.unlink(missing_ok=True)
    mapping = core.write_imatrix(imatrix, tensors, pending)
    digest = sha256_file(pending)
    if path.exists():
        if sha256_file(path) != digest:
            raise ValueError('existing normalized imatrix differs; refusing to overwrite')
        pending.unlink()
    else:
        os.link(pending, path)
        pending.unlink()
        core.sync_directory(scratch)
    return path, mapping, digest


def quantizer_command(quantizer, layer, q8, pending, imatrix, threads):
    # ik embeds this exact argument as quantize.imatrix.file. An absolute path
    # would make GGUF hashes depend on the build machine's directory layout.
    return [str(quantizer.absolute()), '--allow-requantize', '--slab-size', '512',
            '--imatrix', imatrix.name, '--custom-q', core.custom_rules([layer]),
            str(q8.absolute()), str(pending.absolute()), 'IQ3_K', str(threads)]


def run_quantizer(command, scratch, layer):
    log = scratch / f'blk.{layer:02d}.{time.time_ns()}.quantize.log'
    check_no_symlinks(log)
    started = time.monotonic()
    with log.open('xb') as stream:
        child = subprocess.Popen(command, cwd=scratch, stdout=stream, stderr=subprocess.STDOUT)
        try:
            code = child.wait()
        finally:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
        stream.flush()
        os.fsync(stream.fileno())
    if code:
        raise RuntimeError(f'quantizer exited {code}; see {log}')
    return dict(wall_seconds=time.monotonic() - started, returncode=code, log=str(log))


def convert_layer(layer, by_name, meta, imatrix, verified, source, scratch,
                  output, quantizer, cpus, threads, components, tool_hashes):
    selection = [(by_name[name][0], name) for name in core.tensor_names(layer)]
    tensors = [by_name[name][1] for name in core.tensor_names(layer)]
    imat_path, mappings, normalized_sha = prepare_imatrix(imatrix, tensors, scratch, layer)
    context = dict(revision=core.REVISION, layer=layer,
        sources={str(shard.path.relative_to(source)): verified[str(shard.path.relative_to(source))]['sha256']
                 for shard, _ in selection},
        metadata_source=verified[core.METADATA_SHARD]['sha256'],
        imatrix=verified['imatrix_unsloth.gguf_file']['sha256'],
        normalized_imatrix_sha256=normalized_sha,
        quantizer=components, tools=tool_hashes, recipe=core.custom_rules([layer]),
        imatrix_metadata_basename=imat_path.name)
    expected = [dict(t, source_type='Q8_0', target_type=core.recipe(layer, matrix))
                for matrix, t in zip(core.MATRICES, tensors)]
    q8_path = scratch / f'blk.{layer:02d}.q8_0.gguf'
    lut_path = output / f'blk.{layer:02d}.lut.gguf'
    lut_receipt = core.resume(lut_path, context)
    if lut_receipt is not None:
        check_census(lut_path, expected)
        # Non-checkpoint Q8 scratch can be removed after durable progress. Its
        # immutable receipt survives, so a completed LUT never needs extraction.
        if q8_path.exists() or layer in (3, 11):
            q8_receipt = core.resume(q8_path, context)
            if q8_receipt is None:
                raise ValueError('checkpoint Q8 reference is missing')
            check_census(q8_path, expected, q8=True)
        else:
            q8_receipt = receipt_json(q8_path.with_suffix('.gguf.verified.json'))[0]
            if q8_receipt.get('context') != context:
                raise ValueError('cleaned Q8 receipt context mismatch')
    else:
        q8_receipt = core.resume(q8_path, context)
        if q8_receipt is None:
            pending = q8_path.with_suffix('.gguf.pending')
            check_no_symlinks(pending)
            pending.unlink(missing_ok=True)
            write_subset(selection, pending, meta)
            all_sources_unchanged(verified, source)
            q8_receipt = core.publish(pending, q8_path,
                [dict(t, target_type='Q8_0') for t in tensors], context)
        check_census(q8_path, expected, q8=True)
        q8_before = core.fingerprint(q8_path)
        pending = lut_path.with_suffix('.gguf.pending')
        check_no_symlinks(pending)
        pending.unlink(missing_ok=True)
        command = quantizer_command(quantizer, layer, q8_path, pending, imat_path, threads)
        metrics = run_quantizer(command, scratch, layer)
        metrics['execution'] = dict(cpus=cpus, threads=threads)
        if (core.fingerprint(q8_path) != q8_before
                or sha256_file(q8_path) != q8_receipt['sha256']
                or core.fingerprint(q8_path) != q8_before):
            raise ValueError('Q8 extraction changed during quantization')
        if core.quantizer_components(quantizer) != components:
            raise ValueError('quantizer components changed before publication')
        if sha256_file(imat_path) != normalized_sha:
            raise ValueError('normalized imatrix changed during quantization')
        all_sources_unchanged(verified, source)
        lut_receipt = core.publish(pending, lut_path, expected, context, quantizer_metrics=metrics)
    return dict(layer=layer, q8=q8_receipt, lut=lut_receipt, tensors=expected,
                imatrix_mapping=mappings, quantizer_metrics=lut_receipt.get('quantizer_metrics'))


def cleanup_q8(entry, scratch):
    if entry['layer'] in (3, 11):
        return
    path = scratch / f'blk.{entry["layer"]:02d}.q8_0.gguf'
    if Path(entry['q8']['path']) != path:
        raise ValueError('refuse cleanup outside exact owned per-layer scratch path')
    if receipt_json(path.with_suffix('.gguf.verified.json'))[0] != entry['q8']:
        raise ValueError('Q8 cleanup receipt mismatch')
    check_no_symlinks(path)
    if path.exists():
        if sha256_file(path) != entry['q8']['sha256']:
            raise ValueError('Q8 cleanup hash mismatch')
        path.unlink()
        core.sync_directory(scratch)


def ledger(entries):
    payload = sum(tensor_bytes(t['dims'], next(k for k, v in core.TYPES.items() if v[0] == t['target_type']))
                  for entry in entries for t in entry['tensors'])
    return dict(completed_layers=len(entries), completed_tensors=len(entries) * 3,
        completed_canonical_payload_bytes=payload, target_layers=42, target_tensors=126,
        canonical_payload_bytes=CANONICAL_BYTES, tp2_payload_bytes=TP2_BYTES,
        per_node_tp2_payload_bytes=TP2_BYTES // 2)


def public_manifest(manifest, private_manifest_sha256):
    if manifest['state'] != 'S2_READY' or sorted(e['layer'] for e in manifest['layers']) != list(LAYERS):
        raise ValueError('public loader manifest requires all 42 verified layers')
    layers = []
    for entry in manifest['layers']:
        path = Path(entry['lut']['path'])
        actual = check_census(path, entry['tensors'])
        if path.stat().st_size != entry['lut']['bytes'] or sha256_file(path) != entry['lut']['sha256']:
            raise ValueError('bank artifact changed before public manifest publication')
        tensors = [dict(name=t['name'], dims=t['dims'], type=t['type'], type_id=t['type_id'],
            nbytes=t['nbytes'], offset=t['offset'], row_stride=t['nbytes'] // math.prod(t['dims'][1:]))
            for t in actual.tensors]
        layers.append(dict(layer=entry['layer'], artifact=dict(path=path.name,
            sha256=entry['lut']['sha256'], bytes=entry['lut']['bytes'], data_start=actual.data_start), tensors=tensors))
    return dict(schema='e1-lut-bank/v1', state='S2_READY', revision=core.REVISION,
        ik_revision=core.IK_REVISION, imatrix_sha256=core.PINNED['imatrix_unsloth.gguf_file'][1],
        real_pass=False, accepted_quality_verdict='PROXY-PASS',
        accepted_proxy_sha256=HISTORICAL_PROXY_SHA256,
        private_manifest_sha256=private_manifest_sha256,
        offset_convention=OFFSET_CONVENTION, ledger=manifest['ledger'], layers=layers)


def _immutable_manifest(path, value):
    if path.exists():
        old, _ = receipt_json(path)
        if old != value:
            raise ValueError(f'existing immutable manifest differs: {path}')
        return
    core.atomic_json(path, value)


def run(args):
    source, output, scratch = (p.absolute() for p in (args.source, args.output, args.scratch))
    for path in (source, output, scratch):
        check_no_symlinks(path)
    if len({source, output, scratch}) != 3 or any(
            a.is_relative_to(b) for a in (output, scratch) for b in (source,) ):
        raise ValueError('source, output and scratch must be distinct; outputs cannot be inside source')
    output.mkdir(parents=True, exist_ok=True)
    scratch.mkdir(parents=True, exist_ok=True)
    receipts = scratch / 'source-verified'
    check_no_symlinks(receipts)
    receipts.mkdir(exist_ok=True)
    lock_path = scratch / '.conversion.lock'
    output_lock = output / '.conversion.lock'
    check_no_symlinks(lock_path)
    check_no_symlinks(output_lock)
    with lock_path.open('a') as lock, output_lock.open('a') as bank_lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(bank_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        cpus, threads = core.conversion_cpus(args.cpus, args.threads)
        os.sched_setaffinity(0, cpus)
        os.nice(args.nice)
        fetch_done(source)
        core.validate_expected(source)
        verified = {name: core.verify_source(source / name, expected, receipts)
                    for name, expected in core.PINNED.items()}
        components = core.quantizer_components(args.quantizer)
        if args.quantizer_sha256 and sha256_file(args.quantizer) != args.quantizer_sha256:
            raise ValueError('quantizer binary SHA mismatch')
        build, _ = receipt_json(args.build_receipt)
        core.validate_build_receipt(build, components)
        tools = {name: sha256_file(Path(__file__).parent / name)
                 for name in ('gguf.py', 'quantize_s1.py', 'quantize_s2.py')}
        shards = [read_gguf(source / name) for name in core.PINNED if name.endswith('.gguf')]
        meta = next(s for s in shards if s.path == source / core.METADATA_SHARD)
        if meta.metadata.get('general.architecture') != 'glm5-next':
            raise ValueError('source metadata architecture is not glm5-next')
        imatrix = read_gguf(source / 'imatrix_unsloth.gguf_file')
        by_name = inventory(shards)
        entries = []
        for layer in LAYERS:
            print(f'layer {layer}: verify or convert', flush=True)
            entry = convert_layer(layer, by_name, meta, imatrix, verified, source, scratch,
                                  output, args.quantizer, cpus, threads, components, tools)
            all_sources_unchanged(verified, source)
            if core.quantizer_components(args.quantizer) != components:
                raise ValueError('quantizer components changed during conversion')
            entries.append(entry)
            core.atomic_json(scratch / 'conversion-progress.json',
                dict(schema='e1-conversion-progress/v1', layers=entries, ledger=ledger(entries)), replace=True)
            cleanup_q8(entry, scratch)
            selected = [e for e in entries if e['layer'] in (3, 11)]
            if len(selected) == 2:
                s1 = dict(schema='e1-s1/v1', state='S1_READY', stage='S1', revision=core.REVISION,
                    quality_gate='PENDING', quantizer=dict(source_revision=core.IK_REVISION), layers=selected)
                _immutable_manifest(output / 's1-manifest.json', s1)
        if ledger(entries)['completed_canonical_payload_bytes'] != CANONICAL_BYTES:
            raise ValueError('completed bank payload does not equal canonical byte ledger')
        manifest = dict(schema='e1-s2/v1', state='S2_READY', revision=core.REVISION,
            ik_revision=core.IK_REVISION, real_pass=False, layers=entries, ledger=ledger(entries),
            historical_proxy_sha256=HISTORICAL_PROXY_SHA256)
        private = output / 's2-manifest.json'
        _immutable_manifest(private, manifest)
        value = public_manifest(manifest, sha256_file(private))
        # Validate against the serving code, without changing its pins or behavior.
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
        from glm53.lut_manifest import validate_structure
        validate_structure(value)
        _immutable_manifest(output / 'e1-lut-bank.json', value)
        print(json.dumps(dict(state='S2_READY', manifest=str(output / 'e1-lut-bank.json'),
                              quality='real routed activation reports are still required')))
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'run'))
    parser.add_argument('--source', type=Path, default=Path('./e1-work/source'))
    parser.add_argument('--output', type=Path, default=Path('./e1-work/bank'))
    parser.add_argument('--scratch', type=Path, default=Path('./e1-work/scratch'))
    parser.add_argument('--quantizer', type=Path)
    parser.add_argument('--quantizer-sha256', help='optional extra pin; build receipt always pins all components')
    parser.add_argument('--build-receipt', type=Path)
    parser.add_argument('--cpus', help='allowed CPU numbers, e.g. 0-3,8-11; default current affinity')
    parser.add_argument('--threads', type=int, help='default selected CPU count')
    parser.add_argument('--nice', type=int, choices=range(20), default=10)
    args = parser.parse_args(argv)
    if args.action == 'plan':
        print(json.dumps(dict(layers=list(LAYERS), tensors=declared_inventory(),
                              canonical_payload_bytes=CANONICAL_BYTES, tp2_payload_bytes=TP2_BYTES), indent=2))
        return 0
    if not args.quantizer or not args.build_receipt:
        parser.error('run requires --quantizer and --build-receipt')
    args.quantizer = args.quantizer.absolute()
    previous = signal.signal(signal.SIGTERM, lambda *_: (_ for _ in ()).throw(KeyboardInterrupt()))
    try:
        return run(args)
    finally:
        signal.signal(signal.SIGTERM, previous)


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError, KeyboardInterrupt) as exc:
        print(f'conversion stopped: {exc}', file=sys.stderr)
        sys.exit(1)
