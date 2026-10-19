#!/usr/bin/env python3
"""Extract verified real weight fixtures and measure the serving LUT backend.

The complete hardware table measures all three weight recipes, M=1..4 and
K=0..8 with eight distinct experts. Every timed call flushes its owned weight
arenas. CPU timing loads a SHA-verified precompiled module and never uses CUDA.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import re
import statistics
import sys
import time
from unittest import mock

from gguf import check_no_symlinks, read_gguf, sha256_file, tensor_bytes, write_gguf
from quantize_s1 import atomic_json
from quality_checkpoint import stable_json

PROFILES = {'ordinary': (3, [138, 138, 144]),
            'upgraded_gate': (11, [144, 144, 14]),
            'upgraded_down': (12, [138, 138, 14])}
PARTS = ('gate', 'up', 'down')
PLANNED_BYTES = 4 * 2**30
EXPERTS = list(range(8))


def require(value, message):
    if not value:
        raise ValueError(message)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def file_identity(path):
    check_no_symlinks(path)
    st = Path(path).stat()
    import stat
    require(stat.S_ISREG(st.st_mode), 'artifact must be a regular file')
    return dict(dev=st.st_dev, ino=st.st_ino, size=st.st_size,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)


def load_public_tools(source):
    source = Path(source)
    path = source / 'glm53/lut_manifest.py'
    check_no_symlinks(path)
    spec = importlib.util.spec_from_file_location('e1_cost_lut_manifest', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def owned_output(raw):
    path = Path(raw).absolute()
    require('..' not in path.parts, 'output path must not traverse parents')
    check_no_symlinks(path)
    return path


def output_space(path, reserve=2**30, min_free_gib=1):
    parent = Path(path).parent
    while not parent.exists():
        parent = parent.parent
    st = os.statvfs(parent)
    require(st.f_bavail * st.f_frsize >= reserve + min_free_gib * 2**30,
            'insufficient free disk for output plus requested reserve')


def publish_file(pending, final):
    """Publish an fsynced file atomically without replacing any existing name."""
    check_no_symlinks(pending)
    check_no_symlinks(final)
    os.link(pending, final)
    pending.unlink()
    directory = os.open(final.parent, os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def verify_bank(source, manifest, expected_sha, revision):
    import re
    import subprocess
    require(isinstance(revision,str) and re.fullmatch('[a-f0-9]{40}',revision),'pin the reviewed serving source full revision')
    env=dict(os.environ,GIT_OPTIONAL_LOCKS='0')
    require(subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True,timeout=10,env=env).strip()==revision,
            'serving source revision differs from reviewed cost tools')
    require(not subprocess.check_output(['git','-C',str(source),'status','--porcelain','--untracked-files=no'],text=True,timeout=10,env=env),
            'serving source must have no tracked changes')
    public = load_public_tools(source)
    bank = public.load_manifest(manifest, check_files=False)
    require(bank.sha256 == expected_sha, 'bank manifest SHA mismatch')
    return public, bank


def verify_sources(args, tick=lambda: None):
    public, bank = verify_bank(args.serving_source, args.bank_manifest,
                               args.bank_manifest_sha256, args.serving_revision)
    return public, bank, Path(args.bank_manifest).absolute().parent


def copy_range(source, offset, size, output, tick):
    before = file_identity(source)
    result = hashlib.sha256()
    with Path(source).open('rb') as src, Path(output).open('xb') as dst:
        src.seek(offset)
        while size:
            tick()
            block = src.read(min(size, 4 * 2**20))
            require(block, 'source truncated during expert extraction')
            dst.write(block)
            result.update(block)
            size -= len(block)
        dst.flush()
        os.fsync(dst.fileno())
    require(file_identity(source) == before, 'source changed during expert extraction')
    return result.hexdigest()


def extract(args, tick=lambda: None):
    public, bank, source_root = verify_sources(args, tick)
    output = owned_output(args.out)
    require(not output.exists(), 'fixture output exists; retain it and select a fresh owned directory')
    output_space(output, PLANNED_BYTES, args.min_free_disk_gib)
    output.mkdir(parents=True, exist_ok=False)
    result = dict(schema='e1-real-lut-fixtures/v1', state='PREPARING', experts=EXPERTS,
                  bank_manifest_sha256=bank.sha256,
                  source_revision=bank.value['revision'], act_limit=args.act_limit,
                  extractor_sha256=sha256_file(__file__), layers=[], real_pass=False)
    atomic_json(output / '.extract-owner.json', result)
    sources = {}
    for row in bank.value['layers']:
        if row['layer'] not in args.layers:
            continue
        source = source_root / row['artifact']['path']
        sources[row['layer']] = (source, row, public.verify_artifact(source, row, tick))
    for layer in args.layers:
        source_layers = (layer,) * 3
        tick()
        tensors, ranges, payloads, source_parts = [], [], [], []
        for index, (part, source_layer) in enumerate(zip(PARTS, source_layers)):
            source, row, descriptor = sources[source_layer]
            original = descriptor['tensors'][index]
            require(original['experts'] == 288, 'expected actual 288-expert converted layer')
            dims = [original['width'], original['rows'], 8]
            size = tensor_bytes(dims, original['type_id'])
            require(size == original['nbytes'] // 288 * 8, 'expert bytes are not contiguous')
            payload = output / f'blk.{layer}.{part}.experts-0-7.payload.pending'
            payload_hash = copy_range(source, original['offset'], size, payload, tick)
            payloads.append(payload)
            tensors.append(dict(name=f'blk.{layer}.ffn_{part}_exps.weight', dims=dims,
                                type_id=original['type_id'], data=payload))
            provenance = dict(part=part, layer=source_layer, sha256=row['artifact']['sha256'], type_id=original['type_id'])
            source_parts.append(provenance)
            ranges.append(dict(name=tensors[-1]['name'], source_offset=original['offset'], bytes=size,
                               sha256=payload_hash, experts=EXPERTS, dims=dims, type_id=original['type_id'],
                               source=provenance))
        final = output / f'blk.{layer:02d}.experts-0-7.lut.gguf'
        pending = final.with_name(final.name + '.pending')
        receipt = write_gguf(pending, {'general.architecture': (8, 'glm5-next'), 'general.alignment': (4, 32),
                                    'glm5-next.embedding_length': (4, 4096), 'glm5-next.feed_forward_length': (4, 2048),
                                    'glm5-next.expert_count': (4, 8)}, tensors, tick)
        require(all(file_identity(path) == desc['identity'] for path, _, desc in sources.values()),
                'converted source changed before fixture publication')
        entry = dict(layer=layer, fixture=receipt, ranges=ranges, source_parts=source_parts,
                     composite=len(set(source_layers)) > 1,
                     source=dict(path=str(sources[source_layers[0]][0]), sha256=source_parts[0]['sha256']),
                     )
        # Re-read/hash before the final name becomes visible.
        verify_fixture(entry, tick, expected_experts=8)
        publish_file(pending, final)
        # Fixture receipts are portable with their containing directory.
        entry['fixture']['path']=final.name
        result['layers'].append(entry)
        for payload in payloads:
            payload.unlink()  # only exclusive files created by this invocation
    tick()
    result.update(state='FIXTURES_READY')
    atomic_json(output / 'fixture-manifest.json', result)
    return dict(path=str(output / 'fixture-manifest.json'), sha256=sha256_file(output / 'fixture-manifest.json', tick),
                state=result['state'], experts=EXPERTS, real_pass=False)


def verify_fixture(entry, tick=lambda: None, expected_experts=8):
    layer, artifact = entry['layer'], entry['fixture']
    require(layer in (3, 11, 12, 44), 'cost fixtures require actual ordinary/upgraded model layers')
    path = Path(artifact['path'])
    before = file_identity(path)
    parts = entry.get('source_parts')
    if parts is not None:
        require(isinstance(parts, list) and len(parts) == 3, 'fixture requires three part provenance records')
        require(entry.get('composite') in (True, False), 'fixture composite label missing')
    require(before['size'] == artifact['bytes'] and sha256_file(path, tick) == artifact['sha256'], 'fixture SHA/size mismatch')
    parsed = read_gguf(path)
    require(parsed.metadata.get('general.architecture') == 'glm5-next' and len(parsed.tensors) == 3, 'fixture must be an isolated real LUT triplet')
    descriptors, types = [], []
    expected_types = PROFILES['upgraded_gate' if layer == 11 else 'upgraded_down' if layer in (12, 44) else 'ordinary'][1]
    ranges = {item['name']: item for item in entry['ranges']}
    for index, (part, typ) in enumerate(zip(PARTS, expected_types)):
        tensor = parsed.tensor(f'blk.{layer}.ffn_{part}_exps.weight')
        dims = [2048, 4096, expected_experts] if part == 'down' else [4096, 2048, expected_experts]
        require(tensor['dims'] == dims and tensor['type_id'] == typ, 'fixture needs eight distinct experts with actual production geometry/recipe')
        provenance = ranges[tensor['name']]
        if parts is not None:
            require(provenance.get('source') == parts[index] and parts[index]['part'] == part and
                    parts[index]['type_id'] == typ, 'fixture range and part source provenance differ')
        require(provenance['experts'] == list(range(expected_experts)) and provenance['dims'] == dims and
                provenance['bytes'] == tensor['nbytes'] and provenance['type_id'] == typ,
                'fixture expert-range provenance mismatch')
        hashed = hashlib.sha256()
        with path.open('rb') as stream:
            stream.seek(parsed.data_start + tensor['offset'])
            left = tensor['nbytes']
            while left:
                tick()
                block = stream.read(min(left, 4 * 2**20))
                require(block, 'fixture tensor truncated')
                hashed.update(block)
                left -= len(block)
        require(hashed.hexdigest() == provenance['sha256'], 'fixture payload range SHA mismatch')
        descriptors.append(dict(offset=parsed.data_start + tensor['offset'], nbytes=tensor['nbytes'],
                                type_id=typ, width=dims[0], rows=dims[1], experts=expected_experts))
        types.append(typ)
    require(file_identity(path) == before, 'fixture changed during verification')
    return dict(model_layer=layer, internal_li=layer-3, path=str(path.absolute()), identity=before, tensors=descriptors), types


def validate_fixture_manifest(path, expected_sha, tick=lambda: None):
    manifest, actual_sha = stable_json(path, 4 * 2**20)
    require(actual_sha == expected_sha, 'owner-approved fixture manifest SHA mismatch')
    require(manifest.get('schema') == 'e1-real-lut-fixtures/v1' and manifest.get('state') == 'FIXTURES_READY', 'real converted fixture receipt required')
    require(manifest.get('experts') == EXPERTS, 'K8 requires eight distinct experts; never repeat a two-expert fixture')
    require(isinstance(manifest.get('layers'), list) and manifest['layers'], 'no fixture layers')
    require(len({row['layer'] for row in manifest['layers']}) == len(manifest['layers']), 'duplicate fixture layer')
    verified=[]
    for original in manifest['layers']:
        entry=json.loads(json.dumps(original,allow_nan=False))
        artifact=Path(entry['fixture']['path'])
        if not artifact.is_absolute():
            require(artifact.parts==(artifact.name,) and artifact.name not in ('.','..'),'relative fixture path escapes its manifest directory')
            entry['fixture']['path']=str(Path(path).absolute().parent/artifact)
        verified.append(dict(entry=entry,descriptor=verify_fixture(entry,tick)[0]))
    return manifest, verified


def load_precompiled(receipt_path, expected_sha, source, revision, tick=lambda: None, *,
                     expected_build_directory=None, torch_metadata_version=None):
    """Hash receipt/source/module bytes; runtime admission uses its own namespace."""
    receipt, actual = stable_json(receipt_path, 2**20)
    require(actual == expected_sha, 'owner-approved precompile receipt SHA mismatch')
    require(receipt.get('schema') == 'e1-precompile/v1' and receipt.get('state') == 'COMPILED_NO_GPU_CONTEXT' and receipt.get('variant') == 'lut' and
            receipt.get('source_revision') == revision and receipt.get('cuda_initialized') is False, 'fresh reviewed LUT precompile required')
    require(isinstance(receipt.get('torch_version'),str) and receipt['torch_version'],
            'precompile exact Torch runtime version is missing')
    if torch_metadata_version is not None:
        verify_precompiled_torch(receipt, torch_metadata_version)
    require(receipt.get('source_sha256'), 'precompile source hashes absent')
    for relative, expected in receipt['source_sha256'].items():
        require(not Path(relative).is_absolute() and '..' not in Path(relative).parts, 'invalid source hash path')
        path = Path(source) / relative
        before = file_identity(path)
        require(sha256_file(path, tick) == expected and file_identity(path) == before,
                'precompile source changed; rebuild before timing')
    binary = Path(receipt_path).parent / 'ft_tier_host.so'
    before = file_identity(binary)
    build_sha = sha256_file(binary, tick)
    require(file_identity(binary) == before, 'compiled host changed during verification')
    require(receipt.get('outputs', {}).get(binary.name) == build_sha, 'compiled host artifact SHA mismatch')
    declared=Path(receipt.get('build_directory', ''))
    expected=Path(expected_build_directory) if expected_build_directory is not None else Path(receipt_path).parent
    require(declared.is_absolute() and '..' not in declared.parts and declared==expected.absolute(), 'precompile build directory mismatch')
    return binary, build_sha, receipt


def verify_precompiled_torch(receipt, metadata_version, runtime_version=None):
    """Metadata is a pre-import check; the loaded ABI must match exact runtime."""
    recorded = receipt.get('torch_version')
    require(isinstance(recorded,str) and bool(recorded) and metadata_version == recorded.split('+',1)[0],
            'precompile Torch metadata differs from compiled runtime public version')
    require(runtime_version is None or runtime_version == recorded,
            'precompile exact Torch runtime mismatch')


def route_ids(k, experts=8):
    require(type(k) is int and 0 <= k <= experts and experts == 8, 'K grid requires eight distinct actual experts')
    return list(range(k))


def time_cell(call, k, m, samples, warmup, tick, clock=time.perf_counter_ns, *, prepare=None, prepare_clock=time.perf_counter_ns):
    for _ in range(warmup):
        tick()
        call(k, m)
    values, prepared = [], []
    for _ in range(samples):
        tick()
        started = prepare_clock()   # separate clock: the timed call's own clock reads stay exactly begin/end
        if prepare is not None: prepare()
        prepared.append((prepare_clock()-started) / 1e6)
        begin = clock()
        call(k, m)
        values.append((clock()-begin) / 1e6)
    require(all(math.isfinite(v) and v >= 0 for v in values), 'invalid per-call measurement')
    cell = dict(median_ms=statistics.median(values), samples=len(values), raw_ms=values)
    if prepare is not None:
        cell['prepare_median_ms'] = statistics.median(prepared)
    return cell


def decode_cadence(call, m, period_ms, calls, clock=time.perf_counter_ns):
    """Untimed K=0 calls one decode-layer period apart: decode reaches the CPU tier every
    ~1.5 ms, so its workers are awake and recently active while the routed weights stay cold."""
    period = int(period_ms * 1e6)
    for _ in range(calls):
        start = clock()
        call(0, m)
        while clock() - start < period:
            pass


def flush_fixture_weights(module,expected_bytes):
    actual=module.tier_lut_fixture_flush_weights()
    require(type(actual) is int and type(expected_bytes) is int and 0<expected_bytes<=256*2**20 and actual==expected_bytes,
            'owned-weight flush byte receipt differs from the complete loaded TP2 payload')


def validate_source_parts(parts, layer, bank_rows, *, composite=False):
    require(isinstance(parts, list) and len(parts) == 3, 'exactly three source-part receipts required')
    require(not composite or layer == 12, 'composite fixture is exclusively the down-Q6 performance profile')
    expected_layers = (3, 3, 11) if composite else (layer,) * 3
    types = PROFILES['upgraded_gate' if layer == 11 else 'upgraded_down' if layer in (12, 44) else 'ordinary'][1]
    for part, source_layer, typ, proof in zip(PARTS, expected_layers, types, parts):
        require(proof == dict(part=part, layer=source_layer, type_id=typ,
                              sha256=bank_rows[source_layer]['artifact']['sha256']),
                'fixture part provenance differs from the pinned converted source and exact recipe')


def fit_cells(cells):
    require(set(cells) == {str(k) for k in range(9)}, 'fit requires complete K0-8 grid')
    y = [float(cells[str(k)]['median_ms']) for k in range(9)]
    mean = statistics.mean(y)
    b = sum((k - 4) * (value - mean) for k, value in enumerate(y)) / 60
    a = mean - 4 * b
    residuals = [value - a - b*k for k, value in enumerate(y)]
    total = sum((value - mean)**2 for value in y)
    return dict(a_ms=a, b_ms=b, rmse_ms=math.sqrt(statistics.mean(r*r for r in residuals)),
                r_squared=1 - sum(r*r for r in residuals)/total if total else 1.0,
                glm53_ct_lut=','.join(format(v, '.9g') for v in y[1:]),
                use='diagnostic fit; production consumes the exact per-profile K/M median table',
                trellis_n32ctrl_a_ms=0.182, trellis_n32ctrl_b_ms=0.255)


def physical_core(cpu):
    root = Path(f'/sys/devices/system/cpu/cpu{cpu}/topology')
    return dict(package=(root / 'physical_package_id').read_text().strip(),
                core=(root / 'core_id').read_text().strip())


def cpu_selection(cpus, coordinator):
    require(isinstance(cpus, list) and len(cpus) == 42 and len(set(cpus)) == 42 and
            all(type(cpu) is int and 0 <= cpu < 1024 for cpu in cpus),
            'serving costs require 42 distinct physical worker CPUs')
    require(type(coordinator) is int and 0 <= coordinator < 1024 and coordinator not in cpus,
            'coordinator must differ from worker CPUs')
    topology = {str(cpu): physical_core(cpu) for cpu in cpus}
    physical = {(row['package'], row['core']) for row in topology.values()}
    require(len(physical) == 42, 'worker CPUs must occupy distinct physical cores')
    require(physical_core(coordinator) not in topology.values(), 'coordinator needs a separate physical core')
    nodes, workers = [], []
    for cpu in cpus:
        paths = list(Path(f'/sys/devices/system/cpu/cpu{cpu}').glob('node[0-9]*'))
        require(len(paths) == 1, 'each worker must have one discoverable NUMA node')
        node = int(paths[0].name[4:])
        if node not in nodes:
            nodes.append(node)
            workers.append(0)
        workers[nodes.index(node)] += 1
    require(len(nodes) == 2, 'serving costs require exactly two nonempty physical NUMA nodes')
    return topology, dict(nodes=nodes, workers=workers)


def parse_cpus(raw):
    cpus = []
    try:
        for item in raw.split(','):
            if '-' in item:
                first, last = (int(value) for value in item.split('-'))
                require(0 <= first <= last < 1024, 'invalid CPU range')
                cpus.extend(range(first, last + 1))
            else:
                cpus.append(int(item))
    except ValueError as error:
        raise argparse.ArgumentTypeError('CPUs must be comma-separated IDs/ranges') from error
    return cpus


def verify_target_placement(placement, output, profile, layer, stage, result, *, expected):
    """Retain the native diagnostic proof before refusing a target timing claim."""
    okay = (placement.get('placement_verified') is True and placement.get('virtual_tp2') is False and
            placement.get('workers') == expected['workers'] and placement.get('nodes') == expected['nodes'] and
            placement.get('timed_out_jobs') == 0)
    event = dict(event='COST_PLACEMENT_OBSERVED', profile=profile, layer=layer, stage=stage,
                 admitted=okay, placement=placement)
    print(json.dumps(event, sort_keys=True, allow_nan=False), flush=True)
    if okay:
        return
    target = Path(str(output) + '.placement-failed.json')
    require(not target.exists(), 'placement failure receipt exists; preserve it and use a fresh output path')
    failure = dict(schema='e1-lut-cost-placement-failure/v1', state='PLACEMENT-FAILED', complete=False,
                   real_pass=False, production_ready=False, profile=profile, layer=layer, stage=stage,
                   placement=placement, partial_measurement=result,
                   explanation='Native physical placement or deadline receipt failed. This file is diagnostic only and cannot admit a cost table.')
    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(target, failure)
    raise ValueError('physical NUMA placement/deadline unverified; no target timing claim; diagnostic=' + str(target))


def measure(args, tick=lambda: None):
    require(3 <= args.samples <= 10000 and 0 <= args.warmup <= 1000, 'bounded samples/warmup required')
    require(args.activation in ('i16','q8'),'cost measurements require an explicit native activation path')
    require(args.act_limit == 10.0, 'pinned model SwiGLU activation limit must be 10')
    topology, expected_placement = cpu_selection(args.cpus, args.coordinator_cpu)
    original_affinity=set(os.sched_getaffinity(0))
    require({args.coordinator_cpu, *args.cpus} <= original_affinity,
            'start measurement with all selected worker and coordinator CPUs allowed')
    output = owned_output(args.out)
    require(not output.exists(), 'cost output exists; preserve it and use a fresh path')
    output_space(output, min_free_gib=args.min_free_disk_gib)
    public, bank, _ = verify_sources(args, tick)
    fixtures, rows = validate_fixture_manifest(args.fixtures, args.fixtures_sha256, tick)
    selected = selected_profiles(getattr(args, 'profiles', None))
    require(fixtures.get('act_limit') == args.act_limit, 'fixture activation-limit provenance differs from pinned model')
    require(fixtures.get('bank_manifest_sha256') == bank.sha256, 'fixtures belong to different bank manifest')
    bank_rows = {r['layer']: r for r in bank.value['layers']}
    for row in rows:
        entry = row['entry']
        if entry.get('source_parts'):
            validate_source_parts(entry['source_parts'], entry['layer'], bank_rows, composite=entry.get('composite', False))
        else:
            require(entry['source']['sha256'] == bank_rows[entry['layer']]['artifact']['sha256'], 'fixture source differs from bank')
    rows = [row for row in rows if profile_for_layer(row['entry']['layer']) in selected]
    require({profile_for_layer(row['entry']['layer']) for row in rows} == set(selected),
            'requested profiles are absent from the reviewed fixtures')
    binary, build_sha, receipt = load_precompiled(args.precompile, args.precompile_sha256,
                                                args.serving_source, args.serving_revision, tick,
                                                torch_metadata_version=importlib.metadata.version('torch'))
    tick()
    # Import an already compiled extension only. No cpp_extension.load or CUDA API.
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    import torch
    verify_precompiled_torch(receipt, importlib.metadata.version('torch'), str(torch.__version__))
    require(not vars(torch.cuda).get('_initialized'), 'cost measurement must not initialize CUDA')
    hardware = dict(platform=dict(system=platform.system(), release=platform.release(), machine=platform.machine()),
                    cpuinfo=Path('/proc/cpuinfo').read_text(), cpus=args.cpus,
                    python=sys.version, torch=str(torch.__version__), topology=topology,
                    coordinator_topology=physical_core(args.coordinator_cpu))
    result = dict(schema='e1-lut-cost/v1', complete=False, activation=args.activation, bank_manifest_sha256=bank.sha256,
                  output_path=str(output), selected_profiles=selected,
                  measurement=dict(kind='hardware', hardware_sha256=digest(hardware), build_sha256=build_sha,
                                   placement_verified=True, clock='perf_counter_ns', samples_per_cell=args.samples,
                                   per_call=True, includes_python_boundary=True, warmup_calls_per_cell=args.warmup,
                                   seed=args.seed,act_limit=args.act_limit,activation_inputs='deterministic Gaussian CPU floats; timing only',
                                   scope='synchronous serving native TP2 local_forward fixture calls',
                                   treatment='clflush-owned-weights-before-each-call', prewake=bool(args.prewake),
                                   decode_cadence=dict(period_ms=args.cadence_ms, calls=args.cadence_calls) if args.cadence_ms else None,
                                   cache_state='owned weight arenas evicted before each timed call; no model of production cache-hit distribution',
                                   excludes=['GPU split/combine','mapped ingress handoff','FP16 ingress conversion',
                                             'asynchronous coordinator handoff','GPU synchronization','remote network'],
                                   serving_revision=args.serving_revision, precompile_sha256=args.precompile_sha256,
                                   fixture_manifest_sha256=args.fixtures_sha256, hardware=hardware, cpus=args.cpus,
                                   coordinator_cpu=args.coordinator_cpu,coordinator_affinity_before=sorted(original_affinity),nice=os.getpriority(os.PRIO_PROCESS,0),
                                   placement_scope='sampled weight/control/scratch pages; metadata locality not proven'),
                  profiles={}, real_pass=False,
                  production_trellis_baseline=dict(a_ms=0.182, b_ms=0.255,
                      label='production baseline; prior measured fit; not measured by this harness'))
    with contextlib.ExitStack() as stack:
        for name in ('init', '_lazy_init', 'is_available', 'device_count', 'set_device', 'synchronize', 'manual_seed_all'):
            stack.enter_context(mock.patch.object(torch.cuda, name, side_effect=RuntimeError('CPU cost harness attempted CUDA')))
        spec = importlib.util.spec_from_file_location('ft_tier_host', binary)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        require(callable(getattr(module,'tier_lut_fixture_flush_weights',None)),
                'fresh fixture-only owned-weight CLFLUSH API is required for admitted cost measurement')
        torch.set_num_threads(1)
        for item in rows:
            entry, descriptor = item['entry'], item['descriptor']
            layer = entry['layer']
            profile = 'upgraded_gate' if layer == 11 else 'upgraded_down' if layer in (12, 44) else 'ordinary'
            require(profile not in result['profiles'], 'one measured fixture per exact profile required')
            print(json.dumps(dict(event='COST_PROFILE_STARTED', profile=profile, layer=layer,
                                  completed_profiles=sorted(result['profiles']), activation=args.activation)), flush=True)
            module.tier_init_lut_fixture(args.cpus, args.activation, 4096, 2048, 8, False, args.act_limit)
            try:
                module.tier_lut_fixture_add_layer(descriptor)
                os.sched_setaffinity(0, {args.coordinator_cpu})
                placement = dict(module.tier_lut_fixture_stats())
                verify_target_placement(placement, output, profile, layer, 'loaded', result, expected=expected_placement)
                expected_flush_bytes=sum(placement['payload_bytes'])
                generator = torch.Generator(device='cpu').manual_seed(args.seed + layer)
                tensors = {}
                for m in range(1, 5):
                    x = torch.randn((m, 4096), generator=generator, dtype=torch.float32, device='cpu')
                    for k in range(9):
                        selected = torch.tensor([route_ids(k)] * m, dtype=torch.int32, device='cpu').reshape(m, k)
                        weights = torch.full((m, k), 1/max(1, k), dtype=torch.float32, device='cpu')
                        tensors[k, m] = x, selected, weights
                def call(k, m):
                    value = module.tier_forward(layer-3, *tensors[k, m], -1)
                    return value
                # Interleave shuffled cells to reduce deterministic sweep drift.
                cells = [(k, m) for m in range(1, 5) for k in range(9)]
                random.Random(args.seed + layer).shuffle(cells)
                measurements = {str(m): {} for m in range(1, 5)}
                for k, m in cells:
                    if args.cadence_ms:
                        prepare = lambda m=m: (flush_fixture_weights(module, expected_flush_bytes),
                                               decode_cadence(call, m, args.cadence_ms, args.cadence_calls))
                    elif args.prewake:   # workers are awake when the timed layer call arrives
                        prepare = lambda m=m: (flush_fixture_weights(module, expected_flush_bytes), call(0, m))
                    else:
                        prepare = lambda: flush_fixture_weights(module, expected_flush_bytes)
                    measurements[str(m)][str(k)] = time_cell(call, k, m, args.samples, args.warmup, tick,
                                                           prepare=prepare)
                final_stats = dict(module.tier_lut_fixture_stats())
                verify_target_placement(final_stats, output, profile, layer, 'measured', result, expected=expected_placement)
                result['profiles'][profile] = dict(layer=layer, types=PROFILES[profile][1],
                    fixture_sha256=entry['fixture']['sha256'], source_artifact_sha256=entry['source']['sha256'],
                    source_parts=entry.get('source_parts'), composite=entry.get('composite', False),
                    flushed_payload_bytes=expected_flush_bytes,rows=measurements,placement=final_stats,
                    fits={m: fit_cells(cells) for m, cells in measurements.items()})
                print(json.dumps(dict(event='COST_PROFILE_COMPLETE', profile=profile, layer=layer,
                                      cells=36, samples_per_cell=args.samples, activation=args.activation)), flush=True)
            finally:
                os.sched_setaffinity(0,original_affinity)
                module.tier_lut_fixture_reset()
    require(not vars(torch.cuda).get('_initialized'), 'CUDA unexpectedly initialized')
    tick()
    result.update(complete=set(result['profiles']) == set(PROFILES))
    if result['complete']:
        public.validate_cost_table(result, bank, args.activation)
    result['missing_profiles'] = sorted(set(PROFILES)-set(result['profiles']))
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return dict(path=str(output), sha256=sha256_file(output, tick), complete=result['complete'],
                profiles=list(result['profiles']), missing_profiles=result['missing_profiles'], real_pass=False)


def profile_for_layer(layer):
    return 'upgraded_gate' if layer == 11 else 'upgraded_down' if layer in (12, 44) else 'ordinary'


def selected_profiles(raw):
    values = list(PROFILES) if raw is None else raw.split(',')
    require(values and len(values) == len(set(values)) and set(values) <= set(PROFILES),
            'profiles must be a nonempty unique comma-separated subset of the three recipes')
    return [name for name in PROFILES if name in values]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('extract', 'measure', 'inspect-fixtures'))
    parser.add_argument('--profiles', help='comma-separated subset; default all; partial tables cannot admit serving')
    parser.add_argument('--serving-source', type=Path, default=Path(__file__).absolute().parents[2])
    parser.add_argument('--serving-revision', help='reviewed full serving checkout commit')
    parser.add_argument('--bank-manifest', type=Path, default=Path('./e1-work/bank/e1-lut-bank.json'))
    parser.add_argument('--bank-manifest-sha256')
    parser.add_argument('--layers', type=lambda s: [int(x) for x in s.split(',')], default=[3, 11, 12])
    parser.add_argument('--fixtures', type=Path, default=Path('./e1-work/cost-fixtures/fixture-manifest.json'))
    parser.add_argument('--fixtures-sha256')
    parser.add_argument('--precompile', type=Path, default=Path('./e1-work/precompiled/lut/reviewed-precompile.json'))
    parser.add_argument('--precompile-sha256')
    parser.add_argument('--cpus', type=parse_cpus, default=list(range(2, 44)),
                        help='42 distinct physical workers, comma-separated IDs/ranges (default: 2-43)')
    parser.add_argument('--coordinator-cpu', type=int, default=0, help='separate physical coordinator core (default: 0)')
    parser.add_argument('--activation', choices=('i16', 'q8'), default='i16')
    parser.add_argument('--samples', type=int, default=21)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--seed', type=int, default=6100311)
    parser.add_argument('--act-limit', type=float, default=10.0)
    parser.add_argument('--prewake', action='store_true', help='after each flush, issue one untimed K=0 wakeup')
    parser.add_argument('--cadence-ms', type=float, default=0.0, help='after each flush, untimed K=0 decode cadence')
    parser.add_argument('--cadence-calls', type=int, default=8)
    parser.add_argument('--nice', type=int, default=0, help='increase niceness by 0..19; timing records resulting value')
    parser.add_argument('--min-free-disk-gib', type=float, default=1.0, help='free disk reserve after planned output')
    parser.add_argument('--out', type=Path, help='fresh output directory for extract or JSON file for measure')
    args = parser.parse_args(argv)
    require(not (args.prewake and args.cadence_ms), 'choose one wake treatment: --prewake or --cadence-ms')
    require(args.cadence_ms == 0 or (0 < args.cadence_ms <= 20 and 1 <= args.cadence_calls <= 64), 'decode cadence out of bounds')
    require(math.isfinite(args.min_free_disk_gib) and args.min_free_disk_gib >= 0, 'free disk reserve must be nonnegative')
    require(0 <= args.nice <= 19, '--nice must be 0..19')
    if args.action == 'inspect-fixtures':
        value, descriptors = validate_fixture_manifest(args.fixtures, args.fixtures_sha256)
        print(json.dumps(dict(experts=value['experts'], layers=[d['descriptor']['model_layer'] for d in descriptors],
                              real_pass=False)))
        return
    require(args.serving_revision is not None and args.bank_manifest_sha256 is not None,
            'write actions require pinned serving revision and bank manifest SHA256')
    if args.out is None:
        args.out = Path('./e1-work/cost-fixtures' if args.action == 'extract' else './e1-work/lut-cost.json')
    require(args.layers and len(set(args.layers)) == len(args.layers) and set(args.layers) <= {3, 11, 12, 44}, 'invalid fixture layers')
    require(args.act_limit == 10.0, 'fixture/model activation limit must match the pinned model value 10')
    if args.action == 'measure':
        require(args.fixtures_sha256 is not None and args.precompile_sha256 is not None,
                'measurement requires fixture and precompile receipt SHA256')
    if args.nice:
        os.nice(args.nice)
    print(json.dumps(extract(args) if args.action == 'extract' else measure(args)))


if __name__ == '__main__':
    main()
