"""Offline cost-contract fixtures; no model, GPU or native library is loaded."""
import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import lut_cost as cost


class CostHarnessTests(unittest.TestCase):
    def test_profile_selection_defaults_all_and_rejects_duplicate_or_unknown(self):
        self.assertEqual(cost.selected_profiles(None),list(cost.PROFILES))
        self.assertEqual(cost.selected_profiles('upgraded_down,ordinary'),['ordinary','upgraded_down'])
        for value in ('','ordinary,ordinary','unknown',' ordinary'):
            with self.assertRaisesRegex(ValueError,'profiles'):cost.selected_profiles(value)
        self.assertEqual(cost.profile_for_layer(44),'upgraded_down')

    def test_k8_is_eight_distinct_experts_and_zero_is_empty(self):
        self.assertEqual(cost.route_ids(0), [])
        self.assertEqual(cost.route_ids(8), list(range(8)))
        for k, experts in ((9, 8), (8, 2), (2, 2), (-1, 8), (True, 8)):
            with self.assertRaises(ValueError):
                cost.route_ids(k, experts)

    def test_existing_two_expert_receipt_is_rejected_before_payload_reads(self):
        manifest = dict(schema='e1-real-lut-fixtures/v1', state='FIXTURES_READY', experts=[0, 1], layers=[])
        with mock.patch.object(cost, 'stable_json', return_value=(manifest, 'a'*64)), \
             mock.patch.object(cost, 'verify_fixture') as verify:
            with self.assertRaisesRegex(ValueError, 'eight distinct'):
                cost.validate_fixture_manifest('/tmp/unused', 'a'*64)
        verify.assert_not_called()

    def test_per_call_median_uses_individual_samples_and_excludes_warmup(self):
        calls, ticks = [], []
        times = iter([0, 1000000, 2000000, 7000000, 8000000, 11000000])
        measured = cost.time_cell(lambda k, m: calls.append((k, m)), 8, 4, 3, 2,
                                  lambda: ticks.append(True), clock=lambda: next(times))
        self.assertEqual(calls, [(8, 4)]*5)
        self.assertEqual(len(ticks), 5)
        self.assertEqual(measured, dict(median_ms=3.0, samples=3, raw_ms=[1.0, 5.0, 3.0]))

    def test_owned_weight_flush_follows_warmup_and_is_outside_each_timer(self):
        events=[];clock=[0]
        def call(k,m): events.append('call');clock[0]+=1000000
        def flush(): events.append('flush');clock[0]+=1000000000
        def now(): events.append('clock');return clock[0]
        value=cost.time_cell(call,8,4,3,1,lambda:events.append('guard'),clock=now,prepare=flush)
        self.assertEqual(value['raw_ms'],[1.0,1.0,1.0])
        self.assertEqual(events,['guard','call']+['guard','flush','clock','call','clock']*3)

    def test_flush_failure_cannot_publish_or_time_a_call(self):
        call=mock.Mock();clock=mock.Mock()
        with self.assertRaisesRegex(RuntimeError,'flush failed'):
            cost.time_cell(call,1,1,3,0,lambda:None,clock=clock,
                           prepare=mock.Mock(side_effect=RuntimeError('flush failed')))
        call.assert_not_called();clock.assert_not_called()

    def test_flush_byte_receipt_must_cover_the_entire_loaded_owned_payload(self):
        module=mock.Mock();module.tier_lut_fixture_flush_weights.return_value=4096
        cost.flush_fixture_weights(module,4096)
        for returned in (4095,0,None,True):
            module.tier_lut_fixture_flush_weights.return_value=returned
            with self.assertRaisesRegex(ValueError,'flush byte receipt'):
                cost.flush_fixture_weights(module,4096)

    def test_compiled_artifact_hash_and_source_revision_are_bound(self):
        with tempfile.TemporaryDirectory(prefix='e1-cost-test-') as temporary, mock.patch.object(cost.importlib.metadata,'version',return_value='unit'):
            root = Path(temporary)
            source = root/'source';source.mkdir()
            kernel = source/'kernel.cpp';kernel.write_text('// measured source\n')
            binary = root/'ft_tier_host.so';binary.write_bytes(b'precompiled test bytes; never loaded')
            receipt_path = root/'reviewed-precompile.json'
            receipt = dict(schema='e1-precompile/v1', state='COMPILED_NO_GPU_CONTEXT', variant='lut', source_revision='a'*40,
                           cuda_initialized=False, torch_version='unit+build', build_directory=str(root), source_sha256={'kernel.cpp':cost.sha256_file(kernel)},
                           outputs={'ft_tier_host.so':cost.sha256_file(binary)})
            receipt_path.write_text(json.dumps(receipt))
            approved = cost.sha256_file(receipt_path)
            actual, digest, _ = cost.load_precompiled(receipt_path, approved, source, 'a'*40)
            self.assertEqual(actual, binary)
            self.assertEqual(digest, receipt['outputs'][binary.name])
            with self.assertRaisesRegex(ValueError,'Torch metadata'):
                cost.load_precompiled(receipt_path,approved,source,'a'*40,torch_metadata_version='unit+build')
            cost.load_precompiled(receipt_path,approved,source,'a'*40,torch_metadata_version='unit')
            with self.assertRaisesRegex(ValueError, 'fresh reviewed'):
                cost.load_precompiled(receipt_path, approved, source, 'b'*40)
            binary.write_bytes(b'tampered')
            with self.assertRaisesRegex(ValueError, 'artifact SHA'):
                cost.load_precompiled(receipt_path, approved, source, 'a'*40)
            kernel.write_text('// changed source\n')
            with self.assertRaisesRegex(ValueError, 'source changed'):
                cost.load_precompiled(receipt_path, approved, source, 'a'*40)

    def test_precompiled_torch_checks_metadata_before_import_and_exact_runtime_after(self):
        receipt=dict(torch_version='2.11.0+cu130',torch_metadata_version='untrusted-observation')
        cost.verify_precompiled_torch(receipt,'2.11.0')
        cost.verify_precompiled_torch(receipt,'2.11.0','2.11.0+cu130')
        for metadata,runtime in (('2.10.0',None),('2.11.0+cu130',None),('2.11.0+other',None),
                                 ('2.11.0','2.11.0'),('2.11.0','2.11.0+other')):
            with self.subTest(metadata=metadata,runtime=runtime),self.assertRaises(ValueError):
                cost.verify_precompiled_torch(receipt,metadata,runtime)

    def test_range_copy_is_exact_and_source_is_preserved(self):
        with tempfile.TemporaryDirectory(prefix='e1-cost-test-') as temporary:
            root = Path(temporary);source=root/'source';target=root/'range'
            source.write_bytes(bytes(range(256))*8)
            expected=source.read_bytes()[31:121]
            measured=cost.copy_range(source,31,90,target,lambda:None)
            self.assertEqual(target.read_bytes(),expected)
            import hashlib
            self.assertEqual(measured,hashlib.sha256(expected).hexdigest())
            with self.assertRaises(FileExistsError):
                cost.copy_range(source,31,90,target,lambda:None)

    def test_fits_and_legacy_grid_preserve_exact_medians(self):
        cells = {str(k): dict(median_ms=.182 + .255*k) for k in range(9)}
        fit = cost.fit_cells(cells)
        self.assertAlmostEqual(fit['a_ms'], .182)
        self.assertAlmostEqual(fit['b_ms'], .255)
        self.assertAlmostEqual(fit['r_squared'], 1.)
        self.assertLess(fit['rmse_ms'], 1e-15)
        self.assertEqual(len(fit['glm53_ct_lut'].split(',')), 8)
        self.assertAlmostEqual(float(fit['glm53_ct_lut'].split(',')[0]), .437)
        with self.assertRaisesRegex(ValueError, 'complete'):
            cost.fit_cells({'0': dict(median_ms=0)})

    def test_cadence_spaces_untimed_k0_calls_and_cells_record_prepare_time(self):
        import lut_cost
        ticks=iter(range(0,10**10,250_000))   # 0.25 ms per clock read
        calls=[]
        lut_cost.decode_cadence(lambda k,m:calls.append((k,m)),2,1.0,3,clock=lambda:next(ticks))
        self.assertEqual(calls,[(0,2)]*3)
        clock=iter(range(0,10**10,1_000_000))
        prep=iter(range(0,10**10,1_000_000))
        cell=lut_cost.time_cell(lambda k,m:None,3,1,3,0,lambda:None,clock=lambda:next(clock),prepare=lambda:None,prepare_clock=lambda:next(prep))
        self.assertEqual(cell['samples'],3);self.assertEqual(cell['prepare_median_ms'],1.0);self.assertEqual(cell['median_ms'],1.0)

    def test_output_paths_allow_user_locations_but_reject_symlinks_and_traversal(self):
        self.assertEqual(cost.owned_output('e1-work/output'), Path('e1-work/output').absolute())
        with self.assertRaises(ValueError):
            cost.owned_output('e1-work/../output')
        with tempfile.TemporaryDirectory(prefix='e1-cost-test-') as temporary:
            root = Path(temporary)
            (root / 'alias').symlink_to(root)
            with self.assertRaisesRegex(ValueError, 'symlink'):
                cost.owned_output(root / 'alias/output')

    def test_publication_is_atomic_and_never_replaces_a_published_artifact(self):
        with tempfile.TemporaryDirectory(prefix='e1-cost-test-') as temporary:
            root = Path(temporary)
            pending, final = root / 'pending', root / 'published'
            pending.write_bytes(b'first')
            cost.publish_file(pending, final)
            self.assertEqual(final.read_bytes(), b'first')
            self.assertFalse(pending.exists())
            pending.write_bytes(b'second')
            with self.assertRaises(FileExistsError):
                cost.publish_file(pending, final)
            self.assertEqual(final.read_bytes(), b'first')
            self.assertEqual(pending.read_bytes(), b'second')

    def test_cli_requires_pinned_inputs_and_preserves_explicit_activation(self):
        with mock.patch.object(cost, 'extract') as extract:
            with self.assertRaisesRegex(ValueError, 'pinned serving revision'):
                cost.main(['extract'])
        extract.assert_not_called()
        for activation in ('i16', 'q8'):
            argv = ['measure', '--serving-revision', 'a'*40, '--bank-manifest-sha256', 'b'*64,
                    '--fixtures-sha256', 'c'*64, '--precompile-sha256', 'd'*64]
            if activation == 'q8':
                argv += ['--activation', activation]
            with mock.patch.object(cost, 'measure', return_value={}) as measure, contextlib.redirect_stdout(io.StringIO()):
                cost.main(argv)
            args = measure.call_args.args[0]
            self.assertEqual(args.activation, activation)
            self.assertEqual(args.cpus, list(range(2, 44)))
            self.assertEqual(args.coordinator_cpu, 0)
            self.assertEqual(args.layers, [3, 11, 12])
        self.assertEqual(cost.parse_cpus('2-21,30,32-34'), list(range(2, 22))+[30, 32, 33, 34])
        for value in ('1-', '-1', '4-2', '0-1024'):
            with self.assertRaises(argparse.ArgumentTypeError):
                cost.parse_cpus(value)

    def test_topology_accepts_custom_two_node_geometry_and_rejects_smt(self):
        cpus = list(range(100, 142))
        def physical(cpu):
            return dict(package=str(cpu // 121), core=str(cpu))
        def nodes(path, pattern):
            cpu = int(path.name[3:])
            return [Path('node4' if cpu < 121 else 'node7')]
        with mock.patch.object(cost, 'physical_core', side_effect=physical), mock.patch.object(Path, 'glob', nodes):
            topology, expected = cost.cpu_selection(cpus, 99)
        self.assertEqual(len(topology), 42)
        self.assertEqual(expected, dict(nodes=[4, 7], workers=[21, 21]))
        with mock.patch.object(cost, 'physical_core', return_value=dict(package='0', core='0')):
            with self.assertRaisesRegex(ValueError, 'distinct physical'):
                cost.cpu_selection(cpus, 99)
        with self.assertRaisesRegex(ValueError, '42 distinct'):
            cost.cpu_selection(cpus[:40], 99)
        with self.assertRaisesRegex(ValueError, 'coordinator'):
            cost.cpu_selection(cpus, 100)

    def test_placement_failure_retains_diagnostics_and_cannot_publish_costs(self):
        expected = dict(workers=[21, 21], nodes=[4, 7])
        good = dict(placement_verified=True, virtual_tp2=False, timed_out_jobs=0, **expected)
        with tempfile.TemporaryDirectory(prefix='e1-cost-test-') as temporary:
            out = Path(temporary) / 'cost.json'
            with contextlib.redirect_stdout(io.StringIO()):
                cost.verify_target_placement(good, out, 'ordinary', 3, 'loaded', {}, expected=expected)
                with self.assertRaisesRegex(ValueError, 'diagnostic='):
                    cost.verify_target_placement(dict(good, timed_out_jobs=1), out, 'ordinary', 3,
                                                 'measured', {}, expected=expected)
            self.assertFalse(out.exists())
            failure = json.loads(Path(str(out)+'.placement-failed.json').read_text())
            self.assertFalse(failure['complete'])
            self.assertFalse(failure['real_pass'])
            self.assertEqual(failure['placement']['timed_out_jobs'], 1)

    def test_fixture_hash_payload_recipe_and_source_part_provenance_are_checked(self):
        with tempfile.TemporaryDirectory(prefix='e1-cost-test-') as temporary:
            path = Path(temporary) / 'fixture.gguf'
            path.write_bytes(b'header00' + b'gate' + b'up00' + b'down')
            tensors, ranges, parts = [], [], []
            for index, (part, typ) in enumerate(zip(cost.PARTS, cost.PROFILES['ordinary'][1])):
                name = f'blk.3.ffn_{part}_exps.weight'
                dims = [2048, 4096, 8] if part == 'down' else [4096, 2048, 8]
                tensors.append(dict(name=name, dims=dims, type_id=typ, offset=4*index, nbytes=4))
                proof = dict(part=part, layer=3, type_id=typ, sha256='a'*64)
                parts.append(proof)
                ranges.append(dict(name=name, dims=dims, type_id=typ, bytes=4, experts=list(range(8)),
                                   sha256=hashlib.sha256(path.read_bytes()[8+4*index:12+4*index]).hexdigest(), source=proof))
            entry = dict(layer=3, fixture=dict(path=str(path), bytes=path.stat().st_size, sha256=cost.sha256_file(path)),
                         ranges=ranges, source_parts=parts, composite=False)
            parsed = SimpleNamespace(metadata={'general.architecture': 'glm5-next'}, tensors=tensors,
                                     tensor=lambda name: next(row for row in tensors if row['name'] == name), data_start=8)
            with mock.patch.object(cost, 'read_gguf', return_value=parsed):
                descriptor, types = cost.verify_fixture(entry)
                self.assertEqual(types, [138, 138, 144])
                self.assertEqual(descriptor['tensors'][2]['offset'], 16)
                ranges[0]['sha256'] = '0'*64
                with self.assertRaisesRegex(ValueError, 'payload range SHA'):
                    cost.verify_fixture(entry)
                ranges[0]['sha256'] = hashlib.sha256(b'gate').hexdigest()
                path.write_bytes(path.read_bytes()+b'changed')
                with self.assertRaisesRegex(ValueError, 'SHA/size'):
                    cost.verify_fixture(entry)
            bank_rows = {3: dict(artifact=dict(sha256='a'*64))}
            cost.validate_source_parts(parts, 3, bank_rows)
            altered = json.loads(json.dumps(parts))
            altered[0]['sha256'] = 'b'*64
            with self.assertRaisesRegex(ValueError, 'provenance'):
                cost.validate_source_parts(altered, 3, bank_rows)

    def test_complete_harness_table_is_admitted_by_existing_preflight(self):
        """Exercise the whole Python sweep using a non-native recording fixture."""
        public = cost.load_public_tools(Path(__file__).absolute().parents[2])
        bank_rows, fixture_rows = [], []
        for layer, types in cost.PROFILES.values():
            sha = str(layer % 10)*64
            bank_rows.append(dict(layer=layer, artifact=dict(sha256=sha)))
            parts = [dict(part=part, layer=layer, type_id=typ, sha256=sha) for part, typ in zip(cost.PARTS, types)]
            entry = dict(layer=layer, fixture=dict(sha256='d'*64), source=dict(sha256=sha), source_parts=parts)
            fixture_rows.append(dict(entry=entry, descriptor={'model_layer': layer}))
        bank = SimpleNamespace(sha256='b'*64, value=dict(layers=bank_rows))
        topology = {str(cpu): dict(package=str(cpu // 22), core=str(cpu)) for cpu in range(2, 44)}
        placement = dict(placement_verified=True, virtual_tp2=False, workers=[20, 22], nodes=[0, 1],
                         timed_out_jobs=0, payload_bytes=[1024, 1024])
        native = SimpleNamespace(tier_init_lut_fixture=mock.Mock(), tier_lut_fixture_add_layer=mock.Mock(),
                                 tier_lut_fixture_stats=lambda: placement, tier_lut_fixture_reset=mock.Mock(),
                                 tier_lut_fixture_flush_weights=mock.Mock(return_value=2048), tier_forward=mock.Mock())
        class Tensor:
            def reshape(self, *shape):
                return self
        cuda = SimpleNamespace(_initialized=False, **{name: lambda: None for name in
            ('init', '_lazy_init', 'is_available', 'device_count', 'set_device', 'synchronize', 'manual_seed_all')})
        torch = SimpleNamespace(__version__='unit+cpu', cuda=cuda, set_num_threads=lambda value: None,
            Generator=lambda **kw: SimpleNamespace(manual_seed=lambda seed: None), float32='float32', int32='int32',
            randn=lambda *a, **kw: Tensor(), tensor=lambda *a, **kw: Tensor(), full=lambda *a, **kw: Tensor())
        loader = SimpleNamespace(create_module=lambda spec: native, exec_module=lambda module: None)
        with tempfile.TemporaryDirectory(prefix='e1-cost-test-') as temporary:
            args = argparse.Namespace(samples=3, warmup=1, activation='i16', act_limit=10., cpus=list(range(2, 44)),
                coordinator_cpu=0, out=Path(temporary)/'cost.json', min_free_disk_gib=0, profiles=None,
                fixtures=Path(temporary)/'fixture.json', fixtures_sha256='c'*64,
                precompile=Path(temporary)/'precompile.json', precompile_sha256='e'*64,
                serving_source=Path(__file__).absolute().parents[2], serving_revision='a'*40,
                seed=7, prewake=False, cadence_ms=0., cadence_calls=8)
            with contextlib.ExitStack() as stack:
                stack.enter_context(mock.patch.object(cost, 'cpu_selection', return_value=(topology, dict(nodes=[0,1], workers=[20,22]))))
                stack.enter_context(mock.patch.object(cost, 'physical_core', return_value=dict(package='0', core='0')))
                stack.enter_context(mock.patch.object(cost.os, 'sched_getaffinity', return_value=set(range(44))))
                stack.enter_context(mock.patch.object(cost.os, 'sched_setaffinity'))
                stack.enter_context(mock.patch.object(cost, 'verify_sources', return_value=(public, bank, Path(temporary))))
                stack.enter_context(mock.patch.object(cost, 'validate_fixture_manifest', return_value=(dict(act_limit=10., bank_manifest_sha256=bank.sha256), fixture_rows)))
                stack.enter_context(mock.patch.object(cost, 'load_precompiled', return_value=(Path(temporary)/'unused.so', 'f'*64, dict(torch_version='unit+cpu'))))
                stack.enter_context(mock.patch.object(cost.importlib.metadata, 'version', return_value='unit'))
                stack.enter_context(mock.patch.dict(sys.modules, torch=torch))
                stack.enter_context(mock.patch.object(cost.importlib.util, 'module_from_spec', return_value=native))
                stack.enter_context(mock.patch.object(cost.importlib.util, 'spec_from_file_location', return_value=SimpleNamespace(loader=loader)))
                stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                result = cost.measure(args)
            table = json.loads(args.out.read_text())
            self.assertTrue(result['complete'])
            self.assertIs(public.validate_cost_table(table, bank, 'i16'), table)
            self.assertEqual(sum(len(cells) for profile in table['profiles'].values() for cells in profile['rows'].values()), 108)
            self.assertEqual(native.tier_lut_fixture_flush_weights.call_count, 108*3)
            self.assertEqual(native.tier_forward.call_count, 108*4)
            self.assertFalse(table['real_pass'])
            with self.assertRaisesRegex(ValueError, 'complete measured'):
                public.validate_cost_table(dict(table, complete=False), bank, 'i16')
            table['profiles']['ordinary']['rows']['1']['8']['median_ms'] += 1
            with self.assertRaisesRegex(ValueError, 'differs from raw'):
                public.validate_cost_table(table, bank, 'i16')


if __name__ == '__main__':
    unittest.main()
