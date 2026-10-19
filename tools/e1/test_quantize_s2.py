"""Full conversion, restart, bank inventory and privacy-safe metadata fixtures."""
import argparse
import contextlib
import fcntl
import io
import json
import os
from pathlib import Path
import struct
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import quantize_s1 as core
import quantize_s2 as s2
from gguf import read_gguf, sha256_file, tensor_bytes, write_gguf
from test_quantize_s1 import fixture, imat_fixture


class S2Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='e1-bank-fixture-')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_declared_inventory_matches_serving_contract(self):
        from glm53 import lut_manifest
        rows = s2.declared_inventory()
        self.assertEqual(len(rows), 126)
        self.assertEqual(sum(row['nbytes'] for row in rows), lut_manifest.CANONICAL_BYTES)
        for row in rows:
            part = row['name'].split('ffn_')[1].split('_')[0]
            expected = lut_manifest.tensor_layout(row['layer'], part)
            self.assertEqual({key: row[key] for key in expected if key != 'row_stride'},
                             {key: expected[key] for key in expected if key != 'row_stride'})

    def test_fetch_done_requires_exact_terminal_state(self):
        path = self.root / 'FETCH-DONE'
        for value in (b'DONE', b'DONE\n'):
            path.write_bytes(value)
            self.assertEqual(s2.fetch_done(self.root)['state'], 'DONE')
        for value in (b'', b'DONE\n\n', b'DONE-PARTIAL', b'RUNNING', b'DONE\x00'):
            path.write_bytes(value)
            with self.assertRaises(ValueError):
                s2.fetch_done(self.root)
        path.unlink()
        path.symlink_to(self.root / 'missing')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            s2.fetch_done(self.root)

    def test_receipt_is_bounded_and_symlink_refused(self):
        path = self.root / 'receipt.json'
        path.write_text('{"a": 1}')
        self.assertEqual(s2.receipt_json(path)[0], {'a': 1})
        with self.assertRaisesRegex(ValueError, 'bounded'):
            s2.receipt_json(path, 2)
        link = self.root / 'link.json'
        link.symlink_to(path)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            s2.receipt_json(link)

    def test_imatrix_uses_stable_basename_and_existing_adapter_cannot_change(self):
        source = fixture(self.root / 'weights.gguf')
        imat = imat_fixture(self.root / 'imatrix.gguf')
        tensors = [source.tensor(name) for name in core.tensor_names(3)]
        first, second = self.root / 'machine-one', self.root / 'another-directory'
        for directory in (first, second):
            directory.mkdir()
            path, _, digest = s2.prepare_imatrix(imat, tensors, directory, 3)
            command = s2.quantizer_command(Path('/build/llama-quantize'), 3,
                directory / 'q8.gguf', directory / 'out.pending', path, 2)
            self.assertEqual(command[command.index('--imatrix') + 1], 'blk.03.imatrix.dat')
            self.assertEqual(digest, sha256_file(first / path.name))
        self.assertEqual(s2.prepare_imatrix(imat, tensors, first, 3)[2], digest)
        (first / 'blk.03.imatrix.dat').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'refusing to overwrite'):
            s2.prepare_imatrix(imat, tensors, first, 3)

    def test_public_manifest_is_admitted_by_real_preflight(self):
        """Sparse files exercise production geometry without allocating payloads.

        Hashing is stubbed; this tests manifest descriptors/arithmetic and the
        real preflight schema. Other tests exercise real hashes on tiny payloads.
        """
        from glm53.lut_manifest import tensor_layout, validate_structure
        entries, parsed = [], {}
        for layer in s2.LAYERS:
            tensors = [dict(tensor_layout(layer, part)) for part in core.MATRICES]
            offset = 0
            for tensor in tensors:
                tensor['offset'] = offset
                offset += tensor['nbytes']
            path = self.root / f'blk.{layer:02d}.lut.gguf'
            with path.open('wb') as stream:
                stream.truncate(32 + offset)
            parsed[path] = SimpleNamespace(tensors=tensors, data_start=32)
            entries.append(dict(layer=layer, lut=dict(path=str(path), sha256='a' * 64, bytes=32 + offset),
                tensors=[dict(t, target_type=t['type']) for t in tensors]))
        manifest = dict(state='S2_READY', layers=entries, ledger=s2.ledger(entries))
        with mock.patch.object(s2, 'check_census', side_effect=lambda p, _: parsed[p]), \
             mock.patch.object(s2, 'sha256_file', return_value='a' * 64):
            public = s2.public_manifest(manifest, 'b' * 64)
        self.assertEqual(validate_structure(public), public)
        self.assertFalse(public['real_pass'])
        self.assertEqual(public['accepted_proxy_sha256'], s2.HISTORICAL_PROXY_SHA256)
        self.assertTrue(all('/' not in row['artifact']['path'] for row in public['layers']))
        for layer in public['layers']:
            self.assertEqual(sum(t['nbytes'] for t in layer['tensors']), layer['artifact']['bytes'] - 32)

    def test_full_fixture_resumes_after_interruption_and_preserves_final_bytes(self):
        source, output, scratch = (self.root / name for name in ('source', 'bank', 'scratch'))
        (source / 'Q8_0').mkdir(parents=True)
        imat_fixture(source / 'imatrix_unsloth.gguf_file', layers=s2.LAYERS)
        names = [name for name in core.PINNED if name.endswith('.gguf')]
        tensors = [dict(name=name, dims=[256, 256, 2], type_id=8,
                        data=b'\0' * tensor_bytes([256, 256, 2], 8))
                   for layer in s2.LAYERS for name in core.tensor_names(layer)]
        for index, name in enumerate(names):
            write_gguf(source / name, {'general.architecture': (8, 'glm5-next')}, tensors[index::8])
        pinned = {name: ((source / name).stat().st_size, sha256_file(source / name)) for name in core.PINNED}
        (source / 'EXPECTED').write_text(''.join(f'{name} {size} {digest}\n' for name, (size, digest) in pinned.items()))
        (source / 'FETCH-DONE').write_text('DONE\n')
        binary = self.root / 'quantizer'
        binary.write_text('fixture')
        build = self.root / 'build.json'
        build.write_text('{}')
        args = argparse.Namespace(source=source, output=output, scratch=scratch,
            quantizer=binary, quantizer_sha256=sha256_file(binary), build_receipt=build,
            cpus=None, threads=1, nice=0)
        ids = {value[0]: key for key, value in core.TYPES.items()}
        total = sum(tensor_bytes([256, 256, 2], ids[core.recipe(layer, part)])
                    for layer in s2.LAYERS for part in core.MATRICES)
        original_validate = core.validate_triplet
        calls, interrupted = [], False
        def fake_quantize(command, directory, layer):
            nonlocal interrupted
            calls.append(layer)
            if layer == 5 and not interrupted:
                interrupted = True
                raise RuntimeError('fixture interruption')
            q8 = read_gguf(command[-4])
            write_gguf(Path(command[-3]),
                {'general.architecture': (8, 'glm5-next'),
                 'quantize.imatrix.file': (8, command[command.index('--imatrix') + 1])},
                [dict(name=t['name'], dims=t['dims'], type_id=ids[core.recipe(layer, part)],
                      data=b'\0' * tensor_bytes(t['dims'], ids[core.recipe(layer, part)]))
                 for part, t in zip(core.MATRICES, q8.tensors)])
            return dict(wall_seconds=0.1, returncode=0)
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(core, 'PINNED', pinned))
            stack.enter_context(mock.patch.object(core, 'validate_triplet',
                side_effect=lambda t, layer: original_validate(t, layer, production=False)))
            stack.enter_context(mock.patch.object(core, 'quantizer_components', return_value={'binary': 'a' * 64}))
            stack.enter_context(mock.patch.object(core, 'validate_build_receipt'))
            stack.enter_context(mock.patch.object(s2, 'CANONICAL_BYTES', total))
            stack.enter_context(mock.patch.object(s2, 'run_quantizer', side_effect=fake_quantize))
            stack.enter_context(mock.patch.object(s2.os, 'sched_setaffinity'))
            stack.enter_context(mock.patch.object(s2.os, 'nice'))
            # Production schema is separately tested above; this fixture has tiny geometry.
            stack.enter_context(mock.patch('glm53.lut_manifest.validate_structure'))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            with self.assertRaisesRegex(RuntimeError, 'fixture interruption'):
                s2.run(args)
            self.assertFalse((scratch / 'blk.04.q8_0.gguf').exists())
            original = (output / 'blk.03.lut.gguf').read_bytes()
            self.assertEqual(s2.run(args), 0)
            public = json.loads((output / 'e1-lut-bank.json').read_text())
            self.assertEqual(len(public['layers']), 42)
            self.assertEqual(public['ledger']['completed_tensors'], 126)
            self.assertEqual((output / 'blk.03.lut.gguf').read_bytes(), original)
            self.assertEqual(calls.count(3), 1)
            self.assertEqual(calls.count(4), 1)
            self.assertEqual(calls.count(5), 2)
            s1 = json.loads((output / 's1-manifest.json').read_text())
            self.assertEqual(s1['state'], 'S1_READY')
            self.assertEqual([row['layer'] for row in s1['layers']], [3, 11])
            for row in s1['layers']:
                self.assertTrue(Path(row['q8']['path']).exists())
                self.assertEqual(row['q8']['context'], row['lut']['context'])
            before = {p.name: p.read_bytes() for p in output.iterdir()}
            call_count = len(calls)
            self.assertEqual(s2.run(args), 0)
            self.assertEqual(len(calls), call_count)
            self.assertEqual({p.name: p.read_bytes() for p in output.iterdir()}, before)
            with (output / 'blk.06.lut.gguf').open('r+b') as stream:
                stream.seek(-1, 2)
                stream.write(b'\xff')
            with self.assertRaisesRegex(ValueError, 'context/hash mismatch'):
                s2.run(args)

    def test_cleanup_requires_owned_receipt_and_exact_path(self):
        path = self.root / 'blk.04.q8_0.gguf'
        path.write_bytes(b'fixture')
        receipt = dict(path=str(path), sha256=sha256_file(path))
        core.atomic_json(path.with_suffix('.gguf.verified.json'), receipt)
        entry = dict(layer=4, q8=receipt)
        with self.assertRaisesRegex(ValueError, 'exact owned'):
            s2.cleanup_q8(dict(layer=5, q8=receipt), self.root)
        path.write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            s2.cleanup_q8(entry, self.root)
        path.write_bytes(b'fixture')
        s2.cleanup_q8(entry, self.root)
        self.assertFalse(path.exists())
        s2.cleanup_q8(entry, self.root)

    def test_changed_quantizer_or_q8_input_cannot_publish_lut(self):
        for changed in ('q8', 'components'):
            with self.subTest(changed=changed):
                root = self.root / changed
                source, output, scratch = (root / name for name in ('source', 'bank', 'scratch'))
                (source / 'Q8_0').mkdir(parents=True)
                output.mkdir()
                scratch.mkdir()
                meta = fixture(source / core.METADATA_SHARD, kind=8, extras=False)
                imat = imat_fixture(source / 'imatrix_unsloth.gguf_file')
                verified = {name: dict(sha256=sha256_file(source / name), stat=core.fingerprint(source / name))
                            for name in (core.METADATA_SHARD, 'imatrix_unsloth.gguf_file')}
                by_name = {t['name']: (meta, t) for t in meta.tensors}
                ids = {value[0]: key for key, value in core.TYPES.items()}
                def child(command, directory, layer):
                    write_gguf(Path(command[-3]), {}, [
                        dict(name=t['name'], dims=t['dims'], type_id=ids[core.recipe(layer, part)],
                             data=b'\0' * tensor_bytes(t['dims'], ids[core.recipe(layer, part)]))
                        for part, t in zip(core.MATRICES, meta.tensors)])
                    if changed == 'q8':
                        with Path(command[-4]).open('r+b') as stream:
                            stream.seek(-1, 2)
                            stream.write(b'\xff')
                    return {}
                components = {'binary': 'a' * 64}
                current = {'binary': 'b' * 64} if changed == 'components' else components
                with mock.patch.object(s2, 'run_quantizer', side_effect=child), \
                     mock.patch.object(core, 'quantizer_components', return_value=current):
                    with self.assertRaisesRegex(ValueError, 'changed'):
                        s2.convert_layer(3, by_name, meta, imat, verified, source, scratch,
                            output, root / 'quantizer', [0], 1, components, {})
                self.assertFalse((output / 'blk.03.lut.gguf').exists())

    def test_output_lock_covers_runs_with_different_scratch_paths(self):
        source, output, scratch = (self.root / name for name in ('source', 'bank', 'another-scratch'))
        source.mkdir()
        output.mkdir()
        args = argparse.Namespace(source=source, output=output, scratch=scratch)
        with (output / '.conversion.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaises(BlockingIOError):
                s2.run(args)


if __name__ == '__main__':
    unittest.main()
