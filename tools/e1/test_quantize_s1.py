"""Offline conversion fixtures; opt in to local pinned ik integration with E1_QUANTIZER."""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
from unittest import mock

from gguf import read_gguf, sha256_file, tensor_bytes, write_gguf, write_subset
from quantize_s1 import (custom_rules, imatrix_entry, dependency_paths, publish, recipe,
                         resume, tensor_names, METADATA_SHARD, PINNED, validate_expected,
                         validate_triplet, verify_source, write_imatrix, atomic_json,
                         conversion_cpus, GPU_OPTIONS, IK_REVISION, validate_build_receipt)

def f32(values):
    return struct.pack('<' + 'f' * len(values), *values)

def fixture(path, layer=3, kind=0, extras=True):
    names = tensor_names(layer)
    data = f32([((i * 97) % 511 - 255) / 512 for i in range(256 * 256 * 2)])
    tensors = [dict(name=name, dims=[256, 256, 2], type_id=kind,
                    data=data if kind == 0 else b'\0' * tensor_bytes([256, 256, 2], kind))
               for name in names]
    if extras:
        tensors += [dict(name=f'blk.{layer}.ffn_gate_shexp.weight', dims=[256, 256],
                         type_id=0, data=data[:256 * 256 * 4])]
    meta = {'general.architecture': (8, 'glm5-next'),
            'general.type': (8, 'model'), 'general.alignment': (4, 32),
            'split.count': (2, 8), 'split.no': (2, 1),
            'split.tensors.count': (5, len(tensors)),
            'glm5-next.block_count': (4, 46),
            'glm5-next.embedding_length': (4, 256),
            'glm5-next.context_length': (4, 512),
            'glm5-next.feed_forward_length': (4, 256),
            'glm5-next.attention.head_count': (4, 4),
            'glm5-next.attention.layer_norm_rms_epsilon': (6, 1e-5),
            'glm5-next.expert_count': (4, 2),
            'glm5-next.expert_used_count': (4, 1)}
    write_gguf(path, meta, tensors)
    return read_gguf(path)

def imat_fixture(path, layers=(3,), shared=False, zero=False):
    tensors = []
    for layer in layers:
        for matrix, name in zip(('gate', 'up', 'down'), tensor_names(layer)):
            # Merged gate/up is deliberately used to exercise name mapping.
            if matrix == 'gate':
                continue
            if matrix == 'up':
                name = name.replace('ffn_up_exps', 'ffn_gate_up_exps')
            nmat = 1 if shared else 2
            sums = [4.0] * 256 if shared else [4.0] * 256 + [12.0] * 256
            counts = [0.0 if zero else 2.0] if shared else [2.0, 0.0 if zero else 3.0]
            tensors += [dict(name=name + '.in_sum2', dims=[256, nmat], type_id=0, data=f32(sums)),
                        dict(name=name + '.counts', dims=[1, nmat], type_id=0, data=f32(counts))]
    write_gguf(path, {'general.type': (8, 'imatrix'),
                      'imatrix.chunk_count': (4, 2),
                      'imatrix.chunk_size': (4, 512),
                      'imatrix.datasets': (9, (8, ['fixture']))}, tensors)
    return read_gguf(path)

class S1Test(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix='e1-conversion-fixture-')
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_selected_extraction_preserves_bytes_and_removes_split(self):
        src = fixture(self.root / 'source.gguf')
        out = self.root / 'selected.gguf'
        write_subset([(src, n) for n in tensor_names(3)], out)
        selected = read_gguf(out)
        self.assertEqual(set(t['name'] for t in selected.tensors), set(tensor_names(3)))
        self.assertFalse(any(k.startswith('split.') for k in selected.metadata))
        self.assertEqual(selected.metadata['general.architecture'], 'glm5-next')
        for name in tensor_names(3):
            self.assertEqual(src.read_tensor(name), selected.read_tensor(name))
        validate_triplet(selected.tensors, 3, production=False)
        with self.assertRaises(ValueError):
            validate_triplet(src.tensors, 3, production=False)

    def test_iq4_ks_row_scale_is_counted(self):
        self.assertEqual(tensor_bytes([2048, 4096, 288], 144), (8 * 136 + 4) * 4096 * 288)
        self.assertEqual(tensor_bytes([2048, 4096, 288], 144) - tensor_bytes([2048, 4096, 288], 23), 4 * 4096 * 288)

    def test_partial_file_is_never_read_as_complete(self):
        src = fixture(self.root / 'source.gguf')
        path = src.path
        size, digest = path.stat().st_size, sha256_file(path)
        receipts = self.root / 'receipts'
        receipts.mkdir()
        verify_source(path, (size, digest), receipts)
        with path.open('r+b') as f:
            f.truncate(size - 1)
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            verify_source(path, (size, digest), receipts)
        with self.assertRaisesRegex(ValueError, 'partial GGUF'):
            read_gguf(path)

    def test_source_hash_receipt_invalidates_after_mutation(self):
        src = fixture(self.root / 'source.gguf')
        digest = sha256_file(src.path)
        receipts = self.root / 'receipts'
        receipts.mkdir()
        verify_source(src.path, (src.path.stat().st_size, digest), receipts)
        with src.path.open('r+b') as f:
            f.seek(-1, 2)
            f.write(b'\xfe')
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            verify_source(src.path, (src.path.stat().st_size, digest), receipts)

    def test_exact_recipe_no_shared_or_prefix_matches(self):
        self.assertEqual([recipe(3, m) for m in ('gate', 'up', 'down')], ['IQ3_K', 'IQ3_K', 'IQ4_KS'])
        self.assertEqual([recipe(11, m) for m in ('gate', 'up', 'down')], ['IQ4_KS', 'IQ4_KS', 'Q6_K'])
        self.assertEqual(recipe(12, 'down'), 'Q6_K')
        self.assertEqual(recipe(44, 'down'), 'Q6_K')
        rules = custom_rules([3, 11])
        import re
        for rule in rules.split(','):
            pattern, _ = rule.split('=')
            self.assertFalse(re.search(pattern, 'blk.3.ffn_gate_shexp.weight'))
            self.assertFalse(re.search(pattern, 'prefix.blk.3.ffn_gate_exps.weight'))
        with self.assertRaises(ValueError):
            recipe(45, 'gate')

    def test_per_expert_counts_not_first_count(self):
        src = fixture(self.root / 'weights.gguf')
        imat = imat_fixture(self.root / 'imatrix.gguf')
        values, mapping = imatrix_entry(imat, src.tensor(tensor_names(3)[0]))
        self.assertEqual(list(values), [2.0] * 256 + [4.0] * 256)
        self.assertIn('gate_up', mapping['source'])
        self.assertFalse(mapping['broadcast_experts'])
        output = self.root / 'imatrix.dat'
        write_imatrix(imat, [src.tensor(n) for n in tensor_names(3)], output)
        with output.open('rb') as f:
            self.assertEqual(struct.unpack('<i', f.read(4))[0], 3)
            n = struct.unpack('<i', f.read(4))[0]
            self.assertEqual(f.read(n).decode(), tensor_names(3)[0])
            self.assertEqual(struct.unpack('<ii', f.read(8)), (1, 512))
            actual = struct.unpack('<512f', f.read(2048))
            self.assertEqual(list(actual), [2.0] * 256 + [4.0] * 256)

    def test_shared_inputs_broadcast_explicitly(self):
        src = fixture(self.root / 'weights.gguf')
        imat = imat_fixture(self.root / 'imatrix.gguf', shared=True)
        values, mapping = imatrix_entry(imat, src.tensor(tensor_names(3)[0]))
        self.assertEqual(list(values), [2.0] * 512)
        self.assertTrue(mapping['broadcast_experts'])

    def test_zero_count_rejected(self):
        src = fixture(self.root / 'weights.gguf')
        imat = imat_fixture(self.root / 'imatrix.gguf', zero=True)
        with self.assertRaisesRegex(ValueError, 'uncalibrated expert'):
            imatrix_entry(imat, src.tensor(tensor_names(3)[0]))

    def test_atomic_publication_and_resume_hash(self):
        src = fixture(self.root / 'source.gguf')
        pending, final = self.root / 'out.pending', self.root / 'out.gguf'
        selected = [src.tensor(n) for n in tensor_names(3)]
        expected = [dict(t, target_type=t['type']) for t in selected]
        write_subset([(src, n) for n in tensor_names(3)], pending)
        context = {'recipe': 'fixture'}
        receipt = publish(pending, final, expected, context)
        self.assertFalse(pending.exists())
        self.assertEqual(resume(final, context)['sha256'], receipt['sha256'])
        with self.assertRaisesRegex(ValueError, 'context/hash mismatch'):
            resume(final, {'recipe': 'wrong'})
        with final.open('r+b') as f:
            f.seek(-1, 2)
            f.write(b'\xff')
        with self.assertRaisesRegex(ValueError, 'context/hash mismatch'):
            resume(final, context)

    def test_resume_recovers_receipt_from_durable_intent(self):
        src = fixture(self.root / 'source.gguf')
        pending, final = self.root / 'out.pending', self.root / 'out.gguf'
        write_subset([(src, n) for n in tensor_names(3)], pending)
        expected = [dict(src.tensor(n), target_type='F32') for n in tensor_names(3)]
        publish(pending, final, expected, {'recipe': 'fixture'})
        receipt = final.with_suffix('.gguf.verified.json')
        intent = final.with_suffix('.gguf.publishing.json')
        receipt.rename(intent)  # simulate crash after final rename, before receipt
        recovered = resume(final, {'recipe': 'fixture'})
        self.assertEqual(recovered['sha256'], sha256_file(final))
        self.assertTrue(receipt.exists())
        self.assertFalse(intent.exists())

    def test_resume_rejects_adapter_or_normalized_imatrix_change(self):
        src = fixture(self.root / 'source.gguf')
        pending, final = self.root / 'out.pending', self.root / 'out.gguf'
        write_subset([(src, n) for n in tensor_names(3)], pending)
        expected = [dict(src.tensor(n), target_type='F32') for n in tensor_names(3)]
        context = {'tools': {'adapter': 'old'}, 'normalized_imatrix_sha256': 'old'}
        publish(pending, final, expected, context)
        for changed in ({'tools': {'adapter': 'new'}, 'normalized_imatrix_sha256': 'old'},
                        {'tools': {'adapter': 'old'}, 'normalized_imatrix_sha256': 'new'}):
            with self.assertRaisesRegex(ValueError, 'context/hash mismatch'):
                resume(final, changed)

    def test_census_failure_never_publishes(self):
        src = fixture(self.root / 'source.gguf')
        pending, final = self.root / 'out.pending', self.root / 'out.gguf'
        write_subset([(src, t['name']) for t in src.tensors], pending)
        expected = [dict(src.tensor(n), target_type='F32') for n in tensor_names(3)]
        with self.assertRaisesRegex(ValueError, 'census'):
            publish(pending, final, expected, {})
        self.assertFalse(final.exists())

    def test_shared_backend_provenance_includes_split_libraries_and_rejects_gpu(self):
        text = ('libllama.so => /tmp/libllama.so (0x1)\n'
                'libggml-base.so => /tmp/libggml-base.so (0x2)\n'
                'libggml-cpu.so => /tmp/libggml-cpu.so (0x3)\n')
        self.assertEqual([p.name for p in dependency_paths(text)],
                         ['libllama.so', 'libggml-base.so', 'libggml-cpu.so'])
        for line in ('libggml-cuda.so => /tmp/libggml-cuda.so (0x4)',
                     'libcublas.so => /tmp/libcublas.so (0x4)',
                     'libvulkan.so => /tmp/libvulkan.so (0x4)'):
            with self.assertRaisesRegex(ValueError, 'GPU-linked'):
                dependency_paths(text + line)

    def test_symlink_escape_rejected_before_writing(self):
        src = fixture(self.root / 'source.gguf')
        outside = self.root / 'outside'
        outside.mkdir()
        link = self.root / 'owned'
        link.symlink_to(outside, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, 'symlink'):
            write_subset([(src, n) for n in tensor_names(3)], link / 'output.gguf')
        from quantize_s1 import atomic_json
        with self.assertRaisesRegex(ValueError, 'symlink'):
            atomic_json(link / 'receipt.json', {'valid': True})
        self.assertEqual(list(outside.iterdir()), [])

    @unittest.skipUnless(os.environ.get('E1_QUANTIZER'), 'set E1_QUANTIZER for real ik integration')
    def test_real_ik_ordinary_upgraded_and_legacy_imatrix(self):
        binary = os.environ['E1_QUANTIZER']
        for layer in (3, 11):
            with self.subTest(layer=layer):
                src = fixture(self.root / f'float{layer}.gguf', layer, extras=False)
                standalone = self.root / f'standalone{layer}.gguf'
                write_subset([(src, n) for n in tensor_names(layer)], standalone)
                q8 = self.root / f'q8.{layer}.gguf'
                proc = subprocess.run([binary, str(standalone), str(q8), 'Q8_0', '2'],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stdout.decode())
                imat = imat_fixture(self.root / f'imat{layer}.gguf', layers=(layer,))
                dat = self.root / f'imat{layer}.dat'
                write_imatrix(imat, read_gguf(q8).tensors, dat)
                output = self.root / f'quant{layer}.gguf'
                proc = subprocess.run([binary, '--allow-requantize', '--slab-size', '1', '--imatrix', str(dat),
                                       '--custom-q', custom_rules([layer]), str(q8), str(output), 'IQ3_K', '2'],
                                      stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60)
                self.assertEqual(proc.returncode, 0, proc.stdout.decode())
                self.assertNotIn(b'did not find weights', proc.stdout)
                actual = read_gguf(output)
                for matrix, name in zip(('gate', 'up', 'down'), tensor_names(layer)):
                    self.assertEqual(actual.tensor(name)['type'], recipe(layer, matrix))
                validate_triplet(actual.tensors, layer, production=False)
                # The same pinned quantizer and normalized adapter must emit
                # identical complete GGUF bytes with different worker counts.
                for threads in (1, 44):
                    threaded = self.root / f'quant{layer}-threads{threads}.gguf'
                    result = subprocess.run([binary, '--allow-requantize', '--slab-size', '1',
                        '--imatrix', str(dat), '--custom-q', custom_rules([layer]), str(q8),
                        str(threaded), 'IQ3_K', str(threads)], stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT, timeout=60)
                    self.assertEqual(result.returncode, 0, result.stdout.decode())
                    self.assertEqual(sha256_file(threaded), sha256_file(output),
                                     f'layer {layer}: thread count {threads} changed GGUF bytes')

    def test_publisher_pins_are_complete_and_expected_is_exact(self):
        self.assertEqual(len(PINNED), 9)
        self.assertEqual(PINNED[METADATA_SHARD], (9429984,
            '5dc239a59cbd2d3b9d90b647cf697b80c0f4101bfaa2280b3102156e94a24ccf'))
        self.assertEqual(PINNED['imatrix_unsloth.gguf_file'][1],
            '27d2ef6b6be0e314b373080f96322a1e4668b09b192664f3edc231c3b7c2d749')
        path = self.root / 'EXPECTED'
        text = ''.join(f'{name} {size} {sha}\n' for name, (size, sha) in PINNED.items())
        path.write_text(text)
        self.assertEqual(validate_expected(self.root)['kind'], 'canonical-publisher-lfs')
        for bad in (text + text.splitlines()[0] + '\n', text.replace('9429984', '9429983')):
            path.write_text(bad)
            with self.assertRaises(ValueError):
                validate_expected(self.root)

    def test_immutable_json_and_artifacts_never_overwrite(self):
        path = self.root / 'receipt.json'
        atomic_json(path, {'a': 1})
        with self.assertRaises(FileExistsError):
            atomic_json(path, {'a': 2})
        self.assertEqual(json.loads(path.read_text()), {'a': 1})
        atomic_json(path, {'a': 2}, replace=True)
        self.assertEqual(json.loads(path.read_text()), {'a': 2})
        src = fixture(self.root / 'source.gguf')
        pending, final = self.root / 'out.pending', self.root / 'out.gguf'
        write_subset([(src, n) for n in tensor_names(3)], pending)
        expected = [dict(src.tensor(n), target_type='F32') for n in tensor_names(3)]
        final.write_bytes(b'previous publication')
        with self.assertRaises(FileExistsError):
            publish(pending, final, expected, {})
        self.assertEqual(final.read_bytes(), b'previous publication')
        self.assertTrue(pending.exists())

    def test_json_symlink_and_source_symlink_refused(self):
        src = fixture(self.root / 'source.gguf')
        link = self.root / 'link.gguf'
        link.symlink_to(src.path)
        receipts = self.root / 'receipts'
        receipts.mkdir()
        with self.assertRaisesRegex(ValueError, 'symlink'):
            verify_source(link, (src.path.stat().st_size, sha256_file(src.path)), receipts)

    def test_cpu_selection_is_portable_and_validates_affinity(self):
        self.assertEqual(conversion_cpus('2-3,6', 2, {2, 3, 6}), ([2, 3, 6], 2))
        self.assertEqual(conversion_cpus(None, None, {5, 8}), ([5, 8], 2))
        for value, threads in (('0-3', 2), ('3-2', 1), ('2,6', 3), ('2,,3', 1), ('', 1)):
            with self.assertRaises(ValueError):
                conversion_cpus(value, threads, {2, 3, 6})

    def test_build_receipt_pins_source_backends_and_linked_components(self):
        components = {'/build/bin/llama-quantize': 'a' * 64, '/build/libggml.so': 'b' * 64}
        receipt = dict(schema='e1-quantizer-build/v1', source_revision=IK_REVISION,
            tracked_source_clean=True, cmake_cache_sha256='c' * 64,
            disabled_backends={key: 'OFF' for key in GPU_OPTIONS},
            components={Path(path).name: sha for path, sha in components.items()})
        validate_build_receipt(receipt, components)
        for field, value in (('source_revision', '0' * 40), ('tracked_source_clean', False),
                             ('components', {'llama-quantize': 'd' * 64})):
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, 'build receipt mismatch'):
                validate_build_receipt(dict(receipt, **{field: value}), components)


if __name__ == '__main__':
    unittest.main()
