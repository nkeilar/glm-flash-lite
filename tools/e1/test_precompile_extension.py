"""Precompile planning and receipt tests using a compiler stand-in only."""
import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest import mock

import precompile_extension as precompile


class PrecompileTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix='e1-precompile-test-')
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / 'source'
        self.kernel = self.source / 'kernels/cpu_avx2'
        self.kernel.mkdir(parents=True)
        for name in ('ft_tier_ext.cpp', 'ft_tier_cu.cu', 'ft_core.h', 'ft_lut.cpp', 'ft_lut.h'):
            (self.kernel / name).write_text('// fixture\n')
        (self.kernel / 'lut').mkdir()
        for name in ('kernels.cpp', 'kernels.h'):
            (self.kernel / 'lut' / name).write_text('// fixture\n')
        (self.source / 'kernels/k110').mkdir()
        (self.source / precompile.SPLIT_SOURCE).write_text('// fixture\n')
        (self.source / 'glm53').mkdir()
        (self.source / 'glm53/expert_cache.py').write_text('_SRC = "// EC source literal"\n')
        (self.source / 'glm53/k_hcfuse.py').write_text('_SRC = "// fusion source literal"\n')
        self.revision = 'a'*40
        self.toolkit = self.root / 'cuda-toolkit'
        (self.toolkit / 'bin').mkdir(parents=True)
        (self.toolkit / 'bin/nvcc').write_text('fixture compiler; never invoked\n')

    def plan(self, **kwargs):
        options = dict(source=self.source, build_root=self.root / 'build', variant='lut',
                       revision=self.revision, cuda_home=self.toolkit)
        options.update(kwargs)
        with mock.patch.object(precompile.subprocess, 'check_output', side_effect=[self.revision+'\n', '']):
            return precompile.plan(**options)

    @contextlib.contextmanager
    def compiler(self, mutate=None, runtime='2.11.0+cu130'):
        torch = ModuleType('torch')
        torch.__version__ = runtime
        torch.cuda = SimpleNamespace(_initialized=False, _is_compiled=lambda: True,
            **{name: lambda: None for name in ('is_available', 'init', '_lazy_init', 'device_count',
                                              'set_device', 'synchronize', 'manual_seed_all')})
        utils = ModuleType('torch.utils')
        extension = ModuleType('torch.utils.cpp_extension')
        extension.CUDA_HOME = str(self.toolkit)
        calls = []
        def load(**kwargs):
            calls.append(kwargs)
            with self.assertRaisesRegex(RuntimeError, 'must not access CUDA'):
                torch.cuda.init()
            path = Path(kwargs['build_directory']) / (kwargs['name']+'.so')
            path.write_bytes(('fixture compiled module '+kwargs['name']).encode())
            if mutate:
                mutate(kwargs)
            return SimpleNamespace()
        extension.load = load
        utils.cpp_extension = extension
        torch.utils = utils
        with mock.patch.dict(sys.modules, {'torch': torch, 'torch.utils': utils,
                                          'torch.utils.cpp_extension': extension}), \
             mock.patch.object(precompile.importlib.metadata, 'version', return_value='2.11.0'), \
             mock.patch.dict(os.environ, {}):
            yield calls, torch

    def test_plan_binds_all_lut_native_sources_and_guarded_cuda_modules(self):
        spec = self.plan(jobs=3, cuda_arch='9.0')
        self.assertEqual(spec['compiler_jobs'], 3)
        self.assertEqual(spec['cuda_arch'], '9.0')
        self.assertEqual(len(spec['sources']), 3)
        self.assertEqual(spec['cuda_flags'], ['-O3', '-DFT_ENABLE_LUT=1'])
        self.assertIn('-DFT_ENABLE_LUT=1', spec['flags'])
        self.assertEqual(spec['split_cuda_flags'], spec['cuda_flags'])
        self.assertIn(precompile.SPLIT_SOURCE, spec['source_sha256'])
        self.assertIn('kernels/cpu_avx2/ft_lut.h', spec['source_sha256'])
        self.assertIn('kernels/cpu_avx2/lut/kernels.h', spec['source_sha256'])
        self.assertFalse((self.root / 'build').exists())
        (self.source / precompile.SPLIT_SOURCE).unlink()
        with self.assertRaisesRegex(ValueError, 'fast split'):
            self.plan()

    def test_capture_plan_extracts_literals_without_importing_model_modules(self):
        spec = self.plan(variant='capture')
        self.assertEqual(spec['sources'], [])
        self.assertEqual(spec['cuda_name'], 'glm53_ec')
        self.assertEqual(spec['cuda_flags'], ['-O3'])
        self.assertEqual(spec['k101_name'], 'glm53_k101')
        self.assertIn('glm53/expert_cache.py', spec['source_sha256'])
        self.assertIn('glm53/k_hcfuse.py', spec['source_sha256'])
        self.assertFalse((self.root / 'build').exists())
        (self.source / 'glm53/k_hcfuse.py').unlink()
        self.assertNotIn('k101_name', self.plan(variant='capture'))

    def test_plan_rejects_dirty_mismatched_or_invalid_revisions(self):
        for outputs in (['b'*40+'\n'], [self.revision+'\n', ' M kernels/cpu_avx2/ft_core.h\n']):
            with mock.patch.object(precompile.subprocess, 'check_output', side_effect=outputs), self.assertRaises(ValueError):
                precompile.plan(self.source, self.root / 'build', 'lut', self.revision)
        for value in ('a'*7, 'x'*40, None):
            with self.assertRaisesRegex(ValueError, 'full revision'):
                self.plan(revision=value)

    def test_build_locations_are_configurable_and_cannot_replace_sources(self):
        spec = self.plan(build_root=self.source / 'e1-work/precompiled')
        self.assertEqual(spec['build_directory'], str(self.source / 'e1-work/precompiled/lut'))
        for path in (self.source, self.root, self.source / 'kernels/build', self.source / 'glm53/build'):
            with self.assertRaisesRegex(ValueError, 'build directory|source directories'):
                self.plan(build_root=path)
        (self.root / 'alias').symlink_to(self.root / 'elsewhere')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            self.plan(build_root=self.root / 'alias')
        (self.root / 'build/lut').mkdir(parents=True)
        with self.assertRaisesRegex(ValueError, 'must be new'):
            self.plan()

    def test_compiler_budget_and_explicit_architecture_are_bounded(self):
        for jobs in (0, 129, True):
            with self.assertRaisesRegex(ValueError, 'compiler jobs'):
                self.plan(jobs=jobs)
        for arch in ('auto', '8.6;9.0', '8.6 9.0', 'native'):
            with self.assertRaisesRegex(ValueError, 'explicit CUDA architecture'):
                self.plan(cuda_arch=arch)
        self.assertEqual(self.plan(cuda_arch='8.6+PTX')['cuda_arch'], '8.6+PTX')

    def test_lut_build_produces_three_bound_modules_without_cuda_context_or_source_edits(self):
        spec = self.plan()
        original = {path: path.read_bytes() for path in self.source.rglob('*') if path.is_file()}
        with self.compiler() as (calls, torch):
            receipt = precompile.compile_plan(spec)
            self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'], '')
            self.assertEqual(os.environ['MAX_JOBS'], '2')
            self.assertEqual(os.environ['TORCH_CUDA_ARCH_LIST'], '8.6')
            self.assertFalse(torch.cuda._initialized)
        self.assertEqual([call['name'] for call in calls], ['ft_tier_host', 'ft_tier_cu', 'ft_split_fast'])
        self.assertFalse(calls[0]['with_cuda'])
        self.assertTrue(calls[1]['with_cuda'])
        self.assertEqual(set(receipt['outputs']), {'ft_tier_host.so', 'ft_tier_cu.so', 'ft_split_fast.so'})
        self.assertEqual(receipt['schema'], 'e1-precompile/v1')
        self.assertEqual(receipt['state'], 'COMPILED_NO_GPU_CONTEXT')
        self.assertFalse(receipt['cuda_initialized'])
        self.assertEqual(receipt['torch_metadata_version'], '2.11.0')
        self.assertEqual(receipt['torch_version'], '2.11.0+cu130')
        path = Path(spec['build_directory']) / 'reviewed-precompile.json'
        self.assertEqual(json.loads(path.read_text())['outputs'], receipt['outputs'])
        self.assertTrue(all(path.read_bytes() == data for path, data in original.items()))
        with self.compiler(), self.assertRaises(FileExistsError):
            precompile.compile_plan(spec)

    def test_generated_lut_receipt_is_admitted_by_existing_serving_preflight(self):
        from glm53 import lut_manifest as public
        spec = self.plan(source=Path(__file__).absolute().parents[2])
        with self.compiler(), mock.patch.object(public.subprocess, 'check_output', return_value=self.revision+'\n'):
            receipt = precompile.compile_plan(spec)
            path = Path(spec['build_directory']) / 'reviewed-precompile.json'
            admitted = public.load_precompiled(path, precompile.digest(path))
            self.assertEqual(set(admitted['modules']), set(receipt['outputs']))
            binary = Path(spec['build_directory']) / 'ft_tier_host.so'
            binary.write_bytes(b'changed module bytes')
            with self.assertRaisesRegex(ValueError, 'module SHA256 mismatch'):
                public.load_precompiled(path, precompile.digest(path))

    def test_capture_build_uses_literal_generated_sources_and_optional_fusion_module(self):
        spec = self.plan(variant='capture')
        with self.compiler() as (calls, torch):
            receipt = precompile.compile_plan(spec)
        self.assertEqual([call['name'] for call in calls], ['glm53_ec', 'glm53_k101'])
        self.assertEqual(set(receipt['outputs']), {'glm53_ec.so', 'glm53_k101.so'})
        self.assertEqual(Path(spec['cuda_source']).read_text(), '// EC source literal')
        self.assertEqual(Path(spec['k101_source']).read_text(), '// fusion source literal')
        self.assertEqual(len(receipt['generated_source_sha256']), 2)
        self.assertFalse(torch.cuda._initialized)

    def test_changed_sources_or_invalid_literal_cannot_publish_a_receipt(self):
        spec = self.plan()
        def mutate(options):
            if options['name'] == 'ft_split_fast':
                (self.kernel / 'ft_lut.cpp').write_text('// changed during compilation\n')
        with self.compiler(mutate=mutate), self.assertRaisesRegex(RuntimeError, 'source changed'):
            precompile.compile_plan(spec)
        self.assertFalse((Path(spec['build_directory']) / 'reviewed-precompile.json').exists())
        (self.source / 'glm53/expert_cache.py').write_text('_SRC = make_source()\n')
        capture = self.plan(variant='capture', build_root=self.root / 'capture-build')
        with self.compiler(), self.assertRaises(ValueError):
            precompile.compile_plan(capture)
        self.assertFalse((Path(capture['build_directory']) / 'reviewed-precompile.json').exists())

    def test_exact_torch_versions_are_recorded_and_metadata_mismatches_fail(self):
        with mock.patch.object(precompile.importlib.metadata, 'version', return_value='2.11.0'):
            versions = precompile.torch_versions(SimpleNamespace(__version__='2.11.0+cu130'))
        self.assertEqual(versions, dict(torch_version='2.11.0+cu130', torch_runtime_version='2.11.0+cu130',
                                        torch_metadata_version='2.11.0'))
        for metadata in ('2.10.0', '2.11.0+other', '2.11.0+cu130'):
            with mock.patch.object(precompile.importlib.metadata, 'version', return_value=metadata):
                with self.assertRaisesRegex(ValueError, 'metadata'):
                    precompile.torch_versions(SimpleNamespace(__version__='2.11.0+cu130'))

    def test_plan_cli_does_not_import_torch_or_compile(self):
        argv = ['--revision', self.revision, '--source', str(self.source), '--build-root', str(self.root/'build'), '--plan']
        with mock.patch.object(precompile, 'plan', return_value={'schema': 'fixture'}) as plan, \
             mock.patch.object(precompile, 'compile_plan') as compile_plan, contextlib.redirect_stdout(io.StringIO()):
            precompile.main(argv)
        compile_plan.assert_not_called()
        self.assertEqual(plan.call_args.args[2], 'lut')
        self.assertEqual(plan.call_args.kwargs['jobs'], 2)


if __name__ == '__main__':
    unittest.main()
