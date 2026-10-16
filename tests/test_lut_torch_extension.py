"""Opt-in CPU-only PyTorch extension compilation and offline API regression.

Run with RUN_LUT_TORCH_EXTENSION_TESTS=1. Builds use at most two compiler jobs
and temporary directories. Numerical packed-weight tests are separate.
"""
import contextlib
import importlib.util
import inspect
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

ROOT = Path(os.environ.get('LUT_TORCH_ROOT', Path(__file__).resolve().parents[1]))


@unittest.skipUnless(os.environ.get('RUN_LUT_TORCH_EXTENSION_TESTS') == '1', 'explicit CPU extension build opt-in required')
class TorchExtensionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if importlib.util.find_spec('torch') is None:
            raise unittest.SkipTest('PyTorch is unavailable')
        import torch
        cls.torch = torch
        cls.guard = contextlib.ExitStack()
        cls.guard.enter_context(mock.patch.dict(os.environ, {'MAX_JOBS':'2', 'CUDA_VISIBLE_DEVICES':''}))
        cls.guard.enter_context(mock.patch.object(torch.cuda, '_is_compiled', return_value=False))
        for name in ('is_available','init','_lazy_init','device_count','set_device','synchronize','manual_seed_all'):
            cls.guard.enter_context(mock.patch.object(torch.cuda, name, side_effect=AssertionError('CPU-only extension test attempted CUDA')))
        cls.addClassCleanup(cls.guard.close)
        from torch.utils.cpp_extension import load
        cls.temporary = tempfile.TemporaryDirectory(prefix='lut-torch-extension-')
        cls.addClassCleanup(cls.temporary.cleanup)
        kernel = ROOT / 'kernels' / 'cpu_avx2'
        cls.modules = {}
        for mode in ('default','lut','lut_exact'):
            directory = Path(cls.temporary.name) / mode
            directory.mkdir()
            sources = [kernel/'ft_tier_ext.cpp']
            flags = ['-O3','-mavx2','-mfma','-mf16c','-mtune=znver3','-std=c++17',
                     '-DFT_OPT_BSUM_SHUF=1','-DFT_OPT_FLOAT_EXTRACT=1','-I'+str(kernel)]
            if mode != 'default':
                sources += [kernel/'ft_lut.cpp',kernel/'lut/kernels.cpp']
                flags += ['-DFT_ENABLE_LUT=1']
            if mode == 'lut_exact':
                flags += ['-ffp-contract=off']
            options = dict(name='lut_torch_test_'+mode, sources=[str(p) for p in sources],
                           build_directory=str(directory), extra_cflags=flags, extra_ldflags=['-lpthread'],
                           with_cuda=False, verbose=False)
            if 'with_sycl' in inspect.signature(load).parameters:
                options['with_sycl'] = False
            cls.modules[mode] = load(**options)
        if vars(torch.cuda).get('_initialized'):
            raise AssertionError('CPU-only extension test initialized CUDA')

    def test_default_has_original_host_api_and_no_lut_fixture_symbols(self):
        module = self.modules['default']
        self.assertFalse(hasattr(module, 'tier_init_lut_fixture'))
        self.assertEqual(module.tier_stats(), [])

    def test_explicit_lut_variants_reject_startup_and_unloaded_empty_work(self):
        selected, physical = [], set()
        for cpu in sorted(os.sched_getaffinity(0)):
            topology = Path('/sys/devices/system/cpu') / f'cpu{cpu}' / 'topology'
            key = ((topology/'physical_package_id').read_text(), (topology/'core_id').read_text())
            if key not in physical:
                physical.add(key)
                selected.append(cpu)
                if len(selected) == 2:
                    break
        if len(selected) != 2:
            self.skipTest('two distinct physical CPUs required for virtual TP2')
        torch = self.torch
        for mode in ('lut','lut_exact'):
            module = self.modules[mode]
            with self.subTest(mode=mode):
                module.tier_init_lut_fixture(selected, 'i16', 256, 512, 2, True)
                try:
                    stats = module.tier_lut_fixture_stats()
                    self.assertEqual(stats['swz_bytes'], 0)
                    self.assertIs(stats['virtual_tp2'], True)
                    self.assertIs(stats['placement_verified'], False)
                    with self.assertRaisesRegex(RuntimeError, 'startup is unavailable'):
                        module.tier_start(selected[0])
                    x = torch.zeros((1,256), dtype=torch.float32, device='cpu')
                    selection = torch.empty((1,0), dtype=torch.int32, device='cpu')
                    weights = torch.empty((1,0), dtype=torch.float32, device='cpu')
                    with self.assertRaises(RuntimeError):
                        module.tier_forward(0, x, selection, weights, -1)
                finally:
                    module.tier_lut_fixture_reset()
        self.assertFalse(vars(torch.cuda).get('_initialized'))


if __name__ == '__main__':
    unittest.main()
