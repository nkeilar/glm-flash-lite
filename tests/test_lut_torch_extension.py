"""Opt-in CPU-only PyTorch extension compilation and offline API regression.

Run with RUN_LUT_TORCH_EXTENSION_TESTS=1. Builds use at most two compiler jobs
and temporary directories. Numerical packed-weight tests are separate.
"""
import contextlib
import importlib.util
import inspect
import hashlib
import json
import os
import socket
import struct
import threading
import time
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
            ldflags = ['-lpthread']
            if mode != 'default':
                sources += [kernel/'ft_lut.cpp',kernel/'lut/kernels.cpp']
                # Match tools/e1/precompile_extension.py's production host flags;
                # the second LUT build additionally disables FP contraction.
                flags = ['-O3','-mavx2','-mfma','-mf16c','-mtune=broadwell','-fno-semantic-interposition','-flto','-std=c++17',
                         '-DFT_OPT_BSUM_SHUF=1','-DFT_OPT_FLOAT_EXTRACT=1','-DFT_OPT_EXP_EXTRACT=1',
                         '-DFT_OPT_BSUM_MADD_N=32','-DFT_OPT_CTRL_REUSE=1','-I'+str(kernel),'-DFT_ENABLE_LUT=1']
                ldflags += ['-flto=auto','-O3','-mavx2','-mfma','-mf16c','-mtune=broadwell','-fno-semantic-interposition']
            if mode == 'lut_exact':
                flags += ['-ffp-contract=off']
            options = dict(name='lut_torch_test_'+mode, sources=[str(p) for p in sources],
                           build_directory=str(directory), extra_cflags=flags, extra_ldflags=ldflags,
                           with_cuda=False, verbose=False)
            if 'with_sycl' in inspect.signature(load).parameters:
                options['with_sycl'] = False
            cls.modules[mode] = load(**options)
        if vars(torch.cuda).get('_initialized'):
            raise AssertionError('CPU-only extension test initialized CUDA')

    def test_default_has_original_host_api_and_no_lut_fixture_symbols(self):
        module = self.modules['default']
        self.assertFalse(hasattr(module, 'tier_init_lut_fixture'))
        self.assertFalse(hasattr(module, 'tier_init_lut'))
        self.assertFalse(hasattr(module, 'tier_lut_check'))
        self.assertFalse(hasattr(module, 'tier_lut_fixture_flush_weights'))
        for name in ('tier_lut_grouped_ready','tier_lut_forward_grouped','tier_lut_build_grouped_rows'):
            self.assertFalse(hasattr(module, name))
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
                for bad_limit in (0.0, -1.0, float('nan'), float('inf')):
                    with self.assertRaisesRegex(RuntimeError, 'activation limit'):
                        module.tier_init_lut_fixture(selected, 'i16', 256, 512, 2, True, bad_limit)
                module.tier_init_lut_fixture(selected, 'i16', 256, 512, 2, True, act_limit=20.0)
                try:
                    stats = module.tier_lut_fixture_stats()
                    self.assertEqual(stats['swz_bytes'], 0)
                    self.assertEqual(stats['act_limit'], 20.0)
                    self.assertEqual(len(stats['pool_naps']), 2)
                    self.assertEqual(stats['timed_out_jobs'], 0)
                    self.assertEqual(stats['scratch_mapped_bytes'], [0, 0])
                    self.assertEqual(len(stats['controls']), 2)
                    for control in stats['controls']:
                        self.assertIn('policy_mode', control)
                        self.assertIn('policy_nodes', control)
                        self.assertEqual(len(control['thread_policies']), len(control['cpus']))
                        for policy in control['thread_policies']:
                            self.assertEqual(set(policy), {'cpu','node','mode','error','verified','nodes'})
                    self.assertIs(stats['virtual_tp2'], True)
                    self.assertIs(stats['placement_verified'], False)
                    with self.assertRaisesRegex(RuntimeError, 'cannot start a production coordinator'):
                        module.tier_start(selected[0])
                    x = torch.zeros((1,256), dtype=torch.float32, device='cpu')
                    selection = torch.empty((1,0), dtype=torch.int32, device='cpu')
                    weights = torch.empty((1,0), dtype=torch.float32, device='cpu')
                    with self.assertRaises(RuntimeError):
                        module.tier_forward(0, x, selection, weights, -1)
                finally:
                    module.tier_lut_fixture_reset()
                module.tier_init_lut_fixture(selected, 'i16', 256, 512, 2, True)
                try:
                    self.assertEqual(module.tier_lut_fixture_stats()['act_limit'], 10.0)
                finally:
                    module.tier_lut_fixture_reset()
        self.assertFalse(vars(torch.cuda).get('_initialized'))

    def _cpus(self):
        result, seen = [], set()
        for cpu in sorted(os.sched_getaffinity(0)):
            base = Path('/sys/devices/system/cpu') / f'cpu{cpu}' / 'topology'
            core = ((base/'physical_package_id').read_text(), (base/'core_id').read_text())
            if core not in seen:
                seen.add(core); result.append(cpu)
                if len(result) == 2:
                    return result
        self.skipTest('two physical CPUs required')

    def _require_loopback(self):
        try:
            with socket.socket() as peer:peer.bind(('localhost',0))
        except PermissionError as error:
            self.skipTest(f'loopback sockets are forbidden in this environment: {error}')

    def _fixture(self, module, directory, *, delay=None, activation='i16'):
        torch = self.torch
        path = Path(directory) / 'packed-weights.bin'
        descriptor = dict(model_layer=3, internal_li=0, path=str(path), tensors=[])
        with path.open('wb') as output:
            output.write(bytes(64))
            for matrix in range(3):
                width, rows = (512,256) if matrix == 2 else (256,512)
                offset = output.tell()
                for expert in range(2):
                    for row in range(rows):
                        if matrix < 2:
                            packed = struct.pack('<eHH', 0.0005*(expert+1)*(matrix+1), 0,0) + bytes([row%3])*8 + bytes([0x55])*64 + bytes(32)
                        else:
                            packed = struct.pack('<f',0.00002*(expert+1)) + (bytes([128])*8 + bytes([0x88])*128) * 2
                        output.write(packed)
                descriptor['tensors'].append(dict(offset=offset,nbytes=output.tell()-offset,type_id=144 if matrix==2 else 138,
                                                 width=width,rows=rows,experts=2))
                output.write(bytes((-output.tell())%32))
        st = path.stat()
        descriptor['identity'] = dict(dev=st.st_dev,ino=st.st_ino,size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)
        module.tier_init_lut_fixture(self._cpus(),activation,256,512,2,True,20.0)
        module.tier_lut_fixture_add_layer(descriptor)
        if delay:
            module.tier_lut_fixture_fault(*delay)
        x = (torch.arange(4*256,dtype=torch.float32).reshape(4,256).sin()*.02+.03).half()
        sel = torch.tensor([[0,1],[1,0],[0,1],[1,-1]],dtype=torch.int32)
        weights = torch.tensor([[.2,.8],[.35,.65],[.7,.3],[1.,0.]],dtype=torch.float32)
        reference = module.tier_forward(0,x.float(),sel,weights,-1)
        parts = {}
        for token in range(4):
            for expert in range(2):
                parts[token,expert] = module.tier_forward(0,x[token:token+1].float(),torch.tensor([[expert]],dtype=torch.int32),torch.ones((1,1)),-1)[0].clone()
        self.assertGreater(float(reference.abs().max()),0.)
        return x,sel,weights,reference,parts

    def _start_coordinator(self,module,timeout_ms=1000,hidden=256):
        torch = self.torch
        ctrl = torch.zeros(16,dtype=torch.int64)
        hx = torch.zeros(4*hidden,dtype=torch.half)
        picks = torch.zeros(4*8*3,dtype=torch.int32)
        hout = torch.full((4*hidden,),123.,dtype=torch.float32)
        module.tier_lut_fixture_coordinator_start(self._cpus()[0],ctrl,hx,picks,hout,timeout_ms)
        return ctrl,hx,picks,hout

    def test_grouped_row_builder_orders_experts_and_preserves_weight_bits(self):
        torch=self.torch
        sel=torch.tensor([[3,0,2,1],[1,3,2,-1],[0,1,3,2]],dtype=torch.int64)
        take=torch.tensor([1,0,1,1],dtype=torch.uint8)
        expected_tokens=torch.tensor([0,2,0,1,2,0,1,2],dtype=torch.int32)
        selected_slots=torch.tensor([1,8,2,6,11,0,5,10])
        for variant in ('lut','lut_exact'):
            module=self.modules[variant]
            for dtype in (torch.float16,torch.float32):
                with self.subTest(variant=variant,dtype=dtype):
                    w=torch.tensor([[2**-24,-0.,.125,.25],[.5,65504.,-.5,0.],[1.,.3,2**-14,.75]],dtype=dtype)
                    experts,offsets,tokens,weights=module.tier_lut_build_grouped_rows(sel,w,take,4)
                    self.assertEqual(experts.tolist(),[0,2,3]);self.assertEqual(offsets.tolist(),[0,2,5,8])
                    self.assertTrue(torch.equal(tokens,expected_tokens))
                    self.assertTrue(torch.equal(weights.view(torch.int32),w.float().flatten()[selected_slots].view(torch.int32)))
                    for value in (experts,offsets,tokens,weights):
                        self.assertEqual(value.device.type,'cpu');self.assertTrue(value.is_contiguous())
                    self.assertEqual(experts.dtype,torch.int32);self.assertEqual(weights.dtype,torch.float32)
            for selection,weights in ((sel,torch.zeros_like(sel,dtype=torch.float32)),
                                       (sel[:0],torch.empty((0,4))),
                                       (sel[:,:0].contiguous(),torch.empty((3,0)))):
                result=module.tier_lut_build_grouped_rows(selection,weights,torch.zeros(4,dtype=torch.uint8),4)
                self.assertEqual([v.tolist() for v in result],[[],[0],[],[]])

    def test_grouped_row_builder_rejects_invalid_routes(self):
        torch=self.torch;module=self.modules['lut']
        sel=torch.tensor([[0,1],[1,0]],dtype=torch.int64);w=torch.ones((2,2));take=torch.ones(2,dtype=torch.uint8)
        cases=[('selection dtype',sel.int(),w,take,2),('selection shape',sel.flatten(),w,take,2),
               ('selection stride',sel.t(),w,take,2),('weight dtype',sel,w.double(),take,2),
               ('weight shape',sel,w[:1],take,2),('weight stride',sel,w.t(),take,2),
               ('mask dtype',sel,w,take.bool(),2),('mask shape',sel,w,take[:,None],2),
               ('mask count',sel,w,take,3),('mask value',sel,w,torch.tensor([1,2],dtype=torch.uint8),2),
               ('expert count',sel,w,take,0),('expert high',torch.tensor([[0,2],[1,0]]),w,take,2),
               ('expert negative',torch.tensor([[-2,1],[1,0]]),w,take,2),
               ('masked weight',torch.tensor([[-1,1],[1,0]]),w,take,2),
               ('duplicate',torch.tensor([[0,0],[1,0]]),w,take,2)]
        for value in (float('nan'),float('inf'),-float('inf')):
            bad=w.clone();bad[0,0]=value;cases.append(('nonfinite weight',sel,bad,take,2))
        for name,selection,weights,mask,E in cases:
            with self.subTest(case=name),self.assertRaises(RuntimeError):
                module.tier_lut_build_grouped_rows(selection,weights,mask,E)

    def test_grouped_matches_decode_bitwise_for_one_to_four_tokens(self):
        torch=self.torch
        for variant in ('lut','lut_exact'):
            module=self.modules[variant]
            self.assertFalse(module.tier_lut_grouped_ready())
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as directory:
                x,_,_,_,_=self._fixture(module,directory,activation='q8')
                try:
                    self.assertTrue(module.tier_lut_grouped_ready())
                    generator=torch.Generator().manual_seed(28)
                    for m in range(1,5):
                        # Full routes exercise each M; these gapped routes also
                        # exercise decode's non-adjacent M2 activation copies.
                        routes=[torch.tensor([[1,0] if t%2 else [0,1] for t in range(m)],dtype=torch.int32),
                                torch.tensor([[t%2,-1] for t in range(m)],dtype=torch.int32)]
                        for _ in range(5):
                            selection=torch.stack([torch.randperm(2,generator=generator) for _ in range(m)]).int()
                            selection[torch.rand((m,2),generator=generator)<.35]=-1;routes.append(selection)
                        for selection in routes:
                            for dtype in (torch.float16,torch.float32):
                                weights=torch.rand((m,2),generator=generator).to(dtype)
                                weights[selection<0]=0
                                rows=module.tier_lut_build_grouped_rows(selection.long(),weights,torch.ones(2,dtype=torch.uint8),2)
                                reference=module.tier_forward(0,x[:m].float(),selection,weights.float(),-1)
                                output=torch.full((m,256),123.,dtype=torch.float32)
                                ok,why,elapsed,chunks=module.tier_lut_forward_grouped(0,x[:m],*rows,output,1000.)
                                self.assertTrue(ok,why);self.assertEqual(why,'');self.assertGreaterEqual(elapsed,0.)
                                self.assertEqual(chunks,int(rows[0].numel()>0))
                                self.assertTrue(torch.equal(output.view(torch.int32),reference.view(torch.int32)),
                                                f'{variant} M={m}, sel={selection.tolist()}, dtype={dtype}')
                    output=torch.full((4,256),123.)
                    rows=module.tier_lut_build_grouped_rows(torch.empty((4,0),dtype=torch.int64),torch.empty((4,0)),torch.ones(2,dtype=torch.uint8),2)
                    ok,why,_,chunks=module.tier_lut_forward_grouped(0,x,*rows,output,1000.)
                    self.assertTrue(ok,why);self.assertEqual(chunks,0);self.assertEqual(int(torch.count_nonzero(output)),0)
                finally:module.tier_lut_fixture_reset()

    def test_grouped_rejections_leave_output_and_readiness_unchanged(self):
        torch=self.torch;module=self.modules['lut']
        with tempfile.TemporaryDirectory() as directory:
            x,sel,w,_,_=self._fixture(module,directory,activation='q8')
            try:
                rows=module.tier_lut_build_grouped_rows(sel.long(),w,torch.ones(2,dtype=torch.uint8),2)
                expert,offset,tokens,weights=rows
                cases=[]
                def reject(name,**updates):
                    values=dict(layer=0,x_f16=x,experts=expert,offsets=offset,row_token=tokens,row_weight=weights,
                                out=torch.full((4,256),123.),timeout_ms=1000.)
                    values.update(updates);cases.append((name,values))
                reject('layer',layer=41);reject('x dtype',x_f16=x.float())
                reject('x stride',x_f16=x.t().contiguous().t());reject('x shape',x_f16=x.flatten())
                reject('output dtype',out=torch.full((4,256),123.,dtype=torch.float64))
                reject('output shape',out=torch.full((1,256),123.))
                reject('output stride',out=torch.full((256,4),123.).t())
                reject('index dtype',experts=expert.long());reject('index shape',experts=expert[:,None])
                reject('weight dtype',row_weight=weights.half());reject('weight shape',row_weight=weights[:,None])
                reject('offset count',offsets=offset[:1]);reject('row count',row_token=tokens[:-1])
                reject('experts unsorted',experts=expert.flip(0));reject('expert duplicate',experts=torch.tensor([0,0],dtype=torch.int32))
                reject('expert range',experts=torch.tensor([0,2],dtype=torch.int32))
                bad=tokens.clone();bad[1]=bad[0];reject('token duplicate',row_token=bad)
                bad=tokens.clone();bad[0]=4;reject('token range',row_token=bad)
                bad=offset.clone();bad[0]=1;reject('offset start',offsets=bad)
                bad=offset.clone();bad[1]=-1;reject('offset order',offsets=bad)
                for value in (float('nan'),float('inf'),-float('inf')):
                    bad=x.clone();bad[0,0]=value;reject('nonfinite x',x_f16=bad)
                    bad=weights.clone();bad[0]=value;reject('nonfinite weight',row_weight=bad)
                for timeout in (0.,-1.,float('nan'),float('inf')):reject('timeout',timeout_ms=timeout)
                reject('too many tokens',x_f16=torch.zeros((1025,256),dtype=torch.float16),out=torch.full((1025,256),123.))
                reject('too many rows',x_f16=torch.zeros((65,256),dtype=torch.float16),experts=torch.tensor([0],dtype=torch.int32),
                       offsets=torch.tensor([0,65],dtype=torch.int32),row_token=torch.arange(65,dtype=torch.int32),
                       row_weight=torch.ones(65),out=torch.full((65,256),123.))
                for name,values in cases:
                    with self.subTest(case=name):
                        original=values['out'].clone();ok,why,_,chunks=module.tier_lut_forward_grouped(**values)
                        self.assertFalse(ok);self.assertTrue(why);self.assertEqual(chunks,0)
                        self.assertTrue(torch.equal(values['out'],original));self.assertTrue(module.tier_lut_grouped_ready())
                        self.assertEqual(module.tier_lut_stats()['failed_jobs'],0)
            finally:module.tier_lut_fixture_reset()
        with tempfile.TemporaryDirectory() as directory:
            x,sel,w,_,_=self._fixture(module,directory)
            try:
                rows=module.tier_lut_build_grouped_rows(sel.long(),w,torch.ones(2,dtype=torch.uint8),2)
                output=torch.full((4,256),123.)
                ok,why,_,_=module.tier_lut_forward_grouped(0,x,*rows,output,1000.)
                self.assertFalse(ok);self.assertIn('Q8',why.upper());self.assertTrue(bool(torch.all(output==123.)))
                self.assertFalse(module.tier_lut_grouped_ready());self.assertFalse(module.tier_lut_stats()['aborted'])
            finally:module.tier_lut_fixture_reset()

    def test_grouped_busy_during_grouped_and_decode_jobs(self):
        torch=self.torch;module=self.modules['lut']
        for decode in (False,True):
            with self.subTest(decode=decode),tempfile.TemporaryDirectory() as directory:
                x,sel,w,reference,_=self._fixture(module,directory,activation='q8')
                module.tier_lut_fixture_fault(1,1,100)
                rows=module.tier_lut_build_grouped_rows(sel.long(),w,torch.ones(2,dtype=torch.uint8),2)
                output=torch.full((4,256),123.);completed=[];errors=[]
                if decode:self._start_coordinator(module)
                def run():
                    try:
                        if decode:completed.append(module.tier_lut_fixture_job(0,x,sel,w,1,1000))
                        else:completed.append(module.tier_lut_forward_grouped(0,x,*rows,output,1000.))
                    except BaseException as error:errors.append(error)
                thread=threading.Thread(target=run);thread.start()
                try:
                    deadline=time.monotonic()+1
                    while module.tier_lut_grouped_ready() and time.monotonic()<deadline:time.sleep(.001)
                    self.assertTrue(thread.is_alive());self.assertFalse(module.tier_lut_grouped_ready())
                    other=torch.full((4,256),321.)
                    ok,why,_,chunks=module.tier_lut_forward_grouped(0,x,*rows,other,1000.)
                    self.assertFalse(ok);self.assertEqual(why,'busy');self.assertEqual(chunks,0)
                    self.assertTrue(bool(torch.all(other==321.)))
                    thread.join(2);self.assertFalse(thread.is_alive());self.assertEqual(errors,[])
                    self.assertTrue(module.tier_lut_grouped_ready());self.assertEqual(len(completed),1)
                    if decode:torch.testing.assert_close(completed[0],reference,rtol=0,atol=0)
                    else:self.assertTrue(completed[0][0],completed[0][1])
                finally:
                    thread.join(2)
                    if decode:module.tier_stop()
                    module.tier_lut_fixture_reset()

    def test_grouped_runtime_failure_is_sticky_with_and_without_ingress(self):
        torch=self.torch;module=self.modules['lut']
        for fault,ingress in (('timeout',False),('worker',False),('timeout',True)):
            with self.subTest(fault=fault,ingress=ingress),tempfile.TemporaryDirectory() as directory:
                x,sel,w,_,_=self._fixture(module,directory,activation='q8')
                module.tier_lut_fixture_fault(1,1,60 if fault=='timeout' else -1)
                rows=module.tier_lut_build_grouped_rows(sel.long(),w,torch.ones(2,dtype=torch.uint8),2)
                ctrl=None
                try:
                    if ingress:ctrl,_,_,_=self._start_coordinator(module)
                    output=torch.full((4,256),123.)
                    ok,why,_,_=module.tier_lut_forward_grouped(0,x,*rows,output,1. if fault=='timeout' else 1000.)
                    self.assertFalse(ok);self.assertTrue(why);self.assertTrue(bool(torch.all(output==123.)))
                    self.assertFalse(module.tier_lut_grouped_ready());self.assertTrue(module.tier_lut_stats()['aborted'])
                    self.assertEqual(module.tier_lut_stats()['failed_jobs'],1)
                    with self.assertRaisesRegex(RuntimeError,'aborted'):module.tier_lut_check()
                    again=module.tier_lut_forward_grouped(0,x,*rows,output,1000.)
                    self.assertFalse(again[0]);self.assertIn('aborted',again[1]);self.assertTrue(bool(torch.all(output==123.)))
                    self.assertEqual(module.tier_lut_stats()['failed_jobs'],1)
                    if ingress:self.assertGreater(int(ctrl[5]),0);self.assertGreater(int(ctrl[6]),0)
                finally:
                    if ingress:module.tier_stop()
                    module.tier_lut_fixture_reset()

    def test_cpu_ingress_dispatch_and_exactly_once_generations(self):
        torch = self.torch
        for variant in ('lut','lut_exact'):
            module=self.modules[variant]
            with self.subTest(variant=variant), tempfile.TemporaryDirectory() as directory:
                x,sel,w,reference,_ = self._fixture(module,directory)
                try:
                    ctrl,_,_,hout=self._start_coordinator(module)
                    with self.assertRaisesRegex(RuntimeError,'stopped offline LUT fixture'):
                        module.tier_lut_fixture_flush_weights()
                    for sequence in (1,2,9):
                        got=module.tier_lut_fixture_job(0,x,sel,w,sequence,1000)
                        torch.testing.assert_close(got,reference,rtol=1e-6,atol=1e-10)
                        self.assertEqual(int(ctrl[1]),sequence)
                        self.assertEqual(int(ctrl[8]),sequence)
                    with self.assertRaisesRegex(RuntimeError,'reused'):
                        module.tier_lut_fixture_job(0,x,sel,w,9,1000)
                    empty=module.tier_lut_fixture_job(0,x,sel[:,:0].contiguous(),w[:,:0].contiguous(),10,1000)
                    self.assertEqual(float(empty.abs().max()),0.)
                    stats=module.tier_lut_stats()
                    self.assertEqual(stats['completed_jobs'],4)
                    self.assertFalse(stats['production_ready'])
                    module.tier_lut_check()
                finally:
                    module.tier_stop(); module.tier_lut_fixture_reset()

    def test_cpu_ingress_failure_is_sticky_and_stop_drains(self):
        torch=self.torch
        for fault in ('duplicate','nonfinite','bad_layer','worker_failure','cpu_deadline','combine_deadline','explicit_abort'):
            module=self.modules['lut']
            with self.subTest(fault=fault),tempfile.TemporaryDirectory() as directory:
                x,sel,w,_,_=self._fixture(module,directory)
                try:
                    if fault=='worker_failure': module.tier_lut_fixture_fault(1,1,-1)
                    if fault in ('cpu_deadline','combine_deadline','explicit_abort'): module.tier_lut_fixture_fault(1,1,60)
                    ctrl,hx,picks,hout=self._start_coordinator(module,10 if fault=='cpu_deadline' else 1000)
                    original=hout.clone()
                    if fault=='duplicate': sel[0]=torch.tensor([0,0],dtype=torch.int32)
                    if fault=='nonfinite': x[0,0]=float('nan')
                    if fault=='explicit_abort':
                        def abort():
                            deadline=time.monotonic()+1
                            while int(ctrl[0])==0 and time.monotonic()<deadline: time.sleep(.001)
                            module.tier_lut_abort()
                        aborter=threading.Thread(target=abort); aborter.start()
                    with self.assertRaisesRegex(RuntimeError,'aborted'):
                        module.tier_lut_fixture_job(41 if fault=='bad_layer' else 0,x,sel,w,1,5 if fault=='combine_deadline' else 1000)
                    if fault=='explicit_abort': aborter.join(1); self.assertFalse(aborter.is_alive())
                    saved=hx.clone()
                    with self.assertRaisesRegex(RuntimeError,'aborted'):
                        module.tier_lut_fixture_job(0,x,sel,w,2,1000)
                    torch.testing.assert_close(hx,saved,equal_nan=True)
                    module.tier_stop()
                    self.assertEqual(int(ctrl[1]),0)
                    self.assertEqual(int(ctrl[8]),1)
                    self.assertGreater(int(ctrl[6]),0)
                    self.assertFalse(module.tier_lut_stats()['coordinator_active'])
                    # Backend or coordinator error never publishes successful completion.
                    if fault in ('worker_failure','cpu_deadline','nonfinite','duplicate','bad_layer'):
                        torch.testing.assert_close(hout,original)
                finally:
                    module.tier_stop(); module.tier_lut_fixture_reset()

    @unittest.skipUnless(os.environ.get('LUT_TORCH_FIXTURE_MANIFEST'), 'optional actual packed-byte fixtures required')
    def test_actual_packed_bytes_through_cpu_ingress(self):
        torch=self.torch
        manifest_path=Path(os.environ['LUT_TORCH_FIXTURE_MANIFEST'])
        manifest=json.loads(manifest_path.read_text())
        self.assertEqual(manifest['state'],'FIXTURES_READY')
        rows={row['layer']:row for row in manifest['layers']}
        self.assertEqual(set(rows),{3,11})
        descriptors=[]
        for layer,row in sorted(rows.items()):
            path=manifest_path.parent/Path(row['fixture']['path']).name
            self.assertLessEqual(path.stat().st_size,64<<20)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),row['fixture']['sha256'])
            st=path.stat()
            tensors=[]
            for part in ('gate','up','down'):
                tensor=next(t for t in row['ranges'] if t['name'].endswith('ffn_'+part+'_exps.weight'))
                tensors.append(dict(offset=tensor['fixture_offset'],nbytes=tensor['bytes'],type_id=tensor['type_id'],
                                    width=tensor['dims'][0],rows=tensor['dims'][1],experts=tensor['dims'][2]))
            descriptors.append(dict(model_layer=layer,internal_li=layer-3,path=str(path),tensors=tensors,
                                    identity=dict(dev=st.st_dev,ino=st.st_ino,size=st.st_size,mtime_ns=st.st_mtime_ns,ctime_ns=st.st_ctime_ns)))
        for variant in ('lut','lut_exact'):
            module=self.modules[variant]
            with self.subTest(variant=variant):
                module.tier_init_lut_fixture(self._cpus(),'i16',4096,2048,2,True,10.0)
                try:
                    for descriptor in descriptors:module.tier_lut_fixture_add_layer(descriptor)
                    cases=[]
                    for li in (0,8):
                        for m in range(1,5):
                            x=(torch.arange(m*4096,dtype=torch.float32).reshape(m,4096).sin()*.11).half()
                            for k in range(3):
                                sel=torch.tensor([list(range(k))[::(-1 if t%2 else 1)] for t in range(m)],dtype=torch.int32)
                                weights=torch.full((m,k),1./max(k,1),dtype=torch.float32)
                                cases.append((li,x,sel,weights,module.tier_forward(li,x.float(),sel,weights,-1)))
                    self.assertEqual(module.tier_lut_fixture_flush_weights(),sum(module.tier_lut_stats()['payload_bytes']))
                    ctrl,_,_,_=self._start_coordinator(module,hidden=4096)
                    for seq,(li,x,sel,weights,expected) in enumerate(cases,1):
                        got=module.tier_lut_fixture_job(li,x,sel,weights,seq,1000)
                        torch.testing.assert_close(got,expected,rtol=1e-6,atol=1e-7)
                    self.assertEqual(module.tier_lut_stats()['completed_jobs'],24)
                finally:
                    module.tier_stop();module.tier_lut_fixture_reset()

    def test_owned_weight_flush_preserves_values_and_serializes_reset(self):
        torch=self.torch
        for variant in ('lut','lut_exact'):
            module=self.modules[variant]
            with self.subTest(variant=variant),tempfile.TemporaryDirectory() as directory:
                x,sel,w,reference,_=self._fixture(module,directory)
                reset=False;thread=None
                try:
                    expected=sum(module.tier_lut_stats()['payload_bytes'])
                    self.assertGreater(expected,0);self.assertLessEqual(expected,256<<20)
                    for _ in range(3):
                        self.assertEqual(module.tier_lut_fixture_flush_weights(),expected)
                        torch.testing.assert_close(module.tier_forward(0,x.float(),sel,w,-1),reference,rtol=0,atol=0)
                    first=threading.Event();flushed=[];errors=[]
                    def flush():
                        try:
                            for _ in range(50):
                                flushed.append(module.tier_lut_fixture_flush_weights());first.set()
                        except RuntimeError as error:errors.append(str(error))
                        finally:first.set()
                    thread=threading.Thread(target=flush);thread.start()
                    self.assertTrue(first.wait(1))
                    self.assertTrue(flushed)
                    module.tier_lut_fixture_reset();reset=True
                    thread.join(2);self.assertFalse(thread.is_alive())
                    self.assertTrue(all(count==expected for count in flushed))
                    self.assertTrue(all('stopped offline LUT fixture' in error for error in errors))
                    with self.assertRaisesRegex(RuntimeError,'stopped offline LUT fixture'):
                        module.tier_lut_fixture_flush_weights()
                finally:
                    if thread:thread.join(2)
                    if not reset:module.tier_lut_fixture_reset()

    def test_concurrent_fixture_waiter_and_shutdown_keep_storage_owned(self):
        module=self.modules['lut']
        with tempfile.TemporaryDirectory() as directory:
            x,sel,w,_,_=self._fixture(module,directory)
            module.tier_lut_fixture_fault(1,1,60)
            ctrl,_,_,_=self._start_coordinator(module)
            errors=[]
            def run():
                try:module.tier_lut_fixture_job(0,x,sel,w,1,1000)
                except RuntimeError as error:errors.append(str(error))
            thread=threading.Thread(target=run);thread.start()
            try:
                deadline=time.monotonic()+1
                while int(ctrl[0])==0 and time.monotonic()<deadline:time.sleep(.001)
                self.assertEqual(int(ctrl[0]),1)
                with self.assertRaisesRegex(RuntimeError,'one offline ingress waiter'):
                    module.tier_lut_fixture_job(0,x,sel,w,2,1000)
                module.tier_stop()
                thread.join(1)
                self.assertFalse(thread.is_alive())
                self.assertEqual(len(errors),1)
                self.assertEqual(int(ctrl[8]),1)
            finally:
                module.tier_stop();thread.join(1);module.tier_lut_fixture_reset()

    def test_concurrent_remote_start_and_stop_drains_reconnect(self):
        self._require_loopback()
        module=self.modules['lut'];torch=self.torch
        with tempfile.TemporaryDirectory() as directory:
            _,_,_,_,parts=self._fixture(module,directory)
            server=MockRemote(torch,2,'slow_handshake',parts,256)
            errors=[]
            res=[torch.empty(0,dtype=torch.uint8) for _ in range(42)];res[0]=torch.tensor([0,1],dtype=torch.uint8)
            def enable():
                try:module.tier_remote(server.host,server.port,res,list(range(3,45)),10000)
                except BaseException as error:errors.append(error)
            start=threading.Thread(target=enable);start.start();stop=None
            try:
                self.assertTrue(server.handshake_started.wait(1))
                stop=threading.Thread(target=module.tier_stop);stop.start()
                server.handshake_release.set()
                start.join(3);stop.join(3)
                self.assertFalse(start.is_alive());self.assertFalse(stop.is_alive())
                self.assertEqual(errors,[])
                self.assertFalse(module.tier_remote_stats()[0])
            finally:
                server.handshake_release.set();start.join(3)
                if stop:stop.join(3)
                module.tier_stop();module.tier_lut_fixture_reset();server.close()

    def test_remote_success_refusal_bad_reply_and_drop_fallback(self):
        self._require_loopback()
        torch=self.torch
        for protocol,behavior in ((1,'success'),(1,'refuse'),(1,'nan'),(2,'success'),(2,'partial'),(2,'refuse'),
                                  (2,'invalid_mask'),(2,'nan'),(2,'drop'),(2,'trickle'),(2,'bad_seq'),(2,'all_partial'),(2,'remote_only'),(1,'drop'),(1,'trickle')):
            module=self.modules['lut']
            with self.subTest(protocol=protocol,behavior=behavior),tempfile.TemporaryDirectory() as directory:
                x,sel,w,reference,parts=self._fixture(module,directory)
                server=MockRemote(torch,protocol,behavior,parts,256)
                try:
                    res=[torch.empty(0,dtype=torch.uint8) for _ in range(42)];res[0]=torch.tensor([1,1] if behavior=='remote_only' else [0,1],dtype=torch.uint8)
                    module.tier_remote(server.host,server.port,res,list(range(3,45)),10000)
                    module.tier_remote_set(True,1)
                    ctrl,_,_,_=self._start_coordinator(module)
                    got=module.tier_lut_fixture_job(0,x,sel,w,1,1000)
                    torch.testing.assert_close(got,reference,rtol=2e-6,atol=1e-10)
                    self.assertEqual(int(ctrl[1]),1)
                    stats=module.tier_remote_stats()
                    if behavior in ('success','partial','all_partial','remote_only'):
                        self.assertEqual(stats[1],1)
                        self.assertEqual(stats[17],int(behavior in ('partial','all_partial')))
                    else:
                        self.assertEqual(stats[3],1)
                        self.assertEqual(stats[4],4)
                    if behavior=='success':self.assertEqual(stats[11],1)
                finally:
                    module.tier_stop();module.tier_lut_fixture_reset();server.close()


class MockRemote:
    """Loopback protocol peer with bounded lifecycle; no remote service is used."""
    def __init__(self,torch,version,behavior,parts,hidden):
        self.torch,self.version,self.behavior,self.parts,self.hidden=torch,version,behavior,parts,hidden
        self.listener=socket.socket();self.listener.bind(('localhost',0));self.listener.listen(1);self.listener.settimeout(2)
        self.host,self.port=self.listener.getsockname();self.peer=None;self.error=None
        self.handshake_started=threading.Event();self.handshake_release=threading.Event()
        self.thread=threading.Thread(target=self.run,daemon=True);self.thread.start()
    @staticmethod
    def read(peer,n):
        data=bytearray()
        while len(data)<n:
            block=peer.recv(n-len(data))
            if not block:raise EOFError()
            data.extend(block)
        return bytes(data)
    def run(self):
        try:
            self.peer,_=self.listener.accept();self.peer.settimeout(2)
            header=self.read(self.peer,24)
            if self.behavior=='slow_handshake':
                self.handshake_started.set();self.handshake_release.wait(2)
            self.peer.sendall(header)
            self.read(self.peer,24)
            config=('{"version":'+str(self.version)+'}').encode();self.peer.sendall(b'GRTC'+struct.pack('<I',len(config))+config)
            while True:
                header=self.read(self.peer,32 if self.version==2 else 24)
                magic,seq,layer,ntok,npicks,flags=struct.unpack('<4sIiiii',header[:24])
                self.read(self.peer,ntok*self.hidden*2)
                picks=[struct.unpack('<iif',self.read(self.peer,12)) for _ in range(npicks)]
                if self.version==2:self.read(self.peer,struct.unpack('<H',header[28:30])[0]*4)
                if self.behavior=='drop':return
                mask=(1<<1) if self.behavior=='partial' else 0
                if self.behavior=='all_partial':mask=(1<<npicks)-1
                if self.behavior=='invalid_mask':mask=1<<63
                if self.behavior=='bad_seq':seq+=1
                refuse=int(self.behavior=='refuse')
                if self.version==2:
                    reply=struct.pack('<4sIiiiiIHHQ',b'GRT2',seq,layer,ntok,refuse,0,1,0,0,mask)
                else:reply=struct.pack('<4sIiiii',b'GRT1',seq,layer,ntok,npicks,refuse)
                if self.behavior=='trickle':
                    for byte in reply:self.peer.sendall(bytes([byte]));time.sleep(.003)
                    continue
                self.peer.sendall(reply)
                if not refuse:
                    result=self.torch.zeros((ntok,self.hidden),dtype=self.torch.float32)
                    for slot,(token,expert,weight) in enumerate(picks):
                        if not (mask>>slot)&1:result[token]+=self.parts[token,expert]*weight
                    if self.behavior=='nan':result[0,0]=float('nan')
                    self.peer.sendall(result.numpy().tobytes())
        except (EOFError,ConnectionError,OSError):pass
        except BaseException as error:self.error=error
        finally:
            if self.peer:self.peer.close()
    def close(self):
        self.listener.close()
        if self.peer:
            try:self.peer.shutdown(socket.SHUT_RDWR)
            except OSError:pass
        self.thread.join(3)
        if self.thread.is_alive():raise AssertionError('mock remote failed to drain')
        if self.error:raise self.error


if __name__ == '__main__':
    unittest.main()
