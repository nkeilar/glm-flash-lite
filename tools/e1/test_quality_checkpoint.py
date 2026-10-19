#!/usr/bin/env python3
"""Offline synthetic algebra/parser fixtures; never evidence of model quality."""
import array
import copy
import ctypes as C
import contextlib
import errno
import hashlib
import io
import json
import math
import os
from pathlib import Path
import struct
import socket
import shutil
import ssl
import tempfile
import unittest
import urllib.error
from unittest import mock
from types import SimpleNamespace
from quality_checkpoint import (EXL3_SOURCE_SHA, EXL3_REV, MAX_PUBLISHER_RANGE, Replay, SOURCE_REV,
    admit_output, coverage, load_capture, metric, compare, sha256, verify_publisher_payload,
    json_digest, build, real_report, real_binding, validate_s1_real_sources)
from capture_activations import context_hash
from fetch_unsloth_baseline import (DEFAULT_RANGE_MIB, ORIGIN, SUMMARY_SHA256, checked_range,
    fetch_tensor, retry_range, range_bytes)
from gguf import read_gguf, write_gguf
from quantize_s1 import PINNED

class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not shutil.which("g++") or not {"avx2", "fma", "f16c"} <= set(Path("/proc/cpuinfo").read_text().split()):
            raise unittest.SkipTest("CPU replay algebra requires g++ and AVX2/FMA/F16C")
        cls.temporary = tempfile.TemporaryDirectory(prefix="e1-replay-test-")
        path = Path(cls.temporary.name)/"quality_replay.so"
        build(Path(__file__).absolute().parent.parent/"lutbench", path)
        cls.replay = Replay(path)
        cls.h, cls.interm = 256,512
        cls.inputs = [[0.7*math.sin(i*.133+t*.7)+0.2*math.cos(i*.071) for i in range(cls.h)] for t in range(4)]
        cls.packed = {}

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def weights(self, types):
        if tuple(types) in self.packed: return self.packed[tuple(types)]
        result = []
        for part, (typ,width,rows) in enumerate(zip(types,(self.h,self.h,self.interm),(self.interm,self.interm,self.h))):
            stride = self.replay.row_bytes(typ,width)
            data = bytearray(stride*rows)
            for row in range(rows):
                # Independent small rows; include tails large enough to exercise clamp10.
                scale = 0.15 if part < 2 else 0.08
                values = [scale*(math.sin((row+1)*(i+3)*.019+part)+math.cos(i*.17+row)) for i in range(width)]
                x = (C.c_float*width)(*values)
                target = (C.c_ubyte*stride).from_buffer(data,row*stride)
                self.assertEqual(self.replay.lib.e1_quantize(typ,x,width,target),0)
            result.append(bytes(data))
        self.packed[tuple(types)] = result
        return result


    def test_tp2_matches_full_for_profiles_paths_m(self):
        for types in ([138,138,144],[144,144,14],[21,21,23],[23,23,14],[8,8,8]):
            weights = self.weights(types)
            for path in ("i16","q8"):
                for m in (1,2,4):
                    with self.subTest(types=types,path=path,m=m):
                        full = self.replay.forward(weights,types,self.h,self.interm,self.inputs[:m],path=path,tp2=False)
                        split = self.replay.forward(weights,types,self.h,self.interm,self.inputs[:m],path=path,tp2=True)
                        error = sum((a-b)**2 for fa,sa in zip(full,split) for a,b in zip(fa,sa))
                        energy = sum(a*a for row in full for a in row)
                        self.assertLess(math.sqrt(error/energy),1e-5)


    def test_q8_reference_independent_scalar_decode_and_clamp(self):
        weights = self.weights([8,8,8])
        x = [v*100 for v in self.inputs[0]]
        got = self.replay.forward(weights,[8,8,8],self.h,self.interm,[x],reference=True)[0]
        f32 = lambda v:C.c_float(v).value
        x = [f32(v) for v in x]
        def decode(payload,width,rows):
            stride = width//32*34
            result = []
            for r in range(rows):
                row=[]
                for at in range(r*stride,(r+1)*stride,34):
                    scale=struct.unpack_from("<e",payload,at)[0]
                    row.extend(f32(scale*q) for q in struct.unpack_from("32b",payload,at+2))
                result.append(row)
            return result
        gate,up = [decode(w,self.h,self.interm) for w in weights[:2]]
        hidden=[]
        for g,u in zip(gate,up):
            gv=f32(sum(a*b for a,b in zip(g,x))); uv=f32(sum(a*b for a,b in zip(u,x)))
            exp=f32(math.exp(-gv)) if -gv < 88 else float("inf")
            activated=f32(gv/f32(1+exp))
            hidden.append(f32(min(activated,10)*min(max(uv,-10),10)))
        down=decode(weights[2],self.interm,self.h)
        expected=[f32(sum(a*b for a,b in zip(r,hidden))) for r in down]
        self.assertLess(max(abs(a-b) for a,b in zip(got,expected)),1e-5)


    def test_nonfinite_and_wrong_reference_rejected(self):
        w=self.weights([138,138,144])
        with self.assertRaisesRegex(ValueError,"reference must"):
            self.replay.forward(w,[138,138,144],self.h,self.interm,self.inputs[:1],reference=True)
        x=copy.deepcopy(self.inputs[:1]); x[0][3]=float("nan")
        with self.assertRaisesRegex(ValueError,"nonfinite"):
            self.replay.forward(w,[138,138,144],self.h,self.interm,x)


class ProvenanceTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="e1-quality-test-")
        self.root=Path(self.temp.name)
        corpus=[{"id":str(i),"class":c,"text":"synthetic parser test"} for i,c in enumerate(("prose","code","firmware","code"))]
        trajectories=[{"prompt_id":p["id"],"prompt_class":p["class"],"prompt_tokens":[1,2,3],"generated_tokens":list(range(33))} for p in corpus]
        ledger={"schema":"e1-capture-model/v1","model_id":"GLM-5.3-Flash","revision":EXL3_REV,"source_manifest_sha256":EXL3_SOURCE_SHA,"files":[{"path":"fake","sha256":"0"*64}]}
        for name,value in (("corpus.json",corpus),("model-ledger.json",ledger),("trajectories.json",trajectories)):
            (self.root/name).write_text(json.dumps(value))
        self.rows=[]
        with (self.root/"inputs.f32").open("wb") as f:
            for layer in (3,11):
                for p,t in zip(corpus,trajectories):
                    for step in range(1,33):
                        self.rows.append({"row":len(self.rows),"layer":layer,"prompt_id":p["id"],"prompt_class":p["class"],"phase":"decode","step":step,
                                          "context_sha256":context_hash(t["prompt_tokens"]+t["generated_tokens"][:step]),
                                          "experts":[(step*8+i)%288 for i in range(8)],"routing_weights":[.125]*8,
                                          "source_dtype":"float16","stored_dtype":"float32"})
                        f.write(array.array("f",[.1]*4096).tobytes())
        self.write_rows()
        (self.root/"stats.json").write_text("{}")
        (self.root/"runtime-ledger.json").write_text(json.dumps({"torch_metadata_version":"unit","torch_runtime_version":"unit","exllamav3_version":"test-fixture","files":[{"path":"fake","sha256":"0"*64}],"serving_sources":"test-fixture",
                "native_extension":{"name":"exllamav3_ext","path":"/unit/exllamav3_ext.so","sha256":"0"*64,"bytes":1,"identity":{k:1 for k in ("dev","ino","size","mtime_ns","ctime_ns")}},
                "ec_precompile":{"source_revision":"b"*40,"torch_version":"unit","receipt_sha256":"0"*64,"module_sha256":"0"*64},
                "capture_tools":{name:"0"*64 for name in ("capture_activations.py","quality_checkpoint.py","gguf.py","quantize_s1.py","maintenance_guard.py")}}))
        self.manifest={"schema":"e1-real-activations/v1","complete":True,"kind":"offline-exllamav3-routing-input","synthetic":False,
                       "model_id":"GLM-5.3-Flash","source_revision":EXL3_REV,"serving_revision":"b"*40,"exllamav3_version":"test-fixture",
                       "hidden_size":4096,"intermediate_size":2048,"act_limit":10,"top_k":8,"rows":len(self.rows)}
        for key,name in (("runtime_ledger","runtime-ledger.json"),("stats","stats.json"),("corpus","corpus.json"),("model_ledger","model-ledger.json"),("trajectories","trajectories.json"),("inputs","inputs.f32"),("routes","routes.jsonl")):
            self.manifest[key]={"path":name,"sha256":sha256(self.root/name)}
        self.save()


    def tearDown(self): self.temp.cleanup()


    def write_rows(self):
        (self.root/"routes.jsonl").write_text("".join(json.dumps(r)+"\n" for r in self.rows))


    def save(self):
        (self.root/"capture.json").write_text(json.dumps(self.manifest))


    def test_provenance_shaped_fixture_and_coverage(self):
        # Accepted parser shape is not a real quality PASS: no weights/replay.
        _,rows=load_capture(self.root/"capture.json")
        self.assertEqual(coverage(rows,[3,11])[3]["tokens"],128)


    def test_synthetic_missing_provenance_and_hash_rejected(self):
        self.manifest["synthetic"]=True; self.save()
        with self.assertRaisesRegex(ValueError,"synthetic"):
            load_capture(self.root/"capture.json")
        self.manifest["synthetic"]=False; del self.manifest["source_revision"]; self.save()
        with self.assertRaisesRegex(ValueError,"source_revision"):
            load_capture(self.root/"capture.json")
        self.manifest["source_revision"]=EXL3_REV; self.manifest["inputs"]["sha256"]="0"*64; self.save()
        with self.assertRaisesRegex(ValueError,"SHA256 mismatch"):
            load_capture(self.root/"capture.json")


    def test_context_routes_and_nonfinite_fail_closed(self):
        for mutation,pattern in ((lambda r:r.update(context_sha256="0"*64),"context mismatch"),
                                 (lambda r:r.update(experts=[0]*8),"top-8"),
                                 (lambda r:r.update(routing_weights=[float("nan")]*8),"routing weights")):
            original=copy.deepcopy(self.rows[0]); mutation(self.rows[0]); self.write_rows()
            self.manifest["routes"]["sha256"]=sha256(self.root/"routes.jsonl"); self.save()
            with self.assertRaisesRegex(ValueError,pattern): load_capture(self.root/"capture.json")
            self.rows[0]=original
        self.write_rows(); self.manifest["routes"]["sha256"]=sha256(self.root/"routes.jsonl")
        with (self.root/"inputs.f32").open("r+b") as f: f.write(struct.pack("<f",float("inf")))
        self.manifest["inputs"]["sha256"]=sha256(self.root/"inputs.f32"); self.save()
        with self.assertRaisesRegex(ValueError,"nonfinite activation"): load_capture(self.root/"capture.json")


    def test_coverage_rejects_mismatched_layers_and_rare_only_sample(self):
        _,rows=load_capture(self.root/"capture.json")
        with self.assertRaisesRegex(ValueError,"inadequate sample"): coverage(rows[:10]+rows[128:138],[3,11])
        rows[0]["experts"]=[0,1,2,3,4,5,6,7]
        for r in rows: r["experts"]=[0,1,2,3,4,5,6,7]
        with self.assertRaisesRegex(ValueError,"inadequate sample"): coverage(rows,[3,11])


    def test_gate_requires_both_metrics(self):
        control=metric([(1.,1),(2.,3),(3.,1)])
        self.assertTrue(compare(control,control))
        for key in control:
            worse=dict(control); worse[key]+=1e-8
            self.assertFalse(compare(worse,control))


class Response(io.BytesIO):
    def __init__(self,payload,status=206,**headers):
        super().__init__(payload); self.status=status; self.headers=headers; self.read_calls=0
    def read(self,*a): self.read_calls+=1; return super().read(*a)
    def __enter__(self): return self
    def __exit__(self,*a): pass


class FetchTests(unittest.TestCase):
    def response(self,data=b"abc",**overrides):
        values={"Content-Range":"bytes 5-7/100","Content-Length":"3","ETag":"etag"}; values.update(overrides)
        return Response(data,**values)


    def test_range_rejects_before_body_and_accepts_exact(self):
        for status,headers in ((200,{}),(206,{"Content-Range":"bytes 4-6/100"}),(206,{"Content-Length":"100"}),(206,{"ETag":"other"}),(206,{"Content-Encoding":"gzip"})):
            response=self.response(**headers); response.status=status
            with self.assertRaises(ValueError): checked_range(lambda *a,**k:response,"https://example.test",5,3,100,"etag")
            self.assertEqual(response.read_calls,0)
        response=self.response()
        self.assertEqual(checked_range(lambda *a,**k:response,"https://example.test",5,3,100,"etag"),b"abc")


    def test_retry_dns_or_partial_body_reset_then_success_commits_range_once(self):
        class PartialReset(Response):
            def read1(self,n=-1):
                if self.tell(): raise ConnectionResetError("secret signed redirect")
                return super().read(1)
        for kind in ("dns","partial-reset"):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory(prefix="e1-retry-test-") as td:
                root=Path(td); events=[]; requests=[]
                tensor={"name":"test","data_relative_offset":5,"tensor_bytes":3}
                shard={"file":"UD-IQ4_XS/test.gguf","tensor_data_start":0,"file_size":100,"etag":"etag","lfs_sha256":"f"*64}
                def opener(request,**kwargs):
                    requests.append(request.headers["Range"])
                    self.assertEqual((root/"test.payload").stat().st_size,0)
                    self.assertEqual((root/"test.ranges.jsonl").stat().st_size,0)
                    if len(requests)==1:
                        if kind=="dns": raise urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN,"secret signed redirect"))
                        return PartialReset(b"abc",**{"Content-Range":"bytes 5-7/100","Content-Length":"3","ETag":"etag"})
                    return self.response()
                output=io.StringIO()
                with mock.patch("fetch_unsloth_baseline.time.sleep",side_effect=lambda delay:events.append(("sleep",delay))),contextlib.redirect_stdout(output):
                    path,records=fetch_tensor(root,tensor,shard,lambda:events.append("guard"),opener)
                self.assertEqual(path.read_bytes(),b"abc")
                self.assertEqual(len(records),1)
                self.assertEqual(requests,["bytes=5-7","bytes=5-7"])
                self.assertNotIn("secret",output.getvalue())
                self.assertEqual(json.loads(output.getvalue())["event"],"PUBLISHER_RANGE_RETRY")
                index=events.index(("sleep",1.))
                self.assertEqual(events[index-1],"guard"); self.assertEqual(events[index+1],"guard")


    def test_retry_exhaustion_transient_http_and_permanent_failures_are_bounded_and_sanitized(self):
        for status in (429,503):
            body=io.BytesIO(b"do not read")
            failure=urllib.error.HTTPError("https://signed.test/secret",status,"secret",None,body)
            reports=[]; delays=[]; calls=[]
            def opener(*args,**kwargs):
                calls.append(True)
                if len(calls)==1: raise failure
                return self.response()
            self.assertEqual(retry_range(opener,"https://example.test",5,3,100,"etag",lambda:None,lambda:100,sleep=delays.append,report=reports.append),b"abc")
            self.assertTrue(body.closed); self.assertEqual(delays,[1.]); self.assertEqual(len(calls),2)
            self.assertNotIn("secret",json.dumps(reports))
        reports=[]; delays=[]; calls=[]
        def failing(*args,**kwargs):
            calls.append(True)
            raise urllib.error.URLError(socket.gaierror(socket.EAI_AGAIN,"secret signed redirect"))
        with self.assertRaisesRegex(RuntimeError,"exhausted after 3 attempts") as caught:
            retry_range(failing,"https://example.test",5,3,100,"etag",lambda:None,lambda:100,sleep=delays.append,report=reports.append)
        self.assertEqual(len(calls),3); self.assertEqual(delays,[1.,2.])
        self.assertNotIn("secret",str(caught.exception)+json.dumps(reports))
        failures=[urllib.error.HTTPError("https://signed.test/secret",404,"secret",None,io.BytesIO()),
                  urllib.error.URLError(socket.gaierror(socket.EAI_NONAME,"secret")),
                  urllib.error.URLError(ssl.SSLCertVerificationError(1,"secret")),ValueError("publisher ETag changed")]
        for failure in failures:
            calls=[]
            def permanent(*args,**kwargs): calls.append(True); raise failure
            expected=ValueError if isinstance(failure,ValueError) else RuntimeError
            with self.assertRaises(expected) as caught:
                retry_range(permanent,"https://example.test",5,3,100,"etag",lambda:None,lambda:100,
                            sleep=lambda delay:self.fail("permanent failures cannot sleep"),report=lambda event:self.fail("permanent failures cannot retry"))
            self.assertEqual(len(calls),1)
            self.assertNotIn("secret",str(caught.exception))


    def test_retry_deadline_is_checked_before_backoff_or_another_attempt(self):
        calls=[]; guards=[]; remaining=iter([90.,.5])
        def opener(*args,**kwargs): calls.append(True); raise TimeoutError("secret")
        with self.assertRaisesRegex(ValueError,"no retry backoff budget"):
            retry_range(opener,"https://example.test",5,3,100,"etag",lambda:guards.append(True),lambda:next(remaining),
                        sleep=lambda delay:self.fail("deadline forbids backoff"),report=lambda event:self.fail("deadline forbids retry"))
        self.assertEqual(len(calls),1); self.assertEqual(len(guards),2)


    def test_configurable_range_bounds_and_maximum_checked_before_body(self):
        self.assertEqual(DEFAULT_RANGE_MIB,32)
        self.assertEqual(range_bytes(32),32*2**20)
        self.assertEqual(range_bytes(64),MAX_PUBLISHER_RANGE)
        for invalid in (0,-1,65,True,1.5):
            with self.assertRaisesRegex(ValueError,"range-mib"):
                range_bytes(invalid)
        for size in (0,True,MAX_PUBLISHER_RANGE+1):
            with self.assertRaisesRegex(ValueError,"bounded tensor range"):
                checked_range(lambda *a,**k:self.fail("invalid size must not open HTTP"),"https://example.test",0,size,MAX_PUBLISHER_RANGE+1,"etag")
        response=self.response(); response.status=200
        with self.assertRaisesRegex(ValueError,"range server returned 200"):
            checked_range(lambda *a,**k:response,"https://example.test",0,MAX_PUBLISHER_RANGE,MAX_PUBLISHER_RANGE,"etag")
        self.assertEqual(response.read_calls,0)


    def test_old_4mib_prefix_resumes_32mib_ranges_and_old_reader_setting_accepts_them(self):
        with tempfile.TemporaryDirectory(prefix="e1-mixed-ranges-") as td:
            root=Path(td); mib=2**20; prefix=4*mib; total=70*mib; start=100
            name="blk.3.ffn_gate_exps.weight"; shardname="UD-IQ4_XS/test.gguf"
            tensor={"name":name,"data_relative_offset":0,"tensor_bytes":total}
            shard={"file":shardname,"tensor_data_start":start,"file_size":start+total,"etag":"etag","lfs_sha256":"f"*64}
            url=f"https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/{SOURCE_REV}/{shardname}"
            old=b"a"*prefix
            (root/(name+".payload")).write_bytes(old)
            record={"tensor":name,"url":url,"start":start,"end":start+prefix-1,"bytes":prefix,"etag":"etag","lfs_sha256":"f"*64,"sha256":hashlib.sha256(old).hexdigest()}
            (root/(name+".ranges.jsonl")).write_text(json.dumps(record)+"\n")
            requests=[]; reports=[]; ticks=[]
            def opener(request,**kwargs):
                lo,hi=map(int,request.headers["Range"].removeprefix("bytes=").split("-"))
                size=hi-lo+1; requests.append((lo,size))
                return Response(b"x"*size,**{"Content-Range":f"bytes {lo}-{hi}/{start+total}","Content-Length":str(size),"ETag":"etag"})
            path,records=fetch_tensor(root,tensor,shard,lambda:ticks.append(True),opener,
                progress=lambda name,done,total:reports.append(done),request_bytes=range_bytes(32))
            self.assertEqual(requests,[(start+prefix,32*mib),(start+36*mib,32*mib),(start+68*mib,2*mib)])
            self.assertEqual([r["bytes"] for r in records],[4*mib,32*mib,32*mib,2*mib])
            self.assertEqual(reports,[68*mib,70*mib])
            self.assertGreaterEqual(len(ticks),1+2*len(requests))
            with path.open("rb") as f: self.assertEqual(f.read(prefix),old)
            # Range size is operational: reducing the new request size must not
            # invalidate already verified 32 MiB records on a subsequent resume.
            _,retained=fetch_tensor(root,tensor,shard,lambda:None,lambda *a,**k:self.fail("completed payload must not fetch"),request_bytes=range_bytes(4))
            self.assertEqual(retained,records)
            dims=[1,total//4,1]
            source=SimpleNamespace(path=path,data_start=0,tensors=[{"name":name,"dims":dims,"type_id":0,"nbytes":total,"offset":0}])
            summary={"shards":[shard],"expert_tensors":[dict(tensor,shard=shardname,ggml_dimensions=dims,type_id=0)]}
            verify_publisher_payload(source,{"ranges":records},summary)
            malformed=copy.deepcopy(records); malformed[1]["bytes"]=MAX_PUBLISHER_RANGE+1
            with self.assertRaisesRegex(ValueError,"unbounded ranges"):
                verify_publisher_payload(source,{"ranges":malformed},summary)


    def test_range_deadline_rejects_before_body_and_limits_socket_timeout(self):
        response=self.response()
        with mock.patch("fetch_unsloth_baseline.time.monotonic",side_effect=[0.,2.]):
            with self.assertRaisesRegex(ValueError,"body deadline"):
                checked_range(lambda *a,**k:response,"https://example.test",5,3,100,"etag",timeout=1)
        self.assertEqual(response.read_calls,0)
        with tempfile.TemporaryDirectory(prefix="e1-fetch-deadline-") as td:
            observed=[]
            def opener(*a,**k): observed.append(k["timeout"]); return self.response()
            tensor={"name":"test","data_relative_offset":5,"tensor_bytes":3}
            shard={"file":"UD-IQ4_XS/test.gguf","tensor_data_start":0,"file_size":100,"etag":"etag","lfs_sha256":"f"*64}
            fetch_tensor(Path(td),tensor,shard,lambda:None,opener,remaining=lambda:.25)
            self.assertEqual(observed,[.25])


    def test_resume_validates_retained_chunks_and_restores_exact_payload(self):
        with tempfile.TemporaryDirectory(prefix="e1-fetch-test-") as td:
            root=Path(td); payload=b"abc"
            tensor={"name":"test","data_relative_offset":5,"tensor_bytes":3}
            shard={"file":"UD-IQ4_XS/test.gguf","tensor_data_start":0,"file_size":100,"etag":"etag","lfs_sha256":"f"*64}
            calls=[]
            def opener(*args,**kwargs): calls.append(args); return self.response()
            p,records=fetch_tensor(root,tensor,shard,lambda:None,opener)
            with p.open("ab") as f:f.write(b"unlogged")
            p2,r2=fetch_tensor(root,tensor,shard,lambda:None,opener)
            self.assertEqual(p2.read_bytes(),payload); self.assertEqual(records,r2); self.assertEqual(len(calls),1)
            p.write_bytes(b"xyz")
            with self.assertRaisesRegex(ValueError,"resume chunk hash mismatch"):
                fetch_tensor(root,tensor,shard,lambda:None,opener)


    def test_local_ud_payload_must_match_range_hash(self):
        with tempfile.TemporaryDirectory(prefix="e1-ud-test-") as td:
            root=Path(td); payload=b"\0"*272
            name="blk.3.ffn_down_exps.weight"; shardname="UD-IQ4_XS/test.gguf"
            write_gguf(root/"test.gguf",{},[{"name":name,"dims":[256,2,1],"type_id":23,"data":payload}])
            source=read_gguf(root/"test.gguf")
            baseline={"ranges":[{"tensor":name,"url":f"https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/{SOURCE_REV}/{shardname}","start":100,"end":371,"bytes":272,"etag":"etag","lfs_sha256":"f"*64,"sha256":hashlib.sha256(payload).hexdigest()}]}
            summary={"shards":[{"file":shardname,"tensor_data_start":100,"etag":"etag","lfs_sha256":"f"*64}],
                     "expert_tensors":[{"name":name,"shard":shardname,"data_relative_offset":0,"ggml_dimensions":[256,2,1],"type_id":23,"tensor_bytes":272}]}
            verify_publisher_payload(source,baseline,summary)
            baseline["ranges"][0]["sha256"]="0"*64
            with self.assertRaisesRegex(ValueError,"payload/range hash mismatch"): verify_publisher_payload(source,baseline,summary)


    def test_corrupt_resume_chunk_is_rejected_before_payload_read(self):
        with tempfile.TemporaryDirectory(prefix="e1-journal-test-") as td:
            root=Path(td)
            tensor={"name":"test","data_relative_offset":5,"tensor_bytes":3}
            shard={"file":"UD-IQ4_XS/test.gguf","tensor_data_start":0,"file_size":100,"etag":"etag","lfs_sha256":"f"*64}
            _,records=fetch_tensor(root,tensor,shard,lambda:None,lambda *a,**k:self.response())
            for size in (-1,True,MAX_PUBLISHER_RANGE+1,4):
                invalid=dict(records[0],bytes=size)
                (root/"test.ranges.jsonl").write_text(json.dumps(invalid)+"\n")
                with self.assertRaisesRegex(ValueError,"bounds invalid before payload read"):
                    fetch_tensor(root,tensor,shard,lambda:None,lambda *a,**k:self.fail("no network on invalid resume"))


class RealReportTests(unittest.TestCase):
    def fixture(self):
        layers={str(layer):dict(tp2_full_down_max_relative_l2=dict(ours=1e-7,unsloth=1e-7),coverage=dict(tokens=128,prompts=4,classes=["code","firmware","prose"],experts=32,route_occurrences=1024),
                    experts=[dict(expert=e,routing_count=32,ours_sample_errors=[.05]*32,unsloth_sample_errors=[.1]*32) for e in range(32)]) for layer in (3,11)}
        return layers


    def test_real_gate_recomputes_raw_errors_and_frequency_once(self):
        from quality_checkpoint import real_report
        layers=self.fixture()
        value=real_report(layers,"i16",1,10,{"unit":"fixture"})
        self.assertEqual(value["state"],"REAL-PASS")
        self.assertIs(value["real_pass"],True)
        self.assertAlmostEqual(value["aggregate"]["ours"]["routing_frequency_weighted_mean_nmse"],.05)
        for item in layers["11"]["experts"]: item["ours_sample_errors"]=[.2]*32
        self.assertEqual(real_report(layers,"i16",1,10,{})["state"],"REAL-FAIL")


    def test_s1_real_pass_is_a_diagnostic_with_no_production_or_bank_claim(self):
        from quality_checkpoint import real_report, S1_REAL_SCOPE
        binding=dict(scope=S1_REAL_SCOPE)
        result=real_report(self.fixture(),"i16",1,10,binding)
        self.assertEqual(result["state"],"REAL-PASS")
        self.assertIs(result["real_pass"],True)
        self.assertIs(result["synthetic"],False)
        self.assertEqual(result["scope"],S1_REAL_SCOPE)
        self.assertIs(result["production_ready"],False)
        self.assertIs(result["full_bank_bound"],False)
        self.assertNotIn("bank_manifest_sha256",result["binding"])
        with self.assertRaisesRegex(ValueError,"full-bank binding"):
            real_report(self.fixture(),"i16",1,10,dict(binding,bank_manifest_sha256="0"*64))


    def test_real_gate_rejects_missing_layer_nonfinite_census_and_short_coverage(self):
        from quality_checkpoint import real_report
        for mutation in (lambda v:v.pop("11"),lambda v:v["3"]["experts"][0]["ours_sample_errors"].__setitem__(0,float("nan")),
                         lambda v:v["3"]["experts"].pop(),lambda v:v["3"]["coverage"].update(tokens=127)):
            value=self.fixture();mutation(value)
            with self.assertRaises(ValueError): real_report(value,"i16",1,10,{})


    def test_installed_native_extension_is_discovered_without_import_and_bound_to_loaded_bytes(self):
        from capture_activations import discover_native_extension,hash_native_extension,verify_loaded_native_extension
        with tempfile.TemporaryDirectory(prefix="e1-native-test-") as tmp:
            path=Path(tmp)/"exllamav3_ext.so";path.write_bytes(b"dummy native fixture; never imported")
            spec=SimpleNamespace(origin=str(path),loader=object())
            with mock.patch("capture_activations.importlib.util.find_spec",return_value=spec) as find:
                metadata=discover_native_extension()
            find.assert_called_once_with("exllamav3_ext")
            self.assertNotIn("sha256",metadata)
            record=hash_native_extension(metadata)
            self.assertEqual(record["sha256"],sha256(path))
            with mock.patch.dict("sys.modules",{"exllamav3_ext":SimpleNamespace(__file__=str(path))}):
                self.assertEqual(verify_loaded_native_extension(record),record)
                path.write_bytes(b"changed")
                with self.assertRaisesRegex(ValueError,"changed"): verify_loaded_native_extension(record)
            with mock.patch.dict("sys.modules",{"exllamav3_ext":SimpleNamespace(__file__=str(path)+".other")}):
                with self.assertRaisesRegex(ValueError,"differs"): verify_loaded_native_extension(record)


    def test_missing_shadowed_fifo_or_oversized_native_extension_is_rejected(self):
        from capture_activations import discover_native_extension
        with tempfile.TemporaryDirectory(prefix="e1-native-test-") as tmp:
            path=Path(tmp)/"exllamav3_ext.so";os.mkfifo(path)
            specs=(None,SimpleNamespace(origin=str(path.with_suffix(".py")),loader=object()),SimpleNamespace(origin=str(path),loader=object()))
            for spec in specs:
                with mock.patch("capture_activations.importlib.util.find_spec",return_value=spec):
                    with self.assertRaises(ValueError): discover_native_extension()
            path.unlink();path.write_bytes(b"four")
            with mock.patch("capture_activations.importlib.util.find_spec",return_value=SimpleNamespace(origin=str(path),loader=object())),\
                 mock.patch("capture_activations.NATIVE_LIMIT",3):
                with self.assertRaisesRegex(ValueError,"bounded"): discover_native_extension()


    def test_accepted_sequence_ids_track_held_utf8_tokens_and_reject_draft_jumps(self):
        from capture_activations import accepted_generated_tokens
        class Sequence:
            def __init__(self,values): self.values=values
            def torch(self): return self
            def tolist(self): return [self.values]
        seq=Sequence([1,2,3]);job=SimpleNamespace(sequences=[SimpleNamespace(sequence_ids=seq)])
        self.assertEqual(accepted_generated_tokens(job,[1,2],[]),[3])
        seq.values=[1,2,3,4]
        self.assertEqual(accepted_generated_tokens(job,[1,2],[3]),[3,4])
        seq.values=[1,2,3,4,5,6]
        with self.assertRaisesRegex(ValueError,'multiple'): accepted_generated_tokens(job,[1,2],[3,4])


class CapturePostLoadTests(unittest.TestCase):
    def test_routing_hook_requires_untransformed_expert_ingress(self):
        from capture_activations import require_capture_ingress
        require_capture_ingress(SimpleNamespace(router_pre_norm=None,routed_pre_norm=None,latent_in=None))
        for name in ("router_pre_norm","routed_pre_norm","latent_in"):
            with self.subTest(name=name),self.assertRaisesRegex(ValueError,"differs from expert ingress"):
                require_capture_ingress(SimpleNamespace(**{name:object()}))


    def test_generator_contained_errors_surface_before_any_capture_can_publish(self):
        from capture_activations import capture_iteration
        failure=RuntimeError("actual prefill failure")
        for value in (failure,"reported failure"):
            with self.subTest(value=value):
                generator=SimpleNamespace(iterate=lambda:[dict(stage="error",error=value,eos=True)])
                with self.assertRaisesRegex((RuntimeError,ValueError),"failure"):capture_iteration(generator)
        events=[dict(stage="prefill"),dict(stage="streaming",eos=False)]
        self.assertEqual(capture_iteration(SimpleNamespace(iterate=lambda:events)),events)


    def test_actual_routing_payload_writes_little_endian_values_and_context(self):
        from capture_activations import write_capture_row
        vectors,routes=io.BytesIO(),io.StringIO()
        active=dict(id="prompt",**{"class":"code"},prompt=[1,2],generated=[3,4])
        tick=mock.Mock()
        record=write_capture_row(vectors,routes,5,11,active,[.25]*4096,list(range(8)),[.125]*8,tick)
        tick.assert_called_once_with()
        self.assertEqual(len(vectors.getvalue()),4096*4)
        self.assertEqual(struct.unpack("<f",vectors.getvalue()[:4]),(.25,))
        self.assertEqual(json.loads(routes.getvalue()),record)
        self.assertEqual(record["context_sha256"],context_hash([1,2,3,4]))
        self.assertEqual((record["row"],record["layer"],record["step"]),(5,11,2))


    def test_routing_invalid_values_or_expired_guard_write_nothing(self):
        from capture_activations import write_capture_row
        active=dict(id="prompt",**{"class":"code"},prompt=[1],generated=[2])
        for bad in ("activation","expert","weight","context","guard"):
            with self.subTest(bad=bad):
                vectors,routes=io.BytesIO(),io.StringIO();x=[.25]*4096;ids=list(range(8));weights=[.125]*8
                ctx=dict(active);tick=lambda:None
                if bad=="activation":x[0]=float("nan")
                elif bad=="expert":ids[-1]=ids[0]
                elif bad=="weight":weights[0]=float("inf")
                elif bad=="context":ctx["generated"]=[]
                else:tick=mock.Mock(side_effect=ValueError("guard failed"))
                with self.assertRaises(ValueError):write_capture_row(vectors,routes,0,3,ctx,x,ids,weights,tick)
                self.assertEqual(vectors.getvalue(),b"");self.assertEqual(routes.getvalue(),"")


    def test_monitor_failure_blocks_completed_manifest_and_exits_only_own_process(self):
        from capture_activations import CaptureMonitor
        with tempfile.TemporaryDirectory(prefix="e1-publication-test-") as tmp:
            path=Path(tmp)/"capture.json";check=mock.Mock(side_effect=ValueError("heartbeat failed"));exits=mock.Mock()
            monitor=CaptureMonitor(check,exit_process=exits)
            with mock.patch.object(monitor.stop,"wait",return_value=False),contextlib.redirect_stderr(io.StringIO()):monitor.run()
            exits.assert_called_once_with(3)
            with self.assertRaisesRegex(ValueError,"previously failed"):monitor.publish(path,dict(complete=True))
            self.assertFalse(path.exists())


    def test_publication_runs_final_guard_and_keeps_partial_evidence_on_failure(self):
        from capture_activations import CaptureMonitor
        with tempfile.TemporaryDirectory(prefix="e1-publication-test-") as tmp:
            root=Path(tmp);(root/"inputs.f32").write_bytes(b"partial evidence")
            check=mock.Mock(side_effect=ValueError("terminal supervisor"));monitor=CaptureMonitor(check)
            with self.assertRaisesRegex(ValueError,"terminal"):monitor.publish(root/"capture.json",dict(complete=True))
            self.assertFalse((root/"capture.json").exists())
            self.assertEqual((root/"inputs.f32").read_bytes(),b"partial evidence")
            check.side_effect=None;manifest=dict(schema="unit-only",complete=True)
            monitor.publish(root/"capture.json",manifest)
            check.assert_called_with(True)
            self.assertEqual(json.loads((root/"capture.json").read_text()),manifest)
            self.assertTrue(monitor.stop.is_set())
            self.assertFalse((root/"capture.json.pending").exists())


    def test_watchdog_detects_blocked_publication_without_its_io_lock(self):
        import threading
        from capture_activations import CaptureMonitor
        now=[0.];entered=threading.Event();release=threading.Event()
        monitor=CaptureMonitor(lambda force:None,monotonic=lambda:now[0])
        def blocked_write(*_):
            entered.set();release.wait(2)
        with mock.patch('capture_activations.atomic_json',side_effect=blocked_write):
            worker=threading.Thread(target=monitor.publish,args=(Path('/tmp/e1-unit-capture.json'),{}))
            worker.start()
            try:
                self.assertTrue(entered.wait(1))
                monitor.watchdog_check()
                now[0]=21
                with self.assertRaisesRegex(ValueError,'publication stalled'):monitor.watchdog_check()
            finally:
                release.set();worker.join(2)
        self.assertFalse(worker.is_alive())


class PublicAdmissionCompatibilityTests(unittest.TestCase):
    def test_generated_report_shape_is_admitted_and_q8_binding_identical(self):
        """Fictitious raw errors exercise the producer/consumer format only."""
        import importlib.util
        root = Path(__file__).absolute().parents[2]
        spec = importlib.util.spec_from_file_location('e1_public_manifest_fixtures', root/'tests/test_lut_manifest.py')
        fixtures = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(fixtures)
        lm = fixtures.lm
        from quantize_s1 import METADATA_SHARD, custom_rules
        with tempfile.TemporaryDirectory(prefix='e1-binding-fixture-') as temp:
            directory = Path(temp)
            manifest = directory/'e1-lut-bank.json'
            manifest.write_text(json.dumps(fixtures.public_manifest()))
            bank = lm.load_manifest(manifest, check_files=False)
            quant = dict(schema='e1-s1/v1', state='S1_READY', revision=SOURCE_REV,
                         quantizer=dict(source_revision=lm.IK_REVISION), layers=[])
            ud = dict(schema='e1-ud/v1', revision=SOURCE_REV, layers=[])
            for layer in (3, 11):
                shard = f'Q8_0/GLM-5.3-Flash-Q8_0-{2 if layer == 3 else 3:05d}-of-00008.gguf'
                context = dict(revision=SOURCE_REV, layer=layer, imatrix=lm.IMATRIX_SHA256,
                               metadata_source=PINNED[METADATA_SHARD][1], sources={shard: PINNED[shard][1]},
                               recipe=custom_rules([layer]), normalized_imatrix_sha256='e'*64)
                quant['layers'].append(dict(layer=layer, q8=dict(path='fixture-q8.gguf', sha256='b'*64, context=copy.deepcopy(context)),
                                           lut=dict(path='fixture-lut.gguf', sha256='a'*64, context=copy.deepcopy(context))))
                ud['layers'].append(dict(layer=layer, origin=ORIGIN, gguf=dict(path='fixture-ud.gguf', sha256='c'*64)))
            validate_s1_real_sources(quant, ud)
            cap = {key: dict(sha256='d'*64) for key in ('model_ledger','runtime_ledger','corpus','stats','inputs','routes','trajectories')}
            cap.update(source_revision=EXL3_REV, serving_revision='1'*40, exllamav3_version='1.5.1')
            library = directory/'fixture.so'
            library.write_bytes(b'parser fixture, never loaded as native code')
            library.with_suffix('.json').write_text('{}')
            binding = real_binding(SimpleNamespace(s1_real_check=False, bank_manifest=manifest, library=library),
                                   cap, quant, ud, lambda: None, 'f'*64)
            rows = RealReportTests().fixture()
            i16 = real_report(rows, 'i16', 1, 10, binding)
            q8 = real_report(rows, 'q8', 1, 10, binding)
            self.assertIs(lm.validate_real_report(i16, bank, 'i16'), i16)
            self.assertIs(lm.validate_real_report(q8, bank, 'q8'), q8)
            self.assertEqual(i16['binding'], q8['binding'])
            self.assertEqual(i16['binding_sha256'], q8['binding_sha256'])
            mutated = copy.deepcopy(i16)
            mutated['binding']['layers']['3']['ours_sha256'] = '0'*64
            mutated['binding_sha256'] = json_digest(mutated['binding'])
            with self.assertRaisesRegex(ValueError, 'bank'):
                lm.validate_real_report(mutated, bank, 'i16')

    def test_atomic_outputs_and_resource_numbers_fail_closed(self):
        from maintenance_guard import publish_json, resources
        with tempfile.TemporaryDirectory(prefix='e1-publication-fixture-') as temp:
            path = Path(temp)/'receipt.json'
            publish_json(path, {'fixture': True})
            with self.assertRaises(FileExistsError):
                publish_json(path, {'changed': True})
            self.assertEqual(json.loads(path.read_text()), {'fixture': True})
            for value in (-1, float('nan'), float('inf')):
                with self.assertRaises(ValueError):
                    resources(temp, min_ram_gib=value)
                with self.assertRaises(ValueError):
                    resources(temp, min_disk_gib=value)
    def test_conversion_publication_receipt_is_a_replay_artifact(self):
        """The producer's absolute paths support bank and scratch separation."""
        from quantize_s1 import publish
        from quality_checkpoint import artifact
        with tempfile.TemporaryDirectory(prefix='e1-receipt-integration-') as temp:
            directory = Path(temp)
            pending, final = directory/'pending.gguf', directory/'published.gguf'
            expected, tensors = [], []
            for part in ('gate', 'up', 'down'):
                name = f'blk.3.ffn_{part}_exps.weight'
                expected.append(dict(name=name, dims=[32,1,1], target_type='Q8_0'))
                tensors.append(dict(name=name, dims=[32,1,1], type_id=8, data=bytes(34)))
            write_gguf(pending, {'general.architecture': (8, 'glm5-next')}, tensors)
            receipt = publish(pending, final, expected, {'fixture': True})
            self.assertTrue(Path(receipt['path']).is_absolute())
            self.assertEqual(artifact(directory/'other-manifest-directory', receipt), final)
            final.write_bytes(b'altered fixture')
            with self.assertRaisesRegex(ValueError, 'SHA256 mismatch'):
                artifact(directory, receipt)
    def test_neutral_capture_stats_match_public_routed_module_keys(self):
        from capture_activations import prepare
        root = Path(__file__).absolute().parents[2]
        expected = json.loads((root/'data/stats_own_dec.json').read_text())
        with tempfile.TemporaryDirectory(prefix='e1-neutral-stats-fixture-') as temp:
            output = Path(temp)/'stats.json'
            with contextlib.redirect_stdout(io.StringIO()):
                prepare(['prepare-stats', '--out', str(output)])
            generated = json.loads(output.read_text())
            self.assertEqual(set(generated), set(expected))
            self.assertTrue(all(values == [0]*288 for values in generated.values()))
