"""CPU-only manifest, streaming-reader and early-startup admission tests."""
import ast
import builtins
import copy
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "glm53"))
import lut_manifest as lm


def public_manifest():
    layers = []
    for layer in lm.LAYERS:
        tensors, end = [], 0
        for part in lm.PARTS:
            item = lm.tensor_layout(layer, part)
            item["offset"] = end
            end = (end + item["nbytes"] + 31) // 32 * 32
            tensors.append(item)
        layers.append(dict(layer=layer, artifact=dict(path=f"blk.{layer:02d}.lut.gguf", sha256="a"*64,
                           bytes=4096+end, data_start=4096), tensors=tensors))
    return dict(schema="e1-lut-bank/v1", state="S2_READY", revision=lm.SOURCE_REVISION, ik_revision=lm.IK_REVISION,
                imatrix_sha256=lm.IMATRIX_SHA256, real_pass=False, accepted_quality_verdict="PROXY-PASS",
                accepted_proxy_sha256=["b"*64,"c"*64], private_manifest_sha256="d"*64,
                offset_convention=lm.OFFSET_CONVENTION,
                ledger=dict(completed_layers=42, completed_tensors=126, completed_canonical_payload_bytes=lm.CANONICAL_BYTES,
                            target_layers=42, target_tensors=126, canonical_payload_bytes=lm.CANONICAL_BYTES,
                            tp2_payload_bytes=lm.TP2_BYTES, per_node_tp2_payload_bytes=lm.TP2_BYTES//2), layers=layers)


def string(value):
    data = value.encode()
    return struct.pack("<Q",len(data))+data


def small_gguf(path, layer=3, metadata=None):
    """A small packed-layout fixture; never a production-size bank or quality claim."""
    records = metadata if metadata is not None else [("general.architecture",8,string("glm5-next")),
        ("general.alignment",4,struct.pack("<I",32)),
        ("tokenizer.ggml.tokens",9,struct.pack("<IQ",8,2)+string("abc")+string("def"))]
    header = b"GGUF"+struct.pack("<IQQ",3,3,len(records))
    for name,typ,encoded in records:
        header += string(name)+struct.pack("<I",typ)+encoded
    tensors, offsets, end = [], {}, 0
    for part in lm.PARTS:
        type_id = lm.recipe(layer,part)
        typ,block,size,extra = lm.TYPES[type_id]
        dims = [256,256,2]
        stride = dims[0]//block*size+extra
        item = dict(name=f"blk.{layer}.ffn_{part}_exps.weight",dims=dims,type=typ,type_id=type_id,
                    offset=end,nbytes=stride*256*2,row_stride=stride)
        offsets[part] = len(header)+len(string(item["name"]))+4+24
        header += string(item["name"])+struct.pack("<I3QIQ",3,*dims,type_id,end)
        end = (end+item["nbytes"]+31)//32*32
        tensors.append(item)
    start = (len(header)+31)//32*32
    data = header+b"\0"*(start-len(header))+b"\0"*end
    path.write_bytes(data)
    return dict(layer=layer,artifact=dict(path=path.name,sha256=hashlib.sha256(data).hexdigest(),bytes=len(data),data_start=start),
                tensors=tensors),offsets


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="lut-bank-test-")
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def test_complete_inventory_and_exact_tp2_row_scale_ledger(self):
        value = public_manifest()
        lm.validate_structure(value)
        self.assertEqual(sum(t["nbytes"] for row in value["layers"] for t in row["tensors"]),lm.CANONICAL_BYTES)
        self.assertEqual(lm.TP2_BYTES-lm.CANONICAL_BYTES,184025088)
        self.assertEqual([t["type"] for t in value["layers"][8]["tensors"]],["IQ4_KS","IQ4_KS","Q6_K"])
        path = self.root/"bank.json"; path.write_text(json.dumps(value))
        loaded = lm.load_manifest(path,check_files=False)
        self.assertEqual(loaded.sha256,hashlib.sha256(path.read_bytes()).hexdigest())
        with self.assertRaises(FileNotFoundError): lm.load_manifest(path)

    def test_rejects_incomplete_duplicate_wrong_recipe_or_geometry_and_private_fields(self):
        changes = [lambda v:v["layers"].pop(),lambda v:v["layers"].__setitem__(1,copy.deepcopy(v["layers"][0])),
            lambda v:v.__setitem__("state","PARTIAL"),lambda v:v.__setitem__("real_pass",True),
            lambda v:v.__setitem__("revision","0"*40),lambda v:v.__setitem__("imatrix_sha256","0"*64),
            lambda v:v["layers"][0]["tensors"][0].__setitem__("type_id",144),
            lambda v:v["layers"][0]["tensors"][2]["dims"].__setitem__(0,4096),
            lambda v:v["ledger"].__setitem__("target_layers",True),lambda v:v["layers"][0].__setitem__("layer",True),
            lambda v:v["layers"][0]["tensors"][0].__setitem__("offset",True),
            lambda v:v["layers"][0].__setitem__("conversion_context",{}),
            lambda v:v["accepted_proxy_sha256"].__setitem__(1,v["accepted_proxy_sha256"][0])]
        for index,change in enumerate(changes):
            value=public_manifest(); change(value)
            with self.subTest(index=index),self.assertRaises(ValueError): lm.validate_structure(value)

    def test_paths_ranges_strides_and_unbounded_padding_fail_closed(self):
        for path in ("../outside.gguf","/models/bank.gguf","nested/blk.03.lut.gguf","blk.04.lut.gguf","file:outside.gguf"):
            value=public_manifest(); value["layers"][0]["artifact"]["path"]=path
            with self.subTest(path=path),self.assertRaisesRegex(ValueError,"relative basename"): lm.validate_structure(value)
        for field,new in (("offset",1),("offset",2**63),("row_stride",1088),("nbytes",0)):
            value=public_manifest();value["layers"][0]["tensors"][2][field]=new
            with self.subTest(field=field,new=new),self.assertRaises(ValueError): lm.validate_structure(value)
        value=public_manifest();value["layers"][0]["tensors"][1]["offset"]=0
        with self.assertRaisesRegex(ValueError,"overlapping"): lm.validate_structure(value)
        value=public_manifest();value["layers"][0]["artifact"]["bytes"]+=lm.HEADER_LIMIT+65536*3
        with self.assertRaisesRegex(ValueError,"unbounded padding"): lm.validate_structure(value)

    def test_json_duplicate_nonfinite_oversized_and_symlink_rejected(self):
        path=self.root/"bank.json"
        for text,reason in (('{"schema":1,"schema":2}',"duplicate"),('{"x":NaN}',"nonfinite")):
            path.write_text(text)
            with self.assertRaisesRegex(ValueError,reason): lm.load_manifest(path)
        with path.open("wb") as f:f.truncate(lm.JSON_LIMIT+1)
        with self.assertRaisesRegex(ValueError,"bounded JSON"): lm.load_manifest(path)
        link=self.root/"linked.json";link.symlink_to(path)
        with self.assertRaisesRegex(ValueError,"symlinks"): lm.load_manifest(link)
        directory=self.root/"linked";directory.symlink_to(self.root,target_is_directory=True)
        with self.assertRaisesRegex(ValueError,"symlinks"): lm.load_manifest(directory/"bank.json")
        fifo=self.root/"pipe";os.mkfifo(fifo)
        with self.assertRaisesRegex(ValueError,"regular files"): lm.load_manifest(fifo)

    def test_streaming_artifact_verification_returns_stable_absolute_native_descriptors(self):
        for layer in (3,11,12,44):
            path=self.root/f"blk.{layer:02d}.lut.gguf";row,_=small_gguf(path,layer)
            ticks=[];native=lm.verify_artifact(path,row,lambda:ticks.append(True))
            self.assertEqual(native["model_layer"],layer)
            self.assertEqual(native["internal_li"],layer-3)
            self.assertEqual(native["identity"],lm.identity(path.stat()))
            self.assertTrue(Path(native["path"]).is_absolute())
            self.assertEqual([t["type_id"] for t in native["tensors"]],[lm.recipe(layer,p) for p in lm.PARTS])
            self.assertEqual(native["tensors"][0]["offset"],row["artifact"]["data_start"])
            self.assertGreater(len(ticks),0)
        path=self.root/"blk.03.lut.gguf";row,_=small_gguf(path)
        with path.open("r+b") as f:f.seek(-1,os.SEEK_END);f.write(b"x")
        with self.assertRaisesRegex(ValueError,"SHA256"):lm.verify_artifact(path,row)

    def test_mutated_fingerprint_and_manifest_descriptor_mismatch_rejected(self):
        path=self.root/"blk.03.lut.gguf";row,_=small_gguf(path)
        changed=copy.deepcopy(row);changed["tensors"][0]["dims"][2]+=1
        with self.assertRaisesRegex(ValueError,"descriptors"):lm.verify_artifact(path,changed)
        def touch():
            st=path.stat();os.utime(path,ns=(st.st_atime_ns,st.st_mtime_ns+1))
        with self.assertRaisesRegex(ValueError,"changed during"):lm.verify_artifact(path,row,touch)

    def test_header_skips_large_metadata_and_rejects_split_unknown_type_and_bounds(self):
        path=self.root/"small.gguf"
        metadata=[("general.architecture",8,string("glm5-next")),("large.metadata",8,string("x"*(2**20)))]
        row,offsets=small_gguf(path,metadata=metadata)
        class Recording(io.BytesIO):
            maximum=0
            def read(self,n=-1): self.maximum=max(self.maximum,n);return super().read(n)
        stream=Recording(path.read_bytes())
        start,tensors=lm.read_gguf_header(stream,row["artifact"]["bytes"])
        self.assertEqual(start,row["artifact"]["data_start"])
        self.assertLessEqual(stream.maximum,4096)
        for extra,reason in (([("split.count",4,struct.pack("<I",1))],"split"),
                ([("bad.array",9,struct.pack("<IQ",9,1))],"nested"),
                ([("bad.length",8,struct.pack("<Q",lm.HEADER_LIMIT+1))],"bound")):
            small_gguf(path,metadata=[("general.architecture",8,string("glm5-next"))]+extra)
            with path.open("rb") as f,self.assertRaisesRegex(ValueError,reason): lm.read_gguf_header(f,path.stat().st_size)
        row,offsets=small_gguf(path)
        with path.open("r+b") as f:f.seek(offsets["gate"]);f.write(struct.pack("<I",999))
        with path.open("rb") as f,self.assertRaisesRegex(ValueError,"unknown tensor type"):lm.read_gguf_header(f,path.stat().st_size)
        small_gguf(path)
        with path.open("rb") as f,self.assertRaisesRegex(ValueError,"exceeds file bounds"):lm.read_gguf_header(f,path.stat().st_size-10000)

    def test_offline_bank_verifies_all_files_sequentially_without_admitting_serving(self):
        bank=lm.BankManifest(self.root/"bank.json","f"*64,public_manifest())
        order=[]
        def verify(path,row,tick):order.append(row["layer"]);return {"model_layer":row["layer"]}
        with mock.patch.object(lm,"load_manifest",return_value=bank),mock.patch.object(lm,"verify_artifact",side_effect=verify):
            result=lm.verify_bank(bank.path)
        self.assertEqual(order,list(lm.LAYERS));self.assertEqual(len(result["layers"]),42)
        self.assertIs(result["real_pass"],False);self.assertIs(result["serving_admitted"],False)

    def test_env_defaults_and_cost_table_name_remain_separate(self):
        settings=lm.parse_env({"GLM53_CT_LUT":"0.2,0.3"})
        self.assertEqual((settings.format,settings.activation),("trellis","i16"))
        with mock.patch.object(lm,"load_manifest",side_effect=AssertionError("default must not inspect bank")):
            self.assertEqual(lm.preflight({}).format,"trellis")
        for env in ({"GLM53_CT_FORMAT":""},{"GLM53_CT_FORMAT":"other"},{"GLM53_CT_LUT_ACT":"fp16"},
                    {"GLM53_CT_FORMAT":"lut"}):
            with self.subTest(env=env),self.assertRaises(ValueError):lm.preflight(env)

    def test_lut_preflight_requires_owner_approved_real_quality_after_full_bank_structure(self):
        for activation in ("i16","q8"):
            env={"GLM53_CT_FORMAT":"lut","GLM53_CT_LUT_ACT":activation,"GLM53_CT_LUT_MANIFEST":"bank.json"}
            with mock.patch.object(lm,"load_manifest") as load:
                with self.assertRaisesRegex(ValueError,"real i16 report path is required"):lm.preflight(env)
                load.assert_called_once_with(Path("bank.json"))
        with mock.patch.object(lm,"load_manifest",side_effect=ValueError("full bank missing")):
            with self.assertRaisesRegex(ValueError,"full bank missing"):
                lm.preflight({"GLM53_CT_FORMAT":"lut","GLM53_CT_LUT_MANIFEST":"missing.json"})



def real_report(bank, activation="i16"):
    """Fictitious structural data for validator tests; no captured-model claim."""
    capture = dict(model_revision="332ab457b709b7ba30dd9a448be5de03b80a7ac9", serving_revision="1"*40,
                   exllamav3_version="1.5.1", **{key:"2"*64 for key in
                   ("model_ledger_sha256","runtime_ledger_sha256","corpus_sha256","stats_sha256",
                    "inputs_sha256","routes_sha256","trajectories_sha256")})
    replay = {key:"3"*64 for key in ("library_sha256","build_receipt_sha256","quality_checkpoint_sha256",
              "quality_replay_sha256","gguf_sha256","maintenance_guard_sha256")}
    replay.update(codec_revision="d1b493379387518d20c7bc89dab0ec1ce320f811",
                  kernels_cpp_sha256="3daf262cf111df1f2ac04fea05840b147e7efdcd0a7bae4666ecfb6f15362244",
                  kernels_h_sha256="dabec85fd112d544536856e04dc3c722495dd20749c91754cacdd58d9b506669")
    binding = dict(protocol=lm.REAL_PROTOCOL,bank_manifest_sha256=bank.sha256,source_revision=lm.SOURCE_REVISION,
                   imatrix_sha256=lm.IMATRIX_SHA256,capture_manifest_sha256="4"*64,capture=capture,replay=replay,
                   layers={str(row["layer"]):dict(q8_sha256="5"*64,ours_sha256=row["artifact"]["sha256"],unsloth_sha256="6"*64)
                           for row in bank.value["layers"] if row["layer"] in (3,11)})
    layers = {}
    pooled = {"ours":[],"unsloth":[]}
    for layer in ("3","11"):
        experts=[dict(expert=expert,routing_count=32,ours_sample_errors=[.01+expert*.00001]*32,
                      unsloth_sample_errors=[.02+expert*.00001]*32) for expert in range(32)]
        metrics={arm:lm.quality_metrics([(r[arm+"_sample_errors"],r["routing_count"]) for r in experts]) for arm in pooled}
        for arm in pooled:pooled[arm].extend((r[arm+"_sample_errors"],r["routing_count"]) for r in experts)
        layers[layer]=dict(coverage=dict(tokens=128,prompts=4,classes=["code","firmware","prose"],experts=32,
                                        route_occurrences=1024),experts=experts,**metrics,**{"pass":True},
                           tp2_full_down_max_relative_l2={"ours":1e-7,"unsloth":1e-7})
    return dict(schema="e1-real-quality/v1",state="REAL-PASS",complete=True,real_pass=True,synthetic=False,
                activation=activation,batch=2,act_limit=10,tp2="split-intermediate-gate-up/down-columns-before-quantization",
                binding=binding,binding_sha256=lm.json_digest(binding),layers=layers,
                aggregate={**{arm:lm.quality_metrics(values) for arm,values in pooled.items()},"pass":True})


class RealAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory(prefix="real-admission-test-")
        self.addCleanup(self.temp.cleanup)
        self.root=Path(self.temp.name)
        self.bank=lm.BankManifest(self.root/"bank.json","f"*64,public_manifest())

    def environment(self, activation="i16", report=None):
        report=real_report(self.bank) if report is None else report
        path=self.root/"real-i16.json";path.write_text(json.dumps(report,allow_nan=False))
        env=dict(GLM53_CT_FORMAT="lut",GLM53_CT_LUT_ACT=activation,GLM53_CT_LUT_MANIFEST=str(self.bank.path),
                 GLM53_CT_LUT_QUALITY=str(path),GLM53_CT_LUT_QUALITY_SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
        if activation=="q8":
            path=self.root/"real-q8.json";path.write_text(json.dumps(real_report(self.bank,"q8")))
            env.update(GLM53_CT_LUT_Q8_QUALITY=str(path),GLM53_CT_LUT_Q8_QUALITY_SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
        return env

    def admit(self, env):
        with mock.patch.object(lm,"load_manifest",return_value=self.bank):
            return lm.preflight(env,require_production=False)

    def test_real_i16_and_q8_quality_admit_but_production_remains_blocked(self):
        for activation in ("i16","q8"):
            env=self.environment(activation)
            settings=self.admit(env)
            self.assertIs(settings.admission["real_quality_admitted"],True)
            self.assertIs(settings.admission["production_ready"],False)
            self.assertEqual(set(settings.admission["report_sha256"]),{"i16","q8"} if activation=="q8" else {"i16"})
            with mock.patch.object(lm,"load_manifest",return_value=self.bank):
                with self.assertRaisesRegex(RuntimeError,"production LUT coordinator startup is unavailable"):
                    lm.preflight(env)

    def test_owner_approved_digest_is_mandatory_and_same_bytes_are_validated(self):
        for field,value in (("GLM53_CT_LUT_QUALITY_SHA256",None),("GLM53_CT_LUT_QUALITY_SHA256","A"*64),
                            ("GLM53_CT_LUT_QUALITY_SHA256","0"*64),("GLM53_CT_LUT_QUALITY",None)):
            env=self.environment()
            if value is None:env.pop(field)
            else:env[field]=value
            with self.subTest(field=field,value=value),self.assertRaises(ValueError):self.admit(env)
        env=self.environment();path=Path(env["GLM53_CT_LUT_QUALITY"]);path.write_bytes(path.read_bytes()+b" ")
        with self.assertRaisesRegex(ValueError,"owner-approved SHA256"):self.admit(env)
        env=self.environment();path=Path(env["GLM53_CT_LUT_QUALITY"]);link=self.root/"symlink.json";link.symlink_to(path)
        env["GLM53_CT_LUT_QUALITY"]=str(link)
        with self.assertRaisesRegex(ValueError,"symlinks"):self.admit(env)

    def test_proxy_incomplete_or_forged_positive_flags_never_admit(self):
        changes=[lambda v:v.__setitem__("schema","e1-proxy-result/v1"),lambda v:v.__setitem__("state","PROXY-PASS"),
                 lambda v:v.__setitem__("state","PASS"),lambda v:v.__setitem__("complete",1),
                 lambda v:v.__setitem__("real_pass",1),lambda v:v.__setitem__("synthetic",True),
                 lambda v:v.__setitem__("activation","q8"),lambda v:v.__setitem__("batch",True),
                 lambda v:v.__setitem__("batch",3),lambda v:v.__setitem__("act_limit",0),
                 lambda v:v.__setitem__("tp2","unsplit"),lambda v:v.__setitem__("proxy_pass",True),
                 lambda v:v["layers"].pop("11"),lambda v:v["aggregate"].__setitem__("pass",1),
                 lambda v:v["layers"]["3"].__setitem__("pass",1),
                 lambda v:v["layers"]["3"]["ours"].__setitem__("median_expert_nmse",0),
                 lambda v:v["aggregate"]["ours"].__setitem__("median_expert_nmse",0)]
        for index,change in enumerate(changes):
            report=real_report(self.bank);change(report)
            with self.subTest(index=index),self.assertRaises(ValueError):self.admit(self.environment(report=report))

    def test_transitive_binding_pins_and_bank_artifacts_must_match(self):
        fields=[("bank_manifest_sha256",), ("source_revision",), ("imatrix_sha256",), ("protocol",),
                ("capture_manifest_sha256",), ("capture","model_revision"), ("capture","serving_revision"),
                ("capture","exllamav3_version"), ("capture","inputs_sha256"), ("capture","trajectories_sha256"),
                ("replay","codec_revision"), ("replay","kernels_cpp_sha256"), ("replay","kernels_h_sha256"),
                ("replay","build_receipt_sha256"), ("layers","3","ours_sha256"), ("layers","11","unsloth_sha256")]
        for field in fields:
            report=real_report(self.bank);node=report["binding"]
            for key in field[:-1]:node=node[key]
            node[field[-1]]="wrong"
            report["binding_sha256"]=lm.json_digest(report["binding"])
            with self.subTest(field=field),self.assertRaises(ValueError):self.admit(self.environment(report=report))
        report=real_report(self.bank);report["binding"]["capture"]["corpus_sha256"]="7"*64
        with self.assertRaisesRegex(ValueError,"binding SHA256"):self.admit(self.environment(report=report))

    def test_raw_errors_coverage_and_failed_gates_are_recomputed(self):
        changes=[lambda r:r["tp2_full_down_max_relative_l2"].__setitem__("ours",1.1e-5),
                 lambda r:r["tp2_full_down_max_relative_l2"].__setitem__("unsloth",-1),
                 lambda r:r.pop("tp2_full_down_max_relative_l2"),
                 lambda r:r["coverage"].__setitem__("tokens",127),lambda r:r["coverage"].__setitem__("prompts",3),
                 lambda r:r["coverage"].__setitem__("experts",31),lambda r:r["coverage"].__setitem__("route_occurrences",1023),
                 lambda r:r["coverage"].__setitem__("classes",["prose","code"]),
                 lambda r:r["coverage"].__setitem__("classes",["prose","code","firmware","prose"]),
                 lambda r:r["experts"].__setitem__(1,copy.deepcopy(r["experts"][0])),
                 lambda r:r["experts"][0].__setitem__("expert",288),lambda r:r["experts"][0].__setitem__("routing_count",31),
                 lambda r:r["experts"][0]["ours_sample_errors"].pop(),
                 lambda r:r["experts"][0]["ours_sample_errors"].__setitem__(0,-1),
                 lambda r:r["experts"][0]["ours_sample_errors"].__setitem__(0,True)]
        for index,change in enumerate(changes):
            report=real_report(self.bank);change(report["layers"]["3"])
            with self.subTest(index=index),self.assertRaises(ValueError):self.admit(self.environment(report=report))
        report=real_report(self.bank)
        for row in report["layers"].values():
            for record in row["experts"]:record["ours_sample_errors"]=[.04]*record["routing_count"]
            row["ours"]=lm.quality_metrics([(r["ours_sample_errors"],r["routing_count"]) for r in row["experts"]])
        with self.assertRaisesRegex(ValueError,"per-layer quality gate failed"):self.admit(self.environment(report=report))

    def test_q8_needs_both_activations_and_identical_provenance(self):
        for field in ("GLM53_CT_LUT_QUALITY","GLM53_CT_LUT_QUALITY_SHA256","GLM53_CT_LUT_Q8_QUALITY","GLM53_CT_LUT_Q8_QUALITY_SHA256"):
            env=self.environment("q8");env.pop(field)
            with self.subTest(field=field),self.assertRaises(ValueError):self.admit(env)
        for change in (lambda r:r["binding"]["capture"].__setitem__("inputs_sha256","8"*64),
                       lambda r:r.__setitem__("act_limit",20),lambda r:r.__setitem__("activation","i16")):
            env=self.environment("q8");path=Path(env["GLM53_CT_LUT_Q8_QUALITY"]);report=json.loads(path.read_text());change(report)
            report["binding_sha256"]=lm.json_digest(report["binding"]);path.write_text(json.dumps(report))
            env["GLM53_CT_LUT_Q8_QUALITY_SHA256"]=hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):self.admit(env)

    def test_real_report_parser_rejects_duplicate_nonfinite_oversized_fifo_and_changed_identity(self):
        env=self.environment();path=Path(env["GLM53_CT_LUT_QUALITY"])
        for payload in (b'{"schema":1,"schema":2}',b'{"x":NaN}'):
            path.write_bytes(payload);env["GLM53_CT_LUT_QUALITY_SHA256"]=hashlib.sha256(payload).hexdigest()
            with self.assertRaises(ValueError):self.admit(env)
        path.write_bytes(b"x"*128)
        with mock.patch.object(lm,"REAL_JSON_LIMIT",64),self.assertRaisesRegex(ValueError,"bounded JSON"):self.admit(env)
        fifo=self.root/"fifo";os.mkfifo(fifo);env["GLM53_CT_LUT_QUALITY"]=str(fifo)
        with self.assertRaisesRegex(ValueError,"regular files"):self.admit(env)
        env=self.environment()
        with mock.patch.object(lm,"unchanged",side_effect=ValueError("changed during validation")):
            with self.assertRaisesRegex(ValueError,"changed during"):self.admit(env)

    def test_actual_model_limit_must_match_admission_and_can_differ_from_default(self):
        report=real_report(self.bank);report["act_limit"]=20
        settings=self.admit(self.environment(report=report))
        self.assertEqual(lm.validate_model_act_limit(settings,[SimpleNamespace(act_limit=20)]),20)
        for modules in ([],iter(()),[SimpleNamespace(act_limit=10)],[SimpleNamespace(act_limit=0)],
                        [SimpleNamespace(act_limit=float("nan"))],[SimpleNamespace()]):
            with self.subTest(modules=modules),self.assertRaises(ValueError):lm.validate_model_act_limit(settings,modules)

    def test_quality_cli_reports_admission_without_worker_or_framework_imports(self):
        env=self.environment()
        with mock.patch.dict(os.environ,env,clear=True),mock.patch.object(lm,"load_manifest",return_value=self.bank),mock.patch("sys.stdout",new_callable=io.StringIO) as output:
            self.assertEqual(lm.main(["--admit-quality"]),0)
        result=json.loads(output.getvalue())
        self.assertTrue(result["real_quality_admitted"])
        self.assertIs(result["production_ready"],False)


class IntegrationTests(unittest.TestCase):
    def serve_load(self,namespace):
        tree=ast.parse((ROOT/"glm53"/"serve.py").read_text())
        function=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="load")
        exec(compile(ast.Module(body=[function],type_ignores=[]),"serve.load", "exec"),namespace)
        return namespace["load"]

    def test_serve_preflight_runs_before_model_init_or_tier_install_without_torch(self):
        model_init=mock.Mock();argparse=mock.Mock()
        load=self.serve_load({"model_init":model_init,"argparse":argparse,"os":os})
        with mock.patch.dict(os.environ,{"GLM53_CT_FORMAT":"lut"},clear=True):
            with self.assertRaisesRegex(ValueError,"MANIFEST"):load()
        argparse.ArgumentParser.assert_not_called();model_init.init.assert_not_called()
        sentinel=RuntimeError("argument parsing reached")
        model_init.add_args.side_effect=sentinel
        with mock.patch.dict(os.environ,{},clear=True),mock.patch.object(lm,"load_manifest",side_effect=AssertionError("trellis read bank")):
            with self.assertRaisesRegex(RuntimeError,"argument parsing reached"):load()

    def test_serve_module_rejects_lut_before_torch_or_exllamav3_import(self):
        source=(ROOT/"glm53"/"serve.py").read_text()
        original_import=builtins.__import__
        imported=[]
        def guarded_import(name,*args,**kwargs):
            if name.split(".")[0] in ("torch","exllamav3"):
                imported.append(name)
                raise AssertionError("framework import reached")
            return original_import(name,*args,**kwargs)
        namespace={"__name__":"serve_import_fixture","__file__":str(ROOT/"glm53"/"serve.py")}
        with mock.patch.dict(os.environ,{"GLM53_CT_FORMAT":"lut"},clear=True),mock.patch.object(builtins,"__import__",side_effect=guarded_import):
            with self.assertRaisesRegex(ValueError,"MANIFEST"):exec(compile(source,"serve.import","exec"),dict(namespace))
        self.assertEqual(imported,[])
        env={"GLM53_CT_FORMAT":"lut","GLM53_CT_LUT_MANIFEST":"bank.json","GLM53_CT_LUT_QUALITY":"PASS.json"}
        with mock.patch.dict(os.environ,env,clear=True),mock.patch.object(lm,"load_manifest"),mock.patch.object(builtins,"__import__",side_effect=guarded_import):
            with self.assertRaisesRegex(ValueError,"owner-approved real i16 report"):
                exec(compile(source,"serve.import","exec"),dict(namespace))
        self.assertEqual(imported,[])
        with mock.patch.dict(os.environ,{},clear=True),mock.patch.object(lm,"load_manifest",side_effect=AssertionError("trellis read bank")),mock.patch.object(builtins,"__import__",side_effect=guarded_import):
            with self.assertRaisesRegex(AssertionError,"framework import reached"):
                exec(compile(source,"serve.import","exec"),dict(namespace))
        self.assertEqual(imported,["torch"])

    def test_real_admitted_quality_still_blocks_full_serve_import_before_frameworks(self):
        source=(ROOT/"glm53"/"serve.py").read_text()
        original_import=builtins.__import__
        def guarded_import(name,*args,**kwargs):
            if name.split(".")[0] in ("torch","exllamav3"):
                raise AssertionError("framework import reached")
            return original_import(name,*args,**kwargs)
        with tempfile.TemporaryDirectory(prefix="real-startup-test-") as directory:
            bank=lm.BankManifest(Path(directory)/"bank.json","f"*64,public_manifest())
            quality=Path(directory)/"quality.json";quality.write_text(json.dumps(real_report(bank)))
            env=dict(GLM53_CT_FORMAT="lut",GLM53_CT_LUT_MANIFEST=str(bank.path),GLM53_CT_LUT_QUALITY=str(quality),
                     GLM53_CT_LUT_QUALITY_SHA256=hashlib.sha256(quality.read_bytes()).hexdigest())
            namespace={"__name__":"serve_import_fixture","__file__":str(ROOT/"glm53"/"serve.py")}
            with mock.patch.dict(os.environ,env,clear=True),mock.patch.object(lm,"load_manifest",return_value=bank),mock.patch.object(builtins,"__import__",side_effect=guarded_import):
                with self.assertRaisesRegex(RuntimeError,"production LUT coordinator startup is unavailable"):
                    exec(compile(source,"serve.import","exec"),namespace)

    def test_cpu_wrap_and_start_cannot_build_or_allocate_before_admission(self):
        spec=importlib.util.spec_from_file_location("cpu_tier_fixture",ROOT/"glm53"/"cpu_tier.py")
        module=importlib.util.module_from_spec(spec)
        torch_stub=SimpleNamespace(inference_mode=lambda:lambda function:function)
        with mock.patch.dict(sys.modules,{"torch":torch_stub}),mock.patch.dict(os.environ,{},clear=True):
            spec.loader.exec_module(module)
        with mock.patch.object(module,"_build") as build,mock.patch.dict(os.environ,{"GLM53_CT_FORMAT":"lut"},clear=True):
            with self.assertRaisesRegex(ValueError,"MANIFEST"):module.wrap(None)
            with self.assertRaisesRegex(ValueError,"MANIFEST"):module.start()
            build.assert_not_called()
        with mock.patch.object(module,"_build",side_effect=RuntimeError("original build reached")),mock.patch.dict(os.environ,{},clear=True):
            with self.assertRaisesRegex(RuntimeError,"original build reached"):module.wrap(None)
        tree=ast.parse((ROOT/"glm53"/"cpu_tier.py").read_text())
        build=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="_build")
        strings={node.value for node in ast.walk(build) if isinstance(node,ast.Constant) and isinstance(node.value,str)}
        self.assertIn("ft_tier_ext.cpp",strings);self.assertIn("ft_tier_cu.cu",strings)
        self.assertFalse(any("lut" in item.lower() for item in strings))


if __name__=="__main__":unittest.main()
