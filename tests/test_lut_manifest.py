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

    def s1_report(self):
        report = real_report(self.bank)
        report.update(scope="s1-real-check-only", full_bank_bound=False, production_ready=False)
        b = report["binding"]
        del b["bank_manifest_sha256"]
        b.update(scope="s1-real-check-only", quant_manifest_sha256="7"*64, ud_manifest_sha256="8"*64)
        for layer in (3, 11):
            recipe = ",".join("^blk\\."+str(layer)+"\\.ffn_"+part+"_exps\\.weight$="+
                              {138:"iq3_k",144:"iq4_ks",14:"q6_K"}[lm.recipe(layer,part)] for part in lm.PARTS)
            b["layers"][str(layer)]["conversion_context"] = dict(imatrix=lm.IMATRIX_SHA256, layer=layer,
                metadata_source="a"*64, normalized_imatrix_sha256="b"*64, quantizer={"quantizer":"c"*64},
                recipe=recipe, revision=lm.SOURCE_REVISION, sources={"donor":"d"*64}, tools={"adapter":"e"*64})
        report["binding_sha256"] = lm.json_digest(b)
        return report

    def test_s1_real_report_binds_unchanged_tested_layers_into_full_bank(self):
        report = self.s1_report()
        self.assertTrue(lm.validate_real_report(report,self.bank,"i16")["real_pass"])
        mutations = [lambda b:b["layers"]["3"].__setitem__("ours_sha256","0"*64),
                     lambda b:b["layers"]["11"]["conversion_context"].__setitem__("recipe","wrong"),
                     lambda b:b["layers"]["3"]["conversion_context"].__setitem__("revision","0"*40),
                     lambda b:b.__setitem__("scope","proxy"),
                     lambda b:b.__setitem__("bank_manifest_sha256",self.bank.sha256),
                     lambda b:b.__setitem__("quant_manifest_sha256","bad")]
        for mutation in mutations:
            changed = self.s1_report(); mutation(changed["binding"])
            changed["binding_sha256"] = lm.json_digest(changed["binding"])
            with self.assertRaises(ValueError): lm.validate_real_report(changed,self.bank,"i16")
        with self.assertRaises(ValueError): lm.validate_real_report(report,self.bank,"q8")

    def test_real_quality_admits_but_measured_cost_gate_remains_separate(self):
        for activation in ("i16","q8"):
            env=self.environment(activation)
            settings=self.admit(env)
            self.assertIs(settings.admission["real_quality_admitted"],True)
            self.assertIs(settings.admission["production_ready"],False)
            self.assertEqual(set(settings.admission["report_sha256"]),{"i16","q8"} if activation=="q8" else {"i16"})
            env.update(GLM53_CPU_TIER="1",GLM53_EC="1",GLM53_ZC_VRAM="0")
            with mock.patch.object(lm,"load_manifest",return_value=self.bank):
                with self.assertRaisesRegex(ValueError,"measured LUT costs are a separate startup gate"):
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


def measured_costs(bank, activation="i16"):
    """Fictitious structural timing data; no performance measurement claim."""
    profiles = {}
    for index, (name, types) in enumerate(lm.COST_PROFILES.items()):
        profiles[name] = dict(types=types, fixture_sha256="7"*64, source_artifact_sha256="a"*64,
                             rows={str(m):{str(k):dict(median_ms=(index+1)*m+k*.1,
                             samples=3,raw_ms=[(index+1)*m+k*.1]*3) for k in range(9)} for m in range(1,5)})
    cpus=list(range(2,44))
    hardware=dict(cpus=cpus,topology={str(cpu):dict(package=str(int(cpu>=22)),core=str(cpu)) for cpu in cpus})
    return dict(schema="e1-lut-cost/v1",complete=True,activation=activation,bank_manifest_sha256=bank.sha256,
                measurement=dict(kind="hardware",treatment="clflush-owned-weights-before-each-call",act_limit=10,coordinator_cpu=0,hardware_sha256=lm.json_digest(hardware),hardware=hardware,cpus=cpus,build_sha256="9"*64,
                                 placement_verified=True,clock="perf_counter_ns",samples_per_cell=3),profiles=profiles)


class StartupTests(unittest.TestCase):
    environment = RealAdmissionTests.environment

    def setUp(self):
        RealAdmissionTests.setUp(self)
        lm._STARTUP_CACHE.clear()
        self.addCleanup(lm._STARTUP_CACHE.clear)

    def startup_environment(self, activation="i16"):
        env = self.environment(activation)
        path=self.root/"cost.json";path.write_text(json.dumps(measured_costs(self.bank,activation)))
        env.update(GLM53_CPU_TIER="1",GLM53_EC="1",GLM53_ZC_VRAM="0",GLM53_CT_LUT_COST=str(path),
                   GLM53_CT_LUT_COST_SHA256=hashlib.sha256(path.read_bytes()).hexdigest())
        return env

    def precompiled(self, build_sha="9"*64):
        return dict(modules={"ft_tier_host.so":dict(sha256=build_sha)})

    def stage(self):
        path=self.root/"fixture.gguf";row,_=small_gguf(path)
        return dict(manifest_sha256=self.bank.sha256,layers=[lm.verify_artifact(path,row)])

    def test_complete_measured_grid_and_profile_map(self):
        self.assertEqual([lm.cost_profile(layer) for layer in (3,11,12,13,44)],
                         ["ordinary","upgraded_gate","upgraded_down","ordinary","upgraded_down"])
        for activation in ("i16","q8"):
            self.assertIs(lm.validate_cost_table(measured_costs(self.bank,activation),self.bank,activation)["complete"],True)
        changes=[lambda v:v.__setitem__("complete",False),lambda v:v.__setitem__("activation","q8"),
                 lambda v:v.__setitem__("bank_manifest_sha256","0"*64),lambda v:v["profiles"].pop("upgraded_down"),
                 lambda v:v["measurement"].__setitem__("placement_verified",False),
                 lambda v:v["measurement"].__setitem__("kind","projection"),
                 lambda v:v["measurement"].pop("treatment"),
                 lambda v:v["measurement"].__setitem__("treatment","warm-repeated-fixture"),
                 lambda v:v["measurement"].__setitem__("treatment","clflush-after-timed-call"),
                 lambda v:v["measurement"].__setitem__("samples_per_cell",2),
                 lambda v:v["profiles"]["ordinary"]["rows"]["4"].pop("8"),
                 lambda v:v["profiles"]["ordinary"].__setitem__("source_artifact_sha256","0"*64),
                 lambda v:v["profiles"]["ordinary"]["rows"]["1"]["1"].__setitem__("median_ms",.1),
                 lambda v:v["profiles"]["ordinary"]["rows"]["1"]["1"]["raw_ms"].pop(),
                 lambda v:v["profiles"]["ordinary"]["rows"]["1"]["1"]["raw_ms"].__setitem__(0,float("nan"))]
        for index,change in enumerate(changes):
            value=measured_costs(self.bank);change(value)
            with self.subTest(index=index),self.assertRaises(ValueError):lm.validate_cost_table(value,self.bank,"i16")

    def test_s1_recipe_costs_bind_every_tensor_donor_to_full_bank(self):
        value = measured_costs(self.bank)
        value.update(bank_manifest_sha256=None,s1_manifest_sha256="1"*64)
        for name, profile in value["profiles"].items():
            layers = (11,11,11) if name=="upgraded_gate" else (3,3,11) if name=="upgraded_down" else (3,3,3)
            profile["source_parts"] = [dict(part=part,layer=layer,sha256="a"*64,type_id=typ)
                for part,layer,typ in zip(lm.PARTS,layers,profile["types"])]
        self.assertTrue(lm.validate_cost_table(value,self.bank,"i16")["complete"])
        for field, replacement in (("sha256","b"*64),("layer",12),("type_id",144),("part","gate")):
            changed=copy.deepcopy(value)
            changed["profiles"]["upgraded_down"]["source_parts"][2][field]=replacement
            with self.subTest(field=field),self.assertRaises(ValueError):lm.validate_cost_table(changed,self.bank,"i16")
        changed=copy.deepcopy(value);changed["profiles"]["ordinary"]["source_parts"].pop()
        with self.assertRaises(ValueError):lm.validate_cost_table(changed,self.bank,"i16")

    def test_cost_hardware_hash_and_exact_worker_topology_are_required(self):
        value=measured_costs(self.bank);cpus=value["measurement"]["cpus"]
        topology=value["measurement"]["hardware"]["topology"]
        def read(path):
            cpu=path.parent.parent.name[3:]
            row=topology.get(cpu,{"package":"0","core":"0"})
            return row["package" if path.name=="physical_package_id" else "core"]+"\n"
        with mock.patch.object(Path,"read_text",read):
            lm.validate_cost_hardware(value,cpus)
            with self.assertRaisesRegex(ValueError,"worker CPUs differ"):lm.validate_cost_hardware(value,cpus[::-1])
            with self.assertRaisesRegex(ValueError,"coordinator CPU differs"):lm.validate_cost_hardware(value,cpus,1)
        with mock.patch.object(Path,"read_text",return_value="999"):
            with self.assertRaisesRegex(ValueError,"topology differs"):lm.validate_cost_hardware(value,cpus)
        def shared_core(path):
            cpu=path.parent.parent.name[3:]
            row=topology["2"] if cpu=="0" else topology[cpu]
            return row["package" if path.name=="physical_package_id" else "core"]+"\n"
        with mock.patch.object(Path,"read_text",shared_core):
            with self.assertRaisesRegex(ValueError,"separate physical core"):lm.validate_cost_hardware(value,cpus,0)
        for coordinator in (True,-1,1024,2):
            broken=copy.deepcopy(value);broken["measurement"]["coordinator_cpu"]=coordinator
            with self.subTest(coordinator=coordinator),self.assertRaises(ValueError):lm.validate_cost_table(broken,self.bank,"i16")
        for change in (lambda v:v["measurement"].__setitem__("hardware_sha256","0"*64),
                       lambda v:v["measurement"].__setitem__("cpus",cpus[::-1]),
                       lambda v:v["measurement"]["hardware"]["topology"].pop("2")):
            broken=copy.deepcopy(value);change(broken)
            with self.assertRaises(ValueError):lm.validate_cost_table(broken,self.bank,"i16")

    def test_startup_hashes_once_then_rechecks_files_without_allocating_weights(self):
        env=self.startup_environment();stage=self.stage()
        with mock.patch.object(lm,"load_manifest",return_value=self.bank),mock.patch.object(lm,"verify_bank",return_value=stage) as verify,mock.patch.object(lm,"load_precompiled",return_value=self.precompiled()):
            first=lm.preflight(env);second=lm.preflight(env)
            self.assertEqual(first,second);self.assertIs(first.staged_bank,stage)
            self.assertTrue(first.admission["full_bank_verified"]);self.assertFalse(first.admission["production_ready"])
            verify.assert_called_once_with(self.bank.path)
            path=Path(stage["layers"][0]["path"])
            path.write_bytes(path.read_bytes()+b"x")
            with self.assertRaisesRegex(ValueError,"changed before native initialization"):lm.preflight(env)

    def test_measured_activation_limit_must_match_admitted_real_model_limit(self):
        env=self.startup_environment();path=Path(env["GLM53_CT_LUT_COST"])
        value=json.loads(path.read_text());value["measurement"]["act_limit"]=20
        path.write_text(json.dumps(value));env["GLM53_CT_LUT_COST_SHA256"]=hashlib.sha256(path.read_bytes()).hexdigest()
        with mock.patch.object(lm,"load_manifest",return_value=self.bank),mock.patch.object(lm,"verify_bank") as verify:
            with self.assertRaisesRegex(ValueError,"cost activation limit differs"):lm.preflight(env)
            verify.assert_not_called()
        for limit in (0,-1,True,float("inf")):
            value=measured_costs(self.bank);value["measurement"]["act_limit"]=limit
            with self.subTest(limit=limit),self.assertRaises(ValueError):lm.validate_cost_table(value,self.bank,"i16")

    def test_valid_startup_finishes_bank_and_cost_checks_before_framework_import(self):
        env=self.startup_environment();stage=self.stage()
        source=(ROOT/"glm53"/"serve.py").read_text()
        original_import=builtins.__import__;events=[]
        def guarded(name,*args,**kw):
            if name.split(".")[0] in ("torch","exllamav3"):
                events.append("framework");raise RuntimeError("framework reached after admission")
            return original_import(name,*args,**kw)
        def verify(path):events.append("bank_verified");return stage
        with mock.patch.dict(os.environ,env,clear=True),mock.patch.object(lm,"load_manifest",return_value=self.bank),mock.patch.object(lm,"load_precompiled",return_value=self.precompiled()),mock.patch.object(lm,"verify_bank",side_effect=verify),mock.patch.object(builtins,"__import__",side_effect=guarded):
            with self.assertRaisesRegex(RuntimeError,"framework reached after admission"):
                exec(compile(source,"serve.valid-import","exec"),{"__name__":"serve_fixture","__file__":str(ROOT/"glm53"/"serve.py")})
        self.assertEqual(events,["bank_verified","framework"])

    def test_startup_rejects_changed_bank_report_cost_and_unsupported_runtime_before_hashing(self):
        env=self.startup_environment()
        mutations=[("GLM53_CPU_TIER","0"),("GLM53_EC","0"),("GLM53_ZC_VRAM",""),
                   ("GLM53_K_FTSPLIT","2"),("GLM53_LA","yes"),("GLM53_K_HCFUSE","-1"),("GLM53_K_OVL","on"),
                   ("GLM53_CT_VALIDATE","1"),("GLM53_CT_LUT",".1,.2"),("GLM53_CT_MAXBSZ","5"),
                   ("GLM53_CT_MAXN","9"),("GLM53_CT_FORCE_N","9"),("GLM53_CT_TIMEOUT_S","inf"),("GLM53_CT_TIMEOUT_S","0"),
                   ("GLM53_CT_MAXN","7.5"),("GLM53_CT_B","nan"),("GLM53_CT_B","-1"),
                   ("GLM53_CT_LUT_COST_SHA256","0"*64)]
        for key,value in mutations:
            changed=dict(env);changed[key]=value
            with self.subTest(key=key),mock.patch.object(lm,"load_manifest",return_value=self.bank),mock.patch.object(lm,"verify_bank") as verify:
                with self.assertRaises(ValueError):lm.preflight(changed)
                verify.assert_not_called()
        with mock.patch.object(lm,"load_manifest",return_value=self.bank),mock.patch.object(lm,"verify_bank",return_value=dict(manifest_sha256="0"*64)),mock.patch.object(lm,"load_precompiled",return_value=self.precompiled()):
            with self.assertRaisesRegex(ValueError,"manifest changed"):lm.preflight(env)

    def test_batch_policy_rejects_invalid_mode_and_counts_before_bank_reads(self):
        env=dict(GLM53_CT_FORMAT="lut",GLM53_CT_LUT_MANIFEST="unopened.json")
        cases=[{"GLM53_CT_LUT_BATCH_POLICY":mode} for mode in ("", "disjoint", "BOUNDED")]
        cases += [dict(GLM53_CT_LUT_BATCH_POLICY="bounded",GLM53_CT_MAXN=value)
                  for value in ("33", "-1", "7.5", "nan", "inf")]
        cases += [dict(GLM53_CT_LUT_BATCH_POLICY="bounded",GLM53_CT_FORCE_N=value)
                  for value in ("9", "-2", "1.5")]
        with mock.patch.object(lm,"load_manifest") as load:
            for changed in cases:
                with self.subTest(changed=changed),self.assertRaises(ValueError):lm.preflight({**env,**changed})
            load.assert_not_called()
        self.assertEqual(lm.parse_env(env).batch_policy,"shared")
        self.assertEqual(lm.parse_env({**env,"GLM53_CT_LUT_BATCH_POLICY":"bounded","GLM53_CT_MAXN":"32"}).batch_policy,"bounded")

    def test_batch_policy_is_preserved_through_production_admission(self):
        env=dict(self.startup_environment(),GLM53_CT_LUT_BATCH_POLICY="bounded",GLM53_CT_MAXN="32")
        with mock.patch.object(lm,"load_manifest",return_value=self.bank),\
             mock.patch.object(lm,"verify_bank",return_value=self.stage()),\
             mock.patch.object(lm,"load_precompiled",return_value=self.precompiled()):
            self.assertEqual(lm.preflight(env).batch_policy,"bounded")

    def test_lut_admits_gpu_features_and_fast_split_only_with_its_guarded_module(self):
        env=dict(self.startup_environment(),GLM53_LA="1",GLM53_K_HCFUSE="1",GLM53_K_OVL="1")
        with_split=dict(modules={"ft_tier_host.so":dict(sha256="9"*64),lm.SPLIT_MODULE:dict(sha256="8"*64)})
        cases=[(env,self.precompiled(),"manifest changed"),
               (dict(env,GLM53_K_FTSPLIT="1"),self.precompiled(),"guarded fast split module"),
               (dict(env,GLM53_K_FTSPLIT="1"),with_split,"manifest changed")]
        for index,(changed,precompiled,message) in enumerate(cases):
            with self.subTest(index=index),mock.patch.object(lm,"load_manifest",return_value=self.bank),\
                 mock.patch.object(lm,"verify_bank",return_value=dict(manifest_sha256="0"*64)),\
                 mock.patch.object(lm,"load_precompiled",return_value=precompiled):
                with self.assertRaisesRegex(ValueError,message):lm.preflight(changed)

    def test_lut_fast_split_install_imports_only_the_admitted_precompiled_module(self):
        spec=importlib.util.spec_from_file_location("k_ftsplit_fixture",ROOT/"glm53"/"k_ftsplit.py")
        module=importlib.util.module_from_spec(spec)
        with mock.patch.dict(sys.modules,{"torch":SimpleNamespace()}):spec.loader.exec_module(module)
        stock=SimpleNamespace(ft_split=mock.Mock(),ft_combine=mock.Mock())
        tier=SimpleNamespace(S={"settings":SimpleNamespace(format="lut",precompiled=dict(modules={}))},_CU=stock)
        with mock.patch.dict(sys.modules,{"cpu_tier":tier,"lut_manifest":lm}):
            with self.assertRaisesRegex(ValueError,"guarded fast split module"):module.install()
            fast=SimpleNamespace(ft_split=mock.Mock(return_value="fast"))
            with mock.patch.object(module,"_load_lut",return_value=fast) as loader:
                module.install()
            loader.assert_called_once_with(tier.S["settings"])
            sel=SimpleNamespace(numel=lambda:8)
            self.assertEqual(tier._CU.ft_split(sel,1,2),"fast");fast.ft_split.assert_called_once_with(sel,1,2)
            self.assertIs(tier._CU.ft_combine,stock.ft_combine)

    def cpu_module(self, torch=None):
        spec=importlib.util.spec_from_file_location("cpu_tier_startup_fixture",ROOT/"glm53"/"cpu_tier.py")
        module=importlib.util.module_from_spec(spec)
        torch=torch or SimpleNamespace(inference_mode=lambda:lambda function:function)
        with mock.patch.dict(sys.modules,{"torch":torch}):spec.loader.exec_module(module)
        return module

    def test_native_lut_start_uses_full_descriptors_and_model_limit_without_trellis_registration(self):
        job=type("Job",(),{"receive_sample":lambda self,*args,**kwargs:"token"});fake=SimpleNamespace(Job=job)
        patcher=mock.patch.dict(sys.modules,{"exllamav3":SimpleNamespace(),"exllamav3.generator":SimpleNamespace(job=fake),"exllamav3.generator.job":fake})
        patcher.start();self.addCleanup(patcher.stop)
        env=self.startup_environment();stage=self.stage()
        binary=self.root/"host.so";binary.write_bytes(b"structural native-build fixture")
        path=Path(env["GLM53_CT_LUT_COST"]);value=json.loads(path.read_text())
        value["measurement"]["build_sha256"]=hashlib.sha256(binary.read_bytes()).hexdigest()
        path.write_text(json.dumps(value));env["GLM53_CT_LUT_COST_SHA256"]=hashlib.sha256(path.read_bytes()).hexdigest()
        with mock.patch.object(lm,"load_manifest",return_value=self.bank),mock.patch.object(lm,"verify_bank",return_value=stage),mock.patch.object(lm,"load_precompiled",return_value=self.precompiled(value["measurement"]["build_sha256"])):settings=lm.preflight(env)
        module=self.cpu_module();host=mock.Mock();host.__file__=str(binary);module._H=host
        mods=[SimpleNamespace(key=f"model.layers.{layer}.mlp",act_limit=10,hidden_size=4096,intermediate_size=2048,
                              num_local_experts=None,num_experts=288,routing_first=0,_ec=(object(),li))
              for li,layer in enumerate(lm.LAYERS)]
        module.S.update(settings=settings,mods=mods,timeout_ns=2000000000,ctrl=object(),hx=object(),picks=object(),hout=object())
        with mock.patch.dict(os.environ,env,clear=True),mock.patch.object(module,"preflight",return_value=settings),mock.patch.object(module,"_cpus",return_value=list(range(2,44))),mock.patch.object(os,"sched_getaffinity",return_value=set(range(44))),mock.patch.object(lm,"validate_cost_hardware") as hardware,mock.patch("atexit.register") as register:
            module.start()
            hardware.assert_called_once_with(settings.costs,list(range(2,44)),0)
            register.assert_called_once_with(host.tier_stop)
        args=host.tier_init_lut.call_args.args
        self.assertEqual(args[:3],(list(range(2,44)),"i16",stage["layers"]))
        self.assertEqual(args[-2:],(10,2000));host.tier_start.assert_called_once_with(0)
        host.tier_init.assert_not_called();host.tier_add_layer.assert_not_called();host.tier_set_act_limit.assert_not_called()
        self.assertTrue(module.S["started"]);self.assertFalse(module.S["selftest"]["trellis_correctness_comparison"])
        self.assertTrue(job._glm53_lut_token_check)   # the per-token LUT check is installed at LUT start
        host.reset_mock();module.S["started"]=False
        def remote(modules):module.S["rt"]={"backend":"fixture"}
        with mock.patch.dict(os.environ,{**env,"GLM53_RT":"configured"},clear=True),mock.patch.object(module,"_cpus",return_value=list(range(2,44))),mock.patch.object(os,"sched_getaffinity",return_value=set(range(44))),mock.patch.object(module,"_rt_enable",side_effect=remote) as enable,mock.patch.object(module,"_rt_auto_b_start") as auto,mock.patch.object(lm,"validate_cost_hardware"),mock.patch("atexit.register"):
            module._start_lut(settings)
            enable.assert_called_once_with(mods);auto.assert_called_once_with()
        with self.assertRaisesRegex(RuntimeError,"trellis format difference"):module.selftest()
        mods[5].act_limit=20
        with self.assertRaisesRegex(ValueError,"activation limit"):module._start_lut(settings)
        self.assertEqual(host.tier_init_lut.call_count,1)

    def test_lut_remote_plan_bounds_absolute_deadline_payload_and_resolution(self):
        import socket
        module=self.cpu_module();module.S["settings"]=SimpleNamespace(format="lut")
        endpoint=".".join(map(str,(127,0,0,1)))
        class Socket:
            def __init__(self,payload,chunk=65536):self.payload=payload;self.chunk=chunk;self.calls=0
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def settimeout(self,value):self.timeout=value
            def connect(self,address):self.address=address
            def sendall(self,data):self.sent=data
            def recv(self,count):
                self.calls+=1;size=min(count,self.chunk);value=self.payload[:size];self.payload=self.payload[size:];return value
        body=json.dumps({"layers":{"3":[0,1]}}).encode()
        frame=b"GRTC"+struct.pack("<I",len(body))+body
        stream=Socket(frame,chunk=1)
        with mock.patch.object(socket,"socket",return_value=stream),mock.patch.object(module.time,"monotonic",return_value=0):
            self.assertEqual(module._rt_plan(endpoint,1,1),{"layers":{"3":[0,1]}})
            self.assertEqual(stream.address,(endpoint,1))
        stream=Socket(frame,chunk=1);clock=iter(i*.2 for i in range(100))
        with mock.patch.object(socket,"socket",return_value=stream),mock.patch.object(module.time,"monotonic",side_effect=lambda:next(clock)):
            with self.assertRaisesRegex(TimeoutError,"absolute deadline"):module._rt_plan(endpoint,1,1)
        stream=Socket(b"GRTC"+struct.pack("<I",2**20+1))
        with mock.patch.object(socket,"socket",return_value=stream),mock.patch.object(module.time,"monotonic",return_value=0):
            with self.assertRaisesRegex(ValueError,"bounded JSON"):module._rt_plan(endpoint,1,1)
        with mock.patch.object(socket,"socket") as network:
            with self.assertRaises(ValueError):module._rt_plan("hostname",1,1)
            network.assert_not_called()
        module._H=mock.Mock()
        with mock.patch.dict(os.environ,{"GLM53_RT":endpoint+":1"},clear=True),mock.patch.object(module,"_rt_plan",side_effect=TimeoutError("offline")):
            module._rt_enable([])
        self.assertEqual(module.S["rt_plan_error"],"offline");module._H.tier_remote.assert_not_called()

    def test_remote_policy_updates_preserve_all_measured_profile_tails(self):
        module=self.cpu_module()
        plans={m:[0.0]*12+[m+k/10 for k in range(8)] for m in range(1,5)}
        tails={m:plan[12:] for m,plan in plans.items()}
        module.S.update(settings=SimpleNamespace(format="lut"),pol=[0.0]*20,mods=[SimpleNamespace(_ct_pol=plans)])
        module._write_policy(3,.15);module._write_policy(8,1.0)
        for m,plan in plans.items():
            self.assertEqual(plan[3],.15);self.assertEqual(plan[8],1.0);self.assertEqual(plan[12:],tails[m])

    def test_mixed_format_remote_difference_is_labeled_as_a_diagnostic(self):
        module=self.cpu_module();module.S.update(rt={"backend":"fixture"},settings=SimpleNamespace(format="lut"))
        module._H=SimpleNamespace(tier_remote_stats=lambda:[1,1,1,0,0,1,0,0,0,0,0,1,123,0])
        result=module._rt_summary()
        self.assertNotIn("check_rel_rms",result);self.assertAlmostEqual(result["mixed_format_rel_rms"],123/1e9)
        self.assertFalse(result["correctness_gate"])
        module.S["settings"]=SimpleNamespace(format="trellis")
        self.assertIn("check_rel_rms",module._rt_summary())

    def test_lut_failure_is_raised_once_per_token_without_a_stream_sync(self):
        events=[]
        stream=SimpleNamespace(synchronize=lambda:events.append("stream_sync"))
        torch=SimpleNamespace(inference_mode=lambda:lambda function:function,cuda=SimpleNamespace(current_stream=lambda dev:stream))
        module=self.cpu_module(torch);module.S.update(dev="cuda:0",started=True)
        def failure():events.append("check");raise RuntimeError("native failed")
        module._H=SimpleNamespace(tier_lut_check=failure,tier_lut_abort=lambda:events.append("abort"))
        with self.assertRaisesRegex(RuntimeError,"native failed"):module._lut_token_check()
        # The caller (iterate_gen / lookahead) has synchronized the token; the check itself must never sync.
        self.assertEqual(events,["check","abort"]);self.assertFalse(module.S["started"])
        self.assertEqual(module.S["lut_error"],"native failed")

    def test_lut_token_hook_refuses_the_token_before_delivery(self):
        module=self.cpu_module();module.S.update(started=True)
        delivered=[]
        class Job:
            def receive_sample(self,*args,**kwargs):delivered.append((args,kwargs));return "token"
        fake=SimpleNamespace(Job=Job)
        with mock.patch.dict(sys.modules,{"exllamav3":SimpleNamespace(),"exllamav3.generator":SimpleNamespace(job=fake),"exllamav3.generator.job":fake}):
            module._H=SimpleNamespace(tier_lut_check=lambda:None,tier_lut_abort=lambda:None)
            module._install_lut_token_hook();module._install_lut_token_hook()   # idempotent: wrapped once
            self.assertEqual(Job().receive_sample(1,k=2),"token");self.assertEqual(delivered,[((1,),{"k":2})])
            def failure():raise RuntimeError("combine timeout")
            module._H=SimpleNamespace(tier_lut_check=failure,tier_lut_abort=lambda:None)
            with self.assertRaisesRegex(RuntimeError,"combine timeout"):Job().receive_sample(3)
            self.assertEqual(len(delivered),1)   # the failed token never reached the original receive_sample

    def test_lut_combine_hook_no_longer_syncs_per_layer(self):
        tree=ast.parse((ROOT/"glm53"/"cpu_tier.py").read_text())
        wrap=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="wrap")
        comb=next(node for node in ast.walk(wrap) if isinstance(node,ast.FunctionDef) and node.name=="comb")
        names={n.attr if isinstance(n,ast.Attribute) else n.id for n in ast.walk(comb) if isinstance(n,(ast.Attribute,ast.Name))}
        self.assertNotIn("synchronize",names);self.assertNotIn("_lut_token_check",names);self.assertNotIn("_lut_check",names)

    def test_failed_lut_tier_never_falls_back_to_gpu_routing(self):
        module=self.cpu_module();module.S.update(dev="cuda:0",started=False,lut_error="native failed")
        tree=ast.parse((ROOT/"glm53"/"cpu_tier.py").read_text())
        wrap=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="wrap")
        routed=next(node for node in ast.walk(wrap) if isinstance(node,ast.FunctionDef) and node.name=="routed")
        original=mock.Mock(side_effect=AssertionError("failed tier must not fall back to GPU routing"))
        namespace={"S":module.S,"settings":SimpleNamespace(format="lut"),"orig":original,"m":object()}
        exec(compile(ast.Module(body=[routed],type_ignores=[]),"cpu_tier.routed","exec"),namespace)
        for batch in (1,4,4096):
            with self.subTest(batch=batch),self.assertRaisesRegex(RuntimeError,"restart required"):
                namespace["routed"](batch,{},None,{})
        original.assert_not_called()

    def precompile_receipt(self):
        root=ROOT/"kernels"/"cpu_avx2"
        sources={"kernels/cpu_avx2/"+name:hashlib.sha256((root/name).read_bytes()).hexdigest()
                 for name in ("ft_tier_ext.cpp","ft_tier_cu.cu","ft_core.h","ft_lut.cpp","ft_lut.h","lut/kernels.cpp","lut/kernels.h")}
        outputs={}
        for name in ("ft_tier_host.so","ft_tier_cu.so"):
            file=self.root/name;file.write_bytes(("structural binary fixture "+name).encode())
            outputs[name]=hashlib.sha256(file.read_bytes()).hexdigest()
        return dict(schema="e1-precompile/v1",variant="lut",state="COMPILED_NO_GPU_CONTEXT",cuda_initialized=False,
                    source_revision="1"*40,torch_version="fixture",flags=["-DFT_ENABLE_LUT=1"],cuda_flags=["-DFT_ENABLE_LUT=1"],
                    source_sha256=sources,outputs=outputs,build_directory=str(self.root))

    def write_precompile(self, receipt):
        path=self.root/"reviewed-precompile.json";path.write_text(json.dumps(receipt))
        return path,hashlib.sha256(path.read_bytes()).hexdigest()

    def test_precompile_binds_both_guarded_modules_source_revision_and_torch_without_framework_import(self):
        receipt=self.precompile_receipt();path,sha=self.write_precompile(receipt)
        with mock.patch.object(lm.importlib.metadata,"version",return_value="fixture"),mock.patch.object(lm.subprocess,"check_output",return_value="1"*40+"\n"):
            result=lm.load_precompiled(path,sha)
            self.assertEqual(set(result["modules"]),{"ft_tier_host.so","ft_tier_cu.so"})
            self.assertEqual(result["modules"]["ft_tier_host.so"]["identity"],lm.identity((self.root/"ft_tier_host.so").stat()))
            mutations=[lambda r:r.__setitem__("variant","trellis"),lambda r:r.__setitem__("cuda_initialized",True),
                       lambda r:r.__setitem__("cuda_flags",["-O3"]),lambda r:r.__setitem__("flags",["-O3"]),
                       lambda r:r.__setitem__("torch_version","wrong"),lambda r:r.__setitem__("source_revision","0"*40),
                       lambda r:r["source_sha256"].__setitem__("kernels/cpu_avx2/ft_lut.cpp","0"*64),
                       lambda r:r["source_sha256"].pop("kernels/cpu_avx2/ft_tier_cu.cu"),
                       lambda r:r["outputs"].__setitem__("ft_tier_cu.so","0"*64),
                       lambda r:r.__setitem__("build_directory",str(self.root/"other"))]
            for index,change in enumerate(mutations):
                changed=copy.deepcopy(receipt);change(changed);changed_path,changed_sha=self.write_precompile(changed)
                with self.subTest(index=index),self.assertRaises(ValueError):lm.load_precompiled(changed_path,changed_sha)
            path,sha=self.write_precompile(receipt)
            (self.root/"ft_tier_cu.so").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"module SHA256"):lm.load_precompiled(path,sha)
        with self.assertRaisesRegex(ValueError,"PRECOMPILE"):lm.load_precompiled(None,None)

    def test_precompile_admits_the_guarded_fast_split_module_only_with_its_bound_source(self):
        receipt=self.precompile_receipt()
        split=ROOT/"kernels"/"k110"/"ft_split_fast.cu"
        receipt["source_sha256"][lm.SPLIT_SOURCE]=hashlib.sha256(split.read_bytes()).hexdigest()
        receipt["split_cuda_flags"]=["-O3","-DFT_ENABLE_LUT=1"]
        file=self.root/lm.SPLIT_MODULE;file.write_bytes(b"structural binary fixture split")
        receipt["outputs"][lm.SPLIT_MODULE]=hashlib.sha256(file.read_bytes()).hexdigest()
        with mock.patch.object(lm.importlib.metadata,"version",return_value="fixture"),mock.patch.object(lm.subprocess,"check_output",return_value="1"*40+"\n"):
            path,sha=self.write_precompile(receipt)
            self.assertEqual(set(lm.load_precompiled(path,sha)["modules"]),{"ft_tier_host.so","ft_tier_cu.so",lm.SPLIT_MODULE})
            mutations=[lambda r:r.__setitem__("split_cuda_flags",["-O3"]),lambda r:r.pop("split_cuda_flags"),
                       lambda r:r["source_sha256"].pop(lm.SPLIT_SOURCE),
                       lambda r:r["source_sha256"].__setitem__(lm.SPLIT_SOURCE,"0"*64),
                       lambda r:r["source_sha256"].__setitem__("kernels/k110/other.cu","0"*64),
                       lambda r:r["outputs"].__setitem__("other.so","0"*64)]
            for index,change in enumerate(mutations):
                changed=copy.deepcopy(receipt);change(changed);changed_path,changed_sha=self.write_precompile(changed)
                with self.subTest(index=index),self.assertRaises(ValueError):lm.load_precompiled(changed_path,changed_sha)

    def test_precompile_metadata_keeps_receipt_local_suffix_for_exact_runtime_check(self):
        receipt=self.precompile_receipt()
        receipt["torch_version"]="fixture+vendor"
        receipt["torch_metadata_version"]="untrusted-observation"
        path,sha=self.write_precompile(receipt)
        with mock.patch.object(lm.subprocess,"check_output",return_value="1"*40+"\n"):
            with mock.patch.object(lm.importlib.metadata,"version",return_value="fixture"):
                result=lm.load_precompiled(path,sha)
                self.assertEqual(result["receipt"]["torch_version"],"fixture+vendor")
            for version in ("other","fixture+vendor","fixture+other"):
                with self.subTest(version=version),mock.patch.object(lm.importlib.metadata,"version",return_value=version):
                    with self.assertRaisesRegex(ValueError,"Torch metadata"):lm.load_precompiled(path,sha)
        entries={name:dict(path=str(self.root/name),identity=lm.identity((self.root/name).stat()))
                 for name in receipt["outputs"]}
        for version in ("fixture","fixture+other"):
            module=self.cpu_module(SimpleNamespace(inference_mode=lambda:lambda function:function,__version__=version))
            module.S["settings"]=SimpleNamespace(precompiled=dict(receipt=receipt,modules=entries))
            with self.subTest(runtime=version),self.assertRaisesRegex(ValueError,"Torch version mismatch"):
                module._build_lut()

    def test_lut_load_imports_exact_reviewed_modules_without_jit_and_rechecks_identity(self):
        module=self.cpu_module(SimpleNamespace(inference_mode=lambda:lambda function:function,__version__="fixture"))
        receipt=self.precompile_receipt()
        entries={name:dict(path=str(self.root/name),identity=lm.identity((self.root/name).stat())) for name in receipt["outputs"]}
        settings=SimpleNamespace(precompiled=dict(receipt=receipt,modules=entries))
        module.S["settings"]=settings
        specs=[]
        def spec(name,path):
            specs.append((name,path))
            return SimpleNamespace(name=name,path=path,loader=SimpleNamespace(exec_module=lambda loaded:None))
        def loaded(spec):return SimpleNamespace(__file__=str(spec.path))
        with mock.patch.object(module.importlib.util,"spec_from_file_location",side_effect=spec),mock.patch.object(module.importlib.util,"module_from_spec",side_effect=loaded):
            module._build_lut()
            self.assertEqual([name for name,path in specs],["ft_tier_host","ft_tier_cu"])
            self.assertEqual(module._H.__file__,str(self.root/"ft_tier_host.so"))
            (self.root/"ft_tier_host.so").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError,"changed before import"):module._build_lut()

    def policy_torch(self):
        class Tensor:
            def __init__(self,values):self.values=list(values)
            def clone(self):return Tensor(self.values)
            def tolist(self):return list(self.values)
            def __getitem__(self,key):return self.values[key]
            def __setitem__(self,key,value):self.values[key]=value.values if isinstance(value,Tensor) else value
        return SimpleNamespace(inference_mode=lambda:lambda function:function,tensor=lambda values,**kw:Tensor(values),float32=object())

    def test_measured_policy_selects_profile_and_m_without_extra_token_or_k_extrapolation(self):
        module=self.cpu_module(self.policy_torch());settings=SimpleNamespace(costs=measured_costs(self.bank))
        with mock.patch.dict(os.environ,{"GLM53_CT_MAXN":"8"},clear=True):
            plans=module._lut_policy(settings,11,"cpu")
        self.assertEqual(set(plans),{1,2,3,4});self.assertEqual(plans[4][12:20],[8+k*.1 for k in range(1,9)])
        self.assertEqual(plans[4][4],0)
        with mock.patch.dict(os.environ,{"GLM53_CT_MAXN":"9"},clear=True),self.assertRaisesRegex(ValueError,"MAXN"):module._lut_policy(settings,3,"cpu")

    def test_shared_defaults_are_float32_byte_equivalent_and_bounded_preserves_m1(self):
        module=self.cpu_module(self.policy_torch());costs=measured_costs(self.bank)
        for layer in (3,11,12):
            with self.subTest(layer=layer),mock.patch.dict(os.environ,{"GLM53_CT_MAXN":"8","GLM53_CT_B":".31"},clear=True):
                base=module._policy("cpu");base[4]=0.0
                shared=module._lut_policy(SimpleNamespace(costs=costs,batch_policy="shared"),layer,"cpu")
                bounded=module._lut_policy(SimpleNamespace(costs=costs,batch_policy="bounded"),layer,"cpu")
                for m in range(1,5):
                    expected=base.tolist()
                    expected[12:20]=[costs["profiles"][lm.cost_profile(layer)]["rows"][str(m)][str(k)]["median_ms"] for k in range(1,9)]
                    self.assertEqual(struct.pack("20f",*shared[m].tolist()),struct.pack("20f",*expected))
                self.assertEqual(struct.pack("20f",*bounded[1].tolist()),struct.pack("20f",*shared[1].tolist()))

    def test_bounded_policy_caps_union_by_m_and_derives_each_profile_tail(self):
        module=self.cpu_module(self.policy_torch());settings=SimpleNamespace(costs=measured_costs(self.bank),batch_policy="bounded")
        # A nonlinear structural fixture makes a middle K determine the slope.
        settings.costs["profiles"]["ordinary"]["rows"]["2"]["3"]["median_ms"]=30.0
        for requested in (0,5,8,12,24,32):
            for layer in (3,11,12):
                with self.subTest(requested=requested,layer=layer),mock.patch.dict(os.environ,{"GLM53_CT_MAXN":str(requested)},clear=True):
                    plans=module._lut_policy(settings,layer,"cpu")
                for m,plan in plans.items():
                    row=settings.costs["profiles"][lm.cost_profile(layer)]["rows"][str(m)]
                    self.assertEqual(plan[5],min(requested,8*m))
                    self.assertEqual(plan[4],0)
                    self.assertEqual(plan[12:20],[row[str(k)]["median_ms"] for k in range(1,9)])
                    self.assertEqual(plan[3],.104 if m==1 else max(row[str(k)]["median_ms"]/k for k in range(1,9)))

    def test_bounded_runtime_and_remote_updates_keep_caps_and_derived_tails(self):
        torch=self.policy_torch();module=self.cpu_module(torch)
        settings=SimpleNamespace(format="lut",costs=measured_costs(self.bank),batch_policy="bounded")
        with mock.patch.dict(os.environ,{"GLM53_CT_MAXN":"32"},clear=True):
            mods=[SimpleNamespace(key=f"model.layers.{layer}.mlp",_ct_pol=module._lut_policy(settings,layer,"cpu")) for layer in (3,11,12)]
            module.S.update(settings=settings,pol=module._policy("cpu"),mods=mods)
        tails=[[plan[3] for m,plan in mod._ct_pol.items()] for mod in mods]
        rows=[[plan[12:20] for plan in mod._ct_pol.values()] for mod in mods]
        module._H=SimpleNamespace(tier_set_e6=lambda *args:[0,0,0])
        with mock.patch.dict(sys.modules,{"k_overlap":SimpleNamespace(set_after=lambda value:False)}):
            result=module.set_policy(maxn=24,b=.25,b_rt=.15)
            self.assertEqual(result["lut_batch_policy"]["maxn_by_m"],{"1":8,"2":16,"3":24,"4":24})
            self.assertIn("not a proven upper bound",result["lut_batch_policy"]["tail_model"])
            self.assertEqual(set(result["lut_batch_policy"]["tail_ms_per_expert_by_profile"]),set(lm.COST_PROFILES))
            module._write_policy(3,.15)  # The remote auto-B thread uses this same entry point.
            for index,mod in enumerate(mods):
                self.assertEqual(mod._ct_pol[1][3],.15)
                self.assertEqual([plan[3] for m,plan in mod._ct_pol.items() if m>1],tails[index][1:])
                self.assertEqual([plan[12:20] for plan in mod._ct_pol.values()],rows[index])
            result=module.set_policy(maxn=5,force_n=8)
            self.assertEqual(result["lut_batch_policy"]["maxn_by_m"],dict.fromkeys(("1","2","3","4"),5))
            before=module.S["pol"].tolist()
            for change in (dict(maxn=33),dict(maxn=7.5),dict(force_n=9),dict(force_n=-1.5),dict(maxn=float("nan"))):
                with self.subTest(change=change),self.assertRaises(ValueError):module.set_policy(**change)
                self.assertEqual(module.S["pol"].tolist(),before)

    def test_policy_diagnostics_use_cpu_metadata_without_reading_layer_tensors(self):
        class NoDeviceRead:
            def __getitem__(self,key):raise AssertionError("unexpected device scalar read")
            def tolist(self):raise AssertionError("unexpected device tensor copy")
            def items(self):raise AssertionError("unexpected per-layer plan access")
        module=self.cpu_module(self.policy_torch())
        costs=measured_costs(self.bank)
        module.S.update(settings=SimpleNamespace(format="lut",costs=costs,batch_policy="bounded"),
                        pol=NoDeviceRead(),mods=[SimpleNamespace(_ct_pol=NoDeviceRead())])
        result=module._lut_policy_summary(24)
        self.assertEqual(result["maxn_by_m"],{"1":8,"2":16,"3":24,"4":24})
        for name,profile in costs["profiles"].items():
            self.assertEqual(result["tail_ms_per_expert_by_profile"][name],
                             {m:max(cells[str(k)]["median_ms"]/k for k in range(1,9))
                              for m,cells in profile["rows"].items() if int(m)>1})


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

    def test_real_admitted_quality_without_costs_blocks_serve_before_frameworks(self):
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
                     GLM53_CT_LUT_QUALITY_SHA256=hashlib.sha256(quality.read_bytes()).hexdigest(),
                     GLM53_CPU_TIER="1",GLM53_EC="1",GLM53_ZC_VRAM="0")
            namespace={"__name__":"serve_import_fixture","__file__":str(ROOT/"glm53"/"serve.py")}
            with mock.patch.dict(os.environ,env,clear=True),mock.patch.object(lm,"load_manifest",return_value=bank),mock.patch.object(builtins,"__import__",side_effect=guarded_import):
                with self.assertRaisesRegex(ValueError,"measured LUT costs are a separate startup gate"):
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
        self.assertNotIn("-DFT_ENABLE_LUT=1",strings)
        self.assertNotIn("ft_lut.cpp",strings)
        lut_build=next(node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name=="_build_lut")
        lut_strings={node.value for node in ast.walk(lut_build) if isinstance(node,ast.Constant) and isinstance(node.value,str)}
        self.assertIn("ft_tier_host",lut_strings);self.assertIn("ft_tier_cu",lut_strings)
        self.assertNotIn("-O3",lut_strings)  # Production LUT loads reviewed artifacts without JIT.


if __name__=="__main__":unittest.main()
