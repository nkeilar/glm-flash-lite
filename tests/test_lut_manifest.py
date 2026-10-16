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

    def test_lut_preflight_validates_structure_then_refuses_every_quality_value_without_hashing(self):
        for activation in ("i16","q8"):
            for quality in (None,"PASS.json","PROXY-PASS.json"):
                env={"GLM53_CT_FORMAT":"lut","GLM53_CT_LUT_ACT":activation,"GLM53_CT_LUT_MANIFEST":"bank.json"}
                if quality:env["GLM53_CT_LUT_QUALITY"]=quality
                with mock.patch.object(lm,"load_manifest") as load,mock.patch.object(lm,"verify_bank",side_effect=AssertionError("no full hash to refuse")):
                    with self.assertRaisesRegex(RuntimeError,"real-activation quality admission is not implemented"):lm.preflight(env)
                    load.assert_called_once_with(Path("bank.json"))
        with mock.patch.object(lm,"load_manifest",side_effect=ValueError("full bank missing")):
            with self.assertRaisesRegex(ValueError,"full bank missing"):
                lm.preflight({"GLM53_CT_FORMAT":"lut","GLM53_CT_LUT_MANIFEST":"missing.json"})


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
            with self.assertRaisesRegex(RuntimeError,"real-activation quality admission is not implemented"):
                exec(compile(source,"serve.import","exec"),dict(namespace))
        self.assertEqual(imported,[])
        with mock.patch.dict(os.environ,{},clear=True),mock.patch.object(lm,"load_manifest",side_effect=AssertionError("trellis read bank")),mock.patch.object(builtins,"__import__",side_effect=guarded_import):
            with self.assertRaisesRegex(AssertionError,"framework import reached"):
                exec(compile(source,"serve.import","exec"),dict(namespace))
        self.assertEqual(imported,["torch"])

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
