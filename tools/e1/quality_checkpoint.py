#!/usr/bin/env python3
"""Real routed-activation replay and bank-bound quality admission evidence."""
import argparse
import array
import ctypes as C
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import stat
import subprocess
import sys
from gguf import check_no_symlinks
from maintenance_guard import publish_json as atomic_json

HERE = Path(__file__).resolve().parent
LUT_REV = "d1b493379387518d20c7bc89dab0ec1ce320f811"
SOURCE_REV = "a38483c8cd5df544f53d70fb281afe97369d5ab6"
UD_HEADER_SHA = "ab3aa68df71a1ea6018879af4b22cd5eeab5c254a98315d4fe78dfc4c410881a"
MAX_PUBLISHER_RANGE = 64*2**20
MODEL = "GLM-5.3-Flash"
EXL3_REV = "332ab457b709b7ba30dd9a448be5de03b80a7ac9"
EXL3_SOURCE_SHA = "1341c742132c4b42e9f5d1dce4d3d1e07249eed353051c637a77a7adbb8fb859"
CODEC_SHA = {"kernels.cpp":"3daf262cf111df1f2ac04fea05840b147e7efdcd0a7bae4666ecfb6f15362244",
             "kernels.h":"dabec85fd112d544536856e04dc3c722495dd20749c91754cacdd58d9b506669",
             "LICENSE":"153f7caf00796585022b1984edc8c80e5d885c58a68a832ef7bc198696e56380"}
F32 = C.POINTER(C.c_float)


def require(value, message):
    if not value:
        raise ValueError(message)


def admit_output(path):
    """Return a symlink-free local destination chosen by the operator."""
    check_no_symlinks(path)
    return Path(path).absolute()


def sha256(path, tick=lambda:None):
    """Hash a stable regular file in bounded chunks, with admission progress.

    Each input must remain unchanged for the entire read.
    """
    check_no_symlinks(path)
    h = hashlib.sha256()
    fd = os.open(path, os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as f:
        before = os.fstat(f.fileno())
        require(stat.S_ISREG(before.st_mode), "hash input must be a regular file")
        remaining = before.st_size
        while remaining:
            tick()
            data = f.read(min(8 << 20, remaining))
            require(bool(data), "hash input was truncated")
            h.update(data)
            remaining -= len(data)
            tick()
        require(not f.read(1), "hash input grew while reading")
        after = os.fstat(f.fileno())
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns) ==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),
                "hash input changed while reading")
    return h.hexdigest()


def artifact(root, item, tick=lambda:None):
    require(isinstance(item, dict) and item.get("path") and item.get("sha256"), "missing artifact provenance")
    # Conversion receipts may point to a separately selected scratch directory.
    # Every explicit operator path is still checked and fully SHA-verified.
    selected = Path(item["path"])
    candidate = selected if selected.is_absolute() else root / selected
    check_no_symlinks(candidate)
    p = candidate.resolve()
    require(p.is_file(), f"missing artifact: {p}")
    require(sha256(p,tick) == item["sha256"], f"artifact SHA256 mismatch: {p}")
    return p


def build(kernels, output):
    """Compile into scratch, then publish immutable library and source receipt."""
    import tempfile
    from maintenance_guard import publish_file
    kernels = Path(kernels).resolve()
    for name in ("kernels.h", "kernels.cpp", "LICENSE"):
        require(sha256(kernels / name) == CODEC_SHA[name], "unverified pinned codec source: " + name)
    output = admit_output(output)
    require(not output.exists() and not output.with_suffix(".json").exists(), "replay output already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=".replay-building-", suffix=".so", dir=output.parent)
    os.close(fd)
    command = ["g++", "-std=c++17", "-O3", "-shared", "-fPIC", "-mavx2", "-mfma", "-mf16c",
               "-march=broadwell", "-mno-avx512f", "-mno-avxvnni", "-mno-avx512vnni",
               "-ffp-contract=off", "-Wall", "-Wextra", "-I", str(kernels), str(HERE / "quality_replay.cpp"),
               str(kernels / "kernels.cpp"), "-o", temporary]
    try:
        subprocess.run(command, check=True, timeout=180)
        provenance = {"schema": "e1-replay-build/v1", "lutbench_revision": LUT_REV,
                      "library_sha256": sha256(temporary), "command": command,
                      "temporary_output": Path(temporary).name,
                      "sources": {str(p): sha256(p) for p in (HERE / "quality_replay.cpp", kernels / "kernels.cpp", kernels / "kernels.h")}}
        publish_file(temporary, output)
        atomic_json(output.with_suffix(".json"), provenance)
        return provenance
    finally:
        Path(temporary).unlink(missing_ok=True)


def validate_replay_library(path, tick=lambda:None):
    """Verify the CPU library receipt and bytes without loading native code."""
    path = Path(path).resolve()
    receipt_path = path.with_suffix(".json")
    identities = {p:source_identity(p) for p in (path, receipt_path)}
    p, receipt_sha = stable_json(receipt_path)
    library_sha = sha256(path, tick)
    require(p.get("schema")=="e1-replay-build/v1" and p.get("lutbench_revision") == LUT_REV and
            p.get("library_sha256") == library_sha, "replay library provenance mismatch")
    entries=p.get("sources",{})
    require(isinstance(entries,dict) and len(entries)==3 and
            {Path(src).name for src in entries}=={"quality_replay.cpp","kernels.cpp","kernels.h"},"replay receipt must name exact reviewed sources")
    for src, digest in entries.items():
        identities[Path(src)] = source_identity(src)
        expected=sha256(HERE/"quality_replay.cpp",tick) if Path(src).name=="quality_replay.cpp" else CODEC_SHA[Path(src).name]
        require(digest==expected and sha256(src,tick)==expected,"replay source is not the pinned codec/adapter; rebuild")
    source_by_name={Path(src).name:Path(src) for src in entries}
    kernel=source_by_name["kernels.cpp"].parent
    require(source_by_name["kernels.h"].parent==kernel,"replay codec source directories differ")
    expected_command=["g++","-std=c++17","-O3","-shared","-fPIC","-mavx2","-mfma","-mf16c",
        "-march=broadwell","-mno-avx512f","-mno-avxvnni","-mno-avx512vnni","-ffp-contract=off",
        "-Wall","-Wextra","-I",str(kernel),str(source_by_name["quality_replay.cpp"]),str(kernel/"kernels.cpp"),"-o",str(path.parent/p.get("temporary_output",path.name))]
    require(p.get("command")==expected_command,"replay compiler command differs from reviewed CPU-only build")
    require(all(source_identity(src)==identity for src,identity in identities.items()),"replay build input changed during verification")
    return dict(library_sha256=library_sha, library_build_sha256=receipt_sha,
                codec_revision=LUT_REV, identities=identities)


class Replay:
    def __init__(self, path, check_build=True):
        path = Path(path).resolve()
        if check_build:
            validate_replay_library(path)
        self.lib = C.CDLL(str(path))
        self.lib.e1_error.restype = C.c_char_p
        self.lib.e1_row_bytes.argtypes = [C.c_int, C.c_int]
        self.lib.e1_row_bytes.restype = C.c_size_t
        self.lib.e1_quantize.argtypes = [C.c_int, F32, C.c_int, C.c_void_p]
        self.lib.e1_forward.argtypes = [C.c_void_p] * 3 + [C.c_int] * 5 + [F32] + [C.c_int] * 4 + [F32]

    def row_bytes(self, typ, width):
        value = self.lib.e1_row_bytes(typ, width)
        require(value, self.lib.e1_error().decode())
        return value

    def forward(self, weights, types, h, intermediate, inputs, path="i16", reference=False, tp2=True):
        m = len(inputs)
        x = (C.c_float * (h * m))(*(v for row in inputs for v in row))
        buffers = [(C.c_ubyte * len(w)).from_buffer_copy(w) for w in weights]
        for w, typ, n, rows in zip(weights, types, (h, h, intermediate), (intermediate, intermediate, h)):
            require(len(w) == self.row_bytes(typ, n) * rows, "expert weight size mismatch")
        out = (C.c_float * (h * m))()
        status = self.lib.e1_forward(*buffers, *types, h, intermediate, x, m, int(path == "q8"), int(tp2), int(reference), out)
        require(status == 0, self.lib.e1_error().decode())
        return [list(out[t*h:(t+1)*h]) for t in range(m)]


def stable_json(path, limit=32*2**20):
    """Hash the exact bounded regular-file bytes that were parsed."""
    check_no_symlinks(path)
    fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,"rb") as f:
        before=os.fstat(f.fileno())
        require(stat.S_ISREG(before.st_mode),"provenance JSON must be a regular file")
        data=f.read(limit+1)
        after=os.fstat(f.fileno())
        require((before.st_dev,before.st_ino,before.st_size,before.st_mtime_ns,before.st_ctime_ns)==
                (after.st_dev,after.st_ino,after.st_size,after.st_mtime_ns,after.st_ctime_ns),"provenance JSON changed during read")
    require(len(data)<=limit,"provenance JSON exceeds bounded size")
    def pairs(items):
        result = {}
        for key, value in items:
            require(key not in result, "duplicate JSON field")
            result[key] = value
        return result
    def constant(value):
        raise ValueError("nonfinite JSON value")
    return json.loads(data, object_pairs_hook=pairs, parse_constant=constant),hashlib.sha256(data).hexdigest()


def source_identity(path):
    st=Path(path).stat()
    return st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns,st.st_ctime_ns


def load_capture(path, manifest=None):
    path = Path(path).resolve()
    cap = stable_json(path)[0] if manifest is None else manifest
    require(cap.get("schema") == "e1-real-activations/v1" and cap.get("complete") is True, "missing complete real activation capture")
    require(cap.get("kind") == "offline-exllamav3-routing-input" and cap.get("synthetic") is False, "synthetic/unrecognized activation provenance")
    require(cap.get("model_id") == MODEL and cap.get("act_limit") == 10 and cap.get("top_k") == 8, "model/activation semantics mismatch")
    for key in ("source_revision", "serving_revision", "exllamav3_version"):
        require(isinstance(cap.get(key), str) and bool(cap[key]), f"missing capture {key}")
    require(len(cap["source_revision"]) == 40 and len(cap["serving_revision"]) == 40, "capture revisions must be pinned full commits")
    require(cap["source_revision"] == EXL3_REV,"capture must use pinned EXL3 reference-distribution revision")
    root = path.parent
    corpus = stable_json(artifact(root, cap["corpus"]),2**20)[0]
    artifact(root, cap["stats"])
    runtime=stable_json(artifact(root,cap["runtime_ledger"]),8*2**20)[0]
    require(runtime.get("exllamav3_version")==cap["exllamav3_version"] and runtime.get("files") and runtime.get("serving_sources"),"missing/mismatched capture runtime provenance")
    native=runtime.get("native_extension",{})
    require(native.get("name")=="exllamav3_ext" and isinstance(native.get("path"),str) and native["path"] and
            isinstance(native.get("sha256"),str) and len(native["sha256"])==64 and
            type(native.get("bytes")) is int and 0<native["bytes"]<=2**30 and
            all(type(native.get("identity",{}).get(k)) is int and native["identity"][k]>0 for k in ("dev","ino","size","mtime_ns","ctime_ns")) and
            native["identity"]["size"]==native["bytes"],"missing installed EXL3 native extension provenance")
    ec=runtime.get("ec_precompile",{})
    require(ec.get("source_revision")==cap["serving_revision"] and isinstance(ec.get("torch_version"),str) and ec["torch_version"] and
            all(isinstance(ec.get(k),str) and len(ec[k])==64 for k in ("receipt_sha256","module_sha256")),"missing capture EC precompile provenance")
    require(runtime.get("torch_runtime_version")==ec["torch_version"] and
            runtime.get("torch_metadata_version")==ec["torch_version"].split("+",1)[0],
            "capture Torch runtime/metadata differs from the exact EC precompile version")
    require("exllamav3_runtime_version" not in runtime or runtime["exllamav3_runtime_version"]==cap["exllamav3_version"],
            "capture EXL3 imported runtime differs from its metadata version")
    require(set(runtime.get("capture_tools",{}))=={"capture_activations.py","quality_checkpoint.py","gguf.py","quantize_s1.py","maintenance_guard.py"} and
            all(len(v)==64 for v in runtime["capture_tools"].values()),"missing capture tool provenance")
    ledger = stable_json(artifact(root, cap["model_ledger"]),2**20)[0]
    require(ledger.get("schema") == "e1-capture-model/v1" and ledger.get("model_id") == MODEL and
            ledger.get("revision") == cap["source_revision"] and ledger.get("source_manifest_sha256") == EXL3_SOURCE_SHA and ledger.get("files"), "capture/model ledger mismatch")
    trajectories = stable_json(artifact(root, cap["trajectories"]),2**20)[0]
    prompts = {p["id"]:p for p in corpus}
    tokens = {p["prompt_id"]:p for p in trajectories}
    require(len(prompts) == len(corpus) and len(tokens) == len(trajectories) and set(tokens) == set(prompts), "corpus/trajectory examples mismatch")
    data = artifact(root, cap["inputs"])
    rows = artifact(root, cap["routes"])
    require(cap.get("hidden_size") == 4096 and cap.get("intermediate_size") == 2048, "wrong GLM expert dimensions")
    require(type(cap.get("rows")) is int and 0<cap["rows"]<=8192 and rows.stat().st_size<=8*2**20,"capture exceeds bounded replay row budget")
    records = [json.loads(line) for line in rows.read_text().splitlines()]
    require(records and len(records) == cap.get("rows"), "capture row count mismatch")
    require(data.stat().st_size == len(records)*4096*4, "activation binary length mismatch")
    with data.open("rb") as f:
        seen = set()
        for i, record in enumerate(records):
            key = (record.get("prompt_id"), record.get("step"), record.get("layer"))
            require(key not in seen, "duplicate activation example")
            seen.add(key)
            require(record.get("row") == i and record.get("phase") == "decode", "noncanonical or non-decode activation")
            require(record.get("source_dtype") == "float16" and record.get("stored_dtype") == "float32", "capture must preserve serving FP16 ingress")
            require(isinstance(record.get("context_sha256"), str) and len(record["context_sha256"]) == 64, "missing token context provenance")
            require(isinstance(record.get("step"), int) and record["step"] >= 0, "missing decode step")
            prompt_id = record.get("prompt_id")
            require(prompt_id in prompts and record.get("prompt_class") == prompts[prompt_id]["class"] == tokens[prompt_id]["prompt_class"], "capture/corpus prompt mismatch")
            trajectory = tokens[prompt_id]
            step = record["step"]
            require(1 <= step < len(trajectory["generated_tokens"]), "decode context step outside trajectory")
            context = trajectory["prompt_tokens"] + trajectory["generated_tokens"][:step]
            require(all(type(t) is int and t >= 0 for t in context), "invalid trajectory token IDs")
            digest = hashlib.sha256(json.dumps(context,separators=(",", ":")).encode()).hexdigest()
            require(digest == record["context_sha256"], "activation/token context mismatch")
            ids, weights = record.get("experts", []), record.get("routing_weights", [])
            require(len(ids) == len(weights) == 8 and len(set(ids)) == 8 and all(type(e) is int and 0 <= e < 288 for e in ids), "invalid original top-8 routes")
            require(all(type(w) in (float, int) and math.isfinite(w) and 0 < w <= 16 for w in weights), "nonfinite/invalid routing weights")
            x = array.array("f")
            x.frombytes(f.read(4096*4))
            if sys.byteorder != "little":
                x.byteswap()
            require(all(math.isfinite(v) for v in x), "nonfinite activation")
            record["x"] = x
    return cap, records


def coverage(records, layers):
    require(all(type(l) is int and 3 <= l <= 44 for l in layers), "only main routed layers 3..44 are eligible")
    require(11 in layers and any(l not in (11, 12, 44) for l in layers), "need blk11 plus an ordinary layer")
    require(set(r["layer"] for r in records) == set(layers), "activation/weight layers mismatch")
    info = {}
    for layer in layers:
        rows = [r for r in records if r["layer"] == layer]
        examples = {(r["prompt_id"], r["step"], r["context_sha256"]) for r in rows}
        prompts = {r["prompt_id"] for r in rows}
        classes = {r.get("prompt_class") for r in rows}
        experts = {e for r in rows for e in r["experts"]}
        require(len(rows) >= 128 and len(prompts) >= 4 and len(experts) >= 32, f"inadequate sample coverage in blk{layer}: need >=128 tokens,4 prompts,32 experts")
        require({"prose", "code", "firmware"} <= classes, f"missing representative prompt classes in blk{layer}")
        require(all(sum(r["prompt_id"] == p for r in rows) >= 16 for p in prompts), "need >=16 decode tokens per prompt")
        info[layer] = {"tokens": len(rows), "prompts": len(prompts), "classes": sorted(classes), "experts": len(experts), "route_occurrences": len(rows)*8}
        if layer == layers[0]:
            expected = examples
        else:
            require(examples == expected, "layers must use identical token contexts")
    return info


def metric(values):
    require(values and all(math.isfinite(e) and n > 0 for e, n in values), "invalid/empty error statistics")
    return {"median_expert_nmse": statistics.median(e for e, n in values),
            "routing_frequency_weighted_mean_nmse": sum(e*n for e, n in values) / sum(n for e, n in values)}


def compare(ours, unsloth):
    return all(ours[k] <= unsloth[k] for k in ours)


def distribution(values):
    ordered=sorted(values)
    require(ordered and all(math.isfinite(v) for v in ordered),"empty/nonfinite diagnostic distribution")
    return {"mean":statistics.mean(ordered),"median":statistics.median(ordered),
            "p90":ordered[math.ceil(.9*len(ordered))-1],"p99":ordered[math.ceil(.99*len(ordered))-1],"worst":ordered[-1]}


def verify_publisher_payload(source, baseline, summary, tick=lambda:None):
    """Bind every local UD payload byte to a contiguous audited remote range."""
    shards = {s["file"]:s for s in summary["shards"]}
    tensors = {t["name"]:t for t in summary["expert_tensors"]}
    records = baseline.get("ranges", [])
    require({r.get("tensor") for r in records} == {t["name"] for t in source.tensors}, "UD range/tensor census mismatch")
    with source.path.open("rb") as f:
        for tensor in source.tensors:
            expected = tensors[tensor["name"]]
            require(tensor["dims"] == expected["ggml_dimensions"] and tensor["type_id"] == expected["type_id"], "UD profile differs from publisher header")
            shard = shards[expected["shard"]]
            start = shard["tensor_data_start"]+expected["data_relative_offset"]
            ranges = sorted((r for r in records if r["tensor"] == tensor["name"]),key=lambda r:r["start"])
            f.seek(source.data_start+tensor["offset"])
            offset = 0
            for record in ranges:
                tick()
                require(record.get("etag") == shard["etag"] and record.get("lfs_sha256") == shard["lfs_sha256"], "UD shard identity mismatch")
                require(record.get("url") == f"https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/{SOURCE_REV}/{expected['shard']}", "UD range URL mismatch")
                size = record.get("bytes",0)
                require(type(size) is int and 0 < size <= MAX_PUBLISHER_RANGE and record["start"] == start+offset and record["end"] == start+offset+size-1, "UD gaps/overlaps/unbounded ranges")
                require(offset+size <= tensor["nbytes"], "UD range exceeds tensor")
                payload = f.read(size)
                require(len(payload) == size and hashlib.sha256(payload).hexdigest() == record.get("sha256"), "UD local payload/range hash mismatch")
                offset += size
            require(offset == tensor["nbytes"] == expected["tensor_bytes"], "UD ranges do not cover complete tensor")


def json_digest(value):
    normalized=json.loads(json.dumps(value,allow_nan=False))
    return hashlib.sha256(json.dumps(normalized,sort_keys=True,separators=(",", ":"),allow_nan=False).encode()).hexdigest()


REAL_TP2 = "split-intermediate-gate-up/down-columns-before-quantization"
S1_REAL_SCOPE = "s1-real-check-only"


def real_report(layers, activation, batch, act_limit, binding):
    """Recompute the real gate only from checked, paired raw occurrence errors."""
    require(set(map(int,layers))=={3,11},"real checkpoint requires actual layers 3 and 11")
    pooled={arm:[] for arm in ("ours","unsloth")}
    selected={}
    for layer in (3,11):
        row=layers.get(layer,layers.get(str(layer)))
        cov=row["coverage"]
        require(set(cov)=={"tokens","prompts","classes","experts","route_occurrences"},"invalid real coverage fields")
        require(all(type(cov[k]) is int for k in ("tokens","prompts","experts","route_occurrences")) and
                cov["tokens"]>=128 and cov["prompts"]>=4 and cov["experts"]>=32 and
                cov["route_occurrences"]==cov["tokens"]*8 and
                {"prose","code","firmware"}<=set(cov["classes"]),"inadequate real coverage")
        seen=set(); arms={arm:[] for arm in pooled}
        for item in row["experts"]:
            e,n=item["expert"],item["routing_count"]
            require(type(e) is int and 0<=e<288 and e not in seen and type(n) is int and 0<n<=cov["tokens"],"invalid real expert/count")
            seen.add(e)
            for arm in pooled:
                errors=item[arm+"_sample_errors"]
                require(len(errors)==n and all(type(v) in (int,float) and math.isfinite(v) and v>=0 for v in errors),"invalid/nonfinite real sample errors")
                arms[arm].append((statistics.mean(errors),n))
        require(len(seen)==cov["experts"] and sum(n for _,n in arms["ours"])==cov["route_occurrences"],"real routing census differs from raw errors")
        value={arm:metric(values) for arm,values in arms.items()}
        drift=row["tp2_full_down_max_relative_l2"]
        require(set(drift)=={"ours","unsloth"} and all(type(v) in (int,float) and math.isfinite(v) and 0<=v<=1e-5 for v in drift.values()),"invalid real TP2/full drift proof")
        value.update(tp2_full_down_max_relative_l2=drift,coverage=cov,experts=row["experts"],pass_=compare(value["ours"],value["unsloth"]))
        value["pass"]=value.pop("pass_")
        selected[str(layer)]=value
        for arm in pooled: pooled[arm].extend(arms[arm])
    total={arm:metric(values) for arm,values in pooled.items()}
    total["pass"]=compare(total["ours"],total["unsloth"])
    passed=total["pass"] and all(r["pass"] for r in selected.values())
    require(activation in ("i16","q8") and type(batch) is int and batch in (1,2,4) and
            type(act_limit) in (float,int) and math.isfinite(act_limit) and act_limit>0,"invalid real replay configuration")
    result = dict(schema="e1-real-quality/v1",state="REAL-PASS" if passed else "REAL-FAIL",complete=True,real_pass=passed,
                synthetic=False,activation=activation,batch=batch,act_limit=act_limit,tp2=REAL_TP2,
                binding=binding,binding_sha256=json_digest(binding),layers=selected,aggregate=total)
    if binding.get("scope")==S1_REAL_SCOPE:
        require("bank_manifest_sha256" not in binding,"S1 real-check cannot claim a full-bank binding")
        result.update(scope=S1_REAL_SCOPE,production_ready=False,full_bank_bound=False)
    return result


def validate_real_weight_context(context,layer):
    from quantize_s1 import PINNED, METADATA_SHARD, custom_rules
    expected=f"Q8_0/GLM-5.3-Flash-Q8_0-{2 if layer==3 else 3:05d}-of-00008.gguf"
    require(context.get("revision")==SOURCE_REV and context.get("layer")==layer and
            context.get("imatrix")==PINNED["imatrix_unsloth.gguf_file"][1] and
            context.get("metadata_source")==PINNED[METADATA_SHARD][1],"real weight source/imatrix/metadata provenance is not pinned")
    sources=context.get("sources",{})
    require(len(sources)==1 and all(Path(path).parts[-2:]==tuple(expected.split("/")) and digest==PINNED[expected][1]
                                   for path,digest in sources.items()),"real Q8 shard source provenance differs from pinned layer")
    require(context.get("recipe")==custom_rules([layer]),"real LUT recipe context differs from actual layer")
    require(isinstance(context.get("normalized_imatrix_sha256"),str) and len(context["normalized_imatrix_sha256"])==64,
            "missing normalized imatrix hash")


def validate_s1_real_sources(quant, ud):
    """Check the exact S1 three-arm receipts without requiring an S2 bank.

    Artifact payload hashes and publisher ranges are checked by run(), using
    these same parsed manifests. This helper never admits production startup.
    """
    require(quant.get("schema")=="e1-s1/v1" and quant.get("state")=="S1_READY" and
            ud.get("schema")=="e1-ud/v1" and quant.get("revision")==ud.get("revision")==SOURCE_REV,
            "S1 real-check requires ready pinned S1 and publisher UD manifests")
    require(quant.get("quantizer",{}).get("source_revision")=="5bf8f0fe4db98e9560de0f495232f8d3a733c678",
            "quantizer source revision mismatch")
    for manifest in (quant,ud):
        entries=manifest.get("layers",[])
        require(len(entries)==2 and all(type(item.get("layer")) is int for item in entries) and
                {item["layer"] for item in entries}=={3,11},"S1 real-check requires exactly actual layers 3 and 11")
    bound={}
    for layer in (3,11):
        own=next(item for item in quant["layers"] if item["layer"]==layer)
        baseline=next(item for item in ud["layers"] if item["layer"]==layer)
        context=own["q8"].get("context")
        require(isinstance(context,dict) and context==own["lut"].get("context"),
                "Q8/LUT quantization source contexts differ or are missing")
        validate_real_weight_context(context,layer)
        require(baseline.get("origin")==f"https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/{SOURCE_REV}/UD-IQ4_XS",
                "UD origin must be pinned publisher payload")
        digests={"q8_sha256":own["q8"].get("sha256"),"ours_sha256":own["lut"].get("sha256"),
                 "unsloth_sha256":baseline["gguf"].get("sha256")}
        require(all(isinstance(value,str) and len(value)==64 and all(c in "0123456789abcdef" for c in value)
                    for value in digests.values()),"S1 real-check requires exact SHA256 for all three weight arms")
        bound[str(layer)]=dict(digests,conversion_context=context)
    return bound


def _prepare_replay_inputs(quant_manifest, unsloth_manifest, library, tick=lambda:None, *,
                           expected_quant_sha256=None, expected_ud_sha256=None,
                           expected_library_sha256=None, expected_library_build_sha256=None):
    """Common real-run and dry admission; fully hash every actual weight arm."""
    from gguf import read_gguf
    quant_path, ud_path = Path(quant_manifest).resolve(), Path(unsloth_manifest).resolve()
    identities = {p:source_identity(p) for p in (quant_path, ud_path)}
    (quant, quant_sha), (ud, ud_sha) = (stable_json(p) for p in (quant_path, ud_path))
    for actual, expected, label in ((quant_sha, expected_quant_sha256, "quant manifest"),
                                    (ud_sha, expected_ud_sha256, "publisher manifest")):
        require(expected is None or actual == expected, "approved " + label + " SHA256 mismatch")
    bound = validate_s1_real_sources(quant, ud)
    summary_path = (ud_path.parent / ud["header_summary"]["path"]).resolve()
    identities[summary_path] = source_identity(summary_path)
    summary, summary_sha = stable_json(summary_path)
    require(summary_sha == ud["header_summary"]["sha256"] == UD_HEADER_SHA,
            "UD header inventory is not pinned public publisher inventory")
    layers = {}
    for item in quant["layers"]:
        layer = item["layer"]
        baseline = next(row for row in ud["layers"] if row["layer"] == layer)
        require(baseline.get("ranges") and all(r.get("etag") and r.get("lfs_sha256") and r.get("sha256")
                for r in baseline["ranges"]), "missing UD source range provenance")
        sources = {}
        for arm, root, receipt in (("q8", quant_path.parent, item["q8"]),
                                    ("ours", quant_path.parent, item["lut"]),
                                    ("unsloth", ud_path.parent, baseline["gguf"])):
            path = (root / receipt["path"]).resolve()
            identities[path] = source_identity(path)
            source = read_gguf(artifact(root, receipt, tick))
            require(source_identity(path) == identities[path], "weight artifact changed while hashing/parsing")
            sources[arm] = source
        verify_publisher_payload(sources["unsloth"], baseline, summary, tick)
        types = {"q8":[8,8,8], "ours":[144,144,14] if layer == 11 else [138,138,144],
                 "unsloth":[23,23,14] if layer == 11 else [21,21,23]}
        names = [f"blk.{layer}.ffn_{part}_exps.weight" for part in ("gate", "up", "down")]
        descriptors = {}
        for arm, source in sources.items():
            require({tensor["name"] for tensor in source.tensors} == set(names), "expected routed-only per-layer GGUF")
            descriptors[arm] = [source.tensor(name) for name in names]
            require([tensor["type_id"] for tensor in descriptors[arm]] == types[arm],
                    f"incorrect real tensor profile: {arm} blk{layer}")
            require([list(tensor["dims"]) for tensor in descriptors[arm]] ==
                    [[4096,2048,288],[4096,2048,288],[2048,4096,288]], "weight dimensions mismatch")
        layers[layer] = dict(sources=sources, descriptors=descriptors, types=types)
    replay = validate_replay_library(library, tick)
    require(expected_library_sha256 is None or replay["library_sha256"] == expected_library_sha256,
            "approved replay library SHA256 mismatch")
    require(expected_library_build_sha256 is None or replay["library_build_sha256"] == expected_library_build_sha256,
            "approved replay build receipt SHA256 mismatch")
    identities.update(replay.pop("identities"))
    tool_names=("quality_checkpoint.py","quality_replay.cpp","gguf.py","quantize_s1.py","maintenance_guard.py")
    identities.update({HERE/name:source_identity(HERE/name) for name in tool_names})
    tools = {name:sha256(HERE/name,tick) for name in tool_names}
    require(all(source_identity(path) == identity for path, identity in identities.items()),
            "replay admission input changed during verification")
    return dict(quant=quant, ud=ud, layers=layers, identities=identities,
                result=dict(schema="e1-replay-input-admission/v1", state="INPUTS_READY", complete=True,
                            real_pass=False, production_ready=False, capture_required=True,
                            quant_manifest_sha256=quant_sha, unsloth_manifest_sha256=ud_sha,
                            header_summary_sha256=summary_sha, tools_sha256=tools, layers=bound, **replay))


def inspect_replay_inputs(quant_manifest, unsloth_manifest, library, **kwargs):
    """CPU byte/ABI admission; no capture, Torch, GPU access or inference."""
    prepared = _prepare_replay_inputs(quant_manifest, unsloth_manifest, library, **kwargs)
    Replay(library)  # Load the CPU adapter and bind symbols; never call a GEMV.
    require(all(source_identity(path) == identity for path, identity in prepared["identities"].items()),
            "replay admission input changed during native library loading")
    return dict(prepared["result"],cpu_library_loaded=True)


def real_binding(args, cap, quant, ud, tick, capture_sha256, manifest_sha256=None):
    from quantize_s1 import PINNED
    s1_real_check=bool(getattr(args,"s1_real_check",False))
    bank_manifest=getattr(args,"bank_manifest",None)
    require(s1_real_check != bool(bank_manifest),"select exactly one of --s1-real-check or --bank-manifest")
    if s1_real_check:
        bound=validate_s1_real_sources(quant,ud)
        require(isinstance(manifest_sha256,dict) and set(manifest_sha256)=={"quant_manifest_sha256","ud_manifest_sha256"},
                "S1 real-check requires exact parsed manifest SHA256 bindings")
        source_binding=dict(scope=S1_REAL_SCOPE,**manifest_sha256)
        imatrix_sha256=PINNED["imatrix_unsloth.gguf_file"][1]
    else:
        bank_path=Path(bank_manifest).resolve()
        bank,bank_sha256=stable_json(bank_path)
        require(bank.get("schema")=="e1-lut-bank/v1" and bank.get("state")=="S2_READY" and
                bank.get("revision")==SOURCE_REV and bank.get("imatrix_sha256")==PINNED["imatrix_unsloth.gguf_file"][1],"complete pinned S2 public manifest required")
        entries=bank.get("layers",[])
        require(len(entries)==42 and {e["layer"] for e in entries}==set(range(3,45)) and
                sum(len(e["tensors"]) for e in entries)==126 and
                bank.get("ledger",{}).get("canonical_payload_bytes")==143879307264 and
                bank["ledger"].get("tp2_payload_bytes")==144063332352,"S2 bank census/byte ledger mismatch")
        bound={}
        for layer in (3,11):
            own=next(i for i in quant["layers"] if i["layer"]==layer)
            baseline=next(i for i in ud["layers"] if i["layer"]==layer)
            entry=next(i for i in entries if i["layer"]==layer)
            validate_real_weight_context(own["lut"]["context"],layer)
            require(entry["artifact"]["sha256"]==own["lut"]["sha256"],"real replay weights differ from S2 bank")
            require(own["lut"]["context"]["imatrix"]==bank["imatrix_sha256"],"replay imatrix differs from bank")
            bound[str(layer)]=dict(q8_sha256=own["q8"]["sha256"],ours_sha256=own["lut"]["sha256"],unsloth_sha256=baseline["gguf"]["sha256"])
        source_binding=dict(bank_manifest_sha256=bank_sha256)
        imatrix_sha256=bank["imatrix_sha256"]
    capture={key+"_sha256":cap[key]["sha256"] for key in
             ("model_ledger","runtime_ledger","corpus","stats","inputs","routes","trajectories")}
    capture.update(model_revision=cap["source_revision"],serving_revision=cap["serving_revision"],exllamav3_version=cap["exllamav3_version"])
    library=Path(args.library).resolve()
    replay=dict(library_sha256=sha256(library,tick),build_receipt_sha256=sha256(library.with_suffix(".json"),tick),
                codec_revision=LUT_REV,kernels_cpp_sha256=CODEC_SHA["kernels.cpp"],kernels_h_sha256=CODEC_SHA["kernels.h"])
    replay.update({name.replace(".py","").replace(".cpp","")+"_sha256":sha256(HERE/name,tick) for name in
                   ("quality_checkpoint.py","quality_replay.cpp","gguf.py","maintenance_guard.py")})
    return dict(protocol="real-routed-activations/v1",**source_binding,source_revision=SOURCE_REV,
                imatrix_sha256=imatrix_sha256,capture_manifest_sha256=capture_sha256,capture=capture,replay=replay,layers=bound)


def run(args):
    from gguf import read_gguf
    require(bool(getattr(args,"s1_real_check",False)) != bool(getattr(args,"bank_manifest",None)),
            "select exactly one of --s1-real-check or --bank-manifest")
    if getattr(args,"s1_real_check",False):
        require(args.activation=="i16" and args.batch==1,"S1 real-check is restricted to I16/M1")
    quant_path, ud_path = Path(args.quant_manifest).resolve(), Path(args.unsloth_manifest).resolve()
    out = Path(args.out).resolve()
    tick = lambda: None
    prepared = _prepare_replay_inputs(quant_path, ud_path, args.library, tick)
    quant, ud = prepared["quant"], prepared["ud"]
    manifest_sha256=dict(quant_manifest_sha256=prepared["result"]["quant_manifest_sha256"],
                         ud_manifest_sha256=prepared["result"]["unsloth_manifest_sha256"])
    cap_snapshot,capture_sha256=stable_json(args.capture)
    capture_files=[Path(args.capture).resolve()]+[(Path(args.capture).resolve().parent/cap_snapshot[k]["path"]).resolve() for k in
                ("inputs","routes","corpus","stats","runtime_ledger","model_ledger","trajectories")]
    capture_identities={p:source_identity(p) for p in capture_files}
    cap, rows = load_capture(args.capture,cap_snapshot)
    layers = [item["layer"] for item in quant["layers"]]
    require(len(set(layers)) == len(layers) and set(layers) == {l["layer"] for l in ud["layers"]}, "duplicate/mismatched weight layers")
    require(set(layers)=={3,11},"real admission requires actual layers 3 and 11")
    info = coverage(rows, layers)
    require(cap["exllamav3_version"]=="1.5.1","unreviewed capture runtime")
    replay_paths=[Path(args.library).resolve(),Path(args.library).resolve().with_suffix(".json"),HERE/"quality_replay.cpp"]
    replay_identities={p:source_identity(p) for p in replay_paths}
    replay = Replay(args.library)
    binding=real_binding(args,cap,quant,ud,tick,capture_sha256,manifest_sha256)
    if args.activation=="q8":
        require(args.i16_result,"Q8 real replay requires completed matching I16 REAL-PASS")
        prior=stable_json(args.i16_result)[0]
        recomputed=real_report(prior["layers"],prior["activation"],prior["batch"],prior["act_limit"],prior["binding"])
        require(prior==recomputed and prior["state"]=="REAL-PASS" and prior["activation"]=="i16" and
                prior["binding"]==binding and prior["act_limit"]==cap["act_limit"],"Q8 requires identical-provenance I16 REAL-PASS")
    file_identities=dict(capture_identities)
    file_identities.update(replay_identities)
    file_identities.update(prepared["identities"])
    per_layer, all_metrics = {}, {"ours": [], "unsloth": []}
    for layer_item in quant["layers"]:
        layer = layer_item["layer"]
        checked = prepared["layers"][layer]
        sources, descriptors, expected_types = checked["sources"], checked["descriptors"], checked["types"]
        occurrences = {}
        for row in rows:
            if row["layer"] == layer:
                for e, weight in zip(row["experts"], row["routing_weights"]):
                    occurrences.setdefault(e, []).append((row, weight))
        experts, metrics = [], {"ours": [], "unsloth": []}
        per_prompt={arm:{} for arm in metrics}
        layer_sums={row["row"]:{arm:[0.]*4096 for arm in ("q8","ours","unsloth")} for row in rows if row["layer"]==layer}
        drift={arm:0. for arm in metrics}
        for expert, examples in sorted(occurrences.items()):
            weights = {}
            for arm, source in sources.items():
                weights[arm] = []
                with source.path.open("rb") as f:
                    for tensor in descriptors[arm]:
                        size = tensor["nbytes"] // 288
                        f.seek(source.data_start + tensor["offset"] + expert*size)
                        payload = f.read(size)
                        require(len(payload) == size, "truncated expert payload")
                        weights[arm].append(payload)
            errors = {"ours": [], "unsloth": []}
            cursor = 0
            while cursor < len(examples):
                tick()
                m = min(args.batch, len(examples)-cursor)
                if m == 3: m = 2
                chunk = examples[cursor:cursor+m]
                inputs = [row["x"] for row, _ in chunk]
                reference = replay.forward(weights["q8"], expected_types["q8"], 4096, 2048, inputs, reference=True)
                for (row,weight),value in zip(chunk,reference):
                    total=layer_sums[row["row"]]["q8"]
                    for j,v in enumerate(value): total[j] += weight*v
                for arm in errors:
                    candidate = replay.forward(weights[arm], expected_types[arm], 4096, 2048, inputs, path=args.activation)
                    full = replay.forward(weights[arm], expected_types[arm], 4096, 2048, inputs, path=args.activation,tp2=False)
                    for (row,weight),ref,value,unsplit in zip(chunk,reference,candidate,full):
                        split_error=math.sqrt(sum((a-b)**2 for a,b in zip(value,unsplit))/max(1e-30,sum(v*v for v in unsplit)))
                        require(math.isfinite(split_error) and split_error<=1e-5,"actual TP2/full-down arithmetic drift exceeds 1e-5 relative L2")
                        drift[arm]=max(drift[arm],split_error)
                        denominator = sum(float(v)*v for v in ref)
                        require(math.isfinite(denominator) and denominator > 1e-20, "zero/nonfinite reference energy; cannot assign NMSE")
                        error = sum((float(a)-b)**2 for a,b in zip(value,ref)) / denominator
                        require(math.isfinite(error), "nonfinite expert error")
                        errors[arm].append(error)
                        per_prompt[arm].setdefault(row["prompt_id"],[]).append(error)
                        total=layer_sums[row["row"]][arm]
                        for j,v in enumerate(value): total[j] += weight*v
                cursor += m
            item = {"expert": expert, "routing_count": len(examples)}
            for arm in errors:
                mean = statistics.mean(errors[arm])
                item[arm + "_sample_errors"] = errors[arm]
                metrics[arm].append((mean,len(examples)))
                all_metrics[arm].append((mean,len(examples)))
            experts.append(item)
        result = {arm: metric(values) for arm, values in metrics.items()}
        routed={arm:[] for arm in metrics}
        for sums in layer_sums.values():
            energy=sum(v*v for v in sums["q8"])
            require(math.isfinite(energy) and energy>1e-20,"zero/nonfinite reference routed-layer energy")
            for arm in metrics:
                routed[arm].append(sum((a-b)**2 for a,b in zip(sums[arm],sums["q8"]))/energy)
        result["diagnostics"]={"tp2_full_down_max_relative_l2":drift,
                               "routed_layer_sum_nmse":{arm:distribution(values) for arm,values in routed.items()},
                               "paired_per_prompt_nmse":{prompt:{arm:distribution(per_prompt[arm][prompt]) for arm in metrics} for prompt in per_prompt["ours"]}}
        result.update(coverage=info[layer], experts=experts,tp2_full_down_max_relative_l2=drift)
        result["pass"] = compare(result["ours"], result["unsloth"])
        per_layer[layer] = result
    total = {arm: metric(values) for arm,values in all_metrics.items()}
    passed = compare(total["ours"],total["unsloth"]) and all(r["pass"] for r in per_layer.values())
    tick()
    require(all(source_identity(p)==identity for p,identity in file_identities.items()),"capture or weight artifact changed during real replay")
    return real_report(per_layer,args.activation,args.batch,cap["act_limit"],binding)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build", help="build the pinned CPU-only replay adapter")
    b.add_argument("--kernels", default=str(HERE.parent / "lutbench"))
    b.add_argument("--out", default="./e1-work/replay/quality_replay.so")
    q = sub.add_parser("run")
    for flag in ("capture", "quant-manifest", "unsloth-manifest", "library", "out"):
        q.add_argument("--" + flag, required=True)
    source = q.add_mutually_exclusive_group(required=True)
    source.add_argument("--bank-manifest", help="complete public bank for serving-bound evidence")
    source.add_argument("--s1-real-check", action="store_true", help="diagnostic only; cannot admit serving")
    q.add_argument("--activation", choices=("i16", "q8"), default="i16")
    q.add_argument("--batch", type=int, choices=(1, 2, 4), default=1)
    q.add_argument("--i16-result", help="required matching I16 REAL-PASS before Q8")
    from maintenance_guard import add_arguments, configure, resources
    add_arguments(q)
    args = ap.parse_args(argv)
    if args.command == "build":
        print(json.dumps(build(args.kernels, args.out), indent=2))
        return 0
    configure(args)
    output = admit_output(args.out)
    require(not output.exists(), "result path exists; retain old evidence and choose a new path")
    output.parent.mkdir(parents=True, exist_ok=True)
    resources(output.parent, min_ram_gib=args.min_ram_gib, min_disk_gib=args.min_disk_gib)
    try:
        result = run(args)
    except (OSError, ValueError, RuntimeError, TimeoutError, subprocess.SubprocessError, KeyError, TypeError, StopIteration) as error:
        result = dict(schema="e1-real-quality/v1", state="BLOCKED", complete=False,
                      real_pass=False, synthetic=False, reason=str(error))
        if args.s1_real_check:
            result.update(scope=S1_REAL_SCOPE, production_ready=False, full_bank_bound=False)
    atomic_json(output, result)
    print(json.dumps({k: v for k, v in result.items() if k not in ("layers", "binding")}, indent=2))
    return 0 if result["state"] == "REAL-PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
