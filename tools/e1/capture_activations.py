#!/usr/bin/env python3
"""Capture real router inputs in a separate offline EXL3 process. Starts no server."""
import argparse
import array
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import threading
import time

sys.dont_write_bytecode = True
from quality_checkpoint import EXL3_SOURCE_SHA, EXL3_REV, MODEL, admit_output, artifact, coverage, require, sha256
from gguf import check_no_symlinks
from maintenance_guard import publish_json as atomic_json
from maintenance_guard import add_arguments, configure, resources

SOURCE_MANIFEST_SHA = EXL3_SOURCE_SHA
NATIVE_LIMIT = 2**30
CORPUS = [
    {"id":"prose-capital","class":"prose","text":"Explain why Canberra became Australia's capital. Separate the political compromise, the site selection and the design of the new city. State uncertainty explicitly where historical accounts differ. Write several connected paragraphs for a reader unfamiliar with Australian history."},
    {"id":"code-ring","class":"code","text":"Implement a bounded single-producer single-consumer ring buffer in C++17. Explain acquire and release memory ordering, how wraparound works, and how you distinguish full from empty. Provide a short example and discuss the lifetime of objects stored in the buffer."},
    {"id":"firmware-uart","class":"firmware","text":"Review an embedded UART receive design: an interrupt writes bytes into a 256-byte ring, the main loop parses length-prefixed frames and validates CRC16, and malformed or oversized frames must recover without a reboot. Give pseudocode, describe overflow handling and explain how to test interrupt and parser races."},
    {"id":"code-utf8","class":"code","text":"Write a streaming UTF-8 decoder in Python that accepts arbitrarily split byte chunks, yields complete Unicode strings, and reports malformed sequences with their absolute byte offsets. Explain how partial multibyte sequences survive between chunks and give boundary-condition examples."},
]


def context_hash(tokens):
    return hashlib.sha256(json.dumps(tokens,separators=(",", ":")).encode()).hexdigest()


def accepted_generated_tokens(job, prompt, previous):
    """Track accepted IDs, including UTF-8 tokens held by text streaming."""
    require(len(job.sequences)==1,"capture needs one generator sequence")
    rows=job.sequences[0].sequence_ids.torch().tolist()
    require(len(rows)==1 and rows[0][:len(prompt)]==prompt,"generator prompt prefix changed")
    generated=rows[0][len(prompt):]
    require(generated[:len(previous)]==previous and 0<=len(generated)-len(previous)<=1,
            "speculative/multiple accepted tokens would break capture context mapping")
    require(all(type(v) is int and v>=0 for v in generated),"invalid accepted token IDs")
    return generated


def require_capture_ingress(module):
    # Installed BlockSparseMLP passes z to routing, but sends y to experts.
    # These optional transformations would make z differ from expert ingress.
    require(all(getattr(module, name, None) is None for name in
                ("router_pre_norm", "routed_pre_norm", "latent_in")),
            "routing hook differs from expert ingress after normalization or latent projection")


def capture_iteration(generator):
    """EXL3 contains per-job errors in iterate events; surface them immediately."""
    events = list(generator.iterate())
    for event in events:
        if event.get("stage") == "error":
            error = event.get("error")
            if isinstance(error, Exception):
                raise error
            raise ValueError("EXL3 capture job failed: " + str(error))
    return events


class CaptureMonitor:
    """Serialize final publication with the resource monitor's last check."""
    def __init__(self, check, *, exit_process=None, monotonic=time.monotonic, stall_seconds=20):
        self._check = check
        self._exit = os._exit if exit_process is None else exit_process
        self._lock = threading.RLock()
        self.stop = threading.Event()
        self.failure = None
        self._clock = monotonic
        self._stall_seconds = stall_seconds
        self._operation_started = None

    def check(self, force=False):
        with self._lock:
            require(self.failure is None, "capture monitor previously failed")
            started = self._operation_started
            if started is None:
                self._operation_started = self._clock()
            try:
                self._check(force)
            finally:
                if started is None:
                    self._operation_started = None

    def watchdog_check(self):
        """No I/O or publication lock: detect a stuck check or fsync/rename."""
        started = self._operation_started
        require(started is None or self._clock()-started <= self._stall_seconds,
                f"capture guard check or publication stalled beyond its {self._stall_seconds} second I/O budget")

    def watchdog(self):
        while not self.stop.wait(1):
            try:
                self.watchdog_check()
            except Exception as exc:
                self.failure = exc
                self._exit(3)
                return

    def run(self):
        while not self.stop.wait(1):
            with self._lock:
                if self.stop.is_set():
                    return
                try:
                    self.check(force=True)
                except Exception as exc:
                    self.failure = exc
                    # A full logging pipe must not delay the failure exit.
                    self._exit(3)  # Own process only; never publish after failure.
                    return

    def publish(self, path, manifest):
        with self._lock:
            self._operation_started = self._clock()
            try:
                self.check(force=True)
                atomic_json(path, manifest)
                self.stop.set()
            finally:
                self._operation_started = None


def write_capture_row(vectors, routes, count, layer, active, x, ids, weights, tick=lambda: None):
    """Write the actual routing hook payload with replay's validation rules."""
    tick()
    require(len(x) == 4096 and all(math.isfinite(v) for v in x), "invalid/nonfinite activation")
    require(len(ids) == len(weights) == 8 and len(set(ids)) == 8 and
            all(type(e) is int and 0 <= e < 288 for e in ids), "invalid original top-8 routes")
    require(all(type(w) in (float, int) and math.isfinite(w) and 0 < w <= 16 for w in weights),
            "nonfinite/invalid routing weights")
    require(active["generated"], "capture requires an accepted decode context")
    data = array.array("f", x)
    if sys.byteorder != "little":
        data.byteswap()
    record = {"row": count, "layer": layer, "prompt_id": active["id"], "prompt_class": active["class"],
              "phase": "decode", "step": len(active["generated"]), "source_dtype": "float16", "stored_dtype": "float32",
              "context_sha256": context_hash(active["prompt"] + active["generated"]),
              "experts": ids, "routing_weights": weights}
    line = json.dumps(record, allow_nan=False) + "\n"
    require(vectors.write(data.tobytes()) == len(data) * data.itemsize, "short activation write")
    require(routes.write(line) == len(line), "short routing record write")
    return record


def file_identity(path):
    st=path.stat()
    return {k:getattr(st,"st_"+k) for k in ("dev","ino","size","mtime_ns","ctime_ns")}


def verified_model(model_dir, ledger_path, tick=lambda:None, reuse=False, expected_sha256=None):
    from quality_checkpoint import stable_json
    ledger,digest=stable_json(ledger_path,2**20)
    if reuse: require(expected_sha256,"fingerprint reuse needs owner-approved model ledger SHA256")
    if expected_sha256 is not None:
        require(digest==expected_sha256,"model ledger differs from owner-approved SHA256")
    require(ledger.get("schema") == "e1-capture-model/v1" and ledger.get("model_id") == MODEL,
            "need a pinned e1-capture-model/v1 model ledger")
    require(ledger.get("revision") == EXL3_REV, "pin the expected full EXL3 source revision")
    require(ledger.get("source_manifest_sha256")==SOURCE_MANIFEST_SHA,"EXL3 ledger lacks pinned publisher inventory provenance")
    require(ledger.get("files"), "model ledger empty")
    listed = set()
    for item in ledger["files"]:
        tick()
        relative=Path(item["path"])
        require(not relative.is_absolute() and ".." not in relative.parts,"model ledger escapes model directory")
        check_no_symlinks(model_dir/relative)
        if reuse:
            p=(model_dir/item["path"]).resolve()
            require(p.is_file() and item.get("identity")==file_identity(p),"model fingerprint changed; prepare a new fully hashed ledger")
            require(re.fullmatch("[a-f0-9]{64}",item.get("sha256","")),"model ledger lacks full content hash")
        else:
            p = artifact(model_dir,item,tick)
        require(p.is_relative_to(model_dir), "model ledger escapes model directory")
        require(p not in listed,"duplicate model ledger file")
        listed.add(p)
    actual = {p.resolve() for p in model_dir.rglob("*") if p.is_file() and p.suffix in (".json", ".safetensors", ".model", ".py", ".jinja")}
    require(actual <= listed, "ledger must cover every model/tokenizer JSON, safetensors, Python, template and .model file")
    require(any("tokenizer" in p.name for p in listed) and any(p.suffix == ".safetensors" for p in listed),
            "model ledger missing tokenizer or model payload")
    return ledger


def discover_native_extension():
    """Find the installed binary without importing EXL3, Torch or its loader."""
    spec=importlib.util.find_spec("exllamav3_ext")
    require(spec is not None and spec.loader is not None and isinstance(spec.origin,str) and
            any(spec.origin.endswith(suffix) for suffix in importlib.machinery.EXTENSION_SUFFIXES),
            "installed prebuilt exllamav3_ext is required; capture must not JIT compile it")
    origin=Path(spec.origin)
    require(origin.is_absolute(),"native extension origin must be absolute")
    check_no_symlinks(origin)
    path=origin.resolve()
    fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK)
    with os.fdopen(fd,"rb") as stream:
        st=os.fstat(stream.fileno())
        require(stat.S_ISREG(st.st_mode) and 0<st.st_size<=NATIVE_LIMIT,"native extension must be a bounded regular binary")
        identity={k:getattr(st,"st_"+k) for k in ("dev","ino","size","mtime_ns","ctime_ns")}
    require(file_identity(path)==identity,"native extension changed during discovery")
    return dict(name="exllamav3_ext",origin=str(origin),path=str(path),bytes=st.st_size,identity=identity)


def hash_native_extension(record,tick=lambda:None):
    path=Path(record["path"])
    require(file_identity(path)==record["identity"],"native extension changed before hashing")
    fd=os.open(path,os.O_RDONLY|os.O_CLOEXEC|os.O_NOFOLLOW|os.O_NONBLOCK)
    digest=hashlib.sha256()
    with os.fdopen(fd,"rb") as stream:
        before=os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size==record["bytes"]<=NATIVE_LIMIT,"native extension size/type changed")
        require({k:getattr(before,"st_"+k) for k in record["identity"]}==record["identity"],"native extension inode changed")
        remaining=before.st_size
        while remaining:
            tick();chunk=stream.read(min(8*2**20,remaining))
            require(chunk,"native extension truncated while hashing")
            digest.update(chunk);remaining-=len(chunk)
        require(not stream.read(1),"native extension grew while hashing")
    require(file_identity(path)==record["identity"],"native extension changed while hashing")
    value=dict(record,sha256=digest.hexdigest())
    if "sha256" in record: require(value["sha256"]==record["sha256"],"native extension content changed")
    return value


def verify_loaded_native_extension(record,tick=lambda:None):
    module=sys.modules.get("exllamav3_ext")
    require(module is not None and isinstance(getattr(module,"__file__",None),str) and
            Path(module.__file__).resolve()==Path(record["path"]),"loaded EXL3 native binary differs from admitted installed binary")
    return hash_native_extension(record,tick)


def verified_ec_precompile(source,revision,path,approved_sha256,torch_version):
    from quality_checkpoint import stable_json,source_identity
    path=admit_output(path)
    receipt,digest=stable_json(path)
    require(re.fullmatch("[a-f0-9]{64}",approved_sha256 or "") and digest==approved_sha256,
            "capture EC build needs owner-approved receipt SHA256")
    compiled_torch=receipt.get("torch_version")
    require(isinstance(compiled_torch,str) and bool(compiled_torch) and
            compiled_torch.split("+",1)[0]==torch_version,
            "capture EC precompile Torch metadata/public version mismatch")
    require(receipt.get("schema")=="e1-precompile/v1" and receipt.get("variant")=="capture" and
            receipt.get("state")=="COMPILED_NO_GPU_CONTEXT" and receipt.get("cuda_initialized") is False and
            receipt.get("source_revision")==revision,
            "capture EC precompile runtime/revision mismatch")
    build=admit_output(receipt["build_directory"])
    require(path.parent==build and set(receipt.get("outputs",{})) in ({"glm53_ec.so"}, {"glm53_ec.so", "glm53_k101.so"}),"unexpected EC module/build destination")
    sources=receipt.get("source_sha256",{})
    require("glm53/expert_cache.py" in sources,"EC receipt lacks source binding")
    for relative,expected in sources.items():
        part=Path(relative)
        require(not part.is_absolute() and ".." not in part.parts,"EC source receipt escapes serving tree")
        check_no_symlinks(source/part)
        require(sha256(source/part)==expected,"EC precompile source changed")
    for name, expected in receipt["outputs"].items():
        require(sha256(build/name)==expected, "capture precompiled module SHA256 mismatch")
    module=admit_output(build/"glm53_ec.so")
    require(module.is_file() and module.stat().st_size<=128*2**20,"EC module must be a bounded regular file")
    identity=source_identity(module)
    require(sha256(module)==receipt["outputs"]["glm53_ec.so"] and source_identity(module)==identity,"EC precompile module changed")
    return receipt,module,identity,digest


def verify_torch_runtime(torch,receipt):
    require(getattr(torch,"__version__",None)==receipt["torch_version"],
            "capture Torch runtime differs from exact EC precompile version")


def canonical_input(path):
    path=Path(path)
    require(".." not in path.parts,"capture input path must not contain parent traversal")
    check_no_symlinks(path)
    return path.resolve()

def prepare(argv):
    """Prepare hash-verified local model provenance and a fixed capture corpus."""
    from quality_checkpoint import stable_json
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="action", required=True)
    c = sub.add_parser("prepare-corpus")
    c.add_argument("--out", type=Path, default=Path("./e1-work/corpus.json"))
    s = sub.add_parser("prepare-stats", help="neutral expert placement and cache warm ordering")
    s.add_argument("--out", type=Path, default=Path("./e1-work/stats.json"))
    m = sub.add_parser("prepare-model-ledger")
    m.add_argument("--model-dir", type=Path, required=True)
    m.add_argument("--out", type=Path, default=Path("./e1-work/model-ledger.json"))
    args = ap.parse_args(argv)
    out = admit_output(args.out)
    require(not out.exists(), "preparation output exists; choose a new path")
    if args.action == "prepare-corpus":
        atomic_json(out, CORPUS)
    elif args.action == "prepare-stats":
        atomic_json(out, {f"model.language_model.layers.{layer}.mlp": [0] * 288 for layer in range(3, 45)})
    else:
        inventory = Path(__file__).absolute().parent / "exl3-source.json"
        public, digest = stable_json(inventory, 2**20)
        require(digest == SOURCE_MANIFEST_SHA and public["revision"] == EXL3_REV, "pinned model inventory changed")
        model_dir = canonical_input(args.model_dir)
        files = []
        for item in public["files"]:
            file = artifact(model_dir, item)
            require(file.stat().st_size == item["bytes"], "model file size mismatch")
            files.append(dict(item, identity=file_identity(file)))
        ledger = dict(schema="e1-capture-model/v1", model_id=MODEL, revision=EXL3_REV,
                      source_manifest_sha256=SOURCE_MANIFEST_SHA, files=files)
        atomic_json(out, ledger)
        verified_model(model_dir, out)
    print(out)


def preimport_admission(a, started):
    """Verify every model, runtime and source byte before importing frameworks."""
    from quality_checkpoint import stable_json
    require(not any(name in sys.modules for name in ("torch", "exllamav3", "exllamav3_ext", "serve", "expert_cache")),
            "capture admission must precede framework/model imports")
    exl3_version = importlib.metadata.version("exllamav3")
    require(exl3_version == "1.5.1", "capture requires exllamav3 1.5.1")
    torch_metadata = importlib.metadata.version("torch")
    source, model_dir = (canonical_input(p) for p in (a.serving_source, a.model_dir))
    ledger_path, corpus_path, stats = (canonical_input(p) for p in (a.model_ledger, a.corpus, a.stats))
    out = admit_output(a.out)
    require(not out.exists() and not out.is_relative_to(model_dir), "capture needs a new output outside the model directory")
    out.parent.mkdir(parents=True, exist_ok=True)
    configure(a)
    resources(out.parent, min_ram_gib=a.min_ram_gib, min_disk_gib=a.min_disk_gib)
    require(a.layers == "3,11", "serving admission requires actual layers 3 and 11")
    require(32 <= a.tokens <= 256 and 1024 <= a.cache_size <= 8192, "capture bounds: 32..256 tokens, 1024..8192 context")
    require("," not in a.device and bool(a.device), "select one GPU")
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    require(re.fullmatch("[a-f0-9]{40}", revision) and (not a.serving_revision or a.serving_revision == revision), "serving revision mismatch")
    require(not subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no", "--", "glm53", "kernels"], text=True),
            "commit serving source changes before capture")
    receipt, module, module_identity, receipt_digest = verified_ec_precompile(source, revision, canonical_input(a.ec_precompile), a.ec_precompile_sha256, torch_metadata)
    native = hash_native_extension(discover_native_extension())
    ledger, ledger_digest = stable_json(ledger_path, 2**20)
    public, public_sha = stable_json(Path(__file__).absolute().parent / "exl3-source.json", 2**20)
    require(public_sha == SOURCE_MANIFEST_SHA, "model publisher inventory changed")
    fields = ("path", "bytes", "sha256", "publisher_kind", "publisher_hash")
    require([{k: item.get(k) for k in fields} for item in ledger.get("files", [])] ==
            [{k: item[k] for k in fields} for item in public["files"]], "model ledger differs from pinned publisher files")
    verified_model(model_dir, ledger_path)
    model_identities = {model_dir / item["path"]: file_identity(model_dir / item["path"]) for item in ledger["files"]}
    geometry = stable_json(model_dir / "config.json", 2**20)[0].get("text_config", {})
    expected = dict(hidden_size=4096, moe_intermediate_size=2048, num_experts_per_tok=8,
                    num_hidden_layers=45, first_k_dense_replace=3, n_routed_experts=288)
    require(all(type(geometry.get(k)) is int and geometry[k] == v for k, v in expected.items()) and
            geometry.get("model_type") == "glm5_next_text" and geometry.get("swiglu_limit") == 10, "model expert geometry mismatch")
    corpus, corpus_digest = stable_json(corpus_path, 2**20)
    require(isinstance(corpus, list) and 4 <= len(corpus) <= 16 and
            all(isinstance(p, dict) and all(isinstance(p.get(k), str) and p[k] for k in ("id", "class", "text")) for p in corpus) and
            len({p["id"] for p in corpus}) == len(corpus) and {"prose", "code", "firmware"} <= {p["class"] for p in corpus}, "invalid capture corpus")
    stats_value, stats_digest = stable_json(stats)
    require(isinstance(stats_value, dict) and stats_value, "placement stats must be a nonempty JSON object")
    return dict(source=source, model_dir=model_dir, ledger_path=ledger_path, corpus_path=corpus_path, stats=stats, out=out,
                layers=[3, 11], revision=revision, ec_receipt=receipt, ec_module=module, ec_identity=module_identity,
                ec_digest=receipt_digest, native=native, model_peak=sum(i["bytes"] for i in ledger["files"] if i["path"].endswith(".safetensors")) + 8*2**30,
                ledger=ledger, ledger_digest=ledger_digest, model_identities=model_identities, corpus=corpus,
                corpus_digest=corpus_digest, stats_digest=stats_digest, stats_identity=file_identity(stats),
                torch_metadata_version=torch_metadata, exllamav3_version=exl3_version)


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    started = time.monotonic()
    if argv and argv[0] in ("prepare-corpus", "prepare-stats", "prepare-model-ledger"):
        return prepare(argv)
    if argv and argv[0] == "capture":
        argv = argv[1:]
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--serving-source", default=str(Path(__file__).absolute().parents[2]))
    ap.add_argument("--serving-revision", help="optional expected committed serving HEAD")
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-ledger", default="./e1-work/model-ledger.json")
    ap.add_argument("--ec-precompile", required=True)
    ap.add_argument("--ec-precompile-sha256", required=True)
    ap.add_argument("--corpus", default="./e1-work/corpus.json")
    ap.add_argument("--stats", default="./e1-work/stats.json")
    ap.add_argument("--out", default="./e1-work/capture")
    ap.add_argument("--layers", default="3,11")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--cache-size", type=int, default=8192)
    ap.add_argument("--device", default="0", help="CUDA_VISIBLE_DEVICES for this offline capture process")
    add_arguments(ap)
    a = ap.parse_args(argv)
    admitted = preimport_admission(a, started)
    source,model_dir,ledger_path,corpus_path,stats,out=(admitted[k] for k in ("source","model_dir","ledger_path","corpus_path","stats","out"))
    revision,ec_receipt,ec_module,ec_identity,ec_digest=(admitted[k] for k in ("revision","ec_receipt","ec_module","ec_identity","ec_digest"))
    native,model_peak=(admitted[k] for k in ("native","model_peak"))
    ledger,ledger_digest,model_identities,corpus,corpus_digest,layers=(admitted[k] for k in
        ("ledger","ledger_digest","model_identities","corpus","corpus_digest","layers"))
    stats_digest,stats_identity=admitted["stats_digest"],admitted["stats_identity"]
    out.mkdir(parents=True)
    # Every writable runtime cache/build destination belongs to this new process.
    for variable, subdir in {"TORCH_EXTENSIONS_DIR":"torch", "TRITON_CACHE_DIR":"triton", "CUDA_CACHE_PATH":"cuda",
                             "HF_HOME":"hf", "XDG_CACHE_HOME":"cache", "GLM53_EC_BUILD":"ec", "TMPDIR":"tmp",
                             "EXLLAMAV3_CACHE_DIR":"exl3-cache","TORCHINDUCTOR_CACHE_DIR":"inductor",
                             "PYTORCH_KERNEL_CACHE_PATH":"torch-kernels"}.items():
        p = out / subdir
        p.mkdir()
        os.environ[variable] = str(p)
    os.environ.update(CUDA_VISIBLE_DEVICES=a.device, PYTHONDONTWRITEBYTECODE="1", GLM53_CPU_TIER="0", GLM53_RT="",
                      GLM53_LA="0", GLM53_LA_ON="0", GLM53_K_OVL="0", GLM53_K_HCFUSE="0", GLM53_K_FTSPLIT="0",
                      GLM53_ZC_VRAM="0", GLM53_ZC_STATS=str(stats), GLM53_EC_WARM=str(stats), GLM53_EC="1",
                      GLM53_EC_BG="0",GLM53_EC_BG_ON="0",GLM53_CT_FORMAT="trellis",GLM53_CT_ACT="i16",GLM53_ROOT=str(out),HF_HUB_OFFLINE="1",TRANSFORMERS_OFFLINE="1")
    os.environ.pop("GLM53_TRACE_ROUTES",None)
    os.environ.pop("GLM53_TRITON_PIN",None)
    for src, name in ((corpus_path,"corpus.json"),(ledger_path,"model-ledger.json"),(stats,"stats.json")):
        shutil.copyfile(src,out/name)
    require(sha256(out/"corpus.json")==corpus_digest and sha256(out/"model-ledger.json")==ledger_digest and
            sha256(out/"stats.json")==stats_digest and file_identity(stats)==stats_identity,
            "capture preparation inputs changed before copying")
    # Model loaders may persist coop/autotune JSON beside config.json. Give the
    # offline process its own config/tokenizer copies and read-only weight view,
    # so those writes cannot reach the live model's configuration files.
    model_view=out/"model-view"
    model_view.mkdir()
    for record in ledger["files"]:
        origin=model_dir/record["path"]
        target=model_view/record["path"]
        target.parent.mkdir(parents=True,exist_ok=True)
        if origin.suffix==".safetensors": target.symlink_to(origin)
        else:
            shutil.copyfile(origin,target)
            require(sha256(target)==record["sha256"],"source changed while copying offline model config")
    os.environ.update(GLM53_ZC_STATS=str(out/"stats.json"),GLM53_EC_WARM=str(out/"stats.json"))
    os.chdir(out)  # Relative runtime/autotune outputs belong to this scratch.
    resources(out, min_ram_gib=a.min_ram_gib, min_disk_gib=a.min_disk_gib)
    deadline=started+7200
    def check_runtime(force=False):
        require(time.monotonic()<deadline,"two-hour capture limit reached")
        if force:
            resources(out, min_ram_gib=a.min_ram_gib, min_disk_gib=a.min_disk_gib)
    monitor=CaptureMonitor(check_runtime,stall_seconds=60)
    tick=monitor.check
    threading.Thread(target=monitor.run,daemon=True,name="capture-resource-guard").start()
    threading.Thread(target=monitor.watchdog,daemon=True,name="capture-io-watchdog").start()
    sys.path.insert(0,str(source / "glm53"))
    import torch
    verify_torch_runtime(torch,ec_receipt)
    import serve as sx
    import expert_cache
    from quality_checkpoint import source_identity
    require(source_identity(ec_module)==ec_identity and
            sha256(ec_module)==ec_receipt["outputs"]["glm53_ec.so"],"EC build changed before import")
    require("glm53_ec" not in sys.modules and expert_cache._EXT is None,"unexpected preloaded EC module")
    spec=importlib.util.spec_from_file_location("glm53_ec",ec_module)
    require(spec is not None and spec.loader is not None,"EC module loader unavailable")
    extension=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(extension)
    sys.modules["glm53_ec"]=extension
    expert_cache._EXT=extension  # Reuse the verified prebuilt extension.
    from exllamav3 import Generator, Job
    import exllamav3
    verify_loaded_native_extension(native,tick)
    from exllamav3.generator.sampler import ArgmaxSampler
    runtime_root=Path(exllamav3.__file__).resolve().parent
    runtime_sources=[p for p in runtime_root.rglob("*") if p.is_file() and p.suffix in (".py",".cpp",".h",".cuh",".cu",".so")]
    require(runtime_sources,"cannot fingerprint installed exllamav3 runtime")
    runtime={"native_extension":native,"torch_metadata_version":admitted["torch_metadata_version"],
             "torch_runtime_version":torch.__version__,"ec_precompile":{"receipt_sha256":ec_digest,"module_sha256":ec_receipt["outputs"]["glm53_ec.so"],
                              "source_revision":revision,"torch_version":ec_receipt["torch_version"],
                              "torch_metadata_version":admitted["torch_metadata_version"]},
             "exllamav3_version":importlib.metadata.version("exllamav3"),
             "files":[{"path":str(p),"sha256":sha256(p,tick)} for p in sorted(runtime_sources)],
             "capture_tools":{name:sha256(Path(__file__).resolve().parent/name,tick) for name in ("capture_activations.py","quality_checkpoint.py","gguf.py","quantize_s1.py","maintenance_guard.py")},
             "serving_sources":subprocess.check_output(["git","-C",str(source),"ls-tree","-r","HEAD","glm53","kernels"],text=True,timeout=10)}
    atomic_json(out/"runtime-ledger.json",runtime)
    sys.argv = ["offline-capture", "-m", str(model_view), "-cs", str(a.cache_size), "--max-batch-size", "1"]
    torch.set_grad_enabled(False)
    with torch.inference_mode():
        sx.load()  # No uvicorn / sockets / live HTTP calls. This is a new local model.
    tick(force=True)
    verify_loaded_native_extension(native,tick)
    mods = expert_cache.find_moe(sx.model.modules)
    selected = {}
    for m in mods:
        match = re.search(r"(?:^|\.)layers\.(\d+)\.",m.key)
        require(match, f"unrecognized layer key {m.key}; do not guess layer IDs")
        layer = int(match[1])
        if layer not in layers:
            continue
        require(layer not in selected and m.hidden_size == 4096 and m.intermediate_size == 2048,
                "unexpected MoE geometry/duplicate layer")
        require(float(getattr(m,"act_limit",0) or 0) == 10, "offline model must preserve SwiGLU limit 10")
        require_capture_ingress(m)
        selected[layer] = m
    require(set(selected) == set(layers), "selected real MoE layers not found")
    require(all(m.key in json.loads(stats.read_text()) for m in mods), "placement stats must cover every routed module key")
    active = {"enabled":False}
    count = 0
    captured=[]
    trajectories = []
    with (out/"inputs.f32").open("wb") as vectors, (out/"routes.jsonl").open("w") as routes:
        for layer, m in selected.items():
            original = m.routing_fn
            def routed(bsz,cfg,z,params,_original=original,_layer=layer):
                nonlocal count
                sel,w = _original(bsz,cfg,z,params)
                if active["enabled"] and bsz == 1 and not params.get("autosplit_measure") and not params.get("tp_warmup"):
                    require(z.dtype==torch.float16,"source must supply serving FP16 ingress; do not silently change activation precision")
                    x = z.reshape(1,-1).float().cpu().flatten().tolist()
                    ids,weights = sel.reshape(-1).cpu().tolist(),w.float().reshape(-1).cpu().tolist()
                    record = write_capture_row(vectors,routes,count,_layer,active,x,ids,weights,tick)
                    captured.append(record)
                    count += 1
                return sel,w
            m.routing_fn = routed
        gen = Generator(model=sx.model,cache=sx.cache,tokenizer=sx.tok,max_batch_size=1,max_chunk_size=2048,num_draft_tokens=0)
        require(gen.num_draft_tokens==0,"capture must disable speculative decoding")
        for prompt in corpus:
            text = f"[gMASK]<sop><|user|>{prompt['text']}<|assistant|><think></think>"
            ids = sx.tok.encode(text,encode_special_tokens=True)
            require(ids.numel() >= 2 and ids.numel()+a.tokens+1 < a.cache_size, "corpus does not fit declared context")
            active.update(enabled=False,id=prompt["id"],**{"class":prompt["class"]},prompt=ids[0].tolist(),generated=[])
            job = Job(input_ids=ids,max_new_tokens=a.tokens+1,sampler=ArgmaxSampler(),stop_conditions=[],return_logits=False,token_healing=False)
            gen.enqueue(job)
            while gen.num_remaining_jobs():
                # First output comes from prefill. Capture starts with the next
                # forward, whose context is the prompt plus that actual token.
                active["enabled"] = bool(active["generated"])
                tick()
                with torch.inference_mode(): capture_iteration(gen)
                active["generated"]=accepted_generated_tokens(job,active["prompt"],active["generated"])
            require(len(active["generated"]) == a.tokens + 1, "capture job ended before its exact decode budget")
            trajectories.append({"prompt_id":prompt["id"],"prompt_class":prompt["class"],"prompt_tokens":active["prompt"],"generated_tokens":active["generated"]})
            print(f"captured {prompt['id']}: {len(active['generated'])-1} decode inputs per layer",flush=True)
        active["enabled"] = False
        for stream in (vectors,routes):
            stream.flush()
            os.fsync(stream.fileno())
    tick(force=True)
    atomic_json(out/"trajectories.json",trajectories)
    coverage(captured,layers)
    def item(name): return {"path":name,"sha256":sha256(out/name,tick)}
    manifest = {"schema":"e1-real-activations/v1","complete":True,"kind":"offline-exllamav3-routing-input","synthetic":False,
                "model_id":MODEL,"source_revision":ledger["revision"],"serving_revision":revision,
                "exllamav3_version":importlib.metadata.version("exllamav3"),"hidden_size":4096,"intermediate_size":2048,
                "top_k":8,"act_limit":10,"layers":layers,"rows":count,"tokens_per_prompt":a.tokens,
                "device":a.device,"model_peak_reserved_bytes":model_peak,
                "corpus":item("corpus.json"),"model_ledger":item("model-ledger.json"),"stats":item("stats.json"),
                "inputs":item("inputs.f32"),"routes":item("routes.jsonl"),"trajectories":item("trajectories.json"),
                "runtime_ledger":item("runtime-ledger.json"),
                "distribution":"EXL3 offline forward; Q8_0 is local expert-output reference, not full-model activation source"}
    require(not subprocess.check_output(["git","-C",str(source),"status","--porcelain","--untracked-files=no","--","glm53","kernels"],text=True,timeout=10),
            "serving source changed during capture")
    verify_loaded_native_extension(native,tick)
    require(source_identity(ec_module)==ec_identity and sha256(ec_module,tick)==ec_receipt["outputs"]["glm53_ec.so"],"EC module changed during capture")
    require(all(file_identity(p)==identity for p,identity in model_identities.items()),"source model files changed during capture")
    require(sha256(out/"corpus.json",tick)==corpus_digest and sha256(out/"model-ledger.json",tick)==ledger_digest and
            sha256(out/"stats.json",tick)==stats_digest and file_identity(stats)==stats_identity,
            "capture provenance copies changed during execution")
    monitor.publish(out/"capture.json",manifest)
    print(out/"capture.json")


if __name__ == "__main__":
    main()
