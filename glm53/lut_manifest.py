"""LUT bank validation and SHA-bound real-activation quality admission.

PROXY-PASS authorizes conversion only. Owner-approved real reports may admit
quality; production coordinator startup remains unavailable. No Torch or CUDA
is imported by this module.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import statistics
import struct

SOURCE_REVISION = "a38483c8cd5df544f53d70fb281afe97369d5ab6"
IK_REVISION = "5bf8f0fe4db98e9560de0f495232f8d3a733c678"
IMATRIX_SHA256 = "27d2ef6b6be0e314b373080f96322a1e4668b09b192664f3edc231c3b7c2d749"
LAYERS = tuple(range(3, 45))
PARTS = ("gate", "up", "down")
CANONICAL_BYTES = 143879307264
TP2_BYTES = 144063332352
JSON_LIMIT = 512 * 2**10
HEADER_LIMIT = 64 * 2**20
HASH_CHUNK = 4 * 2**20
OFFSET_CONVENTION = "tensor offset is relative to artifact.data_start; row_stride includes IQ4_KS scale"
# ggml type id: (name, values per block, bytes per block, extra bytes per row).
TYPES = {138: ("IQ3_K", 256, 110, 0), 144: ("IQ4_KS", 256, 136, 4), 14: ("Q6_K", 256, 210, 0)}
SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
REAL_JSON_LIMIT = 32 * 2**20
REAL_PROTOCOL = "real-routed-activations/v1"
REAL_ADMISSION_MESSAGE = (
    "Real LUT quality is admitted, but production LUT coordinator startup is unavailable; "
    "restore the default trellis path until production integration is reviewed."
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, label, minimum=0, maximum=2**63 - 1):
    require(type(value) is int and minimum <= value <= maximum, f"invalid integer: {label}")
    return value


def digest(value, label):
    require(isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value),
            f"invalid SHA256: {label}")
    return value


def keys(value, expected, label):
    require(isinstance(value, dict) and set(value) == set(expected), f"invalid fields: {label}")


def no_symlinks(path):
    path = Path(path).absolute()
    for item in (path, *path.parents):
        require(not item.is_symlink(), "manifest and artifact paths must not contain symlinks")
    return path


def identity(st):
    return dict(dev=st.st_dev, ino=st.st_ino, size=st.st_size,
                mtime_ns=st.st_mtime_ns, ctime_ns=st.st_ctime_ns)


@contextlib.contextmanager
def open_regular(path):
    path = no_symlinks(path)
    fd = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(fd)
        require(stat.S_ISREG(before.st_mode), "manifest and artifacts must be regular files")
        stream = os.fdopen(fd, "rb")
        fd = -1
        with stream:
            yield stream, identity(before)
    finally:
        if fd >= 0:
            os.close(fd)


def unchanged(path, stream, before):
    require(identity(os.fstat(stream.fileno())) == before and
            identity(no_symlinks(path).stat(follow_symlinks=False)) == before,
            "manifest or artifact changed during validation")


def unique_pairs(pairs):
    result = {}
    for name, value in pairs:
        require(name not in result, "duplicate JSON field")
        result[name] = value
    return result


def finite_constant(value):
    raise ValueError("nonfinite JSON value")


def recipe(layer, part):
    if part == "down":
        return 14 if layer in (11, 12, 44) else 144
    return 144 if layer == 11 else 138


def tensor_layout(layer, part):
    dims = [2048, 4096, 288] if part == "down" else [4096, 2048, 288]
    type_id = recipe(layer, part)
    name, block, size, extra = TYPES[type_id]
    stride = dims[0] // block * size + extra
    return dict(name=f"blk.{layer}.ffn_{part}_exps.weight", dims=dims, type=name, type_id=type_id,
                row_stride=stride, nbytes=stride * math.prod(dims[1:]))


def validate_structure(value):
    """Validate all 42 triplets and byte arithmetic without opening model files."""
    keys(value, ("schema", "state", "revision", "ik_revision", "imatrix_sha256", "real_pass",
                 "accepted_quality_verdict", "accepted_proxy_sha256", "private_manifest_sha256",
                 "offset_convention", "ledger", "layers"), "public bank manifest")
    require(value["schema"] == "e1-lut-bank/v1" and value["state"] == "S2_READY", "full S2_READY public bank required")
    require(value["revision"] == SOURCE_REVISION and value["ik_revision"] == IK_REVISION and
            value["imatrix_sha256"] == IMATRIX_SHA256, "bank source/quantizer/imatrix provenance mismatch")
    require(value["real_pass"] is False and value["accepted_quality_verdict"] == "PROXY-PASS",
            "conversion manifest must remain proxy-only with real_pass=false")
    digest(value["private_manifest_sha256"], "private manifest")
    proof = value["accepted_proxy_sha256"]
    require(isinstance(proof, list) and len(proof) == 2, "two distinct accepted proxy digests required")
    for sha in proof:
        digest(sha, "accepted proxy")
    require(len(set(proof)) == 2, "two distinct accepted proxy digests required")
    require(value["offset_convention"] == OFFSET_CONVENTION, "unrecognized tensor offset convention")
    expected_ledger = dict(completed_layers=42, completed_tensors=126, completed_canonical_payload_bytes=CANONICAL_BYTES,
                           target_layers=42, target_tensors=126, canonical_payload_bytes=CANONICAL_BYTES,
                           tp2_payload_bytes=TP2_BYTES, per_node_tp2_payload_bytes=TP2_BYTES // 2)
    keys(value["ledger"], expected_ledger, "byte ledger")
    for name, expected in expected_ledger.items():
        require(integer(value["ledger"][name], name) == expected, "full bank byte ledger mismatch")
    require(isinstance(value["layers"], list) and len(value["layers"]) == 42, "all 42 main routed layers required")
    seen, payload, extra_scales = set(), 0, 0
    for row in value["layers"]:
        keys(row, ("layer", "artifact", "tensors"), "layer")
        layer = integer(row["layer"], "model layer", 3, 44)
        require(layer not in seen, "duplicate model layer")
        seen.add(layer)
        artifact = row["artifact"]
        keys(artifact, ("path", "sha256", "bytes", "data_start"), "artifact")
        require(artifact["path"] == f"blk.{layer:02d}.lut.gguf", "artifact path must be its canonical relative basename")
        digest(artifact["sha256"], "artifact")
        size = integer(artifact["bytes"], "artifact bytes", 1)
        start = integer(artifact["data_start"], "GGUF data start", 32, HEADER_LIMIT)
        require(start % 32 == 0 and start < size, "invalid GGUF data start")
        tensors = row["tensors"]
        require(isinstance(tensors, list) and len(tensors) == 3, "exact routed gate/up/down triplet required")
        expected = {tensor_layout(layer, part)["name"]: tensor_layout(layer, part) for part in PARTS}
        names, intervals, layer_bytes = set(), [], 0
        for tensor in tensors:
            keys(tensor, ("name", "dims", "type", "type_id", "nbytes", "offset", "row_stride"), "tensor")
            name = tensor["name"]
            require(isinstance(name, str) and name in expected and name not in names, "unexpected or duplicate routed tensor")
            names.add(name)
            for field, wanted in expected[name].items():
                got = tensor[field]
                if isinstance(wanted, list):
                    require(isinstance(got, list) and all(type(d) is int for d in got), "invalid tensor dimensions")
                elif type(wanted) is int:
                    integer(got, field)
                require(got == wanted, f"tensor recipe/geometry mismatch: {name} {field}")
            offset = integer(tensor["offset"], "tensor offset")
            end = offset + tensor["nbytes"]
            require(offset % 32 == 0 and start + end <= size, "tensor range exceeds artifact or is misaligned")
            intervals.append((offset, end))
            layer_bytes += tensor["nbytes"]
            if name.endswith("ffn_down_exps.weight") and tensor["type_id"] == 144:
                extra_scales += 4 * math.prod(tensor["dims"][1:])
        end = 0
        for begin, following in sorted(intervals):
            require(begin >= end, "overlapping tensor ranges")
            end = following
        require(size <= layer_bytes + HEADER_LIMIT + 3 * 65536, "artifact has unbounded padding")
        payload += layer_bytes
    require(seen == set(LAYERS) and payload == CANONICAL_BYTES and payload + extra_scales == TP2_BYTES,
            "full bank inventory or TP2 row-scale ledger mismatch")
    return value


@dataclass(frozen=True)
class BankManifest:
    path: Path
    sha256: str
    value: dict


def read_json(path, limit, label):
    path = no_symlinks(path)
    with open_regular(path) as (stream, before):
        payload = stream.read(limit + 1)
        require(len(payload) <= limit, f"{label} exceeds bounded JSON size")
        value = json.loads(payload, object_pairs_hook=unique_pairs, parse_constant=finite_constant)
        unchanged(path, stream, before)
    return value, hashlib.sha256(payload).hexdigest()


def load_manifest(path, check_files=True):
    path = no_symlinks(path)
    value, sha256 = read_json(path, JSON_LIMIT, "public manifest")
    validate_structure(value)
    if check_files:
        for row in value["layers"]:
            artifact = row["artifact"]
            file = no_symlinks(path.parent / artifact["path"])
            st = file.stat(follow_symlinks=False)
            require(stat.S_ISREG(st.st_mode) and st.st_size == artifact["bytes"],
                    f"artifact missing, nonregular or wrong size: {artifact['path']}")
    return BankManifest(path=path, sha256=sha256, value=value)


def read_gguf_header(stream, size):
    """Read bounded tensor descriptors; skip tokenizer metadata without loading it."""
    def take(n):
        require(type(n) is int and 0 <= n <= HEADER_LIMIT and stream.tell() + n <= min(size, HEADER_LIMIT),
                "GGUF header exceeds bounded file range")
        data = stream.read(n)
        require(len(data) == n, "truncated GGUF header")
        return data

    def scalar(fmt):
        return struct.unpack("<" + fmt, take(struct.calcsize(fmt)))[0]

    def skip(n):
        require(0 <= n and stream.tell() + n <= min(size, HEADER_LIMIT), "GGUF metadata exceeds header bound")
        stream.seek(n, os.SEEK_CUR)

    def string():
        length = scalar("Q")
        require(length <= 512, "oversized GGUF descriptor string")
        return take(length).decode("utf-8")

    def skip_value(typ):
        if typ in SCALAR_BYTES:
            skip(SCALAR_BYTES[typ])
        elif typ == 8:
            skip(scalar("Q"))
        elif typ == 9:
            item, count = scalar("I"), scalar("Q")
            require(item != 9 and count <= 10_000_000, "nested or oversized GGUF metadata array")
            if item in SCALAR_BYTES:
                skip(count * SCALAR_BYTES[item])
            else:
                require(item == 8, "unrecognized GGUF metadata array type")
                for _ in range(count):
                    skip(scalar("Q"))
        else:
            raise ValueError("unrecognized GGUF metadata type")

    require(take(4) == b"GGUF" and scalar("I") == 3, "standalone little-endian GGUF v3 required")
    nt, nk = scalar("Q"), scalar("Q")
    require(nt == 3 and nk <= 4096, "GGUF must contain three routed tensors and at most 4096 metadata keys")
    metadata, seen, alignment = {}, set(), 32
    for _ in range(nk):
        name, typ = string(), scalar("I")
        require(name not in seen and not name.startswith("split."), "duplicate metadata or split GGUF is forbidden")
        seen.add(name)
        if name == "general.architecture":
            require(typ == 8, "invalid architecture metadata type")
            metadata[name] = string()
        elif name == "general.alignment":
            require(typ == 4, "invalid alignment metadata type")
            alignment = scalar("I")
        else:
            skip_value(typ)
    require(metadata.get("general.architecture") == "glm5-next", "GGUF architecture mismatch")
    require(0 < alignment <= 65536 and alignment & (alignment - 1) == 0, "invalid GGUF alignment")
    tensors, seen = [], set()
    for _ in range(nt):
        name, nd = string(), scalar("I")
        require(name not in seen and nd == 3, "duplicate tensor or invalid GGUF rank")
        seen.add(name)
        dims = [scalar("Q") for _ in range(nd)]
        type_id, offset = scalar("I"), scalar("Q")
        require(type_id in TYPES and all(0 < d <= 2**31 for d in dims), "unknown tensor type or invalid dimensions")
        typ, block, block_bytes, extra = TYPES[type_id]
        require(dims[0] % block == 0, "tensor width not divisible by quantization block")
        stride = dims[0] // block * block_bytes + extra
        tensors.append(dict(name=name, dims=dims, type=typ, type_id=type_id, offset=offset,
                            row_stride=stride, nbytes=stride * math.prod(dims[1:])))
    data_start = (stream.tell() + alignment - 1) // alignment * alignment
    require(data_start <= HEADER_LIMIT, "GGUF data start exceeds header bound")
    end = 0
    for tensor in sorted(tensors, key=lambda t: t["offset"]):
        require(tensor["offset"] % alignment == 0 and tensor["offset"] >= end, "misaligned or overlapping GGUF tensor")
        end = tensor["offset"] + tensor["nbytes"]
        require(data_start + end <= size, "GGUF tensor exceeds file bounds")
    return data_start, tensors


def verify_artifact(path, row, tick=lambda: None):
    artifact = row["artifact"]
    with open_regular(path) as (stream, before):
        require(before["size"] == artifact["bytes"], "artifact length changed")
        start, tensors = read_gguf_header(stream, before["size"])
        expected = {t["name"]: t for t in row["tensors"]}
        require(start == artifact["data_start"] and {t["name"]: t for t in tensors} == expected,
                "GGUF descriptors differ from the public manifest")
        stream.seek(0)
        actual = hashlib.sha256()
        while True:
            tick()
            data = stream.read(HASH_CHUNK)
            if not data:
                break
            actual.update(data)
        require(actual.hexdigest() == artifact["sha256"], "artifact SHA256 mismatch")
        unchanged(path, stream, before)
    ordered = [expected[f"blk.{row['layer']}.ffn_{part}_exps.weight"] for part in PARTS]
    return dict(model_layer=row["layer"], internal_li=row["layer"] - 3, path=str(Path(path).absolute()), identity=before,
                tensors=[dict(offset=start+t["offset"], nbytes=t["nbytes"], type_id=t["type_id"],
                              width=t["dims"][0], rows=t["dims"][1], experts=t["dims"][2], row_stride=t["row_stride"])
                         for t in ordered])


def verify_bank(path, tick=lambda: None):
    """Explicit offline full-file verification. Never grants serving admission.

    The caller supplies resource/deadline supervision through tick. Files are
    streamed sequentially with a 4 MiB hash buffer and no weight allocation.
    """
    bank = load_manifest(path)
    descriptors = []
    for row in sorted(bank.value["layers"], key=lambda r: r["layer"]):
        tick()
        descriptors.append(verify_artifact(bank.path.parent / row["artifact"]["path"], row, tick))
    return dict(schema="e1-lut-verified-bank/v1", manifest_sha256=bank.sha256, real_pass=False,
                serving_admitted=False, canonical_payload_bytes=CANONICAL_BYTES, tp2_payload_bytes=TP2_BYTES,
                layers=descriptors)


def json_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def real_number(value, label, positive=False):
    require(type(value) in (int, float), f"invalid real number: {label}")
    try:
        valid = math.isfinite(value) and (value > 0 if positive else value >= 0)
    except OverflowError:
        valid = False
    require(valid, f"nonfinite or out-of-range real number: {label}")
    return value


def quality_metrics(values):
    means = [statistics.mean(errors) for errors, count in values]
    result = dict(median_expert_nmse=statistics.median(means),
                  routing_frequency_weighted_mean_nmse=math.fsum(mean * count for mean, (_, count) in zip(means, values)) /
                  sum(count for _, count in values))
    for key, value in result.items():
        real_number(value, key)
    return result


def compare_metrics(ours, unsloth):
    return all(ours[key] <= unsloth[key] for key in ours)


def check_reported_metrics(saved, expected):
    keys(saved, expected, "quality metrics")
    for key, value in expected.items():
        got = real_number(saved[key], key)
        require(math.isclose(got, value, rel_tol=1e-12, abs_tol=1e-15), "reported quality metrics differ from raw errors")


def validate_real_report(value, bank, activation):
    """Recompute real gates; an operator-approved file SHA supplies acceptance."""
    keys(value, ("schema", "state", "complete", "real_pass", "synthetic", "activation", "batch", "act_limit",
                 "tp2", "binding", "binding_sha256", "layers", "aggregate"), "real quality report")
    require(value["schema"] == "e1-real-quality/v1" and value["state"] == "REAL-PASS" and
            value["complete"] is True and value["real_pass"] is True and value["synthetic"] is False,
            "complete REAL-PASS required; proxy and synthetic results cannot admit quality")
    require(value["activation"] == activation and activation in ("i16", "q8"), "real quality activation mismatch")
    require(integer(value["batch"], "real replay batch", 1, 4) in (1, 2, 4), "unsupported real replay batch")
    real_number(value["act_limit"], "model activation limit", positive=True)
    require(value["tp2"] == "split-intermediate-gate-up/down-columns-before-quantization", "real replay TP2 protocol mismatch")
    binding = value["binding"]
    keys(binding, ("protocol", "bank_manifest_sha256", "source_revision", "imatrix_sha256", "capture_manifest_sha256",
                   "capture", "replay", "layers"), "real quality binding")
    require(digest(value["binding_sha256"], "quality binding") == json_digest(binding), "real quality binding SHA256 mismatch")
    require(binding["protocol"] == REAL_PROTOCOL and binding["source_revision"] == SOURCE_REVISION and
            binding["imatrix_sha256"] == IMATRIX_SHA256, "real quality source/imatrix/protocol mismatch")
    require(digest(binding["bank_manifest_sha256"], "quality bank manifest") == bank.sha256,
            "real quality report belongs to another bank manifest")
    digest(binding["capture_manifest_sha256"], "capture manifest")
    capture = binding["capture"]
    capture_hashes = ("model_ledger_sha256", "runtime_ledger_sha256", "corpus_sha256", "stats_sha256", "inputs_sha256",
                      "routes_sha256", "trajectories_sha256")
    keys(capture, ("model_revision", "serving_revision", "exllamav3_version", *capture_hashes), "real capture provenance")
    require(capture["model_revision"] == "332ab457b709b7ba30dd9a448be5de03b80a7ac9" and
            capture["exllamav3_version"] == "1.5.1", "real capture model/runtime revision mismatch")
    require(isinstance(capture["serving_revision"], str) and len(capture["serving_revision"]) == 40 and
            all(c in "0123456789abcdef" for c in capture["serving_revision"]), "invalid capture serving revision")
    for key in capture_hashes:
        digest(capture[key], key)
    replay = binding["replay"]
    replay_hashes = ("library_sha256", "build_receipt_sha256", "quality_checkpoint_sha256", "quality_replay_sha256",
                     "gguf_sha256", "maintenance_guard_sha256", "kernels_cpp_sha256", "kernels_h_sha256")
    keys(replay, (*replay_hashes, "codec_revision"), "real replay provenance")
    for key in replay_hashes:
        digest(replay[key], key)
    require(replay["codec_revision"] == "d1b493379387518d20c7bc89dab0ec1ce320f811" and
            replay["kernels_cpp_sha256"] == "3daf262cf111df1f2ac04fea05840b147e7efdcd0a7bae4666ecfb6f15362244" and
            replay["kernels_h_sha256"] == "dabec85fd112d544536856e04dc3c722495dd20749c91754cacdd58d9b506669",
            "real replay codec provenance mismatch")
    keys(binding["layers"], ("3", "11"), "real weight bindings")
    by_layer = {row["layer"]: row for row in bank.value["layers"]}
    for layer in (3, 11):
        weights = binding["layers"][str(layer)]
        keys(weights, ("q8_sha256", "ours_sha256", "unsloth_sha256"), "real weight binding")
        for key in weights:
            digest(weights[key], key)
        require(weights["ours_sha256"] == by_layer[layer]["artifact"]["sha256"], "real replay LUT weight hash differs from bank")
    keys(value["layers"], ("3", "11"), "real quality layers")
    pooled = {"ours": [], "unsloth": []}
    coverage_identity = None
    for layer in ("3", "11"):
        row = value["layers"][layer]
        keys(row, ("coverage", "experts", "ours", "unsloth", "pass", "tp2_full_down_max_relative_l2"), "real quality layer")
        drift = row["tp2_full_down_max_relative_l2"]
        keys(drift, ("ours", "unsloth"), "real TP2/full-down drift")
        for arm, error in drift.items():
            require(real_number(error, "real TP2/full-down drift") <= 1e-5, "real TP2/full-down drift exceeds 1e-5")
        coverage = row["coverage"]
        keys(coverage, ("tokens", "prompts", "classes", "experts", "route_occurrences"), "real sample coverage")
        tokens = integer(coverage["tokens"], "decode tokens", 128, 65536)
        prompts = integer(coverage["prompts"], "capture prompts", 4, tokens // 16)
        experts = integer(coverage["experts"], "routed experts", 32, 288)
        require(integer(coverage["route_occurrences"], "route occurrences") == 8 * tokens, "real top-8 route count mismatch")
        classes = coverage["classes"]
        require(isinstance(classes, list) and 3 <= len(classes) <= 32 and
                all(isinstance(item, str) and 0 < len(item) <= 64 for item in classes) and len(set(classes)) == len(classes) and
                {"prose", "code", "firmware"} <= set(classes), "real capture prompt classes missing or malformed")
        current_identity = (tokens, prompts, sorted(classes))
        require(coverage_identity is None or coverage_identity == current_identity, "real layer capture coverage differs")
        coverage_identity = current_identity
        records = row["experts"]
        require(isinstance(records, list) and len(records) == experts, "real routed expert coverage mismatch")
        seen, total, samples = set(), 0, {"ours": [], "unsloth": []}
        for record in records:
            keys(record, ("expert", "routing_count", "ours_sample_errors", "unsloth_sample_errors"), "real expert error record")
            expert = integer(record["expert"], "real routed expert", 0, 287)
            require(expert not in seen, "duplicate real expert error record")
            seen.add(expert)
            count = integer(record["routing_count"], "real expert routing count", 1, tokens)
            total += count
            for arm in samples:
                errors = record[arm + "_sample_errors"]
                require(isinstance(errors, list) and len(errors) == count, "raw real errors/routing count mismatch")
                for error in errors:
                    real_number(error, "raw real sample error")
                samples[arm].append((errors, count))
        require(total == 8 * tokens, "expert routing counts differ from captured top-8 occurrences")
        expected = {arm: quality_metrics(records) for arm, records in samples.items()}
        for arm in samples:
            check_reported_metrics(row[arm], expected[arm])
            pooled[arm].extend(samples[arm])
        passed = compare_metrics(expected["ours"], expected["unsloth"])
        require(row["pass"] is passed and passed, "real per-layer quality gate failed")
    expected = {arm: quality_metrics(records) for arm, records in pooled.items()}
    aggregate = value["aggregate"]
    keys(aggregate, ("ours", "unsloth", "pass"), "real aggregate")
    for arm in expected:
        check_reported_metrics(aggregate[arm], expected[arm])
    passed = compare_metrics(expected["ours"], expected["unsloth"])
    require(aggregate["pass"] is passed and passed, "real pooled quality gate failed")
    return value


def load_real_report(path, expected_sha256, bank, activation):
    require(path is not None, f"owner-approved real {activation} report path is required")
    digest(expected_sha256, f"owner-approved real {activation} report")
    value, actual = read_json(path, REAL_JSON_LIMIT, "real quality report")
    require(actual == expected_sha256, "real quality report does not match the owner-approved SHA256")
    validate_real_report(value, bank, activation)
    return value


def validate_model_act_limit(settings, modules):
    """Future production initialization must match the actual model to admission."""
    require(settings.format == "lut" and settings.admission is not None, "real LUT admission required")
    expected = settings.admission["act_limit"]
    modules = tuple(modules)
    require(bool(modules), "model MoE modules required for activation-limit validation")
    for module in modules:
        actual = getattr(module, "act_limit", None)
        real_number(actual, "actual model activation limit", positive=True)
        require(actual == expected, "actual model activation limit differs from real quality admission")
    return expected


@dataclass(frozen=True)
class TierSettings:
    format: str
    activation: str
    manifest: Path | None
    quality: Path | None
    quality_sha256: str | None = None
    q8_quality: Path | None = None
    q8_quality_sha256: str | None = None
    admission: dict | None = None


def parse_env(environ=None):
    env = os.environ if environ is None else environ
    fmt, activation = env.get("GLM53_CT_FORMAT", "trellis"), env.get("GLM53_CT_LUT_ACT", "i16")
    require(fmt in ("trellis", "lut"), "GLM53_CT_FORMAT must be trellis or lut")
    require(activation in ("i16", "q8"), "GLM53_CT_LUT_ACT must be i16 or q8")
    manifest, quality = env.get("GLM53_CT_LUT_MANIFEST"), env.get("GLM53_CT_LUT_QUALITY")
    require(fmt != "lut" or bool(manifest), "GLM53_CT_LUT_MANIFEST is required for LUT format")
    q8_quality = env.get("GLM53_CT_LUT_Q8_QUALITY")
    return TierSettings(fmt, activation, Path(manifest) if manifest else None, Path(quality) if quality else None,
                        env.get("GLM53_CT_LUT_QUALITY_SHA256"), Path(q8_quality) if q8_quality else None,
                        env.get("GLM53_CT_LUT_Q8_QUALITY_SHA256"))


def preflight(environ=None, *, require_production=True):
    """Check quality before allocations; production startup remains blocked."""
    settings = parse_env(environ)
    if settings.format == "trellis":
        return settings
    bank = load_manifest(settings.manifest)  # Complete inventory and cheap file checks.
    i16 = load_real_report(settings.quality, settings.quality_sha256, bank, "i16")
    reports = {"i16": settings.quality_sha256}
    if settings.activation == "q8":
        q8 = load_real_report(settings.q8_quality, settings.q8_quality_sha256, bank, "q8")
        require(q8["binding"] == i16["binding"] and q8["act_limit"] == i16["act_limit"],
                "I16 and Q8 real reports must bind identical capture, weights, replay and model limit")
        reports["q8"] = settings.q8_quality_sha256
    admission = dict(schema="e1-real-admission/v1", real_quality_admitted=True, production_ready=False,
                     bank_manifest_sha256=bank.sha256, report_sha256=reports,
                     capture_manifest_sha256=i16["binding"]["capture_manifest_sha256"], act_limit=i16["act_limit"])
    if require_production:
        raise RuntimeError(REAL_ADMISSION_MESSAGE)
    return TierSettings(settings.format, settings.activation, settings.manifest, settings.quality,
                        settings.quality_sha256, settings.q8_quality, settings.q8_quality_sha256, admission)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest")
    parser.add_argument("--admit-quality", action="store_true", help="validate owner-approved real report environment; never start workers")
    parser.add_argument("--verify-files", action="store_true", help="stream and hash every bank file; explicit offline operation")
    args = parser.parse_args(argv)
    if args.admit_quality:
        require(not args.verify_files, "quality admission and full bank hashing are separate operations")
        env = dict(os.environ)
        env["GLM53_CT_FORMAT"] = "lut"
        if args.manifest:
            env["GLM53_CT_LUT_MANIFEST"] = args.manifest
        print(json.dumps(preflight(env, require_production=False).admission, sort_keys=True))
        return 0
    require(bool(args.manifest), "--manifest is required")
    if args.verify_files:
        value = verify_bank(args.manifest)
        print(json.dumps(value, sort_keys=True))
    else:
        bank = load_manifest(args.manifest)
        print(json.dumps(dict(state="STRUCTURE_VALID", manifest_sha256=bank.sha256, real_pass=False, serving_admitted=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
