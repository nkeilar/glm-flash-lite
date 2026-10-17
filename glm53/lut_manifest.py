"""LUT bank validation and SHA-bound real-activation quality admission.

PROXY-PASS authorizes conversion only. Owner-approved real reports may admit
quality. Startup also requires the complete SHA-verified bank and independently
approved measured costs. No Torch or CUDA is imported by this module.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import stat
import statistics
import struct
import subprocess

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
COST_JSON_LIMIT = 4 * 2**20
COST_PROFILES = {"ordinary": [138, 138, 144], "upgraded_gate": [144, 144, 14], "upgraded_down": [138, 138, 14]}
_STARTUP_CACHE = {}


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
    report_s1 = value.get("scope") == "s1-real-check-only"
    report_scope_fields = ("scope", "full_bank_bound", "production_ready") if report_s1 else ()
    keys(value, ("schema", "state", "complete", "real_pass", "synthetic", "activation", "batch", "act_limit",
                 "tp2", "binding", "binding_sha256", "layers", "aggregate", *report_scope_fields), "real quality report")
    if report_s1:
        require(value["full_bank_bound"] is False and value["production_ready"] is False,
                "S1 real report must not claim completed-bank or production admission")
    require(value["schema"] == "e1-real-quality/v1" and value["state"] == "REAL-PASS" and
            value["complete"] is True and value["real_pass"] is True and value["synthetic"] is False,
            "complete REAL-PASS required; proxy and synthetic results cannot admit quality")
    require(value["activation"] == activation and activation in ("i16", "q8"), "real quality activation mismatch")
    require(integer(value["batch"], "real replay batch", 1, 4) in (1, 2, 4), "unsupported real replay batch")
    real_number(value["act_limit"], "model activation limit", positive=True)
    require(value["tp2"] == "split-intermediate-gate-up/down-columns-before-quantization", "real replay TP2 protocol mismatch")
    binding = value["binding"]
    s1_scope = binding.get("scope") == "s1-real-check-only"
    require(s1_scope == report_s1, "real report and binding scope differ")
    scope_fields = ("scope", "quant_manifest_sha256", "ud_manifest_sha256") if s1_scope else ("bank_manifest_sha256",)
    keys(binding, ("protocol", *scope_fields, "source_revision", "imatrix_sha256", "capture_manifest_sha256",
                   "capture", "replay", "layers"), "real quality binding")
    require(digest(value["binding_sha256"], "quality binding") == json_digest(binding), "real quality binding SHA256 mismatch")
    require(binding["protocol"] == REAL_PROTOCOL and binding["source_revision"] == SOURCE_REVISION and
            binding["imatrix_sha256"] == IMATRIX_SHA256, "real quality source/imatrix/protocol mismatch")
    if s1_scope:
        # The approved report predates bank publication. Its exact tested layer
        # artifacts must survive unchanged in the complete validated bank below.
        for field in ("quant_manifest_sha256", "ud_manifest_sha256"):
            digest(binding[field], field)
    else:
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
        keys(weights, ("q8_sha256", "ours_sha256", "unsloth_sha256", *(["conversion_context"] if s1_scope else [])),
             "real weight binding")
        for key in ("q8_sha256", "ours_sha256", "unsloth_sha256"):
            digest(weights[key], key)
        if s1_scope:
            context = weights["conversion_context"]
            keys(context, ("imatrix", "layer", "metadata_source", "normalized_imatrix_sha256", "quantizer", "recipe",
                           "revision", "sources", "tools"), "tested conversion context")
            require(context["revision"] == SOURCE_REVISION and context["imatrix"] == IMATRIX_SHA256 and
                    type(context["layer"]) is int and context["layer"] == layer, "tested conversion provenance mismatch")
            expected_recipe = ",".join("^blk\\." + str(layer) + "\\.ffn_" + part + "_exps\\.weight$=" +
                                       {138: "iq3_k", 144: "iq4_ks", 14: "q6_K"}[recipe(layer, part)] for part in PARTS)
            require(context["recipe"] == expected_recipe, "tested conversion recipe differs from bank")
            for field in ("metadata_source", "normalized_imatrix_sha256"):
                digest(context[field], field)
            for field in ("quantizer", "sources", "tools"):
                require(isinstance(context[field], dict) and bool(context[field]), "tested conversion hash inventory missing")
                for name, sha in context[field].items():
                    require(isinstance(name, str) and bool(name), "invalid tested conversion inventory name")
                    digest(sha, field)
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
    """Match the loaded model to the admitted real replay before bank allocation."""
    require(settings.format == "lut" and settings.admission is not None, "real LUT admission required")
    expected = settings.admission["act_limit"]
    modules = tuple(modules)
    require(bool(modules), "model MoE modules required for activation-limit validation")
    for module in modules:
        actual = getattr(module, "act_limit", None)
        real_number(actual, "actual model activation limit", positive=True)
        require(actual == expected, "actual model activation limit differs from real quality admission")
    return expected


def cost_profile(layer):
    integer(layer, "model layer", 3, 44)
    return "upgraded_gate" if layer == 11 else "upgraded_down" if layer in (12, 44) else "ordinary"


def validate_cost_table(value, bank, activation):
    """Check a complete measured grid; never extrapolate unmeasured profiles."""
    require(isinstance(value, dict) and value.get("schema") == "e1-lut-cost/v1", "measured LUT cost artifact required")
    require(value.get("complete") is True, "complete measured LUT costs required for all three layer profiles")
    require(value.get("activation") == activation, "measured LUT cost activation mismatch")
    s1_cost = value.get("bank_manifest_sha256") is None and "s1_manifest_sha256" in value
    if s1_cost:
        digest(value["s1_manifest_sha256"], "cost S1 manifest")
    else:
        require(digest(value.get("bank_manifest_sha256"), "cost bank manifest") == bank.sha256,
                "measured LUT costs belong to another bank manifest")
    measurement = value.get("measurement")
    require(isinstance(measurement, dict) and measurement.get("kind") == "hardware" and
            measurement.get("placement_verified") is True and measurement.get("clock") == "perf_counter_ns",
            "costs require hardware measurements with verified physical placement")
    require(measurement.get("treatment") == "clflush-owned-weights-before-each-call",
            "serving LUT costs require controlled cold owned weights flushed before each timed call")
    real_number(measurement.get("act_limit"), "measured model activation limit", positive=True)
    for field in ("hardware_sha256", "build_sha256"):
        digest(measurement.get(field), "cost " + field)
    hardware = measurement.get("hardware")
    require(isinstance(hardware, dict) and json_digest(hardware) == measurement["hardware_sha256"],
            "measured cost hardware identity hash mismatch")
    cpus = measurement.get("cpus")
    require(isinstance(cpus, list) and len(cpus) == 42 and all(type(cpu) is int and 0 <= cpu < 1024 for cpu in cpus)
            and len(set(cpus)) == 42 and hardware.get("cpus") == cpus, "measured cost worker CPUs must name 42 distinct workers")
    coordinator = integer(measurement.get("coordinator_cpu"), "measured coordinator CPU", 0, 1023)
    require(coordinator not in cpus, "measured coordinator CPU must differ from worker CPUs")
    topology = hardware.get("topology")
    keys(topology, (str(cpu) for cpu in cpus), "measured worker topology")
    physical = set()
    for row in topology.values():
        keys(row, ("package", "core"), "measured physical core")
        require(all(isinstance(value, str) and value.isdigit() and len(value) <= 8 for value in row.values()),
                "invalid measured physical core")
        physical.add((row["package"], row["core"]))
    require(len(physical) == 42, "measured worker CPUs must use distinct physical cores")
    count = integer(measurement.get("samples_per_cell"), "cost samples per cell", 3, 10000)
    profiles = value.get("profiles")
    keys(profiles, COST_PROFILES, "measured LUT profiles")
    for name, types in COST_PROFILES.items():
        profile = profiles[name]
        require(isinstance(profile, dict) and profile.get("types") == types and
                all(type(item) is int for item in profile["types"]), "measured LUT profile recipe mismatch")
        digest(profile.get("fixture_sha256"), "cost fixture")
        if s1_cost:
            # Timing a recipe may combine actual already-tested tensors. Every
            # donor still must be byte-identical to its full-bank artifact.
            parts = profile.get("source_parts")
            require(isinstance(parts, list) and len(parts) == 3, "three measured tensor source parts required")
            expected_layers = (11, 11, 11) if name == "upgraded_gate" else (3, 3, 11) if name == "upgraded_down" else (3, 3, 3)
            bank_layers = {row["layer"]: row for row in bank.value["layers"]}
            for source, part, source_layer, typ in zip(parts, PARTS, expected_layers, types):
                keys(source, ("part", "layer", "sha256", "type_id"), "measured tensor source")
                require(source["part"] == part and type(source["layer"]) is int and source["layer"] == source_layer and
                        type(source["type_id"]) is int and source["type_id"] == typ and recipe(source_layer, part) == typ,
                        "measured tensor source recipe mismatch")
                require(digest(source["sha256"], "measured tensor source") == bank_layers[source_layer]["artifact"]["sha256"],
                        "measured tensor source differs from admitted bank")
        else:
            digest(profile.get("source_artifact_sha256"), "cost source artifact")
            matching = [row for row in bank.value["layers"] if cost_profile(row["layer"]) == name]
            require(profile["source_artifact_sha256"] in {row["artifact"]["sha256"] for row in matching},
                    "measured LUT fixture does not belong to the admitted bank/profile")
        rows = profile.get("rows")
        keys(rows, (str(m) for m in range(1, 5)), "measured M grid")
        for m, cells in rows.items():
            keys(cells, (str(k) for k in range(9)), "measured K grid")
            for k, cell in cells.items():
                require(isinstance(cell, dict), "invalid measured cost cell")
                samples = integer(cell.get("samples"), "measured cell samples", count, count)
                raw = cell.get("raw_ms")
                require(isinstance(raw, list) and len(raw) == samples, "measured cost samples missing")
                for sample in raw:
                    require(real_number(sample, "measured cost sample", positive=k != "0") <= 3600000,
                            "measured cost exceeds supported job duration")
                median = real_number(cell.get("median_ms"), "measured median", positive=k != "0")
                require(math.isclose(median, statistics.median(raw), rel_tol=1e-12, abs_tol=1e-15),
                        "measured cost median differs from raw samples")
    return value


def validate_cost_hardware(costs, cpus, coordinator=None):
    """Compare stable worker geometry; recorded CPU MHz is intentionally ignored."""
    measurement = costs["measurement"]
    coordinator = measurement["coordinator_cpu"] if coordinator is None else coordinator
    integer(coordinator, "serving coordinator CPU", 0, 1023)
    require(coordinator == measurement["coordinator_cpu"], "serving coordinator CPU differs from measured LUT costs")
    require(list(cpus) == measurement["cpus"], "serving worker CPUs differ from measured LUT costs")
    for cpu in cpus:
        root = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
        current = dict(package=(root / "physical_package_id").read_text().strip(),
                       core=(root / "core_id").read_text().strip())
        require(current == measurement["hardware"]["topology"][str(cpu)],
                "serving physical-core topology differs from measured LUT costs")
    root = Path(f"/sys/devices/system/cpu/cpu{coordinator}/topology")
    core = ((root / "physical_package_id").read_text().strip(), (root / "core_id").read_text().strip())
    workers = {(row["package"], row["core"]) for row in measurement["hardware"]["topology"].values()}
    require(core not in workers, "LUT coordinator must use a separate physical core from workers")


def load_cost_table(path, expected_sha256, bank, activation):
    require(path is not None, "GLM53_CT_LUT_COST is required: measured LUT costs are a separate startup gate")
    digest(expected_sha256, "owner-approved measured cost artifact")
    value, actual = read_json(path, COST_JSON_LIMIT, "measured cost artifact")
    require(actual == expected_sha256, "measured cost artifact does not match the owner-approved SHA256")
    return validate_cost_table(value, bank, activation)


def hash_regular(path):
    with open_regular(path) as (stream, before):
        result = hashlib.sha256()
        while True:
            chunk = stream.read(HASH_CHUNK)
            if not chunk:
                break
            result.update(chunk)
        unchanged(path, stream, before)
    return result.hexdigest(), before


SPLIT_MODULE = "ft_split_fast.so"
SPLIT_SOURCE = "kernels/k110/ft_split_fast.cu"


def load_precompiled(path, expected_sha256):
    """Admit exact reviewed modules without invoking JIT or initializing CUDA."""
    require(path is not None, "GLM53_CT_LUT_PRECOMPILE is required; compile before the serving window")
    digest(expected_sha256, "owner-approved LUT precompile receipt")
    path = no_symlinks(path)
    receipt, actual = read_json(path, JSON_LIMIT, "LUT precompile receipt")
    require(actual == expected_sha256, "LUT precompile receipt does not match the owner-approved SHA256")
    require(isinstance(receipt, dict) and receipt.get("schema") == "e1-precompile/v1" and
            receipt.get("variant") == "lut" and receipt.get("state") == "COMPILED_NO_GPU_CONTEXT" and
            receipt.get("cuda_initialized") is False, "reviewed LUT precompile receipt required")
    revision = receipt.get("source_revision")
    require(isinstance(revision, str) and len(revision) == 40 and all(c in "0123456789abcdef" for c in revision),
            "invalid LUT precompile source revision")
    runtime_version = receipt.get("torch_version")
    require(isinstance(runtime_version, str) and bool(runtime_version) and
            runtime_version.split("+", 1)[0] == importlib.metadata.version("torch"),
            "LUT precompile Torch metadata/public runtime version mismatch")
    require("-DFT_ENABLE_LUT=1" in receipt.get("flags", []) and
            "-DFT_ENABLE_LUT=1" in receipt.get("cuda_flags", []), "both precompiled LUT modules require guarded LUT build flags")
    sources = receipt.get("source_sha256")
    required = {"kernels/cpu_avx2/" + name for name in
                ("ft_tier_ext.cpp", "ft_tier_cu.cu", "ft_core.h", "ft_lut.cpp", "ft_lut.h", "lut/kernels.cpp", "lut/kernels.h")}
    require(isinstance(sources, dict) and required <= set(sources), "precompiled LUT native source bindings missing")
    root = Path(__file__).absolute().parent.parent
    try:
        head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        raise ValueError("reviewed serving checkout revision is unavailable") from error
    require(head == revision, "serving checkout revision differs from the reviewed LUT precompile")
    for relative, expected in sources.items():
        require(isinstance(relative, str) and (relative.startswith("kernels/cpu_avx2/") or relative == SPLIT_SOURCE) and
                not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                "invalid precompile source path")
        digest(expected, "precompile source")
        require(hash_regular(root / relative)[0] == expected, "LUT native source differs from the reviewed precompile")
    outputs = receipt.get("outputs")
    base = {"ft_tier_host.so", "ft_tier_cu.so"}
    require(isinstance(outputs, dict) and set(outputs) in (base, base | {SPLIT_MODULE}), "invalid fields: LUT precompiled modules")
    if SPLIT_MODULE in outputs:   # optional third module: the k110 fast split kernel built with the LUT ingress guard
        require(SPLIT_SOURCE in sources and "-DFT_ENABLE_LUT=1" in receipt.get("split_cuda_flags", []),
                "precompiled fast split module requires its bound source and guarded LUT build flags")
    require(no_symlinks(receipt.get("build_directory", "")) == path.parent,
            "LUT precompile build directory differs from its receipt")
    modules = {}
    for name, expected in outputs.items():
        digest(expected, "precompiled module")
        file = path.parent / name
        actual, before = hash_regular(file)
        require(actual == expected, "precompiled LUT module SHA256 mismatch")
        modules[name] = dict(path=str(file), sha256=actual, identity=before)
    return dict(receipt=receipt, modules=modules)


def validate_staged_identities(staged):
    """Recheck validated files before passing their bounded extents to C++."""
    for row in staged["layers"]:
        path = no_symlinks(row["path"])
        st = path.stat(follow_symlinks=False)
        require(stat.S_ISREG(st.st_mode) and identity(st) == row["identity"],
                "SHA-verified bank artifact changed before native initialization")


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
    cost_path: Path | None = None
    cost_sha256: str | None = None
    costs: dict | None = None
    staged_bank: dict | None = None
    precompile_path: Path | None = None
    precompile_sha256: str | None = None
    precompiled: dict | None = None


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
                        env.get("GLM53_CT_LUT_Q8_QUALITY_SHA256"),
                        cost_path=Path(env["GLM53_CT_LUT_COST"]) if env.get("GLM53_CT_LUT_COST") else None,
                        cost_sha256=env.get("GLM53_CT_LUT_COST_SHA256"),
                        precompile_path=Path(env["GLM53_CT_LUT_PRECOMPILE"]) if env.get("GLM53_CT_LUT_PRECOMPILE") else None,
                        precompile_sha256=env.get("GLM53_CT_LUT_PRECOMPILE_SHA256"))


def preflight(environ=None, *, require_production=True):
    """Admit real quality, then verify measured costs and all bank bytes for startup.

    Quality-only callers leave require_production=False. Verified bank bytes are
    cached only while every original file identity and all admission digests match.
    The native loader rechecks these identities before and after its bounded copy.
    """
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
    costs = staged = precompiled = None
    if require_production:
        env = os.environ if environ is None else environ
        for name in ("GLM53_CPU_TIER", "GLM53_EC"):
            require(env.get(name) == "1", name + "=1 is required for LUT startup")
        require(bool(env.get("GLM53_ZC_VRAM")), "pinned GPU-readable trellis weights require GLM53_ZC_VRAM")
        # GPU-side speed features (T-71): lookahead needs the per-token LUT check (cpu_tier._lut_token_check), and the
        # fast split kernel is admitted only as the guarded precompiled module checked after load_precompiled.
        for name in ("GLM53_LA", "GLM53_K_HCFUSE", "GLM53_K_OVL", "GLM53_K_FTSPLIT"):
            require(env.get(name, "0") in ("0", "1"), name + " must be 0 or 1 for the LUT path")
        require(env.get("GLM53_CT_VALIDATE", "0") != "1", "LUT quality uses real replay and the normal full-model panel; GLM53_CT_VALIDATE is trellis-only")
        maxbsz = int(env.get("GLM53_CT_MAXBSZ", "4096" if env.get("GLM53_CT_VALIDATE") == "1" else "4"))
        require(1 <= maxbsz <= 4, "LUT supports M=1..4; set GLM53_CT_MAXBSZ accordingly")
        require(0 <= int(env.get("GLM53_CT_MAXN", "8")) <= 8, "LUT GLM53_CT_MAXN must be between 0 and 8")
        require(-1 <= int(env.get("GLM53_CT_FORCE_N", "-1")) <= 8, "LUT GLM53_CT_FORCE_N must be between -1 and 8")
        timeout = real_number(float(env.get("GLM53_CT_TIMEOUT_S", "2")), "LUT request timeout", positive=True)
        require(.001 <= timeout <= 3600, "LUT request timeout must be .001..3600 seconds")
        for name, default in (("GLM53_CT_TZC", ".40"), ("GLM53_CT_THIT", ".023"),
                              ("GLM53_CT_A", ".11"), ("GLM53_CT_B", ".104"),
                              ("GLM53_CT_RA", ".37"), ("GLM53_CT_RB", ".03"), ("GLM53_RT_B", ".15")):
            require(real_number(float(env.get(name, default)), name) <= 3600000, "LUT policy cost exceeds supported duration")
        require(not env.get("GLM53_CT_LUT", ""), "use SHA-approved GLM53_CT_LUT_COST for LUT format; GLM53_CT_LUT remains the legacy numeric table")
        costs = load_cost_table(settings.cost_path, settings.cost_sha256, bank, settings.activation)
        require(costs["measurement"]["act_limit"] == admission["act_limit"],
                "measured cost activation limit differs from real quality admission")
        precompiled = load_precompiled(settings.precompile_path, settings.precompile_sha256)
        require(precompiled["modules"]["ft_tier_host.so"]["sha256"] == costs["measurement"]["build_sha256"],
                "precompiled LUT host differs from measured cost build")
        require(env.get("GLM53_K_FTSPLIT", "0") != "1" or SPLIT_MODULE in precompiled["modules"],
                "GLM53_K_FTSPLIT=1 with LUT requires the guarded fast split module in the reviewed precompile")
        cache_key = (str(bank.path), bank.sha256, settings.activation, tuple(sorted(reports.items())), settings.cost_sha256)
        staged = _STARTUP_CACHE.get(cache_key)
        if staged is None:
            staged = verify_bank(bank.path)
            require(staged["manifest_sha256"] == bank.sha256, "bank manifest changed during startup admission")
        validate_staged_identities(staged)
        _STARTUP_CACHE.clear()  # One startup bank; never accumulate model inventories.
        _STARTUP_CACHE[cache_key] = staged
        admission.update(full_bank_verified=True, measured_cost_sha256=settings.cost_sha256)
    return TierSettings(settings.format, settings.activation, settings.manifest, settings.quality,
                        settings.quality_sha256, settings.q8_quality, settings.q8_quality_sha256, admission,
                        settings.cost_path, settings.cost_sha256, costs, staged,
                        settings.precompile_path, settings.precompile_sha256, precompiled)


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
