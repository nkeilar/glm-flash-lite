"""Offline LUT bank validation and a deliberately closed serving admission gate.

The public bank manifest records conversion provenance. Its PROXY-PASS permits
conversion only. Server startup remains disabled until reviewed real-activation
quality admission is implemented. This module imports neither Torch nor CUDA.
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
REAL_ADMISSION_MESSAGE = (
    "LUT server startup is disabled: reviewed real-activation quality admission "
    "is not implemented. PROXY-PASS authorizes conversion only; "
    "GLM53_CT_LUT_QUALITY cannot enable startup."
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
            identity(Path(path).stat(follow_symlinks=False)) == before,
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


def load_manifest(path, check_files=True):
    path = no_symlinks(path)
    with open_regular(path) as (stream, before):
        payload = stream.read(JSON_LIMIT + 1)
        require(len(payload) <= JSON_LIMIT, "public manifest exceeds bounded JSON size")
        value = json.loads(payload, object_pairs_hook=unique_pairs, parse_constant=finite_constant)
        validate_structure(value)
        unchanged(path, stream, before)
    if check_files:
        for row in value["layers"]:
            artifact = row["artifact"]
            file = no_symlinks(path.parent / artifact["path"])
            st = file.stat(follow_symlinks=False)
            require(stat.S_ISREG(st.st_mode) and st.st_size == artifact["bytes"],
                    f"artifact missing, nonregular or wrong size: {artifact['path']}")
    return BankManifest(path=path, sha256=hashlib.sha256(payload).hexdigest(), value=value)


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


@dataclass(frozen=True)
class TierSettings:
    format: str
    activation: str
    manifest: Path | None
    quality: Path | None


def parse_env(environ=None):
    env = os.environ if environ is None else environ
    fmt, activation = env.get("GLM53_CT_FORMAT", "trellis"), env.get("GLM53_CT_LUT_ACT", "i16")
    require(fmt in ("trellis", "lut"), "GLM53_CT_FORMAT must be trellis or lut")
    require(activation in ("i16", "q8"), "GLM53_CT_LUT_ACT must be i16 or q8")
    manifest, quality = env.get("GLM53_CT_LUT_MANIFEST"), env.get("GLM53_CT_LUT_QUALITY")
    require(fmt != "lut" or bool(manifest), "GLM53_CT_LUT_MANIFEST is required for LUT format")
    return TierSettings(fmt, activation, Path(manifest) if manifest else None, Path(quality) if quality else None)


def preflight(environ=None):
    """Run before model allocation, extension builds, or worker startup."""
    settings = parse_env(environ)
    if settings.format == "trellis":
        return settings
    load_manifest(settings.manifest)  # Structure and cheap stat checks only.
    raise RuntimeError(REAL_ADMISSION_MESSAGE)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--verify-files", action="store_true", help="stream and hash every bank file; explicit offline operation")
    args = parser.parse_args(argv)
    if args.verify_files:
        value = verify_bank(args.manifest)
        print(json.dumps(value, sort_keys=True))
    else:
        bank = load_manifest(args.manifest)
        print(json.dumps(dict(state="STRUCTURE_VALID", manifest_sha256=bank.sha256, real_pass=False, serving_admitted=False)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
