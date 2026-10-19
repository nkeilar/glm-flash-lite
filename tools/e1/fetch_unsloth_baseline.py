#!/usr/bin/env python3
"""Bounded/resumable exact publisher tensor fetch. Does not quantize weights."""
import argparse
import errno
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import time
import urllib.parse
import urllib.error
import urllib.request

from gguf import check_no_symlinks, read_gguf, write_gguf
from quality_checkpoint import MAX_PUBLISHER_RANGE, SOURCE_REV, UD_HEADER_SHA, admit_output, require, sha256
from maintenance_guard import publish_json as atomic_json, publish_file, add_arguments, configure, resources

SUMMARY_SHA256 = UD_HEADER_SHA
CHUNK = 4 * 2**20
DEFAULT_RANGE_MIB = 32
RANGE_ATTEMPTS = 3
ORIGIN = f"https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/tree/{SOURCE_REV}/UD-IQ4_XS"


def range_bytes(mib):
    require(type(mib) is int and 1<=mib<=MAX_PUBLISHER_RANGE//2**20,"--range-mib must be an integer from 1 to 64")
    return mib*2**20


def checked_range(opener, url, start, size, total, etag, timeout=90):
    require(type(size) is int and type(start) is int and type(total) is int and
            0 < size <= MAX_PUBLISHER_RANGE and 0 <= start <= total-size, "invalid bounded tensor range")
    end = start+size-1
    require(math.isfinite(timeout) and 0<timeout<=90,"range request deadline expired")
    request = urllib.request.Request(url,headers={"Range":f"bytes={start}-{end}","If-Match":etag,
                  "Accept-Encoding":"identity","User-Agent":"e1-offline-quality/1"})
    request_end=time.monotonic()+timeout
    with opener(request,timeout=timeout) as response:
        # All checks happen before reading the body. A server ignoring Range
        # cannot turn a small requested chunk into an accidental shard download.
        require(response.status == 206, f"range server returned {response.status}; body not read")
        require(response.headers.get("Content-Range") == f"bytes {start}-{end}/{total}", "range Content-Range mismatch")
        require(response.headers.get("Content-Length") == str(size), "range Content-Length mismatch")
        require(response.headers.get("Content-Encoding","identity") == "identity", "encoded range response")
        require(response.headers.get("ETag") == etag, "publisher ETag changed")
        chunks=[]
        received=0
        while received<size:
            remaining=request_end-time.monotonic()
            require(remaining>0,"publisher range body deadline reached")
            sock=getattr(getattr(getattr(response,"fp",None),"raw",None),"_sock",None)
            if sock is not None: sock.settimeout(remaining)
            block=getattr(response,"read1",response.read)(min(65536,size-received))
            require(block,"short publisher range")
            chunks.append(block)
            received+=len(block)
        payload=b"".join(chunks)
        require(len(payload) == size, "short publisher range")
        return payload


def network_failure(error):
    """Return a fixed safe reason and retryability; never expose redirect URLs."""
    if isinstance(error,urllib.error.HTTPError):
        return f"HTTP_{error.code}",error.code==429 or 500<=error.code<=599
    reason=error.reason if isinstance(error,urllib.error.URLError) else error
    if isinstance(reason,socket.gaierror): return f"DNS_{reason.errno}",reason.errno==socket.EAI_AGAIN
    if isinstance(reason,TimeoutError): return "TIMEOUT",True
    if isinstance(reason,ConnectionError): return "CONNECTION_ERROR",True
    transient={errno.ECONNRESET,errno.ECONNREFUSED,errno.ECONNABORTED,errno.ETIMEDOUT,errno.EHOSTUNREACH,errno.ENETUNREACH,errno.EPIPE}
    if isinstance(reason,OSError) and reason.errno in transient:
        return f"NETWORK_ERRNO_{reason.errno}",True
    return "NONTRANSIENT_"+type(reason).__name__,False


def retry_range(opener,url,start,size,total,etag,tick,remaining,attempts=RANGE_ATTEMPTS,sleep=None,report=None):
    require(type(attempts) is int and 1<=attempts<=RANGE_ATTEMPTS,"invalid bounded range attempt count")
    sleep=sleep or time.sleep
    report=report or (lambda event:print(json.dumps(event),flush=True))
    for attempt in range(1,attempts+1):
        tick()
        available=remaining()
        require(math.isfinite(available) and available>0,"publisher deadline reached before range attempt")
        try:
            payload=checked_range(opener,url,start,size,total,etag,timeout=min(90,available))
        except OSError as error:
            reason,retriable=network_failure(error)
            if isinstance(error,urllib.error.HTTPError):
                try: error.close()
                except OSError: pass
        else:
            tick()
            return payload
        # Outside the exception handler: a guard/deadline failure cannot chain
        # the original network exception and leak a signed redirect URL.
        if not retriable:
            raise RuntimeError("non-retryable publisher network failure: "+reason) from None
        if attempt==attempts:
            report({"event":"PUBLISHER_RANGE_RETRIES_EXHAUSTED","attempts":attempt,"reason":reason,"start":start,"bytes":size})
            raise RuntimeError(f"publisher range retry exhausted after {attempt} attempts: {reason}") from None
        tick()
        delay=float(attempt)  # Three total attempts; only 1s and 2s backoffs.
        available=remaining()
        require(math.isfinite(available) and available>delay,"publisher deadline leaves no retry backoff budget")
        report({"event":"PUBLISHER_RANGE_RETRY","attempt":attempt,"next_attempt":attempt+1,"reason":reason,
                "backoff_seconds":delay,"start":start,"bytes":size})
        sleep(delay)
        tick()


def fetch_tensor(out, tensor, shard, tick, opener=urllib.request.urlopen, remaining=lambda:90, progress=lambda *args:None,
                 request_bytes=DEFAULT_RANGE_MIB*2**20):
    require(type(request_bytes) is int and 0<request_bytes<=MAX_PUBLISHER_RANGE,"invalid configured publisher range size")
    name = tensor["name"]
    payload_path, log_path = out/(name+".payload"), out/(name+".ranges.jsonl")
    for p in (out,payload_path,log_path): check_no_symlinks(p)
    start = shard["tensor_data_start"]+tensor["data_relative_offset"]
    total = tensor["tensor_bytes"]
    url = f"https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/{SOURCE_REV}/{urllib.parse.quote(shard['file'])}"
    require(shard.get("lfs_sha256") and shard.get("etag"), "source header lacks publisher identity")
    records = [json.loads(line) for line in log_path.read_text().splitlines()] if log_path.exists() else []
    offset = 0
    if payload_path.exists():
        with payload_path.open("rb") as f:
            for record in records:
                tick()
                require(record["tensor"] == name and record["url"] == url and record["start"] == start+offset and
                        record["etag"] == shard["etag"] and record["lfs_sha256"] == shard["lfs_sha256"], "resume provenance mismatch")
                size=record.get("bytes")
                require(type(size) is int and 0<size<=MAX_PUBLISHER_RANGE and offset+size<=total and
                        record.get("end")==record["start"]+size-1,"resume chunk bounds invalid before payload read")
                block = f.read(record["bytes"])
                require(len(block) == record["bytes"] and hashlib.sha256(block).hexdigest() == record["sha256"], "resume chunk hash mismatch")
                offset += len(block)
                require(offset <= total, "resume range exceeds tensor")
        # Only an unlogged trailing portion of our explicitly owned scratch file
        # can be truncated, after every retained chunk has passed SHA256 checks.
        with payload_path.open("r+b") as f: f.truncate(offset)
    else:
        require(not records, "range ledger exists without payload")
    last_progress=offset
    with payload_path.open("ab") as payload, log_path.open("a") as log:
        while offset < total:
            tick()
            n = min(request_bytes,total-offset)
            block = retry_range(opener,url,start+offset,n,shard["file_size"],shard["etag"],tick,remaining)
            tick()
            payload.write(block)
            payload.flush()
            os.fsync(payload.fileno())
            record = {"tensor":name,"url":url,"start":start+offset,"end":start+offset+n-1,"bytes":n,
                      "etag":shard["etag"],"lfs_sha256":shard["lfs_sha256"],"sha256":hashlib.sha256(block).hexdigest()}
            log.write(json.dumps(record)+"\n")
            log.flush()
            os.fsync(log.fileno())
            records.append(record)
            offset += n
            if offset-last_progress>=64*2**20 or offset==total:
                progress(name,offset,total)
                last_progress=offset
    require(payload_path.stat().st_size == total, "incomplete tensor payload")
    return payload_path, records


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--header-summary", type=Path, default=Path(__file__).absolute().with_name("ud-source.json"))
    ap.add_argument("--layers", default="3,11")
    ap.add_argument("--out", default="./e1-work/ud")
    ap.add_argument("--local-shards", type=Path, help="verify full publisher shards here instead of using HTTP")
    ap.add_argument("--plan-only", action="store_true")
    ap.add_argument("--budget-seconds", type=float, default=21600)
    ap.add_argument("--range-mib", type=int, default=DEFAULT_RANGE_MIB)
    add_arguments(ap)
    args = ap.parse_args(argv)
    request_bytes = range_bytes(args.range_mib)
    require(sha256(args.header_summary) == SUMMARY_SHA256, "publisher inventory differs from pinned public bytes")
    summary = json.loads(args.header_summary.read_text())
    require(summary["revision"] == SOURCE_REV and summary["source_url"] == ORIGIN, "publisher source mismatch")
    layers = sorted({int(v) for v in args.layers.split(",")})
    require(layers and set(layers) <= {3, 11}, "quality replay uses layers 3 and 11")
    shards = {s["file"]: s for s in summary["shards"]}
    tensors = {t["name"]: t for t in summary["expert_tensors"]}
    selected = {layer: [tensors[f"blk.{layer}.ffn_{part}_exps.weight"] for part in ("gate", "up", "down")] for layer in layers}
    plan = dict(revision=SOURCE_REV, layers=layers, summary_sha256=SUMMARY_SHA256,
                tensor_bytes=sum(t["tensor_bytes"] for ts in selected.values() for t in ts))
    if args.plan_only:
        print(json.dumps(plan, indent=2))
        return 0
    require(math.isfinite(args.budget_seconds) and args.budget_seconds > 0, "fetch budget must be positive and finite")
    configure(args)
    end_time = time.monotonic() + args.budget_seconds
    out = admit_output(args.out)
    out.mkdir(parents=True, exist_ok=True)
    import fcntl
    check_no_symlinks(out / ".fetch.lock")
    lock = (out / ".fetch.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def tick():
        require(time.monotonic() < end_time, "publisher fetch budget expired; rerun to resume verified chunks")
        resources(out, min_ram_gib=args.min_ram_gib, min_disk_gib=args.min_disk_gib)
    tick()
    marker = out / "fetch-plan.json"
    check_no_symlinks(marker)
    if marker.exists():
        require(json.loads(marker.read_text()) == plan, "resume directory has a different plan")
    else:
        require(set(p.name for p in out.iterdir()) <= {".fetch.lock"}, "new fetch directory must be empty")
        atomic_json(marker, plan)
    summary_copy = out / "ud-source.json"
    check_no_symlinks(summary_copy)
    if summary_copy.exists():
        require(sha256(summary_copy) == SUMMARY_SHA256, "copied inventory changed")
    else:
        temporary = out / (".inventory-" + str(os.getpid()))
        with temporary.open("xb") as stream:
            stream.write(args.header_summary.read_bytes())
        publish_file(temporary, summary_copy)
    local = {}
    if args.local_shards:
        for name in {t["shard"] for ts in selected.values() for t in ts}:
            file = args.local_shards / name
            check_no_symlinks(file)
            require(file.stat().st_size == shards[name]["file_size"] and sha256(file, tick) == shards[name]["lfs_sha256"], "local publisher shard size/SHA256 mismatch")
            local[name] = file
    complete = dict(schema="e1-ud/v1", revision=SOURCE_REV,
                    header_summary=dict(path=summary_copy.name, sha256=SUMMARY_SHA256), layers=[])
    for layer, items in selected.items():
        descriptors, ranges = [], []
        for tensor in items:
            shard = shards[tensor["shard"]]
            if local:
                name = tensor["name"]
                payload = out / (name + ".payload")
                check_no_symlinks(payload)
                start = shard["tensor_data_start"] + tensor["data_relative_offset"]
                records = []
                temporary = out / ("." + name + ".building")
                with local[tensor["shard"]].open("rb") as source, temporary.open("xb") as target:
                    source.seek(start)
                    offset = 0
                    while offset < tensor["tensor_bytes"]:
                        tick()
                        size = min(request_bytes, tensor["tensor_bytes"] - offset)
                        block = source.read(size)
                        require(len(block) == size, "local publisher tensor truncated")
                        target.write(block)
                        records.append(dict(tensor=name, url=f"https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/{SOURCE_REV}/{tensor['shard']}",
                                            start=start+offset, end=start+offset+size-1, bytes=size, etag=shard["etag"],
                                            lfs_sha256=shard["lfs_sha256"], sha256=hashlib.sha256(block).hexdigest()))
                        offset += size
                if payload.exists():
                    require(sha256(payload, tick) == sha256(temporary, tick), "existing local payload differs")
                    temporary.unlink()
                else:
                    publish_file(temporary, payload)
            else:
                payload, records = fetch_tensor(out, tensor, shard, tick, remaining=lambda: end_time-time.monotonic(), request_bytes=request_bytes)
            descriptors.append(dict(name=tensor["name"], dims=tensor["ggml_dimensions"], type_id=tensor["type_id"], data=payload))
            ranges.extend(records)
        final = out / f"blk.{layer:02d}.ud.gguf"
        check_no_symlinks(final)
        if final.exists():
            existing = read_gguf(final)
            require({t["name"] for t in existing.tensors} == {t["name"] for t in items}, "existing UD tensor mismatch")
            with final.open("rb") as source:
                for descriptor in descriptors:
                    tensor = existing.tensor(descriptor["name"])
                    require(tensor["dims"] == descriptor["dims"] and tensor["type_id"] == descriptor["type_id"], "existing UD shape/type mismatch")
                    source.seek(existing.data_start + tensor["offset"])
                    digest, remaining = hashlib.sha256(), tensor["nbytes"]
                    while remaining:
                        tick()
                        block = source.read(min(CHUNK, remaining))
                        require(block, "existing UD payload truncated")
                        digest.update(block)
                        remaining -= len(block)
                    require(digest.hexdigest() == sha256(descriptor["data"], tick), "existing UD payload SHA256 mismatch")
        else:
            temporary = out / f".blk.{layer:02d}.{os.getpid()}.building.gguf"
            write_gguf(temporary, {"general.architecture": (8, "glm5-next"), "general.alignment": (4, 32)}, descriptors, tick=tick)
            publish_file(temporary, final)
        complete["layers"].append(dict(layer=layer, origin=ORIGIN, gguf=dict(path=final.name, sha256=sha256(final, tick)), ranges=ranges))
    destination = out / "unsloth.json"
    if destination.exists():
        require(json.loads(destination.read_text()) == complete, "existing UD receipt differs")
    else:
        atomic_json(destination, complete)
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
