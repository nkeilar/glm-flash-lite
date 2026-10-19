#!/usr/bin/env python3
"""Fetch or independently verify the pinned public Q8_0 shards and imatrix.

Only ``fetch`` accesses the network. ``verify`` supports offline staging. A
partial file is never an admitted input; all nine complete files are hashed
before FETCH-DONE is published. Published files are never overwritten.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import os
from pathlib import Path
import shutil
import stat
import urllib.error
import urllib.request

from gguf import check_no_symlinks
from quantize_s1 import PINNED, REVISION

CHUNK = 4 * 2**20
ORIGIN = "https://huggingface.co/unsloth/GLM-5.3-Flash-GGUF/resolve/"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def publish_bytes(path, data):
    """Idempotent immutable publication; hard link makes no-clobber atomic."""
    check_no_symlinks(path)
    if path.exists():
        require(path.is_file() and path.read_bytes() == data, f"conflicting published file: {path}")
        return
    pending = path.with_name(path.name + ".pending")
    check_no_symlinks(pending)
    if pending.exists():
        require(pending.is_file() and pending.read_bytes() == data, f"conflicting interrupted publication: {pending}")
    else:
        with pending.open("xb") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    os.link(pending, path)
    pending.unlink()
    sync_directory(path.parent)


def verify_file(path, size, wanted):
    check_no_symlinks(path)
    with os.fdopen(os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK), "rb") as stream:
        before = os.fstat(stream.fileno())
        require(stat.S_ISREG(before.st_mode) and before.st_size == size, f"missing/incomplete source: {path}")
        digest = hashlib.sha256()
        while block := stream.read(CHUNK):
            digest.update(block)
        after = os.fstat(stream.fileno())
        identity = lambda st: (st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
        require(identity(before) == identity(after) == identity(path.stat()),
                f"source changed during verification: {path}")
    require(digest.hexdigest() == wanted, f"source SHA256 mismatch: {path}")
    return wanted


def fetch_file(path, url, size, wanted, opener=urllib.request.urlopen):
    check_no_symlinks(path)
    if path.exists():
        return verify_file(path, size, wanted)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    check_no_symlinks(partial)
    require(not partial.exists() or stat.S_ISREG(partial.stat().st_mode), "partial source must be a regular file")
    offset = partial.stat().st_size if partial.exists() else 0
    require(offset <= size, f"oversized partial source: {partial}")
    headers = {"Accept-Encoding": "identity", "User-Agent": "e1-recipe/1"}
    if offset:
        headers["Range"] = f"bytes={offset}-{size-1}"
    if offset < size:
        request = urllib.request.Request(url, headers=headers)
        with opener(request, timeout=90) as response:
            if offset:
                require(response.status == 206, "server ignored resume Range; partial retained")
                require(response.headers.get("Content-Range") == f"bytes {offset}-{size-1}/{size}",
                        "resume Content-Range mismatch")
            else:
                require(response.status == 200, "unexpected source response status")
            require(response.headers.get("Content-Encoding", "identity") == "identity", "encoded source response")
            require(response.headers.get("Content-Length") == str(size-offset), "source Content-Length mismatch")
            with partial.open("ab") as stream:
                remaining = size-offset
                while remaining:
                    block = response.read(min(CHUNK, remaining))
                    require(block and len(block) <= remaining, "short or oversized source response")
                    stream.write(block)
                    remaining -= len(block)
                require(not response.read(1), "source response exceeds pinned size")
                stream.flush()
                os.fsync(stream.fileno())
    verify_file(partial, size, wanted)
    os.link(partial, path)
    partial.unlink()
    sync_directory(path.parent)
    return wanted


@contextlib.contextmanager
def source_lock(root):
    check_no_symlinks(root)
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".fetch.lock"
    check_no_symlinks(lock)
    with lock.open("a+b") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def expected_bytes():
    return "".join(f"{name} {size} {sha}\n" for name, (size, sha) in PINNED.items()).encode()


def run(args):
    root = args.source.absolute()
    with source_lock(root):
        publish_bytes(root / "EXPECTED", expected_bytes())
        if args.command == "fetch":
            remaining = sum(size - (root/name).stat().st_size if (root/name).exists() else size
                            for name, (size, _) in PINNED.items())
            partial_bytes = sum(p.stat().st_size for p in root.rglob("*.partial") if p.is_file())
            needed = max(0, remaining-partial_bytes) + args.reserve_gib * 2**30
            require(shutil.disk_usage(root).free >= needed, "insufficient disk space for pinned sources plus reserve")
        for name, (size, sha) in PINNED.items():
            if args.command == "fetch":
                fetch_file(root/name, f"{ORIGIN}{REVISION}/{name}", size, sha)
            else:
                verify_file(root/name, size, sha)
            print(f"VERIFIED {sha} {name}", flush=True)
        publish_bytes(root / "FETCH-DONE", b"DONE\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("fetch", "verify", "plan"))
    parser.add_argument("--source", type=Path, default=Path("./e1-work/source"))
    parser.add_argument("--reserve-gib", type=int, default=4)
    args = parser.parse_args(argv)
    require(args.reserve_gib >= 0, "reserve must be nonnegative")
    if args.command == "plan":
        print(expected_bytes().decode(), end="")
        print(f"Total source bytes: {sum(size for size, _ in PINNED.values())}")
        return 0
    try:
        run(args)
    except (OSError, ValueError) as error:
        # Redirect URLs may contain signed query credentials; keep HTTP errors terse.
        if isinstance(error, urllib.error.HTTPError):
            parser.exit(2, f"HTTP {error.code}; partial files retained\n")
        parser.exit(2, f"{error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
