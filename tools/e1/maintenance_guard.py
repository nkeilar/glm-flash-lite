"""Optional local resource checks and immutable atomic artifact publication.

The legacy report field ``maintenance_guard_sha256`` binds this public helper;
there are no server identities, maintenance windows or process supervisors here.
"""
import json
import math
import os
from pathlib import Path
import shutil
import tempfile


def publish_file(temporary, destination):
    """Atomically publish a completed file, failing if the name already exists."""
    temporary, destination = Path(temporary), Path(destination)
    from gguf import check_no_symlinks
    check_no_symlinks(destination)
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    os.link(temporary, destination)  # Same filesystem; exclusive, unlike rename().
    temporary.unlink()
    descriptor = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def publish_json(path, value):
    path = Path(path)
    from gguf import check_no_symlinks
    check_no_symlinks(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.' + path.name + '.', suffix='.partial', dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write('\n')
            stream.flush()
            os.fsync(stream.fileno())
        publish_file(name, path)
    finally:
        Path(name).unlink(missing_ok=True)


def resources(path, *, min_ram_gib=0, min_disk_gib=0):
    """Check explicitly requested headroom; zero disables the corresponding check."""
    if not all(type(v) in (float, int) and math.isfinite(v) and v >= 0 for v in (min_ram_gib, min_disk_gib)):
        raise ValueError("resource headroom must be finite and nonnegative")
    if min_disk_gib and shutil.disk_usage(path).free < min_disk_gib * 2**30:
        raise ValueError('insufficient free disk for requested headroom')
    if min_ram_gib:
        rows = Path('/proc/meminfo').read_text().splitlines()
        available = next(int(row.split()[1]) * 1024 for row in rows if row.startswith('MemAvailable:'))
        if available < min_ram_gib * 2**30:
            raise ValueError('insufficient available RAM for requested headroom')


def configure(args):
    if getattr(args, 'cpus', None):
        cpus = {int(value) for value in args.cpus.split(',')}
        if not cpus or not cpus <= os.sched_getaffinity(0):
            raise ValueError('--cpus must select CPUs in the process affinity')
        os.sched_setaffinity(0, cpus)
    if getattr(args, 'nice', 0):
        os.nice(args.nice)


def add_arguments(parser):
    parser.add_argument('--cpus', help='comma-separated permitted logical CPU IDs')
    parser.add_argument('--nice', type=int, choices=range(0, 20), default=0)
    parser.add_argument('--min-ram-gib', type=float, default=0)
    parser.add_argument('--min-disk-gib', type=float, default=0)
