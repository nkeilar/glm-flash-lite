"""Small, bounded GGUF reader and streaming subset writer for E1 tools.

Layout includes the four-byte per-row IQ4_KS scale used by the pinned
ik_llama.cpp quantizer (it is not an IQ4_XS block stream).
No third-party Python packages, remote reads, or model-loading side effects.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct
from typing import Callable

# name, block elements, block bytes, extra row bytes: ik_llama 5bf8f0fe.
TYPES = {
    0: ('F32', 1, 4, 0), 1: ('F16', 1, 2, 0),
    2: ('Q4_0', 32, 18, 0), 3: ('Q4_1', 32, 20, 0),
    6: ('Q5_0', 32, 22, 0), 7: ('Q5_1', 32, 24, 0),
    8: ('Q8_0', 32, 34, 0), 9: ('Q8_1', 32, 36, 0),
    10: ('Q2_K', 256, 84, 0), 11: ('Q3_K', 256, 110, 0),
    12: ('Q4_K', 256, 144, 0), 13: ('Q5_K', 256, 176, 0),
    14: ('Q6_K', 256, 210, 0), 15: ('Q8_K', 256, 292, 0),
    16: ('IQ2_XXS', 256, 66, 0), 17: ('IQ2_XS', 256, 74, 0),
    18: ('IQ3_XXS', 256, 98, 0), 19: ('IQ1_S', 256, 50, 0),
    20: ('IQ4_NL', 32, 18, 0), 21: ('IQ3_S', 256, 110, 0),
    22: ('IQ2_S', 256, 82, 0), 23: ('IQ4_XS', 256, 136, 0),
    24: ('I8', 1, 1, 0), 25: ('I16', 1, 2, 0),
    26: ('I32', 1, 4, 0), 27: ('I64', 1, 8, 0),
    28: ('F64', 1, 8, 0), 29: ('IQ1_M', 256, 56, 0),
    30: ('BF16', 1, 2, 0), 138: ('IQ3_K', 256, 110, 0),
    144: ('IQ4_KS', 256, 136, 4),
}
SCALARS = {0: 'B', 1: 'b', 2: 'H', 3: 'h', 4: 'I', 5: 'i',
           6: 'f', 7: '?', 10: 'Q', 11: 'q', 12: 'd'}
HEADER_LIMIT = 64 * 2**20


def check_no_symlinks(path: Path | str):
    path = Path(path).absolute()
    for part in (path, *path.parents):
        if part.is_symlink():
            raise ValueError(f'refuse symlink in owned output path: {part}')


def sha256_file(path: Path | str, tick: Callable[[], None] = lambda: None) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk := f.read(4 * 2**20):
            tick()
            digest.update(chunk)
    return digest.hexdigest()


def tensor_bytes(dims: list[int], type_id: int) -> int:
    if type_id not in TYPES:
        raise ValueError(f'unreviewed GGML type id {type_id}')
    _, block, size, extra = TYPES[type_id]
    if not dims or any(d <= 0 for d in dims) or dims[0] % block:
        raise ValueError(f'invalid GGML dimensions {dims} for type {type_id}')
    return (dims[0] // block * size + extra) * math.prod(dims[1:])


def pack_string(value: str) -> bytes:
    data = value.encode('utf-8')
    return struct.pack('<Q', len(data)) + data


def pack_value(typ: int, value) -> bytes:
    if typ in SCALARS:
        return struct.pack('<' + SCALARS[typ], value)
    if typ == 8:
        return pack_string(value)
    if typ == 9:
        item_type, values = value
        return struct.pack('<IQ', item_type, len(values)) + b''.join(
            pack_value(item_type, v) for v in values)
    raise ValueError(f'unknown metadata type {typ}')


@dataclass
class GGUF:
    path: Path
    metadata: dict
    tensors: list[dict]
    data_start: int
    # Encoded values preserve tokenizer arrays and unknown metadata losslessly.
    records: list[tuple[str, int, bytes]]

    def tensor(self, name: str) -> dict:
        matches = [t for t in self.tensors if t['name'] == name]
        if len(matches) != 1:
            raise ValueError(f'expected exactly one tensor {name} in {self.path}')
        return matches[0]

    def read_tensor(self, name: str) -> bytes:
        t = self.tensor(name)
        with self.path.open('rb') as f:
            f.seek(self.data_start + t['offset'])
            data = f.read(t['nbytes'])
        if len(data) != t['nbytes']:
            raise ValueError(f'short tensor data for {name}')
        return data


def read_gguf(path: Path | str) -> GGUF:
    path = Path(path)
    size = path.stat().st_size
    with path.open('rb') as f:
        def take(n):
            if n < 0 or f.tell() + n > HEADER_LIMIT:
                raise ValueError('GGUF header exceeds 64 MiB')
            data = f.read(n)
            if len(data) != n:
                raise ValueError('truncated GGUF header')
            return data

        def scalar(fmt):
            return struct.unpack('<' + fmt, take(struct.calcsize(fmt)))[0]

        def string():
            return take(scalar('Q')).decode('utf-8')

        def value(typ):
            if typ in SCALARS:
                return scalar(SCALARS[typ])
            if typ == 8:
                return string()
            if typ == 9:
                item, count = scalar('I'), scalar('Q')
                if item == 9 or count > 10_000_000:
                    raise ValueError('nested or oversized metadata array')
                return [value(item) for _ in range(count)]
            raise ValueError(f'unknown GGUF metadata type {typ}')

        if take(4) != b'GGUF' or scalar('I') not in (2, 3):
            raise ValueError('expected little-endian GGUF v2/v3')
        nt, nk = scalar('Q'), scalar('Q')
        if nt > 1_000_000 or nk > 100_000:
            raise ValueError('implausible GGUF counts')
        metadata, records = {}, []
        for _ in range(nk):
            key, typ = string(), scalar('I')
            if key in metadata:
                raise ValueError(f'duplicate metadata key {key}')
            start = f.tell()
            metadata[key] = value(typ)
            end = f.tell()
            f.seek(start)
            records.append((key, typ, take(end - start)))
        tensors, names = [], set()
        for _ in range(nt):
            name, nd = string(), scalar('I')
            if not 1 <= nd <= 4 or name in names:
                raise ValueError(f'invalid or duplicate tensor {name}')
            names.add(name)
            dims = [scalar('Q') for _ in range(nd)]
            tid, offset = scalar('I'), scalar('Q')
            nbytes = tensor_bytes(dims, tid)
            tensors.append(dict(name=name, dims=dims, type_id=tid,
                                type=TYPES[tid][0], nbytes=nbytes, offset=offset))
        align = metadata.get('general.alignment', 32)
        if not isinstance(align, int) or align <= 0 or align > 65536 or align & (align - 1):
            raise ValueError('invalid GGUF alignment')
        data_start = (f.tell() + align - 1) // align * align
        end = 0
        for t in sorted(tensors, key=lambda t: t['offset']):
            if t['offset'] % align or t['offset'] < end:
                raise ValueError('overlapping or misaligned tensor ranges')
            end = t['offset'] + t['nbytes']
            if data_start + end > size:
                raise ValueError(f'tensor {t["name"]} exceeds file: partial GGUF')
    return GGUF(path, metadata, tensors, data_start, records)


def write_subset(sources: list[tuple[GGUF, str]], output: Path | str,
                 metadata_source: GGUF | None = None,
                 tick: Callable[[], None] = lambda: None) -> dict:
    """Exclusive-create a standalone GGUF; caller owns fsync/atomic publication.

    Sources may span shards, while metadata comes from the verified first shard.
    split.* keys MUST be removed or llama's loader opens unselected source shards.
    """
    if not sources:
        raise ValueError('empty GGUF selection')
    metadata_source = metadata_source or sources[0][0]
    records = [r for r in metadata_source.records if not r[0].startswith('split.')]
    align = metadata_source.metadata.get('general.alignment', 32)
    selected, offset = [], 0
    for source, name in sources:
        t = dict(source.tensor(name))
        t['offset'] = offset
        offset = (offset + t['nbytes'] + align - 1) // align * align
        selected.append(t)
    if len({t['name'] for t in selected}) != len(selected):
        raise ValueError('duplicate selected tensor')
    header = b'GGUF' + struct.pack('<IQQ', 3, len(selected), len(records))
    for key, typ, encoded in records:
        header += pack_string(key) + struct.pack('<I', typ) + encoded
    for t in selected:
        header += pack_string(t['name']) + struct.pack('<I', len(t['dims']))
        header += struct.pack('<' + 'Q' * len(t['dims']), *t['dims'])
        header += struct.pack('<IQ', t['type_id'], t['offset'])
    header += b'\0' * (-len(header) % align)
    output = Path(output)
    check_no_symlinks(output)
    with output.open('xb') as dest:
        dest.write(header)
        for (source, name), t in zip(sources, selected):
            dest.write(b'\0' * (len(header) + t['offset'] - dest.tell()))
            src_t = source.tensor(name)
            with source.path.open('rb') as src:
                src.seek(source.data_start + src_t['offset'])
                remaining = t['nbytes']
                while remaining:
                    tick()
                    data = src.read(min(remaining, 4 * 2**20))
                    if not data:
                        raise ValueError(f'source truncated while copying {name}')
                    dest.write(data)
                    remaining -= len(data)
        dest.flush()
        import os
        os.fsync(dest.fileno())
    return dict(path=str(output), sha256=sha256_file(output, tick),
                bytes=output.stat().st_size)


def extract_gguf(source: GGUF, tensor_names: list[str], output: Path | str) -> dict:
    return write_subset([(source, name) for name in tensor_names], output)


def write_gguf(path: Path | str, metadata: dict[str, tuple[int, object]],
               tensors: list[dict], tick: Callable[[], None] = lambda: None) -> dict:
    """Exclusive-create a GGUF from explicit typed metadata and payloads.

    metadata values are (GGUF value type, value); array values are themselves
    (item type, list). Each tensor has name,dims,type_id,data. data may be bytes
    or a Path to an exact-size payload file, copied in bounded chunks.
    This is also the tiny-fixture writer shared by quantizer/checkpoint tests.
    """
    align = metadata.get('general.alignment', (4, 32))[1]
    if not isinstance(align, int) or align <= 0 or align > 65536 or align & (align - 1):
        raise ValueError('invalid alignment')
    if len({t['name'] for t in tensors}) != len(tensors):
        raise ValueError('duplicate tensor name')
    records = [(k, typ, pack_value(typ, value)) for k, (typ, value) in metadata.items()]
    selected, offset = [], 0
    for tensor in tensors:
        t = dict(tensor, nbytes=tensor_bytes(tensor['dims'], tensor['type_id']), offset=offset)
        data = t['data']
        actual_size = data.stat().st_size if isinstance(data, Path) else len(data)
        if actual_size != t['nbytes']:
            raise ValueError(f'payload size mismatch for {t["name"]}')
        offset = (offset + t['nbytes'] + align - 1) // align * align
        selected.append(t)
    header = b'GGUF' + struct.pack('<IQQ', 3, len(selected), len(records))
    for key, typ, encoded in records:
        header += pack_string(key) + struct.pack('<I', typ) + encoded
    for t in selected:
        header += pack_string(t['name']) + struct.pack('<I', len(t['dims']))
        header += struct.pack('<' + 'Q' * len(t['dims']), *t['dims'])
        header += struct.pack('<IQ', t['type_id'], t['offset'])
    header += b'\0' * (-len(header) % align)
    path = Path(path)
    check_no_symlinks(path)
    with path.open('xb') as f:
        f.write(header)
        for t in selected:
            f.write(b'\0' * (len(header) + t['offset'] - f.tell()))
            if isinstance(t['data'], Path):
                with t['data'].open('rb') as src:
                    while chunk := src.read(4 * 2**20):
                        tick()
                        f.write(chunk)
            else:
                tick()
                f.write(t['data'])
        f.flush()
        import os
        os.fsync(f.fileno())
    return dict(path=str(path), sha256=sha256_file(path, tick), bytes=path.stat().st_size)
