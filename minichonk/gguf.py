"""Minimal GGUF v3 reader and header writer.

Key/value pairs keep their original encoded bytes so that metadata we do not
change is written back byte for byte. Tensor data is never decoded here.
"""
import dataclasses
import struct
from typing import Callable

MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32

KV_UINT8, KV_INT8, KV_UINT16, KV_INT16, KV_UINT32, KV_INT32, KV_FLOAT32, KV_BOOL = range(8)
KV_STRING, KV_ARRAY, KV_UINT64, KV_INT64, KV_FLOAT64 = range(8, 13)

_KV_SCALARS = {KV_UINT8: "B", KV_INT8: "b", KV_UINT16: "H", KV_INT16: "h", KV_UINT32: "I", KV_INT32: "i",
               KV_FLOAT32: "f", KV_BOOL: "?", KV_UINT64: "Q", KV_INT64: "q", KV_FLOAT64: "d"}

# ggml type id -> (elements per block, bytes per block), from ggml's type_traits.
GGML_TYPE_LAYOUT = {
    0: (1, 4), 1: (1, 2), 2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24), 8: (32, 34), 9: (32, 36),
    10: (256, 80), 11: (256, 112), 12: (256, 144), 13: (256, 176), 14: (256, 210), 15: (256, 292),
    16: (256, 66), 17: (256, 74), 18: (256, 98), 19: (256, 50), 20: (32, 18), 21: (256, 110), 22: (256, 82),
    23: (256, 136), 24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8), 29: (256, 56), 30: (1, 2),
}


class GGUFError(ValueError):
    pass


@dataclasses.dataclass
class KV:
    key: str
    kv_type: int
    raw: bytes  # encoded value, without key and type
    value: object  # decoded scalar or string, None for arrays


@dataclasses.dataclass
class TensorInfo:
    name: str
    dims: list  # ggml order: dims[0] varies fastest, the last dim is the slowest
    ggml_type: int
    offset: int  # relative to the start of the data section

    def row_bytes(self):
        block, size = layout_of(self.ggml_type)
        if self.dims[0] % block:
            raise GGUFError(f"{self.name}: first dim {self.dims[0]} is not a multiple of block size {block}")
        return self.dims[0] // block * size

    def nbytes(self):
        n = self.row_bytes()
        for d in self.dims[1:]:
            n *= d
        return n

    def slab_bytes(self):
        """Size of one index along the last dim, which ggml stores contiguously."""
        return self.nbytes() // self.dims[-1]


@dataclasses.dataclass
class Header:
    version: int
    kvs: list
    tensors: list
    alignment: int
    data_start: int

    def kv(self, key):
        return next((k for k in self.kvs if k.key == key), None)


def layout_of(ggml_type):
    try:
        return GGML_TYPE_LAYOUT[ggml_type]
    except KeyError:
        raise GGUFError(f"unsupported ggml type {ggml_type}") from None


def align_up(n, alignment):
    return (n + alignment - 1) // alignment * alignment


class _Cursor:
    """Sequential reader over a read_at(offset, size) callable with read-ahead."""

    def __init__(self, read_at: Callable[[int, int], bytes], read_ahead=16 * 1024 * 1024):
        self.read_at, self.read_ahead = read_at, read_ahead
        self.buf, self.pos = b"", 0

    def take(self, n):
        while self.pos + n > len(self.buf):
            more = self.read_at(len(self.buf), max(n, self.read_ahead))
            if not more:
                raise GGUFError("truncated GGUF header")
            self.buf += more
        out = self.buf[self.pos:self.pos + n]
        self.pos += n
        return out

    def scalar(self, fmt):
        return struct.unpack("<" + fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self):
        return self.take(self.scalar("Q")).decode("utf-8")


def _read_value(c, kv_type):
    if kv_type in _KV_SCALARS:
        return c.scalar(_KV_SCALARS[kv_type])
    if kv_type == KV_STRING:
        return c.string()
    if kv_type == KV_ARRAY:
        elem, n = c.scalar("I"), c.scalar("Q")
        if elem in _KV_SCALARS:
            c.take(struct.calcsize(_KV_SCALARS[elem]) * n)
        else:
            for _ in range(n):
                _read_value(c, elem)
        return None
    raise GGUFError(f"unknown kv type {kv_type}")


def parse_header(read_at) -> Header:
    c = _Cursor(read_at)
    if c.take(4) != MAGIC:
        raise GGUFError("not a GGUF file")
    version = c.scalar("I")
    if version != 3:
        raise GGUFError(f"unsupported GGUF version {version}")
    n_tensors, n_kv = c.scalar("Q"), c.scalar("Q")
    kvs = []
    for _ in range(n_kv):
        key = c.string()
        kv_type = c.scalar("I")
        start = c.pos
        value = _read_value(c, kv_type)
        kvs.append(KV(key, kv_type, c.buf[start:c.pos], value))
    tensors = []
    for _ in range(n_tensors):
        name = c.string()
        dims = [c.scalar("Q") for _ in range(c.scalar("I"))]
        ggml_type = c.scalar("I")
        tensors.append(TensorInfo(name, dims, ggml_type, c.scalar("Q")))
    align_kv = next((k for k in kvs if k.key == "general.alignment"), None)
    alignment = align_kv.value if align_kv else DEFAULT_ALIGNMENT
    return Header(version, kvs, tensors, alignment, align_up(c.pos, alignment))


def parse_file(path) -> Header:
    with open(path, "rb") as f:
        def read_at(offset, size):
            f.seek(offset)
            return f.read(size)
        return parse_header(read_at)


def check_layout(header: Header, file_size=None):
    """Prove our byte sizes match the file: tensors must sit exactly where we compute them."""
    expected = 0
    for t in header.tensors:
        if t.offset != expected:
            raise GGUFError(f"{t.name}: offset {t.offset}, computed {expected}")
        expected = align_up(t.offset + t.nbytes(), header.alignment)
    if file_size is not None and header.tensors:
        last = header.tensors[-1]
        end = header.data_start + last.offset + last.nbytes()
        if file_size not in (end, header.data_start + expected):
            raise GGUFError(f"file is {file_size} bytes, tensors end at {end}")


def encode_header(header: Header) -> bytes:
    """Encode a header and its padding. Tensor offsets must already be final."""
    out = bytearray(MAGIC + struct.pack("<IQQ", header.version, len(header.tensors), len(header.kvs)))
    for kv in header.kvs:
        key = kv.key.encode("utf-8")
        out += struct.pack("<Q", len(key)) + key + struct.pack("<I", kv.kv_type) + kv.raw
    for t in header.tensors:
        name = t.name.encode("utf-8")
        out += struct.pack("<Q", len(name)) + name + struct.pack("<I", len(t.dims))
        out += struct.pack(f"<{len(t.dims)}Q", *t.dims)
        out += struct.pack("<IQ", t.ggml_type, t.offset)
    out += b"\0" * (align_up(len(out), header.alignment) - len(out))
    return bytes(out)


def encode_scalar(kv_type, value) -> bytes:
    return struct.pack("<" + _KV_SCALARS[kv_type], value)


def encode_string(value) -> bytes:
    data = value.encode("utf-8")
    return struct.pack("<Q", len(data)) + data
