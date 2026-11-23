"""Pure-Python reader for TensorFlow TensorBundle v2 checkpoints (``variables.index`` + ``variables.data-*``).

Reads ``sp_v6.tgz`` and ``og_export.zip`` directly at load time with zero runtime TensorFlow dependency,
and caches the converted PyTorch state dict as ``.safetensors`` next to the archive.
"""

from __future__ import annotations

import os
import struct
import tarfile
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_TABLE_MAGIC = 0xDB4775248B80FB57
# TensorFlow DataType enum values -> (torch.dtype, element size in bytes)
_TF_DTYPES: dict[int, tuple[torch.dtype, int]] = {
    1: (torch.float32, 4),   # DT_FLOAT
    2: (torch.float64, 8),   # DT_DOUBLE
    3: (torch.int32, 4),     # DT_INT32
    4: (torch.uint8, 1),     # DT_UINT8
    5: (torch.int16, 2),     # DT_INT16
    6: (torch.int8, 1),      # DT_INT8
    9: (torch.int64, 8),     # DT_INT64
    10: (torch.bool, 1),     # DT_BOOL
    14: (torch.bfloat16, 2), # DT_BFLOAT16
    19: (torch.float16, 2),  # DT_HALF
}


def _read_varint(buf: bytes | memoryview, pos: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = buf[pos]
        pos += 1
        result |= (b & 0x7F) << shift
        if not (b & 0x80):
            return result, pos
        shift += 7


def _snappy_decompress(src: bytes | memoryview) -> bytes:
    """Decompress a raw Snappy block (LevelDB SSTable block compression type 1)."""
    n = len(src)
    uncompressed_len, pos = _read_varint(src, 0)
    out = bytearray()
    out_extend = out.extend
    while pos < n:
        tag = src[pos]
        pos += 1
        kind = tag & 0x03
        if kind == 0:  # literal
            length = tag >> 2
            if length < 60:
                length += 1
            else:
                extra = length - 59
                length = int.from_bytes(src[pos : pos + extra], "little") + 1
                pos += extra
            out_extend(src[pos : pos + length])
            pos += length
        else:
            if kind == 1:
                length = ((tag >> 2) & 0x07) + 4
                offset = ((tag & 0xE0) << 3) | src[pos]
                pos += 1
            elif kind == 2:
                length = (tag >> 2) + 1
                offset = int.from_bytes(src[pos : pos + 2], "little")
                pos += 2
            else:
                length = (tag >> 2) + 1
                offset = int.from_bytes(src[pos : pos + 4], "little")
                pos += 4
            if offset == 0 or offset > len(out):
                raise ValueError(f"Invalid Snappy copy offset {offset} at output size {len(out)}")
            start = len(out) - offset
            if length <= offset:
                out_extend(out[start : start + length])
            else:
                # Overlapping RLE copy
                for _ in range(length):
                    out.append(out[start])
                    start += 1
    if len(out) != uncompressed_len:
        raise ValueError(f"Snappy length mismatch: expected {uncompressed_len}, got {len(out)}")
    return bytes(out)


def _read_sstable_block(table: bytes, offset: int, size: int) -> bytes:
    comp_type = table[offset + size]
    raw = memoryview(table)[offset : offset + size]
    if comp_type == 0:
        return bytes(raw)
    if comp_type == 1:
        return _snappy_decompress(raw)
    raise ValueError(f"Unsupported LevelDB block compression type {comp_type}")


def _iter_sstable_block(block: bytes):
    """Yield ``(key_bytes, value_bytes)`` entries from a decompressed LevelDB data/index block."""
    num_restarts = struct.unpack_from("<I", block, len(block) - 4)[0]
    limit = len(block) - 4 - 4 * num_restarts
    pos = 0
    key = b""
    while pos < limit:
        shared, pos = _read_varint(block, pos)
        non_shared, pos = _read_varint(block, pos)
        val_len, pos = _read_varint(block, pos)
        key = key[:shared] + block[pos : pos + non_shared]
        pos += non_shared
        val = block[pos : pos + val_len]
        pos += val_len
        yield key, val


def _read_sstable(index_bytes: bytes) -> dict[str, bytes]:
    """Read all key-value pairs from a LevelDB SSTable (TensorFlow ``variables.index``)."""
    if len(index_bytes) < 48:
        raise ValueError("Invalid SSTable: shorter than 48-byte footer")
    magic = struct.unpack_from("<Q", index_bytes, len(index_bytes) - 8)[0]
    if magic != _TABLE_MAGIC:
        raise ValueError(f"Invalid SSTable magic {magic:#x}")
    footer_pos = len(index_bytes) - 48
    # Footer layout: metaindex_handle (2 varints), index_handle (2 varints)
    _, pos = _read_varint(index_bytes, footer_pos)
    _, pos = _read_varint(index_bytes, pos)
    idx_off, pos = _read_varint(index_bytes, pos)
    idx_size, _ = _read_varint(index_bytes, pos)

    index_block = _read_sstable_block(index_bytes, idx_off, idx_size)
    entries: dict[str, bytes] = {}
    for _, handle in _iter_sstable_block(index_block):
        off, p = _read_varint(handle, 0)
        sz, _ = _read_varint(handle, p)
        data_block = _read_sstable_block(index_bytes, off, sz)
        for k, v in _iter_sstable_block(data_block):
            if k:  # empty key holds BundleHeaderProto
                entries[k.decode("utf-8")] = v
    return entries


def _parse_bundle_entry(raw: bytes) -> tuple[int, tuple[int, ...], int, int, int]:
    """Decode ``tensorflow.BundleEntryProto`` -> ``(dtype, shape, shard_id, offset, size)``."""
    pos, n = 0, len(raw)
    dtype = shard_id = offset = size = 0
    shape: list[int] = []
    while pos < n:
        tag, pos = _read_varint(raw, pos)
        field, wire = tag >> 3, tag & 0x07
        if wire == 0:
            val, pos = _read_varint(raw, pos)
            if field == 1:
                dtype = val
            elif field == 3:
                shard_id = val
            elif field == 4:
                offset = val
            elif field == 5:
                size = val
        elif wire == 2:
            length, pos = _read_varint(raw, pos)
            sub = raw[pos : pos + length]
            pos += length
            if field == 2:  # TensorShapeProto { repeated Dim dim = 2; }
                sp = 0
                while sp < len(sub):
                    stag, sp = _read_varint(sub, sp)
                    sfield, swire = stag >> 3, stag & 0x07
                    if swire == 2:
                        slen, sp = _read_varint(sub, sp)
                        dbytes = sub[sp : sp + slen]
                        sp += slen
                        if sfield == 2:  # Dim { int64 size = 1; }
                            dp, dval = 0, 0
                            while dp < len(dbytes):
                                dtag, dp = _read_varint(dbytes, dp)
                                df, dw = dtag >> 3, dtag & 0x07
                                if dw == 0:
                                    v, dp = _read_varint(dbytes, dp)
                                    if df == 1:
                                        dval = v
                                elif dw == 2:
                                    dl, dp = _read_varint(dbytes, dp)
                                    dp += dl
                                elif dw == 5:
                                    dp += 4
                                elif dw == 1:
                                    dp += 8
                            shape.append(dval)
                    elif swire == 0:
                        _, sp = _read_varint(sub, sp)
                    elif swire == 5:
                        sp += 4
                    elif swire == 1:
                        sp += 8
        elif wire == 5:
            pos += 4
        elif wire == 1:
            pos += 8
        else:
            raise ValueError(f"Unsupported protobuf wire type {wire}")
    return dtype, tuple(shape), shard_id, offset, size


def read_bundle_bytes(index_bytes: bytes, data_bytes: bytes) -> dict[str, torch.Tensor]:
    """Read every numeric tensor from a single-shard TF TensorBundle v2 into CPU :class:`torch.Tensor`s."""
    entries = _read_sstable(index_bytes)
    tensors: dict[str, torch.Tensor] = {}
    for name, raw in entries.items():
        dtype_id, shape, shard_id, offset, size = _parse_bundle_entry(raw)
        if dtype_id not in _TF_DTYPES:
            continue  # skip string metadata entries such as _CHECKPOINTABLE_OBJECT_GRAPH
        if shard_id != 0:
            raise ValueError(f"Multi-shard TensorBundle not supported (tensor {name!r} in shard {shard_id})")
        torch_dtype, itemsize = _TF_DTYPES[dtype_id]
        expected = itemsize
        for d in shape:
            expected *= d
        if size != expected:
            raise ValueError(f"Tensor {name!r} byte size {size} != expected {expected} for {shape} {torch_dtype}")
        buf = bytearray(data_bytes[offset : offset + size])
        tensors[name] = torch.frombuffer(buf, dtype=torch_dtype).reshape(shape).clone()
    return tensors


def read_bundle_from_archive(archive_path: str | Path) -> dict[str, torch.Tensor]:
    """Read a single-shard TF SavedModel bundle from a ``.zip``, ``.tgz``/``.tar.gz``, or directory."""
    path = Path(archive_path)
    if path.is_dir():
        idx = next(path.rglob("variables.index"))
        data = idx.with_name("variables.data-00000-of-00001")
        return read_bundle_bytes(idx.read_bytes(), data.read_bytes())
    suffixes = "".join(path.suffixes).lower()
    if suffixes.endswith(".zip"):
        with zipfile.ZipFile(path) as zf:
            names = zf.namelist()
            idx_name = next(n for n in names if n.endswith("variables.index"))
            data_name = next(n for n in names if n.endswith("variables.data-00000-of-00001"))
            return read_bundle_bytes(zf.read(idx_name), zf.read(data_name))
    if suffixes.endswith((".tgz", ".tar.gz", ".tar")):
        with tarfile.open(path, "r:*") as tf_arc:
            members = tf_arc.getmembers()
            idx_m = next(m for m in members if m.name.endswith("variables.index"))
            data_m = next(m for m in members if m.name.endswith("variables.data-00000-of-00001"))
            idx_f = tf_arc.extractfile(idx_m)
            data_f = tf_arc.extractfile(data_m)
            assert idx_f is not None and data_f is not None
            return read_bundle_bytes(idx_f.read(), data_f.read())
    raise ValueError(f"Unsupported TensorBundle container: {path}")


def load_cached_safetensors(
    archive_path: Path,
    cache_name: str,
    convert: Callable[[dict[str, torch.Tensor]], dict[str, torch.Tensor]],
    *,
    map_location="cpu",
) -> dict[str, torch.Tensor]:
    """Load converted tensors from a ``.safetensors`` cache next to ``archive_path`` (creating it on first use)."""
    device = str(map_location)
    if archive_path.suffix == ".safetensors":
        return load_file(str(archive_path), device=device)
    cache_path = archive_path.with_name(cache_name)
    if cache_path.is_file():
        return load_file(str(cache_path), device=device)
    raw = read_bundle_from_archive(archive_path)
    state = {k: v.contiguous() for k, v in convert(raw).items()}
    # Atomic write into the checkpoint directory (ignore permission errors on read-only dirs).
    try:
        fd, tmp = tempfile.mkstemp(dir=cache_path.parent, prefix=f".{cache_name}.", suffix=".tmp")
        os.close(fd)
        try:
            save_file(state, tmp)
            os.replace(tmp, cache_path)
        except Exception:
            if os.path.exists(tmp):
                os.unlink(tmp)
            raise
    except OSError:
        pass
    return {k: v.to(map_location) for k, v in state.items()}
