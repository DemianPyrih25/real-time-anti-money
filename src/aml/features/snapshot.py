"""Engine snapshots: exact, pickle-free state files for restart (M2 spec §7).

Format::

    b"AMLSNAP1" | u32 little-endian header length | header JSON (utf-8, sorted keys) | payload
    payload = zlib.compress(concatenated component bytes, level=1)   (or raw when compress=False)

header = {format: 1, engine_version, spec_hash, clock, next_rank, next_offset, cursors: {W: rank},
n_accounts, n_pairs, live_start, hub_cap, hubs_sha256, byteorder: "little", compression,
components: [{name, typecode, n_items, offset, nbytes}], raw_nbytes, payload_sha256, state_digest,
extra: {...}}. Components, in Engine.STATE_FIELDS order: `ring.<col>` (rows [live_start, end_rank)
only), `Slot.name` for every registry slot, `acct.<name>` (spec.ACCOUNT_COLUMNS), `hub`,
`pairs.keys`, `pairs.pcnt.<W>`, `pairs.<col>` (spec.PAIR_COLUMNS).

Component bytes are little-endian whatever the host (byte-swapped copies on a big-endian host).
This module knows nothing about the engine: `Engine.snapshot` / `Engine.restore` assemble and check
the semantic header; `digest` is the canonical state digest both of them use.

Components are passed as arrays, bytearrays or memoryviews of them. Every memoryview this module
creates is released before it returns: a live export would make the engine's next
`array.append` raise BufferError.
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
import sys
import zlib
from array import array
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from aml.features.spec import SnapshotError

MAGIC = b"AMLSNAP1"
SNAPSHOT_FORMAT = 1
SIDECAR_SUFFIX = ".json"
ZLIB_LEVEL = 1
TYPECODES = ("b", "B", "i", "q", "d")  # the array typecodes a component may have
_PREFIX = len(MAGIC) + 4  # magic + u32 header length
_BIG_ENDIAN = sys.byteorder != "little"
# Header fields the state digest covers (with the component bytes): never the encoding's offsets
# or hashes, never ring.base (compaction timing is invisible).
DIGEST_FIELDS = (
    "clock",
    "next_rank",
    "cursors",
    "n_accounts",
    "n_pairs",
    "live_start",
    "hub_cap",
    "hubs_sha256",
    "engine_version",
    "spec_hash",
)


def _canonical_json(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


class _Views:
    """Creates flat little-endian byte views of components and releases them all on exit."""

    def __init__(self) -> None:
        self._views: list[memoryview] = []

    def __enter__(self) -> _Views:
        return self

    def __exit__(self, *exc: object) -> None:
        for mv in reversed(self._views):
            mv.release()
        self._views.clear()

    def bytes_of(self, buf: Any) -> tuple[memoryview, str]:
        """(byte view, typecode) of an array / bytearray / 1-D memoryview of one."""
        mv = memoryview(buf)
        self._views.append(mv)
        tc = mv.format
        if tc not in TYPECODES or mv.ndim != 1 or not mv.c_contiguous:
            raise ValueError(f"unsupported snapshot component (format {tc!r}, ndim {mv.ndim})")
        if _BIG_ENDIAN and mv.itemsize > 1:
            swapped = array(tc, mv.tobytes())
            swapped.byteswap()
            mv = memoryview(swapped)
            self._views.append(mv)
        b = mv.cast("B")
        self._views.append(b)
        return b, tc


def digest(semantic: dict[str, Any], components: Iterable[tuple[str, Any]]) -> str:
    """sha256 hex over the canonical JSON of `semantic[DIGEST_FIELDS]` and, in order, every
    component's name, typecode, length and little-endian bytes."""
    h = hashlib.sha256()
    h.update(_canonical_json({k: semantic[k] for k in DIGEST_FIELDS}))
    with _Views() as views:
        for name, buf in components:
            b, tc = views.bytes_of(buf)
            h.update(f"\n{name}:{tc}:{b.nbytes}\n".encode())
            h.update(b)
    return h.hexdigest()


def encode(header: dict[str, Any], components: list[tuple[str, Any]], *, compress: bool) -> bytes:
    """Snapshot bytes from the semantic header and (name, array/bytearray/memoryview) components.

    Adds `format`, `byteorder`, `compression`, `components`, `raw_nbytes` and `payload_sha256` to
    a copy of `header`; `peek_header` reads the result back.
    """
    table = []
    co = zlib.compressobj(ZLIB_LEVEL) if compress else None
    h = hashlib.sha256()  # of the payload as stored
    offset = 0
    with _Views() as views:
        chunks: list[Any] = []
        for name, buf in components:
            b, tc = views.bytes_of(buf)
            n = b.nbytes
            table.append(
                {
                    "name": name,
                    "typecode": tc,
                    "n_items": n // array(tc).itemsize,
                    "offset": offset,
                    "nbytes": n,
                }
            )
            offset += n
            piece = co.compress(b) if co is not None else b
            h.update(piece)
            chunks.append(piece)
        if co is not None:
            piece = co.flush()
            h.update(piece)
            chunks.append(piece)
        head = dict(header)
        head.update(
            format=SNAPSHOT_FORMAT,
            byteorder="little",
            compression="zlib" if compress else "none",
            components=table,
            raw_nbytes=offset,
            payload_sha256=h.hexdigest(),
        )
        text = json.dumps(head, sort_keys=True, allow_nan=False).encode("utf-8")
        out = b"".join((MAGIC, struct.pack("<I", len(text)), text, *chunks))
        chunks.clear()
    return out


def _split(data: bytes | bytearray | memoryview) -> tuple[dict[str, Any], memoryview]:
    """(header, payload view) of snapshot bytes; checks only the magic and the header."""
    mv = memoryview(data)
    if mv.nbytes < _PREFIX or bytes(mv[: len(MAGIC)]) != MAGIC:
        raise SnapshotError("not an engine snapshot (bad magic)")
    (hlen,) = struct.unpack("<I", mv[len(MAGIC) : _PREFIX])
    if mv.nbytes < _PREFIX + hlen:
        raise SnapshotError("truncated snapshot header")
    try:
        header = json.loads(bytes(mv[_PREFIX : _PREFIX + hlen]).decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as e:
        raise SnapshotError(f"unreadable snapshot header: {e}") from None
    if not isinstance(header, dict):
        raise SnapshotError("snapshot header is not a JSON object")
    return header, mv[_PREFIX + hlen :]


def peek_header(src: Path | str | bytes) -> dict[str, Any]:
    """The header of a snapshot file or bytes, without reading or checking the payload."""
    if isinstance(src, bytes | bytearray | memoryview):
        return _split(src)[0]
    with open(src, "rb") as f:
        prefix = f.read(_PREFIX)
        if len(prefix) < _PREFIX or prefix[: len(MAGIC)] != MAGIC:
            raise SnapshotError("not an engine snapshot (bad magic)")
        (hlen,) = struct.unpack("<I", prefix[len(MAGIC) :])
        return _split(prefix + f.read(hlen))[0]


def decode(src: Path | str | bytes) -> tuple[dict[str, Any], dict[str, array]]:
    """(header, {component name: array}) after checking magic, format, byteorder and the payload
    sha256 (before decompressing), then the component table. Raises `spec.SnapshotError`."""
    data = src if isinstance(src, bytes | bytearray | memoryview) else Path(src).read_bytes()
    header, payload = _split(data)
    if header.get("format") != SNAPSHOT_FORMAT:
        raise SnapshotError(f"snapshot format {header.get('format')!r} != {SNAPSHOT_FORMAT}")
    if header.get("byteorder") != "little":
        raise SnapshotError(f"snapshot byteorder {header.get('byteorder')!r} != 'little'")
    if hashlib.sha256(payload).hexdigest() != header.get("payload_sha256"):
        raise SnapshotError("snapshot payload sha256 mismatch (corrupt or truncated file)")
    compression = header.get("compression")
    if compression == "zlib":
        try:
            raw = memoryview(zlib.decompress(payload))
        except zlib.error as e:
            raise SnapshotError(f"snapshot payload does not decompress: {e}") from None
    elif compression == "none":
        raw = payload
    else:
        raise SnapshotError(f"unknown snapshot compression {compression!r}")
    if raw.nbytes != header.get("raw_nbytes"):
        raise SnapshotError(
            f"snapshot payload has {raw.nbytes} bytes, the header says {header.get('raw_nbytes')}"
        )
    comps: dict[str, array] = {}
    offset = 0
    entries = header.get("components")
    if not isinstance(entries, list):
        raise SnapshotError("snapshot header has no component table")
    for c in entries:
        try:
            name, tc, n, off, nb = c["name"], c["typecode"], c["n_items"], c["offset"], c["nbytes"]
        except (KeyError, TypeError):
            raise SnapshotError(f"bad component entry {c!r}") from None
        if tc not in TYPECODES or name in comps or off != offset or nb != n * array(tc).itemsize:
            raise SnapshotError(f"bad component entry {c!r}")
        a = array(tc)
        a.frombytes(raw[off : off + nb])
        if _BIG_ENDIAN and a.itemsize > 1:
            a.byteswap()
        comps[name] = a
        offset += nb
    if offset != raw.nbytes:
        raise SnapshotError("snapshot components do not cover the payload")
    return header, comps


def _write_durable(path: Path, data: bytes) -> None:
    """Temp file in the same directory, fsync, os.replace (no temp file is left behind)."""
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


def write_atomic(data: bytes, dst: Path, header: dict[str, Any]) -> dict[str, Any]:
    """Temp file in the same directory, fsync, os.replace; then the `.json` sidecar (header + file
    sha256 and size). Returns the sidecar document."""
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    _write_durable(dst, data)
    sidecar = dict(header)
    sidecar.update(
        file=dst.name, file_sha256=hashlib.sha256(data).hexdigest(), file_nbytes=len(data)
    )
    text = json.dumps(sidecar, indent=2, sort_keys=True, allow_nan=False) + "\n"
    _write_durable(dst.with_name(dst.name + SIDECAR_SUFFIX), text.encode("utf-8"))
    if hasattr(os, "O_DIRECTORY"):  # POSIX: make the renames durable too
        # Best effort: some filesystems (network/FUSE mounts such as a Modal Volume) reject a
        # directory fsync. The files themselves are already fsynced, and jobs vol.commit() after.
        try:
            fd = os.open(dst.parent, os.O_RDONLY | os.O_DIRECTORY)
        except OSError:
            return sidecar
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)
    return sidecar
