"""Wire format of the `transactions` and `alerts` topics (M5).

Core module: stdlib + `aml.features.spec` only.

A transaction is one JSON object `{"v": 1, <spec.INPUT_COLUMNS in order>}` with compact
separators. `json` writes floats with `float.__repr__` (the shortest string that round-trips) and
parses them correctly rounded, so every finite float64 amount survives bit for bit. Decoding is
strict: the exact key order, `v == 1`, ints that are ints (never bools), amounts that are floats,
categoricals that are strings; anything else raises `CodecError`.

Latency headers (set by the replayer, optional for the scorer): `t_sched` and `t_prod` are
`struct.pack(">q", ns)` of `time.monotonic_ns()` (containers on one Docker host share the
kernel's monotonic clock), `point` is `struct.pack(">h", index)`.
"""

from __future__ import annotations

import json
import math
import operator
import struct
import time
from collections.abc import Mapping, Sequence
from typing import Any, NamedTuple

from aml.features.spec import INPUT_COLUMNS

VERSION = 1
KEYS = ("v", *INPUT_COLUMNS)
INT_COLUMNS = ("rank", "row_id", "minute", "src", "dst")
FLOAT_COLUMNS = ("amount_usd", "amount_paid")
STR_COLUMNS = ("payment_format", "payment_currency", "receiving_currency", "from_bank", "to_bank")
_KIND = {c: int for c in INT_COLUMNS} | {c: float for c in FLOAT_COLUMNS}
_KIND |= {c: str for c in STR_COLUMNS}
_KINDS = tuple(_KIND[c] for c in INPUT_COLUMNS)  # KeyError at import if spec gains a column

H_T_SCHED = "t_sched"
H_T_PROD = "t_prod"
H_POINT = "point"
_I64 = struct.Struct(">q")
_I16 = struct.Struct(">h")

now_ns = time.monotonic_ns


class CodecError(ValueError):
    """A message that is not exactly this wire format (the scorer stops: exit 3)."""


class Meta(NamedTuple):
    """The replayer's latency headers of one message."""

    t_sched_ns: int  # monotonic time the event was scheduled for (open loop)
    t_prod_ns: int  # monotonic time taken just before produce()
    point: int  # plan point index (0 = warm-up)


# --- transactions ---------------------------------------------------------------------------------


def event_key(row_id: int) -> bytes:
    """The message key of a transaction or an alert: the decimal row_id."""
    return str(_as_int(row_id, "row_id")).encode("ascii")


def _as_int(x: Any, name: str) -> int:
    if isinstance(x, bool):
        raise CodecError(f"{name}: a bool is not an integer")
    try:
        return operator.index(x)  # numpy bools have no __index__ either
    except TypeError:
        raise CodecError(f"{name}: expected an integer, got {type(x).__name__}") from None


def encode_event(fields: Sequence[Any]) -> bytes:
    """The message value of one event (`fields` in spec.INPUT_COLUMNS order)."""
    if len(fields) != len(INPUT_COLUMNS):
        raise CodecError(f"expected {len(INPUT_COLUMNS)} fields, got {len(fields)}")
    obj: dict[str, Any] = {"v": VERSION}
    for name, kind, x in zip(INPUT_COLUMNS, _KINDS, fields, strict=True):
        if kind is int:
            obj[name] = _as_int(x, name)
        elif kind is float:
            try:
                if isinstance(x, bool | str):
                    raise TypeError
                v = float(x)
            except (TypeError, ValueError):
                raise CodecError(f"{name}: expected a number, got {type(x).__name__}") from None
            if not math.isfinite(v):
                raise CodecError(f"{name}: {v!r} is not finite")
            obj[name] = v
        else:
            if not isinstance(x, str):
                raise CodecError(f"{name}: expected a string, got {type(x).__name__}")
            obj[name] = x
    return json.dumps(obj, separators=(",", ":"), allow_nan=False).encode("ascii")


class _Pairs(list):
    """json.loads object hook result: the (key, value) pairs in document order."""


def _no_constant(name: str) -> Any:
    raise CodecError(f"{name} is not allowed")


def decode_event(value: bytes | bytearray | memoryview | str) -> tuple:
    """The event fields in spec.INPUT_COLUMNS order; CodecError unless exactly the wire format."""
    try:
        obj = json.loads(
            bytes(value) if not isinstance(value, str) else value,
            object_pairs_hook=_Pairs,
            parse_constant=_no_constant,
        )
    except CodecError:
        raise
    except (ValueError, TypeError) as e:  # JSONDecodeError and UnicodeDecodeError are ValueErrors
        raise CodecError(f"not a JSON event: {e}") from None
    if type(obj) is not _Pairs:
        raise CodecError(f"not a JSON object: {type(obj).__name__}")
    keys = tuple(k for k, _ in obj)
    if keys != KEYS:
        raise CodecError(f"keys {list(keys)} != {list(KEYS)}")
    v = obj[0][1]
    if type(v) is not int or v != VERSION:
        raise CodecError(f"v: {v!r} != {VERSION}")
    out = []
    for (name, x), kind in zip(obj[1:], _KINDS, strict=True):
        if type(x) is not kind:
            raise CodecError(f"{name}: expected {kind.__name__}, got {type(x).__name__}")
        out.append(x)
    return tuple(out)


# --- latency headers ------------------------------------------------------------------------------


def pack_meta(meta: Meta) -> list[tuple[str, bytes]]:
    """Kafka headers carrying `meta`."""
    return [
        (H_T_SCHED, _I64.pack(meta.t_sched_ns)),
        (H_T_PROD, _I64.pack(meta.t_prod_ns)),
        (H_POINT, _I16.pack(meta.point)),
    ]


def unpack_meta(headers: Sequence[tuple[str, Any]] | None) -> Meta | None:
    """The latency headers of a message: None when none is present; CodecError when they are
    partial, repeated or of the wrong size. Other headers are ignored."""
    if not headers:
        return None
    got: dict[str, Any] = {}
    for key, value in headers:
        if key in (H_T_SCHED, H_T_PROD, H_POINT):
            if key in got:
                raise CodecError(f"header {key!r} repeated")
            got[key] = value
    if not got:
        return None
    if len(got) != 3:
        missing = sorted({H_T_SCHED, H_T_PROD, H_POINT} - set(got))
        raise CodecError(f"latency headers incomplete: {missing} missing")
    try:
        return Meta(
            _I64.unpack(got[H_T_SCHED])[0],
            _I64.unpack(got[H_T_PROD])[0],
            _I16.unpack(got[H_POINT])[0],
        )
    except (struct.error, TypeError) as e:
        raise CodecError(f"malformed latency header: {e}") from None


# --- alerts ---------------------------------------------------------------------------------------


def encode_alert(record: Mapping[str, Any]) -> bytes:
    """The `alerts` message value: `{"v": 1, **record}` as compact JSON (keyed by event_key)."""
    doc = {"v": VERSION, **record}
    return json.dumps(doc, separators=(",", ":"), allow_nan=False).encode("ascii")
