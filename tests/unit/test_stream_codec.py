"""The M5 wire codec: bit-exact round trips, strict decoding, latency headers."""

from __future__ import annotations

import json
import struct
import sys

import polars as pl
import pytest

from aml.features.spec import INPUT_COLUMNS
from aml.serving import bundle
from aml.streaming.codec import (
    KEYS,
    CodecError,
    Meta,
    decode_event,
    encode_alert,
    encode_event,
    event_key,
    pack_meta,
    unpack_meta,
)

EVENT = (11, 7, 11520, 3, 4, 1234.56, 1234.56, "ACH", "US Dollar", "Euro", "001", "012")


def _bits(x: float) -> bytes:
    return struct.pack("<d", x)


def _same(a: tuple, b: tuple) -> bool:
    if len(a) != len(b):
        return False
    for x, y in zip(a, b, strict=True):
        if type(x) is not type(y):
            return False
        if (_bits(x) != _bits(y)) if isinstance(y, float) else x != y:
            return False
    return True


def _raw(pairs: list[tuple[str, object]]) -> bytes:
    """A JSON object with exactly these pairs, in this order (duplicates allowed)."""
    return ("{" + ",".join(f"{json.dumps(k)}:{json.dumps(v)}" for k, v in pairs) + "}").encode()


def _pairs(fields: tuple = EVENT) -> list[tuple[str, object]]:
    return [("v", 1), *zip(INPUT_COLUMNS, fields, strict=True)]


def test_every_slice_row_round_trips_bit_exact(serving_bundle):
    sl = pl.read_parquet(serving_bundle / bundle.SLICE, columns=list(INPUT_COLUMNS))
    assert sl.height > 0
    for f in sl.iter_rows():
        value = encode_event(f)
        assert _same(decode_event(value), f), f
        assert value.startswith(b'{"v":1,"rank":') and b" " not in value[:20]


@pytest.mark.parametrize(
    "x", [0.1 + 0.2, 5e-324, 2.5e-310, -0.0, 0.0, sys.float_info.max, 1e22, 123456789.01]
)
def test_edge_floats_round_trip(x):
    f = (*EVENT[:5], x, x, *EVENT[7:])
    got = decode_event(encode_event(f))
    assert _bits(got[5]) == _bits(x) and _bits(got[6]) == _bits(x)


def test_a_fifteen_digit_encoder_would_lose_bits():
    """Negative control: the round-trip tests above can see a lossy float format."""
    x = 0.1 + 0.2
    assert _bits(float(f"{x:.15g}")) != _bits(x)
    assert _bits(json.loads(json.dumps(x))) == _bits(x)


def test_ints_strings_and_keys():
    f = (2**31 - 2, 0, 0, 0, 0, 1.0, 2.0, "Bitcoin", "Yuan", "Yuan", "é", "☃")
    assert _same(decode_event(encode_event(f)), f)
    assert list(json.loads(encode_event(f))) == list(KEYS)
    assert event_key(123456) == b"123456"
    with pytest.raises(CodecError):
        event_key(True)


@pytest.mark.parametrize(
    "case",
    [
        "missing",
        "extra",
        "reordered",
        "duplicate",
        "v2",
        "v_bool",
        "bool_for_int",
        "int_for_amount",
        "float_for_id",
        "int_for_category",
        "nan_literal",
        "array",
        "null",
        "string",
        "bad_utf8",
        "truncated",
    ],
)
def test_strict_decode_rejects(case):
    pairs = _pairs()
    if case == "missing":
        raw = _raw(pairs[:-1])
    elif case == "extra":
        raw = _raw([*pairs, ("note", "x")])
    elif case == "reordered":
        raw = _raw([pairs[0], pairs[2], pairs[1], *pairs[3:]])
    elif case == "duplicate":
        raw = _raw([pairs[0], *pairs])
    elif case == "v2":
        raw = _raw([("v", 2), *pairs[1:]])
    elif case == "v_bool":
        raw = _raw([("v", True), *pairs[1:]])
    elif case == "bool_for_int":
        raw = _raw([pairs[0], ("rank", True), *pairs[2:]])
    elif case == "int_for_amount":
        raw = _raw([*pairs[:6], ("amount_usd", 5), *pairs[7:]])
    elif case == "float_for_id":
        raw = _raw([*pairs[:4], ("src", 3.0), *pairs[5:]])
    elif case == "int_for_category":
        raw = _raw([*pairs[:8], ("payment_format", 3), *pairs[9:]])
    elif case == "nan_literal":
        raw = _raw(pairs).replace(b"1234.56,", b"NaN,", 1)
    elif case == "array":
        raw = json.dumps([list(p) for p in pairs]).encode()
    elif case == "null":
        raw = b"null"
    elif case == "string":
        raw = b'"x"'
    elif case == "bad_utf8":
        raw = _raw(pairs).replace(b'"ACH"', b'"\xff\xfe"')
    else:
        raw = _raw(pairs)[:-1]
    assert decode_event(_raw(pairs)) == EVENT  # the unbroken message decodes
    with pytest.raises(CodecError):
        decode_event(raw)


@pytest.mark.parametrize(
    "fields",
    [
        (*EVENT[:5], float("nan"), *EVENT[6:]),
        (*EVENT[:5], float("inf"), *EVENT[6:]),
        (*EVENT[:6], -float("inf"), *EVENT[7:]),
        (True, *EVENT[1:]),
        (*EVENT[:5], "1.0", *EVENT[6:]),
        (*EVENT[:7], 3, *EVENT[8:]),
        (12.5, *EVENT[1:]),
        EVENT[:-1],
    ],
)
def test_encode_refuses(fields):
    with pytest.raises(CodecError):
        encode_event(fields)


def test_headers_round_trip_and_absent():
    for meta in (Meta(123, 456, 3), Meta(2**63 - 1, -(2**63), -32768), Meta(0, 0, 32767)):
        headers = pack_meta(meta)
        assert unpack_meta(headers) == meta
        assert unpack_meta([("trace", b"x"), *headers]) == meta  # other headers are ignored
    for absent in (None, [], [("trace", b"x")]):
        assert unpack_meta(absent) is None


@pytest.mark.parametrize("case", ["short", "partial", "repeated", "null_value", "long_point"])
def test_malformed_headers_raise(case):
    h = pack_meta(Meta(1, 2, 3))
    if case == "short":
        h[0] = ("t_sched", b"\x00" * 7)
    elif case == "partial":
        h = h[:2]
    elif case == "repeated":
        h = [*h, h[0]]
    elif case == "null_value":
        h[1] = ("t_prod", None)
    else:
        h[2] = ("point", b"\x00" * 8)
    with pytest.raises(CodecError):
        unpack_meta(h)


def test_alert_value():
    rec = {"row_id": 5, "score": 0.1 + 0.2, "rules_fired": ["round_trip"], "severities": [0.0]}
    doc = json.loads(encode_alert(rec))
    assert doc == {"v": 1, **rec} and _bits(doc["score"]) == _bits(rec["score"])
    with pytest.raises(ValueError):
        encode_alert({**rec, "score": float("nan")})
