"""Scalar TX helpers (M2 spec §4.1) and the engine's TX group against M1 (§5.1)."""

from __future__ import annotations

import math
import sys
from decimal import ROUND_HALF_UP, Decimal

import duckdb
import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from aml.features.spec import tol_ok
from aml.features.tx_features import (
    TX_FEATURES,
    build_tx_features,
    cents,
    fit_vocab,
    hour_of_day,
    is_round_cents,
)
from tests.unit.test_engine_core import fixture_rows, fixture_spec

# The spec's edge values: half-cents in decimal that are (or are not) halves as doubles, the
# structuring band edges, tiny and huge amounts.
SPECIAL = [
    0.0,
    0.004,
    0.005,
    0.015,
    0.125,
    1.005,
    2.5,
    99.995,
    9999.995,
    8999.99,
    9000.0,
    10000.0,
    0.49999999999999994,
    123456789.125,
    1e12,
    2.0**52 / 100,
]

_CON: duckdb.DuckDBPyConnection | None = None


def _duckdb(xs: list[float]) -> list[int]:
    global _CON
    if _CON is None:
        _CON = duckdb.connect()
    frame = pl.DataFrame({"i": range(len(xs)), "x": xs}, schema={"i": pl.Int64, "x": pl.Float64})
    _CON.register("t", frame)
    rows = _CON.execute("SELECT CAST(round(x * 100) AS BIGINT) FROM t ORDER BY i").fetchall()
    return [r[0] for r in rows]


def _polars(xs: list[float]) -> list[int]:
    frame = pl.DataFrame({"x": xs}, schema={"x": pl.Float64})
    expr = (pl.col("x") * 100).round(0, mode="half_away_from_zero").cast(pl.Int64)
    return frame.select(expr)["x"].to_list()


def _decimal(x: float) -> int:
    # Decimal(y) is the exact value of the double y = x * 100.0
    return int(Decimal(x * 100.0).quantize(Decimal(1), rounding=ROUND_HALF_UP))


def _agree(xs: list[float]) -> None:
    got = [cents(x) for x in xs]
    assert got == [_decimal(x) for x in xs]
    assert got == _polars(xs)
    assert got == _duckdb(xs)


def test_cents_special_values():
    _agree(SPECIAL)
    assert [cents(x) for x in (0.0, 0.004, 0.005, 0.015, 0.125, 2.5, 8999.99)] == [
        0,
        0,
        1,
        2,
        13,
        250,
        899999,
    ]
    assert cents(1.005) == 100  # 1.005 * 100.0 = 100.49999999999999
    assert type(cents(3.0)) is int


_amounts = st.one_of(
    st.floats(min_value=0.0, max_value=1e12, allow_nan=False, allow_infinity=False),
    st.integers(0, 10**12).map(lambda k: k / 1000),  # many near half-cent boundaries
    st.integers(0, 10**11).map(lambda k: k / 200),  # decimal half-cents x.xx5
)


@settings(max_examples=100, deadline=None, derandomize=True)
@given(st.lists(_amounts, min_size=100, max_size=100))
def test_cents_equals_decimal_polars_and_duckdb(xs):
    _agree(xs)  # 100 examples x 100 values = 10^4 amounts


@pytest.mark.parametrize("bad", [-0.01, -1e-300, math.nan, math.inf, -math.inf])
def test_cents_refuses_negative_and_non_finite(bad):
    with pytest.raises(ValueError):
        cents(bad)


def test_round_and_hour_helpers():
    assert is_round_cents(10000, 10000) and is_round_cents(420000, 10000)
    assert not is_round_cents(0, 10000) and not is_round_cents(10001, 10000)
    assert [hour_of_day(m) for m in (0, 59, 60, 1439, 1440, 3 * 1440 + 125)] == [0, 0, 1, 23, 0, 2]


def test_engine_tx_group_equals_m1_build_tx_features(prepared, rules_cfg):
    spec, tx = fixture_spec(prepared, rules_cfg)
    rows = fixture_rows(spec, tx)
    vocab = fit_vocab(tx.filter(pl.col("split") == "train"))
    m1 = build_tx_features(tx, vocab, rules_cfg["round_unit"])
    assert m1["row_id"].to_list() == tx["row_id"].to_list()
    idx = spec.model_index(TX_FEATURES)
    assert idx == tuple(range(len(TX_FEATURES)))
    got = np.array([[r[i] for i in idx] for r in rows], dtype=np.float64)
    for j, name in enumerate(TX_FEATURES):
        want = m1[name].cast(pl.Float64).to_numpy()
        if name == "log_amount_usd":  # libm log1p: bit-equal on Linux, <= 1 ulp elsewhere
            ok = tol_ok("ulp", got[:, j], want, independent=sys.platform != "linux")
        else:
            ok = got[:, j] == want
        assert ok.all(), f"{name}: {int((~ok).sum())} rows differ"
    assert got[:, TX_FEATURES.index("round_amount")].sum() > 0  # the comparison is not vacuous
    assert (got[:, TX_FEATURES.index("payment_format")] >= 0).all()
