"""Per-transaction features on hand-made rows (spec §3.8)."""

from __future__ import annotations

import math

import duckdb
import polars as pl
import pytest

from aml.features.tx_features import (
    CATEGORICAL_FEATURES,
    INPUT_COLUMNS,
    TX_FEATURES,
    build_tx_features,
    fit_vocab,
    round_amount_expr,
    round_amount_sql,
)

ROWS = [
    # row_id, minute, src, dst, from_bank, to_bank, amount_paid, amount_usd, pay, recv, format
    (10, 0, 1, 2, "010", "020", 1000.0000001, 1000.0, "US Dollar", "US Dollar", "ACH"),
    (11, 59, 3, 3, "012", "012", 99.999999, 100.0, "Euro", "Euro", "Reinvestment"),
    (12, 60, 4, 5, "012", "12", 99.99, 117.6, "Euro", "US Dollar", "Wire"),
    (13, 1439, 6, 7, "001", "001", 150.0, 150.0, "US Dollar", "Yen", "Cash"),
    (14, 1440, 8, 9, "001", "0119", 0.0, 0.0, "Yen", "Yen", "Bitcoin"),
    (15, 3 * 1440 + 125, 8, 1, "0119", "010", 200.004, 200.004, "Rupee", "Rupee", "Cheque"),
]


def _frame(rows: list[tuple] = ROWS) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "row_id": pl.Int64,
            "minute": pl.Int64,
            "src": pl.Int32,
            "dst": pl.Int32,
            "from_bank": pl.String,
            "to_bank": pl.String,
            "amount_paid": pl.Float64,
            "amount_usd": pl.Float64,
            "payment_currency": pl.String,
            "receiving_currency": pl.String,
            "payment_format": pl.String,
        },
        orient="row",
    )


def test_columns_match_spec() -> None:
    assert set(CATEGORICAL_FEATURES) <= set(TX_FEATURES)
    assert len(set(TX_FEATURES)) == len(TX_FEATURES) == 9
    assert list(_frame().columns) == INPUT_COLUMNS


def test_round_amount_handles_float_artefacts() -> None:
    amounts = [1000.0000001, 99.999999, 99.99, 150.0, 0.0, 0.001, 200.004, 200.006, 10000.0]
    got = pl.DataFrame({"amount_paid": amounts}).select(r=round_amount_expr(100))["r"].to_list()
    assert got == [True, True, False, False, False, False, True, False, True]
    unit_one = pl.DataFrame({"amount_paid": [5.0, 5.5, 4.9999999, 0.004]})
    assert unit_one.select(r=round_amount_expr(1))["r"].to_list() == [True, False, True, False]


@pytest.mark.parametrize("unit", [0, -100, 0.001, float("nan"), "100", True])
def test_round_amount_rejects_bad_units(unit: object) -> None:
    with pytest.raises(ValueError):
        round_amount_expr(unit)  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        round_amount_sql(unit)  # type: ignore[arg-type]


def test_round_amount_sql_matches_polars() -> None:
    # Includes exact half-cents (0.125 -> 12.5 cents): both sides round half away from zero.
    amounts = [
        1000.0000001,
        99.999999,
        99.99,
        150.0,
        0.0,
        0.125,
        100.125,
        200.004,
        200.006,
        1e7,
        0.5,
        12.345,
        4999.995,
        5000.005,
        99.995,
        100.005,
    ]
    df = pl.DataFrame({"amount_paid": amounts})
    for unit in (100, 1, 0.25, 10000):
        want = df.select(r=round_amount_expr(unit))["r"].to_list()
        got = duckdb.sql(f"SELECT {round_amount_sql(unit)} AS r FROM df").pl()["r"].to_list()
        assert got == want, unit


def test_fit_vocab_sorted_from_given_rows() -> None:
    vocab = fit_vocab(_frame())
    assert set(vocab) == set(CATEGORICAL_FEATURES)
    assert vocab["payment_currency"] == ["Euro", "Rupee", "US Dollar", "Yen"]
    assert vocab["receiving_currency"] == ["Euro", "Rupee", "US Dollar", "Yen"]
    assert vocab["payment_format"] == sorted(r[-1] for r in ROWS)


def test_feature_values_on_hand_made_rows() -> None:
    vocab = {
        "payment_currency": ["Euro", "US Dollar"],  # Yen and Rupee unknown -> -1
        "receiving_currency": ["Euro", "US Dollar", "Yen"],
        "payment_format": ["ACH", "Cash", "Wire"],
    }
    out = build_tx_features(_frame(), vocab, round_unit=100)
    assert out.columns == ["row_id", *TX_FEATURES]
    assert out.schema["log_amount_usd"] == pl.Float64
    for c in CATEGORICAL_FEATURES:
        assert out.schema[c] == pl.Int32
    for c in ("cross_currency", "self_loop", "same_bank", "round_amount", "hour_of_day"):
        assert out.schema[c] == pl.Int8

    rows = {r["row_id"]: r for r in out.iter_rows(named=True)}
    assert out["row_id"].to_list() == [r[0] for r in ROWS]  # input order kept

    for r in ROWS:
        assert math.isclose(rows[r[0]]["log_amount_usd"], math.log1p(r[7]), rel_tol=1e-12)

    assert [rows[i]["payment_currency"] for i in range(10, 16)] == [1, 0, 0, 1, -1, -1]
    assert [rows[i]["receiving_currency"] for i in range(10, 16)] == [1, 0, 1, 2, 2, -1]
    # Reinvestment, Bitcoin and Cheque are not in the vocab.
    assert [rows[i]["payment_format"] for i in range(10, 16)] == [0, -1, 2, 1, -1, -1]

    assert [rows[i]["cross_currency"] for i in range(10, 16)] == [0, 0, 1, 1, 0, 0]
    assert [rows[i]["self_loop"] for i in range(10, 16)] == [0, 1, 0, 0, 0, 0]
    # "012" vs "12" are different banks; a self-loop is also same-bank.
    assert [rows[i]["same_bank"] for i in range(10, 16)] == [0, 1, 0, 1, 0, 0]
    assert [rows[i]["round_amount"] for i in range(10, 16)] == [1, 1, 0, 0, 0, 1]
    assert [rows[i]["hour_of_day"] for i in range(10, 16)] == [0, 0, 1, 23, 0, 2]


def test_null_category_is_missing() -> None:
    df = _frame().with_columns(
        pl.when(pl.col("row_id") == 10)
        .then(None)
        .otherwise(pl.col("payment_format"))
        .alias("payment_format")
    )
    out = build_tx_features(df, fit_vocab(_frame()), round_unit=100)
    assert out.filter(pl.col("row_id") == 10)["payment_format"].item() == -1


def test_missing_input_column_or_bad_vocab_raises() -> None:
    vocab = fit_vocab(_frame())
    with pytest.raises(ValueError, match="lacks columns"):
        build_tx_features(_frame().drop("minute"), vocab, round_unit=100)
    with pytest.raises(ValueError, match="vocab keys"):
        build_tx_features(_frame(), {"payment_format": ["ACH"]}, round_unit=100)


def test_features_on_prepared_fixture(prepared) -> None:
    tx = pl.read_parquet(prepared.transactions)
    vocab = fit_vocab(tx.filter(pl.col("split") == "train"))
    out = build_tx_features(tx, vocab, round_unit=100)
    assert out.height == tx.height
    assert out["row_id"].to_list() == tx["row_id"].to_list()
    assert out.null_count().sum_horizontal().item() == 0
    # Reinvestment is always a self-loop in the fixture (as in the real data).
    reinv = vocab["payment_format"].index("Reinvestment")
    assert out.filter(pl.col("payment_format") == reinv)["self_loop"].min() == 1
    # Every train category is known; codes are within the vocab.
    tr = out.filter(tx["split"] == "train")
    for c in CATEGORICAL_FEATURES:
        assert tr[c].min() >= 0 and tr[c].max() < len(vocab[c])
    assert out["hour_of_day"].is_between(0, 23).all()
