"""aml.data.schemas: the prepared tables pass; corrupted frames are rejected."""

from __future__ import annotations

import pandera.errors as pae
import polars as pl
import pytest

from aml.data import schemas

REJECT = (pae.SchemaError, pae.SchemaErrors, ValueError)


@pytest.fixture(scope="module")
def tables(prepared) -> dict[str, pl.DataFrame]:
    return {
        "tx": pl.read_parquet(prepared.transactions),
        "acc": pl.read_parquet(prepared.accounts),
        "lab": pl.read_parquet(prepared.labels),
    }


def test_prepared_tables_validate(tables) -> None:
    schemas.validate_transactions(tables["tx"])
    schemas.validate_accounts(tables["acc"])
    schemas.validate_labels(tables["lab"])


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda d: d.with_columns(pl.lit(-1.0).alias("amount_usd")), id="neg-amount"),
        pytest.param(lambda d: d.with_columns(pl.lit("val").alias("split")), id="bad-split"),
        pytest.param(
            lambda d: d.with_columns(pl.lit(1, pl.Int8).alias("is_laundering")), id="label"
        ),
        pytest.param(lambda d: d.with_columns(pl.col("row_id") // 2), id="dup-row-id"),
        pytest.param(lambda d: d.with_columns(pl.col("rank").reverse()), id="rank-order"),
        pytest.param(lambda d: d.with_columns(pl.col("src").cast(pl.Int64)), id="src-dtype"),
        pytest.param(
            lambda d: d.with_columns(
                pl.when(pl.col("rank") == 3).then(None).otherwise(pl.col("dst")).alias("dst")
            ),
            id="null-dst",
        ),
        pytest.param(lambda d: d.with_columns(pl.lit(0, pl.Int16).alias("day")), id="day-zero"),
        pytest.param(
            lambda d: d.with_columns(pl.col("from_bank").cast(pl.Categorical)), id="bank-cat"
        ),
        pytest.param(lambda d: d.drop("amount_usd"), id="missing-col"),
    ],
)
def test_transactions_rejected(tables, corrupt) -> None:
    with pytest.raises(REJECT):
        schemas.validate_transactions(corrupt(tables["tx"]))


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param(lambda d: d.with_columns(pl.lit(2, pl.Int8).alias("is_laundering")), id="y=2"),
        pytest.param(lambda d: d.with_columns(pl.lit("SMURF").alias("typology")), id="typology"),
        pytest.param(
            lambda d: d.with_columns(
                pl.when(pl.col("typology") == "OTHER")
                .then(0)
                .otherwise(pl.col("attempt_id"))
                .cast(pl.Int32)
                .alias("attempt_id")
            ),
            id="other-with-attempt",
        ),
        pytest.param(
            lambda d: d.with_columns(pl.lit(None, pl.String).alias("typology")),
            id="positive-without-typology",
        ),
        pytest.param(lambda d: pl.concat([d, d.head(1)]), id="dup-row-id"),
    ],
)
def test_labels_rejected(tables, corrupt) -> None:
    with pytest.raises(REJECT):
        schemas.validate_labels(corrupt(tables["lab"]))


def test_accounts_rejected(tables) -> None:
    acc = tables["acc"]
    dup_key = acc.with_columns(pl.lit("001").alias("bank"), pl.lit("X").alias("account"))
    with pytest.raises(REJECT):
        schemas.validate_accounts(dup_key)
    with pytest.raises(REJECT):
        schemas.validate_accounts(acc.with_columns(pl.col("account_id") + 1))
