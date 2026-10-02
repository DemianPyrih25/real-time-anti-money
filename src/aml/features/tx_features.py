"""Per-transaction features for the M1 LightGBM baseline (PLAN.md §6 M1).

Every feature is a function of the transaction's own fields only, so the as-of rule (PLAN.md §4)
holds trivially: no other event is read. Preprocessing (the category vocabularies) is fitted on
train rows only.
"""

from __future__ import annotations

import math
import numbers
from collections.abc import Iterable

import polars as pl

TX_FEATURES = [
    "log_amount_usd",
    "payment_currency",
    "receiving_currency",
    "cross_currency",
    "payment_format",
    "self_loop",
    "same_bank",
    "round_amount",
    "hour_of_day",
]
CATEGORICAL_FEATURES = ["payment_currency", "receiving_currency", "payment_format"]

# Model inputs must never include these (PLAN.md §4 "Never features"); the whitelist test uses it.
FORBIDDEN_FEATURES = frozenset(
    {
        "row_id",
        "rank",
        "ts",
        "minute",
        "day",
        "split",
        "src",
        "dst",
        "from_bank",
        "from_account",
        "to_bank",
        "to_account",
        "is_laundering",
        "attempt_id",
        "typology",
        "attempt_size",
        "typology_detail",
    }
)
FORBIDDEN_SUBSTRINGS = ("time", "timestamp", "label")

# Transaction columns build_tx_features reads.
INPUT_COLUMNS = [
    "row_id",
    "minute",
    "src",
    "dst",
    "from_bank",
    "to_bank",
    "amount_paid",
    "amount_usd",
    "payment_currency",
    "receiving_currency",
    "payment_format",
]

MINUTES_PER_DAY = 1440


def _unit_cents(unit: float) -> int:
    """`unit` in cents; it must be a positive whole number of cents."""
    if isinstance(unit, bool) or not isinstance(unit, numbers.Real) or not math.isfinite(unit):
        raise ValueError(f"round unit must be a positive number, got {unit!r}")
    cents = float(unit) * 100
    m = round(cents)
    if m <= 0 or abs(cents - m) > 1e-9:
        raise ValueError(f"round unit must be a positive whole number of cents, got {unit!r}")
    return int(m)


def round_amount_expr(unit: float, column: str = "amount_paid") -> pl.Expr:
    """True when the paid amount is a whole multiple of `unit` (in the paid currency).

    Amounts are compared in whole cents so float artefacts (1000.0000001, 99.999999) count as
    round. Rounding is half away from zero, which is what DuckDB's round() does, so
    `round_amount_sql` gives identical flags.
    """
    m = _unit_cents(unit)
    cents = (pl.col(column) * 100).round(0, mode="half_away_from_zero").cast(pl.Int64)
    return (cents > 0) & (cents % m == 0)


def round_amount_sql(unit: float, column: str = "amount_paid") -> str:
    """DuckDB SQL boolean with the same definition as `round_amount_expr` (for the rules)."""
    m = _unit_cents(unit)
    if not column.isidentifier():
        raise ValueError(f"not a plain column name: {column!r}")
    cents = f"CAST(round({column} * 100) AS BIGINT)"
    return f"({cents} > 0 AND {cents} % {m} = 0)"


def cents(x: float) -> int:
    """Whole cents of a finite amount x >= 0, rounded half away from zero (M2 spec §4.1).

    Bit-identical to DuckDB `CAST(round(x * 100) AS BIGINT)` and to polars
    `(x * 100).round(0, "half_away_from_zero")`: all three round the same double y = x * 100.0,
    and y - floor(y) is exact for y < 2^52 (above that y is already a whole number).
    ValueError for a negative, NaN or infinite x.
    """
    if not 0.0 <= x < math.inf:  # NaN fails every comparison
        raise ValueError(f"amount must be finite and >= 0, got {x!r}")
    y = x * 100.0
    f = math.floor(y)  # an int
    return f + 1 if y - f >= 0.5 else f


def is_round_cents(paid_c: int, round_cents: int) -> bool:
    """The round-amount flag on whole cents: paid_c > 0 and a whole multiple of round_cents."""
    return paid_c > 0 and paid_c % round_cents == 0


def hour_of_day(minute: int) -> int:
    """(minute % 1440) // 60, the `hour_of_day` feature of a simulated minute."""
    return (minute % MINUTES_PER_DAY) // 60


def is_forbidden(name: str) -> bool:
    low = name.lower()
    return low in FORBIDDEN_FEATURES or any(s in low for s in FORBIDDEN_SUBSTRINGS)


def assert_whitelisted(feature_names: Iterable[str]) -> None:
    """Raise if any model input is outside TX_FEATURES or forbidden."""
    names = list(feature_names)
    bad = [n for n in names if n not in TX_FEATURES or is_forbidden(n)]
    if bad:
        raise ValueError(f"features not allowed as model inputs: {bad}")


def fit_vocab(train_df: pl.DataFrame) -> dict[str, list[str]]:
    """Sorted categories of each categorical feature, from the rows given (train only)."""
    return {
        c: sorted(train_df.get_column(c).drop_nulls().unique().to_list())
        for c in CATEGORICAL_FEATURES
    }


def _codes(column: str, categories: list[str]) -> pl.Expr:
    # Unknown (or null) -> -1; LightGBM treats negative categorical values as missing.
    return (
        pl.col(column)
        .replace_strict(categories, list(range(len(categories))), default=-1, return_dtype=pl.Int32)
        .fill_null(-1)
    )


def build_tx_features(
    tx: pl.DataFrame, vocab: dict[str, list[str]], round_unit: float
) -> pl.DataFrame:
    """`row_id` + TX_FEATURES, one row per input row, in input order."""
    missing = [c for c in INPUT_COLUMNS if c not in tx.columns]
    if missing:
        raise ValueError(f"transactions frame lacks columns: {missing}")
    if set(vocab) != set(CATEGORICAL_FEATURES):
        raise ValueError(f"vocab keys must be {CATEGORICAL_FEATURES}, got {sorted(vocab)}")
    flag = pl.Int8
    out = tx.select(
        pl.col("row_id"),
        pl.col("amount_usd").log1p().cast(pl.Float64).alias("log_amount_usd"),
        _codes("payment_currency", vocab["payment_currency"]).alias("payment_currency"),
        _codes("receiving_currency", vocab["receiving_currency"]).alias("receiving_currency"),
        (pl.col("payment_currency") != pl.col("receiving_currency"))
        .cast(flag)
        .alias("cross_currency"),
        _codes("payment_format", vocab["payment_format"]).alias("payment_format"),
        (pl.col("src") == pl.col("dst")).cast(flag).alias("self_loop"),
        # Bank codes are compared as strings: "012" and "12" are different banks.
        (pl.col("from_bank") == pl.col("to_bank")).cast(flag).alias("same_bank"),
        round_amount_expr(round_unit).cast(flag).alias("round_amount"),
        ((pl.col("minute") % MINUTES_PER_DAY) // 60).cast(flag).alias("hour_of_day"),
    )
    assert_whitelisted(out.columns[1:])
    return out
