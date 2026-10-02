"""pandera (polars backend) schemas for the canonical tables (M1 spec §2).

`strict=True` everywhere: an unexpected column fails validation. For transactions this is what
keeps any label column out of the model-input table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandera.polars as pa
import polars as pl
from pandera.api.polars.types import PolarsData

from aml.data.patterns import LABEL_TYPOLOGIES
from aml.data.split import SPLITS

if TYPE_CHECKING:
    from aml.features.spec import EngineSpec

_NON_NEG = pa.Check.ge(0)


def _str(**kw) -> pa.Column:
    return pa.Column(pl.String, nullable=False, **kw)


TRANSACTIONS = pa.DataFrameSchema(
    {
        "row_id": pa.Column(pl.Int64, _NON_NEG, nullable=False, unique=True),
        "rank": pa.Column(pl.Int64, _NON_NEG, nullable=False, unique=True),
        "ts": pa.Column(pl.Datetime("us"), nullable=False),
        "minute": pa.Column(pl.Int64, _NON_NEG, nullable=False),
        "day": pa.Column(pl.Int16, pa.Check.ge(1), nullable=False),
        "split": _str(checks=pa.Check.isin(list(SPLITS))),
        "from_bank": _str(),
        "from_account": _str(),
        "to_bank": _str(),
        "to_account": _str(),
        "src": pa.Column(pl.Int32, _NON_NEG, nullable=False),
        "dst": pa.Column(pl.Int32, _NON_NEG, nullable=False),
        "amount_received": pa.Column(pl.Float64, _NON_NEG, nullable=False),
        "receiving_currency": _str(),
        "amount_paid": pa.Column(pl.Float64, _NON_NEG, nullable=False),
        "payment_currency": _str(),
        "payment_format": _str(),
        "amount_usd": pa.Column(pl.Float64, _NON_NEG, nullable=False),
    },
    strict=True,
    ordered=True,
    name="transactions",
)

ACCOUNTS = pa.DataFrameSchema(
    {
        "account_id": pa.Column(pl.Int32, _NON_NEG, nullable=False, unique=True),
        "bank": _str(),
        "account": _str(),
    },
    strict=True,
    ordered=True,
    unique=["bank", "account"],
    name="accounts",
)

LABELS = pa.DataFrameSchema(
    {
        "row_id": pa.Column(pl.Int64, _NON_NEG, nullable=False, unique=True),
        "is_laundering": pa.Column(pl.Int8, pa.Check.isin([0, 1]), nullable=False),
        "attempt_id": pa.Column(pl.Int32, _NON_NEG, nullable=True),
        "typology": pa.Column(pl.String, pa.Check.isin(list(LABEL_TYPOLOGIES)), nullable=True),
        "attempt_size": pa.Column(pl.Int32, pa.Check.ge(1), nullable=True),
        "typology_detail": pa.Column(pl.String, nullable=True),
    },
    strict=True,
    ordered=True,
    name="labels",
)


def validate_transactions(df: pl.DataFrame) -> pl.DataFrame:
    """Schema checks plus: rows ordered by rank, rank = 0..N-1, day consistent with minute."""
    TRANSACTIONS.validate(df, lazy=True)
    n = df.height
    bad = df.select(
        (pl.col("rank") != pl.int_range(n, dtype=pl.Int64)).sum().alias("rank"),
        (pl.col("day") != (pl.col("minute") // 1440 + 1)).sum().alias("day"),
        (pl.col("ts").diff() < pl.duration(microseconds=0)).sum().alias("ts_order"),
    ).row(0, named=True)
    problems = {k: v for k, v in bad.items() if v}
    if problems:
        raise ValueError(f"transactions: inconsistent rows {problems}")
    return df


def validate_accounts(df: pl.DataFrame) -> pl.DataFrame:
    """Schema checks plus: account_id = 0..A-1."""
    ACCOUNTS.validate(df, lazy=True)
    bad = df.select((pl.col("account_id") != pl.int_range(df.height, dtype=pl.Int32)).sum()).item()
    if bad:
        raise ValueError(f"accounts: {bad} account_id values are not 0..A-1 in order")
    return df


def validate_labels(df: pl.DataFrame) -> pl.DataFrame:
    """Schema checks plus: typology is set iff positive; pattern columns set together."""
    LABELS.validate(df, lazy=True)
    bad = df.select(
        (pl.col("typology").is_null() != (pl.col("is_laundering") == 0)).sum().alias("typology"),
        (pl.col("attempt_id").is_null() != pl.col("attempt_size").is_null())
        .sum()
        .alias("attempt_size"),
        (pl.col("attempt_id").is_null() != pl.col("typology").is_in(["OTHER"]).fill_null(True))
        .sum()
        .alias("attempt_id"),
    ).row(0, named=True)
    problems = {k: v for k, v in bad.items() if v}
    if problems:
        raise ValueError(f"labels: inconsistent rows {problems}")
    return df


# --- the M2 feature table (M2 spec §5.10) -------------------------------------------------------


def _domain_ok(column: str, domain: str) -> pl.Expr:
    """True where a value is valid for its FeatureDef.domain (also false on null).

    Polars orders NaN above every number and NaN == NaN is true, so NaN is excluded explicitly
    (is_finite) wherever it is not allowed.
    """
    c = pl.col(column)
    if domain == "flag":
        ok = (c == 0) | (c == 1)
    elif domain == "code":
        ok = c.is_finite() & (c >= -1) & (c == c.floor())
    elif domain == "int":
        ok = c.is_finite() & (c >= 0) & (c == c.floor())
    elif domain == "real":
        ok = c.is_finite() & (c >= 0)
    elif domain == "nullable":
        ok = ~c.is_infinite()  # NaN = undefined (empty window), never +-inf
    else:
        raise ValueError(f"unknown feature domain {domain!r}")
    return c.is_not_null() & ok


def feature_value_checks(spec: EngineSpec) -> dict[str, pl.Expr]:
    """Column -> row-wise validity of a feature-table part (features by domain, tail columns)."""
    from aml.features.spec import INFLOW_COLUMN, SEVERITY_COLUMNS, TRUNC_COLUMNS

    checks = {f.name: _domain_ok(f.name, f.domain) for f in spec.features}
    for s in SEVERITY_COLUMNS:
        checks[s] = _domain_ok(s, "real")  # finite and >= 0
    checks[INFLOW_COLUMN] = pl.col(INFLOW_COLUMN).is_not_null() & (pl.col(INFLOW_COLUMN) >= 0)
    for t in TRUNC_COLUMNS:
        checks[t] = _domain_ok(t, "flag")
    return checks


_FEATURE_SCHEMAS: dict[str, pa.DataFrameSchema] = {}


def features_schema(spec: EngineSpec) -> pa.DataFrameSchema:
    """The pandera schema of a feature-table part for `spec` (cached by spec_hash).

    Column dtypes and order (`spec.table_schema()`), unique non-negative row_id and rank, day >= 1,
    split in SPLITS; one dataframe-level check runs every value check of `feature_value_checks`
    in a single pass (one check per column would cost ~6x more on the 5M-row replay). Feature
    columns are `nullable=True` only because pandera counts NaN as null: nulls are still refused
    by the value checks, NaN only where the domain allows it.
    """
    key = spec.spec_hash()
    schema = _FEATURE_SCHEMAS.get(key)
    if schema is not None:
        return schema
    from aml.features.spec import TABLE_KEY_COLUMNS

    checks = feature_value_checks(spec)
    columns: dict[str, pa.Column] = {
        "row_id": pa.Column(pl.Int64, _NON_NEG, nullable=False, unique=True),
        "rank": pa.Column(pl.Int64, _NON_NEG, nullable=False, unique=True),
        "day": pa.Column(pl.Int16, pa.Check.ge(1), nullable=False),
        "split": _str(checks=pa.Check.isin(list(SPLITS))),
    }
    for name, dtype in spec.table_schema().items():
        if name not in TABLE_KEY_COLUMNS:
            columns[name] = pa.Column(dtype, nullable=True)

    def values_ok(data: PolarsData) -> pl.LazyFrame:
        return data.lazyframe.select(pl.all_horizontal(list(checks.values())).alias("ok"))

    schema = pa.DataFrameSchema(
        columns,
        checks=[pa.Check(values_ok, name="feature_value_domains")],
        strict=True,
        ordered=True,
        name="features",
    )
    _FEATURE_SCHEMAS[key] = schema
    return schema


def validate_features(df: pl.DataFrame, spec: EngineSpec) -> pl.DataFrame:
    """Validate one feature-table part (or row group) against `spec` (M2 spec §5.10).

    dtypes and column order; unique row_id and rank; counts >= 0, integral and never NaN; flags in
    {0, 1}; codes >= -1; log amounts / sums / ports finite and >= 0; means, stds, maxima and
    ratios finite or NaN; severities finite and >= 0; inflow_c >= 0; trunc flags in {0, 1}.
    Raises ValueError naming every column with invalid values (pandera SchemaErrors otherwise).
    """
    try:
        features_schema(spec).validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        checks = feature_value_checks(spec)
        present = {k: v for k, v in checks.items() if k in df.columns}
        if len(present) == len(checks) and df.schema == pl.Schema(spec.table_schema()):
            bad = df.select((~v).sum().alias(k) for k, v in present.items()).row(0, named=True)
            bad = {k: n for k, n in bad.items() if n}
            if bad:
                raise ValueError(f"features: invalid values per column {bad}") from e
        raise
    return df
