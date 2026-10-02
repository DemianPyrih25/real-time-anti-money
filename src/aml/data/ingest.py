"""Raw IBM AML CSV -> typed, time-ordered transactions with account ids and single-currency amounts.

Traps handled here (PLAN.md §4): the header repeats `Account`; bank codes are zero-padded strings;
hex account ids such as `8003E4680` parse as floats, so every column is read as a String and only
the timestamp, amounts and label are typed explicitly.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

RAW_COLUMNS = [
    "timestamp",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "is_laundering",
]

# The header as written in the IBM files (note the repeated `Account`).
EXPECTED_HEADER = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]

TIMESTAMP_FORMAT = "%Y/%m/%d %H:%M"
MINUTES_PER_DAY = 1440


def read_header(csv_path: Path) -> list[str]:
    with Path(csv_path).open(encoding="utf-8", newline="") as f:
        line = f.readline()
    line = line.removeprefix(chr(0xFEFF))  # UTF-8 byte-order mark, if any
    return [h.strip() for h in line.rstrip("\r\n").split(",")]


def read_raw_transactions(csv_path: Path, *, threads: int | None = None) -> pl.DataFrame:
    """All 11 columns as String (never inferred), renamed by position, + `row_id` (file order)."""
    header = read_header(csv_path)
    if header != EXPECTED_HEADER:
        raise ValueError(f"unexpected CSV header in {csv_path}: {header}")
    raw = pl.read_csv(
        csv_path,
        has_header=False,
        skip_rows=1,
        new_columns=RAW_COLUMNS,
        schema={c: pl.String for c in RAW_COLUMNS},
        infer_schema=False,
        # Keep empty fields as "" rather than null: the join to Patterns.txt is on raw strings.
        empty_string_is_null=False,
        n_threads=threads,
    )
    if raw.width != len(RAW_COLUMNS):
        raise ValueError(f"expected {len(RAW_COLUMNS)} columns, got {raw.width}")
    return raw.with_row_index("row_id").with_columns(pl.col("row_id").cast(pl.Int64))


def parse_transactions(raw: pl.DataFrame) -> pl.DataFrame:
    """Typed transactions: ts Datetime(us), amounts Float64, is_laundering Int8; ids stay String."""
    df = raw.select(
        "row_id",
        pl.col("timestamp")
        .str.strptime(pl.Datetime("us"), TIMESTAMP_FORMAT, strict=True)
        .alias("ts"),
        "from_bank",
        "from_account",
        "to_bank",
        "to_account",
        pl.col("amount_received").cast(pl.Float64, strict=True),
        "receiving_currency",
        pl.col("amount_paid").cast(pl.Float64, strict=True),
        "payment_currency",
        "payment_format",
        pl.col("is_laundering").cast(pl.Int8, strict=True),
    )
    bad_label = df.filter(~pl.col("is_laundering").is_in([0, 1])).height
    if bad_label:
        raise ValueError(f"{bad_label} rows have is_laundering outside {{0, 1}}")
    for c in ("ts", "amount_received", "amount_paid", "is_laundering"):
        n_null = df[c].null_count()
        if n_null:
            raise ValueError(f"{n_null} null values in {c}")
    return df


def assign_time(df: pl.DataFrame) -> pl.DataFrame:
    """Add `minute` (since 00:00 of the first timestamp's day) and 1-based `day`."""
    origin = df.select(pl.col("ts").min().dt.truncate("1d")).item()
    minute = (pl.col("ts") - pl.lit(origin)).dt.total_minutes().cast(pl.Int64)
    return df.with_columns(minute.alias("minute")).with_columns(
        (pl.col("minute") // MINUTES_PER_DAY + 1).cast(pl.Int16).alias("day")
    )


def sort_and_rank(df: pl.DataFrame) -> pl.DataFrame:
    """Stable sort by (ts, row_id); `rank` = position in that order (unique global event rank)."""
    out = df.sort(["ts", "row_id"], maintain_order=True).with_row_index("rank")
    out = out.with_columns(pl.col("rank").cast(pl.Int64))
    first = ["row_id", "rank"]
    return out.select(first + [c for c in out.columns if c not in first])


def build_accounts(df: pl.DataFrame) -> pl.DataFrame:
    """Unique (bank, account) pairs, ids assigned in lexicographic (bank, account) string order."""
    pairs = pl.concat(
        [
            df.select(pl.col("from_bank").alias("bank"), pl.col("from_account").alias("account")),
            df.select(pl.col("to_bank").alias("bank"), pl.col("to_account").alias("account")),
        ]
    ).unique()
    pairs = pairs.sort(["bank", "account"])
    return pairs.with_row_index("account_id").with_columns(pl.col("account_id").cast(pl.Int32))


def map_accounts(df: pl.DataFrame, accounts: pl.DataFrame) -> pl.DataFrame:
    """Add `src` / `dst` (Int32 account ids); row order is preserved."""
    out = df
    for side, col in (("from", "src"), ("to", "dst")):
        key = accounts.select(
            pl.col("bank").alias(f"{side}_bank"),
            pl.col("account").alias(f"{side}_account"),
            pl.col("account_id").alias(col),
        )
        out = out.join(
            key,
            on=[f"{side}_bank", f"{side}_account"],
            how="left",
            validate="m:1",
            maintain_order="left",
        )
        n_null = out[col].null_count()
        if n_null:
            raise ValueError(f"{n_null} rows have an unmapped {side} account")
    return out


def fit_fx(df: pl.DataFrame, base_currency: str) -> dict:
    """Infer units-per-base-currency rates from cross-currency rows.

    For each (pay, recv) pair, median of log(amount_received / amount_paid) ~= log u[recv] -
    log u[pay]; solve all pairs by least squares with log u[base] = 0.
    """
    seen = set(df["payment_currency"].unique().to_list()) | set(
        df["receiving_currency"].unique().to_list()
    )
    pairs = (
        df.filter(
            (pl.col("payment_currency") != pl.col("receiving_currency"))
            & (pl.col("amount_paid") > 0)
            & (pl.col("amount_received") > 0)
        )
        .group_by("payment_currency", "receiving_currency")
        .agg(
            (pl.col("amount_received").log() - pl.col("amount_paid").log())
            .median()
            .alias("log_ratio"),
            pl.len().alias("n"),
        )
        .sort("payment_currency", "receiving_currency")
    )
    pay = pairs["payment_currency"].to_list()
    recv = pairs["receiving_currency"].to_list()
    log_ratio = pairs["log_ratio"].to_numpy()

    # Currencies must be connected to the base currency through observed pairs.
    adj: dict[str, set[str]] = defaultdict(set)
    for p, r in zip(pay, recv, strict=True):
        adj[p].add(r)
        adj[r].add(p)
    reach, stack = {base_currency}, [base_currency]
    while stack:
        for nxt in adj[stack.pop()]:
            if nxt not in reach:
                reach.add(nxt)
                stack.append(nxt)
    missing = sorted(seen - reach)
    if missing:
        raise ValueError(
            f"FX: no cross-currency path to {base_currency!r} for {missing}; "
            "cannot infer their rates from the fit rows"
        )

    unknowns = sorted(reach - {base_currency})
    units = {base_currency: 1.0}
    if unknowns:
        col = {c: i for i, c in enumerate(unknowns)}
        a = np.zeros((len(pay), len(unknowns)))
        for i, (p, r) in enumerate(zip(pay, recv, strict=True)):
            if r in col:
                a[i, col[r]] += 1.0
            if p in col:
                a[i, col[p]] -= 1.0
        sol, *_ = np.linalg.lstsq(a, log_ratio, rcond=None)
        units.update({c: float(np.exp(sol[col[c]])) for c in unknowns})
    return {"units_per_base": dict(sorted(units.items())), "n_pairs": len(pay)}


def apply_fx(df: pl.DataFrame, units_per_base: dict[str, float]) -> pl.DataFrame:
    """Add `amount_usd` = amount_paid / units_per_base[payment_currency]."""
    missing = sorted(set(df["payment_currency"].unique().to_list()) - set(units_per_base))
    if missing:
        raise ValueError(f"FX: no rate for payment currencies {missing}")
    rate = pl.col("payment_currency").replace_strict(
        list(units_per_base), list(units_per_base.values()), return_dtype=pl.Float64
    )
    return df.with_columns((pl.col("amount_paid") / rate).alias("amount_usd"))
