"""The prepare-data stage: raw CSV + Patterns.txt -> canonical Parquet tables (M1 spec §2, §3.7).

Raw files must already be in `paths.raw_dir` (the Modal job calls `fetch_dataset` first).
"""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl

from aml.data import ingest, patterns, schemas, split
from aml.io import write_json_atomic, write_parquet_atomic
from aml.paths import DataPaths

TRANSACTION_COLUMNS = [
    "row_id",
    "rank",
    "ts",
    "minute",
    "day",
    "split",
    "from_bank",
    "from_account",
    "to_bank",
    "to_account",
    "src",
    "dst",
    "amount_received",
    "receiving_currency",
    "amount_paid",
    "payment_currency",
    "payment_format",
    "amount_usd",
]
EXPECTED_KEYS = ("rows", "accounts", "positives", "pattern_attempts", "pattern_transactions")


class _Timer:
    def __init__(self) -> None:
        self.timings: dict[str, float] = {}
        self._t = time.perf_counter()

    def lap(self, name: str) -> None:
        now = time.perf_counter()
        self.timings[name] = round(now - self._t, 3)
        self._t = now


def check_expected(counts: dict[str, int], expected: dict | None) -> None:
    """Raise listing every count that differs from `expected` (null expected values are skipped)."""
    expected = expected or {}
    diffs = {
        k: {"expected": expected[k], "got": counts[k]}
        for k in EXPECTED_KEYS
        if expected.get(k) is not None and int(expected[k]) != counts[k]
    }
    if diffs:
        raise ValueError(f"prepare_data: counts differ from data_cfg['expected']: {diffs}")


def prepare_data(
    paths: DataPaths, data_cfg: dict, *, threads: int | None = None, run_eda: bool = True
) -> dict:
    """Build transactions / accounts / labels Parquet and the FX JSON; returns a summary dict."""
    fx_cfg = data_cfg["fx"]
    if fx_cfg["fit_split"] != "train":
        raise ValueError(
            f"fx.fit_split is {fx_cfg['fit_split']!r}: FX must be fitted on train only (PLAN.md §4)"
        )
    ds = data_cfg["dataset"]
    csv_path = paths.raw_dir / ds["transactions_file"]
    pat_path = paths.raw_dir / ds["patterns_file"]
    timer = _Timer()

    raw = ingest.read_raw_transactions(csv_path, threads=threads)
    timer.lap("read_csv")

    # Labels come straight from the raw strings (the join key), before any parsing.
    pat, n_attempts = patterns.parse_patterns_with_count(pat_path)
    labels = patterns.build_labels(raw, pat)
    timer.lap("labels")

    tx = ingest.parse_transactions(raw)
    del raw
    tx = ingest.sort_and_rank(ingest.assign_time(tx))
    timer.lap("parse_sort")

    accounts = ingest.build_accounts(tx)
    tx = ingest.map_accounts(tx, accounts)
    tx = split.assign_split(tx, data_cfg)
    timer.lap("accounts_split")

    fit_rows = tx.filter(pl.col("split") == fx_cfg["fit_split"])
    fx = ingest.fit_fx(fit_rows, fx_cfg["base_currency"])
    del fit_rows
    tx = ingest.apply_fx(tx, fx["units_per_base"])
    timer.lap("fx")

    counts = {
        "rows": tx.height,
        "accounts": accounts.height,
        "positives": int(labels["is_laundering"].sum()),
        "pattern_attempts": n_attempts,
        "pattern_transactions": pat.height,
    }
    tx_pos = int(tx["is_laundering"].sum())
    if tx_pos != counts["positives"] or labels.height != tx.height:
        raise ValueError("labels table disagrees with the transactions")
    check_expected(counts, data_cfg.get("expected"))

    per_split = (
        tx.group_by("split")
        .agg(
            pl.len().alias("rows"), pl.col("is_laundering").sum().cast(pl.Int64).alias("positives")
        )
        .sort("split")
    )
    tx = tx.select(TRANSACTION_COLUMNS)  # drops is_laundering: labels live only in `labels`
    schemas.validate_transactions(tx)
    schemas.validate_accounts(accounts)
    schemas.validate_labels(labels)
    timer.lap("validate")

    fx_json = {
        "base_currency": fx_cfg["base_currency"],
        "units_per_base": fx["units_per_base"],
        "fit_split": fx_cfg["fit_split"],
        "n_pairs": fx["n_pairs"],
    }
    write_parquet_atomic(tx, paths.transactions)
    write_parquet_atomic(accounts, paths.accounts)
    write_parquet_atomic(labels, paths.labels)
    write_json_atomic(fx_json, paths.fx_rates)
    n_days = int(tx["day"].max())
    del tx, accounts, labels
    timer.lap("write")

    summary: dict = {
        "counts": counts,
        "n_days": n_days,
        "splits": {
            r["split"]: {"rows": r["rows"], "positives": r["positives"]}
            for r in per_split.iter_rows(named=True)
        },
        "fx": fx_json,
        "outputs": {
            "transactions": str(paths.transactions),
            "accounts": str(paths.accounts),
            "labels": str(paths.labels),
            "fx_rates": str(paths.fx_rates),
        },
    }
    if run_eda:
        from aml.data.eda import run_eda as _run_eda

        eda = _run_eda(paths, data_cfg, threads=threads, raw_csv=Path(csv_path))
        summary["eda"] = {"report": str(paths.reports / "eda.md"), "tail_days": eda["tail_days"]}
        timer.lap("eda")
    summary["timings_s"] = timer.timings
    write_json_atomic(summary, paths.parquet_dir / "prepare_summary.json")
    return summary
