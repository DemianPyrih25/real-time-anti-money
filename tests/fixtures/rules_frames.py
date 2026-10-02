"""Hand-made transaction frames for the SQL rule tests (unit and leakage)."""

from __future__ import annotations

import copy

import numpy as np
import polars as pl

from aml.rules.sql_baseline import SCENARIOS, TX_COLUMNS


def make_tx(rows: list[dict]) -> pl.DataFrame:
    """Tiny transactions frame from dicts with minute/src/dst (+ optional amounts/format/split)."""
    recs = []
    for k, r in enumerate(rows):
        amount = float(r.get("amount_usd", 123.45))
        recs.append(
            {
                "row_id": k,
                "minute": int(r["minute"]),
                "src": int(r["src"]),
                "dst": int(r["dst"]),
                "amount_paid": float(r.get("amount_paid", amount)),
                "amount_usd": amount,
                "payment_format": r.get("payment_format", "ACH"),
                "split": r.get("split", "val_early"),
            }
        )
    df = pl.DataFrame(recs).with_columns(
        pl.col("row_id").cast(pl.Int64),
        pl.col("minute").cast(pl.Int64),
        pl.col("src").cast(pl.Int32),
        pl.col("dst").cast(pl.Int32),
        (pl.col("minute") // 1440 + 1).cast(pl.Int16).alias("day"),
    )
    df = (
        df.sort(["minute", "row_id"])
        .with_row_index("rank")
        .with_columns(pl.col("rank").cast(pl.Int64))
    )
    return df.select(TX_COLUMNS)


def small_cfg(rules_cfg: dict, window: int, hop: int) -> dict:
    """rules_cfg with every scenario window set to `window` and the round-trip hop gap to `hop`."""
    cfg = copy.deepcopy(rules_cfg)
    for s in SCENARIOS:
        cfg["scenarios"][s]["window_minutes"] = window
    cfg["scenarios"]["round_trip"]["hop_window_minutes"] = hop
    return cfg


def dense_tie_frame(seed: int = 7, n: int = 300, accounts: int = 6, minutes: int = 40):
    """Many transactions per (account, minute): every scenario's same-minute peers are common.

    Few accounts and minutes, in-band (9500, 9900), round (5000, and the in-band amounts),
    high-risk (Cash, Bitcoin) and plain rows, self-loops and reverse edges. The first quarter of
    the minutes is `train`, so a low hub cap turns busy accounts into hubs.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(n):
        minute = int(rng.integers(0, minutes))
        rows.append(
            {
                "minute": minute,
                "src": int(rng.integers(0, accounts)),
                "dst": int(rng.integers(0, accounts)),
                "amount_usd": float(rng.choice([9500.0, 9900.0, 5000.0, 123.45, 777.0])),
                "payment_format": str(rng.choice(["Cash", "Bitcoin", "ACH", "Wire"])),
                "split": "train" if minute < minutes // 4 else "val_early",
            }
        )
    return make_tx(rows)
