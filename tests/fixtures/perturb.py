"""Perturbations of the future for the engine's as-of tests (M1 `perturb` generalised to the
engine's input columns, M2 spec §9.2).

Given a target event at minute m, every other event with minute >= m is changed (accounts,
amounts, formats, currencies, banks), about a fifth are deleted, and new events are added in
minutes m, m + 1 and m + 7 that touch the target's accounts in every way a feature or a rule can
read them: the same pair, reverse edges, 2- and 3-hop returns, scatter-gather siblings, fan-in /
fan-out, self-loops, round, in-band and high-risk amounts. About half of the new minute-m events
are ranked BEFORE the target: same-minute peers must stay invisible in both rank directions.
Events with minute < m are never touched.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from aml.data.split import assign_split
from tests.fixtures.engine_frames import CURRENCIES, ENGINE_TX_COLUMNS, FORMATS

BANKS = ("001", "002", "1", "0119")


def perturb_events(
    tx: pl.DataFrame, targets: pl.DataFrame, m: int, data_cfg: dict, seed: int
) -> pl.DataFrame:
    """Change, delete and add events with minute >= m except `targets` (all in minute m).

    `tx` has ENGINE_TX_COLUMNS in rank order; the result is re-ranked 0..N-1 (row_ids are kept;
    added rows get new ones) and uses only accounts already in `tx`.
    """
    if not (targets["minute"] == m).all():
        raise ValueError("every target must be in minute m")
    rng = np.random.default_rng(seed)
    keep = pl.col("row_id").is_in(targets["row_id"].implode())
    fixed = tx.filter((pl.col("minute") < m) | keep)
    future = tx.filter((pl.col("minute") >= m) & ~keep)
    future = future.filter(pl.Series(rng.random(future.height) > 0.2))  # delete about a fifth
    n = future.height
    us, vs = targets["src"].to_list(), targets["dst"].to_list()
    accounts = np.unique(np.concatenate([tx["src"].to_numpy(), tx["dst"].to_numpy()]))
    touch = np.array(us + vs, dtype=np.int32)

    def pick(values, p: float, current: np.ndarray) -> np.ndarray:
        return np.where(rng.random(n) < p, rng.choice(np.asarray(values), n), current)

    amount = future["amount_usd"].to_numpy() * rng.uniform(0.5, 2.0, n)
    r = rng.random(n)
    amount = np.where(r < 0.15, 5000.0, np.where(r < 0.3, 9500.0, amount))
    future = future.with_columns(
        pl.Series("src", pick(touch, 0.3, future["src"].to_numpy()), dtype=pl.Int32),
        pl.Series("dst", pick(touch, 0.3, future["dst"].to_numpy()), dtype=pl.Int32),
        pl.Series("amount_usd", amount),
        pl.Series(
            "amount_paid", np.where(rng.random(n) < 0.5, amount, 100.0 * rng.integers(1, 99, n))
        ),
        pl.Series("payment_format", pick(FORMATS, 0.4, future["payment_format"].to_numpy())),
        pl.Series("payment_currency", pick(CURRENCIES, 0.3, future["payment_currency"].to_numpy())),
        pl.Series(
            "receiving_currency", pick(CURRENCIES, 0.3, future["receiving_currency"].to_numpy())
        ),
        pl.Series("from_bank", pick(BANKS, 0.3, future["from_bank"].to_numpy())),
        pl.Series("to_bank", pick(BANKS, 0.3, future["to_bank"].to_numpy())),
    )

    last_minute = int(data_cfg["split"]["test"][1]) * 1440 - 1
    new = []
    for u, v, a in zip(us, vs, targets["amount_usd"].to_list(), strict=True):
        x, y, s = (int(rng.choice(accounts)) for _ in range(3))
        edges = (
            (u, v),  # the same pair (a same-minute repeat of a new pair, a later repeat)
            (v, u),  # reverse edge: rev_pair_gap, cyc2, round trip
            (v, x),  # 2-hop return v -> x -> u ...
            (x, u),
            (v, y),  # ... 3-hop return v -> y -> x -> u
            (y, x),
            (s, u),  # scatter-gather: s -> u, s -> x, x -> v
            (s, x),
            (x, v),
            (u, x),  # fan-out / fan-in around both ends
            (y, v),
            (u, u),  # self-loops
            (v, v),
            (x, x),
        )
        for dt in (0, 0, 1, 7):
            for src, dst in edges:
                amt = float(rng.choice([a, 5000.0, 9500.0, a * 1.01, 100.0]))
                new.append(
                    (
                        min(m + dt, last_minute),
                        src,
                        dst,
                        amt,
                        amt if rng.random() < 0.5 else 200.0,
                        str(rng.choice(FORMATS)),
                        str(rng.choice(CURRENCIES)),
                        str(rng.choice(CURRENCIES)),
                        str(rng.choice(BANKS)),
                        str(rng.choice(BANKS)),
                    )
                )
    base = int(tx["row_id"].max()) + 1
    base_rank = int(tx["rank"].max()) + 1
    cols = list(zip(*new, strict=True))
    added = pl.DataFrame(
        {
            "row_id": np.arange(base, base + len(new), dtype=np.int64),
            "rank": np.arange(base_rank, base_rank + len(new), dtype=np.int64),
            "minute": np.array(cols[0], dtype=np.int64),
            "src": np.array(cols[1], dtype=np.int32),
            "dst": np.array(cols[2], dtype=np.int32),
            "amount_usd": list(cols[3]),
            "amount_paid": list(cols[4]),
            "payment_format": list(cols[5]),
            "payment_currency": list(cols[6]),
            "receiving_currency": list(cols[7]),
            "from_bank": list(cols[8]),
            "to_bank": list(cols[9]),
        }
    ).with_columns((pl.col("minute") // 1440 + 1).cast(pl.Int16).alias("day"))
    added = assign_split(added, data_cfg).select(ENGINE_TX_COLUMNS)
    out = pl.concat([fixed, future, added])
    # Half of the new minute-m rows sort before the targets in (minute, rank) order, the rest after.
    is_new = pl.col("row_id") >= base
    early = pl.Series(rng.random(out.height) < 0.5)
    key = (
        pl.when(is_new & (pl.col("minute") == m) & early).then(-1).when(is_new).then(1).otherwise(0)
    )
    out = (
        out.with_columns(key.alias("_k"))
        .sort(["minute", "_k", "rank"])
        .with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("rank"))
        .drop("_k")
    )
    first = out.filter(pl.col("row_id").is_in(targets["row_id"].implode()))["rank"].min()
    if not out.filter(is_new & (pl.col("minute") == m) & (pl.col("rank") < first)).height:
        raise AssertionError("no new same-minute peer is ranked before the target")
    return out


def permute_within_minutes(tx: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Re-rank the events inside every minute at random (row_ids kept, ranks 0..N-1)."""
    rng = np.random.default_rng(seed)
    out = (
        tx.with_columns(pl.Series("_r", rng.random(tx.height)))
        .sort(["minute", "_r"])
        .drop("_r")
        .with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("rank"))
    )
    if out["row_id"].to_list() == tx["row_id"].to_list():
        raise AssertionError("the permutation changed nothing")
    return out


def add_disconnected_component(
    tx: pl.DataFrame,
    data_cfg: dict,
    *,
    first_account: int,
    n_accounts: int,
    n_events: int,
    seed: int,
) -> pl.DataFrame:
    """Add events among fresh accounts [first_account, first_account + n_accounts) at minutes
    spread over the existing span (many earlier than most rows, many sharing a minute with
    existing rows), then re-rank. No existing account is touched."""
    rng = np.random.default_rng(seed)
    minutes = tx["minute"].to_numpy()
    fresh = np.arange(first_account, first_account + n_accounts, dtype=np.int32)
    src = rng.choice(fresh, n_events)
    dst = np.where(rng.random(n_events) < 0.15, src, rng.choice(fresh, n_events))
    # Half of the events reuse an existing minute (ties with old rows), half land anywhere.
    lo, hi = int(minutes.min()), int(minutes.max())
    mins = np.where(
        rng.random(n_events) < 0.5,
        rng.choice(minutes, n_events),
        rng.integers(lo, hi + 1, n_events),
    )
    amount = np.round(rng.lognormal(7, 1.5, n_events), 2)
    base = int(tx["row_id"].max()) + 1
    added = pl.DataFrame(
        {
            "row_id": np.arange(base, base + n_events, dtype=np.int64),
            "rank": np.full(n_events, -1, dtype=np.int64),
            "minute": mins.astype(np.int64),
            "src": src,
            "dst": dst.astype(np.int32),
            "amount_usd": amount,
            "amount_paid": amount,
            "payment_format": rng.choice(np.array(FORMATS), n_events).tolist(),
            "payment_currency": rng.choice(np.array(CURRENCIES), n_events).tolist(),
            "receiving_currency": rng.choice(np.array(CURRENCIES), n_events).tolist(),
            "from_bank": rng.choice(np.array(BANKS), n_events).tolist(),
            "to_bank": rng.choice(np.array(BANKS), n_events).tolist(),
            "day": (mins // 1440 + 1).astype(np.int16),
        }
    )
    added = assign_split(added, data_cfg).select(ENGINE_TX_COLUMNS)
    out = pl.concat([tx.select(ENGINE_TX_COLUMNS), added])
    # Old rows keep their relative order; new rows go first or last within their minute.
    side = pl.Series(rng.random(out.height) < 0.5)
    key = pl.when(pl.col("rank") >= 0).then(0).when(side).then(-1).otherwise(1)
    return (
        out.with_columns(key.alias("_k"), pl.col("rank").alias("_r"))
        .sort(["minute", "_k", "_r", "row_id"])
        .drop("_k", "_r")
        .with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("rank"))
    )
