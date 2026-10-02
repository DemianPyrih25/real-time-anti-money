"""As-of rule for the SQL rules (PLAN.md §4): a transaction in minute m depends only on events in
minutes <= m - 1. Perturbing, adding or deleting any other event in minute >= m must leave its
severities and fired flags unchanged; labels never enter a severity.
"""

from __future__ import annotations

import shutil

import numpy as np
import polars as pl
import pytest

from aml.data.split import assign_split
from aml.paths import DataPaths
from aml.rules.sql_baseline import (
    SCENARIOS,
    TX_COLUMNS,
    apply_thresholds,
    compute_severities,
    connect,
    hub_degree_cap,
    register_transactions,
    run_rules_stage,
)
from tests.fixtures.rules_frames import dense_tie_frame, small_cfg

NO_HUBS = 10**9


def severities(tx: pl.DataFrame, rules_cfg: dict, hub_cap: int) -> pl.DataFrame:
    con = connect(threads=2)
    try:
        register_transactions(con, tx)
        return compute_severities(con, rules_cfg, hub_cap)
    finally:
        con.close()


def loosest(rules_cfg: dict) -> dict[str, float]:
    """Loosest grid value per scenario, so that many rows fire."""
    return {s: float(min(rules_cfg["scenarios"][s]["grid"])) for s in SCENARIOS}


def perturb(
    tx: pl.DataFrame, targets: pl.DataFrame, m: int, data_cfg: dict, seed: int
) -> pl.DataFrame:
    """Change every event with minute >= m except `targets` (all in minute m)."""
    rng = np.random.default_rng(seed)
    keep = pl.col("row_id").is_in(targets["row_id"].to_list())
    fixed = tx.filter((pl.col("minute") < m) | keep)
    future = tx.filter((pl.col("minute") >= m) & ~keep)
    # Delete about a fifth of the future.
    future = future.filter(pl.Series(rng.random(future.height) > 0.2))
    n = future.height
    us, vs = targets["src"].to_list(), targets["dst"].to_list()
    accounts = np.unique(np.concatenate([tx["src"].to_numpy(), tx["dst"].to_numpy()]))
    touch = np.array(us + vs, dtype=np.int32)
    # Re-point some counterparties at the targets' accounts, both directions.
    new_src = np.where(rng.random(n) < 0.3, rng.choice(touch, n), future["src"].to_numpy())
    new_dst = np.where(rng.random(n) < 0.3, rng.choice(touch, n), future["dst"].to_numpy())
    amount = future["amount_usd"].to_numpy() * rng.uniform(0.5, 2.0, n)
    pick = rng.random(n)
    amount = np.where(pick < 0.15, 5000.0, np.where(pick < 0.3, 9500.0, amount))
    formats = np.array(["ACH", "Cash", "Bitcoin", "Wire", "Cheque"])
    fmt = np.where(rng.random(n) < 0.4, rng.choice(formats, n), future["payment_format"].to_numpy())
    future = future.with_columns(
        pl.Series("src", new_src, dtype=pl.Int32),
        pl.Series("dst", new_dst, dtype=pl.Int32),
        pl.Series("amount_usd", amount),
        pl.Series("amount_paid", amount),
        pl.Series("payment_format", fmt),
    )
    # New events in minute m (the same minute as the targets) and just after, touching the
    # targets' accounts: reverse edges, 2-hop paths back, fan-in/out, bursts, pass-through.
    last_minute = int(data_cfg["split"]["test"][1]) * 1440 - 1
    new = []
    for u, v, a in zip(us, vs, targets["amount_usd"].to_list(), strict=True):
        x = int(rng.choice(accounts))
        for dt in (0, 0, 1, 7):
            for s, d in ((v, u), (u, x), (x, u), (v, x), (x, v), (u, v), (u, u), (x, x)):
                amt = float(rng.choice([a, 5000.0, 9500.0, a * 1.01]))
                new.append((min(m + dt, last_minute), s, d, amt, str(rng.choice(formats))))
    base = int(tx["row_id"].max()) + 1
    base_rank = int(tx["rank"].max()) + 1
    added = pl.DataFrame(
        {
            "row_id": np.arange(base, base + len(new), dtype=np.int64),
            "rank": np.arange(base_rank, base_rank + len(new), dtype=np.int64),
            "minute": np.array([r[0] for r in new], dtype=np.int64),
            "src": np.array([r[1] for r in new], dtype=np.int32),
            "dst": np.array([r[2] for r in new], dtype=np.int32),
            "amount_usd": [r[3] for r in new],
            "amount_paid": [r[3] for r in new],
            "payment_format": [r[4] for r in new],
        }
    ).with_columns((pl.col("minute") // 1440 + 1).cast(pl.Int16).alias("day"))
    added = assign_split(added, data_cfg).select(TX_COLUMNS)
    out = pl.concat([fixed.select(TX_COLUMNS), future.select(TX_COLUMNS), added])
    # Same-minute peers must be invisible in either rank direction: about half of the new
    # minute-m rows sort BEFORE the targets in (minute, rank) order, the rest after. Severities
    # are compared by row_id, so renumbering rank is safe.
    new = pl.col("row_id") >= base
    early = pl.Series(rng.random(out.height) < 0.5)
    key = pl.when(new & (pl.col("minute") == m) & early).then(-1).when(new).then(1).otherwise(0)
    out = (
        out.with_columns(key.alias("_k"))
        .sort(["minute", "_k", "rank"])
        .with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("rank"))
        .drop("_k")
    )
    first_target = out.filter(pl.col("row_id").is_in(targets["row_id"].implode()))["rank"].min()
    assert out.filter(new & (pl.col("minute") == m) & (pl.col("rank") < first_target)).height
    return out


@pytest.fixture(scope="module")
def tx(prepared) -> pl.DataFrame:
    return pl.read_parquet(prepared.transactions).select(TX_COLUMNS)


@pytest.fixture(scope="module")
def hub_cap(prepared, rules_cfg) -> int:
    con = connect()
    register_transactions(con, prepared.transactions)
    cap = hub_degree_cap(con, rules_cfg["hub_degree_quantile"])
    con.close()
    return cap


def pick_targets(sev: pl.DataFrame, tx: pl.DataFrame, days: tuple[int, int], seed: int):
    """Target rows: the top event of each scenario in `days`, some random ones, and a row that
    shares its minute with other rows."""
    cand = sev.join(tx.select("row_id", "minute", "src", "dst", "amount_usd"), on="row_id")
    cand = cand.filter(pl.col("day").is_between(*days))
    picks = [cand.sort([s, "row_id"], descending=[True, False]).head(1) for s in SCENARIOS]
    picks.append(cand.sample(4, seed=seed))
    shared = cand.filter(pl.len().over("minute") > 1)
    if shared.height:
        picks.append(shared.sample(1, seed=seed))
    return pl.concat(picks).unique("row_id", keep="first", maintain_order=True)


def check_asof(tx, rules_cfg, data_cfg, cap, targets, seed_base):
    base = severities(tx, rules_cfg, cap)
    thr = loosest(rules_cfg)
    base_fired = apply_thresholds(base, thr)
    changed_future = 0
    for k, row in enumerate(targets.iter_rows(named=True)):
        m = row["minute"]
        tgt = targets.filter(pl.col("row_id") == row["row_id"])
        pert = perturb(tx, tgt, m, data_cfg, seed=seed_base + k)
        got = severities(pert, rules_cfg, cap)
        # The target and every row before minute m keep all severities and flags, exactly.
        protected = tx.filter((pl.col("minute") < m) | (pl.col("row_id") == row["row_id"]))
        ids = protected.select("row_id")
        a = ids.join(base, on="row_id").sort("row_id")
        b = ids.join(got, on="row_id").sort("row_id")
        assert a.height == b.height == protected.height
        assert a.equals(b), (row["row_id"], m)
        fa = ids.join(base_fired, on="row_id").sort("row_id")
        fb = ids.join(apply_thresholds(got, thr), on="row_id").sort("row_id")
        assert fa.equals(fb)
        # The perturbation is not vacuous: later rows did change.
        later = got.join(base, on="row_id", suffix="_b").filter(
            pl.any_horizontal([pl.col(s) != pl.col(f"{s}_b") for s in SCENARIOS])
        )
        changed_future += later.height
    assert changed_future > 0
    # The targets exercise every scenario with a non-zero severity.
    tsev = targets.select(SCENARIOS)
    for s in SCENARIOS:
        assert (tsev[s] > 0).any(), s


def test_rules_asof_after_train(tx, rules_cfg, data_cfg, hub_cap):
    """Targets after the train split, with the train-fitted hub cap (fixed by the as-of cut)."""
    sev = severities(tx, rules_cfg, hub_cap)
    first_after_train = data_cfg["split"]["val_early"][0]
    targets = pick_targets(sev, tx, (first_after_train, 18), seed=0)
    check_asof(tx, rules_cfg, data_cfg, hub_cap, targets, seed_base=100)


def test_rules_asof_in_train_without_hubs(tx, rules_cfg, data_cfg):
    """Train-day targets. The hub set is a train-fitted statistic (PLAN.md §4), so perturbing
    train rows could move it; with no hubs every severity must depend on the past only."""
    sev = severities(tx, rules_cfg, NO_HUBS)
    lo, hi = data_cfg["split"]["train"]
    targets = pick_targets(sev, tx, (lo, hi), seed=1)
    check_asof(tx, rules_cfg, data_cfg, NO_HUBS, targets, seed_base=200)


def test_rules_asof_dense_same_minute_ties(rules_cfg, data_cfg):
    """A tie-heavy frame (many events per account and minute, both directions), short windows:
    the same-minute peers that sort before a target in rank order must stay invisible."""
    cfg = small_cfg(rules_cfg, window=10, hop=5)
    dense = dense_tie_frame(seed=11, n=400)
    sev = severities(dense, cfg, NO_HUBS)
    targets = pick_targets(sev, dense, (1, 1), seed=2)
    check_asof(dense, cfg, data_cfg, NO_HUBS, targets, seed_base=300)


def test_flipping_labels_changes_no_severity(prepared, rules_cfg, data_cfg, tmp_path):
    flipped = DataPaths(tmp_path / "flipped")
    flipped.transactions.parent.mkdir(parents=True)
    shutil.copy(prepared.transactions, flipped.transactions)
    labels = pl.read_parquet(prepared.labels)
    flipped.labels.parent.mkdir(parents=True)
    # The alias matters: `1 - col` is named "literal", which would leave is_laundering unchanged.
    flip = (1 - pl.col("is_laundering")).cast(pl.Int8).alias("is_laundering")
    labels.with_columns(flip).write_parquet(flipped.labels)
    written = pl.read_parquet(flipped.labels)
    assert written.columns == labels.columns
    assert (written["is_laundering"] != labels["is_laundering"]).all()  # really flipped
    run_rules_stage(prepared, tmp_path / "a", rules_cfg, data_cfg, threads=2)
    run_rules_stage(flipped, tmp_path / "b", rules_cfg, data_cfg, threads=2)
    a = pl.read_parquet(tmp_path / "a" / "severities.parquet")
    b = pl.read_parquet(tmp_path / "b" / "severities.parquet")
    assert a.equals(b)
