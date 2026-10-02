"""Greedy threshold tuning of the rule baseline (PLAN.md §6 M1 "Tuning")."""

from __future__ import annotations

import copy
import math

import numpy as np
import polars as pl
import pytest

from aml.rules.sql_baseline import (
    SCENARIOS,
    apply_thresholds,
    scenario_diagnostics,
    tune_thresholds,
    tune_thresholds_detailed,
)

RATES = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1]


def random_severities(n: int = 6000, seed: int = 0) -> tuple[pl.DataFrame, np.ndarray]:
    """Severities on a grid-like scale, weakly informative of y; plus some non-tune rows."""
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.03).astype(np.int8)
    cols = {}
    for j, s in enumerate(SCENARIOS):
        if s == "rapid_pass_through":
            base = rng.random(n) * (rng.random(n) < 0.4)
            cols[s] = np.clip(base + 0.25 * y * rng.random(n), 0, 1)
        else:
            lam = 1.0 + j
            cols[s] = rng.poisson(lam, n) * (rng.random(n) < 0.6) + y * rng.poisson(2 + j, n)
            cols[s] = cols[s].astype(np.float64)
    split = np.where(rng.random(n) < 0.8, "val_early", "train")
    sev = pl.DataFrame({"row_id": np.arange(n, dtype=np.int64), "split": split, **cols})
    return sev, y


def cfg_with_rates(rules_cfg: dict, rates: list[float]) -> dict:
    cfg = copy.deepcopy(rules_cfg)
    cfg["alert_rate"] = rates[0]
    cfg["sensitivity_alert_rates"] = rates[1:]
    return cfg


def tune_stats(sev: pl.DataFrame, y: np.ndarray, thr: dict, tune: str = "val_early"):
    fired = apply_thresholds(sev, thr)["any"].to_numpy()
    m = (sev["split"] == tune).to_numpy()
    alerts = int(fired[m].sum())
    tp = int((fired[m] & (y[m] == 1)).sum())
    return alerts, alerts / m.sum(), tp / max(1, int(y[m].sum()))


def test_alert_rate_cap_and_monotone_recall(rules_cfg):
    sev, y = random_severities()
    cfg = cfg_with_rates(rules_cfg, RATES)
    recalls = []
    for r in RATES:
        thr = tune_thresholds(sev, y, cfg, r)
        assert set(thr) == set(SCENARIOS)
        alerts, rate, recall = tune_stats(sev, y, thr)
        assert rate <= r, (r, alerts)
        n_tune = int((sev["split"] == "val_early").sum())
        assert alerts <= math.floor(r * n_tune)
        for s, t in thr.items():
            assert t is None or t in [float(g) for g in cfg["scenarios"][s]["grid"]]
        recalls.append(recall)
    assert recalls == sorted(recalls)
    assert recalls[-1] > recalls[0] and recalls[-1] > 0.5


def test_deterministic_and_order_free(rules_cfg):
    sev, y = random_severities(seed=1)
    a = tune_thresholds(sev, y, rules_cfg, 0.01)
    b = tune_thresholds(sev, y, rules_cfg, 0.01)
    assert a == b
    perm = np.random.default_rng(5).permutation(sev.height)
    c = tune_thresholds(sev[perm], y[perm], rules_cfg, 0.01)
    assert a == c


def test_only_tune_split_rows_matter(rules_cfg):
    sev, y = random_severities(seed=2)
    thr = tune_thresholds(sev, y, rules_cfg, 0.01)
    other = (sev["split"] != "val_early").to_numpy()
    sev2 = sev.with_columns(
        [
            pl.when(pl.col("split") != "val_early").then(999.0).otherwise(pl.col(s)).alias(s)
            for s in SCENARIOS
        ]
    )
    y2 = y.copy()
    y2[other] = 1 - y2[other]
    assert tune_thresholds(sev2, y2, rules_cfg, 0.01) == thr
    # Non-tune labels may even be missing.
    y3 = pl.Series("y", y, dtype=pl.Int8).scatter(np.flatnonzero(other), None)
    assert tune_thresholds(sev, y3, rules_cfg, 0.01) == thr
    y4 = pl.Series("y", y, dtype=pl.Int8).scatter(np.flatnonzero(~other)[:1], None)
    with pytest.raises(ValueError, match="missing"):
        tune_thresholds(sev, y4, rules_cfg, 0.01)


def test_unused_scenarios_are_none(rules_cfg):
    sev, y = random_severities(seed=3)
    # A scenario that never fires, and one that fires only on negatives.
    sev = sev.with_columns(
        pl.lit(0.0).alias("round_trip"),
        pl.Series("structuring", np.where(y == 1, 0.0, 50.0)),
    )
    thr = tune_thresholds(sev, y, rules_cfg, 0.05)
    assert thr["round_trip"] is None
    assert thr["structuring"] is None
    assert any(v is not None for v in thr.values())
    fired = apply_thresholds(sev, thr)
    assert not fired["fired_round_trip"].any()
    assert not fired["fired_structuring"].any()
    # No positives at all: everything stays off.
    none = tune_thresholds(sev, np.zeros(sev.height, dtype=np.int8), rules_cfg, 0.05)
    assert all(v is None for v in none.values())


def test_greedy_picks_best_ratio_and_skips_ahead(rules_cfg):
    cfg = copy.deepcopy(rules_cfg)
    n = 1000
    y = np.zeros(n, dtype=np.int8)
    y[:10] = 1
    cols = {s: np.zeros(n) for s in SCENARIOS}
    # fan_in: its strictest value (50) flags 2 positives + 0 negatives (ratio 2/2 = 1).
    cols["fan_in_velocity"][:2] = 50
    # fan_out: 3 flags nothing new; only the loosest value (3 == its minimum) flags 8 rows,
    # 4 positive: reachable only by skipping ahead over the empty grid values.
    cols["fan_out_velocity"][2:6] = 3
    cols["fan_out_velocity"][100:104] = 3
    # structuring: 4 positives buried in 200 negatives (ratio 4/204).
    cols["structuring"][6:10] = 2
    cols["structuring"][200:400] = 2
    sev = pl.DataFrame({"row_id": np.arange(n), "split": ["val_early"] * n, **cols})
    thr = tune_thresholds(sev, y, cfg, 0.01)  # budget: 10 alerts
    assert thr["fan_in_velocity"] == 50.0
    assert thr["fan_out_velocity"] == 3.0
    assert thr["structuring"] is None  # 204 alerts never fit
    thr = tune_thresholds(sev, y, cfg, 0.5)
    assert thr["structuring"] == 2.0


def test_apply_thresholds_semantics():
    sev = pl.DataFrame(
        {
            "row_id": [1, 2, 3, 4],
            "split": ["val_early"] * 4,
            **{s: [0.0, 1.0, 2.0, 3.0] for s in SCENARIOS},
        }
    )
    thr = dict.fromkeys(SCENARIOS)
    thr["fan_in_velocity"] = 2.0
    thr["round_trip"] = 0.0  # a zero threshold still needs severity > 0
    out = apply_thresholds(sev, thr)
    assert out.columns == ["row_id", *[f"fired_{s}" for s in SCENARIOS], "any"]
    assert out["fired_fan_in_velocity"].to_list() == [False, False, True, True]
    assert out["fired_round_trip"].to_list() == [False, True, True, True]
    assert out["fired_structuring"].to_list() == [False] * 4
    assert out["any"].to_list() == [False, True, True, True]


def test_bad_inputs(rules_cfg):
    sev, y = random_severities(n=200)
    with pytest.raises(ValueError):
        tune_thresholds(sev, y[:-1], rules_cfg, 0.01)
    with pytest.raises(ValueError):
        tune_thresholds(sev, y, rules_cfg, 0.0)
    with pytest.raises(ValueError):
        tune_thresholds(sev.drop("split"), y, rules_cfg, 0.01)


def test_tuning_on_fixture_severities(prepared, rules_cfg):
    from aml.rules.sql_baseline import (
        compute_severities,
        connect,
        hub_degree_cap,
        register_transactions,
    )

    con = connect(threads=2)
    register_transactions(con, prepared.transactions)
    sev = compute_severities(con, rules_cfg, hub_degree_cap(con, 0.999))
    con.close()
    labels = pl.read_parquet(prepared.labels, columns=["row_id", "is_laundering"])
    y = sev.join(labels, on="row_id", how="left", maintain_order="left")["is_laundering"]
    cfg = cfg_with_rates(rules_cfg, [0.01, 0.05, 0.1])
    prev = -1.0
    for r in (0.01, 0.05, 0.1):
        thr = tune_thresholds(sev, y, cfg, r)
        _, rate, recall = tune_stats(sev, y.to_numpy(), thr)
        assert rate <= r
        assert recall >= prev
        prev = recall
    assert prev > 0


def _naive_setup(sev: pl.DataFrame, y: np.ndarray, cfg: dict, rate: float):
    tune = (sev["split"] == cfg["tune_split"]).to_numpy()
    y_t = y[tune] == 1
    vals = {s: sev[s].to_numpy()[tune] for s in SCENARIOS}
    grids = {
        s: sorted({float(g) for g in cfg["scenarios"][s]["grid"]}, reverse=True) for s in SCENARIOS
    }
    budget = math.floor(rate * int(tune.sum()) + 1e-9)

    def union(t: dict) -> np.ndarray:
        u = np.zeros(int(tune.sum()), dtype=bool)
        for s, v in t.items():
            if v is not None:
                u |= (vals[s] >= v) & (vals[s] > 0)
        return u

    return y_t, grids, budget, union


def naive_greedy(sev: pl.DataFrame, y: np.ndarray, cfg: dict, rate: float, mode="ratio") -> dict:
    """The greedy of the M1 spec written directly over the severities (one rate only).

    mode "ratio": best new-TP / new-alert ratio (ties: more new TPs); "gain": most new TPs
    (ties: fewer new alerts). Remaining ties: scenario order, then the stricter value.
    """
    y_t, grids, budget, union = _naive_setup(sev, y, cfg, rate)
    thr: dict = dict.fromkeys(SCENARIOS)
    while True:
        cur = union(thr)
        best = None
        for s in SCENARIOS:
            looser = grids[s] if thr[s] is None else [g for g in grids[s] if g < thr[s]]
            for g in looser:
                nxt = union({**thr, s: g})
                d_a = int(nxt.sum() - cur.sum())
                d_tp = int((nxt & y_t).sum() - (cur & y_t).sum())
                if nxt.sum() > budget or d_tp <= 0:
                    continue
                key = (d_tp / d_a, d_tp) if mode == "ratio" else (d_tp, -d_a)
                if best is None or key > best[0]:
                    best = (key, s, g)
        if best is None:
            return thr
        thr[best[1]] = best[2]


def naive_best_single(sev: pl.DataFrame, y: np.ndarray, cfg: dict, rate: float) -> dict:
    y_t, grids, budget, union = _naive_setup(sev, y, cfg, rate)
    best = None
    for s in SCENARIOS:
        for g in grids[s]:
            u = union({s: g})
            a, tp = int(u.sum()), int((u & y_t).sum())
            if a > budget or tp == 0:
                continue
            if best is None or (tp, -a) > best[0]:
                best = ((tp, -a), s, g)
    thr: dict = dict.fromkeys(SCENARIOS)
    if best is not None:
        thr[best[1]] = best[2]
    return thr


def naive_tune(sev: pl.DataFrame, y: np.ndarray, cfg: dict, rate: float) -> tuple[str, dict]:
    """Best of the three candidates: most TPs; ties: the ratio greedy, then fewer alerts."""
    y_t, _, _, union = _naive_setup(sev, y, cfg, rate)
    cands = [
        ("ratio_greedy", naive_greedy(sev, y, cfg, rate, "ratio")),
        ("gain_greedy", naive_greedy(sev, y, cfg, rate, "gain")),
        ("best_single", naive_best_single(sev, y, cfg, rate)),
    ]
    best = None
    for i, (name, thr) in enumerate(cands):
        u = union(thr)
        key = (int((u & y_t).sum()), name == "ratio_greedy", -int(u.sum()), -i)
        if best is None or key > best[0]:
            best = (key, name, thr)
    return best[1], best[2]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("rate", [0.002, 0.01, 0.05])
def test_matches_naive_greedy(rules_cfg, seed, rate):
    sev, y = random_severities(n=1500, seed=seed)
    cfg = cfg_with_rates(rules_cfg, [rate])  # no smaller configured rate to fall back on
    thr, info = tune_thresholds_detailed(sev, y, cfg, rate)
    name, want = naive_tune(sev, y, cfg, rate)
    assert (info["candidate"], thr) == (name, want)
    assert tune_thresholds(sev, y, cfg, rate) == want


def _blocking_problem() -> tuple[pl.DataFrame, np.ndarray]:
    """A small precise move (fan_in >= 50: 5 alerts, 3 TP, ratio 0.6) and a large one
    (structuring >= 8: 48 alerts, 20 TP, ratio 0.42) that do not both fit a budget of 50."""
    n = 10_000
    cols = {s: np.zeros(n) for s in SCENARIOS}
    y = np.zeros(n, dtype=np.int8)
    cols["fan_in_velocity"][0:5] = 50
    y[0:3] = 1
    cols["structuring"][100:148] = 8
    y[100:120] = 1
    return pl.DataFrame({"row_id": np.arange(n), "split": ["val_early"] * n, **cols}), y


def test_small_precise_move_does_not_block_a_larger_one(rules_cfg):
    sev, y = _blocking_problem()
    cfg = cfg_with_rates(rules_cfg, [0.005, 0.01])
    # The ratio greedy alone takes fan_in first (3 TP), after which structuring no longer fits.
    assert naive_greedy(sev, y, cfg, 0.005) == {**dict.fromkeys(SCENARIOS), "fan_in_velocity": 50.0}
    thr, info = tune_thresholds_detailed(sev, y, cfg, 0.005)
    assert thr == {**dict.fromkeys(SCENARIOS), "structuring": 8.0}
    assert info["tune_tp"] == 20 and info["tune_alerts"] == 48 <= info["budget"] == 50
    assert info["candidate"] != "ratio_greedy"
    # With room for both, the ratio greedy takes both (23 TP).
    thr, info = tune_thresholds_detailed(sev, y, cfg, 0.01)
    assert thr == {**dict.fromkeys(SCENARIOS), "fan_in_velocity": 50.0, "structuring": 8.0}
    assert info["tune_tp"] == 23 and info["candidate"] == "ratio_greedy"


def test_scenario_diagnostics_explain_infeasible_scenarios(rules_cfg):
    sev, y = _blocking_problem()
    # fan_out fires at its strictest value (50) on 600 rows: never within a budget of 50 or 100.
    fan_out = np.zeros(sev.height)
    fan_out[1000:1600] = 50
    y[1000] = 1
    sev = sev.with_columns(pl.Series("fan_out_velocity", fan_out))
    cfg = cfg_with_rates(rules_cfg, [0.005, 0.01])
    # Pin the grids so the test does not depend on configs/rules.yaml's values.
    cfg["scenarios"]["fan_out_velocity"]["grid"] = [3, 10, 50]
    cfg["scenarios"]["structuring"]["grid"] = [1, 2, 4, 8]
    rates = {"0p005": 0.005, "0p01": 0.01}
    thresholds = {tag: tune_thresholds(sev, y, cfg, r) for tag, r in rates.items()}
    d = scenario_diagnostics(sev, y, cfg, thresholds, rates)
    assert d["tune_rows"] == sev.height
    assert d["scenarios"]["fan_out_velocity"]["alerts_at_strictest"] == 600
    assert d["scenarios"]["structuring"]["alerts_at_strictest"] == 48
    head = d["rates"]["0p005"]
    assert head["budget"] == 50
    assert head["active"] == ["structuring"]
    assert "fan_out_velocity" in head["infeasible"] and "structuring" not in head["infeasible"]
    assert head["scenarios"]["structuring"] == {
        "threshold": 8.0,
        "feasible": True,
        "alone_alerts": 48,
        "alone_tp": 20,
    }
    assert head["scenarios"]["fan_out_velocity"]["alone_alerts"] is None
