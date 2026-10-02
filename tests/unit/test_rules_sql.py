"""SQL rule baseline: every severity equals a brute-force pure-Python reference (PLAN.md §6 M1)."""

from __future__ import annotations

import copy
import math
import warnings
from bisect import bisect_left
from collections import Counter, defaultdict

import numpy as np
import polars as pl
import pytest

from aml.config import rate_tag
from aml.io import read_json
from aml.rules.sql_baseline import (
    HUB_SEGMENTABLE,
    MAX_ROUND_TRIP_PATHS,
    SCENARIOS,
    apply_thresholds,
    check_rules_config,
    compute_severities,
    connect,
    hub_degree_cap,
    load_steps,
    register_transactions,
    render_steps,
    run_rules_stage,
    severity_stats,
)
from tests.fixtures.rules_frames import dense_tie_frame, make_tx, small_cfg

# ---------------------------------------------------------------------------------------------
# Brute-force reference: the scenario definitions written out directly, one event at a time.


def _cents(s: pl.Series) -> list[int]:
    # Half away from zero, as DuckDB's round() and tx_features.round_amount_expr.
    return (s * 100).round(0, mode="half_away_from_zero").cast(pl.Int64).to_list()


def reference_severities(tx: pl.DataFrame, cfg: dict, hub_cap: int) -> pl.DataFrame:
    """Severities per the scenario semantics in PLAN.md §6 M1 / the M1 spec §3.10."""
    tx = tx.sort("rank")
    rid = tx["row_id"].to_list()
    minute = tx["minute"].to_list()
    src = tx["src"].to_list()
    dst = tx["dst"].to_list()
    usd = tx["amount_usd"].to_list()
    fmt = tx["payment_format"].to_list()
    split = tx["split"].to_list()
    usd_c = _cents(tx["amount_usd"])
    paid_c = _cents(tx["amount_paid"])
    n = len(rid)

    sc = cfg["scenarios"]
    round_cents = round(cfg["round_unit"] * 100)
    t_hi = float(cfg["structuring_threshold_usd"])
    t_lo = float(cfg["structuring_band_low"]) * t_hi
    hr_formats = set(cfg["high_risk_formats"])
    is_round = [c > 0 and c % round_cents == 0 for c in paid_c]
    in_band = [t_lo <= a < t_hi for a in usd]
    high_risk = [f in hr_formats for f in fmt]

    deg: Counter[int] = Counter()
    for i in range(n):
        if split[i] == "train":
            deg[src[i]] += 1
            deg[dst[i]] += 1
    hubs = {a for a, d in deg.items() if d > hub_cap}
    excl = {s for s in SCENARIOS if sc[s].get("exclude_hub_senders", False)}

    # Per-account event lists in minute order (rows are already rank-sorted).
    out_all: dict[int, list[int]] = defaultdict(list)
    in_all: dict[int, list[int]] = defaultdict(list)
    out_nsl: dict[int, list[int]] = defaultdict(list)
    in_nsl: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        out_all[src[i]].append(i)
        in_all[dst[i]].append(i)
        if src[i] != dst[i]:
            out_nsl[src[i]].append(i)
            in_nsl[dst[i]].append(i)
    lists = {"out_all": out_all, "in_all": in_all, "out_nsl": out_nsl, "in_nsl": in_nsl}
    mins = {k: {a: [minute[i] for i in lst] for a, lst in d.items()} for k, d in lists.items()}

    def window(kind: str, acct: int, lo: int, hi: int) -> list[int]:
        """Row indices of `acct`'s events of `kind` with minute in [lo, hi]."""
        m = mins[kind].get(acct)
        if not m:
            return []
        return lists[kind][acct][bisect_left(m, lo) : bisect_left(m, hi + 1)]

    w = {s: int(sc[s]["window_minutes"]) for s in SCENARIOS}
    hop = int(sc["round_trip"]["hop_window_minutes"])
    out = {s: [0.0] * n for s in SCENARIOS}
    for i in range(n):
        m, u, v = minute[i], src[i], dst[i]
        win = {s: (m - w[s], m - 1) for s in SCENARIOS}  # the as-of window of each scenario
        out["fan_in_velocity"][i] = float(
            len({src[j] for j in window("in_all", v, *win["fan_in_velocity"])})
        )
        if not ("fan_out_velocity" in excl and u in hubs):
            out["fan_out_velocity"][i] = float(
                len({dst[j] for j in window("out_all", u, *win["fan_out_velocity"])})
            )
        if u != v:
            inflow = sum(usd_c[j] for j in window("in_nsl", u, *win["rapid_pass_through"]))
            if inflow > 0:
                out["rapid_pass_through"][i] = max(
                    0.0, 1.0 - abs(float(usd_c[i]) / float(inflow) - 1.0)
                )
        for s, flag in (
            ("structuring", in_band),
            ("round_amount_burst", is_round),
            ("high_risk_format_burst", high_risk),
        ):
            if flag[i] and not (s in excl and u in hubs):
                out[s][i] = float(1 + sum(flag[j] for j in window("out_all", u, *win[s])))
        if u != v:
            lo, hi = win["round_trip"]
            paths = sum(1 for j in window("out_nsl", v, lo, hi) if dst[j] == u)  # v -> u
            for j in window("out_nsl", v, lo, hi):  # v -> x at t1
                x, t1 = dst[j], minute[j]
                if x in (u, v) or x in hubs:
                    continue
                for k in window("out_nsl", x, t1, min(t1 + hop, hi)):  # x -> u at t2
                    if dst[k] == u:
                        paths += 1
            out["round_trip"][i] = float(min(paths, MAX_ROUND_TRIP_PATHS))
    return pl.DataFrame({"row_id": rid, **out}, schema_overrides={"row_id": pl.Int64})


# ---------------------------------------------------------------------------------------------
# Helpers


def severities(tx: pl.DataFrame, cfg: dict, hub_cap: int) -> pl.DataFrame:
    con = connect(threads=2)
    try:
        register_transactions(con, tx)
        return compute_severities(con, cfg, hub_cap)
    finally:
        con.close()


def sev_of(sev: pl.DataFrame, row_id: int, scenario: str) -> float:
    return sev.filter(pl.col("row_id") == row_id)[scenario].item()


@pytest.fixture(scope="module")
def tx(prepared) -> pl.DataFrame:
    return pl.read_parquet(prepared.transactions)


@pytest.fixture(scope="module")
def fixture_hub_cap(prepared, rules_cfg) -> int:
    con = connect(threads=2)
    register_transactions(con, prepared.transactions)
    cap = hub_degree_cap(con, rules_cfg["hub_degree_quantile"])
    con.close()
    return cap


# ---------------------------------------------------------------------------------------------
# SQL == reference on the fixture


@pytest.mark.parametrize(
    ("window", "hop", "hub_cap"),
    [
        (None, None, None),  # configs/rules.yaml windows and the fitted hub cap
        (90, 45, 12),  # short windows; a low cap turns many intermediates into hubs
        (1, 0, 0),  # the previous minute only; zero hop gap; every train account is a hub
        (4000, 4000, 30),  # hop window == round-trip window (the H <= W edge case)
    ],
)
def test_sql_equals_bruteforce_reference(tx, rules_cfg, fixture_hub_cap, window, hop, hub_cap):
    cfg = rules_cfg if window is None else small_cfg(rules_cfg, window, hop)
    cap = fixture_hub_cap if hub_cap is None else hub_cap
    got = severities(tx, cfg, cap)
    want = reference_severities(tx, cfg, cap)
    assert got.height == tx.height
    assert got["row_id"].to_list() == tx.sort("rank")["row_id"].to_list()
    assert got.schema["row_id"] == pl.Int64
    for s in SCENARIOS:
        assert got.schema[s] == pl.Float64
        a, b = got[s].to_numpy(), want[s].to_numpy()
        diff = np.flatnonzero(a != b)
        assert diff.size == 0, (s, [(int(i), a[i], b[i]) for i in diff[:5]])
    if window is None:
        # The comparison is not vacuous: every scenario fires somewhere on the fixture.
        for s in SCENARIOS:
            assert (got[s] > 0).sum() > 0, s


def _hub_segmentation(cfg: dict, on: bool) -> dict:
    cfg = copy.deepcopy(cfg)
    for s in HUB_SEGMENTABLE:
        cfg["scenarios"][s]["exclude_hub_senders"] = on
    return cfg


@pytest.mark.parametrize("hub_cap", [12, 30])
def test_sql_equals_reference_with_hub_senders_segmented(tx, rules_cfg, hub_cap):
    cfg = _hub_segmentation(small_cfg(rules_cfg, 90, 45), on=True)
    got = severities(tx, cfg, hub_cap)
    _assert_equals_reference(tx, cfg, hub_cap)
    # Not vacuous: segmentation zeroes some rows that fire without it.
    plain = severities(tx, _hub_segmentation(small_cfg(rules_cfg, 90, 45), on=False), hub_cap)
    for s in HUB_SEGMENTABLE:
        assert ((plain[s] > 0) & (got[s] == 0)).sum() > 0, s


def test_hub_sender_segmentation_zeroes_only_hub_senders(rules_cfg):
    # `hub` becomes a hub through train rows (degree 6 > cap 5); `u` does not (degree 2).
    hub, u = 0, 1
    rows = [
        *[{"minute": 10 + k, "src": hub, "dst": 100 + k, "split": "train"} for k in range(6)],
        {"minute": 20, "src": u, "dst": 200, "split": "train"},
        {"minute": 21, "src": u, "dst": 201, "split": "train"},
        {"minute": 30, "src": hub, "dst": 300},
        {"minute": 31, "src": u, "dst": 301},
    ]
    tx_small = make_tx(rows)
    cfg = small_cfg(rules_cfg, 1440, 60)
    plain = severities(tx_small, _hub_segmentation(cfg, on=False), hub_cap=5)
    seg = severities(tx_small, _hub_segmentation(cfg, on=True), hub_cap=5)
    hub_row = tx_small.filter((pl.col("src") == hub) & (pl.col("minute") == 30))["row_id"].item()
    u_row = tx_small.filter((pl.col("src") == u) & (pl.col("minute") == 31))["row_id"].item()
    assert sev_of(plain, hub_row, "fan_out_velocity") == 6.0
    assert sev_of(seg, hub_row, "fan_out_velocity") == 0.0
    assert sev_of(seg, u_row, "fan_out_velocity") == sev_of(plain, u_row, "fan_out_velocity") == 2.0


def test_hub_segmentation_only_for_sender_scenarios(rules_cfg):
    cfg = copy.deepcopy(rules_cfg)
    cfg["scenarios"]["fan_in_velocity"]["exclude_hub_senders"] = True
    with pytest.raises(ValueError, match="exclude_hub_senders"):
        check_rules_config(cfg)
    cfg = copy.deepcopy(rules_cfg)
    cfg["scenarios"]["fan_out_velocity"]["exclude_hub_senders"] = "yes"
    with pytest.raises(ValueError, match="true or false"):
        check_rules_config(cfg)


def _assert_equals_reference(tx: pl.DataFrame, cfg: dict, cap: int) -> None:
    got = severities(tx, cfg, cap)
    want = reference_severities(tx, cfg, cap)
    assert got["row_id"].to_list() == want["row_id"].to_list()
    for s in SCENARIOS:
        a, b = got[s].to_numpy(), want[s].to_numpy()
        diff = np.flatnonzero(a != b)
        assert diff.size == 0, (s, [(int(i), a[i], b[i]) for i in diff[:5]])


@pytest.mark.parametrize(("window", "hop"), [(10, 5), (5, 0), (30, 12)])
@pytest.mark.parametrize("hub_cap", [10**9, 15])
def test_sql_equals_reference_on_dense_same_minute_ties(rules_cfg, window, hop, hub_cap):
    """The fixture has almost no same-(account, minute) ties; the real data has many (hubs pay
    about 12 transactions a minute). A rank-order leak of same-minute peers shows up here."""
    dense = dense_tie_frame()
    shared_src = dense.filter(pl.len().over(["src", "minute"]) > 1).height
    shared_dst = dense.filter(pl.len().over(["dst", "minute"]) > 1).height
    assert shared_src > dense.height // 2 and shared_dst > dense.height // 2
    _assert_equals_reference(dense, small_cfg(rules_cfg, window, hop), hub_cap)


def test_severities_ignore_row_order(tx, rules_cfg, fixture_hub_cap):
    a = severities(tx, rules_cfg, fixture_hub_cap)
    b = severities(tx.sample(fraction=1.0, shuffle=True, seed=3), rules_cfg, fixture_hub_cap)
    assert a.equals(b)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_severities_ignore_rank_order_within_a_minute(rules_cfg, seed):
    """Only the minute orders events: permuting rank (and row order) inside every minute of a
    tie-heavy frame changes no severity."""
    cfg = small_cfg(rules_cfg, window=10, hop=5)
    dense = dense_tie_frame()
    rng = np.random.default_rng(seed)
    permuted = (
        dense.with_columns(pl.Series("_r", rng.random(dense.height)))
        .sort(["minute", "_r"])
        .with_columns(pl.int_range(pl.len(), dtype=pl.Int64).alias("rank"))
        .drop("_r")
        .sample(fraction=1.0, shuffle=True, seed=seed)
    )
    assert permuted["rank"].to_list() != dense["rank"].to_list()
    a = severities(dense, cfg, 15).sort("row_id")
    b = severities(permuted, cfg, 15).sort("row_id")
    assert a.equals(b)


def test_split_and_day_passed_through(tx, rules_cfg, fixture_hub_cap):
    got = severities(tx, rules_cfg, fixture_hub_cap)
    want = tx.sort("rank").select("row_id", "split", "day")
    assert got.select("row_id", "split", "day").equals(want)


def test_hub_cap_matches_eda_and_reference(prepared, rules_cfg, fixture_hub_cap, tx):
    eda = read_json(prepared.reports / "eda.json")
    q = rules_cfg["hub_degree_quantile"]
    assert fixture_hub_cap == eda["train_degree"]["total_deg"][f"q{q:g}"]
    train = tx.filter(pl.col("split") == "train")
    deg = sorted(Counter(train["src"].to_list() + train["dst"].to_list()).values())
    assert fixture_hub_cap == deg[max(0, math.ceil(q * len(deg)) - 1)]
    con = connect()
    register_transactions(con, tx)
    assert hub_degree_cap(con, 1.0) == deg[-1]
    with pytest.raises(ValueError):
        hub_degree_cap(con, 0.0)
    register_transactions(con, tx.filter(pl.col("split") != "train"))
    with pytest.raises(ValueError, match="no train rows"):
        hub_degree_cap(con, 0.5)


def test_stats_left_in_connection(tx, rules_cfg):
    con = connect()
    register_transactions(con, tx)
    compute_severities(con, rules_cfg, 20)
    stats = severity_stats(con)
    assert stats["rows"] == tx.height
    assert stats["hub_accounts"] > 0
    assert stats["round_trip_candidate_paths"] > 0
    tables = {r[0] for r in con.execute("SELECT table_name FROM duckdb_tables()").fetchall()}
    assert "r_paths" not in tables and "r_stats" in tables
    con.close()


# ---------------------------------------------------------------------------------------------
# Hand-made boundary cases


def test_window_bounds_are_m_minus_w_to_m_minus_1(rules_cfg):
    cfg = small_cfg(rules_cfg, window=10, hop=5)
    # Target 0 -> 1 at minute 100; W = 10, so the window is [90, 99].
    rows = [
        {"minute": 100, "src": 0, "dst": 1},  # row 0: target
        {"minute": 89, "src": 2, "dst": 1},  # too early
        {"minute": 90, "src": 3, "dst": 1},  # first visible minute
        {"minute": 99, "src": 4, "dst": 1},  # last visible minute
        {"minute": 99, "src": 4, "dst": 1},  # same sender again: distinct count stays 2
        {"minute": 100, "src": 5, "dst": 1},  # same minute as the target: invisible
        {"minute": 101, "src": 6, "dst": 1},  # future
        {"minute": 95, "src": 0, "dst": 7},  # sender's fan-out: visible
        {"minute": 100, "src": 0, "dst": 8},  # same minute: invisible
    ]
    sev = severities(make_tx(rows), cfg, hub_cap=10**9)
    assert sev_of(sev, 0, "fan_in_velocity") == 2.0
    assert sev_of(sev, 0, "fan_out_velocity") == 1.0
    # Row 5 (same minute as the target) sees the same window as the target.
    assert sev_of(sev, 5, "fan_in_velocity") == 2.0
    # Row 6 at minute 101 sees [91, 100]: senders 4 (99), 0 and 5 (100) into account 1.
    assert sev_of(sev, 6, "fan_in_velocity") == 3.0


def test_pass_through_bounds_and_self_loops(rules_cfg):
    cfg = small_cfg(rules_cfg, window=10, hop=5)
    rows = [
        {"minute": 100, "src": 0, "dst": 1, "amount_usd": 100.0},  # target: u = 0
        {"minute": 95, "src": 2, "dst": 0, "amount_usd": 60.0},  # inflow
        {"minute": 90, "src": 3, "dst": 0, "amount_usd": 40.0},  # inflow at m - W
        {"minute": 89, "src": 3, "dst": 0, "amount_usd": 999.0},  # too early
        {"minute": 100, "src": 3, "dst": 0, "amount_usd": 999.0},  # same minute
        {"minute": 96, "src": 0, "dst": 0, "amount_usd": 999.0},  # self-loop: not an inflow
        {"minute": 100, "src": 0, "dst": 0, "amount_usd": 100.0},  # self-loop target -> 0
        {"minute": 100, "src": 5, "dst": 6, "amount_usd": 100.0},  # no inflow -> 0
        {"minute": 100, "src": 0, "dst": 7, "amount_usd": 150.0},  # 1 - |1.5 - 1| = 0.5
        {"minute": 100, "src": 0, "dst": 8, "amount_usd": 250.0},  # 1 - |2.5 - 1| < 0 -> 0
    ]
    sev = severities(make_tx(rows), cfg, hub_cap=10**9)
    assert sev_of(sev, 0, "rapid_pass_through") == 1.0
    assert sev_of(sev, 6, "rapid_pass_through") == 0.0
    assert sev_of(sev, 7, "rapid_pass_through") == 0.0
    assert sev_of(sev, 8, "rapid_pass_through") == 0.5
    assert sev_of(sev, 9, "rapid_pass_through") == 0.0


def test_pass_through_inflow_excludes_self_loops(rules_cfg):
    """Chosen definition: inflow = non-self-loop edges into u. A Reinvestment u -> u is not money
    from outside, so reinvesting 1000 and then paying 1000 on is not a pass-through (0.0, not
    1.0). The M2 engine's inflow feature must use the same definition (rule parity)."""
    cfg = small_cfg(rules_cfg, window=100, hop=5)
    rows = [
        {"minute": 0, "src": 0, "dst": 0, "amount_usd": 1000.0},  # u reinvests into itself
        {"minute": 10, "src": 0, "dst": 1, "amount_usd": 1000.0},  # then pays the same on
    ]
    assert sev_of(severities(make_tx(rows), cfg, 10**9), 1, "rapid_pass_through") == 0.0
    # The same amount arriving from another account is a full pass-through.
    rows[0]["src"] = 2
    assert sev_of(severities(make_tx(rows), cfg, 10**9), 1, "rapid_pass_through") == 1.0


def test_round_trip_paths(rules_cfg):
    cfg = small_cfg(rules_cfg, window=100, hop=30)
    u, v, w, hub = 0, 1, 2, 3
    rows = [
        {"minute": 1000, "src": u, "dst": v},  # row 0: target u -> v, window [900, 999]
        {"minute": 999, "src": v, "dst": u},  # 2-hop at m - 1: counts
        {"minute": 900, "src": v, "dst": u},  # 2-hop at m - W: counts
        {"minute": 899, "src": v, "dst": u},  # too early
        {"minute": 1000, "src": v, "dst": u},  # same minute: invisible
        {"minute": 950, "src": v, "dst": w},  # 3-hop v -> w at 950 ...
        {"minute": 950, "src": w, "dst": u},  # ... w -> u at 950 (t1 == t2): counts
        {"minute": 980, "src": w, "dst": u},  # ... w -> u at 980 (gap 30 == H): counts
        {"minute": 981, "src": w, "dst": u},  # ... gap 31 > H: no
        {"minute": 949, "src": w, "dst": u},  # ... before the first hop: no
        {"minute": 920, "src": v, "dst": hub},  # via a hub: excluded
        {"minute": 921, "src": hub, "dst": u},
        # Train rows making `hub` a hub (train degree 4 > cap 3); far in the past.
        *[{"minute": 10, "src": hub, "dst": 100 + k, "split": "train"} for k in range(4)],
        {"minute": 1000, "src": u, "dst": u},  # self-loop target: 0
    ]
    tx_small = make_tx(rows)
    sev = severities(tx_small, cfg, hub_cap=3)
    assert sev_of(sev, 0, "round_trip") == 4.0  # 999, 900 two-hop + (950, 950), (950, 980)
    assert sev_of(sev, len(rows) - 1, "round_trip") == 0.0
    # With the hub cap lifted, the path via `hub` counts too.
    sev = severities(tx_small, cfg, hub_cap=10**9)
    assert sev_of(sev, 0, "round_trip") == 5.0
    # First hop at m - W - 1 with the second inside the window: the path is not in the window.
    rows2 = [
        {"minute": 1000, "src": u, "dst": v},
        {"minute": 899, "src": v, "dst": w},
        {"minute": 905, "src": w, "dst": u},
    ]
    assert sev_of(severities(make_tx(rows2), cfg, hub_cap=10**9), 0, "round_trip") == 0.0


def test_round_trip_is_capped(rules_cfg):
    cfg = small_cfg(rules_cfg, window=1000, hop=1000)
    rows = [{"minute": 2000, "src": 0, "dst": 1}]
    rows += [{"minute": 1500 + k, "src": 1, "dst": 0} for k in range(MAX_ROUND_TRIP_PATHS + 20)]
    sev = severities(make_tx(rows), cfg, hub_cap=10**9)
    assert sev_of(sev, 0, "round_trip") == float(MAX_ROUND_TRIP_PATHS)


# ---------------------------------------------------------------------------------------------
# Definitions shared with the features and taken from rules_cfg


def test_round_amount_matches_tx_features(tx, rules_cfg):
    from aml.features.tx_features import round_amount_expr

    unit = rules_cfg["round_unit"]
    crafted = [
        100.0,
        200.0,
        99.99,
        100.01,
        0.0,
        0.004,
        1e9,
        12300.0,
        99.995,
        199.999999,
        100.000001,
        5000.005,
        1234567800.0,
        250.0,
    ]
    small = make_tx(
        [{"minute": k, "src": k, "dst": k + 1, "amount_paid": a} for k, a in enumerate(crafted)]
    )
    for frame in (tx, small):
        sev = severities(frame, rules_cfg, 10**9)
        sql_flag = sev.join(frame.select("row_id", "amount_paid"), on="row_id", how="left")
        sql_flag = sql_flag.with_columns(round_amount_expr(unit).alias("want"))
        assert (sql_flag["round_amount_burst"] > 0).to_list() == sql_flag["want"].to_list()
    assert (severities(tx, rules_cfg, 10**9)["round_amount_burst"] > 0).any()


def test_structuring_band_and_formats_come_from_config(tx, rules_cfg):
    base = severities(tx, rules_cfg, 10**9).join(
        tx.select("row_id", "amount_usd", "payment_format"), on="row_id"
    )
    t = rules_cfg["structuring_threshold_usd"]
    lo = rules_cfg["structuring_band_low"] * t
    in_band = (base["amount_usd"] >= lo) & (base["amount_usd"] < t)
    assert ((base["structuring"] > 0) == in_band).all()
    hr = base["payment_format"].is_in(rules_cfg["high_risk_formats"])
    assert ((base["high_risk_format_burst"] > 0) == hr).all()

    cfg = copy.deepcopy(rules_cfg)
    cfg["structuring_threshold_usd"] = 3000
    cfg["structuring_band_low"] = 0.5
    cfg["high_risk_formats"] = ["Cheque"]
    alt = severities(tx, cfg, 10**9).join(
        tx.select("row_id", "amount_usd", "payment_format"), on="row_id"
    )
    in_band2 = (alt["amount_usd"] >= 1500.0) & (alt["amount_usd"] < 3000.0)
    assert ((alt["structuring"] > 0) == in_band2).all()
    assert ((alt["high_risk_format_burst"] > 0) == (alt["payment_format"] == "Cheque")).all()
    assert in_band2.sum() != in_band.sum()

    cfg["high_risk_formats"] = []
    none = severities(tx, cfg, 10**9)
    assert (none["high_risk_format_burst"] == 0).all()


# ---------------------------------------------------------------------------------------------
# SQL templating and config validation


def test_sql_steps_render_with_numbers_only(rules_cfg):
    steps = load_steps()
    assert [n for n, _ in steps][-1] == "severities"
    for _, sql in render_steps(rules_cfg, 7):
        assert "${" not in sql and "$" not in sql
    bad = copy.deepcopy(rules_cfg)
    bad["scenarios"]["fan_in_velocity"]["window_minutes"] = "1440; DROP TABLE tx"
    with pytest.raises(ValueError):
        render_steps(bad, 7)
    with pytest.raises(ValueError):
        render_steps(rules_cfg, "7")
    with pytest.raises(ValueError):
        render_steps(rules_cfg, True)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda c: c["scenarios"].pop("round_trip"),
        lambda c: c["scenarios"]["round_trip"].update(hop_window_minutes=5000),
        lambda c: c["scenarios"]["structuring"].update(window_minutes=0),
        lambda c: c["scenarios"]["structuring"].update(window_minutes=1.5),
        lambda c: c["scenarios"]["structuring"].update(grid=[]),
        lambda c: c["scenarios"]["structuring"].update(grid=[0, 1]),
        lambda c: c.update(round_unit=0.001),
        lambda c: c.update(structuring_band_low=1.2),
        lambda c: c.update(alert_rate=0.0),
        lambda c: c.update(sensitivity_alert_rates=[float("nan")]),
        # test is touched once; val_late is reserved for thresholds / calibration (PLAN.md §4)
        lambda c: c.update(tune_split="test"),
        lambda c: c.update(tune_split="val_late"),
    ],
)
def test_bad_config_is_rejected(rules_cfg, mutate):
    cfg = copy.deepcopy(rules_cfg)
    mutate(cfg)
    with pytest.raises(ValueError):
        check_rules_config(cfg)


def test_connect_rejects_bad_memory_limit():
    with pytest.raises(ValueError):
        connect(memory_limit="12GB'; DROP TABLE x; --")


# ---------------------------------------------------------------------------------------------
# The stage


@pytest.fixture(scope="module")
def stage(prepared, rules_cfg, data_cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("rules_out")
    # Whether the tiny fixture triggers the infeasibility warning depends on the config grids;
    # test_stage_warns_about_infeasible_scenarios pins that behaviour explicitly.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return out, run_rules_stage(prepared, out, rules_cfg, data_cfg, threads=2)


def test_stage_warns_about_infeasible_scenarios(prepared, rules_cfg, data_cfg, tmp_path):
    # Every Cash/Bitcoin row fires at the only grid value, far above the fixture's 0.5% budget
    # (3 tune alerts): the stage must name the scenario, not stay silent (R1 diagnostics).
    cfg = copy.deepcopy(rules_cfg)
    cfg["scenarios"]["high_risk_format_burst"]["grid"] = [1]
    with pytest.warns(UserWarning, match="can never be switched on at the headline rate"):
        summary = run_rules_stage(prepared, tmp_path, cfg, data_cfg, threads=2)
    assert (
        "high_risk_format_burst" in summary["scenario_diagnostics"]["rates"]["0p005"]["infeasible"]
    )


def test_stage_writes_spec_outputs(stage, rules_cfg, tx):
    out, summary = stage
    for name in ("severities.parquet", "flags.parquet", "thresholds.json", "summary.json"):
        assert (out / name).exists(), name
    tags = [rate_tag(r) for r in [rules_cfg["alert_rate"], *rules_cfg["sensitivity_alert_rates"]]]
    assert tags == ["0p005", "0p001", "0p01"]

    sev = pl.read_parquet(out / "severities.parquet")
    assert sev.columns == ["row_id", *SCENARIOS]
    assert sev.height == tx.height and sev["row_id"].n_unique() == tx.height

    flags = pl.read_parquet(out / "flags.parquet")
    assert flags.columns == [
        "row_id",
        "split",
        "day",
        *[f"rules_any_{t}" for t in tags],
        *[f"fired_{s}" for s in SCENARIOS],
    ]
    for c in flags.columns[3:]:
        assert flags.schema[c] == pl.Boolean
    assert flags["row_id"].to_list() == sev["row_id"].to_list()

    thr = read_json(out / "thresholds.json")
    assert thr["headline_rate_tag"] == "0p005"
    assert thr["hub_cap"] == summary["hub_cap"]
    assert set(thr["thresholds"]) == set(tags)
    for t in tags:
        assert set(thr["thresholds"][t]) == set(SCENARIOS)
        m = thr["metrics"][t]
        for key in ("tune_recall", "tune_alert_rate", "val_late_alert_rate"):
            assert key in m
        assert m["tune_alert_rate"] <= float(t.replace("p", ".")) + 1e-12
        # Flags are the thresholds applied to the severities.
        want = apply_thresholds(sev, thr["thresholds"][t])["any"]
        assert flags[f"rules_any_{t}"].to_list() == want.to_list()
    head = apply_thresholds(sev, thr["thresholds"]["0p005"])
    for s in SCENARIOS:
        assert flags[f"fired_{s}"].to_list() == head[f"fired_{s}"].to_list()
    # Which scenarios are on, why the others are off, and which search found the thresholds.
    assert thr["n_scenarios"] == len(SCENARIOS)
    diag = thr["scenario_diagnostics"]
    for t in tags:
        on = [s for s in SCENARIOS if thr["thresholds"][t][s] is not None]
        assert thr["active_scenarios"][t] == diag["rates"][t]["active"] == on
        assert thr["tuning"][t]["candidate"] in ("ratio_greedy", "gain_greedy", "best_single")
        assert thr["tuning"][t]["tune_alerts"] <= thr["tuning"][t]["budget"]
        for s in diag["rates"][t]["infeasible"]:
            assert thr["thresholds"][t][s] is None  # an infeasible scenario can never be on
    assert summary["scenario_diagnostics"] == diag
    assert read_json(out / "summary.json")["hub_cap"] == summary["hub_cap"]
    assert summary["timings_s"]["total"] >= 0
