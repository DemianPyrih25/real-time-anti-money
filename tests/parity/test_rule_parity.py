"""Exact rule parity (M2 spec §6, §9.2): the engine's 7 severities == M1's scenarios.sql.

Every row is compared with `==` on Float64 (bit-equal) over the M1 grid of (W, H) windows, several
hub caps and hub-sender segmentation sets, on the prepared fixture, tie-heavy frames and
Hypothesis frames; the fired flags are then compared at every grid value of every scenario. The
brute-force reference is held to the same SQL on the same frames (it is validated, not trusted).
"""

from __future__ import annotations

import copy
import os

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from aml.rules.sql_baseline import (
    HUB_SEGMENTABLE,
    SCENARIOS,
    apply_thresholds,
    compute_severities,
    connect,
    hub_accounts,
    register_transactions,
)
from tests.fixtures.engine_frames import (
    NO_HUBS,
    dense_engine_frame,
    engine_tx,
    features_cfg,
    fitted_hub_cap,
    fixture_spec,
    make_spec,
    require_engine,
    run_engine,
    train_hubs,
    with_hub_flags,
)
from tests.fixtures.engine_ref import Reference, reference_frame_cached
from tests.fixtures.rules_frames import small_cfg

GRID_WH = [(1440, 1440), (10, 5), (30, 0)]  # M1's (W, H) grid; H = W and H = 0 are edges
# Hub-sender flag sets: every segmentable scenario is on in one set and off in another, and every
# pair of scenarios differs in some set (a swapped `excl` index cannot pass).
FLAG_SETS = {
    "none": (),
    "all": tuple(HUB_SEGMENTABLE),
    "struct_hr": ("structuring", "high_risk_format_burst"),
    "round_hr": ("round_amount_burst", "high_risk_format_burst"),
}
DENSE_FEATURES = features_cfg(short=5, long=12, sg=5)
SLOW = os.environ.get("AML_HYPOTHESIS_PROFILE") == "slow"


def sql_severities(frame: pl.DataFrame, cfg: dict, hub_cap: int) -> pl.DataFrame:
    con = connect(threads=2)
    try:
        register_transactions(con, frame)
        return compute_severities(con, cfg, hub_cap)
    finally:
        con.close()


def engine_severities(frame: pl.DataFrame, spec) -> pl.DataFrame:
    """row_id, the 7 severities and rule_trunc from an engine run, in rank order."""
    rows = run_engine(frame, spec)
    i = spec.i_sev
    data = {"row_id": frame["row_id"].to_list()}
    for k, s in enumerate(SCENARIOS):
        data[s] = [r[i + k] for r in rows]
    data["rule_trunc"] = [r[spec.i_rule_trunc] for r in rows]
    return pl.DataFrame(data, schema_overrides={s: pl.Float64 for s in SCENARIOS})


def reference_severities(frame: pl.DataFrame, spec) -> pl.DataFrame:
    ref = Reference.from_frame(frame, spec)
    rows = [ref.row(i) for i in range(len(ref))]
    data = {"row_id": frame["row_id"].to_list()}
    for k, s in enumerate(SCENARIOS):
        data[s] = [r[spec.i_sev + k] for r in rows]
    return pl.DataFrame(data, schema_overrides={s: pl.Float64 for s in SCENARIOS})


def mismatches(got: pl.DataFrame, want: pl.DataFrame) -> dict[str, list]:
    """Per scenario, (row_id, got, want) of rows whose severities differ bitwise (rows with
    rule_trunc = 1 in `got` are excluded, as the rules job excludes them)."""
    assert got["row_id"].to_list() == want["row_id"].to_list()
    keep = (
        (got["rule_trunc"] == 0).to_numpy()
        if "rule_trunc" in got.columns
        else np.ones(got.height, bool)
    )
    out = {}
    for s in SCENARIOS:
        a = got[s].to_numpy().view(np.uint64)
        b = want[s].to_numpy().view(np.uint64)
        bad = np.flatnonzero((a != b) & keep)
        if bad.size:
            ids = got["row_id"].to_numpy()
            out[s] = [(int(ids[k]), got[s][k], want[s][k]) for k in bad[:5]]
    return out


def assert_flags_equal(got: pl.DataFrame, want: pl.DataFrame, cfg: dict) -> None:
    """The fired flags agree at every grid value of every scenario (M1 apply_thresholds)."""
    grids = {s: sorted({float(g) for g in cfg["scenarios"][s]["grid"]}) for s in SCENARIOS}
    for k in range(max(len(g) for g in grids.values())):
        thr = {s: (g[k] if k < len(g) else None) for s, g in grids.items()}
        assert apply_thresholds(got, thr).equals(apply_thresholds(want, thr)), thr


def check_parity(frame: pl.DataFrame, cfg: dict, hub_cap: int, feats: dict | None = None) -> dict:
    """Engine (and, when asked, reference) severities == SQL on every row; returns stats."""
    spec = make_spec(frame, cfg, feats, hub_cap=hub_cap)
    sql = sql_severities(frame, cfg, hub_cap)
    eng = engine_severities(frame, spec)
    bad = mismatches(eng, sql)
    assert not bad, bad
    assert (eng["rule_trunc"] == 0).all()
    assert_flags_equal(eng.drop("rule_trunc"), sql.select("row_id", *SCENARIOS), cfg)
    return {"hubs": len(spec.hubs), "nonzero": {s: int((sql[s] > 0).sum()) for s in SCENARIOS}}


# ---------------------------------------------------------------------------------------------
# The hub list is the r_hubs definition


@pytest.mark.parametrize("cap", [0, 3, 30, NO_HUBS])
def test_hub_list_matches_sql_definition(prepared, cap):
    tx = pl.read_parquet(prepared.transactions)
    con = connect(threads=1)
    try:
        register_transactions(con, tx)
        assert hub_accounts(con, cap) == train_hubs(tx, cap)
    finally:
        con.close()


# ---------------------------------------------------------------------------------------------
# The reference is held to the SQL too (no engine needed)


def test_reference_severities_equal_sql_on_fixture(prepared, rules_cfg):
    spec, tx = fixture_spec(prepared, rules_cfg, hub_cap=30)
    ref = reference_frame_cached(tx, spec).select("row_id", *SCENARIOS)
    bad = mismatches(ref, sql_severities(tx, rules_cfg, 30))
    assert not bad, bad


@pytest.mark.parametrize(("w", "h"), GRID_WH)
def test_reference_severities_equal_sql_on_dense_frames(rules_cfg, w, h):
    frame = dense_engine_frame(seed=0)
    cfg = with_hub_flags(small_cfg(rules_cfg, w, h), FLAG_SETS["all"])
    for cap in (20, NO_HUBS):
        spec = make_spec(frame, cfg, DENSE_FEATURES, hub_cap=cap)
        bad = mismatches(reference_severities(frame, spec), sql_severities(frame, cfg, cap))
        assert not bad, (cap, bad)


# ---------------------------------------------------------------------------------------------
# Engine == SQL on the fixture


@pytest.fixture(scope="module")
def fitted_cap(prepared, rules_cfg) -> int:
    return fitted_hub_cap(prepared.transactions, rules_cfg["hub_degree_quantile"])


@pytest.mark.parametrize(
    ("w", "h", "cap", "flags"),
    [
        (1440, 1440, 30, None),  # 9 hubs on the fixture; the config's own hub-sender flags
        (10, 5, 30, None),
        (30, 0, 30, None),
        (10, 5, 3, None),  # every train account is a hub
        (1440, 1440, "fitted", None),  # the fitted cap (no account exceeds it: = 10^9 here)
        (1440, 1440, 30, "all"),
    ],
)
def test_engine_severities_equal_sql_on_fixture(prepared, rules_cfg, fitted_cap, w, h, cap, flags):
    require_engine()
    tx = pl.read_parquet(prepared.transactions)
    cfg = small_cfg(rules_cfg, w, h)
    if flags is not None:
        cfg = with_hub_flags(cfg, FLAG_SETS[flags])
    cap = fitted_cap if cap == "fitted" else cap
    stats = check_parity(tx, cfg, cap)
    if (w, h, cap) == (1440, 1440, 30):  # not vacuous: every scenario fires
        assert all(stats["nonzero"][s] > 0 for s in SCENARIOS), stats


def test_engine_default_config_parity_and_m1_flags(prepared, rules_cfg, fitted_cap):
    """configs/rules.yaml as it is, at the fitted cap: the M1 configuration of the rules job."""
    require_engine()
    tx = pl.read_parquet(prepared.transactions)
    check_parity(tx, rules_cfg, fitted_cap)


def test_tiny_rule_budget_only_lowers_truncated_round_trips(prepared, rules_cfg):
    """With rule_visits = 1 the walk truncates: only rows flagged rule_trunc may differ, and only
    in round_trip, downwards (a lower bound)."""
    require_engine()
    tx = pl.read_parquet(prepared.transactions)
    spec = make_spec(tx, rules_cfg, features_cfg(rule_visits=1), hub_cap=30)
    eng = engine_severities(tx, spec)
    sql = sql_severities(tx, rules_cfg, 30)
    trunc = eng["rule_trunc"] == 1
    assert trunc.any()
    assert not mismatches(eng, sql)  # untruncated rows are exact
    for s in SCENARIOS:
        a, b = eng[s].filter(trunc), sql[s].filter(trunc)
        if s == "round_trip":
            assert (a <= b).all()
        else:
            assert a.equals(b), s


# ---------------------------------------------------------------------------------------------
# Engine == SQL on tie-heavy frames (many events per account and minute, both directions)


def dense_cap(frame: pl.DataFrame) -> int:
    train = frame.filter(pl.col("split") == "train")
    deg = sorted(pl.concat([train["src"], train["dst"]]).value_counts()["count"].to_list())
    return int(deg[len(deg) // 2])


@pytest.mark.parametrize(("w", "h"), GRID_WH)
def test_engine_parity_dense_full_grid(rules_cfg, w, h):
    require_engine()
    frame = dense_engine_frame(seed=0)
    caps = (dense_cap(frame), 3, NO_HUBS)
    for cap in caps:
        flag_sets = FLAG_SETS if cap != NO_HUBS else {"none": ()}
        for name, on in flag_sets.items():
            cfg = with_hub_flags(small_cfg(rules_cfg, w, h), on)
            try:
                check_parity(frame, cfg, cap, DENSE_FEATURES)
            except AssertionError as e:
                raise AssertionError(f"cap={cap} flags={name}: {e}") from None


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_engine_parity_dense_more_seeds(rules_cfg, seed):
    require_engine()
    frame = dense_engine_frame(seed=seed, n=400)
    for w, h in GRID_WH:
        check_parity(frame, small_cfg(rules_cfg, w, h), dense_cap(frame), DENSE_FEATURES)


# ---------------------------------------------------------------------------------------------
# Hypothesis frames

AMOUNTS = (0.0, 0.004, 0.005, 0.015, 99.995, 100.0, 8999.99, 9000.0, 9999.995, 10000.0, 1e11)
FMTS = ("ACH", "Cash", "Bitcoin", "Wire", "Reinvestment", "Cheque")
amount = st.one_of(
    st.sampled_from(AMOUNTS), st.floats(0, 1e7, allow_nan=False, allow_infinity=False)
)


@st.composite
def frames(draw):
    n = draw(st.integers(1, 40))
    rows = []
    for _ in range(n):
        src = draw(st.integers(0, 6))
        dst = src if draw(st.integers(0, 4)) == 0 else draw(st.integers(0, 6))
        a = draw(amount)
        paid = a if draw(st.booleans()) else draw(amount)
        minute = draw(st.integers(0, 24))
        rows.append(
            {
                "minute": minute,
                "src": src,
                "dst": dst,
                "amount_usd": a,
                "amount_paid": paid,
                "payment_format": draw(st.sampled_from(FMTS)),
                "split": "train" if minute < 8 else "val_early",
            }
        )
    w = draw(st.sampled_from([1, 2, 3, 5, 10]))
    h = draw(st.integers(0, w))
    cap = draw(st.sampled_from([0, 2, 5, NO_HUBS]))
    on = [s for s in HUB_SEGMENTABLE if draw(st.booleans())]
    return engine_tx(rows), w, h, cap, on


@settings(
    max_examples=200 if SLOW else 25,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(case=frames())
def test_engine_and_reference_parity_on_hypothesis_frames(rules_cfg, case):
    require_engine()
    frame, w, h, cap, on = case
    cfg = with_hub_flags(small_cfg(copy.deepcopy(rules_cfg), w, h), on)
    spec = make_spec(frame, cfg, features_cfg(short=3, long=7, sg=3), hub_cap=cap)
    sql = sql_severities(frame, cfg, cap)
    assert not mismatches(engine_severities(frame, spec), sql)
    assert not mismatches(reference_severities(frame, spec), sql)
