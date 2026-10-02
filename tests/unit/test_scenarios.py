"""`scenarios.severities` (M2 spec §6): the scalar mirror of scenarios.sql, bit for bit.

The rule inputs (`RuleSupport`) are built here by brute force from small frames, so these tests
check the severity formulas, the hub segmentation and the float operations against the M1 SQL
independently of the engine (the engine's own parity is tested in tests/parity).
"""

from __future__ import annotations

import copy
import math
import random
import struct

import numpy as np
import polars as pl
import pytest

from aml.features.spec import EXACT_INT_LIMIT, EXCL_SCENARIOS, NumericRangeError
from aml.rules import scenarios
from aml.rules.scenarios import RuleSupport, severities
from aml.rules.sql_baseline import (
    HUB_SEGMENTABLE,
    MAX_ROUND_TRIP_PATHS,
    SCENARIOS,
    compute_severities,
    connect,
    hub_accounts,
    register_transactions,
    sql_params,
)
from tests.fixtures.rules_frames import dense_tie_frame, make_tx, small_cfg

P = {"max_round_trip_paths": MAX_ROUND_TRIP_PATHS}
NO_EXCL = (False, False, False, False)
ALL_EXCL = (True, True, True, True)


def rs(**kw) -> RuleSupport:
    """A RuleSupport with neutral defaults (u -> v, nothing flagged, all counts 0)."""
    base = dict.fromkeys(RuleSupport._fields, 0) | {"u": 1, "v": 2}
    base.update(kw)
    base["self_loop"] = int(base["u"] == base["v"])
    return RuleSupport(**base)


def sev(r: RuleSupport, excl=NO_EXCL) -> dict[str, float]:
    out = severities(r, P, excl)
    assert len(out) == len(SCENARIOS) and all(type(x) is float for x in out)
    return dict(zip(SCENARIOS, out, strict=True))


def bits(x: float) -> int:
    return struct.unpack("<Q", struct.pack("<d", x))[0]


# ---------------------------------------------------------------------------------------------
# The contract


def test_rule_support_field_order_is_the_contract():
    assert RuleSupport._fields == (
        "u",
        "v",
        "self_loop",
        "usd_c",
        "in_band",
        "is_round",
        "hr",
        "hub_u",
        "fan_in_uniq_v",
        "fan_out_uniq_u",
        "inflow_c",
        "c2",
        "c3",
        "st_cnt_u",
        "ra_cnt_u",
        "hr_cnt_u",
    )
    assert (
        EXCL_SCENARIOS
        == tuple(HUB_SEGMENTABLE)
        == (
            "fan_out_velocity",
            "structuring",
            "round_amount_burst",
            "high_risk_format_burst",
        )
    )
    # A plain tuple in field order is accepted and gives the same result.
    r = rs(fan_in_uniq_v=3, inflow_c=1000, usd_c=900, in_band=1, st_cnt_u=2)
    assert severities(tuple(r), P, NO_EXCL) == severities(r, P, NO_EXCL)


# ---------------------------------------------------------------------------------------------
# Hand cases


def test_counts_pass_through_as_floats():
    r = rs(fan_in_uniq_v=7, fan_out_uniq_u=4)
    s = sev(r)
    assert s["fan_in_velocity"] == 7.0 and s["fan_out_velocity"] == 4.0
    assert s["rapid_pass_through"] == s["round_trip"] == 0.0
    assert s["structuring"] == s["round_amount_burst"] == s["high_risk_format_burst"] == 0.0


def test_bursts_are_one_plus_count_only_when_the_event_is_flagged():
    s = sev(rs(in_band=1, st_cnt_u=0, is_round=1, ra_cnt_u=4, hr=1, hr_cnt_u=9))
    assert (s["structuring"], s["round_amount_burst"], s["high_risk_format_burst"]) == (
        1.0,
        5.0,
        10.0,
    )
    # Earlier flagged payments alone do not fire: the event's own flag gates the burst.
    s = sev(rs(st_cnt_u=5, ra_cnt_u=5, hr_cnt_u=5))
    assert (s["structuring"], s["round_amount_burst"], s["high_risk_format_burst"]) == (0, 0, 0)


@pytest.mark.parametrize("hub_u", [0, 1])
@pytest.mark.parametrize("excl", [NO_EXCL, ALL_EXCL, (True, False, True, False)])
def test_hub_segmentation(hub_u, excl):
    r = rs(
        hub_u=hub_u,
        fan_in_uniq_v=3,
        fan_out_uniq_u=8,
        in_band=1,
        st_cnt_u=2,
        is_round=1,
        ra_cnt_u=3,
        hr=1,
        hr_cnt_u=4,
        inflow_c=1000,
        usd_c=1000,
        c2=1,
        c3=1,
    )
    s = sev(r, excl)
    want = {
        "fan_in_velocity": 3.0,  # never segmented
        "fan_out_velocity": 8.0,
        "rapid_pass_through": 1.0,  # never segmented
        "round_trip": 2.0,  # never segmented (hubs only restrict intermediates)
        "structuring": 3.0,
        "round_amount_burst": 4.0,
        "high_risk_format_burst": 5.0,
    }
    for name, on in zip(EXCL_SCENARIOS, excl, strict=True):
        if on and hub_u:
            want[name] = 0.0
    assert s == want


def test_pass_through_zero_inflow_and_self_loop_targets():
    assert sev(rs(usd_c=500, inflow_c=0))["rapid_pass_through"] == 0.0
    # A self-loop target is not a query row of r_pass (0), and has no round trip either; the
    # fan counts and bursts still count it (self-loops are included there, as in the SQL).
    s = sev(rs(u=3, v=3, usd_c=500, inflow_c=500, c2=4, c3=4, fan_in_uniq_v=2, in_band=1))
    assert s["rapid_pass_through"] == 0.0 and s["round_trip"] == 0.0
    assert s["fan_in_velocity"] == 2.0 and s["structuring"] == 1.0


@pytest.mark.parametrize(
    ("usd_c", "inflow_c", "want"),
    [
        (10000, 10000, 1.0),
        (15000, 10000, 0.5),
        (5000, 10000, 0.5),
        (20000, 10000, 0.0),  # 1 - |2 - 1| = 0
        (25000, 10000, 0.0),  # negative, clipped
        (0, 10000, 0.0),
        (1, 3, 1.0 - abs(1.0 / 3.0 - 1.0)),
    ],
)
def test_pass_through_values(usd_c, inflow_c, want):
    got = sev(rs(usd_c=usd_c, inflow_c=inflow_c))["rapid_pass_through"]
    assert bits(got) == bits(want)


def test_pass_through_is_bit_identical_to_duckdb():
    """The same IEEE operations as `greatest(0.0, 1.0 - abs(CAST(usd_c AS DOUBLE) /
    CAST(inflow_c AS DOUBLE) - 1.0))` on a HUGEINT inflow sum, on awkward ratios."""
    rnd = random.Random(0)
    pairs = [(rnd.randrange(0, 10**12), rnd.randrange(1, 10**13)) for _ in range(3000)]
    pairs += [(rnd.randrange(1, 10**6), rnd.randrange(1, 10**6)) for _ in range(3000)]
    pairs += [(3, 7), (1, 3), (2, 3), (10**15, 3 * 10**15), (EXACT_INT_LIMIT - 1, 3), (7, 7)]
    pairs += [(EXACT_INT_LIMIT - 1, EXACT_INT_LIMIT - 1), (1, EXACT_INT_LIMIT - 1)]
    df = pl.DataFrame({"usd_c": [a for a, _ in pairs], "inflow": [b for _, b in pairs]})
    con = connect(threads=1)
    try:
        con.register("p", df)
        sql = con.execute(
            """
            WITH s AS (SELECT usd_c, CAST(inflow AS HUGEINT) AS inflow_c FROM p)
            SELECT CASE WHEN inflow_c > 0
                THEN greatest(0.0, 1.0 - abs(CAST(usd_c AS DOUBLE) / CAST(inflow_c AS DOUBLE)
                    - 1.0))
                ELSE 0.0 END AS pt
            FROM s
            """
        ).fetchnumpy()["pt"]
    finally:
        con.close()
    got = np.array([sev(rs(usd_c=a, inflow_c=b))["rapid_pass_through"] for a, b in pairs])
    assert np.array_equal(got.view(np.uint64), np.asarray(sql, np.float64).view(np.uint64))
    assert (got > 0).sum() > 1000 and (got == 0).sum() > 100  # both branches exercised


def test_round_trip_cap():
    for n, want in ((0, 0.0), (1, 1.0), (99, 99.0), (100, 100.0), (101, 100.0), (10**6, 100.0)):
        assert sev(rs(c2=n // 2, c3=n - n // 2))["round_trip"] == want
    assert severities(rs(c2=50, c3=50), {"max_round_trip_paths": 7}, NO_EXCL)[3] == 7.0


def test_values_beyond_exact_floats_raise():
    sev(rs(usd_c=EXACT_INT_LIMIT - 1, inflow_c=EXACT_INT_LIMIT - 1))
    with pytest.raises(NumericRangeError):
        severities(rs(inflow_c=EXACT_INT_LIMIT), P, NO_EXCL)
    with pytest.raises(NumericRangeError):
        severities(rs(usd_c=EXACT_INT_LIMIT, inflow_c=10), P, NO_EXCL)


# ---------------------------------------------------------------------------------------------
# Brute-force rule inputs from a frame -> severities() == the M1 SQL


def _cents(s: pl.Series) -> list[int]:
    # Half away from zero on the same double, as DuckDB's round() (M1 definition).
    return (s * 100).round(0, mode="half_away_from_zero").cast(pl.Int64).to_list()


def reference_rule_support(tx: pl.DataFrame, cfg: dict, hub_cap: int) -> list[RuleSupport]:
    """RuleSupport of every row (rank order) by literal window filters, from the §6 definitions."""
    tx = tx.sort("rank")
    P_ = sql_params(cfg, hub_cap)
    con = connect(threads=1)
    try:
        register_transactions(con, tx)
        hubs = set(hub_accounts(con, hub_cap))
    finally:
        con.close()
    minute = tx["minute"].to_list()
    src, dst = tx["src"].to_list(), tx["dst"].to_list()
    usd = tx["amount_usd"].to_list()
    fmt = tx["payment_format"].to_list()
    usd_c, paid_c = _cents(tx["amount_usd"]), _cents(tx["amount_paid"])
    in_band = [P_["band_low_usd"] <= a < P_["band_high_usd"] for a in usd]
    is_round = [c > 0 and c % P_["round_cents"] == 0 for c in paid_c]
    hr = [f in set(cfg["high_risk_formats"]) for f in fmt]
    n = tx.height
    w_rt, hop = P_["round_trip_window_plus_1"] - 1, P_["hop_window"]

    def win(i: int, w: int) -> list[int]:
        return [j for j in range(n) if minute[i] - w <= minute[j] <= minute[i] - 1]

    out = []
    for i in range(n):
        u, v = src[i], dst[i]
        fi = {src[j] for j in win(i, P_["fan_in_window"]) if dst[j] == v}
        fo = {dst[j] for j in win(i, P_["fan_out_window"]) if src[j] == u}
        inflow = sum(
            usd_c[j] for j in win(i, P_["pass_through_window"]) if dst[j] == u and src[j] != u
        )
        rt = [j for j in win(i, w_rt) if src[j] != dst[j]]
        c2 = sum(1 for j in rt if src[j] == v and dst[j] == u)
        c3 = 0
        for j in rt:
            w = dst[j]
            if src[j] != v or w in (u, v) or w in hubs:
                continue
            c3 += sum(
                1
                for k in rt
                if src[k] == w and dst[k] == u and minute[j] <= minute[k] <= minute[j] + hop
            )

        def burst(flag: list[bool], w: int, u=u, i=i) -> int:
            return sum(1 for j in win(i, w) if src[j] == u and flag[j])

        out.append(
            RuleSupport(
                u=u,
                v=v,
                self_loop=int(u == v),
                usd_c=usd_c[i],
                in_band=int(in_band[i]),
                is_round=int(is_round[i]),
                hr=int(hr[i]),
                hub_u=int(u in hubs),
                fan_in_uniq_v=len(fi),
                fan_out_uniq_u=len(fo),
                inflow_c=inflow,
                c2=c2,
                c3=c3,
                st_cnt_u=burst(in_band, P_["structuring_window"]),
                ra_cnt_u=burst(is_round, P_["round_window"]),
                hr_cnt_u=burst(hr, P_["high_risk_window"]),
            )
        )
    return out


def assert_severities_equal_sql(tx: pl.DataFrame, cfg: dict, hub_cap: int) -> pl.DataFrame:
    P_ = sql_params(cfg, hub_cap)
    excl = tuple(bool(P_[HUB_SEGMENTABLE[s]]) for s in EXCL_SCENARIOS)
    mine = [severities(r, P_, excl) for r in reference_rule_support(tx, cfg, hub_cap)]
    con = connect(threads=2)
    try:
        register_transactions(con, tx)
        sql = compute_severities(con, cfg, hub_cap)
    finally:
        con.close()
    assert sql["row_id"].to_list() == tx.sort("rank")["row_id"].to_list()
    for j, s in enumerate(SCENARIOS):
        a = np.array([row[j] for row in mine], np.float64)
        b = sql[s].to_numpy()
        diff = np.flatnonzero(a.view(np.uint64) != b.view(np.uint64))
        assert diff.size == 0, (s, [(int(i), a[i], b[i]) for i in diff[:5]])
    return sql


@pytest.fixture(scope="module")
def cfg(rules_cfg) -> dict:
    return small_cfg(rules_cfg, window=30, hop=10)


def test_band_edges_match_sql(cfg):
    amounts = [8999.99, 9000.0, 9999.99, 10000.0, 9000.004, 8999.995, 9999.995, 10000.004]
    rows = [{"minute": 100 + k, "src": 1, "dst": 2 + k % 2, "amount_usd": a} for k, a in
            enumerate(amounts)]  # fmt: skip
    sql = assert_severities_equal_sql(make_tx(rows), cfg, 10**9)
    fired = (sql.sort("row_id")["structuring"] > 0).to_list()
    # [0.9 T, T) on the float amount: 9000.0 and 9999.99 in, 8999.99 and 10000.0 out.
    assert fired[:4] == [False, True, True, False]
    assert sql.sort("row_id")["structuring"].to_list()[2] == 2.0  # 1 + the 9000.0 one minute ago


def test_round_cents_match_sql(cfg):
    paid = [100.0, 200.0, 99.995, 100.004, 100.005, 0.0, 99.99, 1e9, 5000.005, 250.0, 0.004]
    rows = [{"minute": 100 + k, "src": 1, "dst": 2, "amount_paid": a} for k, a in enumerate(paid)]
    sql = assert_severities_equal_sql(make_tx(rows), cfg, 10**9)
    assert (sql["round_amount_burst"] > 0).sum() >= 3


def test_round_trip_cap_matches_sql(rules_cfg):
    cfg = small_cfg(rules_cfg, window=1000, hop=1000)
    for n in (99, 100, 101):
        rows = [{"minute": 2000, "src": 0, "dst": 1}]
        rows += [{"minute": 1500 + k, "src": 1, "dst": 0} for k in range(n)]
        sql = assert_severities_equal_sql(make_tx(rows), cfg, 10**9)
        assert sql.filter(pl.col("row_id") == 0)["round_trip"].item() == float(min(n, 100))


def test_self_loops_zero_inflow_and_pass_through_match_sql(cfg):
    rows = [
        {"minute": 100, "src": 0, "dst": 1, "amount_usd": 100.0},
        {"minute": 95, "src": 2, "dst": 0, "amount_usd": 60.0},
        {"minute": 90, "src": 3, "dst": 0, "amount_usd": 40.0},
        {"minute": 96, "src": 0, "dst": 0, "amount_usd": 999.0},  # self-loop: not an inflow
        {"minute": 100, "src": 0, "dst": 0, "amount_usd": 100.0},  # self-loop target
        {"minute": 100, "src": 5, "dst": 6, "amount_usd": 100.0},  # no inflow
        {"minute": 100, "src": 0, "dst": 7, "amount_usd": 150.0},
        {"minute": 60, "src": 8, "dst": 8, "amount_usd": 50.0},  # only a self-loop inflow
        {"minute": 61, "src": 8, "dst": 9, "amount_usd": 50.0},
    ]
    assert_severities_equal_sql(make_tx(rows), cfg, 10**9)


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize(("window", "hop"), [(10, 5), (30, 0)])
@pytest.mark.parametrize("hub_cap", [3, 10, 10**9])
@pytest.mark.parametrize("segment", [False, True])
def test_dense_tie_frames_match_sql(rules_cfg, seed, window, hop, hub_cap, segment):
    cfg = copy.deepcopy(small_cfg(rules_cfg, window, hop))
    for s in HUB_SEGMENTABLE:
        cfg["scenarios"][s]["exclude_hub_senders"] = segment
    tx = dense_tie_frame(seed=seed, n=160)
    sql = assert_severities_equal_sql(tx, cfg, hub_cap)
    if seed == 0 and hub_cap == 10**9:  # not vacuous: every scenario fires somewhere
        for s in SCENARIOS:
            assert (sql[s] > 0).any(), s


def test_module_exports():
    assert scenarios.SCENARIOS == SCENARIOS
    assert math.isclose(EXACT_INT_LIMIT, 2.0**53)
