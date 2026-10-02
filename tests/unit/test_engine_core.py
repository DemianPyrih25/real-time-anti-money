"""The engine core (M2 spec §4, §5, §6): hand cases with literal values, invariants after every
flush, and a brute-force reference written from the spec's definitions.

The helpers at the top (`make_spec`, `Feed`, `random_stream`, `reference_rows`) are shared with
test_ports.py and test_engine_snapshot.py.
"""

from __future__ import annotations

import copy
import math
import random
import struct
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from aml.features import engine as engine_mod
from aml.features.engine import Engine, expected_feature_names
from aml.features.spec import (
    E_MINUTE,
    EXACT_INT_LIMIT,
    I32_LIMIT,
    EngineError,
    EngineSpec,
    FlushStats,
    LateEventError,
    NumericRangeError,
    RankGapError,
    Slot,
    SpecError,
    format_slug,
    tol_ok,
    window_tag,
)
from aml.features.tx_features import cents
from aml.features.windows import window_max
from aml.rules import scenarios
from aml.rules.sql_baseline import SCENARIOS

REPO_ROOT = Path(__file__).resolve().parents[2]
VOCAB = {
    "payment_format": ["ACH", "Bitcoin", "Cash", "Wire"],
    "payment_currency": ["Euro", "US Dollar"],
    "receiving_currency": ["Euro", "US Dollar"],
}
# Rule windows of the small spec (minutes); every one differs from the feature windows 4 / 9.
SMALL_RULES = {
    "fan_in_velocity": 4,
    "fan_out_velocity": 5,
    "rapid_pass_through": 3,
    "round_trip": 7,
    "structuring": 4,
    "round_amount_burst": 6,
    "high_risk_format_burst": 5,
}


def row_bits(row: tuple) -> tuple:
    """A row with every float replaced by its IEEE bytes: bit-equal rows compare equal (NaN too)."""
    return tuple(struct.pack("<d", x) if type(x) is float else x for x in row)


def _yaml(name: str) -> dict:
    with (REPO_ROOT / "configs" / name).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def make_spec(
    *,
    small: bool = False,
    n_accounts: int = 12,
    hubs: tuple[int, ...] = (),
    hub_cap: int = 1000,
    compact_min_rows: int | None = None,
    vocab: dict | None = None,
    features: dict | None = None,
    rules: dict | None = None,
) -> EngineSpec:
    """The default configs (S = 1440, L = 4320, rule windows of M1), or with small=True short
    windows S = 4, L = 9, sg = 4, the SMALL_RULES windows, round-trip hop 3, port cap 3 and gap
    cap 6. `features` / `rules` are merged over the section dicts."""
    fc = copy.deepcopy(_yaml("features.yaml"))
    rc = copy.deepcopy(_yaml("rules.yaml"))
    if small:
        fc["windows"] = {"short": 4, "long": 9, "sg": 4}
        fc["caps"] = {"count": 100, "port": 3, "gap": 6}
        for s, w in SMALL_RULES.items():
            rc["scenarios"][s]["window_minutes"] = w
        rc["scenarios"]["round_trip"]["hop_window_minutes"] = 3
    if compact_min_rows is not None:
        fc["ring"]["compact_min_rows"] = compact_min_rows
    for section, edit in (features or {}).items():
        fc[section].update(edit)
    for key, val in (rules or {}).items():
        rc[key] = val
    return EngineSpec.from_configs(
        fc, rc, n_accounts=n_accounts, vocab=vocab or VOCAB, hub_cap=hub_cap, hubs=hubs
    )


class Feed:
    """Drives one engine with compact events and returns rows as {name: value} dicts."""

    def __init__(self, spec: EngineSpec, eng: Engine | None = None) -> None:
        self.spec = spec
        self.eng = eng if eng is not None else Engine.create(spec)
        self.rank = self.eng.next_rank

    def event(self, minute: int, u: int, v: int, amount: float = 100.0, **kw) -> tuple:
        """The prepared event with the next rank (not processed)."""
        return self.eng.prepare(
            self.rank,
            1000 + self.rank,
            minute,
            u,
            v,
            amount,
            kw.get("paid", amount),
            kw.get("fmt", "ACH"),
            kw.get("pcur", "Euro"),
            kw.get("rcur", "Euro"),
            kw.get("fbank", "001"),
            kw.get("tbank", "002"),
        )

    def __call__(self, minute: int, u: int, v: int, amount: float = 100.0, **kw) -> dict:
        row = self.eng.process(self.event(minute, u, v, amount, **kw))
        self.rank += 1
        return dict(zip(self.spec.row_layout, row, strict=True))


def random_stream(spec: EngineSpec, seed: int, n: int, n_accounts: int = 6) -> list[tuple]:
    """Rank-ordered events in INPUT_COLUMNS order: dense same-minute ties, gaps around every
    window, ~20% self-loops, repeated pairs, band / round / high-risk amounts, an unknown format
    and currency."""
    rng = random.Random(seed)
    S, L = spec.w_short, spec.w_long
    steps = [0, 0, 0, 0, 1, 1, 2, S - 1, S, S + 1, L, L + 1, spec.w_rt + 2]
    amounts = [0.0, 0.004, 0.005, 0.015, 99.995, 100.0, 8999.99, 9000.0, 9999.995, 10000.0, 1e9]
    minute = rng.randrange(3)
    out = []
    for r in range(n):
        minute += rng.choice(steps)
        u = rng.randrange(n_accounts)
        v = u if rng.random() < 0.2 else rng.randrange(n_accounts)
        amt = rng.choice(amounts) if rng.random() < 0.5 else round(rng.uniform(0, 2e4), 2)
        paid = amt if rng.random() < 0.8 else rng.choice([100.0, 5000.0, 0.0])
        fmt = rng.choice(["ACH", "Cash", "Bitcoin", "Wire", "Reinvestment"])
        pcur = rng.choice(["Euro", "US Dollar"])
        rcur = rng.choice(["Euro", "US Dollar", "Yen"])
        out.append((r, 10 * r + 7, minute, u, v, amt, paid, fmt, pcur, rcur, "1", rng.choice("12")))
    return out


def run_stream(spec: EngineSpec, events: list[tuple], *, check_every: int = 0) -> tuple:
    """(rows, engine) for a stream; check_invariants() after every flush when check_every."""
    eng = Engine.create(spec)
    rows = []
    for k, f in enumerate(events):
        ev = eng.prepare(*f)
        if check_every and (eng.clock is None or ev[E_MINUTE] > eng.clock) and k % check_every == 0:
            eng.advance(ev[E_MINUTE])
            eng.check_invariants()
        rows.append(eng.process(ev))
    eng.advance(events[-1][2] + 1)
    eng.check_invariants()
    return rows, eng


# --- brute-force reference (written from M2 spec §4-§6, never from the engine code) -------------


def reference_rows(spec: EngineSpec, events: list[tuple]) -> list[dict]:
    """Every model input except cyc4 / sg_* (explicit path search is tested in test_cycles.py),
    the 7 severities and inflow_c, by literal filters over the event list.

    Each row also carries `_m2` = {(side, window): mean of l^2} for the mean / std tolerance.
    """
    E = _reference_events(spec, events)
    return [_reference_row(spec, E, j) for j in range(len(E))]


def _reference_events(spec: EngineSpec, events: list[tuple]) -> list[dict]:
    """The events with their per-event fields (cents, l, codes, flags, new-pair flag)."""
    codes = {col: {c: i for i, c in enumerate(spec.vocab[col])} for col in spec.vocab}
    E = []
    first_minute: dict[tuple[int, int], int] = {}
    for e in events:
        _, _, m, u, v, usd, paid, fmt, pcur, rcur, fb, tb = e
        pc = cents(paid)
        first_minute.setdefault((u, v), m)
        E.append(
            {
                "m": m,
                "u": u,
                "v": v,
                "c": cents(usd),
                "l": math.log1p(usd),
                "f": codes["payment_format"].get(fmt, -1),
                "band": spec.band_low_usd <= usd < spec.band_high_usd,
                "round": pc > 0 and pc % spec.round_cents == 0,
                "hr": fmt in spec.high_risk_formats,
                "pcur": codes["payment_currency"].get(pcur, -1),
                "rcur": codes["receiving_currency"].get(rcur, -1),
                "cross": pcur != rcur,
                "same_bank": fb == tb,
            }
        )
    for e in E:  # new when scored <=> its minute is the pair's first minute
        e["new"] = e["m"] == first_minute[(e["u"], e["v"])]
    return E


def _reference_row(spec: EngineSpec, E: list[dict], j: int) -> dict:
    """The reference row of target E[j]."""
    S, L, W_pt, W_rt, H = spec.w_short, spec.w_long, spec.w_pt, spec.w_rt, spec.hop
    tS, tL, tP, tR = map(window_tag, (S, L, W_pt, W_rt))
    hub = set(spec.hubs)
    ex_fo, ex_st, ex_ra, ex_hr = spec.excl
    t = E[j]
    m, u, v, c, l_t = t["m"], t["u"], t["v"], t["c"], t["l"]
    A = [e for e in E[:j] if e["m"] < m]  # applied: every earlier minute

    def win(w: int, key: str, x: int) -> list[dict]:
        return [e for e in A if e["m"] >= m - w and e[key] == x]

    def moments(es: list[dict]) -> tuple[float, float, float]:
        if not es:
            return math.nan, math.nan, math.nan
        n = len(es)
        mean = math.fsum(e["l"] for e in es) / n
        var = math.fsum((e["l"] - mean) ** 2 for e in es) / n
        return mean, math.sqrt(var), math.fsum(e["l"] ** 2 for e in es) / n

    def gap(es: list[dict]) -> int:
        return min(m - max(e["m"] for e in es), spec.cap_gap) if es else 0

    def lsum(es: list[dict]) -> float:
        return math.log1p(sum(e["c"] for e in es) / 100)

    r: dict = {"_m2": {}}
    r.update(
        log_amount_usd=l_t,
        payment_currency=t["pcur"],
        receiving_currency=t["rcur"],
        cross_currency=int(t["cross"]),
        payment_format=t["f"],
        self_loop=int(u == v),
        same_bank=int(t["same_bank"]),
        round_amount=int(t["round"]),
        hour_of_day=(m % 1440) // 60,
    )
    sides = {
        ("u", "out"): ("u", u),
        ("u", "in"): ("v", u),
        ("v", "in"): ("v", v),
        ("v", "out"): ("u", v),
    }
    for (side, d), (key, x) in sides.items():
        for w, tag in ((S, tS), (L, tL)):
            es = win(w, key, x)
            other = "v" if key == "u" else "u"
            r[f"{side}_{d}_cnt_{tag}"] = len(es)
            r[f"{side}_{d}_uniq_{tag}"] = len({e[other] for e in es})
            r[f"{side}_{d}_sum_{tag}"] = lsum(es)
            if (side, d) in (("u", "out"), ("v", "in")):
                mean, std, m2 = moments(es)
                r[f"{side}_{d}_mean_{tag}"], r[f"{side}_{d}_std_{tag}"] = mean, std
                r["_m2"][(side, d, w)] = m2
    uo_s, vi_s = win(S, "u", u), win(S, "v", v)
    r[f"u_out_max_{tS}"] = max((e["l"] for e in uo_s), default=math.nan)
    r[f"v_in_max_{tS}"] = max((e["l"] for e in vi_s), default=math.nan)
    r[f"u_amt_dev_{tS}"] = l_t - r[f"u_out_mean_{tS}"]
    r[f"v_amt_dev_{tS}"] = l_t - r[f"v_in_mean_{tS}"]
    # FLOW
    pair = [e for e in A if (e["u"], e["v"]) == (u, v)]
    rev = [e for e in A if (e["u"], e["v"]) == (v, u)]
    r[f"pair_cnt_{tS}"] = sum(e["m"] >= m - S for e in pair)
    r[f"pair_cnt_{tL}"] = sum(e["m"] >= m - L for e in pair)
    inflow = sum(e["c"] for e in win(W_pt, "v", u) if e["u"] != e["v"])
    r[f"u_inflow_{tP}"] = math.log1p(inflow / 100)
    r[f"pt_ratio_{tP}"] = float(c) / float(inflow) if u != v and inflow > 0 else math.nan
    for side, x in (("u", u), ("v", v)):
        o = sum(e["c"] for e in win(L, "u", x))
        i = sum(e["c"] for e in win(L, "v", x))
        r[f"{side}_bal_{tL}"] = (o - i) / (o + i) if o + i else math.nan
    # PORT: ports are counted at the pair's first minute (or now, for a new pair)
    f0 = min((e["m"] for e in pair), default=m)
    r["pair_is_new"] = int(not pair)
    port_out = len({e["v"] for e in A if e["u"] == u and e["m"] < f0})
    port_in = len({e["u"] for e in A if e["v"] == v and e["m"] < f0})
    r["out_port"] = math.log1p(min(port_out, spec.cap_port))
    r["in_port"] = math.log1p(min(port_in, spec.cap_port))
    r["u_out_gap"] = gap([e for e in A if e["u"] == u])
    r["u_in_gap"] = gap([e for e in A if e["v"] == u])
    r["v_in_gap"] = gap([e for e in A if e["v"] == v])
    r["v_out_gap"] = gap([e for e in A if e["u"] == v])
    r["pair_gap"], r["rev_pair_gap"] = gap(pair), gap(rev)
    # CYC: c2, c3 (the round-trip paths); cyc4 and SG are not recomputed here
    lo = m - W_rt
    c2 = c3 = 0
    if u != v:
        c2 = sum(e["m"] >= lo for e in rev)
        for e1 in A:  # v -> w at t1, then w -> u at t2
            if e1["u"] != v or e1["m"] < lo or e1["v"] in (u, v) or e1["v"] in hub:
                continue
            w = e1["v"]
            c3 += sum(
                1 for e2 in A if e2["u"] == w and e2["v"] == u and e1["m"] <= e2["m"] <= e1["m"] + H
            )
    r[f"cyc2_{tR}"] = min(c2, spec.cap_count)
    r[f"cyc3_{tR}"] = min(c3, spec.cap_count)
    # SG: the gather-scatter mins only
    r[f"gs_u_{tS}"] = min(r[f"u_in_uniq_{tS}"], r[f"u_out_uniq_{tS}"])
    r[f"gs_v_{tS}"] = min(r[f"v_in_uniq_{tS}"], r[f"v_out_uniq_{tS}"])
    # RULE
    r["in_band"] = int(t["band"])
    r[f"u_out_inband_{tS}"] = sum(e["band"] for e in uo_s)
    r[f"u_out_round_{tS}"] = sum(e["round"] for e in uo_s)
    r[f"u_out_newcp_{tS}"] = sum(e["new"] for e in uo_s)
    for k, fm in enumerate(spec.vocab["payment_format"]):
        r[f"u_out_fmt_{format_slug(fm)}_{tS}"] = sum(e["f"] == k for e in uo_s)
    r[f"v_in_same_fmt_{tS}"] = sum(e["f"] == t["f"] for e in vi_s) if t["f"] >= 0 else 0
    # severities (§6) at each scenario's own window
    hub_u = u in hub

    def burst(flag: str, w: int, excluded: bool) -> float:
        if not t[flag] or excluded:
            return 0.0
        return float(1 + sum(e[flag] for e in win(w, "u", u)))

    pt = 0.0
    if u != v and inflow > 0:
        pt = max(0.0, 1.0 - abs(float(c) / float(inflow) - 1.0))
    r["severities"] = (
        float(len({e["u"] for e in win(spec.w_fan_in, "v", v)})),
        0.0 if ex_fo and hub_u else float(len({e["v"] for e in win(spec.w_fan_out, "u", u)})),
        pt,
        0.0 if u == v else float(min(c2 + c3, spec.sql_params["max_round_trip_paths"])),
        burst("band", spec.w_struct, ex_st and hub_u),
        burst("round", spec.w_round, ex_ra and hub_u),
        burst("hr", spec.w_hr, ex_hr and hub_u),
    )
    r["inflow_c"] = inflow
    return r


def assert_rows_match_reference(spec: EngineSpec, rows: list[tuple], ref: list[dict]) -> int:
    """Compare engine rows with `reference_rows` under the tolerance classes; returns the number
    of values compared."""
    skip = {f"cyc4_{window_tag(spec.w_rt)}", *(f"sg_{k}_{window_tag(spec.w_sg)}"
                                               for k in ("mids", "srcs"))}  # fmt: skip
    n = 0
    m2_key = {}
    for f in spec.features:
        if f.tol in ("mean", "std"):
            d = "out" if f.name.startswith("u_") else "in"
            m2_key[f.name] = ("u" if d == "out" else "v", d, f.window)
    for k, (row, want) in enumerate(zip(rows, ref, strict=True)):
        assert row[spec.i_rule_trunc] == 0 and row[spec.i_cyc_trunc] == 0
        for i, f in enumerate(spec.features):
            if f.name in skip:
                continue
            m2 = want["_m2"].get(m2_key.get(f.name), 0.0)
            ok = tol_ok(f.tol, row[i], want[f.name], m2 if f.tol in ("mean", "std") else None)
            assert ok.all(), f"row {k} {f.name}: engine {row[i]!r} != reference {want[f.name]!r}"
            n += 1
        sev = tuple(row[spec.i_sev : spec.i_sev + len(SCENARIOS)])
        assert sev == want["severities"], f"row {k} severities {sev} != {want['severities']}"
        assert row[spec.i_inflow] == want["inflow_c"], f"row {k} inflow_c"
        n += len(SCENARIOS) + 1
    return n


# --- spec and layout ------------------------------------------------------------------------------


def test_layout_matches_the_spec_and_row_types():
    spec = make_spec()
    assert expected_feature_names(spec) == spec.feature_names
    feed = Feed(spec)
    row = feed.eng.process(feed.event(10, 1, 2))
    assert len(row) == spec.row_len == spec.n_features + 11
    assert all(type(x) is float for x in row[: spec.i_inflow])  # features and severities
    assert all(type(x) is int for x in row[spec.i_inflow :])  # inflow_c and the trunc flags


def test_engine_refuses_a_spec_whose_layout_it_does_not_know():
    spec = make_spec()
    odd = copy.copy(spec)
    object.__setattr__(odd, "feature_names", tuple(reversed(spec.feature_names)))
    with pytest.raises(SpecError, match="feature layout"):
        Engine.create(odd)


# --- hand cases with literal values (default configs: S = 1440, L = 4320) ------------------------


def test_window_bounds_minute_m_minus_w_counts_and_one_earlier_does_not():
    feed = Feed(make_spec())
    feed(100, 1, 2)
    r = feed(100 + 1440, 1, 3)  # the minute-100 event is at m - S: inside
    assert r["u_out_cnt_1d"] == 1.0 and r["u_out_cnt_3d"] == 1.0 and r["v_in_cnt_1d"] == 0.0
    r = feed(100 + 1441, 1, 4)  # m - S - 1: outside S, inside L
    assert r["u_out_cnt_1d"] == 1.0  # only the minute-1540 event
    assert r["u_out_cnt_3d"] == 2.0
    r = feed(100 + 4320, 5, 2)  # v = 2: its minute-100 in-event is at m - L
    assert r["v_in_cnt_3d"] == 1.0 and r["v_in_cnt_1d"] == 0.0
    r = feed(100 + 4321, 6, 2)
    assert r["v_in_cnt_3d"] == 1.0  # the minute-4420 event of 5 -> 2 only
    feed.eng.check_invariants()


def test_same_minute_peers_are_invisible_in_both_rank_directions():
    feed = Feed(make_spec())
    first = feed.event(10, 1, 2, 500.0)
    r0 = dict(zip(feed.spec.row_layout, feed.eng.process(first), strict=True))
    feed.rank += 1
    r1 = feed(10, 1, 3)
    r2 = feed(10, 2, 1)
    for r in (r0, r1, r2):
        assert r["u_out_cnt_1d"] == r["u_in_cnt_1d"] == r["v_in_cnt_1d"] == 0.0
        assert r["pair_is_new"] == 1.0 and r["rev_pair_gap"] == 0.0 and r["u_out_gap"] == 0.0
        assert math.isnan(r["u_out_max_1d"])
    # the earlier-ranked event, rescored after its peers were queued, still sees none of them
    assert row_bits(feed.eng.score(first)) == row_bits(tuple(r0.values()))
    r = feed(11, 1, 2)
    assert r["u_out_cnt_1d"] == 2.0 and r["u_in_cnt_1d"] == 1.0 and r["pair_cnt_1d"] == 1.0
    assert r["pair_is_new"] == 0.0 and r["pair_gap"] == 1.0 and r["rev_pair_gap"] == 1.0
    assert r["u_out_max_1d"] == math.log1p(500.0)


def test_a_gap_beyond_every_window_right_after_a_pending_minute():
    feed = Feed(make_spec())
    feed(10, 1, 2, 300.0)
    stats = feed.eng.advance(10 + 4321 + 7)  # applies minute 10, then expires it everywhere
    assert stats.n_applied == 1 and stats.n_expired == len(feed.spec.windows_all)
    r = feed(10 + 4321 + 7, 1, 2)
    for name in ("u_out_cnt_1d", "u_out_cnt_3d", "v_in_cnt_3d", "pair_cnt_1d", "pair_cnt_3d"):
        assert r[name] == 0.0
    assert math.isnan(r["u_out_mean_3d"]) and math.isnan(r["u_out_max_1d"])
    assert r["u_out_sum_3d"] == 0.0 and math.isnan(r["u_bal_3d"])
    # lifetime state survives expiry
    assert r["pair_is_new"] == 0.0 and r["u_out_gap"] == 4320.0 and r["pair_gap"] == 4320.0
    feed.eng.check_invariants()


def test_self_loops_count_as_counterparties_but_not_as_inflow_or_paths():
    feed = Feed(make_spec())
    feed(10, 1, 1, 500.0)  # self-loop
    feed(10, 3, 1, 200.0)
    r = feed(20, 1, 4, 50.0)
    assert r["u_in_cnt_1d"] == 2.0 and r["u_in_uniq_1d"] == 2.0  # senders 1 and 3
    assert r["u_out_cnt_1d"] == 1.0 and r["u_out_uniq_1d"] == 1.0  # receiver 1 (itself)
    assert r["inflow_c"] == 20000 and r["u_inflow_12h"] == math.log1p(200.0)
    assert r["pt_ratio_12h"] == 5000 / 20000
    assert r["rapid_pass_through"] == 1.0 - abs(5000 / 20000 - 1.0)
    r = feed(20, 1, 1, 50.0)  # a self-loop target
    assert r["self_loop"] == 1.0 and math.isnan(r["pt_ratio_12h"])
    assert r["rapid_pass_through"] == r["round_trip"] == 0.0
    assert r["cyc2_2d"] == r["cyc3_2d"] == r["cyc4_2d"] == r["sg_mids_1d"] == 0.0
    assert r["rev_pair_gap"] == r["pair_gap"] == 10.0  # (1, 1) is its own reverse pair


def test_a_new_pair_twice_in_one_minute_counts_once_and_shares_its_ports():
    feed = Feed(make_spec())
    feed(1, 1, 9)  # u = 1 has one receiver before minute 5
    rows = [feed(5, 1, 2), feed(5, 1, 3), feed(5, 1, 2)]
    for r in rows:
        assert r["pair_is_new"] == 1.0 and r["out_port"] == math.log1p(1) and r["in_port"] == 0.0
    r = feed(6, 1, 2)
    assert r["pair_is_new"] == 0.0 and r["pair_cnt_1d"] == 2.0
    assert r["out_port"] == math.log1p(1) and r["in_port"] == 0.0  # stored at the first minute
    assert r["u_out_uniq_1d"] == 3.0 and r["u_out_cnt_1d"] == 4.0
    assert r["u_out_newcp_1d"] == 4.0  # 1->9, then all three minute-5 events were new
    r = feed(6, 1, 4)  # a later new receiver: port = receivers before minute 6
    assert r["out_port"] == math.log1p(3)
    feed.eng.advance(7)
    feed.eng.check_invariants()
    assert feed.eng.ever_out[1] == 4 and feed.eng.pairs.n_pairs == 4


def test_window_max_after_its_maximum_expires_and_with_ties():
    feed = Feed(make_spec())
    for minute, amount in ((0, 1000.0), (100, 50.0), (200, 1000.0), (300, 10.0)):
        feed(minute, 1, 2, amount)
    r = feed(1441, 1, 5)  # window [1, 1440]: the minute-0 maximum expired, its tie at 200 stays
    assert r["u_out_max_1d"] == math.log1p(1000.0) and math.isnan(r["v_in_max_1d"])
    r = feed(1641, 1, 5)  # window [201, 1640]: 10.0 at 300 and the 100.0 of the minute-1441 row
    assert r["u_out_max_1d"] == math.log1p(100.0)
    r = feed(1641, 7, 2)  # in side of v = 2: 1000.0 at 200 is outside, 10.0 at 300 inside
    assert r["v_in_max_1d"] == math.log1p(10.0)
    feed.eng.advance(1642)
    feed.eng.check_invariants()


def test_gaps_are_capped_and_zero_means_no_history():
    feed = Feed(make_spec())
    r = feed(3, 1, 2)
    assert (r["u_out_gap"], r["u_in_gap"], r["v_in_gap"], r["v_out_gap"]) == (0.0,) * 4
    r = feed(8, 2, 1)
    assert r["u_out_gap"] == 0.0 and r["u_in_gap"] == 5.0 and r["v_out_gap"] == 5.0
    assert r["v_in_gap"] == 0.0 and r["rev_pair_gap"] == 5.0 and r["pair_gap"] == 0.0
    r = feed(8 + 9000, 1, 2)
    assert r["u_out_gap"] == r["u_in_gap"] == r["pair_gap"] == r["rev_pair_gap"] == 4320.0


def test_moments_reset_to_exact_zero_when_a_window_empties():
    spec = make_spec()
    feed = Feed(spec)
    for amount in (0.1, 0.2, 0.3, 7.77):  # their log sums do not cancel exactly
        feed(0, 1, 2, amount)
    feed(1, 3, 4)
    eng = feed.eng
    eng.advance(1 + 1440)
    s1 = eng.slots.get("out", "s1", spec.w_short)
    s2 = eng.slots.get("out", "s2", spec.w_short)
    assert s1[1] == 0.0 and s2[1] == 0.0 and eng.slots.get("out", "cnt", spec.w_short)[1] == 0
    assert eng.slots.get("out", "s1", spec.w_long)[1] != 0.0  # still inside L
    r = feed(1 + 1440, 1, 9, 42.0)
    assert math.isnan(r["u_out_mean_1d"]) and math.isnan(r["u_out_std_1d"])
    r = feed(2 + 1440, 1, 9, 5.0)
    assert r["u_out_mean_1d"] == math.log1p(42.0) and r["u_out_std_1d"] == 0.0
    assert r["u_amt_dev_1d"] == math.log1p(5.0) - math.log1p(42.0)


def test_amount_flow_features_literal_values():
    feed = Feed(make_spec())
    feed(0, 1, 2, 300.0)  # 1 out 30000
    feed(0, 3, 1, 100.0)  # 1 in 10000
    feed(0, 2, 2, 50.0)  # self-loop of 2: out and in
    r = feed(1, 1, 2, 25.0)
    assert r["u_out_sum_1d"] == math.log1p(300.0) and r["u_in_sum_1d"] == math.log1p(100.0)
    assert r["u_bal_3d"] == (30000 - 10000) / 40000
    assert r["v_bal_3d"] == (5000 - (30000 + 5000)) / (5000 + 35000)
    assert r["v_in_sum_1d"] == math.log1p(350.0) and r["v_out_sum_1d"] == math.log1p(50.0)
    assert r["v_in_mean_1d"] == (math.log1p(300.0) + math.log1p(50.0)) / 2
    assert r["pt_ratio_12h"] == 2500 / 10000 and r["u_inflow_12h"] == math.log1p(100.0)
    assert r["gs_u_1d"] == 1.0 and r["gs_v_1d"] == 1.0  # min(in uniq, out uniq)


def test_format_counts_and_unknown_codes():
    feed = Feed(make_spec())
    feed(0, 1, 2, 123.45, fmt="Cash")  # not round amounts
    feed(0, 1, 3, 123.45, fmt="Cash")
    feed(0, 1, 2, 123.45, fmt="Cheque")  # not in the vocab: code -1, counts nothing
    feed(0, 4, 2, 123.45, fmt="Wire")
    r = feed(1, 1, 2, fmt="Wire", pcur="Yen", rcur="Euro")
    assert r["u_out_fmt_cash_1d"] == 2.0 and r["u_out_fmt_wire_1d"] == 0.0
    assert r["u_out_fmt_ach_1d"] == r["u_out_fmt_bitcoin_1d"] == 0.0 and r["u_out_cnt_1d"] == 3.0
    assert r["v_in_same_fmt_1d"] == 1.0 and r["payment_format"] == 3.0
    assert r["payment_currency"] == -1.0 and r["receiving_currency"] == 0.0
    assert r["cross_currency"] == 1.0
    r = feed(1, 5, 2, fmt="Cheque")
    assert r["payment_format"] == -1.0 and r["v_in_same_fmt_1d"] == 0.0
    r = feed(1, 1, 6, 9500.0, paid=4200.0, fmt="Cash", fbank="7", tbank="7")
    assert r["in_band"] == 1.0 and r["round_amount"] == 1.0 and r["same_bank"] == 1.0
    assert r["structuring"] == 1.0 and r["round_amount_burst"] == 1.0  # no earlier ones
    assert r["high_risk_format_burst"] == 3.0  # 1 + the two earlier Cash payments of u
    assert r["hour_of_day"] == 0.0 and feed(1500, 1, 6)["hour_of_day"] == 1.0


# --- the protocol ---------------------------------------------------------------------------------


def test_late_events_and_rank_gaps_raise_and_change_nothing():
    feed = Feed(make_spec())
    feed(5, 1, 2)
    feed(7, 2, 3)
    eng = feed.eng
    before = (eng.state_digest(), eng.pending_count, eng.next_rank, eng.clock)
    with pytest.raises(LateEventError):
        eng.process(feed.event(6, 1, 2))
    with pytest.raises(RankGapError):
        eng.process(eng.prepare(feed.rank + 1, 0, 8, 1, 2, 1.0, 1.0, "ACH", "Euro", "Euro", 1, 1))
    with pytest.raises(RankGapError):  # a rank gap is refused before the minute flush
        eng.process(eng.prepare(feed.rank + 1, 0, 9, 1, 2, 1.0, 1.0, "ACH", "Euro", "Euro", 1, 1))
    with pytest.raises(LateEventError):
        eng.advance(6)
    with pytest.raises(LateEventError):
        eng.score(feed.event(6, 1, 2))
    with pytest.raises(EngineError, match="advance first"):
        eng.score(feed.event(8, 1, 2))
    assert (eng.state_digest(), eng.pending_count, eng.next_rank, eng.clock) == before
    assert isinstance(feed(7, 3, 4), dict)  # the engine carries on


def test_advance_to_the_clock_is_a_no_op_and_split_advances_equal_one():
    events = random_stream(make_spec(small=True), seed=3, n=120)
    spec = make_spec(small=True)
    rows_a, eng_a = run_stream(spec, events)
    eng_b = Engine.create(spec)
    rows_b = []
    for f in events:
        ev = eng_b.prepare(*f)
        m = ev[E_MINUTE]
        if eng_b.clock is not None and m > eng_b.clock + 2:
            mid = (eng_b.clock + m) // 2 + 1  # a minute with no events (clock < mid < m)
            assert eng_b.advance(mid).n_applied >= 0
            digest = eng_b.state_digest()
            assert eng_b.advance(mid) == FlushStats(0, 0, 0.0)
            assert eng_b.state_digest() == digest
        rows_b.append(eng_b.process(ev))
    eng_b.advance(events[-1][2] + 1)
    assert [row_bits(r) for r in rows_a] == [row_bits(r) for r in rows_b]
    assert eng_a.state_digest() == eng_b.state_digest()


def test_score_is_read_only_and_equals_process():
    spec = make_spec(small=True)
    events = random_stream(spec, seed=5, n=150)
    eng = Engine.create(spec)
    for f in events:
        ev = eng.prepare(*f)
        if eng.clock is not None and ev[E_MINUTE] == eng.clock:
            digest = eng.state_digest()
            scored = eng.score(ev)
            assert eng.state_digest() == digest
            assert row_bits(eng.process(ev)) == row_bits(scored)
        else:
            eng.process(ev)


def test_flush_stats_and_pending_count():
    feed = Feed(make_spec(small=True))
    feed(0, 1, 2)
    feed(0, 2, 3)
    assert feed.eng.pending_count == 2
    stats = feed.eng.advance(2)
    assert isinstance(stats, FlushStats) and stats.n_applied == 2 and stats.n_expired == 0
    assert stats.seconds >= 0.0 and feed.eng.pending_count == 0 and feed.eng.last_flush == stats
    stats = feed.eng.advance(5)  # minute 0 leaves the windows 3 (pass-through) and 4 (short)
    assert stats.n_applied == 0 and stats.n_expired == 2 * 2


# --- compaction, guards ---------------------------------------------------------------------------


def test_walks_right_after_compaction_match_an_uncompacted_twin():
    events = random_stream(make_spec(small=True), seed=11, n=300)
    compacting = make_spec(small=True, compact_min_rows=1)
    plain = make_spec(small=True)
    eng_c, eng_p = Engine.create(compacting), Engine.create(plain)
    compactions = 0
    for f in events:
        base = eng_c.ring.base
        row_c = eng_c.process(eng_c.prepare(*f))
        row_p = eng_p.process(eng_p.prepare(*f))
        assert row_bits(row_c) == row_bits(row_p)
        if eng_c.ring.base != base:
            compactions += 1
            eng_c.check_invariants()
    assert compactions > 10 and eng_c.ring.base > eng_p.ring.base
    # walks on the compacted ring: window max and neighbours agree with the twin
    for a in range(6):
        lo = eng_c.clock - compacting.w_short
        x = window_max(eng_c.ring, eng_c.head_out, eng_c.ring.pmax_out, a, lo)
        y = window_max(eng_p.ring, eng_p.head_out, eng_p.ring.pmax_out, a, lo)
        assert x == y or (math.isnan(x) and math.isnan(y))
        assert eng_c.neighbours(a, "in", compacting.W_max, 50) == eng_p.neighbours(
            a, "in", plain.W_max, 50
        )


def test_counter_overflow_and_cents_range_guards():
    spec = make_spec()
    feed = Feed(spec)
    cnt = feed.eng.slots.get("out", "cnt", spec.w_short)
    cnt[1] = I32_LIMIT  # 2^31 - 1: the next += 1 cannot be stored in array('i')
    feed(0, 1, 2)
    with pytest.raises(OverflowError):
        feed.eng.advance(1)
    feed = Feed(spec)
    feed(0, 1, 2)
    feed.eng.advance(1)
    feed.eng.slots.get("in", "sum_c", spec.w_long)[2] = EXACT_INT_LIMIT
    with pytest.raises(NumericRangeError):
        feed(1, 2, 3)
    with pytest.raises(NumericRangeError):
        feed.event(1, 2, 3, 1e14)  # 10^16 cents >= 2^53


@pytest.mark.parametrize(
    "args, err",
    [
        ((0, 0, 0, 1, 2, -0.01, 1.0), ValueError),
        ((0, 0, 0, 1, 2, math.nan, 1.0), ValueError),
        ((0, 0, 0, 1, 2, 1.0, math.inf), ValueError),
        ((0, 0, 0, 1, 12, 1.0, 1.0), ValueError),  # account id >= n_accounts
        ((0, 0, 0, -1, 2, 1.0, 1.0), ValueError),
        ((I32_LIMIT, 0, 0, 1, 2, 1.0, 1.0), ValueError),
        ((0, 0, I32_LIMIT, 1, 2, 1.0, 1.0), ValueError),
        ((0, 0, -5, 1, 2, 1.0, 1.0), ValueError),
        ((0, 0, 1.5, 1, 2, 1.0, 1.0), ValueError),  # not an integer
    ],
)
def test_prepare_validates_its_inputs(args, err):
    eng = Engine.create(make_spec())
    with pytest.raises(err):
        eng.prepare(*args, "ACH", "Euro", "Euro", "1", "2")


def test_prepare_flags_codes_and_numpy_inputs():
    spec = make_spec()
    eng = Engine.create(spec)
    ev = eng.prepare(np.int64(3), 9, np.int32(61), 4, 4, np.float64(9000.0), 4200.004, "Cash",
                     "Euro", "Yen", "012", "12")  # fmt: skip
    rank, row_id, minute, u, v, usd_c, l_x, usd, paid_c, fmt, pcur, rcur, flags, hour = ev
    assert (rank, row_id, minute, u, v) == (3, 9, 61, 4, 4) and type(rank) is int
    assert usd_c == 900000 and paid_c == 420000 and l_x == math.log1p(9000.0) and usd == 9000.0
    assert (fmt, pcur, rcur, hour) == (2, 0, -1, 1)
    from aml.features.spec import F_CROSS, F_HR, F_IN_BAND, F_ROUND, F_SAME_BANK, F_SELF

    assert flags == F_SELF | F_IN_BAND | F_ROUND | F_HR | F_CROSS  # "012" != "12"
    assert not flags & F_SAME_BANK


# --- invariants and the brute-force reference -----------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("compact", [None, 1])
def test_engine_equals_the_brute_force_reference(seed, compact):
    spec = make_spec(small=True, hubs=(5,), hub_cap=3, compact_min_rows=compact)
    events = random_stream(spec, seed=seed, n=220)
    rows, eng = run_stream(spec, events, check_every=1)
    assert assert_rows_match_reference(spec, rows, reference_rows(spec, events)) > 220 * 70
    assert sum(r[spec.feature_index["pair_is_new"]] == 0.0 for r in rows) > 50  # repeats exist


def test_hub_segmentation_and_rule_windows_differ_from_feature_windows():
    rules = copy.deepcopy(_yaml("rules.yaml"))["scenarios"]
    for s, w in SMALL_RULES.items():
        rules[s]["window_minutes"] = w
    rules["round_trip"]["hop_window_minutes"] = 3
    for s in ("structuring", "round_amount_burst"):
        rules[s]["exclude_hub_senders"] = True
    spec = make_spec(small=True, hubs=(0, 5), hub_cap=3, rules={"scenarios": rules})
    assert spec.excl == (True, True, True, True)
    events = random_stream(spec, seed=21, n=200)
    rows, _ = run_stream(spec, events)
    assert_rows_match_reference(spec, rows, reference_rows(spec, events))
    hub_rows = [r for r, f in zip(rows, events, strict=True) if f[3] in (0, 5)]
    assert hub_rows and all(r[spec.i_sev + 1] == 0.0 for r in hub_rows)  # fan-out segmented


def test_rule_support_is_built_at_each_scenarios_own_window(monkeypatch):
    spec = make_spec(small=True)
    seen = []
    real = scenarios.severities

    def recording(rs, P, excl):
        seen.append(scenarios.RuleSupport(*rs))
        return real(rs, P, excl)

    monkeypatch.setattr(scenarios, "severities", recording)
    events = random_stream(spec, seed=8, n=150)
    rows, _ = run_stream(spec, events)
    ref = reference_rows(spec, events)
    assert len(seen) == len(rows)
    for rs, f, want in zip(seen, events, ref, strict=True):
        assert (rs.u, rs.v, rs.self_loop) == (f[3], f[4], int(f[3] == f[4]))
        assert rs.usd_c == cents(f[5]) and rs.inflow_c == want["inflow_c"]
        assert float(rs.fan_in_uniq_v) == want["severities"][0]


def test_check_invariants_detects_a_corrupted_aggregate():
    spec = make_spec(small=True)
    _, eng = run_stream(spec, random_stream(spec, seed=4, n=60))
    eng.slots.get("in", "cnt", spec.w_long)[3] += 1
    with pytest.raises(AssertionError, match="slot.in.cnt"):
        eng.check_invariants()


def test_slot_kernels_are_generated_from_the_registry():
    spec = make_spec(small=True)
    eng = Engine.create(spec)
    windows = dict(eng._expirers)
    assert set(windows) == set(spec.windows_all)
    src = windows[spec.w_short].__aml_source__
    assert "-= x * x" in src and "= 0.0" in src  # moments, reset at count 0
    assert not hasattr(windows[spec.w_rt], "__aml_source__")  # no slots: the cursor only moves
    assert Slot("in", "nsl_sum_c", spec.w_pt) in eng.slots.arrays


# --- real rules on the prepared fixture -----------------------------------------------------------


def fixture_spec(prepared, rules_cfg) -> tuple[EngineSpec, pl.DataFrame]:
    """The default-config spec of the prepared fixture and its transactions in rank order."""
    from aml.features.tx_features import fit_vocab
    from aml.rules.sql_baseline import connect, hub_accounts, hub_degree_cap, register_transactions

    tx = pl.read_parquet(prepared.transactions).sort("rank")
    con = connect(threads=1)
    try:
        register_transactions(con, prepared.transactions)
        cap = hub_degree_cap(con, rules_cfg["hub_degree_quantile"])
        hubs = hub_accounts(con, cap)
    finally:
        con.close()
    fc = _yaml("features.yaml")
    n = pl.read_parquet(prepared.accounts).height
    vocab = fit_vocab(tx.filter(pl.col("split") == "train"))
    spec = EngineSpec.from_configs(fc, rules_cfg, n_accounts=n, vocab=vocab, hub_cap=cap, hubs=hubs)
    return spec, tx


def fixture_rows(spec: EngineSpec, tx: pl.DataFrame) -> list[tuple]:
    from aml.features.spec import INPUT_COLUMNS

    eng = Engine.create(spec)
    cols = [tx[c].to_list() for c in INPUT_COLUMNS]
    return [eng.process(eng.prepare(*f)) for f in zip(*cols, strict=True)]


def test_engine_severities_equal_the_m1_sql_on_the_fixture(prepared, rules_cfg):
    from aml.rules.sql_baseline import compute_severities, connect, register_transactions

    spec, tx = fixture_spec(prepared, rules_cfg)
    rows = fixture_rows(spec, tx)
    con = connect(threads=1)
    try:
        register_transactions(con, prepared.transactions)
        sql = compute_severities(con, rules_cfg, spec.hub_cap)
    finally:
        con.close()
    assert sql["row_id"].to_list() == tx["row_id"].to_list()
    got = np.array([r[spec.i_sev : spec.i_sev + len(SCENARIOS)] for r in rows])
    for j, s in enumerate(SCENARIOS):
        want = sql[s].to_numpy()
        bad = np.flatnonzero(got[:, j] != want)
        assert bad.size == 0, f"{s}: {bad.size} mismatches, first row {bad[:1]}"
    assert (got > 0).any(axis=0).all()  # every scenario fires somewhere: not vacuous
    assert all(r[spec.i_rule_trunc] == 0 for r in rows)


def test_engine_is_deterministic_across_engines():
    spec = make_spec(small=True)
    events = random_stream(spec, seed=13, n=150)
    rows_a, eng_a = run_stream(spec, events)
    rows_b, eng_b = run_stream(spec, events)
    assert [row_bits(r) for r in rows_a] == [row_bits(r) for r in rows_b]
    assert eng_a.state_digest() == eng_b.state_digest()


def test_module_reads_cycles_and_scenarios_per_call(monkeypatch):
    spec = make_spec(small=True)
    eng = Engine.create(spec)
    monkeypatch.setattr(engine_mod._cycles, "path_counts", lambda *a: (7, 8, 9, 0, 1))
    monkeypatch.setattr(engine_mod._cycles, "sg_counts", lambda *a: (2, 3, 1))
    feed = Feed(spec, eng)
    r = feed(0, 1, 2)
    assert (r["cyc2_7m"], r["cyc3_7m"], r["cyc4_7m"], r["sg_mids_4m"], r["sg_srcs_4m"]) == (
        7.0,
        8.0,
        9.0,
        2.0,
        3.0,
    )
    assert (r["rule_trunc"], r["cyc_trunc"], r["sg_trunc"]) == (0, 1, 1)
    assert r["round_trip"] == 15.0
    monkeypatch.setattr(engine_mod._cycles, "path_counts", lambda *a: (700, 0, 0, 0, 0))
    assert feed(0, 1, 3)["cyc2_7m"] == 100.0  # capped at caps.count
