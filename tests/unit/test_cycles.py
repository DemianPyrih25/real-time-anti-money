"""Path search (M2 spec §4.7): round trip / cycles 2-4, scatter-gather, budgets, neighbours.

Most tests walk `MiniRing`, a `spec.RingView` built directly from an event list, and compare with a
brute-force enumeration written from the spec's definitions. The last section replays M1's
round-trip test scenarios through the engine (`Engine`) and the M1 SQL.
"""

from __future__ import annotations

import copy
import random
from array import array
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from aml.features import cycles
from aml.features.cycles import MinuteMemo, PathMemo
from aml.features.spec import F_SELF, RING_COLUMNS, EngineSpec, window_tag
from aml.rules.sql_baseline import (
    MAX_ROUND_TRIP_PATHS,
    SCENARIOS,
    compute_severities,
    connect,
    hub_accounts,
    register_transactions,
)
from tests.fixtures.rules_frames import make_tx, small_cfg

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _yaml(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------------------------
# A RingView over a fixed list of applied events, and the spec built for a test


class MiniRing:
    """`spec.RingView` over applied events (minute, src, dst[, usd_c]) in rank order.

    `drop` simulates a compaction: the first `drop` rows are deleted and `base` moves to it, so
    chain pointers into them point below `base`.
    """

    def __init__(self, events: list[tuple], n_accounts: int, drop: int = 0) -> None:
        cols = {name: array(tc) for name, tc in RING_COLUMNS}
        self.head_in = array("i", [-1] * n_accounts)
        self.head_out = array("i", [-1] * n_accounts)
        last = -1
        for r, ev in enumerate(events):
            m, s, d = ev[:3]
            assert m >= last, "events must be in minute order"
            last = m
            row = {
                "minute": m,
                "src": s,
                "dst": d,
                "usd_c": ev[3] if len(ev) > 3 else 100 * (r + 1),
                "l": 0.0,
                "fmt": 0,
                "flags": F_SELF if s == d else 0,
                "pid": 0,
                "prev_out": self.head_out[s],
                "prev_in": self.head_in[d],
                "pmax_out": -1,
                "pmax_in": -1,
            }
            for name, _ in RING_COLUMNS:
                cols[name].append(row[name])
            self.head_out[s] = r
            self.head_in[d] = r
        for name, col in cols.items():
            del col[:drop]
            setattr(self, name, col)
        self.base = drop
        self.live_start = drop
        self.end_rank = len(events)


def make_spec(
    *,
    w_rt: int = 100,
    hop: int = 30,
    w_sg: int = 50,
    hubs: tuple[int, ...] = (),
    n_accounts: int = 64,
    rule_visits: int = 2_000_000,
    feat_visits: int = 50_000,
    cap_count: int = 100,
) -> EngineSpec:
    rules_cfg = _yaml("rules.yaml")
    rules_cfg["scenarios"]["round_trip"]["window_minutes"] = w_rt
    rules_cfg["scenarios"]["round_trip"]["hop_window_minutes"] = hop
    features_cfg = _yaml("features.yaml")
    features_cfg["windows"]["sg"] = w_sg
    features_cfg["budgets"] = {"rule_visits": rule_visits, "feat_visits": feat_visits}
    features_cfg["caps"]["count"] = cap_count
    vocab = {
        "payment_currency": ["US Dollar"],
        "receiving_currency": ["US Dollar"],
        "payment_format": ["ACH", "Bitcoin", "Cash", "Wire"],
    }
    return EngineSpec.from_configs(
        features_cfg, rules_cfg, n_accounts=n_accounts, vocab=vocab, hub_cap=10, hubs=hubs
    )


def hub_bytes(spec: EngineSpec) -> bytearray:
    hub = bytearray(spec.n_accounts)
    for h in spec.hubs:
        hub[h] = 1
    return hub


def paths_of(events, spec, clock, u, v, *, drop=0, memo=None):
    ring = MiniRing(events, spec.n_accounts, drop=drop)
    memo = MinuteMemo() if memo is None else memo
    return cycles.path_counts(memo, ring, ring.head_in, hub_bytes(spec), spec, clock, u, v)


def sg_of(events, spec, clock, u, v, *, drop=0):
    ring = MiniRing(events, spec.n_accounts, drop=drop)
    return cycles.sg_counts(MinuteMemo(), ring, ring.head_in, hub_bytes(spec), spec, clock, u, v)


# ---------------------------------------------------------------------------------------------
# Brute force from the spec's definitions (§4.7)


def ref_paths(events, spec, clock, u, v) -> tuple[int, int, int]:
    """(c2, c3, c4) by explicit enumeration over the non-self-loop edges in [clock - W_rt,
    clock - 1]."""
    if u == v:
        return (0, 0, 0)
    hubs, H = set(spec.hubs), spec.hop
    E = [(e[0], e[1], e[2]) for e in events if clock - spec.w_rt <= e[0] <= clock - 1]
    E = [e for e in E if e[1] != e[2]]
    c2 = sum(1 for t, s, d in E if s == v and d == u)
    c3 = 0
    for t1, a, w in E:
        if a != v or w in (u, v) or w in hubs:
            continue
        c3 += sum(1 for t2, s, d in E if s == w and d == u and t1 <= t2 <= t1 + H)
    c4 = 0
    for t1, q, x in E:
        if q != v or x in (u, v) or x in hubs:
            continue
        for t2, s, w in E:
            if s != x or w in (u, v, x) or w in hubs or not t1 <= t2 <= t1 + H:
                continue
            c4 += sum(1 for t3, s3, d3 in E if s3 == w and d3 == u and t2 <= t3 <= t2 + H)
    return (c2, c3, c4)


def ref_sg(events, spec, clock, u, v) -> tuple[int, int]:
    """(sg_mids, sg_srcs), uncapped, by explicit set construction over [clock - W_sg, clock - 1]."""
    if u == v:
        return (0, 0)
    hubs = set(spec.hubs)
    E = [(e[1], e[2]) for e in events if clock - spec.w_sg <= e[0] <= clock - 1]
    su = {s for s, d in E if d == u and s != u and s not in hubs}
    xs = {x for x, d in E if d == v and x not in (u, v)}
    mids, srcs = 0, set()
    for x in xs:
        common = {s for s, d in E if d == x and s != x and s not in hubs} & su - {v}
        if common:
            mids += 1
            srcs |= common
    return (mids, len(srcs))


def random_events(rng: random.Random, n: int, accounts: int, minutes: int, p_self: float = 0.1):
    out = []
    for _ in range(n):
        s = rng.randrange(accounts)
        d = s if rng.random() < p_self else rng.randrange(accounts)
        out.append((rng.randrange(minutes), s, d))
    out.sort(key=lambda e: e[0])
    return out


# ---------------------------------------------------------------------------------------------
# Round trip and cycles: hand cases with literal expected values


U, V, W, X, Q = 0, 1, 2, 3, 4
HUB = 9


def test_cycles_with_equal_minutes_count():
    spec = make_spec()
    events = [(500, V, X), (500, X, W), (500, W, U), (500, V, W)]
    # c3: v -> w -> u (500, 500); c4: v -> x -> w -> u (500, 500, 500)
    assert paths_of(events, spec, 501, U, V) == (0, 1, 1, 0, 0)
    assert ref_paths(events, spec, 501, U, V) == (0, 1, 1)


def test_two_hop_window_bounds():
    spec = make_spec(w_rt=100)
    events = [(399, V, U), (400, V, U), (499, V, U)]
    # Target at minute 500: the window is [400, 499].
    assert paths_of(events, spec, 500, U, V)[:3] == (2, 0, 0)
    # A path whose first hop is at m - W - 1 is outside the window even if the second is inside.
    assert paths_of([(399, V, W), (405, W, U)], spec, 500, U, V)[:3] == (0, 0, 0)
    assert paths_of([(400, V, W), (405, W, U)], spec, 500, U, V)[:3] == (0, 1, 0)


@pytest.mark.parametrize(("gap", "counted"), [(0, 1), (30, 1), (31, 0)])
def test_hop_gap_equal_to_h_counts_and_h_plus_1_not(gap, counted):
    spec = make_spec(w_rt=200, hop=30)
    c3_events = [(400, V, W), (400 + gap, W, U)]
    assert paths_of(c3_events, spec, 500, U, V)[:3] == (0, counted, 0)
    # The same for each hop of a 4-cycle.
    first = [(400, V, X), (400 + gap, X, W), (405 + gap, W, U)]
    assert paths_of(first, spec, 500, U, V)[2] == counted
    second = [(400, V, X), (410, X, W), (410 + gap, W, U)]
    assert paths_of(second, spec, 500, U, V)[2] == counted


def test_later_hop_before_earlier_hop_does_not_count():
    spec = make_spec(w_rt=200, hop=30)
    assert paths_of([(410, W, U), (420, V, W)], spec, 500, U, V)[:3] == (0, 0, 0)
    assert paths_of([(405, X, W), (410, V, X), (420, W, U)], spec, 500, U, V)[:3] == (0, 0, 0)


def test_hub_intermediates_are_excluded():
    spec = make_spec(hubs=(HUB,))
    # v -> hub -> u (3-hops via a hub) and v -> hub -> w -> u, v -> x -> hub -> u (4-hops).
    events = [
        (450, V, HUB),
        (451, HUB, U),
        (452, V, X),
        (453, X, HUB),
        (454, HUB, W),
        (455, W, U),
        (456, HUB, U),
    ]
    assert paths_of(events, spec, 500, U, V)[:3] == (0, 0, 0)
    assert ref_paths(events, spec, 500, U, V) == (0, 0, 0)
    # A hub as the target's receiver v (the path's source) is not an intermediate: it counts.
    hub_v = [(450, HUB, W), (451, W, U), (452, HUB, U)]
    assert paths_of(hub_v, spec, 500, U, HUB)[:3] == (1, 1, 0)
    # Without the hub list every path counts.
    plain = make_spec()
    assert paths_of(events, plain, 500, U, V)[:3] == ref_paths(events, plain, 500, U, V)
    assert paths_of(events, plain, 500, U, V)[:3] == (0, 2, 2)


def test_path_vertices_are_distinct():
    spec = make_spec()
    # v -> x -> v -> u: w == v is no 4-cycle (the last edge is the 2-hop closer).
    assert paths_of([(450, V, X), (451, X, V), (452, V, U)], spec, 500, U, V)[:3] == (1, 0, 0)
    # v -> u -> w -> u: x == u is no 4-cycle.
    assert paths_of([(450, V, U), (451, U, W), (452, W, U)], spec, 500, U, V)[:3] == (1, 0, 0)
    # Target u -> w: w -> x -> w -> u would be a 4-path with q == w (and w == v): excluded.
    assert paths_of([(450, W, X), (451, X, W), (452, W, U)], spec, 500, U, W)[:3] == (1, 0, 0)
    # For the target u -> x the same edges hold the 3-hop x -> w -> u, and no 4-path.
    assert paths_of([(450, W, X), (451, X, W), (452, W, U)], spec, 500, U, X)[:3] == (0, 1, 0)


def test_self_loop_trap_and_self_loop_targets():
    spec = make_spec()
    # v -> v then v -> u: the self-loop pair is no 3-hop path v -> v -> u.
    events = [(450, V, V), (451, V, U), (452, U, U), (453, W, W)]
    assert paths_of(events, spec, 500, U, V) == (1, 0, 0, 0, 0)
    # u -> u -> ... never: self-loops are skipped at every level.
    events2 = [(450, V, W), (451, W, W), (452, W, U), (453, U, U)]
    assert paths_of(events2, spec, 500, U, V)[:3] == (0, 1, 0)
    # A self-loop target gets zeros and builds no memo entry.
    memo = MinuteMemo()
    assert paths_of(events, spec, 500, U, U, memo=memo) == (0, 0, 0, 0, 0)
    assert len(memo) == 0


def test_multi_edges_are_counted_separately():
    spec = make_spec()
    events = [
        (440, V, W),
        (441, V, W),
        (442, V, U),
        (443, V, U),
        (444, V, U),
        (447, V, X),
        (448, X, W),
        (449, X, W),
        (450, W, U),
        (451, W, U),
    ]
    # c2 = 3; c3 = v -> w -> u: 2 x 2; c4 = v -> x -> w -> u: 1 x 2 x 2.
    assert ref_paths(events, spec, 500, U, V) == (3, 4, 4)
    assert paths_of(events, spec, 500, U, V) == (3, 4, 4, 0, 0)


def test_counts_are_raw_and_uncapped():
    spec = make_spec(w_rt=1000, hop=1000)
    events = [(1500 + k, V, U) for k in range(MAX_ROUND_TRIP_PATHS + 20)]
    assert paths_of(events, spec, 2000, U, V)[0] == MAX_ROUND_TRIP_PATHS + 20


def test_memo_is_reused_within_a_minute_and_equals_a_fresh_build():
    rng = random.Random(5)
    spec = make_spec(w_rt=60, hop=20, n_accounts=12)
    events = random_events(rng, 400, 12, 200)
    ring = MiniRing(events, spec.n_accounts)
    hub = hub_bytes(spec)
    memo = MinuteMemo()
    for v in range(12):
        got = cycles.path_counts(memo, ring, ring.head_in, hub, spec, 200, 3, v)
        fresh = cycles.path_counts(MinuteMemo(), ring, ring.head_in, hub, spec, 200, 3, v)
        assert got == fresh
    assert list(memo.path) == [3]  # one entry for the sender, built on the first target
    pm = memo.path[3]
    assert isinstance(pm, PathMemo)
    assert cycles.build_path_memo(ring, ring.head_in, hub, spec, 200, 3) == pm
    memo.clear()
    assert len(memo) == 0


def _paths_vs_bruteforce(seed: int) -> Counter:
    """Compare every (u, v) at a few clocks of a random graph; count non-zero c2, c3, c4."""
    rng = random.Random(seed)
    n_acc = rng.choice([5, 8, 12])
    hubs = tuple(sorted(rng.sample(range(n_acc), rng.choice([0, 1, 2]))))
    spec = make_spec(w_rt=rng.choice([10, 40, 90]), hop=rng.choice([0, 3, 10]), hubs=hubs,
                     n_accounts=n_acc)  # fmt: skip
    events = random_events(rng, rng.choice([60, 200, 400]), n_acc, 120)
    clocks = sorted({rng.randrange(1, 130) for _ in range(6)})
    nonzero = Counter()
    for clock in clocks:
        applied = [e for e in events if e[0] < clock]
        ring = MiniRing(applied, n_acc)
        hub = hub_bytes(spec)
        memo = MinuteMemo()
        for u in range(n_acc):
            for v in range(n_acc):
                got = cycles.path_counts(memo, ring, ring.head_in, hub, spec, clock, u, v)
                assert got[3:] == (0, 0)
                assert got[:3] == ref_paths(applied, spec, clock, u, v), (clock, u, v)
                nonzero.update(k for k in range(3) if got[k])
    return nonzero


@pytest.mark.parametrize("seed", range(12))
def test_path_counts_equal_bruteforce_on_random_graphs(seed):
    _paths_vs_bruteforce(seed)


def test_random_graph_comparisons_are_not_vacuous():
    total = Counter()
    for seed in range(12):
        total += _paths_vs_bruteforce(seed)
    assert min(total[0], total[1], total[2]) > 50, total


def test_walks_stop_at_the_compacted_base():
    """A ring whose rows older than the window are compacted away gives the same counts."""
    rng = random.Random(11)
    spec = make_spec(w_rt=40, hop=10, w_sg=30, n_accounts=8)
    events = random_events(rng, 300, 8, 200)
    clock = 200
    lo = clock - max(spec.w_rt, spec.w_sg)
    drop = sum(1 for e in events if e[0] < lo)
    assert 0 < drop < len(events)
    for u in range(8):
        for v in range(8):
            assert paths_of(events, spec, clock, u, v, drop=drop) == paths_of(
                events, spec, clock, u, v
            )
            assert sg_of(events, spec, clock, u, v, drop=drop) == sg_of(events, spec, clock, u, v)


# ---------------------------------------------------------------------------------------------
# Budgets


def _dense_cycle_graph(width: int) -> list[tuple]:
    """v -> x_i -> w_j -> u for all i, j: width^2 4-paths, width 3-paths via the w_j."""
    xs = list(range(10, 10 + width))
    ws = list(range(40, 40 + width))
    ev = [(400, V, x) for x in xs] + [(410, x, w) for x in xs for w in ws]
    ev += [(415, V, w) for w in ws] + [(420, w, U) for w in ws] + [(430, V, U)]
    return sorted(ev, key=lambda e: e[0])


def test_feat_visits_truncation_gives_a_lower_bound_and_the_flag():
    events = _dense_cycle_graph(6)
    full = make_spec(n_accounts=64)
    exact = paths_of(events, full, 500, U, V)
    assert exact == (1, 6, 36, 0, 0)
    assert exact[:3] == ref_paths(events, full, 500, U, V)
    for budget in (1, 3, 5):
        spec = make_spec(n_accounts=64, feat_visits=budget)
        c2, c3, c4, rule_trunc, cyc_trunc = paths_of(events, spec, 500, U, V)
        assert (c2, c3, rule_trunc, cyc_trunc) == (1, 6, 0, 1)  # D1/D2 exact, D3 stopped
        assert c4 < 36
    # The flag is set only when the walk needed more steps than the budget: the exact count
    # needs `steps` level-3 steps, so that budget is enough and one less is not.
    steps = next(b for b in range(1, 200) if paths_of(events, make_spec(n_accounts=64,
                 feat_visits=b), 500, U, V)[4] == 0)  # fmt: skip
    assert paths_of(events, make_spec(n_accounts=64, feat_visits=steps), 500, U, V) == exact
    low = paths_of(events, make_spec(n_accounts=64, feat_visits=steps - 1), 500, U, V)
    assert low[4] == 1 and low[2] <= 36


def test_rule_visits_truncation_stops_everything():
    events = _dense_cycle_graph(6)
    for budget in (1, 3, 8):
        spec = make_spec(n_accounts=64, rule_visits=budget)
        c2, c3, c4, rule_trunc, cyc_trunc = paths_of(events, spec, 500, U, V)
        assert rule_trunc == 1 and cyc_trunc == 1
        assert c2 <= 1 and c3 <= 6 and c4 == 0


@pytest.mark.parametrize("seed", range(4))
def test_truncated_counts_are_lower_bounds_and_untruncated_ones_exact(seed):
    rng = random.Random(100 + seed)
    events = random_events(rng, 350, 7, 100, p_self=0.05)
    seen = Counter()
    for rule_visits, feat_visits in ((10, 10**6), (10**6, 15), (40, 40)):
        spec = make_spec(w_rt=60, hop=30, n_accounts=7, rule_visits=rule_visits,
                         feat_visits=feat_visits)  # fmt: skip
        for u in range(7):
            for v in range(7):
                c2, c3, c4, rt, ct = paths_of(events, spec, 100, u, v)
                w2, w3, w4 = ref_paths(events, spec, 100, u, v)
                assert c2 <= w2 and c3 <= w3 and c4 <= w4
                if not rt:
                    assert (c2, c3) == (w2, w3)
                if not ct:
                    assert c4 == w4
                assert ct >= rt
                seen[(rt, ct)] += 1
    assert seen[(1, 1)] and seen[(0, 1)]  # both kinds of truncation were exercised


# ---------------------------------------------------------------------------------------------
# Scatter-gather


S1, S2, X2 = 5, 6, 7


def test_sg_siblings_and_sources():
    spec = make_spec(w_sg=100)
    base = [(450, S1, U), (451, S1, X), (452, X, V)]
    assert sg_of(base, spec, 500, U, V) == (1, 1, 0)
    # A second common source of the same sibling: one sibling, two sources.
    two_src = sorted(base + [(453, S2, U), (454, S2, X)])
    assert sg_of(two_src, spec, 500, U, V) == (1, 2, 0)
    # A second sibling fed by the first source.
    two_mid = sorted(base + [(455, S1, X2), (456, X2, V)])
    assert sg_of(two_mid, spec, 500, U, V) == (2, 1, 0)
    for ev in (base, two_src, two_mid):
        assert sg_of(ev, spec, 500, U, V)[:2] == ref_sg(ev, spec, 500, U, V)


def test_sg_exclusions():
    spec = make_spec(w_sg=100, hubs=(HUB,))
    # Hub source: excluded.
    assert sg_of([(450, HUB, U), (451, HUB, X), (452, X, V)], spec, 500, U, V) == (0, 0, 0)
    # s == v: excluded (v -> u, v -> x, x -> v is a cycle, not a scatter-gather).
    assert sg_of([(450, V, U), (451, V, X), (452, X, V)], spec, 500, U, V) == (0, 0, 0)
    # x == u: u's own earlier payment to v is no sibling.
    assert sg_of([(450, S1, U), (451, U, V)], spec, 500, U, V) == (0, 0, 0)
    # Self-loops are neither sources nor siblings.
    assert sg_of([(450, X, X), (451, X, U), (452, X, V), (453, V, V)], spec, 500, U, V) == (
        0,
        0,
        0,
    )
    # Outside the window [m - W_sg, m - 1]: nothing; at m - W_sg: counted.
    assert sg_of([(399, S1, U), (450, S1, X), (452, X, V)], spec, 500, U, V) == (0, 0, 0)
    assert sg_of([(400, S1, U), (450, S1, X), (452, X, V)], spec, 500, U, V) == (1, 1, 0)
    # Self-loop target: zeros, no memo.
    memo = MinuteMemo()
    ring = MiniRing([(450, S1, U)], spec.n_accounts)
    assert cycles.sg_counts(memo, ring, ring.head_in, hub_bytes(spec), spec, 500, U, U) == (
        0,
        0,
        0,
    )
    assert len(memo) == 0


def _sg_vs_bruteforce(seed: int) -> int:
    """Compare every (u, v) at a few clocks of a random graph; count non-zero sg_mids."""
    rng = random.Random(1000 + seed)
    n_acc = rng.choice([6, 10])
    hubs = tuple(sorted(rng.sample(range(n_acc), rng.choice([0, 1]))))
    spec = make_spec(w_sg=rng.choice([5, 30, 80]), hubs=hubs, n_accounts=n_acc, cap_count=1000)
    events = random_events(rng, rng.choice([80, 300]), n_acc, 100)
    nonzero = 0
    for clock in sorted({rng.randrange(1, 110) for _ in range(5)}):
        applied = [e for e in events if e[0] < clock]
        ring = MiniRing(applied, n_acc)
        hub = hub_bytes(spec)
        memo = MinuteMemo()
        for u in range(n_acc):
            for v in range(n_acc):
                got = cycles.sg_counts(memo, ring, ring.head_in, hub, spec, clock, u, v)
                assert got[2] == 0
                assert got[:2] == ref_sg(applied, spec, clock, u, v), (clock, u, v)
                nonzero += got[0] > 0
    return nonzero


@pytest.mark.parametrize("seed", range(10))
def test_sg_equals_bruteforce_on_random_graphs(seed):
    _sg_vs_bruteforce(seed)


def test_random_sg_comparisons_are_not_vacuous():
    assert sum(_sg_vs_bruteforce(seed) for seed in range(10)) > 50


def test_sg_counts_are_capped_and_truncation_is_flagged():
    n = 30
    sources = list(range(10, 10 + n))
    events = [(400, s, U) for s in sources] + [(401, s, X) for s in sources] + [(402, X, V)]
    events.sort(key=lambda e: e[0])
    assert sg_of(events, make_spec(w_sg=200, n_accounts=64), 500, U, V) == (1, n, 0)
    assert sg_of(events, make_spec(w_sg=200, n_accounts=64, cap_count=7), 500, U, V) == (1, 7, 0)
    # Building Ev(v) needs 1 + n steps (x -> v, then x's in-edges): a smaller budget truncates.
    spec = make_spec(w_sg=200, n_accounts=64, feat_visits=n)
    mids, srcs, trunc = sg_of(events, spec, 500, U, V)
    assert trunc == 1 and mids <= 1 and srcs <= n
    assert sg_of(events, make_spec(w_sg=200, n_accounts=64, feat_visits=n + 1), 500, U, V) == (
        1,
        n,
        0,
    )


def test_sg_with_no_source_into_u_is_exactly_zero():
    # Su(u) is empty: no sibling can share a source with u, whatever v's neighbourhood holds.
    events = [(400 + k, 10 + k, V) for k in range(40)]
    spec = make_spec(w_sg=200, n_accounts=64, feat_visits=1)
    memo = MinuteMemo()
    ring = MiniRing(events, spec.n_accounts)
    got = cycles.sg_counts(memo, ring, ring.head_in, hub_bytes(spec), spec, 500, U, V)
    assert got == (0, 0, 0)
    assert memo.su == {U: frozenset()} and memo.ev == {}


# ---------------------------------------------------------------------------------------------
# neighbours (the M6 read API)


def test_neighbours_newest_first_with_cap_and_window():
    spec = make_spec(n_accounts=16)
    events = [(100, 5, U, 11), (300, 6, U, 12), (310, U, U, 13), (320, 7, U, 14), (330, U, 8, 15)]
    ring = MiniRing(events, spec.n_accounts)
    nb = cycles.neighbours
    args = (ring, ring.head_in, ring.head_out, spec, 400)
    assert nb(*args, U, "in", 100, 10) == [(7, 320, 3, 14), (U, 310, 2, 13), (6, 300, 1, 12)]
    assert nb(*args, U, "in", 100, 2) == [(7, 320, 3, 14), (U, 310, 2, 13)]
    assert nb(*args, U, "in", 300, 10)[-1] == (5, 100, 0, 11)  # minute 100 = clock - 300
    assert nb(*args, U, "in", 299, 10)[-1] == (6, 300, 1, 12)
    assert nb(*args, U, "out", 100, 10) == [(8, 330, 4, 15), (U, 310, 2, 13)]
    assert nb(*args, U, "in", 100, 0) == []
    assert nb(*args, 15, "in", 100, 5) == []
    assert nb(ring, ring.head_in, ring.head_out, spec, None, U, "in", 100, 5) == []
    assert nb(*args, np.int64(U), "in", np.int32(100), 1) == [(7, 320, 3, 14)]
    for bad in (
        dict(direction="both"),
        dict(window=0),
        dict(window=spec.W_max + 1),
        dict(cap=-1),
        dict(acct=-1),
        dict(acct=spec.n_accounts),
        dict(acct=True),
        dict(window=1.5),
    ):
        kw = dict(acct=U, direction="in", window=100, cap=3) | bad
        with pytest.raises(ValueError):
            nb(*args, kw["acct"], kw["direction"], kw["window"], kw["cap"])


# ---------------------------------------------------------------------------------------------
# M1's round-trip scenarios on the engine (needs aml.features.engine)


def _engine_available() -> bool:
    from aml.features.engine import Engine

    try:
        Engine.create(make_spec())
    except NotImplementedError:
        return False
    return True


needs_engine = pytest.mark.skipif(
    not _engine_available(), reason="aml.features.engine.Engine is not implemented yet"
)


def replay_frame(tx: pl.DataFrame, spec: EngineSpec) -> list[tuple]:
    """Engine rows of a rules_frames-style frame (rank order); currencies and banks constant."""
    from aml.features.engine import Engine

    eng = Engine.create(spec)
    rows = []
    for r in tx.sort("rank").iter_rows(named=True):
        ev = eng.prepare(
            r["rank"],
            r["row_id"],
            r["minute"],
            r["src"],
            r["dst"],
            r["amount_usd"],
            r["amount_paid"],
            r["payment_format"],
            r.get("payment_currency", "US Dollar"),
            r.get("receiving_currency", "US Dollar"),
            r.get("from_bank", "001"),
            r.get("to_bank", "002"),
        )
        rows.append(eng.process(ev))
    return rows


def frame_spec(tx: pl.DataFrame, rules_cfg: dict, hub_cap: int, **features) -> EngineSpec:
    """EngineSpec for a hand-made frame: hubs from M1's r_hubs definition on the frame."""
    con = connect(threads=1)
    try:
        register_transactions(con, tx)
        hubs = hub_accounts(con, hub_cap)
    finally:
        con.close()
    features_cfg = _yaml("features.yaml")
    for section, values in features.items():
        features_cfg[section] = {**features_cfg[section], **values}
    formats = sorted(set(tx["payment_format"].to_list()))
    vocab = {
        "payment_currency": ["US Dollar"],
        "receiving_currency": ["US Dollar"],
        "payment_format": formats,
    }
    n = int(max(tx["src"].max(), tx["dst"].max())) + 1
    return EngineSpec.from_configs(
        features_cfg, rules_cfg, n_accounts=n, vocab=vocab, hub_cap=hub_cap, hubs=hubs
    )


def engine_severities(tx: pl.DataFrame, spec: EngineSpec) -> pl.DataFrame:
    rows = replay_frame(tx, spec)
    i = spec.i_sev
    data = {s: [row[i + j] for row in rows] for j, s in enumerate(SCENARIOS)}
    ids = tx.sort("rank")["row_id"].to_list()
    return pl.DataFrame({"row_id": ids, **data}, schema_overrides={"row_id": pl.Int64})


def sql_severities(tx: pl.DataFrame, rules_cfg: dict, hub_cap: int) -> pl.DataFrame:
    con = connect(threads=2)
    try:
        register_transactions(con, tx)
        return compute_severities(con, rules_cfg, hub_cap).select("row_id", *SCENARIOS)
    finally:
        con.close()


def _sev(sev: pl.DataFrame, row_id: int, scenario: str) -> float:
    return sev.filter(pl.col("row_id") == row_id)[scenario].item()


@pytest.fixture(scope="module")
def rules_cfg_m1() -> dict:
    return copy.deepcopy(_yaml("rules.yaml"))


@needs_engine
def test_m1_round_trip_paths_on_the_engine(rules_cfg_m1):
    """M1 test_rules_sql.test_round_trip_paths, replayed through the engine."""
    cfg = small_cfg(rules_cfg_m1, window=100, hop=30)
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
        *[{"minute": 10, "src": hub, "dst": 100 + k, "split": "train"} for k in range(4)],
        {"minute": 1000, "src": u, "dst": u},  # self-loop target: 0
    ]
    tx = make_tx(rows)
    for cap, want in ((3, 4.0), (10**9, 5.0)):
        spec = frame_spec(tx, cfg, cap)
        assert (hub in spec.hubs) == (cap == 3)
        got = engine_severities(tx, spec)
        assert _sev(got, 0, "round_trip") == want
        assert _sev(got, len(rows) - 1, "round_trip") == 0.0
        assert got.equals(sql_severities(tx, cfg, cap))
        # The cycle features of the target: c2 = 2, c3 = 2 (+1 via the hub when it is no hub).
        row0 = replay_frame(tx, spec)[tx.sort("rank")["row_id"].to_list().index(0)]
        tag = window_tag(spec.w_rt)
        c2 = row0[spec.feature_index[f"cyc2_{tag}"]]
        c3 = row0[spec.feature_index[f"cyc3_{tag}"]]
        assert (c2, c3) == (2.0, want - 2.0)
    rows2 = [
        {"minute": 1000, "src": u, "dst": v},
        {"minute": 899, "src": v, "dst": w},
        {"minute": 905, "src": w, "dst": u},
    ]
    tx2 = make_tx(rows2)
    got = engine_severities(tx2, frame_spec(tx2, cfg, 10**9))
    assert _sev(got, 0, "round_trip") == 0.0


@needs_engine
def test_m1_round_trip_is_capped_on_the_engine(rules_cfg_m1):
    """M1 test_rules_sql.test_round_trip_is_capped, replayed through the engine."""
    cfg = small_cfg(rules_cfg_m1, window=1000, hop=1000)
    rows = [{"minute": 2000, "src": 0, "dst": 1}]
    rows += [{"minute": 1500 + k, "src": 1, "dst": 0} for k in range(MAX_ROUND_TRIP_PATHS + 20)]
    tx = make_tx(rows)
    spec = frame_spec(tx, cfg, 10**9)
    got = engine_severities(tx, spec)
    assert _sev(got, 0, "round_trip") == float(MAX_ROUND_TRIP_PATHS)
    assert got.equals(sql_severities(tx, cfg, 10**9))
    row0 = replay_frame(tx, spec)[tx.sort("rank")["row_id"].to_list().index(0)]
    assert row0[spec.feature_index[f"cyc2_{window_tag(spec.w_rt)}"]] == float(spec.cap_count)


@needs_engine
@pytest.mark.parametrize(("seed", "hub_cap"), [(0, 10**9), (1, 12), (2, 25)])
def test_engine_cycle_and_sg_features_equal_bruteforce(rules_cfg_m1, seed, hub_cap):
    """The engine's cyc*/sg* columns and trunc flags on a dense frame equal the brute force
    (capped as the features are), with hubs from M1's r_hubs definition."""
    rng = random.Random(seed)
    rows = [
        {
            "minute": rng.randrange(80),
            "src": rng.randrange(8),
            "dst": rng.randrange(8),
            "split": "train" if k < 120 else "val_early",
        }
        for k in range(400)
    ]
    tx = make_tx(rows)
    cfg = small_cfg(rules_cfg_m1, window=20, hop=5)
    spec = frame_spec(tx, cfg, hub_cap, windows={"sg": 10}, caps={"count": 6})
    out = replay_frame(tx, spec)
    events = [(r["minute"], r["src"], r["dst"]) for r in tx.sort("rank").iter_rows(named=True)]
    tag, g = window_tag(spec.w_rt), window_tag(spec.w_sg)
    idx = [spec.feature_index[f"cyc{k}_{tag}"] for k in (2, 3, 4)]
    idx += [spec.feature_index[f"sg_{k}_{g}"] for k in ("mids", "srcs")]
    trunc = (spec.i_rule_trunc, spec.i_cyc_trunc, spec.i_sg_trunc)
    cap = spec.cap_count
    seen = Counter()
    for (m, u, v), row in zip(events, out, strict=True):
        want = [
            min(c, cap) for c in (*ref_paths(events, spec, m, u, v), *ref_sg(events, spec, m, u, v))
        ]
        assert [row[i] for i in idx] == want, (m, u, v)
        assert [row[i] for i in trunc] == [0, 0, 0]
        seen.update(k for k, c in enumerate(want) if c)
    if hub_cap == 10**9:
        assert all(seen[k] for k in range(5)), seen  # every column is non-zero somewhere
    else:
        assert spec.hubs and seen[0], seen  # the low caps make hubs (excluded intermediates)
