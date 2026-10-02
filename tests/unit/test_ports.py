"""The lifetime pair table, per-window pair counts, ports and gaps (M2 spec §4.3 D, §4.5)."""

from __future__ import annotations

import math
from array import array

import pytest

from aml.features.ports import PairTable, capped_gap, pair_key
from aml.features.spec import I32_LIMIT, NumericRangeError
from tests.unit.test_engine_core import Feed, make_spec

# --- PairTable ------------------------------------------------------------------------------------


def test_pair_ids_follow_first_add_order_and_pairs_are_never_deleted():
    t = PairTable((1440, 4320))
    assert t.n_pairs == 0 and t.pid(1, 2) == -1
    assert t.add(1, 2, 0, 0, 5) == 0
    assert t.add(2, 1, 3, 4, 6) == 1  # the reverse pair is a different pair
    assert t.add(7, 7, 1, 1, 6) == 2  # self-loop pairs are pairs too
    assert (t.pid(1, 2), t.pid(2, 1), t.pid(7, 7), t.pid(2, 2)) == (0, 1, 2, -1)
    assert list(t.keys) == [pair_key(1, 2), pair_key(2, 1), pair_key(7, 7)]
    assert list(t.pcnt[1440]) == list(t.pcnt[4320]) == [0, 0, 0]
    assert (list(t.port_out), list(t.port_in), list(t.last_min)) == (
        [0, 3, 1],
        [0, 4, 1],
        [5, 6, 6],
    )
    with pytest.raises(ValueError, match="already exists"):
        t.add(1, 2, 0, 0, 9)
    assert t.n_pairs == 3 and t.nbytes() > 0
    assert [name for name, _ in t.columns()] == [
        "keys",
        "pcnt.1440",
        "pcnt.4320",
        "port_out",
        "port_in",
        "last_min",
    ]


def test_pair_key_is_unique_for_account_ids_below_2_pow_31():
    big = I32_LIMIT - 1
    keys = {pair_key(u, v) for u in (0, 1, big) for v in (0, 1, big)}
    assert len(keys) == 9 and pair_key(big, big) < 2**63


def test_pair_table_round_trips_through_its_columns():
    t = PairTable((3, 9))
    for k, (u, v) in enumerate(((1, 2), (2, 3), (3, 1), (4, 4))):
        t.add(u, v, k, 2 * k, 10 + k)
        t.pcnt[3][k] = k % 2
        t.pcnt[9][k] = k
    cols = dict(t.columns())
    r = PairTable.from_columns(
        (3, 9),
        cols["keys"],
        {3: cols["pcnt.3"], 9: cols["pcnt.9"]},
        cols["port_out"],
        cols["port_in"],
        cols["last_min"],
    )
    assert r.index == t.index and r.n_pairs == 4 and r.pid(3, 1) == 2
    with pytest.raises(ValueError, match="repeat"):
        keys = array("q", [pair_key(1, 2), pair_key(1, 2)])
        PairTable.from_columns((3,), keys, {3: array("i", [0, 0])}, *(array("i", [0, 0]),) * 3)
    with pytest.raises(ValueError, match="pcnt.3"):
        keys = array("q", [pair_key(1, 2)])
        PairTable.from_columns((3,), keys, {3: array("i", [0, 0])}, *(array("i", [0]),) * 3)


def test_pair_ids_are_range_checked(monkeypatch):
    t = PairTable((3,))
    monkeypatch.setattr("aml.features.ports.I32_LIMIT", 2)
    t.add(1, 2, 0, 0, 0)
    t.add(1, 3, 0, 0, 0)
    with pytest.raises(NumericRangeError):
        t.add(1, 4, 0, 0, 0)


def test_capped_gap():
    assert capped_gap(10, -1, 5) == 0  # no history
    assert capped_gap(10, 9, 5) == 1
    assert capped_gap(10, 5, 5) == 5
    assert capped_gap(10, 0, 5) == 5


# --- the engine's pair counts, ports and gaps (default configs: S = 1440, L = 4320) -------------


def test_pair_counts_and_uniq_transitions_per_window():
    spec = make_spec()
    feed = Feed(spec)
    feed(0, 1, 2)
    feed(1, 1, 2)
    feed(2, 1, 3)
    r = feed(3, 1, 2)
    assert r["pair_cnt_1d"] == 2.0 and r["pair_cnt_3d"] == 2.0
    assert r["u_out_uniq_1d"] == 2.0 and r["u_out_cnt_1d"] == 3.0 and r["v_in_uniq_1d"] == 1.0
    eng = feed.eng
    pid = eng.pairs.pid(1, 2)
    assert eng.pairs.pcnt[spec.w_short][pid] == 2  # the minute-3 event is still pending
    r = feed(1441, 1, 2)  # window [1, 1440]: the minute-0 event left S
    assert r["pair_cnt_1d"] == 2.0 and r["pair_cnt_3d"] == 3.0 and r["u_out_uniq_1d"] == 2.0
    r = feed(1444, 1, 9)  # window [4, 1443]: (1, 3) at minute 2 left S, (1, 2) at 1441 stays
    assert r["u_out_uniq_1d"] == 1.0 and r["u_out_uniq_3d"] == 2.0  # 1 -> 9 is pending
    eng.advance(1445)
    eng.check_invariants()
    assert eng.pairs.pcnt[spec.w_short][pid] == 1 and eng.pairs.pcnt[spec.w_long][pid] == 4


def test_a_pair_expiring_and_reappearing_keeps_its_lifetime_fields():
    spec = make_spec()
    feed = Feed(spec)
    feed(0, 5, 1)  # 1 has one sender before (1, 2) starts
    feed(1, 1, 7)
    feed(2, 1, 2)
    feed(2, 4, 2)  # 2's senders before minute 2: none
    r = feed(3, 1, 2)
    assert (r["out_port"], r["in_port"]) == (math.log1p(1), 0.0)
    r = feed(10_000, 1, 2)  # every window is empty again
    assert r["pair_cnt_1d"] == r["pair_cnt_3d"] == r["u_out_uniq_3d"] == 0.0
    assert r["pair_is_new"] == 0.0 and r["pair_gap"] == 4320.0  # capped gap, not "new"
    assert (r["out_port"], r["in_port"]) == (math.log1p(1), 0.0)  # stored at the first minute
    r = feed(10_001, 1, 2)
    assert r["pair_cnt_1d"] == 1.0 and r["pair_gap"] == 1.0 and r["u_out_uniq_1d"] == 1.0
    r = feed(10_001, 9, 2)  # a new sender of 2: its in-port = 2's distinct senders so far
    assert r["pair_is_new"] == 1.0 and r["in_port"] == math.log1p(2)
    feed.eng.advance(10_002)
    feed.eng.check_invariants()


def test_ports_are_lifetime_counts_and_capped_only_in_the_feature():
    spec = make_spec(small=True)  # port cap 3
    feed = Feed(spec)
    for k, v in enumerate((2, 3, 4, 5, 6)):
        feed(k, 1, v)
    r = feed(20, 1, 7)
    assert r["out_port"] == math.log1p(3)  # min(5, cap 3)
    assert feed.eng.ever_out[1] == 5  # the state itself is never saturated
    assert feed.eng.pairs.port_out[feed.eng.pairs.pid(1, 6)] == 4


def test_rev_pair_gaps_and_self_loop_pairs():
    feed = Feed(make_spec())
    feed(5, 2, 1)
    r = feed(9, 1, 2)
    assert r["rev_pair_gap"] == 4.0 and r["pair_gap"] == 0.0 and r["pair_is_new"] == 1.0
    feed(9, 3, 3)
    r = feed(10, 3, 4)
    # a self-loop pair counts 3 as its own receiver and sender
    assert r["u_out_uniq_1d"] == 1.0 and r["u_in_uniq_1d"] == 1.0
    assert r["out_port"] == math.log1p(1)  # the self-loop pair (3, 3) was 3's first receiver
    r = feed(10, 3, 3)
    assert r["pair_is_new"] == 0.0 and r["pair_cnt_1d"] == 1.0 and r["rev_pair_gap"] == 1.0
    eng = feed.eng
    eng.advance(11)
    eng.check_invariants()
    assert eng.ever_out[3] == 2 and eng.ever_in[3] == 1 and eng.ever_in[4] == 1
