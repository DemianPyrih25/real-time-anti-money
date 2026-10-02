"""Stateful property test of the engine (M2 spec §9.2): random event streams, snapshot/restore,
re-scoring of pending events, late events and idle minute flushes. Every returned row is checked
against the brute-force reference, `check_invariants()` runs after every step that changed the
applied state (a minute flush or a restore), and M1's SQL severities are checked over every
finished history.

Tiny windows make every boundary reachable: accounts 0-5 (5 is a hub), short 3, long 7, sg 3;
fan-in 3, fan-out 4, pass-through 2, round trip 5 / hop 2, structuring 3, round 3, high-risk 4;
caps gap 6, port 3, count 100; feat_visits 40 (the spec's setting; at this density the budgets
are rarely reached, so tests/parity/test_oracle.py pins truncation with tiny budgets).
Profiles: the default runs 150 examples of up to 60 steps (derandomized); set
AML_HYPOTHESIS_PROFILE=slow for 2,000.
"""

from __future__ import annotations

import os

import numpy as np
import polars as pl
import pytest
from hypothesis import HealthCheck, settings
from hypothesis import strategies as st
from hypothesis.stateful import RuleBasedStateMachine, initialize, invariant, precondition, rule

from aml.features.spec import EngineSpec, LateEventError
from aml.rules.sql_baseline import (
    SCENARIOS,
    TX_COLUMNS,
    compute_severities,
    connect,
    register_transactions,
)
from tests.fixtures.engine_frames import (
    engine_missing,
    features_cfg,
    load_cfg,
    row_bits,
    rules_windows,
)
from tests.fixtures.engine_ref import Reference, compare_rows

SLOW = os.environ.get("AML_HYPOTHESIS_PROFILE") == "slow"
HUB = 5
N_ACCOUNTS = 6
HUB_FLAGS = {
    "fan_out_velocity": True,
    "structuring": False,
    "round_amount_burst": True,
    "high_risk_format_burst": False,
}
RULES = rules_windows(
    load_cfg("rules"),
    fan_in=3,
    fan_out=4,
    pass_through=2,
    round_trip=5,
    hop=2,
    structuring=3,
    round_burst=3,
    high_risk=4,
    excl=HUB_FLAGS,
)
# Cheque and Yen are outside the train vocabulary: their codes are -1.
VOCAB = {
    "payment_format": ["ACH", "Bitcoin", "Cash", "Reinvestment", "Wire"],
    "payment_currency": ["Euro", "US Dollar"],
    "receiving_currency": ["Euro", "US Dollar"],
}
SPEC = EngineSpec.from_configs(
    features_cfg(
        short=3, long=7, sg=3, gap=6, port=3, count=100, feat_visits=40, compact_min_rows=2
    ),
    RULES,
    n_accounts=N_ACCOUNTS,
    vocab=VOCAB,
    hub_cap=0,  # the SQL check makes exactly account 5 a hub with a future train row
    hubs=[HUB],
)
AMOUNTS = (0.0, 0.004, 0.005, 0.015, 99.995, 100.0, 8999.99, 9000.0, 9999.995, 10000.0, 1e11)
AMOUNT = st.one_of(
    st.sampled_from(AMOUNTS), st.floats(0, 1e7, allow_nan=False, allow_infinity=False)
)
FORMATS = ("ACH", "Cash", "Bitcoin", "Wire", "Reinvestment", "Cheque")
CURRENCIES = ("US Dollar", "Euro", "Yen")
BANKS = ("001", "002", "1")
# Minutes to the next event, drawn from the spec's multiset {0, 0, 0, 1, 2, W - 1, W, W + 1,
# 3 * W_max} with W one of the registered windows: same-minute bursts, exact window edges, and
# gaps that empty every window.
_DT_KINDS = (0, 0, 0, 1, 2, "W-1", "W", "W+1", "3Wmax")


def _dt(kind, w: int) -> int:
    if isinstance(kind, int):
        return kind
    return {"W-1": w - 1, "W": w, "W+1": w + 1, "3Wmax": 3 * SPEC.W_max}[kind]


DT = st.builds(_dt, st.sampled_from(_DT_KINDS), st.sampled_from(SPEC.windows_all))
ACCOUNT = st.integers(0, N_ACCOUNTS - 1)
# One drawn event: dt, repeat-a-pair (0 = yes), which pair, u, self-loop (0 = yes), v, amount,
# paid = amount?, paid, payment currency, receiving = payment?, receiving currency, from bank,
# to bank = from bank?, to bank, format.
EVENT = st.tuples(
    DT,
    st.integers(0, 3),
    st.integers(0, 2**16),
    ACCOUNT,
    st.integers(0, 4),
    ACCOUNT,
    AMOUNT,
    st.booleans(),
    AMOUNT,
    st.sampled_from(CURRENCIES),
    st.booleans(),
    st.sampled_from(CURRENCIES),
    st.sampled_from(BANKS),
    st.booleans(),
    st.sampled_from(BANKS),
    st.sampled_from(FORMATS),
)
# Finished (events, rows) of every example, for the batched SQL check below.
HISTORIES: list[tuple[list[tuple], list[tuple]]] = []


class EngineMachine(RuleBasedStateMachine):
    def __init__(self) -> None:
        super().__init__()
        from aml.features.engine import Engine

        self.Engine = Engine
        self.eng = Engine.create(SPEC)
        self.ref = Reference(SPEC)
        self.events: list[tuple] = []  # engine input tuples (INPUT_COLUMNS order)
        self.rows: list[tuple] = []  # the rows as first returned
        self.minute: int | None = None  # the clock in the model
        self.pending_from = 0  # index of the first event not yet applied
        self.pairs: list[tuple[int, int]] = []
        self.dirty = True  # applied state changed since the last check_invariants()

    def _state(self) -> tuple:
        e = self.eng
        return e.state_digest(), e.next_rank, e.pending_count, e.clock

    @initialize(e=EVENT)
    def first_event(self, e: tuple) -> None:
        self.add_one(e)  # every rule is enabled from the first step on

    @rule(first=EVENT, rest=st.lists(st.tuples(st.sampled_from([0, 0, 1]), EVENT), max_size=9))
    def add_event(self, first: tuple, rest: list[tuple]) -> None:
        """A burst of 1-10 events: the first after a gap from the spec's multiset, the rest in the
        same or the next minute, so windows get dense enough for 4-cycles, scatter-gather and
        the feat_visits budget (the other rules leave the event stream as it is)."""
        self.add_one(first)
        for dt, e in rest:
            self.add_one((dt, *e[1:]))

    def add_one(self, e: tuple) -> None:
        dt, repeat, pick, u, self_loop, v, a, same_paid, paid, pcur, same_cur, rcur = e[:12]
        fb, same_bank, tb, fmt = e[12:]
        m = dt if self.minute is None else self.minute + dt
        if self.pairs and repeat == 0:  # a repeated pair
            u, v = self.pairs[pick % len(self.pairs)]
        elif self_loop == 0:  # about one event in five is a self-loop
            v = u
        paid = a if same_paid else paid
        rcur = pcur if same_cur else rcur
        tb = fb if same_bank else tb
        rank = len(self.events)
        fields = (rank, 1000 + rank, m, u, v, a, paid, fmt, pcur, rcur, fb, tb)
        if self.minute is None or m > self.minute:
            self.pending_from, self.minute = rank, m
            self.dirty = True  # the minute flush applied and expired events
        row = self.eng.process(self.eng.prepare(*fields))
        self.events.append(fields)
        self.rows.append(row)
        self.ref.append(fields)
        if (u, v) not in self.pairs:
            self.pairs.append((u, v))
        want, m2 = self.ref.row_m2(rank)
        bad = compare_rows(row, want, m2, SPEC)
        assert not bad, (fields, bad)

    @precondition(lambda self: self.minute is not None)
    @rule(compress=st.booleans())
    def snapshot_restore(self, compress: bool) -> None:
        """Bytes round trip, then the pending events are re-fed as M5 does after a restart."""
        before = self._state()
        raw = self.eng.snapshot(compress=compress)
        assert isinstance(raw, bytes | bytearray)
        assert self._state() == before  # taking a snapshot changes nothing
        eng2, header = self.Engine.restore(bytes(raw), SPEC)
        assert header["next_rank"] == self.pending_from  # pending events are never saved
        assert header["clock"] == self.minute
        assert eng2.pending_count == 0 and eng2.next_rank == self.pending_from
        for k in range(self.pending_from, len(self.events)):
            row = eng2.process(eng2.prepare(*self.events[k]))
            assert row_bits(row) == row_bits(self.rows[k]), k
        assert eng2.state_digest() == before[0]
        self.eng = eng2
        self.dirty = True

    @precondition(lambda self: self.minute is not None and self.pending_from < len(self.events))
    @rule(pick=st.integers(0, 2**16))
    def rescore_pending(self, pick: int) -> None:
        k = self.pending_from + pick % (len(self.events) - self.pending_from)
        before = self._state()
        row = self.eng.score(self.eng.prepare(*self.events[k]))
        assert row_bits(row) == row_bits(self.rows[k])
        assert self._state() == before

    @precondition(lambda self: self.minute is not None and self.minute > 0)
    @rule(back=st.one_of(st.just(1), st.integers(1, 2**16)))
    def late_event(self, back: int) -> None:
        m = max(0, self.minute - back)
        fields = (len(self.events), 999_999, m, 0, 1, 1.0, 1.0, "ACH", "Euro", "Euro", "1", "1")
        before = self._state()
        with pytest.raises(LateEventError):
            self.eng.process(self.eng.prepare(*fields))
        with pytest.raises(LateEventError):
            self.eng.advance(m)
        assert self._state() == before

    @precondition(lambda self: self.minute is not None)
    @rule(dt=st.sampled_from([0, 1, 3, 8, 30]))
    def idle_advance(self, dt: int) -> None:
        """A minute flush with no event (M5's idle timer; boundary snapshots)."""
        if dt == 0:  # advancing to the current minute is a no-op
            before = self._state()
            stats = self.eng.advance(self.minute)
            assert self._state() == before and stats.n_applied == 0
            return
        stats = self.eng.advance(self.minute + dt)
        assert stats.n_applied == len(self.events) - self.pending_from
        self.minute += dt
        self.pending_from = len(self.events)
        self.dirty = True

    @invariant()
    def consistent(self) -> None:
        e = self.eng
        if self.minute is not None:
            assert e.clock == self.minute
            assert e.next_rank == len(self.events)
            assert e.pending_count == len(self.events) - self.pending_from
        # Aggregates change only when a minute is flushed (or the engine is replaced): checking
        # after those steps is the same check as after every step, at a fraction of the cost.
        if self.dirty:
            e.check_invariants()
            self.dirty = False

    def teardown(self) -> None:
        if self.events:
            self.ref.rows()  # the port definition agrees with the minute-by-minute simulation
            HISTORIES.append((list(self.events), list(self.rows)))


TestEngineMachine = pytest.mark.skipif(engine_missing() is not None, reason=str(engine_missing()))(
    EngineMachine.TestCase
)
TestEngineMachine.settings = settings(
    max_examples=2000 if SLOW else 150,
    stateful_step_count=60,
    derandomize=True,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.data_too_large],
)


def test_sql_severities_match_every_machine_history() -> None:
    """M1's scenarios.sql over the histories == the severities the engine returned.

    All histories run as one SQL frame: example k's accounts are shifted by 10 * k (disjoint, so
    no window or path crosses examples), every real row is non-train, and one far-future train
    self-loop per example makes exactly its account 5 a hub at hub_cap 0.
    """
    if not HISTORIES:
        pytest.skip("run together with TestEngineMachine (it records the histories)")
    recs, engine_rows = [], []
    far = 10**6
    for k, (events, rows) in enumerate(HISTORIES):
        off = 10 * k
        for ev, row in zip(events, rows, strict=True):
            rank, _, m, u, v, a, paid, fmt = ev[:8]
            recs.append((k * 100_000 + rank, m, u + off, v + off, a, paid, fmt, "val_early"))
            engine_rows.append(row)
        hub = HUB + off
        recs.append((k * 100_000 + 99_999, far, hub, hub, 1.0, 1.0, "ACH", "train"))
    frame = pl.DataFrame(
        recs,
        schema={
            "row_id": pl.Int64,
            "minute": pl.Int64,
            "src": pl.Int32,
            "dst": pl.Int32,
            "amount_usd": pl.Float64,
            "amount_paid": pl.Float64,
            "payment_format": pl.String,
            "split": pl.String,
        },
        orient="row",
    )
    frame = (
        frame.sort(["minute", "row_id"])
        .with_row_index("rank")
        .with_columns(
            pl.col("rank").cast(pl.Int64),
            (pl.col("minute") // 1440 + 1).cast(pl.Int16).alias("day"),
        )
        .select(TX_COLUMNS)
    )
    con = connect(threads=2)
    try:
        register_transactions(con, frame)
        sql = compute_severities(con, RULES, 0)
    finally:
        con.close()
    want = sql.filter(pl.col("split") == "val_early").sort("row_id")
    ids = [k * 100_000 + ev[0] for k, (events, _) in enumerate(HISTORIES) for ev in events]
    order = np.argsort(np.asarray(ids))
    got = np.asarray([[r[SPEC.i_sev + j] for j in range(len(SCENARIOS))] for r in engine_rows])
    got = got[order]
    assert want.height == len(ids)
    exp = want.select(SCENARIOS).to_numpy()
    bad = np.flatnonzero((got.view(np.uint64) != exp.view(np.uint64)).any(axis=1))
    assert bad.size == 0, [
        (int(want["row_id"][i]), got[i].tolist(), exp[i].tolist()) for i in bad[:5]
    ]
