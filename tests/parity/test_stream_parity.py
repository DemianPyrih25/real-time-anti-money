"""The inproc stream reproduces the bundle's offline outputs bit for bit (M5 parity).

One run of the fixture slice through `open_runtime` (the scorer's real wiring) checks the online
parity summary, the minute-flush semantics, single-row versus batch predict and the scalar rule
hits; crafted events check that order violations stop before any engine state changes.
"""

from __future__ import annotations

import threading

import numpy as np
import polars as pl
import pytest

from aml.features.engine import Engine
from aml.features.spec import SEVERITY_COLUMNS
from aml.io import read_json
from aml.rules.sql_baseline import SCENARIOS, apply_thresholds
from aml.serving import bundle
from aml.serving.scorer import (
    OrderError,
    ParquetSource,
    open_runtime,
    rule_hits,
)
from tests.fixtures.serving_bundle import fixture_settings


class Collect:
    def __init__(self) -> None:
        self.scored: list = []
        self.timings: list = []

    def observe(self, ev, s, t) -> None:
        self.scored.append(s)
        self.timings.append(t)


@pytest.fixture(scope="module")
def stream(serving_bundle, fixture_tag, tmp_path_factory) -> dict:
    """The full fixture slice through the inproc runtime, with every Engine.advance recorded."""
    calls: list[int] = []
    real = Engine.advance

    def spy(self, minute):
        calls.append(minute)
        return real(self, minute)

    col = Collect()
    settings = fixture_settings(
        serving_bundle, tmp_path_factory.mktemp("parity") / "rt", fixture_tag
    )
    Engine.advance = spy
    try:
        rt = open_runtime(settings, observers=[col])
        result = rt.run(threading.Event())
        code = rt.close(result)
    finally:
        Engine.advance = real
    return {"rt": rt, "col": col, "calls": calls, "result": result, "code": code}


def test_full_slice_parity(stream, serving_bundle):
    rt = stream["rt"]
    n = read_json(serving_bundle / bundle.METADATA_FILE)["rows"]["slice"]
    assert rt.end_offset == n and rt.start_offset == 0 and rt.restored.origin == "bundle"
    assert stream["result"] == "end" and stream["code"] == 0
    par = rt.parity
    assert par["checked"] == n and par["covered"] == [0, n - 1] and par["error"] is None
    assert par["mismatches"] == dict.fromkeys(bundle.CHECKS, 0)
    assert par["digest_ok"] is True and par["alerts_ok"] is True
    a = par["alerts"]
    assert a["ref_count"] >= 1 and a["db_count"] == a["ref_count"]
    assert rt.log1p_ok is True
    assert rt.snapshot is None and not rt.store.snapshot_path.exists()  # the end: no snapshot
    assert len(stream["col"].scored) == n == rt.progress.events
    assert sum(s.alert for s in stream["col"].scored) == a["ref_count"] == rt.db_alerts


def test_flush_semantics(stream, serving_bundle):
    """One minute flush per new minute (> the restored clock), before process(), which never
    flushes; plus the final advance(last minute + 1)."""
    rt, col = stream["rt"], stream["col"]
    clock = rt.restored.header["clock"]
    minutes = pl.read_parquet(serving_bundle / bundle.SLICE, columns=["minute"])["minute"]
    new = sorted({m for m in minutes.to_list() if clock is None or m > clock})
    flushed = [s.minute for s in col.scored if s.flush is not None]
    assert flushed == new
    assert stream["calls"] == [*new, int(minutes[-1]) + 1]
    for s, t in zip(col.scored, col.timings, strict=True):
        if s.flush is None:
            assert t.flush_ns == 0 and t.n_applied == 0
        else:
            assert t.n_applied == s.flush.n_applied and t.flush_ns >= 0
        assert t.t_consume_ns <= t.t_features_ns <= t.t_model_ns <= t.t_done_ns


def test_single_row_predict_equals_batch(stream):
    rt, col = stream["rt"], stream["col"]
    x = np.concatenate([s.x for s in col.scored])
    assert x.dtype == np.float32 and x.shape == (len(col.scored), len(rt.champion.names))
    batch = np.asarray(rt.champion.booster.predict(x, num_threads=1), dtype=np.float64)
    single = np.array([s.score for s in col.scored], dtype=np.float64)
    assert np.array_equal(batch.view(np.uint64), single.view(np.uint64))


def test_rule_hits_equal_m1_apply_thresholds(stream, serving_bundle):
    rt, col = stream["rt"], stream["col"]
    i = rt.champion.spec.i_sev
    sev = np.array([s.row[i : i + len(SEVERITY_COLUMNS)] for s in col.scored], dtype=np.float64)
    sev_df = pl.DataFrame(
        {
            "row_id": [s.row_id for s in col.scored],
            **{c: sev[:, j] for j, c in enumerate(SEVERITY_COLUMNS)},
        }
    )
    rules = read_json(serving_bundle / bundle.THRESHOLDS)["rules"]
    for tag, thr in rules["thresholds"].items():
        fired = apply_thresholds(sev_df, thr)
        for k, r in enumerate(fired.iter_rows(named=True)):
            want = tuple(c for c in SCENARIOS if r[f"fired_{c}"])
            assert rule_hits(sev[k], thr) == want, (tag, k)
    head = rules["thresholds"][rules["headline_rate_tag"]]
    assert [s.rules for s in col.scored] == [rule_hits(row, head) for row in sev]


def test_rule_hits_edge_cases():
    sev = [3.0, 0.0, 0.5, 2.0, 0.0, 7.0, 1.0]
    thr = {
        "fan_in_velocity": 3,  # an int threshold, sev == thr: fires
        "fan_out_velocity": 0.0,  # sev == 0 never fires
        "rapid_pass_through": None,  # None never fires
        "round_trip": 2.0000001,
        "round_amount_burst": 7.0,
        "high_risk_format_burst": 1,
    }  # structuring absent: never fires
    want = ("fan_in_velocity", "round_amount_burst", "high_risk_format_burst")
    assert rule_hits(sev, thr) == want
    df = pl.DataFrame(
        {"row_id": [1], **{c: [v] for c, v in zip(SEVERITY_COLUMNS, sev, strict=True)}}
    )
    fired = apply_thresholds(df, {s: thr.get(s) for s in SCENARIOS}).row(0, named=True)
    assert tuple(s for s in SCENARIOS if fired[f"fired_{s}"]) == want


def _state(rt) -> tuple:
    sc = rt.scorer
    eng = sc.eng
    return eng.state_digest(), eng.next_rank, eng.pending_count, eng.clock, sc.expected_offset


def test_order_violations_leave_the_state_untouched(serving_bundle, fixture_tag, tmp_path):
    rt = open_runtime(fixture_settings(serving_bundle, tmp_path / "rt", fixture_tag))
    sc = rt.scorer
    evs = [rt.source.next(0) for _ in range(6)]
    for ev in evs[:3]:
        sc.step(ev)
    nxt = evs[3]
    before = _state(rt)
    clock = sc.eng.clock
    fields = list(nxt.fields)
    cases = [
        ("offset", evs[2]),  # duplicate
        ("offset", nxt._replace(offset=nxt.offset + 1)),  # gap
        ("rank", evs[4]._replace(offset=nxt.offset)),  # swapped ranks
        ("input", nxt._replace(fields=tuple([*fields[:5], -1.0, *fields[6:]]))),
        ("input", nxt._replace(fields=tuple(fields[:-1]))),
    ]
    if clock >= 1:  # an earlier minute
        cases.append(("late", nxt._replace(fields=tuple([*fields[:2], clock - 1, *fields[3:]]))))
    for kind, ev in cases:
        with pytest.raises(OrderError) as err:
            sc.step(ev)
        assert err.value.kind == kind, (kind, str(err.value))
        assert _state(rt) == before and sc.state_valid
    sc.step(nxt)  # the right event still scores
    assert sc.expected_offset == nxt.offset + 1
    rt.abort()


class ListSource:
    """A Kafka-like source over given events (offsets are message positions)."""

    def __init__(self, events: list) -> None:
        self.events = events
        self.i = 0
        self.exhausted = False

    def start(self, next_offset: int) -> None:
        self.i = next_offset
        self.exhausted = self.i >= len(self.events)

    def next(self, timeout_s: float):
        if self.i >= len(self.events):
            self.exhausted = True
            return None
        self.i += 1
        return self.events[self.i - 1]

    def high_watermark(self) -> int:
        return len(self.events)

    def close(self) -> None:
        pass


def test_run_stops_on_an_order_error_without_snapshot(serving_bundle, fixture_tag, tmp_path):
    meta = read_json(serving_bundle / bundle.METADATA_FILE)

    def factory(settings, n):
        """The slice as a topic whose messages 4 and 5 carry each other's events."""
        src = ParquetSource(settings.bundle_dir / bundle.SLICE, meta["next_rank"], n)
        src.start(0)
        evs = []
        while (ev := src.next(0)) is not None:
            evs.append(ev)
        evs[4], evs[5] = evs[5]._replace(offset=4), evs[4]._replace(offset=5)
        return ListSource(evs)

    settings = fixture_settings(serving_bundle, tmp_path / "rt", fixture_tag)
    rt = open_runtime(settings, source_factory=factory)
    result = rt.run(threading.Event())
    assert result == "order_error" and "rank" in rt.scorer.error
    assert rt.close(result) == 3
    assert rt.snapshot is None and not rt.store.snapshot_path.exists()
    assert rt.scorer.expected_offset == 4 and rt.parity["covered"] == [0, 3]
    assert rt.parity["mismatches"] == dict.fromkeys(bundle.CHECKS, 0)
