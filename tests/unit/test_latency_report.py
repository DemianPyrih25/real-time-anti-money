"""M5 latency report: nearest-rank statistics, the verdict on synthetic runs, the rendered report
(no dates, no epoch values), and the recorder wired into a real inproc runtime on the fixture
bundle (its events stamped as the replayer would)."""

from __future__ import annotations

import json
import platform
import re
import sys
import threading
from types import SimpleNamespace

import numpy as np
import polars as pl
import pytest

from aml.io import read_json
from aml.serving import bundle
from aml.serving.latency import (
    EVENT_SCHEMA,
    EVENTS_FILE,
    REPORT_JSON,
    REPORT_MD,
    SESSIONS_FILE,
    LatencyRecorder,
    adequate,
    build_summary,
    distribution,
    empty_events,
    host_info,
    memory_now,
    nearest_rank,
    render_markdown,
)
from aml.serving.scorer import ParquetSource, Timing, open_runtime
from aml.serving.settings import ServingConfig, plan_points
from aml.streaming.codec import Meta
from tests.fixtures.serving_bundle import CONFIG_DIR, fixture_settings

CFG = ServingConfig.load(CONFIG_DIR / "serving.yaml")
N = CFG.max_events
MS = 1_000_000
TARGETS = [  # the measured uniform and trace points at targets.min_rate_ev_s
    p.index
    for p in plan_points(CFG, N)
    if p.index and p.shape in ("uniform", "trace") and p.rate == CFG.targets.min_rate_ev_s
]
SECTIONS = (
    "Verdict",
    "Method",
    "Rate points",
    "Saturation",
    f"Stages at {CFG.targets.min_rate_ev_s:g} ev/s",
    "Minute flush",
    "Memory",
    "Parity",
    "Hardware and software",
    "Caveats",
)
DATE = re.compile(r"\d{4}-\d{2}-\d{2}")
CALENDAR = re.compile(
    r"\b(?:January|February|March|April|May|June|July|August|September|October|November|"
    r"December|Jan|Feb|Mar|Apr|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|Monday|Tuesday|Wednesday|"
    r"Thursday|Friday|Saturday|Sunday)\b"
)


def synth(
    *,
    service_ns: int = 500_000,
    send_lag_ns: int = 200_000,
    transit_ns: int = 1_000_000,
    flush_every: int = 60,
    flush_ns: int = 3 * MS,
) -> pl.DataFrame:
    """A run of the shipped plan through one FIFO server: event k of a paced point is produced
    send_lag after its schedule and arrives transit later; the server spends service_ns on it,
    plus flush_ns (before it, like the minute flush) on every flush_every-th event. The unpaced
    point is produced every 20 us. Points follow each other after a drain and a settle."""
    decode, model, write = 20_000, 100_000, 50_000
    parts = []
    t0 = 10**12  # an arbitrary monotonic origin (ns)
    for p in plan_points(CFG, N):
        k = np.arange(p.events, dtype=np.int64)
        if p.rate is None:
            t_sched = np.full(p.events, t0, dtype=np.int64)
            t_prod = t0 + k * 20_000
        else:
            t_sched = t0 + np.round(k * (1e9 / p.rate)).astype(np.int64)
            t_prod = t_sched + send_lag_ns
        flush = np.where(k % flush_every == 0, flush_ns, 0).astype(np.int64)
        work = service_ns + flush
        cum = np.cumsum(work)
        # d_k = max(arrival_k, d_{k-1}) + work_k, vectorised.
        t_done = cum + np.maximum.accumulate(t_prod + transit_ns - (cum - work))
        t_consume = t_done - work
        t_features = t_consume + flush + service_ns - model - write
        parts.append(
            pl.DataFrame(
                {
                    "session": np.ones(p.events, dtype=np.int16),
                    "offset": p.start + k,
                    "point": np.full(p.events, p.index, dtype=np.int16),
                    "t_sched": t_sched,
                    "t_prod": t_prod,
                    "t_consume": t_consume,
                    "decode": np.full(p.events, decode, dtype=np.int64),
                    "flush": flush,
                    "n_applied": np.where(flush > 0, 40, 0).astype(np.int32),
                    "t_features": t_features,
                    "t_model": t_features + model,
                    "t_done": t_done,
                    "alert": k % 97 == 0,
                }
            )
        )
        t0 = int(t_done[-1]) + 2_500 * MS  # barrier, settle and lead before the next point
    return pl.concat(parts).cast(EVENT_SCHEMA)


def parity_doc(n: int = N, **over) -> dict:
    alerts = {"db_count": 7, "ref_count": 7, "missing": 0, "extra": 0, "score_mismatch": 0,
              "fired_mismatch": 0, "ok": True}  # fmt: skip
    doc = {
        "checked": n,
        "covered": [0, n - 1],
        "mismatches": dict.fromkeys(bundle.CHECKS, 0),
        "alerts_ok": True,
        "alerts": alerts,
        "digest_ok": True,
        "error": None,
    }
    return {**doc, **over}


def session(n: int = N, **over) -> dict:
    """A session record as LatencyRecorder writes it (one uninterrupted finished run)."""
    doc = {
        "session": 1,
        "status": "finished",
        "transport": "kafka",
        "origin": "bundle",
        "restore_s": 2.5,
        "start_offset": 0,
        "end_offset": n,
        "events": n,
        "alerts": 7,
        "finished": True,
        "final_flush": {"n_applied": 12, "n_expired": 345, "ms": 1.25},
        "parity": parity_doc(n),
        "log1p_ok": True,
        "snapshot": None,
        "memory": {
            "restore": {"vm_rss_mb": 900.0, "vm_hwm_mb": 950.0},
            "end": {"vm_rss_mb": 1000.0, "vm_hwm_mb": 1100.0},
        },
        "state_nbytes": {"restore": 360 * 2**20, "end": 362 * 2**20},
        "bundle": {
            "export_key": "export-test",
            "model_version": "export-test:0123456789ab",
            "alert_tag": "0p005",
            "threshold": 0.5,
            "slice_rows": n,
            "libc": "glibc 2.36",
        },
        "host": host_info({"AML_KAFKA_IMAGE": "apache/kafka:4.3.1"}),
    }
    return {**doc, **over}


def results(s: dict) -> list[str]:
    """The verdict table's results: run, generator, sustained, champion p95, p99, parity."""
    return [c["result"] for c in s["checks"]]


def with_champion(ev: pl.DataFrame, every: int, ns: int) -> pl.DataFrame:
    """Target-point events at offset % every == 0 get a champion path of exactly ns."""
    slow = pl.col("point").is_in(TARGETS) & (pl.col("offset") % every == 0)
    return ev.with_columns(
        pl.when(slow)
        .then(pl.col("t_consume") + pl.col("flush") + ns)
        .otherwise(pl.col("t_done"))
        .alias("t_done")
    )


def big_ints(x) -> list[int]:
    if isinstance(x, bool):
        return []
    if isinstance(x, int):
        return [x] if abs(x) > 1e15 else []
    if isinstance(x, dict):
        return [v for y in x.values() for v in big_ints(y)]
    if isinstance(x, list | tuple):
        return [v for y in x for v in big_ints(y)]
    return []


@pytest.fixture(scope="module")
def base() -> pl.DataFrame:
    return synth()


# --- statistics -----------------------------------------------------------------------------------


def test_nearest_rank_equals_numpy_inverted_cdf():
    rng = np.random.default_rng(7)
    for n in (*range(1, 64), 99, 100, 101, 199, 200, 999, 1000, 1001, 9_999, 10_000):
        x = np.sort(rng.integers(0, 50, n))  # with ties
        for q in (0.0, 0.01, 0.1, 0.25, 0.5, 0.9, 0.95, 0.99, 0.999, 1.0):
            assert nearest_rank(x, q) == np.quantile(x, q, method="inverted_cdf"), (n, q)


def test_adequacy_needs_min_tail_values_beyond_the_quantile():
    assert adequate(10_000, 0.999, 10) and not adequate(9_999, 0.999, 10)
    assert adequate(1_000, 0.99, 10) and not adequate(999, 0.99, 10)
    assert adequate(200, 0.95, 10) and not adequate(199, 0.95, 10)
    d = distribution(np.arange(9_999, dtype=np.int64) * MS, 10)
    assert d["n"] == 9_999 and d["p99.9"] is None and d["p99"] == 9_899.0
    assert d["max_ms"] == 9_998.0 and d["mean_ms"] == 4_999.0
    assert distribution(np.arange(10_000), 10)["p99.9"] is not None
    assert distribution(np.array([], dtype=np.int64), 10)["p50"] is None


# --- verdict --------------------------------------------------------------------------------------


def test_a_keep_up_run_passes(base):
    s = build_summary(CFG, base, [session()])
    assert s["verdict"] == {"overall": "PASS", "latency": "PASS", "parity": "PASS", "reasons": []}
    assert results(s) == ["PASS"] * 6
    pts = {p["index"]: p for p in s["points"]}
    assert TARGETS == [1, 2] and sorted(pts) == [1, 2, 3, 4, 5]  # the warm-up is not a point
    for i in TARGETS:
        p = pts[i]
        assert p["generator_ok"] and p["sustained"]
        assert p["achieved_ev_s"] == pytest.approx(CFG.targets.min_rate_ev_s, rel=0.01)
        assert p["champion_ms"]["max_ms"] == 0.5  # the minute flush is excluded
        assert p["e2e_ms"]["p50"] == 1.7  # from the scheduled send: lag + transit + service
        assert p["e2e_ms"]["max_ms"] == 4.7  # e2e includes the flush
    unpaced = pts[5]
    assert unpaced["sustained"] is None and unpaced["generator_ok"] is None
    assert unpaced["throughput_ev_s"] == pytest.approx(1e9 / (500_000 + 3 * MS / 60), rel=0.01)
    sat = s["saturation"]
    assert sat["sustained_up_to_ev_s"] == 1000 and sat["first_unsustained_ev_s"] is None
    st = s["stages"]
    assert st["points"] == TARGETS and st["n"] == 30_000
    assert set(st["items"]) == {
        "send_lag", "transit", "decode", "features", "model", "alert_write", "champion", "e2e",
        "alert_write_alerted",
    }  # fmt: skip
    assert st["items"]["send_lag"]["p99"] == 0.2 and st["items"]["transit"]["p50"] == 1.0
    assert s["flush"]["ms"]["max_ms"] == 3.0 and s["flush"]["n_applied_max"] == 40


def test_a_backlog_is_not_sustained():
    """20 ms per event at 76 ev/s (13.2 ms apart): each event's champion path is fine, but the
    queue grows, which only the e2e from the scheduled send shows."""
    s = build_summary(CFG, synth(service_ns=20 * MS), [session()])
    pts = {p["index"]: p for p in s["points"]}
    for i in TARGETS:
        p = pts[i]
        assert p["generator_ok"] and not p["sustained"]
        assert p["champion_ms"]["p99"] == 20.0
        assert p["e2e_ms"]["p99"] > 1_000.0
        assert p["achieved_ev_s"] < CFG.latency.min_rate_ratio * CFG.targets.min_rate_ev_s
    assert s["verdict"]["latency"] == "FAIL" and s["verdict"]["overall"] == "FAIL"
    assert results(s) == ["PASS", "PASS", "FAIL", "PASS", "PASS", "PASS"]
    assert s["saturation"]["sustained_up_to_ev_s"] is None
    assert s["saturation"]["first_unsustained_ev_s"] == CFG.targets.min_rate_ev_s


def test_champion_p95_at_the_target_passes_and_one_ns_more_fails(base):
    at = build_summary(CFG, with_champion(base, 10, 50 * MS), [session()])
    assert [p["champion_ms"]["p95"] for p in at["points"] if p["index"] in TARGETS] == [50.0] * 2
    assert at["verdict"]["latency"] == "PASS" and results(at) == ["PASS"] * 6
    over = build_summary(CFG, with_champion(base, 10, 50 * MS + 1), [session()])
    assert over["verdict"]["latency"] == "FAIL"
    assert results(over) == ["PASS", "PASS", "PASS", "FAIL", "PASS", "PASS"]


def test_champion_p99_over_the_target_fails(base):
    s = build_summary(CFG, with_champion(base, 50, 1_500 * MS), [session()])
    assert s["verdict"]["latency"] == "FAIL"
    r = results(s)
    assert r[3] == "PASS" and r[4] == "FAIL" and r[2] == "FAIL"  # e2e p99 > 1 s as well


@pytest.mark.parametrize(
    "case",
    ["restart", "killed_session", "send_lag", "missing", "duplicate", "no_headers", "off_plan"],
)
def test_runs_that_cannot_be_judged_are_invalid(base, case):
    ev, sessions = base, [session()]
    if case == "restart":
        sessions = [
            session(status="signal", finished=False, parity=parity_doc(covered=[0, 39_999])),
            session(session=2, origin="runtime", start_offset=39_000),
        ]
    elif case == "killed_session":  # its record says it started; it never wrote again
        sessions = [session(status="running", finished=False, parity=None), session(session=2)]
    elif case == "send_lag":  # the replayer 6 ms behind its schedule at a target point
        late = pl.col("point") == TARGETS[0]
        ev = ev.with_columns(
            pl.when(late)
            .then(pl.col("t_sched") + 6 * MS)
            .otherwise(pl.col("t_prod"))
            .alias("t_prod")
        )
    elif case == "missing":
        ev = ev.filter(~pl.col("offset").is_between(5_000, 5_009))
    elif case == "duplicate":
        ev = pl.concat([ev, ev.filter(pl.col("offset") == 5_000)])
    elif case == "no_headers":
        ev = ev.with_columns(
            [
                pl.when(pl.col("offset") < 10).then(None).otherwise(pl.col(c)).alias(c)
                for c in ("point", "t_sched", "t_prod")
            ]
        )
    elif case == "off_plan":
        ev = ev.with_columns(
            pl.when(pl.col("offset") == 2_000)
            .then(3)
            .otherwise(pl.col("point"))
            .cast(pl.Int16)
            .alias("point")
        )
    s = build_summary(CFG, ev, sessions)
    v = s["verdict"]
    assert v["latency"] == "INVALID" and v["reasons"], v
    assert v["overall"] in ("INVALID", "FAIL")
    if case in ("restart", "killed_session"):
        assert any("restarted" in r for r in v["reasons"])
        assert v["parity"] == "PASS" and v["overall"] == "INVALID"


def test_a_restore_after_the_end_is_not_a_restart(base):
    after = session(
        session=2,
        origin="runtime",
        start_offset=N,
        events=0,
        parity=parity_doc(checked=0, covered=None),
    )
    s = build_summary(CFG, base, [session(), after])
    assert s["verdict"]["overall"] == "PASS" and s["scoring_sessions"] == 1


@pytest.mark.parametrize(
    "change",
    [
        {"mismatches": {**dict.fromkeys(bundle.CHECKS, 0), "scores": 1}},
        {"alerts_ok": False},
        {"digest_ok": False},
        {"covered": [0, N - 2]},
        {"error": "RuntimeError('boom')"},
    ],
)
def test_parity_failures(base, change):
    s = build_summary(CFG, base, [session(parity=parity_doc(**change))])
    v = s["verdict"]
    assert v["parity"] == "FAIL" and v["latency"] == "PASS" and v["overall"] == "FAIL"
    assert results(s)[5] == "FAIL"


def test_a_digest_that_could_not_be_checked_is_not_a_failure(base):
    s = build_summary(CFG, base, [session(parity=parity_doc(digest_ok=None))])
    assert s["verdict"]["parity"] == "PASS" and s["parity"]["digest_ok"] is None


# --- rendering ------------------------------------------------------------------------------------


def test_report_has_every_section_and_no_dates_or_epoch_values(base):
    host = host_info({"AML_KAFKA_IMAGE": "apache/kafka:4.3.1", "AML_HOST_NOTE": "test host"})
    snap = {"next_offset": N, "seconds": 3.2, "bytes": 380 * 2**20}
    passing = build_summary(CFG, base, [session(host=host, snapshot=snap)])
    empty = build_summary(CFG, empty_events(), [])  # nothing recorded: renders, judged FAIL
    assert empty["verdict"]["overall"] == "FAIL" and empty["verdict"]["latency"] == "INVALID"
    for s, verdict in ((passing, "PASS"), (empty, "FAIL")):
        md = render_markdown(s)
        for h in SECTIONS:
            assert f"\n## {h}\n" in md, h
        assert f"**Verdict: {verdict}**" in md
        text = md + json.dumps(s, allow_nan=False)  # no NaN or inf either
        assert not DATE.search(text) and not CALENDAR.search(text)
        assert not big_ints(s)
    md = render_markdown(passing)
    assert "test host" in md and "apache/kafka:4.3.1" in md and "share this machine" in md


def test_host_info_and_memory():
    h = host_info({"AML_KAFKA_IMAGE": "apache/kafka:4.3.1", "AML_HOST_NOTE": "  "})
    assert h["kernel"] == platform.release() and h["python"] == platform.python_version()
    assert h["kafka_image"] == "apache/kafka:4.3.1" and h["note"] is None  # empty counts as unset
    assert h["versions"]["polars"] and h["versions"]["numpy"]
    m = memory_now()
    assert set(m) == {"vm_rss_mb", "vm_hwm_mb"}
    if sys.platform.startswith("linux") and m["vm_rss_mb"] is not None:
        assert m["vm_rss_mb"] > 0


# --- the recorder ---------------------------------------------------------------------------------


def test_recorder_columns_grow_and_keep_missing_headers_null():
    rec = LatencyRecorder(capacity=1)
    for off in range(3):
        meta = None if off == 1 else Meta(100 + off, 200 + off, 2)
        t = Timing(1_000 + off, 5, 7 if off == 0 else 0, 3 if off == 0 else 0, 1_100, 1_200, 1_300)
        rec.observe(SimpleNamespace(meta=meta), SimpleNamespace(offset=off, alert=off == 2), t)
    f = rec.frame()
    assert f.columns == list(EVENT_SCHEMA)
    assert all(f.schema[c] == t for c, t in EVENT_SCHEMA.items())
    assert f["offset"].to_list() == [0, 1, 2] and f["session"].to_list() == [1, 1, 1]
    assert f["point"].to_list() == [2, None, 2] and f["t_sched"].to_list() == [100, None, 102]
    assert f["t_prod"].to_list() == [200, None, 202] and f["alert"].to_list() == [
        False,
        False,
        True,
    ]
    assert f["flush"].to_list() == [7, 0, 0] and f["n_applied"].to_list() == [3, 0, 0]


class Stamped:
    """The inproc slice with the replayer's headers: each event scheduled 2 ms and produced
    1 ms before it is read, at its plan point."""

    def __init__(self, inner: ParquetSource, plan) -> None:
        self.inner = inner
        self.starts = np.array([p.start for p in plan], dtype=np.int64)
        self.index = [p.index for p in plan]

    @property
    def exhausted(self) -> bool:
        return self.inner.exhausted

    def start(self, next_offset: int) -> None:
        self.inner.start(next_offset)

    def next(self, timeout_s: float):
        ev = self.inner.next(timeout_s)
        if ev is None:
            return None
        point = self.index[int(np.searchsorted(self.starts, ev.offset, side="right")) - 1]
        t = ev.t_consume_ns
        return ev._replace(meta=Meta(t - 2 * MS, t - MS, point))

    def high_watermark(self) -> int | None:
        return self.inner.high_watermark()

    def close(self) -> None:
        self.inner.close()


def stamped(settings, n: int) -> Stamped:
    base = read_json(settings.bundle_dir / bundle.METADATA_FILE)["next_rank"]
    src = ParquetSource(settings.bundle_dir / bundle.SLICE, base, n)
    return Stamped(src, plan_points(settings.cfg, n))


class StopAt:
    def __init__(self, offset: int, stop: threading.Event) -> None:
        self.offset = offset
        self.stop = stop

    def observe(self, ev, s, t) -> None:
        if s.offset == self.offset:
            self.stop.set()


def test_recorder_in_a_runtime_writes_the_raw_files_and_the_report(
    serving_bundle, fixture_tag, tmp_path
):
    settings = fixture_settings(serving_bundle, tmp_path / "runtime", fixture_tag)
    rt = open_runtime(settings, observers=[LatencyRecorder()], source_factory=stamped)
    assert rt.close(rt.run(threading.Event())) == 0
    n = rt.end_offset
    ev = pl.read_parquet(settings.latency_dir / EVENTS_FILE)
    assert ev.columns == list(EVENT_SCHEMA) and ev["session"].unique().to_list() == [1]
    assert ev["offset"].to_list() == list(range(n)) and ev["point"].null_count() == 0
    assert int(ev["alert"].sum()) == rt.progress.alerts >= 1
    assert (ev["t_consume"] <= ev["t_features"]).all() and (ev["t_model"] <= ev["t_done"]).all()
    (doc,) = read_json(settings.latency_dir / SESSIONS_FILE)
    assert doc["status"] == "end" and doc["finished"] and doc["origin"] == "bundle"
    assert doc["events"] == n and doc["final_flush"] is not None and doc["snapshot"] is None
    assert doc["bundle"]["model_version"] == rt.champion.model_version
    summary = read_json(settings.reports_dir / REPORT_JSON)
    assert summary["events"] == n == summary["recorded"] and summary["sessions"] == 1
    assert summary["verdict"]["parity"] == "PASS" and summary["parity"]["covered"] == [[0, n - 1]]
    assert not big_ints(summary)
    md = (settings.reports_dir / REPORT_MD).read_text(encoding="utf-8")
    assert "## Verdict" in md and not DATE.search(md)


def test_recorder_counts_a_restart_and_parity_spans_the_sessions(
    serving_bundle, fixture_tag, tmp_path
):
    settings = fixture_settings(serving_bundle, tmp_path / "runtime", fixture_tag)
    n = read_json(serving_bundle / bundle.METADATA_FILE)["rows"]["slice"]
    stop = threading.Event()
    rt1 = open_runtime(
        settings, observers=[LatencyRecorder(), StopAt(n // 2, stop)], source_factory=stamped
    )
    assert rt1.close(rt1.run(stop)) == 0 and rt1.snapshot is not None
    assert not (settings.reports_dir / REPORT_JSON).exists()  # rendered only once the slice ends
    rt2 = open_runtime(settings, observers=[LatencyRecorder()], source_factory=stamped)
    assert rt2.restored.origin == "runtime"
    assert rt2.close(rt2.run(threading.Event())) == 0
    docs = read_json(settings.latency_dir / SESSIONS_FILE)
    assert [(d["session"], d["status"], d["origin"]) for d in docs] == [
        (1, "signal", "bundle"),
        (2, "end", "runtime"),
    ]
    ev = pl.read_parquet(settings.latency_dir / EVENTS_FILE)
    assert set(ev["offset"].to_list()) == set(range(n))
    assert ev["session"].unique().sort().to_list() == [1, 2]
    summary = read_json(settings.reports_dir / REPORT_JSON)
    v = summary["verdict"]
    assert v["latency"] == "INVALID" and any("restarted" in r for r in v["reasons"])
    assert v["parity"] == "PASS" and summary["parity"]["covered"] == [[0, n - 1]]
