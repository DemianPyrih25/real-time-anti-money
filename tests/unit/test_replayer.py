"""The M5 replayer without a broker: the schedules, the open-loop send loop with a fake clock and
a fake producer, resume, the scorer barrier against a local HTTP server, and the payloads."""

from __future__ import annotations

import json
import shutil
import socket
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
import polars as pl
import pytest

from aml.features.spec import INPUT_COLUMNS
from aml.io import read_json
from aml.serving import bundle
from aml.serving.settings import Point, plan_points
from aml.streaming import replayer
from aml.streaming.codec import decode_event, encode_event, unpack_meta
from aml.streaming.kafka import producer_config
from aml.streaming.replayer import (
    ReplayError,
    load_slice,
    resume_index,
    run_replayer,
    schedule_ns,
    scorer_barrier,
    trace_offsets,
)
from tests.fixtures.serving_bundle import fixture_settings

T0 = 7_000_000_000
MS = 1_000_000


class FakeClock:
    """Monotonic ns that only moves when something sleeps or stalls."""

    def __init__(self) -> None:
        self.now = T0

    def __call__(self) -> int:
        return self.now

    def sleep(self, s: float) -> None:
        self.now += round(s * 1e9)


class FakeProducer:
    def __init__(self, clock: FakeClock, *, stall_at: int | None = None, full_at=()) -> None:
        self.clock = clock
        self.stall_at = stall_at  # this send takes 50 ms
        self.full = set(full_at)  # these sends find the queue full once
        self.sent: list[dict] = []
        self.flushes: list[int] = []  # messages sent when each flush ran

    def produce(self, topic, value=None, key=None, partition=-1, headers=None, on_delivery=None):
        i = len(self.sent)
        if i in self.full:
            self.full.discard(i)
            raise BufferError("queue full")
        self.sent.append(
            {"topic": topic, "value": value, "key": key, "partition": partition, "headers": headers}
        )
        on_delivery(None, None)
        if i == self.stall_at:
            self.clock.now += 50 * MS

    def poll(self, timeout: float = 0) -> int:
        self.clock.sleep(timeout)  # waiting for delivery reports takes time
        return 0

    def flush(self, timeout: float = -1) -> int:
        self.flushes.append(len(self.sent))
        return 0


def _settings(serving_bundle, fixture_tag, tmp_path, **cli):
    return fixture_settings(serving_bundle, tmp_path / "rt", fixture_tag, transport="kafka", **cli)


def _run(settings, producer, clock, **kw):
    kw.setdefault("tail_reader", lambda: (0, 0, None))
    return run_replayer(
        settings, producer_factory=lambda cfg: producer, clock=clock, sleep=clock.sleep, **kw
    )


def _metas(producer: FakeProducer) -> list:
    return [unpack_meta(m["headers"]) for m in producer.sent]


@pytest.fixture(scope="module")
def slice_df(serving_bundle) -> pl.DataFrame:
    return pl.read_parquet(serving_bundle / bundle.SLICE).select(*INPUT_COLUMNS)


# --- schedules ------------------------------------------------------------------------------------


def test_uniform_schedule_is_exact():
    s = schedule_ns(Point(1, "uniform", 76.0, 500, 0))
    assert s.dtype == np.int64
    assert s.tolist() == [round(k * 1e9 / 76.0) for k in range(500)]
    assert schedule_ns(Point(3, "uniform", 1000, 3, 9)).tolist() == [0, MS, 2 * MS]


def test_unpaced_schedule_is_all_zero():
    assert schedule_ns(Point(5, "unpaced", None, 7, 3)).tolist() == [0] * 7


def test_trace_schedule():
    # Minutes 10 (3 events), 11 (2), then a simulated idle gap of 589 minutes, 600 (2), 601 (3).
    minutes = np.array([10, 10, 10, 11, 11, 600, 600, 601, 601, 601], dtype=np.int64)
    sigma = trace_offsets(minutes, gap_cap_s=60.0)
    gaps = np.diff(sigma)
    assert sigma[0] == 0.0 and (gaps > 0).all()
    assert np.allclose(gaps[:2], 20.0)  # evenly spread inside minute 10: 60 / 3
    assert np.allclose(gaps[2:4], [25.0, 30.0])  # minute 10 -> 11, then 60 / 2 inside 11
    assert gaps[4] == 60.0 and gaps.max() == 60.0  # the long idle gap is cut to gap_cap_s
    assert np.allclose(gaps[5:], [30.0, 25.0, 20.0, 20.0])

    rate = 76.0
    p = Point(2, "trace", rate, 10, 0)
    s = schedule_ns(p, sigma)
    assert s[0] == 0 and (np.diff(s) >= 0).all()
    assert abs(int(s[-1]) - round(9 * 1e9 / rate)) <= 1  # spans exactly (n - 1) / R
    assert np.array_equal(s, schedule_ns(p, sigma))  # deterministic
    # The capped gap is the largest one, and the shape survives the rescaling.
    assert np.argmax(np.diff(s)) == 4
    sub = schedule_ns(Point(2, "trace", rate, 4, 5), sigma)  # a point inside the slice
    assert sub[0] == 0 and abs(int(sub[-1]) - round(3 * 1e9 / rate)) <= 1
    one_minute = schedule_ns(Point(2, "trace", rate, 3, 0), sigma)
    assert np.abs(one_minute - schedule_ns(Point(2, "uniform", rate, 3, 0))).max() <= 1
    assert schedule_ns(Point(2, "trace", rate, 1, 4), sigma).tolist() == [0]


def test_trace_offsets_on_the_fixture_slice(slice_df):
    minutes = slice_df["minute"].to_numpy()
    sigma = trace_offsets(minutes, gap_cap_s=60.0)
    assert sigma.shape == minutes.shape and (np.diff(sigma) > 0).all()
    assert np.diff(sigma).max() <= 60.0


# --- the send loop --------------------------------------------------------------------------------


def test_open_loop_a_stall_never_shifts_the_schedule(serving_bundle, fixture_tag, tmp_path):
    settings = _settings(serving_bundle, fixture_tag, tmp_path, max_events=10)
    clock = FakeClock()
    fake = FakeProducer(clock, stall_at=3)
    res = _run(settings, fake, clock, plan=(Point(1, "uniform", 100.0, 10, 0),))
    t = T0 + round(settings.cfg.replayer.lead_s * 1e9)
    metas = _metas(fake)
    assert [m.t_sched_ns for m in metas] == [t + k * 10 * MS for k in range(10)]
    lags = [(m.t_prod_ns - m.t_sched_ns) // MS for m in metas]
    assert lags == [0, 0, 0, 0, 40, 30, 20, 10, 0, 0]  # late events go at once, stamps kept
    assert {m.point for m in metas} == {1} and {m["partition"] for m in fake.sent} == {0}
    assert res.sent == 10 and res.complete and res.delivery_errors == 0 and not res.resumed
    (pt,) = res.points
    assert pt["send_lag_max_ms"] == 40.0 and pt["send_lag_p99_ms"] == 40.0
    assert pt["barrier_ok"] is None  # the first point waits for nothing


def test_buffer_error_retries_with_the_scheduled_stamp(serving_bundle, fixture_tag, tmp_path):
    settings = _settings(serving_bundle, fixture_tag, tmp_path, max_events=10)
    clock = FakeClock()
    fake = FakeProducer(clock, full_at={2})
    res = _run(settings, fake, clock, plan=(Point(1, "uniform", 100.0, 10, 0),))
    t = T0 + round(settings.cfg.replayer.lead_s * 1e9)
    metas = _metas(fake)
    assert len(fake.sent) == 10  # retried, not duplicated or dropped
    assert [m.t_sched_ns for m in metas] == [t + k * 10 * MS for k in range(10)]
    assert metas[2].t_prod_ns == metas[2].t_sched_ns + 10 * MS  # stamped at the retry
    assert res.points[0]["backpressure"] == 1


def test_payloads_are_the_encoded_slice(serving_bundle, fixture_tag, tmp_path, slice_df):
    settings = _settings(serving_bundle, fixture_tag, tmp_path)
    clock = FakeClock()
    fake = FakeProducer(clock)
    res = _run(settings, fake, clock, unpaced=True)
    n = read_json(serving_bundle / bundle.METADATA_FILE)["rows"]["slice"]
    assert res.n == n == slice_df.height and res.sent == n and res.complete
    assert [m["value"] for m in fake.sent] == [encode_event(r) for r in slice_df.iter_rows()]
    assert [m["key"] for m in fake.sent] == [str(r).encode() for r in slice_df["row_id"]]
    assert {m["topic"] for m in fake.sent} == {settings.transactions_topic}
    assert [decode_event(m["value"]) for m in fake.sent] == list(slice_df.iter_rows())
    assert {m.point for m in _metas(fake)} == {0}
    assert {m.t_sched_ns for m in _metas(fake)} == {T0 + round(0.5e9)}  # unpaced: one stamp


def test_the_configured_plan_with_barriers(serving_bundle, fixture_tag, tmp_path, slice_df):
    settings = _settings(serving_bundle, fixture_tag, tmp_path)
    minutes = slice_df["minute"].to_numpy()
    clock = FakeClock()
    fake = FakeProducer(clock)
    calls: list[tuple[int, int]] = []

    def barrier(last: int) -> bool:
        calls.append((last, fake.flushes[-1]))
        return True

    res = _run(settings, fake, clock, barrier=barrier)
    plan = plan_points(settings.cfg, res.n)
    assert len(plan) >= 2 and [d["index"] for d in res.points] == [p.index for p in plan]
    # Before every point but the first: a flush with every earlier event sent, then the barrier.
    assert calls == [(p.start - 1, p.start) for p in plan[1:]]
    assert res.sent == res.n and res.complete
    metas = _metas(fake)
    assert [m.point for m in metas] == [p.index for p in plan for _ in range(p.events)]
    assert all(d["send_lag_max_ms"] == 0.0 for d in res.points)  # the fake clock is exact
    sigma = trace_offsets(minutes, gap_cap_s=settings.cfg.replayer.gap_cap_s)
    for p in plan:  # each point's stamps follow its own schedule from its start
        got = [m.t_sched_ns for m in metas[p.start : p.end]]
        assert [g - got[0] for g in got] == schedule_ns(p, sigma).tolist()
    assert [d["barrier_ok"] for d in res.points] == [None] + [True] * (len(plan) - 1)


def test_stop_ends_the_loop_before_the_next_send(serving_bundle, fixture_tag, tmp_path):
    settings = _settings(serving_bundle, fixture_tag, tmp_path)
    clock = FakeClock()
    stop = threading.Event()

    class StopAfter(FakeProducer):
        def produce(self, *args, **kwargs):
            super().produce(*args, **kwargs)
            if len(self.sent) == 5:
                stop.set()

    fake = StopAfter(clock)
    res = _run(settings, fake, clock, unpaced=True, stop=stop)
    assert res.stopped and res.sent == 5 == len(fake.sent) and not res.complete


# --- resume ---------------------------------------------------------------------------------------


def test_resume_index():
    ranks = np.arange(100, 110, dtype=np.int64)
    assert resume_index(0, 0, 10, None, ranks) == 0
    assert resume_index(0, 4, 10, 103, ranks) == 4
    assert resume_index(0, 10, 10, 109, ranks) == 10  # complete
    for lo, hi, tail in ((0, 4, 104), (0, 4, None), (0, 11, 110), (1, 4, 103), (0, -1, None)):
        with pytest.raises(ReplayError):
            resume_index(lo, hi, 10, tail, ranks)


def test_resume_continues_at_the_high_watermark(serving_bundle, fixture_tag, tmp_path, slice_df):
    settings = _settings(serving_bundle, fixture_tag, tmp_path)
    n, k = slice_df.height, slice_df.height // 3
    base = int(slice_df["rank"][0])
    clock = FakeClock()
    fake = FakeProducer(clock)
    calls: list[int] = []
    res = _run(
        settings,
        fake,
        clock,
        unpaced=True,
        tail_reader=lambda: (0, k, base + k - 1),
        barrier=lambda last: calls.append(last) or True,
    )
    assert res.resumed and res.first_index == k and res.sent == n - k and res.complete
    assert calls == [k - 1] and fake.flushes[0] == 0  # drain the old backlog first
    assert [m["value"] for m in fake.sent] == [encode_event(r) for r in slice_df[k:].iter_rows()]

    made: list = []
    done = run_replayer(
        settings,
        unpaced=True,
        producer_factory=made.append,
        tail_reader=lambda: (0, n, base + n - 1),
    )
    assert done.complete and done.sent == 0 and done.first_index == n and made == []
    with pytest.raises(ReplayError, match="not this run"):
        _run(settings, FakeProducer(clock), clock, unpaced=True, tail_reader=lambda: (0, k, base))


def test_load_slice_refuses_a_slice_out_of_order(serving_bundle, tmp_path, slice_df):
    for name, df in (
        ("swapped", pl.concat([slice_df[1:2], slice_df[0:1], slice_df[2:]])),
        ("late", slice_df.with_columns(pl.col("minute").reverse())),
    ):
        d = tmp_path / name
        (d / "replay").mkdir(parents=True)
        shutil.copy(serving_bundle / bundle.METADATA_FILE, d / bundle.METADATA_FILE)
        df.write_parquet(d / bundle.SLICE)
        with pytest.raises(ReplayError):
            load_slice(d, slice_df.height)
    assert replayer.SLICE == bundle.SLICE and replayer.METADATA_FILE == bundle.METADATA_FILE


def test_producer_config():
    c = producer_config("broker:9092", client_id="x")
    assert c["enable.idempotence"] is True and c["acks"] == "all" and c["linger.ms"] == 0
    assert c["max.in.flight.requests.per.connection"] <= 5 and c["socket.nagle.disable"] is True


# --- barrier --------------------------------------------------------------------------------------


class _Health(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        srv = self.server
        code, body = srv.answers[0] if len(srv.answers) == 1 else srv.answers.pop(0)
        srv.hits += 1
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, *args) -> None:
        pass


@pytest.fixture
def health():
    """A local /health server answering a scripted list (the last answer repeats)."""
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _Health)
    srv.answers, srv.hits = [], 0
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    yield srv, f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def test_barrier_waits_through_503_bodies(health):
    srv, url = health
    srv.answers = [
        (503, {"status": "starting", "last_offset": None}),
        (503, b"<html>not json</html>"),
        (503, {"status": "stalled", "last_offset": 3}),
        (200, {"status": "ok", "last_offset": 4}),
        (200, {"status": "ok", "last_offset": 5}),
    ]
    assert scorer_barrier(url, 5, timeout_s=30, poll_s=0.01) is True
    assert srv.hits == 5


def test_barrier_times_out_or_stops(health):
    srv, url = health
    srv.answers = [(200, {"status": "ok", "last_offset": 2})]
    assert scorer_barrier(url, 5, timeout_s=0.2, poll_s=0.02) is False
    stop = threading.Event()
    stop.set()
    assert scorer_barrier(url, 5, timeout_s=30, poll_s=0.02, stop=stop) is False


def test_barrier_raises_when_the_scorer_failed(health):
    srv, url = health
    srv.answers = [(503, {"status": "failed", "exit_code": 2, "last_offset": None})]
    with pytest.raises(ReplayError, match="exit code 2"):
        scorer_barrier(url, 5, timeout_s=30, poll_s=0.01)


def test_barrier_retries_an_unreachable_scorer():
    with socket.socket() as s:  # a port nobody listens on
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    assert scorer_barrier(f"http://127.0.0.1:{port}", 0, timeout_s=0.2, poll_s=0.02) is False
