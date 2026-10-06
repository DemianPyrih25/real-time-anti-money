"""The Kafka transport against a real broker (M5; skipped without AML_TEST_KAFKA).

Topic reset; the replayer (unpaced) into a fresh topic, then the Kafka-transport scorer to the
end: bit-exact parity, the final digest, the same SQLite rows as the inproc run and the
reference alert set on the alerts topic. A stop and restart of the scorer, and of the replayer,
changes nothing.
"""

from __future__ import annotations

import json
import threading
import time
import uuid

import polars as pl
import pytest
from confluent_kafka import Consumer, KafkaException, Producer, TopicPartition
from confluent_kafka.admin import AdminClient

from aml.features.spec import INPUT_COLUMNS
from aml.serving import bundle
from aml.serving.alerts import read_alerts
from aml.serving.scorer import open_runtime
from aml.serving.state import ALERTS_DB, init
from aml.streaming import kafka, replayer
from aml.streaming.codec import decode_event, encode_event, event_key, unpack_meta
from aml.streaming.replayer import run_replayer
from tests.fixtures.serving_bundle import fixture_settings

pytestmark = pytest.mark.kafka

RUN_TIMEOUT_S = 180.0  # a stuck consumer fails the test instead of hanging it


class Seen:
    """Observer: the offsets and latency headers the scorer consumed."""

    def __init__(self) -> None:
        self.offsets: list[int] = []
        self.metas: list = []

    def observe(self, ev, s, t) -> None:
        self.offsets.append(ev.offset)
        self.metas.append(ev.meta)


class StopAt:
    """Observer: request a stop after the event at `offset` (a SIGTERM there)."""

    def __init__(self, offset: int, stop: threading.Event) -> None:
        self.offset = offset
        self.stop = stop

    def observe(self, ev, s, t) -> None:
        if s.offset == self.offset:
            self.stop.set()


def _settings(serving_bundle, fixture_tag, tmp_path, bootstrap, config):
    return fixture_settings(
        serving_bundle,
        tmp_path / "rt",
        fixture_tag,
        transport="kafka",
        bootstrap=bootstrap,
        config=config,
        exit_at_end=True,
    )


def _run(settings, stop_at: int | None = None):
    stop = threading.Event()
    seen = Seen()
    observers = [seen] if stop_at is None else [seen, StopAt(stop_at, stop)]
    rt = open_runtime(settings, observers=observers)
    watchdog = threading.Timer(RUN_TIMEOUT_S, stop.set)
    watchdog.start()
    try:
        result = rt.run(stop)
    finally:
        watchdog.cancel()
    return rt, result, seen


def _rows(runtime_dir) -> list[dict]:
    return read_alerts(runtime_dir / ALERTS_DB, limit=None)[::-1]  # rank order


def _assert_union(parities: list[dict], n: int) -> None:
    """Every incarnation bit-exact; together they cover offsets [0, n - 1] without a hole."""
    spans = sorted(tuple(p["covered"]) for p in parities if p["covered"])
    reach = -1
    for lo, hi in spans:
        assert lo <= reach + 1, spans
        reach = max(reach, hi)
    assert spans[0][0] == 0 and reach == n - 1, spans
    for p in parities:
        assert p["mismatches"] == dict.fromkeys(bundle.CHECKS, 0) and p["error"] is None


def consume_all(bootstrap: str, topic: str, *, timeout_s: float = 60.0) -> list[tuple]:
    """Every message of partition 0 as (offset, key, value, headers)."""
    lo, hi = kafka.watermarks(bootstrap, topic)
    group = f"aml-t-read-{uuid.uuid4().hex[:8]}"
    c = Consumer(kafka.consumer_config(bootstrap, group_id=group, client_id="aml-test"))
    out: list[tuple] = []
    try:
        c.assign([TopicPartition(topic, 0, lo)])
        deadline = time.monotonic() + timeout_s
        while len(out) < hi - lo and time.monotonic() < deadline:
            msg = c.poll(0.5)
            if msg is None:
                continue
            if msg.error() is not None:
                raise KafkaException(msg.error())
            out.append((msg.offset(), msg.key(), msg.value(), msg.headers()))
    finally:
        c.close()
    assert len(out) == hi - lo, f"read {len(out)} of {hi - lo} messages of {topic}"
    return out


def _produce(bootstrap: str, topic: str, rows: list[tuple]) -> None:
    p = Producer(kafka.producer_config(bootstrap, client_id="aml-test"))
    for r in rows:
        p.produce(topic, value=encode_event(r), key=event_key(r[1]), partition=0)
    assert p.flush(30) == 0


@pytest.fixture(scope="module")
def slice_df(serving_bundle) -> pl.DataFrame:
    return pl.read_parquet(serving_bundle / bundle.SLICE).select(*INPUT_COLUMNS)


@pytest.fixture(scope="module")
def ref_alert_ids(serving_bundle, fixture_tag) -> set[int]:
    """The reference model alerts of the slice at the test's alert tag."""
    al = pl.read_parquet(serving_bundle / bundle.REF_ALERTS)
    ids = set(al.filter(pl.col(f"alert_{fixture_tag}"))["row_id"].to_list())
    assert ids, "fix the fixture: no reference alert"
    return ids


@pytest.fixture(scope="module")
def inproc_rows(kafka_bootstrap, serving_bundle, fixture_tag, tmp_path_factory) -> list[dict]:
    """The alerts table of an uninterrupted inproc run: what the Kafka runs must equal."""
    rt_dir = tmp_path_factory.mktemp("kafka_inproc") / "rt"
    rt = open_runtime(fixture_settings(serving_bundle, rt_dir, fixture_tag))
    result = rt.run(threading.Event())
    assert result == "end" and rt.close(result) == 0
    return _rows(rt_dir)


def _assert_fresh(bootstrap: str, names: list[str]) -> None:
    md = AdminClient(kafka.admin_config(bootstrap)).list_topics(timeout=10)
    for t in names:
        assert set(md.topics[t].partitions) == {0}
        assert kafka.watermarks(bootstrap, t) == (0, 0)


def test_reset_topics_leaves_one_empty_partition(
    kafka_bootstrap, topics, kafka_config, serving_bundle, fixture_tag, tmp_path, slice_df
):
    names = [topics.tx, topics.alerts]
    _produce(kafka_bootstrap, topics.tx, list(slice_df.head(3).iter_rows()))
    assert kafka.watermarks(kafka_bootstrap, topics.tx) == (0, 3)
    settings = _settings(serving_bundle, fixture_tag, tmp_path, kafka_bootstrap, kafka_config)
    stale = settings.runtime_dir / ALERTS_DB
    stale.parent.mkdir(parents=True)
    stale.write_bytes(b"stale")
    init(settings)  # the compose one-shot: runtime wiped, both topics recreated
    assert not stale.exists()
    _assert_fresh(kafka_bootstrap, names)
    kafka.reset_topics(kafka_bootstrap, names)  # right again: the deletion is asynchronous
    _assert_fresh(kafka_bootstrap, names)
    assert kafka.broker_reachable(kafka_bootstrap)


def test_replay_then_kafka_scorer_equals_the_references(
    kafka_bootstrap,
    topics,
    kafka_config,
    serving_bundle,
    fixture_tag,
    tmp_path,
    slice_df,
    inproc_rows,
    ref_alert_ids,
):
    settings = _settings(serving_bundle, fixture_tag, tmp_path, kafka_bootstrap, kafka_config)
    n = slice_df.height
    res = run_replayer(settings, unpaced=True)
    assert res.complete and res.sent == n and res.delivery_errors == 0 and not res.resumed
    msgs = consume_all(kafka_bootstrap, topics.tx)
    assert [m[2] for m in msgs] == [encode_event(r) for r in slice_df.iter_rows()]

    rt, result, seen = _run(settings)
    assert result == "end" and rt.close(result) == 0
    assert rt.restored.origin == "bundle" and rt.start_offset == 0
    par = rt.parity
    assert par["checked"] == n and par["covered"] == [0, n - 1]
    assert par["mismatches"] == dict.fromkeys(bundle.CHECKS, 0) and par["error"] is None
    assert par["digest_ok"] is True and par["alerts_ok"] is True
    assert seen.offsets == list(range(n))
    assert all(m is not None and m.point == 0 for m in seen.metas)  # the replayer's headers
    assert rt.progress.high_watermark == n
    assert _rows(tmp_path / "rt") == inproc_rows
    keys = {int(m[1]) for m in consume_all(kafka_bootstrap, topics.alerts)}
    assert keys == ref_alert_ids == {r["row_id"] for r in inproc_rows}


def test_scorer_stop_then_restart_is_identical(
    kafka_bootstrap,
    topics,
    kafka_config,
    serving_bundle,
    fixture_tag,
    tmp_path,
    slice_df,
    inproc_rows,
    ref_alert_ids,
):
    settings = _settings(serving_bundle, fixture_tag, tmp_path, kafka_bootstrap, kafka_config)
    n = slice_df.height
    assert run_replayer(settings, unpaced=True).complete
    minutes = slice_df["minute"].to_list()
    k = n // 2
    rt1, result, _ = _run(settings, stop_at=k)
    assert result == "signal" and rt1.close(result) == 0
    a = rt1.snapshot["next_offset"]
    assert a == minutes.index(minutes[k])  # the first pending event: k's minute is not applied

    rt2, result, seen = _run(settings)
    assert rt2.restored.origin == "runtime" and rt2.start_offset == a and seen.offsets[0] == a
    assert result == "end" and rt2.close(result) == 0
    assert rt2.parity["digest_ok"] is True and rt2.parity["alerts_ok"] is True
    _assert_union([rt1.parity, rt2.parity], n)
    assert _rows(tmp_path / "rt") == inproc_rows
    # At least once: the alerts of [a, k] went out twice; the key set is the reference.
    keys = [int(m[1]) for m in consume_all(kafka_bootstrap, topics.alerts)]
    assert set(keys) == ref_alert_ids and len(keys) >= len(ref_alert_ids)


class _StopAfter:
    """A real producer that requests a stop after `k` sends (a SIGTERM there)."""

    def __init__(self, inner: Producer, k: int, stop: threading.Event) -> None:
        self.inner = inner
        self.k = k
        self.stop = stop
        self.sent = 0

    def produce(self, *args, **kwargs) -> None:
        self.inner.produce(*args, **kwargs)
        self.sent += 1
        if self.sent == self.k:
            self.stop.set()

    def __getattr__(self, name: str):
        return getattr(self.inner, name)


def test_replayer_stop_then_resume_sends_each_event_once(
    kafka_bootstrap, topics, kafka_config, serving_bundle, fixture_tag, tmp_path, slice_df
):
    settings = _settings(serving_bundle, fixture_tag, tmp_path, kafka_bootstrap, kafka_config)
    n, k = slice_df.height, slice_df.height // 3
    stop = threading.Event()
    first = run_replayer(
        settings,
        unpaced=True,
        producer_factory=lambda cfg: _StopAfter(Producer(cfg), k, stop),
        stop=stop,
    )
    assert first.stopped and first.sent == k and first.delivery_errors == 0
    assert not first.complete
    assert kafka.watermarks(kafka_bootstrap, topics.tx) == (0, k)

    second = run_replayer(settings, unpaced=True)
    assert second.resumed and second.first_index == k and second.sent == n - k
    assert second.complete and second.delivery_errors == 0
    msgs = consume_all(kafka_bootstrap, topics.tx)
    base = int(slice_df["rank"][0])
    assert [m[0] for m in msgs] == list(range(n))
    assert [m[1] for m in msgs] == [str(r).encode() for r in slice_df["row_id"]]
    assert [decode_event(m[2])[0] for m in msgs] == list(range(base, base + n))
    assert {unpack_meta(m[3]).point for m in msgs} == {0}

    again = run_replayer(settings, unpaced=True)
    assert again.complete and again.sent == 0 and again.first_index == n


def test_replayer_cli_and_foreign_topics(
    kafka_bootstrap, topics, kafka_config, serving_bundle, tmp_path, slice_df, capsys
):
    n = slice_df.height
    argv = [
        "--bundle",
        str(serving_bundle),
        "--config",
        str(kafka_config),
        "--bootstrap",
        kafka_bootstrap,
        "--runtime",
        str(tmp_path / "rt"),
        "--reports",
        str(tmp_path / "reports"),
        "--unpaced",
    ]
    assert replayer.main(argv) == 0
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert out["exit_code"] == 0 and out["sent"] == n and out["complete"] is True
    assert out["points"][0]["send_lag_p99_ms"] is not None

    rows = list(slice_df.iter_rows())
    _produce(kafka_bootstrap, topics.tx, rows[:1])  # one more message than the replay has
    assert replayer.main(argv) == 1
    kafka.reset_topics(kafka_bootstrap, [topics.tx])
    _produce(kafka_bootstrap, topics.tx, rows[1:2])  # offset 0 holds the slice's second event
    assert replayer.main(argv) == 1
    assert kafka.watermarks(kafka_bootstrap, topics.tx) == (0, 1)  # nothing was added
