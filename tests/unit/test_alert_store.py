"""The scorer's alert sinks: SQLite (WAL, FULL, idempotent, read-only readers) and Kafka."""

from __future__ import annotations

import sqlite3
import struct
import threading

from aml.serving.alerts import (
    ALERT_COLUMNS,
    KafkaAlertSink,
    SqliteAlertStore,
    count_alerts,
    read_alert,
    read_alerts,
)
from aml.streaming.codec import encode_alert


def _rec(row_id: int, rank: int, **kw) -> dict:
    rec = {
        "row_id": row_id,
        "rank": rank,
        "offset": rank - 100,
        "minute": 11530,
        "day": 9,
        "src": 3,
        "dst": 4,
        "amount_usd": 12345.678901234567,
        "payment_format": "ACH",
        "score": 0.1 + 0.2,
        "threshold": 0.25,
        "rate_tag": "0p005",
        "rules_fired": ["fan_in_velocity"],
        "severities": [3.0, 0.0, 0.1 + 0.7, 0.0, 0.0, 0.0, 5e-324],
        "model_version": "export-x:0123456789ab",
    }
    assert list(rec) == list(ALERT_COLUMNS)
    return {**rec, **kw}


def _bits(x: float) -> bytes:
    return struct.pack("<d", x)


def test_wal_and_full_sync(tmp_path):
    st = SqliteAlertStore(tmp_path / "rt" / "alerts.sqlite")
    try:
        assert st.conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert st.conn.execute("PRAGMA synchronous").fetchone()[0] == 2  # FULL
        assert st.flush(0) == 0 and st.failed == 0
    finally:
        st.close()


def test_bits_round_trip_and_duplicates_ignored(tmp_path):
    path = tmp_path / "alerts.sqlite"
    st = SqliteAlertStore(path)
    rec = _rec(7, 107)
    assert st.write(rec) is True
    assert st.write(rec) is False  # a replayed alert: INSERT OR IGNORE
    assert st.write({**rec, "score": 0.99}) is False  # the first write wins
    assert st.count() == 1 == count_alerts(path)
    got = read_alert(path, 7)
    assert got == rec
    for k in ("amount_usd", "score", "threshold"):
        assert _bits(got[k]) == _bits(rec[k]), k
    assert [_bits(x) for x in got["severities"]] == [_bits(x) for x in rec["severities"]]
    assert st.rows() == [rec]
    st.close()
    assert read_alert(path, 8) is None


def test_reader_sees_commits_while_the_writer_is_open(tmp_path):
    path = tmp_path / "alerts.sqlite"
    st = SqliteAlertStore(path)
    try:
        assert read_alerts(path) == [] and count_alerts(path) == 0
        st.write(_rec(1, 101))
        assert [a["row_id"] for a in read_alerts(path)] == [1]
        st.write(_rec(2, 102))
        assert count_alerts(path) == 2
    finally:
        st.close()


def test_order_limit_and_before_rank(tmp_path):
    path = tmp_path / "alerts.sqlite"
    st = SqliteAlertStore(path)
    for row_id, rank in ((50, 105), (10, 101), (90, 109), (30, 103)):
        st.write(_rec(row_id, rank))
    try:
        assert [a["rank"] for a in read_alerts(path)] == [109, 105, 103, 101]
        assert [a["rank"] for a in read_alerts(path, limit=2)] == [109, 105]
        assert [a["rank"] for a in read_alerts(path, before_rank=105)] == [103, 101]
        assert [a["rank"] for a in read_alerts(path, limit=1, before_rank=105)] == [103]
        assert len(read_alerts(path, limit=None)) == 4
        assert [a["rank"] for a in st.rows()] == [101, 103, 105, 109]
    finally:
        st.close()


def test_missing_database(tmp_path):
    path = tmp_path / "nothing" / "alerts.sqlite"
    assert read_alerts(path) == [] and count_alerts(path) == 0 and read_alert(path, 1) is None
    assert not path.exists()


def test_the_writer_belongs_to_one_thread(tmp_path):
    st = SqliteAlertStore(tmp_path / "alerts.sqlite")
    errors: list[BaseException] = []

    def other() -> None:
        try:
            st.write(_rec(1, 101))
        except BaseException as e:
            errors.append(e)

    t = threading.Thread(target=other)
    t.start()
    t.join()
    st.close()
    assert len(errors) == 1 and isinstance(errors[0], sqlite3.ProgrammingError)


class _FakeProducer:
    def __init__(self, config: dict) -> None:
        self.config = config
        self.sent: list[dict] = []
        self.full = 1  # the first produce() finds the local queue full
        self.callbacks: list = []
        self.polls = 0

    def produce(self, topic, value=None, key=None, partition=None, on_delivery=None):
        if self.full:
            self.full -= 1
            raise BufferError("queue full")
        self.sent.append({"topic": topic, "key": key, "value": value, "partition": partition})
        self.callbacks.append(on_delivery)

    def poll(self, timeout):
        self.polls += 1
        return 0

    def flush(self, timeout):
        return 2


def test_kafka_sink_with_a_fake_producer():
    fake: list[_FakeProducer] = []

    def factory(config: dict) -> _FakeProducer:
        fake.append(_FakeProducer(config))
        return fake[0]

    sink = KafkaAlertSink("broker:9092", "alerts-t", producer_factory=factory)
    rec = _rec(7, 107)
    assert sink.write(rec) is True
    p = fake[0]
    assert p.sent == [
        {"topic": "alerts-t", "key": b"7", "value": encode_alert(rec), "partition": 0}
    ]
    assert p.polls >= 2  # the BufferError retry polled, then the poll(0) after produce
    p.callbacks[0](None, object())
    assert sink.failed == 0
    p.callbacks[0](RuntimeError("broker gone"), object())
    assert sink.failed == 1
    assert sink.flush(10) == 2  # still outstanding: Runtime.close then skips the snapshot
