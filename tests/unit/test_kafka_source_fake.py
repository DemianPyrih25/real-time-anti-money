"""KafkaSource against a fake consumer (no broker): messages become Events, positioning is
`assign` only, fatal client errors stop the scorer and commits are best effort."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
from confluent_kafka import KafkaError, KafkaException

from aml.serving import consumer
from aml.serving.consumer import KafkaSource, SourceError
from aml.serving.scorer import Event, OrderError
from aml.serving.state import RuntimeStateError
from aml.streaming.codec import CodecError, Meta, encode_event, pack_meta

TOPIC = "tx-test"
FIELDS = (100, 7, 11530, 3, 4, 12.5, 0.1 + 0.2, "ACH", "US Dollar", "Euro", "B1", "B2")
INVALID = -1001  # librdkafka's "offset unknown"


class FakeMsg:
    def __init__(self, offset=0, value=None, headers=None, partition=0, error=None) -> None:
        self._offset = offset
        self._value = encode_event(FIELDS) if value is None else value
        self._headers = headers
        self._partition = partition
        self._error = error

    def error(self):
        return self._error

    def offset(self):
        return self._offset

    def partition(self):
        return self._partition

    def topic(self):
        return TOPIC

    def value(self):
        return self._value

    def headers(self):
        return self._headers


class FakeConsumer:
    def __init__(self, config: dict, messages=(), wm=(0, 10), partitions=(0,), topic_error=None):
        self.config = config
        self.messages = list(messages)
        self.wm = wm
        self.cached = (INVALID, INVALID)
        self.partitions = partitions
        self.topic_error = topic_error
        self.assigned: list | None = None
        self.commits: list = []
        self.commit_error = False
        self.wm_calls: list[bool] = []
        self.closed = False

    def list_topics(self, topic=None, timeout=-1):
        tm = SimpleNamespace(error=self.topic_error, partitions=dict.fromkeys(self.partitions))
        return SimpleNamespace(topics={topic: tm})

    def get_watermark_offsets(self, tp, timeout=-1, cached=False):
        assert (tp.topic, tp.partition) == (TOPIC, 0)
        self.wm_calls.append(cached)
        return self.cached if cached else self.wm

    def assign(self, partitions) -> None:
        self.assigned = list(partitions)

    def subscribe(self, *args, **kwargs) -> None:
        raise AssertionError("KafkaSource must position with assign(), never subscribe()")

    def poll(self, timeout=-1):
        return self.messages.pop(0) if self.messages else None

    def commit(self, offsets=None, asynchronous=True) -> None:
        if self.commit_error:
            raise KafkaException(KafkaError(KafkaError._NO_OFFSET))
        self.commits.append(([(tp.topic, tp.partition, tp.offset) for tp in offsets], asynchronous))

    def close(self) -> None:
        self.closed = True


def make(messages=(), *, commit_interval_s: float = 1.0, **kw) -> tuple[KafkaSource, FakeConsumer]:
    made: list[FakeConsumer] = []

    def factory(config: dict) -> FakeConsumer:
        made.append(FakeConsumer(config, messages, **kw))
        return made[0]

    src = KafkaSource(
        "broker:9092",
        TOPIC,
        group_id="aml-test",
        poll_timeout_s=0.1,
        commit_interval_s=commit_interval_s,
        consumer_factory=factory,
    )
    return src, made[0]


class Clock:
    """A settable stand-in for the module's monotonic clock."""

    def __init__(self) -> None:
        self.t = 10**12

    def __call__(self) -> int:
        return self.t


@pytest.fixture
def clock(monkeypatch) -> Clock:
    c = Clock()
    monkeypatch.setattr(consumer, "now_ns", c)
    return c


def test_consumer_config():
    _, fake = make()
    c = fake.config
    assert c["auto.offset.reset"] == "error"
    assert c["enable.auto.commit"] is False and c["enable.auto.offset.store"] is False
    assert c["group.id"] == "aml-test" and c["bootstrap.servers"] == "broker:9092"
    assert c["fetch.wait.max.ms"] == 10 and c["socket.nagle.disable"] is True
    assert callable(c["on_commit"]) and isinstance(c["logger"], logging.Logger)


def test_messages_become_events():
    headers = pack_meta(Meta(111, 222, 3))
    src, fake = make([FakeMsg(5, headers=headers), FakeMsg(6)])
    src.start(5)
    (tp,) = fake.assigned
    assert (tp.topic, tp.partition, tp.offset) == (TOPIC, 0, 5)
    ev = src.next(0.1)
    assert isinstance(ev, Event)
    assert ev.fields == FIELDS and ev.offset == 5 and ev.meta == Meta(111, 222, 3)
    assert ev.t_consume_ns > 0 and ev.decode_ns >= 0
    ev = src.next(0.1)
    assert ev.offset == 6 and ev.meta is None
    assert src.next(0.1) is None and src.exhausted is False
    src.close()
    assert fake.closed


@pytest.mark.parametrize(
    ("kw", "offset", "err", "match"),
    [
        ({"wm": (0, 10)}, 11, RuntimeStateError, "ends at 10"),
        ({"wm": (3, 10)}, 5, RuntimeStateError, "starts at offset 3"),
        ({"topic_error": KafkaError(KafkaError.UNKNOWN_TOPIC_OR_PART)}, 0, SourceError, "init"),
        ({"partitions": (0, 1)}, 0, SourceError, "exactly one"),
    ],
)
def test_start_refuses_a_topic_that_cannot_hold_the_stream(kw, offset, err, match):
    src, fake = make(**kw)
    with pytest.raises(err, match=match):
        src.start(offset)
    assert fake.assigned is None


def test_start_at_the_end_of_the_topic_is_allowed():
    src, fake = make(wm=(0, 10))
    src.start(10)  # a restart after the last message: nothing to read yet
    assert fake.assigned[0].offset == 10


@pytest.mark.parametrize(
    "error",
    [
        KafkaError(KafkaError._AUTO_OFFSET_RESET),
        KafkaError(KafkaError.OFFSET_OUT_OF_RANGE),
        KafkaError(KafkaError.UNKNOWN_TOPIC_OR_PART),
        KafkaError(KafkaError._UNKNOWN_PARTITION),
        KafkaError(KafkaError._TRANSPORT, "fenced", fatal=True),
    ],
)
def test_fatal_errors_raise_source_error(error):
    src, _ = make([FakeMsg(error=error)])
    src.start(0)
    with pytest.raises(SourceError):
        src.next(0.1)


def test_transient_errors_warn_at_most_once_per_second(clock, caplog):
    transport = KafkaError(KafkaError._TRANSPORT, "broker down")
    msgs = [FakeMsg(error=transport), FakeMsg(error=transport), FakeMsg(0)]
    msgs.insert(2, FakeMsg(error=KafkaError(KafkaError._PARTITION_EOF)))
    src, _ = make(msgs)
    src.start(0)
    with caplog.at_level(logging.WARNING, logger=consumer.__name__):
        assert [src.next(0.1) for _ in range(3)] == [None, None, None]
        assert src.next(0.1).offset == 0
        warned = [r for r in caplog.records if r.name == consumer.__name__]
        assert len(warned) == 1 and "broker down" in warned[0].getMessage()
        clock.t += 1_000_000_000
        src._c.messages.append(FakeMsg(error=transport))
        assert src.next(0.1) is None
        assert len([r for r in caplog.records if r.name == consumer.__name__]) == 2


def test_partition_other_than_zero_is_an_order_error():
    src, _ = make([FakeMsg(0, partition=1)])
    src.start(0)
    with pytest.raises(OrderError) as e:
        src.next(0.1)
    assert e.value.kind == "partition"


@pytest.mark.parametrize(
    ("value", "headers"),
    [
        (b'{"v":1}', None),
        (None, [("t_sched", b"\x00" * 8)]),  # partial latency headers
        (b"not json", None),
    ],
)
def test_a_message_that_is_not_the_wire_format_is_a_codec_error(value, headers):
    src, _ = make([FakeMsg(4, value=value, headers=headers)])
    src.start(4)
    with pytest.raises(CodecError, match="offset 4"):
        src.next(0.1)


def test_commits_are_asynchronous_and_best_effort(caplog):
    src, fake = make([FakeMsg(0), FakeMsg(1)], commit_interval_s=0.0)
    src.start(0)
    assert src.next(0.1).offset == 0 and fake.commits == []  # nothing consumed yet
    assert src.next(0.1).offset == 1
    assert fake.commits == [([(TOPIC, 0, 1)], True)]  # event 0 was scored before this poll
    fake.commit_error = True
    with caplog.at_level(logging.WARNING, logger=consumer.__name__):
        assert src.next(0.1) is None  # the failed commit is logged, never raised
        assert src.next(0.1) is None
    assert len([r for r in caplog.records if "commit" in r.getMessage()]) == 1
    fake.commit_error = False
    src.close()
    assert fake.commits[-1] == ([(TOPIC, 0, 2)], True) and fake.closed
    src.close()  # idempotent


def test_high_watermark_is_cached_and_refreshed(clock):
    src, fake = make(wm=(0, 10))
    src.start(0)
    assert fake.wm_calls == [False]
    assert src.high_watermark() == 10 and fake.wm_calls == [False]  # within 0.5 s: cached
    fake.wm = (0, 12)
    clock.t += 600_000_000
    assert src.high_watermark() == 12  # the client knows none yet: one direct request
    assert fake.wm_calls == [False, True, False]
    fake.cached = (0, 15)
    clock.t += 600_000_000
    assert src.high_watermark() == 15 and fake.wm_calls == [False, True, False, True]
