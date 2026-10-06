"""The Kafka transport's event source: partition 0 of the `transactions` topic, positioned by
`assign` at the restored snapshot's next_offset (never `subscribe`, never a committed offset).

Not a core module: it imports `confluent_kafka` at module level. The scorer imports it lazily
(`scorer.kafka_source`), on the Kafka transport only.

A fatal client error, an offset reset (the position left the log) or a vanished topic raise
`SourceError` (exit 1); a message from another partition raises `OrderError` and one that is not
the wire format `CodecError` (both exit 3). Offsets are committed asynchronously and best-effort,
only so that `kafka-consumer-groups` can show the lag.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from confluent_kafka import Consumer, KafkaError, KafkaException, TopicPartition

from aml.serving.scorer import Event, OrderError
from aml.serving.state import RuntimeStateError
from aml.streaming.codec import CodecError, decode_event, now_ns, unpack_meta
from aml.streaming.kafka import consumer_config

log = logging.getLogger(__name__)

HW_REFRESH_NS = 500_000_000  # the high watermark is re-read at most every 0.5 s
WARN_EVERY_NS = 1_000_000_000  # transient client errors: at most one warning per second
METADATA_TIMEOUT_S = 10.0
FATAL_CODES = frozenset(
    {
        KafkaError._AUTO_OFFSET_RESET,
        KafkaError.OFFSET_OUT_OF_RANGE,
        KafkaError.UNKNOWN_TOPIC_OR_PART,
        KafkaError._UNKNOWN_PARTITION,
        KafkaError._UNKNOWN_TOPIC,
    }
)


class SourceError(RuntimeError):
    """A fatal Kafka condition: the scorer stops with exit 1."""


class KafkaSource:
    """The `Source` of the Kafka transport (scorer.Source): one consumer, partition 0 only."""

    exhausted = False  # a topic never ends: the scorer stops at end_offset or on a signal

    def __init__(
        self,
        bootstrap: str,
        topic: str,
        *,
        group_id: str,
        poll_timeout_s: float,
        commit_interval_s: float = 1.0,
        consumer_factory: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        self.topic = topic
        self.poll_timeout_s = poll_timeout_s
        self.commit_interval_ns = int(commit_interval_s * 1e9)
        config = consumer_config(bootstrap, group_id=group_id, client_id="aml-scorer")
        config["on_commit"] = self._on_commit
        self._c = (consumer_factory or Consumer)(config)
        self._tp = TopicPartition(topic, 0)
        self._consumed: int | None = None  # the offset after the last returned message
        self._committed: int | None = None
        self._commit_ns = 0
        self._commit_warned = False
        self._warn_ns: int | None = None
        self._hw: int | None = None
        self._hw_ns: int | None = None
        self._closed = False

    def watermarks(self) -> tuple[int, int]:
        """(low, high) of partition 0, fresh; SourceError unless the topic has exactly it."""
        md = self._c.list_topics(self.topic, timeout=METADATA_TIMEOUT_S)
        tm = md.topics.get(self.topic)
        if tm is None or tm.error is not None:
            why = "not listed" if tm is None else tm.error
            raise SourceError(f"topic {self.topic!r} is not available ({why}): run init")
        if set(tm.partitions) != {0}:
            raise SourceError(
                f"topic {self.topic!r} has partitions {sorted(tm.partitions)}; the stream needs "
                "exactly one (run init)"
            )
        lo, hi = self._c.get_watermark_offsets(
            TopicPartition(self.topic, 0), timeout=METADATA_TIMEOUT_S, cached=False
        )
        return int(lo), int(hi)

    def start(self, next_offset: int) -> None:
        """Assign partition 0 at next_offset; refuse a topic that cannot hold this run's stream
        (RuntimeStateError, exit 2: the runtime state and the topic come from different inits)."""
        lo, hi = self.watermarks()
        if lo != 0:
            raise RuntimeStateError(
                f"topic {self.topic!r} starts at offset {lo}, not 0: it is not this run's "
                "replay (run init)"
            )
        if next_offset > hi:
            raise RuntimeStateError(
                f"the restored state resumes at offset {next_offset} but topic {self.topic!r} "
                f"ends at {hi}: the runtime state is from another run (run init)"
            )
        self._c.assign([TopicPartition(self.topic, 0, next_offset)])
        self._consumed = self._committed = next_offset
        self._hw, self._hw_ns = hi, now_ns()

    def next(self, timeout_s: float | None = None) -> Event | None:
        if self._consumed is None:
            raise RuntimeError("KafkaSource.start() was not called")
        self._maybe_commit()
        msg = self._c.poll(self.poll_timeout_s if timeout_s is None else timeout_s)
        t = now_ns()
        if msg is None:
            return None
        err = msg.error()
        if err is not None:
            if err.fatal() or err.code() in FATAL_CODES:
                raise SourceError(f"Kafka consumer: {err.name()}: {err.str()}")
            if err.code() != KafkaError._PARTITION_EOF:
                self._warn(err)
            return None
        if msg.partition() != 0:
            raise OrderError(
                "partition",
                f"a message from partition {msg.partition()} of {msg.topic()!r}: the stream "
                "reads partition 0 only",
            )
        off = msg.offset()
        d0 = now_ns()
        try:
            fields = decode_event(msg.value())
            meta = unpack_meta(msg.headers())
        except CodecError as e:
            raise CodecError(f"offset {off}: {e}") from e
        decode_ns = now_ns() - d0
        self._consumed = off + 1
        return Event(fields, off, meta, t, decode_ns)

    def high_watermark(self) -> int | None:
        """The partition's end offset as of the last fetch response (cached by the client),
        re-read at most every 0.5 s; one direct request while the client has none yet."""
        now = now_ns()
        if self._hw_ns is not None and now - self._hw_ns < HW_REFRESH_NS:
            return self._hw
        try:
            hi = self._c.get_watermark_offsets(self._tp, cached=True)[1]
            if hi is None or hi < 0:  # unknown until the first fetch response
                hi = self._c.get_watermark_offsets(self._tp, timeout=1.0, cached=False)[1]
            self._hw = int(hi)
        except KafkaException as e:
            self._warn(e)
        self._hw_ns = now
        return self._hw

    def _maybe_commit(self) -> None:
        if self._consumed is None or self._consumed == self._committed:
            return
        now = now_ns()
        if now - self._commit_ns < self.commit_interval_ns:
            return
        self._commit_ns = now
        self._commit(self._consumed)

    def _commit(self, offset: int) -> None:
        try:
            self._c.commit(offsets=[TopicPartition(self.topic, 0, offset)], asynchronous=True)
            self._committed = offset
        except KafkaException as e:  # best effort: lag monitoring only
            self._on_commit(e, None)

    def _on_commit(self, err: Any, partitions: Any) -> None:
        if err is not None and not self._commit_warned:
            self._commit_warned = True
            log.warning("offset commit failed (lag monitoring only; logged once): %s", err)

    def _warn(self, err: Any) -> None:
        now = now_ns()
        if self._warn_ns is None or now - self._warn_ns >= WARN_EVERY_NS:
            self._warn_ns = now
            log.warning("Kafka consumer: %s", err)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._consumed is not None and self._consumed != self._committed:
                self._commit(self._consumed)
        finally:
            self._c.close()
