"""Kafka helpers of the M5 demo: client configs, broker probes, topic reset, watermarks.

`confluent_kafka` is imported inside the functions that talk to a broker, so the replayer (a core
module) can import this one without it. Every client logs through the `aml.kafka.librdkafka`
logger at WARNING, so a missing broker does not flood the output.

Both topics have exactly one partition (offset = position in the replay) and infinite retention:
`reset_topics` deletes and recreates them on every `docker compose up` (init).
"""

from __future__ import annotations

import logging
import socket
import time
import uuid
from collections.abc import Sequence
from typing import Any

log = logging.getLogger(__name__)

LIBRDKAFKA_LOGGER = "aml.kafka.librdkafka"
_rdkafka_log = logging.getLogger(LIBRDKAFKA_LOGGER)
_rdkafka_log.setLevel(logging.WARNING)

DEFAULT_PORT = 9092
TOPIC_CONFIG = {"cleanup.policy": "delete", "retention.ms": "-1", "retention.bytes": "-1"}
RETRY_S = 0.5  # topic deletion and creation are asynchronous on the broker


# --- client configs -------------------------------------------------------------------------------


def producer_config(bootstrap: str, *, client_id: str) -> dict[str, Any]:
    """Idempotent, acks=all and no batching delay: retries never duplicate or reorder."""
    return {
        "bootstrap.servers": bootstrap,
        "client.id": client_id,
        "enable.idempotence": True,
        "acks": "all",
        "linger.ms": 0,
        "max.in.flight.requests.per.connection": 5,
        "socket.nagle.disable": True,
        "message.timeout.ms": 120_000,
        "queue.buffering.max.messages": 200_000,
        "logger": _rdkafka_log,
    }


def consumer_config(bootstrap: str, *, group_id: str, client_id: str) -> dict[str, Any]:
    """Positioned by `assign` only: no auto commit and no offset reset (a bad position is an
    error, never a silent jump). Committed offsets are for lag monitoring only."""
    return {
        "bootstrap.servers": bootstrap,
        "client.id": client_id,
        "group.id": group_id,
        "enable.auto.commit": False,
        "enable.auto.offset.store": False,
        "auto.offset.reset": "error",
        "enable.partition.eof": False,
        "fetch.wait.max.ms": 10,
        "socket.nagle.disable": True,
        "isolation.level": "read_uncommitted",
        "logger": _rdkafka_log,
    }


def admin_config(bootstrap: str) -> dict[str, Any]:
    return {"bootstrap.servers": bootstrap, "client.id": "aml-admin", "logger": _rdkafka_log}


def _probe_config(bootstrap: str) -> dict[str, Any]:
    # A throwaway group: it never subscribes or commits, librdkafka just requires one.
    return consumer_config(
        bootstrap, group_id=f"aml-probe-{uuid.uuid4().hex[:8]}", client_id="aml-probe"
    )


# --- broker ---------------------------------------------------------------------------------------


def _first_host_port(bootstrap: str) -> tuple[str, int]:
    first = bootstrap.split(",")[0].strip()
    if "://" in first:
        first = first.split("://", 1)[1]
    host, sep, port = first.rpartition(":")
    if not sep or not port.isdigit():  # no port given (a bare host or an IPv6 address)
        return first.strip("[]") or "localhost", DEFAULT_PORT
    return host.strip("[]") or "localhost", int(port)


def broker_reachable(bootstrap: str, timeout_s: float = 1.0) -> bool:
    """A TCP connection to the first bootstrap address succeeds (stdlib only, no client)."""
    try:
        with socket.create_connection(_first_host_port(bootstrap), timeout=timeout_s):
            return True
    except OSError:
        return False


def wait_for_broker(bootstrap: str, timeout_s: float = 120.0) -> None:
    """Block until the broker answers a metadata request; TimeoutError after timeout_s."""
    from confluent_kafka import KafkaException
    from confluent_kafka.admin import AdminClient

    deadline = time.monotonic() + timeout_s
    admin = AdminClient(admin_config(bootstrap))
    last: Exception | None = None
    while True:
        left = deadline - time.monotonic()
        try:
            admin.list_topics(timeout=max(0.5, min(5.0, left)))
            return
        except KafkaException as e:
            last = e
        if time.monotonic() >= deadline:
            raise TimeoutError(f"no Kafka broker at {bootstrap} after {timeout_s:g} s ({last})")
        time.sleep(1.0)


# --- topics ---------------------------------------------------------------------------------------


def _error(e: Exception) -> Any:
    """The KafkaError inside a KafkaException (None for other exceptions)."""
    arg = e.args[0] if e.args else None
    return arg if hasattr(arg, "code") else None


def _present(admin: Any, topics: Sequence[str]) -> list[str]:
    listed = admin.list_topics(timeout=10).topics
    return [t for t in topics if t in listed]


def delete_topics(bootstrap: str, topics: Sequence[str], *, timeout_s: float = 60.0) -> None:
    """Delete the topics that exist and wait until the broker no longer lists them."""
    from confluent_kafka import KafkaError, KafkaException
    from confluent_kafka.admin import AdminClient

    deadline = time.monotonic() + timeout_s
    admin = AdminClient(admin_config(bootstrap))
    present = _present(admin, topics)
    if present:
        for t, fut in admin.delete_topics(present, operation_timeout=30).items():
            try:
                fut.result()
            except KafkaException as e:
                err = _error(e)
                if err is None or err.code() != KafkaError.UNKNOWN_TOPIC_OR_PART:
                    raise
                log.info("topic %s was already gone", t)
    while present := _present(admin, present):
        if time.monotonic() >= deadline:
            raise TimeoutError(f"topics {present} still listed {timeout_s:g} s after deletion")
        time.sleep(RETRY_S)


def reset_topics(bootstrap: str, topics: Sequence[str], *, timeout_s: float = 120.0) -> None:
    """Delete the topics if present, then create each with 1 partition (replication 1, infinite
    retention) and verify that it is empty: watermarks (0, 0)."""
    from confluent_kafka import KafkaError, KafkaException
    from confluent_kafka.admin import AdminClient, NewTopic

    deadline = time.monotonic() + timeout_s
    delete_topics(bootstrap, topics, timeout_s=timeout_s)
    admin = AdminClient(admin_config(bootstrap))
    for t in topics:
        while True:
            new = NewTopic(t, num_partitions=1, replication_factor=1, config=dict(TOPIC_CONFIG))
            try:
                admin.create_topics([new], operation_timeout=30)[t].result()
                break
            except KafkaException as e:
                err = _error(e)
                again = err is not None and (
                    err.code() == KafkaError.TOPIC_ALREADY_EXISTS or err.retriable()
                )
                if not again or time.monotonic() >= deadline:
                    raise
            time.sleep(RETRY_S)  # the deletion has not finished on the broker yet
    for t in topics:
        _verify_empty(admin, bootstrap, t, deadline)


def _verify_empty(admin: Any, bootstrap: str, topic: str, deadline: float) -> None:
    """Exactly partition 0 and watermarks (0, 0), once the new partition has a leader."""
    from confluent_kafka import KafkaException

    last: Any = None
    while True:
        try:
            tm = admin.list_topics(topic, timeout=10).topics.get(topic)
            if tm is not None and tm.error is None:
                if set(tm.partitions) != {0}:
                    raise RuntimeError(f"topic {topic} has partitions {sorted(tm.partitions)}")
                wm = watermarks(bootstrap, topic)
                if wm != (0, 0):
                    raise RuntimeError(f"topic {topic} is not empty after reset: {wm}")
                return
            last = None if tm is None else tm.error
        except KafkaException as e:  # e.g. no leader yet for the new partition
            last = e
        if time.monotonic() >= deadline:
            raise TimeoutError(f"topic {topic} not ready after reset ({last})")
        time.sleep(RETRY_S)


def watermarks(bootstrap: str, topic: str, *, timeout_s: float = 10.0) -> tuple[int, int]:
    """(low, high) offsets of partition 0, read fresh from the broker."""
    from confluent_kafka import Consumer, TopicPartition

    c = Consumer(_probe_config(bootstrap))
    try:
        lo, hi = c.get_watermark_offsets(TopicPartition(topic, 0), timeout=timeout_s, cached=False)
        return int(lo), int(hi)
    finally:
        c.close()


def read_at(
    bootstrap: str, topic: str, offset: int, *, timeout_s: float = 10.0
) -> tuple[bytes | None, bytes | None, list | None] | None:
    """(key, value, headers) of the message at `offset` of partition 0; None if none arrives
    within timeout_s."""
    from confluent_kafka import Consumer, KafkaException, TopicPartition

    c = Consumer(_probe_config(bootstrap))
    try:
        c.assign([TopicPartition(topic, 0, offset)])
        deadline = time.monotonic() + timeout_s
        while (left := deadline - time.monotonic()) > 0:
            msg = c.poll(min(left, 1.0))
            if msg is None:
                continue
            if msg.error() is not None:
                raise KafkaException(msg.error())
            return msg.key(), msg.value(), msg.headers()
        return None
    finally:
        c.close()
