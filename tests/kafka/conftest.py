"""Fixtures of the Kafka integration tests (M5): a real broker at AML_TEST_KAFKA, else skipped.

AML_REQUIRE_KAFKA=1 (CI) turns a missing or unreachable broker into a failure instead of a skip.
Every test gets its own topics and consumer group, so tests never see each other's messages.
"""

from __future__ import annotations

import contextlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from tests.fixtures.serving_bundle import CONFIG_DIR

ENV_TEST = "AML_TEST_KAFKA"
ENV_REQUIRE = "AML_REQUIRE_KAFKA"
REQUIRED_WAIT_S = 120.0  # CI: the broker container may still be starting
OPTIONAL_WAIT_S = 10.0


@dataclass(frozen=True)
class Topics:
    tx: str
    alerts: str
    group: str


@pytest.fixture(scope="session")
def kafka_bootstrap() -> str:
    """The broker's host:port; skip without one, unless AML_REQUIRE_KAFKA=1 (then fail)."""
    bootstrap = os.environ.get(ENV_TEST, "").strip()
    required = os.environ.get(ENV_REQUIRE, "").strip() == "1"
    if not bootstrap:
        if required:
            pytest.fail(f"{ENV_REQUIRE}=1 but {ENV_TEST} is not set")
        pytest.skip(f"set {ENV_TEST}=localhost:9092 after `docker compose up -d kafka`")
    from aml.streaming import kafka

    if not required and not kafka.broker_reachable(bootstrap, 1.0):
        pytest.skip(f"no Kafka broker at {bootstrap} ({ENV_TEST})")
    try:
        kafka.wait_for_broker(bootstrap, REQUIRED_WAIT_S if required else OPTIONAL_WAIT_S)
    except TimeoutError as e:
        if required:
            pytest.fail(f"{ENV_REQUIRE}=1: {e}")
        pytest.skip(str(e))
    return bootstrap


@pytest.fixture
def topics(kafka_bootstrap: str):
    """Fresh, empty, uniquely named transactions and alerts topics; deleted afterwards."""
    from aml.streaming import kafka

    tag = uuid.uuid4().hex[:8]
    t = Topics(f"aml-t-{tag}-tx", f"aml-t-{tag}-alerts", f"aml-t-{tag}-group")
    kafka.reset_topics(kafka_bootstrap, [t.tx, t.alerts])
    yield t
    with contextlib.suppress(Exception):  # best effort: the names are never reused
        kafka.delete_topics(kafka_bootstrap, [t.tx, t.alerts], timeout_s=30)


@pytest.fixture
def kafka_config(topics: Topics, tmp_path: Path) -> Path:
    """configs/serving.yaml with this test's topics and consumer group."""
    with (CONFIG_DIR / "serving.yaml").open(encoding="utf-8") as f:
        doc = yaml.safe_load(f)
    doc["kafka"] |= {
        "transactions_topic": topics.tx,
        "alerts_topic": topics.alerts,
        "group_id": topics.group,
    }
    path = tmp_path / "serving.yaml"
    path.write_text(yaml.safe_dump(doc, sort_keys=False), encoding="utf-8")
    return path
