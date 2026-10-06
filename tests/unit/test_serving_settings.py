"""M5 settings: the shipped serving.yaml, strict validation, plan_points, CLI > env > yaml."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pytest
import yaml

from aml.serving.settings import (
    ENV_BOOTSTRAP,
    ENV_BUNDLE,
    ENV_CONFIG,
    ENV_MAX_EVENTS,
    ENV_RUNTIME,
    ConfigError,
    ServingConfig,
    add_cli_args,
    env_value,
    load_settings,
    plan_points,
)
from tests.conftest import CONFIG_DIR

CONFIG = CONFIG_DIR / "serving.yaml"


def _doc() -> dict:
    with CONFIG.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_shipped_yaml_loads():
    cfg = ServingConfig.load(CONFIG)
    lat = cfg.latency
    assert cfg.max_events == 100_000 and cfg.challenger == "none"
    assert cfg.alert_rate_tag == "headline"
    assert lat.warmup.index == 0 and [p.index for p in lat.points] == list(
        range(1, len(lat.points) + 1)
    )
    assert lat.warmup.events + sum(p.events for p in lat.points) == 100_000
    t = cfg.targets
    assert (t.min_rate_ev_s, t.champion_p95_s, t.champion_p99_s) == (76, 0.050, 1.0)
    assert any(p.shape == "uniform" and p.rate == 76 for p in lat.points)
    assert lat.points[-1].shape == "unpaced" and lat.points[-1].rate is None
    assert all(p.rate is not None for p in (lat.warmup, *lat.points[:-1]))


def _unpaced_first(d: dict) -> None:
    pts = d["latency"]["points"]
    pts.insert(0, pts.pop())


def _two_unpaced(d: dict) -> None:
    pts = d["latency"]["points"]
    pts.append(copy.deepcopy(pts[-1]))


def _no_target_point(d: dict) -> None:
    for p in d["latency"]["points"]:
        if p["shape"] == "uniform" and p["rate"] == d["targets"]["min_rate_ev_s"]:
            p["rate"] += 1


def _set(path: str, value: object):
    def mutate(d: dict) -> None:
        *head, last = path.split(".")
        node = d
        for k in head:
            node = node[int(k)] if isinstance(node, list) else node[k]
        if isinstance(node, list):
            node[int(last)] = value
        else:
            node[last] = value

    return mutate


def _delete(path: str):
    def mutate(d: dict) -> None:
        *head, last = path.split(".")
        node = d
        for k in head:
            node = node[k]
        del node[last]

    return mutate


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (_set("note", 1), r"^note: unknown key"),
        (_set("kafka.partitions", 1), r"^kafka\.partitions: unknown key"),
        (_set("latency.points.0.burst", 1), r"^latency\.points\[0\]\.burst: unknown key"),
        (_delete("scorer.port"), r"^scorer\.port: missing"),
        (_set("challenger", "onnx"), r"^challenger: 'onnx'"),
        (_unpaced_first, r"must be the last"),
        (_two_unpaced, r"at most one unpaced"),
        (_set("latency.points.0.shape", "burst"), r"^latency\.points\[0\]\.shape"),
        (_set("latency.points.0.rate", None), r"^latency\.points\[0\]\.rate"),
        (_set("latency.warmup.events", 1), r"replay\.max_events"),
        (_set("replay.max_events", 99_999), r"replay\.max_events"),
        (_set("replay.max_events", True), r"^replay\.max_events: expected an integer"),
        (_no_target_point, r"targets\.min_rate_ev_s"),
        (_set("scorer.stop_timeout_s", 0), r"^scorer\.stop_timeout_s"),
        (_set("latency.min_rate_ratio", 1.5), r"^latency\.min_rate_ratio"),
        (_set("kafka.alerts_topic", ""), r"^kafka\.alerts_topic"),
    ],
)
def test_invalid_config_is_refused(mutate, match):
    d = _doc()
    mutate(d)
    with pytest.raises(ConfigError, match=match):
        ServingConfig.from_dict(d)


def test_unpaced_point_rate_must_be_null():
    d = _doc()
    d["latency"]["points"][-1]["rate"] = 50
    with pytest.raises(ConfigError, match=r"\.rate: an unpaced point has rate null"):
        ServingConfig.from_dict(d)
    with pytest.raises(ConfigError, match="not found"):
        ServingConfig.load(CONFIG.with_name("absent.yaml"))


def test_plan_points_full_size():
    cfg = ServingConfig.load(CONFIG)
    seq = (cfg.latency.warmup, *cfg.latency.points)
    plan = plan_points(cfg, cfg.max_events)
    assert [p.index for p in plan] == [p.index for p in seq]
    assert [p.events for p in plan] == [p.events for p in seq]
    assert plan[0].start == 0 and plan[-1].end == cfg.max_events
    assert all(a.end == b.start for a, b in zip(plan, plan[1:], strict=False))
    assert [(p.shape, p.rate) for p in plan] == [(p.shape, p.rate) for p in seq]


@pytest.mark.parametrize("n", [1234, 1500, 99_999, 7, 1])
def test_plan_points_scaled(n):
    cfg = ServingConfig.load(CONFIG)
    seq = (cfg.latency.warmup, *cfg.latency.points)
    floors = {p.index: p.events * n // cfg.max_events for p in seq}
    floors[seq[-1].index] += n - sum(floors.values())  # the deficit goes to the last point
    plan = plan_points(cfg, n)
    assert sum(p.events for p in plan) == n and plan[0].start == 0 and plan[-1].end == n
    assert all(a.end == b.start for a, b in zip(plan, plan[1:], strict=False))
    assert {p.index: p.events for p in plan} == {i: c for i, c in floors.items() if c}
    assert all(p.events > 0 for p in plan)


def test_plan_points_refuses_more_than_max_events():
    cfg = ServingConfig.load(CONFIG)
    for n in (cfg.max_events + 1, 0):
        with pytest.raises(ConfigError):
            plan_points(cfg, n)


def test_env_value():
    env = {"A": "", "B": "  ", "C": "x"}
    assert env_value(env, "A") is None and env_value(env, "B") is None
    assert env_value(env, "C") == "x" and env_value(env, "D") is None


def test_precedence_cli_env_yaml():
    env = {
        ENV_CONFIG: str(CONFIG),
        ENV_BUNDLE: "/env/bundle",
        ENV_RUNTIME: "",
        ENV_MAX_EVENTS: " ",
        ENV_BOOTSTRAP: "kafka:19092",
    }
    s = load_settings({"bundle": "cli/bundle"}, env)
    assert s.bundle_dir == Path("cli/bundle")  # CLI over env
    assert s.runtime_dir == Path("./runtime")  # an empty env value counts as unset
    assert s.max_events_override is None  # so does whitespace
    assert s.bootstrap == "kafka:19092"  # env over the default
    assert s.alert_rate_tag == "headline" and s.port == 8000  # yaml
    assert s.transport == "inproc" and s.exit_at_end and s.host_note is None

    env2 = {ENV_CONFIG: str(CONFIG), ENV_MAX_EVENTS: "70"}
    s = load_settings({"alert_rate_tag": "0p01", "max_events": 50}, env2)
    assert s.alert_rate_tag == "0p01" and s.max_events_override == 50
    s = load_settings(None, env2, transport="kafka")
    assert s.max_events_override == 70 and not s.exit_at_end
    assert s.demo_events(60) == 60 and s.demo_events(1000) == 70
    assert s.transactions_topic == "transactions" and s.alerts_topic == "alerts"
    assert s.latency_dir == s.reports_dir / "latency"

    ns = add_cli_args(argparse.ArgumentParser()).parse_args(
        ["--bundle", "b", "--max-events", "5", "--config", str(CONFIG)]
    )
    s = load_settings(ns, {ENV_BUNDLE: "/env/bundle"})
    assert s.bundle_dir == Path("b") and s.max_events_override == 5 and s.demo_events(9) == 5

    for bad in ("x", "0"):
        with pytest.raises(ConfigError, match="max-events"):
            load_settings(None, {ENV_CONFIG: str(CONFIG), ENV_MAX_EVENTS: bad})
    with pytest.raises(ConfigError, match="transport"):
        load_settings({"config": CONFIG}, {}, transport="carrier-pigeon")
