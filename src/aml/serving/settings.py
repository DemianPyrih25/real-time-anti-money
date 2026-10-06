"""M5 settings: `configs/serving.yaml` (validated strictly) plus paths from env and CLI.

Core module: stdlib + yaml only (no Kafka or web imports), so M6 can reuse it on Modal.

Precedence for every setting: CLI > environment > serving.yaml > code default. An empty
environment value counts as unset (compose writes `${VAR:-}`).
"""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import yaml

SHAPES = ("uniform", "trace", "unpaced")
TRANSPORTS = ("inproc", "kafka")

ENV_BUNDLE = "AML_BUNDLE_DIR"
ENV_RUNTIME = "AML_RUNTIME_DIR"
ENV_REPORTS = "AML_REPORTS_DIR"
ENV_CONFIG = "AML_CONFIG"
ENV_BOOTSTRAP = "KAFKA_BOOTSTRAP"
ENV_SCORER_URL = "SCORER_URL"
ENV_MAX_EVENTS = "REPLAY_MAX_EVENTS"
ENV_HOST_NOTE = "AML_HOST_NOTE"

DEFAULT_BUNDLE = "./serving"
DEFAULT_RUNTIME = "./runtime"
DEFAULT_REPORTS = "./reports"
DEFAULT_CONFIG = "./configs/serving.yaml"
DEFAULT_BOOTSTRAP = "localhost:9092"
DEFAULT_HOST = "0.0.0.0"
LATENCY_DIR = "latency"  # under the reports dir: the raw latency files (gitignored)


class ConfigError(ValueError):
    """An invalid serving.yaml or setting (the message names the dotted key); exit 2."""


# --- serving.yaml ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class Point:
    """One replay segment: the warm-up (index 0) or a measured point (1.. in config order)."""

    index: int
    shape: str  # uniform | trace | unpaced
    rate: float | None  # events/s; None iff unpaced
    events: int
    start: int = 0  # first offset (set by plan_points)

    @property
    def end(self) -> int:
        return self.start + self.events


@dataclass(frozen=True)
class KafkaConfig:
    transactions_topic: str
    alerts_topic: str
    group_id: str
    poll_timeout_s: float


@dataclass(frozen=True)
class ScorerConfig:
    port: int
    stop_timeout_s: float
    stall_after_s: float


@dataclass(frozen=True)
class ReplayerConfig:
    lead_s: float
    gap_cap_s: float
    barrier_timeout_s: float
    settle_s: float


@dataclass(frozen=True)
class LatencyConfig:
    warmup: Point
    points: tuple[Point, ...]
    min_tail_samples: int
    max_send_lag_ms: float
    min_rate_ratio: float
    e2e_p99_s: float


@dataclass(frozen=True)
class Targets:
    min_rate_ev_s: float
    champion_p95_s: float
    champion_p99_s: float


@dataclass(frozen=True)
class ServingConfig:
    max_events: int
    challenger: str
    alert_rate_tag: str
    kafka: KafkaConfig
    scorer: ScorerConfig
    replayer: ReplayerConfig
    latency: LatencyConfig
    targets: Targets

    @classmethod
    def from_dict(cls, d: Any) -> ServingConfig:
        """Validate a loaded serving.yaml; ConfigError names the first bad dotted key."""
        top = _section(d, "", ("replay", "challenger", "alert", "kafka", "scorer", "replayer",
                               "latency", "targets"))  # fmt: skip
        replay = _section(top["replay"], "replay", ("max_events",))
        max_events = _int(replay["max_events"], "replay.max_events", 1)
        if top["challenger"] != "none":
            raise ConfigError(
                f"challenger: {top['challenger']!r} is not supported; only 'none' (the LightGBM "
                "champion; a challenger is Stretch M4)"
            )
        alert = _section(top["alert"], "alert", ("rate_tag",))
        k = _section(top["kafka"], "kafka", ("transactions_topic", "alerts_topic", "group_id",
                                             "poll_timeout_s"))  # fmt: skip
        kafka = KafkaConfig(
            _str(k["transactions_topic"], "kafka.transactions_topic"),
            _str(k["alerts_topic"], "kafka.alerts_topic"),
            _str(k["group_id"], "kafka.group_id"),
            _num(k["poll_timeout_s"], "kafka.poll_timeout_s", positive=False),
        )
        s = _section(top["scorer"], "scorer", ("port", "stop_timeout_s", "stall_after_s"))
        port = _int(s["port"], "scorer.port", 1)
        if port > 65535:
            raise ConfigError(f"scorer.port: {port} is not a TCP port")
        scorer = ScorerConfig(
            port,
            _num(s["stop_timeout_s"], "scorer.stop_timeout_s"),
            _num(s["stall_after_s"], "scorer.stall_after_s"),
        )
        r = _section(top["replayer"], "replayer", ("lead_s", "gap_cap_s", "barrier_timeout_s",
                                                   "settle_s"))  # fmt: skip
        replayer = ReplayerConfig(
            _num(r["lead_s"], "replayer.lead_s", positive=False),
            _num(r["gap_cap_s"], "replayer.gap_cap_s"),
            _num(r["barrier_timeout_s"], "replayer.barrier_timeout_s"),
            _num(r["settle_s"], "replayer.settle_s", positive=False),
        )
        t = _section(top["targets"], "targets", ("min_rate_ev_s", "champion_p95_s",
                                                 "champion_p99_s"))  # fmt: skip
        targets = Targets(
            _num(t["min_rate_ev_s"], "targets.min_rate_ev_s"),
            _num(t["champion_p95_s"], "targets.champion_p95_s"),
            _num(t["champion_p99_s"], "targets.champion_p99_s"),
        )
        latency = _latency(top["latency"], max_events, targets)
        return cls(
            max_events=max_events,
            challenger="none",
            alert_rate_tag=_str(alert["rate_tag"], "alert.rate_tag"),
            kafka=kafka,
            scorer=scorer,
            replayer=replayer,
            latency=latency,
            targets=targets,
        )

    @classmethod
    def load(cls, path: Path) -> ServingConfig:
        path = Path(path)
        try:
            with path.open(encoding="utf-8") as f:
                doc = yaml.safe_load(f)
        except FileNotFoundError:
            raise ConfigError(f"{path}: serving config not found") from None
        except yaml.YAMLError as e:
            raise ConfigError(f"{path}: not valid YAML ({e})") from None
        return cls.from_dict(doc)


def _join(path: str, key: str) -> str:
    return f"{path}.{key}" if path else key


def _section(d: Any, path: str, keys: tuple[str, ...]) -> Mapping[str, Any]:
    """A mapping with exactly `keys` (unknown and missing keys are refused)."""
    where = path or "serving.yaml"
    if not isinstance(d, Mapping):
        raise ConfigError(f"{where}: expected a mapping, got {type(d).__name__}")
    unknown = [str(x) for x in d if x not in keys]
    if unknown:
        raise ConfigError(f"{_join(path, unknown[0])}: unknown key (allowed: {', '.join(keys)})")
    missing = [x for x in keys if x not in d]
    if missing:
        raise ConfigError(f"{_join(path, missing[0])}: missing")
    return d


def _int(v: Any, path: str, minimum: int) -> int:
    if type(v) is not int or v < minimum:
        raise ConfigError(f"{path}: expected an integer >= {minimum}, got {v!r}")
    return v


def _num(v: Any, path: str, *, positive: bool = True) -> float:
    ok = type(v) in (int, float) and math.isfinite(v) and (v > 0 if positive else v >= 0)
    if not ok:
        raise ConfigError(f"{path}: expected a number {'> 0' if positive else '>= 0'}, got {v!r}")
    return float(v)


def _str(v: Any, path: str) -> str:
    if not isinstance(v, str) or not v.strip():
        raise ConfigError(f"{path}: expected a non-empty string, got {v!r}")
    return v


def _point(d: Any, path: str, index: int) -> Point:
    p = _section(d, path, ("shape", "rate", "events"))
    shape = p["shape"]
    if shape not in SHAPES:
        raise ConfigError(f"{path}.shape: {shape!r} is not one of {', '.join(SHAPES)}")
    if shape == "unpaced":
        if p["rate"] is not None:
            raise ConfigError(f"{path}.rate: an unpaced point has rate null, got {p['rate']!r}")
        rate = None
    else:
        rate = _num(p["rate"], f"{path}.rate")
    return Point(index, shape, rate, _int(p["events"], f"{path}.events", 1))


def _latency(d: Any, max_events: int, targets: Targets) -> LatencyConfig:
    lat = _section(d, "latency", ("warmup", "points", "min_tail_samples", "max_send_lag_ms",
                                  "min_rate_ratio", "e2e_p99_s"))  # fmt: skip
    warmup = _point(lat["warmup"], "latency.warmup", 0)
    raw = lat["points"]
    if not isinstance(raw, list) or not raw:
        raise ConfigError("latency.points: expected a non-empty list")
    points = tuple(_point(p, f"latency.points[{i}]", i + 1) for i, p in enumerate(raw))
    names = ["latency.warmup", *(f"latency.points[{i}]" for i in range(len(points)))]
    seq = (warmup, *points)
    unpaced = [i for i, p in enumerate(seq) if p.shape == "unpaced"]
    if len(unpaced) > 1:
        raise ConfigError(f"{names[unpaced[1]]}.shape: at most one unpaced point")
    if unpaced and unpaced[0] != len(seq) - 1:
        raise ConfigError(f"{names[unpaced[0]]}.shape: the unpaced point must be the last one")
    total = sum(p.events for p in seq)
    if total != max_events:
        raise ConfigError(
            f"latency: warmup.events + sum(points.events) = {total} != replay.max_events "
            f"{max_events}"
        )
    if not any(p.shape == "uniform" and p.rate == targets.min_rate_ev_s for p in points):
        raise ConfigError(
            f"latency.points: no uniform point at targets.min_rate_ev_s {targets.min_rate_ev_s:g}"
        )
    ratio = _num(lat["min_rate_ratio"], "latency.min_rate_ratio")
    if ratio > 1:
        raise ConfigError(f"latency.min_rate_ratio: expected a ratio <= 1, got {ratio!r}")
    return LatencyConfig(
        warmup=warmup,
        points=points,
        min_tail_samples=_int(lat["min_tail_samples"], "latency.min_tail_samples", 1),
        max_send_lag_ms=_num(lat["max_send_lag_ms"], "latency.max_send_lag_ms"),
        min_rate_ratio=ratio,
        e2e_p99_s=_num(lat["e2e_p99_s"], "latency.e2e_p99_s"),
    )


def plan_points(cfg: ServingConfig, n: int) -> tuple[Point, ...]:
    """The warm-up and the points with contiguous offset ranges, in order, covering [0, n).

    n == max_events: the configured counts. n < max_events: each count becomes
    floor(events * n / max_events), the deficit goes to the last point and zero-size points are
    dropped (each keeps its config index). n > max_events raises.
    """
    total = cfg.max_events
    if type(n) is not int or n < 1:
        raise ConfigError(f"plan_points: n must be an integer >= 1, got {n!r}")
    if n > total:
        raise ConfigError(f"plan_points: {n} events > replay.max_events {total}")
    seq = (cfg.latency.warmup, *cfg.latency.points)
    counts = [p.events if n == total else p.events * n // total for p in seq]
    counts[-1] += n - sum(counts)
    out: list[Point] = []
    start = 0
    for p, c in zip(seq, counts, strict=True):
        if c:
            out.append(replace(p, events=c, start=start))
            start += c
    return tuple(out)


# --- settings (paths, env, CLI) -------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    """Everything a scorer, app, replayer or init run reads, resolved once."""

    cfg: ServingConfig
    transport: str  # inproc | kafka
    bundle_dir: Path
    runtime_dir: Path
    reports_dir: Path
    config_path: Path
    bootstrap: str
    scorer_url: str | None
    max_events_override: int | None  # CLI --max-events / REPLAY_MAX_EVENTS
    alert_rate_tag: str  # a thresholds.json rate tag, or "headline"
    exit_at_end: bool  # stop after the last event (inproc CLI) or keep serving (app)
    host: str
    port: int
    host_note: str | None  # AML_HOST_NOTE: free text for the report's hardware section
    case_packs: bool = True  # M6: build a case pack per alert (configs/explain.yaml)

    @property
    def transactions_topic(self) -> str:
        return self.cfg.kafka.transactions_topic

    @property
    def alerts_topic(self) -> str:
        return self.cfg.kafka.alerts_topic

    @property
    def group_id(self) -> str:
        return self.cfg.kafka.group_id

    @property
    def latency_dir(self) -> Path:
        return self.reports_dir / LATENCY_DIR

    def demo_events(self, slice_rows: int) -> int:
        """Events this run covers: min(replay.max_events, the override, the bundle's slice)."""
        n = min(self.cfg.max_events, slice_rows)
        if self.max_events_override is not None:
            n = min(n, self.max_events_override)
        if n < 1:
            raise ConfigError(f"no events to replay (slice rows {slice_rows}, max_events {n})")
        return n


def env_value(environ: Mapping[str, str], name: str) -> str | None:
    """The variable's value, or None when it is unset, empty or whitespace."""
    v = environ.get(name)
    return v if v is not None and v.strip() else None


def add_cli_args(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """The flags every M5 entry point shares (each overrides its env var / yaml value)."""
    p.add_argument("--bundle", help=f"serving bundle dir (env {ENV_BUNDLE}; {DEFAULT_BUNDLE})")
    p.add_argument("--runtime", help=f"runtime state dir (env {ENV_RUNTIME}; {DEFAULT_RUNTIME})")
    p.add_argument("--reports", help=f"reports dir (env {ENV_REPORTS}; {DEFAULT_REPORTS})")
    p.add_argument("--config", help=f"serving.yaml (env {ENV_CONFIG}; {DEFAULT_CONFIG})")
    p.add_argument("--bootstrap", help=f"Kafka host:port (env {ENV_BOOTSTRAP})")
    p.add_argument("--scorer-url", help=f"scorer base URL (env {ENV_SCORER_URL})")
    p.add_argument("--max-events", type=int, help=f"cap on events (env {ENV_MAX_EVENTS})")
    p.add_argument("--alert-rate-tag", help="thresholds.json rate tag (yaml alert.rate_tag)")
    return p


def load_settings(
    ns: argparse.Namespace | Mapping[str, Any] | None = None,
    environ: Mapping[str, str] | None = None,
    *,
    transport: str = "inproc",
    exit_at_end: bool | None = None,
) -> Settings:
    """Resolve settings: CLI values in `ns` (a Namespace or a dict with the argparse dest names:
    bundle, runtime, reports, config, bootstrap, scorer_url, max_events, alert_rate_tag, and
    optionally transport, exit_at_end, host, port, case_packs), then `environ` (default
    os.environ), then serving.yaml, then the defaults. `exit_at_end` defaults to True for inproc
    only; `case_packs` to True.
    """
    args: dict[str, Any] = dict(vars(ns) if isinstance(ns, argparse.Namespace) else (ns or {}))
    env = os.environ if environ is None else environ

    def pick(key: str, env_name: str | None, default: Any) -> Any:
        v = args.get(key)
        if v is not None:
            return v
        e = env_value(env, env_name) if env_name else None
        return e if e is not None else default

    config_path = Path(pick("config", ENV_CONFIG, DEFAULT_CONFIG))
    cfg = ServingConfig.load(config_path)
    tr = args.get("transport") or transport
    if tr not in TRANSPORTS:
        raise ConfigError(f"transport: {tr!r} is not one of {', '.join(TRANSPORTS)}")
    raw_max = pick("max_events", ENV_MAX_EVENTS, None)
    override = None
    if raw_max is not None:
        try:
            override = int(raw_max)
        except (TypeError, ValueError):
            raise ConfigError(
                f"{ENV_MAX_EVENTS} / --max-events: not an integer: {raw_max!r}"
            ) from None
        if override < 1:
            raise ConfigError(f"{ENV_MAX_EVENTS} / --max-events: must be >= 1, got {override}")
    end = args.get("exit_at_end")
    if end is None:
        end = exit_at_end if exit_at_end is not None else tr == "inproc"
    port = args.get("port")
    case_packs = args.get("case_packs")
    return Settings(
        cfg=cfg,
        transport=tr,
        bundle_dir=Path(pick("bundle", ENV_BUNDLE, DEFAULT_BUNDLE)),
        runtime_dir=Path(pick("runtime", ENV_RUNTIME, DEFAULT_RUNTIME)),
        reports_dir=Path(pick("reports", ENV_REPORTS, DEFAULT_REPORTS)),
        config_path=config_path,
        bootstrap=str(pick("bootstrap", ENV_BOOTSTRAP, DEFAULT_BOOTSTRAP)),
        scorer_url=pick("scorer_url", ENV_SCORER_URL, None),
        max_events_override=override,
        alert_rate_tag=str(args.get("alert_rate_tag") or cfg.alert_rate_tag),
        exit_at_end=bool(end),
        host=str(args.get("host") or DEFAULT_HOST),
        port=int(port) if port is not None else cfg.scorer.port,
        host_note=env_value(env, ENV_HOST_NOTE),
        case_packs=True if case_packs is None else bool(case_packs),
    )
