"""M5 latency: the per-event recorder, the statistics and verdict, reports/latency.{md,json}.

Core module: numpy, polars and stdlib only (no Kafka or web imports).

`LatencyRecorder` is a scorer observer (the app attaches it on the Kafka transport, whose messages
carry the replayer's scheduled and produce stamps). It keeps one row of monotonic stamps per scored
event and writes the raw files under reports/latency/: `events.parquet` and `sessions.json`, one
record per scorer session of this run. A session's record is written when it starts, so a session
that is killed later still counts as a restart. At the end of the slice it renders
reports/latency.md and reports/latency.json; `python -m aml.serving.latency report` re-renders them
from the raw files.

Stages (ns, reported in ms):

    send_lag    = t_prod - t_sched                  generator health
    transit     = t_consume - t_prod                client, broker, fetch, consumer backlog
    decode      = decode
    features    = t_features - t_consume - decode - flush
    model       = t_model - t_features
    alert_write = t_done - t_model
    champion    = t_done - t_consume - flush        PLAN §4 "consume -> alert write"
    e2e         = t_done - t_sched                  includes queueing and the minute flush

Quantiles are nearest rank (`np.quantile(..., method="inverted_cdf")`), reported only when
n * (1 - q) >= min_tail_samples; there are no confidence intervals. Durations go out as ms floats:
no report value is an epoch or wall-clock time.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import sys
from collections.abc import Mapping, Sequence
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import polars as pl

from aml.io import read_json, write_json_atomic, write_parquet_atomic, write_text_atomic
from aml.serving.settings import (
    ENV_HOST_NOTE,
    ConfigError,
    Point,
    ServingConfig,
    add_cli_args,
    env_value,
    load_settings,
    plan_points,
)

if TYPE_CHECKING:
    from aml.serving.scorer import Event, Runtime, Scored, Timing

log = logging.getLogger(__name__)

EVENTS_FILE = "events.parquet"  # under reports/latency/ (raw, gitignored, cleared by init)
SESSIONS_FILE = "sessions.json"
REPORT_MD = "latency.md"  # under reports/
REPORT_JSON = "latency.json"
REPORT_FORMAT = 1
ENV_KAFKA_IMAGE = "AML_KAFKA_IMAGE"  # compose: a disclosure string only

QUANTILES = (("p50", 0.5), ("p95", 0.95), ("p99", 0.99), ("p99.9", 0.999))
STAGES = ("send_lag", "transit", "decode", "features", "model", "alert_write", "champion", "e2e")
ENGINE_BENCH_FLUSH_MS = {"p99": 14.54, "max": 124.49}  # reports/engine_bench.md (offline replay)
VERSION_DISTS = ("lightgbm", "numpy", "polars", "pyarrow", "fastapi", "uvicorn", "starlette",
                 "confluent-kafka", "prometheus-client")  # fmt: skip

EVENT_SCHEMA: dict[str, Any] = {
    "session": pl.Int16,  # 1, 2, ... in this run (sessions.json)
    "offset": pl.Int64,
    "point": pl.Int16,  # the header's plan point (0 = warm-up); null without latency headers
    "t_sched": pl.Int64,  # header: scheduled send (monotonic ns); null without headers
    "t_prod": pl.Int64,  # header: just before produce(); null without headers
    "t_consume": pl.Int64,  # when the source returned the message
    "decode": pl.Int64,
    "flush": pl.Int64,  # the minute flush this event triggered; 0 if none
    "n_applied": pl.Int32,
    "t_features": pl.Int64,
    "t_model": pl.Int64,
    "t_done": pl.Int64,  # after the alert write: the end of the champion path
    "alert": pl.Boolean,
}
_META_COLUMNS = ("point", "t_sched", "t_prod")
_INT_COLUMNS = tuple(c for c in EVENT_SCHEMA if c not in ("session", "alert"))


# --- host disclosure ------------------------------------------------------------------------------


def _proc_fields(path: str, keys: Sequence[str]) -> dict[str, str]:
    """`key: value` lines of a /proc file (first occurrence of each key); {} elsewhere."""
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return {}
    out: dict[str, str] = {}
    for line in text.splitlines():
        k, sep, v = line.partition(":")
        k = k.strip()
        if sep and k in keys and k not in out:
            out[k] = v.strip()
    return out


def _kb_to_mb(v: str | None) -> float | None:
    """A /proc '123456 kB' value in MiB."""
    try:
        return int(v.split()[0]) / 1024 if v else None
    except (ValueError, IndexError):
        return None


def memory_now() -> dict[str, float | None]:
    """VmRSS / VmHWM of this process in MiB (/proc/self/status; None where unavailable)."""
    st = _proc_fields("/proc/self/status", ("VmRSS", "VmHWM"))
    return {"vm_rss_mb": _kb_to_mb(st.get("VmRSS")), "vm_hwm_mb": _kb_to_mb(st.get("VmHWM"))}


def host_info(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """The report's hardware and software disclosure (no build strings: those carry dates)."""
    env = os.environ if environ is None else environ
    cpu = _proc_fields("/proc/cpuinfo", ("model name",)).get("model name")
    try:
        affinity: int | None = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):  # not on Windows / macOS
        affinity = None
    versions: dict[str, str | None] = {}
    for dist in VERSION_DISTS:
        try:
            versions[dist] = importlib_metadata.version(dist)
        except importlib_metadata.PackageNotFoundError:
            versions[dist] = None
    libc = " ".join(x for x in platform.libc_ver() if x)
    return {
        "cpu_model": cpu or platform.processor() or None,
        "cpu_count": os.cpu_count(),
        "cpu_affinity": affinity,
        "mem_total_mb": _kb_to_mb(_proc_fields("/proc/meminfo", ("MemTotal",)).get("MemTotal")),
        "system": platform.system(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "libc": libc or None,
        "python": platform.python_version(),
        "versions": versions,
        "kafka_image": env_value(env, ENV_KAFKA_IMAGE),
        "note": env_value(env, ENV_HOST_NOTE),
    }


# --- the recorder ---------------------------------------------------------------------------------


def _grown(a: np.ndarray, capacity: int, n: int) -> np.ndarray:
    b = np.zeros(capacity, dtype=a.dtype)
    b[:n] = a[:n]
    return b


def empty_events() -> pl.DataFrame:
    return pl.DataFrame(schema=EVENT_SCHEMA)


def load_raw(latency_dir: Path) -> tuple[pl.DataFrame, list[dict[str, Any]]]:
    """This run's raw latency files: the events and the session records (empty when absent)."""
    d = Path(latency_dir)
    ev_path, ss_path = d / EVENTS_FILE, d / SESSIONS_FILE
    events = empty_events()
    if ev_path.is_file():
        events = pl.read_parquet(ev_path).select(list(EVENT_SCHEMA)).cast(EVENT_SCHEMA)
    sessions = list(read_json(ss_path)) if ss_path.is_file() else []
    return events, sessions


def _state_total(rt: Runtime) -> int | None:
    try:
        return int(rt.scorer.eng.state_nbytes()["total"])
    except Exception:  # a broken engine: the number is only reported
        return None


class LatencyRecorder:
    """Observer: one row of stamps per scored event (preallocated numpy columns) plus this
    session's record. Every hook runs on the consumer thread; `bind` sizes the columns and
    records the session's start, `on_finish` (the end of the slice) and `on_close` write the raw
    files and render the report once the slice has ended. Only `bind` may raise: a failure
    while writing is logged, never raised into the scorer."""

    def __init__(self, capacity: int = 0, *, report: bool = True) -> None:
        self.report = report
        self.session = 1
        self.n = 0
        self._cols = {c: np.zeros(capacity, dtype=np.int64) for c in _INT_COLUMNS}
        self._alert = np.zeros(capacity, dtype=np.bool_)
        self._meta = np.zeros(capacity, dtype=np.bool_)
        self.prior: pl.DataFrame | None = None  # the earlier sessions' events of this run
        self.sessions: list[dict[str, Any]] = []  # the earlier sessions' records
        self.doc: dict[str, Any] | None = None  # this session's record
        self.cfg: ServingConfig | None = None
        self.reports_dir: Path | None = None
        self.latency_dir: Path | None = None

    def _reserve(self, capacity: int) -> None:
        if capacity <= len(self._alert):
            return
        self._cols = {c: _grown(a, capacity, self.n) for c, a in self._cols.items()}
        self._alert = _grown(self._alert, capacity, self.n)
        self._meta = _grown(self._meta, capacity, self.n)

    def observe(self, ev: Event, s: Scored, t: Timing) -> None:
        i = self.n
        if i == len(self._alert):
            self._reserve(2 * i + 1024)
        c = self._cols
        c["offset"][i] = s.offset
        m = ev.meta
        if m is not None:
            self._meta[i] = True
            c["point"][i] = m.point
            c["t_sched"][i] = m.t_sched_ns
            c["t_prod"][i] = m.t_prod_ns
        c["t_consume"][i] = t.t_consume_ns
        c["decode"][i] = t.decode_ns
        c["flush"][i] = t.flush_ns
        c["n_applied"][i] = t.n_applied
        c["t_features"][i] = t.t_features_ns
        c["t_model"][i] = t.t_model_ns
        c["t_done"][i] = t.t_done_ns
        self._alert[i] = s.alert
        self.n = i + 1

    def frame(self) -> pl.DataFrame:
        """This run's events so far: the earlier sessions' rows, then this session's."""
        n = self.n
        cols: dict[str, Any] = {"session": np.full(n, self.session, dtype=np.int16)}
        cols |= {c: a[:n] for c, a in self._cols.items()}
        cols["alert"] = self._alert[:n]
        cols["_meta"] = self._meta[:n]
        df = (
            pl.DataFrame(cols)
            .with_columns(
                [pl.when(pl.col("_meta")).then(pl.col(c)).alias(c) for c in _META_COLUMNS]
            )
            .select(list(EVENT_SCHEMA))
            .cast(EVENT_SCHEMA)
        )
        return df if self.prior is None else pl.concat([self.prior, df])

    def bind(self, rt: Runtime) -> None:
        """Before the first event: size the columns, load this run's earlier sessions (init
        clears them on every `docker compose up`) and write this session's start."""
        s, ch = rt.settings, rt.champion
        self.cfg, self.reports_dir, self.latency_dir = s.cfg, s.reports_dir, s.latency_dir
        self._reserve(rt.end_offset - rt.start_offset)
        events, self.sessions = load_raw(self.latency_dir)
        self.prior = events if events.height else None
        self.session = len(self.sessions) + 1
        meta = ch.metadata or {}
        plat = meta.get("platform") or {}
        self.doc = {
            "session": self.session,
            "status": "running",
            "transport": s.transport,
            "origin": rt.restored.origin,
            "restore_s": rt.restored.seconds,
            "start_offset": rt.start_offset,
            "end_offset": rt.end_offset,
            "events": 0,
            "alerts": 0,
            "finished": rt.scorer.finished,
            "final_flush": None,
            "parity": rt.parity,
            "log1p_ok": rt.log1p_ok,
            "snapshot": None,
            "memory": {"restore": memory_now(), "end": None},
            "state_nbytes": {"restore": _state_total(rt), "end": None},
            "bundle": {
                "export_key": ch.export_key,
                "model_version": ch.model_version,
                "alert_tag": ch.alert_tag,
                "threshold": ch.threshold,
                "slice_rows": (meta.get("rows") or {}).get("slice"),
                "libc": " ".join(str(x) for x in (plat.get("libc"), plat.get("libc_version")) if x)
                or None,
            },
            "host": {**host_info(), "note": s.host_note},
        }
        self.write()

    def write(self, latency_dir: Path | None = None, frame: pl.DataFrame | None = None) -> Path:
        """Write events.parquet (and sessions.json once bound), atomically."""
        d = Path(latency_dir if latency_dir is not None else self.latency_dir)
        path = write_parquet_atomic(self.frame() if frame is None else frame, d / EVENTS_FILE)
        if self.doc is not None:
            write_json_atomic([*self.sessions, self.doc], d / SESSIONS_FILE)
        return path

    def _save(self, rt: Runtime, status: str, *, render: bool) -> None:
        try:
            sc = rt.scorer
            ff = sc.final_flush
            self.doc.update(
                status=status,
                events=sc.progress.events,
                alerts=sc.progress.alerts,
                finished=sc.finished,
                parity=rt.parity,
                snapshot=rt.snapshot,
            )
            if ff is not None:
                self.doc["final_flush"] = {
                    "n_applied": ff.n_applied,
                    "n_expired": ff.n_expired,
                    "ms": ff.seconds * 1e3,
                }
            self.doc["memory"]["end"] = memory_now()
            self.doc["state_nbytes"]["end"] = _state_total(rt)
            frame = self.frame()
            self.write(frame=frame)
            if render:
                finalize_report(self.cfg, self.reports_dir, frame, [*self.sessions, self.doc])
        except Exception:
            log.exception("the latency files were not written")

    def on_finish(self, rt: Runtime) -> None:
        self._save(rt, "finished", render=self.report)

    def on_close(self, rt: Runtime, result: str) -> None:
        # Re-rendered at close once the slice has ended, with the shutdown snapshot.
        self._save(rt, result, render=self.report and rt.scorer.finished)


# --- statistics -----------------------------------------------------------------------------------


def nearest_rank(sorted_x: np.ndarray, q: float) -> Any:
    """x_(ceil(q n)) of an ascending sample, as np.quantile(x, q, method="inverted_cdf")
    computes it (virtual index n q - 1, kept when it is integral, else rounded up)."""
    n = len(sorted_x)
    if n == 0:
        raise ValueError("nearest_rank of an empty sample")
    idx = n * q - 1
    lo = math.floor(idx)
    k = lo if idx == lo else lo + 1
    return sorted_x[min(max(k, 0), n - 1)]


def adequate(n: int, q: float, min_tail: int) -> bool:
    """A quantile is reported only with at least min_tail values beyond it: n (1 - q) >= min_tail
    (with a tolerance for 1 - q not being exact in binary)."""
    return n * (1 - q) >= min_tail - 1e-9


def quantile_ns(x_ns: np.ndarray, q: float, min_tail: int) -> int | None:
    """The nearest-rank quantile of a sample of ns durations; None when not adequate."""
    n = len(x_ns)
    if n == 0 or not adequate(n, q, min_tail):
        return None
    return int(nearest_rank(np.sort(np.asarray(x_ns, dtype=np.int64)), q))


def distribution(x_ns: np.ndarray, min_tail: int) -> dict[str, Any]:
    """n, mean, max and the QUANTILES of ns durations, in ms; a quantile is None (n/a) when the
    sample has fewer than min_tail values beyond it."""
    x = np.sort(np.asarray(x_ns, dtype=np.int64))
    n = len(x)
    out: dict[str, Any] = {
        "n": n,
        "mean_ms": float(x.mean()) / 1e6 if n else None,
        "max_ms": float(x[-1]) / 1e6 if n else None,
    }
    for label, q in QUANTILES:
        out[label] = float(nearest_rank(x, q)) / 1e6 if n and adequate(n, q, min_tail) else None
    return out


def stage_ns(df: pl.DataFrame) -> dict[str, np.ndarray]:
    """Per-event stage durations in ns (module docstring); send_lag, transit and e2e only when
    every row carries the latency headers."""
    c = {
        k: df[k].to_numpy().astype(np.int64, copy=False)
        for k in ("t_consume", "decode", "flush", "t_features", "t_model", "t_done")
    }
    out = {
        "decode": c["decode"],
        "features": c["t_features"] - c["t_consume"] - c["decode"] - c["flush"],
        "model": c["t_model"] - c["t_features"],
        "alert_write": c["t_done"] - c["t_model"],
        "champion": c["t_done"] - c["t_consume"] - c["flush"],
    }
    if df["t_sched"].null_count() == 0 and df["t_prod"].null_count() == 0:
        ts = df["t_sched"].to_numpy().astype(np.int64, copy=False)
        tp = df["t_prod"].to_numpy().astype(np.int64, copy=False)
        out["send_lag"] = tp - ts
        out["transit"] = c["t_consume"] - tp
        out["e2e"] = c["t_done"] - ts
    return out


def _ns(seconds: float) -> int:
    return round(seconds * 1e9)


def _point(
    p: Point, rows: pl.DataFrame, cfg: ServingConfig
) -> tuple[dict[str, Any], dict[str, int | None]]:
    """One plan point's statistics (ms, for the report) and its verdict quantiles (ns)."""
    lat = cfg.latency
    tail = lat.min_tail_samples
    st = stage_ns(rows)
    n = rows.height
    service = st["features"] + st["model"] + st["alert_write"]
    doc: dict[str, Any] = {
        "index": p.index,
        "shape": p.shape,
        "rate": p.rate,
        "planned": p.events,
        "n": n,
        "achieved_ev_s": None,
        "throughput_ev_s": None,
        "generator_ok": None,
        "sustained": None,
        "send_lag_ms": distribution(st["send_lag"], tail),
        "champion_ms": distribution(st["champion"], tail),
        "e2e_ms": distribution(st["e2e"], tail),
        "service_us": float(service.mean()) / 1e3 if n else None,
        "flush_us_per_event": float(rows["flush"].sum()) / n / 1e3 if n else None,
    }
    q = {
        "send_lag_p99": quantile_ns(st["send_lag"], 0.99, tail),
        "champion_p95": quantile_ns(st["champion"], 0.95, tail),
        "champion_p99": quantile_ns(st["champion"], 0.99, tail),
        "e2e_p99": quantile_ns(st["e2e"], 0.99, tail),
    }
    if n > 1:
        if p.rate is None:  # unpaced: the drain rate under overload
            span = rows["t_done"].max() - rows["t_done"].min()
            doc["throughput_ev_s"] = (n - 1) / (span / 1e9) if span > 0 else None
        else:
            span = rows["t_done"].max() - rows["t_sched"].min()
            doc["achieved_ev_s"] = (n - 1) / (span / 1e9) if span > 0 else None
    if p.rate is not None:
        lag99, e2e99, achieved = q["send_lag_p99"], q["e2e_p99"], doc["achieved_ev_s"]
        gen = lag99 is not None and lag99 <= _ns(lat.max_send_lag_ms / 1e3)
        doc["generator_ok"] = gen
        doc["sustained"] = (
            gen
            and achieved is not None
            and achieved >= lat.min_rate_ratio * p.rate
            and e2e99 is not None
            and e2e99 <= _ns(lat.e2e_p99_s)
        )
    return doc, q


def _off_plan(events: pl.DataFrame, plan: Sequence[Point]) -> int:
    """Events without latency headers or whose header point is not the plan's for its offset."""
    if not plan:
        return events.height
    starts = np.array([p.start for p in plan], dtype=np.int64)
    index = np.array([p.index for p in plan], dtype=np.int64)
    off = events["offset"].to_numpy()
    pos = np.clip(np.searchsorted(starts, off, side="right") - 1, 0, len(plan) - 1)
    got = events["point"].fill_null(-1).to_numpy()
    return int(np.count_nonzero(got != index[pos]))


def _merge(ranges: Sequence[Sequence[int]]) -> list[list[int]]:
    """Inclusive [lo, hi] offset ranges merged where they overlap or touch."""
    out: list[list[int]] = []
    for lo, hi in sorted((int(a), int(b)) for a, b in ranges):
        if out and lo <= out[-1][1] + 1:
            out[-1][1] = max(out[-1][1], hi)
        else:
            out.append([lo, hi])
    return out


def parity_summary(sessions: Sequence[Mapping[str, Any]], n: int) -> dict[str, Any]:
    """Online parity over the run: every session's mismatch counts are 0, the sessions together
    cover offsets [0, n - 1], and the finishing session's alerts check and final state digest
    pass (a digest that was not checkable, on a capped run, is n/a)."""
    docs = [s["parity"] for s in sessions if s.get("parity")]
    mism: dict[str, int] = {}
    for d in docs:
        for k, v in (d.get("mismatches") or {}).items():
            mism[k] = mism.get(k, 0) + int(v)
    final = next((d for d in reversed(docs) if d.get("alerts_ok") is not None), None)
    covered = _merge([d["covered"] for d in docs if d.get("covered")])
    errors = [str(d["error"]) for d in docs if d.get("error")]
    alerts_ok = None if final is None else final.get("alerts_ok")
    digest_ok = None if final is None else final.get("digest_ok")
    ok = (
        final is not None
        and n > 0
        and covered == [[0, n - 1]]
        and not errors
        and not any(mism.values())
        and alerts_ok is True
        and digest_ok is not False
    )
    return {
        "result": "PASS" if ok else "FAIL",
        "checked": sum(int(d.get("checked") or 0) for d in docs),
        "covered": covered,
        "mismatches": mism,
        "alerts_ok": alerts_ok,
        "alerts": None if final is None else final.get("alerts"),
        "digest_ok": digest_ok,
        "errors": errors,
        "finished": final is not None,
        "log1p_ok": sessions[-1].get("log1p_ok") if sessions else None,
    }


def _ms(ns: int | None) -> str:
    return "n/a" if ns is None else f"{ns / 1e6:.3f}"


def _pf(ok: bool, applicable: bool) -> str:
    return "n/a" if not applicable else "PASS" if ok else "FAIL"


def build_summary(
    cfg: ServingConfig, events: pl.DataFrame, sessions: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    """The verdict and the statistics of one run from its raw files.

    latency = INVALID when the run cannot be judged (no session, a restart, offsets not all
    recorded once, events off the plan, no point at targets.min_rate_ev_s, a target point whose
    generator fell behind or with too few events for a verdict quantile); else PASS iff at every
    uniform and trace point at that rate: sustained, champion p95 and p99 within the targets.
    parity = parity_summary. overall = PASS iff both pass; INVALID if parity passes and latency
    is INVALID; else FAIL. Comparisons are on integer ns, so a target is met at equality.
    """
    lat, tg = cfg.latency, cfg.targets
    tail = lat.min_tail_samples
    rate = tg.min_rate_ev_s
    n = max((int(s["end_offset"]) for s in sessions), default=0)
    plan = plan_points(cfg, n) if n else ()
    last: Mapping[str, Any] = sessions[-1] if sessions else {}
    reasons: list[str] = []

    # Run integrity: one scoring session (a restore after the end scores nothing), every offset
    # recorded exactly once, every event on its plan point.
    scoring = [s for s in sessions if int(s["start_offset"]) < int(s["end_offset"])]
    if not sessions:
        reasons.append("no scorer session was recorded")
    elif len(scoring) > 1:
        reasons.append(f"the scorer restarted ({len(scoring)} sessions): one uninterrupted run")
    off = events["offset"].to_numpy()
    once = int(np.unique(off[(off >= 0) & (off < n)]).size)
    complete = n > 0 and events.height == n and once == n
    if not complete:
        reasons.append(f"offsets incomplete: {once} of {n} recorded ({events.height} rows)")
    off_plan = _off_plan(events, plan)
    if off_plan:
        reasons.append(f"{off_plan} events lack latency headers or are off the plan")

    points = [_point(p, events.filter(pl.col("point") == p.index), cfg) for p in plan if p.index]
    target = [(d, q) for d, q in points if d["shape"] in ("uniform", "trace") and d["rate"] == rate]
    if not target:
        reasons.append(f"no measured uniform or trace point at {rate:g} ev/s")
    for d, q in target:
        short = [
            k for k in ("send_lag_p99", "champion_p95", "champion_p99", "e2e_p99") if q[k] is None
        ]
        if short:
            reasons.append(
                f"point {d['index']}: too few events for {', '.join(short)} (n {d['n']})"
            )
        elif not d["generator_ok"]:
            reasons.append(
                f"point {d['index']}: the replayer fell behind its schedule (send-lag p99 "
                f"{_ms(q['send_lag_p99'])} ms > {lat.max_send_lag_ms:g} ms)"
            )
    c2 = bool(target) and all(d["sustained"] for d, _ in target)
    c3 = bool(target) and all(
        q["champion_p95"] is not None and q["champion_p95"] <= _ns(tg.champion_p95_s)
        for _, q in target
    )
    c4 = bool(target) and all(
        q["champion_p99"] is not None and q["champion_p99"] <= _ns(tg.champion_p99_s)
        for _, q in target
    )
    latency = "INVALID" if reasons else "PASS" if c2 and c3 and c4 else "FAIL"
    par = parity_summary(sessions, n)
    if par["result"] == "PASS" and latency == "PASS":
        overall = "PASS"
    elif par["result"] == "PASS" and latency == "INVALID":
        overall = "INVALID"
    else:
        overall = "FAIL"

    def each(fmt: Any) -> str:
        return " / ".join(fmt(d, q) for d, q in target) or "n/a"

    single = len(scoring) == 1 and complete and not off_plan
    checks = [
        {
            "check": "one uninterrupted run",
            "measured": f"{len(scoring)} scoring session(s); {once:,} of {n:,} offsets; "
            f"{off_plan:,} events off the plan",
            "target": "1 session; every offset once, on its plan point",
            "result": "PASS" if single else "INVALID",
        },
        {
            "check": f"generator on schedule at {rate:g} ev/s",
            "measured": "send-lag p99 " + each(lambda d, q: _ms(q["send_lag_p99"])) + " ms",
            "target": f"<= {lat.max_send_lag_ms:g} ms",
            "result": "PASS" if target and all(d["generator_ok"] for d, _ in target) else "INVALID",
        },
        {
            "check": f"{rate:g} ev/s sustained (uniform, trace)",
            "measured": "achieved "
            + each(
                lambda d, q: "n/a" if d["achieved_ev_s"] is None else f"{d['achieved_ev_s']:.1f}"
            )
            + " ev/s; e2e p99 "
            + each(lambda d, q: _ms(q["e2e_p99"]))
            + " ms",
            "target": f">= {lat.min_rate_ratio * rate:.1f} ev/s; e2e p99 <= "
            f"{lat.e2e_p99_s * 1e3:g} ms",
            "result": _pf(c2, bool(target)),
        },
        {
            "check": "champion p95 (consume -> alert write)",
            "measured": each(lambda d, q: _ms(q["champion_p95"])) + " ms",
            "target": f"<= {tg.champion_p95_s * 1e3:g} ms",
            "result": _pf(c3, bool(target)),
        },
        {
            "check": "champion p99",
            "measured": each(lambda d, q: _ms(q["champion_p99"])) + " ms",
            "target": f"<= {tg.champion_p99_s * 1e3:g} ms",
            "result": _pf(c4, bool(target)),
        },
        {
            "check": "parity with the offline references",
            "measured": f"{sum(par['mismatches'].values()):,} mismatches over {par['checked']:,} "
            f"rows; alerts {_word(par['alerts_ok'])}; final digest {_word(par['digest_ok'])}",
            "target": "bit-exact; alerts and digest equal",
            "result": par["result"],
        },
    ]

    uniform = sorted((d for d, _ in points if d["shape"] == "uniform"), key=lambda d: d["rate"])
    up_to = first_fail = None
    for d in uniform:
        if not d["sustained"] and first_fail is None:
            first_fail = d["rate"]
        elif d["sustained"] and first_fail is None:
            up_to = d["rate"]
    unpaced = next((d for d, _ in points if d["shape"] == "unpaced"), None)

    at_rate = events.filter(pl.col("point").is_in([d["index"] for d, _ in target]))
    st = stage_ns(at_rate)
    stages = {k: distribution(st[k], tail) for k in STAGES if k in st}
    alerted = at_rate.filter(pl.col("alert"))
    stages["alert_write_alerted"] = distribution(stage_ns(alerted)["alert_write"], tail)

    fl = events.filter(pl.col("flush") > 0)
    applied = np.sort(fl["n_applied"].to_numpy())
    fin = next((s["final_flush"] for s in reversed(sessions) if s.get("final_flush")), None)

    mem = last.get("memory") or {}
    mem_end = mem.get("end") or mem.get("restore") or {}
    nbytes = last.get("state_nbytes") or {}
    snap = next((s["snapshot"] for s in reversed(sessions) if s.get("snapshot")), None)
    plan0 = plan[0] if plan and plan[0].index == 0 else None

    return {
        "format": REPORT_FORMAT,
        "transport": last.get("transport"),
        "events": n,
        "recorded": events.height,
        "sessions": len(sessions),
        "scoring_sessions": len(scoring),
        "bundle": last.get("bundle"),
        "verdict": {
            "overall": overall,
            "latency": latency,
            "parity": par["result"],
            "reasons": reasons,
        },
        "checks": checks,
        "config": {
            "min_rate_ev_s": rate,
            "champion_p95_ms": tg.champion_p95_s * 1e3,
            "champion_p99_ms": tg.champion_p99_s * 1e3,
            "e2e_p99_ms": lat.e2e_p99_s * 1e3,
            "max_send_lag_ms": lat.max_send_lag_ms,
            "min_rate_ratio": lat.min_rate_ratio,
            "min_tail_samples": tail,
            "warmup_events": plan0.events if plan0 else 0,
            "lead_s": cfg.replayer.lead_s,
            "settle_s": cfg.replayer.settle_s,
            "gap_cap_s": cfg.replayer.gap_cap_s,
        },
        "points": [d for d, _ in points],
        "saturation": {
            "sustained_up_to_ev_s": up_to,
            "first_unsustained_ev_s": first_fail,
            "unpaced_n": 0 if unpaced is None else unpaced["n"],
            "unpaced_throughput_ev_s": None if unpaced is None else unpaced["throughput_ev_s"],
        },
        "stages": {
            "rate": rate,
            "points": [d["index"] for d, _ in target],
            "n": at_rate.height,
            "items": stages,
        },
        "flush": {
            "count": fl.height,
            "n_applied_p50": int(nearest_rank(applied, 0.5)) if fl.height else None,
            "n_applied_max": int(applied[-1]) if fl.height else None,
            "ms": distribution(fl["flush"].to_numpy(), tail),
            "final": fin,
            "offline_bench_ms": ENGINE_BENCH_FLUSH_MS,
        },
        "memory": {
            "state_restore_mb": _mib(nbytes.get("restore")),
            "state_end_mb": _mib(nbytes.get("end")),
            "vm_rss_mb": mem_end.get("vm_rss_mb"),
            "vm_hwm_mb": mem_end.get("vm_hwm_mb"),
            "restore_origin": last.get("origin"),
            "restore_s": last.get("restore_s"),
            "snapshot_mb": None if snap is None else _mib(snap.get("bytes")),
            "snapshot_s": None if snap is None else snap.get("seconds"),
        },
        "parity": par,
        "host": last.get("host"),
    }


def _mib(nbytes: int | None) -> float | None:
    return None if nbytes is None else nbytes / 2**20


def _word(ok: bool | None) -> str:
    return "n/a" if ok is None else "ok" if ok else "FAILED"


# --- rendering ------------------------------------------------------------------------------------


def _f(x: float | None, nd: int = 3) -> str:
    return "n/a" if x is None else f"{x:,.{nd}f}"


def _quads(d: Mapping[str, Any] | None) -> str:
    return " / ".join(_f(None if d is None else d.get(label)) for label, _ in QUANTILES)


def render_markdown(s: Mapping[str, Any]) -> str:
    """reports/latency.md from a build_summary dict."""
    v, c = s["verdict"], s["config"]
    b, h = s.get("bundle") or {}, s.get("host") or {}
    rate = c["min_rate_ev_s"]
    out = [
        "# Streaming latency (M5)",
        "",
        f"Bundle `{b.get('export_key')}` (model {b.get('model_version')}, alert tag "
        f"{b.get('alert_tag')}) · {s['events']:,} events · transport {s['transport']} · "
        f"{s['sessions']} scorer session(s)",
        "",
        f"**Verdict: {v['overall']}** (latency {v['latency']}, parity {v['parity']})",
        "",
        *(f"- Not judged: {r}" for r in v["reasons"]),
        "",
        "## Verdict",
        "",
        "| check | measured | target | result |",
        "|---|---|---|---|",
        *(
            f"| {k['check']} | {k['measured']} | {k['target']} | {k['result']} |"
            for k in s["checks"]
        ),
        "",
        "## Method",
        "",
        "- Open loop: the replayer precomputes every send time and stamps it (`t_sched`) into the "
        "message headers with the produce time (`t_prod`). A sender that falls behind sends at "
        "once and keeps the scheduled stamp, so queueing is never hidden (no coordinated "
        "omission).",
        f"- Fixed counts: a warm-up of {c['warmup_events']:,} events (not counted), then the "
        "points below in order. Before each point the replayer drains the scorer (a barrier on "
        f"`/health` last_offset) and waits {c['settle_s']:g} s; each schedule starts "
        f"{c['lead_s']:g} s after its barrier.",
        f"- Shapes: uniform (event k at k / R), trace (the slice's own minute pattern, idle gaps "
        f"cut to {c['gap_cap_s']:g} s, rescaled to rate R), unpaced (sent as fast as possible: "
        "the saturation point).",
        "- Clock: every stamp is `time.monotonic_ns()`. The replayer, broker and scorer "
        "containers share the Docker host kernel's monotonic clock.",
        "- Champion = consume -> alert write (decode, features, model, SQLite commit and alerts "
        "produce) with the minute flush excluded; e2e = scheduled send -> alert write, which "
        "includes queueing and the minute flush.",
        "- Quantiles are nearest rank; a quantile is reported only when n (1 - q) >= "
        f"{c['min_tail_samples']}. There are no confidence intervals.",
        f"- Sustained at rate R: send-lag p99 <= {c['max_send_lag_ms']:g} ms, achieved >= "
        f"{c['min_rate_ratio']:g} R and e2e p99 <= {c['e2e_p99_ms']:g} ms.",
        "",
        "## Rate points",
        "",
        "| # | shape | R (ev/s) | n | achieved (ev/s) | sustained | send-lag p99 (ms) | "
        "champion p50 / p95 / p99 / p99.9 (ms) | e2e p50 / p95 / p99 / p99.9 (ms) | "
        "service (µs/ev) |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for p in s["points"]:
        if p["rate"] is None:  # unpaced: no schedule, so no send lag or sustained verdict
            rate_s, achieved, lag = "unpaced", _f(p["throughput_ev_s"], 1) + " (drain)", "-"
        else:
            rate_s, achieved = f"{p['rate']:g}", _f(p["achieved_ev_s"], 1)
            lag = _f(p["send_lag_ms"]["p99"])
        sustained = "-" if p["sustained"] is None else "yes" if p["sustained"] else "no"
        out.append(
            f"| {p['index']} | {p['shape']} | {rate_s} | {p['n']:,} | {achieved} | {sustained} | "
            f"{lag} | {_quads(p['champion_ms'])} | {_quads(p['e2e_ms'])} | "
            f"{_f(p['service_us'], 1)} |"
        )
    sat = s["saturation"]
    st = s["stages"]
    out += [
        "",
        "## Saturation",
        "",
        f"- Unpaced point: {_f(sat['unpaced_throughput_ev_s'], 1)} ev/s drained under overload "
        f"(n {sat['unpaced_n']:,}).",
        f"- Paced uniform points: sustained up to {_rate(sat['sustained_up_to_ev_s'])}; first "
        f"not sustained: {_rate(sat['first_unsustained_ev_s'])}.",
        "",
        f"## Stages at {rate:g} ev/s",
        "",
        f"Points {', '.join(str(i) for i in st['points']) or 'none'} (uniform and trace), "
        f"n {st['n']:,}; ms.",
        "",
        "| stage | n | mean | p50 | p95 | p99 | p99.9 | max |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, d in st["items"].items():
        label = "alert_write (alerted events)" if name == "alert_write_alerted" else name
        out.append(
            f"| {label} | {d['n']:,} | {_f(d['mean_ms'])} | "
            + " | ".join(_f(d[q]) for q, _ in QUANTILES)
            + f" | {_f(d['max_ms'])} |"
        )
    fl, fin = s["flush"], s["flush"]["final"] or {}
    m = s["memory"]
    par = s["parity"]
    al = par.get("alerts") or {}
    mism = ", ".join(f"{k} {v}" for k, v in par["mismatches"].items()) or "none checked"
    vers = ", ".join(f"{k} {x}" for k, x in (h.get("versions") or {}).items() if x)
    flush_ms = " / ".join(_f(fl["ms"][q]) for q in ("p50", "p95", "p99"))
    out += [
        "",
        "## Minute flush",
        "",
        f"- {fl['count']:,} flushes; events applied per flush p50 {_n(fl['n_applied_p50'])}, "
        f"max {_n(fl['n_applied_max'])}; flush ms p50 / p95 / p99 / max: {flush_ms} / "
        f"{_f(fl['ms']['max_ms'])}.",
        f"- Final flush: {fin.get('n_applied', 'n/a')} events applied, {_f(fin.get('ms'))} ms.",
        f"- Offline engine bench (reports/engine_bench.md): p99 {fl['offline_bench_ms']['p99']} "
        f"ms, max {fl['offline_bench_ms']['max']} ms.",
        "",
        "## Memory",
        "",
        f"- Engine state (`Engine.state_nbytes`): {_f(m['state_restore_mb'], 1)} MiB after the "
        f"restore, {_f(m['state_end_mb'], 1)} MiB at the end.",
        f"- Scorer process: VmRSS {_f(m['vm_rss_mb'], 1)} MiB, VmHWM {_f(m['vm_hwm_mb'], 1)} MiB "
        "(/proc/self/status).",
        f"- Restore: {_f(m['restore_s'], 2)} s from the {m['restore_origin'] or 'n/a'} snapshot. "
        + (
            f"Shutdown snapshot: {_f(m['snapshot_mb'], 1)} MiB in {_f(m['snapshot_s'], 2)} s."
            if m["snapshot_mb"] is not None
            else "No shutdown snapshot was written yet."
        ),
        "",
        "## Parity",
        "",
        f"- {par['checked']:,} rows checked online against reference/*.parquet, covered "
        f"{par['covered'] or 'none'}; mismatches: {mism}.",
        f"- Alerts table vs the reference alerts: {_word(par['alerts_ok'])} (stored "
        f"{al.get('db_count', 'n/a')}, reference {al.get('ref_count', 'n/a')}, missing "
        f"{al.get('missing', 'n/a')}, extra {al.get('extra', 'n/a')}).",
        f"- Final state digest: {_word(par['digest_ok'])}.",
        "- log1p probe: "
        + (
            "differs from the bundle's host, so bit parity is not expected on this host."
            if par["log1p_ok"] is False
            else "matches the bundle's host."
            if par["log1p_ok"]
            else "n/a."
        ),
        *(f"- Checker error: {e}" for e in par["errors"]),
        "",
        "## Hardware and software",
        "",
        f"- CPU: {h.get('cpu_model') or 'unknown'}; {_n(h.get('cpu_count'))} logical CPUs "
        f"(scorer affinity {_n(h.get('cpu_affinity'))}); MemTotal "
        f"{_f(h.get('mem_total_mb'), 0)} MiB.",
        f"- {h.get('system') or 'OS n/a'} kernel {h.get('kernel') or 'n/a'} "
        f"({h.get('machine') or 'n/a'}); libc {h.get('libc') or 'n/a'} (bundle references: "
        f"{b.get('libc') or 'n/a'}).",
        f"- Python {h.get('python') or 'n/a'}; {vers or 'library versions n/a'}.",
        f"- Kafka image: {h.get('kafka_image') or 'n/a'}. Host note: {h.get('note') or 'none'}.",
        "- The replayer, the broker and the scorer share this machine.",
        "",
        "## Caveats",
        "",
        "- One run on a laptop through the Docker VM: background load and the VM add noise; this "
        "is a demo measurement, not a production benchmark.",
        "- Synthetic data: IBM AML HI-Small is simulated, so the event rates and the trace shape "
        "are the simulator's.",
        "- One partition and one consumer thread; the engine is pure Python under the GIL, so the "
        "throughput is that of one core.",
        "- Rate is confounded with the data segment: each point replays the next part of the "
        "slice, so the per-point service time (µs/ev) is shown to separate the two.",
        "",
    ]
    return "\n".join(out)


def _rate(r: float | None) -> str:
    return "none" if r is None else f"{r:g} ev/s"


def _n(x: int | None) -> str:
    return "n/a" if x is None else f"{x:,}"


def finalize_report(
    cfg: ServingConfig,
    reports_dir: Path,
    events: pl.DataFrame,
    sessions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """build_summary, then write reports/latency.json and reports/latency.md; logs the verdict."""
    summary = build_summary(cfg, events, sessions)
    write_json_atomic(summary, Path(reports_dir) / REPORT_JSON)
    write_text_atomic(render_markdown(summary), Path(reports_dir) / REPORT_MD)
    v = summary["verdict"]
    log.info(
        "VERDICT %s (latency %s, parity %s)%s",
        v["overall"],
        v["latency"],
        v["parity"],
        "".join(f"; {r}" for r in v["reasons"]),
    )
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    """python -m aml.serving.latency report [--reports --config ...]: re-render the report from
    reports/latency/ (the raw files carry the bundle and host facts; the other shared flags are
    accepted and unused)."""
    p = argparse.ArgumentParser(prog="python -m aml.serving.latency", description="M5 latency")
    sub = p.add_subparsers(dest="cmd", required=True)
    add_cli_args(sub.add_parser("report", help="re-render reports/latency.{md,json}"))
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings(args, transport="kafka")
    except ConfigError as e:
        log.error("refused: %s", e)
        return 2
    events, sessions = load_raw(settings.latency_dir)
    if not sessions:
        log.error("no raw latency files in %s (the Kafka scorer writes them)", settings.latency_dir)
        return 1
    summary = finalize_report(settings.cfg, settings.reports_dir, events, sessions)
    print(json.dumps(summary["verdict"], sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
