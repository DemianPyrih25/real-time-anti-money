"""The M5 streaming scorer: one consumer thread owns the engine; features -> LightGBM -> alert.

Core module: no `confluent_kafka`, `fastapi`, `uvicorn`, `starlette` or `prometheus_client` at
module level (M6 `case_eval` reuses it in Modal's cpu_image). The Kafka source and the alerts
producer are imported lazily, only on the Kafka transport.

Per event (`StreamScorer.step`), exactly the computation `bundle._verify` checked offline:

    ev = eng.prepare(*fields)               # a bad input -> OrderError("input"), state untouched
    eng.advance(minute)                     # only when the minute is new: the timed minute flush
    row = eng.process(ev)                   # features + severities; the clock is already there
    x = np.asarray([row[i] for i in idx], np.float64).astype(np.float32)[None]
    score = float(booster.predict(x, num_threads=1)[0])
    alert = threshold is not None and score >= threshold

Order is asserted, never sorted: the offset, the rank lineage (rank - offset == K) and the minute
(>= clock) are checked before any engine call. There is no idle or wall-clock flush; the final
`advance(last minute + 1)` runs after the last event of the slice.

`Runtime` (built by `open_runtime`) wires the bundle, the restored state, the alert sinks, the
source and the online parity check for the CLI, the HTTP app thread and the tests.

M6 case packs (`settings.case_packs`, on by default): `open_runtime` registers the
`aml.explain.casepack` hook as the first `on_alert` hook, configured by `explain.yaml` next to
serving.yaml, writing into the alerts SQLite file's `cases` table. It runs after t_done (untimed)
and only reads the engine, so features, scores, alerts and the state digest are unchanged; a
failing pack is logged and counted (`Runtime.summary()["cases"]`), never raised.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import platform
import signal
import struct
import sys
import threading
import time
from array import array
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import lightgbm as lgb
import numpy as np
import polars as pl
import pyarrow.parquet as pq

from aml.explain.casepack import CONFIG_FILE as EXPLAIN_CONFIG_FILE
from aml.explain.casepack import CaseHook, case_hook, load_config
from aml.features.engine import Engine
from aml.features.spec import (
    E_DST,
    E_MINUTE,
    E_RANK,
    E_ROW_ID,
    E_SRC,
    E_USD,
    INPUT_COLUMNS,
    MINUTES_PER_DAY,
    SEVERITY_COLUMNS,
    EngineSpec,
    FlushStats,
    NumericRangeError,
)
from aml.io import read_json, sha256_file
from aml.rules.sql_baseline import SCENARIOS
from aml.serving import bundle
from aml.serving.alerts import KafkaAlertSink, SqliteAlertStore
from aml.serving.settings import TRANSPORTS, ConfigError, Settings, add_cli_args, load_settings
from aml.serving.state import Restored, RuntimeStateError, StateStore
from aml.streaming.codec import CodecError, Meta, now_ns

log = logging.getLogger(__name__)

N_SEV = len(SEVERITY_COLUMNS)
EXIT_CODES = {"signal": 0, "end": 0, "order_error": 3, "error": 1}
EXIT_REFUSED = 2
EXIT_PARITY = 4  # the slice ended with a parity failure
SINK_FLUSH_S = 10.0
HW_REFRESH_NS = 500_000_000  # high watermark read at most every 0.5 s while busy
SELF_CHECK_MODULES = ("fastapi", "uvicorn", "confluent_kafka", "prometheus_client", "lightgbm",
                      "polars", "pyarrow", "jinja2")  # fmt: skip
SELF_CHECK_DISTS = ("fastapi", "uvicorn", "confluent-kafka", "prometheus-client", "lightgbm",
                    "polars", "pyarrow", "numpy", "jinja2")  # fmt: skip
SELF_CHECK_LOG1P = (0.01, 1.0, 9.99, 1234.56, 9999.99, 1e6, 1e9)


class RefusalError(RuntimeError):
    """A bundle or config problem: the scorer does not start (exit 2)."""


class OrderError(RuntimeError):
    """The message is not the next event of the stream (exit 3), raised before any engine call.

    kind: offset (gap, duplicate or beyond the end), rank (rank - offset != K or rank !=
    next_rank), late (minute < clock), input (Engine.prepare refused it), codec (the source).
    """

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(f"{kind}: {message}")
        self.kind = kind


def startup_exit_code(e: BaseException) -> int:
    """The exit code for an exception from `open_runtime`: 2 for refusals, else 1."""
    return EXIT_REFUSED if isinstance(e, RefusalError | RuntimeStateError | ConfigError) else 1


# --- the model ------------------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Champion:
    """The LightGBM champion of a serving bundle and its decision rule at one rate tag."""

    spec: EngineSpec
    booster: lgb.Booster
    names: tuple[str, ...]  # model inputs, booster order
    idx: tuple[int, ...]  # their positions in the engine row
    alert_tag: str  # the resolved thresholds.json rate tag
    threshold: float | None  # None: this tag never alerts
    rule_head_tag: str
    rule_thresholds: dict[str, float | None]  # scenario -> threshold at the rules' headline tag
    export_key: str | None
    booster_sha256: str
    metadata: dict[str, Any] | None = None  # metadata.json (bundles only)
    bundle_dir: Path | None = None
    thresholds: dict[str, Any] = field(default_factory=dict)  # thresholds.json

    @property
    def model_version(self) -> str:
        return f"{self.export_key}:{self.booster_sha256[:12]}"

    @classmethod
    def from_bundle(cls, bundle_dir: Path, *, alert_tag: str = "headline") -> Champion:
        """Load and check a bundle; RefusalError if it is missing, incomplete or inconsistent.

        Every metadata.files entry is re-hashed except the snapshot, which `Engine.restore`
        checks itself (payload sha256 and state digest)."""
        d = Path(bundle_dir)
        meta_path = d / bundle.METADATA_FILE
        if not meta_path.is_file():
            raise RefusalError(
                f"no serving bundle at {d} ({bundle.METADATA_FILE} missing): run `make pull` "
                "(or `make bundle-fixture` for a synthetic one)"
            )
        meta = read_json(meta_path)
        if meta.get("bundle_format") != bundle.BUNDLE_FORMAT:
            raise RefusalError(f"{meta_path}: bundle_format {meta.get('bundle_format')!r}")
        files = meta.get("files") or {}
        missing = [f for f in bundle.BUNDLE_FILES if f not in files]
        wrong = [
            rel
            for rel, info in files.items()
            if rel != bundle.SNAPSHOT and not _file_matches(d / rel, info)
        ]
        if missing or wrong:
            raise RefusalError(f"bundle {d}: files missing {missing}, differing {wrong}")
        ch = cls._build(
            read_json(d / bundle.FEATURE_SPEC),
            d / bundle.BOOSTER,
            read_json(d / bundle.THRESHOLDS),
            alert_tag=alert_tag,
            export_key=(meta.get("keys") or {}).get("export"),
            booster_sha256=files[bundle.BOOSTER]["sha256"],
            metadata=meta,
            bundle_dir=d,
        )
        if ch.spec.spec_hash() != meta.get("spec_hash"):
            raise RefusalError(f"bundle {d}: feature spec hash differs from metadata.json")
        return ch

    @classmethod
    def from_parts(
        cls,
        feature_spec: dict[str, Any],
        booster_path: Path,
        thresholds: dict[str, Any],
        export_key: str | None = None,
        *,
        alert_tag: str = "headline",
    ) -> Champion:
        """The champion from bundle-format parts (M6 on Modal: no metadata.json)."""
        return cls._build(
            feature_spec,
            Path(booster_path),
            thresholds,
            alert_tag=alert_tag,
            export_key=export_key,
            booster_sha256=sha256_file(Path(booster_path)),
            metadata=None,
            bundle_dir=None,
        )

    @classmethod
    def _build(
        cls,
        feature_spec: dict[str, Any],
        booster_path: Path,
        thresholds: dict[str, Any],
        *,
        alert_tag: str,
        export_key: str | None,
        booster_sha256: str,
        metadata: dict[str, Any] | None,
        bundle_dir: Path | None,
    ) -> Champion:
        try:
            spec = EngineSpec.from_json(feature_spec)
            model = feature_spec["model"]
            names = tuple(model["feature_names"])
            spec.assert_model_inputs(names)
            idx = spec.model_index(names)
        except (ValueError, KeyError, TypeError) as e:  # SpecError is a ValueError
            raise RefusalError(f"feature_spec.json does not describe the model inputs: {e}") from e
        if list(idx) != list(model.get("model_index", idx)):
            raise RefusalError("feature_spec.json model_index differs from the spec's positions")
        cast, call = model.get("float32_cast"), model.get("predict")
        if cast != bundle.FLOAT32_CAST or call != bundle.PREDICT:
            raise RefusalError("the bundle's float32 cast / predict call differ from this scorer's")
        try:
            booster = lgb.Booster(model_file=str(booster_path))
        except Exception as e:  # LightGBMError: missing or unreadable model file
            raise RefusalError(f"booster {booster_path} does not load: {e}") from e
        if tuple(booster.feature_name()) != names:
            raise RefusalError("booster features differ from feature_spec.json")
        try:
            tags = list(thresholds["rate_tags"])
            tag = thresholds["headline_rate_tag"] if alert_tag == "headline" else alert_tag
            if tag not in tags:
                raise RefusalError(f"alert rate tag {alert_tag!r} is not one of {tags}")
            thr = thresholds["model"][tag]["threshold"]
            rules = thresholds["rules"]
            head = rules["headline_rate_tag"]
            rule_thr = dict(rules["thresholds"][head])
        except (KeyError, TypeError) as e:
            raise RefusalError(f"thresholds.json lacks {e}") from e
        if thr is None:
            log.warning("no model alerts at rate tag %s: its threshold is null", tag)
        return cls(
            spec=spec,
            booster=booster,
            names=names,
            idx=tuple(idx),
            alert_tag=tag,
            threshold=None if thr is None else float(thr),
            rule_head_tag=head,
            rule_thresholds=rule_thr,
            export_key=export_key,
            booster_sha256=booster_sha256,
            metadata=metadata,
            bundle_dir=bundle_dir,
            thresholds=thresholds,
        )


def _file_matches(path: Path, info: Mapping[str, Any]) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == info.get("bytes")
        and sha256_file(path) == info.get("sha256")
    )


def rule_hits(sev: Sequence[float], thresholds: Mapping[str, float | None]) -> tuple[str, ...]:
    """The scenarios that fire on one event, in SCENARIOS order: M1's apply_thresholds as a
    scalar (sev >= threshold and sev > 0; a None threshold never fires)."""
    out = []
    for s, v in zip(SCENARIOS, sev, strict=True):
        t = thresholds.get(s)
        if t is not None and v >= float(t) and v > 0:
            out.append(s)
    return tuple(out)


def check_log1p(champion: Champion) -> bool | None:
    """Recompute the bundle's log1p fingerprint over the slice amounts on this host.

    False (a WARNING, never a refusal) means this libm's log1p differs from the one the
    references were computed with, so bit-exact parity is not expected here."""
    meta = champion.metadata or {}
    plat = meta.get("platform") or {}
    want = (plat.get("log1p_probe") or {}).get("sha256")
    if want is None or champion.bundle_dir is None:
        return None
    amounts = pl.read_parquet(champion.bundle_dir / bundle.SLICE, columns=["amount_usd"])
    got = bundle.log1p_fingerprint(amounts["amount_usd"].to_list())["sha256"]
    if got != want:
        log.warning(
            "log1p fingerprint differs from the bundle's (libc %s here, %s %s for the "
            "references): bit-exact parity is not expected on this host",
            " ".join(x for x in platform.libc_ver() if x) or "unknown",
            plat.get("libc"),
            plat.get("libc_version"),
        )
        return False
    return True


# --- the stream -----------------------------------------------------------------------------------


class Event(NamedTuple):
    """One message from a source."""

    fields: tuple  # spec.INPUT_COLUMNS order
    offset: int
    meta: Meta | None  # the replayer's latency headers (Kafka only)
    t_consume_ns: int  # monotonic, when the source returned it
    decode_ns: int


class Scored(NamedTuple):
    """One event's outputs."""

    offset: int
    rank: int
    row_id: int
    minute: int
    src: int
    dst: int
    amount_usd: float
    payment_format: str
    row: tuple  # spec.row_layout: the engine's output
    x: np.ndarray  # (1, n_inputs) float32 model inputs
    score: float  # raw LightGBM probability (float64)
    alert: bool
    rules: tuple[str, ...]  # rule_hits at the rules' headline tag
    flush: FlushStats | None = None  # the minute flush this event triggered
    fields: tuple = ()  # the event's spec.INPUT_COLUMNS values (banks and currencies for M6)


class Timing(NamedTuple):
    """Monotonic stamps of one event (ns); flush_ns is 0 when it triggered no flush."""

    t_consume_ns: int
    decode_ns: int
    flush_ns: int
    n_applied: int
    t_features_ns: int
    t_model_ns: int
    t_done_ns: int  # after the alert write: the end of the champion path


class Source(Protocol):
    """A rank-ordered event stream (ParquetSource inproc; the Kafka transport's KafkaSource)."""

    exhausted: bool  # no event will ever come again

    def start(self, next_offset: int) -> None: ...

    def next(self, timeout_s: float) -> Event | None: ...

    def high_watermark(self) -> int | None: ...

    def close(self) -> None: ...


class AlertSink(Protocol):
    failed: int  # delivery failures

    def write(self, rec: Mapping[str, Any]) -> bool: ...

    def flush(self, timeout_s: float) -> int: ...  # messages still outstanding

    def close(self) -> None: ...


class Observer(Protocol):
    """Called untimed after each event's t_done. Optional duck-typed hooks, all on the consumer
    thread: `bind(rt)` before the first event, `on_finish(rt)` after the final flush and the
    online parity check, `on_close(rt, result)` while the runtime closes (after the snapshot)."""

    def observe(self, ev: Event, s: Scored, t: Timing) -> None: ...


AlertHook = Callable[[Scored, Engine], None]  # M6: after the alert write, engine as of its minute


class ParquetSource:
    """The inproc transport: the events of a rank-contiguous Parquet file, in file order.

    offset = rank - base_rank: the bundle slice uses base_rank = metadata.next_rank; M6 reads
    transactions.parquet with base_rank = header next_rank - next_offset. It yields exactly the
    rows from `start(next_offset)` up to end_offset (default: the file's last rank + 1 -
    base_rank), with Python values (`to_pylist`), and never blocks.
    """

    def __init__(
        self,
        path: Path,
        base_rank: int,
        end_offset: int | None = None,
        *,
        batch_rows: int = 65_536,
    ) -> None:
        self.path = Path(path)
        self.base_rank = int(base_rank)
        if end_offset is None:
            last = pl.scan_parquet(self.path).select(pl.col("rank").max()).collect().item()
            end_offset = int(last) + 1 - self.base_rank
        self.end_offset = int(end_offset)
        self.batch_rows = batch_rows
        self.exhausted = False
        self._batches: Iterator[Any] | None = None
        self._rows: Iterator[tuple] = iter(())
        self._skip_below: int | None = None
        self._left = 0  # rows still to yield

    def start(self, next_offset: int) -> None:
        if not 0 <= next_offset <= self.end_offset:
            raise ValueError(f"next_offset {next_offset} outside [0, {self.end_offset}]")
        first = self.base_rank + next_offset
        pf = pq.ParquetFile(self.path)
        groups = _row_groups_from(pf, first)
        self._batches = pf.iter_batches(
            batch_size=self.batch_rows,
            row_groups=groups,
            columns=list(INPUT_COLUMNS),
            use_threads=False,
        )
        self._skip_below = first
        self._left = self.end_offset - next_offset
        self.exhausted = self._left == 0

    def _load(self) -> bool:
        if self._batches is None:
            raise RuntimeError("ParquetSource.start() was not called")
        for rb in self._batches:
            if self._skip_below is not None:  # leading rows before the start rank only
                keep = np.flatnonzero(rb.column("rank").to_numpy() >= self._skip_below)
                if keep.size == 0:
                    continue
                rb = rb.slice(int(keep[0]))
                self._skip_below = None
            if rb.num_rows:
                cols = [rb.column(c).to_pylist() for c in INPUT_COLUMNS]
                self._rows = zip(*cols, strict=True)
                return True
        return False

    def next(self, timeout_s: float = 0.0) -> Event | None:
        if self.exhausted:
            return None
        row = next(self._rows, None)
        while row is None:
            if not self._load():
                self.exhausted = True  # the file ended before end_offset
                return None
            row = next(self._rows, None)
        self._left -= 1
        self.exhausted = self._left == 0
        return Event(row, row[0] - self.base_rank, None, now_ns(), 0)

    def high_watermark(self) -> int | None:
        return self.end_offset

    def close(self) -> None:
        self._batches = None
        self._rows = iter(())


def _row_groups_from(pf: pq.ParquetFile, first_rank: int) -> list[int]:
    """Row groups from the first one whose rank statistics do not end before first_rank."""
    md = pf.metadata
    j = pf.schema_arrow.get_field_index("rank")
    for i in range(md.num_row_groups):
        st = md.row_group(i).column(j).statistics
        if st is None or not st.has_min_max or st.max >= first_rank:
            return list(range(i, md.num_row_groups))
    return []


@dataclass
class Progress:
    """Counters the consumer thread writes; other threads (the HTTP app) only read them."""

    start_offset: int
    end_offset: int
    events: int = 0  # scored since the restore
    alerts: int = 0
    last_offset: int | None = None
    high_watermark: int | None = None
    last_poll_ns: int | None = None
    last_consume_ns: int | None = None
    finished: bool = False

    @property
    def expected_offset(self) -> int:
        return self.start_offset if self.last_offset is None else self.last_offset + 1


def alert_record(s: Scored, ch: Champion) -> dict[str, Any]:
    """The alert as stored in SQLite and sent on the alerts topic (alerts.ALERT_COLUMNS)."""
    i = ch.spec.i_sev
    return {
        "row_id": s.row_id,
        "rank": s.rank,
        "offset": s.offset,
        "minute": s.minute,
        "day": s.minute // MINUTES_PER_DAY + 1,
        "src": s.src,
        "dst": s.dst,
        "amount_usd": s.amount_usd,
        "payment_format": s.payment_format,
        "score": s.score,
        "threshold": ch.threshold,
        "rate_tag": ch.alert_tag,
        "rules_fired": list(s.rules),
        "severities": list(s.row[i : i + N_SEV]),
        "model_version": ch.model_version,
    }


class StreamScorer:
    """Scores a rank-ordered stream on one thread, from a restored engine.

    K = header next_rank - header next_offset maps offsets to ranks; `next_offset` is the offset
    of the first pending (scored, not yet applied) event, i.e. what a snapshot taken now resumes
    from. `finish()` (the final flush) runs once, right after the event at end_offset - 1.
    """

    def __init__(
        self,
        champion: Champion,
        eng: Engine,
        header: Mapping[str, Any],
        *,
        end_offset: int,
        sinks: Sequence[AlertSink] = (),
        observers: Sequence[Observer] = (),
        on_alert: Sequence[AlertHook] = (),
        on_finish: Sequence[Callable[[StreamScorer], None]] = (),
        expected_final_digest: str | None = None,
        exit_at_end: bool = True,
    ) -> None:
        start = header.get("next_offset")
        if type(start) is not int or not 0 <= start <= end_offset:
            raise ValueError(f"header next_offset {start!r} outside [0, {end_offset}]")
        self.champion = champion
        self.eng = eng
        self.K = int(header["next_rank"]) - start
        self.start_offset = start
        self.end_offset = int(end_offset)
        self.expected_offset = start
        self.sinks = tuple(sinks)
        self.observers = tuple(observers)
        self.on_alert = tuple(on_alert)
        self.on_finish = tuple(on_finish)
        self.expected_final_digest = expected_final_digest
        self.exit_at_end = exit_at_end
        self.progress = Progress(start, self.end_offset)
        self.finished_since_restore = False
        self.final_flush: FlushStats | None = None
        self.error: str | None = None
        self._in_engine = False  # True only while an engine call that raised left it unsafe
        # Restored after the final flush (a SIGTERM after the end): nothing left to do.
        self.finished = start == self.end_offset
        self.final_digest_ok = self._digest_ok() if self.finished else None
        self.progress.finished = self.finished

    @property
    def state_valid(self) -> bool:
        """False once an exception escaped advance/process: no snapshot may be taken."""
        return not self._in_engine

    @property
    def next_offset(self) -> int:
        return (self.eng.next_rank - self.eng.pending_count) - self.K

    def _digest_ok(self) -> bool | None:
        if self.expected_final_digest is None:
            return None
        return self.eng.state_digest() == self.expected_final_digest

    def step(self, ev: Event) -> Scored:
        """Score one event; OrderError (state untouched) unless it is the next one."""
        eng, ch = self.eng, self.champion
        off = ev.offset
        if off != self.expected_offset or off >= self.end_offset:
            raise OrderError(
                "offset",
                f"offset {off}, expected {self.expected_offset} (end_offset {self.end_offset})",
            )
        try:
            ev_p = eng.prepare(*ev.fields)
        except (ValueError, TypeError, NumericRangeError) as e:
            raise OrderError("input", f"offset {off}: {e}") from e
        rank, minute = ev_p[E_RANK], ev_p[E_MINUTE]
        if rank - off != self.K or rank != eng.next_rank:
            raise OrderError(
                "rank", f"offset {off} has rank {rank}; expected {eng.next_rank} (K = {self.K})"
            )
        clock = eng.clock
        if clock is not None and minute < clock:
            raise OrderError("late", f"offset {off} minute {minute} < clock {clock}")
        # The minute flush, timed apart from the event; process() then finds clock == minute.
        flush, flush_ns = None, 0
        if clock is None or minute > clock:
            self._in_engine = True
            t = now_ns()
            flush = eng.advance(minute)
            flush_ns = now_ns() - t
            self._in_engine = False
        self._in_engine = True
        row = eng.process(ev_p)
        self._in_engine = False
        i = ch.spec.i_sev
        rules = rule_hits(row[i : i + N_SEV], ch.rule_thresholds)
        t_features = now_ns()
        x = np.asarray([row[j] for j in ch.idx], np.float64).astype(np.float32)[None]
        score = float(ch.booster.predict(x, num_threads=1)[0])
        t_model = now_ns()
        alert = ch.threshold is not None and score >= ch.threshold
        s = Scored(
            off,
            rank,
            ev_p[E_ROW_ID],
            minute,
            ev_p[E_SRC],
            ev_p[E_DST],
            ev_p[E_USD],
            ev.fields[7],
            row,
            x,
            score,
            alert,
            rules,
            flush,
            ev.fields,
        )
        if alert:
            rec = alert_record(s, ch)
            for sink in self.sinks:  # SQLite first: committed (synchronous=FULL) before t_done
                sink.write(rec)
        t_done = now_ns()
        self.expected_offset = off + 1
        p = self.progress
        p.events += 1
        p.alerts += alert
        p.last_offset = off
        p.last_consume_ns = ev.t_consume_ns
        # Untimed from here on.
        if alert:
            for hook in self.on_alert:
                hook(s, eng)
        t = Timing(
            ev.t_consume_ns,
            ev.decode_ns,
            flush_ns,
            flush.n_applied if flush is not None else 0,
            t_features,
            t_model,
            t_done,
        )
        for o in self.observers:
            o.observe(ev, s, t)
        if off == self.end_offset - 1:
            self.finish()
        return s

    def finish(self) -> FlushStats:
        """The final flush, advance(last minute + 1), once per lineage; then the digest check
        and the on_finish callbacks."""
        if self.finished or self.expected_offset != self.end_offset:
            raise RuntimeError(
                f"finish() at offset {self.expected_offset} (end {self.end_offset}, "
                f"finished {self.finished})"
            )
        eng = self.eng
        self._in_engine = True
        st = eng.advance(eng.clock + 1)
        self._in_engine = False
        self.final_flush = st
        self.finished = self.finished_since_restore = True
        self.final_digest_ok = self._digest_ok()
        for cb in self.on_finish:
            cb(self)
        # /health says "done" only once the end-of-slice parity and reports exist
        self.progress.finished = True
        return st

    def run(self, source: Source, stop: threading.Event, poll_timeout_s: float) -> str:
        """Consume until `stop` ("signal"), the end with exit_at_end ("end"), an order or codec
        error ("order_error") or any other failure ("error", logged). Without exit_at_end a
        finished scorer stays up, idle, until stopped."""
        p = self.progress
        idle_wait = max(poll_timeout_s, 0.05)
        hw_ns = 0
        try:
            while not stop.is_set():
                if self.finished and self.exit_at_end:
                    return "end"
                ev = source.next(poll_timeout_s)
                now = p.last_poll_ns = now_ns()
                if ev is None or now - hw_ns >= HW_REFRESH_NS:
                    p.high_watermark = source.high_watermark()
                    hw_ns = now
                if ev is None:
                    if source.exhausted:
                        if not self.finished:
                            raise OrderError(
                                "offset",
                                f"the source ended at offset {self.expected_offset}, before "
                                f"end_offset {self.end_offset}",
                            )
                        stop.wait(idle_wait)
                    continue
                self.step(ev)
            return "signal"
        except (OrderError, CodecError) as e:
            self.error = str(e)
            log.error("the stream stopped: %s", e)
            return "order_error"
        except Exception as e:
            self.error = repr(e)
            log.exception("the scorer failed")
            return "error"


# --- online parity --------------------------------------------------------------------------------


class _Outputs:
    """Untimed observer: this incarnation's outputs per offset, for the online parity check."""

    def __init__(self, spec: EngineSpec, n_inputs: int) -> None:
        self.n_inputs = n_inputs
        self.i_sev = spec.i_sev
        self.i_trunc = (spec.i_rule_trunc, spec.i_cyc_trunc, spec.i_sg_trunc)
        self.first: int | None = None
        self.n = 0
        self.row_id = array("q")
        self.x = bytearray()
        self.sev = array("d")
        self.trunc = array("q")
        self.p = array("d")

    def observe(self, ev: Event, s: Scored, t: Timing) -> None:
        if self.first is None:
            self.first = s.offset
        elif s.offset != self.first + self.n:
            raise AssertionError(f"parity rows must be contiguous: offset {s.offset}")
        row = s.row
        self.row_id.append(s.row_id)
        self.x += s.x.tobytes()
        self.sev.extend(row[self.i_sev : self.i_sev + N_SEV])
        self.trunc.extend(int(row[i]) for i in self.i_trunc)
        self.p.append(s.score)
        self.n += 1

    def arrays(self) -> dict[str, Any]:
        m = self.n
        return {
            "x32": np.frombuffer(bytes(self.x), dtype=np.float32).reshape(m, self.n_inputs),
            "sev": np.array(self.sev, dtype=np.float64).reshape(m, N_SEV),
            "trunc": np.array(self.trunc, dtype=np.int64).reshape(m, 3),
            "p": np.array(self.p, dtype=np.float64),
            "row_id": pl.Series("row_id", np.array(self.row_id, dtype=np.int64)),
        }


def parity_ok(parity: Mapping[str, Any] | None) -> bool:
    """No mismatch, no failed alerts or digest check, no checker error."""
    return (
        parity is not None
        and parity.get("error") is None
        and not any(parity["mismatches"].values())
        and parity.get("alerts_ok") is not False
        and parity.get("digest_ok") is not False
    )


# --- runtime --------------------------------------------------------------------------------------


class Runtime:
    """The scorer's wiring (bundle, state, sinks, source, online parity), shared by the CLI, the
    app thread and the tests. Build it with `open_runtime`, then `run` and `close` (or `abort`)
    it on the same thread: the engine, the SQLite writer and the snapshot stay on that thread.
    """

    def __init__(
        self,
        settings: Settings,
        champion: Champion,
        store: StateStore,
        restored: Restored,
        *,
        end_offset: int,
        log1p_ok: bool | None,
    ) -> None:
        self.settings = settings
        self.champion = champion
        self.store = store
        self.restored = restored
        self.end_offset = end_offset
        self.start_offset = restored.next_offset
        self.log1p_ok = log1p_ok
        self.alert_store: SqliteAlertStore | None = None
        self.case_hook: CaseHook | None = None  # M6: None when case packs are off
        self.sinks: list[AlertSink] = []
        self.source: Source | None = None
        self.scorer: StreamScorer | None = None
        self.observers: list[Observer] = []  # the caller's (the parity recorder is internal)
        self.outputs = _Outputs(champion.spec, len(champion.names))
        self.parity: dict[str, Any] | None = None
        self.snapshot: dict[str, Any] | None = None  # {next_offset, seconds, bytes} when written
        self.db_alerts: int | None = None
        self.result: str | None = None
        self.exit_code: int | None = None
        self._released = False

    @property
    def progress(self) -> Progress:
        return self.scorer.progress

    def _hooks(self, name: str) -> list[Callable[..., Any]]:
        return [getattr(o, name) for o in self.observers if callable(getattr(o, name, None))]

    def run(self, stop: threading.Event) -> str:
        self.result = self.scorer.run(self.source, stop, self.settings.cfg.kafka.poll_timeout_s)
        return self.result

    def _on_finish(self, scorer: StreamScorer) -> None:
        self.parity = self._check_parity(final=True)
        log.info("end of the slice: parity %s", _parity_line(self.parity))
        for hook in self._hooks("on_finish"):
            hook(self)

    def _check_parity(self, *, final: bool) -> dict[str, Any]:
        """This incarnation's outputs against reference/*.parquet over the offsets it covered;
        at the end of the slice also the alerts table and the final state digest."""
        out, sc = self.outputs, self.scorer
        lo, m = out.first, out.n
        doc: dict[str, Any] = {
            "checked": m,
            "covered": [lo, lo + m - 1] if m else None,
            "mismatches": dict.fromkeys(bundle.CHECKS, 0),
            "alerts_ok": None,
            "alerts": None,
            "digest_ok": sc.final_digest_ok if sc.finished else None,
            "error": None,
        }
        try:
            if m:
                refs = bundle.reference_slice(self.champion.bundle_dir, lo, m)
                doc["mismatches"] = bundle.compare_outputs(**out.arrays(), **refs)
            if final:
                doc["alerts"] = self._alerts_check()
                doc["alerts_ok"] = doc["alerts"]["ok"]
        except Exception as e:
            log.exception("the online parity check failed")
            doc["error"] = repr(e)
        return doc

    def _alerts_check(self) -> dict[str, Any]:
        """The alerts table against the reference alert column at the alert tag over [0, n):
        the same row_ids, score bits and rules_fired."""
        d, n, tag = self.champion.bundle_dir, self.end_offset, self.champion.alert_tag
        fired = [f"fired_{s}" for s in SCENARIOS]
        ref = pl.scan_parquet(d / bundle.REF_ALERTS).select("row_id", f"alert_{tag}", *fired)
        sc = pl.scan_parquet(d / bundle.REF_SCORES).select("row_id", bundle.SCORE)
        ref, sc = ref.head(n).collect(), sc.head(n).collect()
        if not ref["row_id"].equals(sc["row_id"]):
            raise RuntimeError("reference alerts and scores hold different rows")
        want = {
            r["row_id"]: (r[bundle.SCORE], [s for s in SCENARIOS if r[f"fired_{s}"]])
            for r in ref.with_columns(sc[bundle.SCORE]).filter(pl.col(f"alert_{tag}")).to_dicts()
        }
        got = {r["row_id"]: r for r in self.alert_store.rows()}
        both = want.keys() & got.keys()
        doc = {
            "db_count": len(got),
            "ref_count": len(want),
            "missing": len(want.keys() - got.keys()),
            "extra": len(got.keys() - want.keys()),
            "score_mismatch": sum(_bits(got[k]["score"]) != _bits(want[k][0]) for k in both),
            "fired_mismatch": sum(got[k]["rules_fired"] != want[k][1] for k in both),
        }
        checks = ("missing", "extra", "score_mismatch", "fired_mismatch")
        doc["ok"] = not any(doc[k] for k in checks)
        return doc

    def close(self, result: str) -> int:
        """Stop and return the exit code (signal/end 0, end with a parity failure 4, order_error
        3, error 1). Order: alerts already committed in SQLite; drain the alerts producer; the
        snapshot (only on a signal, with a valid state, after at least one event or the final
        flush since the restore, and only if every alert was delivered); the observers'
        on_close (raw latency files); release the source and sinks."""
        if self.exit_code is not None:
            return self.exit_code
        sc = self.scorer
        code = EXIT_CODES.get(result, 1)
        if self.parity is None and self.outputs.n:
            self.parity = self._check_parity(final=False)
        undelivered = 0
        for sink in self.sinks:
            try:
                undelivered += sink.flush(SINK_FLUSH_S) + sink.failed
            except Exception:
                log.exception("alert sink flush failed")
                undelivered += 1
        want = (
            result == "signal"
            and sc.state_valid
            and (sc.progress.events > 0 or sc.finished_since_restore)
        )
        if want and undelivered:
            log.warning(
                "%d alerts not delivered: snapshot skipped; the restart re-produces them",
                undelivered,
            )
        elif want:
            try:
                self._save_snapshot()
            except Exception:
                log.exception("snapshot failed")
                code = 1
        for hook in self._hooks("on_close"):
            try:
                hook(self, result)
            except Exception:
                log.exception("observer on_close failed")
        if result == "end" and not parity_ok(self.parity):
            code = EXIT_PARITY
        try:
            self.db_alerts = self.alert_store.count()
        except Exception:
            log.exception("alerts count failed")
        h = self.case_hook
        if h is not None and h.failed:
            log.warning("%d case packs failed (last: %s)", h.failed, h.last_error)
        self._release()
        self.exit_code = code
        log.info(
            "stopped (%s) at offset %d: %d events, %d alerts, exit %d",
            result,
            sc.expected_offset,
            sc.progress.events,
            sc.progress.alerts,
            code,
        )
        return code

    def _save_snapshot(self) -> None:
        sc, ch = self.scorer, self.champion
        nxt = sc.next_offset
        t0 = time.perf_counter()
        side = self.store.save(
            sc.eng,
            next_offset=nxt,
            extra={
                "writer": __name__,
                "export_key": ch.export_key,
                "booster_sha256": ch.booster_sha256,
                "spec_hash": ch.spec.spec_hash(),
                "end_offset": self.end_offset,
                "transport": self.settings.transport,
                "finished": sc.finished,
            },
        )
        self.snapshot = {
            "next_offset": nxt,
            "seconds": time.perf_counter() - t0,
            "bytes": side["file_nbytes"],
        }
        log.info("snapshot at next_offset %d (%d bytes)", nxt, side["file_nbytes"])

    def abort(self) -> None:
        """Release the handles with no flush, snapshot or hooks (tests: a SIGKILL-like stop)."""
        self._release()
        self.exit_code = -1

    def _release(self) -> None:
        if self._released:
            return
        self._released = True
        for closer in (self.source, *self.sinks):
            if closer is None:
                continue
            try:
                closer.close()
            except Exception:
                log.exception("close failed")

    def summary(self) -> dict[str, Any]:
        """The CLI's one JSON line."""
        sc, h = self.scorer, self.case_hook
        return {
            "transport": self.settings.transport,
            "origin": self.restored.origin,
            "start_offset": self.start_offset,
            "next_offset": sc.next_offset if sc.state_valid else None,
            "end_offset": self.end_offset,
            "events": sc.progress.events,
            "alerts": sc.progress.alerts,
            "db_alerts": self.db_alerts,
            "cases": None if h is None else {"built": h.built, "failed": h.failed},
            "result": self.result,
            "error": sc.error,
            "parity": self.parity,
            "snapshot": self.snapshot,
            "log1p_ok": self.log1p_ok,
            "exit_code": self.exit_code,
        }


def _parity_line(parity: Mapping[str, Any] | None) -> str:
    if parity is None:
        return "not checked"
    bad = {k: v for k, v in parity["mismatches"].items() if v}
    error = f", error {parity['error']}" if parity["error"] else ""
    return (
        f"{parity['checked']} rows, mismatches {bad or 0}, alerts_ok {parity['alerts_ok']}, "
        f"digest_ok {parity['digest_ok']}{error}"
    )


def _bits(x: float) -> bytes:
    return struct.pack("<d", x)


def kafka_source(settings: Settings, end_offset: int) -> Source:
    """The Kafka transport's source (B's KafkaSource), imported only here."""
    from aml.serving.consumer import KafkaSource  # confluent_kafka: the Kafka path only

    return KafkaSource(
        settings.bootstrap,
        settings.transactions_topic,
        group_id=settings.group_id,
        poll_timeout_s=settings.cfg.kafka.poll_timeout_s,
    )


def open_case_hook(
    settings: Settings, champion: Champion, sink: Callable[[dict[str, Any]], Any]
) -> CaseHook | None:
    """The M6 case-pack hook (lenient: failures are logged and counted) writing into `sink`, or
    None when `settings.case_packs` is off or no explain.yaml sits next to serving.yaml (a
    WARNING). An explain.yaml that does not load is a ConfigError (exit 2)."""
    if not settings.case_packs:
        return None
    path = settings.config_path.with_name(EXPLAIN_CONFIG_FILE)
    if not path.is_file():
        log.warning(
            "case packs are off: no %s next to %s", EXPLAIN_CONFIG_FILE, settings.config_path
        )
        return None
    try:
        cfg = load_config(path)
    except Exception as e:  # OSError, YAMLError, ValueError: the operator's file
        raise ConfigError(f"{path}: {e}") from e
    return case_hook(champion, cfg, sink)


def open_runtime(
    settings: Settings,
    *,
    observers: Sequence[Observer] = (),
    on_alert: Sequence[AlertHook] = (),
    source_factory: Callable[[Settings, int], Source] | None = None,
) -> Runtime:
    """Load the champion, probe log1p, restore the state (runtime snapshot, else the bundle's),
    open the alert sinks, the case-pack hook, the source and the scorer, bind the observers and
    start the source at the restored next_offset. Call it on the thread that will run and close
    the runtime.

    `source_factory(settings, end_offset)` overrides the transport's source (default: the
    bundle slice inproc, `kafka_source` on Kafka). The case-pack hook (`open_case_hook`) runs
    before the caller's `on_alert` hooks. Refusals raise RefusalError, RuntimeStateError or
    ConfigError (`startup_exit_code` maps them to 2)."""
    if settings.cfg.challenger != "none":
        raise RefusalError("a challenger is Stretch M4: only challenger 'none' is served")
    champion = Champion.from_bundle(settings.bundle_dir, alert_tag=settings.alert_rate_tag)
    meta = champion.metadata
    n = settings.demo_events(int(meta["rows"]["slice"]))
    log1p_ok = check_log1p(champion)
    store = StateStore(settings.runtime_dir)
    restored = store.restore(champion, end_offset=n)
    rt = Runtime(settings, champion, store, restored, end_offset=n, log1p_ok=log1p_ok)
    try:
        rt.alert_store = SqliteAlertStore(store.alerts_db)
        rt.sinks.append(rt.alert_store)
        if settings.transport == "kafka":
            rt.sinks.append(KafkaAlertSink(settings.bootstrap, settings.alerts_topic))
        rt.case_hook = open_case_hook(settings, champion, rt.alert_store.write_case)
        hooks = [] if rt.case_hook is None else [rt.case_hook]
        if source_factory is not None:
            rt.source = source_factory(settings, n)
        elif settings.transport == "kafka":
            rt.source = kafka_source(settings, n)
        else:
            base = int(meta["next_rank"])
            rt.source = ParquetSource(settings.bundle_dir / bundle.SLICE, base, n)
        rt.observers = list(observers)
        full = n == int(meta["rows"]["slice"])
        rt.scorer = StreamScorer(
            champion,
            restored.eng,
            restored.header,
            end_offset=n,
            sinks=rt.sinks,
            observers=[rt.outputs, *rt.observers],
            on_alert=[*hooks, *on_alert],
            on_finish=[rt._on_finish],
            expected_final_digest=(meta.get("verification") or {}).get("final_state_digest")
            if full
            else None,
            exit_at_end=settings.exit_at_end,
        )
        if rt.scorer.finished:  # restored after the final flush: check what can be checked
            rt.parity = rt._check_parity(final=True)
        # start the source first: a refused start must leave no "running" session behind
        rt.source.start(rt.start_offset)
        for hook in rt._hooks("bind"):
            hook(rt)
    except BaseException:
        rt._release()
        raise
    log.info(
        "%s transport: restored the %s snapshot (%.2f s) at offset %d of %d; alert tag %s; "
        "case packs %s",
        settings.transport,
        restored.origin,
        restored.seconds,
        rt.start_offset,
        n,
        champion.alert_tag,
        "off" if rt.case_hook is None else "on",
    )
    return rt


# --- CLI ------------------------------------------------------------------------------------------


def self_check() -> dict[str, Any]:
    """Image check (no bundle): the serving imports, no torch, versions, libc, log1p digest."""
    for name in SELF_CHECK_MODULES:
        importlib.import_module(name)
    if importlib.util.find_spec("torch") is not None:
        raise RuntimeError("torch is installed: the serving image must be CPU-only, without it")
    librdkafka = importlib.import_module("confluent_kafka").libversion()[0]
    return {
        "python": platform.python_version(),
        "libc": list(platform.libc_ver()),
        "machine": platform.machine(),
        "versions": {d: importlib_metadata.version(d) for d in SELF_CHECK_DISTS},
        "librdkafka": librdkafka,
        "torch_absent": True,
        "log1p_fixed": bundle.log1p_fingerprint(SELF_CHECK_LOG1P),
    }


def add_case_packs_arg(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
    """--case-packs / --no-case-packs (the scorer CLI and the app; default on)."""
    p.add_argument(
        "--case-packs",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="build a case pack per alert into the cases table (default: on)",
    )
    return p


def main(argv: Sequence[str] | None = None, *, observers: Sequence[Observer] = ()) -> int:
    """python -m aml.serving.scorer --transport {inproc,kafka} [--no-case-packs --bundle
    --runtime --max-events ...] | --self-check. Runs on the main thread (the consumer thread
    here); SIGTERM / SIGINT stop it; prints one JSON line and returns the exit code."""
    p = argparse.ArgumentParser(prog="python -m aml.serving.scorer", description="M5 scorer")
    p.add_argument("--transport", choices=TRANSPORTS)
    p.add_argument(
        "--exit-at-end",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="stop after the last event (default: on for inproc, off for kafka)",
    )
    add_case_packs_arg(p)
    p.add_argument("--self-check", action="store_true", help="check the image and exit")
    add_cli_args(p)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.self_check:
        try:
            print(json.dumps(self_check(), sort_keys=True), flush=True)
        except Exception:
            log.exception("self-check failed")
            return 1
        return 0
    if args.transport is None:
        p.error("--transport is required (or --self-check)")
    stop = threading.Event()

    def _stop(signum: int, frame: Any) -> None:
        stop.set()

    signal.signal(signal.SIGTERM, _stop)
    signal.signal(signal.SIGINT, _stop)
    try:
        rt = open_runtime(load_settings(args), observers=observers)
    except Exception as e:
        code = startup_exit_code(e)
        if code == EXIT_REFUSED:
            log.error("refused: %s", e)
        else:
            log.exception("the scorer did not start")
        print(json.dumps({"exit_code": code, "error": str(e)}), flush=True)
        return code
    code = rt.close(rt.run(stop))
    print(json.dumps(rt.summary(), sort_keys=True), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
