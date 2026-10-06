"""The M5 replayer: the bundle's replay slice onto the `transactions` topic, open loop.

Core module: `confluent_kafka` is imported only where a real producer is built, so tests drive
the send loop with a fake producer and a fake clock.

Every send time comes from a schedule computed before the point starts and is never shifted by
feedback (open loop). A sender that falls behind sends at once and keeps the scheduled stamp, so
the latency of late events is measured from when they should have been sent (no coordinated
omission). Each message carries the headers `t_sched` (scheduled), `t_prod` (just before
produce) and `point` (codec.pack_meta).

Shapes: uniform `s_k = k / R`; unpaced `s_k = 0`; trace: the slice's own within-minute spacing
with simulated idle gaps longer than `gap_cap_s` cut to it, rescaled so the point spans exactly
`(n - 1) / R`. Before every point but the first (and after a resume) the producer is flushed and
a barrier waits until the scorer's `/health` reports `last_offset >= point start - 1`, then
`settle_s`: points never overlap.

Resume: a restarted replayer reads the topic's watermarks and continues at the high watermark
after checking that the last message there is the slice event of that position.
"""

from __future__ import annotations

import argparse
import http.client
import json
import logging
import signal
import sys
import threading
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from aml.features.spec import INPUT_COLUMNS
from aml.io import read_json
from aml.serving.settings import (
    ConfigError,
    Point,
    Settings,
    add_cli_args,
    load_settings,
    plan_points,
)
from aml.streaming import kafka
from aml.streaming.codec import (
    CodecError,
    Meta,
    decode_event,
    encode_event,
    event_key,
    pack_meta,
)

log = logging.getLogger(__name__)
# The scorer is always local (compose network or 127.0.0.1): never route /health through a proxy.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))

# The bundle layout (aml.serving.bundle; not imported: it pulls in the training stack).
METADATA_FILE = "metadata.json"
SLICE = "replay/slice.parquet"
FLUSH_S = 60.0
STOP_FLUSH_S = 10.0
BUFFER_POLL_S = 0.01
BARRIER_POLL_S = 0.2
HEALTH_TIMEOUT_S = 2.0
EXIT_REFUSED = 2


class ReplayError(RuntimeError):
    """The slice or the topic is not what this run expects (exit 1)."""


# --- inputs ---------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplaySlice:
    """The first n slice events, pre-encoded: message i is the event of rank base_rank + i."""

    base_rank: int
    ranks: np.ndarray  # int64
    minutes: np.ndarray  # int64, non-decreasing
    keys: list[bytes]
    values: list[bytes]

    @property
    def n(self) -> int:
        return len(self.values)


def slice_rows(bundle_dir: Path) -> int:
    """metadata.rows.slice of the bundle."""
    path = Path(bundle_dir) / METADATA_FILE
    if not path.is_file():
        raise ReplayError(f"no serving bundle at {bundle_dir} ({METADATA_FILE} missing)")
    return int(read_json(path)["rows"]["slice"])


def load_slice(bundle_dir: Path, n: int) -> ReplaySlice:
    """Read and encode the first n events of the replay slice; ReplayError unless their ranks
    are contiguous from metadata.next_rank and their minutes non-decreasing."""
    d = Path(bundle_dir)
    base = int(read_json(d / METADATA_FILE)["next_rank"])
    df = pl.scan_parquet(d / SLICE).select(*INPUT_COLUMNS).head(n).collect()
    if df.height != n:
        raise ReplayError(f"{d / SLICE} has {df.height} rows, fewer than the {n} to replay")
    ranks = df["rank"].to_numpy().astype(np.int64)
    minutes = df["minute"].to_numpy().astype(np.int64)
    if not np.array_equal(ranks, np.arange(base, base + n, dtype=np.int64)):
        raise ReplayError(f"slice ranks are not contiguous from next_rank {base}")
    if n > 1 and bool((np.diff(minutes) < 0).any()):
        raise ReplayError("slice minutes decrease: the slice is not in event order")
    values = [encode_event(row) for row in df.iter_rows()]
    keys = [event_key(r) for r in df["row_id"].to_list()]
    return ReplaySlice(base, ranks, minutes, keys, values)


# --- schedule -------------------------------------------------------------------------------------


def trace_offsets(minutes: np.ndarray, *, gap_cap_s: float) -> np.ndarray:
    """sigma_k (float64 s): the slice's own timeline, gaps longer than gap_cap_s cut to it.

    Event k, the j-th of c events in minute m, sits at tau_k = 60 m + 60 (j + 0.5) / c, i.e.
    spread evenly inside its minute; sigma_0 = 0 and sigma_k = sum of min(tau_i - tau_(i-1),
    gap_cap_s) for i <= k. Minutes must be non-decreasing.
    """
    m = np.asarray(minutes, dtype=np.int64)
    n = m.size
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    starts = np.flatnonzero(np.r_[True, m[1:] != m[:-1]])
    counts = np.diff(np.r_[starts, n])
    j = np.arange(n) - np.repeat(starts, counts)
    tau = 60.0 * m + 60.0 * (j + 0.5) / np.repeat(counts, counts)
    gaps = np.minimum(np.diff(tau), gap_cap_s)
    return np.concatenate([[0.0], np.cumsum(gaps)])


def schedule_ns(p: Point, sigma: np.ndarray | None = None) -> np.ndarray:
    """Send-time offsets (int64 ns from the point's start) of the point's events.

    uniform: k / R; unpaced: 0; trace: sigma over [p.start, p.end) shifted to 0 and rescaled to
    span (n - 1) / R, falling back to uniform for a single event or a zero span."""
    n = p.events
    if p.shape == "unpaced":
        return np.zeros(n, dtype=np.int64)
    if p.rate is None or p.rate <= 0:
        raise ValueError(f"point {p.index}: a {p.shape} point needs a rate > 0")
    k = np.arange(n, dtype=np.float64)
    secs_e9 = k * 1e9 / p.rate
    if p.shape == "trace":
        if sigma is None:
            raise ValueError("a trace point needs the slice's trace_offsets")
        s = np.asarray(sigma[p.start : p.end], dtype=np.float64) - sigma[p.start]
        span = float(s[-1]) if n else 0.0
        if n > 1 and span > 0:
            secs_e9 = s * ((n - 1) * 1e9 / (p.rate * span))
    elif p.shape != "uniform":
        raise ValueError(f"point {p.index}: unknown shape {p.shape!r}")
    return np.rint(secs_e9).astype(np.int64)


# --- topic tail and resume ------------------------------------------------------------------------


def topic_tail(bootstrap: str, topic: str) -> tuple[int, int, int | None]:
    """(low, high, rank of the message at high - 1 or None when the topic is empty)."""
    lo, hi = kafka.watermarks(bootstrap, topic)
    if hi <= 0:
        return lo, hi, None
    got = kafka.read_at(bootstrap, topic, hi - 1)
    if got is None:
        raise ReplayError(f"could not read message {hi - 1} of topic {topic!r}")
    try:
        rank = decode_event(got[1])[0]
    except CodecError as e:
        raise ReplayError(f"message {hi - 1} of topic {topic!r} is not an event: {e}") from e
    return lo, hi, rank


def resume_index(lo: int, hi: int, n: int, tail_rank: int | None, ranks: np.ndarray) -> int:
    """The slice index to send next: hi, after checking that the topic holds this slice's
    first hi events (lo == 0, hi <= n, the message at hi - 1 has rank ranks[hi - 1])."""
    if lo != 0 or not 0 <= hi <= n:
        raise ReplayError(
            f"topic offsets [{lo}, {hi}) are not a prefix of this {n}-event replay: run init"
        )
    if hi > 0 and tail_rank != int(ranks[hi - 1]):
        raise ReplayError(
            f"the message at offset {hi - 1} has rank {tail_rank}, the slice's has "
            f"{int(ranks[hi - 1])}: the topic is not this run's replay (run init)"
        )
    return hi


# --- barrier --------------------------------------------------------------------------------------


def read_health(url: str, *, timeout_s: float = HEALTH_TIMEOUT_S) -> dict[str, Any] | None:
    """GET {url}/health as a dict (a 503 carries its JSON body too); None when unreachable."""
    try:
        with _OPENER.open(url.rstrip("/") + "/health", timeout=timeout_s) as r:
            body = r.read()
    except urllib.error.HTTPError as e:
        with e:
            body = e.read()
    except (OSError, http.client.HTTPException):  # refused, reset, timed out, cut short
        return None
    try:
        doc = json.loads(body)
    except ValueError:
        return None
    return doc if isinstance(doc, dict) else None


def scorer_barrier(
    url: str,
    last_offset: int,
    *,
    timeout_s: float,
    poll_s: float = BARRIER_POLL_S,
    stop: threading.Event | None = None,
) -> bool:
    """Poll the scorer's /health until its `last_offset` reaches `last_offset` (True); False on
    timeout or stop. Unreachable or 503 answers are retried; ReplayError once it reports
    `failed`."""
    deadline = time.monotonic() + timeout_s
    while True:
        doc = read_health(url)
        if doc is not None:
            got = doc.get("last_offset")
            if isinstance(got, int) and got >= last_offset:
                return True
            if doc.get("status") == "failed":
                raise ReplayError(f"the scorer failed (exit code {doc.get('exit_code')})")
        if time.monotonic() >= deadline:
            return False
        if stop is not None:
            if stop.wait(poll_s):
                return False
        else:
            time.sleep(poll_s)


# --- send loop ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayResult:
    n: int  # events in this run's replay
    first_index: int  # where this replayer started (the resume index)
    sent: int
    resumed: bool
    complete: bool  # the topic holds all n events
    stopped: bool  # a stop (SIGTERM) ended the send loop early
    delivery_errors: int
    outstanding: int  # still queued after the final flush
    points: tuple[dict[str, Any], ...]  # per point: sent, send-lag p99 / max (ms), barrier

    def summary(self) -> dict[str, Any]:
        return {**asdict(self), "points": list(self.points)}


def _check_plan(plan: Sequence[Point], n: int) -> None:
    end = 0
    for p in plan:
        if p.start != end or p.events < 1:
            raise ValueError(f"plan point {p.index} starts at {p.start}, expected {end}")
        end = p.end
    if end > n:
        raise ValueError(f"the plan covers {end} events, the replay has {n}")


def _lag_stats(lags: list[int]) -> dict[str, float | None]:
    """Send lag (t_prod - t_sched) p99 (nearest rank) and max, in ms."""
    if not lags:
        return {"send_lag_p99_ms": None, "send_lag_max_ms": None}
    a = np.asarray(lags, dtype=np.int64)
    p99 = np.quantile(a, 0.99, method="inverted_cdf")
    return {"send_lag_p99_ms": float(p99) / 1e6, "send_lag_max_ms": float(a.max()) / 1e6}


class _Sender:
    """The open-loop send loop of one replayer run."""

    def __init__(
        self,
        producer: Any,
        topic: str,
        sl: ReplaySlice,
        *,
        clock: Callable[[], int],
        sleep: Callable[[float], Any],
        stop: threading.Event | None,
    ) -> None:
        self.producer = producer
        self.topic = topic
        self.sl = sl
        self.clock = clock
        self.sleep = sleep
        self.stop = stop
        self.failed = 0  # delivery reports with an error

    def stopping(self) -> bool:
        return self.stop is not None and self.stop.is_set()

    def on_delivery(self, err: Any, msg: Any) -> None:
        if err is not None:
            self.failed += 1
            if self.failed <= 5:
                log.error("delivery failed: %s", err)

    def send_point(
        self, p: Point, k0: int, sched: list[int], lead_ns: int
    ) -> tuple[list[int], int]:
        """Send events k0.. of point p at t_start + sched[k] - sched[k0], t_start = now +
        lead_ns; (send lags, BufferError retries). Stops before a send once the stop is set."""
        producer, clock, values, keys = self.producer, self.clock, self.sl.values, self.sl.keys
        lags: list[int] = []
        retries = 0
        base = sched[k0]
        t_start = clock() + lead_ns
        for k in range(k0, p.events):
            target = t_start + sched[k] - base
            d = target - clock()
            if d > 0:
                self.sleep(d / 1e9)  # never a busy spin
            if self.stopping():
                break
            i = p.start + k
            while True:
                t_prod = clock()
                try:
                    producer.produce(
                        self.topic,
                        value=values[i],
                        key=keys[i],
                        partition=0,
                        headers=pack_meta(Meta(target, t_prod, p.index)),
                        on_delivery=self.on_delivery,
                    )
                    break
                except BufferError:  # the local queue is full: serve deliveries, keep target
                    retries += 1
                    producer.poll(BUFFER_POLL_S)
            producer.poll(0)
            lags.append(t_prod - target)
        return lags, retries


def run_replayer(
    settings: Settings,
    *,
    unpaced: bool = False,
    plan: Sequence[Point] | None = None,
    producer_factory: Callable[[dict[str, Any]], Any] | None = None,
    tail_reader: Callable[[], tuple[int, int, int | None]] | None = None,
    barrier: Callable[[int], bool] | None = None,
    clock: Callable[[], int] = time.monotonic_ns,
    sleep: Callable[[float], Any] | None = None,
    stop: threading.Event | None = None,
) -> ReplayResult:
    """Replay the slice per `plan` (default: plan_points of serving.yaml; `unpaced`: one unpaced
    point) onto the transactions topic, resuming at the topic's high watermark.

    `tail_reader()` -> (low, high, rank at high - 1) defaults to `topic_tail`.
    `barrier(last_offset)` defaults to the scorer's /health at settings.scorer_url (none without
    a URL: the points are then only separated by a producer flush). `sleep` defaults to
    stop.wait (a stop interrupts it) or time.sleep."""
    topic = settings.transactions_topic
    n = settings.demo_events(slice_rows(settings.bundle_dir))
    sl = load_slice(settings.bundle_dir, n)
    if plan is None:
        plan = (Point(0, "unpaced", None, n, 0),) if unpaced else plan_points(settings.cfg, n)
    _check_plan(plan, n)
    rcfg = settings.cfg.replayer
    if sleep is None:
        sleep = stop.wait if stop is not None else time.sleep
    if barrier is None and settings.scorer_url:
        url = settings.scorer_url

        def barrier(last: int) -> bool:
            return scorer_barrier(url, last, timeout_s=rcfg.barrier_timeout_s, stop=stop)

    if tail_reader is None:

        def tail_reader() -> tuple[int, int, int | None]:
            return topic_tail(settings.bootstrap, topic)

    lo, hi, tail_rank = tail_reader()
    first = resume_index(lo, hi, n, tail_rank, sl.ranks)
    resumed = first > 0
    if first == n:
        log.info("topic %s already holds all %d events: nothing to send", topic, n)
        return ReplayResult(
            n=n,
            first_index=first,
            sent=0,
            resumed=resumed,
            complete=True,
            stopped=False,
            delivery_errors=0,
            outstanding=0,
            points=(),
        )
    if resumed:
        log.info("resuming at offset %d of %d (topic %s)", first, n, topic)
    if barrier is None and len(plan) > 1:
        log.warning("no scorer URL: points are separated by a producer flush only")

    if producer_factory is None:
        from confluent_kafka import Producer  # the Kafka path only (core-import rule)

        producer_factory = Producer
    config = kafka.producer_config(settings.bootstrap, client_id="aml-replayer")
    sender = _Sender(producer_factory(config), topic, sl, clock=clock, sleep=sleep, stop=stop)
    producer = sender.producer
    sigma = None
    if any(p.shape == "trace" for p in plan):
        sigma = trace_offsets(sl.minutes, gap_cap_s=rcfg.gap_cap_s)
    lead_ns = round(rcfg.lead_s * 1e9)
    points: list[dict[str, Any]] = []
    sent = 0
    for p in plan:
        if p.end <= first:
            continue
        if sender.stopping():
            break
        k0 = max(first - p.start, 0)
        doc: dict[str, Any] = {
            "index": p.index,
            "shape": p.shape,
            "rate": p.rate,
            "start": p.start,
            "events": p.events,
            "barrier_ok": None,
            "barrier_s": None,
        }
        if points or resumed:  # drain the previous point (or a stopped replayer's backlog)
            producer.flush(FLUSH_S)
            if barrier is not None:
                t0 = time.monotonic()
                ok = barrier(p.start + k0 - 1)
                doc["barrier_ok"], doc["barrier_s"] = ok, round(time.monotonic() - t0, 3)
                if not ok and not sender.stopping():
                    log.warning("barrier before point %d timed out: points overlap", p.index)
                sleep(rcfg.settle_s)
        if p.shape == "trace" and p.events > 1:
            speed = p.rate * float(sigma[p.end - 1] - sigma[p.start]) / (p.events - 1)
            log.info("point %d (trace): simulated time runs %.1fx", p.index, speed)
        lags, retries = sender.send_point(p, k0, schedule_ns(p, sigma).tolist(), lead_ns)
        sent += len(lags)
        points.append({**doc, "sent": len(lags), "backpressure": retries, **_lag_stats(lags)})
        log.info("point %d (%s): %d events sent", p.index, p.shape, len(lags))
    stopped = sender.stopping() and first + sent < n
    outstanding = int(producer.flush(STOP_FLUSH_S if stopped else FLUSH_S))
    if outstanding:
        log.warning("%d messages still queued after the final flush", outstanding)
    errors = sender.failed + (0 if stopped else outstanding)
    return ReplayResult(
        n=n,
        first_index=first,
        sent=sent,
        resumed=resumed,
        complete=first + sent == n and errors == 0,
        stopped=stopped,
        delivery_errors=errors,
        outstanding=outstanding,
        points=tuple(points),
    )


# --- CLI ------------------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    """python -m aml.streaming.replayer [--bundle --config --bootstrap --scorer-url --max-events
    --unpaced]: one JSON summary line; exit 1 on delivery errors or a foreign topic."""
    p = argparse.ArgumentParser(prog="python -m aml.streaming.replayer", description="M5 replayer")
    add_cli_args(p)
    p.add_argument("--unpaced", action="store_true", help="send every event as one unpaced point")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    stop = threading.Event()

    def _stop(signum: int, frame: Any) -> None:
        stop.set()

    previous = {s: signal.signal(s, _stop) for s in (signal.SIGTERM, signal.SIGINT)}
    try:
        settings = load_settings(args, transport="kafka")
        result = run_replayer(settings, unpaced=args.unpaced, stop=stop)
    except ConfigError as e:
        log.error("refused: %s", e)
        print(json.dumps({"exit_code": EXIT_REFUSED, "error": str(e)}), flush=True)
        return EXIT_REFUSED
    except Exception as e:
        log.exception("the replayer failed")
        print(json.dumps({"exit_code": 1, "error": str(e)}), flush=True)
        return 1
    finally:
        for s, h in previous.items():  # in-process callers (tests) get their handlers back
            signal.signal(s, h)
    code = 1 if result.delivery_errors else 0
    print(json.dumps({**result.summary(), "exit_code": code}, sort_keys=True), flush=True)
    return code


if __name__ == "__main__":
    sys.exit(main())
