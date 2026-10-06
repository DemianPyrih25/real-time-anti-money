"""The M5 scorer service: FastAPI around the consumer thread (`python -m aml.serving.app`).

One non-daemon thread (`ScorerThread`) runs `open_runtime`, `Runtime.run` and `Runtime.close`,
so the engine, the SQLite writer and the shutdown snapshot all stay on it. The HTTP handlers only
read its progress, and the alerts endpoints open their own read-only SQLite connection per
request. Endpoints: `/health` (200 ok or done, else 503, always with the same JSON body),
`/metrics` (Prometheus text, read at scrape time) and `/alerts`, `/alerts/{row_id}`.

M6 case pages, from the `cases` table the consumer thread's case-pack hook fills: `/cases` (HTML
list, grouped by case_key), `/cases/{row_id}` (the case page: 5W+H, narrative, drivers and a
Cytoscape.js graph of the causal subgraph) and `/cases/{row_id}.json` (the stored pack); 404 for
an alert without a case. Pages are rendered by jinja2 with autoescape from
src/aml/explain/templates/; the pack is inlined as JSON (`script_json`) and read in the browser
with JSON.parse(element.textContent).

On the Kafka transport the thread also carries the `LatencyRecorder`, which writes
reports/latency/ and renders reports/latency.{md,json} at the end of the slice.

Shutdown: uvicorn turns SIGTERM / SIGINT into a lifespan shutdown, which stops the consumer and
joins it. `main` installs its own handler before uvicorn runs, so the signal uvicorn re-raises
after serving lands there; `main` then joins the thread again (also when a second Ctrl-C skipped
the lifespan shutdown) and returns the thread's exit code.
"""

from __future__ import annotations

import argparse
import asyncio
import bisect
import functools
import itertools
import json
import logging
import signal
import sys
import threading
from collections.abc import AsyncIterator, Callable, Iterator, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Any

import jinja2
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, generate_latest
from prometheus_client.core import (
    CounterMetricFamily,
    GaugeMetricFamily,
    HistogramMetricFamily,
    Metric,
)
from prometheus_client.utils import floatToGoString

from aml.explain.narrative import duration_words
from aml.features.spec import MINUTES_PER_DAY
from aml.serving.alerts import (
    count_alerts,
    count_cases,
    list_cases,
    read_alert,
    read_alerts,
    read_case,
    read_case_json,
)
from aml.serving.latency import LatencyRecorder
from aml.serving.scorer import (
    EXIT_PARITY,
    EXIT_REFUSED,
    Event,
    Observer,
    Runtime,
    Scored,
    Timing,
    add_case_packs_arg,
    open_runtime,
    startup_exit_code,
)
from aml.serving.settings import TRANSPORTS, ConfigError, Settings, add_cli_args, load_settings
from aml.serving.state import ALERTS_DB
from aml.streaming.codec import now_ns

log = logging.getLogger(__name__)

# aml_champion_seconds buckets (s); the latency report uses the raw per-event stamps instead.
HIST_BOUNDS_S = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
UP_STATUSES = ("ok", "done")

# M6 case pages
TEMPLATES_DIR = Path(__file__).resolve().parents[1] / "explain" / "templates"
CASE_TEMPLATE = "case.html.j2"
CASES_TEMPLATE = "cases.html.j2"
CYTOSCAPE_URL = "https://cdnjs.cloudflare.com/ajax/libs/cytoscape/3.30.2/cytoscape.min.js"
CASES_PAGE = 100  # /cases default limit (alerts per page)
_SCRIPT_ESCAPES = str.maketrans({"<": "\\u003c", ">": "\\u003e", "&": "\\u0026"})


class ChampionHistogram:
    """Observer: bucket counts of the champion path (consume -> alert write, minute flush
    excluded), written by the consumer thread and read by /metrics."""

    def __init__(self, bounds_s: Sequence[float] = HIST_BOUNDS_S) -> None:
        self.bounds_s = tuple(bounds_s)
        self._bounds_ns = [round(b * 1e9) for b in self.bounds_s]
        self.counts = [0] * (len(self.bounds_s) + 1)  # the last bucket is +Inf
        self.sum_ns = 0

    def observe(self, ev: Event, s: Scored, t: Timing) -> None:
        ns = t.t_done_ns - t.t_consume_ns - t.flush_ns
        self.counts[bisect.bisect_left(self._bounds_ns, ns)] += 1  # le: value <= bound
        self.sum_ns += ns

    def buckets(self) -> tuple[list[tuple[str, int]], float]:
        """Cumulative (le, count) pairs ending with +Inf, and the sum in seconds."""
        counts, total_ns = list(self.counts), self.sum_ns
        les = [floatToGoString(b) for b in self.bounds_s] + ["+Inf"]
        return list(zip(les, itertools.accumulate(counts), strict=True)), total_ns / 1e9


class ScorerThread(threading.Thread):
    """The consumer thread: open_runtime, run, close, all on it. `rt` is set once the source is
    assigned; `exit_code` when the thread ends (0 clean stop or end, 1 failure, 2 refused,
    3 order error, 4 parity failure at the end of the slice)."""

    def __init__(
        self,
        settings: Settings,
        *,
        observers: Sequence[Observer] = (),
        on_exit: Callable[[int], None] | None = None,
    ) -> None:
        super().__init__(name="aml-consumer", daemon=False)
        self.settings = settings
        self.stop_event = threading.Event()
        self.champion_hist = ChampionHistogram()
        self.observers = (self.champion_hist, *observers)
        self.on_exit = on_exit
        self.rt: Runtime | None = None
        self.running_since_ns: int | None = None
        self.exit_code: int | None = None
        self.error: str | None = None

    def request_stop(self) -> None:
        self.stop_event.set()

    def run(self) -> None:
        code = 1
        try:
            try:
                rt = open_runtime(self.settings, observers=self.observers)
            except Exception as e:
                code = startup_exit_code(e)
                self.error = str(e)
                if code == EXIT_REFUSED:
                    log.error("refused: %s", e)
                else:
                    log.exception("the scorer did not start")
                return
            self.running_since_ns = now_ns()
            self.rt = rt
            code = rt.close(rt.run(self.stop_event))
            if code == EXIT_PARITY:
                self.error = "the end-of-slice parity check failed"
            elif code:
                self.error = rt.scorer.error or f"exit code {code}"
        except Exception as e:
            log.exception("the consumer thread failed")
            self.error, code = repr(e), 1
            if self.rt is not None and self.rt.exit_code is None:
                self.rt.abort()  # release the source and sinks; no snapshot from a failed close
        finally:
            self.exit_code = code
            if self.on_exit is not None:
                try:
                    self.on_exit(code)
                except Exception:
                    log.exception("on_exit failed")


def health_view(t: Any, *, now: int, stall_after_s: float) -> tuple[int, dict[str, Any]]:
    """/health of a consumer thread: (HTTP status, body). Rules, in order:

    failed (503) exit code != 0; stopping (503) stop requested or exited cleanly; stalled (503)
    the thread is dead; starting (503) the source is not assigned yet; done (200) the slice
    ended; stalled (503) lag > 0 and nothing consumed for stall_after_s (since the assignment
    when nothing was consumed yet); else ok (200), also when idle with lag 0.
    """
    rt = t.rt
    p = None if rt is None else rt.progress
    lag = None
    if p is not None and p.high_watermark is not None:
        lag = max(0, p.high_watermark - p.expected_offset)
    consumed_age = None
    idle_s = None
    if p is not None:
        if p.last_consume_ns is not None:
            consumed_age = (now - p.last_consume_ns) / 1e9
        since = p.last_consume_ns if p.last_consume_ns is not None else t.running_since_ns
        idle_s = None if since is None else (now - since) / 1e9
    detail = None
    if t.exit_code not in (None, 0):
        code, status, detail = 503, "failed", t.error
    elif t.exit_code == 0 or t.stop_event.is_set():
        code, status = 503, "stopping"
    elif not t.is_alive():
        code, status, detail = 503, "stalled", "the consumer thread is not running"
    elif p is None:
        code, status, detail = 503, "starting", "restoring the state and assigning the source"
    elif p.finished:
        code, status = 200, "done"
    elif lag and idle_s is not None and idle_s > stall_after_s:
        code, status = 503, "stalled"
        detail = f"lag {lag}, nothing consumed for {idle_s:.0f} s"
    else:
        code, status = 200, "ok"
    return code, {
        "status": status,
        "detail": detail,
        "transport": t.settings.transport,
        "bundle_export_key": None if rt is None else rt.champion.export_key,
        "events": 0 if p is None else p.events,
        "alerts": 0 if p is None else p.alerts,
        "start_offset": None if p is None else p.start_offset,
        # The last offset consumed or covered by the restored state (-1: none): the replayer's
        # barrier waits for it.
        "last_offset": None if p is None else p.expected_offset - 1,
        "end_offset": None if p is None else p.end_offset,
        "lag": lag,
        "last_consumed_age_s": consumed_age,
        "log1p_ok": None if rt is None else rt.log1p_ok,
        "parity": None if rt is None else rt.parity,
        "exit_code": t.exit_code,
    }


class _Metrics:
    """Prometheus collector: the consumer thread's progress, read at scrape time."""

    def __init__(self, app: FastAPI, stall_after_s: float) -> None:
        self.app = app
        self.stall_after_s = stall_after_s

    def collect(self) -> Iterator[Metric]:
        t = getattr(self.app.state, "scorer", None)
        if t is None:
            return
        _, h = health_view(t, now=now_ns(), stall_after_s=self.stall_after_s)
        last = h["last_offset"]
        yield CounterMetricFamily("aml_events", "Events scored since the restore", h["events"])
        yield CounterMetricFamily(
            "aml_alerts", "Model alerts written since the restore", h["alerts"]
        )
        yield GaugeMetricFamily(
            "aml_last_offset",
            "Last offset consumed or restored (-1: none)",
            -1 if last is None else last,
        )
        yield GaugeMetricFamily(
            "aml_consumer_lag_events",
            "High watermark minus the next offset (NaN: unknown)",
            float("nan") if h["lag"] is None else h["lag"],
        )
        yield GaugeMetricFamily(
            "aml_consumer_up", "1 while /health is ok or done", int(h["status"] in UP_STATUSES)
        )
        buckets, total_s = t.champion_hist.buckets()
        yield HistogramMetricFamily(
            "aml_champion_seconds",
            "Consume -> alert write per event, minute flush excluded",
            buckets=buckets,
            sum_value=total_s,
        )


# --- case pages -----------------------------------------------------------------------------------


def script_json(obj: Any) -> str:
    """JSON for a <script type="application/json"> element: ASCII only, with <, > and & as \\u
    escapes, so the data can never close the element (`</script>`) or open a comment; JSON.parse
    of the element's textContent gives `obj` back."""
    return json.dumps(obj, allow_nan=False, separators=(",", ":")).translate(_SCRIPT_ESCAPES)


def _money(x: float | None) -> str:
    return "n/a" if x is None else f"{x:,.2f}"


def _prob(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.4f}"


def _signed(x: float | None) -> str:
    return "n/a" if x is None else f"{x:+.3f}"


def _when(minute: int) -> str:
    hh, mm = divmod(minute % MINUTES_PER_DAY, 60)
    return f"day {minute // MINUTES_PER_DAY + 1}, {hh:02d}:{mm:02d}"


@functools.cache
def templates() -> jinja2.Environment:
    """The case-page templates (autoescape on for every template, undefined names raise)."""
    env = jinja2.Environment(
        loader=jinja2.FileSystemLoader(TEMPLATES_DIR),
        autoescape=True,
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    env.filters.update(money=_money, prob=_prob, signed=_signed, when=_when)
    return env


def render_case(pack: Mapping[str, Any]) -> str:
    """The case page of one pack (aml.explain.casepack keys)."""
    why, how = pack["why"], pack["how"]
    top = max((abs(d["contribution"]) for d in why["drivers"]), default=0.0)
    bars = [
        {**d, "pct": 50.0 * abs(d["contribution"]) / top if top > 0 else 0.0}
        for d in why["drivers"]
    ]
    sg = how["subgraph"]
    return (
        templates()
        .get_template(CASE_TEMPLATE)
        .render(
            pack=pack,
            who=pack["who"],
            what=pack["what"],
            when=pack["when"],
            where=pack["where"],
            why=why,
            how=how,
            typ=why["typology"],
            bars=bars,
            nodes=sg["nodes"],
            edges=sg["edges"],
            n_history=sum(1 for e in sg["edges"] if e["kind"] != "alert"),
            window=duration_words(how["window_minutes"]),
            pack_json=script_json(pack),
            cytoscape_url=CYTOSCAPE_URL,
        )
    )


def render_case_list(
    groups: Sequence[Mapping[str, Any]], *, n_cases: int, limit: int, before_row: int | None
) -> str:
    """The /cases page: `list_cases` groups, with an "older" link when the page is full."""
    listed = [a for g in groups for a in g["alerts"]]
    older = listed[-1]["row_id"] if len(listed) == limit else None
    return (
        templates()
        .get_template(CASES_TEMPLATE)
        .render(
            groups=groups,
            n_cases=n_cases,
            n_listed=len(listed),
            limit=limit,
            before_row=before_row,
            older=older,
        )
    )


def create_app(settings: Settings, *, observers: Sequence[Observer] = ()) -> FastAPI:
    """The service. The lifespan starts the consumer thread, then stops and joins it. When the
    thread exits non-zero, or at the end of the slice with `exit_at_end`, it asks
    `app.state.server` (the uvicorn.Server, set by `main`) to exit."""
    obs: list[Observer] = []
    if settings.transport == "kafka":  # only Kafka messages carry the replayer's stamps
        obs.append(LatencyRecorder())
    obs += observers
    stall_after_s = settings.cfg.scorer.stall_after_s
    stop_timeout_s = settings.cfg.scorer.stop_timeout_s
    db = settings.runtime_dir / ALERTS_DB

    def on_exit(code: int) -> None:  # on the consumer thread
        server = getattr(app.state, "server", None)
        if server is not None and (code != 0 or settings.exit_at_end):
            server.should_exit = True

    @asynccontextmanager
    async def lifespan(app_: FastAPI) -> AsyncIterator[None]:
        t = ScorerThread(settings, observers=obs, on_exit=on_exit)
        app_.state.scorer = t
        t.start()
        try:
            yield
        finally:
            t.request_stop()
            await asyncio.to_thread(t.join, stop_timeout_s)
            if t.is_alive():
                log.error("the consumer thread did not stop within %g s", stop_timeout_s)

    app = FastAPI(title="AML streaming scorer (M5)", lifespan=lifespan)
    registry = CollectorRegistry()
    registry.register(_Metrics(app, stall_after_s))

    @app.get("/health")
    def health() -> JSONResponse:
        code, body = health_view(app.state.scorer, now=now_ns(), stall_after_s=stall_after_s)
        return JSONResponse(body, status_code=code)

    @app.get("/metrics")
    def metrics() -> Response:
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.get("/alerts")
    def alerts(
        limit: Annotated[int, Query(ge=1, le=1000)] = 100,
        before_rank: Annotated[int | None, Query()] = None,
    ) -> dict[str, Any]:
        """Stored alerts, newest rank first (only ranks < before_rank when given)."""
        return {"count": count_alerts(db), "items": read_alerts(db, limit, before_rank)}

    @app.get("/alerts/{row_id}")
    def alert(row_id: int) -> dict[str, Any]:
        rec = read_alert(db, row_id)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"no alert for row_id {row_id}")
        return rec

    @app.get("/cases", response_class=HTMLResponse)
    def cases(
        limit: Annotated[int, Query(ge=1, le=1000)] = CASES_PAGE,
        before_row: Annotated[int | None, Query()] = None,
    ) -> HTMLResponse:
        """Cased alerts, newest first, grouped by case_key (only older than before_row)."""
        groups = list_cases(db, limit, before_row)
        page = render_case_list(groups, n_cases=count_cases(db), limit=limit, before_row=before_row)
        return HTMLResponse(page)

    # Registered before /cases/{row_id}, whose pattern would also match "<row_id>.json".
    @app.get("/cases/{row_id}.json")
    def case_json(row_id: int) -> Response:
        text = read_case_json(db, row_id)
        if text is None:
            raise HTTPException(status_code=404, detail=f"no case for row_id {row_id}")
        return Response(text, media_type="application/json")

    @app.get("/cases/{row_id}", response_class=HTMLResponse)
    def case_page(row_id: int) -> HTMLResponse:
        pack = read_case(db, row_id)
        if pack is None:
            raise HTTPException(status_code=404, detail=f"no case for row_id {row_id}")
        return HTMLResponse(render_case(pack))

    return app


def main(argv: Sequence[str] | None = None) -> int:
    """python -m aml.serving.app [--transport kafka|inproc] [--host --port] [--no-case-packs]
    [paths]: serve until
    SIGTERM / SIGINT (or the end of the slice with --exit-at-end); returns the consumer thread's
    exit code."""
    p = argparse.ArgumentParser(prog="python -m aml.serving.app", description="M5 scorer service")
    p.add_argument("--transport", choices=TRANSPORTS, default="kafka")
    p.add_argument("--host", help="bind address (default 0.0.0.0)")
    p.add_argument("--port", type=int, help="port (serving.yaml scorer.port)")
    p.add_argument(
        "--exit-at-end",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="stop the service after the last event (default: on for inproc, off for kafka)",
    )
    add_case_packs_arg(p)
    add_cli_args(p)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings(args, transport=args.transport)
    except ConfigError as e:
        log.error("refused: %s", e)
        return EXIT_REFUSED
    app = create_app(settings)
    server = uvicorn.Server(
        uvicorn.Config(
            app,
            host=settings.host,
            port=settings.port,
            lifespan="on",
            access_log=False,
            timeout_graceful_shutdown=5,
            log_config=None,  # the basicConfig above: no timestamps in the logs
        )
    )
    app.state.server = server

    def after_uvicorn(signum: int, frame: Any) -> None:
        # uvicorn restores this handler after serving and re-raises the signal it caught: stop
        # the consumer (already stopping) and let main join it, instead of dying here.
        t = getattr(app.state, "scorer", None)
        if t is not None:
            t.request_stop()

    signal.signal(signal.SIGTERM, after_uvicorn)
    signal.signal(signal.SIGINT, after_uvicorn)
    served = True
    try:
        server.run()
    except SystemExit:  # uvicorn exits when it cannot start (e.g. the port is taken)
        log.error("uvicorn did not start")
        served = False
    t = getattr(app.state, "scorer", None)
    if t is None:
        return 1
    t.request_stop()
    t.join(settings.cfg.scorer.stop_timeout_s)
    if t.is_alive():
        log.error("the consumer thread did not stop in time")
        return 1
    code = 1 if t.exit_code is None else t.exit_code
    return code if code or served else 1


if __name__ == "__main__":
    sys.exit(main())
