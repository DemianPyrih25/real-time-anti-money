"""The M5 HTTP app: uvicorn in a thread on a pre-bound 127.0.0.1 socket, queried with urllib (no
httpx), around the inproc consumer thread on the fixture bundle; plus the /health rules on fake
consumer threads."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request
from types import SimpleNamespace

import pytest
import uvicorn

from aml.io import read_json
from aml.serving import bundle
from aml.serving.alerts import count_alerts, read_alerts
from aml.serving.app import create_app, health_view
from aml.serving.scorer import Progress
from aml.serving.state import ALERTS_DB, SNAPSHOT, StateStore
from tests.fixtures.serving_bundle import fixture_settings

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never a proxy
WAIT_S = 120.0
S = 1_000_000_000  # ns


def get(base: str, path: str) -> tuple[int, bytes]:
    try:
        with OPENER.open(base + path, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def wait_for(check, what: str, timeout: float = WAIT_S):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        got = check()
        if got:
            return got
        time.sleep(0.05)
    raise AssertionError(f"timed out waiting for {what}")


class Served:
    """uvicorn.Server(Config(app)) in a daemon thread on a pre-bound free port. `attach` sets
    app.state.server as `main` does, so the consumer thread's exit can stop the server."""

    def __init__(self, app, *, attach: bool) -> None:
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(("127.0.0.1", 0))
        self.base = f"http://127.0.0.1:{self.sock.getsockname()[1]}"
        config = uvicorn.Config(app, lifespan="on", log_config=None, access_log=False)
        self.server = uvicorn.Server(config)
        if attach:
            app.state.server = self.server
        self.thread = threading.Thread(
            target=self.server.run, kwargs={"sockets": [self.sock]}, daemon=True
        )

    def __enter__(self) -> Served:
        self.thread.start()
        wait_for(lambda: self.server.started or not self.thread.is_alive(), "uvicorn to start")
        return self

    def __exit__(self, *exc) -> None:
        self.server.should_exit = True  # lifespan shutdown: stop and join the consumer
        self.thread.join(WAIT_S)
        self.sock.close()

    def health(self) -> tuple[int, dict]:
        code, raw = get(self.base, "/health")
        return code, json.loads(raw)

    def wait_status(self, status: str) -> tuple[int, dict]:
        def check():
            code, body = self.health()
            return (code, body) if body["status"] == status else None

        return wait_for(check, f"/health {status}")


def metric_values(text: str) -> dict[str, float]:
    out = {}
    for line in text.splitlines():
        if line and not line.startswith("#"):
            key, _, value = line.rpartition(" ")
            out[key] = float(value)
    return out


def test_app_serves_the_stream_and_snapshots_on_the_consumer_thread(
    serving_bundle, fixture_tag, tmp_path, monkeypatch
):
    saved_on: list[int] = []
    real_save = StateStore.save

    def spy(self, eng, **kw):
        saved_on.append(threading.get_ident())
        return real_save(self, eng, **kw)

    monkeypatch.setattr(StateStore, "save", spy)
    settings = fixture_settings(
        serving_bundle, tmp_path / "runtime", fixture_tag, exit_at_end=False
    )
    n = read_json(serving_bundle / bundle.METADATA_FILE)["rows"]["slice"]
    db = settings.runtime_dir / ALERTS_DB
    app = create_app(settings)
    with Served(app, attach=True) as srv:
        code, h = srv.wait_status("done")
        assert code == 200 and h["exit_code"] is None and h["transport"] == "inproc"
        assert h["events"] == n and h["last_offset"] == n - 1 and h["end_offset"] == n
        assert h["lag"] == 0 and h["log1p_ok"] is True and h["bundle_export_key"] == "fixture"
        par = h["parity"]
        assert par["mismatches"] == dict.fromkeys(bundle.CHECKS, 0) and par["checked"] == n
        assert par["alerts_ok"] is True and par["digest_ok"] is True

        code, raw = get(srv.base, "/alerts?limit=1000")
        doc = json.loads(raw)
        items = doc["items"]
        assert code == 200 and doc["count"] == count_alerts(db) == h["alerts"] >= 1
        assert len(items) == min(doc["count"], 1000)
        assert items == read_alerts(db, limit=1000)  # newest rank first, bits intact
        ranks = [a["rank"] for a in items]
        assert ranks == sorted(ranks, reverse=True)
        assert json.loads(get(srv.base, "/alerts?limit=1")[1])["items"] == items[:1]
        older = json.loads(get(srv.base, f"/alerts?before_rank={ranks[0]}")[1])["items"]
        assert older == items[1:101]
        assert get(srv.base, "/alerts?limit=0")[0] == 422
        assert get(srv.base, "/alerts?limit=1001")[0] == 422

        code, raw = get(srv.base, f"/alerts/{items[-1]['row_id']}")
        assert code == 200 and json.loads(raw) == items[-1]
        assert get(srv.base, "/alerts/-1")[0] == 404

        code, raw = get(srv.base, "/metrics")
        m = metric_values(raw.decode())
        assert code == 200 and m["aml_events_total"] == n and m["aml_alerts_total"] == h["alerts"]
        assert m["aml_last_offset"] == n - 1 and m["aml_consumer_lag_events"] == 0
        assert m["aml_consumer_up"] == 1
        assert m["aml_champion_seconds_count"] == n == m['aml_champion_seconds_bucket{le="+Inf"}']
        assert m['aml_champion_seconds_bucket{le="0.0005"}'] <= n
        t = app.state.scorer
        assert t.is_alive()
    # should_exit -> the lifespan stops and joins the consumer, which snapshots on its own thread
    assert not srv.thread.is_alive() and not t.is_alive() and t.exit_code == 0
    assert saved_on == [t.ident] and t.ident != threading.main_thread().ident
    assert (settings.runtime_dir / SNAPSHOT).is_file()
    assert t.rt.snapshot["next_offset"] == n and t.rt.result == "signal"


def test_a_missing_bundle_fails_health_with_exit_code_2(tmp_path):
    settings = fixture_settings(tmp_path / "no_bundle", tmp_path / "runtime", "headline")
    app = create_app(settings)
    with Served(app, attach=False) as srv:  # no server to stop: /health stays up
        code, h = srv.wait_status("failed")
        assert code == 503 and h["exit_code"] == 2 and "metadata.json" in h["detail"]
        assert h["events"] == 0 and h["last_offset"] is None and h["parity"] is None
        assert app.state.scorer.exit_code == 2
        assert metric_values(get(srv.base, "/metrics")[1].decode())["aml_consumer_up"] == 0
    # As `main` wires it, the failed consumer stops the server by itself.
    app2 = create_app(settings)
    srv2 = Served(app2, attach=True)
    srv2.thread.start()
    srv2.thread.join(WAIT_S)
    assert not srv2.thread.is_alive() and app2.state.scorer.exit_code == 2


# --- /health rules --------------------------------------------------------------------------------

NOW = 1_000 * S


def fake_thread(rt=None, *, exit_code=None, alive=True, stop=False, since=NOW - 100 * S):
    ev = threading.Event()
    if stop:
        ev.set()
    return SimpleNamespace(
        settings=SimpleNamespace(transport="kafka"),
        rt=rt,
        exit_code=exit_code,
        error="boom" if exit_code else None,
        stop_event=ev,
        is_alive=lambda: alive,
        running_since_ns=since if rt is not None else None,
    )


def fake_rt(**progress):
    return SimpleNamespace(
        progress=Progress(start_offset=0, end_offset=100, **progress),
        champion=SimpleNamespace(export_key="fixture"),
        log1p_ok=True,
        parity=None,
    )


CONSUMING = {"events": 5, "last_offset": 4, "high_watermark": 50}
DONE = {"events": 100, "last_offset": 99, "high_watermark": 100, "finished": True}


@pytest.mark.parametrize(
    "thread, code, status",
    [
        (fake_thread(), 503, "starting"),
        (fake_thread(exit_code=2), 503, "failed"),
        (fake_thread(fake_rt(**CONSUMING), exit_code=3), 503, "failed"),
        (fake_thread(fake_rt(**CONSUMING), stop=True), 503, "stopping"),
        (fake_thread(fake_rt(finished=True), exit_code=0, alive=False), 503, "stopping"),
        (fake_thread(fake_rt(**CONSUMING), alive=False), 503, "stalled"),
        (fake_thread(fake_rt(high_watermark=0)), 200, "ok"),  # idle before the first message
        (fake_thread(fake_rt()), 200, "ok"),  # lag not known yet
        (fake_thread(fake_rt(**CONSUMING, last_consume_ns=NOW - 1 * S)), 200, "ok"),
        (fake_thread(fake_rt(**CONSUMING, last_consume_ns=NOW - 31 * S)), 503, "stalled"),
        (fake_thread(fake_rt(high_watermark=50)), 503, "stalled"),  # nothing consumed since 100 s
        (fake_thread(fake_rt(high_watermark=50), since=NOW - 1 * S), 200, "ok"),
        (fake_thread(fake_rt(**DONE)), 200, "done"),
    ],
)
def test_health_rules(thread, code, status):
    got, body = health_view(thread, now=NOW, stall_after_s=30.0)
    assert (got, body["status"]) == (code, status), body


def test_health_body_while_consuming():
    rt = fake_rt(**CONSUMING, alerts=2, last_consume_ns=NOW - 31 * S)
    code, body = health_view(fake_thread(rt), now=NOW, stall_after_s=30.0)
    assert code == 503 and body["status"] == "stalled" and "lag 45" in body["detail"]
    assert body["events"] == 5 and body["alerts"] == 2 and body["last_offset"] == 4
    assert body["lag"] == 45 and body["last_consumed_age_s"] == 31.0
    assert body["end_offset"] == 100 and body["bundle_export_key"] == "fixture"
    assert body["exit_code"] is None and body["log1p_ok"] is True
