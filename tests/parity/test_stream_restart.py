"""Restart exactness in-process (M5): a SIGTERM-like stop (snapshot at the first pending
offset) or a SIGKILL-like abandon (no snapshot), then a restart, gives exactly the uninterrupted
run's outputs and alerts. Stale or foreign runtime state is refused, never replaced by the
bundle snapshot."""

from __future__ import annotations

import json
import random
import struct
import threading
from collections import Counter

import polars as pl
import pytest

from aml.io import read_json
from aml.serving import bundle
from aml.serving.alerts import read_alerts
from aml.serving.scorer import open_runtime, startup_exit_code
from aml.serving.state import (
    ALERTS_DB,
    KNOWN_FILES,
    RuntimeStateError,
    StateStore,
    clear_latency_dir,
)
from tests.fixtures.serving_bundle import FIXTURE_EXPORT_KEY, fixture_settings

STOPS = ("zero", "minute_first", "mid_minute", "last", "random0", "random1", "random2")


class StopAt:
    """Observer: request a stop after the event at `offset` (the loop stops before the next)."""

    def __init__(self, offset: int, stop: threading.Event) -> None:
        self.offset = offset
        self.stop = stop

    def observe(self, ev, s, t) -> None:
        if s.offset == self.offset:
            self.stop.set()


def _run(settings, stop_at: int | None = None):
    stop = threading.Event()
    rt = open_runtime(settings, observers=[] if stop_at is None else [StopAt(stop_at, stop)])
    return rt, rt.run(stop)


def _rows(runtime_dir) -> list[dict]:
    return read_alerts(runtime_dir / ALERTS_DB, limit=None)[::-1]  # rank order


def _assert_union(parities: list[dict], n: int) -> None:
    """Every incarnation bit-exact; together they cover offsets [0, n - 1] without a hole."""
    spans = sorted(tuple(p["covered"]) for p in parities if p["covered"])
    reach = -1
    for lo, hi in spans:
        assert lo <= reach + 1, spans
        reach = max(reach, hi)
    assert spans[0][0] == 0 and reach == n - 1, spans
    for p in parities:
        assert p["mismatches"] == dict.fromkeys(bundle.CHECKS, 0) and p["error"] is None


@pytest.fixture(scope="module")
def minutes(serving_bundle) -> list[int]:
    return pl.read_parquet(serving_bundle / bundle.SLICE, columns=["minute"])["minute"].to_list()


@pytest.fixture(scope="module")
def stops(minutes) -> dict[str, int]:
    n = len(minutes)
    counts = Counter(minutes)
    busiest = max(counts, key=lambda m: (counts[m], -m))
    first = minutes.index(busiest)
    rnd = random.Random(0).sample(range(n), 3)
    return {
        "zero": 0,
        "minute_first": first,
        "mid_minute": first + counts[busiest] // 2,
        "last": n - 1,
        **{f"random{i}": k for i, k in enumerate(rnd)},
    }


@pytest.fixture(scope="module")
def reference(serving_bundle, fixture_tag, tmp_path_factory) -> list[dict]:
    """The alerts table of an uninterrupted run."""
    rt_dir = tmp_path_factory.mktemp("restart_reference") / "rt"
    rt, result = _run(fixture_settings(serving_bundle, rt_dir, fixture_tag))
    assert result == "end" and rt.close(result) == 0
    rows = _rows(rt_dir)
    assert rows and rows == sorted(rows, key=lambda r: r["rank"])
    return rows


@pytest.mark.parametrize("where", STOPS)
def test_sigterm_like_stop_then_restart(
    where, stops, minutes, reference, serving_bundle, fixture_tag, tmp_path
):
    k, n = stops[where], len(minutes)
    settings = fixture_settings(serving_bundle, tmp_path / "rt", fixture_tag)
    rt1, result = _run(settings, stop_at=k)
    assert result == "signal" and rt1.close(result) == 0
    # The snapshot resumes at the first pending event: the first event of k's minute, or n
    # after the final flush.
    want = n if k == n - 1 else minutes.index(minutes[k])
    assert rt1.snapshot["next_offset"] == want
    side = read_json(rt1.store.snapshot_path.with_name("scorer.snap.json"))
    meta = read_json(serving_bundle / bundle.METADATA_FILE)
    assert side["next_offset"] == want
    assert side["next_rank"] - side["next_offset"] == meta["next_rank"]
    assert side["extra"]["kind"] == "runtime"
    assert side["extra"]["export_key"] == FIXTURE_EXPORT_KEY

    rt2, result = _run(settings)
    assert rt2.restored.origin == "runtime" and rt2.start_offset == want
    assert result == "end" and rt2.close(result) == 0
    assert rt2.parity["alerts_ok"] is True and rt2.parity["digest_ok"] is True
    _assert_union([rt1.parity, rt2.parity], n)
    assert _rows(tmp_path / "rt") == reference


def test_sigkill_like_abandon_then_restart(
    minutes, reference, serving_bundle, fixture_tag, tmp_path
):
    n = len(minutes)
    k1, k2 = n // 3, (2 * n) // 3
    settings = fixture_settings(serving_bundle, tmp_path / "rt", fixture_tag)
    rt1, result = _run(settings, stop_at=k1)
    assert result == "signal" and rt1.close(result) == 0
    a = rt1.snapshot["next_offset"]
    snap = rt1.store.snapshot_path.read_bytes()

    rt2, result = _run(settings, stop_at=k2)  # killed at k2: no close, no snapshot
    assert rt2.restored.origin == "runtime" and rt2.start_offset == a and result == "signal"
    rt2.abort()
    assert rt2.store.snapshot_path.read_bytes() == snap

    rt3, result = _run(settings)  # replays (a, k2] again from the older snapshot
    assert rt3.start_offset == a and result == "end" and rt3.close(result) == 0
    assert rt3.parity["alerts_ok"] is True and rt3.parity["digest_ok"] is True
    _assert_union([rt1.parity, rt3.parity], n)
    assert _rows(tmp_path / "rt") == reference  # no lost and no duplicate alert


def test_no_snapshot_before_any_event(serving_bundle, fixture_tag, tmp_path):
    rt = open_runtime(fixture_settings(serving_bundle, tmp_path / "rt", fixture_tag))
    stop = threading.Event()
    stop.set()
    result = rt.run(stop)
    assert result == "signal" and rt.close(result) == 0
    assert rt.snapshot is None and not rt.store.snapshot_path.exists()


class _Undelivered:
    """An alert sink whose last message never left the producer queue."""

    failed = 0

    def write(self, rec) -> bool:
        return True

    def flush(self, timeout_s: float) -> int:
        return 1

    def close(self) -> None:
        pass


def test_no_snapshot_when_alerts_are_undelivered(minutes, serving_bundle, fixture_tag, tmp_path):
    settings = fixture_settings(serving_bundle, tmp_path / "rt", fixture_tag)
    stop = threading.Event()
    rt = open_runtime(settings, observers=[StopAt(len(minutes) // 2, stop)])
    rt.sinks.append(_Undelivered())
    result = rt.run(stop)
    assert result == "signal" and rt.close(result) == 0
    assert rt.snapshot is None and not rt.store.snapshot_path.exists()


def _rewrite_extra(data: bytes, **changes) -> bytes:
    """The snapshot with `extra` fields changed (payload and state digest untouched)."""
    (hlen,) = struct.unpack("<I", data[8:12])
    header = json.loads(data[12 : 12 + hlen])
    header["extra"] = {**header["extra"], **changes}
    text = json.dumps(header, sort_keys=True).encode()
    return data[:8] + struct.pack("<I", len(text)) + text + data[12 + hlen :]


def test_foreign_or_corrupt_runtime_snapshot_is_refused(
    minutes, serving_bundle, fixture_tag, tmp_path
):
    settings = fixture_settings(serving_bundle, tmp_path / "rt", fixture_tag)
    rt, result = _run(settings, stop_at=len(minutes) // 2)
    assert rt.close(result) == 0
    path = rt.store.snapshot_path
    data = path.read_bytes()
    for changes, match in (
        ({"export_key": "export-other"}, "export_key"),
        ({"booster_sha256": "0" * 64}, "booster_sha256"),
        ({"kind": "boundary"}, "kind"),
    ):
        path.write_bytes(_rewrite_extra(data, **changes))
        with pytest.raises(RuntimeStateError, match=match) as err:
            open_runtime(settings)
        assert startup_exit_code(err.value) == 2
    bad = bytearray(data)
    bad[-1] ^= 0xFF  # one payload byte
    path.write_bytes(bytes(bad))
    with pytest.raises(RuntimeStateError, match="does not restore"):
        open_runtime(settings)
    assert path.read_bytes() == bytes(bad)  # refused, not replaced or skipped

    path.write_bytes(data)
    a = rt.snapshot["next_offset"]
    if a >= 2:  # a run that ends before the snapshot's position
        short = fixture_settings(serving_bundle, tmp_path / "rt", fixture_tag, max_events=a - 1)
        with pytest.raises(RuntimeStateError, match="outside"):
            open_runtime(short)
    again = open_runtime(settings)
    assert again.restored.origin == "runtime" and again.start_offset == a
    again.abort()


def test_reset_deletes_only_the_known_state(serving_bundle, tmp_path):
    rt_dir = tmp_path / "rt"
    for rel in (*KNOWN_FILES, "snapshots/.scorer.snap.tmp-42", "keep.txt", "snapshots/keep"):
        (rt_dir / rel).parent.mkdir(parents=True, exist_ok=True)
        (rt_dir / rel).write_bytes(b"x")
    StateStore(rt_dir).reset()
    left = sorted(p.relative_to(rt_dir).as_posix() for p in rt_dir.rglob("*") if p.is_file())
    assert left == ["keep.txt", "snapshots/keep"]
    StateStore(tmp_path / "fresh").reset()  # a missing dir is created
    assert (tmp_path / "fresh").is_dir()
    with pytest.raises(RuntimeStateError, match="serving bundle"):
        StateStore(serving_bundle).reset()
    with pytest.raises(RuntimeStateError, match="root"):
        StateStore(tmp_path.anchor).reset()
    lat = tmp_path / "reports" / "latency"
    lat.mkdir(parents=True)
    (lat / "events.parquet").write_bytes(b"x")
    (lat / "sub").mkdir()
    assert clear_latency_dir(tmp_path / "reports") == 1 and [p.name for p in lat.iterdir()] == [
        "sub"
    ]
