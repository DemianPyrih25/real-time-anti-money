"""The scorer CLI as a real process (M5): an inproc run, a SIGKILL-like exit and a real SIGTERM
in the middle of the slice, each followed by a re-run that ends identical to an uninterrupted
run in a separate runtime dir."""

from __future__ import annotations

import sys

import polars as pl
import pytest

from aml.io import read_json
from aml.serving import bundle
from aml.serving.alerts import read_alerts
from aml.serving.state import ALERTS_DB, SNAPSHOT
from tests.fixtures.stream_child import KILL_CODE, last_json, run_module, scorer_args


def _rows(runtime_dir) -> list[dict]:
    return read_alerts(runtime_dir / ALERTS_DB, limit=None)[::-1]  # rank order


def _assert_parity_ok(s: dict) -> None:
    p = s["parity"]
    assert p["mismatches"] == dict.fromkeys(bundle.CHECKS, 0) and p["error"] is None
    assert p["alerts_ok"] is True and p["digest_ok"] is True


@pytest.fixture(scope="module")
def cli_run(serving_bundle, fixture_tag, tmp_path_factory):
    """An uninterrupted CLI run: the reference for the restarts."""
    rt_dir = tmp_path_factory.mktemp("cli_reference") / "rt"
    proc = run_module("aml.serving.scorer", *scorer_args(serving_bundle, rt_dir, fixture_tag))
    return proc, rt_dir


@pytest.fixture(scope="module")
def n(serving_bundle) -> int:
    return read_json(serving_bundle / bundle.METADATA_FILE)["rows"]["slice"]


def test_cli_inproc_run(cli_run, n):
    proc, rt_dir = cli_run
    assert proc.returncode == 0, proc.stderr[-4000:]
    s = last_json(proc.stdout)
    assert s["exit_code"] == 0 and s["result"] == "end" and s["origin"] == "bundle"
    assert s["start_offset"] == 0 and s["events"] == s["end_offset"] == s["next_offset"] == n
    _assert_parity_ok(s)
    assert s["db_alerts"] == s["parity"]["alerts"]["ref_count"] >= 1
    assert not (rt_dir / SNAPSHOT).exists()  # the end of the slice writes no snapshot


def test_kill_then_rerun(cli_run, serving_bundle, fixture_tag, tmp_path, n):
    reference = _rows(cli_run[1])
    k = n // 2
    args = scorer_args(serving_bundle, tmp_path / "rt", fixture_tag)
    proc = run_module("tests.fixtures.stream_child", "--kill-at", str(k), *args)
    assert proc.returncode == KILL_CODE, proc.stderr[-4000:]
    assert not (tmp_path / "rt" / SNAPSHOT).exists()
    assert _rows(tmp_path / "rt") == [r for r in reference if r["offset"] <= k]  # durable alerts

    proc = run_module("aml.serving.scorer", *args)
    assert proc.returncode == 0, proc.stderr[-4000:]
    s = last_json(proc.stdout)
    assert s["origin"] == "bundle" and s["start_offset"] == 0  # no snapshot: the bundle's
    _assert_parity_ok(s)
    assert _rows(tmp_path / "rt") == reference


@pytest.mark.skipif(sys.platform == "win32", reason="a real SIGTERM needs POSIX signals")
def test_sigterm_then_rerun(cli_run, serving_bundle, fixture_tag, tmp_path, n):
    reference = _rows(cli_run[1])
    minutes = pl.read_parquet(serving_bundle / bundle.SLICE, columns=["minute"])["minute"]
    k = n // 2
    want = minutes.to_list().index(minutes[k])  # the first event of k's minute
    args = scorer_args(serving_bundle, tmp_path / "rt", fixture_tag)
    proc = run_module("tests.fixtures.stream_child", "--term-at", str(k), *args)
    assert proc.returncode == 0, proc.stderr[-4000:]
    s = last_json(proc.stdout)
    assert s["result"] == "signal" and s["events"] == k + 1
    assert s["snapshot"]["next_offset"] == want
    side = read_json(tmp_path / "rt" / (SNAPSHOT + ".json"))
    assert side["next_offset"] == want and side["extra"]["kind"] == "runtime"

    proc = run_module("aml.serving.scorer", *args)
    assert proc.returncode == 0, proc.stderr[-4000:]
    s = last_json(proc.stdout)
    assert s["origin"] == "runtime" and s["start_offset"] == want
    _assert_parity_ok(s)
    assert _rows(tmp_path / "rt") == reference
