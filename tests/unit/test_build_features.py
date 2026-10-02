"""The offline feature build (M2 spec §8.1): parts, snapshots, restart check, bench and verify.

Most tests run the real engine end to end on the synthetic fixture (M2 spec §9.2 row E): one
shared full build (`real_full`) for the read-only checks, plus the bench, verify (the real DuckDB
oracle) and read-batch runs. `FakeEngine`, a small stand-in with the public Engine API (as-of
lifetime counts per account), is kept where a controllable engine is the point of the test: the
driver's minute flushes checked against closed-form counts, a lossy restore that the restart
check must catch, and the stale-output and refusal paths.
"""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import struct
import zlib
from pathlib import Path

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from aml.data.schemas import validate_features
from aml.features import build
from aml.features.snapshot import MAGIC, SIDECAR_SUFFIX
from aml.features.spec import (
    E_DST,
    E_FLAGS,
    E_FMT,
    E_HOUR,
    E_L,
    E_MINUTE,
    E_PCUR,
    E_RANK,
    E_RCUR,
    E_SRC,
    E_USD_C,
    ENGINE_VERSION,
    F_SELF,
    PARTS_DIR,
    SEVERITY_COLUMNS,
    SNAPSHOTS_DIR,
    TRUNC_COLUMNS,
    EngineError,
    EngineSpec,
    FlushStats,
    LateEventError,
    RankGapError,
    SnapshotError,
    part_name,
)
from aml.features.tx_features import fit_vocab
from aml.io import read_json, write_json_atomic
from aml.paths import DataPaths
from tests.conftest import load_yaml

# --- a stand-in engine with the public Engine API ----------------------------------------------


class FakeEngine:
    """As-of stand-in: per-account lifetime counts of *applied* events (minutes < clock).

    Same protocol as the real engine (pending minute, advance, rank/late checks, snapshot framing
    `MAGIC | u32 | header JSON | payload`), so every driver path is exercised. Row: a few TX
    columns, u_out_cnt_1d / v_in_cnt_1d = earlier-minute counts, u_out_mean_1d NaN on no
    history, fan_in_velocity = v's earlier in-count, inflow_c = u's earlier non-self-loop cents.
    """

    def __init__(self, spec: EngineSpec) -> None:
        self.spec = spec
        self.clock: int | None = None
        self.next_rank = 0
        self.out_cnt: dict[int, int] = {}
        self.in_cnt: dict[int, int] = {}
        self.in_c: dict[int, int] = {}
        self.pending: list[tuple] = []
        self.codes = {c: {v: i for i, v in enumerate(vals)} for c, vals in spec.vocab.items()}
        fi = spec.feature_index
        self.ix = {n: fi[n] for n in fi}

    @classmethod
    def create(cls, spec: EngineSpec) -> FakeEngine:
        return cls(spec)

    def prepare(self, rank, row_id, minute, src, dst, usd, paid, fmt, pcur, rcur, fb, tb):
        return (
            rank,
            row_id,
            minute,
            src,
            dst,
            round(usd * 100),
            math.log1p(usd),
            usd,
            round(paid * 100),
            self.codes["payment_format"].get(fmt, -1),
            self.codes["payment_currency"].get(pcur, -1),
            self.codes["receiving_currency"].get(rcur, -1),
            F_SELF if src == dst else 0,
            (minute % 1440) // 60,
        )

    def advance(self, minute: int) -> FlushStats:
        if self.clock is not None and minute < self.clock:
            raise LateEventError(minute)
        if self.clock is not None and minute == self.clock:
            return FlushStats(0, 0, 0.0)
        n = len(self.pending)
        for ev in self.pending:
            u, v = ev[E_SRC], ev[E_DST]
            self.out_cnt[u] = self.out_cnt.get(u, 0) + 1
            self.in_cnt[v] = self.in_cnt.get(v, 0) + 1
            if u != v:
                self.in_c[v] = self.in_c.get(v, 0) + ev[E_USD_C]
        self.pending.clear()
        self.clock = minute
        return FlushStats(n, 0, 0.0)

    def score(self, ev: tuple) -> tuple:
        ix, row = self.ix, [0.0] * self.spec.row_len
        u, v = ev[E_SRC], ev[E_DST]
        cnt = self.out_cnt.get(u, 0)
        row[ix["log_amount_usd"]] = ev[E_L]
        row[ix["payment_currency"]] = ev[E_PCUR]
        row[ix["receiving_currency"]] = ev[E_RCUR]
        row[ix["payment_format"]] = ev[E_FMT]
        row[ix["self_loop"]] = ev[E_FLAGS] & F_SELF
        row[ix["hour_of_day"]] = ev[E_HOUR]
        row[ix["u_out_cnt_1d"]] = cnt
        row[ix["v_in_cnt_1d"]] = self.in_cnt.get(v, 0)
        row[ix["u_out_mean_1d"]] = math.nan if cnt == 0 else float(cnt % 5)
        row[self.spec.i_sev] = float(self.in_cnt.get(v, 0))
        row[self.spec.i_inflow] = self.in_c.get(u, 0)
        return tuple(row)

    def process(self, ev: tuple) -> tuple:
        m = ev[E_MINUTE]
        if self.clock is None or m > self.clock:
            self.advance(m)
        elif m < self.clock:
            raise LateEventError(m)
        if ev[E_RANK] != self.next_rank:
            raise RankGapError(ev[E_RANK])
        row = self.score(ev)
        self.pending.append(ev)
        self.next_rank += 1
        return row

    @property
    def pending_count(self) -> int:
        return len(self.pending)

    def _state(self) -> dict:
        first = self.pending[0][E_RANK] if self.pending else self.next_rank
        return {
            "clock": self.clock,
            "next_rank": first,
            "out": sorted(self.out_cnt.items()),
            "in": sorted(self.in_cnt.items()),
            "in_c": sorted(self.in_c.items()),
        }

    def state_digest(self) -> str:
        return hashlib.sha256(json.dumps(self._state(), sort_keys=True).encode()).hexdigest()

    def state_nbytes(self) -> dict[str, int]:
        acc = 24 * (len(self.out_cnt) + len(self.in_cnt) + len(self.in_c))
        return {
            "accounts": acc,
            "pairs": 8 * len(self.out_cnt),
            "total": acc + 8 * len(self.out_cnt),
        }

    def snapshot(self, dst=None, *, next_offset=None, extra=None, compress=True):
        state = self._state()
        payload = json.dumps(state, sort_keys=True).encode()
        if compress:
            payload = zlib.compress(payload, 1)
        header = {
            "format": 1,
            "engine_version": ENGINE_VERSION,
            "spec_hash": self.spec.spec_hash(),
            "clock": state["clock"],
            "next_rank": state["next_rank"],
            "next_offset": next_offset,
            "n_pairs": len(self.out_cnt),
            "live_start": 0,
            "compressed": compress,
            "state_digest": self.state_digest(),
            "extra": extra or {},
        }
        hb = json.dumps(header, sort_keys=True).encode()
        data = MAGIC + struct.pack("<I", len(hb)) + hb + payload
        if dst is None:
            return data
        dst = Path(dst)
        dst.write_bytes(data)
        side = {**header, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}
        write_json_atomic(side, dst.with_name(dst.name + SIDECAR_SUFFIX))
        return header

    @classmethod
    def restore(cls, src, spec: EngineSpec):
        data = src if isinstance(src, bytes | bytearray) else Path(src).read_bytes()
        if data[: len(MAGIC)] != MAGIC:
            raise SnapshotError("magic")
        (n,) = struct.unpack("<I", data[len(MAGIC) : len(MAGIC) + 4])
        header = json.loads(data[len(MAGIC) + 4 : len(MAGIC) + 4 + n])
        if header["spec_hash"] != spec.spec_hash():
            raise SnapshotError("spec_hash")
        payload = data[len(MAGIC) + 4 + n :]
        state = json.loads(zlib.decompress(payload) if header["compressed"] else payload)
        eng = cls(spec)
        eng.clock, eng.next_rank = state["clock"], state["next_rank"]
        eng.out_cnt = dict(map(tuple, state["out"]))
        eng.in_cnt = dict(map(tuple, state["in"]))
        eng.in_c = dict(map(tuple, state["in_c"]))
        if eng.state_digest() != header["state_digest"]:
            raise SnapshotError("digest")
        eng._after_restore()
        return eng, header

    def _after_restore(self) -> None:
        pass


class LossyFakeEngine(FakeEngine):
    """Restores every out-count one short (after the digest check): the restart check must
    catch the difference in rows and in the state digest."""

    def _after_restore(self) -> None:
        self.out_cnt = {u: n - 1 for u, n in self.out_cnt.items() if n > 1}


# --- fixtures ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def paths(prepared: DataPaths, tmp_path_factory: pytest.TempPathFactory) -> DataPaths:
    """A private copy of the prepared fixture tables (reports written here stay here)."""
    p = DataPaths(tmp_path_factory.mktemp("feat_volume"))
    for name in ("transactions", "accounts", "fx_rates", "labels"):
        dst = getattr(p, name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(getattr(prepared, name), dst)
    return p


@pytest.fixture(scope="module")
def cfgs(data_cfg: dict, rules_cfg: dict) -> dict[str, dict]:
    return {
        "data": copy.deepcopy(data_cfg),
        "rules": copy.deepcopy(rules_cfg),
        "features": load_yaml("features.yaml"),
        "serving": load_yaml("serving.yaml"),
    }


@pytest.fixture(scope="module")
def tx(paths: DataPaths) -> pl.DataFrame:
    return pl.read_parquet(paths.transactions)


FAKE_INPUTS = {"hubs": [], "hub_cap": 0}


@pytest.fixture
def fake(monkeypatch: pytest.MonkeyPatch) -> type[FakeEngine]:
    monkeypatch.setattr(build, "Engine", FakeEngine)
    return FakeEngine


@pytest.fixture(scope="module")
def fake_full(paths, cfgs, tmp_path_factory) -> dict:
    """One fake-engine full build shared by the read-only driver tests."""
    mp = pytest.MonkeyPatch()
    mp.setattr(build, "Engine", FakeEngine)
    try:
        out = tmp_path_factory.mktemp("fake_full") / "features-x"
        commits: list[int] = []
        summary = build.run_build_features(
            paths, out, cfgs, on_checkpoint=lambda: commits.append(1), **FAKE_INPUTS
        )
    finally:
        mp.undo()
    return {"out": out, "summary": summary, "commits": len(commits)}


@pytest.fixture(scope="module")
def real_full(paths, cfgs, tmp_path_factory) -> dict:
    """One real-engine full build shared by the read-only tests."""
    out = tmp_path_factory.mktemp("real_full") / "features-r"
    commits: list[int] = []
    summary = build.run_build_features(paths, out, cfgs, on_checkpoint=lambda: commits.append(1))
    return {"out": out, "summary": summary, "commits": len(commits)}


def _expected_counts(tx: pl.DataFrame) -> pl.DataFrame:
    """Earlier-minute (as-of) lifetime counts of u's out-events and v's in-events, per row."""
    return tx.sort("rank").select(
        "row_id",
        (pl.col("minute").rank("min").over("src") - 1).cast(pl.Float32).alias("u_out_cnt_1d"),
        (pl.col("minute").rank("min").over("dst") - 1).cast(pl.Float32).alias("v_in_cnt_1d"),
    )


def _table(out: Path) -> pl.DataFrame:
    return pl.read_parquet([str(p) for p in sorted((out / PARTS_DIR).glob("part-d*.parquet"))])


# --- driver (fake engine) ----------------------------------------------------------------------


def test_full_build_writes_one_part_per_day_in_rank_order(fake_full, tx, cfgs, paths):
    out, summary = fake_full["out"], fake_full["summary"]
    n_days = int(tx["day"].max())
    parts = sorted((out / PARTS_DIR).glob("part-d*.parquet"))
    assert [p.name for p in parts] == [part_name(d) for d in range(1, n_days + 1)]
    assert not list((out / PARTS_DIR).glob(".*"))  # no temp files left
    spec = build.load_spec(out)
    schema = build.arrow_schema(spec)
    for p in parts:
        assert pq.read_schema(p).remove_metadata() == schema
    table = _table(out)
    assert table.schema == pl.Schema(spec.table_schema())
    assert table.height == tx.height == summary["rows"]
    assert table["rank"].to_list() == list(range(tx.height))
    keys = tx.sort("rank").select("row_id", "rank", "day", "split")
    assert table.select(keys.columns).equals(keys)
    for d, p in enumerate(parts, start=1):
        days = pl.read_parquet(p, columns=["day"])["day"].unique().to_list()
        assert days == [d] or days == []
    validate_features(table, spec)


def test_rows_are_the_engine_rows_under_the_as_of_rule(fake_full, tx):
    """The driver's own minute flushes must keep same-minute peers invisible (the fake counts
    only applied events): earlier-minute counts, never same-minute ones."""
    table = _table(fake_full["out"])
    want = _expected_counts(tx)
    got = table.select("row_id", "u_out_cnt_1d", "v_in_cnt_1d")
    assert got.sort("row_id").equals(want.sort("row_id"))
    mean = table["u_out_mean_1d"]
    assert mean.is_nan().sum() == (table["u_out_cnt_1d"] == 0).sum() > 0
    assert (table[SEVERITY_COLUMNS[0]] == table["v_in_cnt_1d"].cast(pl.Float64)).all()
    assert table.select(TRUNC_COLUMNS).sum().row(0) == (0, 0, 0)
    log_amt = tx.sort("rank")["amount_usd"].log1p().cast(pl.Float32)
    assert table["log_amount_usd"].to_numpy().view(np.uint32).tolist() == (
        log_amt.to_numpy().view(np.uint32).tolist()
    )


def test_boundary_snapshots_and_progress(real_full, tx, cfgs):
    from aml.features.engine import Engine

    out, summary = real_full["out"], real_full["summary"]
    bounds = build.boundary_minutes(cfgs["data"], cfgs["features"])
    assert bounds == {"val_early": 8640, "test": 11520}
    for split, b in bounds.items():
        snap = out / SNAPSHOTS_DIR / build.snapshot_file(split)
        assert snap.exists() and snap.with_name(snap.name + SIDECAR_SUFFIX).exists()
        first = int(tx.filter(pl.col("minute") >= b)["rank"].min())
        info = summary["snapshots"][split]
        assert info["next_rank"] == first and info["next_offset"] == 0 and info["clock"] == b
        assert info["bytes"] == snap.stat().st_size
        assert info["sha256"] == hashlib.sha256(snap.read_bytes()).hexdigest()
        eng, header = Engine.restore(snap, build.load_spec(out))
        assert header["next_rank"] == first and eng.state_digest() == info["state_digest"]
        assert eng.clock == b and header["next_offset"] == 0
    lines = (out / "progress.jsonl").read_text(encoding="utf-8").splitlines()
    recs = [json.loads(x) for x in lines]
    n_days = int(tx["day"].max())
    assert [r["day"] for r in recs] == list(range(1, n_days + 1))
    assert [r["boundary_minute"] for r in recs[:-1]] == [d * 1440 for d in range(1, n_days)]
    assert recs[-1]["boundary_minute"] == int(tx["minute"].max()) + 1
    assert recs[-1]["final_state_digest"] == summary["final_state_digest"]
    per_day = tx.group_by("day").len().sort("day")["len"].to_list()
    assert [r["events"] for r in recs] == per_day
    for r in recs:
        assert r["memory"]["restored_mb"] > 0 and r["memory"]["n_pairs"] >= 0
        assert r["flush_ms"]["n"] >= 1 and r["us_per_event"]["engine"] > 0
    # the boundary at the start of day 7 (9) closes day 6 (8)
    assert (
        recs[5]["snapshot"] == "val_boundary.snap" and recs[7]["snapshot"] == "test_boundary.snap"
    )
    assert sum("snapshot" in r for r in recs) == 2


def test_summary_fields_and_restart_check(real_full, tx, paths):
    out, s = real_full["out"], real_full["summary"]
    assert read_json(out / "summary.json") == json.loads(json.dumps(s))
    assert s["mode"] == "full" and s["engine_version"] == ENGINE_VERSION
    assert s["spec_hash"] == build.load_spec(out).spec_hash()
    split_rows = tx.group_by("split").len()
    assert s["rows_per_split"] == dict(split_rows.iter_rows())
    assert set(s["trunc_share"]["per_split"]) == set(s["rows_per_split"])
    assert s["trunc_share"]["all"] == dict.fromkeys(TRUNC_COLUMNS, 0.0)
    assert set(s["severity_nonzero_share"]) == set(SEVERITY_COLUMNS)
    n_days = len(s["days"])
    assert [r["day"] for r in s["us_per_event"]["per_day"]] == list(range(1, n_days + 1))
    assert [r["day"] for r in s["memory"]["per_day"]] == list(range(1, n_days + 1))
    h = s["headline"]
    assert h["rows"] == s["rows"] and h["restart_check_pass"] is True
    assert h["rule_trunc_share"] == 0.0 and h["us_per_event_engine"] > 0
    assert s["memory"]["max_restored_mb"] > 0 and s["memory"]["projected_mb"] is None  # no bench
    assert s["flush_ms"]["worst"]["minute"] >= 0
    rc = s["restart_check"]
    assert rc["pass"] and rc["rows_equal"] and rc["digest_equal"] and rc["files_sha256_equal"]
    assert rc["days"] == [7, 8] and rc["from"] == "val_boundary.snap"
    assert rc["rows"] == tx.filter(pl.col("day").is_between(7, 8)).height
    assert not (out / build.RESTART_DIR).exists()
    assert real_full["commits"] >= len(s["days"]) + 3
    assert read_json(out / "vocab.json") == fit_vocab(tx.filter(pl.col("split") == "train"))
    md = (paths.reports / build.ENGINE_BENCH_REPORT).read_text(encoding="utf-8")
    assert "## Full replay" in md and "Restart check" in md


def test_batches_row_groups_and_stale_outputs(fake, fake_full, paths, cfgs, tmp_path):
    """Tiny read batches and small row groups change no value or digest; a rebuild clears the
    previous build's outputs. (Byte-identical files: test_real_tiny_read_batches_...)"""
    c = copy.deepcopy(cfgs)
    c["features"]["replay"]["row_group_rows"] = 250
    out = tmp_path / "f"
    (out / PARTS_DIR).mkdir(parents=True)
    (out / PARTS_DIR / "part-d99.parquet").write_text("stale", encoding="utf-8")
    (out / "verify").mkdir()
    (out / "verify" / "verify.json").write_text("{}", encoding="utf-8")
    (out / "progress.jsonl").write_text('{"day": 0}\n', encoding="utf-8")
    s = build.run_build_features(paths, out, c, batch_rows=7, **FAKE_INPUTS)
    assert not (out / PARTS_DIR / "part-d99.parquet").exists()
    assert not (out / "verify").exists()
    assert '"day": 0' not in (out / "progress.jsonl").read_text(encoding="utf-8")
    ref = fake_full["out"]
    names = sorted(p.name for p in (ref / PARTS_DIR).glob("*.parquet"))
    assert sorted(p.name for p in (out / PARTS_DIR).glob("*.parquet")) == names
    n_groups = 0
    for name in names:
        md = pq.ParquetFile(out / PARTS_DIR / name).metadata
        sizes = [md.row_group(i).num_rows for i in range(md.num_row_groups)]
        assert all(n == 250 for n in sizes[:-1]) and (not sizes or 0 < sizes[-1] <= 250)
        n_groups += len(sizes)
        got = pq.read_table(out / PARTS_DIR / name)
        assert build.tables_bit_equal(got, pq.read_table(ref / PARTS_DIR / name)), name
    assert n_groups > len(names)
    assert s["final_state_digest"] == fake_full["summary"]["final_state_digest"]
    for split, info in s["snapshots"].items():
        assert info["sha256"] == fake_full["summary"]["snapshots"][split]["sha256"]


def test_restart_check_failure_raises_and_keeps_evidence(monkeypatch, paths, cfgs, tmp_path):
    monkeypatch.setattr(build, "Engine", LossyFakeEngine)
    out = tmp_path / "f"
    with pytest.raises(RuntimeError, match="restart check failed"):
        build.run_build_features(paths, out, cfgs, **FAKE_INPUTS)
    result = read_json(out / build.RESTART_DIR / build.RESTART_RESULT_FILE)
    assert not result["pass"] and not result["rows_equal"] and not result["digest_equal"]
    assert not (out / "summary.json").exists()


def test_full_refuses_a_failed_bench_gate(fake, paths, cfgs, tmp_path):
    out = tmp_path / "f"
    write_json_atomic({"gate": {"pass": False}}, out / build.BENCH_DIR / build.BENCH_FILE)
    with pytest.raises(RuntimeError, match="bench gate"):
        build.run_build_features(paths, out, cfgs, **FAKE_INPUTS)
    assert not (out / PARTS_DIR).exists()


def test_bench_mode(paths, cfgs, tmp_path, tx):
    out = tmp_path / "f"
    lines: list[str] = []
    s = build.run_build_features(paths, out, cfgs, mode="bench", log=lines.append)
    doc = read_json(out / build.BENCH_DIR / build.BENCH_FILE)
    assert doc["gate"] == s["gate"] and doc["mode"] == "bench"
    pass_a = read_json(out / build.BENCH_DIR / build.BENCH_PASS_A_FILE)  # saved before pass B
    assert pass_a["pass_a"]["days"] == doc["pass_a"]["days"]
    assert pass_a["projection"]["replay_min"] == doc["projection"]["replay_min"]
    assert pass_a["replay_ok"] == doc["gate"]["replay_ok"]
    for label in ("bench pass A", "bench pass B"):  # one progress line per day and pass
        assert (
            sum(x.startswith(f"{label}: day ") for x in lines)
            == cfgs["features"]["bench"]["last_day"]
        ), label
    assert not (out / "summary.json").exists() and not (out / PARTS_DIR).exists()
    last = cfgs["features"]["bench"]["last_day"]
    assert [d["day"] for d in doc["pass_a"]["days"]] == list(range(1, last + 1))
    assert [d["day"] for d in doc["pass_b"]["days"]] == list(range(1, last + 1))
    assert doc["pass_a"]["days"][-1]["boundary_minute"] == last * 1440 == doc["advance_to"]
    assert doc["pass_a"]["rows"] == tx.filter(pl.col("day") <= last).height
    assert all(d["traced_peak_mb"] > 0 for d in doc["pass_b"]["days"])
    assert doc["live_restored_ratio"] > 0
    assert doc["pass_a"]["days"][-1]["memory"]["restored_mb"] > 0
    assert doc["spec_hash"] == build.load_engine_inputs(paths, cfgs).spec_hash()
    sz = doc["sizing"]
    pairs = tx.select("src", "dst").unique()
    assert sz["lifetime_pairs"] == pairs.height
    assert sz["lifetime_pairs_nsl"] == pairs.filter(pl.col("src") != pl.col("dst")).height
    assert sz["max_in_degree"] == tx.group_by("dst").len()["len"].max()
    usd_c = (tx["amount_usd"] * 100).round(0, mode="half_away_from_zero")
    assert sz["max_usd_c"] == int(usd_c.max())
    assert sz["sums_below_2p53"] is True
    assert sz["max_ring_rows"] == _max_window_rows(tx, 4320)
    p = doc["projection"]
    assert p["rows_total"] == tx.height
    assert p["rows_restart_check"] == tx.filter(pl.col("day").is_between(7, 8)).height
    assert p["replay_min"] > 0 and p["memory_mb"] > 0
    assert set(doc["gate"]) == {
        "replay_max_min",
        "memory_target_mb",
        "replay_ok",
        "memory_ok",
        "pass",
    }
    md = (paths.reports / build.ENGINE_BENCH_REPORT).read_text(encoding="utf-8")
    assert "## Bench" in md and "Gate:" in md


def test_parts_digest_and_absent_bench_gate_are_recorded(real_full, paths):
    """summary.json records each part's SHA-256 and the table's content digest (what the
    stages reading the parts are checked against), and that no bench gated this replay."""
    from aml.features.spec import digest_of_parts, part_sha256s, parts_digest

    out, s = real_full["out"], real_full["summary"]
    hashes = part_sha256s(out)
    assert [p["sha256"] for p in s["parts"]] == [hashes[p["file"]] for p in s["parts"]]
    assert s["features_digest"] == digest_of_parts(hashes) == parts_digest(out)
    assert s["bench_gate"]["status"] == "absent" and s["bench_gate"]["pass"] is None
    assert s["headline"]["bench_gate"] == "absent"
    assert "- Bench gate: absent" in build.render_engine_bench_md(None, s)


def test_bench_gate_is_rechecked_against_the_current_settings(cfgs):
    bench = {
        "gate": {"pass": False},  # judged at bench time against an older memory target
        "projection": {"replay_min": 30.0, "memory_mb": 900.0},
    }
    c = copy.deepcopy(cfgs)
    c["features"]["memory_target_mb"] = 1000
    g = build.bench_gate(bench, c)
    assert g["pass"] is True and g["status"] == "pass" and g["memory_target_mb"] == 1000
    c["features"]["memory_target_mb"] = 800
    assert build.bench_gate(bench, c)["memory_ok"] is False
    slow = {**bench, "projection": {"replay_min": build.BENCH_MAX_REPLAY_MIN + 1, "memory_mb": 1}}
    assert build.bench_gate(slow, c)["replay_ok"] is False
    assert build.bench_gate({"gate": {"pass": True}}, c)["pass"] is True  # no projection
    assert build.bench_gate({"gate": {"pass": False}}, c)["pass"] is False
    assert build.bench_gate(None, c)["status"] == "absent"


def test_full_uses_the_rechecked_gate(fake, paths, cfgs, tmp_path):
    out = tmp_path / "f"
    bench = {  # the fields engine_bench.md renders; judged at bench time with target 800 MB
        "gate": {"pass": False, "replay_max_min": 90.0, "memory_target_mb": 800.0},
        "projection": {"replay_min": 1.0, "memory_mb": 900.0},
        "last_day": 3,
        "advance_to": 4320,
        "pass_a": {"days": []},
        "pass_b": {"days": [], "peak_mb": 900.0},
        "sizing": {},
        "live_restored_ratio": 1.0,
    }
    write_json_atomic(bench, out / build.BENCH_DIR / build.BENCH_FILE)
    c = copy.deepcopy(cfgs)
    c["features"]["memory_target_mb"] = 1000  # raised after the bench: the replay may run
    s = build.run_build_features(paths, out, c, **FAKE_INPUTS)
    assert s["bench_gate"]["status"] == "pass" and s["headline"]["bench_gate"] == "pass"


def test_bench_keeps_pass_a_when_the_traced_pass_dies(monkeypatch, paths, cfgs, tmp_path):
    """Pass A's timing, the sizing and the replay projection are committed before the slow
    traced pass B, so a timeout in pass B does not lose what the gate's replay half needs."""
    commits: list[int] = []

    class Dies(RuntimeError):
        pass

    def die(self, *a, **k):
        raise Dies

    monkeypatch.setattr(build._Traced, "__call__", die)
    out = tmp_path / "f"
    lines: list[str] = []
    with pytest.raises(Dies):
        build.run_build_features(
            paths,
            out,
            cfgs,
            mode="bench",
            on_checkpoint=lambda: commits.append(1),
            log=lines.append,
        )
    doc = read_json(out / build.BENCH_DIR / build.BENCH_PASS_A_FILE)
    assert commits and not (out / build.BENCH_DIR / build.BENCH_FILE).exists()
    last = cfgs["features"]["bench"]["last_day"]
    assert doc["mode"] == "bench_pass_a" and [d["day"] for d in doc["pass_a"]["days"]] == list(
        range(1, last + 1)
    )
    assert doc["projection"]["replay_min"] > 0 and isinstance(doc["replay_ok"], bool)
    assert doc["sizing"]["lifetime_pairs"] > 0
    assert any(x.startswith("bench pass A: day 1 done") for x in lines)
    assert any("saved bench/bench_pass_a.json" in x for x in lines)


def _max_window_rows(tx: pl.DataFrame, w: int) -> int:
    """Brute force: max over clocks m of #events with minute in [m - w, m - 1]."""
    minutes = np.sort(tx["minute"].to_numpy())
    best = 0
    for m in np.unique(minutes) + 1:
        best = max(best, int(np.searchsorted(minutes, m) - np.searchsorted(minutes, m - w)))
    return best


def test_verify_mode(monkeypatch, real_full, paths, tmp_path):
    from aml.features import oracle

    with pytest.raises(FileNotFoundError, match="mode full"):
        build.run_build_features(paths, tmp_path / "none", {}, mode="verify")
    out = tmp_path / "f"
    shutil.copytree(real_full["out"], out)
    # The real DuckDB oracle (M2 spec §9.3) on the real build: 0 mismatches.
    s = build.run_build_features(paths, out, {}, mode="verify", threads=2)
    doc = read_json(out / "verify" / "verify.json")
    assert s["n_mismatches"] == 0 and doc["n_mismatches_total"] == 0
    assert doc["rows"] == real_full["summary"]["rows"] and doc["columns_checked"] >= 65
    assert doc["spec_hash"] == real_full["summary"]["spec_hash"]
    seen = {}

    def fake_oracle(p, features_dir, spec, *, threads=4):
        seen.update(features_dir=features_dir, spec=spec, threads=threads)
        return {"rows": 10, "columns": {"u_out_cnt_1d": {"tol": "exact", "mismatches": 0}}}

    monkeypatch.setattr(oracle, "run_oracle", fake_oracle)
    s = build.run_build_features(paths, out, {}, mode="verify")
    assert s["n_mismatches"] == 0 and seen["threads"] == 4 and seen["features_dir"] == out
    assert seen["spec"] == build.load_spec(out)
    assert read_json(out / "verify" / "verify.json")["n_mismatches_total"] == 0
    # The verdict names the parts it checked (export requires them to be the build's).
    assert (
        read_json(out / "verify" / "verify.json")["features_digest"]
        == (real_full["summary"]["features_digest"])
    )

    def bad_oracle(p, features_dir, spec, *, threads=4):
        return {"columns": {"a": {"mismatches": 2}, "b": {"mismatches": 0}}}

    monkeypatch.setattr(oracle, "run_oracle", bad_oracle)
    with pytest.raises(RuntimeError, match="2 mismatches"):
        build.run_build_features(paths, out, {}, mode="verify", threads=2)
    assert read_json(out / "verify" / "verify.json")["n_mismatches_total"] == 2


def test_mode_and_boundary_validation(cfgs):
    with pytest.raises(ValueError, match="unknown mode"):
        build.run_build_features(None, Path("x"), cfgs, mode="nope")
    data = cfgs["data"]
    assert build.boundary_minutes(data, {"snapshots": {"boundaries": ["test"]}}) == {"test": 11520}
    assert build.boundary_minutes(data, {}) == {}
    with pytest.raises(ValueError, match="not a split"):
        build.boundary_minutes(data, {"snapshots": {"boundaries": ["val"]}})
    with pytest.raises(ValueError, match="no history"):
        build.boundary_minutes(data, {"snapshots": {"boundaries": ["train"]}})
    assert build.snapshot_file("val_early") == "val_boundary.snap"
    assert build.snapshot_file("val_late") == "val_late_boundary.snap"


def _batch(rank, minute, split=None) -> build._Batch:
    rank, minute = np.asarray(rank), np.asarray(minute)
    split = np.asarray(split or ["train"] * len(rank), dtype=object)
    return build._Batch([], minute, minute // 1440 + 1, rank, rank, split)


def test_batch_checks():
    build._check_batch(_batch([5, 6], [10, 10]), 5, 9, {})
    with pytest.raises(ValueError, match="contiguous"):
        build._check_batch(_batch([5, 7], [10, 10]), 5, None, {})
    with pytest.raises(ValueError, match="contiguous"):
        build._check_batch(_batch([6, 7], [10, 10]), 5, None, {})
    with pytest.raises(ValueError, match="decrease"):
        build._check_batch(_batch([5, 6], [10, 9]), 5, None, {})
    with pytest.raises(ValueError, match="decrease"):
        build._check_batch(_batch([5, 6], [10, 11]), 5, 11, {})
    bt = _batch([0, 1], [10, 10])
    bt.day = np.array([1, 2])
    with pytest.raises(ValueError, match="day"):
        build._check_batch(bt, 0, None, {})
    with pytest.raises(ValueError, match="splits"):
        build._check_batch(_batch([0, 1], [10, 11], ["train", "test"]), 0, None, {})
    with pytest.raises(ValueError, match="splits"):
        build._check_batch(_batch([0], [10], ["test"]), 0, None, {1: "train"})


def test_integer_tail_columns_fail_loudly():
    col = np.array([0.0, 3.0, 2.0**52])
    assert build._as_int(col, "x", 0, 2**53 - 1, np.int64).tolist() == [0, 3, 2**52]
    for bad in (1.5, -1.0, math.nan, math.inf, 2.0**53):
        with pytest.raises(EngineError, match="must be an integer"):
            build._as_int(np.array([0.0, bad]), "x", 0, 2**53 - 1, np.int64)


def test_tables_bit_equal_compares_float_bits():
    a = pa.table({"x": pa.array(np.array([1.0, np.nan], np.float32)), "k": [1, 2]})
    assert build.tables_bit_equal(a, a)
    neg = np.array([1.0, np.nan], np.float32)
    neg.view(np.uint32)[1] |= 0x80000000  # -nan: equal as a value, different bits
    b = pa.table({"x": pa.array(neg), "k": [1, 2]})
    assert not build.tables_bit_equal(a, b)
    assert not build.tables_bit_equal(a, a.slice(0, 1))
    assert not build.tables_bit_equal(a, pa.table({"x": a["x"], "k": [1, 3]}))


def test_validate_features_names_bad_columns(real_full):
    out = real_full["out"]
    spec = build.load_spec(out)
    df = pl.read_parquet(out / PARTS_DIR / part_name(2))
    validate_features(df, spec)
    cases = [
        ("u_out_cnt_1d", float("nan")),  # counts are never NaN
        ("u_out_cnt_1d", 1.5),  # counts are integral
        ("self_loop", 2.0),  # flags in {0, 1}
        ("payment_format", -2.0),  # codes >= -1
        ("log_amount_usd", float("inf")),  # real: finite
        ("u_out_mean_1d", float("-inf")),  # nullable: NaN allowed, never inf
        (SEVERITY_COLUMNS[2], -1.0),  # severities >= 0
        ("rule_trunc", 2),
    ]
    for col, val in cases:
        bad = df.with_columns(
            pl.when(pl.int_range(pl.len()) == 3)
            .then(pl.lit(val))
            .otherwise(pl.col(col))
            .cast(df.schema[col])
            .alias(col)
        )
        with pytest.raises(ValueError, match=col):
            validate_features(bad, spec)
    with pytest.raises(Exception, match="row_id"):
        validate_features(df.with_columns(pl.lit(1, pl.Int64).alias("row_id")), spec)
    with pytest.raises(Exception, match="u_out_cnt_1d"):
        validate_features(df.with_columns(pl.col("u_out_cnt_1d").cast(pl.Float64)), spec)
    with pytest.raises(Exception, match="extra"):
        validate_features(df.with_columns(pl.lit(0).alias("extra")), spec)


# --- the real engine (end to end on the fixture) ------------------------------------------------


def test_real_full_build(real_full, tx, cfgs):
    from aml.features.engine import Engine
    from aml.features.tx_features import build_tx_features

    out, s = real_full["out"], real_full["summary"]
    spec = build.load_spec(out)
    table = _table(out)
    assert len(list((out / PARTS_DIR).glob("part-d*.parquet"))) == int(tx["day"].max())
    assert table.height == tx.height == s["rows"]
    assert table["rank"].to_list() == list(range(tx.height))
    assert table.schema == pl.Schema(spec.table_schema())
    validate_features(table, spec)
    keys = tx.sort("rank").select("row_id", "rank", "day", "split")
    assert table.select(keys.columns).equals(keys)
    # The engine's TX group equals M1's build_tx_features (after the same float32 cast).
    m1 = build_tx_features(tx.sort("rank"), fit_vocab(tx.filter(pl.col("split") == "train")),
                           cfgs["rules"]["round_unit"])  # fmt: skip
    for name in spec.group_names("TX"):
        want = m1[name].cast(pl.Float64).to_numpy().astype(np.float32)
        got = table[name].to_numpy()
        if name == "log_amount_usd":
            assert np.all(np.abs(got - want) <= np.spacing(np.abs(want))), name
        else:
            assert np.array_equal(got.view(np.uint32), want.view(np.uint32)), name
    rc = s["restart_check"]
    assert rc["pass"] and rc["rows_equal"] and rc["digest_equal"] and rc["days"] == [7, 8]
    for split, b in build.boundary_minutes(cfgs["data"], cfgs["features"]).items():
        first = int(tx.filter(pl.col("minute") >= b)["rank"].min())
        info = s["snapshots"][split]
        assert info["next_rank"] == first and info["next_offset"] == 0
        eng, header = Engine.restore(out / SNAPSHOTS_DIR / info["file"], spec)
        assert header["next_rank"] == first and eng.state_digest() == info["state_digest"]
    assert s["hub_cap"] == spec.hub_cap and s["n_hubs"] == len(spec.hubs)
    assert set(s["trunc_share"]["per_split"]) == set(s["rows_per_split"])


def test_real_parts_equal_a_plain_process_loop(real_full, tx):
    """The driver's own minute flushes, batching and row groups add nothing: every part value
    equals `eng.process(eng.prepare(...))` over the events in rank order, with the §4.9 cast."""
    from aml.features.engine import Engine
    from aml.features.spec import INPUT_COLUMNS

    out = real_full["out"]
    spec = build.load_spec(out)
    eng = Engine.create(spec)
    ordered = tx.sort("rank")
    rows = [
        eng.process(eng.prepare(*f))
        for f in zip(*(ordered[c].to_list() for c in INPUT_COLUMNS), strict=True)
    ]
    mat = np.asarray(rows, dtype=np.float64)
    table = _table(out)
    feats = mat[:, : spec.n_features].astype(np.float32)
    got = table.select(spec.feature_names).to_numpy()
    assert np.array_equal(got.view(np.uint32), np.ascontiguousarray(feats).view(np.uint32))
    sev = mat[:, spec.i_sev : spec.i_sev + len(SEVERITY_COLUMNS)]
    got = np.ascontiguousarray(table.select(SEVERITY_COLUMNS).to_numpy())
    assert np.array_equal(got.view(np.uint64), np.ascontiguousarray(sev).view(np.uint64))
    assert table["inflow_c"].to_list() == [int(x) for x in mat[:, spec.i_inflow]]
    tail = [spec.i_rule_trunc, spec.i_cyc_trunc, spec.i_sg_trunc]
    assert np.array_equal(table.select(TRUNC_COLUMNS).to_numpy(), mat[:, tail])


def test_real_tiny_read_batches_give_identical_output(real_full, paths, cfgs, tmp_path):
    out = tmp_path / "f"
    s = build.run_build_features(paths, out, cfgs, batch_rows=97)
    ref = real_full["out"]
    for p in sorted((ref / PARTS_DIR).glob("*.parquet")):
        assert build.tables_bit_equal(pq.read_table(p), pq.read_table(out / PARTS_DIR / p.name))
        assert p.read_bytes() == (out / PARTS_DIR / p.name).read_bytes(), p.name
    assert s["final_state_digest"] == real_full["summary"]["final_state_digest"]
    for split, info in s["snapshots"].items():
        assert info["state_digest"] == real_full["summary"]["snapshots"][split]["state_digest"]
