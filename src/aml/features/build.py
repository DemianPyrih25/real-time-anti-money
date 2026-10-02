"""The offline driver: replay transactions.parquet through the engine into the feature table
(M2 spec §8.1). The only source of offline features.

Output layout (inside `paths.features_dir(features_key)`, names from `aml.features.spec`):
feature_spec.json, vocab.json, summary.json (written last), data_version.json (the Modal job),
progress.jsonl, parts/part-dNN.parquet (one per simulated day, rank order, schema
`spec.table_schema()`), snapshots/val_boundary.snap and test_boundary.snap (+ .json sidecars),
bench/bench.json (+ bench/bench_pass_a.json), verify/verify.json. The report `engine_bench.md`
goes to `paths.reports`.

Modes:
- full: the replay. At every day boundary b = (d - 1) * 1440 (and at the end, b = last minute + 1)
  the engine advances to b, its state is restored from a raw snapshot under tracemalloc (the
  restore-measure) and a line is appended to progress.jsonl. Boundary snapshots are written at
  the first minute of the `features.snapshots.boundaries` splits. Then the restart check:
  restore the first boundary snapshot, re-replay up to the next boundary and require bit-equal
  rows and an equal state digest. The bench gate is re-checked first (no bench: recorded as
  "absent"); summary.json records each part's SHA-256 and the table's `features_digest`.
- bench: DuckDB sizing queries, a timing pass (A) and a tracemalloc pass (B) over days
  1..bench.last_day, then the projected full-replay time and memory and the gate. Pass A's
  results and replay projection are saved (bench_pass_a.json) before the slow traced pass.
- verify: the DuckDB feature oracle (`features.oracle.run_oracle`) over all rows; verify.json
  records the parts' digest it checked.

The driver calls the minute flush itself (`Engine.advance` before the first event of every new
minute); `Engine.process` then finds the clock already there, so its output is unchanged and the
flush time is measured apart from the scoring time. No label is read anywhere here.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from itertools import islice
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq

from aml.data.schemas import validate_features
from aml.data.split import SPLITS, split_days
from aml.features.engine import Engine
from aml.features.spec import (
    DRIVER_COLUMNS,
    ENGINE_VERSION,
    EXACT_INT_LIMIT,
    FEATURE_SPEC_FILE,
    FEATURES_DIGEST,
    INPUT_COLUMNS,
    MINUTES_PER_DAY,
    PARTS_DIR,
    PROGRESS_FILE,
    SEVERITY_COLUMNS,
    SNAPSHOT_FILES,
    SNAPSHOTS_DIR,
    SUMMARY_FILE,
    TRUNC_COLUMNS,
    VOCAB_FILE,
    EngineError,
    EngineSpec,
    digest_of_parts,
    part_name,
    part_sha256s,
    parts_digest,
)
from aml.features.tx_features import CATEGORICAL_FEATURES, fit_vocab
from aml.io import read_json, write_json_atomic, write_text_atomic
from aml.paths import DataPaths
from aml.rules import sql_baseline

MODES = ("bench", "full", "verify")
# The split whose rows fit the vocabulary (and, in SQL, the hub list): M1's train split.
TRAIN_SPLIT = "train"
READ_COLUMNS = (*INPUT_COLUMNS, *DRIVER_COLUMNS)
BENCH_DIR = "bench"
BENCH_FILE = "bench.json"
BENCH_PASS_A_FILE = "bench_pass_a.json"  # saved before the slow traced pass; never a gate
VERIFY_DIR = "verify"
VERIFY_FILE = "verify.json"
RESTART_DIR = "restart_check"
RESTART_RESULT_FILE = "result.json"
ENGINE_BENCH_REPORT = "engine_bench.md"
# The bench gate before `full` (M2 spec §8.1): projected replay time; memory uses memory_target_mb.
BENCH_MAX_REPLAY_MIN = 90.0
# A boundary snapshot is the start state of a replay that begins at that boundary: its first
# pending event is that replay's offset 0 (the serving bundle requires next_offset = 0).
BOUNDARY_NEXT_OFFSET = 0
# Planning bytes per lifetime pair (M2 spec §4.3 D) when the engine reports no pair component.
PLANNING_PAIR_BYTES = 125
DUCKDB_MEMORY_LIMIT = "3GB"
MB = 1e6  # reports use decimal megabytes

_PA_TYPES = {
    pl.Int64: pa.int64(),
    pl.Int16: pa.int16(),
    pl.Int8: pa.int8(),
    pl.String: pa.string(),
    pl.Float32: pa.float32(),
    pl.Float64: pa.float64(),
}


# --- inputs -------------------------------------------------------------------------------------


@contextmanager
def _duckdb(threads: int = 1) -> Iterator[Any]:
    """In-memory DuckDB (1 thread by default) with a local spill directory, closed on exit."""
    spill = Path(tempfile.mkdtemp(prefix="aml-features-duckdb-"))
    con = sql_baseline.connect(threads, DUCKDB_MEMORY_LIMIT, spill)
    try:
        yield con
    finally:
        con.close()
        shutil.rmtree(spill, ignore_errors=True)


def load_engine_inputs(
    paths: DataPaths,
    cfgs: dict[str, dict],
    *,
    vocab: dict[str, list[str]] | None = None,
    hubs: list[int] | None = None,
    hub_cap: int | None = None,
) -> EngineSpec:
    """The EngineSpec of a replay: n_accounts = accounts table height, vocab = fit_vocab(train
    rows), hub_cap / hubs on DuckDB with 1 thread (sql_baseline.hub_degree_cap / hub_accounts).
    Explicit vocab / hubs / hub_cap (tests) replace the fitted ones."""
    rules_cfg = cfgs["rules"]
    n_accounts = pq.ParquetFile(paths.accounts).metadata.num_rows
    if vocab is None:
        # Distinct value combinations of train rows: the per-column sorted uniques fit_vocab
        # returns are the same as on all train rows, at a fraction of the memory.
        train = (
            pl.scan_parquet(paths.transactions)
            .filter(pl.col("split") == TRAIN_SPLIT)
            .select(CATEGORICAL_FEATURES)
            .unique()
            .collect()
        )
        if train.height == 0:
            raise ValueError("no train rows: cannot fit the vocabulary")
        vocab = fit_vocab(train)
    if hub_cap is None or hubs is None:
        with _duckdb(1) as con:
            sql_baseline.register_transactions(con, paths.transactions)
            if hub_cap is None:
                hub_cap = sql_baseline.hub_degree_cap(con, float(rules_cfg["hub_degree_quantile"]))
            if hubs is None:
                hubs = sql_baseline.hub_accounts(con, hub_cap)
    return EngineSpec.from_configs(
        cfgs["features"],
        rules_cfg,
        n_accounts=n_accounts,
        vocab=vocab,
        hub_cap=int(hub_cap),
        hubs=hubs,
    )


def boundary_minutes(data_cfg: dict, features_cfg: dict) -> dict[str, int]:
    """Snapshot boundary per `features.snapshots.boundaries` split: the first minute of its first
    day, (day - 1) * 1440 (val_early -> 8640, test -> 11520 on HI-Small)."""
    out: dict[str, int] = {}
    for split in (features_cfg.get("snapshots") or {}).get("boundaries") or []:
        if split not in SPLITS:
            raise ValueError(f"snapshot boundary {split!r} is not a split; splits are {SPLITS}")
        first_day = split_days(data_cfg, split)[0]
        if first_day < 2:
            raise ValueError(f"snapshot boundary {split!r} starts on day {first_day}: no history")
        out[split] = (first_day - 1) * MINUTES_PER_DAY
    if len(set(out.values())) != len(out):
        raise ValueError(f"snapshot boundaries repeat a minute: {out}")
    return dict(sorted(out.items(), key=lambda kv: kv[1]))


def snapshot_file(split: str) -> str:
    """Boundary snapshot file name of a split (spec.SNAPSHOT_FILES, else <split>_boundary.snap)."""
    return SNAPSHOT_FILES.get(split, f"{split}_boundary.snap")


def load_spec(features_dir: Path) -> EngineSpec:
    """The EngineSpec a finished replay was built with (its feature_spec.json)."""
    return EngineSpec.from_json(read_json(Path(features_dir) / FEATURE_SPEC_FILE))


# --- reading ------------------------------------------------------------------------------------


@dataclass
class _Batch:
    cols: list[list]  # INPUT_COLUMNS as Python lists (Engine.prepare arguments)
    minute: np.ndarray
    day: np.ndarray
    rank: np.ndarray
    row_id: np.ndarray
    split: np.ndarray  # object array of split names


def _batches(
    path: Path, batch_rows: int, *, start_rank: int = 0, use_threads: bool = False
) -> Iterator[_Batch]:
    """Rank-ordered batches of the transactions table from `start_rank` on.

    Row groups that end before `start_rank` are skipped via the metadata (rank = row index in
    transactions.parquet); the driver asserts rank contiguity on every batch.
    """
    pf = pq.ParquetFile(path)
    md = pf.metadata
    first, skipped = 0, 0
    for i in range(md.num_row_groups):
        n = md.row_group(i).num_rows
        if first + n > start_rank:
            break
        first += n
        skipped = i + 1
    groups = list(range(skipped, md.num_row_groups))
    if not groups:
        return
    for rb in pf.iter_batches(
        batch_size=batch_rows,
        row_groups=groups,
        columns=list(READ_COLUMNS),
        use_threads=use_threads,
    ):
        rank = rb.column("rank").to_numpy()
        if rank.size == 0 or rank[-1] < start_rank:
            continue
        if rank[0] < start_rank:
            rb = rb.slice(int(np.searchsorted(rank, start_rank)))
            rank = rb.column("rank").to_numpy()
        yield _Batch(
            cols=[rb.column(c).to_pylist() for c in INPUT_COLUMNS],
            minute=rb.column("minute").to_numpy(),
            day=rb.column("day").to_numpy(),
            rank=rank,
            row_id=rb.column("row_id").to_numpy(),
            split=rb.column("split").to_numpy(zero_copy_only=False),
        )


# --- writing ------------------------------------------------------------------------------------


def arrow_schema(spec: EngineSpec) -> pa.Schema:
    """`spec.table_schema()` as an Arrow schema (the Parquet parts' schema)."""
    return pa.schema([(n, _PA_TYPES[type(dt)]) for n, dt in spec.table_schema().items()])


@dataclass
class _SplitStats:
    rows: int = 0
    trunc: np.ndarray = field(default_factory=lambda: np.zeros(len(TRUNC_COLUMNS), np.int64))
    sev_nonzero: np.ndarray = field(
        default_factory=lambda: np.zeros(len(SEVERITY_COLUMNS), np.int64)
    )
    max_inflow_c: int = 0


class _PartWriter:
    """Engine rows of one simulated day -> parts/part-dNN.parquet, row groups of `rg_rows`.

    The replay loop packs every engine row into `buf`, a preallocated bytearray of one row group
    of float64 rows, with `packer.pack_into` (about 3x cheaper per event than `array.extend`,
    which parses each value; a wrong row length or a non-number fails loudly). `inflow_c` and the
    trunc flags travel as doubles and are checked integral before the integer cast. Every full
    row group is cast (features to Float32 by `np.asarray(values, np.float64).astype(np.float32)`,
    M2 spec §4.9), validated and written to `.part-dNN.parquet.tmp`; `close_day` renames it into
    place.
    """

    def __init__(self, parts_dir: Path, spec: EngineSpec, rg_rows: int) -> None:
        if rg_rows < 1:
            raise ValueError(f"row_group_rows must be >= 1, got {rg_rows}")
        self.parts_dir = Path(parts_dir)
        self.spec = spec
        self.rg_rows = int(rg_rows)
        self.schema = arrow_schema(spec)
        self.packer = struct.Struct(f"={spec.row_len}d")  # native float64, no padding
        self.buf = bytearray(self.packer.size * self.rg_rows)
        self.row_ids: list[np.ndarray] = []
        self.n_buf = 0
        self.day: int | None = None
        self.split = ""
        self.rank0 = 0  # rank of the buffer's first row
        self.n_day = 0
        self.writer: pq.ParquetWriter | None = None
        self.tmp: Path | None = None
        self.write_s = 0.0
        self.splits: dict[str, _SplitStats] = {}
        self.parts: list[dict[str, Any]] = []

    @property
    def room(self) -> int:
        return self.rg_rows - self.n_buf

    def open_day(self, day: int, split: str, first_rank: int) -> None:
        if self.day is not None:
            raise RuntimeError(f"part of day {self.day} is still open")
        self.parts_dir.mkdir(parents=True, exist_ok=True)
        self.day, self.split, self.rank0, self.n_day = int(day), str(split), int(first_rank), 0
        self.tmp = self.parts_dir / f".{part_name(day)}.tmp"
        self.writer = pq.ParquetWriter(
            self.tmp, self.schema, compression="zstd", write_statistics=True
        )

    def added(self, row_ids: np.ndarray) -> None:
        """The loop packed len(row_ids) more rows into `buf`; flush when the row group is full."""
        self.row_ids.append(row_ids)
        self.n_buf += len(row_ids)
        if self.n_buf >= self.rg_rows:
            self.flush()

    def flush(self) -> None:
        n = self.n_buf
        if n == 0:
            return
        t = time.perf_counter()
        spec = self.spec
        width = spec.row_len
        mat = np.frombuffer(self.buf, dtype=np.float64, count=n * width).reshape(n, width)
        nf = spec.n_features
        feats = mat[:, :nf].T.astype(np.float32, order="C")  # IEEE round-to-nearest per value
        sev = mat[:, spec.i_sev : spec.i_sev + len(SEVERITY_COLUMNS)].T.copy(order="C")
        inflow = _as_int(mat[:, spec.i_inflow], "inflow_c", 0, EXACT_INT_LIMIT - 1, np.int64)
        trunc = np.stack(
            [
                _as_int(mat[:, i], name, 0, 1, np.int8)
                for i, name in zip(
                    (spec.i_rule_trunc, spec.i_cyc_trunc, spec.i_sg_trunc),
                    TRUNC_COLUMNS,
                    strict=True,
                )
            ]
        )
        del mat  # (copies were taken above; buf is overwritten by the next row group)
        row_id = np.concatenate(self.row_ids).astype(np.int64, copy=False)
        if row_id.size != n:
            raise RuntimeError(f"{row_id.size} row ids for {n} rows")
        arrays = [
            pa.array(row_id),
            pa.array(np.arange(self.rank0, self.rank0 + n, dtype=np.int64)),
            pa.array(np.full(n, self.day, dtype=np.int16)),
            pa.array([self.split] * n, pa.string()),
            *(pa.array(feats[j]) for j in range(nf)),
            *(pa.array(sev[j]) for j in range(sev.shape[0])),
            pa.array(inflow),
            *(pa.array(trunc[j]) for j in range(trunc.shape[0])),
        ]
        table = pa.Table.from_arrays(arrays, schema=self.schema)
        validate_features(pl.from_arrow(table), spec)
        self.writer.write_table(table, row_group_size=n)

        st = self.splits.setdefault(self.split, _SplitStats())
        st.rows += n
        st.trunc += trunc.sum(axis=1, dtype=np.int64)
        st.sev_nonzero += (sev > 0).sum(axis=1)
        st.max_inflow_c = max(st.max_inflow_c, int(inflow.max()))
        self.n_day += n
        self.rank0 += n
        self.row_ids.clear()
        self.n_buf = 0
        self.write_s += time.perf_counter() - t

    def close_day(self) -> dict[str, Any]:
        """Flush the last row group, close the writer and rename the part into place."""
        self.flush()
        t = time.perf_counter()
        self.writer.close()
        final = self.parts_dir / part_name(self.day)
        os.replace(self.tmp, final)
        self.write_s += time.perf_counter() - t
        info = {"day": self.day, "rows": self.n_day, "file": final.name}
        self.parts.append(info)
        self.day, self.writer, self.tmp = None, None, None
        return info

    def write_empty(self, day: int) -> dict[str, Any]:
        """A zero-row part for a simulated day without events (one part per day)."""
        self.open_day(day, "", self.rank0)
        return self.close_day()


def _as_int(col: np.ndarray, name: str, lo: int, hi: int, dtype: Any) -> np.ndarray:
    """Exact integer column from the doubles of the row buffer (fails loudly, never rounds)."""
    ok = np.isfinite(col) & (col == np.floor(col)) & (col >= lo) & (col <= hi)
    if not ok.all():
        bad = col[~ok][:5].tolist()
        raise EngineError(f"engine output {name} must be an integer in [{lo}, {hi}], got {bad}")
    return col.astype(dtype)


# --- the replay loop ----------------------------------------------------------------------------


@dataclass
class _Day:
    """Statistics of one simulated day of a replay."""

    day: int
    first_rank: int | None = None
    events: int = 0
    score_s: float = 0.0
    flush_s: float = 0.0
    flushes: list[float] = field(default_factory=list)  # seconds of every minute flush
    worst: tuple[float, int, int] = (0.0, -1, 0)  # (seconds, minute, events applied)

    def flush_done(self, seconds: float, minute: int, n_applied: int) -> None:
        self.flush_s += seconds
        self.flushes.append(seconds)
        if seconds > self.worst[0]:
            self.worst = (seconds, minute, n_applied)


# boundary(day_stats, b, next_rank, final): after the finished day's part is closed and the engine
# advanced to b; `final` is True for the end-of-data boundary.
Boundary = Callable[[_Day, int, int, bool], None]


def _replay(
    eng: Any,
    source: Iterator[_Batch],
    *,
    start_rank: int,
    writer: _PartWriter | None,
    boundary: Boundary,
    stop_day: int | None = None,
) -> dict[str, Any]:
    """Feed rank-ordered events to the engine; parts per day; `boundary` at every day start.

    Stops at the start of `stop_day` (after its boundary) or at the end of the data (then the
    final boundary is at last minute + 1). Returns the last day, minute and next rank.
    """
    process, prepare, advance = eng.process, eng.prepare, eng.advance
    if writer is not None:
        pack_into, buf, size = writer.packer.pack_into, writer.buf, writer.packer.size
    perf = time.perf_counter
    next_rank = start_rank
    prev_minute: int | None = None
    cur: _Day | None = None
    day_split: dict[int, str] = {}
    stopped = False

    def finish(cur: _Day, upto: int) -> bool:
        """Boundaries at the start of days cur.day + 1 .. upto; True if stop_day was reached."""
        stats = cur
        for dn in range(cur.day + 1, upto + 1):
            if writer is not None:
                if stats.day == cur.day:
                    writer.close_day()
                else:
                    writer.write_empty(stats.day)
            b = (dn - 1) * MINUTES_PER_DAY
            t = perf()
            st = advance(b)
            stats.flush_done(perf() - t, b, st.n_applied)
            boundary(stats, b, next_rank, False)
            if stop_day is not None and dn >= stop_day:
                return True
            stats = _Day(dn, first_rank=next_rank)
        return False

    was_enabled = gc.isenabled()
    gc.disable()  # the loop creates no reference cycles; collection pauses would only add noise
    try:
        for bt in source:
            n = len(bt.minute)
            if n == 0:
                continue
            _check_batch(bt, next_rank, prev_minute, day_split)
            it = zip(*bt.cols, strict=True)
            cuts = (np.flatnonzero(bt.minute[1:] != bt.minute[:-1]) + 1).tolist()
            bounds = [0, *cuts, n]
            for k in range(len(bounds) - 1):
                s, e = bounds[k], bounds[k + 1]
                if k > 0 or prev_minute is None or int(bt.minute[0]) != prev_minute:
                    m, d = int(bt.minute[s]), int(bt.day[s])
                    if cur is None or d != cur.day:
                        if cur is None and stop_day is not None and d >= stop_day:
                            raise ValueError(f"replay starts on day {d} >= stop_day {stop_day}")
                        if cur is not None and finish(cur, d):
                            stopped = True
                            break
                        cur = _Day(d, first_rank=int(bt.rank[s]))
                        if writer is not None:
                            writer.open_day(d, day_split[d], int(bt.rank[s]))
                    t = perf()
                    st = advance(m)
                    cur.flush_done(perf() - t, m, st.n_applied)
                    prev_minute = m
                count = e - s
                if writer is None:
                    t = perf()
                    for f in islice(it, count):
                        process(prepare(*f))
                    cur.score_s += perf() - t
                else:
                    pos = s
                    while pos < e:
                        take = min(e - pos, writer.room)
                        off = writer.n_buf * size
                        t = perf()
                        for f in islice(it, take):
                            pack_into(buf, off, *process(prepare(*f)))
                            off += size
                        cur.score_s += perf() - t
                        writer.added(bt.row_id[pos : pos + take])
                        pos += take
                cur.events += count
                next_rank += count
            if stopped:
                break
            if next(it, None) is not None:
                raise RuntimeError("replay loop left events of a batch unprocessed")
        if cur is None:
            raise ValueError("no events to replay")
        if not stopped:
            if stop_day is not None:
                finish(cur, stop_day)
            else:
                if writer is not None:
                    writer.close_day()
                b = prev_minute + 1
                t = perf()
                st = advance(b)
                cur.flush_done(perf() - t, b, st.n_applied)
                boundary(cur, b, next_rank, True)
    finally:
        if was_enabled:
            gc.enable()
    return {"last_day": cur.day, "last_minute": prev_minute, "next_rank": next_rank}


def _check_batch(
    bt: _Batch, next_rank: int, prev_minute: int | None, day_split: dict[int, str]
) -> None:
    """Contiguous ranks, non-decreasing minutes, day = minute // 1440 + 1, one split per day."""
    n = len(bt.minute)
    if not np.array_equal(bt.rank, np.arange(next_rank, next_rank + n, dtype=bt.rank.dtype)):
        raise ValueError(f"ranks are not contiguous from {next_rank} in a batch of {n} rows")
    if (prev_minute is not None and int(bt.minute[0]) < prev_minute) or (
        n > 1 and bool((bt.minute[1:] < bt.minute[:-1]).any())
    ):
        raise ValueError(f"minutes decrease in the batch starting at rank {next_rank}")
    if not np.array_equal(bt.day, bt.minute // MINUTES_PER_DAY + 1):
        raise ValueError(f"day != minute // 1440 + 1 in the batch starting at rank {next_rank}")
    chg = np.flatnonzero((bt.day[1:] != bt.day[:-1]) | (bt.split[1:] != bt.split[:-1])) + 1
    for i in [0, *chg.tolist()]:
        d, s = int(bt.day[i]), str(bt.split[i])
        if day_split.setdefault(d, s) != s:
            raise ValueError(f"day {d} holds rows of splits {day_split[d]!r} and {s!r}")


# --- measurements -------------------------------------------------------------------------------


def _ru_maxrss_mb() -> float | None:
    try:
        import resource
    except ImportError:  # Windows
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024 / MB  # Linux: KiB


def _restore_measure(eng: Any, spec: EngineSpec) -> dict[str, Any]:
    """Traced bytes of an engine restored from a raw snapshot of `eng` (M2 spec §8.1 step 4).

    The restored engine holds the state without the allocator history of the replay, so this is
    the state's own size; the bench's live/restored ratio converts it to process memory.
    """
    gc.collect()
    t = time.perf_counter()
    raw = eng.snapshot(compress=False)
    tracing = tracemalloc.is_tracing()
    if tracing:
        raise RuntimeError("restore-measure needs tracemalloc off (it starts its own trace)")
    tracemalloc.start()
    try:
        e2, header = Engine.restore(raw, spec)
        restored = tracemalloc.get_traced_memory()[0]
        del e2
    finally:
        tracemalloc.stop()
    del raw
    return {
        "restored_bytes": int(restored),
        "header": header,
        "seconds": time.perf_counter() - t,
    }


def _state_total(nbytes: dict[str, int]) -> int:
    if "total" in nbytes:
        return int(nbytes["total"])
    return int(sum(int(v) for v in nbytes.values()))


def _pair_bytes(nbytes: dict[str, int]) -> int | None:
    keys = [k for k in nbytes if k.startswith("pair")]
    return int(sum(int(nbytes[k]) for k in keys)) if keys else None


def _flush_ms(flushes: list[float]) -> dict[str, Any]:
    if not flushes:
        return {"n": 0, "p50": None, "p99": None, "max": None}
    a = np.asarray(flushes, dtype=np.float64) * 1e3
    return {
        "n": int(a.size),
        "p50": float(np.percentile(a, 50)),
        "p99": float(np.percentile(a, 99)),
        "max": float(a.max()),
    }


def _us(seconds: float, events: int) -> float | None:
    return seconds / events * 1e6 if events else None


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, sort_keys=True, default=_json_default) + "\n")


def _json_default(o: Any) -> Any:
    if hasattr(o, "item") and callable(o.item) and getattr(o, "shape", None) == ():
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _day_record(stats: _Day, b: int, next_rank: int) -> dict[str, Any]:
    """The timing part of a progress line (memory is added by the caller)."""
    eng_s = stats.score_s + stats.flush_s
    return {
        "day": stats.day,
        "boundary_minute": b,
        "events": stats.events,
        "first_rank": stats.first_rank,
        "next_rank": next_rank,
        "us_per_event": {
            "engine": _us(eng_s, stats.events),
            "score": _us(stats.score_s, stats.events),
            "flush": _us(stats.flush_s, stats.events),
        },
        "seconds": {"score": stats.score_s, "flush": stats.flush_s},
        "flush_ms": _flush_ms(stats.flushes),
        "worst_flush": {
            "ms": stats.worst[0] * 1e3,
            "minute": stats.worst[1],
            "n_applied": stats.worst[2],
        },
    }


# --- run state ----------------------------------------------------------------------------------


class _Measured:
    """Boundary callback of a timed replay (full mode and bench pass A).

    At every boundary: write time of the closed part, restore-measure (when `measure`), state
    sizes, ru_maxrss; boundary snapshots; a progress line (when `progress` is set) and a commit.
    """

    def __init__(
        self,
        eng: Any,
        spec: EngineSpec,
        *,
        writer: _PartWriter | None,
        measure: bool,
        progress: Path | None,
        snapshots: dict[int, tuple[str, Path]] | None,
        commit: Callable[[], None],
        dataset: str,
        say: Callable[[str], None] | None = None,
        label: str = "replay",
    ) -> None:
        self.eng, self.spec, self.writer = eng, spec, writer
        self.measure, self.progress = measure, progress
        self.snapshot_at = snapshots or {}
        self.commit, self.dataset = commit, dataset
        self.say, self.label = say, label
        self.days: list[dict[str, Any]] = []
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.flushes: list[float] = []
        self.worst: tuple[float, int, int] = (0.0, -1, 0)
        self.final_digest: str | None = None
        self._write_mark = 0.0
        self.t0 = time.perf_counter()

    def __call__(self, stats: _Day, b: int, next_rank: int, final: bool) -> None:
        rec = _day_record(stats, b, next_rank)
        self.flushes.extend(stats.flushes)
        if stats.worst[0] > self.worst[0]:
            self.worst = stats.worst
        if self.writer is not None:
            rec["seconds"]["write"] = self.writer.write_s - self._write_mark
            self._write_mark = self.writer.write_s
            ev = stats.events
            rec["us_per_event"]["io"] = _us(rec["seconds"]["write"], ev)
        nbytes = {k: int(v) for k, v in self.eng.state_nbytes().items()}
        mem: dict[str, Any] = {
            "state_nbytes": nbytes,
            "state_mb": _state_total(nbytes) / MB,
            "ru_maxrss_mb": _ru_maxrss_mb(),
        }
        if self.measure:
            m = _restore_measure(self.eng, self.spec)
            h = m["header"]
            if h.get("next_rank") != next_rank:
                raise EngineError(f"snapshot next_rank {h.get('next_rank')} != {next_rank}")
            mem.update(
                restored_mb=m["restored_bytes"] / MB,
                n_pairs=h.get("n_pairs"),
                ring_rows_live=next_rank - int(h["live_start"]) if "live_start" in h else None,
            )
            rec["seconds"]["measure"] = m["seconds"]
        rec["memory"] = mem
        if b in self.snapshot_at:
            split, path = self.snapshot_at[b]
            t = time.perf_counter()
            self.snapshots[split] = self._snapshot(split, path, b, next_rank)
            rec["seconds"]["snapshot"] = time.perf_counter() - t
            rec["snapshot"] = path.name
        if final:
            self.final_digest = self.eng.state_digest()
            rec["final_state_digest"] = self.final_digest
        rec["elapsed_s"] = time.perf_counter() - self.t0
        self.days.append(rec)
        if self.progress is not None:
            _append_jsonl(self.progress, rec)
        self.commit()
        if self.say is not None:
            self.say(_day_line(self.label, rec["day"], rec["events"], stats, rec["elapsed_s"]))

    def _snapshot(self, split: str, path: Path, b: int, next_rank: int) -> dict[str, Any]:
        path.parent.mkdir(parents=True, exist_ok=True)
        header = self.eng.snapshot(
            path,
            next_offset=BOUNDARY_NEXT_OFFSET,
            extra={"kind": "boundary", "split": split, "minute": b, "dataset": self.dataset},
        )
        if header.get("next_rank") != next_rank:
            raise EngineError(f"{path.name}: next_rank {header.get('next_rank')} != {next_rank}")
        return {
            "file": path.name,
            "split": split,
            "minute": b,
            "clock": header.get("clock"),
            "next_rank": header.get("next_rank"),
            "next_offset": header.get("next_offset"),
            "state_digest": header.get("state_digest"),
            "raw_nbytes": header.get("raw_nbytes"),
            "bytes": path.stat().st_size,
            "sha256": _sha256_file(path),
        }


class _Traced:
    """Boundary callback of bench pass B: traced current / peak bytes per day (tracemalloc on)."""

    def __init__(self, say: Callable[[str], None] | None = None) -> None:
        self.days: list[dict[str, Any]] = []
        self.say = say
        self.t0 = time.perf_counter()

    def __call__(self, stats: _Day, b: int, next_rank: int, final: bool) -> None:
        cur, peak = tracemalloc.get_traced_memory()
        tracemalloc.reset_peak()
        self.days.append(
            {
                "day": stats.day,
                "boundary_minute": b,
                "events": stats.events,
                "traced_current_mb": cur / MB,
                "traced_peak_mb": peak / MB,
                "us_per_event_traced": _us(stats.score_s + stats.flush_s, stats.events),
            }
        )
        if self.say is not None:
            line = _day_line("bench pass B", stats.day, stats.events, stats, self._elapsed())
            self.say(f"{line}; traced peak {peak / MB:,.1f} MB")

    def _elapsed(self) -> float:
        return time.perf_counter() - self.t0


def _day_line(label: str, day: int, events: int, stats: _Day, elapsed_s: float) -> str:
    """One progress line per replayed day (the Modal job prints it)."""
    us = _us(stats.score_s + stats.flush_s, events)
    us_txt = "n/a" if us is None else f"{us:,.1f}"
    return (
        f"{label}: day {day} done, {events:,} events, {us_txt} µs/event, "
        f"elapsed {elapsed_s / 60:,.1f} min"
    )


# --- full mode ----------------------------------------------------------------------------------


def _clear_outputs(out_dir: Path) -> None:
    """Remove the previous replay's outputs (summary.json first: it marks a finished build)."""
    (out_dir / SUMMARY_FILE).unlink(missing_ok=True)
    for d in (PARTS_DIR, SNAPSHOTS_DIR, RESTART_DIR, VERIFY_DIR):
        shutil.rmtree(out_dir / d, ignore_errors=True)
    for f in (PROGRESS_FILE, FEATURE_SPEC_FILE, VOCAB_FILE):
        (out_dir / f).unlink(missing_ok=True)


def _write_inputs(out_dir: Path, spec: EngineSpec) -> None:
    write_json_atomic(spec.to_json(), out_dir / FEATURE_SPEC_FILE)
    write_json_atomic({k: list(v) for k, v in spec.vocab.items()}, out_dir / VOCAB_FILE)


def _bench_doc(out_dir: Path) -> dict[str, Any] | None:
    path = out_dir / BENCH_DIR / BENCH_FILE
    return read_json(path) if path.exists() else None


def bench_gate(bench: dict[str, Any] | None, cfgs: dict[str, dict]) -> dict[str, Any]:
    """The bench gate before `full`, re-evaluated against the current settings.

    The bench's projected replay minutes and memory are compared with BENCH_MAX_REPLAY_MIN and
    the current `memory_target_mb` (excluded from the features key, so it may change after the
    bench). A bench document without a projection keeps its own `gate.pass`. No bench: status
    "absent", pass None (the replay runs ungated and its memory projection is n/a).
    """
    if bench is None:
        return {
            "status": "absent",
            "pass": None,
            "note": "no bench/bench.json for this features key: the full replay ran ungated "
            "and its memory projection is n/a (`make features-bench` first)",
        }
    proj = bench.get("projection") or {}
    replay_min, memory_mb = proj.get("replay_min"), proj.get("memory_mb")
    if replay_min is None or memory_mb is None:
        ok = bool((bench.get("gate") or {}).get("pass", False))
        return {"status": "pass" if ok else "fail", "pass": ok, "source": "bench gate.pass"}
    target = float(cfgs["features"].get("memory_target_mb", 1000))
    out = {
        "replay_min": float(replay_min),
        "replay_max_min": BENCH_MAX_REPLAY_MIN,
        "memory_mb": float(memory_mb),
        "memory_target_mb": target,
        "replay_ok": float(replay_min) <= BENCH_MAX_REPLAY_MIN,
        "memory_ok": float(memory_mb) < target,
        "source": "bench projection vs the current settings",
    }
    out["pass"] = out["replay_ok"] and out["memory_ok"]
    out["status"] = "pass" if out["pass"] else "fail"
    return out


def _run_full(
    paths: DataPaths,
    out_dir: Path,
    cfgs: dict[str, dict],
    *,
    vocab: dict[str, list[str]] | None,
    hubs: list[int] | None,
    hub_cap: int | None,
    batch_rows: int,
    rg_rows: int,
    use_threads: bool,
    commit: Callable[[], None],
    say: Callable[[str], None],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    timings: dict[str, float] = {}
    bench = _bench_doc(out_dir)
    gate = bench_gate(bench, cfgs)
    if gate["pass"] is False:
        raise RuntimeError(
            f"the bench gate failed ({gate}): stop and report before the full replay"
        )
    if bench is None:
        say(f"WARNING: {gate['note']}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _clear_outputs(out_dir)
    commit()
    spec = load_engine_inputs(paths, cfgs, vocab=vocab, hubs=hubs, hub_cap=hub_cap)
    _write_inputs(out_dir, spec)
    commit()
    timings["load_inputs"] = time.perf_counter() - t0

    bounds = boundary_minutes(cfgs["data"], cfgs["features"])
    snap_paths = {b: (s, out_dir / SNAPSHOTS_DIR / snapshot_file(s)) for s, b in bounds.items()}
    t = time.perf_counter()
    eng = Engine.create(spec)
    writer = _PartWriter(out_dir / PARTS_DIR, spec, rg_rows)
    run = _Measured(
        eng,
        spec,
        writer=writer,
        measure=True,
        progress=out_dir / PROGRESS_FILE,
        snapshots=snap_paths,
        commit=commit,
        dataset=paths.dataset,
        say=say,
        label="full replay",
    )
    end = _replay(
        eng,
        _batches(paths.transactions, batch_rows, use_threads=use_threads),
        start_rank=0,
        writer=writer,
        boundary=run,
    )
    timings["replay"] = time.perf_counter() - t
    missing = [s for b, (s, _) in snap_paths.items() if s not in run.snapshots]
    if missing:
        raise RuntimeError(f"no events reached the snapshot boundaries of {missing}")
    del eng, run.eng
    gc.collect()

    t = time.perf_counter()
    restart = _restart_check(
        paths,
        out_dir,
        spec,
        list(run.snapshots.values()),
        final_digest=run.final_digest,
        batch_rows=batch_rows,
        rg_rows=rg_rows,
        use_threads=use_threads,
    )
    timings["restart_check"] = time.perf_counter() - t
    commit()
    say(f"restart check: {'pass' if restart.get('pass') else restart.get('skipped', 'n/a')}")

    # The parts' content digest: the stages that read the parts record it, and evaluate/export
    # require theirs to equal this one (the features key hashes configs only).
    t = time.perf_counter()
    hashes = part_sha256s(out_dir)
    for part in writer.parts:
        part["sha256"] = hashes[part["file"]]
    timings["digest"] = time.perf_counter() - t

    timings["total"] = time.perf_counter() - t0
    summary = _full_summary(spec, run, writer, end, restart, bench, cfgs, timings, out_dir)
    summary[FEATURES_DIGEST] = digest_of_parts(hashes)
    summary["bench_gate"] = gate
    summary["headline"]["bench_gate"] = gate["status"]
    report_path = paths.reports / ENGINE_BENCH_REPORT
    write_text_atomic(render_engine_bench_md(bench, summary), report_path)
    summary["outputs"]["engine_bench_md"] = str(report_path)
    write_json_atomic(summary, out_dir / SUMMARY_FILE)  # last: marks a finished build
    commit()
    return summary


def _full_summary(
    spec: EngineSpec,
    run: _Measured,
    writer: _PartWriter,
    end: dict[str, Any],
    restart: dict[str, Any],
    bench: dict[str, Any] | None,
    cfgs: dict[str, dict],
    timings: dict[str, float],
    out_dir: Path,
) -> dict[str, Any]:
    days = run.days
    rows = sum(d["events"] for d in days)
    score = sum(d["seconds"]["score"] for d in days)
    flush = sum(d["seconds"]["flush"] for d in days)
    write = sum(d["seconds"].get("write", 0.0) for d in days)
    per_split = {}
    all_trunc = np.zeros(len(TRUNC_COLUMNS), np.int64)
    sev_nz = np.zeros(len(SEVERITY_COLUMNS), np.int64)
    for split in SPLITS:
        st = writer.splits.get(split)
        if st is None:
            continue
        all_trunc += st.trunc
        sev_nz += st.sev_nonzero
        per_split[split] = {
            "rows": st.rows,
            "trunc_share": {
                t: (int(c) / st.rows if st.rows else None)
                for t, c in zip(TRUNC_COLUMNS, st.trunc, strict=True)
            },
            "max_inflow_c": st.max_inflow_c,
        }
    restored = [d["memory"].get("restored_mb") for d in days]
    restored = [r for r in restored if r is not None]
    max_restored = max(restored) if restored else None
    ratio = (bench or {}).get("live_restored_ratio")
    live_peak = ((bench or {}).get("pass_b") or {}).get("peak_mb")
    target = float(cfgs["features"].get("memory_target_mb", 1000))
    projected = None
    if ratio is not None and max_restored is not None:
        projected = max(max_restored * ratio, live_peak or 0.0)
    worst_s, worst_m, worst_n = run.worst
    trunc_all = {
        t: (int(c) / rows if rows else None) for t, c in zip(TRUNC_COLUMNS, all_trunc, strict=True)
    }
    flush_ms = _flush_ms(run.flushes)
    summary = {
        "mode": "full",
        "engine_version": ENGINE_VERSION,
        "spec_hash": spec.spec_hash(),
        "n_features": spec.n_features,
        "n_accounts": spec.n_accounts,
        "hub_cap": spec.hub_cap,
        "n_hubs": len(spec.hubs),
        "rows": rows,
        "rows_per_split": {s: v["rows"] for s, v in per_split.items()},
        "n_days": len(days),
        "last_minute": end["last_minute"],
        "us_per_event": {
            "engine": _us(score + flush, rows),
            "score": _us(score, rows),
            "flush": _us(flush, rows),
            "io": _us(write, rows),
            "per_day": [{"day": d["day"], **d["us_per_event"]} for d in days],
        },
        "flush_ms": {
            **flush_ms,
            "worst": {"ms": worst_s * 1e3, "minute": worst_m, "n_applied": worst_n},
        },
        "memory": {
            "per_day": [
                {
                    "day": d["day"],
                    **{
                        k: d["memory"].get(k)
                        for k in ("restored_mb", "state_mb", "n_pairs", "ring_rows_live")
                    },
                }
                for d in days
            ],
            "max_restored_mb": max_restored,
            "max_state_mb": max(d["memory"]["state_mb"] for d in days),
            "max_ru_maxrss_mb": max(
                (d["memory"]["ru_maxrss_mb"] for d in days if d["memory"]["ru_maxrss_mb"]),
                default=None,
            ),
            "live_restored_ratio": ratio,
            "bench_live_peak_mb": live_peak,
            "projected_mb": projected,
            "target_mb": target,
            "within_target": None if projected is None else projected < target,
        },
        "trunc_share": {
            "per_split": {s: v["trunc_share"] for s, v in per_split.items()},
            "all": trunc_all,
        },
        "severity_nonzero_share": {
            s: (int(c) / rows if rows else None)
            for s, c in zip(SEVERITY_COLUMNS, sev_nz, strict=True)
        },
        "max_inflow_c": max((v["max_inflow_c"] for v in per_split.values()), default=0),
        "snapshots": run.snapshots,
        "restart_check": restart,
        "days": days,  # the progress.jsonl records
        "final_state_digest": run.final_digest,
        "parts": writer.parts,
        "timings_s": timings,
        "outputs": {
            "features_dir": str(out_dir),
            "parts": str(out_dir / PARTS_DIR),
            "progress": str(out_dir / PROGRESS_FILE),
        },
    }
    # The key numbers in one flat block (reports render the summary's first scalar leaves).
    summary["headline"] = {
        "rows": rows,
        "us_per_event_engine": summary["us_per_event"]["engine"],
        "us_per_event_io": summary["us_per_event"]["io"],
        "flush_p99_ms": flush_ms["p99"],
        "flush_max_ms": flush_ms["max"],
        "max_restored_mb": max_restored,
        "projected_memory_mb": projected,
        "memory_within_target": summary["memory"]["within_target"],
        "restart_check_pass": restart.get("pass"),
        **{f"{t}_share": v for t, v in trunc_all.items()},
        "replay_min": timings.get("replay", 0.0) / 60,
    }
    return summary


# --- restart check ------------------------------------------------------------------------------


def _restart_check(
    paths: DataPaths,
    out_dir: Path,
    spec: EngineSpec,
    snaps: list[dict[str, Any]],
    *,
    final_digest: str | None,
    batch_rows: int,
    rg_rows: int,
    use_threads: bool,
) -> dict[str, Any]:
    """Restore the first boundary snapshot, re-replay to the next boundary (or the end) into
    restart_check/, and require bit-equal rows and the boundary's (or the final) state digest.

    Passing deletes restart_check/; failing keeps it (with result.json) and raises.
    """
    if not snaps:
        return {"skipped": "no boundary snapshots configured"}
    snaps = sorted(snaps, key=lambda s: s["minute"])
    start, stop = snaps[0], (snaps[1] if len(snaps) > 1 else None)
    rc_dir = out_dir / RESTART_DIR
    shutil.rmtree(rc_dir, ignore_errors=True)
    eng, header = Engine.restore(out_dir / SNAPSHOTS_DIR / start["file"], spec)
    if header.get("next_rank") != start["next_rank"]:
        raise EngineError(f"restored next_rank {header.get('next_rank')} != {start['next_rank']}")
    stop_day = stop["minute"] // MINUTES_PER_DAY + 1 if stop else None
    writer = _PartWriter(rc_dir / PARTS_DIR, spec, rg_rows)
    digest: dict[str, str] = {}

    def boundary(stats: _Day, b: int, next_rank: int, final: bool) -> None:
        if (stop is not None and b == stop["minute"]) or (stop is None and final):
            digest["at_stop"] = eng.state_digest()

    _replay(
        eng,
        _batches(
            paths.transactions, batch_rows, start_rank=start["next_rank"], use_threads=use_threads
        ),
        start_rank=start["next_rank"],
        writer=writer,
        boundary=boundary,
        stop_day=stop_day,
    )
    expected = stop["state_digest"] if stop is not None else final_digest
    days, mismatched, files_equal, rows = [], [], True, 0
    for part in writer.parts:
        a, b = out_dir / PARTS_DIR / part["file"], rc_dir / PARTS_DIR / part["file"]
        days.append(part["day"])
        rows += part["rows"]
        if not a.exists() or not tables_bit_equal(pq.read_table(a), pq.read_table(b)):
            mismatched.append(part["day"])
        files_equal = files_equal and a.exists() and _sha256_file(a) == _sha256_file(b)
    result = {
        "from": start["file"],
        "to": stop["file"] if stop is not None else "end",
        "days": days,
        "rows": rows,
        "rows_equal": not mismatched,
        "mismatched_days": mismatched,
        "files_sha256_equal": files_equal,
        "state_digest": digest.get("at_stop"),
        "expected_state_digest": expected,
        "digest_equal": digest.get("at_stop") is not None and digest.get("at_stop") == expected,
    }
    result["pass"] = bool(result["rows_equal"] and result["digest_equal"] and rows > 0)
    if not result["pass"]:
        write_json_atomic(result, rc_dir / RESTART_RESULT_FILE)
        raise RuntimeError(f"restart check failed: {result}")
    shutil.rmtree(rc_dir, ignore_errors=True)
    return result


def tables_bit_equal(a: pa.Table, b: pa.Table) -> bool:
    """Same schema and every value bit-equal (floats compared as raw bits, NaN included)."""
    if a.schema != b.schema or a.num_rows != b.num_rows:
        return False
    for name in a.column_names:
        x, y = a.column(name), b.column(name)
        if pa.types.is_floating(x.type):
            bits = np.uint32 if x.type == pa.float32() else np.uint64
            xa = x.to_numpy(zero_copy_only=False)
            ya = y.to_numpy(zero_copy_only=False)
            if not np.array_equal(xa.view(bits), ya.view(bits)):
                return False
        elif not x.equals(y):
            return False
    return True


# --- bench mode ---------------------------------------------------------------------------------


def sizing_queries(spec: EngineSpec) -> dict[str, str]:
    """DuckDB sizing SQL over view `tx` (M2 spec §8.1 bench): lifetime pairs, the largest live
    ring, in-chain lengths, windowed cents sums (vs 2^53) and the largest amount in cents.

    A window [m - W, m - 1] at clock m = t + 1 is `RANGE BETWEEN W - 1 PRECEDING AND CURRENT ROW`
    at an event minute t, so the maxima over event rows are the maxima over all clocks.
    """
    usd_c = "CAST(round(amount_usd * 100) AS BIGINT)"
    wmax, wrt, wpt, wl = spec.W_max - 1, spec.w_rt - 1, spec.w_pt - 1, spec.w_long - 1

    def win(part: str, w: int) -> str:
        return f"PARTITION BY {part} ORDER BY minute RANGE BETWEEN {w} PRECEDING AND CURRENT ROW"

    return {
        "pairs": """
            SELECT count(*) AS lifetime_pairs,
                   count(*) FILTER (WHERE src <> dst) AS lifetime_pairs_nsl
            FROM (SELECT DISTINCT src, dst FROM tx)""",
        "ring": f"""
            WITH c AS (SELECT minute, count(*) AS n FROM tx GROUP BY minute)
            SELECT max(s) AS max_ring_rows FROM (
                SELECT sum(n) OVER (ORDER BY minute RANGE BETWEEN {wmax} PRECEDING
                                    AND CURRENT ROW) AS s FROM c)""",
        "in_degree": """
            SELECT max(n) AS max_in_degree, max(n_nsl) AS max_in_degree_nsl FROM (
                SELECT dst, count(*) AS n, count(*) FILTER (WHERE src <> dst) AS n_nsl
                FROM tx GROUP BY dst)""",
        "in_window": f"""
            SELECT max(c) AS max_in_edges_rt FROM (
                SELECT count(*) OVER ({win("dst", wrt)}) AS c FROM tx)""",
        "in_window_nsl": f"""
            SELECT max(c) AS max_in_edges_rt_nsl FROM (
                SELECT count(*) OVER ({win("dst", wrt)}) AS c FROM tx WHERE src <> dst)""",
        "inflow": f"""
            SELECT max(s) AS max_inflow_c_pt FROM (
                SELECT sum({usd_c}) OVER ({win("dst", wpt)}) AS s FROM tx WHERE src <> dst)""",
        "out_sum": f"""
            SELECT max(s) AS max_out_sum_c_long FROM (
                SELECT sum({usd_c}) OVER ({win("src", wl)}) AS s FROM tx)""",
        "in_sum": f"""
            SELECT max(s) AS max_in_sum_c_long FROM (
                SELECT sum({usd_c}) OVER ({win("dst", wl)}) AS s FROM tx)""",
        "usd": f"SELECT max({usd_c}) AS max_usd_c, count(*) AS rows FROM tx",
    }


def run_sizing(paths: DataPaths, spec: EngineSpec) -> dict[str, Any]:
    """The sizing numbers on DuckDB with 1 thread (plus each query's seconds)."""
    out: dict[str, Any] = {}
    seconds: dict[str, float] = {}
    with _duckdb(1) as con:
        sql_baseline.register_transactions(con, paths.transactions)
        for name, sql in sizing_queries(spec).items():
            t = time.perf_counter()
            cur = con.execute(sql)
            cols = [d[0] for d in cur.description]
            row = cur.fetchone()
            out.update({c: (None if v is None else int(v)) for c, v in zip(cols, row, strict=True)})
            seconds[name] = time.perf_counter() - t
    sums = ("max_inflow_c_pt", "max_out_sum_c_long", "max_in_sum_c_long", "max_usd_c")
    out["sums_below_2p53"] = all((out.get(k) or 0) < EXACT_INT_LIMIT for k in sums)
    out["seconds"] = seconds
    return out


def _count_rows(paths: DataPaths, lo_day: int, hi_day: int | None) -> int:
    """Rows with lo_day <= day < hi_day (hi_day None: to the end)."""
    expr = pl.col("day") >= lo_day
    if hi_day is not None:
        expr = expr & (pl.col("day") < hi_day)
    return int(pl.scan_parquet(paths.transactions).filter(expr).select(pl.len()).collect().item())


def _run_bench(
    paths: DataPaths,
    out_dir: Path,
    cfgs: dict[str, dict],
    *,
    vocab: dict[str, list[str]] | None,
    hubs: list[int] | None,
    hub_cap: int | None,
    batch_rows: int,
    rg_rows: int,
    use_threads: bool,
    commit: Callable[[], None],
    say: Callable[[str], None],
) -> dict[str, Any]:
    t0 = time.perf_counter()
    fcfg = cfgs["features"]
    last_day = int(fcfg["bench"]["last_day"])
    if last_day < 1:
        raise ValueError(f"bench.last_day must be >= 1, got {last_day}")
    timings: dict[str, float] = {}
    spec = load_engine_inputs(paths, cfgs, vocab=vocab, hubs=hubs, hub_cap=hub_cap)
    timings["load_inputs"] = time.perf_counter() - t0
    t = time.perf_counter()
    sizing = run_sizing(paths, spec)
    timings["sizing"] = time.perf_counter() - t
    say(f"bench: sizing SQL done in {timings['sizing'] / 60:,.1f} min")

    # Pass A: timing (with the Parquet writes into a scratch directory) + restore-measures.
    t = time.perf_counter()
    scratch = Path(tempfile.mkdtemp(prefix="aml-bench-parts-"))
    try:
        eng = Engine.create(spec)
        writer = _PartWriter(scratch, spec, rg_rows)
        pass_a = _Measured(
            eng,
            spec,
            writer=writer,
            measure=True,
            progress=None,
            snapshots=None,
            commit=lambda: None,
            dataset=paths.dataset,
            say=say,
            label="bench pass A",
        )
        _replay(
            eng,
            _batches(paths.transactions, batch_rows, use_threads=use_threads),
            start_rank=0,
            writer=writer,
            boundary=pass_a,
            stop_day=last_day + 1,
        )
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    del eng, pass_a.eng
    gc.collect()
    timings["pass_a"] = time.perf_counter() - t

    # Saved before the slow traced pass: if pass B hits the job timeout, the sizing, the timing
    # and the replay projection survive (bench.json, the gate, is written only after pass B).
    proj = _replay_projection(paths, cfgs, pass_a)
    pass_a_doc = _bench_pass_a_doc(spec, sizing, pass_a, proj, last_day)
    write_json_atomic(pass_a_doc, out_dir / BENCH_DIR / BENCH_PASS_A_FILE)
    commit()
    say(
        f"bench pass A: {_f(proj['us_engine'])} µs/event engine + {_f(proj['us_io'])} I/O; "
        f"projected full replay {_f(proj['replay_min'])} min (gate <= {BENCH_MAX_REPLAY_MIN:g}); "
        f"saved {BENCH_DIR}/{BENCH_PASS_A_FILE}; pass B (tracemalloc, several times slower) next"
    )

    # Pass B: the same events with tracemalloc on from before Engine.create; no output.
    t = time.perf_counter()
    traced = _Traced(say)
    tracemalloc.start()
    try:
        eng = Engine.create(spec)
        _replay(
            eng,
            _batches(paths.transactions, batch_rows, use_threads=use_threads),
            start_rank=0,
            writer=None,
            boundary=traced,
            stop_day=last_day + 1,
        )
        del eng
    finally:
        tracemalloc.stop()
    gc.collect()
    timings["pass_b"] = time.perf_counter() - t

    bench = _bench_summary(cfgs, spec, sizing, pass_a, traced, last_day, proj)
    timings["total"] = time.perf_counter() - t0
    bench["timings_s"] = timings
    write_json_atomic(bench, out_dir / BENCH_DIR / BENCH_FILE)
    summary_path = out_dir / SUMMARY_FILE
    full = read_json(summary_path) if summary_path.exists() else None
    report_path = paths.reports / ENGINE_BENCH_REPORT
    write_text_atomic(render_engine_bench_md(bench, full), report_path)
    bench["outputs"] = {
        "bench_json": str(out_dir / BENCH_DIR / BENCH_FILE),
        "engine_bench_md": str(report_path),
    }
    commit()
    return bench


def _replay_projection(
    paths: DataPaths, cfgs: dict[str, dict], pass_a: _Measured
) -> dict[str, Any]:
    """The full replay's projected time from pass A's per-event cost (M2 spec §8.1)."""
    a_days = pass_a.days
    rows = sum(d["events"] for d in a_days)
    eng_s = sum(d["seconds"]["score"] + d["seconds"]["flush"] for d in a_days)
    io_s = sum(d["seconds"].get("write", 0.0) for d in a_days)
    meas = [d["seconds"]["measure"] for d in a_days if "measure" in d["seconds"]]
    rows_total = pq.ParquetFile(paths.transactions).metadata.num_rows
    n_days = int(pl.scan_parquet(paths.transactions).select(pl.col("day").max()).collect().item())
    bounds = sorted(boundary_minutes(cfgs["data"], cfgs["features"]).values())
    if bounds:
        lo = bounds[0] // MINUTES_PER_DAY + 1
        hi = bounds[1] // MINUTES_PER_DAY + 1 if len(bounds) > 1 else None
        rows_restart = _count_rows(paths, lo, hi)
    else:
        rows_restart = 0
    us_engine = _us(eng_s, rows) or 0.0
    us_io = _us(io_s, rows) or 0.0
    measure_s = float(np.mean(meas)) if meas else 0.0
    replay_s = (rows_total + rows_restart) * (us_engine + us_io) / 1e6 + (n_days + 2) * measure_s
    return {
        "rows": rows,
        "us_engine": us_engine,
        "us_io": us_io,
        "measure_s": measure_s,
        "rows_total": rows_total,
        "rows_restart": rows_restart,
        "n_days": n_days,
        "replay_s": replay_s,
        "replay_min": replay_s / 60,
    }


def _bench_pass_a_doc(
    spec: EngineSpec,
    sizing: dict[str, Any],
    pass_a: _Measured,
    proj: dict[str, Any],
    last_day: int,
) -> dict[str, Any]:
    """bench/bench_pass_a.json: what the bench knows before its traced pass (never a gate)."""
    return {
        "mode": "bench_pass_a",
        "note": "written after pass A, before the traced pass B; the gate is bench/bench.json",
        "engine_version": ENGINE_VERSION,
        "spec_hash": spec.spec_hash(),
        "last_day": last_day,
        "sizing": sizing,
        "pass_a": {
            "rows": proj["rows"],
            "us_per_event": {"engine": proj["us_engine"], "io": proj["us_io"]},
            "restore_measure_s": proj["measure_s"],
            "days": pass_a.days,
        },
        "projection": {
            "rows_total": proj["rows_total"],
            "rows_restart_check": proj["rows_restart"],
            "n_days": proj["n_days"],
            "replay_s": proj["replay_s"],
            "replay_min": proj["replay_min"],
        },
        "replay_max_min": BENCH_MAX_REPLAY_MIN,
        "replay_ok": proj["replay_min"] <= BENCH_MAX_REPLAY_MIN,
    }


def _bench_summary(
    cfgs: dict[str, dict],
    spec: EngineSpec,
    sizing: dict[str, Any],
    pass_a: _Measured,
    traced: _Traced,
    last_day: int,
    proj: dict[str, Any],
) -> dict[str, Any]:
    a_days, b_days = pass_a.days, traced.days
    rows = proj["rows"]
    last_a, last_b = a_days[-1], b_days[-1]
    restored_last = last_a["memory"]["restored_mb"]
    ratio = last_b["traced_current_mb"] / restored_last if restored_last else None
    live_peak = max(d["traced_peak_mb"] for d in b_days)

    rows_total, rows_restart, n_days = proj["rows_total"], proj["rows_restart"], proj["n_days"]
    us_engine, us_io, measure_s = proj["us_engine"], proj["us_io"], proj["measure_s"]
    replay_s = proj["replay_s"]

    # Memory: the larger of the bench's live peak and the end-of-replay state, projected from the
    # day-3 restored state plus the lifetime pairs still to come (the ring is at its peak here).
    nb = last_a["memory"]["state_nbytes"]
    pairs_now = last_a["memory"].get("n_pairs") or 0
    pair_bytes = _pair_bytes(nb)
    per_pair = pair_bytes / pairs_now if pair_bytes and pairs_now else PLANNING_PAIR_BYTES
    pairs_end = sizing.get("lifetime_pairs") or pairs_now
    end_restored = restored_last + max(0, pairs_end - pairs_now) * per_pair / MB
    projected_end = end_restored * ratio if ratio is not None else None
    projected_mb = max(live_peak, projected_end or 0.0)
    target = float(cfgs["features"].get("memory_target_mb", 1000))
    replay_min = replay_s / 60
    gate = {
        "replay_max_min": BENCH_MAX_REPLAY_MIN,
        "memory_target_mb": target,
        "replay_ok": replay_min <= BENCH_MAX_REPLAY_MIN,
        "memory_ok": projected_mb < target,
    }
    gate["pass"] = gate["replay_ok"] and gate["memory_ok"]
    return {
        "mode": "bench",
        "engine_version": ENGINE_VERSION,
        "spec_hash": spec.spec_hash(),
        "n_accounts": spec.n_accounts,
        "hub_cap": spec.hub_cap,
        "n_hubs": len(spec.hubs),
        "last_day": last_day,
        "advance_to": last_day * MINUTES_PER_DAY,
        "sizing": sizing,
        "pass_a": {
            "rows": rows,
            "us_per_event": {"engine": us_engine, "io": us_io},
            "flush_ms": {
                **_flush_ms(pass_a.flushes),
                "worst": {
                    "ms": pass_a.worst[0] * 1e3,
                    "minute": pass_a.worst[1],
                    "n_applied": pass_a.worst[2],
                },
            },
            "restore_measure_s": measure_s,
            "days": a_days,
        },
        "pass_b": {"peak_mb": live_peak, "days": b_days},
        "live_restored_ratio": ratio,
        "projection": {
            "rows_total": rows_total,
            "rows_restart_check": rows_restart,
            "n_days": n_days,
            "replay_s": replay_s,
            "replay_min": replay_min,
            "pair_bytes": per_pair,
            "pairs_now": pairs_now,
            "pairs_end": pairs_end,
            "end_restored_mb": end_restored,
            "end_projected_mb": projected_end,
            "memory_mb": projected_mb,
            "note": "planning: days 1..last_day per-event cost; end state = day-3 state + the "
            "remaining lifetime pairs (the ring peaks at the bench boundary); x live/restored",
        },
        "gate": gate,
    }


# --- verify mode --------------------------------------------------------------------------------


def _mismatches(doc: Any) -> int:
    """Total mismatch count of an oracle document: `n_mismatches` if given, else the sum of every
    integer under a key named `mismatches` (any nesting)."""
    if isinstance(doc, dict) and isinstance(doc.get("n_mismatches"), int):
        return int(doc["n_mismatches"])
    total = 0
    stack = [doc]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            for k, v in x.items():
                if k == "mismatches" and isinstance(v, int) and not isinstance(v, bool):
                    total += v
                else:
                    stack.append(v)
        elif isinstance(x, list):
            stack.extend(x)
    return total


def _run_verify(
    paths: DataPaths, out_dir: Path, threads: int | None, commit: Callable[[], None]
) -> dict[str, Any]:
    from aml.features.oracle import run_oracle

    if not (out_dir / SUMMARY_FILE).exists():
        raise FileNotFoundError(f"{out_dir} has no finished replay: run --mode full first")
    spec = load_spec(out_dir)
    t = time.perf_counter()
    doc = dict(run_oracle(paths, out_dir, spec, threads=threads or 4))
    seconds = time.perf_counter() - t
    n_bad = _mismatches(doc)
    doc.setdefault("seconds", seconds)
    doc["n_mismatches_total"] = n_bad
    doc["spec_hash"] = spec.spec_hash()
    doc[FEATURES_DIGEST] = parts_digest(out_dir)  # the parts this verdict is about
    path = out_dir / VERIFY_DIR / VERIFY_FILE
    write_json_atomic(doc, path)
    commit()
    if n_bad:
        raise RuntimeError(f"feature oracle: {n_bad} mismatches (see {path})")
    return {"mode": "verify", "n_mismatches": 0, "seconds": seconds, "verify_json": str(path)}


# --- entry point --------------------------------------------------------------------------------


def run_build_features(
    paths: DataPaths,
    out_dir: Path,
    cfgs: dict[str, dict],
    *,
    mode: str = "full",
    vocab: dict[str, list[str]] | None = None,
    hubs: list[int] | None = None,
    hub_cap: int | None = None,
    batch_rows: int | None = None,
    on_checkpoint: Callable[[], None] | None = None,
    threads: int | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Run one mode of the feature build (§8.1) into `out_dir` and return its summary.

    cfgs: the loaded configs ("data", "rules", "features"). full: the replay, parts, boundary
    snapshots, a restore-measure at every day boundary and the restart check. bench: sizing SQL +
    a timing pass and a traced pass over days 1..bench.last_day. verify: the DuckDB oracle
    (`features.oracle.run_oracle`). `on_checkpoint` runs after every durable write (vol.commit).
    `threads`: Parquet decoding threads (1 / None = single-threaded, as the 1-core replay
    container) and the oracle's DuckDB threads in verify mode (default 4). `log` receives one
    progress line per replayed day (and a few more), e.g. the Modal job's print.
    """
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}; modes are {MODES}")
    out_dir = Path(out_dir)
    commit = on_checkpoint or (lambda: None)
    say = log or (lambda _msg: None)
    if mode == "verify":
        return _run_verify(paths, out_dir, threads, commit)
    fcfg = cfgs["features"]
    batch = int(batch_rows if batch_rows is not None else fcfg["replay"]["batch_rows"])
    rg_rows = int(fcfg["replay"]["row_group_rows"])
    if batch < 1 or rg_rows < 1:
        raise ValueError(f"batch_rows and row_group_rows must be >= 1, got {batch}, {rg_rows}")
    kw = {
        "vocab": vocab,
        "hubs": hubs,
        "hub_cap": hub_cap,
        "batch_rows": batch,
        "rg_rows": rg_rows,
        "use_threads": bool(threads and threads > 1),
        "commit": commit,
        "say": say,
    }
    if mode == "bench":
        return _run_bench(paths, out_dir, cfgs, **kw)
    return _run_full(paths, out_dir, cfgs, **kw)


# --- report -------------------------------------------------------------------------------------


def _f(v: Any, digits: int = 1) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return f"{v:,}"
    return f"{v:,.{digits}f}"


def render_engine_bench_md(bench: dict[str, Any] | None, full: dict[str, Any] | None) -> str:
    """reports/engine_bench.md: one table for the bench, one for the full replay."""
    lines = [
        "# Feature engine benchmark",
        "",
        "Measured by `build_features` (M2 spec §8.1); these numbers replace the planning "
        "estimates. MB = 10^6 bytes. Engine µs/event = scoring + minute flushes per event; I/O = "
        "row conversion, validation and Parquet writes. Restored MB = tracemalloc bytes of an "
        "engine restored from a raw snapshot (the state alone).",
        "",
        "## Bench",
        "",
    ]
    if bench is None:
        lines += ["Not run yet (`make features-bench`).", ""]
    else:
        lines += [
            f"Days 1-{bench['last_day']}, then advance to minute {bench['advance_to']} "
            "(the ring peak). Pass A untraced, pass B with tracemalloc on from before "
            "Engine.create.",
            "",
            "| Day | Events | Engine µs/event | I/O µs/event | Restored MB | State MB "
            "| Traced current MB | Traced peak MB | Pairs | Live ring rows |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        traced = {d["day"]: d for d in bench["pass_b"]["days"]}
        for d in bench["pass_a"]["days"]:
            m, tb = d["memory"], traced.get(d["day"], {})
            lines.append(
                f"| {d['day']} | {_f(d['events'])} | {_f(d['us_per_event']['engine'])} "
                f"| {_f(d['us_per_event'].get('io'))} | {_f(m.get('restored_mb'))} "
                f"| {_f(m.get('state_mb'))} | {_f(tb.get('traced_current_mb'))} "
                f"| {_f(tb.get('traced_peak_mb'))} | {_f(m.get('n_pairs'))} "
                f"| {_f(m.get('ring_rows_live'))} |"
            )
        s, p, g = bench["sizing"], bench["projection"], bench["gate"]
        lines += [
            "",
            f"- Sizing (DuckDB, all rows): lifetime pairs {_f(s.get('lifetime_pairs'))} "
            f"(non-self-loop {_f(s.get('lifetime_pairs_nsl'))}); largest live ring "
            f"{_f(s.get('max_ring_rows'))} rows; max in-degree {_f(s.get('max_in_degree'))}; "
            f"max in-edges per account in a round-trip window {_f(s.get('max_in_edges_rt'))}; "
            f"windowed cent sums below 2^53: {_f(s.get('sums_below_2p53'))}.",
            f"- Live/restored ratio at the bench boundary: {_f(bench['live_restored_ratio'], 2)}; "
            f"live traced peak {_f(bench['pass_b']['peak_mb'])} MB.",
            f"- Projected full replay {_f(p['replay_min'])} min (gate <= {_f(g['replay_max_min'])})"
            f"; projected memory {_f(p['memory_mb'])} MB (target {_f(g['memory_target_mb'])}). "
            f"Gate: **{'pass' if g['pass'] else 'FAIL'}**.",
            "",
        ]
    lines += ["## Full replay", ""]
    if full is None:
        lines += ["Not run yet (`make features`).", ""]
        return "\n".join(lines)
    lines += [
        "| Day | Events | Engine µs/event | Score µs | Flush µs | I/O µs | Flush p99 ms "
        "| Flush max ms | Restored MB | State MB | Pairs | Live ring rows |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for d in full.get("days") or []:
        u, m, fl = d["us_per_event"], d["memory"], d["flush_ms"]
        lines.append(
            f"| {d['day']} | {_f(d['events'])} | {_f(u['engine'])} | {_f(u['score'])} "
            f"| {_f(u['flush'])} | {_f(u.get('io'))} | {_f(fl['p99'], 2)} | {_f(fl['max'], 2)} "
            f"| {_f(m.get('restored_mb'))} | {_f(m.get('state_mb'))} | {_f(m.get('n_pairs'))} "
            f"| {_f(m.get('ring_rows_live'))} |"
        )
    u, mem, rc = full["us_per_event"], full["memory"], full["restart_check"]
    fl = full["flush_ms"]
    gate = full.get("bench_gate") or {}
    if gate.get("status") == "absent":
        gate_txt = "absent: no bench for this features key, the replay ran ungated"
    elif gate:
        gate_txt = f"{gate.get('status')} (re-checked before the replay)"
    else:
        gate_txt = "n/a"
    lines += [
        "",
        f"- {_f(full['rows'])} events: engine {_f(u['engine'])} µs/event (score {_f(u['score'])}"
        f", flush {_f(u['flush'])}), I/O {_f(u['io'])} µs/event; minute flush p50 "
        f"{_f(fl['p50'], 2)} ms, p99 {_f(fl['p99'], 2)} ms, max {_f(fl['worst']['ms'], 2)} ms "
        f"(minute {fl['worst']['minute']}, {_f(fl['worst']['n_applied'])} events applied).",
        f"- Memory: max restored {_f(mem['max_restored_mb'])} MB; projected "
        f"{_f(mem['projected_mb'])} MB with the bench ratio (target {_f(mem['target_mb'])}); "
        f"max ru_maxrss {_f(mem['max_ru_maxrss_mb'])} MB.",
        f"- Bench gate: {gate_txt}.",
        f"- Restart check ({rc.get('from', 'n/a')} -> {rc.get('to', 'n/a')}): "
        f"{'pass' if rc.get('pass') else rc.get('skipped', 'FAIL')}; rows bit-equal "
        f"{_f(rc.get('rows_equal'))}, file sha256 equal {_f(rc.get('files_sha256_equal'))}, "
        f"state digest equal {_f(rc.get('digest_equal'))}.",
        "",
    ]
    return "\n".join(lines)
