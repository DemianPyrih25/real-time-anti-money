"""Engine input frames, specs and a plain driver for the M2 verification suites.

M1's hand-made frames (`rules_frames`) lack the currency and bank columns the engine reads; this
module adds them, builds `EngineSpec`s for frames and for the prepared fixture (vocab, hub cap and
hubs fitted exactly as the offline driver fits them), and feeds a frame through `Engine` in rank
order. Nothing here imports the engine at module import time, so suites that do not need it (the
reference, the oracle) collect and run before the engine exists.
"""

from __future__ import annotations

import copy
import struct
from collections import Counter
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import yaml

from aml.features.spec import INPUT_COLUMNS, EngineSpec
from aml.features.tx_features import CATEGORICAL_FEATURES, fit_vocab
from aml.paths import DataPaths
from aml.rules.sql_baseline import TX_COLUMNS, connect, hub_degree_cap, register_transactions
from tests.fixtures.rules_frames import dense_tie_frame, make_tx

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
# Every column a frame needs to drive the engine (INPUT_COLUMNS) and the SQL rules (TX_COLUMNS).
ENGINE_TX_COLUMNS = tuple(dict.fromkeys((*INPUT_COLUMNS, "day", "split", *TX_COLUMNS)))
CURRENCIES = ("US Dollar", "Euro", "Yen")
BANKS = ("001", "002", "1")
FORMATS = ("ACH", "Bitcoin", "Cash", "Cheque", "Reinvestment", "Wire")
NO_HUBS = 10**9


def load_cfg(name: str) -> dict:
    with (CONFIG_DIR / f"{name}.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def features_cfg(
    *,
    short: int | None = None,
    long: int | None = None,
    sg: int | None = None,
    count: int | None = None,
    port: int | None = None,
    gap: int | None = None,
    rule_visits: int | None = None,
    feat_visits: int | None = None,
    compact_min_rows: int | None = None,
) -> dict:
    """configs/features.yaml with the given engine settings replaced."""
    cfg = copy.deepcopy(load_cfg("features"))
    for section, key, value in (
        ("windows", "short", short),
        ("windows", "long", long),
        ("windows", "sg", sg),
        ("caps", "count", count),
        ("caps", "port", port),
        ("caps", "gap", gap),
        ("budgets", "rule_visits", rule_visits),
        ("budgets", "feat_visits", feat_visits),
        ("ring", "compact_min_rows", compact_min_rows),
    ):
        if value is not None:
            cfg[section][key] = int(value)
    return cfg


def rules_windows(
    rules_cfg: dict,
    *,
    fan_in: int,
    fan_out: int,
    pass_through: int,
    round_trip: int,
    hop: int,
    structuring: int,
    round_burst: int,
    high_risk: int,
    excl: dict[str, bool] | None = None,
) -> dict:
    """rules_cfg with one window per scenario (and optionally the hub-sender flags) replaced."""
    cfg = copy.deepcopy(rules_cfg)
    sc = cfg["scenarios"]
    for name, w in (
        ("fan_in_velocity", fan_in),
        ("fan_out_velocity", fan_out),
        ("rapid_pass_through", pass_through),
        ("round_trip", round_trip),
        ("structuring", structuring),
        ("round_amount_burst", round_burst),
        ("high_risk_format_burst", high_risk),
    ):
        sc[name]["window_minutes"] = int(w)
    sc["round_trip"]["hop_window_minutes"] = int(hop)
    for name, on in (excl or {}).items():
        sc[name]["exclude_hub_senders"] = bool(on)
    return cfg


def with_hub_flags(rules_cfg: dict, on: Iterable[str]) -> dict:
    """rules_cfg with exclude_hub_senders true exactly for the scenarios in `on`."""
    from aml.rules.sql_baseline import HUB_SEGMENTABLE

    on = set(on)
    cfg = copy.deepcopy(rules_cfg)
    for name in HUB_SEGMENTABLE:
        cfg["scenarios"][name]["exclude_hub_senders"] = name in on
    return cfg


# ---------------------------------------------------------------------------------------------
# Frames


def with_engine_columns(
    df: pl.DataFrame,
    *,
    seed: int = 0,
    cross_p: float = 0.3,
    same_bank_p: float = 0.4,
    currencies: Sequence[str] = CURRENCIES,
    banks: Sequence[str] = BANKS,
) -> pl.DataFrame:
    """Add payment/receiving currency and from/to bank columns where missing (deterministic)."""
    rng = np.random.default_rng(seed)
    n = df.height
    pay = rng.choice(np.array(currencies), n)
    recv = np.where(rng.random(n) < cross_p, rng.choice(np.array(currencies), n), pay)
    fb = rng.choice(np.array(banks), n)
    tb = np.where(rng.random(n) < same_bank_p, fb, rng.choice(np.array(banks), n))
    add = {
        "payment_currency": pay,
        "receiving_currency": recv,
        "from_bank": fb,
        "to_bank": tb,
    }
    cols = [
        pl.Series(k, v.tolist(), dtype=pl.String) for k, v in add.items() if k not in df.columns
    ]
    out = df.with_columns(cols) if cols else df
    return out.select(ENGINE_TX_COLUMNS)


def engine_tx(rows: list[dict]) -> pl.DataFrame:
    """M1 `make_tx` plus the engine columns (US Dollar both sides, bank "001" both sides unless a
    row sets payment_currency / receiving_currency / from_bank / to_bank).

    Row k gets row_id k; rank follows (minute, row_id) as in M1.
    """
    df = make_tx(rows)
    extra = pl.DataFrame(
        {
            "row_id": list(range(len(rows))),
            "payment_currency": [r.get("payment_currency", "US Dollar") for r in rows],
            "receiving_currency": [
                r.get("receiving_currency", r.get("payment_currency", "US Dollar")) for r in rows
            ],
            "from_bank": [r.get("from_bank", "001") for r in rows],
            "to_bank": [r.get("to_bank", r.get("from_bank", "001")) for r in rows],
        },
        schema_overrides={"row_id": pl.Int64},
    )
    df = df.join(extra, on="row_id", how="left", maintain_order="left")
    return df.select(ENGINE_TX_COLUMNS)


def dense_engine_frame(
    seed: int = 7, n: int = 300, accounts: int = 6, minutes: int = 40
) -> pl.DataFrame:
    """M1's tie-heavy `dense_tie_frame` with currency and bank columns (some cross, some same)."""
    df = dense_tie_frame(seed=seed, n=n, accounts=accounts, minutes=minutes)
    return with_engine_columns(df, seed=seed + 1000)


def check_frame(frame: pl.DataFrame) -> None:
    """A frame the engine can consume: ranks 0..N-1 in row order and minutes non-decreasing."""
    ranks = frame["rank"].to_list()
    if ranks != list(range(len(ranks))):
        raise ValueError("frame ranks must be 0..N-1 in row order")
    if frame.height and (frame["minute"].diff().drop_nulls() < 0).any():
        raise ValueError("frame minutes must be non-decreasing in rank order")


# ---------------------------------------------------------------------------------------------
# Specs


def train_hubs(frame: pl.DataFrame, hub_cap: int) -> list[int]:
    """The r_hubs definition of scenarios.sql, in Python: train in + out edge count > hub_cap
    (a self-loop counts once as out and once as in)."""
    train = frame.filter(pl.col("split") == "train")
    deg = Counter(train["src"].to_list() + train["dst"].to_list())
    return sorted(int(a) for a, d in deg.items() if d > hub_cap)


def train_vocab(frame: pl.DataFrame) -> dict[str, list[str]]:
    """M1 `fit_vocab` on the train rows (empty lists when there are none)."""
    train = frame.filter(pl.col("split") == "train")
    if train.height == 0:
        return {c: [] for c in CATEGORICAL_FEATURES}
    return fit_vocab(train)


def make_spec(
    frame: pl.DataFrame,
    rules_cfg: dict,
    feats_cfg: dict | None = None,
    *,
    hub_cap: int = NO_HUBS,
    n_accounts: int | None = None,
    vocab: dict | None = None,
    hubs: Sequence[int] | None = None,
) -> EngineSpec:
    """EngineSpec for a frame: vocab and hubs fitted on its train rows unless given."""
    if n_accounts is None:
        n_accounts = int(max(frame["src"].max(), frame["dst"].max())) + 1 if frame.height else 1
    return EngineSpec.from_configs(
        feats_cfg if feats_cfg is not None else features_cfg(),
        rules_cfg,
        n_accounts=n_accounts,
        vocab=train_vocab(frame) if vocab is None else vocab,
        hub_cap=int(hub_cap),
        hubs=train_hubs(frame, hub_cap) if hubs is None else list(hubs),
    )


def read_fixture(paths: DataPaths) -> pl.DataFrame:
    """The prepared fixture's transactions, engine + rules columns, rank order."""
    tx = pl.read_parquet(paths.transactions)
    check_frame(tx)
    return tx.select(ENGINE_TX_COLUMNS)


def fitted_hub_cap(frame_or_path: pl.DataFrame | Path, quantile: float) -> int:
    """M1's DuckDB hub_degree_cap on a frame or a transactions Parquet file."""
    con = connect(threads=1)
    try:
        register_transactions(con, frame_or_path)
        return hub_degree_cap(con, quantile)
    finally:
        con.close()


def fixture_spec(
    paths: DataPaths,
    rules_cfg: dict,
    feats_cfg: dict | None = None,
    *,
    hub_cap: int | None = None,
) -> tuple[EngineSpec, pl.DataFrame]:
    """(spec, transactions) of the prepared fixture: n_accounts = accounts table height, vocab
    from train rows, the hub cap fitted by M1 code unless given, hubs by the r_hubs definition."""
    tx = read_fixture(paths)
    n_accounts = pl.read_parquet(paths.accounts).height
    cap = fitted_hub_cap(paths.transactions, rules_cfg["hub_degree_quantile"])
    cap = cap if hub_cap is None else int(hub_cap)
    spec = make_spec(tx, rules_cfg, feats_cfg, hub_cap=cap, n_accounts=n_accounts)
    return spec, tx


# ---------------------------------------------------------------------------------------------
# Driving the engine


def _is_stub(fn: Any) -> bool:
    """True for a contract stub whose whole body is `raise NotImplementedError`."""
    code = getattr(getattr(fn, "__func__", fn), "__code__", None)
    return code is not None and code.co_names == ("NotImplementedError",)


def engine_missing() -> str | None:
    """Why the engine cannot run yet (the entry points still stubs), or None when it can."""
    from aml.features import cycles, engine
    from aml.rules import scenarios

    eng = engine.Engine
    entry = {
        "Engine.create": eng.create,
        "Engine.restore": eng.restore,
        "Engine.prepare": eng.prepare,
        "Engine.process": eng.process,
        "Engine.score": eng.score,
        "Engine.advance": eng.advance,
        "Engine.snapshot": eng.snapshot,
        "Engine.state_digest": eng.state_digest,
        "cycles.path_counts": cycles.path_counts,
        "cycles.sg_counts": cycles.sg_counts,
        "scenarios.severities": scenarios.severities,
    }
    stubs = [name for name, fn in entry.items() if _is_stub(fn)]
    return f"awaiting the engine (agents B and C): stubs {stubs}" if stubs else None


def build_missing() -> str | None:
    """Like engine_missing, plus the offline driver `run_build_features` (agent E)."""
    from aml.features import build

    reason = engine_missing()
    if _is_stub(build.run_build_features):
        return (reason + "; " if reason else "") + "awaiting run_build_features (agent E)"
    return reason


def require_engine() -> None:
    import pytest

    reason = engine_missing()
    if reason:
        pytest.skip(reason)


def require_build() -> None:
    import pytest

    reason = build_missing()
    if reason:
        pytest.skip(reason)


def frame_fields(frame: pl.DataFrame) -> list[tuple]:
    """The engine input tuples (INPUT_COLUMNS order, plain Python values) of a frame."""
    return list(zip(*(frame.get_column(c).to_list() for c in INPUT_COLUMNS), strict=True))


def run_engine(
    frame: pl.DataFrame,
    spec: EngineSpec,
    *,
    until_minute: int | None = None,
    engine: Any = None,
    check_every: int | None = None,
) -> list[tuple]:
    """Feed a rank-ordered frame through `Engine.process` and return the rows in frame order.

    Stops before the first event later than `until_minute`. `check_every` runs
    `check_invariants()` after every that-many events (and at the end).
    """
    from aml.features.engine import Engine

    check_frame(frame)
    eng = Engine.create(spec) if engine is None else engine
    process, prepare = eng.process, eng.prepare
    rows: list[tuple] = []
    for k, fields in enumerate(frame_fields(frame)):
        if until_minute is not None and fields[2] > until_minute:
            break
        rows.append(process(prepare(*fields)))
        if check_every and (k + 1) % check_every == 0:
            eng.check_invariants()
    if check_every:
        eng.check_invariants()
    return rows


def run_with_day_snapshots(frame: pl.DataFrame, spec: EngineSpec) -> tuple[list[tuple], dict]:
    """Like run_engine, but at the start of every simulated day (minute (d - 1) * 1440, before
    its first event) advance the clock there and take a raw snapshot.

    Returns (rows, {day start minute: snapshot bytes}).
    """
    from aml.features.engine import Engine

    check_frame(frame)
    eng = Engine.create(spec)
    rows: list[tuple] = []
    snaps: dict[int, bytes] = {}
    day_start = -1
    for fields in frame_fields(frame):
        start = (fields[2] // 1440) * 1440
        if start != day_start:
            day_start = start
            eng.advance(start)
            snaps[start] = bytes(eng.snapshot(compress=False))
        rows.append(eng.process(eng.prepare(*fields)))
    return rows, snaps


def resume_engine(
    snapshot: bytes, spec: EngineSpec, frame: pl.DataFrame, *, until_minute: int | None = None
) -> list[tuple]:
    """Restore a snapshot and feed `frame` (whose first rank must be the snapshot's next_rank)."""
    from aml.features.engine import Engine

    eng, header = Engine.restore(snapshot, spec)
    if frame.height and int(frame["rank"][0]) != header["next_rank"]:
        raise ValueError(
            f"frame starts at rank {frame['rank'][0]}, snapshot at {header['next_rank']}"
        )
    rows: list[tuple] = []
    for fields in frame_fields(frame):
        if until_minute is not None and fields[2] > until_minute:
            break
        rows.append(eng.process(eng.prepare(*fields)))
    return rows


_BUILD_SCRIPT = """
import json, sys
from pathlib import Path
from aml.features.build import run_build_features
from aml.paths import DataPaths
root, dataset, out, cfg_file = sys.argv[1:5]
cfgs = json.loads(Path(cfg_file).read_text(encoding="utf-8"))
run_build_features(DataPaths(Path(root), dataset), Path(out), cfgs, mode="full")
"""


def build_in_subprocesses(
    jobs: Sequence[tuple[DataPaths, Path, int]], cfgs: dict, *, timeout: float = 600
) -> None:
    """Run `run_build_features(paths, out_dir, cfgs, mode="full")` for every (paths, out_dir,
    PYTHONHASHSEED) job, each in a fresh interpreter, all at once; raise on any failure."""
    import json
    import os
    import subprocess
    import sys
    import tempfile

    repo = CONFIG_DIR.parent
    with tempfile.TemporaryDirectory() as tmp:
        cfg_file = Path(tmp) / "cfgs.json"
        cfg_file.write_text(json.dumps(cfgs), encoding="utf-8")
        procs = []
        for paths, out_dir, seed in jobs:
            env = dict(os.environ, PYTHONHASHSEED=str(seed), PYTHONIOENCODING="utf-8")
            log = (Path(tmp) / f"log-{len(procs)}.txt").open("w", encoding="utf-8")
            args = [str(paths.root), paths.dataset, str(out_dir), str(cfg_file)]
            procs.append(
                (
                    subprocess.Popen(
                        [sys.executable, "-c", _BUILD_SCRIPT, *args],
                        cwd=repo,
                        env=env,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    ),
                    log,
                )
            )
        failed = []
        for k, (proc, log) in enumerate(procs):
            try:
                code = proc.wait(timeout=timeout)
            finally:
                log.close()
            if code != 0:
                text = (Path(tmp) / f"log-{k}.txt").read_text(encoding="utf-8")
                failed.append(f"job {k} exited {code}: {text[-3000:]}")
        if failed:
            raise AssertionError(" | ".join(failed))


_FIXTURE_BUILDS: dict[str, dict[str, Path]] = {}


def fixture_build_cfgs(data_cfg: dict, rules_cfg: dict) -> dict:
    """The configs a fixture build gets (the repo's, with the fixture's data config)."""
    return {
        "data": data_cfg,
        "rules": rules_cfg,
        "features": load_cfg("features"),
        "lgbm": load_cfg("lgbm"),
        "serving": load_cfg("serving"),
    }


def fixture_builds(prepared: DataPaths, cfgs: dict) -> dict[str, Path]:
    """Output dirs of four full builds of the prepared fixture, run once per session side by
    side, each in a fresh interpreter on its own copy of the inputs:

    - "seed0", "seed1": the original inputs, PYTHONHASHSEED 0 and 1 (determinism);
    - "flipped": every label flipped, "absent": no labels file at all (both PYTHONHASHSEED 0).
    """
    import atexit
    import json
    import shutil
    import tempfile

    key = json.dumps([str(prepared.root), cfgs], sort_keys=True, default=str)
    if key in _FIXTURE_BUILDS:
        return _FIXTURE_BUILDS[key]
    root = Path(tempfile.mkdtemp(prefix="aml-fixture-builds-"))
    atexit.register(shutil.rmtree, root, True)
    jobs, out = [], {}
    labels = pl.read_parquet(prepared.labels)
    for name, seed in (("seed0", 0), ("seed1", 1), ("flipped", 0), ("absent", 0)):
        paths = DataPaths(root / name / "volume", prepared.dataset)
        shutil.copytree(prepared.parquet_dir, paths.parquet_dir)
        if name != "absent":
            paths.labels.parent.mkdir(parents=True)
            lab = labels
            if name == "flipped":
                flip = (1 - pl.col("is_laundering")).cast(pl.Int8).alias("is_laundering")
                lab = labels.with_columns(flip)
            lab.write_parquet(paths.labels)
        out[name] = root / name / "features"
        jobs.append((paths, out[name], seed))
    build_in_subprocesses(jobs, cfgs)
    _FIXTURE_BUILDS[key] = out
    return out


def row_bits(row: Sequence) -> tuple:
    """A row as a hashable bit-exact key (floats by their IEEE bits, NaN == NaN)."""
    return tuple(
        ("f", struct.pack("<d", x)) if isinstance(x, float) else ("i", int(x)) for x in row
    )


def rows_by_id(frame: pl.DataFrame, rows: Sequence[tuple]) -> dict[int, tuple]:
    ids = frame["row_id"].to_list()[: len(rows)]
    return dict(zip(ids, rows, strict=True))


def rows_frame(frame: pl.DataFrame, rows: Sequence[tuple], spec: EngineSpec) -> pl.DataFrame:
    """row_id, rank + the row layout: features and severities Float64, the int tail Int64."""
    n = len(rows)
    cols: dict[str, Any] = {
        "row_id": frame["row_id"].head(n).to_list(),
        "rank": frame["rank"].head(n).to_list(),
    }
    n_float = spec.i_inflow
    for k, name in enumerate(spec.row_layout):
        cols[name] = [r[k] for r in rows]
    schema = {"row_id": pl.Int64, "rank": pl.Int64}
    schema |= {
        name: (pl.Float64 if k < n_float else pl.Int64) for k, name in enumerate(spec.row_layout)
    }
    return pl.DataFrame(cols, schema=schema)
