"""Independent DuckDB feature oracle (M2 spec §9.3).

`oracle.sql` recomputes the engine's windowed quantities with RANGE-frame SQL over per-account,
per-pair and per-format partitions (cents by DuckDB's round, l by polars' log1p, flags in SQL);
this module applies the §5 formulas in polars and compares the result with the engine's feature
table under the tolerance classes (`spec.tol_ok`, independent implementation). Written from the
spec only, never from the engine code.

Covered: every TX, VEL, AMT, FLOW, PORT and RULE feature, cyc2 and the two gather-scatter minima
(75 of 79 at the default config). Not covered here: cyc3 (checked on all real rows through the
round-trip rule parity, c2 + c3), cyc4 and the two scatter-gather counts (checked by the brute
force on the fixture).
"""

from __future__ import annotations

import math
import re
import shutil
import tempfile
import time
from pathlib import Path
from string import Template
from typing import Any

import numpy as np
import polars as pl

from aml.features.spec import (
    ENGINE_VERSION,
    FEATURE_SPEC_FILE,
    INPUT_COLUMNS,
    EngineSpec,
    format_slug,
    scan_feature_table,
    tol_ok,
    window_tag,
)
from aml.features.tx_features import build_tx_features
from aml.io import read_json, write_json_atomic
from aml.paths import DataPaths
from aml.rules.sql_baseline import connect

SQL_PATH = Path(__file__).with_name("oracle.sql")
VERIFY_DIR = "verify"
VERIFY_FILE = "verify.json"
M2_PREFIX = "m2__"  # oracle column holding the window's mean of l^2 for a mean/std feature
# Columns of the transactions table the SQL reads (plus l, added from polars).
SQL_COLUMNS = (
    "row_id",
    "rank",
    "minute",
    "src",
    "dst",
    "amount_usd",
    "amount_paid",
    "payment_format",
)
DEFAULT_MEMORY_LIMIT = "8GB"
_STEP = re.compile(r"^-- step: (\w+)\s*$", re.MULTILINE)
_NAN = float("nan")


# ---------------------------------------------------------------------------------------------
# SQL


def load_steps(path: Path = SQL_PATH) -> list[tuple[str, str]]:
    """(step name, SQL template) pairs in file order."""
    parts = _STEP.split(path.read_text(encoding="utf-8"))
    steps = [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts), 2)]
    if not steps:
        raise ValueError(f"no '-- step:' markers in {path}")
    return steps


def _whole(name: str, value: Any, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an int >= {minimum}, got {value!r}")
    return value


def _double(name: str, value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}")
    return f"CAST({float(value)!r} AS DOUBLE)"  # repr round-trips the double exactly


def render_steps(spec: EngineSpec) -> list[tuple[str, str]]:
    """The oracle SQL with every placeholder filled from the spec (numbers only)."""
    n_formats = len(spec.vocab["payment_format"])
    fmt_counts = "".join(
        f"count(*) FILTER (WHERE fmt = {k}) OVER ws AS u_out_fmt_{k},\n    "
        for k in range(n_formats)
    )
    params = {
        "short": str(_whole("windows.short", spec.w_short)),
        "long": str(_whole("windows.long", spec.w_long)),
        "pass_through": str(_whole("pass-through window", spec.w_pt)),
        "round_trip": str(_whole("round-trip window", spec.w_rt)),
        "round_cents": str(_whole("round_cents", spec.round_cents)),
        "band_low_usd": _double("band_low_usd", spec.band_low_usd),
        "band_high_usd": _double("band_high_usd", spec.band_high_usd),
        "fmt_counts": fmt_counts,
    }
    return [(name, Template(sql).substitute(params)) for name, sql in load_steps()]


# ---------------------------------------------------------------------------------------------
# Features


def not_covered(spec: EngineSpec) -> tuple[str, ...]:
    """Feature names the oracle does not compute (3/4-edge cycles, scatter-gather counts)."""
    rt, sg = window_tag(spec.w_rt), window_tag(spec.w_sg)
    return (f"cyc3_{rt}", f"cyc4_{rt}", f"sg_mids_{sg}", f"sg_srcs_{sg}")


def covered_features(spec: EngineSpec) -> tuple[str, ...]:
    """The spec's model inputs the oracle computes, in spec order."""
    skip = set(not_covered(spec))
    return tuple(n for n in spec.feature_names if n not in skip)


def _read_input(tx: Path | pl.DataFrame) -> pl.DataFrame:
    if isinstance(tx, pl.DataFrame):
        frame = tx.select(INPUT_COLUMNS)
    else:
        frame = pl.read_parquet(tx, columns=list(INPUT_COLUMNS))
    return frame.sort("rank")


def _sql_frame(
    frame: pl.DataFrame,
    spec: EngineSpec,
    *,
    threads: int | None,
    memory_limit: str | None,
    temp_dir: Path | None,
) -> pl.DataFrame:
    inp = frame.select(SQL_COLUMNS).with_columns(pl.col("amount_usd").log1p().alias("l"))
    con = connect(threads, memory_limit, temp_dir)
    try:
        con.register("o_tx_source", inp)
        con.execute("CREATE OR REPLACE TEMP VIEW o_tx AS SELECT * FROM o_tx_source")
        con.execute("CREATE OR REPLACE TEMP TABLE o_fmt (fmt VARCHAR, code INTEGER)")
        formats = list(spec.vocab["payment_format"])
        if formats:
            con.executemany(
                "INSERT INTO o_fmt VALUES (?, ?)", [[f, k] for k, f in enumerate(formats)]
            )
        steps = render_steps(spec)
        for _, sql in steps[:-1]:
            con.execute(sql)
        return con.execute(steps[-1][1]).pl()
    finally:
        con.close()


def _log1p_cents(cents: pl.Series) -> pl.Series:
    """log1p(c / 100) per value with an IEEE true division (exact ints < 2^53 as Float64)."""
    return pl.Series(cents.name, np.log1p(cents.to_numpy() / 100.0), dtype=pl.Float64)


def _expressions(spec: EngineSpec) -> tuple[dict[str, pl.Expr], dict[str, pl.Expr]]:
    """(feature name -> Float64 expression over the SQL result, m2 column -> expression)."""
    f64 = pl.Float64
    S, L = window_tag(spec.w_short), window_tag(spec.w_long)
    P, R = window_tag(spec.w_pt), window_tag(spec.w_rt)

    def num(col: str) -> pl.Expr:
        return pl.col(col).cast(f64)

    def lsum(col: str) -> pl.Expr:  # log1p(cents / 100) of an exact integer sum
        # The division runs in numpy: polars (1.44) divides a column by a scalar as a multiply
        # by its reciprocal (57 / 100 -> 57 * 0.01, one ulp off), which is not the spec's true
        # division and puts ~1.5% of the values 1-2 ulp away. The sums are never null.
        return pl.col(col).cast(f64).map_batches(_log1p_cents, return_dtype=f64)

    def mean(prefix: str, w: str) -> pl.Expr:
        return pl.when(pl.col(f"{prefix}_cnt_{w}") > 0).then(num(f"{prefix}_mean_{w}"))

    def std(prefix: str, w: str) -> pl.Expr:
        var = num(f"{prefix}_var_{w}")
        return (
            pl.when(pl.col(f"{prefix}_cnt_{w}") == 0)
            .then(pl.lit(_NAN))
            .when(var > 0)
            .then(var.sqrt())
            .otherwise(0.0)
        )

    def gap(last: str) -> pl.Expr:
        delta = pl.min_horizontal(pl.col("minute") - pl.col(last), pl.lit(spec.cap_gap))
        return pl.when(pl.col(last).is_null()).then(0.0).otherwise(delta.cast(f64))

    def port(col: str) -> pl.Expr:
        return pl.min_horizontal(pl.col(col), pl.lit(spec.cap_port)).cast(f64).log1p()

    def bal(side: str) -> pl.Expr:
        o, i = pl.col(f"{side}_out_sum_l"), pl.col(f"{side}_in_sum_l")
        return pl.when(o + i > 0).then((o - i).cast(f64) / (o + i).cast(f64))

    e: dict[str, pl.Expr] = {}
    m2: dict[str, pl.Expr] = {}
    for side, d in (("u", "out"), ("u", "in"), ("v", "in"), ("v", "out")):
        for w, t in (("s", S), ("l", L)):
            e[f"{side}_{d}_cnt_{t}"] = num(f"{side}_{d}_cnt_{w}")
            e[f"{side}_{d}_uniq_{t}"] = num(f"{side}_{d}_uniq_{w}")
            e[f"{side}_{d}_sum_{t}"] = lsum(f"{side}_{d}_sum_{w}")
    for side, d in (("u", "out"), ("v", "in")):
        p = f"{side}_{d}"
        for w, t in (("s", S), ("l", L)):
            e[f"{p}_mean_{t}"] = mean(p, w)
            e[f"{p}_std_{t}"] = std(p, w)
            m2[f"{M2_PREFIX}{p}_mean_{t}"] = m2[f"{M2_PREFIX}{p}_std_{t}"] = num(f"{p}_m2_{w}")
        e[f"{p}_max_{S}"] = num(f"{p}_max_s")
        e[f"{side}_amt_dev_{S}"] = pl.col("l") - mean(p, "s")
        m2[f"{M2_PREFIX}{side}_amt_dev_{S}"] = num(f"{p}_m2_s")
    e[f"pair_cnt_{S}"] = num("pair_cnt_s")
    e[f"pair_cnt_{L}"] = num("pair_cnt_l")
    e[f"u_inflow_{P}"] = lsum("inflow_c")
    e[f"pt_ratio_{P}"] = pl.when(~pl.col("self_loop") & (pl.col("inflow_c") > 0)).then(
        num("usd_c") / num("inflow_c")
    )
    e[f"u_bal_{L}"] = bal("u")
    e[f"v_bal_{L}"] = bal("v")
    e["pair_is_new"] = num("is_new")
    e["out_port"] = port("port_out")
    e["in_port"] = port("port_in")
    for name, last in (
        ("u_out_gap", "u_out_last"),
        ("u_in_gap", "u_in_last"),
        ("v_in_gap", "v_in_last"),
        ("v_out_gap", "v_out_last"),
        ("pair_gap", "pair_last"),
        ("rev_pair_gap", "rev_last"),
    ):
        e[name] = gap(last)
    e[f"cyc2_{R}"] = (
        pl.when(pl.col("self_loop"))
        .then(0.0)
        .otherwise(pl.min_horizontal(pl.col("rev_cnt_rt"), pl.lit(spec.cap_count)).cast(f64))
    )
    for side in ("u", "v"):
        e[f"gs_{side}_{S}"] = pl.min_horizontal(
            pl.col(f"{side}_in_uniq_s"), pl.col(f"{side}_out_uniq_s")
        ).cast(f64)
    e["in_band"] = num("in_band")
    e[f"u_out_inband_{S}"] = num("u_out_inband_s")
    e[f"u_out_round_{S}"] = num("u_out_round_s")
    e[f"u_out_newcp_{S}"] = num("u_out_newcp_s")
    for k, fmt in enumerate(spec.vocab["payment_format"]):
        e[f"u_out_fmt_{format_slug(fmt)}_{S}"] = num(f"u_out_fmt_{k}")
    e[f"v_in_same_fmt_{S}"] = num("v_in_same_fmt_s")
    e = {k: v.fill_null(_NAN) for k, v in e.items()}
    m2 = {k: v.fill_null(0.0) for k, v in m2.items()}
    return e, m2


def oracle_features(
    tx: Path | pl.DataFrame,
    spec: EngineSpec,
    *,
    threads: int | None = None,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
) -> pl.DataFrame:
    """row_id + the covered feature columns (Float64, spec order) + `m2__<feature>` helper
    columns for the mean/std-class features, in rank order.

    `tx` is a transactions Parquet path or a frame with spec INPUT_COLUMNS (ranks need not start
    at 0). Labels are never read.
    """
    frame = _read_input(tx)
    raw = _sql_frame(frame, spec, threads=threads, memory_limit=memory_limit, temp_dir=temp_dir)
    if not np.array_equal(raw["row_id"].to_numpy(), frame["row_id"].to_numpy()):
        raise RuntimeError("oracle SQL rows are not the input rows in rank order")
    exprs, m2 = _expressions(spec)
    cols = covered_features(spec)
    sql_part = [c for c in cols if c in exprs]
    out = raw.select(
        pl.col("row_id"),
        *[exprs[c].alias(c) for c in sql_part],
        *[v.alias(k) for k, v in m2.items()],
    )
    del raw  # the widest intermediate on real data
    vocab = {c: list(v) for c, v in spec.vocab.items()}
    tx_feats = build_tx_features(frame, vocab, spec.round_cents / 100)
    tx_part = [c for c in cols if c not in exprs]
    missing = [c for c in tx_part if c not in tx_feats.columns]
    if missing:
        raise RuntimeError(f"oracle has no definition for spec features {missing}")
    out = out.with_columns(tx_feats.select(pl.col(c).cast(pl.Float64) for c in tx_part))
    m2_cols = sorted(k for k in out.columns if k.startswith(M2_PREFIX))
    return out.select("row_id", *cols, *m2_cols)


# ---------------------------------------------------------------------------------------------
# Comparison


def compare_with_engine(
    engine: pl.DataFrame, oracle: pl.DataFrame, spec: EngineSpec, *, float32: bool
) -> dict:
    """Per covered column: rows, mismatches under its tolerance class (`spec.tol_ok`, independent
    implementation; `float32=True` when `engine` is the Float32 feature table), max |error| over
    rows where both are finite, NaN disagreements and a few example row_ids."""
    cols = [c for c in covered_features(spec) if c in oracle.columns]
    absent = [c for c in cols if c not in engine.columns]
    if absent:
        raise ValueError(f"engine table lacks covered columns {absent}")
    ids_e, ids_o = engine["row_id"].to_numpy(), oracle["row_id"].to_numpy()
    aligned = ids_e.shape == ids_o.shape and bool(np.array_equal(ids_e, ids_o))
    rows_missing = rows_extra = 0
    if not aligned:  # align on row_id; rows on one side only are reported
        rows_missing = oracle.join(engine.select("row_id"), on="row_id", how="anti").height
        rows_extra = engine.join(oracle.select("row_id"), on="row_id", how="anti").height
        engine = oracle.select("row_id").join(
            engine, on="row_id", how="inner", maintain_order="left"
        )
        oracle = engine.select("row_id").join(
            oracle, on="row_id", how="left", maintain_order="left"
        )
    ids = oracle["row_id"].to_numpy()
    per_col: dict[str, dict] = {}
    by_tol: dict[str, int] = {}
    for c in cols:
        tol = spec.feature(c).tol
        got = engine[c].to_numpy()
        want = oracle[c].to_numpy()
        m2 = oracle[f"{M2_PREFIX}{c}"].to_numpy() if tol in ("mean", "std") else None
        with np.errstate(invalid="ignore"):  # NaN magnitudes in the float32 ulp bound
            ok = tol_ok(tol, got, want, m2, independent=True, float32=float32)
        bad = np.flatnonzero(~ok)
        g64 = np.asarray(got, dtype=np.float64)
        w64 = np.asarray(want, dtype=np.float64)
        finite = np.isfinite(g64) & np.isfinite(w64)
        err = np.abs(g64[finite] - w64[finite])
        nan_diff = int(np.count_nonzero(np.isnan(g64) != np.isnan(w64)))
        per_col[c] = {
            "tol": tol,
            "mismatches": int(bad.size),
            "nan_disagreements": nan_diff,
            "max_abs_err": float(err.max()) if err.size else 0.0,
            "examples": [
                {"row_id": int(ids[k]), "engine": _jsonable(g64[k]), "oracle": _jsonable(w64[k])}
                for k in bad[:5]
            ],
        }
        by_tol[tol] = by_tol.get(tol, 0) + int(bad.size)
    total = sum(v["mismatches"] for v in per_col.values())
    return {
        "rows": int(ids.size),
        "float32": bool(float32),
        "aligned": aligned,
        "rows_missing": int(rows_missing),
        "rows_extra": int(rows_extra),
        "columns_checked": len(cols),
        "columns": per_col,
        "mismatches_by_tol": by_tol,
        "total_mismatches": int(total),
        "ok": total == 0 and rows_missing == 0 and rows_extra == 0,
    }


def _jsonable(x: float) -> float | str:
    return float(x) if math.isfinite(x) else str(x)


def run_oracle(
    paths: DataPaths,
    features_dir: Path,
    spec: EngineSpec,
    *,
    threads: int | None = 4,
    memory_limit: str | None = DEFAULT_MEMORY_LIMIT,
    raise_on_mismatch: bool = False,
) -> dict:
    """The verify mode of build_features: the oracle over all rows vs the feature parts.

    Writes `<features_dir>/verify/verify.json` and returns the same document: rows, columns
    checked, per column its tolerance class, mismatches, NaN disagreements, max |error| and
    example row_ids; `n_mismatches` = mismatching values + missing + extra rows, `ok`. The caller
    fails the job on `n_mismatches > 0` (after committing the evidence); `raise_on_mismatch`
    raises RuntimeError here instead, after writing the document.
    """
    t0 = time.perf_counter()
    features_dir = Path(features_dir)
    spec_file = features_dir / FEATURE_SPEC_FILE
    if spec_file.exists():
        stored = read_json(spec_file).get("spec_hash")
        if stored != spec.spec_hash():
            raise RuntimeError(f"{spec_file} has spec_hash {stored!r} != {spec.spec_hash()!r}")
    spill = Path(tempfile.mkdtemp(prefix="aml-oracle-duckdb-"))  # local disk, never the Volume
    try:
        ora = oracle_features(
            paths.transactions, spec, threads=threads, memory_limit=memory_limit, temp_dir=spill
        )
    finally:
        shutil.rmtree(spill, ignore_errors=True)
    t_sql = time.perf_counter() - t0
    cols = covered_features(spec)
    eng = scan_feature_table(features_dir, ["row_id", *cols]).collect()
    result = compare_with_engine(eng, ora, spec, float32=True)
    doc = {
        "engine_version": ENGINE_VERSION,
        "spec_hash": spec.spec_hash(),
        "covered": list(cols),
        "not_covered": list(not_covered(spec)),
        **result,
        "n_mismatches": result["total_mismatches"] + result["rows_missing"] + result["rows_extra"],
        "timings_s": {"oracle": t_sql, "total": time.perf_counter() - t0},
    }
    write_json_atomic(doc, features_dir / VERIFY_DIR / VERIFY_FILE)
    if raise_on_mismatch and not doc["ok"]:
        bad = {c: v["mismatches"] for c, v in doc["columns"].items() if v["mismatches"]}
        raise RuntimeError(
            f"feature oracle: {doc['total_mismatches']} mismatching values {bad}, "
            f"{doc['rows_missing']} rows missing, {doc['rows_extra']} extra (see verify.json)"
        )
    return doc
