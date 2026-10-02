"""The seven rule scenarios as scalar predicates on engine output (M2 spec §6, PLAN.md §6 M2).

`severities()` reproduces `scenarios.sql` bit-exactly; the engine calls it inside `_score`, so the
offline replay, the streaming scorer and the rules share one code path. The rules stage
(`run_rules_engine_stage`) reads the severity columns from the feature parts, recomputes the M1
SQL severities, checks parity on every row and tunes thresholds with M1's code
(`sql_baseline.tune_and_write`).
"""

from __future__ import annotations

import shutil
import tempfile
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, NamedTuple, NoReturn

import numpy as np
import polars as pl

from aml.config import rate_tag
from aml.features.spec import (
    EXACT_INT_LIMIT,
    FEATURE_SPEC_FILE,
    FEATURES_DIGEST,
    INFLOW_COLUMN,
    EngineSpec,
    NumericRangeError,
    parts_digest,
    scan_feature_table,
)
from aml.io import read_json, write_json_atomic, write_parquet_atomic, write_text_atomic
from aml.paths import DataPaths
from aml.rules.sql_baseline import (
    DEFAULT_MEMORY_LIMIT,
    RULES_OUTPUTS,
    SCENARIOS,
    alert_rates,
    apply_thresholds,
    check_tune_split,
    compute_severities,
    connect,
    hub_accounts,
    hub_degree_cap,
    register_transactions,
    severity_stats,
    sql_params,
    tune_and_write,
    tune_split_labels,
    tune_thresholds_detailed,
)

PARITY_FILE = "parity.json"
MISMATCHES_FILE = "parity_mismatches.parquet"
PARITY_REPORT = "parity.md"
MAX_MISMATCH_ROWS = 1000
VALIDATION_SPLITS = ("val_early", "val_late")
# Columns the stage reads from the feature parts.
ENGINE_COLUMNS = ("row_id", "rank", "day", "split", *SCENARIOS, INFLOW_COLUMN, "rule_trunc")
_ROUND_TRIP = SCENARIOS.index("round_trip")


class ParityError(RuntimeError):
    """Engine severities differ from the M1 SQL on a row the engine computed exactly (§6)."""


class RuleSupport(NamedTuple):
    """Per-event rule inputs, built inside `Engine._score` (field order is the contract).

    The engine may pass a plain tuple in this order; `severities` reads fields by position.
    Every count is read at **that scenario's own window** (spec.w_* from sql_params), from the
    applied state (events with minute in [m - W, m - 1]):

    u, v            src and dst account ids
    self_loop       1 if u == v (F_SELF), else 0
    usd_c           cents(amount_usd) of this event
    in_band         1 if F_IN_BAND (band_low_usd <= amount_usd < band_high_usd)
    is_round        1 if F_ROUND (paid_c > 0 and paid_c % round_cents == 0)
    hr              1 if F_HR (payment_format string in high_risk_formats)
    hub_u           hub[u] (0/1), the fixed train-fitted hub list
    fan_in_uniq_v   uniq_in[w_fan_in][v]: distinct senders into v (self-loops included)
    fan_out_uniq_u  uniq_out[w_fan_out][u]: distinct receivers of u (self-loops included)
    inflow_c        in nsl_sum_c[w_pt][u]: usd cents of non-self-loop events into u (int)
    c2, c3          raw round-trip path counts from `cycles.path_counts` (uncapped)
    st_cnt_u        out inband[w_struct][u]: u's in-band payments (self-loops included)
    ra_cnt_u        out round[w_round][u]: u's round payments
    hr_cnt_u        out hr[w_hr][u]: u's high-risk-format payments
    """

    u: int
    v: int
    self_loop: int
    usd_c: int
    in_band: int
    is_round: int
    hr: int
    hub_u: int
    fan_in_uniq_v: int
    fan_out_uniq_u: int
    inflow_c: int
    c2: int
    c3: int
    st_cnt_u: int
    ra_cnt_u: int
    hr_cnt_u: int


def severities(
    rs: Sequence[int], P: Mapping[str, int | float], excl: Sequence[bool]
) -> tuple[float, float, float, float, float, float, float]:
    """The 7 severities (float, `sql_baseline.SCENARIOS` order) of one event; pure and scalar.

    P = spec.sql_params (only `max_round_trip_paths` is read: windows were applied when `rs` was
    built); excl = spec.excl, the exclude_hub_senders flags in `spec.EXCL_SCENARIOS` order
    (fan_out_velocity, structuring, round_amount_burst, high_risk_format_burst). Mirrors
    scenarios.sql line by line:

    - fan_in_velocity: the distinct-sender count, never hub-segmented;
    - fan_out_velocity: 0 for a segmented-out hub sender, else the distinct-receiver count;
    - rapid_pass_through: self-loop targets are not query rows (0); with inflow > 0,
      `greatest(0, 1 - abs(CAST(usd_c AS DOUBLE) / CAST(inflow_c AS DOUBLE) - 1))`, the same
      IEEE operations in the same order; else 0;
    - round_trip: `least(c2 + c3, max_round_trip_paths)`, 0 for a self-loop target;
    - structuring / round_amount_burst / high_risk_format_burst: `1 + count` when the event's
      own flag is set and the sender is not segmented out, else 0.

    Raises `spec.NumericRangeError` if inflow_c or usd_c >= 2^53 (the float conversion would
    round, and DuckDB's HUGEINT -> DOUBLE rounding is not guaranteed to agree).
    """
    (
        u,
        v,
        _self_loop,
        usd_c,
        in_band,
        is_round,
        hr,
        hub_u,
        fan_in_uniq_v,
        fan_out_uniq_u,
        inflow_c,
        c2,
        c3,
        st_cnt_u,
        ra_cnt_u,
        hr_cnt_u,
    ) = rs
    if usd_c >= EXACT_INT_LIMIT or inflow_c >= EXACT_INT_LIMIT:
        raise NumericRangeError(
            f"usd_c {usd_c} or inflow_c {inflow_c} >= 2^53: not exactly representable as float"
        )
    ex_fan_out, ex_struct, ex_round, ex_hr = excl
    if u == v:
        pass_through = 0.0
        round_trip = 0.0
    else:
        if inflow_c > 0:
            x = 1.0 - abs(float(usd_c) / float(inflow_c) - 1.0)
            pass_through = x if x > 0.0 else 0.0  # greatest(0.0, x); x is never NaN here
        else:
            pass_through = 0.0
        n = c2 + c3
        cap = P["max_round_trip_paths"]
        round_trip = float(n if n < cap else cap)
    return (
        float(fan_in_uniq_v),
        0.0 if ex_fan_out and hub_u else float(fan_out_uniq_u),
        pass_through,
        round_trip,
        float(1 + st_cnt_u) if in_band and not (ex_struct and hub_u) else 0.0,
        float(1 + ra_cnt_u) if is_round and not (ex_round and hub_u) else 0.0,
        float(1 + hr_cnt_u) if hr and not (ex_hr and hub_u) else 0.0,
    )


# ---------------------------------------------------------------------------------------------
# The rules stage on engine severities (§8.2)


def run_rules_engine_stage(
    paths: DataPaths,
    features_dir: Path,
    out_dir: Path,
    rules_cfg: dict,
    data_cfg: dict,
    *,
    threads: int | None = None,
    memory_limit: str | None = DEFAULT_MEMORY_LIMIT,
    temp_dir: Path | None = None,
    m1_rules_dir: Path | None = None,
    m1_skip_reason: str | None = None,
) -> dict:
    """Rules from engine severities (§8.2): parity with the M1 SQL on every row, then M1 tuning.

    Reads the severity columns, rule_trunc, inflow_c, row_id, rank, split and day from the
    feature parts (`spec.scan_feature_table`) and feature_spec.json; recomputes the SQL
    severities, the hub cap and the hub list with M1 code; checks parity (§6):

    - the six O(1) scenarios: `==` on every row; round_trip: `==` on rows with rule_trunc == 0,
      and engine <= SQL (a lower bound) on truncated rows, which are excluded and reported;
    - the hub cap and hub list equal the feature spec's; max(inflow_c) < 2^53; the same row_ids
      in the same rank order.

    On any failure: parity.json, `paths.reports`/parity.md and (for value mismatches)
    parity_mismatches.parquet (<= 1,000 rows, both values) are written, earlier rules outputs in
    `out_dir` are removed, and `ParityError` is raised. Otherwise the M1 tail
    (`sql_baseline.tune_and_write`) writes severities.parquet, flags.parquet, thresholds.json and
    summary.json (last), with parity.json and parity.md written just before summary.json. The
    report also compares the flags with SQL-tuned flags per alert rate and split and, when
    `m1_rules_dir` holds the M1 stage's outputs, the thresholds and flags with M1's
    (`m1_regression`; a difference fails the stage unless truncated rows whose round trip is
    below the SQL's explain it). `m1_skip_reason` (with `m1_rules_dir=None`) says why M1 was not
    compared, e.g. outputs built from other prepared data. The summary records the parts'
    content digest (`features_digest`), which evaluate and export check.
    """
    check_tune_split(rules_cfg, data_cfg)
    features_dir, out_dir = Path(features_dir), Path(out_dir)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()

    spec = EngineSpec.from_json(read_json(features_dir / FEATURE_SPEC_FILE))
    _check_rule_inputs(spec, rules_cfg)
    digest = parts_digest(features_dir)
    eng = scan_feature_table(features_dir, ENGINE_COLUMNS).collect()
    timings["read_parts"] = time.perf_counter() - t0

    t = time.perf_counter()
    own_tmp = temp_dir is None
    spill = Path(tempfile.mkdtemp(prefix="aml-rules-duckdb-")) if own_tmp else Path(temp_dir)
    con = connect(threads, memory_limit, spill)
    try:
        register_transactions(con, paths.transactions)
        hub_cap = hub_degree_cap(con, float(rules_cfg["hub_degree_quantile"]))
        hubs = hub_accounts(con, hub_cap)
        timings["hub_cap"] = time.perf_counter() - t
        t = time.perf_counter()
        sql = compute_severities(con, rules_cfg, hub_cap)
        stats = severity_stats(con)
        timings["severities"] = time.perf_counter() - t
    finally:
        con.close()
        if own_tmp:
            shutil.rmtree(spill, ignore_errors=True)

    t = time.perf_counter()
    parity, mismatch_rows = compare_severities(eng, sql)
    parity["hubs"] = {
        "hub_cap_engine": spec.hub_cap,
        "hub_cap_sql": hub_cap,
        "n_hubs_engine": len(spec.hubs),
        "n_hubs_sql": len(hubs),
        "equal": spec.hub_cap == hub_cap and list(spec.hubs) == hubs,
    }
    if not parity["hubs"]["equal"]:
        parity["failures"].append("hub cap or hub list differ from the feature spec's")
    parity["features"] = {
        "dir": str(features_dir),
        "spec_hash": spec.spec_hash(),
        "rule_visits": spec.rule_visits,
    }
    timings["parity"] = time.perf_counter() - t

    report_path = paths.reports / PARITY_REPORT
    if parity["failures"]:
        _fail(parity, out_dir, report_path, mismatch_rows)
    (out_dir / MISMATCHES_FILE).unlink(missing_ok=True)

    sev = eng.select(
        pl.col("row_id").cast(pl.Int64),
        pl.col("split"),
        pl.col("day").cast(pl.Int16),
        *[pl.col(s).cast(pl.Float64) for s in SCENARIOS],
    )

    def before_summary(summary: dict[str, Any]) -> dict[str, Any]:
        # The four files are written; parity.json and parity.md go before summary.json.
        t1 = time.perf_counter()
        parity["flag_agreement"], parity["thresholds_equal_sql_tuned"] = _agreement_with_sql(
            sev, sql, summary["thresholds"], paths, rules_cfg
        )
        reg = parity["m1_regression"] = _m1_regression(out_dir, m1_rules_dir, m1_skip_reason)
        if reg["compared"] and not (reg["thresholds_equal"] and reg["flags_equal"]):
            # compare_severities passed, so every row equals the SQL except truncated rows whose
            # round trip is below it; with none of those, truncation cannot explain a difference.
            lower = parity["rule_trunc"]["round_trip_lower_than_sql"]
            if lower == 0:
                parity["failures"].append(
                    "thresholds or flags differ from the M1 rules outputs although every row "
                    "has engine severities equal to the SQL's"
                )
                _fail(parity, out_dir, report_path, None)
            parity["warnings"].append(
                f"thresholds or flags differ from M1's; {lower:,} truncated rows with a round "
                "trip below the SQL can explain it"
            )
        parity["timings_s"] = {"compare_flags": time.perf_counter() - t1}
        write_json_atomic(parity, out_dir / PARITY_FILE)
        write_text_atomic(render_parity_md(parity), report_path)
        keys = ("ok", "rows", "mismatches_total", "rule_trunc", "inflow_c_max", "hubs")
        brief = {k: parity[k] for k in (*keys, "flag_agreement", "m1_regression")}
        return {"parity": brief}

    return tune_and_write(
        sev,
        paths,
        out_dir,
        rules_cfg,
        data_cfg,
        hub_cap=hub_cap,
        stats=stats,
        extra_summary={
            "source": "engine",
            "features_dir": str(features_dir),
            "spec_hash": spec.spec_hash(),
            FEATURES_DIGEST: digest,
            "outputs": {n: str(out_dir / n) for n in (*RULES_OUTPUTS, PARITY_FILE)},
        },
        timings=timings,
        t0=t0,
        before_summary=before_summary,
    )


def _fail(
    parity: dict[str, Any], out_dir: Path, report_path: Path, mismatch_rows: pl.DataFrame | None
) -> NoReturn:
    """Write the failed parity's evidence, remove the stage's earlier outputs, raise."""
    parity["ok"] = False
    for name in RULES_OUTPUTS:  # a failed parity leaves no flags and no completion marker
        (out_dir / name).unlink(missing_ok=True)
    if mismatch_rows is not None and mismatch_rows.height:
        path = write_parquet_atomic(mismatch_rows, out_dir / MISMATCHES_FILE)
        parity["mismatch_file"] = str(path)
    write_json_atomic(parity, out_dir / PARITY_FILE)
    write_text_atomic(render_parity_md(parity), report_path)
    raise ParityError("rule parity failed: " + "; ".join(parity["failures"]) + f" (see {out_dir})")


def _check_rule_inputs(spec: EngineSpec, rules_cfg: dict) -> None:
    """The feature table must have been built with this rules config's engine inputs."""
    want = sql_params(rules_cfg, spec.hub_cap)
    if dict(spec.sql_params) != want:
        diff = sorted(
            k for k in set(want) | set(spec.sql_params) if spec.sql_params.get(k) != want.get(k)
        )
        raise ValueError(
            f"the feature table was built with other rule settings ({diff}); rebuild the features "
            "(`make features`) for this rules config"
        )
    if tuple(spec.high_risk_formats) != tuple(str(f) for f in rules_cfg["high_risk_formats"]):
        raise ValueError(
            "the feature table was built with other high_risk_formats; rebuild the features"
        )


def compare_severities(
    eng: pl.DataFrame, sql: pl.DataFrame
) -> tuple[dict[str, Any], pl.DataFrame | None]:
    """Parity counts of engine vs SQL severities (both in rank order) and the mismatching rows.

    `eng`: ENGINE_COLUMNS; `sql`: `compute_severities` output. Returns (parity document without
    the hub and M1 sections, up to MAX_MISMATCH_ROWS mismatching rows with both values or None).
    """
    failures: list[str] = []
    doc: dict[str, Any] = {
        "ok": True,
        "scenarios": list(SCENARIOS),
        "rows": eng.height,
        "rows_sql": sql.height,
        "failures": failures,
        "warnings": [],
    }
    split = eng["split"].to_numpy()
    splits = sorted(set(eng["split"].unique().to_list()))
    doc["rows_per_split"] = {s: int(np.count_nonzero(split == s)) for s in splits}
    rank = eng["rank"].to_numpy()
    in_order = bool(eng.height == 0 or np.array_equal(rank, np.arange(eng.height)))
    same_rows = eng.height == sql.height and bool(
        np.array_equal(eng["row_id"].to_numpy(), sql["row_id"].to_numpy())
    )
    doc["rank_contiguous"] = in_order
    doc["row_ids_equal"] = same_rows
    nulls = [c for c in ENGINE_COLUMNS if eng[c].null_count()]
    doc["null_columns"] = nulls
    if nulls:
        failures.append(f"feature parts hold nulls in {nulls}")
    if not in_order:
        failures.append("feature parts are not one contiguous rank-ordered table")
    if not same_rows:
        failures.append("feature rows and SQL rows differ (row_id sequence in rank order)")

    inflow = eng[INFLOW_COLUMN]
    inflow_max = int(inflow.max()) if eng.height and inflow.null_count() < eng.height else 0
    doc["inflow_c_max"] = inflow_max
    doc["inflow_c_limit"] = EXACT_INT_LIMIT
    if inflow_max >= EXACT_INT_LIMIT:
        failures.append(f"max inflow_c {inflow_max} >= 2^53")

    trunc = eng["rule_trunc"].fill_null(0).to_numpy() != 0
    n_trunc = int(trunc.sum())
    doc["rule_trunc"] = {
        "rows": n_trunc,
        "share": n_trunc / eng.height if eng.height else 0.0,
        "per_split": {s: int(np.count_nonzero(trunc & (split == s))) for s in splits},
        "round_trip_lower_than_sql": 0,
    }
    doc["mismatches"] = {s: {"all": 0, "validation": 0, "per_split": {}} for s in SCENARIOS}
    doc["mismatches_total"] = 0
    doc["mismatches_validation"] = 0
    if not same_rows:
        return doc, None

    bad_any = np.zeros(eng.height, dtype=bool)
    is_val = np.isin(split, VALIDATION_SPLITS)
    for j, s in enumerate(SCENARIOS):
        a = eng[s].cast(pl.Float64).fill_null(np.nan).to_numpy()
        b = sql[s].to_numpy()
        diff = a != b  # NaN on either side is a mismatch
        if j == _ROUND_TRIP:
            lower = trunc & diff & (a < b)
            doc["rule_trunc"]["round_trip_lower_than_sql"] = int(lower.sum())
            # Truncated rows keep a lower bound: excluded unless the engine exceeds the SQL.
            diff = diff & ~(trunc & (a < b))
        bad_any |= diff
        doc["mismatches"][s] = {
            "all": int(diff.sum()),
            "validation": int(np.count_nonzero(diff & is_val)),
            "per_split": {sp: int(np.count_nonzero(diff & (split == sp))) for sp in splits},
        }
    total = int(bad_any.sum())
    doc["mismatches_total"] = total
    doc["mismatches_validation"] = int(np.count_nonzero(bad_any & is_val))
    if not total:
        return doc, None
    worst = {s: m["all"] for s, m in doc["mismatches"].items() if m["all"]}
    failures.append(f"{total} rows with severity mismatches ({worst})")
    keep = pl.Series("keep", bad_any).arg_true().head(MAX_MISMATCH_ROWS)
    cols = [eng[c].gather(keep) for c in ("row_id", "rank", "day", "split", INFLOW_COLUMN)]
    cols.append(eng["rule_trunc"].gather(keep))
    for s in SCENARIOS:
        cols.append(eng[s].cast(pl.Float64).gather(keep).alias(f"{s}_engine"))
        cols.append(sql[s].gather(keep).alias(f"{s}_sql"))
    return doc, pl.DataFrame(cols)


def _agreement_with_sql(
    sev: pl.DataFrame,
    sql: pl.DataFrame,
    thresholds: dict[str, dict[str, float | None]],
    paths: DataPaths,
    rules_cfg: dict,
) -> tuple[dict[str, dict[str, float | None]], dict[str, bool]]:
    """Per alert rate: share of rows whose union flag from engine severities (engine-tuned
    thresholds) equals the one from SQL severities (SQL-tuned thresholds, as M1 tunes them), on
    val_early, val_late and all rows; and whether the two tunings chose the same thresholds."""
    y = tune_split_labels(sql, paths, rules_cfg["tune_split"])
    split = sev["split"]
    agreement: dict[str, dict[str, float | None]] = {}
    equal: dict[str, bool] = {}
    for r in alert_rates(rules_cfg):
        tag = rate_tag(r)
        thr_sql, _ = tune_thresholds_detailed(sql, y, rules_cfg, r)
        equal[tag] = thr_sql == thresholds[tag]
        same = (
            apply_thresholds(sev, thresholds[tag])["any"] == apply_thresholds(sql, thr_sql)["any"]
        )
        per: dict[str, float | None] = {}
        for name in (*VALIDATION_SPLITS, "all"):
            m = same if name == "all" else same.filter(split == name)
            per[name] = float(m.mean()) if m.len() else None
        agreement[tag] = per
    return agreement, equal


def _m1_regression(
    out_dir: Path, m1_dir: Path | None, skip_reason: str | None = None
) -> dict[str, Any]:
    """The engine-tuned thresholds.json / flags.parquet vs the M1 SQL stage's, when present
    (`skip_reason`: why the caller passed no M1 directory)."""
    if m1_dir is None or not all((Path(m1_dir) / n).exists() for n in RULES_OUTPUTS):
        return {
            "compared": False,
            "dir": None if m1_dir is None else str(m1_dir),
            "reason": skip_reason
            if m1_dir is None and skip_reason
            else "the M1 rules outputs were not found",
        }
    m1_dir = Path(m1_dir)
    mine, theirs = read_json(out_dir / "thresholds.json"), read_json(m1_dir / "thresholds.json")
    flags_mine = pl.read_parquet(out_dir / "flags.parquet")
    flags_m1 = pl.read_parquet(m1_dir / "flags.parquet")
    sev_mine = pl.read_parquet(out_dir / "severities.parquet")
    sev_m1 = pl.read_parquet(m1_dir / "severities.parquet")
    return {
        "compared": True,
        "dir": str(m1_dir),
        "thresholds_equal": mine["thresholds"] == theirs["thresholds"],
        "thresholds_doc_equal": mine == theirs,
        "flags_equal": flags_mine.equals(flags_m1),
        "severities_equal": sev_mine.equals(sev_m1),
    }


# ---------------------------------------------------------------------------------------------
# Report


def _fmt_share(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.6f}"


def render_parity_md(parity: dict) -> str:
    """reports/parity.md from the parity.json document."""
    ok = parity.get("ok", False)
    lines = [
        "# Rule parity: engine severities vs the M1 SQL",
        "",
        f"Result: **{'PASS' if ok else 'FAIL'}**. Every row is compared (M2 spec §6): the six O(1)",
        "scenarios on all rows, round trip on rows whose path walk was not truncated",
        "(`rule_trunc = 0`); truncated rows keep a lower bound and are reported below.",
        "",
        f"- Rows: {parity.get('rows', 0):,} (SQL: {parity.get('rows_sql', 0):,}); same row_ids "
        f"in rank order: {parity.get('row_ids_equal')}",
    ]
    rt = parity.get("rule_trunc", {})
    lines.append(
        f"- Truncated rows (`rule_trunc = 1`): {rt.get('rows', 0):,} "
        f"(share {_fmt_share(rt.get('share'))}); round trip below the SQL on "
        f"{rt.get('round_trip_lower_than_sql', 0):,} of them"
    )
    lines.append(
        f"- Max inflow_c: {parity.get('inflow_c_max', 0):,} (must be < 2^53 = "
        f"{parity.get('inflow_c_limit', EXACT_INT_LIMIT):,})"
    )
    hubs = parity.get("hubs")
    if hubs:
        lines.append(
            f"- Hubs: cap {hubs['hub_cap_engine']} (SQL {hubs['hub_cap_sql']}), "
            f"{hubs['n_hubs_engine']} accounts (SQL {hubs['n_hubs_sql']}); equal: {hubs['equal']}"
        )
    feats = parity.get("features")
    if feats:
        lines.append(f"- Feature table: `{feats['dir']}` (spec_hash `{feats['spec_hash']}`)")
    lines += ["", "## Rows per split", "", "| split | rows | truncated |", "| --- | ---: | ---: |"]
    per_trunc = rt.get("per_split", {})
    for s, n in parity.get("rows_per_split", {}).items():
        lines.append(f"| {s} | {n:,} | {per_trunc.get(s, 0):,} |")
    splits = list(parity.get("rows_per_split", {}))
    lines += [
        "",
        "## Mismatches per scenario",
        "",
        "| scenario | all | validation | " + " | ".join(splits) + " |",
        "| --- | ---: | ---: | " + " | ".join("---:" for _ in splits) + " |",
    ]
    for s, m in parity.get("mismatches", {}).items():
        cells = " | ".join(f"{m['per_split'].get(sp, 0):,}" for sp in splits)
        lines.append(f"| {s} | {m['all']:,} | {m['validation']:,} | {cells} |")
    agree = parity.get("flag_agreement")
    if agree:
        eq = parity.get("thresholds_equal_sql_tuned", {})
        lines += [
            "",
            "## Flags vs SQL-tuned flags",
            "",
            "Share of rows whose union flag from engine severities equals the one from SQL",
            "severities, each tuned with M1's code on its own severities.",
            "",
            "| alert rate | same thresholds | val_early | val_late | all |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
        for tag, per in agree.items():
            lines.append(
                f"| {tag} | {eq.get(tag)} | {_fmt_share(per.get('val_early'))} | "
                f"{_fmt_share(per.get('val_late'))} | {_fmt_share(per.get('all'))} |"
            )
    reg = parity.get("m1_regression")
    if reg is not None:
        lines += ["", "## M1 regression", ""]
        if reg.get("compared"):
            lines.append(
                f"Against `{reg['dir']}`: thresholds equal {reg['thresholds_equal']}, "
                f"thresholds.json equal {reg['thresholds_doc_equal']}, flags equal "
                f"{reg['flags_equal']}, severities equal {reg['severities_equal']}."
            )
        else:
            reason = reg.get("reason") or "the M1 rules outputs were not found"
            lines.append(f"Not compared: {reason}.")
    for title, key in (("Failures", "failures"), ("Warnings", "warnings")):
        items = parity.get(key) or []
        if items:
            lines += ["", f"## {title}", ""] + [f"- {x}" for x in items]
    if parity.get("mismatch_file"):
        lines += [
            "",
            f"Mismatching rows (first {MAX_MISMATCH_ROWS:,}): `{parity['mismatch_file']}`",
        ]
    return "\n".join(lines) + "\n"
