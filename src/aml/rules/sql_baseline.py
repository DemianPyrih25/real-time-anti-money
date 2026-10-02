"""SQL rule baseline, the "incumbent" (PLAN.md §6 M1).

Seven DuckDB scenarios (`scenarios.sql`) give every transaction one numeric severity each; a
scenario fires when severity >= its threshold. Thresholds are tuned greedily on the tune split
for recall under a union alert-rate cap. Every windowed quantity obeys the as-of rule
(PLAN.md §4): a transaction in minute m sees only minutes [m - W, m - 1].
"""

from __future__ import annotations

import math
import numbers
import re
import shutil
import tempfile
import time
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from string import Template
from typing import Any

import duckdb
import numpy as np
import polars as pl

from aml.config import rate_tag
from aml.io import write_json_atomic, write_parquet_atomic
from aml.paths import DataPaths

SCENARIOS = (
    "fan_in_velocity",
    "fan_out_velocity",
    "rapid_pass_through",
    "round_trip",
    "structuring",
    "round_amount_burst",
    "high_risk_format_burst",
)
SQL_PATH = Path(__file__).with_name("scenarios.sql")
# Test is touched once; val_late is reserved for model thresholds and calibration (PLAN.md §4).
FORBIDDEN_TUNE_SPLITS = ("test", "val_late")
MAX_ROUND_TRIP_PATHS = 100
# Sender-keyed scenarios that may segment out hub senders (`exclude_hub_senders: true`): a hub
# is an account whose train-period degree exceeds the hub cap, the same definition the round
# trip uses for intermediates. SQL placeholder name per scenario.
HUB_SEGMENTABLE = {
    "fan_out_velocity": "fan_out_excl_hubs",
    "structuring": "structuring_excl_hubs",
    "round_amount_burst": "round_excl_hubs",
    "high_risk_format_burst": "high_risk_excl_hubs",
}
DEFAULT_MEMORY_LIMIT = "12GB"
# Columns of the transactions table that the scenarios read.
TX_COLUMNS = (
    "row_id",
    "rank",
    "minute",
    "day",
    "split",
    "src",
    "dst",
    "amount_paid",
    "amount_usd",
    "payment_format",
)

_STEP = re.compile(r"^-- step: (\w+)\s*$", re.MULTILINE)


# ---------------------------------------------------------------------------------------------
# SQL loading and parameters


def load_steps(path: Path = SQL_PATH) -> list[tuple[str, str]]:
    """(step name, SQL template) pairs in file order."""
    text = path.read_text(encoding="utf-8")
    parts = _STEP.split(text)
    steps = [(parts[i], parts[i + 1].strip()) for i in range(1, len(parts), 2)]
    if not steps:
        raise ValueError(f"no '-- step:' markers in {path}")
    return steps


def _number(name: str, value: Any) -> int | float:
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(float(value)):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return value


def _minutes(name: str, value: Any, *, minimum: int = 1) -> int:
    v = _number(name, value)
    if float(v) != int(v) or int(v) < minimum:
        raise ValueError(f"{name} must be a whole number of minutes >= {minimum}, got {value!r}")
    return int(v)


def _sql_literal(value: int | float) -> str:
    if isinstance(value, int):
        return str(value)
    # repr round-trips a double exactly; the cast keeps DuckDB from reading it as a DECIMAL.
    return f"CAST({float(value)!r} AS DOUBLE)"


def _round_cents(unit: Any) -> int:
    cents = float(_number("round_unit", unit)) * 100
    if round(cents) <= 0 or abs(cents - round(cents)) > 1e-9:
        raise ValueError(f"round_unit must be a positive whole number of cents, got {unit!r}")
    return int(round(cents))


def scenario_cfg(rules_cfg: dict, name: str) -> dict:
    try:
        return rules_cfg["scenarios"][name]
    except KeyError:
        raise ValueError(f"rules config has no scenario {name!r}") from None


def check_rules_config(rules_cfg: dict) -> None:
    """Raise if the rules config does not define exactly SCENARIOS with valid windows and grids,
    or tunes on a split reserved for something else (PLAN.md §4)."""
    if rules_cfg.get("tune_split") in FORBIDDEN_TUNE_SPLITS:
        raise ValueError(
            f"tune_split {rules_cfg['tune_split']!r} is not allowed: rules are tuned on val_early "
            "(PLAN.md §6 M1); test is touched once and val_late is reserved for thresholds and "
            "calibration"
        )
    names = set(rules_cfg.get("scenarios", {}))
    if names != set(SCENARIOS):
        raise ValueError(f"rules scenarios must be {sorted(SCENARIOS)}, got {sorted(names)}")
    for name in SCENARIOS:
        sc = scenario_cfg(rules_cfg, name)
        _minutes(f"{name}.window_minutes", sc["window_minutes"])
        grid = sc["grid"]
        if not grid:
            raise ValueError(f"{name}.grid is empty")
        for g in grid:
            if float(_number(f"{name}.grid", g)) <= 0:
                raise ValueError(f"{name}.grid values must be > 0, got {g!r}")
        excl = sc.get("exclude_hub_senders", False)
        if not isinstance(excl, bool):
            raise ValueError(f"{name}.exclude_hub_senders must be true or false, got {excl!r}")
        if excl and name not in HUB_SEGMENTABLE:
            raise ValueError(
                f"{name}.exclude_hub_senders is only supported for {sorted(HUB_SEGMENTABLE)}"
            )
    rt = scenario_cfg(rules_cfg, "round_trip")
    hop = _minutes("round_trip.hop_window_minutes", rt["hop_window_minutes"], minimum=0)
    if hop > int(rt["window_minutes"]):
        # The cumulative-count identity in scenarios.sql needs H <= W.
        raise ValueError("round_trip.hop_window_minutes must be <= round_trip.window_minutes")
    for r in alert_rates(rules_cfg):
        if not 0 < float(_number("alert rate", r)) <= 1:
            raise ValueError(f"alert rates must be in (0, 1], got {r!r}")
    low = float(_number("structuring_band_low", rules_cfg["structuring_band_low"]))
    if not 0 < low < 1:
        raise ValueError(f"structuring_band_low must be in (0, 1), got {low!r}")
    if float(_number("structuring_threshold_usd", rules_cfg["structuring_threshold_usd"])) <= 0:
        raise ValueError("structuring_threshold_usd must be > 0")
    _round_cents(rules_cfg["round_unit"])


def sql_params(rules_cfg: dict, hub_cap: int) -> dict[str, int | float]:
    """Every value substituted into scenarios.sql (numbers only)."""
    check_rules_config(rules_cfg)
    windows = {n: int(scenario_cfg(rules_cfg, n)["window_minutes"]) for n in SCENARIOS}
    threshold = float(rules_cfg["structuring_threshold_usd"])
    hop = int(scenario_cfg(rules_cfg, "round_trip")["hop_window_minutes"])
    return {
        "round_cents": _round_cents(rules_cfg["round_unit"]),
        "band_low_usd": float(rules_cfg["structuring_band_low"]) * threshold,
        "band_high_usd": threshold,
        "fan_in_window": windows["fan_in_velocity"],
        "fan_out_window": windows["fan_out_velocity"],
        "pass_through_window": windows["rapid_pass_through"],
        "structuring_window": windows["structuring"],
        "round_window": windows["round_amount_burst"],
        "high_risk_window": windows["high_risk_format_burst"],
        "round_trip_window_plus_1": windows["round_trip"] + 1,
        "hop_window": hop,
        "hop_bucket": max(hop, 1),
        "hub_cap": _minutes("hub_cap", hub_cap, minimum=0),
        "max_round_trip_paths": MAX_ROUND_TRIP_PATHS,
        **{
            param: int(bool(scenario_cfg(rules_cfg, name).get("exclude_hub_senders", False)))
            for name, param in HUB_SEGMENTABLE.items()
        },
    }


def render_steps(rules_cfg: dict, hub_cap: int) -> list[tuple[str, str]]:
    params = {k: _sql_literal(_number(k, v)) for k, v in sql_params(rules_cfg, hub_cap).items()}
    return [(name, Template(sql).substitute(params)) for name, sql in load_steps()]


# ---------------------------------------------------------------------------------------------
# Connection


def connect(
    threads: int | None = None,
    memory_limit: str | None = None,
    temp_dir: Path | None = None,
) -> duckdb.DuckDBPyConnection:
    """In-memory DuckDB with thread count, memory cap and a spill directory set."""
    con = duckdb.connect()
    if threads:
        con.execute(f"SET threads = {int(threads)}")
    if memory_limit:
        if not re.fullmatch(r"\d+(\.\d+)?\s*[KMGT]i?B", memory_limit):
            raise ValueError(f"bad memory_limit {memory_limit!r}")
        con.execute(f"SET memory_limit = '{memory_limit}'")
    if temp_dir is not None:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{_sql_path(Path(temp_dir))}'")
    # Row order is fixed by ORDER BY where it matters; this lets DuckDB stream and spill freely.
    con.execute("SET preserve_insertion_order = false")
    return con


def _sql_path(p: Path) -> str:
    return p.resolve().as_posix().replace("'", "''")


def register_transactions(con: duckdb.DuckDBPyConnection, source: Path | pl.DataFrame) -> None:
    """(Re)define view `tx` over a transactions Parquet file or an in-memory polars frame."""
    cols = ", ".join(TX_COLUMNS)
    if isinstance(source, pl.DataFrame):
        con.register("r_tx_source", source.select(TX_COLUMNS))
        con.execute(f"CREATE OR REPLACE TEMP VIEW tx AS SELECT {cols} FROM r_tx_source")
    else:
        path = _sql_path(Path(source))
        con.execute(f"CREATE OR REPLACE TEMP VIEW tx AS SELECT {cols} FROM read_parquet('{path}')")


# ---------------------------------------------------------------------------------------------
# Severities


def hub_degree_cap(con: duckdb.DuckDBPyConnection, quantile: float) -> int:
    """Train-period total degree (in + out edge count) at `quantile` over accounts active in train.

    Same definition as the EDA's train_degree table (a self-loop counts as one out and one in;
    DuckDB quantile_disc), so the two agree.
    """
    q = float(_number("hub_degree_quantile", quantile))
    if not 0 < q <= 1:
        raise ValueError(f"hub_degree_quantile must be in (0, 1], got {quantile!r}")
    row = con.execute(
        f"""
        WITH e AS (SELECT src, dst FROM tx WHERE split = 'train'),
             d AS (SELECT acct, count(*) AS deg
                   FROM (SELECT src AS acct FROM e UNION ALL SELECT dst AS acct FROM e)
                   GROUP BY acct)
        SELECT quantile_disc(deg, {q!r}) FROM d
        """
    ).fetchone()
    if row is None or row[0] is None:
        raise ValueError("no train rows: cannot fit the hub degree cap")
    return int(row[0])


def hub_accounts(con: duckdb.DuckDBPyConnection, hub_cap: int) -> list[int]:
    """Sorted ids of the accounts whose train-period degree exceeds `hub_cap`.

    The `r_hubs` definition of scenarios.sql (in + out edge count over train rows, a self-loop
    counts once as out and once as in), on view `tx`. The M2 engine takes this list as a fixed
    input, so the rules and the engine segment and exclude exactly the same accounts.
    """
    cap = _minutes("hub_cap", hub_cap, minimum=0)
    rows = con.execute(
        f"""
        WITH e AS (SELECT src, dst FROM tx WHERE split = 'train')
        SELECT acct
        FROM (SELECT src AS acct FROM e UNION ALL SELECT dst AS acct FROM e)
        GROUP BY acct
        HAVING count(*) > {cap}
        ORDER BY acct
        """
    ).fetchall()
    return [int(r[0]) for r in rows]


def compute_severities(
    con: duckdb.DuckDBPyConnection, rules_cfg: dict, hub_cap: int
) -> pl.DataFrame:
    """row_id, split, day and one Float64 severity per scenario for every row, in rank order.

    View `tx` must already be registered (see `register_transactions`). Leaves a one-row table
    `r_stats` in the connection and drops the other working tables.
    """
    con.execute("CREATE OR REPLACE TEMP TABLE r_high_risk_formats (fmt VARCHAR)")
    formats = [str(f) for f in rules_cfg["high_risk_formats"]]
    if formats:
        con.executemany("INSERT INTO r_high_risk_formats VALUES (?)", [[f] for f in formats])
    steps = render_steps(rules_cfg, hub_cap)
    for _, sql in steps[:-1]:
        con.execute(sql)
    sev = con.execute(steps[-1][1]).pl()
    for table in (
        "r_tx",
        "r_sender",
        "r_receiver",
        "r_pass",
        "r_edges",
        "r_closed",
        "r_hubs",
        "r_paths",
        "r_round_trip",
    ):
        con.execute(f"DROP TABLE IF EXISTS {table}")
    return sev.with_columns(
        pl.col("row_id").cast(pl.Int64),
        pl.col("day").cast(pl.Int16),
        *[pl.col(s).cast(pl.Float64) for s in SCENARIOS],
    )


def severity_stats(con: duckdb.DuckDBPyConnection) -> dict[str, int]:
    """Counts left in `r_stats` by the last compute_severities call."""
    cur = con.execute("SELECT * FROM r_stats")
    names = [d[0] for d in cur.description]
    row = cur.fetchone()
    return {n: int(v) for n, v in zip(names, row, strict=True)}


# ---------------------------------------------------------------------------------------------
# Thresholds


def alert_rates(rules_cfg: dict) -> list[float]:
    """Headline alert rate first, then the sensitivity rates (duplicates dropped)."""
    out: list[float] = []
    for r in [rules_cfg["alert_rate"], *rules_cfg.get("sensitivity_alert_rates", [])]:
        if float(r) not in out:
            out.append(float(r))
    return out


def _grids(rules_cfg: dict) -> list[np.ndarray]:
    """Per scenario, the distinct grid values sorted from strict (high) to loose (low)."""
    return [
        np.array(sorted({float(g) for g in scenario_cfg(rules_cfg, s)["grid"]}, reverse=True))
        for s in SCENARIOS
    ]


def _budget(alert_rate: float, n: int) -> int:
    # Largest alert count whose rate is <= alert_rate (the epsilon absorbs float fuzz).
    return int(math.floor(float(alert_rate) * n + 1e-9))


def _first_fire(mat: np.ndarray, grids: list[np.ndarray]) -> np.ndarray:
    """Per row and scenario, the strictest grid index at which it fires (len(grid) = never).

    Grid values are > 0, so severity >= value also implies severity > 0.
    """
    out = np.empty(mat.shape, dtype=np.int32)
    for j, g in enumerate(grids):
        out[:, j] = len(g) - np.searchsorted(g[::-1], mat[:, j], side="right")
    return out


def _greedy(
    first: np.ndarray, y: np.ndarray, sizes: list[int], budget: int, mode: str = "ratio"
) -> list[int | None]:
    """Greedy path of grid indices (None = off) under a union alert budget.

    A move sets one scenario to any looser grid value. Among moves that keep the union within
    the budget and add at least one true positive, take the best new-TP / new-alert ratio
    (mode "ratio"; ties: more new TPs) or the most new TPs (mode "gain"; ties: fewer new
    alerts); then scenario order, then the stricter value. Stop when none fits.
    """
    if mode not in ("ratio", "gain"):
        raise ValueError(f"unknown greedy mode {mode!r}")
    n, k = first.shape
    level: list[int | None] = [None] * k
    union = np.zeros(n, dtype=bool)
    alerts = 0
    while True:
        best: tuple[int, int, int, int] | None = None  # (scenario, grid index, d_tp, d_alerts)
        free = ~union
        free_pos = free & y
        for j in range(k):
            size = sizes[j]
            start = 0 if level[j] is None else level[j] + 1
            if start >= size:
                continue
            # New alerts / TPs if scenario j moved to each grid index (rows not yet alerted).
            new_alerts = np.cumsum(np.bincount(first[free, j], minlength=size + 1))
            new_tp = np.cumsum(np.bincount(first[free_pos, j], minlength=size + 1))
            for idx in range(start, size):
                d_alerts = int(new_alerts[idx])
                if alerts + d_alerts > budget:
                    break  # looser values only add alerts
                d_tp = int(new_tp[idx])
                if d_tp == 0:
                    continue
                if best is None:
                    better = True
                elif mode == "ratio":
                    lhs, rhs = d_tp * best[3], best[2] * d_alerts
                    better = lhs > rhs or (lhs == rhs and d_tp > best[2])
                else:
                    better = d_tp > best[2] or (d_tp == best[2] and d_alerts < best[3])
                if better:
                    best = (j, idx, d_tp, d_alerts)
        if best is None:
            return level
        j, idx, _, d_alerts = best
        level[j] = idx
        union |= first[:, j] <= idx
        alerts += d_alerts


def _best_single(
    first: np.ndarray, y: np.ndarray, sizes: list[int], budget: int
) -> list[int | None]:
    """The single scenario and grid value with the most TPs within the budget (all others off).

    Ties: fewer alerts, then scenario order, then the stricter value. All off if none has a TP.
    """
    best: tuple[int, int, int, int] | None = None  # (scenario, grid index, tp, alerts)
    for j, size in enumerate(sizes):
        alerts = np.cumsum(np.bincount(first[:, j], minlength=size + 1))
        tps = np.cumsum(np.bincount(first[y, j], minlength=size + 1))
        for idx in range(size):
            a, tp = int(alerts[idx]), int(tps[idx])
            if a > budget:
                break
            if tp == 0:
                continue
            if best is None or tp > best[2] or (tp == best[2] and a < best[3]):
                best = (j, idx, tp, a)
    level: list[int | None] = [None] * len(sizes)
    if best is not None:
        level[best[0]] = best[1]
    return level


def _union(first: np.ndarray, level: list[int | None]) -> np.ndarray:
    union = np.zeros(first.shape[0], dtype=bool)
    for j, idx in enumerate(level):
        if idx is not None:
            union |= first[:, j] <= idx
    return union


def _union_tp(first: np.ndarray, y: np.ndarray, level: list[int | None]) -> int:
    return int(np.count_nonzero(_union(first, level) & y))


# Candidate solutions per rate, in tie-break order (the spec's ratio greedy first).
TUNING_CANDIDATES = ("ratio_greedy", "gain_greedy", "best_single")


def _candidates(
    first: np.ndarray, y: np.ndarray, sizes: list[int], budget: int
) -> list[tuple[str, list[int | None]]]:
    return [
        ("ratio_greedy", _greedy(first, y, sizes, budget, "ratio")),
        ("gain_greedy", _greedy(first, y, sizes, budget, "gain")),
        ("best_single", _best_single(first, y, sizes, budget)),
    ]


def _pick(
    first: np.ndarray, y: np.ndarray, cands: list[tuple[str, list[int | None]]]
) -> tuple[str, list[int | None], int]:
    """Most union TPs; ties: the ratio greedy, then fewer union alerts, then candidate order."""
    best: tuple[tuple, str, list[int | None], int] | None = None
    for i, (name, level) in enumerate(cands):
        union = _union(first, level)
        tp = int(np.count_nonzero(union & y))
        key = (tp, name == "ratio_greedy", -int(union.sum()), -i)
        if best is None or key > best[0]:
            best = (key, name, level, tp)
    assert best is not None
    return best[1], best[2], best[3]


def _tune_arrays(
    sev: pl.DataFrame, y: Sequence[int] | np.ndarray | pl.Series, rules_cfg: dict
) -> tuple[np.ndarray, np.ndarray]:
    """Severity matrix and labels of the tune-split rows."""
    missing = [c for c in ("split", *SCENARIOS) if c not in sev.columns]
    if missing:
        raise ValueError(f"severity frame lacks columns: {missing}")
    y_arr = pl.Series("y", y) if not isinstance(y, pl.Series) else y.rename("y")
    if y_arr.len() != sev.height:
        raise ValueError(f"y has {y_arr.len()} rows, severities have {sev.height}")
    is_tune = sev["split"] == rules_cfg["tune_split"]
    y_tune = y_arr.filter(is_tune)
    if y_tune.null_count():
        raise ValueError("labels are missing for some tune-split rows")
    mat = sev.filter(is_tune).select(SCENARIOS).to_numpy().astype(np.float64)
    return np.nan_to_num(mat, nan=0.0), y_tune.cast(pl.Int8).to_numpy() == 1


def tune_thresholds_detailed(
    sev: pl.DataFrame,
    y: Sequence[int] | np.ndarray | pl.Series,
    rules_cfg: dict,
    alert_rate: float,
) -> tuple[dict[str, float | None], dict[str, Any]]:
    """`tune_thresholds` plus how the solution was found: (thresholds, info).

    info: candidate (which search won), from_rate (the configured rate it was found at),
    tune_tp, tune_alerts, budget and tune_rows.
    """
    check_rules_config(rules_cfg)
    rate = float(_number("alert_rate", alert_rate))
    if not 0 < rate <= 1:
        raise ValueError(f"alert_rate must be in (0, 1], got {alert_rate!r}")
    mat, y_tune = _tune_arrays(sev, y, rules_cfg)
    grids = _grids(rules_cfg)
    n = mat.shape[0]
    # Rows no scenario can ever flag change no count.
    keep = (mat > 0).any(axis=1)
    first, y_keep = _first_fire(mat[keep], grids), y_tune[keep]
    sizes = [len(g) for g in grids]

    rates = [rate, *sorted((r for r in alert_rates(rules_cfg) if r < rate), reverse=True)]
    best: tuple[float, str, list[int | None], int] | None = None
    for r in rates:
        name, level, tp = _pick(first, y_keep, _candidates(first, y_keep, sizes, _budget(r, n)))
        if best is None or tp > best[3]:
            best = (r, name, level, tp)
    assert best is not None
    from_rate, name, level, tp = best
    thresholds = {
        s: (None if idx is None else float(grids[j][idx]))
        for j, (s, idx) in enumerate(zip(SCENARIOS, level, strict=True))
    }
    info = {
        "candidate": name,
        "from_rate": from_rate,
        "tune_tp": tp,
        "tune_alerts": int(_union(first, level).sum()),
        "budget": _budget(rate, n),
        "tune_rows": n,
    }
    return thresholds, info


def tune_thresholds(
    sev: pl.DataFrame,
    y: Sequence[int] | np.ndarray | pl.Series,
    rules_cfg: dict,
    alert_rate: float,
) -> dict[str, float | None]:
    """Threshold search on the tune-split rows; None = scenario off. Deterministic.

    `y` is aligned with `sev` rows (only tune-split rows are read; others may be null). The
    union alert rate on the tune split stays <= alert_rate. Three candidate solutions are built
    per rate and the one with the most tune TPs kept (ties: the ratio greedy, then fewer
    alerts): the spec's greedy on the best new-TP / new-alert ratio, a greedy on the most new
    TPs, and the best single scenario/value. The ratio greedy alone can let a small, precise
    move block a much larger one (a 0/1 knapsack), so the other two guard it. All of this is
    also run at every configured alert rate below `alert_rate` and the best-recall solution kept
    (each fits this budget too), so tune recall never falls as the configured rate grows.
    """
    return tune_thresholds_detailed(sev, y, rules_cfg, alert_rate)[0]


def scenario_diagnostics(
    sev: pl.DataFrame,
    y: Sequence[int] | np.ndarray | pl.Series,
    rules_cfg: dict,
    thresholds: dict[str, dict[str, float | None]],
    rates: dict[str, float],
) -> dict[str, Any]:
    """Why each scenario is on or off, from the tune-split rows (no labels beyond tuning's).

    A scenario whose strictest grid value alone flags more tune rows than a rate's budget can
    never be switched on at that rate (e.g. fan-out velocity around hub senders).
    `thresholds`: rate_tag -> scenario -> threshold (None = off); `rates`: rate_tag -> rate.
    """
    mat, y_tune = _tune_arrays(sev, y, rules_cfg)
    n = mat.shape[0]
    grids = _grids(rules_cfg)
    per_scenario = {}
    for j, s in enumerate(SCENARIOS):
        strictest = float(grids[j][0])
        per_scenario[s] = {
            "strictest_value": strictest,
            "alerts_at_strictest": int(np.count_nonzero(mat[:, j] >= strictest)),
            "tp_at_strictest": int(np.count_nonzero((mat[:, j] >= strictest) & y_tune)),
        }
    per_rate = {}
    for tag, thr in thresholds.items():
        budget = _budget(rates[tag], n)
        rows = {}
        for j, s in enumerate(SCENARIOS):
            t = thr.get(s)
            fired = (mat[:, j] >= t) & (mat[:, j] > 0) if t is not None else None
            rows[s] = {
                "threshold": t,
                "feasible": per_scenario[s]["alerts_at_strictest"] <= budget,
                "alone_alerts": None if fired is None else int(np.count_nonzero(fired)),
                "alone_tp": None if fired is None else int(np.count_nonzero(fired & y_tune)),
            }
        per_rate[tag] = {
            "budget": budget,
            "active": [s for s in SCENARIOS if thr.get(s) is not None],
            "infeasible": [s for s in SCENARIOS if not rows[s]["feasible"]],
            "scenarios": rows,
        }
    return {"tune_rows": n, "scenarios": per_scenario, "rates": per_rate}


def apply_thresholds(sev: pl.DataFrame, thresholds: dict[str, float | None]) -> pl.DataFrame:
    """row_id + fired_<scenario> (Boolean) + `any`."""
    fired = []
    for s in SCENARIOS:
        thr = thresholds.get(s)
        if thr is None:
            fired.append(pl.lit(False).alias(f"fired_{s}"))
        else:
            thr = float(_number(f"threshold {s}", thr))
            fired.append(((pl.col(s) >= thr) & (pl.col(s) > 0)).alias(f"fired_{s}"))
    out = sev.select(pl.col("row_id"), *fired)
    return out.with_columns(pl.any_horizontal([f"fired_{s}" for s in SCENARIOS]).alias("any"))


def _rate_metrics(fired: pl.Series, split: pl.Series, y: pl.Series, tune: str) -> dict:
    """Alert counts per split, plus recall / precision on the tune split (y: tune rows only)."""
    is_tune = (split == tune).to_numpy()
    flag = fired.to_numpy()
    tune_fired = flag[is_tune]
    y_tune = y.filter(pl.Series(is_tune)).to_numpy() == 1
    positives = int(y_tune.sum())
    tp = int(np.count_nonzero(tune_fired & y_tune))
    alerts = int(np.count_nonzero(tune_fired))
    per_split = {}
    for s in sorted(split.unique().to_list()):  # 4 names, not a 5M-element Python list
        m = (split == s).to_numpy()
        per_split[s] = {
            "rows": int(m.sum()),
            "alerts": int(np.count_nonzero(flag[m])),
            "alert_rate": float(np.count_nonzero(flag[m]) / m.sum()) if m.any() else None,
        }
    val_late = per_split.get("val_late", {})
    return {
        "tune_rows": int(is_tune.sum()),
        "tune_positives": positives,
        "tune_alerts": alerts,
        "tune_alert_rate": alerts / int(is_tune.sum()) if is_tune.any() else None,
        "tune_recall": tp / positives if positives else None,
        "tune_precision": tp / alerts if alerts else None,
        "val_late_alerts": val_late.get("alerts"),
        "val_late_alert_rate": val_late.get("alert_rate"),
        "per_split": per_split,
    }


# ---------------------------------------------------------------------------------------------
# Stage


def run_rules_stage(
    paths: DataPaths,
    out_dir: Path,
    rules_cfg: dict,
    data_cfg: dict,
    *,
    threads: int | None = None,
    memory_limit: str | None = DEFAULT_MEMORY_LIMIT,
    temp_dir: Path | None = None,
) -> dict:
    """Severities -> hub cap -> tuned thresholds per alert rate -> flags; writes to `out_dir`.

    Outputs: severities.parquet, flags.parquet, thresholds.json, summary.json. Labels: only
    `is_laundering` of the tune-split rows is read, for tuning.
    """
    check_tune_split(rules_cfg, data_cfg)
    out_dir = Path(out_dir)
    timings: dict[str, float] = {}
    t0 = time.perf_counter()
    # DuckDB spills to local disk, never to the Volume.
    own_tmp = temp_dir is None
    spill = Path(tempfile.mkdtemp(prefix="aml-rules-duckdb-")) if own_tmp else Path(temp_dir)
    con = connect(threads, memory_limit, spill)
    try:
        register_transactions(con, paths.transactions)
        quantile = float(rules_cfg["hub_degree_quantile"])
        hub_cap = hub_degree_cap(con, quantile)
        timings["hub_cap"] = time.perf_counter() - t0
        t = time.perf_counter()
        sev = compute_severities(con, rules_cfg, hub_cap)
        stats = severity_stats(con)
        timings["severities"] = time.perf_counter() - t
    finally:
        con.close()
        if own_tmp:
            shutil.rmtree(spill, ignore_errors=True)

    return tune_and_write(
        sev,
        paths,
        out_dir,
        rules_cfg,
        data_cfg,
        hub_cap=hub_cap,
        stats=stats,
        timings=timings,
        t0=t0,
    )


def check_tune_split(rules_cfg: dict, data_cfg: dict) -> None:
    """check_rules_config, and the tune split must be a split of the data config."""
    check_rules_config(rules_cfg)
    tune = rules_cfg["tune_split"]
    if tune not in data_cfg["split"]:
        raise ValueError(f"tune_split {tune!r} is not a split in the data config")


def tune_split_labels(sev: pl.DataFrame, paths: DataPaths, tune: str) -> pl.Series:
    """`is_laundering` aligned with the rows of `sev`, filled only for `tune`-split rows (the other
    rows stay null): the only labels the rules stage reads."""
    labels = pl.read_parquet(paths.labels, columns=["row_id", "is_laundering"])
    tune_labels = (
        sev.filter(pl.col("split") == tune).select("row_id").join(labels, on="row_id", how="inner")
    )
    y = sev.select("row_id").join(tune_labels, on="row_id", how="left", maintain_order="left")
    return y["is_laundering"]


RULES_OUTPUTS = ("severities.parquet", "flags.parquet", "thresholds.json", "summary.json")


def tune_and_write(
    sev: pl.DataFrame,
    paths: DataPaths,
    out_dir: Path,
    rules_cfg: dict,
    data_cfg: dict,
    *,
    hub_cap: int,
    stats: dict[str, int],
    extra_summary: dict[str, Any] | None = None,
    timings: dict[str, float] | None = None,
    t0: float | None = None,
    before_summary: Callable[[dict[str, Any]], dict[str, Any] | None] | None = None,
) -> dict:
    """The rules stage after the severities: tuned thresholds per alert rate -> flags -> files.

    `sev`: row_id, split, day and the SCENARIOS columns (Float64) of every row, in rank order
    (as `compute_severities` returns them). Writes severities.parquet, flags.parquet,
    thresholds.json and summary.json into `out_dir`; labels: only `is_laundering` of the
    tune-split rows is read. `run_rules_stage` (M1, SQL severities) and the M2 engine-severity
    stage share this code, so equal severities give equal files. `extra_summary` is merged into
    summary.json last; `timings` (seconds so far) and `t0` (stage start, `time.perf_counter`)
    keep the stage's timing fields. `before_summary(summary)` runs after the other three files
    are written and before summary.json (the stage's completion marker); a dict it returns is
    merged into the summary.
    """
    check_tune_split(rules_cfg, data_cfg)
    tune = rules_cfg["tune_split"]
    out_dir = Path(out_dir)
    timings = {} if timings is None else timings
    t0 = time.perf_counter() if t0 is None else t0
    quantile = float(rules_cfg["hub_degree_quantile"])

    t = time.perf_counter()
    # Labels: only is_laundering, only for tune-split rows (the other rows stay null).
    y = tune_split_labels(sev, paths, tune)

    rates = alert_rates(rules_cfg)
    headline = rate_tag(rates[0])
    thresholds: dict[str, dict[str, float | None]] = {}
    tuning: dict[str, dict] = {}
    metrics: dict[str, dict] = {}
    flag_cols: list[pl.Series] = []
    headline_fired: pl.DataFrame | None = None
    for r in rates:
        tag = rate_tag(r)
        thr, tuning[tag] = tune_thresholds_detailed(sev, y, rules_cfg, r)
        fired = apply_thresholds(sev, thr)
        thresholds[tag] = thr
        metrics[tag] = {"alert_rate": r, **_rate_metrics(fired["any"], sev["split"], y, tune)}
        flag_cols.append(fired["any"].alias(f"rules_any_{tag}"))
        if tag == headline:
            headline_fired = fired
    assert headline_fired is not None
    diagnostics = scenario_diagnostics(
        sev, y, rules_cfg, thresholds, {rate_tag(r): r for r in rates}
    )
    infeasible = diagnostics["rates"][headline]["infeasible"]
    if infeasible:
        warnings.warn(
            f"rules: {infeasible} can never be switched on at the headline rate {headline}: "
            "their strictest grid value alone exceeds the tune-split alert budget "
            "(see scenario_diagnostics in summary.json)",
            stacklevel=3,
        )
    timings["tuning"] = time.perf_counter() - t

    flags = sev.select("row_id", "split", "day").with_columns(
        *flag_cols, *[headline_fired[f"fired_{s}"] for s in SCENARIOS]
    )
    write_parquet_atomic(sev.select("row_id", *SCENARIOS), out_dir / "severities.parquet")
    write_parquet_atomic(flags, out_dir / "flags.parquet")
    thresholds_doc = {
        "headline_rate_tag": headline,
        "rate_tags": [rate_tag(r) for r in rates],
        "tune_split": tune,
        "hub_degree_quantile": quantile,
        "hub_cap": hub_cap,
        "n_scenarios": len(SCENARIOS),
        "thresholds": thresholds,
        "active_scenarios": {tag: diagnostics["rates"][tag]["active"] for tag in thresholds},
        "tuning": tuning,
        "metrics": {
            tag: {k: v for k, v in m.items() if k != "per_split"} for tag, m in metrics.items()
        },
        "scenario_diagnostics": diagnostics,
    }
    write_json_atomic(thresholds_doc, out_dir / "thresholds.json")
    timings["total"] = time.perf_counter() - t0
    summary = {
        **thresholds_doc,
        "metrics": metrics,
        "rows": sev.height,
        "stats": stats,
        "severity_nonzero_share": {
            s: float((sev[s] > 0).mean()) if sev.height else 0.0 for s in SCENARIOS
        },
        "outputs": {n: str(out_dir / n) for n in RULES_OUTPUTS},
        "timings_s": timings,
    }
    if before_summary is not None:
        summary.update(before_summary(summary) or {})
    if extra_summary:
        summary.update(extra_summary)
    write_json_atomic(summary, out_dir / "summary.json")
    return summary
