"""The evaluation stage: eval frame -> operating points, metrics, breakdowns, bootstrap CIs ->
results.json + results.md (PLAN.md §4 "Metrics").

This is the only place that joins attempt_id / typology to scores. Thresholds come from val_late
only; the test rows are scored once per final model.
"""

from __future__ import annotations

import math
import os
import re
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from aml.config import rate_tag
from aml.eval.bootstrap import STRATA, ReplicateStats, ci, make_clusters, paired_bootstrap
from aml.eval.metrics import (
    ARGMAX_THRESHOLD,
    best_f1_threshold,
    pr_auc,
    prf_at_threshold,
    prf_from_flags,
    roc_auc,
)
from aml.eval.operating_points import (
    alert_budget,
    deployable_point,
    iso_volume_point,
    point_metrics,
    threshold_for_alert_rate,
    union_point,
)
from aml.eval.typology import (
    ALL,
    TYPOLOGIES,
    attempt_detection,
    attempt_ids,
    memorisation_split,
    recall_by_typology,
)
from aml.io import read_json, write_json_atomic, write_text_atomic
from aml.models.gnn import (
    COMPARISON_MODELS,
    FAITHFUL_MODEL,
    GUARD_FIELDS,
    LOOKAHEAD_D10_MODEL,
    add_guard,
    empty_guard,
)
from aml.paths import DataPaths

VAL_SPLIT = "val_late"
TEST_SPLIT = "test"
VIEWS = ("primary", "tail", "full")
EVAL_COLUMNS = [
    "row_id",
    "rank",
    "split",
    "day",
    "minute",
    "src",
    "dst",
    "y",
    "attempt_id",
    "typology",
    "seen_launderer",
]
NA = "n/a"

# Published HI-Small minority-class F1 (%), PLAN.md §12. Not comparable to our table: argmax
# threshold, non-causal sampling, and for the GFP rows probably a different split.
LITERATURE_F1 = (
    ("GIN", 28.7),
    ("PNA", 56.8),
    ("Multi-GIN+EU", 64.8),
    ("Multi-PNA+EU", 68.2),
    ("LightGBM+GFP", 62.9),
    ("XGBoost+GFP", 63.2),
    ("LightGBM, raw transaction features", 21.3),
)
LITERATURE_CAVEATS = (
    "Minority-class F1 at the argmax threshold (0.5), not at a threshold chosen on validation.",
    "GNN rows: non-causal. The published test graph holds all edges and the loaders have no time "
    "constraint; Multi-GNN also drops unsampled target edges from its HI-Small F1.",
    "The GFP rows probably use a different split.",
    "Sources: Altman et al. (arXiv 2306.16424), Egressy et al. (arXiv 2306.11586), "
    "Blanuša et al. (arXiv 2402.08593); see PLAN.md §12.",
)
SYNTHETIC_CAVEATS = (
    "IBM AML HI-Small is synthetic: labels are perfect and instantly available, there is no KYC "
    "or geography, and there may be shortcuts such as payment format (ACH carries most "
    "positives).",
    '"Causal" means features and sampling (as-of rule: only strictly earlier minutes), not '
    "label availability.",
)
# Notes appended by render_markdown when a table uses them.
SEED_MARK = " †"
SEED_NOTE = (
    "† Mean over fewer seeds than the model has: on the other seeds the metric is undefined "
    "(no alerts for precision, no detected attempt for minutes). Per-seed values are in "
    "results.json."
)


def tail_caveat(view: dict[str, Any], pattern_share: float) -> str:
    """The tail caveat from the data: its size, prevalence and pattern-completion share."""
    lo, hi = view["days"]
    text = (
        f"The tail (days {lo}-{hi}) has only {view['rows']:,} transactions, "
        f"{view['positives']:,} of them laundering"
    )
    if view["positives"]:
        text += f" ({100 * view['prevalence']:.0f}% of tail rows"
        if math.isfinite(pattern_share):
            text += f"; {100 * pattern_share:.0f}% of these positives belong to pattern attempts"
        text += ")"
    return text + (
        ", so full-test numbers are dominated by it. The primary period is the headline."
    )


# --------------------------------------------------------------------------- eval frame


def build_eval_frame(paths: DataPaths, data_cfg: dict) -> pl.DataFrame:
    """val_late + test rows sorted by rank, with labels, attempt/typology and seen_launderer.

    seen_launderer = src or dst appears (as src or dst) in a positive train row.
    """
    for s in (VAL_SPLIT, TEST_SPLIT):
        if s not in data_cfg["split"]:
            raise ValueError(f"data_cfg.split has no {s!r}")
    tx = pl.scan_parquet(paths.transactions).select(
        ["row_id", "rank", "split", "day", "minute", "src", "dst"]
    )
    labels = pl.scan_parquet(paths.labels).select(
        ["row_id", "is_laundering", "attempt_id", "typology"]
    )
    train_pos = tx.filter(pl.col("split") == "train").join(
        labels.filter(pl.col("is_laundering") == 1).select("row_id"), on="row_id", how="semi"
    )
    launderers = (
        pl.concat([train_pos.select(acct="src"), train_pos.select(acct="dst")])
        .unique()
        .collect()["acct"]
    )
    ev = (
        tx.filter(pl.col("split").is_in([VAL_SPLIT, TEST_SPLIT]))
        .join(labels, on="row_id", how="left")
        .with_columns(
            y=pl.col("is_laundering").cast(pl.Int8),
            seen_launderer=pl.col("src").is_in(launderers.implode())
            | pl.col("dst").is_in(launderers.implode()),
        )
        .sort("rank")
        .select(EVAL_COLUMNS)
        .collect()
    )
    if ev["y"].null_count():
        raise ValueError("labels are missing for some val_late/test rows")
    return ev


# --------------------------------------------------------------------------- helpers


def _default_threads() -> int:
    try:
        n = len(os.sched_getaffinity(0))
    except AttributeError:
        n = os.cpu_count() or 1
    # The evaluate container requests 4 cores; host CPU counts can be much larger.
    return max(1, min(4, n))


def _aggregate(items: list[Any]) -> Any:
    """Per-seed results (same structure) -> leaves {mean, std, per_seed, n_defined}.

    mean and std are over the seeds where the value is defined (finite); n_defined counts them,
    so a mean over fewer seeds than the model has is visible (the bootstrap uses the same
    convention per replicate). std is None for fewer than 2 defined seeds.
    """
    first = items[0]
    if isinstance(first, dict):
        return {k: _aggregate([it[k] for it in items]) for k in first}
    if isinstance(first, int | float | np.number) and not isinstance(first, bool):
        x = np.array(items, dtype=np.float64)
        fin = x[np.isfinite(x)]
        return {
            "mean": float(fin.mean()) if fin.size else math.nan,
            "std": float(fin.std(ddof=1)) if fin.size > 1 else None,
            "per_seed": [float(v) for v in x],
            "n_defined": int(fin.size),
        }
    return first


def _clean(obj: Any) -> Any:
    """JSON-safe: numpy -> Python, non-finite floats -> None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, list | tuple):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_ | bool):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, float | np.floating):
        return float(obj) if math.isfinite(obj) else None
    return obj


def _view_days(data_cfg: dict, view: str) -> tuple[int, int]:
    lo, hi = data_cfg["test_views"][view]
    return int(lo), int(hi)


def _literature(y: np.ndarray, s: np.ndarray, thr_f1: float) -> dict[str, float]:
    at_thr = prf_at_threshold(y, s, thr_f1)
    at_max = prf_at_threshold(y, s, ARGMAX_THRESHOLD)
    return {
        "f1_thr": at_thr["f1"],
        "precision_thr": at_thr["precision"],
        "recall_thr": at_thr["recall"],
        "alerts_thr": at_thr["alerts"],
        "f1_argmax": at_max["f1"],
        "precision_argmax": at_max["precision"],
        "recall_argmax": at_max["recall"],
        # F1 @ 0.5 of a scale_pos_weight model can flag a large volume; show it.
        "alerts_argmax": at_max["alerts"],
        "pr_auc": pr_auc(y, s),
        "roc_auc": roc_auc(y, s),
    }


def _rate_with_ties(s_val: np.ndarray, rate: float) -> float:
    """The (a) bracket on the other side of threshold_for_alert_rate: the val_late rate if the
    whole boundary tie block at the k-th score were flagged (it then exceeds the budget)."""
    k = alert_budget(s_val.size, rate)
    if k == 0:
        return 0.0
    desc = np.sort(s_val)[::-1]
    return float((s_val >= desc[k - 1]).mean())


# --------------------------------------------------------------------------- evaluate


def evaluate(
    eval_df: pl.DataFrame,
    rule_flags: dict[str, np.ndarray],
    model_scores: dict[str, np.ndarray],
    data_cfg: dict,
    rules_cfg: dict,
    *,
    rule_scenarios: dict[str, np.ndarray] | None = None,
    seeds: dict[str, list[int]] | None = None,
    threads: int | None = None,
) -> dict:
    """All metrics of PLAN.md §4 for the rules and every model; returns a JSON-able dict.

    rule_flags: rate_tag -> Boolean array aligned to eval_df (headline rate required).
    model_scores: name -> raw scores (n_seeds, n_rows) aligned to eval_df.
    rule_scenarios: optional scenario -> Boolean "fired" at the headline rate (breakdown table).
    """
    t0 = time.perf_counter()
    n = eval_df.height
    y = eval_df["y"].to_numpy().astype(np.int8)
    split = eval_df["split"].to_numpy()
    day = eval_df["day"].to_numpy().astype(np.int64)
    minute = eval_df["minute"].to_numpy().astype(np.int64)
    src = eval_df["src"].to_numpy().astype(np.int64)
    attempt = attempt_ids(eval_df["attempt_id"])
    typology = eval_df["typology"]
    seen = eval_df["seen_launderer"].to_numpy().astype(bool)

    headline = float(rules_cfg["alert_rate"])
    rates = {rate_tag(headline): headline}
    skipped = []
    for r in rules_cfg.get("sensitivity_alert_rates", []):
        tag = rate_tag(float(r))
        if tag in rule_flags:
            rates[tag] = float(r)
        else:
            skipped.append(tag)
    head = rate_tag(headline)
    if head not in rule_flags:
        raise ValueError(f"rule_flags has no headline rate {head!r}")
    flags = {tag: np.asarray(rule_flags[tag], dtype=bool) for tag in rates}
    for tag, f in flags.items():
        if f.shape != (n,):
            raise ValueError(f"rule_flags[{tag!r}] has shape {f.shape}, expected ({n},)")

    scores: dict[str, np.ndarray] = {}
    for name, arr in model_scores.items():
        a = np.atleast_2d(np.asarray(arr, dtype=np.float64))
        if a.shape[1] != n:
            raise ValueError(f"{name}: scores have shape {a.shape}, expected (n_seeds, {n})")
        if not np.isfinite(a).all():
            raise ValueError(f"{name}: scores must be finite")
        scores[name] = a

    val = split == VAL_SPLIT
    test = split == TEST_SPLIT
    views = {}
    for v in VIEWS:
        lo, hi = _view_days(data_cfg, v)
        views[v] = (test & (day >= lo) & (day <= hi), lo, hi)

    # Thresholds, from val_late only.
    thresholds: dict[str, dict] = {}
    for name, a in scores.items():
        th: dict[str, Any] = {"rate": {}, "f1": [best_f1_threshold(y[val], s[val]) for s in a]}
        for tag in rates:
            rules_rate = float(flags[tag][val].mean()) if val.any() else math.nan
            per_seed = [threshold_for_alert_rate(s[val], rules_rate) for s in a]
            th["rate"][tag] = {
                "rules_val_late_rate": rules_rate,
                "per_seed": per_seed,
                "model_val_late_rate": [
                    float((s[val] >= t).mean()) if val.any() else math.nan
                    for s, t in zip(a, per_seed, strict=True)
                ],
                # The other bracket: flagging the whole tie block at the cut (over budget).
                "model_val_late_rate_with_ties": [
                    _rate_with_ties(s[val], rules_rate) if val.any() else math.nan for s in a
                ],
            }
        thresholds[name] = th

    results: dict[str, Any] = {
        "meta": {
            "alert_rate": headline,
            "headline_tag": head,
            "rates": rates,
            "skipped_sensitivity_tags": skipped,
            "val_late": {"rows": int(val.sum()), "positives": int(y[val].sum())},
            "views": {},
            "models": {
                name: {
                    "n_seeds": int(a.shape[0]),
                    "seeds": list((seeds or {}).get(name, range(a.shape[0]))),
                }
                for name, a in scores.items()
            },
            "caveats": list(SYNTHETIC_CAVEATS),
        },
        "thresholds": thresholds,
        "views": {},
    }

    for v, (mask, lo, hi) in views.items():
        nd = hi - lo + 1
        yv, srcv, dayv, minv = y[mask], src[mask], day[mask], minute[mask]
        att_v, typ_v, seen_v = attempt[mask], typology.filter(pl.Series(mask)), seen[mask]
        results["meta"]["views"][v] = {
            "days": [lo, hi],
            "n_days": nd,
            "rows": int(mask.sum()),
            "positives": int(yv.sum()),
            "prevalence": float(yv.mean()) if mask.any() else math.nan,
        }
        rf = {tag: f[mask] for tag, f in flags.items()}
        rules_points = {}
        for tag in rates:
            m = point_metrics(yv, rf[tag], srcv, dayv, nd)
            m.update(pr_auc=NA, roc_auc=NA, f1_thr=NA)
            rules_points[tag] = m
        breakdown = {
            "typology": recall_by_typology(rf[head], yv, typ_v),
            "attempts": attempt_detection(rf[head], att_v, typ_v, minv),
            "memorisation": memorisation_split(rf[head], yv, seen_v),
        }
        view_res: dict[str, Any] = {"rules": {"points": rules_points, **breakdown}, "models": {}}
        if rule_scenarios:
            view_res["rules"]["scenarios"] = {
                sc: prf_from_flags(yv, np.asarray(f, dtype=bool)[mask])
                for sc, f in rule_scenarios.items()
            }
        for name, a in scores.items():
            th = thresholds[name]
            per_seed = []
            for i, s_all in enumerate(a):
                s = s_all[mask]
                thr_a = {tag: th["rate"][tag]["per_seed"][i] for tag in rates}
                flag_a = s >= thr_a[head]
                per_seed.append(
                    {
                        "a": {
                            tag: deployable_point(yv, s, thr_a[tag], srcv, dayv, nd)
                            for tag in rates
                        },
                        "b": {
                            tag: iso_volume_point(yv, s, int(rf[tag].sum()), srcv, dayv, nd)
                            for tag in rates
                        },
                        "c": union_point(yv, s, thr_a[head], rf[head], srcv, dayv, nd),
                        "literature": _literature(yv, s, th["f1"][i]),
                        "typology": recall_by_typology(flag_a, yv, typ_v),
                        "attempts": attempt_detection(flag_a, att_v, typ_v, minv),
                        "memorisation": memorisation_split(flag_a, yv, seen_v),
                    }
                )
            view_res["models"][name] = _aggregate(per_seed)
        results["views"][v] = view_res

    tail_mask = views["tail"][0]
    tail_pos = tail_mask & (y == 1)
    pattern_share = float((attempt[tail_pos] >= 0).mean()) if tail_pos.any() else math.nan
    results["meta"]["caveats"].append(tail_caveat(results["meta"]["views"]["tail"], pattern_share))

    ev_cfg = data_cfg.get("evaluation", {})
    B = int(ev_cfg.get("bootstrap_replicates", 0))
    if B > 0:
        results["bootstrap"] = _bootstrap_primary(
            y=y,
            attempt=attempt,
            src=src,
            mask=views["primary"][0],
            rule_flag=flags[head],
            scores=scores,
            thresholds=thresholds,
            head=head,
            B=B,
            seed=int(ev_cfg.get("bootstrap_seed", 0)),
            level=float(ev_cfg.get("ci_level", 0.95)),
            threads=threads if threads is not None else _default_threads(),
        )
        _add_diff_points(results)
    results["meta"]["seconds"] = time.perf_counter() - t0
    return _clean(results)


# Model metrics with a bootstrap CI (primary view, headline rate), and the paired differences.
# M3 appends literature.f1_argmax (F1 at score >= 0.5): an extra statistic on the same
# replicates, so every other key keeps its value (replicates depend only on strata, clusters, B
# and the seed).
BOOT_MODEL_KEYS = (
    "a.precision",
    "a.recall",
    "a.f1",
    "b.precision",
    "b.recall",
    "b.f1",
    "c.union.recall",
    "c.model_same_volume.recall",
    "literature.f1_thr",
    "literature.pr_auc",
    "literature.f1_argmax",
)
BOOT_DIFFS = {
    "a.recall_minus_rules": ("a.recall", "rules.recall"),
    "b.recall_minus_rules": ("b.recall", "rules.recall"),
    "c.union_minus_model_same_volume": ("c.union.recall", "c.model_same_volume.recall"),
}
# Paired model - model differences (every pair, later model minus earlier), same replicates.
# results.md shows the M2 columns unless GNN models are evaluated (M2 output stays byte-identical).
M2_MODEL_DIFF_KEYS = (
    "a.recall",
    "a.precision",
    "b.recall",
    "literature.f1_thr",
    "literature.pr_auc",
)
MODEL_DIFF_KEYS = (*M2_MODEL_DIFF_KEYS, "literature.f1_argmax")
MODEL_DIFF_HEADERS = {
    "a.recall": "Recall (a), pp",
    "a.precision": "Precision (a), pp",
    "b.recall": "Recall (b), pp",
    "literature.f1_thr": "F1 @ val_late-best threshold, pp",
    "literature.pr_auc": "PR-AUC, pp",
    "literature.f1_argmax": "F1 @ 0.5 (argmax), pp",
}


def _interval(samples: np.ndarray, level: float) -> dict[str, float | int]:
    """CI over the finite replicates plus how many replicates were undefined (nan)."""
    lo, hi = ci(samples, level)
    return {"lo": lo, "hi": hi, "undefined": int((~np.isfinite(samples)).sum())}


def _seed_mean(stack: list[np.ndarray]) -> np.ndarray:
    """Per-replicate mean over the seeds where the metric is defined (as `_aggregate`)."""
    arr = np.stack(stack)
    fin = np.isfinite(arr)
    n = fin.sum(axis=0)
    total = np.where(fin, arr, 0.0).sum(axis=0)
    return np.divide(total, n, out=np.full(arr.shape[1], np.nan), where=n > 0)


def _model_pairs(names: list[str]) -> list[tuple[str, str]]:
    return [(names[j], names[i]) for i in range(len(names)) for j in range(i + 1, len(names))]


def _bootstrap_primary(
    *,
    y: np.ndarray,
    attempt: np.ndarray,
    src: np.ndarray,
    mask: np.ndarray,
    rule_flag: np.ndarray,
    scores: dict[str, np.ndarray],
    thresholds: dict[str, dict],
    head: str,
    B: int,
    seed: int,
    level: float,
    threads: int,
) -> dict:
    """Paired stratified cluster bootstrap on the primary view.

    Per replicate: (a) keeps each seed's val_late threshold, K for (b) and |union| for (c) are
    recomputed, and a multi-seed model's value is the mean of its per-seed values.
    """
    t0 = time.perf_counter()
    yv = y[mask]
    strata, cluster_index = make_clusters(yv, attempt[mask], src[mask])
    rs = ReplicateStats(yv, cluster_index)
    rf = rule_flag[mask]
    k_rules = rs.linear(rf)
    stat_fns: dict[str, Callable] = {"rules": rs.prf(rf)}

    def seed_stat(s: np.ndarray, thr_a: float, thr_f1: float) -> Callable:
        flag_a = s >= thr_a
        union = rf | flag_a
        parts = {
            "a": rs.prf(flag_a),
            "b": rs.top_k(s, k_rules),
            "c.union": rs.prf(union),
            "c.model_same_volume": rs.top_k(s, rs.linear(union)),
            "literature.f1": rs.prf(s >= thr_f1),
            "literature.argmax": rs.prf(s >= ARGMAX_THRESHOLD),
        }
        ap = rs.pr_auc(s)

        def fn(counts: np.ndarray) -> dict[str, np.ndarray]:
            out = {}
            for p, f in parts.items():
                for k, v in f(counts).items():
                    out[f"{p}.{k}"] = v
            out["literature.f1_thr"] = out.pop("literature.f1.f1")
            out["literature.f1_argmax"] = out.pop("literature.argmax.f1")
            out["literature.pr_auc"] = ap(counts)
            return out

        return fn

    for name, a in scores.items():
        th = thresholds[name]
        for i, s_all in enumerate(a):
            stat_fns[f"{name}#{i}"] = seed_stat(
                s_all[mask], th["rate"][head]["per_seed"][i], th["f1"][i]
            )
    samples = paired_bootstrap(stat_fns, strata, cluster_index, B, seed, threads=threads)

    rules = {k: samples[f"rules.{k}"] for k in ("precision", "recall", "f1")}
    out: dict[str, Any] = {
        "view": "primary",
        "rate_tag": head,
        "B": B,
        "seed": seed,
        "level": level,
        "strata": {
            name: {
                "rows": int((strata == i).sum()),
                "clusters": int(np.unique(cluster_index[strata == i]).size),
            }
            for i, name in enumerate(STRATA)
        },
        "rules": {k: _interval(v, level) for k, v in rules.items()},
        "models": {},
        "diffs": {},
        "model_diffs": {},
    }
    means: dict[str, dict[str, np.ndarray]] = {}
    for name, a in scores.items():
        n_seeds = a.shape[0]
        mean = {
            key: _seed_mean([samples[f"{name}#{i}.{key}"] for i in range(n_seeds)])
            for key in BOOT_MODEL_KEYS
        }
        means[name] = mean
        out["models"][name] = {k: _interval(v, level) for k, v in mean.items()}
        both = {**mean, **{f"rules.{k}": v for k, v in rules.items()}}
        out["diffs"][name] = {
            d: _interval(both[x] - both[z], level) for d, (x, z) in BOOT_DIFFS.items()
        }
    for m, r in _model_pairs(list(scores)):
        out["model_diffs"][f"{m} - {r}"] = {
            k: _interval(means[m][k] - means[r][k], level) for k in MODEL_DIFF_KEYS
        }
    out["seconds"] = time.perf_counter() - t0
    return out


def _add_diff_points(results: dict) -> None:
    """Point estimates of the paired differences, from the main (non-bootstrap) numbers."""
    head = results["meta"]["headline_tag"]
    pv = results["views"]["primary"]
    rules_recall = pv["rules"]["points"][head]["recall"]
    for name, m in pv["models"].items():
        vals = {
            "a.recall": m["a"][head]["recall"]["mean"],
            "b.recall": m["b"][head]["recall"]["mean"],
            "c.union.recall": m["c"]["union"]["recall"]["mean"],
            "c.model_same_volume.recall": m["c"]["model_same_volume"]["recall"]["mean"],
            "rules.recall": rules_recall,
        }
        for d, (x, z) in BOOT_DIFFS.items():
            results["bootstrap"]["diffs"][name][d]["point"] = _minus(vals[x], vals[z])
    for m, r in _model_pairs(list(pv["models"])):
        a, b = pv["models"][m], pv["models"][r]
        for k, iv in results["bootstrap"]["model_diffs"][f"{m} - {r}"].items():
            iv["point"] = _minus(_mean_at(a, k, head), _mean_at(b, k, head))


def _mean_at(model: dict, key: str, head: str) -> float | None:
    """Seed mean of a dotted metric key; (a)/(b) keys are read at the headline rate."""
    part, metric = key.split(".", 1)
    node = model[part][head] if part in ("a", "b") else model[part]
    return node[metric]["mean"]


def _minus(x: float | None, z: float | None) -> float | None:
    if x is None or z is None:
        return None
    return x - z


# --------------------------------------------------------------------------- markdown


def _val(v: Any) -> float | None:
    if isinstance(v, dict):
        v = v.get("mean")
    if v is None or isinstance(v, str):
        return None
    v = float(v)
    return v if math.isfinite(v) else None


def _seed_mark(v: Any) -> str:
    """SEED_MARK when a per-seed mean covers fewer seeds than the model has."""
    if isinstance(v, dict) and "per_seed" in v:
        n = v.get("n_defined")
        if n is not None and 0 < n < len(v["per_seed"]):
            return SEED_MARK
    return ""


def _pct(v: Any, digits: int = 1) -> str:
    """Percent with mean ± std for per-seed aggregates; "n/a" for missing values."""
    x = _val(v)
    if x is None:
        return NA
    out = f"{100 * x:.{digits}f}"
    if isinstance(v, dict) and v.get("std") is not None:
        out += f" ± {100 * v['std']:.{digits}f}"
    return out + _seed_mark(v)


def _num(v: Any, digits: int = 1) -> str:
    x = _val(v)
    if x is None:
        return NA
    out = f"{x:,.0f}" if float(x).is_integer() else f"{x:,.{digits}f}"
    return out + _seed_mark(v)


def _ci(b: dict | None, key: str) -> str:
    if not b or key not in b:
        return ""
    iv = b[key]
    undefined = iv.get("undefined") or 0
    if iv.get("lo") is None:
        return f" [CI n/a: undefined in {undefined} replicates]" if undefined else ""
    out = f" [{100 * iv['lo']:.1f}, {100 * iv['hi']:.1f}"
    return out + (f"; {undefined} replicates undefined]" if undefined else "]")


def rules_label(results: dict, tag: str | None = None) -> str:
    """'Rules (SQL, k of n scenarios active)' at `tag` (headline by default), from the tuned
    thresholds; plain 'Rules (SQL)' when run_evaluate_stage had no thresholds.json."""
    info = results["meta"].get("rules")
    if not info:
        return "Rules (SQL)"
    tag = tag or results["meta"]["headline_tag"]
    active = info["active"].get(tag)
    if active is None:
        return "Rules (SQL)"
    return f"Rules (SQL, {len(active)} of {info['n_scenarios']} scenarios active)"


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    lines = ["| " + " | ".join(header) + " |", "|" + "|".join(" --- " for _ in header) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return lines


def _label(results: dict, name: str) -> str:
    n = results["meta"]["models"][name]["n_seeds"]
    return f"{name} (1 seed, no std)" if n == 1 else f"{name} ({n} seeds)"


def _has_gnn(results: dict) -> bool:
    """GNN models were evaluated (M3): only then does results.md differ from M2's layout."""
    return bool(results.get("gnn"))


def _models_for(results: dict, view: str) -> list[str]:
    """The models rendered in `view`'s tables (meta.model_views; every model by default).

    gnn_lookahead_d10 is evaluated on every view (its tail rows carry causal-bound scores, so
    every score is finite) but rendered on the primary period only."""
    mv = results["meta"].get("model_views") or {}
    return [m for m in results["meta"]["models"] if view in mv.get(m, VIEWS)]


def _ops_rows(results: dict, view: str, with_ci: bool) -> list[list[str]]:
    head = results["meta"]["headline_tag"]
    vr = results["views"][view]
    shown = set(_models_for(results, view))
    boot = results.get("bootstrap") if with_ci else None
    rb = boot["rules"] if boot else None
    r = vr["rules"]["points"][head]
    rows = [
        [
            rules_label(results),
            "own operating point",
            _num(r["alerts"]),
            _num(r["alerts_per_day"]),
            _num(r["alerted_accounts_per_day"]),
            _pct(r["precision"]) + _ci(rb, "precision"),
            _pct(r["recall"]) + _ci(rb, "recall"),
            _pct(r["f1"]) + _ci(rb, "f1"),
            NA,
        ]
    ]
    for name, m in vr["models"].items():
        if name not in shown:
            continue
        mb = boot["models"].get(name) if boot else None
        lit = m["literature"]
        points = [
            ("(a) deployable threshold", m["a"][head], "a"),
            ("(b) top-K, K = rules' alerts", m["b"][head], "b"),
            ("(c) rules ∪ model (a)", m["c"]["union"], "c.union"),
            ("(c) model alone, same volume", m["c"]["model_same_volume"], "c.model_same_volume"),
        ]
        for label, p, key in points:
            rows.append(
                [
                    _label(results, name),
                    label,
                    _num(p["alerts"]),
                    _num(p["alerts_per_day"]),
                    _num(p["alerted_accounts_per_day"]),
                    _pct(p["precision"]) + _ci(mb, f"{key}.precision"),
                    _pct(p["recall"]) + _ci(mb, f"{key}.recall"),
                    _pct(p["f1"]) + _ci(mb, f"{key}.f1"),
                    _pct(lit["pr_auc"]) + _ci(mb, "literature.pr_auc") if key == "a" else "",
                ]
            )
    return rows


OPS_HEADER = [
    "Model",
    "Operating point",
    "Alerts",
    "Alerts/day",
    "Alerted accounts/day",
    "Precision %",
    "Recall %",
    "F1 %",
    "PR-AUC %",
]


def _view_title(results: dict, view: str) -> str:
    vm = results["meta"]["views"][view]
    lo, hi = vm["days"]
    return (
        f"days {lo}-{hi}: {vm['rows']:,} transactions, {vm['positives']:,} positives "
        f"({_pct(vm['prevalence'], 3)}%)"
    )


def render_markdown(results: dict) -> str:
    """results.json -> results.md: headline (primary) table, other views, sensitivity,
    typology / attempt / memorisation breakdowns, and the separate literature table."""
    meta = results["meta"]
    head = meta["headline_tag"]
    boot = results.get("bootstrap")
    models = list(meta["models"])
    L: list[str] = ["# Evaluation results", ""]
    L += [
        f"One alert = one flagged transaction. Alert budget = {100 * meta['alert_rate']:g}% of "
        "transactions (`configs/rules.yaml: alert_rate`, an analyst-capacity assumption). "
        "Model thresholds are chosen on val_late only and applied unchanged to test. "
        "Percentages are × 100; multi-seed models show mean ± std over seeds.",
        "",
    ]
    if boot:
        L += [
            f"Brackets: {100 * boot['level']:g}% CI from a paired stratified cluster bootstrap "
            f"(B = {boot['B']:,}, seed {boot['seed']}) on the primary period: pattern positives "
            "resampled by attempt, other positives and negatives by sender account; (a) keeps its "
            "threshold, K is recomputed per replicate; multi-seed = mean over seeds per replicate.",
            "",
        ]
    L += ["## Headline: primary period", "", _view_title(results, "primary"), ""]
    L += _table(OPS_HEADER, _ops_rows(results, "primary", with_ci=True))
    L += [""]
    for name in models:
        th = results["thresholds"][name]["rate"][head]
        mv = [_pct(x, 3) for x in th["model_val_late_rate"]]
        L.append(
            f"- {name}: val_late alert rate at (a) = {', '.join(mv)}% per seed "
            f"(rules: {_pct(th['rules_val_late_rate'], 3)}%; tied scores at the cut are left out)."
        )
    L.append("")
    if boot:
        L += ["### Paired differences (primary)", ""]
        rows = []
        for name in models:
            d = boot["diffs"][name]
            rows.append(
                [_label(results, name)]
                + [
                    _pct(d[k].get("point")) + _ci(d, k)
                    for k in (
                        "a.recall_minus_rules",
                        "b.recall_minus_rules",
                        "c.union_minus_model_same_volume",
                    )
                ]
            )
        L += _table(
            [
                "Model",
                "Recall (a) − rules, pp",
                "Recall (b) − rules, pp",
                "Recall union − model at same volume, pp",
            ],
            rows,
        )
        L += [""]
        if boot.get("model_diffs"):
            diff_keys = MODEL_DIFF_KEYS if _has_gnn(results) else M2_MODEL_DIFF_KEYS
            L += ["### Paired model differences (primary)", ""]
            L += _table(
                ["Models", *(MODEL_DIFF_HEADERS[k] for k in diff_keys)],
                [
                    [pair] + [_pct(d[k].get("point")) + _ci(d, k) for k in diff_keys]
                    for pair, d in boot["model_diffs"].items()
                ],
            )
            L += ["", "Same replicates for every model: the later model minus the earlier one.", ""]

    L += _literature_section(results)

    for v in ("tail", "full"):
        L += [f"## Test view: {v}", "", _view_title(results, v), ""]
        L += _table(OPS_HEADER, _ops_rows(results, v, with_ci=False)) + [""]

    L += _sensitivity_section(results)

    pv = results["views"]["primary"]
    if pv["rules"].get("scenarios"):
        L += [f"## Rule scenarios at the headline budget (primary period, {head})", ""]
        rules_info = meta.get("rules") or {}
        active = (rules_info.get("active") or {}).get(head)
        infeasible = set((rules_info.get("infeasible") or {}).get(head) or [])
        rows = []
        for sc, m in pv["rules"]["scenarios"].items():
            if active is not None and sc not in active:
                why = "cannot fit the budget" if sc in infeasible else "not selected by tuning"
                rows.append([sc, f"off ({why})", NA, NA, NA])
            else:
                rows.append(
                    [sc, _num(m["alerts"]), _num(m["tp"]), _pct(m["precision"]), _pct(m["recall"])]
                )
        L += _table(["Scenario", "Alerts", "True positives", "Precision %", "Recall %"], rows)
        L += [""]
        if infeasible:
            L += [
                "A scenario that cannot fit the budget flags more val_early rows at its strictest "
                "grid value than the whole alert budget (e.g. fan-out velocity around hub "
                "senders), so tuning can never switch it on; see `scenario_diagnostics` in the "
                "rules stage's summary.json.",
                "",
            ]

    L += ["## Recall per typology at (a)", ""]
    for v in VIEWS:
        vr = results["views"][v]
        vm = _models_for(results, v)
        rows = []
        for t in (*TYPOLOGIES, ALL):
            rt = vr["rules"]["typology"][t]
            row = [t, _num(rt["positives"]), _pct(rt["recall"])]
            row += [_pct(vr["models"][n]["typology"][t]["recall"]) for n in vm]
            rows.append(row)
        L += [f"**{v}** ({_view_title(results, v)})", ""]
        L += _table(["Typology", "Positives", "Rules %", *[f"{n} %" for n in vm]], rows)
        L += [""]

    L += [
        "## Attempt-level detection at (a)",
        "",
        "An attempt counts if it has at least one row in the view; it is detected if any of its "
        "rows is alerted. Minutes = median time from the attempt's first row in the view to its "
        "first alert (detected attempts only; for multi-seed models the mean over seeds of the "
        "per-seed median).",
        "",
    ]
    for v in VIEWS:
        vr = results["views"][v]
        vm = _models_for(results, v)
        rows = []
        for t in (*TYPOLOGIES[:-1], ALL):
            ra = vr["rules"]["attempts"][t]
            row = [
                t,
                _num(ra["attempts"]),
                _pct(ra["detection_rate"]),
                _num(ra["median_minutes_to_first_alert"]),
            ]
            for n in vm:
                ma = vr["models"][n]["attempts"][t]
                row += [_pct(ma["detection_rate"]), _num(ma["median_minutes_to_first_alert"])]
            rows.append(row)
        hdr = ["Typology", "Attempts", "Rules detected %", "Rules minutes"]
        for n in vm:
            hdr += [f"{n} detected %", f"{n} minutes"]
        L += [f"**{v}**", ""] + _table(hdr, rows) + [""]

    L += [
        "## Memorisation check at (a)",
        "",
        "Positives split by whether they touch an account that laundered in train.",
        "",
    ]
    for v in VIEWS:
        vr = results["views"][v]
        vm = _models_for(results, v)
        rows = []
        for g in ("seen", "unseen"):
            rm = vr["rules"]["memorisation"][g]
            row = [g, _num(rm["positives"]), _pct(rm["recall"])]
            row += [_pct(vr["models"][n]["memorisation"][g]["recall"]) for n in vm]
            rows.append(row)
        L += [f"**{v}**", ""]
        L += _table(["Positives", "Count", "Rules recall %", *[f"{n} recall %" for n in vm]], rows)
        L += [""]

    L += _gnn_sections(results)
    L += _validation_section(results)

    lo, hi = meta["views"]["full"]["days"]
    L += [
        "## Literature reference (published numbers, not directly comparable)",
        "",
    ]
    L += _table(
        ["Method (published)", "HI-Small minority-class F1 %"],
        [[m, f"{f1:.1f}"] for m, f1 in LITERATURE_F1],
    )
    period = (
        "Published figures are on the whole post-validation test period, which is our full view "
        f"(days {lo}-{hi}, tail included): compare them with our full-view F1 @ 0.5 row, not "
        "the primary one."
    )
    L += [""] + [f"- {c}" for c in (period, *LITERATURE_CAVEATS)] + [""]
    L += ["## Caveats", ""] + [f"- {c}" for c in meta["caveats"]]
    if meta.get("skipped_sensitivity_tags"):
        L.append(
            "- Sensitivity rates without rule flags were skipped: "
            + ", ".join(meta["skipped_sensitivity_tags"])
        )
    if any(SEED_MARK in line for line in L):
        L.append(f"- {SEED_NOTE}")
    return "\n".join(L) + "\n"


LITERATURE_HEADER = [
    "Model",
    "F1 @ val_late-best threshold",
    "F1 @ 0.5 (argmax)",
    "Precision @ 0.5",
    "Recall @ 0.5",
    "Alerts @ 0.5",
    "Precision @ thr",
    "Recall @ thr",
    "PR-AUC",
    "ROC-AUC",
]


def _literature_rows(results: dict, view: str, with_ci: bool) -> list[list[str]]:
    boot = results.get("bootstrap") if with_ci else None
    mb_all = boot["models"] if boot else {}
    # M2 rendered no CI for F1 @ 0.5; keep its layout unless GNN models are evaluated.
    argmax_ci = _has_gnn(results)
    rows = [[rules_label(results)] + [NA] * (len(LITERATURE_HEADER) - 1)]
    for name in _models_for(results, view):
        lit = results["views"][view]["models"][name]["literature"]
        mb = mb_all.get(name)
        rows.append(
            [
                _label(results, name),
                _pct(lit["f1_thr"]) + _ci(mb, "literature.f1_thr"),
                _pct(lit["f1_argmax"]) + (_ci(mb, "literature.f1_argmax") if argmax_ci else ""),
                _pct(lit["precision_argmax"]),
                _pct(lit["recall_argmax"]),
                _num(lit["alerts_argmax"]),
                _pct(lit["precision_thr"]),
                _pct(lit["recall_thr"]),
                _pct(lit["pr_auc"]) + _ci(mb, "literature.pr_auc"),
                _pct(lit["roc_auc"]),
            ]
        )
    return rows


def _literature_section(results: dict) -> list[str]:
    """Literature-comparable metrics for the primary view (headline, with CIs) and the full test
    view, which is the period the published numbers use."""
    views = results["meta"]["views"]
    plo, phi = views["primary"]["days"]
    flo, fhi = views["full"]["days"]
    L = ["## Literature-comparable metrics", ""]
    L += [f"**primary** (headline, days {plo}-{phi})", ""]
    L += _table(LITERATURE_HEADER, _literature_rows(results, "primary", with_ci=True)) + [""]
    L += [
        f"**full** (days {flo}-{fhi}: the test period the published numbers use; no CI, the "
        "bootstrap covers the primary period only)",
        "",
    ]
    L += _table(LITERATURE_HEADER, _literature_rows(results, "full", with_ci=False))
    L += [
        "",
        "Rules are binary, so PR-AUC and threshold F1 are n/a; their F1 at their own operating "
        "point is in the headline table. ROC-AUC is shown, not headlined. Compare published "
        "argmax F1 with the full-view F1 @ 0.5.",
        "",
    ]
    return L


def _finite_mean(values: list) -> float | None:
    x = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return sum(x) / len(x) if x else None


def _threshold_warnings(results: dict) -> list[str]:
    """(a) seeds whose deployable threshold flags nothing, or clearly less than the rules' rate,
    on val_late because of tied scores at the cut; quotes the with-ties bracket."""
    out = []
    for name in results["meta"]["models"]:
        seeds = results["meta"]["models"][name]["seeds"]
        for tag, th in results["thresholds"][name]["rate"].items():
            rules_rate = th["rules_val_late_rate"]
            if not rules_rate:
                continue
            with_ties = th.get("model_val_late_rate_with_ties") or [None] * len(th["per_seed"])
            for i, (thr, rate) in enumerate(
                zip(th["per_seed"], th["model_val_late_rate"], strict=True)
            ):
                seed = seeds[i] if i < len(seeds) else i
                bracket = with_ties[i]
                other = (
                    f"; flagging the whole tie block would give {_pct(bracket, 3)}%"
                    if bracket is not None
                    else ""
                )
                if thr is None:
                    out.append(
                        f"- {name} seed {seed} at {tag}: (a) flags nothing, because the top tie "
                        f"block of val_late scores exceeds the budget of {_pct(rules_rate, 3)}%"
                        f"{other}."
                    )
                elif rate is not None and rate < 0.9 * rules_rate:
                    out.append(
                        f"- {name} seed {seed} at {tag}: (a) alerts on {_pct(rate, 3)}% of "
                        f"val_late, under 90% of the rules' {_pct(rules_rate, 3)}%, because tied "
                        f"scores at the cut are left out{other}."
                    )
    return out


def _sensitivity_section(results: dict) -> list[str]:
    meta = results["meta"]
    models = list(meta["models"])
    pv = results["views"]["primary"]
    rules_info = meta.get("rules") or {}
    first = next(iter(results["thresholds"].values()), None)
    L = ["## Sensitivity to the alert budget (primary period)", ""]
    rows = []
    for tag, rate in meta["rates"].items():
        r = pv["rules"]["points"][tag]
        active = (rules_info.get("active") or {}).get(tag)
        rules_val = first["rate"][tag]["rules_val_late_rate"] if first else None
        row = [
            f"{100 * rate:g}%",
            f"{len(active)} of {rules_info['n_scenarios']}" if active is not None else NA,
            _pct(rules_val, 3),
            _num(r["alerts"]),
            _pct(r["recall"]),
        ]
        for name in models:
            m = pv["models"][name]
            th = results["thresholds"][name]["rate"][tag]
            row += [
                _pct(_finite_mean(th["model_val_late_rate"]), 3),
                _num(m["a"][tag]["alerts"]),
                _pct(m["a"][tag]["recall"]),
                _pct(m["b"][tag]["recall"]),
            ]
        rows.append(row)
    hdr = [
        "Alert rate",
        "Rule scenarios active",
        "Rules val_late rate %",
        "Rules alerts",
        "Rules recall %",
    ]
    for name in models:
        hdr += [
            f"{name} (a) val_late rate %",
            f"{name} (a) alerts",
            f"{name} (a) recall %",
            f"{name} (b) recall %",
        ]
    L += _table(hdr, rows) + [""]
    L += [
        "(a) matches the rules' val_late alert rate; tied scores at the cut are left out, so its "
        "realised rate can be lower. (b) uses exactly the rules' test alert count.",
        "",
    ]
    warn = _threshold_warnings(results)
    if warn:
        L += ["Operating-point warnings:", "", *warn, ""]
    return L


# --------------------------------------------------------------------------- validation extras

# Validation-only M2 inputs of results.md (run_evaluate_stage `extras`): kind -> the stage file.
#   gate      lgbm_graph/gate.json          (aml.features.gate.run_gate)
#   ablation  lgbm_graph/ablation.json      (aml.models.lgbm_graph)
#   shap      lgbm_graph/shap_global.json   (aml.models.importance.shap_global)
#   engine    features/summary.json         (build_features; rendered generically)
#   parity    rules_engine/parity.json      (the rules engine stage; rendered generically)
EXTRA_KINDS = ("gate", "ablation", "shap", "engine", "parity")
SHAP_TOP = 20
MAX_SUMMARY_ITEMS = 40
# parity.json keys tried, in order, for the one-line summary (the table lists every scalar).
PARITY_MISMATCH_KEYS = ("mismatches_total", "total_mismatches", "n_mismatches", "mismatches")
PARITY_ROWS_KEYS = ("rows_total", "n_rows", "rows")
PARITY_TRUNC_KEYS = ("rule_trunc_rows", "truncated_rows", "n_truncated", "rule_trunc")


def _scalar_items(doc: Any, prefix: str = "", depth: int = 3) -> list[list[Any]]:
    """[dotted key, value] for the scalar leaves of nested dicts (lists are skipped)."""
    out: list[list[Any]] = []
    if not isinstance(doc, dict):
        return out
    for k, v in doc.items():
        key = f"{prefix}.{k}" if prefix else str(k)
        if isinstance(v, dict):
            if depth > 1:
                out += _scalar_items(v, key, depth - 1)
        elif v is None or isinstance(v, str | bool | int | float):
            out.append([key, v])
    return out[:MAX_SUMMARY_ITEMS]


def _int_total(v: Any) -> int | None:
    """An int, or the sum of the ints in a (nested) dict of ints; else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, dict) and v:
        parts = [_int_total(x) for x in v.values()]
        return None if any(p is None for p in parts) else sum(parts)
    return None


def _first_total(doc: dict, keys: tuple[str, ...]) -> int | None:
    for k in keys:
        if k not in doc:
            continue
        v = doc[k]
        # {"rows": n, "share": ...} (the rules engine stage's rule_trunc block): its row count.
        if isinstance(v, dict) and isinstance(v.get("rows"), int):
            v = v["rows"]
        if (t := _int_total(v)) is not None:
            return t
    return None


# build_features summary.json: identity fields shown before its flat `headline` block.
ENGINE_IDENTITY_KEYS = ("engine_version", "spec_hash", "n_accounts", "hub_cap", "n_hubs")
# parity.json of the rules engine stage (the dotted fields shown; flag agreement is added).
PARITY_ITEM_KEYS = (
    "ok",
    "rows",
    "mismatches_total",
    "mismatches_validation",
    "rule_trunc.rows",
    "rule_trunc.share",
    "rule_trunc.round_trip_lower_than_sql",
    "inflow_c_max",
    "hubs.equal",
    "hubs.hub_cap_engine",
    "hubs.n_hubs_engine",
    "m1_regression.compared",
    "m1_regression.thresholds_equal",
    "m1_regression.flags_equal",
    "m1_regression.severities_equal",
)


def _dotted(doc: dict, key: str) -> tuple[bool, Any]:
    cur: Any = doc
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return False, None
        cur = cur[part]
    return True, cur


def _engine_items(doc: dict) -> list[list[Any]]:
    """The key numbers of a build_features summary (its `headline` block), else every scalar."""
    head = doc.get("headline")
    if not isinstance(head, dict):
        return _scalar_items(doc)
    ident = [[k, doc[k]] for k in ENGINE_IDENTITY_KEYS if k in doc]
    return ident + _scalar_items(head)


def _parity_items(doc: dict) -> list[list[Any]]:
    """The rules engine stage's parity fields (with flag agreement per rate), else every scalar."""
    if "mismatches_total" not in doc:
        return _scalar_items(doc)
    out = []
    for key in PARITY_ITEM_KEYS:
        found, v = _dotted(doc, key)
        if found and (v is None or isinstance(v, str | bool | int | float)):
            out.append([key, v])
    for tag, per in sorted((doc.get("flag_agreement") or {}).items()):
        for split, v in (per or {}).items():
            out.append([f"flag_agreement.{tag}.{split}", v])
    return out


def parity_line(doc: dict) -> str:
    """One line from parity.json: mismatches over rows, truncated rows excluded."""
    mis = _first_total(doc, PARITY_MISMATCH_KEYS)
    rows = _first_total(doc, PARITY_ROWS_KEYS)
    trunc = _first_total(doc, PARITY_TRUNC_KEYS)
    if mis is None:
        return "Rule parity (engine severities vs the M1 SQL): see parity.json (items below)."
    text = f"Rule parity (engine severities vs the M1 SQL): {mis:,} mismatches"
    if rows is not None:
        text += f" over {rows:,} rows"
    if trunc is not None:
        text += f" ({trunc:,} rows with rule_trunc = 1 are excluded)"
    return text + "."


def _compact_gate(doc: dict) -> dict:
    keep = ("group", "psi", "psi_warmup", "oor", "nonzero", "drop", "reasons")
    return {
        "config": doc["config"],
        "rows": doc["rows"],
        "order": list(doc.get("order") or doc["features"]),
        "features": {n: {k: r.get(k) for k in keep} for n, r in doc["features"].items()},
        "kept": doc["kept"],
        "dropped": doc["dropped"],
        "dropped_engine": doc["dropped_engine"],
        "n_engine": doc["n_engine"],
        "drop_share": doc["drop_share"],
        "stop": doc["stop"],
    }


def _compact_ablation(doc: dict) -> dict:
    keep = ("ap", "mean", "std", "delta", "decision", "n_features", "removed", "added")
    out = {
        k: doc.get(k)
        for k in (
            "seeds",
            "margin_std",
            "sigma",
            "threshold",
            "champion",
            "dropped_group",
            "best_group_variant",
            "best_delta",
            "caveat",
        )
    }
    out["variants"] = {v: {k: s.get(k) for k in keep} for v, s in doc["variants"].items()}
    out["order"] = list(doc.get("order") or doc["variants"])
    return out


def _compact_shap(doc: dict) -> dict:
    top = [{"name": n, **doc["features"][n]} for n in doc["ranking"][:SHAP_TOP]]
    groups = [  # a list: the ranking survives JSON's sorted keys
        {"group": g, **{k: doc["groups"][g][k] for k in ("mean_abs", "mean_abs_pos", "n_features")}}
        for g in doc["group_ranking"]
    ]
    keys = (
        "space",
        "n_rows",
        "n_positives",
        "n_negatives",
        "negatives_requested",
        "budget_limited",
        "model_seed",
        "variant",
        "split",
    )
    return {**{k: doc.get(k) for k in keys}, "top": top, "groups": groups}


_COMPACT: dict[str, Callable[[dict], dict]] = {
    "gate": _compact_gate,
    "ablation": _compact_ablation,
    "shap": _compact_shap,
    "engine": lambda doc: {"items": _engine_items(doc)},
    "parity": lambda doc: {"line": parity_line(doc), "items": _parity_items(doc)},
}


def load_validation(extras: dict[str, Path]) -> dict:
    """The validation-only inputs, compacted for results.json (a missing file raises)."""
    unknown = sorted(set(extras) - set(EXTRA_KINDS))
    if unknown:
        raise ValueError(f"unknown extras {unknown}; expected some of {EXTRA_KINDS}")
    out = {}
    for kind in EXTRA_KINDS:
        if kind in extras:
            path = Path(extras[kind])
            out[kind] = {"path": str(path), **_COMPACT[kind](read_json(path))}
    return out


def _f(v: Any, fmt: str) -> str:
    return NA if v is None or (isinstance(v, float) and not math.isfinite(v)) else format(v, fmt)


def _gate_lines(g: dict) -> list[str]:
    cfg = g["config"]
    (wl, wh), (ul, uh) = cfg["psi_train_days"], cfg["warmup_days"]
    n = len(g["order"])
    L = ["### Feature gate (label-free, before any fit)", ""]
    L += [
        f"Kept {len(g['kept'])} of {n} model inputs. Dropped engine features: "
        f"{len(g['dropped_engine'])} of {g['n_engine']} ({100 * g['drop_share']:.1f}%; the gate "
        f"stops for a decision above {100 * cfg['max_drop_share']:g}%). A feature is dropped if "
        f"PSI(warm train days {wl}-{wh} vs val_early) > {cfg['psi_max']:g}, if more than "
        f"{100 * cfg['out_of_range_max']:g}% of its val_early values fall outside the warm-train "
        f"range, or if fewer than {cfg['min_train_nonzero']:,} train rows are non-zero. The "
        f"warm-up PSI (days {ul}-{uh} vs {wl}-{wh}) is a diagnostic only.",
        "",
    ]
    rows = [
        [
            name,
            r["group"],
            _f(r["psi"], ".4f"),
            _f(r["psi_warmup"], ".4f"),
            _f(None if r["oor"] is None else 100 * r["oor"], ".2f"),
            _f(r["nonzero"], ","),
            f"dropped ({', '.join(r['reasons'])})" if r["drop"] else "kept",
        ]
        for name, r in ((n, g["features"][n]) for n in g["order"])
    ]
    hdr = ["Feature", "Group", "PSI", "Warm-up PSI", "Out of range %", "Non-zero train rows"]
    L += _table([*hdr, "Gate"], rows) + [""]
    return L


def _gated_out(variant: str, s: dict) -> bool:
    """A -G variant whose inputs equal full's: the gate removed the whole group, so its fits are
    full's fits and its Δ of 0 measures nothing."""
    return variant.startswith("-") and s.get("removed") == [] and s.get("decision") != "champion"


def _ablation_lines(a: dict) -> list[str]:
    seeds = ", ".join(str(s) for s in a["seeds"] or [])
    L = ["### Group ablation on val_early (tuned parameters fixed)", ""]
    L += [
        f"Average precision on val_early over seeds {seeds}, mean ± std (percent). Δ = mean "
        "AP(variant) − mean AP(full), in percentage points. Pre-registered rule: drop the one "
        f"group with the largest Δ only if Δ > {a['margin_std']:g} σ, where σ = "
        f"{100 * a['sigma']:.2f} pp is the pooled seed std over all variants (bar = "
        f"{100 * a['threshold']:.2f} pp); no_gate and nofmt are reported only.",
        "",
    ]
    rows = []
    for v in a["order"]:
        s = a["variants"][v]
        decision = s.get("decision") or ""
        if _gated_out(v, s):
            decision = "= full (group removed by the gate)"
        rows.append(
            [
                v,
                _f(s.get("n_features"), "d"),
                f"{100 * s['mean']:.2f} ± {100 * s['std']:.2f}",
                f"{100 * s['delta']:+.2f}",
                decision,
            ]
        )
    L += _table(["Variant", "Inputs", "val_early AP %", "Δ pp", "Decision"], rows) + [""]
    if a.get("dropped_group"):
        verdict = f"the {a['dropped_group']} group is dropped; the champion is {a['champion']}"
    else:
        # The largest Δ among groups that were in the model (a gated-out group's fit is full's).
        measured = [
            v
            for v in a["order"]
            if v.startswith("-") and v in a["variants"] and not _gated_out(v, a["variants"][v])
        ]
        best = None
        for v in measured:  # order of the table; strict ">" keeps the first of tied maxima
            if best is None or a["variants"][v]["delta"] > a["variants"][best]["delta"]:
                best = v
        verdict = "the champion is full"
        if best is not None:
            verdict += (
                f" (largest Δ: {best}, {100 * a['variants'][best]['delta']:+.2f} pp, under the "
                f"{100 * a['threshold']:.2f} pp bar)"
            )
    L += [f"Decision: {verdict}."]
    if a.get("caveat"):
        L += ["", f"Note: {a['caveat']}."]
    return L + [""]


def _shap_lines(s: dict) -> list[str]:
    L = [f"### TreeSHAP of the champion (seed {s.get('model_seed')}, {s.get('split')})", ""]
    budget = ""
    if s.get("budget_limited"):
        budget = (
            f" The TreeSHAP time budget stopped the sample at {s['n_negatives']:,} of the "
            f"{s.get('negatives_requested') or 0:,} requested negatives (still a uniform sample)."
        )
    L += [
        f"Mean |contribution| in {s.get('space')} units on {s['n_rows']:,} rows: all "
        f"{s['n_positives']:,} positives and {s['n_negatives']:,} uniformly sampled negatives. A "
        f"group's value is the sum over its features.{budget}",
        "",
    ]
    rows = [
        [
            v["group"],
            _f(v["n_features"], "d"),
            _f(v["mean_abs"], ".4f"),
            _f(v["mean_abs_pos"], ".4f"),
        ]
        for v in s["groups"]
    ]
    L += _table(["Group", "Inputs", "Mean abs (sample)", "Mean abs (positives)"], rows) + [""]
    rows = [
        [str(i), t["name"], t["group"], _f(t["mean_abs"], ".4f"), _f(t["mean_abs_pos"], ".4f")]
        for i, t in enumerate(s["top"], start=1)
    ]
    hdr = ["Rank", "Feature", "Group", "Mean abs (sample)", "Mean abs (positives)"]
    return L + [f"Top {len(rows)} features:", ""] + _table(hdr, rows) + [""]


def _items_lines(title: str, items: list[list[Any]], lead: str | None = None) -> list[str]:
    L = [f"### {title}", ""]
    if lead:
        L += [lead, ""]
    if items:
        rows = [[str(k), _item_value(v)] for k, v in items]
        L += _table(["Item", "Value"], rows) + [""]
    return L


def _item_value(v: Any) -> str:
    if v is None:
        return NA
    if isinstance(v, int) and not isinstance(v, bool):
        return f"{v:,}"
    if isinstance(v, float):
        return f"{v:.6g}"
    return str(v)


def _validation_section(results: dict) -> list[str]:
    """Validation-only M2 sections (only when run_evaluate_stage had extras)."""
    val = results.get("validation")
    if not val:
        return []
    L = [
        "## Validation-only M2 evidence",
        "",
        "Chosen and measured before the test set is touched; no number below uses a test label "
        "(the feature-engine and rule-parity rows cover every split and use no labels).",
        "",
    ]
    if "gate" in val:
        L += _gate_lines(val["gate"])
    if "ablation" in val:
        L += _ablation_lines(val["ablation"])
    if "shap" in val:
        L += _shap_lines(val["shap"])
    if "engine" in val:
        L += _items_lines("Feature engine (build_features summary.json)", val["engine"]["items"])
    if "parity" in val:
        L += _items_lines("Rule parity", val["parity"]["items"], lead=val["parity"]["line"])
    return L


# --------------------------------------------------------------------------- M3 GNN sections

# The pre-registered verdict (M3 spec §13.4), per metric of d = winner_pair[0] - winner_pair[1].
OUTCOMES = ("gnn", "lgbm", "tie")
METRIC_LABELS = {
    "a.recall": "recall (a)",
    "a.precision": "precision (a)",
    "b.recall": "recall (b)",
    "literature.f1_thr": "F1 @ val_late-best threshold",
    "literature.f1_argmax": "F1 @ 0.5 (argmax)",
    "literature.pr_auc": "PR-AUC",
}
MODEL_PHRASES = {"gnn_causal": "the causal GNN", "lgbm_graph": "LightGBM-graph"}
CAUSAL_MODEL, LOOKAHEAD_MODEL = "gnn_causal", "gnn_lookahead"
CONVERGENCE_WINDOW = 5  # best epoch among the last 5 epochs run: "may not have converged"
SUMMARY_KEYS = (
    "protocol",
    "seeds",
    "set_key",
    "final",
    "report_hash",
    "best_val_ap_mean",
    "best_val_ap_std",
    "best_val_ap_mean_fresh",
    "best_val_f1",
    "best_epoch",
    "epochs_run",
    "max_epochs",
    "epoch_cap",
    "batch_size",
    "future_share",
    "sampled_share",
    "scored_splits",
    "test_bounds",
    "gpu",
    "gpu_seconds",
    "cores",
    "memory_mib",
    "gnn_version",
    "features_digest",
    "data_version",
)


def _model_phrase(name: str) -> str:
    return MODEL_PHRASES.get(name, name)


def check_report_hash(summaries: dict[str, dict], current: str) -> None:
    """Refuse when a --final set recorded another hash of the pre-registered `report` rules than
    the current configs/gnn.yaml: no forking paths after the test touch (M3 spec §13.4)."""
    bad = {
        name: s.get("report_hash")
        for name, s in summaries.items()
        if (s.get("final") or s.get("report_hash") is not None) and s.get("report_hash") != current
    }
    if bad:
        raise ValueError(
            f"the pre-registered `report` rules changed after a --final run: recorded {bad}, "
            f"configs/gnn.yaml now hashes to {current!r}. Restore the `report:` section the "
            "--final sets were trained under."
        )


def summary_guard(summary: dict) -> dict[str, int] | None:
    """A summary's as-of guard totals (GUARD_FIELDS), summed over its splits; None if absent."""
    g = summary.get("guard")
    if not isinstance(g, dict) or not g:
        return None
    if all(f in g and not isinstance(g[f], dict) for f in GUARD_FIELDS):
        return {f: int(g[f]) for f in GUARD_FIELDS}
    total = empty_guard()
    for split_guard in g.values():
        total = add_guard(total, split_guard)
    return total


def guard_line(name: str, summary: dict) -> str:
    """'<model>: 0 violations over N sampled edges; 0 target hits' (README evidence)."""
    g = summary_guard(summary)
    if g is None:
        return f"{name}: no guard totals in its summary.json."
    edges, viol = g["edges_checked"], g["violations"]
    if name == FAITHFUL_MODEL:
        return (
            f"{name} (snapshot guard): {viol:,} violations over {edges:,} sampled edges (every "
            "sampled rank <= its snapshot's last rank; the target stays in its snapshot, as "
            "published)."
        )
    hits = g["target_hits"]
    text = f"{name}: {viol:,} violations over {edges:,} sampled edges; {hits:,} target hits"
    if g["dropped_target_copies"]:
        text += f" ({g['dropped_target_copies']:,} sampled copies of the target dropped)"
    return text + "."


def _agg_at(model: dict, key: str, head: str) -> dict | None:
    """The per-seed aggregate {mean, std, per_seed, n_defined} of a dotted metric key."""
    part, metric = key.split(".", 1)
    node = model[part][head] if part in ("a", "b") else model[part]
    return node.get(metric)


def _negated(iv: dict) -> dict:
    # 0.0 - v, not -v: an exact zero stays 0.0 (not -0.0) in results.md.
    neg = {k: (None if v is None else 0.0 - v) for k, v in iv.items() if k in ("point", "lo", "hi")}
    return {
        "point": neg.get("point"),
        "lo": neg.get("hi"),
        "hi": neg.get("lo"),
        "undefined": iv.get("undefined", 0),
    }


def _pair_diff(results: dict, a: str, b: str, key: str) -> dict:
    """a - b of a paired-difference metric: the bootstrap's "a - b" entry, or "b - a" with the
    point negated and the CI bounds negated and swapped. Without a bootstrap, the point only."""
    md = (results.get("bootstrap") or {}).get("model_diffs") or {}
    if f"{a} - {b}" in md and key in md[f"{a} - {b}"]:
        iv = dict(md[f"{a} - {b}"][key])
    elif f"{b} - {a}" in md and key in md[f"{b} - {a}"]:
        iv = _negated(md[f"{b} - {a}"][key])
    else:
        head = results["meta"]["headline_tag"]
        pv = results["views"]["primary"]["models"]
        xa, xb = (_val(_agg_at(pv[m], key, head)) for m in (a, b))
        iv = {"point": _minus(xa, xb), "lo": None, "hi": None, "undefined": 0}
    lo, hi = iv.get("lo"), iv.get("hi")
    iv["significant"] = lo is not None and hi is not None and (lo > 0 or hi < 0)
    return iv


def lookahead_gap(results: dict, summaries: dict[str, dict], report_cfg: dict) -> dict | None:
    """Step 1 of the look-ahead gap on the primary period (M3 spec §13.2.3): per gap metric the
    causal and look-ahead (test bound end, d10) seed means, gap end - causal, gap d10 - causal and
    the tail effect end - d10 (= -(d10 - end)), all from the paired bootstrap. None unless all
    three models were evaluated."""
    names = (CAUSAL_MODEL, LOOKAHEAD_MODEL, LOOKAHEAD_D10_MODEL)
    if not all(n in results["meta"]["models"] for n in names):
        return None
    causal, end, d10 = names
    head = results["meta"]["headline_tag"]
    pv = results["views"]["primary"]["models"]
    rows = {}
    for key in report_cfg["gap_metrics"]:
        rows[key] = {
            "causal": _agg_at(pv[causal], key, head),
            "end": _agg_at(pv[end], key, head),
            "d10": _agg_at(pv[d10], key, head),
            "gap_end": _pair_diff(results, end, causal, key),
            "gap_d10": _pair_diff(results, d10, causal, key),
            "tail": _pair_diff(results, end, d10, key),
        }

    def val_ap(name: str) -> dict:
        """The set's mean and the seeds' sample std (ddof 1, None below 2 seeds: the
        convention of every other ± in results.md), + the mean over seeds other than the HPO
        model seed (causal seed 0 repeats the HPO-selected trial)."""
        s = summaries.get(name) or {}
        vals = [
            float(v["best_val_ap"])
            for v in (s.get("per_seed") or {}).values()
            if isinstance(v, dict) and v.get("best_val_ap") is not None
        ]
        return {
            "mean": s.get("best_val_ap_mean"),
            "std": float(np.std(vals, ddof=1)) if len(vals) > 1 else None,
            "n_seeds": len(vals),
            "mean_fresh": s.get("best_val_ap_mean_fresh"),
        }

    # future share: the `end` set's splits, + the d10 test pass from the d10 summary
    share = dict((summaries.get(end) or {}).get("future_share") or {})
    d10_share = (summaries.get(d10) or {}).get("future_share") or {}
    if "test_d10" in d10_share:
        share["test_d10"] = d10_share["test_d10"]
    n_seeds = int(results["meta"]["models"][end]["n_seeds"])
    return {
        "rate_tag": head,
        "metrics": rows,
        "val_early_pr_auc": {"causal": val_ap(causal), "lookahead": val_ap(end)},
        "future_share": share or None,
        "n_seeds_lookahead": n_seeds,
        "test_sampling_only": n_seeds == 1,
    }


def faithful_section(
    eval_df: pl.DataFrame,
    scores: pl.DataFrame | Path,
    summary: dict,
    report_cfg: dict,
    *,
    views: dict[str, tuple[int, int]] | None = None,
) -> dict:
    """The faithful Multi-GNN reproduction (no bootstrap; never in the model comparison).

    F1 / precision / recall at argmax (score >= 0.5) on the test rows of each view: on sampled
    targets (the published protocol) and on all targets (ours: unsampled targets scored through
    the virtual target edge). Verdict "reproduced" iff the full view's sampled-target F1 (in %)
    lies in report_cfg.reproduced_band; "may not have converged" if the best epoch is among the
    last CONVERGENCE_WINDOW epochs run. views: {"full": (lo, hi), "primary": (lo, hi)} test days
    (default: full = every test row)."""
    sc = scores if isinstance(scores, pl.DataFrame) else pl.read_parquet(scores)
    cols = [c for c in sc.columns if _SCORE_COL.match(c)]
    if len(cols) != 1 or "sampled" not in sc.columns:
        raise ValueError(
            f"faithful scores need one score_s<seed> column and `sampled`, got {sc.columns}"
        )
    col = cols[0]
    test = eval_df.filter(pl.col("split") == TEST_SPLIT).select("row_id", "day", "y")
    j = test.join(
        sc.select("row_id", col, "sampled"), on="row_id", how="left", maintain_order="left"
    )
    if j[col].null_count() or j["sampled"].null_count():
        raise ValueError("faithful scores: missing values for test rows")
    s = j[col].to_numpy().astype(np.float64)
    if not np.isfinite(s).all():
        raise ValueError("faithful scores must be finite")
    y = j["y"].to_numpy().astype(np.int8)
    sampled = j["sampled"].to_numpy().astype(bool)
    day = j["day"].to_numpy().astype(np.int64)
    if views is None:
        views = {"full": (int(day.min()), int(day.max())) if day.size else (0, -1)}
    out_views = {}
    for v, (lo, hi) in views.items():
        mask = (day >= lo) & (day <= hi)
        smp = mask & sampled
        out_views[v] = {
            "days": [int(lo), int(hi)],
            "sampled": {
                **prf_at_threshold(y[smp], s[smp], ARGMAX_THRESHOLD),
                "rows": int(smp.sum()),
            },
            "all": {
                **prf_at_threshold(y[mask], s[mask], ARGMAX_THRESHOLD),
                "rows": int(mask.sum()),
            },
            "sampled_share": float(sampled[mask].mean()) if mask.any() else math.nan,
        }
    f1 = 100.0 * out_views["full"]["sampled"]["f1"]
    lo_b, hi_b = (float(x) for x in report_cfg["reproduced_band"])
    if not math.isfinite(f1):
        verdict, miss = "undefined", None
    elif lo_b <= f1 <= hi_b:
        verdict, miss = "reproduced", 0.0
    else:
        verdict, miss = "not reproduced", (f1 - lo_b if f1 < lo_b else f1 - hi_b)
    best, run = summary.get("best_epoch"), summary.get("epochs_run")
    return {
        "views": out_views,
        "f1_sampled_pct": f1,
        "band": [lo_b, hi_b],
        "published_f1": report_cfg["published_f1"],
        "published_std": report_cfg["published_std"],
        "verdict": verdict,
        "miss_pp": miss,
        "epochs_run": run,
        "max_epochs": summary.get("max_epochs"),
        "epoch_cap": summary.get("epoch_cap"),
        "best_epoch": best,
        "batch_size": summary.get("batch_size"),
        "best_val_f1": summary.get("best_val_f1"),
        "sampled_share_by_split": summary.get("sampled_share"),
        "may_not_have_converged": (
            best is not None and run is not None and best >= run - CONVERGENCE_WINDOW
        ),
        "seeds": summary.get("seeds"),
    }


def _outcome(iv: dict | None) -> str:
    """gnn if the CI lies above 0, lgbm if below, else tie (also when the CI is undefined)."""
    if not iv or iv.get("lo") is None or iv.get("hi") is None:
        return "tie"
    if iv["lo"] > 0:
        return "gnn"
    if iv["hi"] < 0:
        return "lgbm"
    return "tie"


def verdict_of(primary: str, secondary: str) -> str:
    """The pre-registered combination rule of two per-metric outcomes (M3 spec §13.4)."""
    both = {primary, secondary}
    if not both <= set(OUTCOMES):
        raise ValueError(f"outcomes must be in {OUTCOMES}, got {primary!r}, {secondary!r}")
    if both <= {"lgbm", "tie"} and "lgbm" in both:
        return "lgbm"
    if both <= {"gnn", "tie"} and "gnn" in both:
        return "gnn"
    if both == {"tie"}:
        return "tie"
    return "mixed"


def winner_verdict(results: dict, report_cfg: dict) -> dict | None:
    """The pre-registered winner rule on the primary period: d = winner_pair[0] -
    winner_pair[1] per winner metric (primary, secondary) -> {pair, metrics: {metric: {point,
    lo, hi, outcome}}, verdict, sentence}. None if the pair was not evaluated."""
    a, b = report_cfg["winner_pair"]
    if not all(m in results["meta"]["models"] for m in (a, b)):
        return None
    metrics = {}
    for key in report_cfg["winner_metrics"]:
        iv = _pair_diff(results, a, b, key)
        metrics[key] = {**iv, "outcome": _outcome(iv)}
    p, s = (metrics[k]["outcome"] for k in report_cfg["winner_metrics"])
    verdict = verdict_of(p, s)
    pa, pb = _model_phrase(a), _model_phrase(b)
    if verdict == "lgbm":
        sentence = f"{pb} wins"
    elif verdict == "gnn":
        sentence = f"{pa} wins"
    elif verdict == "tie":
        sentence = (
            f"no significant difference; {pb} stays the served champion (cheaper to train, "
            "µs serving)"
        )
    else:
        parts = [
            f"{pa if m['outcome'] == 'gnn' else pb} is better on {METRIC_LABELS.get(k, k)}"
            for k, m in metrics.items()
        ]
        sentence = "mixed: " + "; ".join(parts)
    boot = results.get("bootstrap") or {}
    return {
        "pair": [a, b],
        "metrics": metrics,
        "verdict": verdict,
        "sentence": sentence,
        "ci_level": report_cfg["ci_level"],
        "bootstrap_level": boot.get("level"),
        "level_matches": boot.get("level") == report_cfg["ci_level"],
    }


def training_cost(name: str, summary: dict) -> dict:
    """An estimate of one GNN set's training $: gpu_seconds (the GPU worker calls' wall) x the
    exact shape price (GPU + cores + memory, costplan.shape_usd_h). A lower bound: drivers,
    container startups before the worker's clock and attempts that crashed are not in it.
    gnn_lookahead_d10 shares gnn_lookahead's training (no cost of its own)."""
    from aml.models.gnn import costplan

    out = {
        "gpu": summary.get("gpu"),
        "gpu_seconds": summary.get("gpu_seconds"),
        "cores": summary.get("cores"),
        "memory_mib": summary.get("memory_mib"),
        "usd_h": None,
        "est_usd": None,
        "shared_with": LOOKAHEAD_MODEL if name == LOOKAHEAD_D10_MODEL else None,
    }
    gpu = costplan._gpu_label(summary.get("gpu"))
    secs = _val(summary.get("gpu_seconds"))
    if out["shared_with"] or gpu is None or secs is None:
        return out
    cores = float(summary.get("cores") or costplan.PLANNING["cores"])
    gib = float(summary.get("memory_mib") or costplan.PLANNING["memory_mib"]) / 1024.0
    out["usd_h"] = costplan.shape_usd_h(gpu, cores, gib)
    out["est_usd"] = costplan.usd(out["usd_h"], secs)
    return out


def gnn_block(results: dict, eval_df: pl.DataFrame, gnn: dict, data_cfg: dict) -> dict:
    """results["gnn"]: compact summaries, guard evidence, the look-ahead gap, the faithful
    section, the winner verdict (M3 spec §13.2) and a training-$ estimate per set."""
    from aml.models.gnn import report_hash

    report_cfg = gnn["report_cfg"]
    summaries = dict(gnn.get("summaries") or {})
    faithful = gnn.get("faithful")
    names = [m for m in results["meta"]["models"] if m in COMPARISON_MODELS]
    every = {m: summaries[m] for m in names if m in summaries}
    if faithful:
        every[FAITHFUL_MODEL] = faithful["summary"]
    block = {
        "report_hash": report_hash({"report": report_cfg}),
        "report_cfg": report_cfg,
        "models": names,
        "summaries": {m: {k: s[k] for k in SUMMARY_KEYS if k in s} for m, s in every.items()},
        "guard": {
            m: {"total": summary_guard(s), "line": guard_line(m, s)} for m, s in every.items()
        },
        "training_cost": {m: training_cost(m, s) for m, s in every.items()},
        "lookahead_gap": lookahead_gap(results, summaries, report_cfg),
        "faithful": (
            faithful_section(
                eval_df,
                faithful["scores"],
                faithful["summary"],
                report_cfg,
                views={v: _view_days(data_cfg, v) for v in ("primary", "full")},
            )
            if faithful
            else None
        ),
        "winner": winner_verdict(results, report_cfg),
    }
    return _clean(block)


def _iv_text(iv: dict | None) -> str:
    if not iv:
        return NA
    return _pct(iv.get("point")) + _ci({"k": iv}, "k") + (" *" if iv.get("significant") else "")


def _winner_lines(results: dict, g: dict) -> list[str]:
    w = g.get("winner")
    L = ["## Which model wins (pre-registered rule)", ""]
    if not w:
        return L + ["The pre-registered pair was not evaluated, so there is no verdict.", ""]
    a, b = w["pair"]
    L += [
        f"Pre-registered in `configs/gnn.yaml: report` (hash `{g['report_hash']}`) before any "
        f"`--final` run: d = {a} − {b} on the primary period, paired bootstrap CI "
        f"({100 * w['ci_level']:g}%). Per metric: {_model_phrase(a)} wins it if the CI lies above "
        f"0, {_model_phrase(b)} if below 0, else a tie.",
        "",
    ]
    rows = [
        [
            f"{'primary' if i == 0 else 'secondary'}: {METRIC_LABELS.get(k, k)}",
            _iv_text(m),
            m["outcome"],
        ]
        for i, (k, m) in enumerate(w["metrics"].items())
    ]
    L += _table(["Metric", f"{a} − {b}, pp", "Outcome"], rows) + [""]
    L += [f"**Verdict: {w['sentence']}.**", ""]
    if not w.get("level_matches", True):
        L += [
            f"Warning: the bootstrap level ({w['bootstrap_level']}) differs from the "
            f"pre-registered ci_level ({w['ci_level']}).",
            "",
        ]
    return L


_SHARE_ORDER = ("train", "val_early", "val_late", "val", "test", "test_d10")


def _share_text(shares: Any) -> str:
    """'split x%, ...' in split order (summaries store sorted keys); a split without sampled
    edges (share None) reads 'n/a'."""
    if not isinstance(shares, dict) or not shares:
        return NA
    rank = {k: i for i, k in enumerate(_SHARE_ORDER)}
    keys = sorted(shares, key=lambda k: (rank.get(k, len(rank)), k))
    return ", ".join(
        f"{k} {_pct(shares[k])}%" if _val(shares[k]) is not None else f"{k} {NA}" for k in keys
    )


def _gap_lines(results: dict, g: dict) -> list[str]:
    gap = g.get("lookahead_gap")
    if not gap:
        return []
    views = results["meta"]["views"]
    plo, phi = views["primary"]["days"]
    L = [f"## Look-ahead gap, step 1 (primary period, days {plo}-{phi})", ""]
    L += [
        "The same model, sampler (`last`) and hyperparameters as the causal GNN, trained and "
        "scored with the look-ahead bound: a target's subgraph may hold edges up to the end of "
        "its split's period (train: the end of train; validation: the end of the validation "
        "days; test: the last edge of the data for `end`, the last edge of the primary period "
        "for `d10`), and the target's own edge is dropped. The causal bound admits strictly "
        "earlier minutes only. Gaps are look-ahead minus causal on the same paired bootstrap "
        "replicates; * marks a CI that excludes 0.",
        "",
    ]
    causal = CAUSAL_MODEL
    hdr = [
        "Metric",
        _label(results, causal),
        "Look-ahead, test bound end",
        "Look-ahead, test bound d10",
        "Gap end − causal, pp",
        "Gap d10 − causal, pp",
        "Tail: end − d10, pp",
    ]
    rows = [
        [
            METRIC_LABELS.get(k, k),
            _pct(r["causal"]),
            _pct(r["end"]),
            _pct(r["d10"]),
            _iv_text(r["gap_end"]),
            _iv_text(r["gap_d10"]),
            _iv_text(r["tail"]),
        ]
        for k, r in gap["metrics"].items()
    ]
    L += _table(hdr, rows) + [""]
    va = gap["val_early_pr_auc"]

    def ap(d: dict) -> str:
        if d.get("mean") is None:
            return NA
        out = f"{100 * d['mean']:.1f}"
        if d.get("std") is not None:
            return out + f" ± {100 * d['std']:.1f}"
        return out + (" (1 seed, no std)" if d.get("n_seeds") == 1 else "")

    fresh = va["causal"].get("mean_fresh")
    fresh_text = (
        f" (seeds other than the HPO model seed, whose run repeats the selected trial: "
        f"{100 * fresh:.1f})"
        if fresh is not None
        else ""
    )
    L += [
        f"- val_early PR-AUC of the selected epochs (set summaries; ± = sample std over seeds): "
        f"causal {ap(va['causal'])}{fresh_text}, look-ahead {ap(va['lookahead'])}.",
        f"- Future share (sampled edges ranked at or after the target's minute ÷ all sampled "
        f"edges, look-ahead; test_d10 = the d10 test pass): "
        f"{_share_text(gap.get('future_share'))}.",
        "- With a far bound, `last` takes each busy node's latest edges of the split, whereas "
        "the published loader samples uniformly over the snapshot; the future share quantifies "
        "how much of a subgraph that is.",
        "- Tail: end − d10 is what test edges after the primary period add to the look-ahead "
        "scores of the primary period.",
        f"- {LOOKAHEAD_D10_MODEL} shares {LOOKAHEAD_MODEL}'s validation scores; its tail rows "
        "carry causal-bound scores (so every score is finite), so it is shown on the primary "
        "period only.",
    ]
    if gap.get("test_sampling_only"):
        L.append("- One look-ahead seed: the CI covers test sampling only.")
    return L + [""]


def _faithful_lines(results: dict, g: dict) -> list[str]:
    f = g.get("faithful")
    if not f:
        return []
    full = f["views"]["full"]
    lo, hi = full["days"]
    L = ["## Faithful Multi-GNN reproduction (not in the model comparison)", ""]
    L += [
        "Multi-GIN+EU under the published protocol as recalled from Multi-GNN's code (the "
        "paper does not state every setting; the recalled ones are listed in the faithful "
        "summary's `recalled`): non-temporal snapshot graphs that contain the target, uniform "
        "sampling, per-snapshot normalisation and a timestamp feature (the confined "
        "exemptions). One seed; never part of the comparison above.",
        "",
    ]

    def row(label: str, m: dict) -> list[str]:
        return [
            label,
            _pct(m["f1"], 2),
            _pct(m["precision"], 2),
            _pct(m["recall"], 2),
            _num(m["rows"]),
        ]

    L += [f"**full** test view (days {lo}-{hi}), argmax (score >= 0.5)", ""]
    L += _table(
        ["Targets", "F1 %", "Precision %", "Recall %", "Rows"],
        [
            row("sampled targets (published protocol)", full["sampled"]),
            row("all targets (ours: unsampled ones via the virtual target edge)", full["all"]),
        ],
    )
    band = f"[{f['band'][0]:.2f}, {f['band'][1]:.2f}]"
    pub = f"{f['published_f1']:.2f} ± {f['published_std']:.2f}"
    if f["verdict"] == "reproduced":
        verdict = f"**reproduced** (F1 {f['f1_sampled_pct']:.2f} lies in {band})"
    elif f["verdict"] == "undefined":
        verdict = "**undefined** (no positive test target was sampled)"
    else:
        side = "below" if f["miss_pp"] < 0 else "above"
        verdict = (
            f"**not reproduced**: F1 {f['f1_sampled_pct']:.2f} is {abs(f['miss_pp']):.2f} pp "
            f"{side} the band {band}"
        )
    cap = f["epoch_cap"] if f.get("epoch_cap") is not None else "none"
    epochs = (
        f"- Epochs run {_num(f['epochs_run'])} of {_num(f['max_epochs'])} (gate cap: {cap}); "
        f"best epoch {_num(f['best_epoch'])} (0-based); batch size {_num(f['batch_size'])}."
    )
    if f.get("may_not_have_converged"):
        epochs += (
            f" The best epoch is among the last {CONVERGENCE_WINDOW} run: the run may not have "
            "converged."
        )
    L += [
        "",
        f"Published Multi-GIN+EU on HI-Small: {pub} (Egressy et al., Table 2); pre-registered "
        f"band {band} (± 2 std). Verdict: {verdict}.",
        "",
        f"- Sampled share of test targets: {_pct(full['sampled_share'])}%; by split (training "
        f"summary): {_share_text(f.get('sampled_share_by_split'))}.",
        epochs,
    ]
    if "primary" in f["views"]:
        p = f["views"]["primary"]
        plo, phi = p["days"]
        L.append(
            f"- Primary period (days {plo}-{phi}, for information): F1 sampled targets "
            f"{_pct(p['sampled']['f1'], 2)}, all targets {_pct(p['all']['f1'], 2)}."
        )
    L.append("- One seed: a correct pipeline misses a ± 2 std band about 5% of the time.")
    return L + [""]


def _guard_lines(g: dict) -> list[str]:
    guard = g.get("guard") or {}
    if not guard:
        return []
    L = ["## As-of guard evidence (GNN)", ""]
    L += [
        "Every sampled batch of every GNN run is checked by a code path independent of the "
        "sampler's: each sampled edge's rank must not exceed its target's bound, and a causal "
        "subgraph must not contain the target. Totals over every split the set scored:",
        "",
    ]
    return L + [f"- {v['line']}" for v in guard.values()] + [""]


def _cost_lines(g: dict) -> list[str]:
    costs = g.get("training_cost") or {}
    if not costs:
        return []
    L = ["## GNN training cost (estimate)", ""]
    L += [
        "Per set: GPU-worker wall (gpu_seconds) x the exact shape price (GPU + cores + memory). "
        "A lower bound: drivers, container startups, failed attempts and the HPO search "
        "(`aml-hpo-gnn`) are not in it; reports/cost.md has the billed totals per Modal app "
        "(every protocol trains under `aml-train-gnn`).",
        "",
    ]
    for m, c in costs.items():
        if c.get("shared_with"):
            L.append(f"- {m}: trained with {c['shared_with']} (no cost of its own).")
        elif c.get("est_usd") is None:
            L.append(f"- {m}: {NA} (no GPU wall or GPU name in its summary).")
        else:
            hours = float(c["gpu_seconds"]) / 3600.0
            L.append(
                f"- {m}: ≈ ${c['est_usd']:.2f} ({hours:.2f} h on {c['gpu']}, "
                f"{_num(c.get('cores'), 0)} cores, {_num(c.get('memory_mib'), 0)} MiB, at "
                f"${c['usd_h']:.4f}/h)."
            )
    return L + [""]


def _gnn_sections(results: dict) -> list[str]:
    """The M3 sections of results.md ([] without GNN models: M2's report is unchanged)."""
    g = results.get("gnn")
    if not g:
        return []
    return (
        _winner_lines(results, g)
        + _gap_lines(results, g)
        + _faithful_lines(results, g)
        + _guard_lines(g)
        + _cost_lines(g)
    )


# --------------------------------------------------------------------------- stage


_SCORE_COL = re.compile(r"^score_s(\d+)$")


def _align(keys: pl.DataFrame, df: pl.DataFrame, cols: list[str], what: str) -> pl.DataFrame:
    out = keys.join(df.select(["row_id", *cols]), on="row_id", how="left", maintain_order="left")
    missing = {c: out[c].null_count() for c in cols if out[c].null_count()}
    if missing:
        raise ValueError(f"{what}: missing values for eval rows: {missing}")
    return out


def rules_meta_from(thresholds_json: Path) -> dict | None:
    """Scenario count and, per rate tag, the active (tuned on) and infeasible scenarios, from the
    rules stage's thresholds.json; None if the file is absent."""
    if not thresholds_json.exists():
        return None
    doc = read_json(thresholds_json)
    thr = doc["thresholds"]
    n = doc.get("n_scenarios") or len(next(iter(thr.values()), {}))
    diag = (doc.get("scenario_diagnostics") or {}).get("rates", {})
    return {
        "n_scenarios": int(n),
        "active": {tag: [s for s, v in t.items() if v is not None] for tag, t in thr.items()},
        "infeasible": {tag: list(d.get("infeasible", [])) for tag, d in diag.items()},
    }


def run_evaluate_stage(
    paths: DataPaths,
    model_dirs: dict[str, Path],
    rules_dir: Path,
    out_dir: Path,
    data_cfg: dict,
    rules_cfg: dict,
    *,
    threads: int | None = None,
    extras: dict[str, Path] | None = None,
    gnn: dict | None = None,
) -> dict:
    """Load rule flags and model scores (joined on row_id), evaluate, write results.json/.md.

    `extras` (kind in EXTRA_KINDS -> JSON file) adds the validation-only M2 sections to
    results.md and a `validation` block to results.json; with None both are exactly M1's.

    `gnn` (M3; None = exactly M2's output) = {"model_views": {model: views} (default every
    view; gnn_lookahead_d10: ("primary",)), "summaries": {model: its set summary.json},
    "faithful": {"scores": Path, "summary": dict} | None, "report_cfg": gnn.yaml `report`}.
    The GNN comparison models are ordinary entries of `model_dirs` (same scores layout); the
    faithful run never is (refused). Adds meta.model_views and a `gnn` block (look-ahead gap,
    faithful section, pre-registered winner verdict, guard evidence) and refuses when a --final
    set recorded another `report` hash than report_cfg's.
    """
    if FAITHFUL_MODEL in model_dirs:
        raise ValueError(f"{FAITHFUL_MODEL} never enters the model comparison (its own section)")
    if gnn is not None:
        from aml.models.gnn import report_hash

        unknown = sorted(set(gnn.get("model_views") or {}) - set(model_dirs))
        bad = {
            m: v
            for m, v in (gnn.get("model_views") or {}).items()
            if "primary" not in v or not set(v) <= set(VIEWS)
        }
        if unknown or bad:
            raise ValueError(f"gnn model_views: unknown models {unknown}, bad views {bad}")
        every = dict(gnn.get("summaries") or {})
        if gnn.get("faithful"):
            every[FAITHFUL_MODEL] = gnn["faithful"]["summary"]
        check_report_hash(every, report_hash({"report": gnn["report_cfg"]}))
    ev = build_eval_frame(paths, data_cfg)
    keys = ev.select("row_id")

    flags_df = pl.read_parquet(Path(rules_dir) / "flags.parquet")
    any_cols = [c for c in flags_df.columns if c.startswith("rules_any_")]
    fired_cols = [c for c in flags_df.columns if c.startswith("fired_")]
    al = _align(keys, flags_df, any_cols + fired_cols, "rules flags")
    rule_flags = {c.removeprefix("rules_any_"): al[c].to_numpy().astype(bool) for c in any_cols}
    scenarios = {c.removeprefix("fired_"): al[c].to_numpy().astype(bool) for c in fired_cols}

    model_scores, seeds = {}, {}
    for name, d in model_dirs.items():
        sc = pl.read_parquet(Path(d) / "scores.parquet")
        cols = sorted(
            (c for c in sc.columns if _SCORE_COL.match(c)),
            key=lambda c: int(_SCORE_COL.match(c).group(1)),
        )
        if not cols:
            raise ValueError(f"{name}: no score_s<seed> columns in {d}/scores.parquet")
        a = _align(keys, sc, cols, f"{name} scores")
        model_scores[name] = np.stack([a[c].to_numpy().astype(np.float64) for c in cols])
        seeds[name] = [int(_SCORE_COL.match(c).group(1)) for c in cols]

    results = evaluate(
        ev,
        rule_flags,
        model_scores,
        data_cfg,
        rules_cfg,
        rule_scenarios=scenarios or None,
        seeds=seeds,
        threads=threads,
    )
    results["meta"]["inputs"] = {
        "rules_dir": str(rules_dir),
        "model_dirs": {k: str(v) for k, v in model_dirs.items()},
    }
    rules_meta = rules_meta_from(Path(rules_dir) / "thresholds.json")
    if rules_meta is not None:
        results["meta"]["rules"] = rules_meta
    if extras:
        results["validation"] = _clean(load_validation(extras))
    if gnn is not None:
        mv = gnn.get("model_views") or {}
        results["meta"]["model_views"] = {m: list(mv.get(m, VIEWS)) for m in model_dirs}
        results["gnn"] = gnn_block(results, ev, gnn, data_cfg)
    out_dir = Path(out_dir)
    write_json_atomic(results, out_dir / "results.json")
    write_text_atomic(render_markdown(results), out_dir / "results.md")
    return results
