"""LightGBM on transaction + graph features (M2 spec §8.3): label-free gate, Optuna, group
ablations with a pre-registered decision rule, seed finalists, TreeSHAP, and the optional
payment-format-free finalists.

Inputs: the feature table (`features_dir`: feature_spec.json + parts/, written by
build_features) and `is_laundering` of the train and val_early rows only (M1 `load_labels`);
val_late and test labels are never read here. Every model input passes
`EngineSpec.assert_model_inputs`. Matrices are float32: the stored feature-table values, which
serving feeds the booster after the same float32 cast.

Guardrails (docs/modal/COST_NOTES.md) as in M1: every finished trial (trials.jsonl), ablation
fit (a line in ablation.jsonl plus its booster under ablation/) and finalist
(booster_s<seed>.txt + .json) is written as soon as it exists and `on_checkpoint` is called, so
re-running the same command resumes. checkpoint.json fingerprints the config, the table's
content digest and the gated inputs; checkpoints written for other inputs are discarded. When
the search ends (all trials, or `optuna.timeout_s`), tuning_done.json records it: a re-run
replays the recorded trials and never reopens a finished search, so it reuses every fit and
rewrites the same champion. summary.json (the completion marker) and scores.parquet are removed
before any finished output is rewritten, so a re-run that dies part-way leaves no stale
completion marker. TreeSHAP is reused when its booster file, settings and inputs are unchanged.

Identical fits are never repeated: the winning Optuna trial is the `full` ablation fit of the
selection seed; every ablation fit is saved, so the champion's (and nofmt's) finalists of the
ablation seeds are those fits; variants with identical columns (e.g. a group the gate removed
entirely) share their fits.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from aml.config import config_hash
from aml.data.split import SPLITS
from aml.features.gate import check_gate, check_gate_cfg, run_gate
from aml.features.spec import (
    ABLATION_GROUPS,
    ENGINE_VERSION,
    FEATURE_SPEC_FILE,
    FEATURES_DIGEST,
    EngineSpec,
    parts_digest,
    scan_feature_table,
)
from aml.io import (
    read_json,
    sha256_file,
    write_json_atomic,
    write_parquet_atomic,
    write_text_atomic,
)
from aml.models import lgbm as m1
from aml.models.importance import shap_global
from aml.paths import DataPaths

# Ablation variants (§8.3 step 4): the gated set, one group dropped at a time, no gate, no format.
ABLATION_VARIANTS = (
    "full",
    "-VEL",
    "-AMT",
    "-FLOW",
    "-PORT",
    "-CYC",
    "-SG",
    "-RULE",
    "no_gate",
    "nofmt",
)
GROUP_VARIANTS = tuple(f"-{g}" for g in ABLATION_GROUPS)
# Variants `graph.ablation.variants` may leave out: no_gate only (M2 spec §10, cost cut 3). The
# group variants and full define the pre-registered rule; nofmt is the payment-format finding.
OPTIONAL_VARIANTS = ("no_gate",)
# The pooled-sigma rule needs >= 3 seeds per variant: with 2, the spec's alternative (sigma from
# the champion's finalist seeds, M2 spec §10 cost cut 2) would apply, which is not implemented.
MIN_ABLATION_SEEDS = 3
FEATURE_SET = "graph"
STUDY_NAME = "lgbm_graph"
# lgbm.yaml `graph` sections that are not M1 settings (everything else overrides an M1 key).
GRAPH_ONLY_KEYS = ("gate", "ablation", "shap")
NOFMT_KIND = "lgbm_graph_nofmt"  # model_dir kind of the --nofmt-final outputs
# Wall-clock guardrail of the TreeSHAP step (cost ~ rows x trees x leaves x depth^2): past it, no
# more negatives are explained; shap_global.json records the sample actually used.
SHAP_MAX_SECONDS = 1200.0
NOFMT_SCORE_SPLITS = ("val_late", "test")
TABLE_KEYS = ("row_id", "rank", "day", "split")
# Files in the stage directory.
GATE_FILE = "gate.json"
ABLATION_CHECKPOINT = "ablation.jsonl"
ABLATION_FILE = "ablation.json"
ABLATION_DIR = "ablation"
SHAP_FILE = "shap_global.json"
TUNING_DONE = "tuning_done.json"  # the search has ended: re-runs replay it, never extend it
SUMMARY_FILE = "summary.json"
SCORES_FILE = "scores.parquet"
DECISION_RULE = (
    "sigma = sqrt(mean over all variants of s_v^2), s_v = sample std (ddof 1) of variant v's "
    "seed APs; delta_g = mean AP(-g) - mean AP(full); if max_g delta_g > margin_std * sigma, "
    "the one group with the largest delta is dropped (ties: the first in group order), else "
    "the champion is full; no_gate and nofmt are reported and never change the champion"
)


# --------------------------------------------------------------------------- config


def effective_graph_cfg(lgbm_cfg: dict) -> dict:
    """The M1 LightGBM settings with `graph.optuna` merged over `optuna` (no `graph` key).

    Any other `graph` key that names an M1 setting overrides it the same way (dicts merged one
    level deep, other values replaced); `feature_set` becomes "graph". The graph-only sections
    (gate, ablation, shap) are read from `lgbm_cfg["graph"]` directly. The input is not mutated.
    """
    base = {k: copy.deepcopy(v) for k, v in lgbm_cfg.items() if k != m1.GRAPH_SECTION}
    for k, v in (lgbm_cfg.get(m1.GRAPH_SECTION) or {}).items():
        if k in GRAPH_ONLY_KEYS:
            continue
        if k not in base:
            raise ValueError(
                f"lgbm.graph.{k} is neither an M1 setting nor one of {GRAPH_ONLY_KEYS}"
            )
        if isinstance(base[k], Mapping) and isinstance(v, Mapping):
            base[k] = {**base[k], **copy.deepcopy(dict(v))}
        else:
            base[k] = copy.deepcopy(v)
    if "feature_set" in base:
        base["feature_set"] = FEATURE_SET
    return base


def _seed_list(name: str, seeds: Any, minimum: int) -> list[int]:
    out = [int(s) for s in seeds]
    if len(out) < minimum or len(set(out)) != len(out):
        raise ValueError(f"{name} must be >= {minimum} distinct ints, got {seeds!r}")
    return out


def check_graph_cfg(lgbm_cfg: dict) -> tuple[dict, dict]:
    """(effective M1-style config, validated graph-only sections)."""
    graph = lgbm_cfg.get(m1.GRAPH_SECTION)
    if not isinstance(graph, Mapping):
        raise ValueError("lgbm config has no `graph` section")
    missing = [k for k in GRAPH_ONLY_KEYS if not isinstance(graph.get(k), Mapping)]
    if missing:
        raise ValueError(f"lgbm.graph lacks {missing}")
    eff = effective_graph_cfg(lgbm_cfg)
    m1._check_cfg(eff)
    abl, shap = graph["ablation"], graph["shap"]
    margin = float(abl["margin_std"])
    if not math.isfinite(margin) or margin < 0:
        raise ValueError("lgbm.graph.ablation.margin_std must be a finite number >= 0")
    negatives = int(shap["negatives"])
    if negatives < 0:
        raise ValueError("lgbm.graph.shap.negatives must be >= 0")
    abl_seeds = _seed_list("lgbm.graph.ablation.seeds", abl["seeds"], 1)
    if len(abl_seeds) < MIN_ABLATION_SEEDS:
        raise ValueError(
            f"lgbm.graph.ablation.seeds needs >= {MIN_ABLATION_SEEDS} seeds, got {abl['seeds']!r}: "
            "the pre-registered rule pools each variant's seed std; with 2 seeds the spec's "
            "alternative (sigma from the champion's finalist seeds, M2 spec §10 cost cut 2) "
            "applies, which this stage does not implement"
        )
    return eff, {
        "gate": check_gate_cfg(graph["gate"]),
        "ablation": {
            "seeds": abl_seeds,
            "margin_std": margin,
            "variants": _ablation_variants(abl.get("variants")),
        },
        "shap": {"negatives": negatives, "seed": int(shap["seed"])},
    }


def _ablation_variants(chosen: Any) -> list[str]:
    """`graph.ablation.variants` in ABLATION_VARIANTS order (default: all of them); only
    OPTIONAL_VARIANTS may be left out."""
    if chosen is None:
        return list(ABLATION_VARIANTS)
    names = [str(v) for v in chosen]
    unknown = sorted(set(names) - set(ABLATION_VARIANTS))
    required = [v for v in ABLATION_VARIANTS if v not in OPTIONAL_VARIANTS and v not in names]
    if unknown or required or len(set(names)) != len(names):
        raise ValueError(
            f"lgbm.graph.ablation.variants must be distinct names from {ABLATION_VARIANTS} and "
            f"may leave out only {OPTIONAL_VARIANTS}; unknown {unknown}, missing {required}"
        )
    return [v for v in ABLATION_VARIANTS if v in names]


# --------------------------------------------------------------------------- variants, decision


def variant_columns(spec: EngineSpec, kept: Sequence[str]) -> dict[str, list[str]]:
    """Model inputs of every ablation variant, in spec order.

    full = the gated set; -G = full minus group G; no_gate = all spec model inputs; nofmt = full
    minus the payment-format-derived features.
    """
    kept_set = set(kept)
    full = [n for n in spec.feature_names if n in kept_set]
    if len(full) != len(kept_set):
        raise ValueError(f"gated names are not spec features: {sorted(kept_set - set(full))}")
    out = {"full": full}
    for g in ABLATION_GROUPS:
        out[f"-{g}"] = [n for n in full if spec.feature(n).group != g]
    out["no_gate"] = [f.name for f in spec.features if f.model_input]
    out["nofmt"] = [n for n in full if not spec.feature(n).format_derived]
    empty = [v for v, cols in out.items() if not cols]
    if empty:
        raise ValueError(f"ablation variants without any input: {empty}")
    return out


def ablation_decision(ap: Mapping[str, Sequence[float]], margin_std: float) -> dict:
    """The pre-registered rule: sigma = sqrt(mean_v s_v^2) over all variants (ddof 1);
    delta_g = mean AP(-g) - mean AP(full); drop the one group with max delta_g if it exceeds
    margin_std * sigma, else the champion is `full`. `no_gate` / `nofmt` never change it."""
    unknown = sorted(set(ap) - set(ABLATION_VARIANTS))
    if unknown:
        raise ValueError(f"unknown ablation variants: {unknown}")
    if "full" not in ap:
        raise ValueError("the ablation table needs the `full` variant")
    margin_std = float(margin_std)
    if not math.isfinite(margin_std) or margin_std < 0:
        raise ValueError(f"margin_std must be a finite number >= 0, got {margin_std!r}")
    variants: dict[str, dict[str, Any]] = {}
    for v in ABLATION_VARIANTS:  # fixed order, whatever the mapping's order
        if v not in ap:
            continue
        x = np.asarray(ap[v], dtype=np.float64)
        if x.ndim != 1 or x.size < 2 or not np.isfinite(x).all():
            raise ValueError(f"variant {v!r} needs >= 2 finite seed APs, got {list(ap[v])}")
        variants[v] = {"ap": x.tolist(), "mean": float(x.mean()), "std": float(x.std(ddof=1))}
    sigma = math.sqrt(sum(s["std"] ** 2 for s in variants.values()) / len(variants))
    threshold = margin_std * sigma
    full_mean = variants["full"]["mean"]
    for s in variants.values():
        s["delta"] = s["mean"] - full_mean
    deltas = {v: variants[v]["delta"] for v in GROUP_VARIANTS if v in variants}
    best = None
    for v, d in deltas.items():  # group order; strict ">" keeps the first of tied maxima
        if best is None or d > deltas[best]:
            best = v
    drop = best is not None and deltas[best] > threshold
    champion = best if drop else "full"
    for v, s in variants.items():
        s["decision"] = (
            "champion"
            if v == champion
            else ("reported only" if v in ("no_gate", "nofmt") else "not chosen")
        )
    return {
        "rule": DECISION_RULE,
        "margin_std": margin_std,
        "sigma": sigma,
        "threshold": threshold,
        "variants": variants,
        "order": list(variants),  # ABLATION_VARIANTS order (JSON files sort their keys)
        "deltas": deltas,
        "best_group_variant": best,
        "best_delta": None if best is None else deltas[best],
        "dropped_group": best.removeprefix("-") if drop else None,
        "champion": champion,
    }


# --------------------------------------------------------------------------- table


def load_feature_table(paths: DataPaths, features_dir: Path) -> tuple[EngineSpec, pl.DataFrame]:
    """The engine spec and every row of the feature table (keys + model inputs), rank order.

    Checks: the spec matches this code (from_json), Float32 model inputs, contiguous rank order,
    unique row ids, known splits, and one row per prepared transaction.
    """
    features_dir = Path(features_dir)
    spec = EngineSpec.from_json(read_json(features_dir / FEATURE_SPEC_FILE))
    names = spec.assert_model_inputs(f.name for f in spec.features if f.model_input)
    table = scan_feature_table(features_dir, [*TABLE_KEYS, *names]).collect()
    bad = [n for n in names if table.schema[n] != pl.Float32]
    if bad:
        raise ValueError(f"feature columns must be Float32: {bad}")
    rank = table.get_column("rank")
    if table.height == 0 or not (rank == pl.int_range(table.height, eager=True)).all():
        raise ValueError(f"{features_dir}: ranks are not 0..n-1 in part order")
    if not table.get_column("row_id").is_unique().all():
        raise ValueError(f"{features_dir}: row_id is not unique")
    unknown = set(table.get_column("split").unique().to_list()) - set(SPLITS)
    if unknown:
        raise ValueError(f"{features_dir}: unknown splits {sorted(unknown)}")
    n_tx = pl.scan_parquet(paths.transactions).select(pl.len()).collect().item()
    if table.height != n_tx:
        raise ValueError(
            f"{features_dir}: {table.height} feature rows for {n_tx} transactions (partial build?)"
        )
    return spec, table


def _take(X: np.ndarray, all_names: Sequence[str], names: Sequence[str]) -> np.ndarray:
    """Columns `names` of X (whose columns are `all_names`); X itself when nothing is dropped."""
    if list(names) == list(all_names):
        return X
    pos = {n: i for i, n in enumerate(all_names)}
    return X[:, [pos[n] for n in names]]


# --------------------------------------------------------------------------- checkpoints


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return m1.load_trial_checkpoint(path)  # same format: one object per line, torn tail ignored


def _write_jsonl(records: Sequence[Mapping[str, Any]], path: Path) -> None:
    write_text_atomic("".join(json.dumps(r, sort_keys=True) + "\n" for r in records), path)


def _reset_stale(out_dir: Path, fingerprint: str, *, ablation: bool) -> bool:
    """Delete checkpoints written for other inputs; True if any were deleted."""
    fp_path = out_dir / m1.CHECKPOINT_FINGERPRINT
    stale = fp_path.exists() and read_json(fp_path).get("fingerprint") != fingerprint
    if stale:
        (out_dir / m1.TRIALS_CHECKPOINT).unlink(missing_ok=True)
        (out_dir / TUNING_DONE).unlink(missing_ok=True)
        for f in out_dir.glob("booster_s*.json"):
            f.unlink()
        if ablation:
            (out_dir / ABLATION_CHECKPOINT).unlink(missing_ok=True)
            shutil.rmtree(out_dir / ABLATION_DIR, ignore_errors=True)
    write_json_atomic({"fingerprint": fingerprint}, fp_path)
    return stale


def _tuning_marker(out_dir: Path, fingerprint: str, n_done: int) -> dict[str, Any] | None:
    """tuning_done.json when it records a finished search of these inputs whose trials are all in
    trials.jsonl (`n_done` of them); else None (the search is resumed or started)."""
    path = Path(out_dir) / TUNING_DONE
    if not path.exists():
        return None
    mk = read_json(path)
    n = mk.get("n_trials")
    if mk.get("fingerprint") != fingerprint or not isinstance(n, int) or not 1 <= n <= n_done:
        return None
    return mk


def _plain(obj: Any) -> Any:
    """`obj` as it reads back from a JSON file."""
    return json.loads(json.dumps(obj))


def _unlink_finished(stage_dir: Path) -> None:
    """Remove a stage directory's completion marker (summary.json) and scores."""
    for name in (SUMMARY_FILE, SCORES_FILE):
        (Path(stage_dir) / name).unlink(missing_ok=True)


def _fit_key(params: Mapping[str, Any], names: Sequence[str], categorical: Sequence[str]) -> str:
    """Identity of a fit apart from its seed: the params and the ordered inputs."""
    return config_hash(dict(params), list(names), list(categorical))


def _ablation_paths(out_dir: Path, fit_key: str, seed: int) -> tuple[Path, Path]:
    stem = Path(out_dir) / ABLATION_DIR / f"booster_{fit_key}_s{seed}"
    return stem.with_suffix(".txt"), stem.with_suffix(".json")


def _save_ablation_booster(
    out_dir: Path, fit_key: str, seed: int, booster: lgb.Booster, meta: Mapping[str, Any]
) -> None:
    model, meta_path = _ablation_paths(out_dir, fit_key, seed)
    m1.save_booster(booster, model)
    write_json_atomic(dict(meta), meta_path)  # written last: its presence marks a complete save


def _load_ablation_booster(
    out_dir: Path, fit_key: str, seed: int, lgbm_cfg: dict
) -> lgb.Booster | None:
    """A saved ablation fit (exactly its best_iteration trees), or None."""
    model, meta_path = _ablation_paths(out_dir, fit_key, seed)
    if not (model.exists() and meta_path.exists()):
        return None
    meta = read_json(meta_path)
    if meta.get("fit_key") != fit_key or meta.get("seed") != seed:
        return None
    booster = lgb.Booster(model_file=str(model))
    booster.best_iteration = int(meta["best_iteration"])
    booster.best_score = {
        lgbm_cfg["early_stopping_split"]: {lgbm_cfg["eval_metric"]: float(meta["best_score"])}
    }
    return booster


# --------------------------------------------------------------------------- progress


class _Progress:
    """Wall time per fit after the first measured fit and the projected total (job log)."""

    def __init__(self, say: Callable[[str], None], planned: int) -> None:
        self.say, self.planned, self.first = say, planned, None
        self.t = time.perf_counter()

    def restart(self) -> None:
        self.t = time.perf_counter()

    def fit_done(self, what: str) -> float:
        now = time.perf_counter()
        dt, self.t = now - self.t, now
        if self.first is None:
            self.first = dt
            self.say(
                f"first fit ({what}): {dt:.1f} s; {self.planned} fits planned -> projected "
                f"{dt * self.planned / 60:.1f} min of fitting"
            )
        return dt


# --------------------------------------------------------------------------- stage


def _scores_frame(
    rows: pl.DataFrame, X: np.ndarray, boosters: Mapping[int, lgb.Booster], threads: int | None
) -> pl.DataFrame:
    return rows.select("row_id", "split").with_columns(
        pl.Series(m1.score_column(s), m1.predict(b, X, threads), dtype=pl.Float64)
        for s, b in boosters.items()
    )


def _reload_finalists(
    out_dir: Path, seeds: Sequence[int], names: Sequence[str], spec: EngineSpec
) -> dict[int, lgb.Booster]:
    """The saved finalists, loaded from the files serving reads (scores come from these)."""
    out = {}
    for s in seeds:
        b = lgb.Booster(model_file=str(Path(out_dir) / f"booster_s{s}.txt"))
        if b.feature_name() != list(names):
            raise RuntimeError(f"booster_s{s}.txt inputs differ from the chosen features")
        spec.assert_model_inputs(b.feature_name())
        out[int(s)] = b
    return out


def _val_ap_summary(
    boosters: Mapping[int, lgb.Booster], cfg: dict, selection_seed: int
) -> dict[str, Any]:
    val_ap = {f"s{s}": m1.best_score(b, cfg) for s, b in boosters.items()}
    fresh = [m1.best_score(b, cfg) for s, b in boosters.items() if s != selection_seed]
    return {
        "best_val_ap": val_ap,
        "best_val_ap_mean": float(np.mean(list(val_ap.values()))),
        "best_val_ap_std": float(np.std(list(val_ap.values()))),
        "best_val_ap_mean_fresh": float(np.mean(fresh)) if fresh else None,
        "best_iteration": {f"s{s}": int(b.best_iteration) for s, b in boosters.items()},
    }


def run_lgbm_graph_stage(
    paths: DataPaths,
    features_dir: Path,
    out_dir: Path,
    lgbm_cfg: dict,
    *,
    threads: int | None,
    nofmt_final: bool = False,
    on_checkpoint: Callable[[], None] | None = None,
    nofmt_dir: Path | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Load parts -> gate -> tune -> ablations -> decision -> finalists -> SHAP; resumable from the
    checkpoints in `out_dir`. Raises `spec.GateStopError` (after writing gate.json) if the gate's
    drop-share guard trips.

    `nofmt_final` (a user decision, M2 spec §13) also fits the remaining nofmt seeds and scores
    val_late + test with all nofmt finalists into `nofmt_dir` (default:
    <root>/models/lgbm_graph_nofmt/<out_dir name>). On a finished stage it must reproduce the
    stored champion (same tuned params and variant) and raises otherwise, before rewriting it.
    `log` receives progress lines.
    """
    eff, gcfg = check_graph_cfg(lgbm_cfg)
    train_split, es_split, seeds = m1._check_cfg(eff)
    abl_seeds = gcfg["ablation"]["seeds"]
    abl_variants = gcfg["ablation"]["variants"]
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nofmt_dir = Path(nofmt_dir) if nofmt_dir else paths.model_dir(NOFMT_KIND, out_dir.name)
    say = log or (lambda _msg: None)
    commit = on_checkpoint or (lambda: None)
    timings: dict[str, float] = {}
    t0 = last = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal last
        now = time.perf_counter()
        timings[name] = round(now - last, 3)
        last = now

    # 1. Load: every row (keys + all model inputs), Float32.
    spec, table = load_feature_table(paths, features_dir)
    inputs = [f for f in spec.features if f.model_input]
    names_all = [f.name for f in inputs]
    rows = {s: int(n) for s, n in table.get_column("split").value_counts().iter_rows()}
    # The parts' content: checkpoints of other values (a re-replay under the same features key)
    # are never reused, and evaluate/export check the scores came from these parts.
    digest = parts_digest(features_dir)
    say(f"feature table: {table.height:,} rows, {len(names_all)} model inputs, {rows}")
    lap("load")

    # 2. Gate (label-free; before any fit).
    gate_doc = run_gate(table.select("day", "split", *names_all), inputs, gcfg["gate"])
    gate_doc["spec_hash"] = spec.spec_hash()
    write_json_atomic(gate_doc, out_dir / GATE_FILE)
    commit()
    say(
        f"gate: kept {len(gate_doc['kept'])} of {len(names_all)}; dropped {gate_doc['dropped']} "
        f"(engine share {100 * gate_doc['drop_share']:.1f}%)"
    )
    check_gate(gate_doc)  # GateStopError: nothing is fitted, the user decides
    variants = variant_columns(spec, gate_doc["kept"])
    gated = variants["full"]
    lap("gate")

    # Labels of the fit rows only (train + early stopping), in table (rank) order.
    fit = table.filter(pl.col("split").is_in([train_split, es_split]))
    y_fit = m1.load_labels(paths.labels, fit.get_column("row_id"))
    is_train = (fit.get_column("split") == train_split).to_numpy()
    X_fit = m1.to_matrix(fit, names_all, np.float32)
    X_tr, y_tr = X_fit[is_train], y_fit[is_train]
    X_es, y_es = X_fit[~is_train], y_fit[~is_train]
    del fit, X_fit, y_fit
    positives = {train_split: int(y_tr.sum()), es_split: int(y_es.sum())}
    for s, n in positives.items():
        if n == 0:
            raise ValueError(f"split {s!r} has no positives; cannot train or early-stop")
    lap("labels")

    selection_seed = int(eff["optuna"]["seed"])
    fingerprint = config_hash(
        STUDY_NAME,
        lgbm_cfg,
        spec.spec_hash(),
        rows,
        positives,
        [int(X_tr.shape[0]), int(X_es.shape[0])],
        digest,
        list(gated),
    )
    stale = _reset_stale(out_dir, fingerprint, ablation=True)
    # The finished stage's summary (if any, for these inputs): --nofmt-final must reproduce it.
    summary_path = out_dir / SUMMARY_FILE
    prev = read_json(summary_path) if summary_path.exists() and not stale else None
    categorical = set(spec.categorical_names)

    def cats(names: Sequence[str]) -> list[str]:
        return [n for n in names if n in categorical]

    # 3. Tune on the gated set (M1 search, one Dataset pair, trials.jsonl checkpoint).
    Xg_tr, Xg_es = _take(X_tr, names_all, gated), _take(X_es, names_all, gated)
    pair = m1.make_datasets(
        Xg_tr,
        y_tr,
        Xg_es,
        y_es,
        eff,
        data_seed=selection_seed,
        threads=threads,
        feature_names=gated,
        categorical=cats(gated),
    )
    n_done_trials = len(m1.load_trial_checkpoint(out_dir / m1.TRIALS_CHECKPOINT))
    # A finished search (tuning_done.json for these inputs) is replayed, never extended: tune()
    # gets n_trials = the recorded count, so no new trial runs and the best trial is the same.
    marker = _tuning_marker(out_dir, fingerprint, n_done_trials)
    tune_cfg = eff
    if marker is not None:
        tune_cfg = {**eff, "optuna": {**eff["optuna"], "n_trials": int(marker["n_trials"])}}
    planned = (
        max(0, int(tune_cfg["optuna"]["n_trials"]) - n_done_trials)
        + len(abl_variants) * len(abl_seeds)
        + len(set(seeds) - set(abl_seeds)) * (2 if nofmt_final else 1)
    )
    progress = _Progress(say, planned)

    def trial_done() -> None:
        progress.fit_done("Optuna trial")
        commit()

    lap("datasets")
    progress.restart()
    tuned, trials, best = m1.tune(
        *pair,
        tune_cfg,
        threads=threads,
        checkpoint=out_dir / m1.TRIALS_CHECKPOINT,
        on_checkpoint=trial_done,
        study_name=STUDY_NAME,
    )
    if marker is None:
        write_json_atomic(
            {
                "fingerprint": fingerprint,
                "n_trials": len(trials),
                "timed_out": bool(best.get("timed_out", False)),
                "best_number": best["number"],
                "best_value": best["value"],
            },
            out_dir / TUNING_DONE,
        )
    else:
        best["timed_out"] = bool(marker["timed_out"])
        say(f"tuning: replayed the finished search ({marker['n_trials']} trials, no new trial)")
    best_booster = best.pop("booster", None)
    params = m1.model_params(eff, tuned)
    if nofmt_final and prev is not None and prev.get("best_params") != _plain(params):
        raise RuntimeError(
            "--nofmt-final: the tuned params differ from the finished stage's best_params "
            f"(summary.json in {out_dir}); refusing to rewrite an evaluated champion"
        )
    # From here on the finished outputs are rewritten: drop the completion marker (and the
    # scores) first, so a run that dies below never leaves an old summary.json next to new files.
    _unlink_finished(out_dir)
    if nofmt_final:
        _unlink_finished(nofmt_dir)
    write_json_atomic(params, out_dir / "best_params.json")
    commit()
    say(f"tuning: {len(trials)} trials, best {best['value']} (trial {best['number']})")
    lap("tune")

    # 4. Ablation fits: tuned params fixed, early stopping on val_early, seeds `ablation.seeds`.
    keys = {v: _fit_key(params, cols, cats(cols)) for v, cols in variants.items()}
    ckpt = out_dir / ABLATION_CHECKPOINT
    done: dict[tuple[str, int], dict[str, Any]] = {}
    for r in _read_jsonl(ckpt):
        v, s = r.get("variant"), r.get("seed")
        if (
            v in abl_variants
            and s in abl_seeds
            and r.get("fit_key") == keys[v]
            and _ablation_paths(out_dir, keys[v], s)[1].exists()
        ):
            done[(v, s)] = r
    _write_jsonl(list(done.values()), ckpt)  # drops a torn tail and stale records
    by_fit = {(r["fit_key"], r["seed"]): r for r in done.values()}
    n_resumed_ablation, n_ablation_fits = len(done), 0
    progress.restart()
    for v in abl_variants:
        cols, k = variants[v], keys[v]
        Xv: tuple[np.ndarray, np.ndarray] | None = None
        for s in abl_seeds:
            if (v, s) in done:
                continue
            if (k, s) in by_fit:  # same inputs as an earlier variant: the same fit
                src = by_fit[(k, s)]
                rec = {**src, "variant": v, "seconds": 0.0, "source": f"same_as:{src['variant']}"}
            else:
                if v == "full" and s == selection_seed and best_booster is not None:
                    booster, source = best_booster, f"trial:{best['number']}"
                else:
                    if v == "full" and s == selection_seed:
                        dtrain, dvalid = pair
                    else:
                        if Xv is None:
                            Xv = (_take(X_tr, names_all, cols), _take(X_es, names_all, cols))
                        dtrain, dvalid = m1.make_datasets(
                            Xv[0],
                            y_tr,
                            Xv[1],
                            y_es,
                            eff,
                            data_seed=s,
                            threads=threads,
                            feature_names=cols,
                            categorical=cats(cols),
                        )
                    booster = m1.fit_booster(m1.run_params(params, s, threads), dtrain, dvalid, eff)
                    source = "fit"
                    n_ablation_fits += 1
                seconds = progress.fit_done(f"ablation {v} seed {s}")
                rec = {
                    "variant": v,
                    "seed": s,
                    "fit_key": k,
                    "val_early_ap": m1.best_score(booster, eff),
                    "best_iteration": int(booster.best_iteration),
                    "n_features": len(cols),
                    "seconds": round(seconds, 3),
                    "source": source,
                }
                _save_ablation_booster(
                    out_dir,
                    k,
                    s,
                    booster,
                    {
                        "fit_key": k,
                        "seed": s,
                        "variant": v,
                        "params": params,
                        "feature_names": cols,
                        "best_iteration": rec["best_iteration"],
                        "best_score": rec["val_early_ap"],
                    },
                )
                del booster
            m1._append_jsonl(ckpt, rec)
            done[(v, s)] = by_fit[(k, s)] = rec
            commit()
            say(f"ablation {v} seed {s}: val_early AP {rec['val_early_ap']:.5f} ({rec['source']})")
        del Xv
    best_booster = None

    # 5. Pre-registered decision.
    ap = {v: [done[(v, s)]["val_early_ap"] for s in abl_seeds] for v in abl_variants}
    decision = ablation_decision(ap, gcfg["ablation"]["margin_std"])
    champion = decision["champion"]
    if nofmt_final and prev is not None and prev.get("variant") != champion:
        raise RuntimeError(
            f"--nofmt-final: the ablation champion {champion!r} differs from the finished "
            f"stage's {prev.get('variant')!r}; refusing to rewrite an evaluated champion"
        )
    full_set = set(gated)
    for v, info in decision["variants"].items():
        cols = variants[v]
        col_set = set(cols)
        info.update(
            n_features=len(cols),
            removed=[n for n in gated if n not in col_set],
            added=[n for n in cols if n not in full_set],
            seeds=list(abl_seeds),
            fits=[done[(v, s)]["source"] for s in abl_seeds],
        )
    ablation_doc = {
        **decision,
        "seeds": list(abl_seeds),
        "selection_seed": selection_seed,
        "caveat": (
            "full at the selection seed is the winning Optuna trial (the search maximum), so the "
            "deltas lean against dropping a group"
        ),
        "fits_this_run": n_ablation_fits,
        "resumed_fits": n_resumed_ablation,
    }
    write_json_atomic(ablation_doc, out_dir / ABLATION_FILE)
    commit()
    say(
        f"decision: champion {champion} (sigma {decision['sigma']:.5f}, best "
        f"{decision['best_group_variant']} delta {decision['best_delta']})"
    )
    lap("ablation")

    # 6. Finalists: the champion x lgbm.seeds; ablation fits of the same seeds are reused.
    def finalists(variant: str, ckpt_dir: Path) -> tuple[dict[int, lgb.Booster], list[int]]:
        cols, k = variants[variant], keys[variant]
        reuse = {}
        for s in seeds:
            if s in abl_seeds:
                b = _load_ablation_booster(out_dir, k, s, eff)
                if b is not None:
                    reuse[s] = b
        boosters = m1.train_finalists(
            _take(X_tr, names_all, cols),
            y_tr,
            _take(X_es, names_all, cols),
            y_es,
            tuned,
            eff,
            threads=threads,
            pair=pair if cols == gated else None,
            reuse=reuse,
            checkpoint_dir=ckpt_dir,
            on_checkpoint=commit,
            feature_names=cols,
            categorical=cats(cols),
            extra_meta={"feature_set": FEATURE_SET, "variant": variant, "feature_names": cols},
        )
        return boosters, sorted(reuse)

    champ = variants[champion]
    progress.restart()
    boosters, reused = finalists(champion, out_dir)
    write_json_atomic(list(champ), out_dir / "feature_names.json")
    nofmt_boosters: dict[int, lgb.Booster] = {}
    nofmt_reused: list[int] = []
    nofmt_stale = False
    if nofmt_final:
        nofmt_dir.mkdir(parents=True, exist_ok=True)
        nofmt_stale = _reset_stale(nofmt_dir, fingerprint, ablation=False)
        nofmt_boosters, nofmt_reused = finalists("nofmt", nofmt_dir)
        write_json_atomic(list(variants["nofmt"]), nofmt_dir / "feature_names.json")
    Xc_es = np.ascontiguousarray(_take(X_es, names_all, champ))
    pair = X_tr = X_es = Xg_tr = Xg_es = None  # free the fit matrices before scoring
    lap("finalists")

    # Scores from the saved files (exactly what serving loads), on the float32 table values.
    score_rows = table.filter(pl.col("split").is_in(m1.SCORE_SPLITS))
    loaded = _reload_finalists(out_dir, seeds, champ, spec)
    scores = _scores_frame(score_rows, m1.to_matrix(score_rows, champ, np.float32), loaded, threads)
    write_parquet_atomic(scores, out_dir / SCORES_FILE)
    commit()
    if nofmt_final:
        nf_rows = table.filter(pl.col("split").is_in(NOFMT_SCORE_SPLITS))
        nf_cols = variants["nofmt"]
        nf_loaded = _reload_finalists(nofmt_dir, seeds, nf_cols, spec)
        nf_scores = _scores_frame(
            nf_rows, m1.to_matrix(nf_rows, nf_cols, np.float32), nf_loaded, threads
        )
        write_parquet_atomic(nf_scores, nofmt_dir / SCORES_FILE)
        commit()
    del score_rows, table
    lap("predict")

    # 7. TreeSHAP: seed-0 (else the first) champion finalist on val_early. A document computed
    # for the same booster file, settings and inputs is reused: the time budget makes the sample
    # depend on the container's speed, so a re-run must not replace it with another sample.
    shap_seed = 0 if 0 in seeds else seeds[0]
    shap_key = {
        "booster_sha256": sha256_file(out_dir / f"booster_s{shap_seed}.txt"),
        "fingerprint": fingerprint,
        "model_seed": shap_seed,
        "split": es_split,
        "variant": champion,
        "seed": gcfg["shap"]["seed"],
        "negatives_requested": gcfg["shap"]["negatives"],
    }
    shap_path = out_dir / SHAP_FILE
    old_shap = read_json(shap_path) if shap_path.exists() else None
    shap_reused = old_shap is not None and all(old_shap.get(k) == v for k, v in shap_key.items())
    if shap_reused:
        shap_doc = old_shap
        say("TreeSHAP: reusing shap_global.json (same booster file, settings and inputs)")
    else:
        shap_doc = shap_global(
            loaded[shap_seed],
            Xc_es,
            y_es,
            champ,
            {n: spec.feature(n).group for n in champ},
            negatives=gcfg["shap"]["negatives"],
            seed=gcfg["shap"]["seed"],
            threads=threads,
            max_seconds=SHAP_MAX_SECONDS,
            log=say,
        )
        shap_doc.update(shap_key)
        if shap_doc["budget_limited"]:
            say(f"TreeSHAP stopped at the time budget: {shap_doc['n_negatives']:,} negatives")
        write_json_atomic(shap_doc, shap_path)
        commit()
    lap("shap")

    write_json_atomic(trials, out_dir / "trials.json")
    write_json_atomic(params, out_dir / "best_params.json")
    timings["total"] = round(time.perf_counter() - t0, 3)
    common = {
        "feature_set": FEATURE_SET,
        "features_dir": str(features_dir),
        "spec_hash": spec.spec_hash(),
        "engine_version": ENGINE_VERSION,
        "train_split": train_split,
        "early_stopping_split": es_split,
        "eval_metric": eff["eval_metric"],
        "rows": rows,
        "positives": positives,
        "seeds": seeds,
        "selection_seed": selection_seed,
        "best_params": params,
        FEATURES_DIGEST: digest,
    }
    if nofmt_final:
        nofmt_set = set(variants["nofmt"])
        nf_summary = {
            **common,
            "variant": "nofmt",
            "features": list(variants["nofmt"]),
            "removed": [n for n in gated if n not in nofmt_set],
            "scored_splits": list(NOFMT_SCORE_SPLITS),
            "reused_ablation_fits": nofmt_reused,
            "checkpoints_reset": nofmt_stale,
            **_val_ap_summary(nofmt_boosters, eff, selection_seed),
        }
        write_json_atomic(nf_summary, nofmt_dir / SUMMARY_FILE)
    summary = {
        **common,
        "variant": champion,
        "features": list(champ),
        "n_features": len(champ),
        "gate": {
            "kept": len(gate_doc["kept"]),
            "dropped": gate_doc["dropped"],
            "dropped_engine": gate_doc["dropped_engine"],
            "drop_share": gate_doc["drop_share"],
        },
        "ablation": {
            "seeds": list(abl_seeds),
            "variants": list(abl_variants),
            "sigma": decision["sigma"],
            "threshold": decision["threshold"],
            "deltas": decision["deltas"],
            "dropped_group": decision["dropped_group"],
            "champion": champion,
            "mean_ap": {v: s["mean"] for v, s in decision["variants"].items()},
            "fits_this_run": n_ablation_fits,
            "resumed_fits": n_resumed_ablation,
        },
        **_val_ap_summary(boosters, eff, selection_seed),
        "reused_ablation_fits": reused,
        "best_trial": best["number"],
        "best_trial_value": best["value"],
        "n_trials": len(trials),
        "trials_requested": int(eff["optuna"]["n_trials"]),
        "tuning_timed_out": bool(best.get("timed_out", False)),
        "resumed_trials": int(best.get("resumed_trials", 0)),
        "tuning_replayed": marker is not None,
        "checkpoints_reset": stale,
        "shap_reused": shap_reused,
        "shap_top": shap_doc["ranking"][:10],
        "shap_groups": {g: shap_doc["groups"][g]["mean_abs"] for g in shap_doc["group_ranking"]},
        "nofmt_final": bool(nofmt_final),
        "nofmt_dir": str(nofmt_dir) if nofmt_final else None,
        "trials": trials,
        "threads": threads,
        "timings_s": timings,
    }
    write_json_atomic(summary, out_dir / SUMMARY_FILE)
    say(f"done in {timings['total'] / 60:.1f} min: champion {champion}, {len(champ)} inputs")
    return summary
