"""LightGBM on per-transaction features (PLAN.md §6 M1).

Train on `train_split`, early-stop on `early_stopping_split` (average precision), a small Optuna
TPE search that reuses one constructed `lgb.Dataset` pair, then the best parameters retrained once
per seed. Scores are raw probabilities for the val_early, val_late and test rows.

Long-job guardrails (docs/modal/COST_NOTES.md): the search has a wall-clock bound
(`optuna.timeout_s`), every finished trial is appended to `trials.jsonl` and every finalist is
saved as soon as it is fitted, so a rerun into the same run-key directory resumes instead of
starting over. The winning trial's booster is kept and reused as the finalist of the seed it was
tuned with (the same fit), instead of being trained twice.

Labels: only `is_laundering` is joined (by row_id), and only for the train and early-stopping
rows. Test labels are never used here; evaluation is `aml.eval`'s job.
"""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import optuna
import polars as pl

from aml.config import config_hash
from aml.data.split import SPLITS
from aml.features.tx_features import (
    CATEGORICAL_FEATURES,
    INPUT_COLUMNS,
    TX_FEATURES,
    assert_whitelisted,
    build_tx_features,
    fit_vocab,
    is_forbidden,
)
from aml.io import read_json, write_json_atomic, write_parquet_atomic, write_text_atomic
from aml.paths import DataPaths

SCORE_SPLITS = ("val_early", "val_late", "test")
SEED_KEYS = ("seed", "bagging_seed", "feature_fraction_seed", "data_random_seed")
# Same inputs + same thread count -> same model (Optuna's TPE then replays exactly as well, unless
# the search is cut by its wall-clock bound or resumed from a checkpoint).
FIXED_PARAMS = {"deterministic": True, "force_row_wise": True}
# Checkpoints inside the run-key output directory.
TRIALS_CHECKPOINT = "trials.jsonl"
CHECKPOINT_FINGERPRINT = "checkpoint.json"
# lgbm.yaml section with the M2 graph-model settings; never part of the M1 stage's config.
GRAPH_SECTION = "graph"


def score_column(seed: int) -> str:
    return f"score_s{seed}"


def save_booster(booster: lgb.Booster, path: Path) -> Path:
    """Atomic model save at the best iteration.

    Written as bytes: a text-mode write turns "\\n" into "\\r\\n" on Windows, and LightGBM cannot
    load such a file.
    """
    text = booster.model_to_string(num_iteration=booster.best_iteration)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_bytes(text.encode("utf-8"))
    os.replace(tmp, path)
    return path


def _num_threads(threads: int | None) -> int:
    return int(threads) if threads else 0  # 0 = LightGBM's OpenMP default


def model_params(lgbm_cfg: dict, tuned: dict[str, Any] | None = None) -> dict[str, Any]:
    """base_params + tuned values + the fixed settings; no seeds or thread count."""
    return {
        **lgbm_cfg["base_params"],
        **(tuned or {}),
        "metric": lgbm_cfg["eval_metric"],
        **FIXED_PARAMS,
    }


def run_params(params: dict[str, Any], seed: int, threads: int | None) -> dict[str, Any]:
    return {**params, **dict.fromkeys(SEED_KEYS, int(seed)), "num_threads": _num_threads(threads)}


def to_matrix(
    features: pl.DataFrame, names: Sequence[str] = TX_FEATURES, dtype: Any = np.float64
) -> np.ndarray:
    """The `names` columns, in that order, as a `dtype` matrix (categorical codes stay integral).

    M1 default: TX_FEATURES as float64. The graph path passes names already checked by
    `EngineSpec.assert_model_inputs` and float32, the feature table's storage type, so the model
    sees exactly the values serving feeds it.
    """
    names = list(names)
    if names == TX_FEATURES:
        assert_whitelisted(names)
    else:
        bad = [n for n in names if is_forbidden(n)]
        if bad:
            raise ValueError(f"features not allowed as model inputs: {bad}")
    return features.select(names).to_numpy().astype(dtype, copy=False)


def load_labels(labels_path: Path, row_ids: pl.Series) -> np.ndarray:
    """`is_laundering` for `row_ids`, in their order.

    Only the two needed columns are read, and only the listed rows survive the join, so labels of
    other splits never reach the model code.
    """
    keys = pl.DataFrame({"row_id": row_ids})
    labels = pl.scan_parquet(labels_path).select("row_id", "is_laundering")
    out = (
        keys.lazy()
        .join(labels, on="row_id", how="left", validate="1:1", maintain_order="left")
        .collect()
    )
    y = out.get_column("is_laundering")
    if y.null_count():
        raise ValueError(f"{y.null_count()} rows have no label in {labels_path}")
    return y.cast(pl.Int8).to_numpy()


def make_datasets(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    lgbm_cfg: dict,
    *,
    data_seed: int,
    threads: int | None,
    feature_names: Sequence[str] = TX_FEATURES,
    categorical: Sequence[str] = CATEGORICAL_FEATURES,
) -> tuple[lgb.Dataset, lgb.Dataset]:
    """Construct the (train, valid) pair once.

    `feature_pre_filter=False` lets trials change `min_data_in_leaf` without a rebuild; the other
    dataset-level parameters (max_bin, data_random_seed) must then match at training time, which
    `run_params` guarantees for `data_seed`. `feature_names` name the matrix columns in order;
    `categorical` (a subset of them) are integer codes.
    """
    feature_names, categorical = list(feature_names), list(categorical)
    if not len(feature_names) == X_train.shape[1] == X_valid.shape[1]:
        raise ValueError(
            f"{len(feature_names)} feature names for matrices with {X_train.shape[1]} / "
            f"{X_valid.shape[1]} columns"
        )
    if not set(categorical) <= set(feature_names):
        raise ValueError(f"categorical features not among the inputs: {categorical}")
    params = {
        **lgbm_cfg["base_params"],
        "feature_pre_filter": False,
        "data_random_seed": int(data_seed),
        "num_threads": _num_threads(threads),
    }
    kw = {
        "feature_name": feature_names,
        "categorical_feature": categorical,
        "params": params,
        "free_raw_data": False,
    }
    dtrain = lgb.Dataset(X_train, label=y_train, **kw)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain, **kw)
    dtrain.construct()
    dvalid.construct()
    return dtrain, dvalid


def _data_seed(dtrain: lgb.Dataset) -> int:
    return int(dtrain.params["data_random_seed"])


def fit_booster(
    params: dict[str, Any],
    dtrain: lgb.Dataset,
    dvalid: lgb.Dataset,
    lgbm_cfg: dict,
) -> lgb.Booster:
    """One fit with early stopping on the valid set (best_iteration is always set)."""
    return lgb.train(
        params,
        dtrain,
        num_boost_round=int(lgbm_cfg["num_boost_round"]),
        valid_sets=[dvalid],
        valid_names=[lgbm_cfg["early_stopping_split"]],
        callbacks=[
            lgb.early_stopping(
                int(lgbm_cfg["early_stopping_rounds"]), first_metric_only=True, verbose=False
            )
        ],
    )


def best_score(booster: lgb.Booster, lgbm_cfg: dict) -> float:
    return float(booster.best_score[lgbm_cfg["early_stopping_split"]][lgbm_cfg["eval_metric"]])


def _distribution(spec: dict) -> optuna.distributions.BaseDistribution:
    """int bounds -> IntDistribution, else FloatDistribution (optionally log)."""
    low, high, log = spec["low"], spec["high"], bool(spec.get("log", False))
    ints = isinstance(low, int) and isinstance(high, int)
    kind = spec.get("type") or ("int" if ints else "float")
    if kind == "int":
        return optuna.distributions.IntDistribution(int(low), int(high), log=log)
    return optuna.distributions.FloatDistribution(float(low), float(high), log=log)


def suggest(trial: optuna.Trial, space: dict[str, dict]) -> dict[str, Any]:
    """Sample one point of `space`; int bounds -> suggest_int, else suggest_float."""
    out: dict[str, Any] = {}
    for name, spec in space.items():
        d = _distribution(spec)
        if isinstance(d, optuna.distributions.IntDistribution):
            out[name] = trial.suggest_int(name, d.low, d.high, log=d.log)
        else:
            out[name] = trial.suggest_float(name, d.low, d.high, log=d.log)
    return out


def load_trial_checkpoint(path: Path | None) -> list[dict[str, Any]]:
    """Finished trials appended to `path` (one JSON object per line); a torn last line from a
    killed container is ignored."""
    if path is None or not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            break
    return out


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with Path(path).open("a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(record, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def tune(
    dtrain: lgb.Dataset,
    dvalid: lgb.Dataset,
    lgbm_cfg: dict,
    *,
    threads: int | None,
    checkpoint: Path | None = None,
    on_checkpoint: Callable[[], None] | None = None,
    study_name: str = "lgbm_tx",
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Optuna TPE over `optuna.space`, every trial on the same constructed Dataset pair.

    Returns (best tuned values, trials, best-trial info). With n_trials = 0 the base params are
    used as they are. The search stops after n_trials or `optuna.timeout_s` seconds, whichever
    comes first. With `checkpoint`, trials already in that file are replayed into the study
    (no refit) and only the rest run; every new finished trial is appended (with the search's
    wall time so far, `search_elapsed_s`), then `on_checkpoint` is called (e.g. to commit a
    Modal Volume). A resumed search gets only the rest of the time budget (timeout_s minus the
    replayed trials' wall time; none left: no new trial), and its sampler is seeded with
    seed + (number of replayed trials), so a resume in the random startup phase does not redraw
    the points already tried. A resumed search is therefore not bit-identical to an
    uninterrupted one.

    best-trial info: number, value, resumed_trials, timed_out, and `booster`, the winning
    trial's booster when it was trained in this call (None if it came from the checkpoint).
    """
    ocfg = lgbm_cfg["optuna"]
    n_trials = int(ocfg["n_trials"])
    if n_trials <= 0:
        return {}, [], {"number": None, "value": None}
    seed = _data_seed(dtrain)  # trial seeds must match the Dataset's data_random_seed
    done = load_trial_checkpoint(checkpoint)[:n_trials]
    # A fresh search keeps its configured seed; a resumed one continues with a shifted seed.
    sampler = optuna.samplers.TPESampler(
        seed=int(ocfg["seed"]) + len(done), n_startup_trials=int(ocfg["n_startup_trials"])
    )
    study = optuna.create_study(study_name=study_name, direction="maximize", sampler=sampler)
    dists = {name: _distribution(spec) for name, spec in ocfg["space"].items()}
    # Wall time already spent by the replayed trials (records from before this field: 0).
    spent = max((float(r.get("search_elapsed_s") or 0.0) for r in done), default=0.0)
    if checkpoint is not None and Path(checkpoint).exists():
        # Drop a torn last line before appending, or later trials would hide behind it.
        text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in done)
        write_text_atomic(text, Path(checkpoint))
    for rec in done:
        if set(rec["params"]) != set(dists):
            raise ValueError(f"{checkpoint}: trial params do not match optuna.space")
        study.add_trial(
            optuna.trial.create_trial(
                params=rec["params"],
                distributions=dists,
                value=float(rec["value"]),
                user_attrs={"best_iteration": rec.get("best_iteration")},
            )
        )
    # The best booster trained in this call; strict ">" as Optuna's best_trial is the first max.
    holder: dict[str, Any] = {"number": None, "value": -math.inf, "booster": None}
    if study.trials:
        holder.update(number=study.best_trial.number, value=study.best_trial.value)
    t_start = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        params = run_params(model_params(lgbm_cfg, suggest(trial, ocfg["space"])), seed, threads)
        booster = fit_booster(params, dtrain, dvalid, lgbm_cfg)
        value = best_score(booster, lgbm_cfg)
        trial.set_user_attr("best_iteration", int(booster.best_iteration))
        if value > holder["value"]:
            holder.update(number=trial.number, value=value, booster=booster)
        if checkpoint is not None:
            _append_jsonl(
                checkpoint,
                {
                    "number": trial.number,
                    "params": trial.params,
                    "value": value,
                    "state": "COMPLETE",
                    "best_iteration": int(booster.best_iteration),
                    "search_elapsed_s": round(spent + time.perf_counter() - t_start, 3),
                },
            )
            if on_checkpoint is not None:
                on_checkpoint()
        return value

    remaining = n_trials - len(done)
    timeout = ocfg.get("timeout_s")
    budget = None if timeout is None else float(timeout) - spent
    if remaining > 0 and not study.trials:
        # At least one trial always runs. Optuna checks the timeout before every trial, the first
        # included, so a tiny budget on a fine-grained clock (Linux) would otherwise end the search
        # with no trial at all. The sampler continues across the two calls, so the sequence of
        # trials is the same as one uninterrupted call.
        study.optimize(objective, n_trials=1)
        remaining -= 1
        if budget is not None:
            budget -= time.perf_counter() - t_start
    if remaining > 0 and (budget is None or budget > 0):
        study.optimize(objective, n_trials=remaining, timeout=budget)
    trials = [
        {
            "number": t.number,
            "params": t.params,
            "value": t.value,
            "state": t.state.name,
            "best_iteration": t.user_attrs.get("best_iteration"),
        }
        for t in study.trials
    ]
    best = study.best_trial
    booster = holder["booster"] if holder["number"] == best.number else None
    return (
        dict(best.params),
        trials,
        {
            "number": best.number,
            "value": best.value,
            "resumed_trials": len(done),
            "timed_out": len(study.trials) < n_trials,
            "booster": booster,
        },
    )


def _finalist_meta_path(checkpoint_dir: Path, seed: int) -> Path:
    return Path(checkpoint_dir) / f"booster_s{seed}.json"


def _load_finalist(
    checkpoint_dir: Path | None,
    seed: int,
    params: dict[str, Any],
    lgbm_cfg: dict,
    extra: Mapping[str, Any] | None = None,
) -> lgb.Booster | None:
    """A finalist saved by an earlier, interrupted run with the same params (and the same
    `extra` metadata, e.g. the graph path's feature names), or None."""
    if checkpoint_dir is None:
        return None
    meta_path = _finalist_meta_path(checkpoint_dir, seed)
    model_path = Path(checkpoint_dir) / f"booster_s{seed}.txt"
    if not (meta_path.exists() and model_path.exists()):
        return None
    meta = read_json(meta_path)
    if meta.get("params") != json.loads(json.dumps(params)) or meta.get("seed") != seed:
        return None
    if any(meta.get(k) != json.loads(json.dumps(v)) for k, v in (extra or {}).items()):
        return None
    booster = lgb.Booster(model_file=str(model_path))
    # A loaded model has no early-stopping record; restore it (the file holds exactly the
    # best_iteration trees, so predictions are unchanged).
    booster.best_iteration = int(meta["best_iteration"])
    booster.best_score = {
        lgbm_cfg["early_stopping_split"]: {lgbm_cfg["eval_metric"]: float(meta["best_score"])}
    }
    return booster


def _save_finalist(
    checkpoint_dir: Path,
    seed: int,
    booster: lgb.Booster,
    params: dict[str, Any],
    lgbm_cfg: dict,
    extra: Mapping[str, Any] | None = None,
) -> None:
    save_booster(booster, Path(checkpoint_dir) / f"booster_s{seed}.txt")
    write_json_atomic(
        {
            "seed": seed,
            "params": params,
            "best_iteration": int(booster.best_iteration),
            "best_score": best_score(booster, lgbm_cfg),
            **(extra or {}),
        },
        _finalist_meta_path(checkpoint_dir, seed),
    )


def train_finalists(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    tuned: dict[str, Any],
    lgbm_cfg: dict,
    *,
    threads: int | None,
    pair: tuple[lgb.Dataset, lgb.Dataset] | None = None,
    reuse: dict[int, lgb.Booster] | None = None,
    checkpoint_dir: Path | None = None,
    on_checkpoint: Callable[[], None] | None = None,
    feature_names: Sequence[str] = TX_FEATURES,
    categorical: Sequence[str] = CATEGORICAL_FEATURES,
    extra_meta: Mapping[str, Any] | None = None,
) -> dict[int, lgb.Booster]:
    """The tuned params retrained once per seed.

    A seed changes data_random_seed (the bin-construction sample), so each seed gets its own
    Dataset pair; `pair` is reused for the seed it was built with. `reuse` maps a seed to a
    booster already fitted with exactly these params, inputs and that seed (the winning Optuna
    trial, or an M2 ablation fit), which is taken instead of refitting. With `checkpoint_dir`,
    each finalist is saved as soon as it is fitted and a matching saved finalist is loaded
    instead of refitted. `extra_meta` is stored with each saved finalist and must match on load
    (the graph path passes its feature names, so a finalist of other inputs is never resumed).
    """
    params = model_params(lgbm_cfg, tuned)
    boosters: dict[int, lgb.Booster] = {}
    for seed in lgbm_cfg["seeds"]:
        seed = int(seed)
        loaded = _load_finalist(checkpoint_dir, seed, params, lgbm_cfg, extra_meta)
        if loaded is not None:
            boosters[seed] = loaded
            continue
        if reuse and seed in reuse:
            boosters[seed] = reuse[seed]
        else:
            if pair is not None and _data_seed(pair[0]) == seed:
                dtrain, dvalid = pair
            else:
                dtrain, dvalid = make_datasets(
                    X_train,
                    y_train,
                    X_valid,
                    y_valid,
                    lgbm_cfg,
                    data_seed=seed,
                    threads=threads,
                    feature_names=feature_names,
                    categorical=categorical,
                )
            boosters[seed] = fit_booster(
                run_params(params, seed, threads), dtrain, dvalid, lgbm_cfg
            )
        if checkpoint_dir is not None:
            _save_finalist(checkpoint_dir, seed, boosters[seed], params, lgbm_cfg, extra_meta)
            if on_checkpoint is not None:
                on_checkpoint()
    return boosters


def predict(booster: lgb.Booster, X: np.ndarray, threads: int | None) -> np.ndarray:
    """Raw (uncalibrated) probabilities at the best iteration, float64."""
    num_iteration = booster.best_iteration if booster.best_iteration > 0 else None
    p = booster.predict(X, num_iteration=num_iteration, num_threads=_num_threads(threads))
    return np.asarray(p, dtype=np.float64)


def _check_cfg(lgbm_cfg: dict) -> tuple[str, str, list[int]]:
    tr, es = lgbm_cfg["train_split"], lgbm_cfg["early_stopping_split"]
    for s in (tr, es):
        if s not in SPLITS:
            raise ValueError(f"unknown split {s!r}; expected one of {SPLITS}")
        if s == "test":
            raise ValueError("the test split is never used for training or early stopping")
    if tr == es:
        raise ValueError("train_split and early_stopping_split must differ")
    if tr != "train":
        raise ValueError("final models train on the train split only (PLAN.md §4)")
    if es in ("val_late", "test"):
        raise ValueError(
            "early stopping must not use val_late (reserved for thresholds and calibration) "
            "or test (PLAN.md §4)"
        )
    seeds = [int(s) for s in lgbm_cfg["seeds"]]
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError(f"seeds must be a non-empty list of distinct ints, got {seeds}")
    return tr, es, seeds


def _reset_stale_checkpoints(out_dir: Path, fingerprint: str) -> bool:
    """Delete checkpoints written for other inputs; returns True if any were deleted.

    The Modal jobs key `out_dir` by config, so a mismatch means a direct call reused a directory
    with a different config or data.
    """
    fp_path = out_dir / CHECKPOINT_FINGERPRINT
    stale = fp_path.exists() and read_json(fp_path).get("fingerprint") != fingerprint
    if stale:
        (out_dir / TRIALS_CHECKPOINT).unlink(missing_ok=True)
        for f in out_dir.glob("booster_s*.json"):
            f.unlink()
    write_json_atomic({"fingerprint": fingerprint}, fp_path)
    return stale


def run_lgbm_stage(
    paths: DataPaths,
    out_dir: Path,
    lgbm_cfg: dict,
    rules_cfg: dict,
    *,
    threads: int | None,
    on_checkpoint: Callable[[], None] | None = None,
) -> dict:
    """Features -> labels (train/early-stopping rows only) -> Optuna -> seeds -> scores + files.

    Resumes from the checkpoints in `out_dir` (finished trials, fitted finalists) when they were
    written for the same config and data; `on_checkpoint` runs after each one is written.
    lgbm.yaml's M2 `graph` section is dropped first: it is not an M1 setting, and keeping it out
    of the checkpoint fingerprint keeps the M1 checkpoints valid.
    """
    lgbm_cfg = {k: v for k, v in lgbm_cfg.items() if k != GRAPH_SECTION}
    train_split, es_split, seeds = _check_cfg(lgbm_cfg)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    timings: dict[str, float] = {}
    t0 = last = time.perf_counter()

    def lap(name: str) -> None:
        nonlocal last
        now = time.perf_counter()
        timings[name] = round(now - last, 3)
        last = now

    tx = pl.read_parquet(paths.transactions, columns=["split", *INPUT_COLUMNS])  # rank order
    lap("load")
    vocab = fit_vocab(tx.filter(pl.col("split") == train_split))
    feats = build_tx_features(tx, vocab, rules_cfg["round_unit"]).with_columns(
        tx.get_column("split")
    )
    rows = {s: int(n) for s, n in feats.get_column("split").value_counts().iter_rows()}
    del tx
    lap("features")

    fit = feats.filter(pl.col("split").is_in([train_split, es_split]))
    y_fit = load_labels(paths.labels, fit.get_column("row_id"))
    is_train = (fit.get_column("split") == train_split).to_numpy()
    X_fit = to_matrix(fit)
    X_tr, y_tr = X_fit[is_train], y_fit[is_train]
    X_es, y_es = X_fit[~is_train], y_fit[~is_train]
    del fit, X_fit, y_fit
    positives = {train_split: int(y_tr.sum()), es_split: int(y_es.sum())}
    for s, n in positives.items():
        if n == 0:
            raise ValueError(f"split {s!r} has no positives; cannot train or early-stop")
    lap("labels")

    selection_seed = int(lgbm_cfg["optuna"]["seed"])
    fingerprint = config_hash(
        lgbm_cfg,
        rules_cfg["round_unit"],
        rows,
        positives,
        [int(X_tr.shape[0]), int(X_es.shape[0])],
    )
    stale = _reset_stale_checkpoints(out_dir, fingerprint)
    pair = make_datasets(
        X_tr, y_tr, X_es, y_es, lgbm_cfg, data_seed=selection_seed, threads=threads
    )
    lap("datasets")
    tuned, trials, best = tune(
        *pair,
        lgbm_cfg,
        threads=threads,
        checkpoint=out_dir / TRIALS_CHECKPOINT,
        on_checkpoint=on_checkpoint,
    )
    best_booster = best.pop("booster", None)
    lap("tune")
    params = model_params(lgbm_cfg, tuned)
    write_json_atomic(params, out_dir / "best_params.json")  # before the finalists (resume)
    boosters = train_finalists(
        X_tr,
        y_tr,
        X_es,
        y_es,
        tuned,
        lgbm_cfg,
        threads=threads,
        pair=pair,
        # The winning trial is the finalist of the seed it was tuned with: the same fit.
        reuse={selection_seed: best_booster} if best_booster is not None else None,
        checkpoint_dir=out_dir,
        on_checkpoint=on_checkpoint,
    )
    del pair, X_tr, y_tr, X_es, y_es, best_booster
    lap("finalists")

    score_rows = feats.filter(pl.col("split").is_in(SCORE_SPLITS))
    X_sc = to_matrix(score_rows)
    scores = score_rows.select("row_id", "split").with_columns(
        pl.Series(score_column(s), predict(b, X_sc, threads), dtype=pl.Float64)
        for s, b in boosters.items()
    )
    del X_sc, score_rows, feats
    lap("predict")

    for seed, booster in boosters.items():
        assert_whitelisted(booster.feature_name())
        save_booster(booster, out_dir / f"booster_s{seed}.txt")
    write_parquet_atomic(scores, out_dir / "scores.parquet")
    write_json_atomic(params, out_dir / "best_params.json")
    write_json_atomic(trials, out_dir / "trials.json")
    write_json_atomic(vocab, out_dir / "vocab.json")
    write_json_atomic(list(TX_FEATURES), out_dir / "feature_names.json")
    lap("write")
    timings["total"] = round(time.perf_counter() - t0, 3)

    val_ap = {f"s{s}": best_score(b, lgbm_cfg) for s, b in boosters.items()}
    # The selection seed's val AP is the maximum over the search (winner's curse); the other
    # seeds are fresh draws of the chosen params.
    fresh = [v for s, v in zip(boosters, val_ap.values(), strict=True) if s != selection_seed]
    summary = {
        "feature_set": lgbm_cfg.get("feature_set", "tx"),
        "features": list(TX_FEATURES),
        "train_split": train_split,
        "early_stopping_split": es_split,
        "eval_metric": lgbm_cfg["eval_metric"],
        "round_unit": rules_cfg["round_unit"],
        "rows": rows,
        "positives": positives,
        "seeds": seeds,
        "best_val_ap": val_ap,
        "best_val_ap_mean": float(np.mean(list(val_ap.values()))),
        "best_val_ap_std": float(np.std(list(val_ap.values()))),
        "selection_seed": selection_seed,
        "best_val_ap_mean_fresh": float(np.mean(fresh)) if fresh else None,
        "best_iteration": {f"s{s}": int(b.best_iteration) for s, b in boosters.items()},
        "best_trial": best["number"],
        "best_trial_value": best["value"],
        "best_params": params,
        "n_trials": len(trials),
        "trials_requested": int(lgbm_cfg["optuna"]["n_trials"]),
        "tuning_timed_out": bool(best.get("timed_out", False)),
        "resumed_trials": int(best.get("resumed_trials", 0)),
        "checkpoints_reset": stale,
        "trials": trials,
        "threads": threads,
        "timings_s": timings,
    }
    write_json_atomic(summary, out_dir / "summary.json")
    return summary
