"""LightGBM stage on the prepared fixture (spec §3.9)."""

from __future__ import annotations

import copy
import shutil
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

from aml.features.tx_features import TX_FEATURES, build_tx_features
from aml.io import read_json, write_parquet_atomic
from aml.models.lgbm import (
    SCORE_SPLITS,
    load_labels,
    make_datasets,
    predict,
    run_lgbm_stage,
    score_column,
    train_finalists,
    tune,
)
from aml.paths import DataPaths

THREADS = 2
OUT_FILES = [
    "scores.parquet",
    "best_params.json",
    "trials.json",
    "vocab.json",
    "feature_names.json",
    "summary.json",
]


@pytest.fixture(scope="module")
def stage(prepared, lgbm_cfg, rules_cfg, tmp_path_factory) -> tuple[Path, dict]:
    out = tmp_path_factory.mktemp("lgbm")
    return out, run_lgbm_stage(prepared, out, lgbm_cfg, rules_cfg, threads=THREADS)


def _scores(out: Path) -> pl.DataFrame:
    return pl.read_parquet(out / "scores.parquet")


def test_writes_every_file(stage, lgbm_cfg) -> None:
    out, _ = stage
    expected = OUT_FILES + [f"booster_s{s}.txt" for s in lgbm_cfg["seeds"]]
    for name in expected:
        assert (out / name).is_file(), name
    assert not list(out.glob(".*.tmp-*"))  # atomic writes left no temp files


def test_scores_cover_scored_splits_only(stage, prepared, lgbm_cfg) -> None:
    out, _ = stage
    scores = _scores(out)
    assert scores.columns == ["row_id", "split"] + [score_column(s) for s in lgbm_cfg["seeds"]]
    tx = pl.read_parquet(prepared.transactions, columns=["row_id", "split"])  # rank order
    want = tx.filter(pl.col("split").is_in(SCORE_SPLITS))
    assert scores.select("row_id", "split").equals(want)
    assert set(scores["split"].unique()) == set(SCORE_SPLITS)
    assert scores["row_id"].is_unique().all()


def test_seed_scores_are_valid_and_differ(stage, lgbm_cfg) -> None:
    out, _ = stage
    scores = _scores(out)
    cols = [score_column(s) for s in lgbm_cfg["seeds"]]
    for c in cols:
        assert scores.schema[c] == pl.Float64
        s = scores[c].to_numpy()
        assert np.isfinite(s).all() and (s >= 0).all() and (s <= 1).all()
        assert np.unique(s).size > 1
    a, b = scores[cols[0]].to_numpy(), scores[cols[1]].to_numpy()
    assert not np.allclose(a, b)


def test_summary_trials_and_params(stage, lgbm_cfg) -> None:
    out, summary = stage
    assert read_json(out / "summary.json") == summary
    n = lgbm_cfg["optuna"]["n_trials"]
    trials = read_json(out / "trials.json")
    assert len(trials) == n == summary["n_trials"] == len(summary["trials"])
    for t in trials:
        assert {"number", "params", "value", "state"} <= set(t)
        assert t["state"] == "COMPLETE"
        assert set(t["params"]) == set(lgbm_cfg["optuna"]["space"])
        assert 0.0 <= t["value"] <= 1.0
    assert summary["best_trial_value"] == max(t["value"] for t in trials)
    best = read_json(out / "best_params.json")
    assert best == summary["best_params"]
    assert best["metric"] == "average_precision"
    assert best["deterministic"] is True and best["force_row_wise"] is True
    tuned = trials[summary["best_trial"]]["params"]
    assert all(best[k] == v for k, v in tuned.items())
    for s in lgbm_cfg["seeds"]:
        assert 0.0 <= summary["best_val_ap"][f"s{s}"] <= 1.0
        assert 1 <= summary["best_iteration"][f"s{s}"] <= lgbm_cfg["num_boost_round"]
    assert summary["positives"]["train"] > 0 and summary["positives"]["val_early"] > 0
    assert "test" not in summary["positives"]
    assert summary["timings_s"]["total"] > 0


def test_saved_boosters_reproduce_scores(stage, prepared, lgbm_cfg, rules_cfg) -> None:
    out, _ = stage
    scores = _scores(out)
    vocab = read_json(out / "vocab.json")
    tx = pl.read_parquet(prepared.transactions).filter(pl.col("split").is_in(SCORE_SPLITS))
    X = build_tx_features(tx, vocab, rules_cfg["round_unit"]).select(TX_FEATURES).to_numpy()
    for s in lgbm_cfg["seeds"]:
        path = out / f"booster_s{s}.txt"
        assert b"\r\n" not in path.read_bytes()  # LightGBM cannot parse CRLF model files
        booster = lgb.Booster(model_file=str(path))
        assert booster.feature_name() == TX_FEATURES
        p = booster.predict(X, num_threads=THREADS)
        np.testing.assert_allclose(p, scores[score_column(s)].to_numpy(), rtol=0, atol=1e-12)


def test_rerun_is_reproducible(stage, prepared, lgbm_cfg, rules_cfg, tmp_path) -> None:
    out, summary = stage
    again = run_lgbm_stage(prepared, tmp_path, lgbm_cfg, rules_cfg, threads=THREADS)
    assert _scores(tmp_path).equals(_scores(out))
    assert again["trials"] == summary["trials"]
    assert again["best_params"] == summary["best_params"]


def test_test_labels_are_never_used(stage, prepared, lgbm_cfg, rules_cfg, tmp_path) -> None:
    """Flipping every test label changes no saved score, so the stage cannot have used them."""
    out, _ = stage
    flipped = DataPaths(tmp_path / "volume")
    flipped.transactions.parent.mkdir(parents=True)
    shutil.copy(prepared.transactions, flipped.transactions)
    tx = pl.read_parquet(prepared.transactions, columns=["row_id", "split"])
    test_ids = tx.filter(pl.col("split") == "test")["row_id"]
    labels = pl.read_parquet(prepared.labels).with_columns(
        pl.when(pl.col("row_id").is_in(test_ids.implode()))
        .then(1 - pl.col("is_laundering"))
        .otherwise(pl.col("is_laundering"))
        .cast(pl.Int8)
        .alias("is_laundering")
    )
    assert (labels["is_laundering"] != pl.read_parquet(prepared.labels)["is_laundering"]).sum() > 0
    write_parquet_atomic(labels, flipped.labels)
    run_lgbm_stage(flipped, tmp_path / "out", lgbm_cfg, rules_cfg, threads=THREADS)
    assert _scores(tmp_path / "out").equals(_scores(out))


def test_vocab_is_fitted_on_train_rows_only(prepared, lgbm_cfg, rules_cfg, tmp_path, monkeypatch):
    """PLAN.md §4: preprocessing is fitted on train only. Categories that appear only after the
    train days must not enter vocab.json, and fit_vocab must only ever see train rows."""
    import aml.models.lgbm as lgbm_mod

    vol = DataPaths(tmp_path / "volume")
    vol.labels.parent.mkdir(parents=True)
    shutil.copy(prepared.labels, vol.labels)
    tx = pl.read_parquet(prepared.transactions)
    late = pl.col("split").is_in(["val_early", "test"]) & (pl.col("row_id") % 7 == 0)
    tx = tx.with_columns(
        pl.when(late)
        .then(pl.lit("Carrier Pigeon"))
        .otherwise("payment_format")
        .alias("payment_format"),
        pl.when(late)
        .then(pl.lit("Doubloon"))
        .otherwise("receiving_currency")
        .alias("receiving_currency"),
    )
    write_parquet_atomic(tx, vol.transactions)

    seen: list[set[str]] = []
    real_fit_vocab = lgbm_mod.fit_vocab

    def spy(df):
        seen.append(set(df["split"].unique().to_list()))
        return real_fit_vocab(df)

    monkeypatch.setattr(lgbm_mod, "fit_vocab", spy)
    cfg = copy.deepcopy(lgbm_cfg)
    cfg["optuna"]["n_trials"] = 0
    cfg["seeds"] = [0]
    run_lgbm_stage(vol, tmp_path / "out", cfg, rules_cfg, threads=THREADS)
    assert seen == [{"train"}]
    vocab = read_json(tmp_path / "out" / "vocab.json")
    assert vocab == real_fit_vocab(tx.filter(pl.col("split") == "train"))
    assert "Carrier Pigeon" not in vocab["payment_format"]
    assert "Doubloon" not in vocab["receiving_currency"]


@pytest.mark.parametrize(
    "change",
    [
        {"train_split": "test"},
        {"early_stopping_split": "test"},
        {"early_stopping_split": "train"},
        {"early_stopping_split": "val_late"},  # reserved for thresholds and calibration
        {"train_split": "val_late"},
        {"train_split": "val_early", "early_stopping_split": "train"},  # final models: train only
        {"train_split": "days"},
        {"seeds": [1, 1]},
        {"seeds": []},
    ],
)
def test_rejects_bad_config(change, prepared, lgbm_cfg, rules_cfg, tmp_path) -> None:
    cfg = {**copy.deepcopy(lgbm_cfg), **change}
    with pytest.raises(ValueError):
        run_lgbm_stage(prepared, tmp_path, cfg, rules_cfg, threads=THREADS)


def test_load_labels_keeps_order_and_rejects_missing(prepared) -> None:
    lab = pl.read_parquet(prepared.labels, columns=["row_id", "is_laundering"])
    ids = lab["row_id"].reverse().head(50)
    y = load_labels(prepared.labels, ids)
    want = lab.join(pl.DataFrame({"row_id": ids}), on="row_id").sort(
        pl.col("row_id"), descending=True
    )
    assert y.tolist() == want["is_laundering"].to_list()
    with pytest.raises(ValueError, match="no label"):
        load_labels(prepared.labels, pl.Series("row_id", [10**9], dtype=pl.Int64))


def _toy(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = np.column_stack(
        [rng.normal(5, 2, n)] + [rng.integers(0, k, n) for k in (4, 4, 2, 5, 2, 2, 2, 24)]
    ).astype(np.float64)
    logit = -4 + 1.5 * X[:, 3] + 0.5 * (X[:, 4] == 2) + 0.3 * X[:, 0]
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(np.int8)
    return X, y


def test_one_dataset_pair_is_reused(lgbm_cfg) -> None:
    """Trials and the matching finalist must not silently rebuild the constructed Datasets."""
    X_tr, y_tr = _toy(3000, 0)
    X_va, y_va = _toy(1000, 1)
    cfg = copy.deepcopy(lgbm_cfg)
    seed = cfg["optuna"]["seed"]
    pair = make_datasets(X_tr, y_tr, X_va, y_va, cfg, data_seed=seed, threads=THREADS)
    handles = [d._handle.value for d in pair]
    tuned, trials, best = tune(*pair, cfg, threads=THREADS)
    assert len(trials) == cfg["optuna"]["n_trials"] and best["number"] is not None
    assert [d._handle.value for d in pair] == handles
    boosters = train_finalists(X_tr, y_tr, X_va, y_va, tuned, cfg, threads=THREADS, pair=pair)
    assert [d._handle.value for d in pair] == handles
    assert set(boosters) == set(cfg["seeds"])
    for b in boosters.values():
        p = predict(b, X_va, THREADS)
        assert p.dtype == np.float64 and ((p >= 0) & (p <= 1)).all()


def test_zero_trials_uses_base_params(lgbm_cfg) -> None:
    X_tr, y_tr = _toy(2000, 2)
    X_va, y_va = _toy(500, 3)
    cfg = copy.deepcopy(lgbm_cfg)
    cfg["optuna"]["n_trials"] = 0
    pair = make_datasets(X_tr, y_tr, X_va, y_va, cfg, data_seed=0, threads=THREADS)
    assert tune(*pair, cfg, threads=THREADS) == ({}, [], {"number": None, "value": None})


# --------------------------------------------------------------------------- long-job guardrails


class _Killed(RuntimeError):
    """Stands in for a timeout or preemption of the Modal container."""


def _count_fits(monkeypatch) -> list[int]:
    import aml.models.lgbm as lgbm_mod

    calls: list[int] = []
    real = lgbm_mod.fit_booster

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(lgbm_mod, "fit_booster", counting)
    return calls


def test_winning_trial_is_the_selection_seed_finalist(
    prepared, lgbm_cfg, rules_cfg, tmp_path, monkeypatch
) -> None:
    """The finalist of the seed the search ran with is the winning trial itself: no second fit,
    and its val AP is the search maximum, reported apart from the fresh seeds."""
    calls = _count_fits(monkeypatch)
    summary = run_lgbm_stage(prepared, tmp_path, lgbm_cfg, rules_cfg, threads=THREADS)
    sel = lgbm_cfg["optuna"]["seed"]
    assert sel in lgbm_cfg["seeds"] and summary["selection_seed"] == sel
    n = lgbm_cfg["optuna"]["n_trials"]
    assert len(calls) == n + len(lgbm_cfg["seeds"]) - 1
    assert summary["best_val_ap"][f"s{sel}"] == summary["best_trial_value"]
    fresh = [v for k, v in summary["best_val_ap"].items() if k != f"s{sel}"]
    assert summary["best_val_ap_mean_fresh"] == pytest.approx(np.mean(fresh))


def test_rerun_resumes_from_checkpoints(
    stage, prepared, lgbm_cfg, rules_cfg, tmp_path, monkeypatch
) -> None:
    """Killed after the search and the first finalist: the rerun replays the finished trials
    and loads the saved finalist, fits only what is missing, and ends with the same scores."""
    out, _ = stage
    n, seeds = lgbm_cfg["optuna"]["n_trials"], lgbm_cfg["seeds"]
    commits: list[int] = []

    def kill_after_first_finalist() -> None:
        commits.append(1)
        if len(commits) == n + 1:
            raise _Killed

    with pytest.raises(_Killed):
        run_lgbm_stage(
            prepared,
            tmp_path,
            lgbm_cfg,
            rules_cfg,
            threads=THREADS,
            on_checkpoint=kill_after_first_finalist,
        )
    assert len((tmp_path / "trials.jsonl").read_text(encoding="utf-8").splitlines()) == n
    assert (tmp_path / f"booster_s{seeds[0]}.json").is_file()
    assert not (tmp_path / "summary.json").exists()

    calls = _count_fits(monkeypatch)
    summary = run_lgbm_stage(prepared, tmp_path, lgbm_cfg, rules_cfg, threads=THREADS)
    assert len(calls) == len(seeds) - 1  # only the missing finalist
    assert summary["resumed_trials"] == n and not summary["checkpoints_reset"]
    a, b = _scores(tmp_path), _scores(out)
    assert a.select("row_id", "split").equals(b.select("row_id", "split"))
    for s in seeds:
        c = score_column(s)
        np.testing.assert_allclose(a[c].to_numpy(), b[c].to_numpy(), rtol=0, atol=1e-12)
    assert read_json(tmp_path / "best_params.json") == read_json(out / "best_params.json")


def test_search_has_a_wall_clock_bound(prepared, lgbm_cfg, rules_cfg, tmp_path) -> None:
    cfg = copy.deepcopy(lgbm_cfg)
    cfg["optuna"]["timeout_s"] = 1e-6  # checked after each trial: exactly one runs
    summary = run_lgbm_stage(prepared, tmp_path, cfg, rules_cfg, threads=THREADS)
    assert summary["n_trials"] == 1 < summary["trials_requested"]
    assert summary["tuning_timed_out"] is True
    assert (tmp_path / "booster_s0.txt").is_file()


def test_stale_checkpoints_are_discarded(prepared, lgbm_cfg, rules_cfg, tmp_path) -> None:
    run_lgbm_stage(prepared, tmp_path, lgbm_cfg, rules_cfg, threads=THREADS)
    other = copy.deepcopy(lgbm_cfg)
    other["num_boost_round"] = 30
    summary = run_lgbm_stage(prepared, tmp_path, other, rules_cfg, threads=THREADS)
    assert summary["checkpoints_reset"] is True and summary["resumed_trials"] == 0
    fresh = run_lgbm_stage(prepared, tmp_path / "fresh", other, rules_cfg, threads=THREADS)
    assert fresh["trials"] == summary["trials"]
    assert _scores(tmp_path / "fresh").equals(_scores(tmp_path))


def test_trial_checkpoint_ignores_a_torn_last_line(tmp_path) -> None:
    from aml.models.lgbm import load_trial_checkpoint

    p = tmp_path / "trials.jsonl"
    p.write_text('{"number": 0, "value": 0.5}\n{"number": 1, "val', encoding="utf-8")
    assert load_trial_checkpoint(p) == [{"number": 0, "value": 0.5}]
    assert load_trial_checkpoint(tmp_path / "missing.jsonl") == []


def test_resumed_search_shares_the_time_budget_and_redraws(lgbm_cfg, tmp_path) -> None:
    """A resumed search gets only the rest of `optuna.timeout_s` (records carry the search's
    wall time), and its sampler is re-seeded: in the random startup phase it does not redraw the
    points already tried (with the original seed, trial 1 of a resume repeated trial 0)."""
    import json

    from aml.models.lgbm import load_trial_checkpoint

    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 3)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=1500) > 1.5).astype(np.int8)
    cfg = copy.deepcopy(lgbm_cfg)
    cfg["optuna"].update(n_trials=1, n_startup_trials=5, timeout_s=None)
    names = {"feature_names": ["a", "b", "c"], "categorical": []}
    pair = make_datasets(X, y, X, y, cfg, data_seed=cfg["optuna"]["seed"], threads=1, **names)
    ck = tmp_path / "trials.jsonl"
    tune(*pair, cfg, threads=1, checkpoint=ck)
    (rec,) = load_trial_checkpoint(ck)
    assert rec["search_elapsed_s"] >= 0

    cfg["optuna"]["n_trials"] = 2  # resume: one more trial, still in the random startup phase
    _, trials, best = tune(*pair, cfg, threads=1, checkpoint=ck)
    assert best["resumed_trials"] == 1 and len(trials) == 2
    assert trials[0]["params"] == rec["params"] and trials[1]["params"] != rec["params"]
    recs = load_trial_checkpoint(ck)
    assert recs[1]["search_elapsed_s"] >= recs[0]["search_elapsed_s"]

    # The replayed trials used up the time budget: no new trial, the search counts as timed out.
    ck.write_text(
        "".join(json.dumps({**r, "search_elapsed_s": 100.0}) + "\n" for r in recs),
        encoding="utf-8",
    )
    cfg["optuna"].update(n_trials=4, timeout_s=50)
    _, trials, best = tune(*pair, cfg, threads=1, checkpoint=ck)
    assert len(trials) == 2 and best["timed_out"] is True and best["booster"] is None
