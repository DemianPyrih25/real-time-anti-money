"""LightGBM-graph stage on the fixture feature table (M2 spec §8.3): labels, gate, decision rule,
checkpoints, fit reuse, whitelist, SHAP, --nofmt-final; and the M1 generalisation."""

from __future__ import annotations

import copy
import json
import os
import shutil
import tempfile
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest
import yaml

from aml.features.spec import (
    NON_MODEL_COLUMNS,
    EngineSpec,
    GateStopError,
    parts_digest,
    scan_feature_table,
)
from aml.features.tx_features import TX_FEATURES
from aml.io import read_json, write_json_atomic, write_parquet_atomic
from aml.models import lgbm as m1
from aml.models.importance import sample_rows, shap_global
from aml.models.lgbm_graph import (
    ABLATION_VARIANTS,
    ablation_decision,
    check_graph_cfg,
    effective_graph_cfg,
    run_lgbm_graph_stage,
    variant_columns,
)
from aml.paths import DataPaths
from tests.fixtures.feature_table import build_feature_table

THREADS = 2
REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE_FILES = [
    "gate.json",
    "trials.jsonl",
    "trials.json",
    "best_params.json",
    "ablation.jsonl",
    "ablation.json",
    "scores.parquet",
    "feature_names.json",
    "shap_global.json",
    "summary.json",
    "checkpoint.json",
]


@pytest.fixture(scope="module")
def table_dir(prepared, tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("features")
    build_feature_table(prepared, d)
    return d


@pytest.fixture(scope="module")
def spec(table_dir) -> EngineSpec:
    return EngineSpec.from_json(read_json(table_dir / "feature_spec.json"))


@pytest.fixture(scope="module")
def graph_cfg(lgbm_cfg) -> dict:
    """conftest's small M1 settings; 3 trials, ablation seeds 0-2, finalists 0-3."""
    cfg = copy.deepcopy(lgbm_cfg)
    cfg["seeds"] = [0, 1, 2, 3]
    cfg["graph"]["optuna"] = {"n_trials": 3, "n_startup_trials": 2}
    cfg["graph"]["ablation"]["seeds"] = [0, 1, 2]
    cfg["graph"]["shap"]["negatives"] = 300
    return cfg


@pytest.fixture(scope="module")
def stage(prepared, table_dir, graph_cfg, tmp_path_factory):
    out = tmp_path_factory.mktemp("lgbm_graph")
    nofmt = tmp_path_factory.mktemp("lgbm_graph_nofmt")
    summary = run_lgbm_graph_stage(
        prepared, table_dir, out, graph_cfg, threads=THREADS, nofmt_final=True, nofmt_dir=nofmt
    )
    return out, nofmt, summary


def _table(table_dir: Path, names) -> pl.DataFrame:
    return scan_feature_table(table_dir, ["row_id", "split", *names]).collect()


def _count_fits(monkeypatch) -> list[int]:
    calls: list[int] = []
    real = m1.fit_booster

    def counting(*a, **k):
        calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(m1, "fit_booster", counting)
    return calls


# --------------------------------------------------------------------------- config


def test_effective_graph_cfg_merges_optuna_and_drops_graph():
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "lgbm.yaml").read_text(encoding="utf-8"))
    before = copy.deepcopy(cfg)
    eff = effective_graph_cfg(cfg)
    assert cfg == before  # not mutated
    assert "graph" not in eff and eff["feature_set"] == "graph"
    g = cfg["graph"]["optuna"]
    assert eff["optuna"] == {**cfg["optuna"], **g}
    assert eff["optuna"]["space"] == cfg["optuna"]["space"]
    for k in cfg:
        if k not in ("graph", "optuna", "feature_set"):
            assert eff[k] == cfg[k], k
    # Any M1 key can be overridden; dicts merge one level deep.
    cfg2 = copy.deepcopy(cfg)
    cfg2["graph"]["base_params"] = {"num_leaves": 31}
    eff2 = effective_graph_cfg(cfg2)
    assert eff2["base_params"] == {**cfg["base_params"], "num_leaves": 31}
    cfg2["graph"]["no_such_setting"] = 1
    with pytest.raises(ValueError, match="no_such_setting"):
        effective_graph_cfg(cfg2)


def test_variant_columns(spec):
    kept = [n for n in spec.feature_names if n not in ("cyc3_2d", "out_port")]
    v = variant_columns(spec, kept)
    assert list(v) == list(ABLATION_VARIANTS)
    assert v["full"] == kept and v["no_gate"] == list(spec.feature_names)
    assert v["-CYC"] == [n for n in kept if spec.feature(n).group != "CYC"]
    assert v["nofmt"] == [n for n in kept if n not in spec.format_derived_names]
    assert "payment_format" not in v["nofmt"] and len(spec.format_derived_names) == 9
    with pytest.raises(ValueError):
        variant_columns(spec, ["not_a_feature"])


# --------------------------------------------------------------------------- the decision rule


def _aps(base: float = 0.10, **means: float) -> dict[str, list[float]]:
    """Every variant at `base` (or its own mean) with seed APs mean -/+ 0.001 (std = 0.001414)."""
    out = {}
    for v in ABLATION_VARIANTS:
        m = means.get(v.replace("-", "minus_"), base)
        out[v] = [m - 0.001, m + 0.001]
    return out


def test_decision_sigma_is_the_pooled_seed_std():
    ap = _aps()
    ap["no_gate"] = [0.1, 0.104]  # std 0.002828
    d = ablation_decision(ap, 2.0)
    s2 = (9 * 2e-6 + 8e-6) / 10  # mean of s_v^2: nine variants at 2e-6, no_gate at 8e-6
    assert d["sigma"] == pytest.approx(np.sqrt(s2), rel=1e-9)
    assert d["threshold"] == pytest.approx(2.0 * d["sigma"])
    assert d["variants"]["full"]["std"] == pytest.approx(0.002 / np.sqrt(2))


def test_decision_keeps_full_when_no_group_clears_the_bar():
    d = ablation_decision(_aps(minus_VEL=0.1025), 2.0)  # delta 0.0025 < 2 sigma = 0.002828
    assert d["threshold"] == pytest.approx(2 * np.sqrt(2e-6))
    assert d["champion"] == "full" and d["dropped_group"] is None
    assert d["best_group_variant"] == "-VEL" and d["best_delta"] == pytest.approx(0.0025)
    assert ablation_decision(_aps(minus_VEL=0.1030), 2.0)["champion"] == "-VEL"  # 0.003 > bar
    assert d["variants"]["full"]["decision"] == "champion"


def test_decision_drops_at_most_the_one_best_group():
    d = ablation_decision(_aps(minus_PORT=0.11, minus_SG=0.12), 2.0)
    assert d["champion"] == "-SG" and d["dropped_group"] == "SG"
    assert d["variants"]["-PORT"]["decision"] == "not chosen"
    assert d["variants"]["-SG"]["decision"] == "champion"


def test_decision_ties_go_to_the_first_group():
    d = ablation_decision(_aps(minus_AMT=0.12, minus_RULE=0.12), 2.0)
    assert d["deltas"]["-AMT"] == d["deltas"]["-RULE"]
    assert d["champion"] == "-AMT"


def test_decision_never_picks_no_gate_or_nofmt():
    d = ablation_decision(_aps(no_gate=0.2, nofmt=0.3), 2.0)
    assert d["champion"] == "full"
    assert d["variants"]["nofmt"]["decision"] == "reported only"
    assert d["variants"]["nofmt"]["delta"] == pytest.approx(0.2)


def test_decision_bar_is_strict():
    flat = {v: [0.1, 0.1] for v in ABLATION_VARIANTS}  # sigma = 0: the bar is 0
    d = ablation_decision(flat, 2.0)
    assert d["sigma"] == 0.0 and d["champion"] == "full"  # delta 0 is not > 0
    flat["-CYC"] = [0.1001, 0.1001]
    assert ablation_decision(flat, 2.0)["champion"] == "-CYC"


def test_decision_rejects_bad_tables():
    with pytest.raises(ValueError, match="full"):
        ablation_decision({"-VEL": [0.1, 0.2]}, 2.0)
    with pytest.raises(ValueError, match="unknown"):
        ablation_decision({"full": [0.1, 0.2], "-TIME": [0.1, 0.2]}, 2.0)
    with pytest.raises(ValueError, match=">= 2"):
        ablation_decision({"full": [0.1]}, 2.0)
    with pytest.raises(ValueError, match=">= 2"):
        ablation_decision({"full": [0.1, float("nan")]}, 2.0)
    with pytest.raises(ValueError):
        ablation_decision(_aps(), -1.0)


# --------------------------------------------------------------------------- the stage


def test_stage_writes_every_file(stage, graph_cfg):
    out, _, summary = stage
    for name in STAGE_FILES + [f"booster_s{s}.{e}" for s in (0, 1, 2, 3) for e in ("txt", "json")]:
        assert (out / name).is_file(), name
    assert not list(out.rglob(".*.tmp-*"))
    assert read_json(out / "summary.json") == summary
    assert summary["feature_set"] == "graph" and summary["seeds"] == [0, 1, 2, 3]
    assert summary["n_trials"] == 3 == len(read_json(out / "trials.json"))
    assert set(summary["positives"]) == {"train", "val_early"}


def test_scores_cover_the_scored_splits_in_rank_order(stage, table_dir):
    out, _, _ = stage
    scores = pl.read_parquet(out / "scores.parquet")
    assert scores.columns == ["row_id", "split", *(f"score_s{s}" for s in range(4))]
    keys = _table(table_dir, []).filter(pl.col("split").is_in(m1.SCORE_SPLITS))
    assert scores.select("row_id", "split").equals(keys)
    for c in scores.columns[2:]:
        s = scores[c].to_numpy()
        assert scores.schema[c] == pl.Float64 and ((s >= 0) & (s <= 1)).all()
    assert not np.array_equal(scores["score_s0"].to_numpy(), scores["score_s1"].to_numpy())


def test_model_inputs_are_whitelisted_gated_spec_features(stage, spec):
    out, _, summary = stage
    names = read_json(out / "feature_names.json")
    gate = read_json(out / "gate.json")
    assert names == summary["features"]
    spec.assert_model_inputs(names)
    assert not set(names) & set(NON_MODEL_COLUMNS)
    assert set(names) <= set(gate["kept"]) and not set(names) & set(gate["dropped"])
    champion = summary["variant"]
    assert names == variant_columns(spec, gate["kept"])[champion]
    for s in (0, 1, 2, 3):
        b = lgb.Booster(model_file=str(out / f"booster_s{s}.txt"))
        assert b.feature_name() == names
        spec.assert_model_inputs(b.feature_name())


def test_scores_reproduce_bit_exactly_from_the_saved_boosters(stage, table_dir):
    """What serving does: float32 inputs from the table, the saved file, single-row predict."""
    out, _, _ = stage
    names = read_json(out / "feature_names.json")
    rows = _table(table_dir, names).filter(pl.col("split").is_in(m1.SCORE_SPLITS))
    X = rows.select(names).to_numpy()
    assert X.dtype == np.float32
    scores = pl.read_parquet(out / "scores.parquet")
    for s in (0, 1, 2, 3):
        b = lgb.Booster(model_file=str(out / f"booster_s{s}.txt"))
        assert np.array_equal(b.predict(X, num_threads=1), scores[f"score_s{s}"].to_numpy())
    b = lgb.Booster(model_file=str(out / "booster_s0.txt"))
    one = [b.predict(np.ascontiguousarray(X[i : i + 1]), num_threads=1)[0] for i in range(50)]
    assert np.array_equal(np.array(one), scores["score_s0"].to_numpy()[:50])


def test_ablation_records_and_decision(stage, graph_cfg):
    out, _, summary = stage
    recs = [json.loads(x) for x in (out / "ablation.jsonl").read_text("utf-8").splitlines()]
    seeds = graph_cfg["graph"]["ablation"]["seeds"]
    assert sorted((r["variant"], r["seed"]) for r in recs) == sorted(
        (v, s) for v in ABLATION_VARIANTS for s in seeds
    )
    by = {(r["variant"], r["seed"]): r for r in recs}
    # The winning trial is the full fit of the selection seed (no second fit).
    assert by[("full", 0)]["source"] == f"trial:{summary['best_trial']}"
    assert by[("full", 0)]["val_early_ap"] == summary["best_trial_value"]
    doc = read_json(out / "ablation.json")
    ap = {v: [by[(v, s)]["val_early_ap"] for s in seeds] for v in ABLATION_VARIANTS}
    want = ablation_decision(ap, graph_cfg["graph"]["ablation"]["margin_std"])
    assert doc["champion"] == want["champion"] == summary["variant"]
    assert doc["sigma"] == pytest.approx(want["sigma"]) and doc["order"] == list(ABLATION_VARIANTS)
    gate = read_json(out / "gate.json")
    assert doc["variants"]["no_gate"]["added"] == gate["dropped"]
    for v, info in doc["variants"].items():
        assert info["mean"] == pytest.approx(np.mean(ap[v]))
        if v.startswith("-"):
            assert all(n in gate["kept"] for n in info["removed"]), v
            assert info["n_features"] == len(gate["kept"]) - len(info["removed"]), v
    # Variants with the same inputs share one fit (e.g. a group the gate removed entirely).
    for r in recs:
        if r["source"].startswith("same_as:"):
            src = by[(r["source"].removeprefix("same_as:"), r["seed"])]
            assert src["fit_key"] == r["fit_key"] and src["val_early_ap"] == r["val_early_ap"]


def test_finalists_reuse_the_ablation_fits(stage, graph_cfg):
    out, _, summary = stage
    assert summary["reused_ablation_fits"] == [0, 1, 2]
    doc = read_json(out / "ablation.json")
    champion = summary["variant"]
    for s in (0, 1, 2):
        meta = read_json(out / f"booster_s{s}.json")
        assert meta["best_score"] == doc["variants"][champion]["ap"][s]
        assert meta["variant"] == champion and meta["feature_names"] == summary["features"]


def test_shap_output(stage, spec, table_dir, prepared):
    out, _, summary = stage
    doc = read_json(out / "shap_global.json")
    names = summary["features"]
    assert set(doc["features"]) == set(names)
    assert sorted(doc["ranking"]) == sorted(names)
    assert sum(g["n_features"] for g in doc["groups"].values()) == len(names)
    for g, info in doc["groups"].items():
        members = [n for n in names if spec.feature(n).group == g]
        assert sorted(info["features"]) == sorted(members)
        total = sum(doc["features"][n]["mean_abs"] for n in members)
        assert info["mean_abs"] == pytest.approx(total, rel=1e-12)
    val = _table(table_dir, []).filter(pl.col("split") == "val_early")
    y = m1.load_labels(prepared.labels, val["row_id"])
    assert doc["n_positives"] == int(y.sum()) == summary["positives"]["val_early"]
    assert doc["n_rows"] == min(val.height, int(y.sum()) + 300)
    assert doc["model_seed"] == 0 and doc["split"] == "val_early"


def test_nofmt_final_layout(stage, spec, table_dir):
    out, nofmt, _ = stage
    for name in ["scores.parquet", "feature_names.json", "summary.json", "checkpoint.json"]:
        assert (nofmt / name).is_file(), name
    names = read_json(nofmt / "feature_names.json")
    assert not set(names) & set(spec.format_derived_names)
    gate = read_json(out / "gate.json")
    assert names == variant_columns(spec, gate["kept"])["nofmt"]
    scores = pl.read_parquet(nofmt / "scores.parquet")
    assert set(scores["split"].unique()) == {"val_late", "test"}  # no val_early, no train
    keys = _table(table_dir, []).filter(pl.col("split").is_in(["val_late", "test"]))
    assert scores.select("row_id", "split").equals(keys)
    s = read_json(nofmt / "summary.json")
    assert s["variant"] == "nofmt" and s["reused_ablation_fits"] == [0, 1, 2]
    X = _table(table_dir, names).filter(pl.col("split").is_in(["val_late", "test"]))
    X = X.select(names).to_numpy()
    for seed in (0, 1, 2, 3):
        b = lgb.Booster(model_file=str(nofmt / f"booster_s{seed}.txt"))
        assert b.feature_name() == names
        assert np.array_equal(b.predict(X, num_threads=1), scores[f"score_s{seed}"].to_numpy())


def test_labels_only_from_train_and_val_early(
    stage, prepared, table_dir, graph_cfg, tmp_path, monkeypatch
):
    """The labels file holds train + val_early rows only (val_late / test rows deleted): the
    stage still runs and every output equals the full run's; load_labels sees nothing else.
    Also counts fits: the finalists of the ablation seeds are not refitted."""
    out, _, summary = stage
    vol = DataPaths(tmp_path / "volume")
    for name in ("transactions", "accounts"):
        getattr(vol, name).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(getattr(prepared, name), getattr(vol, name))
    tx = pl.read_parquet(prepared.transactions, columns=["row_id", "split"])
    fit_ids = tx.filter(pl.col("split").is_in(["train", "val_early"]))["row_id"]
    labels = pl.read_parquet(prepared.labels).filter(pl.col("row_id").is_in(fit_ids.implode()))
    write_parquet_atomic(labels, vol.labels)

    split_of = dict(zip(tx["row_id"], tx["split"], strict=True))
    seen: set[str] = set()
    real_load = m1.load_labels

    def spy(path, row_ids):
        seen.update(split_of[r] for r in row_ids.to_list())
        return real_load(path, row_ids)

    monkeypatch.setattr(m1, "load_labels", spy)
    calls = _count_fits(monkeypatch)
    again = run_lgbm_graph_stage(vol, table_dir, tmp_path / "out", graph_cfg, threads=THREADS)
    assert seen == {"train", "val_early"}
    assert pl.read_parquet(tmp_path / "out" / "scores.parquet").equals(
        pl.read_parquet(out / "scores.parquet")
    )
    assert again["best_val_ap"] == summary["best_val_ap"]
    assert read_json(tmp_path / "out" / "ablation.json") == read_json(out / "ablation.json")
    assert not (vol.root / "models").exists()  # no nofmt dir without --nofmt-final

    recs = [json.loads(x) for x in (out / "ablation.jsonl").read_text("utf-8").splitlines()]
    fresh_ablation = sum(r["source"] == "fit" for r in recs)
    # 3 trials + the fresh ablation fits + one finalist (seed 3; seeds 0-2 are ablation fits).
    assert len(calls) == 3 + fresh_ablation + 1


def test_resume_after_a_kill_and_stale_reset(stage, prepared, table_dir, graph_cfg, tmp_path):
    out, _, summary = stage
    n_trials, killed_after = 3, 5  # ablation records committed before the kill

    class Killed(RuntimeError):
        pass

    commits: list[int] = []

    def kill() -> None:
        commits.append(1)
        # gate.json, the trials, the end of the search (tuning_done.json), ablation fits
        if len(commits) == 1 + n_trials + 1 + killed_after:
            raise Killed

    with pytest.raises(Killed):
        run_lgbm_graph_stage(
            prepared, table_dir, tmp_path, graph_cfg, threads=THREADS, on_checkpoint=kill
        )
    assert len((tmp_path / "ablation.jsonl").read_text("utf-8").splitlines()) == killed_after
    assert not (tmp_path / "summary.json").exists()

    again = run_lgbm_graph_stage(prepared, table_dir, tmp_path, graph_cfg, threads=THREADS)
    assert again["resumed_trials"] == n_trials and not again["checkpoints_reset"]
    assert again["ablation"]["resumed_fits"] == killed_after
    assert pl.read_parquet(tmp_path / "scores.parquet").equals(
        pl.read_parquet(out / "scores.parquet")
    )
    assert (
        read_json(tmp_path / "ablation.json")["variants"]
        == read_json(out / "ablation.json")["variants"]
    )

    other = copy.deepcopy(graph_cfg)
    other["num_boost_round"] = 30
    reset = run_lgbm_graph_stage(prepared, table_dir, tmp_path, other, threads=THREADS)
    assert reset["checkpoints_reset"] is True and reset["resumed_trials"] == 0
    assert reset["ablation"]["resumed_fits"] == 0
    lines = (tmp_path / "ablation.jsonl").read_text("utf-8").splitlines()
    keys = {json.loads(x)["fit_key"] for x in lines}
    on_disk = {p.name.split("_")[1] for p in (tmp_path / "ablation").glob("*.json")}
    assert on_disk == keys  # the stale ablation boosters were deleted


def test_gate_stop_writes_gate_json_and_fits_nothing(prepared, table_dir, graph_cfg, tmp_path):
    cfg = copy.deepcopy(graph_cfg)
    cfg["graph"]["gate"]["min_train_nonzero"] = 10**9  # every feature fails the floor
    with pytest.raises(GateStopError, match="engine features"):
        run_lgbm_graph_stage(prepared, table_dir, tmp_path, cfg, threads=THREADS)
    gate = read_json(tmp_path / "gate.json")
    assert gate["stop"] is True and gate["kept"] == []
    for name in ("trials.jsonl", "ablation.jsonl", "summary.json", "checkpoint.json"):
        assert not (tmp_path / name).exists(), name


def test_stage_refuses_a_partial_table(prepared, table_dir, graph_cfg, tmp_path):
    partial = tmp_path / "features"
    shutil.copytree(table_dir, partial)
    (sorted((partial / "parts").glob("*.parquet"))[-1]).unlink()
    with pytest.raises(ValueError, match="partial build"):
        run_lgbm_graph_stage(prepared, partial, tmp_path / "out", graph_cfg, threads=THREADS)


def test_a_finished_search_is_replayed_never_reopened(
    prepared, table_dir, graph_cfg, tmp_path, monkeypatch
):
    """optuna.timeout_s cut the search (1 of 3 trials). A re-run of the finished stage (a resume
    or --nofmt-final) replays it from tuning_done.json instead of running the missing trials:
    no fit, the same champion files, TreeSHAP reused. --nofmt-final fits only the nofmt seeds
    outside the ablation seeds and refuses to rewrite a champion whose params would change."""
    cfg = copy.deepcopy(graph_cfg)
    cfg["graph"]["optuna"]["timeout_s"] = 1e-6  # checked after each trial: one runs
    out = tmp_path / "out"
    first = run_lgbm_graph_stage(prepared, table_dir, out, cfg, threads=THREADS)
    assert first["tuning_timed_out"] is True and first["n_trials"] == 1
    assert first["trials_requested"] == 3 and first["shap_reused"] is False
    marker = read_json(out / "tuning_done.json")
    assert marker["n_trials"] == 1 and marker["timed_out"] is True
    kept = ("best_params.json", "booster_s0.txt", "booster_s3.txt", "shap_global.json")
    before = {n: (out / n).read_bytes() for n in kept}
    scores = pl.read_parquet(out / "scores.parquet")

    asked: list[int] = []
    real_tune = m1.tune

    def spy(*a, **k):
        asked.append(int(a[2]["optuna"]["n_trials"]))
        return real_tune(*a, **k)

    monkeypatch.setattr(m1, "tune", spy)
    calls = _count_fits(monkeypatch)
    again = run_lgbm_graph_stage(prepared, table_dir, out, cfg, threads=THREADS)
    assert asked == [1] and calls == []  # the search was not extended to 3 trials
    assert again["tuning_replayed"] is True and again["tuning_timed_out"] is True
    assert again["n_trials"] == 1 and again["variant"] == first["variant"]
    assert again["shap_reused"] is True and again["best_params"] == first["best_params"]
    for name, data in before.items():
        assert (out / name).read_bytes() == data, name
    assert pl.read_parquet(out / "scores.parquet").equals(scores)

    nofmt = tmp_path / "nofmt"
    nf = run_lgbm_graph_stage(
        prepared, table_dir, out, cfg, threads=THREADS, nofmt_final=True, nofmt_dir=nofmt
    )
    assert len(calls) == 1  # nofmt seed 3; seeds 0-2 are ablation fits
    assert nf["variant"] == first["variant"] and (nofmt / "summary.json").is_file()
    for name, data in before.items():
        assert (out / name).read_bytes() == data, name

    # A finished stage that the replay would not reproduce: refused before anything is touched.
    doc = read_json(out / "summary.json")
    doc["best_params"] = {**doc["best_params"], "num_leaves": 2}
    write_json_atomic(doc, out / "summary.json")
    with pytest.raises(RuntimeError, match="differ from the finished stage"):
        run_lgbm_graph_stage(
            prepared, table_dir, out, cfg, threads=THREADS, nofmt_final=True, nofmt_dir=nofmt
        )
    assert read_json(out / "summary.json") == doc and (out / "scores.parquet").is_file()


def test_a_rerun_that_dies_leaves_no_old_summary(prepared, table_dir, graph_cfg, tmp_path):
    """summary.json marks a finished stage: a re-run removes it (and the scores) before it
    rewrites any finished output, so a run that dies part-way is never taken as finished."""
    out = tmp_path / "out"
    run_lgbm_graph_stage(prepared, table_dir, out, graph_cfg, threads=THREADS)
    commits: list[int] = []

    class Killed(RuntimeError):
        pass

    def kill() -> None:
        commits.append(1)
        if len(commits) == 2:  # gate.json, then the end of the search (best_params.json)
            raise Killed

    with pytest.raises(Killed):
        run_lgbm_graph_stage(
            prepared, table_dir, out, graph_cfg, threads=THREADS, on_checkpoint=kill
        )
    assert not (out / "summary.json").exists() and not (out / "scores.parquet").exists()


def test_other_feature_values_reset_the_checkpoints(prepared, table_dir, graph_cfg, tmp_path):
    """The fingerprint covers the parts' content digest: a feature table re-built under the same
    spec with other values never resumes trials, ablation fits or finalists of the old one."""
    out = tmp_path / "out"
    first = run_lgbm_graph_stage(prepared, table_dir, out, graph_cfg, threads=THREADS)
    assert first["features_digest"] == parts_digest(table_dir)
    other = tmp_path / "features"
    shutil.copytree(table_dir, other)
    part = sorted((other / "parts").glob("*.parquet"))[0]
    name = first["features"][-1]
    df = pl.read_parquet(part)
    df.with_columns((pl.col(name) * 2 + 1).cast(pl.Float32)).write_parquet(part)
    again = run_lgbm_graph_stage(prepared, other, out, graph_cfg, threads=THREADS)
    assert again["checkpoints_reset"] is True and again["resumed_trials"] == 0
    assert again["ablation"]["resumed_fits"] == 0 and again["shap_reused"] is False
    assert again["features_digest"] == parts_digest(other) != first["features_digest"]


def test_ablation_config_checks():
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "lgbm.yaml").read_text(encoding="utf-8"))
    _, g = check_graph_cfg(cfg)
    assert g["ablation"]["seeds"] == [0, 1, 2]
    assert g["ablation"]["variants"] == list(ABLATION_VARIANTS)
    two = copy.deepcopy(cfg)
    two["graph"]["ablation"]["seeds"] = [0, 1]  # spec §10 cost cut 2: its sigma is not built
    with pytest.raises(ValueError, match="cost cut 2"):
        check_graph_cfg(two)
    cut = copy.deepcopy(cfg)
    cut["graph"]["ablation"]["variants"] = [v for v in ABLATION_VARIANTS if v != "no_gate"]
    assert "no_gate" not in check_graph_cfg(cut)[1]["ablation"]["variants"]
    for bad in (["full", "nofmt"], [*ABLATION_VARIANTS, "-TIME"], ["full", *ABLATION_VARIANTS]):
        cut["graph"]["ablation"]["variants"] = bad
        with pytest.raises(ValueError, match="variants"):
            check_graph_cfg(cut)


def test_ablation_without_no_gate(prepared, table_dir, graph_cfg, tmp_path):
    """Spec §10 cost cut 3 is a config edit: no no_gate fits, sigma pools the other variants."""
    cfg = copy.deepcopy(graph_cfg)
    variants = [v for v in ABLATION_VARIANTS if v != "no_gate"]
    cfg["graph"]["ablation"]["variants"] = variants
    s = run_lgbm_graph_stage(prepared, table_dir, tmp_path, cfg, threads=THREADS)
    doc = read_json(tmp_path / "ablation.json")
    assert doc["order"] == variants and s["ablation"]["variants"] == variants
    recs = [json.loads(x) for x in (tmp_path / "ablation.jsonl").read_text("utf-8").splitlines()]
    assert {r["variant"] for r in recs} == set(variants)
    seeds = cfg["graph"]["ablation"]["seeds"]
    by = {(r["variant"], r["seed"]): r["val_early_ap"] for r in recs}
    margin = cfg["graph"]["ablation"]["margin_std"]
    want = ablation_decision({v: [by[(v, x)] for x in seeds] for v in variants}, margin)
    assert doc["sigma"] == pytest.approx(want["sigma"]) and doc["champion"] == want["champion"]


# --------------------------------------------------------------------------- SHAP unit


def _toy_booster(seed: int = 0):
    rng = np.random.default_rng(seed)
    n = 3000
    X = rng.normal(size=(n, 4)).astype(np.float32)
    y = (rng.random(n) < 1 / (1 + np.exp(-(2 * X[:, 0] - X[:, 2] - 3)))).astype(np.int8)
    names = ["a", "b", "c", "d"]
    d = lgb.Dataset(X, label=y, feature_name=names, params={"verbose": -1})
    params = {"objective": "binary", "verbose": -1, "num_threads": 1, "deterministic": True}
    return lgb.train(params, d, num_boost_round=20), X, y, names


def test_shap_global_unit():
    booster, X, y, names = _toy_booster()
    groups = {"a": "G1", "b": "G1", "c": "G2", "d": "G3"}
    doc = shap_global(booster, X, y, names, groups, negatives=500, seed=3)
    idx = sample_rows(y, 500, 3)
    assert doc["n_rows"] == idx.size == int(y.sum()) + 500
    assert doc["n_positives"] == int(y.sum())
    contrib = booster.predict(X[idx], pred_contrib=True)
    raw = booster.predict(X[idx], raw_score=True)
    np.testing.assert_allclose(contrib.sum(axis=1), raw, rtol=0, atol=1e-9)
    want = np.abs(contrib[:, :-1]).mean(axis=0)
    for i, n in enumerate(names):
        assert doc["features"][n]["mean_abs"] == pytest.approx(want[i], rel=1e-12)
    assert doc["groups"]["G1"]["mean_abs"] == pytest.approx(want[0] + want[1], rel=1e-12)
    assert doc["ranking"][0] == "a" and doc["group_ranking"][0] == "G1"
    assert doc["budget_limited"] is False and doc["negatives_requested"] == 500
    timing = ("seconds",)
    again = shap_global(booster, X, y, names, groups, negatives=500, seed=3)
    assert {k: v for k, v in doc.items() if k not in timing} == {
        k: v for k, v in again.items() if k not in timing
    }
    assert not np.array_equal(idx, sample_rows(y, 500, 4))
    assert sample_rows(y, 10**9, 0).size == y.size  # more negatives than exist: all rows
    with pytest.raises(ValueError):
        shap_global(booster, X, y, ["a", "b", "c", "x"], groups, negatives=5, seed=0)
    with pytest.raises(ValueError):
        sample_rows(y, -1, 0)
    with pytest.raises(ValueError, match="no group"):
        shap_global(booster, X, y, names, {"a": "G1"}, negatives=5, seed=0)


def test_shap_time_budget_keeps_positives_and_a_uniform_prefix(monkeypatch):
    from aml.models import importance

    booster, X, y, names = _toy_booster()
    groups = dict.fromkeys(names, "G")
    monkeypatch.setattr(importance, "CHUNK_ROWS", 100)
    doc = shap_global(booster, X, y, names, groups, negatives=2000, seed=1, max_seconds=0.0)
    n_pos = int(y.sum())
    done = -(-n_pos // 100) * 100  # whole chunks until every positive is explained
    assert doc["budget_limited"] is True and doc["n_positives"] == n_pos
    assert doc["n_rows"] == done and doc["n_negatives"] == done - n_pos
    rows = importance.sample_order(y, 2000, 1)[:done]  # positives, then the seeded prefix
    want = np.abs(booster.predict(X[rows], pred_contrib=True)[:, :-1]).mean(axis=0)
    for i, n in enumerate(names):
        assert doc["features"][n]["mean_abs"] == pytest.approx(want[i], rel=1e-12)
    full = shap_global(booster, X, y, names, groups, negatives=2000, seed=1, max_seconds=1e9)
    assert full["budget_limited"] is False and full["n_negatives"] == 2000


# --------------------------------------------------------------------------- M1 generalisation


def test_m1_helpers_keep_their_defaults(prepared):
    tx = pl.read_parquet(prepared.transactions).head(200)
    from aml.features.tx_features import build_tx_features, fit_vocab

    feats = build_tx_features(tx, fit_vocab(tx), 100)
    X = m1.to_matrix(feats)
    assert X.dtype == np.float64 and X.shape == (200, len(TX_FEATURES))
    np.testing.assert_array_equal(X, feats.select(TX_FEATURES).to_numpy().astype(np.float64))
    X32 = m1.to_matrix(feats, ["hour_of_day", "self_loop"], np.float32)
    assert X32.dtype == np.float32 and X32.shape == (200, 2)
    with pytest.raises(ValueError, match="not allowed"):
        m1.to_matrix(feats.with_columns(minute=pl.lit(1)), ["hour_of_day", "minute"])


def test_make_datasets_checks_names(lgbm_cfg):
    X = np.zeros((50, 3), np.float32)
    y = np.r_[np.zeros(45), np.ones(5)].astype(np.int8)
    with pytest.raises(ValueError, match="feature names"):
        m1.make_datasets(X, y, X, y, lgbm_cfg, data_seed=0, threads=1)  # 9 TX names
    with pytest.raises(ValueError, match="categorical"):
        m1.make_datasets(
            X, y, X, y, lgbm_cfg, data_seed=0, threads=1, feature_names=["a", "b", "c"]
        )
    dtr, _ = m1.make_datasets(
        X, y, X, y, lgbm_cfg, data_seed=0, threads=1, feature_names=["a", "b", "c"], categorical=[]
    )
    assert dtr.feature_name == ["a", "b", "c"]


def test_finalist_meta_must_match_to_resume(lgbm_cfg, tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(1500, 3)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=1500) > 1.5).astype(np.int8)
    cfg = copy.deepcopy(lgbm_cfg)
    cfg["seeds"] = [0]
    kw = {"threads": 1, "checkpoint_dir": tmp_path, "feature_names": ["a", "b", "c"]}
    kw["categorical"] = []
    m1.train_finalists(X, y, X, y, {}, cfg, extra_meta={"variant": "full"}, **kw)
    assert read_json(tmp_path / "booster_s0.json")["variant"] == "full"
    calls = _count_fits(monkeypatch)
    m1.train_finalists(X, y, X, y, {}, cfg, extra_meta={"variant": "full"}, **kw)
    assert calls == []  # resumed
    m1.train_finalists(X, y, X, y, {}, cfg, extra_meta={"variant": "-VEL"}, **kw)
    assert calls == [1]  # other inputs: refitted


def test_m1_stage_ignores_the_graph_section(prepared, lgbm_cfg, rules_cfg, tmp_path):
    """The M1 fingerprint (and every output) is the same with or without lgbm.yaml `graph`, so
    the real M1 checkpoints stay valid after M2 appended that section."""
    cfg = copy.deepcopy(lgbm_cfg)
    cfg["optuna"]["n_trials"] = 1
    cfg["seeds"] = [0]
    no_graph = {k: v for k, v in cfg.items() if k != "graph"}
    assert "graph" in cfg
    m1.run_lgbm_stage(prepared, tmp_path / "a", cfg, rules_cfg, threads=THREADS)
    m1.run_lgbm_stage(prepared, tmp_path / "b", no_graph, rules_cfg, threads=THREADS)
    for name in ("checkpoint.json", "scores.parquet", "best_params.json", "booster_s0.txt"):
        a, b = (tmp_path / d / name for d in "ab")
        assert a.read_bytes() == b.read_bytes(), name


# --------------------------------------------------------------------------- the Modal job


def _job(name: str):
    import importlib

    os.environ.setdefault(
        "MODAL_CONFIG_PATH", str(Path(tempfile.gettempdir()) / "aml-tests-no-modal.toml")
    )
    return importlib.import_module(f"modal_jobs.{name}")


class _Remote:
    def __init__(self) -> None:
        self.args: tuple = ()

    def remote(self, *args):
        self.args = args
        return {"ok": True}


def test_train_job_routes_feature_sets(monkeypatch):
    job = _job("train_lgbm")
    common = _job("common")
    tx, graph = _Remote(), _Remote()
    monkeypatch.setattr(job, "train", tx)
    monkeypatch.setattr(job, "train_graph", graph)
    main = job.main.info.raw_f
    main()
    lgbm_cfg, _, _, keys = tx.args
    cfgs = common.load_all_configs()
    assert "graph" not in lgbm_cfg and lgbm_cfg == common.lgbm_tx_cfg(cfgs["lgbm"])
    assert keys["lgbm_tx"] == common.lgbm_key(cfgs) and not graph.args
    main(feature_set="graph", nofmt_final=True)
    lgbm_cfg, _, keys, nofmt = graph.args
    assert "graph" in lgbm_cfg and nofmt is True
    assert keys == {
        "data": common.data_key(cfgs),
        "features": common.features_key(cfgs),
        "lgbm_graph": common.lgbm_graph_key(cfgs),
    }
    with pytest.raises(SystemExit):
        main(feature_set="gnn")
    with pytest.raises(SystemExit):
        main(feature_set="tx", nofmt_final=True)
    assert job.CPU == 8.0 and job.FEATURE_SETS == ("tx", "graph")
