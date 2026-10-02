"""End-to-end M2 pipeline on the synthetic fixture (M2 spec §11.3 steps 4-6), every stage real.

prepare (conftest) -> M1 rules (SQL) + M1 LightGBM-tx -> build_features (full, then verify with
the DuckDB oracle) -> rules from engine severities (parity with the SQL, M1 regression) ->
LightGBM-graph on the real feature table -> evaluate (lgbm_tx + lgbm_graph + rules_engine, with
the validation-only extras) -> export (the serving bundle) -> verify_bundle.

Every library stage is called the way its Modal job calls it, into the Volume layout of
`DataPaths`; the stage directories get the `data_version.json` stamp the jobs write. Only the
fit budgets are small (conftest's lgbm settings, 3 graph trials, 3 ablation seeds).
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

from aml.config import rate_tag
from aml.eval.report import render_markdown, run_evaluate_stage
from aml.features import build
from aml.features.spec import PARTS_DIR, SNAPSHOTS_DIR, scan_feature_table
from aml.io import read_json, write_json_atomic
from aml.models.lgbm import SCORE_SPLITS, predict, run_lgbm_stage
from aml.models.lgbm_graph import (
    ABLATION_FILE,
    ABLATION_VARIANTS,
    GATE_FILE,
    SHAP_FILE,
    run_lgbm_graph_stage,
)
from aml.paths import DataPaths
from aml.rules.scenarios import PARITY_FILE, run_rules_engine_stage
from aml.rules.sql_baseline import run_rules_stage
from aml.serving import bundle
from tests.conftest import load_yaml

pytestmark = pytest.mark.slow

THREADS = 2
BOOT_B = 50
DATA_VERSION = "fixture-v1"
KEYS = {  # stand-ins for modal_jobs.common.all_keys (directory names only)
    "data": "data-e2e",
    "rules": "rules-e2e",
    "rules_engine": "rules_engine-e2e",
    "features": "features-e2e",
    "lgbm_tx": "lgbm_tx-e2e",
    "lgbm_graph": "lgbm_graph-e2e",
    "eval": "eval-e2e",
    "export": "export-e2e",
}
VALIDATION = "## Validation-only M2 evidence"


def _cfgs(data_cfg: dict, rules_cfg: dict, lgbm_cfg: dict) -> dict[str, dict]:
    dcfg = copy.deepcopy(data_cfg)
    dcfg["evaluation"]["bootstrap_replicates"] = BOOT_B
    lcfg = copy.deepcopy(lgbm_cfg)
    lcfg["graph"]["optuna"] = {"n_trials": 3, "n_startup_trials": 2}
    lcfg["graph"]["ablation"]["seeds"] = [0, 1, 2]
    lcfg["graph"]["shap"]["negatives"] = 300
    return {
        "data": dcfg,
        "rules": copy.deepcopy(rules_cfg),
        "lgbm": lcfg,
        "features": load_yaml("features.yaml"),
        "serving": load_yaml("serving.yaml"),
    }


def _stamp(*dirs: Path) -> None:
    for d in dirs:
        write_json_atomic({"data_version": DATA_VERSION}, Path(d) / "data_version.json")


@pytest.fixture(scope="module")
def m2(prepared, data_cfg, rules_cfg, lgbm_cfg, tmp_path_factory) -> dict:
    """Run every stage once into a private copy of the prepared Volume."""
    paths = DataPaths(tmp_path_factory.mktemp("m2_volume"))
    for name in ("transactions", "accounts", "fx_rates", "labels"):
        dst = getattr(paths, name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(getattr(prepared, name), dst)
    cfgs = _cfgs(data_cfg, rules_cfg, lgbm_cfg)
    dcfg, rcfg = cfgs["data"], cfgs["rules"]
    d = {
        "rules": paths.model_dir("rules", KEYS["rules"]),
        "lgbm_tx": paths.model_dir("lgbm_tx", KEYS["lgbm_tx"]),
        "features": paths.features_dir(KEYS["features"]),
        "rules_engine": paths.model_dir("rules_engine", KEYS["rules_engine"]),
        "lgbm_graph": paths.model_dir("lgbm_graph", KEYS["lgbm_graph"]),
    }
    out: dict = {"paths": paths, "cfgs": cfgs, "dirs": d}
    # M1 stages (their outputs are the M1 regression reference and the lgbm_tx comparison).
    run_rules_stage(paths, d["rules"], rcfg, dcfg, threads=THREADS)
    run_lgbm_stage(paths, d["lgbm_tx"], cfgs["lgbm"], rcfg, threads=THREADS)
    # M2: features (full + verify), rules from the engine, LightGBM-graph.
    out["features"] = build.run_build_features(paths, d["features"], cfgs)
    out["verify"] = build.run_build_features(
        paths, d["features"], cfgs, mode="verify", threads=THREADS
    )
    out["rules_engine"] = run_rules_engine_stage(
        paths,
        d["features"],
        d["rules_engine"],
        rcfg,
        dcfg,
        threads=THREADS,
        m1_rules_dir=d["rules"],
    )
    out["lgbm_graph"] = run_lgbm_graph_stage(
        paths, d["features"], d["lgbm_graph"], cfgs["lgbm"], threads=THREADS
    )
    _stamp(d["features"], d["rules_engine"], d["lgbm_graph"])
    # Evaluate as modal_jobs.evaluate does: the rules_engine dir, both models, the extras.
    extras = {
        "gate": d["lgbm_graph"] / GATE_FILE,
        "ablation": d["lgbm_graph"] / ABLATION_FILE,
        "shap": d["lgbm_graph"] / SHAP_FILE,
        "engine": d["features"] / "summary.json",
        "parity": d["rules_engine"] / PARITY_FILE,
    }
    models = {"lgbm_tx": d["lgbm_tx"], "lgbm_graph": d["lgbm_graph"]}
    out["reports"] = paths.reports
    out["results"] = run_evaluate_stage(
        paths, models, d["rules_engine"], paths.reports, dcfg, rcfg, threads=THREADS, extras=extras
    )
    out["export"] = bundle.run_export(
        paths,
        paths.serving_dir,
        cfgs,
        KEYS,
        features_dir=d["features"],
        rules_dir=d["rules_engine"],
        graph_dir=d["lgbm_graph"],
        threads=THREADS,
    )
    return out


def _tx(m2: dict) -> pl.DataFrame:
    return pl.read_parquet(m2["paths"].transactions).sort("rank")


# --------------------------------------------------------------------------- step 4: features


def test_feature_build_parts_snapshots_restart_and_oracle(m2) -> None:
    s, tx, fd = m2["features"], _tx(m2), m2["dirs"]["features"]
    n_days = int(tx["day"].max())
    parts = sorted((fd / PARTS_DIR).glob("part-d*.parquet"))
    assert len(parts) == n_days == s["n_days"] == 18
    assert s["rows"] == tx.height and sum(p["rows"] for p in s["parts"]) == tx.height
    for split, minute in (("val_early", 8640), ("test", 11520)):
        info = s["snapshots"][split]
        first = int(tx.filter(pl.col("minute") >= minute)["rank"][0])
        assert info["minute"] == minute == info["clock"] and info["next_rank"] == first
        assert (fd / SNAPSHOTS_DIR / info["file"]).is_file()
    rc = s["restart_check"]
    assert rc["pass"] and rc["rows_equal"] and rc["digest_equal"] and rc["days"] == [7, 8]
    assert s["headline"]["restart_check_pass"] is True
    v = read_json(fd / "verify" / "verify.json")
    assert m2["verify"]["n_mismatches"] == 0 == v["n_mismatches_total"]
    assert v["rows"] == tx.height and v["columns_checked"] >= 65


# --------------------------------------------------------------------------- rules (engine)


def test_rules_from_engine_severities_reproduce_m1(m2) -> None:
    d = m2["dirs"]
    parity = read_json(d["rules_engine"] / PARITY_FILE)
    assert parity["ok"] and parity["mismatches_total"] == 0 and parity["rows"] == _tx(m2).height
    reg = parity["m1_regression"]
    assert reg["compared"] and reg["thresholds_doc_equal"] and reg["flags_equal"]
    assert reg["severities_equal"]
    for f in ("flags.parquet", "severities.parquet"):
        assert pl.read_parquet(d["rules_engine"] / f).equals(pl.read_parquet(d["rules"] / f)), f
    assert read_json(d["rules_engine"] / "thresholds.json") == read_json(
        d["rules"] / "thresholds.json"
    )


# --------------------------------------------------------------------------- step 5: graph model


def test_graph_stage_on_the_real_feature_table(m2) -> None:
    g, fd = m2["dirs"]["lgbm_graph"], m2["dirs"]["features"]
    s = m2["lgbm_graph"]
    spec = build.load_spec(fd)
    for f in (GATE_FILE, ABLATION_FILE, SHAP_FILE, "scores.parquet", "feature_names.json"):
        assert (g / f).is_file(), f
    assert s["spec_hash"] == spec.spec_hash() and s["variant"] in ABLATION_VARIANTS
    gate = read_json(g / GATE_FILE)
    assert not gate["stop"] and set(gate["kept"]) <= set(spec.feature_names)
    names = read_json(g / "feature_names.json")
    assert names == s["features"] and set(names) <= set(gate["kept"])
    seeds = m2["cfgs"]["lgbm"]["seeds"]
    # Scores = the saved booster files on the parts' Float32 values, bit for bit (what serving
    # computes), for every scored row in rank order.
    table = scan_feature_table(fd, ["row_id", "split", *names]).collect()
    scored = table.filter(pl.col("split").is_in(list(SCORE_SPLITS)))
    sc = pl.read_parquet(g / "scores.parquet")
    assert sc["row_id"].equals(scored["row_id"]) and sc["split"].equals(scored["split"])
    x = scored.select(names).to_numpy().astype(np.float32, copy=False)
    for seed in seeds:
        b = lgb.Booster(model_file=str(g / f"booster_s{seed}.txt"))
        assert b.feature_name() == names
        spec.assert_model_inputs(b.feature_name())
        want = predict(b, x, threads=1)
        assert np.array_equal(sc[f"score_s{seed}"].to_numpy().view(np.uint64), want.view(np.uint64))


# --------------------------------------------------------------------------- step 5: evaluate


def test_evaluation_has_both_models_and_the_engine_rules(m2) -> None:
    res = m2["results"]
    seeds = sorted(m2["cfgs"]["lgbm"]["seeds"])
    assert res["meta"]["models"] == {
        "lgbm_tx": {"n_seeds": len(seeds), "seeds": seeds},
        "lgbm_graph": {"n_seeds": len(seeds), "seeds": seeds},
    }
    assert res["meta"]["inputs"]["rules_dir"] == str(m2["dirs"]["rules_engine"])
    assert set(res["bootstrap"]["model_diffs"]) == {"lgbm_graph - lgbm_tx"}
    assert set(res["bootstrap"]["diffs"]) == {"lgbm_tx", "lgbm_graph"}  # each model - rules
    for view in ("primary", "tail", "full"):
        assert set(res["views"][view]["models"]) == {"lgbm_tx", "lgbm_graph"}
    md = (m2["reports"] / "results.md").read_text(encoding="utf-8")
    assert md == render_markdown(res)
    start, end = md.index(VALIDATION), md.index("## Literature reference")
    assert md.index("## Memorisation check at (a)") < start < end
    block = md[start:end]
    n_rows = _tx(m2).height
    for needle in (
        "### Feature gate (label-free, before any fit)",
        "### Group ablation on val_early (tuned parameters fixed)",
        f"the champion is {m2['lgbm_graph']['variant']}",
        "### TreeSHAP of the champion (seed 0, val_early)",
        "### Feature engine (build_features summary.json)",
        "| restart_check_pass | True |",
        f"| rows | {n_rows:,} |",
        "| us_per_event_engine |",
        "### Rule parity",
        f"Rule parity (engine severities vs the M1 SQL): 0 mismatches over {n_rows:,} rows "
        "(0 rows with rule_trunc = 1 are excluded).",
        "| m1_regression.flags_equal | True |",
    ):
        assert needle in block, needle
    for variant in ABLATION_VARIANTS:
        assert f"| {variant} |" in block, variant
    json.dumps(res, allow_nan=False)


def test_validation_block_is_additive_and_m1_report_unchanged(m2, tmp_path) -> None:
    """Without extras the report is exactly the one with the block removed; and the M1-only
    report (lgbm_tx) is byte-identical whether the rules come from the M1 SQL stage or from
    engine severities (same thresholds and flags)."""
    paths, d, cfgs = m2["paths"], m2["dirs"], m2["cfgs"]
    dcfg, rcfg = cfgs["data"], cfgs["rules"]
    models = {"lgbm_tx": d["lgbm_tx"], "lgbm_graph": d["lgbm_graph"]}
    plain = run_evaluate_stage(
        paths, models, d["rules_engine"], tmp_path / "plain", dcfg, rcfg, threads=THREADS
    )
    assert "validation" not in plain
    md = (m2["reports"] / "results.md").read_text(encoding="utf-8")
    md_plain = (tmp_path / "plain" / "results.md").read_text(encoding="utf-8")
    start, end = md.index(VALIDATION), md.index("## Literature reference")
    assert VALIDATION not in md_plain and md[:start] + md[end:] == md_plain

    m1 = {"lgbm_tx": d["lgbm_tx"]}
    a = run_evaluate_stage(paths, m1, d["rules"], tmp_path / "sql", dcfg, rcfg, threads=THREADS)
    b = run_evaluate_stage(
        paths, m1, d["rules_engine"], tmp_path / "eng", dcfg, rcfg, threads=THREADS
    )
    assert (tmp_path / "sql" / "results.md").read_bytes() == (
        tmp_path / "eng" / "results.md"
    ).read_bytes()
    for r in (a, b):  # everything but the input paths and the timings
        del r["meta"]["inputs"], r["meta"]["seconds"], r["bootstrap"]["seconds"]
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_stage_outputs_name_the_parts_they_read(m2) -> None:
    """The rules and graph stages record the parts' digest; the checks the evaluate and export
    jobs run before touching test (same parts, passed oracle) accept the real outputs."""
    import importlib
    import os
    import tempfile

    os.environ.setdefault(
        "MODAL_CONFIG_PATH", str(Path(tempfile.gettempdir()) / "aml-tests-no-modal.toml")
    )
    common = importlib.import_module("modal_jobs.common")
    d = m2["dirs"]
    digest = m2["features"]["features_digest"]
    assert m2["lgbm_graph"]["features_digest"] == digest == m2["rules_engine"]["features_digest"]
    stages = {"rules_engine": (d["rules_engine"], "rules"), "lgbm_graph": (d["lgbm_graph"], "x")}
    assert common.require_same_feature_table(d["features"], stages) == digest
    assert common.require_features_verified(d["features"])["features_digest"] == digest
    assert read_json(m2["paths"].serving_dir / bundle.METADATA_FILE)["features_digest"] == digest


# --------------------------------------------------------------------------- step 6: export


def test_bundle_from_the_real_stage_outputs(m2) -> None:
    paths, d, res = m2["paths"], m2["dirs"], m2["results"]
    sd = paths.serving_dir
    s = m2["export"]
    assert s["verification"]["mismatches"] == 0 and s["verification"]["rows"] == s["rows"] > 0
    meta = read_json(sd / bundle.METADATA_FILE)
    files = sorted(p.relative_to(sd).as_posix() for p in sd.rglob("*") if p.is_file())
    assert files == sorted([*bundle.BUNDLE_FILES, bundle.METADATA_FILE])
    newest = max((sd / rel).stat().st_mtime_ns for rel in bundle.BUNDLE_FILES)
    assert (sd / bundle.METADATA_FILE).stat().st_mtime_ns >= newest  # written last
    for rel, info in meta["files"].items():
        assert info["sha256"] == bundle._sha256(sd / rel), rel
    assert meta["data_version"] == DATA_VERSION and meta["keys"] == KEYS
    assert meta["champion"] == m2["lgbm_graph"]["variant"]
    spec_doc = read_json(sd / bundle.FEATURE_SPEC)
    assert spec_doc["model"]["feature_names"] == read_json(d["lgbm_graph"] / "feature_names.json")
    assert meta["spec_hash"] == m2["features"]["spec_hash"]
    # The snapshot is the features build's test boundary; the slice starts at its next rank.
    tx = _tx(m2)
    test = tx.filter(pl.col("split") == "test")
    assert meta["next_rank"] == int(test["rank"][0]) and meta["next_offset"] == 0
    assert meta["snapshot_state_digest"] == m2["features"]["snapshots"]["test"]["state_digest"]
    assert s["rows"] == min(m2["cfgs"]["serving"]["replay"]["max_events"], test.height)
    # The bundle's model thresholds are the evaluation's deployable (a) thresholds of seed 0.
    thr = read_json(sd / bundle.THRESHOLDS)
    ev = res["thresholds"]["lgbm_graph"]["rate"]
    seed_pos = res["meta"]["models"]["lgbm_graph"]["seeds"].index(bundle.SEED)
    for tag in thr["rate_tags"]:
        assert thr["model"][tag]["threshold"] == ev[tag]["per_seed"][seed_pos], tag
        assert thr["model"][tag]["rules_val_late_rate"] == ev[tag]["rules_val_late_rate"], tag
    rules_thr = read_json(d["rules_engine"] / "thresholds.json")
    assert thr["rules"]["thresholds"] == rules_thr["thresholds"]
    assert thr["headline_rate_tag"] == rate_tag(m2["cfgs"]["rules"]["alert_rate"])
    # A fresh re-verification (restore, replay, predict) and the file hashes pass.
    again = bundle.verify_bundle(sd)
    assert again["files_checked"] == len(bundle.BUNDLE_FILES)
    assert again["final_state_digest"] == s["verification"]["final_state_digest"]
