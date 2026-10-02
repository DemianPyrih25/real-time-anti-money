"""End-to-end M1 pipeline on the synthetic fixture: rules -> LightGBM -> evaluation.

`prepared` is the real prepare_data stage run on the synthetic CSV; the three library stages are
then called exactly as the Modal jobs call them. Two rule configs run: configs/rules.yaml as it
is (on this tiny fixture its 0.5% budget allows only 3 alerts on the tune split, so the rules
fire nothing and the headline is degenerate, which must still render) and a raised budget where
the rules do fire, so every headline number is defined.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from aml.config import rate_tag
from aml.eval.report import LITERATURE_F1, build_eval_frame, run_evaluate_stage
from aml.io import read_json, write_parquet_atomic
from aml.models.lgbm import run_lgbm_stage
from aml.paths import DataPaths
from aml.rules.sql_baseline import SCENARIOS, run_rules_stage

pytestmark = pytest.mark.slow

THREADS = 2
BOOT_B = 50
MODEL = "lgbm_tx"
RAISED = {"alert_rate": 0.02, "sensitivity_alert_rates": [0.01, 0.05]}


def _run_pipeline(paths: DataPaths, root: Path, data_cfg: dict, rules_cfg: dict, lgbm_cfg: dict):
    rules_dir = root / "models" / "rules"
    model_dir = root / "models" / MODEL
    out_dir = root / "reports"
    rules = run_rules_stage(paths, rules_dir, rules_cfg, data_cfg, threads=THREADS)
    lgbm = run_lgbm_stage(paths, model_dir, lgbm_cfg, rules_cfg, threads=THREADS)
    results = run_evaluate_stage(
        paths, {MODEL: model_dir}, rules_dir, out_dir, data_cfg, rules_cfg, threads=THREADS
    )
    return {
        "paths": paths,
        "rules_dir": rules_dir,
        "model_dir": model_dir,
        "out_dir": out_dir,
        "rules": rules,
        "lgbm": lgbm,
        "results": results,
        "data_cfg": data_cfg,
        "rules_cfg": rules_cfg,
        "lgbm_cfg": lgbm_cfg,
    }


def _cfgs(data_cfg: dict, rules_cfg: dict, raised: bool) -> tuple[dict, dict]:
    dcfg = copy.deepcopy(data_cfg)
    dcfg["evaluation"]["bootstrap_replicates"] = BOOT_B
    rcfg = copy.deepcopy(rules_cfg)
    if raised:
        rcfg.update(copy.deepcopy(RAISED))
    return dcfg, rcfg


@pytest.fixture(scope="module", params=["default", "raised"])
def run(request, prepared, data_cfg, rules_cfg, lgbm_cfg, tmp_path_factory) -> dict:
    dcfg, rcfg = _cfgs(data_cfg, rules_cfg, raised=request.param == "raised")
    out = _run_pipeline(
        prepared, tmp_path_factory.mktemp(f"e2e_{request.param}"), dcfg, rcfg, lgbm_cfg
    )
    out["name"] = request.param
    return out


def _sections(md: str) -> dict[str, str]:
    """'## ' heading -> section text (up to the next '## ' heading)."""
    out: dict[str, str] = {}
    for block in md.split("\n## ")[1:]:
        title, _, body = block.partition("\n")
        out[title.strip()] = body
    return out


def _finite(x) -> bool:
    return isinstance(x, int | float) and not isinstance(x, bool) and math.isfinite(x)


# --------------------------------------------------------------------------- files


def test_every_stage_writes_its_outputs(run) -> None:
    for f in ("severities.parquet", "flags.parquet", "thresholds.json", "summary.json"):
        assert (run["rules_dir"] / f).is_file(), f
    for f in ("scores.parquet", "best_params.json", "trials.json", "summary.json"):
        assert (run["model_dir"] / f).is_file(), f
    for seed in run["lgbm_cfg"]["seeds"]:
        assert (run["model_dir"] / f"booster_s{seed}.txt").is_file()
    res = run["results"]
    assert json.loads((run["out_dir"] / "results.json").read_text(encoding="utf-8")) == res
    json.dumps(res, allow_nan=False)  # strict JSON: nan/inf were written as null
    assert (run["out_dir"] / "results.md").read_text(encoding="utf-8").startswith("# Evaluation")


def test_results_hold_the_rules_and_the_model(run) -> None:
    res = run["results"]
    seeds = sorted(int(s) for s in run["lgbm_cfg"]["seeds"])
    assert res["meta"]["models"] == {MODEL: {"n_seeds": len(seeds), "seeds": seeds}}
    rcfg = run["rules_cfg"]
    head = rate_tag(rcfg["alert_rate"])
    assert res["meta"]["headline_tag"] == head
    tags = {head, *(rate_tag(r) for r in rcfg["sensitivity_alert_rates"])}
    for view in ("primary", "tail", "full"):
        vr = res["views"][view]
        assert set(vr["rules"]["points"]) == tags
        assert vr["rules"]["points"][head]["pr_auc"] == "n/a"  # rules are binary
        assert set(vr["rules"]["scenarios"]) == set(SCENARIOS)
        assert set(vr["models"]) == {MODEL}
        assert len(vr["models"][MODEL]["literature"]["pr_auc"]["per_seed"]) == len(seeds)
    md = (run["out_dir"] / "results.md").read_text(encoding="utf-8")
    headline = _sections(md)["Headline: primary period"]
    # The rules row says how many scenarios tuning actually switched on at the headline rate.
    thr = read_json(run["rules_dir"] / "thresholds.json")["thresholds"][head]
    k = sum(v is not None for v in thr.values())
    rules_row = f"Rules (SQL, {k} of {len(SCENARIOS)} scenarios active)"
    assert f"| {rules_row} | own operating point |" in headline
    assert res["meta"]["rules"]["active"][head] == [s for s in SCENARIOS if thr[s] is not None]
    for label in ("(a) deployable threshold", "(b) top-K", "(c) rules ∪ model (a)"):
        assert f"| {MODEL} ({len(seeds)} seeds) | {label}" in headline


def test_primary_period_is_the_headline(run) -> None:
    res, dcfg = run["results"], run["data_cfg"]
    lo, hi = dcfg["test_views"]["primary"]
    assert res["meta"]["views"]["primary"]["days"] == [lo, hi]
    assert res["bootstrap"]["view"] == "primary"
    assert res["bootstrap"]["B"] == BOOT_B
    md = (run["out_dir"] / "results.md").read_text(encoding="utf-8")
    titles = list(_sections(md))
    assert titles[0] == "Headline: primary period"
    assert titles.index("Test view: tail") > 0 and titles.index("Test view: full") > 0
    assert f"days {lo}-{hi}:" in _sections(md)["Headline: primary period"]
    # The view sizes come from the prepared data, and primary + tail = full.
    ev = build_eval_frame(run["paths"], dcfg)
    test = ev.filter(pl.col("split") == "test")
    views = res["meta"]["views"]
    assert views["primary"]["rows"] == test.filter(pl.col("day").is_between(lo, hi)).height
    assert views["primary"]["rows"] + views["tail"]["rows"] == views["full"]["rows"] == test.height
    assert views["full"]["positives"] == int(test["y"].sum())


def test_literature_table_is_separate(run) -> None:
    md = (run["out_dir"] / "results.md").read_text(encoding="utf-8")
    sections = _sections(md)
    lit = [t for t in sections if t.startswith("Literature reference")]
    assert len(lit) == 1
    for method, f1 in LITERATURE_F1:
        assert f"| {method} | {f1:.1f} |" in sections[lit[0]]
    # Published numbers appear nowhere else (not in our result tables).
    others = "\n".join(body for t, body in sections.items() if t != lit[0])
    for method, _ in LITERATURE_F1:
        assert f"| {method} |" not in others
    # results.json holds only our own numbers ("literature" there = literature-comparable
    # metrics of our models); the published anchors live in the markdown table alone.
    dumped = json.dumps(run["results"])
    assert not any(method in dumped for method, _ in LITERATURE_F1 if "+" in method)


def test_literature_comparable_metrics_cover_the_published_period(run) -> None:
    """Published HI-Small numbers are on the whole test period (our full view): results.md must
    show our full-view F1 @ 0.5 next to the primary headline, and say which to compare."""
    res, dcfg = run["results"], run["data_cfg"]
    md = (run["out_dir"] / "results.md").read_text(encoding="utf-8")
    lit = _sections(md)["Literature-comparable metrics"]
    flo, fhi = dcfg["test_views"]["full"]
    assert f"**full** (days {flo}-{fhi}: the test period the published numbers use" in lit
    full_rows = lit.split("**full**")[1]
    assert "F1 @ 0.5 (argmax)" in full_rows and "Alerts @ 0.5" in full_rows
    full_lit = res["views"]["full"]["models"][MODEL]["literature"]
    f1 = np.mean(full_lit["f1_argmax"]["per_seed"])
    assert f"| {MODEL} (" in full_rows and f"{100 * f1:.1f}" in full_rows
    assert len(full_lit["alerts_argmax"]["per_seed"]) == len(run["lgbm_cfg"]["seeds"])
    ref = _sections(md)[next(t for t in _sections(md) if t.startswith("Literature reference"))]
    assert f"our full view (days {flo}-{fhi}" in ref


# --------------------------------------------------------------------------- numbers


def test_numbers_are_finite_where_defined(run) -> None:
    res = run["results"]
    head = res["meta"]["headline_tag"]
    for view in ("primary", "tail", "full"):
        vm = res["meta"]["views"][view]
        assert vm["positives"] > 0 and _finite(vm["prevalence"])
        r = res["views"][view]["rules"]["points"][head]
        assert r["alerts"] >= 0 and _finite(r["recall"]) and 0 <= r["recall"] <= 1
        lit = res["views"][view]["models"][MODEL]["literature"]
        for k in ("pr_auc", "roc_auc", "f1_thr", "recall_argmax"):
            assert all(_finite(v) and 0 <= v <= 1 for v in lit[k]["per_seed"]), (view, k)
    boot = res["bootstrap"]
    for key in ("literature.pr_auc", "literature.f1_thr", "a.recall", "b.recall"):
        lo, hi = boot["models"][MODEL][key]["lo"], boot["models"][MODEL][key]["hi"]
        assert _finite(lo) and _finite(hi) and 0 <= lo <= hi <= 1, key
    for d in boot["diffs"][MODEL].values():
        assert _finite(d["lo"]) and _finite(d["hi"]) and _finite(d["point"])
        assert d["lo"] <= d["hi"]


def test_headline_is_defined_when_the_rules_fire(run) -> None:
    res = run["results"]
    head = res["meta"]["headline_tag"]
    pv = res["views"]["primary"]
    rules = pv["rules"]["points"][head]
    model = pv["models"][MODEL]
    if run["name"] == "default":
        # 0.5% of the fixture's tune split is 3 alerts: nothing fires, (a) and (b) flag nothing.
        assert rules["alerts"] == 0
        assert model["a"][head]["alerts"]["mean"] == 0 == model["b"][head]["alerts"]["mean"]
        return
    assert rules["alerts"] > 0
    assert model["b"][head]["alerts"]["per_seed"] == [rules["alerts"]] * len(
        run["lgbm_cfg"]["seeds"]
    )
    a = model["a"][head]
    assert all(v > 0 for v in a["alerts"]["per_seed"])
    for k in ("precision", "recall", "f1", "threshold"):
        assert all(_finite(v) for v in a[k]["per_seed"]), k
    for k in ("precision", "recall", "f1"):
        ci = res["bootstrap"]["models"][MODEL][f"a.{k}"]
        assert _finite(ci["lo"]) and _finite(ci["hi"]), k
    md = (run["out_dir"] / "results.md").read_text(encoding="utf-8")
    assert "± " in _sections(md)["Headline: primary period"]  # 2 seeds -> mean ± std


def test_thresholds_respect_the_rules_val_late_volume(run) -> None:
    th = run["results"]["thresholds"][MODEL]["rate"]
    for tag, t in th.items():
        rules_rate = t["rules_val_late_rate"]
        assert all(m <= rules_rate + 1e-12 for m in t["model_val_late_rate"]), tag


def test_stages_agree_on_rows_and_flags(run) -> None:
    """Rule alerts in the report equal the flags the rules stage wrote, per view."""
    res, dcfg = run["results"], run["data_cfg"]
    head = res["meta"]["headline_tag"]
    flags = pl.read_parquet(run["rules_dir"] / "flags.parquet")
    scores = pl.read_parquet(run["model_dir"] / "scores.parquet")
    tx = pl.read_parquet(run["paths"].transactions, columns=["row_id", "split"])
    assert flags.height == tx.height
    assert flags["row_id"].to_list() == tx["row_id"].to_list()  # both in rank order
    scored = tx.filter(pl.col("split").is_in(["val_early", "val_late", "test"]))
    assert scores["row_id"].to_list() == scored["row_id"].to_list()
    for view in ("primary", "tail", "full"):
        lo, hi = dcfg["test_views"][view]
        sub = flags.filter((pl.col("split") == "test") & pl.col("day").is_between(lo, hi))
        n = int(sub[f"rules_any_{head}"].sum())
        assert res["views"][view]["rules"]["points"][head]["alerts"] == n, view
    thr = read_json(run["rules_dir"] / "thresholds.json")
    assert thr["headline_rate_tag"] == head
    assert thr["tune_split"] == run["rules_cfg"]["tune_split"]


def test_shared_constants_agree() -> None:
    """Modules that keep their own copy of a shared constant must not drift apart."""
    from aml.data import ingest, patterns, split
    from aml.eval import report, typology
    from aml.features import tx_features
    from aml.models import lgbm

    assert typology.TYPOLOGIES == patterns.LABEL_TYPOLOGIES
    assert report.VIEWS == split.VIEWS
    assert {report.VAL_SPLIT, report.TEST_SPLIT} <= set(split.SPLITS)
    assert set(lgbm.SCORE_SPLITS) == set(split.SPLITS) - {"train"}
    assert tx_features.MINUTES_PER_DAY == ingest.MINUTES_PER_DAY == 1440


def test_mlflow_metrics_keep_the_headline(run) -> None:
    """modal_jobs.evaluate caps the logged metrics; the bootstrap CIs and the primary view must
    come first and fit under the cap whatever the number of seeds."""
    import importlib
    import os
    import tempfile

    from aml.tracking import _numeric_items  # the order and count log_metrics_flat uses

    os.environ.setdefault("MODAL_CONFIG_PATH", str(Path(tempfile.gettempdir()) / "no-modal.toml"))
    ev = importlib.import_module("modal_jobs.evaluate")
    keys = [k for k, _ in _numeric_items(ev.metrics_for_mlflow(run["results"]), "")]
    assert len(keys) <= ev.MAX_EVAL_METRICS
    assert not any(".per_seed" in k or ".n_defined" in k for k in keys)
    first_other = next(i for i, k in enumerate(keys) if not k.startswith("bootstrap."))
    assert keys[first_other].startswith("views.primary.")
    assert any(k.startswith("bootstrap.diffs.") for k in keys[:first_other])


# --------------------------------------------------------------------------- leakage


def test_test_labels_never_drive_any_choice(
    prepared, data_cfg, rules_cfg, lgbm_cfg, tmp_path_factory
) -> None:
    """Flip every test label: rule thresholds, flags, model scores and the eval thresholds
    (all chosen on train / val_early / val_late) must not move; only test metrics may."""
    dcfg, rcfg = _cfgs(data_cfg, rules_cfg, raised=True)
    base = _run_pipeline(prepared, tmp_path_factory.mktemp("e2e_base"), dcfg, rcfg, lgbm_cfg)

    root = tmp_path_factory.mktemp("e2e_flipped")
    flipped = DataPaths(root / "volume")
    flipped.parquet_dir.mkdir(parents=True)
    for name in ("transactions", "accounts"):
        shutil.copy(getattr(prepared, name), getattr(flipped, name))
    tx = pl.read_parquet(prepared.transactions, columns=["row_id", "split"])
    labels = pl.read_parquet(prepared.labels)
    is_test = labels.join(tx, on="row_id", how="left", maintain_order="left")["split"] == "test"
    labels = labels.with_columns(
        pl.when(pl.Series(is_test))
        .then(1 - pl.col("is_laundering"))
        .otherwise(pl.col("is_laundering"))
        .cast(pl.Int8)
        .alias("is_laundering")
    )
    write_parquet_atomic(labels, flipped.labels)
    other = _run_pipeline(flipped, root, dcfg, rcfg, lgbm_cfg)

    assert read_json(other["rules_dir"] / "thresholds.json") == read_json(
        base["rules_dir"] / "thresholds.json"
    )
    for d, f in (("rules_dir", "flags.parquet"), ("model_dir", "scores.parquet")):
        assert pl.read_parquet(other[d] / f).equals(pl.read_parquet(base[d] / f)), f
    assert other["results"]["thresholds"] == base["results"]["thresholds"]
    # ...while the test metrics do change, so the flip reached the evaluation.
    pos = [r["results"]["meta"]["views"]["full"]["positives"] for r in (base, other)]
    assert pos[0] != pos[1]
    ap = [
        np.mean(
            r["results"]["views"]["primary"]["models"][MODEL]["literature"]["pr_auc"]["per_seed"]
        )
        for r in (base, other)
    ]
    assert ap[0] != ap[1]
