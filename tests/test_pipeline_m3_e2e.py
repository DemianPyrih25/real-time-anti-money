"""End-to-end M3 pipeline on the synthetic fixture (M3 spec §14.2, last row), every GNN stage real
and on CPU, called the way the Modal jobs call them:

build_features (gnn_inputs) -> HPO (2 trials x 2 epochs) -> causal (1 seed, --final) -> look-ahead
(1 seed, test bounds end and d10) -> faithful (1 epoch, --final) -> evaluate with the `gnn`
argument assembled by modal_jobs.evaluate's own helper -> results.md with the GNN rows, the
pre-registered verdict, the look-ahead gap, the faithful section and the guard evidence; every
summary's as-of guard has 0 violations over > 0 sampled edges.

The rules are the real M1 SQL stage; the LightGBM-graph baseline is stand-in scores (its real
stage is covered by test_pipeline_m2_e2e.py). Run keys are the real modal_jobs.common keys of the
small config. Needs torch + PyG (Linux CI / the runner).
"""

from __future__ import annotations

import copy
import importlib
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import polars as pl
import pytest

pytestmark = pytest.mark.slow
torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from aml.eval.report import render_markdown, run_evaluate_stage  # noqa: E402
from aml.io import read_json, write_parquet_atomic  # noqa: E402
from aml.models.gnn import (  # noqa: E402
    BEST_PARAMS_FILE,
    COMPARISON_MODELS,
    HPO_KIND,
    LOOKAHEAD_D10_MODEL,
    SCORES_FILE,
    SUMMARY_FILE,
    check_gnn_cfg,
    read_jsonl,
    set_kind,
)
from aml.models.gnn import hpo as H  # noqa: E402
from aml.models.gnn import train as tr  # noqa: E402
from aml.rules.sql_baseline import run_rules_stage  # noqa: E402
from tests.conftest import load_yaml  # noqa: E402
from tests.fixtures.gnn_graphs import gnn_inputs  # noqa: E402

THREADS = 2
BOOT_B = 40
PROTOCOLS = ("causal", "lookahead", "faithful")
MAKE = {"causal": "gnn", "lookahead": "gnn-lookahead", "faithful": "gnn-faithful"}


def _jobs():
    """modal_jobs.common / evaluate / train_gnn, imported offline (no Modal config)."""
    os.environ.setdefault(
        "MODAL_CONFIG_PATH", str(Path(tempfile.gettempdir()) / "aml-tests-no-modal.toml")
    )
    return tuple(
        importlib.import_module(f"modal_jobs.{m}") for m in ("common", "evaluate", "train_gnn")
    )


def _gnn_cfg(base: dict) -> dict:
    """The small fixture config with one seed per protocol (the sets of this test)."""
    g = copy.deepcopy(base)
    g["protocols"]["causal"]["seeds"] = [0]
    g["protocols"]["lookahead"]["seeds"] = [0]
    return dict(check_gnn_cfg(g))


def _until_done(fn, what: str, calls: int = 4) -> dict:
    """Call a chunked stage until it is done (each call has an unlimited wall here)."""
    for _ in range(calls):
        out = fn()
        if out["status"] == "done":
            return out
        assert out["status"] == "partial", (what, out.get("status"), out.get("error"))
    raise AssertionError(f"{what} not done after {calls} calls")


def _standin_scores(paths, out_dir: Path) -> Path:
    """lgbm_graph stand-in: informative scores for every row, seeds 0 and 1."""
    tx = pl.read_parquet(paths.transactions).select("row_id", "split")
    lab = pl.read_parquet(paths.labels).select("row_id", "is_laundering")
    j = tx.join(lab, on="row_id", how="left")
    y = j["is_laundering"].fill_null(0).to_numpy()
    rng = np.random.default_rng(0)
    cols = {
        f"score_s{k}": pl.Series(np.clip(rng.random(len(y)) * 0.6 + 0.4 * y, 0.0, 1.0))
        for k in (0, 1)
    }
    write_parquet_atomic(j.select("row_id", "split").with_columns(**cols), out_dir / SCORES_FILE)
    return out_dir


@pytest.fixture(scope="module")
def m3(prepared, data_cfg, rules_cfg, tmp_path_factory) -> dict:
    common, ev_job, train_job = _jobs()
    mp = pytest.MonkeyPatch()
    mp.setattr(tr, "MLFLOW_ENABLED", False)  # no tracking store in tests
    dcfg = copy.deepcopy(data_cfg)
    dcfg["evaluation"]["bootstrap_replicates"] = BOOT_B
    build_cfgs = {"data": dcfg, "rules": rules_cfg, "features": load_yaml("features.yaml")}
    paths, fdir, small = gnn_inputs(prepared, tmp_path_factory.mktemp("m3"), build_cfgs)
    g = _gnn_cfg(small)
    cfgs = {
        **build_cfgs,
        "lgbm": load_yaml("lgbm.yaml"),
        "serving": load_yaml("serving.yaml"),
        "gnn": g,
    }
    keys = common.gnn_keys(cfgs)
    out: dict = {"paths": paths, "fdir": fdir, "cfg": g, "cfgs": cfgs, "keys": keys, "sets": {}}
    try:
        # HPO (the job's hpo_gpu body): 2 trials x 2 epochs, val_early only.
        hpo_dir = paths.gnn_set_dir(HPO_KIND, keys["gnn_hpo"])
        out["hpo"] = _until_done(
            lambda: H.run_hpo(
                paths,
                fdir,
                hpo_dir,
                g,
                data_cfg=dcfg,
                device="cpu",
                runtime=g["runtime"],
                budget_s=1e9,
            ),
            "hpo",
        )
        out["hpo_dir"] = hpo_dir
        best = read_json(hpo_dir / BEST_PARAMS_FILE)
        out["best"] = best
        # The sets (the job's train_gpu body), keyed exactly as train_gnn keys them.
        for p in PROTOCOLS:
            params = tr.effective_params(g, p, best if p in train_job.HPO_PROTOCOLS else None)
            plan = train_job.set_plan(cfgs, p, params, train_job.protocol_seeds(g, p))
            set_dir = paths.gnn_set_dir(plan["set_kind"], plan["set_key"])
            bounds = (
                tuple(g["protocols"]["lookahead"]["test_bounds"]) if p == "lookahead" else ("end",)
            )
            res = _until_done(
                lambda p=p, plan=plan, set_dir=set_dir, params=params, bounds=bounds: tr.run_set(
                    paths,
                    fdir,
                    set_dir,
                    g,
                    data_cfg=dcfg,
                    protocol=p,
                    seeds=plan["seeds"],
                    run_keys=plan["run_keys"],
                    params=params,
                    final=True,
                    device="cpu",
                    runtime=g["runtime"],
                    budget_s=1e9,
                    test_bounds=bounds,
                ),
                p,
            )
            out["sets"][p] = {**plan, "dir": set_dir, "result": res}
        # The rules (real M1 SQL stage) and the LightGBM-graph stand-in.
        rules_dir = paths.model_dir("rules", "rules-e2e")
        run_rules_stage(paths, rules_dir, rules_cfg, dcfg, threads=THREADS)
        graph_dir = _standin_scores(paths, paths.model_dir("lgbm_graph", "lgbm_graph-standin"))
        # evaluate --with-gnn: the job's spec and its own checks, then the library stage.
        sets = out["sets"]
        la_key = sets["lookahead"]["set_key"]
        spec = {
            "models": {
                "gnn_causal": {
                    "kind": "gnn_causal",
                    "key": sets["causal"]["set_key"],
                    "make": MAKE["causal"],
                },
                "gnn_lookahead": {
                    "kind": "gnn_lookahead",
                    "key": la_key,
                    "make": MAKE["lookahead"],
                },
                LOOKAHEAD_D10_MODEL: {
                    "kind": LOOKAHEAD_D10_MODEL,
                    "key": la_key,
                    "make": MAKE["lookahead"],
                },
            },
            "faithful": {
                "kind": "gnn_faithful",
                "key": sets["faithful"]["set_key"],
                "make": MAKE["faithful"],
            },
            "model_views": {LOOKAHEAD_D10_MODEL: ["primary"]},
            "report_cfg": g["report"],
        }
        marker = paths.parquet_dir / common.DATA_MARKER
        version = read_json(marker).get("data_version") if marker.exists() else None
        gnn_dirs, faithful_dir, gnn_arg = ev_job._gnn_inputs(paths, spec, version, fdir)
        model_dirs = {"lgbm_graph": graph_dir, **gnn_dirs}
        out.update(spec=spec, gnn_dirs=gnn_dirs, faithful_dir=faithful_dir, rules_dir=rules_dir)
        out["results"] = run_evaluate_stage(
            paths,
            model_dirs,
            rules_dir,
            paths.reports,
            dcfg,
            rules_cfg,
            threads=THREADS,
            gnn=gnn_arg,
        )
        out["eval_key"] = common.eval_key(cfgs, gnn_keys=ev_job.gnn_eval_keys(spec))
    finally:
        mp.undo()
    return out


def _guard_total(summary: dict) -> dict:
    from aml.eval.report import summary_guard

    total = summary_guard(summary)
    assert total is not None, summary.keys()
    return total


def test_hpo_ran_its_trials_and_picked_best_params(m3) -> None:
    s = read_json(m3["hpo_dir"] / SUMMARY_FILE)
    g = m3["cfg"]
    assert s["n_trials"] == g["hpo"]["n_trials"] == 2
    assert set(m3["best"]) == {"lr", "final_dropout", "w_pos"}
    records = read_jsonl(m3["paths"].gnn_optuna_log(m3["keys"]["gnn_hpo"]))
    assert len(records) == 2 and records[0]["params"] == H.trial0_params(g)
    assert s["violations"] == 0 and s["edges_checked"] > 0


@pytest.mark.parametrize("protocol", PROTOCOLS)
def test_every_set_is_final_scored_and_guarded(m3, protocol) -> None:
    st = m3["sets"][protocol]
    d = st["dir"]
    assert d == m3["paths"].gnn_set_dir(set_kind(protocol), st["set_key"])
    summary = read_json(d / SUMMARY_FILE)
    assert summary["final"] is True and summary["features_digest"] is not None
    from aml.models.gnn import report_hash

    assert summary["report_hash"] == report_hash(m3["cfg"])
    scores = pl.read_parquet(d / SCORES_FILE)
    assert set(scores["split"].unique()) == {"val_early", "val_late", "test"}
    assert [c for c in scores.columns if c.startswith("score_s")] == ["score_s0"]
    assert scores["score_s0"].is_finite().all()
    total = _guard_total(summary)
    assert total["violations"] == 0 and total["edges_checked"] > 0
    if protocol == "causal":
        assert total["target_hits"] == 0
    if protocol == "faithful":
        assert "sampled" in scores.columns and summary["epochs_run"] == 1
    if protocol == "lookahead":
        d10 = m3["paths"].gnn_set_dir(LOOKAHEAD_D10_MODEL, st["set_key"])
        d10_sum = read_json(d10 / SUMMARY_FILE)
        assert d10_sum["final"] is True and _guard_total(d10_sum)["violations"] == 0
        d10_scores = pl.read_parquet(d10 / SCORES_FILE)
        assert d10_scores.height == scores.height
        val = pl.col("split") != "test"
        assert d10_scores.filter(val).sort("row_id").equals(scores.filter(val).sort("row_id"))
        assert summary["dropped_target_copies"] > 0  # the look-ahead target copies are dropped
        assert set(summary["future_share"]) >= {"val_early"}
        # review LEAK-1: the d10 test pass sampled under the d10 bound, not the end one
        d10_future = d10_sum["guard"]["test_d10"]["future_edges"]
        assert d10_future < summary["guard"]["test"]["future_edges"]


def test_evaluation_with_gnn_models(m3) -> None:
    res = m3["results"]
    assert list(res["meta"]["models"]) == ["lgbm_graph", *COMPARISON_MODELS[:3]]
    assert res["meta"]["model_views"][LOOKAHEAD_D10_MODEL] == ["primary"]
    assert res["meta"]["models"]["gnn_causal"]["seeds"] == [0]
    md = (m3["paths"].reports / "results.md").read_text(encoding="utf-8")
    assert md == render_markdown(res)
    for needle in (
        "| gnn_causal (1 seed, no std) | (a) deployable threshold |",
        "| gnn_lookahead_d10 (1 seed, no std) |",
        "| gnn_causal - lgbm_graph |",
        "## Which model wins (pre-registered rule)",
        "## Look-ahead gap, step 1 (primary period, days 9-10)",
        "## Faithful Multi-GNN reproduction (not in the model comparison)",
        "## As-of guard evidence (GNN)",
        "- gnn_causal: 0 violations over ",
        "- gnn_faithful (snapshot guard): 0 violations over ",
        "One look-ahead seed: the CI covers test sampling only.",
    ):
        assert needle in md, needle
    gnn = res["gnn"]
    assert gnn["winner"]["pair"] == ["gnn_causal", "lgbm_graph"]
    assert gnn["winner"]["verdict"] in ("gnn", "lgbm", "tie", "mixed")
    assert gnn["faithful"]["verdict"] in ("reproduced", "not reproduced", "undefined")
    assert gnn["faithful"]["epochs_run"] == 1
    assert set(gnn["guard"]) == {*COMPARISON_MODELS[:3], "gnn_faithful"}
    for name, gd in gnn["guard"].items():
        assert gd["total"]["violations"] == 0 and gd["total"]["edges_checked"] > 0, name
    gap = gnn["lookahead_gap"]
    pair = res["bootstrap"]["model_diffs"]["gnn_lookahead - gnn_causal"]
    for key, row in gap["metrics"].items():
        assert row["gap_end"]["point"] == pair[key]["point"], key
    assert "test_d10" in gap["future_share"] and "test" in gap["future_share"]
    assert m3["eval_key"].startswith("eval-")
    json.dumps(res, allow_nan=False)
    # review EVAL-5: MLflow gets the compact GNN block right after the bootstrap, every
    # model's primary view, and gnn_lookahead_d10 on its primary view only; nothing is cut
    from aml.tracking import _numeric_items  # the order and count log_metrics_flat uses

    _common, ev, _train = _jobs()
    m = ev.metrics_for_mlflow(res)
    keys = [k for k, _ in _numeric_items(m, "")]
    assert len(keys) <= ev.MAX_EVAL_METRICS
    first = {k.split(".")[0]: i for i, k in reversed(list(enumerate(keys)))}
    assert list(m)[:2] == ["bootstrap", "gnn"] and first["gnn"] < first["views"]
    for name in res["meta"]["models"]:
        assert any(k.startswith(f"views.primary.models.{name}.") for k in keys), name
    assert not any(k.startswith(f"views.tail.models.{LOOKAHEAD_D10_MODEL}.") for k in keys)
    assert any(k.startswith("gnn.gap.") for k in keys) and any(
        k.startswith("gnn.winner.") for k in keys
    )


def test_chunked_driver_loop_resumes_to_the_same_results(m3, prepared, tmp_path, monkeypatch):
    """modal_jobs.train_gnn.drive_chunks (the CPU driver loop) over the real worker bodies with
    a budget that ends every call after one unit, and one worker call that dies (Modal retries
    used up: the driver ends with "error", a re-submission resumes). The causal set and the HPO
    trial log equal the module fixture's straight runs bit for bit, and no "partial" is taken
    for a lack of progress."""
    _common, _ev, train_job = _jobs()
    monkeypatch.setattr(tr, "MLFLOW_ENABLED", False)
    build_cfgs = {k: m3["cfgs"][k] for k in ("data", "rules", "features")}
    paths, fdir, _ = gnn_inputs(prepared, tmp_path, build_cfgs)
    assert fdir == m3["fdir"]  # the same feature parts (shared build)
    g, dcfg = m3["cfg"], m3["cfgs"]["data"]
    c = m3["sets"]["causal"]
    budget = {"attempt_wall_s": 1e-6, "max_set_wall_s": 1e9, "timeout_s": 1800}

    def drive(work, set_dir, die_at=None):
        calls = []

        def call(budget_s):
            calls.append(budget_s)
            if len(calls) == die_at:
                raise RuntimeError("retries exhausted (simulated)")
            return work(budget_s)

        out = train_job.drive_chunks(call, budget=budget, set_dir=set_dir, log=lambda m: None)
        return out, calls

    set_dir = paths.gnn_set_dir(c["set_kind"], c["set_key"])

    def train(budget_s):
        return tr.run_set(
            paths,
            fdir,
            set_dir,
            g,
            data_cfg=dcfg,
            protocol="causal",
            seeds=c["seeds"],
            run_keys=c["run_keys"],
            params=c["params"],
            final=True,
            device="cpu",
            runtime=g["runtime"],
            budget_s=budget_s,
        )

    out, calls = drive(train, set_dir, die_at=2)
    assert out["status"] == "error" and len(calls) == 2
    out, calls = drive(train, set_dir)
    assert out["status"] == "done" and out["set_complete"] and len(calls) >= 2
    assert all(ch["status"] == "partial" for ch in out["chunks"][:-1])
    want = pl.read_parquet(c["dir"] / SCORES_FILE)
    assert pl.read_parquet(set_dir / SCORES_FILE).equals(want)

    hpo_key = m3["keys"]["gnn_hpo"]
    hpo_dir = paths.gnn_set_dir(HPO_KIND, hpo_key)

    def search(budget_s):
        return H.run_hpo(
            paths,
            fdir,
            hpo_dir,
            g,
            data_cfg=dcfg,
            device="cpu",
            runtime=g["runtime"],
            budget_s=budget_s,
        )

    out, calls = drive(search, hpo_dir)
    assert out["status"] == "done" and len(calls) >= 4  # 2 trials x 2 epochs, one per call

    def trials(log):
        return [
            {k: v for k, v in r.items() if k not in ("seconds", "guard")} for r in read_jsonl(log)
        ]

    log, ref_log = paths.gnn_optuna_log(hpo_key), m3["paths"].gnn_optuna_log(hpo_key)
    assert trials(log) == trials(ref_log)
    # Guard totals count the edges actually sampled: the val_early eval cache is rebuilt in every
    # call, so a chunked search checks (and counts) val_early more often than a straight one.
    for a, b in zip(read_jsonl(log), read_jsonl(ref_log), strict=True):
        assert a["guard"]["train"] == b["guard"]["train"]
        assert all(v["violations"] == 0 for v in a["guard"].values())
    assert read_json(hpo_dir / BEST_PARAMS_FILE) == m3["best"]
