"""Evaluate rules and models on val_late + test; write reports/results.{md,json} and cost.md.

    uv run modal run -m modal_jobs.evaluate [--with-nofmt] [--with-gnn] [--skip-cost]

Rules: the engine-severity rules (/data/models/rules_engine/<rules_engine_key>/, whose flags equal
the M1 SQL rules'). Models: lgbm_tx (M1) and lgbm_graph (M2), plus lgbm_graph_nofmt with
`--with-nofmt` (a second test touch: run it only after the user approves, M2 spec §13).
results.md also gets validation-only M2 sections (gate, ablations, SHAP, engine, rule parity)
from the stage outputs that exist.

`--with-gnn` (M3, `make eval-gnn`) adds the finished --final GNN sets in report order:
gnn_causal (required), gnn_lookahead + gnn_lookahead_d10 (both or neither; d10 rendered on the
primary period only), gnn_pna (if finished), and gnn_faithful in its own section (never in the
comparison). It adds the look-ahead gap, the faithful reproduction, the pre-registered winner
verdict and the guard evidence, and refuses if gnn.yaml's `report:` hash differs from the one
the --final sets recorded. Without it the stage is exactly M2's.

This touches the test set (MLflow tag `test_touch` lists every evaluated model). The cost report
is best-effort: if `modal billing report` fails, cost.md says so and the evaluation still succeeds.
"""

from __future__ import annotations

# subprocess / datetime / UTC stay importable from here (tests patch ev.subprocess.run, which is
# the module common's billing helpers call).
import subprocess  # noqa: F401
from datetime import UTC, datetime  # noqa: F401
from pathlib import Path

import modal

from modal_jobs.common import (
    DATA_ROOT,
    all_keys,
    cpu_job,
    data_paths,
    eval_key,
    fetch_billing_report,
    jsonable,
    load_all_configs,
    print_summary,
    require_data,
    require_same_feature_table,
    require_stage_output,
    vol,
)

APP_NAME = "aml-evaluate"
CPU = 4.0
MAX_EVAL_METRICS = 8000  # the results dict is large; the full JSON is logged as an artifact
app = modal.App(APP_NAME)

# model name -> (model_dir kind, run-key name, make target that builds it)
MODELS = {
    "lgbm_tx": ("lgbm_tx", "lgbm_tx", "lgbm"),
    "lgbm_graph": ("lgbm_graph", "lgbm_graph", "lgbm-graph"),
    "lgbm_graph_nofmt": ("lgbm_graph_nofmt", "lgbm_graph", "lgbm-graph"),  # + --nofmt-final
}
# Validation-only inputs of results.md: extras kind -> (stage, file name).
EXTRA_FILES = {
    "gate": ("lgbm_graph", "gate.json"),
    "ablation": ("lgbm_graph", "ablation.json"),
    "shap": ("lgbm_graph", "shap_global.json"),
    "engine": ("features", "summary.json"),
    "parity": ("rules_engine", "parity.json"),
}


def stage_dirs(paths, keys: dict, with_nofmt: bool) -> tuple[Path, dict[str, Path], dict]:
    """(rules dir, model dirs in report order, {stage: dir} for the extras)."""
    rules_dir = paths.model_dir("rules_engine", keys["rules_engine"])
    names = ["lgbm_tx", "lgbm_graph"] + (["lgbm_graph_nofmt"] if with_nofmt else [])
    model_dirs = {n: paths.model_dir(MODELS[n][0], keys[MODELS[n][1]]) for n in names}
    by_stage = {
        "lgbm_graph": model_dirs["lgbm_graph"],
        "features": paths.features_dir(keys["features"]),
        "rules_engine": rules_dir,
    }
    return rules_dir, model_dirs, by_stage


def feature_table_stages(
    rules_dir: Path, model_dirs: dict[str, Path]
) -> dict[str, tuple[Path, str]]:
    """The evaluated stages that read the feature parts -> (directory, make target)."""
    out = {"rules_engine": (rules_dir, "rules")}
    for name in ("lgbm_graph", "lgbm_graph_nofmt"):
        if name in model_dirs:
            out[name] = (model_dirs[name], MODELS[name][2])
    return out


def extra_paths(by_stage: dict[str, Path]) -> dict[str, Path]:
    """The validation-only inputs that exist (a missing one is left out of results.md)."""
    out = {}
    for kind, (stage, name) in EXTRA_FILES.items():
        p = Path(by_stage[stage]) / name
        if p.exists():
            out[kind] = p
    return out


PER_SEED_DETAIL = ("per_seed", "n_defined")  # in results.json; not worth an MLflow metric each


def _drop_per_seed(obj):
    if isinstance(obj, dict):
        return {k: _drop_per_seed(v) for k, v in obj.items() if k not in PER_SEED_DETAIL}
    return obj


def _iv(d: dict | None) -> dict | None:
    if not isinstance(d, dict):
        return None
    return {k: d.get(k) for k in ("point", "lo", "hi")}


def gnn_metrics(g: dict | None) -> dict | None:
    """A compact M3 block for MLflow: the winner metrics' CIs, the look-ahead gap rows, the
    faithful F1 and its miss (results.json holds the whole `gnn` block)."""
    if not g:
        return None
    out: dict = {}
    w = g.get("winner") or {}
    if w.get("metrics"):
        out["winner"] = {k: _iv(m) for k, m in w["metrics"].items()}
    gap = g.get("lookahead_gap") or {}
    if gap.get("metrics"):
        out["gap"] = {
            k: {
                "gap_end": _iv(r.get("gap_end")),
                "gap_d10": _iv(r.get("gap_d10")),
                "tail": _iv(r.get("tail")),
            }
            for k, r in gap["metrics"].items()
        }
    f = g.get("faithful") or {}
    if f:
        out["faithful"] = {"f1_sampled_pct": f.get("f1_sampled_pct"), "miss_pp": f.get("miss_pp")}
    return out or None


def metrics_for_mlflow(results: dict) -> dict:
    """The results in headline-first order (bootstrap CIs, the compact GNN block, primary, tail
    and full views, then thresholds and meta), without per-seed lists and counts (mean and std
    stay; results.json has them). Models rendered on the primary period only
    (meta.model_views, e.g. gnn_lookahead_d10) keep only their primary view here.

    MAX_EVAL_METRICS keeps the first metrics in key order: about 6,400 with every M3 model,
    so nothing is cut; the headline and the GNN block come first either way.
    """
    views = dict(results.get("views", {}))
    model_views = (results.get("meta") or {}).get("model_views") or {}
    narrow = [m for m, v in model_views.items() if list(v) == ["primary"]]
    if narrow:
        for name in ("tail", "full"):
            if name in views and isinstance(views[name].get("models"), dict):
                models = {k: v for k, v in views[name]["models"].items() if k not in narrow}
                views[name] = {**views[name], "models": models}
    ordered = {
        "bootstrap": results.get("bootstrap"),
        "gnn": gnn_metrics(results.get("gnn")),
        "views": {v: views[v] for v in ("primary", "tail", "full") if v in views},
        "thresholds": results.get("thresholds"),
        "meta": results.get("meta"),
    }
    return _drop_per_seed({k: v for k, v in ordered.items() if v is not None})


def gnn_stage_dirs(paths, gnn: dict) -> tuple[dict[str, Path], Path | None]:
    """(GNN comparison model -> set dir in report order, the faithful set dir or None) from
    the local entrypoint's train_gnn.gnn_eval_models(...) spec."""
    from aml.models.gnn import COMPARISON_MODELS, FAITHFUL_MODEL

    models = gnn["models"]
    if FAITHFUL_MODEL in models or "gnn_causal" not in models:
        raise ValueError(f"GNN models must include gnn_causal and never {FAITHFUL_MODEL}")
    dirs = {
        n: paths.gnn_set_dir(models[n]["kind"], models[n]["key"])
        for n in COMPARISON_MODELS
        if n in models
    }
    fa = gnn.get("faithful")
    return dirs, (paths.gnn_set_dir(fa["kind"], fa["key"]) if fa else None)


def _gnn_inputs(paths, gnn: dict, version: str | None, features_dir: Path):
    """Check the GNN sets (finished, this data, these feature parts) and build run_evaluate_stage's
    `gnn` argument. Returns (comparison dirs, faithful dir, gnn argument)."""
    from aml.io import read_json
    from aml.models.gnn import FAITHFUL_MODEL, SCORES_FILE, SUMMARY_FILE

    dirs, faithful_dir = gnn_stage_dirs(paths, gnn)
    makes = {n: gnn["models"][n]["make"] for n in dirs}
    if faithful_dir is not None:
        makes[FAITHFUL_MODEL] = gnn["faithful"]["make"]
    every = {**dirs, **({FAITHFUL_MODEL: faithful_dir} if faithful_dir is not None else {})}
    for name, d in every.items():
        require_stage_output(d, name, makes[name], data_version=version)
    # Every GNN set read the feature parts (its edge attributes): same parts as on the Volume.
    require_same_feature_table(features_dir, {n: (d, makes[n]) for n, d in every.items()})
    summaries = {n: read_json(d / SUMMARY_FILE) for n, d in dirs.items()}
    faithful = None
    if faithful_dir is not None:
        faithful = {
            "scores": faithful_dir / SCORES_FILE,
            "summary": read_json(faithful_dir / SUMMARY_FILE),
        }
    arg = {
        "model_views": gnn.get("model_views") or {},
        "summaries": summaries,
        "faithful": faithful,
        "report_cfg": gnn["report_cfg"],
    }
    return dirs, faithful_dir, arg


@app.function(**cpu_job(cpu=CPU, memory_mib=16384, timeout=3600))
def evaluate(
    data_cfg: dict, rules_cfg: dict, keys: dict, with_nofmt: bool = False, gnn: dict | None = None
) -> dict:
    import mlflow

    from aml import tracking
    from aml.eval.report import run_evaluate_stage
    from aml.models.gnn import FAITHFUL_MODEL

    paths = data_paths(data_cfg)
    version = require_data(paths, keys["data"])
    rules_dir, model_dirs, by_stage = stage_dirs(paths, keys, with_nofmt)
    # Outputs built from other prepared data (a prepare fix or a changed download under the same
    # config) would be joined by row_id onto the new eval frame without any error.
    require_stage_output(rules_dir, "rules_engine", "rules", data_version=version)
    for name, d in model_dirs.items():
        require_stage_output(d, name, MODELS[name][2], data_version=version)
    # The rules and graph models must come from the feature parts on the Volume now (a
    # re-replay under the same features key would otherwise go unnoticed).
    require_stage_output(by_stage["features"], "features", "features", data_version=version)
    require_same_feature_table(by_stage["features"], feature_table_stages(rules_dir, model_dirs))
    gnn_arg, touched = None, sorted(model_dirs)
    if gnn:
        gnn_dirs, faithful_dir, gnn_arg = _gnn_inputs(paths, gnn, version, by_stage["features"])
        model_dirs = {**model_dirs, **gnn_dirs}  # report order: lgbm_*, then the GNN models
        touched = sorted([*model_dirs, *([FAITHFUL_MODEL] if faithful_dir is not None else [])])
    extras = extra_paths(by_stage)
    missing = sorted(set(EXTRA_FILES) - set(extras))
    if missing:
        print(f"validation sections without input (left out of results.md): {missing}")

    out_dir = paths.reports
    # Opened (and tagged as touching test) before the evaluation runs, so a failed attempt is
    # recorded too.
    with tracking.start_run_for_key(APP_NAME, keys["eval"], run_name=keys["eval"]):
        tracking.tag_test_touch(touched)
        params = {
            "keys": keys,
            "evaluation": data_cfg.get("evaluation"),
            "with_nofmt": bool(with_nofmt),
        }
        if gnn:
            params["gnn"] = {n: m["key"] for n, m in gnn["models"].items()}
            params["gnn_faithful"] = (gnn.get("faithful") or {}).get("key")
        tracking.log_params_flat(params)
        mlflow.set_tags({"data_version": str(version)})
        results = jsonable(
            run_evaluate_stage(
                paths,
                model_dirs,
                rules_dir,
                out_dir,
                data_cfg,
                rules_cfg,
                extras=extras or None,
                gnn=gnn_arg,
            )
        )
        logged = tracking.log_metrics_flat(metrics_for_mlflow(results), max_items=MAX_EVAL_METRICS)
        for name in ("results.md", "results.json"):
            if (out_dir / name).exists():
                mlflow.log_artifact(str(out_dir / name))
    vol.commit()
    return {
        "run_key": keys["eval"],
        "models": list(model_dirs),
        "rules_dir": str(rules_dir),
        "extras": sorted(extras),
        "reports": [str(out_dir / "results.md"), str(out_dir / "results.json")],
        "n_metrics_logged": len(logged),
    }


@app.function(**cpu_job(cpu=1.0, memory_mib=2048, timeout=600))
def earliest_run_start() -> int | None:
    """Earliest start (ms since epoch) over this project's MLflow runs: the cost window start."""
    from aml.tracking import earliest_run_start_ms

    return earliest_run_start_ms("aml-")


@app.function(**cpu_job(cpu=1.0, memory_mib=2048, timeout=600))
def write_cost_report(report: list | dict | None, error: str | None) -> str:
    from aml.io import write_text_atomic
    from aml.paths import DataPaths
    from aml.tracking import render_cost_report

    path = DataPaths(Path(DATA_ROOT)).reports / "cost.md"
    write_text_atomic(render_cost_report(report, error), path)
    vol.commit()
    return str(path)


def gnn_eval_keys(gnn: dict) -> dict[str, str]:
    """eval_key's gnn_keys: every evaluated GNN model (faithful included) -> its set key."""
    from aml.models.gnn import FAITHFUL_MODEL

    out = {n: m["key"] for n, m in gnn["models"].items()}
    if gnn.get("faithful"):
        out[FAITHFUL_MODEL] = gnn["faithful"]["key"]
    return out


@app.local_entrypoint()
def main(
    with_nofmt: bool = False,
    skip_cost: bool = False,
    cost_only: bool = False,
    with_gnn: bool = False,
) -> None:
    # --cost-only refreshes reports/cost.md without re-running the evaluation (no test touch).
    if not cost_only:
        cfgs = load_all_configs()
        keys = all_keys(cfgs)
        if with_gnn:
            from aml.models.gnn import check_gnn_cfg
            from modal_jobs.train_gnn import gnn_eval_models

            check_gnn_cfg(cfgs["gnn"])
            gnn = gnn_eval_models(cfgs)
            keys["eval"] = eval_key(cfgs, with_nofmt=with_nofmt, gnn_keys=gnn_eval_keys(gnn))
            print(f"== GNN models: {list(gnn['models'])}; faithful: {gnn['faithful'] is not None}")
            summary = evaluate.remote(cfgs["data"], cfgs["rules"], keys, with_nofmt, gnn)
        else:
            keys["eval"] = eval_key(cfgs, with_nofmt=with_nofmt)
            summary = evaluate.remote(cfgs["data"], cfgs["rules"], keys, with_nofmt)
        print_summary(f"{APP_NAME} {keys['eval']}", summary, keys=["models", "extras", "reports"])
    if skip_cost:
        return
    # Cost logging is best-effort; it must never fail the evaluation.
    try:
        start_ms = earliest_run_start.remote()
        report, error = fetch_billing_report(start_ms)
        path = write_cost_report.remote(report, error)
        print(f"== cost report: {path}" + (f" (unavailable: {error})" if error else ""))
    except Exception as e:
        print(f"== cost report skipped: {type(e).__name__}: {e}")
