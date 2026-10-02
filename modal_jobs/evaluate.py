"""Evaluate rules and models on val_late + test; write reports/results.{md,json} and cost.md.

    uv run modal run -m modal_jobs.evaluate [--with-nofmt] [--skip-cost]

Rules: the engine-severity rules (/data/models/rules_engine/<rules_engine_key>/, whose flags equal
the M1 SQL rules'). Models: lgbm_tx (M1) and lgbm_graph (M2), plus lgbm_graph_nofmt with
`--with-nofmt` (a second test touch: run it only after the user approves, M2 spec §13).
results.md also gets validation-only M2 sections (gate, ablations, SHAP, engine, rule parity)
from the stage outputs that exist.

This touches the test set (MLflow tag `test_touch` lists every evaluated model). The cost report
is best-effort: if `modal billing report` fails, cost.md says so and the evaluation still succeeds.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import modal

from modal_jobs.common import (
    DATA_ROOT,
    all_keys,
    cpu_job,
    data_paths,
    eval_key,
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
MAX_EVAL_METRICS = 2000  # the results dict is large; the full JSON is logged as an artifact
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


def metrics_for_mlflow(results: dict) -> dict:
    """The results in headline-first order (bootstrap CIs, primary, tail and full views, then
    thresholds and meta), without per-seed lists and counts (mean and std stay; results.json
    has them).

    MAX_EVAL_METRICS keeps the first metrics in key order, so the headline is never cut.
    """
    views = results.get("views", {})
    ordered = {
        "bootstrap": results.get("bootstrap"),
        "views": {v: views[v] for v in ("primary", "tail", "full") if v in views},
        "thresholds": results.get("thresholds"),
        "meta": results.get("meta"),
    }
    return _drop_per_seed({k: v for k, v in ordered.items() if v is not None})


@app.function(**cpu_job(cpu=CPU, memory_mib=16384, timeout=3600))
def evaluate(data_cfg: dict, rules_cfg: dict, keys: dict, with_nofmt: bool = False) -> dict:
    import mlflow

    from aml import tracking
    from aml.eval.report import run_evaluate_stage

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
    extras = extra_paths(by_stage)
    missing = sorted(set(EXTRA_FILES) - set(extras))
    if missing:
        print(f"validation sections without input (left out of results.md): {missing}")

    out_dir = paths.reports
    # Opened (and tagged as touching test) before the evaluation runs, so a failed attempt is
    # recorded too.
    with tracking.start_run_for_key(APP_NAME, keys["eval"], run_name=keys["eval"]):
        tracking.tag_test_touch(sorted(model_dirs))
        tracking.log_params_flat(
            {
                "keys": keys,
                "evaluation": data_cfg.get("evaluation"),
                "with_nofmt": bool(with_nofmt),
            }
        )
        mlflow.set_tags({"data_version": str(version)})
        results = jsonable(
            run_evaluate_stage(
                paths, model_dirs, rules_dir, out_dir, data_cfg, rules_cfg, extras=extras or None
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


def fetch_billing_report(start_ms: int | None) -> tuple[list | dict | None, str | None]:
    """Run `modal billing report --json` locally. Never raises: returns (report, error)."""
    start = datetime.fromtimestamp(start_ms / 1000, tz=UTC) if start_ms else datetime.now(UTC)
    cmd = [
        sys.executable,
        "-m",
        "modal",
        "billing",
        "report",
        "--start",
        start.strftime("%Y-%m-%d"),
        "--resolution",
        "h",
        "--json",
    ]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=180,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"billing CLI failed to run: {type(e).__name__}"
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["no stderr"]
        return None, f"billing CLI exited {proc.returncode}: {tail[0][:300]}"
    try:
        return json.loads(proc.stdout), None
    except json.JSONDecodeError:
        return None, "billing CLI output was not JSON"


@app.local_entrypoint()
def main(with_nofmt: bool = False, skip_cost: bool = False, cost_only: bool = False) -> None:
    # --cost-only refreshes reports/cost.md without re-running the evaluation (no test touch).
    if not cost_only:
        cfgs = load_all_configs()
        keys = all_keys(cfgs)
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
