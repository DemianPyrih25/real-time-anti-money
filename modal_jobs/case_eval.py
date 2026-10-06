"""Case packs over a whole period: typology-matcher fitting and tuning (val) or their evaluation
plus the full-period parity check (test) (PLAN.md §6 M6, M6 spec §5).

    uv run modal run --detach -m modal_jobs.case_eval --period val     # make cases-val
    uv run modal run --detach -m modal_jobs.case_eval --period test    # make cases

The period's boundary snapshot (val: day 7, test: day 9) is restored and every row of the period
streams through the M5 scorer inproc on 1 core; the on_alert hook builds each alert's case pack
(aml.explain.case_eval). Every event's features, severities, scores and alerts must equal the
offline references and the final state digest the replay's: otherwise the job fails, after
writing its report. Labels are joined only after the stream (eval only).

Outputs in /data/reports (`make pull-reports`):
- val: typology_tree_val.json (the decision tree fitted on the val true-positive alerts, the
  primary matcher: copy it to configs/typology_tree.json before `make cases`),
  typology_match_val.json (the tuned decision-list thresholds, the baseline: copy them into
  configs/explain.yaml `typology:`; plus the tree's train and cross-validated accuracy next to
  the list's) and case_eval_val.json (parity, recall by typology, the 10 highest-scoring false
  positives with their narratives).
- test: case_eval.json (parity; the frozen tree's and the frozen decision list's accuracy on the
  primary and full views, with the majority baseline, per-typology recall and confusion) and
  cases/case_<row_id>.json (3 example case packs). It reads test labels (MLflow tag test_touch).
  The local entrypoint loads the frozen tree with configs/explain.yaml (configs/typology_tree.json
  next to it) and passes it to the remote; under typology_model: tree it refuses the single-leaf
  placeholder before any compute.

Needs the feature table to have passed the DuckDB oracle (`make features-verify`), and rules and
model outputs built from its current parts, like `make export`. The champion is read from the
serving bundle when it was exported under the current export key, else computed the same way.
"""

from __future__ import annotations

from pathlib import Path

import modal

from modal_jobs.common import (
    CONFIG_DIR,
    all_keys,
    cpu_job,
    data_paths,
    export_key,
    jsonable,
    load_all_configs,
    print_summary,
    require_data,
    require_features_verified,
    require_same_feature_table,
    require_stage_output,
    vol,
)

APP_NAME = "aml-case-eval"
CPU = 1.0  # the scorer is one thread: pure-Python engine + one-row LightGBM predicts
MEMORY_MIB = 6144
# Per-attempt timeout (COST_NOTES): a period is about 0.5-1 M events at well under 1 ms each,
# plus the chunked parity reads; 1 h leaves margin for a slower core.
TIMEOUT_S = 3600
PERIODS = ("val", "test")
MAX_METRICS = 2000
WORKER_KW = cpu_job(cpu=CPU, memory_mib=MEMORY_MIB, timeout=TIMEOUT_S)
app = modal.App(APP_NAME)


class ParityError(RuntimeError):
    """The stream differs from the offline references (the report is written first)."""


def stage_dirs(paths, keys: dict) -> tuple[Path, Path, Path]:
    """(features, rules_engine, lgbm_graph) directories of the current keys."""
    return (
        paths.features_dir(keys["features"]),
        paths.model_dir("rules_engine", keys["rules_engine"]),
        paths.model_dir("lgbm_graph", keys["lgbm_graph"]),
    )


def load_explain_cfg() -> dict:
    """configs/explain.yaml, validated, with the frozen tree (configs/typology_tree.json) under
    `typology_tree` (laptop)."""
    from aml.explain.casepack import CONFIG_FILE, load_config

    return load_config(CONFIG_DIR / CONFIG_FILE)


def case_eval_key(cfgs: dict[str, dict], explain_cfg: dict, period: str) -> str:
    """The champion (export key), the explain config (with the frozen tree), the period and the
    test views."""
    from aml.config import run_key

    if period not in PERIODS:
        raise ValueError(f"period must be one of {PERIODS}, got {period!r}")
    views = cfgs["data"].get("test_views")
    return run_key("case_eval", export_key(cfgs), explain_cfg, period, views)


@app.function(**WORKER_KW)
def case_eval(cfgs: dict, keys: dict, explain_cfg: dict, period: str) -> dict:
    import mlflow

    from aml import tracking
    from aml.config import run_key as hash_key
    from aml.explain.case_eval import TEST_TOUCH_MODELS, run_case_eval

    paths = data_paths(cfgs["data"])
    version = require_data(paths, keys["data"])
    features_dir, rules_dir, graph_dir = stage_dirs(paths, keys)
    require_stage_output(features_dir, "features", "features", data_version=version)
    require_stage_output(rules_dir, "rules_engine", "rules", data_version=version)
    require_stage_output(graph_dir, "lgbm_graph", "lgbm-graph", data_version=version)
    # The references are the feature table: it must have passed the DuckDB oracle, and the rules
    # and model must come from its current parts (as for the export).
    verified = require_features_verified(features_dir)
    require_same_feature_table(
        features_dir,
        {"rules_engine": (rules_dir, "rules"), "lgbm_graph": (graph_dir, "lgbm-graph")},
    )
    run_key = keys["case_eval"]
    try:
        # Opened before compute: a failed attempt shows up as a FAILED run (and in the cost window).
        with tracking.start_run_for_key(APP_NAME, run_key, run_name=run_key):
            if period == "test":
                tracking.tag_test_touch(list(TEST_TOUCH_MODELS))  # reads test labels
            # The frozen tree as its hash and provenance, not one param per node.
            tree = explain_cfg.get("typology_tree")
            explain = {k: v for k, v in explain_cfg.items() if k != "typology_tree"}
            explain["typology_tree"] = (
                None
                if tree is None
                else {
                    "hash": hash_key("typology_tree", tree["tree"]),
                    "trained_on": tree.get("trained_on"),
                }
            )
            tracking.log_params_flat({"keys": keys, "period": period, "explain": explain})
            tracking.log_params_flat({"features_verify": verified})
            mlflow.set_tags({"data_version": str(version), "period": period})
            summary = jsonable(
                run_case_eval(
                    paths,
                    period,
                    cfgs["data"],
                    explain_cfg,
                    keys,
                    features_dir=features_dir,
                    rules_dir=rules_dir,
                    graph_dir=graph_dir,
                    out_dir=paths.reports,
                    data_version=version,
                    log=lambda msg: print(msg, flush=True),  # progress: `modal app logs`
                )
            )
            tracking.log_metrics_flat(summary, max_items=MAX_METRICS)
            for report in summary["reports"]:
                mlflow.log_artifact(report)
            if not summary["parity_ok"]:
                raise ParityError(f"{period}: full-period parity failed: {summary['parity']}")
    finally:
        vol.commit()  # the reports (and the MLflow run) also when parity failed
    return {"run_key": run_key, **summary}


@app.local_entrypoint()
def main(period: str = "test") -> None:
    if period not in PERIODS:
        raise SystemExit(f"--period must be one of {PERIODS}, got {period!r}")
    from aml.explain.case_eval import TREE_REPORT, require_frozen_tree, typology_snippet
    from aml.explain.casepack import TREE_FILE

    cfgs = load_all_configs()
    keys = all_keys(cfgs)
    explain_cfg = load_explain_cfg()
    if period == "test":  # test measures the frozen tree: refuse the placeholder before compute
        try:
            require_frozen_tree(explain_cfg)
        except ValueError as e:
            raise SystemExit(str(e)) from None
    keys["case_eval"] = case_eval_key(cfgs, explain_cfg, period)
    summary = case_eval.remote(cfgs, keys, explain_cfg, period)
    print_summary(f"{APP_NAME} {period} {keys['case_eval']}", summary)
    parity = summary.get("parity") or {}
    bad = {k: v for k, v in (parity.get("mismatches") or {}).items() if v}
    digest_ok = parity.get("digest_ok")
    print(f"  parity: mismatches {bad or 0}, alerts {parity.get('alerts')}, digest_ok {digest_ok}")
    print(f"  accuracy: {summary.get('accuracy')}")
    for report in summary.get("reports") or []:
        print(f"  report: {report}")
    if period == "val" and summary.get("tuned"):
        print("== copy into configs/explain.yaml, then `make cases`:")
        print(typology_snippet(summary["tuned"]), end="")
    if period == "val" and summary.get("tree_report"):
        print(
            f"== freeze the tree: `make pull-reports`, then copy reports/{TREE_REPORT} to "
            f"configs/{TREE_FILE} before `make cases`"
        )
