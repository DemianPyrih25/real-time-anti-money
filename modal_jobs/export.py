"""Serving bundle: LightGBM-graph seed 0, feature spec, thresholds, calibration, the test-boundary
snapshot, the replay slice and its offline reference outputs (PLAN §6 M2, M2 spec §8.6).

    uv run modal run -m modal_jobs.export

Output: /data/models/serving/ (pulled by `make pull`); metadata.json is written last, after the
slice replayed from the snapshot reproduces the reference features and scores bit for bit.
Needs a feature table that passed the DuckDB oracle (`make features-verify`) and rules and
model outputs built from its current parts.
"""

from __future__ import annotations

import modal

from modal_jobs.common import (
    all_keys,
    cpu_job,
    data_paths,
    jsonable,
    load_all_configs,
    print_summary,
    require_data,
    require_features_verified,
    require_same_feature_table,
    require_stage_output,
    vol,
)

APP_NAME = "aml-export"
CPU = 4.0
app = modal.App(APP_NAME)


@app.function(**cpu_job(cpu=CPU, memory_mib=16384, timeout=3600))
def export(cfgs: dict, keys: dict) -> dict:
    import mlflow

    from aml import tracking
    from aml.serving.bundle import run_export

    paths = data_paths(cfgs["data"])
    version = require_data(paths, keys["data"])
    features_dir = paths.features_dir(keys["features"])
    rules_dir = paths.model_dir("rules_engine", keys["rules_engine"])
    graph_dir = paths.model_dir("lgbm_graph", keys["lgbm_graph"])
    require_stage_output(features_dir, "features", "features", data_version=version)
    require_stage_output(rules_dir, "rules_engine", "rules", data_version=version)
    require_stage_output(graph_dir, "lgbm_graph", "lgbm-graph", data_version=version)
    # The bundle's references are the feature table: it must have passed the DuckDB oracle, and
    # the rules and model must come from its current parts.
    verified = require_features_verified(features_dir)
    require_same_feature_table(
        features_dir,
        {"rules_engine": (rules_dir, "rules"), "lgbm_graph": (graph_dir, "lgbm-graph")},
    )
    bundle_dir = paths.serving_dir
    # Opened before compute: a failed attempt shows up as a FAILED run (and in the cost window).
    with tracking.start_run_for_key(APP_NAME, keys["export"], run_name=keys["export"]):
        tracking.log_params_flat({"keys": keys, "serving": cfgs["serving"]})
        tracking.log_params_flat({"features_verify": verified})
        mlflow.set_tags({"data_version": str(version)})
        summary = jsonable(
            run_export(
                paths,
                bundle_dir,
                cfgs,
                keys,
                features_dir=features_dir,
                rules_dir=rules_dir,
                graph_dir=graph_dir,
                threads=int(CPU),
                features_verify=verified,
            )
        )
        tracking.log_metrics_flat(summary)
    vol.commit()
    return jsonable({"run_key": keys["export"], "bundle_dir": str(bundle_dir), **summary})


@app.local_entrypoint()
def main() -> None:
    cfgs = load_all_configs()
    keys = all_keys(cfgs)
    summary = export.remote(cfgs, keys)
    print_summary(f"{APP_NAME} {keys['export']}", summary)
    v = summary.get("verification") or {}
    print(f"  verification: {v.get('rows')} slice rows replayed, mismatches {v.get('mismatches')}")
