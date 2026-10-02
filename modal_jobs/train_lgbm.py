"""LightGBM: Optuna on val_early, then the seed finalists (M1 transaction features, M2 graph).

    uv run modal run --detach -m modal_jobs.train_lgbm                      # M1: --feature-set tx
    uv run modal run --detach -m modal_jobs.train_lgbm --feature-set graph  # M2 (make lgbm-graph)
    uv run modal run --detach -m modal_jobs.train_lgbm --feature-set graph --nofmt-final

Outputs: /data/models/lgbm_tx/<lgbm_key>/ (M1) and /data/models/lgbm_graph/<lgbm_graph_key>/
(M2: gate, trials, ablations, finalists, SHAP; read by modal_jobs.evaluate). `--nofmt-final` also
writes /data/models/lgbm_graph_nofmt/<lgbm_graph_key>/ (val_late + test scores of the
payment-format-free model); run it only after the user approves that extra test touch (M2 spec
§13). Test labels are never read. Every finished trial, ablation fit and finalist is checkpointed
and committed, so re-running the same command after a timeout or preemption resumes. A search
that has ended (all trials, or `optuna.timeout_s`) is recorded in tuning_done.json and replayed,
never extended, so `--nofmt-final` on a finished stage reuses every fit and refuses to change the
evaluated champion. If the graph gate would drop more than `graph.gate.max_drop_share` of the
engine features, the job writes gate.json and fails with GateStopError before any fit: the user
decides.
"""

from __future__ import annotations

import modal

from modal_jobs.common import (
    cpu_job,
    data_key,
    data_paths,
    features_key,
    jsonable,
    lgbm_graph_key,
    lgbm_key,
    lgbm_tx_cfg,
    load_all_configs,
    print_summary,
    require_data,
    require_stage_output,
    reset_if_other_data,
    vol,
)

APP_NAME = "aml-train-lgbm"
CPU = 8.0
FEATURE_SETS = ("tx", "graph")
app = modal.App(APP_NAME)


@app.function(**cpu_job(cpu=CPU, memory_mib=16384, timeout=3 * 3600))
def train(lgbm_cfg: dict, rules_cfg: dict, data_cfg: dict, keys: dict) -> dict:
    import mlflow

    from aml import tracking
    from aml.models.lgbm import run_lgbm_stage

    paths = data_paths(data_cfg)
    version = require_data(paths, keys["data"])
    out_dir = paths.model_dir("lgbm_tx", keys["lgbm_tx"])
    # Checkpoints from other prepared data under the same config key must not be resumed.
    if reset_if_other_data(out_dir, version):
        print(f"{out_dir}: built from other prepared data; starting over")
    vol.commit()
    # Opened before compute: a failed or timed-out attempt shows up as a FAILED run (and in the
    # cost window); a retry resumes the same run and the checkpoints below.
    with tracking.start_run_for_key(APP_NAME, keys["lgbm_tx"], run_name=keys["lgbm_tx"]):
        tracking.log_params_flat(lgbm_cfg, prefix="lgbm")
        tracking.log_params_flat(
            {"data_key": keys["data"], "round_unit": rules_cfg.get("round_unit")}
        )
        mlflow.set_tags({"data_version": str(version)})
        # Every finished trial and finalist is committed, so a timeout or preemption keeps them.
        summary = jsonable(
            run_lgbm_stage(
                paths, out_dir, lgbm_cfg, rules_cfg, threads=int(CPU), on_checkpoint=vol.commit
            )
        )
        trials = summary.pop("trials", None) or []
        best = out_dir / "best_params.json"
        if best.exists():
            from aml.io import read_json

            tracking.log_params_flat(read_json(best), prefix="best")
        tracking.log_metrics_flat(summary)
        tracking.log_trials(trials)
    vol.commit()
    return jsonable(
        {"run_key": keys["lgbm_tx"], "out_dir": str(out_dir), "n_trials": len(trials), **summary}
    )


@app.function(**cpu_job(cpu=CPU, memory_mib=16384, timeout=3 * 3600))
def train_graph(lgbm_cfg: dict, data_cfg: dict, keys: dict, nofmt_final: bool = False) -> dict:
    import mlflow

    from aml import tracking
    from aml.features.spec import GateStopError
    from aml.io import read_json
    from aml.models.lgbm_graph import (
        ABLATION_FILE,
        GATE_FILE,
        NOFMT_KIND,
        SHAP_FILE,
        effective_graph_cfg,
        run_lgbm_graph_stage,
    )

    paths = data_paths(data_cfg)
    version = require_data(paths, keys["data"])
    features_dir = paths.features_dir(keys["features"])
    require_stage_output(features_dir, "features", "features", data_version=version)
    out_dir = paths.model_dir("lgbm_graph", keys["lgbm_graph"])
    if reset_if_other_data(out_dir, version):
        print(f"{out_dir}: built from other prepared data; starting over")
    nofmt_dir = paths.model_dir(NOFMT_KIND, keys["lgbm_graph"])
    if nofmt_final and reset_if_other_data(nofmt_dir, version):
        print(f"{nofmt_dir}: built from other prepared data; starting over")
    vol.commit()
    with tracking.start_run_for_key(APP_NAME, keys["lgbm_graph"], run_name=keys["lgbm_graph"]):
        tracking.log_params_flat(effective_graph_cfg(lgbm_cfg), prefix="lgbm")
        tracking.log_params_flat(lgbm_cfg.get("graph") or {}, prefix="graph")
        tracking.log_params_flat(
            {"data_key": keys["data"], "features_key": keys["features"], "feature_set": "graph"}
        )
        mlflow.set_tags({"data_version": str(version), "nofmt_final": str(bool(nofmt_final))})
        try:
            summary = jsonable(
                run_lgbm_graph_stage(
                    paths,
                    features_dir,
                    out_dir,
                    lgbm_cfg,
                    threads=int(CPU),
                    nofmt_final=nofmt_final,
                    nofmt_dir=nofmt_dir,
                    on_checkpoint=vol.commit,
                    log=lambda msg: print(msg, flush=True),
                )
            )
        except GateStopError:
            vol.commit()  # gate.json stays on the Volume for the user's decision
            if (out_dir / GATE_FILE).exists():
                mlflow.log_artifact(str(out_dir / GATE_FILE))
            raise
        trials = summary.pop("trials", None) or []
        tracking.log_params_flat(read_json(out_dir / "best_params.json"), prefix="best")
        tracking.log_metrics_flat(summary)
        tracking.log_trials(trials)
        for name in (GATE_FILE, ABLATION_FILE, SHAP_FILE):
            mlflow.log_artifact(str(out_dir / name))
    vol.commit()
    return jsonable(
        {"run_key": keys["lgbm_graph"], "out_dir": str(out_dir), "n_trials": len(trials), **summary}
    )


@app.local_entrypoint()
def main(feature_set: str = "tx", nofmt_final: bool = False) -> None:
    if feature_set not in FEATURE_SETS:
        raise SystemExit(f"--feature-set must be one of {FEATURE_SETS}, got {feature_set!r}")
    if nofmt_final and feature_set != "graph":
        raise SystemExit("--nofmt-final belongs to --feature-set graph")
    cfgs = load_all_configs()
    if feature_set == "tx":
        keys = {"data": data_key(cfgs), "lgbm_tx": lgbm_key(cfgs)}
        # The M1 stage gets lgbm.yaml without `graph`: its fingerprint and MLflow params are the
        # M1 ones, so the M1 checkpoints stay valid.
        summary = train.remote(lgbm_tx_cfg(cfgs["lgbm"]), cfgs["rules"], cfgs["data"], keys)
        print_summary(f"{APP_NAME} {keys['lgbm_tx']}", summary)
        return
    keys = {
        "data": data_key(cfgs),
        "features": features_key(cfgs),
        "lgbm_graph": lgbm_graph_key(cfgs),
    }
    summary = train_graph.remote(cfgs["lgbm"], cfgs["data"], keys, nofmt_final)
    print_summary(f"{APP_NAME} graph {keys['lgbm_graph']}", summary)
