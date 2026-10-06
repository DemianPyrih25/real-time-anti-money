"""The causal GNN's hyperparameter search on a Modal GPU, in chunks under a CPU driver (M3 spec §8).

    uv run modal run --detach -m modal_jobs.hpo_gnn               # make gnn-hpo
    uv run modal run -m modal_jobs.hpo_gnn --plan-only            # the gate table only

Needs a finished gnn_bench whose decision.json equals gnn.yaml's decided values. The entrypoint
runs the cost gate, then ONE driver call runs the GPU worker (hpo.run_hpo) in chunks until it
returns done / failed / stopped / busy or the wall guard fires (STOPPED.json), exactly like
modal_jobs.train_gnn. The Optuna trial log /data/optuna/<gnn_hpo_key>.jsonl is the study's source
of truth: every finished trial is appended and committed, so a re-run resumes. Output:
/data/models/gnn_hpo/<gnn_hpo_key>/best_params.json (read by `make gnn` / `make gnn-lookahead`).
HPO reads train and val_early labels only.
"""

from __future__ import annotations

import traceback
from typing import Any

import modal

from modal_jobs.common import (
    cpu_job,
    data_paths,
    gpu_job,
    jsonable,
    load_all_configs,
    print_summary,
    vol,
)

APP_NAME = "aml-hpo-gnn"
app = modal.App(APP_NAME)

STATIC_GPU = "L4"
STATIC_CORES = 8.0
STATIC_MEMORY_MIB = 32768
STATIC_WORKER_TIMEOUT = 7200 + 1800  # §12.1 fallback; each call sets T via with_options
WORKER_KW = gpu_job(
    gpu=STATIC_GPU,
    cpu=STATIC_CORES,
    memory_mib=STATIC_MEMORY_MIB,
    timeout=STATIC_WORKER_TIMEOUT,
    retries=True,  # transient errors only: deterministic ones return "failed"
)
DRIVER_KW = cpu_job(cpu=0.25, memory_mib=1024, timeout=21600)
JOB = "hpo"  # its RUN_ORDER id (costplan)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _failed(e: BaseException) -> dict[str, Any]:
    tb = traceback.format_exc()
    _log(tb)
    return {"status": "failed", "error": f"{type(e).__name__}: {e}", "traceback": tb[-4000:]}


@app.function(**WORKER_KW)
def hpo_gpu(spec: dict) -> dict:
    """One chunk of the search (hpo.run_hpo within spec["budget_s"]). Deterministic errors
    return "failed"; transient ones are re-raised (Modal retries; the search resumes from the
    trial log and the current trial's checkpoint). Only `Exception` is caught."""
    try:
        from aml.models.gnn import HPO_KIND
        from aml.models.gnn.hpo import run_hpo
        from aml.models.gnn.train import is_transient
    except Exception as e:  # a broken image or import: deterministic, never retried
        return _failed(e)
    paths = data_paths(spec["data_cfg"])
    try:
        out = run_hpo(
            paths,
            paths.features_dir(spec["keys"]["features"]),
            paths.gnn_set_dir(HPO_KIND, spec["keys"]["gnn_hpo"]),
            spec["gnn_cfg"],
            data_cfg=spec["data_cfg"],
            device="cuda",
            runtime=spec["runtime"],
            budget_s=float(spec["budget_s"]),
            on_checkpoint=vol.commit,
            on_reload=vol.reload,
            log=_log,
        )
    except Exception as e:
        if is_transient(e):
            raise
        return _failed(e)
    vol.commit()
    return jsonable(out)


@app.function(**DRIVER_KW)
def hpo_driver(spec: dict) -> dict:
    """Checks the inputs, stamps the HPO dir (other prepared data: emptied, and its trial log
    removed), then runs the worker in chunks (train_gnn.drive_chunks)."""
    from aml.models.gnn import HPO_KIND
    from aml.models.gnn.hpo import finished_hpo  # torch-free
    from modal_jobs.common import reset_if_other_data
    from modal_jobs.train_gnn import (
        DRIVER_DEADLINE_MARGIN_S,
        drive_chunks,
        prepare_set_dirs,
        require_inputs,
    )

    paths = data_paths(spec["data_cfg"])
    version = require_inputs(paths, spec["keys"])
    hpo_key = spec["keys"]["gnn_hpo"]
    hpo_dir = paths.gnn_set_dir(HPO_KIND, hpo_key)
    trial_log = paths.gnn_optuna_log(hpo_key)
    # The trial log lives outside hpo_dir; it belongs to the trials, so it goes with them.
    if reset_if_other_data(hpo_dir, version) and trial_log.exists():
        trial_log.unlink()
        _log(f"{trial_log}: trials from other prepared data removed")
    prepare_set_dirs([hpo_dir], version)
    vol.commit()
    # the worker's own rule: built from the current data and feature parts
    stored = finished_hpo(paths, paths.features_dir(spec["keys"]["features"]), hpo_dir)
    if stored is not None:
        return {"status": "done", "summary": stored, "chunks": []}
    worker = hpo_gpu.with_options(**spec["options"])
    _log(f"HPO {hpo_key}: {spec['budget']}")
    res = drive_chunks(
        lambda budget_s: worker.remote({**spec, "budget_s": budget_s}),
        budget=spec["budget"],
        set_dir=hpo_dir,
        commit=vol.commit,
        deadline_s=DRIVER_KW["timeout"] - DRIVER_DEADLINE_MARGIN_S,
    )
    return jsonable(res)


@app.local_entrypoint()
def main(plan_only: bool = False) -> None:
    from aml.models.gnn import GnnStopError, check_gnn_cfg
    from modal_jobs.train_gnn import (
        BUSY_HINT,
        budget_rows,
        check_decision,
        chunk_budget,
        print_gate,
        run_gate,
        volume_state,
        worker_options,
    )

    cfgs = load_all_configs()
    g = check_gnn_cfg(cfgs["gnn"])
    state = volume_state(cfgs)
    if plan_only:
        print_gate(run_gate(cfgs, state, [JOB]))
        return
    decision = check_decision(state, g)
    if JOB in state["done"]:
        print_summary(
            f"{APP_NAME} {state['keys']['gnn_hpo']} (already finished)", state["hpo_summary"]
        )
        _log(f"best params: {state['best_params']}")
        return
    gate = run_gate(cfgs, state, [JOB])
    print_gate(gate)
    if not gate["allowed"]:
        raise GnnStopError("the cost gate refused: " + "; ".join(gate["reasons"]))
    budget = chunk_budget(budget_rows(gate, [JOB]), g)
    options = worker_options(g, decision, protocol="causal", timeout=budget["timeout_s"])
    spec = {
        "data_cfg": cfgs["data"],
        "gnn_cfg": g,
        "keys": state["keys"],
        "runtime": {**g["runtime"], "gpu": options["gpu"]},
        "budget": budget,
        "options": options,
    }
    _log(
        f"== submitting HPO {state['keys']['gnn_hpo']} ({g['hpo']['n_trials']} trials) on "
        f"{options['gpu']} ({budget['attempt_wall_s']:.0f} s per chunk, timeout "
        f"{budget['timeout_s']} s, wall guard {budget['max_set_wall_s']:.0f} s)"
    )
    res = hpo_driver.remote(spec)
    print_summary(f"{APP_NAME} {state['keys']['gnn_hpo']}", res)
    if isinstance(res.get("summary"), dict):
        print_summary("HPO summary", res["summary"])
    status = res.get("status")
    if status == "stopped":
        raise GnnStopError(f"the driver stopped the search ({res.get('reason')}): re-run to resume")
    if status == "busy":
        raise SystemExit(f"{APP_NAME}: busy ({res.get('error')}): {BUSY_HINT}")
    if status != "done":
        raise SystemExit(f"{APP_NAME}: {status}: {res.get('error') or res}")
