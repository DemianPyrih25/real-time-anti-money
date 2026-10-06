"""gnn_bench: measure the GNN on real HI-Small batches before anything long runs (M3 spec §10).

    uv run modal run -m modal_jobs.smoke --gpu T4                 # make smoke GPU=T4, first
    uv run modal run --detach -m modal_jobs.gnn_bench             # make gnn-bench
    uv run modal run -m modal_jobs.gnn_bench --plan-only          # the gate table only

The entrypoint runs the cost gate (planning constants), then ONE CPU driver call runs the GPU
containers in order: L4 (every cell group), T4 (build, grid, stress, val, faithful), and the
4-core rerun of the winner only if its sampler wait at 3 workers is below
bench.rerun_4core_if_wait_below. Each container appends its cells to cells.jsonl (resume skips
recorded cells; a finished container is not called again). The driver then applies the
pre-registered decision rules (costplan.decide) and writes decision.json, plan.json, bench.json,
/data/reports/gnn_bench.md and summary.json (last). Every cell runs the as-of guard.

The lead copies decision.json's values into configs/gnn.yaml (sampler.batch_size,
max_edges_per_step, max_edges_per_eval_step, protocols.faithful.batch_size, runtime.*); HPO and
training refuse to start until they are equal. If the determinism overhead alone breaks the M3
cap (decision rule 6), the driver writes everything and then stops with GnnStopError: ask the
user. No labels beyond train and val_early are read.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
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

APP_NAME = "aml-gnn-bench"
app = modal.App(APP_NAME)

# Bench container shapes (M3 spec §10.1, §12.2): 8 cores / 32 GiB per GPU; the rerun at 4 / 16.
BENCH_CORES = 8
BENCH_MEMORY_MIB = 32768
WORKER_KW = gpu_job(gpu="L4", cpu=float(BENCH_CORES), memory_mib=BENCH_MEMORY_MIB, timeout=3600)
DRIVER_KW = cpu_job(cpu=0.25, memory_mib=1024, timeout=10800)
JOB = "bench"  # its RUN_ORDER id (costplan)
SUMMARY_DECISION_KEYS = (
    "gpu",
    "cores",
    "memory_mib",
    "memory_limit_mib",
    "num_workers",
    "batch_size",
    "max_edges_per_step",
    "max_edges_per_eval_step",
    "precision",
    "bit_deterministic",
    "det_overhead_ratio",
    "usd_epoch",
    "epoch_s",
    "rerun_4core",
    "ask_user",
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def planned_containers(gnn_cfg: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The GPU containers of the bench, in order (one per bench.gpus entry)."""
    from aml.models.gnn.costplan import CONTAINER_GROUPS

    return [
        {
            "name": gpu,
            "gpu": gpu,
            "cores": BENCH_CORES,
            "memory_mib": BENCH_MEMORY_MIB,
            "groups": list(CONTAINER_GROUPS[gpu]),
            "rerun": None,
        }
        for gpu in gnn_cfg["bench"]["gpus"]
    ]


def bench_counts(cells: list[Mapping[str, Any]]) -> dict | None:
    """The split counts the build cell measured ({n_pos, n_neg, n_val_early, ...})."""
    for c in cells:
        if c.get("group") == "build" and c.get("status", "ok") == "ok" and c.get("counts"):
            return dict(c["counts"])
    return None


def guard_totals(cells: list[Mapping[str, Any]]) -> dict[str, Any]:
    """GUARD totals per cell group and overall (the bench's real-data as-of evidence)."""
    from aml.models.gnn import add_guard

    by_group: dict[str, dict] = {}
    total = None
    for c in cells:
        if c.get("guard") is None:
            continue
        by_group[c["group"]] = add_guard(by_group.get(c["group"]), c["guard"])
        total = add_guard(total, c["guard"])
    return {"total": total, "by_group": by_group}


def decided_snippet(decision: Mapping[str, Any]) -> str:
    """The values the lead copies into configs/gnn.yaml (M3 spec §10.3)."""
    fa = decision.get("faithful") or {}
    return "\n".join(
        [
            "sampler:",
            f"  batch_size: {decision.get('batch_size')}",
            f"  max_edges_per_step: {decision.get('max_edges_per_step')}",
            f"  max_edges_per_eval_step: {decision.get('max_edges_per_eval_step')}",
            "protocols:",
            "  faithful:",
            f"    batch_size: {fa.get('batch_size')}   # on {fa.get('gpu')}",
            "runtime:",
            f"  gpu: {decision.get('gpu')}",
            f"  cpu: {decision.get('cores')}",
            f"  memory_mib: {decision.get('memory_mib')}",
            f"  num_workers: {decision.get('num_workers')}",
        ]
    )


def run_bench(
    gnn_cfg: Mapping[str, Any],
    bench_dir: Path,
    report_path: Path,
    *,
    call: Callable[[Mapping[str, Any]], Mapping[str, Any]],
    reload: Callable[[], None] | None = None,
    commit: Callable[[], None] | None = None,
    meta: Mapping[str, Any] | None = None,
    log: Callable[[str], None] = _log,
    clock: Callable[[], float] = time.monotonic,
) -> dict[str, Any]:
    """The bench driver's work (torch-free; `call(container)` runs one GPU container and
    returns {"cells": [...]}). A finished bench (summary.json) is returned as is. bench.json
    records each container's status and wall seconds, so a re-run never calls a finished one
    again; a failed container stops the bench ("failed", nothing decided). Then
    costplan.decide / plan_runs / gate / render_bench_md write decision.json, plan.json,
    report_path and summary.json (LAST). meta: {bench_key, data_version, features_digest, ...}
    copied into the summary."""
    from aml.io import read_json, write_json_atomic, write_text_atomic
    from aml.models.gnn import (
        BENCH_FILE,
        CELLS_FILE,
        DECISION_FILE,
        GNN_VERSION,
        PLAN_FILE,
        SUMMARY_FILE,
        costplan,
        read_jsonl,
    )

    bench_dir = Path(bench_dir)
    reload = reload or (lambda: None)
    commit = commit or (lambda: None)
    if (bench_dir / SUMMARY_FILE).exists():
        return {"status": "done", "summary": read_json(bench_dir / SUMMARY_FILE), "resumed": True}
    state_path = bench_dir / BENCH_FILE
    state = read_json(state_path) if state_path.exists() else {"containers": {}}
    state.setdefault("containers", {})

    def save() -> None:
        write_json_atomic(jsonable({**dict(meta or {}), **state}), state_path)
        commit()

    def run(container: dict[str, Any]) -> bool:
        name = container["name"]
        if (state["containers"].get(name) or {}).get("status") == "done":
            log(f"container {name}: done earlier, skipped")
            return True
        log(
            f"container {name}: {container['gpu']} x {container['cores']} cores, groups "
            f"{container['groups']}"
        )
        t0 = clock()
        try:
            out = call(container)
        except Exception as e:  # recorded; the bench stops (a re-run resumes at this container)
            state["containers"][name] = {
                **container,
                "status": "failed",
                "error": f"{type(e).__name__}: {e}",
                "seconds": clock() - t0,
                "ended_at": time.time(),  # the cost gate's billing-lag part (call_records)
            }
            save()
            log(f"container {name} failed: {type(e).__name__}: {e}")
            return False
        state["containers"][name] = {
            **container,
            "status": "done",
            "seconds": clock() - t0,
            "ended_at": time.time(),
            "n_cells": len(out.get("cells") or []),
        }
        save()
        return True

    def failed() -> dict[str, Any]:
        return {"status": "failed", "containers": state["containers"]}

    for c in planned_containers(gnn_cfg):
        if not run(c):
            return failed()
    reload()
    cells = read_jsonl(bench_dir / CELLS_FILE)
    rerun = costplan.rerun_4core(cells, gnn_cfg)
    state["rerun"] = rerun
    if rerun is not None:
        c = {
            "name": "rerun",
            "gpu": rerun["gpu"],
            "cores": int(rerun["cores"]),
            "memory_mib": int(rerun["memory_mib"]),
            "groups": list(costplan.CONTAINER_GROUPS["rerun"]),
            "rerun": dict(rerun),
        }
        if not run(c):
            return failed()
        reload()
        cells = read_jsonl(bench_dir / CELLS_FILE)
    try:
        decision = costplan.decide(cells, gnn_cfg)
    except ValueError as e:
        state["decide_error"] = str(e)
        save()
        log(f"no decision: {e}")
        return {"status": "failed", "error": str(e), "containers": state["containers"]}
    counts = bench_counts(cells)
    plan = costplan.plan_runs(gnn_cfg, decision=decision, counts=counts)
    containers = [c for c in state["containers"].values() if c.get("status") == "done"]
    bench_usd = sum(
        costplan.usd(
            costplan.plan_usd_h(c["gpu"], c["cores"], c["memory_mib"] / 1024),
            c["seconds"],
            float(gnn_cfg["budget"]["overhead"]),
        )
        for c in containers
    )
    spent = bench_usd + float(gnn_cfg["budget"]["dev_allowance_usd"])
    gate = costplan.gate(
        plan, gnn_cfg=gnn_cfg, spent_usd=spent, done=["bench"], job="dev", metered_usd=None
    )
    write_json_atomic(jsonable(decision), bench_dir / DECISION_FILE)
    write_json_atomic(
        jsonable(
            {
                "rows": plan,
                "gate": gate,
                "spent_estimate_usd": spent,
                "spent_source": "bench container wall x shape price x overhead + dev allowance "
                "(the entrypoints' gate uses the billing report)",
                "note": "the workspace budget pre-check needs `modal billing summary`, which only "
                "the local entrypoints run, so this gate reports it as unavailable; `make "
                "gnn-plan` shows the full gate",
            }
        ),
        bench_dir / PLAN_FILE,
    )
    write_text_atomic(costplan.render_bench_md(cells, decision, plan), Path(report_path))
    save()
    guard = guard_totals(cells)
    summary = {
        **dict(meta or {}),
        "status": "done",
        "decision": {k: decision.get(k) for k in SUMMARY_DECISION_KEYS if k in decision},
        "faithful": decision.get("faithful"),
        "counts": counts,
        "n_cells": len(cells),
        "containers": [
            {k: c.get(k) for k in ("name", "gpu", "cores", "memory_mib", "seconds", "n_cells")}
            for c in containers
        ],
        "gpu_seconds": sum(float(c["seconds"]) for c in containers),
        "bench_usd_estimate": bench_usd,
        "guard": guard["total"],
        "guard_by_group": guard["by_group"],
        "gnn_version": GNN_VERSION,
        "report": str(report_path),
    }
    write_json_atomic(jsonable(summary), bench_dir / SUMMARY_FILE)
    commit()
    return {"status": "done", "summary": summary, "decision": decision}


def _mlflow_decision_run(bench_key: str, result: Mapping[str, Any]) -> None:
    try:
        from aml import tracking

        with tracking.start_run_for_key(APP_NAME, bench_key, run_name=bench_key):
            s = result.get("summary") or {}
            tracking.log_params_flat({"decision": s.get("decision") or {}})
            tracking.log_metrics_flat(
                {
                    "gpu_seconds": s.get("gpu_seconds"),
                    "bench_usd_estimate": s.get("bench_usd_estimate"),
                }
            )
    except Exception as e:
        _log(f"warning: MLflow logging of the bench decision failed: {type(e).__name__}: {e}")


@app.function(**WORKER_KW)
def bench_gpu(spec: dict) -> dict:
    """One bench container: bench.run_bench_container for spec["groups"] (cells appended to
    cells.jsonl and committed one by one). Exceptions propagate (no retries): the driver
    records the failure."""
    from aml.models.gnn import BENCH_KIND
    from aml.models.gnn.bench import run_bench_container

    paths = data_paths(spec["data_cfg"])
    t0 = time.monotonic()
    cells = run_bench_container(
        paths,
        paths.features_dir(spec["keys"]["features"]),
        paths.gnn_set_dir(BENCH_KIND, spec["keys"]["gnn_bench"]),
        spec["gnn_cfg"],
        data_cfg=spec["data_cfg"],
        gpu=spec["gpu"],
        cores=int(spec["cores"]),
        memory_mib=int(spec["memory_mib"]),
        groups=list(spec["groups"]),
        device="cuda",
        rerun=spec.get("rerun"),
        on_checkpoint=vol.commit,
        log=_log,
    )
    vol.commit()
    # bench.run_bench_container logs this container's MLflow run itself.
    return {"cells": jsonable(cells), "seconds": time.monotonic() - t0}


def container_options(container: Mapping[str, Any]) -> dict[str, Any]:
    """`with_options` of a bench container (thread caps follow the cores when they differ)."""
    cores, mem = float(container["cores"]), int(container["memory_mib"])
    opts: dict[str, Any] = {
        "gpu": container["gpu"],
        "cpu": (cores, cores),
        "memory": (mem, mem + 8192),
    }
    if cores != float(BENCH_CORES):
        opts["env"] = gpu_job(cpu=cores)["env"]
    return opts


@app.function(**DRIVER_KW)
def bench_driver(spec: dict) -> dict:
    """Checks the inputs, stamps the bench dir, runs the containers and decides (run_bench)."""
    from aml.features.spec import FEATURES_DIGEST
    from aml.io import read_json
    from aml.models.gnn import BENCH_KIND, BENCH_REPORT, GnnStopError
    from modal_jobs.train_gnn import prepare_set_dirs, require_inputs

    paths = data_paths(spec["data_cfg"])
    version = require_inputs(paths, spec["keys"])
    bench_key = spec["keys"]["gnn_bench"]
    bench_dir = paths.gnn_set_dir(BENCH_KIND, bench_key)
    prepare_set_dirs([bench_dir], version)
    vol.commit()
    features_dir = paths.features_dir(spec["keys"]["features"])
    meta = {
        "bench_key": bench_key,
        "data_version": version,
        FEATURES_DIGEST: read_json(features_dir / "summary.json").get(FEATURES_DIGEST),
    }

    def call(container: Mapping[str, Any]) -> Mapping[str, Any]:
        fn = bench_gpu.with_options(**container_options(container))
        return fn.remote(
            {**spec, **{k: container[k] for k in ("gpu", "cores", "memory_mib", "groups", "rerun")}}
        )

    res = run_bench(
        spec["gnn_cfg"],
        bench_dir,
        paths.reports / BENCH_REPORT,
        call=call,
        reload=vol.reload,
        commit=vol.commit,
        meta=meta,
    )
    if res["status"] == "done" and not res.get("resumed"):
        _mlflow_decision_run(bench_key, res)
        vol.commit()
    ask = ((res.get("summary") or {}).get("decision") or {}).get("ask_user")
    if ask:
        raise GnnStopError(f"gnn_bench decision rule 6: {ask} (outputs are written; ask the user)")
    return jsonable(res)


@app.local_entrypoint()
def main(plan_only: bool = False) -> None:
    from aml.models.gnn import GnnStopError, check_gnn_cfg
    from modal_jobs.train_gnn import print_gate, run_gate, volume_state

    cfgs = load_all_configs()
    g = check_gnn_cfg(cfgs["gnn"])
    state = volume_state(cfgs)
    if JOB in state["done"] and not plan_only:
        print_summary(
            f"{APP_NAME} {state['keys']['gnn_bench']} (already finished)", state["bench_summary"]
        )
        if state["decision"]:
            _log(decided_snippet(state["decision"]))
        return
    gate = run_gate(cfgs, state, [JOB])
    print_gate(gate)
    if plan_only:
        return
    if not gate["allowed"]:
        raise GnnStopError("the cost gate refused: " + "; ".join(gate["reasons"]))
    spec = {"data_cfg": cfgs["data"], "gnn_cfg": g, "keys": state["keys"]}
    _log(f"== submitting gnn_bench {state['keys']['gnn_bench']} on {g['bench']['gpus']}")
    res = bench_driver.remote(spec)
    print_summary(f"{APP_NAME} {state['keys']['gnn_bench']}", res)
    if res.get("status") != "done":
        raise SystemExit(f"{APP_NAME}: {res.get('status')}: {res.get('error') or res}")
    print_summary("bench summary", res["summary"])
    decision = res.get("decision") or state.get("decision")
    if decision:
        _log("== copy into configs/gnn.yaml (then commit it before any --final run):")
        _log(decided_snippet(decision))
