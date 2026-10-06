"""The resumable GNN hyperparameter search (M3 spec §8; owner E).

One GPU container (chunked by the driver like run_set): the graph, the loaders and the
val_early eval cache are built once and shared by every trial. A small local search around
Multi-GNN's tuned config: lr log-uniform, final_dropout uniform, w_pos log-uniform
(gnn.yaml hpo.space); trial 0 = the train section's values via `enqueue_trial`; model seed
hpo.model_seed for every trial (common random numbers: identical init and negatives per epoch).

Resume: the trial log `/data/optuna/<gnn_hpo_key>.jsonl` (paths.gnn_optuna_log) is the source of
truth. Before every ask() the study is rebuilt from it (`optuna.trial.create_trial` +
`study.add_trial`) and `study.sampler = TPESampler(n_startup_trials, seed=sampler_seed_base + k)`
is assigned, so trial k proposes identical params on every rebuild (F20). Asked params are
written to `trial_<k>/params.json` BEFORE training; a resumed unfinished trial asserts the
re-asked params equal them and resumes from `trial_<k>/last.pt` (train.py's run-dir layout),
re-reporting its finished epochs to the pruner first.

HPO dir `/data/models/gnn_hpo/<gnn_hpo_key>/`: trials.json (the trial table), best_params.json
({lr, final_dropout, w_pos}; argmax value over COMPLETE trials, ties -> lower number; written
last before summary), summary.json (LAST: {hpo_key, n_trials, n_complete, n_pruned, n_failed,
best_trial, best_value, best_params, trials, guard (GUARD totals per split), edges_checked,
violations, target_hits, seconds, gpu_seconds, features_digest, data_version, spec_hash,
gnn_version}), data_version.json, writer.json, calls.jsonl, trial_<n>/.
HPO never builds a val_late or test loader and never reads their labels (asserted: its engine
allows no scoring split and load_graph gets LABEL_SPLITS).
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aml.models.gnn import (
    BEST_PARAMS_FILE,
    GNN_VERSION,
    HPO_KIND,
    LABEL_SPLITS,
    PARAMS_FILE,
    RUNNING_FILE,
    STOPPED_FILE,
    SUMMARY_FILE,
    TRIALS_FILE,
    add_guard,
    append_jsonl,
    empty_guard,
    read_jsonl,
    trial_dir_name,
)

if TYPE_CHECKING:
    import optuna

    from aml.paths import DataPaths

# Trial-log record states (optuna.trial.TrialState names).
TRIAL_STATES = ("COMPLETE", "PRUNED", "FAIL")
OBJECTIVE_DIRECTION = "maximize"  # val_early PR-AUC of the best epoch
EXPERIMENT = "aml-hpo-gnn"  # MLflow: one study run (key = hpo key) + one run per trial
HPO_PARAMS = ("lr", "final_dropout", "w_pos")  # the search space, in suggestion order


def run_hpo(
    paths: DataPaths,
    features_dir: Path,
    hpo_dir: Path,
    gnn_cfg: dict,
    *,
    data_cfg: dict,
    device: str,
    runtime: dict,
    budget_s: float,
    on_checkpoint: Callable[[], None] | None = None,
    on_reload: Callable[[], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Run (resume) up to hpo.n_trials trials of hpo.max_epochs each (early stopping as §7.4,
    MedianPruner(n_startup_trials, n_warmup_steps) on per-epoch val_early PR-AUC).

    hpo_dir: paths.gnn_set_dir(HPO_KIND, gnn_hpo_key); the trial log is
    paths.gnn_optuna_log(hpo_dir.name). data_cfg: configs/data.yaml (graph.load_graph).
    budget_s / statuses / hooks as train.run_set (an attempt that runs out of wall returns
    "partial" after checkpointing the current trial's epoch).
    An OOM that survives the in-process split -> the trial is logged FAIL (with the batch's edge
    count), empty_cache(), the search continues. Any other deterministic error -> FAILED.json
    and "failed"; transient errors are re-raised (Modal retries; the search resumes).

    Returns {"status": "done" | "partial" | "failed" | "stopped" | "busy", "hpo_dir",
    "n_trials_done", "best_params": dict | None, "best_value": float | None, "best_trial":
    int | None, "next": {"trial", "epoch"} | None, "elapsed_s", "gpu_seconds", "guard", "error":
    str | None, "summary": the summary.json content when done, else None}.
    """
    from aml.models.gnn import train as tr

    log = log or tr._print
    commit = on_checkpoint or tr._noop
    hpo_dir = Path(hpo_dir)
    clock = tr.Clock(budget_s)
    res: dict[str, Any] = {
        "status": "failed",
        "hpo_dir": str(hpo_dir),
        "n_trials_done": 0,
        "best_params": None,
        "best_value": None,
        "best_trial": None,
        "next": None,
        "elapsed_s": 0.0,
        "gpu_seconds": 0.0,
        "guard": empty_guard(),
        "error": None,
        "summary": None,
    }
    stored = finished_hpo(paths, features_dir, hpo_dir)
    if stored is not None:
        return {**res, **_result_from_summary(stored), "status": "done", "summary": stored}
    if on_reload is not None:
        on_reload()
    if (hpo_dir / STOPPED_FILE).exists():
        return {**res, "status": "stopped"}
    hpo_dir.mkdir(parents=True, exist_ok=True)
    lease = tr.WriterLease(hpo_dir)
    if not lease.acquire():
        holder = lease.holder() or {}
        return {**res, "status": "busy", "error": f"writer lease held by {holder.get('call_id')}"}
    commit()
    st = tr._CallState()
    try:
        out = _run_hpo_body(
            paths,
            features_dir,
            hpo_dir,
            gnn_cfg,
            data_cfg=data_cfg,
            device=device,
            runtime=runtime,
            clock=clock,
            lease=lease,
            commit=commit,
            log=log,
            st=st,
            res=res,
        )
    except Exception as exc:
        if tr.is_transient(exc):
            raise
        fail_dir = st.run_dir or hpo_dir
        if st.run_dir is not None:
            running = Path(st.run_dir) / RUNNING_FILE
            st.epoch = (tr._read_json(running) or {}).get("epoch_next", st.epoch)
            running.unlink(missing_ok=True)
        tr._write_failed(fail_dir, f"{type(exc).__name__}: {exc}", exc, epoch=st.epoch, st=st)
        log(f"FAILED ({fail_dir}): {type(exc).__name__}: {exc}")
        out = {**res, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    finally:
        st.close_engine()
    out["elapsed_s"] = clock.elapsed()
    out["gpu_seconds"] = clock.elapsed() if device == "cuda" else 0.0
    out["guard"] = st.guard
    append_jsonl(
        hpo_dir / tr.CALLS_FILE,
        {
            "call_id": lease.call_id,
            "status": out["status"],
            "elapsed_s": round(clock.elapsed(), 3),
            "device": device,
            "ended_at": round(time.time(), 3),
        },
    )
    lease.release()
    commit()
    return out


def finished_hpo(paths: DataPaths, features_dir: Path, hpo_dir: Path) -> dict | None:
    """The stored HPO summary if it was built from the prepared data and feature parts on the
    Volume now (run_hpo and the CPU driver short-circuit on it; torch-free)."""
    from aml.models.gnn import train as tr

    stored = tr._read_json(Path(hpo_dir) / SUMMARY_FILE)
    if stored is not None and tr.built_from_current(stored, paths, features_dir):
        return stored
    return None


def _run_hpo_body(
    paths,
    features_dir,
    hpo_dir: Path,
    gnn_cfg: dict,
    *,
    data_cfg,
    device,
    runtime,
    clock,
    lease,
    commit,
    log,
    st,
    res: dict,
) -> dict:
    from aml.models.gnn import graph as G
    from aml.models.gnn import train as tr

    h = gnn_cfg["hpo"]
    n_trials = int(h["n_trials"])
    hpo_key = hpo_dir.name
    log_path = paths.gnn_optuna_log(hpo_key)
    version, digest = tr.current_inputs(paths, features_dir)
    if tr._reset_if_other_data(hpo_dir, version):
        log(f"{hpo_dir}: built from other prepared data; starting over")
        lease.refresh()
    records = _load_records(log_path, version, digest, log)
    if records is None or (hpo_dir / SUMMARY_FILE).exists():
        # Outputs of other inputs (a current summary was returned by run_hpo): start over.
        _clear_outputs(hpo_dir)
        records = records or []
    records = records[:n_trials]
    g = None
    if len(records) < n_trials:
        t0 = time.perf_counter()
        g = G.load_graph(
            paths,
            features_dir,
            gnn_cfg,
            data_cfg=data_cfg,
            label_splits=LABEL_SPLITS,
            protocol="causal",
            log=log,
        )
        log(f"graph: {g.n_edges} edges ({time.perf_counter() - t0:.1f}s)")
        engine = tr.TemporalEngine(
            g,
            gnn_cfg,
            protocol="causal",
            params=tr.effective_params(gnn_cfg, "causal", trial0_params(gnn_cfg)),
            device=device,
            runtime=runtime,
            allowed_splits=(),  # validation only: no val_late / test loader, ever
            log=log,
        )
        st.engine = engine
        for k in range(len(records), n_trials):
            rec = _run_trial(
                engine,
                gnn_cfg,
                hpo_dir,
                k,
                records,
                clock=clock,
                lease=lease,
                commit=commit,
                log=log,
                st=st,
            )
            if rec is None:  # out of wall: the trial resumes in the next call
                nxt = {"trial": k, "epoch": st.epoch}
                st.run_dir = st.run_key = st.seed = None
                return {**res, "status": "partial", "n_trials_done": len(records), "next": nxt}
            if rec.get("status") == "failed":
                st.run_dir = st.run_key = st.seed = None
                return {**res, "status": "failed", "error": rec["error"]}
            rec["data_version"], rec["features_digest"] = version, digest
            append_jsonl(log_path, rec)
            commit()
            records.append(rec)
            st.run_dir = st.run_key = st.seed = None
            log(f"trial {k}: {rec['state']} value {rec['value']} params {rec['params']}")
    summary = _finalize(
        paths, features_dir, hpo_dir, gnn_cfg, records, g, clock, device, runtime, commit, log
    )
    return {
        **res,
        **_result_from_summary(summary),
        "status": "done",
        "summary": summary,
    }


def _run_trial(
    engine,
    gnn_cfg: dict,
    hpo_dir: Path,
    k: int,
    records: list[dict],
    *,
    clock,
    lease,
    commit,
    log,
    st,
) -> dict | None:
    """Ask trial k (rebuilt study), persist its params, train it (resume) with pruning.
    Returns its log record, None when out of wall, or {"status": "failed", "error"}."""
    from aml.models.gnn import train as tr

    h, t = gnn_cfg["hpo"], gnn_cfg["train"]
    study = rebuild_study(records, gnn_cfg, k)
    trial = study.ask(_distributions(gnn_cfg))
    if trial.number != k:
        raise AssertionError(f"the rebuilt study asked trial {trial.number}, expected {k}")
    tp = {name: float(trial.params[name]) for name in HPO_PARAMS}
    trial_dir = hpo_dir / trial_dir_name(k)
    pfile = trial_dir / PARAMS_FILE
    saved = tr._read_json(pfile)
    if saved is not None and tr._plain(saved) != tr._plain(tp):
        raise ValueError(
            f"{pfile}: trial {k} was started with {saved}, the rebuilt study now asks {tp} "
            "(the trial log or the search config changed)"
        )
    if saved is None:
        tr._write_json(pfile, tp)
        commit()
    params = tr.effective_params(gnn_cfg, "causal", tp)
    engine.set_params(params)
    seed = int(h["model_seed"])
    run_key = f"{hpo_dir.name}-t{k}"
    max_e = int(h["max_epochs"])
    spec = tr.RunSpec(
        run_key=run_key,
        run_dir=trial_dir,
        protocol="causal",
        seed=seed,
        params=tr._plain(params),
        max_epochs=max_e,
        min_epochs=min(int(t["min_epochs"]), max_e),
        patience=int(t["patience"]),
        early_stopping=True,
        deterministic=bool(t["deterministic"]),
        matmul_precision=str(t["matmul_precision"]),
        fingerprint=tr.fingerprint(engine.g, run_key, "causal", seed, params, engine.preprocess),
        graph_meta=tr.graph_meta(engine.g, "causal", engine.preprocess),
        mlflow_experiment=EXPERIMENT,
        mlflow_tags={"trial": str(k), "hpo_key": hpo_dir.name, "gnn_version": str(GNN_VERSION)},
    )
    st.run_dir, st.run_key, st.seed = trial_dir, run_key, seed

    reported: set[int] = set()

    def report(history: list[dict]) -> None:
        for r in history:
            v, e = tr._finite_or_none(r.get("metric")), int(r["epoch"])
            if v is not None and e not in reported:  # NaN (no positives) is not reported
                trial.report(v, e)
                reported.add(e)

    def on_epoch(epoch: int, metric: float, history: list[dict]) -> bool:
        report(history[-1:])
        return bool(trial.should_prune())

    def on_resume(history: list[dict]) -> bool:
        report(history)
        return bool(history) and bool(trial.should_prune())

    t0 = time.perf_counter()
    try:
        out = tr.train_run(
            engine,
            spec,
            clock,
            on_checkpoint=commit,
            on_epoch=on_epoch,
            on_resume=on_resume,
            heartbeat=lease.refresh,
            log=log,
        )
    except tr.GnnOOMError as e:
        tr._free_cuda(engine.device)
        history = read_jsonl(trial_dir / tr.HISTORY_FILE)
        (trial_dir / RUNNING_FILE).unlink(missing_ok=True)
        rec = trial_record(
            k,
            tp,
            "FAIL",
            None,
            _intermediate(history),
            None,
            _seconds(history) + (time.perf_counter() - t0),
            _history_guard(history),
        )
        rec["error"] = f"CUDA OOM at the minimal split: {e.edges} edges ({e.subgraphs} subgraphs)"
        return rec
    st.guard = add_guard(st.guard, out.guard)
    st.epoch = out.epoch_next
    if out.status == "partial":
        return None
    if out.status == "failed":
        return {"status": "failed", "error": out.error}
    history = out.history or read_jsonl(trial_dir / tr.HISTORY_FILE)
    iv = _intermediate(history)
    best_epoch = history[-1].get("best_epoch") if history else None
    if out.status == "pruned":
        last = iv[max(iv)] if iv else None
        return trial_record(
            k, tp, "PRUNED", last, iv, best_epoch, _seconds(history), _history_guard(history)
        )
    value = tr._finite_or_none((out.summary or {}).get("metric"))
    if value is None:
        rec = trial_record(
            k, tp, "FAIL", None, iv, best_epoch, _seconds(history), _history_guard(history)
        )
        rec["error"] = "non-finite objective (val_early PR-AUC undefined)"
        return rec
    best_epoch = (out.summary or {}).get("best_epoch", best_epoch)
    return trial_record(
        k, tp, "COMPLETE", value, iv, best_epoch, _seconds(history), _history_guard(history)
    )


def _finalize(
    paths,
    features_dir,
    hpo_dir: Path,
    gnn_cfg: dict,
    records,
    g,
    clock,
    device,
    runtime,
    commit,
    log,
) -> dict:
    from aml.models.gnn import train as tr

    best = best_params(records)
    if best is None:
        raise ValueError(f"no COMPLETE trial among {len(records)}: no best params to train with")
    number, value, params = best
    guard: dict[str, dict] = {}
    for r in records:
        for split, gv in (r.get("guard") or {}).items():
            guard[split] = add_guard(guard.get(split), gv)
    info = _graph_info(g, paths, features_dir)
    calls = read_jsonl(hpo_dir / tr.CALLS_FILE)
    gpu_s = sum(float(c.get("elapsed_s") or 0) for c in calls if c.get("device") == "cuda")
    if device == "cuda":
        gpu_s += clock.elapsed()
    states = [r["state"] for r in records]
    summary = tr._plain(
        {
            "hpo_key": hpo_dir.name,
            "n_trials": int(gnn_cfg["hpo"]["n_trials"]),
            "n_complete": states.count("COMPLETE"),
            "n_pruned": states.count("PRUNED"),
            "n_failed": states.count("FAIL"),
            "best_trial": number,
            "best_value": value,
            "best_params": params,
            "trials": records,
            "guard": guard,
            **tr._guard_totals(guard),
            "seconds": float(sum(float(r.get("seconds") or 0) for r in records)),
            # gpu / gpu_seconds / cores / memory_mib: the cost gate's fallback (billing fails)
            "gpu": _device_name(device),
            "gpu_seconds": gpu_s,
            "cores": runtime.get("cpu"),
            "memory_mib": runtime.get("memory_mib"),
            "model_seed": int(gnn_cfg["hpo"]["model_seed"]),
            "max_epochs": int(gnn_cfg["hpo"]["max_epochs"]),
            "gnn_version": GNN_VERSION,
            **info,
        }
    )
    tr._write_json(hpo_dir / TRIALS_FILE, records)
    tr._write_json(hpo_dir / BEST_PARAMS_FILE, params)  # last before the summary
    tr._write_json(hpo_dir / SUMMARY_FILE, summary)
    commit()
    _mlflow_study(hpo_dir.name, records, summary, log)
    return summary


def _device_name(device: str) -> str | None:
    if device != "cuda":
        return None
    import torch

    return str(torch.cuda.get_device_name(0))


def _graph_info(g, paths, features_dir) -> dict:
    """features_digest / data_version / spec_hash of the search: the graph's, or (every trial
    finished in an earlier call, whose records match the current inputs) the prepare marker's
    and the feature build summary's."""
    from aml.models.gnn import train as tr

    if g is not None:
        return {
            "features_digest": g.features_digest,
            "data_version": g.data_version,
            "spec_hash": g.spec_hash,
        }
    dv, fd = tr.current_inputs(paths, features_dir)
    build = tr._read_json(Path(features_dir) / SUMMARY_FILE) or {}
    return {"features_digest": fd, "data_version": dv, "spec_hash": build.get("spec_hash")}


def _result_from_summary(summary: dict) -> dict:
    return {
        "n_trials_done": len(summary.get("trials") or []),
        "best_params": summary.get("best_params"),
        "best_value": summary.get("best_value"),
        "best_trial": summary.get("best_trial"),
    }


def _clear_outputs(hpo_dir: Path) -> None:
    """Remove the search outputs and trial dirs of other inputs (keeps the lease and stamps)."""
    import shutil

    for name in (SUMMARY_FILE, BEST_PARAMS_FILE, TRIALS_FILE):
        (hpo_dir / name).unlink(missing_ok=True)
    for d in hpo_dir.glob("trial_*"):
        shutil.rmtree(d)


def _load_records(
    log_path: Path, version: str | None, digest: str | None, log
) -> list[dict] | None:
    """The trial log, or None when it was written for other prepared data or feature parts
    (then it is moved aside and the search starts over)."""
    records = read_jsonl(log_path)
    other_data = any(r.get("data_version", version) != version for r in records)
    other_parts = digest is not None and any(
        r.get("features_digest", digest) != digest for r in records
    )
    if other_data or other_parts:
        stale = log_path.with_name(f"{log_path.name}.other-data-{int(time.time())}")
        log_path.rename(stale)
        log(f"{log_path}: written for other inputs; moved to {stale.name}")
        return None
    numbers = [int(r["number"]) for r in records]
    if numbers != list(range(len(records))):
        raise ValueError(f"{log_path}: trial numbers {numbers} are not 0..{len(records) - 1}")
    for r in records:
        if r["state"] not in TRIAL_STATES:
            raise ValueError(f"{log_path}: unknown trial state {r['state']!r}")
    return records


def _intermediate(history: list[dict]) -> dict[int, float]:
    out: dict[int, float] = {}
    for r in history:
        v = r.get("metric")
        if v is not None and math.isfinite(float(v)):
            out[int(r["epoch"])] = float(v)
    return out


def _seconds(history: list[dict]) -> float:
    return float(sum(float(r.get("seconds") or 0.0) for r in history))


def _history_guard(history: list[dict]) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for r in history:
        for split, gv in (r.get("guard") or {}).items():
            out[split] = add_guard(out.get(split), gv)
    return out


def _distributions(gnn_cfg: dict) -> dict[str, Any]:
    import optuna

    space = gnn_cfg["hpo"]["space"]
    if set(space) != set(HPO_PARAMS):
        raise ValueError(f"hpo.space must be exactly {HPO_PARAMS}, got {sorted(space)}")
    return {
        k: optuna.distributions.FloatDistribution(
            float(space[k]["low"]), float(space[k]["high"]), log=bool(space[k].get("log", False))
        )
        for k in HPO_PARAMS
    }


def trial0_params(gnn_cfg: dict) -> dict[str, float]:
    """The enqueued trial 0: {lr, final_dropout, w_pos} of gnn.yaml's train section."""
    t = gnn_cfg["train"]
    return {k: float(t[k]) for k in HPO_PARAMS}


def trial_record(
    number: int,
    params: dict[str, float],
    state: str,
    value: float | None,
    intermediate_values: dict[int, float],
    best_epoch: int | None,
    seconds: float,
    guard: dict[str, int],
) -> dict[str, Any]:
    """One trial-log line: {number, params, state, value, intermediate_values (epoch -> value,
    JSON keys are strings), best_epoch, seconds, guard}."""
    if state not in TRIAL_STATES:
        raise ValueError(f"unknown trial state {state!r}; expected one of {TRIAL_STATES}")
    if state == "COMPLETE" and (value is None or not math.isfinite(float(value))):
        raise ValueError("a COMPLETE trial needs a finite value")
    return {
        "number": int(number),
        "params": {k: float(v) for k, v in params.items()},
        "state": state,
        "value": None if value is None else float(value),
        "intermediate_values": {str(int(k)): float(v) for k, v in intermediate_values.items()},
        "best_epoch": None if best_epoch is None else int(best_epoch),
        "seconds": float(seconds),
        "guard": json.loads(json.dumps(guard)),
    }


def rebuild_study(records: list[dict[str, Any]], gnn_cfg: dict, next_number: int) -> optuna.Study:
    """An in-memory study holding `records` (create_trial + add_trial; distributions from
    hpo.space), trial 0 enqueued if not yet recorded, `study.sampler = TPESampler(
    n_startup_trials=hpo.n_startup_trials, seed=hpo.sampler_seed_base + next_number)` and the
    MedianPruner: the next ask() is a pure function of (records, gnn_cfg, next_number)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    h = gnn_cfg["hpo"]
    dists = _distributions(gnn_cfg)
    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=int(h["pruner"]["n_startup_trials"]),
        n_warmup_steps=int(h["pruner"]["n_warmup_steps"]),
    )
    study = optuna.create_study(direction=OBJECTIVE_DIRECTION, pruner=pruner)
    ordered = sorted(records, key=lambda r: int(r["number"]))
    if [int(r["number"]) for r in ordered] != list(range(len(ordered))):
        raise ValueError("trial records must be numbered 0..n-1")
    for r in ordered:
        state = optuna.trial.TrialState[r["state"]]
        value = r.get("value") if r["state"] != "FAIL" else None
        study.add_trial(
            optuna.trial.create_trial(
                params={k: float(r["params"][k]) for k in HPO_PARAMS},
                distributions=dists,
                value=value,
                state=state,
                intermediate_values={
                    int(k): float(v) for k, v in (r.get("intermediate_values") or {}).items()
                },
            )
        )
    if not ordered:
        study.enqueue_trial(trial0_params(gnn_cfg))
    study.sampler = optuna.samplers.TPESampler(
        n_startup_trials=int(h["n_startup_trials"]),
        seed=int(h["sampler_seed_base"]) + int(next_number),
    )
    return study


def best_params(records: list[dict[str, Any]]) -> tuple[int, float, dict[str, float]] | None:
    """(number, value, params) of the best COMPLETE trial (max value; ties -> lower number), or
    None if no trial completed."""
    done = [
        r
        for r in records
        if r["state"] == "COMPLETE"
        and r.get("value") is not None
        and math.isfinite(float(r["value"]))
    ]
    if not done:
        return None
    best = min(done, key=lambda r: (-float(r["value"]), int(r["number"])))
    return (
        int(best["number"]),
        float(best["value"]),
        {k: float(best["params"][k]) for k in HPO_PARAMS},
    )


def _mlflow_study(hpo_key: str, records: list[dict], summary: dict, log) -> None:
    from aml.models.gnn import train as tr

    with tr._mlflow_run(EXPERIMENT, hpo_key, {"kind": HPO_KIND}, log) as run:
        if run is None:
            return
        from aml import tracking

        tr._mlflow_call(log, tracking.log_trials, records)
        tr._mlflow_call(log, tracking.log_params_flat, summary["best_params"], "best")
        tr._mlflow_call(
            log,
            tracking.log_metrics_flat,
            {
                "best_value": summary["best_value"],
                "best_trial": summary["best_trial"],
                "n_complete": summary["n_complete"],
                "n_pruned": summary["n_pruned"],
                "n_failed": summary["n_failed"],
                "edges_checked": summary["edges_checked"],
                "violations": summary["violations"],
            },
        )
