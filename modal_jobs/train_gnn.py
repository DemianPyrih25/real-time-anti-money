"""Train one GNN protocol's seed set on a Modal GPU, in chunks under a CPU driver (M3 spec §7, §12).

    uv run modal run --detach -m modal_jobs.train_gnn --protocol causal --final      # make gnn
    uv run modal run --detach -m modal_jobs.train_gnn --protocol lookahead --final [--seeds 0]
    uv run modal run --detach -m modal_jobs.train_gnn --protocol faithful --final
    uv run modal run --detach -m modal_jobs.train_gnn --protocol pna --final
    uv run modal run -m modal_jobs.train_gnn --plan-only                         # make gnn-plan
    make gnn-dev    # --protocol causal --seeds 0 --max-epochs 2 --dev (validation only)

The local entrypoint validates the arguments and configs, reads the Volume state (bench decision,
HPO best params, finished sets), runs the cost gate (billing report + conservative projection of
every remaining planned run, M3 spec §11.4) and submits ONE CPU driver call. The driver checks the
inputs, stamps the set directories, then calls the GPU worker in chunks with constant options per
set (gpu, cores, memory, the per-attempt timeout T of §12.1) until it returns done, failed,
stopped or busy, or the set's GPU wall exceeds wall_guard_factor x the projection (+ one chunk):
then it writes STOPPED.json and returns. It never retries a failed or busy result: re-run the
same command to resume (every epoch is checkpointed; score files are written once).

Non-dev runs are always `--final` (they score test once per seed); `--dev` runs score validation
only, under their own keys and the gnn_dev set kind. Test labels are never read here.

This module also holds the torch-free helpers the other GNN jobs (gnn_bench, hpo_gnn, evaluate
--with-gnn) import lazily: the Volume state, the gate wiring, the chunk budget and the driver loop.
"""

from __future__ import annotations

import json
import math
import time
import traceback
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

import modal

from modal_jobs.common import (
    DATA_MARKER,
    WORKER_MAX_RETRIES,
    billing_window_start_ms,
    cpu_job,
    data_key,
    data_paths,
    features_key,
    fetch_billing_report,
    fetch_billing_summary,
    gnn_keys,
    gnn_run_key,
    gnn_set_key,
    gpu_job,
    jsonable,
    load_all_configs,
    print_summary,
    read_volume_json,
    read_volume_jsonl,
    require_data,
    require_features_verified,
    require_stage_output,
    reset_if_other_data,
    vol,
)

APP_NAME = "aml-train-gnn"
app = modal.App(APP_NAME)

# The worker's static shape; every set overrides gpu / cores / memory / timeout per call with
# `with_options` from the bench decision (constant per set). The static timeout is the §12.1
# fallback (chunk_wall_s + 1800 for the default chunk wall of 7200 s).
STATIC_GPU = "L4"
STATIC_CORES = 8.0
STATIC_MEMORY_MIB = 32768
STATIC_WORKER_TIMEOUT = 7200 + 1800
WORKER_KW = gpu_job(
    gpu=STATIC_GPU,
    cpu=STATIC_CORES,
    memory_mib=STATIC_MEMORY_MIB,
    timeout=STATIC_WORKER_TIMEOUT,
    retries=True,  # transient errors only: deterministic ones return "failed" (§7.6)
)
DRIVER_KW = cpu_job(cpu=0.25, memory_mib=1024, timeout=43200)

# `make gnn-dev`: the gate's "dev" run (seed 0, 2 epochs, causal sampling, validation only).
DEV_PROTOCOL = "causal"
DEV_SEEDS = (0,)
DEV_MAX_EPOCHS = 2
# Protocols trained with HPO's best {lr, final_dropout, w_pos} (PNA and faithful use their own).
HPO_PROTOCOLS = ("causal", "lookahead")
# make targets that build each evaluated set (error messages).
MAKE_TARGETS = {
    "causal": "gnn",
    "lookahead": "gnn-lookahead",
    "pna": "gnn-pna",
    "faithful": "gnn-faithful",
}
# The driver stops a set whose worker returned "partial" twice in a row without progress.
NO_PROGRESS_LIMIT = 2
# A driver starts no chunk that could outlast its own timeout minus this margin (seconds); the
# margin also covers the startup + load of a last attempt that the unclean-start counter ends.
DRIVER_DEADLINE_MARGIN_S = 600
# A "busy" result: what the entrypoints tell the user.
BUSY_HINT = (
    "another worker call holds the set's writer lease. Check `modal app list` for a running "
    "worker; if none is running (e.g. its retries were used up), the lease goes stale 15 "
    "minutes after its last heartbeat: re-run the same command then"
)


def _log(msg: str) -> None:
    print(msg, flush=True)


# --- arguments ---------------------------------------------------------------------------------


def protocol_seeds(gnn_cfg: Mapping[str, Any], protocol: str) -> list[int]:
    """The set's seed list in gnn.yaml order (faithful: its single seed)."""
    p = gnn_cfg["protocols"][protocol]
    return [int(p["seed"])] if protocol == "faithful" else [int(s) for s in p["seeds"]]


def parse_seeds(text: str | int | None) -> list[int]:
    """'1,2' -> [1, 2]; '' or None -> [] (all seeds). The Makefile passes a comma list."""
    if text is None:
        return []
    out = []
    for part in str(text).replace(" ", "").split(","):
        if not part:
            continue
        if not part.isdigit():
            raise SystemExit(f"--seeds must be a comma-separated list of ints, got {text!r}")
        out.append(int(part))
    if len(set(out)) != len(out):
        raise SystemExit(f"--seeds has duplicates: {text!r}")
    return out


def check_args(
    gnn_cfg: Mapping[str, Any],
    protocol: str,
    seeds: str | int | None,
    *,
    final: bool,
    dev: bool,
    max_epochs: int | None,
) -> tuple[list[int], list[int]]:
    """Validate the submission (M3 spec §12.3). Returns (submitted seeds, the set's seeds).

    --dev and --max-epochs exclude --final; --max-epochs needs --dev; a non-dev run must be
    --final (a validation-only real run would finish its set without test scores, and score
    files are written once). --seeds must be a subset of the protocol's list; a dev set is the
    submitted seeds."""
    from aml.models.gnn import PROTOCOLS

    if protocol not in PROTOCOLS:
        raise SystemExit(f"--protocol must be one of {PROTOCOLS}, got {protocol!r}")
    if final and (dev or max_epochs):
        raise SystemExit("--dev and --max-epochs are mutually exclusive with --final")
    if max_epochs and not dev:
        raise SystemExit("--max-epochs belongs to --dev runs")
    if max_epochs is not None and max_epochs < 0:
        raise SystemExit(f"--max-epochs must be positive, got {max_epochs}")
    if dev and protocol != DEV_PROTOCOL:
        raise SystemExit(f"--dev runs the {DEV_PROTOCOL} protocol (the gate's dev run)")
    if not dev and not final:
        raise SystemExit(
            "a non-dev run must be --final (it scores test once per seed); use --dev for a "
            "validation-only check"
        )
    allowed = protocol_seeds(gnn_cfg, protocol)
    wanted = parse_seeds(seeds)
    extra = sorted(set(wanted) - set(allowed))
    if extra:
        raise SystemExit(f"--seeds {extra} are not in protocols.{protocol} seeds {allowed}")
    submitted = [s for s in allowed if s in wanted] if wanted else list(allowed)
    return submitted, (submitted if dev else allowed)


# --- keys and the Volume state ------------------------------------------------------------------


def set_plan(
    cfgs: Mapping[str, dict],
    protocol: str,
    params: Mapping[str, Any],
    set_seeds: Iterable[int],
    *,
    dev: bool = False,
    max_epochs: int | None = None,
) -> dict[str, Any]:
    """Run keys of every seed of a set, its set key and its directory (Volume path, posix)."""
    from aml.models.gnn import set_kind

    seeds = [int(s) for s in set_seeds]
    run_keys = {
        s: gnn_run_key(cfgs, protocol, s, dict(params), dev=dev, max_epochs=max_epochs)
        for s in seeds
    }
    key = gnn_set_key(cfgs, protocol, [run_keys[s] for s in seeds])
    kind = set_kind(protocol, dev=dev)
    return {
        "protocol": protocol,
        "dev": dev,
        "max_epochs": max_epochs,
        "params": dict(params),
        "seeds": seeds,
        "run_keys": run_keys,
        "set_kind": kind,
        "set_key": key,
        "set_dir": data_paths(cfgs["data"]).gnn_set_dir(kind, key).as_posix(),
    }


def _current(doc: Mapping[str, Any] | None, version: str | None) -> bool:
    """A finished stage's summary built from the prepared data on the Volume now."""
    if doc is None:
        return False
    found = doc.get("data_version")
    return version is None or found is None or found == version


def volume_state(
    cfgs: Mapping[str, dict],
    *,
    read: Callable[[Any], dict | None] | None = None,
    params_fn: Callable[..., dict] | None = None,
    read_lines: Callable[[Any], list[dict] | None] | None = None,
) -> dict[str, Any]:
    """Everything the GNN entrypoints need from the Volume (laptop: `vol.read_file`, no
    container): keys, the prepared data's version, the bench decision and summary, the HPO
    summary and best params, every protocol's set (keys, dir, summary), the dev set, `done` =
    the RUN_ORDER runs already finished and `progress` = what the started ones already trained
    (run_progress; both costplan.gate inputs). read_lines reads a JSON-lines file; without it
    (a test's fake `read`) progress is empty."""
    from aml.models.gnn import (
        BENCH_FILE,
        BENCH_KIND,
        BEST_PARAMS_FILE,
        DECISION_FILE,
        HPO_KIND,
        PROTOCOLS,
        SEED_SUMMARY_FILE,
        SUMMARY_FILE,
    )
    from aml.models.gnn.train import CALLS_FILE

    if read is None:
        read, read_lines = read_volume_json, read_lines or read_volume_jsonl
    if params_fn is None:
        from aml.models.gnn.train import effective_params as params_fn
    g = cfgs["gnn"]
    paths = data_paths(cfgs["data"])
    keys = {"data": data_key(cfgs), "features": features_key(cfgs), **gnn_keys(cfgs)}
    marker = read(paths.parquet_dir / DATA_MARKER) or {}
    version = marker.get("data_version")
    bench_dir = paths.gnn_set_dir(BENCH_KIND, keys["gnn_bench"])
    hpo_dir = paths.gnn_set_dir(HPO_KIND, keys["gnn_hpo"])
    st: dict[str, Any] = {
        "keys": keys,
        "data_version": version,
        "data_key_on_volume": marker.get("data_key"),
        "decision": read(bench_dir / DECISION_FILE),
        "bench_summary": read(bench_dir / SUMMARY_FILE),
        "bench_state": read(bench_dir / BENCH_FILE),  # every container, failed ones too
        "hpo_summary": read(hpo_dir / SUMMARY_FILE),
        "hpo_dir": hpo_dir.as_posix(),
        "bench_dir": bench_dir.as_posix(),
    }
    # best_params.json is written last before the HPO summary: trust it only once HPO finished.
    hpo_done = _current(st["hpo_summary"], version)
    st["best_params"] = read(hpo_dir / BEST_PARAMS_FILE) if hpo_done else None
    sets: dict[str, dict | None] = {}
    for protocol in PROTOCOLS:
        if protocol in HPO_PROTOCOLS and st["best_params"] is None:
            sets[protocol] = None  # its run keys need HPO's best params
            continue
        best = st["best_params"] if protocol in HPO_PROTOCOLS else None
        plan = set_plan(cfgs, protocol, params_fn(g, protocol, best), protocol_seeds(g, protocol))
        plan["summary"] = read(Path(plan["set_dir"]) / SUMMARY_FILE)
        sets[protocol] = plan
    st["sets"] = sets
    dev = set_plan(
        cfgs,
        DEV_PROTOCOL,
        params_fn(g, DEV_PROTOCOL, None),
        DEV_SEEDS,
        dev=True,
        max_epochs=DEV_MAX_EPOCHS,
    )
    dev["summary"] = read(Path(dev["set_dir"]) / SUMMARY_FILE)
    st["dev"] = dev
    la = sets["lookahead"]
    st["lookahead_s0_seed_summary"] = None
    if la is not None:
        first = la["run_keys"][la["seeds"][0]]
        st["lookahead_s0_seed_summary"] = read(paths.gnn_run_dir(first) / SEED_SUMMARY_FILE)

    def set_done(p: str) -> bool:
        return sets[p] is not None and _current(sets[p]["summary"], version)

    done = set()
    if _current(st["bench_summary"], version):
        done.add("bench")
    if _current(dev["summary"], version):
        done.add("dev")
    if hpo_done:
        done.add("hpo")
    for p in ("causal", "faithful", "pna"):
        if set_done(p):
            done.add(p)
    if set_done("lookahead"):
        done |= {"lookahead_s0", "lookahead_rest"}
    elif _current(st["lookahead_s0_seed_summary"], version):
        done.add("lookahead_s0")
    st["done"] = sorted(done)
    st["counts"] = (st["bench_summary"] or {}).get("counts")
    st["progress"] = run_progress(cfgs, st, read=read, read_lines=read_lines) if read_lines else {}
    # every worker call of the HPO, dev and set dirs (calls.jsonl): the spend floor and lag
    st["calls"] = {}
    if read_lines:
        dirs = [st["hpo_dir"], dev["set_dir"]] + [s["set_dir"] for s in sets.values() if s]
        for d in dirs:
            st["calls"][d] = read_lines(Path(d) / CALLS_FILE) or []
    return st


def run_progress(
    cfgs: Mapping[str, dict],
    state: Mapping[str, Any],
    *,
    read: Callable[[Any], dict | None],
    read_lines: Callable[[Any], list[dict] | None],
) -> dict[str, dict]:
    """What every started, unfinished RUN_ORDER run already trained: {run id: {"units_done",
    "epochs_done", "seconds"}}. A seed counts as a unit once its seed_summary.json exists, an HPO
    trial once it is in the trial log; the unfinished one counts its history.jsonl epochs.
    seconds = their recorded epoch seconds (a lower bound of the GPU wall they cost). Only run
    dirs stamped with the current prepared data and feature parts count (checkpoint.json), as
    the workers would resume them. The gate charges a started run only for what is left, since
    its spend so far is already in the measured spend (a re-submission is not charged twice)."""
    from aml.features.spec import FEATURES_DIGEST
    from aml.models.gnn import (
        FINGERPRINT_FILE,
        HISTORY_FILE,
        SEED_SUMMARY_FILE,
        SUMMARY_FILE,
        trial_dir_name,
    )

    paths = data_paths(cfgs["data"])
    version = state["data_version"]
    digest = (read(paths.features_dir(state["keys"]["features"]) / SUMMARY_FILE) or {}).get(
        FEATURES_DIGEST
    )

    def current(doc: Mapping[str, Any] | None) -> bool:
        return _current(doc, version) and (
            digest is None or (doc or {}).get("features_digest") in (None, digest)
        )

    def epochs_of(run_dir: Path) -> tuple[int, float]:
        hist = read_lines(run_dir / HISTORY_FILE) or []
        if not hist or not current(read(run_dir / FINGERPRINT_FILE)):
            return 0, 0.0
        return len(hist), float(sum(float(r.get("seconds") or 0.0) for r in hist))

    def seeds(run_keys: Mapping[int, str], which: Iterable[int]) -> dict:
        units, epochs, secs = 0, 0, 0.0
        for s in which:
            run_dir = paths.gnn_run_dir(run_keys[s])
            n, sec = epochs_of(run_dir)
            if not n:
                continue
            secs += sec
            if current(read(run_dir / SEED_SUMMARY_FILE)):
                units += 1
            else:
                epochs += n
        return {"units_done": units, "epochs_done": epochs, "seconds": secs}

    out: dict[str, dict] = {}
    done = set(state["done"])
    if "dev" not in done and state.get("dev"):
        d = state["dev"]
        out["dev"] = seeds(d["run_keys"], d["seeds"])
    if "hpo" not in done:
        hpo_key = state["keys"]["gnn_hpo"]
        recs = read_lines(paths.gnn_optuna_log(hpo_key)) or []
        if not all(current(r) for r in recs):
            recs = []  # run_hpo moves a log of other inputs aside and starts over
        n_trials = int(cfgs["gnn"]["hpo"]["n_trials"])
        recs = recs[:n_trials]
        n, sec = (0, 0.0)
        if len(recs) < n_trials:
            n, sec = epochs_of(Path(state["hpo_dir"]) / trial_dir_name(len(recs)))
        out["hpo"] = {
            "units_done": len(recs),
            "epochs_done": n,
            "seconds": sec + float(sum(float(r.get("seconds") or 0.0) for r in recs)),
        }
    sets = state.get("sets") or {}
    for run in ("causal", "faithful", "pna"):  # one RUN_ORDER id per set
        if run not in done and sets.get(run):
            out[run] = seeds(sets[run]["run_keys"], sets[run]["seeds"])
    la = sets.get("lookahead")
    if la:
        if "lookahead_s0" not in done:
            out["lookahead_s0"] = seeds(la["run_keys"], la["seeds"][:1])
        if "lookahead_rest" not in done:
            out["lookahead_rest"] = seeds(la["run_keys"], la["seeds"][1:])
    return {k: v for k, v in out.items() if v["units_done"] or v["epochs_done"]}


def check_decision(state: Mapping[str, Any], gnn_cfg: Mapping[str, Any]) -> dict:
    """HPO and training refuse to start unless gnn_bench finished and gnn.yaml's decided values
    equal decision.json (M3 spec §10.3)."""
    from aml.models.gnn import GnnStopError, decision_mismatches

    decision = state.get("decision")
    if decision is None or "bench" not in state["done"]:
        raise GnnStopError(
            f"no finished gnn_bench for the current configs ({state['bench_dir']}): run "
            "`make gnn-bench` and copy decision.json's values into configs/gnn.yaml"
        )
    bad = decision_mismatches(gnn_cfg, decision)
    if bad:
        raise GnnStopError(
            "configs/gnn.yaml's decided values differ from the bench's decision.json:\n  "
            + "\n  ".join(bad),
            {"mismatches": bad},
        )
    if decision.get("ask_user"):
        _log(f"warning: the bench decision asked the user: {decision['ask_user']}")
    return decision


# --- the cost gate (M3 spec §11.4) --------------------------------------------------------------


def _float(v: Any) -> float | None:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _shapes(state: Mapping[str, Any], gnn_cfg: Mapping[str, Any]) -> tuple[dict, dict]:
    """(the runs' shape, the faithful run's shape): runtime's, the faithful one from
    decision.faithful (its measured GPU, cores, memory) where recorded."""
    rt = gnn_cfg["runtime"]
    shape = {"gpu": rt["gpu"], "cores": rt["cpu"], "memory_mib": rt["memory_mib"]}
    fd = (state.get("decision") or {}).get("faithful") or {}
    fshape = {
        "gpu": fd.get("gpu") or shape["gpu"],
        "cores": fd.get("cores") or shape["cores"],
        "memory_mib": fd.get("memory_mib") or shape["memory_mib"],
    }
    return shape, fshape


def _stage_dirs(state: Mapping[str, Any]) -> list[tuple[str, dict | None, str | None, tuple]]:
    """(name, summary, dir, RUN_ORDER ids) of the HPO, dev and protocol set dirs."""
    out = [
        ("hpo", state.get("hpo_summary"), state.get("hpo_dir"), ("hpo",)),
        (
            "dev",
            (state.get("dev") or {}).get("summary"),
            (state.get("dev") or {}).get("set_dir"),
            ("dev",),
        ),
    ]
    for p, s in (state.get("sets") or {}).items():
        runs = ("lookahead_s0", "lookahead_rest") if p == "lookahead" else (p,)
        out.append((p, (s or {}).get("summary"), (s or {}).get("set_dir"), runs))
    return out


def _bench_containers(state: Mapping[str, Any]) -> list[dict]:
    """The bench containers that ran: bench.json's (failed ones too), else the summary's."""
    recorded = list(((state.get("bench_state") or {}).get("containers") or {}).values())
    return recorded or list((state.get("bench_summary") or {}).get("containers") or [])


def spend_records(state: Mapping[str, Any], gnn_cfg: Mapping[str, Any]) -> list[dict]:
    """{gpu, cores, memory_mib, gpu_seconds} of every GNN stage on the Volume: the fallback (and
    lag-proof floor) of the measured M3 spend. Bench: every container bench.json records (a
    failed bench has no summary). A dir with a summary: its gpu_seconds; without one (started,
    stopped or failed): max(its runs' recorded epoch seconds, its calls.jsonl GPU seconds), so
    load, scoring and failed calls count too. Faithful at its decided shape."""
    shape, fshape = _shapes(state, gnn_cfg)
    out = []
    for c in _bench_containers(state):
        if c.get("seconds") is not None:
            out.append(
                {
                    "gpu": c["gpu"],
                    "cores": c["cores"],
                    "memory_mib": c["memory_mib"],
                    "gpu_seconds": c["seconds"],
                }
            )
    progress = state.get("progress") or {}
    calls = state.get("calls") or {}
    for name, summary, d, runs in _stage_dirs(state):
        secs = _float((summary or {}).get("gpu_seconds"))
        if secs is None:
            epochs = sum(float((progress.get(r) or {}).get("seconds") or 0.0) for r in runs)
            logged = sum(
                float(c.get("elapsed_s") or 0.0)
                for c in calls.get(d) or []
                if c.get("device") == "cuda"
            )
            secs = max(epochs, logged) if (epochs or logged) else None
        if secs is None:
            continue
        out.append({**(fshape if name == "faithful" else shape), "gpu_seconds": secs})
    return out


def call_records(state: Mapping[str, Any], gnn_cfg: Mapping[str, Any]) -> list[dict]:
    """Every timestamped GPU worker call (calls.jsonl lines with ended_at) and bench container
    (bench.json, ended_at) as {gpu, cores, memory_mib, elapsed_s, ended_at}: what the billing
    report's last full hour does not show yet (costplan.lag_usd)."""
    shape, fshape = _shapes(state, gnn_cfg)
    out = []
    for c in _bench_containers(state):
        if c.get("ended_at") is not None and c.get("seconds") is not None:
            out.append(
                {
                    "gpu": c["gpu"],
                    "cores": c["cores"],
                    "memory_mib": c["memory_mib"],
                    "elapsed_s": c["seconds"],
                    "ended_at": c["ended_at"],
                }
            )
    calls = state.get("calls") or {}
    for name, _summary, d, _runs in _stage_dirs(state):
        sh = fshape if name == "faithful" else shape
        for c in calls.get(d) or []:
            if c.get("device") == "cuda" and c.get("ended_at") is not None:
                out.append({**sh, "elapsed_s": c.get("elapsed_s"), "ended_at": c["ended_at"]})
    return out


def measured_spend(
    state: Mapping[str, Any],
    gnn_cfg: Mapping[str, Any],
    *,
    fetch_report: Callable | None = None,
    fetch_summary: Callable | None = None,
) -> dict[str, Any]:
    """The gate's spend (costplan.measured_spend): max(billing report over the M3 apps + the
    recorded GPU calls after its last full hour, the Volume floor) + budget.dev_allowance_usd;
    the billing report lists full hours only, so a run that just finished may be missing from
    it. Plus the workspace's metered cost for the budget pre-check (`modal billing summary`;
    None if unavailable, and then the gate refuses)."""
    from aml.models.gnn import costplan

    fetch_report = fetch_report or fetch_billing_report
    fetch_summary = fetch_summary or fetch_billing_summary
    report, err = fetch_report(billing_window_start_ms())
    spent = costplan.measured_spend(
        report if err is None else None,
        spend_records(state, gnn_cfg),
        gnn_cfg,
        report_error=err,
        calls=call_records(state, gnn_cfg),
    )
    if spent.get("warning"):
        _log(f"warning: {spent['warning']}")
    summary, serr = fetch_summary()
    metered = _float((summary or {}).get("metered_cost")) if serr is None else None
    if metered is None:
        _log(f"warning: the workspace metered cost is unavailable ({serr})")
    return {
        **spent,
        "spent_usd": spent["usd"],
        "billing_error": err,
        "metered_usd": metered,
        "metered_error": serr,
    }


def plan_rows(gnn_cfg: Mapping[str, Any], state: Mapping[str, Any]) -> list[dict]:
    """costplan.plan_runs with the bench's measured numbers once decision.json exists, the
    §11.1 planning constants before."""
    from aml.models.gnn import costplan

    decision = state.get("decision")
    if decision is None:
        return costplan.plan_runs(gnn_cfg)
    return costplan.plan_runs(gnn_cfg, decision=decision, counts=state.get("counts"))


def run_gate(
    cfgs: Mapping[str, dict],
    state: Mapping[str, Any],
    jobs: Iterable[str],
    *,
    spent: Mapping[str, Any] | None = None,
    fetch_report: Callable | None = None,
    fetch_summary: Callable | None = None,
) -> dict[str, Any]:
    """costplan.gate for the RUN_ORDER job(s) of one submission (M3 spec §11.4): the measured
    spend + the conservative $ of every planned run not done, the cut order, the faithful epoch
    cap, the workspace pre-check. The gate refuses a job it cuts or whose cut / cap gnn.yaml has
    not applied yet; nothing is submitted unless "allowed"."""
    from aml.models.gnn import costplan

    g = cfgs["gnn"]
    jobs = list(jobs)
    plan = plan_rows(g, state)
    if spent is None:
        spent = measured_spend(state, g, fetch_report=fetch_report, fetch_summary=fetch_summary)
    result = costplan.gate(
        plan,
        gnn_cfg=g,
        spent_usd=spent["spent_usd"],
        done=list(state["done"]),
        job=jobs[0] if len(jobs) == 1 else jobs,
        metered_usd=spent["metered_usd"],
        progress={
            run: {k: p[k] for k in ("units_done", "epochs_done")}
            for run, p in (state.get("progress") or {}).items()
        },
    )
    allowed = bool(result["allowed"])
    return {
        "jobs": jobs,
        "result": result,
        "allowed": allowed,
        "reasons": [] if allowed else [result.get("refuse_reason") or "refused by the cost gate"],
        "spent": dict(spent),
        "plan": plan,
    }


def print_gate(gate: Mapping[str, Any]) -> None:
    from aml.models.gnn import costplan

    sp = gate["spent"]
    billing = "n/a" if sp.get("billing_usd") is None else f"${sp['billing_usd']:.2f}"
    lag = float(sp.get("lag_usd") or 0.0)
    _log(
        f"== M3 spend: billing {billing} (+ ${lag:.2f} after its last full hour), Volume floor "
        f"${sp['volume_usd']:.2f}, dev allowance ${sp['dev_allowance_usd']:.2f} -> "
        f"${sp['spent_usd']:.2f}"
    )
    _log(costplan.render_gate(gate["result"]))


def next_job(state: Mapping[str, Any]) -> str | None:
    """The first planned run in RUN_ORDER that has not finished (the --plan-only default)."""
    from aml.models.gnn.costplan import RUN_ORDER

    return next((r for r in RUN_ORDER if r not in state["done"]), None)


def budget_rows(gate: Mapping[str, Any], jobs: Iterable[str]) -> list[dict]:
    """The plan rows a submission's chunk budget is sized from: the gate's rows of its jobs that
    are still planned (a started run: what is left, as the gate charged it, so the wall guard of
    a re-submission matches what the gate allowed), else the plan's (e.g. look-ahead seed 0
    trained but its set not yet scored)."""
    jobs = set(jobs)
    rows = [
        r
        for r in (gate.get("result") or {}).get("rows") or []
        if r.get("run") in jobs and r.get("status") == "planned"
    ]
    return rows or [r for r in gate["plan"] if r.get("run") in jobs]


def full_attempts() -> int:
    """The attempts of one worker call that can each run a full per-attempt timeout T: Modal
    retries a timed-out input (WORKER_MAX_RETRIES), and the unclean-start counter
    (train.MAX_STARTS) ends the start after the second unclean one right after its load."""
    from aml.models.gnn.train import MAX_STARTS

    return min(WORKER_MAX_RETRIES + 1, MAX_STARTS - 1)


def chunk_budget(
    rows: Iterable[Mapping[str, Any]], gnn_cfg: Mapping[str, Any], *, scale: float = 1.0
) -> dict[str, Any]:
    """Wall and timeout of one set (M3 spec §12.1), from the plan rows of its job(s): projected =
    scale x Σ conservative run seconds; attempt_wall = min(chunk_wall_s, factor x projected);
    T = max(1800, ceil(attempt_wall + max(600, 3 x conservative epoch s))); the driver stops
    once the set's GPU wall reaches factor x projected (so at most one more chunk), and starts
    no chunk that could pass set_wall_limit_s = factor x projected + chunk_wall_s (the bound the
    gate's budget pre-check charged: costplan.job_bound_usd). full_attempts = the attempts of
    one call that can each run a full T (Modal retries a timed-out input; the unclean-start
    counter ends the next one right after its load)."""
    from aml.models.gnn import costplan

    rows = list(rows)
    if not rows:
        raise SystemExit("the cost plan has no row for this job: cannot size its chunks")
    rt = gnn_cfg["runtime"]
    try:
        projected = scale * sum(float(r["run_s"]["conservative"]) for r in rows)
        epoch_cons = max(float(r["epoch_s"]["conservative"]) for r in rows)
    except (KeyError, TypeError) as e:
        raise SystemExit(f"the cost plan rows lack run_s / epoch_s: {e!r}") from e
    factor = float(rt["wall_guard_factor"])
    attempt = float(costplan.attempt_wall_s(rt["chunk_wall_s"], projected, factor))
    timeout = int(costplan.attempt_timeout_s(attempt, epoch_cons))
    startup = max(float((r.get("startup_s") or {}).get("conservative") or 0.0) for r in rows)
    return {
        "projected_s": projected,
        "epoch_s_cons": epoch_cons,
        "attempt_wall_s": attempt,
        "timeout_s": timeout,
        "max_set_wall_s": factor * projected,
        "set_wall_limit_s": factor * projected + float(rt["chunk_wall_s"]),
        "startup_s": startup,
        "full_attempts": full_attempts(),
        "wall_guard_factor": factor,
    }


def worker_options(
    gnn_cfg: Mapping[str, Any], decision: Mapping[str, Any], *, protocol: str, timeout: int
) -> dict[str, Any]:
    """`with_options` of a set's GPU worker: the decided GPU, cores (request = limit), memory
    (request, request + 8 GiB), timeout T; faithful: decision.faithful's GPU, cores and memory
    (the shape its bench cell was measured on). The thread caps follow the cores (env replaces
    the static env only when the cores differ from it)."""
    rt = gnn_cfg["runtime"]
    cores, mem, gpu = float(rt["cpu"]), int(rt["memory_mib"]), rt["gpu"]
    if protocol == "faithful":
        fd = decision.get("faithful") or {}
        gpu = fd.get("gpu")
        cores = float(fd.get("cores") or cores)
        mem = int(fd.get("memory_mib") or mem)
    opts: dict[str, Any] = {
        "gpu": gpu or rt["gpu"],
        "cpu": (cores, cores),
        "memory": (mem, mem + 8192),
        "timeout": int(timeout),
    }
    if cores != STATIC_CORES:
        opts["env"] = gpu_job(cpu=cores)["env"]
    return opts


def set_runtime(
    gnn_cfg: Mapping[str, Any], decision: Mapping[str, Any], *, protocol: str, gpu: str
) -> dict[str, Any]:
    """The worker's runtime section: gnn.yaml runtime with the call's GPU; faithful: its
    measured cores, loader workers and memory from decision.faithful (worker_options' shape)."""
    rt = {**gnn_cfg["runtime"], "gpu": gpu}
    if protocol == "faithful":
        fd = decision.get("faithful") or {}
        pairs = (("cpu", "cores"), ("num_workers", "num_workers"), ("memory_mib", "memory_mib"))
        for key, src in pairs:
            if fd.get(src) is not None:
                rt[key] = fd[src]
    return rt


# --- the driver loop (CPU container; torch-free) ------------------------------------------------


def _progress(res: Mapping[str, Any]) -> str | None:
    sig = {k: res.get(k) for k in ("next", "seeds_trained", "seeds_scored", "n_trials_done")}
    if all(v is None for v in sig.values()):
        return None
    return json.dumps(sig, sort_keys=True, default=str)


def drive_chunks(
    call: Callable[[float], Mapping[str, Any]],
    *,
    budget: Mapping[str, Any],
    set_dir: Path,
    commit: Callable[[], None] | None = None,
    log: Callable[[str], None] = _log,
    clock: Callable[[], float] = time.monotonic,
    deadline_s: float | None = None,
) -> dict[str, Any]:
    """Call the GPU worker (`call(attempt_wall_s)`) until it returns done / failed / stopped /
    busy. Stops with STOPPED.json in `set_dir` when the set's GPU wall (measured here, queueing
    included) reaches budget.max_set_wall_s; when another chunk (attempt_wall_s + startup_s)
    could pass budget.set_wall_limit_s (what the gate's pre-check charged); when two consecutive
    "partial" results show no progress; or when another chunk could outlast the driver's own
    deadline_s (its Modal timeout minus a margin): one call can take full_attempts x the
    per-attempt timeout budget.timeout_s (Modal retries a timed-out input) +
    DRIVER_DEADLINE_MARGIN_S, and the driver returns cleanly instead of being killed mid-chunk.
    A raised error (retries exhausted, a timeout) ends the loop as "error"; nothing is retried
    here."""
    from aml.io import write_json_atomic
    from aml.models.gnn import DRIVER_STOP_STATUSES, STOPPED_FILE

    commit = commit or (lambda: None)
    used, chunks, last_sig, same = 0.0, [], None, 0

    def stop(reason: str, last: Mapping[str, Any] | None) -> dict[str, Any]:
        doc = {
            "reason": reason,
            "used_s": used,
            "max_set_wall_s": budget["max_set_wall_s"],
            "chunks": chunks,
            "last": dict(last or {}),
        }
        write_json_atomic(jsonable(doc), Path(set_dir) / STOPPED_FILE)
        commit()
        log(f"STOPPED ({reason}) after {len(chunks)} chunks, {used:.0f} s of GPU wall")
        return {"status": "stopped", "reason": reason, "chunks": chunks, "used_s": used}

    last: Mapping[str, Any] | None = None
    if budget.get("timeout_s"):
        attempts = int(budget.get("full_attempts") or full_attempts())
        chunk_max = attempts * float(budget["timeout_s"]) + DRIVER_DEADLINE_MARGIN_S
    else:
        chunk_max = float(budget["attempt_wall_s"])
    limit = budget.get("set_wall_limit_s")
    next_chunk = float(budget["attempt_wall_s"]) + float(budget.get("startup_s") or 0.0)
    while True:
        if used >= budget["max_set_wall_s"]:
            return stop("wall_guard", last)
        if limit is not None and chunks and used + next_chunk > float(limit):
            return stop("wall_limit", last)
        if deadline_s is not None and chunks and used + chunk_max > deadline_s:
            return stop("driver_deadline", last)
        t0 = clock()
        try:
            res = call(float(budget["attempt_wall_s"]))
        except Exception as e:
            used += clock() - t0
            log(f"worker call failed: {type(e).__name__}: {e}")
            return {
                "status": "error",
                "error": f"{type(e).__name__}: {e}",
                "chunks": chunks,
                "used_s": used,
            }
        dt = clock() - t0
        used += dt
        last = res
        status = res.get("status")
        chunks.append({"status": status, "seconds": dt, "next": res.get("next")})
        log(f"chunk {len(chunks)}: {status} (set GPU wall {used:.0f} s)")
        if status in DRIVER_STOP_STATUSES:
            return {**res, "chunks": chunks, "used_s": used}
        if status != "partial":
            return {
                "status": "error",
                "error": f"unknown worker status {status!r}",
                "chunks": chunks,
                "used_s": used,
            }
        sig = _progress(res)
        same = same + 1 if sig is not None and sig == last_sig else 0
        last_sig = sig
        if same >= NO_PROGRESS_LIMIT - 1:
            return stop("no_progress", res)


def require_inputs(paths: Any, keys: Mapping[str, str]) -> str | None:
    """The job preconditions (M3 spec §4.1): prepared data of this config, the feature build,
    and its passed DuckDB oracle. Returns the prepared data's version."""
    version = require_data(paths, keys["data"])
    features_dir = paths.features_dir(keys["features"])
    require_stage_output(features_dir, "features", "features", data_version=version)
    require_features_verified(features_dir)
    return version


def prepare_set_dirs(dirs: Iterable[Path], version: str | None) -> None:
    """Stamp each stage dir with the data version (emptied if built from other data) and remove
    STOPPED.json: a new submission resumes a stopped set."""
    from aml.models.gnn import STOPPED_FILE

    for d in dirs:
        if reset_if_other_data(Path(d), version):
            _log(f"{d}: built from other prepared data; starting over")
        stopped = Path(d) / STOPPED_FILE
        if stopped.exists():
            stopped.unlink()


def _failed(e: BaseException) -> dict[str, Any]:
    tb = traceback.format_exc()
    _log(tb)
    return {"status": "failed", "error": f"{type(e).__name__}: {e}", "traceback": tb[-4000:]}


# --- evaluate --with-gnn ------------------------------------------------------------------------


def gnn_eval_models(
    cfgs: Mapping[str, dict],
    *,
    read: Callable[[Any], dict | None] | None = None,
    params_fn: Callable[..., dict] | None = None,
) -> dict[str, Any]:
    """The GNN sets `evaluate --with-gnn` evaluates (M3 spec §13.1): gnn_causal (required), the
    two look-ahead models (both or neither), gnn_pna if its set finished, gnn_faithful for its
    own section. -> {"models": {name: {kind, key, make}} in report order, "faithful": {kind, key,
    make} | None, "model_views", "report_cfg"}."""
    from aml.models.gnn import (
        FINGERPRINT_FILE,
        LOOKAHEAD_D10_MODEL,
        PRIMARY_ONLY_MODELS,
        SET_KINDS,
        SUMMARY_FILE,
    )

    read = read or read_volume_json
    st = volume_state(cfgs, read=read, params_fn=params_fn)
    version = st["data_version"]
    paths = data_paths(cfgs["data"])

    def finished(p: str) -> dict | None:
        s = st["sets"].get(p)
        if s is None or not _current(s["summary"], version):
            if s is not None and any(
                read(paths.gnn_run_dir(k) / FINGERPRINT_FILE) is not None
                for k in s["run_keys"].values()
            ):
                # Started but not assembled (still running, stopped, or a look-ahead seed cut
                # after seed 0 trained): evaluating now would touch test without its section.
                raise SystemExit(
                    f"{SET_KINDS[p]} {s['set_key']} has trained runs but no finished set: "
                    f"run `make {MAKE_TARGETS[p]}` until the set is assembled, then evaluate"
                )
            return None
        if not s["summary"].get("final"):
            raise SystemExit(f"{SET_KINDS[p]} {s['set_key']} was not trained with --final")
        return s

    causal = finished("causal")
    if causal is None:
        raise SystemExit(
            "gnn_causal: no finished --final set for the current configs: run `make gnn-hpo` "
            "and `make gnn` first"
        )
    models = {
        "gnn_causal": {"kind": SET_KINDS["causal"], "key": causal["set_key"], "make": "gnn"},
    }
    la = finished("lookahead")
    if la is not None:
        d10_dir = paths.gnn_set_dir(LOOKAHEAD_D10_MODEL, la["set_key"])
        if not _current(read(d10_dir / SUMMARY_FILE), version):
            raise SystemExit(
                f"gnn_lookahead {la['set_key']} finished but {LOOKAHEAD_D10_MODEL} has no "
                "summary: the two look-ahead models are evaluated both or neither"
            )
        make = MAKE_TARGETS["lookahead"]
        models["gnn_lookahead"] = {
            "kind": SET_KINDS["lookahead"],
            "key": la["set_key"],
            "make": make,
        }
        models[LOOKAHEAD_D10_MODEL] = {
            "kind": LOOKAHEAD_D10_MODEL,
            "key": la["set_key"],
            "make": make,
        }
    pna = finished("pna")
    if pna is not None:
        models["gnn_pna"] = {"kind": SET_KINDS["pna"], "key": pna["set_key"], "make": "gnn-pna"}
    fa = finished("faithful")
    faithful = (
        {"kind": SET_KINDS["faithful"], "key": fa["set_key"], "make": "gnn-faithful"}
        if fa is not None
        else None
    )
    return {
        "models": models,
        "faithful": faithful,
        "model_views": {m: ["primary"] for m in PRIMARY_ONLY_MODELS if m in models},
        "report_cfg": cfgs["gnn"]["report"],
    }


# --- Modal functions ----------------------------------------------------------------------------


@app.function(**WORKER_KW)
def train_gpu(spec: dict) -> dict:
    """One chunk: train.run_set within spec["budget_s"] of wall. Deterministic errors return
    "failed" (FAILED.json in the run dir); transient ones are re-raised so Modal retries the input
    and the run resumes from its last checkpoint. Only `Exception` is caught: Modal's preemption
    interrupt must propagate."""
    try:
        from aml.models.gnn.train import is_transient, run_set
    except Exception as e:  # a broken image or import: deterministic, never retried
        return _failed(e)
    paths = data_paths(spec["data_cfg"])
    try:
        out = run_set(
            paths,
            paths.features_dir(spec["keys"]["features"]),
            paths.gnn_set_dir(spec["set_kind"], spec["set_key"]),
            spec["gnn_cfg"],
            data_cfg=spec["data_cfg"],
            protocol=spec["protocol"],
            seeds=[int(s) for s in spec["seeds"]],
            run_keys={int(s): k for s, k in spec["run_keys"].items()},
            params=spec["params"],
            final=bool(spec["final"]),
            device="cuda",
            runtime=spec["runtime"],
            budget_s=float(spec["budget_s"]),
            test_bounds=tuple(spec["test_bounds"]),
            dev=bool(spec["dev"]),
            max_epochs=spec["max_epochs"],
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
def train_driver(spec: dict) -> dict:
    """Checks the inputs, stamps the set dirs, then runs the worker in chunks (drive_chunks)."""
    from aml.models.gnn import LOOKAHEAD_D10_MODEL
    from aml.models.gnn.train import finished_set  # torch-free

    paths = data_paths(spec["data_cfg"])
    version = require_inputs(paths, spec["keys"])
    set_dir = paths.gnn_set_dir(spec["set_kind"], spec["set_key"])
    dirs = [set_dir]
    if spec["protocol"] == "lookahead" and not spec["dev"]:
        dirs.append(paths.gnn_set_dir(LOOKAHEAD_D10_MODEL, spec["set_key"]))
    prepare_set_dirs(dirs, version)
    vol.commit()
    # the worker's own rule: current data and feature parts, final if this call is
    features_dir = paths.features_dir(spec["keys"]["features"])
    stored = finished_set(paths, features_dir, set_dir, spec["protocol"], bool(spec["final"]))
    if stored is not None:
        return {"status": "done", "summary": stored, "chunks": []}
    worker = train_gpu.with_options(**spec["options"])
    _log(f"{spec['protocol']} set {spec['set_key']}: seeds {spec['seeds']}, {spec['budget']}")
    res = drive_chunks(
        lambda budget_s: worker.remote({**spec, "budget_s": budget_s}),
        budget=spec["budget"],
        set_dir=set_dir,
        commit=vol.commit,
        deadline_s=DRIVER_KW["timeout"] - DRIVER_DEADLINE_MARGIN_S,
    )
    return jsonable(res)


def jobs_for(
    gnn_cfg: Mapping[str, Any], protocol: str, submitted: list[int], *, dev: bool
) -> list[str]:
    """The RUN_ORDER ids a submission covers (look-ahead: its first seed and/or the rest)."""
    if dev:
        return ["dev"]
    if protocol != "lookahead":
        return [protocol]
    first = protocol_seeds(gnn_cfg, "lookahead")[0]
    jobs = []
    if first in submitted:
        jobs.append("lookahead_s0")
    if any(s != first for s in submitted):
        jobs.append("lookahead_rest")
    return jobs


def _check_report_hash(state: Mapping[str, Any], gnn_cfg: Mapping[str, Any]) -> None:
    """No --final run under another `report` section than the --final sets already recorded."""
    from aml.models.gnn import GnnStopError, report_hash

    current = report_hash(gnn_cfg)
    recorded = {
        p: s["summary"].get("report_hash")
        for p, s in (state.get("sets") or {}).items()
        if s and s.get("summary") and s["summary"].get("final")
    }
    bad = {p: h for p, h in recorded.items() if h != current}
    if bad:
        raise GnnStopError(
            f"configs/gnn.yaml `report:` hashes to {current!r}, but finished --final sets "
            f"recorded {bad}: restore the pre-registered section"
        )


@app.local_entrypoint()
def main(
    protocol: str = "causal",
    seeds: str = "",
    final: bool = False,
    dev: bool = False,
    max_epochs: int = 0,
    plan_only: bool = False,
) -> None:
    from aml.models.gnn import GnnStopError, check_gnn_cfg
    from aml.models.gnn.train import effective_params

    cfgs = load_all_configs()
    g = check_gnn_cfg(cfgs["gnn"])
    state = volume_state(cfgs)
    if plan_only:
        job = next_job(state)
        if job is None:
            _log("every planned M3 run has finished")
            return
        print_gate(run_gate(cfgs, state, [job]))
        return
    submitted, set_seeds = check_args(
        g, protocol, seeds, final=final, dev=dev, max_epochs=max_epochs or None
    )
    decision = check_decision(state, g)
    best = None
    if protocol in HPO_PROTOCOLS and not dev:
        if "hpo" not in state["done"] or state["best_params"] is None:
            raise SystemExit(
                f"--protocol {protocol} needs HPO's best_params.json: run `make gnn-hpo`"
            )
        best = state["best_params"]
    params = effective_params(g, protocol, best)
    plan = set_plan(cfgs, protocol, params, set_seeds, dev=dev, max_epochs=max_epochs or None)
    known = (state["dev"] if dev else state["sets"].get(protocol)) or {}
    if known.get("set_key") == plan["set_key"] and _current(
        known.get("summary"), state["data_version"]
    ):
        print_summary(f"{APP_NAME} {plan['set_key']} (already finished)", known["summary"])
        return
    if final:
        _check_report_hash(state, g)
    jobs = jobs_for(g, protocol, submitted, dev=dev)
    gate = run_gate(cfgs, state, jobs)
    print_gate(gate)
    if not gate["allowed"]:
        raise GnnStopError("the cost gate refused: " + "; ".join(gate["reasons"]))
    # A seed subset of a causal / PNA set runs that share of the set's projection.
    scale = 1.0 if protocol == "lookahead" or dev else len(submitted) / len(set_seeds)
    budget = chunk_budget(budget_rows(gate, jobs), g, scale=scale)
    options = worker_options(g, decision, protocol=protocol, timeout=budget["timeout_s"])
    test_bounds = ["end"]
    if protocol == "lookahead":
        test_bounds = list(g["protocols"]["lookahead"]["test_bounds"])
    spec = {
        "data_cfg": cfgs["data"],
        "gnn_cfg": g,
        "keys": state["keys"],
        "protocol": protocol,
        "seeds": submitted,
        "run_keys": plan["run_keys"],
        "params": params,
        "final": bool(final),
        "dev": bool(dev),
        "max_epochs": max_epochs or None,
        "set_kind": plan["set_kind"],
        "set_key": plan["set_key"],
        "test_bounds": test_bounds,
        "runtime": set_runtime(g, decision, protocol=protocol, gpu=options["gpu"]),
        "budget": budget,
        "options": options,
    }
    _log(
        f"== submitting {protocol} seeds {submitted} -> {plan['set_kind']}/{plan['set_key']} on "
        f"{options['gpu']} ({budget['attempt_wall_s']:.0f} s per chunk, timeout "
        f"{budget['timeout_s']} s, set wall guard {budget['max_set_wall_s']:.0f} s)"
    )
    res = train_driver.remote(spec)
    print_summary(f"{APP_NAME} {plan['set_key']}", res)
    if isinstance(res.get("summary"), dict):
        print_summary("set summary", res["summary"])
    status = res.get("status")
    if status == "stopped":
        raise GnnStopError(f"the driver stopped the set ({res.get('reason')}): re-run to resume")
    if status == "busy":
        raise SystemExit(f"{APP_NAME}: busy ({res.get('error')}): {BUSY_HINT}")
    if status != "done":
        raise SystemExit(f"{APP_NAME}: {status}: {res.get('error') or res}")
