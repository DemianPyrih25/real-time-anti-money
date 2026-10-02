"""Feature-engine replay: bench, full replay, or the DuckDB oracle check (PLAN.md §6 M2).

    uv run modal run --detach -m modal_jobs.build_features --mode bench     # make features-bench
    uv run modal run --detach -m modal_jobs.build_features --mode full      # make features
    uv run modal run --detach -m modal_jobs.build_features --mode verify    # make features-verify

Output: /data/features/<dataset>/<features_key>/ (parts, snapshots, progress.jsonl, summary.json
written last; bench/bench.json, with bench/bench_pass_a.json saved before the slow traced pass;
verify/verify.json) and /data/reports/engine_bench.md. Each replayed day prints one progress
line (`modal app logs`). The replay is pure Python on 1 core (bench, full); verify runs DuckDB on
4 cores. There is no mid-replay resume: a failed or preempted replay is simply re-run.
"""

from __future__ import annotations

import modal

from modal_jobs.common import (
    cpu_job,
    data_key,
    data_paths,
    features_key,
    jsonable,
    load_all_configs,
    print_summary,
    require_data,
    require_stage_output,
    reset_if_other_data,
    rules_key,
    vol,
)

APP_NAME = "aml-build-features"
REPLAY_CPU = 1.0  # the engine is single-threaded pure Python
VERIFY_CPU = 4.0
REPLAY_MEMORY_MIB = 6144
VERIFY_MEMORY_MIB = 16384
# Per-attempt timeouts (COST_NOTES): bench 2 h. Full: max(1 h, 2 x the bench projection); the
# real-data bench projected 6.9 min for the replay + restart check, so 1 h. Verify: 1 h.
BENCH_TIMEOUT_S = 2 * 3600
FULL_TIMEOUT_S = 3600
VERIFY_TIMEOUT_S = 3600
MODES = ("bench", "full", "verify")
# Summary entries printed locally (the full documents are on the Volume and in MLflow).
_PRINT = (
    "run_key",
    "out_dir",
    "rows",
    "us_per_event",
    "memory",
    "restart_check",
    "gate",
    "bench_gate",
    "projection",
    "n_mismatches",
    "m1_closed_pairs",
    "features_digest",
)

app = modal.App(APP_NAME)


def _no_lists(doc: dict) -> dict:
    """The summary without list values (per-day records stay in summary.json / progress.jsonl)."""
    return {
        k: _no_lists(v) if isinstance(v, dict) else v
        for k, v in doc.items()
        if not isinstance(v, list)
    }


def _run(mode: str, cfgs: dict, keys: dict, cpu: float) -> dict:
    """Inside the container: one build mode under an MLflow run keyed by features key + mode."""
    import mlflow

    from aml import tracking
    from aml.features.build import run_build_features
    from aml.io import read_json

    paths = data_paths(cfgs["data"])
    version = require_data(paths, keys["data"])
    out_dir = paths.features_dir(keys["features"])
    if mode == "verify":
        require_stage_output(out_dir, "features", "features", data_version=version)
    else:
        # A replay of other prepared data under the same config key must not be reused.
        if reset_if_other_data(out_dir, version):
            print(f"{out_dir}: built from other prepared data; starting over")
        vol.commit()
    run_key = f"{keys['features']}-{mode}"
    # Opened before compute: a failed attempt shows up as a FAILED run (and in the cost window).
    with tracking.start_run_for_key(APP_NAME, run_key, run_name=run_key):
        tracking.log_params_flat(cfgs["features"], prefix="features")
        tracking.log_params_flat(
            {"mode": mode, "data_key": keys["data"], "features_key": keys["features"]}
        )
        mlflow.set_tags({"data_version": str(version), "mode": mode})
        summary = jsonable(
            run_build_features(
                paths,
                out_dir,
                cfgs,
                mode=mode,
                on_checkpoint=vol.commit,
                threads=int(cpu),
                log=lambda msg: print(msg, flush=True),  # one line per day: `modal app logs`
            )
        )
        if mode == "bench":
            # The lifetime non-self-loop pairs must reproduce M1's closed_pairs (M2 spec §8.1).
            m1 = paths.model_dir("rules", keys["rules"]) / "summary.json"
            closed = read_json(m1).get("stats", {}).get("closed_pairs") if m1.exists() else None
            nsl = summary["sizing"].get("lifetime_pairs_nsl")
            summary["m1_closed_pairs"] = {"m1": closed, "engine_sql": nsl, "equal": closed == nsl}
        tracking.log_metrics_flat(_no_lists(summary))
    vol.commit()
    return jsonable({"run_key": keys["features"], "out_dir": str(out_dir), **summary})


@app.function(**cpu_job(cpu=REPLAY_CPU, memory_mib=REPLAY_MEMORY_MIB, timeout=BENCH_TIMEOUT_S))
def bench(cfgs: dict, keys: dict) -> dict:
    return _run("bench", cfgs, keys, REPLAY_CPU)


@app.function(**cpu_job(cpu=REPLAY_CPU, memory_mib=REPLAY_MEMORY_MIB, timeout=FULL_TIMEOUT_S))
def full(cfgs: dict, keys: dict) -> dict:
    return _run("full", cfgs, keys, REPLAY_CPU)


@app.function(**cpu_job(cpu=VERIFY_CPU, memory_mib=VERIFY_MEMORY_MIB, timeout=VERIFY_TIMEOUT_S))
def verify(cfgs: dict, keys: dict) -> dict:
    return _run("verify", cfgs, keys, VERIFY_CPU)


@app.local_entrypoint()
def main(mode: str = "full") -> None:
    if mode not in MODES:
        raise SystemExit(f"--mode must be one of {MODES}, got {mode!r}")
    cfgs = load_all_configs()
    keys = {"data": data_key(cfgs), "rules": rules_key(cfgs), "features": features_key(cfgs)}
    fn = {"bench": bench, "full": full, "verify": verify}[mode]
    summary = fn.remote(cfgs, keys)
    print_summary(f"{APP_NAME} {mode} {keys['features']}", summary, keys=list(_PRINT))
