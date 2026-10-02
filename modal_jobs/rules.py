"""Rule baseline: severities, greedy thresholds on val_early, flags (PLAN §6 M1, M2 spec §8.2).

    uv run modal run -m modal_jobs.rules                  # --source engine (the default from M2 on)
    uv run modal run -m modal_jobs.rules --source sql     # the M1 stage, unchanged

engine: severities from the feature engine's parts (`make features` first), checked against the
M1 SQL on every row (parity.json, /data/reports/parity.md; the job fails on a mismatch), then
tuned with M1's code. Output: /data/models/rules_engine/<rules_engine_key>/.
sql: the M1 SQL stage. Output: /data/models/rules/<rules_key>/.
Both are read by modal_jobs.evaluate (M2 reads rules_engine).
"""

from __future__ import annotations

import modal

from modal_jobs.common import (
    built_from,
    cpu_job,
    data_key,
    data_paths,
    features_key,
    jsonable,
    load_all_configs,
    print_summary,
    require_data,
    require_stage_output,
    rules_engine_key,
    rules_key,
    stamp_data_version,
    vol,
)

APP_NAME = "aml-rules"
CPU = 8.0
SOURCES = ("engine", "sql")
app = modal.App(APP_NAME)


@app.function(**cpu_job(cpu=CPU, memory_mib=16384, timeout=3600))
def rules(rules_cfg: dict, data_cfg: dict, keys: dict) -> dict:
    import mlflow

    from aml import tracking
    from aml.rules.sql_baseline import run_rules_stage

    paths = data_paths(data_cfg)
    version = require_data(paths, keys["data"])
    out_dir = paths.model_dir("rules", keys["rules"])
    # Opened before compute: a failed attempt shows up as a FAILED run (and in the cost window).
    with tracking.start_run_for_key(APP_NAME, keys["rules"], run_name=keys["rules"]):
        tracking.log_params_flat(rules_cfg, prefix="rules")
        tracking.log_params_flat({"data_key": keys["data"], "out_dir": str(out_dir)})
        mlflow.set_tags({"data_version": str(version)})
        summary = jsonable(run_rules_stage(paths, out_dir, rules_cfg, data_cfg, threads=int(CPU)))
        stamp_data_version(out_dir, version)
        tracking.log_metrics_flat(summary)
        thresholds = out_dir / "thresholds.json"
        if thresholds.exists():
            mlflow.log_artifact(str(thresholds))
    vol.commit()
    return jsonable(
        {"run_key": keys["rules"], "out_dir": str(out_dir), "data_version": version, **summary}
    )


@app.function(**cpu_job(cpu=CPU, memory_mib=16384, timeout=3600))
def rules_engine(rules_cfg: dict, data_cfg: dict, keys: dict) -> dict:
    import mlflow

    from aml import tracking
    from aml.rules.scenarios import run_rules_engine_stage

    paths = data_paths(data_cfg)
    version = require_data(paths, keys["data"])
    features_dir = paths.features_dir(keys["features"])
    require_stage_output(features_dir, "features", "features", data_version=version)
    out_dir = paths.model_dir("rules_engine", keys["rules_engine"])
    # The M1 SQL stage's outputs, compared when built from this prepared data (m1_regression in
    # parity.json). rules_key hashes configs only: after a re-prepare that changed the data, the
    # old M1 directory under the same key must not be compared.
    m1_dir, m1_skip = paths.model_dir("rules", keys["rules"]), None
    if not built_from(m1_dir, version):
        m1_skip = (
            f"the M1 rules outputs in {m1_dir} are missing or were built from other prepared "
            "data (refresh them with `uv run modal run -m modal_jobs.rules --source sql`)"
        )
        print(f"m1_regression not compared: {m1_skip}")
        m1_dir = None
    # Opened before compute: a failed attempt shows up as a FAILED run (and in the cost window).
    with tracking.start_run_for_key(APP_NAME, keys["rules_engine"], run_name=keys["rules_engine"]):
        tracking.log_params_flat(rules_cfg, prefix="rules")
        tracking.log_params_flat(
            {
                "source": "engine",
                "data_key": keys["data"],
                "features_key": keys["features"],
                "out_dir": str(out_dir),
            }
        )
        mlflow.set_tags({"data_version": str(version)})
        try:
            summary = jsonable(
                run_rules_engine_stage(
                    paths,
                    features_dir,
                    out_dir,
                    rules_cfg,
                    data_cfg,
                    threads=int(CPU),
                    m1_rules_dir=m1_dir,
                    m1_skip_reason=m1_skip,
                )
            )
        finally:
            # A parity failure leaves parity.json / parity_mismatches.parquet / parity.md as
            # evidence: keep them on the Volume.
            vol.commit()
        stamp_data_version(out_dir, version)
        tracking.log_metrics_flat(summary)
        for name in ("thresholds.json", "parity.json"):
            if (out_dir / name).exists():
                mlflow.log_artifact(str(out_dir / name))
    vol.commit()
    return jsonable(
        {
            "run_key": keys["rules_engine"],
            "out_dir": str(out_dir),
            "data_version": version,
            **summary,
        }
    )


@app.local_entrypoint()
def main(source: str = "engine") -> None:
    if source not in SOURCES:
        raise SystemExit(f"--source must be one of {SOURCES}, got {source!r}")
    cfgs = load_all_configs()
    if source == "sql":
        keys = {"data": data_key(cfgs), "rules": rules_key(cfgs)}
        summary = rules.remote(cfgs["rules"], cfgs["data"], keys)
        title = f"{APP_NAME} {keys['rules']}"
    else:
        keys = {
            "data": data_key(cfgs),
            "rules": rules_key(cfgs),
            "features": features_key(cfgs),
            "rules_engine": rules_engine_key(cfgs),
        }
        summary = rules_engine.remote(cfgs["rules"], cfgs["data"], keys)
        title = f"{APP_NAME} (engine) {keys['rules_engine']}"
    print_summary(title, summary)
    head = summary["headline_rate_tag"]
    diag = summary["scenario_diagnostics"]["rates"][head]
    print(f"  active scenarios at {head}: {diag['active']}")
    if diag["infeasible"]:
        off = diag["infeasible"]
        print(f"  WARNING: cannot fit the {head} budget even at their strictest value: {off}")
    parity = summary.get("parity")
    if parity:
        trunc = parity["rule_trunc"]
        print(
            f"  parity: ok={parity['ok']} mismatches={parity['mismatches_total']} "
            f"truncated rows={trunc['rows']} m1_regression={parity['m1_regression']}"
        )
