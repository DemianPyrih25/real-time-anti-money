"""Download HI-Small, build the canonical Parquet tables + labels, run the EDA (PLAN §6 M1).

uv run modal run -m modal_jobs.prepare_data [--force-download]
"""

from __future__ import annotations

import time

import modal
from modal.exception import NotFoundError

from modal_jobs.common import (
    DATA_MARKER,
    cpu_job,
    data_key,
    data_paths,
    data_version,
    jsonable,
    load_all_configs,
    print_summary,
    thread_env,
    vol,
)

APP_NAME = "aml-prepare-data"
CPU = 8.0
app = modal.App(APP_NAME)


@app.function(**cpu_job(cpu=CPU, memory_mib=16384, timeout=3600))
def prepare(data_cfg: dict, run_key: str, force_download: bool = False) -> dict:
    import mlflow

    from aml import tracking
    from aml.data.download import fetch_dataset
    from aml.data.prepare import prepare_data
    from aml.io import write_json_atomic

    paths = data_paths(data_cfg)
    # The tables are rewritten in place: drop the marker first, so a failure part-way can never
    # leave new tables under the previous run's key (the Volume commits partial writes).
    (paths.parquet_dir / DATA_MARKER).unlink(missing_ok=True)
    # The run opens before any compute, so a failed attempt is recorded (FAILED, with its start
    # time for the cost window) instead of leaving no trace.
    with tracking.start_run_for_key(APP_NAME, run_key, run_name=run_key):
        tracking.log_params_flat(data_cfg, prefix="data")
        tracking.log_params_flat({"run_key": run_key})
        t0 = time.perf_counter()
        checksums = fetch_dataset(data_cfg, paths.raw_dir, force=force_download)
        vol.commit()  # keep the raw files (and the marker removal) even if a later step fails
        t_download = time.perf_counter() - t0
        # Tags, not params: a forced re-download of a changed file must be able to update them.
        mlflow.set_tags({f"sha256.{name}": value for name, value in checksums.items()})

        summary = jsonable(prepare_data(paths, data_cfg, threads=int(CPU)))
        summary["download_seconds"] = t_download
        version = data_version(paths)
        write_json_atomic(
            {
                "data_key": run_key,
                "data_version": version,
                "checksums": checksums,
                "summary": summary,
            },
            paths.parquet_dir / DATA_MARKER,
        )
        mlflow.set_tags({"data_version": version})
        tracking.log_metrics_flat(summary)
        eda_md = paths.reports / "eda.md"
        if eda_md.exists():
            mlflow.log_artifact(str(eda_md))
    vol.commit()
    return jsonable(
        {"run_key": run_key, "data_version": version, "checksums": checksums, **summary}
    )


def _kaggle_secrets() -> list[modal.Secret]:
    """The `kaggle` Secret only if it exists; the public URL needs none."""
    secret = modal.Secret.from_name("kaggle")
    try:
        secret.hydrate()
    except NotFoundError:
        print("no 'kaggle' Secret found: public download URL only")
        return []
    return [secret]


@app.local_entrypoint()
def main(force_download: bool = False) -> None:
    cfgs = load_all_configs()
    key = data_key(cfgs)
    secrets = _kaggle_secrets()
    # with_options(secrets=...) replaces the static secrets, including the one Modal builds
    # from `env=`, so the thread caps are passed again.
    fn = prepare.with_options(secrets=secrets, env=thread_env(CPU)) if secrets else prepare
    summary = fn.remote(cfgs["data"], key, force_download)
    print_summary(f"{APP_NAME} {key}", summary)
