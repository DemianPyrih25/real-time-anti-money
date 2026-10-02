"""MLflow UI over the Volume's file store.

    uv run modal serve -m modal_jobs.mlflow_ui     # Ctrl-C stops it

The URL is public while it runs (synthetic-data metrics only). Restart it to see runs
committed after it started.
"""

from __future__ import annotations

import subprocess

import modal

from modal_jobs.common import DATA_ROOT, MLFLOW_URI, cpu_image, vol

APP_NAME = "aml-mlflow-ui"
PORT = 5000
app = modal.App(APP_NAME)


@app.function(
    image=cpu_image,
    cpu=1.0,
    memory=(2048, 4096),
    timeout=3600,
    max_containers=1,
    # Not a batch job: keep the container for a short while between page loads.
    scaledown_window=120,
    volumes={DATA_ROOT: vol},
)
@modal.web_server(PORT, startup_timeout=120)
def ui() -> None:
    # A web server must bind 0.0.0.0; MLflow rejects Host headers it does not know.
    subprocess.Popen(
        [
            "mlflow",
            "server",
            "--backend-store-uri",
            MLFLOW_URI,
            "--host",
            "0.0.0.0",
            "--port",
            str(PORT),
            "--allowed-hosts",
            "*.modal.run",
            "--workers",
            "1",
        ]
    )
