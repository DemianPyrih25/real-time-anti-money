"""Smoke test of gpu_image: CUDA, versions, one GINEConv forward, package data files (M1).

    uv run modal run -m modal_jobs.smoke            # L4 (default)
    uv run modal run -m modal_jobs.smoke --gpu T4   # before gnn_bench benchmarks T4

Must pass on a GPU type before any real job uses it (docs/modal/getting-started.md §5).
"""

from __future__ import annotations

import modal

from modal_jobs.common import (
    DATA_ROOT,
    check_package_files,
    gpu_image,
    package_data_files,
    print_summary,
    thread_env,
    vol,
)

APP_NAME = "aml-smoke"
DEFAULT_GPU = "L4"
ALLOWED_GPUS = ("L4", "T4")
app = modal.App(APP_NAME)


def _versions() -> dict[str, str]:
    import importlib.metadata as md
    import platform

    names = ["torch", "torch_geometric", "pyg_lib", "polars", "duckdb", "lightgbm", "mlflow"]
    out = {"python": platform.python_version()}
    for n in names:
        try:
            out[n] = md.version(n)
        except md.PackageNotFoundError:
            out[n] = "missing"
    return out


def _gine_forward(device: str) -> list[int]:
    """One GINEConv forward on a 4-node directed cycle with edge features."""
    import torch
    from torch_geometric.nn import GINEConv

    torch.manual_seed(0)
    x = torch.randn(4, 8, device=device)
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], device=device)
    edge_attr = torch.randn(4, 8, device=device)
    mlp = torch.nn.Sequential(torch.nn.Linear(8, 16), torch.nn.ReLU(), torch.nn.Linear(16, 16))
    conv = GINEConv(mlp, edge_dim=8).to(device)
    with torch.no_grad():
        out = conv(x, edge_index, edge_attr)
    assert out.shape == (4, 16), out.shape
    assert bool(torch.isfinite(out).all()), "non-finite GINEConv output"
    return list(out.shape)


def _check_pyg_lib() -> None:
    import pyg_lib  # noqa: F401
    import torch_geometric.typing as pyg_typing

    assert pyg_typing.WITH_PYG_LIB, "torch_geometric does not see pyg_lib"


@app.function(
    image=gpu_image,
    gpu=DEFAULT_GPU,  # overridden per call with .with_options(gpu=...)
    cpu=2.0,
    memory=(8192, 12288),
    timeout=900,
    max_containers=1,
    scaledown_window=2,
    volumes={DATA_ROOT: vol},
    env=thread_env(2.0),
)
def gpu_check(gpu: str, expected_files: list[str], run_key: str) -> dict:
    import torch

    from aml import tracking

    # Opened before the checks: a failed smoke test is recorded as a FAILED run (and counts in
    # the cost window) instead of leaving no trace.
    with tracking.start_run_for_key(APP_NAME, run_key, run_name=run_key):
        assert torch.cuda.is_available(), "CUDA is not available"
        versions = _versions()
        info = {
            "gpu_requested": gpu,
            "device_name": torch.cuda.get_device_name(0),
            "cuda_runtime": torch.version.cuda,
            "versions": versions,
            "gine_out_shape": _gine_forward("cuda"),
            "package_files": check_package_files(expected_files),
        }
        _check_pyg_lib()
        print(info)
        tracking.log_params_flat(
            {"gpu": gpu, "device": info["device_name"], "cuda": info["cuda_runtime"], **versions}
        )
    vol.commit()
    return info


@app.function(
    image=gpu_image,  # no GPU: PyG + pyg-lib must also import in CPU-only containers
    cpu=1.0,
    memory=(4096, 8192),
    timeout=600,
    max_containers=1,
    scaledown_window=2,
    volumes={DATA_ROOT: vol},
    env=thread_env(1.0),
)
def cpu_check(expected_files: list[str]) -> dict:
    import torch

    _check_pyg_lib()
    return {
        "cuda_available": torch.cuda.is_available(),
        "versions": _versions(),
        "gine_out_shape": _gine_forward("cpu"),
        "package_files": check_package_files(expected_files),
    }


@app.local_entrypoint()
def main(gpu: str = DEFAULT_GPU) -> None:
    gpu = gpu.upper()
    if gpu not in ALLOWED_GPUS:
        raise SystemExit(f"--gpu must be one of {ALLOWED_GPUS} (COST_NOTES.md), got {gpu!r}")
    files = package_data_files()
    if not files:
        print("warning: no package data files under src/aml (expected *.sql)")
    cpu_info = cpu_check.remote(files)
    print_summary("cpu_check (gpu_image, no GPU)", cpu_info, keys=["versions", "package_files"])
    gpu_info = gpu_check.with_options(gpu=gpu).remote(gpu, files, f"smoke-{gpu.lower()}")
    print_summary(f"gpu_check ({gpu})", gpu_info)
    print(f"smoke OK on {gpu_info['device_name']} ({len(files)} package data files present)")
