"""Shared Modal scaffolding: Volume, images, config loading, run keys, guardrail presets.

Run every stage in module mode (`uv run modal run -m modal_jobs.<stage>`) so this file ships.
Importing this module never contacts Modal: Volume and Secret handles hydrate lazily.
Cost guardrails follow docs/modal/COST_NOTES.md.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import modal

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"
AML_SRC = REPO_ROOT / "src" / "aml"

VOLUME_NAME = "aml-data"
DATA_ROOT = "/data"
MLFLOW_URI = "file:///data/mlflow"
vol = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# Exact versions from uv.lock (tests/unit/test_image_pins.py keeps them equal).
PINS: dict[str, str] = {
    # main dependencies (pyproject [project].dependencies)
    "polars": "1.44.2",
    "duckdb": "1.5.6",
    "pyarrow": "25.0.1",
    "pandera": "0.33.1",
    "lightgbm": "4.7.0",
    "optuna": "5.0.0",
    "scikit-learn": "1.9.1",
    "mlflow": "3.16.1",
    "numpy": "2.5.3",
    "pyyaml": "6.0.3",
    "kaggle": "2.2.4",
    "scipy": "1.18.1",
    # gnn group (gpu_image only)
    "torch": "2.14.0",
    "torch_geometric": "2.8.0.post1",
    "pyg_lib": "0.9.0",
}
EXTRAS = {"pandera": "[polars]"}
MAIN_PACKAGES = (
    "polars",
    "duckdb",
    "pyarrow",
    "pandera",
    "lightgbm",
    "optuna",
    "scikit-learn",
    "mlflow",
    "numpy",
    "pyyaml",
    "kaggle",
    "scipy",
)
GNN_PACKAGES = ("torch", "torch_geometric", "pyg_lib")


def requirement(name: str) -> str:
    return f"{name}{EXTRAS.get(name, '')}=={PINS[name]}"


MAIN_REQUIREMENTS = [requirement(n) for n in MAIN_PACKAGES]
# Every main-group package, transitive ones included, exported from uv.lock, so the images run the
# versions the tests ran against (not whatever is newest on PyPI when an image is rebuilt).
# Regenerate after changing the lock with the command in the file's header;
# tests/unit/test_image_pins.py keeps it equal to uv.lock.
MAIN_REQUIREMENTS_FILE = REPO_ROOT / "modal_jobs" / "requirements-main.txt"

TORCH, CUDA_TAG = PINS["torch"], "cu126"  # Modal's host driver supports CUDA 12.x and 13.0
TORCH_INDEX_URL = f"https://download.pytorch.org/whl/{CUDA_TAG}"
PYG_FIND_LINKS = f"https://data.pyg.org/whl/torch-{TORCH}+{CUDA_TAG}.html"

IMAGE_ENV = {
    "MLFLOW_TRACKING_URI": MLFLOW_URI,
    # MLflow 3.16 refuses the file store without this opt-in. A Volume has no file locking,
    # so the file store (one directory per run) is safer there than SQLite.
    "MLFLOW_ALLOW_FILE_STORE": "true",
    "MLFLOW_DISABLE_AGENT_HINT": "1",
    # The images have no git binary; MLflow's git lookup would print a long warning per run.
    "GIT_PYTHON_REFRESH": "quiet",
    "PYTHONUNBUFFERED": "1",
}
# The default ignore ships only .py files, which would drop *.sql and templates/.
AML_SOURCE_IGNORE = ["**/__pycache__", "**/*.pyc"]

cpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("libgomp1")  # LightGBM's OpenMP runtime
    .uv_pip_install(requirements=[str(MAIN_REQUIREMENTS_FILE)])
    .env(IMAGE_ENV)
    .add_local_python_source("aml", ignore=AML_SOURCE_IGNORE)
)

# The verified definition (docs/modal/getting-started.md §4), then the project's main deps.
# No torch-scatter / torch-sparse: PyG 2.8 uses pyg-lib plus native torch scatter.
gpu_image = (
    modal.Image.debian_slim(python_version="3.12")
    .uv_pip_install(f"torch=={TORCH}", index_url=TORCH_INDEX_URL)
    .uv_pip_install(
        requirement("torch_geometric"), requirement("pyg_lib"), find_links=PYG_FIND_LINKS
    )
    .apt_install("libgomp1")
    .uv_pip_install(requirements=[str(MAIN_REQUIREMENTS_FILE)])
    .env(IMAGE_ENV)
    .add_local_python_source("aml", ignore=AML_SOURCE_IGNORE)
)


# --- guardrail presets (COST_NOTES.md) ------------------------------------------------------


def thread_env(cpu: float) -> dict[str, str]:
    """Thread caps = the CPU request: CPU is billed on max(request, usage)."""
    n = str(max(1, int(cpu)))
    return {
        "POLARS_MAX_THREADS": n,
        "OMP_NUM_THREADS": n,
        "OPENBLAS_NUM_THREADS": n,
        "MKL_NUM_THREADS": n,
    }


def cpu_job(cpu: float, memory_mib: int, timeout: int) -> dict[str, Any]:
    """`@app.function` kwargs for a batch CPU job: one container, no retries, fast scale-down.

    memory = (request, limit): a leak is OOM-killed at the limit instead of billed.
    """
    return {
        "image": cpu_image,
        "cpu": cpu,
        "memory": (memory_mib, memory_mib + 4096),
        "timeout": timeout,
        "max_containers": 1,
        "scaledown_window": 2,
        "volumes": {DATA_ROOT: vol},
        "env": thread_env(cpu),
    }


# cuBLAS needs a fixed workspace for deterministic GEMMs under use_deterministic_algorithms.
CUBLAS_WORKSPACE_CONFIG = ":4096:8"


WORKER_MAX_RETRIES = 2  # Modal retries of a GNN worker input (a timed-out attempt is retried)


def gpu_job(
    *,
    gpu: str = "L4",
    cpu: float = 8.0,
    memory_mib: int = 32768,
    timeout: int = 3600,
    retries: bool = False,
) -> dict[str, Any]:
    """`@app.function` kwargs for a GNN GPU worker (M3 spec §12): one container, fast scale-down.

    cpu = (request, limit) with limit = request: extra sampler workers are throttled, never
    billed above the request. memory = (request, request + 8 GiB). `retries` adds
    Retries(max_retries=2, initial_delay=0) (the HPO / train workers only: their runs resume from
    checkpoints, and deterministic errors return instead of raising). Drivers use cpu_job.
    """
    kw: dict[str, Any] = {
        "image": gpu_image,
        "gpu": gpu,
        "cpu": (float(cpu), float(cpu)),
        "memory": (memory_mib, memory_mib + 8192),
        "timeout": timeout,
        "max_containers": 1,
        "scaledown_window": 2,
        "volumes": {DATA_ROOT: vol},
        "env": {**thread_env(cpu), "CUBLAS_WORKSPACE_CONFIG": CUBLAS_WORKSPACE_CONFIG},
    }
    if retries:
        kw["retries"] = modal.Retries(max_retries=WORKER_MAX_RETRIES, initial_delay=0.0)
    return kw


# --- configs and run keys ------------------------------------------------------------------

# data.yaml sections that change prepare_data's outputs, EDA included (`evaluation` only affects
# eval). This is the prepared-data marker's key.
DATA_STAGE_SECTIONS = ("dataset", "expected", "split", "test_views", "fx")
# The sections that change the Parquet tables rules and models read: test_views only feed the
# EDA and evaluation, and the download URL only says where the same files come from. Rules and
# models chain on this key, so editing a test view or the URL does not force a paid re-run.
DATA_CONTENT_SECTIONS = ("dataset", "expected", "split", "fx")
DATASET_SOURCE_KEYS = ("public_url_template",)
# Rules config values the LightGBM features read (the shared round-amount unit).
LGBM_RULES_KEYS = ("round_unit",)
# M2 settings for LightGBM on graph features live in lgbm.yaml under this key. They are dropped
# before hashing the M1 key (and from the config the M1 stage gets), so M1 outputs stay valid.
LGBM_GRAPH_SECTION = "graph"
# Rules config values the feature engine reads besides the scenario settings. Grids, alert rates
# and tune_split are excluded: re-tuning the rules never forces a feature replay.
FEATURES_RULES_KEYS = (
    "round_unit",
    "high_risk_formats",
    "structuring_threshold_usd",
    "structuring_band_low",
    "hub_degree_quantile",
)
# features.yaml sections that only size or gate the bench, never the replay's outputs.
FEATURES_KEY_EXCLUDED = ("bench", "memory_target_mb")


def load_all_configs() -> dict[str, dict]:
    """Load configs/*.yaml on the laptop; remote functions receive plain dicts."""
    from aml.config import load_configs

    return load_configs(CONFIG_DIR)


def data_key(cfgs: dict[str, dict]) -> str:
    from aml.config import run_key

    data = cfgs["data"]
    return run_key("data", {k: data.get(k) for k in DATA_STAGE_SECTIONS})


def data_content_key(cfgs: dict[str, dict]) -> str:
    from aml.config import run_key

    data = cfgs["data"]
    parts = {k: data.get(k) for k in DATA_CONTENT_SECTIONS}
    parts["dataset"] = {
        k: v for k, v in (data.get("dataset") or {}).items() if k not in DATASET_SOURCE_KEYS
    }
    return run_key("data_content", parts)


def rules_key(cfgs: dict[str, dict]) -> str:
    from aml.config import run_key

    return run_key("rules", data_content_key(cfgs), cfgs["rules"])


def lgbm_tx_cfg(lgbm_cfg: dict) -> dict:
    """The M1 LightGBM config: lgbm.yaml without the M2 `graph` section.

    Pass this (not the whole lgbm config) to the M1 stage: its checkpoint fingerprint and MLflow
    params hash the config it gets, so the `graph` section must not reach it.
    """
    return {k: v for k, v in lgbm_cfg.items() if k != LGBM_GRAPH_SECTION}


def lgbm_key(cfgs: dict[str, dict]) -> str:
    from aml.config import run_key

    # Only what the tx features read from the rules config: a rules-scenario edit (grids,
    # windows, alert rate) must not force a paid, bit-identical LightGBM re-run. The M2 `graph`
    # section is dropped, so the M1 key is unchanged by it.
    rules_part = {k: cfgs["rules"][k] for k in LGBM_RULES_KEYS}
    return run_key("lgbm_tx", data_content_key(cfgs), lgbm_tx_cfg(cfgs["lgbm"]), rules_part)


def features_key(cfgs: dict[str, dict]) -> str:
    """The feature engine replay: data, features.yaml (minus bench sizing), the rules settings
    the engine reads (no grids, alert rates or tune_split) and ENGINE_VERSION.

    ENGINE_VERSION is the engine's explicit opt-in to code in the key (M1 keys exclude code).
    """
    from aml.config import run_key
    from aml.features.spec import ENGINE_VERSION

    r = cfgs["rules"]
    scenarios = {
        s: {k: v for k, v in sc.items() if k != "grid"} for s, sc in r["scenarios"].items()
    }
    rules_part = {k: r[k] for k in FEATURES_RULES_KEYS} | {"scenarios": scenarios}
    feats = {k: v for k, v in cfgs["features"].items() if k not in FEATURES_KEY_EXCLUDED}
    return run_key("features", data_content_key(cfgs), feats, rules_part, ENGINE_VERSION)


def rules_engine_key(cfgs: dict[str, dict]) -> str:
    """Rules from engine severities: the M1 rules config (tuning included) + the replay."""
    from aml.config import run_key

    return run_key("rules_engine", rules_key(cfgs), features_key(cfgs))


def lgbm_graph_key(cfgs: dict[str, dict]) -> str:
    from aml.config import run_key

    return run_key("lgbm_graph", features_key(cfgs), cfgs["lgbm"])


def eval_key(
    cfgs: dict[str, dict], with_nofmt: bool = False, gnn_keys: dict[str, str] | None = None
) -> str:
    """The evaluation key. `gnn_keys` (evaluated GNN model -> its set key, M3) is appended only
    when non-empty, so without GNN models the key is M2's, byte for byte."""
    from aml.config import run_key

    parts: list[Any] = [
        rules_engine_key(cfgs),
        lgbm_key(cfgs),
        lgbm_graph_key(cfgs),
        cfgs["data"],
        cfgs["rules"],
        bool(with_nofmt),
    ]
    if gnn_keys:
        parts.append(dict(sorted(gnn_keys.items())))
    return run_key("eval", *parts)


def export_inputs(cfgs: dict[str, dict]) -> dict:
    """What run_export reads from serving.yaml: only `replay`. The M5 demo settings in the same
    file never re-key or rebuild the bundle (the canonical JSON equals the M2 config's)."""
    return {"replay": {"max_events": cfgs["serving"]["replay"]["max_events"]}}


def export_key(cfgs: dict[str, dict]) -> str:
    from aml.config import run_key

    return run_key("export", lgbm_graph_key(cfgs), rules_engine_key(cfgs), export_inputs(cfgs))


def all_keys(cfgs: dict[str, dict]) -> dict[str, str]:
    return {
        "data": data_key(cfgs),
        "rules": rules_key(cfgs),
        "rules_engine": rules_engine_key(cfgs),
        "features": features_key(cfgs),
        "lgbm_tx": lgbm_key(cfgs),
        "lgbm_graph": lgbm_graph_key(cfgs),
        "eval": eval_key(cfgs),
        "export": export_key(cfgs),
    }


# --- M3 (GNN) run keys -----------------------------------------------------------------------

# gnn.yaml `sampler` values gnn_bench decides: excluded from the bench key (no circularity).
BENCH_DECIDED = ("batch_size", "max_edges_per_step", "max_edges_per_eval_step")
# gnn.yaml sections no training key hashes: hardware, money, the pre-registered report rules
# (hashed separately, aml.models.gnn.report_hash) and the bench grid (re-keys only the bench).
GNN_UNKEYED = ("runtime", "budget", "report", "bench")
# The Modal apps whose spend counts against the M3 cap (the cost gate): the GNN jobs and M3's
# dev apps (the Linux test runner, the GPU smoke test); == aml.models.gnn.costplan's.
GNN_APPS = ("aml-gnn-bench", "aml-hpo-gnn", "aml-train-gnn")
DEV_APPS = ("aml-linux-runner", "aml-smoke")
M3_APPS = GNN_APPS + DEV_APPS


def gnn_graph_key(cfgs: dict[str, dict]) -> str:
    """The graph encoding and the replay it reads, + GNN_VERSION (the GNN's opt-in to code)."""
    from aml.config import run_key
    from aml.models.gnn import GNN_VERSION

    return run_key(
        "gnn_graph",
        features_key(cfgs),
        data_content_key(cfgs),
        cfgs["gnn"]["graph"],
        GNN_VERSION,
    )


def gnn_bench_key(cfgs: dict[str, dict]) -> str:
    """The bench: everything it measures, without the values it decides."""
    from aml.config import run_key

    g = cfgs["gnn"]
    sampler = {k: v for k, v in g["sampler"].items() if k not in BENCH_DECIDED}
    faithful = {k: g["protocols"]["faithful"][k] for k in ("fanout",)}
    pna = {k: g["protocols"]["pna"][k] for k in ("hidden", "towers")}
    return run_key(
        "gnn_bench",
        gnn_graph_key(cfgs),
        sampler,
        g["model"],
        g["train"]["neg_rate"],
        g["bench"],
        faithful,
        pna,
    )


def gnn_hpo_key(cfgs: dict[str, dict]) -> str:
    from aml.config import run_key

    g = cfgs["gnn"]
    return run_key("gnn_hpo", gnn_graph_key(cfgs), g["sampler"], g["model"], g["train"], g["hpo"])


def gnn_run_key(
    cfgs: dict[str, dict],
    protocol: str,
    seed: int,
    params: dict[str, Any],
    *,
    dev: bool = False,
    max_epochs: int | None = None,
) -> str:
    """One (protocol, seed) training run; `params` = the effective hyperparameters
    (aml.models.gnn.train.effective_params). The protocol's seed list is not part of it, so a
    cut of the look-ahead seeds keeps seed 0's run."""
    from aml.config import run_key

    g = cfgs["gnn"]
    proto = {k: v for k, v in g["protocols"][protocol].items() if k not in ("seeds", "seed")}
    return run_key(
        "gnn_dev" if dev else "gnn",
        gnn_graph_key(cfgs),
        protocol,
        int(seed),
        g["sampler"],
        g["model"],
        g["train"],
        proto,
        params,
        max_epochs,
    )


def gnn_set_key(cfgs: dict[str, dict], protocol: str, run_keys: list[str]) -> str:
    """A protocol's set = its ordered per-seed run keys (cfgs is unused; the signature matches
    the other key functions)."""
    from aml.config import run_key

    del cfgs
    if not run_keys or not all(isinstance(k, str) for k in run_keys):
        raise ValueError(f"a set needs its per-seed run keys, got {run_keys!r}")
    return run_key(f"gnn_{protocol}", list(run_keys))


def gnn_keys(cfgs: dict[str, dict]) -> dict[str, str]:
    """The GNN keys computable from configs alone. Per-run and set keys need HPO's
    best_params.json and are computed by the job entrypoints."""
    return {
        "gnn_graph": gnn_graph_key(cfgs),
        "gnn_bench": gnn_bench_key(cfgs),
        "gnn_hpo": gnn_hpo_key(cfgs),
    }


# --- stage helpers (used inside containers) ------------------------------------------------

DATA_MARKER = "prepare_summary.json"
# Written into every rules / model output directory: the prepared data it was built from.
DATA_VERSION_FILE = "data_version.json"


def data_paths(data_cfg: dict):
    from aml.paths import DataPaths

    return DataPaths(Path(DATA_ROOT), data_cfg["dataset"]["name"])


def data_version(paths: Any) -> str:
    """Content fingerprint of the prepared tables rules and models read: SHA-256 over their bytes.

    Run keys hash configs only; this catches a prepare code fix or a changed Kaggle file under an
    unchanged config. prepare_data writes byte-identical files for identical inputs, so an
    unchanged re-run keeps the version.
    """
    h = hashlib.sha256()
    for p in (paths.transactions, paths.accounts, paths.labels, paths.fx_rates):
        h.update(p.name.encode())
        with open(p, "rb") as f:
            for chunk in iter(lambda f=f: f.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()[:16]


def require_data(paths: Any, expected_key: str) -> str | None:
    """Fail fast if prepare_data has not run for this data config (stale or missing outputs).

    Returns the prepared data's version (None for a marker written before versions existed).
    """
    marker = paths.parquet_dir / DATA_MARKER
    if not marker.exists():
        raise FileNotFoundError(f"{marker} missing: run `make data` first")
    doc = json.loads(marker.read_text(encoding="utf-8"))
    found = doc.get("data_key")
    if found != expected_key:
        raise RuntimeError(
            f"prepared data has data_key={found!r}, configs expect {expected_key!r}: "
            "re-run `make data`"
        )
    return doc.get("data_version")


def stamp_data_version(out_dir: Path, version: str | None) -> None:
    """Record which prepared data a rules / model output directory is built from."""
    from aml.io import write_json_atomic

    write_json_atomic({"data_version": version}, Path(out_dir) / DATA_VERSION_FILE)


def stamped_data_version(out_dir: Path) -> tuple[bool, str | None]:
    """(stamp exists, the version in it)."""
    path = Path(out_dir) / DATA_VERSION_FILE
    if not path.exists():
        return False, None
    return True, json.loads(path.read_text(encoding="utf-8")).get("data_version")


def built_from(out_dir: Path, version: str | None) -> bool:
    """True if `out_dir` carries a data-version stamp equal to `version`."""
    return stamped_data_version(out_dir) == (True, version)


def reset_if_other_data(out_dir: Path, version: str | None) -> bool:
    """Empty `out_dir` when it holds outputs or checkpoints built from other prepared data, then
    stamp it with `version`. Returns True if it was emptied."""
    exists, found = stamped_data_version(out_dir)
    stale = exists and found != version
    if stale:
        shutil.rmtree(out_dir)
    stamp_data_version(out_dir, version)
    return stale


def require_stage_output(
    out_dir: Path, what: str, make_target: str, data_version: str | None = None
) -> None:
    """Fail if a stage's output is missing or was built from other prepared data.

    `data_version` is the current prepared data's version (from require_data); None skips the
    check (a marker from before versions existed).
    """
    if not (out_dir / "summary.json").exists():
        raise FileNotFoundError(f"{what} output {out_dir} missing: run `make {make_target}` first")
    if data_version is None:
        return
    _, found = stamped_data_version(out_dir)
    if found != data_version:
        raise RuntimeError(
            f"{what} output {out_dir} was built from prepared data version {found!r}, but the "
            f"Volume now holds {data_version!r}: re-run `make {make_target}`"
        )


def _summary(stage_dir: Path) -> dict:
    return json.loads((Path(stage_dir) / "summary.json").read_text(encoding="utf-8"))


def require_same_feature_table(
    features_dir: Path, stages: dict[str, tuple[Path, str]]
) -> str | None:
    """Stage outputs that read the feature parts must come from the parts the features
    directory holds now: their summary's `features_digest` must equal the build summary's.

    The features key hashes configs (and ENGINE_VERSION) only, so a re-replay under the same key
    (an engine fix without a version bump) is caught only here. `stages`: name -> (directory,
    make target that rebuilds it). Returns the digest (None: a build summary without one, which
    predates the digest; nothing is checked).
    """
    from aml.features.spec import FEATURES_DIGEST

    digest = _summary(features_dir).get(FEATURES_DIGEST)
    if digest is None:
        return None
    for name, (stage_dir, target) in stages.items():
        found = _summary(stage_dir).get(FEATURES_DIGEST)
        if found != digest:
            raise RuntimeError(
                f"{name} output {stage_dir} was built from other feature parts ({FEATURES_DIGEST} "
                f"{found!r}; {features_dir} now holds {digest!r}): re-run `make {target}`"
            )
    return digest


def require_features_verified(features_dir: Path) -> dict[str, Any]:
    """The DuckDB oracle (`make features-verify`) must have passed on this feature table:
    verify/verify.json with 0 mismatches, the build's spec_hash and (when recorded) its parts
    digest. Returns the verify summary."""
    from aml.features.build import VERIFY_DIR, VERIFY_FILE
    from aml.features.spec import FEATURES_DIGEST

    path = Path(features_dir) / VERIFY_DIR / VERIFY_FILE
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `make features-verify` first")
    v = json.loads(path.read_text(encoding="utf-8"))
    build = _summary(features_dir)
    bad = []
    if v.get("n_mismatches_total") != 0:
        bad.append(f"mismatches={v.get('n_mismatches_total')!r}")
    if v.get("spec_hash") != build.get("spec_hash"):
        bad.append(f"spec_hash {v.get('spec_hash')!r} != {build.get('spec_hash')!r}")
    if build.get(FEATURES_DIGEST) is not None and v.get(FEATURES_DIGEST) != build[FEATURES_DIGEST]:
        bad.append(f"{FEATURES_DIGEST} {v.get(FEATURES_DIGEST)!r} != {build[FEATURES_DIGEST]!r}")
    if bad:
        raise RuntimeError(
            f"the feature oracle has not passed on {features_dir} ({'; '.join(bad)}): run "
            "`make features-verify`"
        )
    return {
        "rows": v.get("rows"),
        "columns_checked": v.get("columns_checked"),
        "n_mismatches_total": 0,
        "spec_hash": v.get("spec_hash"),
        FEATURES_DIGEST: v.get(FEATURES_DIGEST),
    }


def package_data_files() -> list[str]:
    """Non-.py files under src/aml (posix paths relative to the package), as shipped to Modal."""
    out = []
    for p in sorted(AML_SRC.rglob("*")):
        rel = p.relative_to(AML_SRC)
        if not p.is_file() or p.suffix in {".py", ".pyc"}:
            continue
        # add_local_python_source skips dot-prefixed files/dirs and __pycache__.
        if any(part.startswith(".") or part == "__pycache__" for part in rel.parts):
            continue
        out.append(rel.as_posix())
    return out


def check_package_files(expected: list[str]) -> dict[str, Any]:
    """Inside a container: assert every expected package data file shipped with `aml`."""
    import aml

    pkg = Path(aml.__file__).resolve().parent
    missing = [f for f in expected if not (pkg / f).is_file()]
    assert not missing, f"package data files missing in the container: {missing}"
    return {"aml_dir": str(pkg), "n_files": len(expected)}


def jsonable(obj: Any) -> Any:
    """Round-trip through JSON so remote results are plain values (numpy -> Python, Path -> str)."""
    return json.loads(json.dumps(obj, default=_json_default))


def _json_default(o: Any) -> Any:
    if hasattr(o, "item") and callable(o.item) and getattr(o, "shape", None) == ():
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def print_summary(title: str, summary: dict[str, Any], keys: list[str] | None = None) -> None:
    """Short local printout of a remote summary (top-level scalars only unless keys given)."""
    print(f"== {title}")
    for k, v in summary.items():
        if keys is not None and k not in keys:
            continue
        if keys is None and isinstance(v, dict | list):
            continue
        print(f"  {k}: {v}")


# --- local helpers (laptop; no Modal container) ---------------------------------------------

BILLING_WINDOW_DAYS = 62  # the cost gate sums the M3 apps' spend over this window


def read_volume_json(path: str | Path) -> dict | None:
    """Read a JSON file from the Volume from the laptop (`vol.read_file`, no container).

    `path` is a container path under /data or a Volume-relative path; None if it is missing.
    """
    p = Path(path).as_posix()
    if p == DATA_ROOT or p.startswith(DATA_ROOT + "/"):
        p = p[len(DATA_ROOT) :] or "/"
    try:
        data = b"".join(vol.read_file(p))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None
    return json.loads(data.decode("utf-8"))


def read_volume_jsonl(path: str | Path) -> list[dict] | None:
    """read_volume_json for a JSON-lines file (history.jsonl, a trial log): its records, None if
    missing. A torn last line (no trailing newline) is ignored, as aml.models.gnn.read_jsonl
    does; a malformed line is skipped (the callers only estimate progress from it)."""
    p = Path(path).as_posix()
    if p == DATA_ROOT or p.startswith(DATA_ROOT + "/"):
        p = p[len(DATA_ROOT) :] or "/"
    try:
        data = b"".join(vol.read_file(p))
    except (FileNotFoundError, modal.exception.NotFoundError):
        return None
    *complete, _torn = data.decode("utf-8").split("\n")
    out = []
    for line in complete:
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def _modal_cli_json(args: list[str], timeout: int = 180) -> tuple[Any, str | None]:
    """Run `python -m modal <args>` and parse its JSON stdout. Never raises: (value, error)."""
    cmd = [sys.executable, "-m", "modal", *args]
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"billing CLI failed to run: {type(e).__name__}"
    if proc.returncode != 0:
        tail = (proc.stderr or "").strip().splitlines()[-1:] or ["no stderr"]
        return None, f"billing CLI exited {proc.returncode}: {tail[0][:300]}"
    try:
        return json.loads(proc.stdout), None
    except json.JSONDecodeError:
        return None, "billing CLI output was not JSON"


# `modal billing report` refuses hourly reports over 7 days and daily reports over 31 days.
HOURLY_REPORT_MAX_DAYS = 7
DAILY_REPORT_PIECE_DAYS = 30


def _report_rows(report: Any) -> list[dict]:
    if isinstance(report, list):
        return [r for r in report if isinstance(r, dict)]
    if isinstance(report, dict):
        for key in ("rows", "items", "data", "report", "results"):
            if isinstance(report.get(key), list):
                return [r for r in report[key] if isinstance(r, dict)]
    return []


def _row_start(row: dict) -> datetime | None:
    raw = row.get("interval_start")
    if not isinstance(raw, str):
        return None
    try:
        ts = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    return ts if ts.tzinfo else ts.replace(tzinfo=UTC)


def fetch_billing_report(
    start_ms: int | None, resolution: str = "h"
) -> tuple[list | dict | None, str | None]:
    """Run `modal billing report --json` locally from the day of `start_ms` (ms since epoch;
    None = today). Never raises: returns (report, error). (Moved here from evaluate.py.)

    An hourly report longer than HOURLY_REPORT_MAX_DAYS is fetched as daily rows up to the start
    of today (UTC; the end date is exclusive) plus hourly rows from today, merged into one list
    (rows from each part are kept only on their side of today's midnight, so nothing is counted
    twice)."""
    now = datetime.now(UTC)
    start = datetime.fromtimestamp(start_ms / 1000, tz=UTC) if start_ms else now
    day = "%Y-%m-%d"
    if resolution != "h" or (now - start).days < HOURLY_REPORT_MAX_DAYS - 1:
        return _modal_cli_json(
            [
                "billing",
                "report",
                "--start",
                start.strftime(day),
                "--resolution",
                resolution,
                "--json",
            ]
        )
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)
    daily: list[dict] = []
    lo = start.replace(hour=0, minute=0, second=0, microsecond=0)
    while lo < midnight:  # daily reports are refused above 31 days: fetch 30-day pieces
        hi = min(lo + timedelta(days=DAILY_REPORT_PIECE_DAYS), midnight)
        part, err = _modal_cli_json(
            [
                "billing",
                "report",
                "--start",
                lo.strftime(day),
                "--end",
                hi.strftime(day),
                "--resolution",
                "d",
                "--json",
            ]
        )
        if err is not None:
            return None, err
        daily += [r for r in _report_rows(part) if (s := _row_start(r)) is None or lo <= s < hi]
        lo = hi
    hourly, err = _modal_cli_json(
        ["billing", "report", "--start", midnight.strftime(day), "--resolution", "h", "--json"]
    )
    if err is not None:
        return None, err
    rows = daily
    rows += [r for r in _report_rows(hourly) if (s := _row_start(r)) is None or s >= midnight]
    return rows, None


def fetch_billing_summary() -> tuple[dict | None, str | None]:
    """Run `modal billing summary --json` (this cycle: {metered_cost, billed_cost, adjustments,
    metered_cost_breakdown}, amounts as strings). Never raises: returns (summary, error)."""
    out, err = _modal_cli_json(["billing", "summary", "--json"])
    if err is None and not isinstance(out, dict):
        return None, "billing summary output was not a JSON object"
    return out, err


def billing_window_start_ms(days: int = BILLING_WINDOW_DAYS) -> int:
    """Start of the cost gate's billing window: now - `days`, in ms since epoch."""
    return int((time.time() - days * 86400) * 1000)
