"""Shared Modal scaffolding: Volume, images, config loading, run keys, guardrail presets.

Run every stage in module mode (`uv run modal run -m modal_jobs.<stage>`) so this file ships.
Importing this module never contacts Modal: Volume and Secret handles hydrate lazily.
Cost guardrails follow docs/modal/COST_NOTES.md.
"""

from __future__ import annotations

import hashlib
import json
import shutil
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


def eval_key(cfgs: dict[str, dict], with_nofmt: bool = False) -> str:
    from aml.config import run_key

    return run_key(
        "eval",
        rules_engine_key(cfgs),
        lgbm_key(cfgs),
        lgbm_graph_key(cfgs),
        cfgs["data"],
        cfgs["rules"],
        bool(with_nofmt),
    )


def export_key(cfgs: dict[str, dict]) -> str:
    from aml.config import run_key

    return run_key("export", lgbm_graph_key(cfgs), rules_engine_key(cfgs), cfgs["serving"])


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
