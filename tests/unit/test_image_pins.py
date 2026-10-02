"""The Modal image pins equal uv.lock, and the images follow the verified definition."""

from __future__ import annotations

import importlib
import os
import re
import shutil
import subprocess
import tempfile
import tomllib
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


def _common():
    # Keep the Modal client away from the user's token file; importing never contacts Modal.
    os.environ.setdefault(
        "MODAL_CONFIG_PATH", str(Path(tempfile.gettempdir()) / "aml-tests-no-modal.toml")
    )
    return importlib.import_module("modal_jobs.common")


def _norm(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _base(req: str) -> str:
    return _norm(re.split(r"[\[<>=!~; ]", req, maxsplit=1)[0])


@pytest.fixture(scope="module")
def lock_versions() -> dict[str, set[str]]:
    lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))
    out: dict[str, set[str]] = {}
    for pkg in lock["package"]:
        # uv.lock records local versions (torch 2.14.0+cu126, pyg-lib 0.9.0+pt214cu126).
        out.setdefault(_norm(pkg["name"]), set()).add(pkg["version"].split("+")[0])
    return out


@pytest.fixture(scope="module")
def pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_every_pin_equals_uv_lock(lock_versions):
    pins = _common().PINS
    wrong = {
        name: (version, sorted(lock_versions.get(_norm(name), set())))
        for name, version in pins.items()
        if lock_versions.get(_norm(name)) != {version}
    }
    assert not wrong, f"PINS differ from uv.lock (pin, lock): {wrong}"


def test_pins_cover_project_dependencies(pyproject):
    common = _common()
    main = {_base(r) for r in pyproject["project"]["dependencies"]}
    gnn = {_base(r) for r in pyproject["dependency-groups"]["gnn"]}
    assert {_norm(n) for n in common.MAIN_PACKAGES} == main
    assert {_norm(n) for n in common.GNN_PACKAGES} == gnn
    assert {_norm(n) for n in common.PINS} == main | gnn


def test_requirements_are_exact_and_cpu_image_has_no_torch():
    common = _common()
    for req in common.MAIN_REQUIREMENTS:
        name, sep, version = req.partition("==")
        assert sep and version == common.PINS[re.sub(r"\[.*\]", "", name)], req
    names = {_base(r) for r in common.MAIN_REQUIREMENTS}
    assert not names & {"torch", "torch-geometric", "pyg-lib", "torch-scatter", "torch-sparse"}
    assert "pandera[polars]==0.33.1" in common.MAIN_REQUIREMENTS


def test_gpu_image_sources_match_verified_definition(pyproject):
    common = _common()
    assert common.TORCH == "2.14.0" and common.CUDA_TAG == "cu126"
    assert common.TORCH_INDEX_URL == "https://download.pytorch.org/whl/cu126"
    assert common.PYG_FIND_LINKS == "https://data.pyg.org/whl/torch-2.14.0+cu126.html"
    assert common.PYG_FIND_LINKS in pyproject["tool"]["uv"]["find-links"]
    index = {i["name"]: i["url"] for i in pyproject["tool"]["uv"]["index"]}
    assert index["pytorch-cu126"] == common.TORCH_INDEX_URL
    assert common.PINS["torch_geometric"] == "2.8.0.post1"
    assert common.PINS["pyg_lib"] == "0.9.0"


def test_images_ship_package_data_and_mlflow_env():
    common = _common()
    # The default ignore ships only .py files and would drop *.sql / templates.
    assert common.AML_SOURCE_IGNORE == ["**/__pycache__", "**/*.pyc"]
    assert common.IMAGE_ENV["MLFLOW_TRACKING_URI"] == "file:///data/mlflow"
    assert common.IMAGE_ENV["MLFLOW_ALLOW_FILE_STORE"] == "true"


EXPORT_CMD = ["uv", "export", "--frozen", "--no-default-groups", "--no-hashes", "--no-emit-project"]


def _requirement_lines(text: str) -> list[str]:
    """The requirement lines (name==version [; marker]), without comments or blanks."""
    lines = (ln.strip() for ln in text.splitlines())
    return [ln for ln in lines if ln and not ln.startswith("#")]


def _parse(line: str) -> tuple[str, str, str]:
    req, _, marker = line.partition(";")
    name, sep, version = req.strip().partition("==")
    assert sep, line
    return _norm(re.sub(r"\[.*\]", "", name)), version.strip(), marker.strip()


def test_image_requirements_file_equals_uv_lock(lock_versions):
    """Images install every main-group package (transitive ones too) at the uv.lock version."""
    common = _common()
    text = common.MAIN_REQUIREMENTS_FILE.read_text(encoding="utf-8")
    rows = [_parse(ln) for ln in _requirement_lines(text)]
    assert rows
    wrong = {
        n: (v, sorted(lock_versions.get(n, set())))
        for n, v, _ in rows
        if v not in lock_versions.get(n, set())
    }
    assert not wrong, f"requirements-main.txt differs from uv.lock: {wrong}"
    version_of = {n: v for n, v, _ in rows}
    for pkg in common.MAIN_PACKAGES:  # the direct pins are in it, at the same versions
        assert version_of.get(_norm(pkg)) == common.PINS[pkg], pkg
    assert not set(version_of) & {"torch", "torch-geometric", "pyg-lib", "modal", "pytest"}
    # No index / find-links options: the torch index and PyG links stay in their own layers.
    assert not any(ln.startswith("-") for ln in _requirement_lines(text))


@pytest.mark.skipif(shutil.which("uv") is None, reason="uv not installed")
def test_image_requirements_file_is_a_fresh_export():
    common = _common()
    proc = subprocess.run(
        EXPORT_CMD, cwd=REPO_ROOT, capture_output=True, text=True, encoding="utf-8", timeout=120
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    saved = common.MAIN_REQUIREMENTS_FILE.read_text(encoding="utf-8")
    assert _requirement_lines(proc.stdout) == _requirement_lines(saved), (
        "modal_jobs/requirements-main.txt is stale: regenerate it with the command in its header"
    )
