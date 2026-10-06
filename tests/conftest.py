"""Shared pytest fixtures: the tiny synthetic dataset and configs adapted to it."""

from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

from tests.fixtures.synthetic import SyntheticDataset, make_synthetic

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(name: str) -> dict:
    with (CONFIG_DIR / name).open() as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="session")
def synthetic(tmp_path_factory: pytest.TempPathFactory) -> SyntheticDataset:
    return make_synthetic(tmp_path_factory.mktemp("raw"))


@pytest.fixture(scope="session")
def data_cfg(synthetic: SyntheticDataset) -> dict:
    """configs/data.yaml with the expected counts replaced by the fixture's."""
    cfg = copy.deepcopy(load_yaml("data.yaml"))
    cfg["expected"] = dict(synthetic.expected)
    return cfg


@pytest.fixture(scope="session")
def rules_cfg() -> dict:
    return copy.deepcopy(load_yaml("rules.yaml"))


@pytest.fixture(scope="session")
def prepared(tmp_path_factory: pytest.TempPathFactory, synthetic: SyntheticDataset, data_cfg: dict):
    """Run the real prepare_data stage on the fixture; returns the DataPaths of its outputs."""
    import shutil

    from aml.data.prepare import prepare_data
    from aml.paths import DataPaths

    paths = DataPaths(tmp_path_factory.mktemp("volume"))
    paths.raw_dir.mkdir(parents=True)
    for f in (synthetic.transactions_csv, synthetic.patterns_txt):
        shutil.copy(f, paths.raw_dir / f.name)
    prepare_data(paths, data_cfg, threads=2)
    return paths


@pytest.fixture(scope="session")
def serving_bundle(
    prepared, data_cfg: dict, rules_cfg: dict, tmp_path_factory: pytest.TempPathFactory
) -> Path:
    """One exported fixture bundle (real engine, test-built rules and model) for the M5 tests.

    Read-only by convention: copy it before changing anything. Fails loudly when the fixture is
    degenerate for streaming tests (no model alert, no minute with two events)."""
    from tests.fixtures.serving_bundle import (
        build_fixture_bundle,
        build_world,
        check_fixture,
        copy_prepared,
    )

    root = tmp_path_factory.mktemp("serving_world")
    cfgs = {
        "data": copy.deepcopy(data_cfg),
        "rules": copy.deepcopy(rules_cfg),
        "features": load_yaml("features.yaml"),
        "serving": load_yaml("serving.yaml"),
    }
    world = build_world(copy_prepared(prepared, root / "volume"), cfgs)
    out = root / "bundle_shared"
    build_fixture_bundle(world, out)
    problems = check_fixture(out)
    if problems:
        pytest.fail(f"fix the fixture: {'; '.join(problems)}")
    return out


@pytest.fixture(scope="session")
def fixture_tag(serving_bundle: Path) -> str:
    """The alert rate tag the M5 tests use: the headline tag if it alerts in the slice, else the
    tag with the most alerts."""
    from tests.fixtures.serving_bundle import pick_tag

    return pick_tag(serving_bundle)


@pytest.fixture(scope="session")
def lgbm_cfg() -> dict:
    cfg = copy.deepcopy(load_yaml("lgbm.yaml"))
    cfg["optuna"]["n_trials"] = 3
    cfg["optuna"]["n_startup_trials"] = 2
    cfg["num_boost_round"] = 60
    cfg["early_stopping_rounds"] = 10
    cfg["seeds"] = [0, 1]
    return cfg
