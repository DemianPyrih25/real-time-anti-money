"""Model inputs are whitelisted: no ids, timestamps, split or label columns (PLAN.md §4)."""

from __future__ import annotations

from pathlib import Path

import lightgbm as lgb
import polars as pl
import pytest

from aml.data.prepare import TRANSACTION_COLUMNS
from aml.features.tx_features import (
    CATEGORICAL_FEATURES,
    FORBIDDEN_FEATURES,
    TX_FEATURES,
    assert_whitelisted,
    build_tx_features,
    fit_vocab,
    is_forbidden,
)
from aml.io import read_json
from aml.models.lgbm import run_lgbm_stage

LABEL_COLUMNS = ["is_laundering", "attempt_id", "typology", "attempt_size", "typology_detail"]


def test_whitelist_has_no_forbidden_names() -> None:
    assert not any(is_forbidden(f) for f in TX_FEATURES)
    assert set(CATEGORICAL_FEATURES) <= set(TX_FEATURES)
    assert_whitelisted(TX_FEATURES)


def test_every_raw_and_label_column_except_features_is_rejected() -> None:
    # Transaction columns that are not features, and every label-table column.
    for name in [c for c in TRANSACTION_COLUMNS if c not in TX_FEATURES] + LABEL_COLUMNS:
        if name in {"amount_received", "amount_paid", "amount_usd"}:
            continue  # raw amounts are not forbidden, just not in the whitelist
        assert is_forbidden(name), name
    assert set(LABEL_COLUMNS) <= FORBIDDEN_FEATURES


@pytest.mark.parametrize(
    "name",
    [
        *sorted(FORBIDDEN_FEATURES),
        "event_time",
        "Timestamp",
        "label_rate",
        "minute_of_time",
        "amount_paid",  # not forbidden, but not whitelisted either
        "unknown",
    ],
)
def test_assert_whitelisted_rejects(name: str) -> None:
    with pytest.raises(ValueError):
        assert_whitelisted([*TX_FEATURES, name])


def test_build_tx_features_output_has_no_forbidden_columns(prepared) -> None:
    tx = pl.read_parquet(prepared.transactions)
    out = build_tx_features(tx, fit_vocab(tx.filter(pl.col("split") == "train")), 100)
    # row_id is the join key, not a model input; everything else must be the whitelist.
    assert out.columns == ["row_id", *TX_FEATURES]
    assert not [c for c in out.columns[1:] if is_forbidden(c)]


@pytest.fixture(scope="module")
def stage_dir(prepared, lgbm_cfg, rules_cfg, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("lgbm_whitelist")
    run_lgbm_stage(prepared, out, lgbm_cfg, rules_cfg, threads=2)
    return out


def test_saved_feature_names_are_whitelisted(stage_dir: Path) -> None:
    names = read_json(stage_dir / "feature_names.json")
    assert names == TX_FEATURES
    assert_whitelisted(names)


def test_booster_inputs_are_whitelisted(stage_dir: Path, lgbm_cfg) -> None:
    for s in lgbm_cfg["seeds"]:
        booster = lgb.Booster(model_file=str(stage_dir / f"booster_s{s}.txt"))
        names = booster.feature_name()
        assert set(names) <= set(TX_FEATURES)
        assert not [n for n in names if is_forbidden(n)]
        assert names == TX_FEATURES


# ---------------------------------------------------------------------------------------------
# M2: the engine's model inputs (M2 spec §5.9). Names are checked literally, not only through
# the spec's own metadata.

ENGINE_EDGE_LEVEL = {
    "pair_is_new",
    "out_port",
    "in_port",
    "u_out_gap",
    "u_in_gap",
    "v_in_gap",
    "v_out_gap",
    "pair_gap",
    "rev_pair_gap",
}


@pytest.fixture(scope="module")
def engine_spec(prepared, rules_cfg):
    from tests.fixtures.engine_frames import fixture_spec

    return fixture_spec(prepared, rules_cfg)[0]


def test_engine_model_inputs_pass_the_whitelist(engine_spec) -> None:
    names = list(engine_spec.feature_names)
    assert len(names) == 79 and len(set(names)) == 79
    assert engine_spec.assert_model_inputs(names) == names
    assert not [n for n in names if is_forbidden(n)]
    assert names[: len(TX_FEATURES)] == TX_FEATURES
    assert engine_spec.categorical_names == tuple(CATEGORICAL_FEATURES)


def test_engine_non_model_columns_are_rejected(engine_spec) -> None:
    from aml.features.spec import NON_MODEL_COLUMNS

    names = list(engine_spec.feature_names)
    raw = [c for c in TRANSACTION_COLUMNS if c not in TX_FEATURES]
    for col in [*NON_MODEL_COLUMNS, *LABEL_COLUMNS, *raw, "hub", "u_is_hub", "label_rate"]:
        with pytest.raises(ValueError):
            engine_spec.assert_model_inputs([*names, col])
    with pytest.raises(ValueError):  # repeated input
        engine_spec.assert_model_inputs([*names, names[0]])


def test_no_model_input_encodes_hub_status_severities_or_labels(engine_spec) -> None:
    from aml.rules.sql_baseline import SCENARIOS

    for n in engine_spec.feature_names:
        low = n.lower()
        assert "hub" not in low, n
        assert not any(w in low for w in ("laundering", "attempt", "typology", "label")), n
        assert n not in SCENARIOS, n


def test_format_ablation_removes_exactly_the_format_derived_features(engine_spec) -> None:
    slugs = [f.lower().replace(" ", "_") for f in engine_spec.vocab["payment_format"]]
    expected = {"payment_format", "v_in_same_fmt_1d", *(f"u_out_fmt_{s}_1d" for s in slugs)}
    assert set(engine_spec.format_derived_names) == expected and len(expected) == 9
    kept = [n for n in engine_spec.feature_names if n not in expected]
    assert len(kept) == 70
    assert not [n for n in kept if "fmt" in n or "format" in n]


def test_gnn_edge_attributes_are_edge_level_only(engine_spec) -> None:
    edge = set(engine_spec.gnn_edge_attr_names)
    assert edge <= set(TX_FEATURES) | ENGINE_EDGE_LEVEL
    assert edge == set(TX_FEATURES) | ENGINE_EDGE_LEVEL  # all of them are available to M3
    # No windowed node aggregate (counts, sums, moments, paths) is an edge attribute.
    assert not [n for n in edge if n.endswith(("_1d", "_3d", "_12h", "_2d"))]


def test_booster_feature_names_pass_the_engine_whitelist(engine_spec, tmp_path) -> None:
    import numpy as np

    names = list(engine_spec.feature_names)
    rng = np.random.default_rng(0)
    x = rng.random((200, len(names)))
    y = (x[:, 0] > 0.5).astype(int)
    params = {"objective": "binary", "verbose": -1, "num_threads": 1, "min_data_in_leaf": 5}
    booster = lgb.train(params, lgb.Dataset(x, y, feature_name=names), num_boost_round=3)
    path = tmp_path / "booster.txt"
    booster.save_model(str(path))
    loaded = lgb.Booster(model_file=str(path))
    assert engine_spec.assert_model_inputs(loaded.feature_name()) == names
    leaky = lgb.train(
        params,
        lgb.Dataset(np.c_[x, np.arange(200)], y, feature_name=[*names, "rank"]),
        num_boost_round=3,
    )
    with pytest.raises(ValueError):
        engine_spec.assert_model_inputs(leaky.feature_name())
