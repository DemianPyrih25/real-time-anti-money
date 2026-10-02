"""The rules stage on engine severities (M2 spec §6, §8.2), end to end on the synthetic fixture.

build_features -> run_rules_engine_stage: every row's severities equal the M1 SQL's, thresholds
and flags equal the M1 stage's, truncated rows are the only exclusions and are reported, and an
injected mismatch fails the stage with evidence and without flags.
"""

from __future__ import annotations

import copy
import os
import shutil
import warnings
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from aml.features.spec import (
    FEATURE_SPEC_FILE,
    INPUT_COLUMNS,
    PARTS_DIR,
    SEVERITY_COLUMNS,
    TRUNC_COLUMNS,
    EngineSpec,
    part_name,
    part_paths,
)
from aml.io import read_json, write_json_atomic
from aml.paths import DataPaths
from aml.rules.scenarios import (
    MISMATCHES_FILE,
    PARITY_FILE,
    PARITY_REPORT,
    ParityError,
    compare_severities,
    render_parity_md,
    run_rules_engine_stage,
)
from aml.rules.sql_baseline import (
    RULES_OUTPUTS,
    SCENARIOS,
    compute_severities,
    connect,
    hub_accounts,
    hub_degree_cap,
    register_transactions,
    run_rules_stage,
)

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"


def _engine_available() -> bool:
    from aml.features.engine import Engine
    from tests.unit.test_cycles import make_spec

    try:
        Engine.create(make_spec())
    except NotImplementedError:
        return False
    return True


needs_engine = pytest.mark.skipif(
    not _engine_available(), reason="aml.features.engine.Engine is not implemented yet"
)


# ---------------------------------------------------------------------------------------------
# Building the feature table


def _local_build(paths: DataPaths, out_dir: Path, cfgs: dict[str, dict]) -> None:
    """A minimal replay driver (feature_spec.json + one part per day): used while
    features.build.run_build_features is a stub, or always with AML_TEST_LOCAL_BUILD=1 (to test
    this stage independently of the real driver)."""
    from aml.features.engine import Engine
    from aml.features.tx_features import fit_vocab

    tx = pl.read_parquet(paths.transactions).sort("rank")
    con = connect(threads=1)
    try:
        register_transactions(con, paths.transactions)
        hub_cap = hub_degree_cap(con, cfgs["rules"]["hub_degree_quantile"])
        hubs = hub_accounts(con, hub_cap)
    finally:
        con.close()
    spec = EngineSpec.from_configs(
        cfgs["features"],
        cfgs["rules"],
        n_accounts=pl.read_parquet(paths.accounts).height,
        vocab=fit_vocab(tx.filter(pl.col("split") == "train")),
        hub_cap=hub_cap,
        hubs=hubs,
    )
    write_json_atomic(spec.to_json(), out_dir / FEATURE_SPEC_FILE)
    eng = Engine.create(spec)
    rows = [eng.process(eng.prepare(*r)) for r in tx.select(list(INPUT_COLUMNS)).iter_rows()]
    n = spec.n_features
    feats = np.asarray([r[:n] for r in rows], np.float64).astype(np.float32)
    data = {
        "row_id": tx["row_id"],
        "rank": tx["rank"],
        "day": tx["day"].cast(pl.Int16),
        "split": tx["split"],
        **{name: feats[:, j] for j, name in enumerate(spec.feature_names)},
        **{s: [r[spec.i_sev + j] for r in rows] for j, s in enumerate(SEVERITY_COLUMNS)},
        "inflow_c": [r[spec.i_inflow] for r in rows],
        **{t: [r[spec.i_rule_trunc + j] for r in rows] for j, t in enumerate(TRUNC_COLUMNS)},
    }
    table = pl.DataFrame(data).cast(spec.table_schema())
    (out_dir / PARTS_DIR).mkdir(parents=True, exist_ok=True)
    for (day,), part in table.group_by("day", maintain_order=True):
        part.write_parquet(out_dir / PARTS_DIR / part_name(int(day)))


def build_features(paths: DataPaths, out_dir: Path, cfgs: dict[str, dict]) -> Path:
    from aml.features.build import run_build_features

    out_dir.mkdir(parents=True, exist_ok=True)
    if os.environ.get("AML_TEST_LOCAL_BUILD") == "1":
        _local_build(paths, out_dir, cfgs)
        return out_dir
    try:
        run_build_features(paths, out_dir, cfgs, mode="full")
    except NotImplementedError:
        _local_build(paths, out_dir, cfgs)
    return out_dir


@pytest.fixture(scope="module")
def volume(prepared, tmp_path_factory) -> DataPaths:
    """A private copy of the prepared fixture: the stage writes reports/parity.md into it."""
    root = tmp_path_factory.mktemp("engine_rules") / "vol"
    shutil.copytree(prepared.root, root)
    return DataPaths(root, prepared.dataset)


@pytest.fixture(scope="module")
def cfgs(rules_cfg, data_cfg) -> dict[str, dict]:
    features = yaml.safe_load((CONFIG_DIR / "features.yaml").read_text(encoding="utf-8"))
    return {"data": data_cfg, "rules": rules_cfg, "features": features}


@pytest.fixture(scope="module")
def features_dir(volume, cfgs, tmp_path_factory) -> Path:
    if not _engine_available():
        pytest.skip("aml.features.engine.Engine is not implemented yet")
    return build_features(volume, tmp_path_factory.mktemp("features"), cfgs)


@pytest.fixture(scope="module")
def m1_dir(volume, cfgs, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("rules_m1")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run_rules_stage(volume, out, cfgs["rules"], cfgs["data"], threads=2)
    return out


@pytest.fixture(scope="module")
def stage(volume, cfgs, features_dir, m1_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("rules_engine")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        summary = run_rules_engine_stage(
            volume,
            features_dir,
            out,
            cfgs["rules"],
            cfgs["data"],
            threads=2,
            m1_rules_dir=m1_dir,
        )
    return out, summary


# ---------------------------------------------------------------------------------------------
# End to end


@needs_engine
def test_parity_is_exact_on_every_row(stage, volume):
    out, summary = stage
    parity = read_json(out / PARITY_FILE)
    assert parity["ok"] is True and parity["failures"] == []
    assert parity["row_ids_equal"] and parity["rank_contiguous"]
    assert parity["rows"] == parity["rows_sql"] == pl.read_parquet(volume.transactions).height
    assert parity["mismatches_total"] == 0
    assert all(m["all"] == 0 for m in parity["mismatches"].values())
    assert parity["rule_trunc"]["rows"] == 0  # the default budget never truncates
    assert parity["hubs"]["equal"]
    assert 0 < parity["inflow_c_max"] < 2**53
    assert not (out / MISMATCHES_FILE).exists()
    assert summary["source"] == "engine" and summary["parity"]["ok"] is True
    assert set(summary["outputs"]) == {*RULES_OUTPUTS, PARITY_FILE}
    report = (volume.reports / PARITY_REPORT).read_text(encoding="utf-8")
    assert "**PASS**" in report and "| fan_in_velocity | 0 |" in report


@needs_engine
def test_thresholds_and_flags_equal_the_m1_stage(stage, m1_dir):
    out, summary = stage
    reg = read_json(out / PARITY_FILE)["m1_regression"]
    assert reg == {
        "compared": True,
        "dir": str(m1_dir),
        "thresholds_equal": True,
        "thresholds_doc_equal": True,
        "flags_equal": True,
        "severities_equal": True,
    }
    assert read_json(out / "thresholds.json") == read_json(m1_dir / "thresholds.json")
    for name in ("flags.parquet", "severities.parquet"):
        assert pl.read_parquet(out / name).equals(pl.read_parquet(m1_dir / name)), name
    m1 = read_json(m1_dir / "summary.json")
    for key in ("thresholds", "metrics", "rows", "stats", "severity_nonzero_share", "hub_cap"):
        assert summary[key] == m1[key], key
    parity = read_json(out / PARITY_FILE)
    for tag, per in parity["flag_agreement"].items():
        assert per == {"val_early": 1.0, "val_late": 1.0, "all": 1.0}, tag
        assert parity["thresholds_equal_sql_tuned"][tag] is True


@needs_engine
def test_tiny_rule_visits_truncated_rows_are_the_only_exclusions(volume, cfgs, m1_dir, tmp_path):
    small = copy.deepcopy(cfgs)
    small["features"]["budgets"]["rule_visits"] = 1
    fdir = build_features(volume, tmp_path / "features", small)
    eng = pl.concat([pl.read_parquet(p) for p in part_paths(fdir)])
    trunc = eng["rule_trunc"].to_numpy() == 1
    assert trunc.any()
    assert (eng["cyc_trunc"].to_numpy()[trunc] == 1).all()  # rule_trunc implies cyc_trunc

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        run_rules_engine_stage(
            volume, fdir, tmp_path / "out", small["rules"], small["data"], threads=2,
            m1_rules_dir=m1_dir,
        )  # fmt: skip
    parity = read_json(tmp_path / "out" / PARITY_FILE)
    assert parity["ok"] is True and parity["mismatches_total"] == 0
    rt = parity["rule_trunc"]
    assert rt["rows"] == int(trunc.sum()) and rt["share"] == pytest.approx(trunc.mean())
    assert sum(rt["per_split"].values()) == rt["rows"]
    # Independently of the stage: only round_trip on truncated rows differs, as a lower bound.
    con = connect(threads=2)
    try:
        register_transactions(con, volume.transactions)
        sql = compute_severities(con, small["rules"], parity["hubs"]["hub_cap_sql"])
    finally:
        con.close()
    for s in SCENARIOS:
        a, b = eng[s].to_numpy(), sql[s].to_numpy()
        if s == "round_trip":
            assert (a[~trunc] == b[~trunc]).all() and (a[trunc] <= b[trunc]).all()
            assert int((a[trunc] < b[trunc]).sum()) == rt["round_trip_lower_than_sql"]
        else:
            assert (a == b).all(), s
    assert rt["round_trip_lower_than_sql"] > 0  # the budget did cut some round trips
    reg = parity["m1_regression"]
    assert reg["compared"]
    if not (reg["thresholds_equal"] and reg["flags_equal"]):
        assert any("truncated rows" in w for w in parity["warnings"])
    report = (volume.reports / PARITY_REPORT).read_text(encoding="utf-8")
    assert f"Truncated rows (`rule_trunc = 1`): {rt['rows']:,}" in report


def _copy_features(src: Path, dst: Path) -> Path:
    shutil.copytree(src, dst)
    return dst


@needs_engine
def test_injected_mismatch_fails_without_flags(volume, cfgs, features_dir, tmp_path):
    fdir = _copy_features(features_dir, tmp_path / "features")
    part = part_paths(fdir)[2]
    df = pl.read_parquet(part)
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) == 3)
        .then(pl.col("fan_in_velocity") + 1.0)
        .otherwise(pl.col("fan_in_velocity"))
        .alias("fan_in_velocity")
    )
    df.write_parquet(part)
    bad_row = df["row_id"][3]
    out = tmp_path / "out"
    out.mkdir()
    for name in RULES_OUTPUTS:  # a stale earlier run must not survive a failed parity
        (out / name).write_text("stale", encoding="utf-8")
    with pytest.raises(ParityError, match="severity mismatches"):
        run_rules_engine_stage(volume, fdir, out, cfgs["rules"], cfgs["data"], threads=2)
    assert not any((out / n).exists() for n in RULES_OUTPUTS)
    parity = read_json(out / PARITY_FILE)
    assert parity["ok"] is False and parity["mismatches_total"] == 1
    assert parity["mismatches"]["fan_in_velocity"]["all"] == 1
    assert sum(m["all"] for m in parity["mismatches"].values()) == 1
    mism = pl.read_parquet(out / MISMATCHES_FILE)
    assert mism.height == 1 and mism["row_id"].item() == bad_row
    row = mism.row(0, named=True)
    assert row["fan_in_velocity_engine"] == row["fan_in_velocity_sql"] + 1.0
    assert "**FAIL**" in (volume.reports / PARITY_REPORT).read_text(encoding="utf-8")


@needs_engine
def test_m1_difference_fails_unless_a_truncated_row_is_below_the_sql(
    volume, cfgs, features_dir, m1_dir, tmp_path
):
    """Truncated rows can explain a difference from M1 only if one of them has a round trip
    below the SQL. Rows flagged rule_trunc = 1 whose severities still equal the SQL's explain
    nothing: a stale or regressed M1 directory still fails the stage."""
    fdir = _copy_features(features_dir, tmp_path / "features")
    part = part_paths(fdir)[2]
    df = pl.read_parquet(part)
    df = df.with_columns(
        pl.when(pl.int_range(pl.len()) < 3)
        .then(pl.lit(1, dtype=df["rule_trunc"].dtype))
        .otherwise(pl.col("rule_trunc"))
        .alias("rule_trunc")
    )
    df.write_parquet(part)
    m1 = _copy_features(m1_dir, tmp_path / "m1")
    thr = read_json(m1 / "thresholds.json")
    tag = next(iter(thr["thresholds"]))
    scen = next(iter(thr["thresholds"][tag]))
    thr["thresholds"][tag][scen] = 12345.0 if thr["thresholds"][tag][scen] != 12345.0 else 1.0
    write_json_atomic(thr, m1 / "thresholds.json")
    out = tmp_path / "out"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with pytest.raises(ParityError, match="equal to the SQL"):
            run_rules_engine_stage(
                volume, fdir, out, cfgs["rules"], cfgs["data"], threads=2, m1_rules_dir=m1
            )
    parity = read_json(out / PARITY_FILE)
    assert (
        parity["rule_trunc"]["rows"] == 3 and parity["rule_trunc"]["round_trip_lower_than_sql"] == 0
    )
    assert parity["m1_regression"]["thresholds_equal"] is False
    assert not any((out / n).exists() for n in RULES_OUTPUTS)


def test_m1_regression_says_why_it_was_not_compared(tmp_path):
    from aml.rules.scenarios import _m1_regression

    reg = _m1_regression(tmp_path, None, "built from other prepared data")
    assert reg == {"compared": False, "dir": None, "reason": "built from other prepared data"}
    eng, sql = _frames()
    doc, _ = compare_severities(eng, sql)
    doc["m1_regression"] = reg
    assert "Not compared: built from other prepared data." in render_parity_md(doc)
    missing = _m1_regression(tmp_path, tmp_path / "nope")
    assert missing["compared"] is False and "not found" in missing["reason"]


@needs_engine
def test_feature_table_from_other_rule_settings_is_refused(volume, cfgs, features_dir, tmp_path):
    other = copy.deepcopy(cfgs["rules"])
    other["scenarios"]["fan_in_velocity"]["window_minutes"] = 720
    with pytest.raises(ValueError, match="other rule settings"):
        run_rules_engine_stage(volume, features_dir, tmp_path, other, cfgs["data"], threads=2)


@needs_engine
def test_hub_list_mismatch_fails(volume, cfgs, features_dir, tmp_path):
    """A feature table built with another hub list (same cap) fails the hub check."""
    fdir = _copy_features(features_dir, tmp_path / "features")
    spec = EngineSpec.from_json(read_json(fdir / FEATURE_SPEC_FILE))
    extra = next(a for a in range(spec.n_accounts) if a not in spec.hubs)
    forged = EngineSpec.from_configs(
        cfgs["features"],
        cfgs["rules"],
        n_accounts=spec.n_accounts,
        vocab=spec.vocab,
        hub_cap=spec.hub_cap,
        hubs=(*spec.hubs, extra),
    )
    write_json_atomic(forged.to_json(), fdir / FEATURE_SPEC_FILE)
    with pytest.raises(ParityError, match="hub"):
        run_rules_engine_stage(volume, fdir, tmp_path / "out", cfgs["rules"], cfgs["data"])
    assert read_json(tmp_path / "out" / PARITY_FILE)["hubs"]["equal"] is False


# ---------------------------------------------------------------------------------------------
# compare_severities and the report on hand-made frames (no engine needed)


def _frames(n: int = 6) -> tuple[pl.DataFrame, pl.DataFrame]:
    splits = ["train", "train", "val_early", "val_early", "val_late", "test"][:n]
    base = {s: [float(k % 3) for k in range(n)] for s in SCENARIOS}
    sql = pl.DataFrame(
        {"row_id": list(range(100, 100 + n)), "split": splits, "day": [1] * n, **base}
    )
    eng = pl.DataFrame(
        {
            "row_id": list(range(100, 100 + n)),
            "rank": list(range(n)),
            "day": [1] * n,
            "split": splits,
            **base,
            "inflow_c": [0, 5, 10, 0, 0, 7],
            "rule_trunc": [0] * n,
        }
    )
    return eng, sql


def _set(df: pl.DataFrame, col: str, i: int, value) -> pl.DataFrame:
    vals = df[col].to_list()
    vals[i] = value
    return df.with_columns(pl.Series(col, vals, dtype=df[col].dtype))


def test_compare_counts_mismatches_per_scenario_and_split():
    eng, sql = _frames()
    doc, rows = compare_severities(eng, sql)
    assert doc["ok"] and not doc["failures"] and rows is None and doc["mismatches_total"] == 0
    assert doc["rows_per_split"] == {"test": 1, "train": 2, "val_early": 2, "val_late": 1}
    assert doc["inflow_c_max"] == 10
    eng = _set(eng, "structuring", 2, 9.0)
    eng = _set(eng, "fan_in_velocity", 2, 9.0)
    eng = _set(eng, "fan_in_velocity", 5, 9.0)
    doc, rows = compare_severities(eng, sql)
    assert doc["mismatches_total"] == 2 and doc["mismatches_validation"] == 1
    assert doc["mismatches"]["fan_in_velocity"]["per_split"] == {
        "test": 1,
        "train": 0,
        "val_early": 1,
        "val_late": 0,
    }
    assert doc["mismatches"]["fan_in_velocity"]["validation"] == 1
    assert doc["mismatches"]["structuring"]["all"] == 1
    assert rows is not None and rows["row_id"].to_list() == [102, 105]
    assert rows["fan_in_velocity_engine"].to_list() == [9.0, 9.0]
    assert rows["fan_in_velocity_sql"].to_list() == [2.0, 2.0]
    assert doc["failures"] and "severity mismatches" in doc["failures"][0]


def test_compare_truncated_rows_keep_a_lower_bound_only():
    eng, sql = _frames()
    eng = _set(eng, "rule_trunc", 1, 1)
    eng = _set(eng, "rule_trunc", 4, 1)
    eng = _set(eng, "round_trip", 1, 0.0)  # below the SQL's 1.0: excluded, reported
    doc, _ = compare_severities(eng, sql)
    assert doc["mismatches_total"] == 0 and not doc["failures"]
    assert doc["rule_trunc"]["rows"] == 2 and doc["rule_trunc"]["round_trip_lower_than_sql"] == 1
    assert doc["rule_trunc"]["per_split"]["train"] == 1
    # A truncated row may never exceed the SQL, and its other severities are checked.
    over = _set(eng, "round_trip", 4, 5.0)
    assert compare_severities(over, sql)[0]["mismatches"]["round_trip"]["all"] == 1
    other = _set(eng, "structuring", 1, 7.0)
    assert compare_severities(other, sql)[0]["mismatches"]["structuring"]["all"] == 1


def test_compare_structural_failures():
    eng, sql = _frames()
    doc, rows = compare_severities(eng, sql.reverse())
    assert not doc["row_ids_equal"] and rows is None and doc["failures"]
    doc, _ = compare_severities(_set(eng, "rank", 0, 7), sql)
    assert not doc["rank_contiguous"] and doc["failures"]
    doc, _ = compare_severities(_set(eng, "inflow_c", 0, 2**53), sql)
    assert any("2^53" in f for f in doc["failures"])
    doc, _ = compare_severities(_set(eng, "structuring", 0, None), sql)
    assert doc["null_columns"] == ["structuring"] and doc["mismatches"]["structuring"]["all"] == 1
    doc, _ = compare_severities(_set(eng, "structuring", 0, float("nan")), sql)
    assert doc["mismatches"]["structuring"]["all"] == 1  # NaN never equals


def test_render_parity_md():
    eng, sql = _frames()
    doc, _ = compare_severities(eng, sql)
    text = render_parity_md(doc)
    assert text.startswith("# Rule parity") and "**PASS**" in text
    assert "| round_trip | 0 | 0 |" in text and "| val_early | 2 | 0 |" in text
    doc["failures"].append("something broke")
    doc["ok"] = False
    doc["flag_agreement"] = {"0p005": {"val_early": 1.0, "val_late": 0.5, "all": None}}
    doc["thresholds_equal_sql_tuned"] = {"0p005": False}
    doc["m1_regression"] = {"compared": False, "dir": None}
    text = render_parity_md(doc)
    assert "**FAIL**" in text and "- something broke" in text
    assert "| 0p005 | False | 1.000000 | 0.500000 | n/a |" in text
    assert "Not compared" in text
