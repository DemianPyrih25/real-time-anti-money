"""The serving bundle (M2 spec §8.6) on the synthetic fixture, with the real engine.

The features come from a real `run_build_features`. The rules_engine and lgbm_graph stage outputs
are built here in their documented layouts (thresholds.json / flags.parquet as M1's rules stage
writes them; booster_s0.txt, scores.parquet, feature_names.json, gate.json, summary.json as the
lgbm_graph stage writes them), so the bundle's own checks can be steered (tampered references,
other data, other feature tables). tests/test_pipeline_m2_e2e.py exports the real stages' outputs.
"""

from __future__ import annotations

import copy
import json
import math
import shutil
from pathlib import Path

import lightgbm as lgb
import numpy as np
import polars as pl
import pytest

from aml.config import rate_tag
from aml.eval.operating_points import threshold_for_alert_rate
from aml.features import build
from aml.features.spec import (
    DRIVER_COLUMNS,
    INPUT_COLUMNS,
    SEVERITY_COLUMNS,
    TRUNC_COLUMNS,
    scan_feature_table,
)
from aml.io import read_json, write_json_atomic, write_parquet_atomic
from aml.models.lgbm import load_labels, predict, save_booster
from aml.paths import DataPaths
from aml.rules.sql_baseline import SCENARIOS, apply_thresholds
from aml.serving import bundle
from tests.conftest import load_yaml

NAMES = [
    "log_amount_usd",
    "payment_format",
    "hour_of_day",
    "u_out_cnt_1d",
    "v_in_cnt_1d",
    "u_out_mean_1d",
]
RULE_THRESHOLDS = {  # rate -> fan_in_velocity threshold (other scenarios off)
    0.005: 3.0,
    0.001: 5.0,
    0.01: 2.0,
}
KEYS = {"data": "data-t", "features": "features-t", "lgbm_graph": "lgbm_graph-t", "export": "e"}


# --- stage outputs in their documented layouts ---------------------------------------------------


def make_rules_dir(features_dir: Path, out: Path) -> dict:
    """thresholds.json + flags.parquet in the M1 rules-stage layout, from the part severities."""
    sev = scan_feature_table(features_dir, ["row_id", "split", "day", *SEVERITY_COLUMNS]).collect()
    tags = [rate_tag(r) for r in RULE_THRESHOLDS]
    thresholds = {
        rate_tag(r): {s: (t if s == "fan_in_velocity" else None) for s in SCENARIOS}
        for r, t in RULE_THRESHOLDS.items()
    }
    head = tags[0]
    fired = apply_thresholds(sev, thresholds[head])
    flags = sev.select("row_id", "split", "day").with_columns(
        *[apply_thresholds(sev, thresholds[t])["any"].alias(f"rules_any_{t}") for t in tags],
        *[fired[f"fired_{s}"] for s in SCENARIOS],
    )
    write_parquet_atomic(flags, out / "flags.parquet")
    doc = {"headline_rate_tag": head, "rate_tags": tags, "thresholds": thresholds}
    write_json_atomic(doc, out / "thresholds.json")
    spec_hash = build.load_spec(features_dir).spec_hash()
    write_json_atomic({"rows": sev.height, "spec_hash": spec_hash}, out / "summary.json")
    return doc


def make_graph_dir(paths: DataPaths, features_dir: Path, out: Path) -> lgb.Booster:
    """A small LightGBM on float32 model inputs, early-stopped on val_early, and its scores."""
    t = scan_feature_table(features_dir, ["row_id", "split", *NAMES]).collect()
    tr, va = t.filter(pl.col("split") == "train"), t.filter(pl.col("split") == "val_early")
    params = {
        "objective": "binary",
        "num_leaves": 7,
        "learning_rate": 0.1,
        "min_data_in_leaf": 5,
        "verbose": -1,
        "deterministic": True,
        "force_row_wise": True,
        "seed": 0,
        "num_threads": 2,
        "metric": "average_precision",
    }

    def ds(df: pl.DataFrame, ref=None) -> lgb.Dataset:
        y = load_labels(paths.labels, df["row_id"])
        x = df.select(NAMES).to_numpy().astype(np.float32)
        return lgb.Dataset(
            x, y, feature_name=NAMES, categorical_feature=["payment_format"], reference=ref
        )

    dtr = ds(tr)
    booster = lgb.train(
        params,
        dtr,
        num_boost_round=40,
        valid_sets=[ds(va, dtr)],
        callbacks=[lgb.early_stopping(5, verbose=False)],
    )
    save_booster(booster, out / "booster_s0.txt")
    scored = t.filter(pl.col("split").is_in(["val_early", "val_late", "test"]))
    x = scored.select(NAMES).to_numpy().astype(np.float32)
    scores = scored.select("row_id", "split").with_columns(
        pl.Series("score_s0", predict(booster, x, threads=2)),
        pl.Series("score_s1", predict(booster, x, threads=1) * 0.5),
    )
    write_parquet_atomic(scores, out / "scores.parquet")
    write_json_atomic(NAMES, out / "feature_names.json")
    write_json_atomic({"kept": NAMES, "dropped": []}, out / "gate.json")
    spec_hash = build.load_spec(features_dir).spec_hash()
    summary = {"variant": "full", "ablation": {"champion": "full"}, "spec_hash": spec_hash}
    write_json_atomic(summary, out / "summary.json")  # the lgbm_graph stage's layout
    return booster


def make_inputs(paths: DataPaths, features_dir: Path, root: Path) -> dict:
    rules_dir, graph_dir = root / "rules_engine" / "k", root / "lgbm_graph" / "k"
    rules = make_rules_dir(features_dir, rules_dir)
    make_graph_dir(paths, features_dir, graph_dir)
    for d in (features_dir, rules_dir, graph_dir):
        write_json_atomic({"data_version": "v1"}, d / "data_version.json")
    return {"features_dir": features_dir, "rules_dir": rules_dir, "graph_dir": graph_dir, **rules}


# --- fixtures ----------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def paths(prepared: DataPaths, tmp_path_factory: pytest.TempPathFactory) -> DataPaths:
    p = DataPaths(tmp_path_factory.mktemp("export_volume"))
    for name in ("transactions", "accounts", "fx_rates", "labels"):
        dst = getattr(p, name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(getattr(prepared, name), dst)
    return p


@pytest.fixture(scope="module")
def cfgs(data_cfg: dict, rules_cfg: dict) -> dict[str, dict]:
    return {
        "data": copy.deepcopy(data_cfg),
        "rules": copy.deepcopy(rules_cfg),
        "features": load_yaml("features.yaml"),
        "serving": load_yaml("serving.yaml"),
    }


@pytest.fixture(scope="module")
def world(paths, cfgs) -> dict:
    """Real-engine features + test-built rules and model outputs, and one exported bundle."""
    features_dir = paths.features_dir("features-real")
    build.run_build_features(paths, features_dir, cfgs)
    inputs = make_inputs(paths, features_dir, paths.root / "models")
    summary = bundle.run_export(
        paths,
        paths.serving_dir,
        cfgs,
        KEYS,
        features_dir=inputs["features_dir"],
        rules_dir=inputs["rules_dir"],
        graph_dir=inputs["graph_dir"],
    )
    return {**inputs, "summary": summary}


def _export(paths, cfgs, world, bundle_dir=None, **kw):
    return bundle.run_export(
        paths,
        bundle_dir or paths.serving_dir,
        cfgs,
        KEYS,
        features_dir=world["features_dir"],
        rules_dir=world["rules_dir"],
        graph_dir=world["graph_dir"],
        **kw,
    )


# --- the bundle ---------------------------------------------------------------------------------


def test_bundle_has_every_file_and_metadata_last(world, paths, cfgs):
    d = paths.serving_dir
    files = sorted(p.relative_to(d).as_posix() for p in d.rglob("*") if p.is_file())
    assert files == sorted([*bundle.BUNDLE_FILES, bundle.METADATA_FILE])
    meta = read_json(d / bundle.METADATA_FILE)
    assert set(meta["files"]) == set(bundle.BUNDLE_FILES)
    for rel, info in meta["files"].items():
        assert (
            info["sha256"] == bundle._sha256(d / rel) and info["bytes"] == (d / rel).stat().st_size
        )
    newest = max((d / rel).stat().st_mtime_ns for rel in bundle.BUNDLE_FILES)
    assert (d / bundle.METADATA_FILE).stat().st_mtime_ns >= newest
    tx = pl.read_parquet(paths.transactions)
    test = tx.filter(pl.col("split") == "test").sort("rank")
    n = min(cfgs["serving"]["replay"]["max_events"], test.height)
    spec = build.load_spec(world["features_dir"])
    assert meta["bundle_format"] == bundle.BUNDLE_FORMAT == 1
    assert meta["keys"] == KEYS and meta["data_version"] == "v1"
    assert meta["spec_hash"] == spec.spec_hash() and meta["engine_version"] == 1
    assert meta["next_rank"] == int(test["rank"][0]) and meta["next_offset"] == 0
    assert meta["rows"]["slice"] == n and meta["rows"]["model_inputs"] == len(NAMES)
    assert meta["verification"]["mismatches"] == 0 and meta["verification"]["rows"] == n
    assert meta["champion"] == "full" and meta["snapshot_state_digest"]
    assert meta["libraries"]["lightgbm"] == lgb.__version__
    assert world["summary"]["verification"] == meta["verification"]


def test_metadata_records_platform_oracle_and_parts(world, paths, cfgs, tmp_path):
    """M5's bit-exact parity depends on the host's libm (log1p) as well as the libraries:
    metadata.json records where the references were computed and a log1p fingerprint over the
    slice amounts that M5 can recompute; also the oracle verdict and the parts' digest."""
    import platform
    import sys

    from aml.features.spec import parts_digest

    meta = read_json(paths.serving_dir / bundle.METADATA_FILE)
    p = meta["platform"]
    assert p["sys_platform"] == sys.platform and p["machine"] == platform.machine()
    amounts = pl.read_parquet(paths.serving_dir / bundle.SLICE)["amount_usd"].to_list()
    assert p["log1p_probe"] == {
        "values": f"sorted distinct {bundle.SLICE} amount_usd",
        **bundle.log1p_fingerprint(amounts),
    }
    assert bundle.log1p_fingerprint(amounts[::-1] + amounts[:5]) == bundle.log1p_fingerprint(
        amounts
    )
    assert bundle.log1p_fingerprint([0.5])["sha256"] != bundle.log1p_fingerprint([0.25])["sha256"]
    assert meta["features_digest"] == parts_digest(world["features_dir"])
    assert meta["features_verify"] is None  # run_export called directly: no oracle summary given
    verdict = {"n_mismatches_total": 0, "rows": 5, "spec_hash": meta["spec_hash"]}
    _export(paths, cfgs, world, bundle_dir=tmp_path / "b", features_verify=verdict)
    assert read_json(tmp_path / "b" / bundle.METADATA_FILE)["features_verify"] == verdict


def test_slice_and_references(world, paths):
    d = paths.serving_dir
    tx = pl.read_parquet(paths.transactions)
    sl = pl.read_parquet(d / bundle.SLICE)
    assert sl.columns == [*INPUT_COLUMNS, *DRIVER_COLUMNS]  # no labels, no extra columns
    want = tx.filter(pl.col("split") == "test").sort("rank").head(sl.height).select(sl.columns)
    assert sl.equals(want)
    ref = pl.read_parquet(d / bundle.REF_FEATURES)
    assert ref.columns == ["row_id", "rank", *NAMES, *SEVERITY_COLUMNS, *TRUNC_COLUMNS]
    assert ref["row_id"].equals(sl["row_id"]) and ref["rank"].equals(sl["rank"])
    parts = scan_feature_table(world["features_dir"], ref.columns).collect()
    assert ref.equals(parts.filter(pl.col("row_id").is_in(sl["row_id"].implode())).sort("rank"))
    sc = pl.read_parquet(d / bundle.REF_SCORES)
    assert sc.columns == ["row_id", "score_s0"]
    offline = pl.read_parquet(world["graph_dir"] / "scores.parquet")
    assert sc.equals(sl.select("row_id").join(offline.select(sc.columns), on="row_id"))
    al = pl.read_parquet(d / bundle.REF_ALERTS)
    tags = world["rate_tags"]
    assert al.columns == [
        "row_id",
        *[f"alert_{t}" for t in tags],
        *[f"fired_{s}" for s in SCENARIOS],
        *[f"rules_any_{t}" for t in tags],
    ]
    flags = pl.read_parquet(world["rules_dir"] / "flags.parquet")
    cols = [c for c in al.columns if c.startswith(("fired_", "rules_any_"))]
    assert al.select("row_id", *cols).equals(
        sl.select("row_id").join(flags.select("row_id", *cols), on="row_id")
    )
    thr = read_json(d / bundle.THRESHOLDS)["model"]
    for t in tags:
        want_alert = sc["score_s0"] >= thr[t]["threshold"] if thr[t]["threshold"] else False
        assert (al[f"alert_{t}"] == want_alert).all()


def test_thresholds_and_calibration_from_val_late(world, paths):
    d = paths.serving_dir
    doc = read_json(d / bundle.THRESHOLDS)
    scores = pl.read_parquet(world["graph_dir"] / "scores.parquet")
    flags = pl.read_parquet(world["rules_dir"] / "flags.parquet")
    val = scores.filter(pl.col("split") == "val_late").join(flags, on="row_id", how="left")
    s = val["score_s0"].to_numpy()
    assert doc["headline_rate_tag"] == world["headline_rate_tag"]
    assert doc["rate_tags"] == world["rate_tags"] and doc["val_late_rows"] == val.height
    for t in doc["rate_tags"]:
        rate = float(val[f"rules_any_{t}"].mean())
        want = threshold_for_alert_rate(s, rate)
        got = doc["model"][t]["threshold"]
        assert (got is None and math.isinf(want)) or got == want
        assert doc["model"][t]["rules_val_late_rate"] == rate
    assert doc["rules"]["thresholds"] == world["thresholds"]
    y = load_labels(paths.labels, val["row_id"])
    assert doc["val_late_positives"] == int(y.sum())
    cal = read_json(d / bundle.CALIBRATION)
    assert cal["fit_split"] == "val_late" and cal["out_of_bounds"] == "clip"
    assert np.all(np.diff(cal["x"]) >= 0) and np.all(np.diff(cal["y"]) >= 0)
    assert 0 <= min(cal["y"]) <= max(cal["y"]) <= 1


def test_model_files(world, paths):
    d = paths.serving_dir
    assert (d / bundle.BOOSTER).read_bytes() == (world["graph_dir"] / "booster_s0.txt").read_bytes()
    doc = read_json(d / bundle.FEATURE_SPEC)
    spec = build.load_spec(world["features_dir"])
    assert {k: v for k, v in doc.items() if k != "model"} == spec.to_json()
    m = doc["model"]
    assert m["feature_names"] == NAMES and m["categorical"] == ["payment_format"]
    assert m["model_index"] == list(spec.model_index(NAMES)) and m["gated"] == NAMES
    assert m["float32_cast"] == bundle.FLOAT32_CAST and m["champion"] == "full"
    assert bundle._champion({"ablation": {"champion": "-SG"}}) == "-SG"
    assert bundle._champion({"champion": "full"}) == "full" and bundle._champion({}) is None
    pre = read_json(d / bundle.PREPROCESS)
    assert pre["normalisation_stats"] is None and pre["hubs"] == list(spec.hubs)
    assert pre["fx"] == read_json(paths.fx_rates) and pre["vocab"] == read_json(
        world["features_dir"] / "vocab.json"
    )
    side = read_json(d / bundle.SNAPSHOT_SIDECAR)
    assert side["next_offset"] == 0
    assert (
        bundle.read_snapshot_header(d / bundle.SNAPSHOT)["next_rank"]
        == read_json(d / bundle.METADATA_FILE)["next_rank"]
    )


def test_verify_bundle_rechecks_files(world, paths, cfgs, tmp_path):
    d = tmp_path / "serving"
    shutil.copytree(paths.serving_dir, d)
    out = bundle.verify_bundle(d)
    assert out["metadata"] and out["files_checked"] == len(bundle.BUNDLE_FILES)
    al = pl.read_parquet(d / bundle.REF_ALERTS)
    col = next(c for c in al.columns if c.startswith("alert_"))
    write_parquet_atomic(al.with_columns(~pl.col(col)), d / bundle.REF_ALERTS)
    with pytest.raises(bundle.BundleVerificationError, match="alerts"):
        bundle.verify_bundle(d)


@pytest.mark.parametrize("what", ["features", "scores", "severities"])
def test_tampered_reference_fails_before_metadata(world, paths, cfgs, tmp_path, monkeypatch, what):
    real = bundle.build_references

    def tampered(*args, **kw):
        feats, scores, alerts = real(*args, **kw)
        i = pl.int_range(pl.len()) == 7
        if what == "features":  # one model input, one row, the smallest float32 step
            v = feats["u_out_cnt_1d"].to_numpy().copy()
            v[7] = np.nextafter(v[7], np.float32(np.inf))
            feats = feats.with_columns(pl.Series("u_out_cnt_1d", v))
        elif what == "scores":
            v = scores["score_s0"].to_numpy().copy()
            v[7] = np.nextafter(v[7], 1.0)
            scores = scores.with_columns(pl.Series("score_s0", v))
        else:
            s = SEVERITY_COLUMNS[0]
            feats = feats.with_columns(pl.when(i).then(pl.col(s) + 1).otherwise(pl.col(s)))
        return feats, scores, alerts

    monkeypatch.setattr(bundle, "build_references", tampered)
    d = tmp_path / "serving"
    with pytest.raises(bundle.BundleVerificationError, match=what):
        _export(paths, cfgs, world, bundle_dir=d)
    assert (d / bundle.REF_FEATURES).exists() and not (d / bundle.METADATA_FILE).exists()


def test_old_contents_removed_and_inputs_protected(world, paths, cfgs, tmp_path):
    d = tmp_path / "serving"
    (d / "model").mkdir(parents=True)
    (d / "model" / "booster_s9.txt").write_text("old", encoding="utf-8")
    (d / bundle.METADATA_FILE).write_text("{}", encoding="utf-8")
    c = copy.deepcopy(cfgs)
    c["serving"]["replay"]["max_events"] = 50
    s = _export(paths, c, world, bundle_dir=d)
    assert not (d / "model" / "booster_s9.txt").exists() and s["rows"] == 50
    assert read_json(d / bundle.METADATA_FILE)["rows"]["slice"] == 50
    for bad in (paths.root, world["features_dir"], world["graph_dir"].parent):
        with pytest.raises(ValueError, match="refusing"):
            _export(paths, c, world, bundle_dir=bad)
    assert (world["features_dir"] / "summary.json").exists()
    other = tmp_path / "other"
    (other / "keep").mkdir(parents=True)
    with pytest.raises(ValueError, match="non-bundle entries"):
        _export(paths, c, world, bundle_dir=other)
    assert (other / "keep").exists()


def test_inputs_from_other_data_are_refused(world, paths, cfgs, tmp_path):
    rules = tmp_path / "rules"
    shutil.copytree(world["rules_dir"], rules)
    write_json_atomic({"data_version": "v2"}, rules / "data_version.json")
    other = {**world, "rules_dir": rules}
    with pytest.raises(ValueError, match="different prepared data"):
        _export(paths, cfgs, other, bundle_dir=tmp_path / "serving")
    # Rules or model outputs built from another feature table are refused before any write.
    for kind in ("rules_dir", "graph_dir"):
        d = tmp_path / kind
        shutil.copytree(world[kind], d)
        doc = read_json(d / "summary.json")
        write_json_atomic({**doc, "spec_hash": "0" * 16}, d / "summary.json")
        with pytest.raises(ValueError, match="feature spec"):
            _export(paths, cfgs, {**world, kind: d}, bundle_dir=tmp_path / f"serving-{kind}")
        assert not (tmp_path / f"serving-{kind}").exists()


def test_snapshot_header_reader(tmp_path):
    p = tmp_path / "x.snap"
    p.write_bytes(b"NOTASNAP" + b"\0" * 8)
    with pytest.raises(ValueError, match="not an engine snapshot"):
        bundle.read_snapshot_header(p)
    hb = json.dumps({"next_rank": 5}).encode()
    p.write_bytes(bundle.MAGIC + len(hb).to_bytes(4, "little") + hb + b"payload")
    assert bundle.read_snapshot_header(p) == {"next_rank": 5}
