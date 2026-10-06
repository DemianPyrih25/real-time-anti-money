"""A synthetic serving bundle without Modal: the M2 export on the tiny fixture, real engine.

The rules_engine and lgbm_graph stage outputs are built here in their documented layouts
(thresholds.json / flags.parquet as M1's rules stage writes them; booster_s0.txt, scores.parquet,
feature_names.json, gate.json, summary.json as the lgbm_graph stage writes them), so tests can
steer them. tests/unit/test_export.py uses the builders; tests/conftest.py exports one shared
bundle for the M5 tests; `python -m tests.fixtures.serving_bundle --out DIR` writes one for the
demo (`make bundle-fixture`). No pytest import: the module also runs as a script.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl
import yaml

from aml.config import rate_tag
from aml.features import build
from aml.features.spec import SEVERITY_COLUMNS, scan_feature_table
from aml.io import read_json, write_json_atomic, write_parquet_atomic
from aml.models.lgbm import load_labels, predict, save_booster
from aml.paths import DataPaths
from aml.rules.sql_baseline import SCENARIOS, apply_thresholds
from aml.serving import bundle
from aml.serving.settings import Settings, load_settings
from tests.fixtures.synthetic import make_synthetic

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = REPO_ROOT / "configs"
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
FIXTURE_EXPORT_KEY = "fixture"  # metadata.keys.export of every bundle built here
KEYS = {
    "data": "data-t",
    "features": "features-t",
    "lgbm_graph": "lgbm_graph-t",
    "export": FIXTURE_EXPORT_KEY,
}
TABLES = ("transactions", "accounts", "fx_rates", "labels")


def load_yaml(name: str) -> dict:
    with (CONFIG_DIR / name).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


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


# --- the world and the bundle --------------------------------------------------------------------


def fixture_configs(expected: dict[str, int]) -> dict[str, dict]:
    """The repo configs, with data.yaml's expected counts replaced by the fixture's."""
    data = load_yaml("data.yaml")
    data["expected"] = dict(expected)
    return {
        "data": data,
        "rules": load_yaml("rules.yaml"),
        "features": load_yaml("features.yaml"),
        "serving": load_yaml("serving.yaml"),
    }


def copy_prepared(prepared: DataPaths, root: Path) -> DataPaths:
    """A private Volume layout holding copies of the prepared tables."""
    paths = DataPaths(Path(root))
    for name in TABLES:
        dst = getattr(paths, name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(getattr(prepared, name), dst)
    return paths


def prepare_synthetic(root: Path) -> tuple[DataPaths, dict[str, dict]]:
    """The synthetic dataset through the real prepare_data stage (no EDA report)."""
    from aml.data.prepare import prepare_data

    syn = make_synthetic(Path(root) / "synthetic")
    cfgs = fixture_configs(syn.expected)
    paths = DataPaths(Path(root) / "volume")
    paths.raw_dir.mkdir(parents=True, exist_ok=True)
    for f in (syn.transactions_csv, syn.patterns_txt):
        shutil.copy(f, paths.raw_dir / f.name)
    prepare_data(paths, cfgs["data"], threads=2, run_eda=False)
    return paths, cfgs


def build_world(paths: DataPaths, cfgs: dict[str, dict]) -> dict[str, Any]:
    """Real-engine features plus test-built rules and model outputs on prepared tables."""
    features_dir = paths.features_dir("features-fixture")
    build.run_build_features(paths, features_dir, cfgs)
    inputs = make_inputs(paths, features_dir, paths.root / "models")
    return {"paths": paths, "cfgs": cfgs, **inputs}


def build_fixture_bundle(world: dict[str, Any], out: Path) -> dict[str, Any]:
    """Export the world's bundle into `out` (metadata.keys.export = "fixture").

    Refuses an `out` that holds a bundle with another export key (a pulled real bundle)."""
    out = Path(out)
    meta_path = out / bundle.METADATA_FILE
    if meta_path.exists():
        found = (read_json(meta_path).get("keys") or {}).get("export")
        if found != FIXTURE_EXPORT_KEY:
            raise ValueError(
                f"{out} holds a serving bundle with export key {found!r}: refusing to overwrite "
                "it with the synthetic fixture"
            )
    return bundle.run_export(
        world["paths"],
        out,
        world["cfgs"],
        KEYS,
        features_dir=world["features_dir"],
        rules_dir=world["rules_dir"],
        graph_dir=world["graph_dir"],
    )


# --- checks the M5 tests rely on -----------------------------------------------------------------


def alert_counts(bundle_dir: Path) -> dict[str, int]:
    """Model alerts in the slice per rate tag (the reference alert columns)."""
    tags = read_json(Path(bundle_dir) / bundle.THRESHOLDS)["rate_tags"]
    al = pl.read_parquet(Path(bundle_dir) / bundle.REF_ALERTS)
    return {t: int(al[f"alert_{t}"].sum()) for t in tags}


def pick_tag(bundle_dir: Path) -> str:
    """The headline tag if it alerts at least once in the slice, else the tag alerting most."""
    counts = alert_counts(bundle_dir)
    head = read_json(Path(bundle_dir) / bundle.THRESHOLDS)["headline_rate_tag"]
    return head if counts[head] >= 1 else max(counts, key=lambda t: counts[t])


def fixture_settings(
    bundle_dir: Path,
    runtime_dir: Path,
    alert_rate_tag: str,
    *,
    transport: str = "inproc",
    **cli: Any,
) -> Settings:
    """Settings of a test run on a fixture bundle: the shipped serving.yaml, no environment;
    `cli` holds further argparse dest names (max_events, exit_at_end, bootstrap, ...)."""
    args = {
        "bundle": bundle_dir,
        "runtime": runtime_dir,
        "reports": Path(runtime_dir).parent / "reports",
        "config": CONFIG_DIR / "serving.yaml",
        "alert_rate_tag": alert_rate_tag,
        "transport": transport,
        **cli,
    }
    return load_settings(args, {})


def check_fixture(bundle_dir: Path) -> list[str]:
    """What makes the fixture useless for the M5 tests (empty when it is fine)."""
    problems = []
    if not any(alert_counts(bundle_dir).values()):
        problems.append("no model alert in the slice at any rate tag")
    minutes = pl.read_parquet(Path(bundle_dir) / bundle.SLICE, columns=["minute"])["minute"]
    if minutes.n_unique() == minutes.len():
        problems.append("no minute with >= 2 events in the slice")
    return problems


def main(argv: list[str] | None = None) -> int:
    """python -m tests.fixtures.serving_bundle --out DIR [--work DIR]"""
    p = argparse.ArgumentParser(prog="python -m tests.fixtures.serving_bundle")
    p.add_argument("--out", required=True, type=Path, help="bundle directory to (re)write")
    p.add_argument("--work", type=Path, help="keep the intermediate files here (default: temp)")
    args = p.parse_args(argv)
    meta_path = args.out / bundle.METADATA_FILE
    if meta_path.exists():  # refuse before the (slower) build, too
        found = (read_json(meta_path).get("keys") or {}).get("export")
        if found != FIXTURE_EXPORT_KEY:
            print(f"refusing: {args.out} holds a bundle with export key {found!r}", file=sys.stderr)
            return 2
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmp:
        work = args.work or Path(tmp)
        paths, cfgs = prepare_synthetic(work)
        # Export into a staging dir, then copy: --out may be a mount point (the Docker recipe
        # mounts ./serving there), which the export cannot delete and recreate.
        staged = Path(tmp) / "bundle"
        summary = build_fixture_bundle(build_world(paths, cfgs), staged)
        args.out.mkdir(parents=True, exist_ok=True)
        for child in args.out.iterdir():  # only an older fixture bundle can be here
            shutil.rmtree(child) if child.is_dir() else child.unlink()
        for src in staged.iterdir():
            dst = args.out / src.name
            shutil.copytree(src, dst) if src.is_dir() else shutil.copy2(src, dst)
    problems = check_fixture(args.out)
    out = {"bundle_dir": str(args.out), "rows": summary["rows"], "problems": problems}
    print(json.dumps(out, sort_keys=True))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
