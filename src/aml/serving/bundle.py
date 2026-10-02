"""The serving bundle: model, feature spec, thresholds, calibration, state snapshot, replay slice
and offline reference outputs, verified before metadata.json is written (M2 spec §8.6).

Layout of `paths.serving_dir` (old contents removed first; metadata.json last, so a bundle
without it is incomplete): model/booster_s0.txt, model/feature_spec.json, model/preprocess.json,
model/thresholds.json, model/calibration.json, state/test_boundary.snap (+ .json),
replay/slice.parquet, reference/features.parquet, reference/scores.parquet,
reference/alerts.parquet, metadata.json.

Verification (any failure raises and leaves no metadata.json): restore the snapshot, replay the
slice through `Engine.process` + a final `advance`; the model inputs after the float32 cast equal
the reference bit for bit, so do the severities, trunc flags and rule flags;
`booster.predict(x_float32, num_threads=1)` equals the offline `score_s0` bit for bit and gives
the reference alerts; row-by-row predict equals batch predict for the first 1,000 rows.

Labels: only `is_laundering` of val_late rows is read, for the F1 threshold and the isotonic
calibration (PLAN §4: val_late = thresholds and calibration). Test labels are never read.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import struct
import sys
import time
from array import array
from importlib import metadata
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from aml.eval.metrics import best_f1_threshold
from aml.eval.operating_points import threshold_for_alert_rate
from aml.features.build import load_spec, snapshot_file
from aml.features.engine import Engine
from aml.features.snapshot import MAGIC, SIDECAR_SUFFIX
from aml.features.spec import (
    DRIVER_COLUMNS,
    ENGINE_VERSION,
    FEATURES_DIGEST,
    INPUT_COLUMNS,
    SEVERITY_COLUMNS,
    SNAPSHOTS_DIR,
    SUMMARY_FILE,
    TRUNC_COLUMNS,
    EngineSpec,
    scan_feature_table,
)
from aml.io import read_json, write_json_atomic, write_parquet_atomic
from aml.models.lgbm import load_labels, score_column
from aml.paths import DataPaths
from aml.rules.sql_baseline import SCENARIOS, apply_thresholds

BUNDLE_FORMAT = 1
METADATA_FILE = "metadata.json"
SEED = 0
SNAPSHOT_SPLIT = "test"  # the bundle replays the start of the test period
CALIBRATION_SPLIT = "val_late"
ROW_BY_ROW_ROWS = 1000
FLOAT32_CAST = "np.asarray(values, np.float64).astype(np.float32)"
PREDICT = "booster.predict(x_float32, num_threads=1)"
DATA_VERSION_FILE = "data_version.json"  # stamped by the Modal jobs into every stage directory

BOOSTER = "model/booster_s0.txt"
FEATURE_SPEC = "model/feature_spec.json"
PREPROCESS = "model/preprocess.json"
THRESHOLDS = "model/thresholds.json"
CALIBRATION = "model/calibration.json"
SNAPSHOT = f"state/{snapshot_file(SNAPSHOT_SPLIT)}"
SNAPSHOT_SIDECAR = SNAPSHOT + SIDECAR_SUFFIX
SLICE = "replay/slice.parquet"
REF_FEATURES = "reference/features.parquet"
REF_SCORES = "reference/scores.parquet"
REF_ALERTS = "reference/alerts.parquet"
BUNDLE_FILES = (
    BOOSTER,
    FEATURE_SPEC,
    PREPROCESS,
    THRESHOLDS,
    CALIBRATION,
    SNAPSHOT,
    SNAPSHOT_SIDECAR,
    SLICE,
    REF_FEATURES,
    REF_SCORES,
    REF_ALERTS,
)
SCORE = score_column(SEED)  # "score_s0"
# Top-level names inside a bundle directory (the only entries export may delete).
_BUNDLE_ENTRIES = {*(f.split("/")[0] for f in BUNDLE_FILES), METADATA_FILE}
_LIBRARIES = ("lightgbm", "numpy", "polars", "pyarrow", "duckdb", "scikit-learn", "pandera")


class BundleVerificationError(RuntimeError):
    """The bundle does not reproduce its reference outputs (no metadata.json is written)."""


# --- small helpers ------------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _copy_atomic(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(f".{dst.name}.tmp-{os.getpid()}")
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def read_snapshot_header(path: Path) -> dict[str, Any]:
    """The JSON header of a snapshot file (M2 spec §7 framing), without reading the payload."""
    with open(path, "rb") as f:
        head = f.read(len(MAGIC) + 4)
        if len(head) < len(MAGIC) + 4 or head[: len(MAGIC)] != MAGIC:
            raise ValueError(f"{path} is not an engine snapshot")
        (n,) = struct.unpack("<I", head[len(MAGIC) :])
        return json.loads(f.read(n).decode("utf-8"))


def _finite_or_none(x: float) -> float | None:
    return float(x) if x is not None and math.isfinite(x) else None


def _data_version(d: Path) -> tuple[bool, str | None]:
    p = Path(d) / DATA_VERSION_FILE
    return (True, read_json(p).get("data_version")) if p.exists() else (False, None)


def _library_versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": sys.version.split()[0]}
    for name in _LIBRARIES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def log1p_fingerprint(values: Any) -> dict[str, Any]:
    """SHA-256 of `math.log1p` (little-endian float64 bits) over the sorted distinct `values`.

    math.log1p is the one libm call in the feature path that IEEE 754 does not pin down
    (sqrt is exact), so two hosts give bit-equal features only if their libm agrees on it. M5
    recomputes this over the bundle's slice amounts: a different digest means bit-exact parity
    is not expected on that host and the tolerance classes (M2 spec §4.9) apply.
    """
    xs = sorted({float(x) for x in values})
    h = hashlib.sha256()
    for x in xs:
        h.update(struct.pack("<d", math.log1p(x)))
    return {"n": len(xs), "sha256": h.hexdigest()}


def platform_info(slice_amounts: Any) -> dict[str, Any]:
    """Where the reference outputs were computed: OS, machine, libc and the log1p fingerprint
    over the replay slice's distinct `amount_usd` values."""
    libc, libc_version = platform.libc_ver()
    return {
        "sys_platform": sys.platform,
        "machine": platform.machine(),
        "libc": libc or None,
        "libc_version": libc_version or None,
        "python_implementation": platform.python_implementation(),
        "log1p_probe": {
            "values": f"sorted distinct {SLICE} amount_usd",
            **log1p_fingerprint(slice_amounts),
        },
    }


def _check_bundle_dir(bundle_dir: Path, protected: list[Path]) -> None:
    """The bundle directory is emptied first: refuse one that contains any stage's inputs or
    holds anything but bundle entries (an earlier, possibly incomplete, bundle)."""
    b = bundle_dir.resolve()
    for p in protected:
        q = Path(p).resolve()
        if b == q or b in q.parents:
            raise ValueError(f"bundle_dir {bundle_dir} would contain {p}: refusing to empty it")
    if bundle_dir.exists():
        foreign = sorted(
            e.name
            for e in bundle_dir.iterdir()
            if e.name not in _BUNDLE_ENTRIES and not e.name.startswith(".")
        )
        if foreign:
            raise ValueError(
                f"bundle_dir {bundle_dir} holds non-bundle entries {foreign}: refusing to empty it"
            )


def _champion(graph_summary: dict) -> str | None:
    """The lgbm_graph champion variant: `variant` in its summary.json (also `ablation.champion`)."""
    for v in (
        graph_summary.get("variant"),
        (graph_summary.get("ablation") or {}).get("champion"),
        graph_summary.get("champion"),
    ):
        if v:
            return str(v)
    return None


def _check_same_table(spec_hash: str, summaries: dict[str, Path]) -> None:
    """Stage outputs that record a feature-spec hash must come from this feature table."""
    for name, path in summaries.items():
        if path.exists():
            found = read_json(path).get("spec_hash")
            if found is not None and found != spec_hash:
                raise ValueError(
                    f"{name} outputs were built from feature spec {found}, the features dir holds "
                    f"{spec_hash}: re-run that stage on this feature table"
                )


def _rate_tags(rules_thr: dict) -> tuple[list[str], str]:
    tags = list(rules_thr.get("rate_tags") or rules_thr["thresholds"])
    head = rules_thr["headline_rate_tag"]
    if head not in tags:
        raise ValueError(f"rules thresholds: headline tag {head!r} not in {tags}")
    return tags, head


def rule_flags(
    sev: pl.DataFrame, rules_thresholds: dict, tags: list[str], head: str
) -> pl.DataFrame:
    """row_id + fired_<scenario> (headline rate) + rules_any_<tag>, with M1's apply_thresholds."""
    head_fired = apply_thresholds(sev, rules_thresholds[head])
    cols = [head_fired["row_id"], *(head_fired[f"fired_{s}"] for s in SCENARIOS)]
    for tag in tags:
        cols.append(apply_thresholds(sev, rules_thresholds[tag])["any"].alias(f"rules_any_{tag}"))
    return pl.DataFrame(cols)


def _rows_differ(a: pl.DataFrame, b: pl.DataFrame) -> int:
    """Rows where any column of `a` (row_id excluded) differs from the same column of `b`."""
    cols = [c for c in a.columns if c != "row_id"]
    if not cols:
        return 0
    diff = pl.DataFrame([(a[c] != b[c]).alias(c) for c in cols])
    return int(diff.select(pl.any_horizontal(pl.all())).to_series().sum())


def model_alerts(
    row_id: pl.Series, score: np.ndarray, model_thr: dict, tags: list[str]
) -> pl.DataFrame:
    """row_id + alert_<tag> = score >= the tag's threshold (null threshold = never)."""
    cols = {"row_id": row_id}
    for tag in tags:
        thr = model_thr[tag]["threshold"]
        cols[f"alert_{tag}"] = (
            np.zeros(len(score), dtype=bool) if thr is None else np.asarray(score) >= thr
        )
    return pl.DataFrame(cols)


# --- building -----------------------------------------------------------------------------------


def _model_thresholds(
    paths: DataPaths, scores: pl.DataFrame, flags: pl.DataFrame, tags: list[str], head: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    """thresholds.json (model part) and calibration.json from the seed-0 val_late scores.

    Per rate tag: `threshold_for_alert_rate(val_late scores, rules' val_late rate)` (the M1
    evaluation's deployable threshold); the val_late F1 threshold; isotonic calibration.
    """
    from sklearn.isotonic import IsotonicRegression

    val = scores.filter(pl.col("split") == CALIBRATION_SPLIT).select("row_id", SCORE)
    if val.height == 0:
        raise ValueError(f"no {CALIBRATION_SPLIT} rows in the model scores")
    any_cols = [f"rules_any_{t}" for t in tags]
    val = val.join(
        flags.select("row_id", *any_cols), on="row_id", how="left", maintain_order="left"
    )
    if val.select(pl.any_horizontal(pl.col(any_cols).is_null())).to_series().any():
        raise ValueError(f"rules flags are missing for some {CALIBRATION_SPLIT} rows")
    s = val[SCORE].to_numpy().astype(np.float64)
    y = load_labels(paths.labels, val["row_id"])  # val_late labels only
    model: dict[str, Any] = {}
    for tag in tags:
        rate = float(val[f"rules_any_{tag}"].mean())
        thr = threshold_for_alert_rate(s, rate)
        model[tag] = {
            "threshold": _finite_or_none(thr),
            "rules_val_late_rate": rate,
            "model_val_late_rate": float((s >= thr).mean()),
        }
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(s, y)
    thresholds = {
        "seed": SEED,
        "score": f"{SCORE}: raw LightGBM probability, {PREDICT}",
        "headline_rate_tag": head,
        "rate_tags": tags,
        "model": model,
        "alert_rule": "alert_<tag> = score >= model[tag].threshold (null: no alert at this rate)",
        "val_late_f1": {"threshold": _finite_or_none(best_f1_threshold(y, s))},
        "val_late_rows": int(s.size),
        "val_late_positives": int(np.asarray(y).sum()),
    }
    calibration = {
        "method": "isotonic",
        "fit_split": CALIBRATION_SPLIT,
        "seed": SEED,
        "out_of_bounds": "clip",
        "x": iso.X_thresholds_.tolist(),
        "y": iso.y_thresholds_.tolist(),
        "use": "display only (case packs); every threshold and metric uses the raw score",
    }
    return thresholds, calibration


def build_references(
    features_dir: Path,
    names: list[str],
    slice_df: pl.DataFrame,
    scores: pl.DataFrame,
    flags: pl.DataFrame,
    thresholds: dict[str, Any],
    rules_thresholds: dict[str, Any],
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """(features, scores, alerts) reference frames of the slice, from the offline outputs.

    Features come from the feature-table parts (model inputs Float32, severities, trunc flags);
    scores are the lgbm_graph stage's offline `score_s0`; rule flags are recomputed from the
    part severities with M1's apply_thresholds and must equal the rules stage's flags.
    """
    r0, r1 = int(slice_df["rank"][0]), int(slice_df["rank"][-1])
    cols = ["row_id", "rank", *names, *SEVERITY_COLUMNS, *TRUNC_COLUMNS]
    feats = (
        scan_feature_table(features_dir, cols)
        .filter(pl.col("rank").is_between(r0, r1))
        .collect()
        .sort("rank")
    )
    if not feats["row_id"].equals(slice_df["row_id"]):
        raise ValueError("feature parts do not hold exactly the slice rows")
    keys = slice_df.select("row_id")
    sc = keys.join(scores.select("row_id", SCORE), on="row_id", how="left", maintain_order="left")
    if sc[SCORE].null_count():
        raise ValueError(f"{sc[SCORE].null_count()} slice rows have no offline {SCORE}")
    tags, head = thresholds["rate_tags"], thresholds["headline_rate_tag"]
    rules = rule_flags(feats.select("row_id", *SEVERITY_COLUMNS), rules_thresholds, tags, head)
    stage = keys.join(flags.select(rules.columns), on="row_id", how="left", maintain_order="left")
    if not stage.equals(rules):
        raise ValueError("rule flags from the part severities differ from the rules stage's")
    alerts = model_alerts(sc["row_id"], sc[SCORE].to_numpy(), thresholds["model"], tags).join(
        rules, on="row_id", how="left", maintain_order="left"
    )
    return feats, sc, alerts


def run_export(
    paths: DataPaths,
    bundle_dir: Path,
    cfgs: dict[str, dict],
    keys: dict[str, str],
    *,
    features_dir: Path,
    rules_dir: Path,
    graph_dir: Path,
    threads: int | None = None,
    features_verify: dict[str, Any] | None = None,
) -> dict:
    """Build the bundle in `bundle_dir`, verify it (restore the snapshot, replay the slice: model
    inputs, severities, rule flags and seed-0 scores bit-equal to the reference), then write
    metadata.json. Any failure raises and leaves no metadata.json.

    `features_verify`: the passed DuckDB-oracle summary of the feature table (the Modal job
    requires it), recorded in metadata.json. metadata.json also records the platform the
    references were computed on (`platform`, with a log1p fingerprint for M5's parity check).
    """
    t0 = time.perf_counter()
    bundle_dir, features_dir = Path(bundle_dir), Path(features_dir)
    rules_dir, graph_dir = Path(rules_dir), Path(graph_dir)
    _check_bundle_dir(
        bundle_dir, [paths.root / "parquet", paths.labels, features_dir, rules_dir, graph_dir]
    )
    versions = {d.name: _data_version(d) for d in (features_dir, rules_dir, graph_dir)}
    stamped = {v for found, v in versions.values() if found}
    if len(stamped) > 1:
        raise ValueError(f"stage outputs are built from different prepared data: {versions}")
    data_version = next(iter(stamped)) if stamped else None

    # Inputs: spec, champion booster, snapshot header, rules, scores.
    spec = load_spec(features_dir)
    build_summary = features_dir / SUMMARY_FILE
    features_digest = (
        read_json(build_summary).get(FEATURES_DIGEST) if build_summary.exists() else None
    )
    graph_summary = graph_dir / "summary.json"
    _check_same_table(
        spec.spec_hash(), {"lgbm_graph": graph_summary, "rules_engine": rules_dir / "summary.json"}
    )
    booster_src = graph_dir / f"booster_s{SEED}.txt"
    booster = lgb.Booster(model_file=str(booster_src))
    names = list(booster.feature_name())
    spec.assert_model_inputs(names)
    fn_path = graph_dir / "feature_names.json"
    if fn_path.exists():
        listed = read_json(fn_path)
        if isinstance(listed, list) and list(listed) != names:
            raise ValueError("feature_names.json differs from the seed-0 booster's features")
    snap_src = features_dir / SNAPSHOTS_DIR / snapshot_file(SNAPSHOT_SPLIT)
    header = read_snapshot_header(snap_src)
    if header.get("next_offset") != 0:
        raise ValueError(f"{snap_src.name}: next_offset {header.get('next_offset')!r} != 0")
    rules_thr = read_json(rules_dir / "thresholds.json")
    tags, head = _rate_tags(rules_thr)
    flags = pl.read_parquet(rules_dir / "flags.parquet")
    scores = pl.read_parquet(graph_dir / "scores.parquet", columns=["row_id", "split", SCORE])
    gate_path = graph_dir / "gate.json"
    gated = read_json(gate_path).get("kept") if gate_path.exists() else None
    champion = _champion(read_json(graph_summary)) if graph_summary.exists() else None

    # The slice: the first max_events test rows in rank order, from the snapshot's next rank.
    max_events = int(cfgs["serving"]["replay"]["max_events"])
    if max_events < 1:
        raise ValueError(f"serving.replay.max_events must be >= 1, got {max_events}")
    slice_df = (
        pl.scan_parquet(paths.transactions)
        .filter(pl.col("split") == SNAPSHOT_SPLIT)
        .select(*INPUT_COLUMNS, *DRIVER_COLUMNS)
        .sort("rank")
        .head(max_events)
        .collect()
    )
    n = slice_df.height
    r0 = int(header["next_rank"])
    expected = np.arange(r0, r0 + n, dtype=np.int64)
    if n == 0 or not np.array_equal(slice_df["rank"].to_numpy(), expected):
        raise ValueError(f"the {SNAPSHOT_SPLIT} slice does not start at next_rank {r0}")

    thresholds, calibration = _model_thresholds(paths, scores, flags, tags, head)
    thresholds["rules"] = {"headline_rate_tag": head, "thresholds": rules_thr["thresholds"]}
    ref_feats, ref_scores, ref_alerts = build_references(
        features_dir, names, slice_df, scores, flags, thresholds, rules_thr["thresholds"]
    )

    # Write: old contents first removed; metadata.json only after the verification.
    if bundle_dir.exists():
        shutil.rmtree(bundle_dir)
    bundle_dir.mkdir(parents=True)
    _copy_atomic(booster_src, bundle_dir / BOOSTER)
    spec_doc = spec.to_json()
    spec_doc["model"] = {
        "seed": SEED,
        "booster": BOOSTER,
        "feature_names": names,
        "model_index": list(spec.model_index(names)),
        "categorical": [n_ for n_ in names if spec.feature(n_).categorical],
        "gated": gated,
        "champion": champion,
        "float32_cast": FLOAT32_CAST,
        "predict": PREDICT,
    }
    write_json_atomic(spec_doc, bundle_dir / FEATURE_SPEC)
    write_json_atomic(_preprocess(paths, cfgs, spec), bundle_dir / PREPROCESS)
    write_json_atomic(thresholds, bundle_dir / THRESHOLDS)
    write_json_atomic(calibration, bundle_dir / CALIBRATION)
    _copy_atomic(snap_src, bundle_dir / SNAPSHOT)
    sidecar = snap_src.with_name(snap_src.name + SIDECAR_SUFFIX)
    if not sidecar.exists():
        raise FileNotFoundError(f"snapshot sidecar {sidecar} missing")
    _copy_atomic(sidecar, bundle_dir / SNAPSHOT_SIDECAR)
    write_parquet_atomic(slice_df, bundle_dir / SLICE)
    write_parquet_atomic(ref_feats, bundle_dir / REF_FEATURES)
    write_parquet_atomic(ref_scores, bundle_dir / REF_SCORES)
    write_parquet_atomic(ref_alerts, bundle_dir / REF_ALERTS)

    verification = _verify(bundle_dir)

    files = {
        rel: {"sha256": _sha256(bundle_dir / rel), "bytes": (bundle_dir / rel).stat().st_size}
        for rel in BUNDLE_FILES
    }
    meta = {
        "bundle_format": BUNDLE_FORMAT,
        "keys": dict(keys),
        "data_version": data_version,
        "engine_version": ENGINE_VERSION,
        "spec_hash": spec.spec_hash(),
        FEATURES_DIGEST: features_digest,
        "features_verify": features_verify,
        "libraries": _library_versions(),
        "platform": platform_info(slice_df["amount_usd"].to_list()),
        "files": files,
        "rows": {
            "slice": n,
            "val_late": thresholds["val_late_rows"],
            "model_inputs": len(names),
        },
        "next_rank": r0,
        "next_offset": header.get("next_offset"),
        "snapshot_state_digest": header.get("state_digest"),
        "seed": SEED,
        "champion": champion,
        "verification": verification,
        "sources": {
            "features_dir": str(features_dir),
            "rules_dir": str(rules_dir),
            "graph_dir": str(graph_dir),
        },
    }
    write_json_atomic(meta, bundle_dir / METADATA_FILE)  # last: marks a complete bundle
    return {
        "bundle_dir": str(bundle_dir),
        "rows": n,
        "next_rank": r0,
        "model_inputs": len(names),
        "verification": verification,
        "seconds": time.perf_counter() - t0,
    }


def _preprocess(paths: DataPaths, cfgs: dict[str, dict], spec: EngineSpec) -> dict[str, Any]:
    inputs = spec.to_json()
    return {
        "normalisation_stats": None,
        "note": "Tree models need no normalisation statistics, so none are fitted or applied. "
        "The train-fitted preprocessing the engine uses is listed here.",
        "fit_split": "train",
        "fx": read_json(paths.fx_rates),
        "vocab": {k: list(v) for k, v in spec.vocab.items()},
        "hub_degree_quantile": cfgs["rules"]["hub_degree_quantile"],
        "hub_cap": spec.hub_cap,
        "hubs": list(spec.hubs),
        "windows": inputs["inputs"]["windows"],
        "rule_windows": inputs["derived"]["rule_windows"],
    }


# --- verification -------------------------------------------------------------------------------


def _bits_equal(a: np.ndarray, b: np.ndarray, bits: type) -> np.ndarray:
    """Row-wise: every value of the row has the same bit pattern."""
    a = np.ascontiguousarray(a)
    b = np.ascontiguousarray(b)
    if a.shape != b.shape:
        raise BundleVerificationError(f"shape {a.shape} != reference {b.shape}")
    eq = a.view(bits) == b.view(bits)
    return eq if eq.ndim == 1 else eq.all(axis=1)


def _verify(bundle_dir: Path) -> dict[str, Any]:
    """Replay the slice from the bundle's snapshot and compare with the reference outputs."""
    t0 = time.perf_counter()
    spec_doc = read_json(bundle_dir / FEATURE_SPEC)
    spec = EngineSpec.from_json(spec_doc)
    names = list(spec_doc["model"]["feature_names"])
    idx = list(spec.model_index(names))
    booster = lgb.Booster(model_file=str(bundle_dir / BOOSTER))
    if list(booster.feature_name()) != names:
        raise BundleVerificationError("booster features differ from feature_spec.json")
    thresholds = read_json(bundle_dir / THRESHOLDS)
    tags, head = thresholds["rate_tags"], thresholds["headline_rate_tag"]

    slice_df = pl.read_parquet(bundle_dir / SLICE)
    n = slice_df.height
    eng, header = Engine.restore(bundle_dir / SNAPSHOT, spec)
    r0 = int(slice_df["rank"][0])
    if header.get("next_rank") != r0 or header.get("next_offset") != 0:
        raise BundleVerificationError(
            f"snapshot next_rank/next_offset {header.get('next_rank')}/"
            f"{header.get('next_offset')} != {r0}/0"
        )
    buf = array("d")
    extend, process, prepare = buf.extend, eng.process, eng.prepare
    for f in zip(*(slice_df[c].to_list() for c in INPUT_COLUMNS), strict=True):
        extend(process(prepare(*f)))
    flush = eng.advance(int(slice_df["minute"][-1]) + 1)
    mat = np.frombuffer(buf, dtype=np.float64).reshape(n, spec.row_len)

    ref = pl.read_parquet(bundle_dir / REF_FEATURES)
    ref_scores = pl.read_parquet(bundle_dir / REF_SCORES)
    ref_alerts = pl.read_parquet(bundle_dir / REF_ALERTS)
    same = [r["row_id"].equals(slice_df["row_id"]) for r in (ref, ref_scores, ref_alerts)]
    if not all(same):
        raise BundleVerificationError("reference rows differ from the slice rows")

    x = np.asarray(mat[:, idx], np.float64).astype(np.float32)
    bad: dict[str, int] = {}
    ok = _bits_equal(x, ref.select(names).to_numpy().astype(np.float32, copy=False), np.uint32)
    bad["features"] = int((~ok).sum())
    sev = mat[:, spec.i_sev : spec.i_sev + len(SEVERITY_COLUMNS)]
    ok = _bits_equal(sev, ref.select(SEVERITY_COLUMNS).to_numpy().astype(np.float64), np.uint64)
    bad["severities"] = int((~ok).sum())
    trunc = mat[:, [spec.i_rule_trunc, spec.i_cyc_trunc, spec.i_sg_trunc]]
    bad["trunc"] = int((trunc != ref.select(TRUNC_COLUMNS).to_numpy()).any(axis=1).sum())

    sev_df = pl.DataFrame(
        {"row_id": slice_df["row_id"], **{s: sev[:, j] for j, s in enumerate(SEVERITY_COLUMNS)}}
    )
    rules = rule_flags(sev_df, thresholds["rules"]["thresholds"], tags, head)
    bad["rule_flags"] = _rows_differ(rules, ref_alerts)
    p = np.asarray(booster.predict(x, num_threads=1), dtype=np.float64)
    ref_p = ref_scores[SCORE].to_numpy().astype(np.float64)
    bad["scores"] = int((~_bits_equal(p, ref_p, np.uint64)).sum())
    alerts = model_alerts(slice_df["row_id"], p, thresholds["model"], tags)
    bad["alerts"] = _rows_differ(alerts, ref_alerts)
    k = min(ROW_BY_ROW_ROWS, n)
    single = np.array([booster.predict(x[i : i + 1], num_threads=1)[0] for i in range(k)])
    bad["row_by_row"] = int((~_bits_equal(single.astype(np.float64), p[:k], np.uint64)).sum())
    failed = {kk: v for kk, v in bad.items() if v}
    if failed:
        raise BundleVerificationError(f"bundle verification failed (rows per check): {failed}")
    return {
        "rows": n,
        "row_by_row_rows": k,
        "checks": sorted(bad),
        "mismatches": 0,
        "final_flush": {"n_applied": flush.n_applied, "n_expired": flush.n_expired},
        "final_state_digest": eng.state_digest(),
        "seconds": time.perf_counter() - t0,
    }


def verify_bundle(bundle_dir: Path, *, threads: int | None = None) -> dict:
    """Re-run the bundle verification on an existing bundle directory (no writes).

    Also checks every file's sha256 and size against metadata.json when it exists. Predictions
    always use one thread (`threads` is accepted for the common signature)."""
    bundle_dir = Path(bundle_dir)
    out = _verify(bundle_dir)
    meta_path = bundle_dir / METADATA_FILE
    out["metadata"] = meta_path.exists()
    if meta_path.exists():
        files = read_json(meta_path)["files"]
        wrong = [
            rel
            for rel, info in files.items()
            if not (bundle_dir / rel).exists()
            or (bundle_dir / rel).stat().st_size != info["bytes"]
            or _sha256(bundle_dir / rel) != info["sha256"]
        ]
        if wrong:
            raise BundleVerificationError(f"files differ from metadata.json: {wrong}")
        out["files_checked"] = len(files)
    return out
