"""Determinism (M2 spec §4.9, §9.2): the feature table and the snapshots are a pure function of
the inputs. Two fresh interpreters with PYTHONHASHSEED 0 and 1 build the fixture's feature table;
parts, snapshots and digests must be identical. In-process, two engine runs give the same rows,
digest and snapshot bytes.
"""

from __future__ import annotations

import hashlib
import json

import numpy as np
import polars as pl
import pytest

from aml.features.spec import INPUT_COLUMNS
from tests.fixtures.engine_frames import (
    dense_engine_frame,
    features_cfg,
    fixture_build_cfgs,
    fixture_builds,
    make_spec,
    require_build,
    require_engine,
    row_bits,
)
from tests.fixtures.rules_frames import small_cfg


def sha256(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frame_bits(df: pl.DataFrame) -> list:
    """Every column as raw values (floats by their bit patterns)."""
    out = []
    for c in df.columns:
        x = df[c].to_numpy()
        out.append((c, str(df.schema[c]), x.view(f"u{x.itemsize}") if x.dtype.kind == "f" else x))
    return out


def same_frame(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    fa, fb = frame_bits(a), frame_bits(b)
    return len(fa) == len(fb) and all(
        ca == cb and ta == tb and np.array_equal(xa, xb)
        for (ca, ta, xa), (cb, tb, xb) in zip(fa, fb, strict=True)
    )


@pytest.fixture(scope="module")
def two_builds(prepared, data_cfg, rules_cfg):
    """The fixture built in two fresh interpreters with PYTHONHASHSEED 0 and 1."""
    require_build()
    builds = fixture_builds(prepared, fixture_build_cfgs(data_cfg, rules_cfg))
    return builds["seed0"], builds["seed1"]


def test_parts_identical_across_hash_seeds(two_builds):
    a, b = two_builds
    pa = sorted(p.name for p in (a / "parts").glob("*.parquet"))
    pb = sorted(p.name for p in (b / "parts").glob("*.parquet"))
    assert pa == pb and len(pa) == 18
    for name in pa:
        fa, fb = pl.read_parquet(a / "parts" / name), pl.read_parquet(b / "parts" / name)
        assert same_frame(fa, fb), name
        assert sha256(a / "parts" / name) == sha256(b / "parts" / name), name


def test_snapshots_identical_across_hash_seeds(two_builds):
    a, b = two_builds
    names = sorted(p.name for p in (a / "snapshots").glob("*.snap"))
    assert names == sorted(p.name for p in (b / "snapshots").glob("*.snap")) and len(names) == 2
    for name in names:
        ha = json.loads((a / "snapshots" / f"{name}.json").read_text(encoding="utf-8"))
        hb = json.loads((b / "snapshots" / f"{name}.json").read_text(encoding="utf-8"))
        assert ha["state_digest"] == hb["state_digest"], name
        assert ha["payload_sha256"] == hb["payload_sha256"], name
        assert sha256(a / "snapshots" / name) == sha256(b / "snapshots" / name), name


def test_spec_and_vocab_identical_across_hash_seeds(two_builds):
    a, b = two_builds
    for name in ("feature_spec.json", "vocab.json"):
        assert (a / name).read_bytes() == (b / name).read_bytes(), name


def test_two_engine_runs_are_identical(rules_cfg):
    """Same process, two engines: rows, final digest and snapshot bytes (both compressions)."""
    require_engine()
    from aml.features.engine import Engine

    frame = dense_engine_frame(seed=5, n=400)
    spec = make_spec(frame, small_cfg(rules_cfg, 10, 5), features_cfg(short=5, long=12, sg=5))
    out = []
    for _ in range(2):
        eng = Engine.create(spec)
        cols = [frame[c].to_list() for c in INPUT_COLUMNS]
        rows = [row_bits(eng.process(eng.prepare(*f))) for f in zip(*cols, strict=True)]
        eng.advance(int(frame["minute"].max()) + 1)
        out.append((rows, eng.state_digest(), eng.snapshot(), eng.snapshot(compress=False)))
    assert out[0] == out[1]
