"""aml.features.gate: PSI, the NaN / zero / quantile / per-value bins, out-of-range support, the
learnability floor and the drop-share guard (M2 spec §8.3 step 2)."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable

import numpy as np
import polars as pl
import pytest

from aml.features.gate import check_gate, check_gate_cfg, psi, run_gate
from aml.features.spec import EngineSpec, FeatureDef, GateStopError, scan_feature_table
from aml.io import read_json
from tests.fixtures.feature_table import build_feature_table

CFG = {
    "psi_max": 0.25,
    "psi_bins": 10,
    "psi_eps": 1.0e-4,
    "psi_train_days": [4, 6],
    "warmup_days": [1, 3],
    "out_of_range_max": 0.01,
    "min_train_nonzero": 100,
    "max_drop_share": 0.25,
}
Gen = Callable[[np.random.Generator, int, int], np.ndarray]


def fdef(name: str, group: str = "VEL", domain: str = "real", categorical=False) -> FeatureDef:
    return FeatureDef(name, group, "u", None, "exact", domain=domain, categorical=categorical)


def make_table(cols: dict[str, Gen], n_day: int = 2000, n_val: int = 2000, seed: int = 0):
    """Train days 1-6 (n_day rows each), val_early day 7, val_late day 8; columns from `cols`."""
    rng = np.random.default_rng(seed)
    blocks = [(d, "train", n_day) for d in range(1, 7)]
    blocks += [(7, "val_early", n_val), (8, "val_late", 300)]
    parts = []
    for day, split, n in blocks:
        data = {"day": pl.Series(np.full(n, day), dtype=pl.Int16), "split": [split] * n}
        for name, gen in cols.items():
            data[name] = pl.Series(np.asarray(gen(rng, day, n), dtype=np.float32))
        parts.append(pl.DataFrame(data))
    return pl.concat(parts)


def normal(rng, day, n):
    return rng.normal(5.0, 2.0, n)


def counts(rng, day, n):
    return rng.poisson(1.5, n)


def flags(rng, day, n):
    return rng.random(n) < 0.3


# --------------------------------------------------------------------------- psi


def test_psi_known_values():
    # Unsmoothed PSI([.5, .5] -> [.9, .1]) = 0.4 ln(1.8) + 0.4 ln(5) = 0.4 ln 9.
    assert psi([50, 50], [90, 10], 1e-12) == pytest.approx(0.4 * math.log(9), rel=1e-9)
    # Same proportions (any sample sizes) -> 0; PSI is symmetric.
    assert psi([10, 20, 30], [20, 40, 60], 1e-4) == pytest.approx(0.0, abs=1e-15)
    p, q = np.array([5, 40, 30, 25]), np.array([30, 30, 20, 20])
    assert psi(p, q, 1e-4) == pytest.approx(psi(q, p, 1e-4), rel=1e-12)
    # The smoothing keeps empty bins finite: (count + eps N) / (N (1 + k eps)).
    got = psi([100, 0], [0, 100], 1e-4)
    a, b = (100 + 0.01) / (100 * 1.0002), 0.01 / (100 * 1.0002)
    assert got == pytest.approx(2 * (a - b) * math.log(a / b), rel=1e-12)
    assert math.isfinite(got) and got > 10


@pytest.mark.parametrize(
    "p, q, eps",
    [([1, 2], [1, 2, 3], 1e-4), ([], [], 1e-4), ([1, 2], [1, 2], 0.0), ([0, 0], [1, 1], 1e-4)],
)
def test_psi_rejects_bad_input(p, q, eps):
    with pytest.raises(ValueError):
        psi(p, q, eps)
    with pytest.raises(ValueError):
        psi([1, -1], [1, 1], 1e-4)


# --------------------------------------------------------------------------- binning, drop rules


def test_stationary_features_are_all_kept():
    cols = {"real": normal, "count": counts, "flag": flags}
    feats = [fdef("real"), fdef("count", domain="int"), fdef("flag", domain="flag")]
    doc = run_gate(make_table(cols), feats, CFG)
    assert doc["kept"] == doc["order"] == ["real", "count", "flag"] and doc["dropped"] == []
    assert doc["stop"] is False and doc["drop_share"] == 0.0
    assert doc["rows"] == {"warm_train": 6000, "val_early": 2000, "train": 12000, "warmup": 6000}
    for r in doc["features"].values():
        assert r["psi"] < 0.02 and r["psi_warmup"] < 0.02 and r["reasons"] == []
    # NaN bin + zero bin + 10 quantile bins for a continuous feature; per value for a flag.
    assert doc["features"]["real"]["kind"] == "quantile" and doc["features"]["real"]["n_bins"] == 12
    assert doc["features"]["flag"]["kind"] == "value" and doc["features"]["flag"]["n_bins"] == 3
    json.dumps(doc, allow_nan=False)


def test_quantile_bins_keep_distinct_edges_only():
    doc = run_gate(make_table({"c": lambda r, d, n: r.integers(0, 3, n)}), [fdef("c")], CFG)
    r = doc["features"]["c"]
    assert r["edges"] == [1.0, 2.0]  # data values, ties collapsed
    assert r["n_bins"] == 2 + len(r["edges"]) + 1


def test_nan_bin_detects_a_missingness_shift():
    def nan_shift(rng, day, n):
        x = rng.normal(5, 2, n)
        x[rng.random(n) < (0.6 if day == 7 else 0.1)] = np.nan
        return x

    def nan_same(rng, day, n):
        x = rng.normal(5, 2, n)
        x[rng.random(n) < 0.3] = np.nan
        return x

    feats = [fdef("shift", domain="nullable"), fdef("same", domain="nullable")]
    doc = run_gate(make_table({"shift": nan_shift, "same": nan_same}), feats, CFG)
    assert doc["features"]["shift"]["reasons"] == ["psi"]
    assert doc["features"]["shift"]["nan_share_val"] == pytest.approx(0.6, abs=0.03)
    assert not doc["features"]["same"]["drop"]


def test_zero_bin_detects_a_sparsity_shift():
    def zero_shift(rng, day, n):
        x = rng.normal(5, 2, n)
        x[rng.random(n) < (0.9 if day == 7 else 0.5)] = 0.0
        return x

    doc = run_gate(make_table({"z": zero_shift}), [fdef("z")], CFG)
    assert "psi" in doc["features"]["z"]["reasons"]


def _unit(rng, day, n):
    x = rng.random(n)
    if day != 7:
        x[:2] = 0.0, 1.0  # the warm-train range is exactly [0, 1]
    return x


def test_out_of_range_share():
    def oor(rng, day, n):
        x = _unit(rng, day, n)
        if day == 7:
            x[:30] = 5.0  # above the warm-train max
            x[30:40] = -1.0  # below its min
            x[40:60] = np.nan  # not counted at all
        return x

    def inside(rng, day, n):
        x = _unit(rng, day, n)
        if day == 7:
            x[:15] = 5.0
        return x

    doc = run_gate(make_table({"oor": oor, "inside": inside}), [fdef("oor"), fdef("inside")], CFG)
    assert doc["features"]["oor"]["oor"] == pytest.approx(40 / 1980)
    assert doc["features"]["oor"]["reasons"] == ["oor"]
    assert doc["features"]["inside"]["oor"] == pytest.approx(15 / 2000)  # under 1%: kept
    assert not doc["features"]["inside"]["drop"]


def _sparse(n_nonzero_per_day: dict[int, int]) -> Gen:
    def gen(rng, day, n):
        x = np.zeros(n)
        k = n_nonzero_per_day.get(day, 0)
        x[rng.choice(n, size=k, replace=False)] = rng.integers(1, 4, k)
        return x

    return gen


def test_count_floor_is_a_row_count_not_a_share():
    # 99 non-zero train rows -> dropped; 100 -> kept (the val_early values stay in range).
    per_day_99 = {1: 16, 2: 16, 3: 16, 4: 17, 5: 17, 6: 17, 7: 6}
    per_day_100 = {**per_day_99, 6: 18}
    cols = {"r99": _sparse(per_day_99), "r100": _sparse(per_day_100)}
    doc = run_gate(make_table(cols), [fdef("r99", "CYC", "int"), fdef("r100", "CYC", "int")], CFG)
    assert doc["features"]["r99"]["nonzero"] == 99
    assert doc["features"]["r99"]["reasons"] == ["nonzero"]
    assert doc["features"]["r100"]["nonzero"] == 100 and not doc["features"]["r100"]["drop"]


def test_count_floor_keeps_a_0p04_percent_nonzero_feature():
    """Round trip is non-zero on 0.039% of the real rows: a share floor would drop it."""
    n_day, k = 70_000, 28  # 28 / 70,000 = 0.04% per day
    cols = {"rt": _sparse(dict.fromkeys(range(1, 8), k))}
    doc = run_gate(make_table(cols, n_day=n_day, n_val=n_day), [fdef("rt", "CYC", "int")], CFG)
    r = doc["features"]["rt"]
    assert r["nonzero"] == 6 * k and r["nonzero"] / (6 * n_day) == pytest.approx(0.0004)
    assert not r["drop"], r


def test_categorical_and_flag_features_bin_per_value():
    def codes(rng, day, n):
        x = rng.integers(0, 3, n).astype(float)
        if day == 7:
            x[rng.random(n) < 0.3] = -1  # unseen category in val_early
        return x

    feats = [
        fdef("code", "TX", "code", categorical=True),
        fdef("flag", "RULE", "flag"),
    ]
    doc = run_gate(make_table({"code": codes, "flag": flags}), feats, CFG)
    code = doc["features"]["code"]
    assert code["kind"] == "value" and code["n_bins"] == 1 + 4  # NaN + {-1, 0, 1, 2}
    assert set(code["reasons"]) == {"psi", "oor"}
    assert doc["dropped_tx"] == ["code"]
    assert doc["features"]["flag"]["kind"] == "value" and not doc["features"]["flag"]["drop"]


def test_warmup_psi_is_reported_only():
    def warmup_shift(rng, day, n):
        return rng.normal(20 if day <= 3 else 5, 2, n)

    doc = run_gate(make_table({"w": warmup_shift}), [fdef("w")], CFG)
    r = doc["features"]["w"]
    assert r["psi_warmup"] > CFG["psi_max"] and r["psi"] < 0.05 and not r["drop"]


# --------------------------------------------------------------------------- the guard


def _many(n_engine: int, n_bad: int, tx_bad: bool = False):
    def bad(rng, day, n):
        return rng.normal(50 if day == 7 else 5, 1, n)

    cols, feats = {}, []
    for i in range(n_engine):
        cols[f"e{i}"] = bad if i < n_bad else normal
        feats.append(fdef(f"e{i}"))
    cols["t"] = bad if tx_bad else normal
    feats.append(fdef("t", "TX"))
    return make_table(cols, n_day=400, n_val=400), feats


def test_guard_trips_above_the_drop_share():
    doc = run_gate(*_many(10, 3), CFG)
    assert doc["dropped_engine"] == ["e0", "e1", "e2"] and doc["drop_share"] == 0.3
    assert doc["stop"] is True
    with pytest.raises(GateStopError, match="3 of 10 engine features"):
        check_gate(doc)


def test_guard_is_strict_and_ignores_tx_drops():
    doc = run_gate(*_many(8, 2, tx_bad=True), CFG)
    assert doc["drop_share"] == 0.25 and doc["stop"] is False  # exactly the share: no stop
    assert doc["dropped_tx"] == ["t"] and doc["n_engine"] == 8
    check_gate(doc)  # does not raise


def test_gate_config_is_validated():
    for change in (
        {"psi_eps": 0},
        {"psi_train_days": [6, 4]},
        {"psi_bins": 0},
        {"max_drop_share": -0.1},
    ):
        with pytest.raises(ValueError):
            check_gate_cfg({**CFG, **change})
    cfg = dict(CFG)
    del cfg["psi_max"]
    with pytest.raises(ValueError, match="psi_max"):
        check_gate_cfg(cfg)
    table = make_table({"x": normal})
    with pytest.raises(ValueError, match="lacks columns"):
        run_gate(table, [fdef("missing")], CFG)
    with pytest.raises(ValueError, match="warm-train rows"):
        run_gate(table.filter(pl.col("split") != "val_early"), [fdef("x")], CFG)


# --------------------------------------------------------------------------- the feature table


def test_gate_on_the_fixture_feature_table(prepared, tmp_path):
    from aml.data.schemas import validate_features
    from aml.features.spec import part_paths

    spec = build_feature_table(prepared, tmp_path)
    assert spec == EngineSpec.from_json(read_json(tmp_path / "feature_spec.json"))
    for p in part_paths(tmp_path):  # the fixture is a schema-valid table (§5.10)
        validate_features(pl.read_parquet(p), spec)
    table = scan_feature_table(tmp_path, ["day", "split", *spec.feature_names]).collect()
    doc = run_gate(table, spec.features, copy.deepcopy(CFG))
    json.dumps(doc, allow_nan=False)
    assert doc["order"] == list(spec.feature_names)
    assert doc["kept"] == [n for n in spec.feature_names if n not in set(doc["dropped"])]
    assert doc["n_engine"] == len(spec.feature_names) - len(spec.group_names("TX"))
    for name, r in doc["features"].items():
        want = (
            (r["psi"] > CFG["psi_max"])
            or (r["oor"] > CFG["out_of_range_max"])
            or (r["nonzero"] < CFG["min_train_nonzero"])
        )
        assert r["drop"] == want, name
    # cyc4 is seeded at ~0.3% non-zero on ~4,300 train rows: under the 100-row floor.
    assert "nonzero" in doc["features"]["cyc4_2d"]["reasons"]
