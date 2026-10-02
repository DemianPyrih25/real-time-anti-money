"""Feature parity (M2 spec §9.2-§9.3): the DuckDB oracle, the brute-force reference and the engine.

The oracle (`aml.features.oracle`, set-based SQL) and the reference (`tests.fixtures.engine_ref`,
literal per-event filters and path enumeration) are two independent implementations of §4-§6; they
are checked against each other and against hand-computed values first, then the engine is checked
against both: every covered column under its tolerance class (float64 side), and all 79 features,
the severities and inflow_c against the reference.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from aml.features.oracle import (
    M2_PREFIX,
    SQL_COLUMNS,
    compare_with_engine,
    covered_features,
    not_covered,
    oracle_features,
    run_oracle,
)
from aml.features.spec import FEATURE_SPEC_FILE, INPUT_COLUMNS, PARTS_DIR, part_name, window_tag
from aml.io import read_json, write_json_atomic
from tests.fixtures.engine_frames import (
    dense_engine_frame,
    engine_tx,
    features_cfg,
    fixture_spec,
    make_spec,
    require_engine,
    rows_frame,
    rules_windows,
    run_engine,
)
from tests.fixtures.engine_ref import (
    Reference,
    compare_rows,
    reference_frame,
    reference_frame_cached,
)
from tests.fixtures.rules_frames import small_cfg

FIXTURE_HUB_CAP = 30  # 9 hubs on the fixture (its fitted cap, 43, gives none)
DENSE_FEATURES = {"short": 5, "long": 12, "sg": 5}


def dense_hub_cap(frame: pl.DataFrame) -> int:
    """A cap that makes the busier half of a dense frame's train accounts hubs."""
    train = frame.filter(pl.col("split") == "train")
    deg = sorted(pl.concat([train["src"], train["dst"]]).value_counts()["count"].to_list())
    return int(deg[len(deg) // 2]) if deg else 0


def unlike_windows(rules_cfg: dict) -> dict:
    """Rule windows that differ from every feature window (and from each other)."""
    return rules_windows(
        rules_cfg,
        fan_in=600,
        fan_out=900,
        pass_through=300,
        round_trip=2000,
        hop=700,
        structuring=500,
        round_burst=800,
        high_risk=1000,
    )


def assert_parity(res: dict) -> None:
    bad = {c: v for c, v in res["columns"].items() if v["mismatches"]}
    assert res["ok"] and not bad, bad


# ---------------------------------------------------------------------------------------------
# Module data


@pytest.fixture(scope="module")
def fx(prepared, rules_cfg):
    spec, tx = fixture_spec(prepared, rules_cfg, hub_cap=FIXTURE_HUB_CAP)
    assert 0 < len(spec.hubs) < 20
    return spec, tx


@pytest.fixture(scope="module")
def fx_ref(fx) -> pl.DataFrame:
    spec, tx = fx
    return reference_frame_cached(tx, spec)


@pytest.fixture(scope="module")
def fx_oracle(fx) -> pl.DataFrame:
    spec, tx = fx
    return oracle_features(tx, spec, threads=2)


@pytest.fixture(scope="module")
def fx_engine(fx) -> pl.DataFrame:
    require_engine()
    spec, tx = fx
    return rows_frame(tx, run_engine(tx, spec), spec)


def dense_case(seed: int, rules_cfg: dict):
    frame = dense_engine_frame(seed=seed)
    spec = make_spec(
        frame,
        small_cfg(rules_cfg, 10, 5),
        features_cfg(**DENSE_FEATURES),
        hub_cap=dense_hub_cap(frame),
    )
    return frame, spec


# ---------------------------------------------------------------------------------------------
# Oracle == reference (no engine needed)


def test_oracle_covers_all_but_longer_paths_and_scatter_gather(fx):
    spec, _ = fx
    cov = covered_features(spec)
    assert len(cov) == 75 and len(spec.feature_names) == 79
    assert set(spec.feature_names) - set(cov) == {"cyc3_2d", "cyc4_2d", "sg_mids_1d", "sg_srcs_1d"}
    assert set(not_covered(spec)) == set(spec.feature_names) - set(cov)
    # Labels never enter: the oracle reads only engine input columns.
    assert set(SQL_COLUMNS) <= set(INPUT_COLUMNS)


def test_oracle_equals_reference_on_fixture(fx, fx_ref, fx_oracle):
    spec, tx = fx
    assert fx_oracle["row_id"].to_list() == tx["row_id"].to_list()
    assert_parity(compare_with_engine(fx_ref, fx_oracle, spec, float32=False))
    # Not vacuous: every covered windowed column is non-zero somewhere, hubs exist, the longer
    # paths and scatter-gather (reference only) fire on the fixture.
    for c in covered_features(spec):
        assert (fx_ref[c].fill_nan(0) != 0).any(), c
    for c in not_covered(spec):
        assert (fx_ref[c] > 0).any(), c


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_oracle_equals_reference_on_dense_frames(rules_cfg, seed):
    frame, spec = dense_case(seed, rules_cfg)
    assert 0 < len(spec.hubs) < 6
    res = compare_with_engine(
        reference_frame(frame, spec), oracle_features(frame, spec, threads=1), spec, float32=False
    )
    assert_parity(res)


def test_oracle_equals_reference_with_rule_windows_unlike_feature_windows(fx, rules_cfg):
    _, tx = fx
    spec = make_spec(tx, unlike_windows(rules_cfg), hub_cap=FIXTURE_HUB_CAP)
    assert {"u_inflow_5h", "pt_ratio_5h", "cyc2_2000m"} <= set(spec.feature_names)
    sub = tx.head(3000)
    res = compare_with_engine(
        reference_frame(sub, spec), oracle_features(sub, spec, threads=2), spec, float32=False
    )
    assert_parity(res)


def test_compare_rows_detects_every_kind_of_difference(fx):
    """The row comparator is not vacuous: a one-ulp change, a NaN, a wrong severity, a wrong
    inflow_c, a non-int flag and an over-count under a trunc flag are all reported."""
    spec, tx = fx
    ref = Reference.from_frame(tx.head(400), spec)
    k = next(i for i in range(len(ref)) if ref.row(i)[spec.feature_index["v_in_cnt_1d"]] > 0)
    row, m2 = ref.row_m2(k)
    assert compare_rows(row, row, m2, spec) == []

    def changed(pos, value):
        return tuple(value if j == pos else x for j, x in enumerate(row))

    fi = spec.feature_index
    up = math.nextafter(row[fi["log_amount_usd"]], math.inf)
    assert compare_rows(changed(fi["log_amount_usd"], up), row, m2, spec)  # ulp: bit-equal
    assert not compare_rows(changed(fi["log_amount_usd"], up), row, m2, spec, independent=True)
    assert compare_rows(changed(fi["v_in_cnt_1d"], math.nan), row, m2, spec)
    assert compare_rows(changed(fi["v_in_mean_1d"], row[fi["v_in_mean_1d"]] + 1e-6), row, m2, spec)
    assert compare_rows(changed(spec.i_sev, row[spec.i_sev] + 1.0), row, m2, spec)
    assert compare_rows(changed(spec.i_inflow, row[spec.i_inflow] + 1), row, m2, spec)
    assert compare_rows(changed(spec.i_sg_trunc, True), row, m2, spec)
    # Under sg_trunc a scatter-gather count may only be lower.
    trunc = changed(spec.i_sg_trunc, 1)
    sg = fi["sg_mids_1d"]
    low = tuple(-1.0 if j == sg else x for j, x in enumerate(trunc))
    high = tuple(row[sg] + 1.0 if j == sg else x for j, x in enumerate(trunc))
    assert not compare_rows(low, row, m2, spec)
    assert compare_rows(high, row, m2, spec)


# ---------------------------------------------------------------------------------------------
# Hand-computed cases (literal expected values)

WINDOW_CASE = [
    # target T = row 0: u = 0 -> v = 1 at m = 100; S = 10 -> [90, 99], L = 30 -> [70, 99]
    {"minute": 100, "src": 0, "dst": 1, "amount_usd": 100.0},  # 0: T
    {"minute": 90, "src": 0, "dst": 2, "amount_usd": 50.0},  # 1: at m - S: in S
    {"minute": 89, "src": 0, "dst": 3, "amount_usd": 7.0},  # 2: m - S - 1: L only
    {"minute": 99, "src": 0, "dst": 2, "amount_usd": 25.0},  # 3: repeat receiver
    {"minute": 95, "src": 0, "dst": 0, "amount_usd": 10.0},  # 4: self-loop, own counterparty
    {"minute": 100, "src": 0, "dst": 1, "amount_usd": 1.0},  # 5: same minute, same pair
    {"minute": 100, "src": 4, "dst": 1, "amount_usd": 1.0},  # 6: same minute into v
    {"minute": 96, "src": 5, "dst": 0, "amount_usd": 200.0},  # 7: inflow into u
    {"minute": 97, "src": 1, "dst": 0, "amount_usd": 30.0},  # 8: v -> u (reverse pair)
    {"minute": 98, "src": 1, "dst": 6, "amount_usd": 5.0},  # 9: v pays on
    {"minute": 105, "src": 0, "dst": 1, "amount_usd": 3.0},  # 10: the pair again
    {"minute": 105, "src": 0, "dst": 7, "amount_usd": 3.0},  # 11: a new receiver later
    {"minute": 60, "src": 0, "dst": 3, "amount_usd": 1.0},  # 12: outside L
]


def window_case(rules_cfg):
    frame = engine_tx(WINDOW_CASE)
    cfg = small_cfg(rules_cfg, 10, 5)
    cfg["scenarios"]["rapid_pass_through"]["window_minutes"] = 5  # W_pt = 5 -> [95, 99]
    spec = make_spec(frame, cfg, features_cfg(short=10, long=30, sg=10, gap=25))
    return frame, spec


def _values(frame, spec, i_row):
    """The row of row_id i_row from the reference and from the oracle (covered columns)."""
    ref = reference_frame(frame, spec).filter(pl.col("row_id") == i_row).row(0, named=True)
    ora = oracle_features(frame, spec, threads=1).filter(pl.col("row_id") == i_row)
    return ref, ora.row(0, named=True)


def test_hand_case_windows_ports_gaps_flow(rules_cfg):
    frame, spec = window_case(rules_cfg)
    ref, ora = _values(frame, spec, 0)
    l = math.log1p  # noqa: E741
    want = {
        "u_out_cnt_10m": 3.0,  # rows 1, 3, 4 (row 2 at m - S - 1 is out, row 5 same minute)
        "u_out_uniq_10m": 2.0,  # receivers 2 and 0 (the self-loop counts u itself)
        "u_out_cnt_30m": 4.0,
        "u_out_uniq_30m": 3.0,
        "u_out_sum_10m": l(85.0),
        "u_out_sum_30m": l(92.0),
        "u_out_max_10m": l(50.0),
        "u_in_cnt_10m": 3.0,  # rows 4 (self-loop), 7, 8
        "u_in_uniq_10m": 3.0,
        "v_in_cnt_10m": 0.0,  # row 6 shares the minute: invisible
        "v_in_sum_10m": 0.0,
        "v_out_cnt_10m": 2.0,  # rows 8, 9
        "v_out_uniq_10m": 2.0,
        "pair_cnt_10m": 0.0,
        "pair_is_new": 1.0,
        "out_port": l(3.0),  # receivers before minute 100: 3, 2, 0
        "in_port": 0.0,
        "u_out_gap": 1.0,
        "u_in_gap": 3.0,
        "v_in_gap": 0.0,  # no history
        "v_out_gap": 2.0,
        "pair_gap": 0.0,
        "rev_pair_gap": 3.0,
        "cyc2_10m": 1.0,  # 1 -> 0 at 97
        "gs_u_10m": 2.0,
        "gs_v_10m": 0.0,
        "u_inflow_5m": l(230.0),  # rows 7 and 8 (W_pt = 5); the self-loop row 4 is not inflow
        "pt_ratio_5m": 10000 / 23000,
        "u_bal_30m": (9200 - 24000) / (9200 + 24000),  # out 92.00; in 10 + 200 + 30
    }
    for name, value in want.items():
        assert ref[name] == value, (name, ref[name], value)
        assert ora[name] == pytest.approx(value, rel=1e-15, abs=0), name
    for name in ("v_in_mean_10m", "v_in_std_10m", "v_in_max_10m", "v_amt_dev_10m"):
        assert math.isnan(ref[name]) and math.isnan(ora[name]), name
    ls = [l(50.0), l(25.0), l(10.0)]
    mean = math.fsum(ls) / 3
    std = math.sqrt(math.fsum((x - mean) ** 2 for x in ls) / 3)
    assert ref["u_out_mean_10m"] == pytest.approx(mean, rel=1e-15)
    assert ref["u_out_std_10m"] == pytest.approx(std, rel=1e-12)
    assert ora["u_out_std_10m"] == pytest.approx(std, rel=1e-12)
    assert ref["u_amt_dev_10m"] == pytest.approx(l(100.0) - mean, rel=1e-14)
    # Row 5 (same minute, same new pair) gets the same pair features and ports as T.
    r5, o5 = _values(frame, spec, 5)
    for name in ("pair_is_new", "out_port", "in_port", "pair_cnt_10m", "pair_gap"):
        assert r5[name] == ref[name] and o5[name] == ora[name], name
    # Row 10: the pair at 105 sees both minute-100 events; the port stays the pair's own.
    r10, o10 = _values(frame, spec, 10)
    for name, value in {"pair_cnt_10m": 2.0, "pair_is_new": 0.0, "pair_gap": 5.0}.items():
        assert r10[name] == value and o10[name] == value, name
    assert r10["out_port"] == o10["out_port"] == l(3.0)
    # Row 11: a new receiver at 105: receivers before 105 are 3, 2, 0, 1 -> port 4.
    r11, o11 = _values(frame, spec, 11)
    assert r11["out_port"] == o11["out_port"] == l(4.0)
    # [95, 104]: rows 4 (self-loop pair new at 95), 0 and 5 (new at 100); row 3 repeats a pair
    assert r11["u_out_newcp_10m"] == o11["u_out_newcp_10m"] == 3.0


CYCLE_CASE = [
    # target u = 0 -> v = 1 at m = 100; W_rt = 10 -> [90, 99], H = 3; account 9 is a hub
    {"minute": 100, "src": 0, "dst": 1},  # 0: target
    {"minute": 95, "src": 1, "dst": 0},  # c2
    {"minute": 92, "src": 1, "dst": 2},  # c3 via 2: 92 <= 94 <= 95
    {"minute": 94, "src": 2, "dst": 0},
    {"minute": 91, "src": 1, "dst": 3},  # via 3: gap 4 > H, no
    {"minute": 95, "src": 3, "dst": 0},
    {"minute": 93, "src": 1, "dst": 4},  # via 4: equal minutes count
    {"minute": 93, "src": 4, "dst": 0},
    {"minute": 93, "src": 1, "dst": 9},  # via the hub 9: excluded
    {"minute": 94, "src": 9, "dst": 0},
    {"minute": 91, "src": 1, "dst": 5},  # c4: 1 -> 5 -> 6 -> 0 at 91, 93, 95
    {"minute": 93, "src": 5, "dst": 6},
    {"minute": 95, "src": 6, "dst": 0},
    {"minute": 97, "src": 6, "dst": 0},  # last hop gap 4 > H: no
    {"minute": 89, "src": 1, "dst": 0},  # outside the window
    {"minute": 100, "src": 1, "dst": 0},  # same minute: invisible
    {"minute": 96, "src": 0, "dst": 0},  # self-loops are never path edges
    {"minute": 96, "src": 1, "dst": 1},
    {"minute": 100, "src": 0, "dst": 0},  # self-loop target: 0
]


def test_hand_case_cycles(rules_cfg):
    frame = engine_tx(CYCLE_CASE)
    spec = make_spec(
        frame, small_cfg(rules_cfg, 10, 3), features_cfg(short=10, long=30, sg=10), hubs=[9]
    )
    ref = Reference.from_frame(frame, spec)
    i0 = frame["row_id"].to_list().index(0)
    assert ref._paths(i0) == (1, 2, 1)
    row = ref.row(i0)
    k = spec.feature_index
    assert (row[k["cyc2_10m"]], row[k["cyc3_10m"]], row[k["cyc4_10m"]]) == (1.0, 2.0, 1.0)
    assert row[spec.i_sev + 3] == 3.0  # round_trip = min(c2 + c3, 100)
    i_self = frame["row_id"].to_list().index(18)
    assert ref._paths(i_self) == (0, 0, 0)
    ora = oracle_features(frame, spec, threads=1)
    assert ora.filter(pl.col("row_id") == 0)["cyc2_10m"].item() == 1.0
    assert ora.filter(pl.col("row_id") == 18)["cyc2_10m"].item() == 0.0


SG_CASE = [
    # target u = 0 -> v = 1 at m = 100; W_sg = 10 -> [90, 99]; account 9 is a hub
    {"minute": 100, "src": 0, "dst": 1},  # 0: target
    {"minute": 91, "src": 2, "dst": 0},  # s = 2 feeds u ...
    {"minute": 92, "src": 2, "dst": 3},  # ... and sibling x = 3 ...
    {"minute": 93, "src": 3, "dst": 1},  # ... which pays v
    {"minute": 94, "src": 4, "dst": 0},  # a second source of the same sibling
    {"minute": 95, "src": 4, "dst": 3},
    {"minute": 94, "src": 5, "dst": 0},  # sibling 6 via source 5
    {"minute": 95, "src": 5, "dst": 6},
    {"minute": 96, "src": 6, "dst": 1},
    {"minute": 94, "src": 9, "dst": 0},  # a hub source: excluded
    {"minute": 95, "src": 9, "dst": 7},
    {"minute": 96, "src": 7, "dst": 1},
    {"minute": 95, "src": 1, "dst": 0},  # s = v: excluded
    {"minute": 95, "src": 1, "dst": 8},
    {"minute": 96, "src": 8, "dst": 1},
    {"minute": 97, "src": 0, "dst": 1},  # x = u: excluded
    {"minute": 100, "src": 2, "dst": 9},  # same minute: invisible
]


def test_hand_case_scatter_gather(rules_cfg):
    frame = engine_tx(SG_CASE)
    spec = make_spec(
        frame, small_cfg(rules_cfg, 10, 3), features_cfg(short=10, long=30, sg=10), hubs=[9]
    )
    ref = Reference.from_frame(frame, spec)
    i0 = frame["row_id"].to_list().index(0)
    assert ref._sg(i0) == (2, 3)  # siblings {3, 6}, sources {2, 4, 5}
    row = ref.row(i0)
    assert (row[spec.feature_index["sg_mids_10m"]], row[spec.feature_index["sg_srcs_10m"]]) == (
        2.0,
        3.0,
    )


def test_hand_case_rule_support_and_formats(rules_cfg):
    rows = [
        {"minute": 50, "src": 0, "dst": 1, "amount_usd": 9500.0, "payment_format": "Cash"},
        {"minute": 45, "src": 0, "dst": 2, "amount_usd": 9000.0, "payment_format": "Cash"},
        {"minute": 46, "src": 0, "dst": 2, "amount_usd": 8999.99, "payment_format": "ACH"},
        {"minute": 47, "src": 0, "dst": 3, "amount_usd": 550.0, "payment_format": "Zelle"},
        {"minute": 48, "src": 4, "dst": 1, "amount_usd": 1.0, "payment_format": "Cash"},
        {"minute": 49, "src": 5, "dst": 1, "amount_usd": 1.0, "payment_format": "Wire"},
        {"minute": 50, "src": 6, "dst": 1, "amount_usd": 1.0, "payment_format": "Zelle"},
        {
            "minute": 49,
            "src": 0,
            "dst": 1,
            "amount_usd": 1.0,
            "amount_paid": 10000.004,
            "payment_format": "ACH",
            "split": "train",
        },
    ]
    frame = engine_tx(rows)
    spec = make_spec(frame, small_cfg(rules_cfg, 10, 5), features_cfg(short=10, long=30, sg=10))
    assert spec.vocab["payment_format"] == ("ACH",)  # fitted on the one train row
    ref = reference_frame(frame, spec)
    ora = oracle_features(frame, spec, threads=1)
    r0 = ref.filter(pl.col("row_id") == 0).row(0, named=True)
    o0 = ora.filter(pl.col("row_id") == 0).row(0, named=True)
    want = {
        "in_band": 1.0,
        "u_out_inband_10m": 1.0,  # 9000.0 is in [9000, 10000); 8999.99 is not
        "u_out_round_10m": 2.0,  # paid 9000.00 and 10000.00 (half away from zero on 1000000.4)
        "u_out_newcp_10m": 3.0,  # 0->2 at 45, 0->3 at 47, 0->1 at 49 (0->2 at 46 repeats)
        "u_out_fmt_ach_10m": 2.0,
        "v_in_same_fmt_10m": 0.0,  # Cash is not in the vocab: unknown matches nothing
        "payment_format": -1.0,
    }
    for name, value in want.items():
        assert r0[name] == value and o0[name] == value, (name, r0[name], o0[name])
    r7 = ref.filter(pl.col("row_id") == 7).row(0, named=True)
    assert r7["round_amount"] == 1.0 and r7["payment_format"] == 0.0


# ---------------------------------------------------------------------------------------------
# run_oracle end to end on a feature table built from the reference


def _write_table(features_dir, tx, table, spec):
    parts = features_dir / PARTS_DIR
    parts.mkdir(parents=True)
    keyed = table.join(tx.select("row_id", "day", "split"), on="row_id", maintain_order="left")
    schema = spec.table_schema()
    keyed = keyed.select([pl.col(c).cast(t) for c, t in schema.items()])
    for (day,), part in keyed.group_by("day", maintain_order=True):
        part.write_parquet(parts / part_name(day))
    write_json_atomic(spec.to_json(), features_dir / FEATURE_SPEC_FILE)


def test_run_oracle_passes_on_a_true_table_and_fails_on_a_tampered_one(
    prepared, fx, fx_ref, tmp_path
):
    spec, tx = fx
    good = tmp_path / "good"
    _write_table(good, tx, fx_ref, spec)
    doc = run_oracle(prepared, good, spec, threads=2, memory_limit=None)
    assert doc["ok"] and doc["rows"] == tx.height and doc["columns_checked"] == 75
    assert doc["n_mismatches"] == 0
    assert read_json(good / "verify" / "verify.json")["total_mismatches"] == 0

    bad = tmp_path / "bad"
    victim = int(tx["row_id"][1234])
    tampered = fx_ref.with_columns(
        pl.when(pl.col("row_id") == victim)
        .then(pl.col("u_out_cnt_1d") + 1)
        .otherwise(pl.col("u_out_cnt_1d"))
        .alias("u_out_cnt_1d")
    )
    _write_table(bad, tx, tampered, spec)
    doc = run_oracle(prepared, bad, spec, threads=2, memory_limit=None)  # the caller decides
    assert doc["n_mismatches"] == 1 and not doc["ok"]
    with pytest.raises(RuntimeError, match="1 mismatching"):
        run_oracle(prepared, bad, spec, threads=2, memory_limit=None, raise_on_mismatch=True)
    doc = read_json(bad / "verify" / "verify.json")
    assert doc["columns"]["u_out_cnt_1d"]["examples"][0]["row_id"] == victim
    assert not doc["ok"]


def test_run_oracle_refuses_another_spec(prepared, fx, fx_ref, tmp_path, rules_cfg):
    spec, tx = fx
    _write_table(tmp_path, tx, fx_ref, spec)
    other = make_spec(tx, rules_cfg, hub_cap=FIXTURE_HUB_CAP + 1)
    with pytest.raises(RuntimeError, match="spec_hash"):
        run_oracle(prepared, tmp_path, other, threads=2)


def test_float32_table_comparison_uses_the_float32_rules(fx, fx_ref, fx_oracle):
    spec, tx = fx
    cols = covered_features(spec)
    f32 = fx_ref.with_columns(pl.col(c).cast(pl.Float32) for c in cols)
    assert_parity(compare_with_engine(f32, fx_oracle, spec, float32=True))
    # A one-ulp32 error in an exact column is caught.
    nudged = f32.with_columns(
        pl.Series("u_out_cnt_1d", np.nextafter(f32["u_out_cnt_1d"].to_numpy(), np.float32(1e9)))
    )
    res = compare_with_engine(nudged, fx_oracle, spec, float32=True)
    assert res["columns"]["u_out_cnt_1d"]["mismatches"] == tx.height


def test_m2_columns_cover_every_moment_feature(fx, fx_oracle):
    spec, _ = fx
    for fd in spec.features:
        assert (f"{M2_PREFIX}{fd.name}" in fx_oracle.columns) == (fd.tol in ("mean", "std"))


# ---------------------------------------------------------------------------------------------
# The engine (needs agents B and C)


def test_engine_equals_oracle_on_fixture(fx, fx_engine, fx_oracle):
    spec, _ = fx
    assert_parity(compare_with_engine(fx_engine, fx_oracle, spec, float32=False))


def test_engine_equals_reference_on_fixture(fx, fx_engine, fx_ref):
    """All 79 features (cyc3/cyc4 and scatter-gather included), the 7 severities and inflow_c.
    At the default budgets nothing is truncated on the fixture."""
    spec, tx = fx
    names = spec.row_layout
    ref_rows = fx_ref.select(names).rows()
    eng_rows = fx_engine.select(names).rows()
    m2cols = [c for c in fx_ref.columns if c.startswith(M2_PREFIX)]
    m2s = fx_ref.select(m2cols).rows(named=True)
    bad = []
    for k, (g, w) in enumerate(zip(eng_rows, ref_rows, strict=True)):
        g = _typed(g, spec)
        m2 = {c[len(M2_PREFIX) :]: v for c, v in m2s[k].items()}
        for msg in compare_rows(g, w, m2, spec):
            bad.append((int(tx["row_id"][k]), msg))
    assert not bad, bad[:10]
    for t in ("rule_trunc", "cyc_trunc", "sg_trunc"):
        assert (fx_engine[t] == 0).all(), t


def _typed(row, spec):
    """A rows_frame row back to the engine's types (floats, then int tail)."""
    return tuple(float(x) if k < spec.i_inflow else int(x) for k, x in enumerate(row))


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_engine_equals_oracle_and_reference_on_dense_frames(rules_cfg, seed):
    require_engine()
    frame, spec = dense_case(seed, rules_cfg)
    rows = run_engine(frame, spec, check_every=50)
    eng = rows_frame(frame, rows, spec)
    assert_parity(
        compare_with_engine(eng, oracle_features(frame, spec, threads=1), spec, float32=False)
    )
    ref = Reference.from_frame(frame, spec)
    ref_rows, m2s = ref.rows()
    bad = [(k, m) for k, r in enumerate(rows) for m in compare_rows(r, ref_rows[k], m2s[k], spec)]
    assert not bad, bad[:10]


def test_engine_equals_oracle_with_rule_windows_unlike_feature_windows(fx, rules_cfg):
    require_engine()
    _, tx = fx
    spec = make_spec(tx, unlike_windows(rules_cfg), hub_cap=FIXTURE_HUB_CAP)
    eng = rows_frame(tx, run_engine(tx, spec), spec)
    assert_parity(
        compare_with_engine(eng, oracle_features(tx, spec, threads=2), spec, float32=False)
    )


@pytest.mark.parametrize(("feat_visits", "rule_visits"), [(1, 2_000_000), (8, 2_000_000), (3, 4)])
def test_engine_truncation_gives_lower_bounds_only(rules_cfg, feat_visits, rule_visits):
    """Tiny budgets (§4.7): a truncated row may only under-count its cycle / scatter-gather
    columns (and the round-trip severity under rule_trunc); every other value stays exact, and
    rule_trunc implies cyc_trunc."""
    require_engine()
    frame = dense_engine_frame(seed=3, n=400)
    feats = features_cfg(**DENSE_FEATURES, feat_visits=feat_visits, rule_visits=rule_visits)
    spec = make_spec(frame, small_cfg(rules_cfg, 10, 5), feats, hub_cap=dense_hub_cap(frame))
    rows = run_engine(frame, spec)
    ref_rows, m2s = Reference.from_frame(frame, spec).rows()
    bad = [(k, m) for k, r in enumerate(rows) for m in compare_rows(r, ref_rows[k], m2s[k], spec)]
    assert not bad, bad[:10]
    rule_t = [r[spec.i_rule_trunc] for r in rows]
    cyc_t = [r[spec.i_cyc_trunc] for r in rows]
    sg_t = [r[spec.i_sg_trunc] for r in rows]
    assert all(c for r, c in zip(rule_t, cyc_t, strict=True) if r)
    assert any(cyc_t) and any(sg_t)
    assert any(rule_t) == (rule_visits < 100)
    # Not vacuous: some truncated row really is a strict lower bound of the reference.
    i4 = spec.feature_index["cyc4_10m"]
    assert any(r[i4] < w[i4] for r, w, t in zip(rows, ref_rows, cyc_t, strict=True) if t)


def test_reference_cents_equal_duckdb_round_and_the_engine_cents():
    """The reference's cents (the §4.1 formula) are DuckDB's CAST(round(x * 100) AS BIGINT), the
    oracle's and M1's definition, and the engine's tx_features.cents."""
    import duckdb

    from aml.features import tx_features
    from tests.fixtures.engine_ref import ref_cents

    rng = np.random.default_rng(0)
    special = [0.0, 0.004, 0.005, 0.015, 1.005, 99.995, 9999.995, 8999.99, 0.125, 2.675, 1e11]
    xs = (
        special
        + rng.uniform(0, 1e12, 2000).tolist()
        + np.round(rng.uniform(0, 1e5, 2000), 3).tolist()
    )
    con = duckdb.connect()
    want = [
        r[0]
        for r in con.execute(
            "SELECT CAST(round(x * 100) AS BIGINT) FROM (SELECT unnest(?) AS x)", [xs]
        ).fetchall()
    ]
    con.close()
    got = [ref_cents(x) for x in xs]
    assert got == want
    if hasattr(tx_features, "cents"):
        assert [tx_features.cents(x) for x in xs] == want
    with pytest.raises(ValueError):
        ref_cents(-0.01)
    with pytest.raises(ValueError):
        ref_cents(math.nan)


def test_oracle_lsum_is_a_true_division(rules_cfg):
    """lsum = log1p(cents / 100) with an IEEE division (§5). polars divides a column by a scalar
    as a multiply by the reciprocal (57 / 100 -> 57 * 0.01, one ulp off), which once put the
    oracle 2 ulp from the engine on a 57-cent window sum."""
    assert 57 / 100 != 57 * 0.01  # the case is not vacuous
    rows = [
        {"minute": 1, "src": 1, "dst": 2, "amount_usd": 0.57, "split": "train"},
        {"minute": 3, "src": 2, "dst": 3, "amount_usd": 10.0},  # u = 2: in-sum and inflow 57
        {"minute": 3, "src": 4, "dst": 2, "amount_usd": 5.0},  # v = 2: in-sum 57
        {"minute": 4, "src": 1, "dst": 5, "amount_usd": 7.0},  # u = 1: out-sum 57
    ]
    frame = engine_tx(rows)
    spec = make_spec(frame, small_cfg(rules_cfg, 10, 5), features_cfg(short=10, long=30, sg=10))
    ora = oracle_features(frame, spec, threads=1)
    want = math.log1p(57 / 100).hex()
    S, P = window_tag(spec.w_short), window_tag(spec.w_pt)
    cases = {1: (f"u_in_sum_{S}", f"u_inflow_{P}"), 2: (f"v_in_sum_{S}",), 3: (f"u_out_sum_{S}",)}
    for rid, cols in cases.items():
        row = ora.filter(pl.col("row_id") == rid).row(0, named=True)
        assert {c: row[c].hex() for c in cols} == dict.fromkeys(cols, want)
