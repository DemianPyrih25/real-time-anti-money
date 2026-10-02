"""The engine spec (M2 spec §3-§5): generated names, groups, registry, validation, JSON, hashing."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from aml.features import spec as S
from aml.features.spec import (
    ABLATION_GROUPS,
    GROUPS,
    NON_MODEL_COLUMNS,
    ROW_TAIL,
    EngineSpec,
    Slot,
    SpecError,
    assert_model_inputs,
    build_features,
    format_slug,
    model_index,
    tol_ok,
    window_tag,
)
from aml.features.tx_features import (
    FORBIDDEN_FEATURES,
    FORBIDDEN_SUBSTRINGS,
    TX_FEATURES,
    fit_vocab,
    is_forbidden,
)
from aml.rules.sql_baseline import SCENARIOS, hub_degree_cap, sql_params

REPO_ROOT = Path(__file__).resolve().parents[2]
HI_SMALL_FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]
VOCAB = {
    "payment_currency": ["Euro", "US Dollar", "Yuan"],
    "receiving_currency": ["Euro", "US Dollar", "Yuan"],
    "payment_format": HI_SMALL_FORMATS,
}

# §5 at the default config and the 7 HI-Small payment formats, in row order.
DEFAULT_NAMES = [
    # TX (9)
    *TX_FEATURES,
    # VEL (16)
    *[
        f"{p}_{w}"
        for p in (
            "u_out_cnt",
            "u_out_uniq",
            "u_in_cnt",
            "u_in_uniq",
            "v_in_cnt",
            "v_in_uniq",
            "v_out_cnt",
            "v_out_uniq",
        )
        for w in ("1d", "3d")
    ],
    # AMT (20)
    *[f"{p}_sum_{w}" for p in ("u_out", "u_in", "v_in", "v_out") for w in ("1d", "3d")],
    *[f"{p}_{s}_{w}" for s in ("mean", "std") for p in ("u_out", "v_in") for w in ("1d", "3d")],
    "u_out_max_1d",
    "v_in_max_1d",
    "u_amt_dev_1d",
    "v_amt_dev_1d",
    # FLOW (6)
    "pair_cnt_1d",
    "pair_cnt_3d",
    "u_inflow_12h",
    "pt_ratio_12h",
    "u_bal_3d",
    "v_bal_3d",
    # PORT (9)
    "pair_is_new",
    "out_port",
    "in_port",
    "u_out_gap",
    "u_in_gap",
    "v_in_gap",
    "v_out_gap",
    "pair_gap",
    "rev_pair_gap",
    # CYC (3)
    "cyc2_2d",
    "cyc3_2d",
    "cyc4_2d",
    # SG (4)
    "sg_mids_1d",
    "sg_srcs_1d",
    "gs_u_1d",
    "gs_v_1d",
    # RULE (12)
    "in_band",
    "u_out_inband_1d",
    "u_out_round_1d",
    "u_out_newcp_1d",
    *[
        f"u_out_fmt_{s}_1d"
        for s in ("ach", "bitcoin", "cash", "cheque", "credit_card", "reinvestment", "wire")
    ],
    "v_in_same_fmt_1d",
]
FORMAT_DERIVED = [
    "payment_format",
    *[n for n in DEFAULT_NAMES if n.startswith("u_out_fmt_")],
    "v_in_same_fmt_1d",
]
GNN_EDGE_ATTRS = [
    *TX_FEATURES,
    "pair_is_new",
    "out_port",
    "in_port",
    "u_out_gap",
    "u_in_gap",
    "v_in_gap",
    "v_out_gap",
    "pair_gap",
    "rev_pair_gap",
]


def _yaml(name: str) -> dict:
    with (REPO_ROOT / "configs" / name).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture
def fcfg() -> dict:
    return copy.deepcopy(_yaml("features.yaml"))


@pytest.fixture
def rcfg() -> dict:
    return copy.deepcopy(_yaml("rules.yaml"))


def make(fcfg: dict, rcfg: dict, **kw) -> EngineSpec:
    args = {"n_accounts": 1000, "vocab": VOCAB, "hub_cap": 115, "hubs": [7, 3, 500]} | kw
    return EngineSpec.from_configs(fcfg, rcfg, **args)


@pytest.fixture
def spec(fcfg: dict, rcfg: dict) -> EngineSpec:
    return make(fcfg, rcfg)


# --- names, groups, metadata --------------------------------------------------------------------


def test_default_names_are_generated_from_configs_and_vocab(spec, fcfg, rcfg):
    assert len(DEFAULT_NAMES) == 79
    assert list(spec.feature_names) == DEFAULT_NAMES
    assert [f.name for f in build_features(fcfg, rcfg, VOCAB)] == DEFAULT_NAMES
    assert build_features(fcfg, rcfg, VOCAB) == spec.features
    assert spec.n_features == 79 and spec.FEATURES is spec.features


def test_groups_partition_the_features(spec):
    counts = Counter(f.group for f in spec.features)
    assert counts == {
        "TX": 9,
        "VEL": 16,
        "AMT": 20,
        "FLOW": 6,
        "PORT": 9,
        "CYC": 3,
        "SG": 4,
        "RULE": 12,
    }
    assert set(counts) == set(GROUPS) and tuple(g for g in GROUPS if g != "TX") == ABLATION_GROUPS
    parts = [n for g in GROUPS for n in spec.group_names(g)]
    assert sorted(parts) == sorted(spec.feature_names) and len(set(parts)) == len(parts)
    assert list(spec.group_names("TX")) == TX_FEATURES  # M1's TX features, in M1 order
    with pytest.raises(KeyError):
        spec.group_names("HUB")


def test_names_are_allowed_model_inputs(spec):
    assert assert_model_inputs(spec.feature_names, spec) == list(spec.feature_names)
    assert spec.assert_model_inputs(reversed(spec.feature_names))
    for name in spec.feature_names:
        assert not is_forbidden(name), name
        assert not any(s in name.lower() for s in FORBIDDEN_SUBSTRINGS), name
        assert name not in NON_MODEL_COLUMNS
    for bad in [*NON_MODEL_COLUMNS, *sorted(FORBIDDEN_FEATURES), "hub_u", "unknown", "event_time"]:
        with pytest.raises(ValueError):
            spec.assert_model_inputs([*spec.feature_names[:3], bad])
    with pytest.raises(ValueError, match="repeated"):
        assert_model_inputs(["self_loop", "in_band", "self_loop"], spec)


def test_feature_metadata(spec):
    f = spec.feature_map
    assert all(x.dtype == "f32" and x.model_input for x in spec.features)
    assert spec.categorical_names == ("payment_currency", "receiving_currency", "payment_format")
    assert list(spec.format_derived_names) == FORMAT_DERIVED and len(FORMAT_DERIVED) == 9
    assert list(spec.gnn_edge_attr_names) == GNN_EDGE_ATTRS
    # tolerance classes (§4.9)
    for name, feat in f.items():
        if name == "log_amount_usd" or any(
            k in name for k in ("_sum_", "u_inflow_", "_port", "_max_")
        ):
            want = "ulp"
        elif "_mean_" in name or "_amt_dev_" in name:
            want = "mean"
        elif "_std_" in name:
            want = "std"
        else:
            want = "exact"
        assert feat.tol == want, name
    # domains, sides, windows, per-format indices
    assert {n for n, x in f.items() if x.domain == "flag"} == {
        "cross_currency",
        "self_loop",
        "same_bank",
        "round_amount",
        "pair_is_new",
        "in_band",
    }
    assert {n for n, x in f.items() if x.domain == "code"} == set(spec.categorical_names)
    assert {x.domain for x in spec.features} <= set(S.DOMAINS)
    assert {x.side for x in spec.features} <= set(S.SIDES)
    assert {x.tol for x in spec.features} <= set(S.TOL_CLASSES)
    assert f["u_out_cnt_3d"].window == 4320 and f["u_out_cnt_3d"].direction == "out"
    assert f["u_inflow_12h"].window == 720 and f["cyc3_2d"].window == 2880
    assert f["pair_is_new"].window is None and f["log_amount_usd"].side == "tx"
    fmt_counts = [x for x in spec.features if x.stat == "fmt_cnt"]
    assert [x.fmt for x in fmt_counts] == list(range(len(HI_SMALL_FORMATS)))
    assert all(x.fmt is None for x in spec.features if x.stat != "fmt_cnt")


@pytest.mark.parametrize(
    ("minutes", "tag"),
    [(720, "12h"), (1440, "1d"), (2880, "2d"), (4320, "3d"), (60, "1h"), (90, "90m"), (1, "1m")],
)
def test_window_tag(minutes, tag):
    assert window_tag(minutes) == tag


@pytest.mark.parametrize("bad", [0, -60, 1.5, True, "1440", None])
def test_window_tag_rejects(bad):
    with pytest.raises(SpecError):
        window_tag(bad)


def test_format_slug():
    assert format_slug("Credit Card") == "credit_card"
    assert format_slug("ACH") == "ach" and format_slug("Wire-Transfer 2") == "wire_transfer_2"


def test_names_follow_config_windows_and_vocab(fcfg, rcfg):
    fcfg["windows"] = {"short": 720, "long": 2880, "sg": 90}
    rcfg["scenarios"]["rapid_pass_through"]["window_minutes"] = 360
    rcfg["scenarios"]["round_trip"].update(window_minutes=4320, hop_window_minutes=60)
    vocab = {**VOCAB, "payment_format": ["Cash", "Wire Transfer", "Bitcoin"]}
    sp = make(fcfg, rcfg, vocab=vocab)
    names = sp.feature_names
    assert sp.n_features == 79 - 7 + 3
    for n in (
        "u_out_cnt_12h",
        "u_out_cnt_2d",
        "u_out_max_12h",
        "pair_cnt_2d",
        "u_inflow_6h",
        "pt_ratio_6h",
        "u_bal_2d",
        "cyc4_3d",
        "sg_mids_90m",
        "gs_v_12h",
        "u_out_fmt_wire_transfer_12h",
        "v_in_same_fmt_12h",
    ):
        assert n in names, n
    assert [x.name for x in sp.features if x.stat == "fmt_cnt"] == [
        "u_out_fmt_cash_12h",
        "u_out_fmt_wire_transfer_12h",
        "u_out_fmt_bitcoin_12h",
    ]
    sp.assert_model_inputs(names)


def test_no_formats_gives_no_format_counts(fcfg, rcfg):
    sp = make(fcfg, rcfg, vocab={**VOCAB, "payment_format": []})
    assert sp.n_features == 72 and not any("u_out_fmt_" in n for n in sp.feature_names)
    assert not [s for s in sp.slots if s.kind == "fmt"]


# --- registry -----------------------------------------------------------------------------------


def _slot_bytes(slots) -> dict[int, int]:
    out: Counter = Counter()
    for s in slots:
        out[s.window] += {"i": 4, "q": 8, "d": 8}[s.typecode]
    return dict(out)


def test_default_registry_matches_the_memory_table(spec):
    slots = spec.slots
    assert list(slots) == sorted(set(slots))
    assert len(slots) == 39 and spec.slot_bytes_per_account == 208
    assert _slot_bytes(slots) == {1440: 136, 4320: 64, 720: 8}  # §4.3 B
    assert spec.windows_all == (720, 1440, 2880, 4320) and spec.W_max == 4320
    assert spec.uniq_windows == (1440, 4320)
    by_window = {w: {(s.side, s.kind, s.pred) for s in slots if s.window == w} for w in (720, 4320)}
    assert by_window[720] == {("in", "nsl_sum_c", -1)}
    assert by_window[4320] == {
        (side, kind, -1) for side in ("out", "in") for kind in ("cnt", "uniq", "sum_c", "s1", "s2")
    }
    short = {(s.side, s.kind) for s in slots if s.window == 1440 and s.kind != "fmt"}
    assert short == {
        *[(side, k) for side in ("out", "in") for k in ("cnt", "uniq", "sum_c", "s1", "s2")],
        ("out", "inband"),
        ("out", "round"),
        ("out", "newcp"),
        ("out", "hr"),
    }
    fmts = sorted((s.side, s.pred) for s in slots if s.kind == "fmt")
    assert fmts == sorted((side, k) for side in ("out", "in") for k in range(7))


def test_registry_supports_rule_windows_other_than_feature_windows(fcfg, rcfg):
    sc = rcfg["scenarios"]
    sc["fan_in_velocity"]["window_minutes"] = 60
    sc["fan_out_velocity"]["window_minutes"] = 120
    sc["rapid_pass_through"]["window_minutes"] = 30
    sc["structuring"]["window_minutes"] = 45
    sc["round_amount_burst"]["window_minutes"] = 1440  # = short: shares the feature slot
    sc["high_risk_format_burst"]["window_minutes"] = 4320  # = long, but no feature hr slot
    sc["round_trip"].update(window_minutes=10, hop_window_minutes=0)
    fcfg["windows"]["sg"] = 15
    sp = make(fcfg, rcfg)
    slots = set(sp.slots)
    assert len(slots) == len(sp.slots)
    for w in (60, 120):
        assert {Slot("in", "uniq", w), Slot("out", "uniq", w)} <= slots
    assert Slot("in", "nsl_sum_c", 30) in slots
    assert {Slot("out", "inband", 45), Slot("out", "inband", 1440)} <= slots
    assert [s for s in sp.slots if s.kind == "round"] == [Slot("out", "round", 1440)]
    assert [s for s in sp.slots if s.kind == "hr"] == [Slot("out", "hr", 4320)]
    assert sp.uniq_windows == (60, 120, 1440, 4320)
    assert sp.windows_all == (10, 15, 30, 45, 60, 120, 1440, 4320) and sp.W_max == 4320
    assert (sp.w_fan_in, sp.w_fan_out, sp.w_pt, sp.w_struct, sp.w_round, sp.w_hr) == (
        60,
        120,
        30,
        45,
        1440,
        4320,
    )
    assert (sp.w_rt, sp.hop) == (10, 0)
    # the feature names follow the rule windows they read
    assert "u_inflow_30m" in sp.feature_names and "cyc2_10m" in sp.feature_names
    # a long rule window raises W_max (the ring keeps what every window needs)
    sc["round_trip"]["window_minutes"] = 10080
    assert make(fcfg, rcfg).W_max == 10080


def test_slot_names_and_typecodes():
    assert Slot("out", "fmt", 1440, 3).name == "slot.out.fmt.1440.3"
    assert Slot("in", "cnt", 720).name == "slot.in.cnt.720"
    assert Slot("in", "s2", 60).typecode == "d" and Slot("out", "sum_c", 60).typecode == "q"
    assert set(S.SLOT_TYPECODES) == {
        "cnt",
        "uniq",
        "sum_c",
        "s1",
        "s2",
        "nsl_sum_c",
        "inband",
        "round",
        "hr",
        "newcp",
        "fmt",
    }


def test_rule_constants_come_from_sql_params(spec, rcfg):
    P = sql_params(rcfg, 115)
    assert spec.sql_params == P and spec.sql_params is not P
    assert spec.hub_cap == 115 and spec.hubs == (3, 7, 500)
    assert spec.excl == (True, False, False, True)  # rules.yaml exclude_hub_senders
    assert S.EXCL_SCENARIOS == (
        "fan_out_velocity",
        "structuring",
        "round_amount_burst",
        "high_risk_format_burst",
    )
    assert (spec.round_cents, spec.band_low_usd, spec.band_high_usd) == (10000, 9000.0, 10000.0)
    assert spec.max_round_trip_paths == 100 and spec.high_risk_formats == ("Cash", "Bitcoin")


# --- validation ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "edit",
    [
        lambda f, r: f["windows"].update(long=1440),  # short == long: names collide
        lambda f, r: f["windows"].update(short=0),
        lambda f, r: f["windows"].update(sg=-5),
        lambda f, r: f["windows"].update(short=True),
        lambda f, r: f["windows"].update(short=1.5),
        lambda f, r: f["windows"].update(short=2**31),
        lambda f, r: f["windows"].update(medium=100),
        lambda f, r: f["caps"].pop("gap"),
        lambda f, r: f.pop("budgets"),
        lambda f, r: f["caps"].update(count=0),
        lambda f, r: f["caps"].update(port=0),
        lambda f, r: f["budgets"].update(feat_visits=0),
        lambda f, r: f["budgets"].update(rule_visits=-1),
        lambda f, r: f["ring"].update(compact_min_rows=0),
        lambda f, r: r["scenarios"]["round_trip"].update(hop_window_minutes=3000),
        lambda f, r: r["scenarios"]["fan_in_velocity"].update(window_minutes=0),
        lambda f, r: r.update(high_risk_formats="Cash"),
        lambda f, r: r.update(high_risk_formats=["Cash", ""]),
        lambda f, r: r.update(high_risk_formats=["Cash", 3]),
    ],
)
def test_config_validation_errors(fcfg, rcfg, edit):
    edit(fcfg, rcfg)
    with pytest.raises(ValueError):
        make(fcfg, rcfg)


@pytest.mark.parametrize(
    "kw",
    [
        {"n_accounts": 0},
        {"n_accounts": 2**31},
        {"hub_cap": -1},
        {"hubs": [3, 3]},
        {"hubs": [1000]},  # >= n_accounts
        {"hubs": [-1]},
        {"hubs": "123"},
        {"vocab": {k: v for k, v in VOCAB.items() if k != "payment_format"}},
        {"vocab": {**VOCAB, "payment_format": ["ACH", "ACH"]}},
        {"vocab": {**VOCAB, "payment_format": ["Credit Card", "credit-card"]}},  # slug collision
        {"vocab": {**VOCAB, "payment_format": ["ACH", ""]}},
        {"vocab": {**VOCAB, "payment_currency": "Euro"}},
    ],
)
def test_input_validation_errors(fcfg, rcfg, kw):
    with pytest.raises(ValueError):  # SpecError, or M1's ValueError from sql_params (hub_cap)
        make(fcfg, rcfg, **kw)


def test_validation_messages_name_the_problem(fcfg, rcfg):
    fcfg["windows"]["long"] = fcfg["windows"]["short"]
    with pytest.raises(SpecError, match="must differ"):
        make(fcfg, rcfg)


def test_inputs_are_normalised(fcfg, rcfg):
    sp = make(fcfg, rcfg, hubs=np.array([9, 2], dtype=np.int64), n_accounts=np.int64(50))
    assert sp.hubs == (2, 9) and type(sp.hubs[0]) is int and sp.n_accounts == 50
    assert sp.vocab["payment_format"] == tuple(HI_SMALL_FORMATS)
    assert make(fcfg, rcfg, hubs=[]).hubs == ()


# --- JSON, hashing, identity --------------------------------------------------------------------


def test_json_round_trip(spec):
    doc = spec.to_json()
    text = json.dumps(doc, allow_nan=False)
    for src in (doc, text, json.loads(text)):
        back = EngineSpec.from_json(src)
        assert back == spec and back.spec_hash() == spec.spec_hash()
        assert back.features == spec.features and back.slots == spec.slots
    assert doc["engine_version"] == S.ENGINE_VERSION and doc["spec_hash"] == spec.spec_hash()
    assert doc["inputs"]["hubs"] == [3, 7, 500]
    assert [f["name"] for f in doc["derived"]["features"]] == DEFAULT_NAMES
    assert doc["derived"]["row_layout"] == [*DEFAULT_NAMES, *ROW_TAIL]
    # callers may extend the document (the serving bundle adds its model inputs)
    assert EngineSpec.from_json({**doc, "model_inputs": DEFAULT_NAMES[:5]}) == spec


def test_spec_hash_is_stable(fcfg, rcfg, spec):
    h = spec.spec_hash()
    assert len(h) == 16 and int(h, 16) >= 0
    assert make(copy.deepcopy(fcfg), copy.deepcopy(rcfg)).spec_hash() == h
    # dict key order does not matter
    rev = {k: fcfg[k] for k in reversed(list(fcfg))}
    rrev = {k: rcfg[k] for k in reversed(list(rcfg))}
    vocab_rev = {k: VOCAB[k] for k in reversed(list(VOCAB))}
    assert make(rev, rrev, vocab=vocab_rev, hubs=[500, 7, 3]).spec_hash() == h
    # driver-only sections are not engine inputs
    fcfg["replay"]["batch_rows"] = 7
    fcfg["bench"]["last_day"] = 1
    assert make(fcfg, rcfg).spec_hash() == h
    # the grids and alert rates are not either
    rcfg["alert_rate"] = 0.002
    rcfg["scenarios"]["structuring"]["grid"] = [1, 2]
    assert make(fcfg, rcfg).spec_hash() == h


@pytest.mark.parametrize(
    "change",
    [
        lambda f, r, kw: kw.update(n_accounts=1001),
        lambda f, r, kw: kw.update(hubs=[3, 7]),
        lambda f, r, kw: kw.update(hub_cap=116),
        lambda f, r, kw: kw.update(vocab={**VOCAB, "payment_format": HI_SMALL_FORMATS[::-1]}),
        lambda f, r, kw: f["windows"].update(sg=720),
        lambda f, r, kw: f["caps"].update(port=255),
        lambda f, r, kw: f["budgets"].update(feat_visits=10),
        lambda f, r, kw: f["ring"].update(compact_min_rows=10),
        lambda f, r, kw: r["scenarios"]["round_trip"].update(hop_window_minutes=720),
        lambda f, r, kw: r["scenarios"]["structuring"].update(exclude_hub_senders=True),
        lambda f, r, kw: r.update(high_risk_formats=["Cash"]),
        lambda f, r, kw: r.update(round_unit=1000),
    ],
)
def test_spec_hash_changes_with_every_input(fcfg, rcfg, spec, change):
    kw: dict = {}
    change(fcfg, rcfg, kw)
    assert make(fcfg, rcfg, **kw).spec_hash() != spec.spec_hash()


def test_spec_hash_changes_with_engine_version(fcfg, rcfg, spec, monkeypatch):
    doc = spec.to_json()
    monkeypatch.setattr(S, "ENGINE_VERSION", S.ENGINE_VERSION + 1)
    assert make(fcfg, rcfg).spec_hash() != spec.spec_hash()
    with pytest.raises(SpecError, match="engine version"):
        EngineSpec.from_json(doc)


@pytest.mark.parametrize(
    "tamper",
    [
        lambda d: d.update(format=2),
        lambda d: d.update(spec_hash="0" * 16),
        lambda d: d["inputs"].update(hubs=[3, 7]),  # inputs changed under the stored hash
        lambda d: d.pop("inputs"),
    ],
)
def test_from_json_refuses_mismatches(spec, tamper):
    doc = copy.deepcopy(spec.to_json())
    tamper(doc)
    with pytest.raises(SpecError):
        EngineSpec.from_json(doc)


def test_spec_is_frozen_hashable_and_picklable(spec):
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.w_short = 1  # type: ignore[misc]
    assert hash(spec) == hash(EngineSpec.from_json(spec.to_json()))
    back = pickle.loads(pickle.dumps(spec))  # Modal ships arguments by pickle
    assert back == spec and back.spec_hash() == spec.spec_hash()
    assert copy.deepcopy(spec) == spec


# --- constants and layouts ----------------------------------------------------------------------


def test_input_and_event_constants():
    assert S.INPUT_COLUMNS == (
        "rank",
        "row_id",
        "minute",
        "src",
        "dst",
        "amount_usd",
        "amount_paid",
        "payment_format",
        "payment_currency",
        "receiving_currency",
        "from_bank",
        "to_bank",
    )
    assert S.DRIVER_COLUMNS == ("day", "split")
    assert not {"is_laundering", "label", "attempt_id"} & set(S.INPUT_COLUMNS)
    index = {name: getattr(S, f"E_{name.upper()}") for name in S.EVENT_FIELDS}
    assert list(index.values()) == list(range(len(S.EVENT_FIELDS))) == list(range(14))
    assert (S.E_RANK, S.E_MINUTE, S.E_USD_C, S.E_L, S.E_FLAGS, S.E_HOUR) == (0, 2, 5, 6, 12, 13)
    bits = list(S.FLAG_BITS.values())
    assert len(set(bits)) == 7 and all(b & (b - 1) == 0 and 0 < b < 256 for b in bits)
    assert S.FLAG_BITS["F_NEW"] == S.F_NEW and S.FLAG_BITS["F_SELF"] == S.F_SELF


def test_row_layout_and_indices(spec):
    assert S.SEVERITY_COLUMNS == SCENARIOS
    assert S.TRUNC_COLUMNS == ("rule_trunc", "cyc_trunc", "sg_trunc")
    assert (*SCENARIOS, "inflow_c", "rule_trunc", "cyc_trunc", "sg_trunc") == ROW_TAIL
    assert ("row_id", "rank", "day", "split", *ROW_TAIL) == NON_MODEL_COLUMNS
    assert spec.row_layout == (*DEFAULT_NAMES, *ROW_TAIL) and spec.ROW_LAYOUT is spec.row_layout
    assert spec.row_len == 79 + 11
    lay = spec.row_layout
    assert lay[spec.i_sev : spec.i_sev + 7] == SCENARIOS
    assert lay[spec.i_inflow] == "inflow_c" and lay[spec.i_rule_trunc] == "rule_trunc"
    assert lay[spec.i_cyc_trunc] == "cyc_trunc" and lay[spec.i_sg_trunc] == "sg_trunc"
    names = ["cyc3_2d", "log_amount_usd", "in_band"]
    idx = model_index(names, spec)
    assert idx == spec.model_index(names) == (61, 0, 67)
    assert [lay[i] for i in idx] == names
    for bad in (["rule_trunc"], ["nope"]):
        with pytest.raises(ValueError):
            spec.model_index(bad)


def test_table_schema(spec):
    schema = spec.table_schema()
    assert list(schema) == ["row_id", "rank", "day", "split", *spec.row_layout]
    assert schema["row_id"] == pl.Int64 and schema["day"] == pl.Int16
    assert schema["split"] == pl.String
    assert all(schema[n] == pl.Float32 for n in spec.feature_names)
    assert all(schema[s] == pl.Float64 for s in SCENARIOS)
    assert schema["inflow_c"] == pl.Int64 and all(schema[t] == pl.Int8 for t in S.TRUNC_COLUMNS)


def test_state_column_constants():
    ring = dict(S.RING_COLUMNS)
    assert list(ring) == [
        "minute",
        "src",
        "dst",
        "usd_c",
        "l",
        "fmt",
        "flags",
        "pid",
        "prev_out",
        "prev_in",
        "pmax_out",
        "pmax_in",
    ]
    size = {"i": 4, "q": 8, "d": 8, "b": 1, "B": 1}
    assert sum(size[t] for t in ring.values()) == 50  # §4.3 A: 50 B per ring row
    assert set(S.RingView.__annotations__) == {"base", "live_start", "end_rank", *ring}
    assert [n for n, _ in S.ACCOUNT_COLUMNS] == [
        "last_out",
        "last_in",
        "ever_out",
        "ever_in",
        "head_out",
        "head_in",
    ]
    assert [n for n, _ in S.PAIR_COLUMNS] == ["port_out", "port_in", "last_min"]
    assert S.FlushStats(3, 5, 0.25)._fields == ("n_applied", "n_expired", "seconds")


def test_exceptions():
    for exc in (S.LateEventError, S.RankGapError, S.NumericRangeError, S.SnapshotError):
        assert issubclass(exc, S.EngineError)
    assert issubclass(S.SpecError, ValueError) and issubclass(S.GateStopError, Exception)


def test_part_helpers(tmp_path):
    assert S.part_name(7) == "part-d07.parquet" and S.part_name(18) == "part-d18.parquet"
    with pytest.raises(FileNotFoundError):
        S.part_paths(tmp_path)
    parts = tmp_path / S.PARTS_DIR
    parts.mkdir()
    for day, rows in ((10, [3]), (2, [1]), (9, [2])):
        pl.DataFrame({"rank": rows, "x": [day]}).write_parquet(parts / S.part_name(day))
    (parts / ".part-d11.parquet.tmp").write_bytes(b"partial")
    assert [p.name for p in S.part_paths(tmp_path)] == [
        "part-d02.parquet",
        "part-d09.parquet",
        "part-d10.parquet",
    ]
    assert S.scan_feature_table(tmp_path, ["rank"]).collect()["rank"].to_list() == [1, 2, 3]


# --- tolerance classes --------------------------------------------------------------------------


def _up(x: float, n: int = 1) -> float:
    for _ in range(n):
        x = math.nextafter(x, math.inf)
    return x


def test_tol_exact_and_ulp():
    nan = float("nan")
    assert tol_ok("exact", [1.5, nan, 0.0], [1.5, nan, 0.0]).all()
    assert not tol_ok("exact", [0.0], [-0.0]).any()  # bit-equal, not ==
    assert not tol_ok("exact", [_up(1.5)], [1.5]).any()
    assert not tol_ok("exact", [nan], [1.0]).any()
    # ulp: bit-equal on the same path; <= 1 ulp of the compared dtype otherwise
    assert not tol_ok("ulp", [_up(3.0)], [3.0]).any()
    assert tol_ok("ulp", [_up(3.0)], [3.0], independent=True).all()
    assert not tol_ok("ulp", [_up(3.0, 2)], [3.0], independent=True).any()
    assert tol_ok("ulp", [nan], [nan], independent=True).all()
    assert not tol_ok("ulp", [nan], [3.0], independent=True).any()
    # float32 table: the same cast for exact, 1 ulp32 for ulp
    x = 0.1 + 1e-12
    assert tol_ok("exact", np.float32(x), x, float32=True).all()
    up32 = np.nextafter(np.float32(x), np.float32(1.0))
    assert not tol_ok("exact", up32, x, float32=True).any()
    assert tol_ok("ulp", up32, x, float32=True).all()
    assert not tol_ok("ulp", np.nextafter(up32, np.float32(1.0)), x, float32=True).any()


def test_tol_mean_and_std():
    m2 = 400.0  # sqrt = 20: mean bound 2e-8, std bound sqrt(4e-7)
    assert tol_ok("mean", 5.0 + 1.9e-8, 5.0, m2).all()
    assert not tol_ok("mean", 5.0 + 2.1e-8, 5.0, m2).any()
    assert tol_ok("mean", 5.0 + 0.9e-9, 5.0, 0.0).all()  # max(1, sqrt(m2))
    bound = math.sqrt(1e-9 * 400.0)
    assert tol_ok("std", 0.5 + 0.99 * bound, 0.5, m2).all()
    assert not tol_ok("std", 0.5 + 1.01 * bound, 0.5, m2).any()
    # identical amounts: std ~1e-7 instead of 0 is inside the bound
    assert tol_ok("std", 1e-7, 0.0, 25.0).all()
    assert tol_ok("mean", [float("nan")], [float("nan")], [1.0]).all()
    # float32 table: + 2 ulp32
    g = np.float32(5.0)
    assert tol_ok("mean", np.nextafter(np.nextafter(g, 9), 9), 5.0, 1.0, float32=True).all()
    assert not tol_ok("mean", np.nextafter(g, 9), 5.0, 1.0).any()
    with pytest.raises(ValueError, match="m2"):
        tol_ok("std", 1.0, 1.0)
    with pytest.raises(ValueError):
        tol_ok("loose", 1.0, 1.0)


# --- configs and the fixture --------------------------------------------------------------------


def test_config_files():
    f = _yaml("features.yaml")
    assert f["windows"] == {"short": 1440, "long": 4320, "sg": 1440}
    assert f["caps"] == {"count": 100, "port": 16383, "gap": 4320}
    assert f["budgets"]["rule_visits"] > 1084**2  # never hit on HI-Small (max in-degree 1,084)
    assert f["snapshots"]["boundaries"] == ["val_early", "test"]
    assert f["memory_target_mb"] == 1000 and f["bench"]["last_day"] == 3
    assert _yaml("serving.yaml") == {"replay": {"max_events": 100000}}
    g = _yaml("lgbm.yaml")["graph"]
    assert set(g) == {"optuna", "gate", "ablation", "shap"}
    assert g["gate"]["min_train_nonzero"] == 100 and g["ablation"]["margin_std"] == 2.0


def test_spec_from_the_prepared_fixture(prepared, rules_cfg, fcfg):
    from aml.rules.sql_baseline import connect, register_transactions

    tx = pl.read_parquet(prepared.transactions)
    vocab = fit_vocab(tx.filter(pl.col("split") == "train"))
    con = connect(threads=1)
    try:
        register_transactions(con, prepared.transactions)
        cap = hub_degree_cap(con, rules_cfg["hub_degree_quantile"])
    finally:
        con.close()
    n = pl.read_parquet(prepared.accounts).height
    sp = EngineSpec.from_configs(fcfg, rules_cfg, n_accounts=n, vocab=vocab, hub_cap=cap, hubs=[])
    assert sp.n_features == 72 + len(vocab["payment_format"])
    sp.assert_model_inputs(sp.feature_names)
    assert EngineSpec.from_json(json.dumps(sp.to_json())) == sp
