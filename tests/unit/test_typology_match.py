"""The typology matchers (M6): the fixed decision list (baseline), its evidence, the inputs read
through the engine spec's feature names, accuracy and the grid tuning; the decision tree
(primary): the JSON walk with missing values, its evidence, the round trip and the val fit with
cross-validation."""

from __future__ import annotations

import copy
import dataclasses
import json
import math

import pytest
import yaml

from aml.data.patterns import LABEL_TYPOLOGIES
from aml.eval.typology import TYPOLOGIES as EVAL_TYPOLOGIES
from aml.explain.typology_match import (
    MAX_GRID,
    MISSING_SPLIT,
    ORDER,
    OTHER,
    ROLES,
    TREE_FIT,
    TYPOLOGIES,
    MatcherParams,
    RulesMatcher,
    TreeMatcher,
    accuracy,
    check_grid,
    fit_tree,
    input_names,
    inputs,
    match,
    tree_fit_params,
    tune,
)
from aml.features.spec import EngineSpec
from tests.fixtures.serving_bundle import CONFIG_DIR

# The matcher's engine features at the shipped configs (windows 1 d / 2 d / 1 d / 12 h).
NAMES = {
    "cyc2": "cyc2_2d",
    "cyc3": "cyc3_2d",
    "cyc4": "cyc4_2d",
    "sg_mids": "sg_mids_1d",
    "sg_srcs": "sg_srcs_1d",
    "gs_u": "gs_u_1d",
    "u_out_uniq": "u_out_uniq_1d",
    "u_in_uniq": "u_in_uniq_1d",
    "v_in_uniq": "v_in_uniq_1d",
    "pt_ratio": "pt_ratio_12h",
}
P = MatcherParams()  # cycle 1, sg 1, gs 2, fan-out 5, fan-in 5, stack 0.2, bipartite 2, random 1/1


def feats(**roles: float | None) -> dict[str, float | None]:
    """Matcher inputs by role (all counts 0, pt_ratio undefined unless given)."""
    base: dict[str, float | None] = dict.fromkeys(ROLES, 0.0)
    base["pt_ratio"] = None
    base.update(roles)
    return {NAMES[r]: v for r, v in base.items()}


def load_yaml(name: str) -> dict:
    with (CONFIG_DIR / name).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def small_spec(**windows: int) -> EngineSpec:
    feats_cfg = load_yaml("features.yaml")
    feats_cfg["windows"].update(windows)
    cats = {
        "payment_currency": "US Dollar",
        "receiving_currency": "US Dollar",
        "payment_format": "ACH",
    }
    vocab = {k: [v] for k, v in cats.items()}
    return EngineSpec.from_configs(
        feats_cfg, load_yaml("rules.yaml"), n_accounts=4, vocab=vocab, hub_cap=1000, hubs=[]
    )


def test_labels_are_the_ground_truth_vocabulary():
    assert TYPOLOGIES == EVAL_TYPOLOGIES == tuple(LABEL_TYPOLOGIES)
    assert (*ORDER, OTHER) == (
        "CYCLE",
        "SCATTER-GATHER",
        "GATHER-SCATTER",
        "FAN-OUT",
        "FAN-IN",
        "STACK",
        "BIPARTITE",
        "RANDOM",
        "OTHER",
    )
    assert set(ORDER) | {OTHER} == set(TYPOLOGIES)


@pytest.mark.parametrize(
    "roles, label, snippet",
    [
        (
            dict(cyc3=2),
            "CYCLE",
            "closes 2 temporal cycles back to the sender in the previous 2 days "
            "(cyc2_2d = 0, cyc3_2d = 2, cyc4_2d = 0)",
        ),
        (
            dict(sg_mids=3, sg_srcs=1),
            "SCATTER-GATHER",
            "3 sibling accounts of the sender fed by 1 common source in the previous day "
            "(sg_mids_1d = 3, sg_srcs_1d = 1)",
        ),
        (dict(gs_u=4), "GATHER-SCATTER", "at least 4 distinct accounts in the previous day"),
        (
            dict(u_out_uniq=14),
            "FAN-OUT",
            "the sender paid 14 distinct receivers in the previous day (u_out_uniq_1d = 14)",
        ),
        (
            dict(v_in_uniq=14),
            "FAN-IN",
            "the receiver had 14 distinct payers in the previous day (v_in_uniq_1d = 14)",
        ),
        (dict(pt_ratio=0.95), "STACK", "95% of the sender's inflow in the previous 12 hours"),
        (dict(u_out_uniq=3, v_in_uniq=2), "BIPARTITE", "a many-to-many link"),
        (dict(u_in_uniq=1), "RANDOM", "a hop in a chain (u_in_uniq_1d = 1, u_out_uniq_1d = 0)"),
        (dict(), "OTHER", "no typology rule fired"),
    ],
)
def test_each_rule_names_its_typology_with_evidence(roles, label, snippet):
    got, evidence = match(feats(**roles), P)
    assert got == label
    assert len(evidence) == 1 and snippet in evidence[0], evidence


def test_the_fixed_order_decides_and_later_rules_become_evidence():
    f = feats(cyc2=1, v_in_uniq=9, u_out_uniq=3)
    label, evidence = match(f, P)
    assert label == "CYCLE" and len(evidence) == 3
    assert evidence[1].startswith("also consistent with FAN-IN: the receiver had 9 distinct")
    assert evidence[2].startswith("also consistent with BIPARTITE: ")
    label, evidence = match(f, MatcherParams(cycle_min=None))  # null switches a rule off
    assert label == "FAN-IN" and len(evidence) == 2
    assert match(f, MatcherParams(cycle_min=2))[0] == "FAN-IN"  # thresholds are inclusive (>=)
    assert match(feats(cyc2=1, cyc4=1), MatcherParams(cycle_min=2))[0] == "CYCLE"


def test_random_out_max_null_means_unbounded():
    f = feats(u_in_uniq=1, u_out_uniq=9)
    off = {"fan_out_min": None, "bipartite_min": None}
    assert match(f, MatcherParams(**off))[0] == OTHER  # 9 receivers > random_out_max 1
    assert match(f, MatcherParams(**off, random_out_max=None))[0] == "RANDOM"
    assert match(f, MatcherParams(**off, random_out_max=None, random_in_min=None))[0] == OTHER


@pytest.mark.parametrize(
    "pt, fires",
    [(0.8, True), (1.2, True), (1.0, True), (0.79, False), (1.21, False), (None, False),
     (math.nan, False)],
)  # fmt: skip
def test_stack_is_a_symmetric_band_and_undefined_never_fires(pt, fires):
    assert (match(feats(pt_ratio=pt), P)[0] == "STACK") is fires


def test_undefined_inputs_never_fire():
    assert match({n: None for n in NAMES.values()}, P) == (
        OTHER,
        ["no typology rule fired: the alert matches no known laundering pattern"],
    )
    assert match(feats(cyc2=None, cyc3=1), P)[0] == "CYCLE"  # an undefined count adds 0


def test_inputs_are_found_by_role_under_any_window_tag():
    f = {k.rsplit("_", 1)[0] + "_12h": v for k, v in feats(u_out_uniq=7).items()}
    label, evidence = match(f, P)
    assert label == "FAN-OUT" and "in the previous 12 hours (u_out_uniq_12h = 7)" in evidence[0]
    missing = dict(feats())
    del missing["gs_u_1d"]
    with pytest.raises(KeyError, match="gs_u"):
        match(missing, P)
    with pytest.raises(KeyError, match="u_out_uniq"):
        match({**feats(), "u_out_uniq_3d": 1.0}, P)  # two windows of one role: ambiguous


def test_input_names_follow_the_spec_windows():
    spec = small_spec()
    assert input_names(spec) == NAMES
    row = [float(i) for i in range(spec.row_len)]
    row[spec.feature_index["pt_ratio_12h"]] = math.nan
    got = inputs(row, spec)
    assert list(got) == list(NAMES.values())
    assert got["pt_ratio_12h"] is None
    assert all(got[n] == spec.feature_index[n] for n in got if n != "pt_ratio_12h")
    other = small_spec(short=720)
    assert input_names(other)["u_out_uniq"] == "u_out_uniq_12h"
    assert input_names(other)["gs_u"] == "gs_u_12h"
    names = input_names(other)
    names["cyc2"] = "changed"  # a copy: the cached names stay intact
    assert input_names(other)["cyc2"] == "cyc2_2d"


def test_matcher_params_are_checked_exactly():
    cfg = load_yaml("explain.yaml")["typology"]
    p = MatcherParams.from_dict(cfg)
    assert p.to_dict() == cfg and MatcherParams.from_dict(p.to_dict()) == p
    with pytest.raises(ValueError, match="unknown"):
        MatcherParams.from_dict({**cfg, "extra": 1})
    short = dict(cfg)
    del short["gs_min"]
    with pytest.raises(ValueError, match="missing"):
        MatcherParams.from_dict(short)
    for bad in (True, -1, "3", math.inf, math.nan):
        with pytest.raises(ValueError):
            MatcherParams(fan_in_min=bad)
    with pytest.raises(ValueError):
        MatcherParams.from_dict([1, 2])


def test_grid_checks():
    grid = load_yaml("explain.yaml")["typology_grid"]
    keys = check_grid(grid)
    fields = list(MatcherParams().to_dict())
    assert list(keys) == [k for k in fields if k in grid]
    assert math.prod(len(v) for v in grid.values()) <= MAX_GRID
    for bad in (
        {},
        {"nope": [1]},
        {"sg_min": []},
        {"sg_min": "12"},
        {"sg_min": [1, -2]},
        {"sg_min": list(range(11)), "gs_min": list(range(10)), "fan_in_min": list(range(10))},
    ):
        with pytest.raises(ValueError):
            check_grid(bad)


def test_accuracy_counts_confusion_and_baselines():
    rows = [
        (feats(v_in_uniq=9), "FAN-IN"),
        (feats(v_in_uniq=9), "FAN-OUT"),
        (feats(), "OTHER"),
        (feats(cyc2=1), "CYCLE"),
        (feats(), "STACK"),
    ]
    acc = accuracy(rows, P)
    assert (acc["n"], acc["correct"], acc["accuracy"]) == (5, 3, 0.6)
    per = acc["per_typology"]
    assert per["FAN-IN"] == {
        "support": 1,
        "predicted": 2,
        "correct": 1,
        "recall": 1.0,
        "precision": 0.5,
    }
    assert per["FAN-OUT"]["recall"] == 0.0 and per["FAN-OUT"]["precision"] is None
    assert per["BIPARTITE"]["recall"] is None and per["BIPARTITE"]["support"] == 0
    assert acc["macro_recall"] == pytest.approx(0.6)
    assert acc["majority"] == {"label": "FAN-OUT", "accuracy": 0.2}  # ties -> TYPOLOGIES order
    assert acc["confusion"]["FAN-OUT"]["FAN-IN"] == 1 and acc["confusion"]["STACK"][OTHER] == 1
    assert sum(sum(r.values()) for r in acc["confusion"].values()) == 5
    assert set(acc["confusion"]) == set(TYPOLOGIES) == set(acc["confusion"]["CYCLE"])
    assert acc["params"] == P.to_dict()
    assert json.loads(json.dumps(acc, allow_nan=False)) == acc
    empty = accuracy([], P)
    assert empty["n"] == 0 and empty["accuracy"] is None and empty["macro_recall"] is None
    assert empty["majority"] == {"label": None, "accuracy": None}
    with pytest.raises(ValueError, match="unknown ground-truth"):
        accuracy([(feats(), "LAYERING")], P)


def test_tune_maximises_accuracy_with_ties_to_the_first_point():
    rows = [(feats(v_in_uniq=4), "FAN-IN")] * 3 + [(feats(v_in_uniq=3), OTHER)] * 2
    best, info = tune(rows, {"fan_in_min": [3, 4, 5]})
    assert best == MatcherParams(fan_in_min=4)
    assert info["combinations"] == 3 and info["n_rows"] == 5
    assert info["best"]["accuracy"] == 1.0 and info["best"] == accuracy(rows, best)
    assert info["grid"] == {"fan_in_min": [3, 4, 5]} and info["base"] == P.to_dict()
    assert tune(rows, {"fan_in_min": [5, 6]})[0].fan_in_min == 5  # a tie: the first value
    # Keys run in MatcherParams field order (sg_min before fan_in_min), values as listed.
    tied, info = tune(rows, {"fan_in_min": [6, 5], "sg_min": [2, 1]})
    assert (tied.sg_min, tied.fan_in_min) == (2, 6) and info["combinations"] == 4
    assert list(info["grid"]) == ["sg_min", "fan_in_min"]
    base = MatcherParams(cycle_min=None, random_out_max=3)
    got, info = tune(rows, {"fan_in_min": [4]}, base=base)
    assert got == dataclasses.replace(base, fan_in_min=4) and info["base"] == base.to_dict()
    first, info = tune([], {"gs_min": [3, 2]})
    assert first.gs_min == 3 and info["n_rows"] == 0 and info["best"]["accuracy"] is None


def test_tune_on_the_config_grid_is_well_formed():
    cfg = load_yaml("explain.yaml")
    rows = [
        (feats(cyc3=1), "CYCLE"),
        (feats(sg_mids=2, sg_srcs=1), "SCATTER-GATHER"),
        (feats(u_out_uniq=6), "FAN-OUT"),
        (feats(v_in_uniq=3), "FAN-IN"),
        (feats(pt_ratio=0.75), "STACK"),
        (feats(), OTHER),
    ]
    base = MatcherParams.from_dict(cfg["typology"])
    best, info = tune(rows, copy.deepcopy(cfg["typology_grid"]), base=base)
    assert info["combinations"] == math.prod(len(v) for v in cfg["typology_grid"].values())
    assert best.fan_in_min == 3 and best.stack_pt_tol == 0.3  # the points that fit these rows
    assert info["best"]["correct"] == 6
    assert json.loads(json.dumps(info, allow_nan=False)) == info


def test_accuracy_takes_matcher_objects_and_callables():
    rows = [
        (feats(v_in_uniq=9), "FAN-IN"),
        (feats(), "OTHER"),
        (feats(cyc2=1), "STACK"),
    ]
    by_params = accuracy(rows, P)
    assert by_params["model"] == "rules" and by_params["correct"] == 2
    # A case pack's features hold more than the list's inputs (two windows of one role).
    wide = [({**f, "u_out_uniq_3d": 99.0, "g_n_edges": 3}, t) for f, t in rows]
    rules = RulesMatcher(P, tuple(NAMES.values()))
    assert accuracy(wide, rules) == by_params
    assert rules.match(wide[0][0]) == match(rows[0][0], P)
    assert rules.label(wide[2][0]) == "CYCLE"
    with pytest.raises(KeyError, match="u_out_uniq"):
        accuracy(wide, P)  # the bare thresholds cannot narrow a wide dict
    with pytest.raises(KeyError):
        rules.match({"g_n_edges": 3})
    by_label = accuracy(rows, lambda f: "FAN-IN")
    by_pair = accuracy(rows, lambda f: ("FAN-IN", ["why"]))
    assert by_label == by_pair and by_label["model"] == "callable" and by_label["params"] is None
    assert (by_label["correct"], by_label["confusion"]["STACK"]["FAN-IN"]) == (1, 1)
    with pytest.raises(ValueError, match="unknown typology"):
        accuracy(rows, lambda f: "LAYERING")
    with pytest.raises(TypeError):
        accuracy(rows, 3)


# --- the decision tree ----------------------------------------------------------------------------

TREE = {
    "format": 1,
    "tree": {
        "feature": "g_cycle_back",
        "threshold": 0.5,
        "left": {
            "feature": "u_out_uniq_1d",
            "threshold": 3.5,
            "left": {"label": "OTHER", "n": 20, "dist": {"STACK": 5, "OTHER": 15}},
            "right": {"label": "FAN-OUT", "n": 34, "dist": {"FAN-OUT": 28, "GATHER-SCATTER": 6}},
        },
        "right": {"label": "CYCLE", "n": 10, "dist": {"CYCLE": 10}},
    },
    "trained_on": None,
}
PLACEHOLDER = CONFIG_DIR / "typology_tree.json"


def test_the_tree_walk_and_its_evidence():
    t = TreeMatcher.from_dict(TREE)
    label, evidence = t.match({"g_cycle_back": 0, "u_out_uniq_1d": 7.0, "other": 1.0})
    assert label == "FAN-OUT" == t.label({"g_cycle_back": 0, "u_out_uniq_1d": 7.0})
    assert evidence == [
        "no earlier payment path of at most two hops leads from the receiver back to the sender "
        "(g_cycle_back = 0 <= 0.5)",
        "the sender paid 7 distinct counterparties in the previous day (u_out_uniq_1d = 7 > 3.5)",
        "82% of 34 validation cases at this leaf were FAN-OUT",
    ]
    label, evidence = t.match({"g_cycle_back": 1, "u_out_uniq_1d": 0})
    assert label == "CYCLE" and len(evidence) == 2
    assert evidence[0].startswith("an earlier payment path of at most two hops leads from")
    assert evidence[1] == "100% of 10 validation cases at this leaf were CYCLE"
    assert t.label({"g_cycle_back": 0, "u_out_uniq_1d": 3.5}) == OTHER  # left = value <= threshold


@pytest.mark.parametrize("missing", [{}, {"g_cycle_back": None, "u_out_uniq_1d": math.nan}])
def test_missing_and_undefined_values_go_left(missing):
    t = TreeMatcher.from_dict(TREE)
    label, evidence = t.match(missing)
    assert label == OTHER == t.label(missing)
    assert evidence == [
        "g_cycle_back is undefined (missing values take the <= 0.5 branch)",
        "u_out_uniq_1d is undefined (missing values take the <= 3.5 branch)",
        "75% of 20 validation cases at this leaf were OTHER",
    ]


def test_tree_round_trip_and_shape():
    t = TreeMatcher.from_dict(TREE)
    doc = t.to_dict()
    assert doc == TREE and json.loads(json.dumps(doc, allow_nan=False)) == doc
    assert TreeMatcher.from_dict(doc) == t
    doc["tree"]["threshold"] = 9.0  # a deep copy: the matcher is unchanged
    assert t.root["threshold"] == 0.5
    shape = (t.trivial, t.depth, t.n_leaves, t.inputs)
    assert shape == (False, 2, 3, ("g_cycle_back", "u_out_uniq_1d"))
    report = {**TREE, "trained_on": {"period": "val", "rows": 64, "run_key": "k"}, "info": {"x": 1}}
    back = TreeMatcher.from_dict(report)  # typology_tree_val.json: info is dropped
    assert back.trained_on == report["trained_on"] and "info" not in back.to_dict()
    assert back.root == t.root and back != t
    zero = copy.deepcopy(TREE)
    zero["tree"]["right"]["dist"] = {"CYCLE": 10, "STACK": 0}  # zeros are dropped
    assert TreeMatcher.from_dict(zero) == t


def test_the_placeholder_is_one_trivial_leaf():
    doc = json.loads(PLACEHOLDER.read_text(encoding="utf-8"))
    assert set(doc) <= {"format", "tree", "trained_on", "info"}
    t = TreeMatcher.from_dict(doc)
    if doc["trained_on"] is None:  # the committed placeholder (a fitted tree replaces it)
        assert t.trivial and t.root == {"label": OTHER, "n": 0, "dist": {}}
        assert t.match({}) == (OTHER, ["no validation case reached this leaf (label OTHER)"])
        assert (t.depth, t.n_leaves, t.inputs) == (0, 1, ())


def _bad(path: list, value) -> dict:
    doc = copy.deepcopy(TREE)
    node = doc
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    return doc


@pytest.mark.parametrize(
    "doc",
    [
        [1],
        {**TREE, "format": 2},
        {**TREE, "format": True},
        {k: v for k, v in TREE.items() if k != "tree"},
        {**TREE, "extra": 1},
        {**TREE, "trained_on": "val"},
        _bad(["tree", "threshold"], math.nan),
        _bad(["tree", "threshold"], True),
        _bad(["tree", "threshold"], "0.5"),
        _bad(["tree", "feature"], ""),
        _bad(["tree", "right", "label"], "LAYERING"),
        _bad(["tree", "right", "n"], 11),  # dist sums to 10
        _bad(["tree", "right", "n"], -1),
        _bad(["tree", "right", "dist"], {"SMURF": 10}),
        _bad(["tree", "right", "dist"], {"CYCLE": 10.0}),
        _bad(["tree", "right", "extra"], 1),
        _bad(["tree", "left"], "leaf"),
    ],
)
def test_bad_tree_documents_are_refused(doc):
    with pytest.raises(ValueError):
        TreeMatcher.from_dict(doc)


def test_accuracy_of_the_tree():
    t = TreeMatcher.from_dict(TREE)
    rows = [
        ({"g_cycle_back": 1}, "CYCLE"),
        ({"g_cycle_back": 0, "u_out_uniq_1d": 9}, "FAN-OUT"),
        ({"g_cycle_back": 0, "u_out_uniq_1d": 9}, "STACK"),
        ({}, OTHER),
    ]
    acc = accuracy(rows, t)
    assert (acc["n"], acc["correct"], acc["accuracy"]) == (4, 3, 0.75)
    assert acc["model"] == "tree" and acc["params"] is None
    assert acc["confusion"]["STACK"]["FAN-OUT"] == 1
    assert acc["per_typology"]["CYCLE"]["recall"] == 1.0
    assert acc["majority"] == {"label": "FAN-OUT", "accuracy": 0.25}  # ties -> TYPOLOGIES order
    assert json.loads(json.dumps(acc, allow_nan=False)) == acc


def _separable(n: int = 30) -> list[tuple[dict, str]]:
    """Four typologies separated by three features, plus noise and a sometimes-undefined
    input: a depth-3 tree classifies them all."""
    rows = []
    for i in range(n):
        lo, hi = float(i % 3), 10.0 + i % 5
        common = {"noise": float(i % 7), "m": None if i % 2 else float(i)}
        rows.append(({"a": hi, "b": 0.0, "c": 0.0, **common}, "FAN-OUT"))
        rows.append(({"a": lo, "b": hi, "c": 0.0, **common}, "FAN-IN"))
        rows.append(({"a": lo, "b": lo, "c": 1.0, **common}, "CYCLE"))
        rows.append(({"a": lo, "b": lo, "c": 0.0, **common}, OTHER))
    return rows


def test_fit_tree_separates_a_separable_set_with_cross_validation():
    rows = _separable()
    tree, info = fit_tree(rows)
    assert tree.trained_on is None and not tree.trivial and tree.depth <= TREE_FIT["max_depth"]
    assert info["train_accuracy"] == 1.0 == accuracy(rows, tree)["accuracy"]
    cv = info["cv"]
    assert cv is not None and cv["folds"] == 5 and len(cv["fold_accuracy"]) == 5
    assert info["cv_accuracy"] == cv["accuracy"] >= 0.95 and cv["n"] == len(rows)
    assert sum(sum(r.values()) for r in cv["confusion"].values()) == len(rows)
    assert info["majority"] == {"label": "FAN-OUT", "accuracy": 0.25}
    assert info["params"] == TREE_FIT and info["rows"] == 120 and info["features"] == 5
    assert info["classes"]["CYCLE"] == 30 and sum(info["classes"].values()) == 120
    top = [d["feature"] for d in info["importances"]]
    assert set(top) == {"a", "b", "c"} and len(top) <= 10
    imps = [d["importance"] for d in info["importances"]]
    assert imps == sorted(imps, reverse=True) and all(v > 0 for v in imps)
    assert set(tree.inputs) <= {"a", "b", "c"} and info["inputs"] == list(tree.inputs)
    for leaf in (n for n, _ in tree._nodes() if "label" in n):
        assert leaf["n"] >= TREE_FIT["min_samples_leaf"]
        assert leaf["dist"] == {leaf["label"]: leaf["n"]}
    label, evidence = tree.match(rows[0][0])
    assert label == "FAN-OUT" and evidence[-1].startswith("100% of 30 validation cases")
    doc = {**tree.to_dict(), "info": info}
    assert json.loads(json.dumps(doc, allow_nan=False)) == doc
    assert TreeMatcher.from_dict(doc) == tree
    again, info2 = fit_tree(list(rows))  # deterministic for a seed
    assert again == tree and info2 == info


def test_fit_tree_splits_undefined_from_defined_and_says_so():
    rows = [({"pt_ratio_12h": 0.9 + 0.01 * (i % 20)}, "STACK") for i in range(20)]
    rows += [({"pt_ratio_12h": None}, OTHER) for _ in range(20)]
    tree, info = fit_tree(rows, min_samples_leaf=5)
    assert info["train_accuracy"] == 1.0 and tree.inputs == ("pt_ratio_12h",)
    assert tree.root["threshold"] < MISSING_SPLIT  # the MISSING stand-in sits below any value
    label, evidence = tree.match({"pt_ratio_12h": 0.95})
    assert label == "STACK"
    assert evidence[0] == (
        "the payment is 0.95 times the sender's inflow of the previous 12 hours "
        "(pt_ratio_12h = 0.95, defined)"
    )
    assert tree.match({})[0] == OTHER and tree.match({})[1][0] == "pt_ratio_12h is undefined"


def test_fit_tree_edge_cases():
    tree, info = fit_tree([])
    assert tree.trivial and tree.root == {"label": OTHER, "n": 0, "dist": {}}
    assert info["train_accuracy"] is None and info["cv"] is None and info["importances"] == []
    one = [({"a": float(i)}, "FAN-IN") for i in range(12)]
    tree, info = fit_tree(one)
    assert tree.trivial and tree.root["label"] == "FAN-IN" and info["train_accuracy"] == 1.0
    assert info["cv"]["folds"] == 5 and info["cv_accuracy"] == 1.0
    few = [({"a": 0.0}, "FAN-IN"), ({"a": 1.0}, "CYCLE")]  # no class has 2 rows: no CV
    assert fit_tree(few, min_samples_leaf=1)[1]["cv"] is None
    small = [({"a": float(i)}, "FAN-IN") for i in range(3)] + [({"a": 9.0}, "CYCLE")]
    assert fit_tree(small, min_samples_leaf=1)[1]["cv"]["folds"] == 3  # = the largest class
    with pytest.raises(ValueError, match="unknown ground-truth"):
        fit_tree([({"a": 1.0}, "LAYERING")])
    for bad in (
        {"max_depth": 0},
        {"min_samples_leaf": 0},
        {"seed": -1},
        {"max_depth": True},
        {"max_depth": 99},
    ):
        with pytest.raises(ValueError):
            fit_tree(one, **bad)
    assert tree_fit_params() == TREE_FIT and tree_fit_params({"seed": 3})["seed"] == 3
    for bad in ({"depth": 3}, [1], {"min_samples_leaf": 2.5}):
        with pytest.raises(ValueError):
            tree_fit_params(bad)
