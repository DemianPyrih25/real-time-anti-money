"""case_eval (M6 spec §5) end to end on the synthetic fixture, through the library functions the
Modal job calls (not Modal): the val tree fit and list tuning, then the test evaluation with the
fitted tree and the tuned thresholds copied in (frozen). Full-period parity is exact on both
periods, the reports are strict JSON and their counts agree with an independent recount from the
offline scores and the labels.
"""

from __future__ import annotations

import ast
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import yaml

from aml.explain import case_eval as ce
from aml.explain.casepack import CONFIG_FILE, check_config, load_config
from aml.explain.typology_match import OTHER, TYPOLOGIES, MatcherParams, TreeMatcher
from aml.features.build import snapshot_file
from aml.features.spec import SNAPSHOTS_DIR, SUMMARY_FILE
from aml.io import read_json, write_json_atomic
from aml.paths import DataPaths
from aml.serving import bundle
from tests.fixtures.serving_bundle import CONFIG_DIR, FIXTURE_EXPORT_KEY, REPO_ROOT
from tests.unit.test_import_lazy import FORBIDDEN, _import_time_imports

pytestmark = pytest.mark.slow

CHUNK = 500  # several parity chunks per period on the fixture
ACC_KEYS = {
    "n",
    "correct",
    "accuracy",
    "macro_recall",
    "majority",
    "per_typology",
    "confusion",
    "model",
    "params",
}
SUMMARY_ACC = ("n", "correct", "accuracy", "macro_recall")
VAL_SPLITS = ["val_early", "val_late"]
# The val fit on the fixture's few true-positive alerts: leaves of one row are allowed.
FIT = {"max_depth": 5, "min_samples_leaf": 1, "seed": 0}
# Frozen on test when the fixture's val fit is a single leaf (one truth class only).
HAND_TREE = {
    "format": 1,
    "tree": {
        "feature": "g_n_edges",
        "threshold": 0.5,
        "left": {"label": "OTHER", "n": 2, "dist": {"OTHER": 2}},
        "right": {"label": "FAN-IN", "n": 3, "dist": {"FAN-IN": 2, "OTHER": 1}},
    },
    "trained_on": None,
}


def strict_json(path: Path):
    """The file as JSON, refusing NaN and Infinity."""

    def refuse(const: str):
        raise ValueError(f"non-finite JSON constant {const}")

    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=refuse)


@pytest.fixture(scope="module")
def world(serving_bundle, data_cfg) -> dict:
    """The fixture bundle's world (its metadata.sources), read-only."""
    meta = read_json(serving_bundle / bundle.METADATA_FILE)
    src = {k: Path(v) for k, v in meta["sources"].items()}
    paths = DataPaths(src["features_dir"].parents[2])
    assert paths.features_dir(src["features_dir"].name) == src["features_dir"]
    return {
        "paths": paths,
        "features_dir": src["features_dir"],
        "rules_dir": src["rules_dir"],
        "graph_dir": src["graph_dir"],
        "bundle_dir": serving_bundle,
        "data_cfg": data_cfg,
        # independent of the committed tree (fitted on the real champion's inputs)
        "cfg": check_config({**load_config(CONFIG_DIR / CONFIG_FILE), "typology_tree": None}),
        "keys": {"data": "data-t", "export": FIXTURE_EXPORT_KEY},
    }


def run_period(world: dict, period: str, cfg: dict, out: Path, tag: str, **kw) -> dict:
    return ce.run_case_eval(
        world["paths"],
        period,
        world["data_cfg"],
        cfg,
        world["keys"],
        features_dir=world["features_dir"],
        rules_dir=world["rules_dir"],
        graph_dir=world["graph_dir"],
        out_dir=out,
        bundle_dir=world["bundle_dir"],
        alert_tag=tag,
        chunk_rows=CHUNK,
        **kw,
    )


@pytest.fixture(scope="module")
def runs(world, fixture_tag, tmp_path_factory) -> dict:
    """`make cases-val`, the lead copying the fitted tree and the tuned thresholds in, then
    `make cases`. The shipped config holds the single-leaf placeholder tree, so val's packs use
    the decision list; test's packs use the frozen tree."""
    out = tmp_path_factory.mktemp("case_eval") / "reports"
    logs: list[str] = []
    val_cfg = {**world["cfg"], "typology_tree_fit": FIT}
    val = run_period(world, "val", val_cfg, out, fixture_tag, log=logs.append)
    tuned = strict_json(out / ce.VAL_REPORT)["tuned"]
    fitted = TreeMatcher.from_dict(strict_json(out / ce.TREE_REPORT))
    tree = TreeMatcher.from_dict(HAND_TREE) if fitted.trivial else fitted
    frozen = {
        **world["cfg"],
        "typology": tuned,
        "typology_model": "tree",
        "typology_tree": tree.to_dict(),
    }
    test = run_period(world, "test", frozen, out, fixture_tag)
    return {
        "out": out,
        "val": val,
        "test": test,
        "tuned": tuned,
        "fitted": fitted,
        "tree": tree,
        "logs": logs,
    }


def offline_alerts(world: dict, tag: str, splits: list[str]) -> pl.DataFrame:
    """The rows the champion alerts on, recounted from the offline scores (score_s0 >= the tag's
    threshold), with day, rank and labels."""
    thr = read_json(world["bundle_dir"] / bundle.THRESHOLDS)["model"][tag]["threshold"]
    assert thr is not None
    scores = pl.read_parquet(world["graph_dir"] / "scores.parquet").select(
        "row_id", "split", bundle.SCORE
    )
    tx = pl.read_parquet(world["paths"].transactions, columns=["row_id", "rank", "day"])
    lab = pl.read_parquet(world["paths"].labels, columns=["row_id", "is_laundering", "typology"])
    return (
        scores.filter(pl.col("split").is_in(splits) & (pl.col(bundle.SCORE) >= thr))
        .join(tx, on="row_id")
        .join(lab, on="row_id")
        .sort("rank")
    )


def labels_of(world: dict, ids: list[int]) -> dict[int, tuple[int, str | None]]:
    lab = pl.read_parquet(world["paths"].labels, columns=["row_id", "is_laundering", "typology"])
    got = lab.filter(pl.col("row_id").is_in(ids))
    return {r: (y, t) for r, y, t in got.iter_rows()}


def truth_counts(frame: pl.DataFrame) -> dict[str, int]:
    tp = frame.filter(pl.col("is_laundering") == 1)
    c = Counter((t or OTHER) for t in tp["typology"].to_list())
    return {t: c.get(t, 0) for t in TYPOLOGIES}


# --- val: parity, tuning, error analysis --------------------------------------------------------


def test_val_parity_is_exact(runs, world, fixture_tag):
    v = runs["val"]
    rows = ce.period_rows(world["paths"].transactions, "val")
    assert v["parity_ok"] is True and v["champion_source"] == "bundle"
    assert v["parity"]["mismatches"] == dict.fromkeys(ce.PARITY_CHECKS, 0)
    assert v["parity"]["digest_ok"] is True
    assert v["events"] == v["rows"] == rows["rows"]
    doc = strict_json(runs["out"] / ce.VAL_ANALYSIS)
    p = doc["parity"]
    assert p["rows"] == rows["rows"] and p["chunks"] == math.ceil(rows["rows"] / CHUNK)
    assert p["alerts"]["stream"] == p["alerts"]["reference"] == v["alerts"]
    # The val stream ends in the state the replay snapshotted at the start of test.
    assert p["digest"]["against"] == snapshot_file("test") and p["digest"]["ok"] is True
    assert v["alerts"] == offline_alerts(world, fixture_tag, VAL_SPLITS).height
    assert v["alerts"] == doc["packs"] == doc["alerts"]["alerts"]
    n = rows["rows"]
    assert runs["logs"] and runs["logs"][-1].startswith(f"val: {n:,}/{n:,} events")


def test_val_tuning_report(runs, world, fixture_tag):
    doc = strict_json(runs["out"] / ce.VAL_REPORT)
    cfg = world["cfg"]
    tuned = doc["tuned"]
    assert tuned == runs["tuned"] == runs["val"]["tuned"]
    assert MatcherParams.from_dict(tuned).to_dict() == tuned
    grid = cfg["typology_grid"]
    assert all(tuned[k] in values for k, values in grid.items())
    assert all(tuned[k] == cfg["typology"][k] for k in set(tuned) - set(grid))
    acc = doc["accuracy"]
    assert set(acc) == ACC_KEYS and acc["params"] == tuned
    off = offline_alerts(world, fixture_tag, VAL_SPLITS)
    n_tp = off.filter(pl.col("is_laundering") == 1).height
    assert acc["n"] == doc["tuning"]["n_rows"] == runs["val"]["true_positive_alerts"] == n_tp
    support = {t: acc["per_typology"][t]["support"] for t in TYPOLOGIES}
    assert support == truth_counts(off)
    assert sum(sum(row.values()) for row in acc["confusion"].values()) == n_tp
    assert set(doc["at_config"]) == ACC_KEYS and doc["at_config"]["n"] == n_tp
    assert doc["at_config"]["params"] == MatcherParams.from_dict(cfg["typology"]).to_dict()
    assert doc["tuning"]["combinations"] == math.prod(len(v) for v in grid.values())
    assert doc["rate_tag"] == fixture_tag and doc["days"] == [7, 8]
    # The tree next to the list, on the same rows.
    cmp = doc["compare"]
    tree_doc = strict_json(runs["out"] / ce.TREE_REPORT)
    info = tree_doc["info"]
    assert cmp["rows"] == n_tp == info["rows"] and cmp["majority"] == acc["majority"]
    assert (
        cmp["rules_tuned"] == acc["accuracy"]
        and cmp["rules_at_config"] == doc["at_config"]["accuracy"]
    )
    assert cmp["tree_train"] == info["train_accuracy"] and cmp["tree_cv"] == info["cv_accuracy"]
    assert doc["tree"]["file"] == ce.TREE_REPORT and doc["tree"]["params"] == FIT
    v = runs["val"]
    assert v["tree_report"] == str(runs["out"] / ce.TREE_REPORT)
    assert v["accuracy"]["rules"] == {k: acc[k] for k in SUMMARY_ACC}
    assert v["accuracy"]["tree_train"] == info["train_accuracy"]
    assert v["accuracy"]["tree_cv"] == info["cv_accuracy"]


def test_val_tree_report(runs, world, fixture_tag):
    doc = strict_json(runs["out"] / ce.TREE_REPORT)
    assert set(doc) == {"format", "tree", "trained_on", "info"} and doc["format"] == 1
    fitted = runs["fitted"]
    assert fitted == TreeMatcher.from_dict(doc) and fitted.to_dict()["tree"] == doc["tree"]
    off = offline_alerts(world, fixture_tag, VAL_SPLITS)
    n_tp = off.filter(pl.col("is_laundering") == 1).height
    assert doc["trained_on"] == {"period": "val", "rows": n_tp, "run_key": None}
    info = doc["info"]
    assert info["params"] == FIT and info["rows"] == n_tp
    assert info["classes"] == truth_counts(off)
    assert (info["depth"], info["leaves"]) == (fitted.depth, fitted.n_leaves)
    assert info["inputs"] == list(fitted.inputs) and len(info["importances"]) <= 10
    assert info["train"]["n"] == n_tp and info["train"]["accuracy"] == info["train_accuracy"]
    if n_tp:
        assert 0.0 <= info["train_accuracy"] <= 1.0
        assert info["majority"]["label"] in TYPOLOGIES
    cv = info["cv"]
    if cv is not None:
        assert 2 <= cv["folds"] <= 5 and len(cv["fold_accuracy"]) == cv["folds"]
        assert cv["n"] == n_tp and cv["accuracy"] == info["cv_accuracy"]
    # The error analysis labels the false positives with the val-fitted tree (unless one leaf).
    tm = strict_json(runs["out"] / ce.VAL_ANALYSIS)["typology_match"]
    assert tm["model"] == ("rules" if fitted.trivial else "tree")
    assert tm["tree"]["file"] == ce.TREE_REPORT


def test_val_error_analysis(runs, world, fixture_tag):
    doc = strict_json(runs["out"] / ce.VAL_ANALYSIS)
    rec = doc["recall_by_typology"]
    assert list(rec) == [*TYPOLOGIES, "ALL"]
    for r in rec.values():
        assert 0 <= r["detected"] <= r["positives"]
        assert r["missed"] == r["positives"] - r["detected"]
        assert (r["recall"] is None) == (r["positives"] == 0)
    assert rec["ALL"]["positives"] == sum(rec[t]["positives"] for t in TYPOLOGIES)
    # Recount: the val positives and the offline alerts among them.
    tx = pl.read_parquet(world["paths"].transactions, columns=["row_id", "split"])
    lab = pl.read_parquet(world["paths"].labels, columns=["row_id", "is_laundering"])
    pos = tx.filter(pl.col("split").is_in(VAL_SPLITS)).join(lab, on="row_id")
    pos = pos.filter(pl.col("is_laundering") == 1)["row_id"].to_list()
    alerted = set(offline_alerts(world, fixture_tag, VAL_SPLITS)["row_id"].to_list())
    assert rec["ALL"]["positives"] == len(pos)
    assert rec["ALL"]["detected"] == len(alerted & set(pos))
    al = doc["alerts"]
    assert al["alerts"] == al["true_positives"] + al["false_positives"] == runs["val"]["alerts"]
    fps = doc["false_positives"]
    assert len(fps) == min(ce.N_FALSE_POSITIVES, al["false_positives"])
    scores = [f["score"] for f in fps]
    assert scores == sorted(scores, reverse=True)
    labels = labels_of(world, [f["id"] for f in fps])
    n_inputs = len(read_json(world["bundle_dir"] / bundle.FEATURE_SPEC)["model"]["feature_names"])
    for f in fps:
        assert labels[f["id"]][0] == 0
        assert f["narrative"].startswith(f"On day {f['day']} at {f['time']}, account ")
        assert f["typology"] in TYPOLOGIES and f["score"] >= doc["threshold"]
        assert len(f["drivers"]) == min(world["cfg"]["top_drivers"], n_inputs)
    assert sum(doc["false_positive_typology"].values()) == al["false_positives"]
    assert doc["typology_match"]["tuned"] == runs["tuned"]


# --- test: parity, frozen accuracy, examples ----------------------------------------------------


def test_test_parity_and_frozen_accuracy(runs, world, fixture_tag):
    res = runs["test"]
    assert res["parity_ok"] is True and res["parity"]["digest_ok"] is True
    assert res["parity"]["mismatches"] == dict.fromkeys(ce.PARITY_CHECKS, 0)
    rows = ce.period_rows(world["paths"].transactions, "test")
    assert res["events"] == res["rows"] == rows["rows"]
    doc = strict_json(runs["out"] / ce.TEST_REPORT)
    digest = doc["parity"]["digest"]
    assert digest["against"] == f"final_state_digest in {SUMMARY_FILE}"
    assert digest["expected"] == digest["final_state_digest"] and digest["ok"] is True
    views = world["data_cfg"]["test_views"]
    assert doc["views"] == {v: list(views[v]) for v in ce.VIEWS}
    tm = doc["typology_match"]
    rules = tm["rules"]
    assert rules["params"] == runs["tuned"] == rules["val_tuned"]
    assert rules["source"] == "val-tuned" == res["params_source"]
    tree = runs["tree"]
    assert tm["model"] == "tree" == res["typology_model"]
    from_val = tree == runs["fitted"]
    assert tm["tree"]["source"] == (
        "val-fitted" if from_val else "config (differs from the val-fitted tree)"
    )
    assert res["tree_source"] == tm["tree"]["source"]
    assert tm["tree"]["inputs"] == list(tree.inputs) and tm["tree"]["depth"] == tree.depth
    off = offline_alerts(world, fixture_tag, ["test"])
    assert res["alerts"] == off.height >= 1  # the fixture slice (inside test) alerts at this tag
    for view, (lo, hi) in doc["views"].items():
        sub = off.filter(pl.col("day").is_between(lo, hi))
        n_tp = sub.filter(pl.col("is_laundering") == 1).height
        both = tm[view]
        assert both["n"] == n_tp and both["majority"] == both["rules"]["majority"]
        for name, acc in (("rules", both["rules"]), ("tree", both["tree"])):
            assert set(acc) == ACC_KEYS and acc["model"] == name and acc["n"] == n_tp
            support = {t: acc["per_typology"][t]["support"] for t in TYPOLOGIES}
            assert support == truth_counts(sub)
            assert sum(sum(r.values()) for r in acc["confusion"].values()) == n_tp
            assert res["accuracy"][view][name] == {k: acc[k] for k in SUMMARY_ACC}
        assert both["rules"]["params"] == runs["tuned"] and both["tree"]["params"] is None
        assert both["tree"]["majority"] == both["majority"] == res["accuracy"][view]["majority"]
        assert doc["alerts"][view]["alerts"] == sub.height
    assert tm["full"]["n"] >= tm["primary"]["n"]


def test_test_packs_use_the_frozen_tree(runs):
    """The test packs are labelled by the frozen tree passed in, from their own stored
    features (the examples are full packs)."""
    tree, out = runs["tree"], runs["out"]
    for e in strict_json(out / ce.TEST_REPORT)["examples"]:
        typ = strict_json(out / e["file"])["why"]["typology"]
        assert typ["model"] == "tree" and e["label"] == typ["label"]
        assert tree.match(typ["features"]) == (typ["label"], typ["evidence"])
        assert set(tree.inputs) <= set(typ["features"])


def test_test_needs_a_fitted_tree(world, tmp_path):
    placeholder = {"format": 1, "tree": {"label": OTHER, "n": 0, "dist": {}}, "trained_on": None}
    cfg = check_config({**world["cfg"], "typology_model": "tree", "typology_tree": placeholder})
    with pytest.raises(ValueError, match="typology_tree.json"):
        ce.require_frozen_tree(cfg)
    with pytest.raises(ValueError, match="typology_tree.json"):  # refused before streaming
        run_period(world, "test", cfg, tmp_path, "headline")
    assert ce.require_frozen_tree({**cfg, "typology_model": "rules"}) is None
    hand = check_config({**cfg, "typology_tree": HAND_TREE})
    assert ce.require_frozen_tree(hand) == TreeMatcher.from_dict(HAND_TREE)
    assert ce.require_frozen_tree({**hand, "typology_model": "rules"}) is not None  # measured too


def test_tree_source(tmp_path):
    tree = TreeMatcher.from_dict(HAND_TREE)
    report = tmp_path / ce.TREE_REPORT
    assert ce.tree_source(tree, report) == {
        "source": "config (no val tree report)",
        "val_trained_on": None,
    }
    write_json_atomic({**HAND_TREE, "trained_on": {"period": "val"}, "info": {"x": 1}}, report)
    assert ce.tree_source(tree, report) == {
        "source": "val-fitted",
        "val_trained_on": {"period": "val"},
    }
    other = TreeMatcher.from_dict({**HAND_TREE, "tree": HAND_TREE["tree"]["left"]})
    assert ce.tree_source(other, report)["source"].startswith("config (differs")


def test_example_case_packs(runs, world, fixture_tag):
    out = runs["out"]
    doc = strict_json(out / ce.TEST_REPORT)
    ex = doc["examples"]
    files = sorted((out / ce.CASES_DIR).glob("case_*.json"))
    tp = offline_alerts(world, fixture_tag, ["test"]).filter(pl.col("is_laundering") == 1)
    kinds = {t or OTHER for t in tp["typology"].to_list()}
    assert len(ex) == len(files) == min(ce.N_EXAMPLES, len(kinds)) == runs["test"]["examples"]
    assert len({e["truth"] for e in ex}) == len(ex)
    labels = labels_of(world, [e["id"] for e in ex])
    for e in ex:
        pack = strict_json(out / e["file"])
        assert list(pack)[:5] == ["pack_format", "id", "case_key", "rank", "who"]
        assert pack["id"] == e["id"] and labels[e["id"]] == (1, e["truth"])
        assert pack["why"]["typology"]["label"] == e["label"]
        assert e["matched"] == (e["label"] == e["truth"])
        assert pack["why"]["score"] >= pack["why"]["threshold"]
        assert pack["who"]["subject"]["bank"] is not None  # the event's fields reached the pack
        alert = pack["how"]["subgraph"]["edges"][0]
        assert alert["kind"] == "alert" and alert["rank"] == pack["rank"]
        assert all(
            ed["minute"] <= alert["minute"] - 1 for ed in pack["how"]["subgraph"]["edges"][1:]
        )


# --- the pieces ---------------------------------------------------------------------------------


def test_period_rows(world):
    tx = world["paths"].transactions
    v, t = ce.period_rows(tx, "val"), ce.period_rows(tx, "test")
    assert v["days"] == [7, 8] and v["splits"] == VAL_SPLITS and t["days"][0] == 9
    assert t["first_rank"] == v["last_rank"] + 1
    n_all = pl.scan_parquet(tx).select(pl.len()).collect().item()
    assert t["last_rank"] == n_all - 1
    assert snapshot_file(ce.PERIODS["val"].snapshot_split) == "val_boundary.snap"
    assert snapshot_file(ce.PERIODS["test"].snapshot_split) == "test_boundary.snap"
    with pytest.raises(KeyError):
        ce.period_rows(tx, "train")


def test_champion_parts_reproduce_the_bundle(world):
    parts = ce.champion_parts(
        world["paths"],
        features_dir=world["features_dir"],
        rules_dir=world["rules_dir"],
        graph_dir=world["graph_dir"],
    )
    b = world["bundle_dir"]
    for key, rel in (
        ("thresholds", bundle.THRESHOLDS),
        ("calibration", bundle.CALIBRATION),
        ("feature_spec", bundle.FEATURE_SPEC),
    ):
        assert json.loads(json.dumps(parts[key])) == read_json(b / rel), key
    assert parts["booster"].read_bytes() == (b / bundle.BOOSTER).read_bytes()


def test_load_champion_prefers_a_matching_bundle(world, fixture_tag, tmp_path):
    dirs = {k: world[k] for k in ("features_dir", "rules_dir", "graph_dir")}
    b, paths = world["bundle_dir"], world["paths"]
    ch, cal, info = ce.load_champion(
        paths, FIXTURE_EXPORT_KEY, **dirs, bundle_dir=b, alert_tag=fixture_tag
    )
    assert info == {"source": "bundle", "bundle_dir": str(b)}
    assert cal == read_json(b / bundle.CALIBRATION) and ch.bundle_dir == b
    other, cal2, info2 = ce.load_champion(
        paths, "export-other", **dirs, bundle_dir=b, alert_tag=fixture_tag
    )
    assert info2["source"] == "computed" and "export key" in info2["reason"]
    assert other.threshold == ch.threshold and other.names == ch.names
    assert other.alert_tag == ch.alert_tag and other.bundle_dir is None
    assert other.model_version == f"export-other:{ch.booster_sha256[:12]}"
    assert json.loads(json.dumps(cal2)) == cal
    fd = world["features_dir"]
    assert ce.bundle_mismatch(b, FIXTURE_EXPORT_KEY, fd) is None
    assert ce.bundle_mismatch(b, FIXTURE_EXPORT_KEY, fd, data_version="v1") is None
    assert "prepared data" in ce.bundle_mismatch(b, FIXTURE_EXPORT_KEY, fd, data_version="v0")
    assert "export key" in ce.bundle_mismatch(b, None, fd)
    assert ce.bundle_mismatch(tmp_path, FIXTURE_EXPORT_KEY, fd).startswith("no serving bundle")


def test_parity_catches_a_changed_reference(world, fixture_tag):
    """One offline score one ulp off: exactly that row is reported, and the run is not ok."""
    dirs = {k: world[k] for k in ("features_dir", "rules_dir", "graph_dir")}
    ch, cal, _ = ce.load_champion(
        world["paths"],
        FIXTURE_EXPORT_KEY,
        **dirs,
        bundle_dir=world["bundle_dir"],
        alert_tag=fixture_tag,
    )
    rows = ce.period_rows(world["paths"].transactions, "test")
    refs = ce.load_references(world["features_dir"], world["rules_dir"], world["graph_dir"], "test")
    vals = refs.scores[bundle.SCORE].to_numpy().copy()
    vals[0] = np.nextafter(vals[0], 2.0)
    bumped = refs.scores.with_columns(pl.Series(bundle.SCORE, vals))
    st = ce.stream_period(
        ch,
        world["cfg"],
        snapshot=world["features_dir"] / SNAPSHOTS_DIR / snapshot_file("test"),
        transactions=world["paths"].transactions,
        rows=rows,
        calibration=cal,
        references=ce.References(world["features_dir"], bumped, refs.flags),
        chunk_rows=CHUNK,
    )
    bad = st.parity["mismatches"]
    assert bad["scores"] == 1 and st.parity["rows"] == st.events == rows["rows"]
    assert bad["features"] == bad["severities"] == bad["rule_flags"] == bad["rules_fired"] == 0
    assert len(st.alerts) == st.parity["alerts"]["stream"]
    assert ce.parity_ok({**st.parity, "digest": {"ok": True}}, rows["rows"]) is False
    good = {**st.parity, "mismatches": dict.fromkeys(ce.PARITY_CHECKS, 0)}
    good["alerts"] = {"stream": 3, "reference": 3}
    assert ce.parity_ok({**good, "digest": {"ok": True}}, rows["rows"]) is True
    assert ce.parity_ok({**good, "digest": {"ok": None}}, rows["rows"]) is True
    assert ce.parity_ok({**good, "digest": {"ok": False}}, rows["rows"]) is False
    assert ce.parity_ok({**good, "digest": {"ok": True}}, rows["rows"] + 1) is False
    assert ce.parity_ok(None, rows["rows"]) is False


def _rec(i: int, truth: str | None, label: str, y: int = 1, score: float = 0.5) -> dict:
    return {
        "id": i,
        "rank": i,
        "day": 9,
        "score": score,
        "label": label,
        "features": {"id": i},
        "y": y,
        "truth": truth,
    }


def test_pick_examples_prefers_matched_typologies():
    alerts = [
        _rec(1, "FAN-IN", "FAN-OUT"),
        _rec(2, "FAN-IN", "FAN-IN"),
        _rec(3, "CYCLE", "STACK"),
        _rec(4, OTHER, OTHER),
        _rec(5, "STACK", "STACK"),
        _rec(6, None, "FAN-IN", y=0),
    ]
    assert [a["id"] for a in ce.pick_examples(alerts)] == [2, 5, 4]
    assert [a["id"] for a in ce.pick_examples(alerts, n=4)] == [2, 5, 4, 3]
    assert [a["id"] for a in ce.pick_examples(alerts[:1])] == [1]  # unmatched fills up
    assert ce.pick_examples([alerts[-1]]) == []


def _fp_pack(i: int, score: float) -> str:
    return json.dumps(
        {
            "id": i,
            "when": {"day": 7, "time": "01:02"},
            "why": {
                "score": score,
                "calibrated": None,
                "typology": {"label": OTHER},
                "rules_fired": [],
                "drivers": [{"feature": "f", "value": 1.0, "display": "1", "contribution": 0.2}],
            },
            "narrative": f"On day 7 at 01:02, account {i} ...",
        }
    )


def test_false_positives_are_the_highest_scores():
    alerts = [
        {**_rec(i, None, OTHER, y=0, score=s), "pack": _fp_pack(i, s)}
        for i, s in ((1, 0.6), (2, 0.9), (3, 0.6), (4, 0.7))
    ]
    alerts.append({**_rec(5, "FAN-IN", OTHER, y=1, score=0.99), "pack": _fp_pack(5, 0.99)})
    fps = ce.false_positives(alerts, n=3)
    assert [f["id"] for f in fps] == [2, 4, 1]  # score, then the earlier rank
    assert fps[0]["drivers"] == [{"feature": "f", "display": "1", "contribution": 0.2}]
    assert ce.alert_counts(alerts) == {
        "alerts": 5,
        "true_positives": 1,
        "false_positives": 4,
        "precision": 0.2,
    }
    counts = ce.label_counts(alerts)
    assert counts[OTHER] == 5 and list(counts) == list(TYPOLOGIES)
    assert ce.matcher_rows(alerts) == [({"id": 5}, "FAN-IN")]


def test_recall_by_typology():
    rec = ce.recall_by_typology([(1, "FAN-IN"), (2, "FAN-IN"), (3, OTHER)], {2, 3, 99})
    assert rec["FAN-IN"] == {"positives": 2, "detected": 1, "missed": 1, "recall": 0.5}
    assert rec[OTHER] == {"positives": 1, "detected": 1, "missed": 0, "recall": 1.0}
    assert rec["CYCLE"] == {"positives": 0, "detected": 0, "missed": 0, "recall": None}
    assert rec["ALL"] == {"positives": 3, "detected": 2, "missed": 1, "recall": 2 / 3}
    with pytest.raises(ValueError, match="unknown typology"):
        ce.recall_by_typology([(1, "SMURFING")], set())


def test_params_source_and_the_snippet(world, tmp_path):
    params = dict(world["cfg"]["typology"])
    report = tmp_path / ce.VAL_REPORT
    assert ce.params_source(params, report)["source"] == "config (no val tuning report)"
    write_json_atomic({"tuned": params}, report)
    assert ce.params_source(params, report) == {"source": "val-tuned", "val_tuned": params}
    changed = {**params, "fan_in_min": 99}
    assert ce.params_source(changed, report)["source"].startswith("config (differs")
    snippet = ce.typology_snippet({**params, "random_out_max": None})
    assert snippet.startswith("typology:\n  cycle_min: ")
    assert yaml.safe_load(snippet)["typology"] == {**params, "random_out_max": None}


def test_case_eval_imports_no_web_or_kafka_package():
    path = REPO_ROOT / "src" / "aml" / "explain" / "case_eval.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad = [(n, m) for n, m in _import_time_imports(tree.body) if m.split(".")[0] in FORBIDDEN]
    assert not bad, f"aml.explain.case_eval imports {bad} at module level"
