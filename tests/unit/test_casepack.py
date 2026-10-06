"""Case packs (M6 spec §1): the on_alert hook on the fixture bundle's inproc stream, and the
causal subgraph on a hand-made engine.

The stream test runs the real runtime (`open_runtime`) with the case hook: one pack per alert,
every history edge at least one minute before its alert, TreeSHAP drivers that sum to the raw
log-odds, a known typology label from the configured matcher (the decision list under the
placeholder tree; a hand-made tree in a second hook) with the matcher's inputs (model inputs,
the list's inputs, the subgraph's shape), a deterministic narrative, and parity unchanged.
"""

from __future__ import annotations

import ast
import copy
import json
import logging
import math
import threading

import numpy as np
import polars as pl
import pytest

from aml.explain.casepack import (
    CONFIG_KEYS,
    G_STATS,
    PACK_FORMAT,
    TREE_FILE,
    build_case_pack,
    calibrate,
    case_hook,
    check_config,
    drivers,
    feature_names,
    frozen_tree,
    graph_stats,
    load_config,
    subgraph,
    typology_matcher,
)
from aml.explain.narrative import narrative
from aml.explain.typology_match import (
    FEATURE_TEXT,
    TYPOLOGIES,
    MatcherParams,
    RulesMatcher,
    TreeMatcher,
    input_names,
    inputs,
    match,
)
from aml.features.engine import Engine
from aml.features.spec import INPUT_COLUMNS, MINUTES_PER_DAY
from aml.features.tx_features import cents
from aml.serving import bundle
from aml.serving.alerts import read_alerts
from aml.serving.scorer import Champion, open_runtime, parity_ok
from tests.fixtures.engine_frames import engine_tx, frame_fields, make_spec
from tests.fixtures.serving_bundle import CONFIG_DIR, REPO_ROOT, fixture_settings
from tests.unit.test_import_lazy import FORBIDDEN, _import_time_imports

EXPLAIN_CONFIG = CONFIG_DIR / "explain.yaml"
SAME_KEYS = ("pack_format", "id", "case_key", "rank", "when", "why", "how", "model_version")
# A hand-made frozen tree over the subgraph's shape (every pack carries these inputs).
HAND_TREE = {
    "format": 1,
    "tree": {
        "feature": "g_n_edges",
        "threshold": 0.5,
        "left": {"label": "OTHER", "n": 12, "dist": {"OTHER": 12}},
        "right": {
            "feature": "g_src_out_deg",
            "threshold": 2.5,
            "left": {"label": "STACK", "n": 10, "dist": {"STACK": 7, "OTHER": 3}},
            "right": {"label": "FAN-OUT", "n": 10, "dist": {"FAN-OUT": 10}},
        },
    },
    "trained_on": None,
}


def tree_cfg(cfg: dict, tree: dict = HAND_TREE, model: str = "tree") -> dict:
    return check_config({**cfg, "typology_model": model, "typology_tree": tree})


@pytest.fixture(scope="module")
def run(serving_bundle, fixture_tag, tmp_path_factory) -> dict:
    """The fixture slice through the inproc runtime with the case hook (strict) plus a second
    hook that builds each pack again from the event's input fields (banks known), and once more
    with the hand-made tree as the frozen matcher."""
    # independent of the committed tree (fitted on the real champion's inputs)
    cfg = check_config({**load_config(EXPLAIN_CONFIG), "typology_tree": None})
    ch = Champion.from_bundle(serving_bundle, alert_tag=fixture_tag)
    sl = pl.read_parquet(serving_bundle / bundle.SLICE)
    fields = {r[1]: r for r in zip(*(sl[c].to_list() for c in INPUT_COLUMNS), strict=True)}
    packs: list[dict] = []
    with_fields: list[dict] = []
    with_tree: list[dict] = []
    seen: list = []
    hook = case_hook(ch, cfg, packs.append, strict=True)
    by_tree = tree_cfg(cfg)

    def again(s, eng) -> None:
        f = fields[s.row_id]
        pack = build_case_pack(s, eng, ch, cfg=cfg, calibration=hook.calibration, fields=f)
        with_fields.append(pack)
        with_tree.append(build_case_pack(s, eng, ch, cfg=by_tree, fields=f))
        seen.append(s)

    settings = fixture_settings(
        serving_bundle, tmp_path_factory.mktemp("cases") / "rt", fixture_tag
    )
    rt = open_runtime(settings, on_alert=[hook, again])
    result = rt.run(threading.Event())
    code = rt.close(result)
    return {
        "cfg": cfg,
        "ch": ch,
        "rt": rt,
        "hook": hook,
        "packs": packs,
        "with_fields": with_fields,
        "with_tree": with_tree,
        "seen": seen,
        "slice": {r["rank"]: r for r in sl.to_dicts()},
        "db_ids": [a["row_id"] for a in read_alerts(rt.store.alerts_db, limit=None)],
        "result": result,
        "code": code,
    }


def test_one_case_pack_per_alert(run):
    packs, seen, hook = run["packs"], run["seen"], run["hook"]
    assert len(packs) >= 1
    assert [p["id"] for p in packs] == [s.row_id for s in seen] == run["db_ids"][::-1]
    assert len(packs) == run["rt"].progress.alerts == hook.built and hook.failed == 0
    assert all(s.alert for s in seen)
    assert len({p["id"] for p in packs}) == len(packs)


def test_case_packs_change_no_output(run):
    rt = run["rt"]
    assert run["result"] == "end" and run["code"] == 0 and parity_ok(rt.parity)
    assert rt.parity["mismatches"] == dict.fromkeys(bundle.CHECKS, 0)
    assert rt.parity["alerts_ok"] is True and rt.parity["digest_ok"] is True


def test_history_is_strictly_before_the_alert_minute(run):
    window = run["packs"][0]["how"]["window_minutes"]
    n_history = 0
    for p, s in zip(run["packs"], run["seen"], strict=True):
        edges = p["how"]["subgraph"]["edges"]
        alert = {"src": s.src, "dst": s.dst, "minute": s.minute, "amount_usd": s.amount_usd,
                 "kind": "alert", "rank": s.rank}  # fmt: skip
        assert edges[0] == alert and [e["kind"] for e in edges].count("alert") == 1
        ranks = [e["rank"] for e in edges]
        assert len(set(ranks)) == len(ranks) and ranks[1:] == sorted(ranks[1:])
        for e in edges[1:]:
            assert e["kind"] == "history"
            assert s.minute - window <= e["minute"] <= s.minute - 1  # the as-of rule
            assert e["rank"] < s.rank and e["amount_usd"] >= 0
            n_history += 1
            row = run["slice"].get(e["rank"])
            if row is not None:  # a slice event: the edge is that payment, as stored
                assert (e["src"], e["dst"], e["minute"]) == (row["src"], row["dst"], row["minute"])
                assert round(e["amount_usd"] * 100) == cents(row["amount_usd"])
    assert n_history > 0  # not vacuous


def test_subgraph_nodes_and_roles(run):
    for p, s in zip(run["packs"], run["seen"], strict=True):
        sg = p["how"]["subgraph"]
        ids = [n["id"] for n in sg["nodes"]]
        assert len(set(ids)) == len(ids)
        assert sg["nodes"][0] == {"id": s.src, "role": "subject"}
        others = sg["nodes"][1:]
        if s.dst != s.src:
            assert sg["nodes"][1] == {"id": s.dst, "role": "counterparty"}
            others = sg["nodes"][2:]
        assert all(n["role"] == "other" for n in others)
        assert [n["id"] for n in others] == sorted(n["id"] for n in others)
        ends = {a for e in sg["edges"] for a in (e["src"], e["dst"])}
        assert ends == set(ids)


def test_drivers_sum_to_the_raw_log_odds(run):
    ch, top = run["ch"], run["cfg"]["top_drivers"]
    for p, s in zip(run["packs"], run["seen"], strict=True):
        why = p["why"]
        raw = float(ch.booster.predict(s.x, raw_score=True, num_threads=1)[0])
        parts = [d["contribution"] for d in why["drivers"]]
        total = math.fsum([*parts, why["others_contribution"], why["base_value"]])
        assert abs(total - raw) <= 1e-6 and abs(why["log_odds"] - raw) <= 1e-6
        assert len(parts) == min(top, len(ch.names))
        mags = [abs(c) for c in parts]
        assert mags == sorted(mags, reverse=True)
        names = [d["feature"] for d in why["drivers"]]
        assert len(set(names)) == len(names) and set(names) <= set(ch.names)
        for d in why["drivers"]:
            v = float(s.x[0, ch.names.index(d["feature"])])
            assert d["value"] == (v if math.isfinite(v) else None)
            assert isinstance(d["display"], str) and d["display"]
    s = run["seen"][0]
    full = drivers(ch, s.x, len(ch.names))  # every input listed: nothing left over
    raw = float(ch.booster.predict(s.x, raw_score=True, num_threads=1)[0])
    assert full["others_contribution"] == 0.0
    total = math.fsum([d["contribution"] for d in full["drivers"]] + [full["base_value"]])
    assert abs(total - raw) <= 1e-6


def test_typology_features_are_the_matchers_inputs(run):
    ch = run["ch"]
    spec, names = ch.spec, list(ch.names)
    assert set(input_names(spec).values()) <= set(feature_names(names, spec))
    for p, s in zip(run["packs"], run["seen"], strict=True):
        f = p["why"]["typology"]["features"]
        assert tuple(f) == feature_names(names, spec)
        assert all(
            v is None or (isinstance(v, int | float) and math.isfinite(v)) for v in f.values()
        )
        x = np.asarray(s.x, dtype=np.float32).reshape(-1)
        own = inputs(s.row, spec)  # the decision list's inputs keep the engine's float64 values
        for i, name in enumerate(names):
            v = float(x[i])
            expected = own[name] if name in own else (v if math.isfinite(v) else None)
            assert f[name] == expected, name
        assert {k: f[k] for k in own} == own
        sg = p["how"]["subgraph"]
        assert {k: f[k] for k in G_STATS} == graph_stats(sg, s.src, s.dst)
        assert all(isinstance(f[k], int) and f[k] >= 0 for k in G_STATS)
        assert f["g_n_edges"] == sum(1 for e in sg["edges"] if e["kind"] == "history")


def test_typology_is_a_known_label_reproduced_by_the_matcher(run):
    params = MatcherParams.from_dict(run["cfg"]["typology"])
    ch = run["ch"]
    used = typology_matcher(run["cfg"], ch)
    for p in run["packs"]:
        typ = p["why"]["typology"]
        assert list(typ) == ["label", "model", "evidence", "features"]
        assert typ["label"] in TYPOLOGIES and typ["model"] == used.model
        assert typ["evidence"] and all(isinstance(e, str) and e for e in typ["evidence"])
        assert used.match(typ["features"]) == (typ["label"], typ["evidence"])
        if typ["model"] == "rules":  # the shipped placeholder tree: the decision list
            rule_f = {n: typ["features"][n] for n in input_names(ch.spec).values()}
            assert match(rule_f, params) == (typ["label"], typ["evidence"])


def test_a_frozen_tree_labels_the_packs(run):
    tree = TreeMatcher.from_dict(HAND_TREE)
    for p, pt in zip(run["with_fields"], run["with_tree"], strict=True):
        typ = pt["why"]["typology"]
        assert typ["model"] == "tree"
        assert tree.match(typ["features"]) == (typ["label"], typ["evidence"])
        assert typ["evidence"][-1].endswith(f"validation cases at this leaf were {typ['label']}")
        assert typ["features"] == p["why"]["typology"]["features"]
        assert "Matched typology (decision tree): " in pt["narrative"]
        assert {k: pt[k] for k in ("id", "who", "what", "when", "where", "how")} == {
            k: p[k] for k in ("id", "who", "what", "when", "where", "how")
        }


def test_narrative_is_deterministic_and_the_pack_is_json(run):
    for p in run["packs"]:
        back = json.loads(json.dumps(p, allow_nan=False))
        assert back == p
        assert narrative(back) == narrative(p) == p["narrative"]
        when, sub = p["when"], p["who"]["subject"]
        assert p["narrative"].startswith(f"On day {when['day']} at {when['time']}, account ")
        assert f"account {sub['account']}" in p["narrative"]


def test_the_5w_h_fields(run):
    ch, cfg = run["ch"], run["cfg"]
    cal = run["hook"].calibration
    assert cal is not None and cal["x"]  # the bundle's calibration.json
    for p, s in zip(run["packs"], run["seen"], strict=True):
        day = s.minute // MINUTES_PER_DAY + 1
        hh, mm = divmod(s.minute % MINUTES_PER_DAY, 60)
        row = run["slice"][s.rank]
        assert p["pack_format"] == PACK_FORMAT and p["rank"] == s.rank
        assert p["id"] == s.row_id and p["case_key"] == f"{s.src}-d{day}"
        assert p["who"]["subject"]["account"] == s.src
        assert p["who"]["counterparty"]["account"] == s.dst
        assert p["when"] == {"day": day, "time": f"{hh:02d}:{mm:02d}", "minute": s.minute}
        assert p["what"]["amount_usd"] == row["amount_usd"] == s.amount_usd
        assert p["what"]["payment_format"] == row["payment_format"]
        why = p["why"]
        assert why["score"] == s.score and why["threshold"] == ch.threshold
        assert why["score"] >= why["threshold"] and why["rate_tag"] == ch.alert_tag
        assert why["rules_fired"] == list(s.rules)
        assert why["calibrated"] == float(np.interp(s.score, cal["x"], cal["y"]))
        assert 0.0 <= why["calibrated"] <= 1.0
        assert p["how"]["window_minutes"] == min(cfg["window_minutes"], ch.spec.W_max)
        assert (p["how"]["cap_1hop"], p["how"]["cap_2hop"]) == (cfg["cap_1hop"], cfg["cap_2hop"])
        assert p["model_version"] == ch.model_version and p["export_key"] == "fixture"


def test_fields_give_the_banks_and_the_engine_fallback_agrees(run):
    for p, pf, s in zip(run["packs"], run["with_fields"], run["seen"], strict=True):
        row = run["slice"][s.rank]
        fb, tb = str(row["from_bank"]), str(row["to_bank"])
        assert pf["who"]["subject"]["bank"] == fb and pf["who"]["counterparty"]["bank"] == tb
        assert pf["where"] == {"from_bank": fb, "to_bank": tb, "cross_bank": fb != tb}
        assert pf["what"]["amount_paid"] == row["amount_paid"]
        assert pf["what"]["payment_currency"] == row["payment_currency"]
        assert pf["what"]["receiving_currency"] == row["receiving_currency"]
        assert f"(bank {fb})" in pf["narrative"]
        # Without the fields (unless the scorer carries them): the engine's copy of the event.
        assert p["where"]["cross_bank"] == pf["where"]["cross_bank"]
        assert p["who"]["subject"]["bank"] in (None, fb)
        assert abs(p["what"]["amount_paid"] - row["amount_paid"]) <= 0.005 + 1e-9
        for k in ("payment_currency", "receiving_currency"):
            assert p["what"][k] in (None, row[k])
        assert {k: p[k] for k in SAME_KEYS} == {k: pf[k] for k in SAME_KEYS}
        assert p["what"]["amount_usd"] == pf["what"]["amount_usd"]


def test_hook_failures_are_counted_or_raised(run):
    """After the run the engine clock has moved past every alert minute: building a pack then
    would see that minute's own payments, so it is refused."""
    ch, cfg, s = run["ch"], run["cfg"], run["seen"][0]
    eng = run["rt"].scorer.eng
    assert eng.clock > s.minute
    sink: list[dict] = []
    lenient = case_hook(ch, cfg, sink.append)
    lenient(s, eng)  # logged and counted, never raised: the stream goes on
    assert (lenient.built, lenient.failed, sink) == (0, 1, [])
    assert "clock" in lenient.last_error
    strict = case_hook(ch, cfg, sink.append, strict=True)
    with pytest.raises(ValueError, match="clock"):
        strict(s, eng)
    assert strict.failed == 1
    calibration = {"x": [0.0, 1.0], "y": [0.0, 1.0]}
    assert case_hook(ch, cfg, sink.append, calibration=calibration).calibration == calibration
    assert calibrate(0.25, calibration) == 0.25 and calibrate(0.25, None) is None
    assert calibrate(0.25, {"x": [], "y": []}) is None


# --- the subgraph on a hand-made engine -----------------------------------------------------------

# rank: (minute, src, dst); the alert is the last event: 1 -> 2 at minute 5000. W_max = 4320, so
# the window starts at minute 680.
EVENTS = [
    dict(minute=600, src=10, dst=1),  # 0: older than the window
    dict(minute=680, src=11, dst=1),  # 1: minute = m - W: inside
    dict(minute=3500, src=7, dst=3),  # 2: hop 2, the payer of a payer
    dict(minute=3600, src=3, dst=9),  # 3: a payee of a payer: not outward
    dict(minute=4000, src=3, dst=1),  # 4: hop 1, into the subject
    dict(minute=4100, src=1, dst=4),  # 5: hop 1, out of the subject
    dict(minute=4150, src=6, dst=4),  # 6: a payer of a payee: not outward
    dict(minute=4200, src=4, dst=5),  # 7: hop 2, the payee of a payee
    dict(minute=4300, src=2, dst=8),  # 8: hop 1, out of the counterparty
    dict(minute=4400, src=1, dst=2),  # 9: an earlier subject -> counterparty payment
    dict(minute=5000, src=12, dst=1),  # 10: same minute as the alert: pending, invisible
    dict(minute=5000, src=1, dst=2, amount_usd=9800.0),  # 11: the alert
]


@pytest.fixture(scope="module")
def small_engine(rules_cfg) -> Engine:
    frame = engine_tx(EVENTS)
    assert frame["rank"].to_list() == list(range(len(EVENTS)))
    eng = Engine.create(make_spec(frame, rules_cfg))
    for f in frame_fields(frame):
        eng.process(eng.prepare(*f))
    assert eng.clock == 5000 and eng.spec.W_max == 4320
    return eng


def sg_of(eng: Engine, src: int = 1, dst: int = 2, **kw) -> dict:
    args = {"window": 4320, "cap_1hop": 10, "cap_2hop": 4} | kw
    return subgraph(eng, src, dst, 5000, 11, 9800.0, **args)


def history(sg: dict) -> list[tuple[int, int, int, int]]:
    return [(e["rank"], e["src"], e["dst"], e["minute"]) for e in sg["edges"][1:]]


def test_subgraph_is_causal_capped_and_continues_outward(small_engine):
    digest = small_engine.state_digest()
    sg = sg_of(small_engine)
    assert sg["edges"][0] == {"src": 1, "dst": 2, "minute": 5000, "amount_usd": 9800.0,
                              "kind": "alert", "rank": 11}  # fmt: skip
    assert history(sg) == [
        (1, 11, 1, 680),
        (2, 7, 3, 3500),
        (4, 3, 1, 4000),
        (5, 1, 4, 4100),
        (7, 4, 5, 4200),
        (8, 2, 8, 4300),
        (9, 1, 2, 4400),
    ]
    assert all(e["kind"] == "history" and e["amount_usd"] == 123.45 for e in sg["edges"][1:])
    assert sg["nodes"] == [
        {"id": 1, "role": "subject"},
        {"id": 2, "role": "counterparty"},
        *({"id": a, "role": "other"} for a in (3, 4, 5, 7, 8, 11)),
    ]
    capped = sg_of(small_engine, cap_1hop=1, cap_2hop=0)  # the newest payment per direction
    assert [r for r, *_ in history(capped)] == [4, 8, 9]
    short = sg_of(small_engine, window=1000)  # minutes >= 4000
    assert [r for r, *_ in history(short)] == [4, 5, 7, 8, 9]
    assert sg_of(small_engine, window=10**6) == sg  # clamped to W_max
    assert small_engine.state_digest() == digest  # read-only


def test_a_self_loop_alert_has_one_centre(small_engine):
    sg = sg_of(small_engine, src=1, dst=1)
    assert sg["nodes"][0] == {"id": 1, "role": "subject"}
    assert all(n["role"] == "other" for n in sg["nodes"][1:])  # account 2 is a neighbour now
    assert {"id": 2, "role": "other"} in sg["nodes"]
    assert [r for r, *_ in history(sg)] == [1, 2, 4, 5, 7, 8, 9]


def test_the_subgraph_needs_the_engine_at_the_alert_minute(small_engine):
    with pytest.raises(ValueError, match="clock"):
        subgraph(small_engine, 1, 2, 4999, 11, 1.0, window=4320, cap_1hop=1, cap_2hop=1)


def _edges(pairs: list[tuple[int, int]], alert: tuple[int, int] = (1, 2)) -> dict:
    hist = [{"src": a, "dst": b, "minute": 10 + i, "amount_usd": 1.0, "kind": "history",
             "rank": i} for i, (a, b) in enumerate(pairs)]  # fmt: skip
    first = {"src": alert[0], "dst": alert[1], "minute": 99, "amount_usd": 1.0, "kind": "alert",
             "rank": 99}  # fmt: skip
    return {"nodes": [], "edges": [first, *hist]}


def test_graph_stats_on_a_hand_checkable_subgraph():
    """Alert 1 -> 2. The sender paid 3 and 4 (3 twice) and itself; 3 and 4 both pay 6 (a
    common sink), 7 is paid by 4 and by 5, which the sender never paid (not a common sink); 9
    and 8 paid the sender; 12, 13 and 14 paid the receiver, which paid 8, and 8 paid the
    sender: a path back from the receiver."""
    pairs = [(1, 3), (1, 4), (1, 3), (3, 6), (4, 6), (4, 7), (5, 7), (9, 1), (2, 8),
             (12, 2), (13, 2), (14, 2), (8, 1), (1, 1)]  # fmt: skip
    sg = _edges(pairs)
    assert graph_stats(sg, 1, 2) == {
        "g_src_out_deg": 2,  # 3, 4 (the self-payment is no counterparty)
        "g_src_in_deg": 2,  # 9, 8
        "g_dst_in_deg": 3,  # 12, 13, 14
        "g_dst_out_deg": 1,  # 8
        "g_common_sinks": 1,  # 6
        "g_cycle_back": 1,  # 2 -> 8 -> 1
        "g_n_nodes": 12,  # 1 2 3 4 5 6 7 8 9 12 13 14
        "g_n_edges": 14,
    }
    assert list(graph_stats(sg, 1, 2)) == list(G_STATS)
    assert graph_stats(_edges([(2, 1)]), 1, 2)["g_cycle_back"] == 1  # directly back
    assert graph_stats(_edges([(2, 8), (9, 1)]), 1, 2)["g_cycle_back"] == 0
    assert graph_stats(_edges([(2, 8), (8, 9), (9, 1)]), 1, 2)["g_cycle_back"] == 0  # 3 hops
    # Only history edges count: the alert edge alone leaves an empty shape.
    assert set(graph_stats(_edges([]), 1, 2).values()) == {0}
    assert set(G_STATS) <= set(FEATURE_TEXT)  # every statistic reads as a sentence


def test_graph_stats_of_the_engine_subgraph(small_engine):
    # History of sg_of: 11->1, 7->3, 3->1, 1->4, 4->5, 2->8, 1->2 (see the test above).
    assert graph_stats(sg_of(small_engine), 1, 2) == {
        "g_src_out_deg": 2,  # 4, 2
        "g_src_in_deg": 2,  # 11, 3
        "g_dst_in_deg": 1,  # 1
        "g_dst_out_deg": 1,  # 8
        "g_common_sinks": 0,  # 4 pays 5, 2 pays 8
        "g_cycle_back": 0,
        "g_n_nodes": 8,
        "g_n_edges": 7,
    }


def test_the_matcher_choice(run, caplog):
    ch, cfg = run["ch"], run["cfg"]
    names = tuple(input_names(ch.spec).values())
    rules = RulesMatcher(MatcherParams.from_dict(cfg["typology"]), names)
    placeholder = {"format": 1, "tree": {"label": "OTHER", "n": 0, "dist": {}}, "trained_on": None}
    assert typology_matcher(tree_cfg(cfg), ch) == TreeMatcher.from_dict(HAND_TREE)
    assert frozen_tree(tree_cfg(cfg)) == TreeMatcher.from_dict(HAND_TREE)
    assert typology_matcher(tree_cfg(cfg, model="rules"), ch) == rules  # the tree is ignored
    assert typology_matcher(tree_cfg(cfg, placeholder), ch) == rules  # one leaf: no tree
    assert frozen_tree(tree_cfg(cfg, placeholder)) is None
    assert typology_matcher(tree_cfg(cfg, None), ch) == rules  # no tree file
    alien = copy.deepcopy(HAND_TREE)
    alien["tree"]["feature"] = "not_a_feature"
    with pytest.raises(ValueError, match="not_a_feature"):
        typology_matcher(tree_cfg(cfg, alien), ch)
    with pytest.raises(ValueError, match="not_a_feature"):
        case_hook(ch, tree_cfg(cfg, alien), [].append, strict=True)
    with caplog.at_level(logging.WARNING, logger="aml.explain.casepack"):
        lenient = case_hook(ch, tree_cfg(cfg, alien), [].append)
    assert lenient.matcher == rules and "not_a_feature" in caplog.text
    assert case_hook(ch, tree_cfg(cfg), [].append).matcher.model == "tree"


# --- config and imports ---------------------------------------------------------------------------


def test_the_explain_config():
    cfg = load_config(EXPLAIN_CONFIG)
    assert set(cfg) == set(CONFIG_KEYS)
    assert (cfg["window_minutes"], cfg["cap_1hop"], cfg["cap_2hop"], cfg["top_drivers"]) == (
        4320,
        10,
        4,
        5,
    )
    MatcherParams.from_dict(cfg["typology"])
    assert cfg["typology_model"] == "tree"
    assert cfg["typology_tree_fit"] == {"max_depth": 5, "min_samples_leaf": 10, "seed": 0}
    assert TreeMatcher.from_dict(cfg["typology_tree"]).to_dict() == cfg["typology_tree"]
    assert check_config({k: v for k, v in cfg.items() if k != "typology_grid"})
    assert check_config(cfg) == cfg  # idempotent
    old = {k: v for k, v in cfg.items() if not k.startswith("typology_") or k == "typology_grid"}
    filled = check_config(old)  # an explain.yaml without the tree keys: the decision list
    assert filled["typology_model"] == "rules" and filled["typology_tree"] is None
    assert filled["typology_tree_fit"] == cfg["typology_tree_fit"]
    for bad in (
        {**cfg, "extra": 1},
        {**cfg, "cap_1hop": -1},
        {**cfg, "window_minutes": 0},
        {**cfg, "top_drivers": True},
        {**cfg, "cap_2hop": 2.5},
        {**cfg, "typology": {**cfg["typology"], "gs_min": -1}},
        {k: v for k, v in cfg.items() if k != "typology"},
        {**cfg, "typology_grid": {"nope": [1]}},
        {**cfg, "typology_model": "gnn"},
        {**cfg, "typology_tree": {"format": 2, "tree": HAND_TREE["tree"]}},
        {**cfg, "typology_tree_fit": {"max_depth": 0}},
        {**cfg, "typology_tree_fit": {"leaves": 4}},
    ):
        with pytest.raises(ValueError):
            check_config(bad)


def test_load_config_reads_the_tree_next_to_it(tmp_path):
    text = EXPLAIN_CONFIG.read_text(encoding="utf-8")
    (tmp_path / "explain.yaml").write_text(text, encoding="utf-8")
    assert load_config(tmp_path / "explain.yaml")["typology_tree"] is None  # no tree file
    (tmp_path / TREE_FILE).write_text(json.dumps(HAND_TREE), encoding="utf-8")
    cfg = load_config(tmp_path / "explain.yaml")
    assert cfg["typology_tree"] == HAND_TREE and frozen_tree(cfg) is not None
    (tmp_path / TREE_FILE).write_text(json.dumps({**HAND_TREE, "format": 7}), encoding="utf-8")
    with pytest.raises(ValueError, match=TREE_FILE):
        load_config(tmp_path / "explain.yaml")
    (tmp_path / "explain.yaml").write_text(text + "\ntypology_tree: {}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="typology_tree"):
        load_config(tmp_path / "explain.yaml")


def test_explain_imports_no_web_or_kafka_package():
    for name in ("__init__", "casepack", "typology_match", "narrative"):
        path = REPO_ROOT / "src" / "aml" / "explain" / f"{name}.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        bad = [(n, m) for n, m in _import_time_imports(tree.body) if m.split(".")[0] in FORBIDDEN]
        assert not bad, f"aml.explain.{name} imports {bad} at module level"
