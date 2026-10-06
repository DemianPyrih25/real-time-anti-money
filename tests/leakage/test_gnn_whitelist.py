"""The GNN sees only whitelisted inputs (PLAN.md §4; M3 spec §4.2, §4.3, §5.6, §9, §14.2).

- Edge attributes = exactly EngineSpec.gnn_edge_attr_names (set and order), each column the §4.3
  encoding of that feature-table column and nothing else; no id, time, split or label column; no
  windowed node aggregate; no column that is an affine copy of rank, minute, row_id or the label.
- The HeteroData handed to a loader holds only edge_index and time per edge type and num_nodes.
- The model's node input is the 1-column ego flag; edge rows are EA[gid] (a rev_to copy shares its
  forward row); the target's own attributes enter only through tgt_attr.
- The faithful exemptions (`timestamp`, per-snapshot z-score) are exactly those and refuse every
  other protocol; gnn_faithful never enters the model comparison.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from tests.conftest import load_yaml
from tests.fixtures.gnn_graphs import GNN_EDGE_ATTRS, gnn_inputs, hand_graph_h1, make_host_graph
from tests.fixtures.gnn_ref import (
    LOADER_PARTS,
    REV_TYPE,
    TO_TYPE,
    engine_batches,
    gnn_missing,
    load_gnn_yaml,
    ref_hetero,
    require_gnn,
)

NODE_AGGREGATE_SUFFIXES = ("_1d", "_3d", "_12h", "_2d")
ID_TIME_LABEL = (
    "row_id",
    "rank",
    "minute",
    "day",
    "split",
    "ts",
    "timestamp",
    "src",
    "dst",
    "from_account",
    "to_account",
    "from_bank",
    "to_bank",
    "is_laundering",
    "attempt_id",
    "typology",
    "attempt_size",
    "typology_detail",
    "label",
    "y",
)
NON_FAITHFUL = ("causal", "lookahead", "pna")
# §4.5 does not pin the exception type of assert_gnn_edge_inputs (M2's whitelist raises
# ValueError; an `assert_*` helper may raise AssertionError): either counts as a refusal.
REFUSED = (ValueError, AssertionError)


def _bad_names(names) -> list[str]:
    out = []
    for n in names:
        low = n.lower()
        words = any(w in low for w in ("label", "laundering", "typology", "attempt", "hub"))
        if low in ID_TIME_LABEL or low.endswith(NODE_AGGREGATE_SUFFIXES) or words:
            out.append(n)
    return out


# --- names and constants (no torch) ------------------------------------------------------------


@pytest.fixture(scope="module")
def engine_spec(prepared, rules_cfg):
    from tests.fixtures.engine_frames import fixture_spec

    return fixture_spec(prepared, rules_cfg)[0]


def test_gnn_edge_attributes_are_the_whitelisted_edge_level_names(engine_spec) -> None:
    from aml.features.tx_features import is_forbidden

    names = list(engine_spec.gnn_edge_attr_names)
    assert tuple(names) == GNN_EDGE_ATTRS
    assert _bad_names(names) == []
    assert not [n for n in names if is_forbidden(n)]
    assert engine_spec.assert_model_inputs(names) == names


def test_faithful_exemptions_are_exactly_two() -> None:
    from aml.models.gnn import FAITHFUL_EXEMPT_FEATURES, FAITHFUL_EXEMPT_NORM

    assert FAITHFUL_EXEMPT_FEATURES == ("timestamp",)
    assert FAITHFUL_EXEMPT_NORM == "per_snapshot"
    faithful = load_gnn_yaml()["protocols"]["faithful"]
    assert faithful["norm"] == FAITHFUL_EXEMPT_NORM
    assert set(FAITHFUL_EXEMPT_FEATURES) <= set(faithful["edge_features"])
    # the exempt feature is outside the causal whitelist
    assert not set(FAITHFUL_EXEMPT_FEATURES) & set(GNN_EDGE_ATTRS)
    # besides the exemption, Multi-GNN's edge features are raw transaction fields and ports
    rest = [f for f in faithful["edge_features"] if f not in FAITHFUL_EXEMPT_FEATURES]
    assert _bad_names(rest) == []


def test_gnn_faithful_is_never_a_comparison_model() -> None:
    from aml.models import gnn

    assert gnn.FAITHFUL_MODEL == "gnn_faithful"
    assert gnn.FAITHFUL_MODEL not in gnn.COMPARISON_MODELS
    assert set(gnn.PRIMARY_ONLY_MODELS) <= set(gnn.COMPARISON_MODELS)
    report = load_gnn_yaml()["report"]
    assert report["winner_pair"][0] in gnn.COMPARISON_MODELS
    assert gnn.FAITHFUL_MODEL not in report["winner_pair"]


def test_assert_gnn_edge_inputs_is_exact(engine_spec) -> None:
    reason = gnn_missing("graph.assert_gnn_edge_inputs")
    if reason:
        pytest.skip(reason)
    from aml.models.gnn.graph import assert_gnn_edge_inputs

    names = list(engine_spec.gnn_edge_attr_names)
    for p in NON_FAITHFUL:
        assert_gnn_edge_inputs(names, p, engine_spec)
        for extra in ("timestamp", "rank", "minute", "row_id", "is_laundering", "u_out_cnt_1d"):
            with pytest.raises(REFUSED):
                assert_gnn_edge_inputs([*names, extra], p, engine_spec)
        with pytest.raises(REFUSED):  # set and order
            assert_gnn_edge_inputs(names[::-1], p, engine_spec)
        with pytest.raises(REFUSED):
            assert_gnn_edge_inputs(names[:-1], p, engine_spec)
        with pytest.raises(REFUSED):
            assert_gnn_edge_inputs([*names, names[0]], p, engine_spec)


def test_faithful_builders_refuse_other_protocols() -> None:
    """The faithful data (timestamp feature, per-snapshot norm) cannot be built under another
    protocol: a deliberate error before any input is touched (the inputs here are None)."""
    pytest.importorskip("torch")
    reason = gnn_missing(
        "faithful.require_faithful", "faithful.faithful_edge_attrs", "faithful.snapshot_ea"
    )
    if reason:
        pytest.skip(reason)
    from aml.models.gnn import faithful

    deliberate = (ValueError, RuntimeError, AssertionError, PermissionError)
    faithful.require_faithful("faithful")
    for p in NON_FAITHFUL:
        with pytest.raises(deliberate):
            faithful.require_faithful(p)
        with pytest.raises(deliberate):
            faithful.faithful_edge_attrs(None, None, None, protocol=p)
        with pytest.raises(deliberate):
            faithful.snapshot_ea(None, 0, protocol=p)
    assert faithful.FAITHFUL_COLUMNS[0] == "timestamp"
    assert _bad_names([c for c in faithful.FAITHFUL_COLUMNS if c != "timestamp"]) == []


# --- the graph handed to the loader -----------------------------------------------------------


def test_hetero_holds_only_topology_and_time() -> None:
    require_gnn("graph.build_hetero")
    import torch

    from aml.models.gnn.graph import build_hetero
    from tests.fixtures.gnn_graphs import random_tied_graph

    g = random_tied_graph(n_nodes=200, n_edges=2000, seed=7)
    data = build_hetero(make_host_graph(g))
    assert data.node_types == ["acct"]
    assert set(data.edge_types) == {TO_TYPE, REV_TYPE}
    assert set(data["acct"].keys()) == {"num_nodes"} and data["acct"].num_nodes == g.n_nodes
    ref = ref_hetero(g.src, g.dst, g.n_nodes)
    for et in (TO_TYPE, REV_TYPE):
        assert set(data[et].keys()) <= {"edge_index", "time"}, (et, list(data[et].keys()))
        assert torch.equal(data[et].edge_index, ref[et].edge_index), et
        assert torch.equal(data[et].time, ref[et].time), et


# --- the fixture's real graph -----------------------------------------------------------------


@pytest.fixture(scope="module")
def real(prepared, data_cfg, rules_cfg, tmp_path_factory):
    reason = gnn_missing("graph.load_graph")
    if reason:
        pytest.skip(reason)
    from aml.features.build import load_spec
    from aml.models.gnn import LABEL_SPLITS
    from aml.models.gnn.graph import load_graph

    paths, features_dir, gnn_cfg = gnn_inputs(
        prepared,
        tmp_path_factory.mktemp("gnn_wl"),
        {"data": data_cfg, "rules": rules_cfg, "features": load_yaml("features.yaml")},
    )
    g = load_graph(paths, features_dir, gnn_cfg, data_cfg=data_cfg, label_splits=LABEL_SPLITS)
    return g, load_spec(features_dir), features_dir, gnn_cfg


def test_ea_columns_are_exactly_the_engine_edge_attributes(real) -> None:
    g, spec, _, _ = real
    names = tuple(spec.gnn_edge_attr_names)
    assert tuple(g.attr_columns) == names == GNN_EDGE_ATTRS
    assert g.ea.shape == (g.n_edges, len(names)) and g.ea.dtype == np.float32
    assert list(g.preprocess["columns"]) == list(names)
    assert _bad_names(g.attr_columns) == []
    cat = [names[i] for i in g.preprocess["cat_idx"]]
    assert cat == ["payment_currency", "receiving_currency", "payment_format"]
    assert sorted(g.preprocess["num_idx"] + g.preprocess["cat_idx"]) == list(range(len(names)))


def test_ea_is_the_encoding_of_those_columns_only(real) -> None:
    """Every EA column equals the §4.3 encoding of its own feature-table column (categorical
    code; log1p on gaps; z-score with train-row statistics; flags as is), recomputed here."""
    from aml.features.spec import scan_feature_table

    g, spec, features_dir, gnn_cfg = real
    gc = gnn_cfg["graph"]
    names = list(spec.gnn_edge_attr_names)
    table = scan_feature_table(features_dir, ["rank", "split", *names]).collect().sort("rank")
    assert table["rank"].to_list() == list(range(g.n_edges))
    train = (table["split"] == "train").to_numpy()
    bad = []
    for j, name in enumerate(names):
        v = table[name].to_numpy().astype(np.float64)
        if name in gc["log1p"]:
            v = np.log1p(v)
        if name in gc["zscore"]:
            mu = v[train].mean()
            sds = (v[train].std(ddof=0), v[train].std(ddof=1))
            wants = [(v - mu) / max(sd, gc["std_floor"]) for sd in sds]
        else:
            wants = [v]
        got = g.ea[:, j].astype(np.float64)
        if not any(np.allclose(got, w, rtol=1e-4, atol=1e-4) for w in wants):
            bad.append(name)
    assert not bad, bad


def test_ea_encodes_no_time_id_or_label(real) -> None:
    g, *_ = real
    loaded = g.y >= 0
    probes = {
        "rank": np.arange(g.n_edges, dtype=np.float64),
        "minute": g.minute.astype(np.float64),
        "row_id": g.row_id.astype(np.float64),
    }
    bad = []
    for j, name in enumerate(g.attr_columns):
        col = g.ea[:, j].astype(np.float64)
        if col.std() == 0:
            continue
        for what, p in probes.items():
            if abs(np.corrcoef(col, p)[0, 1]) > 0.99:
                bad.append((name, what))
        y = g.y[loaded].astype(np.float64)
        if y.std() > 0 and abs(np.corrcoef(col[loaded], y)[0, 1]) > 0.99:
            bad.append((name, "label"))
    assert not bad, bad


def test_labels_are_loaded_only_for_the_allowed_splits(real) -> None:
    from aml.data.split import SPLITS

    g, *_ = real
    assert tuple(g.label_splits) == ("train", "val_early")
    for code, split in enumerate(SPLITS):
        y = g.y[g.split_code == code]
        if split in g.label_splits:
            assert (y >= 0).all(), split
        else:
            assert (y == -1).all(), split


# --- the model's inputs ----------------------------------------------------------------------


def test_model_inputs_are_the_ego_flag_and_ea_rows() -> None:
    require_gnn(*LOADER_PARTS, "transforms.to_model_inputs")
    import torch

    from aml.models.gnn.transforms import to_model_inputs

    g = hand_graph_h1()
    hg = make_host_graph(g)
    ea = torch.from_numpy(hg.ea)
    for protocol in ("causal", "lookahead"):
        (fb,) = engine_batches(
            hg, np.arange(g.n_edges), protocol=protocol, fanout=[5, 5], batch_size=64
        )
        x, ei_fwd, ea_fwd, ei_rev, ea_rev, tgt_src, tgt_dst, tgt_attr = to_model_inputs(
            fb, ea, "cpu"
        )
        assert x.shape == (fb.n_nodes, 1)
        assert torch.equal(x[:, 0], fb.ego.float())
        assert torch.equal(ei_fwd, fb.ei_fwd) and torch.equal(ei_rev, fb.ei_rev)
        assert torch.equal(ea_fwd, ea[fb.gid_fwd]) and torch.equal(ea_rev, ea[fb.gid_rev])
        assert torch.equal(tgt_attr, ea[fb.tgt_gid])
        assert torch.equal(tgt_src, fb.tgt_src) and torch.equal(tgt_dst, fb.tgt_dst)
        # the target's own row reaches the model only through tgt_attr
        own = fb.tgt_gid[fb.node_sub[fb.ei_fwd[1]]] == fb.gid_fwd
        own_rev = fb.tgt_gid[fb.node_sub[fb.ei_rev[1]]] == fb.gid_rev
        assert not own.any() and not own_rev.any(), protocol
        # a FlatBatch carries index tensors only: no float data crosses the worker boundary
        floats = [
            n
            for n, v in zip(fb._fields, fb, strict=True)
            if isinstance(v, torch.Tensor) and v.is_floating_point()
        ]
        assert floats == [], floats


def test_faithful_snapshot_holds_only_its_prefix() -> None:
    """§9: the snapshot for last rank L holds exactly the transactions of rank <= L (`to` in rank
    order, `rev_to` every one of them flipped incl. self-loops), topology only."""
    require_gnn("faithful.snapshot_hetero")
    import torch

    from aml.models.gnn.faithful import snapshot_hetero
    from tests.fixtures.gnn_graphs import random_tied_graph

    g = random_tied_graph(n_nodes=200, n_edges=2000, seed=8)
    hg = make_host_graph(g)
    last = 1200
    data = snapshot_hetero(hg, last)
    assert set(data["acct"].keys()) == {"num_nodes"} and data["acct"].num_nodes == g.n_nodes
    src, dst = torch.as_tensor(g.src[: last + 1]), torch.as_tensor(g.dst[: last + 1])
    want = {TO_TYPE: torch.stack([src, dst]), REV_TYPE: torch.stack([dst, src])}
    assert set(data.edge_types) == set(want)
    for et, ei in want.items():
        assert set(data[et].keys()) <= {"edge_index", "time"}, (et, list(data[et].keys()))
        assert torch.equal(data[et].edge_index, ei), et


def test_evaluate_refuses_gnn_faithful_as_a_comparison_model() -> None:
    from modal_jobs import evaluate

    if not hasattr(evaluate, "gnn_stage_dirs"):
        pytest.skip("awaiting evaluate --with-gnn (F)")
    from aml.paths import DataPaths

    spec = {"kind": "gnn_causal", "key": "k", "make": "gnn"}
    models = {"gnn_causal": spec, "gnn_faithful": dict(spec, kind="gnn_faithful")}
    with pytest.raises(ValueError):
        evaluate.gnn_stage_dirs(DataPaths(Path("/nonexistent")), {"models": models})
