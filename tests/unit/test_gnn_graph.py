"""The GNN host graph (M3 spec §4): load_graph on the real fixture feature table, encodings,
train-only normalisation, first ranks, bounds, label times vs the independent guard bounds,
labels, the whitelist, PNA degree histograms and the loader's HeteroData.

Torch-free tests run on the laptop too; the HeteroData tests skip without torch / PyG."""

from __future__ import annotations

import copy
import json

import numpy as np
import polars as pl
import pytest

from aml.data.split import SPLITS
from aml.features.build import load_spec
from aml.features.spec import scan_feature_table
from aml.models.gnn import (
    BOUND_KEYS,
    FAITHFUL_LABEL_SPLITS,
    LABEL_SPLITS,
    NODE,
    PROTOCOLS,
    REV,
    TO,
)
from aml.models.gnn import graph as G
from tests.conftest import load_yaml
from tests.fixtures import gnn_graphs as fx


@pytest.fixture(scope="module")
def inputs(prepared, data_cfg, rules_cfg, tmp_path_factory):
    cfgs = {"data": data_cfg, "rules": rules_cfg, "features": load_yaml("features.yaml")}
    return fx.gnn_inputs(prepared, tmp_path_factory.mktemp("gnn_graph"), cfgs)


@pytest.fixture(scope="module")
def graph(inputs, data_cfg):
    paths, fdir, gcfg = inputs
    return G.load_graph(paths, fdir, gcfg, data_cfg=data_cfg, label_splits=LABEL_SPLITS)


@pytest.fixture(scope="module")
def tx(inputs) -> pl.DataFrame:
    return pl.read_parquet(inputs[0].transactions)


@pytest.fixture(scope="module")
def parts(inputs) -> pl.DataFrame:
    return scan_feature_table(inputs[1]).collect()


def _torch():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    return torch


# --- load_graph on the fixture ------------------------------------------------------------------


def test_load_graph_fields(graph, tx, inputs):
    paths, fdir, _ = inputs
    g = graph
    m = tx.height
    assert g.n_edges == m and g.n_nodes == pl.read_parquet(paths.accounts).height
    expect = {
        "src": np.int64,
        "dst": np.int64,
        "minute": np.int64,
        "day": np.int16,
        "split_code": np.int8,
        "row_id": np.int64,
        "first_rank": np.int64,
        "rev_gid": np.int64,
        "ea": np.float32,
        "y": np.int8,
    }
    for name, dt in expect.items():
        a = getattr(g, name)
        assert a.dtype == dt, name
        assert a.flags["C_CONTIGUOUS"], name
        assert len(a) == (m if name != "rev_gid" else int((g.src != g.dst).sum())), name
    assert np.array_equal(g.src, tx["src"].to_numpy()) and np.array_equal(g.dst, tx["dst"])
    assert np.array_equal(g.row_id, tx["row_id"].to_numpy())
    assert np.array_equal(g.minute, tx["minute"].to_numpy())
    assert [SPLITS[c] for c in g.split_code[[0, -1]]] == [tx["split"][0], tx["split"][-1]]
    assert np.array_equal(g.rev_gid, np.flatnonzero(g.src != g.dst))
    assert np.array_equal(g.first_rank, fx.first_rank_of(g.minute))
    assert g.attr_columns == fx.GNN_EDGE_ATTRS == load_spec(fdir).gnn_edge_attr_names
    assert g.ea.shape == (m, 18)
    summary = json.loads((fdir / "summary.json").read_text(encoding="utf-8"))
    assert g.features_digest == summary["features_digest"]
    assert g.spec_hash == load_spec(fdir).spec_hash()
    assert g.label_splits == LABEL_SPLITS


def test_bounds_match_the_data(graph, tx, data_cfg):
    g = graph
    rank = tx["rank"].to_numpy()
    split = tx["split"].to_numpy()
    day = tx["day"].to_numpy()
    want = {
        "train_last": int(rank[split == "train"].max()),
        "val_last": int(rank[split == "val_late"].max()),
        "d10_last": int(rank[day <= data_cfg["test_views"]["primary"][1]].max()),
        "data_last": tx.height - 1,
        "val_early_first": int(rank[split == "val_early"].min()),
        "val_late_first": int(rank[split == "val_late"].min()),
        "test_first": int(rank[split == "test"].min()),
    }
    assert tuple(g.bounds) == BOUND_KEYS and g.bounds == want
    # the fixture has tail days: d10 is a real bound inside test
    assert g.bounds["test_first"] <= g.bounds["d10_last"] < g.bounds["data_last"]


def test_bounds_must_match_the_day_ranges(inputs, data_cfg):
    paths, fdir, gcfg = inputs
    bad = copy.deepcopy(data_cfg)
    bad["split"]["train"] = [1, 5]
    bad["split"]["val_early"] = [6, 7]
    with pytest.raises(ValueError, match="day range"):
        G.load_graph(paths, fdir, gcfg, data_cfg=bad, label_splits=LABEL_SPLITS)


def test_hi_small_bounds_are_asserted(data_cfg):
    hi = copy.deepcopy(data_cfg)
    hi["dataset"]["name"] = "hi_small"
    G._check_hi_small(dict(G.HI_SMALL_BOUNDS), G.HI_SMALL_ROWS, hi)
    wrong = dict(G.HI_SMALL_BOUNDS, d10_last=G.HI_SMALL_BOUNDS["d10_last"] - 1)
    with pytest.raises(ValueError, match="d10_last"):
        G._check_hi_small(wrong, G.HI_SMALL_ROWS, hi)
    G._check_hi_small(wrong, G.HI_SMALL_ROWS - 1, hi)  # not HI-Small: nothing to compare


def test_edge_attribute_encoding(graph, parts, inputs):
    _, fdir, gcfg = inputs
    g, pre = graph, graph.preprocess
    spec = load_spec(fdir)
    cat = gcfg["graph"]["categorical"]
    assert pre["columns"] == list(fx.GNN_EDGE_ATTRS) and pre["fitted_on"] == "train"
    assert pre["cat_idx"] == [fx.GNN_EDGE_ATTRS.index(c) for c in cat]
    assert pre["num_idx"] == [i for i in range(18) if i not in pre["cat_idx"]]
    assert pre["cat_sizes"] == [len(spec.vocab[c]) + 1 for c in cat]
    assert set(pre["mu"]) == set(pre["sigma"]) == set(gcfg["graph"]["zscore"])
    assert pre["log1p"] == [c for c in fx.GNN_EDGE_ATTRS if c in gcfg["graph"]["log1p"]]
    train = parts.filter(pl.col("split") == "train")
    for i, c in enumerate(fx.GNN_EDGE_ATTRS):
        raw = parts[c].to_numpy().astype(np.float64)
        if c in cat:  # the float of the train-vocab code; the model embeds code + 1 >= 0
            assert np.array_equal(g.ea[:, i], raw.astype(np.float32))
            emb = g.ea[:, i].astype(np.int64) + 1
            assert emb.min() >= 0 and emb.max() < pre["cat_sizes"][cat.index(c)]
            continue
        if c in fx.FLAGS:  # flags pass as is
            assert set(np.unique(raw)) <= {0.0, 1.0}
            assert np.array_equal(g.ea[:, i], raw.astype(np.float32))
            continue
        log = c in gcfg["graph"]["log1p"]
        tr = train[c].to_numpy().astype(np.float64)
        tr = np.log1p(tr) if log else tr
        mu, sd = tr.mean(), max(tr.std(), gcfg["graph"]["std_floor"])
        assert pre["mu"][c] == pytest.approx(mu, rel=1e-12, abs=1e-12)
        assert pre["sigma"][c] == pytest.approx(sd, rel=1e-12)
        want = ((np.log1p(raw) if log else raw) - mu) / sd
        np.testing.assert_allclose(g.ea[:, i], want.astype(np.float32), rtol=1e-6, atol=1e-6)
    # gaps: 0 (no history) stays 0 under log1p, a real gap >= 1 maps to >= log(2)
    gaps = parts["u_out_gap"].to_numpy()
    assert (gaps == 0).any() and (gaps >= 1).any()


def test_norm_stats_change_only_with_train_rows(parts, inputs):
    _, fdir, gcfg = inputs
    spec = load_spec(fdir)
    cols = ["split", *fx.GNN_EDGE_ATTRS]
    base = G.fit_preprocess(parts.select(cols), gcfg, spec)
    not_train = pl.col("split") != "train"
    shifted = parts.with_columns(
        pl.when(not_train).then(pl.col(c) * 7 + 3).otherwise(pl.col(c)).alias(c)
        for c in ("log_amount_usd", "u_out_gap", "hour_of_day")
    )
    assert G.fit_preprocess(shifted.select(cols), gcfg, spec) == base
    one_train = parts.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.col("log_amount_usd") + 50)
        .otherwise(pl.col("log_amount_usd"))
        .alias("log_amount_usd")
    )
    assert parts["split"][0] == "train"
    moved = G.fit_preprocess(one_train.select(cols), gcfg, spec)
    assert moved["mu"]["log_amount_usd"] != base["mu"]["log_amount_usd"]
    assert moved["mu"]["u_out_gap"] == base["mu"]["u_out_gap"]


def test_fit_preprocess_rejects_bad_configs(parts, inputs):
    _, fdir, gcfg = inputs
    spec = load_spec(fdir)
    table = parts.select("split", *fx.GNN_EDGE_ATTRS)
    for mutate, msg in (
        (lambda c: c["graph"]["zscore"].remove("hour_of_day"), "flags"),
        (lambda c: c["graph"]["categorical"].pop(), "categorical"),
        (lambda c: c["graph"]["log1p"].append("payment_format"), "categorical"),
        (lambda c: c["graph"]["zscore"].append("u_out_cnt_1d"), "not GNN edge attributes"),
    ):
        bad = copy.deepcopy(gcfg)
        mutate(bad)
        with pytest.raises(ValueError, match=msg):
            G.fit_preprocess(table, bad, spec)


def test_frozen_preprocess(graph, inputs, data_cfg):
    paths, fdir, gcfg = inputs
    same = G.load_graph(
        paths,
        fdir,
        gcfg,
        data_cfg=data_cfg,
        label_splits=LABEL_SPLITS,
        preprocess=copy.deepcopy(graph.preprocess),
    )
    assert np.array_equal(same.ea, graph.ea)
    frozen = copy.deepcopy(graph.preprocess)
    frozen["mu"]["log_amount_usd"] += 1.0
    other = G.load_graph(
        paths, fdir, gcfg, data_cfg=data_cfg, label_splits=LABEL_SPLITS, preprocess=frozen
    )
    i = fx.GNN_EDGE_ATTRS.index("log_amount_usd")
    delta = graph.ea[:, i].astype(np.float64) - other.ea[:, i]
    np.testing.assert_allclose(delta, 1.0 / frozen["sigma"]["log_amount_usd"], rtol=1e-5)
    keep = [j for j in range(18) if j != i]
    assert np.array_equal(other.ea[:, keep], graph.ea[:, keep])
    wrong = copy.deepcopy(graph.preprocess)
    wrong["vocab_sizes"]["payment_format"] += 1
    with pytest.raises(ValueError, match="vocab"):
        G.load_graph(
            paths, fdir, gcfg, data_cfg=data_cfg, label_splits=LABEL_SPLITS, preprocess=wrong
        )


def test_encode_rejects_bad_values(graph, parts):
    pre = graph.preprocess
    table = parts.select(*fx.GNN_EDGE_ATTRS).head(50)
    bad_code = table.with_columns(pl.lit(99.0, pl.Float32).alias("payment_format"))
    with pytest.raises(ValueError, match="payment_format"):
        G.encode_edge_attrs(bad_code, pre)
    nan = table.with_columns(pl.lit(float("nan"), pl.Float32).alias("hour_of_day"))
    with pytest.raises(ValueError, match="non-finite"):
        G.encode_edge_attrs(nan, pre)


def test_labels(graph, inputs, data_cfg):
    paths, fdir, gcfg = inputs
    g = graph
    lab = pl.read_parquet(paths.labels).select("row_id", "is_laundering")
    by_row = dict(zip(lab["row_id"].to_list(), lab["is_laundering"].to_list(), strict=True))
    loaded = np.isin(g.split_code, [SPLITS.index(s) for s in LABEL_SPLITS])
    assert (g.y[~loaded] == -1).all()
    want = np.array([by_row[r] for r in g.row_id[loaded].tolist()], dtype=np.int8)
    assert np.array_equal(g.y[loaded], want) and want.sum() > 0
    train = G.split_gids(g, "train")
    assert np.array_equal(G.labels_for(g, train), want[: len(train)])
    for s in ("val_late", "test"):
        with pytest.raises(ValueError, match="not loaded"):
            G.labels_for(g, G.split_gids(g, s)[:3])
    with pytest.raises(ValueError, match="test labels"):
        G.load_graph(
            paths, fdir, gcfg, data_cfg=data_cfg, label_splits=("train", "val_early", "test")
        )
    with pytest.raises(ValueError, match="reads the labels"):
        G.load_graph(paths, fdir, gcfg, data_cfg=data_cfg, label_splits=FAITHFUL_LABEL_SPLITS)
    fg = G.load_graph(
        paths,
        fdir,
        gcfg,
        data_cfg=data_cfg,
        label_splits=FAITHFUL_LABEL_SPLITS,
        protocol="faithful",
    )
    assert (fg.y[G.split_gids(fg, "val_late")] >= 0).all()
    assert (fg.y[G.split_gids(fg, "test")] == -1).all()


def test_split_gids(graph, tx):
    total = 0
    for s in SPLITS:
        gids = G.split_gids(graph, s)
        assert gids.dtype == np.int64 and np.all(np.diff(gids) == 1)  # contiguous, rank order
        assert len(gids) == int((tx["split"] == s).sum())
        total += len(gids)
    assert total == graph.n_edges
    with pytest.raises(ValueError):
        G.split_gids(graph, "val")


# --- label times and guard bounds ---------------------------------------------------------------


def _expected_bounds(g, gid: int, protocol: str, test_bound: str) -> int:
    """The §5.2 table, one gid at a time (a plain reference)."""
    first = int(np.flatnonzero(g.minute == g.minute[gid])[0])
    if protocol in ("causal", "pna"):
        return first - 1
    s = SPLITS[g.split_code[gid]]
    b = g.bounds
    if s == "train":
        return b["train_last"]
    if s in ("val_early", "val_late"):
        return b["val_last"]
    return b["data_last"] if test_bound == "end" else max(b["d10_last"], first - 1)


@pytest.mark.parametrize(
    ("protocol", "test_bound"),
    [("causal", "end"), ("pna", "end"), ("lookahead", "end"), ("lookahead", "d10")],
)
def test_label_times_and_guard_bounds_agree(graph, protocol, test_bound):
    g = graph
    gids = np.arange(g.n_edges, dtype=np.int64)
    lt = G.label_times(g, gids, protocol=protocol, test_bound=test_bound)
    gb = G.guard_bounds(g, gids, protocol=protocol, test_bound=test_bound)
    assert lt.dtype == gb.dtype == np.int64
    assert np.array_equal(lt, gb)
    sample = np.unique(np.r_[0, g.n_edges - 1, np.linspace(0, g.n_edges - 1, 400).astype(int)])
    for gid in sample.tolist():
        assert lt[gid] == _expected_bounds(g, gid, protocol, test_bound), gid
    if protocol != "lookahead":
        assert lt[0] == -1  # the first minute: an empty subgraph
        assert np.all(g.minute[np.maximum(lt, 0)][lt >= 0] < g.minute[lt >= 0])
    elif test_bound == "d10":
        test = g.split_code == SPLITS.index("test")
        tail = test & (np.arange(g.n_edges) > g.bounds["d10_last"])
        assert tail.any() and np.array_equal(lt[tail], g.first_rank[tail] - 1)
        assert (lt[test & ~tail] == g.bounds["d10_last"]).all()


def test_label_times_rejects(graph):
    gids = np.arange(5)
    with pytest.raises(ValueError, match="non-temporal"):
        G.label_times(graph, gids, protocol="faithful")
    with pytest.raises(ValueError, match="non-temporal"):
        G.guard_bounds(graph, gids, protocol="faithful")
    with pytest.raises(ValueError, match="protocol"):
        G.label_times(graph, gids, protocol="hpo")
    with pytest.raises(ValueError, match="test bound"):
        G.guard_bounds(graph, gids, protocol="lookahead", test_bound="d9")
    with pytest.raises(ValueError, match="outside"):
        G.label_times(graph, np.array([graph.n_edges]), protocol="causal")


def test_guard_bounds_never_call_label_times(graph, monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("guard_bounds must not call label_times")

    monkeypatch.setattr(G, "label_times", boom)
    gids = np.arange(graph.n_edges)
    for protocol, tb in (("causal", "end"), ("lookahead", "end"), ("lookahead", "d10")):
        assert len(G.guard_bounds(graph, gids, protocol=protocol, test_bound=tb)) == len(gids)


def test_bounds_on_a_tied_hand_graph():
    t = fx.tied_minute_graph()
    day_min = 1440
    minute = np.r_[[0] * 5, [6 * day_min] * 6, [8 * day_min] * 2, 10 * day_min, 10 * day_min + 1]
    g = fx.ArrayGraph(src=t.src, dst=t.dst, minute=minute.astype(np.int64), n_nodes=t.n_nodes)
    split = ["train"] * 5 + ["val_early"] * 6 + ["test"] * 4  # days 1 | 7 | 9, 9, 11, 11
    hg = fx.make_host_graph(g, split=split, primary_last_day=10)
    assert hg.bounds["d10_last"] == 12 and hg.bounds["val_last"] == 10  # no val_late rows
    gids = np.arange(g.n_edges)
    causal = G.label_times(hg, gids, protocol="causal")
    assert causal.tolist() == [-1] * 5 + [4] * 6 + [10, 10, 12, 13]
    assert np.array_equal(causal, G.guard_bounds(hg, gids, protocol="causal"))
    for tb, want in (
        ("end", [4] * 5 + [10] * 6 + [14] * 4),
        ("d10", [4] * 5 + [10] * 6 + [12, 12, 12, 13]),  # tail rows keep their causal bound
    ):
        lt = G.label_times(hg, gids, protocol="lookahead", test_bound=tb)
        assert lt.tolist() == want
        assert np.array_equal(lt, G.guard_bounds(hg, gids, protocol="lookahead", test_bound=tb))


# --- labels, whitelist and degrees on hand-made host graphs -------------------------------------


def test_train_degree_histograms_use_train_edges_only():
    g = fx.random_tied_graph(n_nodes=60, n_edges=900, per_minute=6, seed=3)
    split = ["train"] * 500 + ["val_early"] * 150 + ["val_late"] * 100 + ["test"] * 150
    hg = fx.make_host_graph(g, split=split)
    fwd, rev = G.train_degree_histograms(hg)
    tr = np.arange(500)
    deg_in = np.zeros(60, int)
    deg_rev = np.zeros(60, int)
    for e in tr:  # brute force
        deg_in[g.dst[e]] += 1
        if g.src[e] != g.dst[e]:
            deg_rev[g.src[e]] += 1
    assert fwd.dtype == rev.dtype == np.int64
    assert np.array_equal(fwd, np.bincount(deg_in)) and np.array_equal(rev, np.bincount(deg_rev))
    assert fwd.sum() == rev.sum() == 60
    # rewiring every non-train edge changes nothing
    g2 = fx.ArrayGraph(
        src=np.r_[g.src[:500], np.zeros(400, np.int64)],
        dst=np.r_[g.dst[:500], np.ones(400, np.int64)],
        minute=g.minute,
        n_nodes=60,
    )
    f2, r2 = G.train_degree_histograms(fx.make_host_graph(g2, split=split))
    assert np.array_equal(f2, fwd) and np.array_equal(r2, rev)


def test_assert_gnn_edge_inputs(inputs):
    spec = load_spec(inputs[1])
    names = list(fx.GNN_EDGE_ATTRS)
    for p in PROTOCOLS:
        G.assert_gnn_edge_inputs(names, p, spec)
    for bad in (
        names[::-1],
        names[:-1],
        [*names, "u_out_cnt_1d"],
        [*names, "timestamp"],
        [*names, "row_id"],
        [*names[:-1], "rev_pair_gap", "rev_pair_gap"],
    ):
        with pytest.raises(AssertionError):
            G.assert_gnn_edge_inputs(bad, "causal", spec)
    from aml.models.gnn.faithful import FAITHFUL_COLUMNS

    G.assert_gnn_edge_inputs(FAITHFUL_COLUMNS, "faithful", spec)  # the timestamp exemption
    for p in ("causal", "lookahead", "pna"):
        with pytest.raises(AssertionError):
            G.assert_gnn_edge_inputs(FAITHFUL_COLUMNS, p, spec)
    for bad in (["timestamp", "minute"], ["amount_received", "label"], ["v_in_cnt_12h"]):
        with pytest.raises(AssertionError):
            G.assert_gnn_edge_inputs(bad, "faithful", spec)


# --- the loader's HeteroData (torch) ------------------------------------------------------------


def test_build_hetero_whitelist_counts_and_order():
    torch = _torch()
    g = fx.tied_minute_graph()
    hg = fx.make_host_graph(g)
    d = G.build_hetero(hg)
    assert set(d.node_types) == {NODE} and set(d.edge_types) == {TO, REV}
    assert set(d[NODE].keys()) == {"num_nodes"} and d[NODE].num_nodes == g.n_nodes
    for et in (TO, REV):
        assert set(d[et].keys()) == {"edge_index", "time"}
        assert d[et].edge_index.dtype == d[et].time.dtype == torch.int64
        t = d[et].time
        assert bool((t[1:] > t[:-1]).all())  # unique, strictly increasing (rank order)
    m, loops = g.n_edges, int((g.src == g.dst).sum())
    assert d[TO].num_edges == m and d[REV].num_edges == m - loops
    # to: e_id == gid == time; rev: e_id -> rev_gid -> the forward edge, endpoints swapped
    assert d[TO].edge_index.tolist() == [g.src.tolist(), g.dst.tolist()]
    assert d[TO].time.tolist() == list(range(m))
    rg = hg.rev_gid
    assert d[REV].time.tolist() == rg.tolist()
    assert d[REV].edge_index.tolist() == [g.dst[rg].tolist(), g.src[rg].tolist()]


def test_build_hetero_snapshot_variants():
    _torch()
    g = fx.hand_graph_h1()
    hg = fx.make_host_graph(g)
    snap = G.build_hetero(hg, temporal=False, last_rank=5, reverse_self_loops=True)
    for et in (TO, REV):
        assert set(snap[et].keys()) == {"edge_index"} and snap[et].num_edges == 6
    assert snap[REV].edge_index.tolist() == [g.dst[:6].tolist(), g.src[:6].tolist()]
    pre = G.build_hetero(hg, last_rank=6)  # self-loop 6 has no reverse copy
    assert pre[TO].time.tolist() == list(range(7))
    assert pre[REV].time.tolist() == [0, 1, 2, 3, 4, 5]
    full = G.build_hetero(hg, reverse_self_loops=True)
    assert full[REV].num_edges == g.n_edges and full[REV].time.tolist() == list(range(8))
    with pytest.raises(ValueError):
        G.build_hetero(hg, last_rank=8)


def test_build_hetero_on_the_fixture_graph(graph):
    _torch()
    d = G.build_hetero(graph)
    assert d[TO].num_edges == graph.n_edges
    assert d[REV].num_edges == len(graph.rev_gid)
    assert np.array_equal(d[REV].time.numpy(), graph.rev_gid)
