"""Multi-GNN's snapshot data (M3 spec §9): ports, snapshots, per-snapshot z-scores, EA_f on the
real fixture graph, the confinement to protocol "faithful" and the shuffled loader's seeding.

Torch-free tests run on the laptop too; the snapshot / loader tests skip without torch / PyG."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from aml.features.build import load_spec
from aml.models.gnn import FAITHFUL_EXEMPT_NORM, FAITHFUL_LABEL_SPLITS, NODE, PROTOCOLS, REV, TO
from aml.models.gnn import faithful as F
from aml.models.gnn import graph as G
from tests.conftest import load_yaml
from tests.fixtures import gnn_graphs as fx


def brute_ports(src, dst) -> tuple[list[int], list[int]]:
    """First-appearance ports, edges in rank order: a pair's port is fixed when it first occurs
    (in-port at v = number of distinct senders v had before; out-port at u likewise)."""
    senders: dict[int, list[int]] = {}
    receivers: dict[int, list[int]] = {}
    port: dict[tuple[int, int], tuple[int, int]] = {}
    for u, v in zip(src.tolist(), dst.tolist(), strict=True):
        if (u, v) not in port:
            port[(u, v)] = (len(senders.setdefault(v, [])), len(receivers.setdefault(u, [])))
            senders[v].append(u)
            receivers[u].append(v)
    pairs = [port[(u, v)] for u, v in zip(src.tolist(), dst.tolist(), strict=True)]
    return [p[0] for p in pairs], [p[1] for p in pairs]


@pytest.fixture(scope="module")
def inputs(prepared, data_cfg, rules_cfg, tmp_path_factory):
    cfgs = {"data": data_cfg, "rules": rules_cfg, "features": load_yaml("features.yaml")}
    return fx.gnn_inputs(prepared, tmp_path_factory.mktemp("gnn_faithful"), cfgs)


@pytest.fixture(scope="module")
def fgraph(inputs, data_cfg):
    paths, fdir, gcfg = inputs
    return G.load_graph(
        paths,
        fdir,
        gcfg,
        data_cfg=data_cfg,
        label_splits=FAITHFUL_LABEL_SPLITS,
        protocol="faithful",
    )


# --- ports ---------------------------------------------------------------------------------------


def test_ports_equal_first_appearance_reference():
    g = fx.random_tied_graph(n_nodes=300, n_edges=5000, per_minute=10, seed=11)
    in_p, out_p = F.multignn_ports(g.src, g.dst, g.rank)
    want_in, want_out = brute_ports(g.src, g.dst)
    assert in_p.dtype == out_p.dtype == np.int64
    assert in_p.tolist() == want_in and out_p.tolist() == want_out
    assert in_p.min() == 0 and out_p.min() == 0  # 0-based
    # input order does not matter: ranks decide (rank tie-break among same-minute pairs)
    perm = np.random.default_rng(0).permutation(g.n_edges)
    pi, po = F.multignn_ports(g.src[perm], g.dst[perm], g.rank[perm])
    assert np.array_equal(pi, in_p[perm]) and np.array_equal(po, out_p[perm])
    # a parallel edge shares its pair's port; a reverse edge is another pair
    h = fx.hand_graph_h1()
    hi, ho = F.multignn_ports(h.src, h.dst, h.rank)
    assert (hi.tolist(), ho.tolist()) == brute_ports(h.src, h.dst)
    # edges 0 v->x, 1 x->u, 2 u->x, 3 y->u, 4 u->v, 5 x->v, 6 z->z, 7 v->u
    assert hi.tolist() == [0, 0, 1, 1, 0, 1, 0, 2]
    assert ho.tolist() == [0, 0, 0, 0, 1, 1, 0, 1]


def test_ports_are_prefix_stable():
    g = fx.random_tied_graph(n_nodes=200, n_edges=3000, per_minute=8, seed=4)
    in_all, out_all = F.multignn_ports(g.src, g.dst, g.rank)
    for k in (1, 17, 1000, 2999):
        pi, po = F.multignn_ports(g.src[:k], g.dst[:k], g.rank[:k])
        assert np.array_equal(pi, in_all[:k]) and np.array_equal(po, out_all[:k])


def test_ports_reject_bad_input():
    with pytest.raises(ValueError, match="unique"):
        F.multignn_ports(np.array([0, 1]), np.array([1, 0]), np.array([3, 3]))
    with pytest.raises(ValueError):
        F.multignn_ports(np.array([0]), np.array([1, 2]), np.array([0]))
    e = F.multignn_ports(np.array([], int), np.array([], int), np.array([], int))
    assert [len(x) for x in e] == [0, 0]


# --- EA_f on the fixture graph -------------------------------------------------------------------


def test_faithful_edge_attrs_on_the_fixture(fgraph, inputs):
    paths, fdir, _ = inputs
    raw = F.load_faithful_raw(fgraph, paths, fdir, protocol="faithful")
    assert raw.columns == list(F.FAITHFUL_COLUMNS) and raw.height == fgraph.n_edges
    assert all(dt == pl.Float64 for dt in raw.dtypes)
    tx = pl.read_parquet(paths.transactions)
    vocab = load_spec(fdir).vocab
    assert np.array_equal(raw["timestamp"].to_numpy(), tx["minute"] - tx["minute"].min())
    assert np.array_equal(raw["amount_received"].to_numpy(), tx["amount_received"].to_numpy())
    for c in F.CODED_COLUMNS:
        cats = list(vocab[c])
        want = [cats.index(x) if x in cats else -1 for x in tx[c].to_list()]
        assert raw[c].to_numpy().tolist() == want
    want_in, want_out = brute_ports(fgraph.src, fgraph.dst)
    assert raw["mg_in_port"].to_numpy().tolist() == want_in
    assert raw["mg_out_port"].to_numpy().tolist() == want_out
    assert F.snapshot_last_ranks(fgraph) == {
        "train": fgraph.bounds["train_last"],
        "val": fgraph.bounds["val_last"],
        "test": fgraph.bounds["data_last"],
    }
    # an unknown category maps to -1
    vocab2 = {k: list(v) for k, v in vocab.items()}
    vocab2["payment_format"] = vocab2["payment_format"][1:]
    raw2 = F.faithful_edge_attrs(fgraph, tx, vocab2, protocol="faithful")
    first = vocab["payment_format"][0]
    gone = (tx["payment_format"] == first).to_numpy()
    assert gone.any() and (raw2["payment_format"].to_numpy()[gone] == -1).all()
    with pytest.raises(ValueError, match="rank order"):
        F.faithful_edge_attrs(fgraph, tx.reverse(), vocab, protocol="faithful")


def test_per_snapshot_zscore(fgraph, inputs):
    paths, fdir, _ = inputs
    raw = F.load_faithful_raw(fgraph, paths, fdir, protocol="faithful")
    for name, last in F.snapshot_last_ranks(fgraph).items():
        ea = F.snapshot_ea(raw, last, protocol="faithful")
        assert ea.dtype == np.float32 and ea.shape == (last + 1, 6), name
        x = raw.head(last + 1).to_numpy()
        want = (x - x.mean(0)) / np.maximum(x.std(0, ddof=1), 1e-6)
        np.testing.assert_allclose(ea, want.astype(np.float32), rtol=1e-5, atol=1e-5)
        np.testing.assert_allclose(ea.mean(0), 0.0, atol=1e-4)
    # rows after the snapshot never change its statistics; rows inside do
    last = fgraph.bounds["train_last"]
    later = raw.with_columns(
        pl.when(pl.int_range(pl.len()) > last)
        .then(pl.col("amount_received") * 1000)
        .otherwise(pl.col("amount_received"))
    )
    base = F.snapshot_ea(raw, last, protocol="faithful")
    assert np.array_equal(F.snapshot_ea(later, last, protocol="faithful"), base)
    inside = raw.with_columns(
        pl.when(pl.int_range(pl.len()) == 0)
        .then(pl.col("amount_received") + 1e6)
        .otherwise(pl.col("amount_received"))
    )
    assert not np.array_equal(F.snapshot_ea(inside, last, protocol="faithful")[1:, 1], base[1:, 1])
    # a constant column stays finite (std floor)
    const = raw.with_columns(pl.lit(5.0).alias("timestamp"))
    assert (F.snapshot_ea(const, last, protocol="faithful")[:, 0] == 0).all()


def test_faithful_builders_raise_outside_the_protocol(fgraph, inputs):
    paths, fdir, _ = inputs
    raw = F.load_faithful_raw(fgraph, paths, fdir, protocol="faithful")
    tx = pl.read_parquet(paths.transactions)
    vocab = load_spec(fdir).vocab
    for p in [p for p in PROTOCOLS if p != "faithful"] + ["hpo", ""]:
        with pytest.raises(ValueError, match="confined"):
            F.require_faithful(p)
        with pytest.raises(ValueError, match="confined"):
            F.faithful_edge_attrs(fgraph, tx, vocab, protocol=p)
        with pytest.raises(ValueError, match="confined"):
            F.snapshot_ea(raw, 10, protocol=p)
        with pytest.raises(ValueError, match="confined"):
            F.load_faithful_raw(fgraph, paths, fdir, protocol=p)
    pre = F.faithful_preprocess()
    assert pre["columns"] == list(F.FAITHFUL_COLUMNS) and pre["num_idx"] == list(range(6))
    assert pre["cat_idx"] == [] and pre["cat_sizes"] == [] and pre["norm"] == FAITHFUL_EXEMPT_NORM


def test_recalled_list_marks_what_the_papers_do_not_state():
    """review FID-1..5: the faithful summary's (R) list (train.RECALLED, copied into the README)
    marks every recalled or own choice the faithful run makes, and qualifies the (V) parts."""
    from aml.models.gnn.train import RECALLED

    text = " | ".join(RECALLED)
    for needle in (
        "App. C says reverse-MP runs needed a reduced batch",  # FID-1: 8192 is a CLI default
        "Multi-GNN's tuned GIN hidden 66",  # FID-2: 64 is App. F.5's runtime-table size
        "sorted train-vocab order",  # FID-3: code order of a numeric feature
        "first-appearance order",
        "whole-day split",  # FID-4: the split / fanout labels, the unlisted (R) items
        "App. E.1 states 60/20/20",
        "per edge type, uniform, non-disjoint",
        "Adam, constant lr, no weight decay",
        "mean over edge types",
        "on sampled targets only",
        "placement ours",  # FID-5: where layer dropout is applied
    ):
        assert needle in text, needle


# --- snapshots and the loader (torch) ------------------------------------------------------------


def test_snapshot_prefixes_flip_every_edge(fgraph):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from torch_geometric.loader import LinkNeighborLoader

    for name, last in F.snapshot_last_ranks(fgraph).items():
        snap = F.snapshot_hetero(fgraph, last)
        n = last + 1
        assert snap[NODE].num_nodes == fgraph.n_nodes and set(snap[NODE].keys()) == {"num_nodes"}
        for et in (TO, REV):
            assert set(snap[et].keys()) == {"edge_index"} and snap[et].num_edges == n, name
        fwd = np.stack([fgraph.src, fgraph.dst])[:, :n]
        assert np.array_equal(snap[TO].edge_index.numpy(), fwd)
        assert np.array_equal(snap[REV].edge_index.numpy(), fwd[::-1])
    snap = F.snapshot_hetero(fgraph, fgraph.bounds["val_last"])
    seeds = torch.from_numpy(G.split_gids(fgraph, "val_early")[:64])
    ld = LinkNeighborLoader(
        snap,
        num_neighbors={TO: [10, 5], REV: [10, 5]},
        edge_label_index=(TO, snap[TO].edge_index[:, seeds]),
        batch_size=64,
    )
    b = next(iter(ld))
    n_id = b[NODE].n_id
    src, dst = torch.from_numpy(fgraph.src), torch.from_numpy(fgraph.dst)
    # e_id == gid on both types: rev edges are the forward gid's endpoints, flipped
    assert torch.equal(n_id[b[TO].edge_index[0]], src[b[TO].e_id])
    assert torch.equal(n_id[b[REV].edge_index[0]], dst[b[REV].e_id])
    assert torch.equal(n_id[b[REV].edge_index[1]], src[b[REV].e_id])
    assert int(b[TO].e_id.max()) <= fgraph.bounds["val_last"]


def test_faithful_loader_shuffle_is_seeded_per_epoch(fgraph):
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from aml.models.gnn.sampler import make_faithful_loader

    last = fgraph.bounds["train_last"]
    snap = F.snapshot_hetero(fgraph, last)
    seeds = G.split_gids(fgraph, "train")
    runtime = {"num_workers": 0, "loader_timeout_s": 60}
    ld = make_faithful_loader(
        snap, seeds, last_rank=last, fanout=[5, 3], batch_size=512, shuffle=True, seed=3,
        runtime=runtime, device="cpu",
    )  # fmt: skip
    assert ld.generator is None  # the order comes from FaithfulEpochSampler alone

    def order(epoch: int) -> list[int]:
        ld.sampler.set_epoch(epoch)
        return torch.cat([fb.seed_pos for fb in ld]).tolist()

    o0, o1 = order(0), order(1)
    assert sorted(o0) == list(range(len(seeds))) and o0 != list(range(len(seeds)))
    assert order(0) == o0 and o1 != o0
    # the spec's order: randperm from a generator seeded seed * 1000 + epoch (§9)
    want = torch.randperm(len(seeds), generator=torch.Generator().manual_seed(3 * 1000 + 1))
    assert o1 == want.tolist()
    fb = next(iter(ld))
    assert fb.node_sub is None and fb.sampled is not None and fb.sampled.any()
    with pytest.raises(ValueError, match="inside the snapshot"):
        make_faithful_loader(
            snap, np.array([last + 1]), last_rank=last, fanout=[5, 3], batch_size=8,
            shuffle=False, seed=0, runtime=runtime, device="cpu",
        )  # fmt: skip
    with pytest.raises(ValueError, match="non-temporal"):
        make_faithful_loader(
            G.build_hetero(fgraph), seeds, last_rank=last, fanout=[5, 3], batch_size=8,
            shuffle=False, seed=0, runtime=runtime, device="cpu",
        )  # fmt: skip
    # review LEAK-3: the guard bound comes from the seeds' split, not from the snapshot, so a
    # train loader built over the val snapshot cannot run
    val_snap = F.snapshot_hetero(fgraph, fgraph.bounds["val_last"])
    with pytest.raises(ValueError, match="wrong snapshot"):
        make_faithful_loader(
            val_snap, seeds, last_rank=last, fanout=[5, 3], batch_size=8, shuffle=True, seed=0,
            runtime=runtime, device="cpu",
        )  # fmt: skip


def test_faithful_shuffle_order_survives_a_resume_with_workers(fgraph):
    """review TR-2: with persistent workers, a resumed run's first epoch has the same order as
    the uninterrupted run's (DataLoader used to draw the workers' base seed from the shared
    shuffle generator at the first iter() of a process, shifting that epoch's permutation)."""
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from aml.models.gnn.sampler import close_loader, make_faithful_loader

    last = fgraph.bounds["train_last"]
    snap = F.snapshot_hetero(fgraph, last)
    seeds = G.split_gids(fgraph, "train")
    runtime = {"num_workers": 2, "loader_timeout_s": 120}

    def loader():
        return make_faithful_loader(
            snap, seeds, last_rank=last, fanout=[5, 3], batch_size=256, shuffle=True, seed=1,
            runtime=runtime, device="cpu",
        )  # fmt: skip

    def order(ld, epoch: int) -> list[int]:
        ld.sampler.set_epoch(epoch)  # FaithfulEngine.train_epoch's pattern: set, then iter()
        it = iter(ld)
        torch.manual_seed(1000 + epoch)  # seed_epoch re-seeds the global RNG after iter()
        return torch.cat([fb.seed_pos for fb in it]).tolist()

    straight = loader()
    full = [order(straight, e) for e in range(3)]
    close_loader(straight)
    resumed = loader()
    again = [order(resumed, e) for e in (1, 2)]
    close_loader(resumed)
    assert again == full[1:] and full[1] != full[2]
