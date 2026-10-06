"""The GNN's temporal sampler is causal and exact (M3 spec §5, §14.2; PLAN.md §4 as-of rule).

1. The numpy reference (tests/fixtures/gnn_ref.py, written from the spec only) is proven against
   raw PyG / pyg-lib before it judges anything: identical per-subgraph edge multisets, node
   multisets (per-tree dedup) and account-level ego on the hand graph, the tied-minute graph
   and 2,000 seeds of the random tied graph, and identical trees through NeighborLoader.
2. The real `make_loader` + `FlattenTransform`, for every seed of the tied graph and > 2,000
   seeds of the random tied graph: no sampled edge later than first_rank(target) - 1, no
   sampled minute >= the target's (either rank side, either type, hop 2 included), the target
   absent, and the `last` multisets equal the reference; h1's pinned values; workers 0 vs 2
   give identical FlatBatches.
3. Mutations prove the checks have teeth: an off-by-one label time is reported, a strict-`<`
   filter fails the inclusivity pin, a look-ahead batch without the target drop is caught, and
   root-only ego flags fail the h1 pin.
"""

from __future__ import annotations

import numpy as np
import pytest

from tests.fixtures.gnn_graphs import (
    hand_graph_h1,
    make_host_graph,
    random_tied_graph,
    tied_minute_graph,
)
from tests.fixtures.gnn_ref import (
    INCLUSIVITY_CASES,
    LOADER_PARTS,
    REV_TYPE,
    TO_TYPE,
    RefSampler,
    asof_problems,
    day_splits,
    engine_batches,
    flat_observed,
    h1_pin_problems,
    inclusivity_problems,
    raw_loader,
    raw_observed,
    ref_hetero,
    ref_problems,
    require_gnn,
    split_by_days,
)

FAN = [25, 10]  # gnn.yaml sampler.fanout
H1_FAN = [5, 5]
# Ends of days 1..17 as fractions of the minute range: days 1-6 cover 80% (train), day 7 the next
# 8% (val_early, ~2,400 of 30,000 edges), day 8 4%, days 9-18 the rest.
RANDOM_DAY_CUTS = [0.14, 0.27, 0.4, 0.54, 0.67, 0.8, 0.88, 0.92]
RANDOM_DAY_CUTS += [0.93, 0.94, 0.95, 0.96, 0.97, 0.98, 0.985, 0.99, 0.995]


def graphs() -> dict:
    return {"h1": hand_graph_h1(), "tied": tied_minute_graph()}


@pytest.fixture(scope="module")
def random_graph():
    return random_tied_graph()


def _problems(obs, g, bounds, refs=None, *, drop_target=False, causal=True) -> list[str]:
    out = []
    for k, o in enumerate(obs):
        out += asof_problems(o, bound=int(bounds[k]), minute=g.minute, causal=causal)
        if refs is not None:
            out += ref_problems(o, refs[k], drop_target=drop_target)
    return out


# --- 1. the reference is proven against raw PyG -------------------------------------------------


def test_reference_reproduces_the_h1_pins() -> None:
    """numpy only (runs on the laptop): §14.1's verified PyG facts for target A."""
    g = hand_graph_h1()
    pin = g.pinned["A"]
    ref = RefSampler(g.src, g.dst, g.n_nodes, H1_FAN).subgraph(pin["gid"], pin["causal_bound"])
    assert ref.gids("to") == pin["to_gids"]
    assert ref.gids("rev_to") == pin["rev_to_gids"]
    assert sorted(ref.node_counts().elements()) == sorted(pin["n_id"])
    assert ref.n_nodes == len(pin["n_id"])
    assert ref.ego_count == sum(pin["account_ego"])
    assert max(ref.gids("to")) == pin["max_time"]["to"]
    assert max(ref.gids("rev_to")) == pin["max_time"]["rev_to"]
    assert not set(ref.all_gids().tolist()) & set(pin["absent_gids"])
    la = RefSampler(g.src, g.dst, g.n_nodes, H1_FAN).subgraph(pin["gid"], pin["lookahead_bound"])
    assert {et: la.target_copies(et) for et in ("to", "rev_to")} == pin["lookahead_target_copies"]
    later = {et: sum(1 for e in la.edges(et) if e.gid > pin["gid"]) for et in ("to", "rev_to")}
    assert later == pin["lookahead_later_edges"]


def test_reference_reproduces_the_tied_graph_pins() -> None:
    g = tied_minute_graph()
    sampler = RefSampler(g.src, g.dst, g.n_nodes, g.pinned["fanout"])
    for name in ("T1", "T2", "T3"):
        pin = g.pinned[name]
        got = set(sampler.subgraph(pin["gid"], pin["causal_bound"]).all_gids().tolist())
        assert not got & set(pin["same_minute"]), name
        assert set(pin["cycle"]) <= got, name
    t3 = g.pinned["T3"]
    assert t3["unseen_closer"] not in sampler.subgraph(t3["gid"], t3["causal_bound"]).all_gids()


@pytest.mark.parametrize("name", ["h1", "tied"])
@pytest.mark.parametrize("fanout", [[5, 5], [2, 1], [-1, -1]])
@pytest.mark.parametrize("far", [False, True])
def test_reference_equals_pyg_on_hand_graphs(name, fanout, far) -> None:
    require_gnn()
    g = graphs()[name]
    seeds = np.arange(g.n_edges)
    bounds = np.full(g.n_edges, g.n_edges - 1) if far else g.causal_bound
    obs = raw_observed(g.src, g.dst, g.n_nodes, seeds, bounds, fanout)
    ref = RefSampler(g.src, g.dst, g.n_nodes, fanout)
    bad = [
        p
        for o, s, b in zip(obs, seeds, bounds, strict=True)
        for p in ref_problems(o, ref.subgraph(s, b))
    ]
    assert not bad, bad[:5]
    assert [o.seed_pos for o in obs] == list(range(g.n_edges))
    if name == "h1" and fanout == H1_FAN and not far:
        assert h1_pin_problems(obs[4], g.pinned) == []


@pytest.mark.parametrize("fanout", [FAN, [3, 2]])
@pytest.mark.parametrize("far", [False, True])
def test_reference_equals_pyg_on_the_random_tied_graph(random_graph, fanout, far) -> None:
    """2,000 seeds (hubs, ties, self-loops, parallel and reverse edges)."""
    require_gnn()
    g = random_graph
    seeds = np.sort(np.random.default_rng(0).choice(g.n_edges, 2000, replace=False))
    bounds = np.minimum(g.n_edges - 1, seeds + 3000) if far else g.causal_bound[seeds]
    obs = raw_observed(g.src, g.dst, g.n_nodes, seeds, bounds, fanout)
    ref = RefSampler(g.src, g.dst, g.n_nodes, fanout)
    bad = [
        p
        for o, s, b in zip(obs, seeds, bounds, strict=True)
        for p in ref_problems(o, ref.subgraph(s, b))
    ]
    assert not bad, bad[:5]
    n_edges = sum(len(o.all_gids()) for o in obs)
    assert n_edges > 20 * len(seeds)  # rich subgraphs: the comparison is not vacuous


def test_reference_trees_equal_pyg_neighbor_loader(random_graph) -> None:
    """Per tree (not only per merged subgraph): NeighborLoader with disjoint node seeds and edge
    times returns one tree per root; nodes (deduplicated within the tree) and edges equal."""
    require_gnn()
    import torch
    from torch_geometric.loader import NeighborLoader

    g = random_graph
    seeds = np.sort(np.random.default_rng(1).choice(g.n_edges, 1000, replace=False))
    roots = np.concatenate([g.src[seeds], g.dst[seeds]])
    bounds = np.concatenate([g.causal_bound[seeds], g.causal_bound[seeds]])
    loader = NeighborLoader(
        ref_hetero(g.src, g.dst, g.n_nodes),
        num_neighbors={TO_TYPE: FAN, REV_TYPE: FAN},
        input_nodes=("acct", torch.as_tensor(roots)),
        input_time=torch.as_tensor(bounds),
        time_attr="time",
        temporal_strategy="last",
        disjoint=True,
        batch_size=len(roots),
        shuffle=False,
    )
    (b,) = list(loader)
    n_id, tree = b["acct"].n_id.numpy(), b["acct"].batch.numpy()
    rev_gid = g.rev_gid
    to_ei, to_g = b[TO_TYPE].edge_index.numpy(), b[TO_TYPE].e_id.numpy()
    rv_ei, rv_g = b[REV_TYPE].edge_index.numpy(), rev_gid[b[REV_TYPE].e_id.numpy()]
    ref = RefSampler(g.src, g.dst, g.n_nodes, FAN)
    bad = []
    for k in range(len(roots)):
        t = ref.tree(roots[k], bounds[k])
        nodes = n_id[tree == k].tolist()
        ok = (
            nodes[0] == roots[k]
            and sorted(nodes) == sorted(t.nodes)
            and len(set(nodes)) == len(nodes)
            and sorted(to_g[tree[to_ei[1]] == k].tolist()) == sorted(e.gid for e in t.edges["to"])
            and sorted(rv_g[tree[rv_ei[1]] == k].tolist())
            == sorted(e.gid for e in t.edges["rev_to"])
        )
        if not ok:
            bad.append(int(k))
    assert not bad, bad[:10]


def _raw_sampler(name: str):
    g = graphs()[name]

    def sample(seeds, label_times):
        return raw_observed(g.src, g.dst, g.n_nodes, seeds, label_times, H1_FAN)

    return sample


@pytest.mark.parametrize("case", INCLUSIVITY_CASES, ids=[c[0] for c in INCLUSIVITY_CASES])
def test_inclusivity_pin_and_its_strict_mutant(case) -> None:
    """The filter is inclusive (F2): label time == rank(e) samples e, rank(e) - 1 does not. A
    strict `<` filter (emulated by bound - 1) must fail the pin."""
    require_gnn()
    name, target, edge = case
    sample = _raw_sampler(name)
    assert inclusivity_problems(sample, target, edge) == []

    def strict(seeds, label_times):
        return sample(seeds, [int(t) - 1 for t in label_times])

    assert inclusivity_problems(strict, target, edge) != []


@pytest.mark.parametrize("case", INCLUSIVITY_CASES, ids=[c[0] for c in INCLUSIVITY_CASES])
def test_inclusivity_pin_through_make_loader(case) -> None:
    """The real loader hands its label times to PyG unchanged (no hidden - 1, no strict <): the
    pin holds with the target's label time set through `label_times` (and the guard's bound set
    to the same value, so only the sampling is under test)."""
    require_gnn(*LOADER_PARTS)
    from aml.models.gnn import graph, sampler

    name, target, edge = case
    g = graphs()[name]
    hg = make_host_graph(g)
    seeds = np.arange(g.n_edges)

    def sample(targets, label_times):
        (t,), (lt,) = targets, label_times
        bounds = g.causal_bound.copy()
        bounds[t] = lt

        def fixed(g_, gids, **kw):
            return bounds[np.asarray(gids)]

        with pytest.MonkeyPatch.context() as mp:
            for mod in (graph, sampler):
                mp.setattr(mod, "label_times", fixed, raising=False)
                mp.setattr(mod, "guard_bounds", fixed, raising=False)
            batches = engine_batches(hg, seeds, protocol="causal", fanout=H1_FAN, batch_size=8)
        obs, _ = flat_observed(hg, batches)
        return [obs[t]]

    assert inclusivity_problems(sample, target, edge) == []


def test_checker_reports_an_off_by_one_label_time() -> None:
    """Label time = first rank of the minute (no - 1): the checker reports a same-minute edge;
    the correct bound passes for every seed."""
    require_gnn()
    g = tied_minute_graph()
    seeds = np.arange(g.n_edges)
    good = raw_observed(g.src, g.dst, g.n_nodes, seeds, g.causal_bound, H1_FAN)
    assert _problems(good, g, g.causal_bound) == []
    bad = raw_observed(g.src, g.dst, g.n_nodes, seeds, g.first_rank, H1_FAN)
    found = _problems(bad, g, g.causal_bound)
    assert any("same-or-later-minute" in p for p in found), found
    # T1's first-of-minute edge e->a (rank 5) is the one an off-by-one bound lets in
    t1 = g.pinned["T1"]
    assert 5 in bad[t1["gid"]].all_gids() and 5 in t1["same_minute"]


# --- 2. the real loader -------------------------------------------------------------------------


def _random_host(g):
    day = split_by_days(g.minute, RANDOM_DAY_CUTS)
    return make_host_graph(g, split=day_splits(day), day=day, primary_last_day=10)


def test_every_seed_of_the_tied_graph_is_causal_and_exact() -> None:
    require_gnn(*LOADER_PARTS)
    g = tied_minute_graph()
    hg = make_host_graph(g)
    seeds = np.arange(g.n_edges)
    obs, guard = flat_observed(
        hg, engine_batches(hg, seeds, protocol="causal", fanout=H1_FAN, batch_size=4)
    )
    assert [o.seed_pos for o in obs] == list(range(g.n_edges))
    ref = RefSampler(g.src, g.dst, g.n_nodes, H1_FAN)
    refs = [ref.subgraph(s, g.causal_bound[s]) for s in seeds]
    assert _problems(obs, g, g.causal_bound, refs) == []
    for name in ("T1", "T2", "T3"):
        pin = g.pinned[name]
        got = set(obs[pin["gid"]].all_gids().tolist())
        assert not got & set(pin["same_minute"]), name
        assert set(pin["cycle"]) <= got, name
    assert g.pinned["T3"]["unseen_closer"] not in obs[g.pinned["T3"]["gid"]].all_gids()
    total = sum(len(o.all_gids()) for o in obs)
    assert guard["edges_checked"] == total
    assert guard["violations"] == 0 and guard["target_hits"] == 0


def test_random_tied_graph_seeds_are_causal_and_exact(random_graph) -> None:
    """> 2,000 val_early seeds at the real fanout [25, 10]: every subgraph causal; the `last`
    multisets equal the reference on 200 seeds (incl. the 50 with the largest subgraphs)."""
    require_gnn(*LOADER_PARTS)
    g = random_graph
    hg = _random_host(g)
    seeds = np.flatnonzero(hg.split_code == 1)
    assert len(seeds) > 2000
    obs, guard = flat_observed(
        hg, engine_batches(hg, seeds, protocol="causal", fanout=FAN, batch_size=512)
    )
    assert [o.seed_pos for o in obs] == list(range(len(seeds)))
    assert [o.gid for o in obs] == seeds.tolist()
    bounds = g.causal_bound[seeds]
    assert _problems(obs, g, bounds) == []
    sizes = np.array([len(o.all_gids()) for o in obs])
    pick = np.unique(
        np.concatenate(
            [
                np.argsort(-sizes, kind="stable")[:50],
                np.random.default_rng(2).choice(len(seeds), 150, replace=False),
            ]
        )
    )
    ref = RefSampler(g.src, g.dst, g.n_nodes, FAN)
    bad = [p for k in pick for p in ref_problems(obs[k], ref.subgraph(seeds[k], bounds[k]))]
    assert not bad, bad[:5]
    assert guard["edges_checked"] == int(sizes.sum()) > 100_000
    assert guard["violations"] == 0 and guard["target_hits"] == 0
    assert guard["future_edges"] == 0 and guard["dropped_target_copies"] == 0
    assert guard["max_slack"] <= 0


def test_h1_pinned_values_through_make_loader() -> None:
    require_gnn(*LOADER_PARTS)
    g = hand_graph_h1()
    hg = make_host_graph(g)
    obs, guard = flat_observed(
        hg,
        engine_batches(hg, np.arange(g.n_edges), protocol="causal", fanout=H1_FAN, batch_size=64),
    )
    assert h1_pin_problems(obs[4], g.pinned) == []
    b = g.pinned["B"]
    assert max(obs[b["gid"]].all_gids()) <= b["causal_bound"]
    for gid in (0, 1, 2):  # minute 0: label time -1, an empty subgraph (two roots)
        assert obs[gid].n_nodes == 2 and len(obs[gid].all_gids()) == 0
    assert guard["violations"] == 0 and guard["target_hits"] == 0


def test_workers_0_and_2_give_identical_flat_batches(random_graph) -> None:
    require_gnn(*LOADER_PARTS)
    hg = _random_host(random_graph)
    seeds = np.flatnonzero(hg.split_code == 1)
    runs = [
        engine_batches(hg, seeds, protocol="causal", fanout=FAN, batch_size=256, num_workers=w)
        for w in (0, 2)
    ]
    assert len(runs[0]) == len(runs[1]) == -(-len(seeds) // 256)
    for a, b in zip(*runs, strict=True):
        assert a._fields == b._fields
        for name, x, y in zip(a._fields, a, b, strict=True):
            if x is None or isinstance(x, int):
                assert x == y, name
            else:
                assert x.dtype == y.dtype and x.device.type == "cpu", name
                assert x.shape == y.shape and bool((x == y).all()), name


# --- 3. mutations: B's transform on raw batches ----------------------------------------------


def _transform_batches(g, hg, seeds, label_times, guard_bound, **kw):
    """§5.1's raw loader over gnn_ref's own HeteroData with B's FlattenTransform."""
    from aml.models.gnn.transforms import FlattenTransform

    ft = FlattenTransform(
        seed_gids=np.asarray(seeds, np.int64),
        rev_gid=hg.rev_gid,
        guard_bound=np.asarray(guard_bound, np.int64),
        first_rank=hg.first_rank,
        **kw,
    )
    data = ref_hetero(g.src, g.dst, g.n_nodes)
    return list(raw_loader(data, seeds, label_times, H1_FAN, batch_size=64, transform=ft))


def test_lookahead_without_the_target_drop_is_caught() -> None:
    """drop_target=False on a look-ahead batch: the guard raises or reports the target copies,
    and the checker sees them (h1 target A: 2 per edge type, F9); with the drop they are gone."""
    require_gnn(*LOADER_PARTS)
    from aml.models.gnn import LeakError

    g = hand_graph_h1()
    hg = make_host_graph(g)
    seeds = np.arange(g.n_edges)
    bound = np.full(g.n_edges, hg.bounds["train_last"])  # every h1 edge is train
    ref = RefSampler(g.src, g.dst, g.n_nodes, H1_FAN)
    try:
        batches = _transform_batches(
            g, hg, seeds, bound, bound, protocol="lookahead", drop_target=False
        )
    except LeakError:
        pass  # the guard refused the undropped target: detected
    else:
        obs, guard = flat_observed(hg, batches)
        copies = sum(int((o.all_gids() == o.gid).sum()) for o in obs)
        assert copies == sum(
            ref.subgraph(s, b).target_copies(et)
            for s, b in zip(seeds, bound, strict=True)
            for et in ("to", "rev_to")
        )
        assert int((obs[4].all_gids() == 4).sum()) == 4
        assert any("copies of the target" in p for p in _problems(obs, g, bound, causal=False))
        assert guard["target_hits"] == copies  # §5.5: hits = (gid == tgt_gid[sub]).sum()
    dropped, guard = flat_observed(
        hg, _transform_batches(g, hg, seeds, bound, bound, protocol="lookahead", drop_target=True)
    )
    refs = [ref.subgraph(s, b) for s, b in zip(seeds, bound, strict=True)]
    assert _problems(dropped, g, bound, refs, drop_target=True, causal=False) == []
    assert guard["dropped_target_copies"] == sum(
        r.target_copies(et) for r in refs for et in ("to", "rev_to")
    )


def test_root_only_ego_fails_the_h1_pin() -> None:
    require_gnn(*LOADER_PARTS)
    g = hand_graph_h1()
    hg = make_host_graph(g)
    seeds = np.arange(g.n_edges)
    kw = {"protocol": "causal", "drop_target": False}
    for ego, ok in (("account", True), ("root", False)):
        batches = _transform_batches(g, hg, seeds, g.causal_bound, g.causal_bound, ego=ego, **kw)
        obs, _ = flat_observed(hg, batches)
        assert (h1_pin_problems(obs[4], g.pinned) == []) is ok, ego


def test_off_by_one_label_time_through_make_loader_is_caught(monkeypatch) -> None:
    """make_loader fed label time = first rank of the minute (the classic off-by-one), with the
    guard's bound shifted the same way so only an independent check can notice: the guard (target
    hits) or the checker (same-minute edges) must report it."""
    require_gnn(*LOADER_PARTS)
    from aml.models.gnn import LeakError, graph, sampler

    g = tied_minute_graph()
    hg = make_host_graph(g)
    calls: list[str] = []

    def off_by_one(name):
        def fn(g_, gids, **kw):
            calls.append(name)
            return np.asarray(g_.first_rank, np.int64)[np.asarray(gids)]

        return fn

    for mod in (graph, sampler):
        monkeypatch.setattr(mod, "label_times", off_by_one("label_times"), raising=False)
        monkeypatch.setattr(mod, "guard_bounds", off_by_one("guard_bounds"), raising=False)
    seeds = np.arange(g.n_edges)
    try:
        batches = engine_batches(hg, seeds, protocol="causal", fanout=H1_FAN, batch_size=64)
    except LeakError:
        return
    assert "label_times" in calls, "make_loader must take edge_label_time from label_times (§5.1)"
    obs, _ = flat_observed(hg, batches)
    found = _problems(obs, g, g.causal_bound)
    assert any("same-or-later-minute" in p for p in found), found
