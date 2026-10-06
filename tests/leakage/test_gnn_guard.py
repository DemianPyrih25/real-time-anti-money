"""The runtime as-of guard (M3 spec §5.5) is independent of the sampler's bound and counts exactly.

- One seed's edge_label_time corrupted to +1, with the loader consistent with it (the sampler
  really takes the extra edge): the transform raises LeakError, because its bound comes from
  `guard_bounds`, a code path that never calls `label_times`. Shown on raw batches and through
  `make_loader` with `label_times` monkeypatched.
- `guard_bounds` never calls `label_times` (patched to raise) and equals the reference bounds.
- Exact guard counts on h1 (causal; look-ahead incl. the dropped target copies) against the
  reference sampler.
- Faithful batches get the snapshot guard: a sampled rank beyond the snapshot raises.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from tests.fixtures.gnn_graphs import (
    hand_graph_h1,
    make_host_graph,
    random_tied_graph,
    tied_minute_graph,
)
from tests.fixtures.gnn_ref import (
    LOADER_PARTS,
    REV_TYPE,
    TO_TYPE,
    RefSampler,
    day_splits,
    engine_batches,
    flat_observed,
    raw_loader,
    ref_hetero,
    ref_label_times,
    require_gnn,
    split_by_days,
)

H1_FAN = [5, 5]
FAN = [25, 10]


def _corruptible(g) -> int:
    """A seed whose bound + 1 admits an edge its subgraph samples: the first edge of its minute
    touches the target's src or dst (so hop 1 takes it under the corrupted bound)."""
    first = g.first_rank
    for s in range(g.n_edges):
        e = int(first[s])
        if e == s:
            continue
        ends = {int(g.src[s]), int(g.dst[s])}
        if int(g.dst[e]) in ends or int(g.src[e]) in ends:
            return s
    raise AssertionError("no corruptible seed")


def _raw_transform_batches(g, hg, label_times, guard_bound, fanout, batch_size=64):
    from aml.models.gnn.transforms import FlattenTransform

    seeds = np.arange(g.n_edges, dtype=np.int64)
    ft = FlattenTransform(
        protocol="causal",
        seed_gids=seeds,
        rev_gid=hg.rev_gid,
        guard_bound=np.asarray(guard_bound, np.int64),
        first_rank=hg.first_rank,
        drop_target=False,
    )
    data = ref_hetero(g.src, g.dst, g.n_nodes)
    return list(raw_loader(data, seeds, label_times, fanout, batch_size=batch_size, transform=ft))


@pytest.mark.parametrize("name", ["h1", "tied", "random"])
def test_guard_raises_on_a_corrupted_label_time(name) -> None:
    require_gnn(*LOADER_PARTS)
    from aml.models.gnn import LeakError
    from aml.models.gnn.graph import guard_bounds

    g = {"h1": hand_graph_h1, "tied": tied_minute_graph}.get(name, _small_random)()
    hg = make_host_graph(g)
    fanout = FAN if name == "random" else H1_FAN
    seeds = np.arange(g.n_edges)
    bound = guard_bounds(hg, seeds, protocol="causal")
    np.testing.assert_array_equal(bound, g.causal_bound)
    # consistent bounds: clean
    _, guard = flat_observed(hg, _raw_transform_batches(g, hg, g.causal_bound, bound, fanout))
    assert guard["violations"] == 0 and guard["target_hits"] == 0
    # one seed + 1, the loader samples with it, the guard's bound is independent
    s = 4 if name == "h1" else _corruptible(g)
    lt = g.causal_bound.copy()
    lt[s] += 1
    with pytest.raises(LeakError) as err:
        _raw_transform_batches(g, hg, lt, bound, fanout)
    assert err.value.detail  # the counts and offending seeds travel with the error
    assert isinstance(pickle.loads(pickle.dumps(err.value)), LeakError)


def _small_random():
    return random_tied_graph(n_nodes=300, n_edges=3000, per_minute=10, seed=3)


def test_guard_raises_through_make_loader(monkeypatch) -> None:
    """make_loader's label times come from graph.label_times; corrupting one seed's (+1) makes
    the real loader sample a same-minute edge and its own guard must refuse the batch."""
    require_gnn(*LOADER_PARTS)
    from aml.models.gnn import LeakError, graph, sampler

    g = hand_graph_h1()
    hg = make_host_graph(g)
    real = graph.label_times
    calls: list[int] = []

    def corrupted(g_, gids, **kw):
        out = np.array(real(g_, gids, **kw), dtype=np.int64, copy=True)
        gids = np.asarray(gids)
        out[gids == 4] += 1  # target A: bound 3 admits y->u (rank 3, same minute)
        calls.append(len(gids))
        return out

    for mod in (graph, sampler):
        monkeypatch.setattr(mod, "label_times", corrupted, raising=False)
    with pytest.raises(LeakError):
        engine_batches(hg, np.arange(g.n_edges), protocol="causal", fanout=H1_FAN, batch_size=64)
    assert calls, "make_loader must take edge_label_time from graph.label_times (§5.1)"


@pytest.mark.parametrize("protocol", ["causal", "pna", "lookahead"])
def test_guard_bounds_never_call_label_times(monkeypatch, protocol) -> None:
    require_gnn("graph.guard_bounds", torch=False)
    from aml.models.gnn import graph

    def boom(*a, **kw):
        raise AssertionError("guard_bounds must not call label_times (§5.5)")

    monkeypatch.setattr(graph, "label_times", boom)
    g = _small_random()
    cuts = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, *np.linspace(0.85, 0.99, 9)]
    day = split_by_days(g.minute, cuts)
    hg = make_host_graph(g, split=day_splits(day), day=day, primary_last_day=10)
    gids = np.arange(g.n_edges)
    for tb in ("end", "d10"):
        got = graph.guard_bounds(hg, gids, protocol=protocol, test_bound=tb)
        want = ref_label_times(
            gids,
            minute=g.minute,
            split_code=hg.split_code,
            bounds=hg.bounds,
            protocol=protocol,
            test_bound=tb,
        )
        np.testing.assert_array_equal(np.asarray(got, np.int64), want)


def _ref_totals(g, seeds, bounds, fanout, *, drop: bool) -> dict:
    """Reference guard totals over seeds. `drop`: look-ahead (target copies removed)."""
    sampler = RefSampler(g.src, g.dst, g.n_nodes, fanout)
    first = g.first_rank
    out = {"pre_edges": 0, "post_edges": 0, "pre_future": 0, "post_future": 0, "copies": 0}
    slack_pre, slack_post = [], []
    for s, b in zip(seeds, bounds, strict=True):
        r = sampler.subgraph(s, b)
        out["copies"] += sum(r.target_copies(et) for et in ("to", "rev_to"))
        out["pre_edges"] += r.n_edges()
        out["post_edges"] += r.n_edges(drop_target=drop)
        out["pre_future"] += r.future_count(int(first[s]))
        out["post_future"] += r.future_count(int(first[s]), drop_target=drop)
        for acc, d in ((slack_pre, False), (slack_post, drop)):
            m = r.max_slack(drop_target=d)
            if m is not None:
                acc.append(m)
    out["pre_slack"] = max(slack_pre) if slack_pre else None
    out["post_slack"] = max(slack_post) if slack_post else None
    return out


@pytest.mark.parametrize("batch_size", [64, 1])
def test_exact_guard_counts_on_h1_causal(batch_size) -> None:
    require_gnn(*LOADER_PARTS)
    from aml.models.gnn import NO_SLACK

    g = hand_graph_h1()
    hg = make_host_graph(g)
    seeds = np.arange(g.n_edges)
    obs, guard = flat_observed(
        hg,
        engine_batches(hg, seeds, protocol="causal", fanout=H1_FAN, batch_size=batch_size),
    )
    ref = _ref_totals(g, seeds, g.causal_bound, H1_FAN, drop=False)
    assert ref["pre_edges"] == 55 and ref["pre_slack"] == 0  # hand-checkable on the h1 drawing
    assert guard == {
        "edges_checked": ref["pre_edges"],
        "violations": 0,
        "target_hits": 0,
        "max_slack": ref["pre_slack"] if ref["pre_slack"] is not None else NO_SLACK,
        "future_edges": 0,
        "dropped_target_copies": 0,
    }
    # target A alone: 5 `to` + 4 `rev_to` edges, its last edge exactly at the bound (slack 0)
    a = RefSampler(g.src, g.dst, g.n_nodes, H1_FAN).subgraph(4, 2)
    assert a.n_edges() == 9 and a.max_slack() == 0
    assert len(obs[4].all_gids()) == 9


def test_exact_guard_counts_on_h1_lookahead() -> None:
    """Look-ahead (every h1 edge is train: bound train_last = 7): 0 violations, every target copy
    dropped and counted (target A: 2 + 2), 0 hits left after the drop (§5.3). `edges_checked`,
    `future_edges` and `max_slack` cover ALL sampled edges, the dropped copies included (§5.3:
    future share = edges with rank >= the target minute's first rank / all sampled edges)."""
    require_gnn(*LOADER_PARTS)
    g = hand_graph_h1()
    hg = make_host_graph(g)
    seeds = np.arange(g.n_edges)
    bound = np.full(g.n_edges, hg.bounds["train_last"])
    assert hg.bounds["train_last"] == g.pinned["A"]["lookahead_bound"]
    obs, guard = flat_observed(
        hg, engine_batches(hg, seeds, protocol="lookahead", fanout=H1_FAN, batch_size=64)
    )
    ref = _ref_totals(g, seeds, bound, H1_FAN, drop=True)
    assert guard["violations"] == 0
    assert guard["dropped_target_copies"] == ref["copies"]
    a = RefSampler(g.src, g.dst, g.n_nodes, H1_FAN).subgraph(4, 7)
    assert a.target_copies("to") == 2 and a.target_copies("rev_to") == 2
    assert not (obs[4].all_gids() == 4).any()
    assert guard["target_hits"] == 0
    assert guard["edges_checked"] == ref["pre_edges"] == 182
    assert guard["future_edges"] == ref["pre_future"] == 121
    assert guard["max_slack"] == ref["pre_slack"] == 0
    assert sum(len(o.all_gids()) for o in obs) == ref["post_edges"] == 182 - 30


def _faithful_hetero(g, last_rank: int):
    """§9's snapshot, built here: ranks <= last_rank, `to` in rank order and `rev_to` = every edge
    of the snapshot flipped (self-loops included), so e_id == gid on both stores."""
    import torch
    from torch_geometric.data import HeteroData

    src = torch.as_tensor(g.src[: last_rank + 1])
    dst = torch.as_tensor(g.dst[: last_rank + 1])
    data = HeteroData()
    data["acct"].num_nodes = g.n_nodes
    data[TO_TYPE].edge_index = torch.stack([src, dst])
    data[REV_TYPE].edge_index = torch.stack([dst, src])
    return data


def test_faithful_snapshot_guard_refuses_edges_beyond_the_snapshot() -> None:
    """§5.5: faithful batches get a snapshot guard (every sampled rank <= the snapshot's last
    rank). The non-temporal, non-disjoint [100, 100] loader over the train snapshot passes; the
    same loader over the whole graph (edges after the snapshot reachable) must raise."""
    require_gnn("transforms.FlattenTransform.__init__", "transforms.FlattenTransform.__call__")
    import torch
    from torch_geometric.loader import LinkNeighborLoader

    from aml.models.gnn import LeakError, add_guard, empty_guard
    from aml.models.gnn.transforms import FlattenTransform

    g = _small_random()
    last = g.n_edges // 2
    seeds = np.arange(last + 1, dtype=np.int64)

    def batches(data):
        ft = FlattenTransform(
            protocol="faithful",
            seed_gids=seeds,
            rev_gid=np.arange(g.n_edges, dtype=np.int64),
            guard_bound=np.full(len(seeds), last, dtype=np.int64),
            first_rank=None,
        )
        loader = LinkNeighborLoader(
            data,
            num_neighbors={TO_TYPE: [100, 100], REV_TYPE: [100, 100]},
            edge_label_index=(TO_TYPE, data[TO_TYPE].edge_index[:, torch.as_tensor(seeds)]),
            batch_size=256,
            shuffle=False,
            transform=ft,
        )
        return list(loader)

    total = empty_guard()
    for fb in batches(_faithful_hetero(g, last)):
        assert int(fb.gid_fwd.max()) <= last and int(fb.gid_rev.max()) <= last
        total = add_guard(total, fb.guard.tolist())
    assert total["violations"] == 0 and total["edges_checked"] > 0
    with pytest.raises(LeakError):
        batches(_faithful_hetero(g, g.n_edges - 1))


def test_faithful_snapshot_guard_is_exact_at_the_boundary() -> None:
    """The snapshot guard's bound is exactly the snapshot's last rank: make_faithful_loader's
    batches over the snapshot reach slack 0 (the edge AT the last rank is sampled and allowed),
    and a snapshot with one more edge (rank last + 1) is refused by a guard bounded at last."""
    require_gnn(
        "faithful.snapshot_hetero",
        "sampler.make_faithful_loader",
        "transforms.FlattenTransform.__init__",
        "transforms.FlattenTransform.__call__",
    )
    from aml.models.gnn import LeakError, add_guard, empty_guard
    from aml.models.gnn.faithful import snapshot_hetero
    from aml.models.gnn.sampler import make_faithful_loader
    from aml.models.gnn.transforms import FlattenTransform

    g = _small_random()
    hg = make_host_graph(g)
    last = g.n_edges // 3
    seeds = np.arange(last + 1, dtype=np.int64)
    loader = make_faithful_loader(
        snapshot_hetero(hg, last),
        seeds,
        last_rank=last,
        fanout=[-1, -1],  # every neighbour: the edge at rank `last` is surely sampled
        batch_size=128,
        shuffle=False,
        seed=0,
        runtime={"num_workers": 0, "loader_timeout_s": 120},
        device="cpu",
    )
    total = empty_guard()
    for fb in loader:
        total = add_guard(total, fb.guard.tolist())
    assert total["violations"] == 0 and total["max_slack"] == 0, total

    beyond = np.arange(last + 2, dtype=np.int64)
    ft = FlattenTransform(
        protocol="faithful",
        seed_gids=beyond,
        rev_gid=None,
        guard_bound=np.full(len(beyond), last, dtype=np.int64),
        first_rank=None,
    )
    data = _faithful_hetero(g, last + 1)
    raw = _every_neighbour_loader(data, beyond, ft)
    with pytest.raises(LeakError) as err:
        list(raw)
    assert err.value.detail["guard"]["max_slack"] == 1


def _every_neighbour_loader(data, seeds, transform):
    """Non-temporal, non-disjoint LinkNeighborLoader taking every neighbour (2 hops)."""
    import torch
    from torch_geometric.loader import LinkNeighborLoader

    return LinkNeighborLoader(
        data,
        num_neighbors={TO_TYPE: [-1, -1], REV_TYPE: [-1, -1]},
        edge_label_index=(TO_TYPE, data[TO_TYPE].edge_index[:, torch.as_tensor(seeds)]),
        batch_size=128,
        shuffle=False,
        transform=transform,
    )
