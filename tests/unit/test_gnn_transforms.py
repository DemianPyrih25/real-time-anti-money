"""FlattenTransform, the as-of guard, the edge-cap split, device inputs, the eval cache and the
epoch sampler (M3 spec §5), through the real make_loader on hand-made and random tied graphs."""

from __future__ import annotations

import time

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch import nn  # noqa: E402
from torch_geometric.loader import LinkNeighborLoader  # noqa: E402
from torch_geometric.nn import GINEConv  # noqa: E402

from aml.models.gnn import GUARD_FIELDS, NO_SLACK, REV, TO, LeakError, add_guard  # noqa: E402
from aml.models.gnn import graph as G  # noqa: E402
from aml.models.gnn import sampler as S  # noqa: E402
from aml.models.gnn import transforms as T  # noqa: E402
from tests.fixtures import gnn_graphs as fx  # noqa: E402

H1 = fx.H1_PINNED
RUNTIME = {"num_workers": 0, "loader_timeout_s": 60}


def scfg(fanout=(5, 5), batch_size=64, eval_batch_size=64, ego="account") -> dict:
    return {
        "fanout": list(fanout),
        "temporal_strategy": "last",
        "ego": ego,
        "batch_size": batch_size,
        "eval_batch_size": eval_batch_size,
        "max_edges_per_step": None,
        "max_edges_per_eval_step": None,
        "eval_cache_max_gb": 0.5,
    }


@pytest.fixture(scope="module")
def h1():
    g = fx.hand_graph_h1()
    hg = fx.make_host_graph(g)
    return g, hg, G.build_hetero(hg)


@pytest.fixture(scope="module")
def rnd():
    g = fx.random_tied_graph(n_nodes=400, n_edges=6000, per_minute=12, seed=5)
    # 12 days of 500 edges (ties kept within a day): train d1-6, val d7 / d8, test d9-12
    day = np.repeat(np.arange(1, 13), 500).astype(np.int64)
    minute = (day - 1) * 1440 + g.minute  # g.minute < 500: monotone, tied within a day
    g = fx.ArrayGraph(src=g.src, dst=g.dst, minute=minute, n_nodes=g.n_nodes)
    split = ["train"] * 3000 + ["val_early"] * 500 + ["val_late"] * 500 + ["test"] * 2000
    hg = fx.make_host_graph(g, split=split, primary_last_day=10)
    assert hg.bounds["test_first"] < hg.bounds["d10_last"] < hg.bounds["data_last"]
    return g, hg, G.build_hetero(hg)


def loader(hg, hetero, seeds, *, protocol="causal", cfg=None, role="eval", **kw):
    return S.make_loader(
        hg,
        hetero,
        np.asarray(seeds, dtype=np.int64),
        protocol=protocol,
        role=role,
        sampler_cfg=cfg or scfg(),
        runtime=kw.pop("runtime", RUNTIME),
        device="cpu",
        **kw,
    )


def raw_and_flat(ld):
    """(raw HeteroData batches, FlatBatches) of one loader pass (transform applied by hand)."""
    tf = ld.transform
    ld.transform = None
    try:
        raw = list(ld)
    finally:
        ld.transform = tf
    return raw, [tf(b) for b in raw]


def guard_dict(fb) -> dict:
    return dict(zip(GUARD_FIELDS, fb.guard.tolist(), strict=True))


# --- FlatBatch layout ----------------------------------------------------------------------------


def test_flatbatch_shapes_dtypes_and_layout(rnd):
    g, hg, d = rnd
    seeds = G.split_gids(hg, "val_early")
    ld = loader(hg, d, seeds, cfg=scfg(fanout=(25, 10), eval_batch_size=128))
    raw, flat = raw_and_flat(ld)
    pos = []
    for rb, fb in zip(raw, flat, strict=True):
        assert isinstance(fb, T.FlatBatch)
        b, n = int(fb.tgt_gid.numel()), fb.n_nodes
        assert n == int(rb["acct"].n_id.numel())
        for f in fb._fields:
            v = getattr(fb, f)
            if isinstance(v, torch.Tensor):
                assert v.device.type == "cpu", f
                assert v.dtype == (torch.bool if f == "ego" else torch.int64), f
        assert fb.ego.shape == (n,) and fb.node_sub.shape == (n,)
        assert fb.ei_fwd.shape == (2, fb.gid_fwd.numel())
        assert fb.ei_rev.shape == (2, fb.gid_rev.numel())
        for t in (fb.tgt_src, fb.tgt_dst, fb.tgt_gid, fb.seed_pos, fb.sub_edges):
            assert t.shape == (b,)
        assert fb.sampled is None and fb.guard.shape == (6,)
        assert int(fb.sub_edges.sum()) == fb.gid_fwd.numel() + fb.gid_rev.numel()
        for ei in (fb.ei_fwd, fb.ei_rev):
            assert ei.numel() == 0 or (int(ei.min()) >= 0 and int(ei.max()) < n)
        assert torch.equal(fb.node_sub[fb.tgt_src], torch.arange(b))
        assert torch.equal(fb.node_sub[fb.tgt_dst], torch.arange(b))
        assert torch.equal(fb.tgt_gid, torch.from_numpy(seeds)[fb.seed_pos])
        # roots are the target's accounts; every edge is the gid's own (src, dst)
        n_id = rb["acct"].n_id
        assert torch.equal(n_id[fb.tgt_src], torch.from_numpy(g.src)[fb.tgt_gid])
        assert torch.equal(n_id[fb.tgt_dst], torch.from_numpy(g.dst)[fb.tgt_gid])
        src, dst = torch.from_numpy(g.src), torch.from_numpy(g.dst)
        assert torch.equal(n_id[fb.ei_fwd[0]], src[fb.gid_fwd])
        assert torch.equal(n_id[fb.ei_fwd[1]], dst[fb.gid_fwd])
        assert torch.equal(n_id[fb.ei_rev[0]], dst[fb.gid_rev])  # reversed
        assert torch.equal(n_id[fb.ei_rev[1]], src[fb.gid_rev])
        assert not bool((src[fb.gid_rev] == dst[fb.gid_rev]).any())  # no self-loop reversed
        pos += fb.seed_pos.tolist()
    assert pos == list(range(len(seeds)))  # every seed, in order, never down-sampled


def test_device_inputs_gather_by_gid(rnd):
    g, hg, d = rnd
    seeds = G.split_gids(hg, "train")[-300:]
    fb = next(iter(loader(hg, d, seeds, cfg=scfg(eval_batch_size=300))))
    ea = torch.from_numpy(hg.ea)
    x, eif, eaf, eir, ear, ts, td, ta = T.to_model_inputs(fb, ea, "cpu")
    assert x.shape == (fb.n_nodes, 1) and x.dtype == torch.float32
    assert torch.equal(x[:, 0], fb.ego.float())
    assert torch.equal(eaf, ea[fb.gid_fwd]) and torch.equal(ear, ea[fb.gid_rev])
    assert torch.equal(ta, ea[fb.tgt_gid])
    assert torch.equal(eif, fb.ei_fwd) and torch.equal(eir, fb.ei_rev)
    assert torch.equal(ts, fb.tgt_src) and torch.equal(td, fb.tgt_dst)
    # a reverse copy carries its forward edge's row
    common = np.intersect1d(fb.gid_fwd.numpy(), fb.gid_rev.numpy())
    assert len(common)
    k = int(common[0])
    row_f = eaf[(fb.gid_fwd == k).nonzero()[0, 0]]
    row_r = ear[(fb.gid_rev == k).nonzero()[0, 0]]
    assert torch.equal(row_f, row_r)
    y = torch.from_numpy(hg.y)
    assert torch.equal(T.batch_labels(fb, y), y[fb.tgt_gid].long())
    bad = y.clone()
    bad[int(fb.tgt_gid[0])] = -1
    with pytest.raises(ValueError, match="no loaded label"):
        T.batch_labels(fb, bad)


# --- ego flags, self-loops, empty subgraphs (hand graph h1) --------------------------------------


def test_h1_pinned_ego_and_guard(h1):
    g, hg, d = h1
    pa = H1["A"]
    ld = loader(hg, d, [pa["gid"]], cfg=scfg(fanout=H1["fanout"]))
    (rb,), (fb,) = raw_and_flat(ld)
    assert rb["acct"].n_id.tolist() == pa["n_id"]
    assert fb.ego.int().tolist() == pa["account_ego"]
    assert sorted(fb.gid_fwd.tolist()) == pa["to_gids"]
    assert sorted(fb.gid_rev.tolist()) == pa["rev_to_gids"]
    assert fb.node_sub.tolist() == pa["batch"]
    assert guard_dict(fb) == {
        "edges_checked": 9,
        "violations": 0,
        "target_hits": 0,
        "max_slack": 0,  # max sampled time 2 == bound 2 (inclusive)
        "future_edges": 0,
        "dropped_target_copies": 0,
    }
    root = loader(hg, d, [pa["gid"]], cfg=scfg(fanout=H1["fanout"], ego="root"))
    assert next(iter(root)).ego.int().tolist() == pa["root_ego"]


def test_self_loop_target_and_empty_subgraph(h1):
    g, hg, d = h1
    # z->z (gid 6, minute 2): both roots are z; nothing earlier touches z
    fb = next(iter(loader(hg, d, [6], cfg=scfg(fanout=H1["fanout"]))))
    assert fb.n_nodes == 2 and fb.ego.tolist() == [True, True]
    assert fb.gid_fwd.numel() == fb.gid_rev.numel() == 0 and fb.sub_edges.tolist() == [0]
    # look-ahead: the self-loop is sampled once per tree (to only) and dropped
    la = next(iter(loader(hg, d, [6], protocol="lookahead", cfg=scfg(fanout=H1["fanout"]))))
    gd = guard_dict(la)
    assert gd["dropped_target_copies"] == 2 and gd["target_hits"] == 0
    assert 6 not in la.gid_fwd.tolist() and la.ego.tolist() == [True, True]
    # gid 0 is in the first minute: causal bound -1 -> an empty subgraph (F10)
    e = next(iter(loader(hg, d, [0], cfg=scfg(fanout=H1["fanout"]))))
    assert e.n_nodes == 2 and e.gid_fwd.numel() == e.gid_rev.numel() == 0
    assert guard_dict(e) == {f: (NO_SLACK if f == "max_slack" else 0) for f in GUARD_FIELDS}
    assert torch.equal(e.guard, T.zero_guard())


def test_account_ego_marks_every_copy_of_the_target_accounts(rnd):
    g, hg, d = rnd
    seeds = G.split_gids(hg, "test")
    ld = loader(hg, d, seeds, cfg=scfg(fanout=(25, 10), eval_batch_size=300))
    raw, flat = raw_and_flat(ld)
    extra = 0
    for rb, fb in zip(raw, flat, strict=True):
        n_id = rb["acct"].n_id
        u = n_id[fb.tgt_src][fb.node_sub]
        v = n_id[fb.tgt_dst][fb.node_sub]
        assert torch.equal(fb.ego, (n_id == u) | (n_id == v))
        extra += int(fb.ego.sum()) - 2 * int(fb.tgt_gid.numel())
    assert extra > 0  # copies of u / v inside the other endpoint's tree do occur


# --- the guard -----------------------------------------------------------------------------------


def _manual_loader(hg, d, seeds, label_time, transform, fanout=(5, 5)):
    """A loader whose edge_label_time is given explicitly (mutations), with `transform`."""
    return LinkNeighborLoader(
        d,
        num_neighbors={TO: list(fanout), REV: list(fanout)},
        edge_label_index=(TO, d[TO].edge_index[:, torch.as_tensor(seeds)]),
        edge_label_time=torch.as_tensor(label_time, dtype=torch.int64),
        time_attr="time",
        temporal_strategy="last",
        disjoint=True,
        batch_size=len(seeds),
        shuffle=False,
        transform=transform,
    )


def test_guard_raises_on_a_corrupted_bound(rnd):
    g, hg, d = rnd
    seeds = G.split_gids(hg, "val_early")[:200]
    lt = G.label_times(hg, seeds, protocol="causal")
    gb = G.guard_bounds(hg, seeds, protocol="causal")
    ok = T.FlattenTransform(
        protocol="causal", seed_gids=seeds, rev_gid=hg.rev_gid, guard_bound=gb,
        first_rank=hg.first_rank,
    )  # fmt: skip
    fb = next(iter(_manual_loader(hg, d, seeds, lt, ok)))
    assert guard_dict(fb)["violations"] == 0 and guard_dict(fb)["edges_checked"] > 500
    # the loader samples with first_rank (off by one: same-minute edges and the target itself)
    with pytest.raises(LeakError) as e:
        next(iter(_manual_loader(hg, d, seeds, hg.first_rank[seeds], ok)))
    problems = {p["problem"] for p in e.value.detail["findings"]}
    assert "edge later than its bound" in problems
    # one seed's guard bound corrupted to -1 while the loader is consistent with label_times
    i = next(k for k in range(len(seeds)) if int(fb.sub_edges[k]) > 0)
    gb_bad = gb.copy()
    gb_bad[i] = -1
    bad = T.FlattenTransform(
        protocol="causal", seed_gids=seeds, rev_gid=hg.rev_gid, guard_bound=gb_bad,
        first_rank=hg.first_rank,
    )  # fmt: skip
    with pytest.raises(LeakError) as e:
        next(iter(_manual_loader(hg, d, seeds, lt, bad)))
    detail = e.value.detail
    assert detail["guard"]["violations"] == int(fb.sub_edges[i])
    assert set(detail["findings"][0]["seed_pos"]) == {i}


def test_causal_subgraphs_hold_only_earlier_minutes(rnd):
    """Independent of ranks and bounds: every sampled edge's minute < its target's minute."""
    g, hg, d = rnd
    ld = loader(hg, d, np.arange(g.n_edges), cfg=scfg(fanout=(25, 10), eval_batch_size=1500))
    minute = torch.from_numpy(g.minute)
    n, total = 0, None
    for fb in ld:
        tm = minute[fb.tgt_gid]
        for gid, ei in ((fb.gid_fwd, fb.ei_fwd), (fb.gid_rev, fb.ei_rev)):
            assert bool((minute[gid] < tm[fb.node_sub[ei[1]]]).all())
            n += gid.numel()
        total = add_guard(total, fb.guard.tolist())
    assert n > 100_000 and total["edges_checked"] == n
    assert total["violations"] == total["target_hits"] == total["future_edges"] == 0
    assert total["max_slack"] <= 0


def test_guard_checks_that_time_is_the_rank(h1):
    g, hg, _ = h1
    bad = G.build_hetero(hg)
    bad[REV].time = torch.arange(bad[REV].num_edges)  # sorted, but not the forward rank
    pb = H1["B"]
    ld = loader(hg, bad, [pb["gid"]], cfg=scfg(fanout=H1["fanout"]))
    with pytest.raises(LeakError) as e:
        next(iter(ld))
    problems = {p["problem"] for p in e.value.detail["findings"]}
    assert "time != gid (rank)" in problems and "target edge sampled" in problems


def test_causal_target_hit_raises(h1):
    g, hg, d = h1
    pa = H1["A"]
    tf = T.FlattenTransform(
        protocol="causal", seed_gids=np.array([pa["gid"]]), rev_gid=hg.rev_gid,
        guard_bound=np.array([pa["lookahead_bound"]]), first_rank=hg.first_rank,
    )  # fmt: skip
    with pytest.raises(LeakError, match="target edge sampled") as e:
        next(iter(_manual_loader(hg, d, [pa["gid"]], [pa["lookahead_bound"]], tf)))
    assert e.value.detail["guard"]["target_hits"] == 4


def test_lookahead_drops_every_target_copy(h1):
    g, hg, d = h1
    pa = H1["A"]
    ld = loader(hg, d, [pa["gid"]], protocol="lookahead", cfg=scfg(fanout=H1["fanout"]))
    (rb,), (fb,) = raw_and_flat(ld)
    times = torch.cat([rb[TO].time, rb[REV].time])
    gd = guard_dict(fb)
    assert gd["edges_checked"] == times.numel()
    assert gd["dropped_target_copies"] == sum(pa["lookahead_target_copies"].values())
    assert gd["target_hits"] == 0 and gd["violations"] == 0
    assert gd["future_edges"] == int((times >= hg.first_rank[pa["gid"]]).sum())
    assert gd["max_slack"] == int(times.max()) - pa["lookahead_bound"]
    assert pa["gid"] not in fb.gid_fwd.tolist() + fb.gid_rev.tolist()
    later = int((fb.gid_fwd > pa["gid"]).sum()) + int((fb.gid_rev > pa["gid"]).sum())
    assert later == sum(pa["lookahead_later_edges"].values())  # non-vacuous: the future is seen
    assert int(fb.sub_edges.sum()) == times.numel() - gd["dropped_target_copies"]
    # without the drop, the target copies are a leak
    tf = T.FlattenTransform(
        protocol="lookahead", seed_gids=np.array([pa["gid"]]), rev_gid=hg.rev_gid,
        guard_bound=np.array([pa["lookahead_bound"]]), first_rank=hg.first_rank,
        drop_target=False,
    )  # fmt: skip
    with pytest.raises(LeakError, match="target edge sampled"):
        tf(rb)


def test_lookahead_parallel_edges_stay(rnd):
    g, hg, d = rnd
    seeds = G.split_gids(hg, "val_late")
    ld = loader(hg, d, seeds, protocol="lookahead", cfg=scfg(fanout=(25, 10), eval_batch_size=600))
    flat = list(ld)
    total = G.label_times(hg, seeds, protocol="lookahead")
    assert (total == hg.bounds["val_last"]).all()
    dropped = kept_parallel = 0
    for fb in flat:
        tgt = fb.tgt_gid
        for gid, ei in ((fb.gid_fwd, fb.ei_fwd), (fb.gid_rev, fb.ei_rev)):
            sub = fb.node_sub[ei[1]]
            assert not bool((gid == tgt[sub]).any())
            same_pair = (torch.from_numpy(g.src)[gid] == torch.from_numpy(g.src)[tgt[sub]]) & (
                torch.from_numpy(g.dst)[gid] == torch.from_numpy(g.dst)[tgt[sub]]
            )
            kept_parallel += int(same_pair.sum())
        dropped += guard_dict(fb)["dropped_target_copies"]
    assert dropped > 0 and kept_parallel > 0


def test_faithful_transform_sampled_mask_and_snapshot_guard(rnd):
    from aml.models.gnn import faithful as F

    g, hg, _ = rnd
    last = hg.bounds["val_last"]
    snap = F.snapshot_hetero(hg, last)
    seeds = G.split_gids(hg, "val_early")
    ld = S.make_faithful_loader(
        snap, seeds, last_rank=last, fanout=[5, 3], batch_size=256, shuffle=False, seed=0,
        runtime=RUNTIME, device="cpu",
    )  # fmt: skip
    raw, flat = raw_and_flat(ld)
    n_sampled = 0
    for rb, fb in zip(raw, flat, strict=True):
        assert fb.node_sub is None and fb.sub_edges is None
        e_to = set(rb[TO].e_id.tolist())
        mask = [int(t) in e_to for t in fb.tgt_gid]  # Multi-GNN: target id among batch edges
        assert fb.sampled.tolist() == mask
        n_sampled += sum(mask)
        assert torch.equal(fb.gid_rev, rb[REV].e_id)  # every edge flipped: rev e_id == gid
        ego = torch.zeros(fb.n_nodes, dtype=torch.bool)
        ego[torch.unique(rb[TO].edge_label_index.reshape(-1))] = True
        assert torch.equal(fb.ego, ego)
        tg = set(fb.tgt_gid.tolist())
        hits = sum(int(x) in tg for x in fb.gid_fwd.tolist() + fb.gid_rev.tolist())
        gd = guard_dict(fb)
        assert gd["target_hits"] == hits and gd["violations"] == 0
        assert gd["edges_checked"] == fb.gid_fwd.numel() + fb.gid_rev.numel()
        assert gd["max_slack"] <= 0
    assert 0 < n_sampled < len(seeds)
    # review perf-3: the mark-array counts equal the sort-based torch.isin ones
    for rb, fb in zip(raw, flat, strict=True):
        tgt = fb.tgt_gid
        hits = int(torch.isin(fb.gid_fwd, tgt).sum()) + int(torch.isin(fb.gid_rev, tgt).sum())
        assert guard_dict(fb)["target_hits"] == hits
        assert torch.equal(fb.sampled, torch.isin(tgt, rb[TO].e_id))
    with pytest.raises(ValueError, match="not splittable"):
        T.split_flat_batch(flat[0], 1)
    tight = T.FlattenTransform(
        protocol="faithful", seed_gids=seeds, rev_gid=None,
        guard_bound=np.full(len(seeds), hg.bounds["val_early_first"] - 1), first_rank=None,
    )  # fmt: skip
    with pytest.raises(LeakError, match="snapshot"):
        tight(raw[0])


def test_transform_argument_checks(h1):
    _, hg, _ = h1
    kw = {"seed_gids": np.array([4]), "rev_gid": hg.rev_gid, "guard_bound": np.array([2])}
    with pytest.raises(ValueError):
        T.FlattenTransform(protocol="causal", first_rank=hg.first_rank, ego="tree", **kw)
    with pytest.raises(ValueError):
        T.FlattenTransform(protocol="causal", first_rank=hg.first_rank, drop_target=True, **kw)
    with pytest.raises(ValueError):
        T.FlattenTransform(protocol="lookahead", first_rank=None, drop_target=True, **kw)
    with pytest.raises(ValueError):
        T.FlattenTransform(
            protocol="faithful", seed_gids=np.array([1, 2]), rev_gid=None,
            guard_bound=np.array([3, 4]), first_rank=None,
        )  # fmt: skip


# --- the edge-cap split --------------------------------------------------------------------------


class RefGINE(nn.Module):
    """A small flat 2-layer GINE (eval mode) for split-vs-unsplit equality."""

    def __init__(self, d: int = 18, h: int = 8) -> None:
        super().__init__()
        torch.manual_seed(0)
        self.node = nn.Linear(1, h)
        self.ef, self.er = nn.Linear(d, h), nn.Linear(d, h)
        mk = lambda: nn.Sequential(nn.Linear(h, h), nn.ReLU(), nn.Linear(h, h))  # noqa: E731
        self.cf = nn.ModuleList([GINEConv(mk(), edge_dim=h) for _ in range(2)])
        self.cr = nn.ModuleList([GINEConv(mk(), edge_dim=h) for _ in range(2)])
        self.bn = nn.ModuleList([nn.BatchNorm1d(h) for _ in range(2)])
        self.out = nn.Linear(3 * h, 2)

    def forward(self, x, eif, eaf, eir, ear, ts, td, ta):
        h, ef, er = self.node(x), self.ef(eaf), self.er(ear)
        for i in range(2):
            agg = (self.cf[i](h, eif, ef) + self.cr[i](h, eir, er)) / 2
            h = (h + torch.relu(self.bn[i](agg))) / 2
        return self.out(torch.cat([h[ts], h[td], self.ef(ta)], 1))


def _greedy_groups(sizes: list[int], cap: int) -> list[int]:
    """Reference grouping: seeds per part (consecutive, greedy; a too-big subgraph alone)."""
    groups: list[int] = []
    cur = acc = 0
    for e in sizes:
        if cur and acc + e > cap:
            groups.append(cur)
            cur = acc = 0
        cur += 1
        acc += e
    return [*groups, cur]


def _check_parts(fb, parts, cap):
    assert [int(p.tgt_gid.numel()) for p in parts] == _greedy_groups(fb.sub_edges.tolist(), cap)
    assert sum(p.n_nodes for p in parts) == fb.n_nodes
    assert sum(p.gid_fwd.numel() for p in parts) == fb.gid_fwd.numel()
    assert sum(p.gid_rev.numel() for p in parts) == fb.gid_rev.numel()
    assert torch.equal(torch.cat([p.tgt_gid for p in parts]), fb.tgt_gid)
    assert torch.equal(torch.cat([p.seed_pos for p in parts]), fb.seed_pos)
    assert torch.equal(torch.cat([p.sub_edges for p in parts]), fb.sub_edges)
    a = 0
    for k, p in enumerate(parts):
        b = a + int(p.tgt_gid.numel())
        assert int(p.sub_edges.sum()) <= cap or p.tgt_gid.numel() == 1
        assert int(p.sub_edges.sum()) == p.gid_fwd.numel() + p.gid_rev.numel()
        keep = ((fb.node_sub >= a) & (fb.node_sub < b)).nonzero().reshape(-1)  # old index
        assert torch.equal(p.ego, fb.ego[keep])
        assert torch.equal(p.node_sub, fb.node_sub[keep] - a)
        for ei_p, gid_p, ei, gid in (
            (p.ei_fwd, p.gid_fwd, fb.ei_fwd, fb.gid_fwd),
            (p.ei_rev, p.gid_rev, fb.ei_rev, fb.gid_rev),
        ):
            sel = (fb.node_sub[ei[1]] >= a) & (fb.node_sub[ei[1]] < b)
            assert torch.equal(keep[ei_p], ei[:, sel])  # relabelled back == the original edges
            assert torch.equal(gid_p, gid[sel])
        assert torch.equal(keep[p.tgt_src], fb.tgt_src[a:b])
        assert torch.equal(keep[p.tgt_dst], fb.tgt_dst[a:b])
        assert torch.equal(p.guard, fb.guard if k == 0 else T.zero_guard())
        a = b
    total = None
    for p in parts:
        total = add_guard(total, p.guard.tolist())
    assert total == add_guard(None, fb.guard.tolist())


def test_split_flat_batch_conserves_and_relabels(rnd):
    g, hg, d = rnd
    seeds = G.split_gids(hg, "test")[:500]
    fb = next(iter(loader(hg, d, seeds, cfg=scfg(fanout=(25, 10), eval_batch_size=500))))
    total = fb.gid_fwd.numel() + fb.gid_rev.numel()
    same = T.split_flat_batch(fb, total)
    assert len(same) == 1 and same[0] is fb
    for cap in (total // 2, total // 7, int(fb.sub_edges.max()), 1):
        parts = T.split_flat_batch(fb, cap)
        assert len(parts) > 1
        _check_parts(fb, parts, cap)
    singles = T.split_flat_batch(fb, 1)  # every non-empty subgraph alone
    assert all(int(p.sub_edges.sum()) <= 1 or p.tgt_gid.numel() == 1 for p in singles)
    nested = T.split_flat_batch(T.split_flat_batch(fb, total // 2)[0], total // 6)
    assert len(nested) > 1  # parts can be split again (OOM halving)
    with pytest.raises(ValueError):
        T.split_flat_batch(fb, 0)


def test_eval_logits_split_equal_unsplit(rnd):
    g, hg, d = rnd
    seeds = G.split_gids(hg, "test")
    fb = next(iter(loader(hg, d, seeds, cfg=scfg(fanout=(25, 10), eval_batch_size=400))))
    ea = torch.from_numpy(hg.ea)
    models = [RefGINE().eval()]
    try:
        from aml.models.gnn.model import MultiGINe

        pre = fx.fixture_preprocess()
        torch.manual_seed(1)
        models.append(
            MultiGINe(
                num_idx=pre["num_idx"], cat_idx=pre["cat_idx"], cat_sizes=pre["cat_sizes"],
                hidden=8,
            ).eval()
        )  # fmt: skip
    except NotImplementedError:
        pass
    total = fb.gid_fwd.numel() + fb.gid_rev.numel()
    for model in models:
        with torch.no_grad():
            whole = model(*T.to_model_inputs(fb, ea, "cpu"))
            for cap in (total // 3, total // 11):
                parts = T.split_flat_batch(fb, cap)
                split = torch.cat([model(*T.to_model_inputs(p, ea, "cpu")) for p in parts])
                torch.testing.assert_close(split, whole, rtol=1e-6, atol=1e-6)


# --- caps ----------------------------------------------------------------------------------------


def test_resolve_edge_caps(monkeypatch):
    assert T.resolve_edge_caps(scfg(), "cpu") == (None, None)
    cfg = scfg()
    cfg["max_edges_per_step"] = 1000
    assert T.resolve_edge_caps(cfg, "cpu") == (1000, 4000)
    cfg["max_edges_per_eval_step"] = 1500
    assert T.resolve_edge_caps(cfg, "cuda") == (1000, 1500)

    class Props:
        total_memory = 24 * 2**30

    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _i: Props())
    train = int(np.floor(0.6 * 24 * 2**30 / 5900))
    assert T.resolve_edge_caps(scfg(), "cuda") == (train, 4 * train)
    assert T.resolve_edge_caps(scfg(), "cuda", memory_fraction=0.5, bytes_per_edge=1000) == (
        int(0.5 * 24 * 2**30 / 1000),
        4 * int(0.5 * 24 * 2**30 / 1000),
    )
    for bad in (0, -5, 1.5, True):
        cfg = scfg()
        cfg["max_edges_per_step"] = bad
        with pytest.raises(ValueError):
            T.resolve_edge_caps(cfg, "cpu")


# --- the eval cache ------------------------------------------------------------------------------


def _same(a: T.FlatBatch, b: T.FlatBatch) -> bool:
    for f in a._fields:
        x, y = getattr(a, f), getattr(b, f)
        if isinstance(x, torch.Tensor):
            if x.dtype != y.dtype or not torch.equal(x, y):
                return False
        elif x != y:
            return False
    return True


def test_eval_cache_replays_int64_with_zero_guards(rnd):
    g, hg, d = rnd
    seeds = G.split_gids(hg, "val_early")
    ld = loader(hg, d, seeds, cfg=scfg(fanout=(25, 10), eval_batch_size=128))
    ref = list(ld)
    cap = max(int(fb.sub_edges.sum()) for fb in ref) // 2
    cache = T.EvalCache(max_gb=0.5)
    first = list(cache.batches(ld, cap))
    assert cache.cached and cache.nbytes > 0
    expect = [p for fb in ref for p in T.split_flat_batch(fb, cap)]
    assert len(first) == len(expect) > len(ref)
    assert all(_same(a, b) for a, b in zip(first, expect, strict=True))
    second = list(cache.batches(ld, cap))
    for a, b in zip(second, first, strict=True):
        assert _same(a._replace(guard=b.guard), b) and torch.equal(a.guard, T.zero_guard())
    g1 = g2 = None
    for p in first:
        g1 = add_guard(g1, p.guard.tolist())
    for p in first + second:
        g2 = add_guard(g2, p.guard.tolist())
    assert g1 == g2 and g1["edges_checked"] == sum(
        p.gid_fwd.numel() + p.gid_rev.numel() for p in first
    )
    st = cache.stats()
    assert st["cached"] and st["streamed_passes"] == 1 and st["cached_passes"] == 1
    assert st["batches"] == len(first) and st["edges"] == g1["edges_checked"]
    assert st["max_edges_per_batch"] == max(int(fb.sub_edges.sum()) for fb in ref)
    assert st["max_edges_per_part"] <= cap or st["max_edges_per_part"] == max(
        int(fb.sub_edges.max()) for fb in ref
    )
    with pytest.raises(ValueError, match="max_edges"):
        cache.batches(ld, None)


def test_eval_cache_overflow_and_abandoned_pass(rnd):
    g, hg, d = rnd
    seeds = G.split_gids(hg, "val_early")
    ld = loader(hg, d, seeds, cfg=scfg(fanout=(25, 10), eval_batch_size=128))
    tiny = T.EvalCache(max_gb=1e-6)  # 1,000 bytes: overflows on the first batch
    a = list(tiny.batches(ld, None))
    b = list(tiny.batches(ld, None))
    assert not tiny.cached and tiny.stats()["overflow"]
    assert tiny.stats()["streamed_passes"] == 2 and tiny.stats()["cached_passes"] == 0
    assert all(_same(x, y) for x, y in zip(a, b, strict=True))  # `last` is deterministic
    assert not bool(torch.equal(b[0].guard, T.zero_guard()))  # streamed passes keep guards
    early = T.EvalCache(max_gb=0.5)
    it = early.batches(ld, None)
    next(it)
    it.close()  # an abandoned pass is not cached
    assert not early.cached
    assert len(list(early.batches(ld, None))) == len(a) and early.cached


# --- the epoch sampler ---------------------------------------------------------------------------


def test_epoch_positions_is_a_pure_function():
    pos = np.array([3, 17, 40])
    neg = np.setdiff1d(np.arange(1000), pos)
    p = S.epoch_positions(pos, neg, 0.1, seed=2, epoch=5)
    assert p.dtype == np.int64 and len(p) == 3 + round(0.1 * 997)
    assert set(pos) <= set(p.tolist()) and len(np.unique(p)) == len(p)
    assert set(p.tolist()) - set(pos) <= set(neg.tolist())
    assert np.array_equal(p, S.epoch_positions(pos, neg, 0.1, seed=2, epoch=5))
    assert not np.array_equal(p, S.epoch_positions(pos, neg, 0.1, seed=2, epoch=6))
    assert not np.array_equal(p, S.epoch_positions(pos, neg, 0.1, seed=3, epoch=5))
    assert len(S.epoch_positions(pos, neg, 1.0, 0, 0)) == 1000
    s = S.EpochSubsetSampler(pos, neg, 0.1, seed=2)
    assert len(s) == len(p)
    s.set_epoch(5)
    assert list(s) == p.tolist() and len(s) == len(p)
    s.set_epoch(0)
    assert list(s) == S.epoch_positions(pos, neg, 0.1, 2, 0).tolist()
    with pytest.raises(ValueError):
        S.EpochSubsetSampler(pos, np.r_[neg, 3], 0.1, 0)
    with pytest.raises(ValueError):
        S.epoch_positions(pos, neg, 0.0, 0, 0)


def test_train_loader_follows_the_epoch_subset(rnd):
    g, hg, d = rnd
    seeds = G.split_gids(hg, "train")
    y = G.labels_for(hg, seeds)
    sampler = S.EpochSubsetSampler(np.flatnonzero(y == 1), np.flatnonzero(y == 0), 0.2, seed=7)
    ld = loader(hg, d, seeds, role="train", cfg=scfg(batch_size=100), sampler=sampler)
    for epoch in (0, 1):
        sampler.set_epoch(epoch)
        got = torch.cat([fb.seed_pos for fb in ld]).tolist()
        assert (
            got
            == S.epoch_positions(
                np.flatnonzero(y == 1), np.flatnonzero(y == 0), 0.2, 7, epoch
            ).tolist()
        )
        assert (y[got] == 1).sum() == (y == 1).sum()
    with pytest.raises(ValueError, match="EpochSubsetSampler"):
        loader(hg, d, seeds, role="train")
    with pytest.raises(ValueError, match="sampler must be None"):
        loader(hg, d, seeds, sampler=sampler)
    with pytest.raises(ValueError):
        loader(hg, d, seeds, protocol="faithful")


def test_workers_give_identical_batches(rnd):
    g, hg, d = rnd
    seeds = G.split_gids(hg, "train")
    y = G.labels_for(hg, seeds)
    runs = []
    for workers in (0, 2):
        sampler = S.EpochSubsetSampler(np.flatnonzero(y == 1), np.flatnonzero(y == 0), 0.3, 1)
        ld = loader(
            hg, d, seeds, role="train", cfg=scfg(batch_size=128, fanout=(25, 10)),
            sampler=sampler, runtime={"num_workers": workers, "loader_timeout_s": 60},
        )  # fmt: skip
        out = []
        for epoch in (0, 1):
            sampler.set_epoch(epoch)
            out.append(list(ld))
        runs.append(out)
        if workers:
            # persistent workers live between epochs; close_loader stops them at once (left to
            # the cyclic GC, a PyG loader's shutdown waits DataLoader's 5 s join per worker)
            procs = list(ld._iterator._workers)
            assert procs and all(p.is_alive() for p in procs)
            t0 = time.perf_counter()
            S.close_loader(ld)
            S.close_loader(ld)  # idempotent
            assert time.perf_counter() - t0 < 4.0
            assert ld._iterator is None and not any(p.is_alive() for p in procs)
    for e0, e1 in zip(*runs, strict=True):
        assert len(e0) == len(e1) and all(_same(a, b) for a, b in zip(e0, e1, strict=True))


def test_zero_guard():
    z = T.zero_guard()
    assert z.dtype == torch.int64 and z.tolist() == [0, 0, 0, NO_SLACK, 0, 0]
