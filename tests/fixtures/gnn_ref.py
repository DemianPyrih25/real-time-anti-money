"""Independent reference for the causal GNN sampler (M3 spec §5), written only from the spec.

- `RefSampler`: per seed and tree, the exact hop-1 / hop-2 `last` edge multisets under the
  INCLUSIVE bound (an edge whose time equals the bound is eligible, F2). Every hop is bounded by
  the seed's label time; `last` takes the k latest eligible in-edges per frontier node and edge
  type; a node enters a tree once and is expanded once (per-tree dedup, F3/F4). `to` in-edges of
  a node are the transactions into it; `rev_to` in-edges are the non-self-loop transactions out
  of it, reversed, with the forward rank as time (§4.2).
- `RefSubgraph`: a target's subgraph = the tree of its src account + the tree of its dst account
  (the two disjoint trees PyG merges with `batch % num_pos`, F4), with account-level ego flags
  (§5.4) and the target's own copies (look-ahead, F9).
- `ref_label_times`: the per-protocol bounds of §5.2, recomputed from first principles.
- `Observed`: the neutral per-subgraph view of a sampled batch (a raw PyG LinkNeighborLoader batch
  or a FlatBatch), `asof_problems` (the checker every leakage suite runs on it),
  `ref_problems` (exact comparison with the reference) and the pins (§14.1, inclusivity).

numpy only at import; torch / PyG are imported inside the functions that need them.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

TO_NAME, REV_NAME = "to", "rev_to"
EDGE_TYPES = (TO_NAME, REV_NAME)
SPLIT_NAMES = ("train", "val_early", "val_late", "test")  # aml.data.split.SPLITS order
NODE_TYPE = "acct"
TO_TYPE = (NODE_TYPE, TO_NAME, NODE_TYPE)
REV_TYPE = (NODE_TYPE, REV_NAME, NODE_TYPE)


# --- the reference sampler ----------------------------------------------------------------------


@dataclass(frozen=True)
class RefEdge:
    hop: int  # 1 or 2
    gid: int  # forward transaction id (= rank = time); a rev_to copy carries its forward gid
    src: int  # message source account
    dst: int  # message target account (the frontier node it was sampled for)


@dataclass(frozen=True)
class RefTree:
    root: int
    bound: int
    nodes: tuple[int, ...]  # accounts in discovery order, root first, each at most once
    edges: Mapping[str, tuple[RefEdge, ...]]  # edge type -> sampled edges, hop order


@dataclass(frozen=True)
class RefSubgraph:
    gid: int  # the target
    bound: int  # its label time
    u: int  # target src account
    v: int  # target dst account
    trees: tuple[RefTree, RefTree]  # (tree of u, tree of v)

    def edges(self, et: str, *, drop_target: bool = False) -> list[RefEdge]:
        out = [e for t in self.trees for e in t.edges[et]]
        return [e for e in out if e.gid != self.gid] if drop_target else out

    def gids(self, et: str, *, drop_target: bool = False) -> list[int]:
        """Sorted multiset of the sampled gids of one edge type."""
        return sorted(e.gid for e in self.edges(et, drop_target=drop_target))

    def all_gids(self, *, drop_target: bool = False) -> np.ndarray:
        return np.array(
            [e.gid for et in EDGE_TYPES for e in self.edges(et, drop_target=drop_target)],
            dtype=np.int64,
        )

    @property
    def n_nodes(self) -> int:
        return sum(len(t.nodes) for t in self.trees)

    def node_counts(self) -> Counter:
        """Multiset of accounts over both trees (u's tree may hold a copy of v and vice versa)."""
        return Counter(a for t in self.trees for a in t.nodes)

    @property
    def ego_count(self) -> int:
        """Nodes flagged by the account-level ego rule: every copy of u and of v (§5.4)."""
        return sum(1 for t in self.trees for a in t.nodes if a in (self.u, self.v))

    def target_copies(self, et: str) -> int:
        return sum(1 for e in self.edges(et) if e.gid == self.gid)

    def n_edges(self, *, drop_target: bool = False) -> int:
        return sum(len(self.edges(et, drop_target=drop_target)) for et in EDGE_TYPES)

    def future_count(self, first_rank: int, *, drop_target: bool = False) -> int:
        """Sampled edges with rank >= the first rank of the target's minute (§5.3 diagnostic)."""
        return int((self.all_gids(drop_target=drop_target) >= first_rank).sum())

    def max_slack(self, *, drop_target: bool = False) -> int | None:
        g = self.all_gids(drop_target=drop_target)
        return int(g.max()) - self.bound if len(g) else None


def _in_lists(target: np.ndarray, source: np.ndarray, gid: np.ndarray, n_nodes: int):
    """Per node, its in-edges sorted by time (= gid): (indptr, sources, gids)."""
    order = np.lexsort((gid, target))
    counts = np.bincount(target, minlength=n_nodes)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    return indptr, source[order].astype(np.int64), gid[order].astype(np.int64)


class RefSampler:
    """The exact `last` temporal sampler of §5 on a rank-ordered transaction graph.

    fanout: per hop, the number of in-edges taken per frontier node and edge type (-1 = all).
    """

    def __init__(self, src: Sequence[int], dst: Sequence[int], n_nodes: int, fanout: Sequence[int]):
        src = np.asarray(src, dtype=np.int64)
        dst = np.asarray(dst, dtype=np.int64)
        if src.shape != dst.shape or src.ndim != 1:
            raise ValueError("src and dst must be 1-d arrays of the same length")
        self.src, self.dst, self.n_nodes = src, dst, int(n_nodes)
        self.fanout = tuple(int(k) for k in fanout)
        gid = np.arange(len(src), dtype=np.int64)
        nsl = src != dst  # self-loops get no reverse copy
        self._in = {
            TO_NAME: _in_lists(dst, src, gid, self.n_nodes),  # u -> w is an in-edge of w
            REV_NAME: _in_lists(src[nsl], dst[nsl], gid[nsl], self.n_nodes),  # w -> x: x -> w
        }

    def tree(self, root: int, bound: int) -> RefTree:
        root, bound = int(root), int(bound)
        nodes, seen, frontier = [root], {root}, [root]
        edges: dict[str, list[RefEdge]] = {et: [] for et in EDGE_TYPES}
        for hop, k in enumerate(self.fanout, start=1):
            new: list[int] = []
            for w in frontier:
                for et in EDGE_TYPES:
                    indptr, sources, gids = self._in[et]
                    lo, hi = int(indptr[w]), int(indptr[w + 1])
                    # eligible: time <= bound (inclusive); `last`: the k latest of them
                    cut = lo + int(np.searchsorted(gids[lo:hi], bound, side="right"))
                    first = lo if k < 0 else max(lo, cut - k)
                    for j in range(first, cut):
                        s = int(sources[j])
                        edges[et].append(RefEdge(hop, int(gids[j]), s, w))
                        if s not in seen:
                            seen.add(s)
                            nodes.append(s)
                            new.append(s)
            frontier = new
        return RefTree(root, bound, tuple(nodes), {et: tuple(v) for et, v in edges.items()})

    def subgraph(self, gid: int, bound: int) -> RefSubgraph:
        gid = int(gid)
        u, v = int(self.src[gid]), int(self.dst[gid])
        return RefSubgraph(gid, int(bound), u, v, (self.tree(u, bound), self.tree(v, bound)))


# --- bounds (§5.2) ----------------------------------------------------------------------------


def ref_first_rank(minute: Sequence[int]) -> np.ndarray:
    """First rank of each gid's minute, by a plain loop over the minute column."""
    first: dict[int, int] = {}
    vals = np.asarray(minute).tolist()
    for r, m in enumerate(vals):
        first.setdefault(m, r)
    return np.array([first[m] for m in vals], dtype=np.int64)


def ref_label_times(
    gids: Iterable[int],
    *,
    minute: Sequence[int],
    split_code: Sequence[int],
    bounds: Mapping[str, int],
    protocol: str,
    test_bound: str = "end",
) -> np.ndarray:
    """§5.2: causal / PNA = first rank of the minute - 1 (every split); look-ahead = the split's
    bound (train: train_last; val_early, val_late: val_last; test: data_last for `end`,
    max(d10_last, causal bound) for `d10`)."""
    first = ref_first_rank(minute)
    codes = np.asarray(split_code)
    out = []
    for g in gids:
        g = int(g)
        causal = int(first[g]) - 1
        if protocol in ("causal", "pna"):
            out.append(causal)
            continue
        if protocol != "lookahead":
            raise ValueError(f"no temporal bound for protocol {protocol!r}")
        split = SPLIT_NAMES[int(codes[g])]
        if split == "train":
            out.append(int(bounds["train_last"]))
        elif split in ("val_early", "val_late"):
            out.append(int(bounds["val_last"]))
        elif test_bound == "end":
            out.append(int(bounds["data_last"]))
        elif test_bound == "d10":
            out.append(max(int(bounds["d10_last"]), causal))
        else:
            raise ValueError(f"unknown test bound {test_bound!r}")
    return np.array(out, dtype=np.int64)


# --- the observed view of a sampled batch --------------------------------------------------------


@dataclass
class Observed:
    """One subgraph of a sampled batch, in global terms."""

    gid: int  # the target gid
    seed_pos: int  # its position in the loader's seed array
    accounts: np.ndarray  # (n,) account of each node; -1 = not determined (isolated copy)
    ego: np.ndarray  # (n,) bool
    roots: tuple[int, int]  # node positions (in this subgraph) of the target's src / dst roots
    to_gids: np.ndarray  # sampled `to` gids (with multiplicity)
    rev_gids: np.ndarray  # sampled `rev_to` copies as their forward gids
    problems: list[str] = field(default_factory=list)  # structural problems seen on extraction

    @property
    def n_nodes(self) -> int:
        return len(self.accounts)

    def all_gids(self) -> np.ndarray:
        return np.concatenate([self.to_gids, self.rev_gids]).astype(np.int64)

    def gids(self, et: str) -> list[int]:
        return sorted((self.to_gids if et == TO_NAME else self.rev_gids).tolist())


def _per_subgraph(
    *,
    node_sub: np.ndarray,
    accounts: np.ndarray,
    ego: np.ndarray,
    tgt_src: np.ndarray,
    tgt_dst: np.ndarray,
    tgt_gid: np.ndarray,
    seed_pos: np.ndarray,
    ei_to: np.ndarray,
    gid_to: np.ndarray,
    ei_rev: np.ndarray,
    gid_rev: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
) -> list[Observed]:
    """Split a flat batch into subgraphs and check its structure against the transactions:
    every node's account (given, or inferred from the edges and roots touching it) must agree
    with the gids of all those edges, and no edge may cross subgraphs."""
    n, n_sub = len(node_sub), len(tgt_gid)
    problems: list[str] = []
    acc = accounts.astype(np.int64).copy()
    claims = (
        (ei_to[0], src[gid_to], "`to` sources"),
        (ei_to[1], dst[gid_to], "`to` targets"),
        (ei_rev[0], dst[gid_rev], "`rev_to` sources (forward dst)"),
        (ei_rev[1], src[gid_rev], "`rev_to` targets (forward src)"),
        (tgt_src, src[tgt_gid], "target src roots"),
        (tgt_dst, dst[tgt_gid], "target dst roots"),
    )
    for nodes, want, _ in claims:
        unset = acc[nodes] < 0
        acc[nodes[unset]] = want[unset]
    for nodes, want, what in claims:
        bad = acc[nodes] != want
        if bad.any():
            problems.append(f"{int(bad.sum())} {what} disagree with the node accounts")
    for name, ei in (("to", ei_to), ("rev_to", ei_rev)):
        cross = node_sub[ei[0]] != node_sub[ei[1]]
        if cross.any():
            problems.append(f"{int(cross.sum())} `{name}` edges cross subgraphs")
    if n and (node_sub.min() < 0 or node_sub.max() >= n_sub):
        raise AssertionError(f"subgraph ids outside [0, {n_sub}): {problems}")

    def groups(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(keys, kind="stable")
        return order, np.searchsorted(keys[order], np.arange(n_sub + 1))

    node_order, node_cut = groups(node_sub)
    pos_in_sub = np.empty(n, dtype=np.int64)
    for b in range(n_sub):
        idx = node_order[node_cut[b] : node_cut[b + 1]]
        pos_in_sub[idx] = np.arange(len(idx))
    to_order, to_cut = groups(node_sub[ei_to[1]])
    rev_order, rev_cut = groups(node_sub[ei_rev[1]])
    out = []
    for b in range(n_sub):
        idx = node_order[node_cut[b] : node_cut[b + 1]]
        sub_problems = list(problems)
        roots = []
        for r, side in ((int(tgt_src[b]), "src"), (int(tgt_dst[b]), "dst")):
            if node_sub[r] != b:
                sub_problems.append(f"the {side} root lies in subgraph {int(node_sub[r])}")
            roots.append(int(pos_in_sub[r]))
        out.append(
            Observed(
                gid=int(tgt_gid[b]),
                seed_pos=int(seed_pos[b]),
                accounts=acc[idx],
                ego=ego[idx].astype(bool),
                roots=(roots[0], roots[1]),
                to_gids=gid_to[to_order[to_cut[b] : to_cut[b + 1]]].astype(np.int64),
                rev_gids=gid_rev[rev_order[rev_cut[b] : rev_cut[b + 1]]].astype(np.int64),
                problems=sub_problems,
            )
        )
    return out


def account_ego(n_id: np.ndarray, batch: np.ndarray, eli: np.ndarray) -> np.ndarray:
    """§5.4: node i is flagged iff its account is its subgraph target's src or dst account."""
    u_acc, v_acc = n_id[eli[0]], n_id[eli[1]]
    return (n_id == u_acc[batch]) | (n_id == v_acc[batch])


def observed_from_pyg(batch: Any, seed_gids: np.ndarray, rev_gid: np.ndarray, src, dst):
    """Observed subgraphs of a raw PyG LinkNeighborLoader batch over `ref_hetero` data (the `to`
    store in rank order, so e_id == gid; `rev_to` e_id -> forward gid through rev_gid). Ego flags
    are computed with the account-level rule."""
    n_id = batch[NODE_TYPE].n_id.numpy()
    bvec = batch[NODE_TYPE].batch.numpy()
    eli = batch[TO_TYPE].edge_label_index.numpy()
    input_id = batch[TO_TYPE].input_id.numpy()
    src, dst = np.asarray(src, np.int64), np.asarray(dst, np.int64)
    obs = _per_subgraph(
        node_sub=bvec,
        accounts=n_id,
        ego=account_ego(n_id, bvec, eli),
        tgt_src=eli[0],
        tgt_dst=eli[1],
        tgt_gid=np.asarray(seed_gids, np.int64)[input_id],
        seed_pos=input_id,
        ei_to=batch[TO_TYPE].edge_index.numpy(),
        gid_to=batch[TO_TYPE].e_id.numpy(),
        ei_rev=batch[REV_TYPE].edge_index.numpy(),
        gid_rev=np.asarray(rev_gid, np.int64)[batch[REV_TYPE].e_id.numpy()],
        src=src,
        dst=dst,
    )
    # a raw batch also carries the sampled times: they must equal the gids (time = rank)
    for et in (TO_TYPE, REV_TYPE):
        e_id = batch[et].e_id.numpy()
        gid = e_id if et == TO_TYPE else np.asarray(rev_gid, np.int64)[e_id]
        if not np.array_equal(batch[et].time.numpy(), gid):
            for o in obs:
                o.problems.append(f"`{et[1]}` time != forward gid")
    return obs


def observed_from_flat(fb: Any, src, dst) -> list[Observed]:
    """Observed subgraphs of a FlatBatch (§5.6). Accounts are inferred from the gids of the edges
    and roots, and every inference must agree (a check of the local indices and of rev_gid)."""
    src, dst = np.asarray(src, np.int64), np.asarray(dst, np.int64)
    n = int(fb.n_nodes)
    if fb.node_sub is None:
        raise ValueError("a temporal FlatBatch carries node_sub")
    return _per_subgraph(
        node_sub=fb.node_sub.numpy(),
        accounts=np.full(n, -1, dtype=np.int64),
        ego=fb.ego.numpy(),
        tgt_src=fb.tgt_src.numpy(),
        tgt_dst=fb.tgt_dst.numpy(),
        tgt_gid=fb.tgt_gid.numpy(),
        seed_pos=fb.seed_pos.numpy(),
        ei_to=fb.ei_fwd.numpy(),
        gid_to=fb.gid_fwd.numpy(),
        ei_rev=fb.ei_rev.numpy(),
        gid_rev=fb.gid_rev.numpy(),
        src=src,
        dst=dst,
    )


# --- checkers -----------------------------------------------------------------------------------


def asof_problems(obs: Observed, *, bound: int, minute: np.ndarray, causal: bool) -> list[str]:
    """The leakage checker: every sampled edge (both types, every hop) has time <= bound; the
    target is absent; under the causal rule no sampled edge shares or follows the target's
    minute (either rank side)."""
    out = list(obs.problems)
    g = obs.all_gids()
    if len(g) and int(g.max()) > bound:
        out.append(f"target {obs.gid}: {int((g > bound).sum())} edges later than bound {bound}")
    if (g == obs.gid).any():
        out.append(f"target {obs.gid}: {int((g == obs.gid).sum())} copies of the target")
    if causal and len(g):
        m = int(minute[obs.gid])
        late = g[np.asarray(minute)[g] >= m]
        if len(late):
            out.append(f"target {obs.gid}: same-or-later-minute edges {sorted(set(late.tolist()))}")
    return out


def ref_problems(obs: Observed, ref: RefSubgraph, *, drop_target: bool = False) -> list[str]:
    """Exact comparison with the reference: both edge multisets, the node count (per-tree dedup),
    the node accounts, the roots and the account-level ego flags."""
    out = list(obs.problems)
    tag = f"target {ref.gid}"
    if obs.gid != ref.gid:
        out.append(f"{tag}: observed target {obs.gid}")
    for et in EDGE_TYPES:
        got, want = obs.gids(et), ref.gids(et, drop_target=drop_target)
        if got != want:
            out.append(f"{tag}: `{et}` gids {got} != reference {want}")
    if obs.n_nodes != ref.n_nodes:
        out.append(f"{tag}: {obs.n_nodes} nodes != reference {ref.n_nodes}")
    known = obs.accounts[obs.accounts >= 0]
    got_counts, want_counts = Counter(known.tolist()), ref.node_counts()
    if got_counts - want_counts:
        out.append(f"{tag}: accounts {dict(got_counts)} not within reference {dict(want_counts)}")
    if len(known) == obs.n_nodes and got_counts != want_counts:
        out.append(f"{tag}: accounts {dict(got_counts)} != reference {dict(want_counts)}")
    if obs.roots and tuple(int(obs.accounts[r]) for r in obs.roots) != (ref.u, ref.v):
        out.append(f"{tag}: roots are accounts {[int(obs.accounts[r]) for r in obs.roots]}")
    if int(obs.ego.sum()) != ref.ego_count:
        out.append(f"{tag}: {int(obs.ego.sum())} ego nodes != reference {ref.ego_count}")
    want_ego = np.isin(obs.accounts, [ref.u, ref.v])
    sure = obs.accounts >= 0
    if not np.array_equal(obs.ego[sure], want_ego[sure]):
        out.append(f"{tag}: ego flags are not the account-level rule")
    return out


def h1_pin_problems(obs: Observed, pinned: Mapping[str, Any]) -> list[str]:
    """§14.1's verified facts for hand graph h1, target A (bound 2, fanout [5, 5]), checked
    order-free: nodes, account-level ego (4 flags; root-only would give 2), edge multisets, max
    time per type, absent gids."""
    a = pinned["A"]
    out = list(obs.problems)
    if obs.gid != a["gid"]:
        out.append(f"observed target {obs.gid} != {a['gid']}")
    if Counter(obs.accounts.tolist()) != Counter(a["n_id"]):
        out.append(f"node accounts {sorted(obs.accounts.tolist())} != {sorted(a['n_id'])}")
    if int(obs.ego.sum()) != sum(a["account_ego"]):
        out.append(f"{int(obs.ego.sum())} ego flags != pinned {sum(a['account_ego'])}")
    u, v = obs.accounts[obs.roots[0]], obs.accounts[obs.roots[1]]
    if not np.array_equal(obs.ego, np.isin(obs.accounts, [u, v])):
        out.append("ego flags are not every copy of u and v")
    if obs.gids(TO_NAME) != a["to_gids"]:
        out.append(f"`to` gids {obs.gids(TO_NAME)} != {a['to_gids']}")
    if obs.gids(REV_NAME) != a["rev_to_gids"]:
        out.append(f"`rev_to` gids {obs.gids(REV_NAME)} != {a['rev_to_gids']}")
    for et, arr in ((TO_NAME, obs.to_gids), (REV_NAME, obs.rev_gids)):
        top = int(arr.max()) if len(arr) else None
        if top != a["max_time"][et]:
            out.append(f"max `{et}` time {top} != {a['max_time'][et]}")
    present = set(obs.all_gids().tolist())
    if present & set(a["absent_gids"]):
        out.append(f"gids {sorted(present & set(a['absent_gids']))} must be absent")
    return out


Sampler = Callable[[Sequence[int], Sequence[int]], list[Observed]]

# Inclusivity pins: (graph builder name, target gid, an eligible edge e adjacent to the target).
# At label time rank(e) the filter must take e (inclusive); at rank(e) - 1 it must not.
INCLUSIVITY_CASES = (
    ("h1", 4, 2),  # target A u->v; e = u->x (rank 2): rev hop 1 at u, `to` hop 2 at x
    ("tied", 11, 10),  # target T2 a->d; e = c->a (rank 10): `to` hop 1 at a
)


def inclusivity_problems(sample: Sampler, target: int, edge: int) -> list[str]:
    """`sample(seed_gids, label_times)` -> observed subgraphs. Pin: label time == rank(edge)
    includes edge, rank(edge) - 1 excludes it."""
    out = []
    (at,) = sample([target], [edge])
    if edge not in at.all_gids().tolist():
        out.append(f"label time {edge} == rank of edge {edge}: edge not sampled (filter not <=)")
    (below,) = sample([target], [edge - 1])
    if edge in below.all_gids().tolist():
        out.append(f"label time {edge - 1}: edge {edge} sampled although later than the bound")
    return out


# --- raw PyG helpers (independent of aml.models.gnn) ------------------------------------------


def ref_hetero(src: Sequence[int], dst: Sequence[int], n_nodes: int):
    """§4.2's HeteroData, built here: `to` = all edges in rank order, `rev_to` = non-self-loop
    edges reversed in rank order, time = the forward rank on both; only edge_index and time."""
    import torch
    from torch_geometric.data import HeteroData

    src_t = torch.as_tensor(np.asarray(src, np.int64))
    dst_t = torch.as_tensor(np.asarray(dst, np.int64))
    rank = torch.arange(len(src_t), dtype=torch.int64)
    nsl = src_t != dst_t
    data = HeteroData()
    data[NODE_TYPE].num_nodes = int(n_nodes)
    data[TO_TYPE].edge_index = torch.stack([src_t, dst_t])
    data[TO_TYPE].time = rank
    data[REV_TYPE].edge_index = torch.stack([dst_t[nsl], src_t[nsl]])
    data[REV_TYPE].time = rank[nsl]
    return data


def raw_loader(
    data: Any,
    seed_gids: Sequence[int],
    label_times: Sequence[int],
    fanout: Sequence[int],
    *,
    batch_size: int,
    transform: Any = None,
    **kw: Any,
):
    """§5.1's LinkNeighborLoader on `data` (no transform unless given)."""
    import torch
    from torch_geometric.loader import LinkNeighborLoader

    seeds = torch.as_tensor(np.asarray(seed_gids, np.int64))
    return LinkNeighborLoader(
        data,
        num_neighbors={TO_TYPE: list(fanout), REV_TYPE: list(fanout)},
        edge_label_index=(TO_TYPE, data[TO_TYPE].edge_index[:, seeds]),
        edge_label_time=torch.as_tensor(np.asarray(label_times, np.int64)),
        time_attr="time",
        temporal_strategy="last",
        disjoint=True,
        batch_size=batch_size,
        shuffle=False,
        transform=transform,
        **kw,
    )


def raw_observed(
    src, dst, n_nodes: int, seed_gids, label_times, fanout, *, batch_size: int = 4096
) -> list[Observed]:
    """Every seed's observed subgraph through raw PyG (ordered by seed position)."""
    data = ref_hetero(src, dst, n_nodes)
    rev_gid = np.flatnonzero(np.asarray(src) != np.asarray(dst)).astype(np.int64)
    out: list[Observed] = []
    seeds = np.asarray(seed_gids, np.int64)
    for batch in raw_loader(data, seeds, label_times, fanout, batch_size=batch_size):
        out.extend(observed_from_pyg(batch, seeds, rev_gid, src, dst))
    return sorted(out, key=lambda o: o.seed_pos)


# --- adapters for aml.models.gnn (skip while a part is a stub) --------------------------------

GNN_OWNERS = {
    "graph": "B",
    "sampler": "B",
    "transforms": "B",
    "faithful": "B",
    "model": "C",
    "costplan": "C",
    "train": "E",
    "hpo": "E",
    "bench": "E",
}


def _is_stub(obj: Any) -> bool:
    """True for a contract stub whose whole body is `raise NotImplementedError(...)`."""
    code = getattr(getattr(obj, "__func__", obj), "__code__", None)
    return code is not None and code.co_names == ("NotImplementedError",)


def gnn_missing(*parts: str) -> str | None:
    """Why `parts` ("module.name" or "module.Class.method" under aml.models.gnn) cannot run yet:
    the stubs among them with their owners, or None."""
    import importlib

    stubs = []
    for part in parts:
        mod, *attrs = part.split(".")
        obj: Any = importlib.import_module(f"aml.models.gnn.{mod}")
        for a in attrs:
            obj = getattr(obj, a, None)
        if obj is None or _is_stub(obj):
            stubs.append(f"{part} ({GNN_OWNERS.get(mod, '?')})")
    return f"awaiting the GNN implementation: stubs {stubs}" if stubs else None


def require_gnn(*parts: str, torch: bool = True) -> None:
    """Skip unless every part in `parts` is implemented and (torch=True) torch and PyG import.
    graph.py is torch-free (load_graph, label_times, guard_bounds run on the laptop too)."""
    import pytest

    if torch:
        pytest.importorskip("torch")
        pytest.importorskip("torch_geometric")
    reason = gnn_missing(*parts)
    if reason:
        pytest.skip(reason)


LOADER_PARTS = (
    "graph.build_hetero",
    "graph.label_times",
    "graph.guard_bounds",
    "sampler.make_loader",
    "transforms.FlattenTransform.__init__",
    "transforms.FlattenTransform.__call__",
)


def load_gnn_yaml() -> dict:
    import yaml

    from tests.fixtures.gnn_graphs import REPO_ROOT

    with (REPO_ROOT / "configs" / "gnn.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def sampler_cfg(fanout: Sequence[int], batch_size: int) -> dict:
    """gnn.yaml's sampler section with `fanout` and both batch sizes replaced."""
    cfg = dict(load_gnn_yaml()["sampler"])
    cfg.update(fanout=list(fanout), batch_size=int(batch_size), eval_batch_size=int(batch_size))
    return cfg


def runtime_cfg(num_workers: int = 0) -> dict:
    cfg = dict(load_gnn_yaml()["runtime"])
    cfg.update(num_workers=int(num_workers), cpu=2)
    return cfg


def engine_batches(
    g: Any,
    seed_gids: np.ndarray,
    *,
    protocol: str,
    fanout: Sequence[int],
    batch_size: int,
    test_bound: str = "end",
    num_workers: int = 0,
) -> list[Any]:
    """Every FlatBatch of the real make_loader (role eval) over `seed_gids`, on build_hetero(g)."""
    from aml.models.gnn.graph import build_hetero
    from aml.models.gnn.sampler import make_loader

    loader = make_loader(
        g,
        build_hetero(g),
        np.asarray(seed_gids, dtype=np.int64),
        protocol=protocol,
        role="eval",
        sampler_cfg=sampler_cfg(fanout, batch_size),
        runtime=runtime_cfg(num_workers),
        device="cpu",
        test_bound=test_bound,
    )
    return list(loader)


def flat_observed(g: Any, batches: Iterable[Any]) -> tuple[list[Observed], dict[str, int]]:
    """(observed subgraphs ordered by seed position, guard totals) of FlatBatches."""
    from aml.models.gnn import add_guard, empty_guard

    obs: list[Observed] = []
    total = empty_guard()
    for fb in batches:
        obs.extend(observed_from_flat(fb, g.src, g.dst))
        total = add_guard(total, fb.guard.tolist())
    return sorted(obs, key=lambda o: o.seed_pos), total


def split_by_days(minute: np.ndarray, day_of_minute_fraction: Sequence[float]) -> np.ndarray:
    """A day per gid from cut points over the minute range: day d (1-based) covers minutes whose
    position in [min, max] lies below cut d. Whole minutes stay in one day (ties never straddle
    a split, as on the real data)."""
    minute = np.asarray(minute, np.int64)
    span = max(1, int(minute.max() - minute.min()) + 1)
    frac = (minute - minute.min()) / span
    return (np.searchsorted(np.asarray(day_of_minute_fraction), frac, side="right") + 1).astype(
        np.int16
    )


def day_splits(day: np.ndarray, data_cfg: Mapping[str, Any] | None = None) -> list[str]:
    """Split name per day from the data.yaml day ranges (default: train 1-6, val_early 7,
    val_late 8, test 9-18)."""
    ranges = (data_cfg or {}).get("split") or {
        "train": [1, 6],
        "val_early": [7, 7],
        "val_late": [8, 8],
        "test": [9, 18],
    }
    out = []
    for d in np.asarray(day).tolist():
        out.append(next(s for s in SPLIT_NAMES if ranges[s][0] <= d <= ranges[s][1]))
    return out
