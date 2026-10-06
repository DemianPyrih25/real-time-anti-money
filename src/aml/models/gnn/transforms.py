"""Batch flattening, the runtime as-of guard, the edge-cap split and the eval-tree cache
(M3 spec §5.4-§5.8; owner B).

`FlattenTransform` runs in the loader workers: it turns a sampled HeteroData into a FlatBatch of
CPU int64 index tensors (only indices cross the worker boundary), computes the ego flag, checks
every sampled edge against its seed's bound (an independent `guard_bounds` array) and raises
LeakError on a violation. Edge attributes are gathered on the device by gid (`to_model_inputs`).

torch is imported inside the functions (the module itself imports without torch).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

from aml.models.gnn import (
    CAUSAL_BOUND_PROTOCOLS,
    GUARD_FIELDS,
    NO_SLACK,
    NODE,
    PROTOCOLS,
    REV,
    TO,
    LeakError,
)

if TYPE_CHECKING:
    from torch import Tensor
    from torch_geometric.data import HeteroData

# Planning value of saved activations per sampled edge (F18: H 64, 2 layers, EU, fp32); the bench
# measures the real value. A null cap in gnn.yaml = floor(fraction * device bytes / this).
BYTES_PER_EDGE_PLANNING = 5900
DEFAULT_MEMORY_FRACTION = 0.6
EVAL_CAP_FACTOR = 4  # null max_edges_per_eval_step = 4 * max_edges_per_step
EGO_MODES = ("account", "root")
CACHE_GB = 1e9  # EvalCache.max_gb is in 10^9 bytes
DETAIL_LIMIT = 20  # offending ids kept in a LeakError's detail


class FlatBatch(NamedTuple):
    """One sampled batch, flattened (CPU tensors only; int64 unless noted).

    B = seeds in the batch, N = sampled node copies, Ef / Er = sampled `to` / `rev_to` edges.
    Local node indices refer to rows of the batch's node list (`n_id` order); node copies are
    per tree (disjoint sampling), so an account can appear more than once.
    """

    n_nodes: int  # N
    ego: Tensor  # (N,) bool: account-level ego flag (§5.4); faithful: every seed endpoint
    node_sub: Tensor | None  # (N,) subgraph id 0..B-1 (= batch % B); None for faithful
    ei_fwd: Tensor  # (2, Ef) local [src; dst] of sampled `to` edges
    gid_fwd: Tensor  # (Ef,) gid = `to` e_id
    ei_rev: Tensor  # (2, Er) local [src; dst] of sampled `rev_to` edges (already reversed)
    gid_rev: Tensor  # (Er,) gid of the forward edge = rev_gid[`rev_to` e_id]
    tgt_src: Tensor  # (B,) local root of the target's src (edge_label_index[0])
    tgt_dst: Tensor  # (B,) local root of the target's dst (edge_label_index[1])
    tgt_gid: Tensor  # (B,) target gid = seed_gids[input_id]
    seed_pos: Tensor  # (B,) input_id = position in the loader's seed array
    sub_edges: Tensor | None  # (B,) Ef + Er per subgraph (split planning); None for faithful
    sampled: Tensor | None  # (B,) bool, faithful only: the target's gid is among gid_fwd
    guard: Tensor  # (6,) int64 in GUARD_FIELDS order (aml.models.gnn)


def _guard_tensor(values: dict[str, int]) -> Tensor:
    import torch

    return torch.tensor([int(values[f]) for f in GUARD_FIELDS], dtype=torch.int64)


def zero_guard() -> Tensor:
    """A (6,) int64 guard of a batch that sampled nothing new (max_slack = NO_SLACK)."""
    return _guard_tensor({f: (NO_SLACK if f == "max_slack" else 0) for f in GUARD_FIELDS})


def _from_numpy(a: np.ndarray, name: str) -> Tensor:
    import torch

    a = np.ascontiguousarray(a, dtype=np.int64).reshape(-1)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D")
    return torch.from_numpy(a)


class FlattenTransform:
    """HeteroData batch -> FlatBatch, with the as-of guard (M3 spec §5.5, §5.6).

    protocol: "causal" | "lookahead" | "pna" | "faithful".
    seed_gids: (S,) int64, the loader's seed array (tgt_gid = seed_gids[input_id]).
    rev_gid: int64 map `rev_to` e_id -> forward gid (graph.rev_gid); None = identity (faithful
        snapshots flip every edge, so rev e_id == gid).
    guard_bound: (S,) int64 aligned with seed_gids, from graph.guard_bounds (independent of
        label_times); faithful: the snapshot's last rank for every seed.
    first_rank: graph.first_rank (counts `future_edges`: sampled rank >= first rank of the
        target's minute); None for faithful (future_edges = 0). Required for look-ahead.
    ego: "account" (production; every copy of the target's src/dst accounts in its subgraph) or
        "root" (only the two roots; exists for the mutation test). Faithful ignores it.
    drop_target: look-ahead step 1: remove every sampled edge whose gid equals its own
        subgraph's target gid, from both types and any hop (nodes stay, possibly isolated).
        Only valid with protocol "lookahead".

    Per batch and edge type: sub = batch[ei[1]] (assert == batch[ei[0]]: no edge crosses
    subgraphs), bnd = guard_bound[input_id][sub], violations = (time > bnd).sum(), max_slack =
    max(time - bnd), hits = (gid == tgt_gid[sub]).sum(). causal/pna: violations or hits ->
    LeakError; lookahead: violations -> LeakError, hits dropped and counted in
    dropped_target_copies (target_hits after the drop = 0; with drop_target=False the remaining
    hits raise LeakError); faithful: violations (gid > last rank) -> LeakError, hits are allowed
    and counted in target_hits.

    Guard counts cover every SAMPLED edge (before the look-ahead drop): edges_checked,
    violations, max_slack and future_edges (time >= first rank of the target's minute, so the
    target's own copies count as future). The batch's root layout (subgraph i holds seed i's two
    roots) and `time == gid` per sampled edge are asserted too: the bound of an edge is looked
    up through them.
    """

    def __init__(
        self,
        *,
        protocol: str,
        seed_gids: np.ndarray,
        rev_gid: np.ndarray | None,
        guard_bound: np.ndarray,
        first_rank: np.ndarray | None,
        ego: str = "account",
        drop_target: bool = False,
    ) -> None:
        if protocol not in PROTOCOLS:
            raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
        if ego not in EGO_MODES:
            raise ValueError(f"ego must be one of {EGO_MODES}, got {ego!r}")
        if drop_target and protocol != "lookahead":
            raise ValueError("drop_target is look-ahead step 1 only")
        if protocol == "lookahead" and first_rank is None:
            raise ValueError("look-ahead needs first_rank (the future-share diagnostic)")
        self.protocol = protocol
        self.ego = ego
        self.drop_target = bool(drop_target)
        self.seed_gids = _from_numpy(seed_gids, "seed_gids")
        self.guard_bound = _from_numpy(guard_bound, "guard_bound")
        if self.guard_bound.shape != self.seed_gids.shape:
            raise ValueError("guard_bound must align with seed_gids")
        self.rev_gid = None if rev_gid is None else _from_numpy(rev_gid, "rev_gid")
        self.first_rank = None
        if protocol != "faithful" and first_rank is not None:
            self.first_rank = _from_numpy(first_rank, "first_rank")
        self.snapshot_last: int | None = None
        if protocol == "faithful":
            bounds = np.unique(np.asarray(guard_bound))
            if len(bounds) != 1:
                raise ValueError("a faithful loader samples one snapshot: one guard bound")
            self.snapshot_last = int(bounds[0])
        elif self.rev_gid is None:
            raise ValueError("temporal protocols need rev_gid (graph.rev_gid)")

    def __call__(self, batch: HeteroData) -> FlatBatch:
        if self.protocol == "faithful":
            return self._faithful(batch)
        return self._temporal(batch)

    # --- causal / look-ahead / PNA (disjoint temporal trees) -----------------------------------

    def _temporal(self, batch: HeteroData) -> FlatBatch:
        import torch

        node, to = batch[NODE], batch[TO]
        n_id, bvec = node.n_id, node.batch
        n = int(n_id.numel())
        eli, input_id = to.edge_label_index, to.input_id
        b = int(input_id.numel())
        tgt_gid = self.seed_gids[input_id]
        bound_b = self.guard_bound[input_id]
        ar = torch.arange(b, dtype=torch.int64)
        if (
            eli.shape != (2, b)
            or (n and int(bvec.max()) >= b)
            or not torch.equal(bvec[eli[0]], ar)
            or not torch.equal(bvec[eli[1]], ar)
        ):
            raise LeakError(
                "unexpected root layout: subgraph i must hold seed i's two roots",
                {"protocol": self.protocol, "seed_pos": input_id[:DETAIL_LIMIT].tolist()},
            )
        if self.ego == "account":
            u_acc, v_acc = n_id[eli[0]], n_id[eli[1]]
            ego = (n_id == u_acc[bvec]) | (n_id == v_acc[bvec])
        else:
            ego = torch.zeros(n, dtype=torch.bool)
            ego[eli.reshape(-1)] = True
        fr_b = self.first_rank[tgt_gid] if self.first_rank is not None else None

        g = {f: 0 for f in GUARD_FIELDS}
        g["max_slack"] = NO_SLACK
        problems: list[dict[str, Any]] = []
        out: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
        for et, name in ((TO, "to"), (REV, "rev_to")):
            store = batch[et]
            ei, e_id, t = store.edge_index, store.e_id, store.time
            gid = e_id if et == TO else self.rev_gid[e_id]
            sub = bvec[ei[1]]
            if int(e_id.numel()):
                if not torch.equal(sub, bvec[ei[0]]):
                    problems.append({"edge_type": name, "problem": "edge crosses subgraphs"})
                if not torch.equal(t, gid):
                    problems.append({"edge_type": name, "problem": "time != gid (rank)"})
                diff = t - bound_b[sub]
                viol = diff > 0
                hit = gid == tgt_gid[sub]
                n_viol, n_hit = int(viol.sum()), int(hit.sum())
                g["edges_checked"] += int(e_id.numel())
                g["max_slack"] = max(g["max_slack"], int(diff.max()))
                if fr_b is not None:
                    g["future_edges"] += int((t >= fr_b[sub]).sum())
                if n_viol:
                    g["violations"] += n_viol
                    problems.append(self._detail(name, "edge later than its bound", viol, sub,
                                                 gid, t, bound_b, input_id))  # fmt: skip
                if n_hit:
                    if self.protocol in CAUSAL_BOUND_PROTOCOLS or not self.drop_target:
                        g["target_hits"] += n_hit
                        problems.append(self._detail(name, "target edge sampled", hit, sub,
                                                     gid, t, bound_b, input_id))  # fmt: skip
                    else:
                        keep = ~hit
                        ei, gid, sub = ei[:, keep], gid[keep], sub[keep]
                        g["dropped_target_copies"] += n_hit
            out[name] = (ei, gid, sub)
        if problems:
            raise LeakError(
                f"as-of guard ({self.protocol}): {problems[0]['problem']} "
                f"({len(problems)} finding(s))",
                {"protocol": self.protocol, "guard": dict(g), "findings": problems},
            )
        ei_f, gid_f, sub_f = out["to"]
        ei_r, gid_r, sub_r = out["rev_to"]
        sub_edges = torch.bincount(sub_f, minlength=b) + torch.bincount(sub_r, minlength=b)
        return FlatBatch(
            n_nodes=n,
            ego=ego,
            node_sub=bvec,
            ei_fwd=ei_f,
            gid_fwd=gid_f,
            ei_rev=ei_r,
            gid_rev=gid_r,
            tgt_src=eli[0].contiguous(),
            tgt_dst=eli[1].contiguous(),
            tgt_gid=tgt_gid,
            seed_pos=input_id,
            sub_edges=sub_edges,
            sampled=None,
            guard=_guard_tensor(g),
        )

    def _detail(self, name, problem, mask, sub, gid, t, bound_b, input_id) -> dict[str, Any]:
        idx = mask.nonzero().reshape(-1)[:DETAIL_LIMIT]
        return {
            "edge_type": name,
            "problem": problem,
            "count": int(mask.sum()),
            "seed_pos": input_id[sub[idx]].tolist(),
            "gid": gid[idx].tolist(),
            "time": t[idx].tolist(),
            "bound": bound_b[sub[idx]].tolist(),
        }

    # --- faithful (non-temporal, non-disjoint snapshot) ----------------------------------------

    def _faithful(self, batch: HeteroData) -> FlatBatch:
        import torch

        node, to, rev = batch[NODE], batch[TO], batch[REV]
        n = int(node.n_id.numel())
        eli, input_id = to.edge_label_index, to.input_id
        tgt_gid = self.seed_gids[input_id]
        gid_f = to.e_id
        gid_r = rev.e_id if self.rev_gid is None else self.rev_gid[rev.e_id]
        last = int(self.snapshot_last)
        g = {f: 0 for f in GUARD_FIELDS}
        g["max_slack"] = NO_SLACK
        for gid in (gid_f, gid_r):
            if int(gid.numel()):
                g["edges_checked"] += int(gid.numel())
                g["violations"] += int((gid > last).sum())
                g["max_slack"] = max(g["max_slack"], int(gid.max()) - last)
        tgt_out = int((tgt_gid > last).sum())
        if g["violations"] or tgt_out:
            raise LeakError(
                f"faithful snapshot guard: {g['violations']} sampled edges and {tgt_out} targets "
                f"beyond rank {last}",
                {"protocol": "faithful", "guard": dict(g), "targets_beyond": tgt_out},
            )
        # Every gid is now <= last: mark arrays over [0, last] replace three sort-based
        # torch.isin calls (~10x faster on 1.2M-edge batches, same counts).
        mark = torch.zeros(last + 1, dtype=torch.bool)
        mark[tgt_gid] = True
        g["target_hits"] += int(mark[gid_f].sum()) + int(mark[gid_r].sum())
        seen = torch.zeros(last + 1, dtype=torch.bool)
        seen[gid_f] = True
        sampled = seen[tgt_gid]
        ego = torch.zeros(n, dtype=torch.bool)
        ego[eli.reshape(-1)] = True
        return FlatBatch(
            n_nodes=n,
            ego=ego,
            node_sub=None,
            ei_fwd=to.edge_index,
            gid_fwd=gid_f,
            ei_rev=rev.edge_index,
            gid_rev=gid_r,
            tgt_src=eli[0].contiguous(),
            tgt_dst=eli[1].contiguous(),
            tgt_gid=tgt_gid,
            seed_pos=input_id,
            sub_edges=None,
            sampled=sampled,
            guard=_guard_tensor(g),
        )


# --- the edge-cap split -----------------------------------------------------------------------


def split_flat_batch(fb: FlatBatch, max_edges: int) -> list[FlatBatch]:
    """Deterministic split under an edge cap (M3 spec §5.7): consecutive subgraph ids grouped
    greedily so each part's sum(sub_edges) <= max_edges (a single larger subgraph is its own
    part); each part keeps its nodes (node_sub in the part), relabelled with cumsum(mask) - 1,
    and its seeds' tgt_*/seed_pos/sub_edges; node_sub is re-based to 0 in every part. Edges keep
    their relative order. The parent's guard goes to the first part, the others get
    zero_guard() (parts sum to the parent). Returns [fb] if it fits. Faithful batches (node_sub
    None) are not splittable: ValueError."""
    import torch

    if fb.node_sub is None or fb.sub_edges is None:
        raise ValueError("faithful batches are not splittable (non-disjoint)")
    if int(max_edges) < 1:
        raise ValueError(f"max_edges must be >= 1, got {max_edges}")
    sizes = fb.sub_edges.tolist()
    if sum(sizes) <= max_edges:
        return [fb]
    groups: list[tuple[int, int]] = []
    start, acc = 0, 0
    for i, e in enumerate(sizes):
        if i > start and acc + e > max_edges:
            groups.append((start, i))
            start, acc = i, 0
        acc += e
    groups.append((start, len(sizes)))
    if len(groups) == 1:
        return [fb]
    sub_f = fb.node_sub[fb.ei_fwd[1]]
    sub_r = fb.node_sub[fb.ei_rev[1]]
    parts = []
    for k, (a, b) in enumerate(groups):
        nmask = (fb.node_sub >= a) & (fb.node_sub < b)
        new = torch.cumsum(nmask.to(torch.int64), 0) - 1
        fm = (sub_f >= a) & (sub_f < b)
        rm = (sub_r >= a) & (sub_r < b)
        parts.append(
            FlatBatch(
                n_nodes=int(nmask.sum()),
                ego=fb.ego[nmask],
                node_sub=fb.node_sub[nmask] - a,
                ei_fwd=new[fb.ei_fwd[:, fm]],
                gid_fwd=fb.gid_fwd[fm],
                ei_rev=new[fb.ei_rev[:, rm]],
                gid_rev=fb.gid_rev[rm],
                tgt_src=new[fb.tgt_src[a:b]],
                tgt_dst=new[fb.tgt_dst[a:b]],
                tgt_gid=fb.tgt_gid[a:b],
                seed_pos=fb.seed_pos[a:b],
                sub_edges=fb.sub_edges[a:b],
                sampled=None,
                guard=fb.guard if k == 0 else zero_guard(),
            )
        )
    return parts


def resolve_edge_caps(
    sampler_cfg: dict,
    device: str,
    *,
    memory_fraction: float = DEFAULT_MEMORY_FRACTION,
    bytes_per_edge: float = BYTES_PER_EDGE_PLANNING,
) -> tuple[int | None, int | None]:
    """(train cap, eval cap) in edges. Configured values win. A null train cap on CUDA =
    floor(memory_fraction * torch.cuda.get_device_properties(0).total_memory / bytes_per_edge);
    a null eval cap = EVAL_CAP_FACTOR * the train cap (None when the train cap is None); on CPU
    a null train cap means no split (None)."""
    train = sampler_cfg.get("max_edges_per_step")
    evalc = sampler_cfg.get("max_edges_per_eval_step")
    for name, v in (("max_edges_per_step", train), ("max_edges_per_eval_step", evalc)):
        if v is not None and (isinstance(v, bool) or not isinstance(v, int) or v < 1):
            raise ValueError(f"sampler.{name} must be null or an int >= 1, got {v!r}")
    if train is None and str(device).startswith("cuda"):
        import torch

        total = torch.cuda.get_device_properties(0).total_memory
        train = int(math.floor(float(memory_fraction) * total / float(bytes_per_edge)))
    if evalc is None and train is not None:
        evalc = EVAL_CAP_FACTOR * int(train)
    return (None if train is None else int(train)), (None if evalc is None else int(evalc))


# --- device inputs and labels -----------------------------------------------------------------


def to_model_inputs(fb: FlatBatch, ea_dev: Tensor, device: str) -> tuple:
    """(x, ei_fwd, ea_fwd, ei_rev, ea_rev, tgt_src, tgt_dst, tgt_attr) on `device`:
    x = ego.float()[:, None] (N, 1); ea_fwd = EA[gid_fwd]; ea_rev = EA[gid_rev] (a reverse copy
    shares its forward row); tgt_attr = EA[tgt_gid]; gathers happen on the device (`ea_dev` is
    the device-resident EA; faithful passes the snapshot's EA_f)."""
    nb = str(device).startswith("cuda")

    def dev(t: Tensor) -> Tensor:
        return t.to(device, non_blocking=nb)

    x = dev(fb.ego).float().unsqueeze(1)
    ei_f, ei_r = dev(fb.ei_fwd), dev(fb.ei_rev)
    ea_f = ea_dev.index_select(0, dev(fb.gid_fwd))
    ea_r = ea_dev.index_select(0, dev(fb.gid_rev))
    tgt_attr = ea_dev.index_select(0, dev(fb.tgt_gid))
    return x, ei_f, ea_f, ei_r, ea_r, dev(fb.tgt_src), dev(fb.tgt_dst), tgt_attr


def batch_labels(fb: FlatBatch, y_dev: Tensor) -> Tensor:
    """int64 labels Y[tgt_gid] on y_dev's device (training only); ValueError if any is -1."""
    y = y_dev.index_select(0, fb.tgt_gid.to(y_dev.device)).long()
    if bool((y < 0).any()):
        raise ValueError(f"{int((y < 0).sum())} batch targets have no loaded label")
    return y


# --- the eval-tree cache ------------------------------------------------------------------------

_INT32_FIELDS = (
    "node_sub",
    "ei_fwd",
    "gid_fwd",
    "ei_rev",
    "gid_rev",
    "tgt_src",
    "tgt_dst",
    "tgt_gid",
    "seed_pos",
    "sub_edges",
)
_INT32_MAX = 2**31 - 1


def _n_edges(fb: FlatBatch) -> int:
    return int(fb.gid_fwd.numel()) + int(fb.gid_rev.numel())


def _compact(fb: FlatBatch) -> dict[str, Any]:
    """An int32 host copy of a FlatBatch (guard dropped: a replay carries zero_guard())."""
    import torch

    out: dict[str, Any] = {"n_nodes": fb.n_nodes, "ego": fb.ego.clone()}
    for f in _INT32_FIELDS:
        t = getattr(fb, f)
        if t is None:
            out[f] = None
            continue
        if t.numel() and (int(t.max()) > _INT32_MAX or int(t.min()) < -_INT32_MAX):
            raise ValueError(f"{f} does not fit int32: the eval cache cannot hold it")
        out[f] = t.to(torch.int32)
    out["sampled"] = None if fb.sampled is None else fb.sampled.clone()
    return out


def _nbytes(c: dict[str, Any]) -> int:
    return sum(
        int(v.numel()) * int(v.element_size())
        for v in c.values()
        if v is not None and hasattr(v, "element_size")
    )


def _expand(c: dict[str, Any]) -> FlatBatch:
    import torch

    kw = {f: (None if c[f] is None else c[f].to(torch.int64)) for f in _INT32_FIELDS}
    return FlatBatch(
        n_nodes=int(c["n_nodes"]),
        ego=c["ego"],
        sampled=c["sampled"],
        guard=zero_guard(),
        **kw,
    )


class EvalCache:
    """Host-RAM cache of one eval loader's (already split) FlatBatches (M3 spec §5.8).

    With `last` and fixed bounds, the val_early trees are identical every epoch, seed and HPO
    trial. The first `batches()` pass streams from the loader, splits each batch by `max_edges`,
    yields the parts and keeps int32 copies (local indices and gids < 2^31) while the running
    size (the copies' bytes, ~ 12 * (Ef + Er) + 5 * N) stays <= max_gb (10^9 bytes); past the cap
    it drops the copies and streams on every later pass. A complete cached pass is replayed from
    RAM (converted back to int64) with zero_guard() (its edges were checked when sampled), so
    summing `guard` over every yielded part counts each sampled edge once. A pass the consumer
    abandons early is not cached (the next pass streams again). One cache per (loader,
    max_edges): another max_edges raises ValueError.
    """

    def __init__(self, max_gb: float) -> None:
        if not max_gb >= 0:
            raise ValueError(f"max_gb must be >= 0, got {max_gb}")
        self.max_bytes = int(float(max_gb) * CACHE_GB)
        self._parts: list[dict[str, Any]] | None = None
        self._overflow = False
        self._max_edges: int | None = None
        self._bound = False
        self._nbytes = 0
        self._edges = 0
        self._batches = 0
        self._max_batch_edges = 0
        self._max_part_edges = 0
        self.streamed_passes = 0
        self.cached_passes = 0

    def batches(self, loader: Iterable[FlatBatch], max_edges: int | None) -> Iterator[FlatBatch]:
        """Yield the eval pass's parts (max_edges None: no split)."""
        if self._bound and max_edges != self._max_edges:
            raise ValueError(
                f"this EvalCache was filled with max_edges={self._max_edges}, got {max_edges}"
            )
        self._bound, self._max_edges = True, max_edges
        if self._parts is not None:
            return self._replay()
        return self._stream(loader, max_edges)

    def _replay(self) -> Iterator[FlatBatch]:
        self.cached_passes += 1
        for c in self._parts or []:
            yield _expand(c)

    def _stream(self, loader: Iterable[FlatBatch], max_edges: int | None) -> Iterator[FlatBatch]:
        keep = not self._overflow and self.max_bytes > 0
        pending: list[dict[str, Any]] = []
        size = edges = n_batches = max_batch = max_part = 0
        for fb in loader:
            e = _n_edges(fb)
            max_batch = max(max_batch, e)
            parts = (
                [fb]
                if max_edges is None or fb.node_sub is None
                else split_flat_batch(fb, max_edges)
            )
            for p in parts:
                pe = _n_edges(p)
                edges += pe
                n_batches += 1
                max_part = max(max_part, pe)
                if keep:
                    c = _compact(p)
                    size += _nbytes(c)
                    if size > self.max_bytes:
                        keep, pending = False, []
                        self._overflow = True
                    else:
                        pending.append(c)
                yield p
        self.streamed_passes += 1
        self._edges, self._batches = edges, n_batches
        self._max_batch_edges, self._max_part_edges = max_batch, max_part
        if keep:
            self._parts, self._nbytes = pending, size

    @property
    def cached(self) -> bool:
        """True once a complete pass is held in RAM."""
        return self._parts is not None

    @property
    def nbytes(self) -> int:
        return self._nbytes if self._parts is not None else 0

    def stats(self) -> dict:
        """{"cached": bool, "bytes": int, "gb": float, "batches": int, "edges": int,
        "max_edges_per_batch": int (largest loader batch, before the split),
        "max_edges_per_part": int, "overflow": bool, "streamed_passes": int,
        "cached_passes": int}; batches/edges describe the last complete streamed pass."""
        return {
            "cached": self.cached,
            "bytes": self.nbytes,
            "gb": self.nbytes / CACHE_GB,
            "batches": self._batches,
            "edges": self._edges,
            "max_edges_per_batch": self._max_batch_edges,
            "max_edges_per_part": self._max_part_edges,
            "overflow": self._overflow,
            "streamed_passes": self.streamed_passes,
            "cached_passes": self.cached_passes,
        }
