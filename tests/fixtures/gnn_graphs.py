"""Shared GNN test fixtures (M3 spec §14.1).

- `hand_graph_h1()`: the 8-edge hand graph whose sampler facts were verified with real PyG 2.8 /
  pyg-lib 0.9 (M3 spec F4, F9); `H1_PINNED` holds them.
- `tied_minute_graph()`: 3 minutes with peers ranked before and after the targets, reverse and
  parallel edges, self-loops, a 2-cycle and a 3-cycle closing on targets (incl. a cycle closed
  by a same-minute edge, which a causal sampler must not see).
- `random_tied_graph(...)`: a larger random graph with tied minutes, hubs and self-loops.
- `make_host_graph(graph, ...)`: an aml.models.gnn.graph.HostGraph for an array graph, every
  field derived here independently of graph.py (numpy only).
- `gnn_inputs(prepared, tmp_path, cfgs)`: the real `build.run_build_features` on the synthetic
  dataset (built once per session and shared read-only) plus a small gnn config.

Torch-free at import: the laptop collects every test module.
"""

from __future__ import annotations

import copy
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]

# EngineSpec.gnn_edge_attr_names on the real spec (M2 spec §5.9), in order.
GNN_EDGE_ATTRS = (
    "log_amount_usd",
    "payment_currency",
    "receiving_currency",
    "cross_currency",
    "payment_format",
    "self_loop",
    "same_bank",
    "round_amount",
    "hour_of_day",
    "pair_is_new",
    "out_port",
    "in_port",
    "u_out_gap",
    "u_in_gap",
    "v_in_gap",
    "v_out_gap",
    "pair_gap",
    "rev_pair_gap",
)
CATEGORICAL = ("payment_currency", "receiving_currency", "payment_format")
FLAGS = ("cross_currency", "self_loop", "same_bank", "round_amount", "pair_is_new")
FIXTURE_VOCAB_SIZES = {"payment_currency": 5, "receiving_currency": 5, "payment_format": 4}


@dataclass(frozen=True, eq=False)
class ArrayGraph:
    """A transaction graph as rank-ordered arrays (gid = rank = position)."""

    src: np.ndarray  # int64 (M,)
    dst: np.ndarray  # int64 (M,)
    minute: np.ndarray  # int64 (M,), non-decreasing
    n_nodes: int
    names: dict[str, int] = field(default_factory=dict)  # account name -> id
    targets: dict[str, int] = field(default_factory=dict)  # target name -> gid
    pinned: dict[str, Any] = field(default_factory=dict)  # verified facts (see each builder)

    def __post_init__(self) -> None:
        for name in ("src", "dst", "minute"):
            a = getattr(self, name)
            if a.dtype != np.int64 or a.ndim != 1 or len(a) != len(self.src):
                raise ValueError(f"{name} must be int64 (M,)")
        if len(self.minute) and np.any(np.diff(self.minute) < 0):
            raise ValueError("minute must be non-decreasing in rank")
        ids = np.concatenate([self.src, self.dst])
        if len(ids) and (ids.min() < 0 or ids.max() >= self.n_nodes):
            raise ValueError("account ids must lie in [0, n_nodes)")

    @property
    def n_edges(self) -> int:
        return len(self.src)

    @property
    def rank(self) -> np.ndarray:
        return np.arange(self.n_edges, dtype=np.int64)

    @property
    def first_rank(self) -> np.ndarray:
        """First rank of each gid's minute (brute force over the minute column)."""
        return first_rank_of(self.minute)

    @property
    def causal_bound(self) -> np.ndarray:
        """first_rank - 1 per gid (the causal edge_label_time)."""
        return self.first_rank - 1

    @property
    def rev_gid(self) -> np.ndarray:
        """Forward gid of each rev_to edge (non-self-loops, rank order)."""
        return np.flatnonzero(self.src != self.dst).astype(np.int64)


def _arr(values) -> np.ndarray:
    return np.asarray(values, dtype=np.int64)


def first_rank_of(minute: np.ndarray) -> np.ndarray:
    """int64 first rank of each position's minute (a plain loop: the reference)."""
    first: dict[int, int] = {}
    for r, m in enumerate(np.asarray(minute).tolist()):
        first.setdefault(m, r)
    return np.array([first[m] for m in np.asarray(minute).tolist()], dtype=np.int64)


# --- the hand graph h1 (M3 spec §14.1, F4, F9) ------------------------------------------------

# Verified with torch 2.14 / PyG 2.8.0.post1 / pyg-lib 0.9.0: LinkNeighborLoader on
# build_hetero-style data (TO = all edges, REV = non-self-loop edges reversed, time = rank on
# both), num_neighbors {TO: [5, 5], REV: [5, 5]}, temporal_strategy "last", disjoint, one seed.
H1_PINNED: dict[str, Any] = {
    "fanout": [5, 5],
    "A": {
        "gid": 4,
        "causal_bound": 2,  # first rank of minute 1 (= 3) - 1
        "n_id": [0, 1, 2, 2, 1, 0],  # u, v, then x/v in u's tree and x/u in v's tree
        "batch": [0, 0, 0, 0, 0, 0],
        "edge_label_index": [[0], [1]],
        "account_ego": [1, 1, 0, 0, 1, 1],
        "root_ego": [1, 1, 0, 0, 0, 0],
        "max_time": {"to": 2, "rev_to": 2},
        "to_gids": [0, 0, 1, 2, 2],  # sorted multiset of sampled `to` gids
        "rev_to_gids": [0, 1, 1, 2],  # sorted multiset of sampled `rev_to` forward gids
        "absent_gids": [3, 4],  # y->u (same minute, ranked before) and the target itself
        "lookahead_bound": 7,  # every edge is in split train: train_last = data_last = 7
        "lookahead_target_copies": {"to": 2, "rev_to": 2},  # once per tree per type
        "lookahead_later_edges": {"to": 4, "rev_to": 4},  # sampled edges with time > 4
    },
    "B": {"gid": 7, "causal_bound": 6},  # first rank of minute 3 (= 7) - 1
}


def hand_graph_h1() -> ArrayGraph:
    """Accounts u=0, v=1, x=2, y=3, z=4; rank (minute):
    0 v->x (0) | 1 x->u (0) | 2 u->x (0) | 3 y->u (1) | 4 u->v (1) target A | 5 x->v (2) |
    6 z->z (2) self-loop | 7 v->u (3) target B (a 2-cycle with A)."""
    edges = [(1, 2), (2, 0), (0, 2), (3, 0), (0, 1), (2, 1), (4, 4), (1, 0)]
    return ArrayGraph(
        src=_arr([e[0] for e in edges]),
        dst=_arr([e[1] for e in edges]),
        minute=_arr([0, 0, 0, 1, 1, 2, 2, 3]),
        n_nodes=5,
        names={"u": 0, "v": 1, "x": 2, "y": 3, "z": 4},
        targets={"A": 4, "B": 7},
        pinned=copy.deepcopy(H1_PINNED),
    )


# --- the tied-minute graph ----------------------------------------------------------------------


def tied_minute_graph() -> ArrayGraph:
    """Accounts a..f = 0..5 and g = 6 (no edges); rank (minute):

    minute 0:  0 a->b | 1 b->c | 2 d->a | 3 b->a (reverse of 0) | 4 d->d (self-loop)
    minute 1:  5 e->a (peer before T1) | 6 a->b (parallel of 0, peer before T1) |
               7 c->a  T1: closes the 3-cycle a->b->c->a (ranks 0, 1) |
               8 a->c (reverse of T1, peer after) | 9 f->f (self-loop, after) |
               10 c->a (parallel of T1, after)
    minute 2:  11 a->d  T2: closes the 2-cycle with 2 (d->a) | 12 b->c (parallel of 1) |
               13 c->e | 14 e->c  T3: a 2-cycle with 13, closed in the SAME minute

    pinned[target] = {gid, causal_bound, same_minute (gids of the target's minute: never in its
    causal subgraph, either type, either rank side), cycle (earlier edges closing a cycle on the
    target: sampled with fanout >= [5, 5])}.
    """
    a, b, c, d, e, f = range(6)
    edges = [
        (a, b, 0),
        (b, c, 0),
        (d, a, 0),
        (b, a, 0),
        (d, d, 0),
        (e, a, 1),
        (a, b, 1),
        (c, a, 1),
        (a, c, 1),
        (f, f, 1),
        (c, a, 1),
        (a, d, 2),
        (b, c, 2),
        (c, e, 2),
        (e, c, 2),
    ]
    minute = _arr([x[2] for x in edges])
    targets = {"T1": 7, "T2": 11, "T3": 14}
    cycles = {"T1": [0, 1], "T2": [2], "T3": []}
    bound = first_rank_of(minute) - 1
    pinned: dict[str, Any] = {"fanout": [5, 5]}
    for name, gid in targets.items():
        pinned[name] = {
            "gid": gid,
            "causal_bound": int(bound[gid]),
            "same_minute": np.flatnonzero(minute == minute[gid]).tolist(),
            "cycle": cycles[name],
        }
    pinned["T3"]["unseen_closer"] = 13  # c->e closes T3's 2-cycle within its own minute
    return ArrayGraph(
        src=_arr([x[0] for x in edges]),
        dst=_arr([x[1] for x in edges]),
        minute=minute,
        n_nodes=7,
        names=dict(zip("abcdefg", range(7), strict=True)),
        targets=targets,
        pinned=pinned,
    )


# --- a random graph with tied minutes -----------------------------------------------------------


def random_tied_graph(
    n_nodes: int = 2000,
    n_edges: int = 30000,
    per_minute: int = 30,
    n_hubs: int = 5,
    self_loop_share: float = 0.1,
    seed: int = 0,
) -> ArrayGraph:
    """Random edges in rank order with ~per_minute events per minute (Poisson-like tie sizes),
    hubs (accounts 0..n_hubs-1 take ~30% of endpoints), self-loops (self_loop_share), and
    repeated / reversed earlier pairs (~10% each), so parallel and reverse edges are common."""
    rng = np.random.default_rng(seed)
    n_minutes = max(1, n_edges // per_minute)
    minute = np.sort(rng.integers(0, n_minutes, n_edges)).astype(np.int64)

    def endpoints(k: int) -> np.ndarray:
        hub = rng.random(k) < 0.3
        out = rng.integers(n_hubs, n_nodes, k)
        out[hub] = rng.integers(0, n_hubs, int(hub.sum()))
        return out.astype(np.int64)

    src, dst = endpoints(n_edges), endpoints(n_edges)
    kind = rng.random(n_edges)
    for i in range(1, n_edges):
        j = int(rng.integers(0, i))
        if kind[i] < 0.10:  # repeat an earlier pair (parallel edge)
            src[i], dst[i] = src[j], dst[j]
        elif kind[i] < 0.20:  # reverse an earlier pair
            src[i], dst[i] = dst[j], src[j]
    loops = rng.random(n_edges) < self_loop_share
    dst[loops] = src[loops]
    return ArrayGraph(src=src, dst=dst, minute=minute, n_nodes=n_nodes)


# --- a HostGraph for array graphs -----------------------------------------------------------------


def fixture_preprocess(columns: tuple[str, ...] = GNN_EDGE_ATTRS) -> dict:
    """A `preprocess` dict for fixture EAs (identity normalisation, FIXTURE_VOCAB_SIZES)."""
    cat_idx = [columns.index(c) for c in CATEGORICAL if c in columns]
    num_idx = [i for i in range(len(columns)) if i not in cat_idx]
    zs = [c for c in columns if c not in CATEGORICAL and c not in FLAGS]
    return {
        "mu": {c: 0.0 for c in zs},
        "sigma": {c: 1.0 for c in zs},
        "vocab_sizes": {c: FIXTURE_VOCAB_SIZES[c] for c in CATEGORICAL if c in columns},
        "fitted_on": "train",
        "columns": list(columns),
        "num_idx": num_idx,
        "cat_idx": cat_idx,
        "cat_sizes": [FIXTURE_VOCAB_SIZES[columns[i]] + 1 for i in cat_idx],
    }


def fixture_edge_attrs(g: ArrayGraph, seed: int = 0) -> np.ndarray:
    """float32 (M, 18) EA in GNN_EDGE_ATTRS order: standard-normal numerics, categorical codes in
    -1..vocab-1 (as float), 0/1 flags with self_loop = (src == dst)."""
    rng = np.random.default_rng(seed)
    m = g.n_edges
    ea = rng.standard_normal((m, len(GNN_EDGE_ATTRS))).astype(np.float32)
    for c in CATEGORICAL:
        i = GNN_EDGE_ATTRS.index(c)
        ea[:, i] = rng.integers(-1, FIXTURE_VOCAB_SIZES[c], m).astype(np.float32)
    for c in FLAGS:
        ea[:, GNN_EDGE_ATTRS.index(c)] = (rng.random(m) < 0.3).astype(np.float32)
    ea[:, GNN_EDGE_ATTRS.index("self_loop")] = (g.src == g.dst).astype(np.float32)
    return np.ascontiguousarray(ea)


def make_host_graph(
    g: ArrayGraph,
    *,
    split: list[str] | np.ndarray | None = None,
    day: np.ndarray | None = None,
    ea: np.ndarray | None = None,
    y: np.ndarray | None = None,
    row_id: np.ndarray | None = None,
    primary_last_day: int | None = None,
    label_splits: tuple[str, ...] = ("train", "val_early"),
):
    """A HostGraph for an array graph, every field derived here (independent of graph.py).

    split: per-gid split names (default: all "train"; must be non-decreasing in SPLITS order).
    day: per-gid day (default minute // 1440 + 1). ea: (M, 18) float32 (default
    fixture_edge_attrs). y: per-gid 0/1 labels (default: 1 for gids % 7 == 3), kept only for
    `label_splits` (else -1). row_id: default a fixed permutation of 100..100+M-1 (row_id !=
    rank, as on the real data). primary_last_day: the primary view's last day (default: the
    last day, so d10_last = data_last).

    Bounds of a split absent from the graph (fixture convention): first = M, last = the last
    rank of the latest earlier split present (-1 if none).
    """
    from aml.data.split import SPLITS
    from aml.models.gnn.graph import HostGraph

    m = g.n_edges
    split_arr = np.array(["train"] * m if split is None else list(split), dtype=object)
    if len(split_arr) != m or not set(split_arr) <= set(SPLITS):
        raise ValueError(f"split must name one of {SPLITS} per edge")
    split_code = np.array([SPLITS.index(s) for s in split_arr], dtype=np.int8)
    if m and np.any(np.diff(split_code) < 0):
        raise ValueError("splits must be contiguous day ranges in rank order")
    day_arr = (g.minute // 1440 + 1 if day is None else np.asarray(day)).astype(np.int16)
    if ea is None:
        ea = fixture_edge_attrs(g)
    ea = np.ascontiguousarray(ea, dtype=np.float32)
    if ea.shape != (m, len(GNN_EDGE_ATTRS)):
        raise ValueError(f"ea must be (M, {len(GNN_EDGE_ATTRS)})")
    labels = (np.arange(m) % 7 == 3).astype(np.int8) if y is None else np.asarray(y, np.int8)
    keep = np.isin(split_arr, list(label_splits))
    y_arr = np.where(keep, labels, -1).astype(np.int8)
    if row_id is None:
        row_id = np.random.default_rng(1).permutation(m) + 100
    row_id = np.asarray(row_id, dtype=np.int64)

    def last_of(codes: list[int]) -> int:
        hits = np.flatnonzero(np.isin(split_code, codes))
        return int(hits[-1]) if len(hits) else -1

    def first_of(code: int) -> int:
        hits = np.flatnonzero(split_code == code)
        return int(hits[0]) if len(hits) else m

    last_day = int(day_arr.max()) if m else 0
    pday = last_day if primary_last_day is None else int(primary_last_day)
    d10 = np.flatnonzero(day_arr <= pday)
    bounds = {
        "train_last": last_of([0]),
        "val_last": last_of([0, 1, 2]),
        "d10_last": int(d10[-1]) if len(d10) else -1,
        "data_last": m - 1,
        "val_early_first": first_of(1),
        "val_late_first": first_of(2),
        "test_first": first_of(3),
    }
    return HostGraph(
        n_nodes=g.n_nodes,
        n_edges=m,
        src=np.ascontiguousarray(g.src, dtype=np.int64),
        dst=np.ascontiguousarray(g.dst, dtype=np.int64),
        minute=np.ascontiguousarray(g.minute, dtype=np.int64),
        day=day_arr,
        split_code=split_code,
        row_id=row_id,
        first_rank=g.first_rank,
        rev_gid=g.rev_gid,
        bounds=bounds,
        ea=ea,
        y=y_arr,
        label_splits=tuple(label_splits),
        attr_columns=GNN_EDGE_ATTRS,
        preprocess=fixture_preprocess(),
        features_digest="fixture",
        spec_hash="fixture",
        data_version=None,
    )


# --- the real feature build on the synthetic dataset ------------------------------------------

FIXTURE_FEATURES_KEY = "features-gnn-fixture"
_BUILT: dict[str, Path] = {}  # (prepared root, configs) -> the shared, read-only features_dir


def small_gnn_cfg(gnn_cfg: dict) -> dict:
    """gnn.yaml shrunk for CPU tests: H 8, fanout [5, 3], batch 64, eval batch 128, max_epochs 2
    (min 1, patience 1), HPO 2 trials x 2 epochs, PNA hidden 10, faithful fanout [5, 3] / batch
    64 / 1 epoch, no loader workers. Passes check_gnn_cfg."""
    from aml.models.gnn import check_gnn_cfg

    c = copy.deepcopy(gnn_cfg)
    c["model"]["hidden"] = 8
    c["sampler"].update(fanout=[5, 3], batch_size=64, eval_batch_size=128, eval_cache_max_gb=0.5)
    c["train"].update(max_epochs=2, min_epochs=1, patience=1)
    c["hpo"].update(n_trials=2, max_epochs=2)
    c["protocols"]["pna"]["hidden"] = 10
    c["protocols"]["faithful"].update(fanout=[5, 3], batch_size=64, max_epochs=1)
    c["bench"].update(batch_sizes=[64], num_workers=[0], warmup_steps=1, timed_steps=2)
    c["bench"]["faithful"].update(batch_fallback=[64, 32])
    c["runtime"].update(cpu=2, num_workers=0)
    return dict(check_gnn_cfg(c))


def gnn_inputs(prepared, tmp_path: Path, cfgs: dict[str, dict]) -> tuple[Any, Path, dict]:
    """(paths, features_dir, gnn_cfg_small) on the synthetic dataset.

    prepared: conftest's `prepared` DataPaths. cfgs: {"data", "rules", "features"} as the
    feature build reads them (+ optional "gnn"; default configs/gnn.yaml). `paths` is a private
    copy of the prepared tables under tmp_path (stage dirs written there stay private); the
    feature table is built once per session per (prepared root, configs) with the real
    `build.run_build_features` and shared READ-ONLY. Wrap in a module-scoped fixture:

        @pytest.fixture(scope="module")
        def inputs(prepared, data_cfg, rules_cfg, tmp_path_factory):
            return gnn_inputs(prepared, tmp_path_factory.mktemp("gnn"),
                              {"data": data_cfg, "rules": rules_cfg,
                               "features": load_yaml("features.yaml")})
    """
    from aml.config import config_hash
    from aml.features import build
    from aml.paths import DataPaths

    tmp_path = Path(tmp_path)
    paths = DataPaths(tmp_path / "volume", prepared.dataset)
    for name in ("transactions", "accounts", "fx_rates", "labels"):
        dst = getattr(paths, name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(getattr(prepared, name), dst)
    marker = prepared.parquet_dir / "prepare_summary.json"
    if marker.exists():
        shutil.copy(marker, paths.parquet_dir / marker.name)

    build_cfgs = {k: cfgs[k] for k in ("data", "rules", "features")}
    cache_key = f"{Path(prepared.root).resolve()}|{config_hash(build_cfgs)}"
    features_dir = _BUILT.get(cache_key)
    if features_dir is None or not (features_dir / "summary.json").exists():
        features_dir = tmp_path / "shared_features" / FIXTURE_FEATURES_KEY
        build.run_build_features(paths, features_dir, build_cfgs)
        _BUILT[cache_key] = features_dir

    gnn_cfg = cfgs.get("gnn")
    if gnn_cfg is None:
        from aml.config import load_config

        gnn_cfg = load_config("gnn", REPO_ROOT / "configs")
    return paths, features_dir, small_gnn_cfg(gnn_cfg)
