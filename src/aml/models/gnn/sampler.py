"""Temporal link-neighbour loaders and the per-epoch negative subset (M3 spec §5.1; owner B).

Causal, look-ahead, PNA, HPO and bench use one LinkNeighborLoader over `build_hetero(g)`:
per-type fanout {TO: f, REV: f}, `time_attr="time"`, `temporal_strategy="last"`, `disjoint=True`
(forced by temporal sampling anyway: one tree per endpoint, merged by `batch % num_pos`),
`edge_label_time = label_times(...)`, `shuffle=False`, `transform=FlattenTransform(...)` with
`filter_per_worker=True` (the transform runs in the workers and returns a FlatBatch of CPU index
tensors). Faithful uses a non-temporal, non-disjoint, uniform loader over a snapshot (§9).

Loaders over the same HeteroData with the same fanout share one pyg NeighborSampler (its CSC
copy of the graph, ~0.25 GB on HI-Small, is built once per container, before workers fork).
The edge label times are per loader, so sharing the sampler never mixes bounds.

DataLoader options valid only with workers (`persistent_workers`, `prefetch_factor`, `timeout`,
`worker_init_fn`) are passed only when num_workers > 0 (F13).
"""

from __future__ import annotations

import weakref
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import numpy as np
import torch

from aml.models.gnn import LOADER_ROLES, REV, TEMPORAL_PROTOCOLS, TEST_BOUNDS, TO

if TYPE_CHECKING:
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import LinkNeighborLoader
    from torch_geometric.sampler import NeighborSampler

    from aml.models.gnn.graph import HostGraph

EPOCH_SEED_SALT = 0x5EED
MAIN_THREADS = 2  # torch.set_num_threads in the main process
PREFETCH_FACTOR = 4
FAITHFUL_GENERATOR_STRIDE = 1000  # faithful shuffle: generator seed = seed * 1000 + epoch (§9)

# (id(hetero), fanout, temporal_strategy) -> (weakref to the hetero, its NeighborSampler)
_SHARED: dict[tuple, tuple[weakref.ref, Any]] = {}


def epoch_positions(
    pos_idx: np.ndarray, neg_idx: np.ndarray, rate: float, seed: int, epoch: int
) -> np.ndarray:
    """int64 positions into the train seed array for one epoch: all of `pos_idx` + exactly
    round(rate * len(neg_idx)) of `neg_idx` drawn without replacement, concatenated, then
    permuted; rng = np.random.default_rng([seed, epoch, EPOCH_SEED_SALT]). A pure function of
    (pos_idx, neg_idx, rate, seed, epoch): resume and every HPO trial see the same subset."""
    pos = np.asarray(pos_idx, dtype=np.int64).reshape(-1)
    neg = np.asarray(neg_idx, dtype=np.int64).reshape(-1)
    if not 0.0 < float(rate) <= 1.0:
        raise ValueError(f"negative rate must be in (0, 1], got {rate}")
    if int(seed) < 0 or int(epoch) < 0:
        raise ValueError(f"seed and epoch must be >= 0, got {seed}, {epoch}")
    k = n_negatives(len(neg), rate)
    rng = np.random.default_rng([int(seed), int(epoch), EPOCH_SEED_SALT])
    chosen = rng.choice(neg, size=k, replace=False) if k else np.empty(0, np.int64)
    return rng.permutation(np.concatenate([pos, chosen.astype(np.int64)]))


def n_negatives(n_neg: int, rate: float) -> int:
    """round(rate * n_neg): the negatives drawn per epoch (Python's round)."""
    return int(round(float(rate) * int(n_neg)))


class EpochSubsetSampler(torch.utils.data.Sampler[int]):
    """The train loader's index sampler: yields `epoch_positions(...)` of the current epoch.
    `set_epoch(e)` must be called before every `iter(loader)` (also with persistent workers:
    the main process iterates the sampler). Before the first set_epoch the epoch is 0."""

    def __init__(self, pos_idx: np.ndarray, neg_idx: np.ndarray, rate: float, seed: int) -> None:
        super().__init__()
        self.pos_idx = np.ascontiguousarray(pos_idx, dtype=np.int64).reshape(-1)
        self.neg_idx = np.ascontiguousarray(neg_idx, dtype=np.int64).reshape(-1)
        both = np.concatenate([self.pos_idx, self.neg_idx])
        if both.size and both.min() < 0:
            raise ValueError("positions must be >= 0")
        if len(np.unique(both)) != len(both):
            raise ValueError("pos_idx and neg_idx must be distinct positions (no overlap)")
        if not 0.0 < float(rate) <= 1.0:
            raise ValueError(f"negative rate must be in (0, 1], got {rate}")
        if int(seed) < 0:
            raise ValueError(f"seed must be >= 0, got {seed}")
        self.rate = float(rate)
        self.seed = int(seed)
        self.epoch = 0
        self.max_position = int(both.max()) if both.size else -1

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError(f"epoch must be >= 0, got {epoch}")
        self.epoch = int(epoch)

    def positions(self, epoch: int | None = None) -> np.ndarray:
        """The positions of `epoch` (default: the current epoch)."""
        e = self.epoch if epoch is None else int(epoch)
        return epoch_positions(self.pos_idx, self.neg_idx, self.rate, self.seed, e)

    def __iter__(self) -> Iterator[int]:
        return iter(self.positions().tolist())

    def __len__(self) -> int:
        """n_pos + round(rate * n_neg): constant over epochs."""
        return len(self.pos_idx) + n_negatives(len(self.neg_idx), self.rate)


class FaithfulEpochSampler(torch.utils.data.Sampler[int]):
    """The faithful train loader's shuffle: epoch e yields torch.randperm(n) drawn from a FRESH
    torch.Generator seeded with seed * FAITHFUL_GENERATOR_STRIDE + e, so the order is a pure
    function of (seed, epoch). The loader gets no `generator`: with one, DataLoader draws its
    workers' base seed from that same generator when an iterator is created (with persistent
    workers: the first epoch in a process), which shifted the first epoch after a resume.
    Without one the base seed comes from the global RNG, which seed_epoch re-seeds after iter().
    `set_epoch(e)` must be called before every `iter(loader)`; before it the epoch is 0."""

    def __init__(self, n: int, seed: int) -> None:
        super().__init__()
        if int(n) < 1 or int(seed) < 0:
            raise ValueError(f"need n >= 1 and seed >= 0, got {n}, {seed}")
        self.n, self.seed, self.epoch = int(n), int(seed), 0

    def set_epoch(self, epoch: int) -> None:
        if int(epoch) < 0:
            raise ValueError(f"epoch must be >= 0, got {epoch}")
        self.epoch = int(epoch)

    def order(self, epoch: int | None = None) -> torch.Tensor:
        """The permutation of `epoch` (default: the current epoch)."""
        e = self.epoch if epoch is None else int(epoch)
        gen = torch.Generator()
        gen.manual_seed(self.seed * FAITHFUL_GENERATOR_STRIDE + e)
        return torch.randperm(self.n, generator=gen)

    def __iter__(self) -> Iterator[int]:
        return iter(self.order().tolist())

    def __len__(self) -> int:
        return self.n


def worker_init(_worker_id: int) -> None:
    """DataLoader worker_init_fn: torch.set_num_threads(1)."""
    torch.set_num_threads(1)


def close_loader(loader: Any) -> None:
    """Stop a loader's persistent workers now; call it before dropping a train loader.

    A PyG loader's iterator references the loader (its collate_fn is a bound method), so a
    dropped persistent-worker loader is freed only by the cyclic GC, and a shutdown from there
    waits DataLoader's 5 s join timeout per worker (35 s at 7 workers, measured on the runner;
    an explicit shutdown takes ~0.02 s). Safe on loaders without workers and when repeated."""
    it = getattr(loader, "_iterator", None)
    if it is None:
        return
    loader._iterator = None
    shutdown = getattr(it, "_shutdown_workers", None)
    if shutdown is not None:
        shutdown()


def _worker_kwargs(runtime: dict, *, persistent: bool) -> tuple[int, dict]:
    w = int(runtime["num_workers"])
    if w < 0:
        raise ValueError(f"num_workers must be >= 0, got {w}")
    if w == 0:
        return 0, {}
    return w, {
        "persistent_workers": bool(persistent),
        "prefetch_factor": PREFETCH_FACTOR,
        "timeout": int(runtime["loader_timeout_s"]),
        "worker_init_fn": worker_init,
    }


def shared_neighbor_sampler(
    hetero: HeteroData, fanout: list[int], *, temporal_strategy: str, share_memory: bool
) -> NeighborSampler:
    """The temporal, disjoint NeighborSampler of `hetero` with per-type `fanout` (one per
    (hetero, fanout, strategy) in this process)."""
    from torch_geometric.sampler import NeighborSampler

    for k in [k for k, (ref, _) in _SHARED.items() if ref() is None]:
        del _SHARED[k]
    key = (id(hetero), tuple(int(f) for f in fanout), str(temporal_strategy))
    hit = _SHARED.get(key)
    if hit is not None and hit[0]() is hetero:
        return hit[1]
    ns = NeighborSampler(
        hetero,
        num_neighbors={TO: list(key[1]), REV: list(key[1])},
        disjoint=True,
        temporal_strategy=temporal_strategy,
        time_attr="time",
        share_memory=share_memory,
    )
    _SHARED[key] = (weakref.ref(hetero), ns)
    return ns


def make_loader(
    g: HostGraph,
    hetero: HeteroData,
    seed_gids: np.ndarray,
    *,
    protocol: str,
    role: str,
    sampler_cfg: dict,
    runtime: dict,
    device: str,
    test_bound: str = "end",
    sampler: EpochSubsetSampler | None = None,
) -> LinkNeighborLoader:
    """The temporal loader of one seed array (M3 spec §5.1).

    protocol: "causal" | "lookahead" | "pna" (HPO, bench and dev pass "causal").
    role: "train" -> batch_size = sampler_cfg["batch_size"], `sampler` (an EpochSubsetSampler
        over positions of seed_gids) required, persistent workers; "eval" -> eval_batch_size,
        sampler None, seeds in the given (rank) order, never down-sampled (the caller passes
        `split_gids(g, split)` and asserts the count).
    seed_gids: int64 gids (train: all train gids; eval: `split_gids(g, split)`).
    sampler_cfg: gnn.yaml `sampler` (fanout, temporal_strategy, batch sizes, ego).
    runtime: gnn.yaml `runtime` (num_workers, loader_timeout_s).
    device: "cuda" -> pin_memory=True.
    The transform is FlattenTransform(protocol=protocol, seed_gids=seed_gids, rev_gid=g.rev_gid,
    guard_bound=guard_bounds(g, seed_gids, ...), first_rank=g.first_rank, ego=sampler_cfg["ego"],
    drop_target=(protocol == "lookahead")); edge_label_time = label_times(g, seed_gids, ...).
    Batches carry `seed_pos` = positions into seed_gids (input_id, also under `sampler`).
    """
    from torch_geometric.loader import LinkNeighborLoader

    from aml.models.gnn.graph import guard_bounds, label_times
    from aml.models.gnn.transforms import FlattenTransform

    if protocol not in TEMPORAL_PROTOCOLS:
        raise ValueError(f"make_loader serves {TEMPORAL_PROTOCOLS}, got {protocol!r}")
    if role not in LOADER_ROLES:
        raise ValueError(f"unknown loader role {role!r}; expected one of {LOADER_ROLES}")
    if test_bound not in TEST_BOUNDS:
        raise ValueError(f"unknown test bound {test_bound!r}; expected one of {TEST_BOUNDS}")
    if "time" not in hetero[TO] or "time" not in hetero[REV]:
        raise ValueError("a temporal loader needs build_hetero(g, temporal=True)")
    if hetero[TO].num_edges != g.n_edges:
        raise ValueError("hetero must hold every edge of g (build_hetero(g) without last_rank)")
    seeds = np.ascontiguousarray(seed_gids, dtype=np.int64).reshape(-1)
    if not len(seeds):
        raise ValueError("no seeds")
    if role == "train":
        if not isinstance(sampler, EpochSubsetSampler):
            raise ValueError("a train loader needs an EpochSubsetSampler over the seed positions")
        if sampler.max_position >= len(seeds):
            raise ValueError("the sampler's positions exceed the seed array")
        batch_size = int(sampler_cfg["batch_size"])
    else:
        if sampler is not None:
            raise ValueError("an eval loader takes every seed in order: sampler must be None")
        batch_size = int(sampler_cfg["eval_batch_size"])
    fanout = [int(f) for f in sampler_cfg["fanout"]]
    strategy = str(sampler_cfg.get("temporal_strategy", "last"))

    lt = label_times(g, seeds, protocol=protocol, test_bound=test_bound)
    transform = FlattenTransform(
        protocol=protocol,
        seed_gids=seeds,
        rev_gid=g.rev_gid,
        guard_bound=guard_bounds(g, seeds, protocol=protocol, test_bound=test_bound),
        first_rank=g.first_rank,
        ego=str(sampler_cfg.get("ego", "account")),
        drop_target=(protocol == "lookahead"),
    )
    workers, extra = _worker_kwargs(runtime, persistent=(role == "train"))
    ns = shared_neighbor_sampler(
        hetero, fanout, temporal_strategy=strategy, share_memory=workers > 0
    )
    return LinkNeighborLoader(
        hetero,
        num_neighbors={TO: fanout, REV: fanout},
        edge_label_index=(TO, hetero[TO].edge_index[:, torch.from_numpy(seeds)]),
        edge_label_time=torch.from_numpy(lt),
        time_attr="time",
        temporal_strategy=strategy,
        disjoint=True,
        neighbor_sampler=ns,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        transform=transform,
        filter_per_worker=True,
        num_workers=workers,
        pin_memory=str(device).startswith("cuda"),
        **extra,
    )


def make_faithful_loader(
    snapshot: HeteroData,
    seed_gids: np.ndarray,
    *,
    last_rank: int,
    fanout: list[int],
    batch_size: int,
    shuffle: bool,
    seed: int,
    runtime: dict,
    device: str,
) -> LinkNeighborLoader:
    """Multi-GNN's loader (M3 spec §9) over a snapshot from `faithful.snapshot_hetero`:
    non-temporal, non-disjoint, uniform, per-type fanout, seeds = the snapshot split's gids.

    last_rank: the last rank the seeds' split may see (faithful.snapshot_last_ranks of the
        seeds' split: train_last for train, val_last for val, data_last for test), passed by
        the caller independently of `snapshot`: a snapshot holding other ranks raises
        ValueError, so a loader built over the wrong snapshot cannot run (the snapshot guard
        alone, bounded by the snapshot it samples from, could never fire).
    The transform is FlattenTransform(protocol="faithful", rev_gid = identity (all edges are
    flipped, so rev e_id == gid), guard_bound = last_rank for every seed, first_rank=None).
    shuffle=True: sampler = FaithfulEpochSampler(len(seeds), seed); the caller calls
    `loader.sampler.set_epoch(epoch)` before every epoch (order = f(seed, epoch), also across
    resumes). Shuffled (train) loaders keep their workers between epochs.
    """
    from torch_geometric.loader import LinkNeighborLoader

    from aml.models.gnn.transforms import FlattenTransform

    if "time" in snapshot[TO] or "time" in snapshot[REV]:
        raise ValueError("a faithful snapshot is non-temporal (faithful.snapshot_hetero)")
    n = int(snapshot[TO].num_edges)
    if int(snapshot[REV].num_edges) != n:
        raise ValueError("a faithful snapshot flips every edge (rev e_id == gid)")
    if n - 1 != int(last_rank):
        raise ValueError(
            f"the snapshot holds ranks <= {n - 1}, but its seeds' split may see ranks <= "
            f"{int(last_rank)}: a loader over the wrong snapshot"
        )
    seeds = np.ascontiguousarray(seed_gids, dtype=np.int64).reshape(-1)
    if not len(seeds) or seeds.min() < 0 or seeds.max() >= n:
        raise ValueError(f"faithful seeds must be gids inside the snapshot [0, {n})")
    transform = FlattenTransform(
        protocol="faithful",
        seed_gids=seeds,
        rev_gid=None,
        guard_bound=np.full(len(seeds), int(last_rank), dtype=np.int64),
        first_rank=None,
    )
    sampler = FaithfulEpochSampler(len(seeds), int(seed)) if shuffle else None
    workers, extra = _worker_kwargs(runtime, persistent=bool(shuffle))
    return LinkNeighborLoader(
        snapshot,
        num_neighbors={TO: [int(f) for f in fanout], REV: [int(f) for f in fanout]},
        edge_label_index=(TO, snapshot[TO].edge_index[:, torch.from_numpy(seeds)]),
        batch_size=int(batch_size),
        shuffle=False,
        sampler=sampler,
        transform=transform,
        filter_per_worker=True,
        num_workers=workers,
        pin_memory=str(device).startswith("cuda"),
        **extra,
    )
