"""Multi-GNN's published setup, data side (M3 spec §9; owner B). Protocol "faithful" ONLY.

Snapshot graphs: train = ranks <= bounds.train_last, val = <= bounds.val_last, test = all; the
reverse store flips ALL edges (self-loops included), both stores in rank order, so `e_id == gid`
on both types. Edge features EA_f (6 columns, Multi-GNN's set, recalled): `timestamp` (minutes
since the first event), `amount_received` (raw), `receiving_currency`, `payment_format` (numeric
train-vocab codes in sorted vocab order: ours; Multi-GNN recalled as first-appearance codes,
unverified; the order matters because the codes are fed as numbers), `mg_in_port`,
`mg_out_port`; z-scored per snapshot over the snapshot's own edges. `timestamp`
(FAITHFUL_EXEMPT_FEATURES) and the per-snapshot normalisation (FAITHFUL_EXEMPT_NORM) are the
two confined exemptions from PLAN §4: every builder takes
`protocol` and raises ValueError unless it is "faithful". Nothing from this run enters the model
comparison.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from aml.models.gnn import FAITHFUL_EXEMPT_NORM

if TYPE_CHECKING:
    import polars as pl
    from torch_geometric.data import HeteroData

    from aml.models.gnn.graph import HostGraph
    from aml.paths import DataPaths

# EA_f column order (= gnn.yaml protocols.faithful.edge_features)
FAITHFUL_COLUMNS = (
    "timestamp",
    "amount_received",
    "receiving_currency",
    "payment_format",
    "mg_in_port",
    "mg_out_port",
)
SNAPSHOTS = ("train", "val", "test")  # last rank: bounds.train_last, bounds.val_last, data_last
CODED_COLUMNS = ("receiving_currency", "payment_format")  # train-vocab codes, -1 unknown
# transactions.parquet columns the faithful features read
FAITHFUL_TX_COLUMNS = ("rank", "minute", "src", "dst", "amount_received", *CODED_COLUMNS)
SNAPSHOT_STD_FLOOR = 1e-6


def require_faithful(protocol: str) -> None:
    """ValueError unless protocol == "faithful" (the exemptions are confined to it)."""
    if protocol != "faithful":
        raise ValueError(
            f"Multi-GNN's snapshot data (timestamp feature, {FAITHFUL_EXEMPT_NORM} "
            f"normalisation) is confined to protocol 'faithful', got {protocol!r}"
        )


def snapshot_last_ranks(g: HostGraph) -> dict[str, int]:
    """{"train": bounds.train_last, "val": bounds.val_last, "test": bounds.data_last}."""
    b = g.bounds
    return {"train": int(b["train_last"]), "val": int(b["val_last"]), "test": int(b["data_last"])}


def multignn_ports(
    src: np.ndarray, dst: np.ndarray, rank: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Multi-GNN port numbers per edge (int64, 0-based), (in_port, out_port): for each (u, v)
    pair its first rank; the in-port of u->v = the ordinal of first_rank(u, v) among v's
    distinct in-neighbours ordered by first rank (rank tie-break); the out-port likewise at u
    among u's distinct out-neighbours. Prefix-stable, so computed once over all edges.
    Returned in the input order (edges need not be passed in rank order; ranks must be
    unique)."""
    src = np.asarray(src, dtype=np.int64).reshape(-1)
    dst = np.asarray(dst, dtype=np.int64).reshape(-1)
    rank = np.asarray(rank, dtype=np.int64).reshape(-1)
    m = len(src)
    if len(dst) != m or len(rank) != m:
        raise ValueError("src, dst and rank must have one entry per edge")
    if m == 0:
        return np.empty(0, np.int64), np.empty(0, np.int64)
    if min(int(src.min()), int(dst.min())) < 0:
        raise ValueError("account ids must be >= 0")
    order = np.argsort(rank, kind="stable")
    if np.any(rank[order][1:] == rank[order][:-1]):
        raise ValueError("ranks must be unique")
    s, d = src[order], dst[order]
    n = int(max(int(s.max()), int(d.max()))) + 1
    # first_pos = position (in rank order) of each distinct pair's first edge
    _, first_pos, inv = np.unique(s * n + d, return_index=True, return_inverse=True)
    pu, pv = s[first_pos], d[first_pos]
    in_pair = _ordinal_within(pv, first_pos)
    out_pair = _ordinal_within(pu, first_pos)
    in_port = np.empty(m, np.int64)
    out_port = np.empty(m, np.int64)
    in_port[order] = in_pair[inv.reshape(-1)]
    out_port[order] = out_pair[inv.reshape(-1)]
    return in_port, out_port


def _ordinal_within(group: np.ndarray, key: np.ndarray) -> np.ndarray:
    """0-based rank of each item's `key` among the items of its `group` (keys unique)."""
    o = np.lexsort((key, group))
    g = group[o]
    start = np.ones(len(g), dtype=bool)
    start[1:] = g[1:] != g[:-1]
    pos = np.arange(len(g), dtype=np.int64)
    ordinal = pos - np.maximum.accumulate(np.where(start, pos, 0))
    out = np.empty(len(g), np.int64)
    out[o] = ordinal
    return out


def _codes(values: pl.Series, categories: list[str]) -> np.ndarray:
    import polars as pl

    return (
        values.replace_strict(
            list(categories), list(range(len(categories))), default=-1, return_dtype=pl.Int64
        )
        .fill_null(-1)
        .to_numpy()
        .astype(np.float64)
    )


def faithful_edge_attrs(
    g: HostGraph, transactions: pl.DataFrame, vocab: dict, *, protocol: str
) -> pl.DataFrame:
    """The raw (unnormalised) EA_f frame in rank order, columns FAITHFUL_COLUMNS (Float64).
    transactions: rank-ordered `rank, amount_received, receiving_currency, payment_format`
    (+ optional minute, src, dst, cross-checked against g); vocab: the feature table's
    vocab.json (train codes, -1 unknown). timestamp = minute - the first event's minute; ports
    from multignn_ports over g's edges in rank order. Raises ValueError unless protocol ==
    "faithful"."""
    import polars as pl

    require_faithful(protocol)
    need = ("rank", "amount_received", *CODED_COLUMNS)
    missing = [c for c in need if c not in transactions.columns]
    if missing:
        raise ValueError(f"transactions lack columns {missing}")
    m = g.n_edges
    if transactions.height != m:
        raise ValueError(f"{transactions.height} transactions for a graph of {m} edges")
    if not np.array_equal(transactions.get_column("rank").to_numpy(), np.arange(m)):
        raise ValueError("transactions must be in rank order (rank == 0..M-1)")
    for c, want in (("minute", g.minute), ("src", g.src), ("dst", g.dst)):
        if c in transactions.columns and not np.array_equal(
            transactions.get_column(c).to_numpy(), want
        ):
            raise ValueError(f"transactions.{c} differs from the graph's")
    missing = [c for c in CODED_COLUMNS if c not in vocab]
    if missing:
        raise ValueError(f"vocab lacks {missing}")
    amount = transactions.get_column("amount_received").cast(pl.Float64).to_numpy()
    if not np.isfinite(amount).all():
        raise ValueError("amount_received has non-finite values")
    in_port, out_port = multignn_ports(g.src, g.dst, np.arange(m, dtype=np.int64))
    cols = {
        "timestamp": (g.minute - g.minute[0]).astype(np.float64),
        "amount_received": amount,
        "receiving_currency": _codes(
            transactions.get_column("receiving_currency"), vocab["receiving_currency"]
        ),
        "payment_format": _codes(
            transactions.get_column("payment_format"), vocab["payment_format"]
        ),
        "mg_in_port": in_port.astype(np.float64),
        "mg_out_port": out_port.astype(np.float64),
    }
    return pl.DataFrame({c: pl.Series(c, cols[c], dtype=pl.Float64) for c in FAITHFUL_COLUMNS})


def load_faithful_raw(
    g: HostGraph, paths: DataPaths, features_dir: Path, *, protocol: str
) -> pl.DataFrame:
    """faithful_edge_attrs over the prepared transactions (FAITHFUL_TX_COLUMNS, column
    projection) and the feature table's vocab.json. Raises unless protocol == "faithful"."""
    import polars as pl

    from aml.features.spec import VOCAB_FILE
    from aml.io import read_json

    require_faithful(protocol)
    tx = pl.scan_parquet(paths.transactions).select(list(FAITHFUL_TX_COLUMNS)).collect()
    vocab = read_json(Path(features_dir) / VOCAB_FILE)
    return faithful_edge_attrs(g, tx, vocab, protocol=protocol)


def snapshot_hetero(g: HostGraph, last_rank: int) -> HeteroData:
    """The snapshot's HeteroData: ranks <= last_rank, TO = [src; dst], REV = all of them
    flipped (self-loops included), both in rank order (e_id == gid), no `time`, plus
    `[NODE].num_nodes` (= graph.build_hetero(g, temporal=False, last_rank=last_rank,
    reverse_self_loops=True))."""
    from aml.models.gnn.graph import build_hetero

    return build_hetero(g, temporal=False, last_rank=last_rank, reverse_self_loops=True)


def snapshot_ea(raw: pl.DataFrame, last_rank: int, *, protocol: str) -> np.ndarray:
    """float32 (last_rank + 1, 6): rows 0..last_rank of `raw`, every column z-scored with the
    mean and std of those rows (float64; std with ddof 1 as torch.std, floored at 1e-6).
    Raises ValueError unless protocol == "faithful"."""
    require_faithful(protocol)
    if tuple(raw.columns) != FAITHFUL_COLUMNS:
        raise ValueError(f"raw EA_f columns {raw.columns} != {list(FAITHFUL_COLUMNS)}")
    n = int(last_rank) + 1
    if not 0 < n <= raw.height:
        raise ValueError(f"last_rank {last_rank} outside [0, {raw.height - 1}]")
    out = np.empty((n, len(FAITHFUL_COLUMNS)), dtype=np.float32)
    for i, c in enumerate(FAITHFUL_COLUMNS):
        v = raw.get_column(c).to_numpy()[:n].astype(np.float64)
        sd = float(v.std(ddof=1)) if n > 1 else 0.0
        z = (v - float(v.mean())) / max(sd, SNAPSHOT_STD_FLOOR)
        if not np.isfinite(z).all():
            raise ValueError(f"EA_f column {c!r} is not finite after normalisation")
        out[:, i] = z
    return out


def faithful_preprocess() -> dict:
    """The model-column layout of EA_f (build_model's `preprocess`): 6 numeric columns, no
    categoricals (Multi-GNN feeds its codes as numbers), normalisation per snapshot."""
    k = len(FAITHFUL_COLUMNS)
    return {
        "columns": list(FAITHFUL_COLUMNS),
        "num_idx": list(range(k)),
        "cat_idx": [],
        "cat_sizes": [],
        "norm": FAITHFUL_EXEMPT_NORM,
        "fitted_on": "snapshot",
    }
