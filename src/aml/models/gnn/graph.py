"""The transaction graph in host memory (M3 spec §4; owner B).

`load_graph` reads the prepared transactions and the feature engine's parts once per container
and returns a HostGraph: topology in rank order, time = rank, per-gid side arrays, the derived
bounds, the 18 encoded GNN edge attributes and (selected) labels. The HeteroData handed to a
loader holds only `edge_index` and `time` per edge type plus the node count (`build_hetero`).

Edge ids: `gid` = global rank. The `to` store is in rank order, so its `e_id == gid ==` feature
row; the `rev_to` store holds the non-self-loop edges reversed, in rank order, with
`rev_gid[e_id]` = the forward gid. Every message edge carries its own engine row, computed as of
its own minute - 1, so every attribute in a causal subgraph is itself causal.

torch / PyG are imported inside the functions that need them (HostGraph is numpy only).
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from aml.models.gnn import (
    BOUND_KEYS,
    CAUSAL_BOUND_PROTOCOLS,
    FAITHFUL_EXEMPT_FEATURES,
    NODE,
    PROTOCOLS,
    REV,
    TEST_BOUNDS,
    TO,
    label_splits_for,
)

if TYPE_CHECKING:
    import polars as pl
    from torch_geometric.data import HeteroData

    from aml.features.spec import EngineSpec
    from aml.paths import DataPaths

# transactions.parquet columns the graph reads (§4.1)
TX_COLUMNS = ("row_id", "rank", "minute", "day", "split", "src", "dst")
MINUTES_PER_DAY = 1440
PREPARE_MARKER = "prepare_summary.json"
# HI-Small bounds (M3 spec §2.1), asserted when the loaded data is HI-Small with the PLAN split.
HI_SMALL_ROWS = 5_078_345
HI_SMALL_BOUNDS = {
    "train_last": 3_248_920,
    "val_last": 4_214_444,
    "d10_last": 5_077_236,
    "data_last": 5_078_344,
    "val_early_first": 3_248_921,
    "val_late_first": 3_731_672,
    "test_first": 4_214_445,
}
# Look-ahead (step 1) bound key per split (§5.2); test uses data_last ("end") or the d10 rule.
LOOKAHEAD_BOUND = {
    "train": "train_last",
    "val_early": "val_last",
    "val_late": "val_last",
    "test": "data_last",
}
# Engine node aggregates carry a window tag (_1d, _3d, _12h, _2d, _90m ...): never a GNN input.
_WINDOW_TAG = re.compile(r"_\d+[dhm](_|$)")


@dataclass(frozen=True, eq=False)
class HostGraph:
    """The graph of one container (host RAM, ~2-3 GB on HI-Small). Arrays are indexed by gid
    (= rank, 0..n_edges-1) unless noted; all are C-contiguous numpy arrays.

    n_nodes      accounts.parquet height (every account is a node; no stored node features)
    n_edges      M = transactions height
    src, dst     int64 (M,)   account ids, 0 <= id < n_nodes
    minute       int64 (M,)   non-decreasing in rank
    day          int16 (M,)   1-based simulated day
    split_code   int8  (M,)   index into aml.data.split.SPLITS
    row_id       int64 (M,)   the transactions/labels/scores key (row_id != rank)
    first_rank   int64 (M,)   first rank of the gid's minute (np.unique(minute, return_index))
    rev_gid      int64 (M - n_self_loops,)  flatnonzero(src != dst): rev_to e_id -> forward gid
    bounds       dict[str, int], keys BOUND_KEYS (aml.models.gnn), all ranks:
                   train_last       last rank of split train            (HI-Small 3,248,920)
                   val_last         last rank of split val_late = end of day 8   (4,214,444)
                   d10_last         last rank with day <= data_cfg test_views.primary[1]
                                                                          (5,077,236)
                   data_last        M - 1                                (5,078,344)
                   val_early_first  first rank of val_early              (3,248,921)
                   val_late_first   first rank of val_late               (3,731,672)
                   test_first       first rank of test                   (4,214,445)
                 load_graph asserts every split is non-empty and contiguous in rank order and that
                 the bounds match data_cfg's day ranges.
    ea           float32 (M, len(attr_columns)), row = gid: the encoded GNN edge attributes
                 (§4.3), column order = attr_columns. Categorical columns hold float(code), code
                 in -1..vocab-1 (the model embeds code + 1); log1p columns are log1p'd before the
                 z-score; flags pass as is.
    y            int8 (M,)   label of every gid whose split is in label_splits; -1 = not loaded
    label_splits splits whose labels were loaded (LABEL_SPLITS, or FAITHFUL_LABEL_SPLITS)
    attr_columns exactly EngineSpec.gnn_edge_attr_names (set and order)
    preprocess   {"mu": {col: float}, "sigma": {col: float} (after std_floor), "vocab_sizes":
                 {col: int}, "fitted_on": "train", "columns": [...attr_columns],
                 "num_idx": [ea column indices fed to the numeric Linear], "cat_idx": [ea column
                 indices of the categoricals, in graph.categorical order], "cat_sizes":
                 [vocab size + 1 per cat_idx entry (embedding rows; row 0 = unknown code -1)],
                 "log1p": [columns log1p'd before the z-score, in column order]}.
                 The z-scored columns are the keys of mu/sigma. Saved as graph_meta.json in every
                 run dir and inside checkpoints.
    features_digest  the feature table's parts digest (spec.parts_digest of the parts read; the
                     build summary's `features_digest` for unchanged parts)
    spec_hash        EngineSpec.spec_hash() of the parts
    data_version     the prepared data's version (the prepare marker's `data_version`; None if
                     the marker has none, e.g. the synthetic fixture)
    """

    n_nodes: int
    n_edges: int
    src: np.ndarray
    dst: np.ndarray
    minute: np.ndarray
    day: np.ndarray
    split_code: np.ndarray
    row_id: np.ndarray
    first_rank: np.ndarray
    rev_gid: np.ndarray
    bounds: dict[str, int]
    ea: np.ndarray
    y: np.ndarray
    label_splits: tuple[str, ...]
    attr_columns: tuple[str, ...]
    preprocess: dict
    features_digest: str
    spec_hash: str
    data_version: str | None


# --- loading ---------------------------------------------------------------------------------


def load_graph(
    paths: DataPaths,
    features_dir: Path,
    gnn_cfg: dict,
    *,
    data_cfg: dict,
    label_splits: tuple[str, ...],
    preprocess: dict | None = None,
    protocol: str = "causal",
    log: Callable[[str], None] | None = None,
) -> HostGraph:
    """Read transactions (`row_id, rank, minute, day, split, src, dst`), the feature parts
    (`rank, row_id` + exactly `spec.gnn_edge_attr_names`, polars column projection),
    feature_spec.json, vocab.json, accounts.parquet (height) and labels for `label_splits` only
    (`aml.models.lgbm.load_labels`), and build the HostGraph.

    data_cfg: configs/data.yaml (split day ranges and test_views.primary for `d10_last`).
    label_splits must equal `label_splits_for(protocol)`; "test" is never allowed.
    preprocess: given = frozen normalisation stats (perturbation tests, as M2 freezes vocab and
    hubs); None = fitted on train rows (`fit_preprocess`).
    protocol: selects the whitelist check (`assert_gnn_edge_inputs`); the faithful data
    (faithful.py) is built from this graph separately.

    Fails fast (ValueError / AssertionError) unless: parts `rank == arange(M)`; parts `row_id`
    equals transactions `row_id` per rank; `minute` non-decreasing in rank; `0 <= src, dst <
    n_nodes`; attribute names equal `spec.gnn_edge_attr_names` (set and order); every attribute
    finite; bounds equal data_cfg's day ranges. The job preconditions (require_data,
    require_stage_output(features), require_features_verified) are checked by the job.
    """
    import polars as pl

    from aml.data.split import SPLITS
    from aml.features.spec import FEATURE_SPEC_FILE, EngineSpec, parts_digest, scan_feature_table
    from aml.io import read_json
    from aml.models.lgbm import load_labels

    say = log or (lambda _msg: None)
    t0 = time.perf_counter()
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    label_splits = tuple(label_splits)
    if "test" in label_splits:
        raise ValueError("test labels are never read outside aml.eval")
    if label_splits != label_splits_for(protocol):
        raise ValueError(
            f"protocol {protocol!r} reads the labels of {label_splits_for(protocol)}, "
            f"got {label_splits}"
        )
    features_dir = Path(features_dir)

    spec = EngineSpec.from_json(read_json(features_dir / FEATURE_SPEC_FILE))
    names = tuple(spec.gnn_edge_attr_names)
    assert_gnn_edge_inputs(names, protocol, spec)

    tx = pl.scan_parquet(paths.transactions).select(list(TX_COLUMNS)).collect()
    m = tx.height
    if m == 0:
        raise ValueError(f"{paths.transactions} has no rows")
    rank = tx.get_column("rank").to_numpy()
    if not np.array_equal(rank, np.arange(m, dtype=rank.dtype)):
        raise ValueError(f"{paths.transactions}: rank is not 0..M-1 in file order")
    del rank

    table = scan_feature_table(features_dir, ["rank", "row_id", "split", *names]).collect()
    if table.height != m:
        raise ValueError(f"{features_dir}: {table.height} feature rows for {m} transactions")
    if not np.array_equal(table.get_column("rank").to_numpy(), np.arange(m)):
        raise ValueError(f"{features_dir}: ranks are not 0..M-1 in part order")
    row_id = np.ascontiguousarray(tx.get_column("row_id").to_numpy(), dtype=np.int64)
    if not np.array_equal(table.get_column("row_id").to_numpy(), row_id):
        raise ValueError(f"{features_dir}: row_id differs from transactions.parquet per rank")
    if not (table.get_column("split") == tx.get_column("split")).all():
        raise ValueError(f"{features_dir}: split differs from transactions.parquet per rank")
    bad = [c for c in names if table.schema[c] != pl.Float32]
    if bad:
        raise ValueError(f"GNN edge attributes must be Float32: {bad}")
    say(f"graph: read {m:,} transactions and parts ({time.perf_counter() - t0:.1f}s)")

    n_nodes = int(pl.scan_parquet(paths.accounts).select(pl.len()).collect().item())
    src = np.ascontiguousarray(tx.get_column("src").to_numpy(), dtype=np.int64)
    dst = np.ascontiguousarray(tx.get_column("dst").to_numpy(), dtype=np.int64)
    for name, a in (("src", src), ("dst", dst)):
        if a.min() < 0 or a.max() >= n_nodes:
            raise ValueError(f"{name} outside [0, {n_nodes}) (accounts.parquet height)")
    minute = np.ascontiguousarray(tx.get_column("minute").to_numpy(), dtype=np.int64)
    if np.any(minute[1:] < minute[:-1]):
        raise ValueError("minute is not non-decreasing in rank")
    day = np.ascontiguousarray(tx.get_column("day").to_numpy(), dtype=np.int16)
    # prepare's definition (ingest.assign_time); guard_bounds' d10 rule relies on day being a
    # function of minute (no minute straddles a day boundary)
    if not np.array_equal(day, minute // MINUTES_PER_DAY + 1):
        raise ValueError("day != minute // 1440 + 1 (prepare_data's definition)")
    split_names = tx.get_column("split")
    unknown = set(split_names.unique().to_list()) - set(SPLITS)
    if unknown:
        raise ValueError(f"unknown splits {sorted(unknown)}")
    split_code = np.ascontiguousarray(
        split_names.replace_strict(list(SPLITS), list(range(len(SPLITS))), return_dtype=pl.Int8)
        .to_numpy()
        .astype(np.int8)
    )
    del tx, split_names
    bounds = _derive_bounds(split_code, day, data_cfg)
    _check_hi_small(bounds, m, data_cfg)

    first_rank = _first_rank(minute)
    rev_gid = np.flatnonzero(src != dst).astype(np.int64)

    if preprocess is None:
        preprocess = fit_preprocess(table.select("split", *names), gnn_cfg, spec)
    else:
        preprocess = _check_frozen_preprocess(preprocess, names, spec)
    ea = encode_edge_attrs(table.select(*names), preprocess)
    del table
    say(f"graph: encoded {len(names)} edge attributes ({time.perf_counter() - t0:.1f}s)")

    y = np.full(m, -1, dtype=np.int8)
    codes = [SPLITS.index(s) for s in label_splits]
    keep = np.isin(split_code, codes)
    y[keep] = load_labels(paths.labels, pl.Series("row_id", row_id[keep]))
    if y[keep].size and not set(np.unique(y[keep]).tolist()) <= {0, 1}:
        raise ValueError("labels must be 0/1")

    marker = paths.parquet_dir / PREPARE_MARKER
    data_version = None
    if marker.exists():
        data_version = json.loads(marker.read_text(encoding="utf-8")).get("data_version")
    g = HostGraph(
        n_nodes=n_nodes,
        n_edges=m,
        src=src,
        dst=dst,
        minute=minute,
        day=day,
        split_code=split_code,
        row_id=row_id,
        first_rank=first_rank,
        rev_gid=rev_gid,
        bounds=bounds,
        ea=ea,
        y=y,
        label_splits=label_splits,
        attr_columns=names,
        preprocess=preprocess,
        features_digest=parts_digest(features_dir),
        spec_hash=spec.spec_hash(),
        data_version=data_version,
    )
    say(
        f"graph: {n_nodes:,} nodes, {m:,} edges ({m - len(rev_gid):,} self-loops), labels of "
        f"{label_splits} ({int(keep.sum()):,} rows), bounds {bounds} "
        f"({time.perf_counter() - t0:.1f}s)"
    )
    return g


def _first_rank(minute: np.ndarray) -> np.ndarray:
    """First rank of each gid's minute (minute is non-decreasing): the start of its run."""
    m = len(minute)
    start = np.ones(m, dtype=bool)
    start[1:] = minute[1:] != minute[:-1]
    return np.maximum.accumulate(np.where(start, np.arange(m, dtype=np.int64), 0))


def _derive_bounds(split_code: np.ndarray, day: np.ndarray, data_cfg: dict) -> dict[str, int]:
    """BOUND_KEYS from the per-gid split codes; asserts every split is non-empty, contiguous in
    rank order (in SPLITS order) and equals data_cfg's day ranges, and derives d10_last from
    test_views.primary."""
    from aml.data.split import SPLITS, split_days, view_days

    m = len(split_code)
    if np.any(split_code[1:] < split_code[:-1]):
        raise ValueError("splits are not contiguous in rank order (SPLITS order)")
    first, last = {}, {}
    for i, s in enumerate(SPLITS):
        idx = np.flatnonzero(split_code == i)
        if not len(idx):
            raise ValueError(f"split {s!r} has no rows")
        first[s], last[s] = int(idx[0]), int(idx[-1])
        lo, hi = split_days(data_cfg, s)
        in_range = (day >= lo) & (day <= hi)
        if not np.array_equal(in_range, split_code == i):
            raise ValueError(f"split {s!r} rows differ from data.yaml's day range [{lo}, {hi}]")
    p_hi = view_days(data_cfg, "primary")[1]
    primary = np.flatnonzero(day <= p_hi)
    return {
        "train_last": last["train"],
        "val_last": last["val_late"],
        "d10_last": int(primary[-1]) if len(primary) else -1,
        "data_last": m - 1,
        "val_early_first": first["val_early"],
        "val_late_first": first["val_late"],
        "test_first": first["test"],
    }


def _check_hi_small(bounds: Mapping[str, int], m: int, data_cfg: dict) -> None:
    """On HI-Small with the PLAN split, the derived bounds must equal M3 spec §2.1."""
    if data_cfg.get("dataset", {}).get("name") != "hi_small" or m != HI_SMALL_ROWS:
        return
    plan_split = {"train": [1, 6], "val_early": [7, 7], "val_late": [8, 8], "test": [9, 18]}
    if {k: list(v) for k, v in data_cfg["split"].items()} != plan_split:
        return
    keys = [k for k in BOUND_KEYS if k != "d10_last"]
    if list(data_cfg["test_views"]["primary"]) == [9, 10]:
        keys.append("d10_last")
    bad = {k: (bounds[k], HI_SMALL_BOUNDS[k]) for k in keys if bounds[k] != HI_SMALL_BOUNDS[k]}
    if bad:
        raise ValueError(f"HI-Small bounds differ from M3 spec §2.1 (got, want): {bad}")


# --- edge attributes -------------------------------------------------------------------------


def fit_preprocess(table: pl.DataFrame, gnn_cfg: dict, spec: EngineSpec) -> dict:
    """Normalisation stats (the `preprocess` dict of HostGraph) from `table`'s rows with
    `split == "train"` only: log1p first for `graph.log1p` columns, then float64 mean and
    population std (ddof 0) of `graph.zscore` columns (sigma = max(std, std_floor)); vocab sizes
    from the spec's vocab. Every column that is neither categorical nor z-scored must be a 0/1
    flag of the spec (domain "flag")."""
    gcfg = gnn_cfg["graph"]
    if gcfg["norm_split"] != "train":
        raise ValueError("graph.norm_split must be 'train' (PLAN §4)")
    columns = list(spec.gnn_edge_attr_names)
    cat = [str(c) for c in gcfg["categorical"]]
    log1p = {str(c) for c in gcfg["log1p"]}
    zs = {str(c) for c in gcfg["zscore"]}
    unknown = sorted((set(cat) | log1p | zs) - set(columns))
    if unknown:
        raise ValueError(f"graph config names columns that are not GNN edge attributes: {unknown}")
    spec_cat = {c for c in columns if spec.feature(c).categorical}
    if set(cat) != spec_cat:
        raise ValueError(f"graph.categorical {sorted(cat)} != the spec's categoricals {spec_cat}")
    if set(cat) & (log1p | zs):
        raise ValueError("categorical columns cannot be transformed")
    loose = [c for c in columns if c not in spec_cat and c not in zs]
    not_flags = [c for c in loose if spec.feature(c).domain != "flag"]
    if not_flags:
        raise ValueError(f"columns neither categorical nor z-scored must be flags: {not_flags}")
    missing = [c for c in ("split", *columns) if c not in table.columns]
    if missing:
        raise ValueError(f"table lacks columns {missing}")

    train = table.filter(table.get_column("split") == "train")
    if train.height == 0:
        raise ValueError("no train rows: cannot fit the GNN edge-attribute normalisation")
    floor = float(gcfg["std_floor"])
    mu: dict[str, float] = {}
    sigma: dict[str, float] = {}
    for c in columns:
        if c not in zs:
            continue
        v = train.get_column(c).to_numpy().astype(np.float64)
        if c in log1p:
            v = np.log1p(v)
        if not np.isfinite(v).all():
            raise ValueError(f"column {c!r} has non-finite train values (after log1p)")
        mu[c] = float(v.mean())
        sigma[c] = float(max(float(v.std()), floor))
    vocab_sizes = {c: len(spec.vocab[c]) for c in cat}
    return {
        "mu": mu,
        "sigma": sigma,
        "vocab_sizes": vocab_sizes,
        "fitted_on": "train",
        "columns": columns,
        "num_idx": [i for i, c in enumerate(columns) if c not in spec_cat],
        "cat_idx": [columns.index(c) for c in cat],
        "cat_sizes": [vocab_sizes[c] + 1 for c in cat],
        "log1p": [c for c in columns if c in log1p],
    }


def _check_frozen_preprocess(pre: dict, names: Sequence[str], spec: EngineSpec) -> dict:
    keys = ("mu", "sigma", "vocab_sizes", "columns", "num_idx", "cat_idx", "cat_sizes", "log1p")
    missing = [k for k in keys if k not in pre]
    if missing:
        raise ValueError(f"frozen preprocess lacks {missing}")
    if list(pre["columns"]) != list(names):
        raise ValueError("frozen preprocess columns differ from the spec's GNN edge attributes")
    sizes = {c: len(spec.vocab[c]) for c in pre["vocab_sizes"]}
    if sizes != dict(pre["vocab_sizes"]):
        raise ValueError(f"frozen vocab sizes {pre['vocab_sizes']} != the parts' {sizes}")
    return pre


def encode_edge_attrs(table: pl.DataFrame, preprocess: dict) -> np.ndarray:
    """(rows, len(columns)) float32 in preprocess["columns"] order: categoricals as float(code)
    (integer codes in -1..vocab-1, checked), log1p then z-score as configured (float64 math),
    flags as is. Rows in the table's (rank) order. Raises ValueError on a non-finite value."""
    columns = list(preprocess["columns"])
    missing = [c for c in columns if c not in table.columns]
    if missing:
        raise ValueError(f"table lacks columns {missing}")
    # categorical column -> vocab size (embedding rows - 1)
    cat_pairs = zip(preprocess["cat_idx"], preprocess["cat_sizes"], strict=True)
    cats = {columns[i]: int(n) - 1 for i, n in cat_pairs}
    log1p = set(preprocess.get("log1p", ()))
    mu, sigma = preprocess["mu"], preprocess["sigma"]
    out = np.empty((table.height, len(columns)), dtype=np.float32)
    for i, c in enumerate(columns):
        v = table.get_column(c).to_numpy()
        if c in cats:
            if v.size and (v.min() < -1 or v.max() > cats[c] - 1 or np.any(v != np.round(v))):
                raise ValueError(f"categorical {c!r} codes outside -1..{cats[c] - 1}")
            out[:, i] = v
            continue
        v = v.astype(np.float64)
        if c in log1p:
            v = np.log1p(v)
        if c in mu:
            v = (v - float(mu[c])) / float(sigma[c])
        if not np.isfinite(v).all():
            raise ValueError(f"column {c!r} has non-finite encoded values")
        out[:, i] = v
    return out


# --- the loader's HeteroData ------------------------------------------------------------------


def build_hetero(
    g: HostGraph,
    *,
    temporal: bool = True,
    last_rank: int | None = None,
    reverse_self_loops: bool = False,
) -> HeteroData:
    """The CPU HeteroData a loader samples from. Holds ONLY (whitelist test):
    `[NODE].num_nodes`, and per edge type `edge_index` (int64, rank order) and (temporal) `time`
    (int64, = forward rank, strictly increasing within each type).

    TO = all edges [src; dst]; REV = the non-self-loop edges [dst; src] (reverse_self_loops=True:
    all edges flipped, as Multi-GNN's snapshots; then rev e_id == gid). last_rank: keep only
    ranks <= last_rank (faithful snapshots; None = all). temporal=False omits `time`.
    """
    import torch
    from torch_geometric.data import HeteroData

    n = g.n_edges if last_rank is None else int(last_rank) + 1
    if not 0 < n <= g.n_edges:
        raise ValueError(f"last_rank {last_rank} outside [0, {g.n_edges - 1}]")
    src = torch.from_numpy(np.ascontiguousarray(g.src[:n], dtype=np.int64))
    dst = torch.from_numpy(np.ascontiguousarray(g.dst[:n], dtype=np.int64))
    data = HeteroData()
    data[NODE].num_nodes = int(g.n_nodes)
    data[TO].edge_index = torch.stack([src, dst])
    if reverse_self_loops:
        data[REV].edge_index = torch.stack([dst, src])
        rev_time = torch.arange(n, dtype=torch.int64)
    else:
        rg = np.ascontiguousarray(g.rev_gid[: int(np.searchsorted(g.rev_gid, n))], np.int64)
        if len(rg) > 1 and not np.all(rg[1:] > rg[:-1]):
            raise ValueError("rev_gid must be strictly increasing (rank order)")
        rev_time = torch.from_numpy(rg)
        data[REV].edge_index = torch.stack([dst[rev_time], src[rev_time]])
    if temporal:
        data[TO].time = torch.arange(n, dtype=torch.int64)
        data[REV].time = rev_time
    return data


# --- seeds, label times, guard bounds, labels ------------------------------------------------


def split_gids(g: HostGraph, split: str) -> np.ndarray:
    """int64 gids of one split in rank order (eval seed arrays; never down-sampled)."""
    from aml.data.split import SPLITS

    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return np.flatnonzero(g.split_code == SPLITS.index(split)).astype(np.int64)


def _gids(g: HostGraph, gids: Any) -> np.ndarray:
    a = np.asarray(gids)
    if a.ndim != 1 or (a.size and not np.issubdtype(a.dtype, np.integer)):
        raise ValueError("gids must be a 1-D integer array")
    a = a.astype(np.int64, copy=False)
    if a.size and (a.min() < 0 or a.max() >= g.n_edges):
        raise ValueError(f"gids outside [0, {g.n_edges})")
    return a


def _check_bound_args(protocol: str, test_bound: str) -> None:
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    if protocol == "faithful":
        raise ValueError("faithful samples non-temporal snapshots: it has no label times")
    if test_bound not in TEST_BOUNDS:
        raise ValueError(f"unknown test bound {test_bound!r}; expected one of {TEST_BOUNDS}")


def label_times(
    g: HostGraph, gids: np.ndarray, *, protocol: str, test_bound: str = "end"
) -> np.ndarray:
    """int64 `edge_label_time` per seed gid (M3 spec §5.2; never hardcoded):

    causal / pna (also HPO, bench, dev): first_rank[g] - 1 (-1 for the first minute: an empty
        subgraph, F10). pyg-lib's filter is inclusive, so every same-minute edge is excluded.
    lookahead: train -> bounds.train_last; val_early, val_late -> bounds.val_last;
        test & test_bound "end" -> bounds.data_last;
        test & test_bound "d10" -> max(bounds.d10_last, first_rank[g] - 1) (tail rows get their
        causal bound, so every test row is scored).
    faithful: raises ValueError (non-temporal snapshots).
    """
    from aml.data.split import SPLITS

    _check_bound_args(protocol, test_bound)
    gids = _gids(g, gids)
    if protocol in CAUSAL_BOUND_PROTOCOLS:
        return (g.first_rank[gids] - 1).astype(np.int64)
    out = np.empty(len(gids), dtype=np.int64)
    codes = g.split_code[gids]
    for i, s in enumerate(SPLITS):
        sel = codes == i
        if s == "test" and test_bound == "d10":
            out[sel] = np.maximum(g.bounds["d10_last"], g.first_rank[gids[sel]] - 1)
        else:
            out[sel] = g.bounds[LOOKAHEAD_BOUND[s]]
    return out


def guard_bounds(
    g: HostGraph, gids: np.ndarray, *, protocol: str, test_bound: str = "end"
) -> np.ndarray:
    """The same bounds as `label_times`, computed by an INDEPENDENT code path for the runtime
    guard (M3 spec §5.5); must never call label_times (tested):
    causal / pna: np.searchsorted(g.minute, g.minute[gids], side="left") - 1;
    lookahead: the split bound looked up from split_code (train -> train_last, val -> val_last,
    test -> data_last or, for "d10", d10_last for targets on a day <= the day of d10_last and
    the searchsorted causal bound for later (tail) days).
    """
    from aml.data.split import SPLITS

    _check_bound_args(protocol, test_bound)
    gids = _gids(g, gids)
    if protocol in CAUSAL_BOUND_PROTOCOLS:
        return np.searchsorted(g.minute, g.minute[gids], side="left").astype(np.int64) - 1
    b = g.bounds
    by_code = np.array([b[LOOKAHEAD_BOUND[s]] for s in SPLITS], dtype=np.int64)
    out = by_code[g.split_code[gids].astype(np.int64)]
    if test_bound == "d10":
        test = g.split_code[gids] == SPLITS.index("test")
        last_day = int(g.day[b["d10_last"]]) if b["d10_last"] >= 0 else -1
        tail = test & (g.day[gids].astype(np.int64) > last_day)
        out[test & ~tail] = b["d10_last"]
        causal = np.searchsorted(g.minute, g.minute[gids[tail]], side="left").astype(np.int64)
        out[tail] = causal - 1
    return out


def labels_for(g: HostGraph, gids: np.ndarray) -> np.ndarray:
    """int8 labels of `gids`; ValueError if any gid's label was not loaded (y == -1)."""
    from aml.data.split import SPLITS

    gids = _gids(g, gids)
    y = g.y[gids]
    bad = y < 0
    if bad.any():
        splits = sorted({SPLITS[int(c)] for c in np.unique(g.split_code[gids[bad]])})
        raise ValueError(
            f"{int(bad.sum())} labels were not loaded (splits {splits}; loaded {g.label_splits})"
        )
    return y.astype(np.int8, copy=True)


# --- whitelist and PNA degrees ----------------------------------------------------------------


def assert_gnn_edge_inputs(names: Any, protocol: str, spec: EngineSpec) -> None:
    """Whitelist (M3 spec §14): non-faithful protocols use exactly spec.gnn_edge_attr_names (set
    and order); no label / rank / minute / row_id / id column and no engine node aggregate
    (`_1d/_3d/_12h/_2d` names). Faithful may use FAITHFUL_EXEMPT_FEATURES (and only then): its
    names must be GNN edge attributes or Multi-GNN's columns (faithful.FAITHFUL_COLUMNS).
    Raises AssertionError on violation."""
    from aml.features.tx_features import is_forbidden

    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    names = [str(n) for n in names]
    allowed = tuple(spec.gnn_edge_attr_names)
    exempt = FAITHFUL_EXEMPT_FEATURES if protocol == "faithful" else ()
    bad = [
        n
        for n in names
        if n not in exempt
        and (
            is_forbidden(n)
            or n == "id"
            or n.endswith("_id")
            or _WINDOW_TAG.search(n) is not None
            or n in {"y", "is_laundering", "label"}
        )
    ]
    if len(set(names)) != len(names):
        bad.append(f"duplicates in {names}")
    if protocol == "faithful":
        from aml.models.gnn.faithful import FAITHFUL_COLUMNS

        outside = [n for n in names if n not in allowed and n not in FAITHFUL_COLUMNS]
        if outside:
            bad.append(f"not GNN edge attributes or Multi-GNN columns: {outside}")
    elif tuple(names) != allowed:
        bad.append(f"{names} != spec.gnn_edge_attr_names {list(allowed)} (set and order)")
    if bad:
        raise AssertionError(f"GNN edge inputs not allowed under {protocol!r}: {bad}")


def train_degree_histograms(g: HostGraph) -> tuple[np.ndarray, np.ndarray]:
    """PNA `deg` histograms (int64 bincounts of in-degrees) from TRAIN edges only:
    (forward in-degree over `to`, in-degree over `rev_to` = out-degree without self-loops).
    Every account counts (degree 0 included), as PyG's PNA example does. The source is ours
    (the full train graph, one histogram per direction); Multi-GNN recalled (unverified): one
    sampled train batch, both directions pooled into one histogram. It only sets PNA's
    avg_deg scaler constants (absorbed by the post-layer Linear)."""
    from aml.data.split import SPLITS

    train = g.split_code == SPLITS.index("train")
    fwd_in = np.bincount(g.dst[train], minlength=g.n_nodes)
    rev_in = np.bincount(g.src[train & (g.src != g.dst)], minlength=g.n_nodes)
    return np.bincount(fwd_in).astype(np.int64), np.bincount(rev_in).astype(np.int64)
