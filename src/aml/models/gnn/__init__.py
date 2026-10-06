"""Causal Multi-GINe+EU (M3): shared constants, errors, guard counters and config validation.

**Torch-free by contract**: modal_jobs, the laptop and `costplan` import this module without
torch (tests/unit/test_modal_jobs.py imports it with torch blocked). Module map (M3 spec §4-§10):

- `graph.py`      HostGraph (topology, time, bounds, encoded edge attributes), label times
- `sampler.py`    the temporal LinkNeighborLoader and the per-epoch negative subset
- `transforms.py` FlatBatch, the runtime as-of guard, the edge-cap split, the eval-tree cache
- `faithful.py`   Multi-GNN's snapshot data (the confined exemptions)
- `model.py`      MultiGINe (GINE or PNA convs, edge updates, virtual target edge)
- `train.py`      run_set: seeds of one protocol, checkpoints, scoring, set assembly
- `hpo.py`        run_hpo: the resumable Optuna search
- `bench.py`      gnn_bench cells on real batches
- `costplan.py`   $ formulas, the bench decision, the plan and the cost gate (torch-free)

Only `src/aml/models/gnn/` imports torch, and modal_jobs import these modules lazily.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

# Bumped by hand on any change of GNN semantics (graph encoding, sampler, model, loss). Like the
# feature engine's ENGINE_VERSION it is the GNN's explicit opt-in to code in the run keys.
GNN_VERSION = 1

# --- graph vocabulary ------------------------------------------------------------------------

NODE = "acct"
TO = (NODE, "to", NODE)  # every transaction, in rank order: e_id == gid == rank
REV = (NODE, "rev_to", NODE)  # non-self-loop transactions reversed; time = the forward rank

# Training protocols = the keys of gnn.yaml `protocols`. HPO, bench and the dev run sample with
# the causal bound and pass protocol="causal" (the dev run's set lives under DEV_KIND).
PROTOCOLS = ("causal", "lookahead", "pna", "faithful")
CAUSAL_BOUND_PROTOCOLS = ("causal", "pna")  # label time = first rank of the minute - 1
TEMPORAL_PROTOCOLS = ("causal", "lookahead", "pna")  # disjoint temporal trees (faithful: not)
TEST_BOUNDS = ("end", "d10")  # look-ahead test bound: end of data, last rank of the primary view
LOADER_ROLES = ("train", "eval")

# Splits whose labels a protocol may read (test labels are read only in aml.eval).
LABEL_SPLITS = ("train", "val_early")
FAITHFUL_LABEL_SPLITS = ("train", "val_early", "val_late")  # selection on days 7-8 (§9)

# HostGraph.bounds keys (graph.py documents each).
BOUND_KEYS = (
    "train_last",
    "val_last",
    "d10_last",
    "data_last",
    "val_early_first",
    "val_late_first",
    "test_first",
)

# --- Volume directory kinds (/data/models/<kind>/<key>/, M3 spec §3.3) --------------------

RUN_KIND = "gnn"  # one (protocol, seed) run
BENCH_KIND = "gnn_bench"
HPO_KIND = "gnn_hpo"
DEV_KIND = "gnn_dev"  # the dev run's set: val scores only, never evaluated
SET_KINDS = {
    "causal": "gnn_causal",
    "lookahead": "gnn_lookahead",  # test bound: end of data
    "pna": "gnn_pna",
    "faithful": "gnn_faithful",
}
LOOKAHEAD_D10_MODEL = "gnn_lookahead_d10"  # same val scores; test bound d10 (primary view only)
FAITHFUL_MODEL = "gnn_faithful"  # never in the model comparison (asserted in evaluate)
# The GNN models that enter evaluate's comparison, in report order.
COMPARISON_MODELS = ("gnn_causal", "gnn_lookahead", "gnn_lookahead_d10", "gnn_pna")
PRIMARY_ONLY_MODELS = (LOOKAHEAD_D10_MODEL,)  # evaluated on every view, rendered on primary only
DIR_KINDS = (
    RUN_KIND,
    BENCH_KIND,
    HPO_KIND,
    DEV_KIND,
    *SET_KINDS.values(),
    LOOKAHEAD_D10_MODEL,
)

# --- file names -------------------------------------------------------------------------------

SUMMARY_FILE = "summary.json"  # every stage dir; written LAST (the completion marker)
SCORES_FILE = "scores.parquet"  # set dir: row_id Int64, split String, score_s<k> Float64
WRITER_FILE = "writer.json"  # set dir: the writer lease {call_id, heartbeat}
STOPPED_FILE = "STOPPED.json"  # set dir: written by the driver when the set's wall guard fired
# run dir (/data/models/gnn/<run_key>/)
FINGERPRINT_FILE = "checkpoint.json"
GRAPH_META_FILE = "graph_meta.json"  # the preprocess dict (+ attr columns) of the run
LAST_CKPT = "last.pt"
BEST_CKPT = "best.pt"
HISTORY_FILE = "history.jsonl"
RUNNING_FILE = "running.json"  # {epoch_next, starts}: the unclean-start counter
FAILED_FILE = "FAILED.json"  # {error, traceback, epoch, ...}: a deterministic failure
SEED_SUMMARY_FILE = "seed_summary.json"
# bench dir
CELLS_FILE = "cells.jsonl"
BENCH_FILE = "bench.json"
DECISION_FILE = "decision.json"
PLAN_FILE = "plan.json"
BENCH_REPORT = "gnn_bench.md"  # under /data/reports/
# HPO dir
TRIALS_FILE = "trials.json"
BEST_PARAMS_FILE = "best_params.json"  # {lr, final_dropout, w_pos} only
PARAMS_FILE = "params.json"  # trial_<n>/params.json, written before the trial trains


def score_column(seed: int) -> str:
    """Score column of a seed; equals aml.models.lgbm.score_column (the evaluate layout)."""
    return f"score_s{int(seed)}"


def scores_file(seed: int, test_bound: str = "end") -> str:
    """A run dir's score file: scores_s<k>.parquet, or scores_d10_s<k>.parquet (look-ahead d10)."""
    if test_bound not in TEST_BOUNDS:
        raise ValueError(f"unknown test bound {test_bound!r}; expected one of {TEST_BOUNDS}")
    return (
        f"scores_s{int(seed)}.parquet"
        if test_bound == "end"
        else f"scores_d10_s{int(seed)}.parquet"
    )


def trial_dir_name(number: int) -> str:
    """An HPO trial's run dir under the HPO dir: trial_<n>/."""
    return f"trial_{int(number)}"


def set_kind(protocol: str, *, dev: bool = False) -> str:
    """The set directory kind of a protocol (the dev run's set is DEV_KIND)."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    return DEV_KIND if dev else SET_KINDS[protocol]


def label_splits_for(protocol: str) -> tuple[str, ...]:
    """The splits whose labels `protocol` may load (faithful: + val_late, its exemption)."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    return FAITHFUL_LABEL_SPLITS if protocol == "faithful" else LABEL_SPLITS


# --- faithful exemptions (M3 spec §9; PLAN §6) ------------------------------------------------

# The only edge feature outside the §4 whitelist, and only under protocol "faithful".
FAITHFUL_EXEMPT_FEATURES = ("timestamp",)
# The second confined exemption: z-score per snapshot (stats over the snapshot's own edges).
FAITHFUL_EXEMPT_NORM = "per_snapshot"

# --- run status (train.run_set, hpo.run_hpo; the driver loops on "partial") -------------------

RUN_STATUSES = ("done", "partial", "failed", "stopped", "busy")
DRIVER_STOP_STATUSES = ("done", "failed", "stopped", "busy")  # the driver stops calling the worker

# --- errors -----------------------------------------------------------------------------------


class LeakError(RuntimeError):
    """The runtime as-of guard (M3 spec §5.5) found a sampled edge later than its target's bound,
    the target itself in a causal subgraph, an edge crossing subgraphs, or a faithful edge beyond
    its snapshot. Deterministic: never retried; the run fails fast (FAILED.json).

    `detail` holds the counts (GUARD_FIELDS) and the offending seed positions / gids."""

    def __init__(self, message: str, detail: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.detail = dict(detail or {})


class GnnStopError(RuntimeError):
    """A pre-registered stop: the cost gate, the workspace-budget pre-check, a bench decision rule
    (e.g. the determinism overhead alone breaks the $20 cap) or a decision.json / gnn.yaml
    mismatch says to ask the user before spending more. Raised by local entrypoints and drivers;
    nothing is submitted after it."""

    def __init__(self, message: str, detail: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.detail = dict(detail or {})


# --- the as-of guard counters (M3 spec §5.5) ---------------------------------------------------

# FlatBatch.guard is an int64 tensor in this order. Every field adds up over batches except
# max_slack, which is combined with max(). `max_slack` = max(time - bound) over checked edges
# (<= 0 when clean); a batch without edges reports NO_SLACK.
GUARD_FIELDS = (
    "edges_checked",
    "violations",
    "target_hits",
    "max_slack",
    "future_edges",
    "dropped_target_copies",
)
GUARD_MAX_FIELDS = ("max_slack",)
NO_SLACK = -(2**62)


def empty_guard() -> dict[str, int]:
    """Zero counts (max_slack = NO_SLACK): the identity of add_guard."""
    return {f: (NO_SLACK if f in GUARD_MAX_FIELDS else 0) for f in GUARD_FIELDS}


def add_guard(total: Mapping[str, int] | None, values: Mapping[str, int] | Sequence[int]) -> dict:
    """`total` + one batch's guard (a dict, or the 6 values of FlatBatch.guard.tolist())."""
    if isinstance(values, Mapping):
        vals = {f: int(values[f]) for f in GUARD_FIELDS}
    else:
        seq = [int(v) for v in values]
        if len(seq) != len(GUARD_FIELDS):
            raise ValueError(f"a guard has {len(GUARD_FIELDS)} values, got {len(seq)}")
        vals = dict(zip(GUARD_FIELDS, seq, strict=True))
    out = dict(total) if total is not None else empty_guard()
    for f in GUARD_FIELDS:
        out[f] = max(out[f], vals[f]) if f in GUARD_MAX_FIELDS else out[f] + vals[f]
    return out


def guard_is_clean(guard: Mapping[str, int], protocol: str) -> bool:
    """0 violations, and (causal/PNA) 0 target hits. Look-ahead target copies are dropped and
    counted in `dropped_target_copies`; faithful keeps its targets (published protocol)."""
    if int(guard["violations"]) != 0:
        return False
    return protocol not in CAUSAL_BOUND_PROTOCOLS or int(guard["target_hits"]) == 0


# --- JSON lines (history.jsonl, cells.jsonl, the Optuna trial log) ----------------------------


def append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    """Append one JSON line and fsync. A torn last line left by a crash (no trailing newline) is
    cut off first, so the log stays parseable."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, sort_keys=True, default=_json_default) + "\n"
    with open(path, "a+b") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        if size:
            f.seek(size - 1)
            if f.read(1) != b"\n":
                f.seek(0)
                data = f.read()
                f.truncate(data.rfind(b"\n") + 1)
        f.seek(0, os.SEEK_END)
        f.write(line.encode("utf-8"))
        f.flush()
        os.fsync(f.fileno())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """The records of a JSON-lines file ([] if missing). A torn last line (no trailing newline) is
    ignored; a malformed complete line raises ValueError."""
    path = Path(path)
    if not path.exists():
        return []
    # The last element is the torn tail ("" after a final newline): never parsed.
    *complete, _torn = path.read_text(encoding="utf-8").split("\n")
    out = []
    for i, ln in enumerate(complete, start=1):
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except json.JSONDecodeError as e:
            raise ValueError(f"{path}:{i}: malformed JSON line") from e
    return out


def _json_default(o: Any) -> Any:
    if hasattr(o, "item") and callable(o.item) and getattr(o, "shape", None) == ():
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


# --- decided values (gnn_bench -> gnn.yaml) ---------------------------------------------------

# gnn.yaml values gnn_bench decides; the lead copies decision.json into gnn.yaml, and the HPO and
# train entrypoints refuse to start unless they are equal (decision_mismatches == []).
DECIDED_SAMPLER = ("batch_size", "max_edges_per_step", "max_edges_per_eval_step")
# decision.json field <- gnn.yaml runtime field
DECIDED_RUNTIME = {
    "gpu": "gpu",
    "cores": "cpu",
    "memory_mib": "memory_mib",
    "num_workers": "num_workers",
}


def decided_values(gnn_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """The decided values of gnn.yaml in decision.json's vocabulary:
    {gpu, cores, memory_mib, num_workers, batch_size, max_edges_per_step,
    max_edges_per_eval_step, faithful_batch_size}."""
    rt, sp = gnn_cfg["runtime"], gnn_cfg["sampler"]
    out: dict[str, Any] = {k: rt[v] for k, v in DECIDED_RUNTIME.items()}
    out.update({k: sp[k] for k in DECIDED_SAMPLER})
    out["faithful_batch_size"] = gnn_cfg["protocols"]["faithful"]["batch_size"]
    return out


def decision_mismatches(gnn_cfg: Mapping[str, Any], decision: Mapping[str, Any]) -> list[str]:
    """Human-readable differences between gnn.yaml's decided values and decision.json
    (decision["faithful"]["batch_size"] is compared with protocols.faithful.batch_size)."""
    want = decided_values(gnn_cfg)
    got = {k: decision.get(k) for k in want if k != "faithful_batch_size"}
    got["faithful_batch_size"] = (decision.get("faithful") or {}).get("batch_size")
    return [
        f"{k}: gnn.yaml {want[k]!r} != decision.json {got[k]!r}"
        for k in want
        if not _same_value(want[k], got[k])
    ]


def _same_value(a: Any, b: Any) -> bool:
    if isinstance(a, bool) or isinstance(b, bool):
        return a is b
    if isinstance(a, int | float) and isinstance(b, int | float):
        return float(a) == float(b)
    return a == b


def report_hash(gnn_cfg: Mapping[str, Any]) -> str:
    """config_hash of the pre-registered `report` section (M3 spec §13.4): stored in every
    `--final` set summary; evaluate refuses if it changed after the first --final set."""
    from aml.config import config_hash

    return config_hash("gnn_report", gnn_cfg["report"])


# --- config validation (configs/gnn.yaml, M3 spec §3.1) ---------------------------------------

MAX_CHUNK_WALL_S = 14_400  # runtime.chunk_wall_s upper bound (4 h of planned wall per call)
MAX_WALL_GUARD_FACTOR = 2.0  # runtime.wall_guard_factor upper bound

SECTIONS = (
    "graph",
    "sampler",
    "model",
    "train",
    "hpo",
    "protocols",
    "bench",
    "runtime",
    "budget",
    "report",
)
GPUS = ("L4", "T4")
CONVS = ("gine", "pna")
MATMUL_PRECISIONS = ("highest", "high", "medium")
HPO_PARAMS = ("lr", "final_dropout", "w_pos")
MIN_FAITHFUL_EPOCH_CAP = 50  # below this the cost gate refuses and asks the user (§11.4)


class _Checker:
    def __init__(self) -> None:
        self.errors: list[str] = []

    def fail(self, path: str, msg: str) -> None:
        self.errors.append(f"{path}: {msg}")

    def section(self, cfg: Any, path: str, keys: Iterable[str]) -> dict:
        if not isinstance(cfg, Mapping):
            self.fail(path, f"must be a mapping, got {type(cfg).__name__}")
            return {}
        missing = [k for k in keys if k not in cfg]
        if missing:
            self.fail(path, f"missing keys {missing}")
        return dict(cfg)

    def int_(self, d: Mapping, key: str, path: str, lo: int | None = None, hi: int | None = None):
        v = d.get(key)
        if isinstance(v, bool) or not isinstance(v, int):
            self.fail(f"{path}.{key}", f"must be an int, got {v!r}")
            return None
        if (lo is not None and v < lo) or (hi is not None and v > hi):
            self.fail(f"{path}.{key}", f"{v} outside [{lo}, {hi}]")
        return v

    def num(
        self,
        d: Mapping,
        key: str,
        path: str,
        lo: float | None = None,
        hi: float | None = None,
        *,
        lo_open: bool = False,
        hi_open: bool = False,
    ):
        v = d.get(key)
        if isinstance(v, bool) or not isinstance(v, int | float) or not math.isfinite(v):
            self.fail(f"{path}.{key}", f"must be a finite number, got {v!r}")
            return None
        bad_lo = lo is not None and (v <= lo if lo_open else v < lo)
        bad_hi = hi is not None and (v >= hi if hi_open else v > hi)
        if bad_lo or bad_hi:
            lb, rb = ("(" if lo_open else "["), (")" if hi_open else "]")
            self.fail(f"{path}.{key}", f"{v} outside {lb}{lo}, {hi}{rb}")
        return v

    def bool_(self, d: Mapping, key: str, path: str):
        v = d.get(key)
        if not isinstance(v, bool):
            self.fail(f"{path}.{key}", f"must be true/false, got {v!r}")
        return v

    def choice(self, d: Mapping, key: str, path: str, options: Sequence[Any]):
        v = d.get(key)
        if v not in options:
            self.fail(f"{path}.{key}", f"must be one of {list(options)}, got {v!r}")
        return v

    def str_list(self, d: Mapping, key: str, path: str, *, nonempty: bool = True) -> list[str]:
        v = d.get(key)
        if not isinstance(v, list) or not all(isinstance(x, str) and x for x in v):
            self.fail(f"{path}.{key}", f"must be a list of names, got {v!r}")
            return []
        if nonempty and not v:
            self.fail(f"{path}.{key}", "must not be empty")
        if len(set(v)) != len(v):
            self.fail(f"{path}.{key}", f"duplicate names in {v}")
        return v

    def int_list(self, d: Mapping, key: str, path: str, lo: int = 1) -> list[int]:
        v = d.get(key)
        ok = isinstance(v, list) and bool(v)
        ok = ok and all(isinstance(x, int) and not isinstance(x, bool) and x >= lo for x in v)
        if not ok:
            self.fail(f"{path}.{key}", f"must be a non-empty list of ints >= {lo}, got {v!r}")
            return []
        return v

    def seeds(self, d: Mapping, key: str, path: str) -> list[int]:
        v = self.int_list(d, key, path, lo=0)
        if len(set(v)) != len(v):
            self.fail(f"{path}.{key}", f"seeds must be distinct, got {v}")
        return v

    def opt_int(self, d: Mapping, key: str, path: str, lo: int = 1):
        if d.get(key) is None:
            return None
        return self.int_(d, key, path, lo=lo)


def check_gnn_cfg(cfg: Mapping[str, Any]) -> Mapping[str, Any]:
    """Validate configs/gnn.yaml (M3 spec §3.1) and return `cfg` itself (unchanged, so run keys
    hash exactly what was loaded). Raises ValueError listing every problem.

    Checks types and ranges of every key, `train.max_epochs >= train.min_epochs`, fanouts with
    one entry per model layer, distinct seeds, trial 0 (the train section's lr / final_dropout /
    w_pos) inside the HPO space, `lookahead.test_bounds == [end, d10]`, `faithful.epoch_cap`
    null or 1..faithful.max_epochs, `faithful.norm == FAITHFUL_EXEMPT_NORM`, a complete
    `report` section whose band equals published_f1 ± 2 × published_std, and the budget.
    """
    c = _Checker()
    top = c.section(cfg, "gnn", SECTIONS)
    extra = sorted(set(top) - set(SECTIONS))
    if extra:
        c.fail("gnn", f"unknown sections {extra}")

    g = c.section(
        top.get("graph", {}),
        "graph",
        ("edge_attr_columns", "categorical", "log1p", "zscore", "norm_split", "std_floor"),
    )
    if g.get("edge_attr_columns") != "gnn_edge_attr":
        c.fail("graph.edge_attr_columns", "must be 'gnn_edge_attr' (EngineSpec.gnn_edge_attr)")
    cat = c.str_list(g, "categorical", "graph")
    log1p = c.str_list(g, "log1p", "graph", nonempty=False)
    zs = c.str_list(g, "zscore", "graph", nonempty=False)
    both = sorted(set(cat) & (set(log1p) | set(zs)))
    if both:
        c.fail("graph", f"categorical columns cannot be transformed: {both}")
    if g.get("norm_split") != "train":
        c.fail("graph.norm_split", "must be 'train' (PLAN §4: statistics from train rows only)")
    c.num(g, "std_floor", "graph", 0.0, None, lo_open=True)

    layers = None
    m = c.section(
        top.get("model", {}),
        "model",
        (
            "conv",
            "hidden",
            "layers",
            "layer_dropout",
            "readout_hidden",
            "edge_updates",
            "batch_norm",
        ),
    )
    c.choice(m, "conv", "model", CONVS)
    c.int_(m, "hidden", "model", lo=1)
    layers = c.int_(m, "layers", "model", lo=1)
    c.num(m, "layer_dropout", "model", 0.0, 1.0, hi_open=True)
    c.int_list(m, "readout_hidden", "model")
    c.bool_(m, "edge_updates", "model")
    c.bool_(m, "batch_norm", "model")

    s = c.section(
        top.get("sampler", {}),
        "sampler",
        (
            "fanout",
            "temporal_strategy",
            "ego",
            "batch_size",
            "eval_batch_size",
            "max_edges_per_step",
            "max_edges_per_eval_step",
            "eval_cache_max_gb",
        ),
    )
    fan = c.int_list(s, "fanout", "sampler")
    if fan and layers is not None and len(fan) != layers:
        c.fail("sampler.fanout", f"needs one entry per layer ({layers}), got {fan}")
    # `last` makes eval trees deterministic (the eval cache and look-ahead step 1 rely on it).
    c.choice(s, "temporal_strategy", "sampler", ("last",))
    c.choice(s, "ego", "sampler", ("account",))
    c.int_(s, "batch_size", "sampler", lo=1)
    c.int_(s, "eval_batch_size", "sampler", lo=1)
    c.opt_int(s, "max_edges_per_step", "sampler")
    c.opt_int(s, "max_edges_per_eval_step", "sampler")
    c.num(s, "eval_cache_max_gb", "sampler", 0.0, None)

    t = c.section(
        top.get("train", {}),
        "train",
        (
            "neg_rate",
            "w_pos",
            "lr",
            "final_dropout",
            "max_epochs",
            "min_epochs",
            "patience",
            "deterministic",
            "matmul_precision",
        ),
    )
    c.num(t, "neg_rate", "train", 0.0, 1.0, lo_open=True)
    c.num(t, "w_pos", "train", 0.0, None, lo_open=True)
    c.num(t, "lr", "train", 0.0, None, lo_open=True)
    c.num(t, "final_dropout", "train", 0.0, 1.0, hi_open=True)
    max_e = c.int_(t, "max_epochs", "train", lo=1)
    min_e = c.int_(t, "min_epochs", "train", lo=1)
    if max_e is not None and min_e is not None and max_e < min_e:
        c.fail("train", f"max_epochs {max_e} < min_epochs {min_e}")
    c.int_(t, "patience", "train", lo=1)
    c.bool_(t, "deterministic", "train")
    c.choice(t, "matmul_precision", "train", MATMUL_PRECISIONS)

    h = c.section(
        top.get("hpo", {}),
        "hpo",
        (
            "n_trials",
            "max_epochs",
            "n_startup_trials",
            "pruner",
            "sampler_seed_base",
            "model_seed",
            "space",
        ),
    )
    c.int_(h, "n_trials", "hpo", lo=1)
    c.int_(h, "max_epochs", "hpo", lo=1)
    c.int_(h, "n_startup_trials", "hpo", lo=0)
    pr = c.section(h.get("pruner", {}), "hpo.pruner", ("n_startup_trials", "n_warmup_steps"))
    c.int_(pr, "n_startup_trials", "hpo.pruner", lo=0)
    c.int_(pr, "n_warmup_steps", "hpo.pruner", lo=0)
    c.int_(h, "sampler_seed_base", "hpo", lo=0)
    c.int_(h, "model_seed", "hpo", lo=0)
    space = c.section(h.get("space", {}), "hpo.space", HPO_PARAMS)
    extra = sorted(set(space) - set(HPO_PARAMS))
    if extra:
        c.fail("hpo.space", f"unknown parameters {extra}; the space is {list(HPO_PARAMS)}")
    for name in HPO_PARAMS:
        if name not in space:
            continue
        p = f"hpo.space.{name}"
        d = c.section(space[name], p, ("low", "high"))
        lo = c.num(d, "low", p)
        hi = c.num(d, "high", p)
        log = d.get("log", False)
        if not isinstance(log, bool):
            c.fail(f"{p}.log", f"must be true/false, got {log!r}")
        if lo is not None and hi is not None:
            if lo >= hi:
                c.fail(p, f"low {lo} >= high {hi}")
            if log and lo <= 0:
                c.fail(p, "a log-uniform range needs low > 0")
            v = t.get(name)
            if isinstance(v, int | float) and not isinstance(v, bool) and not lo <= v <= hi:
                c.fail(p, f"trial 0 (train.{name} = {v}) lies outside [{lo}, {hi}]")

    _check_protocols(c, top.get("protocols", {}), layers)
    _check_bench(c, top.get("bench", {}))

    rt = c.section(
        top.get("runtime", {}),
        "runtime",
        (
            "gpu",
            "cpu",
            "memory_mib",
            "num_workers",
            "loader_timeout_s",
            "chunk_wall_s",
            "wall_guard_factor",
        ),
    )
    c.choice(rt, "gpu", "runtime", GPUS)
    c.num(rt, "cpu", "runtime", 0.0, None, lo_open=True)
    c.int_(rt, "memory_mib", "runtime", lo=1024)
    c.int_(rt, "num_workers", "runtime", lo=0)
    c.int_(rt, "loader_timeout_s", "runtime", lo=1)
    # Upper bounds: both size every chunk's timeout T and the set's wall guard, and runtime is
    # unkeyed (an edit takes effect at once), so a typo must not scale a chunk's worst case.
    c.int_(rt, "chunk_wall_s", "runtime", lo=60, hi=MAX_CHUNK_WALL_S)
    c.num(rt, "wall_guard_factor", "runtime", 1.0, MAX_WALL_GUARD_FACTOR)

    b = c.section(
        top.get("budget", {}),
        "budget",
        (
            "m3_cap_usd",
            "workspace_budget_usd",
            "m6_reserve_usd",
            "overhead",
            "dev_allowance_usd",
        ),
    )
    cap = c.num(b, "m3_cap_usd", "budget", 0.0, None, lo_open=True)
    ws = c.num(b, "workspace_budget_usd", "budget", 0.0, None, lo_open=True)
    res = c.num(b, "m6_reserve_usd", "budget", 0.0, None)
    c.num(b, "overhead", "budget", 1.0, None)
    c.num(b, "dev_allowance_usd", "budget", 0.0, None)
    if None not in (cap, ws, res) and cap > ws - res:
        c.fail("budget", f"m3_cap_usd {cap} > workspace_budget_usd - m6_reserve_usd {ws - res}")

    _check_report(c, top.get("report", {}))

    if c.errors:
        raise ValueError("invalid gnn config:\n  " + "\n  ".join(c.errors))
    return cfg


def _check_protocols(c: _Checker, protos: Any, layers: int | None) -> None:
    p = c.section(protos, "protocols", PROTOCOLS)
    extra = sorted(set(p) - set(PROTOCOLS))
    if extra:
        c.fail("protocols", f"unknown protocols {extra}")

    causal = c.section(p.get("causal", {}), "protocols.causal", ("seeds",))
    c.seeds(causal, "seeds", "protocols.causal")

    la = c.section(p.get("lookahead", {}), "protocols.lookahead", ("seeds", "test_bounds"))
    c.seeds(la, "seeds", "protocols.lookahead")
    if la.get("test_bounds") != list(TEST_BOUNDS):
        c.fail("protocols.lookahead.test_bounds", f"must be {list(TEST_BOUNDS)}")

    path = "protocols.pna"
    pna = c.section(
        p.get("pna", {}),
        path,
        ("seeds", "hidden", "towers", "lr", "layer_dropout", "final_dropout", "w_pos"),
    )
    c.seeds(pna, "seeds", path)
    hid = c.int_(pna, "hidden", path, lo=1)
    tow = c.int_(pna, "towers", path, lo=1)
    if hid is not None and tow is not None and hid % tow:
        c.fail(path, f"hidden {hid} must be divisible by towers {tow} (PNAConv)")
    c.num(pna, "lr", path, 0.0, None, lo_open=True)
    c.num(pna, "layer_dropout", path, 0.0, 1.0, hi_open=True)
    c.num(pna, "final_dropout", path, 0.0, 1.0, hi_open=True)
    c.num(pna, "w_pos", path, 0.0, None, lo_open=True)

    path = "protocols.faithful"
    f = c.section(
        p.get("faithful", {}),
        path,
        (
            "seed",
            "fanout",
            "batch_size",
            "max_epochs",
            "epoch_cap",
            "lr",
            "w_pos",
            "layer_dropout",
            "final_dropout",
            "edge_features",
            "norm",
            "reverse_self_loops",
        ),
    )
    c.int_(f, "seed", path, lo=0)
    fan = c.int_list(f, "fanout", path)
    if fan and layers is not None and len(fan) != layers:
        c.fail(f"{path}.fanout", f"needs one entry per layer ({layers}), got {fan}")
    c.int_(f, "batch_size", path, lo=1)
    max_e = c.int_(f, "max_epochs", path, lo=1)
    if f.get("epoch_cap") is not None:
        c.int_(f, "epoch_cap", path, lo=1, hi=max_e)
    c.num(f, "lr", path, 0.0, None, lo_open=True)
    c.num(f, "w_pos", path, 0.0, None, lo_open=True)
    c.num(f, "layer_dropout", path, 0.0, 1.0, hi_open=True)
    c.num(f, "final_dropout", path, 0.0, 1.0, hi_open=True)
    feats = c.str_list(f, "edge_features", path)
    exempt = [x for x in FAITHFUL_EXEMPT_FEATURES if x not in feats]
    if feats and exempt:
        c.fail(f"{path}.edge_features", f"Multi-GNN's feature set includes {exempt}")
    # The faithful code is not driven by these two keys (faithful.FAITHFUL_COLUMNS and
    # snapshot_hetero's flip are fixed): they must state what runs, or an edit would re-key
    # the run and change nothing else.
    from aml.models.gnn.faithful import FAITHFUL_COLUMNS

    if feats and feats != list(FAITHFUL_COLUMNS):
        c.fail(
            f"{path}.edge_features",
            f"must equal faithful.FAITHFUL_COLUMNS {list(FAITHFUL_COLUMNS)} (the code is not "
            "driven by this list)",
        )
    if f.get("norm") != FAITHFUL_EXEMPT_NORM:
        c.fail(f"{path}.norm", f"must be {FAITHFUL_EXEMPT_NORM!r}")
    c.bool_(f, "reverse_self_loops", path)
    if isinstance(f.get("reverse_self_loops"), bool) and f["reverse_self_loops"] is not True:
        c.fail(
            f"{path}.reverse_self_loops",
            "must be true (faithful.snapshot_hetero flips every edge, self-loops included)",
        )


def _check_bench(c: _Checker, bench: Any) -> None:
    keys = (
        "gpus",
        "batch_sizes",
        "num_workers",
        "warmup_steps",
        "timed_steps",
        "lookahead",
        "faithful",
        "pna",
        "memory_fraction",
        "rerun_4core_if_wait_below",
        "tie_band",
    )
    b = c.section(bench, "bench", keys)
    gpus = c.str_list(b, "gpus", "bench")
    bad = [x for x in gpus if x not in GPUS]
    if bad:
        c.fail("bench.gpus", f"unsupported GPUs {bad}; allowed {list(GPUS)}")
    c.int_list(b, "batch_sizes", "bench")
    c.int_list(b, "num_workers", "bench", lo=0)
    c.int_(b, "warmup_steps", "bench", lo=0)
    c.int_(b, "timed_steps", "bench", lo=1)
    la = c.section(b.get("lookahead", {}), "bench.lookahead", ("timed_steps", "val_batches"))
    c.int_(la, "timed_steps", "bench.lookahead", lo=1)
    c.int_(la, "val_batches", "bench.lookahead", lo=1)
    fk = ("warmup_steps", "timed_steps", "val_batches", "test_batches", "batch_fallback")
    f = c.section(b.get("faithful", {}), "bench.faithful", fk)
    c.int_(f, "warmup_steps", "bench.faithful", lo=0)
    for k in ("timed_steps", "val_batches", "test_batches"):
        c.int_(f, k, "bench.faithful", lo=1)
    fb = c.int_list(f, "batch_fallback", "bench.faithful")
    if fb and any(a <= b_ for a, b_ in zip(fb, fb[1:], strict=False)):
        c.fail("bench.faithful.batch_fallback", f"must be strictly decreasing, got {fb}")
    pna = c.section(b.get("pna", {}), "bench.pna", ("warmup_steps", "timed_steps"))
    c.int_(pna, "warmup_steps", "bench.pna", lo=0)
    c.int_(pna, "timed_steps", "bench.pna", lo=1)
    c.num(b, "memory_fraction", "bench", 0.0, 1.0, lo_open=True)
    c.num(b, "rerun_4core_if_wait_below", "bench", 0.0, 1.0)
    c.num(b, "tie_band", "bench", 0.0, 1.0, hi_open=True)


def _check_report(c: _Checker, report: Any) -> None:
    keys = (
        "winner_pair",
        "winner_metrics",
        "gap_metrics",
        "ci_level",
        "reproduced_band",
        "published_f1",
        "published_std",
    )
    r = c.section(report, "report", keys)
    pair = c.str_list(r, "winner_pair", "report")
    if pair and (len(pair) != 2 or pair[0] not in COMPARISON_MODELS):
        c.fail("report.winner_pair", f"must be [<GNN comparison model>, <baseline>], got {pair}")
    wm = c.str_list(r, "winner_metrics", "report")
    if wm and len(wm) != 2:
        c.fail("report.winner_metrics", f"must be [primary, secondary], got {wm}")
    c.str_list(r, "gap_metrics", "report")
    c.num(r, "ci_level", "report", 0.0, 1.0, lo_open=True, hi_open=True)
    f1 = c.num(r, "published_f1", "report", 0.0, 100.0)
    sd = c.num(r, "published_std", "report", 0.0, None, lo_open=True)
    band = r.get("reproduced_band")
    ok = isinstance(band, list) and len(band) == 2
    ok = ok and all(isinstance(x, int | float) and not isinstance(x, bool) for x in band)
    if not ok:
        c.fail("report.reproduced_band", f"must be [low, high], got {band!r}")
    elif f1 is not None and sd is not None:
        want = (f1 - 2 * sd, f1 + 2 * sd)
        if abs(band[0] - want[0]) > 0.011 or abs(band[1] - want[1]) > 0.011:
            c.fail(
                "report.reproduced_band",
                f"{band} != published_f1 ± 2 × published_std = [{want[0]:.2f}, {want[1]:.2f}]",
            )
