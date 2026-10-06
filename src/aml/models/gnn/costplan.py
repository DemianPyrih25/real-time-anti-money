"""GNN cost formulas, the bench decision, the run plan and the cost gate (M3 spec §10.2-§11.4;
owner C). Pure Python, **no torch**: the CPU bench driver and every local entrypoint import it.

Every $ figure is a planning estimate until gnn_bench measures it. Prices: GPU_price.md
(`shape $/h = GPU $/h + cores x 0.0473 + GiB x 0.008`). `shape_usd_h` is that exact formula
(1.4344 / 1.2244 / 1.1172 / 0.9072 for L4/T4 x 8 cores 32 GiB / 4 cores 16 GiB) and ranks cells
in `decide`; every budget projection (plan, gate) uses `plan_usd_h` = the exact price rounded UP
to the cent, which is GPU_price.md's table (1.44 / 1.23 / 1.12 / 0.91) and never under-counts.
`overhead` (gnn.yaml budget.overhead, 1.3) covers retries and debugging (COST_NOTES).

Bench cells (one JSON line each in cells.jsonl, written by bench.run_bench_container) share
these fields: {cell_id, group (CELL_GROUPS), gpu, device_name, device_bytes, cores, memory_mib,
batch_size, num_workers, status ("ok" | "oom" | "error"), error, seconds, host_peak_mib (the
container's memory, sampled during the cell; rule 5 reads it), host_peak_source,
host_peak_mib_proxy (the old ru_maxrss proxy, for reference), guard (GUARD totals)}; per group
(all times in seconds, memory in bytes):
  build        build_s, n_nodes, n_edges, counts {n_pos, n_neg, n_val_early, n_val_late, n_test,
               n_test_d10}
  grid, rerun, pna, lookahead (train steps)
               warmup_steps, timed_steps, step_s_mean, step_s_p90 (wall incl. sampler wait),
               gpu_s_mean (CUDA events), wait_s_mean, wait_share (= wait / step), edges_mean,
               edges_max, nodes_mean, nodes_max, peak_mem_bytes, bytes_per_edge
               (rerun may add the val fields below for its own val pass)
  stress       edges, peak_mem_bytes, bytes_per_edge, max_edges_per_step
  val          val_s (uncached pass that fills the cache), val_cached_s, cache_gb, cache_fits,
               eval_batches, eval_edges_max, eval_peak_mem_bytes, eval_bytes_per_edge,
               max_edges_per_eval_step
  lookahead    (+) val_batches, val_first_batch_s (a fresh loader's first batch: worker fork),
               val_s_per_batch (the other batches), edges_per_target_train,
               edges_per_target_val, future_share
  faithful     tried (batch sizes in order), batch_size (the one that fit, None if none),
               warmup_steps, timed_steps, step_s_mean, edges_mean, edges_max, peak_mem_bytes,
               val_s_per_batch, test_s_per_batch (first batch apart: val_first_batch_s,
               test_first_batch_s), val_edges_mean, test_edges_mean, sampled_share {train, val,
               test}
  determinism  bit_deterministic, max_param_diff, det_overhead_ratio (train step time with
               deterministic algorithms / without, >= 1; = det_overhead_ratio_train),
               det_overhead_ratio_eval (eval forwards), det_overhead_ratio_faithful (faithful
               train steps; None if not measured), cpu_gpu_max_logit_diff

decision.json (`decide`): {gpu, cores, memory_mib, memory_limit_mib, num_workers, batch_size,
max_edges_per_step, max_edges_per_eval_step, eval_cache {fits, gb}, faithful {gpu, batch_size,
usd_epoch, cores, num_workers, memory_mib} (the shape its cell was measured on: the faithful run
keeps it), bytes_per_edge {train, eval}, precision ("high" on L4, "highest" on T4),
bit_deterministic, det_overhead_ratio (train), det_overhead_ratios {train, eval, faithful},
usd_epoch, epoch_s, cell_id, rerun_4core (bool),
ask_user (str | None: a rule says to stop and ask), projection_usd {deterministic,
without_determinism} | None, counts, measured (the timings plan_runs reads), cells (the measured
per-cell table: one row per grid / rerun cell with its epoch_s, usd_epoch and eligibility)}. Its
decided values are compared with gnn.yaml by aml.models.gnn.decision_mismatches.

plan.json (`plan_runs`): one row per RUN_ORDER entry: {run, protocol, seeds, units (seed runs or
HPO trials), n_trials, max_epochs, epochs {central, conservative} (per unit), gpu, cores,
memory_mib, shape_usd_h, price_usd_h, batch_size, epoch_s, scoring_s, startup_s (per container),
work_s, n_containers, run_s, usd (x overhead; each a {central, conservative} dict),
usd_epoch_cons, overhead, chunk_wall_s, epoch_cap (faithful), cut_alt (HPO at 4 trials) | None,
source ("bench" | "planning"), status ("planned"; "cut" when gnn.yaml already applied the cut)}.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Collection, Mapping, Sequence
from typing import Any

GPU_USD_H = {"L4": 0.80, "T4": 0.59}
CORE_USD_H = 0.0473
GIB_USD_H = 0.008
GNN_APPS = ("aml-gnn-bench", "aml-hpo-gnn", "aml-train-gnn")  # == modal_jobs.common.GNN_APPS
# M3's dev apps (the Linux test runner, the GPU smoke test): billed to M3 as well.
DEV_APPS = ("aml-linux-runner", "aml-smoke")  # == modal_jobs.common.DEV_APPS
M3_APPS = GNN_APPS + DEV_APPS  # the apps whose billing the gate counts as M3 spend
BILLING_INTERVAL_S = 3600  # the gate reads the hourly billing report (full intervals only)

CELL_GROUPS = (
    "build",
    "grid",
    "stress",
    "val",
    "faithful",
    "lookahead",
    "pna",
    "determinism",
    "rerun",
)
# The cell groups each bench container runs (M3 spec §10.1), in order.
CONTAINER_GROUPS = {
    "L4": ("build", "grid", "stress", "val", "faithful", "lookahead", "pna", "determinism"),
    "T4": ("build", "grid", "stress", "val", "faithful"),
    "rerun": ("build", "rerun"),  # the winner GPU at 4 cores / 16 GiB, workers 3
}

# Planned runs in submission order (protects the never-cut items), the cut order and the
# never-cut set (M3 spec §11.4; PLAN §2.3).
RUN_ORDER = ("bench", "dev", "hpo", "causal", "lookahead_s0", "faithful", "lookahead_rest", "pna")
CUT_ORDER = ("pna", "hpo_4", "lookahead_1")  # 1. drop PNA 2. HPO 8 -> 4 3. look-ahead 3 -> 1
NEVER_CUT = ("causal", "lookahead_s0", "faithful")
MIN_FAITHFUL_EPOCHS = 50  # a faithful epoch cap below this: refuse and ask the user

CUT_TARGET = {"pna": "pna", "hpo_4": "hpo", "lookahead_1": "lookahead_rest"}
HPO_CUT_TRIALS = 4
CENTRAL, CONS = "central", "conservative"
WHICH = (CENTRAL, CONS)
CENTRAL_EPOCH_SHARE = 0.6  # central projection: 60% of max epochs (no credit for faithful)
DEV_MAX_EPOCHS = 2  # `make gnn-dev` passes --max-epochs 2
RERUN_CORES, RERUN_MEMORY_MIB = 4, 16384
MIN_MEMORY_MIB, MEMORY_STEP_MIB = 16384, 8192  # rule 5
ELIGIBLE_CAP_FACTOR = 1.5  # rule 1: edge cap >= 1.5 x the cell's mean batch edges
PRECISION = {"L4": "high", "T4": "highest"}  # rule 3 (TF32 on L4; "highest" elsewhere)
CPU_LABEL = "cpu"  # the GPU label of bench cells measured on a CPU device (tests)

# HI-Small (M3 spec §2.1): train positives / negatives and the scored split sizes.
HI_SMALL_COUNTS = {
    "n_pos": 2_534,
    "n_neg": 3_246_387,
    "n_val_early": 482_751,
    "n_val_late": 482_773,
    "n_test": 863_900,
    "n_test_d10": 863_900,  # d10 scores every test row (tail rows with their causal bound)
}

# The §11.1 planning constants (replaced by the bench's measurements once decision.json exists).
PLANNING: dict[str, Any] = {
    "gpu": "L4",
    "cores": 8,
    "memory_mib": 32768,
    # causal edges (fwd + rev) per target, F19; conservative x 1.5, look-ahead x 1.2
    "edges_per_target": {"train": 148, "val_early": 164, "val_late": 170, "test": 184},
    "edges_factor": {CENTRAL: 1.0, CONS: 1.5},
    "lookahead_edges_factor": 1.2,
    # GPU ns per sampled edge and fixed seconds per step. Deterministic algorithms: GINE sums
    # via index_add_ (model.IndexAddSum), one radix sort of E row indices per aggregation
    # (~16 B x a few passes per edge, < 2 ns at 250 GB/s) instead of scatter_add_'s E x H
    # keys (~50 ns per edge per aggregation at H 64); that is inside the conservative margin
    # (220 / 75 ns = 1.8x central), so no separate term; the bench measures it (rule 6).
    "ns_per_edge": {CENTRAL: {"train": 120, "eval": 40}, CONS: {"train": 220, "eval": 75}},
    "step_overhead_s": {"train": 0.010, "eval": 0.005},
    # faithful edges per batch of FAITHFUL_REF_BATCH targets (train / val / test snapshot)
    "faithful_edges_per_batch": {"train": 1.15e6, "val": 1.5e6, "test": 1.85e6},
    "faithful_ref_batch": 8192,
    "faithful_build_s": {CENTRAL: 60, CONS: 120},  # snapshots + EA_f, in every container
    "pna_epoch_factor": {CENTRAL: 1.5, CONS: 3.0},  # x the headline epoch and scoring
    "startup_s": {CENTRAL: 180, CONS: 300},  # per container: image, graph load, CSC
    "epoch_overhead_s": {CENTRAL: 2, CONS: 3},  # checkpoint, commit, metrics
    # gnn_bench: container seconds (L4 all groups, T4 fewer, the optional 4-core rerun at the
    # L4 shape), plus driver and T4 smoke; conservative = x 1.5 on everything
    "bench_container_s": {"L4": 720, "T4": 480, "rerun": 300},
    "bench_fixed_usd": 0.03,
    "bench_cons_factor": 1.5,
}

# Splits each scoring pass covers (look-ahead scores test twice: bounds `end` and `d10`).
SCORED_FINAL = ("val_early", "val_late", "test")
SCORED_LOOKAHEAD = ("val_early", "val_late", "test", "test_d10")
SCORED_DEV = ("val_early", "val_late")
_SPLIT_COUNT = {
    "val_early": "n_val_early",
    "val_late": "n_val_late",
    "test": "n_test",
    "test_d10": "n_test_d10",
}
_SPLIT_EDGES = {
    "val_early": "val_early",
    "val_late": "val_late",
    "test": "test",
    "test_d10": "test",
}


def cell_id(
    gpu: str,
    cores: int,
    group: str,
    batch_size: int | None = None,
    num_workers: int | None = None,
) -> str:
    """Unique id of a bench cell, e.g. "L4-8c-grid-b2048-w7" (resume skips recorded ids)."""
    parts = [f"{gpu}-{int(cores)}c-{group}"]
    if batch_size is not None:
        parts.append(f"b{int(batch_size)}")
    if num_workers is not None:
        parts.append(f"w{int(num_workers)}")
    return "-".join(parts)


# --- prices and formulas (§10.2) ---------------------------------------------------------------


def shape_usd_h(gpu: str, cores: float, gib: float) -> float:
    """$/h of a container shape: GPU_USD_H[gpu] + cores x CORE_USD_H + gib x GIB_USD_H (exact).
    gpu CPU_LABEL ("cpu", bench cells run on a CPU device in tests) has no GPU part."""
    if gpu != CPU_LABEL and gpu not in GPU_USD_H:
        raise ValueError(f"no price for GPU {gpu!r}; known: {sorted(GPU_USD_H)}")
    return GPU_USD_H.get(gpu, 0.0) + float(cores) * CORE_USD_H + float(gib) * GIB_USD_H


def plan_usd_h(gpu: str, cores: float, gib: float) -> float:
    """The budget price of a shape: shape_usd_h rounded UP to the cent (= GPU_price.md's table:
    L4/T4 x 8 cores 32 GiB = 1.44 / 1.23, x 4 cores 16 GiB = 1.12 / 0.91)."""
    return math.ceil(round(shape_usd_h(gpu, cores, gib) * 100, 6)) / 100


def steps_per_epoch(n_pos: int, n_neg: int, neg_rate: float, batch_size: int) -> int:
    """ceil((n_pos + round(neg_rate x n_neg)) / batch_size)."""
    if int(batch_size) < 1:
        raise ValueError(f"batch_size must be >= 1, got {batch_size}")
    return math.ceil((int(n_pos) + round(float(neg_rate) * int(n_neg))) / int(batch_size))


def epoch_seconds(steps: int, t_step: float, t_val: float, t_ovh: float) -> float:
    """steps x t_step + t_val + t_ovh (t_val: the cached val pass if the cache fits, else the
    uncached pass)."""
    return steps * t_step + t_val + t_ovh


def run_seconds(
    *, startup_s: float, n_containers: int, epochs: float, epoch_s: float, scoring_s: float
) -> float:
    """startup_s x n_containers + epochs x epoch_s + scoring_s."""
    return startup_s * n_containers + epochs * epoch_s + scoring_s


def usd(shape_usd_h_: float, seconds: float, overhead: float = 1.0) -> float:
    """shape $/h / 3600 x seconds x overhead."""
    return shape_usd_h_ / 3600.0 * seconds * overhead


def attempt_wall_s(chunk_wall_s: float, projected_set_s: float, wall_guard_factor: float) -> float:
    """min(chunk_wall_s, wall_guard_factor x projected_set_s): one worker call's wall budget."""
    return min(float(chunk_wall_s), float(wall_guard_factor) * float(projected_set_s))


def attempt_timeout_s(attempt_wall_s_: float, epoch_s_cons: float) -> int:
    """Per-attempt Modal timeout T = max(1800, ceil(attempt_wall_s + max(600, 3 x
    epoch_s_cons))) (M3 spec §12.1)."""
    return max(1800, math.ceil(attempt_wall_s_ + max(600.0, 3.0 * epoch_s_cons)))


def memory_request_mib(peak_host_mib: float) -> tuple[int, int]:
    """Rule 5: request = max(16 GiB, 1.5 x peak host memory) rounded up to 8 GiB; limit =
    request + 8 GiB. Returns (request, limit) in MiB."""
    want = max(float(MIN_MEMORY_MIB), 1.5 * float(peak_host_mib or 0.0))
    req = math.ceil(round(want, 6) / MEMORY_STEP_MIB) * MEMORY_STEP_MIB
    return int(req), int(req + MEMORY_STEP_MIB)


def run_limits(row: Mapping[str, Any], runtime: Mapping[str, Any]) -> dict:
    """The worker limits of one planned set (M3 spec §7.6, §12.1), from its conservative
    projection: {projected_s, attempt_wall_s, timeout_s (T, constant per set), set_wall_limit_s
    (the driver writes STOPPED.json beyond it: wall_guard_factor x projection + one chunk)}."""
    projected = float(row["run_s"][CONS])
    a = attempt_wall_s(runtime["chunk_wall_s"], projected, runtime["wall_guard_factor"])
    return {
        "projected_s": projected,
        "attempt_wall_s": a,
        "timeout_s": attempt_timeout_s(a, float(row["epoch_s"][CONS])),
        "set_wall_limit_s": float(runtime["wall_guard_factor"]) * projected
        + float(runtime["chunk_wall_s"]),
    }


# --- planning helpers -----------------------------------------------------------------------


def _counts(*sources: Mapping[str, Any] | None) -> dict[str, int]:
    out = dict(HI_SMALL_COUNTS)
    for src in sources:
        for k, v in (src or {}).items():
            if k in out and v is not None:
                out[k] = int(v)
    return out


def _batches(n: int, batch: int) -> int:
    return math.ceil(n / batch) if n > 0 else 0


def _passes(counts: Mapping[str, int], splits: Sequence[str], batch: int) -> tuple[int, float]:
    """(eval batches, mean planning edges per target) of a scoring pass over `splits`."""
    n_b, n_seeds, edges = 0, 0, 0.0
    ept = PLANNING["edges_per_target"]
    for s in splits:
        n = counts[_SPLIT_COUNT[s]]
        n_b += _batches(n, batch)
        n_seeds += n
        edges += n * ept[_SPLIT_EDGES[s]]
    return n_b, (edges / n_seeds if n_seeds else 0.0)


def _plan_step(which: str, kind: str, edges_per_target: float, batch: int) -> float:
    """Planning seconds of one train / eval step (§11.1): fixed + batch x edges x ns/edge."""
    ns = PLANNING["ns_per_edge"][which][kind]
    return PLANNING["step_overhead_s"][kind] + batch * edges_per_target * ns * 1e-9


def _both(fn) -> dict[str, float]:
    return {w: float(fn(w)) for w in WHICH}


class _Timing:
    """Per-epoch and scoring seconds of one model kind (headline, look-ahead, PNA, faithful) on
    the run's shape, from the planning constants or the bench's measurements."""

    def __init__(self, cfg: Mapping[str, Any], counts: Mapping[str, int], measured: Mapping | None):
        self.cfg, self.counts, self.m = cfg, counts, measured
        s = cfg["sampler"]
        self.batch = int(measured["batch_size"]) if measured else int(s["batch_size"])
        self.eval_batch = int(s["eval_batch_size"])
        self.steps = steps_per_epoch(
            counts["n_pos"], counts["n_neg"], cfg["train"]["neg_rate"], self.batch
        )
        self.n_val = _batches(counts["n_val_early"], self.eval_batch)
        ept = PLANNING["edges_per_target"]
        self.ept_val = float(ept["val_early"])

    # -- headline / look-ahead (temporal GINE) ------------------------------------------------

    def epoch_s(self, which: str, *, lookahead: bool = False) -> tuple[float, str]:
        ovh = PLANNING["epoch_overhead_s"][which]
        m = self.m
        if m is not None and (not lookahead or m.get("lookahead")):
            if lookahead:
                la = m["lookahead"]
                if which == CENTRAL and la.get("val_cached_s") is not None:
                    t_val = float(la["val_cached_s"])  # its val trees are cached too (fixed bound)
                else:
                    first = float(la.get("val_first_batch_s") or 0.0)
                    t_val = self.n_val * la["val_s_per_batch"] + first
                return self.steps * la["step_s"] + t_val + ovh, "bench"
            t_val = m["val_cached_s"] if (which == CENTRAL and m.get("cache_fits")) else m["val_s"]
            return epoch_seconds(self.steps, m["train_step_s"], t_val, ovh), "bench"
        f = PLANNING["edges_factor"][which] * (
            PLANNING["lookahead_edges_factor"] if lookahead else 1.0
        )
        ept = PLANNING["edges_per_target"]
        t_step = _plan_step(which, "train", ept["train"] * f, self.batch)
        t_val = self.n_val * _plan_step(which, "eval", ept["val_early"] * f, self.eval_batch)
        return epoch_seconds(self.steps, t_step, t_val, ovh), "planning"

    def scoring_s(
        self, which: str, splits: Sequence[str], k: int, *, lookahead: bool = False
    ) -> float:
        """One scoring pass over `splits` for k seeds: every batch sampled once, k forwards
        (budgeted as k x the per-batch time, as §11.2)."""
        if k <= 0 or not splits:
            return 0.0
        n_b, mean_ept = _passes(self.counts, splits, self.eval_batch)
        m = self.m
        if m is not None and (not lookahead or m.get("lookahead")):
            first = 0.0  # the headline per-batch time already spreads its loader's fork
            if lookahead:
                per_batch = m["lookahead"]["val_s_per_batch"]
                first = len(splits) * float(m["lookahead"].get("val_first_batch_s") or 0.0)
            else:
                per_batch = m["val_s"] / max(1, int(m["eval_batches"]))
            return n_b * k * per_batch * (mean_ept / self.ept_val) + first
        f = PLANNING["edges_factor"][which] * (
            PLANNING["lookahead_edges_factor"] if lookahead else 1.0
        )
        return n_b * k * _plan_step(which, "eval", mean_ept * f, self.eval_batch)

    # -- PNA ------------------------------------------------------------------------------

    def pna_factor(self, which: str) -> tuple[float, str]:
        m = self.m
        if m is not None and m.get("pna"):
            return m["pna"]["step_s"] / m["train_step_s"], "bench"
        return PLANNING["pna_epoch_factor"][which], "planning"

    # -- faithful -------------------------------------------------------------------------

    def faithful(self, which: str, batch: int) -> tuple[float, float, str]:
        """(epoch_s, test scoring_s, source) of the faithful run at `batch`."""
        c = self.counts
        n_train = c["n_pos"] + c["n_neg"]  # every train edge, every epoch (no down-sampling)
        steps = _batches(n_train, batch)
        n_val = _batches(c["n_val_early"] + c["n_val_late"], batch)  # the days 7-8 graph
        n_test = _batches(c["n_test"], batch)
        ovh = PLANNING["epoch_overhead_s"][which]
        fm = (self.m or {}).get("faithful")
        if fm:
            # a faithful eval loader forks its workers once per pass (val: every epoch)
            first_v = float(fm.get("val_first_batch_s") or 0.0)
            first_t = float(fm.get("test_first_batch_s") or 0.0)
            epoch = steps * fm["step_s"] + n_val * fm["val_s_per_batch"] + first_v + ovh
            return epoch, n_test * fm["test_s_per_batch"] + first_t, "bench"
        f = PLANNING["edges_factor"][which] * batch / PLANNING["faithful_ref_batch"]
        e = PLANNING["faithful_edges_per_batch"]
        ns, fixed = PLANNING["ns_per_edge"][which], PLANNING["step_overhead_s"]
        t_train = fixed["train"] + e["train"] * f * ns["train"] * 1e-9
        t_val = fixed["eval"] + e["val"] * f * ns["eval"] * 1e-9
        t_test = fixed["eval"] + e["test"] * f * ns["eval"] * 1e-9
        return steps * t_train + n_val * t_val + ovh, n_test * t_test, "planning"

    def startup_s(self, which: str) -> float:
        """Per container: the planning startup, raised to the measured graph build (x 1.5
        conservative) when that is longer."""
        return _at_least(PLANNING["startup_s"][which], (self.m or {}).get("startup_s"), which)

    def faithful_build_s(self, which: str) -> float:
        """The faithful snapshot build, paid in every container (each chunk rebuilds it)."""
        build = ((self.m or {}).get("faithful") or {}).get("build_s")
        return _at_least(PLANNING["faithful_build_s"][which], build, which)


def _at_least(planning: float, measured: float | None, which: str) -> float:
    if measured is None:
        return float(planning)
    return float(max(planning, measured if which == CENTRAL else 1.5 * measured))


def _started(row: Mapping[str, Any]) -> bool:
    """The run already trained some of its epochs (gate `progress`)."""
    return bool(row.get("done_units") or row.get("done_epochs"))


def _cost(
    row: Mapping[str, Any], which: str, *, units: float | None = None, epochs: float | None = None
) -> dict[str, float]:
    """{work_s, n_containers, run_s, usd} of a planned row (optionally with other units/epochs).
    A set needs one container per chunk_wall_s of work (each pays its startup). A started run
    (gate `progress`: done_units finished seeds / trials, done_epochs of the unfinished one) pays
    only the epochs it has left; its scoring and startups are charged in full."""
    if row.get("fixed"):
        return {
            "work_s": row["run_s"][which],
            "n_containers": row["n_containers"][which],
            "run_s": row["run_s"][which],
            "usd": row["usd"][which],
        }
    u = row["units"] if units is None else units
    e = row["epochs"][which] if epochs is None else epochs
    done = float(row.get("done_units") or 0.0) * e + float(row.get("done_epochs") or 0.0)
    left = max(0.0, u * e - done)
    work = left * row["epoch_s"][which] + (row["scoring_s"][which] if u > 0 else 0.0)
    n = max(1, math.ceil(round(work / row["chunk_wall_s"], 9))) if work > 0 else 0
    run_s = run_seconds(
        startup_s=row["startup_s"][which],
        n_containers=n,
        epochs=left,
        epoch_s=row["epoch_s"][which],
        scoring_s=row["scoring_s"][which] if u > 0 else 0.0,
    )
    return {
        "work_s": work,
        "n_containers": n,
        "run_s": run_s,
        "usd": usd(row["price_usd_h"], run_s, row["overhead"]),
    }


def _finish(row: dict) -> dict:
    costs = {w: _cost(row, w) for w in WHICH}
    for k in ("work_s", "n_containers", "run_s", "usd"):
        row[k] = {w: costs[w][k] for w in WHICH}
    row["usd_epoch_cons"] = usd(row["price_usd_h"], row["epoch_s"][CONS], row["overhead"])
    return row


def plan_runs(
    gnn_cfg: Mapping[str, Any],
    *,
    decision: Mapping[str, Any] | None = None,
    counts: Mapping[str, int] | None = None,
) -> list[dict]:
    """plan.json rows, one per RUN_ORDER entry (schema in the module docstring). Uses the bench
    decision's measured timings (decision["measured"]) and shape when given, else the §11.1
    planning constants on L4 x 8 cores x 32 GiB (HI-Small counts of §2.1 unless `counts` or
    the bench's build cell gives others). The train batch is the decision's, else gnn.yaml's
    sampler.batch_size; the eval batch is sampler.eval_batch_size. Conservative = max epochs
    (no early-stopping credit), conservative startup / epoch overhead and the uncached val pass;
    central = 60% of max epochs (faithful and dev: max in both). A set needs one container per
    runtime.chunk_wall_s of work. Every $ is at plan_usd_h x budget.overhead."""
    measured = (decision or {}).get("measured")
    c = _counts((measured or {}).get("counts"), counts)
    t = _Timing(gnn_cfg, c, measured)
    if decision is not None:
        gpu, cores = str(decision["gpu"]), int(decision["cores"])
        mem = int(decision["memory_mib"])
        fd = decision.get("faithful") or {}
        fgpu = str(fd.get("gpu") or gpu)
        fbatch = fd.get("batch_size")
        # the faithful run keeps the shape its cell was measured on (perf-2)
        fcores, fmem = int(fd.get("cores") or cores), int(fd.get("memory_mib") or mem)
    else:
        gpu, cores, mem = PLANNING["gpu"], PLANNING["cores"], PLANNING["memory_mib"]
        fgpu, fbatch, fcores, fmem = gpu, None, cores, mem
    fbatch = int(fbatch or gnn_cfg["protocols"]["faithful"]["batch_size"])
    overhead = float(gnn_cfg["budget"]["overhead"])
    chunk = float(gnn_cfg["runtime"]["chunk_wall_s"])
    protos, tr, hpo = gnn_cfg["protocols"], gnn_cfg["train"], gnn_cfg["hpo"]

    def base(
        run, protocol, seeds, units, max_epochs, central_epochs, g=gpu, c=cores, m=mem
    ) -> dict:
        return {
            "run": run,
            "protocol": protocol,
            "seeds": [int(s) for s in seeds],
            "units": units,
            "n_trials": None,
            "max_epochs": max_epochs,
            "epochs": {CENTRAL: float(central_epochs), CONS: float(max_epochs)},
            "gpu": g,
            "cores": c,
            "memory_mib": m,
            "shape_usd_h": shape_usd_h(g, c, m / 1024),
            "price_usd_h": plan_usd_h(g, c, m / 1024),
            "batch_size": t.batch,
            "startup_s": _both(t.startup_s),
            "overhead": overhead,
            "chunk_wall_s": chunk,
            "epoch_cap": None,
            "cut_alt": None,
            "source": "planning",
            "status": "planned",
        }

    def temporal(row: dict, splits: Sequence[str], *, lookahead=False, factor=None) -> dict:
        k = row["units"]
        src = "planning"
        for w in WHICH:
            ep, src = t.epoch_s(w, lookahead=lookahead)
            sc = t.scoring_s(w, splits, k, lookahead=lookahead)
            if factor is not None:
                f, src = factor(w)
                ep, sc = ep * f, sc * f
            row.setdefault("epoch_s", {})[w] = ep
            row.setdefault("scoring_s", {})[w] = sc
        row["source"] = src
        return _finish(row)

    rows: list[dict] = []

    # bench: L4, T4 (if benched) and the 4-core rerun, + driver and T4 smoke
    secs = PLANNING["bench_container_s"]
    shapes = [(g, 8, 32, secs[g]) for g in gnn_cfg["bench"]["gpus"] if g in secs] + [
        ("L4", RERUN_CORES, RERUN_MEMORY_MIB // 1024, secs["rerun"])
    ]
    b_s = sum(s for *_, s in shapes)
    b_usd = sum(plan_usd_h(g, k, gib) / 3600 * s for g, k, gib, s in shapes)
    b_usd += PLANNING["bench_fixed_usd"]
    fac = {CENTRAL: 1.0, CONS: PLANNING["bench_cons_factor"]}
    bench = base("bench", None, [], 1, 0, 0)
    bench.update(
        fixed=True,
        epoch_s={w: 0.0 for w in WHICH},
        scoring_s={w: 0.0 for w in WHICH},
        work_s={w: b_s * fac[w] for w in WHICH},
        n_containers={w: len(shapes) for w in WHICH},
        run_s={w: b_s * fac[w] for w in WHICH},
        usd={w: b_usd * fac[w] * overhead for w in WHICH},
        usd_epoch_cons=0.0,
    )
    rows.append(bench)

    dev = base("dev", "causal", [0], 1, DEV_MAX_EPOCHS, DEV_MAX_EPOCHS)
    rows.append(temporal(dev, SCORED_DEV))

    n_trials = int(hpo["n_trials"])
    h = base(
        "hpo",
        "causal",
        [hpo["model_seed"]],
        n_trials,
        int(hpo["max_epochs"]),
        CENTRAL_EPOCH_SHARE * int(hpo["max_epochs"]),
    )
    h["n_trials"] = n_trials
    rows.append(temporal(h, ()))
    if n_trials > HPO_CUT_TRIALS:
        alt = {w: _cost(h, w, units=HPO_CUT_TRIALS) for w in WHICH}
        h["cut_alt"] = {
            "n_trials": HPO_CUT_TRIALS,
            "run_s": {w: alt[w]["run_s"] for w in WHICH},
            "usd": {w: alt[w]["usd"] for w in WHICH},
        }

    max_e = int(tr["max_epochs"])
    central_e = CENTRAL_EPOCH_SHARE * max_e
    causal_seeds = list(protos["causal"]["seeds"])
    rows.append(
        temporal(
            base("causal", "causal", causal_seeds, len(causal_seeds), max_e, central_e),
            SCORED_FINAL,
        )
    )

    la = list(protos["lookahead"]["seeds"])
    rows.append(
        temporal(
            base("lookahead_s0", "lookahead", la[:1], len(la[:1]), max_e, central_e),
            SCORED_LOOKAHEAD,
            lookahead=True,
        )
    )
    la_rest = temporal(
        base("lookahead_rest", "lookahead", la[1:], len(la[1:]), max_e, central_e),
        SCORED_LOOKAHEAD,
        lookahead=True,
    )

    fp = protos["faithful"]
    f_max = int(fp["max_epochs"])
    f_e = min(f_max, int(fp["epoch_cap"])) if fp.get("epoch_cap") is not None else f_max
    fr = base("faithful", "faithful", [fp["seed"]], 1, f_max, f_e, g=fgpu, c=fcores, m=fmem)
    fr["epochs"] = {CENTRAL: float(f_e), CONS: float(f_e)}
    fr["epoch_cap"] = fp.get("epoch_cap")
    fr["batch_size"] = fbatch
    fr["startup_s"] = {w: fr["startup_s"][w] + t.faithful_build_s(w) for w in WHICH}
    fr["epoch_s"], fr["scoring_s"] = {}, {}
    for w in WHICH:
        fr["epoch_s"][w], fr["scoring_s"][w], fr["source"] = t.faithful(w, fbatch)
    rows.append(_finish(fr))

    rows.append(la_rest)
    if not la[1:]:
        la_rest.update(status="cut", cut="lookahead_1")  # gnn.yaml already keeps seed 0 only

    pna_seeds = list(protos["pna"]["seeds"])
    rows.append(
        temporal(
            base("pna", "pna", pna_seeds, len(pna_seeds), max_e, central_e),
            SCORED_FINAL,
            factor=t.pna_factor,
        )
    )
    assert tuple(r["run"] for r in rows) == RUN_ORDER
    return rows


# --- the bench decision (§10.3) ----------------------------------------------------------------


def _ok(cells: Sequence[Mapping[str, Any]], *groups: str) -> list[Mapping[str, Any]]:
    return [c for c in cells if c.get("status") == "ok" and c.get("group") in groups]


def _gib(memory_mib: float) -> float:
    return float(memory_mib) / 1024.0


def _on_cpu(cells, gpu: str) -> bool:
    """Cells measured on a CPU device record no device memory (bench._device_info)."""
    same = [c for c in cells if c.get("gpu") == gpu]
    return bool(same) and not any(c.get("device_bytes") for c in same)


def _train_cap(cells, gpu: str, memory_fraction: float) -> tuple[int | None, float | None]:
    """(max_edges_per_step, bytes per edge) of a GPU from its stress cells (the worst bytes per
    edge; the grid cells' own figures if no stress cell succeeded). (None, None) if unmeasured
    (and always on a CPU device: no device memory, no split, M3 spec §5.7)."""
    src = [
        c
        for c in _ok(cells, "stress")
        if c["gpu"] == gpu and c.get("bytes_per_edge") and c.get("device_bytes")
    ]
    if not src:
        src = [
            c
            for c in _ok(cells, "grid")
            if c["gpu"] == gpu and c.get("bytes_per_edge") and c.get("device_bytes")
        ]
    if not src:
        return None, None
    bpe = max(float(c["bytes_per_edge"]) for c in src)
    dev = max(int(c["device_bytes"]) for c in src)
    return int(math.floor(memory_fraction * dev / bpe)), bpe


def _t_val(v: Mapping[str, Any], *, cached: bool) -> float:
    """The val_early pass: cached if asked and the cache fits (and was timed), else uncached."""
    if cached and v.get("cache_fits") and v.get("val_cached_s") is not None:
        return float(v["val_cached_s"])
    return float(v["val_s"])


def _val_cell(cells, cell: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """The val pass of a grid / rerun cell's container (a rerun cell may carry its own)."""
    if cell.get("val_s") is not None:
        return cell
    same = [c for c in _ok(cells, "val") if c["gpu"] == cell["gpu"]]
    exact = [c for c in same if int(c["cores"]) == int(cell["cores"])]
    pool = exact or sorted(same, key=lambda c: -int(c["cores"]))
    return pool[0] if pool else None


def _entries(cells, cfg, counts, memory_mib: int, groups=("grid", "rerun")) -> list[dict]:
    """The per-cell table: epoch seconds and $/epoch of every ok train-step cell (rule 1, 2)."""
    b = cfg["bench"]
    ovh = PLANNING["epoch_overhead_s"][CENTRAL]
    out = []
    for c in _ok(cells, *groups):
        gpu, cores, batch = c["gpu"], int(c["cores"]), int(c["batch_size"])
        cap, _ = _train_cap(cells, gpu, float(b["memory_fraction"]))
        v = _val_cell(cells, c)
        steps = steps_per_epoch(counts["n_pos"], counts["n_neg"], cfg["train"]["neg_rate"], batch)
        e: dict[str, Any] = {
            "cell_id": c.get("cell_id") or cell_id(gpu, cores, c["group"], batch, c["num_workers"]),
            "group": c["group"],
            "gpu": gpu,
            "cores": cores,
            "batch_size": batch,
            "num_workers": int(c["num_workers"]),
            "step_s_mean": float(c["step_s_mean"]),
            "step_s_p90": c.get("step_s_p90"),
            "wait_share": c.get("wait_share"),
            "edges_mean": float(c.get("edges_mean") or 0.0),
            "peak_mem_bytes": c.get("peak_mem_bytes"),
            "edge_cap": cap,
            "steps": steps,
            "val_cell": (v or {}).get("cell_id"),
            "epoch_s": None,
            "usd_epoch": None,
            "eligible": False,
            "why": None,
        }
        if v is None:
            e["why"] = "no val pass on this GPU"
        elif cap is None and not _on_cpu(cells, gpu):
            e["why"] = "no bytes/edge measured on this GPU"
        elif cap is not None and cap < ELIGIBLE_CAP_FACTOR * e["edges_mean"]:
            e["why"] = f"edge cap {cap} < {ELIGIBLE_CAP_FACTOR} x mean batch edges"
        else:
            e["eligible"] = True
        if v is not None:
            t_val = _t_val(v, cached=True)
            e["epoch_s"] = epoch_seconds(steps, e["step_s_mean"], t_val, ovh)
            e["usd_epoch"] = usd(shape_usd_h(gpu, cores, _gib(memory_mib)), e["epoch_s"])
        out.append(e)
    return out


def _pick(entries: Sequence[dict], tie_band: float) -> dict:
    """Rule 2: argmin $/epoch; within tie_band of the best prefer L4, then the smaller batch."""
    best = min(e["usd_epoch"] for e in entries)
    band = [e for e in entries if e["usd_epoch"] <= best * (1.0 + tie_band)]
    band = [e for e in band if e["gpu"] == "L4"] or band
    small = min(e["batch_size"] for e in band)
    band = [e for e in band if e["batch_size"] == small]
    return min(band, key=lambda e: (e["usd_epoch"], e["cell_id"]))


def _peak_mib(cells) -> float:
    vals = [float(c["host_peak_mib"]) for c in cells if c.get("host_peak_mib") is not None]
    return max(vals) if vals else 0.0


def _ref_step(cells, like: Mapping[str, Any]) -> float | None:
    """The headline grid step time matching a lookahead / PNA cell (same GPU, cores, batch,
    workers; else the fastest grid cell of that GPU and batch)."""
    grid = [c for c in _ok(cells, "grid") if c["gpu"] == like["gpu"]]
    same_b = [c for c in grid if int(c["batch_size"]) == int(like["batch_size"])]
    exact = [
        c
        for c in same_b
        if int(c["cores"]) == int(like["cores"]) and c["num_workers"] == like["num_workers"]
    ]
    pool = exact or same_b
    return min(float(c["step_s_mean"]) for c in pool) if pool else None


def _measured(cells, cfg, counts, best: dict, v: Mapping[str, Any], faithful: dict) -> dict:
    """The timings plan_runs needs, on the decided shape (look-ahead and PNA are measured on L4
    only: their ratio to the matching L4 grid cell is applied to the decided cell)."""
    builds = [float(c["build_s"]) for c in _ok(cells, "build") if c.get("build_s") is not None]
    m: dict[str, Any] = {
        "counts": dict(counts),
        "batch_size": best["batch_size"],
        "train_step_s": best["step_s_mean"],
        "val_s": float(v["val_s"]),
        "val_cached_s": float(v.get("val_cached_s") or v["val_s"]),
        "cache_fits": bool(v.get("cache_fits")),
        "eval_batches": int(v.get("eval_batches") or 1),
        "startup_s": max(builds) if builds else None,
        "lookahead": None,
        "pna": None,
        "faithful": None,
    }
    val_pb = m["val_s"] / m["eval_batches"]
    la = _ok(cells, "lookahead")
    if la:
        c = la[0]
        ref = _ref_step(cells, c)
        l4_val = [x for x in _ok(cells, "val") if x["gpu"] == c["gpu"]]
        if ref and l4_val and c.get("val_s_per_batch"):
            l4_pb = float(l4_val[0]["val_s"]) / max(1, int(l4_val[0].get("eval_batches") or 1))
            m["lookahead"] = {
                "step_s": best["step_s_mean"] * float(c["step_s_mean"]) / ref,
                "val_s_per_batch": val_pb * float(c["val_s_per_batch"]) / l4_pb,
                "val_first_batch_s": c.get("val_first_batch_s"),
                "val_cached_s": None,
            }
            # TemporalEngine caches the look-ahead val trees too (the val bound is fixed): the
            # central epoch replays them, the headline cached pass scaled by edges per target
            la_ept = c.get("edges_per_target_val")
            head_ept = None
            if v.get("eval_edges") and v.get("eval_rows"):
                head_ept = float(v["eval_edges"]) / float(v["eval_rows"])
            if m["cache_fits"] and la_ept and head_ept:
                ratio = float(la_ept) / head_ept
                fits = float(v.get("cache_gb") or 0.0) * ratio <= float(
                    cfg["sampler"]["eval_cache_max_gb"]
                )
                if fits:
                    m["lookahead"]["val_cached_s"] = m["val_cached_s"] * ratio
    pn = _ok(cells, "pna")
    if pn:
        ref = _ref_step(cells, pn[0])
        if ref:
            m["pna"] = {"step_s": best["step_s_mean"] * float(pn[0]["step_s_mean"]) / ref}
    if faithful.get("cell") is not None:
        f = faithful["cell"]
        m["faithful"] = {
            "step_s": float(f["step_s_mean"]),
            "val_s_per_batch": float(f["val_s_per_batch"]),
            "test_s_per_batch": float(f["test_s_per_batch"]),
            "val_first_batch_s": f.get("val_first_batch_s"),
            "test_first_batch_s": f.get("test_first_batch_s"),
            "build_s": None if f.get("build_s") is None else float(f["build_s"]),
        }
    return m


DET_KINDS = ("train", "eval", "faithful")


def det_ratios(det: Mapping[str, Any]) -> dict[str, float] | None:
    """{train, eval, faithful} step-time ratios (deterministic algorithms / without) of a
    determinism cell; eval and faithful fall back to the train ratio when not measured. None
    without a train ratio."""
    r = det.get("det_overhead_ratio_train") or det.get("det_overhead_ratio")
    if not r or float(r) <= 0:
        return None
    out = {"train": float(r)}
    for k in ("eval", "faithful"):
        v = det.get(f"det_overhead_ratio_{k}")
        out[k] = float(v) if v and float(v) > 0 else float(r)
    return out


def _scale_measured(m: Mapping[str, Any], factors: Mapping[str, float] | float) -> dict:
    """Every measured GPU time x the factor of its kind (rule 6: deterministic algorithms off):
    train steps (headline, look-ahead, PNA) x factors["train"]; eval passes (val, cached val,
    look-ahead val, faithful val / test) x factors["eval"]; faithful train steps x
    factors["faithful"]. A float scales every kind. Other fields (first batches, build) stay."""
    f = factors if isinstance(factors, Mapping) else {k: float(factors) for k in DET_KINDS}
    out = copy.deepcopy(dict(m))

    def scale(d: dict | None, key: str, kind: str) -> None:
        if d is not None and d.get(key) is not None:
            d[key] = d[key] * float(f[kind])

    scale(out, "train_step_s", "train")
    scale(out, "val_s", "eval")
    scale(out, "val_cached_s", "eval")
    la, pn, fa = out.get("lookahead"), out.get("pna"), out.get("faithful")
    scale(la, "step_s", "train")
    scale(la, "val_s_per_batch", "eval")
    scale(la, "val_cached_s", "eval")
    scale(pn, "step_s", "train")
    scale(fa, "step_s", "faithful")
    scale(fa, "val_s_per_batch", "eval")
    scale(fa, "test_s_per_batch", "eval")
    return out


def _decide_faithful(cells, cfg, counts, memory_mib: int) -> dict:
    """Rule 4: among GPUs whose faithful cell fit at the largest batch, the cheaper $/epoch,
    each priced at the shape its cell was measured on (its cores and loader workers: the
    faithful run keeps that shape, perf-2)."""
    fc = [c for c in _ok(cells, "faithful") if c.get("batch_size")]
    if not fc:
        return {
            "gpu": None,
            "batch_size": None,
            "usd_epoch": None,
            "cores": None,
            "num_workers": None,
            "cell": None,
        }
    top = max(int(c["batch_size"]) for c in fc)
    ovh = PLANNING["epoch_overhead_s"][CENTRAL]
    n_train = counts["n_pos"] + counts["n_neg"]
    best = None
    for c in sorted(fc, key=lambda c: (c["gpu"] != "L4", c["gpu"])):
        if int(c["batch_size"]) != top:
            # the largest batch first (8192 = the recalled Multi-GNN CLI default, PLAN §6; not
            # a published value): a GPU that needed a smaller batch loses
            continue
        n_val = _batches(counts["n_val_early"] + counts["n_val_late"], top)
        ep = _batches(n_train, top) * float(c["step_s_mean"])
        ep += n_val * float(c["val_s_per_batch"]) + float(c.get("val_first_batch_s") or 0.0)
        ep += ovh
        u = usd(shape_usd_h(c["gpu"], int(c["cores"]), _gib(memory_mib)), ep)
        if best is None or u < best["usd_epoch"]:
            best = {
                "gpu": c["gpu"],
                "batch_size": top,
                "usd_epoch": u,
                "epoch_s": ep,
                "cores": int(c["cores"]),
                "num_workers": None if c.get("num_workers") is None else int(c["num_workers"]),
                "cell": c,
            }
    return best


def decide(cells: Sequence[Mapping[str, Any]], cfg: Mapping[str, Any]) -> dict:
    """The pre-registered decision rules (M3 spec §10.3) on the measured cells; cfg = the whole
    gnn config. 1. a cell is eligible if its edge cap >= 1.5 x its mean batch edges; 2. GPU,
    batch, workers and shape = argmin headline usd_epoch, within bench.tie_band prefer L4, then
    batch 2048 (T4 wins only if cheaper by more than the band); 3. precision follows the GPU;
    4. faithful GPU: among GPUs whose faithful cell fit at the largest batch (8192, the recalled
    CLI default), the cheaper faithful usd_epoch at the cell's own shape (a GPU needing a smaller
    batch loses); 5. memory_request_mib of the peak host memory over every cell; 6. if the
    determinism overhead alone pushes the conservative M3 projection (after cuts 1-3, spent 0)
    above budget.m3_cap_usd, `ask_user` explains it (the driver then raises GnnStopError); the
    "without determinism" projection divides train steps, eval passes and faithful steps by
    their own measured ratios; no faithful batch fitting also sets `ask_user`;
    7. returns decision.json (schema in the module docstring). ValueError if no cell is
    eligible."""
    b = cfg["bench"]
    counts = _counts(*(c.get("counts") for c in _ok(cells, "build")))
    mem_req, mem_lim = memory_request_mib(_peak_mib(cells))
    table = _entries(cells, cfg, counts, mem_req)
    eligible = [e for e in table if e["eligible"]]
    if not eligible:
        why = sorted({f"{e['cell_id']}: {e['why']}" for e in table}) or ["no ok grid cell"]
        raise ValueError("gnn_bench: no eligible cell (rule 1):\n  " + "\n  ".join(why))
    best = _pick(eligible, float(b["tie_band"]))
    by_id = {c.get("cell_id"): c for c in cells}
    raw = by_id.get(best["cell_id"]) or {}
    v = _val_cell(cells, raw or best)
    if v is None:
        raise ValueError(f"gnn_bench: no val pass for {best['cell_id']}")

    frac = float(b["memory_fraction"])
    cap, bpe = _train_cap(cells, best["gpu"], frac)
    dev_bytes = max(
        [
            int(c["device_bytes"])
            for c in cells
            if c.get("gpu") == best["gpu"] and c.get("device_bytes")
        ]
        or [0]
    )
    if v.get("eval_bytes_per_edge") and dev_bytes:
        eval_bpe = float(v["eval_bytes_per_edge"])
        eval_cap = int(math.floor(frac * dev_bytes / eval_bpe))
    else:
        eval_bpe = None
        eval_cap = v.get("max_edges_per_eval_step") or (4 * cap if cap is not None else None)

    faithful = _decide_faithful(cells, cfg, counts, mem_req)
    det = (_ok(cells, "determinism") or [{}])[0]
    ratios = det_ratios(det)
    ratio = ratios["train"] if ratios else det.get("det_overhead_ratio")
    asks = []
    if faithful["gpu"] is None:
        tried = b["faithful"]["batch_fallback"]
        asks.append(f"no faithful batch fits on any GPU (tried {tried}): ask the user")

    decision: dict[str, Any] = {
        "gpu": best["gpu"],
        "cores": best["cores"],
        "memory_mib": mem_req,
        "memory_limit_mib": mem_lim,
        "num_workers": best["num_workers"],
        "batch_size": best["batch_size"],
        "max_edges_per_step": cap,
        "max_edges_per_eval_step": eval_cap,
        "eval_cache": {"fits": bool(v.get("cache_fits")), "gb": v.get("cache_gb")},
        "faithful": {
            **{k: faithful[k] for k in ("gpu", "batch_size", "usd_epoch", "cores", "num_workers")},
            "memory_mib": mem_req if faithful["gpu"] is not None else None,
        },
        "bytes_per_edge": {"train": bpe, "eval": eval_bpe},
        "precision": PRECISION.get(best["gpu"], "highest"),
        "bit_deterministic": det.get("bit_deterministic"),
        "det_overhead_ratio": ratio,
        "det_overhead_ratios": ratios,
        "usd_epoch": best["usd_epoch"],
        "epoch_s": best["epoch_s"],
        "cell_id": best["cell_id"],
        "rerun_4core": any(c.get("group") == "rerun" for c in _ok(cells, "rerun")),
        "ask_user": None,
        "projection_usd": None,
        "counts": counts,
        "measured": _measured(cells, cfg, counts, best, v, faithful),
        "cells": table,
    }
    if ratios:
        cap_usd = float(cfg["budget"]["m3_cap_usd"])
        on = _projection(plan_runs(cfg, decision=decision), cap_usd)[0]
        inv = {k: 1.0 / v for k, v in ratios.items()}
        off_dec = {**decision, "measured": _scale_measured(decision["measured"], inv)}
        off = _projection(plan_runs(cfg, decision=off_dec), cap_usd)[0]
        decision["projection_usd"] = {"deterministic": on, "without_determinism": off}
        if on > cap_usd >= off:
            asks.append(
                f"deterministic algorithms slow a train step x{ratios['train']:.2f}, an eval "
                f"forward x{ratios['eval']:.2f} and a faithful step x{ratios['faithful']:.2f}: "
                f"the conservative M3 projection is ${on:.2f} with them and ${off:.2f} without "
                f"(cap ${cap_usd:.2f}); ask the user whether to keep them (PLAN) or turn them off"
            )
    decision["ask_user"] = "; ".join(asks) or None
    return decision


def rerun_4core(cells: Sequence[Mapping[str, Any]], gnn_cfg: Mapping[str, Any]) -> dict | None:
    """The 4-core rerun rule: if the winning GPU's best batch at 3 workers (8 cores) has a
    sampler wait share < bench.rerun_4core_if_wait_below, return {"gpu", "batch_size",
    "num_workers": 3, "cores": 4, "memory_mib": 16384}; else None. (3 = the smallest
    bench.num_workers; the winner = rules 1-2 over the 8-core grid cells.)"""
    b = gnn_cfg["bench"]
    counts = _counts(*(c.get("counts") for c in _ok(cells, "build")))
    mem_req, _ = memory_request_mib(_peak_mib(cells))
    grid = [e for e in _entries(cells, gnn_cfg, counts, mem_req, ("grid",)) if e["eligible"]]
    if not grid:
        return None
    win = _pick(grid, float(b["tie_band"]))
    w3 = min(int(w) for w in b["num_workers"])
    pool = [
        e
        for e in grid
        if e["gpu"] == win["gpu"] and e["cores"] == win["cores"] and e["num_workers"] == w3
    ]
    if not pool:
        return None
    best = min(pool, key=lambda e: (e["usd_epoch"], e["cell_id"]))
    share = best.get("wait_share")
    if share is None or float(share) >= float(b["rerun_4core_if_wait_below"]):
        return None
    return {
        "gpu": best["gpu"],
        "batch_size": best["batch_size"],
        "num_workers": w3,
        "cores": RERUN_CORES,
        "memory_mib": RERUN_MEMORY_MIB,
    }


# --- the cost gate (§11.4) ----------------------------------------------------------------------


def _planned_usd(rows: Sequence[Mapping[str, Any]]) -> float:
    return sum(float(r["usd"][CONS]) for r in rows if r["status"] == "planned")


def _apply_cuts(rows: list[dict], cap: float, spent: float) -> tuple[float, list[str]]:
    """CUT_ORDER on runs not yet run (in place), while spent + planned conservative $ > cap. A
    started run is not cut: its epochs are paid for, and cut 2 (hpo.n_trials) would re-key it."""
    by_run = {r["run"]: r for r in rows}
    projected, cuts = spent + _planned_usd(rows), []
    for cut in CUT_ORDER:
        if projected <= cap:
            break
        r = by_run.get(CUT_TARGET[cut])
        if r is None or r["status"] != "planned" or _started(r):
            continue
        if cut == "hpo_4":
            alt = r.get("cut_alt")
            if not alt or float(alt["usd"][CONS]) >= float(r["usd"][CONS]):
                continue
            r.update(
                n_trials=alt["n_trials"],
                units=alt["n_trials"],
                usd=dict(alt["usd"]),
                run_s=dict(alt["run_s"]),
                cut=cut,
            )
        else:
            r.update(status="cut", cut=cut)
        cuts.append(cut)
        projected = spent + _planned_usd(rows)
    return projected, cuts


def job_bound_usd(rows: Sequence[Mapping[str, Any]], runtime: Mapping[str, Any]) -> float:
    """What one submission of these planned rows can spend before its driver stops it (money
    review MONEY-1): price x set_wall_limit_s / 3600, set_wall_limit_s = wall_guard_factor x the
    rows' conservative run seconds + chunk_wall_s (run_limits; drive_chunks starts no chunk
    that could pass it), and at least their conservative $. A fixed row (the bench: per-
    container timeouts, no chunk driver) counts its conservative $."""
    rows = list(rows)
    fixed = sum(float(r["usd"][CONS]) for r in rows if r.get("fixed"))
    var = [r for r in rows if not r.get("fixed")]
    if not var:
        return fixed
    projected = sum(float(r["run_s"][CONS]) for r in var)
    limit = float(runtime["wall_guard_factor"]) * projected + float(runtime["chunk_wall_s"])
    price = max(float(r["price_usd_h"]) for r in var)
    return fixed + max(sum(float(r["usd"][CONS]) for r in var), usd(price, limit))


def _projection(plan: Sequence[Mapping[str, Any]], cap: float, spent: float = 0.0) -> tuple:
    rows = [copy.deepcopy(dict(r)) for r in plan]
    return _apply_cuts(rows, cap, spent)


def _faithful_cap(f: dict, budget_usd: float) -> int:
    """The most faithful epochs whose conservative $ (startups, test pass and build included)
    fits in budget_usd; 0 if not even the fixed part fits."""
    for e in range(int(f["epochs"][CONS]), -1, -1):
        if _cost(f, CONS, epochs=e)["usd"] <= budget_usd:
            return e
    return 0


def gate(
    plan: Sequence[Mapping[str, Any]],
    *,
    gnn_cfg: Mapping[str, Any],
    spent_usd: float,
    done: Collection[str],
    job: str | Sequence[str],
    metered_usd: float | None,
    progress: Mapping[str, Mapping[str, float]] | None = None,
) -> dict:
    """The cost gate (M3 spec §11.4) run by every GPU job's local entrypoint before submitting.

    plan: plan_runs(...) rows; spent_usd: measured M3 spend INCLUDING budget.dev_allowance_usd
    (see `measured_spend`); done: RUN_ORDER ids already finished (from the Volume summaries);
    job: the RUN_ORDER id about to be submitted (or several, e.g. both look-ahead rows when one
    submission trains every look-ahead seed); metered_usd: the workspace's metered cost this
    cycle (`modal billing summary --json`), None if unavailable (then the pre-check refuses);
    progress: {run id: {"units_done", "epochs_done"}} of started runs not done (finished seeds /
    HPO trials, checkpointed epochs of the unfinished one; modal_jobs.train_gnn.run_progress):
    their spend is already in spent_usd, so they are charged only for what is left.
    projected = spent + conservative $ of every planned run not done (job included); while
    projected > m3_cap_usd apply CUT_ORDER cuts of runs not yet run (a started run is not cut);
    if still above, faithful epoch_cap = the most epochs whose conservative $ fits cap - spent -
    other remaining (the fixed part included), refusing below MIN_FAITHFUL_EPOCHS (or when
    nothing is left to cut). A cap changes the faithful run key, so for a started faithful run
    it is computed as a restart from epoch 0 (and says so in the warnings).
    The job is refused when the gate cuts it, when a cut / cap concerning it is not yet applied
    in gnn.yaml, or when the budget pre-check fails: metered + the job's bound (job_bound_usd:
    what the submission can spend before its driver stops it, >= its conservative $) <=
    workspace_budget_usd - m6_reserve_usd. The M3-cap projection charges the conservative $.

    Returns {"job", "allowed": bool, "refuse_reason": str | None, "spent_usd",
    "projected_usd", "cap_usd", "cuts": [cut ids applied], "faithful_epoch_cap": int | None,
    "budget_check": {"metered_usd", "job_usd", "job_bound_usd", "limit_usd", "ok"},
    "actions": [gnn.yaml edits
    the cuts / cap need], "warnings": [...], "rows": plan rows with status "planned" | "done" |
    "cut"}. The job must not be submitted unless allowed; the lead applies cuts by editing
    gnn.yaml (hpo.n_trials, protocols.lookahead.seeds, protocols.faithful.epoch_cap)."""
    b = gnn_cfg["budget"]
    cap = float(b["m3_cap_usd"])
    limit = float(b["workspace_budget_usd"]) - float(b["m6_reserve_usd"])
    jobs = [job] if isinstance(job, str) else list(job)
    done = set(done)
    bad = sorted((set(jobs) | done) - set(RUN_ORDER))
    if bad or not jobs:
        raise ValueError(f"unknown run ids {bad or jobs}; expected ids from {RUN_ORDER}")
    rows = [copy.deepcopy(dict(r)) for r in plan]
    by_run = {r["run"]: r for r in rows}
    missing = [j for j in jobs if j not in by_run]
    if missing:
        raise ValueError(f"job(s) {missing} are not in the plan")
    for r in rows:
        if r["run"] in done:
            r["status"] = "done"
        elif r.get("status") != "cut":
            r["status"] = "planned"
    unknown = sorted(set(progress or {}) - set(RUN_ORDER))
    if unknown:
        raise ValueError(f"progress for unknown run ids {unknown}; expected ids from {RUN_ORDER}")
    for r in rows:
        p = (progress or {}).get(r["run"]) or {}
        if r["status"] != "planned" or r.get("fixed"):
            continue
        du, de = float(p.get("units_done") or 0.0), float(p.get("epochs_done") or 0.0)
        if du < 0 or de < 0:
            raise ValueError(f"negative progress for {r['run']}: {p}")
        if du or de:
            r["done_units"], r["done_epochs"] = du, de
            _finish(r)

    spent = float(spent_usd)
    projected, cuts = _apply_cuts(rows, cap, spent)
    refuse: list[str] = []
    actions: list[str] = []
    warnings: list[str] = []
    f_cap = None
    if projected > cap:
        f = by_run.get("faithful")
        if f is not None and f["status"] == "planned":
            if _started(f):  # a new epoch_cap re-keys the faithful run: it starts over
                warnings.append(
                    f"the faithful run has trained {f['done_epochs']:g} epochs under its current "
                    "key; an epoch cap changes the key, so the cap below assumes a restart from "
                    "epoch 0 (ask the user before giving up those epochs)"
                )
                f["done_units"] = f["done_epochs"] = 0.0
                _finish(f)
            others = _planned_usd([r for r in rows if r is not f])
            f_cap = _faithful_cap(f, cap - spent - others)
            if f_cap < MIN_FAITHFUL_EPOCHS:
                refuse.append(
                    f"even after cuts {cuts or '[]'} the faithful run fits only {f_cap} epochs "
                    f"(< {MIN_FAITHFUL_EPOCHS}) in the ${cap:.2f} cap: ask the user (raise the "
                    "budget and pay the difference, or accept a lower cap)"
                )
            else:
                f["epochs"] = {w: min(float(f["epochs"][w]), float(f_cap)) for w in WHICH}
                f["epoch_cap"] = f_cap
                _finish(f)
                projected = spent + _planned_usd(rows)
        else:
            refuse.append(
                f"projected ${projected:.2f} > cap ${cap:.2f} with nothing left to cut: "
                "ask the user"
            )

    seeds0 = list(gnn_cfg["protocols"]["lookahead"]["seeds"][:1])
    edits = {
        "pna": "skip the PNA run (cut 1)",
        "hpo_4": f"hpo.n_trials: {HPO_CUT_TRIALS} (cut 2)",
        "lookahead_1": f"protocols.lookahead.seeds: {seeds0} (cut 3)",
    }
    actions += [edits[c] for c in cuts]
    cfg_cap = gnn_cfg["protocols"]["faithful"].get("epoch_cap")
    if f_cap is not None and f_cap >= MIN_FAITHFUL_EPOCHS:
        actions.append(f"protocols.faithful.epoch_cap: {f_cap} (a documented deviation)")

    for j in jobs:
        r = by_run[j]
        if r["status"] == "cut":
            how = f"cut {r.get('cut')}" if r.get("cut") else "cut"
            refuse.append(f"the gate cuts {j!r} ({how}); do not submit it")
        elif j == "hpo" and "hpo_4" in cuts:
            refuse.append(f"apply cut 2 first: set hpo.n_trials: {HPO_CUT_TRIALS} in gnn.yaml")
        elif j == "faithful" and f_cap is not None and (cfg_cap is None or cfg_cap > f_cap):
            refuse.append(f"apply the cap first: set protocols.faithful.epoch_cap: {f_cap}")
    earlier = [
        r
        for r in RUN_ORDER[: min(RUN_ORDER.index(j) for j in jobs)]
        if by_run[r]["status"] == "planned"
    ]
    if earlier:
        warnings.append(f"runs earlier in the run order are not done: {earlier}")

    planned_jobs = [by_run[j] for j in jobs if by_run[j]["status"] == "planned"]
    job_usd = sum(float(r["usd"][CONS]) for r in planned_jobs)
    bound = job_bound_usd(planned_jobs, gnn_cfg["runtime"])
    ok = metered_usd is not None and float(metered_usd) + bound <= limit
    if metered_usd is None:
        refuse.append("the workspace metered cost is unavailable (modal billing summary failed)")
    elif not ok:
        refuse.append(
            f"budget pre-check: metered ${float(metered_usd):.2f} + job bound ${bound:.2f} "
            f"(conservative ${job_usd:.2f}) > ${limit:.2f} (workspace budget - M6 reserve)"
        )
    return {
        "job": job if isinstance(job, str) else list(jobs),
        "allowed": not refuse,
        "refuse_reason": "; ".join(refuse) or None,
        "spent_usd": spent,
        "projected_usd": projected,
        "cap_usd": cap,
        "cuts": cuts,
        "faithful_epoch_cap": f_cap,
        "budget_check": {
            "metered_usd": None if metered_usd is None else float(metered_usd),
            "job_usd": job_usd,
            "job_bound_usd": bound,
            "limit_usd": limit,
            "ok": ok,
        },
        "actions": actions,
        "warnings": warnings,
        "rows": rows,
    }


def raise_if_refused(result: Mapping[str, Any]) -> None:
    """Raise aml.models.gnn.GnnStopError (nothing is submitted after it) unless the gate
    allowed the job; `detail` carries the gate's numbers (rows left out)."""
    if result["allowed"]:
        return
    from aml.models.gnn import GnnStopError

    detail = {k: v for k, v in result.items() if k != "rows"}
    raise GnnStopError(f"cost gate refused {result['job']}: {result['refuse_reason']}", detail)


# --- measured spend ----------------------------------------------------------------------------


def _gpu_label(name: Any) -> str | None:
    s = str(name or "")
    return next((g for g in GPU_USD_H if g in s), None)


def spent_from_billing(report: Any, apps: Sequence[str] = M3_APPS) -> float:
    """Sum of `modal billing report --json` costs over `apps` (default M3_APPS: the GNN jobs
    and M3's dev apps; aml.tracking.cost_by_app parses the report; an app absent from the
    report counts 0)."""
    from aml.tracking import cost_by_app

    by_app = cost_by_app(report)
    return float(sum(by_app.get(a, 0) for a in apps))


def spent_from_summaries(summaries: Sequence[Mapping[str, Any]]) -> float:
    """The fallback when the billing CLI fails: sum of gpu_seconds x the shape's budget price of
    every summary that records {gpu, gpu_seconds} (cores / memory_mib default to the planning
    shape, 8 cores / 32 GiB). A lower bound: startups and drivers are not in gpu_seconds."""
    total = 0.0
    for s in summaries:
        gpu, sec = _gpu_label(s.get("gpu")), s.get("gpu_seconds")
        if gpu is None or sec is None:
            continue
        cores = float(s.get("cores") or PLANNING["cores"])
        mem = float(s.get("memory_mib") or PLANNING["memory_mib"])
        total += usd(plan_usd_h(gpu, cores, _gib(mem)), float(sec))
    return total


def billed_through(report: Any) -> float | None:
    """The end (unix seconds) of the last interval in an hourly billing report (rows'
    `interval_start`, ISO UTC, + 1 h); None if no row carries one. The report lists complete
    intervals only, so later usage is not in it yet."""
    from datetime import UTC, datetime

    from aml.tracking import billing_rows

    ends = []
    for row in billing_rows(report):
        s = row.get("interval_start")
        if not s:
            continue
        try:
            t = datetime.fromisoformat(str(s).replace("Z", "+00:00"))
        except ValueError:
            continue
        if t.tzinfo is None:
            t = t.replace(tzinfo=UTC)
        ends.append(t.timestamp() + BILLING_INTERVAL_S)
    return max(ends) if ends else None


def lag_usd(calls: Sequence[Mapping[str, Any]], through: float | None) -> float:
    """$ of recorded worker calls / bench containers after the billing report's last interval
    (`through`): per record {gpu, cores?, memory_mib?, elapsed_s, ended_at (unix s)}, the part
    of elapsed_s after `through`, at its shape's budget price. 0 without `through`. A lower
    bound: an attempt that crashed before recording itself is not in it."""
    if through is None:
        return 0.0
    total = 0.0
    for c in calls:
        gpu, ended, el = _gpu_label(c.get("gpu")), c.get("ended_at"), c.get("elapsed_s")
        if gpu is None or ended is None or el is None:
            continue
        after = min(float(el), max(0.0, float(ended) - float(through)))
        if after <= 0:
            continue
        cores = float(c.get("cores") or PLANNING["cores"])
        mem = float(c.get("memory_mib") or PLANNING["memory_mib"])
        total += usd(plan_usd_h(gpu, cores, _gib(mem)), after)
    return total


def measured_spend(
    report: Any,
    summaries: Sequence[Mapping[str, Any]],
    gnn_cfg: Mapping[str, Any],
    *,
    report_error: str | None = None,
    calls: Sequence[Mapping[str, Any]] = (),
) -> dict:
    """The gate's `spent_usd`: max(billing over M3_APPS + lag, the Volume floor) +
    budget.dev_allowance_usd. The Volume floor = the summaries' / call logs' gpu_seconds x
    price (spend_records); lag = lag_usd(calls) after the report's last full interval
    (billed_through), since the hourly report lists complete intervals only. report None (CLI
    failed): the Volume floor alone, with a warning. Returns {usd, billing_usd, lag_usd,
    billed_through, volume_usd, dev_allowance_usd, source, warning}."""
    allowance = float(gnn_cfg["budget"]["dev_allowance_usd"])
    vol = spent_from_summaries(summaries)
    warning, lag, through = None, 0.0, None
    if report is None:
        billing, base, source = None, vol, "volume"
        warning = (
            f"billing report unavailable ({report_error or 'no report'}): using the Volume "
            "summaries' gpu_seconds x shape price, a lower bound"
        )
    else:
        billing = spent_from_billing(report)
        through = billed_through(report)
        lag = lag_usd(calls, through)
        base, source = (billing + lag, "billing") if billing + lag >= vol else (vol, "volume")
    return {
        "usd": base + allowance,
        "billing_usd": billing,
        "lag_usd": lag,
        "billed_through": through,
        "volume_usd": vol,
        "dev_allowance_usd": allowance,
        "source": source,
        "warning": warning,
    }


# --- rendering -----------------------------------------------------------------------------------


def _usd(x: Any) -> str:
    return "-" if x is None else f"${float(x):.2f}"


def render_gate(result: Mapping[str, Any]) -> str:
    """The `--plan-only` table: one line per run with status and $, the cuts, the cap, the
    budget pre-check and the verdict."""
    job = result["job"]
    verdict = "ALLOWED" if result["allowed"] else "REFUSED"
    lines = [
        f"M3 cost gate for {job}: {verdict}",
        f"cap {_usd(result['cap_usd'])} | spent {_usd(result['spent_usd'])} (incl. dev "
        f"allowance) | projected {_usd(result['projected_usd'])} (conservative)",
        "",
        f"{'run':<15} {'status':<8} {'seeds/trials':<14} {'epochs':>7} {'central':>9} "
        f"{'conserv.':>9}  source",
    ]
    for r in result["rows"]:
        units = f"{r['n_trials']} trials" if r.get("n_trials") else str(r.get("seeds") or "-")
        ep = r["epochs"][CONS] if r["run"] != "bench" else "-"
        ep = f"{ep:g}" if isinstance(ep, float) else ep
        started = ""
        if _started(r):
            started = (
                f"  [started: {r.get('done_units') or 0:g} done + "
                f"{r.get('done_epochs') or 0:g} epochs; $ = what is left]"
            )
        lines.append(
            f"{r['run']:<15} {r['status']:<8} {units:<14} {ep:>7} {_usd(r['usd'][CENTRAL]):>9} "
            f"{_usd(r['usd'][CONS]):>9}  {r.get('source', '-')}"
            + (f"  [{r['cut']}]" if r.get("cut") else "")
            + started
        )
    bc = result["budget_check"]
    lines += [
        "",
        f"cuts: {', '.join(result['cuts']) or 'none'}",
        f"faithful epoch cap: {result['faithful_epoch_cap'] or 'none'}",
        f"budget pre-check: metered {_usd(bc['metered_usd'])} + job bound "
        f"{_usd(bc.get('job_bound_usd', bc['job_usd']))} (conservative {_usd(bc['job_usd'])}; "
        f"the wall guard + one chunk) <= {_usd(bc['limit_usd'])}: "
        f"{'ok' if bc['ok'] else 'FAILED'}",
    ]
    lines += [f"action: {a}" for a in result.get("actions", [])]
    lines += [f"warning: {w}" for w in result.get("warnings", [])]
    if result["refuse_reason"]:
        lines.append(f"refused: {result['refuse_reason']}")
    return "\n".join(lines) + "\n"


def _fmt(x: Any, spec: str = ".3f") -> str:
    if x is None:
        return "-"
    if isinstance(x, bool):
        return "yes" if x else "no"
    if isinstance(x, int | float):
        return format(x, spec)
    return str(x)


def render_bench_md(
    cells: Sequence[Mapping[str, Any]], decision: Mapping[str, Any], plan: Sequence[Mapping]
) -> str:
    """reports/gnn_bench.md: the grid table, the decision and the plan."""
    d = decision
    out = [
        "# GNN benchmark (gnn_bench)",
        "",
        "Measured on real HI-Small batches (M3 spec §10). $ per epoch = exact shape price x "
        "(train steps x mean step + val_early pass + 2 s); the plan uses the budget price (rounded "
        "up to the cent) and x budget overhead.",
        "",
        "## Train-step cells",
        "",
        "| cell | step ms (mean / p90) | sampler wait | edges / batch | peak GPU GB | epoch s "
        "| $ / epoch | eligible |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for e in d.get("cells", []):
        p90 = e.get("step_s_p90")
        step = f"{1000 * e['step_s_mean']:.1f} / " + (f"{1000 * p90:.1f}" if p90 else "-")
        peak = e.get("peak_mem_bytes")
        out.append(
            f"| {e['cell_id']} | {step} | {_fmt(e.get('wait_share'), '.1%')} | "
            f"{_fmt(e.get('edges_mean'), ',.0f')} | {_fmt(peak / 1e9 if peak else None, '.2f')} | "
            f"{_fmt(e.get('epoch_s'), '.1f')} | {_fmt(e.get('usd_epoch'), '.4f')} | "
            f"{'yes' if e['eligible'] else 'no: ' + str(e.get('why'))} |"
        )
    fc = _ok(cells, "faithful")
    if fc:
        out += [
            "",
            "## Faithful cells",
            "",
            "| cell | batch (tried) | step s | val s / batch | test s / batch | edges / batch |",
            "| --- | --- | ---: | ---: | ---: | ---: |",
        ]
        for c in fc:
            out.append(
                f"| {c.get('cell_id')} | {c.get('batch_size')} ({c.get('tried')}) | "
                f"{_fmt(c.get('step_s_mean'))} | {_fmt(c.get('val_s_per_batch'))} | "
                f"{_fmt(c.get('test_s_per_batch'))} | {_fmt(c.get('edges_mean'), ',.0f')} |"
            )
    f = d.get("faithful") or {}
    bpe = d.get("bytes_per_edge") or {}
    cache = d.get("eval_cache") or {}
    out += [
        "",
        "## Decision",
        "",
        f"- GPU {d['gpu']}, {d['cores']} cores, memory {d['memory_mib']} MiB (limit "
        f"{d['memory_limit_mib']}), {d['num_workers']} loader workers, batch {d['batch_size']} "
        f"(cell {d.get('cell_id')}), precision {d['precision']}.",
        f"- Epoch {_fmt(d.get('epoch_s'), '.1f')} s, {_fmt(d.get('usd_epoch'), '.4f')} $ per "
        "epoch (exact shape price).",
        f"- Edge caps: train {d['max_edges_per_step']} ({_fmt(bpe.get('train'), ',.0f')} B/edge), "
        f"eval {d['max_edges_per_eval_step']} ({_fmt(bpe.get('eval'), ',.0f')} B/edge); eval "
        f"cache fits: {_fmt(cache.get('fits'))} ({_fmt(cache.get('gb'), '.2f')} GB).",
        f"- Faithful: GPU {f.get('gpu')}, batch {f.get('batch_size')}, {f.get('cores')} cores, "
        f"{f.get('num_workers')} loader workers (its measured shape), "
        f"{_fmt(f.get('usd_epoch'), '.4f')} $ per epoch.",
        f"- Bitwise deterministic on the GPU: {_fmt(d.get('bit_deterministic'))}; deterministic "
        f"algorithms overhead x{_fmt(d.get('det_overhead_ratio'), '.2f')} (train step), "
        f"x{_fmt((d.get('det_overhead_ratios') or {}).get('eval'), '.2f')} (eval forward), "
        f"x{_fmt((d.get('det_overhead_ratios') or {}).get('faithful'), '.2f')} (faithful step).",
        f"- 4-core rerun measured: {_fmt(d.get('rerun_4core'))}.",
    ]
    if d.get("ask_user"):
        out.append(f"- **Stop and ask the user:** {d['ask_user']}")
    out += [
        "",
        "## Plan (conservative = max epochs; x budget overhead)",
        "",
        "| run | protocol | seeds / trials | epochs (central / max) | epoch s (central / cons.) "
        "| containers | central $ | conservative $ | source |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | --- |",
    ]
    for r in plan:
        units = f"{r['n_trials']} trials" if r.get("n_trials") else str(r.get("seeds") or "-")
        ep, es = r["epochs"], r["epoch_s"]
        out.append(
            f"| {r['run']} | {r.get('protocol') or '-'} | {units} | "
            f"{ep[CENTRAL]:g} / {ep[CONS]:g} | {es[CENTRAL]:.1f} / {es[CONS]:.1f} | "
            f"{r['n_containers'][CONS]} | {_usd(r['usd'][CENTRAL])} | {_usd(r['usd'][CONS])} | "
            f"{r.get('source')}{' (cut)' if r.get('status') == 'cut' else ''} |"
        )
    total = {w: sum(float(r["usd"][w]) for r in plan if r.get("status") != "cut") for w in WHICH}
    out += ["", f"Total: central {_usd(total[CENTRAL])}, conservative {_usd(total[CONS])}.", ""]
    return "\n".join(out)
