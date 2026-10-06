"""Training of one protocol's seed set: checkpoints, resume, guards, scoring, set assembly
(M3 spec §7, §9 loop; owner E).

A GPU worker call runs `run_set` once per chunk; the CPU driver (modal_jobs.train_gnn) calls it
again while it returns "partial". Everything is resumable from the Volume: per-epoch atomic
checkpoints with every RNG state, data order and negatives = f(seed, epoch), score files written
once, the set's summary.json written last.

Run dir `/data/models/gnn/<run_key>/` (file names in aml.models.gnn):
  checkpoint.json   fingerprint {run_key, gnn_version, features_digest, data_version, spec_hash,
                    params, protocol, seed, preprocess_hash}; a mismatch empties the dir
  graph_meta.json   {"preprocess", "attr_columns", "bounds", "protocol", "n_nodes", "n_edges"}
  best.pt           state_dict of the best epoch (written first when improved)
  last.pt           {model, optimizer, epoch_done, best_metric, best_epoch, bad_epochs,
                     rng: {torch_cpu, torch_cuda_all, numpy, python}, best_model (the best
                     state_dict again, so a resume can rewrite a best.pt torn by a crash),
                     history (every epoch line so far), fingerprint, preprocess}
  history.jsonl     one line per epoch: {epoch, train_loss, seconds, train_s, val_s, metric,
                    val_early_pr_auc (faithful: val_f1_sampled, val_f1_all, val_sampled_share),
                    steps, edges, oom_splits, guard {train, val_early | val}, peak_gpu_bytes,
                    val_cached, improved, best_epoch, bad_epochs}; rewritten from last.pt on resume
  running.json      {epoch_next, starts}: the unclean-start counter (§7.6)
  FAILED.json       {error, traceback, epoch, run_key, protocol, seed}
  seed_summary.json written when the seed's training ends (below)
  scores_s<k>.parquet [scores_d10_s<k>.parquet]   row_id Int64, split String, score_s<k> Float64
                    (faithful adds sampled Boolean); written once per unit, never overwritten
  scoring.json      {unit: {pass_id, guard, rows, seconds, test_bound, bounds, sampled_share}}:
                    the scoring passes that produced the score files (a pass shared by several
                    seeds carries one pass_id, so set totals count its edges once)

seed_summary.json: {run_key, protocol, seed, params, best_epoch, best_val_ap (faithful:
best_val_f1), metric, epochs_run, max_epochs, epoch_cap, stopped_early, oom_splits, guard: {split:
GUARD totals}, future_share: {split: float} (look-ahead), seconds, gpu (device name),
determinism (set_determinism flags), gnn_version, features_digest, data_version, spec_hash}.

Set dir `/data/models/<set_kind>/<set_key>/`: scores.parquet (row_id Int64, split String,
score_s<k> Float64 per seed of the set's list; val_early and val_late, + test only if final;
faithful adds `sampled`), writer.json, calls.jsonl (one line per worker call: call_id, status,
elapsed_s, device, ended_at), data_version.json, summary.json (LAST). Look-ahead also writes
`gnn_lookahead_d10/<set_key>/` (val rows from the `end` files, test rows from the d10 files) with
its own summary.json. Set summary.json: {model, protocol, set_key, seeds, run_keys, params,
per_seed: {"<seed>": {best_epoch, best_val_ap, epochs_run, stopped_early}}, best_val_ap_mean,
best_val_ap_std, best_val_ap_mean_fresh (seeds != hpo.model_seed; null if none),
features_digest, data_version, spec_hash, gnn_version, report_hash (final sets), guard: {split:
GUARD totals}, edges_checked, violations, target_hits, future_share: {split: float} (look-ahead),
scored_splits, test_bound(s), final, dev, timings, gpu, gpu_seconds, determinism}; faithful adds
{epochs_run, max_epochs, epoch_cap, best_epoch, best_val_f1, batch_size, sampled_share: {split:
float}, exemptions, recalled}.

torch is imported inside functions only: the laptop entrypoints import `effective_params`.
"""

from __future__ import annotations

import contextlib
import json
import math
import os
import random
import shutil
import time
import traceback
import uuid
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from aml.models.gnn import (
    BEST_CKPT,
    FAILED_FILE,
    FAITHFUL_EXEMPT_FEATURES,
    FAITHFUL_EXEMPT_NORM,
    FAITHFUL_MODEL,
    FINGERPRINT_FILE,
    GNN_VERSION,
    GRAPH_META_FILE,
    HISTORY_FILE,
    LAST_CKPT,
    LOOKAHEAD_D10_MODEL,
    PROTOCOLS,
    RUNNING_FILE,
    SCORES_FILE,
    SEED_SUMMARY_FILE,
    STOPPED_FILE,
    SUMMARY_FILE,
    TEST_BOUNDS,
    WRITER_FILE,
    add_guard,
    append_jsonl,
    empty_guard,
    label_splits_for,
    read_jsonl,
    report_hash,
    score_column,
    scores_file,
    set_kind,
)

if TYPE_CHECKING:
    from torch import Tensor, nn

    from aml.models.gnn.graph import HostGraph
    from aml.models.gnn.transforms import FlatBatch
    from aml.paths import DataPaths

LEASE_FRESH_S = 15 * 60  # a writer lease with an older heartbeat is stale
MAX_STARTS = 3  # the third unclean start at the same epoch writes FAILED.json
UNIT_MARGIN = 1.25  # stop when elapsed + 1.25 x the next unit's estimate > the attempt wall
EXPERIMENT = "aml-train-gnn"  # MLflow experiment of the training runs (= the Modal app name)
# set / HPO dir: one line per worker call {call_id, status, elapsed_s, device, ended_at (unix s)}:
# the gpu_seconds fallback and the cost gate's spend floor / billing lag (train_gnn.spend_records)
CALLS_FILE = "calls.jsonl"
SCORING_FILE = "scoring.json"  # run dir: the scoring passes behind its score files
VAL_SPLITS = ("val_early", "val_late")
EPOCH_SALT = 0x7E5  # torch is re-seeded per (seed, epoch): resume == an uninterrupted run
HPO_PARAMS = ("lr", "final_dropout", "w_pos")
# The faithful run's settings that are recalled (R) from Multi-GNN's code or are our choices
# beside recalled ones (M3 spec §9), listed in its summary and the README: nothing here is
# stated by the papers (the (V) parts are marked).
RECALLED = (
    "lr 0.006213",
    "w_pos 6.275",
    "layer_dropout 0.0098 (value R; applied to the node states after each layer's residual: "
    "placement ours, Multi-GNN's GINe forward may not apply it)",
    "final_dropout 0.1053",
    "readout 3H->50->25->2",
    "batch_size 8192 (Multi-GNN CLI default; Egressy et al. App. C says reverse-MP runs needed "
    "a reduced batch, value not stated)",
    "max_epochs 100",
    "hidden 64 / 2 layers (ours, the headline model's; App. F.5's 64 / 2 is the runtime-table "
    "configuration; Multi-GNN's tuned GIN hidden 66, R)",
    "edge features (timestamp, amount_received, receiving_currency, payment_format, ports)",
    "receiving_currency / payment_format codes: sorted train-vocab order (-1 unknown), fed as "
    "z-scored numbers (ours); Multi-GNN recalled: first-appearance order in the raw CSV with "
    "one dict shared by both currency columns (R, unverified)",
    "per-snapshot z-normalisation",
    "reverse edges = flip of all edges incl. self-loops",
    "ego IDs on every seed endpoint (AddEgoIds)",
    "loss, val selection and test F1 on sampled targets only",
    "no early stopping; the best-val epoch",
    "Adam, constant lr, no weight decay",
    "BatchNorm after each conv",
    "residual (h + relu(bn(agg))) / 2",
    "mean over edge types (to_hetero aggr='mean')",
    "per-direction edge encoders and edge-update MLPs",
    "whole-day split d1-6 / d7-8 / d9-18 (App. E.1 states 60/20/20 by two timestamps, V; "
    "actual 64.0 / 19.0 / 17.0%)",
    "fanout [100, 100] per edge type, uniform, non-disjoint (App. F.4 states 100 one- and "
    "two-hop neighbours, V)",
)


class GnnOOMError(RuntimeError):
    """A CUDA out-of-memory that survived the in-process split (a single subgraph, or a faithful
    batch, still does not fit). Deterministic: run_set fails fast; an HPO trial is logged FAIL."""

    def __init__(self, message: str, edges: int = 0, subgraphs: int = 0) -> None:
        super().__init__(message)
        self.edges = int(edges)
        self.subgraphs = int(subgraphs)

    def __reduce__(self):  # survives pickling (DataLoader workers, Modal results)
        return (type(self), (str(self), self.edges, self.subgraphs))


# --- small public helpers ------------------------------------------------------------------------


def effective_params(gnn_cfg: dict, protocol: str, hpo_best: dict | None) -> dict:
    """The hyperparameters a run trains with (hashed into its run key):
    {conv, hidden, layers, layer_dropout, final_dropout, lr, w_pos, neg_rate} (+ towers for PNA).
    causal / lookahead: the train section with HPO's best {lr, final_dropout, w_pos} applied
    (hpo_best None -> the train section's trial-0 values, as the dev run uses);
    pna: its protocol values (conv "pna", hidden, towers, lr, dropouts, w_pos), neg_rate from
    train; faithful: its protocol values with neg_rate 1.0 (no down-sampling).
    hidden / layers default to gnn_cfg["model"]. hpo_best is ignored for pna and faithful."""
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    m, t = gnn_cfg["model"], gnn_cfg["train"]
    out: dict[str, Any] = {
        "conv": str(m["conv"]),
        "hidden": int(m["hidden"]),
        "layers": int(m["layers"]),
        "layer_dropout": float(m["layer_dropout"]),
        "final_dropout": float(t["final_dropout"]),
        "lr": float(t["lr"]),
        "w_pos": float(t["w_pos"]),
        "neg_rate": float(t["neg_rate"]),
    }
    if protocol in ("causal", "lookahead"):
        if hpo_best is not None:
            if set(hpo_best) != set(HPO_PARAMS):
                raise ValueError(f"HPO best params must be exactly {HPO_PARAMS}, got {hpo_best}")
            out.update({k: float(hpo_best[k]) for k in HPO_PARAMS})
    elif protocol == "pna":
        p = gnn_cfg["protocols"]["pna"]
        out.update(
            conv="pna",
            hidden=int(p["hidden"]),
            towers=int(p["towers"]),
            lr=float(p["lr"]),
            layer_dropout=float(p["layer_dropout"]),
            final_dropout=float(p["final_dropout"]),
            w_pos=float(p["w_pos"]),
        )
    else:  # faithful
        f = gnn_cfg["protocols"]["faithful"]
        out.update(
            lr=float(f["lr"]),
            w_pos=float(f["w_pos"]),
            layer_dropout=float(f["layer_dropout"]),
            final_dropout=float(f["final_dropout"]),
            neg_rate=1.0,
        )
    return out


def class_weights(neg_rate: float, w_pos: float) -> tuple[float, float]:
    """(1.0, w_pos * neg_rate): Multi-GNN's full-data CE weights [1, w_pos] in expectation when
    negatives are kept at rate neg_rate (M3 spec §7.2)."""
    r, w = float(neg_rate), float(w_pos)
    if not (0.0 < r <= 1.0) or not (w > 0.0 and math.isfinite(w)):
        raise ValueError(f"need 0 < neg_rate <= 1 and w_pos > 0, got {neg_rate}, {w_pos}")
    return (1.0, w * r)


def weighted_loss(logits: Tensor, y: Tensor, cw: Tensor, total_weight: float | Tensor) -> Tensor:
    """F.cross_entropy(logits, y, weight=cw, reduction="sum") / total_weight, where total_weight
    = sum of cw[y_i] over the WHOLE batch (all parts of a split batch): one optimizer step per
    batch equals PyTorch's weighted mean on the unsplit batch."""
    import torch.nn.functional as F

    return F.cross_entropy(logits, y, weight=cw, reduction="sum") / total_weight


def set_determinism(seed: int, *, deterministic: bool, matmul_precision: str, device: str) -> dict:
    """Seed random / numpy / torch (+ cuda); use_deterministic_algorithms(deterministic,
    warn_only=True); cudnn.benchmark False; float32 matmul precision `matmul_precision` on an
    L4-class GPU (TF32, compute capability >= 8), "highest" on T4 and CPU. Returns {torch,
    torch_geometric, pyg_lib, cuda, gpu, compute_capability, device, deterministic,
    matmul_precision, tf32, cublas_workspace_config} for MLflow and the summaries."""
    import torch

    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed % 2**32)  # noqa: NPY002 - the legacy global RNG is seeded too
    torch.manual_seed(seed)
    cuda = device == "cuda"
    if cuda:
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(bool(deterministic), warn_only=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = bool(deterministic)
    gpu, capability = None, None
    if cuda:
        gpu = torch.cuda.get_device_name(0)
        capability = list(torch.cuda.get_device_capability(0))
    precision = matmul_precision if cuda and capability and capability[0] >= 8 else "highest"
    torch.set_float32_matmul_precision(precision)
    return {
        "torch": torch.__version__,
        "torch_geometric": _version("torch_geometric"),
        "pyg_lib": _version("pyg_lib"),
        "cuda": torch.version.cuda,
        "gpu": gpu,
        "compute_capability": capability,
        "device": device,
        "deterministic": bool(deterministic),
        "matmul_precision": precision,
        "tf32": precision != "highest",
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }


def is_transient(exc: BaseException) -> bool:
    """True only for the DataLoader timeout (RuntimeError starting "DataLoader timed out"),
    OSError, ConnectionError and TimeoutError: these are re-raised so Modal retries. Everything
    else (LeakError, ValueError, AssertionError, KeyError, an OOM at the minimal split, ...) is
    deterministic: FAILED.json + {"status": "failed"}."""
    if isinstance(exc, OSError | ConnectionError | TimeoutError):
        return True
    return isinstance(exc, RuntimeError) and str(exc).startswith("DataLoader timed out")


def save_torch_atomic(obj: Any, path: Path) -> Path:
    """torch.save to a temp file in the same directory, fsync, os.replace (no .tmp survives)."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with open(tmp, "wb") as f:
            torch.save(obj, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)
    return path


def load_torch(path: Path, device: str = "cpu") -> Any:
    """torch.load with weights_only=True (checkpoints hold tensors and plain values only)."""
    import torch

    return torch.load(Path(path), map_location=device, weights_only=True)


# --- early stopping (M3 spec §7.4) --------------------------------------------------------------


def early_stop_update(
    best_metric: float, best_epoch: int | None, bad_epochs: int, metric: float | None, epoch: int
) -> tuple[bool, float, int, int]:
    """One epoch of the early-stopping bookkeeping: (improved, best_metric, best_epoch,
    bad_epochs). Best = max metric, ties -> the earlier epoch; a NaN metric (no positives)
    counts as -inf, and the first epoch is always the best so far."""
    m = float(metric) if metric is not None and math.isfinite(float(metric)) else -math.inf
    if best_epoch is None or m > best_metric:
        return True, m, int(epoch), 0
    return False, best_metric, best_epoch, bad_epochs + 1


def should_stop(
    epoch_done: int,
    bad_epochs: int,
    *,
    max_epochs: int,
    min_epochs: int,
    patience: int,
    early_stopping: bool = True,
) -> bool:
    """After epoch `epoch_done` (0-based): stop at max_epochs, or (early stopping) when
    epoch_done + 1 >= min_epochs and bad_epochs >= patience."""
    n = epoch_done + 1
    if n >= max_epochs:
        return True
    return bool(early_stopping) and n >= min_epochs and bad_epochs >= patience


# --- the wall budget and the writer lease (M3 spec §7.6) ----------------------------------------


class Clock:
    """One worker call's wall budget: before each unit (epoch, trial, scoring pass), stop if
    elapsed + UNIT_MARGIN x the unit's estimate > budget_s. A call always runs at least one unit
    (otherwise a unit longer than the budget would loop forever on "partial"); the per-attempt
    Modal timeout and the unclean-start counter bound that unit."""

    def __init__(self, budget_s: float, now: Callable[[], float] = time.monotonic) -> None:
        self.budget_s = float(budget_s)
        self._now = now
        self.t0 = now()
        self.units_done = 0
        self.last_epoch_s: float | None = None

    def elapsed(self) -> float:
        return self._now() - self.t0

    def out_of_time(self, estimate_s: float | None) -> bool:
        if self.units_done == 0:
            return False
        est = max(0.0, float(estimate_s or 0.0))
        return self.elapsed() + UNIT_MARGIN * est > self.budget_s

    def unit_done(self) -> None:
        self.units_done += 1


def current_call_id() -> str:
    """The Modal function call id (constant across retries of one input), else a process id."""
    try:
        import modal

        cid = modal.current_function_call_id()
    except Exception:  # noqa: BLE001 - not inside Modal, or an SDK without it
        cid = None
    return str(cid) if cid else f"local-{os.getpid()}-{_PROCESS_TOKEN}"


_PROCESS_TOKEN = uuid.uuid4().hex[:8]


class WriterLease:
    """`writer.json` in a set (or HPO) dir: {call_id, heartbeat}. A fresh lease (heartbeat <
    LEASE_FRESH_S old) held by another call means another writer is active -> "busy". Released on
    every clean return, so the driver's next chunk (a new call) can take it; a crashed call's
    retry has the same call id and takes it back."""

    def __init__(
        self,
        stage_dir: Path,
        *,
        call_id: str | None = None,
        now: Callable[[], float] = time.time,
    ) -> None:
        self.path = Path(stage_dir) / WRITER_FILE
        self.call_id = call_id or current_call_id()
        self._now = now

    def holder(self) -> dict | None:
        try:
            return json.loads(self.path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            return None

    def acquire(self) -> bool:
        doc = self.holder()
        if doc and doc.get("call_id") != self.call_id:
            age = self._now() - float(doc.get("heartbeat") or 0.0)
            if age < LEASE_FRESH_S:
                return False
        self.refresh()
        return True

    def refresh(self) -> None:
        from aml.io import write_json_atomic

        write_json_atomic({"call_id": self.call_id, "heartbeat": self._now()}, self.path)

    def release(self) -> None:
        doc = self.holder()
        if doc and doc.get("call_id") == self.call_id:
            self.path.unlink(missing_ok=True)


# --- one training step and the eval forward (M3 spec §5.7, §7.2) ---------------------------------


def _edges_of(fb: FlatBatch) -> int:
    return int(fb.gid_fwd.numel()) + int(fb.gid_rev.numel())


def _cuda_oom():
    import torch

    return torch.cuda.OutOfMemoryError


def _free_cuda(device: str) -> None:
    if device == "cuda":
        import torch

        torch.cuda.empty_cache()


def batch_labels_nosync(fb: FlatBatch, y: Tensor, y_host: np.ndarray | None) -> Tensor:
    """Y[tgt_gid] (int64, on y's device) as transforms.batch_labels, but the "label not
    loaded" (-1) check runs on the host copy `y_host` (HostGraph.y), so a training step never
    waits for the GPU. Without y_host: transforms.batch_labels (checks on the device)."""
    from aml.models.gnn import transforms as T

    if y_host is None:
        return T.batch_labels(fb, y)
    bad = int((y_host[fb.tgt_gid.numpy()] < 0).sum())
    if bad:
        raise ValueError(f"{bad} batch targets have no loaded label")
    return y.index_select(0, fb.tgt_gid.to(y.device, non_blocking=True)).long()


def _part_labels(fb: FlatBatch, parts: list, labels: Tensor) -> list[Tensor]:
    """The labels of each part: split_flat_batch keeps the batch's seeds in order, in
    consecutive groups, so the parts' labels are consecutive slices of the batch's."""
    if len(parts) == 1:
        return [labels]
    import torch

    sizes = [int(p.tgt_gid.numel()) for p in parts]
    if not torch.equal(torch.cat([p.tgt_gid for p in parts]), fb.tgt_gid):
        raise AssertionError("split_flat_batch reordered the batch's seeds")
    return list(torch.split(labels, sizes))


def _forward_backward(model, parts, labels, ea, cw, total, device) -> Tensor:
    from aml.models.gnn import transforms as T

    acc = None
    for p, lab in zip(parts, labels, strict=True):
        inputs = T.to_model_inputs(p, ea, device)
        loss = weighted_loss(model(*inputs), lab, cw, total)
        loss.backward()
        acc = loss.detach() if acc is None else acc + loss.detach()
    return acc


def train_batch(
    model: nn.Module,
    opt: Any,
    fb: FlatBatch,
    *,
    ea: Tensor,
    y: Tensor,
    cw: Tensor,
    cap: int | None,
    device: str,
    y_host: np.ndarray | None = None,
) -> tuple[Tensor, int]:
    """One optimizer step on one sampled batch (M3 spec §7.2, §5.7): split under the edge cap
    `cap` (None: no split), gradients accumulated over the parts with every part's loss
    normalised by the WHOLE batch's class weight, one Adam step. A CUDA OOM empties the cache,
    halves the cap for this batch and redoes it; an OOM with one subgraph per part raises
    GnnOOMError. y_host (HostGraph.y): the label check on the host, so the step does not sync
    the GPU. Returns (the batch loss as a detached 0-dim device tensor, OOM splits)."""
    from aml.models.gnn import transforms as T

    oom_t = _cuda_oom()
    labels = batch_labels_nosync(fb, y, y_host)
    total = cw[labels].sum()
    n_sub = int(fb.tgt_gid.numel())
    edges = _edges_of(fb)
    cap_now, ooms = cap, 0
    while True:
        parts = [fb] if cap_now is None or edges <= cap_now else T.split_flat_batch(fb, cap_now)
        opt.zero_grad(set_to_none=True)
        loss = None
        # An OOM is cleaned up below, outside the handler (it pins the failed step's tensors).
        with contextlib.suppress(oom_t):
            loss = _forward_backward(
                model, parts, _part_labels(fb, parts, labels), ea, cw, total, device
            )
        if loss is not None:
            opt.step()
            return loss, ooms
        opt.zero_grad(set_to_none=True)
        _free_cuda(device)
        if len(parts) >= n_sub:
            raise GnnOOMError(
                f"CUDA OOM on a training batch of {edges} edges even with one subgraph per part",
                edges,
                n_sub,
            )
        cap_now = max(1, min(cap_now if cap_now is not None else edges, edges) // 2)
        ooms += 1


def faithful_train_batch(
    model: nn.Module,
    opt: Any,
    fb: FlatBatch,
    *,
    ea: Tensor,
    y: Tensor,
    cw: Tensor,
    device: str,
    y_host: np.ndarray | None = None,
) -> Tensor | None:
    """One faithful step (M3 spec §9): CE [1, w_pos] on the SAMPLED targets only (Multi-GNN's
    mask), weight-normalised over them; a batch without sampled targets is skipped (None).
    Faithful batches are not splittable: a CUDA OOM raises GnnOOMError."""
    import torch

    from aml.models.gnn import transforms as T

    if fb.sampled is None:
        return None
    idx_cpu = torch.nonzero(fb.sampled, as_tuple=False).reshape(-1)  # host: no GPU sync
    if not idx_cpu.numel():
        return None
    oom = False
    opt.zero_grad(set_to_none=True)
    try:
        inputs = T.to_model_inputs(fb, ea, device)
        labels = batch_labels_nosync(fb, y, y_host)
        idx = idx_cpu.to(labels.device, non_blocking=True)
        lab = labels.index_select(0, idx)
        logits = model(*inputs).index_select(0, idx)
        loss = weighted_loss(logits, lab, cw, cw[lab].sum())
        loss.backward()
    except _cuda_oom():
        oom = True
    if oom:
        opt.zero_grad(set_to_none=True)
        _free_cuda(device)
        edges = _edges_of(fb)
        raise GnnOOMError(f"CUDA OOM on a faithful batch of {edges} edges", edges, 1)
    opt.step()
    return loss.detach()


def eval_parts(
    models: Mapping[int, nn.Module],
    part: FlatBatch,
    *,
    ea: Tensor,
    device: str,
    cap: int | None,
) -> list[tuple[np.ndarray, dict[int, np.ndarray]]]:
    """Scores of one eval part for every model (eval mode, the caller holds inference_mode):
    [(seed positions, {model key: float64 scores})]. A CUDA OOM splits the part further (exact
    in eval mode); a single subgraph (or a faithful batch) that still OOMs raises GnnOOMError."""
    from aml.models.gnn import model as M
    from aml.models.gnn import transforms as T

    out = None
    try:
        inputs = T.to_model_inputs(part, ea, device)
        out = {k: M.scores_from_logits(m(*inputs)) for k, m in models.items()}
    except _cuda_oom():
        out = None
    if out is not None:
        return [(part.seed_pos.cpu().numpy().astype(np.int64), out)]
    _free_cuda(device)
    edges, n_sub = _edges_of(part), int(part.tgt_gid.numel())
    if part.node_sub is None or n_sub <= 1:
        raise GnnOOMError(f"CUDA OOM on an eval part of {edges} edges ({n_sub} targets)", edges, 1)
    sub_cap = max(1, min(cap if cap is not None else edges, edges) // 2)
    subs = T.split_flat_batch(part, sub_cap)
    if len(subs) <= 1:
        raise GnnOOMError(f"CUDA OOM on an eval part of {edges} edges", edges, n_sub)
    res: list[tuple[np.ndarray, dict[int, np.ndarray]]] = []
    for s in subs:
        res += eval_parts(models, s, ea=ea, device=device, cap=sub_cap)
    return res


def _cap_parts(fb: FlatBatch, cap: int | None) -> list:
    from aml.models.gnn import transforms as T

    if cap is None or fb.node_sub is None or _edges_of(fb) <= cap:
        return [fb]
    return T.split_flat_batch(fb, cap)


def _guard_of(fb: FlatBatch) -> list[int]:
    return [int(v) for v in fb.guard.tolist()]


def future_share(guard: Mapping[str, int]) -> float | None:
    """Sampled edges at or after the target's minute / all sampled edges (look-ahead)."""
    n = int(guard.get("edges_checked", 0))
    return float(guard["future_edges"]) / n if n else None


# --- engines: everything built once per container ------------------------------------------------


@dataclass
class ScoreUnit:
    """One scoring pass: `splits` scored with test bound `test_bound` into scores_file(k, bound)."""

    name: str
    splits: tuple[str, ...]
    test_bound: str = "end"


@dataclass
class ScorePass:
    unit: ScoreUnit
    gids: np.ndarray
    split: np.ndarray  # split name per row
    scores: dict[int, np.ndarray]
    sampled: np.ndarray | None
    guard: dict[str, int]
    seconds: float
    pass_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])


def _split_names(g: HostGraph, gids: np.ndarray) -> np.ndarray:
    from aml.data.split import SPLITS

    return np.asarray(SPLITS, dtype=object)[g.split_code[gids]]


def _require_train_seeds(g: HostGraph, gids: np.ndarray) -> None:
    """Training targets (and the negatives drawn from them) are train-split rows only: g.y also
    holds val_early labels (early stopping, HPO selection), which must never be trained on."""
    from aml.data.split import SPLITS

    gids = np.asarray(gids, dtype=np.int64)
    if not len(gids) or not (g.split_code[gids] == SPLITS.index("train")).all():
        from aml.models.gnn import LeakError

        bad = int((g.split_code[gids] != SPLITS.index("train")).sum()) if len(gids) else 0
        raise LeakError(
            f"{bad} train seeds lie outside the train split (or there are none)",
            {"n_seeds": int(len(gids)), "outside_train": bad},
        )


# The snapshot each split's seeds may see (faithful.snapshot_last_ranks keys).
FAITHFUL_SNAPSHOT_OF = {"train": "train", "val_early": "val", "val_late": "val", "test": "test"}


class TemporalEngine:
    """Graph, device tensors, loaders and the val_early eval cache of a causal / look-ahead /
    PNA set or the HPO (protocol "causal"), built once per container (M3 spec §7.1).

    allowed_splits: the splits `score` may build loaders for (HPO: none; a non-final set:
    val_early, val_late; a final set: + test). The val_early loader of `validate` always exists.
    """

    metric_name = "val_early_pr_auc"
    val_guard_key = "val_early"
    early_stopping = True

    def __init__(
        self,
        g: HostGraph,
        gnn_cfg: dict,
        *,
        protocol: str,
        params: dict,
        device: str,
        runtime: dict,
        allowed_splits: tuple[str, ...] = VAL_SPLITS,
        log: Callable[[str], None] | None = None,
    ) -> None:
        import torch

        from aml.models.gnn import graph as G
        from aml.models.gnn import sampler as S
        from aml.models.gnn import transforms as T

        if protocol not in ("causal", "lookahead", "pna"):
            raise ValueError(f"TemporalEngine serves causal / lookahead / pna, got {protocol!r}")
        self.g, self.gnn_cfg, self.protocol = g, gnn_cfg, protocol
        self.device, self.runtime = device, dict(runtime)
        self.allowed_splits = tuple(allowed_splits)
        self.log = log or _print
        self.sampler_cfg = dict(gnn_cfg["sampler"])
        self.preprocess = g.preprocess
        if device == "cuda":
            torch.set_num_threads(S.MAIN_THREADS)
        t0 = time.perf_counter()
        self.hetero = G.build_hetero(g)
        self.ea = torch.from_numpy(np.ascontiguousarray(g.ea, dtype=np.float32)).to(device)
        self.y = torch.from_numpy(np.ascontiguousarray(g.y, dtype=np.int8)).to(device)
        self.train_gids = G.split_gids(g, "train")
        _require_train_seeds(g, self.train_gids)
        y_train = G.labels_for(g, self.train_gids)
        self.pos_idx = np.flatnonzero(y_train == 1).astype(np.int64)
        self.neg_idx = np.flatnonzero(y_train == 0).astype(np.int64)
        if len(self.pos_idx) == 0:
            raise ValueError("the train split has no positives")
        self.val_gids = G.split_gids(g, "val_early")
        self.val_y = G.labels_for(g, self.val_gids).astype(np.int8)
        self.val_loader = S.make_loader(
            g,
            self.hetero,
            self.val_gids,
            protocol=protocol,
            role="eval",
            sampler_cfg=self.sampler_cfg,
            runtime=self.runtime,
            device=device,
        )
        self.cache = T.EvalCache(float(self.sampler_cfg["eval_cache_max_gb"]))
        self.train_cap, self.eval_cap = T.resolve_edge_caps(
            self.sampler_cfg, device, memory_fraction=float(gnn_cfg["bench"]["memory_fraction"])
        )
        self._train: tuple[int, float, Any, Any] | None = None  # (seed, rate, loader, sampler)
        self.deg = None
        self.set_params(params)
        self.build_s = time.perf_counter() - t0

    def set_params(self, params: dict) -> None:
        """The run's hyperparameters (HPO: per trial): class weights, model shape, PNA degrees."""
        import torch

        from aml.models.gnn import graph as G

        self.params = dict(params)
        self.cw = torch.tensor(
            class_weights(self.params["neg_rate"], self.params["w_pos"]),
            dtype=torch.float32,
            device=self.device,
        )
        if self.params["conv"] == "pna" and self.deg is None:
            fwd, rev = G.train_degree_histograms(self.g)
            self.deg = (torch.from_numpy(np.asarray(fwd)), torch.from_numpy(np.asarray(rev)))

    # -- model / data ---------------------------------------------------------------------------

    def new_model(self) -> nn.Module:
        from aml.models.gnn import model as M

        return M.build_model(
            self.gnn_cfg, self.protocol, self.preprocess, self.params, deg=self.deg
        ).to(self.device)

    def train_loader(self, seed: int):
        """(loader, sampler) of `seed`: the negatives are f(seed, epoch); built once per seed
        (HPO trials share the model seed, hence one loader)."""
        from aml.models.gnn import sampler as S

        rate = float(self.params["neg_rate"])
        if self._train is None or self._train[:2] != (int(seed), rate):
            self.close()  # stop the previous loader's workers first
            sampler = S.EpochSubsetSampler(self.pos_idx, self.neg_idx, rate, int(seed))
            loader = S.make_loader(
                self.g,
                self.hetero,
                self.train_gids,
                protocol=self.protocol,
                role="train",
                sampler_cfg=self.sampler_cfg,
                runtime=self.runtime,
                device=self.device,
                sampler=sampler,
            )
            self._train = (int(seed), rate, loader, sampler)
        return self._train[2], self._train[3]

    def close(self) -> None:
        """Stop the train loader's persistent workers (sampler.close_loader) and drop it."""
        from aml.models.gnn import sampler as S

        if self._train is not None:
            S.close_loader(self._train[2])
        self._train = None

    # -- one epoch --------------------------------------------------------------------------

    def train_epoch(self, model: nn.Module, opt: Any, seed: int, epoch: int) -> dict:
        import torch

        loader, sampler = self.train_loader(seed)
        sampler.set_epoch(int(epoch))
        it = iter(loader)
        seed_epoch(seed, epoch, self.device)  # after iter(): see seed_epoch
        model.train()
        guard = empty_guard()
        loss_acc = torch.zeros((), device=self.device)
        steps = edges = ooms = 0
        for fb in it:
            guard = add_guard(guard, _guard_of(fb))
            loss, n_oom = train_batch(
                model,
                opt,
                fb,
                ea=self.ea,
                y=self.y,
                cw=self.cw,
                cap=self.train_cap,
                device=self.device,
                y_host=self.g.y,
            )
            loss_acc += loss
            steps += 1
            edges += _edges_of(fb)
            ooms += n_oom
        if steps == 0:
            raise ValueError("the train loader yielded no batch")
        return {
            "train_loss": float(loss_acc.item()) / steps,
            "steps": steps,
            "edges": edges,
            "oom_splits": ooms,
            "guard": guard,
        }

    def validate(self, model: nn.Module) -> dict:
        import torch

        from aml.eval.metrics import pr_auc

        model.eval()
        n = len(self.val_gids)
        scores = np.full(n, np.nan)
        seen = np.zeros(n, dtype=np.int64)
        guard = empty_guard()
        was_cached = bool(self.cache.cached)
        with torch.inference_mode():
            for part in self.cache.batches(self.val_loader, self.eval_cap):
                guard = add_guard(guard, _guard_of(part))
                for pos, out in eval_parts(
                    {0: model}, part, ea=self.ea, device=self.device, cap=self.eval_cap
                ):
                    scores[pos] = out[0]
                    seen[pos] += 1
        _check_coverage(seen, "val_early")
        ap = pr_auc(self.val_y, scores)
        return {
            "metric": ap,
            "val_early_pr_auc": ap,
            "guard": guard,
            "val_cached": was_cached,
            "cache": self.cache.stats(),
        }

    # -- scoring --------------------------------------------------------------------------------

    def score_units(self, final: bool, test_bounds: tuple[str, ...]) -> list[ScoreUnit]:
        units = [ScoreUnit("val_early", ("val_early",)), ScoreUnit("val_late", ("val_late",))]
        if final:
            units += [
                ScoreUnit("test" if b == "end" else "test_d10", ("test",), b) for b in test_bounds
            ]
        return units

    def score(self, models: Mapping[int, nn.Module], unit: ScoreUnit) -> ScorePass:
        import torch

        from aml.models.gnn import graph as G
        from aml.models.gnn import sampler as S

        (split,) = unit.splits
        if split not in self.allowed_splits:
            raise AssertionError(f"a {split} loader is not allowed here ({self.allowed_splits})")
        t0 = time.perf_counter()
        gids = G.split_gids(self.g, split)
        loader = S.make_loader(
            self.g,
            self.hetero,
            gids,
            protocol=self.protocol,
            role="eval",
            sampler_cfg=self.sampler_cfg,
            runtime=self.runtime,
            device=self.device,
            test_bound=unit.test_bound,
        )
        n = len(gids)
        out = {k: np.full(n, np.nan) for k in models}
        seen = np.zeros(n, dtype=np.int64)
        guard = empty_guard()
        for m in models.values():
            m.eval()
        with torch.inference_mode():
            for fb in loader:
                guard = add_guard(guard, _guard_of(fb))
                for part in _cap_parts(fb, self.eval_cap):
                    for pos, sc in eval_parts(
                        models, part, ea=self.ea, device=self.device, cap=self.eval_cap
                    ):
                        seen[pos] += 1
                        for k in models:
                            out[k][pos] = sc[k]
        del loader
        _check_coverage(seen, unit.name)
        return ScorePass(
            unit,
            gids,
            np.full(n, split, dtype=object),
            out,
            None,
            guard,
            time.perf_counter() - t0,
        )


class FaithfulEngine:
    """Multi-GNN's published setup (M3 spec §9): snapshot graphs (train / val / test), EA_f
    z-scored per snapshot, non-temporal non-disjoint uniform [100, 100] loaders, CE on sampled
    targets, best val F1 at argmax over sampled val targets (days 7-8), no early stopping.

    The val labels and loader are built on first use (the bench times faithful batches on a
    graph loaded with the causal label splits)."""

    metric_name = "val_f1_sampled"
    val_guard_key = "val"
    early_stopping = False

    def __init__(
        self,
        g: HostGraph,
        gnn_cfg: dict,
        *,
        paths: DataPaths,
        features_dir: Path,
        params: dict,
        device: str,
        runtime: dict,
        allow_test: bool,
        log: Callable[[str], None] | None = None,
    ) -> None:
        import torch

        from aml.models.gnn import faithful as Fm
        from aml.models.gnn import graph as G
        from aml.models.gnn import sampler as S

        self.g, self.gnn_cfg, self.protocol = g, gnn_cfg, "faithful"
        self.device, self.runtime, self.allow_test = device, dict(runtime), bool(allow_test)
        self.log = log or _print
        self.fcfg = dict(gnn_cfg["protocols"]["faithful"])
        self.preprocess = Fm.faithful_preprocess()
        if device == "cuda":
            torch.set_num_threads(S.MAIN_THREADS)
        t0 = time.perf_counter()
        self.raw = Fm.load_faithful_raw(g, paths, features_dir, protocol="faithful")
        self.lasts = Fm.snapshot_last_ranks(g)
        self.snap: dict[str, Any] = {}
        self.ea: dict[str, Any] = {}
        for name in ("train", "val"):
            self.build_snapshot(name)
        self.y = torch.from_numpy(np.ascontiguousarray(g.y, dtype=np.int8)).to(device)
        self.train_gids = G.split_gids(g, "train")
        _require_train_seeds(g, self.train_gids)
        self.val_gids = np.concatenate([G.split_gids(g, s) for s in VAL_SPLITS])
        self.val_split = _split_names(g, self.val_gids)
        self._val_y: np.ndarray | None = None
        self._val_loader = None
        self._train: tuple[int, Any] | None = None
        self.set_params(params)
        self.build_s = time.perf_counter() - t0

    def set_params(self, params: dict) -> None:
        import torch

        self.params = dict(params)
        self.cw = torch.tensor(
            class_weights(1.0, self.params["w_pos"]), dtype=torch.float32, device=self.device
        )

    def build_snapshot(self, name: str) -> None:
        """The snapshot's HeteroData and its device EA_f (z-scored over the snapshot)."""
        import torch

        from aml.models.gnn import faithful as Fm

        if name in self.snap:
            return
        last = int(self.lasts[name])
        self.snap[name] = Fm.snapshot_hetero(self.g, last)
        ea = Fm.snapshot_ea(self.raw, last, protocol="faithful")
        self.ea[name] = torch.from_numpy(np.ascontiguousarray(ea, dtype=np.float32)).to(self.device)

    def last_rank_for(self, gids: np.ndarray) -> int:
        """The last rank the seeds may see, from their split alone (the latest split among
        them: train -> train_last, val_early / val_late -> val_last, test -> data_last)."""
        from aml.data.split import SPLITS

        codes = self.g.split_code[np.asarray(gids, dtype=np.int64)]
        if not len(codes):
            raise ValueError("no seeds")
        return int(self.lasts[FAITHFUL_SNAPSHOT_OF[SPLITS[int(codes.max())]]])

    def loader(
        self,
        snapshot: str,
        gids: np.ndarray,
        *,
        shuffle: bool,
        seed: int,
        batch_size: int | None = None,
    ):
        """A faithful loader of `gids` over snapshot `snapshot`; its guard bound comes from the
        seeds' split (last_rank_for), so a loader over another split's snapshot raises."""
        from aml.models.gnn import sampler as S

        self.build_snapshot(snapshot)
        return S.make_faithful_loader(
            self.snap[snapshot],
            gids,
            last_rank=self.last_rank_for(gids),
            fanout=list(self.fcfg["fanout"]),
            batch_size=int(batch_size or self.fcfg["batch_size"]),
            shuffle=shuffle,
            seed=int(seed),
            runtime=self.runtime,
            device=self.device,
        )

    @property
    def val_y(self) -> np.ndarray:
        from aml.models.gnn import graph as G

        if self._val_y is None:
            self._val_y = G.labels_for(self.g, self.val_gids).astype(np.int8)
        return self._val_y

    @property
    def val_loader(self):
        if self._val_loader is None:
            self._val_loader = self.loader("val", self.val_gids, shuffle=False, seed=0)
        return self._val_loader

    def new_model(self) -> nn.Module:
        from aml.models.gnn import model as M

        return M.build_model(self.gnn_cfg, "faithful", self.preprocess, self.params).to(self.device)

    def train_loader(self, seed: int):
        if self._train is None or self._train[0] != int(seed):
            self.close()
            loader = self.loader("train", self.train_gids, shuffle=True, seed=seed)
            self._train = (int(seed), loader)
        return self._train[1]

    def close(self) -> None:
        """Stop the train loader's persistent workers (sampler.close_loader) and drop it."""
        from aml.models.gnn import sampler as S

        if self._train is not None:
            S.close_loader(self._train[1])
        self._train = None

    def train_epoch(self, model: nn.Module, opt: Any, seed: int, epoch: int) -> dict:
        import torch

        loader = self.train_loader(seed)
        loader.sampler.set_epoch(int(epoch))  # order = f(seed, epoch): sampler.FaithfulEpochSampler
        it = iter(loader)
        seed_epoch(seed, epoch, self.device)  # after iter(): see seed_epoch
        model.train()
        guard = empty_guard()
        loss_acc = torch.zeros((), device=self.device)
        steps = edges = skipped = 0
        for fb in it:
            guard = add_guard(guard, _guard_of(fb))
            loss = faithful_train_batch(
                model,
                opt,
                fb,
                ea=self.ea["train"],
                y=self.y,
                cw=self.cw,
                device=self.device,
                y_host=self.g.y,
            )
            edges += _edges_of(fb)
            if loss is None:
                skipped += 1
                continue
            loss_acc += loss
            steps += 1
        if steps == 0:
            raise ValueError("no faithful train batch had a sampled target")
        return {
            "train_loss": float(loss_acc.item()) / steps,
            "steps": steps,
            "skipped_batches": skipped,
            "edges": edges,
            "oom_splits": 0,
            "guard": guard,
        }

    def eval_pass(self, models: Mapping[int, nn.Module], snapshot: str, loader, n: int):
        """Scores of every seed of `loader` for every model, the `sampled` mask (Multi-GNN's)
        and the snapshot guard totals."""
        import torch

        out = {k: np.full(n, np.nan) for k in models}
        sampled = np.zeros(n, dtype=bool)
        seen = np.zeros(n, dtype=np.int64)
        guard = empty_guard()
        for m in models.values():
            m.eval()
        with torch.inference_mode():
            for fb in loader:
                guard = add_guard(guard, _guard_of(fb))
                pos_all = fb.seed_pos.cpu().numpy().astype(np.int64)
                sampled[pos_all] = fb.sampled.cpu().numpy().astype(bool)
                for pos, sc in eval_parts(
                    models, fb, ea=self.ea[snapshot], device=self.device, cap=None
                ):
                    seen[pos] += 1
                    for k in models:
                        out[k][pos] = sc[k]
        _check_coverage(seen, snapshot)
        return out, sampled, guard

    def validate(self, model: nn.Module) -> dict:
        from aml.eval.metrics import ARGMAX_THRESHOLD, prf_at_threshold

        n = len(self.val_gids)
        out, sampled, guard = self.eval_pass({0: model}, "val", self.val_loader, n)
        s, y = out[0], self.val_y
        f1_s = prf_at_threshold(y[sampled], s[sampled], ARGMAX_THRESHOLD)["f1"]
        f1_all = prf_at_threshold(y, s, ARGMAX_THRESHOLD)["f1"]
        return {
            "metric": f1_s,
            "val_f1_sampled": f1_s,
            "val_f1_all": f1_all,
            "val_sampled_share": float(sampled.mean()) if len(sampled) else None,
            "guard": guard,
            "val_cached": False,
        }

    def score_units(self, final: bool, test_bounds: tuple[str, ...]) -> list[ScoreUnit]:
        del test_bounds
        units = [ScoreUnit("val", VAL_SPLITS)]
        if final:
            units.append(ScoreUnit("test", ("test",)))
        return units

    def score(self, models: Mapping[int, nn.Module], unit: ScoreUnit) -> ScorePass:
        from aml.models.gnn import graph as G

        t0 = time.perf_counter()
        if unit.name == "val":
            gids, split, loader, snap = self.val_gids, self.val_split, self.val_loader, "val"
        elif unit.name == "test":
            if not self.allow_test:
                raise AssertionError("a test loader is not allowed here (final=False)")
            gids = G.split_gids(self.g, "test")
            split = np.full(len(gids), "test", dtype=object)
            loader, snap = self.loader("test", gids, shuffle=False, seed=0), "test"
        else:
            raise ValueError(f"unknown faithful scoring unit {unit.name!r}")
        out, sampled, guard = self.eval_pass(models, snap, loader, len(gids))
        return ScorePass(unit, gids, split, out, sampled, guard, time.perf_counter() - t0)


def _check_coverage(seen: np.ndarray, what: str) -> None:
    if len(seen) and not (seen == 1).all():
        bad = int((seen != 1).sum())
        raise AssertionError(f"{what}: {bad} seeds were not scored exactly once")


# --- one run: checkpoints, resume, the unclean-start counter (M3 spec §7.4-§7.6) -----------------


@dataclass
class RunSpec:
    """One (protocol, seed) training run (or one HPO trial) in `run_dir`."""

    run_key: str
    run_dir: Path
    protocol: str
    seed: int
    params: dict
    max_epochs: int
    min_epochs: int
    patience: int
    early_stopping: bool
    deterministic: bool
    matmul_precision: str
    fingerprint: dict
    graph_meta: dict
    mlflow_experiment: str | None = EXPERIMENT
    mlflow_tags: dict = field(default_factory=dict)
    epoch_cap: int | None = None


@dataclass
class RunOutcome:
    status: str  # "trained" | "partial" | "failed" | "pruned"
    epoch_next: int
    summary: dict | None = None
    error: str | None = None
    history: list[dict] = field(default_factory=list)
    guard: dict = field(default_factory=empty_guard)  # this call's totals


def fingerprint(
    g: HostGraph, run_key: str, protocol: str, seed: int, params: dict, preprocess: dict
) -> dict:
    """checkpoint.json of a run dir: everything that must match for a resume (M3 spec §7.5)."""
    from aml.config import config_hash

    return _plain(
        {
            "run_key": run_key,
            "gnn_version": GNN_VERSION,
            "features_digest": g.features_digest,
            "data_version": g.data_version,
            "spec_hash": g.spec_hash,
            "params": params,
            "protocol": protocol,
            "seed": int(seed),
            "preprocess_hash": config_hash(preprocess),
        }
    )


def graph_meta(g: HostGraph, protocol: str, preprocess: dict) -> dict:
    return _plain(
        {
            "preprocess": preprocess,
            "attr_columns": list(preprocess.get("columns") or g.attr_columns),
            "bounds": dict(g.bounds),
            "protocol": protocol,
            "n_nodes": int(g.n_nodes),
            "n_edges": int(g.n_edges),
        }
    )


def fingerprint_matches(run_dir: Path, fp: Mapping[str, Any]) -> bool:
    path = Path(run_dir) / FINGERPRINT_FILE
    try:
        return json.loads(path.read_text(encoding="utf-8")) == _plain(fp)
    except (FileNotFoundError, json.JSONDecodeError):
        return False


def prepare_run_dir(run_dir: Path, fp: Mapping[str, Any], meta: Mapping[str, Any]) -> bool:
    """Stamp `run_dir` with its fingerprint; a dir written for other inputs is emptied first
    (the `_reset_stale_checkpoints` pattern). Stray temp files of a crashed write are removed.
    Returns True if the dir was reset."""
    from aml.io import write_json_atomic

    run_dir = Path(run_dir)
    fp_path = run_dir / FINGERPRINT_FILE
    stale = fp_path.exists() and not fingerprint_matches(run_dir, fp)
    if stale:
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    for tmp in run_dir.glob(".*.tmp-*"):
        tmp.unlink(missing_ok=True)
    if not fp_path.exists():
        write_json_atomic(_plain(fp), fp_path)
    write_json_atomic(_plain(meta), run_dir / GRAPH_META_FILE)
    return stale


def train_run(
    engine: Any,
    spec: RunSpec,
    clock: Clock,
    *,
    on_checkpoint: Callable[[], None] | None = None,
    on_epoch: Callable[[int, float, list[dict]], bool] | None = None,
    on_resume: Callable[[list[dict]], bool] | None = None,
    heartbeat: Callable[[], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> RunOutcome:
    """Train (resume) one run to its end, or until the wall budget says stop.

    on_epoch(epoch, metric, history) -> prune? (HPO: trial.report + should_prune);
    on_resume(history) -> prune? (HPO: re-report the epochs of a resumed trial);
    heartbeat: refreshes the writer lease after every epoch. Exceptions propagate (run_set and
    run_hpo classify them); a "partial" outcome leaves last.pt for the next call."""
    import torch

    log = log or _print
    commit = on_checkpoint or _noop
    run_dir = Path(spec.run_dir)
    if prepare_run_dir(run_dir, spec.fingerprint, spec.graph_meta):
        log(f"{run_dir.name}: checkpoints of other inputs found; starting over")
    summary_path = run_dir / SEED_SUMMARY_FILE
    if summary_path.exists():
        doc = json.loads(summary_path.read_text(encoding="utf-8"))
        hist = read_jsonl(run_dir / HISTORY_FILE)
        return RunOutcome("trained", int(doc.get("epochs_run", 0)), summary=doc, history=hist)
    (run_dir / FAILED_FILE).unlink(missing_ok=True)  # a new attempt (the user re-ran it)

    ck = load_torch(run_dir / LAST_CKPT) if (run_dir / LAST_CKPT).exists() else None
    if ck is not None and _plain(ck.get("fingerprint")) != _plain(spec.fingerprint):
        raise ValueError(f"{run_dir}: last.pt was written for other inputs")
    epoch_next = int(ck["epoch_done"]) + 1 if ck else 0
    history = list(ck["history"]) if ck else []

    # The unclean-start counter: a running.json left at the same epoch = an unclean exit.
    running = run_dir / RUNNING_FILE
    prev = _read_json(running)
    starts = int(prev.get("starts", 0)) + 1 if prev and prev.get("epoch_next") == epoch_next else 1
    if starts >= MAX_STARTS:
        running.unlink(missing_ok=True)
        msg = f"no progress at epoch {epoch_next} after {MAX_STARTS - 1} attempts"
        _write_failed(run_dir, msg, None, epoch=epoch_next, spec=spec)
        commit()
        return RunOutcome("failed", epoch_next, error=msg, history=history)
    _write_json(running, {"epoch_next": epoch_next, "starts": starts})
    commit()

    det = set_determinism(
        spec.seed,
        deterministic=spec.deterministic,
        matmul_precision=spec.matmul_precision,
        device=engine.device,
    )
    model = engine.new_model()
    opt = torch.optim.Adam(model.parameters(), lr=float(spec.params["lr"]))
    best_metric, best_epoch, bad = -math.inf, None, 0
    best_state = None
    if ck:
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        best_metric = float(ck["best_metric"])
        best_epoch = None if ck["best_epoch"] is None else int(ck["best_epoch"])
        bad = int(ck["bad_epochs"])
        best_state = ck.get("best_model")
        _restore_rng(ck["rng"], engine.device)
        if best_state is not None:  # best.pt may be torn by a crash between the two writes
            save_torch_atomic(best_state, run_dir / BEST_CKPT)
        _rewrite_history(run_dir / HISTORY_FILE, history)
        log(f"{spec.run_key}: resuming at epoch {epoch_next}")
        if on_resume is not None and on_resume(history):
            return _finish_pruned(run_dir, history, epoch_next, commit)

    call_guard = empty_guard()
    stop = bool(ck) and should_stop(
        epoch_next - 1,
        bad,
        max_epochs=spec.max_epochs,
        min_epochs=spec.min_epochs,
        patience=spec.patience,
        early_stopping=spec.early_stopping,
    )
    with _mlflow_run(spec.mlflow_experiment, spec.run_key, spec.mlflow_tags, log) as mlf:
        if mlf is not None:
            _mlflow_call(log, _log_params, spec, det)
        epoch = epoch_next
        while not stop:
            est = _epoch_estimate(history, clock)
            if clock.out_of_time(est):
                running.unlink(missing_ok=True)
                commit()
                return RunOutcome("partial", epoch, history=history, guard=call_guard)
            rec, metric = _run_epoch(engine, model, opt, spec, epoch)
            improved, best_metric, best_epoch, bad = early_stop_update(
                best_metric, best_epoch, bad, metric, epoch
            )
            if improved:
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                save_torch_atomic(best_state, run_dir / BEST_CKPT)
            rec.update(improved=improved, best_epoch=best_epoch, bad_epochs=bad)
            rec = _plain(rec)
            history.append(rec)
            save_torch_atomic(
                {
                    "model": model.state_dict(),
                    "optimizer": opt.state_dict(),
                    "epoch_done": epoch,
                    "best_metric": best_metric,
                    "best_epoch": best_epoch,
                    "bad_epochs": bad,
                    "rng": _capture_rng(engine.device),
                    "best_model": best_state,
                    "history": history,
                    "fingerprint": _plain(spec.fingerprint),
                    "preprocess": _plain(engine.preprocess),
                },
                run_dir / LAST_CKPT,
            )
            append_jsonl(run_dir / HISTORY_FILE, rec)
            _write_json(running, {"epoch_next": epoch + 1, "starts": 1})
            if heartbeat is not None:
                heartbeat()
            commit()
            for split_guard in rec["guard"].values():
                call_guard = add_guard(call_guard, split_guard)
            clock.last_epoch_s = float(rec["seconds"])
            clock.unit_done()
            log(_epoch_line(spec, rec))
            if mlf is not None:
                _mlflow_call(log, _log_epoch, rec)
            if on_epoch is not None and on_epoch(epoch, metric, history):
                return _finish_pruned(run_dir, history, epoch + 1, commit, call_guard)
            stop = should_stop(
                epoch,
                bad,
                max_epochs=spec.max_epochs,
                min_epochs=spec.min_epochs,
                patience=spec.patience,
                early_stopping=spec.early_stopping,
            )
            epoch += 1
        if best_epoch is None:
            raise ValueError(f"{spec.run_key}: no epoch was trained (max_epochs {spec.max_epochs})")
        summary = _seed_summary(engine, spec, history, best_epoch, best_metric, det)
        _write_json(summary_path, summary)
        running.unlink(missing_ok=True)
        commit()
        if mlf is not None:
            _mlflow_call(
                log, _log_metrics, {"best_epoch": best_epoch, "best_metric": summary["metric"]}
            )
    return RunOutcome("trained", epoch, summary=summary, history=history, guard=call_guard)


def _finish_pruned(run_dir, history, epoch_next, commit, guard=None) -> RunOutcome:
    (Path(run_dir) / RUNNING_FILE).unlink(missing_ok=True)
    commit()
    return RunOutcome("pruned", epoch_next, history=history, guard=guard or empty_guard())


def _run_epoch(engine: Any, model: nn.Module, opt: Any, spec: RunSpec, epoch: int):
    import torch

    cuda = engine.device == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    tr = engine.train_epoch(model, opt, spec.seed, epoch)
    if cuda:
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    va = engine.validate(model)
    t2 = time.perf_counter()
    metric = va["metric"]
    rec: dict[str, Any] = {
        "epoch": int(epoch),
        "train_loss": tr["train_loss"],
        "metric": _finite_or_none(metric),
        engine.metric_name: _finite_or_none(metric),
        "seconds": t2 - t0,
        "train_s": t1 - t0,
        "val_s": t2 - t1,
        "steps": tr["steps"],
        "edges": tr["edges"],
        "oom_splits": tr["oom_splits"],
        "guard": {"train": tr["guard"], engine.val_guard_key: va["guard"]},
        "peak_gpu_bytes": int(torch.cuda.max_memory_allocated()) if cuda else None,
        "val_cached": bool(va.get("val_cached", False)),
    }
    for k in ("val_f1_all", "val_sampled_share", "skipped_batches"):
        if k in va or k in tr:
            rec[k] = _finite_or_none(va.get(k, tr.get(k)))
    return rec, metric


def _seed_summary(engine, spec: RunSpec, history, best_epoch, best_metric, det) -> dict:
    guard: dict[str, dict] = {}
    for rec in history:
        for split, gv in rec["guard"].items():
            guard[split] = add_guard(guard.get(split), gv)
    best = _finite_or_none(best_metric)
    out = {
        "run_key": spec.run_key,
        "protocol": spec.protocol,
        "seed": int(spec.seed),
        "params": spec.params,
        "best_epoch": int(best_epoch),
        "metric": best,
        ("best_val_f1" if spec.protocol == "faithful" else "best_val_ap"): best,
        "epochs_run": len(history),
        "max_epochs": int(spec.max_epochs),
        "epoch_cap": spec.epoch_cap,
        "stopped_early": len(history) < spec.max_epochs,
        "oom_splits": int(sum(int(r.get("oom_splits") or 0) for r in history)),
        "guard": guard,
        "seconds": float(sum(float(r["seconds"]) for r in history)),
        "gpu": det.get("gpu"),
        "determinism": det,
        "gnn_version": GNN_VERSION,
        "features_digest": engine.g.features_digest,
        "data_version": engine.g.data_version,
        "spec_hash": engine.g.spec_hash,
    }
    if spec.protocol == "lookahead":
        out["future_share"] = {s: future_share(gv) for s, gv in guard.items()}
    return _plain(out)


def _epoch_estimate(history: list[dict], clock: Clock) -> float | None:
    if history:
        return max(float(r["seconds"]) for r in history[-3:])
    return clock.last_epoch_s


def _epoch_line(spec: RunSpec, rec: dict) -> str:
    m = rec.get("metric")
    ms = "nan" if m is None else f"{m:.4f}"
    return (
        f"[{spec.protocol} s{spec.seed}] epoch {rec['epoch']}: loss {rec['train_loss']:.4f} "
        f"metric {ms} best {rec['best_epoch']} ({rec['seconds']:.1f}s, {rec['steps']} steps, "
        f"{rec['guard']['train']['edges_checked']} train edges checked)"
    )


def seed_epoch(seed: int, epoch: int, device: str) -> None:
    """Seed torch (CPU and CUDA) as a pure function of (seed, epoch), so dropout of epoch e is
    the same in an uninterrupted run and a resumed one. Engines call it AFTER iter(loader): a
    DataLoader draws its workers' base seed from the global RNG when an iterator is created,
    and persistent workers create one only in the first epoch (a resumed run: at its first)."""
    import torch

    s = int(np.random.SeedSequence([int(seed), int(epoch), EPOCH_SALT]).generate_state(1)[0])
    torch.manual_seed(s)
    if device == "cuda":
        torch.cuda.manual_seed_all(s)


def _capture_rng(device: str) -> dict:
    import torch

    kind, key, pos, has_gauss, cached = np.random.get_state()  # noqa: NPY002
    return {
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda_all": torch.cuda.get_rng_state_all() if device == "cuda" else [],
        "numpy": {
            "kind": str(kind),
            "key": [int(x) for x in key],
            "pos": int(pos),
            "has_gauss": int(has_gauss),
            "cached": float(cached),
        },
        "python": _plain_rng(random.getstate()),
    }


def _plain_rng(state: Any) -> Any:
    if isinstance(state, tuple | list):
        return [_plain_rng(x) for x in state]
    return state


def _restore_rng(rng: Mapping[str, Any], device: str) -> None:
    import torch

    torch.set_rng_state(rng["torch_cpu"])
    cuda_states = list(rng.get("torch_cuda_all") or [])
    if device == "cuda" and cuda_states and len(cuda_states) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(cuda_states)
    n = rng["numpy"]
    np.random.set_state(  # noqa: NPY002
        (n["kind"], np.asarray(n["key"], dtype=np.uint32), n["pos"], n["has_gauss"], n["cached"])
    )
    version, internal, gauss = rng["python"]
    random.setstate((version, tuple(internal), gauss))


def _rewrite_history(path: Path, history: list[dict]) -> None:
    from aml.io import write_text_atomic

    text = "".join(json.dumps(r, sort_keys=True) + "\n" for r in history)
    write_text_atomic(text, path)


# --- run_set (M3 spec §7.1, §7.7) ----------------------------------------------------------------


@dataclass
class _CallState:
    run_dir: Path | None = None
    run_key: str | None = None
    seed: int | None = None
    epoch: int | None = None
    guard: dict = field(default_factory=empty_guard)
    engine: Any = None  # closed when the call ends (stops persistent loader workers)

    def close_engine(self) -> None:
        """Stop the engine's persistent loader workers now (also when an exception
        propagates), not later in the cyclic GC (sampler.close_loader)."""
        engine, self.engine = self.engine, None
        if engine is not None:
            engine.close()


def run_set(
    paths: DataPaths,
    features_dir: Path,
    set_dir: Path,
    gnn_cfg: dict,
    *,
    data_cfg: dict,
    protocol: str,
    seeds: list[int],
    run_keys: dict[int, str],
    params: dict,
    final: bool,
    device: str,
    runtime: dict,
    budget_s: float,
    test_bounds: tuple[str, ...] = ("end",),
    dev: bool = False,
    max_epochs: int | None = None,
    on_checkpoint: Callable[[], None] | None = None,
    on_reload: Callable[[], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Train (resume) `seeds` of one protocol sequentially in this container, then score and
    assemble (M3 spec §7.1, §7.7). Never raises for deterministic errors (§7.6); re-raises
    transient ones (is_transient) so Modal retries the input and the run resumes.

    set_dir: paths.gnn_set_dir(set_kind(protocol, dev=dev), gnn_set_key(...)).
    data_cfg: configs/data.yaml (passed to graph.load_graph).
    seeds: this submission's seeds (a subset of the set's list, in order).
    run_keys: seed -> run key for EVERY seed of the set's list (dict order = set order); the set
        is assembled when all of them have their score files.
    params: effective_params(...). final: score test (never with dev; HPO never calls this).
    device: "cuda" | "cpu". runtime: gnn.yaml runtime (num_workers, loader_timeout_s, ...).
    budget_s: this attempt's wall budget (costplan.attempt_wall_s); before each epoch or scoring
        pass, if elapsed + UNIT_MARGIN x the unit's estimate > budget_s: checkpoint and return
        "partial" (a call always completes at least one unit).
    test_bounds: ("end",) or, look-ahead, ("end", "d10"). max_epochs: the dev override.
    on_checkpoint: vol.commit (after every durable write); on_reload: vol.reload (before reading
        the writer lease).

    Returns {"status": "done" | "partial" | "failed" | "stopped" | "busy",
             "protocol", "set_dir", "seeds", "seeds_trained": [...], "seeds_scored": [...],
             "next": {"seed", "epoch"} | {"phase": "scoring"} | None (where a partial resumes),
             "elapsed_s", "gpu_seconds" (this call's wall on the device), "guard" (this call's
             GUARD totals), "error": str | None, "failed_run_key": str | None,
             "set_complete": bool, "summary": the set summary.json content when the set is
             assembled, else None}.
    "done" with set_complete False: this submission's seeds are trained and scored, but other
    seeds of the set's list are not (look-ahead seed 0 first). "stopped": STOPPED.json exists in
    set_dir (the driver's wall guard fired; a new submission's driver deletes it first). "busy":
    another call holds a fresh writer lease. A finished set short-circuits to "done" with the
    stored summary.
    """
    log = log or _print
    commit = on_checkpoint or _noop
    set_dir = Path(set_dir)
    clock = Clock(budget_s)
    res: dict[str, Any] = {
        "status": "failed",
        "protocol": protocol,
        "set_dir": str(set_dir),
        "seeds": [int(s) for s in seeds],
        "seeds_trained": [],
        "seeds_scored": [],
        "next": None,
        "elapsed_s": 0.0,
        "gpu_seconds": 0.0,
        "guard": empty_guard(),
        "error": None,
        "failed_run_key": None,
        "set_complete": False,
        "summary": None,
    }
    stored = finished_set(paths, features_dir, set_dir, protocol, final)
    if stored is not None:
        return {**res, "status": "done", "set_complete": True, "summary": stored}
    if on_reload is not None:
        on_reload()
    if (set_dir / STOPPED_FILE).exists():
        return {**res, "status": "stopped"}
    set_dir.mkdir(parents=True, exist_ok=True)
    lease = WriterLease(set_dir)
    if not lease.acquire():
        holder = lease.holder() or {}
        return {**res, "status": "busy", "error": f"writer lease held by {holder.get('call_id')}"}
    commit()
    st = _CallState()
    try:
        out = _run_set_body(
            paths,
            features_dir,
            set_dir,
            gnn_cfg,
            data_cfg=data_cfg,
            protocol=protocol,
            seeds=[int(s) for s in seeds],
            run_keys={int(k): str(v) for k, v in run_keys.items()},
            params=params,
            final=bool(final),
            device=device,
            runtime=runtime,
            clock=clock,
            test_bounds=tuple(test_bounds),
            dev=bool(dev),
            max_epochs=max_epochs,
            lease=lease,
            commit=commit,
            log=log,
            st=st,
            res=res,
        )
    except Exception as exc:
        if is_transient(exc):
            raise
        fail_dir = st.run_dir or set_dir
        if st.run_dir is not None:  # a failed return is a clean return: reset the counter
            running = Path(st.run_dir) / RUNNING_FILE
            st.epoch = (_read_json(running) or {}).get("epoch_next", st.epoch)
            running.unlink(missing_ok=True)
        _write_failed(fail_dir, f"{type(exc).__name__}: {exc}", exc, epoch=st.epoch, st=st)
        log(f"FAILED ({fail_dir}): {type(exc).__name__}: {exc}")
        out = {
            **res,
            "status": "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "failed_run_key": st.run_key,
        }
    finally:
        st.close_engine()
    out["elapsed_s"] = clock.elapsed()
    out["gpu_seconds"] = clock.elapsed() if device == "cuda" else 0.0
    out["guard"] = st.guard
    append_jsonl(
        set_dir / CALLS_FILE,
        {
            "call_id": lease.call_id,
            "status": out["status"],
            "elapsed_s": round(clock.elapsed(), 3),
            "device": device,
            "ended_at": round(time.time(), 3),
        },
    )
    lease.release()
    commit()
    return out


def _run_set_body(
    paths,
    features_dir,
    set_dir: Path,
    gnn_cfg: dict,
    *,
    data_cfg,
    protocol,
    seeds,
    run_keys,
    params,
    final,
    device,
    runtime,
    clock: Clock,
    test_bounds,
    dev,
    max_epochs,
    lease: WriterLease,
    commit,
    log,
    st: _CallState,
    res: dict,
) -> dict:
    from aml.models.gnn import graph as G

    _check_run_set_args(gnn_cfg, protocol, seeds, run_keys, params, final, dev, test_bounds)
    t0 = time.perf_counter()
    g = G.load_graph(
        paths,
        features_dir,
        gnn_cfg,
        data_cfg=data_cfg,
        label_splits=label_splits_for(protocol),
        protocol=protocol,
        log=log,
    )
    load_s = time.perf_counter() - t0
    if _reset_if_other_data(set_dir, g.data_version):  # the job normally did this already
        log(f"{set_dir}: built from other prepared data; starting over")
        lease.refresh()
    log(f"graph: {g.n_edges} edges, {g.n_nodes} nodes ({load_s:.1f}s)")
    allowed = VAL_SPLITS + (("test",) if final else ())
    if protocol == "faithful":
        engine: Any = FaithfulEngine(
            g,
            gnn_cfg,
            paths=paths,
            features_dir=features_dir,
            params=params,
            device=device,
            runtime=runtime,
            allow_test=final,
            log=log,
        )
    else:
        engine = TemporalEngine(
            g,
            gnn_cfg,
            protocol=protocol,
            params=params,
            device=device,
            runtime=runtime,
            allowed_splits=allowed,
            log=log,
        )
    st.engine = engine
    specs = {
        s: _run_spec(engine, gnn_cfg, protocol, s, run_keys[s], params, dev, max_epochs, paths)
        for s in run_keys
    }

    # 1) train this submission's seeds in order
    for seed in seeds:
        spec = specs[seed]
        st.run_dir, st.run_key, st.seed = spec.run_dir, spec.run_key, seed
        out = train_run(engine, spec, clock, on_checkpoint=commit, heartbeat=lease.refresh, log=log)
        st.guard = add_guard(st.guard, out.guard)
        st.epoch = out.epoch_next
        if out.status == "failed":
            return {
                **res,
                "status": "failed",
                "error": out.error,
                "failed_run_key": spec.run_key,
                "next": {"seed": seed, "epoch": out.epoch_next},
            }
        if out.status == "partial":
            return {**res, "status": "partial", "next": {"seed": seed, "epoch": out.epoch_next}}
        res["seeds_trained"].append(seed)
    st.run_dir = st.run_key = st.seed = st.epoch = None

    # 2) one scoring pass per unit over every trained seed of the set whose files lack it
    units = engine.score_units(final, test_bounds)
    trained = [
        s
        for s in run_keys
        if (specs[s].run_dir / SEED_SUMMARY_FILE).exists()
        and fingerprint_matches(specs[s].run_dir, specs[s].fingerprint)
    ]
    todo = _scoring_todo(specs, trained, units, g, protocol)
    if todo:
        est = _scoring_estimate(specs, trained, todo, units, engine)
        if clock.out_of_time(est):
            return {**res, "status": "partial", "next": {"phase": "scoring"}}
        _score_seeds(engine, specs, todo, units, final, commit, log, st)
        clock.unit_done()
        lease.refresh()
        commit()
    res["seeds_scored"] = sorted({s for need in todo.values() for s in need})

    # 3) assemble when every seed of the set's list has its score files
    missing = [s for s in run_keys if _scoring_todo(specs, [s], units, g, protocol)]
    not_trained = [s for s in run_keys if s not in trained]
    if missing or not_trained:
        log(f"set incomplete: seeds without scores {sorted(set(missing) | set(not_trained))}")
        return {**res, "status": "done", "set_complete": False}
    summary = _assemble_set(
        paths,
        set_dir,
        gnn_cfg,
        g=g,
        protocol=protocol,
        specs=specs,
        params=params,
        final=final,
        dev=dev,
        test_bounds=test_bounds,
        clock=clock,
        load_s=load_s,
        engine=engine,
        commit=commit,
    )
    log(f"set assembled: {set_dir}")
    return {**res, "status": "done", "set_complete": True, "summary": summary}


def _check_run_set_args(gnn_cfg, protocol, seeds, run_keys, params, final, dev, test_bounds):
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    if final and dev:
        raise ValueError("a dev run never scores test (final and dev are exclusive)")
    if dev and protocol != "causal":
        raise ValueError("the dev run is a causal run")
    if not run_keys:
        raise ValueError("run_keys is empty")
    if not seeds or len(set(seeds)) != len(seeds) or not set(seeds) <= set(run_keys):
        raise ValueError(f"seeds {seeds} must be distinct and a subset of {list(run_keys)}")
    if len(set(run_keys.values())) != len(run_keys):
        raise ValueError("run keys must be distinct per seed")
    want_bounds = TEST_BOUNDS if protocol == "lookahead" else ("end",)
    if tuple(test_bounds) != want_bounds:
        raise ValueError(f"{protocol}: test_bounds must be {want_bounds}, got {test_bounds}")
    if protocol == "faithful":
        fseed = int(gnn_cfg["protocols"]["faithful"]["seed"])
        if list(run_keys) != [fseed]:
            raise ValueError(f"the faithful set is seed {fseed} only, got {list(run_keys)}")
    want = effective_params(gnn_cfg, protocol, {k: params[k] for k in HPO_PARAMS})
    if protocol in ("pna", "faithful"):
        want = effective_params(gnn_cfg, protocol, None)
    if _plain(want) != _plain(params):
        raise ValueError(f"params do not match effective_params for {protocol}: {params}")


def _run_spec(engine, gnn_cfg, protocol, seed, run_key, params, dev, max_epochs, paths) -> RunSpec:
    t = gnn_cfg["train"]
    epoch_cap = None
    early = True
    if protocol == "faithful":
        f = gnn_cfg["protocols"]["faithful"]
        epoch_cap = f.get("epoch_cap")
        max_e = min(int(f["max_epochs"]), int(epoch_cap) if epoch_cap else 10**9)
        early = False
    else:
        max_e = int(t["max_epochs"])
    if max_epochs is not None:
        max_e = min(max_e, int(max_epochs))
    pre = engine.preprocess
    fp = fingerprint(engine.g, run_key, protocol, seed, params, pre)
    return RunSpec(
        run_key=run_key,
        run_dir=paths.gnn_run_dir(run_key),
        protocol=protocol,
        seed=int(seed),
        params=_plain(params),
        max_epochs=max_e,
        min_epochs=min(int(t["min_epochs"]), max_e),
        patience=int(t["patience"]),
        early_stopping=early,
        deterministic=bool(t["deterministic"]),
        matmul_precision=str(t["matmul_precision"]),
        fingerprint=fp,
        graph_meta=graph_meta(engine.g, protocol, pre),
        mlflow_experiment=EXPERIMENT,
        mlflow_tags={
            "protocol": protocol,
            "seed": str(seed),
            "gnn_version": str(GNN_VERSION),
            "dev": str(bool(dev)),
            "device": engine.device,
        },
        epoch_cap=None if epoch_cap is None else int(epoch_cap),
    )


# --- scoring and set assembly --------------------------------------------------------------------


def _file_splits(path: Path) -> set[str]:
    import polars as pl

    if not path.exists():
        return set()
    return set(pl.read_parquet(path, columns=["split"]).get_column("split").unique().to_list())


def _scoring_todo(specs, seeds, units, g: HostGraph, protocol: str) -> dict[str, list[int]]:
    """unit name -> seeds whose score file lacks the unit's splits. A d10 file scored under
    another d10 bound is refused (test is touched once per seed)."""
    todo: dict[str, list[int]] = {}
    for s in seeds:
        run_dir = specs[s].run_dir
        for u in units:
            path = run_dir / scores_file(s, u.test_bound)
            if set(u.splits) <= _file_splits(path):
                if u.test_bound == "d10":
                    rec = (_read_json(run_dir / SCORING_FILE) or {}).get(u.name) or {}
                    old = (rec.get("bounds") or {}).get("d10_last")
                    if old is not None and int(old) != int(g.bounds["d10_last"]):
                        raise ValueError(
                            f"{path} was scored with d10_last {old}, the graph now has "
                            f"{g.bounds['d10_last']} (data test_views changed?): refusing to "
                            "touch test again; delete the file to re-score deliberately"
                        )
                continue
            todo.setdefault(u.name, []).append(s)
    return todo


def _scoring_estimate(specs, seeds, todo, units, engine) -> float:
    """Conservative seconds of the scoring passes: the slowest per-epoch validation pass per
    row (the uncached one), times the rows of every unit, times the number of models."""
    from aml.data.split import SPLITS

    val_s = max(
        (
            float(r.get("val_s") or 0.0)
            for s in seeds
            for r in read_jsonl(specs[s].run_dir / HISTORY_FILE)
        ),
        default=0.0,
    )
    per_row = val_s / max(1, len(engine.val_gids))
    rows = {name: int((engine.g.split_code == i).sum()) for i, name in enumerate(SPLITS)}
    by_name = {u.name: u for u in units}
    return sum(
        per_row * sum(rows[s] for s in by_name[name].splits) * len(ss) for name, ss in todo.items()
    )


def _score_seeds(engine, specs, todo, units, final, commit, log, st: _CallState) -> None:
    import polars as pl

    from aml.io import write_json_atomic, write_parquet_atomic

    seeds = sorted({s for ss in todo.values() for s in ss}, key=list(specs).index)
    models = {}
    for s in seeds:
        m = engine.new_model()
        m.load_state_dict(load_torch(specs[s].run_dir / BEST_CKPT, engine.device))
        m.eval()
        models[s] = m
    new_rows: dict[tuple[int, str], list[pl.DataFrame]] = {}
    meta: dict[int, dict] = {
        s: dict(_read_json(specs[s].run_dir / SCORING_FILE) or {}) for s in seeds
    }
    for u in units:
        need = todo.get(u.name) or []
        if not need:
            continue
        sp = engine.score({s: models[s] for s in need}, u)
        st.guard = add_guard(st.guard, sp.guard)
        g = engine.g
        for s in need:
            cols = {
                "row_id": pl.Series("row_id", g.row_id[sp.gids], dtype=pl.Int64),
                "split": pl.Series("split", sp.split.astype(str), dtype=pl.String),
                score_column(s): pl.Series(score_column(s), sp.scores[s], dtype=pl.Float64),
            }
            if sp.sampled is not None:
                cols["sampled"] = pl.Series("sampled", sp.sampled, dtype=pl.Boolean)
            new_rows.setdefault((s, u.test_bound), []).append(pl.DataFrame(cols))
            rec = {
                "pass_id": sp.pass_id,
                "guard": sp.guard,
                "rows": len(sp.gids),
                "seconds": sp.seconds,
                "test_bound": u.test_bound,
                "bounds": dict(g.bounds),
                "seeds": need,
            }
            if sp.sampled is not None:
                rec["sampled_share"] = {
                    name: float(sp.sampled[sp.split == name].mean())
                    for name in u.splits
                    if (sp.split == name).any()
                }
            if engine.protocol == "lookahead":
                rec["future_share"] = future_share(sp.guard)
            meta[s][u.name] = _plain(rec)
        log(f"scored {u.name} ({len(sp.gids)} rows) for seeds {need} in {sp.seconds:.1f}s")
    for (s, bound), frames in new_rows.items():
        path = specs[s].run_dir / scores_file(s, bound)
        old = [pl.read_parquet(path)] if path.exists() else []
        df = pl.concat(old + frames, how="vertical")
        if df.get_column("row_id").n_unique() != df.height:
            raise AssertionError(f"{path}: duplicate rows after adding a scoring pass")
        write_parquet_atomic(df, path)
    for s in seeds:
        write_json_atomic(meta[s], specs[s].run_dir / SCORING_FILE)
    commit()
    if final:
        for s in seeds:
            if any(u.splits == ("test",) and s in (todo.get(u.name) or []) for u in units):
                models_touched = [f"gnn_{engine.protocol}"]
                if engine.protocol == "lookahead":
                    models_touched.append(LOOKAHEAD_D10_MODEL)
                _mlflow_tag_test(specs[s], models_touched, log)


def _read_scores(run_dir: Path, seed: int, bound: str, splits: Iterable[str]):
    import polars as pl

    df = pl.read_parquet(Path(run_dir) / scores_file(seed, bound))
    return df.filter(pl.col("split").is_in(list(splits)))


def _set_frame(specs, seeds, splits_by_bound: list[tuple[str, tuple[str, ...]]], faithful: bool):
    """The set's scores: (row_id, split[, sampled]) + score_s<k> per seed, rows in split order
    (val_early, val_late, test) and rank order within a split; every seed must score the same
    rows."""
    import polars as pl

    from aml.data.split import SPLITS

    out = None
    for s in seeds:
        parts = [_read_scores(specs[s].run_dir, s, b, sp) for b, sp in splits_by_bound]
        df = pl.concat(parts, how="vertical")
        order = {name: i for i, name in enumerate(SPLITS)}
        df = df.with_columns(
            pl.col("split").replace_strict(order, return_dtype=pl.Int8).alias("_o")
        )
        df = df.with_row_index("_i").sort(["_o", "_i"]).drop(["_o", "_i"])
        keys = ["row_id", "split"] + (["sampled"] if faithful else [])
        if out is None:
            out = df.select([*keys, score_column(s)])
        else:
            if not df.select(["row_id", "split"]).equals(out.select(["row_id", "split"])):
                raise AssertionError(f"seed {s} scored other rows than seed {seeds[0]}")
            out = out.with_columns(df.get_column(score_column(s)))
    return out


def _scoring_guards(specs, seeds, unit_names) -> dict[str, dict]:
    """Totals per unit over the UNIQUE scoring passes behind the seeds' score files."""
    seen: set[str] = set()
    out: dict[str, dict] = {}
    for s in seeds:
        meta = _read_json(specs[s].run_dir / SCORING_FILE) or {}
        for name in unit_names:
            rec = meta.get(name)
            if not rec or rec["pass_id"] in seen:
                continue
            seen.add(rec["pass_id"])
            out[name] = add_guard(out.get(name), rec["guard"])
    return out


def _assemble_set(
    paths,
    set_dir: Path,
    gnn_cfg: dict,
    *,
    g: HostGraph,
    protocol: str,
    specs: dict[int, RunSpec],
    params: dict,
    final: bool,
    dev: bool,
    test_bounds: tuple[str, ...],
    clock: Clock,
    load_s: float,
    engine: Any,
    commit,
) -> dict:
    from aml.io import write_json_atomic, write_parquet_atomic

    seeds = list(specs)
    faithful = protocol == "faithful"
    splits = VAL_SPLITS + (("test",) if final else ())
    seed_sums = {
        s: json.loads((specs[s].run_dir / SEED_SUMMARY_FILE).read_text("utf-8")) for s in seeds
    }
    for s, doc in seed_sums.items():
        if doc.get("features_digest") != g.features_digest:
            raise AssertionError(f"seed {s} was trained on other feature parts")
    units = [u.name for u in engine.score_units(final, test_bounds)]
    # Per split: the training passes (train, val_early | val) plus the scoring passes (val_early,
    # val_late, test, test_d10 | val, test); a shared scoring pass counts once.
    guards: dict[str, dict] = {}
    for doc in seed_sums.values():
        for split, gv in doc["guard"].items():
            guards[split] = add_guard(guards.get(split), gv)
    for split, gv in _scoring_guards(specs, seeds, units).items():
        guards[split] = add_guard(guards.get(split), gv)
    vals = [seed_sums[s]["metric"] for s in seeds]
    finite = [v for v in vals if v is not None]
    model_seed = int(gnn_cfg["hpo"]["model_seed"])
    fresh = [seed_sums[s]["metric"] for s in seeds if s != model_seed]
    fresh = [v for v in fresh if v is not None]
    calls = read_jsonl(set_dir / CALLS_FILE)
    gpu_s = sum(float(c.get("elapsed_s") or 0) for c in calls if c.get("device") == "cuda")
    if engine.device == "cuda":
        gpu_s += clock.elapsed()
    dets = [doc.get("determinism") or {} for doc in seed_sums.values()]
    metric_key = "best_val_f1" if faithful else "best_val_ap"
    base = {
        "protocol": protocol,
        "set_key": set_dir.name,
        "seeds": seeds,
        "run_keys": {str(s): specs[s].run_key for s in seeds},
        "params": _plain(params),
        "per_seed": {
            str(s): {
                "best_epoch": seed_sums[s]["best_epoch"],
                metric_key: seed_sums[s]["metric"],
                "epochs_run": seed_sums[s]["epochs_run"],
                "stopped_early": seed_sums[s]["stopped_early"],
            }
            for s in seeds
        },
        # val_early PR-AUC over the seeds (faithful: its val F1 is in best_val_f1 below)
        "best_val_ap_mean": float(np.mean(finite)) if finite and not faithful else None,
        "best_val_ap_std": float(np.std(finite)) if finite and not faithful else None,
        "best_val_ap_mean_fresh": float(np.mean(fresh)) if fresh and not faithful else None,
        "hpo_model_seed": model_seed,
        "features_digest": g.features_digest,
        "data_version": g.data_version,
        "spec_hash": g.spec_hash,
        "gnn_version": GNN_VERSION,
        "report_hash": report_hash(gnn_cfg) if final else None,
        "guard": guards,
        "scored_splits": list(splits),
        "final": bool(final),
        "dev": bool(dev),
        "timings": {
            "graph_load_s": load_s,
            "train_s": float(sum(d["seconds"] for d in seed_sums.values())),
            "this_call_s": clock.elapsed(),
        },
        # gpu / gpu_seconds / cores / memory_mib: the cost gate's fallback when billing fails
        "gpu": next((d.get("gpu") for d in dets if d.get("gpu")), None),
        "gpus": sorted({d.get("gpu") for d in dets if d.get("gpu")}),
        "gpu_seconds": gpu_s,
        "cores": engine.runtime.get("cpu"),
        "memory_mib": engine.runtime.get("memory_mib"),
        "determinism": dets[0] if dets else None,
        "determinism_consistent": all(_plain(d) == _plain(dets[0]) for d in dets),
        "bounds": dict(g.bounds),
    }
    base.update(_guard_totals(guards))
    if faithful:
        doc = seed_sums[seeds[0]]
        meta = _read_json(specs[seeds[0]].run_dir / SCORING_FILE) or {}
        share: dict[str, float] = {}
        for rec in meta.values():
            share.update(rec.get("sampled_share") or {})
        fcfg = gnn_cfg["protocols"]["faithful"]
        base.update(
            {
                "epochs_run": doc["epochs_run"],
                "max_epochs": int(fcfg["max_epochs"]),
                "epoch_cap": fcfg.get("epoch_cap"),
                "epochs_planned": doc["max_epochs"],
                "best_epoch": doc["best_epoch"],
                "best_val_f1": doc["metric"],
                "batch_size": int(fcfg["batch_size"]),
                "sampled_share": share,
                "exemptions": {
                    "features": list(FAITHFUL_EXEMPT_FEATURES),
                    "norm": FAITHFUL_EXEMPT_NORM,
                    "label_splits": list(label_splits_for("faithful")),
                },
                "recalled": list(RECALLED),
            }
        )
    if protocol == "lookahead":
        base["future_share"] = {k: future_share(v) for k, v in guards.items()}
    end_parts = [("end", splits)]
    model = FAITHFUL_MODEL if faithful else set_kind(protocol, dev=dev)
    if protocol == "lookahead":
        d10_dir = paths.gnn_set_dir(LOOKAHEAD_D10_MODEL, set_dir.name)
        _reset_if_other_data(d10_dir, g.data_version)
        d10_parts = [("end", VAL_SPLITS)] + ([("d10", ("test",))] if final else [])
        frame = _set_frame(specs, seeds, d10_parts, False)
        write_parquet_atomic(frame, d10_dir / SCORES_FILE)
        _stamp_data_version(d10_dir, g.data_version)
        d10 = {**base, "model": LOOKAHEAD_D10_MODEL, "test_bound": "d10", "views": ["primary"]}
        d10["guard"] = {k: v for k, v in guards.items() if k != "test"}
        d10.update(_guard_totals(d10["guard"]))
        d10["future_share"] = {k: future_share(v) for k, v in d10["guard"].items()}
        write_json_atomic(_plain(d10), d10_dir / SUMMARY_FILE)
        base["guard"] = {k: v for k, v in guards.items() if k != "test_d10"}
        base.update(_guard_totals(base["guard"]))
        base["future_share"] = {k: future_share(v) for k, v in base["guard"].items()}
        base["d10_set_dir"] = str(d10_dir)
    frame = _set_frame(specs, seeds, end_parts, faithful)
    write_parquet_atomic(frame, set_dir / SCORES_FILE)
    _stamp_data_version(set_dir, g.data_version)
    summary = _plain(
        {**base, "model": model, "test_bound": "end", "test_bounds": list(test_bounds)}
    )
    write_json_atomic(summary, set_dir / SUMMARY_FILE)  # LAST: the completion marker
    commit()
    return summary


def _guard_totals(guards: Mapping[str, Mapping[str, int]]) -> dict:
    tot = empty_guard()
    for gv in guards.values():
        tot = add_guard(tot, gv)
    return {
        "edges_checked": tot["edges_checked"],
        "violations": tot["violations"],
        "target_hits": tot["target_hits"],
        "dropped_target_copies": tot["dropped_target_copies"],
    }


def current_inputs(paths, features_dir: Path) -> tuple[str | None, str | None]:
    """(prepared data version, feature parts digest) as the prepare marker and the feature
    build summary record them: a cheap staleness check before a short-circuit (run keys hash
    configs only; a re-prepare or re-replay under the same keys changes these)."""
    from aml.features.spec import FEATURES_DIGEST
    from aml.models.gnn.graph import PREPARE_MARKER

    dv = (_read_json(paths.parquet_dir / PREPARE_MARKER) or {}).get("data_version")
    fd = (_read_json(Path(features_dir) / SUMMARY_FILE) or {}).get(FEATURES_DIGEST)
    return dv, fd


def built_from_current(doc: Mapping[str, Any], paths, features_dir: Path) -> bool:
    """A stored summary was built from the prepared data and feature parts on the Volume now."""
    dv, fd = current_inputs(paths, features_dir)
    return doc.get("data_version") == dv and (fd is None or doc.get("features_digest") == fd)


def finished_set(paths, features_dir, set_dir: Path, protocol: str, final: bool) -> dict | None:
    """The stored set summary if the set is complete for this call: built from the current
    data and feature parts, and final if the call is (a non-final set does not satisfy a final
    call: its seeds still need their test scores). run_set and the CPU driver
    (modal_jobs.train_gnn) both short-circuit on it; torch-free."""
    doc = _read_json(Path(set_dir) / SUMMARY_FILE)
    if doc is None or (final and not doc.get("final")):
        return None
    if not built_from_current(doc, paths, features_dir):
        return None
    if protocol == "lookahead":
        d10 = paths.gnn_set_dir(LOOKAHEAD_D10_MODEL, Path(set_dir).name) / SUMMARY_FILE
        if not d10.exists():
            return None
    return doc


# --- small utilities -----------------------------------------------------------------------------


def _print(msg: str) -> None:
    print(msg, flush=True)


def _noop() -> None:
    return None


def _version(module: str) -> str | None:
    try:
        return str(__import__(module).__version__)
    except Exception:  # noqa: BLE001 - optional package
        return None


def _finite_or_none(v: Any) -> float | None:
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _json_default(o: Any) -> Any:
    if hasattr(o, "item") and callable(o.item) and getattr(o, "shape", None) == ():
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")


def _plain(obj: Any) -> Any:
    """JSON round trip: numpy scalars -> Python, tuples -> lists, keys -> str."""
    return json.loads(json.dumps(obj, default=_json_default))


def _read_json(path: Path) -> dict | None:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def _write_json(path: Path, obj: Any) -> None:
    from aml.io import write_json_atomic

    write_json_atomic(_plain(obj), Path(path))


def _write_failed(
    where: Path,
    msg: str,
    exc: BaseException | None,
    *,
    epoch: int | None,
    spec: RunSpec | None = None,
    st: _CallState | None = None,
) -> None:
    tb = "".join(traceback.format_exception(exc)) if exc is not None else None
    doc = {
        "error": msg,
        "type": type(exc).__name__ if exc is not None else None,
        "traceback": tb,
        "epoch": epoch,
        "run_key": spec.run_key if spec else (st.run_key if st else None),
        "protocol": spec.protocol if spec else None,
        "seed": spec.seed if spec else (st.seed if st else None),
        "detail": getattr(exc, "detail", None),
    }
    Path(where).mkdir(parents=True, exist_ok=True)
    _write_json(Path(where) / FAILED_FILE, doc)


def _stamp_data_version(stage_dir: Path, version: str | None) -> None:
    """data_version.json as modal_jobs.common.stamp_data_version writes it."""
    path = Path(stage_dir) / "data_version.json"
    if _read_json(path) != {"data_version": version}:
        _write_json(path, {"data_version": version})


def _reset_if_other_data(stage_dir: Path, version: str | None) -> bool:
    """modal_jobs.common.reset_if_other_data, also for the look-ahead d10 dir the job does not
    see: empty a dir stamped with other prepared data, then stamp it. True if it was emptied."""
    doc = _read_json(Path(stage_dir) / "data_version.json")
    stale = doc is not None and doc.get("data_version") != version
    if stale:
        shutil.rmtree(stage_dir)
    _stamp_data_version(stage_dir, version)
    return stale


# --- MLflow (telemetry: a tracking error never fails a run) -------------------------------------


class _MlflowRun:
    def __init__(self, experiment: str | None, run_key: str, tags: dict, log) -> None:
        self.experiment, self.run_key, self.tags, self.log = experiment, run_key, tags, log
        self.cm = None

    def __enter__(self):
        if not self.experiment or not MLFLOW_ENABLED:
            return None
        try:
            from aml import tracking

            self.cm = tracking.start_run_for_key(
                self.experiment, self.run_key, run_name=self.run_key, tags=self.tags
            )
            return self.cm.__enter__()
        except Exception as e:  # noqa: BLE001 - telemetry only
            self.log(f"MLflow unavailable ({type(e).__name__}: {e}); continuing without it")
            self.cm = None
            return None

    def __exit__(self, et, ev, tb):
        if self.cm is not None:
            try:
                self.cm.__exit__(et, ev, tb)
            except Exception as e:  # noqa: BLE001
                if e is not ev:
                    self.log(f"MLflow end_run failed ({type(e).__name__}: {e})")
        return False


MLFLOW_ENABLED = True  # tests may switch it off for speed


def _mlflow_run(experiment, run_key, tags, log) -> _MlflowRun:
    return _MlflowRun(experiment, run_key, dict(tags), log)


def _mlflow_call(log, fn, *args) -> None:
    try:
        fn(*args)
    except Exception as e:  # noqa: BLE001 - telemetry only
        log(f"MLflow logging failed ({type(e).__name__}: {e})")


def _log_params(spec: RunSpec, det: dict) -> None:
    from aml import tracking

    tracking.log_params_flat(
        {
            "params": spec.params,
            "protocol": spec.protocol,
            "seed": spec.seed,
            "max_epochs": spec.max_epochs,
            "min_epochs": spec.min_epochs,
            "patience": spec.patience,
            "determinism": det,
        }
    )
    import mlflow

    mlflow.set_tags({"gpu": str(det.get("gpu")), "run_dir": str(spec.run_dir)})


def _log_epoch(rec: dict) -> None:
    from aml import tracking

    keep = {
        k: rec[k]
        for k in rec
        if k not in ("epoch", "guard", "improved", "val_cached") and rec[k] is not None
    }
    keep["edges_checked"] = sum(gv["edges_checked"] for gv in rec["guard"].values())
    tracking.log_metrics_flat(keep, step=int(rec["epoch"]))


def _log_metrics(d: dict) -> None:
    from aml import tracking

    tracking.log_metrics_flat({k: v for k, v in d.items() if v is not None})


def _mlflow_tag_test(spec: RunSpec, models: list[str], log) -> None:
    if not MLFLOW_ENABLED:
        return
    try:
        from aml import tracking

        run_id = tracking.find_run_for_key(spec.mlflow_experiment or EXPERIMENT, spec.run_key)
        if run_id is not None:
            tracking.tag_test_touch(models, run_id=run_id)
    except Exception as e:  # noqa: BLE001 - telemetry only
        log(f"MLflow test-touch tag failed ({type(e).__name__}: {e})")
