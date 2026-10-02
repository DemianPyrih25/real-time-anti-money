"""Label-free feature gate before any fit: PSI, out-of-range support, learnability floor
(M2 spec §8.3, step 2).

Per feature, with p = warm train (days `psi_train_days`, where every window <= 3 d is complete)
and q = val_early:

- bins: a NaN bin, an exact-0 bin, then <= `psi_bins` quantile bins of the non-zero, non-NaN
  warm-train values (distinct edges only; right-closed, so x <= edge falls in that edge's bin).
  Categorical and flag features (FeatureDef `categorical` or `domain` in {"flag", "code"}) use the
  NaN bin and one bin per distinct value seen in either sample;
- psi = sum (q - p) ln(q / p) over the bins, both sides smoothed as
  (count + eps * N) / (N * (1 + k * eps)) with k = the number of bins (PSI is symmetric);
- oor = share of val_early non-NaN values outside [min, max] of the warm-train non-NaN values;
- nonzero = train rows (days 1-6) that are non-NaN and != 0 (a count, not a share: round trip is
  non-zero on 0.039% of the real rows and is still learnable);
- drop if psi > psi_max or oor > out_of_range_max or nonzero < min_train_nonzero;
- psi_warmup = PSI(days `warmup_days` vs `psi_train_days`) on the same bins: reported only.

If the dropped engine (non-TX) features exceed `max_drop_share` of the engine features, the doc
says `stop`; the caller writes gate.json and raises `GateStopError` (`check_gate`) so the user
decides. TX drops are reported but do not count towards that share. No label is read anywhere.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl

from aml.features.spec import FeatureDef, GateStopError

GATE_KEYS = (
    "psi_max",
    "psi_bins",
    "psi_eps",
    "psi_train_days",
    "warmup_days",
    "out_of_range_max",
    "min_train_nonzero",
    "max_drop_share",
)
VALUE_BIN_DOMAINS = ("flag", "code")  # one bin per value (plus the NaN bin)
TRAIN_SPLIT = "train"
VAL_SPLIT = "val_early"
REASONS = ("psi", "oor", "nonzero")


def psi(p_counts: np.ndarray, q_counts: np.ndarray, eps: float) -> float:
    """Smoothed population stability index between two count vectors over the same bins.

    Proportions are (count + eps * N) / (N * (1 + k * eps)), k = number of bins, so an empty bin
    never gives log(0) and both vectors still sum to 1.
    """
    p = np.asarray(p_counts, dtype=np.float64)
    q = np.asarray(q_counts, dtype=np.float64)
    if p.ndim != 1 or p.shape != q.shape or p.size == 0:
        raise ValueError(f"count vectors must be 1-D, non-empty and aligned: {p.shape}, {q.shape}")
    if (p < 0).any() or (q < 0).any():
        raise ValueError("counts must be >= 0")
    eps = float(eps)
    if not (math.isfinite(eps) and eps > 0):
        raise ValueError(f"psi_eps must be a positive number, got {eps!r}")
    n_p, n_q, k = float(p.sum()), float(q.sum()), p.size
    if n_p <= 0 or n_q <= 0:
        raise ValueError("both samples need at least one row")
    ps = (p + eps * n_p) / (n_p * (1.0 + k * eps))
    qs = (q + eps * n_q) / (n_q * (1.0 + k * eps))
    return float(np.sum((qs - ps) * np.log(qs / ps)))


def check_gate_cfg(gate_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Validated copy of lgbm.yaml `graph.gate`."""
    missing = [k for k in GATE_KEYS if k not in gate_cfg]
    if missing:
        raise ValueError(f"lgbm.graph.gate lacks {missing}")
    cfg = {k: gate_cfg[k] for k in GATE_KEYS}
    for k in ("psi_train_days", "warmup_days"):
        lo, hi = (int(d) for d in cfg[k])
        if not 1 <= lo <= hi:
            raise ValueError(f"lgbm.graph.gate.{k} must be [lo, hi] with 1 <= lo <= hi")
        cfg[k] = [lo, hi]
    cfg["psi_bins"] = int(cfg["psi_bins"])
    cfg["min_train_nonzero"] = int(cfg["min_train_nonzero"])
    if cfg["psi_bins"] < 1 or cfg["min_train_nonzero"] < 0:
        raise ValueError("psi_bins must be >= 1 and min_train_nonzero >= 0")
    for k in ("psi_max", "psi_eps", "out_of_range_max", "max_drop_share"):
        cfg[k] = float(cfg[k])
        if not math.isfinite(cfg[k]) or cfg[k] < 0:
            raise ValueError(f"lgbm.graph.gate.{k} must be a finite number >= 0")
    if cfg["psi_eps"] <= 0:
        raise ValueError("lgbm.graph.gate.psi_eps must be > 0")
    return cfg


def _is_value_binned(f: FeatureDef) -> bool:
    return bool(f.categorical) or f.domain in VALUE_BIN_DOMAINS


def _quantile_edges(ref: np.ndarray, n_bins: int) -> np.ndarray:
    """Distinct interior edges of <= n_bins quantile bins of the non-zero, non-NaN `ref` values.

    `inverted_cdf` returns data values, so the edges in gate.json are values that occur.
    """
    nz = ref[~np.isnan(ref) & (ref != 0)]
    if nz.size == 0 or n_bins <= 1:
        return np.empty(0, dtype=np.float64)
    probs = np.arange(1, n_bins, dtype=np.float64) / n_bins
    return np.unique(np.quantile(nz, probs, method="inverted_cdf"))


def _quantile_counts(x: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """[NaN, exact 0, value bins...]; value bin i holds edges[i-1] < x <= edges[i]."""
    nan = np.isnan(x)
    v = x[~nan]
    zero = v == 0
    idx = np.searchsorted(edges, v[~zero], side="left")
    return np.concatenate(
        [[nan.sum(), zero.sum()], np.bincount(idx, minlength=edges.size + 1)]
    ).astype(np.int64)


def _value_counts(x: np.ndarray, values: np.ndarray) -> np.ndarray:
    """[NaN, one bin per entry of `values` (sorted, and every non-NaN x is among them)]."""
    nan = np.isnan(x)
    idx = np.searchsorted(values, x[~nan])
    return np.concatenate([[nan.sum()], np.bincount(idx, minlength=values.size)]).astype(np.int64)


def _psi_value_binned(ref: np.ndarray, other: np.ndarray, eps: float) -> tuple[float, int]:
    values = np.unique(np.concatenate([ref[~np.isnan(ref)], other[~np.isnan(other)]]))
    p, q = _value_counts(ref, values), _value_counts(other, values)
    return psi(p, q, eps), int(p.size)


def _finite_or_none(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def gate_feature(
    f: FeatureDef,
    warm: np.ndarray,
    val: np.ndarray,
    train: np.ndarray,
    warmup: np.ndarray,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """The gate record of one feature from its float64 values on the four row sets."""
    eps = cfg["psi_eps"]
    if _is_value_binned(f):
        kind, edges = "value", None
        p_val, n_bins = _psi_value_binned(warm, val, eps)
        p_warmup = _psi_value_binned(warm, warmup, eps)[0] if warmup.size else None
    else:
        kind = "quantile"
        edges = _quantile_edges(warm, cfg["psi_bins"])
        ref = _quantile_counts(warm, edges)
        p_val, n_bins = psi(ref, _quantile_counts(val, edges), eps), int(ref.size)
        p_warmup = psi(ref, _quantile_counts(warmup, edges), eps) if warmup.size else None

    warm_ok = warm[~np.isnan(warm)]
    val_ok = val[~np.isnan(val)]
    lo = float(warm_ok.min()) if warm_ok.size else math.nan
    hi = float(warm_ok.max()) if warm_ok.size else math.nan
    if val_ok.size == 0:
        oor = 0.0
    elif warm_ok.size == 0:
        oor = 1.0  # nothing to compare with: every val_early value is unsupported
    else:
        oor = float(((val_ok < lo) | (val_ok > hi)).mean())
    nonzero = int(np.count_nonzero(~np.isnan(train) & (train != 0)))

    reasons = []
    if p_val > cfg["psi_max"]:
        reasons.append("psi")
    if oor > cfg["out_of_range_max"]:
        reasons.append("oor")
    if nonzero < cfg["min_train_nonzero"]:
        reasons.append("nonzero")
    return {
        "group": f.group,
        "kind": kind,
        "n_bins": n_bins,
        "edges": None if edges is None else [float(e) for e in edges],
        "psi": p_val,
        "psi_warmup": p_warmup,
        "oor": oor,
        "nonzero": nonzero,
        "warm_range": [_finite_or_none(lo), _finite_or_none(hi)],
        "nan_share_warm": float(np.isnan(warm).mean()) if warm.size else None,
        "nan_share_val": float(np.isnan(val).mean()) if val.size else None,
        "drop": bool(reasons),
        "reasons": reasons,
    }


def run_gate(table: pl.DataFrame, features: Sequence[FeatureDef], gate_cfg: dict) -> dict:
    """The gate document (per-feature psi, psi_warmup, oor, nonzero, drop reasons; kept and
    dropped lists; the drop share and whether the guard trips). `table` holds day, split and the
    feature columns (Float32); no label is read."""
    cfg = check_gate_cfg(gate_cfg)
    names = [f.name for f in features]
    missing = [c for c in ("day", "split", *names) if c not in table.columns]
    if missing:
        raise ValueError(f"gate table lacks columns: {missing}")
    day = table.get_column("day").to_numpy()
    split = table.get_column("split").to_numpy()
    (w_lo, w_hi), (u_lo, u_hi) = cfg["psi_train_days"], cfg["warmup_days"]
    train = split == TRAIN_SPLIT
    warm = train & (day >= w_lo) & (day <= w_hi)
    warmup = train & (day >= u_lo) & (day <= u_hi)
    val = split == VAL_SPLIT
    if not warm.any() or not val.any():
        raise ValueError(
            f"the gate needs warm-train rows (days {w_lo}-{w_hi}) and {VAL_SPLIT} rows; "
            f"got {int(warm.sum())} and {int(val.sum())}"
        )

    per: dict[str, dict[str, Any]] = {}
    for f in features:
        x = table.get_column(f.name).to_numpy().astype(np.float64)
        per[f.name] = gate_feature(f, x[warm], x[val], x[train], x[warmup], cfg)

    kept = [n for n in names if not per[n]["drop"]]
    dropped = [n for n in names if per[n]["drop"]]
    engine = [f.name for f in features if f.group != "TX"]
    dropped_engine = [n for n in engine if per[n]["drop"]]
    share = len(dropped_engine) / len(engine) if engine else 0.0
    return {
        "config": cfg,
        "rows": {
            "warm_train": int(warm.sum()),
            VAL_SPLIT: int(val.sum()),
            TRAIN_SPLIT: int(train.sum()),
            "warmup": int(warmup.sum()),
        },
        "features": per,
        "order": names,  # spec order (JSON files sort their keys)
        "kept": kept,
        "dropped": dropped,
        "dropped_by_reason": {r: [n for n in dropped if r in per[n]["reasons"]] for r in REASONS},
        "dropped_tx": [n for n in dropped if per[n]["group"] == "TX"],
        "n_engine": len(engine),
        "dropped_engine": dropped_engine,
        "drop_share": share,
        "stop": share > cfg["max_drop_share"],
    }


def check_gate(doc: Mapping[str, Any]) -> None:
    """Raise `GateStopError` when the gate's drop-share guard tripped (write gate.json first)."""
    if doc["stop"]:
        cfg = doc["config"]
        raise GateStopError(
            f"the gate would drop {len(doc['dropped_engine'])} of {doc['n_engine']} engine "
            f"features ({100 * doc['drop_share']:.1f}% > {100 * cfg['max_drop_share']:g}%): "
            f"{doc['dropped_engine']}. Nothing was fitted; read gate.json and decide."
        )
