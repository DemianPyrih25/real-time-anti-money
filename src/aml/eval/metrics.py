"""Ranking and alerting metrics (PLAN.md §4 "Metrics").

Every function takes optional non-negative integer weights `w`: a row with weight k counts as k
identical copies of that row. This is how the cluster bootstrap reweights rows, and the weighted
result equals the unweighted metric on physically replicated rows. Degenerate cases (no
positives, no negatives for ROC-AUC, no alerts for precision) return nan instead of raising. F1 with
positives but no alerts is 0 (2 TP / (alerts + positives) is well defined there): a model that
flags nothing has the worst F1, not a missing one, so seed means cannot drop it. Scores must be
finite; NaN scores are a data bug and raise.
"""

from __future__ import annotations

import numpy as np

# The papers' "argmax" operating point: flag when the positive-class probability >= 0.5.
ARGMAX_THRESHOLD = 0.5

NAN = float("nan")


def _as_binary(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y)
    if y.dtype != bool and not np.isin(y, (0, 1)).all():
        raise ValueError("labels must be 0/1")
    return y.astype(np.float64)


def _as_scores(s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=np.float64)
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    return s


def _as_weights(w: np.ndarray | None, n: int) -> np.ndarray | None:
    if w is None:
        return None
    w = np.asarray(w, dtype=np.float64)
    if w.shape != (n,):
        raise ValueError(f"weights have shape {w.shape}, expected ({n},)")
    if (w < 0).any():
        raise ValueError("weights must be non-negative")
    return w


def _num(x: float) -> int | float:
    """Counts come out as ints when they are whole numbers (JSON-friendly)."""
    x = float(x)
    return int(x) if x.is_integer() else x


def tie_curve(
    y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(thresholds, tp, fp) at each distinct score, descending; flag = s >= threshold.

    Tied scores form one group, so the curve does not depend on row order.
    """
    y = _as_binary(y)
    s = _as_scores(s)
    w = _as_weights(w, y.size)
    if y.size == 0:
        empty = np.zeros(0)
        return empty, empty, empty
    order = np.argsort(-s, kind="stable")
    s_o, y_o = s[order], y[order]
    if w is None:
        pos, neg = y_o, 1.0 - y_o
    else:
        w_o = w[order]
        pos, neg = y_o * w_o, (1.0 - y_o) * w_o
    last = np.r_[np.flatnonzero(s_o[1:] != s_o[:-1]), s_o.size - 1]
    return s_o[last], np.cumsum(pos)[last], np.cumsum(neg)[last]


def pr_auc(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
    """Average precision, step-wise like sklearn.average_precision_score: sum (R_k - R_k-1) P_k."""
    _, tp, fp = tie_curve(y, s, w)
    if tp.size == 0 or tp[-1] <= 0:
        return NAN
    flagged = tp + fp
    precision = np.divide(tp, flagged, out=np.zeros_like(tp), where=flagged > 0)
    recall = tp / tp[-1]
    return float(np.sum(np.diff(recall, prepend=0.0) * precision))


def roc_auc(y: np.ndarray, s: np.ndarray, w: np.ndarray | None = None) -> float:
    """Area under the ROC curve (ties count one half), like sklearn.roc_auc_score."""
    _, tp, fp = tie_curve(y, s, w)
    if tp.size == 0 or tp[-1] <= 0 or fp[-1] <= 0:
        return NAN
    tpr = np.r_[0.0, tp / tp[-1]]
    fpr = np.r_[0.0, fp / fp[-1]]
    return float(np.trapezoid(tpr, fpr))


def prf(tp: float, alerts: float, positives: float) -> dict[str, float]:
    """Precision, recall and F1 from counts.

    Precision is nan without alerts, recall and F1 are nan without positives; with positives but
    no alerts F1 is 0.
    """
    precision = tp / alerts if alerts > 0 else NAN
    recall = tp / positives if positives > 0 else NAN
    f1 = 2.0 * tp / (alerts + positives) if positives > 0 else NAN
    return {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def prf_from_flags(
    y: np.ndarray, flags: np.ndarray, w: np.ndarray | None = None
) -> dict[str, int | float]:
    """precision, recall, f1, alerts, tp, positives for a set of flagged rows.

    `flags` is Boolean per row, or a non-negative count of flagged copies per row (what
    `top_k_flags` returns with weights). Counts already include the weights, so `w` then only
    weights the positives; a count above the row's weight is an error.
    """
    y = _as_binary(y)
    f = np.asarray(flags)
    if f.shape != y.shape:
        raise ValueError(f"flags have shape {f.shape}, expected {y.shape}")
    w = _as_weights(w, y.size)
    if f.dtype == bool:
        flagged = f.astype(np.float64) if w is None else f * w
    else:
        flagged = f.astype(np.float64)
        if (flagged < 0).any() or (w is not None and (flagged > w).any()):
            raise ValueError("flag counts must be in [0, w]")
    alerts = float(flagged.sum())
    tp = float((flagged * y).sum())
    positives = float(y.sum() if w is None else (y * w).sum())
    out: dict[str, int | float] = dict(prf(tp, alerts, positives))
    out.update(alerts=_num(alerts), tp=_num(tp), positives=_num(positives))
    return out


def prf_at_threshold(
    y: np.ndarray, s: np.ndarray, thr: float, w: np.ndarray | None = None
) -> dict[str, int | float]:
    """prf_from_flags with flag = s >= thr."""
    return prf_from_flags(y, _as_scores(s) >= thr, w)


def best_f1_threshold(y: np.ndarray, s: np.ndarray) -> float:
    """The score threshold (flag = s >= thr) with the highest F1; the highest such threshold on
    ties. nan when there are no positives."""
    thr, tp, fp = tie_curve(y, s)
    if tp.size == 0 or tp[-1] <= 0:
        return NAN
    f1 = 2.0 * tp / (tp + fp + tp[-1])
    return float(thr[int(np.argmax(f1))])
