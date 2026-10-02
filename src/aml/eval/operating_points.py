"""Operating points (PLAN.md §4 "Metrics"). One alert = one flagged transaction.

(a) deployable threshold: chosen on val_late so the model's alert rate matches the rules'
    val_late rate, then applied unchanged to test;
(b) iso-volume top-K: K = the rules' alert count in the view (evaluation only);
(c) rules + model: the union of rule alerts and model alerts at (a), reported with the union's
    volume next to the model alone at that same volume (top-|union|).
"""

from __future__ import annotations

import math

import numpy as np

from aml.eval.metrics import prf_from_flags

# Guards `rate * n` against float error when the rate came from a count (rules' A / n).
_EPS = 1e-6


def alert_budget(n: int, rate: float) -> int:
    """Largest alert count k with k / n <= rate."""
    if n <= 0 or not rate > 0:  # also catches a nan rate
        return 0
    return min(n, math.floor(rate * n + _EPS))


def threshold_for_alert_rate(s: np.ndarray, rate: float) -> float:
    """Smallest threshold with mean(s >= thr) <= rate.

    With k = floor(rate * n) this is the k-th highest score. Ties: if the tie block of the k-th
    highest score extends past k, the whole block is left out and the threshold moves up to the
    next higher distinct score, so the model never exceeds the budget on the rows it was chosen
    on. Returns +inf when the budget is zero (nothing flagged).
    """
    s = np.asarray(s, dtype=np.float64)
    if np.isnan(s).any():
        raise ValueError("scores contain NaN")
    k = alert_budget(s.size, rate)
    if k == 0:
        return math.inf
    desc = np.sort(s)[::-1]
    if k == s.size:
        return float(desc[-1])
    t = desc[k - 1]
    if desc[k] < t:
        return float(t)
    higher = desc[:k][desc[:k] > t]
    return float(higher[-1]) if higher.size else math.inf


def top_k_order(s: np.ndarray) -> np.ndarray:
    """Row order for top-K: score descending, ties broken by position (earlier row first)."""
    return np.argsort(-np.asarray(s, dtype=np.float64), kind="stable")


def top_k_flags(s: np.ndarray, k: int, w: np.ndarray | None = None) -> np.ndarray:
    """Flag the k highest-scored rows, ties broken by position (deterministic).

    Without weights returns a Boolean mask with exactly min(k, n) rows set. With integer weights
    (row i = w[i] copies) returns the number of flagged copies per row: rows are taken in top-K
    order until k copies are flagged, the last row possibly partially.
    """
    s = np.asarray(s, dtype=np.float64)
    order = top_k_order(s)
    k = max(0, int(k))
    if w is None:
        flags = np.zeros(s.size, dtype=bool)
        flags[order[:k]] = True
        return flags
    w_o = np.asarray(w, dtype=np.int64)[order]
    before = np.cumsum(w_o) - w_o
    counts = np.zeros(s.size, dtype=np.int64)
    counts[order] = np.clip(k - before, 0, w_o)
    return counts


def alerted_accounts_per_day(
    flags: np.ndarray, src: np.ndarray, day: np.ndarray, n_days: int
) -> float:
    """Distinct (sender account, day) pairs with at least one alert, per simulated day.

    A proxy for analyst cases: one case per alerted account per day.
    """
    if n_days <= 0:
        return float("nan")
    f = np.asarray(flags) > 0
    key = np.asarray(day, dtype=np.int64)[f] << 32 | np.asarray(src, dtype=np.int64)[f]
    return float(np.unique(key).size / n_days)


def point_metrics(
    y: np.ndarray, flags: np.ndarray, src: np.ndarray, day: np.ndarray, n_days: int
) -> dict[str, int | float]:
    """precision/recall/F1 plus alert volume (alerts, per day, alerted accounts per day)."""
    m = prf_from_flags(y, np.asarray(flags, dtype=bool))
    m["alerts_per_day"] = float(m["alerts"] / n_days) if n_days > 0 else float("nan")
    m["alerted_accounts_per_day"] = alerted_accounts_per_day(flags, src, day, n_days)
    return m


def deployable_point(
    y: np.ndarray, s: np.ndarray, thr: float, src: np.ndarray, day: np.ndarray, n_days: int
) -> dict[str, int | float]:
    """(a): flag = s >= thr, thr chosen on val_late."""
    m = point_metrics(y, np.asarray(s) >= thr, src, day, n_days)
    m["threshold"] = float(thr)
    return m


def iso_volume_point(
    y: np.ndarray, s: np.ndarray, k: int, src: np.ndarray, day: np.ndarray, n_days: int
) -> dict[str, int | float]:
    """(b): the model's top-K rows in the view."""
    m = point_metrics(y, top_k_flags(s, k), src, day, n_days)
    m["k"] = int(k)
    return m


def union_point(
    y: np.ndarray,
    s: np.ndarray,
    thr: float,
    rule_flags: np.ndarray,
    src: np.ndarray,
    day: np.ndarray,
    n_days: int,
) -> dict[str, dict[str, int | float]]:
    """(c): rules ∪ model at (a), and the model alone at the union's volume."""
    union = np.asarray(rule_flags, dtype=bool) | (np.asarray(s) >= thr)
    k = int(union.sum())
    return {
        "union": point_metrics(y, union, src, day, n_days),
        "model_same_volume": iso_volume_point(y, s, k, src, day, n_days),
    }


def operating_point_metrics(
    y: np.ndarray,
    s: np.ndarray,
    *,
    thr: float,
    rule_flags: np.ndarray,
    src: np.ndarray,
    day: np.ndarray,
    n_days: int,
) -> dict[str, dict]:
    """(a), (b) and (c) for one model seed in one view; K for (b) = rules' alerts in the view."""
    k = int(np.asarray(rule_flags, dtype=bool).sum())
    return {
        "a": deployable_point(y, s, thr, src, day, n_days),
        "b": iso_volume_point(y, s, k, src, day, n_days),
        "c": union_point(y, s, thr, rule_flags, src, day, n_days),
    }
