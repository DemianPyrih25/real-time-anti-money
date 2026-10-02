"""Paired, stratified cluster bootstrap (PLAN.md §4 "Uncertainty").

Strata: pattern positives (cluster = attempt_id), other positives (cluster = sender account),
negatives (cluster = sender account). Each replicate draws, within every stratum, as many clusters
as the stratum has, with replacement. A replicate is a vector of cluster counts; the weight of a
row is the count of its cluster (an integer, "this row appears k times").

Replicates are generated from (seed, replicate index) alone, so every statistic and every model
sees the same replicates (paired), whatever the chunk size.

Speed: statistics work on cluster counts directly. A weighted row sum is a dot product between the
counts and a per-cluster aggregate, restricted to the clusters that matter (positives, flagged
rows); PR-AUC needs only the tie groups of positive scores and, per group, the weighted number of
negatives scored at or above it (a sparse group x cluster matrix); top-K walks a short prefix of a
precomputed score order. Nothing touches every row per replicate.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import numpy as np
import scipy.sparse as sp

from aml.eval.operating_points import top_k_order
from aml.eval.typology import attempt_ids

STRATA = ("pattern_positive", "other_positive", "negative")

# A statistic maps cluster counts (n_clusters, b) to (b,) values or a dict of such arrays.
StatFn = Callable[[np.ndarray], np.ndarray | dict[str, np.ndarray]]


def make_clusters(y: np.ndarray, attempt_id: Any, src: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(strata, cluster_index): stratum per row (index into STRATA) and a cluster id per row.

    Cluster ids are 0..C-1, numbered stratum by stratum, so every stratum is a contiguous range.
    attempt_id < 0, NaN or null means "not a pattern row".
    """
    y = np.asarray(y).astype(bool)
    a = attempt_ids(attempt_id)
    src = np.asarray(src, dtype=np.int64)
    if not (a.shape == src.shape == y.shape):
        raise ValueError("y, attempt_id and src must have the same length")
    strata = np.where(y & (a >= 0), 0, np.where(y, 1, 2)).astype(np.int8)
    key = np.where(strata == 0, a, src)
    if (key < 0).any() or (key >= 1 << 40).any():
        raise ValueError("cluster keys must be in [0, 2**40)")
    _, cluster_index = np.unique(strata.astype(np.int64) << 40 | key, return_inverse=True)
    return strata, cluster_index.astype(np.int64)


def cluster_strata(strata: np.ndarray, cluster_index: np.ndarray) -> np.ndarray:
    """Stratum of each cluster, shape (C,)."""
    out = np.empty(int(cluster_index.max()) + 1 if cluster_index.size else 0, dtype=np.int8)
    out[cluster_index] = strata
    return out


def _stratum_layout(strata: np.ndarray, cluster_index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cs = cluster_strata(strata, cluster_index)
    if (np.diff(cs) < 0).any():
        raise ValueError("clusters must be numbered stratum by stratum (use make_clusters)")
    sizes = np.bincount(cs, minlength=len(STRATA))
    return sizes, np.r_[0, np.cumsum(sizes)[:-1]]


def _chunk_counts(
    sizes: np.ndarray, offsets: np.ndarray, seed: int, start: int, b: int
) -> np.ndarray:
    counts = np.empty((int(sizes.sum()), b), dtype=np.float64)
    for j in range(b):
        rng = np.random.default_rng([seed, start + j])
        for n_s, off in zip(sizes, offsets, strict=True):
            if n_s:
                counts[off : off + n_s, j] = np.bincount(
                    rng.integers(0, n_s, size=n_s), minlength=n_s
                )
    return counts


def replicate_counts(
    strata: np.ndarray, cluster_index: np.ndarray, B: int, seed: int, chunk: int = 32
) -> Iterator[np.ndarray]:
    """Yield cluster counts as float64 arrays of shape (C, b), b <= chunk, B replicates in all.

    Replicate r uses np.random.default_rng([seed, r]): the same replicates for any chunk size.
    """
    sizes, offsets = _stratum_layout(strata, cluster_index)
    for start in range(0, B, chunk):
        yield _chunk_counts(sizes, offsets, seed, start, min(chunk, B - start))


def replicate_weights(
    strata: np.ndarray, cluster_index: np.ndarray, B: int, seed: int, chunk: int = 32
) -> Iterator[np.ndarray]:
    """Yield integer row weights of shape (b, n): the same replicates as `replicate_counts`."""
    for counts in replicate_counts(strata, cluster_index, B, seed, chunk):
        yield counts[cluster_index].T.astype(np.int64)


def paired_bootstrap(
    stat_fns: dict[str, StatFn],
    strata: np.ndarray,
    cluster_index: np.ndarray,
    B: int,
    seed: int,
    chunk: int = 32,
    threads: int | None = None,
) -> dict[str, np.ndarray]:
    """Evaluate every statistic on the same B replicates; returns {name: (B,) samples}.

    A statistic returning a dict contributes one entry per key, named "<name>.<key>". Chunks run
    on `threads` threads (numpy/scipy release the GIL in the heavy parts); the result does not
    depend on `threads` or `chunk`.
    """
    sizes, offsets = _stratum_layout(strata, cluster_index)

    def work(start: int) -> dict[str, np.ndarray]:
        counts = _chunk_counts(sizes, offsets, seed, start, min(chunk, B - start))
        res: dict[str, np.ndarray] = {}
        for name, fn in stat_fns.items():
            out = fn(counts)
            items = out.items() if isinstance(out, dict) else [(None, out)]
            for key, v in items:
                res[name if key is None else f"{name}.{key}"] = np.asarray(v, dtype=np.float64)
        return res

    starts = range(0, B, chunk)
    if threads and threads > 1:
        with ThreadPoolExecutor(max_workers=threads) as pool:
            results = list(pool.map(work, starts))
    else:
        results = [work(s) for s in starts]
    if not results:
        return {}
    return {k: np.concatenate([r[k] for r in results]) for k in results[0]}


def ci(samples: np.ndarray, level: float = 0.95) -> tuple[float, float]:
    """Percentile interval over the finite samples; (nan, nan) if there are none."""
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return float("nan"), float("nan")
    alpha = (1.0 - level) / 2.0
    lo, hi = np.quantile(x, [alpha, 1.0 - alpha])
    return float(lo), float(hi)


def _prf_vec(tp: np.ndarray, alerts: np.ndarray, positives: np.ndarray) -> dict[str, np.ndarray]:
    nan = np.full(tp.shape, np.nan)
    ok_a, ok_p = alerts > 0, positives > 0
    return {
        "precision": np.divide(tp, alerts, out=nan.copy(), where=ok_a),
        "recall": np.divide(tp, positives, out=nan.copy(), where=ok_p),
        # As metrics.prf: F1 is 0 (not nan) with positives but no alerts.
        "f1": np.divide(2.0 * tp, alerts + positives, out=nan.copy(), where=ok_p),
    }


class ReplicateStats:
    """Builders of fast statistics for one set of rows (e.g. the primary test view).

    Each builder precomputes what it needs once and returns a StatFn for `paired_bootstrap`.
    Weighted results equal the metrics in aml.eval.metrics with row weights = counts[cluster].
    """

    def __init__(self, y: np.ndarray, cluster_index: np.ndarray) -> None:
        self.y = np.asarray(y).astype(bool)
        self.cluster_index = np.asarray(cluster_index, dtype=np.int64)
        self.n_clusters = int(self.cluster_index.max()) + 1 if self.cluster_index.size else 0
        self.positives = self.linear(self.y)

    def linear(self, values: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """sum_i w_i * values_i per replicate."""
        agg = np.bincount(
            self.cluster_index,
            weights=np.asarray(values, dtype=np.float64),
            minlength=self.n_clusters,
        )
        nz = np.flatnonzero(agg)
        v = agg[nz]
        return lambda counts: v @ counts[nz]

    def prf(self, flags: np.ndarray) -> StatFn:
        """precision / recall / f1 of a fixed set of flagged rows."""
        f = np.asarray(flags).astype(bool)
        tp, alerts = self.linear(f & self.y), self.linear(f)
        return lambda counts: _prf_vec(tp(counts), alerts(counts), self.positives(counts))

    def top_k(self, s: np.ndarray, k_fn: Callable[[np.ndarray], np.ndarray]) -> StatFn:
        """precision / recall / f1 of the weighted top-K rows, K = k_fn(counts) per replicate.

        Same order and partial last row as operating_points.top_k_flags(s, K, w).
        """
        order = top_k_order(s)
        y_o = self.y[order].astype(np.float64)
        ci_o = self.cluster_index[order]
        n = order.size

        def fn(counts: np.ndarray) -> dict[str, np.ndarray]:
            k = k_fn(counts)
            length = min(n, 2 * int(k.max(initial=0)) + 1024)
            while True:
                w = counts[ci_o[:length]]
                cum = np.cumsum(w, axis=0)
                if length == n or (cum[-1] >= k).all():
                    break
                length = min(n, 2 * length)
            take = np.clip(k[None, :] - (cum - w), 0.0, w)
            tp = y_o[:length] @ take
            return _prf_vec(tp, np.minimum(k, cum[-1]), self.positives(counts))

        return fn

    def pr_auc(self, s: np.ndarray) -> Callable[[np.ndarray], np.ndarray]:
        """Weighted average precision, as metrics.pr_auc(y, s, w)."""
        s = np.asarray(s, dtype=np.float64)
        if np.isnan(s).any():
            raise ValueError("scores contain NaN")
        t_asc = np.unique(s[self.y])
        g = t_asc.size
        if g == 0:
            return lambda counts: np.full(counts.shape[1], np.nan)
        # Group of a row = number of distinct positive scores strictly above it; a row counts at
        # every positive group whose threshold it reaches (s >= t).
        group = g - np.searchsorted(t_asc, s, side="right")
        neg = ~self.y & (group < g)

        def matrix(mask: np.ndarray) -> sp.csr_matrix:
            data = np.ones(int(mask.sum()))
            ij = (group[mask], self.cluster_index[mask])
            return sp.csr_matrix((data, ij), shape=(g, self.n_clusters))

        m_pos, m_neg = matrix(self.y), matrix(neg)

        def fn(counts: np.ndarray) -> np.ndarray:
            tp_g = m_pos @ counts
            tp = np.cumsum(tp_g, axis=0)
            fp = np.cumsum(m_neg @ counts, axis=0)
            flagged = tp + fp
            precision = np.divide(tp, flagged, out=np.zeros_like(tp), where=flagged > 0)
            total = tp[-1]
            return np.divide(
                (tp_g * precision).sum(axis=0),
                total,
                out=np.full(total.shape, np.nan),
                where=total > 0,
            )

        return fn
