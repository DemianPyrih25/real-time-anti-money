"""Global TreeSHAP importances of the champion booster (M2 spec §8.3, step 7).

LightGBM's exact TreeSHAP (`pred_contrib=True`) on a validation sample: every positive plus a
uniform sample of negatives. Contributions are in raw-score units (log-odds for the binary
objective); per row they sum, with the expected value, to the raw score. Importance = mean
|contribution|; a group's importance is the sum of its features' importances.

Cost guardrail: TreeSHAP costs about rows x trees x leaves x depth^2, so a large tuned model can
make 100k rows expensive. Rows are explained in chunks (positives first, then the negatives in a
seeded random order); with `max_seconds`, no further chunk starts once the projected total would
exceed it. The negatives explained are then a prefix of the same seeded permutation, i.e. still a
uniform sample, and the document records how many were used (`budget_limited`).
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence

import lightgbm as lgb
import numpy as np

SPACE = "raw score (log-odds)"
CHUNK_ROWS = 8192


def sample_order(y: np.ndarray, negatives: int, seed: int) -> np.ndarray:
    """Row indices in explanation order: every positive (ascending), then min(negatives,
    #negatives) negatives in a seeded random order (any prefix is a uniform sample)."""
    y = np.asarray(y)
    if negatives < 0:
        raise ValueError(f"negatives must be >= 0, got {negatives}")
    pos = np.flatnonzero(y == 1)
    neg = np.flatnonzero(y != 1)
    neg = np.random.default_rng(int(seed)).permutation(neg)[: min(int(negatives), neg.size)]
    return np.concatenate([pos, neg])


def sample_rows(y: np.ndarray, negatives: int, seed: int) -> np.ndarray:
    """The full sample (no time budget), sorted: all positives + the seeded negatives."""
    return np.sort(sample_order(y, negatives, seed))


def _num_threads(threads: int | None) -> int:
    return int(threads) if threads else 0


def _nan_none(x: float) -> float | None:
    return float(x) if math.isfinite(x) else None


def shap_global(
    booster: lgb.Booster,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    groups: Mapping[str, str],
    *,
    negatives: int,
    seed: int,
    threads: int | None = None,
    max_seconds: float | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """`pred_contrib=True` on all positives + `negatives` uniform negatives (seed `seed`) of the
    val_early rows given: mean |contribution| (log-odds) per feature and per group (`groups`:
    feature -> group), on the sample and on positives only."""
    names = list(feature_names)
    if booster.feature_name() != names:
        raise ValueError("feature_names must be the booster's inputs, in order")
    if X.ndim != 2 or X.shape[1] != len(names) or X.shape[0] != len(y):
        raise ValueError(f"X {X.shape} does not match {len(names)} features and {len(y)} labels")
    missing = [n for n in names if n not in groups]
    if missing:
        raise ValueError(f"no group for features: {missing}")
    y = np.asarray(y)
    order = sample_order(y, negatives, seed)
    if order.size == 0:
        raise ValueError("no rows to explain")
    n_pos = int((y == 1).sum())
    num_iteration = booster.best_iteration if booster.best_iteration > 0 else None
    say = log or (lambda _msg: None)

    k = len(names)
    abs_sum, abs_sum_pos = np.zeros(k), np.zeros(k)
    bias_sum, done = 0.0, 0
    t0 = time.perf_counter()
    while done < order.size:
        if max_seconds is not None and done >= n_pos and done > 0:
            per_row = (time.perf_counter() - t0) / done
            if per_row * min(done + CHUNK_ROWS, order.size) > max_seconds:
                break  # the positives are always explained; stop adding negatives
        idx = order[done : done + CHUNK_ROWS]
        contrib = np.asarray(
            booster.predict(
                X[idx],
                pred_contrib=True,
                num_iteration=num_iteration,
                num_threads=_num_threads(threads),
            ),
            dtype=np.float64,
        )
        if contrib.shape != (idx.size, k + 1):
            raise ValueError(f"unexpected contribution shape {contrib.shape}")
        phi = np.abs(contrib[:, :-1])
        abs_sum += phi.sum(axis=0)
        abs_sum_pos += phi[y[idx] == 1].sum(axis=0)
        bias_sum += float(contrib[:, -1].sum())
        if done == 0:
            dt = time.perf_counter() - t0
            say(
                f"TreeSHAP: {dt:.1f} s for the first {idx.size:,} rows; projected "
                f"{dt / idx.size * order.size / 60:.1f} min for {order.size:,} rows"
            )
        done += idx.size

    n_neg = done - n_pos
    mean_abs = abs_sum / done
    mean_abs_pos = abs_sum_pos / n_pos if n_pos else np.full(k, np.nan)
    features = {
        n: {
            "group": groups[n],
            "mean_abs": float(mean_abs[i]),
            "mean_abs_pos": _nan_none(mean_abs_pos[i]),
        }
        for i, n in enumerate(names)
    }
    group_doc: dict[str, dict] = {}
    for i, n in enumerate(names):
        g = group_doc.setdefault(
            groups[n], {"mean_abs": 0.0, "mean_abs_pos": 0.0, "n_features": 0, "features": []}
        )
        g["mean_abs"] += float(mean_abs[i])
        g["mean_abs_pos"] += float(mean_abs_pos[i])
        g["n_features"] += 1
        g["features"].append(n)
    for g in group_doc.values():
        g["mean_abs_pos"] = _nan_none(g["mean_abs_pos"])
    # Stable descending order: ties keep the input (spec) order.
    ranking = [names[i] for i in np.argsort(-mean_abs, kind="stable")]
    group_ranking = sorted(group_doc, key=lambda g: -group_doc[g]["mean_abs"])
    return {
        "space": SPACE,
        "n_rows": int(done),
        "n_positives": n_pos,
        "n_negatives": int(n_neg),
        "negatives_requested": int(negatives),
        "budget_limited": bool(done < order.size),
        "max_seconds": max_seconds,
        "seconds": round(time.perf_counter() - t0, 3),
        "seed": int(seed),
        "num_iteration": num_iteration,
        "expected_value": bias_sum / done,
        "features": features,
        "ranking": ranking,
        "groups": group_doc,
        "group_ranking": group_ranking,
    }
