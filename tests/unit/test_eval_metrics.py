"""aml.eval.metrics: == sklearn unweighted, weighted == replicated rows, nan on degenerate."""

from __future__ import annotations

import math

import numpy as np
import pytest
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from aml.eval.metrics import (
    ARGMAX_THRESHOLD,
    best_f1_threshold,
    pr_auc,
    prf_at_threshold,
    prf_from_flags,
    roc_auc,
    tie_curve,
)


def _data(seed: int, n: int = 2000, ties: bool = True):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(np.int8)
    s = rng.random(n) * 0.7 + y * rng.random(n) * 0.5
    if ties:  # coarse scores: many ties, like a tree model on few features
        s = np.round(s, 2)
    return y, s


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("ties", [True, False])
def test_unweighted_equals_sklearn(seed, ties):
    y, s = _data(seed, ties=ties)
    assert pr_auc(y, s) == pytest.approx(average_precision_score(y, s), rel=1e-12)
    assert roc_auc(y, s) == pytest.approx(roc_auc_score(y, s), rel=1e-12)
    thr = float(np.quantile(s, 0.9))
    flags = s >= thr
    m = prf_from_flags(y, flags)
    assert m["precision"] == pytest.approx(precision_score(y, flags))
    assert m["recall"] == pytest.approx(recall_score(y, flags))
    assert m["f1"] == pytest.approx(f1_score(y, flags))
    assert m["alerts"] == int(flags.sum())
    assert m["tp"] == int((flags & (y == 1)).sum())
    assert m["positives"] == int(y.sum())
    assert prf_at_threshold(y, s, thr) == m


@pytest.mark.parametrize("seed", range(5))
def test_integer_weights_equal_physical_replication(seed):
    y, s = _data(seed, n=1500)
    w = np.random.default_rng(100 + seed).integers(0, 4, size=y.size)
    yr, sr = np.repeat(y, w), np.repeat(s, w)
    assert pr_auc(y, s, w) == pytest.approx(pr_auc(yr, sr), rel=1e-12)
    assert pr_auc(y, s, w) == pytest.approx(average_precision_score(yr, sr), rel=1e-12)
    assert roc_auc(y, s, w) == pytest.approx(roc_auc(yr, sr), rel=1e-12)
    assert roc_auc(y, s, w) == pytest.approx(roc_auc_score(yr, sr), rel=1e-12)
    flags = s >= 0.6
    assert prf_from_flags(y, flags, w) == pytest.approx(prf_from_flags(yr, np.repeat(flags, w)))
    # sklearn's sample_weight agrees too
    assert pr_auc(y, s, w) == pytest.approx(average_precision_score(y, s, sample_weight=w))


def test_row_order_does_not_matter():
    y, s = _data(7)
    p = np.random.default_rng(0).permutation(y.size)
    assert pr_auc(y, s) == pytest.approx(pr_auc(y[p], s[p]), rel=1e-12)
    assert roc_auc(y, s) == pytest.approx(roc_auc(y[p], s[p]), rel=1e-12)


def test_degenerate_cases_return_nan():
    s = np.array([0.1, 0.5, 0.9])
    assert math.isnan(pr_auc(np.zeros(3), s))
    assert math.isnan(roc_auc(np.zeros(3), s))
    assert math.isnan(roc_auc(np.ones(3), s))
    assert math.isnan(pr_auc(np.zeros(0), np.zeros(0)))
    assert math.isnan(best_f1_threshold(np.zeros(3), s))
    # all weight on negatives
    assert math.isnan(pr_auc(np.array([1, 0, 0]), s, w=np.array([0, 1, 1])))

    no_alerts = prf_from_flags(np.array([1, 0, 1]), np.zeros(3, dtype=bool))
    assert math.isnan(no_alerts["precision"])
    # F1 = 2 TP / (alerts + positives) is defined: flagging nothing is the worst F1, not missing.
    assert no_alerts["f1"] == 0.0
    assert no_alerts["recall"] == 0.0 and no_alerts["alerts"] == 0

    no_pos = prf_from_flags(np.zeros(3), np.array([True, False, False]))
    assert math.isnan(no_pos["recall"]) and math.isnan(no_pos["f1"])
    assert no_pos["precision"] == 0.0


def test_nan_scores_raise():
    with pytest.raises(ValueError):
        pr_auc(np.array([0, 1]), np.array([0.1, np.nan]))


def test_count_flags_from_weighted_top_k():
    y = np.array([1, 0, 1, 0])
    counts = np.array([2, 1, 0, 0])  # 2 copies of row 0 and 1 of row 1 flagged
    w = np.array([2, 1, 3, 0])
    m = prf_from_flags(y, counts, w)
    assert (m["alerts"], m["tp"], m["positives"]) == (3, 2, 5)  # positives weighted by w
    assert prf_from_flags(y, counts)["positives"] == 2
    with pytest.raises(ValueError):
        prf_from_flags(y, counts, w=np.array([1, 1, 1, 1]))  # 2 copies of a weight-1 row


@pytest.mark.parametrize("seed", range(4))
def test_best_f1_threshold_is_the_brute_force_argmax(seed):
    y, s = _data(seed)
    thr = best_f1_threshold(y, s)
    cands = np.unique(s)
    f1s = np.array([f1_score(y, s >= t) for t in cands])
    best = f1s.max()
    assert f1_score(y, s >= thr) == pytest.approx(best)
    # ties in F1 -> the highest threshold
    assert thr == pytest.approx(cands[np.flatnonzero(np.isclose(f1s, best))].max())


def test_tie_curve_groups_ties():
    thr, tp, fp = tie_curve(np.array([1, 0, 1, 0]), np.array([0.5, 0.5, 0.2, 0.9]))
    assert thr.tolist() == [0.9, 0.5, 0.2]
    assert tp.tolist() == [0, 1, 2]
    assert fp.tolist() == [1, 2, 2]


def test_argmax_threshold_constant():
    assert ARGMAX_THRESHOLD == 0.5
