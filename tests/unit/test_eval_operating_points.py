"""aml.eval.operating_points: budgets, ties, deterministic top-K, the (a)/(b)/(c) helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aml.eval.metrics import prf_from_flags
from aml.eval.operating_points import (
    alert_budget,
    alerted_accounts_per_day,
    operating_point_metrics,
    threshold_for_alert_rate,
    top_k_flags,
)


def test_threshold_without_ties_flags_exactly_k():
    s = np.random.default_rng(0).permutation(np.arange(1000) / 1000)
    thr = threshold_for_alert_rate(s, 0.005)
    assert (s >= thr).sum() == 5
    assert thr == pytest.approx(0.995)


def test_threshold_with_ties_never_exceeds_budget():
    # k = 3; the 3rd-highest score (0.5) is tied with 2 more rows -> the block is left out
    s = np.array([0.9, 0.8, 0.5, 0.5, 0.5, 0.1, 0.1, 0.0, 0.0, 0.0])
    thr = threshold_for_alert_rate(s, 0.3)
    assert thr == 0.8
    assert (s >= thr).sum() == 2
    # tie block that ends exactly at k is kept
    assert threshold_for_alert_rate(s, 0.5) == 0.5
    # everything tied -> nothing can be flagged within the budget
    assert threshold_for_alert_rate(np.full(10, 0.3), 0.5) == math.inf


@pytest.mark.parametrize("seed", range(10))
def test_threshold_is_the_smallest_score_within_budget(seed):
    rng = np.random.default_rng(seed)
    s = np.round(rng.random(5000), 2)  # heavy ties
    rate = float(rng.uniform(0.001, 0.05))
    thr = threshold_for_alert_rate(s, rate)
    assert (s >= thr).mean() <= rate + 1e-9
    lower = np.unique(s[s < thr])
    if lower.size:  # the next lower distinct score would break the budget
        assert (s >= lower[-1]).mean() > rate
    # deterministic, and independent of row order
    assert thr == threshold_for_alert_rate(s, rate)
    assert thr == threshold_for_alert_rate(rng.permutation(s), rate)


def test_threshold_edge_cases():
    s = np.array([0.2, 0.4, 0.6])
    assert threshold_for_alert_rate(s, 0.0) == math.inf
    assert threshold_for_alert_rate(s, 1.0) == 0.2
    assert threshold_for_alert_rate(np.zeros(0), 0.5) == math.inf
    assert threshold_for_alert_rate(s, float("nan")) == math.inf
    # a rate that came from a count (A / n) gives exactly A despite float error
    assert alert_budget(1_000_003, 5_017 / 1_000_003) == 5_017


def test_top_k_breaks_ties_by_position():
    s = np.array([0.5, 0.9, 0.5, 0.5, 0.1])
    assert top_k_flags(s, 2).tolist() == [True, True, False, False, False]
    assert top_k_flags(s, 3).tolist() == [True, True, True, False, False]
    assert top_k_flags(s, 0).sum() == 0
    assert top_k_flags(s, 99).all()


@pytest.mark.parametrize("seed", range(5))
def test_weighted_top_k_equals_physical_replication(seed):
    rng = np.random.default_rng(seed)
    n = 300
    s = np.round(rng.random(n), 1)
    y = (rng.random(n) < 0.2).astype(np.int8)
    w = rng.integers(0, 4, n)
    k = int(rng.integers(1, int(w.sum())))
    counts = top_k_flags(s, k, w)
    assert counts.sum() == k
    assert ((counts >= 0) & (counts <= w)).all()
    rep = top_k_flags(np.repeat(s, w), k)
    # copies of row i are adjacent after np.repeat; sum the flags back per original row
    per_row = np.bincount(np.repeat(np.arange(n), w), weights=rep, minlength=n)
    assert np.array_equal(per_row, counts)
    assert prf_from_flags(y, counts, w) == pytest.approx(prf_from_flags(np.repeat(y, w), rep))


def test_alerted_accounts_per_day():
    flags = np.array([True, True, True, False, True])
    src = np.array([7, 7, 8, 9, 7])
    day = np.array([9, 9, 9, 9, 10])
    # (7, 9), (8, 9), (7, 10) -> 3 cases over 2 days
    assert alerted_accounts_per_day(flags, src, day, 2) == 1.5
    assert alerted_accounts_per_day(np.zeros(5, bool), src, day, 2) == 0.0


def test_operating_point_metrics_volumes():
    rng = np.random.default_rng(0)
    n = 2000
    y = (rng.random(n) < 0.05).astype(np.int8)
    s = rng.random(n) + y * 0.3
    rule = rng.random(n) < 0.02
    src = rng.integers(0, 300, n)
    day = rng.integers(9, 11, n)
    thr = float(np.quantile(s, 0.97))
    m = operating_point_metrics(y, s, thr=thr, rule_flags=rule, src=src, day=day, n_days=2)
    assert m["a"]["alerts"] == int((s >= thr).sum())
    assert m["a"]["threshold"] == thr
    assert m["b"]["alerts"] == m["b"]["k"] == int(rule.sum())  # (b): K = the rules' alerts
    union = rule | (s >= thr)
    assert m["c"]["union"]["alerts"] == int(union.sum())
    assert m["c"]["union"]["recall"] == pytest.approx(y[union].sum() / y.sum())
    # the model alone gets exactly the union's volume
    assert m["c"]["model_same_volume"]["alerts"] == int(union.sum())
    assert m["a"]["alerts_per_day"] == m["a"]["alerts"] / 2
