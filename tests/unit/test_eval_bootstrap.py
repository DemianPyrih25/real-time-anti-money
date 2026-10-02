"""aml.eval.bootstrap: clusters, reproducible paired replicates, fast stats == direct metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest

from aml.eval.bootstrap import (
    STRATA,
    ReplicateStats,
    ci,
    cluster_strata,
    make_clusters,
    paired_bootstrap,
    replicate_counts,
    replicate_weights,
)
from aml.eval.metrics import pr_auc, prf_from_flags
from aml.eval.operating_points import top_k_flags


def _problem(seed: int = 0, n: int = 3000):
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.04
    attempt = np.where(y & (rng.random(n) < 0.6), rng.integers(0, 15, n), -1)
    src = rng.integers(0, 400, n)
    s = np.round(rng.random(n) * 0.7 + y * rng.random(n) * 0.4, 2)  # ties
    rule = rng.random(n) < 0.03
    return y, attempt, src, s, rule


def test_make_clusters_strata_and_keys():
    y = np.array([1, 1, 1, 0, 0, 1])
    attempt = np.array([5, 5, -1, -1, -1, 7])
    src = np.array([1, 2, 1, 1, 3, 1])
    strata, cidx = make_clusters(y, attempt, src)
    assert strata.tolist() == [0, 0, 1, 2, 2, 0]
    assert STRATA[0] == "pattern_positive"
    # pattern rows cluster by attempt (rows 0 and 1 share attempt 5 despite different senders)
    assert cidx[0] == cidx[1] != cidx[5]
    # the same sender is a different cluster in the "other positive" and "negative" strata
    assert cidx[2] != cidx[3]
    cs = cluster_strata(strata, cidx)
    assert (np.diff(cs) >= 0).all()  # contiguous stratum ranges
    assert cs.size == 5


def test_replicate_counts_per_stratum_and_reproducible():
    y, attempt, src, _, _ = _problem()
    strata, cidx = make_clusters(y, attempt, src)
    cs = cluster_strata(strata, cidx)
    a = np.concatenate(list(replicate_counts(strata, cidx, 50, seed=3, chunk=7)), axis=1)
    b = np.concatenate(list(replicate_counts(strata, cidx, 50, seed=3, chunk=32)), axis=1)
    assert a.shape == (cs.size, 50)
    assert np.array_equal(a, b)  # chunk size does not change the replicates
    for k in range(len(STRATA)):  # each stratum draws as many clusters as it has
        assert (a[cs == k].sum(axis=0) == (cs == k).sum()).all()
    c = np.concatenate(list(replicate_counts(strata, cidx, 50, seed=4)), axis=1)
    assert not np.array_equal(a, c)
    w = np.concatenate(list(replicate_weights(strata, cidx, 50, seed=3, chunk=7)), axis=0)
    assert np.array_equal(w, a[cidx].T.astype(np.int64))


@pytest.mark.parametrize("seed", range(3))
def test_fast_stats_equal_direct_weighted_metrics(seed):
    y, attempt, src, s, rule = _problem(seed)
    strata, cidx = make_clusters(y, attempt, src)
    rs = ReplicateStats(y, cidx)
    flag = s >= 0.6
    k_rules = rs.linear(rule)
    fns = {"ap": rs.pr_auc(s), "prf": rs.prf(flag), "topk": rs.top_k(s, k_rules), "k": k_rules}
    counts = next(replicate_counts(strata, cidx, 20, seed=seed))
    got = {name: fn(counts) for name, fn in fns.items()}
    for r in range(counts.shape[1]):
        w = counts[cidx, r].astype(np.int64)
        assert got["ap"][r] == pytest.approx(pr_auc(y, s, w), rel=1e-10)
        direct = prf_from_flags(y, flag, w)
        for key in ("precision", "recall", "f1"):
            assert got["prf"][key][r] == pytest.approx(direct[key], rel=1e-12, nan_ok=True)
        k = int(got["k"][r])
        assert k == int((w * rule).sum())
        topk = prf_from_flags(y, top_k_flags(s, k, w), w)
        for key in ("precision", "recall", "f1"):
            assert got["topk"][key][r] == pytest.approx(topk[key], rel=1e-12, nan_ok=True)


def test_top_k_extends_its_prefix_when_needed():
    # the K target is far larger than the default prefix of 2K + 1024 unweighted rows
    y, attempt, src, s, _ = _problem(1, n=4000)
    strata, cidx = make_clusters(y, attempt, src)
    rs = ReplicateStats(y, cidx)
    stat = rs.top_k(s, lambda c: np.full(c.shape[1], 3500.0))
    counts = next(replicate_counts(strata, cidx, 4, seed=0))
    got = stat(counts)
    for r in range(4):
        w = counts[cidx, r].astype(np.int64)
        direct = prf_from_flags(y, top_k_flags(s, 3500, w), w)
        assert got["recall"][r] == pytest.approx(direct["recall"])


def test_paired_bootstrap_is_paired_seeded_and_thread_invariant():
    y, attempt, src, s, rule = _problem(2)
    strata, cidx = make_clusters(y, attempt, src)
    rs = ReplicateStats(y, cidx)
    fns = {"m1": rs.pr_auc(s), "m1_again": rs.pr_auc(s), "rules": rs.prf(rule)}
    one = paired_bootstrap(fns, strata, cidx, B=40, seed=9, chunk=8)
    three = paired_bootstrap(fns, strata, cidx, B=40, seed=9, chunk=5, threads=3)
    assert set(one) == {"m1", "m1_again", "rules.precision", "rules.recall", "rules.f1"}
    assert one["m1"].shape == (40,)
    assert np.array_equal(one["m1"], one["m1_again"])  # same replicates for every statistic
    for k in one:
        assert np.array_equal(one[k], three[k], equal_nan=True)
    other = paired_bootstrap(fns, strata, cidx, B=40, seed=10)
    assert not np.array_equal(one["m1"], other["m1"])
    # the bootstrap distribution is centred near the point estimate
    assert np.mean(one["m1"]) == pytest.approx(pr_auc(y, s), abs=0.1)


def test_ci():
    lo, hi = ci(np.arange(1001, dtype=float), 0.9)
    assert (lo, hi) == pytest.approx((50.0, 950.0))
    assert ci(np.array([np.nan, 1.0, np.nan, 3.0]), 1.0) == (1.0, 3.0)
    assert all(math.isnan(x) for x in ci(np.full(5, np.nan)))
