"""Look-ahead step 1 moves only the bound (M3 spec §5.2, §5.3).

- `label_times` / `guard_bounds` per protocol and split equal the bounds derived here: causal and
  PNA first_rank - 1; look-ahead train -> train_last, val_early / val_late -> val_last, test ->
  data_last (`end`) or max(d10_last, causal) (`d10`: tail rows keep their causal bound).
- Through the real loader, for every split and both test bounds: every sampled edge <= the
  bound, every copy of the target dropped (parallel transactions of the pair stay), later-than-
  target edges do occur (the bound is really far), the `last` multisets equal the reference
  after the drop, dropped copies and the future share equal the brute-force counts.
- On the fixture's real graph (load_graph): bounds equal the data.yaml day ranges.
"""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from tests.conftest import load_yaml
from tests.fixtures.gnn_graphs import (
    gnn_inputs,
    hand_graph_h1,
    make_host_graph,
    random_tied_graph,
    tied_minute_graph,
)
from tests.fixtures.gnn_ref import (
    LOADER_PARTS,
    SPLIT_NAMES,
    RefSampler,
    asof_problems,
    day_splits,
    engine_batches,
    flat_observed,
    ref_label_times,
    ref_problems,
    require_gnn,
    split_by_days,
)

FAN = [25, 10]
# Ends of days 1..17 (fractions of the minute range): train 50%, val_early 10%, val_late 10%,
# days 9-10 10%, tail days 11-18 20%.
DAY_CUTS = [0.083, 0.167, 0.25, 0.333, 0.417, 0.5, 0.6, 0.7, 0.75, 0.8]
DAY_CUTS += [0.825, 0.85, 0.875, 0.9, 0.925, 0.95, 0.975]
PRIMARY_LAST_DAY = 10


@pytest.fixture(scope="module")
def la_graph():
    g = random_tied_graph(n_nodes=600, n_edges=3000, per_minute=10, seed=5)
    day = split_by_days(g.minute, DAY_CUTS)
    hg = make_host_graph(g, split=day_splits(day), day=day, primary_last_day=PRIMARY_LAST_DAY)
    return g, hg


def _ref(hg, gids, protocol, test_bound="end"):
    return ref_label_times(
        gids,
        minute=hg.minute,
        split_code=hg.split_code,
        bounds=hg.bounds,
        protocol=protocol,
        test_bound=test_bound,
    )


def test_fixture_bounds_are_the_split_ends(la_graph) -> None:
    g, hg = la_graph
    code = hg.split_code
    b = hg.bounds
    assert b["train_last"] == np.flatnonzero(code == 0)[-1]
    assert b["val_last"] == np.flatnonzero(code == 2)[-1]
    assert b["data_last"] == g.n_edges - 1
    assert b["d10_last"] == np.flatnonzero(hg.day <= PRIMARY_LAST_DAY)[-1] < b["data_last"]
    assert b["val_early_first"] == np.flatnonzero(code == 1)[0]
    assert b["test_first"] == np.flatnonzero(code == 3)[0]


@pytest.mark.parametrize("protocol", ["causal", "pna", "lookahead"])
@pytest.mark.parametrize("test_bound", ["end", "d10"])
def test_label_times_and_guard_bounds_equal_the_derived_bounds(
    la_graph, protocol, test_bound
) -> None:
    require_gnn("graph.label_times", "graph.guard_bounds", torch=False)
    from aml.models.gnn.graph import guard_bounds, label_times

    g, hg = la_graph
    for code, split in enumerate(SPLIT_NAMES):
        gids = np.flatnonzero(hg.split_code == code)
        want = _ref(hg, gids, protocol, test_bound)
        got = label_times(hg, gids, protocol=protocol, test_bound=test_bound)
        assert np.asarray(got).dtype == np.int64, split
        np.testing.assert_array_equal(got, want, err_msg=f"label_times {split}")
        got = guard_bounds(hg, gids, protocol=protocol, test_bound=test_bound)
        np.testing.assert_array_equal(np.asarray(got, np.int64), want, err_msg=f"guard {split}")


def test_lookahead_bounds_explicitly(la_graph) -> None:
    """The §5.2 table, stated value by value (not through the reference function)."""
    require_gnn("graph.label_times", torch=False)
    from aml.models.gnn.graph import label_times

    g, hg = la_graph
    b = hg.bounds

    def lt(split: str, tb: str = "end") -> np.ndarray:
        gids = np.flatnonzero(hg.split_code == SPLIT_NAMES.index(split))
        return np.asarray(label_times(hg, gids, protocol="lookahead", test_bound=tb))

    assert set(lt("train").tolist()) == {b["train_last"]}
    assert set(lt("val_early").tolist()) == {b["val_last"]}
    assert set(lt("val_late").tolist()) == {b["val_last"]}
    assert set(lt("test").tolist()) == {b["data_last"]}
    test = np.flatnonzero(hg.split_code == 3)
    d10 = lt("test", "d10")
    primary = hg.day[test] <= PRIMARY_LAST_DAY
    assert primary.any() and (~primary).any()
    assert set(d10[primary].tolist()) == {b["d10_last"]}
    tail = test[~primary]
    np.testing.assert_array_equal(d10[~primary], g.first_rank[tail] - 1)  # causal bound
    # the first tail minute starts right after day 10: its causal bound is d10_last itself
    assert (d10[~primary] >= b["d10_last"]).all() and d10[~primary].max() > b["d10_last"]
    # causal: first rank of the minute - 1, -1 for the first minute
    causal = np.asarray(label_times(hg, np.arange(g.n_edges), protocol="causal"))
    np.testing.assert_array_equal(causal, g.first_rank - 1)
    assert causal[0] == -1


LOADER_CASES = [
    ("train", "end"),
    ("val_early", "end"),
    ("val_late", "end"),
    ("test", "end"),
    ("test", "d10"),
]


@pytest.mark.parametrize(("split", "test_bound"), LOADER_CASES)
def test_lookahead_subgraphs_respect_their_bound(la_graph, split, test_bound) -> None:
    require_gnn(*LOADER_PARTS)
    g, hg = la_graph
    seeds = np.flatnonzero(hg.split_code == SPLIT_NAMES.index(split))
    bounds = _ref(hg, seeds, "lookahead", test_bound)
    obs, guard = flat_observed(
        hg,
        engine_batches(
            hg, seeds, protocol="lookahead", fanout=FAN, batch_size=1024, test_bound=test_bound
        ),
    )
    assert [o.gid for o in obs] == seeds.tolist()
    ref = RefSampler(g.src, g.dst, g.n_nodes, FAN)
    refs = [ref.subgraph(s, b) for s, b in zip(seeds, bounds, strict=True)]
    bad = []
    for o, r, bnd in zip(obs, refs, bounds, strict=True):
        # tail rows under d10 carry the causal bound: they must also pass the causal check
        causal = bool(bnd == g.first_rank[o.gid] - 1)
        bad += asof_problems(o, bound=int(bnd), minute=g.minute, causal=causal)
        bad += ref_problems(o, r, drop_target=True)
    assert not bad, bad[:5]

    later = sum(int((o.all_gids() > o.gid).sum()) for o in obs)
    assert later > 0  # non-vacuity: the far bound really admits edges after the target
    copies = sum(r.target_copies(et) for r in refs for et in ("to", "rev_to"))
    assert copies > 0 and guard["dropped_target_copies"] == copies
    assert guard["violations"] == 0
    assert guard["target_hits"] == 0  # after the drop
    # future share (§5.3) = sampled edges with rank >= the target minute's first rank / all
    # sampled edges, the dropped copies included; brute force from the reference and from the
    # kept edges actually observed (+ the copies, whose rank is >= that first rank)
    first = g.first_rank
    kept = sum(int(len(o.all_gids())) for o in obs)
    kept_future = sum(int((o.all_gids() >= first[o.gid]).sum()) for o in obs)
    brute = sum(r.future_count(int(first[r.gid])) for r in refs)
    assert guard["edges_checked"] == sum(r.n_edges() for r in refs) == kept + copies
    assert guard["future_edges"] == brute == kept_future + copies
    assert 0 < guard["future_edges"] / guard["edges_checked"] < 1


def test_parallel_and_reverse_edges_of_the_target_stay() -> None:
    """Tied graph, look-ahead to the end: T1 = c->a (rank 7) is dropped from its own subgraph,
    its parallel c->a (rank 10) and its reverse a->c (rank 8), both later, stay."""
    require_gnn(*LOADER_PARTS)
    g = tied_minute_graph()
    hg = make_host_graph(g)
    obs, guard = flat_observed(
        hg,
        engine_batches(
            hg, np.arange(g.n_edges), protocol="lookahead", fanout=[5, 5], batch_size=64
        ),
    )
    t1 = obs[g.pinned["T1"]["gid"]].all_gids().tolist()
    assert 7 not in t1 and 10 in t1 and 8 in t1
    assert guard["violations"] == 0 and guard["dropped_target_copies"] > 0


def test_h1_lookahead_bound_is_end_of_train() -> None:
    require_gnn("graph.label_times", torch=False)
    from aml.models.gnn.graph import label_times

    g = hand_graph_h1()
    hg = make_host_graph(g)
    got = label_times(hg, np.arange(g.n_edges), protocol="lookahead")
    assert set(np.asarray(got).tolist()) == {g.pinned["A"]["lookahead_bound"]}


# --- the fixture's real graph ---------------------------------------------------------------


@pytest.fixture(scope="module")
def real_graph(prepared, data_cfg, rules_cfg, tmp_path_factory):
    require_gnn("graph.load_graph", torch=False)
    from aml.models.gnn import LABEL_SPLITS
    from aml.models.gnn.graph import load_graph

    paths, features_dir, gnn_cfg = gnn_inputs(
        prepared,
        tmp_path_factory.mktemp("gnn_la"),
        {"data": data_cfg, "rules": rules_cfg, "features": load_yaml("features.yaml")},
    )
    g = load_graph(paths, features_dir, gnn_cfg, data_cfg=data_cfg, label_splits=LABEL_SPLITS)
    tx = pl.read_parquet(paths.transactions, columns=["rank", "day", "split", "minute"])
    return g, tx


def test_real_graph_bounds_equal_the_data_cfg_day_ranges(real_graph, data_cfg) -> None:
    g, tx = real_graph
    day = tx["day"].to_numpy()
    rank = tx["rank"].to_numpy()
    sp = data_cfg["split"]
    want = {
        "train_last": int(rank[day <= sp["train"][1]].max()),
        "val_last": int(rank[day <= sp["val_late"][1]].max()),
        "d10_last": int(rank[day <= data_cfg["test_views"]["primary"][1]].max()),
        "data_last": int(rank.max()),
        "val_early_first": int(rank[day >= sp["val_early"][0]].min()),
        "val_late_first": int(rank[day >= sp["val_late"][0]].min()),
        "test_first": int(rank[day >= sp["test"][0]].min()),
    }
    assert {k: int(v) for k, v in g.bounds.items()} == want
    assert want["d10_last"] < want["data_last"]  # the fixture has tail days
    np.testing.assert_array_equal(g.minute, tx["minute"].to_numpy())


@pytest.mark.parametrize("test_bound", ["end", "d10"])
def test_real_graph_label_times(real_graph, test_bound) -> None:
    require_gnn("graph.label_times", torch=False)
    from aml.models.gnn.graph import label_times

    g, _ = real_graph
    gids = np.arange(g.n_edges)
    for protocol in ("causal", "lookahead"):
        got = label_times(g, gids, protocol=protocol, test_bound=test_bound)
        np.testing.assert_array_equal(got, _ref(g, gids, protocol, test_bound), err_msg=protocol)
