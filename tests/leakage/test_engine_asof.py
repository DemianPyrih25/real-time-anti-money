"""As-of rule for the feature engine (PLAN.md §4, M2 spec §9.2): an event in minute m depends only
on events in minutes <= m - 1.

- Perturbation: changing, deleting and adding events with minute >= m (same-minute peers ranked
  before the target included, and events touching u and v in every way a feature reads them)
  leaves the target's and every earlier row's 79 features, 7 severities, inflow_c and trunc flags
  bit-identical, while later rows do change. Targets: the top row of each feature group, a random
  row and a shared-minute row, in train days and after train (hubs and vocab fixed from the
  unperturbed data, as the offline driver fixes them).
- Labels: the offline build gives the same parts and snapshot digests with every label flipped
  and with no labels file at all.
- Locality: a disconnected component of fresh accounts changes no existing row.
- Within-minute permutation: re-ranking rows inside every minute changes no row (exact and ulp
  classes bit-equal, mean/std classes within tolerance; rows with a trunc flag exempt).
- Prefix = streaming: replaying (events < m) and minute m gives the full run's minute-m rows.
"""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from aml.features.spec import GROUPS, MINUTES_PER_DAY, window_tag
from tests.fixtures.engine_frames import (
    dense_engine_frame,
    features_cfg,
    fixture_build_cfgs,
    fixture_builds,
    fixture_spec,
    make_spec,
    read_fixture,
    require_build,
    require_engine,
    resume_engine,
    row_bits,
    rows_by_id,
    run_engine,
    run_with_day_snapshots,
)
from tests.fixtures.engine_ref import moment_ok, same_bits
from tests.fixtures.perturb import (
    add_disconnected_component,
    permute_within_minutes,
    perturb_events,
)
from tests.fixtures.rules_frames import small_cfg

HUB_CAP = 30  # 9 hubs on the fixture
FRESH = 32  # spare account ids for the locality test
LOOKAHEAD = 8  # minutes after the target that the perturbed replay covers (added events: +0..+7)


@pytest.fixture(scope="module")
def base(prepared, rules_cfg):
    """The unperturbed fixture run: (spec, tx, rows by row_id, day-start snapshots).

    The rows come from one plain uninterrupted run. A second run advances the clock to every day
    start and snapshots there; its rows must be bit-equal (advance(b) then advance(m) is
    advance(m)), and the snapshots let a perturbed replay start at the target's day.
    """
    require_engine()
    spec0, tx = fixture_spec(prepared, rules_cfg, hub_cap=HUB_CAP)
    spec = make_spec(
        tx,
        rules_cfg,
        hub_cap=HUB_CAP,
        n_accounts=spec0.n_accounts + FRESH,
        vocab={k: list(v) for k, v in spec0.vocab.items()},
        hubs=spec0.hubs,
    )
    rows = run_engine(tx, spec)
    rows2, snaps = run_with_day_snapshots(tx, spec)
    assert [row_bits(r) for r in rows2] == [row_bits(r) for r in rows]
    return spec, tx, rows_by_id(tx, rows), snaps


def group_score(row: tuple, idx: list[int]) -> int:
    return sum(1 for k in idx if not math.isnan(row[k]) and row[k] != 0)


def pick_targets(spec, tx: pl.DataFrame, rows: dict, days: tuple[int, int], seed: int):
    """The top row of each engine feature group in `days` (most non-zero features of the group,
    then the lowest row_id), one random row and one row sharing its minute with another row that
    touches the same accounts."""
    cand = tx.filter(pl.col("day").is_between(*days))
    ids = cand["row_id"].to_list()
    picks: list[int] = []
    for g in GROUPS[1:]:
        idx = [spec.feature_index[n] for n in spec.group_names(g)]
        best = max(ids, key=lambda r: (group_score(rows[r], idx), -r))
        assert group_score(rows[best], idx) > 0, g
        picks.append(best)
    rng = np.random.default_rng(seed)
    picks.append(int(rng.choice(ids)))
    shared = cand.filter(
        (pl.len().over(["minute", "src"]) > 1) | (pl.len().over(["minute", "dst"]) > 1)
    )
    if shared.height == 0:
        shared = cand.filter(pl.len().over("minute") > 1)
    picks.append(int(shared["row_id"][int(rng.integers(shared.height))]))
    return cand.filter(pl.col("row_id").is_in(list(dict.fromkeys(picks))))


def check_asof(spec, tx, rows, targets, data_cfg, seed_base, snaps=None):
    """Perturb the future of each target and replay: protected rows stay bit-identical.

    With `snaps` the replay restores the snapshot taken at the start of the target's day (the
    events before it are identical in both streams); otherwise it replays from the first event.
    """
    changed_later = 0
    for k, t in enumerate(targets.iter_rows(named=True)):
        m, rid = t["minute"], t["row_id"]
        tgt = targets.filter(pl.col("row_id") == rid)
        pert = perturb_events(tx, tgt, m, data_cfg, seed=seed_base + k)
        start = 0
        if snaps is None:
            got = rows_by_id(pert, run_engine(pert, spec, until_minute=m + LOOKAHEAD))
        else:
            start = (m // MINUTES_PER_DAY) * MINUTES_PER_DAY
            rest = pert.filter(pl.col("minute") >= start)
            out = resume_engine(snaps[start], spec, rest, until_minute=m + LOOKAHEAD)
            got = rows_by_id(rest, out)
        protected = tx.filter(
            pl.col("minute").is_between(start, m - 1) | (pl.col("row_id") == rid)
        )["row_id"].to_list()
        bad = [r for r in protected if row_bits(got[r]) != row_bits(rows[r])]
        assert not bad, (rid, m, bad[:5])
        later = tx.filter(pl.col("minute").is_between(m + 1, m + LOOKAHEAD))["row_id"].to_list()
        changed_later += sum(1 for r in later if r in got and row_bits(got[r]) != row_bits(rows[r]))
    assert changed_later > 0  # the perturbation is not vacuous


def test_engine_asof_in_train_days(base, data_cfg):
    spec, tx, rows, snaps = base
    lo, hi = data_cfg["split"]["train"]
    targets = pick_targets(spec, tx, rows, (lo, hi), seed=1)
    assert targets.height >= 7
    check_asof(spec, tx, rows, targets, data_cfg, seed_base=200, snaps=snaps)


def test_engine_asof_after_train(base, data_cfg):
    spec, tx, rows, snaps = base
    lo = data_cfg["split"]["val_early"][0]
    targets = pick_targets(spec, tx, rows, (lo, 18), seed=0)
    assert targets.height >= 7
    check_asof(spec, tx, rows, targets, data_cfg, seed_base=100, snaps=snaps)


def test_engine_asof_without_snapshots(base, data_cfg):
    """Two targets replayed from the very first event (no snapshot on the path)."""
    spec, tx, rows, _ = base
    targets = pick_targets(spec, tx, rows, (2, 3), seed=4).head(2)
    check_asof(spec, tx, rows, targets, data_cfg, seed_base=400)


def test_engine_asof_dense_same_minute_ties(rules_cfg, data_cfg):
    """A tie-heavy frame with short windows: the same-minute peers ranked before a target must
    stay invisible to every feature (fan, moments, ports, cycles, scatter-gather)."""
    require_engine()
    frame = dense_engine_frame(seed=11, n=400)
    spec = make_spec(frame, small_cfg(rules_cfg, 10, 5), features_cfg(short=5, long=12, sg=5))
    rows = rows_by_id(frame, run_engine(frame, spec))
    targets = pick_targets(spec, frame, rows, (1, 1), seed=2)
    check_asof(spec, frame, rows, targets, data_cfg, seed_base=300)


# ---------------------------------------------------------------------------------------------
# Labels never enter


def _parts(out_dir) -> dict[str, pl.DataFrame]:
    return {p.name: pl.read_parquet(p) for p in sorted((out_dir / "parts").glob("*.parquet"))}


def _sidecar_digests(out_dir) -> dict[str, str]:
    import json

    return {
        p.name: json.loads(p.read_text(encoding="utf-8"))["state_digest"]
        for p in sorted((out_dir / "snapshots").glob("*.snap.json"))
    }


def frames_bit_equal(a: pl.DataFrame, b: pl.DataFrame) -> bool:
    if a.columns != b.columns or a.schema != b.schema or a.height != b.height:
        return False
    for c in a.columns:
        x, y = a[c].to_numpy(), b[c].to_numpy()
        if x.dtype.kind == "f":
            x, y = x.view(f"u{x.itemsize}"), y.view(f"u{y.itemsize}")
        if not np.array_equal(x, y):
            return False
    return True


def test_labels_never_reach_the_feature_table(prepared, data_cfg, rules_cfg):
    """Builds with the original labels, every label flipped and no labels file at all (fresh
    interpreters, same hash seed) give bit-identical parts and equal snapshot digests."""
    require_build()
    builds = fixture_builds(prepared, fixture_build_cfgs(data_cfg, rules_cfg))
    base = _parts(builds["seed0"])
    assert len(base) == 18
    digests = _sidecar_digests(builds["seed0"])
    assert len(digests) == 2
    for variant in ("flipped", "absent"):
        parts = _parts(builds[variant])
        assert parts.keys() == base.keys(), variant
        for name in base:
            assert frames_bit_equal(parts[name], base[name]), (variant, name)
        assert _sidecar_digests(builds[variant]) == digests, variant


# ---------------------------------------------------------------------------------------------
# Locality, permutation, prefix


def test_disconnected_component_changes_no_existing_row(base, data_cfg):
    spec, tx, rows, _ = base
    first = spec.n_accounts - FRESH
    assert int(max(tx["src"].max(), tx["dst"].max())) < first
    more = add_disconnected_component(
        tx, data_cfg, first_account=first, n_accounts=FRESH, n_events=600, seed=5
    )
    got = rows_by_id(more, run_engine(more, spec))
    bad = [r for r in tx["row_id"].to_list() if row_bits(got[r]) != row_bits(rows[r])]
    assert not bad, bad[:5]
    fresh = more.filter(pl.col("src") >= first)["row_id"].to_list()
    i = spec.feature_index["u_out_cnt_3d"]
    assert any(got[r][i] > 0 for r in fresh)  # the component is not inert


def derived_m2(row: tuple, spec) -> dict[str, float]:
    """m2 of each mean/std-class feature from the row's own mean and std (m2 = std^2 + mean^2):
    only the scale of the tolerance bound, as the golden test does off-platform."""
    out = {}
    for fd in spec.features:
        if fd.tol not in ("mean", "std"):
            continue
        t = window_tag(fd.window)
        mean = row[spec.feature_index[f"{fd.side}_{fd.direction}_mean_{t}"]]
        std = row[spec.feature_index[f"{fd.side}_{fd.direction}_std_{t}"]]
        out[fd.name] = 0.0 if math.isnan(mean) else std * std + mean * mean
    return out


def permutation_mismatches(spec, a: dict, b: dict) -> list:
    """Rows of b differing from a beyond the permutation rule (trunc-flagged rows exempt)."""
    tr = (spec.i_rule_trunc, spec.i_cyc_trunc, spec.i_sg_trunc)
    bad = []
    for rid, ra in a.items():
        rb = b[rid]
        if any(ra[k] or rb[k] for k in tr):
            continue
        m2 = derived_m2(ra, spec)
        for k, fd in enumerate(spec.features):
            ok = (
                moment_ok(fd.tol, rb[k], ra[k], m2[fd.name])
                if fd.tol in ("mean", "std")
                else same_bits(rb[k], ra[k])
            )
            if not ok:
                bad.append((rid, fd.name, ra[k], rb[k]))
        if row_bits(ra[spec.n_features :]) != row_bits(rb[spec.n_features :]):
            bad.append((rid, "tail", ra[spec.n_features :], rb[spec.n_features :]))
    return bad


def test_within_minute_permutation_changes_no_row(base):
    spec, tx, rows, _ = base
    perm = permute_within_minutes(tx, seed=3)
    got = rows_by_id(perm, run_engine(perm, spec))
    bad = permutation_mismatches(spec, rows, got)
    assert not bad, bad[:5]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_within_minute_permutation_dense(rules_cfg, seed):
    require_engine()
    frame = dense_engine_frame(seed=seed, n=400)
    spec = make_spec(frame, small_cfg(rules_cfg, 10, 5), features_cfg(short=5, long=12, sg=5))
    a = rows_by_id(frame, run_engine(frame, spec))
    perm = permute_within_minutes(frame, seed=seed + 10)
    b = rows_by_id(perm, run_engine(perm, spec))
    bad = permutation_mismatches(spec, a, b)
    assert not bad, bad[:5]


def test_prefix_replay_equals_streaming(base, data_cfg):
    spec, tx, rows, _ = base
    multi = tx.filter(pl.len().over("minute") > 1)
    picks = []
    for lo, hi in ((2, 2), tuple(data_cfg["split"]["val_early"]), (9, 10)):
        day = multi.filter(pl.col("day").is_between(lo, hi))
        picks.append(int(day["minute"][day.height // 2]))
    for m in picks:
        prefix = tx.filter(pl.col("minute") <= m)
        got = rows_by_id(prefix, run_engine(prefix, spec))
        ids = prefix.filter(pl.col("minute") == m)["row_id"].to_list()
        assert len(ids) > 1
        assert all(row_bits(got[r]) == row_bits(rows[r]) for r in ids), m


def test_fixture_reader_matches_prepared_order(prepared):
    tx = read_fixture(prepared)
    assert tx["rank"].to_list() == list(range(tx.height))
