"""Path search on the ring: round trip, temporal cycles 2-4, scatter-gather, neighbours (M2 §4.7).

Contract between the engine (`Engine._score`) and this module
---------------------------------------------------------------
- Inputs are the engine's own objects, passed explicitly (no engine import here):
  `ring` (`spec.RingView`: columns, `base`, chains), `head_in` / `head_out` (array('i') by
  account, newest applied rank, -1 none), `hub` (bytearray by account, 1 = hub, from spec.hubs),
  `spec` (EngineSpec: w_rt, hop, w_sg, rule_visits, feat_visits, cap_count) and `clock` (the
  minute being scored; every ring row has minute <= clock - 1).
- The memo lives on the engine as `engine.memo = MinuteMemo()` (a TRANSIENT field, never in the
  snapshot or the state digest) and is cleared by `Engine.advance` via `memo.clear()`. Within one
  minute the ring does not change, so an entry built for the first event of an account serves
  every later same-minute event of it (hub bursts of up to 11,193 events/minute). Entries are
  built lazily by `path_counts` / `sg_counts` on a miss; `Engine.score` may fill it too.
- Every search walks in-chains only (`head_in`, `prev_in`), newest first; no search ever walks an
  out-chain, so a hub's out-edges are never expanded. All search walks skip self-loop rows
  (src == dst). Walks stop at `r < ring.base` or at the first row older than the window bound.
  Budgets count steps in walk order, so any truncation point is deterministic.
- Self-loop targets (u == v): `path_counts` -> (0, 0, 0, 0, 0), `sg_counts` -> (0, 0, 0), and no
  memo entry is built for them.

Engine side, per target u -> v (after the TX/window features are read):
    args = (memo, ring, head_in, hub, spec, clock, u, v)
    c2, c3, c4, rule_trunc, cyc_trunc = cycles.path_counts(*args)
    sg_mids, sg_srcs, sg_trunc = cycles.sg_counts(*args)
The engine caps cyc features with `min(c, spec.cap_count)` and feeds raw c2, c3 to the
round-trip rule (`scenarios.RuleSupport`), which caps c2 + c3 at `max_round_trip_paths`.

The path memo (§4.7) counts exactly the paths of the spec's pseudo-code, but groups its walks:
each distinct intermediate's in-chain is walked once per memo (not once per parallel edge), and
the per-edge hop conditions are applied with two pointers over the parallel edges' minutes.
Steps (the budget unit) are chain entries examined inside the window bound of their walk:
`rule_visits` bounds the level-1 walk (in-edges w -> u) plus the level-2 walks (in-edges x -> w,
one walk per distinct non-hub w); `feat_visits` bounds the level-3 walks (in-edges q -> x, one
walk per distinct x). On HI-Small the rule walk needs at most indeg(u) + sum of indeg(w) <=
1,084 + 1,084^2 = 1,176,140 < 2,000,000 steps, so `rule_trunc` cannot fire there.
"""

from __future__ import annotations

import operator
from array import array
from typing import NamedTuple

from aml.features.spec import EngineSpec, RingView


class PathMemo(NamedTuple):
    """Backward path counts into one account u for the current minute (§4.7, `lo = clock - W_rt`).

    d1[w] = edges w -> u (w != u) in [lo, clock - 1]: the 2-hop closers (c2 = d1[v]).
    d2[x] = paths x -> w -> u with w not a hub, x not in {w, u}, t3 - H <= t2 <= t3, t2 >= lo
            (c3 = d2[v]).
    d3[q] = paths q -> x -> w -> u with x and w not hubs, u, x, w, q pairwise distinct,
            t2 - H <= t1 <= t2, t1 >= lo (c4 = d3[v]); a lower bound when cyc_trunc = 1.
    Every path counts once per combination of parallel edges. Keys that no target can read
    (x == u in d2, q == u in d3) are never stored.
    rule_trunc = 1 if the D1/D2 walk needed more than spec.rule_visits steps (everything stopped;
    d1/d2 are lower bounds, d3 is not built); cyc_trunc = 1 if the D3 walk needed more than
    spec.feat_visits steps (D3 stopped), or rule_trunc.
    """

    d1: dict[int, int]
    d2: dict[int, int]
    d3: dict[int, int]
    rule_trunc: int
    cyc_trunc: int


class EvMemo(NamedTuple):
    """Two-hop in-neighbourhood of a receiver v for the SG window [clock - W_sg, clock - 1].

    mids: x -> frozenset of senders s of x (s != x, s not a hub), for every x with an edge x -> v
    in the window (x != v), in chain order of first appearance (insertion order is the
    deterministic iteration order). trunc = 1 if building it exceeded spec.feat_visits (partial).
    """

    mids: dict[int, frozenset[int]]
    trunc: int


class MinuteMemo:
    """The per-minute memo (`engine.memo`): path memos and SG sets keyed by account id."""

    __slots__ = ("ev", "path", "su")

    def __init__(self) -> None:
        self.path: dict[int, PathMemo] = {}  # by sender u
        self.su: dict[int, frozenset[int]] = {}  # by sender u: Su(u)
        self.ev: dict[int, EvMemo] = {}  # by receiver v: Ev(v)

    def clear(self) -> None:
        self.path.clear()
        self.su.clear()
        self.ev.clear()

    def __len__(self) -> int:
        return len(self.path) + len(self.su) + len(self.ev)


# --- round trip and cycles ----------------------------------------------------------------------


def build_path_memo(
    ring: RingView, head_in: array, hub: bytearray, spec: EngineSpec, clock: int, u: int
) -> PathMemo:
    """D1/D2/D3 for account u (§4.7); budgets: spec.rule_visits counts the D1 and D2 walk steps,
    spec.feat_visits the D3 walk steps (see the module docstring)."""
    base = ring.base
    minute = ring.minute
    src = ring.src
    prev_in = ring.prev_in
    lo = clock - spec.w_rt
    hop = spec.hop
    rule_budget = spec.rule_visits

    # Level 1: in-edges (w -> u, t3), newest first. t3s[w] = the minutes of the edges w -> u
    # (descending) for the non-hub intermediates w.
    d1: dict[int, int] = {}
    t3s: dict[int, list[int]] = {}
    rv = 0
    rule_trunc = 0
    r = head_in[u]
    while r >= base:
        i = r - base
        t3 = minute[i]
        if t3 < lo:
            break
        rv += 1
        if rv > rule_budget:
            rule_trunc = 1
            break
        w = src[i]
        if w != u:
            d1[w] = d1.get(w, 0) + 1
            if not hub[w]:
                ts = t3s.get(w)
                if ts is None:
                    t3s[w] = [t3]
                else:
                    ts.append(t3)
        r = prev_in[i]

    # Level 2: for each intermediate w, its in-edges (x -> w, t2) with t2 <= t3 <= t2 + H for
    # k >= 1 of the edges w -> u; one walk per w. ext[x] collects (t2, w, k) for level 3.
    d2: dict[int, int] = {}
    ext: dict[int, list[tuple[int, int, int]]] = {}
    if not rule_trunc:
        for w, ts in t3s.items():
            n = len(ts)
            t_hi = ts[0]
            bound = ts[-1] - hop
            if bound < lo:
                bound = lo
            a = b = 0  # ts[a:b] = the t3 in [t2, t2 + H] (ts is descending)
            r = head_in[w]
            while r >= base:
                i = r - base
                t2 = minute[i]
                if t2 < bound:
                    break
                rv += 1
                if rv > rule_budget:
                    rule_trunc = 1
                    break
                r = prev_in[i]
                if t2 > t_hi:
                    continue
                x = src[i]
                # A self-loop of w, or x == u (closes no path to any target). Hot loop: two
                # compares are faster than `x in (w, u)`.
                if x == w or x == u:  # noqa: SIM109
                    continue
                if n == 1:
                    k = 1  # bound <= t2 <= t_hi: the one edge w -> u is within the hop
                else:
                    lim = t2 + hop
                    while a < n and ts[a] > lim:
                        a += 1
                    while b < n and ts[b] >= t2:
                        b += 1
                    k = b - a
                    if not k:
                        continue
                d2[x] = d2.get(x, 0) + k
                if not hub[x]:
                    e = ext.get(x)
                    if e is None:
                        ext[x] = [(t2, w, k)]
                    else:
                        e.append((t2, w, k))
            if rule_trunc:
                break

    d3: dict[int, int] = {}
    cyc_trunc = rule_trunc
    if not rule_trunc:
        cyc_trunc = _level3(ring, head_in, ext, d3, lo, hop, u, spec.feat_visits)
    return PathMemo(d1, d2, d3, rule_trunc, cyc_trunc)


def _level3(
    ring: RingView,
    head_in: array,
    ext: dict[int, list[tuple[int, int, int]]],
    d3: dict[int, int],
    lo: int,
    hop: int,
    u: int,
    budget: int,
) -> int:
    """Fill d3 from the level-2 paths ending at each non-hub x; returns 1 if the budget stopped it.

    For an in-edge (q -> x, t1) every level-2 path (x -> w at t2, weight k) with
    t1 <= t2 <= t1 + H and w != q extends to k paths q -> x -> w -> u.
    """
    base = ring.base
    minute = ring.minute
    src = ring.src
    prev_in = ring.prev_in
    fv = 0
    for x, ents in ext.items():
        r = head_in[x]
        if len(ents) == 1:
            t2, w, k = ents[0]
            bound = t2 - hop
            if bound < lo:
                bound = lo
            while r >= base:
                i = r - base
                t1 = minute[i]
                if t1 < bound:
                    break
                fv += 1
                if fv > budget:
                    return 1
                if t1 <= t2:
                    q = src[i]
                    if q != x and q != w and q != u:
                        d3[q] = d3.get(q, 0) + k
                r = prev_in[i]
            continue
        # Several level-2 paths through x: a sliding window over them, by t2 descending.
        ents.sort(reverse=True)
        n = len(ents)
        bound = ents[-1][0] - hop
        if bound < lo:
            bound = lo
        a = b = 0  # ents[a:b] = the paths with t1 <= t2 <= t1 + H
        total = 0
        per_w: dict[int, int] = {}
        while r >= base:
            i = r - base
            t1 = minute[i]
            if t1 < bound:
                break
            fv += 1
            if fv > budget:
                return 1
            r = prev_in[i]
            while b < n and ents[b][0] >= t1:
                _, w, k = ents[b]
                total += k
                per_w[w] = per_w.get(w, 0) + k
                b += 1
            lim = t1 + hop
            while a < n and ents[a][0] > lim:  # a <= b: these were all added above
                _, w, k = ents[a]
                total -= k
                per_w[w] -= k
                a += 1
            if not total:
                continue
            q = src[i]
            if q == x or q == u:  # noqa: SIM109 - hot loop, see level 2
                continue
            k = total - per_w.get(q, 0)
            if k:
                d3[q] = d3.get(q, 0) + k
    return 0


def path_counts(
    memo: MinuteMemo,
    ring: RingView,
    head_in: array,
    hub: bytearray,
    spec: EngineSpec,
    clock: int,
    u: int,
    v: int,
) -> tuple[int, int, int, int, int]:
    """(c2, c3, c4, rule_trunc, cyc_trunc) for the target u -> v, raw (uncapped) counts.

    Builds `memo.path[u]` on a miss. u == v -> (0, 0, 0, 0, 0).
    """
    if u == v:
        return (0, 0, 0, 0, 0)
    pm = memo.path.get(u)
    if pm is None:
        pm = build_path_memo(ring, head_in, hub, spec, clock, u)
        memo.path[u] = pm
    return (pm.d1.get(v, 0), pm.d2.get(v, 0), pm.d3.get(v, 0), pm.rule_trunc, pm.cyc_trunc)


# --- scatter-gather -----------------------------------------------------------------------------


def build_su(
    ring: RingView, head_in: array, hub: bytearray, spec: EngineSpec, clock: int, u: int
) -> frozenset[int]:
    """Su(u): accounts w with an edge w -> u in [clock - W_sg, clock - 1], w != u, w not a hub."""
    base = ring.base
    minute = ring.minute
    src = ring.src
    prev_in = ring.prev_in
    lo = clock - spec.w_sg
    out: set[int] = set()
    r = head_in[u]
    while r >= base:
        i = r - base
        if minute[i] < lo:
            break
        w = src[i]
        if w != u and not hub[w]:
            out.add(w)
        r = prev_in[i]
    return frozenset(out)


def build_ev(
    ring: RingView, head_in: array, hub: bytearray, spec: EngineSpec, clock: int, v: int
) -> EvMemo:
    """Ev(v) (see EvMemo); every chain step (both levels) counts against spec.feat_visits."""
    base = ring.base
    minute = ring.minute
    src = ring.src
    prev_in = ring.prev_in
    lo = clock - spec.w_sg
    budget = spec.feat_visits
    steps = 0
    mids: dict[int, frozenset[int]] = {}
    r = head_in[v]
    while r >= base:
        i = r - base
        if minute[i] < lo:
            break
        steps += 1
        if steps > budget:
            return EvMemo(mids, 1)
        x = src[i]
        r = prev_in[i]
        if x == v or x in mids:
            continue
        senders: set[int] = set()
        r2 = head_in[x]
        while r2 >= base:
            j = r2 - base
            if minute[j] < lo:
                break
            steps += 1
            if steps > budget:
                mids[x] = frozenset(senders)  # partial: still a lower bound downstream
                return EvMemo(mids, 1)
            s = src[j]
            if s != x and not hub[s]:
                senders.add(s)
            r2 = prev_in[j]
        mids[x] = frozenset(senders)
    return EvMemo(mids, 0)


def sg_counts(
    memo: MinuteMemo,
    ring: RingView,
    head_in: array,
    hub: bytearray,
    spec: EngineSpec,
    clock: int,
    u: int,
    v: int,
) -> tuple[int, int, int]:
    """(sg_mids, sg_srcs, sg_trunc) for the target u -> v, capped at spec.cap_count (§4.7).

    Builds `memo.su[u]` and `memo.ev[v]` on a miss, then (u != v)::

        mids = 0; srcs = set(); work = 0; trunc = Ev.trunc
        for x, sx in Ev.mids.items():
            if x == u: continue
            work += 1 + min(len(sx), len(Su))
            if work > spec.feat_visits: trunc = 1; break
            common = (sx & Su) - {v}
            if common: mids += 1; srcs |= common

    sg_mids = siblings x of u (x not in {u, v}, x -> v in the window) sharing a non-hub source
    s not in {u, v, x} with u (s -> u and s -> x in the window); sg_srcs = distinct such s.
    u == v -> (0, 0, 0). An empty Su(u) gives exactly (0, 0, 0) without building Ev(v): no
    source can exist, so nothing is truncated.
    """
    if u == v:
        return (0, 0, 0)
    su = memo.su.get(u)
    if su is None:
        su = build_su(ring, head_in, hub, spec, clock, u)
        memo.su[u] = su
    if not su:
        return (0, 0, 0)
    ev = memo.ev.get(v)
    if ev is None:
        ev = build_ev(ring, head_in, hub, spec, clock, v)
        memo.ev[v] = ev
    budget = spec.feat_visits
    n_su = len(su)
    mids = 0
    srcs: set[int] = set()
    work = 0
    trunc = ev.trunc
    for x, sx in ev.mids.items():
        if x == u:
            continue
        n = len(sx)
        work += 1 + (n if n < n_su else n_su)
        if work > budget:
            trunc = 1
            break
        if not n:
            continue
        common = sx & su
        if common and v in common:
            common = common - {v}
        if common:
            mids += 1
            srcs |= common
    cap = spec.cap_count
    n_srcs = len(srcs)
    return (mids if mids < cap else cap, n_srcs if n_srcs < cap else cap, trunc)


# --- M6 read API --------------------------------------------------------------------------------


def neighbours(
    ring: RingView,
    head_in: array,
    head_out: array,
    spec: EngineSpec,
    clock: int | None,
    acct: int,
    direction: str,
    window: int,
    cap: int,
) -> list[tuple[int, int, int, int]]:
    """M6 read API (`Engine.neighbours` delegates here): [(counterparty, minute, rank, usd_c)],
    newest first, at most `cap` entries.

    direction "in" walks head_in/prev_in (counterparty = src), "out" walks head_out/prev_out
    (counterparty = dst; capped at read time, so a hub's out-chain is cut at `cap`). Applied events
    with minute in [clock - window, clock - 1]; self-loops are included (they are real events).
    ValueError for another direction, an account id outside [0, n_accounts), cap < 0, or window
    outside [1, spec.W_max] (older rows may be compacted away, which would make the answer depend
    on compaction timing). clock None (nothing applied yet) -> [].
    """
    if direction == "in":
        head, prev, other = head_in, ring.prev_in, ring.src
    elif direction == "out":
        head, prev, other = head_out, ring.prev_out, ring.dst
    else:
        raise ValueError(f"direction must be 'in' or 'out', got {direction!r}")
    acct, window, cap = _index("acct", acct), _index("window", window), _index("cap", cap)
    if not 0 <= acct < len(head):
        raise ValueError(f"account id {acct} outside [0, {len(head)})")
    if not 1 <= window <= spec.W_max:
        raise ValueError(f"window must be in [1, W_max={spec.W_max}], got {window}")
    if cap < 0:
        raise ValueError(f"cap must be >= 0, got {cap}")
    if clock is None or cap == 0:
        return []
    base = ring.base
    minute = ring.minute
    usd_c = ring.usd_c
    lo = clock - window
    out: list[tuple[int, int, int, int]] = []
    r = head[acct]
    while r >= base:
        i = r - base
        t = minute[i]
        if t < lo:
            break
        out.append((other[i], t, r, usd_c[i]))
        if len(out) >= cap:
            break
        r = prev[i]
    return out


def _index(name: str, value: object) -> int:
    """An integer argument (numpy integers accepted, bool refused) as a Python int."""
    if not isinstance(value, bool):
        try:
            return operator.index(value)  # type: ignore[arg-type]
        except TypeError:
            pass
    raise ValueError(f"{name} must be an integer, got {value!r}")
