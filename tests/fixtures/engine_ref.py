"""Independent brute-force reference of the M2 feature engine (M2 spec §9.1).

Written only from the M2 spec §4-§6 (and M1's scenarios.sql for the rules), never from the engine
code. For one event e = u -> v at minute m it recomputes every model input, the 7 rule severities
and inflow_c from the event list by literal filters, with no state carried between events:

- windows: the events with minute in [m - W, m - 1] (filtered from each account's full list);
- "applied" history: the events with minute <= m - 1 (same-minute peers are never visible);
- cycles cyc2-4 and scatter-gather: explicit enumeration of the edges / paths in the window;
- ports: the definition (receivers of u strictly before the pair's first minute), cross-checked by
  a separate minute-by-minute simulation in `Reference.rows()`;
- moments: two-pass (fsum mean, then fsum of squared deviations).

Feature names are generated here from the §5 naming rule and must equal the spec's names. Budgets
are not simulated: the trunc flags are 0 and every count is the exact (uncapped-by-budget) value;
`compare_rows` lets a truncated engine row be <= the reference in the affected columns only.
"""

from __future__ import annotations

import math
import re
import struct
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import polars as pl

from aml.features.spec import INPUT_COLUMNS, TOL_VAR, EngineSpec, tol_ok
from aml.rules.sql_baseline import HUB_SEGMENTABLE, SCENARIOS

NAN = float("nan")
MINUTES_PER_DAY = 1440
TX_NAMES = (
    "log_amount_usd",
    "payment_currency",
    "receiving_currency",
    "cross_currency",
    "payment_format",
    "self_loop",
    "same_bank",
    "round_amount",
    "hour_of_day",
)
# Columns a truncated engine row may under-count (<= the reference), by the flag that allows it.
RULE_TRUNC_STATS = ("cyc2", "cyc3", "cyc4")
CYC_TRUNC_STATS = ("cyc4",)
SG_TRUNC_STATS = ("sg_mids", "sg_srcs")


def ref_cents(x: float) -> int:
    """Whole cents of a non-negative finite amount, half away from zero on the double x * 100
    (M2 spec §4.1; equal to DuckDB's CAST(round(x * 100) AS BIGINT))."""
    x = float(x)
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"amount must be finite and >= 0, got {x!r}")
    y = x * 100.0
    f = math.floor(y)
    return int(f) + (1 if y - f >= 0.5 else 0)


def ref_window_tag(minutes: int) -> str:
    """§5: W // 1440 "d" if whole days, else W // 60 "h" if whole hours, else W "m"."""
    w = int(minutes)
    if w % MINUTES_PER_DAY == 0:
        return f"{w // MINUTES_PER_DAY}d"
    if w % 60 == 0:
        return f"{w // 60}h"
    return f"{w}m"


def ref_slug(fmt: str) -> str:
    """§5: lower-case, every non-alphanumeric character -> "_"."""
    return re.sub(r"[^0-9a-z]", "_", fmt.lower())


def _moments(ls: list[float]) -> tuple[float, float, float]:
    """(mean, population std, mean of squares) by two passes; NaN, NaN, 0.0 when empty."""
    n = len(ls)
    if n == 0:
        return NAN, NAN, 0.0
    mean = math.fsum(ls) / n
    var = math.fsum((x - mean) * (x - mean) for x in ls) / n
    return mean, math.sqrt(var), math.fsum(x * x for x in ls) / n


class Reference:
    """Brute-force features of an event list (rank order). Build, then `row(i)` / `rows()`.

    `extend` appends events (mappings with the INPUT_COLUMNS keys, or tuples in that order); a
    row depends only on events of earlier minutes, so a growing list gives stable rows.
    """

    def __init__(self, spec: EngineSpec, events: Iterable[Any] = ()) -> None:
        self.spec = spec
        P = spec.sql_params
        self.P = P
        self.S, self.L, self.W_sg = spec.w_short, spec.w_long, spec.w_sg
        self.W_pt = int(P["pass_through_window"])
        self.W_rt = int(P["round_trip_window_plus_1"]) - 1
        self.H = int(P["hop_window"])
        self.band = (float(P["band_low_usd"]), float(P["band_high_usd"]))
        self.round_cents = int(P["round_cents"])
        self.hubs = frozenset(int(h) for h in spec.hubs)
        self.hr_formats = frozenset(spec.high_risk_formats)
        self.excl = {s: bool(P[param]) for s, param in HUB_SEGMENTABLE.items()}
        self.codes = {col: {x: k for k, x in enumerate(vals)} for col, vals in spec.vocab.items()}
        self.formats = tuple(spec.vocab["payment_format"])
        # raw and derived columns, by event index (= position in rank order)
        self.rank: list[int] = []
        self.row_id: list[int] = []
        self.minute: list[int] = []
        self.src: list[int] = []
        self.dst: list[int] = []
        self.usd: list[float] = []
        self.c: list[int] = []
        self.l: list[float] = []  # noqa: E741 - the spec's name
        self.pc: list[int] = []
        self.fmt_s: list[str] = []
        self.fmt: list[int] = []
        self.pcur: list[int] = []
        self.rcur: list[int] = []
        self.cross: list[bool] = []
        self.same_bank: list[bool] = []
        self.in_band: list[bool] = []
        self.is_round: list[bool] = []
        self.hr: list[bool] = []
        self.out_of: dict[int, list[int]] = defaultdict(list)
        self.in_of: dict[int, list[int]] = defaultdict(list)
        self.pair_of: dict[tuple[int, int], list[int]] = defaultdict(list)
        self.extend(events)

    # --- input ----------------------------------------------------------------------------------

    @classmethod
    def from_frame(cls, frame: pl.DataFrame, spec: EngineSpec) -> Reference:
        cols = [frame.get_column(c).to_list() for c in INPUT_COLUMNS]
        return cls(spec, zip(*cols, strict=True))

    def extend(self, events: Iterable[Any]) -> None:
        for ev in events:
            self.append(ev)

    def append(self, ev: Any) -> int:
        if isinstance(ev, Mapping):
            f = {k: ev[k] for k in INPUT_COLUMNS}
        else:
            f = dict(zip(INPUT_COLUMNS, ev, strict=True))
        i = len(self.rank)
        m = int(f["minute"])
        if i and m < self.minute[-1]:
            raise ValueError("events must be in rank order with non-decreasing minutes")
        u, v = int(f["src"]), int(f["dst"])
        a, paid = float(f["amount_usd"]), float(f["amount_paid"])
        self.rank.append(int(f["rank"]))
        self.row_id.append(int(f["row_id"]))
        self.minute.append(m)
        self.src.append(u)
        self.dst.append(v)
        self.usd.append(a)
        self.c.append(ref_cents(a))
        self.l.append(math.log1p(a))
        pc = ref_cents(paid)
        self.pc.append(pc)
        fmt = str(f["payment_format"])
        self.fmt_s.append(fmt)
        self.fmt.append(self.codes["payment_format"].get(fmt, -1))
        self.pcur.append(self.codes["payment_currency"].get(str(f["payment_currency"]), -1))
        self.rcur.append(self.codes["receiving_currency"].get(str(f["receiving_currency"]), -1))
        self.cross.append(str(f["payment_currency"]) != str(f["receiving_currency"]))
        self.same_bank.append(str(f["from_bank"]) == str(f["to_bank"]))
        self.in_band.append(self.band[0] <= a < self.band[1])
        self.is_round.append(pc > 0 and pc % self.round_cents == 0)
        self.hr.append(fmt in self.hr_formats)
        self.out_of[u].append(i)
        self.in_of[v].append(i)
        self.pair_of[(u, v)].append(i)
        return i

    def __len__(self) -> int:
        return len(self.rank)

    # --- one event ------------------------------------------------------------------------------

    def _was_new(self, j: int) -> bool:
        """Event j's pair had no event in an earlier minute (it was new when j was scored)."""
        mj = self.minute[j]
        return not any(self.minute[k] < mj for k in self.pair_of[(self.src[j], self.dst[j])])

    def _paths(self, i: int) -> tuple[int, int, int]:
        """(c2, c3, c4): earlier temporal paths v -> ... -> u closing on u -> v (§4.7, §6).

        Every edge has minute in [m - W_rt, m - 1] and is not a self-loop; consecutive hops are
        time-ordered with gap <= H; intermediates are not hubs; all vertices are distinct.
        """
        m, u, v = self.minute[i], self.src[i], self.dst[i]
        if u == v:
            return 0, 0, 0
        M, src, hubs, H = self.minute, self.src, self.hubs, self.H
        lo, hi = m - self.W_rt, m - 1

        def into(x: int) -> list[int]:
            return [j for j in self.in_of[x] if lo <= M[j] <= hi and src[j] != x]

        into_u = into(u)
        c2 = sum(1 for j in into_u if src[j] == v)  # the single edge v -> u
        c3 = c4 = 0
        for j_wu in into_u:  # w -> u
            w, t_wu = src[j_wu], M[j_wu]
            if w == v or w in hubs:
                continue
            into_w = into(w)
            for j_xw in into_w:  # x -> w
                x, t_xw = src[j_xw], M[j_xw]
                if x == v:  # 3-hop path v -> w -> u
                    if t_xw <= t_wu <= t_xw + H:
                        c3 += 1
                    continue
                if x == u or x in hubs or not (t_xw <= t_wu <= t_xw + H):
                    continue
                for j_vx in into(x):  # v -> x: 4-hop path v -> x -> w -> u
                    if src[j_vx] == v and M[j_vx] <= t_xw <= M[j_vx] + H:
                        c4 += 1
        return c2, c3, c4

    def _sg(self, i: int) -> tuple[int, int]:
        """(siblings x, distinct sources s) of scatter-gather patterns s -> u, s -> x, x -> v in
        [m - W_sg, m - 1] (uncapped), with s not a hub and s, u, v, x distinct (§4.7)."""
        m, u, v = self.minute[i], self.src[i], self.dst[i]
        if u == v:
            return 0, 0
        M, src, hubs = self.minute, self.src, self.hubs
        lo, hi = m - self.W_sg, m - 1

        def senders(x: int, *, exclude_hubs: bool) -> set[int]:
            return {
                src[j]
                for j in self.in_of[x]
                if lo <= M[j] <= hi and src[j] != x and not (exclude_hubs and src[j] in hubs)
            }

        su = senders(u, exclude_hubs=True)
        mids, srcs = 0, set()
        for x in senders(v, exclude_hubs=False):
            if x == u:
                continue
            common = (senders(x, exclude_hubs=True) & su) - {v}
            if common:
                mids += 1
                srcs |= common
        return mids, len(srcs)

    def compute(self, i: int) -> tuple[dict[str, float], tuple[float, ...], int, dict[str, float]]:
        """(features by name, severities in SCENARIOS order, inflow_c, m2 by feature name)."""
        sp, P = self.spec, self.P
        M, src, dst, c, l = self.minute, self.src, self.dst, self.c, self.l  # noqa: E741
        m, u, v = M[i], src[i], dst[i]
        S, L = self.S, self.L
        tS, tL = ref_window_tag(S), ref_window_tag(L)

        def applied(lst: list[int]) -> list[int]:
            return [j for j in lst if M[j] <= m - 1]

        def within(lst: list[int], w: int) -> list[int]:
            return [j for j in lst if m - w <= M[j] <= m - 1]

        sides = {
            ("u", "out"): self.out_of[u],
            ("u", "in"): self.in_of[u],
            ("v", "in"): self.in_of[v],
            ("v", "out"): self.out_of[v],
        }
        f: dict[str, float] = {}
        m2: dict[str, float] = {}

        # TX (§5.1)
        f["log_amount_usd"] = l[i]
        f["payment_currency"] = float(self.pcur[i])
        f["receiving_currency"] = float(self.rcur[i])
        f["cross_currency"] = float(self.cross[i])
        f["payment_format"] = float(self.fmt[i])
        f["self_loop"] = float(u == v)
        f["same_bank"] = float(self.same_bank[i])
        f["round_amount"] = float(self.is_round[i])
        f["hour_of_day"] = float((m % MINUTES_PER_DAY) // 60)

        # VEL (§5.2) and the AMT sums (§5.3); a self-loop is its own counterparty
        uniq: dict[tuple[str, str, int], int] = {}
        sums: dict[tuple[str, str, int], int] = {}
        for (side, d), lst in sides.items():
            for w in (S, L):
                ev = within(lst, w)
                cps = {dst[j] for j in ev} if d == "out" else {src[j] for j in ev}
                t = ref_window_tag(w)
                f[f"{side}_{d}_cnt_{t}"] = float(len(ev))
                f[f"{side}_{d}_uniq_{t}"] = float(len(cps))
                uniq[(side, d, w)] = len(cps)
                total = sum(c[j] for j in ev)
                sums[(side, d, w)] = total
                f[f"{side}_{d}_sum_{t}"] = math.log1p(total / 100)

        # AMT moments, max and deviation (§5.3)
        for side, d in (("u", "out"), ("v", "in")):
            lst = sides[(side, d)]
            for w in (S, L):
                t = ref_window_tag(w)
                mean, std, sq = _moments([l[j] for j in within(lst, w)])
                f[f"{side}_{d}_mean_{t}"] = mean
                f[f"{side}_{d}_std_{t}"] = std
                m2[f"{side}_{d}_mean_{t}"] = m2[f"{side}_{d}_std_{t}"] = sq
                if w == S:
                    f[f"{side}_amt_dev_{t}"] = l[i] - mean
                    m2[f"{side}_amt_dev_{t}"] = sq
            ev = within(lst, S)
            f[f"{side}_{d}_max_{tS}"] = max(l[j] for j in ev) if ev else NAN

        # FLOW (§5.4)
        pair = self.pair_of[(u, v)]
        for w in (S, L):
            f[f"pair_cnt_{ref_window_tag(w)}"] = float(len(within(pair, w)))
        inflow_c = sum(c[j] for j in within(self.in_of[u], self.W_pt) if src[j] != dst[j])
        tP = ref_window_tag(self.W_pt)
        f[f"u_inflow_{tP}"] = math.log1p(inflow_c / 100)
        f[f"pt_ratio_{tP}"] = float(c[i]) / float(inflow_c) if u != v and inflow_c > 0 else NAN
        for side in ("u", "v"):
            o, n_in = sums[(side, "out", L)], sums[(side, "in", L)]
            f[f"{side}_bal_{tL}"] = (o - n_in) / (o + n_in) if o + n_in else NAN

        # PORT (§4.5, §5.5)
        f["pair_is_new"] = 0.0 if applied(pair) else 1.0
        first = min(M[j] for j in pair if M[j] <= m)  # the pair's first minute (<= m)
        port_out = len({dst[j] for j in self.out_of[u] if M[j] < first})
        port_in = len({src[j] for j in self.in_of[v] if M[j] < first})
        f["out_port"] = math.log1p(min(port_out, sp.cap_port))
        f["in_port"] = math.log1p(min(port_in, sp.cap_port))

        def gap(lst: list[int]) -> float:
            ts = [M[j] for j in applied(lst)]
            return float(min(m - max(ts), sp.cap_gap)) if ts else 0.0

        f["u_out_gap"] = gap(self.out_of[u])
        f["u_in_gap"] = gap(self.in_of[u])
        f["v_in_gap"] = gap(self.in_of[v])
        f["v_out_gap"] = gap(self.out_of[v])
        f["pair_gap"] = gap(pair)
        f["rev_pair_gap"] = gap(self.pair_of.get((v, u), []))

        # CYC (§4.7, §5.6)
        c2, c3, c4 = self._paths(i)
        tR = ref_window_tag(self.W_rt)
        for k, n in ((2, c2), (3, c3), (4, c4)):
            f[f"cyc{k}_{tR}"] = float(min(n, sp.cap_count))

        # SG (§4.7, §5.7)
        mids, n_srcs = self._sg(i)
        tG = ref_window_tag(self.W_sg)
        f[f"sg_mids_{tG}"] = float(min(mids, sp.cap_count))
        f[f"sg_srcs_{tG}"] = float(min(n_srcs, sp.cap_count))
        f[f"gs_u_{tS}"] = float(min(uniq[("u", "in", S)], uniq[("u", "out", S)]))
        f[f"gs_v_{tS}"] = float(min(uniq[("v", "in", S)], uniq[("v", "out", S)]))

        # RULE (§5.8)
        out_u = within(self.out_of[u], S)
        f["in_band"] = float(self.in_band[i])
        f[f"u_out_inband_{tS}"] = float(sum(self.in_band[j] for j in out_u))
        f[f"u_out_round_{tS}"] = float(sum(self.is_round[j] for j in out_u))
        f[f"u_out_newcp_{tS}"] = float(sum(self._was_new(j) for j in out_u))
        for k, fmt in enumerate(self.formats):
            f[f"u_out_fmt_{ref_slug(fmt)}_{tS}"] = float(sum(self.fmt[j] == k for j in out_u))
        same = sum(self.fmt[j] == self.fmt[i] for j in within(self.in_of[v], S))
        f[f"v_in_same_fmt_{tS}"] = float(same) if self.fmt[i] >= 0 else 0.0

        # Severities (§6 = M1 scenarios.sql), each count at its scenario's own window
        hub_u = u in self.hubs
        fan_in = len({src[j] for j in within(self.in_of[v], int(P["fan_in_window"]))})
        fan_out = len({dst[j] for j in within(self.out_of[u], int(P["fan_out_window"]))})

        def burst(flags: list[bool], window_key: str, scenario: str) -> float:
            if not flags[i] or (self.excl[scenario] and hub_u):
                return 0.0
            return float(1 + sum(flags[j] for j in within(self.out_of[u], int(P[window_key]))))

        sev = {
            "fan_in_velocity": float(fan_in),
            "fan_out_velocity": 0.0 if self.excl["fan_out_velocity"] and hub_u else float(fan_out),
            "rapid_pass_through": 0.0
            if u == v or inflow_c <= 0
            else max(0.0, 1.0 - abs(float(c[i]) / float(inflow_c) - 1.0)),
            "round_trip": 0.0 if u == v else float(min(c2 + c3, int(P["max_round_trip_paths"]))),
            "structuring": burst(self.in_band, "structuring_window", "structuring"),
            "round_amount_burst": burst(self.is_round, "round_window", "round_amount_burst"),
            "high_risk_format_burst": burst(self.hr, "high_risk_window", "high_risk_format_burst"),
        }
        return f, tuple(sev[s] for s in SCENARIOS), inflow_c, m2

    def row_m2(self, i: int) -> tuple[tuple, dict[str, float]]:
        """(the engine row layout of event i: features, severities, inflow_c, trunc flags 0;
        the m2 of its mean/std-class features)."""
        f, sev, inflow_c, m2 = self.compute(i)
        names = self.spec.feature_names
        if set(f) != set(names):
            raise AssertionError(
                "reference names differ from the spec's: "
                f"missing {sorted(set(names) - set(f))}, extra {sorted(set(f) - set(names))}"
            )
        return (*(f[n] for n in names), *sev, inflow_c, 0, 0, 0), m2

    def row(self, i: int) -> tuple:
        return self.row_m2(i)[0]

    # --- all events -----------------------------------------------------------------------------

    def simulate_ports(self) -> list[tuple[int, int]]:
        """(port_out, port_in) of every event by a minute-by-minute simulation of the lifetime
        pair relation: score a whole minute against the state, then apply it (§4.5)."""
        receivers: dict[int, set[int]] = defaultdict(set)
        senders: dict[int, set[int]] = defaultdict(set)
        port_of: dict[tuple[int, int], tuple[int, int]] = {}
        out: list[tuple[int, int]] = [(0, 0)] * len(self)
        i = 0
        while i < len(self):
            k = i
            while k < len(self) and self.minute[k] == self.minute[i]:
                k += 1
            minute_events = range(i, k)
            for j in minute_events:  # scoring sees only earlier minutes
                key = (self.src[j], self.dst[j])
                out[j] = port_of.get(key, (len(receivers[key[0]]), len(senders[key[1]])))
            for j in minute_events:  # new pairs keep the minute-start ports
                port_of.setdefault((self.src[j], self.dst[j]), out[j])
            for j in minute_events:
                receivers[self.src[j]].add(self.dst[j])
                senders[self.dst[j]].add(self.src[j])
            i = k
        return out

    def rows(self) -> tuple[list[tuple], list[dict[str, float]]]:
        """Rows and m2 dicts of every event; the port definition is cross-checked against the
        minute-by-minute simulation."""
        sim = self.simulate_ports()
        cap = self.spec.cap_port
        i_out = self.spec.feature_index["out_port"]
        i_in = self.spec.feature_index["in_port"]
        rows, m2s = [], []
        for i in range(len(self)):
            r, m2 = self.row_m2(i)
            want = (math.log1p(min(sim[i][0], cap)), math.log1p(min(sim[i][1], cap)))
            if (r[i_out], r[i_in]) != want:
                raise AssertionError(f"port definition != simulation at event {i}")
            rows.append(r)
            m2s.append(m2)
        return rows, m2s


def reference_frame(frame: pl.DataFrame, spec: EngineSpec) -> pl.DataFrame:
    """row_id, rank, the row layout (float64 features/severities, int tail) and `m2__<feature>`
    columns for the mean/std-class features, in the frame's rank order."""
    ref = Reference.from_frame(frame, spec)
    rows, m2s = ref.rows()
    n_float = spec.i_inflow
    data: dict[str, Any] = {"row_id": frame["row_id"].to_list(), "rank": frame["rank"].to_list()}
    schema: dict[str, Any] = {"row_id": pl.Int64, "rank": pl.Int64}
    for k, name in enumerate(spec.row_layout):
        data[name] = [r[k] for r in rows]
        schema[name] = pl.Float64 if k < n_float else pl.Int64
    for fd in spec.features:
        if fd.tol in ("mean", "std"):
            data[f"m2__{fd.name}"] = [d[fd.name] for d in m2s]
            schema[f"m2__{fd.name}"] = pl.Float64
    return pl.DataFrame(data, schema=schema)


_FRAME_CACHE: dict[tuple, pl.DataFrame] = {}


def reference_frame_cached(frame: pl.DataFrame, spec: EngineSpec) -> pl.DataFrame:
    """`reference_frame`, computed once per (spec, events) in a test session (the fixture's
    reference takes seconds and several suites need the same one)."""
    key = (
        spec.spec_hash(),
        hash(tuple(frame.select(INPUT_COLUMNS).hash_rows(seed=0).to_list())),
    )
    if key not in _FRAME_CACHE:
        _FRAME_CACHE[key] = reference_frame(frame, spec)
    return _FRAME_CACHE[key]


# ---------------------------------------------------------------------------------------------
# Comparison under the tolerance classes (§4.9)


def bits(x: float) -> bytes:
    return struct.pack("<d", float(x))


def same_bits(a: float, b: float) -> bool:
    """Bit-equal floats, any NaN equal to any NaN."""
    a, b = float(a), float(b)
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    return bits(a) == bits(b)


def moment_ok(tol: str, got: float, want: float, m2: float) -> bool:
    """Scalar form of `spec.tol_ok` for the mean / std classes on float64 values (§4.9):
    |got - want| <= 1e-9 * max(1, sqrt(m2)) (mean) or sqrt(1e-9 * max(1, m2)) (std)."""
    if math.isnan(got) or math.isnan(want):
        return math.isnan(got) and math.isnan(want)
    if got == want:
        return True
    if tol == "mean":
        bound = TOL_VAR * max(1.0, math.sqrt(max(m2, 0.0)))
    else:
        bound = math.sqrt(TOL_VAR * max(1.0, m2))
    return abs(got - want) <= bound


def compare_rows(
    got: Sequence,
    want: Sequence,
    m2: Mapping[str, float],
    spec: EngineSpec,
    *,
    independent: bool = False,
) -> list[str]:
    """Mismatches of an engine row against a reference row (empty list = equal).

    exact / ulp features and the severities are bit-equal (independent=True: ulp within 1 ulp);
    mean / std features use `tol_ok` with the reference's m2. Where the engine's row carries a
    trunc flag, the affected counts (and the round-trip severity) may only be <= the reference.
    """
    bad: list[str] = []
    n = spec.n_features
    if len(got) != len(spec.row_layout) or len(want) != len(spec.row_layout):
        return [f"row length {len(got)} / {len(want)} != {len(spec.row_layout)}"]
    rule_t, cyc_t, sg_t = got[spec.i_rule_trunc], got[spec.i_cyc_trunc], got[spec.i_sg_trunc]
    for name, flag in (("rule_trunc", rule_t), ("cyc_trunc", cyc_t), ("sg_trunc", sg_t)):
        if type(flag) is not int or flag not in (0, 1):
            bad.append(f"{name}={flag!r} is not an int 0/1")
    lower_bound: set[str] = set()
    if rule_t:
        lower_bound |= set(RULE_TRUNC_STATS)
    if rule_t or cyc_t:
        lower_bound |= set(CYC_TRUNC_STATS)
    if sg_t:
        lower_bound |= set(SG_TRUNC_STATS)
    for k, fd in enumerate(spec.features):
        g, w = got[k], want[k]
        if not isinstance(g, float):
            bad.append(f"{fd.name}: engine value {g!r} is not a float")
            continue
        if fd.stat in lower_bound:
            if not g <= w:
                bad.append(f"{fd.name}: truncated {g!r} > reference {w!r}")
            continue
        if fd.tol in ("mean", "std"):
            ok = moment_ok(fd.tol, g, w, m2[fd.name])
        elif fd.tol == "ulp" and independent:
            ok = bool(tol_ok("ulp", g, w, independent=True)[0])
        else:
            ok = same_bits(g, w)
        if not ok:
            bad.append(f"{fd.name} [{fd.tol}]: engine {g!r} != reference {w!r}")
    for k, s in enumerate(SCENARIOS):
        g, w = got[n + k], want[n + k]
        if s == "round_trip" and rule_t:
            if not g <= w:
                bad.append(f"{s}: truncated {g!r} > reference {w!r}")
        elif not (isinstance(g, float) and same_bits(g, w)):
            bad.append(f"severity {s}: engine {g!r} != reference {w!r}")
    g_in, w_in = got[spec.i_inflow], want[spec.i_inflow]
    if type(g_in) is not int or g_in != w_in:
        bad.append(f"inflow_c: engine {g_in!r} != reference {w_in!r}")
    return bad


def compare_frames(
    got: pl.DataFrame,
    want: pl.DataFrame,
    spec: EngineSpec,
    columns: Sequence[str],
    *,
    independent: bool,
    float32: bool = False,
) -> dict[str, int]:
    """Mismatch count per column between two frames aligned by row_id (tolerance classes; the
    m2 columns come from `want`). A missing or extra row_id raises."""
    if got["row_id"].to_list() != want["row_id"].to_list():
        raise AssertionError("frames are not aligned by row_id")
    out = {}
    for name in columns:
        tol = spec.feature(name).tol
        m2 = want[f"m2__{name}"].to_numpy() if tol in ("mean", "std") else None
        ok = tol_ok(
            tol,
            got[name].to_numpy(),
            want[name].to_numpy(),
            m2,
            independent=independent,
            float32=float32,
        )
        out[name] = int(np.count_nonzero(~ok))
    return out
