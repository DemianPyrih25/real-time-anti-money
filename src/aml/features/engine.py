"""The causal feature engine: one scalar state machine for every model input (M2 spec §4).

The public API below is what the offline driver (`features.build`), the rules, the serving bundle,
the M5 scorer and the tests call. Internal attribute names other agents rely on:

- `ring` (`windows.Ring`, a `spec.RingView`), `head_in`, `head_out` (array('i') by account),
  `hub` (bytearray by account) and `memo` (`cycles.MinuteMemo`, cleared by `advance`): passed
  to `features.cycles` explicitly.
- `spec` (the frozen `EngineSpec`).

As-of rule as code (§4.2): state = applied events with minute <= clock - 1, expired per window to
[m - W, m - 1]; events of the current minute are pending (scored, never applied) until `advance`
moves past their minute. Same-minute peers are invisible in both rank directions.

M5 per message::

    row = eng.process(eng.prepare(*fields))           # fields in spec.INPUT_COLUMNS order
    x = np.asarray([row[i] for i in model_idx], np.float64).astype(np.float32)[None]
    booster.predict(x, num_threads=1)

The engine is not thread-safe; one consumer thread owns it. After an exception raised inside
`advance` (OverflowError of a counter, a broken invariant) the engine must be discarded.

Speed (pure Python, one core): the per-event paths are closures over local array references,
built once per engine by `_bind`. The per-slot updates of apply and expiry are generated from the
window registry (`_slot_statements`), so expiry is the exact mirror of apply by construction and
any configured window set runs without per-slot loops.
"""

from __future__ import annotations

import bisect
import hashlib
import json
import math
import operator
import re
import sys
from collections import Counter, defaultdict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from time import perf_counter
from typing import Any, ClassVar

from aml.features import cycles as _cycles
from aml.features import snapshot as _snap
from aml.features.ports import PairTable
from aml.features.spec import (
    ACCOUNT_COLUMNS,
    E_MINUTE,
    E_RANK,
    ENGINE_VERSION,
    EXACT_INT_LIMIT,
    F_CROSS,
    F_HR,
    F_IN_BAND,
    F_NEW,
    F_ROUND,
    F_SAME_BANK,
    F_SELF,
    I32_LIMIT,
    MINUTES_PER_DAY,
    NO_PRED,
    PAIR_COLUMNS,
    RING_COLUMNS,
    EngineError,
    EngineSpec,
    FlushStats,
    LateEventError,
    NumericRangeError,
    RankGapError,
    Slot,
    SnapshotError,
    SpecError,
    format_slug,
    window_tag,
)
from aml.features.tx_features import TX_FEATURES, cents
from aml.features.windows import Ring, Slots, filled, window_max
from aml.rules import scenarios as _scenarios

_NOOP = FlushStats(0, 0, 0.0)
_FMT_CODE_MAX = 127  # ring.fmt is array('b')


def expected_feature_names(spec: EngineSpec) -> tuple[str, ...]:
    """The model inputs in the order `Engine` writes them (M2 spec §5.1-§5.8).

    The scorer is written for exactly this order; `Engine` refuses a spec whose generated names
    differ, so a change in `spec.build_features` cannot silently shift columns.
    """
    S, L = window_tag(spec.w_short), window_tag(spec.w_long)
    P, R, G = window_tag(spec.w_pt), window_tag(spec.w_rt), window_tag(spec.w_sg)
    names = list(TX_FEATURES)
    for side, d, stat in (
        ("u", "out", "cnt"),
        ("u", "out", "uniq"),
        ("u", "in", "cnt"),
        ("u", "in", "uniq"),
        ("v", "in", "cnt"),
        ("v", "in", "uniq"),
        ("v", "out", "cnt"),
        ("v", "out", "uniq"),
    ):
        names += [f"{side}_{d}_{stat}_{S}", f"{side}_{d}_{stat}_{L}"]
    for side, d in (("u", "out"), ("u", "in"), ("v", "in"), ("v", "out")):
        names += [f"{side}_{d}_sum_{S}", f"{side}_{d}_sum_{L}"]
    for stat in ("mean", "std"):
        for side, d in (("u", "out"), ("v", "in")):
            names += [f"{side}_{d}_{stat}_{S}", f"{side}_{d}_{stat}_{L}"]
    names += [f"u_out_max_{S}", f"v_in_max_{S}", f"u_amt_dev_{S}", f"v_amt_dev_{S}"]
    names += [f"pair_cnt_{S}", f"pair_cnt_{L}", f"u_inflow_{P}", f"pt_ratio_{P}"]
    names += [f"u_bal_{L}", f"v_bal_{L}"]
    names += ["pair_is_new", "out_port", "in_port", "u_out_gap", "u_in_gap", "v_in_gap"]
    names += ["v_out_gap", "pair_gap", "rev_pair_gap"]
    names += [f"cyc2_{R}", f"cyc3_{R}", f"cyc4_{R}"]
    names += [f"sg_mids_{G}", f"sg_srcs_{G}", f"gs_u_{S}", f"gs_v_{S}"]
    names += ["in_band", f"u_out_inband_{S}", f"u_out_round_{S}", f"u_out_newcp_{S}"]
    names += [f"u_out_fmt_{format_slug(fm)}_{S}" for fm in spec.vocab["payment_format"]]
    names.append(f"v_in_same_fmt_{S}")
    return tuple(names)


def component_layout(spec: EngineSpec, n_pairs: int, ring_rows: int) -> list[tuple[str, str, int]]:
    """(name, typecode, n_items) of every snapshot component, in STATE_FIELDS order (§7)."""
    n = spec.n_accounts
    out = [(f"ring.{name}", tc, ring_rows) for name, tc in RING_COLUMNS]
    out += [(s.name, s.typecode, n) for s in spec.slots]
    out += [(f"acct.{name}", tc, n) for name, tc in ACCOUNT_COLUMNS]
    out.append(("hub", "B", n))
    out.append(("pairs.keys", "q", n_pairs))
    out += [(f"pairs.pcnt.{w}", "i", n_pairs) for w in spec.uniq_windows]
    out += [(f"pairs.{name}", tc, n_pairs) for name, tc in PAIR_COLUMNS]
    return out


def hubs_sha256(hubs: tuple[int, ...]) -> str:
    """sha256 of the canonical JSON list of hub ids (snapshot and bundle headers)."""
    return hashlib.sha256(json.dumps(list(hubs), separators=(",", ":")).encode()).hexdigest()


def _hub_bytes(spec: EngineSpec) -> bytearray:
    hub = bytearray(spec.n_accounts)
    for h in spec.hubs:
        hub[h] = 1
    return hub


def _index(name: str, value: Any) -> int:
    try:
        return operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be an integer, got {value!r}") from None


class Engine:
    """Incremental causal state over the rank-ordered event stream.

    `set(vars(engine)) == STATE_FIELDS | DERIVED_FIELDS | TRANSIENT_FIELDS` is a test, so an
    attribute nobody adds to the snapshot fails CI.
    """

    # Serialised by the snapshot (§7): clock, next_rank and cursors in the header, the rest as
    # components (ring rows [live_start, end_rank), slots, acct.*, hub, pairs.*).
    STATE_FIELDS: ClassVar[tuple[str, ...]] = (
        "clock",
        "next_rank",
        "cursors",
        "ring",
        "slots",
        "last_out",
        "last_in",
        "ever_out",
        "ever_in",
        "head_out",
        "head_in",
        "hub",
        "pairs",
    )
    # Rebuilt from the spec and the state objects by `_bind` (code dicts, compiled kernels).
    DERIVED_FIELDS: ClassVar[tuple[str, ...]] = (
        "spec",
        "_prep",
        "_hubs_sha256",
        "_score",
        "_apply",
        "_expirers",
    )
    # Never saved: the pending minute, port bumps, the per-minute path memo, the last flush.
    TRANSIENT_FIELDS: ClassVar[tuple[str, ...]] = (
        "pending",
        "bump_out",
        "bump_in",
        "memo",
        "last_flush",
    )

    clock: int | None  # the minute being scored; None before the first event
    next_rank: int  # the rank the next processed event must carry

    def __init__(self, spec: EngineSpec) -> None:
        if not isinstance(spec, EngineSpec):
            raise TypeError(f"spec must be an EngineSpec, got {type(spec).__name__}")
        n = spec.n_accounts
        self._setup(
            spec,
            clock=None,
            next_rank=0,
            cursors={w: 0 for w in spec.windows_all},
            ring=Ring(spec.compact_min_rows, 0),
            slots=Slots(spec),
            accounts={
                name: filled(tc, n, -1 if name.startswith(("last_", "head_")) else 0)
                for name, tc in ACCOUNT_COLUMNS
            },
            hub=_hub_bytes(spec),
            pairs=PairTable(spec.uniq_windows),
        )

    def _setup(
        self,
        spec: EngineSpec,
        *,
        clock: int | None,
        next_rank: int,
        cursors: dict[int, int],
        ring: Ring,
        slots: Slots,
        accounts: dict[str, Any],
        hub: bytearray,
        pairs: PairTable,
    ) -> None:
        """Assign every field (state, transients, derived) in one place."""
        _check_engine_spec(spec)
        self.spec = spec
        self.clock = clock
        self.next_rank = next_rank
        self.cursors = cursors
        self.ring = ring
        self.slots = slots
        for name, _ in ACCOUNT_COLUMNS:
            setattr(self, name, accounts[name])
        self.hub = hub
        self.pairs = pairs
        self.pending: list[tuple[tuple, int]] = []  # (prepared event, new-pair flag), rank order
        self.bump_out: dict[int, int] = {}  # account -> new receivers in the pending minute
        self.bump_in: dict[int, int] = {}
        self.memo = _cycles.MinuteMemo()
        self.last_flush = _NOOP
        self._bind()

    def _bind(self) -> None:
        """(Re)build the derived fields: code dicts and the compiled per-event kernels."""
        spec = self.spec
        vocab = spec.vocab
        self._prep = (
            {c: i for i, c in enumerate(vocab["payment_format"])},
            {c: i for i, c in enumerate(vocab["payment_currency"])},
            {c: i for i, c in enumerate(vocab["receiving_currency"])},
            frozenset(spec.high_risk_formats),
            spec.band_low_usd,
            spec.band_high_usd,
            spec.round_cents,
            spec.n_accounts,
        )
        self._hubs_sha256 = hubs_sha256(spec.hubs)
        self._score = _make_scorer(self)
        self._apply = _make_apply(self)
        self._expirers = tuple((w, _make_expirer(self, w)) for w in spec.windows_all)

    # --- construction ---------------------------------------------------------------------------

    @classmethod
    def create(cls, spec: EngineSpec) -> Engine:
        """An empty engine (clock None, next_rank 0)."""
        return cls(spec)

    @classmethod
    def restore(cls, src: Path | bytes, spec: EngineSpec) -> tuple[Engine, dict[str, Any]]:
        """(engine, header) from a snapshot file or bytes; raises `spec.SnapshotError` on a wrong
        magic / byteorder / payload sha256 / engine_version / spec_hash or a digest mismatch."""
        if not isinstance(spec, EngineSpec):
            raise TypeError(f"spec must be an EngineSpec, got {type(spec).__name__}")
        header, comps = _snap.decode(src)

        def need(ok: bool, what: str) -> None:
            if not ok:
                raise SnapshotError(f"snapshot does not match: {what}")

        need(header.get("engine_version") == ENGINE_VERSION, f"engine_version "
             f"{header.get('engine_version')!r} != {ENGINE_VERSION}")  # fmt: skip
        need(header.get("spec_hash") == spec.spec_hash(), f"spec_hash "
             f"{header.get('spec_hash')!r} != {spec.spec_hash()!r}")  # fmt: skip
        need(header.get("n_accounts") == spec.n_accounts, "n_accounts")
        need(header.get("hub_cap") == spec.hub_cap, "hub_cap")
        need(header.get("hubs_sha256") == hubs_sha256(spec.hubs), "hubs_sha256")
        clock, next_rank = header.get("clock"), header.get("next_rank")
        live_start, n_pairs = header.get("live_start"), header.get("n_pairs")
        ints = (next_rank, live_start, n_pairs)
        need(all(type(x) is int for x in ints), "next_rank / live_start / n_pairs types")
        need(clock is None or (type(clock) is int and 0 <= clock < I32_LIMIT), "clock")
        need(0 <= live_start <= next_rank < I32_LIMIT and n_pairs >= 0, "rank range")
        raw_cursors = header.get("cursors")
        need(isinstance(raw_cursors, dict), "cursors")
        try:
            cursors = {int(w): r for w, r in raw_cursors.items()}
        except ValueError:
            raise SnapshotError("snapshot cursors keys are not windows") from None
        need(set(cursors) == set(spec.windows_all), "cursor windows")
        need(all(type(r) is int and live_start <= r <= next_rank for r in cursors.values()),
             "cursor ranks")  # fmt: skip
        need(cursors[spec.W_max] == live_start, "cursor[W_max] == live_start")
        layout = component_layout(spec, n_pairs, next_rank - live_start)
        got = [(c["name"], c["typecode"], c["n_items"]) for c in header["components"]]
        need(got == layout, "component layout")
        hub = bytearray(comps["hub"])
        need(hub == _hub_bytes(spec), "hub flags")
        try:
            ring = Ring.from_columns(
                spec.compact_min_rows,
                live_start,
                next_rank,
                {name: comps[f"ring.{name}"] for name, _ in RING_COLUMNS},
            )
            slots = Slots(spec, {s: comps[s.name] for s in spec.slots})
            pairs = PairTable.from_columns(
                spec.uniq_windows,
                comps["pairs.keys"],
                {w: comps[f"pairs.pcnt.{w}"] for w in spec.uniq_windows},
                comps["pairs.port_out"],
                comps["pairs.port_in"],
                comps["pairs.last_min"],
            )
        except ValueError as e:
            raise SnapshotError(f"snapshot state is inconsistent: {e}") from None
        eng = cls.__new__(cls)
        eng._setup(
            spec,
            clock=clock,
            next_rank=next_rank,
            cursors={w: cursors[w] for w in spec.windows_all},
            ring=ring,
            slots=slots,
            accounts={name: comps[f"acct.{name}"] for name, _ in ACCOUNT_COLUMNS},
            hub=hub,
            pairs=pairs,
        )
        need(eng.state_digest() == header.get("state_digest"), "state digest")
        return eng, header

    # --- the event path -------------------------------------------------------------------------

    def prepare(
        self,
        rank: int,
        row_id: int,
        minute: int,
        src: int,
        dst: int,
        amount_usd: float,
        amount_paid: float,
        payment_format: str,
        payment_currency: str,
        receiving_currency: str,
        from_bank: str,
        to_bank: str,
    ) -> tuple:
        """The prepared event (indexed by `spec.E_*`), arguments in `spec.INPUT_COLUMNS` order.

        Pure (reads only the spec and code dicts). ValueError on a negative or non-finite amount,
        an account id outside [0, n_accounts), or a rank / minute outside [0, 2^31 - 1);
        NumericRangeError if cents(amount_usd) >= 2^53.
        """
        fmt_code, pcur_code, rcur_code, hr_formats, band_lo, band_hi, round_c, n = self._prep
        rank, row_id = _index("rank", rank), _index("row_id", row_id)
        minute, src, dst = _index("minute", minute), _index("src", src), _index("dst", dst)
        if not (0 <= rank < I32_LIMIT and 0 <= minute < I32_LIMIT):
            raise ValueError(f"rank {rank} / minute {minute} outside [0, {I32_LIMIT})")
        if not (0 <= src < n and 0 <= dst < n):
            raise ValueError(f"account ids ({src}, {dst}) outside [0, {n})")
        usd, paid = float(amount_usd), float(amount_paid)
        usd_c, paid_c = cents(usd), cents(paid)  # ValueError if negative or not finite
        if usd_c >= EXACT_INT_LIMIT:
            raise NumericRangeError(f"amount_usd {usd!r} is too large for exact cents sums")
        flags = (
            (F_SELF if src == dst else 0)
            | (F_IN_BAND if band_lo <= usd < band_hi else 0)
            | (F_ROUND if paid_c > 0 and paid_c % round_c == 0 else 0)
            | (F_HR if payment_format in hr_formats else 0)
            | (F_CROSS if payment_currency != receiving_currency else 0)
            | (F_SAME_BANK if from_bank == to_bank else 0)
        )
        return (
            rank,
            row_id,
            minute,
            src,
            dst,
            usd_c,
            math.log1p(usd),
            usd,
            paid_c,
            fmt_code.get(payment_format, -1),
            pcur_code.get(payment_currency, -1),
            rcur_code.get(receiving_currency, -1),
            flags,
            (minute % MINUTES_PER_DAY) // 60,
        )

    def process(self, ev: tuple) -> tuple:
        """Advance to the event's minute if it is later (the minute flush), score it, queue it.

        Returns the row (`spec.row_layout`). `spec.LateEventError` if its minute is earlier than
        the clock, `spec.RankGapError` if its rank is not `next_rank`; both leave the state as
        it was.
        """
        m = ev[E_MINUTE]
        clock = self.clock
        if clock is not None and m < clock:
            raise LateEventError(f"event rank {ev[E_RANK]} minute {m} < clock {clock}")
        if ev[E_RANK] != self.next_rank:
            raise RankGapError(f"event rank {ev[E_RANK]} != next_rank {self.next_rank}")
        if clock is None or m > clock:
            self.advance(m)
        row, new = self._score(ev)
        self.pending.append((ev, new))
        self.next_rank += 1
        return row

    def score(self, ev: tuple) -> tuple:
        """The row `process` would return, without queuing the event (state digest unchanged).

        The event's minute must equal the clock (or the clock must be None): LateEventError for
        an earlier minute, EngineError for a later one (advance first).
        """
        m, clock = ev[E_MINUTE], self.clock
        if clock is not None and m != clock:
            if m < clock:
                raise LateEventError(f"event minute {m} < clock {clock}")
            raise EngineError(f"event minute {m} > clock {clock}: advance first")
        return self._score(ev)[0]

    def advance(self, minute: int) -> FlushStats:
        """Apply the pending minute, expire every window relative to `minute`, set the clock.

        A no-op for minute == clock; `spec.LateEventError` for an earlier minute. `advance(b)` then
        `advance(m)` (no events between) is identical to `advance(m)`.
        """
        m = _index("minute", minute)
        if not 0 <= m < I32_LIMIT:
            raise ValueError(f"minute {m} outside [0, {I32_LIMIT})")
        clock = self.clock
        if clock is not None and m <= clock:
            if m == clock:
                return _NOOP
            raise LateEventError(f"advance to minute {m} < clock {clock}")
        t0 = perf_counter()
        pending = self.pending
        n_applied = len(pending)
        if n_applied:
            self._apply(pending, clock)  # rank order; then ever_out/ever_in += the new pairs
            pending.clear()
        ring = self.ring
        end, base = ring.end_rank, ring.base
        cursors = self.cursors
        n_expired = 0
        for w, expire in self._expirers:  # ascending windows; each is independent
            cur = cursors[w]
            if cur < end:
                new = expire(cur, end, m - w, base)
                n_expired += new - cur
                cursors[w] = new
        ring.live_start = cursors[self.spec.W_max]
        ring.maybe_compact()
        self.clock = m
        self.memo.clear()
        stats = FlushStats(n_applied, n_expired, perf_counter() - t0)
        self.last_flush = stats
        return stats

    @property
    def pending_count(self) -> int:
        """Events of the current minute scored but not yet applied."""
        return len(self.pending)

    # --- snapshots ------------------------------------------------------------------------------

    def _semantic(self) -> dict[str, Any]:
        """The header fields that define the applied state (`snapshot.DIGEST_FIELDS`)."""
        spec, ring = self.spec, self.ring
        return {
            "engine_version": ENGINE_VERSION,
            "spec_hash": spec.spec_hash(),
            "clock": self.clock,
            # = the rank of the first pending event, else next_rank: all lower ranks are applied
            "next_rank": ring.end_rank,
            "cursors": {str(w): self.cursors[w] for w in spec.windows_all},
            "n_accounts": spec.n_accounts,
            "n_pairs": self.pairs.n_pairs,
            "live_start": ring.live_start,
            "hub_cap": spec.hub_cap,
            "hubs_sha256": self._hubs_sha256,
        }

    @contextmanager
    def _components(self) -> Iterator[list[tuple[str, Any]]]:
        """(name, buffer) of every state component in STATE_FIELDS order; the live ring rows are
        zero-copy views, released on exit (an exported buffer would block array appends)."""
        ring = self.ring
        dead = ring.live_start - ring.base
        views: list[memoryview] = []
        try:
            comps: list[tuple[str, Any]] = []
            for name, col in ring.columns():
                if dead:
                    mv = memoryview(col)
                    views.append(mv)
                    col = mv[dead:]
                    views.append(col)
                comps.append((f"ring.{name}", col))
            comps += [(s.name, a) for s, a in self.slots.arrays.items()]
            comps += [(f"acct.{name}", getattr(self, name)) for name, _ in ACCOUNT_COLUMNS]
            comps.append(("hub", self.hub))
            comps += [(f"pairs.{name}", a) for name, a in self.pairs.columns()]
            yield comps
        finally:
            for mv in reversed(views):
                mv.release()

    def snapshot(
        self,
        dst: Path | None = None,
        *,
        next_offset: int | None = None,
        extra: dict | None = None,
        compress: bool = True,
    ) -> bytes | dict:
        """The applied state (never the pending events) in the §7 format.

        dst None -> the snapshot bytes; else an atomic write of `dst` plus a `.json` sidecar
        (header + file sha256 and size) and the sidecar document is returned. The header's
        next_rank is the rank of the first pending event (else `next_rank`); next_offset is the
        caller's Kafka offset of that event (None offline).
        """
        if next_offset is not None:
            next_offset = _index("next_offset", next_offset)
        header = self._semantic()
        header["next_offset"] = next_offset
        header["extra"] = dict(extra or {})
        with self._components() as comps:
            header["state_digest"] = _snap.digest(header, comps)
            data = _snap.encode(header, comps, compress=compress)
        if dst is None:
            return data
        return _snap.write_atomic(data, Path(dst), _snap.peek_header(data))

    def state_digest(self) -> str:
        """sha256 over the semantic header fields and every state component (excludes ring.base
        and the pending events, so an uninterrupted and a restored engine agree)."""
        with self._components() as comps:
            return _snap.digest(self._semantic(), comps)

    def state_nbytes(self) -> dict[str, int]:
        """Bytes per state component (ring, slots, accounts, pairs, transients) and the total.

        Arrays are counted with their allocated capacity; the pair index and the transients are
        estimates (tracemalloc is the measurement)."""
        size = sys.getsizeof
        out = {
            "ring": self.ring.nbytes(),
            "slots": self.slots.nbytes(),
            "accounts": sum(size(getattr(self, name)) for name, _ in ACCOUNT_COLUMNS)
            + size(self.hub),
            "pairs": self.pairs.nbytes(),
            "transients": size(self.pending)
            + sum(size(ev) + 64 for ev, _ in self.pending)
            + size(self.bump_out)
            + size(self.bump_in)
            + sum(size(d) for d in (self.memo.path, self.memo.su, self.memo.ev)),
        }
        out["total"] = sum(out.values())
        return out

    # --- reads for M6 and tests -----------------------------------------------------------------

    def neighbours(
        self, acct: int, direction: str, window: int, cap: int
    ) -> list[tuple[int, int, int, int]]:
        """[(counterparty, minute, rank, usd_c)] newest first; see `cycles.neighbours`."""
        return _cycles.neighbours(
            self.ring,
            self.head_in,
            self.head_out,
            self.spec,
            self.clock,
            acct,
            direction,
            window,
            cap,
        )

    def check_invariants(self) -> None:
        """Tests: every aggregate equals a recount from the ring (AssertionError otherwise)."""
        _check_invariants(self)


# --- spec checks ----------------------------------------------------------------------------------


def _check_engine_spec(spec: EngineSpec) -> None:
    got, want = spec.feature_names, expected_feature_names(spec)
    if got != want:
        diff = [(i, g, w) for i, (g, w) in enumerate(zip(got, want, strict=False)) if g != w]
        raise SpecError(
            f"the spec's feature layout differs from the engine's ({len(got)} vs {len(want)} "
            f"names; first differences {diff[:3]}): update Engine and bump ENGINE_VERSION"
        )
    if len(spec.vocab["payment_format"]) > _FMT_CODE_MAX:
        raise SpecError(f"at most {_FMT_CODE_MAX} payment formats fit the ring's fmt column")


# --- generated slot kernels -----------------------------------------------------------------------

_SIDE_KEY = {"out": "u", "in": "v"}  # out slots are keyed by src, in slots by dst
_FLAG_KINDS = {"inband": F_IN_BAND, "round": F_ROUND, "hr": F_HR, "newcp": F_NEW}
_EVENT_VARS = ("u", "v", "c", "x", "fl", "f", "pid")  # src, dst, usd_c, l, flags, fmt, pair id


class _Kernel:
    """Source builder for one generated function: arrays it touches become closure variables."""

    def __init__(self) -> None:
        self.env: dict[str, Any] = {}
        self._ids: dict[int, str] = {}

    def ref(self, obj: Any) -> str:
        name = self._ids.get(id(obj))
        if name is None:
            name = f"a{len(self.env)}"
            self.env[name] = obj
            self._ids[id(obj)] = name
        return name

    def build(self, name: str, params: str, lines: list[str]) -> Callable:
        body = "\n".join(f"        {line}" for line in lines)
        src = f"def _make({', '.join(self.env)}):\n    def {name}({params}):\n{body}\n"
        src += f"    return {name}\n"
        ns: dict[str, Any] = {}
        exec(compile(src, f"<aml.features.engine {name}>", "exec"), ns)  # noqa: S102
        fn = ns["_make"](**self.env)
        fn.__aml_source__ = src  # for debugging and tests
        return fn


def _slot_statements(eng: Engine, k: _Kernel, windows: tuple[int, ...], *, add: bool) -> list[str]:
    """Statements adding (add=True) or removing one event's contribution to every slot and pair
    count of `windows` (§4.4 steps 4-5 and their inverse). Event variables: `_EVENT_VARS`.

    Removal resets a side's s1 / s2 to exactly 0.0 when that side's count reaches 0.
    """
    spec, arrays, pcnt = eng.spec, eng.slots.arrays, eng.pairs.pcnt
    n_fmt = len(spec.vocab["payment_format"])
    op = "+=" if add else "-="
    plain: list[str] = []
    cond: dict[str, list[str]] = defaultdict(list)

    def slot(side: str, kind: str, w: int, pred: int = NO_PRED) -> str | None:
        a = arrays.get(Slot(side, kind, w, pred))
        return None if a is None else k.ref(a)

    for w in windows:
        for side in ("out", "in"):
            key = _SIDE_KEY[side]
            cnt, sum_c, s1, s2 = (slot(side, kind, w) for kind in ("cnt", "sum_c", "s1", "s2"))
            if (s1 or s2) and not cnt:
                raise SpecError(f"window {w} has {side} moments without a {side} count")
            if cnt and (add or not (s1 or s2)):
                plain.append(f"{cnt}[{key}] {op} 1")
            elif cnt:
                plain += [f"n_{side} = {cnt}[{key}] - 1", f"{cnt}[{key}] = n_{side}"]
            if sum_c:
                plain.append(f"{sum_c}[{key}] {op} c")
            moments = [m for m in ((s1, "x"), (s2, "x * x")) if m[0]]
            if moments and add:
                plain += [f"{a}[{key}] += {val}" for a, val in moments]
            elif moments:
                plain.append(f"if n_{side}:")
                plain += [f"    {a}[{key}] -= {val}" for a, val in moments]
                plain.append("else:")
                plain += [f"    {a}[{key}] = 0.0" for a, _ in moments]
            nsl = slot(side, "nsl_sum_c", w)
            if nsl:
                cond[f"not fl & {F_SELF}"].append(f"{nsl}[{key}] {op} c")
            for kind, bit in _FLAG_KINDS.items():
                a = slot(side, kind, w)
                if a:
                    cond[f"fl & {bit}"].append(f"{a}[{key}] {op} 1")
            fmts = [arrays.get(Slot(side, "fmt", w, p)) for p in range(n_fmt)]
            if any(a is not None for a in fmts):
                if any(a is None for a in fmts):
                    raise SpecError(f"window {w}: {side} format counts must cover every format")
                cond["f >= 0"].append(f"{k.ref(tuple(fmts))}[f][{key}] {op} 1")
        if w in pcnt:  # distinct counterparties: the pair count's 0 <-> 1 transitions
            uo, ui = slot("out", "uniq", w), slot("in", "uniq", w)
            if not (uo and ui):
                raise SpecError(f"uniq window {w} needs both uniq slots")
            pc = k.ref(pcnt[w])
            if add:
                plain += [f"n_p = {pc}[pid] + 1", f"{pc}[pid] = n_p", "if n_p == 1:"]
            else:
                plain += [f"n_p = {pc}[pid] - 1", f"{pc}[pid] = n_p", "if not n_p:"]
            plain += [f"    {uo}[u] {op} 1", f"    {ui}[v] {op} 1"]
    for test, stmts in cond.items():
        plain.append(f"if {test}:")
        plain += [f"    {s}" for s in stmts]
    return plain


def _make_add(eng: Engine) -> Callable:
    """add(u, v, c, x, fl, f, pid): one applied event into every slot and pair count."""
    k = _Kernel()
    lines = _slot_statements(eng, k, eng.spec.windows_all, add=True)
    return k.build("add", ", ".join(_EVENT_VARS), lines or ["pass"])


def _make_expirer(eng: Engine, w: int) -> Callable:
    """expire(cursor, end, lo, base) -> new cursor: removes ring rows [cursor, ...) with minute <
    lo from window w (the exact inverse of `add` for w's slots)."""
    ring = eng.ring
    k = _Kernel()
    stmts = _slot_statements(eng, k, (w,), add=False)
    if not stmts:  # nothing to subtract: the cursor only moves (minutes are non-decreasing)
        minute = ring.minute

        def expire(cursor: int, end: int, lo: int, base: int) -> int:
            return bisect.bisect_left(minute, lo, cursor - base, end - base) + base

        return expire
    text = "\n".join(stmts)
    cols = dict(zip(_EVENT_VARS, (ring.src, ring.dst, ring.usd_c, ring.l, ring.flags, ring.fmt,
                                  ring.pid), strict=True))  # fmt: skip
    lines = ["while cursor < end:", "    i = cursor - base"]
    lines += [f"    if {k.ref(ring.minute)}[i] >= lo:", "        break"]
    lines += [f"    {var} = {k.ref(col)}[i]" for var, col in cols.items()
              if re.search(rf"\b{var}\b", text)]  # fmt: skip
    lines += [f"    {s}" for s in stmts]
    lines += ["    cursor += 1", "return cursor"]
    return k.build("expire", "cursor, end, lo, base", lines)


# --- apply ----------------------------------------------------------------------------------------


def _make_apply(eng: Engine) -> Callable[[list, int], None]:
    """apply(pending, p): the pending minute p into the state (§4.4 `_apply`, then
    `_commit_ports`)."""
    ring = eng.ring
    r_l, r_pmax_out, r_pmax_in = ring.l, ring.pmax_out, ring.pmax_in
    (ap_minute, ap_src, ap_dst, ap_usd_c, ap_l, ap_fmt, ap_flags, ap_pid, ap_prev_out,
     ap_prev_in, ap_pmax_out, ap_pmax_in) = (col.append for _, col in ring.columns())  # fmt: skip
    pairs = eng.pairs
    index_get, pairs_add, last_min = pairs.index.get, pairs.add, pairs.last_min
    head_out, head_in = eng.head_out, eng.head_in
    ever_out, ever_in = eng.ever_out, eng.ever_in
    last_out, last_in = eng.last_out, eng.last_in
    bump_out, bump_in = eng.bump_out, eng.bump_in
    add = _make_add(eng)

    def apply(pending: list, p: int) -> None:
        base = ring.base
        live = ring.live_start
        for ev, new in pending:
            r, _, _, u, v, c, x, _, _, f, _, _, fl, _ = ev
            if r != ring.end_rank:
                raise EngineError(f"apply: rank {r} != ring end_rank {ring.end_rank}")
            if new:
                fl |= F_NEW
            # 1. suffix-max chains (§4.6): pop the earlier rows with l <= x, stop at live_start
            q = head_out[u]
            while q >= live and r_l[q - base] <= x:
                q = r_pmax_out[q - base]
            pmo = q if q >= live else -1
            q = head_in[v]
            while q >= live and r_l[q - base] <= x:
                q = r_pmax_in[q - base]
            pmi = q if q >= live else -1
            # 2. pair; ports are the minute-start values (ever_* change only after the minute)
            pid = index_get((u << 32) | v)
            if pid is None:
                pid = pairs_add(u, v, ever_out[u], ever_in[v], p)
                bump_out[u] = bump_out.get(u, 0) + 1
                bump_in[v] = bump_in.get(v, 0) + 1
            # 3. ring row and chain heads
            ap_minute(p)
            ap_src(u)
            ap_dst(v)
            ap_usd_c(c)
            ap_l(x)
            ap_fmt(f)
            ap_flags(fl)
            ap_pid(pid)
            ap_prev_out(head_out[u])
            ap_prev_in(head_in[v])
            ap_pmax_out(pmo)
            ap_pmax_in(pmi)
            ring.end_rank = r + 1
            head_out[u] = r
            head_in[v] = r
            # 4-5. every slot and per-window pair count
            add(u, v, c, x, fl, f, pid)
            # 6. last minutes
            last_min[pid] = p
            last_out[u] = p
            last_in[v] = p
        for a, n in bump_out.items():  # _commit_ports
            ever_out[a] += n
        for a, n in bump_in.items():
            ever_in[a] += n
        bump_out.clear()
        bump_in.clear()

    return apply


# --- scoring --------------------------------------------------------------------------------------


def _make_scorer(eng: Engine) -> Callable[[tuple], tuple[tuple, int]]:
    """score(ev) -> (row, new): the read-only `_score` of §4.8 (it may fill the minute memo)."""
    spec = eng.spec
    S, L = spec.w_short, spec.w_long
    get = eng.slots.get
    oc_s, oc_l, ic_s, ic_l = (get(d, "cnt", w) for d in ("out", "in") for w in (S, L))
    ou_s, ou_l, iu_s, iu_l = (get(d, "uniq", w) for d in ("out", "in") for w in (S, L))
    os_s, os_l, is_s, is_l = (get(d, "sum_c", w) for d in ("out", "in") for w in (S, L))
    o1_s, o1_l, i1_s, i1_l = (get(d, "s1", w) for d in ("out", "in") for w in (S, L))
    o2_s, o2_l, i2_s, i2_l = (get(d, "s2", w) for d in ("out", "in") for w in (S, L))
    inflow_a = get("in", "nsl_sum_c", spec.w_pt)
    inband_s, round_s, newcp_s = (get("out", kind, S) for kind in ("inband", "round", "newcp"))
    n_fmt = len(spec.vocab["payment_format"])
    fmt_out = tuple(get("out", "fmt", S, k) for k in range(n_fmt))
    fmt_in = tuple(get("in", "fmt", S, k) for k in range(n_fmt))
    # rule support, each at its scenario's own window
    fan_in, fan_out = get("in", "uniq", spec.w_fan_in), get("out", "uniq", spec.w_fan_out)
    st_a, ra_a = get("out", "inband", spec.w_struct), get("out", "round", spec.w_round)
    hr_a = get("out", "hr", spec.w_hr)
    pairs = eng.pairs
    index_get, last_min = pairs.index.get, pairs.last_min
    pc_s, pc_l = pairs.pcnt[S], pairs.pcnt[L]
    port_out, port_in = pairs.port_out, pairs.port_in
    last_out, last_in = eng.last_out, eng.last_in
    ever_out, ever_in = eng.ever_out, eng.ever_in
    head_out, head_in, hub = eng.head_out, eng.head_in, eng.hub
    ring, memo = eng.ring, eng.memo
    r_minute, r_l, r_pmax_out, r_pmax_in = ring.minute, ring.l, ring.pmax_out, ring.pmax_in
    cap_count, cap_port, cap_gap = spec.cap_count, spec.cap_port, spec.cap_gap
    P, excl = spec.sql_params, spec.excl
    limit = EXACT_INT_LIMIT
    nan, log1p, sqrt, fl = math.nan, math.log1p, math.sqrt, float
    f_in_band, f_round, f_hr, f_cross, f_same = F_IN_BAND, F_ROUND, F_HR, F_CROSS, F_SAME_BANK
    cyc, scen = _cycles, _scenarios  # module attributes are read per call (tests may patch them)

    def score(ev: tuple) -> tuple[tuple, int]:
        _, _, m, u, v, c, x, _, _, f, pcur, rcur, flags, hour = ev
        # windowed counts (VEL)
        uoc_s, uoc_l, uou_s, uou_l = oc_s[u], oc_l[u], ou_s[u], ou_l[u]
        uic_s, uic_l, uiu_s, uiu_l = ic_s[u], ic_l[u], iu_s[u], iu_l[u]
        vic_s, vic_l, viu_s, viu_l = ic_s[v], ic_l[v], iu_s[v], iu_l[v]
        voc_s, voc_l, vou_s, vou_l = oc_s[v], oc_l[v], ou_s[v], ou_l[v]
        # cents sums: exact ints, converted to float only below 2^53 (an OR of non-negative ints
        # is >= 2^53 iff one of them is)
        uos_s, uos_l, uis_s, uis_l = os_s[u], os_l[u], is_s[u], is_l[u]
        vis_s, vis_l, vos_s, vos_l = is_s[v], is_l[v], os_s[v], os_l[v]
        inflow = inflow_a[u]
        u_tot, v_tot = uos_l + uis_l, vos_l + vis_l
        if (
            uos_s | uos_l | uis_s | uis_l | vis_s | vis_l | vos_s | vos_l | inflow | u_tot | v_tot
        ) >= limit:
            raise NumericRangeError(f"a windowed cents sum reached 2^53 at minute {m} ({u}, {v})")
        # log-amount moments: mean = s1/n, var = s2/n - mean^2, std = sqrt(max(var, 0))
        if uoc_s:
            uo_mean_s = o1_s[u] / uoc_s
            var = o2_s[u] / uoc_s - uo_mean_s * uo_mean_s
            uo_std_s = sqrt(var) if var > 0.0 else 0.0
        else:
            uo_mean_s = uo_std_s = nan
        if uoc_l:
            uo_mean_l = o1_l[u] / uoc_l
            var = o2_l[u] / uoc_l - uo_mean_l * uo_mean_l
            uo_std_l = sqrt(var) if var > 0.0 else 0.0
        else:
            uo_mean_l = uo_std_l = nan
        if vic_s:
            vi_mean_s = i1_s[v] / vic_s
            var = i2_s[v] / vic_s - vi_mean_s * vi_mean_s
            vi_std_s = sqrt(var) if var > 0.0 else 0.0
        else:
            vi_mean_s = vi_std_s = nan
        if vic_l:
            vi_mean_l = i1_l[v] / vic_l
            var = i2_l[v] / vic_l - vi_mean_l * vi_mean_l
            vi_std_l = sqrt(var) if var > 0.0 else 0.0
        else:
            vi_mean_l = vi_std_l = nan
        # window max over the short window: walk the suffix-max chain from the head (§4.6)
        lo = m - S
        base = ring.base
        p = head_out[u]
        if p >= base and r_minute[p - base] >= lo:
            q = r_pmax_out[p - base]
            while q >= base and r_minute[q - base] >= lo:
                p = q
                q = r_pmax_out[q - base]
            uo_max = r_l[p - base]
        else:
            uo_max = nan
        p = head_in[v]
        if p >= base and r_minute[p - base] >= lo:
            q = r_pmax_in[p - base]
            while q >= base and r_minute[q - base] >= lo:
                p = q
                q = r_pmax_in[q - base]
            vi_max = r_l[p - base]
        else:
            vi_max = nan
        # pair, ports, gaps (lifetime state; a real gap is >= 1, 0 = no history)
        pid = index_get((u << 32) | v)
        if pid is None:
            new, pc1, pc3, po, pi, pgap = 1, 0, 0, ever_out[u], ever_in[v], 0
        else:
            new, pc1, pc3, po, pi = 0, pc_s[pid], pc_l[pid], port_out[pid], port_in[pid]
            pgap = m - last_min[pid]
            if pgap > cap_gap:
                pgap = cap_gap
        rpid = index_get((v << 32) | u)
        if rpid is None:
            rgap = 0
        else:
            rgap = m - last_min[rpid]
            if rgap > cap_gap:
                rgap = cap_gap
        t = last_out[u]
        g_uo = 0 if t < 0 else (m - t if m - t < cap_gap else cap_gap)
        t = last_in[u]
        g_ui = 0 if t < 0 else (m - t if m - t < cap_gap else cap_gap)
        t = last_in[v]
        g_vi = 0 if t < 0 else (m - t if m - t < cap_gap else cap_gap)
        t = last_out[v]
        g_vo = 0 if t < 0 else (m - t if m - t < cap_gap else cap_gap)
        # paths (backward search with the per-minute memo) and scatter-gather
        if u != v:
            self_loop = 0
            c2, c3, c4, rule_trunc, cyc_trunc = cyc.path_counts(
                memo, ring, head_in, hub, spec, m, u, v
            )
            sg_mids, sg_srcs, sg_trunc = cyc.sg_counts(memo, ring, head_in, hub, spec, m, u, v)
            pt = fl(c) / fl(inflow) if inflow > 0 else nan
        else:
            self_loop = 1
            c2 = c3 = c4 = rule_trunc = cyc_trunc = sg_mids = sg_srcs = sg_trunc = 0
            pt = nan
        in_band = 1 if flags & f_in_band else 0
        is_round = 1 if flags & f_round else 0
        sev = scen.severities(
            (
                u,
                v,
                self_loop,
                c,
                in_band,
                is_round,
                1 if flags & f_hr else 0,
                hub[u],
                fan_in[v],
                fan_out[u],
                inflow,
                c2,
                c3,
                st_a[u],
                ra_a[u],
                hr_a[u],
            ),
            P,
            excl,
        )
        row = (
            # TX
            x,
            fl(pcur),
            fl(rcur),
            1.0 if flags & f_cross else 0.0,
            fl(f),
            fl(self_loop),
            1.0 if flags & f_same else 0.0,
            fl(is_round),
            fl(hour),
            # VEL
            fl(uoc_s),
            fl(uoc_l),
            fl(uou_s),
            fl(uou_l),
            fl(uic_s),
            fl(uic_l),
            fl(uiu_s),
            fl(uiu_l),
            fl(vic_s),
            fl(vic_l),
            fl(viu_s),
            fl(viu_l),
            fl(voc_s),
            fl(voc_l),
            fl(vou_s),
            fl(vou_l),
            # AMT: lsum(cents) = log1p(cents / 100) of the exact int
            log1p(uos_s / 100),
            log1p(uos_l / 100),
            log1p(uis_s / 100),
            log1p(uis_l / 100),
            log1p(vis_s / 100),
            log1p(vis_l / 100),
            log1p(vos_s / 100),
            log1p(vos_l / 100),
            uo_mean_s,
            uo_mean_l,
            vi_mean_s,
            vi_mean_l,
            uo_std_s,
            uo_std_l,
            vi_std_s,
            vi_std_l,
            uo_max,
            vi_max,
            x - uo_mean_s,
            x - vi_mean_s,
            # FLOW
            fl(pc1),
            fl(pc3),
            log1p(inflow / 100),
            pt,
            (uos_l - uis_l) / u_tot if u_tot else nan,
            (vos_l - vis_l) / v_tot if v_tot else nan,
            # PORT
            fl(new),
            log1p(po if po < cap_port else cap_port),
            log1p(pi if pi < cap_port else cap_port),
            fl(g_uo),
            fl(g_ui),
            fl(g_vi),
            fl(g_vo),
            fl(pgap),
            fl(rgap),
            # CYC
            fl(c2 if c2 < cap_count else cap_count),
            fl(c3 if c3 < cap_count else cap_count),
            fl(c4 if c4 < cap_count else cap_count),
            # SG
            fl(sg_mids),
            fl(sg_srcs),
            fl(uiu_s if uiu_s < uou_s else uou_s),
            fl(viu_s if viu_s < vou_s else vou_s),
            # RULE
            fl(in_band),
            fl(inband_s[u]),
            fl(round_s[u]),
            fl(newcp_s[u]),
            *[fl(a[u]) for a in fmt_out],
            fl(fmt_in[f][v]) if f >= 0 else 0.0,
            # tail: severities, inflow_c, trunc flags
            *sev,
            inflow,
            rule_trunc,
            cyc_trunc,
            sg_trunc,
        )
        return row, new

    return score


# --- invariants (tests) ---------------------------------------------------------------------------


def _check_invariants(eng: Engine) -> None:
    """Recount every aggregate from the ring and the pair table (slow; tests only)."""
    spec, ring, pairs = eng.spec, eng.ring, eng.pairs
    base, live, end = ring.base, ring.live_start, ring.end_rank
    clock, n = eng.clock, spec.n_accounts

    def check(ok: bool, what: str) -> None:
        if not ok:
            raise AssertionError(f"engine invariant broken: {what}")

    check(base <= live <= end, f"base {base} <= live_start {live} <= end_rank {end}")
    for name, col in ring.columns():
        check(len(col) == end - base, f"ring column {name} length")
    minute = ring.minute
    rows = range(end - base)
    check(all(minute[i - 1] <= minute[i] for i in range(1, len(rows))), "ring minutes sorted")
    if clock is None:
        check(end == 0 and not eng.pending and eng.next_rank == 0, "fresh engine state")
        return
    check(all(minute[i] <= clock - 1 for i in rows), "ring rows are earlier than the clock")
    # pending: the current minute, contiguous ranks after the applied ones
    for j, (ev, _) in enumerate(eng.pending):
        check(ev[E_RANK] == end + j and ev[E_MINUTE] == clock, f"pending event {j}")
    check(eng.next_rank == end + len(eng.pending), "next_rank")
    check(not eng.bump_out and not eng.bump_in, "port bumps committed")
    # cursors: rows below cursor[W] are older than clock - W, rows from it are not
    check(eng.cursors[spec.W_max] == live, "cursor[W_max] == live_start")
    for w, cur in eng.cursors.items():
        check(live <= cur <= end, f"cursor[{w}] range")
        check(all(minute[i] < clock - w for i in range(cur - base)), f"cursor[{w}] expired rows")
        check(all(minute[i] >= clock - w for i in range(cur - base, end - base)), f"cursor[{w}]")
    # pair table
    keys, index = pairs.keys, pairs.index
    check(len(index) == len(keys), "pair index size")
    check(all(index.get(key) == p for p, key in enumerate(keys)), "pair index")
    for name, col in pairs.columns():
        check(len(col) == len(keys), f"pair column {name} length")
    eo = Counter(key >> 32 for key in keys)
    ei = Counter(key & 0xFFFFFFFF for key in keys)
    check(all(eng.ever_out[a] == eo.get(a, 0) for a in range(n)), "ever_out = distinct receivers")
    check(all(eng.ever_in[a] == ei.get(a, 0) for a in range(n)), "ever_in = distinct senders")
    src, dst, pid, flags = ring.src, ring.dst, ring.pid, ring.flags
    for i in rows:
        u, v = src[i], dst[i]
        check(keys[pid[i]] == (u << 32) | v, f"ring row {base + i} pair id")
        check(bool(flags[i] & F_SELF) == (u == v), f"ring row {base + i} self-loop flag")
    # chains, heads and last minutes over the stored rows
    for side, key_col, head, prev, last in (
        ("out", src, eng.head_out, ring.prev_out, eng.last_out),
        ("in", dst, eng.head_in, ring.prev_in, eng.last_in),
    ):
        newest: dict[int, int] = {}
        for i in rows:
            a = key_col[i]
            want = newest.get(a, -1)
            got = prev[i]
            check(got == want if want >= 0 else got < base, f"prev_{side}[{base + i}]")
            newest[a] = base + i
        for a in range(n):
            h = head[a]
            if a in newest:
                check(h == newest[a], f"head_{side}[{a}]")
            else:
                check(h < base, f"head_{side}[{a}] is not stored")
            if h >= base:
                check(last[a] == minute[h - base], f"last_{side}[{a}]")
            check((h == -1) == (last[a] == -1), f"last_{side}[{a}] / head sentinel")
    pair_last: dict[int, int] = {}
    for i in rows:
        pair_last[pid[i]] = minute[i]
    check(all(pairs.last_min[p] == t for p, t in pair_last.items()), "pair last_min")
    check(all(t <= clock - 1 for t in pairs.last_min), "pair last_min is earlier than the clock")
    # windowed aggregates, recounted per window from rows [cursor[W], end)
    arrays = eng.slots.arrays
    usd_c, l_col, fmt = ring.usd_c, ring.l, ring.fmt
    value = {
        "cnt": lambda i: 1,
        "sum_c": lambda i: usd_c[i],
        "s1": lambda i: l_col[i],
        "s2": lambda i: l_col[i] * l_col[i],
        "nsl_sum_c": lambda i: 0 if flags[i] & F_SELF else usd_c[i],
    } | {kind: (lambda i, b=bit: 1 if flags[i] & b else 0) for kind, bit in _FLAG_KINDS.items()}
    for w in spec.windows_all:
        live_rows = range(eng.cursors[w] - base, end - base)
        pcounts = Counter(pid[i] for i in live_rows)
        if w in pairs.pcnt:
            col = pairs.pcnt[w]
            check(all(col[p] == pcounts.get(p, 0) for p in range(len(keys))), f"pcnt[{w}]")
        for s, a in arrays.items():
            if s.window != w:
                continue
            keyed = src if s.side == "out" else dst
            want: dict[int, Any] = defaultdict(int)
            if s.kind == "uniq":
                for p in pcounts:
                    key = keys[p]
                    want[key >> 32 if s.side == "out" else key & 0xFFFFFFFF] += 1
            elif s.kind == "fmt":
                for i in live_rows:
                    want[keyed[i]] += fmt[i] == s.pred
            else:
                for i in live_rows:
                    want[keyed[i]] += value[s.kind](i)
            for acct in range(n):
                got, exp = a[acct], want.get(acct, 0)
                if s.kind in ("s1", "s2"):
                    cnt = arrays[Slot(s.side, "cnt", w)][acct]
                    ok = got == 0.0 if cnt == 0 else abs(got - exp) <= 1e-7 * max(1.0, abs(exp))
                else:
                    ok = got == exp
                check(ok, f"{s.name}[{acct}] = {got!r}, recount {exp!r}")
    # window max (short window) for every account with live rows
    lo = clock - spec.w_short
    for side, key_col, head, pmax in (
        ("out", src, eng.head_out, ring.pmax_out),
        ("in", dst, eng.head_in, ring.pmax_in),
    ):
        best: dict[int, float] = {}
        for i in rows:
            if minute[i] >= lo:
                a = key_col[i]
                best[a] = max(best.get(a, -math.inf), l_col[i])
        for a in range(n):
            got = window_max(ring, head, pmax, a, lo)
            check(got == best[a] if a in best else math.isnan(got), f"max_{side}[{a}]")
