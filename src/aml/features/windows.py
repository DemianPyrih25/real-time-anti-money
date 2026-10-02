"""Windowed engine state: the ring of applied events and the per-account slot arrays (M2 §4.3-§4.6).

The attribute names below are a cross-module contract: `features.cycles` walks a `Ring` through
`spec.RingView`, and the snapshot names components after them (`ring.<col>`, `Slot.name`).

Ring (§4.3 A): columnar and append-only, index = rank - base; one `array.array` attribute per
`spec.RING_COLUMNS` entry (50 B/row). Rows enter only when `Engine.advance` applies the pending
minute, in rank order. `live_start` = cursor[W_max]. Compaction (only inside advance): when
`live_start - base >= max(compact_min_rows, (end_rank - live_start) // 2)`, `del col[:live_start -
base]` on every column and `base = live_start`. Pointer values (prev_*, pmax_*) are absolute ranks,
so compaction never rewrites them; every walk checks `r >= base` first. The column arrays are
never replaced (only appended to and trimmed in place), so code may bind them once.

Slots (§4.3 B, §4.4): one per-account array per `spec.slots` entry, typecode `Slot.typecode`,
length n_accounts. Out slots are keyed by src, in slots by dst.

Window max (§4.6): `pmax_out[r]` = rank of the latest earlier out-event of the same src with a
strictly larger l, else -1; insertion stops at `live_start` so an uninterrupted engine and a
restored one hold identical pointers. Query = walk from the head while minute >= m - W.
"""

from __future__ import annotations

import math
import sys
from array import array
from collections.abc import Mapping

from aml.features.spec import RING_COLUMNS, EngineSpec, Slot

RING_COLUMN_NAMES = tuple(name for name, _ in RING_COLUMNS)


def filled(typecode: str, n: int, value: int | float = 0) -> array:
    """A new array of n copies of `value` (zero-filled from bytes when value is 0)."""
    if value == 0:
        return array(typecode, bytes(array(typecode).itemsize * n))
    return array(typecode, [value]) * n


class Ring:
    """Columnar ring of applied events; implements `spec.RingView`.

    Attributes (the contract): `base`, `live_start`, `end_rank` (ints) and the columns `minute`,
    `src`, `dst`, `usd_c`, `l`, `fmt`, `flags`, `pid`, `prev_out`, `prev_in`, `pmax_out`,
    `pmax_in` (`array.array`, typecodes from `spec.RING_COLUMNS`).

    `Engine.advance` appends through the columns' bound `append` methods (the per-event hot path)
    and then sets `end_rank` itself; `append` below is the checked one-row form.
    """

    __slots__ = ("base", "compact_min_rows", "end_rank", "live_start", *RING_COLUMN_NAMES)

    def __init__(self, compact_min_rows: int, start_rank: int = 0) -> None:
        """Empty ring whose first appended row gets rank `start_rank` (base = live_start = it)."""
        if compact_min_rows < 1:
            raise ValueError(f"compact_min_rows must be >= 1, got {compact_min_rows}")
        if start_rank < 0:
            raise ValueError(f"start_rank must be >= 0, got {start_rank}")
        self.compact_min_rows = int(compact_min_rows)
        self.base = self.live_start = self.end_rank = int(start_rank)
        for name, typecode in RING_COLUMNS:
            setattr(self, name, array(typecode))

    @classmethod
    def from_columns(
        cls,
        compact_min_rows: int,
        live_start: int,
        end_rank: int,
        columns: Mapping[str, array],
    ) -> Ring:
        """A ring holding rows [live_start, end_rank) (a restored snapshot): base = live_start."""
        ring = cls(compact_min_rows, live_start)
        n = end_rank - live_start
        if n < 0:
            raise ValueError(f"end_rank {end_rank} < live_start {live_start}")
        for name, typecode in RING_COLUMNS:
            col = columns[name]
            if not isinstance(col, array) or col.typecode != typecode or len(col) != n:
                raise ValueError(f"ring column {name!r}: expected {n} items of {typecode!r}")
            setattr(ring, name, col)
        ring.end_rank = end_rank
        return ring

    def columns(self) -> tuple[tuple[str, array], ...]:
        """(name, array) for every column, in `spec.RING_COLUMNS` order."""
        return tuple((name, getattr(self, name)) for name in RING_COLUMN_NAMES)

    def append(
        self,
        rank: int,
        minute: int,
        src: int,
        dst: int,
        usd_c: int,
        l: float,  # noqa: E741 - the spec's column name
        fmt: int,
        flags: int,
        pid: int,
        prev_out: int,
        prev_in: int,
        pmax_out: int,
        pmax_in: int,
    ) -> None:
        """Append one applied event; `rank` must equal `end_rank`."""
        if rank != self.end_rank:
            raise ValueError(f"ring append: rank {rank} != end_rank {self.end_rank}")
        self.minute.append(minute)
        self.src.append(src)
        self.dst.append(dst)
        self.usd_c.append(usd_c)
        self.l.append(l)
        self.fmt.append(fmt)
        self.flags.append(flags)
        self.pid.append(pid)
        self.prev_out.append(prev_out)
        self.prev_in.append(prev_in)
        self.pmax_out.append(pmax_out)
        self.pmax_in.append(pmax_in)
        self.end_rank = rank + 1

    def maybe_compact(self) -> bool:
        """Drop dead rows below `live_start` when the §4.3 rule says so; True if it compacted."""
        dead = self.live_start - self.base
        if dead <= 0 or dead < max(self.compact_min_rows, (self.end_rank - self.live_start) // 2):
            return False
        for name in RING_COLUMN_NAMES:
            del getattr(self, name)[:dead]
        self.base = self.live_start
        return True

    def __len__(self) -> int:
        """Stored rows (end_rank - base), dead rows included."""
        return self.end_rank - self.base

    def nbytes(self) -> int:
        """Bytes held by the column buffers (allocated capacity included)."""
        return sum(sys.getsizeof(getattr(self, name)) for name in RING_COLUMN_NAMES)


class Slots:
    """Per-account windowed aggregates from the registry (`spec.slots`)."""

    __slots__ = ("arrays",)

    def __init__(self, spec: EngineSpec, arrays: Mapping[Slot, array] | None = None) -> None:
        """Zeroed arrays, one per registry slot, length spec.n_accounts; or the given `arrays`
        (a restored snapshot), checked against the registry."""
        n = spec.n_accounts
        if arrays is None:
            self.arrays: dict[Slot, array] = {s: filled(s.typecode, n) for s in spec.slots}
            return
        if set(arrays) != set(spec.slots):
            raise ValueError("slot arrays do not match the spec's registry")
        for s in spec.slots:
            a = arrays[s]
            if not isinstance(a, array) or a.typecode != s.typecode or len(a) != n:
                raise ValueError(f"{s.name}: expected {n} items of {s.typecode!r}")
        self.arrays = {s: arrays[s] for s in spec.slots}

    def get(self, side: str, kind: str, window: int, pred: int = -1) -> array:
        """The array of Slot(side, kind, window, pred); KeyError if not registered."""
        try:
            return self.arrays[Slot(side, kind, window, pred)]
        except KeyError:
            raise KeyError(f"slot not registered: {Slot(side, kind, window, pred)}") from None

    def nbytes(self) -> int:
        return sum(sys.getsizeof(a) for a in self.arrays.values())


def window_max(ring: Ring, head: array, pmax: array, acct: int, lo: int) -> float:
    """Max l over the account's chain events with minute >= lo (§4.6); NaN if there are none.

    `head` / `pmax` are head_out + ring.pmax_out (out side) or head_in + ring.pmax_in (in side).
    `lo` must be >= clock - W_max (older rows may be dead or compacted). Ties keep the latest
    equal maximum. `Engine` inlines this walk in its scorer; the two are tested equal.
    """
    base = ring.base
    minute = ring.minute
    p = head[acct]
    if p < base or minute[p - base] < lo:
        return math.nan
    q = pmax[p - base]
    while q >= base and minute[q - base] >= lo:
        p = q
        q = pmax[q - base]
    return ring.l[p - base]
