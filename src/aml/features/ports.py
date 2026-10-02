"""The lifetime pair table: pair ids, per-window pair counts, ports, last minutes (M2 §4.3 D, §4.5).

A pair (u, v) gets id 0, 1, 2, ... in first-apply order (self-loop pairs included); key
`pair_key(u, v) = (u << 32) | v`. Pairs are never deleted: ports and `pair_is_new` need the
lifetime relation. Parallel columns by pid (`spec.PAIR_COLUMNS` + one `pcnt[W]` per uniq window):

- `pcnt[W][pid]`: applied u->v events in the window W; 0 <-> 1 transitions drive uniq_out/uniq_in.
- `port_out[pid]`: distinct receivers of u in minutes strictly before the pair's first minute
  (= ever_out[u] at the start of that minute); `port_in[pid]` likewise for v's senders. All new
  counterparties of one minute share a port.
- `last_min[pid]`: the pair's last applied minute (pair_gap; rev_pair_gap reads key (v, u)).

The engine updates `pcnt` and `last_min` in place and reads `index` directly in its hot paths;
`add` is the only way a pair is created.
"""

from __future__ import annotations

import sys
from array import array
from collections.abc import Mapping

from aml.features.spec import I32_LIMIT, NumericRangeError


def pair_key(u: int, v: int) -> int:
    """Dict key of the ordered pair (u, v): (u << 32) | v (account ids < 2^31 - 1)."""
    return (u << 32) | v


def capped_gap(minute: int, last: int, cap: int) -> int:
    """min(minute - last, cap) if last >= 0 else 0 (0 = no history; a real gap is >= 1)."""
    if last < 0:
        return 0
    gap = minute - last
    return gap if gap < cap else cap


class PairTable:
    """Lifetime pair relation with per-window counts (attribute names are the snapshot's)."""

    __slots__ = ("index", "keys", "last_min", "pcnt", "port_in", "port_out", "windows")

    def __init__(self, uniq_windows: tuple[int, ...]) -> None:
        self.windows = tuple(uniq_windows)
        self.index: dict[int, int] = {}  # pair_key -> pid
        self.keys = array("q")  # pid order
        self.pcnt: dict[int, array] = {w: array("i") for w in self.windows}
        self.port_out = array("i")
        self.port_in = array("i")
        self.last_min = array("i")

    @classmethod
    def from_columns(
        cls,
        uniq_windows: tuple[int, ...],
        keys: array,
        pcnt: Mapping[int, array],
        port_out: array,
        port_in: array,
        last_min: array,
    ) -> PairTable:
        """A table from saved columns (a restored snapshot); the index is rebuilt from `keys`."""
        t = cls(uniq_windows)
        n = len(keys)
        cols = {"port_out": port_out, "port_in": port_in, "last_min": last_min}
        cols |= {f"pcnt.{w}": pcnt[w] for w in t.windows}
        if keys.typecode != "q" or set(pcnt) != set(t.windows):
            raise ValueError("pair table columns do not match the uniq windows")
        for name, col in cols.items():
            if col.typecode != "i" or len(col) != n:
                raise ValueError(f"pair column {name!r}: expected {n} items of 'i'")
        t.index = dict(zip(keys, range(n), strict=True))
        if len(t.index) != n:
            raise ValueError("pair table keys repeat")
        t.keys = keys
        t.pcnt = {w: pcnt[w] for w in t.windows}
        t.port_out, t.port_in, t.last_min = port_out, port_in, last_min
        return t

    @property
    def n_pairs(self) -> int:
        return len(self.keys)

    def pid(self, u: int, v: int) -> int:
        """The pair id of (u, v), or -1 if the pair has no applied event."""
        return self.index.get((u << 32) | v, -1)

    def add(self, u: int, v: int, port_out: int, port_in: int, minute: int) -> int:
        """Register a new pair (pcnt = 0 in every window) and return its id."""
        key = (u << 32) | v
        if key in self.index:
            raise ValueError(f"pair ({u}, {v}) already exists")
        pid = len(self.keys)
        if pid >= I32_LIMIT:
            raise NumericRangeError(f"pair id {pid} does not fit array('i')")
        self.index[key] = pid
        self.keys.append(key)
        for col in self.pcnt.values():
            col.append(0)
        self.port_out.append(port_out)
        self.port_in.append(port_in)
        self.last_min.append(minute)
        return pid

    def columns(self) -> tuple[tuple[str, array], ...]:
        """(component name, array) in snapshot order: keys, pcnt.<W>..., port_out, port_in,
        last_min."""
        return (
            ("keys", self.keys),
            *((f"pcnt.{w}", self.pcnt[w]) for w in self.windows),
            ("port_out", self.port_out),
            ("port_in", self.port_in),
            ("last_min", self.last_min),
        )

    def nbytes(self) -> int:
        """Bytes of the columns plus an estimate of the index dict (its table and int objects)."""
        cols = sum(sys.getsizeof(a) for _, a in self.columns())
        n = len(self.keys)
        # Each entry holds a key int (> 2^30: 32 B) and a pid int (28 B; ids < 257 are cached).
        return cols + sys.getsizeof(self.index) + n * (32 + 28)
