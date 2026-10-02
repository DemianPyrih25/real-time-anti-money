"""The causal feature engine's contract (M2 spec §3-§7): names, layouts, registry and constants.

Everything the engine, the rules, the drivers, the models and the tests share is defined here, so
modules written separately fit together. Nothing here computes a feature.

Feature names are *generated* from the configs and the train vocabulary (`build_features`), so the
per-run layouts live on an `EngineSpec` instance:

- `spec.features` / `spec.FEATURES`: the model inputs (`FeatureDef`), in row order. At the default
  config and the 7 HI-Small payment formats there are 79:
  TX (9, M1's `TX_FEATURES`), VEL (16), AMT (20), FLOW (6), PORT (9), CYC (3), SG (4), RULE (12).
- `spec.row_layout` / `spec.ROW_LAYOUT`: the engine's output row = the feature names, then the fixed
  tail `ROW_TAIL` = the 7 severities (`SCENARIOS` order), `inflow_c`, `rule_trunc`, `cyc_trunc`,
  `sg_trunc`. The row never holds row_id / rank / day / split (the driver adds them).
- `spec.model_index(names)`: positions of model inputs in the row (M5: `row[i] for i in idx`).

Cross-module contracts pinned here (the stub docstrings refer back to them):

- The prepared event (`Engine.prepare`) is a plain tuple indexed by the `E_*` constants; flag bits
  are `F_*` (`F_NEW` is set only on the ring copy, at apply time).
- `RingView` is the read-only accessor contract of `windows.Ring` that `features.cycles` walks.
  The engine calls `cycles.path_counts` / `cycles.sg_counts` with its own ring, `head_in`, `hub`
  and `memo` (`cycles.MinuteMemo`, cleared by `advance`); see features/cycles.py.
- Severities: `rules.scenarios.severities(rs, spec.sql_params, spec.excl)` with `rs` a
  `rules.scenarios.RuleSupport` (or a plain tuple in its field order), called inside `_score`.
- `FlushStats` is what `Engine.advance` returns.
- `Slot` is one per-account windowed aggregate; `spec.slots` is the deduplicated registry (§4.4).
- `tol_ok` implements the tolerance classes of §4.9 (never a looser rule than those).
- The feature table layout on the Volume (`part_name`, `part_paths`, `scan_feature_table`,
  `spec.table_schema()`).
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import numbers
import re
from array import array
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Protocol

import numpy as np
import polars as pl

from aml.features.tx_features import CATEGORICAL_FEATURES, TX_FEATURES, is_forbidden
from aml.io import sha256_file
from aml.rules.sql_baseline import HUB_SEGMENTABLE, SCENARIOS, sql_params

# Bumped by hand on any change of engine semantics (state, features, row layout). It is part of the
# features run key and of `spec_hash`, so old feature tables, snapshots and goldens are refused.
ENGINE_VERSION = 1
SPEC_FORMAT = 1  # layout version of EngineSpec.to_json()

MINUTES_PER_DAY = 1440
# Ranks, minutes, account ids, pair ids and windows must be < I32_LIMIT (they live in array('i')).
I32_LIMIT = 2**31 - 1
# Any integer (cents sum) converted to float must be < EXACT_INT_LIMIT, else NumericRangeError.
EXACT_INT_LIMIT = 2**53

if array("i").itemsize != 4 or array("q").itemsize != 8:  # snapshots store raw array bytes
    raise RuntimeError("the engine needs a platform with 4-byte array('i') and 8-byte array('q')")

# --- exceptions ---------------------------------------------------------------------------------


class SpecError(ValueError):
    """Invalid engine configuration, or a feature spec document that does not match this code."""


class EngineError(RuntimeError):
    """Base class of the engine's runtime errors."""


class LateEventError(EngineError):
    """An event's minute is earlier than the engine clock (M5 stops on this)."""


class RankGapError(EngineError):
    """An event's rank is not the next contiguous rank."""


class NumericRangeError(EngineError):
    """A value is outside the exactly representable range (cents sum >= 2^53, id >= 2^31 - 1)."""


class SnapshotError(EngineError):
    """A snapshot is corrupt or belongs to another engine version / spec."""


class GateStopError(RuntimeError):
    """The feature gate would drop more engine features than `max_drop_share`: the user decides."""


# --- inputs, the prepared event and flags (§4.1) ------------------------------------------------

# Transactions columns the engine reads, in `Engine.prepare` argument order. No label column exists
# anywhere in the engine API.
INPUT_COLUMNS = (
    "rank",
    "row_id",
    "minute",
    "src",
    "dst",
    "amount_usd",
    "amount_paid",
    "payment_format",
    "payment_currency",
    "receiving_currency",
    "from_bank",
    "to_bank",
)
# Also read by the offline driver (never passed to the engine).
DRIVER_COLUMNS = ("day", "split")

# The prepared event: a plain tuple, indexed by these constants.
EVENT_FIELDS = (
    "rank",  # int
    "row_id",  # int (carried for the driver; never in the output row)
    "minute",  # int
    "src",  # int account id u
    "dst",  # int account id v
    "usd_c",  # int: cents(amount_usd), half away from zero (tx_features.cents)
    "l",  # float: math.log1p(amount_usd)
    "usd",  # float: amount_usd as given
    "paid_c",  # int: cents(amount_paid)
    "fmt",  # int: payment_format train-vocab code, -1 unknown
    "pcur",  # int: payment_currency code, -1 unknown
    "rcur",  # int: receiving_currency code, -1 unknown
    "flags",  # int: F_* bits (never F_NEW)
    "hour",  # int: (minute % 1440) // 60
)
E_RANK = 0
E_ROW_ID = 1
E_MINUTE = 2
E_SRC = 3
E_DST = 4
E_USD_C = 5
E_L = 6
E_USD = 7
E_PAID_C = 8
E_FMT = 9
E_PCUR = 10
E_RCUR = 11
E_FLAGS = 12
E_HOUR = 13

F_SELF = 1 << 0  # src == dst
F_IN_BAND = 1 << 1  # band_low_usd <= amount_usd < band_high_usd (sql_params floats)
F_ROUND = 1 << 2  # paid_c > 0 and paid_c % round_cents == 0
F_HR = 1 << 3  # payment_format string in high_risk_formats
F_CROSS = 1 << 4  # payment_currency string != receiving_currency string
F_SAME_BANK = 1 << 5  # from_bank string == to_bank string
F_NEW = 1 << 6  # set at scoring: the pair (u, v) had no applied event (ring rows only)
FLAG_BITS = {
    "F_SELF": F_SELF,
    "F_IN_BAND": F_IN_BAND,
    "F_ROUND": F_ROUND,
    "F_HR": F_HR,
    "F_CROSS": F_CROSS,
    "F_SAME_BANK": F_SAME_BANK,
    "F_NEW": F_NEW,
}

# --- the output row and the feature table (§4.8, §5.10) -----------------------------------------

SEVERITY_COLUMNS = SCENARIOS
INFLOW_COLUMN = "inflow_c"
TRUNC_COLUMNS = ("rule_trunc", "cyc_trunc", "sg_trunc")
# The fixed end of every row: severities (float), inflow_c (int), trunc flags (int 0/1).
ROW_TAIL = (*SEVERITY_COLUMNS, INFLOW_COLUMN, *TRUNC_COLUMNS)
# Key columns the driver adds in front of every feature-table row.
TABLE_KEY_COLUMNS = ("row_id", "rank", "day", "split")
# Never model inputs. Deliberately not features either: hub flags, severities (the hub-segmented
# ones would encode the account's own hub status), account age / first-seen, global rates, skew.
NON_MODEL_COLUMNS = (*TABLE_KEY_COLUMNS, *ROW_TAIL)

GROUPS = ("TX", "VEL", "AMT", "FLOW", "PORT", "CYC", "SG", "RULE")
ABLATION_GROUPS = GROUPS[1:]
SIDES = ("tx", "u", "v", "pair", "path")
TOL_CLASSES = ("exact", "ulp", "mean", "std")
# Value domain of a feature (for validation and the gate's binning):
#   flag     0/1, never NaN
#   code     integer >= -1 (train-vocab code, -1 unknown), never NaN
#   int      integer >= 0, never NaN (counts, gaps, hour)
#   real     finite float >= 0, never NaN (log amounts, log sums, ports)
#   nullable finite float or NaN when undefined (means, stds, maxima, deviations, ratios)
DOMAINS = ("flag", "code", "int", "real", "nullable")

# Feature-table layout inside /data/features/<dataset>/<features_key>/ (§3.5).
FEATURE_SPEC_FILE = "feature_spec.json"
VOCAB_FILE = "vocab.json"
SUMMARY_FILE = "summary.json"
PROGRESS_FILE = "progress.jsonl"
PARTS_DIR = "parts"
SNAPSHOTS_DIR = "snapshots"
# split whose first minute gets a boundary snapshot -> file name (sidecar: name + ".json")
SNAPSHOT_FILES = {"val_early": "val_boundary.snap", "test": "test_boundary.snap"}


def part_name(day: int) -> str:
    """File name of one simulated day's part: part_name(7) -> "part-d07.parquet"."""
    return f"part-d{int(day):02d}.parquet"


def part_paths(features_dir: Path) -> list[Path]:
    """The finished parts of a feature table, in day order (= rank order across parts)."""
    paths = sorted((Path(features_dir) / PARTS_DIR).glob("part-d*.parquet"))
    if not paths:
        raise FileNotFoundError(f"no feature parts under {Path(features_dir) / PARTS_DIR}")
    return paths


def scan_feature_table(features_dir: Path, columns: Iterable[str] | None = None) -> pl.LazyFrame:
    """All parts as one lazy frame in rank order (optionally only `columns`)."""
    lf = pl.scan_parquet([str(p) for p in part_paths(features_dir)])
    return lf if columns is None else lf.select(list(columns))


# Key of the feature table's content digest in the build summary and in the summaries of the
# stages that read the parts (rules_engine, lgbm_graph).
FEATURES_DIGEST = "features_digest"


def part_sha256s(features_dir: Path) -> dict[str, str]:
    """SHA-256 of every finished part file, by file name in day order."""
    return {p.name: sha256_file(p) for p in part_paths(features_dir)}


def digest_of_parts(part_hashes: Mapping[str, str]) -> str:
    """The feature table's content digest: SHA-256 over the (file name, file SHA-256) pairs."""
    h = hashlib.sha256()
    for name in sorted(part_hashes):
        h.update(f"{name} {part_hashes[name]}\n".encode())
    return h.hexdigest()


def parts_digest(features_dir: Path) -> str:
    """Content digest of a feature table's parts (run keys hash configs only: a re-replay under
    the same key with other values, e.g. an engine fix without an ENGINE_VERSION bump, changes
    this digest and not the key)."""
    return digest_of_parts(part_sha256s(features_dir))


# --- ring, per-account and pair columns (§4.3; names are also snapshot component names) ---------

# (column, array typecode) of windows.Ring, 50 bytes per row. Pointers are absolute ranks, -1 none.
RING_COLUMNS = (
    ("minute", "i"),
    ("src", "i"),
    ("dst", "i"),
    ("usd_c", "q"),
    ("l", "d"),
    ("fmt", "b"),
    ("flags", "B"),  # F_* bits incl. F_NEW
    ("pid", "i"),  # pair id of (src, dst)
    ("prev_out", "i"),  # rank of the previous applied event with the same src
    ("prev_in", "i"),  # rank of the previous applied event with the same dst
    ("pmax_out", "i"),  # suffix-max chain over the src's out-events (§4.6)
    ("pmax_in", "i"),  # suffix-max chain over the dst's in-events
)
# Unwindowed per-account arrays (engine attributes; array('i') of length n_accounts).
ACCOUNT_COLUMNS = (
    ("last_out", "i"),  # last applied minute as src, -1 none (self-loops included)
    ("last_in", "i"),  # last applied minute as dst, -1 none
    ("ever_out", "i"),  # distinct receivers ever (= the next out port)
    ("ever_in", "i"),  # distinct senders ever
    ("head_out", "i"),  # newest applied rank as src, -1 none
    ("head_in", "i"),  # newest applied rank as dst, -1 none
)
# Pair-table columns by pair id, besides `keys` (array('q'), key = (u << 32) | v, pid order) and one
# `pcnt.<W>` array('i') per uniq window W.
PAIR_COLUMNS = (("port_out", "i"), ("port_in", "i"), ("last_min", "i"))


class RingView(Protocol):
    """Read-only accessor contract of `windows.Ring`, the one `features.cycles` walks.

    - Columns are `array.array` attributes named as in RING_COLUMNS; row index = rank - base.
    - `base`: rank stored at index 0 (moves forward only at compaction, inside `Engine.advance`).
    - `live_start`: = cursor[W_max]; rows below it are expired from every window (they may still
      be stored until compaction). `end_rank`: the rank the next applied event gets.
    - Only applied events are in the ring (minute <= clock - 1); pending events never are.
    - Chains: `head_in[x]` (an engine array, passed explicitly) is the newest rank with dst x,
      then `prev_in[r - base]` the previous one, and so on, rank-descending; -1 ends a chain.
      Out-chains likewise with `head_out` / `prev_out`. Self-loops are in both chains.
    - Every walk checks `r >= base` **before** indexing (a negative index wraps silently) and stops
      at the first row older than its window bound. Nothing mutates the ring while scoring.

    Walk idiom::

        base, minute, src, prev_in = ring.base, ring.minute, ring.src, ring.prev_in
        r = head_in[u]
        while r >= base:
            i = r - base
            t = minute[i]
            if t < lo:
                break
            w = src[i]          # the sender of this in-edge of u
            ...
            r = prev_in[i]
    """

    base: int
    live_start: int
    end_rank: int
    minute: array
    src: array
    dst: array
    usd_c: array
    l: array  # noqa: E741 - the spec's column name
    fmt: array
    flags: array
    pid: array
    prev_out: array
    prev_in: array
    pmax_out: array
    pmax_in: array


class FlushStats(NamedTuple):
    """What `Engine.advance` returns (M5 reports it as the minute flush)."""

    n_applied: int  # pending events applied (the previous minute's events)
    n_expired: int  # (window, ring row) expiry steps performed, summed over all windows
    seconds: float  # wall time of the flush (a statistic only; never an input to any output)


# --- window registry (§4.4) ---------------------------------------------------------------------

# kind -> array typecode of the per-account slot array.
SLOT_TYPECODES = {
    "cnt": "i",  # +1 per event
    "uniq": "i",  # distinct counterparties, driven by the pair table's pcnt[W] 0 <-> 1 transitions
    "sum_c": "q",  # + usd_c
    "s1": "d",  # + l
    "s2": "d",  # + l * l (the same double at apply and expire)
    "nsl_sum_c": "q",  # + usd_c if not F_SELF
    "inband": "i",  # +1 if F_IN_BAND
    "round": "i",  # +1 if F_ROUND
    "hr": "i",  # +1 if F_HR
    "newcp": "i",  # +1 if F_NEW (new pair when scored)
    "fmt": "i",  # +1 if the event's fmt code == pred (unknown -1 matches nothing)
}
NO_PRED = -1


class Slot(NamedTuple):
    """One per-account windowed aggregate: out slots are keyed by src, in slots by dst."""

    side: str  # "out" | "in"
    kind: str  # a SLOT_TYPECODES key
    window: int  # minutes W: the slot holds events with minute in [m - W, m - 1]
    pred: int = NO_PRED  # the vocab format index for kind "fmt"; NO_PRED otherwise

    @property
    def typecode(self) -> str:
        return SLOT_TYPECODES[self.kind]

    @property
    def name(self) -> str:
        """Snapshot component name: slot.<side>.<kind>.<window>[.<pred>]."""
        tail = "" if self.pred == NO_PRED else f".{self.pred}"
        return f"slot.{self.side}.{self.kind}.{self.window}{tail}"


def build_slots(
    *,
    short: int,
    long: int,
    n_formats: int,
    fan_in: int,
    fan_out: int,
    pass_through: int,
    structuring: int,
    round_burst: int,
    high_risk: int,
) -> tuple[Slot, ...]:
    """The sorted, deduplicated slot set registered by the features and the rules (§4.4).

    A uniq window keeps both sides (`uniq_out[W]`, `uniq_in[W]`): one pcnt[W] transition updates
    both. Rule and feature windows need not be equal; equal ones share slots.
    """
    slots: set[Slot] = set()
    for w in (short, long):  # features, both windows
        for side in ("out", "in"):
            for kind in ("cnt", "uniq", "sum_c", "s1", "s2"):
                slots.add(Slot(side, kind, w))
    for kind in ("inband", "round", "newcp"):  # features, short window
        slots.add(Slot("out", kind, short))
    for k in range(n_formats):
        slots.add(Slot("out", "fmt", short, k))
        slots.add(Slot("in", "fmt", short, k))
    for w in (fan_in, fan_out):  # rules
        slots.add(Slot("out", "uniq", w))
        slots.add(Slot("in", "uniq", w))
    slots.add(Slot("in", "nsl_sum_c", pass_through))
    slots.add(Slot("out", "inband", structuring))
    slots.add(Slot("out", "round", round_burst))
    slots.add(Slot("out", "hr", high_risk))
    return tuple(sorted(slots))


# --- feature definitions (§5) -------------------------------------------------------------------


@dataclass(frozen=True)
class FeatureDef:
    """Metadata of one model input. Fields after `doc` are additive machine-readable metadata.

    stat: what is computed (e.g. "cnt", "uniq", "sum", "mean", "gap", "cyc3", "fmt_cnt"; TX features
    use their own name); direction: "out" / "in" for per-account aggregates, else None; fmt: the
    vocab format index of a per-format count, else None; domain: a DOMAINS value.
    """

    name: str
    group: str
    side: str
    window: int | None
    tol: str
    dtype: str = "f32"
    model_input: bool = True
    categorical: bool = False
    format_derived: bool = False
    gnn_edge_attr: bool = False
    doc: str = ""
    stat: str = ""
    direction: str | None = None
    fmt: int | None = None
    domain: str = "real"

    def to_json(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def window_tag(minutes: int) -> str:
    """Window length in minutes -> name tag: 720 -> "12h", 1440 -> "1d", 90 -> "90m"."""
    w = _int("window", minutes, 1)
    if w % MINUTES_PER_DAY == 0:
        return f"{w // MINUTES_PER_DAY}d"
    if w % 60 == 0:
        return f"{w // 60}h"
    return f"{w}m"


_NON_ALNUM = re.compile(r"[^0-9a-z]")


def format_slug(fmt: str) -> str:
    """Payment format -> name slug: lower-case, non-alphanumerics -> "_" ("Credit Card" -> ...)."""
    return _NON_ALNUM.sub("_", str(fmt).lower())


_TX_META = {  # name -> (tol, domain, categorical, format_derived, doc)
    "log_amount_usd": ("ulp", "real", False, False, "log1p(amount_usd)"),
    "payment_currency": ("exact", "code", True, False, "train-vocab code, -1 unknown"),
    "receiving_currency": ("exact", "code", True, False, "train-vocab code, -1 unknown"),
    "cross_currency": ("exact", "flag", False, False, "payment currency != receiving currency"),
    "payment_format": ("exact", "code", True, True, "train-vocab code, -1 unknown"),
    "self_loop": ("exact", "flag", False, False, "src == dst"),
    "same_bank": ("exact", "flag", False, False, "from_bank == to_bank (bank code strings)"),
    "round_amount": ("exact", "flag", False, False, "paid amount a whole multiple of round_unit"),
    "hour_of_day": ("exact", "int", False, False, "(minute % 1440) // 60"),
}


def _feature_defs(
    *, short: int, long: int, sg: int, pass_through: int, round_trip: int, formats: tuple[str, ...]
) -> tuple[FeatureDef, ...]:
    """The model inputs in row order (§5). Windows are already validated minutes."""
    if short == long:
        raise SpecError(f"windows.short and windows.long must differ, both are {short}")
    if tuple(TX_FEATURES) != tuple(_TX_META):
        raise RuntimeError("M1 TX_FEATURES changed: update _TX_META")
    S, L, P = window_tag(short), window_tag(long), window_tag(pass_through)
    R, G = window_tag(round_trip), window_tag(sg)
    both = ((short, S), (long, L))
    out: list[FeatureDef] = []

    def add(name, group, side, window, tol, stat, direction, domain, doc, **kw) -> None:
        out.append(FeatureDef(name, group, side, window, tol, doc=doc, stat=stat,
                              direction=direction, domain=domain, **kw))  # fmt: skip

    # TX: the transaction itself (M1 definitions and order); all are GNN edge attributes.
    for name, (tol, domain, cat, fmt_d, doc) in _TX_META.items():
        add(name, "TX", "tx", None, tol, name, None, domain, doc, categorical=cat,
            format_derived=fmt_d, gnn_edge_attr=True)  # fmt: skip

    # VEL: velocity and fan, both windows.
    for side, d, stat, doc in (
        ("u", "out", "cnt", "sender burst"),
        ("u", "out", "uniq", "fan-out (smurfing): distinct receivers"),
        ("u", "in", "cnt", "u collecting before paying on"),
        ("u", "in", "uniq", "u as a gather point / mule: distinct senders"),
        ("v", "in", "cnt", "receiver burst"),
        ("v", "in", "uniq", "fan-in (collection): distinct senders"),
        ("v", "out", "cnt", "v forwarding"),
        ("v", "out", "uniq", "v scattering onwards: distinct receivers"),
    ):
        for w, t in both:
            add(f"{side}_{d}_{stat}_{t}", "VEL", side, w, "exact", stat, d, "int", doc)

    # AMT: amount statistics (l = log1p(amount_usd); sums in usd cents).
    for side, d in (("u", "out"), ("u", "in"), ("v", "in"), ("v", "out")):
        for w, t in both:
            add(f"{side}_{d}_sum_{t}", "AMT", side, w, "ulp", "sum", d, "real", "log1p(usd sum)")
    for stat, doc in (("mean", "mean of l (typical ticket)"), ("std", "population std of l")):
        for side, d in (("u", "out"), ("v", "in")):
            for w, t in both:
                add(f"{side}_{d}_{stat}_{t}", "AMT", side, w, stat, stat, d, "nullable", doc)
    for side, d in (("u", "out"), ("v", "in")):
        add(f"{side}_{d}_max_{S}", "AMT", side, short, "ulp", "max", d, "nullable", "largest l")
    for side, d in (("u", "out"), ("v", "in")):
        doc = f"l - {side}_{d}_mean_{S} (an unusual amount)"
        add(f"{side}_amt_dev_{S}", "AMT", side, short, "mean", "amt_dev", d, "nullable", doc)

    # FLOW: pair repetition and flow.
    for w, t in both:
        add(f"pair_cnt_{t}", "FLOW", "pair", w, "exact", "pair_cnt", None, "int", "earlier u->v")
    doc = "log1p(non-self-loop usd into u), the pass-through rule's inflow"
    add(f"u_inflow_{P}", "FLOW", "u", pass_through, "ulp", "inflow", "in", "real", doc)
    doc = "usd_c / inflow_c: forwards what just came in"
    add(f"pt_ratio_{P}", "FLOW", "u", pass_through, "exact", "pt_ratio", None, "nullable", doc)
    for side in ("u", "v"):
        doc = "(out - in) / (out + in) in usd cents"
        add(f"{side}_bal_{L}", "FLOW", side, long, "exact", "bal", None, "nullable", doc)

    # PORT: ports and time deltas (lifetime; gaps capped, 0 = no history). GNN edge attributes.
    edge = {"gnn_edge_attr": True}
    doc = "(u, v) has no applied event"
    add("pair_is_new", "PORT", "pair", None, "exact", "pair_is_new", None, "flag", doc, **edge)
    for name, d, who in (("out_port", "out", "u's receivers"), ("in_port", "in", "v's senders")):
        doc = f"log1p(min({who} before the pair's first minute, cap))"
        add(name, "PORT", "pair", None, "ulp", "port", d, "real", doc, **edge)
    for side, d in (("u", "out"), ("u", "in"), ("v", "in"), ("v", "out")):
        doc = "minutes since the side's last earlier-minute event, capped; 0 = none"
        add(f"{side}_{d}_gap", "PORT", side, None, "exact", "gap", d, "int", doc, **edge)
    for name, doc in (("pair_gap", "u->v, 0 = new"), ("rev_pair_gap", "v->u, 0 = never")):
        doc = f"minutes since the last {doc}, capped"
        add(name, "PORT", "pair", None, "exact", name, None, "int", doc, **edge)

    # CYC: temporal cycles closing on the new edge (the round-trip rule's window and hop).
    for k in (2, 3, 4):
        doc = f"earlier {k}-edge paths v -> ... -> u, capped"
        add(f"cyc{k}_{R}", "CYC", "path", round_trip, "exact", f"cyc{k}", None, "int", doc)

    # SG: scatter-gather and gather-scatter.
    doc = "siblings x of u fed by a common non-hub source that pay v"
    add(f"sg_mids_{G}", "SG", "path", sg, "exact", "sg_mids", None, "int", doc)
    doc = "distinct sources of those siblings"
    add(f"sg_srcs_{G}", "SG", "path", sg, "exact", "sg_srcs", None, "int", doc)
    for side in ("u", "v"):
        doc = f"min({side}_in_uniq_{S}, {side}_out_uniq_{S}): gathers then scatters"
        add(f"gs_{side}_{S}", "SG", side, short, "exact", "gs", None, "int", doc)

    # RULE: rule support, so every scenario becomes a predicate on engine output.
    add("in_band", "RULE", "tx", None, "exact", "in_band", None, "flag", "in [band_low * T, T)")
    for kind, doc in (("inband", "in-band"), ("round", "round"), ("newcp", "new-pair")):
        doc = f"u's {doc} payments"
        add(f"u_out_{kind}_{S}", "RULE", "u", short, "exact", f"{kind}_cnt", "out", "int", doc)
    fmt_derived = {"format_derived": True}
    for k, fmt in enumerate(formats):
        name, doc = f"u_out_fmt_{format_slug(fmt)}_{S}", f"u's {fmt} payments"
        add(name, "RULE", "u", short, "exact", "fmt_cnt", "out", "int", doc, fmt=k, **fmt_derived)
    doc = "v's incoming payments in this event's format (0 if unknown)"
    name = f"v_in_same_fmt_{S}"
    add(name, "RULE", "v", short, "exact", "same_fmt_cnt", "in", "int", doc, **fmt_derived)

    names = [f.name for f in out]
    dup = sorted(n for n, k in Counter(names).items() if k > 1)
    if dup:
        raise SpecError(f"feature names collide (choose other windows or formats): {dup}")
    bad = [n for n in names if is_forbidden(n) or n in NON_MODEL_COLUMNS]
    if bad:
        raise SpecError(f"generated feature names are forbidden as model inputs: {bad}")
    return tuple(out)


def build_features(
    features_cfg: Mapping[str, Any], rules_cfg: Mapping[str, Any], vocab: Mapping[str, Any]
) -> tuple[FeatureDef, ...]:
    """The model inputs generated from configs/features.yaml, configs/rules.yaml and the vocab."""
    nums = _features_numbers(features_cfg)
    P = sql_params(dict(rules_cfg), 0)  # validates the rules config (M1 checks); windows only
    return _feature_defs(
        short=nums["w_short"],
        long=nums["w_long"],
        sg=nums["w_sg"],
        pass_through=int(P["pass_through_window"]),
        round_trip=int(P["round_trip_window_plus_1"]) - 1,
        formats=_check_vocab(vocab)["payment_format"],
    )


# --- EngineSpec ---------------------------------------------------------------------------------

# sql_params keys the engine reads (stored verbatim; the engine never re-derives constants).
SQL_PARAM_WINDOWS = (
    "fan_in_window",
    "fan_out_window",
    "pass_through_window",
    "structuring_window",
    "round_window",
    "high_risk_window",
    "round_trip_window_plus_1",
)
# Hub-segmentable scenarios, in the order of `EngineSpec.excl` and of severities()' `excl`.
EXCL_SCENARIOS = tuple(HUB_SEGMENTABLE)
_FEATURES_SECTIONS = {
    "windows": {"short": "w_short", "long": "w_long", "sg": "w_sg"},
    "caps": {"count": "cap_count", "port": "cap_port", "gap": "cap_gap"},
    "budgets": {"rule_visits": "rule_visits", "feat_visits": "feat_visits"},
    "ring": {"compact_min_rows": "compact_min_rows"},
}


def _derived() -> Any:
    return field(init=False, repr=False, compare=False)


@dataclass(frozen=True)
class EngineSpec:
    """Frozen engine configuration: every constant the engine reads, and the derived layouts.

    Build it with `from_configs` (offline) or `from_json` (a saved feature_spec.json, M5). Never
    mutate the dict fields. Derived attributes (after `hubs`) are computed on construction.

    M2 spec notation -> attribute: S = w_short, L = w_long, W_sg = w_sg, W_pt = w_pt,
    W_rt = w_rt, H = hop, fan-in / fan-out / structuring / round-burst / high-risk windows =
    w_fan_in / w_fan_out / w_struct / w_round / w_hr, P = sql_params, caps.count / port / gap =
    cap_count / cap_port / cap_gap, W_max = W_max.
    """

    n_accounts: int
    w_short: int
    w_long: int
    w_sg: int
    cap_count: int  # cyc*/sg* feature counts are capped here
    cap_port: int  # port features = log1p(min(port, cap_port))
    cap_gap: int  # gap features = min(minutes, cap_gap)
    rule_visits: int  # D1/D2 walk steps per (account, minute)
    feat_visits: int  # D3 steps per (account, minute); SG work per target / per Ev build
    compact_min_rows: int
    sql_params: dict[str, int | float] = field(repr=False)  # verbatim sql_baseline.sql_params
    high_risk_formats: tuple[str, ...]
    vocab: dict[str, tuple[str, ...]] = field(repr=False)  # CATEGORICAL_FEATURES -> categories
    hub_cap: int
    hubs: tuple[int, ...] = field(repr=False)  # sorted train-fitted hub account ids

    # derived from sql_params (rule windows in minutes; H = round-trip hop window)
    w_fan_in: int = _derived()
    w_fan_out: int = _derived()
    w_pt: int = _derived()
    w_struct: int = _derived()
    w_round: int = _derived()
    w_hr: int = _derived()
    w_rt: int = _derived()
    hop: int = _derived()
    excl: tuple[bool, bool, bool, bool] = _derived()  # EXCL_SCENARIOS order
    round_cents: int = _derived()
    band_low_usd: float = _derived()
    band_high_usd: float = _derived()
    max_round_trip_paths: int = _derived()
    # derived layouts
    features: tuple[FeatureDef, ...] = _derived()
    feature_names: tuple[str, ...] = _derived()
    n_features: int = _derived()
    row_layout: tuple[str, ...] = _derived()
    slots: tuple[Slot, ...] = _derived()
    windows_all: tuple[int, ...] = _derived()  # every distinct window with an expiry cursor
    uniq_windows: tuple[int, ...] = _derived()  # windows with a pcnt[W] pair column
    W_max: int = _derived()
    # row positions of the tail
    i_sev: int = _derived()  # first severity; severities are row[i_sev : i_sev + 7]
    i_inflow: int = _derived()
    i_rule_trunc: int = _derived()
    i_cyc_trunc: int = _derived()
    i_sg_trunc: int = _derived()
    feature_map: dict[str, FeatureDef] = _derived()
    feature_index: dict[str, int] = _derived()
    _hash: str = _derived()

    def __post_init__(self) -> None:
        put = object.__setattr__
        put(self, "n_accounts", _int("n_accounts", self.n_accounts, 1))
        for section in _FEATURES_SECTIONS.values():
            for attr in section.values():
                put(self, attr, _int(attr, getattr(self, attr), 1))
        P = _check_sql_params(self.sql_params)
        put(self, "sql_params", P)
        put(self, "high_risk_formats", _check_formats(self.high_risk_formats))
        put(self, "vocab", _check_vocab(self.vocab))
        put(self, "hub_cap", _int("hub_cap", self.hub_cap, 0))
        if P["hub_cap"] != self.hub_cap:
            raise SpecError(f"sql_params hub_cap {P['hub_cap']} != hub_cap {self.hub_cap}")
        put(self, "hubs", _check_hubs(self.hubs, self.n_accounts))

        put(self, "w_fan_in", P["fan_in_window"])
        put(self, "w_fan_out", P["fan_out_window"])
        put(self, "w_pt", P["pass_through_window"])
        put(self, "w_struct", P["structuring_window"])
        put(self, "w_round", P["round_window"])
        put(self, "w_hr", P["high_risk_window"])
        put(self, "w_rt", P["round_trip_window_plus_1"] - 1)
        put(self, "hop", P["hop_window"])
        put(self, "excl", tuple(bool(P[HUB_SEGMENTABLE[s]]) for s in EXCL_SCENARIOS))
        put(self, "round_cents", P["round_cents"])
        put(self, "band_low_usd", P["band_low_usd"])
        put(self, "band_high_usd", P["band_high_usd"])
        put(self, "max_round_trip_paths", P["max_round_trip_paths"])

        feats = _feature_defs(
            short=self.w_short,
            long=self.w_long,
            sg=self.w_sg,
            pass_through=self.w_pt,
            round_trip=self.w_rt,
            formats=self.vocab["payment_format"],
        )
        names = tuple(f.name for f in feats)
        put(self, "features", feats)
        put(self, "feature_names", names)
        put(self, "n_features", len(names))
        put(self, "row_layout", (*names, *ROW_TAIL))
        n = len(names)
        put(self, "i_sev", n)
        put(self, "i_inflow", n + len(SEVERITY_COLUMNS))
        put(self, "i_rule_trunc", n + len(SEVERITY_COLUMNS) + 1)
        put(self, "i_cyc_trunc", n + len(SEVERITY_COLUMNS) + 2)
        put(self, "i_sg_trunc", n + len(SEVERITY_COLUMNS) + 3)
        put(self, "feature_map", {f.name: f for f in feats})
        put(self, "feature_index", {name: i for i, name in enumerate(names)})

        slots = build_slots(
            short=self.w_short,
            long=self.w_long,
            n_formats=len(self.vocab["payment_format"]),
            fan_in=self.w_fan_in,
            fan_out=self.w_fan_out,
            pass_through=self.w_pt,
            structuring=self.w_struct,
            round_burst=self.w_round,
            high_risk=self.w_hr,
        )
        windows_all = tuple(sorted({s.window for s in slots} | {self.w_rt, self.w_sg}))
        put(self, "slots", slots)
        put(self, "windows_all", windows_all)
        put(self, "uniq_windows", tuple(sorted({s.window for s in slots if s.kind == "uniq"})))
        put(self, "W_max", windows_all[-1])
        put(self, "_hash", _sha16(self._hash_doc()))

    # --- construction -------------------------------------------------------------------------

    @classmethod
    def from_configs(
        cls,
        features_cfg: Mapping[str, Any],
        rules_cfg: Mapping[str, Any],
        *,
        n_accounts: int,
        vocab: Mapping[str, Any],
        hub_cap: int,
        hubs: Iterable[int],
    ) -> EngineSpec:
        """Spec from configs/features.yaml + configs/rules.yaml and the train-fitted inputs.

        `vocab` = tx_features.fit_vocab(train rows); `hub_cap`, `hubs` = sql_baseline's
        hub_degree_cap / hub_accounts (the r_hubs definition). The rules config is validated by
        M1's checks inside `sql_params`, which is stored verbatim.
        """
        nums = _features_numbers(features_cfg)
        P = sql_params(dict(rules_cfg), hub_cap)
        return cls(
            n_accounts=n_accounts,
            **nums,
            sql_params=P,
            high_risk_formats=rules_cfg["high_risk_formats"],
            vocab=vocab,
            hub_cap=hub_cap,
            hubs=hubs,
        )

    def inputs_json(self) -> dict[str, Any]:
        """The constructor inputs as JSON (what `spec_hash` and `from_json` are built from)."""
        doc: dict[str, Any] = {"n_accounts": self.n_accounts}
        for section, keys in _FEATURES_SECTIONS.items():
            doc[section] = {k: getattr(self, attr) for k, attr in keys.items()}
        doc["sql_params"] = dict(self.sql_params)
        doc["high_risk_formats"] = list(self.high_risk_formats)
        doc["vocab"] = {k: list(v) for k, v in self.vocab.items()}
        doc["hub_cap"] = self.hub_cap
        doc["hubs"] = list(self.hubs)
        return doc

    def to_json(self) -> dict[str, Any]:
        """JSON document of the spec: inputs, derived layouts and the shared constants.

        Callers may add top-level keys (e.g. the serving bundle's model inputs); `from_json`
        ignores keys it does not know.
        """
        return {
            "format": SPEC_FORMAT,
            "engine_version": ENGINE_VERSION,
            "spec_hash": self._hash,
            "inputs": self.inputs_json(),
            "derived": {
                "rule_windows": {
                    "fan_in": self.w_fan_in,
                    "fan_out": self.w_fan_out,
                    "pass_through": self.w_pt,
                    "structuring": self.w_struct,
                    "round_burst": self.w_round,
                    "high_risk": self.w_hr,
                    "round_trip": self.w_rt,
                    "hop": self.hop,
                },
                "exclude_hub_senders": dict(zip(EXCL_SCENARIOS, self.excl, strict=True)),
                "windows_all": list(self.windows_all),
                "uniq_windows": list(self.uniq_windows),
                "W_max": self.W_max,
                "slots": [list(s) for s in self.slots],
                "features": [f.to_json() for f in self.features],
                "row_layout": list(self.row_layout),
            },
            "constants": {
                "input_columns": list(INPUT_COLUMNS),
                "event_fields": list(EVENT_FIELDS),
                "flag_bits": dict(FLAG_BITS),
                "severity_columns": list(SEVERITY_COLUMNS),
                "trunc_columns": list(TRUNC_COLUMNS),
                "non_model_columns": list(NON_MODEL_COLUMNS),
                "groups": list(GROUPS),
                "float32_cast": "np.asarray(values, np.float64).astype(np.float32)",
            },
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any] | str) -> EngineSpec:
        """Rebuild a spec from `to_json()` output (a dict or JSON text).

        Refuses a document from another engine version or whose stored spec_hash differs from
        the one this code computes (a feature definition changed without an ENGINE_VERSION bump).
        """
        if isinstance(doc, str):
            doc = json.loads(doc)
        if not isinstance(doc, Mapping) or "inputs" not in doc:
            raise SpecError("not a feature spec document (no 'inputs')")
        if doc.get("format") != SPEC_FORMAT:
            raise SpecError(f"feature spec format {doc.get('format')!r} != {SPEC_FORMAT}")
        if doc.get("engine_version") != ENGINE_VERSION:
            raise SpecError(
                f"feature spec written by engine version {doc.get('engine_version')!r}; this code "
                f"is version {ENGINE_VERSION}: rebuild the features"
            )
        inp = doc["inputs"]
        kw: dict[str, Any] = {}
        for section, keys in _FEATURES_SECTIONS.items():
            for k, attr in keys.items():
                kw[attr] = inp[section][k]
        spec = cls(
            n_accounts=inp["n_accounts"],
            **kw,
            sql_params=dict(inp["sql_params"]),
            high_risk_formats=inp["high_risk_formats"],
            vocab=inp["vocab"],
            hub_cap=inp["hub_cap"],
            hubs=inp["hubs"],
        )
        stored = doc.get("spec_hash")
        if stored != spec.spec_hash():
            old = [f.get("name") for f in doc.get("derived", {}).get("features", [])]
            changed = sorted(set(old) ^ set(spec.feature_names))
            raise SpecError(
                f"feature spec hash {stored!r} != {spec.spec_hash()!r} computed by this code "
                f"(names added/removed: {changed or 'none'}); bump ENGINE_VERSION and rebuild"
            )
        return spec

    # --- identity -----------------------------------------------------------------------------

    def spec_hash(self) -> str:
        """16 hex chars of sha256 over ENGINE_VERSION, the inputs and the generated layouts."""
        return self._hash

    def _hash_doc(self) -> dict[str, Any]:
        return {
            "engine_version": ENGINE_VERSION,
            "inputs": self.inputs_json(),
            "features": [
                {k: v for k, v in f.to_json().items() if k != "doc"} for f in self.features
            ],
            "slots": [list(s) for s in self.slots],
            "row_tail": list(ROW_TAIL),
        }

    def __hash__(self) -> int:
        return hash(self._hash)

    # --- layouts and lookups ------------------------------------------------------------------

    @property
    def FEATURES(self) -> tuple[FeatureDef, ...]:
        return self.features

    @property
    def ROW_LAYOUT(self) -> tuple[str, ...]:
        return self.row_layout

    @property
    def row_len(self) -> int:
        return len(self.row_layout)

    def feature(self, name: str) -> FeatureDef:
        try:
            return self.feature_map[name]
        except KeyError:
            raise KeyError(f"not a feature of this spec: {name!r}") from None

    def group_names(self, group: str) -> tuple[str, ...]:
        if group not in GROUPS:
            raise KeyError(f"unknown feature group {group!r}; groups are {GROUPS}")
        return tuple(f.name for f in self.features if f.group == group)

    @property
    def categorical_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features if f.categorical)

    @property
    def format_derived_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features if f.format_derived)

    @property
    def gnn_edge_attr_names(self) -> tuple[str, ...]:
        return tuple(f.name for f in self.features if f.gnn_edge_attr)

    @property
    def slot_bytes_per_account(self) -> int:
        return sum(array(s.typecode).itemsize for s in self.slots)

    def model_index(self, names: Iterable[str]) -> tuple[int, ...]:
        return model_index(names, self)

    def assert_model_inputs(self, names: Iterable[str]) -> list[str]:
        return assert_model_inputs(names, self)

    def table_schema(self) -> dict[str, pl.DataType]:
        """Column -> polars dtype of a feature-table part (§5.10), in column order."""
        schema: dict[str, pl.DataType] = {
            "row_id": pl.Int64(),
            "rank": pl.Int64(),
            "day": pl.Int16(),
            "split": pl.String(),
        }
        schema.update({n: pl.Float32() for n in self.feature_names})
        schema.update({s: pl.Float64() for s in SEVERITY_COLUMNS})
        schema[INFLOW_COLUMN] = pl.Int64()
        schema.update({t: pl.Int8() for t in TRUNC_COLUMNS})
        return schema


# --- model-input checks -------------------------------------------------------------------------


def assert_model_inputs(names: Iterable[str], spec: EngineSpec) -> list[str]:
    """Raise unless every name is a model-input feature of `spec`, none is M1-forbidden
    (`tx_features.is_forbidden`), none is in NON_MODEL_COLUMNS and none repeats. Returns the list.
    """
    names = [str(n) for n in names]
    bad = [
        n
        for n in names
        if n not in spec.feature_map
        or not spec.feature_map[n].model_input
        or is_forbidden(n)
        or n in NON_MODEL_COLUMNS
    ]
    dup = sorted(n for n, k in Counter(names).items() if k > 1)
    if bad or dup:
        raise ValueError(f"not allowed as model inputs: {bad}; repeated: {dup}")
    return names


def model_index(names: Iterable[str], spec: EngineSpec) -> tuple[int, ...]:
    """Positions of `names` in the engine row (= in spec.feature_names). Unknown names raise."""
    names = list(names)
    missing = [n for n in names if n not in spec.feature_index]
    if missing:
        raise ValueError(f"not features of this spec: {missing}")
    return tuple(spec.feature_index[n] for n in names)


# --- tolerance classes (§4.9) -------------------------------------------------------------------

TOL_VAR = 1e-9  # variance-scale tolerance of the running float64 moments


def tol_ok(
    tol: str,
    got: Any,
    want: Any,
    m2: Any = None,
    *,
    independent: bool = False,
    float32: bool = False,
) -> np.ndarray:
    """Element-wise check of `got` against the reference `want` under a tolerance class (§4.9).

    - independent=False: the same code path on the same platform; independent=True: an
      independent implementation (oracle, brute force, other libm). `float32=True` means `got`
      comes from the Float32 feature table (implies independent) and `want` is float64.
    - exact: bit-equal (NaN == NaN; after the same float32 cast when float32).
    - ulp: bit-equal on the same path, else <= 1 ulp of the compared dtype.
    - mean: |got - want| <= 1e-9 * max(1, sqrt(m2)) (+ 2 ulp32 when float32).
    - std: |got - want| <= sqrt(1e-9 * max(1, m2)) (+ 2 ulp32 when float32).
    m2 = the window's mean of l^2 from the reference (required for mean/std). Returns a bool array.
    """
    if tol not in TOL_CLASSES:
        raise ValueError(f"unknown tolerance class {tol!r}; classes are {TOL_CLASSES}")
    independent = independent or float32
    dt = np.float32 if float32 else np.float64
    g = np.ascontiguousarray(np.atleast_1d(np.asarray(got, dtype=np.float64)).astype(dt))
    w64 = np.atleast_1d(np.asarray(want, dtype=np.float64))
    w = np.ascontiguousarray(w64.astype(dt))
    both_nan = np.isnan(g) & np.isnan(w)
    if tol == "exact" or (tol == "ulp" and not independent):
        bits = np.uint32 if float32 else np.uint64
        return (g.view(bits) == w.view(bits)) | both_nan
    if tol == "ulp":
        with np.errstate(invalid="ignore"):
            ok = np.abs(g - w) <= np.spacing(np.maximum(np.abs(g), np.abs(w)))
        return ok | (g == w) | both_nan
    if m2 is None:
        raise ValueError(f"tolerance class {tol!r} needs m2 (the window's mean of l^2)")
    m2 = np.asarray(m2, dtype=np.float64)
    if tol == "mean":
        bound = TOL_VAR * np.maximum(1.0, np.sqrt(np.maximum(m2, 0.0)))
    else:
        bound = np.sqrt(TOL_VAR * np.maximum(1.0, m2))
    with np.errstate(invalid="ignore"):  # NaN rows are settled by both_nan below
        if float32:
            mag = np.maximum(np.abs(g), np.abs(w)).astype(np.float32)
            bound = bound + 2.0 * np.spacing(mag).astype(np.float64)
        ok = np.abs(g.astype(np.float64) - w64) <= bound
    return ok | (g.astype(np.float64) == w64) | both_nan


# --- validation helpers -------------------------------------------------------------------------


def _int(name: str, value: Any, minimum: int) -> int:
    """A whole number in [minimum, I32_LIMIT); integral floats are accepted (as M1 does)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise SpecError(f"{name} must be a whole number, got {value!r}")
    if not isinstance(value, numbers.Integral) and not (
        math.isfinite(float(value)) and float(value).is_integer()
    ):
        raise SpecError(f"{name} must be a whole number, got {value!r}")
    v = int(value)
    if not minimum <= v < I32_LIMIT:
        raise SpecError(f"{name} must be in [{minimum}, {I32_LIMIT}), got {value!r}")
    return v


def _features_numbers(features_cfg: Mapping[str, Any]) -> dict[str, int]:
    """The engine sections of configs/features.yaml (windows, caps, budgets, ring), validated.

    Other top-level sections (replay, snapshots, bench, memory_target_mb) belong to the driver.
    """
    if not isinstance(features_cfg, Mapping):
        raise SpecError("features config must be a mapping")
    out: dict[str, int] = {}
    for section, keys in _FEATURES_SECTIONS.items():
        sec = features_cfg.get(section)
        if not isinstance(sec, Mapping):
            raise SpecError(f"features config lacks the {section!r} section")
        if set(sec) != set(keys):
            raise SpecError(
                f"features.{section} must have exactly {sorted(keys)}, got {sorted(sec)}"
            )
        for k, attr in keys.items():
            out[attr] = _int(f"features.{section}.{k}", sec[k], 1)
    if out["w_short"] == out["w_long"]:
        raise SpecError("windows.short and windows.long must differ")
    return out


def _check_sql_params(P: Mapping[str, Any]) -> dict[str, int | float]:
    if not isinstance(P, Mapping):
        raise SpecError("sql_params must be a mapping")
    need = (
        *SQL_PARAM_WINDOWS,
        "hop_window",
        "round_cents",
        "band_low_usd",
        "band_high_usd",
        "hub_cap",
        "max_round_trip_paths",
        *HUB_SEGMENTABLE.values(),
    )
    missing = [k for k in need if k not in P]
    if missing:
        raise SpecError(f"sql_params lacks {missing}")
    out = dict(P)
    for k in SQL_PARAM_WINDOWS:
        out[k] = _int(f"sql_params.{k}", P[k], 2 if k == "round_trip_window_plus_1" else 1)
    out["hop_window"] = _int("sql_params.hop_window", P["hop_window"], 0)
    if out["hop_window"] > out["round_trip_window_plus_1"] - 1:
        raise SpecError("round_trip.hop_window_minutes must be <= round_trip.window_minutes")
    out["round_cents"] = _int("sql_params.round_cents", P["round_cents"], 1)
    out["hub_cap"] = _int("sql_params.hub_cap", P["hub_cap"], 0)
    out["max_round_trip_paths"] = _int(
        "sql_params.max_round_trip_paths", P["max_round_trip_paths"], 1
    )
    for k in HUB_SEGMENTABLE.values():
        if isinstance(P[k], bool) or P[k] not in (0, 1):
            raise SpecError(f"sql_params.{k} must be 0 or 1, got {P[k]!r}")
        out[k] = int(P[k])
    for k in ("band_low_usd", "band_high_usd"):
        v = P[k]
        if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v):
            raise SpecError(f"sql_params.{k} must be a finite number, got {v!r}")
        out[k] = float(v)
    if not 0 < out["band_low_usd"] < out["band_high_usd"]:
        raise SpecError("sql_params needs 0 < band_low_usd < band_high_usd")
    return out


def _check_formats(formats: Any) -> tuple[str, ...]:
    if isinstance(formats, str) or not isinstance(formats, Iterable):
        raise SpecError(f"high_risk_formats must be a list of format names, got {formats!r}")
    out = tuple(formats)
    if any(not isinstance(f, str) or not f for f in out):
        raise SpecError(f"high_risk_formats must be non-empty strings, got {list(out)!r}")
    return out


def _check_vocab(vocab: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    if not isinstance(vocab, Mapping) or set(vocab) != set(CATEGORICAL_FEATURES):
        got = sorted(vocab) if isinstance(vocab, Mapping) else vocab
        raise SpecError(f"vocab keys must be {CATEGORICAL_FEATURES}, got {got!r}")
    out: dict[str, tuple[str, ...]] = {}
    for col in CATEGORICAL_FEATURES:
        vals = vocab[col]
        if isinstance(vals, str) or not isinstance(vals, Iterable):
            raise SpecError(f"vocab[{col!r}] must be a list of strings")
        vals = tuple(vals)
        if any(not isinstance(x, str) or not x for x in vals):
            raise SpecError(f"vocab[{col!r}] must hold non-empty strings, got {list(vals)!r}")
        if len(set(vals)) != len(vals):
            raise SpecError(f"vocab[{col!r}] has repeated categories")
        out[col] = vals
    slugs = [format_slug(f) for f in out["payment_format"]]
    if len(set(slugs)) != len(slugs):
        raise SpecError(f"payment formats collide as name slugs: {slugs}")
    return out


def _check_hubs(hubs: Iterable[Any], n_accounts: int) -> tuple[int, ...]:
    if isinstance(hubs, str | bytes) or not isinstance(hubs, Iterable):
        raise SpecError("hubs must be a list of account ids")
    ids = sorted(_int("hub id", h, 0) for h in hubs)
    if len(set(ids)) != len(ids):
        raise SpecError("hubs has repeated account ids")
    if ids and ids[-1] >= n_accounts:
        raise SpecError(f"hub id {ids[-1]} >= n_accounts {n_accounts}")
    return tuple(ids)


def _sha16(obj: Any) -> str:
    text = json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
