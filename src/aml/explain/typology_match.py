"""The typology matchers (M6): a small decision tree (primary) and a fixed decision list (baseline).

The tree (`TreeMatcher`) is fitted on labelled val true-positive alerts (`fit_tree`, sklearn's
DecisionTreeClassifier), frozen as plain JSON in configs/typology_tree.json and measured on test.
Inference walks the JSON in pure Python (no sklearn at runtime). A node is either a split
{"feature", "threshold", "left", "right"} (left: value <= threshold; an undefined or missing value
goes left) or a leaf {"label", "n", "dist": {label: count}} (the val alerts that reached it). Its
inputs are a case pack's `why.typology.features`: the alert's model inputs and the shape of its
causal case subgraph (`g_*`, aml.explain.casepack). The evidence is the decision path in words
plus the leaf's purity. `fit_tree` reports the train accuracy, a stratified k-fold
cross-validated accuracy on the same rows (folds not grouped by attempt: optimistic), the
majority baseline and the feature importances.

The decision list is label-free: it reads ten causal engine features of the alerted row
(`inputs`), named as the engine spec generates them (at the shipped configs: `cyc2_2d`,
`cyc3_2d`, `cyc4_2d`, `sg_mids_1d`, `sg_srcs_1d`, `gs_u_1d`, `u_out_uniq_1d`, `u_in_uniq_1d`,
`v_in_uniq_1d`, `pt_ratio_12h`). The rules are tried in this fixed order and the first that fires
names the typology:

1. CYCLE           cyc2 + cyc3 + cyc4 >= cycle_min: a temporal cycle closes on the payment
2. SCATTER-GATHER  sg_mids >= sg_min: the receiver is also paid by siblings of the sender
3. GATHER-SCATTER  gs_u >= gs_min: the sender both gathered from and paid many accounts
4. FAN-OUT         u_out_uniq >= fan_out_min
5. FAN-IN          v_in_uniq >= fan_in_min
6. STACK           |pt_ratio - 1| <= stack_pt_tol: the sender passes its inflow straight on
7. BIPARTITE       u_out_uniq >= bipartite_min and v_in_uniq >= bipartite_min
8. RANDOM          u_in_uniq >= random_in_min and u_out_uniq <= random_out_max: a chain hop
9. OTHER           no rule fired ("no pattern")

Only the thresholds are tuned (`tune`, on labelled val alerts); a null threshold switches its
rule off, and a null random_out_max means no upper bound. `RulesMatcher` wraps the thresholds as
a matcher object that narrows a case pack's features to the list's ten inputs. Labels never enter
the inputs of either matcher.
"""

from __future__ import annotations

import copy
import dataclasses
import itertools
import math
import numbers
import re
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, ClassVar, NamedTuple

from aml.explain.narrative import duration_words, plural
from aml.features.spec import EngineSpec, window_tag

TYPOLOGIES = (
    "FAN-OUT",
    "FAN-IN",
    "CYCLE",
    "SCATTER-GATHER",
    "GATHER-SCATTER",
    "STACK",
    "BIPARTITE",
    "RANDOM",
    "OTHER",
)
OTHER = "OTHER"
# Matcher input roles = engine feature-name prefixes; the suffix is the feature's window tag.
ROLES = (
    "cyc2",
    "cyc3",
    "cyc4",
    "sg_mids",
    "sg_srcs",
    "gs_u",
    "u_out_uniq",
    "u_in_uniq",
    "v_in_uniq",
    "pt_ratio",
)
MAX_GRID = 1000  # tuning-grid combinations ("a few hundred" in the spec)
_TAG = re.compile(r"(\d+)([mhd])")
_TAG_MINUTES = {"m": 1, "h": 60, "d": 1440}


# --- parameters -------------------------------------------------------------------------------


def _threshold(name: str, v: Any) -> float | int | None:
    if v is None:
        return None
    if isinstance(v, bool) or not isinstance(v, numbers.Real) or not math.isfinite(v) or v < 0:
        raise ValueError(f"typology.{name} must be null or a finite number >= 0, got {v!r}")
    return v


@dataclass(frozen=True)
class MatcherParams:
    """The decision list's thresholds (configs/explain.yaml `typology`, frozen after val)."""

    cycle_min: float | None = 1
    sg_min: float | None = 1
    gs_min: float | None = 2
    fan_out_min: float | None = 5
    fan_in_min: float | None = 5
    stack_pt_tol: float | None = 0.2
    bipartite_min: float | None = 2
    random_in_min: float | None = 1
    random_out_max: float | None = 1

    def __post_init__(self) -> None:
        for f in dataclasses.fields(self):
            _threshold(f.name, getattr(self, f.name))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> MatcherParams:
        """Exactly the fields, no more and no fewer (a frozen config is checked, not padded)."""
        if not isinstance(d, Mapping):
            raise ValueError(f"typology must be a mapping, got {d!r}")
        names = [f.name for f in dataclasses.fields(cls)]
        unknown, missing = sorted(set(d) - set(names)), [n for n in names if n not in d]
        if unknown or missing:
            raise ValueError(f"typology keys: unknown {unknown}, missing {missing}")
        return cls(**{n: d[n] for n in names})

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def check_grid(grid: Mapping[str, Sequence[Any]]) -> tuple[str, ...]:
    """Validate a tuning grid; returns its keys in MatcherParams field order."""
    if not isinstance(grid, Mapping) or not grid:
        raise ValueError("the typology grid must be a non-empty mapping")
    names = [f.name for f in dataclasses.fields(MatcherParams)]
    unknown = sorted(set(grid) - set(names))
    if unknown:
        raise ValueError(f"typology grid: unknown keys {unknown}")
    size = 1
    for k, values in grid.items():
        if isinstance(values, str | bytes) or not isinstance(values, Sequence) or not values:
            raise ValueError(f"typology grid {k!r} must be a non-empty list")
        for v in values:
            _threshold(k, v)
        size *= len(values)
    if size > MAX_GRID:
        raise ValueError(f"typology grid has {size} combinations (max {MAX_GRID})")
    return tuple(n for n in names if n in grid)


# --- inputs -----------------------------------------------------------------------------------


def input_names(spec: EngineSpec) -> dict[str, str]:
    """Role -> the engine feature name the matcher reads for it, from the spec's windows."""
    return dict(_input_names(spec))


@lru_cache(maxsize=8)
def _input_names(spec: EngineSpec) -> tuple[tuple[str, str], ...]:
    S, R, G, P = (window_tag(w) for w in (spec.w_short, spec.w_rt, spec.w_sg, spec.w_pt))
    names = {
        "cyc2": f"cyc2_{R}",
        "cyc3": f"cyc3_{R}",
        "cyc4": f"cyc4_{R}",
        "sg_mids": f"sg_mids_{G}",
        "sg_srcs": f"sg_srcs_{G}",
        "gs_u": f"gs_u_{S}",
        "u_out_uniq": f"u_out_uniq_{S}",
        "u_in_uniq": f"u_in_uniq_{S}",
        "v_in_uniq": f"v_in_uniq_{S}",
        "pt_ratio": f"pt_ratio_{P}",
    }
    missing = [n for n in names.values() if n not in spec.feature_index]
    if missing:
        raise ValueError(f"the engine spec lacks the matcher's features {missing}")
    return tuple(names.items())


def inputs(row: Sequence[float], spec: EngineSpec) -> dict[str, float | None]:
    """{engine feature name: value} of the matcher's inputs, read from an engine row
    (`spec.row_layout`); NaN (undefined, e.g. pt_ratio without inflow) becomes None."""
    out: dict[str, float | None] = {}
    for _, name in _input_names(spec):
        v = float(row[spec.feature_index[name]])
        out[name] = v if math.isfinite(v) else None
    return out


class _Vals(NamedTuple):
    """One row's inputs by role (NaN = undefined: a comparison with it never fires)."""

    cyc2: float
    cyc3: float
    cyc4: float
    sg_mids: float
    sg_srcs: float
    gs_u: float
    u_out_uniq: float
    u_in_uniq: float
    v_in_uniq: float
    pt_ratio: float


def _resolve(features: Mapping[str, Any]) -> dict[str, str]:
    """Role -> the one key of `features` named <role>_<window tag>."""
    out = {}
    for role in ROLES:
        n = len(role) + 1
        keys = [k for k in features if k.startswith(role + "_") and _TAG.fullmatch(k[n:])]
        if len(keys) != 1:
            raise KeyError(f"the matcher needs exactly one {role}_<window> input, got {keys}")
        out[role] = keys[0]
    return out


def _vals(features: Mapping[str, Any], names: Mapping[str, str]) -> _Vals:
    def get(role: str) -> float:
        v = features[names[role]]
        return math.nan if v is None else float(v)

    return _Vals(*(get(r) for r in ROLES))


def _window(name: str) -> str:
    """The window of a feature name in words: u_out_uniq_1d -> "day", cyc2_2d -> "2 days"."""
    n, unit = _TAG.fullmatch(name.rsplit("_", 1)[1]).groups()
    return duration_words(int(n) * _TAG_MINUTES[unit])


# --- the decision list (fixed in code; only the thresholds are tuned) --------------------------


def _count(x: float) -> float:
    return 0.0 if math.isnan(x) else x


def _ge(x: float, t: float | None) -> bool:
    return t is not None and x >= t


def _cycle(v: _Vals, p: MatcherParams) -> bool:
    return _ge(_count(v.cyc2) + _count(v.cyc3) + _count(v.cyc4), p.cycle_min)


def _scatter_gather(v: _Vals, p: MatcherParams) -> bool:
    return _ge(v.sg_mids, p.sg_min)


def _gather_scatter(v: _Vals, p: MatcherParams) -> bool:
    return _ge(v.gs_u, p.gs_min)


def _fan_out(v: _Vals, p: MatcherParams) -> bool:
    return _ge(v.u_out_uniq, p.fan_out_min)


def _fan_in(v: _Vals, p: MatcherParams) -> bool:
    return _ge(v.v_in_uniq, p.fan_in_min)


def _stack(v: _Vals, p: MatcherParams) -> bool:
    return p.stack_pt_tol is not None and abs(v.pt_ratio - 1.0) <= p.stack_pt_tol


def _bipartite(v: _Vals, p: MatcherParams) -> bool:
    return _ge(v.u_out_uniq, p.bipartite_min) and _ge(v.v_in_uniq, p.bipartite_min)


def _random(v: _Vals, p: MatcherParams) -> bool:
    hi = p.random_out_max
    return _ge(v.u_in_uniq, p.random_in_min) and (hi is None or v.u_out_uniq <= hi)


RULES = (
    ("CYCLE", _cycle),
    ("SCATTER-GATHER", _scatter_gather),
    ("GATHER-SCATTER", _gather_scatter),
    ("FAN-OUT", _fan_out),
    ("FAN-IN", _fan_in),
    ("STACK", _stack),
    ("BIPARTITE", _bipartite),
    ("RANDOM", _random),
)
ORDER = tuple(label for label, _ in RULES)  # then OTHER


def _label(v: _Vals, p: MatcherParams) -> str:
    for label, fires in RULES:
        if fires(v, p):
            return label
    return OTHER


def _n(x: float) -> str:
    return "n/a" if math.isnan(x) else (str(int(x)) if x.is_integer() else f"{x:.3g}")


def _reason(label: str, v: _Vals, names: Mapping[str, str]) -> str:
    """Why `label` fired, in words, with the engine feature values."""
    if label == "CYCLE":
        c = [_count(v.cyc2), _count(v.cyc3), _count(v.cyc4)]
        return (
            f"the payment closes {plural(int(sum(c)), 'temporal cycle')} back to the sender in "
            f"the previous {_window(names['cyc2'])} ({names['cyc2']} = {_n(v.cyc2)}, "
            f"{names['cyc3']} = {_n(v.cyc3)}, {names['cyc4']} = {_n(v.cyc4)})"
        )
    if label == "SCATTER-GATHER":
        return (
            f"the receiver is also paid by {plural(int(v.sg_mids), 'sibling account')} of the "
            f"sender fed by {plural(int(_count(v.sg_srcs)), 'common source')} in the previous "
            f"{_window(names['sg_mids'])} ({names['sg_mids']} = {_n(v.sg_mids)}, "
            f"{names['sg_srcs']} = {_n(v.sg_srcs)})"
        )
    if label == "GATHER-SCATTER":
        return (
            f"the sender both received from and paid at least {_n(v.gs_u)} distinct accounts "
            f"in the previous {_window(names['gs_u'])} ({names['gs_u']} = {_n(v.gs_u)})"
        )
    if label == "FAN-OUT":
        return (
            f"the sender paid {_n(v.u_out_uniq)} distinct receivers in the previous "
            f"{_window(names['u_out_uniq'])} ({names['u_out_uniq']} = {_n(v.u_out_uniq)})"
        )
    if label == "FAN-IN":
        return (
            f"the receiver had {_n(v.v_in_uniq)} distinct payers in the previous "
            f"{_window(names['v_in_uniq'])} ({names['v_in_uniq']} = {_n(v.v_in_uniq)})"
        )
    if label == "STACK":
        return (
            f"the payment is {v.pt_ratio:.0%} of the sender's inflow in the previous "
            f"{_window(names['pt_ratio'])}: the money passes straight through "
            f"({names['pt_ratio']} = {v.pt_ratio:.3f})"
        )
    if label == "BIPARTITE":
        return (
            f"the sender paid {_n(v.u_out_uniq)} and the receiver was paid by "
            f"{_n(v.v_in_uniq)} distinct accounts in the previous "
            f"{_window(names['u_out_uniq'])}: a many-to-many link "
            f"({names['u_out_uniq']} = {_n(v.u_out_uniq)}, "
            f"{names['v_in_uniq']} = {_n(v.v_in_uniq)})"
        )
    if label == "RANDOM":
        return (
            f"the sender received from {_n(v.u_in_uniq)} and paid {_n(v.u_out_uniq)} distinct "
            f"accounts in the previous {_window(names['u_in_uniq'])}: a hop in a chain "
            f"({names['u_in_uniq']} = {_n(v.u_in_uniq)}, "
            f"{names['u_out_uniq']} = {_n(v.u_out_uniq)})"
        )
    raise ValueError(f"no rule for {label!r}")


# --- the matcher ------------------------------------------------------------------------------


def match(features: Mapping[str, float | None], p: MatcherParams) -> tuple[str, list[str]]:
    """(typology label, evidence) for one alert. `features` = `inputs(row, spec)` (keys
    <role>_<window tag>). The evidence starts with the reason for the label, then names every
    later rule that also fired ("also consistent with ...")."""
    names = _resolve(features)
    v = _vals(features, names)
    fired = [label for label, fires in RULES if fires(v, p)]
    if not fired:
        return OTHER, ["no typology rule fired: the alert matches no known laundering pattern"]
    evidence = [_reason(fired[0], v, names)]
    evidence += [f"also consistent with {t}: {_reason(t, v, names)}" for t in fired[1:]]
    return fired[0], evidence


@dataclass(frozen=True)
class RulesMatcher:
    """The decision list as a matcher object (the reported baseline). `names` = the engine names
    of its ten inputs (`input_names(spec).values()`): a wider features dict, e.g. a case pack's
    `why.typology.features` (where u_out_uniq_1d and u_out_uniq_3d would be ambiguous), is
    narrowed to them first (KeyError when one is absent). None: the dict holds only its inputs."""

    params: MatcherParams
    names: tuple[str, ...] | None = None
    model: ClassVar[str] = "rules"

    def narrow(self, features: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.names is None:
            return features
        return {n: features[n] for n in self.names}

    def label(self, features: Mapping[str, Any]) -> str:
        f = self.narrow(features)
        return _label(_vals(f, _resolve(f)), self.params)

    def match(self, features: Mapping[str, Any]) -> tuple[str, list[str]]:
        return match(self.narrow(features), self.params)


# --- accuracy and tuning ------------------------------------------------------------------------


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def _truth(truth: Any) -> str:
    if truth not in TYPOLOGIES:
        raise ValueError(f"unknown ground-truth typology {truth!r}")
    return truth


def _prepared(rows: Sequence[tuple[Mapping[str, Any], str]]) -> list[tuple[_Vals, str]]:
    return [(_vals(f, _resolve(f)), _truth(truth)) for f, truth in rows]


def _report(pairs: Iterable[tuple[str, str]]) -> dict[str, Any]:
    """Scores of (truth, predicted) pairs: overall, macro recall, the majority-class baseline,
    per-typology recall and precision, and the confusion matrix (truth -> predicted -> count)."""
    conf = {t: dict.fromkeys(TYPOLOGIES, 0) for t in TYPOLOGIES}
    for truth, pred in pairs:
        if pred not in TYPOLOGIES:
            raise ValueError(f"the matcher returned an unknown typology {pred!r}")
        conf[truth][pred] += 1
    counts = {t: sum(conf[t].values()) for t in TYPOLOGIES}
    n = sum(counts.values())
    correct = sum(conf[t][t] for t in TYPOLOGIES)
    per: dict[str, dict[str, Any]] = {}
    for t in TYPOLOGIES:
        predicted = sum(conf[x][t] for x in TYPOLOGIES)
        per[t] = {
            "support": counts[t],
            "predicted": predicted,
            "correct": conf[t][t],
            "recall": _ratio(conf[t][t], counts[t]),
            "precision": _ratio(conf[t][t], predicted),
        }
    recalls = [d["recall"] for d in per.values() if d["recall"] is not None]
    top = max(TYPOLOGIES, key=lambda t: counts[t]) if n else None  # ties -> TYPOLOGIES order
    return {
        "n": n,
        "correct": correct,
        "accuracy": _ratio(correct, n),
        "macro_recall": sum(recalls) / len(recalls) if recalls else None,
        "majority": {"label": top, "accuracy": _ratio(counts[top], n) if n else None},
        "per_typology": per,
        "confusion": conf,
    }


def _accuracy(prepared: list[tuple[_Vals, str]], p: MatcherParams) -> dict[str, Any]:
    scores = _report((truth, _label(v, p)) for v, truth in prepared)
    return {**scores, "model": RulesMatcher.model, "params": p.to_dict()}


def _predictor(m: Any) -> Callable[[Mapping[str, Any]], str]:
    """features -> label of a matcher object (`.label` or `.match`) or a callable."""
    if callable(getattr(m, "label", None)):
        return m.label
    if callable(getattr(m, "match", None)):
        return lambda f: m.match(f)[0]
    if callable(m):

        def call(f: Mapping[str, Any]) -> str:
            out = m(f)
            return out[0] if isinstance(out, tuple) else out

        return call
    raise TypeError(f"not a typology matcher: {m!r}")


def accuracy(rows: Sequence[tuple[Mapping[str, Any], str]], m: Matcher) -> dict[str, Any]:
    """Accuracy of a matcher on labelled rows (features, ground-truth typology): overall, macro
    recall, the majority-class baseline, per-typology recall and precision, the confusion matrix
    (truth -> predicted -> count), `model` ("rules", "tree" or "callable") and `params` (the
    decision list's thresholds; None for other matchers). Undefined ratios are None (JSON-safe).

    `m`: MatcherParams (the decision list, over rows that hold only its inputs), a matcher object
    (RulesMatcher, TreeMatcher: anything with `.label(features)` or `.match(features)`), or a
    callable features -> label or (label, evidence)."""
    if isinstance(m, MatcherParams):
        return _accuracy(_prepared(rows), m)
    if isinstance(m, RulesMatcher):
        return _accuracy(_prepared([(m.narrow(f), t) for f, t in rows]), m.params)
    predict = _predictor(m)
    pairs = [(_truth(truth), predict(f)) for f, truth in rows]
    return {**_report(pairs), "model": getattr(m, "model", "callable"), "params": None}


def tune(
    rows: Sequence[tuple[Mapping[str, Any], str]],
    grid: Mapping[str, Sequence[Any]],
    *,
    base: MatcherParams | None = None,
) -> tuple[MatcherParams, dict[str, Any]]:
    """The grid point with the highest accuracy on the labelled rows; ties -> the first point in
    grid order (keys in MatcherParams field order, values as listed). Fields not in the grid
    keep `base` (default MatcherParams()). With no rows, the first point is returned."""
    keys = check_grid(grid)
    base = MatcherParams() if base is None else base
    prepared = _prepared(rows)
    best, best_correct, n_points = None, -1, 0
    for point in itertools.product(*(grid[k] for k in keys)):
        p = dataclasses.replace(base, **dict(zip(keys, point, strict=True)))
        correct = sum(_label(v, p) == truth for v, truth in prepared)
        n_points += 1
        if correct > best_correct:
            best, best_correct = p, correct
    info = {
        "n_rows": len(prepared),
        "combinations": n_points,
        "grid": {k: list(grid[k]) for k in keys},
        "base": base.to_dict(),
        "best": _accuracy(prepared, best),
    }
    return best, info


# --- the decision tree (primary): fitted on val, frozen as JSON, walked in pure Python ----------

TREE_FORMAT = 1
TREE_DOC_KEYS = ("format", "tree", "trained_on", "info")  # info: fit_tree's report, not kept
SPLIT_KEYS = frozenset({"feature", "threshold", "left", "right"})
LEAF_KEYS = frozenset({"label", "n", "dist"})
MAX_TREE_DEPTH = 32
MISSING = -1e30  # fit_tree's stand-in for an undefined input: below every real value, so left
MISSING_SPLIT = -1e29  # a threshold below this splits undefined (left) from defined (right)
TREE_FIT = {"max_depth": 5, "min_samples_leaf": 10, "seed": 0}  # fit_tree's defaults
_FIT_MIN = {"max_depth": 1, "min_samples_leaf": 1, "seed": 0}
CV_FOLDS = 5
TOP_IMPORTANCES = 10
SCORE_KEYS = ("n", "correct", "accuracy", "macro_recall", "per_typology", "confusion")

# Feature -> the sentence of a decision step. Engine features are keyed by their stem (the name
# without its _<window> tag; {w} = the window in words), the others by name; {v} = the value. A
# pair of sentences is a 0/1 flag: (value <= threshold, value > threshold). Unlisted features
# fall back to "<name> = <value> <op> <threshold>".
FEATURE_TEXT: dict[str, str | tuple[str, str]] = {
    "u_out_uniq": "the sender paid {v} distinct counterparties in the previous {w}",
    "u_in_uniq": "the sender was paid by {v} distinct counterparties in the previous {w}",
    "v_in_uniq": "the receiver was paid by {v} distinct counterparties in the previous {w}",
    "v_out_uniq": "the receiver paid {v} distinct counterparties in the previous {w}",
    "u_out_cnt": "the sender made {v} payments in the previous {w}",
    "u_in_cnt": "the sender received {v} payments in the previous {w}",
    "v_in_cnt": "the receiver received {v} payments in the previous {w}",
    "v_out_cnt": "the receiver made {v} payments in the previous {w}",
    "pair_cnt": "the sender paid this receiver {v} times in the previous {w}",
    "cyc2": "{v} two-payment cycles close back to the sender in the previous {w}",
    "cyc3": "{v} three-payment cycles close back to the sender in the previous {w}",
    "cyc4": "{v} four-payment cycles close back to the sender in the previous {w}",
    "sg_mids": "{v} siblings of the sender (fed by a common source) also paid the receiver in "
    "the previous {w}",
    "sg_srcs": "the receiver's sibling payers had {v} distinct common sources in the previous {w}",
    "gs_u": "the sender both received from and paid at least {v} distinct accounts in the "
    "previous {w}",
    "gs_v": "the receiver both received from and paid at least {v} distinct accounts in the "
    "previous {w}",
    "pt_ratio": "the payment is {v} times the sender's inflow of the previous {w}",
    "g_src_out_deg": "the sender paid {v} distinct accounts in the case subgraph",
    "g_src_in_deg": "the sender was paid by {v} distinct accounts in the case subgraph",
    "g_dst_in_deg": "the receiver was paid by {v} distinct accounts in the case subgraph",
    "g_dst_out_deg": "the receiver paid {v} distinct accounts in the case subgraph",
    "g_common_sinks": "{v} accounts in the case subgraph were paid by two or more of the "
    "sender's receivers",
    "g_cycle_back": (
        "no earlier payment path of at most two hops leads from the receiver back to the sender",
        "an earlier payment path of at most two hops leads from the receiver back to the sender",
    ),
    "g_n_nodes": "the case subgraph has {v} accounts on earlier payments",
    "g_n_edges": "the case subgraph has {v} earlier payments",
    "pair_is_new": (
        "the sender had paid this receiver before",
        "the sender had never paid this receiver before",
    ),
}


def _value(x: Any) -> float | None:
    """A tree input as a float; None, NaN and infinities are undefined (None)."""
    if x is None:
        return None
    v = float(x)
    return v if math.isfinite(v) else None


def _fmt(x: float) -> str:
    return str(int(x)) if float(x).is_integer() and abs(x) < 1e15 else f"{x:.4g}"


def _split_tag(name: str) -> tuple[str, int | None]:
    """(stem, window minutes) of an engine feature name; (name, None) without a window tag."""
    stem, _, tag = name.rpartition("_")
    m = _TAG.fullmatch(tag)
    if not stem or m is None or int(m.group(1)) < 1:
        return name, None
    return stem, int(m.group(1)) * _TAG_MINUTES[m.group(2)]


def _text(name: str, v: float, left: bool) -> str:
    entry = FEATURE_TEXT.get(name)
    window = None
    if entry is None:
        stem, minutes = _split_tag(name)
        if minutes is not None:
            entry, window = FEATURE_TEXT.get(stem), duration_words(minutes)
    if entry is None:
        return ""
    if isinstance(entry, tuple):
        return entry[0] if left else entry[1]
    return entry.format(v=_fmt(v), w=window)


def _step(name: str, v: float | None, left: bool, t: float) -> str:
    """One decision on the path, in words, with the value and the threshold."""
    ts = _fmt(t)
    if v is None:
        if t < MISSING_SPLIT:
            return f"{name} is undefined"
        return f"{name} is undefined (missing values take the <= {ts} branch)"
    vs = _fmt(v)
    if t < MISSING_SPLIT and not left:
        detail = f"{name} = {vs}, defined"
    else:
        detail = f"{name} = {vs} {'<=' if left else '>'} {ts}"
    text = _text(name, v, left)
    return f"{text} ({detail})" if text else detail


def _purity(leaf: Mapping[str, Any]) -> str:
    n, label = leaf["n"], leaf["label"]
    if n == 0:
        return f"no validation case reached this leaf (label {label})"
    share = leaf["dist"].get(label, 0) / n
    return f"{share:.0%} of {n} validation cases at this leaf were {label}"


def _count_value(x: Any, what: str) -> int:
    if isinstance(x, bool) or not isinstance(x, int) or x < 0:
        raise ValueError(f"{what} must be an integer >= 0, got {x!r}")
    return x


def _node(d: Any, depth: int) -> dict[str, Any]:
    """A validated, normalised copy of a tree node (dist in TYPOLOGIES order, zeros dropped)."""
    if depth > MAX_TREE_DEPTH:
        raise ValueError(f"the typology tree is deeper than {MAX_TREE_DEPTH}")
    if not isinstance(d, Mapping):
        raise ValueError(f"a tree node must be a mapping, got {d!r}")
    keys = set(d)
    if keys == LEAF_KEYS:
        label = d["label"]
        if label not in TYPOLOGIES:
            raise ValueError(f"tree leaf label {label!r} is not a typology")
        n = _count_value(d["n"], "a leaf's n")
        dist = d["dist"]
        if not isinstance(dist, Mapping) or set(dist) - set(TYPOLOGIES):
            raise ValueError(f"a leaf's dist must map typologies to counts, got {dist!r}")
        counts = {t: _count_value(dist[t], f"dist[{t!r}]") for t in TYPOLOGIES if t in dist}
        if sum(counts.values()) != n:
            raise ValueError(f"a leaf's dist sums to {sum(counts.values())}, its n is {n}")
        return {"label": label, "n": n, "dist": {t: c for t, c in counts.items() if c}}
    if keys == SPLIT_KEYS:
        name, t = d["feature"], d["threshold"]
        if not isinstance(name, str) or not name:
            raise ValueError(f"a split's feature must be a non-empty string, got {name!r}")
        if isinstance(t, bool) or not isinstance(t, numbers.Real) or not math.isfinite(t):
            raise ValueError(f"a split's threshold must be a finite number, got {t!r}")
        return {
            "feature": name,
            "threshold": float(t),
            "left": _node(d["left"], depth + 1),
            "right": _node(d["right"], depth + 1),
        }
    raise ValueError(
        f"a tree node has keys {sorted(map(str, keys))}: a split has {sorted(SPLIT_KEYS)}, "
        f"a leaf {sorted(LEAF_KEYS)}"
    )


@dataclass(frozen=True)
class TreeMatcher:
    """A frozen decision tree over a case pack's typology features (module docstring). Build it
    with `from_dict` (validated, normalised) or `fit_tree`; `root` is the top node."""

    root: dict[str, Any]
    trained_on: dict[str, Any] | None = None
    model: ClassVar[str] = "tree"

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> TreeMatcher:
        """From the file document {"format": 1, "tree": node, "trained_on": null | {...}}; an
        "info" key (fit_tree's report in typology_tree_val.json) is allowed and dropped."""
        if not isinstance(d, Mapping):
            raise ValueError(f"a typology tree document must be a mapping, got {d!r}")
        unknown = sorted(set(d) - set(TREE_DOC_KEYS))
        if unknown:
            raise ValueError(f"typology tree: unknown keys {unknown}")
        fmt = d.get("format")
        if isinstance(fmt, bool) or fmt != TREE_FORMAT:
            raise ValueError(f"typology tree format must be {TREE_FORMAT}, got {fmt!r}")
        if "tree" not in d:
            raise ValueError("the typology tree document has no `tree`")
        trained_on = d.get("trained_on")
        if trained_on is not None and not isinstance(trained_on, Mapping):
            raise ValueError(f"typology tree trained_on must be null or a mapping: {trained_on!r}")
        meta = None if trained_on is None else copy.deepcopy(dict(trained_on))
        return cls(_node(d["tree"], 0), meta)

    def to_dict(self) -> dict[str, Any]:
        """The file document (configs/typology_tree.json), a deep copy."""
        return {
            "format": TREE_FORMAT,
            "tree": copy.deepcopy(self.root),
            "trained_on": copy.deepcopy(self.trained_on),
        }

    def _nodes(self) -> Iterator[tuple[dict[str, Any], int]]:
        """(node, depth) in preorder, left before right."""
        stack = [(self.root, 0)]
        while stack:
            node, depth = stack.pop()
            yield node, depth
            if "label" not in node:
                stack.append((node["right"], depth + 1))
                stack.append((node["left"], depth + 1))

    @property
    def trivial(self) -> bool:
        """A single leaf (the placeholder): it says nothing about the alert."""
        return "label" in self.root

    @property
    def inputs(self) -> tuple[str, ...]:
        """The features its splits read, in preorder of first use."""
        return tuple(dict.fromkeys(n["feature"] for n, _ in self._nodes() if "label" not in n))

    @property
    def depth(self) -> int:
        return max(d for _, d in self._nodes())

    @property
    def n_leaves(self) -> int:
        return sum(1 for n, _ in self._nodes() if "label" in n)

    def label(self, features: Mapping[str, Any]) -> str:
        """The leaf label for `features` (a missing or undefined value goes left)."""
        node = self.root
        while "label" not in node:
            v = _value(features.get(node["feature"]))
            node = node["left" if v is None or v <= node["threshold"] else "right"]
        return node["label"]

    def match(self, features: Mapping[str, Any]) -> tuple[str, list[str]]:
        """(typology label, evidence): the decision path as sentences, then the leaf's purity."""
        node, evidence = self.root, []
        while "label" not in node:
            name, t = node["feature"], node["threshold"]
            v = _value(features.get(name))
            left = v is None or v <= t
            evidence.append(_step(name, v, left, t))
            node = node["left" if left else "right"]
        evidence.append(_purity(node))
        return node["label"], evidence


Matcher = MatcherParams | RulesMatcher | TreeMatcher | Callable[[Mapping[str, Any]], Any]


def tree_fit_params(d: Mapping[str, Any] | None = None) -> dict[str, int]:
    """fit_tree's settings (configs/explain.yaml `typology_tree_fit`), defaults filled in."""
    if d is None:
        return dict(TREE_FIT)
    if not isinstance(d, Mapping):
        raise ValueError(f"typology_tree_fit must be a mapping, got {d!r}")
    unknown = sorted(set(d) - set(TREE_FIT))
    if unknown:
        raise ValueError(f"typology_tree_fit: unknown keys {unknown}")
    out = {**TREE_FIT, **d}
    for k, v in out.items():
        if isinstance(v, bool) or not isinstance(v, int) or v < _FIT_MIN[k]:
            raise ValueError(
                f"typology_tree_fit.{k} must be an integer >= {_FIT_MIN[k]}, got {v!r}"
            )
    if out["max_depth"] > MAX_TREE_DEPTH:
        raise ValueError(f"typology_tree_fit.max_depth must be <= {MAX_TREE_DEPTH}")
    return out


def _leaf(truths: Sequence[str]) -> dict[str, Any]:
    """One leaf for all rows (no rows: OTHER); ties -> the alphabetical first, as sklearn."""
    c = Counter(truths)
    label = max(sorted(c), key=lambda k: c[k]) if c else OTHER
    return {"label": label, "n": len(truths), "dist": {t: c[t] for t in TYPOLOGIES if c[t]}}


def _fit(x: Any, y: Any, p: Mapping[str, int]) -> Any:
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(
        max_depth=p["max_depth"], min_samples_leaf=p["min_samples_leaf"], random_state=p["seed"]
    )
    return clf.fit(x, y)


def _export(clf: Any, x: Any, truths: Sequence[str], names: Sequence[str]) -> dict[str, Any]:
    """The fitted sklearn tree as a JSON node: thresholds as fitted, each leaf's label as
    sklearn predicts it (argmax, ties -> classes_ order) and its rows' truth counts."""
    t = clf.tree_
    counts: dict[int, Counter] = defaultdict(Counter)
    for leaf, truth in zip(clf.apply(x).tolist(), truths, strict=True):
        counts[int(leaf)][str(truth)] += 1
    classes = [str(c) for c in clf.classes_]

    def build(i: int) -> dict[str, Any]:
        left, right = int(t.children_left[i]), int(t.children_right[i])
        if left == right:  # both TREE_LEAF: a leaf
            c = counts[i]
            label = max(classes, key=lambda k: c[k])
            dist = {k: c[k] for k in TYPOLOGIES if c[k]}
            return {"label": label, "n": sum(c.values()), "dist": dist}
        return {
            "feature": names[int(t.feature[i])],
            "threshold": float(t.threshold[i]),
            "left": build(left),
            "right": build(right),
        }

    return _node(build(0), 0)


def _scores(rep: Mapping[str, Any]) -> dict[str, Any]:
    return {k: rep[k] for k in SCORE_KEYS}


def _cross_validate(
    rows: Sequence[tuple[Mapping[str, Any], str]],
    x: Any,
    truths: list[str],
    names: list[str],
    p: Mapping[str, int],
) -> dict[str, Any] | None:
    """Stratified k-fold (shuffled, seeded) out-of-fold scores of the same tree settings, each
    fold's tree exported and walked like the frozen one; None when no class has 2 rows."""
    import numpy as np
    from sklearn.model_selection import StratifiedKFold

    k = min(CV_FOLDS, max(Counter(truths).values()))
    if k < 2:
        return None
    y = np.asarray(truths)
    with warnings.catch_warnings():
        # A class with fewer than k rows: sklearn warns and still splits (the folds are reported).
        warnings.filterwarnings("ignore", message="The least populated class", category=UserWarning)
        splits = list(StratifiedKFold(n_splits=k, shuffle=True, random_state=p["seed"]).split(x, y))
    pred: list[str] = [OTHER] * len(rows)
    folds = []
    for train_idx, test_idx in splits:
        clf = _fit(x[train_idx], y[train_idx], p)
        fold = TreeMatcher(_export(clf, x[train_idx], y[train_idx].tolist(), names))
        hits = 0
        for i in test_idx.tolist():
            pred[i] = fold.label(rows[i][0])
            hits += pred[i] == truths[i]
        folds.append(hits / len(test_idx))
    return {"folds": k, "fold_accuracy": folds, **_scores(_report(zip(truths, pred, strict=True)))}


def fit_tree(
    rows: Sequence[tuple[Mapping[str, Any], str]],
    *,
    max_depth: int = TREE_FIT["max_depth"],
    min_samples_leaf: int = TREE_FIT["min_samples_leaf"],
    seed: int = TREE_FIT["seed"],
) -> tuple[TreeMatcher, dict[str, Any]]:
    """The decision tree fitted on labelled rows (features, ground-truth typology) with sklearn's
    DecisionTreeClassifier (imported here only), exported to the JSON form (trained_on None: the
    caller records it). The columns are every feature name in the rows, in first-seen order; an
    undefined value becomes MISSING, below every real value, so it goes left as at inference.
    Deterministic for a seed (the tree's and the CV split's).

    info: rows, features, params, missing_value, depth, leaves, inputs, classes (truth counts),
    majority (the baseline on these rows), train_accuracy and train (in-sample scores of the
    exported tree), cv_accuracy and cv (stratified k-fold out-of-fold scores, k = min(CV_FOLDS,
    the largest class), None below 2; folds are not grouped by attempt), importances (the top
    TOP_IMPORTANCES features by sklearn's impurity importance, > 0)."""
    import numpy as np

    p = tree_fit_params(
        {"max_depth": max_depth, "min_samples_leaf": min_samples_leaf, "seed": seed}
    )
    truths = [_truth(t) for _, t in rows]
    names = list(dict.fromkeys(name for f, _ in rows for name in f))
    cv: dict[str, Any] | None = None
    importances: list[dict[str, Any]] = []
    if not rows or not names:
        tree = TreeMatcher(_node(_leaf(truths), 0))
    else:
        cells = [[_value(f.get(n)) for n in names] for f, _ in rows]
        x = np.array([[MISSING if v is None else v for v in r] for r in cells], dtype=np.float64)
        clf = _fit(x, truths, p)
        tree = TreeMatcher(_export(clf, x, truths, names))
        cv = _cross_validate(rows, x, truths, names, p)
        imp = [float(v) for v in clf.feature_importances_]
        order = sorted((j for j in range(len(names)) if imp[j] > 0), key=lambda j: (-imp[j], j))
        importances = [{"feature": names[j], "importance": imp[j]} for j in order]
    train = accuracy(rows, tree)
    counts = Counter(truths)
    info = {
        "rows": len(rows),
        "features": len(names),
        "params": p,
        "missing_value": MISSING,
        "depth": tree.depth,
        "leaves": tree.n_leaves,
        "inputs": list(tree.inputs),
        "classes": {t: counts[t] for t in TYPOLOGIES},
        "majority": train["majority"],
        "train_accuracy": train["accuracy"],
        "cv_accuracy": None if cv is None else cv["accuracy"],
        "train": _scores(train),
        "cv": cv,
        "importances": importances[:TOP_IMPORTANCES],
    }
    return tree, info
