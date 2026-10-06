"""Case packs (M6 spec §1): one alert's evidence as a JSON-safe dict in FinCEN 5W+H order.

`case_hook(champion, cfg, sink)` is the scorer's `on_alert` hook (`AlertHook = (Scored, Engine)
-> None`). StreamScorer calls it on the consumer thread right after the alert write, after
t_done (untimed), while the engine is still at the alert's minute m: its state holds only applied
events (minutes <= m - 1), so the subgraph read through `Engine.neighbours` is causal by
construction, and a guard refuses an engine whose clock has moved on. Nothing here writes to the
engine, so scores and alerts are unchanged.

Pack keys (PACK_FORMAT 1), in 5W+H order:
- pack_format, id (= row_id), case_key ("<src>-d<day>": subject account and simulated day), rank;
- who: subject / counterparty = {account, bank};
- what: amount_usd, amount_paid, payment_format, payment_currency, receiving_currency;
- when: day (minute // 1440 + 1), time ("HH:MM" of that day), minute;
- where: from_bank, to_bank, cross_bank;
- why: score (raw probability), calibrated (display only), threshold, rate_tag, rules_fired,
  drivers [{feature, value, display, contribution}] (top |TreeSHAP| in log-odds), base_value,
  others_contribution (the remaining features), log_odds (= the sum of all three), typology
  {label, model ("tree" | "rules"), evidence, features};
- how: subgraph {nodes [{id, role}], edges [{src, dst, minute, amount_usd, kind, rank}]},
  window_minutes, cap_1hop, cap_2hop;
- model_version, export_key, narrative.

`why.typology.features` holds the inputs of both typology matchers (`typology_features`): every
model input of the alert row by name (float(s.x[0][i])), the decision list's ten engine inputs
(the engine row's float64 values, also where they are model inputs), then the shape of the pack's
own causal subgraph over its history edges (`graph_stats`, names prefixed "g_"). Undefined values
are None. The matcher (`typology_matcher`) is the frozen tree (configs/typology_tree.json) when
`typology_model` is "tree" and the tree is more than one leaf, else the decision list.

Banks and the paid amount/currencies are not in `Scored`: they come from `fields` (the event's
INPUT_COLUMNS tuple, also read from `s.fields` when the scorer carries it), else from the engine's
pending copy of the event (amount_paid from its cents, currencies from the vocab codes, the bank
ids unknown: None, cross_bank from the same-bank flag).
"""

from __future__ import annotations

import logging
import math
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import yaml

from aml.explain import typology_match
from aml.explain.narrative import narrative
from aml.explain.typology_match import MatcherParams, RulesMatcher, TreeMatcher
from aml.features.spec import (
    E_FLAGS,
    E_PAID_C,
    E_PCUR,
    E_RANK,
    E_RCUR,
    E_ROW_ID,
    F_SAME_BANK,
    INPUT_COLUMNS,
    MINUTES_PER_DAY,
)
from aml.io import read_json

if TYPE_CHECKING:
    from aml.features.engine import Engine
    from aml.features.spec import EngineSpec
    from aml.serving.scorer import Champion, Scored

log = logging.getLogger(__name__)

PACK_FORMAT = 1
CONFIG_FILE = "explain.yaml"  # next to serving.yaml in configs/
TREE_FILE = "typology_tree.json"  # the frozen typology tree, next to explain.yaml
TYPOLOGY_MODELS = ("tree", "rules")
_INTS = {"window_minutes": 1, "cap_1hop": 0, "cap_2hop": 0, "top_drivers": 1}  # key -> minimum
CONFIG_KEYS = (
    *_INTS,
    "typology_model",
    "typology",
    "typology_grid",
    "typology_tree_fit",
    "typology_tree",
)
DIRECTIONS = ("in", "out")
# The shape of a pack's causal subgraph (history edges only), typology-tree inputs.
G_STATS = (
    "g_src_out_deg",
    "g_src_in_deg",
    "g_dst_in_deg",
    "g_dst_out_deg",
    "g_common_sinks",
    "g_cycle_back",
    "g_n_nodes",
    "g_n_edges",
)


# --- config -----------------------------------------------------------------------------------


def check_config(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """A validated copy of configs/explain.yaml (ValueError names the first bad key), with the
    frozen tree document under `typology_tree` (None: none). An absent typology_model is
    "rules"; an absent typology_tree_fit gets fit_tree's defaults."""
    if not isinstance(cfg, Mapping):
        raise ValueError("the explain config must be a mapping")
    unknown = sorted(set(cfg) - set(CONFIG_KEYS))
    if unknown:
        raise ValueError(f"explain config: unknown keys {unknown}")
    out: dict[str, Any] = {}
    for key, lo in _INTS.items():
        v = cfg.get(key)
        if isinstance(v, bool) or not isinstance(v, int) or v < lo:
            raise ValueError(f"explain.{key} must be an integer >= {lo}, got {v!r}")
        out[key] = v
    model = cfg.get("typology_model", "rules")
    if model not in TYPOLOGY_MODELS:
        raise ValueError(f"explain.typology_model must be one of {TYPOLOGY_MODELS}, got {model!r}")
    out["typology_model"] = model
    out["typology"] = MatcherParams.from_dict(cfg.get("typology")).to_dict()
    grid = cfg.get("typology_grid")
    if grid is not None:
        typology_match.check_grid(grid)
        out["typology_grid"] = {k: list(v) for k, v in grid.items()}
    out["typology_tree_fit"] = typology_match.tree_fit_params(cfg.get("typology_tree_fit"))
    tree = cfg.get("typology_tree")
    try:
        out["typology_tree"] = None if tree is None else TreeMatcher.from_dict(tree).to_dict()
    except ValueError as e:
        raise ValueError(f"explain.typology_tree ({TREE_FILE}): {e}") from e
    return out


def load_config(path: Path) -> dict[str, Any]:
    """configs/explain.yaml, validated, plus the frozen tree: TREE_FILE next to it (None when
    absent) under `typology_tree`."""
    path = Path(path)
    with path.open(encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if isinstance(raw, Mapping):
        if "typology_tree" in raw:
            raise ValueError(f"typology_tree is read from {TREE_FILE}, not from {path.name}")
        tree_path = path.with_name(TREE_FILE)
        raw = {**raw, "typology_tree": read_json(tree_path) if tree_path.is_file() else None}
    return check_config(raw)


def rules_matcher(cfg: Mapping[str, Any], spec: EngineSpec) -> RulesMatcher:
    """The decision list at the config's thresholds, reading its inputs by the spec's names."""
    names = tuple(typology_match.input_names(spec).values())
    return RulesMatcher(MatcherParams.from_dict(cfg["typology"]), names)


def frozen_tree(cfg: Mapping[str, Any]) -> TreeMatcher | None:
    """The config's tree when it is more than one leaf (the placeholder is one), else None."""
    doc = cfg.get("typology_tree")
    if doc is None:
        return None
    tree = TreeMatcher.from_dict(doc)
    return None if tree.trivial else tree


def typology_matcher(cfg: Mapping[str, Any], ch: Champion) -> TreeMatcher | RulesMatcher:
    """The packs' typology matcher: the frozen tree when typology_model is "tree" and the tree
    is more than one leaf, else the decision list. ValueError when the tree reads a feature the
    packs of this champion do not carry (`feature_names`)."""
    tree = frozen_tree(cfg) if cfg.get("typology_model") == "tree" else None
    if tree is None:
        return rules_matcher(cfg, ch.spec)
    missing = sorted(set(tree.inputs) - set(feature_names(ch.names, ch.spec)))
    if missing:
        raise ValueError(
            f"the typology tree reads features this champion's case packs lack: {missing} "
            f"(refit it with `make cases-val`, or set typology_model: rules)"
        )
    return tree


def load_calibration(ch: Champion) -> dict[str, Any] | None:
    """The bundle's calibration.json (display only), or None without a bundle directory."""
    if ch.bundle_dir is None:
        return None
    from aml.serving import bundle  # the bundle layout; imported only when a bundle is used

    path = Path(ch.bundle_dir) / bundle.CALIBRATION
    return read_json(path) if path.is_file() else None


# --- pieces -----------------------------------------------------------------------------------


def _finite(x: Any) -> float | None:
    v = float(x)
    return v if math.isfinite(v) else None


def calibrate(score: float, calibration: Mapping[str, Any] | None) -> float | None:
    """np.interp of the raw score on the isotonic calibration points (clipped at the ends)."""
    if not calibration or not calibration.get("x"):
        return None
    return float(np.interp(score, calibration["x"], calibration["y"]))


def display(spec: EngineSpec, name: str, value: float | None) -> str:
    """A model input as text: vocab strings for the categorical codes, whole numbers for counts
    and flags, 4 significant digits otherwise, n/a when undefined."""
    if value is None:
        return "n/a"
    f = spec.feature_map.get(name)
    if f is not None and f.categorical:
        cats = spec.vocab.get(name, ())
        code = int(value)
        return cats[code] if 0 <= code < len(cats) else "unknown"
    if value.is_integer() and abs(value) < 1e15:
        return str(int(value))
    return f"{value:.4g}"


def drivers(ch: Champion, x: np.ndarray, top: int) -> dict[str, Any]:
    """The top-`top` TreeSHAP contributions (log-odds) of the model inputs `x` (1, n float32),
    by |contribution| (ties: model input order), plus the base value and the rest; the three
    parts sum to the raw log-odds."""
    contrib = np.asarray(
        ch.booster.predict(x, pred_contrib=True, num_threads=1), dtype=np.float64
    ).reshape(-1)
    n = len(ch.names)
    if contrib.size != n + 1:
        raise ValueError(f"pred_contrib gave {contrib.size} values for {n} model inputs")
    c = contrib[:n].tolist()
    order = sorted(range(n), key=lambda j: (-abs(c[j]), j))
    values = np.asarray(x, dtype=np.float32).reshape(-1)
    out = []
    for j in order[:top]:
        v = _finite(values[j])
        name = ch.names[j]
        out.append(
            {
                "feature": name,
                "value": v,
                "display": display(ch.spec, name, v),
                "contribution": c[j],
            }
        )
    return {
        "drivers": out,
        "base_value": float(contrib[n]),
        "others_contribution": math.fsum(c[j] for j in order[top:]),
        "log_odds": math.fsum(contrib.tolist()),
    }


def subgraph(
    eng: Engine,
    src: int,
    dst: int,
    minute: int,
    rank: int,
    amount_usd: float,
    *,
    window: int,
    cap_1hop: int,
    cap_2hop: int,
) -> dict[str, list[dict[str, Any]]]:
    """The causal subgraph of the alert src -> dst at `minute`, from the engine's windowed
    adjacency (`Engine.neighbours`, minutes in [minute - window, minute - 1], window clamped to
    the spec's W_max).

    Hop 1: the newest `cap_1hop` in- and out-payments of src and of dst. Hop 2: from each hop-1
    neighbour (not src or dst), the newest `cap_2hop` payments continuing outward (the payers of
    a payer, the payees of a payee). Edges are deduplicated by rank; the alert edge comes first
    (kind "alert"), then the history edges in rank order (kind "history"). Nodes: subject (src),
    counterparty (dst), then the others by id. ValueError unless the engine clock is `minute`
    (a later clock would show the alert minute's own payments)."""
    if eng.clock != minute:
        raise ValueError(
            f"case packs are built as of the alert's minute {minute}, the engine clock is "
            f"{eng.clock}: call it from the on_alert hook"
        )
    w = min(int(window), eng.spec.W_max)
    hist: dict[int, dict[str, Any]] = {}

    def walk(acct: int, direction: str, cap: int) -> list[int]:
        found = []
        for cp, t, r, usd_c in eng.neighbours(acct, direction, w, cap):
            a, b = (cp, acct) if direction == "in" else (acct, cp)
            if r not in hist:
                e = {"src": a, "dst": b, "minute": t, "amount_usd": usd_c / 100}
                hist[r] = {**e, "kind": "history", "rank": r}
            found.append(cp)
        return found

    centres = (src,) if src == dst else (src, dst)
    frontier: dict[tuple[int, str], None] = {}  # insertion-ordered set of (neighbour, direction)
    for c in centres:
        for d in DIRECTIONS:
            for cp in walk(c, d, cap_1hop):
                if cp not in centres:
                    frontier[(cp, d)] = None
    for cp, d in frontier:
        walk(cp, d, cap_2hop)
    alert = {"src": src, "dst": dst, "minute": minute, "amount_usd": float(amount_usd)}
    edges = [{**alert, "kind": "alert", "rank": rank}, *(hist[r] for r in sorted(hist))]
    others = sorted({a for e in hist.values() for a in (e["src"], e["dst"])} - set(centres))
    nodes = [{"id": src, "role": "subject"}]
    if dst != src:
        nodes.append({"id": dst, "role": "counterparty"})
    nodes += [{"id": a, "role": "other"} for a in others]
    return {"nodes": nodes, "edges": edges}


# --- typology features ----------------------------------------------------------------------------


def graph_stats(sg: Mapping[str, Any], src: int, dst: int) -> dict[str, int]:
    """The shape of a pack's causal subgraph over its history edges (the alert edge is left out),
    capped like the subgraph itself (cap_1hop, cap_2hop):
    - g_src_out_deg, g_src_in_deg, g_dst_in_deg, g_dst_out_deg: distinct counterparties of the
      sender / receiver (other accounts: a self-payment is no counterparty);
    - g_common_sinks: accounts paid by >= 2 distinct receivers of the sender;
    - g_cycle_back: 1 when a history path dst -> src or dst -> x -> src exists, else 0;
    - g_n_nodes: distinct accounts on history edges; g_n_edges: history edges."""
    hist = [(int(e["src"]), int(e["dst"])) for e in sg["edges"] if e["kind"] == "history"]
    pays: dict[int, set[int]] = defaultdict(set)  # account -> its payees (self excluded)
    paid_by: dict[int, set[int]] = defaultdict(set)  # account -> its payers (self excluded)
    for a, b in hist:
        if a != b:
            pays[a].add(b)
            paid_by[b].add(a)
    receivers = pays.get(src, set())
    payers_of: dict[int, int] = defaultdict(int)  # account -> how many of src's receivers pay it
    for r in receivers:
        for y in pays.get(r, ()):
            payers_of[y] += 1
    back = src in pays.get(dst, ()) or any(src in pays.get(x, ()) for x in pays.get(dst, ()))
    return {
        "g_src_out_deg": len(receivers),
        "g_src_in_deg": len(paid_by.get(src, ())),
        "g_dst_in_deg": len(paid_by.get(dst, ())),
        "g_dst_out_deg": len(pays.get(dst, ())),
        "g_common_sinks": sum(1 for k in payers_of.values() if k >= 2),
        "g_cycle_back": int(back),
        "g_n_nodes": len({a for e in hist for a in e}),
        "g_n_edges": len(hist),
    }


def feature_names(names: Sequence[str], spec: EngineSpec) -> tuple[str, ...]:
    """The keys of a pack's `why.typology.features`, in order: the model inputs `names`, the
    decision list's inputs that are not model inputs, then G_STATS."""
    rules = [n for n in typology_match.input_names(spec).values() if n not in names]
    return (*names, *rules, *G_STATS)


def typology_features(
    s: Scored, names: Sequence[str], spec: EngineSpec, sg: Mapping[str, Any]
) -> dict[str, float | int | None]:
    """The typology matchers' inputs of the alert `s` (module docstring), JSON-safe: finite
    numbers or None. `names` = the champion's model inputs (the columns of `s.x`)."""
    x = np.asarray(s.x, dtype=np.float32).reshape(-1)
    if x.size != len(names):
        raise ValueError(f"the alert row has {x.size} model inputs, the champion {len(names)}")
    feats: dict[str, float | int | None] = {n: _finite(x[i]) for i, n in enumerate(names)}
    feats.update(typology_match.inputs(s.row, spec))  # float64 engine values (None: undefined)
    clash = sorted(set(G_STATS) & set(feats))
    if clash:
        raise ValueError(f"model inputs clash with the subgraph statistics: {clash}")
    feats.update(graph_stats(sg, int(s.src), int(s.dst)))
    return feats


def _decode(cats: Sequence[str], code: int) -> str | None:
    return cats[code] if 0 <= code < len(cats) else None


def _details(s: Scored, eng: Engine, fields: Sequence[Any] | None) -> dict[str, Any]:
    """amount_paid, currencies and banks of the alerted event (see the module docstring)."""
    f = fields if fields is not None else getattr(s, "fields", None)
    if f:
        rec = dict(zip(INPUT_COLUMNS, f, strict=True))
        if int(rec["row_id"]) != s.row_id:
            raise ValueError(f"fields of row {rec['row_id']} given for alert row {s.row_id}")
        fb, tb = str(rec["from_bank"]), str(rec["to_bank"])
        return {
            "amount_paid": float(rec["amount_paid"]),
            "payment_currency": str(rec["payment_currency"]),
            "receiving_currency": str(rec["receiving_currency"]),
            "from_bank": fb,
            "to_bank": tb,
            "cross_bank": fb != tb,
        }
    # The engine's transient pending list: (prepared event, new-pair flag) of the current minute;
    # in the on_alert hook its last entry is the alerted event.
    pending = eng.pending
    ev = pending[-1][0] if pending else None
    if ev is None or ev[E_ROW_ID] != s.row_id or ev[E_RANK] != s.rank:
        raise ValueError(
            f"alert row {s.row_id} is not the engine's last pending event: pass its fields"
        )
    vocab = eng.spec.vocab
    return {
        "amount_paid": ev[E_PAID_C] / 100,
        "payment_currency": _decode(vocab["payment_currency"], ev[E_PCUR]),
        "receiving_currency": _decode(vocab["receiving_currency"], ev[E_RCUR]),
        "from_bank": None,
        "to_bank": None,
        "cross_bank": not (ev[E_FLAGS] & F_SAME_BANK),
    }


# --- the pack ---------------------------------------------------------------------------------


def build_case_pack(
    s: Scored,
    eng: Engine,
    ch: Champion,
    *,
    cfg: Mapping[str, Any],
    calibration: Mapping[str, Any] | None = None,
    fields: Sequence[Any] | None = None,
    matcher: TreeMatcher | RulesMatcher | None = None,
) -> dict[str, Any]:
    """The case pack of the alert `s` (keys in the module docstring); call it from the
    on_alert hook, while `eng` is at the alert's minute. `cfg` = configs/explain.yaml (checked);
    `matcher` defaults to `typology_matcher(cfg, ch)`."""
    spec = eng.spec
    minute, src, dst = int(s.minute), int(s.src), int(s.dst)
    day = minute // MINUTES_PER_DAY + 1
    hh, mm = divmod(minute % MINUTES_PER_DAY, 60)
    window = min(int(cfg["window_minutes"]), spec.W_max)
    caps = {"cap_1hop": int(cfg["cap_1hop"]), "cap_2hop": int(cfg["cap_2hop"])}
    sg = subgraph(eng, src, dst, minute, int(s.rank), s.amount_usd, window=window, **caps)
    d = _details(s, eng, fields)
    feats = typology_features(s, ch.names, spec, sg)
    m = typology_matcher(cfg, ch) if matcher is None else matcher
    label, evidence = m.match(feats)
    pack: dict[str, Any] = {
        "pack_format": PACK_FORMAT,
        "id": int(s.row_id),
        "case_key": f"{src}-d{day}",
        "rank": int(s.rank),
        "who": {
            "subject": {"account": src, "bank": d["from_bank"]},
            "counterparty": {"account": dst, "bank": d["to_bank"]},
        },
        "what": {
            "amount_usd": float(s.amount_usd),
            "amount_paid": d["amount_paid"],
            "payment_format": str(s.payment_format),
            "payment_currency": d["payment_currency"],
            "receiving_currency": d["receiving_currency"],
        },
        "when": {"day": day, "time": f"{hh:02d}:{mm:02d}", "minute": minute},
        "where": {
            "from_bank": d["from_bank"],
            "to_bank": d["to_bank"],
            "cross_bank": d["cross_bank"],
        },
        "why": {
            "score": float(s.score),
            "calibrated": calibrate(s.score, calibration),
            "threshold": ch.threshold,
            "rate_tag": ch.alert_tag,
            "rules_fired": list(s.rules),
            **drivers(ch, s.x, int(cfg["top_drivers"])),
            "typology": {
                "label": label,
                "model": m.model,
                "evidence": evidence,
                "features": feats,
            },
        },
        "how": {"subgraph": sg, "window_minutes": window, **caps},
        "model_version": ch.model_version,
        "export_key": ch.export_key,
    }
    pack["narrative"] = narrative(pack)
    return pack


class CaseHook:
    """The scorer's on_alert hook: build the alert's case pack and hand it to `sink` (on the
    consumer thread, e.g. SqliteAlertStore.write_case or list.append).

    strict=False (the app): a failure is logged and counted, never raised, so a case-pack bug
    cannot stop the stream; a frozen tree that reads features these packs lack is logged and the
    decision list labels the cases. strict=True (tests, case_eval): the failure propagates.
    `matcher` is the typology matcher the packs use (`typology_matcher`)."""

    def __init__(
        self,
        ch: Champion,
        cfg: Mapping[str, Any],
        sink: Callable[[dict[str, Any]], Any],
        *,
        calibration: Mapping[str, Any] | None = None,
        strict: bool = False,
    ) -> None:
        self.champion = ch
        self.cfg = check_config(cfg)
        self.sink = sink
        self.calibration = load_calibration(ch) if calibration is None else dict(calibration)
        self.strict = strict
        try:
            self.matcher = typology_matcher(self.cfg, ch)
        except ValueError as e:
            if strict:
                raise
            log.warning("%s: the decision list labels the cases", e)
            self.matcher = rules_matcher(self.cfg, ch.spec)
        self.built = 0
        self.failed = 0
        self.last_error: str | None = None

    def __call__(self, s: Scored, eng: Engine) -> None:
        try:
            pack = build_case_pack(
                s,
                eng,
                self.champion,
                cfg=self.cfg,
                calibration=self.calibration,
                matcher=self.matcher,
            )
            self.sink(pack)
        except Exception as e:
            self.failed += 1
            self.last_error = repr(e)
            if self.strict:
                raise
            log.exception("the case pack of alert row %s failed", s.row_id)
            return
        self.built += 1


def case_hook(
    ch: Champion,
    cfg: Mapping[str, Any],
    sink: Callable[[dict[str, Any]], Any],
    *,
    calibration: Mapping[str, Any] | None = None,
    strict: bool = False,
) -> CaseHook:
    """The on_alert hook (an `AlertHook`) that builds case packs into `sink`. `calibration`
    defaults to the champion's bundle calibration.json (None without a bundle)."""
    return CaseHook(ch, cfg, sink, calibration=calibration, strict=strict)
