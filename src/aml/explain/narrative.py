"""The template narrative of a case pack (M6 spec §3): deterministic text in 5W+H order.

`narrative(pack)` reads only the pack (the JSON is the source of truth) and never its own
"narrative" key, so a pack read back from JSON gives the same text. No LLM, no randomness: the
same pack always gives the same string.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

_UNITS = ((1440, "day"), (60, "hour"), (1, "minute"))


def plural(n: int, word: str) -> str:
    """plural(1, "account") -> "1 account"; plural(3, "account") -> "3 accounts"."""
    return f"{n} {word}" if n == 1 else f"{n} {word}s"


def duration_words(minutes: int) -> str:
    """A window length in words, for "in the previous ...": 1440 -> "day", 4320 -> "3 days",
    720 -> "12 hours", 90 -> "90 minutes"."""
    m = int(minutes)
    for size, unit in _UNITS:
        if m >= size and m % size == 0:
            n = m // size
            return unit if n == 1 else plural(n, unit)
    raise ValueError(f"a window must be a positive number of minutes, got {minutes!r}")


def _num(x: Any) -> str:
    """A feature value as text: whole numbers without decimals, others with 4 significant
    digits, None (undefined) as n/a."""
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return "n/a"
    v = float(x)
    return str(int(v)) if v.is_integer() and abs(v) < 1e15 else f"{v:.4g}"


def _bank(party: Mapping[str, Any]) -> str:
    bank = party.get("bank")
    return f" (bank {bank})" if bank is not None else ""


def _who_what_when(pack: Mapping[str, Any]) -> str:
    who, what, when = pack["who"], pack["what"], pack["when"]
    sub, cp = who["subject"], who["counterparty"]
    to = (
        "to itself" if sub["account"] == cp["account"] else f"to account {cp['account']}{_bank(cp)}"
    )
    text = (
        f"On day {when['day']} at {when['time']}, account {sub['account']}{_bank(sub)} sent "
        f"{what['amount_usd']:,.2f} USD by {what['payment_format']} {to}."
    )
    pc, rc, paid = what.get("payment_currency"), what.get("receiving_currency"), what["amount_paid"]
    if pc is not None and rc is not None and rc != pc:
        text += f" The sender paid {paid:,.2f} {pc}; the receiver was credited in {rc}."
    elif pc is not None and pc != "US Dollar":
        text += f" The sender paid {paid:,.2f} {pc}."
    return text


def _where(pack: Mapping[str, Any]) -> str:
    where = pack["where"]
    fb, tb, cross = where.get("from_bank"), where.get("to_bank"), where.get("cross_bank")
    if cross is None:
        return ""
    if cross:
        return f"It crossed banks ({fb} to {tb})." if fb is not None else "It crossed banks."
    return f"Both accounts are at bank {fb}." if fb is not None else "Both accounts share a bank."


def _score(why: Mapping[str, Any]) -> str:
    text = f"The model scored it {why['score']:.4f}"
    if why.get("calibrated") is not None:
        text += f" (calibrated probability {why['calibrated']:.3f})"
    thr = why.get("threshold")
    if thr is None:
        return text + "."
    return text + f", at or above the alert threshold {thr:.4f} (rate tag {why['rate_tag']})."


def _drivers(why: Mapping[str, Any]) -> str:
    drivers: Sequence[Mapping[str, Any]] = why.get("drivers") or ()
    if not drivers:
        return ""
    items = "; ".join(
        f"{d['feature']} = {d.get('display') or _num(d.get('value'))} ({d['contribution']:+.3f})"
        for d in drivers
    )
    return f"Top drivers (log-odds contributions): {items}; base value {why['base_value']:+.3f}."


def _rules(why: Mapping[str, Any]) -> str:
    rules = list(why.get("rules_fired") or ())
    return f"Rules fired: {', '.join(rules)}." if rules else "No rule fired."


def _typology(why: Mapping[str, Any]) -> str:
    typ = why["typology"]
    evidence = list(typ.get("evidence") or ())
    detail = f" ({'; '.join(evidence)})" if evidence else ""
    by = " (decision tree)" if typ.get("model") == "tree" else ""
    return f"Matched typology{by}: {typ['label']}{detail}."


def _how(pack: Mapping[str, Any]) -> str:
    how = pack["how"]
    sg = how["subgraph"]
    n_hist = sum(1 for e in sg["edges"] if e["kind"] != "alert")
    span = duration_words(how["window_minutes"])
    if n_hist == 0:
        return f"Neither account has an earlier payment in the {span} before the alert."
    return (
        f"The causal subgraph shows {plural(len(sg['nodes']), 'account')} and "
        f"{plural(n_hist, 'earlier payment')} from the {span} before the alert."
    )


def narrative(pack: Mapping[str, Any]) -> str:
    """Who / what / when, where, why (score, drivers, rules, typology), how (the subgraph)."""
    why = pack["why"]
    parts = (
        _who_what_when(pack),
        _where(pack),
        _score(why),
        _drivers(why),
        _rules(why),
        _typology(why),
        _how(pack),
    )
    return " ".join(p for p in parts if p)
