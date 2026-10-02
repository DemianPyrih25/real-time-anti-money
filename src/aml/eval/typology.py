"""Typology, attempt-level and memorisation breakdowns (PLAN.md §4 "Metrics")."""

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl

# The 8 pattern typologies of Patterns.txt, then "OTHER" = integration positives not in it.
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
ALL = "ALL"


def attempt_ids(attempt_id: Any) -> np.ndarray:
    """int64 attempt ids with -1 for "not a pattern row" (accepts polars nulls, NaN or None)."""
    if isinstance(attempt_id, pl.Series):
        return attempt_id.cast(pl.Int64).fill_null(-1).to_numpy()
    a = np.asarray(attempt_id)
    if a.dtype == object:
        a = np.array([-1 if v is None else v for v in a], dtype=np.float64)
    if a.dtype.kind == "f":
        a = np.where(np.isnan(a), -1, a)
    return a.astype(np.int64)


def _typology_series(typology: Any, n: int) -> pl.Series:
    t = typology if isinstance(typology, pl.Series) else pl.Series(list(typology), dtype=pl.String)
    if t.len() != n:
        raise ValueError("typology has the wrong length")
    return t.cast(pl.String).alias("t")


def _rate(num: float, den: float) -> float:
    return float(num / den) if den > 0 else float("nan")


def recall_by_typology(flags: np.ndarray, y: np.ndarray, typology: Any) -> dict[str, dict]:
    """{typology: {positives, detected, recall}} over positives, all 9 typologies plus ALL.

    Positives with a null typology count as OTHER.
    """
    f = np.asarray(flags) > 0
    y = np.asarray(y).astype(bool)
    df = pl.DataFrame({"t": _typology_series(typology, y.size), "f": f, "y": y}).filter(pl.col("y"))
    df = df.with_columns(pl.col("t").fill_null("OTHER"))
    counts = {
        r["t"]: (r["positives"], r["detected"])
        for r in df.group_by("t")
        .agg(positives=pl.len(), detected=pl.col("f").sum())
        .iter_rows(named=True)
    }
    unknown = set(counts) - set(TYPOLOGIES)
    if unknown:
        raise ValueError(f"unknown typologies: {sorted(unknown)}")
    out = {}
    for t in (*TYPOLOGIES, ALL):
        p, d = (df.height, int(df["f"].sum())) if t == ALL else counts.get(t, (0, 0))
        out[t] = {"positives": int(p), "detected": int(d), "recall": _rate(d, p)}
    return out


def attempt_detection(
    flags: np.ndarray, attempt_id: Any, typology: Any, minute: np.ndarray
) -> dict[str, dict]:
    """Attempt-level detection per typology (8 pattern typologies plus ALL).

    Attempts = those with >= 1 row in the given rows (the view). An attempt is detected if any of
    its rows is alerted; time to first alert = minute of its first alerted row - minute of its
    first row in the view (median and mean over detected attempts).
    """
    a = attempt_ids(attempt_id)
    df = pl.DataFrame(
        {
            "a": a,
            "t": _typology_series(typology, a.size),
            "m": np.asarray(minute, dtype=np.int64),
            "f": np.asarray(flags) > 0,
        }
    ).filter(pl.col("a") >= 0)
    per_attempt = df.group_by("a").agg(
        t=pl.col("t").first(),
        first_minute=pl.col("m").min(),
        first_alert=pl.col("m").filter(pl.col("f")).min(),
    )
    per_attempt = per_attempt.with_columns(ttfa=pl.col("first_alert") - pl.col("first_minute"))
    out = {}
    for t in (*TYPOLOGIES[:-1], ALL):
        sub = per_attempt if t == ALL else per_attempt.filter(pl.col("t") == t)
        ttfa = sub["ttfa"].drop_nulls()
        n, d = sub.height, ttfa.len()
        out[t] = {
            "attempts": int(n),
            "detected": int(d),
            "detection_rate": _rate(d, n),
            "median_minutes_to_first_alert": float(ttfa.median()) if d else float("nan"),
            "mean_minutes_to_first_alert": float(ttfa.mean()) if d else float("nan"),
        }
    return out


def memorisation_split(flags: np.ndarray, y: np.ndarray, seen_launderer: np.ndarray) -> dict:
    """Recall on positives that touch an account that laundered in train ("seen") vs the rest."""
    f = np.asarray(flags) > 0
    y = np.asarray(y).astype(bool)
    seen = np.asarray(seen_launderer).astype(bool)
    out: dict[str, Any] = {}
    for name, mask in (("seen", y & seen), ("unseen", y & ~seen)):
        p, d = int(mask.sum()), int((mask & f).sum())
        out[name] = {"positives": p, "detected": d, "recall": _rate(d, p)}
    out["seen_share"] = _rate(out["seen"]["positives"], int(y.sum()))
    return out
