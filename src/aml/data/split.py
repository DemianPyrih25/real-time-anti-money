"""Day-boundary split and test views (configs/data.yaml; inclusive 1-based day ranges)."""

from __future__ import annotations

import polars as pl

SPLITS = ("train", "val_early", "val_late", "test")
VIEWS = ("primary", "tail", "full")


def _range(cfg_ranges: dict, name: str) -> tuple[int, int]:
    lo, hi = cfg_ranges[name]
    lo, hi = int(lo), int(hi)
    if lo > hi:
        raise ValueError(f"{name}: empty day range [{lo}, {hi}]")
    return lo, hi


def split_days(data_cfg: dict, split: str) -> tuple[int, int]:
    if split not in SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {SPLITS}")
    return _range(data_cfg["split"], split)


def view_days(data_cfg: dict, view: str) -> tuple[int, int]:
    if view not in VIEWS:
        raise ValueError(f"unknown view {view!r}; expected one of {VIEWS}")
    return _range(data_cfg["test_views"], view)


def n_days(data_cfg: dict, view: str) -> int:
    lo, hi = view_days(data_cfg, view)
    return hi - lo + 1


def split_expr(data_cfg: dict, split: str) -> pl.Expr:
    lo, hi = split_days(data_cfg, split)
    return pl.col("day").is_between(lo, hi, closed="both")


def view_expr(data_cfg: dict, view: str) -> pl.Expr:
    lo, hi = view_days(data_cfg, view)
    return pl.col("day").is_between(lo, hi, closed="both")


def check_split_config(data_cfg: dict) -> None:
    """Raise if split ranges overlap or a test view leaves the test split."""
    ranges = sorted((split_days(data_cfg, s), s) for s in SPLITS)
    for (a, name_a), (b, name_b) in zip(ranges, ranges[1:], strict=False):
        if b[0] <= a[1]:
            raise ValueError(f"split ranges overlap: {name_a} {a} and {name_b} {b}")
    t_lo, t_hi = split_days(data_cfg, "test")
    for v in VIEWS:
        lo, hi = view_days(data_cfg, v)
        if lo < t_lo or hi > t_hi:
            raise ValueError(f"test view {v} [{lo}, {hi}] is outside test [{t_lo}, {t_hi}]")


def assign_split(df: pl.DataFrame, data_cfg: dict) -> pl.DataFrame:
    """Add a String `split` column from `day`; raise if any day falls outside every split."""
    check_split_config(data_cfg)
    expr = pl.lit(None, dtype=pl.String)
    for s in reversed(SPLITS):
        expr = pl.when(split_expr(data_cfg, s)).then(pl.lit(s)).otherwise(expr)
    out = df.with_columns(expr.alias("split"))
    unassigned = out.filter(pl.col("split").is_null())
    if unassigned.height:
        days = sorted(unassigned["day"].unique().to_list())
        raise ValueError(f"{unassigned.height} rows on days {days} are in no split")
    return out
