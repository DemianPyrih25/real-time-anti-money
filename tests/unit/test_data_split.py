"""aml.data.split: day-boundary split and test views."""

from __future__ import annotations

import copy

import polars as pl
import pytest

from aml.data import split
from aml.data.split import SPLITS, VIEWS


def _days(n: int) -> pl.DataFrame:
    return pl.DataFrame({"day": pl.Series(range(1, n + 1), dtype=pl.Int16)})


def test_assign_split_by_day(data_cfg: dict) -> None:
    out = split.assign_split(_days(18), data_cfg)
    got = dict(zip(out["day"].to_list(), out["split"].to_list(), strict=True))
    for s in SPLITS:
        lo, hi = data_cfg["split"][s]
        assert all(got[d] == s for d in range(lo, hi + 1))
    assert out.schema["split"] == pl.String
    assert got[6] == "train" and got[7] == "val_early" and got[8] == "val_late" and got[9] == "test"


def test_unassigned_day_raises(data_cfg: dict) -> None:
    with pytest.raises(ValueError, match="no split"):
        split.assign_split(_days(19), data_cfg)


def test_overlapping_ranges_raise(data_cfg: dict) -> None:
    cfg = copy.deepcopy(data_cfg)
    cfg["split"]["val_early"] = [6, 7]
    with pytest.raises(ValueError, match="overlap"):
        split.assign_split(_days(18), cfg)


def test_view_outside_test_raises(data_cfg: dict) -> None:
    cfg = copy.deepcopy(data_cfg)
    cfg["test_views"]["primary"] = [8, 10]
    with pytest.raises(ValueError, match="outside test"):
        split.check_split_config(cfg)


def test_exprs_and_views(data_cfg: dict) -> None:
    df = _days(18)
    assert df.filter(split.split_expr(data_cfg, "train")).height == 6
    assert df.filter(split.split_expr(data_cfg, "test")).height == 10
    assert split.view_days(data_cfg, "primary") == (9, 10)
    assert split.view_days(data_cfg, "tail") == (11, 18)
    assert {v: split.n_days(data_cfg, v) for v in VIEWS} == {"primary": 2, "tail": 8, "full": 10}
    assert df.filter(split.view_expr(data_cfg, "tail"))["day"].to_list() == list(range(11, 19))
    with pytest.raises(ValueError):
        split.view_days(data_cfg, "train")
    with pytest.raises(ValueError):
        split.split_expr(data_cfg, "val")


def test_prepared_split_column(prepared, data_cfg: dict) -> None:
    tx = pl.read_parquet(prepared.transactions, columns=["day", "split"])
    for s in SPLITS:
        lo, hi = data_cfg["split"][s]
        sub = tx.filter(pl.col("split") == s)
        assert sub.height > 0
        assert sub["day"].min() >= lo and sub["day"].max() <= hi
