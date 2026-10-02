"""aml.data.eda: the report files and the facts they re-derive."""

from __future__ import annotations

import re

import polars as pl
import pytest

from aml.data.eda import DEGREE_QUANTILES, run_eda
from aml.data.split import SPLITS, VIEWS
from aml.io import read_json
from aml.paths import DataPaths

SECTIONS = (
    "overview",
    "per_day",
    "tail_days",
    "csv",
    "rows_per_minute",
    "payment",
    "patterns",
    "hubs",
    "repeat_accounts",
    "splits",
    "train_degree",
    "ties",
)


@pytest.fixture(scope="module")
def eda(prepared) -> dict:
    return read_json(prepared.reports / "eda.json")


@pytest.fixture(scope="module")
def md(prepared) -> str:
    return (prepared.reports / "eda.md").read_text(encoding="utf-8")


def test_files_and_sections(eda, md) -> None:
    assert set(SECTIONS) <= set(eda)
    for heading in (
        "## Overview",
        "## Rows and positives per day",
        "## Tail days",
        "## CSV traps",
        "## Rows per minute",
        "## Payment format and positives",
        "## Patterns file",
        "## Hubs",
        "## Repeat accounts",
        "## Split sizes and prevalence",
        "## Train-period degree distribution",
        "## Same-minute ties",
    ):
        assert heading in md


def test_no_calendar_dates_in_report(md) -> None:
    assert not re.search(r"\d{4}[/-]\d{2}[/-]\d{2}", md)


def test_overview_matches_fixture(eda, synthetic) -> None:
    exp, o = synthetic.expected, eda["overview"]
    assert o["rows"] == exp["rows"] and o["accounts"] == exp["accounts"]
    assert o["positives"] == exp["positives"]
    assert o["positive_pct"] == pytest.approx(100 * exp["positives"] / exp["rows"])
    assert (o["first_day"], o["last_day"], o["n_days"]) == (1, synthetic.n_days, synthetic.n_days)
    per_day = eda["per_day"]
    assert [d["day"] for d in per_day] == list(range(1, synthetic.n_days + 1))
    assert sum(d["rows"] for d in per_day) == exp["rows"]
    assert sum(d["positives"] for d in per_day) == exp["positives"]


def test_tail_days_detected(eda, data_cfg) -> None:
    lo, hi = data_cfg["test_views"]["tail"]
    assert eda["tail_days"]["days"] == list(range(lo, hi + 1))
    assert eda["tail_days"]["matches_config_tail_view"] is True
    assert 0 < eda["tail_days"]["share_of_test_positives"] <= 1


def test_csv_traps(eda) -> None:
    c = eda["csv"]
    assert c["sorted_by_time"] is False and c["inversions"] > 0
    assert c["first_out_of_order_row_id"] >= 1
    assert c["header"]["checked"] and c["header"]["duplicate_names"] == ["Account"]
    assert c["header"]["matches_expected"] is True
    assert c["zero_padded_banks"] > 0
    assert c["bank_codes_colliding_as_int"] >= 1  # "012" and "12"
    assert c["float_looking_account_ids"] > 0
    assert c["account_strings_at_several_banks"] >= 1


def test_payment_and_patterns(eda, synthetic) -> None:
    p = eda["payment"]
    formats = {f["payment_format"]: f for f in p["formats"]}
    assert formats["ACH"]["share_of_positives"] > 0.9
    assert p["reinvestment_self_loop_share"] == 1.0
    assert (p["wire_positives"] or 0) == 0 and p["cross_currency_positives"] == 0
    pat = eda["patterns"]
    assert pat["attempts"] == synthetic.expected["pattern_attempts"]
    assert pat["pattern_transactions"] == synthetic.expected["pattern_transactions"]
    assert pat["other_positives"] == (
        synthetic.expected["positives"] - synthetic.expected["pattern_transactions"]
    )
    assert pat["share_of_positives_covered"] == pytest.approx(
        synthetic.expected["pattern_transactions"] / synthetic.expected["positives"]
    )


def test_splits_and_views(eda, prepared) -> None:
    tx = pl.read_parquet(prepared.transactions, columns=["split"])
    assert set(eda["splits"]) == set(SPLITS) | {f"test_{v}" for v in VIEWS}
    for s in SPLITS:
        assert eda["splits"][s]["rows"] == tx.filter(pl.col("split") == s).height
    assert eda["splits"]["test_full"]["rows"] == eda["splits"]["test"]["rows"]
    r = eda["repeat_accounts"]
    assert r["train_launderer_accounts"] > 0
    assert 0 <= r["test"]["share_touching"] <= 1


def test_train_degree_and_hubs(eda, prepared, data_cfg) -> None:
    deg = eda["train_degree"]
    lo, hi = data_cfg["split"]["train"]
    tx = pl.read_parquet(prepared.transactions).filter(pl.col("day").is_between(lo, hi))
    out_deg = tx.group_by("src").len()
    assert deg["out_deg"]["max"] == out_deg["len"].max()
    for col in ("in_deg", "out_deg", "total_deg"):
        qs = [deg[col][f"q{q:g}"] for q in DEGREE_QUANTILES]
        assert qs == sorted(qs) and qs[-1] <= deg[col]["max"]
    all_out = pl.read_parquet(prepared.transactions).group_by("src").len()
    assert eda["hubs"]["max_out_degree"] == all_out["len"].max()


def test_ties(eda, prepared) -> None:
    tx = pl.read_parquet(prepared.transactions, columns=["minute", "src"])
    shared = tx.filter(pl.len().over("minute") > 1).height / tx.height
    shared_src = tx.filter(pl.len().over("src", "minute") > 1).height / tx.height
    assert eda["ties"]["share_rows_in_shared_minute"] == pytest.approx(shared)
    assert eda["ties"]["share_rows_in_shared_src_minute"] == pytest.approx(shared_src)
    assert eda["ties"]["share_rows_in_shared_minute"] > 0


def test_run_eda_without_raw_csv(prepared, data_cfg, tmp_path) -> None:
    # Same Parquet, a root with no raw files: the header check is skipped, not failed.
    paths = DataPaths(tmp_path)
    for src, dst in (
        (prepared.transactions, paths.transactions),
        (prepared.accounts, paths.accounts),
        (prepared.labels, paths.labels),
    ):
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes())
    out = run_eda(paths, data_cfg, threads=1)
    assert out["csv"]["header"] == {"checked": False}
    assert (paths.reports / "eda.md").exists() and (paths.reports / "eda.json").exists()
