"""aml.data.prepare: end-to-end on the synthetic fixture (the `prepared` session fixture)."""

from __future__ import annotations

import copy
import csv
import shutil
from datetime import datetime

import numpy as np
import polars as pl
import pytest

from aml.data.prepare import TRANSACTION_COLUMNS, prepare_data
from aml.io import read_json
from aml.paths import DataPaths


@pytest.fixture(scope="module")
def tx(prepared) -> pl.DataFrame:
    return pl.read_parquet(prepared.transactions)


@pytest.fixture(scope="module")
def labels(prepared) -> pl.DataFrame:
    return pl.read_parquet(prepared.labels)


def test_counts_equal_fixture(prepared, tx, labels, synthetic) -> None:
    exp = synthetic.expected
    acc = pl.read_parquet(prepared.accounts)
    assert tx.height == exp["rows"]
    assert acc.height == exp["accounts"]
    assert int(labels["is_laundering"].sum()) == exp["positives"]
    assert labels["attempt_id"].drop_nulls().n_unique() == exp["pattern_attempts"]
    assert labels["attempt_id"].is_not_null().sum() == exp["pattern_transactions"]


def test_transactions_have_no_label_column(tx) -> None:
    assert tx.columns == TRANSACTION_COLUMNS
    banned = ("launder", "label", "typology", "attempt")
    assert not [c for c in tx.columns if any(b in c for b in banned)]


def test_transactions_sorted_by_stable_rank(tx) -> None:
    assert tx["rank"].to_list() == list(range(tx.height))
    order = np.lexsort((tx["row_id"].to_numpy(), tx["minute"].to_numpy()))
    assert np.array_equal(order, np.arange(tx.height))
    assert tx["ts"].is_sorted()
    assert tx["row_id"].sort().to_list() == list(range(tx.height))


def test_dtypes(tx) -> None:
    s = tx.schema
    assert s["row_id"] == pl.Int64 and s["rank"] == pl.Int64 and s["minute"] == pl.Int64
    assert s["day"] == pl.Int16 and s["src"] == pl.Int32 and s["dst"] == pl.Int32
    assert s["ts"] == pl.Datetime("us") and s["amount_usd"] == pl.Float64
    for c in ("from_bank", "from_account", "to_bank", "to_account", "split"):
        assert s[c] == pl.String


def test_labels_one_row_per_transaction(tx, labels) -> None:
    assert labels.height == tx.height
    assert labels["row_id"].to_list() == list(range(tx.height))


def test_exact_duplicate_row_kept(tx) -> None:
    raw_cols = [
        "ts",
        "from_bank",
        "from_account",
        "to_bank",
        "to_account",
        "amount_received",
        "receiving_currency",
        "amount_paid",
        "payment_currency",
        "payment_format",
    ]
    dup = tx.filter(pl.struct(raw_cols).is_duplicated())
    assert dup.height >= 2
    assert dup["row_id"].is_unique().all()


def test_fx_json(prepared, synthetic) -> None:
    fx = read_json(prepared.fx_rates)
    assert fx["base_currency"] == "US Dollar" and fx["fit_split"] == "train"
    assert fx["n_pairs"] > 0
    for cur, true in synthetic.fx_units_per_usd.items():
        assert fx["units_per_base"][cur] == pytest.approx(true, rel=0.01)


def test_summary_written(prepared, synthetic) -> None:
    summary = read_json(prepared.parquet_dir / "prepare_summary.json")
    assert summary["counts"] == synthetic.expected
    assert sum(v["rows"] for v in summary["splits"].values()) == synthetic.expected["rows"]
    assert (
        sum(v["positives"] for v in summary["splits"].values()) == (synthetic.expected["positives"])
    )
    assert "timings_s" in summary


def _raw_paths(tmp_path, synthetic) -> DataPaths:
    paths = DataPaths(tmp_path)
    paths.raw_dir.mkdir(parents=True)
    for f in (synthetic.transactions_csv, synthetic.patterns_txt):
        shutil.copy(f, paths.raw_dir / f.name)
    return paths


def test_count_mismatch_raises_before_writing(tmp_path, synthetic, data_cfg) -> None:
    paths = _raw_paths(tmp_path, synthetic)
    cfg = copy.deepcopy(data_cfg)
    cfg["expected"]["accounts"] += 1
    with pytest.raises(ValueError, match="accounts"):
        prepare_data(paths, cfg, threads=1, run_eda=False)
    assert not paths.transactions.exists()


def _scale_non_train_cross_currency(csv_path, train_last_day: int, factor: float) -> int:
    """Multiply Amount Received on cross-currency rows after the train days, in place.

    Uses the csv module: polars would rename the duplicate `Account` header, which ingest rejects.
    """
    with csv_path.open(newline="", encoding="utf-8") as f:
        header, *body = list(csv.reader(f))
    days = [datetime.strptime(r[0], "%Y/%m/%d %H:%M").date() for r in body]
    first = min(days)
    changed = 0
    for r, d in zip(body, days, strict=True):
        # columns: 5 Amount Received, 6 Receiving Currency, 8 Payment Currency
        if (d - first).days + 1 > train_last_day and r[6] != r[8]:
            r[5] = f"{float(r[5]) * factor:.2f}"
            changed += 1
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        csv.writer(f, lineterminator="\n").writerows([header, *body])
    return changed


def test_fx_is_fitted_on_train_rows_only(prepared, tmp_path, synthetic, data_cfg) -> None:
    """PLAN.md §4: preprocessing is fitted on train only. Changing the FX implied by every
    val/test cross-currency row must leave the fitted rates and every amount_usd unchanged."""
    paths = _raw_paths(tmp_path, synthetic)
    csv_path = paths.raw_dir / synthetic.transactions_csv.name
    changed = _scale_non_train_cross_currency(csv_path, data_cfg["split"]["train"][1], 3.0)
    assert changed > 0
    prepare_data(paths, data_cfg, threads=1, run_eda=False)
    base_fx, new_fx = read_json(prepared.fx_rates), read_json(paths.fx_rates)
    assert new_fx["units_per_base"] == base_fx["units_per_base"]  # exactly
    cols = ["row_id", "split", "amount_usd", "amount_received"]
    a = pl.read_parquet(prepared.transactions, columns=cols)
    b = pl.read_parquet(paths.transactions, columns=cols)
    assert a["row_id"].to_list() == b["row_id"].to_list()
    assert a["amount_usd"].to_list() == b["amount_usd"].to_list()
    # The perturbation is real: some non-train amounts moved, no train amount did.
    moved = a["amount_received"] != b["amount_received"]
    assert moved.any()
    assert (a.filter(moved)["split"] != "train").all()


def test_fx_fit_sees_train_rows_only(tmp_path, synthetic, data_cfg, monkeypatch) -> None:
    from aml.data import ingest

    real_fit_fx = ingest.fit_fx
    seen: list[set[str]] = []

    def spy(df, base_currency):
        seen.append(set(df["split"].unique().to_list()))
        return real_fit_fx(df, base_currency)

    monkeypatch.setattr(ingest, "fit_fx", spy)
    prepare_data(_raw_paths(tmp_path, synthetic), data_cfg, threads=1, run_eda=False)
    assert seen == [{"train"}]


@pytest.mark.parametrize("split_name", ["val_early", "test", "all"])
def test_fx_fit_split_must_be_train(tmp_path, synthetic, data_cfg, split_name) -> None:
    paths = _raw_paths(tmp_path, synthetic)
    cfg = copy.deepcopy(data_cfg)
    cfg["fx"]["fit_split"] = split_name
    with pytest.raises(ValueError, match="train only"):
        prepare_data(paths, cfg, threads=1, run_eda=False)
    assert not paths.transactions.exists()


def test_null_expected_values_are_skipped(tmp_path, synthetic, data_cfg) -> None:
    paths = _raw_paths(tmp_path, synthetic)
    cfg = copy.deepcopy(data_cfg)
    cfg["expected"] = {k: None for k in cfg["expected"]}
    summary = prepare_data(paths, cfg, threads=1, run_eda=False)
    assert summary["counts"] == synthetic.expected
    assert "eda" not in summary
    assert not (paths.reports / "eda.md").exists()
