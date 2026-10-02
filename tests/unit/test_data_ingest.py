"""aml.data.ingest: raw strings, typing, time, rank, accounts and FX."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest

from aml.data import ingest, split
from aml.data.ingest import RAW_COLUMNS

HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
    "Amount Paid,Payment Currency,Payment Format,Is Laundering"
)
# Rows 0-3 are real lines from HI-Small_Trans.csv; the rest exercise the traps.
ROWS = [
    "2022/09/01 00:20,010,8000EBD30,010,8000EBD30,3697.34,US Dollar,3697.34,US Dollar,Reinvestment,0",  # noqa: E501 (verbatim data line)
    "2022/09/01 00:20,03208,8000F4580,001,8000F5340,0.01,US Dollar,0.01,US Dollar,Cheque,0",
    "2022/09/01 00:00,03209,8000F4670,03209,8000F4670,14675.57,US Dollar,14675.57,US Dollar,Reinvestment,0",  # noqa: E501 (verbatim data line)
    "2022/09/01 00:02,012,8000F5030,012,8000F5030,2806.97,US Dollar,2806.97,US Dollar,Reinvestment,0",  # noqa: E501 (verbatim data line)
    "2022/09/02 23:59,012,8003E4680,12,8003E4680,100.00,US Dollar,100.00,US Dollar,ACH,1",
    "2022/09/01 00:20,12,8003E4680,012,8000F5030,85.00,Euro,100.00,US Dollar,Wire,0",
]


def _write_csv(tmp_path: Path, rows: list[str] = ROWS, header: str = HEADER) -> Path:
    p = tmp_path / "trans.csv"
    p.write_text("\n".join([header, *rows]) + "\n", encoding="utf-8")
    return p


@pytest.fixture()
def small_raw(tmp_path: Path) -> pl.DataFrame:
    return ingest.read_raw_transactions(_write_csv(tmp_path))


def test_read_raw_all_strings_and_row_id(small_raw: pl.DataFrame) -> None:
    assert small_raw.columns == ["row_id", *RAW_COLUMNS]
    assert small_raw["row_id"].dtype == pl.Int64
    assert small_raw["row_id"].to_list() == list(range(len(ROWS)))
    assert all(small_raw[c].dtype == pl.String for c in RAW_COLUMNS)


def test_raw_strings_preserved_verbatim(small_raw: pl.DataFrame) -> None:
    # Zero-padded banks and float-looking hex ids survive untouched.
    assert small_raw["from_bank"].to_list()[:4] == ["010", "03208", "03209", "012"]
    assert small_raw.row(4, named=True)["from_account"] == "8003E4680"
    assert small_raw.row(4, named=True)["to_bank"] == "12"
    assert small_raw.row(1, named=True)["amount_paid"] == "0.01"


def test_crlf_csv(tmp_path: Path) -> None:
    p = tmp_path / "crlf.csv"
    p.write_bytes(("\r\n".join([HEADER, *ROWS]) + "\r\n").encode())
    raw = ingest.read_raw_transactions(p)
    assert raw.height == len(ROWS)
    assert raw["is_laundering"].to_list() == [r[-1] for r in ROWS]


def test_bad_header_raises(tmp_path: Path) -> None:
    bad = HEADER.replace("From Bank,Account", "From Bank,From Account")
    with pytest.raises(ValueError, match="header"):
        ingest.read_raw_transactions(_write_csv(tmp_path, header=bad))


def test_parse_types(small_raw: pl.DataFrame) -> None:
    tx = ingest.parse_transactions(small_raw)
    assert tx.schema["ts"] == pl.Datetime("us")
    assert tx.schema["amount_paid"] == pl.Float64
    assert tx.schema["is_laundering"] == pl.Int8
    assert tx.schema["from_bank"] == pl.String
    assert tx["amount_received"][5] == pytest.approx(85.0)


def test_parse_rejects_bad_label(tmp_path: Path) -> None:
    rows = [*ROWS[:1], ROWS[1][:-1] + "2"]
    raw = ingest.read_raw_transactions(_write_csv(tmp_path, rows=rows))
    with pytest.raises(ValueError, match="is_laundering"):
        ingest.parse_transactions(raw)


def test_assign_time(small_raw: pl.DataFrame) -> None:
    tx = ingest.assign_time(ingest.parse_transactions(small_raw))
    assert tx["minute"].to_list() == [20, 20, 0, 2, 1440 + 1439, 20]
    assert tx["day"].to_list() == [1, 1, 1, 1, 2, 1]
    assert tx.schema["day"] == pl.Int16 and tx.schema["minute"] == pl.Int64


def test_rank_is_stable_sort_by_ts_then_row_id(small_raw: pl.DataFrame) -> None:
    tx = ingest.sort_and_rank(ingest.assign_time(ingest.parse_transactions(small_raw)))
    assert tx.columns[:2] == ["row_id", "rank"]
    assert tx["rank"].to_list() == list(range(len(ROWS)))
    # Minute 20 holds rows 0, 1 and 5: ties are broken by row_id.
    assert tx["row_id"].to_list() == [2, 3, 0, 1, 5, 4]


def test_rank_on_synthetic(synthetic) -> None:
    raw = ingest.read_raw_transactions(synthetic.transactions_csv)
    tx = ingest.sort_and_rank(ingest.assign_time(ingest.parse_transactions(raw)))
    assert tx.height == synthetic.expected["rows"]
    order = np.lexsort((raw["row_id"].to_numpy(), tx.sort("row_id")["minute"].to_numpy()))
    assert np.array_equal(tx["row_id"].to_numpy(), order)
    assert tx["rank"].is_unique().all()
    # The file really is unsorted, so the sort is not a no-op.
    assert not tx["row_id"].is_sorted()


def test_accounts_keyed_by_bank_and_account(small_raw: pl.DataFrame) -> None:
    tx = ingest.parse_transactions(small_raw)
    acc = ingest.build_accounts(tx)
    keys = list(zip(acc["bank"], acc["account"], strict=True))
    assert keys == sorted(keys)
    assert acc["account_id"].to_list() == list(range(acc.height))
    # Banks "012" and "12" stay distinct, so the same id string gives two accounts.
    assert ("012", "8003E4680") in keys and ("12", "8003E4680") in keys
    assert acc.height == len(set(keys))

    mapped = ingest.map_accounts(tx, acc)
    assert mapped.schema["src"] == pl.Int32 and mapped.schema["dst"] == pl.Int32
    assert mapped["row_id"].to_list() == tx["row_id"].to_list()
    r4 = mapped.row(4, named=True)
    assert r4["src"] != r4["dst"]  # 012/8003E4680 -> 12/8003E4680 is not a self-loop
    r0 = mapped.row(0, named=True)
    assert r0["src"] == r0["dst"]
    lookup = {(b, a): i for i, b, a in acc.iter_rows()}
    for r in mapped.iter_rows(named=True):
        assert lookup[(r["from_bank"], r["from_account"])] == r["src"]
        assert lookup[(r["to_bank"], r["to_account"])] == r["dst"]


def test_synthetic_accounts(synthetic) -> None:
    tx = ingest.parse_transactions(ingest.read_raw_transactions(synthetic.transactions_csv))
    acc = ingest.build_accounts(tx)
    assert acc.height == synthetic.expected["accounts"]
    banks = set(acc["bank"].to_list())
    assert {"012", "12"} <= banks
    assert acc.group_by("account").len().filter(pl.col("len") > 1).height >= 1
    float_like = acc.filter(pl.col("account").str.contains(r"^\d+E\d+$"))
    assert float_like.height > 0


def test_fx_recovers_fixture_rates(synthetic, data_cfg: dict) -> None:
    tx = ingest.assign_time(
        ingest.parse_transactions(ingest.read_raw_transactions(synthetic.transactions_csv))
    )
    tx = split.assign_split(tx, data_cfg)
    base = data_cfg["fx"]["base_currency"]
    fx = ingest.fit_fx(tx.filter(pl.col("split") == "train"), base)
    units = fx["units_per_base"]
    assert fx["n_pairs"] > 0
    assert set(units) == set(synthetic.fx_units_per_usd)
    for cur, true in synthetic.fx_units_per_usd.items():
        assert units[cur] == pytest.approx(true, rel=0.01), cur

    out = ingest.apply_fx(tx, units)
    expected = out["amount_paid"] / out["payment_currency"].replace_strict(units)
    assert np.allclose(out["amount_usd"].to_numpy(), expected.to_numpy())
    usd = out.filter(pl.col("payment_currency") == base)
    assert np.array_equal(usd["amount_usd"].to_numpy(), usd["amount_paid"].to_numpy())
    # Both legs of a cross-currency row convert to (nearly) the same USD amount.
    cross = out.filter(
        (pl.col("payment_currency") != pl.col("receiving_currency")) & (pl.col("amount_paid") > 100)
    )
    via_recv = cross["amount_received"] / cross["receiving_currency"].replace_strict(units)
    assert np.allclose(cross["amount_usd"].to_numpy(), via_recv.to_numpy(), rtol=0.02)


def _fx_frame(rows: list[tuple[str, str, float, float]]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=["payment_currency", "receiving_currency", "amount_paid", "amount_received"],
        orient="row",
    )


def test_fit_fx_chain_and_median() -> None:
    # Euro links to base directly; Yen only through Euro. An outlier does not move the median.
    df = _fx_frame(
        [
            ("US Dollar", "Euro", 100.0, 85.0),
            ("US Dollar", "Euro", 200.0, 170.0),
            ("US Dollar", "Euro", 10.0, 999.0),
            ("Euro", "Yen", 85.0, 15000.0),
            ("US Dollar", "US Dollar", 5.0, 5.0),
        ]
    )
    units = ingest.fit_fx(df, "US Dollar")["units_per_base"]
    assert units["US Dollar"] == 1.0
    assert units["Euro"] == pytest.approx(0.85, rel=1e-9)
    assert units["Yen"] == pytest.approx(150.0, rel=1e-9)


def test_fit_fx_unreachable_currency_raises() -> None:
    df = _fx_frame([("US Dollar", "Euro", 100.0, 85.0), ("Rupee", "Rupee", 10.0, 10.0)])
    with pytest.raises(ValueError, match="Rupee"):
        ingest.fit_fx(df, "US Dollar")


def test_apply_fx_missing_rate_raises() -> None:
    df = _fx_frame([("Bitcoin", "Bitcoin", 1.0, 1.0)])
    with pytest.raises(ValueError, match="Bitcoin"):
        ingest.apply_fx(df, {"US Dollar": 1.0})
