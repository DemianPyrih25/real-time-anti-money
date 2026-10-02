"""aml.data.patterns: Patterns.txt parsing and the labels table."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from aml.data import ingest, patterns
from aml.data.ingest import RAW_COLUMNS
from aml.data.patterns import OTHER, TYPOLOGIES

# Real excerpt from HI-Small_Patterns.txt (first rows of two blocks), plus a bare header block.
FAN_OUT_1 = "2022/09/01 00:06,021174,800737690,012,80011F990,2848.96,Euro,2848.96,Euro,ACH,1"
FAN_OUT_2 = "2022/09/01 04:33,021174,800737690,020,80020C5B0,8630.40,Euro,8630.40,Euro,ACH,1"
CYCLE_1 = "2022/09/01 00:03,01467,8013C4030,020,80BC62F10,58702.10,Yuan,58702.10,Yuan,ACH,1"
STACK_1 = "2022/09/01 01:00,0222,812D127D0,0050202,812D129C0,20085.71,Saudi Riyal,20085.71,Saudi Riyal,ACH,1"  # noqa: E501 (verbatim data line)

PATTERNS_TEXT = "\n".join(
    [
        "BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out",
        FAN_OUT_1,
        FAN_OUT_2,
        "END LAUNDERING ATTEMPT - FAN-OUT",
        "",
        "BEGIN LAUNDERING ATTEMPT - CYCLE:  Max 10 hops",
        CYCLE_1,
        "END LAUNDERING ATTEMPT - CYCLE",
        "",
        "BEGIN LAUNDERING ATTEMPT - STACK",
        STACK_1,
        STACK_1,  # an exact duplicate row inside an attempt
        "END LAUNDERING ATTEMPT - STACK",
        "",
    ]
)

HEADER = (
    "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,"
    "Amount Paid,Payment Currency,Payment Format,Is Laundering"
)
NEG = "2022/09/01 00:20,010,8000EBD30,010,8000EBD30,3697.34,US Dollar,3697.34,US Dollar,Reinvestment,0"  # noqa: E501 (verbatim data line)
INTEGRATION = "2022/09/01 02:00,010,8000EBD30,012,80011F990,500.00,US Dollar,500.00,US Dollar,ACH,1"


def _raw_tx(tmp_path: Path, rows: list[str]) -> pl.DataFrame:
    p = tmp_path / "t.csv"
    p.write_text("\n".join([HEADER, *rows]) + "\n", encoding="utf-8")
    return ingest.read_raw_transactions(p)


def test_parse_headers_details_and_rows() -> None:
    df, n = patterns.parse_patterns_text(PATTERNS_TEXT)
    assert n == 3
    assert df.columns == ["attempt_id", "typology", "typology_detail", *RAW_COLUMNS]
    assert df.schema["attempt_id"] == pl.Int32
    assert all(df.schema[c] == pl.String for c in RAW_COLUMNS)
    assert df["attempt_id"].to_list() == [0, 0, 1, 2, 2]
    assert df["typology"].to_list() == ["FAN-OUT", "FAN-OUT", "CYCLE", "STACK", "STACK"]
    assert df["typology_detail"].to_list() == [
        "Max 16-degree Fan-Out",
        "Max 16-degree Fan-Out",
        "Max 10 hops",
        None,
        None,
    ]
    first = df.row(0, named=True)
    assert [first[c] for c in RAW_COLUMNS] == FAN_OUT_1.split(",")


def test_parse_crlf_equals_lf(tmp_path: Path) -> None:
    lf = tmp_path / "lf.txt"
    crlf = tmp_path / "crlf.txt"
    lf.write_bytes(PATTERNS_TEXT.encode())
    crlf.write_bytes(PATTERNS_TEXT.replace("\n", "\r\n").encode())
    a, b = patterns.parse_patterns(lf), patterns.parse_patterns(crlf)
    assert a.equals(b)
    assert not b["is_laundering"].str.contains("\r").any()


@pytest.mark.parametrize(
    ("text", "match"),
    [
        (
            "BEGIN LAUNDERING ATTEMPT - CYCLE\n" + CYCLE_1 + "\nEND LAUNDERING ATTEMPT - STACK\n",
            "END",
        ),
        ("END LAUNDERING ATTEMPT - CYCLE\n", "without BEGIN"),
        ("BEGIN LAUNDERING ATTEMPT - CYCLE\n" + CYCLE_1 + "\n", "unterminated"),
        (CYCLE_1 + "\n", "outside"),
        ("BEGIN LAUNDERING ATTEMPT - CYCLE\n1,2,3\nEND LAUNDERING ATTEMPT - CYCLE\n", "fields"),
        ("BEGIN LAUNDERING ATTEMPT - SMURF\nEND LAUNDERING ATTEMPT - SMURF\n", "typology"),
        (
            "BEGIN LAUNDERING ATTEMPT - CYCLE\nBEGIN LAUNDERING ATTEMPT - CYCLE\n",
            "inside an open block",
        ),
    ],
)
def test_parse_errors(text: str, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        patterns.parse_patterns_text(text)


def test_build_labels_duplicates_and_other(tmp_path: Path) -> None:
    # The STACK row appears twice in both files: the pairs match 1:1 by occurrence order.
    rows = [NEG, STACK_1, FAN_OUT_2, INTEGRATION, CYCLE_1, STACK_1, FAN_OUT_1, NEG]
    raw = _raw_tx(tmp_path, rows)
    pat, _ = patterns.parse_patterns_text(PATTERNS_TEXT)
    labels = patterns.build_labels(raw, pat)

    assert labels.columns == [
        "row_id",
        "is_laundering",
        "attempt_id",
        "typology",
        "attempt_size",
        "typology_detail",
    ]
    assert labels["row_id"].to_list() == list(range(len(rows)))
    assert labels["is_laundering"].to_list() == [0, 1, 1, 1, 1, 1, 1, 0]
    assert labels["attempt_id"].to_list() == [None, 2, 0, None, 1, 2, 0, None]
    assert labels["typology"].to_list() == [
        None,
        "STACK",
        "FAN-OUT",
        OTHER,
        "CYCLE",
        "STACK",
        "FAN-OUT",
        None,
    ]
    assert labels["attempt_size"].to_list() == [None, 2, 2, None, 1, 2, 2, None]
    assert labels["typology_detail"][2] == "Max 16-degree Fan-Out"
    assert labels.schema["attempt_id"] == pl.Int32
    assert labels.schema["attempt_size"] == pl.Int32
    assert labels.schema["is_laundering"] == pl.Int8


def test_build_labels_unmatched_pattern_row_raises(tmp_path: Path) -> None:
    # The file lists STACK_1 twice but the transactions hold it once.
    raw = _raw_tx(tmp_path, [STACK_1, FAN_OUT_1, FAN_OUT_2, CYCLE_1])
    pat, _ = patterns.parse_patterns_text(PATTERNS_TEXT)
    with pytest.raises(ValueError, match="match no transaction"):
        patterns.build_labels(raw, pat)


def test_build_labels_pattern_row_must_be_positive(tmp_path: Path) -> None:
    text = "BEGIN LAUNDERING ATTEMPT - CYCLE\n" + NEG + "\nEND LAUNDERING ATTEMPT - CYCLE\n"
    pat, _ = patterns.parse_patterns_text(text)
    with pytest.raises(ValueError, match="is_laundering"):
        patterns.build_labels(_raw_tx(tmp_path, [NEG]), pat)


def test_synthetic_labels(synthetic) -> None:
    raw = ingest.read_raw_transactions(synthetic.transactions_csv)
    pat, n = patterns.parse_patterns_with_count(synthetic.patterns_txt)
    assert n == synthetic.expected["pattern_attempts"]
    assert pat.height == synthetic.expected["pattern_transactions"]
    assert set(pat["typology"].unique().to_list()) == set(TYPOLOGIES)

    labels = patterns.build_labels(raw, pat)
    assert labels.height == synthetic.expected["rows"]
    assert int(labels["is_laundering"].sum()) == synthetic.expected["positives"]
    in_pattern = labels.filter(pl.col("attempt_id").is_not_null())
    # Every pattern row is matched exactly once (distinct row_ids, one per pattern row).
    assert in_pattern.height == pat.height
    assert in_pattern["row_id"].is_unique().all()
    assert (in_pattern["is_laundering"] == 1).all()
    others = labels.filter(pl.col("typology") == OTHER)
    assert others.height == synthetic.expected["positives"] - pat.height
    assert others["attempt_id"].null_count() == others.height
    assert labels.filter(pl.col("is_laundering") == 0)["typology"].null_count() == (
        labels.height - synthetic.expected["positives"]
    )
    sizes = pat.group_by("attempt_id").len()
    joined = in_pattern.join(sizes, on="attempt_id")
    assert (joined["attempt_size"].cast(pl.UInt32) == joined["len"]).all()


def test_parse_ignores_leading_bom() -> None:
    df, n = patterns.parse_patterns_text(chr(0xFEFF) + PATTERNS_TEXT)
    assert n == 3 and df.height == 5
