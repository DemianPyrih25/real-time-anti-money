"""Patterns.txt -> pattern rows, and the separate labels table keyed by row_id.

The file holds BEGIN/END blocks, one per laundering attempt, e.g.

    BEGIN LAUNDERING ATTEMPT - FAN-OUT:  Max 16-degree Fan-Out
    <11 CSV fields, same format as the transactions file>
    END LAUNDERING ATTEMPT - FAN-OUT
    <blank line>

Some headers carry no detail ("BEGIN LAUNDERING ATTEMPT - STACK"). Pattern rows are joined to
the transactions on all 11 raw strings, so exact duplicate rows pair up 1:1 by occurrence order.
"""

from __future__ import annotations

import re
from pathlib import Path

import polars as pl

from aml.data.ingest import RAW_COLUMNS

TYPOLOGIES = (
    "FAN-OUT",
    "FAN-IN",
    "CYCLE",
    "SCATTER-GATHER",
    "GATHER-SCATTER",
    "STACK",
    "BIPARTITE",
    "RANDOM",
)
OTHER = "OTHER"  # positives absent from Patterns.txt ("integration" laundering)
LABEL_TYPOLOGIES = (*TYPOLOGIES, OTHER)

_BEGIN = re.compile(r"^BEGIN LAUNDERING ATTEMPT - (?P<typ>[A-Z][A-Z-]*)\s*(?::(?P<detail>.*))?$")
_END = re.compile(r"^END LAUNDERING ATTEMPT - (?P<typ>[A-Z][A-Z-]*)\s*$")

_PATTERN_SCHEMA = {
    "attempt_id": pl.Int32,
    "typology": pl.String,
    "typology_detail": pl.String,
    **{c: pl.String for c in RAW_COLUMNS},
}


def parse_patterns_text(text: str) -> tuple[pl.DataFrame, int]:
    """Parse the file content; returns (pattern rows, number of BEGIN/END blocks)."""
    rows: list[list] = []
    n_blocks = 0
    current: tuple[int, str, str | None] | None = None  # (attempt_id, typology, detail)
    text = text.removeprefix(chr(0xFEFF))  # UTF-8 byte-order mark, if any
    for lineno, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("BEGIN"):
            m = _BEGIN.match(line)
            if m is None:
                raise ValueError(f"line {lineno}: malformed BEGIN header: {line!r}")
            if current is not None:
                raise ValueError(f"line {lineno}: BEGIN inside an open block")
            typ = m["typ"]
            if typ not in TYPOLOGIES:
                raise ValueError(f"line {lineno}: unknown typology {typ!r}")
            detail = (m["detail"] or "").strip() or None
            current = (n_blocks, typ, detail)
            n_blocks += 1
        elif line.startswith("END"):
            m = _END.match(line)
            if m is None:
                raise ValueError(f"line {lineno}: malformed END line: {line!r}")
            if current is None:
                raise ValueError(f"line {lineno}: END without BEGIN")
            if m["typ"] != current[1]:
                raise ValueError(
                    f"line {lineno}: END typology {m['typ']!r} != BEGIN typology {current[1]!r}"
                )
            current = None
        else:
            if current is None:
                raise ValueError(f"line {lineno}: data row outside a BEGIN/END block")
            fields = raw_line.rstrip("\r\n").split(",")
            if len(fields) != len(RAW_COLUMNS):
                raise ValueError(
                    f"line {lineno}: expected {len(RAW_COLUMNS)} fields, got {len(fields)}"
                )
            rows.append([*current, *fields])
    if current is not None:
        raise ValueError(f"unterminated block for attempt {current[0]} ({current[1]})")
    df = pl.DataFrame(rows, schema=_PATTERN_SCHEMA, orient="row")
    return df, n_blocks


def parse_patterns_with_count(path: Path) -> tuple[pl.DataFrame, int]:
    """(pattern rows, number of attempts). The count includes blocks with no rows."""
    # newline="" keeps CR characters so splitlines() handles CRLF and LF alike.
    with Path(path).open(encoding="utf-8", newline="") as f:
        return parse_patterns_text(f.read())


def parse_patterns(path: Path) -> pl.DataFrame:
    """attempt_id Int32, typology, typology_detail (nullable) + the 11 raw String columns."""
    return parse_patterns_with_count(path)[0]


def _dup_index(df: pl.DataFrame) -> pl.DataFrame:
    """0-based occurrence index within groups of identical 11-column rows (current row order)."""
    return df.with_columns(pl.int_range(pl.len()).over(RAW_COLUMNS).alias("_dup"))


def build_labels(raw_tx: pl.DataFrame, patterns: pl.DataFrame) -> pl.DataFrame:
    """One row per transaction: row_id, is_laundering, attempt_id, typology, attempt_size,
    typology_detail. `raw_tx` is the output of `read_raw_transactions` (file order)."""
    bad = patterns.filter(pl.col("is_laundering") != "1")
    if bad.height:
        raise ValueError(f"{bad.height} pattern rows have is_laundering != '1'")

    # Every pattern row carries is_laundering == "1" (checked above) and it is a join key, so
    # only positive transactions can match: join against those alone (5k rows, not 5M).
    positives = _dup_index(
        raw_tx.filter(pl.col("is_laundering") == "1").select("row_id", *RAW_COLUMNS)
    )
    pat = _dup_index(patterns).with_columns(
        pl.len().over("attempt_id").cast(pl.Int32).alias("attempt_size")
    )
    matched = pat.join(positives, on=[*RAW_COLUMNS, "_dup"], how="left", maintain_order="left")
    unmatched = matched.filter(pl.col("row_id").is_null())
    if unmatched.height:
        example = unmatched.select("attempt_id", *RAW_COLUMNS).row(0)
        raise ValueError(f"{unmatched.height} pattern rows match no transaction (first: {example})")
    if matched["row_id"].n_unique() != matched.height:
        raise ValueError("a transaction matched more than one pattern row")

    info = matched.select("row_id", "attempt_id", "typology", "attempt_size", "typology_detail")
    labels = (
        raw_tx.select("row_id", pl.col("is_laundering").cast(pl.Int8, strict=True))
        .join(info, on="row_id", how="left", maintain_order="left")
        .with_columns(
            pl.when(pl.col("is_laundering") == 1)
            .then(pl.col("typology").fill_null(OTHER))
            .otherwise(None)
            .alias("typology")
        )
    )
    return labels.select(
        "row_id", "is_laundering", "attempt_id", "typology", "attempt_size", "typology_detail"
    ).sort("row_id")
