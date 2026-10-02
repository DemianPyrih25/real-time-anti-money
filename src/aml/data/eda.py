"""EDA: re-derive the PLAN.md §4 data facts by code (DuckDB SQL over the prepared Parquet).

Writes `reports/eda.md` and `reports/eda.json`. Everything is reported by simulated day index, never
by calendar date. `row_id` is the CSV's file order, so the "is the CSV sorted" check runs on the
Parquet; only the header check reads the raw CSV (its first line).
"""

from __future__ import annotations

from pathlib import Path

import duckdb

from aml.data.ingest import EXPECTED_HEADER, read_header
from aml.data.split import SPLITS, VIEWS, split_days, view_days
from aml.io import write_json_atomic, write_text_atomic
from aml.paths import DataPaths

# A day is "tail" when it is sparse (< 1% of the busiest day's rows) or mostly laundering.
TAIL_SPARSE_FRACTION = 0.01
TAIL_POSITIVE_SHARE = 0.5
DEGREE_QUANTILES = (0.5, 0.9, 0.99, 0.999, 0.9999)


def _sql_path(p: Path) -> str:
    return Path(p).resolve().as_posix().replace("'", "''")


def _rows(con: duckdb.DuckDBPyConnection, sql: str) -> list[dict]:
    cur = con.execute(sql)
    names = [d[0] for d in cur.description]
    return [dict(zip(names, r, strict=True)) for r in cur.fetchall()]


def _one(con: duckdb.DuckDBPyConnection, sql: str) -> dict:
    return _rows(con, sql)[0]


def _connect(paths: DataPaths, threads: int | None) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    if threads:
        con.execute(f"SET threads = {int(threads)}")
    con.execute(f"CREATE VIEW acc AS SELECT * FROM read_parquet('{_sql_path(paths.accounts)}')")
    # One narrow working table: transactions joined to the label columns the EDA needs.
    con.execute(
        f"""
        CREATE TEMP TABLE t AS
        SELECT tx.row_id, tx.rank, tx.minute, tx.day, tx.split, tx.src, tx.dst,
               tx.payment_format, tx.payment_currency, tx.receiving_currency,
               lab.is_laundering::INTEGER AS y, lab.attempt_id, lab.typology
        FROM read_parquet('{_sql_path(paths.transactions)}') tx
        JOIN read_parquet('{_sql_path(paths.labels)}') lab USING (row_id)
        """
    )
    return con


def _in_days(lo: int, hi: int) -> str:
    return f"day BETWEEN {int(lo)} AND {int(hi)}"


def _overview(con) -> dict:
    o = _one(
        con,
        """
        SELECT count(*) AS rows, sum(y) AS positives,
               100.0 * sum(y) / count(*) AS positive_pct,
               min(day) AS first_day, max(day) AS last_day
        FROM t
        """,
    )
    o["accounts"] = _one(con, "SELECT count(*) AS n FROM acc")["n"]
    o["n_days"] = o["last_day"] - o["first_day"] + 1
    return o


def _per_day(con) -> list[dict]:
    return _rows(
        con,
        f"""
        WITH d AS (SELECT day, count(*) AS rows, sum(y) AS positives FROM t GROUP BY day)
        SELECT day, rows, positives, positives::DOUBLE / rows AS positive_share,
               rows < {TAIL_SPARSE_FRACTION} * max(rows) OVER () AS sparse,
               positives::DOUBLE / rows >= {TAIL_POSITIVE_SHARE} AS mostly_laundering
        FROM d ORDER BY day
        """,
    )


def _tail(per_day: list[dict], data_cfg: dict) -> dict:
    days = [d["day"] for d in per_day if d["sparse"] or d["mostly_laundering"]]
    rows = sum(d["rows"] for d in per_day if d["day"] in days)
    pos = sum(d["positives"] for d in per_day if d["day"] in days)
    t_lo, t_hi = split_days(data_cfg, "test")
    test = [d for d in per_day if t_lo <= d["day"] <= t_hi]
    test_rows = sum(d["rows"] for d in test)
    test_pos = sum(d["positives"] for d in test)
    v_lo, v_hi = view_days(data_cfg, "tail")
    return {
        "days": days,
        "rows": rows,
        "positives": pos,
        "positive_share": pos / rows if rows else None,
        "share_of_test_rows": rows / test_rows if test_rows else None,
        "share_of_test_positives": pos / test_pos if test_pos else None,
        "matches_config_tail_view": days == list(range(v_lo, v_hi + 1)),
        "rule": f"rows < {TAIL_SPARSE_FRACTION:g} x busiest day, or positive share >= "
        f"{TAIL_POSITIVE_SHARE:g}",
    }


def _csv_traps(con, raw_csv: Path | None) -> dict:
    order = _one(
        con,
        """
        WITH o AS (SELECT row_id, minute, lag(minute) OVER (ORDER BY row_id) AS prev FROM t)
        SELECT count(*) FILTER (WHERE minute < prev) AS inversions,
               min(row_id) FILTER (WHERE minute < prev) AS first_out_of_order_row_id
        FROM o
        """,
    )
    banks = _one(
        con,
        r"""
        WITH b AS (SELECT DISTINCT bank FROM acc)
        SELECT count(*) AS distinct_banks,
               count(*) FILTER (WHERE bank LIKE '0%' AND length(bank) > 1) AS zero_padded_banks,
               (SELECT count(*) FROM (
                    SELECT TRY_CAST(bank AS BIGINT) AS v FROM b
                    GROUP BY v HAVING count(*) > 1 AND v IS NOT NULL)
               ) AS bank_codes_colliding_as_int
        FROM b
        """,
    )
    ids = _one(
        con,
        r"""
        SELECT count(*) FILTER (
                   WHERE regexp_full_match(account, '[0-9]+(\.[0-9]*)?[eE][+-]?[0-9]+')
               ) AS float_looking_account_ids,
               count(*) FILTER (WHERE regexp_full_match(account, '[0-9]+'))
                   AS all_digit_account_ids,
               count(DISTINCT account) AS distinct_account_strings
        FROM acc
        """,
    )
    shared = _one(
        con,
        """
        SELECT count(*) AS account_strings_at_several_banks FROM (
            SELECT account FROM acc GROUP BY account HAVING count(*) > 1)
        """,
    )
    header: dict = {"checked": False}
    if raw_csv is not None and Path(raw_csv).exists():
        names = read_header(Path(raw_csv))
        dups = sorted({n for n in names if names.count(n) > 1})
        header = {
            "checked": True,
            "columns": names,
            "duplicate_names": dups,
            "matches_expected": names == EXPECTED_HEADER,
        }
    return {
        "sorted_by_time": order["inversions"] == 0,
        **order,
        "header": header,
        **banks,
        **ids,
        **shared,
    }


def _rows_per_minute(con, tail_days: list[int]) -> dict:
    tail = ",".join(str(int(d)) for d in tail_days) or "-1"
    return _one(
        con,
        f"""
        WITH m AS (SELECT minute, count(*) AS n FROM t GROUP BY minute)
        SELECT count(*) AS occupied_minutes, avg(n) AS mean, quantile_disc(n, 0.5) AS p50,
               quantile_disc(n, 0.99) AS p99, max(n) AS max,
               (SELECT count(*) FROM t WHERE day NOT IN ({tail}))::DOUBLE
                 / (1440 * (SELECT count(DISTINCT day) FROM t WHERE day NOT IN ({tail})))
                 AS mean_over_non_tail_span
        FROM m
        """,
    )


def _payment(con) -> dict:
    formats = _rows(
        con,
        """
        SELECT payment_format, count(*) AS rows, sum(y) AS positives,
               sum(y)::DOUBLE / NULLIF((SELECT sum(y) FROM t), 0) AS share_of_positives,
               sum(y)::DOUBLE / count(*) AS prevalence
        FROM t GROUP BY payment_format ORDER BY positives DESC, payment_format
        """,
    )
    other = _one(
        con,
        """
        SELECT
          avg(CASE WHEN src = dst THEN 1.0 ELSE 0.0 END)
            FILTER (WHERE payment_format = 'Reinvestment') AS reinvestment_self_loop_share,
          sum(y) FILTER (WHERE payment_format = 'Wire') AS wire_positives,
          count(*) FILTER (WHERE payment_currency <> receiving_currency) AS cross_currency_rows,
          coalesce(sum(y) FILTER (WHERE payment_currency <> receiving_currency), 0)
            AS cross_currency_positives,
          count(*) FILTER (WHERE src = dst) AS self_loop_rows,
          coalesce(sum(y) FILTER (WHERE src = dst), 0) AS self_loop_positives
        FROM t
        """,
    )
    return {"formats": formats, **other}


def _patterns(con) -> dict:
    tot = _one(
        con,
        """
        SELECT count(DISTINCT attempt_id) AS attempts,
               count(*) FILTER (WHERE attempt_id IS NOT NULL) AS pattern_transactions,
               count(*) FILTER (WHERE typology = 'OTHER') AS other_positives,
               count(*) FILTER (WHERE attempt_id IS NOT NULL)::DOUBLE
                 / NULLIF(sum(y), 0) AS share_of_positives_covered
        FROM t
        """,
    )
    by_typ = _rows(
        con,
        """
        SELECT typology, count(DISTINCT attempt_id) AS attempts, count(*) AS transactions
        FROM t WHERE typology IS NOT NULL
        GROUP BY typology ORDER BY typology
        """,
    )
    return {**tot, "typologies": by_typ}


def _hubs(con, top: int = 5) -> dict:
    out: dict = {}
    for name, col, other in (("out", "src", "dst"), ("in", "dst", "src")):
        out[f"top_{name}_degree"] = _rows(
            con,
            f"""
            SELECT {col} AS account_id, count(*) AS degree,
                   count(DISTINCT {other}) AS distinct_counterparties
            FROM t GROUP BY {col} ORDER BY degree DESC, account_id LIMIT {int(top)}
            """,
        )
        out[f"max_{name}_degree"] = out[f"top_{name}_degree"][0]["degree"]
    return out


def _repeat_accounts(con, data_cfg: dict) -> dict:
    tr_lo, tr_hi = split_days(data_cfg, "train")
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE train_launderers AS
        SELECT src AS a FROM t WHERE y = 1 AND {_in_days(tr_lo, tr_hi)}
        UNION SELECT dst FROM t WHERE y = 1 AND {_in_days(tr_lo, tr_hi)}
        """
    )
    out = {"train_launderer_accounts": _one(con, "SELECT count(*) AS n FROM train_launderers")["n"]}
    scopes = {"test": split_days(data_cfg, "test")}
    scopes.update({f"view_{v}": view_days(data_cfg, v) for v in VIEWS})
    for name, (lo, hi) in scopes.items():
        out[name] = _one(
            con,
            f"""
            SELECT count(*) AS positives,
                   count(*) FILTER (WHERE src IN (SELECT a FROM train_launderers)
                                       OR dst IN (SELECT a FROM train_launderers)) AS touching,
                   touching::DOUBLE / NULLIF(count(*), 0) AS share_touching
            FROM t WHERE y = 1 AND {_in_days(lo, hi)}
            """,
        )
    return out


def _splits(con, data_cfg: dict) -> dict:
    total = _one(con, "SELECT count(*) AS n FROM t")["n"]
    out: dict = {}
    scopes = [(s, split_days(data_cfg, s)) for s in SPLITS]
    scopes += [(f"test_{v}", view_days(data_cfg, v)) for v in VIEWS]
    for name, (lo, hi) in scopes:
        r = _one(
            con,
            f"SELECT count(*) AS rows, coalesce(sum(y), 0) AS positives FROM t "
            f"WHERE {_in_days(lo, hi)}",
        )
        r["days"] = [lo, hi]
        r["share_of_rows"] = r["rows"] / total if total else None
        r["prevalence"] = r["positives"] / r["rows"] if r["rows"] else None
        out[name] = r
    return out


def _train_degrees(con, data_cfg: dict) -> dict:
    lo, hi = split_days(data_cfg, "train")
    qs = ", ".join(str(q) for q in DEGREE_QUANTILES)
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE train_deg AS
        WITH e AS (SELECT src, dst FROM t WHERE {_in_days(lo, hi)}),
             o AS (SELECT src AS a, count(*) AS n FROM e GROUP BY src),
             i AS (SELECT dst AS a, count(*) AS n FROM e GROUP BY dst)
        SELECT coalesce(o.a, i.a) AS a, coalesce(o.n, 0) AS out_deg, coalesce(i.n, 0) AS in_deg,
               coalesce(o.n, 0) + coalesce(i.n, 0) AS total_deg
        FROM o FULL OUTER JOIN i ON o.a = i.a
        """
    )
    out: dict = {
        "definition": "edge (transaction) counts over accounts active in train; a self-loop "
        "counts once as out and once as in",
        "accounts": _one(con, "SELECT count(*) AS n FROM train_deg")["n"],
        "quantiles": list(DEGREE_QUANTILES),
    }
    for col in ("in_deg", "out_deg", "total_deg"):
        r = _one(con, f"SELECT quantile_disc({col}, [{qs}]) AS q, max({col}) AS mx FROM train_deg")
        out[col] = {**{f"q{q:g}": v for q, v in zip(DEGREE_QUANTILES, r["q"], strict=True)}}
        out[col]["max"] = r["mx"]
    return out


def _ties(con) -> dict:
    return _one(
        con,
        """
        WITH m AS (SELECT minute, count(*) AS n FROM t GROUP BY minute),
             sm AS (SELECT src, minute, count(*) AS n FROM t GROUP BY src, minute)
        SELECT (SELECT coalesce(sum(n) FILTER (WHERE n > 1), 0) FROM m)::DOUBLE
                 / (SELECT count(*) FROM t)
                 AS share_rows_in_shared_minute,
               (SELECT coalesce(sum(n) FILTER (WHERE n > 1), 0) FROM sm)::DOUBLE
                 / (SELECT count(*) FROM t) AS share_rows_in_shared_src_minute
        """,
    )


def compute_eda(
    paths: DataPaths, data_cfg: dict, *, threads: int | None = None, raw_csv: Path | None = None
) -> dict:
    """All EDA facts as a JSON-able dict (no files written)."""
    con = _connect(paths, threads)
    try:
        per_day = _per_day(con)
        tail = _tail(per_day, data_cfg)
        return {
            "overview": _overview(con),
            "per_day": per_day,
            "tail_days": tail,
            "csv": _csv_traps(con, raw_csv),
            "rows_per_minute": _rows_per_minute(con, tail["days"]),
            "payment": _payment(con),
            "patterns": _patterns(con),
            "hubs": _hubs(con),
            "repeat_accounts": _repeat_accounts(con, data_cfg),
            "splits": _splits(con, data_cfg),
            "train_degree": _train_degrees(con, data_cfg),
            "ties": _ties(con),
        }
    finally:
        con.close()


def _fmt(v) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _table(rows: list[dict], cols: list[str]) -> str:
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "".join("| " + " | ".join(_fmt(r.get(c)) for c in cols) + " |\n" for r in rows)


def _kv(d: dict) -> str:
    return _table([{"fact": k, "value": v} for k, v in d.items()], ["fact", "value"])


def render_markdown(eda: dict) -> str:
    o, csv, pay, pat = eda["overview"], eda["csv"], eda["payment"], eda["patterns"]
    header, tail, deg = csv["header"], eda["tail_days"], eda["train_degree"]
    hub_cols = ["account_id", "degree", "distinct_counterparties"]
    sections = [
        (
            "Overview",
            [
                _kv(
                    {
                        k: o[k]
                        for k in (
                            "rows",
                            "accounts",
                            "positives",
                            "positive_pct",
                            "first_day",
                            "last_day",
                            "n_days",
                        )
                    }
                ),
            ],
        ),
        (
            "Rows and positives per day",
            [
                _table(
                    eda["per_day"],
                    ["day", "rows", "positives", "positive_share", "sparse", "mostly_laundering"],
                ),
            ],
        ),
        (
            "Tail days",
            [
                _kv(
                    {"days": ", ".join(str(d) for d in tail["days"]) or "none"}
                    | {k: v for k, v in tail.items() if k != "days"}
                ),
            ],
        ),
        (
            "CSV traps",
            [
                _kv(
                    {
                        "sorted by time": csv["sorted_by_time"],
                        "rows earlier than the previous row (file order)": csv["inversions"],
                        "first out-of-order row_id": csv["first_out_of_order_row_id"],
                        "header checked": header["checked"],
                        "duplicate header names": ", ".join(header.get("duplicate_names", []))
                        or "none",
                        "distinct bank codes": csv["distinct_banks"],
                        "zero-padded bank codes": csv["zero_padded_banks"],
                        "bank codes that collide if parsed as int": csv[
                            "bank_codes_colliding_as_int"
                        ],
                        "float-looking account ids": csv["float_looking_account_ids"],
                        "all-digit account ids": csv["all_digit_account_ids"],
                        "account strings at several banks": csv["account_strings_at_several_banks"],
                    }
                ),
            ],
        ),
        ("Rows per minute", [_kv(eda["rows_per_minute"])]),
        (
            "Payment format and positives",
            [
                _table(
                    pay["formats"],
                    ["payment_format", "rows", "positives", "share_of_positives", "prevalence"],
                ),
                _kv({k: v for k, v in pay.items() if k != "formats"}),
            ],
        ),
        (
            "Patterns file",
            [
                _kv({k: v for k, v in pat.items() if k != "typologies"}),
                "Per typology (`OTHER` = positives absent from the patterns file):\n",
                _table(pat["typologies"], ["typology", "attempts", "transactions"]),
            ],
        ),
        (
            "Hubs (whole period, transaction counts)",
            [
                "Top out-degree:\n",
                _table(eda["hubs"]["top_out_degree"], hub_cols),
                "Top in-degree:\n",
                _table(eda["hubs"]["top_in_degree"], hub_cols),
            ],
        ),
        (
            "Repeat accounts (positives touching an account that laundered in train)",
            [
                f"Accounts that laundered in train: "
                f"{_fmt(eda['repeat_accounts']['train_launderer_accounts'])}\n",
                _table(
                    [
                        {"scope": k, **v}
                        for k, v in eda["repeat_accounts"].items()
                        if isinstance(v, dict)
                    ],
                    ["scope", "positives", "touching", "share_touching"],
                ),
            ],
        ),
        (
            "Split sizes and prevalence",
            [
                _table(
                    [
                        {"split": k, **v, "days": f"{v['days'][0]}-{v['days'][1]}"}
                        for k, v in eda["splits"].items()
                    ],
                    ["split", "days", "rows", "positives", "share_of_rows", "prevalence"],
                ),
            ],
        ),
        (
            "Train-period degree distribution",
            [
                f"{deg['definition']}; accounts: {_fmt(deg['accounts'])}.\n",
                _table(
                    [{"degree": c, **deg[c]} for c in ("in_deg", "out_deg", "total_deg")],
                    ["degree", *[f"q{q:g}" for q in DEGREE_QUANTILES], "max"],
                ),
            ],
        ),
        ("Same-minute ties", [_kv(eda["ties"])]),
    ]
    out = [
        "# EDA: IBM AML HI-Small (re-derived by code)\n",
        "Simulated day indices (day 1 = the first timestamp's day); no calendar dates. "
        "The data is synthetic: labels are perfect and complete.\n",
    ]
    for title, blocks in sections:
        out.append(f"## {title}\n")
        out.extend(blocks)
    return "\n".join(out)


def run_eda(
    paths: DataPaths, data_cfg: dict, *, threads: int | None = None, raw_csv: Path | None = None
) -> dict:
    """Compute the EDA, write `reports/eda.md` and `reports/eda.json`, return the dict."""
    if raw_csv is None:
        default = paths.raw_dir / data_cfg["dataset"]["transactions_file"]
        raw_csv = default if default.exists() else None
    eda = compute_eda(paths, data_cfg, threads=threads, raw_csv=raw_csv)
    write_json_atomic(eda, paths.reports / "eda.json")
    write_text_atomic(render_markdown(eda), paths.reports / "eda.md")
    return eda
