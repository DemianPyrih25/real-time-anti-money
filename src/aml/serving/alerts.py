"""Alert sinks of the M5 scorer: SQLite (`runtime/alerts.sqlite`) and the Kafka `alerts` topic.

Core module: `confluent_kafka` is imported only inside `KafkaAlertSink`.

An alert record is a dict with the keys `ALERT_COLUMNS` (`scorer.alert_record` builds it);
`rules_fired` and `severities` are lists, stored as compact JSON text. Writes are idempotent
(`INSERT OR IGNORE` on row_id) and committed one by one with `synchronous=FULL`, so an alert is
durable before the scorer moves on and a restart that re-scores it changes nothing. The writer
connection belongs to the consumer thread; HTTP readers open their own read-only connection per
request (`read_alerts`, `read_alert`, `count_alerts`).

M6: the same file holds the `cases` table, one case pack (`aml.explain.casepack`) per alert,
keyed by the alert's row_id: `write_case` (consumer thread, after the alert write; idempotent
and committed the same way) and the readers `read_case`, `read_case_json`, `list_cases` and
`count_cases`. Besides the pack JSON, a row keeps the few fields the list page groups and labels
by (case_key, day, subject, typology); `list_cases` joins the alerts table for the rest.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from aml.features.spec import MINUTES_PER_DAY
from aml.streaming.codec import encode_alert, event_key

log = logging.getLogger(__name__)

ALERT_COLUMNS = (
    "row_id",
    "rank",
    "offset",
    "minute",
    "day",
    "src",
    "dst",
    "amount_usd",
    "payment_format",
    "score",
    "threshold",
    "rate_tag",
    "rules_fired",
    "severities",
    "model_version",
)
_JSON_COLUMNS = ("rules_fired", "severities")
_COLS = ", ".join(f'"{c}"' for c in ALERT_COLUMNS)  # "offset" is an SQL keyword
DDL = """CREATE TABLE IF NOT EXISTS alerts(
  row_id INTEGER PRIMARY KEY, rank INTEGER NOT NULL, "offset" INTEGER NOT NULL,
  minute INTEGER NOT NULL, day INTEGER NOT NULL, src INTEGER NOT NULL, dst INTEGER NOT NULL,
  amount_usd REAL NOT NULL, payment_format TEXT NOT NULL, score REAL NOT NULL,
  threshold REAL NOT NULL, rate_tag TEXT NOT NULL, rules_fired TEXT NOT NULL,
  severities TEXT NOT NULL, model_version TEXT NOT NULL)"""
_INSERT = f"INSERT OR IGNORE INTO alerts({_COLS}) VALUES ({', '.join('?' * len(ALERT_COLUMNS))})"
_SELECT = f"SELECT {_COLS} FROM alerts"

CASE_COLUMNS = ("row_id", "case_key", "day", "subject", "typology", "pack")
CASES_DDL = """CREATE TABLE IF NOT EXISTS cases(
  row_id INTEGER PRIMARY KEY, case_key TEXT NOT NULL, day INTEGER NOT NULL,
  subject INTEGER NOT NULL, typology TEXT NOT NULL, pack TEXT NOT NULL)"""
_INSERT_CASE = f"INSERT OR IGNORE INTO cases({', '.join(CASE_COLUMNS)}) VALUES (?, ?, ?, ?, ?, ?)"
_LIST_CASES = (
    "SELECT c.row_id, c.case_key, c.day, c.subject, c.typology, a.rank, a.minute, a.dst, "
    "a.amount_usd, a.payment_format, a.score, a.rules_fired "
    "FROM cases c JOIN alerts a ON a.row_id = c.row_id"
)


def _params(rec: Mapping[str, Any]) -> tuple:
    return tuple(
        json.dumps(list(rec[c]), separators=(",", ":"), allow_nan=False)
        if c in _JSON_COLUMNS
        else rec[c]
        for c in ALERT_COLUMNS
    )


def _row(values: tuple) -> dict[str, Any]:
    d = dict(zip(ALERT_COLUMNS, values, strict=True))
    for c in _JSON_COLUMNS:
        d[c] = json.loads(d[c])
    return d


def _case_params(pack: Mapping[str, Any]) -> tuple:
    """The cases row of a pack (ValueError on NaN or infinity: packs are strict JSON)."""
    return (
        int(pack["id"]),
        str(pack["case_key"]),
        int(pack["when"]["day"]),
        int(pack["who"]["subject"]["account"]),
        str(pack["why"]["typology"]["label"]),
        json.dumps(pack, separators=(",", ":"), allow_nan=False),
    )


class SqliteAlertStore:
    """The alerts and cases tables; open, write and close them on one thread (the consumer's)."""

    failed = 0  # the AlertSink interface: SQLite errors raise instead

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        mode = self.conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]
        if str(mode).lower() != "wal":
            self.conn.close()
            raise RuntimeError(f"{self.path}: SQLite refused WAL mode (got {mode!r})")
        self.conn.execute("PRAGMA synchronous=FULL")
        self.conn.execute(DDL)
        self.conn.execute(CASES_DDL)
        self.conn.commit()

    def write(self, rec: Mapping[str, Any]) -> bool:
        """Insert and commit one alert; False if its row_id was already stored (a replay)."""
        cur = self.conn.execute(_INSERT, _params(rec))
        self.conn.commit()
        return cur.rowcount == 1

    def write_case(self, pack: Mapping[str, Any]) -> bool:
        """Insert and commit one case pack; False if its row_id was already stored (a replay
        re-builds the same pack; the first write wins)."""
        cur = self.conn.execute(_INSERT_CASE, _case_params(pack))
        self.conn.commit()
        return cur.rowcount == 1

    def count_cases(self) -> int:
        return int(self.conn.execute("SELECT count(*) FROM cases").fetchone()[0])

    def flush(self, timeout_s: float = 0.0) -> int:
        """Outstanding writes: none, every write is committed before `write` returns."""
        return 0

    def count(self) -> int:
        return int(self.conn.execute("SELECT count(*) FROM alerts").fetchone()[0])

    def rows(self) -> list[dict[str, Any]]:
        """Every stored alert in rank order (writer thread only)."""
        return [_row(v) for v in self.conn.execute(f"{_SELECT} ORDER BY rank")]

    def close(self) -> None:
        self.conn.close()


class KafkaAlertSink:
    """At-least-once copy of each alert on the `alerts` topic (key = row_id, partition 0)."""

    def __init__(
        self,
        bootstrap: str,
        topic: str,
        *,
        producer_factory: Callable[[dict[str, Any]], Any] | None = None,
    ) -> None:
        if producer_factory is None:
            from confluent_kafka import Producer  # the Kafka path only (core-import rule)

            from aml.streaming.kafka import producer_config

            config = producer_config(bootstrap, client_id="aml-scorer-alerts")
            producer_factory = Producer
        else:
            config = {"bootstrap.servers": bootstrap}
        self.topic = topic
        self.failed = 0  # delivery-callback errors
        self._producer = producer_factory(config)

    def _on_delivery(self, err: Any, msg: Any) -> None:
        if err is not None:
            self.failed += 1
            log.warning("alert delivery failed: %s", err)

    def write(self, rec: Mapping[str, Any]) -> bool:
        key, value = event_key(rec["row_id"]), encode_alert(rec)
        while True:
            try:
                self._producer.produce(
                    self.topic, value=value, key=key, partition=0, on_delivery=self._on_delivery
                )
                break
            except BufferError:  # local queue full: serve delivery reports, then retry
                self._producer.poll(0.05)
        self._producer.poll(0)
        return True

    def flush(self, timeout_s: float) -> int:
        """Wait up to timeout_s for deliveries; the number of messages still outstanding."""
        return int(self._producer.flush(timeout_s))

    def close(self) -> None:
        self._producer.flush(1.0)


def _connect_ro(path: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{Path(path).resolve().as_uri()}?mode=ro", uri=True)


def _read(path: Path, sql: str, params: tuple) -> list[tuple]:
    """Rows of a read-only query; [] when the database or its table does not exist yet."""
    p = Path(path)
    if not p.is_file():
        return []
    conn = _connect_ro(p)
    try:
        return conn.execute(sql, params).fetchall()
    except sqlite3.OperationalError as e:
        if "no such table" in str(e):
            return []
        raise
    finally:
        conn.close()


def read_alerts(
    path: Path, limit: int | None = 100, before_rank: int | None = None
) -> list[dict[str, Any]]:
    """Stored alerts, newest rank first; only ranks < before_rank when given."""
    where, params = ("WHERE rank < ? ", (before_rank,)) if before_rank is not None else ("", ())
    lim = ""
    if limit is not None:
        lim, params = " LIMIT ?", (*params, int(limit))
    return [_row(v) for v in _read(path, f"{_SELECT} {where}ORDER BY rank DESC{lim}", params)]


def read_alert(path: Path, row_id: int) -> dict[str, Any] | None:
    rows = _read(path, f"{_SELECT} WHERE row_id = ?", (int(row_id),))
    return _row(rows[0]) if rows else None


def count_alerts(path: Path) -> int:
    rows = _read(path, "SELECT count(*) FROM alerts", ())
    return int(rows[0][0]) if rows else 0


def read_case_json(path: Path, row_id: int) -> str | None:
    """The stored case pack of an alert as JSON text, or None."""
    rows = _read(path, "SELECT pack FROM cases WHERE row_id = ?", (int(row_id),))
    return str(rows[0][0]) if rows else None


def read_case(path: Path, row_id: int) -> dict[str, Any] | None:
    """The stored case pack of an alert, or None."""
    text = read_case_json(path, row_id)
    return None if text is None else json.loads(text)


def count_cases(path: Path) -> int:
    rows = _read(path, "SELECT count(*) FROM cases", ())
    return int(rows[0][0]) if rows else 0


def list_cases(
    path: Path, limit: int | None = 100, before_row: int | None = None
) -> list[dict[str, Any]]:
    """The newest `limit` cased alerts (rank descending; only alerts older than the alert
    `before_row` when given, none if that row is unknown), grouped by case_key in order of each
    group's newest alert.

    A group is {case_key, subject, day, alerts}; an alert is {row_id, rank, minute, time,
    counterparty, amount_usd, payment_format, score, typology, rules_fired}. A group can continue
    on the next page."""
    where, params = "", ()
    if before_row is not None:
        where = " WHERE a.rank < (SELECT rank FROM alerts WHERE row_id = ?)"
        params = (int(before_row),)
    lim = ""
    if limit is not None:
        lim, params = " LIMIT ?", (*params, int(limit))
    rows = _read(path, f"{_LIST_CASES}{where} ORDER BY a.rank DESC{lim}", params)
    groups: dict[str, dict[str, Any]] = {}
    for row_id, key, day, subject, typ, rank, minute, dst, usd, fmt, score, fired in rows:
        g = groups.setdefault(key, {"case_key": key, "subject": subject, "day": day, "alerts": []})
        hh, mm = divmod(minute % MINUTES_PER_DAY, 60)
        g["alerts"].append(
            {
                "row_id": row_id,
                "rank": rank,
                "minute": minute,
                "time": f"{hh:02d}:{mm:02d}",
                "counterparty": dst,
                "amount_usd": usd,
                "payment_format": fmt,
                "score": score,
                "typology": typ,
                "rules_fired": json.loads(fired),
            }
        )
    return list(groups.values())
