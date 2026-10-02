"""Tiny synthetic dataset in the IBM AML HI-Small file format, with planted laundering patterns.

It reproduces the traps of the real files (PLAN.md §4) at a size that runs in seconds:
- rows are written unsorted; the header repeats `Account`;
- bank codes are zero-padded strings, and banks "012" and "12" both exist;
- some account ids look like scientific-notation floats (e.g. "8012E4567");
- one account string exists at two banks (accounts are keyed by (bank, account));
- one exact duplicate normal row;
- cross-currency rows with consistent FX rates; Reinvestment is always a self-loop;
- laundering is ACH (plus one Bitcoin row), never Wire or cross-currency;
- a Patterns file with BEGIN/END blocks in all 8 typologies, rows unsorted inside a block,
  plus "integration" positives that are labelled but absent from the Patterns file;
- 10 normal days followed by 8 sparse "tail" days where most rows are laundering.

Everything is deterministic given `seed`.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

HEADER = [
    "Timestamp",
    "From Bank",
    "Account",
    "To Bank",
    "Account",
    "Amount Received",
    "Receiving Currency",
    "Amount Paid",
    "Payment Currency",
    "Payment Format",
    "Is Laundering",
]

# Arbitrary origin of simulated time (only minute offsets from it matter).
ORIGIN = datetime(2000, 1, 1)

# Units of currency per 1 US Dollar.
FX = {
    "US Dollar": 1.0,
    "Euro": 0.85,
    "Yuan": 7.0,
    "Rupee": 83.0,
    "Yen": 150.0,
    "Swiss Franc": 0.9,
}
NORMAL_FORMATS = ["Cheque", "Credit Card", "ACH", "Wire", "Cash", "Bitcoin"]
NORMAL_FORMAT_P = [0.25, 0.25, 0.2, 0.1, 0.12, 0.08]
BANKS = ["010", "03208", "001", "0119", "020", "0048309", "012", "12"]

TYPOLOGIES = [
    "FAN-OUT",
    "FAN-IN",
    "CYCLE",
    "SCATTER-GATHER",
    "GATHER-SCATTER",
    "STACK",
    "BIPARTITE",
    "RANDOM",
]


@dataclass(frozen=True)
class Tx:
    minute: int
    from_bank: str
    from_account: str
    to_bank: str
    to_account: str
    amount_received: float
    receiving_currency: str
    amount_paid: float
    payment_currency: str
    payment_format: str
    is_laundering: int

    def row(self) -> list[str]:
        ts = (ORIGIN + timedelta(minutes=self.minute)).strftime("%Y/%m/%d %H:%M")
        return [
            ts,
            self.from_bank,
            self.from_account,
            self.to_bank,
            self.to_account,
            f"{self.amount_received:.2f}",
            self.receiving_currency,
            f"{self.amount_paid:.2f}",
            self.payment_currency,
            self.payment_format,
            str(self.is_laundering),
        ]


@dataclass
class SyntheticDataset:
    transactions_csv: Path
    patterns_txt: Path
    expected: dict[str, int]
    fx_units_per_usd: dict[str, float] = field(default_factory=lambda: dict(FX))
    n_days: int = 18


def _make_accounts(rng: np.random.Generator, n: int) -> list[tuple[str, str, str]]:
    """Return (bank, account, home_currency) triples with unique (bank, account) keys."""
    ids: set[str] = set()
    out: list[tuple[str, str, str]] = []
    currencies = list(FX)
    cur_p = np.array([0.6, 0.15, 0.1, 0.05, 0.05, 0.05])
    i = 0
    while len(out) < n:
        if i % 10 == 0:
            acct = f"80{i % 100:02d}E{int(rng.integers(1000, 9999))}"  # parses as a float
        else:
            acct = "80" + format(int(rng.integers(0, 16**7)), "07X")
        i += 1
        if acct in ids:
            continue
        ids.add(acct)
        bank = BANKS[int(rng.integers(0, len(BANKS)))]
        cur = currencies[int(rng.choice(len(currencies), p=cur_p))]
        out.append((bank, acct, cur))
    # The same account string at a second bank: a distinct account.
    b0, a0, c0 = out[1]
    other_bank = next(b for b in BANKS if b != b0)
    out.append((other_bank, a0, c0))
    return out


def _amount(rng: np.random.Generator, round_p: float = 0.08) -> float:
    if rng.random() < round_p:
        return float(100 * int(rng.integers(1, 200)))
    return float(np.round(np.exp(rng.normal(7.0, 1.6)), 2)) + 0.01


def _tx(
    minute: int,
    src: tuple[str, str, str],
    dst: tuple[str, str, str],
    amount_paid: float,
    fmt: str,
    label: int,
    cross: bool = False,
) -> Tx:
    pay_cur = src[2]
    recv_cur = dst[2] if cross and dst[2] != src[2] else pay_cur
    amount_received = float(np.round(amount_paid / FX[pay_cur] * FX[recv_cur], 2))
    return Tx(
        minute=minute,
        from_bank=src[0],
        from_account=src[1],
        to_bank=dst[0],
        to_account=dst[1],
        amount_received=amount_received,
        receiving_currency=recv_cur,
        amount_paid=amount_paid,
        payment_currency=pay_cur,
        payment_format=fmt,
        is_laundering=label,
    )


def _attempt_edges(
    rng: np.random.Generator, typology: str, pool: list[tuple[str, str, str]]
) -> tuple[list[tuple[int, int]], str]:
    """Edges (src index, dst index) into `pool` for one attempt, plus the header detail."""
    idx = [int(i) for i in rng.permutation(len(pool))]
    if typology == "FAN-OUT":
        k = int(rng.integers(3, 7))
        return [(idx[0], idx[j]) for j in range(1, k + 1)], f"Max {k}-degree Fan-Out"
    if typology == "FAN-IN":
        k = int(rng.integers(3, 7))
        return [(idx[j], idx[0]) for j in range(1, k + 1)], f"Max {k}-degree Fan-In"
    if typology == "CYCLE":
        n = int(rng.integers(2, 5))
        nodes = idx[:n]
        return [(nodes[j], nodes[(j + 1) % n]) for j in range(n)], f"Max {n} hops"
    if typology == "SCATTER-GATHER":
        k = int(rng.integers(2, 5))
        mids = idx[2 : 2 + k]
        return [(idx[0], m) for m in mids] + [(m, idx[1]) for m in mids], ""
    if typology == "GATHER-SCATTER":
        k = int(rng.integers(2, 4))
        ins, outs = idx[1 : 1 + k], idx[1 + k : 1 + 2 * k]
        edges = [(i, idx[0]) for i in ins] + [(idx[0], o) for o in outs]
        return edges, f"Max {k}-degree Fan-In"
    if typology == "STACK":
        a, b, c = idx[0:2], idx[2:4], idx[4:6]
        return [(x, y) for x in a for y in b] + [(y, z) for y in b for z in c], ""
    if typology == "BIPARTITE":
        a, b = idx[0:2], idx[2:5]
        return [(x, y) for x in a for y in b], ""
    if typology == "RANDOM":
        n = int(rng.integers(2, 5))
        walk = idx[: n + 1]
        return [(walk[j], walk[j + 1]) for j in range(n)], f"Max {n} hops"
    raise ValueError(typology)


def make_synthetic(
    out_dir: Path,
    *,
    seed: int = 0,
    n_accounts: int = 400,
    normal_days: int = 10,
    tail_days: int = 8,
    tx_per_day: int = 700,
    tail_normal_per_day: int = 4,
    attempts_per_typology: int = 5,
    integration_positives: int = 40,
) -> SyntheticDataset:
    """Write `HI-Small_Trans.csv` and `HI-Small_Patterns.txt` into `out_dir`."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    accounts = _make_accounts(rng, n_accounts)
    n_all = len(accounts)
    launder_pool = accounts[: n_all // 3]
    day = 1440
    n_days = normal_days + tail_days

    txs: list[Tx] = []

    # Normal activity on the normal days; sparse normal rows in the tail.
    for d in range(n_days):
        n = tx_per_day if d < normal_days else tail_normal_per_day
        for _ in range(n):
            minute = d * day + int(rng.integers(0, day))
            fmt = str(
                rng.choice(NORMAL_FORMATS + ["Reinvestment"], p=[*NORMAL_FORMAT_P[:-1], 0.04, 0.04])
            )
            s = accounts[int(rng.integers(0, n_all))]
            if fmt == "Reinvestment":
                txs.append(_tx(minute, s, s, _amount(rng), fmt, 0))
                continue
            t = accounts[int(rng.integers(0, n_all))]
            cross = fmt in ("Wire", "Cheque") and rng.random() < 0.3
            txs.append(_tx(minute, s, t, _amount(rng), fmt, 0, cross=cross))

    # Planted pattern attempts. Starts are spread over days 1..10; late attempts spill into
    # the tail, as in the real data (the tail holds pattern completions).
    pattern_blocks: list[tuple[str, str, list[Tx]]] = []
    bitcoin_done = False
    for typology in TYPOLOGIES:
        for a in range(attempts_per_typology):
            edges, detail = _attempt_edges(rng, typology, launder_pool)
            start = int(rng.integers(0, normal_days * day - day // 2))
            if a == attempts_per_typology - 1:
                start = (normal_days - 1) * day + int(rng.integers(0, day // 2))
            gap = int(rng.integers(30, 400))
            block: list[Tx] = []
            minute = start
            for j, (si, di) in enumerate(edges):
                minute += int(rng.integers(1, gap))
                if j == 0 and a == 0:
                    minute = start  # two edges may share a minute (same-minute ties)
                fmt = "ACH"
                if not bitcoin_done and typology == "RANDOM":
                    fmt, bitcoin_done = "Bitcoin", True
                amt = float(np.round(rng.uniform(2_000, 60_000), 2))
                block.append(_tx(minute, launder_pool[si], launder_pool[di], amt, fmt, 1))
            # Rows inside a block are not strictly time-sorted in the real file.
            if len(block) > 2:
                block[-1], block[-2] = block[-2], block[-1]
            header = f"{typology}:  {detail}" if detail else typology
            pattern_blocks.append((typology, header, block))
            txs.extend(block)

    # "Integration" laundering: labelled positives that the Patterns file does not list.
    for _ in range(integration_positives):
        minute = int(rng.integers(0, n_days * day))
        s = launder_pool[int(rng.integers(0, len(launder_pool)))]
        t = accounts[int(rng.integers(0, n_all))]
        if s == t:
            continue
        txs.append(_tx(minute, s, t, float(np.round(rng.uniform(1_000, 30_000), 2)), "ACH", 1))

    # One exact duplicate of a normal row (all 11 columns identical).
    txs.append(txs[5])

    # Keep only minutes inside the simulated span.
    txs = [t for t in txs if t.minute < n_days * day]
    pattern_blocks = [
        (ty, h, [t for t in b if t.minute < n_days * day]) for ty, h, b in pattern_blocks
    ]

    # Write the CSV unsorted.
    order = rng.permutation(len(txs))
    csv_path = out_dir / "HI-Small_Trans.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(HEADER)
        for i in order:
            w.writerow(txs[int(i)].row())

    pat_path = out_dir / "HI-Small_Patterns.txt"
    with pat_path.open("w", newline="") as f:
        for typology, header, block in pattern_blocks:
            f.write(f"BEGIN LAUNDERING ATTEMPT - {header}\n")
            for t in block:
                f.write(",".join(t.row()) + "\n")
            f.write(f"END LAUNDERING ATTEMPT - {typology}\n\n")

    keys = {(t.from_bank, t.from_account) for t in txs} | {(t.to_bank, t.to_account) for t in txs}
    expected = {
        "rows": len(txs),
        "accounts": len(keys),
        "positives": sum(t.is_laundering for t in txs),
        "pattern_attempts": len(pattern_blocks),
        "pattern_transactions": sum(len(b) for _, _, b in pattern_blocks),
    }
    return SyntheticDataset(
        transactions_csv=csv_path, patterns_txt=pat_path, expected=expected, n_days=n_days
    )


if __name__ == "__main__":  # pragma: no cover
    import sys

    ds = make_synthetic(Path(sys.argv[1] if len(sys.argv) > 1 else "synthetic_out"))
    print(ds.expected)
