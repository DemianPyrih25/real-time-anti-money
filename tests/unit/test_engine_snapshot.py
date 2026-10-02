"""Engine snapshots and exact restart (M2 spec §7): the format, header checks, pending events,
STATE_FIELDS completeness and a restart in a fresh subprocess at 10 boundaries."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import struct
import subprocess
import sys
from array import array

import pytest

from aml.features import engine as engine_mod
from aml.features import snapshot as snap
from aml.features.engine import Engine, component_layout
from aml.features.spec import ACCOUNT_COLUMNS, ENGINE_VERSION, SnapshotError
from tests.unit.test_engine_core import (
    REPO_ROOT,
    Feed,
    make_spec,
    random_stream,
    row_bits,
    run_stream,
)


def _rewrite(data: bytes, **changes) -> bytes:
    """The snapshot with header fields replaced (the payload is untouched)."""
    header = snap.peek_header(data)
    (hlen,) = struct.unpack("<I", data[8:12])
    header.update(changes)
    text = json.dumps(header, sort_keys=True).encode()
    return snap.MAGIC + struct.pack("<I", len(text)) + text + data[12 + hlen :]


def _engine(spec, seed: int = 0, n: int = 200) -> tuple[Engine, list[tuple]]:
    events = random_stream(spec, seed=seed, n=n)
    eng = Engine.create(spec)
    for f in events:
        eng.process(eng.prepare(*f))
    return eng, events


# --- the format -----------------------------------------------------------------------------------


@pytest.mark.parametrize("compress", [True, False])
def test_encode_decode_round_trip(compress):
    source = array("i", range(10))
    comps = [
        ("a", array("i", [1, -2, 3])),
        ("b", array("q", [2**40, -1])),
        ("c", array("d", [0.5, math.nan, -0.0])),
        ("d", array("b", [-1, 5])),
        ("e", bytearray(b"\x00\x01")),
        ("f", array("B", [])),
    ]
    with memoryview(source) as mv, mv[3:] as tail:
        data = snap.encode({"x": 1}, [*comps, ("g", tail)], compress=compress)
    source.append(10)  # encode released every view it made
    comps[0][1].append(4)
    header, out = snap.decode(data)
    assert header["x"] == 1 and header["compression"] == ("zlib" if compress else "none")
    assert header["format"] == snap.SNAPSHOT_FORMAT and header["byteorder"] == "little"
    assert [c["name"] for c in header["components"]] == list("abcdefg")
    assert [c["typecode"] for c in header["components"]] == ["i", "q", "d", "b", "B", "B", "i"]
    assert out["a"] == array("i", [1, -2, 3]) and out["b"] == array("q", [2**40, -1])
    assert out["c"].tobytes() == array("d", [0.5, math.nan, -0.0]).tobytes()
    assert out["d"] == array("b", [-1, 5]) and out["e"] == array("B", [0, 1])
    assert len(out["f"]) == 0 and list(out["g"]) == list(range(3, 10))
    assert header["raw_nbytes"] == 12 + 16 + 24 + 2 + 2 + 0 + 28
    assert snap.peek_header(data) == header
    if not compress:
        assert len(data) == 12 + len(json.dumps(header, sort_keys=True)) + header["raw_nbytes"]


def test_digest_covers_semantic_fields_names_typecodes_and_bytes():
    sem = {k: 0 for k in snap.DIGEST_FIELDS}
    one = [("a", array("i", [1]))]
    d0 = snap.digest(sem, one)
    assert d0 == snap.digest(sem, [("a", array("i", [1]))])
    assert d0 != snap.digest(sem, [("a", array("i", [2]))])
    assert d0 != snap.digest(sem, [("b", array("i", [1]))])
    assert d0 != snap.digest(sem, [("a", array("q", [1]))])
    assert d0 != snap.digest({**sem, "clock": 1}, one)
    assert d0 == snap.digest({**sem, "next_offset": 9, "extra": {"x": 1}}, one)  # not semantic


@pytest.mark.parametrize(
    "damage, match",
    [
        (lambda d: b"XXXXXXXX" + d[8:], "magic"),
        (lambda d: d[:10], "magic|truncated"),
        (lambda d: d[:14], "truncated"),
        (lambda d: d[:-3], "sha256"),
        (lambda d: d[:-3] + bytes([d[-3] ^ 1]) + d[-2:], "sha256"),
        (lambda d: _rewrite(d, byteorder="big"), "byteorder"),
        (lambda d: _rewrite(d, format=2), "format"),
        (lambda d: _rewrite(d, compression="lz4"), "compression"),
        (lambda d: _rewrite(d, raw_nbytes=1), "bytes"),
        (
            lambda d: _rewrite(
                d,
                components=[
                    {"name": "a", "typecode": "i", "n_items": 99, "offset": 0, "nbytes": 400}
                ],
            ),
            "cover|component",
        ),  # fmt: skip
        (
            lambda d: _rewrite(
                d,
                components=[
                    {"name": "a", "typecode": "x", "n_items": 100, "offset": 0, "nbytes": 400}
                ],
            ),
            "component",
        ),  # fmt: skip
    ],
)
def test_decode_refuses_damaged_snapshots(damage, match):
    data = snap.encode({}, [("a", array("i", range(100)))], compress=True)
    snap.decode(data)
    with pytest.raises(SnapshotError, match=match):
        snap.decode(damage(data))


# --- engine snapshots -----------------------------------------------------------------------------


def test_bytes_round_trip_restores_an_identical_state():
    spec = make_spec(small=True, hubs=(5,), hub_cap=3, compact_min_rows=4)
    events = random_stream(spec, seed=2, n=400)
    split = next(k for k in range(250, 400) if events[k][2] > events[k - 1][2])
    eng = Engine.create(spec)
    for f in events[:split]:
        eng.process(eng.prepare(*f))
    eng.advance(events[split][2])  # applies everything before the split: nothing pending
    assert eng.ring.base < eng.ring.live_start  # dead rows exist and are not saved
    data = eng.snapshot()
    eng2, header = Engine.restore(data, spec)
    assert eng2.state_digest() == eng.state_digest() == header["state_digest"]
    assert (eng2.clock, eng2.next_rank, eng2.cursors) == (eng.clock, eng.next_rank, eng.cursors)
    assert eng2.ring.base == eng2.ring.live_start == eng.ring.live_start
    assert len(eng2.ring) == eng.ring.end_rank - eng.ring.live_start
    for name, _ in ACCOUNT_COLUMNS:
        assert getattr(eng2, name) == getattr(eng, name)
    assert eng2.slots.arrays == eng.slots.arrays and eng2.hub == eng.hub
    assert eng2.pairs.index == eng.pairs.index
    assert dict(eng2.pairs.columns()) == dict(eng.pairs.columns())
    for f in events[split:]:  # both continue identically
        assert row_bits(eng.process(eng.prepare(*f))) == row_bits(eng2.process(eng2.prepare(*f)))
    eng.advance(events[-1][2] + 1)
    eng2.advance(events[-1][2] + 1)
    assert eng.state_digest() == eng2.state_digest()
    eng2.check_invariants()


def test_restore_refuses_another_spec_or_engine_version(monkeypatch):
    spec = make_spec(small=True)
    eng, _ = _engine(spec, n=50)
    data = eng.snapshot()
    with pytest.raises(SnapshotError, match="spec_hash"):
        Engine.restore(data, make_spec(small=True, features={"caps": {"gap": 7}}))
    with pytest.raises(SnapshotError, match="n_accounts|spec_hash"):
        Engine.restore(data, make_spec(small=True, n_accounts=13))
    monkeypatch.setattr(engine_mod, "ENGINE_VERSION", ENGINE_VERSION + 1)
    with pytest.raises(SnapshotError, match="engine_version"):
        Engine.restore(data, spec)


def test_restore_refuses_corrupt_or_inconsistent_snapshots():
    spec = make_spec(small=True)
    eng, _ = _engine(spec, n=80)
    data = eng.snapshot()
    with pytest.raises(SnapshotError, match="sha256"):
        Engine.restore(data[:-5] + bytes([data[-5] ^ 0x10]) + data[-4:], spec)
    header, comps = snap.decode(data)
    keep = {k: header[k] for k in (*snap.DIGEST_FIELDS, "next_offset", "extra", "state_digest")}
    comps["acct.ever_out"][3] += 1  # a valid file whose state differs from its digest
    with pytest.raises(SnapshotError, match="state digest"):
        Engine.restore(snap.encode(keep, list(comps.items()), compress=True), spec)
    del comps["hub"]
    with pytest.raises(SnapshotError, match="component layout"):
        Engine.restore(snap.encode(keep, list(comps.items()), compress=False), spec)
    w = str(spec.W_max)
    bad = _rewrite(data, cursors={**header["cursors"], w: header["cursors"][w] + 1})
    with pytest.raises(SnapshotError, match="cursor"):
        Engine.restore(bad, spec)


def test_pending_events_are_never_saved_and_are_refed_after_restore():
    spec = make_spec(small=True)
    eng, events = _engine(spec, seed=6, n=120)
    feed = Feed(spec, eng)
    m = eng.clock + 3
    first = eng.next_rank
    pending = [feed.event(m, 1, 2), None, None]
    rows = [eng.process(pending[0])]
    feed.rank += 1
    for k in (1, 2):
        pending[k] = feed.event(m, k, 3, 777.0)
        rows.append(eng.process(pending[k]))
        feed.rank += 1
    assert eng.pending_count == 3
    data = eng.snapshot(next_offset=42, extra={"why": "test"})
    header = snap.peek_header(data)
    assert header["next_rank"] == first and header["clock"] == m and eng.next_rank == first + 3
    assert header["next_offset"] == 42 and header["extra"] == {"why": "test"}
    eng2, _ = Engine.restore(data, spec)
    assert (eng2.pending_count, eng2.next_rank, eng2.clock) == (0, first, m)
    assert [row_bits(eng2.process(ev)) for ev in pending] == [row_bits(r) for r in rows]
    eng.advance(m + 1)
    eng2.advance(m + 1)
    assert eng.state_digest() == eng2.state_digest()


def test_state_fields_are_complete():
    state, derived = Engine.STATE_FIELDS, Engine.DERIVED_FIELDS
    transient = Engine.TRANSIENT_FIELDS
    every = set(state) | set(derived) | set(transient)
    assert len(every) == len(state) + len(derived) + len(transient)  # disjoint
    spec = make_spec(small=True, hubs=(5,), hub_cap=3)
    fresh = Engine.create(spec)
    assert set(vars(fresh)) == every
    eng, _ = _engine(spec, n=150)
    assert set(vars(eng)) == every
    restored, header = Engine.restore(eng.snapshot(), spec)
    assert set(vars(restored)) == every
    # every state field is saved: a header field or components with its prefix
    names = [c["name"] for c in header["components"]]
    rows = header["next_rank"] - header["live_start"]
    layout = component_layout(spec, header["n_pairs"], rows)
    assert names == [n for n, _, _ in layout]
    prefix = {"ring": "ring.", "slots": "slot.", "hub": "hub", "pairs": "pairs."}
    prefix |= {name: f"acct.{name}" for name, _ in ACCOUNT_COLUMNS}
    for field in state:
        if field in ("clock", "next_rank", "cursors"):
            assert field in header
        else:
            assert any(n.startswith(prefix[field]) for n in names), field


def test_every_state_field_moves_the_digest():
    spec = make_spec(small=True)
    eng, _ = _engine(spec, n=150)
    eng.advance(eng.clock + 1)
    base = eng.state_digest()

    def bumped(arr) -> str:
        arr[0] += 1
        try:
            return eng.state_digest()
        finally:
            arr[0] -= 1

    ring = eng.ring
    live = ring.live_start - ring.base
    for name, col in ring.columns():
        old = col[live]
        col[live] = old + 1 if name != "l" else old + 0.5
        assert eng.state_digest() != base, name
        col[live] = old
    for s, a in eng.slots.arrays.items():
        assert bumped(a) != base, s.name
    for name, _ in ACCOUNT_COLUMNS:
        assert bumped(getattr(eng, name)) != base, name
    assert bumped(eng.hub) != base
    for name, col in eng.pairs.columns():
        assert bumped(col) != base, name
    eng.clock += 1
    assert eng.state_digest() != base
    eng.clock -= 1
    eng.cursors[spec.w_pt] -= 1
    assert eng.state_digest() != base
    eng.cursors[spec.w_pt] += 1
    assert eng.state_digest() == base


def test_snapshot_file_and_sidecar(tmp_path):
    spec = make_spec(small=True)
    eng, _ = _engine(spec, n=100)
    dst = tmp_path / "snaps" / "x.snap"
    side = eng.snapshot(dst, next_offset=0, extra={"day": 7})
    raw = dst.read_bytes()
    assert sorted(os.listdir(dst.parent)) == ["x.snap", "x.snap.json"]  # no temp files left
    assert side["file_sha256"] == hashlib.sha256(raw).hexdigest()
    assert side["file_nbytes"] == len(raw) and side["file"] == "x.snap"
    assert json.loads((tmp_path / "snaps" / "x.snap.json").read_text()) == side
    header = snap.peek_header(dst)
    assert {k: v for k, v in side.items() if not k.startswith("file")} == header
    assert header["next_offset"] == 0 and header["extra"] == {"day": 7}
    eng2, h2 = Engine.restore(dst, spec)  # from a path
    assert h2 == header and eng2.state_digest() == eng.state_digest()
    raw_snap = eng.snapshot(compress=False)
    assert snap.peek_header(raw_snap)["compression"] == "none" and len(raw_snap) > len(raw)
    assert Engine.restore(raw_snap, spec)[0].state_digest() == eng.state_digest()


def test_state_nbytes():
    spec = make_spec(small=True)
    eng, _ = _engine(spec, n=100)
    nb = eng.state_nbytes()
    assert set(nb) == {"ring", "slots", "accounts", "pairs", "transients", "total"}
    assert nb["total"] == sum(v for k, v in nb.items() if k != "total")
    assert nb["slots"] >= spec.slot_bytes_per_account * spec.n_accounts
    assert nb["ring"] >= 50 * len(eng.ring)


# --- exact restart in a fresh interpreter ---------------------------------------------------------

_CHILD = r"""
import json, struct, sys
from pathlib import Path
from aml.features.engine import Engine
from aml.features.spec import EngineSpec

def bits(row):
    return [struct.pack("<d", x).hex() if type(x) is float else x for x in row]

work = Path(sys.argv[1])
doc = json.loads((work / "input.json").read_text(encoding="utf-8"))
spec = EngineSpec.from_json(doc["spec"])
events = [tuple(e) for e in doc["events"]]
out = []
for name in doc["snapshots"]:
    eng, header = Engine.restore(work / name, spec)
    k = header["next_rank"]
    rows = [bits(eng.process(eng.prepare(*e))) for e in events[k:]]
    eng.advance(doc["end_minute"])
    eng.check_invariants()
    out.append({"name": name, "next_rank": k, "rows": rows, "digest": eng.state_digest()})
(work / "output.json").write_text(json.dumps(out), encoding="utf-8")
"""


def _bits(row: tuple) -> list:
    return [struct.pack("<d", x).hex() if type(x) is float else x for x in row]


def test_restart_in_a_fresh_subprocess_is_exact(tmp_path):
    spec = make_spec(small=True, hubs=(5,), hub_cap=3, compact_min_rows=4)
    events = random_stream(spec, seed=31, n=400)
    end_minute = events[-1][2] + 1
    rows_a, eng_a = run_stream(spec, events)  # uninterrupted (ends with advance(end_minute))
    digest_a = eng_a.state_digest()

    rng = random.Random(7)
    ks = range(1, len(events))
    inside = set(rng.sample([k for k in ks if events[k][2] == events[k - 1][2]], 3))
    empty = set(rng.sample([k for k in ks if events[k][2] > events[k - 1][2] + 1], 3))
    plain = set(rng.sample([k for k in ks if k not in inside | empty], 2))
    eng = Engine.create(spec)
    rows_b, snaps, kinds = [], [], []

    def take(kind: str) -> None:
        name = f"s{len(snaps)}.snap"
        if len(snaps) % 2:
            eng.snapshot(tmp_path / name, extra={"kind": kind})
        else:
            (tmp_path / name).write_bytes(eng.snapshot(compress=len(snaps) % 4 == 0))
        snaps.append(name)
        kinds.append((kind, eng.pending_count, eng.ring.base == eng.ring.live_start))

    compactions = 0
    for k, f in enumerate(events):
        if k in empty:
            eng.advance(events[k - 1][2] + 1)  # a minute with no events, then snapshot
            take("empty minute")
        if k in inside or k in plain:
            take("inside a pending minute" if k in inside else "boundary")
        base = eng.ring.base
        rows_b.append(eng.process(eng.prepare(*f)))
        if eng.ring.base != base and compactions < 2 and k > 50:
            compactions += 1
            take("right after a compaction")
    eng.advance(end_minute)
    # snapshots and the extra empty-minute advances change nothing for the live engine
    assert [_bits(r) for r in rows_b] == [_bits(r) for r in rows_a]
    assert eng.state_digest() == digest_a
    assert len(snaps) == 10 and compactions == 2
    assert all(n > 0 for kind, n, _ in kinds if kind == "inside a pending minute")
    assert all(n == 0 for kind, n, _ in kinds if kind == "empty minute")
    assert all(compacted for kind, _, compacted in kinds if kind == "right after a compaction")

    doc = {
        "spec": spec.to_json(),
        "events": [list(e) for e in events],
        "snapshots": snaps,
        "end_minute": end_minute,
    }
    (tmp_path / "input.json").write_text(json.dumps(doc), encoding="utf-8")
    (tmp_path / "child.py").write_text(_CHILD, encoding="utf-8")
    env = {**os.environ, "PYTHONHASHSEED": "4242", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, str(tmp_path / "child.py"), str(tmp_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads((tmp_path / "output.json").read_text(encoding="utf-8"))
    assert [o["name"] for o in out] == snaps
    want = [_bits(r) for r in rows_a]
    for o, (kind, _, _) in zip(out, kinds, strict=True):
        assert o["rows"] == want[o["next_rank"] :], f"{o['name']} ({kind}): rows differ"
        assert o["digest"] == digest_a, f"{o['name']} ({kind}): final digest differs"
