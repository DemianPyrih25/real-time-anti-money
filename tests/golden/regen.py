"""Regenerate the engine golden file (M2 spec §9.2): `uv run python -m tests.golden.regen`.

The golden file holds every output value of the engine for the first 2,000 events of the
synthetic fixture (the same data as the `prepared` test fixture), floats as `float.hex`, with a
header recording ENGINE_VERSION, the spec hash and the platform. Regenerate it only for an
intended change of engine semantics (with an ENGINE_VERSION bump) and review the diff.
"""

from __future__ import annotations

import argparse
import copy
import csv
import platform
import shutil
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from aml.features.spec import ENGINE_VERSION, EngineSpec
from aml.paths import DataPaths
from tests.fixtures.engine_frames import fixture_spec, load_cfg, run_engine

GOLDEN_PATH = Path(__file__).with_name("engine_fixture_first2000.csv")
N_EVENTS = 2000
HUB_CAP = 30  # 9 hubs on the fixture, so hub exclusion and segmentation are exercised
REGEN_CMD = "uv run python -m tests.golden.regen"


def platform_tag() -> str:
    """Where libm results were recorded (log1p may differ by 1 ulp between platforms)."""
    return f"{sys.platform}-{platform.machine().lower()}"


def golden_inputs(paths: DataPaths, rules_cfg: dict) -> tuple[EngineSpec, pl.DataFrame]:
    """(spec, the first N_EVENTS events) of the prepared fixture at the default configs."""
    spec, tx = fixture_spec(paths, rules_cfg, load_cfg("features"), hub_cap=HUB_CAP)
    return spec, tx.head(N_EVENTS)


def encode(x) -> str:
    """A float as float.hex with trailing mantissa zeros dropped (float.fromhex reads it back
    bit for bit); an int as a decimal integer."""
    if isinstance(x, float):
        h = x.hex()
        if "p" in h:
            mant, exp = h.split("p")
            if "." in mant:
                mant = mant.rstrip("0").rstrip(".")
            h = f"{mant}p{exp}"
        return h
    return str(int(x))


def decode(s: str, is_float: bool):
    return float.fromhex(s) if is_float else int(s)


def write_golden(path: Path, spec: EngineSpec, tx: pl.DataFrame, rows: Sequence[tuple]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        f.write(f"# engine_version={ENGINE_VERSION}\n")
        f.write(f"# spec_hash={spec.spec_hash()}\n")
        f.write(f"# platform={platform_tag()}\n")
        f.write(f"# rows={len(rows)}\n")
        f.write(f"# regenerate: {REGEN_CMD}\n")
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["row_id", "rank", *spec.row_layout])
        for rid, rank, row in zip(tx["row_id"].to_list(), tx["rank"].to_list(), rows, strict=True):
            w.writerow([rid, rank, *(encode(x) for x in row)])


def read_golden(path: Path) -> tuple[dict[str, str], list[str], list[list[str]]]:
    """(header fields, column names, data rows as strings)."""
    header: dict[str, str] = {}
    with path.open(encoding="utf-8", newline="") as f:
        lines = f.read().splitlines()
    k = 0
    while k < len(lines) and lines[k].startswith("#"):
        key, _, value = lines[k][1:].strip().partition("=")
        header[key.strip()] = value.strip()
        k += 1
    reader = csv.reader(lines[k:])
    cols = next(reader)
    return header, cols, list(reader)


def build_fixture(root: Path) -> DataPaths:
    """The synthetic fixture prepared exactly as tests/conftest.py prepares it."""
    from aml.data.prepare import prepare_data
    from tests.fixtures.synthetic import make_synthetic

    ds = make_synthetic(root / "raw")
    data_cfg = copy.deepcopy(load_cfg("data"))
    data_cfg["expected"] = dict(ds.expected)
    paths = DataPaths(root / "volume")
    paths.raw_dir.mkdir(parents=True)
    for f in (ds.transactions_csv, ds.patterns_txt):
        shutil.copy(f, paths.raw_dir / f.name)
    prepare_data(paths, data_cfg, threads=2)
    return paths


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=GOLDEN_PATH)
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        paths = build_fixture(Path(tmp))
        spec, tx = golden_inputs(paths, load_cfg("rules"))
        rows = run_engine(tx, spec, check_every=500)
    write_golden(args.out, spec, tx, rows)
    print(f"wrote {args.out} ({len(rows)} rows, spec_hash {spec.spec_hash()}, {platform_tag()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
