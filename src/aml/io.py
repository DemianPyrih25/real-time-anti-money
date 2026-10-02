"""Atomic file writes: write to a temp file in the same directory, then os.replace.

A reader (or a Volume commit) never sees a half-written file.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import polars as pl


def _tmp(path: Path) -> Path:
    return path.with_name(f".{path.name}.tmp-{os.getpid()}")


def write_parquet_atomic(df: pl.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    df.write_parquet(tmp, compression="zstd", statistics=True)
    os.replace(tmp, path)
    return path


def write_text_atomic(text: str, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp(path)
    # LF even on Windows: LightGBM refuses CRLF model files.
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return path


def write_json_atomic(obj: Any, path: Path) -> Path:
    return write_text_atomic(json.dumps(obj, indent=2, sort_keys=True, default=_json_default), path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    """Hex SHA-256 of a file's bytes (read in 1 MiB chunks)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _json_default(o: Any) -> Any:
    # numpy scalars / arrays and Paths
    if hasattr(o, "item") and callable(o.item) and getattr(o, "shape", None) == ():
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(f"not JSON serialisable: {type(o)}")
