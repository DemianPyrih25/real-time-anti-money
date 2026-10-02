"""Download the raw IBM AML files and record SHA-256 checksums.

The public Kaggle URL works anonymously; the `kaggle` client (Modal Secret `kaggle`, env
`KAGGLE_API_TOKEN` or `KAGGLE_USERNAME` + `KAGGLE_KEY`) is only a fallback. Kaggle may serve a
single file zipped, so a zip is unpacked transparently.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

from aml.io import read_json, write_json_atomic

CHECKSUMS_FILE = "checksums.json"
_CHUNK = 1 << 20

# First bytes a genuine file must contain (guards against saving an HTML error page).
_SIGNATURES = {".csv": b"Timestamp", ".txt": b"LAUNDERING ATTEMPT"}


def public_url(data_cfg: dict, file_name: str) -> str:
    ds = data_cfg["dataset"]
    return ds["public_url_template"].format(dataset=ds["kaggle_dataset"], file=file_name)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def download_file(url: str, dest: Path, *, timeout: int = 120) -> Path:
    """Stream `url` to `dest` via a `.part` file, then atomically rename. Redirects are followed;
    HTTP errors raise `urllib.error.HTTPError`."""
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "aml-prepare/0.1"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, part.open("wb") as out:
            shutil.copyfileobj(resp, out, length=_CHUNK)
        os.replace(part, dest)
    finally:
        part.unlink(missing_ok=True)
    return dest


def _unpack_to(src: Path, dest: Path, file_name: str) -> None:
    """Move `src` to `dest`, extracting `file_name` first if `src` is a zip archive."""
    if zipfile.is_zipfile(src):
        with zipfile.ZipFile(src) as zf:
            names = [n for n in zf.namelist() if not n.endswith("/")]
            member = next((n for n in names if Path(n).name == file_name), None)
            if member is None and len(names) == 1:
                member = names[0]
            if member is None:
                raise ValueError(f"{file_name} not found in downloaded zip: {names}")
            part = dest.with_suffix(dest.suffix + ".part")
            with zf.open(member) as fin, part.open("wb") as fout:
                shutil.copyfileobj(fin, fout, length=_CHUNK)
        os.replace(part, dest)
        src.unlink(missing_ok=True)
    else:
        os.replace(src, dest)


def _check_signature(path: Path, file_name: str) -> None:
    sig = _SIGNATURES.get(Path(file_name).suffix.lower())
    if sig is None:
        return
    with path.open("rb") as f:
        head = f.read(4096)
    if sig not in head:
        raise ValueError(
            f"{file_name} does not look like the expected file (first bytes: {head[:80]!r})"
        )


def _has_kaggle_credentials() -> bool:
    env = os.environ
    return bool(env.get("KAGGLE_API_TOKEN")) or bool(
        env.get("KAGGLE_USERNAME") and env.get("KAGGLE_KEY")
    )


def _kaggle_download(dataset: str, file_name: str, dest: Path) -> None:
    from kaggle.api.kaggle_api_extended import KaggleApi  # heavy; only needed on fallback

    api = KaggleApi()
    api.authenticate()
    with tempfile.TemporaryDirectory(dir=dest.parent, prefix=".kaggle-") as tmp:
        api.dataset_download_file(dataset, file_name, path=tmp, force=True, quiet=True)
        files = [p for p in Path(tmp).iterdir() if p.is_file()]
        if len(files) != 1:
            raise RuntimeError(f"kaggle client produced {len(files)} files for {file_name}")
        _unpack_to(files[0], dest, file_name)


def _fetch_one(data_cfg: dict, file_name: str, dest: Path) -> None:
    """Public URL first, Kaggle client second; `dest` is replaced only by a checked file."""
    tmp = dest.with_name(f".{dest.name}.download")
    staged = dest.with_name(f".{dest.name}.staged")
    try:
        try:
            download_file(public_url(data_cfg, file_name), tmp)
            _unpack_to(tmp, staged, file_name)
            _check_signature(staged, file_name)
        except Exception as public_err:
            if not _has_kaggle_credentials():
                raise RuntimeError(
                    f"public download of {file_name} failed and no Kaggle credentials are set"
                ) from public_err
            _kaggle_download(data_cfg["dataset"]["kaggle_dataset"], file_name, staged)
            _check_signature(staged, file_name)
        os.replace(staged, dest)
    finally:
        tmp.unlink(missing_ok=True)
        staged.unlink(missing_ok=True)


def fetch_dataset(data_cfg: dict, raw_dir: Path, *, force: bool = False) -> dict[str, str]:
    """Ensure the transactions and patterns files are in `raw_dir`; returns {file: sha256}.

    A file is skipped when it exists and its SHA-256 matches `raw_dir/checksums.json`.
    """
    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    ck_path = raw_dir / CHECKSUMS_FILE
    recorded: dict[str, str] = read_json(ck_path) if ck_path.exists() else {}
    ds = data_cfg["dataset"]
    out: dict[str, str] = {}
    for file_name in (ds["transactions_file"], ds["patterns_file"]):
        dest = raw_dir / file_name
        if not force and dest.exists() and recorded.get(file_name):
            digest = sha256_file(dest)
            if digest == recorded[file_name]:
                out[file_name] = digest
                continue
        _fetch_one(data_cfg, file_name, dest)
        out[file_name] = sha256_file(dest)
    write_json_atomic({**recorded, **out}, ck_path)
    return out
