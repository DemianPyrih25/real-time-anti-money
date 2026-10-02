"""aml.data.download: no network; `download_file` is monkeypatched or fed file:// URLs."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

import pytest

from aml.data import download
from aml.io import read_json

CSV_BYTES = b"Timestamp,From Bank,Account\n2022/09/01 00:20,010,8000EBD30\n"
TXT_BYTES = b"BEGIN LAUNDERING ATTEMPT - STACK\nEND LAUNDERING ATTEMPT - STACK\n\n"


@pytest.fixture()
def fake_download(monkeypatch, data_cfg):
    """Serve the two files from memory; records every requested URL."""
    ds = data_cfg["dataset"]
    content = {ds["transactions_file"]: CSV_BYTES, ds["patterns_file"]: TXT_BYTES}
    calls: list[str] = []

    def fake(url: str, dest: Path, *, timeout: int = 120) -> Path:
        calls.append(url)
        name = next(n for n in content if url.endswith(n))
        Path(dest).write_bytes(content[name])
        return Path(dest)

    monkeypatch.setattr(download, "download_file", fake)
    monkeypatch.delenv("KAGGLE_API_TOKEN", raising=False)
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    monkeypatch.delenv("KAGGLE_KEY", raising=False)
    return calls, content


def test_public_url(data_cfg) -> None:
    url = download.public_url(data_cfg, "HI-Small_Trans.csv")
    assert url == (
        "https://www.kaggle.com/api/v1/datasets/download/"
        "ealtman2019/ibm-transactions-for-anti-money-laundering-aml/HI-Small_Trans.csv"
    )


def test_sha256_file(tmp_path) -> None:
    p = tmp_path / "f.bin"
    p.write_bytes(b"abc" * 1000)
    assert download.sha256_file(p) == hashlib.sha256(b"abc" * 1000).hexdigest()


def test_download_file_local_url(tmp_path) -> None:
    src = tmp_path / "src.csv"
    src.write_bytes(CSV_BYTES)
    dest = tmp_path / "out" / "dest.csv"
    assert download.download_file(src.resolve().as_uri(), dest) == dest
    assert dest.read_bytes() == CSV_BYTES
    assert not dest.with_suffix(".csv.part").exists()


def test_fetch_then_skip_by_checksum(tmp_path, data_cfg, fake_download) -> None:
    calls, content = fake_download
    raw = tmp_path / "raw"
    sums = download.fetch_dataset(data_cfg, raw)
    assert len(calls) == 2
    assert sums == {n: hashlib.sha256(b).hexdigest() for n, b in content.items()}
    assert read_json(raw / download.CHECKSUMS_FILE) == sums
    for n, b in content.items():
        assert (raw / n).read_bytes() == b

    assert download.fetch_dataset(data_cfg, raw) == sums
    assert len(calls) == 2  # both files skipped

    download.fetch_dataset(data_cfg, raw, force=True)
    assert len(calls) == 4


def test_checksum_mismatch_redownloads(tmp_path, data_cfg, fake_download) -> None:
    calls, content = fake_download
    raw = tmp_path / "raw"
    download.fetch_dataset(data_cfg, raw)
    csv_name = data_cfg["dataset"]["transactions_file"]
    (raw / csv_name).write_bytes(b"Timestamp,corrupted\n")
    sums = download.fetch_dataset(data_cfg, raw)
    assert len(calls) == 3 and calls[-1].endswith(csv_name)
    assert (raw / csv_name).read_bytes() == content[csv_name]
    assert sums[csv_name] == hashlib.sha256(content[csv_name]).hexdigest()


def test_zip_response_is_unpacked(tmp_path, monkeypatch, data_cfg) -> None:
    def fake(url: str, dest: Path, *, timeout: int = 120) -> Path:
        name = url.rsplit("/", 1)[-1]
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr(name, CSV_BYTES if name.endswith(".csv") else TXT_BYTES)
        Path(dest).write_bytes(buf.getvalue())
        return Path(dest)

    monkeypatch.setattr(download, "download_file", fake)
    raw = tmp_path / "raw"
    download.fetch_dataset(data_cfg, raw)
    assert (raw / data_cfg["dataset"]["transactions_file"]).read_bytes() == CSV_BYTES
    assert (raw / data_cfg["dataset"]["patterns_file"]).read_bytes() == TXT_BYTES
    assert sorted(p.name for p in raw.iterdir()) == sorted(
        [
            download.CHECKSUMS_FILE,
            *(data_cfg["dataset"][k] for k in ("transactions_file", "patterns_file")),
        ]
    )


def test_html_error_page_rejected(tmp_path, monkeypatch, data_cfg) -> None:
    def fake(url: str, dest: Path, *, timeout: int = 120) -> Path:
        Path(dest).write_bytes(b"<html>Please sign in</html>")
        return Path(dest)

    monkeypatch.setattr(download, "download_file", fake)
    for k in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(k, raising=False)
    raw = tmp_path / "raw"
    raw.mkdir()
    csv = raw / data_cfg["dataset"]["transactions_file"]
    csv.write_bytes(CSV_BYTES)  # an existing file without a recorded checksum
    with pytest.raises(RuntimeError) as err:
        download.fetch_dataset(data_cfg, raw)
    assert "does not look like" in str(err.value.__cause__)
    # The existing file is not clobbered by the bad response, and no temp files remain.
    assert csv.read_bytes() == CSV_BYTES
    assert sorted(p.name for p in raw.iterdir()) == [csv.name]


def _failing(url: str, dest: Path, *, timeout: int = 120) -> Path:
    raise OSError("network down")


def test_public_failure_without_credentials_raises(tmp_path, monkeypatch, data_cfg) -> None:
    monkeypatch.setattr(download, "download_file", _failing)
    for k in ("KAGGLE_API_TOKEN", "KAGGLE_USERNAME", "KAGGLE_KEY"):
        monkeypatch.delenv(k, raising=False)
    with pytest.raises(RuntimeError, match="no Kaggle credentials"):
        download.fetch_dataset(data_cfg, tmp_path / "raw")


def test_public_failure_falls_back_to_kaggle(tmp_path, monkeypatch, data_cfg) -> None:
    monkeypatch.setattr(download, "download_file", _failing)
    monkeypatch.setenv("KAGGLE_API_TOKEN", "dummy-not-a-real-token")
    used: list[tuple[str, str]] = []

    def fake_kaggle(dataset: str, file_name: str, dest: Path) -> None:
        used.append((dataset, file_name))
        dest.write_bytes(CSV_BYTES if file_name.endswith(".csv") else TXT_BYTES)

    monkeypatch.setattr(download, "_kaggle_download", fake_kaggle)
    sums = download.fetch_dataset(data_cfg, tmp_path / "raw")
    ds = data_cfg["dataset"]
    assert used == [
        (ds["kaggle_dataset"], ds["transactions_file"]),
        (ds["kaggle_dataset"], ds["patterns_file"]),
    ]
    assert set(sums) == {ds["transactions_file"], ds["patterns_file"]}
