"""Runtime state of the M5 scorer: the restart snapshot and the alerts database.

Core module: `aml.streaming.kafka` (confluent_kafka) is imported only by `init`.

The runtime dir (a named volume in compose, `./runtime` locally) holds `snapshots/scorer.snap`
(+ the `.json` sidecar `Engine.snapshot` writes) and `alerts.sqlite`. `init` wipes it on every
`docker compose up`, so stale state is refused, never worked around: a runtime snapshot that does
not restore, belongs to another bundle or breaks the offset lineage raises `RuntimeStateError`
(exit 2) and the scorer never falls back to the bundle snapshot.

Offset lineage: offset 0 is the bundle's `next_rank`, so every runtime snapshot satisfies
`next_rank - next_offset == metadata.next_rank`.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from aml.features.engine import Engine
from aml.features.spec import SnapshotError
from aml.serving import bundle
from aml.serving.settings import (
    LATENCY_DIR,
    ConfigError,
    Settings,
    add_cli_args,
    load_settings,
)

if TYPE_CHECKING:
    from aml.serving.scorer import Champion

log = logging.getLogger(__name__)

SNAPSHOT = "snapshots/scorer.snap"
ALERTS_DB = "alerts.sqlite"
SNAPSHOT_KIND = "runtime"
# The only names `reset` deletes (plus the snapshot writer's stray temp files).
KNOWN_FILES = (
    SNAPSHOT,
    SNAPSHOT + ".json",
    ALERTS_DB,
    ALERTS_DB + "-wal",
    ALERTS_DB + "-shm",
    ALERTS_DB + "-journal",
)
BROKER_WAIT_S = 120.0


class RuntimeStateError(RuntimeError):
    """Runtime or bundle state that must not be used (exit 2): run init / re-pull the bundle."""


@dataclass(frozen=True)
class Restored:
    eng: Engine
    header: dict[str, Any]
    origin: str  # "runtime" | "bundle"
    path: Path
    seconds: float

    @property
    def next_offset(self) -> int:
        return int(self.header["next_offset"])


class StateStore:
    """The runtime dir: restore (runtime snapshot, else the bundle's), save, reset."""

    def __init__(self, runtime_dir: Path) -> None:
        self.runtime_dir = Path(runtime_dir)
        self.snapshot_path = self.runtime_dir / SNAPSHOT
        self.alerts_db = self.runtime_dir / ALERTS_DB

    def restore(self, champion: Champion, *, end_offset: int) -> Restored:
        """The engine to start from and its header (next_offset = the first offset to consume).

        The runtime snapshot if it exists (refused, never skipped, when it fails a check), else
        the bundle snapshot (next_offset 0, state digest = metadata.snapshot_state_digest).
        """
        meta = champion.metadata
        if meta is None or champion.bundle_dir is None:
            raise RuntimeStateError("restore needs a champion loaded from a serving bundle")
        base_rank = int(meta["next_rank"])
        t0 = time.perf_counter()
        if self.snapshot_path.exists():
            path, origin = self.snapshot_path, "runtime"
            eng, header = _restore(path, champion, "runtime snapshot")
            extra = header.get("extra") or {}
            off = header.get("next_offset")
            problems = []
            if extra.get("kind") != SNAPSHOT_KIND:
                problems.append(f"extra.kind {extra.get('kind')!r} != {SNAPSHOT_KIND!r}")
            if extra.get("export_key") != champion.export_key:
                problems.append(f"export_key {extra.get('export_key')!r} != bundle's")
            if extra.get("booster_sha256") != champion.booster_sha256:
                problems.append("booster_sha256 differs from the bundle's booster")
            if type(off) is not int:
                problems.append(f"next_offset {off!r} is not an integer")
            elif header.get("next_rank") - off != base_rank:
                problems.append(
                    f"next_rank - next_offset = {header.get('next_rank') - off} != bundle "
                    f"next_rank {base_rank} (offset 0 must be the bundle's first event)"
                )
            elif not 0 <= off <= end_offset:
                problems.append(f"next_offset {off} outside [0, {end_offset}]")
            if problems:
                raise RuntimeStateError(
                    f"runtime snapshot {path} is not this run's state ({'; '.join(problems)}): "
                    "run init (a fresh `docker compose up`); the scorer never falls back to the "
                    "bundle snapshot"
                )
        else:
            path, origin = champion.bundle_dir / bundle.SNAPSHOT, "bundle"
            eng, header = _restore(path, champion, "bundle snapshot")
            if header.get("next_offset") != 0:
                raise RuntimeStateError(f"{path}: next_offset {header.get('next_offset')!r} != 0")
            if header.get("next_rank") != base_rank:
                raise RuntimeStateError(
                    f"{path}: next_rank {header.get('next_rank')} != metadata {base_rank}"
                )
            if header.get("state_digest") != meta.get("snapshot_state_digest"):
                raise RuntimeStateError(f"{path}: state digest differs from metadata.json")
        return Restored(eng, header, origin, path, time.perf_counter() - t0)

    def save(self, eng: Engine, *, next_offset: int, extra: dict[str, Any]) -> dict[str, Any]:
        """Write the runtime snapshot (temp file, fsync, os.replace) and return its sidecar."""
        return eng.snapshot(
            self.snapshot_path, next_offset=next_offset, extra={**extra, "kind": SNAPSHOT_KIND}
        )

    def reset(self) -> None:
        """Delete the known state files (only those); refuse a root or a bundle directory."""
        d = self.runtime_dir
        r = d.resolve()
        if r.parent == r:
            raise RuntimeStateError(f"refusing to reset {d}: it is a filesystem root")
        if (d / bundle.METADATA_FILE).exists():
            raise RuntimeStateError(f"refusing to reset {d}: it holds a serving bundle")
        d.mkdir(parents=True, exist_ok=True)
        for rel in KNOWN_FILES:
            (d / rel).unlink(missing_ok=True)
        snaps = self.snapshot_path.parent
        if snaps.is_dir():
            for tmp in snaps.glob(f".{self.snapshot_path.name}*.tmp-*"):
                tmp.unlink(missing_ok=True)
            if not any(snaps.iterdir()):
                snaps.rmdir()


def _restore(path: Path, champion: Champion, what: str) -> tuple[Engine, dict[str, Any]]:
    try:
        return Engine.restore(path, champion.spec)
    except (SnapshotError, OSError, ValueError, KeyError, TypeError) as e:
        raise RuntimeStateError(f"{what} {path} does not restore: {e}") from e


def clear_latency_dir(reports_dir: Path) -> int:
    """Delete the raw latency files (the files directly in reports/latency/); their count."""
    d = Path(reports_dir) / LATENCY_DIR
    n = 0
    if d.is_dir():
        for p in d.iterdir():
            if p.is_file():
                p.unlink()
                n += 1
    return n


def init(settings: Settings, *, topics: bool = True) -> None:
    """The compose one-shot: wait for the broker, wipe the runtime dir and the raw latency
    files, recreate both topics (1 partition each). Every `docker compose up` starts fresh."""
    kafka = None
    if topics:
        from aml.streaming import kafka  # confluent_kafka: the Kafka path only

        kafka.wait_for_broker(settings.bootstrap, BROKER_WAIT_S)
    StateStore(settings.runtime_dir).reset()
    n = clear_latency_dir(settings.reports_dir)
    log.info("runtime %s reset; %d raw latency files removed", settings.runtime_dir, n)
    if kafka is not None:
        kafka.reset_topics(settings.bootstrap, [settings.transactions_topic, settings.alerts_topic])
        log.info("topics %s and %s recreated", settings.transactions_topic, settings.alerts_topic)


def main(argv: Sequence[str] | None = None) -> int:
    """python -m aml.serving.state init [--no-topics] [--runtime --reports --config --bootstrap]"""
    p = argparse.ArgumentParser(prog="python -m aml.serving.state", description="M5 runtime state")
    sub = p.add_subparsers(dest="cmd", required=True)
    q = add_cli_args(sub.add_parser("init", help="fresh runtime dir, latency files and topics"))
    q.add_argument("--no-topics", action="store_true", help="skip the broker and the topics")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings(args, transport="kafka")
        init(settings, topics=not args.no_topics)
    except (ConfigError, RuntimeStateError) as e:
        log.error("init refused: %s", e)
        return 2
    except Exception:
        log.exception("init failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
