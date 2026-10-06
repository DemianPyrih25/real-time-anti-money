"""Test-only child process: the scorer CLI with a fault right after one offset.

`python -m tests.fixtures.stream_child --kill-at K <scorer args>` exits with `os._exit(137)`
after event K was fully processed (alert committed, observers run): a SIGKILL-like stop with no
cleanup, which also works on Windows. `--term-at K` raises a real SIGTERM there (POSIX), so the
scorer stops before event K + 1 and writes its snapshot. Production code has no fault hook: this
module adds one as an observer.

Also the parent side: `run_module` runs `python -m <module>` from the repo root with an
environment free of the M5 variables.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
SERVING_CONFIG = REPO_ROOT / "configs" / "serving.yaml"
M5_ENV = (
    "AML_BUNDLE_DIR",
    "AML_RUNTIME_DIR",
    "AML_REPORTS_DIR",
    "AML_CONFIG",
    "KAFKA_BOOTSTRAP",
    "SCORER_URL",
    "REPLAY_MAX_EVENTS",
    "AML_HOST_NOTE",
)
KILL_CODE = 137


class _Fault:
    """Observer: the fault fires once, after the event at `offset`."""

    def __init__(self, offset: int, kind: str) -> None:
        self.offset = offset
        self.kind = kind

    def observe(self, ev: Any, s: Any, t: Any) -> None:
        if s.offset != self.offset:
            return
        if self.kind == "kill":
            os._exit(KILL_CODE)
        # Delivered to this (the consumer) thread before raise_signal returns, so the scorer's
        # handler sets its stop event before the loop asks for the next event.
        signal.raise_signal(signal.SIGTERM)


def main(argv: list[str] | None = None) -> int:
    from aml.serving import scorer

    p = argparse.ArgumentParser(prog="python -m tests.fixtures.stream_child")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--kill-at", type=int)
    g.add_argument("--term-at", type=int)
    args, rest = p.parse_known_args(argv)
    if args.kill_at is not None:
        fault = _Fault(args.kill_at, "kill")
    else:
        fault = _Fault(args.term_at, "term")
    return scorer.main(rest, observers=[fault])


# --- parent side ------------------------------------------------------------------------------


def child_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if k not in M5_ENV}
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONPATH"] = os.pathsep.join([str(REPO_ROOT), *(p for p in sys.path if p)])
    return env


def run_module(module: str, *args: str, timeout: float = 600) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", module, *args],
        cwd=REPO_ROOT,
        env=child_env(),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def scorer_args(bundle_dir: Path, runtime_dir: Path, alert_rate_tag: str) -> list[str]:
    """CLI arguments of an inproc scorer run on a fixture bundle."""
    return [
        "--transport",
        "inproc",
        "--bundle",
        str(bundle_dir),
        "--runtime",
        str(runtime_dir),
        "--reports",
        str(Path(runtime_dir).parent / "reports"),
        "--config",
        str(SERVING_CONFIG),
        "--alert-rate-tag",
        alert_rate_tag,
    ]


def last_json(stdout: str) -> dict[str, Any]:
    """The scorer's summary: the last stdout line."""
    return json.loads(stdout.strip().splitlines()[-1])


if __name__ == "__main__":
    sys.exit(main())
