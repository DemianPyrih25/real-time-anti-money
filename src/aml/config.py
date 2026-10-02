"""Config loading and hashing.

Configs are loaded by the local entrypoint (laptop), hashed into run keys and passed to remote
functions as plain dicts. Remote code never reads `configs/` from disk.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

CONFIG_NAMES = ("data", "rules", "lgbm", "features", "serving")


def load_config(name: str, config_dir: Path) -> dict[str, Any]:
    with (config_dir / f"{name}.yaml").open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_configs(config_dir: Path, names: tuple[str, ...] = CONFIG_NAMES) -> dict[str, dict]:
    return {n: load_config(n, config_dir) for n in names}


def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def config_hash(*parts: Any, length: int = 12) -> str:
    """Deterministic short hash of any JSON-able parts (dict key order does not matter)."""
    return hashlib.sha256(canonical_json(list(parts)).encode()).hexdigest()[:length]


def run_key(stage: str, *parts: Any) -> str:
    """e.g. run_key("rules", data_cfg, rules_cfg) -> "rules-3f2a9c1b0d4e"."""
    return f"{stage}-{config_hash(stage, *parts)}"


def rate_tag(rate: float) -> str:
    """Alert rate -> column-safe tag: 0.005 -> "0p005"."""
    return f"{rate:g}".replace(".", "p").replace("-", "m")
