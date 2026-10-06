"""The M5 core imports no Kafka or web package at module level (M6 reuses it in Modal's
cpu_image, which lacks them): an AST scan, and the inproc CLI in a process that cannot import
them."""

from __future__ import annotations

import ast
import subprocess
import sys

import pytest

from aml.serving import bundle
from tests.fixtures.stream_child import REPO_ROOT, child_env, last_json, scorer_args

FORBIDDEN = ("confluent_kafka", "fastapi", "uvicorn", "starlette", "prometheus_client")
CORE = (
    "aml.serving.settings",
    "aml.serving.scorer",
    "aml.serving.state",
    "aml.serving.alerts",
    "aml.serving.latency",
    "aml.streaming.codec",
    "aml.streaming.replayer",
)


def _is_type_checking(test: ast.expr) -> bool:
    return (isinstance(test, ast.Name) and test.id == "TYPE_CHECKING") or (
        isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING"
    )


def _import_time_imports(stmts: list[ast.stmt]) -> list[tuple[int, str]]:
    """(line, module) of every import that runs when the module is imported: everything outside
    function bodies and `if TYPE_CHECKING:` blocks (class bodies and try/if/with blocks run)."""
    out: list[tuple[int, str]] = []
    for node in stmts:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            continue
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            out.append((node.lineno, node.module or ""))
        elif isinstance(node, ast.If) and _is_type_checking(node.test):
            out += _import_time_imports(node.orelse)
        else:
            for name in ("body", "orelse", "finalbody", "handlers"):
                block = getattr(node, name, None)
                if isinstance(block, list):
                    out += _import_time_imports(block)
    return out


@pytest.mark.parametrize("module", CORE)
def test_core_modules_import_no_kafka_or_web_package(module):
    path = REPO_ROOT / "src" / (module.replace(".", "/") + ".py")
    assert path.is_file(), f"{module} is a core module and must exist"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    bad = [
        (line, name)
        for line, name in _import_time_imports(tree.body)
        if name.split(".")[0] in FORBIDDEN
    ]
    assert not bad, f"{module} imports {bad} at module level: import them where they are used"


def test_the_scan_sees_nested_imports():
    src = (
        "import os\nfrom typing import TYPE_CHECKING\nif TYPE_CHECKING:\n    import fastapi\n"
        "try:\n    import uvicorn\nexcept ImportError:\n    pass\n"
        "class A:\n    import starlette\n"
        "def f():\n    import confluent_kafka\n"
    )
    names = [n for _, n in _import_time_imports(ast.parse(src).body)]
    assert names == ["os", "typing", "uvicorn", "starlette"]


_BLOCKED_SCRIPT = """
import sys
for m in {forbidden!r}:
    sys.modules[m] = None  # any import of it raises ImportError
import importlib
for m in {core!r}:
    importlib.import_module(m)
from aml.serving import scorer
sys.exit(scorer.main(sys.argv[1:]))
"""


def test_inproc_cli_runs_without_kafka_or_web_packages(serving_bundle, fixture_tag, tmp_path):
    script = _BLOCKED_SCRIPT.format(forbidden=FORBIDDEN, core=CORE)
    args = scorer_args(serving_bundle, tmp_path / "runtime", fixture_tag)
    proc = subprocess.run(
        [sys.executable, "-c", script, *args],
        cwd=REPO_ROOT,
        env=child_env(),
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-4000:]
    s = last_json(proc.stdout)
    assert s["exit_code"] == 0 and s["result"] == "end"
    p = s["parity"]
    assert p["mismatches"] == dict.fromkeys(bundle.CHECKS, 0)
    assert p["alerts_ok"] is True and p["digest_ok"] is True
