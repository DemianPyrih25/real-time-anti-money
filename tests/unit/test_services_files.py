"""M5 infrastructure, read statically: docker-compose.yml, the Dockerfiles, .dockerignore, CI,
the Makefile targets and .gitignore. Nothing here needs Docker or a broker."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from aml.serving.settings import (
    ENV_BOOTSTRAP,
    ENV_BUNDLE,
    ENV_CONFIG,
    ENV_HOST_NOTE,
    ENV_MAX_EVENTS,
    ENV_REPORTS,
    ENV_RUNTIME,
    ENV_SCORER_URL,
    ServingConfig,
)
from tests.conftest import CONFIG_DIR, REPO_ROOT

KAFKA_IMAGE = "apache/kafka:4.3.1"
MEM_LIMITS = {"kafka": "1g", "init": "512m", "scorer": "3g", "replayer": "768m"}
COMMANDS = {
    "init": ["python", "-m", "aml.serving.state", "init"],
    "scorer": ["python", "-m", "aml.serving.app", "--transport", "kafka"],
    "replayer": ["python", "-m", "aml.streaming.replayer"],
}
DOCKERFILES = {"scorer": "services/scorer.Dockerfile", "replayer": "services/replayer.Dockerfile"}
SYNC = "uv sync --frozen --no-default-groups --group serve --no-install-project"
# Code, package data and templates the images need (stray build or run artefacts are not checked)
SOURCE_SUFFIXES = {".py", ".sql", ".yaml", ".json", ".txt", ".j2", ".jinja", ".html", ".md"}


def _load(rel: str) -> Any:
    with (REPO_ROOT / rel).open(encoding="utf-8") as f:
        return yaml.safe_load(f)


@pytest.fixture(scope="module")
def compose() -> dict:
    return _load("docker-compose.yml")


@pytest.fixture(scope="module")
def cfg() -> ServingConfig:
    return ServingConfig.load(CONFIG_DIR / "serving.yaml")


def _seconds(v: str) -> int:
    m = re.fullmatch(r"(\d+)(s|m)", v)
    assert m, v
    return int(m[1]) * (60 if m[2] == "m" else 1)


def _module_file(module: str) -> Path:
    return REPO_ROOT / "src" / (module.replace(".", "/") + ".py")


def _mounts(service: dict) -> dict[str, tuple[str, bool]]:
    """Short-syntax volumes as {target: (source, read_only)}."""
    out = {}
    for v in service.get("volumes", []):
        src, dst, *mode = v.split(":")
        out[dst] = (src, mode == ["ro"])
    return out


# --- docker-compose.yml ---------------------------------------------------------------------------


def test_compose_services_limits_and_volume(compose):
    svc = compose["services"]
    assert set(svc) == set(MEM_LIMITS)
    assert {n: s.get("mem_limit") for n, s in svc.items()} == MEM_LIMITS
    assert set(compose["volumes"]) == {"runtime"}


def test_compose_kafka_broker(compose):
    k = compose["services"]["kafka"]
    env = k["environment"]
    assert k["image"] == KAFKA_IMAGE
    assert env["KAFKA_PROCESS_ROLES"] == "broker,controller"  # KRaft, no ZooKeeper
    assert env["KAFKA_HEAP_OPTS"].split() == ["-Xms512m", "-Xmx512m"]
    assert env["KAFKA_AUTO_CREATE_TOPICS_ENABLE"] == "false"  # only init creates topics
    advertised = env["KAFKA_ADVERTISED_LISTENERS"].split(",")
    assert "EXTERNAL://localhost:9092" in advertised
    assert "command" not in k and "healthcheck" not in k  # init waits for the broker itself


def test_compose_ports_bound_to_localhost(compose, cfg):
    port = cfg.scorer.port
    ports = {n: s.get("ports", []) for n, s in compose["services"].items()}
    assert ports == {
        "kafka": ["127.0.0.1:9092:9092"],
        "init": [],
        "scorer": [f"127.0.0.1:{port}:{port}"],
        "replayer": [],
    }


def test_compose_depends_on_conditions(compose):
    dep = {
        n: {d: c["condition"] for d, c in s.get("depends_on", {}).items()}
        for n, s in compose["services"].items()
    }
    assert dep == {
        "kafka": {},
        "init": {"kafka": "service_started"},
        "scorer": {"kafka": "service_started", "init": "service_completed_successfully"},
        "replayer": {"init": "service_completed_successfully", "scorer": "service_healthy"},
    }


def test_compose_exec_form_commands_name_real_modules(compose):
    svc = compose["services"]
    for name, cmd in COMMANDS.items():
        s = svc[name]
        assert s["command"] == cmd, name  # a list: no shell between tini and python
        assert _module_file(cmd[2]).is_file(), cmd[2]
        assert s["init"] is True and s["restart"] == "no", name
        # init runs in the scorer image
        assert s["build"]["dockerfile"] == DOCKERFILES.get(name, DOCKERFILES["scorer"]), name


def test_compose_scorer_health_and_stop(compose, cfg):
    s = compose["services"]["scorer"]
    hc = s["healthcheck"]
    assert hc["test"][0] == "CMD"  # exec form
    assert f"http://127.0.0.1:{cfg.scorer.port}/health" in hc["test"][-1]
    assert hc["test"][1] == "python" and "urllib" in hc["test"][-1]
    assert _seconds(hc["start_period"]) >= 300
    assert s["stop_signal"] == "SIGTERM"
    grace = _seconds(s["stop_grace_period"])
    assert grace >= 100 and grace > cfg.scorer.stop_timeout_s  # the consumer join + snapshot


def test_compose_env_and_mounts_match_settings(compose, cfg):
    svc = compose["services"]
    advertised = svc["kafka"]["environment"]["KAFKA_ADVERTISED_LISTENERS"]
    for name in COMMANDS:
        s = svc[name]
        env, mounts = s["environment"], _mounts(s)
        assert set(env) == {
            ENV_BUNDLE, ENV_RUNTIME, ENV_REPORTS, ENV_CONFIG, ENV_BOOTSTRAP, ENV_SCORER_URL,
            ENV_MAX_EVENTS, ENV_HOST_NOTE, "AML_KAFKA_IMAGE",
        }  # fmt: skip
        assert env[ENV_BUNDLE] == "/bundle" and env[ENV_RUNTIME] == "/runtime"
        assert env[ENV_REPORTS] == "/reports"
        assert env[ENV_CONFIG] == "/app/configs/serving.yaml"
        assert f"://{env[ENV_BOOTSTRAP]}" in advertised  # the internal listener
        assert env[ENV_SCORER_URL] == f"http://scorer:{cfg.scorer.port}"
        assert env[ENV_MAX_EVENTS] == "${REPLAY_MAX_EVENTS:-}"  # unset -> empty -> yaml value
        assert env[ENV_HOST_NOTE] == "${AML_HOST_NOTE:-}"
        assert env["AML_KAFKA_IMAGE"] == KAFKA_IMAGE
        assert mounts["/app/configs"] == ("./configs", True)
        # the bundle is never written: every ./serving mount is read-only
        assert all(ro for src, ro in mounts.values() if src == "./serving"), name
    sc, rp, it = (_mounts(svc[n]) for n in ("scorer", "replayer", "init"))
    assert sc["/bundle"] == ("./serving", True) and rp["/bundle"] == ("./serving", True)
    assert sc["/runtime"] == ("runtime", False) and it["/runtime"] == ("runtime", False)
    assert sc["/reports"] == ("./reports", False) and it["/reports"] == ("./reports", False)
    assert "/bundle" not in it


# --- Dockerfiles ----------------------------------------------------------------------------------


def _instructions(rel: str) -> list[str]:
    """The Dockerfile's instructions, continuation lines joined, comments and blanks dropped."""
    out: list[str] = []
    cur = ""
    for raw in (REPO_ROOT / rel).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cur = f"{cur} {line}" if cur else line
        if cur.endswith("\\"):
            cur = cur[:-1].rstrip()
            continue
        out.append(re.sub(r"\s+", " ", cur))
        cur = ""
    assert not cur, f"{rel}: dangling continuation"
    return out


@pytest.mark.parametrize("name", sorted(DOCKERFILES))
def test_dockerfile(name, compose):
    ins = _instructions(DOCKERFILES[name])
    # Debian glibc (log1p bit parity with Modal's debian_slim), not Alpine's musl
    assert re.fullmatch(r"FROM python:3\.12-slim-(bookworm|trixie)", ins[0]), ins[0]
    assert sum(i.startswith("FROM ") for i in ins) == 1
    assert any(i.startswith("RUN ") and "libgomp1" in i for i in ins)  # LightGBM's OpenMP
    uv = [i for i in ins if "astral-sh/uv" in i]
    assert len(uv) == 1 and re.search(r"astral-sh/uv:\d+\.\d+\.\d+ ", uv[0]), uv  # pinned tag
    assert any(i.startswith("ENV ") and "PYTHONPATH=/app/src" in i for i in ins)
    sync = ins.index(f"RUN {SYNC}")
    check = ins.index("RUN python -m aml.serving.scorer --self-check")
    assert sync < ins.index("COPY src ./src") < check  # deps layer cached apart from the code
    assert not any(re.search(r"torch|--all-groups|--group gnn", i, re.I) for i in ins)
    assert ins[-1].startswith("CMD [")
    assert json.loads(ins[-1][4:]) == compose["services"][name]["command"]  # JSON (exec) form


def test_dockerfiles_differ_only_in_expose_and_cmd(cfg):
    scorer, replayer = _instructions(DOCKERFILES["scorer"]), _instructions(DOCKERFILES["replayer"])
    assert scorer[:-2] == replayer[:-1]  # shared layers come from the build cache
    assert scorer[-2] == f"EXPOSE {cfg.scorer.port}"
    assert not any(i.startswith("EXPOSE") for i in replayer)


# --- .dockerignore --------------------------------------------------------------------------------


def _pattern_re(pat: str) -> re.Pattern[str]:
    """A .dockerignore pattern (relative to the context root) as a regex over a POSIX path."""
    out, i, pat = [], 0, pat.strip("/")
    while i < len(pat):
        if pat.startswith("**/", i):
            out.append("(?:.*/)?")
            i += 3
        elif pat.startswith("**", i):
            out.append(".*")
            i += 2
        elif pat[i] == "*":
            out.append("[^/]*")
            i += 1
        elif pat[i] == "?":
            out.append("[^/]")
            i += 1
        elif pat[i] == "[":
            j = pat.index("]", i)
            out.append(pat[i : j + 1])
            i = j + 1
        else:
            out.append(re.escape(pat[i]))
            i += 1
    return re.compile("".join(out))


def _ignored(rel: str, patterns: list[re.Pattern[str]]) -> bool:
    """Docker excludes a path when a pattern matches it or one of its parent directories."""
    parts = rel.split("/")
    prefixes = ["/".join(parts[: i + 1]) for i in range(len(parts))]
    return any(rx.fullmatch(p) for rx in patterns for p in prefixes)


def test_dockerignore():
    lines = (REPO_ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    pats = [ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")]
    assert not any(p.startswith("!") for p in pats)  # no re-includes (the matcher has none)
    rx = [_pattern_re(p) for p in pats]
    excluded = [
        "serving", "serving/metadata.json", "serving/replay/slice.parquet", "x.parquet",
        "src/aml/x.parquet", ".venv/bin/python", ".git/HEAD", "API.txt", "configs/API.txt",
        "tests/conftest.py", "reports/latency.json", "runtime/alerts.sqlite",
        "modal_jobs/common.py", "README.md",
    ]  # fmt: skip
    assert [p for p in excluded if not _ignored(p, rx)] == []
    kept = ["pyproject.toml", "uv.lock"] + [
        p.relative_to(REPO_ROOT).as_posix()
        for d in ("src", "configs")
        for p in (REPO_ROOT / d).rglob("*")
        if p.is_file() and p.suffix in SOURCE_SUFFIXES and "__pycache__" not in p.parts
    ]
    must = {"src/aml/serving/scorer.py", "src/aml/features/engine.py", "configs/serving.yaml"}
    assert must <= set(kept)
    # root-anchored data folders must not drop the same-named packages (src/aml/data, features)
    assert [p for p in kept if _ignored(p, rx)] == []


# --- CI, Makefile, .gitignore ---------------------------------------------------------------------


def test_ci_kafka_service_and_compose_job():
    jobs = _load(".github/workflows/ci.yml")["jobs"]
    test = jobs["test"]
    assert test["env"]["AML_TEST_KAFKA"] == "localhost:9092"
    assert test["env"]["AML_REQUIRE_KAFKA"] == "1"  # a missing broker fails, never skips
    k = test["services"]["kafka"]
    assert k["image"] == KAFKA_IMAGE and "9092:9092" in k["ports"]
    assert k["env"]["KAFKA_ADVERTISED_LISTENERS"] == "PLAINTEXT://localhost:9092"
    assert k["env"]["KAFKA_AUTO_CREATE_TOPICS_ENABLE"] == "false"
    runs = [s.get("run") for s in jobs["compose"]["steps"]]
    assert runs.index("docker compose config --quiet") < runs.index("docker compose build")


def test_makefile_targets_and_gitignore():
    lines = (REPO_ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
    phony = {t for ln in lines if ln.startswith(".PHONY:") for t in ln.split(":", 1)[1].split()}
    recipes = {
        "demo": "docker compose up --build",
        "demo-down": "docker compose down --volumes",
        "bundle-fixture": "docker compose run --rm --build --no-deps -v ./tests:/app/tests:ro"
        " -v ./serving:/out -e PYTHONPATH=/app/src:/app scorer"
        " python -m tests.fixtures.serving_bundle --out /out",
    }
    for target, recipe in recipes.items():
        assert target in phony, target
        assert lines[lines.index(f"{target}:") + 1] == f"\t{recipe}", target
    assert any(
        ln.startswith('\t@echo "M5:') and all(t in ln.split() for t in recipes) for ln in lines
    )
    assert (REPO_ROOT / "tests" / "fixtures" / "serving_bundle.py").is_file()
    gitignore = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert "/runtime/" in gitignore and "/reports/latency/" in gitignore
