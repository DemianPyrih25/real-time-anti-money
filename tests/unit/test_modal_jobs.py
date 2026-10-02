"""modal_jobs import offline, follow the guardrails, and compute stable run keys; repo infra."""

from __future__ import annotations

import copy
import importlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGES = ("smoke", "prepare_data", "rules", "train_lgbm", "evaluate", "mlflow_ui")
# M2 stage modules, checked by the same import and guardrail tests once they exist.
M2_STAGES = tuple(
    s for s in ("build_features", "export") if (REPO_ROOT / "modal_jobs" / f"{s}.py").exists()
)
NO_MODAL_CONFIG = str(Path(tempfile.gettempdir()) / "aml-tests-no-modal.toml")
# The M1 run keys of the current configs: M1 outputs on the Volume live under them (the reported
# results used rules-851aa23fb506 and lgbm_tx-875618a5b3f2). M2 must not re-key them (M2 spec §0);
# update only on a deliberate M1 config change.
M1_KEYS = {
    "data": "data-92dd21b72722",
    "rules": "rules-851aa23fb506",
    "lgbm_tx": "lgbm_tx-875618a5b3f2",
}

# Imports every stage module with DNS/network to anything but localhost blocked, then reports
# each app's name, functions and function specs.
_IMPORT_SCRIPT = r"""
import importlib, json, socket, sys

_orig_getaddrinfo = socket.getaddrinfo
def _guard(host, *a, **k):
    if host not in ("localhost", "127.0.0.1", "::1", None):
        raise RuntimeError(f"network lookup during import: {host}")
    return _orig_getaddrinfo(host, *a, **k)
socket.getaddrinfo = _guard

out = {}
for name in sys.argv[1:]:
    mod = importlib.import_module(f"modal_jobs.{name}")
    fns = {}
    for fname, fn in mod.app.registered_functions.items():
        spec = getattr(fn, "_spec_", None) or getattr(fn, "_spec", None)
        fns[fname] = None if spec is None else {
            "gpus": spec.gpus,
            "cpu": spec.cpu,
            "memory": list(spec.memory) if isinstance(spec.memory, tuple) else spec.memory,
            "volumes": sorted(str(k) for k in spec.volumes),
            "secrets": len(spec.secrets),
        }
    out[name] = {
        "app": mod.app.name,
        "APP_NAME": getattr(mod, "APP_NAME", None),
        "functions": fns,
        "entrypoints": sorted(mod.app.registered_entrypoints),
    }
print("RESULT=" + json.dumps(out))
"""


def _common():
    os.environ.setdefault("MODAL_CONFIG_PATH", NO_MODAL_CONFIG)
    return importlib.import_module("modal_jobs.common")


@pytest.fixture(scope="module")
def imported() -> dict:
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("MODAL_TOKEN", "MODAL_PROFILE"))
    }
    env.update(
        MODAL_CONFIG_PATH=NO_MODAL_CONFIG,
        MODAL_SERVER_URL="https://modal-server.invalid",
        PYTHONIOENCODING="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-c", _IMPORT_SCRIPT, *STAGES, *M2_STAGES],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    line = next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT="))
    return json.loads(line.removeprefix("RESULT="))


def test_modules_import_offline_and_apps_are_named(imported):
    assert set(imported) == set(STAGES) | set(M2_STAGES)
    for stage, info in imported.items():
        assert info["app"].startswith("aml-"), stage
        assert info["app"] == info["APP_NAME"], stage
        assert info["functions"], f"{stage} registers no function"
        if stage != "mlflow_ui":
            assert info["entrypoints"], f"{stage} has no local entrypoint"


def test_function_specs_follow_guardrails(imported):
    for stage, info in imported.items():
        for fname, spec in info["functions"].items():
            if spec is None:  # private Modal attribute moved: covered by cpu_job tests below
                continue
            where = f"{stage}.{fname}"
            assert spec["volumes"] == ["/data"], where
            assert isinstance(spec["memory"], list) and spec["memory"][1] > spec["memory"][0], where
            assert spec["cpu"], where
            # The GPU is for the smoke test (and later GNN training) only.
            if stage == "smoke" and fname == "gpu_check":
                assert spec["gpus"] == "L4", where
            else:
                assert not spec["gpus"], where
            # The kaggle Secret is attached at call time (only if it exists), never statically;
            # the one allowed Secret is the anonymous one Modal builds from `env=`.
            assert spec["secrets"] <= 1, where


def test_cpu_job_preset():
    common = _common()
    kw = common.cpu_job(cpu=8.0, memory_mib=16384, timeout=3600)
    assert kw["image"] is common.cpu_image
    assert kw["cpu"] == 8.0 and kw["memory"] == (16384, 20480)
    assert kw["max_containers"] == 1 and kw["scaledown_window"] == 2
    assert kw["timeout"] == 3600 and "retries" not in kw and "gpu" not in kw
    assert set(kw["volumes"]) == {"/data"}
    assert kw["env"]["POLARS_MAX_THREADS"] == "8" and kw["env"]["OMP_NUM_THREADS"] == "8"


def test_stage_cpu_matches_thread_env(imported):
    # the thread caps equal the CPU request in every stage module
    for stage in ("prepare_data", "rules", "train_lgbm", "evaluate"):
        mod = importlib.import_module(f"modal_jobs.{stage}")
        assert mod.CPU in (4.0, 8.0)
        fn = next(iter(imported[stage]["functions"].values()))
        if fn is not None:
            assert fn["cpu"] == mod.CPU


def test_run_keys_are_stable_and_scoped():
    common = _common()
    cfgs = common.load_all_configs()
    keys = common.all_keys(cfgs)
    assert keys == common.all_keys(copy.deepcopy(cfgs))
    assert keys["data"].startswith("data-") and keys["rules"].startswith("rules-")
    assert keys["lgbm_tx"].startswith("lgbm_tx-") and keys["eval"].startswith("eval-")

    # evaluation settings do not invalidate the prepared data or the trained models
    c = copy.deepcopy(cfgs)
    c["data"]["evaluation"]["bootstrap_replicates"] += 1
    k = common.all_keys(c)
    assert (k["data"], k["rules"], k["lgbm_tx"]) == (keys["data"], keys["rules"], keys["lgbm_tx"])
    assert k["eval"] != keys["eval"]

    # a model config change re-keys that model and evaluation only
    c = copy.deepcopy(cfgs)
    c["lgbm"]["seeds"] = [0]
    k = common.all_keys(c)
    assert k["lgbm_tx"] != keys["lgbm_tx"] and k["rules"] == keys["rules"]
    assert k["eval"] != keys["eval"]

    # a split change re-keys everything downstream of the data
    c = copy.deepcopy(cfgs)
    c["data"]["split"]["train"] = [1, 5]
    k = common.all_keys(c)
    assert all(k[s] != keys[s] for s in keys)


@pytest.mark.parametrize(
    "edit",
    [
        lambda c: c["rules"].update(alert_rate=0.004),
        lambda c: c["rules"]["scenarios"]["fan_in_velocity"].update(grid=[5, 10]),
        lambda c: c["rules"]["scenarios"]["round_trip"].update(hop_window_minutes=720),
        lambda c: c["rules"].update(high_risk_formats=["Cash"]),
    ],
)
def test_rules_scenario_edits_do_not_rekey_lightgbm(edit):
    """LightGBM reads only round_unit from the rules config: iterating on the scenarios must not
    force a paid, bit-identical LightGBM HPO re-run."""
    common = _common()
    cfgs = common.load_all_configs()
    keys = common.all_keys(cfgs)
    c = copy.deepcopy(cfgs)
    edit(c)
    k = common.all_keys(c)
    assert (k["data"], k["lgbm_tx"]) == (keys["data"], keys["lgbm_tx"])
    assert k["rules"] != keys["rules"] and k["eval"] != keys["eval"]


def test_round_unit_rekeys_lightgbm_and_rules():
    common = _common()
    cfgs = common.load_all_configs()
    keys = common.all_keys(cfgs)
    c = copy.deepcopy(cfgs)
    c["rules"]["round_unit"] = 1000
    k = common.all_keys(c)
    assert k["lgbm_tx"] != keys["lgbm_tx"] and k["rules"] != keys["rules"]
    assert k["data"] == keys["data"]


@pytest.mark.parametrize(
    "edit",
    [
        lambda c: c["data"]["test_views"].update(primary=[9, 9]),
        lambda c: c["data"]["dataset"].update(public_url_template="https://example.invalid/{file}"),
    ],
)
def test_eval_only_data_edits_keep_the_rules_and_model_keys(edit):
    """test_views feed only the EDA and evaluation, the URL only where the same files come from:
    the prepared-data marker re-keys (prepare re-runs, cheaply), rules and models do not."""
    common = _common()
    cfgs = common.load_all_configs()
    keys = common.all_keys(cfgs)
    c = copy.deepcopy(cfgs)
    edit(c)
    k = common.all_keys(c)
    assert (k["rules"], k["lgbm_tx"]) == (keys["rules"], keys["lgbm_tx"])
    assert k["eval"] != keys["eval"]
    assert common.data_content_key(c) == common.data_content_key(cfgs)


# --- M2 run keys -----------------------------------------------------------------------------


def test_m1_run_keys_are_unchanged_by_m2():
    common = _common()
    cfgs = common.load_all_configs()
    assert set(cfgs) == {"data", "rules", "lgbm", "features", "serving"}
    keys = common.all_keys(cfgs)
    assert {k: keys[k] for k in M1_KEYS} == M1_KEYS
    assert set(keys) == {
        "data",
        "rules",
        "rules_engine",
        "features",
        "lgbm_tx",
        "lgbm_graph",
        "eval",
        "export",
    }
    for name, key in keys.items():
        assert key.startswith(f"{name}-"), name


def test_lgbm_key_ignores_the_graph_section():
    from aml.config import run_key

    common = _common()
    cfgs = common.load_all_configs()
    assert "graph" in cfgs["lgbm"]
    keys = common.all_keys(cfgs)
    # the M1 formula on the M1 config, written out
    m1_lgbm = {k: v for k, v in cfgs["lgbm"].items() if k != "graph"}
    rules_part = {"round_unit": cfgs["rules"]["round_unit"]}
    m1 = run_key("lgbm_tx", common.data_content_key(cfgs), m1_lgbm, rules_part)
    assert common.lgbm_key(cfgs) == m1 == M1_KEYS["lgbm_tx"]
    assert common.lgbm_tx_cfg(cfgs["lgbm"]) == m1_lgbm and "graph" in cfgs["lgbm"]

    for edit in (
        lambda c: c["lgbm"].pop("graph"),
        lambda c: c["lgbm"]["graph"]["optuna"].update(n_trials=3),
        lambda c: c["lgbm"]["graph"]["gate"].update(psi_max=0.1),
    ):
        c = copy.deepcopy(cfgs)
        edit(c)
        k = common.all_keys(c)
        assert k["lgbm_tx"] == keys["lgbm_tx"]
        assert k["lgbm_graph"] != keys["lgbm_graph"] and k["eval"] != keys["eval"]
        assert k["export"] != keys["export"]
        assert (k["features"], k["rules_engine"]) == (keys["features"], keys["rules_engine"])

    # an M1 setting re-keys both models
    c = copy.deepcopy(cfgs)
    c["lgbm"]["base_params"]["num_leaves"] = 31
    k = common.all_keys(c)
    assert k["lgbm_tx"] != keys["lgbm_tx"] and k["lgbm_graph"] != keys["lgbm_graph"]


@pytest.mark.parametrize(
    "edit",
    [
        lambda c: c["rules"].update(alert_rate=0.004),
        lambda c: c["rules"].update(sensitivity_alert_rates=[0.002]),
        lambda c: c["rules"].update(tune_split="train"),
        lambda c: c["rules"]["scenarios"]["fan_in_velocity"].update(grid=[5, 10]),
        lambda c: c["rules"]["scenarios"]["round_trip"].update(grid=[1]),
        lambda c: c["features"].update(memory_target_mb=2000),
        lambda c: c["features"]["bench"].update(last_day=2),
    ],
)
def test_rules_retuning_and_bench_sizing_never_force_a_replay(edit):
    common = _common()
    cfgs = common.load_all_configs()
    keys = common.all_keys(cfgs)
    c = copy.deepcopy(cfgs)
    edit(c)
    k = common.all_keys(c)
    assert (k["features"], k["lgbm_graph"], k["lgbm_tx"]) == (
        keys["features"],
        keys["lgbm_graph"],
        keys["lgbm_tx"],
    )


@pytest.mark.parametrize(
    "edit",
    [
        lambda c: c["rules"]["scenarios"]["fan_in_velocity"].update(window_minutes=720),
        lambda c: c["rules"]["scenarios"]["round_trip"].update(hop_window_minutes=720),
        lambda c: c["rules"]["scenarios"]["structuring"].update(exclude_hub_senders=True),
        lambda c: c["rules"].update(high_risk_formats=["Cash"]),
        lambda c: c["rules"].update(round_unit=1000),
        lambda c: c["rules"].update(structuring_threshold_usd=5000),
        lambda c: c["rules"].update(structuring_band_low=0.8),
        lambda c: c["rules"].update(hub_degree_quantile=0.99),
        lambda c: c["features"]["windows"].update(long=2880),
        lambda c: c["features"]["caps"].update(port=255),
        lambda c: c["features"]["budgets"].update(feat_visits=1000),
        lambda c: c["features"]["ring"].update(compact_min_rows=1024),
        lambda c: c["features"]["replay"].update(row_group_rows=1024),
        lambda c: c["features"]["snapshots"].update(boundaries=["test"]),
        lambda c: c["data"]["split"].update(val_early=[7, 7], val_late=[8, 9], test=[10, 18]),
    ],
)
def test_engine_inputs_rekey_the_replay_and_everything_downstream(edit):
    common = _common()
    cfgs = common.load_all_configs()
    keys = common.all_keys(cfgs)
    c = copy.deepcopy(cfgs)
    edit(c)
    k = common.all_keys(c)
    for name in ("features", "rules_engine", "lgbm_graph", "eval", "export"):
        assert k[name] != keys[name], name


def test_features_key_tracks_the_engine_version(monkeypatch):
    common = _common()
    from aml.features import spec

    cfgs = common.load_all_configs()
    keys = common.all_keys(cfgs)
    monkeypatch.setattr(spec, "ENGINE_VERSION", spec.ENGINE_VERSION + 1)
    k = common.all_keys(cfgs)
    assert k["features"] != keys["features"] and k["export"] != keys["export"]
    assert {n: k[n] for n in M1_KEYS} == M1_KEYS


def test_serving_and_eval_options_rekey_only_their_stage():
    common = _common()
    cfgs = common.load_all_configs()
    keys = common.all_keys(cfgs)
    c = copy.deepcopy(cfgs)
    c["serving"]["replay"]["max_events"] = 10
    k = common.all_keys(c)
    assert {n for n in keys if k[n] != keys[n]} == {"export"}
    assert common.eval_key(cfgs, with_nofmt=True) != common.eval_key(cfgs) == keys["eval"]
    # rules re-tuning re-keys the engine-severity rules, not the replay
    c = copy.deepcopy(cfgs)
    c["rules"]["alert_rate"] = 0.004
    k = common.all_keys(c)
    assert k["rules_engine"] != keys["rules_engine"] and k["features"] == keys["features"]


def test_m2_paths():
    from aml.paths import DataPaths

    p = DataPaths(Path("/data"))
    assert p.features_root == Path("/data/features/hi_small")
    assert p.features_dir("features-abc") == Path("/data/features/hi_small/features-abc")
    assert p.serving_dir == Path("/data/models/serving")
    assert p.model_dir("rules_engine", "k") == Path("/data/models/rules_engine/k")


def test_package_data_files(tmp_path, monkeypatch):
    common = _common()
    pkg = tmp_path / "aml"
    for rel in (
        "a.py",
        "rules/scenarios.sql",
        "explain/templates/case.md.j2",
        "__pycache__/x.pyc",
        "rules/__pycache__/y.sql",
        ".hidden",
        "data/.tmp-1.sql",
    ):
        (pkg / rel).parent.mkdir(parents=True, exist_ok=True)
        (pkg / rel).write_text("x", encoding="utf-8")
    monkeypatch.setattr(common, "AML_SRC", pkg)
    assert common.package_data_files() == ["explain/templates/case.md.j2", "rules/scenarios.sql"]


def test_real_package_data_files_resolve_locally():
    common = _common()
    files = common.package_data_files()
    assert all("/" in f or "." in f for f in files)
    assert common.check_package_files(files)["n_files"] == len(files)
    with pytest.raises(AssertionError):
        common.check_package_files(["does/not/exist.sql"])


def test_require_data_marker(tmp_path):
    common = _common()
    from aml.io import write_json_atomic
    from aml.paths import DataPaths

    paths = DataPaths(tmp_path)
    with pytest.raises(FileNotFoundError):
        common.require_data(paths, "data-abc")
    write_json_atomic({"data_key": "data-old"}, paths.parquet_dir / common.DATA_MARKER)
    with pytest.raises(RuntimeError, match="re-run"):
        common.require_data(paths, "data-abc")
    # A marker written before data versions existed: accepted, version unknown.
    write_json_atomic({"data_key": "data-abc"}, paths.parquet_dir / common.DATA_MARKER)
    assert common.require_data(paths, "data-abc") is None
    write_json_atomic(
        {"data_key": "data-abc", "data_version": "v1"}, paths.parquet_dir / common.DATA_MARKER
    )
    assert common.require_data(paths, "data-abc") == "v1"


def test_data_version_fingerprints_the_prepared_tables(prepared, tmp_path):
    common = _common()
    from aml.paths import DataPaths

    v = common.data_version(prepared)
    assert v == common.data_version(prepared) and len(v) == 16
    copy_paths = DataPaths(tmp_path)
    for name in ("transactions", "accounts", "labels", "fx_rates"):
        dst = getattr(copy_paths, name)
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(getattr(prepared, name), dst)
    assert common.data_version(copy_paths) == v  # content, not location
    with copy_paths.fx_rates.open("a", encoding="utf-8") as f:
        f.write(" ")
    assert common.data_version(copy_paths) != v


def test_stage_outputs_must_match_the_prepared_data(tmp_path):
    """A prepare fix or a changed download keeps the config keys; the version stamp then stops
    evaluation from joining old rule flags / model scores onto the new eval frame."""
    common = _common()
    out = tmp_path / "rules"
    out.mkdir()
    with pytest.raises(FileNotFoundError, match="make rules"):
        common.require_stage_output(out, "rules", "rules", data_version="v2")
    (out / "summary.json").write_text("{}", encoding="utf-8")
    common.require_stage_output(out, "rules", "rules")  # no version known: not checked
    with pytest.raises(RuntimeError, match="re-run `make rules`"):
        common.require_stage_output(out, "rules", "rules", data_version="v2")  # no stamp
    common.stamp_data_version(out, "v1")
    with pytest.raises(RuntimeError, match="re-run `make rules`"):
        common.require_stage_output(out, "rules", "rules", data_version="v2")
    common.stamp_data_version(out, "v2")
    common.require_stage_output(out, "rules", "rules", data_version="v2")


def test_checkpoints_from_other_data_are_reset(tmp_path):
    common = _common()
    out = tmp_path / "lgbm"
    assert common.reset_if_other_data(out, "v1") is False  # fresh directory
    (out / "trials.jsonl").write_text("{}", encoding="utf-8")
    assert common.reset_if_other_data(out, "v1") is False  # same data: keep the checkpoints
    assert (out / "trials.jsonl").exists()
    assert common.reset_if_other_data(out, "v2") is True
    assert not (out / "trials.jsonl").exists()
    assert common.stamped_data_version(out) == (True, "v2")


def test_m1_rules_dir_is_compared_only_when_built_from_this_data(tmp_path):
    """The rules job compares against the M1 SQL directory only if its stamp is the current
    prepared data's (rules_key hashes configs only; a re-prepare keeps the key)."""
    common = _common()
    m1 = tmp_path / "rules"
    assert common.built_from(m1, "v1") is False  # missing
    common.stamp_data_version(m1, "v0")
    assert common.built_from(m1, "v1") is False  # other prepared data
    common.stamp_data_version(m1, "v1")
    assert common.built_from(m1, "v1") is True


def _write_summary(d: Path, **doc) -> None:
    d.mkdir(parents=True, exist_ok=True)
    (d / "summary.json").write_text(json.dumps(doc), encoding="utf-8")


def test_stages_must_come_from_the_current_feature_parts(tmp_path):
    common = _common()
    feats, rules, graph = tmp_path / "features", tmp_path / "rules", tmp_path / "graph"
    stages = {"rules_engine": (rules, "rules"), "lgbm_graph": (graph, "lgbm-graph")}
    _write_summary(feats, spec_hash="s")  # a build summary from before the digest
    _write_summary(rules)
    _write_summary(graph)
    assert common.require_same_feature_table(feats, stages) is None
    _write_summary(feats, spec_hash="s", features_digest="d2")
    _write_summary(rules, features_digest="d2")
    _write_summary(graph, features_digest="d1")  # trained on the parts of an earlier replay
    with pytest.raises(RuntimeError, match="re-run `make lgbm-graph`"):
        common.require_same_feature_table(feats, stages)
    _write_summary(graph)  # no digest at all: not from these parts either
    with pytest.raises(RuntimeError, match="lgbm_graph output"):
        common.require_same_feature_table(feats, stages)
    _write_summary(graph, features_digest="d2")
    assert common.require_same_feature_table(feats, stages) == "d2"


def test_export_needs_a_passed_feature_oracle(tmp_path):
    common = _common()
    feats = tmp_path / "features"
    _write_summary(feats, spec_hash="s", features_digest="d")
    with pytest.raises(FileNotFoundError, match="make features-verify"):
        common.require_features_verified(feats)
    verify = feats / "verify" / "verify.json"
    verify.parent.mkdir()
    good = {"n_mismatches_total": 0, "spec_hash": "s", "features_digest": "d", "rows": 7}
    for bad in ({"n_mismatches_total": 2}, {"spec_hash": "x"}, {"features_digest": "old"}):
        verify.write_text(json.dumps({**good, **bad}), encoding="utf-8")
        with pytest.raises(RuntimeError, match="oracle has not passed"):
            common.require_features_verified(feats)
    verify.write_text(json.dumps(good), encoding="utf-8")
    got = common.require_features_verified(feats)
    assert got["n_mismatches_total"] == 0 and got["rows"] == 7 and got["features_digest"] == "d"


def test_evaluate_checks_every_stage_that_reads_the_parts():
    os.environ.setdefault("MODAL_CONFIG_PATH", NO_MODAL_CONFIG)
    job = importlib.import_module("modal_jobs.evaluate")
    models = {"lgbm_tx": Path("tx"), "lgbm_graph": Path("g"), "lgbm_graph_nofmt": Path("n")}
    assert job.feature_table_stages(Path("r"), models) == {
        "rules_engine": (Path("r"), "rules"),
        "lgbm_graph": (Path("g"), "lgbm-graph"),
        "lgbm_graph_nofmt": (Path("n"), "lgbm-graph"),
    }


def test_jsonable_converts_numpy_and_paths():
    import numpy as np

    common = _common()
    out = common.jsonable(
        {"a": np.float64(1.5), "b": np.arange(2), "c": Path("x"), "d": [np.int32(3)]}
    )
    assert out == {"a": 1.5, "b": [0, 1], "c": "x", "d": [3]}


class _Proc:
    def __init__(self, rc: int, out: str = "", err: str = ""):
        self.returncode, self.stdout, self.stderr = rc, out, err


def test_billing_report_never_raises(monkeypatch):
    os.environ.setdefault("MODAL_CONFIG_PATH", NO_MODAL_CONFIG)
    ev = importlib.import_module("modal_jobs.evaluate")
    calls = []

    def run_ok(cmd, **kw):
        calls.append(cmd)
        return _Proc(0, json.dumps([{"description": "aml-rules", "cost": "0.1"}]))

    monkeypatch.setattr(ev.subprocess, "run", run_ok)
    start_ms = 1_700_000_000_000
    report, err = ev.fetch_billing_report(start_ms)
    assert err is None and report[0]["description"] == "aml-rules"
    # The start day is derived, not written out: no calendar dates in repo files (CLAUDE.md).
    day = ev.datetime.fromtimestamp(start_ms / 1000, tz=ev.UTC).strftime("%Y-%m-%d")
    assert calls[0][calls[0].index("--start") + 1] == day
    assert "--json" in calls[0] and "h" in calls[0]

    monkeypatch.setattr(ev.subprocess, "run", lambda cmd, **kw: _Proc(2, err="boom\nno auth"))
    report, err = ev.fetch_billing_report(None)
    assert report is None and "no auth" in err

    monkeypatch.setattr(ev.subprocess, "run", lambda cmd, **kw: _Proc(0, "not json"))
    assert ev.fetch_billing_report(None)[0] is None

    def boom(cmd, **kw):
        raise OSError("missing")

    monkeypatch.setattr(ev.subprocess, "run", boom)
    assert ev.fetch_billing_report(None) == (None, "billing CLI failed to run: OSError")


# --- repo infrastructure -------------------------------------------------------------------


def test_makefile_targets():
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "export PYTHONIOENCODING := utf-8" in text
    for target in (
        "sync",
        "lint",
        "fmt",
        "test",
        "smoke",
        "data",
        "rules",
        "lgbm",
        "eval",
        "mlflow-ui",
        "pull-reports",
        "clean-local",
    ):
        assert f"\n{target}:" in text, target
    assert "run --detach -m modal_jobs.train_lgbm" in text
    assert "modal_jobs.smoke --gpu $(GPU)" in text


def test_makefile_m2_targets():
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    recipes = {
        "features-bench": "run --detach -m modal_jobs.build_features --mode bench",
        "features": "run --detach -m modal_jobs.build_features --mode full",
        "features-verify": "run --detach -m modal_jobs.build_features --mode verify",
        "rules": "run -m modal_jobs.rules",
        "lgbm-graph": "run --detach -m modal_jobs.train_lgbm --feature-set graph",
        "export": "run -m modal_jobs.export",
        "pull": "volume get --force $(VOLUME) /models/serving .",
    }
    for target, cmd in recipes.items():
        assert f"\n{target}:\n\t$(MODAL) {cmd}\n" in text, target


def test_m2_stage_shapes(imported):
    """Shapes of M2 spec §8: export 4 cores / 16 GiB; build_features 1 core (bench, full) or
    4 cores (verify), never a GPU."""
    if "export" in imported:
        mod = importlib.import_module("modal_jobs.export")
        assert mod.CPU == 4.0 and imported["export"]["app"] == "aml-export"
        for spec in imported["export"]["functions"].values():
            if spec is not None:
                assert spec["cpu"] == 4.0 and spec["memory"][0] == 16384
    if "build_features" in imported:
        assert imported["build_features"]["app"] == "aml-build-features"
        for spec in imported["build_features"]["functions"].values():
            if spec is not None:
                assert spec["cpu"] in (1.0, 4.0) and spec["memory"][0] >= 6144
                assert not spec["gpus"]


def test_gitignore_keeps_secrets_and_data_out_but_packages_in():
    # Reads .gitignore as text, so the test needs no git.
    lines = {
        ln.strip()
        for ln in (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.lstrip().startswith("#")
    }
    required = {
        # secrets
        "API.txt",
        "**/API.txt",
        ".env",
        ".modal.toml",
        "kaggle.json",
        # data and run outputs (root-anchored so src/aml/data, src/aml/serving stay tracked)
        "/data/",
        "/serving/",
        "/models/",
        "/features/",
        "HI-*_Trans.csv",
        "HI-*_Patterns.txt",
        "*.parquet",
        "mlruns/",
        "synthetic_out/",
        "*.part",
        "/*.csv",
        # python
        ".venv/",
        "__pycache__/",
        # only the README is published; local agent files never are
        "*.md",
        "!/README.md",
        "CLAUDE.md",
        ".claude/",
    }
    assert required <= lines, sorted(required - lines)
    # Unanchored or broad patterns that would hide source packages, SQL, configs or reports.
    forbidden = {
        "data",
        "data/",
        "serving/",
        "models/",
        "features/",
        "rules/",
        "eval/",
        "reports/",
        "configs/",
        "*.sql",
        "*.py",
        "*.yaml",
        "*.txt",
        "*.json",
        "uv.lock",
    }
    assert not (lines & forbidden), sorted(lines & forbidden)
