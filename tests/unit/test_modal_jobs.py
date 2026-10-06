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
# M3 (GNN) stage modules, checked the same way once they exist.
M3_STAGES = tuple(
    s
    for s in ("gnn_bench", "hpo_gnn", "train_gnn")
    if (REPO_ROOT / "modal_jobs" / f"{s}.py").exists()
)
# M6 stage modules, checked the same way once they exist.
M6_STAGES = tuple(s for s in ("case_eval",) if (REPO_ROOT / "modal_jobs" / f"{s}.py").exists())
# The only functions with a static GPU (M3 spec §12.2): the smoke test and the GNN workers.
GPU_FUNCTIONS = {
    ("smoke", "gpu_check"),
    ("gnn_bench", "bench_gpu"),
    ("hpo_gnn", "hpo_gpu"),
    ("train_gnn", "train_gpu"),
}
NO_MODAL_CONFIG = str(Path(tempfile.gettempdir()) / "aml-tests-no-modal.toml")
# The M1 run keys of the current configs: M1 outputs on the Volume live under them (the reported
# results used rules-851aa23fb506 and lgbm_tx-875618a5b3f2). M2 must not re-key them (M2 spec §0);
# update only on a deliberate M1 config change.
M1_KEYS = {
    "data": "data-92dd21b72722",
    "rules": "rules-851aa23fb506",
    "lgbm_tx": "lgbm_tx-875618a5b3f2",
}
# The M2 run keys of the current configs (the reported M2 results used features-1e538c068454,
# rules_engine-1cdefaafac26 and lgbm_graph-abd44403c57a). M3 must not re-key them (M3 spec §0).
M2_KEYS = {
    "rules_engine": "rules_engine-1cdefaafac26",
    "features": "features-1e538c068454",
    "lgbm_graph": "lgbm_graph-abd44403c57a",
    "eval": "eval-603c659874ad",
    "export": "export-6109de3b6779",
}
M2_EVAL_NOFMT_KEY = "eval-26f3ddcb870b"

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


# The lead copies gnn_bench's decision (sampler caps, runtime shape) into configs/gnn.yaml after
# `make gnn-bench`. The tests below pin the pre-bench baseline so they don't depend on that copy.
_PRE_BENCH_DECIDED = {
    "sampler": {"batch_size": 2048, "max_edges_per_step": None, "max_edges_per_eval_step": None},
    "runtime": {"gpu": "L4", "cpu": 8, "memory_mib": 32768, "num_workers": 7},
    "faithful_batch_size": 8192,
}


@pytest.fixture(autouse=True)
def _pin_pre_bench_decided(monkeypatch):
    common = _common()
    real = common.load_all_configs

    def pinned(*args, **kwargs):
        cfgs = real(*args, **kwargs)
        g = cfgs.get("gnn")
        if g is not None:
            g["sampler"].update(_PRE_BENCH_DECIDED["sampler"])
            g["runtime"].update(_PRE_BENCH_DECIDED["runtime"])
            g["protocols"]["faithful"]["batch_size"] = _PRE_BENCH_DECIDED["faithful_batch_size"]
        return cfgs

    monkeypatch.setattr(common, "load_all_configs", pinned)
    for name, mod in list(sys.modules.items()):
        if name.startswith("modal_jobs.") and getattr(mod, "load_all_configs", None) is real:
            monkeypatch.setattr(mod, "load_all_configs", pinned)


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
        [sys.executable, "-c", _IMPORT_SCRIPT, *STAGES, *M2_STAGES, *M3_STAGES, *M6_STAGES],
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
    assert set(imported) == set(STAGES) | set(M2_STAGES) | set(M3_STAGES) | set(M6_STAGES)
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
            # The GPU is for the smoke test and the GNN workers only (static L4; T4 per call).
            if (stage, fname) in GPU_FUNCTIONS:
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
    assert set(cfgs) == {"data", "rules", "lgbm", "features", "serving", "gnn"}
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


def test_export_key_ignores_m5_settings():
    """Only serving.replay feeds the export key: the M5 demo settings (Kafka, scorer, replayer,
    latency plan, targets) never re-key or rebuild the bundle, and the pinned M2 key holds."""
    common = _common()
    cfgs = common.load_all_configs()
    assert common.export_inputs(cfgs) == {"replay": {"max_events": 100000}}
    assert common.export_key(cfgs) == M2_KEYS["export"]
    c = copy.deepcopy(cfgs)
    for name in [k for k in c["serving"] if k != "replay"]:
        c["serving"][name] = {"changed": True}
    c["serving"]["new_m5_section"] = {"x": 1}
    assert common.export_key(c) == M2_KEYS["export"]


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
    start_ms = int(ev.datetime.now(ev.UTC).timestamp() * 1000) - 86_400_000  # < 7-day limit
    report, err = ev.fetch_billing_report(start_ms)
    assert err is None and report[0]["description"] == "aml-rules"
    # The start day is derived, not written out: no calendar dates in repo files (project rule).
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


# --- M3 (GNN): keys, presets, local helpers, the contract package -------------------------------

M3_KEY_NAMES = ("gnn_graph", "gnn_bench", "gnn_hpo")
GNN_PARAMS = {  # stands in for train.effective_params(...) of the causal protocol
    "lr": 0.006213,
    "final_dropout": 0.1053,
    "w_pos": 6.275,
    "layer_dropout": 0.0098,
    "hidden": 64,
    "layers": 2,
    "neg_rate": 0.1,
    "conv": "gine",
}
RUN_KINDS = ("run_causal", "run_lookahead", "run_pna", "run_faithful")


def _gnn_key_table(common, cfgs: dict) -> dict[str, str]:
    """Every GNN key kind of one config: the config-only keys, one run per protocol, a set."""
    out = dict(common.gnn_keys(cfgs))
    for kind in RUN_KINDS:
        out[kind] = common.gnn_run_key(cfgs, kind.removeprefix("run_"), 0, GNN_PARAMS)
    out["set_causal"] = common.gnn_set_key(cfgs, "causal", [out["run_causal"]])
    return out


def test_m1_and_m2_run_keys_are_unchanged_by_m3():
    common = _common()
    cfgs = common.load_all_configs()
    keys = common.all_keys(cfgs)
    assert keys == {**M1_KEYS, **M2_KEYS}
    assert common.eval_key(cfgs, with_nofmt=True) == M2_EVAL_NOFMT_KEY
    # eval_key without GNN models is M2's key, byte for byte
    for none in (None, {}):
        assert common.eval_key(cfgs, gnn_keys=none) == M2_KEYS["eval"]
        assert common.eval_key(cfgs, True, gnn_keys=none) == M2_EVAL_NOFMT_KEY


def test_eval_key_with_gnn_models():
    common = _common()
    cfgs = common.load_all_configs()
    a = common.eval_key(cfgs, gnn_keys={"gnn_causal": "gnn_causal-1", "gnn_pna": "gnn_pna-2"})
    b = common.eval_key(cfgs, gnn_keys={"gnn_pna": "gnn_pna-2", "gnn_causal": "gnn_causal-1"})
    assert a == b and a.startswith("eval-") and a != M2_KEYS["eval"]
    assert common.eval_key(cfgs, gnn_keys={"gnn_causal": "gnn_causal-3"}) != a
    one = {"gnn_causal": "gnn_causal-1"}
    assert common.eval_key(cfgs, True, gnn_keys=one) != common.eval_key(cfgs, gnn_keys=one)


def test_gnn_keys_are_stable_and_prefixed():
    common = _common()
    cfgs = common.load_all_configs()
    table = _gnn_key_table(common, cfgs)
    assert table == _gnn_key_table(common, copy.deepcopy(cfgs))
    assert set(common.gnn_keys(cfgs)) == set(M3_KEY_NAMES)
    for name in M3_KEY_NAMES:
        assert table[name].startswith(f"{name}-"), name
    assert all(table[k].startswith("gnn-") for k in RUN_KINDS)
    assert len({table[k] for k in RUN_KINDS}) == len(RUN_KINDS)
    assert table["set_causal"].startswith("gnn_causal-")
    assert set(common.all_keys(cfgs)) == set(M1_KEYS) | set(M2_KEYS)  # GNN keys not in all_keys


RUNS = set(RUN_KINDS) | {"set_causal"}


@pytest.mark.parametrize(
    "edit, changed",
    [
        # the graph encoding re-keys every GNN stage
        (lambda g: g["graph"].update(std_floor=1e-5), {"gnn_graph", "gnn_bench", "gnn_hpo"} | RUNS),
        (lambda g: g["sampler"].update(fanout=[20, 10]), {"gnn_bench", "gnn_hpo"} | RUNS),
        # values the bench decides never re-key the bench
        (lambda g: g["sampler"].update(batch_size=4096), {"gnn_hpo"} | RUNS),
        (lambda g: g["sampler"].update(max_edges_per_step=10**6), {"gnn_hpo"} | RUNS),
        (lambda g: g["sampler"].update(max_edges_per_eval_step=4 * 10**6), {"gnn_hpo"} | RUNS),
        (lambda g: g["protocols"]["faithful"].update(batch_size=4096), {"run_faithful"}),
        (lambda g: g["model"].update(hidden=66), {"gnn_bench", "gnn_hpo"} | RUNS),
        (lambda g: g["train"].update(lr=0.01), {"gnn_hpo"} | RUNS),
        (lambda g: g["train"].update(neg_rate=0.2), {"gnn_bench", "gnn_hpo"} | RUNS),
        (lambda g: g["hpo"].update(n_trials=4), {"gnn_hpo"}),
        (lambda g: g["protocols"]["pna"].update(lr=0.001), {"run_pna"}),
        (lambda g: g["protocols"]["pna"].update(hidden=25), {"gnn_bench", "run_pna"}),
        (
            lambda g: g["protocols"]["faithful"].update(fanout=[50, 50]),
            {"gnn_bench", "run_faithful"},
        ),
        (lambda g: g["protocols"]["faithful"].update(epoch_cap=80), {"run_faithful"}),
        (lambda g: g["bench"].update(timed_steps=20), {"gnn_bench"}),
        # seed lists, hardware, money and the report never re-key a run
        (lambda g: g["protocols"]["causal"].update(seeds=[0, 1, 2]), set()),
        (lambda g: g["protocols"]["lookahead"].update(seeds=[0]), set()),
        (lambda g: g["runtime"].update(num_workers=3, gpu="T4", cpu=4, memory_mib=16384), set()),
        (lambda g: g["budget"].update(m3_cap_usd=15.0, overhead=1.5), set()),
        (lambda g: g["report"].update(published_f1=60.0), set()),
    ],
)
def test_gnn_keys_track_exactly_what_they_hash(edit, changed):
    common = _common()
    cfgs = common.load_all_configs()
    base, m12 = _gnn_key_table(common, cfgs), common.all_keys(cfgs)
    c = copy.deepcopy(cfgs)
    edit(c["gnn"])
    got = _gnn_key_table(common, c)
    assert {k for k in base if got[k] != base[k]} == changed
    assert common.all_keys(c) == m12  # no GNN edit touches an M1/M2 key


def test_gnn_keys_track_gnn_version_and_the_replay(monkeypatch):
    common = _common()
    import aml.models.gnn as gnn

    cfgs = common.load_all_configs()
    base = _gnn_key_table(common, cfgs)
    monkeypatch.setattr(gnn, "GNN_VERSION", gnn.GNN_VERSION + 1)
    bumped = _gnn_key_table(common, cfgs)
    assert all(bumped[k] != base[k] for k in base)
    assert common.all_keys(cfgs) == {**M1_KEYS, **M2_KEYS}
    monkeypatch.undo()
    c = copy.deepcopy(cfgs)
    c["features"]["windows"]["long"] = 2880  # a feature replay change re-keys the graph
    assert all(v != base[k] for k, v in _gnn_key_table(common, c).items())


def test_gnn_run_and_set_keys():
    common = _common()
    cfgs = common.load_all_configs()
    k0 = common.gnn_run_key(cfgs, "causal", 0, GNN_PARAMS)
    assert common.gnn_run_key(cfgs, "causal", 1, GNN_PARAMS) != k0
    assert common.gnn_run_key(cfgs, "lookahead", 0, GNN_PARAMS) != k0
    assert common.gnn_run_key(cfgs, "causal", 0, {**GNN_PARAMS, "lr": 0.01}) != k0
    dev = common.gnn_run_key(cfgs, "causal", 0, GNN_PARAMS, dev=True, max_epochs=2)
    assert dev.startswith("gnn_dev-") and dev != k0
    assert common.gnn_run_key(cfgs, "causal", 0, GNN_PARAMS, dev=True) != dev
    assert common.gnn_run_key(cfgs, "causal", 0, GNN_PARAMS, max_epochs=2) != k0
    assert common.gnn_run_key(cfgs, "causal", "0", GNN_PARAMS) == k0  # seeds hash as ints
    k1 = common.gnn_run_key(cfgs, "causal", 1, GNN_PARAMS)
    s = common.gnn_set_key(cfgs, "causal", [k0, k1])
    assert s.startswith("gnn_causal-") and s != common.gnn_set_key(cfgs, "causal", [k1, k0])
    assert common.gnn_set_key(cfgs, "lookahead", [k0, k1]).startswith("gnn_lookahead-")
    with pytest.raises(ValueError):
        common.gnn_set_key(cfgs, "causal", [])
    with pytest.raises(KeyError):
        common.gnn_run_key(cfgs, "nope", 0, GNN_PARAMS)


def test_gpu_job_preset():
    common = _common()
    kw = common.gpu_job()
    assert kw["image"] is common.gpu_image and kw["gpu"] == "L4"
    assert kw["cpu"] == (8.0, 8.0) and kw["memory"] == (32768, 32768 + 8192)
    assert kw["timeout"] == 3600 and kw["max_containers"] == 1 and kw["scaledown_window"] == 2
    assert set(kw["volumes"]) == {"/data"} and "retries" not in kw
    assert kw["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert kw["env"]["OMP_NUM_THREADS"] == "8" and kw["env"]["POLARS_MAX_THREADS"] == "8"
    assert not {"min_containers", "buffer_containers"} & set(kw)
    kw = common.gpu_job(gpu="T4", cpu=4, memory_mib=16384, timeout=5400, retries=True)
    assert kw["gpu"] == "T4" and kw["cpu"] == (4.0, 4.0) and kw["memory"] == (16384, 24576)
    assert kw["timeout"] == 5400 and kw["env"]["OMP_NUM_THREADS"] == "4"
    r = kw["retries"]
    assert r.max_retries == 2 and r.initial_delay.total_seconds() == 0.0
    # drivers stay on the CPU preset
    drv = common.cpu_job(cpu=0.25, memory_mib=1024, timeout=10800)
    assert drv["cpu"] == 0.25 and drv["memory"] == (1024, 5120) and "gpu" not in drv
    assert drv["env"]["OMP_NUM_THREADS"] == "1"


_NO_TORCH_SCRIPT = r"""
import importlib, socket, sys
for name in ("torch", "torch_geometric", "pyg_lib"):
    sys.modules[name] = None  # any import of them raises ImportError
def _guard(host, *a, **k):
    if host not in ("localhost", "127.0.0.1", "::1", None):
        raise RuntimeError(f"network lookup during import: {host}")
    return _orig(host, *a, **k)
_orig = socket.getaddrinfo
socket.getaddrinfo = _guard
import aml.models.gnn, aml.models.gnn.costplan, modal_jobs.common
for name in sys.argv[1:]:
    importlib.import_module(f"modal_jobs.{name}")
print("IMPORTED_OK")
"""


def test_gnn_stage_modules_and_the_contract_import_without_torch():
    """aml.models.gnn, its costplan and every M3 stage module import with torch blocked (the
    laptop and the CPU drivers have no torch; Linux CI has it, so it is blocked here)."""
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("MODAL_TOKEN", "MODAL_PROFILE"))
    }
    env.update(
        MODAL_CONFIG_PATH=NO_MODAL_CONFIG,
        MODAL_SERVER_URL="https://modal-server.invalid",
        PYTHONIOENCODING="utf-8",
    )
    proc = subprocess.run(
        [sys.executable, "-c", _NO_TORCH_SCRIPT, *M3_STAGES],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert proc.returncode == 0 and "IMPORTED_OK" in proc.stdout, proc.stderr[-3000:]


def test_gnn_constants_agree_across_modules():
    common = _common()
    import aml.models.gnn as gnn
    from aml.models.gnn import costplan
    from aml.models.lgbm import score_column

    assert common.GNN_APPS == costplan.GNN_APPS
    assert common.GNN_APPS == ("aml-gnn-bench", "aml-hpo-gnn", "aml-train-gnn")
    # review MONEY-7: the M3 cap also counts the dev apps (test runner, GPU smoke test)
    assert common.DEV_APPS == costplan.DEV_APPS == ("aml-linux-runner", "aml-smoke")
    assert common.M3_APPS == costplan.M3_APPS == common.GNN_APPS + common.DEV_APPS
    assert common.BENCH_DECIDED == gnn.DECIDED_SAMPLER
    assert set(common.GNN_UNKEYED) == {"runtime", "budget", "report", "bench"}
    assert all(gnn.score_column(s) == score_column(s) for s in (0, 3, 12))
    assert gnn.COMPARISON_MODELS == (
        "gnn_causal",
        "gnn_lookahead",
        "gnn_lookahead_d10",
        "gnn_pna",
    )
    assert gnn.FAITHFUL_MODEL not in gnn.COMPARISON_MODELS
    assert gnn.FAITHFUL_EXEMPT_FEATURES == ("timestamp",)
    assert gnn.FAITHFUL_EXEMPT_NORM == "per_snapshot"
    assert gnn.TO == ("acct", "to", "acct") and gnn.REV == ("acct", "rev_to", "acct")
    assert gnn.GUARD_FIELDS == (
        "edges_checked",
        "violations",
        "target_hits",
        "max_slack",
        "future_edges",
        "dropped_target_copies",
    )
    groups = set(costplan.CONTAINER_GROUPS["L4"]) | set(costplan.CONTAINER_GROUPS["rerun"])
    assert groups == set(costplan.CELL_GROUPS)
    assert costplan.cell_id("T4", 8, "grid", 4096, 3) == "T4-8c-grid-b4096-w3"
    assert costplan.cell_id("L4", 8, "build") == "L4-8c-build"


def test_m3_paths_and_file_names():
    from aml.models import gnn
    from aml.paths import DataPaths

    p = DataPaths(Path("/data"))
    assert p.optuna_dir == Path("/data/optuna")
    assert p.gnn_optuna_log("gnn_hpo-abc") == Path("/data/optuna/gnn_hpo-abc.jsonl")
    assert p.gnn_run_dir("gnn-abc") == Path("/data/models/gnn/gnn-abc")
    for kind in ("gnn_bench", "gnn_hpo", "gnn_causal", "gnn_lookahead_d10", "gnn_faithful"):
        assert p.gnn_set_dir(kind, "k") == Path(f"/data/models/{kind}/k")
    for bad in ("gnn", "lgbm_graph", "gnn_other"):
        with pytest.raises(ValueError):
            p.gnn_set_dir(bad, "k")
    assert gnn.scores_file(2) == "scores_s2.parquet"
    assert gnn.scores_file(2, "d10") == "scores_d10_s2.parquet"
    with pytest.raises(ValueError):
        gnn.scores_file(0, "late")
    assert gnn.trial_dir_name(3) == "trial_3"
    assert gnn.set_kind("lookahead") == "gnn_lookahead"
    assert gnn.set_kind("causal", dev=True) == "gnn_dev"
    assert gnn.label_splits_for("causal") == ("train", "val_early")
    assert gnn.label_splits_for("faithful") == ("train", "val_early", "val_late")
    with pytest.raises(ValueError):
        gnn.set_kind("dev")


def test_gnn_yaml_passes_its_validator():
    from aml.models.gnn import check_gnn_cfg

    common = _common()
    g = common.load_all_configs()["gnn"]
    assert check_gnn_cfg(g) is g
    assert g["protocols"]["causal"]["seeds"] == [0, 1, 2, 3, 4]
    assert g["protocols"]["lookahead"]["test_bounds"] == ["end", "d10"]
    assert g["report"]["reproduced_band"] == [62.35, 67.23]


@pytest.mark.parametrize(
    "edit, where",
    [
        (lambda g: g["train"].update(min_epochs=40), "max_epochs 30 < min_epochs 40"),
        (lambda g: g["protocols"]["causal"].update(seeds=[0, 1, 1]), "protocols.causal.seeds"),
        (lambda g: g["protocols"]["lookahead"].update(test_bounds=["end"]), "test_bounds"),
        (lambda g: g["protocols"]["faithful"].update(epoch_cap=0), "epoch_cap"),
        (lambda g: g["protocols"]["faithful"].update(epoch_cap=101), "epoch_cap"),
        (lambda g: g["protocols"]["faithful"].update(norm="global"), "faithful.norm"),
        (lambda g: g["protocols"]["pna"].update(hidden=21), "divisible by towers"),
        (lambda g: g["report"].update(reproduced_band=[62.0, 67.0]), "reproduced_band"),
        (lambda g: g["report"].pop("winner_metrics"), "report: missing keys"),
        (lambda g: g["sampler"].update(fanout=[25, 10, 5]), "one entry per layer"),
        (lambda g: g["sampler"].update(temporal_strategy="uniform"), "temporal_strategy"),
        (lambda g: g["sampler"].update(batch_size=0), "sampler.batch_size"),
        (lambda g: g["sampler"].update(max_edges_per_step=1.5), "max_edges_per_step"),
        (lambda g: g["train"].update(lr=0.05), "trial 0"),
        (lambda g: g["hpo"]["space"]["lr"].update(low=0.0), "log-uniform"),
        (lambda g: g["graph"].update(norm_split="val_early"), "norm_split"),
        (lambda g: g["graph"].update(log1p=["payment_format"]), "categorical"),
        (lambda g: g["budget"].update(m3_cap_usd=29.5), "m3_cap_usd"),
        (lambda g: g["runtime"].update(gpu="A10"), "runtime.gpu"),
        (lambda g: g["runtime"].update(chunk_wall_s=72000), "runtime.chunk_wall_s"),
        (lambda g: g["runtime"].update(wall_guard_factor=15), "runtime.wall_guard_factor"),
        (
            lambda g: g["protocols"]["faithful"].update(reverse_self_loops=False),
            "faithful.reverse_self_loops",
        ),
        (
            lambda g: g["protocols"]["faithful"].update(edge_features=["timestamp", "amount"]),
            "FAITHFUL_COLUMNS",
        ),
        (lambda g: g["bench"]["faithful"].update(batch_fallback=[4096, 8192]), "decreasing"),
        (lambda g: g.update(extra={}), "unknown sections"),
        (lambda g: g.pop("report"), "missing keys ['report']"),
    ],
)
def test_check_gnn_cfg_rejects(edit, where):
    from aml.models.gnn import check_gnn_cfg

    common = _common()
    g = copy.deepcopy(common.load_all_configs()["gnn"])
    edit(g)
    with pytest.raises(ValueError, match="invalid gnn config") as e:
        check_gnn_cfg(g)
    assert where in str(e.value)


def test_check_gnn_cfg_accepts_the_gate_cuts_and_decided_values():
    """The cuts the cost gate applies by editing gnn.yaml (M3 spec §11.4) and any decided
    values the lead copies from decision.json stay valid."""
    from aml.models.gnn import check_gnn_cfg

    common = _common()
    g = copy.deepcopy(common.load_all_configs()["gnn"])
    g["hpo"]["n_trials"] = 4
    g["protocols"]["lookahead"]["seeds"] = [0]
    g["protocols"]["faithful"]["epoch_cap"] = 80
    g["sampler"].update(batch_size=4096, max_edges_per_step=2_400_000, max_edges_per_eval_step=None)
    g["protocols"]["faithful"]["batch_size"] = 4096
    g["runtime"].update(gpu="T4", cpu=4, memory_mib=16384, num_workers=3)
    assert check_gnn_cfg(g) is g


def test_decided_values_and_mismatches():
    from aml.models.gnn import decided_values, decision_mismatches

    common = _common()
    g = copy.deepcopy(common.load_all_configs()["gnn"])
    want = decided_values(g)
    assert want == {
        "gpu": "L4",
        "cores": 8,
        "memory_mib": 32768,
        "num_workers": 7,
        "batch_size": 2048,
        "max_edges_per_step": None,
        "max_edges_per_eval_step": None,
        "faithful_batch_size": 8192,
    }
    decision = {k: v for k, v in want.items() if k != "faithful_batch_size"}
    decision.update(cores=8.0, faithful={"gpu": "L4", "batch_size": 8192}, bit_deterministic=True)
    assert decision_mismatches(g, decision) == []
    assert len(decision_mismatches(g, {**decision, "num_workers": True})) == 1  # bool != int
    assert len(decision_mismatches(g, {**decision, "max_edges_per_step": 10**6})) == 1
    decision.update(batch_size=4096, faithful={"gpu": "L4", "batch_size": 4096})
    bad = decision_mismatches(g, decision)
    assert len(bad) == 2 and bad[0].startswith("batch_size") and "faithful_batch_size" in bad[1]
    assert len(decision_mismatches(g, {})) == 6  # the two null edge caps match a missing value


def test_report_hash_tracks_only_the_report():
    from aml.models.gnn import report_hash

    common = _common()
    g = copy.deepcopy(common.load_all_configs()["gnn"])
    h = report_hash(g)
    g["train"]["lr"] = 0.01
    assert report_hash(g) == h
    g["report"]["winner_metrics"] = ["literature.pr_auc", "a.recall"]
    assert report_hash(g) != h


def test_guard_counters():
    from aml.models import gnn

    total = gnn.empty_guard()
    assert total["max_slack"] == gnn.NO_SLACK and total["edges_checked"] == 0
    total = gnn.add_guard(total, [10, 0, 0, -3, 2, 1])
    total = gnn.add_guard(total, dict(zip(gnn.GUARD_FIELDS, [5, 0, 0, -1, 0, 2], strict=True)))
    total = gnn.add_guard(total, [0, 0, 0, gnn.NO_SLACK, 0, 0])  # a batch without edges
    assert total == {
        "edges_checked": 15,
        "violations": 0,
        "target_hits": 0,
        "max_slack": -1,
        "future_edges": 2,
        "dropped_target_copies": 3,
    }
    assert gnn.add_guard(None, [1, 0, 0, 0, 0, 0])["edges_checked"] == 1
    with pytest.raises(ValueError):
        gnn.add_guard(total, [1, 2, 3])
    assert gnn.guard_is_clean(total, "causal") and gnn.guard_is_clean(total, "lookahead")
    assert not gnn.guard_is_clean({**total, "violations": 1}, "lookahead")
    assert not gnn.guard_is_clean({**total, "target_hits": 1}, "causal")
    assert gnn.guard_is_clean({**total, "target_hits": 4}, "faithful")


def test_errors_keep_their_detail_through_pickle():
    import pickle

    from aml.models.gnn import GnnStopError, LeakError

    e = pickle.loads(pickle.dumps(LeakError("future edge", {"violations": 1, "seed_pos": [3]})))
    assert isinstance(e, RuntimeError) and str(e) == "future edge"
    assert e.detail == {"violations": 1, "seed_pos": [3]}
    s = pickle.loads(pickle.dumps(GnnStopError("over budget")))
    assert s.detail == {} and str(s) == "over budget"


def test_jsonl_append_survives_a_torn_line(tmp_path):
    from aml.models.gnn import append_jsonl, read_jsonl

    path = tmp_path / "log" / "trials.jsonl"
    assert read_jsonl(path) == []
    append_jsonl(path, {"number": 0, "value": 0.5})
    append_jsonl(path, {"number": 1, "value": None})
    with open(path, "ab") as f:
        f.write(b'{"number": 2, "val')  # a crash mid-append
    assert [r["number"] for r in read_jsonl(path)] == [0, 1]
    append_jsonl(path, {"number": 2, "value": 0.7})  # the torn tail is cut first
    assert read_jsonl(path) == [
        {"number": 0, "value": 0.5},
        {"number": 1, "value": None},
        {"number": 2, "value": 0.7},
    ]
    path.write_text('{"a": 1}\nnot json\n{"a": 2}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="malformed"):
        read_jsonl(path)


class _FakeVol:
    def __init__(self, files: dict[str, bytes]):
        self.files, self.calls = files, []

    def read_file(self, path: str):
        self.calls.append(path)
        if path not in self.files:
            raise FileNotFoundError(path)
        data = self.files[path]
        yield data[:3]
        yield data[3:]


def test_read_volume_json(monkeypatch):
    common = _common()
    vol = _FakeVol({"/models/gnn_bench/k/decision.json": b'{"gpu": "L4", "batch_size": 2048}'})
    monkeypatch.setattr(common, "vol", vol)
    want = {"gpu": "L4", "batch_size": 2048}
    assert common.read_volume_json("/data/models/gnn_bench/k/decision.json") == want
    assert common.read_volume_json(Path("/data/models/gnn_bench/k/decision.json")) == want
    assert common.read_volume_json("/models/gnn_bench/k/decision.json") == want
    assert common.read_volume_json("/data/models/gnn_bench/k/plan.json") is None
    assert vol.calls[-1] == "/models/gnn_bench/k/plan.json"
    assert common.read_volume_json("/database/x.json") is None
    assert vol.calls[-1] == "/database/x.json"  # only the /data prefix is stripped


def test_read_volume_jsonl(monkeypatch):
    """history.jsonl / trial logs from the laptop: records in order, a torn last line ignored
    (as aml.models.gnn.read_jsonl), a malformed line skipped, None if missing."""
    common = _common()
    lines = b'{"epoch": 0}\n{"epoch": 1}\nnot json\n\n{"epoch": 2}\n{"epo'
    vol = _FakeVol({"/models/gnn/k/history.jsonl": lines})
    monkeypatch.setattr(common, "vol", vol)
    got = common.read_volume_jsonl("/data/models/gnn/k/history.jsonl")
    assert got == [{"epoch": 0}, {"epoch": 1}, {"epoch": 2}]
    assert vol.calls[-1] == "/models/gnn/k/history.jsonl"
    assert common.read_volume_jsonl("/data/models/gnn/k/none.jsonl") is None


def test_fetch_billing_summary_never_raises(monkeypatch):
    common = _common()
    calls = []
    doc = {"metered_cost": "1.23", "billed_cost": "0", "adjustments": {}}

    def run_ok(cmd, **kw):
        calls.append(cmd)
        return _Proc(0, json.dumps(doc))

    monkeypatch.setattr(common.subprocess, "run", run_ok)
    assert common.fetch_billing_summary() == (doc, None)
    assert calls[0][-3:] == ["billing", "summary", "--json"]
    monkeypatch.setattr(common.subprocess, "run", lambda cmd, **kw: _Proc(0, "[1, 2]"))
    out, err = common.fetch_billing_summary()
    assert out is None and "not a JSON object" in err
    monkeypatch.setattr(common.subprocess, "run", lambda cmd, **kw: _Proc(1, err="x\nno auth"))
    assert common.fetch_billing_summary() == (None, "billing CLI exited 1: no auth")

    def boom(cmd, **kw):
        raise subprocess.TimeoutExpired(cmd, 180)

    monkeypatch.setattr(common.subprocess, "run", boom)
    assert common.fetch_billing_summary() == (None, "billing CLI failed to run: TimeoutExpired")


def test_common_billing_report_never_raises(monkeypatch):
    common = _common()
    calls = []

    def run_ok(cmd, **kw):
        calls.append(cmd)
        return _Proc(0, json.dumps([{"description": "aml-train-gnn", "cost": "0.4"}]))

    monkeypatch.setattr(common.subprocess, "run", run_ok)
    start_ms = common.billing_window_start_ms(1)  # within the hourly report's 7-day limit
    report, err = common.fetch_billing_report(start_ms)
    assert err is None and report[0]["description"] == "aml-train-gnn"
    # The start day is derived, not written out: no calendar dates in repo files (project rule).
    day = common.datetime.fromtimestamp(start_ms / 1000, tz=common.UTC).strftime("%Y-%m-%d")
    assert calls[0][calls[0].index("--start") + 1] == day
    assert calls[0][calls[0].index("--resolution") + 1] == "h" and "--json" in calls[0]
    common.fetch_billing_report(start_ms, resolution="d")
    assert calls[1][calls[1].index("--resolution") + 1] == "d"
    monkeypatch.setattr(common.subprocess, "run", lambda cmd, **kw: _Proc(0, "not json"))
    assert common.fetch_billing_report(None) == (None, "billing CLI output was not JSON")
    now_ms = common.billing_window_start_ms(0)
    assert abs(now_ms - common.billing_window_start_ms() - 62 * 86_400_000) < 60_000


def test_long_hourly_billing_report_is_daily_then_hourly(monkeypatch):
    """`modal billing report` refuses hourly reports over 7 days: a longer window is fetched as
    daily rows up to today's midnight plus hourly rows from it, merged without double counting."""
    from datetime import timedelta

    common = _common()
    midnight = common.datetime.now(common.UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    before = (midnight - timedelta(days=3)).isoformat()
    today = (midnight + timedelta(hours=1)).isoformat()
    calls = []

    def run(cmd, **kw):
        calls.append(cmd)
        res = cmd[cmd.index("--resolution") + 1]
        rows = (
            [
                {"description": "aml-gnn-bench", "interval_start": before, "cost": "0.2"},
                {"description": "aml-gnn-bench", "interval_start": today, "cost": "9"},
            ]
            if res == "d"
            else [
                {"description": "aml-train-gnn", "interval_start": today, "cost": "0.4"},
                {"description": "aml-train-gnn", "interval_start": before, "cost": "9"},
            ]
        )
        return _Proc(0, json.dumps(rows))

    monkeypatch.setattr(common.subprocess, "run", run)
    rows, err = common.fetch_billing_report(common.billing_window_start_ms())
    assert err is None
    res = [c[c.index("--resolution") + 1] for c in calls]
    assert res[-1] == "h" and set(res[:-1]) == {"d"} and len(res) >= 3  # 62 days: >= 2 pieces
    daily = [(c[c.index("--start") + 1], c[c.index("--end") + 1]) for c in calls[:-1]]
    fmt = "%Y-%m-%d"
    spans = [(common.datetime.strptime(a, fmt), common.datetime.strptime(b, fmt)) for a, b in daily]
    assert all((b - a).days <= 31 for a, b in spans)  # Modal's daily limit
    assert all(spans[i][1] == spans[i + 1][0] for i in range(len(spans) - 1))  # contiguous
    assert daily[-1][1] == midnight.strftime(fmt)
    assert calls[-1][calls[-1].index("--start") + 1] == midnight.strftime(fmt)
    assert sorted(float(r["cost"]) for r in rows) == [0.2, 0.4]  # each row counted once
    monkeypatch.setattr(common.subprocess, "run", lambda cmd, **kw: _Proc(2, err="no auth"))
    assert common.fetch_billing_report(common.billing_window_start_ms())[0] is None


def test_makefile_m3_targets():
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    train = "run --detach -m modal_jobs.train_gnn"
    recipes = {
        "gnn-bench": "run --detach -m modal_jobs.gnn_bench",
        "gnn-plan": "run -m modal_jobs.train_gnn --plan-only",
        "gnn-dev": f"{train} --protocol causal --seeds 0 --max-epochs 2 --dev",
        "gnn-hpo": "run --detach -m modal_jobs.hpo_gnn",
        "gnn": f"{train} --protocol causal --final",
        "gnn-lookahead": f"{train} --protocol lookahead --final$(if $(SEEDS), --seeds $(SEEDS))",
        "gnn-faithful": f"{train} --protocol faithful --final",
        "gnn-pna": f"{train} --protocol pna --final",
        "eval-gnn": "run -m modal_jobs.evaluate --with-gnn",
    }
    phony = {
        name
        for line in text.splitlines()
        if line.startswith(".PHONY:")
        for name in line.removeprefix(".PHONY:").split()
    }
    for target, cmd in recipes.items():
        assert f"\n{target}:\n\t$(MODAL) {cmd}\n" in text, target
        assert target in phony, target
    assert "\nSEEDS ?=\n" in text and "modal_jobs.smoke --gpu $(GPU)" in text


# --- M3 (GNN) job modules: guardrails, arguments, the driver loop, the gate wiring -------------

GNN_JOBS = ("gnn_bench", "hpo_gnn", "train_gnn")
# (module, worker, driver, worker timeout, driver timeout, worker retries) per M3 spec §12.2.
GNN_JOB_SHAPES = {
    "gnn_bench": ("bench_gpu", "bench_driver", 3600, 10800, False),
    "hpo_gnn": ("hpo_gpu", "hpo_driver", 7200 + 1800, 21600, True),
    "train_gnn": ("train_gpu", "train_driver", 7200 + 1800, 43200, True),
}


def _job(name: str):
    _common()
    return importlib.import_module(f"modal_jobs.{name}")


@pytest.mark.parametrize("stage", GNN_JOBS)
def test_gnn_job_guardrail_kwargs(stage, imported):
    common = _common()
    mod = _job(stage)
    worker, driver, w_timeout, d_timeout, retries = GNN_JOB_SHAPES[stage]
    w, d = mod.WORKER_KW, mod.DRIVER_KW
    assert mod.APP_NAME in common.GNN_APPS
    # GPU worker: static L4, cpu request = limit, memory limit above the request, one container.
    assert w["image"] is common.gpu_image and w["gpu"] == "L4"
    assert w["cpu"] == (8.0, 8.0) and w["memory"][1] > w["memory"][0] == 32768
    assert w["timeout"] == w_timeout and w["max_containers"] == 1 and w["scaledown_window"] == 2
    assert w["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8" and w["env"]["OMP_NUM_THREADS"] == "8"
    if retries:  # only hpo_gpu and train_gpu retry (transient errors; runs resume)
        assert w["retries"].max_retries == 2 and w["retries"].initial_delay.total_seconds() == 0
    else:
        assert "retries" not in w
    # CPU driver: a quarter core, 1 GiB, no GPU, no retries, one container.
    assert d["image"] is common.cpu_image and "gpu" not in d and "retries" not in d
    assert d["cpu"] == 0.25 and d["memory"] == (1024, 5120) and d["timeout"] == d_timeout
    assert d["max_containers"] == 1 and d["scaledown_window"] == 2
    for kw in (w, d):
        assert not {"min_containers", "buffer_containers"} & set(kw)
        assert set(kw["volumes"]) == {"/data"}
    # The decorated functions carry exactly these kwargs (the imported specs).
    fns = imported[stage]["functions"]
    assert set(fns) == {worker, driver}
    if fns[worker] is not None:
        assert fns[worker]["gpus"] == "L4" and fns[worker]["memory"] == list(w["memory"])
    if fns[driver] is not None:
        assert not fns[driver]["gpus"] and fns[driver]["memory"] == list(d["memory"])


def test_train_gnn_arguments():
    job = _job("train_gnn")
    g = _common().load_all_configs()["gnn"]
    assert (
        job.parse_seeds("1,2") == [1, 2] and job.parse_seeds("") == [] and job.parse_seeds(0) == [0]
    )
    with pytest.raises(SystemExit, match="comma-separated"):
        job.parse_seeds("1;2")
    with pytest.raises(SystemExit, match="duplicates"):
        job.parse_seeds("1,1")
    ok = {"final": True, "dev": False, "max_epochs": None}
    assert job.check_args(g, "causal", "", **ok) == ([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    assert job.check_args(g, "lookahead", "2,0", **ok) == ([0, 2], [0, 1, 2])  # set order
    assert job.check_args(g, "faithful", "", **ok) == ([0], [0])
    dev = {"final": False, "dev": True, "max_epochs": 2}
    assert job.check_args(g, "causal", "0", **dev) == ([0], [0])  # a dev set = its seeds
    for kw, msg in (
        ({"final": True, "dev": True, "max_epochs": None}, "mutually exclusive"),
        ({"final": True, "dev": False, "max_epochs": 2}, "mutually exclusive"),
        ({"final": False, "dev": False, "max_epochs": 2}, "belongs to --dev"),
        ({"final": False, "dev": False, "max_epochs": None}, "must be --final"),
    ):
        with pytest.raises(SystemExit, match=msg):
            job.check_args(g, "causal", "", **kw)
    with pytest.raises(SystemExit, match="not in protocols.lookahead"):
        job.check_args(g, "lookahead", "3", **ok)
    with pytest.raises(SystemExit, match="--protocol"):
        job.check_args(g, "dev", "", **ok)
    with pytest.raises(SystemExit, match="--dev runs the causal protocol"):
        job.check_args(g, "lookahead", "0", **dev)
    assert job.jobs_for(g, "lookahead", [0], dev=False) == ["lookahead_s0"]
    assert job.jobs_for(g, "lookahead", [1, 2], dev=False) == ["lookahead_rest"]
    assert job.jobs_for(g, "lookahead", [0, 1, 2], dev=False) == ["lookahead_s0", "lookahead_rest"]
    assert job.jobs_for(g, "causal", [0], dev=True) == ["dev"]
    assert job.jobs_for(g, "pna", [0, 1, 2], dev=False) == ["pna"]


def test_worker_options_follow_the_decision():
    job = _job("train_gnn")
    g = copy.deepcopy(_common().load_all_configs()["gnn"])
    decision = {"faithful": {"gpu": "T4", "batch_size": 8192}}
    opts = job.worker_options(g, decision, protocol="causal", timeout=5400)
    assert opts == {"gpu": "L4", "cpu": (8.0, 8.0), "memory": (32768, 40960), "timeout": 5400}
    assert job.worker_options(g, decision, protocol="faithful", timeout=1)["gpu"] == "T4"
    # review perf-2: faithful keeps the shape its bench cell was measured on, also when the
    # headline sets moved to the 4-core rerun shape
    fd = {"gpu": "L4", "batch_size": 8192, "cores": 8, "num_workers": 7, "memory_mib": 24576}
    rt4 = copy.deepcopy(g)
    rt4["runtime"].update(cpu=4, memory_mib=16384, num_workers=3)
    fo = job.worker_options(rt4, {"faithful": fd}, protocol="faithful", timeout=1)
    assert fo["cpu"] == (8.0, 8.0) and fo["memory"] == (24576, 32768) and "env" not in fo
    run_rt = job.set_runtime(rt4, {"faithful": fd}, protocol="faithful", gpu="L4")
    assert (run_rt["cpu"], run_rt["num_workers"], run_rt["memory_mib"]) == (8, 7, 24576)
    assert job.set_runtime(rt4, {"faithful": fd}, protocol="causal", gpu="L4")["num_workers"] == 3
    g["runtime"].update(cpu=4, memory_mib=16384)
    opts = job.worker_options(g, decision, protocol="causal", timeout=1)
    assert opts["cpu"] == (4.0, 4.0) and opts["memory"] == (16384, 24576)
    assert opts["env"]["OMP_NUM_THREADS"] == "4"
    assert opts["env"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"  # env replaces the static env


class _Clock:
    """drive_chunks reads the clock before and after each worker call: one call = `step` s."""

    def __init__(self, step: float):
        self.t, self.step, self.n = 0.0, step, 0

    def __call__(self) -> float:
        self.n += 1
        if self.n % 2 == 0:
            self.t += self.step
        return self.t


def _drive(job, results, tmp_path, *, max_wall=10_000.0, step=100.0):
    calls = []

    def call(budget_s):
        calls.append(budget_s)
        r = results[len(calls) - 1]
        if isinstance(r, Exception):
            raise r
        return r

    budget = {"attempt_wall_s": 1800.0, "max_set_wall_s": max_wall}
    out = job.drive_chunks(
        call, budget=budget, set_dir=tmp_path, log=lambda m: None, clock=_Clock(step)
    )
    return out, calls


def test_drive_chunks_loop(tmp_path):
    from aml.io import read_json
    from aml.models.gnn import STOPPED_FILE

    job = _job("train_gnn")
    part = [{"status": "partial", "next": {"seed": 0, "epoch": e}} for e in (3, 6)]
    out, calls = _drive(job, [*part, {"status": "done", "summary": {"x": 1}}], tmp_path)
    assert out["status"] == "done" and out["summary"] == {"x": 1}
    assert calls == [1800.0] * 3 and len(out["chunks"]) == 3
    assert out["used_s"] == pytest.approx(300.0) and not (tmp_path / STOPPED_FILE).exists()
    for status in ("failed", "busy", "stopped"):  # never retried by the driver
        out, calls = _drive(job, [{"status": status, "error": "e"}, part[0]], tmp_path)
        assert out["status"] == status and len(calls) == 1
    # A raised error (retries exhausted, a timeout) ends the loop; nothing is retried.
    out, calls = _drive(job, [part[0], TimeoutError("timed out"), part[1]], tmp_path)
    assert out["status"] == "error" and "TimeoutError" in out["error"] and len(calls) == 2
    out, _ = _drive(job, [{"status": "weird"}], tmp_path)
    assert out["status"] == "error" and "weird" in out["error"]
    assert not (tmp_path / STOPPED_FILE).exists()
    # Wall guard: the set's wall reaches the limit -> STOPPED.json, at most one chunk beyond.
    many = [{"status": "partial", "next": {"seed": 0, "epoch": e}} for e in range(10)]
    out, calls = _drive(job, many, tmp_path, max_wall=250.0)
    assert out["status"] == "stopped" and out["reason"] == "wall_guard" and len(calls) == 3
    doc = read_json(tmp_path / STOPPED_FILE)
    assert doc["reason"] == "wall_guard" and doc["used_s"] == pytest.approx(300.0)
    assert doc["last"]["next"] == {"seed": 0, "epoch": 2}
    (tmp_path / STOPPED_FILE).unlink()
    # Two consecutive partial results without progress -> stopped.
    out, calls = _drive(job, [part[0], part[1], part[1], part[1]], tmp_path)
    assert out["status"] == "stopped" and out["reason"] == "no_progress" and len(calls) == 3
    assert read_json(tmp_path / STOPPED_FILE)["reason"] == "no_progress"


class _FakeVolume:
    def __init__(self) -> None:
        self.docs: dict[str, dict] = {}
        self.lines: dict[str, list[dict]] = {}

    def put(self, path, doc: dict) -> None:
        self.docs[Path(path).as_posix()] = doc

    def read(self, path) -> dict | None:
        doc = self.docs.get(Path(path).as_posix())
        return copy.deepcopy(doc)

    def put_lines(self, path, records: list[dict]) -> None:
        self.lines[Path(path).as_posix()] = records

    def read_lines(self, path) -> list[dict] | None:
        return copy.deepcopy(self.lines.get(Path(path).as_posix()))


def _fake_params(gnn_cfg, protocol, best):
    """Stands in for train.effective_params (E): any deterministic dict works for the keys."""
    return {"protocol": protocol, "lr": (best or {}).get("lr", gnn_cfg["train"]["lr"])}


def _gnn_volume(cfgs: dict, *, version: str = "v1") -> tuple[_FakeVolume, dict]:
    """A Volume where bench, HPO and the causal set finished and look-ahead seed 0 trained."""
    from aml.models import gnn

    common = _common()
    job = _job("train_gnn")
    paths = common.data_paths(cfgs["data"])
    keys = {"data": common.data_key(cfgs), **common.gnn_keys(cfgs)}
    v = _FakeVolume()
    v.put(
        paths.parquet_dir / common.DATA_MARKER, {"data_key": keys["data"], "data_version": version}
    )
    bench = paths.gnn_set_dir(gnn.BENCH_KIND, keys["gnn_bench"])
    decision = {
        k: x for k, x in gnn.decided_values(cfgs["gnn"]).items() if k != "faithful_batch_size"
    }
    decision["faithful"] = {
        "gpu": "T4",
        "batch_size": cfgs["gnn"]["protocols"]["faithful"]["batch_size"],
    }
    v.put(bench / gnn.DECISION_FILE, decision)
    v.put(
        bench / gnn.SUMMARY_FILE,
        {
            "data_version": version,
            "counts": {"n_pos": 10},
            "containers": [{"gpu": "L4", "cores": 8, "memory_mib": 32768, "seconds": 600.0}],
        },
    )
    hpo = paths.gnn_set_dir(gnn.HPO_KIND, keys["gnn_hpo"])
    best = {"lr": 0.01, "final_dropout": 0.1, "w_pos": 5.0}
    v.put(hpo / gnn.SUMMARY_FILE, {"data_version": version, "gpu_seconds": 1000.0})
    v.put(hpo / gnn.BEST_PARAMS_FILE, best)
    g = cfgs["gnn"]
    causal = job.set_plan(cfgs, "causal", _fake_params(g, "causal", best), [0, 1, 2, 3, 4])
    v.put(Path(causal["set_dir"]) / gnn.SUMMARY_FILE, {"data_version": version, "final": True})
    la = job.set_plan(cfgs, "lookahead", _fake_params(g, "lookahead", best), [0, 1, 2])
    v.put(paths.gnn_run_dir(la["run_keys"][0]) / gnn.SEED_SUMMARY_FILE, {"data_version": version})
    return v, {"decision": decision, "best": best, "causal": causal, "lookahead": la}


def test_volume_state_reads_keys_and_done_runs():
    job = _job("train_gnn")
    cfgs = _common().load_all_configs()
    v, made = _gnn_volume(cfgs)
    st = job.volume_state(cfgs, read=v.read, params_fn=_fake_params)
    assert st["done"] == ["bench", "causal", "hpo", "lookahead_s0"]
    assert st["data_version"] == "v1" and st["counts"] == {"n_pos": 10}
    assert st["best_params"] == made["best"] and st["decision"] == made["decision"]
    assert st["sets"]["causal"]["set_key"] == made["causal"]["set_key"]
    assert st["sets"]["causal"]["set_key"].startswith("gnn_causal-")
    assert st["sets"]["lookahead"]["run_keys"] == made["lookahead"]["run_keys"]
    assert st["sets"]["faithful"]["set_kind"] == "gnn_faithful"
    assert st["dev"]["set_kind"] == "gnn_dev" and st["dev"]["max_epochs"] == 2
    assert st["dev"]["run_keys"][0].startswith("gnn_dev-")
    assert job.next_job(st) == "dev"
    # Outputs of other prepared data do not count as done.
    st = job.volume_state(
        cfgs, read=_gnn_volume(cfgs, version="v1")[0].read, params_fn=_fake_params
    )
    v2, _ = _gnn_volume(cfgs)
    marker = Path(_common().data_paths(cfgs["data"]).parquet_dir / _common().DATA_MARKER)
    v2.docs[marker.as_posix()]["data_version"] = "v2"
    assert job.volume_state(cfgs, read=v2.read, params_fn=_fake_params)["done"] == []
    # Without a finished HPO its best params are not trusted: no causal / look-ahead keys.
    v3, _ = _gnn_volume(cfgs)
    hpo = [p for p in v3.docs if "/gnn_hpo/" in p and p.endswith("summary.json")]
    del v3.docs[hpo[0]]
    st = job.volume_state(cfgs, read=v3.read, params_fn=_fake_params)
    assert st["best_params"] is None and st["sets"]["causal"] is None
    assert "hpo" not in st["done"] and "causal" not in st["done"]


def test_run_gate_wiring(monkeypatch):
    """The entrypoints' gate: costplan.measured_spend on the billing report and the Volume
    summaries, the metered cost for the pre-check, ONE costplan.gate call per submission (both
    look-ahead rows when one submission trains every look-ahead seed)."""
    from aml.models.gnn import costplan

    job = _job("train_gnn")
    cfgs = _common().load_all_configs()
    v, _ = _gnn_volume(cfgs)
    st = job.volume_state(cfgs, read=v.read, params_fn=_fake_params)
    seen = []

    def fake_gate(plan, *, gnn_cfg, spent_usd, done, job, metered_usd, progress=None):
        seen.append(
            {
                "plan": plan,
                "spent": spent_usd,
                "done": done,
                "job": job,
                "m": metered_usd,
                "progress": progress,
            }
        )
        return {"job": job, "allowed": True, "refuse_reason": None}

    monkeypatch.setattr(costplan, "plan_runs", lambda g, decision=None, counts=None: ["rows"])
    monkeypatch.setattr(costplan, "gate", fake_gate)
    report = lambda start_ms: ([{"description": "aml-train-gnn", "cost": "3"}], None)  # noqa: E731
    summary = lambda: ({"metered_cost": "7.25"}, None)  # noqa: E731
    gate = job.run_gate(
        cfgs, st, ["lookahead_s0", "lookahead_rest"], fetch_report=report, fetch_summary=summary
    )
    assert gate["allowed"] and gate["reasons"] == [] and gate["plan"] == ["rows"]
    (call,) = seen
    assert call["job"] == ["lookahead_s0", "lookahead_rest"] and call["done"] == st["done"]
    assert call["progress"] == {}  # a fake `read` without read_lines: no progress
    sp = gate["spent"]
    # max(billing 3.00, Volume floor) + the dev allowance; metered from the billing summary.
    allowance = cfgs["gnn"]["budget"]["dev_allowance_usd"]
    assert sp["billing_usd"] == pytest.approx(3.0) and call["m"] == 7.25
    assert sp["volume_usd"] < 3.0 and sp["source"] == "billing"
    assert call["spent"] == sp["spent_usd"] == pytest.approx(3.0 + allowance)
    job.run_gate(cfgs, st, ["pna"], spent=sp)
    assert seen[-1]["job"] == "pna"  # a single job is passed as a string
    # Billing CLI down: the Volume floor alone (a lower bound, with a warning).
    down = lambda start_ms: (None, "billing CLI exited 1: no auth")  # noqa: E731
    gate = job.run_gate(cfgs, st, ["pna"], fetch_report=down, fetch_summary=lambda: (None, "x"))
    sp = gate["spent"]
    assert sp["billing_usd"] is None and sp["source"] == "volume" and "unavailable" in sp["warning"]
    assert sp["spent_usd"] == pytest.approx(sp["volume_usd"] + allowance) and seen[-1]["m"] is None
    monkeypatch.undo()
    # The real gate: an unavailable metered cost refuses (the workspace pre-check cannot run).
    gate = job.run_gate(cfgs, st, ["pna"], spent={**sp, "metered_usd": None})
    assert not gate["allowed"] and "metered cost is unavailable" in gate["reasons"][0]
    gate = job.run_gate(cfgs, st, ["pna"], spent={**sp, "metered_usd": 1.0})
    assert gate["result"]["job"] == "pna" and gate["allowed"] is gate["result"]["allowed"]


def test_spend_records_from_volume_summaries():
    job = _job("train_gnn")
    cfgs = _common().load_all_configs()
    v, made = _gnn_volume(cfgs)
    st = job.volume_state(cfgs, read=v.read, params_fn=_fake_params)
    st["sets"]["faithful"]["summary"] = {"gpu_seconds": 50.0}
    recs = job.spend_records(st, cfgs["gnn"])
    assert {"gpu": "L4", "cores": 8, "memory_mib": 32768, "gpu_seconds": 600.0} in recs  # bench
    assert {"gpu": "L4", "cores": 8, "memory_mib": 32768, "gpu_seconds": 1000.0} in recs  # hpo
    assert {"gpu": "T4", "cores": 8, "memory_mib": 32768, "gpu_seconds": 50.0} in recs
    assert len(recs) == 3


def test_spend_floor_counts_failed_calls_and_containers_and_the_billing_lag():
    """review MONEY-8: the Volume floor counts a failed bench's containers (bench.json, failed
    ones too) and an unfinished set's worker calls (calls.jsonl GPU seconds when longer than its
    epochs); the gate adds the recorded GPU seconds after the billing report's last full hour."""
    from aml.models.gnn import costplan

    job = _job("train_gnn")
    cfgs = _common().load_all_configs()
    g = cfgs["gnn"]
    v, _ = _gnn_volume(cfgs)
    st = job.volume_state(cfgs, read=v.read, params_fn=_fake_params)
    st["bench_summary"] = None  # the bench failed: no summary, only bench.json
    shape = {"cores": 8, "memory_mib": 32768}
    st["bench_state"] = {
        "containers": {
            "L4": {**shape, "gpu": "L4", "seconds": 700.0, "status": "done", "ended_at": 1e3},
            "T4": {**shape, "gpu": "T4", "seconds": 300.0, "status": "failed", "ended_at": 5e3},
        }
    }
    fa = st["sets"]["faithful"]
    fa["summary"] = None  # a stopped faithful set: 2 epochs (100 s) but 3 calls (2,000 s)
    st["progress"] = {"faithful": {"units_done": 0, "epochs_done": 2, "seconds": 100.0}}
    st["calls"] = {
        fa["set_dir"]: [
            {"status": "partial", "elapsed_s": 1500.0, "device": "cuda", "ended_at": 4_000.0},
            {"status": "failed", "elapsed_s": 500.0, "device": "cuda", "ended_at": 7_000.0},
            {"status": "failed", "elapsed_s": 9.0, "device": "cpu", "ended_at": 7_100.0},
        ]
    }
    recs = job.spend_records(st, g)
    assert {"gpu": "T4", "cores": 8, "memory_mib": 32768, "gpu_seconds": 300.0} in recs
    assert {"gpu": "L4", "cores": 8, "memory_mib": 32768, "gpu_seconds": 700.0} in recs
    fshape = {"gpu": "T4", "cores": 8, "memory_mib": 32768}  # decision.faithful.gpu
    assert {**fshape, "gpu_seconds": 2000.0} in recs
    calls = job.call_records(st, g)
    assert {**fshape, "elapsed_s": 500.0, "ended_at": 7_000.0} in calls and len(calls) == 4
    # an hourly report whose last interval starts at 01:00 (unix 3,600 s) is billed through
    # 7,200 s: only what ran after that is added
    report = [
        {"description": "aml-train-gnn", "cost": "2.00", "interval_start": "1970-01-01T01:00:00"}
    ]
    assert costplan.billed_through(report) == 7_200.0
    lag = costplan.lag_usd(calls, 7_200.0)
    assert lag == 0.0  # everything ended before 7,200 s
    lag = costplan.lag_usd(calls, 6_700.0)  # the failed call ran 500 s, 300 s of them after
    assert lag == pytest.approx(costplan.usd(costplan.plan_usd_h("T4", 8, 32), 300.0))
    sp = costplan.measured_spend(report, recs, g, calls=calls)
    assert sp["billed_through"] == 7_200.0 and sp["lag_usd"] == 0.0
    late = [{**report[0], "interval_start": "1970-01-01T00:00:00"}]  # billed through 3,600 s
    sp = costplan.measured_spend(late, recs, g, calls=calls)
    after = {"L4": 0.0, "T4": 300.0 + 400.0 + 500.0}  # bench T4 (300 s), faithful 400 + 500 s
    want = costplan.usd(costplan.plan_usd_h("T4", 8, 32), after["T4"])
    assert sp["lag_usd"] == pytest.approx(want)
    assert sp["usd"] == pytest.approx(
        max(2.0 + want, sp["volume_usd"]) + g["budget"]["dev_allowance_usd"]
    )


def _history(n: int, seconds: float = 10.0) -> list[dict]:
    return [{"epoch": e, "seconds": seconds} for e in range(n)]


def test_run_progress_credits_what_started_runs_trained():
    """A re-submission after a stop: what the started runs already trained (finished seeds,
    checkpointed epochs) reaches the gate, so their partial spend is not charged twice. Only run
    dirs of the current prepared data and feature parts count."""
    from aml.features.spec import FEATURES_DIGEST
    from aml.models import gnn
    from aml.models.gnn import costplan

    job = _job("train_gnn")
    common = _common()
    cfgs = common.load_all_configs()
    v, made = _gnn_volume(cfgs)
    paths = common.data_paths(cfgs["data"])
    v.put(paths.features_dir(common.features_key(cfgs)) / "summary.json", {FEATURES_DIGEST: "d1"})
    fp = {"data_version": "v1", "features_digest": "d1"}
    st0 = job.volume_state(cfgs, read=v.read, params_fn=_fake_params)
    la, sets = made["lookahead"], st0["sets"]

    def run_dir(plan, seed):
        return paths.gnn_run_dir(plan["run_keys"][seed])

    # look-ahead seed 1 finished (12 epochs), seed 2 checkpointed 4 epochs
    for seed, n in ((1, 12), (2, 4)):
        v.put_lines(run_dir(la, seed) / gnn.HISTORY_FILE, _history(n))
        v.put(run_dir(la, seed) / gnn.FINGERPRINT_FILE, fp)
    v.put(run_dir(la, 1) / gnn.SEED_SUMMARY_FILE, {**fp, "seconds": 120.0})
    # faithful: other prepared data; PNA: other feature parts -> no credit (a reset follows)
    v.put_lines(run_dir(sets["faithful"], 0) / gnn.HISTORY_FILE, _history(60))
    v.put(run_dir(sets["faithful"], 0) / gnn.FINGERPRINT_FILE, {**fp, "data_version": "v0"})
    v.put_lines(run_dir(sets["pna"], 0) / gnn.HISTORY_FILE, _history(3))
    v.put(run_dir(sets["pna"], 0) / gnn.FINGERPRINT_FILE, {**fp, "features_digest": "d0"})
    # the dev run checkpointed 1 epoch
    dev_dir = paths.gnn_run_dir(st0["dev"]["run_keys"][0])
    v.put_lines(dev_dir / gnn.HISTORY_FILE, _history(1, 30.0))
    v.put(dev_dir / gnn.FINGERPRINT_FILE, fp)

    st = job.volume_state(cfgs, read=v.read, params_fn=_fake_params, read_lines=v.read_lines)
    assert st["progress"] == {
        "dev": {"units_done": 0, "epochs_done": 1, "seconds": 30.0},
        "lookahead_rest": {"units_done": 1, "epochs_done": 4, "seconds": 160.0},
    }
    recs = job.spend_records(st, cfgs["gnn"])  # the Volume floor includes the started runs
    assert {"gpu": "L4", "cores": 8, "memory_mib": 32768, "gpu_seconds": 160.0} in recs
    assert {"gpu": "L4", "cores": 8, "memory_mib": 32768, "gpu_seconds": 30.0} in recs
    # the real gate charges the started look-ahead rest only for what is left
    report = lambda start_ms: ([], None)  # noqa: E731
    summary = lambda: ({"metered_cost": "1.00"}, None)  # noqa: E731
    gate = job.run_gate(cfgs, st, ["lookahead_rest"], fetch_report=report, fetch_summary=summary)
    fresh = job.run_gate(
        cfgs, {**st, "progress": {}}, ["lookahead_rest"], fetch_report=report, fetch_summary=summary
    )
    row = {r["run"]: r for r in gate["result"]["rows"]}["lookahead_rest"]
    full = {r["run"]: r for r in fresh["result"]["rows"]}["lookahead_rest"]
    assert row["done_units"] == 1 and row["done_epochs"] == 4
    assert row["usd"]["conservative"] < full["usd"]["conservative"]
    assert "[started: 1 done + 4 epochs" in costplan.render_gate(gate["result"])

    # HPO under way: 2 trials in the log + 3 epochs of trial 2; a stale log counts nothing
    v3, _ = _gnn_volume(cfgs)
    hpo = [p for p in v3.docs if "/gnn_hpo/" in p and p.endswith("summary.json")]
    del v3.docs[hpo[0]]
    v3.put(paths.features_dir(common.features_key(cfgs)) / "summary.json", {FEATURES_DIGEST: "d1"})
    log = paths.gnn_optuna_log(common.gnn_keys(cfgs)["gnn_hpo"])
    v3.put_lines(log, [{"number": k, "seconds": 50.0, **fp} for k in (0, 1)])
    trial = Path(st0["hpo_dir"]) / gnn.trial_dir_name(2)
    v3.put_lines(trial / gnn.HISTORY_FILE, _history(3))
    v3.put(trial / gnn.FINGERPRINT_FILE, fp)
    st = job.volume_state(cfgs, read=v3.read, params_fn=_fake_params, read_lines=v3.read_lines)
    assert st["progress"]["hpo"] == {"units_done": 2, "epochs_done": 3, "seconds": 130.0}
    v3.put_lines(log, [{"number": 0, "seconds": 50.0, **fp, "data_version": "v0"}])
    st = job.volume_state(cfgs, read=v3.read, params_fn=_fake_params, read_lines=v3.read_lines)
    assert "hpo" not in st["progress"]  # run_hpo would move that log aside and start over


def test_drive_chunks_stops_before_the_driver_deadline(tmp_path):
    """No chunk starts that could outlast the driver's own Modal timeout: the driver writes
    STOPPED.json and returns cleanly (a re-run resumes) instead of being killed mid-chunk."""
    from aml.io import read_json
    from aml.models.gnn import STOPPED_FILE

    job = _job("train_gnn")
    calls = []

    def call(budget_s):
        calls.append(budget_s)
        return {"status": "partial", "next": {"seed": 0, "epoch": len(calls)}}

    budget = {"attempt_wall_s": 1800.0, "max_set_wall_s": 1e9, "timeout_s": 2400}
    out = job.drive_chunks(
        call,
        budget=budget,
        set_dir=tmp_path,
        log=lambda m: None,
        clock=_Clock(1000.0),
        deadline_s=7000.0,
    )
    # review MONEY-5: one call can take 2 full attempts of T (Modal retries a timed-out input;
    # the unclean-start counter ends the third after its load) + the margin: 2 x 2400 + 600 =
    # 5400 s. Chunk 1 at 0 s, chunk 2 at 1000 s (1000 + 5400 <= 7000), not a third at 2000 s.
    assert out["status"] == "stopped" and out["reason"] == "driver_deadline" and len(calls) == 2
    assert read_json(tmp_path / STOPPED_FILE)["reason"] == "driver_deadline"
    # the first chunk always runs, even with a deadline shorter than one chunk
    (tmp_path / STOPPED_FILE).unlink()
    calls.clear()
    out = job.drive_chunks(
        call, budget=budget, set_dir=tmp_path, log=lambda m: None, clock=_Clock(1.0), deadline_s=1
    )
    assert len(calls) == 1 and out["reason"] == "driver_deadline"
    # the drivers pass their own timeout minus the margin
    assert job.DRIVER_KW["timeout"] - job.DRIVER_DEADLINE_MARGIN_S > job.STATIC_WORKER_TIMEOUT
    # chunk_budget ties the attempts to the worker's Retries and the unclean-start counter
    from aml.models.gnn.train import MAX_STARTS

    rows = [{"run_s": {"conservative": 4000.0}, "epoch_s": {"conservative": 30.0}}]
    b = job.chunk_budget(rows, _common().load_all_configs()["gnn"])
    assert b["full_attempts"] == min(_common().WORKER_MAX_RETRIES + 1, MAX_STARTS - 1) == 2


def test_drive_chunks_stops_before_the_charged_wall_limit(tmp_path):
    """review MONEY-1: no chunk starts that could take the set past set_wall_limit_s (factor x
    projection + one chunk wall: what the gate's budget pre-check charged); the first chunk
    always runs."""
    job = _job("train_gnn")
    calls = []

    def call(budget_s):
        calls.append(budget_s)
        return {"status": "partial", "next": {"seed": 0, "epoch": len(calls)}}

    budget = {
        "attempt_wall_s": 1800.0,
        "max_set_wall_s": 1e9,
        "set_wall_limit_s": 5000.0,
        "startup_s": 300.0,
    }
    out = job.drive_chunks(
        call, budget=budget, set_dir=tmp_path, log=lambda m: None, clock=_Clock(1000.0)
    )
    # chunks at 0, 1000, 2000 s (2000 + 1800 + 300 <= 5000); not at 3000 s (5100 > 5000)
    assert out["status"] == "stopped" and out["reason"] == "wall_limit" and len(calls) == 3
    g = _common().load_all_configs()["gnn"]
    rows = [
        {
            "run_s": {"conservative": 4000.0},
            "epoch_s": {"conservative": 30.0},
            "startup_s": {"conservative": 300.0},
        }
    ]
    b = job.chunk_budget(rows, g)
    rt = g["runtime"]
    assert b["set_wall_limit_s"] == pytest.approx(
        rt["wall_guard_factor"] * 4000.0 + rt["chunk_wall_s"]
    )
    assert b["startup_s"] == 300.0


class _NoopVol:
    def commit(self) -> None:
        pass

    def reload(self) -> None:
        pass


@pytest.mark.filterwarnings("ignore:The .* function is executing locally")
def test_drivers_short_circuit_only_a_current_finished_set(tmp_path, monkeypatch):
    """train_driver / hpo_driver return a stored summary without a GPU call only under the
    worker's own rule: current prepared data and feature parts, and final if the call is."""
    from aml.features.spec import FEATURES_DIGEST
    from aml.io import write_json_atomic
    from aml.paths import DataPaths

    tg, hg = _job("train_gnn"), _job("hpo_gnn")
    cfgs = _common().load_all_configs()
    paths = DataPaths(tmp_path, cfgs["data"]["dataset"]["name"])
    for mod in (tg, hg):
        monkeypatch.setattr(mod, "data_paths", lambda data_cfg: paths)
        monkeypatch.setattr(mod, "vol", _NoopVol())
    monkeypatch.setattr(tg, "require_inputs", lambda p, keys: "v1")
    write_json_atomic({"data_version": "v1"}, paths.parquet_dir / "prepare_summary.json")
    write_json_atomic({FEATURES_DIGEST: "d1"}, paths.features_dir("features-x") / "summary.json")
    worker_calls = []

    class Worker:
        def with_options(self, **kw):
            return self

        def remote(self, spec):
            worker_calls.append(spec)
            return {"status": "done", "summary": None}

    monkeypatch.setattr(tg, "train_gpu", Worker())
    monkeypatch.setattr(hg, "hpo_gpu", Worker())
    budget = {"attempt_wall_s": 10.0, "max_set_wall_s": 100.0, "timeout_s": 1800}
    spec = {
        "data_cfg": cfgs["data"],
        "keys": {"features": "features-x", "gnn_hpo": "gnn_hpo-x"},
        "protocol": "causal",
        "dev": False,
        "final": True,
        "set_kind": "gnn_causal",
        "set_key": "gnn_causal-x",
        "seeds": [0],
        "options": {},
        "budget": budget,
    }
    set_dir = paths.gnn_set_dir("gnn_causal", "gnn_causal-x")
    current = {"final": True, "data_version": "v1", FEATURES_DIGEST: "d1"}
    for doc, final, called in (
        (current, True, False),  # finished: no GPU call
        ({**current, FEATURES_DIGEST: "d0"}, True, True),  # other feature parts
        ({**current, "final": False}, True, True),  # a final call needs the test scores
        ({**current, "final": False}, False, False),
    ):
        worker_calls.clear()
        write_json_atomic(doc, set_dir / "summary.json")
        out = tg.train_driver.local({**spec, "final": final})
        assert out["status"] == "done" and bool(worker_calls) is called, (doc, final)
        if not called:
            assert out["summary"] == doc
    hpo_dir = paths.gnn_set_dir("gnn_hpo", "gnn_hpo-x")
    for doc, called in (
        ({"data_version": "v1", FEATURES_DIGEST: "d1"}, False),
        ({"data_version": "v1", FEATURES_DIGEST: "d0"}, True),
    ):
        worker_calls.clear()
        write_json_atomic(doc, hpo_dir / "summary.json")
        hg.hpo_driver.local(spec)
        assert bool(worker_calls) is called, doc


def test_plan_only_runs_the_real_gate_offline(monkeypatch, capsys):
    """`make gnn-plan` / `--plan-only` end to end with the real configs, costplan and job
    wiring on an empty Volume and stubbed billing CLIs: nothing is submitted, the gate table
    matches the plan (M3 spec §11.3: PNA is the only cut), and a failing billing summary
    refuses (the workspace pre-check cannot run)."""
    from aml.models.gnn import costplan

    tg, hg, gb = _job("train_gnn"), _job("hpo_gnn"), _job("gnn_bench")
    monkeypatch.setattr(tg, "read_volume_json", lambda path: None)
    monkeypatch.setattr(tg, "read_volume_jsonl", lambda path: None)
    monkeypatch.setattr(tg, "fetch_billing_report", lambda start_ms: ([], None))
    monkeypatch.setattr(tg, "fetch_billing_summary", lambda: ({"metered_cost": "1.20"}, None))
    drivers = _Remote({"status": "done"})
    for mod, name in ((tg, "train_driver"), (hg, "hpo_driver"), (gb, "bench_driver")):
        monkeypatch.setattr(mod, name, drivers)
    cfgs = _common().load_all_configs()
    plan = costplan.plan_runs(cfgs["gnn"])
    want = cfgs["gnn"]["budget"]["dev_allowance_usd"] + sum(
        r["usd"]["conservative"] for r in plan if r["run"] != "pna"
    )
    tg.main.info.raw_f(plan_only=True)
    out = capsys.readouterr().out
    assert "M3 cost gate for bench: ALLOWED" in out and "cuts: pna" in out
    assert f"projected ${want:.2f} (conservative)" in out
    assert "budget pre-check: metered $1.20 + job" in out and "action: skip the PNA run" in out
    hg.main.info.raw_f(plan_only=True)
    assert "M3 cost gate for hpo: ALLOWED" in capsys.readouterr().out
    gb.main.info.raw_f(plan_only=True)
    assert "M3 cost gate for bench: ALLOWED" in capsys.readouterr().out
    monkeypatch.setattr(tg, "fetch_billing_summary", lambda: (None, "billing CLI exited 1"))
    tg.main.info.raw_f(plan_only=True)
    out = capsys.readouterr().out
    assert "REFUSED" in out and "metered cost is unavailable" in out
    assert drivers.calls == []


def test_budget_rows_follow_what_the_gate_charged():
    """A started run's chunk budget and wall guard are sized from what is left (the gate's
    row), not from the full plan; a trained-but-unscored look-ahead seed 0 (done in the gate)
    falls back to the plan's row."""
    job = _job("train_gnn")
    plan = [{"run": "causal", "run_s": {"conservative": 5000.0}}, {"run": "lookahead_s0"}]
    left = {"run": "causal", "status": "planned", "run_s": {"conservative": 1200.0}}
    gate = {"plan": plan, "result": {"rows": [left, {"run": "lookahead_s0", "status": "done"}]}}
    assert job.budget_rows(gate, ["causal"]) == [left]
    assert job.budget_rows(gate, ["lookahead_s0"]) == [plan[1]]
    assert job.budget_rows({"plan": plan}, ["causal"]) == [plan[0]]


def test_chunk_budget_formula():
    """§12.1 with the real costplan formulas (owner C): attempt wall = min(chunk_wall_s, 1.5 x
    projection); T = max(1800, ceil(attempt + max(600, 3 x conservative epoch)))."""
    job = _job("train_gnn")
    g = _common().load_all_configs()["gnn"]
    rows = [{"run_s": {"conservative": 4000.0}, "epoch_s": {"conservative": 30.0}}]
    b = job.chunk_budget(rows, g)
    assert b["attempt_wall_s"] == pytest.approx(6000.0) and b["timeout_s"] == 6600
    assert b["max_set_wall_s"] == pytest.approx(6000.0)
    b = job.chunk_budget(rows * 2, g, scale=0.5)  # two rows, half the seeds
    assert b["projected_s"] == pytest.approx(4000.0)
    long = [{"run_s": {"conservative": 20000.0}, "epoch_s": {"conservative": 400.0}}]
    b = job.chunk_budget(long, g)
    assert b["attempt_wall_s"] == 7200 and b["timeout_s"] == 7200 + 1200
    short = [{"run_s": {"conservative": 100.0}, "epoch_s": {"conservative": 5.0}}]
    assert job.chunk_budget(short, g)["timeout_s"] == 1800
    with pytest.raises(SystemExit, match="no row"):
        job.chunk_budget([], g)


def _entry_state(cfgs: dict, **over) -> dict:
    """A volume_state result for the entrypoint tests (bench and HPO finished)."""
    from aml.models.gnn import decided_values

    g = cfgs["gnn"]
    decision = {k: v for k, v in decided_values(g).items() if k != "faithful_batch_size"}
    decision["faithful"] = {"gpu": "T4", "batch_size": g["protocols"]["faithful"]["batch_size"]}
    common = _common()
    st = {
        "keys": {"data": "data-x", "features": "features-x", **common.gnn_keys(cfgs)},
        "data_version": "v1",
        "decision": decision,
        "bench_summary": {"data_version": "v1"},
        "hpo_summary": {"data_version": "v1"},
        "best_params": {"lr": 0.01, "final_dropout": 0.1, "w_pos": 5.0},
        "bench_dir": "/data/models/gnn_bench/k",
        "done": ["bench", "hpo"],
        "sets": {},
        "dev": {},
    }
    st.update(over)
    return st


class _Remote:
    def __init__(self, result: dict) -> None:
        self.calls: list[tuple] = []
        self.result = result

    def remote(self, *args):
        self.calls.append(args)
        return self.result


def _patch_entry(monkeypatch, job, state: dict, *, allowed: bool = True, result=None):
    import aml.models.gnn.train as train

    gates = []

    def fake_gate(cfgs, st, jobs, **kw):
        gates.append(list(jobs))
        return {
            "allowed": allowed,
            "reasons": [] if allowed else ["x: refused"],
            "plan": [{"run": j} for j in ("dev", "hpo", "causal", "lookahead_s0", "faithful")],
            "results": {},
        }

    monkeypatch.setattr(job, "volume_state", lambda cfgs: state)
    monkeypatch.setattr(job, "run_gate", fake_gate)
    monkeypatch.setattr(job, "print_gate", lambda gate: None)
    monkeypatch.setattr(
        job,
        "chunk_budget",
        lambda rows, g, scale=1.0: {
            "rows": [r["run"] for r in rows],
            "scale": scale,
            "attempt_wall_s": 1000.0,
            "timeout_s": 2400,
            "max_set_wall_s": 3000.0,
        },
    )
    monkeypatch.setattr(train, "effective_params", _fake_params)
    driver = _Remote(result or {"status": "done", "summary": {"protocol": "x"}})
    monkeypatch.setattr(job, "train_driver", driver)
    return driver, gates


def test_train_entrypoint_submits_one_driver_call(monkeypatch):
    job = _job("train_gnn")
    common = _common()
    cfgs = common.load_all_configs()
    state = _entry_state(cfgs)
    driver, gates = _patch_entry(monkeypatch, job, state)
    job.main.info.raw_f(protocol="lookahead", seeds="0", final=True)
    (spec,) = driver.calls[0]
    g = cfgs["gnn"]
    params = _fake_params(g, "lookahead", state["best_params"])
    run_keys = {s: common.gnn_run_key(cfgs, "lookahead", s, params) for s in (0, 1, 2)}
    assert gates == [["lookahead_s0"]] and spec["seeds"] == [0]
    assert spec["run_keys"] == run_keys and spec["params"] == params  # every seed of the set
    assert spec["set_kind"] == "gnn_lookahead"
    assert spec["set_key"] == common.gnn_set_key(cfgs, "lookahead", list(run_keys.values()))
    assert spec["test_bounds"] == ["end", "d10"] and spec["final"] is True and not spec["dev"]
    assert spec["budget"]["rows"] == ["lookahead_s0"] and spec["budget"]["scale"] == 1.0
    assert spec["options"] == {
        "gpu": "L4",
        "cpu": (8.0, 8.0),
        "memory": (32768, 40960),
        "timeout": 2400,
    }
    assert spec["runtime"]["gpu"] == "L4" and spec["keys"] == state["keys"]
    # faithful runs on the decided faithful GPU; a causal seed subset scales the projection.
    job.main.info.raw_f(protocol="faithful", final=True)
    assert driver.calls[1][0]["options"]["gpu"] == "T4"
    assert driver.calls[1][0]["runtime"]["gpu"] == "T4"
    job.main.info.raw_f(protocol="causal", seeds="0,1", final=True)
    assert driver.calls[2][0]["budget"]["scale"] == pytest.approx(2 / 5)
    # the dev run: trial-0 params, the gnn_dev kind, max_epochs in the key, no test.
    job.main.info.raw_f(protocol="causal", seeds="0", dev=True, max_epochs=2)
    dev = driver.calls[3][0]
    assert dev["set_kind"] == "gnn_dev" and dev["final"] is False and dev["max_epochs"] == 2
    assert dev["params"] == _fake_params(g, "causal", None)
    assert dev["run_keys"][0] == common.gnn_run_key(
        cfgs, "causal", 0, dev["params"], dev=True, max_epochs=2
    )
    assert gates[-1] == ["dev"]


def test_train_entrypoint_refusals(monkeypatch):
    from aml.models.gnn import GnnStopError

    job = _job("train_gnn")
    cfgs = _common().load_all_configs()
    # The gate refuses -> nothing is submitted.
    driver, _ = _patch_entry(monkeypatch, job, _entry_state(cfgs), allowed=False)
    with pytest.raises(GnnStopError, match="cost gate refused"):
        job.main.info.raw_f(protocol="pna", final=True)
    assert driver.calls == []
    # decision.json differs from gnn.yaml's decided values.
    state = _entry_state(cfgs)
    state["decision"] = {**state["decision"], "batch_size": 4096}
    driver, _ = _patch_entry(monkeypatch, job, state)
    with pytest.raises(GnnStopError, match="batch_size"):
        job.main.info.raw_f(protocol="pna", final=True)
    # No bench yet.
    driver, _ = _patch_entry(monkeypatch, job, _entry_state(cfgs, done=[], decision=None))
    with pytest.raises(GnnStopError, match="make gnn-bench"):
        job.main.info.raw_f(protocol="faithful", final=True)
    # causal / look-ahead need HPO's best params.
    driver, _ = _patch_entry(monkeypatch, job, _entry_state(cfgs, done=["bench"]))
    with pytest.raises(SystemExit, match="make gnn-hpo"):
        job.main.info.raw_f(protocol="causal", final=True)
    # A --final set recorded another `report` hash.
    sets = {"causal": {"set_key": "k", "summary": {"final": True, "report_hash": "old"}}}
    driver, _ = _patch_entry(monkeypatch, job, _entry_state(cfgs, sets=sets))
    with pytest.raises(GnnStopError, match="report"):
        job.main.info.raw_f(protocol="pna", final=True)
    assert driver.calls == []
    # A stopped set or a failed worker ends the command with an error.
    for result, err in (
        ({"status": "stopped", "reason": "wall_guard"}, GnnStopError),
        ({"status": "failed", "error": "LeakError: boom"}, SystemExit),
    ):
        driver, _ = _patch_entry(monkeypatch, job, _entry_state(cfgs), result=result)
        with pytest.raises(err):
            job.main.info.raw_f(protocol="pna", final=True)
        assert len(driver.calls) == 1


def test_train_entrypoint_short_circuits_a_finished_set(monkeypatch):
    job = _job("train_gnn")
    common = _common()
    cfgs = common.load_all_configs()
    g = cfgs["gnn"]
    plan = job.set_plan(cfgs, "pna", _fake_params(g, "pna", None), [0, 1, 2])
    sets = {"pna": {**plan, "summary": {"final": True, "data_version": "v1"}}}
    driver, gates = _patch_entry(monkeypatch, job, _entry_state(cfgs, sets=sets))
    job.main.info.raw_f(protocol="pna", final=True)
    assert driver.calls == [] and gates == []
    # --plan-only: the gate table of the next unfinished run, nothing submitted.
    driver, gates = _patch_entry(monkeypatch, job, _entry_state(cfgs, done=["bench", "dev"]))
    job.main.info.raw_f(plan_only=True)
    assert driver.calls == [] and gates == [["hpo"]]


def test_hpo_entrypoint(monkeypatch):
    job = _job("hpo_gnn")
    tg = _job("train_gnn")
    cfgs = _common().load_all_configs()
    state = _entry_state(cfgs, done=["bench"], hpo_summary=None, best_params=None)
    driver, gates = _patch_entry(monkeypatch, tg, state)
    hpo = _Remote({"status": "done", "summary": {"best_trial": 0}})
    monkeypatch.setattr(job, "hpo_driver", hpo)
    job.main.info.raw_f()
    (spec,) = hpo.calls[0]
    assert gates == [["hpo"]] and spec["budget"]["rows"] == ["hpo"]
    assert spec["options"]["gpu"] == "L4" and spec["keys"] == state["keys"]
    assert driver.calls == []  # the train driver is never touched
    _patch_entry(monkeypatch, tg, _entry_state(cfgs))  # HPO already finished: no call
    job.main.info.raw_f()
    assert len(hpo.calls) == 1


def test_bench_entrypoint(monkeypatch):
    job = _job("gnn_bench")
    tg = _job("train_gnn")
    cfgs = _common().load_all_configs()
    state = _entry_state(cfgs, done=[], decision=None, bench_summary=None)
    _, gates = _patch_entry(monkeypatch, tg, state)
    decision = {"gpu": "L4", "batch_size": 4096, "faithful": {"gpu": "L4", "batch_size": 8192}}
    bench = _Remote({"status": "done", "summary": {"n_cells": 3}, "decision": decision})
    monkeypatch.setattr(job, "bench_driver", bench)
    job.main.info.raw_f()
    (spec,) = bench.calls[0]
    assert gates == [["bench"]] and set(spec) == {"data_cfg", "gnn_cfg", "keys"}
    snippet = job.decided_snippet(decision)
    assert "  batch_size: 4096" in snippet and "    batch_size: 8192   # on L4" in snippet
    job.main.info.raw_f(plan_only=True)
    assert len(bench.calls) == 1 and gates[-1] == ["bench"]


def _cell(gpu, cores, group, **kw):
    from aml.models.gnn import costplan

    return {
        "cell_id": costplan.cell_id(gpu, cores, group, kw.get("batch_size")),
        "group": group,
        "gpu": gpu,
        "cores": cores,
        "status": "ok",
        "guard": [100, 0, 0, -1, 0, 0],
        **kw,
    }


def test_run_bench_orchestration(tmp_path, monkeypatch):
    from aml.io import read_json
    from aml.models.gnn import (
        BENCH_FILE,
        CELLS_FILE,
        DECISION_FILE,
        PLAN_FILE,
        SUMMARY_FILE,
        append_jsonl,
        costplan,
    )

    job = _job("gnn_bench")
    g = _common().load_all_configs()["gnn"]
    calls, fail = [], {"T4"}

    def call(c):
        calls.append({k: c[k] for k in ("name", "gpu", "cores", "memory_mib", "groups", "rerun")})
        if c["name"] in fail:
            raise RuntimeError("container died")
        cells = [_cell(c["gpu"], c["cores"], grp) for grp in c["groups"]]
        if c["name"] == "L4":
            cells[0]["counts"] = {"n_pos": 2534, "n_neg": 3246387}
        for cell in cells:
            append_jsonl(tmp_path / CELLS_FILE, cell)
        return {"cells": cells}

    decided = {"gpu": "L4", "batch_size": 2048, "ask_user": None, "faithful": {"gpu": "L4"}}
    monkeypatch.setattr(
        costplan,
        "rerun_4core",
        lambda cells, cfg: {
            "gpu": "L4",
            "batch_size": 2048,
            "num_workers": 3,
            "cores": 4,
            "memory_mib": 16384,
        },
    )
    monkeypatch.setattr(costplan, "decide", lambda cells, cfg: dict(decided))
    monkeypatch.setattr(costplan, "plan_runs", lambda cfg, decision, counts: [{"run": "dev"}])
    monkeypatch.setattr(costplan, "gate", lambda plan, **kw: {"allowed": True, **kw})
    monkeypatch.setattr(costplan, "render_bench_md", lambda cells, d, plan: "# bench\n")
    monkeypatch.setattr(costplan, "plan_usd_h", lambda gpu, cores, gib: 3600.0)
    monkeypatch.setattr(costplan, "usd", lambda h, s, overhead=1.0: h / 3600 * s * overhead)
    report = tmp_path / "reports" / "gnn_bench.md"
    meta = {"bench_key": "gnn_bench-x", "data_version": "v1"}
    run = lambda: job.run_bench(g, tmp_path, report, call=call, meta=meta, log=lambda m: None)  # noqa: E731
    # T4 fails: recorded, nothing decided, no summary.
    out = run()
    assert out["status"] == "failed" and [c["name"] for c in calls] == ["L4", "T4"]
    state = read_json(tmp_path / BENCH_FILE)
    assert state["containers"]["L4"]["status"] == "done"
    assert "container died" in state["containers"]["T4"]["error"]
    assert not (tmp_path / SUMMARY_FILE).exists() and not (tmp_path / DECISION_FILE).exists()
    # Re-run: L4 is not called again; T4, then the 4-core rerun; then the decision.
    fail.clear()
    out = run()
    assert out["status"] == "done" and [c["name"] for c in calls] == ["L4", "T4", "T4", "rerun"]
    assert calls[0]["groups"] == list(costplan.CONTAINER_GROUPS["L4"])
    assert calls[2]["groups"] == list(costplan.CONTAINER_GROUPS["T4"])
    assert calls[0]["cores"] == 8 and calls[0]["memory_mib"] == 32768
    assert calls[3]["cores"] == 4 and calls[3]["memory_mib"] == 16384
    assert calls[3]["groups"] == list(costplan.CONTAINER_GROUPS["rerun"])
    assert calls[3]["rerun"]["num_workers"] == 3
    assert read_json(tmp_path / DECISION_FILE) == decided
    plan = read_json(tmp_path / PLAN_FILE)
    assert plan["rows"] == [{"run": "dev"}] and plan["gate"]["job"] == "dev"
    assert plan["gate"]["done"] == ["bench"] and plan["gate"]["metered_usd"] is None
    assert report.read_text(encoding="utf-8") == "# bench\n"
    s = read_json(tmp_path / SUMMARY_FILE)
    assert s["status"] == "done" and s["bench_key"] == "gnn_bench-x" and s["data_version"] == "v1"
    assert s["counts"] == {"n_pos": 2534, "n_neg": 3246387}
    assert [c["name"] for c in s["containers"]] == ["L4", "T4", "rerun"]
    n_cells = len(read_json(tmp_path / BENCH_FILE)["containers"])
    assert n_cells == 3 and s["n_cells"] == 8 + 5 + 2
    assert s["guard"]["edges_checked"] == 100 * s["n_cells"] and s["guard"]["violations"] == 0
    assert s["decision"]["batch_size"] == 2048
    assert (tmp_path / SUMMARY_FILE).stat().st_mtime_ns >= (tmp_path / PLAN_FILE).stat().st_mtime_ns
    # A finished bench is returned as is.
    assert run()["resumed"] is True and len(calls) == 4
    # costplan.decide finds no eligible cell: failed, no summary.
    (tmp_path / SUMMARY_FILE).unlink()

    def no_cell(cells, cfg):
        raise ValueError("no eligible cell")

    monkeypatch.setattr(costplan, "decide", no_cell)
    out = run()
    assert out["status"] == "failed" and "no eligible" in out["error"] and len(calls) == 4


def test_bench_container_options():
    job = _job("gnn_bench")
    assert job.container_options({"gpu": "T4", "cores": 8, "memory_mib": 32768}) == {
        "gpu": "T4",
        "cpu": (8.0, 8.0),
        "memory": (32768, 40960),
    }
    opts = job.container_options({"gpu": "L4", "cores": 4, "memory_mib": 16384})
    assert opts["cpu"] == (4.0, 4.0) and opts["memory"] == (16384, 24576)
    assert opts["env"]["OMP_NUM_THREADS"] == "4"


def test_gnn_eval_models():
    from aml.models import gnn

    job = _job("train_gnn")
    common = _common()
    cfgs = common.load_all_configs()
    v, made = _gnn_volume(cfgs)
    out = job.gnn_eval_models(cfgs, read=v.read, params_fn=_fake_params)
    assert out["models"] == {
        "gnn_causal": {"kind": "gnn_causal", "key": made["causal"]["set_key"], "make": "gnn"}
    }
    assert out["faithful"] is None and out["model_views"] == {}
    assert out["report_cfg"] == cfgs["gnn"]["report"]
    paths = common.data_paths(cfgs["data"])
    la = made["lookahead"]
    # A started but unassembled set (a run dir with its fingerprint, no set summary) refuses:
    # evaluating now would touch test without that protocol's section.
    fp = paths.gnn_run_dir(la["run_keys"][0]) / gnn.FINGERPRINT_FILE
    v.put(fp, {"run_key": la["run_keys"][0]})
    with pytest.raises(SystemExit, match="no finished set.*make gnn-lookahead"):
        job.gnn_eval_models(cfgs, read=v.read, params_fn=_fake_params)
    g0 = cfgs["gnn"]
    fa0 = job.set_plan(cfgs, "faithful", _fake_params(g0, "faithful", None), [0])
    v.put(paths.gnn_run_dir(fa0["run_keys"][0]) / gnn.FINGERPRINT_FILE, {"run_key": "x"})
    del v.docs[fp.as_posix()]
    with pytest.raises(SystemExit, match="no finished set.*make gnn-faithful"):
        job.gnn_eval_models(cfgs, read=v.read, params_fn=_fake_params)
    v.put(fp, {"run_key": la["run_keys"][0]})
    v.put(Path(la["set_dir"]) / gnn.SUMMARY_FILE, {"final": True})
    with pytest.raises(SystemExit, match="both or neither"):
        job.gnn_eval_models(cfgs, read=v.read, params_fn=_fake_params)
    v.put(paths.gnn_set_dir("gnn_lookahead_d10", la["set_key"]) / gnn.SUMMARY_FILE, {"final": True})
    g = cfgs["gnn"]
    fa = job.set_plan(cfgs, "faithful", _fake_params(g, "faithful", None), [0])
    v.put(Path(fa["set_dir"]) / gnn.SUMMARY_FILE, {"final": True})
    out = job.gnn_eval_models(cfgs, read=v.read, params_fn=_fake_params)
    assert list(out["models"]) == ["gnn_causal", "gnn_lookahead", "gnn_lookahead_d10"]
    assert out["models"]["gnn_lookahead_d10"] == {
        "kind": "gnn_lookahead_d10",
        "key": la["set_key"],
        "make": "gnn-lookahead",
    }
    assert out["faithful"] == {"kind": "gnn_faithful", "key": fa["set_key"], "make": "gnn-faithful"}
    assert out["model_views"] == {"gnn_lookahead_d10": ["primary"]}
    v.put(Path(fa["set_dir"]) / gnn.SUMMARY_FILE, {"final": False})
    with pytest.raises(SystemExit, match="--final"):
        job.gnn_eval_models(cfgs, read=v.read, params_fn=_fake_params)
    v2, _ = _gnn_volume(cfgs)
    del v2.docs[(Path(made["causal"]["set_dir"]) / gnn.SUMMARY_FILE).as_posix()]
    with pytest.raises(SystemExit, match="make gnn"):
        job.gnn_eval_models(cfgs, read=v2.read, params_fn=_fake_params)


def _gnn_spec() -> dict:
    return {
        "models": {
            "gnn_causal": {"kind": "gnn_causal", "key": "gnn_causal-1", "make": "gnn"},
            "gnn_lookahead": {"kind": "gnn_lookahead", "key": "gnn_lookahead-2", "make": "gl"},
            "gnn_lookahead_d10": {
                "kind": "gnn_lookahead_d10",
                "key": "gnn_lookahead-2",
                "make": "gl",
            },
        },
        "faithful": {"kind": "gnn_faithful", "key": "gnn_faithful-3", "make": "gnn-faithful"},
        "model_views": {"gnn_lookahead_d10": ["primary"]},
        "report_cfg": {"x": 1},
    }


def test_evaluate_with_gnn_entrypoint(monkeypatch):
    ev = _job("evaluate")
    tg = _job("train_gnn")
    common = _common()
    spec = _gnn_spec()
    monkeypatch.setattr(tg, "gnn_eval_models", lambda cfgs: spec)
    remote = _Remote({"models": [], "extras": [], "reports": []})
    monkeypatch.setattr(ev, "evaluate", remote)
    ev.main.info.raw_f(with_gnn=True, skip_cost=True)
    data, rules, keys, nofmt, gnn = remote.calls[0]
    cfgs = common.load_all_configs()
    gk = {
        "gnn_causal": "gnn_causal-1",
        "gnn_lookahead": "gnn_lookahead-2",
        "gnn_lookahead_d10": "gnn_lookahead-2",
        "gnn_faithful": "gnn_faithful-3",
    }
    assert ev.gnn_eval_keys(spec) == gk and gnn is spec and nofmt is False
    assert keys["eval"] == common.eval_key(cfgs, gnn_keys=gk) != M2_KEYS["eval"]
    assert {k: v for k, v in keys.items() if k != "eval"} == {
        k: v for k, v in common.all_keys(cfgs).items() if k != "eval"
    }
    ev.main.info.raw_f(skip_cost=True)  # without --with-gnn: M2's call, M2's key
    assert len(remote.calls[1]) == 4 and remote.calls[1][2]["eval"] == M2_KEYS["eval"]


def test_evaluate_checks_the_gnn_sets(tmp_path):
    from aml.io import write_json_atomic
    from aml.paths import DataPaths

    ev = _job("evaluate")
    common = _common()
    paths = DataPaths(tmp_path)
    spec = _gnn_spec()
    dirs, faithful_dir = ev.gnn_stage_dirs(paths, spec)
    assert list(dirs) == ["gnn_causal", "gnn_lookahead", "gnn_lookahead_d10"]
    assert dirs["gnn_lookahead_d10"] == paths.model_dir("gnn_lookahead_d10", "gnn_lookahead-2")
    assert faithful_dir == paths.model_dir("gnn_faithful", "gnn_faithful-3")
    bad = copy.deepcopy(spec)
    bad["models"]["gnn_faithful"] = bad["faithful"]
    with pytest.raises(ValueError, match="never gnn_faithful"):
        ev.gnn_stage_dirs(paths, bad)
    feats = tmp_path / "features"
    _write_summary(feats, features_digest="d1")
    with pytest.raises(FileNotFoundError, match="make gnn"):
        ev._gnn_inputs(paths, spec, "v1", feats)
    for name, d in [*dirs.items(), ("gnn_faithful", faithful_dir)]:
        _write_summary(d, features_digest="d1", final=True, name=name)
        common.stamp_data_version(d, "v1")
    _, _, arg = ev._gnn_inputs(paths, spec, "v1", feats)
    assert (
        set(arg["summaries"]) == set(dirs)
        and arg["summaries"]["gnn_causal"]["name"] == "gnn_causal"
    )
    assert arg["faithful"]["scores"] == faithful_dir / "scores.parquet"
    assert arg["faithful"]["summary"]["name"] == "gnn_faithful"
    assert arg["model_views"] == {"gnn_lookahead_d10": ["primary"]} and arg["report_cfg"] == {
        "x": 1
    }
    common.stamp_data_version(dirs["gnn_lookahead"], "v0")  # other prepared data
    with pytest.raises(RuntimeError, match="make gl"):
        ev._gnn_inputs(paths, spec, "v1", feats)
    common.stamp_data_version(dirs["gnn_lookahead"], "v1")
    write_json_atomic({"features_digest": "d0"}, faithful_dir / "summary.json")  # older parts
    with pytest.raises(RuntimeError, match="make gnn-faithful"):
        ev._gnn_inputs(paths, spec, "v1", feats)


_TRAIN_PARAMS_SCRIPT = r"""
import json, sys
for name in ("torch", "torch_geometric", "pyg_lib"):
    sys.modules[name] = None
import yaml
from aml.models.gnn.train import effective_params
g = yaml.safe_load(open("configs/gnn.yaml", encoding="utf-8"))
best = {"lr": 0.01, "final_dropout": 0.2, "w_pos": 4.0}
out = {p: effective_params(g, p, best if p in ("causal", "lookahead") else None)
       for p in ("causal", "lookahead", "pna", "faithful")}
out["dev"] = effective_params(g, "causal", None)
print("RESULT=" + json.dumps(out))
"""


def test_effective_params_import_without_torch():
    """Contract (owner E): every GNN entrypoint computes run keys on the laptop from
    train.effective_params, so aml.models.gnn.train must import and run without torch."""
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("MODAL_TOKEN", "MODAL_PROFILE"))
    }
    proc = subprocess.run(
        [sys.executable, "-c", _TRAIN_PARAMS_SCRIPT],
        cwd=REPO_ROOT,
        env={**env, "PYTHONIOENCODING": "utf-8"},
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=300,
    )
    assert proc.returncode == 0, proc.stderr[-3000:]
    out = json.loads(
        next(ln for ln in proc.stdout.splitlines() if ln.startswith("RESULT=")).removeprefix(
            "RESULT="
        )
    )
    assert out["causal"]["lr"] == 0.01 and out["dev"]["lr"] == 0.006213  # dev: trial 0
    assert out["faithful"]["neg_rate"] == 1.0 and out["pna"]["hidden"] == 20
    json.dumps(out)  # hashable into run keys


# --- M6: the case_eval job ------------------------------------------------------------------


def test_case_eval_job_guardrail_kwargs(imported):
    common = _common()
    job = _job("case_eval")
    w = job.WORKER_KW
    assert job.APP_NAME == "aml-case-eval" == imported["case_eval"]["app"]
    assert job.APP_NAME not in common.M3_APPS  # never charged to the GNN cost gate
    # One CPU container: 1 core (the scorer is single-threaded), no GPU, no retries.
    assert w["image"] is common.cpu_image and "gpu" not in w and "retries" not in w
    assert w["cpu"] == job.CPU == 1.0
    assert w["memory"] == (job.MEMORY_MIB, job.MEMORY_MIB + 4096)
    assert w["timeout"] == job.TIMEOUT_S <= 3600
    assert w["max_containers"] == 1 and w["scaledown_window"] == 2
    assert w["env"]["OMP_NUM_THREADS"] == "1" and w["env"]["POLARS_MAX_THREADS"] == "1"
    assert not {"min_containers", "buffer_containers"} & set(w)
    assert set(w["volumes"]) == {"/data"}
    fns = imported["case_eval"]["functions"]
    assert set(fns) == {"case_eval"} and imported["case_eval"]["entrypoints"] == ["main"]
    spec = fns["case_eval"]
    if spec is not None:
        assert not spec["gpus"] and spec["cpu"] == 1.0 and spec["memory"] == list(w["memory"])


def test_case_eval_key_tracks_the_champion_the_explain_config_and_the_period():
    from aml.explain.case_eval import PERIODS

    common = _common()
    job = _job("case_eval")
    assert tuple(PERIODS) == job.PERIODS  # the job's choices are the library's periods
    cfgs = common.load_all_configs()
    explain = job.load_explain_cfg()
    k = {p: job.case_eval_key(cfgs, explain, p) for p in job.PERIODS}
    assert k["val"] != k["test"] and all(v.startswith("case_eval-") for v in k.values())
    assert job.case_eval_key(copy.deepcopy(cfgs), copy.deepcopy(explain), "test") == k["test"]
    tuned = copy.deepcopy(explain)
    tuned["typology"]["fan_in_min"] = 99  # frozen thresholds copied in after the val run
    assert job.case_eval_key(cfgs, tuned, "test") != k["test"]
    c = copy.deepcopy(cfgs)
    c["serving"]["replay"]["max_events"] = 10  # re-keys the export, i.e. the champion
    assert job.case_eval_key(c, explain, "test") != k["test"]
    c = copy.deepcopy(cfgs)
    c["data"]["test_views"]["primary"] = [9, 9]
    assert job.case_eval_key(c, explain, "test") != k["test"]
    c = copy.deepcopy(cfgs)
    for name in [n for n in c["serving"] if n != "replay"]:  # the M5 demo settings
        c["serving"][name] = {"changed": True}
    assert job.case_eval_key(c, explain, "test") == k["test"]
    refit = copy.deepcopy(explain)
    refit["typology_tree"] = _HAND_TREE  # a new frozen tree copied in after the val run
    assert job.case_eval_key(cfgs, refit, "test") != k["test"]
    with pytest.raises(ValueError, match="period"):
        job.case_eval_key(cfgs, explain, "train")


def test_case_eval_stage_dirs():
    from aml.paths import DataPaths

    job = _job("case_eval")
    keys = {
        "features": "features-k",
        "rules_engine": "rules_engine-k",
        "lgbm_graph": "lgbm_graph-k",
    }
    assert job.stage_dirs(DataPaths(Path("/data")), keys) == (
        Path("/data/features/hi_small/features-k"),
        Path("/data/models/rules_engine/rules_engine-k"),
        Path("/data/models/lgbm_graph/lgbm_graph-k"),
    )


def test_case_eval_entrypoint(monkeypatch, capsys):
    job = _job("case_eval")
    common = _common()
    cfgs = common.load_all_configs()
    explain = job.load_explain_cfg()
    summary = {
        "period": "val",
        "parity_ok": True,
        "parity": {"mismatches": {"scores": 0}, "alerts": {"stream": 2, "reference": 2}},
        "tuned": dict(explain["typology"]),
        "accuracy": {"rules": {"n": 2}, "tree_cv": 0.5},
        "tree_report": "/data/reports/typology_tree_val.json",
        "reports": ["/data/reports/typology_match_val.json"],
    }
    remote = _Remote(summary)
    monkeypatch.setattr(job, "case_eval", remote)
    job.main.info.raw_f(period="val")
    _, keys, sent, period = remote.calls[0]
    assert period == "val" and sent == explain
    assert keys["case_eval"] == job.case_eval_key(cfgs, explain, "val")
    assert {k: v for k, v in keys.items() if k != "case_eval"} == common.all_keys(cfgs)
    out = capsys.readouterr().out
    assert "typology:\n  cycle_min: " in out and "make cases" in out
    assert "reports/typology_tree_val.json to configs/typology_tree.json" in out
    with pytest.raises(SystemExit, match="--period"):
        job.main.info.raw_f(period="train")
    assert len(remote.calls) == 1


_HAND_TREE = {
    "format": 1,
    "tree": {
        "feature": "g_n_edges",
        "threshold": 0.5,
        "left": {"label": "OTHER", "n": 1, "dist": {"OTHER": 1}},
        "right": {"label": "FAN-IN", "n": 1, "dist": {"FAN-IN": 1}},
    },
    "trained_on": None,
}


def test_case_eval_test_entrypoint_needs_the_frozen_tree(monkeypatch):
    """--period test sends the frozen tree (configs/typology_tree.json) with the explain
    config, and refuses the single-leaf placeholder under typology_model: tree before any
    remote call."""
    from aml.explain.casepack import check_config

    job = _job("case_eval")
    explain = job.load_explain_cfg()
    placeholder = {"format": 1, "tree": {"label": "OTHER", "n": 0, "dist": {}}, "trained_on": None}
    remote = _Remote({"period": "test", "parity_ok": True, "parity": {}, "reports": []})
    monkeypatch.setattr(job, "case_eval", remote)
    stub = check_config({**explain, "typology_model": "tree", "typology_tree": placeholder})
    monkeypatch.setattr(job, "load_explain_cfg", lambda: stub)
    with pytest.raises(SystemExit, match="typology_tree.json"):
        job.main.info.raw_f(period="test")
    assert remote.calls == []
    frozen = check_config({**explain, "typology_model": "tree", "typology_tree": _HAND_TREE})
    monkeypatch.setattr(job, "load_explain_cfg", lambda: frozen)
    job.main.info.raw_f(period="test")
    _, _, sent, period = remote.calls[0]
    assert period == "test" and sent["typology_tree"] == _HAND_TREE


def test_makefile_m6_targets():
    text = (REPO_ROOT / "Makefile").read_text(encoding="utf-8")
    recipes = {
        "cases-val": "run --detach -m modal_jobs.case_eval --period val",
        "cases": "run --detach -m modal_jobs.case_eval --period test",
    }
    phony = {
        name
        for line in text.splitlines()
        if line.startswith(".PHONY:")
        for name in line.removeprefix(".PHONY:").split()
    }
    for target, cmd in recipes.items():
        assert f"\n{target}:\n\t$(MODAL) {cmd}\n" in text, target
        assert target in phony, target
    help_block = text.split("\nhelp:\n", 1)[1].split("\n\n", 1)[0]
    assert "M6: cases-val" in help_block and " cases " in help_block
