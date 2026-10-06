"""GNN HPO (M3 spec §8, §14.2 row E): the enqueued trial 0, the JSONL rebuild and per-trial
reseeding (restart-reproducible proposals), params persisted before training, pruning, OOM ->
FAIL, the best-params tie rule, and no val_late / test loader. The Optuna logic runs on the
laptop; the end-to-end search needs torch (CPU, synthetic fixture)."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from aml.models.gnn import (
    BEST_PARAMS_FILE,
    HPO_KIND,
    PARAMS_FILE,
    SUMMARY_FILE,
    TRIALS_FILE,
    read_jsonl,
    trial_dir_name,
)
from aml.models.gnn import hpo as H
from tests.conftest import load_yaml


@pytest.fixture(autouse=True)
def _restore_torch_globals():
    """set_determinism changes process-wide torch flags; give other suites theirs back."""
    try:
        import torch as _t
    except ImportError:
        yield
        return
    det = _t.are_deterministic_algorithms_enabled()
    warn = _t.is_deterministic_algorithms_warn_only_enabled()
    prec = _t.get_float32_matmul_precision()
    yield
    _t.use_deterministic_algorithms(det, warn_only=warn)
    _t.set_float32_matmul_precision(prec)


@pytest.fixture(scope="module")
def gcfg() -> dict:
    return load_yaml("gnn.yaml")


def scripted_records(gcfg: dict, values: list[float | None], states=None) -> list[dict]:
    """Trial records whose params are what the rebuilt study asks (so they form a valid log)."""
    recs: list[dict] = []
    for k, v in enumerate(values):
        st = H.rebuild_study(recs, gcfg, k)
        t = st.ask(H._distributions(gcfg))
        state = (states or {}).get(k, "COMPLETE" if v is not None else "FAIL")
        iv = {e: (v or 0.0) * (e + 1) / 4 for e in range(4)} if v is not None else {}
        recs.append(H.trial_record(k, t.params, state, v, iv, 3 if v else None, 1.0, {}))
    return recs


# --- the Optuna logic (laptop) -------------------------------------------------------------------


def test_trial0_is_the_enqueued_train_section(gcfg):
    t0 = H.trial0_params(gcfg)
    assert t0 == {k: float(gcfg["train"][k]) for k in ("lr", "final_dropout", "w_pos")}
    trial = H.rebuild_study([], gcfg, 0).ask(H._distributions(gcfg))
    assert trial.number == 0 and trial.params == t0


def test_rebuild_and_per_trial_reseeding_give_identical_proposals(gcfg):
    recs = scripted_records(gcfg, [0.31, 0.42, None, 0.27, 0.5])
    for k in range(1, 6):
        a = H.rebuild_study(recs[:k], gcfg, k).ask(H._distributions(gcfg))
        b = H.rebuild_study(copy.deepcopy(recs[:k]), gcfg, k).ask(H._distributions(gcfg))
        assert a.number == b.number == k
        assert a.params == b.params  # a restart asks exactly the same point
    # the JSONL round trip (string epoch keys) keeps the proposals
    jl = json.loads(json.dumps(recs))
    for k in range(1, 6):
        a = H.rebuild_study(recs[:k], gcfg, k).ask(H._distributions(gcfg))
        b = H.rebuild_study(jl[:k], gcfg, k).ask(H._distributions(gcfg))
        assert a.params == b.params
    # the sampler seed is per trial: base + k
    c = copy.deepcopy(gcfg)
    c["hpo"]["sampler_seed_base"] += 1
    a = H.rebuild_study(recs[:2], gcfg, 2).ask(H._distributions(gcfg))
    b = H.rebuild_study(recs[:2], c, 2).ask(H._distributions(c))
    assert a.params != b.params
    # every proposal lies in the search space
    space = gcfg["hpo"]["space"]
    for r in recs:
        for name, v in r["params"].items():
            assert space[name]["low"] <= v <= space[name]["high"]
    with pytest.raises(ValueError):
        H.rebuild_study([recs[1]], gcfg, 1)  # numbers must be 0..n-1


def test_best_params_tie_rule_and_trial_record(gcfg):
    recs = scripted_records(gcfg, [0.5, 0.7, 0.7, 0.9, None], states={3: "PRUNED"})
    number, value, params = H.best_params(recs)
    assert (number, value) == (1, 0.7)  # PRUNED never wins; a tie goes to the lower number
    assert params == recs[1]["params"] and set(params) == {"lr", "final_dropout", "w_pos"}
    assert H.best_params([r for r in recs if r["state"] != "COMPLETE"]) is None
    r = H.trial_record(3, {"lr": 0.01}, "PRUNED", 0.2, {0: 0.1, 2: 0.2}, 1, 2.5, {"train": {}})
    assert r["intermediate_values"] == {"0": 0.1, "2": 0.2}
    assert set(r) == {
        "number",
        "params",
        "state",
        "value",
        "intermediate_values",
        "best_epoch",
        "seconds",
        "guard",
    }
    with pytest.raises(ValueError):
        H.trial_record(0, {}, "COMPLETE", float("nan"), {}, None, 0.0, {})
    with pytest.raises(ValueError):
        H.trial_record(0, {}, "RUNNING", None, {}, None, 0.0, {})


# --- the search on the fixture (torch) ------------------------------------------------------------


@pytest.fixture(scope="module")
def cfgs(data_cfg, rules_cfg):
    return {"data": data_cfg, "rules": rules_cfg, "features": load_yaml("features.yaml")}


@pytest.fixture
def env(prepared, cfgs, tmp_path, monkeypatch):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from aml.models.gnn import train as tr
    from tests.fixtures.gnn_graphs import gnn_inputs

    monkeypatch.setattr(tr, "MLFLOW_ENABLED", False)
    paths, fdir, cfg = gnn_inputs(prepared, tmp_path, cfgs)
    return paths, fdir, cfg, cfgs["data"]


def hpo(env, *, cfg=None, budget_s=1e9, key="gnn_hpo-test"):
    paths, fdir, base, data_cfg = env
    cfg = cfg or base
    return H.run_hpo(
        paths,
        fdir,
        paths.gnn_set_dir(HPO_KIND, key),
        cfg,
        data_cfg=data_cfg,
        device="cpu",
        runtime=cfg["runtime"],
        budget_s=budget_s,
    )


def test_run_hpo_end_to_end_without_late_or_test_data(env, monkeypatch):
    import polars as pl

    import aml.models.lgbm as lgbm
    from aml.models.gnn import sampler as S

    paths = env[0]
    tx = pl.read_parquet(paths.transactions, columns=["row_id", "split"])
    allowed = set(tx.filter(pl.col("split").is_in(["train", "val_early"]))["row_id"])
    seen: list[int] = []
    real_labels = lgbm.load_labels

    def spy_labels(path, row_ids):
        seen.extend(row_ids.to_list())
        return real_labels(path, row_ids)

    roles: list[tuple[str, set]] = []
    real_make = S.make_loader

    def spy_make(g, hetero, seed_gids, **kw):
        roles.append((kw["role"], set(g.split_code[np.asarray(seed_gids)].tolist())))
        return real_make(g, hetero, seed_gids, **kw)

    monkeypatch.setattr(lgbm, "load_labels", spy_labels)
    monkeypatch.setattr(S, "make_loader", spy_make)
    res = hpo(env)
    assert res["status"] == "done", res
    hdir = paths.gnn_set_dir(HPO_KIND, "gnn_hpo-test")
    log = read_jsonl(paths.gnn_optuna_log("gnn_hpo-test"))
    assert [r["number"] for r in log] == [0, 1]
    assert log[0]["params"] == H.trial0_params(env[2])
    assert all(r["state"] in H.TRIAL_STATES for r in log)
    best = json.loads((hdir / BEST_PARAMS_FILE).read_text("utf-8"))
    assert set(best) == {"lr", "final_dropout", "w_pos"} and best == res["best_params"]
    s = json.loads((hdir / SUMMARY_FILE).read_text("utf-8"))
    assert s["n_trials"] == 2 and s["violations"] == 0 and s["edges_checked"] > 0
    assert s["target_hits"] == 0 and set(s["guard"]) == {"train", "val_early"}
    assert json.loads((hdir / TRIALS_FILE).read_text("utf-8")) == s["trials"]
    for k in (0, 1):
        tdir = hdir / trial_dir_name(k)
        assert json.loads((tdir / PARAMS_FILE).read_text("utf-8")) == log[k]["params"]
    # HPO never reads val_late / test labels nor builds a loader over them
    assert seen and set(seen) <= allowed
    assert {r for r, _ in roles} == {"train", "eval"}
    assert all(splits <= {0, 1} for _, splits in roles)
    # a finished search short-circuits
    again = hpo(env)
    assert again["status"] == "done" and again["summary"] == s
    # a summary of other feature parts does not: the search is re-finalised from its log
    (hdir / SUMMARY_FILE).write_text(json.dumps({**s, "features_digest": "old"}), "utf-8")
    third = hpo(env)
    assert third["status"] == "done" and third["summary"]["features_digest"] == s["features_digest"]
    assert third["summary"]["trials"] == s["trials"]
    assert json.loads((hdir / BEST_PARAMS_FILE).read_text("utf-8")) == best
    # a log of other inputs is moved aside and the search starts over
    log_path = paths.gnn_optuna_log("gnn_hpo-test")
    stale = [{**r, "features_digest": "old"} for r in log]
    log_path.write_text("".join(json.dumps(r) + chr(10) for r in stale), "utf-8")
    (hdir / SUMMARY_FILE).unlink()
    fourth = hpo(env)
    assert fourth["status"] == "done"
    moved = [f for f in log_path.parent.iterdir() if ".jsonl.other-data-" in f.name]
    assert len(moved) == 1
    assert [r["features_digest"] for r in read_jsonl(log_path)] == [s["features_digest"]] * 2


def test_hpo_resumes_after_partial_and_matches_a_straight_search(
    prepared, cfgs, tmp_path, monkeypatch
):
    pytest.importorskip("torch")
    from aml.models.gnn import train as tr
    from tests.fixtures.gnn_graphs import gnn_inputs

    monkeypatch.setattr(tr, "MLFLOW_ENABLED", False)
    env_a = (*gnn_inputs(prepared, tmp_path / "a", cfgs), cfgs["data"])
    env_b = (*gnn_inputs(prepared, tmp_path / "b", cfgs), cfgs["data"])
    assert hpo(env_a)["status"] == "done"
    statuses = []
    for _ in range(10):
        r = hpo(env_b, budget_s=0.0)  # one epoch per call
        statuses.append(r["status"])
        if r["status"] != "partial":
            break
    assert statuses == ["partial"] * 3 + ["done"], statuses  # 2 trials x 2 epochs
    la = read_jsonl(env_a[0].gnn_optuna_log("gnn_hpo-test"))
    lb = read_jsonl(env_b[0].gnn_optuna_log("gnn_hpo-test"))
    drop = ("seconds", "guard")  # a resumed call re-streams the val cache: more edges checked
    assert [{k: v for k, v in r.items() if k not in drop} for r in la] == [
        {k: v for k, v in r.items() if k not in drop} for r in lb
    ]


def test_params_json_mismatch_fails_fast(env):
    paths = env[0]
    hdir = paths.gnn_set_dir(HPO_KIND, "gnn_hpo-test")
    pfile = hdir / trial_dir_name(0) / PARAMS_FILE
    pfile.parent.mkdir(parents=True)
    pfile.write_text(json.dumps({"lr": 0.0111, "final_dropout": 0.1, "w_pos": 5.0}), "utf-8")
    res = hpo(env)
    assert res["status"] == "failed" and "params.json" in res["error"]
    assert read_jsonl(paths.gnn_optuna_log("gnn_hpo-test")) == []


def fake_train_run(script: dict[int, list[float]], calls: list):
    """A scripted train.train_run: per trial k the per-epoch metrics script[k], reported
    through the real on_epoch hook (trial.report + should_prune)."""
    from aml.models.gnn import train as tr

    def run(engine, spec, clock, *, on_checkpoint=None, on_epoch=None, on_resume=None, **kw):
        k = int(spec.run_key.rsplit("-t", 1)[1])
        calls.append(k)
        hist = []
        for epoch, m in enumerate(script[k][: spec.max_epochs]):
            hist.append({"epoch": epoch, "metric": m, "seconds": 0.01, "guard": {}})
            if on_epoch is not None and on_epoch(epoch, m, hist):
                return tr.RunOutcome("pruned", epoch + 1, history=hist)
        best = int(np.argmax(script[k][: spec.max_epochs]))
        summary = {"metric": float(np.max(script[k][: spec.max_epochs])), "best_epoch": best}
        return tr.RunOutcome("trained", len(hist), summary=summary, history=hist)

    return run


def pruning_cfg(cfg: dict) -> dict:
    c = copy.deepcopy(cfg)
    c["hpo"].update(n_trials=5, max_epochs=6)
    c["hpo"]["pruner"].update(n_startup_trials=3, n_warmup_steps=3)
    return c


def test_pruning_after_warmup_and_restart_reproducible(env, monkeypatch):
    from aml.models.gnn import train as tr

    good = [0.2, 0.4, 0.6, 0.8, 0.85, 0.9]
    script = {0: good, 1: good, 2: good, 3: [0.01] * 6, 4: [0.5, 0.6, 0.7, 0.8, 0.9, 0.95]}
    calls: list[int] = []
    monkeypatch.setattr(tr, "train_run", fake_train_run(script, calls))
    cfg = pruning_cfg(env[2])
    res = hpo(env, cfg=cfg)
    assert res["status"] == "done", res
    log = read_jsonl(env[0].gnn_optuna_log("gnn_hpo-test"))
    states = [r["state"] for r in log]
    assert states == ["COMPLETE", "COMPLETE", "COMPLETE", "PRUNED", "COMPLETE"]
    # no pruning before the warm-up steps: trial 3 reported epochs 0..3, pruned at step 3
    assert sorted(int(e) for e in log[3]["intermediate_values"]) == [0, 1, 2, 3]
    assert res["best_trial"] == 4 and res["best_value"] == 0.95
    # a restart after trial 2 (a crash in trial 3) asks the same params for trials 3 and 4
    calls2: list[int] = []
    crash = fake_train_run(script, calls2)

    def crash_at_3(engine, spec, clock, **kw):
        if spec.run_key.endswith("-t3") and not calls2.count(3):
            calls2.append(3)
            raise KeyboardInterrupt("preempted")
        return crash(engine, spec, clock, **kw)

    monkeypatch.setattr(tr, "train_run", crash_at_3)
    with pytest.raises(KeyboardInterrupt):
        hpo(env, cfg=cfg, key="gnn_hpo-restart")
    res2 = hpo(env, cfg=cfg, key="gnn_hpo-restart")
    assert res2["status"] == "done"
    log2 = read_jsonl(env[0].gnn_optuna_log("gnn_hpo-restart"))
    assert [r["params"] for r in log2] == [r["params"] for r in log]
    assert [r["state"] for r in log2] == states


def test_oom_trial_is_logged_fail_and_the_search_continues(env, monkeypatch):
    from aml.models.gnn import train as tr

    real = tr.train_run

    def oom_in_trial_0(engine, spec, clock, **kw):
        if spec.run_key.endswith("-t0"):
            raise tr.GnnOOMError("CUDA OOM on a training batch of 12345 edges", 12345, 1)
        return real(engine, spec, clock, **kw)

    monkeypatch.setattr(tr, "train_run", oom_in_trial_0)
    res = hpo(env)
    assert res["status"] == "done", res
    log = read_jsonl(env[0].gnn_optuna_log("gnn_hpo-test"))
    assert log[0]["state"] == "FAIL" and "12345 edges" in log[0]["error"]
    assert log[0]["value"] is None and log[1]["state"] == "COMPLETE"
    assert res["best_trial"] == 1
    s = res["summary"]
    assert s["n_failed"] == 1 and s["n_complete"] == 1


def test_no_complete_trial_fails(env, monkeypatch):
    from aml.models.gnn import train as tr

    def always_oom(engine, spec, clock, **kw):
        raise tr.GnnOOMError("OOM", 7, 1)

    monkeypatch.setattr(tr, "train_run", always_oom)
    res = hpo(env)
    assert res["status"] == "failed" and "no COMPLETE trial" in res["error"]
    assert [r["state"] for r in read_jsonl(env[0].gnn_optuna_log("gnn_hpo-test"))] == ["FAIL"] * 2
