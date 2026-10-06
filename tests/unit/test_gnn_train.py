"""GNN training: loss, determinism, checkpoints, exact resume, the money guards, scoring and set
assembly (M3 spec §7, §9, §14.2 row E). CPU only, on the synthetic fixture (tests/fixtures/
gnn_graphs.gnn_inputs) with the real graph / sampler / model code."""

from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import numpy as np
import polars as pl
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from aml.models.gnn import (  # noqa: E402
    BEST_CKPT,
    FAILED_FILE,
    HISTORY_FILE,
    LAST_CKPT,
    LOOKAHEAD_D10_MODEL,
    RUNNING_FILE,
    SCORES_FILE,
    SEED_SUMMARY_FILE,
    STOPPED_FILE,
    SUMMARY_FILE,
    TEST_BOUNDS,
    WRITER_FILE,
    LeakError,
    read_jsonl,
    scores_file,
    set_kind,
)
from aml.models.gnn import train as tr  # noqa: E402
from tests.conftest import load_yaml  # noqa: E402
from tests.fixtures.gnn_graphs import gnn_inputs  # noqa: E402

VOLATILE = ("seconds", "train_s", "val_s", "peak_gpu_bytes", "val_cached")


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
def cfgs(data_cfg, rules_cfg):
    return {"data": data_cfg, "rules": rules_cfg, "features": load_yaml("features.yaml")}


@pytest.fixture
def env(prepared, cfgs, tmp_path, monkeypatch):
    """(paths, features_dir, gnn_cfg, data_cfg) with private Volume dirs; MLflow off."""
    monkeypatch.setattr(tr, "MLFLOW_ENABLED", False)
    paths, fdir, cfg = gnn_inputs(prepared, tmp_path, cfgs)
    return paths, fdir, cfg, cfgs["data"]


def run(
    env,
    *,
    protocol="causal",
    seeds=(0,),
    set_seeds=None,
    final=False,
    budget_s=1e9,
    set_key="set-a",
    cfg=None,
    **kw,
):
    paths, fdir, base_cfg, data_cfg = env
    cfg = cfg or base_cfg
    set_seeds = list(set_seeds or seeds)
    keys = {int(s): f"gnn-test-{protocol}-{set_key}-s{s}" for s in set_seeds}
    set_dir = paths.gnn_set_dir(set_kind(protocol, dev=kw.get("dev", False)), set_key)
    return tr.run_set(
        paths,
        fdir,
        set_dir,
        cfg,
        data_cfg=data_cfg,
        protocol=protocol,
        seeds=list(seeds),
        run_keys=keys,
        params=tr.effective_params(cfg, protocol, None),
        final=final,
        device="cpu",
        runtime=cfg["runtime"],
        budget_s=budget_s,
        test_bounds=TEST_BOUNDS if protocol == "lookahead" else ("end",),
        **kw,
    )


def run_dir(env, protocol="causal", set_key="set-a", seed=0) -> Path:
    return env[0].gnn_run_dir(f"gnn-test-{protocol}-{set_key}-s{seed}")


def stable_history(path: Path) -> list[dict]:
    """History without wall times and the val guard (a resumed call re-streams the val cache,
    so it checks those edges again; the train guard must match)."""
    out = []
    for r in read_jsonl(path):
        d = {k: v for k, v in r.items() if k not in VOLATILE and k != "guard"}
        d["train_guard"] = r["guard"]["train"]
        out.append(d)
    return out


def tensors_equal(a, b) -> bool:
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(tensors_equal(a[k], b[k]) for k in a)
    if isinstance(a, list | tuple):
        return len(a) == len(b) and all(tensors_equal(x, y) for x, y in zip(a, b, strict=True))
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    return a == b


def three_epoch_cfg(env) -> dict:
    cfg = copy.deepcopy(env[2])
    cfg["train"].update(max_epochs=3, min_epochs=3, patience=3)
    return cfg


# --- pure helpers ---------------------------------------------------------------------------------


def test_effective_params_and_class_weights(env):
    cfg = env[2]
    base = tr.effective_params(cfg, "causal", None)
    assert base == {
        "conv": "gine",
        "hidden": 8,
        "layers": 2,
        "layer_dropout": cfg["model"]["layer_dropout"],
        "final_dropout": cfg["train"]["final_dropout"],
        "lr": cfg["train"]["lr"],
        "w_pos": cfg["train"]["w_pos"],
        "neg_rate": cfg["train"]["neg_rate"],
    }
    best = {"lr": 0.01, "final_dropout": 0.2, "w_pos": 3.0}
    assert tr.effective_params(cfg, "lookahead", best) == {**base, **best}
    with pytest.raises(ValueError):
        tr.effective_params(cfg, "causal", {"lr": 0.01})
    pna = tr.effective_params(cfg, "pna", best)  # HPO's values never reach PNA
    assert pna["conv"] == "pna" and pna["hidden"] == 10 and pna["towers"] == 5
    assert pna["lr"] == cfg["protocols"]["pna"]["lr"]
    f = tr.effective_params(cfg, "faithful", None)
    assert f["neg_rate"] == 1.0 and f["w_pos"] == cfg["protocols"]["faithful"]["w_pos"]
    assert tr.class_weights(0.1, 6.275) == pytest.approx((1.0, 0.6275))
    assert tr.class_weights(1.0, 6.275) == (1.0, 6.275)  # faithful: Multi-GNN's [1, w_pos]
    for bad in ((0.0, 6.0), (1.5, 6.0), (0.1, 0.0)):
        with pytest.raises(ValueError):
            tr.class_weights(*bad)


def test_early_stopping_rule_on_scripted_metrics():
    kw = {"max_epochs": 10, "min_epochs": 3, "patience": 2}
    best, best_epoch, bad, stopped_at, improved = -np.inf, None, 0, None, []
    for epoch, m in enumerate([0.10, 0.30, 0.30, 0.20, 0.25, 0.9, 0.95]):
        imp, best, best_epoch, bad = tr.early_stop_update(best, best_epoch, bad, m, epoch)
        improved.append(imp)
        if tr.should_stop(epoch, bad, **kw):
            stopped_at = epoch
            break
    # a tie (epoch 2) is no improvement: the earlier epoch stays best; two bad epochs after
    # epoch 1 -> stop after epoch 3 (epoch + 1 = 4 >= min_epochs 3)
    assert improved == [True, True, False, False]
    assert (best_epoch, best, stopped_at) == (1, 0.30, 3)
    # min_epochs holds the stop back; max_epochs always stops
    assert not tr.should_stop(0, 5, max_epochs=10, min_epochs=3, patience=1)
    assert tr.should_stop(9, 0, max_epochs=10, min_epochs=3, patience=1)
    assert not tr.should_stop(5, 9, max_epochs=10, min_epochs=1, patience=1, early_stopping=False)
    # NaN (no positives) never beats a finite metric, but the first epoch is always the best
    assert tr.early_stop_update(-np.inf, None, 0, float("nan"), 0)[:3] == (True, -np.inf, 0)
    assert tr.early_stop_update(0.2, 0, 0, float("nan"), 1) == (False, 0.2, 0, 1)
    assert tr.early_stop_update(-np.inf, 0, 1, 0.1, 2) == (True, 0.1, 2, 0)


def test_is_transient_classification():
    assert tr.is_transient(RuntimeError("DataLoader timed out after 120 seconds"))
    assert tr.is_transient(OSError("EIO"))
    assert tr.is_transient(ConnectionError("reset"))
    assert tr.is_transient(TimeoutError())
    assert tr.is_transient(FileNotFoundError("x"))  # an OSError: the Volume may lag
    for exc in (
        LeakError("future edge", {"violations": 1}),
        ValueError("x"),
        AssertionError(),
        KeyError("k"),
        RuntimeError("CUDA error: an illegal memory access"),
        torch.cuda.OutOfMemoryError("CUDA out of memory"),
        tr.GnnOOMError("minimal split", 10, 1),
    ):
        assert not tr.is_transient(exc), exc
    import pickle

    e = pickle.loads(pickle.dumps(tr.GnnOOMError("m", 12, 3)))
    assert (e.edges, e.subgraphs, str(e)) == (12, 3, "m")


def test_save_torch_atomic_leaves_no_tmp(tmp_path, monkeypatch):
    p = tr.save_torch_atomic({"a": torch.arange(3)}, tmp_path / "x.pt")
    assert torch.equal(tr.load_torch(p)["a"], torch.arange(3))

    def boom(obj, f):
        f.write(b"partial")
        raise RuntimeError("disk full")

    monkeypatch.setattr(torch, "save", boom)
    with pytest.raises(RuntimeError):
        tr.save_torch_atomic({"a": 1}, tmp_path / "x.pt")
    assert sorted(f.name for f in tmp_path.iterdir()) == ["x.pt"]  # the old file survives


# --- the training step: loss normaliser, split, OOM -------------------------------------------


class SumModel(torch.nn.Module):
    """A BatchNorm/dropout-free model with MultiGINe's signature: per-subgraph local, so a
    split batch must give exactly the unsplit gradients."""

    def __init__(self, n_cols: int) -> None:
        super().__init__()
        self.e = torch.nn.Linear(n_cols, 4)
        self.x = torch.nn.Linear(1, 4)
        self.out = torch.nn.Linear(12, 2)

    def forward(self, x, ei_fwd, ea_fwd, ei_rev, ea_rev, tgt_src, tgt_dst, tgt_attr):
        h = self.x(x)
        agg = torch.zeros_like(h).index_add(0, ei_fwd[1], self.e(ea_fwd))
        agg = agg.index_add(0, ei_rev[1], self.e(ea_rev))
        h = torch.relu(h + agg)
        return self.out(torch.cat([h[tgt_src], h[tgt_dst], self.e(tgt_attr)], 1))


class OOMAbove(torch.nn.Module):
    """Raises a CUDA OOM when a forward gets more than `limit` edges (simulated memory)."""

    def __init__(self, inner, limit: int) -> None:
        super().__init__()
        self.inner, self.limit, self.ooms = inner, limit, 0

    def forward(self, *a):
        if a[1].shape[1] + a[3].shape[1] > self.limit:
            self.ooms += 1
            raise torch.cuda.OutOfMemoryError("simulated CUDA OOM")
        return self.inner(*a)


@pytest.fixture
def engine(env):
    from aml.models.gnn import graph as G

    paths, fdir, cfg, data_cfg = env
    g = G.load_graph(
        paths, fdir, cfg, data_cfg=data_cfg, label_splits=("train", "val_early"), protocol="causal"
    )
    return tr.TemporalEngine(
        g,
        cfg,
        protocol="causal",
        params=tr.effective_params(cfg, "causal", None),
        device="cpu",
        runtime=cfg["runtime"],
        allowed_splits=(),
    )


def first_batch(engine):
    loader, sampler = engine.train_loader(0)
    sampler.set_epoch(0)
    fbs = list(loader)
    # the busiest batch, so a cap splits it into several parts
    return max(fbs, key=lambda fb: int(fb.gid_fwd.numel() + fb.gid_rev.numel()))


def grads_after(model_fn, fb, engine, cap, y_host=None):
    model = model_fn()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)  # lr 0: keep the gradients to compare
    loss, ooms = tr.train_batch(
        model, opt, fb, ea=engine.ea, y=engine.y, cw=engine.cw, cap=cap, device="cpu", y_host=y_host
    )
    return {n: p.grad.clone() for n, p in model.named_parameters()}, float(loss), ooms


def test_loss_weights_and_full_batch_normaliser_under_split(engine):
    from aml.models.gnn import transforms as T

    fb = first_batch(engine)
    n_cols = engine.ea.shape[1]
    torch.manual_seed(0)
    proto = SumModel(n_cols)

    def make():
        return copy.deepcopy(proto)

    edges = int(fb.gid_fwd.numel() + fb.gid_rev.numel())
    cap = max(1, edges // 4)
    assert len(T.split_flat_batch(fb, cap)) >= 2
    g_full, loss_full, _ = grads_after(make, fb, engine, None)
    g_split, loss_split, _ = grads_after(make, fb, engine, cap, y_host=engine.g.y)
    assert loss_split == pytest.approx(loss_full, rel=1e-5)
    for k in g_full:
        torch.testing.assert_close(g_split[k], g_full[k], rtol=1e-5, atol=1e-7)
    # the unsplit loss is PyTorch's weighted mean with weights [1, w_pos * r]
    w = tr.class_weights(engine.params["neg_rate"], engine.params["w_pos"])
    assert torch.allclose(engine.cw, torch.tensor(w))
    with torch.no_grad():
        logits = make()(*T.to_model_inputs(fb, engine.ea, "cpu"))
        y = T.batch_labels(fb, engine.y)
        ref = torch.nn.functional.cross_entropy(logits, y, weight=engine.cw)
        mine = tr.weighted_loss(logits, y, engine.cw, engine.cw[y].sum())
    torch.testing.assert_close(mine, ref)
    assert loss_full == pytest.approx(float(ref), rel=1e-6)
    # a per-part normaliser would be wrong: the parts' own weights differ from the batch's
    parts = T.split_flat_batch(fb, cap)
    per_part = [engine.cw[T.batch_labels(p, engine.y)].sum() for p in parts]
    assert float(sum(per_part)) == pytest.approx(float(engine.cw[y].sum()))


def test_training_targets_are_train_split_rows_only(engine):
    """review LEAK-2: every target of a train epoch (positives and the negatives drawn) is a
    train-split row; g.y also holds val_early labels, which select epochs and HPO trials and
    must never be trained on. The engines refuse train seeds outside the train split."""
    from aml.data.split import SPLITS

    g = engine.g
    loader, sampler = engine.train_loader(0)
    sampler.set_epoch(0)
    tg = np.concatenate([fb.tgt_gid.numpy() for fb in loader])
    assert len(tg) and (g.split_code[tg] == SPLITS.index("train")).all()
    assert (g.split_code == SPLITS.index("val_early")).any()  # labels the loader must not use
    with pytest.raises(LeakError, match="outside the train split"):
        tr._require_train_seeds(g, np.flatnonzero(g.split_code <= 1))
    tr._require_train_seeds(g, engine.train_gids)


def test_host_label_check_equals_the_device_one(engine):
    from aml.models.gnn import transforms as T

    fb = first_batch(engine)
    a = tr.batch_labels_nosync(fb, engine.y, engine.g.y)
    assert a.dtype == torch.int64 and torch.equal(a, T.batch_labels(fb, engine.y))
    y_host = engine.g.y.copy()
    y_host[int(fb.tgt_gid[0])] = -1  # a target whose label was not loaded
    with pytest.raises(ValueError, match="no loaded label"):
        tr.batch_labels_nosync(fb, engine.y, y_host)


def test_train_batch_oom_splits_and_continues(engine):
    fb = first_batch(engine)
    edges = int(fb.gid_fwd.numel() + fb.gid_rev.numel())
    torch.manual_seed(0)
    proto = SumModel(engine.ea.shape[1])
    g_full, loss_full, _ = grads_after(lambda: copy.deepcopy(proto), fb, engine, None)
    wrapped = OOMAbove(copy.deepcopy(proto), limit=edges // 3)
    g_oom, loss_oom, ooms = grads_after(lambda: wrapped, fb, engine, None)
    assert ooms >= 1 and wrapped.ooms >= 1
    assert loss_oom == pytest.approx(loss_full, rel=1e-5)
    for k in g_full:
        torch.testing.assert_close(g_oom[f"inner.{k}"], g_full[k], rtol=1e-5, atol=1e-7)
    # a single subgraph that still does not fit -> GnnOOMError (fail fast / HPO FAIL)
    with pytest.raises(tr.GnnOOMError) as ei:
        grads_after(lambda: OOMAbove(copy.deepcopy(proto), limit=0), fb, engine, None)
    assert ei.value.edges == edges


# --- run_set: a whole set on the fixture ----------------------------------------------------------


def test_causal_final_set_scores_guard_and_layout(env):
    res = run(env, seeds=(0, 1), final=True)
    assert res["status"] == "done" and res["set_complete"], res
    assert res["seeds_trained"] == [0, 1] and res["seeds_scored"] == [0, 1]
    paths = env[0]
    set_dir = paths.gnn_set_dir("gnn_causal", "set-a")
    s = json.loads((set_dir / SUMMARY_FILE).read_text("utf-8"))
    assert (
        s["final"] and s["report_hash"] and s["scored_splits"] == ["val_early", "val_late", "test"]
    )
    assert s["violations"] == 0 and s["target_hits"] == 0 and s["edges_checked"] > 0
    assert set(s["guard"]) == {"train", "val_early", "val_late", "test"}
    assert all(g["violations"] == 0 and g["target_hits"] == 0 for g in s["guard"].values())
    assert s["features_digest"] and s["per_seed"].keys() == {"0", "1"}
    assert s["best_val_ap_mean_fresh"] == s["per_seed"]["1"]["best_val_ap"]  # seed 0 = HPO's
    df = pl.read_parquet(set_dir / SCORES_FILE)
    assert df.columns == ["row_id", "split", "score_s0", "score_s1"]
    assert df.schema["row_id"] == pl.Int64 and df.schema["score_s0"] == pl.Float64
    tx = pl.read_parquet(paths.transactions, columns=["row_id", "split"])
    want = tx.filter(pl.col("split") != "train")
    assert df.height == want.height and set(df["row_id"]) == set(want["row_id"])
    assert df.select(pl.col("score_s0", "score_s1").is_finite().all()).row(0) == (True, True)
    assert df["split"].to_list() == sorted(
        df["split"].to_list(), key=["val_early", "val_late", "test"].index
    )
    # the stage files: data version stamp, calls, lease released, no temp files anywhere
    assert (set_dir / "data_version.json").exists() and not (set_dir / WRITER_FILE).exists()
    assert len(read_jsonl(set_dir / tr.CALLS_FILE)) == 1
    assert not [p for p in paths.root.rglob(".*.tmp-*")]
    for seed in (0, 1):
        d = run_dir(env, seed=seed)
        assert {p.name for p in d.iterdir()} >= {
            "checkpoint.json",
            "graph_meta.json",
            LAST_CKPT,
            BEST_CKPT,
            HISTORY_FILE,
            SEED_SUMMARY_FILE,
            scores_file(seed),
            tr.SCORING_FILE,
        }
        assert not (d / RUNNING_FILE).exists() and not (d / FAILED_FILE).exists()
        ss = json.loads((d / SEED_SUMMARY_FILE).read_text("utf-8"))
        assert ss["epochs_run"] == len(read_jsonl(d / HISTORY_FILE)) <= 2
        assert ss["guard"]["train"]["edges_checked"] > 0
    # a finished set short-circuits without loading the graph
    from aml.models.gnn import graph as G

    def no_load(*a, **k):
        raise AssertionError("graph loaded for a finished set")

    orig = G.load_graph
    G.load_graph = no_load
    try:
        again = run(env, seeds=(0, 1), final=True)
    finally:
        G.load_graph = orig
    assert again["status"] == "done" and again["summary"] == s


def test_a_summary_of_other_feature_parts_does_not_short_circuit(env, monkeypatch):
    assert run(env)["status"] == "done"
    set_dir = env[0].gnn_set_dir("gnn_causal", "set-a")
    doc = json.loads((set_dir / SUMMARY_FILE).read_text("utf-8"))
    current = doc["features_digest"]
    (set_dir / SUMMARY_FILE).write_text(json.dumps({**doc, "features_digest": "old"}), "utf-8")
    trained: list[int] = []
    real = tr.TemporalEngine.train_epoch

    def spy(self, model, opt, seed, epoch):
        trained.append(epoch)
        return real(self, model, opt, seed, epoch)

    monkeypatch.setattr(tr.TemporalEngine, "train_epoch", spy)
    res = run(env)
    assert res["status"] == "done" and res["summary"]["features_digest"] == current
    assert trained == []  # the run dirs match the current parts: only re-assembled


def test_final_false_writes_no_test_rows_and_reads_no_late_labels(env, monkeypatch):
    import aml.models.lgbm as lgbm
    from aml.models.gnn import sampler as S

    paths = env[0]
    tx = pl.read_parquet(paths.transactions, columns=["row_id", "split"])
    allowed = set(tx.filter(pl.col("split").is_in(["train", "val_early"]))["row_id"])
    seen_rows: list[int] = []
    real = lgbm.load_labels

    def spy(path, row_ids):
        seen_rows.extend(row_ids.to_list())
        return real(path, row_ids)

    monkeypatch.setattr(lgbm, "load_labels", spy)
    loaders: list[int] = []
    real_make = S.make_loader

    def spy_make(g, hetero, seed_gids, **kw):
        loaders.append(int(np.asarray(seed_gids).size))
        if kw["role"] == "eval":
            splits = set(g.split_code[np.asarray(seed_gids)].tolist())
            assert splits <= {1, 2}, f"an eval loader over split codes {splits}"
        return real_make(g, hetero, seed_gids, **kw)

    monkeypatch.setattr(S, "make_loader", spy_make)
    res = run(env, final=False)
    assert res["status"] == "done" and res["set_complete"]
    assert seen_rows and set(seen_rows) <= allowed
    assert loaders  # the spy saw the loaders
    monkeypatch.setattr(S, "make_loader", real_make)
    df = pl.read_parquet(paths.gnn_set_dir("gnn_causal", "set-a") / SCORES_FILE)
    assert set(df["split"]) == {"val_early", "val_late"}
    assert set(pl.read_parquet(run_dir(env) / scores_file(0))["split"]) == {"val_early", "val_late"}
    s = json.loads((paths.gnn_set_dir("gnn_causal", "set-a") / SUMMARY_FILE).read_text("utf-8"))
    assert not s["final"] and s["report_hash"] is None
    # a later final submission of the same set scores test once and keeps the val rows
    before = pl.read_parquet(run_dir(env) / scores_file(0)).filter(pl.col("split") != "test")
    res = run(env, final=True)
    assert res["status"] == "done" and res["summary"]["final"]
    after = pl.read_parquet(run_dir(env) / scores_file(0))
    assert set(after["split"]) == {"val_early", "val_late", "test"}
    assert after.filter(pl.col("split") != "test").equals(before)
    with pytest.raises(ValueError):  # dev never scores test
        tr._check_run_set_args(
            env[2], "causal", [0], {0: "k"}, res["summary"]["params"], True, True, ("end",)
        )


def test_score_files_written_once_and_set_assembled_only_when_complete(env, monkeypatch):
    paths = env[0]
    set_dir = paths.gnn_set_dir("gnn_causal", "set-a")
    first = run(env, seeds=(0,), set_seeds=(0, 1))
    assert first["status"] == "done" and not first["set_complete"] and first["summary"] is None
    assert not (set_dir / SUMMARY_FILE).exists() and not (set_dir / SCORES_FILE).exists()
    f0 = run_dir(env, seed=0) / scores_file(0)
    stamp, content = f0.stat().st_mtime_ns, f0.read_bytes()
    scored: list[list[int]] = []
    real_score = tr.TemporalEngine.score

    def spy(self, models, unit):
        scored.append(sorted(models))
        return real_score(self, models, unit)

    monkeypatch.setattr(tr.TemporalEngine, "score", spy)
    time.sleep(0.01)
    second = run(env, seeds=(1,), set_seeds=(0, 1))
    assert second["status"] == "done" and second["set_complete"]
    assert second["seeds_trained"] == [1] and scored == [[1], [1]]  # val_early, val_late
    assert f0.stat().st_mtime_ns == stamp and f0.read_bytes() == content
    df = pl.read_parquet(set_dir / SCORES_FILE)
    assert df.columns == ["row_id", "split", "score_s0", "score_s1"]
    # a re-run with the set summary removed re-assembles without any scoring pass
    (set_dir / SUMMARY_FILE).unlink()
    scored.clear()
    third = run(env, seeds=(0, 1))
    assert third["status"] == "done" and third["set_complete"] and scored == []
    assert pl.read_parquet(set_dir / SCORES_FILE).equals(df)


def test_lookahead_d10_file_layout(env):
    res = run(env, protocol="lookahead", final=True)
    assert res["status"] == "done" and res["set_complete"], res
    paths = env[0]
    d = run_dir(env, "lookahead")
    end = pl.read_parquet(d / scores_file(0, "end"))
    d10 = pl.read_parquet(d / scores_file(0, "d10"))
    assert set(end["split"]) == {"val_early", "val_late", "test"} and set(d10["split"]) == {"test"}
    assert d10.height == end.filter(pl.col("split") == "test").height
    la_dir = paths.gnn_set_dir("gnn_lookahead", "set-a")
    d10_dir = paths.gnn_set_dir(LOOKAHEAD_D10_MODEL, "set-a")
    la = pl.read_parquet(la_dir / SCORES_FILE)
    l10 = pl.read_parquet(d10_dir / SCORES_FILE)
    assert la.select("row_id", "split").equals(l10.select("row_id", "split"))
    val = pl.col("split") != "test"
    assert la.filter(val).equals(l10.filter(val))  # the same val scores
    assert l10.filter(~val)["score_s0"].to_list() == d10["score_s0"].to_list()
    s = json.loads((la_dir / SUMMARY_FILE).read_text("utf-8"))
    s10 = json.loads((d10_dir / SUMMARY_FILE).read_text("utf-8"))
    assert s["model"] == "gnn_lookahead" and s10["model"] == LOOKAHEAD_D10_MODEL
    assert s10["views"] == ["primary"] and s10["test_bound"] == "d10"
    assert "test_d10" in s10["guard"] and "test" not in s10["guard"]
    assert "test" in s["guard"] and "test_d10" not in s["guard"]
    assert s["violations"] == 0 and s10["violations"] == 0
    assert s["dropped_target_copies"] > 0  # look-ahead step 1 drops the target's own edge
    assert set(s["future_share"]) == set(s["guard"]) and s["future_share"]["train"] > 0
    # review LEAK-1: the d10 test pass really ran under the d10 bound (a regression to `end`
    # would make the two passes identical while every guard still reads 0 violations)
    assert s10["guard"]["test_d10"]["future_edges"] < s["guard"]["test"]["future_edges"]
    end_t = end.filter(pl.col("split") == "test").sort("row_id")
    d10_t = d10.sort("row_id")
    assert end_t["row_id"].to_list() == d10_t["row_id"].to_list()
    assert (end_t["score_s0"] != d10_t["score_s0"]).any()
    meta = json.loads((d / tr.SCORING_FILE).read_text("utf-8"))
    assert meta["test_d10"]["bounds"]["d10_last"] is not None
    # a d10 file scored under another d10 bound is refused, never silently re-touched
    meta["test_d10"]["bounds"]["d10_last"] = -5
    (d / tr.SCORING_FILE).write_text(json.dumps(meta), "utf-8")
    (la_dir / SUMMARY_FILE).unlink()
    again = run(env, protocol="lookahead", final=True)
    assert again["status"] == "failed" and "d10_last" in again["error"]


def test_faithful_set(env):
    res = run(env, protocol="faithful", final=True)
    assert res["status"] == "done" and res["set_complete"], res
    paths = env[0]
    set_dir = paths.gnn_set_dir("gnn_faithful", "set-a")
    df = pl.read_parquet(set_dir / SCORES_FILE)
    assert df.columns == ["row_id", "split", "sampled", "score_s0"]
    assert df.schema["sampled"] == pl.Boolean
    assert set(df["split"]) == {"val_early", "val_late", "test"}
    assert df["sampled"].any() and df["score_s0"].is_finite().all()
    s = json.loads((set_dir / SUMMARY_FILE).read_text("utf-8"))
    assert s["model"] == "gnn_faithful" and s["epochs_run"] == 1 and s["max_epochs"] == 1
    assert s["exemptions"]["features"] == ["timestamp"]
    assert s["exemptions"]["norm"] == "per_snapshot"
    assert set(s["sampled_share"]) == {"val_early", "val_late", "test"}
    assert s["violations"] == 0 and s["recalled"]
    assert set(s["guard"]) == {"train", "val", "test"}
    h = read_jsonl(run_dir(env, "faithful") / HISTORY_FILE)
    assert h[0]["val_f1_sampled"] is None or 0 <= h[0]["val_f1_sampled"] <= 1


def test_faithful_train_batches_stay_in_the_train_snapshot(env, monkeypatch):
    """review LEAK-3: every faithful training batch samples ranks <= train_last (the train
    graph = train edges only, as published); the loader's guard bound comes from the seeds'
    split, so a train loader over the val snapshot raises instead of training on days 7-8."""
    seen: list[int] = []
    orig = tr.faithful_train_batch

    def spy(model, opt, fb, **kw):
        for gid in (fb.gid_fwd, fb.gid_rev):
            if int(gid.numel()):
                seen.append(int(gid.max()))
        return orig(model, opt, fb, **kw)

    monkeypatch.setattr(tr, "faithful_train_batch", spy)
    res = run(env, protocol="faithful", final=True)
    assert res["status"] == "done" and res["set_complete"], res
    bounds = res["summary"]["bounds"]
    assert seen and max(seen) <= bounds["train_last"] < bounds["val_last"]


def test_pna_and_dev_sets(env):
    res = run(env, protocol="pna", final=False)
    assert res["status"] == "done" and res["set_complete"], res
    s = res["summary"]
    assert s["params"]["conv"] == "pna" and s["violations"] == 0
    dev = run(env, protocol="causal", dev=True, max_epochs=1, set_key="dev-a")
    assert dev["status"] == "done" and dev["summary"]["model"] == "gnn_dev"
    assert dev["summary"]["per_seed"]["0"]["epochs_run"] == 1
    assert not dev["summary"]["final"]


# --- resume, the money guards ---------------------------------------------------------------


@pytest.mark.parametrize("workers", [0, 2])
def test_resume_after_kill_equals_uninterrupted_run(prepared, cfgs, tmp_path, monkeypatch, workers):
    """3 epochs straight == 2 epochs + a kill (BaseException, as a preemption) + resume:
    state_dict, optimizer, history and scores bitwise equal on CPU, also with loader worker
    processes (persistent train workers draw their base seed only at the first epoch of a
    process, so the per-epoch reseed must come after iter(loader))."""
    monkeypatch.setattr(tr, "MLFLOW_ENABLED", False)
    env_a = (*gnn_inputs(prepared, tmp_path / "a", cfgs), cfgs["data"])
    env_b = (*gnn_inputs(prepared, tmp_path / "b", cfgs), cfgs["data"])
    cfg = three_epoch_cfg(env_a)
    cfg["runtime"].update(num_workers=workers, loader_timeout_s=60)
    assert run(env_a, cfg=cfg)["status"] == "done"

    real = tr.TemporalEngine.train_epoch

    def killed_at_2(self, model, opt, seed, epoch):
        if epoch == 2:
            raise KeyboardInterrupt("preempted")
        return real(self, model, opt, seed, epoch)

    monkeypatch.setattr(tr.TemporalEngine, "train_epoch", killed_at_2)
    with pytest.raises(KeyboardInterrupt):
        run(env_b, cfg=cfg)
    rb = run_dir(env_b)
    assert json.loads((rb / RUNNING_FILE).read_text("utf-8")) == {"epoch_next": 2, "starts": 1}
    assert tr.load_torch(rb / LAST_CKPT)["epoch_done"] == 1
    monkeypatch.setattr(tr.TemporalEngine, "train_epoch", real)
    res = run(env_b, cfg=cfg)  # the retry: same call id, so it takes its lease back
    assert res["status"] == "done", res
    ra = run_dir(env_a)
    a, b = tr.load_torch(ra / LAST_CKPT), tr.load_torch(rb / LAST_CKPT)
    assert a["epoch_done"] == b["epoch_done"] == 2
    assert tensors_equal(a["model"], b["model"])
    assert tensors_equal(a["optimizer"], b["optimizer"])
    assert tensors_equal(tr.load_torch(ra / BEST_CKPT), tr.load_torch(rb / BEST_CKPT))
    assert stable_history(ra / HISTORY_FILE) == stable_history(rb / HISTORY_FILE)
    assert len(read_jsonl(rb / HISTORY_FILE)) == 3
    sa = pl.read_parquet(ra / scores_file(0))
    sb = pl.read_parquet(rb / scores_file(0))
    assert sa.equals(sb)


def test_wall_budget_partial_then_resume_completes(prepared, cfgs, tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "MLFLOW_ENABLED", False)
    env_a = (*gnn_inputs(prepared, tmp_path / "a", cfgs), cfgs["data"])
    env_b = (*gnn_inputs(prepared, tmp_path / "b", cfgs), cfgs["data"])
    straight = run(env_a, seeds=(0, 1))
    assert straight["status"] == "done"
    statuses, nexts = [], []
    for _ in range(10):
        r = run(env_b, seeds=(0, 1), budget_s=0.0)  # every call runs exactly one unit
        statuses.append(r["status"])
        nexts.append(r["next"])
        if r["status"] != "partial":
            break
    # seed 0: epochs 0, 1; seed 1: epochs 0, 1; then the scoring pass; then done
    assert statuses == ["partial"] * 4 + ["done"], statuses
    assert nexts[:4] == [
        {"seed": 0, "epoch": 1},
        {"seed": 1, "epoch": 0},
        {"seed": 1, "epoch": 1},
        {"phase": "scoring"},
    ]
    a = pl.read_parquet(env_a[0].gnn_set_dir("gnn_causal", "set-a") / SCORES_FILE)
    b = pl.read_parquet(env_b[0].gnn_set_dir("gnn_causal", "set-a") / SCORES_FILE)
    assert a.equals(b)
    calls = read_jsonl(env_b[0].gnn_set_dir("gnn_causal", "set-a") / tr.CALLS_FILE)
    assert [c["status"] for c in calls] == statuses


def test_clock_runs_at_least_one_unit():
    t = [0.0]
    c = tr.Clock(10.0, now=lambda: t[0])
    t[0] = 100.0
    assert not c.out_of_time(50.0)  # nothing done yet in this call: always run one unit
    c.unit_done()
    assert c.out_of_time(0.0)
    t[0] = 1.0
    assert not c.out_of_time(7.0) and c.out_of_time(7.3)  # 1 + 1.25 x 7.3 > 10


def test_fingerprint_mismatch_resets_the_run_dir(env):
    assert run(env, budget_s=0.0)["status"] == "partial"
    d = run_dir(env)
    assert tr.load_torch(d / LAST_CKPT)["epoch_done"] == 0
    fp = json.loads((d / "checkpoint.json").read_text("utf-8"))
    fp["features_digest"] = "other-parts"
    (d / "checkpoint.json").write_text(json.dumps(fp), "utf-8")
    (d / "stray.txt").write_text("from the other inputs", "utf-8")
    res = run(env)
    assert res["status"] == "done"
    assert not (d / "stray.txt").exists()  # emptied, then trained from epoch 0
    assert [r["epoch"] for r in read_jsonl(d / HISTORY_FILE)] == [0, 1]
    assert (
        json.loads((d / "checkpoint.json").read_text("utf-8"))["features_digest"] != "other-parts"
    )


def test_fail_fast_returns_failed_with_failed_json(env, monkeypatch):
    def boom(self, model):
        raise LeakError("a sampled edge after its bound", {"violations": 1})

    monkeypatch.setattr(tr.TemporalEngine, "validate", boom)
    res = run(env)
    assert res["status"] == "failed" and "LeakError" in res["error"]
    assert res["failed_run_key"] == run_dir(env).name
    d = run_dir(env)
    doc = json.loads((d / FAILED_FILE).read_text("utf-8"))
    assert doc["type"] == "LeakError" and "Traceback" in doc["traceback"]
    assert doc["epoch"] == 0 and doc["detail"] == {"violations": 1}
    assert not (d / RUNNING_FILE).exists()  # a failed return is a clean return
    set_dir = env[0].gnn_set_dir("gnn_causal", "set-a")
    assert not (set_dir / WRITER_FILE).exists()
    # transient errors propagate (Modal retries the input; the run resumes)

    def flaky(self, model):
        raise OSError("Volume I/O error")

    monkeypatch.setattr(tr.TemporalEngine, "validate", flaky)
    with pytest.raises(OSError):
        run(env)
    # deterministic errors outside a run (bad arguments) -> FAILED.json in the set dir
    paths, fdir, cfg, data_cfg = env
    res = tr.run_set(
        paths,
        fdir,
        paths.gnn_set_dir("gnn_causal", "set-b"),
        cfg,
        data_cfg=data_cfg,
        protocol="causal",
        seeds=[0],
        run_keys={0: "k0"},
        params={**tr.effective_params(cfg, "causal", None), "hidden": 99},
        final=False,
        device="cpu",
        runtime=cfg["runtime"],
        budget_s=1e9,
    )
    assert res["status"] == "failed" and "effective_params" in res["error"]
    assert (paths.gnn_set_dir("gnn_causal", "set-b") / FAILED_FILE).exists()


def test_unclean_start_counter_stops_at_the_third_start(env, monkeypatch):
    calls = []

    def crash(self, model, opt, seed, epoch):
        calls.append(epoch)
        raise KeyboardInterrupt("container killed")

    monkeypatch.setattr(tr.TemporalEngine, "train_epoch", crash)
    for _ in range(2):
        with pytest.raises(KeyboardInterrupt):
            run(env)
    d = run_dir(env)
    assert json.loads((d / RUNNING_FILE).read_text("utf-8")) == {"epoch_next": 0, "starts": 2}
    res = run(env)
    assert res["status"] == "failed" and "no progress at epoch 0" in res["error"]
    assert calls == [0, 0]  # the third start trains nothing
    assert "no progress" in json.loads((d / FAILED_FILE).read_text("utf-8"))["error"]
    assert not (d / RUNNING_FILE).exists()


def test_writer_lease_busy_and_stopped(env):
    set_dir = env[0].gnn_set_dir("gnn_causal", "set-a")
    set_dir.mkdir(parents=True)
    lease = {"call_id": "fc-other", "heartbeat": time.time()}
    (set_dir / WRITER_FILE).write_text(json.dumps(lease), "utf-8")
    res = run(env)
    assert res["status"] == "busy" and "fc-other" in res["error"]
    assert not run_dir(env).exists()  # nothing was trained
    lease["heartbeat"] = time.time() - tr.LEASE_FRESH_S - 1  # a stale lease is taken over
    (set_dir / WRITER_FILE).write_text(json.dumps(lease), "utf-8")
    (set_dir / STOPPED_FILE).write_text("{}", "utf-8")
    assert run(env)["status"] == "stopped"
    (set_dir / STOPPED_FILE).unlink()
    assert run(env)["status"] == "done"
    assert not (set_dir / WRITER_FILE).exists()


def test_oom_in_a_run_splits_and_continues(env, monkeypatch):
    from aml.models.gnn import model as M

    real = M.build_model
    wrapped = []

    def build(*a, **k):
        m = OOMAbove(real(*a, **k), limit=300)
        wrapped.append(m)
        return m

    monkeypatch.setattr(M, "build_model", build)
    res = run(env)
    assert res["status"] == "done", res
    hist = read_jsonl(run_dir(env) / HISTORY_FILE)
    assert sum(r["oom_splits"] for r in hist) > 0 and wrapped[0].ooms > 0
    s = json.loads((run_dir(env) / SEED_SUMMARY_FILE).read_text("utf-8"))
    assert s["oom_splits"] == sum(r["oom_splits"] for r in hist)


def test_mlflow_run_id_reused_on_resume(env, tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "MLFLOW_ENABLED", True)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    assert run(env, budget_s=0.0)["status"] == "partial"
    from aml import tracking

    key = run_dir(env).name
    first = tracking.find_run_for_key(tr.EXPERIMENT, key)
    assert first is not None
    assert run(env, final=True)["status"] == "done"
    client = tracking._client()
    exp = client.get_experiment_by_name(tr.EXPERIMENT)
    runs = client.search_runs([exp.experiment_id], f"tags.run_key = '{key}'")
    assert [r.info.run_id for r in runs] == [first]
    steps = [m.step for m in client.get_metric_history(first, "train_loss")]
    assert steps == [0, 1]
    tags = runs[0].data.tags
    assert tags["protocol"] == "causal" and tags.get("test_touch") == "true"
    assert os.environ["MLFLOW_TRACKING_URI"].startswith("file:")
