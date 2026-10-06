"""gnn_bench cells (M3 spec §10.1, §14.2 row E) on the synthetic fixture, on CPU: every field of
the costplan cell schema, the stress-batch selection, the faithful OOM fallback, guard totals,
resume by cell id, and the cells feeding costplan.decide."""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from aml.models.gnn import CELLS_FILE, read_jsonl  # noqa: E402
from aml.models.gnn import bench as B  # noqa: E402
from aml.models.gnn import costplan as C  # noqa: E402
from aml.models.gnn import train as tr  # noqa: E402
from tests.conftest import load_yaml  # noqa: E402
from tests.fixtures.gnn_graphs import gnn_inputs, make_host_graph, random_tied_graph  # noqa: E402

COMMON = (
    "cell_id",
    "group",
    "gpu",
    "device_name",
    "device_bytes",
    "cores",
    "memory_mib",
    "batch_size",
    "num_workers",
    "status",
    "error",
    "seconds",
    "host_peak_mib",
    "host_peak_source",
    "host_peak_mib_proxy",
    "guard",
)
STEPS = (
    "warmup_steps",
    "timed_steps",
    "step_s_mean",
    "step_s_p90",
    "gpu_s_mean",
    "wait_s_mean",
    "wait_share",
    "edges_mean",
    "edges_max",
    "nodes_mean",
    "nodes_max",
    "peak_mem_bytes",
    "bytes_per_edge",
)
FIELDS = {
    "build": ("build_s", "n_nodes", "n_edges", "counts"),
    "grid": STEPS,
    "rerun": STEPS + ("val_s", "val_cached_s", "cache_fits"),
    "pna": STEPS,
    "lookahead": STEPS
    + (
        "val_batches",
        "val_first_batch_s",
        "val_s_per_batch",
        "edges_per_target_train",
        "edges_per_target_val",
        "future_share",
    ),
    "stress": ("edges", "peak_mem_bytes", "bytes_per_edge", "max_edges_per_step"),
    "val": (
        "val_s",
        "val_cached_s",
        "cache_gb",
        "cache_fits",
        "eval_batches",
        "eval_edges_max",
        "eval_peak_mem_bytes",
        "eval_bytes_per_edge",
        "max_edges_per_eval_step",
    ),
    "faithful": (
        "tried",
        "batch_size",
        "warmup_steps",
        "timed_steps",
        "step_s_mean",
        "edges_mean",
        "edges_max",
        "peak_mem_bytes",
        "val_s_per_batch",
        "test_s_per_batch",
        "val_first_batch_s",
        "test_first_batch_s",
        "val_edges_mean",
        "test_edges_mean",
        "sampled_share",
    ),
    "determinism": (
        "bit_deterministic",
        "max_param_diff",
        "det_overhead_ratio",
        "det_overhead_ratio_train",
        "det_overhead_ratio_eval",
        "det_overhead_ratio_faithful",
        "cpu_gpu_max_logit_diff",
    ),
}
SAMPLING = ("grid", "rerun", "pna", "lookahead", "stress", "val", "faithful", "determinism")


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


@pytest.fixture(scope="module")
def bench_run(prepared, cfgs, tmp_path_factory):
    """Every L4 group plus a rerun container on the fixture (CPU), shared by the tests."""
    mp = pytest.MonkeyPatch()
    mp.setattr(tr, "MLFLOW_ENABLED", False)
    det = torch.are_deterministic_algorithms_enabled()
    warn = torch.is_deterministic_algorithms_warn_only_enabled()
    paths, fdir, cfg = gnn_inputs(prepared, tmp_path_factory.mktemp("bench"), cfgs)
    cfg = copy.deepcopy(cfg)
    cfg["bench"]["lookahead"].update(timed_steps=3, val_batches=3)
    cfg["bench"]["faithful"].update(warmup_steps=1, timed_steps=2, val_batches=2, test_batches=2)
    cfg["bench"]["pna"].update(warmup_steps=1, timed_steps=2)
    bench_dir = paths.gnn_set_dir("gnn_bench", "gnn_bench-test")
    kw = dict(data_cfg=cfgs["data"], device="cpu", memory_mib=4096)
    cells = B.run_bench_container(
        paths, fdir, bench_dir, cfg, gpu="L4", cores=2, groups=C.CONTAINER_GROUPS["L4"], **kw
    )
    rerun = B.run_bench_container(
        paths,
        fdir,
        bench_dir,
        cfg,
        gpu="L4",
        cores=1,
        groups=C.CONTAINER_GROUPS["rerun"],
        rerun={"batch_size": 64, "num_workers": 0},
        **kw,
    )
    yield {"paths": paths, "fdir": fdir, "cfg": cfg, "dir": bench_dir, "kw": kw}, cells, rerun
    mp.undo()
    torch.use_deterministic_algorithms(det, warn_only=warn)


def test_every_cell_has_its_schema_and_a_clean_guard(bench_run):
    ctx, cells, rerun = bench_run
    by_group = {c["group"]: c for c in cells + rerun}
    assert set(by_group) == set(C.CONTAINER_GROUPS["L4"]) | {"rerun"}
    for c in cells + rerun:
        missing = [f for f in COMMON + FIELDS[c["group"]] if f not in c]
        assert not missing, (c["cell_id"], missing)
        assert c["status"] == "ok", (c["cell_id"], c["error"])
        # the faithful id is fixed before the fitting batch is known (stable on resume)
        b = None if c["group"] == "faithful" else c["batch_size"]
        assert c["cell_id"] == C.cell_id(c["gpu"], c["cores"], c["group"], b, c["num_workers"])
        g = c["guard"]
        assert g["violations"] == 0
        if c["group"] != "faithful":  # faithful keeps its targets (published protocol)
            assert g["target_hits"] == 0, c["cell_id"]
        if c["group"] in SAMPLING:
            assert g["edges_checked"] > 0, c["cell_id"]
    grid = [c for c in cells if c["group"] == "grid"]
    assert [(c["batch_size"], c["num_workers"]) for c in grid] == [(64, 0)]
    assert grid[0]["timed_steps"] == 2 and grid[0]["edges_mean"] > 0
    assert 0 <= grid[0]["wait_share"] <= 1
    b = by_group["build"]["counts"]
    assert b["n_pos"] > 0 and b["n_neg"] > 0 and b["n_test_d10"] == b["n_test"]
    assert by_group["val"]["cache_fits"] and by_group["val"]["val_cached_s"] is not None
    assert by_group["val"]["eval_batches"] >= 1
    f = by_group["faithful"]
    assert f["tried"] == [64] and f["batch_size"] == 64
    assert f["guard"]["target_hits"] > 0  # the sampled targets are in their batch graph
    assert set(f["sampled_share"]) == {"train", "val", "test"}
    la = by_group["lookahead"]
    assert la["future_share"] > 0 and la["dropped_target_copies"] > 0  # step-1 bounds
    d = by_group["determinism"]
    assert d["bit_deterministic"] is True and d["max_param_diff"] == 0.0  # CPU
    assert d["cpu_gpu_max_logit_diff"] is None  # no GPU here
    # review perf-1: eval forwards and faithful steps get their own ratio (rule 6)
    for k in ("train", "eval", "faithful"):
        assert d[f"det_overhead_ratio_{k}"] > 0, k
    assert d["det_overhead_ratio"] == d["det_overhead_ratio_train"]
    # review perf-4: a fresh eval loader's first batch is timed apart
    assert f["val_first_batch_s"] > 0 and f["test_first_batch_s"] > 0
    assert la["val_first_batch_s"] > 0
    assert by_group["pna"]["hidden"] == ctx["cfg"]["protocols"]["pna"]["hidden"]
    # CPU: no device memory figures
    assert (
        by_group["stress"]["bytes_per_edge"] is None and by_group["build"]["device_bytes"] is None
    )


def test_resume_skips_recorded_cells(bench_run, monkeypatch):
    ctx, cells, _ = bench_run
    from aml.models.gnn import graph as G

    def no_load(*a, **k):
        raise AssertionError("graph loaded although every cell is recorded")

    monkeypatch.setattr(G, "load_graph", no_load)
    again = B.run_bench_container(
        ctx["paths"],
        ctx["fdir"],
        ctx["dir"],
        ctx["cfg"],
        gpu="L4",
        cores=2,
        groups=C.CONTAINER_GROUPS["L4"],
        **ctx["kw"],
    )
    assert [c["cell_id"] for c in again] == [c["cell_id"] for c in cells]
    lines = read_jsonl(ctx["dir"] / CELLS_FILE)
    assert len(lines) == len({c["cell_id"] for c in lines})  # nothing recorded twice


def test_stress_batch_is_the_top_train_degree_seeds():
    ag = random_tied_graph(n_nodes=300, n_edges=3000, seed=3)
    m = ag.n_edges
    split = ["train"] * (m // 2) + ["val_early"] * (m - m // 2)
    g = make_host_graph(ag, split=split)
    train = np.arange(m // 2, dtype=np.int64)
    got = B.stress_gids(g, train, 64)
    # brute force: degree over train edges (in + out), score per edge, top 64, ties -> lower gid
    deg = {}
    for i in train:
        for a in (int(g.src[i]), int(g.dst[i])):
            deg[a] = deg.get(a, 0) + 1
    score = {int(i): deg[int(g.src[i])] + deg[int(g.dst[i])] for i in train}
    want = sorted(sorted(score, key=lambda i: (-score[i], i))[:64])
    assert got.tolist() == want and got.dtype == np.int64


def test_faithful_oom_fallback_records_the_batch_that_fits(bench_run, monkeypatch, tmp_path):
    ctx, _, _ = bench_run
    real = tr.faithful_train_batch
    seen: list[int] = []

    def oom_above_32(model, opt, fb, **kw):
        n = int(fb.tgt_gid.numel())
        seen.append(n)
        if n > 32:
            raise tr.GnnOOMError(f"simulated OOM at {n} targets", 10**6, 1)
        return real(model, opt, fb, **kw)

    monkeypatch.setattr(tr, "faithful_train_batch", oom_above_32)
    cells = B.run_bench_container(
        ctx["paths"],
        ctx["fdir"],
        tmp_path / "bench_oom",
        ctx["cfg"],
        gpu="T4",
        cores=2,
        groups=("faithful",),
        **ctx["kw"],
    )
    (f,) = cells
    assert f["status"] == "ok" and f["tried"] == [64, 32] and f["batch_size"] == 32
    assert max(seen) == 64

    def always(model, opt, fb, **kw):
        raise tr.GnnOOMError("simulated OOM", 1, 1)

    monkeypatch.setattr(tr, "faithful_train_batch", always)
    (f,) = B.run_bench_container(
        ctx["paths"],
        ctx["fdir"],
        tmp_path / "bench_oom2",
        ctx["cfg"],
        gpu="T4",
        cores=2,
        groups=("faithful",),
        **ctx["kw"],
    )
    assert f["status"] == "oom" and f["batch_size"] is None and f["tried"] == [64, 32]
    assert "no faithful batch fits" in f["error"]


def test_cells_feed_costplan_decide(bench_run):
    """On a GPU the cells carry device memory; emulate it on the CPU cells and decide."""
    ctx, cells, rerun = bench_run
    gpu_cells = []
    for c in copy.deepcopy(cells + rerun):
        c["device_bytes"] = 24 * 2**30
        if c["group"] in ("grid", "rerun", "stress"):
            c["peak_mem_bytes"] = 2 * 2**30
            c["bytes_per_edge"] = 6000.0
        if c["group"] == "stress":
            c["max_edges_per_step"] = int(0.6 * c["device_bytes"] / 6000.0)
        if c["group"] in ("val", "rerun"):
            c["eval_bytes_per_edge"] = 2500.0
            c["max_edges_per_eval_step"] = int(0.6 * c["device_bytes"] / 2500.0)
        gpu_cells.append(c)
    decision = C.decide(gpu_cells, ctx["cfg"])
    assert decision["gpu"] == "L4" and decision["batch_size"] == 64
    assert decision["max_edges_per_step"] == int(0.6 * 24 * 2**30 / 6000.0)
    assert decision["faithful"]["batch_size"] == 64
    assert decision["bit_deterministic"] is True
    assert (
        decision["counts"]["n_pos"]
        == next(c for c in cells if c["group"] == "build")["counts"]["n_pos"]
    )
    json.dumps(decision)  # decision.json is plain JSON
    assert C.rerun_4core(gpu_cells, ctx["cfg"]) is None or isinstance(
        C.rerun_4core(gpu_cells, ctx["cfg"]), dict
    )


def test_peak_host_mib_is_positive_on_linux():
    v = B.peak_host_mib()
    assert isinstance(v, float) and v >= 0.0


def test_cells_record_the_container_memory(bench_run):
    """review MONEY-2 / perf-6: host_peak_mib (rule 5's input) is the container's memory
    sampled during the cell (cgroup usage, else MemTotal - MemAvailable), never self + children
    ru_maxrss (a child spawned during `import torch` reports the parent's RSS); the old proxy is
    kept for reference."""
    _, cells, rerun = bench_run
    for c in cells + rerun:
        assert c["host_peak_source"] in ("cgroup", "meminfo", "ru_maxrss_self"), c["cell_id"]
        assert c["host_peak_mib"] > 0 and c["host_peak_mib_proxy"] >= 0
    v, src = B.container_memory_mib()
    assert (v is None) == (src is None) and (v is None or v > 0)
    # the window max: a sample at mark(), the poller's samples and one at peak()
    vals = iter([100.0, 300.0, 200.0, 50.0, 60.0])
    mp = B.MemPeak(poll_s=3600.0, read=lambda: (next(vals), "fake"))
    mp.mark()
    assert mp.peak() == (300.0, "fake")
    assert mp.peak() == (300.0, "fake")  # 200 < 300
    mp.mark()  # a new window (one per cell)
    assert mp.peak() == (60.0, "fake")
    mp.stop()
    none = B.MemPeak(read=lambda: (None, None))
    none.mark()
    assert none.peak() == (None, None)
    none.stop()
