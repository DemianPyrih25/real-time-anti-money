"""gnn_bench cells on real HI-Small batches (M3 spec §10.1; library owner E, job owner F).

One GPU container per call runs a list of cell groups (costplan.CONTAINER_GROUPS: L4 -> all
groups; T4 -> build, grid, stress, val, faithful; the optional 4-core rerun -> build, rerun).
Every cell runs the §5.5 guard (0 violations required: a violation raises LeakError), appends one
JSON line to `<bench_dir>/cells.jsonl` (aml.models.gnn.append_jsonl, then on_checkpoint) and is
skipped on resume when a line with its `cell_id` already exists. The cell record schema is in
costplan's module docstring; `costplan.decide` consumes the records on the CPU driver.

Every train step is the production step (train.train_batch / train.faithful_train_batch), and
every eval forward is train.eval_parts. The bench synchronises the device after every timed step
(production does not: its label check runs on the host), so a timed step is its serial cost,
slightly pessimistic. Grid cells: 5 warm-up + 30 timed train steps on real seeds
drawn with r = train.neg_rate (seed 0, epoch 0); step wall time (mean, p90), GPU time (CUDA
events; on CPU the compute wall), sampler wait (time blocked in `next`), edges and nodes per
batch, peak `max_memory_allocated`. Stress: one unsplit train batch of the B train seeds with the
highest deg(src) + deg(dst) (train degrees) per batch size -> bytes/edge -> max_edges_per_step =
floor(bench.memory_fraction x device bytes / bytes/edge) (an OOM halves the seeds until it
fits). Val: one full val_early pass at the container's best grid cell (uncached, filling the
cache), then the cached pass. Faithful: batch fallback 8192 -> 4096 -> 2048 on OOM (train steps,
val and test batches must all fit). Eval timings (faithful val / test, look-ahead val) time
the first batch of a fresh loader apart (`*_first_batch_s`: worker fork + first sample, paid
once per pass) and average the rest. Determinism: same init and 3 batches twice ->
bit_deterministic, max param diff; 10 steps with deterministic algorithms on vs off -> overhead
ratio (train), the same 10 batches as eval forwards (eval ratio) and faithful train steps
(faithful ratio, when this container ran the faithful cell); CPU vs GPU logits on one batch.
Device memory fields are None on CPU (tests). host_peak_mib = the container's memory (cgroup
usage, else MemTotal - MemAvailable), sampled every MEM_POLL_S during the cell, max.
"""

from __future__ import annotations

import gc
import math
import os
import threading
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from aml.models.gnn import (
    CELLS_FILE,
    LABEL_SPLITS,
    add_guard,
    append_jsonl,
    empty_guard,
    read_jsonl,
)

if TYPE_CHECKING:
    from aml.paths import DataPaths

EXPERIMENT = "aml-gnn-bench"  # MLflow: one run per container keyed <bench_key>-<gpu>-<cores>
DETERMINISM_BATCHES = 3
OVERHEAD_STEPS = 10
FAITHFUL_DET_STEPS = 3  # faithful train steps timed per determinism mode
MEM_POLL_S = 0.5  # container-memory sampling period during a cell
# Container memory: cgroup v1 (Modal's gVisor exposes only usage / limit there), cgroup v2.
CGROUP_USAGE_FILES = (
    "/sys/fs/cgroup/memory/memory.usage_in_bytes",
    "/sys/fs/cgroup/memory.current",
)


def run_bench_container(
    paths: DataPaths,
    features_dir: Path,
    bench_dir: Path,
    gnn_cfg: dict,
    *,
    data_cfg: dict,
    gpu: str,
    cores: int,
    memory_mib: int,
    groups: Sequence[str],
    device: str,
    rerun: dict | None = None,
    on_checkpoint: Callable[[], None] | None = None,
    log: Callable[[str], None] | None = None,
) -> list[dict]:
    """Run `groups` (in costplan.CELL_GROUPS order) in this container and return every cell
    record of this (gpu, cores) container, including ones recorded by an earlier attempt.

    gpu: "L4" | "T4" (the label in cell ids; "cpu" in tests); cores / memory_mib: the
    container's shape (recorded in every cell and used for $/h). device: "cuda" | "cpu".
    rerun: for the "rerun" group, costplan.rerun_4core(...)'s cell ({"batch_size",
    "num_workers"}). A cell that OOMs is recorded with status "oom" (the faithful group then
    tries the next fallback batch); other exceptions propagate (the driver records the failure).
    """
    from aml.models.gnn import costplan as C

    unknown = sorted(set(groups) - set(C.CELL_GROUPS))
    if unknown:
        raise ValueError(f"unknown bench groups {unknown}; expected {C.CELL_GROUPS}")
    if "rerun" in groups and not rerun:
        raise ValueError("the rerun group needs costplan.rerun_4core(...)'s cell")
    b = _Bench(
        paths,
        features_dir,
        Path(bench_dir),
        gnn_cfg,
        data_cfg=data_cfg,
        gpu=gpu,
        cores=int(cores),
        memory_mib=int(memory_mib),
        device=device,
        on_checkpoint=on_checkpoint,
        log=log,
    )
    try:
        for group in (g for g in C.CELL_GROUPS if g in groups):
            if group == "rerun":
                b.group_rerun(rerun)
            else:
                getattr(b, f"group_{group}")()
    finally:
        b.mem.stop()
    cells = b.own_cells()
    b.log_mlflow(cells)
    return cells


def peak_host_mib() -> float:
    """The old proxy, kept as `host_peak_mib_proxy` in every cell: cgroup `memory.peak` if
    present, else self + children ru_maxrss (0.0 where neither exists, e.g. Windows). On Modal
    neither peak file exists, and ru_maxrss(children) is the RSS of the largest exited child (a
    child spawned during `import torch` reports the parent's ~4 GB): not the container's peak."""
    for p in ("/sys/fs/cgroup/memory.peak", "/sys/fs/cgroup/memory/memory.max_usage_in_bytes"):
        try:
            return int(Path(p).read_text(encoding="utf-8").strip()) / 2**20
        except (OSError, ValueError):
            continue
    try:
        import resource
    except ImportError:
        return 0.0
    kib = (
        resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        + resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    )
    return kib / 1024.0  # Linux reports KiB


def _stat_value(path: Path, names: Sequence[str]) -> int | None:
    try:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0] in names:
                return int(parts[1])
    except (OSError, ValueError):
        return None
    return None


def container_memory_mib() -> tuple[float | None, str | None]:
    """The container's memory in use now, in MiB, and its source: the cgroup usage (v1
    memory.usage_in_bytes, present on Modal; v2 memory.current) minus the inactive file cache
    when memory.stat has it ("cgroup"), else /proc/meminfo MemTotal - MemAvailable ("meminfo");
    (None, None) where neither exists (Windows). Never adds ru_maxrss of children."""
    for p in CGROUP_USAGE_FILES:
        try:
            usage = int(Path(p).read_text(encoding="utf-8").strip())
        except (OSError, ValueError):
            continue
        stat = Path(p).parent / "memory.stat"
        inactive = _stat_value(stat, ("total_inactive_file", "inactive_file")) or 0
        return max(0, usage - inactive) / 2**20, "cgroup"
    total = _stat_value(Path("/proc/meminfo"), ("MemTotal:",))
    avail = _stat_value(Path("/proc/meminfo"), ("MemAvailable:",))
    if total is not None and avail is not None:
        return max(0, total - avail) / 1024.0, "meminfo"  # kB
    return None, None


def _self_maxrss_mib() -> float:
    try:
        import resource
    except ImportError:
        return 0.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class MemPeak:
    """The running max of `read()` (container_memory_mib), sampled every `poll_s` by a daemon
    thread started on the first mark(). mark() starts a new window (one per cell); peak() ->
    (max over the window incl. a sample now, source), or (None, None) without a source."""

    def __init__(self, poll_s: float = MEM_POLL_S, read=container_memory_mib) -> None:
        self.poll_s, self._read = float(poll_s), read
        self._lock = threading.Lock()
        self._max: float | None = None
        self._src: str | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        v, src = self._read()
        if v is None:
            return
        with self._lock:
            self._src = src
            self._max = v if self._max is None else max(self._max, v)

    def _loop(self) -> None:
        while not self._stop.wait(self.poll_s):
            self._sample()

    def mark(self) -> None:
        with self._lock:
            self._max = None
        self._sample()
        if self._thread is None and not self._stop.is_set():
            self._thread = threading.Thread(target=self._loop, name="mem-peak", daemon=True)
            self._thread.start()

    def peak(self) -> tuple[float | None, str | None]:
        self._sample()
        with self._lock:
            return self._max, self._src

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None


# --- the container ------------------------------------------------------------------------------


class _Bench:
    def __init__(
        self,
        paths,
        features_dir,
        bench_dir: Path,
        gnn_cfg: dict,
        *,
        data_cfg: dict,
        gpu: str,
        cores: int,
        memory_mib: int,
        device: str,
        on_checkpoint,
        log,
    ) -> None:
        from aml.models.gnn import train as tr

        self.paths, self.features_dir, self.dir = paths, Path(features_dir), bench_dir
        self.cfg, self.data_cfg = gnn_cfg, data_cfg
        self.bcfg = gnn_cfg["bench"]
        self.gpu, self.cores, self.memory_mib, self.device = gpu, cores, memory_mib, device
        self.commit = on_checkpoint or tr._noop
        self.log = log or tr._print
        self.dir.mkdir(parents=True, exist_ok=True)
        self.cells_path = self.dir / CELLS_FILE
        self.recorded = {c["cell_id"]: c for c in read_jsonl(self.cells_path)}
        self.device_name, self.device_bytes = _device_info(device)
        self.g = None
        self.engine = None
        self.fengine = None
        self.mem = MemPeak()

    # -- bookkeeping ---------------------------------------------------------------------------

    def own_cells(self) -> list[dict]:
        return [
            c
            for c in read_jsonl(self.cells_path)
            if c.get("gpu") == self.gpu and int(c.get("cores", -1)) == self.cores
        ]

    def cell(
        self,
        group: str,
        fn: Callable[[], dict],
        *,
        batch_size: int | None = None,
        num_workers: int | None = None,
    ) -> dict:
        """Run one cell unless recorded; an OOM records status "oom"."""
        from aml.models.gnn import costplan as C
        from aml.models.gnn import train as tr

        cid = C.cell_id(self.gpu, self.cores, group, batch_size, num_workers)
        if cid in self.recorded:
            return self.recorded[cid]
        self.log(f"bench cell {cid} ...")
        self.mem.mark()
        t0 = time.perf_counter()
        rec: dict[str, Any] = {
            "cell_id": cid,
            "group": group,
            "gpu": self.gpu,
            "device_name": self.device_name,
            "device_bytes": self.device_bytes,
            "cores": self.cores,
            "memory_mib": self.memory_mib,
            "batch_size": batch_size,
            "num_workers": num_workers,
            "status": "ok",
            "error": None,
        }
        try:
            fields = fn()
        except (tr.GnnOOMError, tr._cuda_oom()) as e:
            tr._free_cuda(self.device)
            fields = {"guard": empty_guard()}
            rec.update(status="oom", error=f"{type(e).__name__}: {e}")
        guard = fields.pop("guard", None) or empty_guard()
        if fields.get("_status"):
            rec.update(status=fields.pop("_status"), error=fields.pop("_error", None))
        rec.update(fields)
        mem, src = self.mem.peak()
        rec.update(
            seconds=time.perf_counter() - t0,
            # rule 5 reads host_peak_mib: the container's memory, else this process's max RSS
            host_peak_mib=mem if mem is not None else _self_maxrss_mib(),
            host_peak_source=src or "ru_maxrss_self",
            host_peak_mib_proxy=peak_host_mib(),
            guard=guard,
        )
        rec = tr._plain(rec)
        append_jsonl(self.cells_path, rec)
        self.commit()
        self.recorded[cid] = rec
        self.log(f"bench cell {cid}: {rec['status']} ({rec['seconds']:.1f}s)")
        return rec

    def ensure_graph(self) -> dict:
        """Load the graph and the causal engine once per container; returns their timings."""
        if self.engine is not None:
            return {}
        from aml.models.gnn import graph as G
        from aml.models.gnn import train as tr

        t0 = time.perf_counter()
        self.g = G.load_graph(
            self.paths,
            self.features_dir,
            self.cfg,
            data_cfg=self.data_cfg,
            label_splits=LABEL_SPLITS,
            protocol="causal",
            log=self.log,
        )
        t1 = time.perf_counter()
        self.engine = tr.TemporalEngine(
            self.g,
            self.cfg,
            protocol="causal",
            params=tr.effective_params(self.cfg, "causal", None),
            device=self.device,
            runtime=self.cfg["runtime"],
            allowed_splits=(),
            log=self.log,
        )
        return {"graph_load_s": t1 - t0, "engine_build_s": time.perf_counter() - t1}

    def fresh_model(self, seed: int = 0, *, deterministic: bool | None = None):
        import torch

        from aml.models.gnn import train as tr

        t = self.cfg["train"]
        tr.set_determinism(
            seed,
            deterministic=bool(t["deterministic"] if deterministic is None else deterministic),
            matmul_precision=str(t["matmul_precision"]),
            device=self.device,
        )
        model = self.engine.new_model()
        return model, torch.optim.Adam(model.parameters(), lr=float(self.engine.params["lr"]))

    def train_loader(self, batch_size: int, workers: int, *, protocol: str = "causal"):
        from aml.models.gnn import sampler as S

        e = self.engine
        sampler = S.EpochSubsetSampler(e.pos_idx, e.neg_idx, float(e.params["neg_rate"]), 0)
        loader = S.make_loader(
            self.g,
            e.hetero,
            e.train_gids,
            protocol=protocol,
            role="train",
            sampler_cfg={**e.sampler_cfg, "batch_size": int(batch_size)},
            runtime={**e.runtime, "num_workers": int(workers)},
            device=self.device,
            sampler=sampler,
        )
        return loader, sampler

    def eval_loader(self, gids, workers: int, *, protocol: str = "causal"):
        from aml.models.gnn import sampler as S

        e = self.engine
        return S.make_loader(
            self.g,
            e.hetero,
            gids,
            protocol=protocol,
            role="eval",
            sampler_cfg=e.sampler_cfg,
            runtime={**e.runtime, "num_workers": int(workers)},
            device=self.device,
        )

    def best_grid(self) -> tuple[int, int]:
        """(batch, workers) of this container's fastest ok grid cell per epoch."""
        from aml.models.gnn import costplan as C

        cells = self.own_cells()
        grid = [c for c in cells if c["group"] == "grid" and c["status"] == "ok"]
        if not grid:
            sampler_cfg, runtime = self.cfg["sampler"], self.cfg["runtime"]
            return int(sampler_cfg["batch_size"]), int(runtime["num_workers"])
        rate = float(self.cfg["train"]["neg_rate"])
        build = [c for c in cells if c["group"] == "build" and c["status"] == "ok"]
        if build:
            n_pos, n_neg = int(build[0]["counts"]["n_pos"]), int(build[0]["counts"]["n_neg"])
        else:
            self.ensure_graph()
            n_pos, n_neg = len(self.engine.pos_idx), len(self.engine.neg_idx)

        def epoch_train_s(c: dict) -> float:
            return C.steps_per_epoch(n_pos, n_neg, rate, int(c["batch_size"])) * c["step_s_mean"]

        best = min(grid, key=lambda c: (epoch_train_s(c), c["cell_id"]))
        return int(best["batch_size"]), int(best["num_workers"])

    # -- groups ------------------------------------------------------------------------------

    def group_build(self) -> None:
        def fn() -> dict:
            from aml.data.split import SPLITS

            t = self.ensure_graph()
            g, e = self.g, self.engine
            n = {s: int((g.split_code == i).sum()) for i, s in enumerate(SPLITS)}
            is_test = g.split_code == SPLITS.index("test")
            d10 = int((is_test & (np.arange(g.n_edges) <= g.bounds["d10_last"])).sum())
            return {
                "build_s": t.get("graph_load_s", 0.0) + t.get("engine_build_s", 0.0),
                **t,
                "n_nodes": int(g.n_nodes),
                "n_edges": int(g.n_edges),
                "counts": {
                    "n_pos": int(len(e.pos_idx)),
                    "n_neg": int(len(e.neg_idx)),
                    "n_val_early": n["val_early"],
                    "n_val_late": n["val_late"],
                    "n_test": n["test"],
                    "n_test_d10": n["test"],  # the d10 pass scores every test row
                    "n_test_primary": d10,
                },
            }

        self.cell("build", fn)

    def group_grid(self) -> None:
        for b in self.bcfg["batch_sizes"]:
            for w in self.bcfg["num_workers"]:
                self.cell(
                    "grid",
                    lambda b=b, w=w: self.train_steps_cell(
                        b, w, int(self.bcfg["warmup_steps"]), int(self.bcfg["timed_steps"])
                    ),
                    batch_size=int(b),
                    num_workers=int(w),
                )

    def train_steps_cell(
        self, batch_size: int, workers: int, warmup: int, timed: int, *, protocol: str = "causal"
    ) -> dict:
        from aml.models.gnn import sampler as S

        self.ensure_graph()
        model, opt = self.fresh_model()
        loader, sampler = self.train_loader(batch_size, workers, protocol=protocol)
        try:
            return _time_steps(self, model, opt, loader, sampler, warmup, timed, cw=self.engine.cw)
        finally:
            S.close_loader(loader)  # persistent workers: not via the cyclic GC (5 s per worker)
            del loader
            gc.collect()

    def group_stress(self) -> None:
        for b in self.bcfg["batch_sizes"]:
            self.cell("stress", lambda b=b: self.stress_cell(int(b)), batch_size=int(b))

    def stress_cell(self, batch_size: int) -> dict:
        import torch

        from aml.models.gnn import sampler as S
        from aml.models.gnn import train as tr

        self.ensure_graph()
        g, e = self.g, self.engine
        tried: list[int] = []
        b = min(int(batch_size), len(e.train_gids))
        cuda = self.device == "cuda"
        guard = empty_guard()
        while b >= 1:
            tried.append(b)
            gids = stress_gids(g, e.train_gids, b)
            loader = S.make_loader(
                g,
                e.hetero,
                gids,
                protocol="causal",
                role="train",
                sampler_cfg={**e.sampler_cfg, "batch_size": b},
                runtime={**e.runtime, "num_workers": 0},
                device=self.device,
                sampler=S.EpochSubsetSampler(np.arange(b), np.empty(0, np.int64), 0.5, 0),
            )
            fb = next(iter(loader))
            guard = add_guard(guard, tr._guard_of(fb))
            model, opt = self.fresh_model()
            static = _mem_static(cuda)
            ok = False
            try:
                labels = tr.batch_labels_nosync(fb, e.y, e.g.y)
                opt.zero_grad(set_to_none=True)
                tr._forward_backward(
                    model, [fb], [labels], e.ea, e.cw, e.cw[labels].sum(), self.device
                )
                opt.step()
                ok = True
            except tr._cuda_oom():
                pass
            if ok:
                edges = tr._edges_of(fb)
                peak = int(torch.cuda.max_memory_allocated()) if cuda else None
                bpe = (peak - static) / edges if cuda and edges else None
                frac = float(self.bcfg["memory_fraction"])
                return {
                    "seeds_used": b,
                    "tried": tried,
                    "edges": edges,
                    "nodes": int(fb.n_nodes),
                    "static_mem_bytes": static,
                    "peak_mem_bytes": peak,
                    "bytes_per_edge": bpe,
                    "max_edges_per_step": (
                        int(math.floor(frac * self.device_bytes / bpe)) if bpe else None
                    ),
                    "guard": guard,
                }
            del model, opt, fb, loader
            tr._free_cuda(self.device)
            b //= 2
        raise tr.GnnOOMError("the stress batch does not fit even with one seed", 0, 1)

    def group_val(self) -> None:
        b, w = self.best_grid()
        self.cell("val", lambda: self.val_cell(w), batch_size=b, num_workers=w)

    def val_cell(self, workers: int) -> dict:
        import torch

        from aml.models.gnn import train as tr
        from aml.models.gnn import transforms as T

        self.ensure_graph()
        e = self.engine
        model, _ = self.fresh_model()
        model.eval()
        loader = self.eval_loader(e.val_gids, workers)
        cache = T.EvalCache(float(e.sampler_cfg["eval_cache_max_gb"]))
        cuda = self.device == "cuda"
        static = _mem_static(cuda)
        guard = empty_guard()
        times = []
        for _ in range(2):  # the uncached pass fills the cache, then the cached pass
            if times and not cache.cached:
                break
            t0 = time.perf_counter()
            with torch.inference_mode():
                for part in cache.batches(loader, e.eval_cap):
                    guard = add_guard(guard, tr._guard_of(part))
                    tr.eval_parts({0: model}, part, ea=e.ea, device=self.device, cap=e.eval_cap)
            if cuda:
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        st = cache.stats()
        peak = int(torch.cuda.max_memory_allocated()) if cuda else None
        emax = int(st["max_edges_per_part"])
        bpe = (peak - static) / emax if cuda and emax else None
        frac = float(self.bcfg["memory_fraction"])
        del loader
        gc.collect()
        return {
            "val_s": times[0],
            "val_cached_s": times[1] if len(times) > 1 else None,
            "cache_gb": st["gb"],
            "cache_fits": bool(st["cached"]),
            "eval_batches": int(st["batches"]),
            "eval_edges": int(st["edges"]),
            "eval_edges_max": emax,
            "eval_batch_edges_max": int(st["max_edges_per_batch"]),
            "eval_cap": e.eval_cap,
            "eval_peak_mem_bytes": peak,
            "eval_bytes_per_edge": bpe,
            "max_edges_per_eval_step": (
                int(math.floor(frac * self.device_bytes / bpe)) if bpe else None
            ),
            "eval_rows": int(len(e.val_gids)),
            "guard": guard,
        }

    def group_faithful(self) -> None:
        _, w = self.best_grid()
        self.cell("faithful", lambda: self.faithful_cell(w), num_workers=w)

    def faithful_cell(self, workers: int) -> dict:

        from aml.models.gnn import graph as G
        from aml.models.gnn import train as tr

        self.ensure_graph()
        fcfg = self.bcfg["faithful"]
        t0 = time.perf_counter()
        if self.fengine is None:
            self.fengine = tr.FaithfulEngine(
                self.g,
                self.cfg,
                paths=self.paths,
                features_dir=self.features_dir,
                params=tr.effective_params(self.cfg, "faithful", None),
                device=self.device,
                runtime={**self.cfg["runtime"], "num_workers": int(workers)},
                allow_test=False,  # no test scores: only timed forward passes, no test labels
                log=self.log,
            )
        fe = self.fengine
        fe.build_snapshot("test")
        build_s = time.perf_counter() - t0
        test_gids = G.split_gids(self.g, "test")
        cuda = self.device == "cuda"
        tried: list[int] = []
        guard = empty_guard()
        last_err = None
        for b in fcfg["batch_fallback"]:
            b = int(b)
            tried.append(b)
            try:
                out = self.faithful_batch(fe, b, test_gids, cuda)
            except (tr.GnnOOMError, tr._cuda_oom()) as e:
                last_err = f"batch {b}: {type(e).__name__}: {e}"
                self.log(f"faithful batch {b} does not fit ({last_err}); trying the next one")
                gc.collect()
                tr._free_cuda(self.device)
                continue
            guard = add_guard(guard, out.pop("guard"))
            return {"tried": tried, "batch_size": b, "build_s": build_s, **out, "guard": guard}
        return {
            "_status": "oom",
            "_error": f"no faithful batch fits (tried {tried}); last: {last_err}",
            "tried": tried,
            "batch_size": None,
            "build_s": build_s,
            "guard": guard,
        }

    def faithful_batch(self, fe, b: int, test_gids, cuda: bool) -> dict:
        import torch

        from aml.models.gnn import sampler as S
        from aml.models.gnn import train as tr

        fcfg = self.bcfg["faithful"]
        tr.set_determinism(
            0,
            deterministic=bool(self.cfg["train"]["deterministic"]),
            matmul_precision=str(self.cfg["train"]["matmul_precision"]),
            device=self.device,
        )
        model = fe.new_model()
        opt = torch.optim.Adam(model.parameters(), lr=float(fe.params["lr"]))
        loader = fe.loader("train", fe.train_gids, shuffle=True, seed=0, batch_size=b)
        loader.sampler.set_epoch(0)
        static = _mem_static(cuda)
        warm, timed = int(fcfg["warmup_steps"]), int(fcfg["timed_steps"])
        guard = empty_guard()
        steps, edges, sampled = [], [], []
        it, epoch = iter(loader), 0
        model.train()
        for i in range(warm + timed):
            t0 = time.perf_counter()
            fb, it, epoch = _next(loader, it, None, epoch)
            tr.faithful_train_batch(
                model,
                opt,
                fb,
                ea=fe.ea["train"],
                y=fe.y,
                cw=fe.cw,
                device=self.device,
                y_host=fe.g.y,
            )
            _sync(cuda)
            guard = add_guard(guard, tr._guard_of(fb))
            if i >= warm:
                steps.append(time.perf_counter() - t0)
                edges.append(tr._edges_of(fb))
                sampled.append(float(fb.sampled.float().mean()))
        del it
        out: dict[str, Any] = {
            "warmup_steps": warm,
            "timed_steps": timed,
            "step_s_mean": float(np.mean(steps)),
            "step_s_p90": float(np.percentile(steps, 90)),
            "edges_mean": float(np.mean(edges)),
            "edges_max": int(np.max(edges)),
        }
        share = {"train": float(np.mean(sampled))}
        for name, gids, n_batches in (
            ("val", fe.val_gids, int(fcfg["val_batches"])),
            ("test", test_gids, int(fcfg["test_batches"])),
        ):
            ev = fe.loader(name, gids, shuffle=False, seed=0, batch_size=b)
            s_per, e_per, sh, gv, first = _time_eval(fe, ev, name, n_batches, model, cuda)
            guard = add_guard(guard, gv)
            out[f"{name}_s_per_batch"] = s_per
            out[f"{name}_first_batch_s"] = first
            out[f"{name}_edges_mean"] = e_per
            out[f"{name}_batches"] = n_batches
            share[name] = sh
            del ev
        out["sampled_share"] = share
        peak = int(torch.cuda.max_memory_allocated()) if cuda else None
        out["peak_mem_bytes"] = peak
        out["static_mem_bytes"] = static
        out["guard"] = guard
        S.close_loader(loader)  # persistent workers: not via the cyclic GC (5 s per worker)
        del loader, model, opt
        gc.collect()
        return out

    def group_lookahead(self) -> None:
        b, w = self.best_grid()
        self.cell("lookahead", lambda: self.lookahead_cell(b, w), batch_size=b, num_workers=w)

    def lookahead_cell(self, b: int, w: int) -> dict:
        import torch

        from aml.models.gnn import train as tr

        self.ensure_graph()
        la = self.bcfg["lookahead"]
        out = self.train_steps_cell(b, w, 1, int(la["timed_steps"]), protocol="lookahead")
        guard = out.pop("guard")
        model, _ = self.fresh_model()
        model.eval()
        loader = self.eval_loader(self.engine.val_gids, w, protocol="lookahead")
        cuda = self.device == "cuda"
        n_batches = int(la["val_batches"])
        times, edges, targets = [], 0, 0
        vguard = empty_guard()
        it = iter(loader)
        with torch.inference_mode():
            for _ in range(n_batches):
                t0 = time.perf_counter()
                try:
                    fb = next(it)  # the first one includes the worker fork (timed apart)
                except StopIteration:
                    break
                vguard = add_guard(vguard, tr._guard_of(fb))
                for part in tr._cap_parts(fb, self.engine.eval_cap):
                    tr.eval_parts(
                        {0: model},
                        part,
                        ea=self.engine.ea,
                        device=self.device,
                        cap=self.engine.eval_cap,
                    )
                _sync(cuda)
                times.append(time.perf_counter() - t0)
                edges += tr._edges_of(fb)
                targets += int(fb.tgt_gid.numel())
        del it, loader
        gc.collect()
        tot = add_guard(guard, vguard)
        rest = times[1:] or times[:1]
        return {
            **out,
            "val_batches": len(times),
            "val_first_batch_s": times[0] if times else None,
            "val_s_per_batch": float(np.mean(rest)) if rest else None,
            "edges_per_target_train": out["edges_mean"] / b,
            "edges_per_target_val": edges / targets if targets else None,
            "future_share": tr.future_share(tot),
            "future_share_train": tr.future_share(guard),
            "future_share_val": tr.future_share(vguard),
            "dropped_target_copies": int(tot["dropped_target_copies"]),
            "guard": tot,
        }

    def group_pna(self) -> None:
        b, w = self.best_grid()
        self.cell("pna", lambda: self.pna_cell(b, w), batch_size=b, num_workers=w)

    def pna_cell(self, b: int, w: int) -> dict:
        import torch

        from aml.models.gnn import graph as G
        from aml.models.gnn import model as M
        from aml.models.gnn import sampler as S
        from aml.models.gnn import train as tr

        self.ensure_graph()
        params = tr.effective_params(self.cfg, "pna", None)
        fwd, rev = G.train_degree_histograms(self.g)
        deg = (torch.from_numpy(np.asarray(fwd)), torch.from_numpy(np.asarray(rev)))
        t = self.cfg["train"]
        tr.set_determinism(
            0,
            deterministic=bool(t["deterministic"]),
            matmul_precision=str(t["matmul_precision"]),
            device=self.device,
        )
        model = M.build_model(self.cfg, "pna", self.engine.preprocess, params, deg=deg)
        model = model.to(self.device)
        opt = torch.optim.Adam(model.parameters(), lr=float(params["lr"]))
        cw = torch.tensor(
            tr.class_weights(params["neg_rate"], params["w_pos"]),
            dtype=torch.float32,
            device=self.device,
        )
        p = self.bcfg["pna"]
        loader, sampler = self.train_loader(b, w)
        try:
            out = _time_steps(
                self,
                model,
                opt,
                loader,
                sampler,
                int(p["warmup_steps"]),
                int(p["timed_steps"]),
                cw=cw,
            )
        finally:
            S.close_loader(loader)  # persistent workers: not via the cyclic GC (5 s per worker)
            del loader
            gc.collect()
        return {**out, "hidden": params["hidden"], "towers": params["towers"]}

    def group_determinism(self) -> None:
        b, _ = self.best_grid()
        self.cell("determinism", lambda: self.determinism_cell(b), batch_size=b, num_workers=0)

    def determinism_cell(self, b: int) -> dict:
        import torch

        from aml.models.gnn import model as M
        from aml.models.gnn import train as tr

        self.ensure_graph()
        e, cuda = self.engine, self.device == "cuda"
        loader, sampler = self.train_loader(b, 0)
        sampler.set_epoch(0)
        batches, guard = [], empty_guard()
        it = iter(loader)
        for _ in range(max(DETERMINISM_BATCHES, OVERHEAD_STEPS)):
            fb, it, _ep = _next(loader, it, sampler, 0)
            guard = add_guard(guard, tr._guard_of(fb))
            batches.append(fb)
        del it, loader

        def run(n: int, deterministic: bool) -> tuple[dict, float]:
            model, opt = self.fresh_model(0, deterministic=deterministic)
            _sync(cuda)
            t0 = time.perf_counter()
            for fb in batches[:n]:
                tr.train_batch(
                    model,
                    opt,
                    fb,
                    ea=e.ea,
                    y=e.y,
                    cw=e.cw,
                    cap=e.train_cap,
                    device=self.device,
                    y_host=e.g.y,
                )
            _sync(cuda)
            return {
                k: v.detach().clone() for k, v in model.state_dict().items()
            }, time.perf_counter() - t0

        def forwards(deterministic: bool) -> float:
            """The same batches as eval forwards (eval mode, inference_mode, eval_parts)."""
            model, _ = self.fresh_model(0, deterministic=deterministic)
            model.eval()
            with torch.inference_mode():
                tr.eval_parts({0: model}, batches[0], ea=e.ea, device=self.device, cap=e.eval_cap)
                _sync(cuda)  # batch 0 warms the kernels of this mode
                t0 = time.perf_counter()
                for fb in batches[:OVERHEAD_STEPS]:
                    tr.eval_parts({0: model}, fb, ea=e.ea, device=self.device, cap=e.eval_cap)
                _sync(cuda)
            return time.perf_counter() - t0

        a, _ = run(DETERMINISM_BATCHES, True)
        b_, _ = run(DETERMINISM_BATCHES, True)
        bit = all(torch.equal(a[k], b_[k]) for k in a)
        diff = max(float((a[k].double() - b_[k].double()).abs().max()) for k in a if a[k].numel())
        run(2, True)  # warm the kernels of both modes before timing
        run(2, False)
        _, t_on = run(OVERHEAD_STEPS, True)
        _, t_off = run(OVERHEAD_STEPS, False)
        ev_on, ev_off = forwards(True), forwards(False)
        # faithful batches keep their targets (published protocol): their snapshot guard is
        # recorded apart, so this cell's guard keeps the causal invariant (0 target hits)
        f_on, f_off, f_guard = self.faithful_det_times()
        cpu_diff = None
        if cuda:
            model, _ = self.fresh_model(0)
            model.eval()
            cpu_model = M.build_model(self.cfg, "causal", e.preprocess, e.params)
            cpu_model.load_state_dict({k: v.cpu() for k, v in model.state_dict().items()})
            cpu_model.eval()
            from aml.models.gnn import transforms as T

            with torch.inference_mode():
                fb = batches[0]
                g_log = model(*T.to_model_inputs(fb, e.ea, "cuda")).double().cpu()
                c_log = cpu_model(*T.to_model_inputs(fb, e.ea.cpu(), "cpu")).double()
            cpu_diff = float((g_log - c_log).abs().max())
        # restore the configured mode for whatever runs next in this process
        self.fresh_model(0)

        def ratio(on: float | None, off: float | None) -> float | None:
            return on / off if on is not None and off else None

        return {
            "bit_deterministic": bool(bit),
            "max_param_diff": diff,
            "det_on_s": t_on,
            "det_off_s": t_off,
            "det_overhead_ratio": ratio(t_on, t_off),  # train steps (decision.json keeps it)
            "det_overhead_ratio_train": ratio(t_on, t_off),
            "det_eval_on_s": ev_on,
            "det_eval_off_s": ev_off,
            "det_overhead_ratio_eval": ratio(ev_on, ev_off),
            "det_faithful_on_s": f_on,
            "det_faithful_off_s": f_off,
            "det_overhead_ratio_faithful": ratio(f_on, f_off),
            "cpu_gpu_max_logit_diff": cpu_diff,
            "steps": OVERHEAD_STEPS,
            "faithful_guard": f_guard,
            "guard": guard,
        }

    def faithful_det_times(self) -> tuple[float | None, float | None, dict]:
        """Seconds of FAITHFUL_DET_STEPS faithful train steps with deterministic algorithms on
        and off (after one warm-up step each), on batches of this container's faithful cell;
        (None, None, empty guard) if this process holds no faithful engine with a fitting batch
        (e.g. the faithful cell was recorded by an earlier attempt)."""
        import torch

        from aml.models.gnn import sampler as S
        from aml.models.gnn import train as tr

        fe = self.fengine
        fit = [
            c
            for c in self.own_cells()
            if c["group"] == "faithful" and c["status"] == "ok" and c.get("batch_size")
        ]
        guard = empty_guard()
        if fe is None or not fit:
            return None, None, guard
        cuda = self.device == "cuda"
        loader = fe.loader(
            "train", fe.train_gids, shuffle=True, seed=0, batch_size=int(fit[0]["batch_size"])
        )
        loader.sampler.set_epoch(0)
        batches, it, epoch = [], iter(loader), 0
        for _ in range(1 + FAITHFUL_DET_STEPS):
            fb, it, epoch = _next(loader, it, None, epoch)
            guard = add_guard(guard, tr._guard_of(fb))
            batches.append(fb)
        del it
        S.close_loader(loader)  # persistent workers: not via the cyclic GC (5 s per worker)
        del loader

        def steps(deterministic: bool) -> float:
            tr.set_determinism(
                0,
                deterministic=deterministic,
                matmul_precision=str(self.cfg["train"]["matmul_precision"]),
                device=self.device,
            )
            model = fe.new_model()
            opt = torch.optim.Adam(model.parameters(), lr=float(fe.params["lr"]))
            model.train()
            t0 = time.perf_counter()
            for i, fb in enumerate(batches):
                if i == 1:  # batch 0 warms the kernels of this mode
                    _sync(cuda)
                    t0 = time.perf_counter()
                tr.faithful_train_batch(
                    model,
                    opt,
                    fb,
                    ea=fe.ea["train"],
                    y=fe.y,
                    cw=fe.cw,
                    device=self.device,
                    y_host=fe.g.y,
                )
            _sync(cuda)
            return time.perf_counter() - t0

        on, off = steps(True), steps(False)
        gc.collect()
        return on, off, guard

    def group_rerun(self, rerun: dict) -> None:
        b, w = int(rerun["batch_size"]), int(rerun["num_workers"])

        def fn() -> dict:
            out = self.train_steps_cell(
                b, w, int(self.bcfg["warmup_steps"]), int(self.bcfg["timed_steps"])
            )
            guard = out.pop("guard")
            val = self.val_cell(w)
            guard = add_guard(guard, val.pop("guard"))
            return {**out, **val, "guard": guard}

        self.cell("rerun", fn, batch_size=b, num_workers=w)

    # -- MLflow --------------------------------------------------------------------------------

    def log_mlflow(self, cells: list[dict]) -> None:
        from aml.models.gnn import train as tr

        key = f"{self.dir.name}-{self.gpu}-{self.cores}"
        tags = {"gpu": self.gpu, "device_name": str(self.device_name), "cores": str(self.cores)}
        with tr._mlflow_run(EXPERIMENT, key, tags, self.log) as run:
            if run is None:
                return
            from aml import tracking

            metrics = {
                c["cell_id"]: {
                    k: v for k, v in c.items() if k not in ("guard", "faithful_guard", "counts")
                }
                for c in cells
            }
            tr._mlflow_call(self.log, tracking.log_metrics_flat, metrics)
            tr._mlflow_call(
                self.log,
                tracking.log_params_flat,
                {"versions": _versions(), "device_bytes": self.device_bytes},
            )


# --- measurement helpers -------------------------------------------------------------------------


def stress_gids(g, train_gids: np.ndarray, b: int) -> np.ndarray:
    """The stress batch (M3 spec §10.1): the `b` train gids with the highest deg(src) +
    deg(dst), degrees = in + out degree over TRAIN edges; ties -> lower gid; rank order."""
    train_gids = np.asarray(train_gids, dtype=np.int64)
    src, dst = g.src[train_gids], g.dst[train_gids]
    deg = np.bincount(src, minlength=g.n_nodes) + np.bincount(dst, minlength=g.n_nodes)
    score = deg[src] + deg[dst]
    order = np.lexsort((train_gids, -score))  # score desc, then gid asc
    return np.sort(train_gids[order[: int(b)]])


def _device_info(device: str) -> tuple[str, int | None]:
    if device == "cuda":
        import torch

        p = torch.cuda.get_device_properties(0)
        return str(p.name), int(p.total_memory)
    return "cpu", None


def _versions() -> dict:
    from aml.models.gnn import train as tr

    return {m: tr._version(m) for m in ("torch", "torch_geometric", "pyg_lib")} | {
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    }


def _sync(cuda: bool) -> None:
    if cuda:
        import torch

        torch.cuda.synchronize()


def _mem_static(cuda: bool) -> int | None:
    """Reset the peak counter; the memory allocated now (model, EA, labels)."""
    if not cuda:
        return None
    import torch

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    return int(torch.cuda.memory_allocated())


def _next(loader, it, sampler, epoch: int):
    """The next batch, starting the loader's next epoch when one ends."""
    try:
        return next(it), it, epoch
    except StopIteration:
        epoch += 1
        if sampler is not None:
            sampler.set_epoch(epoch)
        elif hasattr(getattr(loader, "sampler", None), "set_epoch"):
            loader.sampler.set_epoch(epoch)  # a faithful train loader (FaithfulEpochSampler)
        it = iter(loader)
        return next(it), it, epoch


def _time_steps(bench: _Bench, model, opt, loader, sampler, warmup: int, timed: int, *, cw) -> dict:
    """Production train steps (train.train_batch) on `loader`: wall (incl. sampler wait),
    device time, wait, edges, nodes and peak memory over the timed steps."""
    import torch

    from aml.models.gnn import train as tr

    e, cuda = bench.engine, bench.device == "cuda"
    sampler.set_epoch(0)
    static = _mem_static(cuda)
    it, epoch = iter(loader), 0
    step_s, dev_s, wait_s, edges, nodes = [], [], [], [], []
    guard, ooms = empty_guard(), 0
    model.train()
    for i in range(warmup + timed):
        t0 = time.perf_counter()
        fb, it, epoch = _next(loader, it, sampler, epoch)
        t1 = time.perf_counter()
        if cuda:
            ev0, ev1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            ev0.record()
        _, n_oom = tr.train_batch(
            model,
            opt,
            fb,
            ea=e.ea,
            y=e.y,
            cw=cw,
            cap=e.train_cap,
            device=bench.device,
            y_host=e.g.y,
        )
        if cuda:
            ev1.record()
        _sync(cuda)
        t2 = time.perf_counter()
        guard = add_guard(guard, tr._guard_of(fb))
        ooms += n_oom
        if i >= warmup:
            step_s.append(t2 - t0)
            wait_s.append(t1 - t0)
            dev_s.append(ev0.elapsed_time(ev1) / 1000.0 if cuda else t2 - t1)
            edges.append(tr._edges_of(fb))
            nodes.append(int(fb.n_nodes))
    del it
    peak = int(torch.cuda.max_memory_allocated()) if cuda else None
    emax = int(np.max(edges))
    mean_step = float(np.mean(step_s))
    return {
        "warmup_steps": int(warmup),
        "timed_steps": int(timed),
        "step_s_mean": mean_step,
        "step_s_p90": float(np.percentile(step_s, 90)),
        "gpu_s_mean": float(np.mean(dev_s)),
        "wait_s_mean": float(np.mean(wait_s)),
        "wait_share": float(np.mean(wait_s)) / mean_step if mean_step > 0 else None,
        "edges_mean": float(np.mean(edges)),
        "edges_max": emax,
        "nodes_mean": float(np.mean(nodes)),
        "nodes_max": int(np.max(nodes)),
        "static_mem_bytes": static,
        "peak_mem_bytes": peak,
        "bytes_per_edge": (peak - static) / emax if cuda and emax else None,
        "train_cap": e.train_cap,
        "oom_splits": int(ooms),
        "guard": guard,
    }


def _time_eval(fe, loader, snapshot: str, n_batches: int, model, cuda: bool):
    """(seconds per eval batch, edges per batch, sampled share, guard totals, first batch s)
    over the first n_batches of a fresh faithful eval loader. The first batch (worker fork +
    first sample, paid once per pass) is timed apart; the mean is over the others (the first
    one alone if n_batches is 1)."""
    import torch

    from aml.models.gnn import train as tr

    times, edges, sampled = [], [], []
    guard = empty_guard()
    model.eval()
    it = iter(loader)
    with torch.inference_mode():
        for _ in range(n_batches):
            t0 = time.perf_counter()
            try:
                fb = next(it)
            except StopIteration:
                break
            tr.eval_parts({0: model}, fb, ea=fe.ea[snapshot], device=fe.device, cap=None)
            _sync(cuda)
            times.append(time.perf_counter() - t0)
            edges.append(tr._edges_of(fb))
            sampled.append(float(fb.sampled.float().mean()))
            guard = add_guard(guard, tr._guard_of(fb))
    model.train()
    del it
    if not times:
        return None, None, None, guard, None
    rest = times[1:] or times[:1]
    return (
        float(np.mean(rest)),
        float(np.mean(edges)),
        float(np.mean(sampled)),
        guard,
        float(times[0]),
    )
