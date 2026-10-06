"""End-to-end as-of rule for the causal GNN on the synthetic fixture (PLAN.md §4; M3 spec §14.2).

A Multi-GINe is trained for one epoch on CPU (a plain loop here, so the suite needs only the
graph, sampler, transform and model, not train.py), then its weights AND the edge-attribute
preprocessing are frozen (`load_graph(preprocess=frozen)`, as M2 freezes hubs and vocab). For
targets in train days and in test days, every event with minute >= the target's is changed,
deleted or added (same-minute peers ranked before the target, reverse and parallel edges of u and
v, cycles, self-loops; tests/fixtures/perturb.py), the real `build.run_build_features` re-runs on
the perturbed data (with the unperturbed vocab, hubs and hub_cap) and `load_graph` reloads it.
Re-scored alone in eval mode and matched by row_id (ranks shift), the target's causal score is
bit-identical, while later rows' scores do change. Also: flipping every label leaves the graph
(topology, time, bounds, edge attributes) bit-identical, and eval scores do not depend on the
batch (alone vs batched vs shuffled).
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest

from tests.conftest import load_yaml
from tests.fixtures.engine_frames import read_fixture
from tests.fixtures.gnn_graphs import REPO_ROOT, gnn_inputs
from tests.fixtures.gnn_ref import LOADER_PARTS, require_gnn, runtime_cfg
from tests.fixtures.perturb import perturb_events

PARTS = (
    *LOADER_PARTS,
    "graph.load_graph",
    "transforms.to_model_inputs",
    "model.MultiGINe.__init__",
    "model.MultiGINe.forward",
    "model.scores_from_logits",
)
LATER = 40  # later positions of the target's split re-scored to show the perturbation bites
DAYS = ((3, 5), (9, 10))  # targets in train days and in test days (the primary view)


@dataclass
class Base:
    paths: Any
    features_dir: Path
    gnn_cfg: dict
    build_cfgs: dict
    data_cfg: dict
    g: Any  # HostGraph
    model: Any
    spec: Any  # the unperturbed EngineSpec (vocab, hubs, hub_cap)
    targets: dict  # DAYS -> 2 target row ids


def _loader(g, seeds, gnn_cfg, *, batch_size, role="eval", sampler=None):
    from aml.models.gnn.graph import build_hetero
    from aml.models.gnn.sampler import make_loader

    sc = dict(gnn_cfg["sampler"], batch_size=batch_size, eval_batch_size=batch_size)
    return make_loader(
        g,
        build_hetero(g),
        np.asarray(seeds, np.int64),
        protocol="causal",
        role=role,
        sampler_cfg=sc,
        runtime=runtime_cfg(0),
        device="cpu",
        sampler=sampler,
    )


def _model(g, gnn_cfg):
    from aml.models.gnn.model import MultiGINe

    m, t, pp = gnn_cfg["model"], gnn_cfg["train"], g.preprocess
    return MultiGINe(
        num_idx=list(pp["num_idx"]),
        cat_idx=list(pp["cat_idx"]),
        cat_sizes=list(pp["cat_sizes"]),
        hidden=m["hidden"],
        layers=m["layers"],
        layer_dropout=m["layer_dropout"],
        final_dropout=t["final_dropout"],
        readout_hidden=tuple(m["readout_hidden"]),
        edge_updates=m["edge_updates"],
        conv="gine",
    )


def _train_one_epoch(g, gnn_cfg):
    """One pass over every train seed (weighted CE [1, w_pos], Adam), then eval mode."""
    import torch
    import torch.nn.functional as F

    from aml.models.gnn.transforms import to_model_inputs

    torch.manual_seed(0)
    model = _model(g, gnn_cfg)
    t = gnn_cfg["train"]
    opt = torch.optim.Adam(model.parameters(), lr=t["lr"])
    weight = torch.tensor([1.0, float(t["w_pos"])])
    ea = torch.from_numpy(g.ea)
    model.train()
    steps = 0
    for fb in _loader(g, np.flatnonzero(g.split_code == 0), gnn_cfg, batch_size=64):
        y = torch.from_numpy(g.y[fb.tgt_gid.numpy()].astype(np.int64))
        assert (y >= 0).all()
        loss = F.cross_entropy(model(*to_model_inputs(fb, ea, "cpu")), y, weight=weight)
        opt.zero_grad()
        loss.backward()
        opt.step()
        steps += 1
    assert steps > 10
    return model.eval()


def _score(model, g, fbs) -> dict[int, float]:
    """row_id -> float64 score of every target of `fbs` (eval mode, inference)."""
    import torch

    from aml.models.gnn.model import scores_from_logits
    from aml.models.gnn.transforms import to_model_inputs

    ea = torch.from_numpy(g.ea)
    out: dict[int, float] = {}
    with torch.inference_mode():
        for fb in fbs:
            s = scores_from_logits(model(*to_model_inputs(fb, ea, "cpu")))
            for gid, v in zip(fb.tgt_gid.tolist(), np.asarray(s).tolist(), strict=True):
                out[int(g.row_id[gid])] = v
    return out


def _alone(g, gnn_cfg, seeds, positions):
    """FlatBatches of one subgraph each, for `positions` of `seeds` (the loader runs in order and
    stops after the last wanted position)."""
    want, last = set(int(p) for p in positions), int(max(positions))
    for fb in _loader(g, seeds, gnn_cfg, batch_size=1):
        assert len(fb.tgt_gid) == 1, "eval_batch_size 1 must give one subgraph per batch"
        p = int(fb.seed_pos[0])
        if p in want:
            yield fb
        if p >= last:
            return


def _split_seeds(g, gid: int) -> tuple[np.ndarray, int]:
    seeds = np.flatnonzero(g.split_code == g.split_code[gid])
    return seeds, int(np.searchsorted(seeds, gid))


@pytest.fixture(scope="module")
def base(prepared, data_cfg, rules_cfg, tmp_path_factory) -> Base:
    require_gnn(*PARTS)
    from aml.features.build import load_spec
    from aml.models.gnn import LABEL_SPLITS
    from aml.models.gnn.graph import load_graph

    build_cfgs = {"data": data_cfg, "rules": rules_cfg, "features": load_yaml("features.yaml")}
    paths, features_dir, gnn_cfg = gnn_inputs(
        prepared, tmp_path_factory.mktemp("gnn_asof"), build_cfgs
    )
    g = load_graph(paths, features_dir, gnn_cfg, data_cfg=data_cfg, label_splits=LABEL_SPLITS)
    model = _train_one_epoch(g, gnn_cfg)
    spec = load_spec(features_dir)
    targets = {days: _targets(g, days) for days in DAYS}
    return Base(paths, features_dir, gnn_cfg, build_cfgs, data_cfg, g, model, spec, targets)


def _history(g) -> np.ndarray:
    """Per gid: earlier-minute transactions touching its src or dst (a subgraph-size proxy)."""
    inc = np.concatenate([g.src, g.dst])
    rank = np.concatenate([np.arange(g.n_edges)] * 2)
    order = np.lexsort((rank, inc))
    inc, rank = inc[order], rank[order]
    out = np.zeros(g.n_edges, dtype=np.int64)
    for gid in range(g.n_edges):
        for a in {int(g.src[gid]), int(g.dst[gid])}:
            lo = np.searchsorted(inc, a, side="left")
            out[gid] += int(
                np.searchsorted(rank[lo : np.searchsorted(inc, a, "right")], g.first_rank[gid])
            )
    return out


def _targets(g, days: tuple[int, int]) -> list[int]:
    """Row ids: the row with the longest history in `days`, and another longest-history row
    that shares its minute with another row."""
    hist = _history(g)
    cand = np.flatnonzero((g.day >= days[0]) & (g.day <= days[1]))
    first = int(cand[np.argmax(hist[cand])])
    shared = cand[(np.bincount(g.minute)[g.minute[cand]] > 1) & (cand != first)]
    return [int(g.row_id[first]), int(g.row_id[shared[np.argmax(hist[shared])]])]


def _write_dataset(src, frame: pl.DataFrame, root: Path):
    """A prepared-data directory for a perturbed frame (ENGINE_TX_COLUMNS): the transactions get
    every prepared column back (kept rows from the original, added rows synthesised), labels of
    added rows are 0, the other tables are copied."""
    from aml.paths import DataPaths

    full = pl.read_parquet(src.transactions)
    dst = DataPaths(root / "volume", src.dataset)
    dst.parquet_dir.mkdir(parents=True)
    dst.labels.parent.mkdir(parents=True)
    for name in ("accounts", "fx_rates"):
        shutil.copy(getattr(src, name), getattr(dst, name))
    marker = src.parquet_dir / "prepare_summary.json"
    if marker.exists():
        shutil.copy(marker, dst.parquet_dir / marker.name)
    extra = [c for c in full.columns if c not in frame.columns]
    epoch = full["ts"][0] - timedelta(minutes=int(full["minute"][0]))
    tx = (
        frame.join(full.select("row_id", *extra), on="row_id", how="left", maintain_order="left")
        .with_columns(
            pl.coalesce(pl.col("ts"), pl.lit(epoch) + pl.duration(minutes=pl.col("minute"))),
            pl.coalesce(pl.col("from_account"), pl.col("src").cast(pl.String)),
            pl.coalesce(pl.col("to_account"), pl.col("dst").cast(pl.String)),
            pl.coalesce(pl.col("amount_received"), pl.col("amount_paid")),
        )
        .select(full.columns)
        .cast(dict(full.schema))
    )
    tx.write_parquet(dst.transactions)
    labels = pl.read_parquet(src.labels)
    added = tx.filter(~pl.col("row_id").is_in(labels["row_id"].implode())).select("row_id")
    added = added.with_columns(pl.lit(0, dtype=pl.Int8).alias("is_laundering"))
    pl.concat([labels, added], how="diagonal").cast(dict(labels.schema)).write_parquet(dst.labels)
    return dst


_BUILD_SCRIPT = """
import json, sys
from pathlib import Path
from aml.features.build import run_build_features
from aml.paths import DataPaths
root, dataset, out, args = sys.argv[1:5]
a = json.loads(Path(args).read_text(encoding="utf-8"))
run_build_features(DataPaths(Path(root), dataset), Path(out), a["cfgs"], vocab=a["vocab"],
                   hubs=a["hubs"], hub_cap=a["hub_cap"])
"""


def _build_all(b: Base, jobs: list[tuple[Any, Path]], tmp: Path) -> None:
    """The real feature build of every (paths, out_dir), each in a fresh interpreter, all at
    once, with the unperturbed vocab, hubs and hub_cap (as M2's as-of test fixes them)."""
    args = tmp / "build_args.json"
    args.write_text(
        json.dumps(
            {
                "cfgs": b.build_cfgs,
                "vocab": {k: list(v) for k, v in b.spec.vocab.items()},
                "hubs": [int(h) for h in b.spec.hubs],
                "hub_cap": int(b.spec.hub_cap),
            }
        ),
        encoding="utf-8",
    )
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    procs = []
    for paths, out in jobs:
        log = out.parent / "build.log"
        cmd = [sys.executable, "-c", _BUILD_SCRIPT, str(paths.root), paths.dataset, str(out)]
        with log.open("w", encoding="utf-8") as f:
            procs.append(
                (
                    subprocess.Popen(
                        [*cmd, str(args)],
                        cwd=REPO_ROOT,
                        env=env,
                        stdout=f,
                        stderr=subprocess.STDOUT,
                    ),
                    log,
                )
            )
    failed = [
        f"{log}: {log.read_text(encoding='utf-8')[-3000:]}"
        for proc, log in procs
        if proc.wait(timeout=900) != 0
    ]
    assert not failed, failed


@pytest.fixture(scope="module")
def perturbed(base, tmp_path_factory) -> dict[int, Any]:
    """row_id -> the HostGraph of the data perturbed at that target (frozen preprocessing)."""
    from aml.models.gnn import LABEL_SPLITS
    from aml.models.gnn.graph import load_graph

    tmp = tmp_path_factory.mktemp("gnn_asof_perturbed")
    tx = read_fixture(base.paths)
    jobs = {}
    for k, row_id in enumerate(r for days in DAYS for r in base.targets[days]):
        tgt = tx.filter(pl.col("row_id") == row_id)
        pert = perturb_events(tx, tgt, int(tgt["minute"][0]), base.data_cfg, seed=700 + k)
        root = tmp / f"t{k}"
        jobs[row_id] = (_write_dataset(base.paths, pert, root), root / "features")
    _build_all(base, list(jobs.values()), tmp)
    return {
        row_id: load_graph(
            paths,
            out,
            base.gnn_cfg,
            data_cfg=base.data_cfg,
            label_splits=LABEL_SPLITS,
            preprocess=base.g.preprocess,
        )
        for row_id, (paths, out) in jobs.items()
    }


def _check_target(b: Base, row_id: int, g1) -> None:
    g0 = b.g
    gid0 = int(np.flatnonzero(g0.row_id == row_id)[0])
    seeds0, pos0 = _split_seeds(g0, gid0)
    s0 = _score(b.model, g0, _alone(g0, b.gnn_cfg, seeds0, range(pos0, pos0 + LATER)))
    n_ctx = int(g0.first_rank[gid0])  # rows of earlier minutes
    assert n_ctx > 0

    gid1 = int(np.flatnonzero(g1.row_id == row_id)[0])
    assert gid1 != gid0  # new same-minute peers ranked before it: comparing by row_id matters
    # the strictly earlier past is untouched, so the causal subgraph's inputs are identical
    np.testing.assert_array_equal(g1.row_id[:n_ctx], g0.row_id[:n_ctx])
    np.testing.assert_array_equal(g1.src[:n_ctx], g0.src[:n_ctx])
    assert g1.ea[:n_ctx].tobytes() == g0.ea[:n_ctx].tobytes()
    assert g1.ea[gid1].tobytes() == g0.ea[gid0].tobytes()  # its own features: as of minute - 1
    assert g1.first_rank[gid1] == n_ctx
    seeds1, pos1 = _split_seeds(g1, gid1)
    s1 = _score(b.model, g1, _alone(g1, b.gnn_cfg, seeds1, range(pos1, pos1 + LATER)))

    a, z = s0[row_id], s1[row_id]
    assert np.float64(a).tobytes() == np.float64(z).tobytes(), (row_id, a, z)
    common = [r for r in s0 if r != row_id and r in s1]
    assert common, "no later row survived the perturbation"
    assert any(s0[r] != s1[r] for r in common)  # the perturbation is not vacuous


@pytest.mark.parametrize("days", DAYS, ids=["train_days", "test_days"])
def test_causal_score_is_bit_identical_under_future_perturbation(base, perturbed, days) -> None:
    assert len(base.targets[days]) == 2
    for row_id in base.targets[days]:
        _check_target(base, row_id, perturbed[row_id])


def test_label_flip_leaves_the_graph_bit_identical(base, tmp_path) -> None:
    from aml.models.gnn import LABEL_SPLITS
    from aml.models.gnn.graph import load_graph
    from aml.paths import DataPaths

    src = base.paths
    dst = DataPaths(tmp_path / "volume", src.dataset)
    shutil.copytree(src.parquet_dir, dst.parquet_dir)
    dst.labels.parent.mkdir(parents=True)
    labels = pl.read_parquet(src.labels)
    flip = (1 - pl.col("is_laundering")).cast(pl.Int8).alias("is_laundering")
    labels.with_columns(flip).write_parquet(dst.labels)
    g0 = base.g
    g1 = load_graph(
        dst, base.features_dir, base.gnn_cfg, data_cfg=base.data_cfg, label_splits=LABEL_SPLITS
    )
    for name in ("src", "dst", "minute", "day", "split_code", "row_id", "first_rank", "rev_gid"):
        a, b = getattr(g0, name), getattr(g1, name)
        assert a.dtype == b.dtype and np.array_equal(a, b), name
    assert g1.ea.tobytes() == g0.ea.tobytes()
    assert (g1.n_nodes, g1.n_edges, g1.bounds) == (g0.n_nodes, g0.n_edges, g0.bounds)
    assert g1.preprocess == g0.preprocess and tuple(g1.attr_columns) == tuple(g0.attr_columns)
    assert (g1.features_digest, g1.spec_hash) == (g0.features_digest, g0.spec_hash)
    loaded = g0.y >= 0
    assert loaded.any() and np.array_equal(g1.y[loaded], 1 - g0.y[loaded])
    assert np.array_equal(g1.y[~loaded], g0.y[~loaded])


def test_eval_scores_do_not_depend_on_the_batch(base) -> None:
    """Alone vs batched (rank order) vs shuffled (the train sampler's permutation): <= 1e-6."""
    from aml.models.gnn.sampler import EpochSubsetSampler

    g, cfg = base.g, base.gnn_cfg
    seeds = np.flatnonzero(g.split_code == 0)
    first = range(0, 150)
    alone = _score(base.model, g, _alone(g, cfg, seeds, first))
    batched = _score(base.model, g, _loader(g, seeds, cfg, batch_size=64))
    y = g.y[seeds]
    sampler = EpochSubsetSampler(np.flatnonzero(y == 1), np.flatnonzero(y == 0), 1.0, 3)
    sampler.set_epoch(0)
    fbs = list(_loader(g, seeds, cfg, batch_size=64, role="train", sampler=sampler))
    order = np.concatenate([fb.seed_pos.numpy() for fb in fbs])
    assert sorted(order.tolist()) == list(range(len(seeds)))
    assert not np.array_equal(order, np.sort(order))  # really shuffled
    shuffled = _score(base.model, g, fbs)
    assert len(alone) == len(first)
    for r, v in alone.items():
        assert abs(batched[r] - v) <= 1e-6 and abs(shuffled[r] - v) <= 1e-6, r
