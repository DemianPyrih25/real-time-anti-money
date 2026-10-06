"""MultiGINe+EU, build_model and scores_from_logits (M3 spec §6, §14.2; CPU only).

The in-graph reference below is written from the spec's description of Multi-GNN's readout
(every message edge updated at every layer, the target read out as an in-graph `to` edge), not
from model.py's forward.
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

import torch.nn.functional as F  # noqa: E402

from aml.config import load_config  # noqa: E402
from aml.models.gnn.model import (  # noqa: E402
    EdgeEncoder,
    MultiGINe,
    build_model,
    count_parameters,
    scores_from_logits,
)
from tests.fixtures.gnn_graphs import (  # noqa: E402
    CATEGORICAL,
    FIXTURE_VOCAB_SIZES,
    GNN_EDGE_ATTRS,
    REPO_ROOT,
    fixture_preprocess,
    make_host_graph,
    random_tied_graph,
)

PRE = fixture_preprocess()
N_COLS = len(GNN_EDGE_ATTRS)
CAT_SIZES = PRE["cat_sizes"]
HI_SMALL_CAT_SIZES = [16, 16, 8]  # 15 currencies, 7 payment formats (+1: unknown code -1)


@pytest.fixture(scope="module")
def gnn_cfg() -> dict:
    return load_config("gnn", REPO_ROOT / "configs")


def model_kw(**kw) -> dict:
    out = {
        "num_idx": PRE["num_idx"],
        "cat_idx": PRE["cat_idx"],
        "cat_sizes": CAT_SIZES,
        "hidden": 16,
        "layers": 2,
    }
    out.update(kw)
    return out


def make_model(seed: int = 0, **kw) -> MultiGINe:
    torch.manual_seed(seed)
    return MultiGINe(**model_kw(**kw))


def random_ea(rng: np.random.Generator, n: int) -> torch.Tensor:
    ea = rng.standard_normal((n, N_COLS)).astype(np.float32)
    for c in CATEGORICAL:
        ea[:, GNN_EDGE_ATTRS.index(c)] = rng.integers(-1, FIXTURE_VOCAB_SIZES[c], n)
    return torch.from_numpy(ea)


def subgraph(rng: np.random.Generator, n_nodes: int = 6, n_edges: int = 12) -> dict:
    """One flat subgraph: roots 0 (src) and 1 (dst), random `to` edges, the reverse copies of
    the non-self-loop ones (same attribute rows), ego flags on the roots."""
    ei = torch.from_numpy(rng.integers(0, n_nodes, (2, n_edges)))
    ea = random_ea(rng, n_edges)
    keep = ei[0] != ei[1]
    x = torch.zeros(n_nodes, 1)
    x[:2] = 1.0
    return {
        "x": x,
        "ei_fwd": ei,
        "ea_fwd": ea,
        "ei_rev": ei[:, keep].flip(0),
        "ea_rev": ea[keep],
        "tgt_src": torch.tensor([0]),
        "tgt_dst": torch.tensor([1]),
        "tgt_attr": random_ea(rng, 1),
    }


def concat(parts: list[dict]) -> dict:
    """Disjoint union of flat subgraphs (local node ids offset), one target per part."""
    out: dict[str, list] = {k: [] for k in parts[0]}
    off = 0
    for p in parts:
        for k, v in p.items():
            out[k].append(v + off if k in ("ei_fwd", "ei_rev", "tgt_src", "tgt_dst") else v)
        off += p["x"].shape[0]
    return {k: torch.cat(v, dim=1 if k.startswith("ei_") else 0) for k, v in out.items()}


def args_of(b: dict) -> tuple:
    keys = ("x", "ei_fwd", "ea_fwd", "ei_rev", "ea_rev", "tgt_src", "tgt_dst", "tgt_attr")
    return tuple(b[k] for k in keys)


def batch(seed: int = 0, n_sub: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    return concat([subgraph(rng, 5 + k % 3, 8 + 3 * k) for k in range(n_sub)])


# --- shapes, gradients, determinism -----------------------------------------------------------


def test_output_shape_and_finite():
    m = make_model()
    b = batch()
    for train in (True, False):
        m.train(train)
        out = m(*args_of(b))
        assert out.shape == (5, 2) and out.dtype == torch.float32
        assert torch.isfinite(out).all()


def test_backward_reaches_every_parameter():
    """No unused parameters: emlp_rev has L - 1 modules and the last emlp_fwd updates `et`."""
    m = make_model(layers=3)
    m.train()
    out = m(*args_of(batch()))
    F.cross_entropy(out, torch.tensor([0, 1, 0, 1, 1])).backward()
    missing = [n for n, p in m.named_parameters() if p.grad is None]
    assert missing == []
    assert all(torch.isfinite(p.grad).all() for p in m.parameters())


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_edge_update_mlps(layers):
    m = make_model(layers=layers)
    assert len(m.emlp_fwd) == layers and len(m.emlp_rev) == layers - 1
    assert len(m.conv_fwd) == len(m.conv_rev) == len(m.bn) == layers
    m2 = make_model(layers=layers, edge_updates=False)
    assert len(m2.emlp_fwd) == len(m2.emlp_rev) == 0
    assert m2(*args_of(batch())).shape == (5, 2)


def test_cpu_determinism_bitwise():
    def run() -> tuple[dict, torch.Tensor]:
        m = make_model(seed=3)
        opt = torch.optim.Adam(m.parameters(), lr=0.01)
        b = batch(seed=1)
        y = torch.tensor([1, 0, 0, 1, 0])
        for _ in range(3):
            torch.manual_seed(11)  # dropout masks
            m.train()
            opt.zero_grad()
            F.cross_entropy(m(*args_of(b)), y).backward()
            opt.step()
        m.eval()
        with torch.no_grad():
            return copy.deepcopy(m.state_dict()), m(*args_of(b))

    sd1, out1 = run()
    sd2, out2 = run()
    assert torch.equal(out1, out2)
    assert sd1.keys() == sd2.keys()
    assert all(torch.equal(sd1[k], sd2[k]) for k in sd1)


# --- locality and the readout -----------------------------------------------------------------


def test_eval_scores_do_not_depend_on_batch_mates():
    rng = np.random.default_rng(5)
    parts = [subgraph(rng, 4 + k, 6 + 2 * k) for k in range(6)]
    m = make_model()
    with torch.no_grad():
        m.train()
        m(*args_of(concat(parts)))  # move the BatchNorm running stats off their init
        m.eval()
        full = m(*args_of(concat(parts)))
        alone = torch.cat([m(*args_of(p)) for p in parts])
        order = [3, 0, 5, 1, 4, 2]
        shuffled = m(*args_of(concat([parts[i] for i in order])))
    assert torch.allclose(full, alone, atol=1e-6, rtol=0)
    assert torch.allclose(full[order], shuffled, atol=1e-6, rtol=0)


def in_graph_reference(m: MultiGINe, b: dict, k: torch.Tensor) -> torch.Tensor:
    """Multi-GNN's edge readout (eval mode): every message edge updated at every layer, the
    targets read out as the in-graph `to` edges at positions k."""
    ei_f, ei_r = b["ei_fwd"], b["ei_rev"]
    h = m.node_emb(b["x"])
    ef, er = m.enc_fwd(b["ea_fwd"]), m.enc_rev(b["ea_rev"])
    for i in range(m.layers):
        agg = (m.conv_fwd[i](h, ei_f, ef) + m.conv_rev[i](h, ei_r, er)) / 2
        h = (h + F.relu(m.bn[i](agg))) / 2
        ef_next = ef + m.emlp_fwd[i](torch.cat([h[ei_f[0]], h[ei_f[1]], ef], -1)) / 2
        if i < m.layers - 1:  # a last-layer reverse update could not reach the readout anyway
            er = er + m.emlp_rev[i](torch.cat([h[ei_r[0]], h[ei_r[1]], er], -1)) / 2
        ef = ef_next
    z = torch.cat([F.relu(h[ei_f[0, k]]), F.relu(h[ei_f[1, k]]), ef[k]], -1)
    return m.readout(z)


@pytest.mark.parametrize("layers", [1, 2, 3])
def test_virtual_edge_readout_equals_in_graph_readout(layers):
    """F16: a target that is in the graph as a `to` message edge (plus its reverse copy) gets the
    same logits from the virtual target edge as from Multi-GNN's in-graph readout."""
    b = batch(seed=7, n_sub=4)
    k = torch.tensor([0, 9, 21, 30])  # in-graph `to` edges used as targets
    m = make_model(seed=2, layers=layers)
    with torch.no_grad():
        m.train()
        m(*args_of(b))
        m.eval()
        virtual = m(
            b["x"],
            b["ei_fwd"],
            b["ea_fwd"],
            b["ei_rev"],
            b["ea_rev"],
            b["ei_fwd"][0, k],
            b["ei_fwd"][1, k],
            b["ea_fwd"][k],
        )
        ref = in_graph_reference(m, b, k)
    assert torch.allclose(virtual, ref, atol=1e-6, rtol=0)


def test_virtual_readout_on_a_faithful_style_batch():
    """Real non-temporal, non-disjoint LinkNeighborLoader batches (Multi-GNN mode, F14): for every
    target sampled into its batch graph, the virtual readout equals the in-graph readout."""
    from torch_geometric.data import HeteroData
    from torch_geometric.loader import LinkNeighborLoader

    g = random_tied_graph(n_nodes=300, n_edges=3000, seed=4)
    ea = random_ea(np.random.default_rng(0), g.n_edges)
    to, rev = ("acct", "to", "acct"), ("acct", "rev_to", "acct")
    data = HeteroData()
    data["acct"].num_nodes = g.n_nodes
    data[to].edge_index = torch.from_numpy(np.stack([g.src, g.dst]))
    data[rev].edge_index = torch.from_numpy(np.stack([g.dst, g.src]))  # all edges flipped (§9)
    seeds = np.random.default_rng(1).permutation(g.n_edges)[:256]
    loader = LinkNeighborLoader(
        data,
        num_neighbors={to: [5, 5], rev: [5, 5]},
        edge_label_index=(to, data[to].edge_index[:, torch.from_numpy(seeds)]),
        batch_size=128,
        shuffle=False,
    )
    m = make_model(seed=1).eval()
    n_sampled = 0
    for hb in loader:
        tgt_src, tgt_dst = hb[to].edge_label_index
        tgt_gid = torch.from_numpy(seeds)[hb[to].input_id]
        x = torch.zeros(hb["acct"].num_nodes, 1)
        x[torch.unique(hb[to].edge_label_index.reshape(-1))] = 1.0
        b = {
            "x": x,
            "ei_fwd": hb[to].edge_index,
            "ea_fwd": ea[hb[to].e_id],
            "ei_rev": hb[rev].edge_index,
            "ea_rev": ea[hb[rev].e_id],  # rev e_id == gid when every edge is flipped
        }
        pos = {int(e): i for i, e in enumerate(hb[to].e_id.tolist())}
        sampled = torch.tensor([int(t) in pos for t in tgt_gid])
        k = torch.tensor([pos[int(t)] for t in tgt_gid[sampled]], dtype=torch.long)
        with torch.no_grad():
            virtual = m(
                *args_of({**b, "tgt_src": tgt_src, "tgt_dst": tgt_dst, "tgt_attr": ea[tgt_gid]})
            )
            ref = in_graph_reference(m, b, k)
        assert torch.allclose(virtual[sampled], ref, atol=1e-6, rtol=0)
        n_sampled += int(sampled.sum())
    assert n_sampled > 100  # non-vacuous


def test_empty_subgraph_scores_are_finite():
    """Bound -1 (first minute): no message edges at all, only the two roots per target."""
    m = make_model()
    e = torch.zeros(2, 0, dtype=torch.long)
    ea = torch.zeros(0, N_COLS)
    x = torch.ones(4, 1)
    tgt = random_ea(np.random.default_rng(0), 2)
    args = (x, e, ea, e, ea, torch.tensor([0, 2]), torch.tensor([1, 3]), tgt)
    for train in (True, False):
        m.train(train)
        out = m(*args)
        assert out.shape == (2, 2) and torch.isfinite(out).all()
    # an empty subgraph next to a full one keeps both finite
    full = batch(seed=2, n_sub=1)
    empty = {
        "x": torch.ones(2, 1),
        "ei_fwd": e,
        "ea_fwd": ea,
        "ei_rev": e,
        "ea_rev": ea,
        "tgt_src": torch.tensor([0]),
        "tgt_dst": torch.tensor([1]),
        "tgt_attr": tgt[:1],
    }
    m.eval()
    with torch.no_grad():
        both = m(*args_of(concat([full, empty])))
        alone = m(*args_of(empty))
    assert torch.isfinite(both).all()
    assert torch.allclose(both[1:], alone, atol=1e-6, rtol=0)


# --- encoders -----------------------------------------------------------------------------------


def test_edge_encoder_linear_plus_embeddings():
    torch.manual_seed(0)
    enc = EdgeEncoder(PRE["num_idx"], PRE["cat_idx"], CAT_SIZES, 8)
    ea = random_ea(np.random.default_rng(3), 20)
    ea[0, PRE["cat_idx"]] = -1.0  # unknown codes -> embedding row 0
    want = enc.lin(ea[:, PRE["num_idx"]])
    for k, col in enumerate(PRE["cat_idx"]):
        want = want + enc.emb[k].weight[ea[:, col].long() + 1]
    assert torch.allclose(enc(ea), want, atol=1e-6)
    row0 = enc.lin(ea[:1, PRE["num_idx"]]) + sum(e.weight[0] for e in enc.emb)
    assert torch.allclose(enc(ea[:1]), row0, atol=1e-6)
    bad = ea[:1].clone()
    bad[0, PRE["cat_idx"][0]] = float(CAT_SIZES[0])  # code == vocab size: outside the table
    with pytest.raises(IndexError):
        enc(bad)


def test_edge_encoder_numeric_only_and_validation():
    enc = EdgeEncoder(range(6), [], [], 8)  # faithful EA_f
    assert len(enc.emb) == 0 and enc(torch.randn(3, 6)).shape == (3, 8)
    with pytest.raises(ValueError):
        EdgeEncoder([0, 1], [1], [3], 8)  # overlapping columns
    with pytest.raises(ValueError):
        EdgeEncoder([0], [1, 2], [3], 8)  # sizes do not match
    with pytest.raises(ValueError):
        EdgeEncoder([], [], [], 8)


def test_per_direction_encoders():
    m = make_model()
    assert m.enc_fwd is not m.enc_rev
    fwd = {id(p) for p in m.enc_fwd.parameters()}
    assert fwd.isdisjoint(id(p) for p in m.enc_rev.parameters())
    m.eval()
    e = torch.zeros(2, 0, dtype=torch.long)
    no_edges = (
        torch.ones(2, 1),
        e,
        torch.zeros(0, N_COLS),
        e,
        torch.zeros(0, N_COLS),
        torch.tensor([0]),
        torch.tensor([1]),
        random_ea(np.random.default_rng(0), 1),
    )
    b = args_of(batch())

    def bump(enc):
        with torch.no_grad():
            for p in enc.parameters():
                p.add_(0.5)

    with torch.no_grad():
        base_empty, base = m(*no_edges), m(*b)
        bump(m.enc_rev)  # reverse encoder: only reverse message edges see it
        assert torch.equal(m(*no_edges), base_empty)
        assert not torch.allclose(m(*b), base)
        bump(m.enc_fwd)  # forward encoder: the target edge itself goes through it
        assert not torch.allclose(m(*no_edges), base_empty)


# --- parameters, build_model ------------------------------------------------------------------


def expected_params(hidden, layers, n_num, cat_sizes, readout=(50, 25), eu=True, bn=True):
    """Analytic count of §6.1's architecture (GINE)."""

    def lin(i, o):
        return i * o + o

    enc = (lin(n_num, hidden) if n_num else 0) + sum(cat_sizes) * hidden
    gine = 3 * lin(hidden, hidden)  # nn: Lin(H,H), Lin(H,H); edge Linear(edge_dim=H, H)
    emlp = lin(3 * hidden, hidden) + lin(hidden, hidden)
    dims = [3 * hidden, *readout, 2]
    head = sum(lin(a, b) for a, b in zip(dims[:-1], dims[1:], strict=True))
    n_emlp = (2 * layers - 1) if eu else 0
    return (
        lin(1, hidden)
        + 2 * enc
        + 2 * layers * gine
        + 2 * hidden * layers * bn
        + n_emlp * emlp
        + head
    )


def test_parameter_count(gnn_cfg):
    """§6.1's architecture at H 64, L 2 on HI-Small's 18-column EA: 117,985 parameters. (The
    spec's "about 134K" was measured on a probe with 2L edge-update MLPs and a shared encoder;
    §6.1 binds emlp_rev to L - 1 modules.)"""
    pre = {**PRE, "cat_sizes": HI_SMALL_CAT_SIZES}
    m = build_model(gnn_cfg, "causal", pre, {})
    assert m.hidden == 64 and m.layers == 2
    assert count_parameters(m) == expected_params(64, 2, 15, HI_SMALL_CAT_SIZES) == 117_985
    f = build_model(gnn_cfg, "faithful", {"num_idx": list(range(6))}, {})
    assert count_parameters(f) == expected_params(64, 2, 6, [])
    small = make_model(hidden=8, layers=3, edge_updates=False, batch_norm=False)
    assert count_parameters(small) == expected_params(8, 3, 15, CAT_SIZES, eu=False, bn=False)


def test_build_model_takes_params_over_config(gnn_cfg):
    params = {"hidden": 12, "layers": 3, "layer_dropout": 0.2, "final_dropout": 0.3}
    m = build_model(gnn_cfg, "causal", PRE, params)
    assert (m.hidden, m.layers, m.layer_dropout, m.conv) == (12, 3, 0.2, "gine")
    drops = [d.p for d in m.readout if isinstance(d, torch.nn.Dropout)]
    assert drops == [0.3, 0.3]
    assert [lin.out_features for lin in m.readout if isinstance(lin, torch.nn.Linear)] == [
        *gnn_cfg["model"]["readout_hidden"],
        2,
    ]
    d = build_model(gnn_cfg, "lookahead", PRE, {})
    assert d.hidden == gnn_cfg["model"]["hidden"]
    assert d.layer_dropout == gnn_cfg["model"]["layer_dropout"]
    fd = [x.p for x in d.readout if isinstance(x, torch.nn.Dropout)]
    assert fd == [gnn_cfg["train"]["final_dropout"]] * 2


def test_build_model_validation(gnn_cfg):
    with pytest.raises(ValueError, match="protocol"):
        build_model(gnn_cfg, "nope", PRE, {})
    with pytest.raises(ValueError, match="cannot use conv"):
        build_model(gnn_cfg, "causal", PRE, {"conv": "pna"})
    with pytest.raises(ValueError, match="deg"):
        build_model(gnn_cfg, "pna", PRE, {})
    with pytest.raises(ValueError, match="numeric only"):
        build_model(gnn_cfg, "faithful", PRE, {})
    with pytest.raises(ValueError, match="cover"):
        build_model(gnn_cfg, "causal", {**PRE, "num_idx": PRE["num_idx"][:-1]}, {})
    with pytest.raises(ValueError):
        MultiGINe(**model_kw(conv="gat"))
    with pytest.raises(ValueError):
        MultiGINe(**model_kw(final_dropout=1.0))


# --- PNA --------------------------------------------------------------------------------------


def hist(degrees: np.ndarray) -> np.ndarray:
    return np.bincount(degrees)


def test_pna_variant(gnn_cfg):
    rng = np.random.default_rng(0)
    deg_fwd = hist(rng.integers(0, 6, 50))
    deg_rev = hist(rng.integers(0, 3, 50))
    torch.manual_seed(0)
    m = build_model(gnn_cfg, "pna", PRE, {}, deg=(deg_fwd, deg_rev))
    pna = gnn_cfg["protocols"]["pna"]
    assert (m.conv, m.hidden) == ("pna", pna["hidden"])
    assert m.layer_dropout == pna["layer_dropout"]
    for convs, h in ((m.conv_fwd, deg_fwd), (m.conv_rev, deg_rev)):
        want = float((np.arange(len(h)) * h).sum() / h.sum())
        for c in convs:
            assert type(c).__name__ == "PNAConv" and c.towers == pna["towers"]
            assert c.aggr_module.init_avg_deg_lin == pytest.approx(want)
    b = batch(seed=3)
    m.train()
    out = m(*args_of(b))
    assert out.shape == (5, 2) and torch.isfinite(out).all()
    F.cross_entropy(out, torch.tensor([0, 1, 0, 0, 1])).backward()
    assert all(p.grad is not None for p in m.parameters())
    m.eval()
    e = torch.zeros(2, 0, dtype=torch.long)
    with torch.no_grad():
        empty = m(
            torch.ones(2, 1),
            e,
            torch.zeros(0, N_COLS),
            e,
            torch.zeros(0, N_COLS),
            torch.tensor([0]),
            torch.tensor([1]),
            b["tgt_attr"][:1],
        )
    assert torch.isfinite(empty).all()
    with pytest.raises(ValueError, match="divisible"):
        MultiGINe(**model_kw(hidden=12, conv="pna", pna_deg=(deg_fwd, deg_rev), pna_towers=5))


def test_pna_degrees_come_from_train_edges_only(gnn_cfg):
    """graph.train_degree_histograms -> PNAConv `deg`: histograms of TRAIN in-degrees (`to`) and
    of train non-self-loop out-degrees (`rev_to` in-degrees); val/test edges never count."""
    from aml.models.gnn import graph

    g = random_tied_graph(n_nodes=200, n_edges=2000, seed=2)
    split = ["train"] * 1200 + ["val_early"] * 300 + ["val_late"] * 200 + ["test"] * 300
    hg = make_host_graph(g, split=split)
    try:
        got_fwd, got_rev = graph.train_degree_histograms(hg)
    except NotImplementedError:
        pytest.skip("graph.train_degree_histograms not implemented yet (owner B)")
    tr = np.arange(g.n_edges) < 1200
    nsl = g.src != g.dst
    want_fwd = np.bincount(np.bincount(g.dst[tr], minlength=g.n_nodes))
    want_rev = np.bincount(np.bincount(g.src[tr & nsl], minlength=g.n_nodes))
    np.testing.assert_array_equal(np.asarray(got_fwd), want_fwd)
    np.testing.assert_array_equal(np.asarray(got_rev), want_rev)
    m = build_model(gnn_cfg, "pna", PRE, {}, deg=(got_fwd, got_rev))
    want = float((np.arange(len(want_fwd)) * want_fwd).sum() / want_fwd.sum())
    assert m.conv_fwd[0].aggr_module.init_avg_deg_lin == pytest.approx(want)


# --- scores ---------------------------------------------------------------------------------


def test_scores_from_logits_matches_softmax_and_is_monotone():
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2000, 2, generator=g) * 4
    s = scores_from_logits(logits)
    assert isinstance(s, np.ndarray) and s.dtype == np.float64 and s.shape == (2000,)
    p1 = torch.softmax(logits.double(), -1)[:, 1].numpy()
    np.testing.assert_allclose(s, p1, atol=1e-12, rtol=0)
    d = (logits[:, 1].double() - logits[:, 0].double()).numpy()
    order = np.argsort(d, kind="stable")
    assert np.all(np.diff(s[order]) >= 0)
    assert np.array_equal(s >= 0.5, d >= 0)


def test_scores_threshold_is_exact_for_tiny_margins():
    one = np.float32(1.0)
    rows = [
        [0.0, 0.0],  # tie -> 0.5 (>= 0.5)
        [1e-30, 0.0],  # z1 < z0 by a margin float64's sigmoid rounds to 0.5
        [0.0, 1e-30],
        [float(np.nextafter(one, np.float32(2))), 1.0],
        [1.0, float(np.nextafter(one, np.float32(2)))],
        [80.0, -80.0],
        [-80.0, 80.0],
    ]
    logits = torch.tensor(rows, dtype=torch.float32, requires_grad=True)
    s = scores_from_logits(logits * 1)
    z = logits.detach()
    assert np.array_equal(s >= 0.5, (z[:, 1] >= z[:, 0]).numpy())
    assert s[0] == 0.5 and s[1] < 0.5 <= s[2]
    assert np.all(np.isfinite(s)) and s.min() >= 0.0 and s.max() <= 1.0
    with pytest.raises(ValueError):
        scores_from_logits(torch.zeros(3))


def test_gine_index_add_sum_equals_the_default_sum():
    """review perf-1: GINE sums its messages with index_add_ (deterministic CUDA mode then sorts
    the E row indices, not E x H coordinates as scatter_add_ does). It equals PyG's default
    SumAggregation: outputs and every gradient bitwise on CPU, with deterministic algorithms on
    and off, also with no edges; the state_dict keys are the same."""
    from torch_geometric.nn import GINEConv

    from aml.models.gnn.model import IndexAddSum, _gine, _mlp

    h, n, e = 16, 50, 400
    torch.manual_seed(0)
    mine = _gine(h)
    ref = GINEConv(_mlp(h, h), edge_dim=h)  # aggr "add": PyG's scatter sum
    ref.load_state_dict(mine.state_dict())  # strict: the same keys
    assert isinstance(mine.aggr_module, IndexAddSum)
    assert list(mine.state_dict()) == list(ref.state_dict())
    ei = torch.randint(0, n, (2, e), generator=torch.Generator().manual_seed(1))

    def run(conv, ei_, n_, e_):
        x = torch.randn(n_, h, generator=torch.Generator().manual_seed(2)).requires_grad_()
        ea = torch.randn(e_, h, generator=torch.Generator().manual_seed(3)).requires_grad_()
        conv.zero_grad()
        out = conv(x, ei_, ea)
        out.square().sum().backward()
        grads = [None if p.grad is None else p.grad.clone() for p in conv.parameters()]
        return [out.detach(), x.grad, ea.grad, *grads]

    def same(a, b) -> bool:
        return all(
            (x is None and y is None) or (x is not None and y is not None and torch.equal(x, y))
            for x, y in zip(a, b, strict=True)
        )

    det, warn = (
        torch.are_deterministic_algorithms_enabled(),
        torch.is_deterministic_algorithms_warn_only_enabled(),
    )
    try:
        for flag in (False, True):
            torch.use_deterministic_algorithms(flag, warn_only=True)
            assert same(run(mine, ei, n, e), run(ref, ei, n, e))
        empty = torch.empty(2, 0, dtype=torch.long)
        assert same(run(mine, empty, 5, 0), run(ref, empty, 5, 0))
    finally:
        torch.use_deterministic_algorithms(det, warn_only=warn)


# --- the Stretch M4 path stays open -------------------------------------------------------------


def test_torch_export_with_dynamic_sizes():
    """No data-dependent Python control flow: torch.export with dynamic node/edge/target counts
    reproduces the eager logits (keeps the Stretch M4 ONNX path open)."""
    from torch.export import Dim, export

    m = make_model().eval()
    b = batch(seed=4)
    ne, nr, nn_, nt = Dim("ne"), Dim("nr"), Dim("nn"), Dim("nt")
    dyn = ({0: nn_}, {1: ne}, {0: ne}, {1: nr}, {0: nr}, {0: nt}, {0: nt}, {0: nt})
    ep = export(m, args_of(b), dynamic_shapes=dyn)
    other = batch(seed=9, n_sub=3)
    with torch.no_grad():
        for bb in (b, other):
            assert torch.allclose(ep.module()(*args_of(bb)), m(*args_of(bb)), atol=1e-6)
