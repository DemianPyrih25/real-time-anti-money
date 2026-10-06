"""Multi-GINe+EU with a virtual target edge (M3 spec §6; owner C).

Reverse message passing (separate `to` / `rev_to` convs and encoders: the to_hetero
equivalent), ports and gaps in the edge attributes, the ego flag as the only node input, edge
updates, and a readout on [relu(h_src), relu(h_dst), et] where `et` is the forward encoding of
the target's own attributes with the same per-layer edge update an in-graph edge gets. When the
target is a message edge (faithful), this equals Multi-GNN's in-graph edge readout (F16); when it
is not (causal), the target is still scored. One class serves every protocol; no data-dependent
Python control flow beyond tensor sizes (Stretch M4 ONNX stays open).

Parameter count at H 64, L 2 on the 18-column EA (15 numeric + 3 categoricals with HI-Small's
16 / 16 / 8 embedding rows): 117,985. The spec's "about 134K" came from a probe model with a
shared encoder and L reverse edge-update MLPs; §6.1 binds L - 1 (the last one is never used).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch_geometric.nn.aggr import Aggregation

PNA_AGGREGATORS = ("mean", "min", "max", "std")
PNA_SCALERS = ("identity", "amplification", "attenuation")
CONVS = ("gine", "pna")


class EdgeEncoder(nn.Module):
    """Linear(len(num_idx), hidden) on the numeric columns + the sum of one
    Embedding(cat_sizes[i], hidden) per categorical column, indexed by code + 1 (-1 -> row 0).
    Faithful: cat_idx empty (6 numeric columns)."""

    def __init__(
        self,
        num_idx: Sequence[int],
        cat_idx: Sequence[int],
        cat_sizes: Sequence[int],
        hidden: int,
    ) -> None:
        super().__init__()
        num = [int(i) for i in num_idx]
        cat = [int(i) for i in cat_idx]
        sizes = [int(s) for s in cat_sizes]
        if len(cat) != len(sizes):
            raise ValueError(f"cat_idx {cat} and cat_sizes {sizes} differ in length")
        if not num and not cat:
            raise ValueError("an edge encoder needs at least one column")
        if len(set(num + cat)) != len(num) + len(cat) or min(num + cat) < 0:
            raise ValueError(f"column indices must be distinct and >= 0: {num} / {cat}")
        if any(s < 1 for s in sizes):
            raise ValueError(f"embedding sizes must be >= 1, got {sizes}")
        self.hidden = int(hidden)
        # Buffers follow the module to its device; not persistent (the preprocess dict holds them).
        self.register_buffer("num_idx", torch.tensor(num, dtype=torch.long), persistent=False)
        self.register_buffer("cat_idx", torch.tensor(cat, dtype=torch.long), persistent=False)
        self.lin = nn.Linear(len(num), hidden) if num else None
        self.emb = nn.ModuleList(nn.Embedding(s, hidden) for s in sizes)

    def forward(self, ea: Tensor) -> Tensor:
        """(E, n_cols) float32 -> (E, hidden)."""
        out = self.lin(ea.index_select(1, self.num_idx)) if self.lin is not None else None
        if len(self.emb):
            # Codes are stored as float(code) in -1..vocab-1 (exact small ints); an index outside
            # the table raises instead of being clamped (a vocab mismatch must fail loudly).
            codes = ea.index_select(1, self.cat_idx).long() + 1
            for k, emb in enumerate(self.emb):
                e = emb(codes[:, k])
                out = e if out is None else out + e
        return out


def _lin(i: int, o: int) -> nn.Linear:
    return nn.Linear(i, o)


def _mlp(i: int, h: int) -> nn.Sequential:
    return nn.Sequential(_lin(i, h), nn.ReLU(), _lin(h, h))


class IndexAddSum(Aggregation):
    """A sum Aggregation that adds the messages with `index_add_` along the edge dimension.

    PyG's default SumAggregation calls `scatter_add_` with the index broadcast to (E, H). Under
    `use_deterministic_algorithms(True)`, CUDA runs that as an index_put_ over E x H flattened
    coordinates: a radix sort of E x H keys per aggregation (4 per forward). `index_add_` along
    dim 0 sorts only the E row indices (H-wide slices), 64x fewer keys at H 64, and is
    deterministic too. Same sum (bitwise on CPU); no parameters, so state_dict keys are
    unchanged. A CSR input (`ptr`) keeps PyG's own segment path."""

    def forward(
        self,
        x: Tensor,
        index: Tensor | None = None,
        ptr: Tensor | None = None,
        dim_size: int | None = None,
        dim: int = -2,
        max_num_elements: int | None = None,
    ) -> Tensor:
        if index is None or ptr is not None:
            return self.reduce(x, index, ptr, dim_size, dim, reduce="sum")
        d = dim if dim >= 0 else x.dim() + dim
        size = list(x.size())
        # Aggregation.__call__ always fills dim_size; no int(): it stays symbolic under export
        size[d] = dim_size if dim_size is not None else int(index.max()) + 1
        return x.new_zeros(size).index_add_(d, index, x)


def _gine(hidden: int) -> nn.Module:
    from torch_geometric.nn import GINEConv

    # eps fixed 0 (F15); the sum via index_add_ (deterministic without an E x H sort)
    return GINEConv(_mlp(hidden, hidden), edge_dim=hidden, aggr=IndexAddSum())


def _pna(hidden: int, deg: Tensor, towers: int) -> nn.Module:
    from torch_geometric.nn import PNAConv

    return PNAConv(
        hidden,
        hidden,
        aggregators=list(PNA_AGGREGATORS),
        scalers=list(PNA_SCALERS),
        deg=deg,
        edge_dim=hidden,
        towers=towers,
        pre_layers=1,
        post_layers=1,
        divide_input=False,
    )


def _as_hist(deg: Any, name: str) -> Tensor:
    """A PNAConv `deg` histogram (hist[k] = number of nodes with in-degree k) as int64."""
    t = deg if isinstance(deg, Tensor) else torch.as_tensor(np.asarray(deg))
    t = t.detach().to("cpu", torch.long).reshape(-1)
    if t.numel() == 0 or int(t.min()) < 0 or int(t.sum()) <= 0:
        raise ValueError(f"{name}: a degree histogram needs non-negative counts and >= 1 node")
    return t


class MultiGINe(nn.Module):
    """h = node_emb(x) (Linear(1, H)); ef = enc_fwd(ea_fwd); er = enc_rev(ea_rev);
    et = enc_fwd(tgt_attr). Per layer l: agg = (conv_fwd[l](h, ei_fwd, ef) + conv_rev[l](h,
    ei_rev, er)) / 2; h = (h + relu(bn[l](agg))) / 2; h = dropout(h, layer_dropout); if l < L-1:
    ef += emlp_fwd[l](cat[h[src], h[dst], ef]) / 2 and er likewise with emlp_rev[l]; always
    et += emlp_fwd[l](cat[h[tgt_src], h[tgt_dst], et]) / 2. Readout z = cat[relu(h[tgt_src]),
    relu(h[tgt_dst]), et] -> Lin(3H, r0)-ReLU-Drop(fd)-Lin(r0, r1)-ReLU-Drop(fd)-Lin(r1, 2).

    conv "gine": GINEConv(Seq(Lin(H,H), ReLU, Lin(H,H)), edge_dim=H, aggr=IndexAddSum) (eps 0;
    the sum via index_add_, see IndexAddSum).
    conv "pna": PNAConv(H, H, aggregators=PNA_AGGREGATORS, scalers=PNA_SCALERS, deg=pna_deg[0]
    (fwd) / pna_deg[1] (rev), edge_dim=H, towers=pna_towers, pre_layers=1, post_layers=1,
    divide_input=False); pna_deg from graph.train_degree_histograms (train edges only).
    emlp = Seq(Lin(3H, H), ReLU, Lin(H, H)); emlp_fwd has L modules, emlp_rev L - 1 (no unused
    parameters); bn[l] = BatchNorm1d(H) when batch_norm (else identity); edge_updates=False
    drops the emlp updates (et is then the plain forward encoding of tgt_attr).
    """

    def __init__(
        self,
        *,
        num_idx: Sequence[int],
        cat_idx: Sequence[int],
        cat_sizes: Sequence[int],
        hidden: int = 64,
        layers: int = 2,
        layer_dropout: float = 0.0098,
        final_dropout: float = 0.1053,
        readout_hidden: Sequence[int] = (50, 25),
        edge_updates: bool = True,
        batch_norm: bool = True,
        conv: str = "gine",
        pna_deg: tuple[Tensor, Tensor] | None = None,
        pna_towers: int = 5,
    ) -> None:
        super().__init__()
        if conv not in CONVS:
            raise ValueError(f"conv must be one of {CONVS}, got {conv!r}")
        hidden, layers = int(hidden), int(layers)
        if hidden < 1 or layers < 1:
            raise ValueError(f"hidden and layers must be >= 1, got {hidden} / {layers}")
        for name, p in (("layer_dropout", layer_dropout), ("final_dropout", final_dropout)):
            if not 0.0 <= float(p) < 1.0:
                raise ValueError(f"{name} must lie in [0, 1), got {p}")
        self.hidden, self.layers, self.conv = hidden, layers, conv
        self.layer_dropout = float(layer_dropout)
        self.edge_updates = bool(edge_updates)

        self.node_emb = _lin(1, hidden)
        self.enc_fwd = EdgeEncoder(num_idx, cat_idx, cat_sizes, hidden)
        self.enc_rev = EdgeEncoder(num_idx, cat_idx, cat_sizes, hidden)
        if conv == "gine":
            self.conv_fwd = nn.ModuleList(_gine(hidden) for _ in range(layers))
            self.conv_rev = nn.ModuleList(_gine(hidden) for _ in range(layers))
        else:
            if pna_deg is None or len(pna_deg) != 2:
                raise ValueError("conv 'pna' needs pna_deg = (fwd, rev) degree histograms")
            if hidden % int(pna_towers):
                raise ValueError(f"hidden {hidden} must be divisible by towers {pna_towers}")
            deg_fwd, deg_rev = _as_hist(pna_deg[0], "deg fwd"), _as_hist(pna_deg[1], "deg rev")
            self.conv_fwd = nn.ModuleList(
                _pna(hidden, deg_fwd, int(pna_towers)) for _ in range(layers)
            )
            self.conv_rev = nn.ModuleList(
                _pna(hidden, deg_rev, int(pna_towers)) for _ in range(layers)
            )
        self.bn = nn.ModuleList(
            nn.BatchNorm1d(hidden) if batch_norm else nn.Identity() for _ in range(layers)
        )
        n_fwd, n_rev = (layers, layers - 1) if self.edge_updates else (0, 0)
        self.emlp_fwd = nn.ModuleList(_mlp(3 * hidden, hidden) for _ in range(n_fwd))
        self.emlp_rev = nn.ModuleList(_mlp(3 * hidden, hidden) for _ in range(n_rev))

        dims = [3 * hidden, *(int(d) for d in readout_hidden)]
        head: list[nn.Module] = []
        for a, b in zip(dims[:-1], dims[1:], strict=True):
            head += [_lin(a, b), nn.ReLU(), nn.Dropout(float(final_dropout))]
        head.append(_lin(dims[-1], 2))
        self.readout = nn.Sequential(*head)

    def forward(
        self,
        x: Tensor,
        ei_fwd: Tensor,
        ea_fwd: Tensor,
        ei_rev: Tensor,
        ea_rev: Tensor,
        tgt_src: Tensor,
        tgt_dst: Tensor,
        tgt_attr: Tensor,
    ) -> Tensor:
        """x (N, 1) float; ei_* (2, E*) int64 local; ea_* (E*, n_cols) float32; tgt_src/tgt_dst
        (B,) int64 local roots; tgt_attr (B, n_cols) -> logits (B, 2). An empty subgraph (no
        edges) still yields finite logits."""
        h = self.node_emb(x)
        ef = self.enc_fwd(ea_fwd)
        er = self.enc_rev(ea_rev)
        et = self.enc_fwd(tgt_attr)  # the target as a `to` edge that sends no message
        last = self.layers - 1
        for i in range(self.layers):
            agg = (self.conv_fwd[i](h, ei_fwd, ef) + self.conv_rev[i](h, ei_rev, er)) / 2
            h = (h + F.relu(self.bn[i](agg))) / 2
            # placement ours: Multi-GNN's GINe / PNA forward may not apply `dropout` at all
            # (recalled, unverified; listed in train.RECALLED)
            h = F.dropout(h, self.layer_dropout, self.training)
            if self.edge_updates:
                if i < last:  # last-layer message-edge updates never reach the output
                    ef = ef + self.emlp_fwd[i](torch.cat([h[ei_fwd[0]], h[ei_fwd[1]], ef], -1)) / 2
                    er = er + self.emlp_rev[i](torch.cat([h[ei_rev[0]], h[ei_rev[1]], er], -1)) / 2
                et = et + self.emlp_fwd[i](torch.cat([h[tgt_src], h[tgt_dst], et], -1)) / 2
        z = torch.cat([F.relu(h[tgt_src]), F.relu(h[tgt_dst]), et], -1)
        return self.readout(z)


def build_model(
    gnn_cfg: Mapping[str, Any],
    protocol: str,
    preprocess: Mapping[str, Any],
    params: Mapping[str, Any],
    deg: tuple[Any, Any] | None = None,
) -> MultiGINe:
    """The model of one run. Architecture from gnn_cfg["model"] (readout_hidden, edge_updates,
    batch_norm, and the defaults of hidden / layers / layer_dropout), overridden by the protocol's
    section (pna: hidden, towers, dropouts; faithful: dropouts) and then by `params` (the
    effective hyperparameters of train.effective_params: hidden, layers, layer_dropout,
    final_dropout, conv, towers). Columns from `preprocess` (num_idx, cat_idx, cat_sizes; every
    column of preprocess["columns"] is used exactly once; the faithful preprocess has 6 numeric
    columns and no categoricals). protocol "pna" (and only it) uses conv "pna" and requires `deg`
    = (fwd, rev) in-degree histograms (hist[k] = nodes with in-degree k, train edges only);
    `deg` is ignored for GINE. Not moved to a device; seed torch before calling."""
    from aml.models.gnn import PROTOCOLS

    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    m = gnn_cfg["model"]
    hp: dict[str, Any] = {
        "hidden": m["hidden"],
        "layers": m["layers"],
        "layer_dropout": m["layer_dropout"],
        "final_dropout": gnn_cfg["train"]["final_dropout"],
        "conv": "pna" if protocol == "pna" else m["conv"],
        "towers": gnn_cfg["protocols"]["pna"]["towers"],
    }
    proto_keys = {
        "pna": ("hidden", "towers", "layer_dropout", "final_dropout"),
        "faithful": ("layer_dropout", "final_dropout"),
    }.get(protocol, ())
    hp.update({k: gnn_cfg["protocols"][protocol][k] for k in proto_keys})
    hp.update({k: params[k] for k in hp if k in params})
    if (hp["conv"] == "pna") != (protocol == "pna"):
        raise ValueError(f"protocol {protocol!r} cannot use conv {hp['conv']!r}")
    if hp["conv"] == "pna" and deg is None:
        raise ValueError("protocol 'pna' needs deg = graph.train_degree_histograms(g)")

    num_idx = [int(i) for i in preprocess["num_idx"]]
    cat_idx = [int(i) for i in preprocess.get("cat_idx", [])]
    cat_sizes = [int(s) for s in preprocess.get("cat_sizes", [])]
    columns = preprocess.get("columns")
    if columns is not None and sorted(num_idx + cat_idx) != list(range(len(columns))):
        raise ValueError(
            f"num_idx {num_idx} + cat_idx {cat_idx} must cover the {len(columns)} columns once"
        )
    if protocol == "faithful" and cat_idx:
        raise ValueError("the faithful edge attributes are numeric only (no embeddings, §9)")

    return MultiGINe(
        num_idx=num_idx,
        cat_idx=cat_idx,
        cat_sizes=cat_sizes,
        hidden=int(hp["hidden"]),
        layers=int(hp["layers"]),
        layer_dropout=float(hp["layer_dropout"]),
        final_dropout=float(hp["final_dropout"]),
        readout_hidden=tuple(int(d) for d in m["readout_hidden"]),
        edge_updates=bool(m["edge_updates"]),
        batch_norm=bool(m["batch_norm"]),
        conv=str(hp["conv"]),
        pna_deg=deg if hp["conv"] == "pna" else None,
        pna_towers=int(hp["towers"]),
    )


_BELOW_HALF = float(np.nextafter(0.5, 0.0))


def scores_from_logits(logits: Tensor) -> np.ndarray:
    """float64 scores = sigmoid(z1 - z0) (== softmax p1), computed in float64 on the host;
    score >= 0.5 <=> z1 >= z0 exactly (a negative margin too small for float64's sigmoid is
    clamped just below 0.5), so argmax <=> score >= 0.5 up to exact ties (measure-zero)."""
    if logits.dim() != 2 or logits.shape[1] != 2:
        raise ValueError(f"logits must be (B, 2), got {tuple(logits.shape)}")
    z = logits.detach().to(device="cpu", dtype=torch.float64)
    d = z[:, 1] - z[:, 0]  # exact sign: float32 values are exact in float64
    s = torch.sigmoid(d)
    s = torch.where(d < 0, torch.clamp(s, max=_BELOW_HALF), s)
    return s.numpy().astype(np.float64, copy=False)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))
