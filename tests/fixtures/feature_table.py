"""A schema-valid synthetic feature table (M2 spec §5.10) built from the prepared fixture.

For the model and evaluation tests (gate, LightGBM-graph, evaluate) without the engine. The layout
is the Volume's: feature_spec.json, vocab.json, summary.json and parts/part-dNN.parquet (one per
simulated day, rank order, `spec.table_schema()` dtypes, features cast to Float32).

Values: the TX group is M1's `build_tx_features`; most windowed features (counts, distinct
counterparties, usd sums, log-amount moments and maxima, pair counts, inflow, balances, ports,
gaps, rule counts, cyc2) are computed causally from strictly earlier minutes with vectorised
searchsorted windows; cyc3, cyc4 and the scatter-gather counts are seeded sparse noise; the
severities are 0 and the trunc flags 0. This is NOT the engine's output and never a reference for
it (the brute force and the oracle are): it only has the engine table's shape and plausible,
label-free values.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import polars as pl
import yaml

from aml.features.spec import (
    FEATURE_SPEC_FILE,
    INFLOW_COLUMN,
    PARTS_DIR,
    SEVERITY_COLUMNS,
    SUMMARY_FILE,
    TRUNC_COLUMNS,
    VOCAB_FILE,
    EngineSpec,
    format_slug,
    part_name,
    window_tag,
)
from aml.features.tx_features import TX_FEATURES, build_tx_features, fit_vocab
from aml.io import write_json_atomic, write_parquet_atomic
from aml.paths import DataPaths

CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs"
BIG = 1 << 32  # key = id * BIG + minute; minutes are far below 2^32
# Share of non-zero rows of the noise features (cyc3 / cyc4 stay under the gate's 100-row floor).
NOISE_NONZERO = {"cyc3": 0.01, "cyc4": 0.003, "sg_mids": 0.05, "sg_srcs": 0.05}
TX_COLUMNS = [
    "row_id",
    "rank",
    "day",
    "split",
    "minute",
    "src",
    "dst",
    "from_bank",
    "to_bank",
    "amount_paid",
    "amount_usd",
    "payment_currency",
    "receiving_currency",
    "payment_format",
]


def _load_cfg(name: str) -> dict:
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def default_cfgs() -> tuple[dict, dict]:
    """configs/features.yaml and configs/rules.yaml."""
    return _load_cfg("features.yaml"), _load_cfg("rules.yaml")


class _Stream:
    """Events keyed by an int id, sorted by (id, minute); windows [m - W, m - 1] by bisection."""

    def __init__(self, key: np.ndarray, minute: np.ndarray) -> None:
        k = key.astype(np.int64) * BIG + minute.astype(np.int64)
        self.order = np.argsort(k, kind="stable")
        self.k = k[self.order]

    def bounds(self, qkey: np.ndarray, qmin: np.ndarray, window: int | None):
        """[lo, hi) positions of the qkey rows with minute in [m - W, m - 1] (W None: all < m)."""
        base = qkey.astype(np.int64) * BIG
        start = 0 if window is None else np.maximum(qmin.astype(np.int64) - window, 0)
        lo = np.searchsorted(self.k, base + start, side="left")
        hi = np.searchsorted(self.k, base + qmin.astype(np.int64), side="left")
        return lo, hi

    def sum(self, values: np.ndarray, qkey, qmin, window) -> np.ndarray:
        v = np.asarray(values)[self.order]
        cs = np.concatenate([np.zeros(1, dtype=v.dtype), np.cumsum(v)])
        lo, hi = self.bounds(qkey, qmin, window)
        return cs[hi] - cs[lo]

    def count(self, qkey, qmin, window) -> np.ndarray:
        lo, hi = self.bounds(qkey, qmin, window)
        return (hi - lo).astype(np.int64)

    def distinct(self, other: np.ndarray, qkey, qmin, window) -> np.ndarray:
        cp = np.asarray(other)[self.order].tolist()
        lo, hi = self.bounds(qkey, qmin, window)
        return np.array([len(set(cp[a:b])) for a, b in zip(lo, hi, strict=True)], dtype=np.int64)

    def max(self, values: np.ndarray, qkey, qmin, window) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)[self.order]
        lo, hi = self.bounds(qkey, qmin, window)
        return np.array([v[a:b].max() if b > a else np.nan for a, b in zip(lo, hi, strict=True)])

    def gap(self, qkey, qmin, cap: int) -> np.ndarray:
        """min(m - last earlier minute, cap); 0 = no earlier event."""
        base = qkey.astype(np.int64) * BIG
        idx = np.searchsorted(self.k, base + qmin, side="left") - 1
        safe = np.maximum(idx, 0)
        has = (idx >= 0) & (self.k[safe] >= base)
        last = self.k[safe] - base
        return np.where(has, np.minimum(qmin - last, cap), 0).astype(np.int64)


def _lsum(cents: np.ndarray) -> np.ndarray:
    return np.log1p(cents / 100.0)


def feature_frame(paths: DataPaths, spec: EngineSpec, rules_cfg: dict, seed: int = 0):
    """(key columns, {feature: float64 values}, inflow_c) for every transaction, rank order."""
    tx = pl.read_parquet(paths.transactions, columns=TX_COLUMNS).sort("rank")
    vocab = {k: list(v) for k, v in spec.vocab.items()}
    tx_feats = build_tx_features(tx, vocab, rules_cfg["round_unit"])
    n = tx.height
    m = tx["minute"].to_numpy().astype(np.int64)
    u = tx["src"].to_numpy().astype(np.int64)
    v = tx["dst"].to_numpy().astype(np.int64)
    usd = tx["amount_usd"].to_numpy()
    cents = (tx["amount_usd"] * 100).round(0, mode="half_away_from_zero").cast(pl.Int64).to_numpy()
    lv = np.log1p(usd)
    fmt = tx_feats["payment_format"].to_numpy().astype(np.int64)
    self_loop = u == v
    in_band = (usd >= spec.band_low_usd) & (usd < spec.band_high_usd)
    rnd = tx_feats["round_amount"].to_numpy().astype(bool)
    n_acc = spec.n_accounts
    S, L, PT, RT = spec.w_short, spec.w_long, spec.w_pt, spec.w_rt

    out_s, in_s = _Stream(u, m), _Stream(v, m)
    pair = u * n_acc + v
    rev = v * n_acc + u
    pair_s = _Stream(pair, m)
    f: dict[str, np.ndarray] = {
        name: tx_feats[name].to_numpy().astype(np.float64) for name in TX_FEATURES
    }

    tag = window_tag
    # VEL and AMT on both windows: (side, direction) -> stream, query account, counterparty.
    sides = {("u", "out"): (out_s, u), ("u", "in"): (in_s, u), ("v", "in"): (in_s, v)}
    sides[("v", "out")] = (out_s, v)
    cps = {"out": v, "in": u}  # the counterparty of each stream event
    uniq_1d = {}
    for (side, d), (stream, acct) in sides.items():
        for w in (S, L):
            t = tag(w)
            f[f"{side}_{d}_cnt_{t}"] = stream.count(acct, m, w).astype(np.float64)
            uq = stream.distinct(cps[d], acct, m, w)
            f[f"{side}_{d}_uniq_{t}"] = uq.astype(np.float64)
            if w == S:
                uniq_1d[(side, d)] = uq
            f[f"{side}_{d}_sum_{t}"] = _lsum(stream.sum(cents, acct, m, w))
    for side, d in (("u", "out"), ("v", "in")):
        stream, acct = sides[(side, d)]
        for w in (S, L):
            t = tag(w)
            cnt = stream.count(acct, m, w)
            s1, s2 = stream.sum(lv, acct, m, w), stream.sum(lv * lv, acct, m, w)
            with np.errstate(invalid="ignore", divide="ignore"):
                mean = np.where(cnt > 0, s1 / np.maximum(cnt, 1), np.nan)
                var = np.where(cnt > 0, s2 / np.maximum(cnt, 1) - mean * mean, np.nan)
            f[f"{side}_{d}_mean_{t}"] = mean
            f[f"{side}_{d}_std_{t}"] = np.where(cnt > 0, np.sqrt(np.maximum(var, 0.0)), np.nan)
        f[f"{side}_{d}_max_{tag(S)}"] = stream.max(lv, acct, m, S)
        f[f"{side}_amt_dev_{tag(S)}"] = lv - f[f"{side}_{d}_mean_{tag(S)}"]

    # FLOW.
    for w in (S, L):
        f[f"pair_cnt_{tag(w)}"] = pair_s.count(pair, m, w).astype(np.float64)
    inflow = in_s.sum(np.where(self_loop, 0, cents), u, m, PT)
    f[f"u_inflow_{tag(PT)}"] = _lsum(inflow)
    with np.errstate(invalid="ignore", divide="ignore"):
        f[f"pt_ratio_{tag(PT)}"] = np.where(
            ~self_loop & (inflow > 0), cents / np.maximum(inflow, 1), np.nan
        )
    for side, acct in (("u", u), ("v", v)):
        o, i = out_s.sum(cents, acct, m, L), in_s.sum(cents, acct, m, L)
        with np.errstate(invalid="ignore", divide="ignore"):
            f[f"{side}_bal_{tag(L)}"] = np.where(o + i > 0, (o - i) / np.maximum(o + i, 1), np.nan)

    # PORT: lifetime pair relation; a pair's ports are fixed at its first minute.
    is_new = pair_s.count(pair, m, None) == 0
    f["pair_is_new"] = is_new.astype(np.float64)
    first = (
        pl.DataFrame({"pair": pair, "u": u, "v": v, "m": m})
        .group_by("pair")
        .agg(pl.col("u").first(), pl.col("v").first(), pl.col("m").min().alias("fm"))
    )
    fm_of = dict(zip(first["pair"].to_list(), first["fm"].to_list(), strict=True))
    fm = np.array([fm_of[p] for p in pair.tolist()], dtype=np.int64)
    pu, pv, pfm = (first[c].to_numpy().astype(np.int64) for c in ("u", "v", "fm"))
    port_out = _Stream(pu, pfm).count(u, fm, None)
    port_in = _Stream(pv, pfm).count(v, fm, None)
    f["out_port"] = np.log1p(np.minimum(port_out, spec.cap_port).astype(np.float64))
    f["in_port"] = np.log1p(np.minimum(port_in, spec.cap_port).astype(np.float64))
    for (side, d), (stream, acct) in sides.items():
        f[f"{side}_{d}_gap"] = stream.gap(acct, m, spec.cap_gap).astype(np.float64)
    f["pair_gap"] = pair_s.gap(pair, m, spec.cap_gap).astype(np.float64)
    f["rev_pair_gap"] = pair_s.gap(rev, m, spec.cap_gap).astype(np.float64)

    # CYC / SG: cyc2 exact; the rest seeded sparse noise.
    rng = np.random.default_rng(seed)
    c2 = np.where(self_loop, 0, pair_s.count(rev, m, RT))
    f[f"cyc2_{tag(RT)}"] = np.minimum(c2, spec.cap_count).astype(np.float64)
    for k in (3, 4):
        nz = rng.random(n) < NOISE_NONZERO[f"cyc{k}"]
        f[f"cyc{k}_{tag(RT)}"] = np.where(nz & ~self_loop, rng.integers(1, 4, n), 0).astype(float)
    g = tag(spec.w_sg)
    for name in ("sg_mids", "sg_srcs"):
        nz = rng.random(n) < NOISE_NONZERO[name]
        f[f"{name}_{g}"] = np.where(nz & ~self_loop, rng.integers(1, 6, n), 0).astype(float)
    for side in ("u", "v"):
        gs = np.minimum(uniq_1d[(side, "in")], uniq_1d[(side, "out")])
        f[f"gs_{side}_{tag(S)}"] = gs.astype(np.float64)

    # RULE.
    f["in_band"] = in_band.astype(np.float64)
    f[f"u_out_inband_{tag(S)}"] = out_s.sum(in_band.astype(np.int64), u, m, S).astype(float)
    f[f"u_out_round_{tag(S)}"] = out_s.sum(rnd.astype(np.int64), u, m, S).astype(float)
    f[f"u_out_newcp_{tag(S)}"] = out_s.sum(is_new.astype(np.int64), u, m, S).astype(float)
    same = np.zeros(n, dtype=np.int64)
    for k, name in enumerate(spec.vocab["payment_format"]):
        hit = (fmt == k).astype(np.int64)
        f[f"u_out_fmt_{format_slug(name)}_{tag(S)}"] = out_s.sum(hit, u, m, S).astype(float)
        same = np.where(fmt == k, in_s.sum(hit, v, m, S), same)
    f[f"v_in_same_fmt_{tag(S)}"] = same.astype(np.float64)

    missing = [x for x in spec.feature_names if x not in f]
    extra = sorted(set(f) - set(spec.feature_names))
    if missing or extra:
        raise RuntimeError(f"fixture features out of sync with the spec: {missing} / {extra}")
    keys = tx.select("row_id", "rank", "day", "split")
    return keys, f, inflow.astype(np.int64)


def build_feature_table(
    paths: DataPaths,
    out_dir: Path,
    *,
    features_cfg: dict | None = None,
    rules_cfg: dict | None = None,
    seed: int = 0,
    hub_cap: int = 1_000_000,
) -> EngineSpec:
    """Write the table under `out_dir` (a features_dir); returns its EngineSpec.

    The vocab is fitted on train rows (as build_features does); there are no hubs.
    """
    default_f, default_r = default_cfgs()
    features_cfg = copy.deepcopy(features_cfg or default_f)
    rules_cfg = copy.deepcopy(rules_cfg or default_r)
    tx = pl.read_parquet(paths.transactions, columns=TX_COLUMNS)
    vocab = fit_vocab(tx.filter(pl.col("split") == "train"))
    n_accounts = pl.read_parquet(paths.accounts).height
    spec = EngineSpec.from_configs(
        features_cfg, rules_cfg, n_accounts=n_accounts, vocab=vocab, hub_cap=hub_cap, hubs=()
    )
    keys, f, inflow = feature_frame(paths, spec, rules_cfg, seed)
    n = keys.height
    cols: dict[str, pl.Series] = {c: keys[c] for c in keys.columns}
    for name in spec.feature_names:
        x = np.asarray(f[name], dtype=np.float64)
        if x.shape != (n,) or np.isinf(x).any():
            raise RuntimeError(f"bad fixture values for {name}")
        cols[name] = pl.Series(name, x.astype(np.float32), dtype=pl.Float32)
    for s in SEVERITY_COLUMNS:
        cols[s] = pl.Series(s, np.zeros(n), dtype=pl.Float64)
    cols[INFLOW_COLUMN] = pl.Series(INFLOW_COLUMN, inflow, dtype=pl.Int64)
    for t in TRUNC_COLUMNS:
        cols[t] = pl.Series(t, np.zeros(n, dtype=np.int8), dtype=pl.Int8)
    schema = spec.table_schema()
    table = pl.DataFrame([cols[c] for c in schema]).cast(schema)
    if table.schema != pl.Schema(schema):
        raise RuntimeError("fixture table does not match spec.table_schema()")

    out_dir = Path(out_dir)
    for (day,), part in table.group_by("day", maintain_order=True):
        write_parquet_atomic(part, out_dir / PARTS_DIR / part_name(int(day)))
    write_json_atomic(spec.to_json(), out_dir / FEATURE_SPEC_FILE)
    write_json_atomic({k: list(v) for k, v in spec.vocab.items()}, out_dir / VOCAB_FILE)
    write_json_atomic(
        {"fixture": True, "rows": n, "spec_hash": spec.spec_hash(), "seed": seed},
        out_dir / SUMMARY_FILE,
    )
    return spec
