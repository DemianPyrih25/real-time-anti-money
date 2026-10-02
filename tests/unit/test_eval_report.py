"""aml.eval.report: eval frame, evaluate() invariants, markdown, and the stage end to end."""

from __future__ import annotations

import copy
import json

import numpy as np
import polars as pl
import pytest

from aml.config import rate_tag
from aml.eval.report import (
    EVAL_COLUMNS,
    build_eval_frame,
    evaluate,
    render_markdown,
    run_evaluate_stage,
)
from aml.eval.typology import TYPOLOGIES
from aml.io import read_json, write_parquet_atomic
from aml.paths import DataPaths

SPLIT_DAYS = {"train": (1, 6), "val_early": (7, 7), "val_late": (8, 8), "test": (9, 18)}


def _split_of(day: np.ndarray) -> np.ndarray:
    out = np.empty(day.size, dtype=object)
    for name, (lo, hi) in SPLIT_DAYS.items():
        out[(day >= lo) & (day <= hi)] = name
    return out


def _mini_paths(root, seed: int = 0, per_day: int = 2500) -> DataPaths:
    """A hand-made transactions + labels pair in the canonical layout (no data layer needed)."""
    rng = np.random.default_rng(seed)
    days = np.r_[np.repeat(np.arange(1, 11), per_day), np.repeat(np.arange(11, 19), 20)]
    n = days.size
    minute = (days - 1) * 1440 + rng.integers(0, 1440, n)
    row_id = rng.permutation(n)  # file order differs from time order
    order = np.lexsort((row_id, minute))
    rank = np.empty(n, dtype=np.int64)
    rank[order] = np.arange(n)
    src = rng.integers(0, 600, n)
    dst = rng.integers(0, 600, n)
    tail = days >= 11
    y = (rng.random(n) < np.where(tail, 0.6, 0.01)).astype(np.int8)
    pattern = (y == 1) & (rng.random(n) < 0.6)
    attempt = np.where(pattern, rng.integers(0, 40, n), -1)
    typ = np.where(y == 1, "OTHER", None).astype(object)
    typ[pattern] = np.array(TYPOLOGIES[:8], dtype=object)[attempt[pattern] % 8]
    tx = pl.DataFrame(
        {
            "row_id": pl.Series(row_id, dtype=pl.Int64),
            "rank": pl.Series(rank, dtype=pl.Int64),
            "minute": pl.Series(minute, dtype=pl.Int64),
            "day": pl.Series(days, dtype=pl.Int16),
            "split": pl.Series(_split_of(days).tolist(), dtype=pl.String),
            "src": pl.Series(src, dtype=pl.Int32),
            "dst": pl.Series(dst, dtype=pl.Int32),
        }
    ).sort("rank")
    labels = pl.DataFrame(
        {
            "row_id": pl.Series(row_id, dtype=pl.Int64),
            "is_laundering": pl.Series(y, dtype=pl.Int8),
            "attempt_id": pl.Series(np.where(pattern, attempt, None).tolist(), dtype=pl.Int32),
            "typology": pl.Series(typ.tolist(), dtype=pl.String),
            "attempt_size": pl.Series(np.where(pattern, 5, None).tolist(), dtype=pl.Int32),
            "typology_detail": pl.Series([None] * n, dtype=pl.String),
        }
    ).sort("row_id")
    paths = DataPaths(root)
    write_parquet_atomic(tx, paths.transactions)
    write_parquet_atomic(labels, paths.labels)
    return paths


def _cfgs(data_cfg: dict, rules_cfg: dict, B: int = 60) -> tuple[dict, dict]:
    d = copy.deepcopy(data_cfg)
    d["evaluation"]["bootstrap_replicates"] = B
    return d, copy.deepcopy(rules_cfg)


def _rates(rules_cfg: dict) -> list[float]:
    return [rules_cfg["alert_rate"], *rules_cfg["sensitivity_alert_rates"]]


def _fake_inputs(ev: pl.DataFrame, rules_cfg: dict, seed: int = 0):
    """Rule flags (weakly informative) per rate and model scores (informative, with ties)."""
    rng = np.random.default_rng(seed)
    y = ev["y"].to_numpy()
    base = rng.random(ev.height) - 0.3 * y
    flags = {rate_tag(r): base < r for r in _rates(rules_cfg)}
    s = [
        np.round(rng.random(ev.height) * 0.6 + y * rng.random(ev.height) * 0.5, 3) for _ in range(3)
    ]
    scores = {"lgbm_tx": np.stack(s[:2]), "single": s[2][None, :]}
    return flags, scores


@pytest.fixture(scope="module")
def mini(tmp_path_factory):
    paths = _mini_paths(tmp_path_factory.mktemp("mini"))
    return paths, build_eval_frame(paths, {"split": SPLIT_DAYS})


@pytest.fixture(scope="module")
def evaluated(mini, data_cfg, rules_cfg):
    _, ev = mini
    dcfg, rcfg = _cfgs(data_cfg, rules_cfg)
    flags, scores = _fake_inputs(ev, rcfg)
    return evaluate(ev, flags, scores, dcfg, rcfg, threads=2), ev, flags, scores, dcfg, rcfg


def test_build_eval_frame(mini):
    paths, ev = mini
    assert ev.columns == EVAL_COLUMNS
    assert set(ev["split"].unique()) == {"val_late", "test"}
    assert ev["rank"].is_sorted()
    assert ev["y"].dtype == pl.Int8 and ev["y"].null_count() == 0
    tx = pl.read_parquet(paths.transactions)
    lab = pl.read_parquet(paths.labels)
    assert ev.height == tx.filter(pl.col("split").is_in(["val_late", "test"])).height
    # seen_launderer re-derived in plain Python
    j = tx.join(lab, on="row_id")
    train_pos = j.filter((pl.col("split") == "train") & (pl.col("is_laundering") == 1))
    bad = set(train_pos["src"]) | set(train_pos["dst"])
    expect = [s in bad or d in bad for s, d in zip(ev["src"], ev["dst"], strict=True)]
    assert ev["seen_launderer"].to_list() == expect
    assert 0 < sum(expect) < ev.height
    # labels joined by row_id
    y_by_id = dict(zip(lab["row_id"], lab["is_laundering"], strict=True))
    assert ev["y"].to_list() == [y_by_id[r] for r in ev["row_id"]]


def test_evaluate_invariants(evaluated):
    res, ev, flags, scores, dcfg, rcfg = evaluated
    json.dumps(res, allow_nan=False)  # strict JSON
    head = res["meta"]["headline_tag"]
    assert head == rate_tag(rcfg["alert_rate"])
    assert set(res["meta"]["rates"]) == {rate_tag(r) for r in _rates(rcfg)}
    assert set(res["views"]) == {"primary", "tail", "full"}
    val = (ev["split"] == "val_late").to_numpy()
    for name in scores:
        for tag in res["meta"]["rates"]:
            th = res["thresholds"][name]["rate"][tag]
            assert th["rules_val_late_rate"] == pytest.approx(flags[tag][val].mean())
            # (a) never exceeds the rules' val_late volume
            for rate in th["model_val_late_rate"]:
                assert rate <= th["rules_val_late_rate"] + 1e-12
    day = ev["day"].to_numpy()
    test = (ev["split"] == "test").to_numpy()
    for v, (lo, hi) in dcfg["test_views"].items():
        vr = res["views"][v]
        mask = test & (day >= lo) & (day <= hi)
        assert res["meta"]["views"][v]["rows"] == int(mask.sum())
        for tag in res["meta"]["rates"]:
            rp = vr["rules"]["points"][tag]
            assert rp["alerts"] == int(flags[tag][mask].sum())
            assert rp["pr_auc"] == "n/a" and rp["f1_thr"] == "n/a"
            for name in scores:
                b = vr["models"][name]["b"][tag]
                assert b["alerts"]["mean"] == rp["alerts"]  # iso-volume: K = rules' alerts
        for name in scores:
            c = vr["models"][name]["c"]
            assert c["model_same_volume"]["alerts"]["mean"] == c["union"]["alerts"]["mean"]
    one = res["views"]["primary"]["models"]["single"]["a"][head]["recall"]
    assert one["std"] is None and len(one["per_seed"]) == 1
    two = res["views"]["primary"]["models"]["lgbm_tx"]["a"][head]["recall"]
    assert two["std"] is not None
    assert two["mean"] == pytest.approx(np.mean(two["per_seed"]))


def test_bootstrap_block(evaluated):
    res, *_ = evaluated
    boot = res["bootstrap"]
    assert boot["B"] == 60 and boot["view"] == "primary"
    assert (
        sum(s["rows"] for s in boot["strata"].values()) == res["meta"]["views"]["primary"]["rows"]
    )
    for name in ("lgbm_tx", "single"):
        for key, iv in boot["models"][name].items():
            assert iv["lo"] <= iv["hi"], key
        d = boot["diffs"][name]["a.recall_minus_rules"]
        pv = res["views"]["primary"]
        head = res["meta"]["headline_tag"]
        point = (
            pv["models"][name]["a"][head]["recall"]["mean"] - pv["rules"]["points"][head]["recall"]
        )
        assert d["point"] == pytest.approx(point)
        assert d["lo"] <= d["hi"]


def test_thresholds_depend_on_val_late_only(mini, data_cfg, rules_cfg):
    _, ev = mini
    dcfg, rcfg = _cfgs(data_cfg, rules_cfg, B=0)
    flags, scores = _fake_inputs(ev, rcfg)
    base = evaluate(ev, flags, scores, dcfg, rcfg)
    test = (ev["split"] == "test").to_numpy()
    rng = np.random.default_rng(5)
    y = ev["y"].to_numpy().copy()
    y[test] = rng.permutation(y[test])
    scores2 = {k: v.copy() for k, v in scores.items()}
    for v in scores2.values():
        v[:, test] = rng.random((v.shape[0], int(test.sum())))
    other = evaluate(ev.with_columns(y=pl.Series(y, dtype=pl.Int8)), flags, scores2, dcfg, rcfg)
    assert other["thresholds"] == base["thresholds"]
    assert "bootstrap" not in base
    md = render_markdown(base)  # renders without CIs
    assert "Paired differences" not in md and "## Headline: primary period" in md


def test_multi_seed_bootstrap_is_the_mean_over_seeds(mini, data_cfg, rules_cfg):
    _, ev = mini
    dcfg, rcfg = _cfgs(data_cfg, rules_cfg, B=40)
    flags, scores = _fake_inputs(ev, rcfg)
    s = scores["single"][0]
    res = evaluate(ev, flags, {"one": s[None, :], "dup": np.stack([s, s])}, dcfg, rcfg)
    # two identical seeds -> the per-replicate mean equals the single seed, so identical CIs
    assert res["bootstrap"]["models"]["dup"] == res["bootstrap"]["models"]["one"]
    assert res["bootstrap"]["diffs"]["dup"] == res["bootstrap"]["diffs"]["one"]


def test_render_markdown(evaluated):
    res, *_ = evaluated
    md = render_markdown(res)
    for needle in (
        "## Headline: primary period",
        "days 9-10",
        "(a) deployable threshold",
        "(b) top-K, K = rules' alerts",
        "(c) rules ∪ model (a)",
        "(c) model alone, same volume",
        "Paired differences",
        "## Test view: tail",
        "## Test view: full",
        "## Sensitivity to the alert budget",
        "| 0.1% |",
        "| 1% |",
        "## Recall per typology at (a)",
        "## Attempt-level detection at (a)",
        "## Memorisation check at (a)",
        "single (1 seed, no std)",
        "lgbm_tx (2 seeds)",
        "## Literature reference (published numbers, not directly comparable)",
        "| Multi-PNA+EU | 68.2 |",
        "| LightGBM, raw transaction features | 21.3 |",
        "argmax",
        "synthetic",
    ):
        assert needle in md, needle
    # rules: PR-AUC and threshold F1 are n/a; evaluate() alone does not know the active scenarios
    lit = md.split("## Literature-comparable metrics")[1].split("\n## ")[0]
    assert "| Rules (SQL) |" + " n/a |" * 9 in lit
    # both the headline period and the full test period (the published numbers' period)
    assert "**primary** (headline, days 9-10)" in lit
    assert "**full** (days 9-18: the test period the published numbers use" in lit
    assert lit.count("| lgbm_tx (2 seeds) |") == 2
    assert lit.count("| single (1 seed, no std) |") == 2
    # the literature table comes after our results, in its own section
    assert md.index("## Literature reference") > md.index("## Headline: primary period")


def _write_stage_inputs(paths: DataPaths, rules_cfg: dict, root, seed: int = 0):
    """flags.parquet and scores.parquet for every transaction, rows shuffled."""
    tx = (
        pl.read_parquet(paths.transactions)
        .select("row_id", "split", "day")
        .sample(fraction=1.0, shuffle=True, seed=seed)
    )
    rng = np.random.default_rng(seed)
    n = tx.height
    flags = tx.with_columns(
        **{f"rules_any_{rate_tag(r)}": pl.Series(rng.random(n) < r) for r in _rates(rules_cfg)},
        fired_fan_in_velocity=pl.Series(rng.random(n) < 0.002),
        fired_structuring=pl.Series(rng.random(n) < 0.002),
    )
    rules_dir = root / "rules"
    write_parquet_atomic(flags, rules_dir / "flags.parquet")
    model_dir = root / "lgbm"
    scores = tx.select("row_id", "split").with_columns(
        score_s1=pl.Series(rng.random(n)), score_s0=pl.Series(rng.random(n))
    )
    write_parquet_atomic(scores, model_dir / "scores.parquet")
    return rules_dir, {"lgbm_tx": model_dir}


def test_run_evaluate_stage_on_mini(mini, data_cfg, rules_cfg, tmp_path):
    paths, ev = mini
    dcfg, rcfg = _cfgs(data_cfg, rules_cfg, B=30)
    rules_dir, model_dirs = _write_stage_inputs(paths, rcfg, tmp_path)
    out = tmp_path / "reports"
    res = run_evaluate_stage(paths, model_dirs, rules_dir, out, dcfg, rcfg, threads=1)
    assert json.loads((out / "results.json").read_text(encoding="utf-8")) == res
    assert (out / "results.md").read_text(encoding="utf-8") == render_markdown(res)
    assert res["meta"]["models"]["lgbm_tx"]["seeds"] == [0, 1]  # sorted by seed number
    assert set(res["views"]["primary"]["rules"]["scenarios"]) == {"fan_in_velocity", "structuring"}
    # the stage aligns inputs on row_id: evaluate() on hand-aligned arrays gives the same numbers
    fl = ev.select("row_id").join(
        pl.read_parquet(rules_dir / "flags.parquet"), on="row_id", maintain_order="left"
    )
    sc = ev.select("row_id").join(
        pl.read_parquet(model_dirs["lgbm_tx"] / "scores.parquet"),
        on="row_id",
        maintain_order="left",
    )
    direct = evaluate(
        ev,
        {rate_tag(r): fl[f"rules_any_{rate_tag(r)}"].to_numpy() for r in _rates(rcfg)},
        {"lgbm_tx": np.stack([sc["score_s0"].to_numpy(), sc["score_s1"].to_numpy()])},
        dcfg,
        rcfg,
    )
    assert direct["views"]["primary"]["models"] == res["views"]["primary"]["models"]


def test_run_evaluate_stage_rejects_missing_scores(mini, data_cfg, rules_cfg, tmp_path):
    paths, _ = mini
    dcfg, rcfg = _cfgs(data_cfg, rules_cfg, B=0)
    rules_dir, model_dirs = _write_stage_inputs(paths, rcfg, tmp_path)
    p = model_dirs["lgbm_tx"] / "scores.parquet"
    write_parquet_atomic(pl.read_parquet(p).filter(pl.col("split") != "test"), p)
    with pytest.raises(ValueError, match="missing"):
        run_evaluate_stage(paths, model_dirs, rules_dir, tmp_path / "out", dcfg, rcfg)


def test_run_evaluate_stage_on_prepared_fixture(prepared, data_cfg, rules_cfg, tmp_path):
    """End to end on the real data layer's output for the synthetic fixture."""
    dcfg, rcfg = _cfgs(data_cfg, rules_cfg, B=30)
    rules_dir, model_dirs = _write_stage_inputs(prepared, rcfg, tmp_path, seed=1)
    out = tmp_path / "reports"
    res = run_evaluate_stage(prepared, model_dirs, rules_dir, out, dcfg, rcfg, threads=2)
    ev = build_eval_frame(prepared, dcfg)
    assert res["meta"]["val_late"]["rows"] == int((ev["split"] == "val_late").sum())
    total = sum(res["meta"]["views"][v]["rows"] for v in ("primary", "tail"))
    assert total == res["meta"]["views"]["full"]["rows"] == int((ev["split"] == "test").sum())
    assert res["meta"]["views"]["full"]["positives"] > 0
    assert (out / "results.md").exists()
    json.dumps(res, allow_nan=False)


# --------------------------------------------------------------------------- honesty details


def test_undefined_seed_metrics_are_marked_not_hidden(mini, data_cfg, rules_cfg):
    """A seed whose val_late scores are all tied gets an (a) threshold of +inf and flags
    nothing. Its precision is undefined: the mean covers the other seed and is marked; F1 is 0,
    so it is not dropped; the markdown warns about the threshold and quotes the other bracket."""
    _, ev = mini
    dcfg, rcfg = _cfgs(data_cfg, rules_cfg, B=40)
    flags, scores = _fake_inputs(ev, rcfg)
    tied = np.full(ev.height, 0.3)
    res = evaluate(ev, flags, {"m": np.stack([tied, scores["single"][0]])}, dcfg, rcfg)
    head = res["meta"]["headline_tag"]
    th = res["thresholds"]["m"]["rate"][head]
    assert th["per_seed"][0] is None  # +inf, written as null
    assert th["model_val_late_rate"][0] == 0.0
    assert th["model_val_late_rate_with_ties"][0] == 1.0  # the whole tie block
    a = res["views"]["primary"]["models"]["m"]["a"][head]
    assert a["alerts"]["per_seed"][0] == 0
    assert a["precision"]["per_seed"][0] is None and a["precision"]["n_defined"] == 1
    assert a["f1"]["per_seed"][0] == 0.0 and a["f1"]["n_defined"] == 2
    assert a["f1"]["mean"] == pytest.approx(np.mean(a["f1"]["per_seed"]))
    boot = res["bootstrap"]["models"]["m"]["a.precision"]
    assert boot["lo"] is not None and boot["undefined"] == 0  # seed mean over defined seeds
    md = render_markdown(res)
    assert "†" in md and "Mean over fewer seeds than the model has" in md
    assert f"m seed 0 at {head}: (a) flags nothing" in md
    assert "flagging the whole tie block would give 100.000%" in md


def test_seed_mean_and_ci_conventions():
    from aml.eval.report import _aggregate, _ci, _pct, _seed_mean

    agg = _aggregate([{"p": float("nan")}, {"p": 0.5}, {"p": 0.7}])["p"]
    assert agg["mean"] == pytest.approx(0.6) and agg["n_defined"] == 2
    assert _pct(agg).endswith("†")
    full = _aggregate([{"p": 0.4}, {"p": 0.6}])["p"]
    assert not _pct(full).endswith("†")
    nan = float("nan")
    got = _seed_mean([np.array([nan, 1.0, nan]), np.array([0.5, 0.0, nan])])
    assert got[0] == 0.5 and got[1] == 0.5 and np.isnan(got[2])
    assert _ci({"k": {"lo": None, "hi": None, "undefined": 7}}, "k") == (
        " [CI n/a: undefined in 7 replicates]"
    )
    assert _ci({"k": {"lo": 0.1, "hi": 0.2, "undefined": 3}}, "k") == (
        " [10.0, 20.0; 3 replicates undefined]"
    )
    assert _ci({"k": {"lo": 0.1, "hi": 0.2, "undefined": 0}}, "k") == " [10.0, 20.0]"


def test_tail_caveat_comes_from_the_data(evaluated):
    res, ev, *_ = evaluated
    tv = res["meta"]["views"]["tail"]
    caveat = next(c for c in res["meta"]["caveats"] if c.startswith("The tail"))
    lo, hi = tv["days"]
    assert f"The tail (days {lo}-{hi}) has only {tv['rows']:,} transactions" in caveat
    assert f"{tv['positives']:,} of them laundering" in caveat
    assert f"({100 * tv['prevalence']:.0f}% of tail rows" in caveat
    assert "holds only completions" not in caveat
    tail = ev.filter((pl.col("split") == "test") & (pl.col("day") >= lo) & (pl.col("y") == 1))
    share = tail["attempt_id"].is_not_null().mean()
    assert f"{100 * share:.0f}% of these positives belong to pattern attempts" in caveat


def test_paired_model_differences(evaluated):
    res, *_ = evaluated
    from aml.eval.report import MODEL_DIFF_KEYS

    md_all = res["bootstrap"]["model_diffs"]
    assert set(md_all) == {"single - lgbm_tx"}
    d = md_all["single - lgbm_tx"]
    assert set(d) == set(MODEL_DIFF_KEYS)
    head = res["meta"]["headline_tag"]
    pv = res["views"]["primary"]["models"]
    want = pv["single"]["a"][head]["recall"]["mean"] - pv["lgbm_tx"]["a"][head]["recall"]["mean"]
    assert d["a.recall"]["point"] == pytest.approx(want)
    want = (
        pv["single"]["literature"]["pr_auc"]["mean"] - pv["lgbm_tx"]["literature"]["pr_auc"]["mean"]
    )
    assert d["literature.pr_auc"]["point"] == pytest.approx(want)
    for iv in d.values():
        assert iv["lo"] <= iv["hi"]
    md = render_markdown(res)
    assert "### Paired model differences (primary)" in md
    assert "| single - lgbm_tx |" in md


def test_identical_models_have_a_zero_paired_difference(mini, data_cfg, rules_cfg):
    _, ev = mini
    dcfg, rcfg = _cfgs(data_cfg, rules_cfg, B=30)
    flags, scores = _fake_inputs(ev, rcfg)
    s = scores["single"][0]
    res = evaluate(ev, flags, {"one": s[None, :], "dup": np.stack([s, s])}, dcfg, rcfg)
    for iv in res["bootstrap"]["model_diffs"]["dup - one"].values():
        assert iv["lo"] == iv["hi"] == 0.0 and iv["point"] == 0.0


def test_sensitivity_table_shows_realised_rates(evaluated):
    res, *_ = evaluated
    md = render_markdown(res)
    sens = md.split("## Sensitivity to the alert budget")[1].split("\n## ")[0]
    for col in ("Rules val_late rate %", "lgbm_tx (a) val_late rate %", "lgbm_tx (a) alerts"):
        assert col in sens


def test_rules_label_and_scenarios_follow_thresholds_json(mini, data_cfg, rules_cfg, tmp_path):
    from aml.io import write_json_atomic
    from aml.rules.sql_baseline import SCENARIOS

    paths, _ = mini
    dcfg, rcfg = _cfgs(data_cfg, rules_cfg, B=0)
    rules_dir, model_dirs = _write_stage_inputs(paths, rcfg, tmp_path)
    tags = [rate_tag(r) for r in _rates(rcfg)]
    head = tags[0]
    thr = {t: {**dict.fromkeys(SCENARIOS), "fan_in_velocity": 5.0} for t in tags}
    thr[tags[-1]]["round_trip"] = 2.0  # two scenarios on at the loosest rate
    doc = {
        "n_scenarios": len(SCENARIOS),
        "thresholds": thr,
        "scenario_diagnostics": {"rates": {head: {"infeasible": ["structuring"]}}},
    }
    write_json_atomic(doc, rules_dir / "thresholds.json")
    res = run_evaluate_stage(paths, model_dirs, rules_dir, tmp_path / "out", dcfg, rcfg)
    assert res["meta"]["rules"]["active"][head] == ["fan_in_velocity"]
    md = render_markdown(res)
    assert "| Rules (SQL, 1 of 7 scenarios active) | own operating point |" in md
    assert "Rules (SQL, 7 scenarios)" not in md
    scen = md.split("## Rule scenarios at the headline budget")[1].split("\n## ")[0]
    assert "| structuring | off (cannot fit the budget) |" in scen
    sens = md.split("## Sensitivity to the alert budget")[1].split("\n## ")[0]
    assert "| 1 of 7 |" in sens and "| 2 of 7 |" in sens


# --------------------------------------------------------------------------- M2 validation extras


def _write_extras(root) -> dict:
    """Real gate / ablation / SHAP documents (from the M2 code) and generic engine / parity ones,
    written as the stages write them (sorted keys)."""
    import lightgbm as lgb

    from aml.features.gate import run_gate
    from aml.features.spec import FeatureDef
    from aml.io import write_json_atomic
    from aml.models.importance import shap_global
    from aml.models.lgbm_graph import ABLATION_VARIANTS, ablation_decision

    rng = np.random.default_rng(0)
    days = np.repeat(np.arange(1, 8), 300)
    split = np.where(days <= 6, "train", "val_early")
    zz = rng.normal(5, 1, days.size)
    table = pl.DataFrame(
        {
            "day": pl.Series(days, dtype=pl.Int16),
            "split": split.tolist(),
            "zz_stable": pl.Series(zz, dtype=pl.Float32),
            "aa_shift": pl.Series(zz + 10 * (days == 7), dtype=pl.Float32),
        }
    )
    feats = [FeatureDef(n, "VEL", "u", None, "exact") for n in ("zz_stable", "aa_shift")]
    cfg = {
        "psi_max": 0.25,
        "psi_bins": 10,
        "psi_eps": 1e-4,
        "psi_train_days": [4, 6],
        "warmup_days": [1, 3],
        "out_of_range_max": 0.01,
        "min_train_nonzero": 100,
        "max_drop_share": 0.75,
    }
    ap = {v: [0.10 + 0.001 * i, 0.102 + 0.001 * i] for i, v in enumerate(ABLATION_VARIANTS)}
    ablation = {**ablation_decision(ap, 2.0), "seeds": [0, 1], "caveat": "a caveat"}
    for v, info in ablation["variants"].items():
        info.update(n_features=10, removed=[f"x{v}"] if v.startswith("-") else [], added=[])
    X = rng.normal(size=(800, 3)).astype(np.float32)
    y = (X[:, 0] + rng.normal(size=800) > 1.5).astype(np.int8)
    names = ["f1", "f2", "f3"]
    booster = lgb.train(
        {"objective": "binary", "verbose": -1, "num_threads": 1},
        lgb.Dataset(X, label=y, feature_name=names),
        num_boost_round=10,
    )
    groups = {"f1": "TX", "f2": "VEL", "f3": "VEL"}
    shap = shap_global(booster, X, y, names, groups, negatives=100, seed=0)
    shap.update(model_seed=0, split="val_early", variant="full")
    docs = {
        "gate": run_gate(table, feats, cfg),
        "ablation": ablation,
        "shap": shap,
        "engine": {"rows": {"train": 10}, "us_per_event": 12.5, "restart_check": {"ok": True}},
        "parity": {"mismatches": {"fan_in_velocity": {"val_early": 0}}, "rows": 1000},
    }
    return {kind: write_json_atomic(doc, root / f"{kind}.json") for kind, doc in docs.items()}


def test_validation_sections_are_additive(mini, data_cfg, rules_cfg, tmp_path):
    from aml.models.lgbm_graph import ABLATION_VARIANTS

    paths, _ = mini
    dcfg, rcfg = _cfgs(data_cfg, rules_cfg, B=0)
    rules_dir, model_dirs = _write_stage_inputs(paths, rcfg, tmp_path)
    plain = run_evaluate_stage(paths, model_dirs, rules_dir, tmp_path / "a", dcfg, rcfg)
    extras = _write_extras(tmp_path / "extras")
    res = run_evaluate_stage(
        paths, model_dirs, rules_dir, tmp_path / "b", dcfg, rcfg, extras=extras
    )
    assert "validation" not in plain and set(res["validation"]) == set(extras)
    md_plain = (tmp_path / "a" / "results.md").read_text(encoding="utf-8")
    md = (tmp_path / "b" / "results.md").read_text(encoding="utf-8")
    assert "Validation-only" not in md_plain
    assert md == render_markdown(res)
    # The validation block does not depend on key order: results.json (sorted keys) gives the
    # same block. (M1 sections follow the in-memory order and are not checked here.)
    from aml.eval.report import _validation_section

    reloaded = json.loads((tmp_path / "b" / "results.json").read_text(encoding="utf-8"))
    assert _validation_section(reloaded) == _validation_section(res)
    # Purely additive: removing the block gives exactly the plain report.
    start = md.index("## Validation-only M2 evidence")
    end = md.index("## Literature reference")
    assert md[:start] + md[end:] == md_plain
    assert md.index("## Memorisation check at (a)") < start
    block = md[start:end]
    for needle in (
        "### Feature gate (label-free, before any fit)",
        "| zz_stable | VEL |",
        "| aa_shift | VEL |",
        "dropped (psi, oor)",
        "### Group ablation on val_early (tuned parameters fixed)",
        "Decision: the RULE group is dropped; the champion is -RULE.",  # delta 0.7 pp > bar
        "| -RULE | 10 | 10.80 ± 0.14 | +0.70 | champion |",
        "Note: a caveat.",
        "### TreeSHAP of the champion (seed 0, val_early)",
        "Top 3 features:",
        "### Feature engine (build_features summary.json)",
        "| us_per_event | 12.5 |",
        "| restart_check.ok | True |",
        "### Rule parity",
        "Rule parity (engine severities vs the M1 SQL): 0 mismatches over 1,000 rows.",
    ):
        assert needle in block, needle
    # Spec order, variant order and the SHAP group ranking survive the sorted-key JSON files.
    assert block.index("| zz_stable |") < block.index("| aa_shift |")
    first_cells = [ln.split("|")[1].strip() for ln in block.splitlines() if ln.startswith("| ")]
    assert [c for c in first_cells if c in ABLATION_VARIANTS] == list(ABLATION_VARIANTS)
    ranking = read_json(extras["shap"])["group_ranking"]
    assert [c for c in first_cells if c in ("TX", "VEL")][:2] == ranking


def test_ablation_text_marks_a_group_the_gate_removed():
    """A -G variant with the same inputs as full (the gate removed the whole group) shares
    full's fits: its Δ of 0 is no measurement, so it is labelled and never the 'largest Δ'."""
    from aml.eval.report import _ablation_lines, _compact_ablation, _validation_section
    from aml.models.lgbm_graph import ABLATION_VARIANTS, ablation_decision

    ap = {v: [0.090, 0.092] for v in ABLATION_VARIANTS}  # every group variant: -1 pp
    ap["full"] = ap["-CYC"] = ap["no_gate"] = ap["nofmt"] = [0.100, 0.102]
    ap["-AMT"] = [0.099, 0.101]  # -0.1 pp: the largest measured Δ
    doc = ablation_decision(ap, 2.0)
    for v, info in doc["variants"].items():
        info.update(n_features=10, removed=[] if v in ("full", "-CYC") else [f"x{v}"], added=[])
    doc.update(seeds=[0, 1], caveat=None)
    assert doc["champion"] == "full" and doc["best_group_variant"] == "-CYC"
    text = "\n".join(_ablation_lines(_compact_ablation(doc)))
    assert "| -CYC | 10 | 10.10 ± 0.14 | +0.00 | = full (group removed by the gate) |" in text
    assert "| -AMT | 10 | 10.00 ± 0.14 | -0.10 | not chosen |" in text
    assert "Decision: the champion is full (largest Δ: -AMT, -0.10 pp, under the" in text
    for info in doc["variants"].values():
        info["removed"] = []  # every group removed by the gate: no measured Δ to quote
    text = "\n".join(_ablation_lines(_compact_ablation(doc)))
    assert "Decision: the champion is full." in text
    intro = "\n".join(_validation_section({"validation": {"engine": {"items": []}}}))
    assert "no number below uses a test label" in intro and "test row" not in intro


def test_extras_are_checked(mini, data_cfg, rules_cfg, tmp_path):
    paths, _ = mini
    dcfg, rcfg = _cfgs(data_cfg, rules_cfg, B=0)
    rules_dir, model_dirs = _write_stage_inputs(paths, rcfg, tmp_path)
    out = tmp_path / "o"
    with pytest.raises(ValueError, match="unknown extras"):
        run_evaluate_stage(paths, model_dirs, rules_dir, out, dcfg, rcfg, extras={"x": tmp_path})
    missing = {"parity": tmp_path / "missing.json"}
    with pytest.raises(FileNotFoundError):
        run_evaluate_stage(paths, model_dirs, rules_dir, out, dcfg, rcfg, extras=missing)


def test_parity_line():
    from aml.eval.report import parity_line

    doc = {
        "mismatches": {"structuring": {"train": 2, "test": 1}, "round_trip": {"train": 0}},
        "rows": {"train": 10, "test": 5},
        "rule_trunc_rows": 0,
    }
    assert parity_line(doc) == (
        "Rule parity (engine severities vs the M1 SQL): 3 mismatches over 15 rows "
        "(0 rows with rule_trunc = 1 are excluded)."
    )
    assert parity_line({"mismatches_total": 0}).endswith(": 0 mismatches.")
    assert "see parity.json" in parity_line({"ok": True, "mismatches": ["a"]})
    # The rules engine stage's parity.json: rule_trunc is a block with a row count and a share.
    doc = {"mismatches_total": 0, "rows": 7, "rule_trunc": {"rows": 2, "share": 2 / 7}}
    assert parity_line(doc).endswith(
        ": 0 mismatches over 7 rows (2 rows with rule_trunc = 1 are excluded)."
    )


def test_evaluate_job_reads_rules_engine_and_graph_models(tmp_path, monkeypatch):
    import importlib
    import os
    import tempfile
    from pathlib import Path

    os.environ.setdefault(
        "MODAL_CONFIG_PATH", str(Path(tempfile.gettempdir()) / "aml-tests-no-modal.toml")
    )
    job = importlib.import_module("modal_jobs.evaluate")
    common = importlib.import_module("modal_jobs.common")
    paths = DataPaths(tmp_path)
    keys = {
        "rules_engine": "rules_engine-1",
        "lgbm_tx": "lgbm_tx-2",
        "lgbm_graph": "lgbm_graph-3",
        "features": "features-4",
    }
    rules_dir, model_dirs, by_stage = job.stage_dirs(paths, keys, with_nofmt=False)
    assert rules_dir == paths.model_dir("rules_engine", "rules_engine-1")
    assert model_dirs == {
        "lgbm_tx": paths.model_dir("lgbm_tx", "lgbm_tx-2"),
        "lgbm_graph": paths.model_dir("lgbm_graph", "lgbm_graph-3"),
    }
    _, with_nofmt, _ = job.stage_dirs(paths, keys, with_nofmt=True)
    assert list(with_nofmt) == ["lgbm_tx", "lgbm_graph", "lgbm_graph_nofmt"]
    assert with_nofmt["lgbm_graph_nofmt"] == paths.model_dir("lgbm_graph_nofmt", "lgbm_graph-3")
    assert job.extra_paths(by_stage) == {}
    for stage, name in (("lgbm_graph", "gate.json"), ("features", "summary.json")):
        by_stage[stage].mkdir(parents=True)
        (by_stage[stage] / name).write_text("{}", encoding="utf-8")
    assert job.extra_paths(by_stage) == {
        "gate": by_stage["lgbm_graph"] / "gate.json",
        "engine": by_stage["features"] / "summary.json",
    }

    sent: list[tuple] = []

    class _Remote:
        def remote(self, *args):
            sent.append(args)
            return {"models": [], "extras": [], "reports": []}

    monkeypatch.setattr(job, "evaluate", _Remote())
    job.main.info.raw_f(with_nofmt=True, skip_cost=True)
    _, _, keys_sent, flag = sent[0]
    cfgs = common.load_all_configs()
    base = common.all_keys(cfgs)
    assert flag is True
    assert keys_sent["eval"] == common.eval_key(cfgs, with_nofmt=True) != base["eval"]
    assert {k: v for k, v in keys_sent.items() if k != "eval"} == {
        k: v for k, v in base.items() if k != "eval"
    }
