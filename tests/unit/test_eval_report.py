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


# --------------------------------------------------------------------------- M3: GNN models

GNN_COMPARISON = ("gnn_causal", "gnn_lookahead", "gnn_lookahead_d10")
M2_REPORT_ORDER = ("lgbm_tx", "lgbm_graph")  # modal_jobs.evaluate.stage_dirs


def _gnn_report_cfg() -> dict:
    from tests.conftest import load_yaml

    return copy.deepcopy(load_yaml("gnn.yaml")["report"])


def _labelled(paths: DataPaths) -> pl.DataFrame:
    tx = pl.read_parquet(paths.transactions).select("row_id", "split", "day")
    lab = pl.read_parquet(paths.labels).select("row_id", "is_laundering")
    return tx.join(lab, on="row_id", how="left").sort("row_id")


def _write_scores(paths, root, name, seeds, *, seed, signal=0.35, test_shift=0.0):
    """scores.parquet (row_id, split, score_s<k>) for every row: informative, in [0, 1]."""
    j = _labelled(paths)
    y = j["is_laundering"].fill_null(0).to_numpy()
    test = (j["split"] == "test").to_numpy()
    rng = np.random.default_rng(seed)
    cols = {}
    for k in seeds:
        s = rng.random(len(y)) * 0.6 + y * signal + rng.normal(0, 0.05, len(y)) + test * test_shift
        cols[f"score_s{k}"] = pl.Series(np.clip(s, 0.0, 1.0))
    d = root / name
    write_parquet_atomic(j.select("row_id", "split").with_columns(**cols), d / "scores.parquet")
    return d


def _guard(edges: int, dropped: int = 0) -> dict:
    return {
        "edges_checked": edges,
        "violations": 0,
        "target_hits": 0,
        "max_slack": -1,
        "future_edges": dropped,
        "dropped_target_copies": dropped,
    }


def _gnn_summary(protocol: str, seeds: list[int], report_cfg: dict, **extra) -> dict:
    from aml.models.gnn import report_hash

    doc = {
        "protocol": protocol,
        "seeds": seeds,
        "final": True,
        "report_hash": report_hash({"report": report_cfg}),
        "best_val_ap_mean": 0.41,
        "best_val_ap_std": 0.02,
        "guard": {"val_early": _guard(1000), "val_late": _guard(2000), "test": _guard(3000)},
    }
    doc.update(extra)
    return doc


@pytest.fixture
def gnn_stage(mini, data_cfg, rules_cfg, tmp_path):
    """Stage inputs for lgbm_tx, lgbm_graph, the three GNN comparison models and gnn_faithful."""
    paths, _ = mini
    dcfg, rcfg = _cfgs(data_cfg, rules_cfg, B=40)
    rules_dir, model_dirs = _write_stage_inputs(paths, rcfg, tmp_path)
    model_dirs["lgbm_graph"] = _write_scores(paths, tmp_path, "lgbm_graph", [0, 1], seed=11)
    report_cfg = _gnn_report_cfg()
    gnn_dirs = {
        "gnn_causal": _write_scores(paths, tmp_path, "gnn_causal", [0, 1, 2], seed=12),
        "gnn_lookahead": _write_scores(paths, tmp_path, "gnn_lookahead", [0], seed=13),
        "gnn_lookahead_d10": _write_scores(
            paths, tmp_path, "gnn_lookahead_d10", [0], seed=13, test_shift=0.05
        ),
    }
    la_guard = {"val_early": _guard(500, dropped=40), "test": _guard(900, dropped=60)}
    l4 = {"gpu": "NVIDIA L4", "cores": 8, "memory_mib": 32768}
    summaries = {
        "gnn_causal": _gnn_summary(
            "causal",
            [0, 1, 2],
            report_cfg,
            per_seed={str(k): {"best_val_ap": v} for k, v in enumerate((0.39, 0.41, 0.43))},
            best_val_ap_std=0.0163,  # the training summary's ddof 0 (not what results.md shows)
            best_val_ap_mean_fresh=0.42,
            gpu_seconds=3600.0,
            **l4,
        ),
        "gnn_lookahead": _gnn_summary(
            "lookahead",
            [0],
            report_cfg,
            guard=la_guard,
            per_seed={"0": {"best_val_ap": 0.47}},
            best_val_ap_mean=0.47,
            best_val_ap_std=0.0,  # ddof 0 of one seed: no std to show
            future_share={"train": 0.25, "val_early": 0.4, "test": 0.5},
        ),
        "gnn_lookahead_d10": _gnn_summary(
            "lookahead",
            [0],
            report_cfg,
            guard=la_guard,
            future_share={"train": 0.25, "val_early": 0.4, "test_d10": 0.3},
            gpu_seconds=1800.0,
            **l4,
        ),
    }
    fa_dir = _write_scores(paths, tmp_path, "gnn_faithful", [0], seed=14, signal=0.5)
    fa = pl.read_parquet(fa_dir / "scores.parquet")
    sampled = np.random.default_rng(15).random(fa.height) < 0.9
    write_parquet_atomic(fa.with_columns(sampled=pl.Series(sampled)), fa_dir / "scores.parquet")
    fa_summary = _gnn_summary(
        "faithful",
        [0],
        report_cfg,
        epochs_run=100,
        max_epochs=100,
        epoch_cap=None,
        best_epoch=97,
        batch_size=8192,
        sampled_share={"val_early": 0.95, "val_late": 0.96},
        guard={"test": _guard(7000)},
        gpu="NVIDIA L4",
        gpu_seconds=7200.0,
        cores=8,
        memory_mib=24576,
    )
    gnn = {
        "model_views": {"gnn_lookahead_d10": ["primary"]},
        "summaries": summaries,
        "faithful": {"scores": fa_dir / "scores.parquet", "summary": fa_summary},
        "report_cfg": report_cfg,
    }
    return {
        "paths": paths,
        "dcfg": dcfg,
        "rcfg": rcfg,
        "rules_dir": rules_dir,
        "m2_dirs": {n: model_dirs[n] for n in M2_REPORT_ORDER},
        "all_dirs": {**{n: model_dirs[n] for n in M2_REPORT_ORDER}, **gnn_dirs},
        "gnn": gnn,
        "out": tmp_path,
    }


def _stage(s: dict, out: str, gnn: dict | None = None, dirs: dict | None = None) -> dict:
    return run_evaluate_stage(
        s["paths"],
        dirs or s["all_dirs"],
        s["rules_dir"],
        s["out"] / out,
        s["dcfg"],
        s["rcfg"],
        gnn=gnn,
    )


def test_f1_argmax_is_bootstrapped_and_in_model_diffs(evaluated):
    from aml.eval.report import BOOT_MODEL_KEYS, M2_MODEL_DIFF_KEYS, MODEL_DIFF_KEYS

    res, *_ = evaluated
    assert BOOT_MODEL_KEYS[-1] == MODEL_DIFF_KEYS[-1] == "literature.f1_argmax"
    assert MODEL_DIFF_KEYS[:-1] == M2_MODEL_DIFF_KEYS  # existing keys unchanged, in order
    boot = res["bootstrap"]
    for name in ("lgbm_tx", "single"):
        iv = boot["models"][name]["literature.f1_argmax"]
        assert iv["lo"] <= iv["hi"]
    d = boot["model_diffs"]["single - lgbm_tx"]["literature.f1_argmax"]
    pv = res["views"]["primary"]["models"]
    want = (
        pv["single"]["literature"]["f1_argmax"]["mean"]
        - (pv["lgbm_tx"]["literature"]["f1_argmax"]["mean"])
    )
    assert d["point"] == pytest.approx(want) and d["lo"] <= d["hi"]


def test_f1_argmax_bootstrap_matches_the_weighted_metric(mini, data_cfg, rules_cfg):
    """A replicate's literature.f1_argmax equals metrics.prf_at_threshold(0.5) under that
    replicate's row weights (B = 1, so the CI is the replicate itself)."""
    from aml.eval.bootstrap import make_clusters, replicate_weights
    from aml.eval.metrics import ARGMAX_THRESHOLD, prf_at_threshold
    from aml.eval.typology import attempt_ids

    _, ev = mini
    dcfg, rcfg = _cfgs(data_cfg, rules_cfg, B=1)
    flags, scores = _fake_inputs(ev, rcfg)
    res = evaluate(ev, flags, {"one": scores["single"]}, dcfg, rcfg)
    test = (ev["split"] == "test").to_numpy()
    lo, hi = dcfg["test_views"]["primary"]
    day = ev["day"].to_numpy()
    mask = test & (day >= lo) & (day <= hi)
    y = ev["y"].to_numpy()[mask]
    strata, clusters = make_clusters(
        y, attempt_ids(ev["attempt_id"])[mask], ev["src"].to_numpy()[mask]
    )
    (w,) = next(replicate_weights(strata, clusters, 1, dcfg["evaluation"]["bootstrap_seed"]))
    want = prf_at_threshold(y, scores["single"][0][mask], ARGMAX_THRESHOLD, w)["f1"]
    iv = res["bootstrap"]["models"]["one"]["literature.f1_argmax"]
    assert iv["lo"] == pytest.approx(want) and iv["hi"] == pytest.approx(want)


def _restore_report_order(res: dict, rules_cfg: dict) -> dict:
    """results.json is written with sorted keys; restore the insertion order results.md was
    rendered in (models in report order, rates headline first, scenarios in SCENARIOS order)."""
    from aml.rules.sql_baseline import SCENARIOS

    def order(d: dict, keys) -> dict:
        first = {k: d[k] for k in keys if k in d}
        return {**first, **{k: v for k, v in d.items() if k not in first}}

    tags = [rate_tag(r) for r in _rates(rules_cfg)]
    res["meta"]["models"] = order(res["meta"]["models"], M2_REPORT_ORDER)
    res["meta"]["rates"] = order(res["meta"]["rates"], tags)
    res["thresholds"] = order(res["thresholds"], M2_REPORT_ORDER)
    for th in res["thresholds"].values():
        th["rate"] = order(th["rate"], tags)
    for v in res["views"].values():
        v["models"] = order(v["models"], M2_REPORT_ORDER)
        if "scenarios" in v["rules"]:
            v["rules"]["scenarios"] = order(v["rules"]["scenarios"], SCENARIOS)
    for k in ("models", "diffs"):
        res["bootstrap"][k] = order(res["bootstrap"][k], M2_REPORT_ORDER)
    return res


# SHA-256 of the M2 reports/results.md (real data) that tests/golden/m2_results.json rendered
# to. Only README.md is published, so the Markdown itself is pinned by its hash.
M2_RESULTS_MD_SHA256 = "c040c21dc8cad96bb6d8a0c9d6fc179322a65e281fd5310b67fc7473e11c6372"


def test_committed_m2_results_render_byte_identical():
    """Regression against the M2 outputs (real data): the current renderer turns the M2
    results.json (frozen in tests/golden/; no GNN models, no literature.f1_argmax yet) into
    exactly the M2 results.md, and the pinned headline numbers are the README's."""
    import hashlib

    from tests.conftest import REPO_ROOT, load_yaml

    res = json.loads((REPO_ROOT / "tests" / "golden" / "m2_results.json").read_text("utf-8"))
    assert "gnn" not in res and "model_views" not in res["meta"]
    assert list(res["meta"]["models"]) == sorted(M2_REPORT_ORDER)  # sorted keys on disk
    res = _restore_report_order(res, load_yaml("rules.yaml"))
    md = render_markdown(res).encode("utf-8")
    assert hashlib.sha256(md).hexdigest() == M2_RESULTS_MD_SHA256
    head = res["meta"]["headline_tag"]
    a = res["views"]["primary"]["models"]["lgbm_graph"]["a"][head]["recall"]
    ci = res["bootstrap"]["models"]["lgbm_graph"]["a.recall"]
    assert (round(100 * a["mean"], 1), round(100 * a["std"], 1)) == (68.4, 0.6)
    assert (round(100 * ci["lo"], 1), round(100 * ci["hi"], 1)) == (60.0, 75.7)


def test_m2_values_and_cis_identical_with_and_without_gnn_models(mini, data_cfg, rules_cfg):
    _, ev = mini
    dcfg, rcfg = _cfgs(data_cfg, rules_cfg, B=50)
    flags, scores = _fake_inputs(ev, rcfg)
    rng = np.random.default_rng(7)
    m2 = {"lgbm_tx": scores["lgbm_tx"], "lgbm_graph": np.round(rng.random((2, ev.height)), 3)}
    gnn = {n: np.round(rng.random((1 + (n == "gnn_causal"), ev.height)), 3) for n in GNN_COMPARISON}
    plain = evaluate(ev, flags, m2, dcfg, rcfg, threads=2)
    both = evaluate(ev, flags, {**m2, **gnn}, dcfg, rcfg, threads=3)
    for name in m2:
        assert both["thresholds"][name] == plain["thresholds"][name]
        assert both["bootstrap"]["models"][name] == plain["bootstrap"]["models"][name]
        assert both["bootstrap"]["diffs"][name] == plain["bootstrap"]["diffs"][name]
    for v in ("primary", "tail", "full"):
        assert both["views"][v]["rules"] == plain["views"][v]["rules"]
        for name in m2:
            assert both["views"][v]["models"][name] == plain["views"][v]["models"][name]
    for k in ("rules", "strata"):
        assert both["bootstrap"][k] == plain["bootstrap"][k]
    pair = "lgbm_graph - lgbm_tx"
    assert both["bootstrap"]["model_diffs"][pair] == plain["bootstrap"]["model_diffs"][pair]


def test_without_gnn_the_report_keeps_the_m2_layout(gnn_stage):
    s = gnn_stage
    res = _stage(s, "m2", dirs=s["m2_dirs"])
    assert "gnn" not in res and "model_views" not in res["meta"]
    md = (s["out"] / "m2" / "results.md").read_text(encoding="utf-8")
    assert md == render_markdown(res)
    for needle in ("F1 @ 0.5 (argmax), pp", "Which model wins", "Look-ahead gap", "guard"):
        assert needle not in md, needle
    table = md.split("### Paired model differences (primary)")[1].split("\n\n")[1]
    assert table.splitlines()[0] == (
        "| Models | Recall (a), pp | Precision (a), pp | Recall (b), pp | "
        "F1 @ val_late-best threshold, pp | PR-AUC, pp |"
    )
    # The faithful run never enters the comparison, with or without the gnn argument.
    dirs = {**s["m2_dirs"], "gnn_faithful": s["m2_dirs"]["lgbm_tx"]}
    with pytest.raises(ValueError, match="never enters the model comparison"):
        _stage(s, "x", dirs=dirs)
    with pytest.raises(ValueError, match="never enters the model comparison"):
        _stage(s, "y", gnn=s["gnn"], dirs={**s["all_dirs"], "gnn_faithful": s["out"]})


def _section(md: str, title: str) -> str:
    return md.split(title)[1].split("\n## ")[0]


def test_gnn_report_sections_and_primary_only_rendering(gnn_stage):
    s = gnn_stage
    res = _stage(s, "gnn", gnn=s["gnn"])
    out = s["out"] / "gnn"
    assert json.loads((out / "results.json").read_text(encoding="utf-8")) == res
    md = (out / "results.md").read_text(encoding="utf-8")
    assert md == render_markdown(res)
    assert list(res["meta"]["models"]) == [*M2_REPORT_ORDER, *GNN_COMPARISON]
    assert res["meta"]["model_views"]["gnn_lookahead_d10"] == ["primary"]
    assert res["meta"]["model_views"]["gnn_causal"] == ["primary", "tail", "full"]
    # d10 is evaluated on every view (finite-score invariant) but rendered on the primary only.
    assert "gnn_lookahead_d10" in res["views"]["tail"]["models"]
    d10 = "| gnn_lookahead_d10 (1 seed, no std) |"
    assert d10 in _section(md, "## Headline: primary period")
    assert "| gnn_lookahead_d10 - gnn_lookahead |" in md
    lit = _section(md, "## Literature-comparable metrics")
    primary_lit, full_lit = lit.split("**full**")
    assert d10 in primary_lit and d10 not in full_lit
    assert "| gnn_causal (3 seeds) |" in full_lit
    for title in ("## Test view: tail", "## Test view: full"):
        assert "gnn_lookahead_d10" not in _section(md, title), title
        assert "gnn_lookahead (1 seed, no std)" in _section(md, title), title
    for title in (
        "## Recall per typology at (a)",
        "## Attempt-level detection at (a)",
        "## Memorisation check at (a)",
    ):
        blocks = _section(md, title).split("**")
        prim = next(b for i, b in enumerate(blocks) if blocks[i - 1].startswith("primary"))
        assert "gnn_lookahead_d10" in prim, title
        for v in ("tail", "full"):
            other = next(b for i, b in enumerate(blocks) if blocks[i - 1].startswith(v))
            assert "gnn_lookahead_d10" not in other, (title, v)
            assert "gnn_lookahead " in other or "gnn_lookahead detected" in other, (title, v)
    # GNN layout: F1 @ 0.5 in the model differences.
    assert "F1 @ 0.5 (argmax), pp" in md
    for title in (
        "## Which model wins (pre-registered rule)",
        "## Look-ahead gap, step 1 (primary period, days 9-10)",
        "## Faithful Multi-GNN reproduction (not in the model comparison)",
        "## As-of guard evidence (GNN)",
    ):
        assert title in md, title
    assert md.index("## Memorisation check at (a)") < md.index("## Which model wins")
    assert md.index("## As-of guard evidence (GNN)") < md.index("## Literature reference")
    guard = _section(md, "## As-of guard evidence (GNN)")
    assert "- gnn_causal: 0 violations over 6,000 sampled edges; 0 target hits." in guard
    assert (
        "- gnn_lookahead: 0 violations over 1,400 sampled edges; 0 target hits (100 sampled "
        "copies of the target dropped)." in guard
    )
    assert "- gnn_faithful (snapshot guard): 0 violations over 7,000 sampled edges" in guard
    gap = _section(md, "## Look-ahead gap, step 1")
    assert "One look-ahead seed: the CI covers test sampling only." in gap
    # review EVAL-4: the d10 test pass's future share and the `last`-vs-uniform caveat
    assert "train 25.0%, val_early 40.0%, test 50.0%, test_d10 30.0%." in gap
    assert "whereas the published loader samples uniformly" in gap
    # review EVAL-1: sample std (ddof 1) over the per-seed values, none for one seed, and the
    # causal mean without the HPO seed's run
    assert (
        "causal 41.0 ± 2.0 (seeds other than the HPO model seed, whose run repeats the "
        "selected trial: 42.0), look-ahead 47.0 (1 seed, no std)." in gap
    )
    va = res["gnn"]["lookahead_gap"]["val_early_pr_auc"]
    assert va["causal"]["std"] == pytest.approx(0.02) and va["lookahead"]["std"] is None
    # review EVAL-7: a training-$ estimate per set (gpu_seconds x the exact shape price)
    cost = _section(md, "## GNN training cost (estimate)")
    from aml.models.gnn import costplan

    l4_h = costplan.shape_usd_h("L4", 8, 32)
    assert f"- gnn_causal: ≈ ${l4_h:.2f} (1.00 h on NVIDIA L4, 8 cores, 32,768 MiB" in cost
    assert "- gnn_lookahead: n/a" in cost  # no GPU wall in its summary
    assert "- gnn_lookahead_d10: trained with gnn_lookahead (no cost of its own)." in cost
    f_usd = 2 * costplan.shape_usd_h("L4", 8, 24)
    assert f"- gnn_faithful: ≈ ${f_usd:.2f} (2.00 h on NVIDIA L4, 8 cores, 24,576 MiB" in cost
    tc = res["gnn"]["training_cost"]
    assert tc["gnn_causal"]["est_usd"] == pytest.approx(l4_h)
    assert res["gnn"]["summaries"]["gnn_causal"]["cores"] == 8
    faithful = _section(md, "## Faithful Multi-GNN reproduction")
    assert "the run may not have converged" in faithful  # best epoch 97 of 100
    assert res["gnn"]["winner"]["pair"] == ["gnn_causal", "lgbm_graph"]
    assert f"**Verdict: {res['gnn']['winner']['sentence']}.**" in md
    json.dumps(res, allow_nan=False)


def test_gap_numbers_equal_the_paired_model_differences(gnn_stage):
    s = gnn_stage
    res = _stage(s, "gap", gnn=s["gnn"])
    gap = res["gnn"]["lookahead_gap"]
    md_all = res["bootstrap"]["model_diffs"]
    assert list(gap["metrics"]) == s["gnn"]["report_cfg"]["gap_metrics"]
    pv = res["views"]["primary"]["models"]
    for key, row in gap["metrics"].items():
        for col, pair in (
            ("gap_end", "gnn_lookahead - gnn_causal"),
            ("gap_d10", "gnn_lookahead_d10 - gnn_causal"),
        ):
            iv = md_all[pair][key]
            got = {k: row[col][k] for k in ("point", "lo", "hi")}
            assert got == {k: iv[k] for k in ("point", "lo", "hi")}, (key, col)
        # tail = end - d10 = -(d10 - end): point negated, CI bounds negated and swapped
        iv = md_all["gnn_lookahead_d10 - gnn_lookahead"][key]
        assert row["tail"]["point"] == pytest.approx(-iv["point"])
        assert (row["tail"]["lo"], row["tail"]["hi"]) == (-iv["hi"], -iv["lo"])
        for col in ("gap_end", "gap_d10", "tail"):
            r = row[col]
            assert r["significant"] == (r["lo"] > 0 or r["hi"] < 0)
        part, metric = key.split(".", 1)
        assert row["causal"]["mean"] == pv["gnn_causal"][part][metric]["mean"]
    # Without the d10 model there is no gap section.
    no_d10 = copy.deepcopy(s["gnn"])
    no_d10["model_views"] = {}
    del no_d10["summaries"]["gnn_lookahead_d10"]
    dirs = {k: v for k, v in s["all_dirs"].items() if k != "gnn_lookahead_d10"}
    res = _stage(s, "nod10", gnn=no_d10, dirs=dirs)
    assert res["gnn"]["lookahead_gap"] is None
    assert "Look-ahead gap" not in render_markdown(res)


def test_faithful_section_band_verdict_and_convergence():
    from aml.eval.report import faithful_section

    report_cfg = _gnn_report_cfg()
    # 40 test rows on day 9 and 10 on day 12. Sampled day-9 targets: TP 10, FP 5, FN 6 -> F1
    # 20/31 = 64.5% (in the band); 4 unsampled positives on day 12 are scored low.
    y = [1] * 16 + [0] * 24 + [1] * 4 + [0] * 6
    s = [0.9] * 10 + [0.1] * 6 + [0.8] * 5 + [0.2] * 19 + [0.3] * 4 + [0.1] * 6
    sampled = [True] * 40 + [False] * 4 + [True] * 6
    day = [9] * 40 + [12] * 10
    n = len(y)
    ev = pl.DataFrame(
        {
            "row_id": [*range(n), 1000],
            "split": ["test"] * n + ["val_late"],
            "day": [*day, 8],
            "y": [*y, 1],
        }
    )
    scores = pl.DataFrame(
        {"row_id": list(range(n)), "split": ["test"] * n, "score_s0": s, "sampled": sampled}
    )
    summary = {"epochs_run": 100, "max_epochs": 100, "best_epoch": 60, "batch_size": 8192}
    views = {"primary": (9, 10), "full": (9, 18)}
    out = faithful_section(ev, scores, summary, report_cfg, views=views)
    assert out["f1_sampled_pct"] == pytest.approx(100 * 20 / 31)
    assert out["verdict"] == "reproduced" and out["miss_pp"] == 0.0
    full = out["views"]["full"]
    assert full["sampled"]["rows"] == 46 and full["all"]["rows"] == 50
    assert full["all"]["f1"] == pytest.approx(20 / (20 + 5 + 10))  # + 4 unsampled misses
    assert full["sampled_share"] == pytest.approx(46 / 50)
    assert out["views"]["primary"]["sampled"]["f1"] == pytest.approx(20 / 31)
    assert out["may_not_have_converged"] is False
    out = faithful_section(ev, scores, {**summary, "best_epoch": 95}, report_cfg, views=views)
    assert out["may_not_have_converged"] is True
    # Below the band: two more false positives among the sampled targets.
    s2 = list(s)
    s2[21:23] = [0.95, 0.95]  # two sampled negatives scored above 0.5
    worse = scores.with_columns(score_s0=pl.Series(s2))
    out = faithful_section(ev, worse, summary, report_cfg, views=views)
    assert out["verdict"] == "not reproduced"
    assert out["miss_pp"] == pytest.approx(100 * 20 / 33 - report_cfg["reproduced_band"][0])
    with pytest.raises(ValueError, match="missing values"):
        faithful_section(ev, scores.head(10), summary, report_cfg)
    with pytest.raises(ValueError, match="sampled"):
        faithful_section(ev, scores.drop("sampled"), summary, report_cfg)


def _verdict_results(primary: tuple, secondary: tuple, reverse: bool = False) -> dict:
    def iv(lo_hi: tuple) -> dict:
        lo, hi = lo_hi
        return {"point": (lo + hi) / 2, "lo": lo, "hi": hi, "undefined": 0}

    diffs = {"a.recall": iv(primary), "literature.pr_auc": iv(secondary)}
    if reverse:  # the bootstrap holds lgbm_graph - gnn_causal instead
        diffs = {
            k: {"point": -d["point"], "lo": -d["hi"], "hi": -d["lo"]} for k, d in diffs.items()
        }
    key = "lgbm_graph - gnn_causal" if reverse else "gnn_causal - lgbm_graph"
    return {
        "meta": {"models": {"lgbm_graph": {}, "gnn_causal": {}}, "headline_tag": "0p005"},
        "bootstrap": {"level": 0.95, "model_diffs": {key: diffs}},
    }


WIN, LOSE, TIE = (0.01, 0.05), (-0.05, -0.01), (-0.02, 0.03)
_ID = {WIN: "gnn", LOSE: "lgbm", TIE: "tie"}
TIE_SENTENCE = (
    "no significant difference; LightGBM-graph stays the served champion (cheaper to train, "
    "µs serving)"
)
VERDICT_TABLE = [
    (WIN, WIN, "gnn", "the causal GNN wins"),
    (WIN, TIE, "gnn", "the causal GNN wins"),
    (TIE, WIN, "gnn", "the causal GNN wins"),
    (LOSE, LOSE, "lgbm", "LightGBM-graph wins"),
    (LOSE, TIE, "lgbm", "LightGBM-graph wins"),
    (TIE, LOSE, "lgbm", "LightGBM-graph wins"),
    (TIE, TIE, "tie", TIE_SENTENCE),
    (
        WIN,
        LOSE,
        "mixed",
        "mixed: the causal GNN is better on recall (a); LightGBM-graph is better on PR-AUC",
    ),
    (
        LOSE,
        WIN,
        "mixed",
        "mixed: LightGBM-graph is better on recall (a); the causal GNN is better on PR-AUC",
    ),
]


@pytest.mark.parametrize(
    "primary, secondary, verdict, sentence",
    VERDICT_TABLE,
    ids=[f"{_ID[p]}-{_ID[s]}" for p, s, *_ in VERDICT_TABLE],
)
@pytest.mark.parametrize("reverse", [False, True])
def test_winner_verdict_all_nine_combinations(primary, secondary, verdict, sentence, reverse):
    from aml.eval.report import winner_verdict

    w = winner_verdict(_verdict_results(primary, secondary, reverse), _gnn_report_cfg())
    assert w["verdict"] == verdict and w["sentence"] == sentence
    want = {WIN: "gnn", LOSE: "lgbm", TIE: "tie"}
    assert w["metrics"]["a.recall"]["outcome"] == want[primary]
    assert w["metrics"]["literature.pr_auc"]["outcome"] == want[secondary]
    assert w["metrics"]["a.recall"]["lo"] == pytest.approx(primary[0])
    assert w["level_matches"] is True


def test_winner_verdict_edge_cases():
    from aml.eval.report import verdict_of, winner_verdict

    assert len({(p, s) for p, s, *_ in VERDICT_TABLE}) == 9
    res = _verdict_results(WIN, WIN)
    md = res["bootstrap"]["model_diffs"]["gnn_causal - lgbm_graph"]
    md["a.recall"].update(lo=None, hi=None)  # an undefined CI is no evidence: a tie
    w = winner_verdict(res, _gnn_report_cfg())
    assert w["metrics"]["a.recall"]["outcome"] == "tie" and w["verdict"] == "gnn"
    md["a.recall"].update(lo=0.0, hi=0.02)  # a CI touching 0 does not exclude it
    assert winner_verdict(res, _gnn_report_cfg())["metrics"]["a.recall"]["outcome"] == "tie"
    res["bootstrap"]["level"] = 0.9
    assert winner_verdict(res, _gnn_report_cfg())["level_matches"] is False
    del res["meta"]["models"]["lgbm_graph"]
    assert winner_verdict(res, _gnn_report_cfg()) is None
    with pytest.raises(ValueError):
        verdict_of("gnn", "win")


def test_report_hash_refusal(gnn_stage):
    from aml.eval.report import check_report_hash

    check_report_hash({"a": {"final": True, "report_hash": "h"}}, "h")
    check_report_hash({"dev": {"final": False}}, "h")  # validation-only sets record none
    with pytest.raises(ValueError, match="pre-registered `report` rules changed"):
        check_report_hash({"a": {"final": True, "report_hash": "old"}}, "h")
    with pytest.raises(ValueError, match="pre-registered"):
        check_report_hash({"a": {"final": True}}, "h")  # a --final set must record it
    s = gnn_stage
    edited = copy.deepcopy(s["gnn"])
    edited["report_cfg"]["winner_metrics"] = ["literature.pr_auc", "a.recall"]
    with pytest.raises(ValueError, match="pre-registered `report` rules changed"):
        _stage(s, "h", gnn=edited)
    assert not (s["out"] / "h" / "results.md").exists()  # refused before evaluating
    bad_views = {**s["gnn"], "model_views": {"gnn_lookahead_d10": ["tail"]}}
    with pytest.raises(ValueError, match="model_views"):
        _stage(s, "v", gnn=bad_views)


def test_guard_totals_and_lines():
    from aml.eval.report import guard_line, summary_guard

    per_split = {"val_early": _guard(10), "test": {**_guard(5, dropped=2), "max_slack": -3}}
    total = summary_guard({"guard": per_split})
    assert total["edges_checked"] == 15 and total["dropped_target_copies"] == 2
    assert total["max_slack"] == -1  # max over splits
    assert summary_guard({"guard": _guard(7)})["edges_checked"] == 7  # already a total
    assert summary_guard({}) is None
    assert guard_line("gnn_pna", {"guard": per_split}) == (
        "gnn_pna: 0 violations over 15 sampled edges; 0 target hits (2 sampled copies of the "
        "target dropped)."
    )
    assert guard_line("gnn_pna", {}) == "gnn_pna: no guard totals in its summary.json."
    bad = {"guard": {"test": {**_guard(5), "violations": 2, "target_hits": 1}}}
    assert guard_line("gnn_causal", bad).startswith("gnn_causal: 2 violations over 5")


def test_share_text_orders_splits_and_handles_missing_shares():
    """Summaries store sorted keys; the gap and faithful lines list splits in split order, and a
    split without sampled edges (share None) reads n/a, not 'n/a%'."""
    from aml.eval.report import NA, _share_text

    shares = {"test": 0.5, "train": 0.25, "val_late": None, "val_early": 0.125, "zz": 1.0}
    assert _share_text(shares) == (
        f"train 25.0%, val_early 12.5%, val_late {NA}, test 50.0%, zz 100.0%"
    )
    assert _share_text({}) == NA and _share_text(None) == NA
