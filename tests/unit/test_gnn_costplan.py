"""The GNN cost plan (M3 spec §10.2-§11.4): prices, the §11.2 arithmetic, the bench decision
rules, the cost gate and the worker limits. Torch-free: runs on the laptop too."""

from __future__ import annotations

import copy
import json
import math
import re

import pytest

from aml.config import load_config
from aml.models.gnn import check_gnn_cfg, decision_mismatches
from aml.models.gnn import costplan as cp
from tests.fixtures.gnn_graphs import REPO_ROOT

C, K = cp.CENTRAL, cp.CONS


@pytest.fixture(scope="module")
def gnn_cfg() -> dict:
    return load_config("gnn", REPO_ROOT / "configs")


@pytest.fixture(scope="module")
def cfg4096(gnn_cfg) -> dict:
    """§11.2's setting: train batch 4096, eval batch 8192."""
    c = copy.deepcopy(gnn_cfg)
    c["sampler"]["batch_size"] = 4096
    assert c["sampler"]["eval_batch_size"] == 8192
    return c


@pytest.fixture(scope="module")
def plan(cfg4096) -> list[dict]:
    return cp.plan_runs(cfg4096)


def rows_by_run(rows) -> dict:
    return {r["run"]: r for r in rows}


# --- prices and formulas ----------------------------------------------------------------------


def test_shape_prices():
    """GPU_price.md: 0.80 + 0.38 + 0.26 = 1.44 etc. (each part rounded to the cent); the exact
    formula is a fraction of a cent lower and the budget price rounds it back up."""
    shapes = {("L4", 8, 32): 1.44, ("T4", 8, 32): 1.23, ("L4", 4, 16): 1.12, ("T4", 4, 16): 0.91}
    exact = {("L4", 8, 32): 1.4344, ("T4", 8, 32): 1.2244, ("L4", 4, 16): 1.1172}
    for shape, table in shapes.items():
        assert cp.plan_usd_h(*shape) == table
        assert cp.shape_usd_h(*shape) == pytest.approx(table, abs=0.011)
        assert cp.shape_usd_h(*shape) <= cp.plan_usd_h(*shape)
    for shape, x in exact.items():
        assert cp.shape_usd_h(*shape) == pytest.approx(x, abs=1e-12)
    assert cp.plan_usd_h("L4", 8, 32) / 3600 == pytest.approx(0.0004)  # §11.1: $0.0004/s
    with pytest.raises(ValueError):
        cp.shape_usd_h("A100", 8, 32)


def test_basic_formulas():
    assert 2_534 + round(0.1 * 3_246_387) == 327_173  # §11.1
    assert cp.steps_per_epoch(2_534, 3_246_387, 0.1, 4096) == 80
    assert cp.steps_per_epoch(2_534, 3_246_387, 0.1, 2048) == 160
    assert cp.steps_per_epoch(10, 0, 0.1, 10) == 1
    assert cp.steps_per_epoch(11, 0, 0.1, 10) == 2
    with pytest.raises(ValueError):
        cp.steps_per_epoch(1, 1, 0.1, 0)
    assert cp.epoch_seconds(80, 0.1, 3.0, 2.0) == pytest.approx(13.0)
    assert cp.run_seconds(startup_s=180, n_containers=2, epochs=10, epoch_s=5, scoring_s=7) == 417
    assert cp.usd(1.44, 3600) == pytest.approx(1.44)
    assert cp.usd(1.44, 1800, 1.3) == pytest.approx(0.936)


def test_memory_request_rule():
    """Rule 5: max(16 GiB, 1.5 x peak) rounded up to 8 GiB; limit = request + 8 GiB."""
    assert cp.memory_request_mib(0) == (16384, 24576)
    assert cp.memory_request_mib(10_000) == (16384, 24576)  # 15,000 -> 16 GiB
    assert cp.memory_request_mib(10_923) == (24576, 32768)  # 16,384.5 -> 24 GiB
    assert cp.memory_request_mib(16_384) == (24576, 32768)  # 24,576 exactly
    assert cp.memory_request_mib(20_000) == (32768, 40960)


def test_timeout_formula():
    """§12.1: attempt wall = min(chunk_wall_s, 1.5 x projection); T = max(1800, ceil(wall +
    max(600, 3 x conservative epoch s)))."""
    assert cp.attempt_wall_s(7200, 4838.7, 1.5) == 7200
    assert cp.attempt_wall_s(7200, 400, 1.5) == 600
    assert cp.attempt_timeout_s(7200, 29.0) == 7800
    assert cp.attempt_timeout_s(7200, 400.0) == 8400  # 3 x 400 > 600
    assert cp.attempt_timeout_s(600, 29.0) == 1800  # floor
    assert cp.attempt_timeout_s(1300.2, 10.0) == 1901  # ceil
    # about 2 x the projection or less for any set longer than 20 min
    for proj in (1200.0, 3000.0, 4838.7, 20000.0):
        t = cp.attempt_timeout_s(cp.attempt_wall_s(7200, proj, 1.5), 29.0)
        assert t <= max(2 * proj, 1800)


def test_run_limits(plan, gnn_cfg):
    rt = gnn_cfg["runtime"]
    row = rows_by_run(plan)["causal"]
    lim = cp.run_limits(row, rt)
    proj = row["run_s"][K]
    assert lim["projected_s"] == proj
    assert lim["attempt_wall_s"] == min(rt["chunk_wall_s"], rt["wall_guard_factor"] * proj)
    assert lim["timeout_s"] == cp.attempt_timeout_s(lim["attempt_wall_s"], row["epoch_s"][K])
    assert lim["set_wall_limit_s"] == pytest.approx(
        rt["wall_guard_factor"] * proj + rt["chunk_wall_s"]
    )


# --- the §11.2 arithmetic, reproduced from the planning constants -----------------------------
# The spec rounds intermediate values for display (12.1 s, 29.0 s, ...), so totals are compared
# with a tolerance of one displayed unit: seconds within 0.5%, dollars within $0.011.


def usd_close(x: float, spec: float) -> bool:
    return abs(x - spec) <= 0.011


def secs_close(x: float, spec: float) -> bool:
    return abs(x - spec) <= 0.005 * spec


def test_planning_steps_and_epochs(cfg4096):
    t = cp._Timing(cfg4096, cp.HI_SMALL_COUNTS, None)
    assert (t.steps, t.n_val, t.batch, t.eval_batch) == (80, 59, 4096, 8192)
    ept = cp.PLANNING["edges_per_target"]
    # headline: 80 x (10 + 4096*148*120 ns = 82.7 ms) + 59 x (5 + 8192*164*40 ns = 58.7 ms) + 2
    assert 1000 * cp._plan_step(C, "train", ept["train"], 4096) == pytest.approx(82.7, abs=0.05)
    assert 1000 * cp._plan_step(C, "eval", ept["val_early"], 8192) == pytest.approx(58.7, abs=0.05)
    assert round(80 * cp._plan_step(C, "train", 148, 4096), 1) == 6.6
    assert round(59 * cp._plan_step(C, "eval", 164, 8192), 1) == 3.5
    assert round(t.epoch_s(C)[0], 1) == 12.1
    # conservative: edges x 1.5, 220 / 75 ns: 80 x 210 ms = 16.8 s + 59 x 156 ms = 9.2 s + 3
    assert round(1000 * cp._plan_step(K, "train", 148 * 1.5, 4096)) == 210
    assert round(1000 * cp._plan_step(K, "eval", 164 * 1.5, 8192)) == 156
    assert round(t.epoch_s(K)[0], 1) == 29.0
    # look-ahead (edges x 1.2): 80 x 97.2 ms + 59 x 69.5 ms + 2 = 13.9 s; cons 250 / 186 ms, 34.0 s
    assert 1000 * cp._plan_step(C, "train", 148 * 1.2, 4096) == pytest.approx(97.2, abs=0.1)
    assert 1000 * cp._plan_step(C, "eval", 164 * 1.2, 8192) == pytest.approx(69.5, abs=0.05)
    assert round(t.epoch_s(C, lookahead=True)[0], 1) == 13.9
    assert round(1000 * cp._plan_step(K, "train", 148 * 1.8, 4096)) == 250
    assert round(1000 * cp._plan_step(K, "eval", 164 * 1.8, 8192)) == 186
    assert round(t.epoch_s(K, lookahead=True)[0], 1) == 34.0


def test_planning_scoring_pass(cfg4096):
    """val_early + val_late + test = 1,829,424 seeds = 224 batches; 224 x k x 62 / 166 ms."""
    t = cp._Timing(cfg4096, cp.HI_SMALL_COUNTS, None)
    n_b, mean_ept = cp._passes(cp.HI_SMALL_COUNTS, cp.SCORED_FINAL, 8192)
    assert (
        n_b == 224
        and sum(cp.HI_SMALL_COUNTS[k] for k in ("n_val_early", "n_val_late", "n_test")) == 1_829_424
    )
    assert round(1000 * cp._plan_step(C, "eval", mean_ept, 8192)) == 62
    assert round(1000 * cp._plan_step(K, "eval", mean_ept * 1.5, 8192)) == 166
    assert round(t.scoring_s(C, cp.SCORED_FINAL, 5)) == 70  # headline set: "+ 70"
    assert round(t.scoring_s(K, cp.SCORED_FINAL, 5)) == 186  # "+ 186"
    # dev: val_early + val_late = 118 batches
    assert cp._passes(cp.HI_SMALL_COUNTS, cp.SCORED_DEV, 8192)[0] == 118
    # look-ahead also scores test with the d10 bound: §11.2 counts 2,693,324 seeds = 329 batches
    # over the total; per split (as the scoring pass batches) it is 59 + 59 + 106 + 106 = 330.
    n_la, _ = cp._passes(cp.HI_SMALL_COUNTS, cp.SCORED_LOOKAHEAD, 8192)
    assert math.ceil(2_693_324 / 8192) == 329 and n_la == 330


def test_planning_runs_match_section_11_2(plan, cfg4096):
    r = rows_by_run(plan)
    assert all(x["source"] == "planning" for x in plan)
    assert [x["run"] for x in plan] == list(cp.RUN_ORDER)
    assert all(x["price_usd_h"] == 1.44 for x in plan if x["run"] != "bench")

    # headline set (5 seeds): 5 x 18 x 12.1 + 70 + 180 = 1,339 s -> $0.70; cons 4,836 s -> $2.51
    h = r["causal"]
    assert h["epochs"] == {C: 18.0, K: 30.0} and h["units"] == 5
    assert secs_close(h["run_s"][C], 1339) and usd_close(h["usd"][C], 0.70)
    assert secs_close(h["run_s"][K], 4836) and usd_close(h["usd"][K], 2.51)
    assert h["n_containers"] == {C: 1, K: 1}

    # HPO: 8 x 9 x 12.1 + 180 = 1,051 s -> $0.55; 8 x 15 x 29.0 + 300 = 3,780 s -> $1.97; 4: $1.06
    p = r["hpo"]
    assert p["epochs"] == {C: 9.0, K: 15.0} and p["n_trials"] == 8 and p["scoring_s"][K] == 0
    assert secs_close(p["run_s"][C], 1051) and usd_close(p["usd"][C], 0.55)
    assert secs_close(p["run_s"][K], 3780) and usd_close(p["usd"][K], 1.97)
    assert p["cut_alt"]["n_trials"] == 4 and usd_close(p["cut_alt"]["usd"][K], 1.06)
    assert usd_close(p["usd"][K] - p["cut_alt"]["usd"][K], 0.91)  # "cut 2 (HPO -> 4: -$0.91)"

    # dev (2 epochs + val_early/val_late scoring): 211 s -> $0.11; 378 s -> $0.20
    d = r["dev"]
    assert d["epochs"] == {C: 2.0, K: 2.0}
    assert secs_close(d["run_s"][C], 211) and usd_close(d["usd"][C], 0.11)
    assert secs_close(d["run_s"][K], 378) and usd_close(d["usd"][K], 0.20)

    # look-ahead: one seed (cut) $0.24 / $0.72; seeds 1-2 in a second submission
    s0, rest = r["lookahead_s0"], r["lookahead_rest"]
    assert s0["seeds"] == [0] and rest["seeds"] == [1, 2]
    assert usd_close(s0["usd"][C], 0.24) and usd_close(s0["usd"][K], 0.72)
    # the set of 3 in one container: central 1,004 s -> $0.52, cons 3,556 s -> $1.85
    t = cp._Timing(cfg4096, cp.HI_SMALL_COUNTS, None)
    set3 = {}
    for w, e, start in ((C, 18, 180), (K, 30, 300)):
        ep = t.epoch_s(w, lookahead=True)[0]
        set3[w] = 3 * e * ep + t.scoring_s(w, cp.SCORED_LOOKAHEAD, 3, lookahead=True) + start
    assert secs_close(set3[C], 1004) and usd_close(cp.usd(1.44, set3[C], 1.3), 0.52)
    assert secs_close(set3[K], 3556) and usd_close(cp.usd(1.44, set3[K], 1.3), 1.85)
    # splitting into seed 0 then seeds 1-2 adds one startup (+$0.09 / +$0.16)
    for w, extra in ((C, 0.09), (K, 0.16)):
        split = s0["usd"][w] + rest["usd"][w]
        assert usd_close(split - cp.usd(1.44, set3[w], 1.3), extra)
    assert usd_close(s0["usd"][C] + rest["usd"][C], 0.61)  # §11.3 row
    assert usd_close(s0["usd"][K] + rest["usd"][K], 2.01)

    # bench: L4 720 s ($0.29) + T4 480 s ($0.16) + rerun 300 s ($0.09) + $0.03 -> $0.575 -> $0.75;
    # conservative x 1.5 -> $1.12
    b = r["bench"]
    assert usd_close(b["usd"][C] / 1.3, 0.575) and usd_close(b["usd"][C], 0.75)
    assert usd_close(b["usd"][K], 1.12)


def test_faithful_and_pna_rows_and_the_documented_differences(plan):
    """Faithful: 100 epochs, 397 train steps, 118 val, 106 test batches; central epoch 58.8 +
    7.7 + 2 = 68.5 s -> 7,098 s -> $3.69. PNA central 1,225 s -> $0.64.

    Two deliberate differences from §11.2's conservative arithmetic (both make the gate more
    conservative, neither changes a cut): the faithful snapshot build (120 s) is paid in each of
    the 3 chunk containers, not once ($9.80 -> $9.93), and the PNA set's 8,4xx s of work exceeds
    one chunk (runtime.chunk_wall_s 7,200 s), so it pays a second startup ($4.40 -> $4.56)."""
    r = rows_by_run(plan)
    f = r["faithful"]
    assert f["epochs"] == {C: 100.0, K: 100.0} and f["batch_size"] == 8192
    assert math.ceil(3_248_921 / 8192) == 397
    assert math.ceil((482_751 + 482_773) / 8192) == 118 and math.ceil(863_900 / 8192) == 106
    assert round(f["epoch_s"][C], 1) == pytest.approx(68.4, abs=0.1)  # 58.8 + 7.7 + 2
    assert round(f["epoch_s"][K], 1) == 178.1  # 154.6 + 20.5 + 3
    assert round(f["scoring_s"][C]) == 8 and round(f["scoring_s"][K]) == 23
    assert secs_close(f["run_s"][C], 7098) and usd_close(f["usd"][C], 3.69)
    assert f["n_containers"] == {C: 1, K: 3}
    spec_cons_s = 18_853  # 17,810 + 23 + 600 + 300 + 120 (build once)
    assert secs_close(f["run_s"][K] - 2 * 120, spec_cons_s)
    assert usd_close(cp.usd(1.44, spec_cons_s, 1.3), 9.80)
    assert usd_close(f["usd"][K], 9.93)

    p = r["pna"]
    assert round(p["epoch_s"][C], 1) == pytest.approx(18.15, abs=0.05)  # 1.5 x 12.1
    assert round(p["epoch_s"][K]) == 87  # 3 x 29.0
    assert round(p["scoring_s"][K]) == 335 and abs(p["scoring_s"][C] - 62) < 1
    assert secs_close(p["run_s"][C], 1225) and usd_close(p["usd"][C], 0.64)
    assert p["n_containers"] == {C: 1, K: 2}
    assert secs_close(p["run_s"][K] - 300, 8465)
    assert usd_close(cp.usd(1.44, 8465, 1.3), 4.40) and usd_close(p["usd"][K], 4.56)


def test_section_11_3_totals_and_the_first_cut(plan, cfg4096):
    """Central: subtotal without PNA $6.41, total $7.05; conservative: $17.61 / $22.01 in the
    spec (+ the two documented differences: $17.74 / $22.30) -> cut 1 (PNA), no further cut."""
    no_pna = [x for x in plan if x["run"] != "pna"]
    # §11.3 adds rows rounded to the cent: allow half a cent per row (8 rows)
    assert abs(sum(x["usd"][C] for x in no_pna) - 6.41) <= 0.035
    assert abs(sum(x["usd"][C] for x in plan) - 7.05) <= 0.04
    diff = 0.13 + 0.16  # faithful build x 2 containers, PNA second startup
    assert abs(sum(x["usd"][K] for x in no_pna) - (17.61 + 0.13)) <= 0.035
    assert abs(sum(x["usd"][K] for x in plan) - (22.01 + diff)) <= 0.04
    g = cp.gate(plan, gnn_cfg=cfg4096, spent_usd=0.0, done=(), job="bench", metered_usd=0.0)
    assert g["allowed"] and g["cuts"] == ["pna"] and g["faithful_epoch_cap"] is None
    assert g["projected_usd"] == pytest.approx(sum(x["usd"][K] for x in no_pna))
    assert g["projected_usd"] <= 20.0


def override(plan, usd_cons: dict[str, float]) -> list[dict]:
    rows = copy.deepcopy(plan)
    for r in rows:
        if r["run"] in usd_cons:
            r["usd"] = {C: usd_cons[r["run"]], K: usd_cons[r["run"]]}
    return rows


def test_stress_case_reproduces_the_faithful_cap(plan, cfg4096):
    """§11.3 stress case: faithful ≈ 44,700 s ≈ $17.9 ($0.179 / epoch). Never-cut + HPO = 1.12 +
    0.20 + 1.97 + 2.51 + 2.01 + 17.9 = $25.71 -> cut 2 (-$0.91) -> cut 3 (-$1.29) -> $23.51 ->
    faithful epoch cap = floor((20 - 5.61) / 0.179) = 80."""
    assert round(12_958 * 3 * 1.15) == 44_705
    vals = {
        "bench": 1.12,
        "dev": 0.20,
        "hpo": 1.97,
        "causal": 2.51,
        "lookahead_s0": 0.72,
        "lookahead_rest": 1.29,
        "pna": 4.40,
    }
    rows = override(plan, vals)
    by = rows_by_run(rows)
    by["hpo"]["cut_alt"]["usd"] = {C: 1.06, K: 1.06}
    f = by["faithful"]  # $0.179 per epoch, nothing fixed (the spec's stress arithmetic)
    f.update(price_usd_h=1.44, overhead=1.0, chunk_wall_s=1e9)
    f["startup_s"] = {C: 0.0, K: 0.0}
    f["scoring_s"] = {C: 0.0, K: 0.0}
    f["epoch_s"] = {C: 447.5, K: 447.5}
    f["usd"] = {C: 17.9, K: 17.9}
    assert cp._cost(f, K)["usd"] == pytest.approx(17.9)

    never_and_hpo = sum(vals[k] for k in ("bench", "dev", "hpo", "causal")) + 0.72 + 1.29 + 17.9
    assert never_and_hpo == pytest.approx(25.71)
    assert never_and_hpo - 0.91 - 1.29 == pytest.approx(23.51)
    assert math.floor((20 - 5.61) / 0.179) == 80

    g = cp.gate(rows, gnn_cfg=cfg4096, spent_usd=0.0, done=(), job="causal", metered_usd=0.0)
    assert g["cuts"] == ["pna", "hpo_4", "lookahead_1"]
    assert g["faithful_epoch_cap"] == 80
    assert g["projected_usd"] == pytest.approx(5.61 + 80 * 0.179)
    assert g["allowed"]  # causal is never cut; the cap concerns the faithful submission
    assert "protocols.faithful.epoch_cap: 80 (a documented deviation)" in g["actions"]
    # the faithful job is refused until gnn.yaml carries the cap
    gf = cp.gate(rows, gnn_cfg=cfg4096, spent_usd=0.0, done=(), job="faithful", metered_usd=0.0)
    assert not gf["allowed"] and "epoch_cap: 80" in gf["refuse_reason"]
    capped = copy.deepcopy(cfg4096)
    capped["protocols"]["faithful"]["epoch_cap"] = 80
    gf2 = cp.gate(rows, gnn_cfg=capped, spent_usd=0.0, done=(), job="faithful", metered_usd=0.0)
    assert gf2["allowed"], gf2["refuse_reason"]


# --- the gate -------------------------------------------------------------------------------------


def independent_faithful_usd(f: dict, epochs: int) -> float:
    """§10.2 written out: price/3600 x (startup x containers + E x epoch_s + scoring) x overhead,
    one container per chunk_wall_s of work."""
    work = epochs * f["epoch_s"][K] + f["scoring_s"][K]
    n = max(1, math.ceil(work / f["chunk_wall_s"]))
    return f["price_usd_h"] / 3600 * (f["startup_s"][K] * n + work) * f["overhead"]


def test_gate_cut_order_only_as_needed(plan, cfg4096):
    total_cons = sum(x["usd"][K] for x in plan)
    by = rows_by_run(plan)
    after_pna = total_cons - by["pna"]["usd"][K]
    hpo_save = by["hpo"]["usd"][K] - by["hpo"]["cut_alt"]["usd"][K]
    la_save = by["lookahead_rest"]["usd"][K]

    def run(spent, **kw):
        return cp.gate(plan, gnn_cfg=cfg4096, spent_usd=spent, job="bench", metered_usd=0.0, **kw)

    assert run(0.0, done=())["cuts"] == ["pna"]
    assert run(20.0 - total_cons - 1e-9, done=())["cuts"] == []  # at the cap: no cut
    g = run(20.0 - after_pna + 0.5, done=())
    assert g["cuts"] == ["pna", "hpo_4"]
    assert g["projected_usd"] == pytest.approx(20.5 - hpo_save)
    g = run(20.0 - after_pna + hpo_save + 0.5, done=())
    assert g["cuts"] == ["pna", "hpo_4", "lookahead_1"]
    assert g["projected_usd"] == pytest.approx(20.5 - la_save)
    assert g["faithful_epoch_cap"] is None
    # only cuts of runs not yet run: HPO done -> cut 2 is skipped, cut 3 is used instead
    done3 = ("bench", "dev", "hpo")
    g = run(20.0 - (after_pna - sum(by[r]["usd"][K] for r in done3)) + 0.5, done=done3)
    assert g["cuts"] == ["pna", "lookahead_1"]
    assert rows_by_run(g["rows"])["hpo"]["status"] == "done"
    # the never-cut set is never cut
    for spent in (0.0, 5.0, 10.0):
        rows = rows_by_run(run(spent, done=())["rows"])
        assert all(rows[n]["status"] == "planned" for n in cp.NEVER_CUT)


def test_gate_faithful_cap_formula_and_refusal(plan, cfg4096):
    by = rows_by_run(plan)
    f = by["faithful"]
    others_after_cuts = (
        sum(
            x["usd"][K]
            for x in plan
            if x["run"] not in ("pna", "lookahead_rest", "faithful", "hpo")
        )
        + by["hpo"]["cut_alt"]["usd"][K]
    )
    spent = 6.0
    g = cp.gate(plan, gnn_cfg=cfg4096, spent_usd=spent, done=(), job="dev", metered_usd=0.0)
    assert g["cuts"] == ["pna", "hpo_4", "lookahead_1"]
    budget = 20.0 - spent - others_after_cuts
    want = max(e for e in range(0, 101) if independent_faithful_usd(f, e) <= budget)
    assert g["faithful_epoch_cap"] == want and cp.MIN_FAITHFUL_EPOCHS <= want < 100
    assert g["allowed"]  # dev may run; the faithful job needs the cap in gnn.yaml first
    fr = rows_by_run(g["rows"])["faithful"]
    assert fr["epochs"][K] == want and fr["usd"][K] == pytest.approx(
        independent_faithful_usd(f, want)
    )
    assert g["projected_usd"] <= 20.0
    # the fixed part counts: a pure $/epoch division would allow more epochs
    per_epoch = cp.usd(f["price_usd_h"], f["epoch_s"][K], f["overhead"])
    assert math.floor(budget / per_epoch) > want

    # below 50 epochs: refuse every job and ask the user
    for job in ("dev", "causal", "faithful"):
        g = cp.gate(plan, gnn_cfg=cfg4096, spent_usd=10.0, done=(), job=job, metered_usd=0.0)
        assert g["faithful_epoch_cap"] < cp.MIN_FAITHFUL_EPOCHS
        assert not g["allowed"] and "ask the user" in g["refuse_reason"]
    # faithful already done and still over the cap: nothing left to cut
    g = cp.gate(
        plan,
        gnn_cfg=cfg4096,
        spent_usd=19.0,
        done=("bench", "faithful"),
        job="causal",
        metered_usd=0.0,
    )
    assert not g["allowed"] and "nothing left to cut" in g["refuse_reason"]


def test_gate_refuses_cut_jobs_until_gnn_yaml_applies_them(plan, cfg4096):
    g = cp.gate(plan, gnn_cfg=cfg4096, spent_usd=0.0, done=(), job="pna", metered_usd=0.0)
    assert not g["allowed"] and "cuts 'pna'" in g["refuse_reason"]
    by = rows_by_run(plan)
    spent = 20.0 - (sum(x["usd"][K] for x in plan) - by["pna"]["usd"][K]) + 0.3
    g = cp.gate(plan, gnn_cfg=cfg4096, spent_usd=spent, done=(), job="hpo", metered_usd=0.0)
    assert g["cuts"] == ["pna", "hpo_4"]
    assert not g["allowed"] and "hpo.n_trials: 4" in g["refuse_reason"]
    assert "hpo.n_trials: 4 (cut 2)" in g["actions"]
    # once gnn.yaml says 4 trials, the plan has no HPO cut left and the job may run
    c4 = copy.deepcopy(cfg4096)
    c4["hpo"]["n_trials"] = 4
    p4 = cp.plan_runs(c4)
    assert rows_by_run(p4)["hpo"]["cut_alt"] is None
    g = cp.gate(p4, gnn_cfg=c4, spent_usd=spent, done=(), job="hpo", metered_usd=0.0)
    assert g["allowed"] and "hpo_4" not in g["cuts"]
    # look-ahead seeds [0] in gnn.yaml: the rest row is cut by the config, costs nothing
    c1 = copy.deepcopy(cfg4096)
    c1["protocols"]["lookahead"]["seeds"] = [0]
    p1 = rows_by_run(cp.plan_runs(c1))
    assert p1["lookahead_rest"]["status"] == "cut" and p1["lookahead_rest"]["usd"][K] == 0
    g = cp.gate(
        list(p1.values()), gnn_cfg=c1, spent_usd=0.0, done=(), job="lookahead_rest", metered_usd=0.0
    )
    assert not g["allowed"]
    # one submission covering every look-ahead seed when cut 3 applies: refused
    spent3 = spent + by["hpo"]["usd"][K]
    g = cp.gate(
        plan,
        gnn_cfg=cfg4096,
        spent_usd=spent3,
        done=(),
        job=["lookahead_s0", "lookahead_rest"],
        metered_usd=0.0,
    )
    assert "lookahead_1" in g["cuts"] and not g["allowed"]
    assert "protocols.lookahead.seeds: [0] (cut 3)" in g["actions"]


def test_gate_budget_precheck(plan, cfg4096):
    """metered + this job's bound <= workspace budget - M6 reserve ($29). review MONEY-1: the
    bound is what the submission can spend before its driver stops it, price x (wall guard
    factor x conservative run seconds + one chunk wall), not its conservative $."""
    row = rows_by_run(plan)["causal"]
    job_usd = row["usd"][K]
    rt = cfg4096["runtime"]
    bound = cp.usd(
        row["price_usd_h"], rt["wall_guard_factor"] * row["run_s"][K] + rt["chunk_wall_s"]
    )
    assert bound > job_usd
    assert cp.job_bound_usd([row], rt) == pytest.approx(bound)
    assert cp.run_limits(row, rt)["set_wall_limit_s"] == pytest.approx(
        rt["wall_guard_factor"] * row["run_s"][K] + rt["chunk_wall_s"]
    )
    ok = cp.gate(
        plan, gnn_cfg=cfg4096, spent_usd=0, done=(), job="causal", metered_usd=29.0 - bound - 0.01
    )
    assert ok["allowed"] and ok["budget_check"]["ok"]
    assert ok["budget_check"]["limit_usd"] == 29.0
    assert ok["budget_check"]["job_usd"] == pytest.approx(job_usd)
    assert ok["budget_check"]["job_bound_usd"] == pytest.approx(bound)
    assert "job bound" in cp.render_gate(ok)
    bad = cp.gate(
        plan, gnn_cfg=cfg4096, spent_usd=0, done=(), job="causal", metered_usd=29.0 - bound + 0.01
    )
    assert not bad["allowed"] and "budget pre-check" in bad["refuse_reason"]
    # the old charge (conservative $ only) would have let this one through
    assert 29.0 - bound + 0.01 + job_usd <= 29.0
    # the faithful run: about $14 of bound for a $9.93 conservative charge (planning)
    f = rows_by_run(plan)["faithful"]
    assert 14.0 < cp.job_bound_usd([f], rt) < 15.0 and f["usd"][K] < 10.0
    # the bench (fixed row, per-container timeouts) is charged its conservative $
    b = rows_by_run(plan)["bench"]
    assert cp.job_bound_usd([b], rt) == pytest.approx(b["usd"][K])
    none = cp.gate(plan, gnn_cfg=cfg4096, spent_usd=0, done=(), job="causal", metered_usd=None)
    assert not none["allowed"] and "unavailable" in none["refuse_reason"]


def test_gate_bookkeeping(plan, cfg4096):
    done = ("bench", "dev")
    g = cp.gate(plan, gnn_cfg=cfg4096, spent_usd=1.0, done=done, job="causal", metered_usd=0.0)
    by = rows_by_run(g["rows"])
    assert by["bench"]["status"] == by["dev"]["status"] == "done"
    planned = [r for r in g["rows"] if r["status"] == "planned"]
    assert g["projected_usd"] == pytest.approx(1.0 + sum(r["usd"][K] for r in planned))
    assert g["warnings"] == ["runs earlier in the run order are not done: ['hpo']"]
    assert plan == cp.plan_runs(cfg4096)  # the input plan is not mutated
    # a done job costs nothing more
    g = cp.gate(
        plan, gnn_cfg=cfg4096, spent_usd=1.0, done=("causal",), job="causal", metered_usd=0.0
    )
    assert g["budget_check"]["job_usd"] == 0.0
    # plan.json round trip
    rt = json.loads(json.dumps(plan))
    g2 = cp.gate(rt, gnn_cfg=cfg4096, spent_usd=1.0, done=done, job="causal", metered_usd=0.0)
    assert g2["projected_usd"] == pytest.approx(
        cp.gate(plan, gnn_cfg=cfg4096, spent_usd=1.0, done=done, job="causal", metered_usd=0.0)[
            "projected_usd"
        ]
    )
    with pytest.raises(ValueError):
        cp.gate(plan, gnn_cfg=cfg4096, spent_usd=0, done=(), job="train", metered_usd=0.0)
    with pytest.raises(ValueError):
        cp.gate(plan, gnn_cfg=cfg4096, spent_usd=0, done=("nope",), job="dev", metered_usd=0.0)


def test_gate_charges_a_started_run_only_for_what_is_left(plan, cfg4096):
    """A re-submission after a stop (wall guard, driver deadline, exhausted retries): the
    partial spend is already in spent_usd, so the started run pays only its remaining epochs.
    Charging it in full again would refuse this faithful re-submission, cut look-ahead seeds
    1-2 and demand an epoch cap (which re-keys the faithful run: a restart from epoch 0)."""
    by = rows_by_run(plan)
    f = by["faithful"]
    done = ("bench", "dev", "hpo", "causal", "lookahead_s0")
    spent = sum(by[r]["usd"][K] for r in done) + 0.6 * f["usd"][K]  # 60 of 100 epochs paid

    def run(progress=None, job="faithful"):
        return cp.gate(
            plan,
            gnn_cfg=cfg4096,
            spent_usd=spent,
            done=done,
            job=job,
            metered_usd=0.0,
            progress=progress,
        )

    blind = run()
    assert not blind["allowed"] and blind["cuts"] == ["pna", "lookahead_1"]
    assert "epoch_cap" in blind["refuse_reason"]
    g = run({"faithful": {"units_done": 0, "epochs_done": 60}})
    assert g["allowed"], g["refuse_reason"]
    assert g["cuts"] == ["pna"] and g["faithful_epoch_cap"] is None
    fr = rows_by_run(g["rows"])["faithful"]
    assert fr["usd"][K] == pytest.approx(independent_faithful_usd(f, 40))  # startup, test: full
    assert g["budget_check"]["job_usd"] == pytest.approx(fr["usd"][K])
    left = sum(x["usd"][K] for x in g["rows"] if x["status"] == "planned")
    assert g["projected_usd"] == pytest.approx(spent + left) and g["projected_usd"] <= 20.0
    assert "[started: 0 done + 60 epochs" in cp.render_gate(g)
    assert plan == cp.plan_runs(cfg4096)  # the input plan is not mutated
    # finished seeds count whole units: 2 of 5 causal seeds done + 7 epochs of the third
    c = by["causal"]
    g = cp.gate(
        plan,
        gnn_cfg=cfg4096,
        spent_usd=0.0,
        done=("bench",),
        job="causal",
        metered_usd=0.0,
        progress={"causal": {"units_done": 2, "epochs_done": 7}},
    )
    e = c["epochs"][K]
    want = cp._cost({**c, "units": 1}, K, epochs=c["units"] * e - (2 * e + 7))["usd"]
    assert rows_by_run(g["rows"])["causal"]["usd"][K] == pytest.approx(want)
    # progress of a finished run changes nothing; unknown run ids and negatives are errors
    g = run({"causal": {"units_done": 1, "epochs_done": 3}})
    assert rows_by_run(g["rows"])["causal"]["status"] == "done"
    with pytest.raises(ValueError, match="unknown run ids"):
        run({"train": {"units_done": 1}})
    with pytest.raises(ValueError, match="negative"):
        run({"faithful": {"epochs_done": -1}})


def test_gate_never_cuts_a_started_run(plan, cfg4096):
    by = rows_by_run(plan)
    after_pna = sum(x["usd"][K] for x in plan) - by["pna"]["usd"][K]
    hpo_save = by["hpo"]["usd"][K] - by["hpo"]["cut_alt"]["usd"][K]
    spent = 20.0 - after_pna + hpo_save + 0.5  # cuts 1-3 are all needed

    def run(progress):
        return cp.gate(
            plan,
            gnn_cfg=cfg4096,
            spent_usd=spent,
            done=(),
            job="dev",
            metered_usd=0.0,
            progress=progress,
        )

    assert run(None)["cuts"] == ["pna", "hpo_4", "lookahead_1"]
    # a started HPO keeps its trial count (cut 2 would change the HPO key: a restart) and a
    # started look-ahead rest keeps its seeds
    g = run({"hpo": {"units_done": 2, "epochs_done": 4}, "lookahead_rest": {"units_done": 1}})
    assert "hpo_4" not in g["cuts"] and "lookahead_1" not in g["cuts"]
    rows = rows_by_run(g["rows"])
    assert rows["hpo"]["status"] == rows["lookahead_rest"]["status"] == "planned"


def test_a_cap_on_a_started_faithful_run_is_priced_as_a_restart(plan, cfg4096):
    """epoch_cap is in the faithful run key: a cap demanded while the run is under way starts a
    new run from epoch 0, so the cap is computed without the progress credit, with a warning."""
    started = {"faithful": {"epochs_done": 30}}
    kw = dict(gnn_cfg=cfg4096, done=(), job="dev", metered_usd=0.0)
    # $6 spent: a fresh faithful run needs a cap; 30 epochs already trained: finishing fits
    assert cp.gate(plan, spent_usd=6.0, **kw)["faithful_epoch_cap"] is not None
    g = cp.gate(plan, spent_usd=6.0, **kw, progress=started)
    assert g["faithful_epoch_cap"] is None and g["allowed"]
    assert not any("restart" in w for w in g["warnings"])
    # $8 spent: even finishing does not fit -> the cap is priced as a restart
    fresh = cp.gate(plan, spent_usd=8.0, **kw)
    g = cp.gate(plan, spent_usd=8.0, **kw, progress=started)
    assert g["faithful_epoch_cap"] == fresh["faithful_epoch_cap"] >= cp.MIN_FAITHFUL_EPOCHS
    assert any("restart from epoch 0" in w for w in g["warnings"])
    assert rows_by_run(g["rows"])["faithful"]["usd"][K] == pytest.approx(
        rows_by_run(fresh["rows"])["faithful"]["usd"][K]
    )


def test_raise_if_refused(plan, cfg4096):
    from aml.models.gnn import GnnStopError

    ok = cp.gate(plan, gnn_cfg=cfg4096, spent_usd=0.0, done=(), job="dev", metered_usd=0.0)
    cp.raise_if_refused(ok)  # allowed: no error
    bad = cp.gate(plan, gnn_cfg=cfg4096, spent_usd=0.0, done=(), job="pna", metered_usd=0.0)
    with pytest.raises(GnnStopError, match="cost gate refused pna") as e:
        cp.raise_if_refused(bad)
    assert e.value.detail["cuts"] == ["pna"] and "rows" not in e.value.detail


def test_plan_follows_gnn_yaml(gnn_cfg):
    """plan_runs reads the configured batch (2048 until the bench decides), seeds, trials and
    the faithful epoch cap."""
    rows = rows_by_run(cp.plan_runs(gnn_cfg))
    assert rows["causal"]["batch_size"] == gnn_cfg["sampler"]["batch_size"]
    assert rows["causal"]["seeds"] == gnn_cfg["protocols"]["causal"]["seeds"]
    assert rows["pna"]["seeds"] == gnn_cfg["protocols"]["pna"]["seeds"]
    assert rows["hpo"]["n_trials"] == gnn_cfg["hpo"]["n_trials"]
    capped = copy.deepcopy(gnn_cfg)
    capped["protocols"]["faithful"]["epoch_cap"] = 70
    f = rows_by_run(cp.plan_runs(capped))["faithful"]
    assert f["epochs"] == {C: 70.0, K: 70.0} and f["max_epochs"] == 100 and f["epoch_cap"] == 70
    # a smaller batch pays more fixed per-step cost: never cheaper than the 4096 plan
    c4096 = copy.deepcopy(gnn_cfg)
    c4096["sampler"]["batch_size"] = 4096
    assert rows["causal"]["usd"][K] >= rows_by_run(cp.plan_runs(c4096))["causal"]["usd"][K]


# --- the bench decision -----------------------------------------------------------------------

GB = 10**9


def make_cells(
    gpu: str,
    *,
    cores: int = 8,
    scale: float = 1.0,
    step4096: float = 2.0,
    edges: float = 300_000.0,
    bpe: float = 6000.0,
    wait: float = 0.05,
    faithful_batch: int | None = 8192,
    f_scale: float = 1.0,
    peak_mib: float = 9000.0,
    extras: bool = False,
    ratio: float = 1.1,
) -> list[dict]:
    """Synthetic bench cells of one container. Step time 40 ms x scale at batch 2048 and 7
    workers (3 workers: +10%); 4096 costs step4096 x the 2048 step."""
    dev = (24 if gpu == "L4" else 16) * GB
    base = {
        "gpu": gpu,
        "cores": cores,
        "memory_mib": 32768,
        "status": "ok",
        "device_bytes": dev,
        "device_name": f"NVIDIA {gpu}",
        "host_peak_mib": peak_mib,
        "error": None,
    }

    def cell(group, b=None, w=None, **kw):
        return {
            **base,
            "group": group,
            "cell_id": cp.cell_id(gpu, cores, group, b, w),
            "batch_size": b,
            "num_workers": w,
            **kw,
        }

    out = [cell("build", build_s=90.0, counts=dict(cp.HI_SMALL_COUNTS))]
    for b in (2048, 4096):
        for w in (3, 7):
            s = 0.040 * scale * (step4096 if b == 4096 else 1.0) * (1.1 if w == 3 else 1.0)
            e = edges * b / 2048
            out.append(
                cell(
                    "grid",
                    b,
                    w,
                    step_s_mean=s,
                    step_s_p90=1.2 * s,
                    wait_share=wait,
                    edges_mean=e,
                    edges_max=2 * e,
                    peak_mem_bytes=int(e * bpe),
                    bytes_per_edge=bpe,
                )
            )
        e = 3 * edges * b / 2048
        out.append(
            cell(
                "stress",
                b,
                edges=e,
                peak_mem_bytes=int(e * bpe),
                bytes_per_edge=bpe,
                max_edges_per_step=int(0.6 * dev / bpe),
            )
        )
    out.append(
        cell(
            "val",
            val_s=6.0 * scale,
            val_cached_s=2.0 * scale,
            cache_gb=1.0,
            cache_fits=True,
            eval_batches=59,
            eval_bytes_per_edge=1500.0,
        )
    )
    tried = [8192] if faithful_batch == 8192 else [8192, 4096, 2048][: 2 if faithful_batch else 3]
    out.append(
        cell(
            "faithful",
            faithful_batch,
            7,
            tried=tried,
            step_s_mean=0.15 * scale * f_scale,
            val_s_per_batch=0.065 * scale * f_scale,
            test_s_per_batch=0.08 * scale * f_scale,
            edges_mean=1.15e6,
        )
    )
    if faithful_batch is None:
        out[-1]["status"] = "oom"
    if extras:
        out.append(
            cell(
                "lookahead",
                2048,
                7,
                step_s_mean=0.040 * scale * 1.2,
                val_batches=20,
                val_s_per_batch=6.0 * scale / 59 * 1.2,
            )
        )
        out.append(cell("pna", 2048, 7, step_s_mean=0.040 * scale * 1.5))
        out.append(
            cell(
                "determinism",
                bit_deterministic=False,
                max_param_diff=1e-7,
                det_overhead_ratio=ratio,
                cpu_gpu_max_logit_diff=1e-5,
            )
        )
    return out


def by_cell(decision) -> dict:
    return {e["cell_id"]: e for e in decision["cells"]}


def test_decide_prefers_l4_within_the_tie_band(gnn_cfg):
    """T4 wins only if cheaper by more than bench.tie_band (5%)."""
    assert gnn_cfg["bench"]["tie_band"] == 0.05
    l4 = make_cells("L4", extras=True)
    # T4 price / L4 price at 8 cores, 16 GiB = 1.0964 / 1.3064: T4 about 16% cheaper at equal time
    close = cp.decide(l4 + make_cells("T4", scale=1.2), gnn_cfg)
    t = by_cell(close)
    l4_best = min(e["usd_epoch"] for e in t.values() if e["gpu"] == "L4")
    t4_best = min(e["usd_epoch"] for e in t.values() if e["gpu"] == "T4")
    assert t4_best < l4_best <= t4_best * 1.05
    assert close["gpu"] == "L4" and close["precision"] == "high"
    far = cp.decide(l4 + make_cells("T4", scale=1.0), gnn_cfg)
    assert far["gpu"] == "T4" and far["precision"] == "highest"
    t = by_cell(far)
    assert min(e["usd_epoch"] for e in t.values() if e["gpu"] == "L4") > 1.05 * far["usd_epoch"]


def test_decide_batch_and_workers(gnn_cfg):
    # equal epochs at 2048 and 4096: the tie goes to 2048; 7 workers are faster than 3
    d = cp.decide(make_cells("L4", extras=True), gnn_cfg)
    assert (d["batch_size"], d["num_workers"], d["cores"]) == (2048, 7, 8)
    assert d["cell_id"] == "L4-8c-grid-b2048-w7"
    # 4096 within 5%: still 2048; 4096 cheaper by more than 5%: 4096
    d = cp.decide(make_cells("L4", step4096=1.95, extras=True), gnn_cfg)
    assert d["batch_size"] == 2048
    d = cp.decide(make_cells("L4", step4096=1.6, extras=True), gnn_cfg)
    assert d["batch_size"] == 4096
    # the decision's epoch / $ follow §10.2: steps x step + cached val pass + 2 s, exact price
    e = by_cell(d)[d["cell_id"]]
    assert e["epoch_s"] == pytest.approx(80 * 0.040 * 1.6 + 2.0 + 2.0)
    assert d["usd_epoch"] == pytest.approx(
        cp.shape_usd_h("L4", 8, d["memory_mib"] / 1024) / 3600 * e["epoch_s"]
    )


def test_decide_rule_1_edge_cap(gnn_cfg):
    """A cell whose edge cap is < 1.5 x its mean batch edges is not eligible, however cheap."""
    cells = make_cells("L4", extras=True)
    cap = math.floor(0.6 * (24 * GB) / 6000.0)
    for c in cells:
        if c["cell_id"] == "L4-8c-grid-b4096-w7":
            c["edges_mean"] = cap / 1.5 + 1
            c["step_s_mean"] = 0.001  # by far the cheapest
    d = cp.decide(cells, gnn_cfg)
    t = by_cell(d)
    assert (
        not t["L4-8c-grid-b4096-w7"]["eligible"] and "edge cap" in t["L4-8c-grid-b4096-w7"]["why"]
    )
    assert d["cell_id"] != "L4-8c-grid-b4096-w7"
    assert d["max_edges_per_step"] == cap
    assert d["bytes_per_edge"]["train"] == 6000.0
    assert d["max_edges_per_eval_step"] == math.floor(0.6 * (24 * GB) / 1500.0)
    for c in cells:
        if c["group"] == "grid":
            c["edges_mean"] = cap  # nothing eligible
    with pytest.raises(ValueError, match="no eligible cell"):
        cp.decide(cells, gnn_cfg)


def test_decide_faithful_keeps_its_measured_shape(gnn_cfg):
    """review perf-2: the faithful run keeps the shape its cell was measured on (8 cores, its
    workers) even when the 4-core rerun wins the headline shape; its $ / epoch is priced there,
    plan_runs and the faithful worker use it. Eval passes add one first batch (the worker
    fork) per pass instead of spreading it over the timed batches (perf-4)."""
    t4 = make_cells("T4", scale=1.2, wait=0.5, f_scale=2.0)  # its faithful epoch is dearer
    low = make_cells("L4", extras=True, wait=0.05) + t4
    rerun = make_cells("L4", cores=4)
    rr = [c for c in rerun if c["cell_id"] == "L4-4c-grid-b2048-w3"][0]
    rr.update(
        group="rerun",
        cell_id="L4-4c-rerun-b2048-w3",
        step_s_mean=0.044,
        val_s=6.0,
        val_cached_s=2.0,
        cache_fits=True,
        eval_batches=59,
    )
    cells = low + [rerun[0], rr]
    for c in cells:
        if c["group"] == "faithful" and c["gpu"] == "L4":
            c.update(val_first_batch_s=1.3, test_first_batch_s=1.4)
    d = cp.decide(cells, gnn_cfg)
    assert (d["cores"], d["num_workers"]) == (4, 3)
    f = d["faithful"]
    assert (f["cores"], f["num_workers"], f["memory_mib"]) == (8, 7, d["memory_mib"])
    ep = 397 * 0.15 + 118 * 0.065 + 1.3 + 2.0
    assert f["gpu"] == "L4"
    price = cp.shape_usd_h("L4", 8, d["memory_mib"] / 1024)
    assert f["usd_epoch"] == pytest.approx(price / 3600 * ep)
    rows = rows_by_run(cp.plan_runs(gnn_cfg, decision=d))
    assert rows["causal"]["cores"] == 4 and rows["faithful"]["cores"] == 8
    assert rows["faithful"]["epoch_s"][C] == pytest.approx(ep)
    assert rows["faithful"]["scoring_s"][C] == pytest.approx(106 * 0.08 + 1.4)


def test_lookahead_central_epoch_replays_its_eval_cache(gnn_cfg):
    """review perf-4: TemporalEngine caches the look-ahead val trees too, so the central
    look-ahead epoch uses the headline cached pass scaled by edges per target; the
    conservative one the uncached batches + the first batch (the worker fork)."""
    cells = make_cells("L4", extras=True)
    for c in cells:
        if c["group"] == "val":
            c.update(eval_edges=59 * 8192 * 160, eval_rows=59 * 8192)
        if c["group"] == "lookahead":
            c.update(edges_per_target_val=200.0, val_first_batch_s=1.2)
    d = cp.decide(cells, gnn_cfg)
    la = d["measured"]["lookahead"]
    assert la["val_cached_s"] == pytest.approx(2.0 * 200.0 / 160.0)
    rows = rows_by_run(cp.plan_runs(gnn_cfg, decision=d))
    steps = cp.steps_per_epoch(2_534, 3_246_387, 0.1, d["batch_size"])
    r = rows["lookahead_s0"]
    assert r["epoch_s"][C] == pytest.approx(steps * la["step_s"] + 2.5 + 2)
    cons = steps * la["step_s"] + 59 * la["val_s_per_batch"] + 1.2 + 3
    assert r["epoch_s"][K] == pytest.approx(cons)
    # a cache that would not fit (eval_cache_max_gb) gets no cached credit
    small = copy.deepcopy(gnn_cfg)
    small["sampler"]["eval_cache_max_gb"] = 1.1  # 1.0 GB x 200 / 160 does not fit
    assert cp.decide(cells, small)["measured"]["lookahead"]["val_cached_s"] is None


def test_rule_6_scales_each_kind_by_its_own_ratio(gnn_cfg):
    """review perf-1: the 'without determinism' projection divides train steps, eval passes and
    faithful steps by their own measured ratios (an eval forward pays more for deterministic
    scatter than a train step); eval and faithful fall back to the train ratio."""
    m = {
        "train_step_s": 1.0,
        "val_s": 2.0,
        "val_cached_s": 1.0,
        "lookahead": {
            "step_s": 1.0,
            "val_s_per_batch": 1.0,
            "val_cached_s": 4.0,
            "val_first_batch_s": 1.3,
        },
        "pna": {"step_s": 1.0},
        "faithful": {
            "step_s": 1.0,
            "val_s_per_batch": 1.0,
            "test_s_per_batch": 1.0,
            "build_s": 50.0,
        },
    }
    out = cp._scale_measured(m, {"train": 0.5, "eval": 0.25, "faithful": 0.1})
    assert out["train_step_s"] == 0.5 and out["val_s"] == 0.5 and out["val_cached_s"] == 0.25
    la = {"step_s": 0.5, "val_s_per_batch": 0.25, "val_cached_s": 1.0, "val_first_batch_s": 1.3}
    assert out["lookahead"] == la
    assert out["pna"] == {"step_s": 0.5}
    fa = {"step_s": 0.1, "val_s_per_batch": 0.25, "test_s_per_batch": 0.25, "build_s": 50.0}
    assert out["faithful"] == fa
    assert m["train_step_s"] == 1.0  # not mutated
    same = {"train": 1.3, "eval": 1.3, "faithful": 1.3}
    assert cp.det_ratios({"det_overhead_ratio": 1.3}) == same
    assert cp.det_ratios({"det_overhead_ratio": None}) is None
    uniform = cp.decide(make_cells("L4", extras=True, ratio=1.3), gnn_cfg)
    cells = make_cells("L4", extras=True, ratio=1.3)
    for c in cells:
        if c["group"] == "determinism":
            c.update(det_overhead_ratio_eval=3.0, det_overhead_ratio_faithful=2.0)
    d = cp.decide(cells, gnn_cfg)
    assert d["det_overhead_ratios"] == {"train": 1.3, "eval": 3.0, "faithful": 2.0}
    p, pu = d["projection_usd"], uniform["projection_usd"]
    assert p["deterministic"] == pytest.approx(pu["deterministic"])
    assert p["without_determinism"] < pu["without_determinism"]
    assert "x1.30" in cp.render_bench_md(cells, d, cp.plan_runs(gnn_cfg, decision=d))


def test_decide_faithful_gpu_rule(gnn_cfg):
    """Rule 4: the largest batch first (8192, the recalled CLI default); among GPUs that fit
    it, the cheaper faithful epoch."""
    l4 = make_cells("L4", extras=True)
    d = cp.decide(l4 + make_cells("T4", scale=1.2, f_scale=0.9), gnn_cfg)
    assert d["faithful"]["gpu"] == "T4" and d["faithful"]["batch_size"] == 8192  # cheaper
    d = cp.decide(l4 + make_cells("T4", scale=1.2, f_scale=0.5, faithful_batch=4096), gnn_cfg)
    assert d["faithful"]["gpu"] == "L4" and d["faithful"]["batch_size"] == 8192  # T4 needed less
    no_fit = make_cells("L4", extras=True, faithful_batch=None)
    d = cp.decide(no_fit, gnn_cfg)
    assert d["faithful"]["gpu"] is None and "no faithful batch fits" in d["ask_user"]


def test_decide_memory_and_record(gnn_cfg):
    cells = make_cells("L4", extras=True, peak_mib=12_000.0) + make_cells("T4", scale=1.2)
    d = cp.decide(cells, gnn_cfg)
    assert (d["memory_mib"], d["memory_limit_mib"]) == (24576, 32768)  # 1.5 x 12,000 -> 24 GiB
    keys = {
        "gpu",
        "cores",
        "memory_mib",
        "memory_limit_mib",
        "num_workers",
        "batch_size",
        "max_edges_per_step",
        "max_edges_per_eval_step",
        "eval_cache",
        "faithful",
        "bytes_per_edge",
        "precision",
        "bit_deterministic",
        "det_overhead_ratio",
        "usd_epoch",
        "epoch_s",
        "rerun_4core",
        "ask_user",
        "cells",
    }
    assert keys <= set(d)
    assert d["eval_cache"] == {"fits": True, "gb": 1.0} and d["rerun_4core"] is False
    assert d["bit_deterministic"] is False and d["det_overhead_ratio"] == 1.1
    json.dumps(d)  # decision.json
    # copying the decided values into gnn.yaml leaves no mismatch and a valid config
    c = copy.deepcopy(gnn_cfg)
    c["sampler"].update(
        batch_size=d["batch_size"],
        max_edges_per_step=d["max_edges_per_step"],
        max_edges_per_eval_step=d["max_edges_per_eval_step"],
    )
    c["protocols"]["faithful"]["batch_size"] = d["faithful"]["batch_size"]
    c["runtime"].update(
        gpu=d["gpu"], cpu=d["cores"], memory_mib=d["memory_mib"], num_workers=d["num_workers"]
    )
    check_gnn_cfg(c)
    assert decision_mismatches(c, d) == []
    assert decision_mismatches(gnn_cfg, d) != []


def test_plan_from_the_bench_decision(gnn_cfg):
    d = cp.decide(make_cells("L4", extras=True) + make_cells("T4", scale=1.3), gnn_cfg)
    rows = rows_by_run(cp.plan_runs(gnn_cfg, decision=d))
    m = d["measured"]
    steps = cp.steps_per_epoch(2_534, 3_246_387, 0.1, d["batch_size"])
    h = rows["causal"]
    assert h["source"] == "bench" and h["gpu"] == d["gpu"] and h["memory_mib"] == d["memory_mib"]
    assert h["price_usd_h"] == cp.plan_usd_h(d["gpu"], d["cores"], d["memory_mib"] / 1024)
    assert h["epoch_s"][C] == pytest.approx(steps * m["train_step_s"] + m["val_cached_s"] + 2)
    assert h["epoch_s"][K] == pytest.approx(steps * m["train_step_s"] + m["val_s"] + 3)
    assert h["startup_s"] == {C: 180.0, K: 300.0}  # build 90 s < the planning startup
    for run in ("dev", "hpo", "lookahead_s0", "lookahead_rest", "pna", "faithful"):
        assert rows[run]["source"] == "bench", run
    la = rows["lookahead_s0"]
    assert la["epoch_s"][C] == pytest.approx(
        steps * m["lookahead"]["step_s"] + 59 * m["lookahead"]["val_s_per_batch"] + 2
    )
    assert m["pna"]["step_s"] == pytest.approx(1.5 * m["train_step_s"])
    f = rows["faithful"]
    assert f["gpu"] == d["faithful"]["gpu"] and f["batch_size"] == d["faithful"]["batch_size"]
    assert f["epoch_s"][C] == pytest.approx(397 * 0.15 + 118 * 0.065 + 2)
    json.dumps(list(rows.values()))


def test_decide_on_cpu_device_cells(gnn_cfg):
    """Cells measured on a CPU device (the bench tests) record no device memory and no bytes per
    edge: no edge cap (no split, §5.7), every cell eligible, the CPU shape price."""
    cells = make_cells("L4", extras=True)
    for c in cells:
        c.update(gpu="cpu", device_bytes=None, device_name="cpu")
        c["cell_id"] = c["cell_id"].replace("L4", "cpu")
        for k in ("bytes_per_edge", "max_edges_per_step", "eval_bytes_per_edge"):
            if k in c:
                c[k] = None
        if c["group"] == "val":
            c["val_cached_s"] = None  # only the uncached pass was timed
    d = cp.decide(cells, gnn_cfg)
    assert d["gpu"] == "cpu" and d["precision"] == "highest"
    assert d["max_edges_per_step"] is None and d["max_edges_per_eval_step"] is None
    assert all(e["eligible"] for e in d["cells"])
    e = by_cell(d)[d["cell_id"]]
    assert e["epoch_s"] == pytest.approx(160 * 0.040 + 6.0 + 2.0)  # the uncached val pass
    assert cp.shape_usd_h("cpu", 8, 16) == pytest.approx(8 * 0.0473 + 16 * 0.008)
    rows = cp.plan_runs(gnn_cfg, decision=d)
    g = cp.gate(rows, gnn_cfg=gnn_cfg, spent_usd=0.0, done=(), job="dev", metered_usd=0.0)
    assert g["allowed"]
    json.dumps(d)


def test_measured_faithful_build_is_paid_per_container(gnn_cfg):
    cells = make_cells("L4", extras=True)
    for c in cells:
        if c["group"] == "faithful":
            c["build_s"] = 200.0  # longer than the planning 60 / 120 s
        if c["group"] == "build":
            c["build_s"] = 400.0  # longer than the planning 180 / 300 s
    d = cp.decide(cells, gnn_cfg)
    rows = rows_by_run(cp.plan_runs(gnn_cfg, decision=d))
    assert rows["causal"]["startup_s"] == {C: 400.0, K: 600.0}
    assert rows["faithful"]["startup_s"] == {C: 600.0, K: 900.0}  # + 200 / 300 build
    f = rows["faithful"]
    n = f["n_containers"][K]
    assert f["run_s"][K] == pytest.approx(900.0 * n + f["work_s"][K])


def test_decide_rule_6_determinism_overhead(gnn_cfg):
    """Ask the user only if the determinism overhead ALONE pushes the conservative projection
    (after cuts 1-3, nothing spent) above the cap."""
    cap = gnn_cfg["budget"]["m3_cap_usd"]
    fine = cp.decide(make_cells("L4", extras=True, ratio=1.3), gnn_cfg)
    proj = fine["projection_usd"]
    assert fine["ask_user"] is None and proj["deterministic"] > proj["without_determinism"]
    assert proj["deterministic"] == pytest.approx(
        cp._projection(cp.plan_runs(gnn_cfg, decision=fine), cap)[0]
    )
    hit = None
    for i in range(200):  # slow every GPU step down until the projection crosses the cap
        scale = 1.03**i
        d = cp.decide(make_cells("L4", extras=True, ratio=1.3, scale=scale), gnn_cfg)
        if d["projection_usd"]["deterministic"] > cap:
            hit = d
            break
    assert hit is not None
    p = hit["projection_usd"]
    assert p["without_determinism"] <= cap < p["deterministic"]
    assert "deterministic algorithms" in hit["ask_user"]
    # far beyond: the overhead is not what breaks the cap -> the gate's cuts handle it
    slow = cp.decide(make_cells("L4", extras=True, ratio=1.3, scale=scale * 3), gnn_cfg)
    assert slow["projection_usd"]["without_determinism"] > cap and slow["ask_user"] is None
    one = cp.decide(make_cells("L4", extras=True, ratio=1.0, scale=scale), gnn_cfg)
    assert one["ask_user"] is None


def test_rerun_4core_rule(gnn_cfg):
    """Rerun the winner at 4 cores / 16 GiB, 3 workers, only if its sampler wait share at 3
    workers (8 cores) is < 10% of the step."""
    assert gnn_cfg["bench"]["rerun_4core_if_wait_below"] == 0.10
    low = make_cells("L4", extras=True, wait=0.05) + make_cells("T4", scale=1.2, wait=0.5)
    r = cp.rerun_4core(low, gnn_cfg)
    assert r == {"gpu": "L4", "batch_size": 2048, "num_workers": 3, "cores": 4, "memory_mib": 16384}
    high = make_cells("L4", extras=True, wait=0.12) + make_cells("T4", scale=1.2, wait=0.01)
    assert cp.rerun_4core(high, gnn_cfg) is None  # the winner (L4) waits too long
    assert cp.rerun_4core([], gnn_cfg) is None
    # the rerun cell joins the candidates: cheaper at 4 cores -> decided
    rerun = make_cells("L4", cores=4)
    rr = [c for c in rerun if c["cell_id"] == "L4-4c-grid-b2048-w3"][0]
    rr.update(
        group="rerun",
        cell_id="L4-4c-rerun-b2048-w3",
        step_s_mean=0.044,
        val_s=6.0,
        val_cached_s=2.0,
        cache_fits=True,
        eval_batches=59,
    )
    d = cp.decide(low + [rerun[0], rr], gnn_cfg)
    assert (d["cores"], d["num_workers"], d["rerun_4core"]) == (4, 3, True)


# --- spend, rendering ---------------------------------------------------------------------------


def test_spent_from_billing_and_summaries(gnn_cfg):
    report = [
        {"description": "aml-train-gnn", "cost": "1.25"},
        {"description": "aml-train-gnn", "cost": "0.50"},
        {"description": "aml-gnn-bench", "cost": "0.70"},
        {"description": "aml-hpo-gnn", "cost": "0.30"},
        {"description": "aml-evaluate", "cost": "9.00"},  # not an M3 GNN app
    ]
    allowance = gnn_cfg["budget"]["dev_allowance_usd"]
    assert cp.spent_from_billing(report) == pytest.approx(2.75)
    # review MONEY-7: the dev apps (Linux test runner, GPU smoke test) are M3 spend too
    dev = [
        {"description": "aml-linux-runner", "cost": "0.83"},
        {"description": "aml-smoke", "cost": "0.01"},
    ]
    assert cp.spent_from_billing(report + dev) == pytest.approx(2.75 + 0.84)
    assert cp.spent_from_billing(report + dev, cp.GNN_APPS) == pytest.approx(2.75)
    assert cp.spent_from_billing({"rows": report}) == pytest.approx(2.75)
    assert cp.spent_from_billing([]) == 0.0
    sums = [
        {"gpu": "L4", "cores": 8, "memory_mib": 32768, "gpu_seconds": 3600},
        {"gpu": "NVIDIA T4", "gpu_seconds": 1800},  # planning shape: 8 cores, 32 GiB
        {"gpu_seconds": 100},  # no GPU recorded: cannot be priced
    ]
    assert cp.spent_from_summaries(sums) == pytest.approx(1.44 + 1.23 / 2)
    m = cp.measured_spend(report, sums, gnn_cfg)
    assert m["usd"] == pytest.approx(2.75 + allowance) and m["source"] == "billing"
    lag = cp.measured_spend(report[:1], sums, gnn_cfg)  # billing lags the Volume
    assert lag["usd"] == pytest.approx(1.44 + 0.615 + allowance) and lag["source"] == "volume"
    down = cp.measured_spend(None, sums, gnn_cfg, report_error="exit 1")
    assert down["source"] == "volume" and "exit 1" in down["warning"]
    assert down["usd"] == pytest.approx(1.44 + 0.615 + allowance)
    # review MONEY-8: recorded GPU calls after the report's last full hour are added to it
    hourly = [{**r, "interval_start": "2026-01-01T10:00:00"} for r in report]
    through = cp.billed_through(hourly)
    assert through == pytest.approx(cp.billed_through([hourly[0]])) and through % 3600 == 0
    call = {"gpu": "L4", "cores": 8, "memory_mib": 32768, "elapsed_s": 1800.0}
    calls = [{**call, "ended_at": through + 900.0}]
    m = cp.measured_spend(hourly, sums, gnn_cfg, calls=calls)
    assert m["lag_usd"] == pytest.approx(1.44 / 4)  # 900 s after the last interval
    assert m["usd"] == pytest.approx(2.75 + 0.36 + allowance) and m["source"] == "billing"
    assert cp.measured_spend(report, sums, gnn_cfg, calls=calls)["lag_usd"] == 0.0  # no intervals


DATE = re.compile(r"\b\d{4}-\d{2}-\d{2}\b")


def test_render_gate_and_bench_md(plan, cfg4096, gnn_cfg):
    g = cp.gate(plan, gnn_cfg=cfg4096, spent_usd=0.5, done=("bench",), job="dev", metered_usd=3.0)
    text = cp.render_gate(g)
    assert "M3 cost gate for dev: ALLOWED" in text
    for run in cp.RUN_ORDER:
        assert run in text
    assert "cuts: pna" in text and "budget pre-check" in text and not DATE.search(text)
    bad = cp.gate(plan, gnn_cfg=cfg4096, spent_usd=0.5, done=(), job="pna", metered_usd=3.0)
    assert "REFUSED" in cp.render_gate(bad) and "refused:" in cp.render_gate(bad)

    cells = make_cells("L4", extras=True) + make_cells("T4", scale=1.2)
    d = cp.decide(cells, gnn_cfg)
    md = cp.render_bench_md(cells, d, cp.plan_runs(gnn_cfg, decision=d))
    assert md.startswith("# GNN benchmark")
    for s in (
        "## Train-step cells",
        "## Faithful cells",
        "## Decision",
        "## Plan",
        "L4-8c-grid-b2048-w7",
        "faithful",
        "Total:",
    ):
        assert s in md
    assert not DATE.search(md)


def test_constants_are_consistent():
    assert set(cp.CUT_TARGET) == set(cp.CUT_ORDER)
    assert set(cp.CUT_TARGET.values()) <= set(cp.RUN_ORDER)
    assert not set(cp.CUT_TARGET.values()) & set(cp.NEVER_CUT)
    assert set(cp.NEVER_CUT) <= set(cp.RUN_ORDER)
    assert cp.RUN_ORDER.index("lookahead_s0") < cp.RUN_ORDER.index("faithful")
    assert cp.RUN_ORDER.index("faithful") < cp.RUN_ORDER.index("lookahead_rest")
    assert cp.RUN_ORDER[-1] == "pna"
