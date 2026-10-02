"""aml.tracking against a temporary file:// MLflow store."""

from __future__ import annotations

import math
import re

import numpy as np
import pytest

from aml import tracking


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_TRACKING_URI", (tmp_path / "mlruns").as_uri())
    monkeypatch.setenv("MLFLOW_ALLOW_FILE_STORE", "true")
    yield tmp_path / "mlruns"
    import mlflow

    while mlflow.active_run() is not None:
        mlflow.end_run()


def _run(run_id):
    return tracking._client().get_run(run_id)


def test_ensure_experiment_is_idempotent(store):
    a = tracking.ensure_experiment("aml-test")
    b = tracking.ensure_experiment("aml-test")
    assert a == b
    assert tracking.ensure_experiment("aml-other") != a
    assert store.exists()


def test_start_run_for_key_resumes(store):
    with tracking.start_run_for_key("aml-test", "stage-abc", tags={"extra": 1}) as run:
        first = run.info.run_id
        tracking.log_params_flat({"x": 1})
    with tracking.start_run_for_key("aml-test", "stage-abc") as run:
        assert run.info.run_id == first
    with tracking.start_run_for_key("aml-test", "stage-other") as run:
        assert run.info.run_id != first
    r = _run(first)
    assert r.data.tags[tracking.RUN_KEY_TAG] == "stage-abc"
    assert r.data.tags["extra"] == "1"
    assert r.info.status == "FINISHED"
    assert tracking.find_run_for_key("aml-test", "stage-abc") == first
    assert tracking.find_run_for_key("aml-test", "missing") is None
    assert tracking.find_run_for_key("aml-missing", "stage-abc") is None


def test_run_is_marked_failed_on_error(store):
    with pytest.raises(ValueError), tracking.start_run_for_key("aml-test", "k-fail") as run:
        rid = run.info.run_id
        raise ValueError("stage crashed")
    assert _run(rid).info.status == "FAILED"


def test_log_params_flat(store):
    params = {
        "a": {"b": 1, "c": {"d": "x"}},
        "lst": [1, 2.5],
        "none": None,
        "long": "y" * 7000,
        "bad key:with*chars": True,
    }
    with tracking.start_run_for_key("aml-test", "p") as run:
        logged = tracking.log_params_flat(params, prefix="cfg")
        rid = run.info.run_id
    got = _run(rid).data.params
    assert got["cfg.a.b"] == "1" and got["cfg.a.c.d"] == "x"
    assert got["cfg.lst"] == "[1,2.5]"
    assert got["cfg.none"] == "None"
    assert len(got["cfg.long"]) == tracking.MAX_PARAM_VALUE_LENGTH
    assert got["cfg.bad key_with_chars"] == "True"
    assert logged == got


def test_log_params_on_resume_skips_changed_values(store):
    with tracking.start_run_for_key("aml-test", "p2"):
        tracking.log_params_flat({"a": 1, "b": 2})
    with tracking.start_run_for_key("aml-test", "p2") as run:
        with pytest.warns(UserWarning, match="different value"):
            logged = tracking.log_params_flat({"a": 1, "b": 3, "c": 4})
        rid = run.info.run_id
    assert logged == {"c": "4"}
    assert _run(rid).data.params == {"a": "1", "b": "2", "c": "4"}


def test_many_params_are_batched(store):
    with tracking.start_run_for_key("aml-test", "many") as run:
        tracking.log_params_flat({f"k{i}": i for i in range(250)})
        rid = run.info.run_id
    assert len(_run(rid).data.params) == 250


def test_log_metrics_flat_skips_nan_and_non_numbers(store):
    d = {
        "a": 1,
        "b": {"c": 2.5, "nan": float("nan"), "inf": math.inf, "s": "text", "none": None},
        "per_seed": [0.1, 0.2],
        "np": np.float64(0.75),
        "np_nan": np.float32("nan"),
        "flag": True,
        "rows": [{"ap": 0.3}],
    }
    with tracking.start_run_for_key("aml-test", "m") as run:
        logged = tracking.log_metrics_flat(d, prefix="val")
        rid = run.info.run_id
    expected = {
        "val.a": 1.0,
        "val.b.c": 2.5,
        "val.per_seed.0": 0.1,
        "val.per_seed.1": 0.2,
        "val.np": 0.75,
        "val.flag": 1.0,
        "val.rows.0.ap": 0.3,
    }
    assert logged == pytest.approx(expected)
    assert _run(rid).data.metrics == pytest.approx(expected)


def test_log_metrics_flat_cap(store):
    with tracking.start_run_for_key("aml-test", "cap") as run:
        with pytest.warns(UserWarning):
            logged = tracking.log_metrics_flat({f"m{i}": i for i in range(10)}, max_items=4)
        rid = run.info.run_id
    assert len(logged) == 4 and len(_run(rid).data.metrics) == 4


def test_logging_needs_an_active_run(store):
    with pytest.raises(RuntimeError, match="no active MLflow run"):
        tracking.log_metrics_flat({"a": 1})


def test_tag_test_touch_and_trials(store):
    with tracking.start_run_for_key("aml-evaluate", "e") as run:
        tracking.tag_test_touch(["lgbm_tx", "gnn"])
        n = tracking.log_trials(
            [{"number": 0, "value": 0.2, "state": "COMPLETE"}, {"number": 1, "value": None}]
        )
        rid = run.info.run_id
    r = _run(rid)
    assert r.data.tags[tracking.TEST_TOUCH_TAG] == "true"
    assert r.data.tags["test_touch.models"] == "gnn,lgbm_tx"
    assert r.data.tags["test_touch.lgbm_tx"] == "true"
    assert n == 2 and r.data.metrics["trial_value"] == pytest.approx(0.2)


def test_earliest_run_start(store):
    assert tracking.earliest_run_start_ms() is None
    with tracking.start_run_for_key("not-this-project", "k0"):  # earliest, but another project
        pass
    with tracking.start_run_for_key("aml-a", "k1") as r1:
        pass
    with tracking.start_run_for_key("aml-b", "k2"):
        pass
    with tracking.start_run_for_key("not-this-project", "k3"):
        pass
    assert tracking.earliest_run_start_ms("aml-") == r1.info.start_time


def test_earliest_run_start_file_store_matches_mlflow_search(store):
    # The fast path reads meta.yaml files only; it must agree with MLflow's own search.
    from mlflow.entities import ViewType

    for exp, key in (("aml-x", "a"), ("aml-y", "b"), ("aml-x", "c")):
        with tracking.start_run_for_key(exp, key):
            pass
    client = tracking._client()
    exps = [e.experiment_id for e in client.search_experiments() if e.name.startswith("aml-")]
    runs = client.search_runs(
        exps, run_view_type=ViewType.ALL, order_by=["attributes.start_time ASC"], max_results=1
    )
    assert tracking.tracking_uri().startswith("file:")
    assert tracking.earliest_run_start_ms("aml-") == runs[0].info.start_time


def test_sanitize_key_limits():
    long_a, long_b = "a" * 300 + "x", "a" * 300 + "y"
    ka, kb = tracking.sanitize_key(long_a), tracking.sanitize_key(long_b)
    assert len(ka) <= tracking.MAX_KEY_LENGTH and ka != kb
    assert tracking.sanitize_key("../x:y") == "_x_y"
    assert tracking.flatten({"a": {"b": {"c": 1}}, "d": 2}) == {"a.b.c": 1, "d": 2}


def test_render_cost_report():
    rows = [
        {"description": "aml-rules", "cost": "0.12", "interval_start": "2030-01-01T00:00:00Z"},
        {"description": "aml-rules", "cost": "0.03", "interval_start": "2030-01-01T01:00:00Z"},
        {"description": "aml-smoke", "cost": 0.002},
        {"description": "someone-else", "cost": "1.00"},
    ]
    md = tracking.render_cost_report(rows)
    assert "| `aml-rules` | 0.15 |" in md
    assert "| `aml-smoke` | 0.0020 |" in md
    assert "**0.15**" in md  # 0.152 rounds to 0.15
    assert "Other apps" in md and "1.00" in md
    assert not re.search(r"\d{4}-\d{2}-\d{2}", md), "no calendar dates in committed reports"
    assert (
        tracking.cost_by_app({"items": rows})["aml-rules"]
        == tracking.cost_by_app(rows)["aml-rules"]
    )

    md = tracking.render_cost_report(None, error="billing CLI exited 1")
    assert "unavailable" in md and "billing CLI exited 1" in md
