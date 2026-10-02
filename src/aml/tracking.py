"""MLflow helpers: tracking URI, experiments created once, runs keyed by `run_key`, flat logging.

The tracking URI comes from `MLFLOW_TRACKING_URI` (the Modal images set `file:///data/mlflow`).
MLflow 3.16 refuses the file store unless `MLFLOW_ALLOW_FILE_STORE=true`; the images set it and
this module sets it too, so the one store layout works in containers, tests and the UI.
The file store is used on purpose: a Modal Volume has no file locking, so SQLite is unsafe there.

Also renders `reports/cost.md` from `modal billing report --json` rows (cost logging, PLAN §5).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import warnings
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from decimal import Decimal, InvalidOperation
from numbers import Real
from typing import Any

DEFAULT_TRACKING_URI = "file:///data/mlflow"
RUN_KEY_TAG = "run_key"
TEST_TOUCH_TAG = "test_touch"

# MLflow limits (mlflow.utils.validation, 3.16).
MAX_KEY_LENGTH = 250
MAX_PARAM_VALUE_LENGTH = 6000
MAX_TAG_VALUE_LENGTH = 8000
_PARAM_BATCH = 100
_METRIC_BATCH = 1000

# Colons are valid on Linux but not on Windows; keep keys portable.
_BAD_KEY_CHARS = re.compile(r"[^\w.\- ]")


def tracking_uri() -> str:
    return os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


def _mlflow():
    """Import mlflow configured for this project's store (import is slow; done lazily)."""
    if tracking_uri().startswith("file:"):
        os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(tracking_uri())
    return mlflow


def _client():
    mlflow = _mlflow()
    return mlflow.MlflowClient(tracking_uri=tracking_uri())


def ensure_experiment(name: str) -> str:
    """Return the id of experiment `name`, creating (or restoring) it if needed. Idempotent."""
    from mlflow.exceptions import MlflowException

    client = _client()
    exp = client.get_experiment_by_name(name)
    if exp is None:
        try:
            return client.create_experiment(name)
        except MlflowException:
            # Created concurrently by someone else: fall through and read it back.
            exp = client.get_experiment_by_name(name)
            if exp is None:
                raise
    if exp.lifecycle_stage == "deleted":
        client.restore_experiment(exp.experiment_id)
    return exp.experiment_id


def find_run_for_key(experiment: str, run_key: str) -> str | None:
    """Run id of the newest active run in `experiment` tagged `run_key`, or None."""
    client = _client()
    exp = client.get_experiment_by_name(experiment)
    if exp is None:
        return None
    runs = client.search_runs(
        [exp.experiment_id],
        filter_string=f"tags.{RUN_KEY_TAG} = '{_quote(run_key)}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )
    return runs[0].info.run_id if runs else None


@contextmanager
def start_run_for_key(
    experiment: str,
    run_key: str,
    run_name: str | None = None,
    tags: Mapping[str, Any] | None = None,
) -> Iterator[Any]:
    """Start (or resume) the MLflow run tagged `run_key` in `experiment`.

    Re-running a stage with the same config resumes its run instead of creating an orphan.
    Yields the `mlflow.ActiveRun`.
    """
    mlflow = _mlflow()
    exp_id = ensure_experiment(experiment)
    run_id = find_run_for_key(experiment, run_key)
    extra = {str(k): _truncate(str(v), MAX_TAG_VALUE_LENGTH) for k, v in (tags or {}).items()}
    if run_id is not None:
        with mlflow.start_run(run_id=run_id) as run:
            if extra:
                mlflow.set_tags(extra)
            yield run
    else:
        with mlflow.start_run(
            experiment_id=exp_id,
            run_name=run_name or run_key,
            tags={RUN_KEY_TAG: run_key, **extra},
        ) as run:
            yield run


def flatten(d: Mapping[str, Any], prefix: str = "", sep: str = ".") -> dict[str, Any]:
    """Flatten nested dicts; lists/tuples stay values (callers decide how to render them)."""
    out: dict[str, Any] = {}
    for k, v in d.items():
        key = f"{prefix}{sep}{k}" if prefix else str(k)
        if isinstance(v, Mapping):
            out.update(flatten(v, key, sep))
        else:
            out[key] = v
    return out


def sanitize_key(key: str) -> str:
    """Make `key` a valid, portable MLflow param/metric key (<= 250 chars)."""
    k = _BAD_KEY_CHARS.sub("_", key).lstrip(".") or "_"
    if len(k) > MAX_KEY_LENGTH:
        digest = hashlib.sha256(key.encode()).hexdigest()[:8]
        k = f"{k[: MAX_KEY_LENGTH - 9]}~{digest}"
    return k


def log_params_flat(
    d: Mapping[str, Any], prefix: str = "", run_id: str | None = None
) -> dict[str, str]:
    """Log a nested dict as flat params (`a.b.c`), values stringified and truncated.

    Params are immutable in MLflow: when a resumed run already holds a key with a different
    value, that key is skipped with a warning instead of failing the stage. Returns what was
    logged.
    """
    mlflow = _mlflow()
    run_id = run_id or _active_run_id(mlflow)
    client = _client()
    existing = client.get_run(run_id).data.params
    params: dict[str, str] = {}
    for k, v in flatten(d, prefix).items():
        key = sanitize_key(k)
        value = _truncate(_param_str(v), MAX_PARAM_VALUE_LENGTH)
        if key in existing:
            if existing[key] != value:
                warnings.warn(
                    f"param {key!r} already logged with a different value; kept the old one",
                    stacklevel=2,
                )
            continue
        params[key] = value
    from mlflow.entities import Param

    items = [Param(k, v) for k, v in params.items()]
    for i in range(0, len(items), _PARAM_BATCH):
        client.log_batch(run_id, params=items[i : i + _PARAM_BATCH])
    return params


def log_metrics_flat(
    d: Mapping[str, Any],
    prefix: str = "",
    step: int | None = None,
    run_id: str | None = None,
    max_items: int | None = None,
) -> dict[str, float]:
    """Log every finite number in a nested dict as a metric (`a.b.c`; list items get `.i`).

    Strings, None, nan and inf are skipped. `max_items` caps the count (in key order) so a huge
    results dict cannot flood the file store. Returns what was logged.
    """
    mlflow = _mlflow()
    run_id = run_id or _active_run_id(mlflow)
    metrics: dict[str, float] = {}
    for k, v in _numeric_items(d, prefix):
        metrics[sanitize_key(k)] = v
    if max_items is not None and len(metrics) > max_items:
        warnings.warn(f"{len(metrics)} metrics; logging only the first {max_items}", stacklevel=2)
        metrics = dict(list(metrics.items())[:max_items])
    from mlflow.entities import Metric

    ts = _now_ms()
    items = [Metric(k, v, ts, step or 0) for k, v in metrics.items()]
    client = _client()
    for i in range(0, len(items), _METRIC_BATCH):
        client.log_batch(run_id, metrics=items[i : i + _METRIC_BATCH])
    return metrics


def log_trials(trials: Iterable[Mapping[str, Any]], metric: str = "trial_value") -> int:
    """Log Optuna-style trials ({number, value, state, params}) as a stepped metric + JSON."""
    mlflow = _mlflow()
    trials = list(trials)
    for i, t in enumerate(trials):
        value = _as_float(t.get("value"))
        if value is not None:
            mlflow.log_metric(metric, value, step=int(t.get("number", i)))
    mlflow.log_dict(_jsonable(trials), "trials.json")
    return len(trials)


def tag_test_touch(models: list[str], run_id: str | None = None) -> dict[str, str]:
    """Mark a run as a test-set evaluation of `models` (PLAN §4: test is touched once per model)."""
    mlflow = _mlflow()
    run_id = run_id or _active_run_id(mlflow)
    tags = {TEST_TOUCH_TAG: "true", f"{TEST_TOUCH_TAG}.models": ",".join(sorted(models))}
    tags.update({f"{TEST_TOUCH_TAG}.{sanitize_key(m)}": "true" for m in models})
    client = _client()
    for k, v in tags.items():
        client.set_tag(run_id, k, _truncate(v, MAX_TAG_VALUE_LENGTH))
    return tags


def earliest_run_start_ms(experiment_prefix: str = "aml-") -> int | None:
    """Earliest start time (ms since epoch) over all runs in experiments named `prefix*`."""
    uri = tracking_uri()
    if uri.startswith("file:"):
        return _earliest_start_from_file_store(_file_store_root(uri), experiment_prefix)
    from mlflow.entities import ViewType

    client = _client()
    exps = [
        e
        for e in client.search_experiments(view_type=ViewType.ALL)
        if e.name.startswith(experiment_prefix)
    ]
    if not exps:
        return None
    runs = client.search_runs(
        [e.experiment_id for e in exps],
        run_view_type=ViewType.ALL,
        order_by=["attributes.start_time ASC"],
        max_results=1,
    )
    return int(runs[0].info.start_time) if runs else None


def _file_store_root(uri: str):
    from pathlib import Path
    from urllib.parse import unquote, urlparse
    from urllib.request import url2pathname

    parsed = urlparse(uri)
    return Path(url2pathname(unquote(parsed.path)))


def _earliest_start_from_file_store(root, experiment_prefix: str) -> int | None:
    """Read only the small meta.yaml files of a file store.

    The MLflow client's search_runs loads every run's metrics and params, which takes minutes on
    a network Volume once the HPO stages have logged hundreds of runs.
    """
    import yaml

    best: int | None = None
    if not root.is_dir():
        return None
    for exp_dir in root.iterdir():
        meta = exp_dir / "meta.yaml"
        if not exp_dir.is_dir() or not meta.is_file():
            continue
        exp = yaml.safe_load(meta.read_text(encoding="utf-8")) or {}
        if not str(exp.get("name", "")).startswith(experiment_prefix):
            continue
        for run_meta in exp_dir.glob("*/meta.yaml"):
            run = yaml.safe_load(run_meta.read_text(encoding="utf-8")) or {}
            start = run.get("start_time")
            if isinstance(start, int) and (best is None or start < best):
                best = start
    return best


# --- cost report ------------------------------------------------------------------------------


def billing_rows(report: Any) -> list[dict[str, Any]]:
    """Normalise `modal billing report --json` output to a list of row dicts."""
    if report is None:
        return []
    if isinstance(report, Mapping):
        for key in ("rows", "items", "data", "report", "results"):
            if isinstance(report.get(key), list):
                return [r for r in report[key] if isinstance(r, Mapping)]
        return [dict(report)] if "cost" in report else []
    if isinstance(report, list):
        return [r for r in report if isinstance(r, Mapping)]
    return []


def cost_by_app(report: Any) -> dict[str, Decimal]:
    """Sum the cost per app name (`description`) over all intervals of a billing report."""
    totals: dict[str, Decimal] = {}
    for row in billing_rows(report):
        name = str(row.get("description") or row.get("object_id") or "unknown")
        cost = _as_decimal(row.get("cost"))
        if cost is not None:
            totals[name] = totals.get(name, Decimal(0)) + cost
    return totals


def render_cost_report(report: Any, error: str | None = None, app_prefix: str = "aml-") -> str:
    """Markdown for `reports/cost.md`. `report` None means the billing CLI was unavailable.

    No dates are written (project rule): the window is described, not printed.
    """
    lines = [
        "# Compute cost (Modal)",
        "",
        "Source: `modal billing report --json`, hourly resolution, from the start of the day of",
        "this project's earliest MLflow run until the last complete hour before the report was",
        "made (so the evaluation app that writes this file is only partly included).",
        "Costs are metered usage before credits.",
        "",
    ]
    if report is None:
        lines += [
            "**The billing report was unavailable**, so no costs are listed.",
            f"Reason: {error or 'unknown'}",
            "",
            "Re-run `make eval` or check `uv run modal billing report --for today --json`.",
        ]
        return "\n".join(lines) + "\n"

    totals = cost_by_app(report)
    ours = {k: v for k, v in totals.items() if k.startswith(app_prefix)}
    other = sum((v for k, v in totals.items() if not k.startswith(app_prefix)), Decimal(0))
    total = sum(ours.values(), Decimal(0))
    lines += ["| App | Cost (USD) |", "| --- | ---: |"]
    lines += [f"| `{k}` | {_usd(v)} |" for k, v in sorted(ours.items())]
    lines += [f"| **Total ({app_prefix}\\*)** | **{_usd(total)}** |", ""]
    if not ours:
        lines += ["No project app has billed a complete interval in this window yet.", ""]
    if other:
        lines += [f"Other apps in the workspace over the same window: {_usd(other)} USD.", ""]
    lines += ["Budget caps per milestone: PLAN.md §2.3.", ""]
    if error:
        lines += [f"Note: {error}", ""]
    return "\n".join(lines)


# --- internals --------------------------------------------------------------------------------


def _active_run_id(mlflow: Any) -> str:
    run = mlflow.active_run()
    if run is None:
        raise RuntimeError("no active MLflow run: use start_run_for_key(...) or pass run_id")
    return run.info.run_id


def _numeric_items(d: Any, prefix: str) -> Iterator[tuple[str, float]]:
    if isinstance(d, Mapping):
        for k, v in d.items():
            yield from _numeric_items(v, f"{prefix}.{k}" if prefix else str(k))
    elif isinstance(d, list | tuple):
        for i, v in enumerate(d):
            yield from _numeric_items(v, f"{prefix}.{i}" if prefix else str(i))
    else:
        value = _as_float(d)
        if value is not None and prefix:
            yield prefix, value


def _as_float(v: Any) -> float | None:
    """Finite float for numbers (incl. numpy scalars and bools), else None."""
    if isinstance(v, str | bytes) or v is None:
        return None
    if hasattr(v, "item") and getattr(v, "shape", None) == ():
        v = v.item()
    if isinstance(v, bool):
        return float(v)
    if isinstance(v, Real | Decimal):
        f = float(v)
        return f if math.isfinite(f) else None
    return None


def _as_decimal(v: Any) -> Decimal | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        d = Decimal(str(v))
    except (InvalidOperation, ValueError):
        return None
    return d if d.is_finite() else None


def _param_str(v: Any) -> str:
    if isinstance(v, str):
        return v
    if isinstance(v, list | tuple | Mapping):
        return json.dumps(_jsonable(v), sort_keys=True, separators=(",", ":"))
    return str(v)


def _jsonable(obj: Any) -> Any:
    return json.loads(json.dumps(obj, default=_json_default))


def _json_default(o: Any) -> Any:
    if hasattr(o, "item") and callable(o.item) and getattr(o, "shape", None) == ():
        return o.item()
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def _truncate(s: str, n: int) -> str:
    return s if len(s) <= n else s[: n - 3] + "..."


def _quote(s: str) -> str:
    return s.replace("\\", "\\\\").replace("'", "\\'")


def _usd(d: Decimal) -> str:
    return f"{d.quantize(Decimal('0.01')):,}" if d >= Decimal("0.01") or d == 0 else f"{d:.4f}"


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)
