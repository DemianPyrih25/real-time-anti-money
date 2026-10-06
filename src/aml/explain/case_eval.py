"""The case_eval stage (M6 spec §5): case packs over a whole period, full-period parity, and the
typology matchers fitted or tuned on val and measured on test. `modal_jobs.case_eval` calls
`run_case_eval`; the tests call the same functions on the synthetic fixture.

1. The champion: the serving bundle (`Champion.from_bundle`) when its metadata.keys.export is the
   current export key and it was exported from this feature table, else the same parts computed
   the way `bundle.run_export` computes them (`champion_parts`; val_late labels only).
2. The period's boundary snapshot (val: val_boundary.snap at the start of day 7, test:
   test_boundary.snap at the start of day 9) is restored and the period's rows stream in rank
   order through `StreamScorer` from a `ParquetSource` over transactions.parquet. The on_alert
   hook builds every alert's case pack (strict: a failure stops the run).
3. Full-period parity: every event's model inputs, severities, trunc flags, rule flags, score and
   alerts against the offline references (`bundle.build_references` + `bundle.compare_outputs`,
   chunk by chunk), plus the scorer's own alert decisions and rule hits; then the final state
   digest (test: the replay's final_state_digest; val: the test boundary snapshot's, after
   advancing to its clock). Every count must be 0.
4. Eval only, after the stream: labels.parquet (is_laundering, typology) joined on the alerted
   rows (val also on the period's positives, for recall by typology). The matcher rows are the
   true-positive alerts: (the pack's typology features, the labelled typology; OTHER for a
   positive outside the Patterns file).
   - val: `fit_tree` (the primary matcher) -> typology_tree_val.json, to copy to
     configs/typology_tree.json; `tune` (the decision-list baseline) -> typology_match_val.json,
     which also holds the tree's train and cross-validated accuracy side by side with the
     list's; error analysis (recall per typology at the alert tag, the 10 highest-scoring false
     positives with their narratives) -> case_eval_val.json.
   - test: the frozen tree (configs/typology_tree.json, required under typology_model: tree; the
     packs are built with it) and the frozen decision-list thresholds (recomputed from the
     stored features) on the true-positive alerts of the primary and full views, with the
     majority baseline, per-typology recall and the confusion matrices -> case_eval.json; 3
     example packs (true positives, different typologies) -> cases/case_<row_id>.json.
"""

from __future__ import annotations

import dataclasses
import json
import time
from array import array
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, NamedTuple

import lightgbm as lgb
import numpy as np
import polars as pl

from aml.explain.casepack import (
    TREE_FILE,
    build_case_pack,
    check_config,
    frozen_tree,
    rules_matcher,
    typology_matcher,
)
from aml.explain.typology_match import (
    OTHER,
    TYPOLOGIES,
    MatcherParams,
    TreeMatcher,
    accuracy,
    fit_tree,
    tune,
)
from aml.features.build import load_spec, snapshot_file
from aml.features.engine import Engine
from aml.features.spec import FEATURES_DIGEST, SEVERITY_COLUMNS, SNAPSHOTS_DIR, SUMMARY_FILE
from aml.io import read_json, write_text_atomic
from aml.paths import DataPaths
from aml.rules.sql_baseline import SCENARIOS
from aml.serving import bundle
from aml.serving.scorer import Champion, ParquetSource, RefusalError, StreamScorer

if TYPE_CHECKING:
    from aml.serving.scorer import Event, Scored, Timing


class Period(NamedTuple):
    name: str
    splits: tuple[str, ...]  # whole splits, rank-contiguous
    snapshot_split: str  # its boundary snapshot (features.snapshots.boundaries)
    next_split: str | None  # the split whose boundary snapshot ends it (None: end of the data)


PERIODS = {
    "val": Period("val", ("val_early", "val_late"), "val_early", "test"),
    "test": Period("test", ("test",), "test", None),
}
VAL_REPORT = "typology_match_val.json"
TREE_REPORT = "typology_tree_val.json"  # copied to configs/typology_tree.json (frozen)
VAL_ANALYSIS = "case_eval_val.json"
SUMMARY_ACC = ("n", "correct", "accuracy", "macro_recall")
TEST_REPORT = "case_eval.json"
CASES_DIR = "cases"
VIEWS = ("primary", "full")  # data.yaml test_views measured on test
N_FALSE_POSITIVES = 10
N_EXAMPLES = 3
PARITY_CHUNK_ROWS = 200_000  # references built and compared per chunk: bounded memory
PROGRESS_EVERY = 100_000
EXTRA_CHECKS = ("alert_flags", "rules_fired")  # the scorer's own decisions vs the references
PARITY_CHECKS = (*bundle.CHECKS, *EXTRA_CHECKS)
TEST_TOUCH_MODELS = ("lgbm_graph", "typology_match")
N_SEV = len(SEVERITY_COLUMNS)


# --- the period and the champion ----------------------------------------------------------------


def period_rows(transactions: Path, period: str) -> dict[str, Any]:
    """The period's rows in transactions.parquet: count, first/last rank and day range. They must
    be rank-contiguous (ValueError otherwise, or when there are none)."""
    p = PERIODS[period]
    st = (
        pl.scan_parquet(transactions)
        .filter(pl.col("split").is_in(list(p.splits)))
        .select(
            pl.len().alias("rows"),
            pl.col("rank").min().alias("first_rank"),
            pl.col("rank").max().alias("last_rank"),
            pl.col("day").min().alias("first_day"),
            pl.col("day").max().alias("last_day"),
        )
        .collect()
        .row(0, named=True)
    )
    n = int(st["rows"])
    if n == 0:
        raise ValueError(f"no {period} rows (splits {list(p.splits)}) in {transactions}")
    r0, r1 = int(st["first_rank"]), int(st["last_rank"])
    if r1 - r0 + 1 != n:
        raise ValueError(f"the {period} rows are not rank-contiguous: {n} rows in [{r0}, {r1}]")
    return {
        "period": period,
        "splits": list(p.splits),
        "rows": n,
        "first_rank": r0,
        "last_rank": r1,
        "days": [int(st["first_day"]), int(st["last_day"])],
    }


def champion_parts(
    paths: DataPaths, *, features_dir: Path, rules_dir: Path, graph_dir: Path
) -> dict[str, Any]:
    """feature_spec.json, the booster path, thresholds.json and calibration.json of the seed-0
    lgbm_graph champion, computed as `bundle.run_export` computes them (val_late labels only)."""
    features_dir, rules_dir, graph_dir = Path(features_dir), Path(rules_dir), Path(graph_dir)
    spec = load_spec(features_dir)
    booster_path = graph_dir / f"booster_s{bundle.SEED}.txt"
    names = list(lgb.Booster(model_file=str(booster_path)).feature_name())
    spec.assert_model_inputs(names)
    listed_path = graph_dir / "feature_names.json"
    if listed_path.exists():
        listed = read_json(listed_path)
        if isinstance(listed, list) and list(listed) != names:
            raise ValueError("feature_names.json differs from the seed-0 booster's features")
    rules_thr = read_json(rules_dir / "thresholds.json")
    tags, head = bundle._rate_tags(rules_thr)
    flags = pl.read_parquet(rules_dir / "flags.parquet")
    scores = pl.read_parquet(
        graph_dir / "scores.parquet", columns=["row_id", "split", bundle.SCORE]
    )
    thresholds, calibration = bundle._model_thresholds(paths, scores, flags, tags, head)
    thresholds["rules"] = {"headline_rate_tag": head, "thresholds": rules_thr["thresholds"]}
    gate_path, summary_path = graph_dir / "gate.json", graph_dir / "summary.json"
    doc = spec.to_json()
    doc["model"] = {
        "seed": bundle.SEED,
        "booster": bundle.BOOSTER,
        "feature_names": names,
        "model_index": list(spec.model_index(names)),
        "categorical": [n for n in names if spec.feature(n).categorical],
        "gated": read_json(gate_path).get("kept") if gate_path.exists() else None,
        "champion": bundle._champion(read_json(summary_path)) if summary_path.exists() else None,
        "float32_cast": bundle.FLOAT32_CAST,
        "predict": bundle.PREDICT,
    }
    return {
        "feature_spec": doc,
        "booster": booster_path,
        "thresholds": thresholds,
        "calibration": calibration,
    }


def bundle_mismatch(
    bundle_dir: Path,
    export_key: str | None,
    features_dir: Path,
    data_version: str | None = None,
) -> str | None:
    """Why the serving bundle cannot stand for the current champion (None: it can): missing,
    another export key, another feature spec or parts, or other prepared data."""
    meta_path = Path(bundle_dir) / bundle.METADATA_FILE
    if not meta_path.is_file():
        return f"no serving bundle at {bundle_dir}"
    meta = read_json(meta_path)
    found = (meta.get("keys") or {}).get("export")
    if export_key is None or found != export_key:
        return f"the bundle's export key {found!r} is not the current {export_key!r}"
    if meta.get("spec_hash") != load_spec(features_dir).spec_hash():
        return "the bundle's feature spec differs from the feature table's"
    summary_path = Path(features_dir) / SUMMARY_FILE
    digest = read_json(summary_path).get(FEATURES_DIGEST) if summary_path.exists() else None
    if digest is not None and meta.get(FEATURES_DIGEST) != digest:
        return "the bundle was exported from other feature parts"
    if data_version is not None and meta.get("data_version") != data_version:
        return f"the bundle's prepared data {meta.get('data_version')!r} is not {data_version!r}"
    return None


def load_champion(
    paths: DataPaths,
    export_key: str | None,
    *,
    features_dir: Path,
    rules_dir: Path,
    graph_dir: Path,
    bundle_dir: Path | None = None,
    alert_tag: str = "headline",
    data_version: str | None = None,
) -> tuple[Champion, dict[str, Any] | None, dict[str, Any]]:
    """(champion, calibration.json, where they came from). The bundle (default
    paths.serving_dir) when `bundle_mismatch` finds nothing, else `champion_parts`."""
    bdir = Path(paths.serving_dir if bundle_dir is None else bundle_dir)
    reason = bundle_mismatch(bdir, export_key, features_dir, data_version)
    if reason is None:
        try:
            ch = Champion.from_bundle(bdir, alert_tag=alert_tag)
        except RefusalError as e:
            reason = f"the bundle was refused: {e}"
        else:
            cal = read_json(bdir / bundle.CALIBRATION)
            return ch, cal, {"source": "bundle", "bundle_dir": str(bdir)}
    parts = champion_parts(
        paths, features_dir=features_dir, rules_dir=rules_dir, graph_dir=graph_dir
    )
    ch = Champion.from_parts(
        parts["feature_spec"],
        parts["booster"],
        parts["thresholds"],
        export_key,
        alert_tag=alert_tag,
    )
    return ch, parts["calibration"], {"source": "computed", "reason": reason}


# --- the stream ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class References:
    """The offline outputs of a period: the feature-table parts, the lgbm_graph stage's scores
    (row_id, score_s0) and the rules stage's flags."""

    features_dir: Path
    scores: pl.DataFrame
    flags: pl.DataFrame


def load_references(
    features_dir: Path, rules_dir: Path, graph_dir: Path, period: str
) -> References:
    """The period's offline scores and rule flags (the parts are read chunk by chunk later)."""
    splits = list(PERIODS[period].splits)
    scores = (
        pl.scan_parquet(Path(graph_dir) / "scores.parquet")
        .filter(pl.col("split").is_in(splits))
        .select("row_id", bundle.SCORE)
        .collect()
    )
    flags = (
        pl.scan_parquet(Path(rules_dir) / "flags.parquet")
        .filter(pl.col("split").is_in(splits))
        .collect()
    )
    return References(Path(features_dir), scores, flags)


ChunkCheck = Callable[[dict[str, np.ndarray]], tuple[dict[str, int], int]]


def reference_check(ch: Champion, refs: References) -> ChunkCheck:
    """check(chunk) -> (mismatching rows per PARITY_CHECKS, reference alerts at the alert tag) for
    a rank-contiguous chunk of the stream's outputs (see ParityRecorder)."""
    names = list(ch.names)
    thr, tag = ch.thresholds, ch.alert_tag
    rules_thr = thr["rules"]["thresholds"]

    def check(chunk: dict[str, np.ndarray]) -> tuple[dict[str, int], int]:
        keys = pl.DataFrame({"row_id": chunk["row_id"], "rank": chunk["rank"]})
        feats, sc, alerts = bundle.build_references(
            refs.features_dir, names, keys, refs.scores, refs.flags, thr, rules_thr
        )
        bad = bundle.compare_outputs(
            x32=chunk["x32"],
            sev=chunk["sev"],
            trunc=chunk["trunc"],
            p=chunk["p"],
            row_id=keys["row_id"],
            ref_features=feats,
            ref_scores=sc,
            ref_alerts=alerts,
            names=names,
            thresholds=thr,
        )
        ref_alert = alerts[f"alert_{tag}"].to_numpy().astype(bool)
        bad["alert_flags"] = int((ref_alert != chunk["alert"]).sum())
        ref_bits = np.zeros(keys.height, dtype=np.int64)
        for j, s in enumerate(SCENARIOS):
            ref_bits |= alerts[f"fired_{s}"].to_numpy().astype(np.int64) << j
        bad["rules_fired"] = int((ref_bits != chunk["rules"]).sum())
        return bad, int(ref_alert.sum())

    return check


class ParityRecorder:
    """Untimed scorer observer: every event's outputs, compared with the offline references every
    `chunk_rows` events by `check` (bounded memory); `finish()` compares the rest and returns the
    parity document (rows, chunks, mismatches per PARITY_CHECKS, stream vs reference alerts)."""

    def __init__(self, ch: Champion, check: ChunkCheck, chunk_rows: int = PARITY_CHUNK_ROWS):
        if chunk_rows < 1:
            raise ValueError(f"chunk_rows must be >= 1, got {chunk_rows}")
        spec = ch.spec
        self.check = check
        self.chunk_rows = int(chunk_rows)
        self.n_inputs = len(ch.names)
        self.i_sev = spec.i_sev
        self.i_trunc = (spec.i_rule_trunc, spec.i_cyc_trunc, spec.i_sg_trunc)
        self.bits = {s: 1 << j for j, s in enumerate(SCENARIOS)}
        self.mismatches = dict.fromkeys(PARITY_CHECKS, 0)
        self.rows = 0
        self.chunks = 0
        self.stream_alerts = 0
        self.reference_alerts = 0
        self.seconds = 0.0
        self.next_rank: int | None = None
        self._clear()

    def _clear(self) -> None:
        self._rank, self._row_id = array("q"), array("q")
        self._x, self._alert = bytearray(), bytearray()
        self._sev, self._p = array("d"), array("d")
        self._trunc, self._rules = array("q"), array("q")

    def observe(self, ev: Event, s: Scored, t: Timing) -> None:
        if self.next_rank is not None and s.rank != self.next_rank:
            raise AssertionError(f"parity rows must be rank-contiguous: rank {s.rank}")
        self.next_rank = s.rank + 1
        row = s.row
        self._rank.append(s.rank)
        self._row_id.append(s.row_id)
        self._x += s.x.tobytes()
        self._sev.extend(row[self.i_sev : self.i_sev + N_SEV])
        self._trunc.extend(int(row[i]) for i in self.i_trunc)
        self._p.append(s.score)
        self._alert.append(1 if s.alert else 0)
        self._rules.append(sum(self.bits[r] for r in s.rules))
        if len(self._p) >= self.chunk_rows:
            self._flush()

    def _flush(self) -> None:
        m = len(self._p)
        if not m:
            return
        t0 = time.perf_counter()
        chunk = {
            "rank": np.array(self._rank, dtype=np.int64),
            "row_id": np.array(self._row_id, dtype=np.int64),
            "x32": np.frombuffer(bytes(self._x), dtype=np.float32).reshape(m, self.n_inputs),
            "sev": np.array(self._sev, dtype=np.float64).reshape(m, N_SEV),
            "trunc": np.array(self._trunc, dtype=np.int64).reshape(m, 3),
            "p": np.array(self._p, dtype=np.float64),
            "alert": np.frombuffer(bytes(self._alert), dtype=np.uint8).astype(bool),
            "rules": np.array(self._rules, dtype=np.int64),
        }
        bad, ref_alerts = self.check(chunk)
        for k in PARITY_CHECKS:
            self.mismatches[k] += int(bad[k])
        self.rows += m
        self.chunks += 1
        self.stream_alerts += int(chunk["alert"].sum())
        self.reference_alerts += int(ref_alerts)
        self._clear()
        self.seconds += time.perf_counter() - t0

    def finish(self) -> dict[str, Any]:
        self._flush()
        return {
            "rows": self.rows,
            "chunks": self.chunks,
            "mismatches": dict(self.mismatches),
            "alerts": {"stream": self.stream_alerts, "reference": self.reference_alerts},
            "seconds": self.seconds,
        }


class PackHook:
    """The on_alert hook of case_eval: the alert's case pack (strict: a failure stops the run),
    kept compact (id, rank, day, score, matcher label, model and inputs, the pack as JSON text).
    The packs' matcher is `typology_matcher` (ValueError up front when the frozen tree does not
    fit the champion). The stream loop sets `fields` to the current event's INPUT_COLUMNS tuple
    before each step, so the packs carry banks and the paid amount."""

    def __init__(
        self, ch: Champion, cfg: Mapping[str, Any], calibration: Mapping[str, Any] | None
    ) -> None:
        self.ch, self.cfg = ch, check_config(cfg)
        self.matcher = typology_matcher(self.cfg, ch)
        self.calibration = None if calibration is None else dict(calibration)
        self.fields: tuple | None = None
        self.alerts: list[dict[str, Any]] = []

    def __call__(self, s: Scored, eng: Engine) -> None:
        pack = build_case_pack(
            s,
            eng,
            self.ch,
            cfg=self.cfg,
            calibration=self.calibration,
            fields=self.fields,
            matcher=self.matcher,
        )
        typ = pack["why"]["typology"]
        self.alerts.append(
            {
                "id": pack["id"],
                "rank": pack["rank"],
                "day": pack["when"]["day"],
                "score": pack["why"]["score"],
                "label": typ["label"],
                "model": typ["model"],
                "features": typ["features"],
                "pack": json.dumps(pack, allow_nan=False, separators=(",", ":")),
            }
        )


@dataclass
class Streamed:
    eng: Engine
    events: int
    alerts: list[dict[str, Any]]  # PackHook records, stream order
    parity: dict[str, Any] | None  # ParityRecorder.finish(), None without references
    final_state_digest: str  # after the scorer's final flush
    seconds: float


def stream_period(
    ch: Champion,
    cfg: Mapping[str, Any],
    *,
    snapshot: Path,
    transactions: Path,
    rows: Mapping[str, Any],
    calibration: Mapping[str, Any] | None = None,
    references: References | None = None,
    chunk_rows: int = PARITY_CHUNK_ROWS,
    log: Callable[[str], None] | None = None,
    progress_every: int = PROGRESS_EVERY,
) -> Streamed:
    """Restore `snapshot` and stream the period `rows` (period_rows) through StreamScorer with
    the case-pack hook and, given `references`, the parity recorder. The snapshot must resume
    at the period's first rank with next_offset 0; the scorer's final flush runs after the last
    event."""
    n = int(rows["rows"])
    if n < 1:
        raise ValueError("a period needs at least one row")
    eng, header = Engine.restore(Path(snapshot), ch.spec)
    start = header.get("next_offset")
    if start != 0 or header.get("next_rank") != rows["first_rank"]:
        raise ValueError(
            f"{Path(snapshot).name} resumes at rank {header.get('next_rank')} offset {start}, "
            f"the period starts at rank {rows['first_rank']} offset 0"
        )
    hook = PackHook(ch, cfg, calibration)
    recorder = None
    if references is not None:
        recorder = ParityRecorder(ch, reference_check(ch, references), chunk_rows)
    scorer = StreamScorer(
        ch,
        eng,
        header,
        end_offset=n,
        observers=[recorder] if recorder is not None else (),
        on_alert=[hook],
    )
    source = ParquetSource(Path(transactions), int(header["next_rank"]) - start, n)
    t0 = time.perf_counter()
    source.start(scorer.start_offset)
    try:
        while not scorer.finished:
            ev = source.next(0.0)
            if ev is None:
                raise RuntimeError(
                    f"{transactions} ended at offset {scorer.expected_offset} of {n}"
                )
            hook.fields = ev.fields
            scorer.step(ev)
            done = scorer.progress.events
            if log is not None and (done % progress_every == 0 or scorer.finished):
                log(
                    f"{rows['period']}: {done:,}/{n:,} events, {len(hook.alerts):,} alerts, "
                    f"{time.perf_counter() - t0:,.0f} s"
                )
    finally:
        source.close()
    parity = recorder.finish() if recorder is not None else None
    return Streamed(
        eng=eng,
        events=scorer.progress.events,
        alerts=hook.alerts,
        parity=parity,
        final_state_digest=eng.state_digest(),
        seconds=time.perf_counter() - t0,
    )


def check_digest(
    eng: Engine, period: str, *, features_dir: Path, rows: Mapping[str, Any]
) -> dict[str, Any]:
    """The final state against the replay's. test (the end of the data): the build summary's
    final_state_digest. val: advance to the next boundary snapshot's clock (`advance(b)` after
    the final flush equals the replay's day-boundary advance) and compare with its digest."""
    out: dict[str, Any] = {"final_state_digest": eng.state_digest(), "clock": eng.clock}
    nxt = PERIODS[period].next_split
    if nxt is None:
        path = Path(features_dir) / SUMMARY_FILE
        expected = read_json(path).get("final_state_digest") if path.exists() else None
        out.update(against=f"final_state_digest in {SUMMARY_FILE}", expected=expected)
        out["ok"] = None if expected is None else out["final_state_digest"] == expected
        return out
    snap = Path(features_dir) / SNAPSHOTS_DIR / snapshot_file(nxt)
    h = bundle.read_snapshot_header(snap)
    out.update(against=snap.name, expected=h.get("state_digest"), boundary_clock=h.get("clock"))
    if h.get("next_rank") != int(rows["last_rank"]) + 1:
        out["ok"] = False
        out["reason"] = f"{snap.name} next_rank {h.get('next_rank')} != {rows['last_rank'] + 1}"
        return out
    clock = h.get("clock")
    if isinstance(clock, int) and eng.clock is not None and clock > eng.clock:
        eng.advance(clock)
    out["boundary_state_digest"] = eng.state_digest()
    out["ok"] = out["boundary_state_digest"] == h.get("state_digest")
    return out


def parity_ok(parity: Mapping[str, Any] | None, n_rows: int) -> bool:
    """Every event checked, no mismatch, the same alert count and no failed digest check."""
    if not parity:
        return False
    alerts = parity["alerts"]
    return (
        parity["rows"] == n_rows
        and not any(parity["mismatches"].values())
        and alerts["stream"] == alerts["reference"]
        and (parity.get("digest") or {}).get("ok") is not False
    )


# --- labels (eval only, after the stream) --------------------------------------------------------


def join_labels(labels_path: Path, alerts: list[dict[str, Any]]) -> None:
    """Add `y` (is_laundering) and `truth` (the typology of a positive, OTHER when it has none;
    None for a negative) to every alert record. Only the alerted rows' labels are read."""
    if not alerts:
        return
    keys = pl.DataFrame({"row_id": pl.Series("row_id", [a["id"] for a in alerts], pl.Int64)})
    lab = pl.scan_parquet(labels_path).select("row_id", "is_laundering", "typology")
    got = keys.lazy().join(lab, on="row_id", how="left", maintain_order="left").collect()
    missing = got["is_laundering"].null_count()
    if missing:
        raise ValueError(f"{missing} alerted rows have no label in {labels_path}")
    ys, ts = got["is_laundering"].to_list(), got["typology"].to_list()
    for a, y, t in zip(alerts, ys, ts, strict=True):
        a["y"] = int(y)
        a["truth"] = (t or OTHER) if a["y"] else None


def period_positives(paths: DataPaths, rows: Mapping[str, Any]) -> list[tuple[int, str]]:
    """(row_id, typology) of the period's positives (OTHER when a positive has no typology)."""
    ids = (
        pl.scan_parquet(paths.transactions)
        .filter(pl.col("rank").is_between(int(rows["first_rank"]), int(rows["last_rank"])))
        .select("row_id")
    )
    lab = (
        pl.scan_parquet(paths.labels)
        .filter(pl.col("is_laundering") == 1)
        .select("row_id", "typology")
    )
    got = ids.join(lab, on="row_id", how="inner").sort("row_id").collect()
    return [(int(r), t or OTHER) for r, t in got.iter_rows()]


# --- analysis -----------------------------------------------------------------------------------


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def alert_counts(alerts: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Alerts, true and false positives and precision of labelled alert records."""
    n = len(alerts)
    tp = sum(1 for a in alerts if a["y"] == 1)
    return {
        "alerts": n,
        "true_positives": tp,
        "false_positives": n - tp,
        "precision": _ratio(tp, n),
    }


def matcher_rows(alerts: Sequence[Mapping[str, Any]]) -> list[tuple[dict, str]]:
    """(the pack's typology features, ground-truth typology) of the true-positive alerts."""
    return [(a["features"], a["truth"]) for a in alerts if a["y"] == 1]


def _summary(acc: Mapping[str, Any] | None) -> dict[str, Any] | None:
    return None if acc is None else {k: acc[k] for k in SUMMARY_ACC}


def recall_by_typology(
    positives: Sequence[tuple[int, str]], alerted: set[int]
) -> dict[str, dict[str, Any]]:
    """{typology: positives, detected (alerted), missed, recall} over every typology, plus ALL."""
    counts = {t: [0, 0] for t in TYPOLOGIES}
    for row_id, t in positives:
        if t not in counts:
            raise ValueError(f"unknown typology {t!r}")
        counts[t][0] += 1
        counts[t][1] += row_id in alerted
    counts["ALL"] = [sum(c[0] for c in counts.values()), sum(c[1] for c in counts.values())]
    return {
        t: {"positives": p, "detected": d, "missed": p - d, "recall": _ratio(d, p)}
        for t, (p, d) in counts.items()
    }


def false_positives(
    alerts: Sequence[Mapping[str, Any]], n: int = N_FALSE_POSITIVES
) -> list[dict[str, Any]]:
    """The `n` highest-scoring false-positive alerts (ties: earlier rank) as an analyst reads
    them: when, score, matched typology, rules, top drivers and the narrative."""
    fps = sorted((a for a in alerts if a["y"] == 0), key=lambda a: (-a["score"], a["rank"]))
    out = []
    for a in fps[:n]:
        p = json.loads(a["pack"])
        why = p["why"]
        out.append(
            {
                "id": p["id"],
                "day": p["when"]["day"],
                "time": p["when"]["time"],
                "score": why["score"],
                "calibrated": why["calibrated"],
                "typology": why["typology"]["label"],
                "rules_fired": why["rules_fired"],
                "drivers": [
                    {k: d[k] for k in ("feature", "display", "contribution")}
                    for d in why["drivers"]
                ],
                "narrative": p["narrative"],
            }
        )
    return out


def label_counts(alerts: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Matcher labels of alert records, in TYPOLOGIES order."""
    c = Counter(a["label"] for a in alerts)
    return {t: c[t] for t in TYPOLOGIES}


def pick_examples(
    alerts: Sequence[Mapping[str, Any]], n: int = N_EXAMPLES
) -> list[Mapping[str, Any]]:
    """Up to `n` true-positive alerts with different ground-truth typologies, in TYPOLOGIES order:
    first the earliest alert of each typology the matcher labels correctly, then (to fill up)
    the earliest of the remaining typologies whatever its label."""
    tps = sorted((a for a in alerts if a["y"] == 1), key=lambda a: a["rank"])
    picks: list[Mapping[str, Any]] = []
    used: set[str] = set()
    for need_match in (True, False):
        for t in TYPOLOGIES:
            if len(picks) == n:
                return picks
            if t in used:
                continue
            a = next(
                (a for a in tps if a["truth"] == t and (a["label"] == t or not need_match)), None
            )
            if a is not None:
                picks.append(a)
                used.add(t)
    return picks


def params_source(params: Mapping[str, Any], val_report: Path) -> dict[str, Any]:
    """Whether the thresholds measured on test are the val-tuned ones (typology_match_val.json)."""
    if not Path(val_report).is_file():
        return {"source": "config (no val tuning report)", "val_tuned": None}
    tuned = read_json(val_report).get("tuned")
    same = tuned is not None and MatcherParams.from_dict(tuned) == MatcherParams.from_dict(params)
    return {
        "source": "val-tuned" if same else "config (differs from the val-tuned values)",
        "val_tuned": tuned,
    }


def tree_source(tree: TreeMatcher, val_report: Path) -> dict[str, Any]:
    """Whether the tree measured on test is the val-fitted one (typology_tree_val.json)."""
    path = Path(val_report)
    if not path.is_file():
        return {"source": "config (no val tree report)", "val_trained_on": None}
    try:
        fitted = TreeMatcher.from_dict(read_json(path))
    except ValueError as e:
        return {"source": f"config (unreadable val tree report: {e})", "val_trained_on": None}
    same = fitted.root == tree.root
    return {
        "source": "val-fitted" if same else "config (differs from the val-fitted tree)",
        "val_trained_on": fitted.trained_on,
    }


def require_frozen_tree(cfg: Mapping[str, Any]) -> TreeMatcher | None:
    """The frozen tree test measures (None: typology_model is rules and no tree is fitted).
    ValueError when typology_model is tree but the config holds no fitted tree (the
    single-leaf placeholder), since the packs would then silently use the decision list."""
    tree = frozen_tree(cfg)
    if tree is None and cfg.get("typology_model") == "tree":
        raise ValueError(
            f"typology_model is tree but configs/{TREE_FILE} holds no fitted tree: run "
            f"`make cases-val`, `make pull-reports`, copy reports/{TREE_REPORT} to "
            f"configs/{TREE_FILE}, then `make cases` (or set typology_model: rules)"
        )
    return tree


def typology_snippet(params: Mapping[str, Any]) -> str:
    """The `typology:` block of configs/explain.yaml for these thresholds."""
    lines = ["typology:"]
    for k, v in MatcherParams.from_dict(params).to_dict().items():
        lines.append(f"  {k}: {'null' if v is None else v}")
    return "\n".join(lines) + "\n"


# --- the stage ----------------------------------------------------------------------------------


@dataclass
class PeriodResult:
    period: str
    rows: dict[str, Any]
    champion: Champion
    champion_info: dict[str, Any]
    events: int
    alerts: list[dict[str, Any]]  # PackHook records + y / truth (join_labels)
    parity: dict[str, Any]  # ParityRecorder doc + digest + ok
    seconds: dict[str, float]


def evaluate_period(
    paths: DataPaths,
    period: str,
    explain_cfg: Mapping[str, Any],
    export_key: str | None,
    *,
    features_dir: Path,
    rules_dir: Path,
    graph_dir: Path,
    bundle_dir: Path | None = None,
    alert_tag: str = "headline",
    data_version: str | None = None,
    chunk_rows: int = PARITY_CHUNK_ROWS,
    log: Callable[[str], None] | None = None,
) -> PeriodResult:
    """Steps 1-3 of the module docstring, then the eval-only label join (no writes)."""
    if period not in PERIODS:
        raise ValueError(f"period must be one of {list(PERIODS)}, got {period!r}")
    cfg = check_config(explain_cfg)
    t0 = time.perf_counter()
    ch, calibration, info = load_champion(
        paths,
        export_key,
        features_dir=features_dir,
        rules_dir=rules_dir,
        graph_dir=graph_dir,
        bundle_dir=bundle_dir,
        alert_tag=alert_tag,
        data_version=data_version,
    )
    rows = period_rows(paths.transactions, period)
    refs = load_references(features_dir, rules_dir, graph_dir, period)
    t1 = time.perf_counter()
    snap = Path(features_dir) / SNAPSHOTS_DIR / snapshot_file(PERIODS[period].snapshot_split)
    st = stream_period(
        ch,
        cfg,
        snapshot=snap,
        transactions=paths.transactions,
        rows=rows,
        calibration=calibration,
        references=refs,
        chunk_rows=chunk_rows,
        log=log,
    )
    digest = check_digest(st.eng, period, features_dir=features_dir, rows=rows)
    parity = {**st.parity, "digest": digest}
    parity["ok"] = parity_ok(parity, rows["rows"])
    t2 = time.perf_counter()
    join_labels(paths.labels, st.alerts)  # eval only: after the stream and the parity check
    t3 = time.perf_counter()
    return PeriodResult(
        period=period,
        rows=rows,
        champion=ch,
        champion_info=info,
        events=st.events,
        alerts=st.alerts,
        parity=parity,
        seconds={
            "inputs": t1 - t0,
            "stream": st.seconds,
            "parity_checks": st.parity["seconds"],
            "labels": t3 - t2,
        },
    )


def _json_default(o: Any) -> Any:
    if hasattr(o, "item") and callable(o.item) and getattr(o, "shape", None) == ():
        return o.item()
    return str(o)


def write_json(doc: Any, path: Path) -> Path:
    """Strict JSON (no NaN), keys in the document's order (5W+H for packs)."""
    text = json.dumps(doc, indent=2, allow_nan=False, default=_json_default)
    return write_text_atomic(text + "\n", Path(path))


def _common(res: PeriodResult, keys: Mapping[str, str]) -> dict[str, Any]:
    ch = res.champion
    return {
        "period": res.period,
        "days": res.rows["days"],
        "rows": res.rows["rows"],
        "rate_tag": ch.alert_tag,
        "threshold": ch.threshold,
        "model_version": ch.model_version,
        "champion": res.champion_info,
        "keys": dict(keys),
    }


def write_val_reports(
    res: PeriodResult,
    cfg: Mapping[str, Any],
    keys: Mapping[str, str],
    out_dir: Path,
    positives: Sequence[tuple[int, str]],
) -> tuple[list[Path], dict[str, Any]]:
    """typology_tree_val.json (the tree fitted on the val true-positive alerts, with fit_tree's
    report under `info`: copy it to configs/typology_tree.json), typology_match_val.json (the
    tuned decision list, and the tree's train and cross-validated accuracy side by side with the
    list's) and case_eval_val.json (parity and the error analysis). Returns (paths, summary)."""
    grid = cfg.get("typology_grid")
    if not grid:
        raise ValueError("configs/explain.yaml has no typology_grid: nothing to tune on val")
    base = MatcherParams.from_dict(cfg["typology"])
    names = rules_matcher(cfg, res.champion.spec).names
    rows = matcher_rows(res.alerts)  # the packs' typology features: the tree's inputs
    rule_rows = [({n: f[n] for n in names}, t) for f, t in rows]  # the decision list's ten
    best, info = tune(rule_rows, grid, base=base)
    at_config = accuracy(rule_rows, base)
    fitted, tinfo = fit_tree(rows, **cfg["typology_tree_fit"])
    trained_on = {"period": res.period, "rows": len(rows), "run_key": keys.get("case_eval")}
    tree = dataclasses.replace(fitted, trained_on=trained_on)
    cv_folds = None if tinfo["cv"] is None else tinfo["cv"]["folds"]
    compare = {
        "rows": len(rows),
        "majority": info["best"]["majority"],
        "tree_cv": tinfo["cv_accuracy"],
        "tree_train": tinfo["train_accuracy"],
        "rules_tuned": info["best"]["accuracy"],
        "rules_at_config": at_config["accuracy"],
        "cv_folds": cv_folds,
        "note": "tree_cv is the out-of-fold accuracy of the same tree settings (stratified "
        "k-fold on these rows, not grouped by attempt, so optimistic; test is the measure); "
        "tree_train and rules_tuned are in-sample (fitted or tuned on these rows)",
    }
    tree_part = {
        "file": TREE_REPORT,
        "copy_to": f"configs/{TREE_FILE}",
        "params": tinfo["params"],
        "depth": tree.depth,
        "leaves": tree.n_leaves,
        "inputs": list(tree.inputs),
        "train_accuracy": tinfo["train_accuracy"],
        "cv_accuracy": tinfo["cv_accuracy"],
        "cv_folds": cv_folds,
        "importances": tinfo["importances"],
    }
    head = _common(res, keys)
    tuning = {
        **head,
        "rows_note": "true-positive alerts of the period; truth = the labelled typology, OTHER "
        "for positives outside the Patterns file",
        "compare": compare,
        "tree": tree_part,
        "tuned": best.to_dict(),
        "accuracy": info["best"],
        "at_config": at_config,
        "tuning": {k: info[k] for k in ("n_rows", "combinations", "grid", "base")},
        "next_step": f"copy {TREE_REPORT} to configs/{TREE_FILE} and `tuned` into "
        "configs/explain.yaml `typology:`, then `make cases`",
    }
    # The error analysis labels with the val matchers (the stream used the config's): the
    # fitted tree under typology_model tree (unless it is one leaf), else the tuned list.
    primary = (
        tree
        if cfg["typology_model"] == "tree" and not tree.trivial
        else dataclasses.replace(rules_matcher(cfg, res.champion.spec), params=best)
    )
    val_label = {a["id"]: primary.label(a["features"]) for a in res.alerts}
    fps = [
        {**fp, "typology_val_matcher": val_label[fp["id"]]} for fp in false_positives(res.alerts)
    ]
    neg = [{**a, "label": val_label[a["id"]]} for a in res.alerts if a["y"] == 0]
    alerted = {a["id"] for a in res.alerts}
    analysis = {
        **head,
        "parity": res.parity,
        "packs": len(res.alerts),
        "alerts": alert_counts(res.alerts),
        "recall_by_typology": recall_by_typology(positives, alerted),
        "false_positives": fps,
        "false_positive_typology": label_counts(neg),
        "typology_match": {
            "model": primary.model,  # the false positives' labels
            "compare": compare,
            "tree": tree_part,
            "tuned": best.to_dict(),
            "accuracy": info["best"],
            "report": VAL_REPORT,
        },
        "seconds": res.seconds,
    }
    out = Path(out_dir)
    paths = [
        write_json({**tree.to_dict(), "info": tinfo}, out / TREE_REPORT),
        write_json(tuning, out / VAL_REPORT),
        write_json(analysis, out / VAL_ANALYSIS),
    ]
    summary = {
        "tuned": best.to_dict(),
        "accuracy": {
            "rules": _summary(info["best"]),
            "tree_train": tinfo["train_accuracy"],
            "tree_cv": tinfo["cv_accuracy"],
            "cv_folds": cv_folds,
            "majority": compare["majority"],
        },
        "tree": {"depth": tree.depth, "leaves": tree.n_leaves},
        "tree_report": str(paths[0]),
    }
    return paths, summary


def write_examples(out_dir: Path, picks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """cases/case_<row_id>.json for each pick (old case_*.json files removed first)."""
    d = Path(out_dir) / CASES_DIR
    d.mkdir(parents=True, exist_ok=True)
    for old in d.glob("case_*.json"):
        old.unlink()
    index = []
    for a in picks:
        path = write_json(json.loads(a["pack"]), d / f"case_{a['id']}.json")
        index.append(
            {
                "file": f"{CASES_DIR}/{path.name}",
                "id": a["id"],
                "day": a["day"],
                "truth": a["truth"],
                "label": a["label"],
                "matched": a["label"] == a["truth"],
            }
        )
    return index


def write_test_reports(
    res: PeriodResult,
    cfg: Mapping[str, Any],
    keys: Mapping[str, str],
    out_dir: Path,
    test_views: Mapping[str, Sequence[int]],
) -> tuple[list[Path], dict[str, Any]]:
    """case_eval.json (parity; per view, on the same true-positive alerts, the accuracy of the
    frozen tree and of the frozen decision list, recomputed from the stored features, each with
    per-typology recall and its confusion matrix, plus the majority baseline) and the example
    packs. The stream built the packs with `typology_matcher` (`typology_match.model`)."""
    params = MatcherParams.from_dict(cfg["typology"])
    rules = rules_matcher(cfg, res.champion.spec)
    tree = frozen_tree(cfg)
    primary = typology_matcher(cfg, res.champion)
    out = Path(out_dir)
    views = {v: [int(test_views[v][0]), int(test_views[v][1])] for v in VIEWS}
    by_view = {v: [a for a in res.alerts if lo <= a["day"] <= hi] for v, (lo, hi) in views.items()}
    acc: dict[str, dict[str, Any]] = {}
    for v, al in by_view.items():
        rows = matcher_rows(al)
        by_rules = accuracy(rows, rules)
        acc[v] = {
            "n": by_rules["n"],
            "majority": by_rules["majority"],
            "tree": None if tree is None else accuracy(rows, tree),
            "rules": by_rules,
        }
    tree_part = None
    if tree is not None:
        tree_part = {
            "trained_on": tree.trained_on,
            **tree_source(tree, out / TREE_REPORT),
            "depth": tree.depth,
            "leaves": tree.n_leaves,
            "inputs": list(tree.inputs),
        }
    examples = write_examples(out, pick_examples(res.alerts))
    doc = {
        **_common(res, keys),
        "parity": res.parity,
        "packs": len(res.alerts),
        "views": views,
        "alerts": {v: alert_counts(al) for v, al in by_view.items()},
        "typology_match": {
            "model": primary.model,  # the matcher of the packs (and of the examples' labels)
            "rows_note": "true-positive alerts of each view; truth = the labelled typology, "
            "OTHER for positives outside the Patterns file; the decision list is the baseline",
            "tree": tree_part,
            "rules": {
                "params": params.to_dict(),
                **params_source(params.to_dict(), out / VAL_REPORT),
            },
            **acc,
        },
        "examples": examples,
        "seconds": res.seconds,
    }
    paths = [write_json(doc, out / TEST_REPORT)]
    paths += [out / e["file"] for e in examples]
    summary = {
        "typology_model": primary.model,
        "accuracy": {
            v: {
                "tree": _summary(a["tree"]),
                "rules": _summary(a["rules"]),
                "majority": a["majority"],
            }
            for v, a in acc.items()
        },
        "params_source": doc["typology_match"]["rules"]["source"],
        "tree_source": None if tree_part is None else tree_part["source"],
        "examples": len(examples),
    }
    return paths, summary


def run_case_eval(
    paths: DataPaths,
    period: str,
    data_cfg: Mapping[str, Any],
    explain_cfg: Mapping[str, Any],
    keys: Mapping[str, str],
    *,
    features_dir: Path,
    rules_dir: Path,
    graph_dir: Path,
    out_dir: Path,
    bundle_dir: Path | None = None,
    alert_tag: str = "headline",
    data_version: str | None = None,
    chunk_rows: int = PARITY_CHUNK_ROWS,
    log: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """The whole stage for one period: evaluate it, write its reports into `out_dir`, return a
    summary (`parity_ok` False: a parity count or the digest failed; the reports say which)."""
    t0 = time.perf_counter()
    cfg = check_config(explain_cfg)
    if period == "val":
        cfg = {**cfg, "typology_tree": None}  # val fits the tree; its packs never use a frozen one
    # Fail before minutes of streaming.
    if period == "val" and not cfg.get("typology_grid"):
        raise ValueError("configs/explain.yaml has no typology_grid: nothing to tune on val")
    if period == "test":
        require_frozen_tree(cfg)
    res = evaluate_period(
        paths,
        period,
        cfg,
        keys.get("export"),
        features_dir=features_dir,
        rules_dir=rules_dir,
        graph_dir=graph_dir,
        bundle_dir=bundle_dir,
        alert_tag=alert_tag,
        data_version=data_version,
        chunk_rows=chunk_rows,
        log=log,
    )
    if period == "val":
        positives = period_positives(paths, res.rows)
        written, part = write_val_reports(res, cfg, keys, out_dir, positives)
    else:
        written, part = write_test_reports(res, cfg, keys, out_dir, data_cfg["test_views"])
    par = res.parity
    return {
        "period": period,
        "rows": res.rows["rows"],
        "events": res.events,
        "alerts": len(res.alerts),
        "true_positive_alerts": sum(1 for a in res.alerts if a["y"] == 1),
        "champion_source": res.champion_info["source"],
        "rate_tag": res.champion.alert_tag,
        "parity_ok": par["ok"],
        "parity": {
            "mismatches": par["mismatches"],
            "alerts": par["alerts"],
            "digest_ok": par["digest"].get("ok"),
            "final_state_digest": par["digest"]["final_state_digest"],
        },
        **part,
        "reports": [str(p) for p in written],
        "seconds": time.perf_counter() - t0,
    }
