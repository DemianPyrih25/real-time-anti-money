"""aml.eval.typology: typology recall, attempt detection + time to first alert, memorisation."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from aml.eval.typology import (
    ALL,
    TYPOLOGIES,
    attempt_detection,
    attempt_ids,
    memorisation_split,
    recall_by_typology,
)


def test_recall_by_typology():
    y = np.array([1, 1, 1, 1, 0, 1, 0])
    typ = ["FAN-IN", "FAN-IN", "CYCLE", "OTHER", None, None, None]  # a null positive -> OTHER
    flags = np.array([True, False, True, False, True, True, False])
    r = recall_by_typology(flags, y, typ)
    assert list(r) == [*TYPOLOGIES, ALL]
    assert r["FAN-IN"] == {"positives": 2, "detected": 1, "recall": 0.5}
    assert r["CYCLE"]["recall"] == 1.0
    assert r["OTHER"] == {"positives": 2, "detected": 1, "recall": 0.5}
    assert r["STACK"]["positives"] == 0 and math.isnan(r["STACK"]["recall"])
    assert r[ALL] == {"positives": 5, "detected": 3, "recall": 0.6}


def test_recall_by_typology_rejects_unknown():
    with pytest.raises(ValueError):
        recall_by_typology(np.array([True]), np.array([1]), ["NOT-A-TYPOLOGY"])


def test_attempt_detection_and_time_to_first_alert():
    # attempt 0 (FAN-OUT): first row at minute 100, first alert at 130 -> 30 minutes
    # attempt 1 (FAN-OUT): first row alerted -> 0 minutes
    # attempt 2 (CYCLE): never alerted
    attempt = pl.Series([0, 0, 0, 1, 1, 2, 2, None], dtype=pl.Int32)
    typ = pl.Series(["FAN-OUT"] * 5 + ["CYCLE"] * 2 + [None])
    minute = np.array([100, 130, 160, 500, 520, 10, 20, 0])
    flags = np.array([False, True, True, True, False, False, False, True])
    r = attempt_detection(flags, attempt, typ, minute)
    assert "OTHER" not in r  # integration positives are not attempts
    assert r["FAN-OUT"]["attempts"] == 2 and r["FAN-OUT"]["detected"] == 2
    assert r["FAN-OUT"]["median_minutes_to_first_alert"] == 15.0
    assert r["FAN-OUT"]["mean_minutes_to_first_alert"] == 15.0
    assert r["CYCLE"] == {
        "attempts": 1,
        "detected": 0,
        "detection_rate": 0.0,
        "median_minutes_to_first_alert": pytest.approx(float("nan"), nan_ok=True),
        "mean_minutes_to_first_alert": pytest.approx(float("nan"), nan_ok=True),
    }
    assert r[ALL]["attempts"] == 3 and r[ALL]["detection_rate"] == pytest.approx(2 / 3)
    assert r["STACK"]["attempts"] == 0 and math.isnan(r["STACK"]["detection_rate"])


def test_attempt_ids_normalises_nulls():
    assert attempt_ids(pl.Series([3, None], dtype=pl.Int32)).tolist() == [3, -1]
    assert attempt_ids(np.array([2.0, np.nan])).tolist() == [2, -1]
    assert attempt_ids(np.array([1, None], dtype=object)).tolist() == [1, -1]


def test_memorisation_split():
    y = np.array([1, 1, 1, 0, 1])
    seen = np.array([True, True, False, True, False])
    flags = np.array([True, False, True, True, False])
    r = memorisation_split(flags, y, seen)
    assert r["seen"] == {"positives": 2, "detected": 1, "recall": 0.5}
    assert r["unseen"] == {"positives": 2, "detected": 1, "recall": 0.5}
    assert r["seen_share"] == 0.5
