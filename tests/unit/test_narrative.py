"""The template narrative (M6 spec §3): deterministic 5W+H text built only from the pack."""

from __future__ import annotations

import copy
import json

import pytest

from aml.explain.narrative import duration_words, narrative, plural

M = 8 * 1440 + 14 * 60 + 7  # day 9, 14:07

PACK = {
    "pack_format": 1,
    "id": 77,
    "case_key": "123-d9",
    "rank": 5000,
    "who": {
        "subject": {"account": 123, "bank": "0012"},
        "counterparty": {"account": 456, "bank": "0204"},
    },
    "what": {
        "amount_usd": 9800.0,
        "amount_paid": 9800.0,
        "payment_format": "ACH",
        "payment_currency": "US Dollar",
        "receiving_currency": "US Dollar",
    },
    "when": {"day": 9, "time": "14:07", "minute": M},
    "where": {"from_bank": "0012", "to_bank": "0204", "cross_bank": True},
    "why": {
        "score": 0.91234,
        "calibrated": 0.41,
        "threshold": 0.83,
        "rate_tag": "0p005",
        "rules_fired": ["fan_in_velocity"],
        "drivers": [
            {"feature": "v_in_uniq_1d", "value": 14.0, "display": "14", "contribution": 1.234},
            {"feature": "payment_format", "value": 0.0, "display": "ACH", "contribution": -0.5},
        ],
        "base_value": -3.0,
        "others_contribution": 0.1,
        "log_odds": -2.166,
        "typology": {
            "label": "FAN-IN",
            "evidence": [
                "the receiver had 14 distinct payers in the previous day (v_in_uniq_1d = 14)"
            ],
            "features": {"v_in_uniq_1d": 14.0},
        },
    },
    "how": {
        "subgraph": {
            "nodes": [
                {"id": 123, "role": "subject"},
                {"id": 456, "role": "counterparty"},
                {"id": 9, "role": "other"},
            ],
            "edges": [
                {
                    "src": 123,
                    "dst": 456,
                    "minute": M,
                    "amount_usd": 9800.0,
                    "kind": "alert",
                    "rank": 5000,
                },
                {
                    "src": 9,
                    "dst": 456,
                    "minute": 100,
                    "amount_usd": 5.0,
                    "kind": "history",
                    "rank": 10,
                },
            ],
        },
        "window_minutes": 4320,
        "cap_1hop": 10,
        "cap_2hop": 4,
    },
    "model_version": "fixture:0123456789ab",
    "export_key": "fixture",
}

EXPECTED = (
    "On day 9 at 14:07, account 123 (bank 0012) sent 9,800.00 USD by ACH to account 456 "
    "(bank 0204). It crossed banks (0012 to 0204). The model scored it 0.9123 (calibrated "
    "probability 0.410), at or above the alert threshold 0.8300 (rate tag 0p005). Top drivers "
    "(log-odds contributions): v_in_uniq_1d = 14 (+1.234); payment_format = ACH (-0.500); base "
    "value -3.000. Rules fired: fan_in_velocity. Matched typology: FAN-IN (the receiver had 14 "
    "distinct payers in the previous day (v_in_uniq_1d = 14)). The causal subgraph shows 3 "
    "accounts and 1 earlier payment from the 3 days before the alert."
)


def pack(**changes) -> dict:
    """A deep copy of PACK; `changes` maps "section.key" (or "key") to a new value."""
    p = copy.deepcopy(PACK)
    for path, value in changes.items():
        *head, last = path.split(".")
        d = p
        for k in head:
            d = d[k]
        d[last] = value
    return p


def test_the_narrative_reads_the_pack_in_5w_h_order():
    assert narrative(PACK) == EXPECTED


def test_the_narrative_is_deterministic_and_reads_only_the_pack():
    assert narrative(PACK) == narrative(copy.deepcopy(PACK))
    back = json.loads(json.dumps(PACK, sort_keys=True))  # other key order, through JSON
    assert narrative(back) == EXPECTED
    assert narrative({**PACK, "narrative": "stale text"}) == EXPECTED
    reordered = dict(reversed(list(PACK.items())))
    assert narrative(reordered) == EXPECTED


def test_unknown_and_empty_parts():
    p = pack(
        **{
            "who.subject.bank": None,
            "who.counterparty": {"account": 123, "bank": None},
            "what.payment_currency": "Euro",
            "what.receiving_currency": "Euro",
            "what.amount_paid": 9000.5,
            "where.from_bank": None,
            "where.to_bank": None,
            "where.cross_bank": None,
            "why.calibrated": None,
            "why.threshold": None,
            "why.rules_fired": [],
            "why.drivers": [],
            "why.typology": {"label": "OTHER", "evidence": ["no typology rule fired"]},
            "how.subgraph": {
                "nodes": [{"id": 123, "role": "subject"}],
                "edges": [
                    {
                        "src": 123,
                        "dst": 123,
                        "minute": M,
                        "amount_usd": 9800.0,
                        "kind": "alert",
                        "rank": 5000,
                    }
                ],
            },
        }
    )
    assert narrative(p) == (
        "On day 9 at 14:07, account 123 sent 9,800.00 USD by ACH to itself. The sender paid "
        "9,000.50 Euro. The model scored it 0.9123. No rule fired. Matched typology: OTHER (no "
        "typology rule fired). Neither account has an earlier payment in the 3 days before the "
        "alert."
    )


def test_currencies_banks_and_display_fallbacks():
    p = pack(
        **{
            "what.payment_currency": "Euro",
            "what.amount_paid": 8330.0,
            "where.cross_bank": False,
            "where.to_bank": "0012",
            "why.drivers": [{"feature": "u_out_mean_1d", "value": 6.123456, "contribution": 0.25}],
        }
    )
    text = narrative(p)
    assert "The sender paid 8,330.00 Euro; the receiver was credited in US Dollar." in text
    assert "Both accounts are at bank 0012." in text
    assert "u_out_mean_1d = 6.123 (+0.250)" in text  # no display: the value, 4 digits
    p = pack(**{"where.cross_bank": False, "where.from_bank": None, "where.to_bank": None})
    assert "Both accounts share a bank." in narrative(p)
    p = pack(**{"where.from_bank": None, "where.to_bank": None})
    assert "It crossed banks." in narrative(p)
    p = pack(**{"why.drivers": [{"feature": "pt_ratio_12h", "value": None, "contribution": 0.0}]})
    assert "pt_ratio_12h = n/a (+0.000)" in narrative(p)


@pytest.mark.parametrize(
    "minutes, words",
    [(1440, "day"), (4320, "3 days"), (720, "12 hours"), (60, "hour"), (90, "90 minutes"),
     (1, "minute"), (2880, "2 days")],
)  # fmt: skip
def test_duration_words(minutes, words):
    assert duration_words(minutes) == words


@pytest.mark.parametrize("bad", [0, -5])
def test_duration_words_refuses_non_positive_windows(bad):
    with pytest.raises(ValueError):
        duration_words(bad)


def test_plural():
    assert plural(1, "account") == "1 account" and plural(0, "account") == "0 accounts"
