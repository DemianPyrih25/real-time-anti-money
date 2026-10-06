"""M6 case pages (spec §4): the `cases` table, the case-pack hook `open_runtime` registers, and
the /cases routes, served by uvicorn in a thread on a pre-bound socket and read with urllib, on
the fixture bundle's inproc stream."""

from __future__ import annotations

import copy
import json
import logging
import re
import shutil
import threading
import urllib.error
import urllib.request

import polars as pl
import pytest

from aml.serving import bundle
from aml.serving.alerts import (
    ALERT_COLUMNS,
    SqliteAlertStore,
    count_alerts,
    count_cases,
    list_cases,
    read_alerts,
    read_case,
    read_case_json,
)
from aml.serving.app import CYTOSCAPE_URL, create_app, render_case, render_case_list, script_json
from aml.serving.scorer import Champion, open_case_hook, open_runtime, parity_ok
from aml.serving.settings import ConfigError
from aml.serving.state import ALERTS_DB
from tests.fixtures.serving_bundle import CONFIG_DIR, fixture_settings
from tests.unit.test_serving_app import Served

OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))  # never a proxy
DATA_OPEN = '<script type="application/json" id="case-data">'
CASE_LINK = re.compile(r'href="/cases/(-?\d+)"')
EVIL = "</script><script>alert(1)</script><img src=x onerror=alert(2)>"


def fetch(base: str, path: str) -> tuple[int, str, bytes]:
    """(status, content type, body), also for error statuses."""
    try:
        with OPENER.open(base + path, timeout=10) as r:
            return r.status, r.headers.get("content-type", ""), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("content-type", ""), e.read()


def esc(s: str) -> str:
    """The escaping jinja2's autoescape applies (markupsafe)."""
    for a, b in (("&", "&amp;"), ("<", "&lt;"), (">", "&gt;"), ('"', "&#34;"), ("'", "&#39;")):
        s = s.replace(a, b)
    return s


def embedded(page: str) -> dict:
    """The pack inlined in a case page, read the way the page script reads it."""
    start = page.index(DATA_OPEN) + len(DATA_OPEN)
    return json.loads(page[start : page.index("</script>", start)])


@pytest.fixture(scope="module")
def served(serving_bundle, fixture_tag, tmp_path_factory):
    """The app (case packs on by default) after the whole fixture slice, still serving."""
    root = tmp_path_factory.mktemp("cases_http")
    settings = fixture_settings(serving_bundle, root / "runtime", fixture_tag, exit_at_end=False)
    app = create_app(settings)
    db = settings.runtime_dir / ALERTS_DB
    with Served(app, attach=True) as srv:
        code, health = srv.wait_status("done")
        yield {
            "base": srv.base,
            "app": app,
            "db": db,
            "code": code,
            "health": health,
            "alerts": read_alerts(db, limit=None),  # newest rank first
            "slice": pl.read_parquet(serving_bundle / bundle.SLICE),
        }


# --- the stream writes one case per alert -----------------------------------------------------


def test_every_alert_has_a_case_and_m5_outputs_are_unchanged(served):
    db, alerts, h = served["db"], served["alerts"], served["health"]
    assert served["code"] == 200 and len(alerts) == h["alerts"] >= 1
    par = h["parity"]  # the end-of-slice check ran with the case hook on
    assert par["mismatches"] == dict.fromkeys(bundle.CHECKS, 0)
    assert par["alerts_ok"] is True and par["digest_ok"] is True
    assert count_cases(db) == count_alerts(db) == len(alerts)
    rt = served["app"].state.scorer.rt
    assert rt.case_hook is not None and rt.case_hook.failed == 0
    assert rt.summary()["cases"] == {"built": len(alerts), "failed": 0}
    banks = {
        r["row_id"]: (str(r["from_bank"]), str(r["to_bank"]))
        for r in served["slice"].select("row_id", "from_bank", "to_bank").to_dicts()
    }
    for a in alerts:
        pack = read_case(db, a["row_id"])
        assert pack is not None and pack["id"] == a["row_id"] and pack["rank"] == a["rank"]
        assert pack["case_key"] == f"{a['src']}-d{a['day']}"
        assert pack["who"]["subject"]["account"] == a["src"]
        assert pack["who"]["counterparty"]["account"] == a["dst"]
        assert pack["why"]["score"] == a["score"] and pack["why"]["rules_fired"] == a["rules_fired"]
        # Scored carries the event's fields, so the banks are the event's own.
        fb, tb = banks[a["row_id"]]
        assert (pack["who"]["subject"]["bank"], pack["who"]["counterparty"]["bank"]) == (fb, tb)
        assert pack["where"] == {"from_bank": fb, "to_bank": tb, "cross_bank": fb != tb}
    listed = [x["row_id"] for g in list_cases(db, None) for x in g["alerts"]]
    assert sorted(listed) == sorted(a["row_id"] for a in alerts)


# --- routes -----------------------------------------------------------------------------------


def test_case_json_is_the_stored_pack(served):
    base, db = served["base"], served["db"]
    for a in served["alerts"]:
        code, ctype, raw = fetch(base, f"/cases/{a['row_id']}.json")
        assert code == 200 and ctype.startswith("application/json")
        assert raw.decode("utf-8") == read_case_json(db, a["row_id"])
        assert json.loads(raw) == read_case(db, a["row_id"])


def test_case_page_shows_the_pack(served):
    base, db = served["base"], served["db"]
    for a in served["alerts"]:
        code, ctype, raw = fetch(base, f"/cases/{a['row_id']}")
        assert code == 200 and ctype.startswith("text/html")
        page = raw.decode("utf-8")
        pack = read_case(db, a["row_id"])
        subject = pack["who"]["subject"]["account"]
        assert f'Account <span class="mono">{subject}</span>' in page
        assert f'<script src="{CYTOSCAPE_URL}"' in page
        assert embedded(page) == pack  # the graph script's data, exactly the stored pack
        assert page.count("</script>") == 3  # data, Cytoscape, the page script
        assert 'href="/cases"' in page and f'href="/cases/{a["row_id"]}.json"' in page
        assert esc(pack["narrative"]) in page
        why = pack["why"]
        assert esc(why["typology"]["label"]) in page
        for line in why["typology"]["evidence"]:
            assert esc(line) in page
        for d in why["drivers"]:
            assert f'<span class="mono">{esc(d["feature"])}</span>' in page
        assert page.count('<div class="drow">') == len(why["drivers"])
        assert f"All {len(pack['how']['subgraph']['edges'])} payments" in page


def test_unknown_case_is_404(served):
    base = served["base"]
    known = {a["row_id"] for a in served["alerts"]}
    plain = next(r for r in served["slice"]["row_id"].to_list() if r not in known)  # no alert
    for row_id in (-1, plain, max(known) + 10**9):
        assert fetch(base, f"/cases/{row_id}")[0] == 404
        assert fetch(base, f"/cases/{row_id}.json")[0] == 404
    assert fetch(base, "/cases/abc")[0] == 422


def test_case_list_groups_the_newest_alerts(served):
    base, db, alerts = served["base"], served["db"], served["alerts"]
    code, ctype, raw = fetch(base, "/cases?limit=1000")
    assert code == 200 and ctype.startswith("text/html")
    page = raw.decode("utf-8")
    links = CASE_LINK.findall(page)
    assert sorted(int(x) for x in links) == sorted(a["row_id"] for a in alerts[:1000])
    keys = list(dict.fromkeys(f"{a['src']}-d{a['day']}" for a in alerts))  # first appearance
    for k in keys:
        assert f'id="case-{k}"' in page
    groups = list_cases(db, None)
    assert [g["case_key"] for g in groups] == keys
    for g in groups:
        ranks = [x["rank"] for x in g["alerts"]]
        assert ranks == sorted(ranks, reverse=True)
        assert {x["row_id"] for x in g["alerts"]} == {
            a["row_id"] for a in alerts if f"{a['src']}-d{a['day']}" == g["case_key"]
        }
    assert [int(x) for x in links] == [x["row_id"] for g in groups for x in g["alerts"]]

    first = fetch(base, "/cases?limit=1")[2].decode("utf-8")
    assert CASE_LINK.findall(first) == [str(alerts[0]["row_id"])]
    if len(alerts) > 1:
        assert f"/cases?before_row={alerts[0]['row_id']}&amp;limit=1" in first
        second = fetch(base, f"/cases?limit=1&before_row={alerts[0]['row_id']}")[2]
        assert CASE_LINK.findall(second.decode("utf-8")) == [str(alerts[1]["row_id"])]
    oldest = fetch(base, f"/cases?before_row={alerts[-1]['row_id']}")[2].decode("utf-8")
    assert CASE_LINK.findall(oldest) == [] and "No cases here yet" in oldest
    assert fetch(base, "/cases?limit=0")[0] == 422
    assert fetch(base, "/cases?limit=1001")[0] == 422


# --- rendering is injection-safe --------------------------------------------------------------


def test_script_json_cannot_close_the_script_element():
    obj = {"a": "</script><script>x()</script>", "b": "<!-- & -->", "c": "  é", "d": 0.1 + 0.2}
    text = script_json(obj)
    assert text.isascii() and not set("<>&") & set(text)
    assert json.loads(text) == obj


def test_hostile_pack_text_is_escaped(served):
    pack = copy.deepcopy(read_case(served["db"], served["alerts"][0]["row_id"]))
    pack["case_key"] = EVIL
    pack["narrative"] = EVIL
    pack["what"]["payment_format"] = EVIL
    pack["why"]["typology"]["evidence"] = [EVIL]
    pack["why"]["drivers"][0]["display"] = EVIL
    page = render_case(pack)
    assert "<script>alert(1)" not in page and "<img src=x" not in page
    assert esc(EVIL) in page
    assert page.count("</script>") == 3
    assert embedded(page) == pack

    alert = {"row_id": 1, "rank": 5, "minute": 61, "time": "01:01", "counterparty": 2,
             "amount_usd": 1.5, "payment_format": EVIL, "score": 0.5, "typology": EVIL,
             "rules_fired": [EVIL]}  # fmt: skip
    groups = [{"case_key": EVIL, "subject": 3, "day": 1, "alerts": [alert]}]
    listing = render_case_list(groups, n_cases=1, limit=100, before_row=None)
    assert "<script" not in listing and "<img" not in listing and esc(EVIL) in listing


def test_case_page_names_the_typology_matcher(served):
    """A tree-labelled pack shows the model and its decision path; a pack written before the
    tree (no `model`) renders as the decision list's."""
    pack = copy.deepcopy(read_case(served["db"], served["alerts"][0]["row_id"]))
    typ = pack["why"]["typology"]
    typ.update(
        label="FAN-OUT",
        model="tree",
        evidence=[
            "the sender paid 7 distinct counterparties in the previous day "
            "(u_out_uniq_1d = 7 > 3.5)",
            "82% of 34 validation cases at this leaf were FAN-OUT",
        ],
    )
    page = render_case(pack)
    assert "decision tree" in page and "typology <b>FAN-OUT</b> (tree)" in page
    for line in typ["evidence"]:
        assert esc(line) in page
    assert embedded(page) == pack
    del typ["model"]
    old = render_case(pack)
    assert "decision list" in old and "typology <b>FAN-OUT</b> (rules)" in old


# --- the cases table --------------------------------------------------------------------------


def _alert(row_id: int, rank: int, src: int, day: int) -> dict:
    rec = {
        "row_id": row_id,
        "rank": rank,
        "offset": rank - 100,
        "minute": (day - 1) * 1440 + rank - 40,
        "day": day,
        "src": src,
        "dst": 4,
        "amount_usd": 100.0 + rank,
        "payment_format": "ACH",
        "score": 0.1 + 0.2,
        "threshold": 0.25,
        "rate_tag": "0p005",
        "rules_fired": ["fan_in_velocity"],
        "severities": [0.0] * 7,
        "model_version": "export-x:0123456789ab",
    }
    assert list(rec) == list(ALERT_COLUMNS)
    return rec


def _pack(row_id: int, src: int, day: int, **extra) -> dict:
    return {
        "id": row_id,
        "case_key": f"{src}-d{day}",
        "who": {"subject": {"account": src}},
        "when": {"day": day},
        "why": {"typology": {"label": "FAN-IN", "evidence": ["</script>"]}, "score": 0.1 + 0.2},
        **extra,
    }


def test_cases_table_is_idempotent_and_grouped_newest_first(tmp_path):
    path = tmp_path / "rt" / ALERTS_DB
    assert read_case(path, 1) is None and list_cases(path) == [] and count_cases(path) == 0
    st = SqliteAlertStore(path)
    try:
        assert list_cases(path) == [] and read_case_json(path, 1) is None
        for row_id, rank, src, day in ((1, 101, 7, 9), (2, 102, 8, 9), (3, 103, 7, 9),
                                       (4, 104, 7, 10)):  # fmt: skip
            st.write(_alert(row_id, rank, src, day))
            assert st.write_case(_pack(row_id, src, day)) is True
            assert st.write_case(_pack(row_id, src, day, extra=1)) is False  # the first wins
        assert st.count_cases() == count_cases(path) == 4  # a reader while the writer is open
        assert read_case(path, 3) == _pack(3, 7, 9)
        assert json.loads(read_case_json(path, 3)) == _pack(3, 7, 9)

        def ids(groups: list[dict]) -> list[tuple[str, list[int]]]:
            return [(g["case_key"], [a["row_id"] for a in g["alerts"]]) for g in groups]

        groups = list_cases(path)
        assert ids(groups) == [("7-d10", [4]), ("7-d9", [3, 1]), ("8-d9", [2])]
        assert (groups[1]["subject"], groups[1]["day"]) == (7, 9)
        assert groups[1]["alerts"][0] == {
            "row_id": 3,
            "rank": 103,
            "minute": 8 * 1440 + 63,
            "time": "01:03",
            "counterparty": 4,
            "amount_usd": 203.0,
            "payment_format": "ACH",
            "score": 0.1 + 0.2,
            "typology": "FAN-IN",
            "rules_fired": ["fan_in_velocity"],
        }
        assert ids(list_cases(path, limit=2)) == [("7-d10", [4]), ("7-d9", [3])]
        assert ids(list_cases(path, limit=2, before_row=3)) == [("8-d9", [2]), ("7-d9", [1])]
        assert list_cases(path, before_row=999) == []  # an unknown cursor row
        st.write_case(_pack(5, 7, 9))  # a case without its alert row is not listed
        assert count_cases(path) == 5 and len(ids(list_cases(path))) == 3
        with pytest.raises(ValueError):
            st.write_case(_pack(6, 7, 9, bad=float("nan")))  # packs are strict JSON
        assert read_case(path, 6) is None and len(st.rows()) == 4
    finally:
        st.close()


# --- the switch -------------------------------------------------------------------------------


def test_case_packs_are_on_by_default_and_need_explain_yaml(
    serving_bundle, fixture_tag, tmp_path, caplog
):
    on = fixture_settings(serving_bundle, tmp_path / "a", fixture_tag)
    kafka = fixture_settings(serving_bundle, tmp_path / "b", fixture_tag, transport="kafka")
    off = fixture_settings(serving_bundle, tmp_path / "c", fixture_tag, case_packs=False)
    assert on.case_packs is True and kafka.case_packs is True and off.case_packs is False
    ch = Champion.from_bundle(serving_bundle, alert_tag=fixture_tag)
    sink: list[dict] = []
    assert open_case_hook(off, ch, sink.append) is None
    hook = open_case_hook(on, ch, sink.append)
    assert hook is not None and hook.strict is False and hook.calibration is not None
    cfg_dir = tmp_path / "cfg"
    cfg_dir.mkdir()
    shutil.copy(CONFIG_DIR / "serving.yaml", cfg_dir / "serving.yaml")
    lone = fixture_settings(
        serving_bundle, tmp_path / "d", fixture_tag, config=cfg_dir / "serving.yaml"
    )
    with caplog.at_level(logging.WARNING, logger="aml.serving.scorer"):
        assert open_case_hook(lone, ch, sink.append) is None
    assert "case packs are off" in caplog.text
    (cfg_dir / "explain.yaml").write_text("window_minutes: 0\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="explain.yaml"):
        open_case_hook(lone, ch, sink.append)
    assert sink == []


def test_without_case_packs_the_m5_run_is_unchanged(serving_bundle, fixture_tag, tmp_path):
    settings = fixture_settings(serving_bundle, tmp_path / "rt", fixture_tag, case_packs=False)
    rt = open_runtime(settings)
    result = rt.run(threading.Event())
    assert rt.close(result) == 0 and result == "end" and parity_ok(rt.parity)
    assert rt.case_hook is None and rt.summary()["cases"] is None
    db = rt.store.alerts_db
    assert count_alerts(db) == rt.progress.alerts >= 1 and count_cases(db) == 0
