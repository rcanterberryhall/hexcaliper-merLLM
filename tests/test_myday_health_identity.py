"""tests/test_myday_health_identity.py — the Ecosystem panel must report the
health it observed, and only about the app it observed it from (#93).

The defect was one flag asked to answer two questions. ``/api/merllm/myday``
computed ``ok`` three different ways in one response:

* ``soonstone`` used ``status is not None``, which the probe sets for *any*
  HTTP reply — so a 404 or a 502 scored as healthy, and the tile painted a
  green dot, a green border and a latency figure beside a feed status of
  ``unknown``.
* ``parsival``/``lancellmot`` used "the summary body parsed", so a running app
  whose summary endpoint answered non-2xx was reported with the literal word
  ``offline``, which ``README.md`` reserves for an app that is unreachable.
* The four heartbeat sites used ``status < 400``, the correct health reading,
  and so had the second problem only.

Both halves are tested here, because the fix is to split the question:
``reachable`` (an HTTP response arrived at all) is now separate from ``ok``
(it was healthy), ``detail`` names what answered, and the frontend has a third
state to paint rather than forcing every outcome into up-or-offline.
"""
import json
import os
import re
import sys
from pathlib import Path

import httpx
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

WEB = Path(__file__).resolve().parents[1] / "web"


# ── Fixtures and a per-URL fake httpx ────────────────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    """Return a TestClient with an isolated DB."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "gpu_router",
                   "metrics", "notifications"):
            sys.modules.pop(mod, None)
    from fastapi.testclient import TestClient
    import app as app_mod
    return TestClient(app_mod.app, raise_server_exceptions=True)


def _resp(url, status, *, json_body=None, text=None):
    """A real httpx.Response, so .json() and .raise_for_status() behave.

    Using the real class rather than a mock matters here: the pre-fix code
    reached its failure path through ``raise_for_status()`` and the post-fix
    code reaches it through ``status_code``, and a stubbed response that
    silently does neither would pass on both refs for the wrong reason.
    """
    req = httpx.Request("GET", url)
    if json_body is not None:
        return httpx.Response(status, json=json_body, request=req)
    return httpx.Response(status, text=text or "", request=req)


# The five probe URLs are distinguished by their real routes: soonstone serves
# /health, scrivener is probed at /, the rest at /healthz.
def _route(url):
    if "attention" in url:
        return "parsival"
    if "pending" in url:
        return "lancellmot"
    if url.endswith("/health"):
        return "soonstone"
    if url.endswith("/healthz"):
        return "healthz"
    return "root"


def _patch(monkeypatch, answers):
    """Point httpx at ``answers``: route name → Response or Exception.

    Anything unrouted answers healthily, so each test states only the one
    service it is about.
    """
    import app as app_mod

    def _default(url):
        name = _route(url)
        if name == "parsival":
            return _resp(url, 200, json_body={"active_situations": 3,
                                              "new_investigating": 2,
                                              "overdue_followups": 1,
                                              "cold_start": False})
        if name == "lancellmot":
            return _resp(url, 200, json_body={"acquisition_pending": 2,
                                              "escalation_pending": 1,
                                              "total_pending": 3})
        if name == "soonstone":
            return _resp(url, 200, json_body={"status": "ok"})
        return _resp(url, 200, text="ok")

    class _FakeClient:
        def __init__(self, *a, **kw):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get(self, url, **kw):
            answer = answers.get(_route(url))
            if answer is None:
                return _default(url)
            if isinstance(answer, Exception):
                raise answer
            return answer(url) if callable(answer) else answer

    monkeypatch.setattr(app_mod.httpx, "AsyncClient", _FakeClient)


def _myday(client, monkeypatch, answers):
    _patch(monkeypatch, answers)
    resp = client.get("/api/merllm/myday")
    assert resp.status_code == 200, "a sick companion never fails the panel"
    return resp.json()


_SITE_KEYS = ["soonstone", "havelock", "scrivener", "codex", "warden"]
_ALL_KEYS = ["parsival", "lancellmot", "merllm", *_SITE_KEYS]


# ── Soonstone: an HTTP error is not a health certificate ─────────────────────

def test_soonstone_bad_gateway_is_not_ok(client, monkeypatch):
    """A 502 behind a live proxy must not score as healthy.

    Pre-fix this read ok=True, because `status is not None` admits every
    status; the HTML body is unparseable, so data_status was null and the
    tile's only row read "Feed status unknown" beside a green dot.
    """
    body = _myday(client, monkeypatch, {
        "soonstone": lambda u: _resp(u, 502, text="<html>502 Bad Gateway</html>"),
    })
    s = body["soonstone"]
    assert s["ok"] is False, "a 502 is not a healthy feed"
    assert s["reachable"] is True, "something answered, so it is not offline either"
    assert s["detail"] == "HTTP 502", f"the tile needs to name what answered, got {s['detail']!r}"
    assert s["data_status"] is None, "an HTML error body carries no feed verdict"


def test_soonstone_wrong_port_404_is_not_ok(client, monkeypatch):
    """`soonstone_url` is settable in Settings, and every app shares this host.

    A mistyped port lands on a live neighbour, which answers 404 for /health
    with a JSON body that has no `status` field — pre-fix a green tile with a
    plausible latency, reporting a service that was never contacted.
    """
    body = _myday(client, monkeypatch, {
        "soonstone": lambda u: _resp(u, 404, json_body={"detail": "Not Found"}),
    })
    s = body["soonstone"]
    assert s["ok"] is False, "a 404 from the wrong host is not soonstone being up"
    assert s["reachable"] is True
    assert s["detail"] == "HTTP 404"
    assert s["data_status"] is None, "a neighbour's 404 body is not a feed verdict"


def _stale_feed(client, monkeypatch):
    """soonstone answering the 503 it uses for an ingest older than 90 minutes."""
    return _myday(client, monkeypatch, {
        "soonstone": lambda u: _resp(u, 503, json_body={"status": "stale",
                                                        "ingest_metars": None}),
    })["soonstone"]


def test_soonstone_stale_feed_keeps_its_verdict(client, monkeypatch):
    """The 503 the old tolerance was written for must survive the fix.

    Asserts only what holds on both refs, so it is a real no-regression guard:
    soonstone/api/health.py answers 503 with `status: stale`, and the site's
    own verdict — not the status code — is what the tile has to show.
    """
    s = _stale_feed(client, monkeypatch)
    assert s["data_status"] == "stale", "the site's own verdict must reach the tile"
    assert s["latency_ms"] is not None, "it answered, so it was timed"


def test_soonstone_stale_feed_is_reachable(client, monkeypatch):
    """And a stale feed is still an answer, so the tile must not read offline."""
    s = _stale_feed(client, monkeypatch)
    assert s["reachable"] is True, "503 is an answer"
    assert s["ok"] is False, "but 503 is not a healthy feed either"


def test_soonstone_healthy_feed_is_ok(client, monkeypatch):
    """A 200 with `status: ok` still reads healthy.

    Passes on both refs by design — without it the suite could be satisfied by
    a fix that simply calls everything degraded.
    """
    s = _myday(client, monkeypatch, {})["soonstone"]
    assert s["ok"] is True
    assert s["data_status"] == "ok"


def test_parsival_healthy_summary_still_populates_its_counts(client, monkeypatch):
    """The other both-refs guard: a working summary still fills the tile's rows."""
    p = _myday(client, monkeypatch, {})["parsival"]
    assert p["ok"] is True
    assert p["active_situations"] == 3
    assert p["overdue_followups"] == 1
    assert p["cold_start"] is False


# ── Reachable-but-failing is distinguishable from absent ─────────────────────

def test_unreachable_sites_are_not_reachable(client, monkeypatch):
    """No HTTP response at all — the one case README:220 calls offline."""
    exc = httpx.ConnectError("connection refused")
    body = _myday(client, monkeypatch, {k: exc for k in
                                        ("soonstone", "healthz", "root",
                                         "parsival", "lancellmot")})
    for key in ["parsival", "lancellmot", *_SITE_KEYS]:
        section = body[key]
        assert section["ok"] is False, key
        assert section["reachable"] is False, key
        assert section["detail"] is None, f"{key}: nothing answered, so nothing to quote"
    assert body["soonstone"]["latency_ms"] is None


def test_heartbeat_site_error_is_reachable(client, monkeypatch):
    """A 500 from a live site is not the same observation as silence.

    Paired with the test above: the two differ only in `reachable`, which is
    the distinction the panel had no way to paint before.
    """
    body = _myday(client, monkeypatch, {
        "healthz": lambda u: _resp(u, 500, text="internal server error"),
    })
    for key in ("havelock", "codex", "warden"):
        section = body[key]
        assert section["ok"] is False, key
        assert section["reachable"] is True, key
        assert section["detail"] == "HTTP 500", key
    assert body["scrivener"]["ok"] is True, "only the /healthz sites were sick"


# ── Parsival / LanceLLMot: a failed summary is not an offline app ────────────

def test_parsival_summary_error_is_reachable(client, monkeypatch):
    """parsival's /attention/summary reads SQLite, so a locked DB answers 500.

    The app is up and the tile's own href still works; pre-fix the tile read
    the literal word `offline` and suppressed the overdue-follow-up row at the
    same moment it gave a reason to believe there was nothing to see.
    """
    body = _myday(client, monkeypatch, {
        "parsival": lambda u: _resp(u, 500, text="Internal Server Error"),
    })
    p = body["parsival"]
    assert p["reachable"] is True, "a 500 came from a running app"
    assert p["ok"] is False, "but the summary is not in hand"
    assert p["detail"] == "HTTP 500"
    assert p["overdue_followups"] == 0, "no count was read, so none is claimed"
    assert body["lancellmot"]["ok"] is True, "only parsival was sick"


def test_parsival_unreadable_body_is_reachable(client, monkeypatch):
    """A 2xx whose body will not parse is a third outcome, not an outage."""
    body = _myday(client, monkeypatch, {
        "parsival": lambda u: _resp(u, 200, text="<html>cached page</html>"),
    })
    p = body["parsival"]
    assert p["reachable"] is True
    assert p["ok"] is False
    assert p["detail"] and "body" in p["detail"], \
        f"the detail should name the body, got {p['detail']!r}"


def test_parsival_unreachable_is_offline(client, monkeypatch):
    """The genuine outage still reads as one."""
    body = _myday(client, monkeypatch, {
        "parsival": httpx.ConnectTimeout("timed out"),
    })
    p = body["parsival"]
    assert p["reachable"] is False
    assert p["ok"] is False


# ── Shape: every section answers the same three questions ────────────────────

def test_every_section_carries_the_same_health_fields(client, monkeypatch):
    """One dot is painted from these, so one vocabulary has to serve them all."""
    body = _myday(client, monkeypatch, {})
    for key in _ALL_KEYS:
        section = body[key]
        for field in ("ok", "reachable", "detail"):
            assert field in section, f"{key} is missing {field}"
        assert section["ok"] is True, key
        assert section["reachable"] is True, key
        assert section["detail"] is None, key


def test_merllm_self_section_needs_no_probe(client, monkeypatch):
    """This response is merLLM's own evidence of being up."""
    body = _myday(client, monkeypatch, {
        k: httpx.ConnectError("down") for k in
        ("soonstone", "healthz", "root", "parsival", "lancellmot")
    })
    assert body["merllm"]["ok"] is True
    assert body["merllm"]["reachable"] is True


# ── Frontend: the shipped assets ─────────────────────────────────────────────

def _code(path):
    """The file with whole-line // comments dropped.

    Every assertion below is a text match and this repo comments the *why*
    above each step, so an un-stripped needle can be satisfied by prose
    describing what the code omits.
    """
    lines = (WEB / path).read_text().split("\n")
    return "\n".join("" if l.strip().startswith("//") else l for l in lines)


def _fn(src, name):
    """The body of a top-level `function name(` up to the next top-level one."""
    start = src.index(f"function {name}(")
    rest = src[start + 1:]
    m = re.search(r"\n(?:async )?function ", rest)
    return rest[: m.start()] if m else rest


@pytest.fixture(scope="module")
def app_js():
    return _code("app.js")


def test_comment_stripping_is_in_effect(app_js):
    """Guard on the guard: the needles below must not be matching prose."""
    raw = (WEB / "app.js").read_text()
    needle = "One ecosystem tile"
    assert needle in raw, "the comment above mydayTile is expected in the source"
    assert needle not in app_js, "comments were not stripped"
    assert 'url.split("//")' in app_js, "stripping must not eat code containing //"


def test_offline_is_claimed_only_when_nothing_answered(app_js):
    """`down` must require !reachable, not merely !ok."""
    body = _fn(app_js, "mydayState")
    assert re.search(r"app\.reachable \? \"degraded\" : \"down\"", body), \
        f"mydayState must reach `down` only via reachable, got: {body!r}"
    assert re.search(r"app\.ok \? \"up\"", body), "a healthy app is up"


def test_tile_paints_three_states(app_js):
    """The two-state flag is gone, and `offline` belongs to one branch only."""
    body = _fn(app_js, "mydayTile")
    assert "online" not in body, "the two-state `online` flag must be gone"
    assert body.count("myday-offline-msg") == 1, \
        "the word offline must be emitted from exactly one branch"
    assert re.search(r'st === "down"', body), "the down branch must be explicit"
    assert re.search(r'st === "degraded"', body), "the degraded branch must be explicit"
    assert "myday-degraded-msg" in body, "a degraded tile has to say what answered"


def test_degraded_tile_shows_what_answered(app_js):
    """The reason the server sent is what the tile prints."""
    body = _fn(app_js, "mydayTile")
    assert re.search(r"const msg = detail \|\|", body), \
        "the server's detail must be preferred over any canned wording"
    assert "esc(msg)" in body, "the detail is server text and has to be escaped"


def test_latency_is_shown_whenever_it_was_measured(app_js):
    """A degraded app answered, and how fast is part of the observation."""
    body = _fn(app_js, "mydayTile")
    assert re.search(r'st !== "down" && latencyMs != null', body), \
        "latency must not be gated on the answer having been the wanted one"


def test_attention_override_cannot_repaint_an_unhealthy_tile(app_js):
    """`cls` means "healthy, wants attention" — never "well after all"."""
    body = _fn(app_js, "mydayTile")
    assert re.search(r'\(st === "up" && cls\) \|\| _MYDAY_CLS\[st\]', body), \
        "the caller's class override must be confined to the up state"


def test_no_tile_is_painted_from_a_two_state_flag(app_js):
    """All five call sites moved to the three-state vocabulary."""
    body = _fn(app_js, "renderMyDay")
    assert "online:" not in body, "no tile may still be passing `online:`"
    assert body.count("mydayTile({") == 5, "expected the five ecosystem tiles"
    assert body.count("mydayState(") == 3, \
        "parsival, lancellmot and the heartbeat loop derive their state"


def test_soonstone_tile_needs_a_verdict_before_claiming_health(app_js):
    """`unknown` was a word meaning no-information printed beside a green dot."""
    body = _fn(app_js, "renderMyDay")
    assert '|| "unknown"' not in body, \
        "a missing feed verdict must not be rendered as a feed status"
    assert "s.reachable" in body, "the soonstone tile must consult reachable"
    assert re.search(r'feed === "ok" \? "up" : "degraded"', body), \
        "a feed verdict that is not ok is a degraded tile, not a green one"
    assert re.search(r'rows: feed \?', body), \
        "the feed row is shown when a verdict was read, not when ok is true"


def test_every_myday_class_the_js_emits_has_a_css_rule(app_js):
    """Derived cross-file check: a mistyped class is an invisible dot."""
    emitted = set(re.findall(r'"(myday-[a-z-]+)"', app_js))
    emitted |= set(re.findall(r'myday-card (myday-[a-z-]+)"', app_js))
    styled = set(re.findall(r"\.(myday-[a-z-]+)", (WEB / "styles.css").read_text()))

    assert "myday-degraded" in emitted, "the degraded card class should be emitted"
    assert "myday-degraded-msg" in emitted, "the degraded message class should be emitted"
    assert not emitted - styled, f"app.js emits myday classes with no CSS rule: {emitted - styled}"


def test_every_dot_state_has_a_css_rule(app_js):
    """The amber dot is the whole point: it must not render as no dot."""
    body = _fn(app_js, "mydayTile")
    start = app_js.index("const _MYDAY_DOT = {")
    decl = app_js[start: app_js.index("};", start)]
    assert re.search(r"degraded:\s*\"warn\"", decl), \
        "the degraded state needs a dot class of its own"
    css = (WEB / "styles.css").read_text()
    for cls in re.findall(r'"(up|warn|down)"', decl):
        assert f".myday-dot.{cls}" in css, f"no rule for .myday-dot.{cls}"
