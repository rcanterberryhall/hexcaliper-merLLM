"""tests/test_instance_card_health_identity.py — the Ollama instance cards must
not report an outage they have not observed (#89).

Two halves, because the defect was a disagreement between them:

* The backend half — ``/api/merllm/activity`` reports an unreachable instance as
  ``loaded: null``, while the SSE stream carries no ``loaded`` at all. So
  ``null`` is the endpoint's outage sentinel, and a frontend cache that is
  *initialised* with it claims an outage nobody measured.
* The frontend half — ``renderInstanceCard`` must distinguish "not observed
  yet" from "observed unreachable", must not claim the element (and so suppress
  ``renderOverview``'s accurate ``/api/merllm/status`` header) until it can
  state the health, and the first full poll must repaint even while the stream
  is driving, because the next push may be minutes away.
"""
import json
import os
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))

WEB = Path(__file__).resolve().parents[1] / "web"


# ── Backend: what the two transports actually say about instance health ───────

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


class _UnreachableClient:
    """An httpx.AsyncClient whose context manager fails to connect."""

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        raise RuntimeError("connection refused")

    async def __aexit__(self, *a):
        return False


class _EmptyPsClient:
    """A reachable instance with no model loaded."""

    def __init__(self, *a, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False

    async def get(self, url, **kw):
        class _R:
            status_code = 200

            @staticmethod
            def json():
                return {"models": []}

        return _R()


def test_unreachable_instance_is_reported_as_loaded_null(client, monkeypatch):
    """An instance that cannot be probed comes back as loaded: null."""
    import app as app_mod

    monkeypatch.setattr(app_mod.httpx, "AsyncClient", _UnreachableClient)
    body = client.get("/api/merllm/activity").json()

    assert body["gpu0"]["loaded"] is None
    assert body["gpu1"]["loaded"] is None


def test_reachable_instance_with_no_model_is_reported_as_empty_list(client, monkeypatch):
    """A reachable-but-idle instance is [], never null.

    Paired with the test above so neither can be satisfied by an endpoint that
    answers null unconditionally: null is reserved for an outage, which is what
    makes it the wrong value to seed a cache with.
    """
    import app as app_mod

    monkeypatch.setattr(app_mod.httpx, "AsyncClient", _EmptyPsClient)
    body = client.get("/api/merllm/activity").json()

    assert body["gpu0"]["loaded"] == []
    assert body["gpu1"]["loaded"] == []


def test_sse_snapshot_carries_no_instance_health(tmp_path, monkeypatch):
    """The stream pushes activity only — no url, no loaded.

    This is why _mergeAndRenderInstance has to supply those two fields from the
    cache, and therefore why the cache's initial value is on screen.
    """
    import asyncio

    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "gpu_router", "metrics"):
            sys.modules.pop(mod, None)
    import app as app_mod

    q: asyncio.Queue = asyncio.Queue()
    app_mod._activity_sse_queues.append(q)
    try:
        app_mod._activity_set(app_mod.config.OLLAMA_0_URL, "qwen3:30b", "/api/chat")
        pushed = json.loads(q.get_nowait())
    finally:
        app_mod._activity_sse_queues.remove(q)

    assert "gpu0" in pushed and "gpu1" in pushed
    for key in ("gpu0", "gpu1"):
        entry = pushed[key]
        if entry is None:
            continue
        assert "loaded" not in entry, f"{key} carries loaded; the merge would be redundant"
        assert "url" not in entry, f"{key} carries url; the merge would be redundant"
    assert pushed["gpu0"] is not None, "the GPU that just started work should be in the push"


# ── Frontend: the shipped assets ─────────────────────────────────────────────

def _code(path):
    """The file with whole-line // comments dropped.

    Every assertion below is a text match, and this repo's house style puts a
    *why* comment above each step — so an un-stripped needle can be satisfied
    by prose describing what the code omits. Only whole-line comments are
    dropped, because `url.split("//")` is real code in this file.
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
    needle = "Per-GPU cache so SSE updates"
    assert needle in raw, "the comment above _instCache is expected in the source"
    assert needle not in app_js, "comments were not stripped"
    assert 'url.split("//")' in app_js, "stripping must not eat code containing //"


def test_instance_cache_starts_unknown_not_unreachable(app_js):
    """The cache seeds a known/unknown flag, not just the null sentinel."""
    start = app_js.index("const _instCache = {")
    decl = app_js[start: app_js.index("};", start)]
    for gpu in ("gpu0", "gpu1"):
        assert re.search(rf"{gpu}:.*known: false", decl), \
            f"_instCache.{gpu} must start as not-yet-observed, got: {decl!r}"


def test_merge_path_passes_the_unknown_flag_through(app_js):
    """A stream render says which of the two meanings its `loaded` has."""
    body = _fn(app_js, "_mergeAndRenderInstance")
    assert "loadedKnown:" in body, \
        "_mergeAndRenderInstance must tell the renderer whether the cache is seeded"
    assert "cached.known" in body


def test_renderer_separates_unknown_from_unreachable(app_js):
    """`unreachable` is reachable only once health has been observed."""
    body = _fn(app_js, "renderInstanceCard")

    assert "inst.loadedKnown !== false" in body, \
        "only an explicit false may mean unknown, or the full-poll callers change meaning"
    assert re.search(r"reachable\s*=\s*known &&", body), \
        "reachable must require known, or an unseeded cache still reports an outage"

    i_unknown = body.index("checking")
    i_unreach = body.index("unreachable")
    assert i_unknown < i_unreach, \
        "the unknown branch must be taken before the unreachable one"

    assert "dot-unknown" in body, "the unknown state needs a dot of its own"
    assert re.search(r'dot\s*=\s*!known \?\s*"dot-unknown"', body), \
        "the unknown state must not borrow dot-ok or dot-fail"


def test_unknown_render_does_not_claim_the_element(app_js):
    """renderOverview's accurate /status header must survive an unknown render."""
    body = _fn(app_js, "renderInstanceCard")
    assert re.search(r'if \(known\) el\.dataset\.activityReady = "1";', body), \
        "activityReady must be set only by a render that can state the health"

    overview = _fn(app_js, "renderOverview")
    assert 'activityReady !== "1"' in overview, \
        "renderOverview must still back off once an activity render has claimed the element"


def test_first_seed_repaints_even_while_the_stream_is_driving(app_js):
    """Otherwise the unknown render stands until the next push, which may not come."""
    body = _fn(app_js, "refresh")
    assert "_instCacheKnown()" in body, "the seed must notice it is the first one"
    assert re.search(r"if \(!_activityEventSrc \|\| wasUnknown\)", body), \
        "a first seed must repaint even with the stream open"
    assert "_cacheInstances(a)" in body

    order = body.index("wasUnknown = !_instCacheKnown()") < body.index("_cacheInstances(a)")
    assert order, "the unknown check must be read before the cache is overwritten"


def test_every_dot_class_the_js_emits_has_a_rule(app_js):
    """Derived cross-file check: a mistyped class is a silently invisible dot."""
    emitted = set(re.findall(r'"(dot-[a-z]+)"', app_js))
    styled = set(re.findall(r"\.(dot-[a-z]+)\s*\{", (WEB / "styles.css").read_text()))

    assert "dot-unknown" in emitted, "expected the unknown dot to be emitted by app.js"
    assert not emitted - styled, f"app.js emits dot classes with no CSS rule: {emitted - styled}"
