"""tests/test_v1_route.py — OpenAI-compatible /v1 proxy routes.

Covers ``proxy_v1_chat_completions`` and ``proxy_v1_models``:
- streaming requests relay upstream SSE bytes verbatim as text/event-stream
- ``X-Priority`` maps strictly via ``priority_from_name`` (missing falls to
  BACKGROUND — this route has no legacy clients, unlike ``_priority``'s
  missing→CHAT compatibility on the native routes)
- a CHAT queue timeout surfaces as a clean OpenAI-shaped error, streaming
  (SSE error frame + [DONE]) and non-streaming (503 + Retry-After) alike
- non-streaming requests forward the upstream JSON and status
- /v1/models relays Ollama-0's OpenAI-format model list
"""
import contextlib
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


SSE_CHUNKS = [
    b'data: {"choices":[{"delta":{"content":"hel"}}]}\n\n',
    b'data: {"choices":[{"delta":{"content":"lo"}}]}\n\n',
    b"data: [DONE]\n\n",
]


class _FakeStreamResp:
    def __init__(self, chunks):
        self._chunks = chunks

    def raise_for_status(self):
        return None

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c


class _FakeStreamCM:
    def __init__(self, chunks):
        self._chunks = chunks

    async def __aenter__(self):
        return _FakeStreamResp(self._chunks)

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeAsyncClient:
    """Mimic ``httpx.AsyncClient`` for both .stream() and .post()/.get()."""

    calls: list = []

    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, method, url, json=None):  # noqa: A002
        _FakeAsyncClient.calls.append(("stream", method, url, json))
        return _FakeStreamCM(SSE_CHUNKS)

    async def post(self, url, json=None):  # noqa: A002
        _FakeAsyncClient.calls.append(("post", url, json))
        return _FakeResp({"id": "chatcmpl-1", "choices": []})

    async def get(self, url):
        _FakeAsyncClient.calls.append(("get", url))
        return _FakeResp({"object": "list",
                          "data": [{"id": "gpt-oss:20b-agent"}]})


class _FakeResp:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.content = json.dumps(payload).encode()
        self.headers = {"content-type": "application/json"}


@pytest.fixture
def app_mod(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "gpu_router",
                   "metrics", "notifications", "scheduler"):
            sys.modules.pop(mod, None)
    import app as m
    _FakeAsyncClient.calls = []
    monkeypatch.setattr(m.httpx, "AsyncClient", _FakeAsyncClient)
    return m


@pytest.fixture
def client(app_mod):
    from fastapi.testclient import TestClient
    return TestClient(app_mod.app, raise_server_exceptions=True)


def _wire_dispatch(app_mod, monkeypatch, target="http://fake-gpu:11434"):
    """Replace queue tracking/dispatch with recorders resolving to ``target``."""
    qm = app_mod.queue_manager
    tracked = []

    def fake_track(source, request_type, model, priority, batch_job_id=None):
        tracked.append({"source": source, "request_type": request_type,
                        "model": model, "priority": priority})
        return "tid-1"

    async def fake_wait(tid):
        return target

    @contextlib.asynccontextmanager
    async def fake_dispatched(tid):
        yield target

    monkeypatch.setattr(qm, "track_request", fake_track)
    monkeypatch.setattr(qm, "wait_for_dispatch", fake_wait)
    monkeypatch.setattr(qm, "dispatched", fake_dispatched)
    monkeypatch.setattr(qm, "release", lambda tid: None)
    monkeypatch.setattr(qm, "fail_request", lambda tid, err="": None)
    return tracked


# ── streaming ─────────────────────────────────────────────────────────────────

def test_stream_relays_sse_bytes_and_chat_priority(app_mod, client, monkeypatch):
    tracked = _wire_dispatch(app_mod, monkeypatch)
    r = client.post("/v1/chat/completions",
                    json={"model": "gpt-oss:20b", "stream": True,
                          "messages": [{"role": "user", "content": "hi"}]},
                    headers={"X-Priority": "chat", "X-Source": "hermes"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.content == b"".join(SSE_CHUNKS)
    assert tracked == [{"source": "hermes", "request_type": "chat_completions",
                        "model": "gpt-oss:20b",
                        "priority": app_mod.queue_manager.Priority.CHAT}]
    kind, method, url, body = _FakeAsyncClient.calls[0]
    assert (kind, method) == ("stream", "POST")
    assert url == "http://fake-gpu:11434/v1/chat/completions"
    assert body["stream"] is True


def test_missing_priority_falls_to_background(app_mod, client, monkeypatch):
    tracked = _wire_dispatch(app_mod, monkeypatch)
    client.post("/v1/chat/completions",
                json={"model": "m", "stream": True, "messages": []})
    assert tracked[0]["priority"] == app_mod.queue_manager.Priority.BACKGROUND
    assert tracked[0]["source"] == "direct"


def test_stream_dispatch_timeout_emits_openai_error(app_mod, client, monkeypatch):
    _wire_dispatch(app_mod, monkeypatch)
    qm = app_mod.queue_manager

    async def timeout_wait(tid):
        raise qm.DispatchTimeout("chat request exceeded INTERACTIVE_QUEUE_TIMEOUT")

    monkeypatch.setattr(qm, "wait_for_dispatch", timeout_wait)
    r = client.post("/v1/chat/completions",
                    json={"model": "m", "stream": True, "messages": []},
                    headers={"X-Priority": "chat"})
    assert r.status_code == 200  # headers already sent on a stream
    text = r.content.decode()
    frames = [ln for ln in text.split("\n") if ln.startswith("data: ")]
    err = json.loads(frames[0][len("data: "):])
    assert err["error"]["code"] == "gpu_queue_timeout"
    assert err["error"]["type"] == "server_error"
    assert frames[-1] == "data: [DONE]"


# ── non-streaming ─────────────────────────────────────────────────────────────

def test_nonstream_forwards_upstream_json(app_mod, client, monkeypatch):
    tracked = _wire_dispatch(app_mod, monkeypatch)
    r = client.post("/v1/chat/completions",
                    json={"model": "m", "messages": []},
                    headers={"X-Priority": "chat", "X-Source": "hermes"})
    assert r.status_code == 200
    assert r.json()["id"] == "chatcmpl-1"
    assert tracked[0]["request_type"] == "chat_completions"
    kind, url, body = _FakeAsyncClient.calls[0]
    assert kind == "post"
    assert url == "http://fake-gpu:11434/v1/chat/completions"


def test_nonstream_dispatch_timeout_is_503_openai_shape(app_mod, client,
                                                        monkeypatch):
    _wire_dispatch(app_mod, monkeypatch)
    qm = app_mod.queue_manager

    @contextlib.asynccontextmanager
    async def timeout_dispatched(tid):
        raise qm.DispatchTimeout("timed out")
        yield  # pragma: no cover

    monkeypatch.setattr(qm, "dispatched", timeout_dispatched)
    r = client.post("/v1/chat/completions",
                    json={"model": "m", "messages": []},
                    headers={"X-Priority": "chat"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "gpu_queue_timeout"
    assert "Retry-After" in r.headers


# ── /v1/models ────────────────────────────────────────────────────────────────

def test_v1_models_relays_ollama0(app_mod, client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert r.json()["data"][0]["id"] == "gpt-oss:20b-agent"
    kind, url = _FakeAsyncClient.calls[0]
    assert kind == "get"
    assert url.endswith("/v1/models")
    assert url.startswith(app_mod.config.OLLAMA_0_URL)
