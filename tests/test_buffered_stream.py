"""tests/test_buffered_stream.py — Tests for #64 buffer-stream path.

Covers ``_stream_and_accumulate`` and ``_buffered_stream_proxy``:
- wire response shape matches a ``stream=False`` Ollama call
- ``_activity_inc`` ticks once per NDJSON line so the instance card's
  tok counter updates mid-call for batch + extractor traffic.
"""
import asyncio
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


class _FakeStreamResp:
    """Mimic the object returned by ``httpx.AsyncClient.stream`` CM."""

    def __init__(self, lines: list[bytes]):
        # Split each NDJSON line across two chunks to exercise the
        # buffer-split path (mid-line chunk boundary).
        self._chunks = []
        for ln in lines:
            mid = max(1, len(ln) // 2)
            self._chunks.append(ln[:mid])
            self._chunks.append(ln[mid:] + b"\n")

    def raise_for_status(self):
        return None

    async def aiter_bytes(self):
        for c in self._chunks:
            yield c


class _FakeStreamCM:
    def __init__(self, lines: list[bytes]):
        self._lines = lines

    async def __aenter__(self):
        return _FakeStreamResp(self._lines)

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _FakeClient:
    """Mimic ``httpx.AsyncClient`` for the .stream() path used by accumulator."""

    def __init__(self, lines: list[bytes]):
        self._lines = lines
        self.stream_calls: list[tuple[str, str, dict]] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    def stream(self, method: str, url: str, json: dict):  # noqa: A002
        self.stream_calls.append((method, url, json))
        return _FakeStreamCM(self._lines)


@pytest.fixture
def app_mod(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "test.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "gpu_router",
                   "metrics", "notifications"):
            sys.modules.pop(mod, None)
    import app as app_mod
    return app_mod


def _run(coro):
    return asyncio.run(coro)


def test_accumulate_generate_wire_shape(app_mod, monkeypatch):
    """/api/generate: accumulated payload has .response + .done + metadata."""
    lines = [
        json.dumps({"response": "Hello "}).encode(),
        json.dumps({"response": "world"}).encode(),
        json.dumps({
            "response":           "",
            "done":               True,
            "done_reason":        "stop",
            "prompt_eval_count":  7,
            "eval_count":         3,
        }).encode(),
    ]
    fake = _FakeClient(lines)
    monkeypatch.setattr(app_mod.httpx, "AsyncClient", lambda **kw: fake)

    payload = _run(app_mod._stream_and_accumulate(
        "http://fake:11434", "/api/generate", {"model": "m", "prompt": "p"}
    ))

    assert payload["response"] == "Hello world"
    assert payload["done"] is True
    assert payload["done_reason"] == "stop"
    assert payload["prompt_eval_count"] == 7
    # Wire body flipped stream=True even though caller body had no flag.
    assert fake.stream_calls[0][2]["stream"] is True


def test_accumulate_chat_wire_shape(app_mod, monkeypatch):
    """/api/chat: accumulated payload has message.content + thinking + done."""
    lines = [
        json.dumps({"message": {"role": "assistant", "content": "Sure, "}}).encode(),
        json.dumps({"message": {"role": "assistant", "thinking": "planning..."}}).encode(),
        json.dumps({"message": {"role": "assistant", "content": "here:"}}).encode(),
        json.dumps({
            "message":    {"role": "assistant", "content": ""},
            "done":       True,
            "done_reason": "stop",
        }).encode(),
    ]
    fake = _FakeClient(lines)
    monkeypatch.setattr(app_mod.httpx, "AsyncClient", lambda **kw: fake)

    payload = _run(app_mod._stream_and_accumulate(
        "http://fake:11434", "/api/chat",
        {"model": "m", "messages": [{"role": "user", "content": "hi"}], "stream": False},
    ))

    assert payload["message"]["role"] == "assistant"
    assert payload["message"]["content"] == "Sure, here:"
    assert payload["message"]["thinking"] == "planning..."
    assert payload["done"] is True
    # Caller's stream=False is overridden on the wire.
    assert fake.stream_calls[0][2]["stream"] is True


def test_accumulate_ticks_activity_per_line(app_mod, monkeypatch):
    """Every non-empty NDJSON line increments the instance card's chunk counter."""
    lines = [
        json.dumps({"response": "a"}).encode(),
        json.dumps({"response": "b"}).encode(),
        json.dumps({"response": "c"}).encode(),
        json.dumps({"response": "", "done": True}).encode(),
    ]
    fake = _FakeClient(lines)
    monkeypatch.setattr(app_mod.httpx, "AsyncClient", lambda **kw: fake)

    target = app_mod.config.OLLAMA_0_URL
    app_mod._activity_set(target, "m", "/api/generate")
    try:
        _run(app_mod._stream_and_accumulate(
            target, "/api/generate", {"model": "m", "prompt": "p"}
        ))
        entry = app_mod._activity[app_mod._gpu_label(target)]
        # 4 non-empty NDJSON lines → chunks == 4.
        assert entry is not None
        assert entry["chunks"] == 4
        # Rolling text buffer fed from the per-line append.
        assert entry.get("text", "").endswith("abc")
    finally:
        app_mod._activity_clear(target)


def test_buffered_stream_proxy_returns_json_response(app_mod, monkeypatch):
    """_buffered_stream_proxy serializes the accumulated dict to a JSON Response."""
    lines = [
        json.dumps({"response": "ok"}).encode(),
        json.dumps({"response": "", "done": True, "done_reason": "stop"}).encode(),
    ]
    fake = _FakeClient(lines)
    monkeypatch.setattr(app_mod.httpx, "AsyncClient", lambda **kw: fake)

    resp = _run(app_mod._buffered_stream_proxy(
        "http://fake:11434", "/api/generate", {"model": "m", "prompt": "p"}
    ))

    assert resp.media_type == "application/json"
    body = json.loads(bytes(resp.body))
    assert body["response"] == "ok"
    assert body["done"] is True
    # _activity_clear ran in finally → card shows idle.
    assert app_mod._activity[app_mod._gpu_label("http://fake:11434")] is None
