"""Tests for B8 — transparent wait experience.

Covers:
 - queue_status NDJSON emitted before tokens when GPU slot is busy
 - context_tokens injected into the done line
 - queue_position field in batch job status
"""
import asyncio
import json
import os
import sys
import time

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "api"))


# ── Helpers ───────────────────────────────────────────────────────────────────


def _ndjson_lines(content: bytes) -> list[dict]:
    """Parse all non-empty NDJSON lines from a response body."""
    lines = []
    for raw in content.splitlines():
        raw = raw.strip()
        if raw:
            try:
                lines.append(json.loads(raw))
            except Exception:
                pass
    return lines


def _fresh_app(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "merllm.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "mode_manager",
                   "metrics", "geoip"):
            sys.modules.pop(mod, None)
    import app as app_mod
    from fastapi.testclient import TestClient
    return TestClient(app_mod.app, raise_server_exceptions=True)


# ── Mock httpx streaming client ───────────────────────────────────────────────


def _make_stream_client(content: bytes):
    """Build a mock httpx.AsyncClient that streams the given bytes."""

    class _FakeStream:
        def __init__(self):
            self._content = content

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        async def aiter_bytes(self):
            yield self._content

    class _FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def stream(self, *args, **kwargs):
            return _FakeStream()

    return lambda **kwargs: _FakeClient()


# ── context_tokens injection ──────────────────────────────────────────────────


def test_context_tokens_injected_into_done_line(tmp_path, monkeypatch):
    """The done NDJSON line must carry context_tokens = prompt_eval_count."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "merllm.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "mode_manager",
                   "metrics", "geoip"):
            sys.modules.pop(mod, None)

    import app as app_mod
    from fastapi.testclient import TestClient

    done_chunk = json.dumps({
        "model": "test", "created_at": "2026-01-01T00:00:00Z",
        "message": {"role": "assistant", "content": ""},
        "done": True, "done_reason": "stop",
        "prompt_eval_count": 128, "eval_count": 32,
    }).encode() + b"\n"
    token_chunk = json.dumps({
        "message": {"role": "assistant", "content": "hello"},
        "done": False,
    }).encode() + b"\n"

    monkeypatch.setattr(app_mod.httpx, "AsyncClient",
                        _make_stream_client(token_chunk + done_chunk))

    client = TestClient(app_mod.app, raise_server_exceptions=True)
    resp = client.post("/api/chat", json={"model": "test", "messages": [], "stream": True})

    assert resp.status_code == 200
    lines = _ndjson_lines(resp.content)
    done_line = next((l for l in lines if l.get("done")), None)
    assert done_line is not None, "No done line in response"
    assert done_line.get("context_tokens") == 128


# ── queue_status on busy slot ─────────────────────────────────────────────────


def test_queue_status_emitted_when_slot_busy(tmp_path, monkeypatch):
    """When the GPU slot is held, queue_status NDJSON must appear before tokens."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "merllm.db"))
    monkeypatch.setenv("INTERACTIVE_QUEUE_TIMEOUT", "5")
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "mode_manager",
                   "metrics", "geoip"):
            sys.modules.pop(mod, None)

    import queue_manager as qm
    import app as app_mod
    from fastapi.testclient import TestClient

    target = "http://host.docker.internal:11434"
    # Patch gpu_slot_busy so it reports the slot as busy (simulating contention).
    monkeypatch.setattr(qm, "gpu_slot_busy", lambda t: True)
    # acquire_gpu_slot should succeed immediately so the stream proceeds.
    async def _instant_acquire(t, p):
        return True
    monkeypatch.setattr(qm, "acquire_gpu_slot", _instant_acquire)
    monkeypatch.setattr(qm, "release_gpu_slot", lambda t: None)

    done_chunk = json.dumps({"done": True, "prompt_eval_count": 10}).encode() + b"\n"
    monkeypatch.setattr(app_mod.httpx, "AsyncClient",
                        _make_stream_client(done_chunk))

    client = TestClient(app_mod.app, raise_server_exceptions=True)
    resp = client.post("/api/generate", json={"model": "test", "prompt": "hi", "stream": True})

    lines = _ndjson_lines(resp.content)
    types = [l.get("type") for l in lines]
    assert "queue_status" in types, f"Expected queue_status in {types}"
    qs = next(l for l in lines if l.get("type") == "queue_status")
    assert qs.get("reason")
    assert qs.get("estimated_wait_seconds") is not None


# ── batch queue_position ──────────────────────────────────────────────────────


def test_batch_queue_position_in_status(tmp_path, monkeypatch):
    """Queued batch jobs must report their queue_position in status."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "merllm.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "mode_manager",
                   "metrics", "geoip"):
            sys.modules.pop(mod, None)
    import queue_manager as qm
    id1 = qm.submit_batch_job("test", "model", "first")
    id2 = qm.submit_batch_job("test", "model", "second")
    id3 = qm.submit_batch_job("test", "model", "third")

    s1 = qm.get_job_status(id1)
    s2 = qm.get_job_status(id2)
    s3 = qm.get_job_status(id3)

    assert s1["queue_position"] == 0
    assert s2["queue_position"] == 1
    assert s3["queue_position"] == 2
    assert "estimated_start" in s1


def test_completed_job_has_no_queue_position(tmp_path, monkeypatch):
    """Completed jobs must not include queue_position in their status."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "merllm.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "mode_manager",
                   "metrics", "geoip"):
            sys.modules.pop(mod, None)
    import queue_manager as qm
    import db
    job_id = qm.submit_batch_job("test", "model", "hello")
    db.update_batch_job(job_id, status="completed", completed_at=time.time(), result="ok")
    status = qm.get_job_status(job_id)
    assert "queue_position" not in status
