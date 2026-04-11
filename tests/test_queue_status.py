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
        if mod in ("app", "db", "config", "queue_manager", "gpu_router",
                   "metrics"):
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
        if mod in ("app", "db", "config", "queue_manager", "gpu_router",
                   "metrics"):
            sys.modules.pop(mod, None)

    import app as app_mod
    import gpu_router
    from fastapi.testclient import TestClient

    # Stub model swap so the dispatcher does not try to reach a real Ollama.
    async def _noop_reload(gpu, model):
        gpu.model = model
    monkeypatch.setattr(gpu_router, "_reload_model", _noop_reload)

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
    """When all GPU slots are held, queue_status NDJSON must appear before tokens."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "merllm.db"))
    monkeypatch.setenv("INTERACTIVE_QUEUE_TIMEOUT", "5")
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "gpu_router",
                   "metrics"):
            sys.modules.pop(mod, None)

    import queue_manager as qm
    import config
    import app as app_mod
    from fastapi.testclient import TestClient

    # Patch gpu_slot_busy so every GPU reports as busy (simulating contention).
    monkeypatch.setattr(qm, "gpu_slot_busy", lambda t: True)

    # Stub the dispatcher so the request proceeds immediately to streaming.
    target = config.OLLAMA_0_URL

    async def _instant_dispatch(tid):
        entry = qm._tracked.get(tid)
        if entry is not None:
            entry.target = target
            entry.status = "running"
        return target

    monkeypatch.setattr(qm, "wait_for_dispatch", _instant_dispatch)
    monkeypatch.setattr(qm, "release", lambda tid=None: None)

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


# ── queue-wait heartbeat ──────────────────────────────────────────────────────


def test_queue_wait_emits_heartbeats_during_long_wait(tmp_path, monkeypatch):
    """During a queue wait longer than QUEUE_HEARTBEAT_INTERVAL_SECONDS, the
    streaming proxy must emit periodic keepalive NDJSON chunks so a caller
    with a between-chunk read-gap timeout shorter than the wait does not
    disconnect.

    Regression: 2026-04-11 feedback-slot wedge. parsival's
    ``requests.post(timeout=60, stream=True)`` was tripping its read-gap
    timer during BACKGROUND-drain queue waits because merllm emitted a
    single ``queue_status`` line at t=0 and then nothing until the slot
    was popped. The downstream consequence was a GeneratorExit at
    ``await wait_for_dispatch`` that leaked the pending entry and wedged
    the GPU slot via a dead waiter.
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "merllm.db"))
    monkeypatch.setenv("QUEUE_HEARTBEAT_INTERVAL_SECONDS", "1")
    monkeypatch.setenv("INTERACTIVE_QUEUE_TIMEOUT", "30")
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "gpu_router",
                   "metrics"):
            sys.modules.pop(mod, None)

    import queue_manager as qm
    import config
    import app as app_mod
    from fastapi.testclient import TestClient

    monkeypatch.setattr(qm, "gpu_slot_busy", lambda t: True)

    target = config.OLLAMA_0_URL

    async def _slow_dispatch(tid):
        # Sleep long enough that at least two heartbeat intervals elapse
        # before the dispatcher resolves.
        await asyncio.sleep(2.5)
        entry = qm._tracked.get(tid)
        if entry is not None:
            entry.target = target
            entry.status = "running"
        return target

    monkeypatch.setattr(qm, "wait_for_dispatch", _slow_dispatch)
    monkeypatch.setattr(qm, "release", lambda tid=None: None)
    monkeypatch.setattr(qm, "cancel_tracked", lambda tid, reason="": None)

    done_chunk = json.dumps({"done": True, "prompt_eval_count": 5}).encode() + b"\n"
    monkeypatch.setattr(app_mod.httpx, "AsyncClient",
                        _make_stream_client(done_chunk))

    client = TestClient(app_mod.app, raise_server_exceptions=True)
    resp = client.post("/api/generate",
                       json={"model": "test", "prompt": "hi", "stream": True})

    assert resp.status_code == 200
    lines = _ndjson_lines(resp.content)
    heartbeats = [l for l in lines
                  if l.get("type") == "queue_status" and l.get("waiting") is True]
    assert len(heartbeats) >= 2, (
        f"Expected ≥2 waiting heartbeats during a 2.5s wait with 1s "
        f"interval, got {len(heartbeats)}: {lines}"
    )
    # Each heartbeat must expose elapsed_seconds so a debug UI can render
    # a growing wait indicator.
    assert all("elapsed_seconds" in h for h in heartbeats)
    # Heartbeat elapsed_seconds must be monotonically non-decreasing.
    elapsed = [h["elapsed_seconds"] for h in heartbeats]
    assert elapsed == sorted(elapsed), f"heartbeats not monotonic: {elapsed}"


# ── batch queue_position ──────────────────────────────────────────────────────


def test_batch_queue_position_in_status(tmp_path, monkeypatch):
    """Queued batch jobs must report their queue_position in status."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "merllm.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "gpu_router",
                   "metrics"):
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


# ── Reasoning-model num_ctx pinning on the proxy path ────────────────────────


def test_normalize_reasoning_body_pins_num_ctx_for_qwen3(tmp_path, monkeypatch):
    """qwen3:* proxy requests must get num_ctx pinned if the caller omitted it.

    Regression: 2026-04-11. The dispatcher's _reload_model already pinned
    num_ctx, but if the dispatcher's GpuState already thought the right
    model was loaded the warm-load was skipped and the user request hit
    Ollama unmodified — at which point Ollama's auto-fit picked the
    model's training maximum (32k), forcing CPU offload of the output
    layer and turning sub-second loads into 12 s loads.
    """
    monkeypatch.setenv("DB_PATH", str(tmp_path / "merllm.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "gpu_router",
                   "metrics"):
            sys.modules.pop(mod, None)
    import app as app_mod

    body = {"model": "qwen3:32b", "prompt": "hi"}
    app_mod._normalize_reasoning_body(body)
    assert body["options"]["num_ctx"] == 8192


def test_normalize_reasoning_body_respects_explicit_num_ctx(tmp_path, monkeypatch):
    """If the caller already pinned num_ctx, the proxy must not overwrite it."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "merllm.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "gpu_router",
                   "metrics"):
            sys.modules.pop(mod, None)
    import app as app_mod

    body = {"model": "qwen3:32b", "prompt": "hi", "options": {"num_ctx": 16384}}
    app_mod._normalize_reasoning_body(body)
    assert body["options"]["num_ctx"] == 16384


def test_normalize_reasoning_body_skips_non_qwen3(tmp_path, monkeypatch):
    """Models outside the qwen3:* family must not have options injected."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "merllm.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "gpu_router",
                   "metrics"):
            sys.modules.pop(mod, None)
    import app as app_mod

    body = {"model": "llama3:8b", "prompt": "hi"}
    app_mod._normalize_reasoning_body(body)
    assert "options" not in body


def test_completed_job_has_no_queue_position(tmp_path, monkeypatch):
    """Completed jobs must not include queue_position in their status."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "merllm.db"))
    for mod in list(sys.modules.keys()):
        if mod in ("app", "db", "config", "queue_manager", "gpu_router",
                   "metrics"):
            sys.modules.pop(mod, None)
    import queue_manager as qm
    import db
    job_id = qm.submit_batch_job("test", "model", "hello")
    db.update_batch_job(job_id, status="completed", completed_at=time.time(), result="ok")
    status = qm.get_job_status(job_id)
    assert "queue_position" not in status
