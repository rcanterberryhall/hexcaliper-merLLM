"""
app.py — merLLM: centralized LLM traffic control for the Hexcaliper ecosystem.

Exposes a drop-in Ollama API proxy on :11400. Both LanceLLMot and Parsival
point their OLLAMA_BASE_URL here. merLLM round-robins requests across both
GPU-pinned Ollama instances with priority queuing and health tracking.

Additional endpoints:
  GET/POST  /api/merllm/status    — GPU state, queue depth, health
  GET/POST  /api/merllm/settings  — view/update configuration
  GET       /api/merllm/default-model — current default model
  POST      /api/merllm/gpu/{gpu}/reset — manual GPU health reset
  GET       /api/merllm/activity  — live per-instance request state + loaded models
  POST      /api/batch/submit     — queue a batch job (runs at low priority)
  GET       /api/batch/status     — poll job status
  GET       /api/batch/results/{id} — retrieve completed output
  GET       /api/merllm/metrics/current  — latest system snapshot
  GET       /api/merllm/metrics/history  — time-series data
  GET       /api/merllm/diagnostics      — connectivity + container health
  GET       /api/merllm/logs/{service}   — recent log lines
  WS        /ws/ssh               — browser SSH terminal (xterm.js)
  WS        /ws/vnc               — browser VNC viewer (noVNC)
"""
import asyncio
import json
import logging
import os
import sqlite3
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

# Configure root logger so all merLLM modules are visible in container logs
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
# Suppress noisy httpx request-level logging (merLLM logs its own proxy calls)
logging.getLogger("httpx").setLevel(logging.WARNING)

_req_log = logging.getLogger("merllm.requests")
log = logging.getLogger("merllm")

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
import db
import gpu_router
import metrics
import notifications
import queue_manager

# ── Per-instance activity tracking ───────────────────────────────────────────
# Keyed by "gpu0" / "gpu1". Each entry is either None (idle) or a dict with:
#   model, endpoint, started_at (epoch float), chunks (NDJSON lines received),
#   text (rolling token buffer, last _TEXT_TAIL chars).
# Single-threaded asyncio — no lock needed.
_activity: dict[str, dict | None] = {"gpu0": None, "gpu1": None}

# ── SSE push for real-time activity updates ────────────────────────────────
# One asyncio.Queue per connected SSE client. Updated on every token.
_activity_sse_queues: list[asyncio.Queue] = []
_last_sse_push: float = 0.0
_SSE_PUSH_MIN_INTERVAL = 0.1  # max 10 pushes/sec during active generation


def _push_activity_sse(force: bool = False) -> None:
    """Push current activity snapshot to all connected SSE clients.

    Rate-limited to _SSE_PUSH_MIN_INTERVAL seconds between pushes unless
    force=True (used on request start/end so those transitions are always seen).
    """
    global _last_sse_push
    if not _activity_sse_queues:
        return
    now = time.time()
    if not force and now - _last_sse_push < _SSE_PUSH_MIN_INTERVAL:
        return
    _last_sse_push = now
    data = json.dumps(_activity_snapshot())
    for q in list(_activity_sse_queues):
        try:
            q.put_nowait(data)
        except asyncio.QueueFull:
            pass


def _gpu_label(url: str) -> str:
    return "gpu0" if url == config.OLLAMA_0_URL else "gpu1"


def _activity_set(url: str, model: str, endpoint: str) -> None:
    _activity[_gpu_label(url)] = {
        "model":      model,
        "endpoint":   endpoint,
        "started_at": time.time(),
        "chunks":     0,
    }
    _push_activity_sse(force=True)


def _activity_inc(url: str) -> None:
    entry = _activity.get(_gpu_label(url))
    if entry is not None:
        entry["chunks"] += 1
    _push_activity_sse()


def _activity_clear(url: str) -> None:
    _activity[_gpu_label(url)] = None
    _push_activity_sse(force=True)


_TEXT_TAIL = 600   # characters kept in the rolling text buffer


def _activity_append_token(url: str, line: bytes, path: str) -> None:
    """Parse one NDJSON line and append its token text to the rolling buffer."""
    entry = _activity.get(_gpu_label(url))
    if entry is None:
        return
    try:
        obj = json.loads(line)
        if obj.get("done"):
            return
        if path == "/api/generate":
            token = obj.get("response", "")
        elif path == "/api/chat":
            token = obj.get("message", {}).get("content", "")
        else:
            return
        if token:
            entry["text"] = (entry.get("text", "") + token)[-_TEXT_TAIL:]
    except Exception:
        pass


def _activity_snapshot() -> dict:
    now = time.time()
    result = {}
    for key, entry in _activity.items():
        if entry is None:
            result[key] = None
        else:
            result[key] = {**entry, "elapsed_sec": round(now - entry["started_at"], 1)}
    return result

# ── App lifecycle ─────────────────────────────────────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load persisted settings
    saved = db.get_settings()
    if saved:
        config.apply_overrides(saved)

    # Startup diagnostics: database integrity
    try:
        integrity = db.conn().execute("PRAGMA integrity_check").fetchone()
        if integrity and integrity[0] != "ok":
            log.error("database integrity check failed: %s", integrity[0])
        else:
            log.info("database integrity check passed")
    except Exception as exc:
        log.error("database integrity check error: %s", exc)

    # Startup diagnostics: check Ollama instances
    for name, url in [("gpu0", config.OLLAMA_0_URL), ("gpu1", config.OLLAMA_1_URL)]:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(5.0)) as client:
                r = await client.get(f"{url}/api/tags")
                models = len(r.json().get("models", []))
                log.info("Ollama %s reachable (%d models)", name, models)
        except Exception as exc:
            log.warning("Ollama %s unreachable at startup: %s", name, exc)

    # Recover any jobs that were running when the process was last killed
    recovered = db.requeue_orphaned_jobs()
    if recovered:
        log.info("recovered %d orphaned job(s) → queued", recovered)

    # Background tasks
    asyncio.create_task(metrics.collection_loop())
    asyncio.create_task(queue_manager.worker_loop())
    asyncio.create_task(gpu_router.reclaim_loop())

    log.info("started — routing=%s, default_model=%s",
             "round_robin", config.DEFAULT_MODEL)
    yield
    log.info("shutting down")


app = FastAPI(title="merLLM", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def request_logging(request: Request, call_next):
    """Log every HTTP request with method, path, status, duration, and user."""
    start = time.monotonic()
    response = await call_next(request)
    ms = int((time.monotonic() - start) * 1000)
    user = request.headers.get("CF-Access-Authenticated-User-Email", "anonymous")
    status = response.status_code
    msg = "%s %s %d %dms [%s]", request.method, request.url.path, status, ms, user
    if status >= 500:
        _req_log.error(*msg)
    elif status >= 400:
        _req_log.warning(*msg)
    else:
        _req_log.info(*msg)
    return response


# Static web UI
_web_dir = os.path.join(os.path.dirname(__file__), "..", "web")
if os.path.isdir(_web_dir):
    app.mount("/web", StaticFiles(directory=_web_dir, html=True), name="web")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _client_ip(request: Request) -> str:
    return (
        request.headers.get("cf-connecting-ip") or
        request.headers.get("x-real-ip") or
        (request.client.host if request.client else "127.0.0.1")
    )


def _priority(request: Request) -> int:
    h = request.headers.get("x-priority", "interactive").lower()
    return queue_manager.PRIORITY_BATCH if h == "batch" else queue_manager.PRIORITY_INTERACTIVE


async def _generate_stream(target: str, path: str, body: dict, on_done=None):
    """
    Async generator that streams NDJSON from Ollama, line by line.

    Injects ``context_tokens`` (= prompt_eval_count) into the final done line
    so callers can display context-window utilisation.

    on_done: optional zero-argument callable called in the finally block.
    """
    buf = b""
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
            async with client.stream("POST", f"{target}{path}", json=body) as resp:
                async for chunk in resp.aiter_bytes():
                    buf += chunk
                    while b"\n" in buf:
                        line, buf = buf.split(b"\n", 1)
                        if line.strip():
                            _activity_inc(target)
                            _activity_append_token(target, line, path)
                            # Inject context_tokens into the done line.
                            try:
                                obj = json.loads(line)
                                if obj.get("done"):
                                    pec = obj.get("prompt_eval_count")
                                    if pec is not None:
                                        obj["context_tokens"] = pec
                                    line = json.dumps(obj).encode()
                            except Exception:
                                pass
                            yield line + b"\n"
                        else:
                            yield b"\n"
    finally:
        _activity_clear(target)
        if on_done:
            on_done()


async def _stream_proxy(target: str, path: str, body: dict,
                        on_done=None) -> StreamingResponse:
    """Stream an Ollama response, tracking activity per instance.

    on_done: optional zero-argument callable invoked in the generator's finally
    block, after the stream ends. Use this to release GPU slots or other
    resources that must be held for the full duration of streaming.
    """
    model = body.get("model", "")
    _activity_set(target, model, path)
    return StreamingResponse(_generate_stream(target, path, body, on_done),
                             media_type="application/x-ndjson")


def _queue_reason(target: str) -> str:
    """Human-readable reason a request is waiting for a GPU slot."""
    in_flight = queue_manager.queue_depth()["gpus"].get(target, {}).get("in_flight", 0)
    if in_flight:
        return "GPU slot occupied by another request"
    return "GPU busy"


async def _queued_stream(target: str, path: str, body: dict,
                         priority: int) -> StreamingResponse:
    """
    Streaming proxy that emits a ``queue_status`` NDJSON line if the GPU slot
    is not immediately available, giving the caller transparent visibility into
    queue state before generation begins.
    """
    model = body.get("model", "")
    _activity_set(target, model, path)

    async def generate():
        # If the slot is busy, tell the caller before we start waiting.
        if queue_manager.gpu_slot_busy(target):
            reason = _queue_reason(target)
            est    = config.INTERACTIVE_QUEUE_TIMEOUT
            yield json.dumps({
                "type":                    "queue_status",
                "reason":                  reason,
                "estimated_wait_seconds":  est,
            }).encode() + b"\n"

        acquired = await queue_manager.acquire_gpu_slot(target, priority)
        if not acquired:
            yield json.dumps({
                "error": "GPU busy — timed out waiting for slot",
                "done":  True,
            }).encode() + b"\n"
            _activity_clear(target)
            return

        on_done = lambda: queue_manager.release_gpu_slot(target)
        async for chunk in _generate_stream(target, path, body, on_done):
            yield chunk

    return StreamingResponse(generate(), media_type="application/x-ndjson")


async def _proxy(target: str, path: str, body: dict) -> Response:
    """Non-streaming Ollama proxy (e.g. embeddings), tracking activity per instance."""
    model = body.get("model", "")
    _activity_set(target, model, path)
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
            resp = await client.post(f"{target}{path}", json=body)
        return Response(content=resp.content,
                        media_type=resp.headers.get("content-type", "application/json"),
                        status_code=resp.status_code)
    finally:
        _activity_clear(target)


async def _proxy_get(target: str, path: str) -> Response:
    async with httpx.AsyncClient(timeout=30) as client:
        resp = await client.get(f"{target}{path}")
    return Response(content=resp.content,
                    media_type=resp.headers.get("content-type", "application/json"),
                    status_code=resp.status_code)


# ── Ollama API proxy ──────────────────────────────────────────────────────────


@app.post("/api/generate")
async def proxy_generate(request: Request):
    """Proxy POST /api/generate — round-robin routed across GPUs."""
    body = await request.json()
    model = body.get("model", "")
    priority = _priority(request)

    target = gpu_router.get_target_url(model)
    stream = body.get("stream", True)

    if stream:
        return await _queued_stream(target, "/api/generate", body, priority)

    acquired = await queue_manager.acquire_gpu_slot(target, priority)
    if not acquired:
        return JSONResponse(
            status_code=503,
            content={
                "error": "GPU busy — all slots occupied",
                "estimated_wait_seconds": config.INTERACTIVE_QUEUE_TIMEOUT,
            },
            headers={"Retry-After": str(config.INTERACTIVE_QUEUE_TIMEOUT)},
        )
    try:
        resp = await _proxy(target, "/api/generate", body)
        gpu_router.record_activity(target)
        return resp
    finally:
        queue_manager.release_gpu_slot(target)


@app.post("/api/chat")
async def proxy_chat(request: Request):
    """Proxy POST /api/chat — round-robin routed across GPUs."""
    body = await request.json()
    model = body.get("model", "")
    priority = _priority(request)

    target = gpu_router.get_target_url(model)
    stream = body.get("stream", True)

    if stream:
        return await _queued_stream(target, "/api/chat", body, priority)

    acquired = await queue_manager.acquire_gpu_slot(target, priority)
    if not acquired:
        return JSONResponse(
            status_code=503,
            content={
                "error": "GPU busy — all slots occupied",
                "estimated_wait_seconds": config.INTERACTIVE_QUEUE_TIMEOUT,
            },
            headers={"Retry-After": str(config.INTERACTIVE_QUEUE_TIMEOUT)},
        )
    try:
        resp = await _proxy(target, "/api/chat", body)
        gpu_router.record_activity(target)
        return resp
    finally:
        queue_manager.release_gpu_slot(target)


@app.post("/api/embeddings")
async def proxy_embeddings(request: Request):
    """Proxy POST /api/embeddings — round-robin routed across GPUs."""
    body = await request.json()
    model = body.get("model", "")
    target = gpu_router.get_target_url(model)
    resp = await _proxy(target, "/api/embeddings", body)
    gpu_router.record_activity(target)
    return resp


@app.get("/api/tags")
async def proxy_tags():
    """Return model list from GPU 0 (both instances share the same model store)."""
    return await _proxy_get(config.OLLAMA_0_URL, "/api/tags")


@app.post("/api/show")
async def proxy_show(request: Request):
    body = await request.json()
    return await _proxy(config.OLLAMA_0_URL, "/api/show", body)


@app.post("/api/pull")
async def proxy_pull(request: Request):
    """Proxy model pull to GPU 0 (shared model store)."""
    body = await request.json()
    stream = body.get("stream", True)
    if stream:
        return await _stream_proxy(config.OLLAMA_0_URL, "/api/pull", body)
    return await _proxy(config.OLLAMA_0_URL, "/api/pull", body)


@app.get("/api/ps")
async def proxy_ps():
    """Aggregate loaded models from both Ollama instances."""
    results = {"models": []}
    for url in [config.OLLAMA_0_URL, config.OLLAMA_1_URL]:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(f"{url}/api/ps")
                if r.status_code == 200:
                    results["models"].extend(r.json().get("models", []))
        except Exception:
            pass
    return results


# ── merLLM status and control ─────────────────────────────────────────────────


@app.get("/api/merllm/status")
async def merllm_status():
    """
    Return current mode, GPU layout, queue depth, Ollama instance health,
    and active alert flags.
    """
    ollama_health = {}
    for label, url in [("gpu0", config.OLLAMA_0_URL), ("gpu1", config.OLLAMA_1_URL)]:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"{url}/api/tags")
                ollama_health[label] = {"ok": r.status_code == 200, "url": url}
        except Exception as exc:
            ollama_health[label] = {"ok": False, "url": url, "error": str(exc)}

    latest = db.get_latest_metrics()

    return {
        **gpu_router.status(),
        "queue":         queue_manager.queue_depth(),
        "ollama":        ollama_health,
        "gpu_metrics":   metrics.gpu_snapshot(),
        "batch_counts":  db.count_batch_jobs_by_status(),
        "warnings":      _build_warnings(ollama_health, latest),
    }


def _build_warnings(ollama_health: dict, latest: dict) -> list[str]:
    warnings = []
    for label, h in ollama_health.items():
        if not h["ok"]:
            warnings.append(f"Ollama {label} unreachable: {h.get('error', 'unknown')}")
    ram_total = latest.get("ram.total", {}).get("value", 0)
    ram_avail = latest.get("ram.available", {}).get("value", 0)
    if ram_total and ram_avail / ram_total < 0.10:
        warnings.append("RAM available < 10%")
    for i in range(4):
        temp = latest.get(f"gpu{i}.temp", {}).get("value")
        if temp and temp > 85:
            warnings.append(f"GPU {i} temperature {temp:.0f}°C — thermal throttle risk")
    return warnings


@app.get("/api/merllm/activity")
async def merllm_activity():
    """
    Per-instance live activity state: what model is running, on which endpoint,
    for how long, and how many response chunks have been received so far.
    Also fetches loaded models from each instance via /api/ps.
    """
    snapshot = _activity_snapshot()

    loaded: dict[str, list[dict] | None] = {}
    for label, url in [("gpu0", config.OLLAMA_0_URL), ("gpu1", config.OLLAMA_1_URL)]:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"{url}/api/ps")
                if r.status_code == 200:
                    models = r.json().get("models", [])
                    loaded[label] = [
                        {
                            "name":        m.get("name", ""),
                            "size_vram_mb": round(m.get("size_vram", 0) / 1024 ** 2),
                            "expires_at":  m.get("expires_at"),
                        }
                        for m in models
                    ]
                else:
                    loaded[label] = []
        except Exception:
            loaded[label] = None  # instance unreachable

    return {
        "gpu0": {
            "url":    config.OLLAMA_0_URL,
            "loaded": loaded.get("gpu0"),
            "active": snapshot.get("gpu0"),
        },
        "gpu1": {
            "url":    config.OLLAMA_1_URL,
            "loaded": loaded.get("gpu1"),
            "active": snapshot.get("gpu1"),
        },
    }


@app.get("/api/merllm/activity/stream")
async def activity_stream():
    """
    Server-sent events stream of per-instance activity.

    Pushes a snapshot whenever a token arrives (rate-limited to 10/sec)
    and immediately on request start/end. Keepalive comment sent every 5s
    when idle. The frontend subscribes to this instead of polling.
    """
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    _activity_sse_queues.append(q)

    async def generate():
        try:
            while True:
                try:
                    data = await asyncio.wait_for(q.get(), timeout=5.0)
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            try:
                _activity_sse_queues.remove(q)
            except ValueError:
                pass

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/merllm/default-model")
async def default_model():
    """Return the current default model for client apps to query."""
    return {"model": config.DEFAULT_MODEL}


@app.post("/api/merllm/gpu/{gpu}/reset")
async def reset_gpu(gpu: str):
    """Manually reset a faulted GPU to healthy."""
    url_map = {"gpu0": config.OLLAMA_0_URL, "gpu1": config.OLLAMA_1_URL}
    url = url_map.get(gpu)
    if not url:
        raise HTTPException(status_code=422, detail="gpu must be 'gpu0' or 'gpu1'")
    ok = gpu_router.reset_gpu(url)
    if not ok:
        raise HTTPException(status_code=404, detail="GPU not found")
    return {"ok": True, **gpu_router.status()}


@app.get("/api/merllm/settings")
async def get_settings():
    return {
        "ollama_0_url":              config.OLLAMA_0_URL,
        "ollama_1_url":              config.OLLAMA_1_URL,
        "default_model":             config.DEFAULT_MODEL,
        "reclaim_timeout":           config.RECLAIM_TIMEOUT,
        "health_backoff_base":       config.HEALTH_BACKOFF_BASE,
        "health_backoff_cap":        config.HEALTH_BACKOFF_CAP,
        "health_fault_timeout":      config.HEALTH_FAULT_TIMEOUT,
        "metrics_interval_sec":      config.METRICS_INTERVAL_SEC,
        "notification_webhook_url":  config.NOTIFICATION_WEBHOOK_URL,
    }


@app.post("/api/merllm/settings")
async def save_settings(request: Request):
    body = await request.json()

    # DEFAULT_MODEL change requires explicit confirmation.
    new_model = body.get("default_model")
    if new_model and new_model != config.DEFAULT_MODEL:
        if not body.get("confirm_model_change"):
            return JSONResponse(
                status_code=409,
                content={
                    "ok": False,
                    "error": "Changing default_model will reload both GPUs when idle. "
                             "Resend with confirm_model_change: true to proceed.",
                },
            )
        gpu_router.set_pending_default_model(new_model)

    db.save_settings(body)
    config.apply_overrides(body)
    return {"ok": True}


# ── Batch API ─────────────────────────────────────────────────────────────────


@app.post("/api/batch/submit")
async def batch_submit(request: Request):
    """
    Queue a generation request for batch processing (runs at low priority).

    Body: {
      "source_app": "parsival" | "lancellmot" | ...,
      "prompt": "...",
      "model": "qwen3:32b",       # optional, defaults to DEFAULT_MODEL
      "options": {}               # optional Ollama options
    }
    """
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt is required")
    if len(prompt) > config.BATCH_MAX_PROMPT_LEN:
        raise HTTPException(
            status_code=422,
            detail=f"Prompt exceeds maximum length of {config.BATCH_MAX_PROMPT_LEN} characters "
                   f"({len(prompt)} submitted). Set BATCH_MAX_PROMPT_LEN to override.",
        )

    job_id = queue_manager.submit_batch_job(
        source_app=body.get("source_app", "unknown"),
        model=body.get("model", config.DEFAULT_MODEL),
        prompt=prompt,
        options=body.get("options"),
    )
    return {"ok": True, "id": job_id}


@app.get("/api/batch/status")
async def batch_status_list():
    return db.list_batch_jobs()


@app.get("/api/batch/status/{job_id}")
async def batch_status(job_id: str):
    status = queue_manager.get_job_status(job_id)
    if not status:
        raise HTTPException(status_code=404, detail="Job not found")
    return status


@app.get("/api/batch/results/{job_id}")
async def batch_results(job_id: str):
    result = queue_manager.get_job_result(job_id)
    if result is None:
        status = queue_manager.get_job_status(job_id)
        if not status:
            raise HTTPException(status_code=404, detail="Job not found")
        raise HTTPException(status_code=409, detail=f"Job status: {status['status']}")
    return result


@app.post("/api/batch/{job_id}/cancel")
async def batch_cancel(job_id: str):
    ok = db.cancel_batch_job(job_id)
    if not ok:
        raise HTTPException(status_code=409, detail="Job cannot be cancelled (not queued)")
    return {"ok": True}


@app.post("/api/batch/{job_id}/requeue")
async def batch_requeue(job_id: str):
    ok = db.requeue_batch_job(job_id)
    if not ok:
        raise HTTPException(status_code=409, detail="Only failed jobs can be requeued")
    return {"ok": True}


@app.post("/api/batch/drain")
async def batch_drain():
    """Cancel all queued jobs."""
    n = db.drain_queued_jobs()
    return {"ok": True, "cancelled": n}


@app.post("/api/batch/retry-failed")
async def batch_retry_failed():
    """Requeue all failed jobs and re-enqueue them for processing."""
    n = db.requeue_all_failed_jobs()
    # Re-enqueue each requeued job for processing.
    for job in db.list_batch_jobs(status="queued"):
        asyncio.ensure_future(queue_manager._run_batch_job_async(job["id"]))
    return {"ok": True, "requeued": n}


@app.delete("/api/batch/completed")
async def batch_delete_completed(older_than_days: Optional[int] = None):
    """Delete completed and cancelled jobs, optionally filtered by age."""
    n = db.delete_terminal_jobs(older_than_days)
    return {"ok": True, "deleted": n}


# ── Fan controller fault log ──────────────────────────────────────────────────

@app.post("/api/fan/fault")
async def fan_fault(request: Request):
    """
    Receive a fault event from the iDRAC fan controller and store it.

    Expected body:
      {
        "type":              "gpu_fault_onset" | "gpu_fault_cleared" | ...,
        "message":           "human-readable description",
        "timestamp":         "ISO-8601 string (informational; DB uses server time)",
        "fan_speed_applied": 80   (optional integer)
      }
    """
    try:
        body = await request.json()
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid JSON: {e}")

    event_type = body.get("type", "unknown")
    message    = body.get("message", "")
    fan_speed  = body.get("fan_speed_applied")
    if fan_speed is not None:
        try:
            fan_speed = int(fan_speed)
        except (ValueError, TypeError):
            fan_speed = None

    row_id = db.insert_fan_fault(event_type, message, fan_speed)
    return {"ok": True, "id": row_id}


@app.get("/api/fan/faults")
async def fan_faults(limit: int = 100):
    """Return the most recent fan controller fault events, newest first."""
    return db.list_fan_faults(limit=min(limit, 500))


# ── Metrics ───────────────────────────────────────────────────────────────────


@app.get("/api/merllm/metrics/current")
async def metrics_current():
    return db.get_latest_metrics()


@app.get("/api/merllm/metrics/history")
async def metrics_history(metric: str, range: str = "1h"):
    ranges = {"1h": 3600, "6h": 21600, "24h": 86400, "7d": 604800}
    seconds = ranges.get(range, 3600)
    since = time.time() - seconds
    return db.get_metrics_history(metric, since)


@app.get("/api/merllm/metrics/alerts")
async def metrics_alerts():
    latest = db.get_latest_metrics()
    status = await merllm_status()
    return {"warnings": status["warnings"]}


@app.post("/api/merllm/metrics/thresholds")
async def metrics_thresholds(request: Request):
    body = await request.json()
    db.save_settings({"alert_thresholds": body})
    return {"ok": True}


# ── Database backup ───────────────────────────────────────────────────────────


@app.get("/api/merllm/myday")
async def merllm_myday():
    """
    Aggregate 'My Day' attention summary from all Hexcaliper services.

    Fetches from Parsival (/page/api/attention/summary) and LanceLLMot
    (/api/status/pending) concurrently.  If a service is unreachable,
    its section is omitted and an error flag is set.

    Also returns merLLM's own batch job queue counts.

    :return: Dict with ``parsival``, ``lancellmot``, and ``merllm`` sections.
    :rtype: dict
    """
    _myday_log = logging.getLogger("merllm.myday")

    async def _fetch(url: str) -> dict | None:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(url)
                r.raise_for_status()
                return r.json()
        except Exception as exc:
            _myday_log.warning("myday fetch failed for %s: %s", url, exc)
            return None

    parsival_url    = f"{config.PARSIVAL_URL}/page/api/attention/summary"
    lancellmot_url  = f"{config.LANCELLMOT_URL}/api/status/pending"

    parsival_data, lancellmot_data = await asyncio.gather(
        _fetch(parsival_url),
        _fetch(lancellmot_url),
    )

    queued_jobs    = db.list_batch_jobs(status="queued")
    completed_jobs = db.list_batch_jobs(status="completed")
    failed_jobs    = db.list_batch_jobs(status="failed")

    return {
        "parsival": {
            "ok":               parsival_data is not None,
            "active_situations": parsival_data.get("active_situations", 0) if parsival_data else 0,
            "new_investigating": parsival_data.get("new_investigating", 0) if parsival_data else 0,
            "overdue_followups": parsival_data.get("overdue_followups", 0) if parsival_data else 0,
            "cold_start":        parsival_data.get("cold_start", True) if parsival_data else True,
        },
        "lancellmot": {
            "ok":                   lancellmot_data is not None,
            "acquisition_pending":  lancellmot_data.get("acquisition_pending", 0) if lancellmot_data else 0,
            "escalation_pending":   lancellmot_data.get("escalation_pending", 0) if lancellmot_data else 0,
            "total_pending":        lancellmot_data.get("total_pending", 0) if lancellmot_data else 0,
        },
        "merllm": {
            "ok":              True,
            "queued_jobs":     len(queued_jobs),
            "completed_jobs":  len(completed_jobs),
            "failed_jobs":     len(failed_jobs),
        },
    }


@app.get("/api/merllm/events")
async def merllm_events(request: Request):
    """
    Server-Sent Events stream for real-time browser notifications.

    Emits job_complete or job_failed events when batch jobs finish.
    Clients connect once and keep the connection open.
    """
    async def event_generator():
        q = notifications.add_sse_listener()
        try:
            yield "data: {\"type\": \"connected\"}\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    data = await asyncio.wait_for(q.get(), timeout=30)
                    yield f"data: {data}\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"  # SSE comment keeps connection alive
        finally:
            notifications.remove_sse_listener(q)

    return StreamingResponse(event_generator(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


@app.post("/api/merllm/backup")
async def trigger_backup():
    """Create an online SQLite backup and rotate old files."""
    backup_dir = Path(config.BACKUP_DIR)
    db_path    = Path(config.DB_PATH)
    if not db_path.exists():
        raise HTTPException(status_code=500, detail=f"Database not found: {db_path}")
    backup_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime())
    dest  = backup_dir / f"merllm-{stamp}.db"
    try:
        src = sqlite3.connect(str(db_path))
        dst = sqlite3.connect(str(dest))
        src.backup(dst)
        dst.close()
        src.close()
        size = dest.stat().st_size
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail=f"Backup failed: {exc}")
    all_backups = sorted(backup_dir.glob("merllm-*.db"))
    rotated: list[str] = []
    if len(all_backups) > config.BACKUP_KEEP_DAYS:
        for old in all_backups[:-config.BACKUP_KEEP_DAYS]:
            old.unlink(missing_ok=True)
            rotated.append(str(old))
    return {"ok": True, "backup": str(dest), "size_bytes": size, "rotated": rotated}


# ── Diagnostics ───────────────────────────────────────────────────────────────


@app.get("/api/merllm/diagnostics")
async def diagnostics():
    """Connectivity checks for all Ollama instances."""
    results = {}
    for label, url in [("ollama_gpu0", config.OLLAMA_0_URL),
                        ("ollama_gpu1", config.OLLAMA_1_URL)]:
        try:
            async with httpx.AsyncClient(timeout=3) as client:
                r = await client.get(f"{url}/api/tags")
                results[label] = {
                    "ok":       r.status_code == 200,
                    "url":      url,
                    "status":   r.status_code,
                    "latency_ms": None,
                }
        except Exception as exc:
            results[label] = {"ok": False, "url": url, "error": str(exc)}

    # Docker container health
    containers = _docker_status()

    return {"connectivity": results, "containers": containers}


def _docker_status() -> list[dict]:
    try:
        import docker
        client = docker.from_env()
        result = []
        for c in client.containers.list(all=True):
            stats = {}
            try:
                raw = c.stats(stream=False)
                cpu_delta = (raw["cpu_stats"]["cpu_usage"]["total_usage"] -
                             raw["precpu_stats"]["cpu_usage"]["total_usage"])
                sys_delta = (raw["cpu_stats"]["system_cpu_usage"] -
                             raw["precpu_stats"]["system_cpu_usage"])
                num_cpus = raw["cpu_stats"].get("online_cpus", 1)
                cpu_pct = (cpu_delta / sys_delta) * num_cpus * 100 if sys_delta else 0
                mem_usage = raw["memory_stats"].get("usage", 0)
                stats = {"cpu_pct": round(cpu_pct, 1), "mem_mb": round(mem_usage / 1024**2, 1)}
            except Exception:
                pass
            result.append({
                "name":    c.name,
                "status":  c.status,
                "image":   c.image.tags[0] if c.image.tags else str(c.image.id)[:12],
                **stats,
            })
        return result
    except Exception:
        return []


# Services backed by Docker containers — names read from config (env-var overridable)
_LOG_DOCKER = {
    "lancellmot-api":   config.LOG_CONTAINER_LANCELLMOT_API,
    "lancellmot-nginx": config.LOG_CONTAINER_LANCELLMOT_NGINX,
    "parsival-api":     config.LOG_CONTAINER_PARSIVAL_API,
    "parsival-nginx":   config.LOG_CONTAINER_PARSIVAL_NGINX,
    "merllm-api":       config.LOG_CONTAINER_MERLLM_API,
}

_ALL_LOG_SERVICES = set(_LOG_DOCKER)


@app.get("/api/merllm/logs/{service}")
async def service_logs(service: str, lines: int = 100):
    """Tail recent log lines from Docker containers."""
    if service not in _ALL_LOG_SERVICES:
        raise HTTPException(status_code=422,
                            detail=f"Unknown service. Valid: {sorted(_ALL_LOG_SERVICES)}")

    container_name = _LOG_DOCKER[service]
    try:
        import docker
        client = docker.from_env()
        container = client.containers.get(container_name)
        raw = container.logs(tail=lines, timestamps=False).decode("utf-8", errors="replace")
        return {"service": service, "container": container_name, "lines": raw.splitlines()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))




# ── Fan controller proxy ──────────────────────────────────────────────────────


@app.get("/api/merllm/fans/status")
async def fans_status():
    """Current temperatures, fan speed, and active thresholds from the iDRAC fan controller."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{config.FAN_CONTROLLER_URL}/status")
            return Response(content=r.content, media_type="application/json",
                            status_code=r.status_code)
    except Exception as exc:
        return JSONResponse({"error": str(exc), "available": False}, status_code=503)


@app.get("/api/merllm/fans/settings")
async def fans_get_settings():
    """Active fan controller override values (empty object if none set)."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{config.FAN_CONTROLLER_URL}/settings")
            return Response(content=r.content, media_type="application/json",
                            status_code=r.status_code)
    except Exception as exc:
        return JSONResponse({"error": str(exc), "available": False}, status_code=503)


@app.post("/api/merllm/fans/settings")
async def fans_save_settings(request: Request):
    """Push threshold/speed overrides to the fan controller (merges with existing)."""
    body = await request.json()
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.post(f"{config.FAN_CONTROLLER_URL}/settings", json=body)
            return Response(content=r.content, media_type="application/json",
                            status_code=r.status_code)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


@app.delete("/api/merllm/fans/settings")
async def fans_reset_settings():
    """Remove all fan controller overrides, reverting to env-var defaults."""
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.delete(f"{config.FAN_CONTROLLER_URL}/settings")
            return Response(content=r.content, media_type="application/json",
                            status_code=r.status_code)
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)


@app.get("/api/merllm/fans/faults")
async def fans_faults(limit: int = 100):
    """Fault history stored in merLLM DB (does not proxy to fan controller)."""
    return db.list_fan_faults(limit=min(limit, 500))


# ── WebSocket: SSH terminal ───────────────────────────────────────────────────


@app.websocket("/ws/ssh")
async def ws_ssh(websocket: WebSocket):
    """
    Browser SSH terminal via xterm.js.

    Requires asyncssh and a valid SSH_USER + SSH_KEY_PATH configured.
    The connection authenticates using an SSH key — merLLM never stores
    passwords. Authentication is gated by Cloudflare Access at the nginx layer.
    """
    await websocket.accept()
    if not config.SSH_USER:
        await websocket.send_text("SSH_USER not configured.\r\n")
        await websocket.close()
        return
    try:
        import asyncssh

        async with asyncssh.connect(
            config.SSH_HOST,
            port=config.SSH_PORT,
            username=config.SSH_USER,
            client_keys=[config.SSH_KEY_PATH] if os.path.exists(config.SSH_KEY_PATH) else None,
            known_hosts=None,
        ) as conn:
            async with conn.create_process(term_type="xterm-256color") as proc:

                async def ws_to_proc():
                    try:
                        while True:
                            data = await websocket.receive_text()
                            proc.stdin.write(data)
                    except WebSocketDisconnect:
                        pass

                async def proc_to_ws():
                    try:
                        while True:
                            data = await proc.stdout.read(4096)
                            if not data:
                                break
                            await websocket.send_text(data)
                    except WebSocketDisconnect:
                        pass

                await asyncio.gather(ws_to_proc(), proc_to_ws())

    except Exception as exc:
        try:
            await websocket.send_text(f"\r\nSSH error: {exc}\r\n")
            await websocket.close()
        except Exception:
            pass


# ── WebSocket: VNC proxy ──────────────────────────────────────────────────────


@app.websocket("/ws/vnc")
async def ws_vnc(websocket: WebSocket):
    """
    noVNC WebSocket-to-TCP proxy.

    Proxies the browser WebSocket connection to the VNC server at
    VNC_HOST:VNC_PORT. The VNC server must be started separately (e.g.
    TigerVNC or x11vnc). This endpoint only proxies the connection.
    """
    await websocket.accept()
    try:
        reader, writer = await asyncio.open_connection(config.VNC_HOST, config.VNC_PORT)

        async def ws_to_vnc():
            try:
                while True:
                    data = await websocket.receive_bytes()
                    writer.write(data)
                    await writer.drain()
            except (WebSocketDisconnect, Exception):
                writer.close()

        async def vnc_to_ws():
            try:
                while True:
                    data = await reader.read(4096)
                    if not data:
                        break
                    await websocket.send_bytes(data)
            except (WebSocketDisconnect, Exception):
                pass

        await asyncio.gather(ws_to_vnc(), vnc_to_ws())

    except Exception as exc:
        try:
            await websocket.close(code=1011, reason=str(exc))
        except Exception:
            pass


# ── Root redirect ─────────────────────────────────────────────────────────────


@app.get("/")
async def root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="/web/index.html")


@app.get("/health")
async def health():
    return {"ok": True, "routing": "round_robin"}
