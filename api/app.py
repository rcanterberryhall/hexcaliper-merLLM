"""
app.py — merLLM: centralized LLM traffic control for the Hexcaliper ecosystem.

Exposes a drop-in Ollama API proxy on :11400. Both LanceLLMot and Parsival
point their OLLAMA_BASE_URL here. Requests flow through a late-binding GPU
dispatcher with a 5-bucket strict priority queue (chat > embeddings > short >
feedback > background) and per-GPU health tracking — the target GPU is
chosen only when one actually becomes idle, then drained top-down with
FIFO inside each bucket.

Additional endpoints:
  GET/POST  /api/merllm/status    — GPU state, queue depth, health
  GET/POST  /api/merllm/settings  — view/update configuration
  GET       /api/merllm/default-model — current default model
  POST      /api/merllm/gpu/{gpu}/reset — manual GPU health reset
  GET       /api/merllm/queue     — unified GPU queue (all active/waiting requests)
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
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException, Body
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
    result["queue"] = queue_manager.active_queue()
    result["queue_summary"] = queue_manager.queue_depth()
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

    # Restore operator pause state before recovery kicks in so re-enqueued
    # jobs respect it immediately. A power outage during a pause must not
    # silently resume bulk work — the setting lives in SQLite so it travels
    # with the DB through a restart.
    if saved.get("queue_paused"):
        queue_manager.set_paused(True, persist=False)
        log.info("queue resumed in PAUSED state from persisted setting")

    # Boot-reconcile the scheduler: probe each GPU, rehydrate slot_state,
    # wipe stale pending_work rows. MUST run before any track_request so the
    # tick loop sees a valid slot lineup. _boot_reconcile wipes _buckets_v2,
    # _tid_to_job, and _inflight — doing this after the batch re-kick below
    # would orphan those tasks' dispatch futures (they'd await forever while
    # buckets sit empty). _ensure_tick starts the loop.
    await queue_manager._boot_reconcile()
    queue_manager._ensure_tick()

    # Wire queue-change notifications into the activity SSE stream
    queue_manager.set_queue_change_callback(lambda: _push_activity_sse(force=True))

    # Recover any jobs that were running when the process was last killed.
    # ``requeue_orphaned_jobs`` only resets DB status; we also have to
    # re-enqueue an async task for each one or they sit queued forever
    # (the in-memory ``_run_batch_job_async`` future is process-local).
    #
    # Deferred retries (``retry_after`` in the future) get a delayed
    # schedule rather than an immediate run, so a job that was backing off
    # for 10 minutes resumes its backoff instead of firing instantly on
    # restart. Each re-enqueued job also gets a shadow entry in the
    # tracked queue so the Overview UI shows the backlog from the first
    # frame instead of looking empty while async tasks spin up.
    recovered = db.requeue_orphaned_jobs()
    if recovered:
        log.info("recovered %d orphaned job(s) → queued", recovered)
    pending = db.list_batch_jobs(status="queued")
    now = time.time()
    immediate = 0
    deferred  = 0
    for job in pending:
        retry_after = job.get("retry_after")
        if retry_after and retry_after > now:
            delay = retry_after - now
            deferred += 1
            # ``call_later`` with a bound lambda to capture ``job_id`` by
            # value. The sleep runs in the event loop, not a thread, so
            # hundreds of deferred jobs cost effectively nothing.
            asyncio.get_event_loop().call_later(
                delay,
                lambda jid=job["id"]: asyncio.ensure_future(
                    queue_manager._run_batch_job_async(jid)
                ),
            )
        else:
            immediate += 1
            asyncio.ensure_future(queue_manager._run_batch_job_async(job["id"]))
    if pending:
        log.info(
            "re-enqueued %d queued batch job(s): %d immediate, %d deferred by retry_after",
            len(pending), immediate, deferred,
        )

    # Background tasks
    asyncio.create_task(metrics.collection_loop())
    asyncio.create_task(gpu_router.recovery_loop())

    log.info("started — routing=%s, default_model=%s",
             "fsm", config.DEFAULT_MODEL)
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


def _priority(request: Request) -> queue_manager.Priority:
    """Parse the X-Priority header into one of the five priority buckets.

    Accepts canonical names (``chat`` / ``embeddings`` / ``short`` /
    ``feedback`` / ``background``) and the legacy ``interactive`` / ``batch``
    aliases.

    Missing values fall back to ``CHAT`` for backwards compatibility with
    clients that predate the header — the same behavior as the old
    two-tier system. Unknown (non-empty) values fall back to ``BACKGROUND``
    so typos cannot silently escalate work to a higher lane.

    Note: ``proxy_embeddings`` does not call this parser. It defaults
    embeds to ``Priority.EMBEDDINGS`` and only honors one explicit
    override — ``X-Priority: chat`` (or the ``interactive`` alias)
    routes the embed to ``Priority.CHAT`` so a chat-path RAG embed
    jumps ahead of ingest-chunk embeds (merLLM#58). Any other header
    value still falls through to ``EMBEDDINGS``, so a missing or wrong
    ``X-Priority`` from a non-chat client cannot route an embed into
    the wrong bucket (preserves merLLM#38).

    TODO: once parsival and lancellmot are both sending explicit
    ``X-Priority`` on every call, tighten the missing-header default to
    ``BACKGROUND`` so we catch any new code path that forgets to declare.
    """
    raw = request.headers.get("x-priority")
    if raw is None:
        return queue_manager.Priority.CHAT
    return queue_manager.priority_from_name(
        raw, default=queue_manager.Priority.BACKGROUND
    )


def _source(request: Request) -> str:
    return request.headers.get("x-source", "direct")


def _normalize_reasoning_body(body: dict) -> None:
    """Pin ``num_ctx`` for known large reasoning models on the proxy path.

    Without this, a user-facing chat/generate request that omits ``num_ctx``
    reaches Ollama with no context size, and Ollama auto-fits the KV cache
    to whatever VRAM is free at load time. On a 24 GB Tesla P40 with no
    co-resident model, qwen3:32b auto-fits to its 32 k training maximum,
    forces CPU offload of the output layer, and the load takes ~12 s
    instead of ~330 ms — every subsequent token is also slower because
    the KV cache straddles the PCIe bus.

    The dispatcher's ``_reload_model`` already pins ``num_ctx`` for the
    warm-load it sends on a swap, but if the dispatcher's ``GpuState``
    already thinks the right model is loaded the warm-load is skipped and
    the user request hits Ollama unmodified — at which point Ollama may
    have evicted the model on its own ``keep_alive`` timer and reload it
    from scratch with the user's (empty) options. Pinning here closes that
    gap so every entry point converges on the same KV-cache size.

    Mirrors the same defaults the batch path applies in
    ``queue_manager._run_batch_job_async``. Only fills in MISSING keys —
    explicit caller values are respected.
    """
    model = body.get("model", "")
    if not model.startswith("qwen3:"):
        return
    options = body.get("options")
    if not isinstance(options, dict):
        options = {}
        body["options"] = options
    options.setdefault("num_ctx", 8192)


async def _generate_stream(target: str, path: str, body: dict, on_done=None):
    """
    Async generator that streams NDJSON from Ollama, line by line.

    Injects ``context_tokens`` (= prompt_eval_count) into the final done line
    so callers can display context-window utilisation.

    on_done: optional zero-argument callable called in the finally block.
    """
    buf = b""
    try:
        # Bounded read timeout (defence-in-depth behind the tick's busy-slot
        # timeout sweep) so a hung Ollama stream cannot pin a slot forever —
        # see queue_manager._sweep_busy_timeouts for the primary deadlock guard.
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(config.PROXY_READ_TIMEOUT_SECONDS),
        ) as client:
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


def _any_gpu_busy() -> bool:
    """True if any configured GPU is currently busy or swapping."""
    return any(queue_manager.gpu_slot_busy(url)
               for url in (config.OLLAMA_0_URL, config.OLLAMA_1_URL))


async def _queued_stream(path: str, body: dict,
                         tracking_id: str) -> StreamingResponse:
    """Streaming proxy backed by the FSM dispatcher.

    Emits a ``queue_status`` NDJSON line if no GPU is immediately available,
    then awaits dispatch, then streams the Ollama response. The assigned
    target URL is chosen by the dispatcher at dispatch time.

    Slot lifecycle is owned by ``queue_manager.dispatched``: release on
    normal completion, fail_request on exception, cancel_tracked (inside
    ``wait_for_dispatch``) on pre-dispatch caller exit. The streaming
    generator itself is thin — no flags, no outer cancel_tracked.

    Keepalive: ``wait_for_dispatch`` is run as a task so this generator
    can emit ``queue_status`` heartbeats while it blocks. Each heartbeat
    resets the caller's between-chunk read-gap timeout; parsival's
    ``requests.post(timeout=60)`` would otherwise trip when a BACKGROUND
    drain pushes a feedback wait past 60 s (the 2026-04-11 wedge). Heartbeat
    lines carry no ``response``/``thinking`` tokens so ``llm._collect_stream``
    silently drops them — no caller change required.
    """
    async def generate():
        if _any_gpu_busy():
            yield json.dumps({
                "type":                    "queue_status",
                "reason":                  "all GPU slots occupied — queued for dispatch",
                "estimated_wait_seconds":  config.INTERACTIVE_QUEUE_TIMEOUT,
            }).encode() + b"\n"

        dispatch_task = asyncio.create_task(
            queue_manager.wait_for_dispatch(tracking_id)
        )
        waiting_since = time.time()
        try:
            while True:
                done, _ = await asyncio.wait(
                    {dispatch_task},
                    timeout=config.QUEUE_HEARTBEAT_INTERVAL_SECONDS,
                )
                if dispatch_task in done:
                    break
                yield json.dumps({
                    "type":             "queue_status",
                    "waiting":          True,
                    "elapsed_seconds":  int(time.time() - waiting_since),
                }).encode() + b"\n"
            target = dispatch_task.result()
        except queue_manager.DispatchTimeout:
            yield json.dumps({
                "error": "GPU busy — timed out waiting for slot",
                "done":  True,
            }).encode() + b"\n"
            return
        except BaseException:
            # wait_for_dispatch self-cleans; cancel the pending task so it
            # unwinds quickly and re-raise for the ASGI layer.
            if not dispatch_task.done():
                dispatch_task.cancel()
            raise

        _activity_set(target, body.get("model", ""), path)
        released = False
        try:
            def _on_done():
                nonlocal released
                _activity_clear(target)
                queue_manager.release(tracking_id)
                released = True

            async for chunk in _generate_stream(target, path, body, _on_done):
                yield chunk
        except BaseException as exc:
            _activity_clear(target)
            if not released:
                queue_manager.fail_request(tracking_id, str(exc))
            raise

    return StreamingResponse(generate(), media_type="application/x-ndjson")


async def _stream_and_accumulate(target: str, path: str, body: dict) -> dict:
    """Buffer-stream Ollama NDJSON and return the single-JSON payload shape.

    Forces ``stream=True`` on the wire so every NDJSON line ticks
    ``_activity_inc`` + ``_activity_append_token`` (the instance card's tok
    counter + rolling text buffer), then accumulates the streamed tokens into
    the same dict shape a ``stream=False`` request would have returned.
    Caller-facing wire contract is unchanged; only the Ollama-facing request
    is rewritten. ``stream`` is wire-format only — KV cache / num_ctx /
    num_predict are identical in both modes.
    """
    wire_body = {**body, "stream": True}
    text, thinking = "", ""
    final: dict = {}
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(config.PROXY_READ_TIMEOUT_SECONDS),
    ) as client:
        async with client.stream("POST", f"{target}{path}", json=wire_body) as resp:
            resp.raise_for_status()
            buf = b""
            async for chunk in resp.aiter_bytes():
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    _activity_inc(target)
                    _activity_append_token(target, line, path)
                    try:
                        obj = json.loads(line)
                    except Exception:
                        continue
                    if path == "/api/generate":
                        text += obj.get("response", "") or ""
                    elif path == "/api/chat":
                        msg = obj.get("message") or {}
                        text     += msg.get("content", "")  or ""
                        thinking += msg.get("thinking", "") or ""
                    if obj.get("done"):
                        final = obj
    if path == "/api/generate":
        final["response"] = text
    elif path == "/api/chat":
        msg = dict(final.get("message") or {"role": "assistant"})
        msg["content"] = text
        if thinking:
            msg["thinking"] = thinking
        final["message"] = msg
    return final


async def _buffered_stream_proxy(target: str, path: str, body: dict) -> Response:
    """Non-streaming wire contract, streamed internally for live card updates.

    Used for generate/chat when the caller sends ``stream: False``. Wraps
    ``_activity_set`` / ``_activity_clear`` around ``_stream_and_accumulate``
    so the instance card shows model+endpoint for the full call duration
    while the tok counter ticks per line.
    """
    model = body.get("model", "")
    _activity_set(target, model, path)
    try:
        payload = await _stream_and_accumulate(target, path, body)
        return Response(content=json.dumps(payload).encode(),
                        media_type="application/json")
    finally:
        _activity_clear(target)


async def _proxy(target: str, path: str, body: dict) -> Response:
    """Non-streaming Ollama proxy (e.g. embeddings), tracking activity per instance."""
    model = body.get("model", "")
    _activity_set(target, model, path)
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(config.PROXY_READ_TIMEOUT_SECONDS),
        ) as client:
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


def _dispatch_timeout_response() -> JSONResponse:
    return JSONResponse(
        status_code=503,
        content={
            "error": "GPU busy — timed out waiting for slot",
            "estimated_wait_seconds": config.INTERACTIVE_QUEUE_TIMEOUT,
        },
        headers={"Retry-After": str(config.INTERACTIVE_QUEUE_TIMEOUT)},
    )


async def _proxy_nonstream(path: str, body: dict, tid: str) -> Response:
    """Non-streaming proxy backed by the FSM dispatcher.

    The ``queue_manager.dispatched`` CM owns slot lifecycle: release on
    happy path, fail_request on exception, cancel_tracked on pre-dispatch
    caller exit. No handler-side cleanup flags needed.

    For generate/chat, routes through ``_buffered_stream_proxy`` so the
    instance card's tok counter ticks per NDJSON line even when the caller
    sent ``stream: False``. Embeddings/show/pull stay on the plain ``_proxy``
    path — no NDJSON to accumulate.
    """
    try:
        async with queue_manager.dispatched(tid) as target:
            if path in ("/api/generate", "/api/chat"):
                resp = await _buffered_stream_proxy(target, path, body)
            else:
                resp = await _proxy(target, path, body)
            gpu_router.record_activity(target)
            return resp
    except queue_manager.DispatchTimeout:
        return _dispatch_timeout_response()


@app.post("/api/generate")
async def proxy_generate(request: Request):
    """Proxy POST /api/generate through the late-binding dispatcher.

    Reasoning-model defaults (e.g. ``num_ctx`` for ``qwen3:*``) are pinned
    here via :func:`_normalize_reasoning_body` before the body is tracked
    or dispatched, so callers that omit the option still hit the same KV
    cache footprint as the dispatcher's warm-load swap path.
    """
    body = await request.json()
    _normalize_reasoning_body(body)
    model = body.get("model", "")
    priority = _priority(request)
    source = _source(request)
    stream = body.get("stream", True)

    tid = queue_manager.track_request(source, "generate", model, priority)

    if stream:
        return await _queued_stream("/api/generate", body, tid)
    return await _proxy_nonstream("/api/generate", body, tid)


@app.post("/api/chat")
async def proxy_chat(request: Request):
    """Proxy POST /api/chat through the late-binding dispatcher.

    Reasoning-model defaults (e.g. ``num_ctx`` for ``qwen3:*``) are pinned
    here via :func:`_normalize_reasoning_body` before the body is tracked
    or dispatched, so callers that omit the option still hit the same KV
    cache footprint as the dispatcher's warm-load swap path.
    """
    body = await request.json()
    _normalize_reasoning_body(body)
    model = body.get("model", "")
    priority = _priority(request)
    source = _source(request)
    stream = body.get("stream", True)

    tid = queue_manager.track_request(source, "chat", model, priority)

    if stream:
        return await _queued_stream("/api/chat", body, tid)
    return await _proxy_nonstream("/api/chat", body, tid)


@app.post("/api/embeddings")
async def proxy_embeddings(request: Request):
    """Proxy POST /api/embeddings through the late-binding dispatcher.

    Embedding traffic defaults to ``Priority.EMBEDDINGS`` — a dedicated
    bucket so a burst of ingest-chunk embeds never sits behind a 32b
    chat in ``BACKGROUND`` (see merLLM#38). The only override honored
    here is ``X-Priority: chat`` (or the legacy ``interactive`` alias),
    which routes that single request to ``Priority.CHAT`` so a chat-path
    RAG embed jumps ahead of ingest-chunk embeds (merLLM#58). Any other
    ``X-Priority`` value — missing, unknown, or a lower bucket like
    ``background`` — falls through to ``EMBEDDINGS``, so #38's narrow
    rule (header can't silently route embeds into the wrong bucket) is
    preserved for every non-chat caller.
    """
    body = await request.json()
    model = body.get("model", "")
    raw_priority = request.headers.get("x-priority")
    if raw_priority and raw_priority.strip().lower() in ("chat", "interactive"):
        priority = queue_manager.Priority.CHAT
    else:
        priority = queue_manager.Priority.EMBEDDINGS
    source = _source(request)

    tid = queue_manager.track_request(source, "embeddings", model, priority)
    return await _proxy_nonstream("/api/embeddings", body, tid)


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
        "queue":              queue_manager.queue_depth(),
        "queue_paused":       queue_manager.is_paused(),
        "queue_paused_since": queue_manager.paused_since(),
        "scheduler_status":   queue_manager.scheduler_status(),
        "ollama":             ollama_health,
        "gpu_metrics":        metrics.gpu_snapshot(),
        "warnings":           _build_warnings(ollama_health, latest),
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
        temp = latest.get(f"gpu{i}.temp_c", {}).get("value")
        if temp and temp >= config.GPU_TEMP_PAUSE_C:
            warnings.append(
                f"GPU {i} temperature {temp:.0f}°C — dispatch paused "
                f"(resume below {config.GPU_TEMP_RESUME_C}°C)"
            )
        elif temp and temp > 80:
            warnings.append(f"GPU {i} temperature {temp:.0f}°C — approaching thermal pause")
    return warnings


@app.get("/api/merllm/queue")
async def merllm_queue():
    """
    Unified GPU queue: all active, waiting, and recently completed requests
    across every priority bucket.

    ``buckets`` reports pre-dispatch depth per bucket (how many waiters each
    priority lane currently holds). The dashboard uses this to show the five
    drain lanes.
    """
    return {
        "queue":        queue_manager.active_queue(),
        "summary":      queue_manager.queue_depth(),
        "buckets":      queue_manager.pipe_depth(),
        "paused":       queue_manager.is_paused(),
        "paused_since": queue_manager.paused_since(),
    }


@app.post("/api/merllm/queue/pause")
async def merllm_queue_pause():
    """Operator pause: stop dispatching new GPU work (in-flight continues).

    Persisted across restarts via the ``settings`` table.
    """
    changed = queue_manager.set_paused(True)
    return {
        "ok":           True,
        "paused":       True,
        "changed":      changed,
        "paused_since": queue_manager.paused_since(),
    }


@app.post("/api/merllm/queue/resume")
async def merllm_queue_resume():
    """Resume dispatching. Wakes the dispatcher so queued heads run now."""
    changed = queue_manager.set_paused(False)
    return {"ok": True, "paused": False, "changed": changed}


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
    """Manually reset a wedged GPU slot — re-probe and feed result into the FSM.

    Used when a slot has been UNREACHABLE long enough to stop being probed
    by the recovery loop (past HEALTH_FAULT_TIMEOUT) and the operator wants
    to force a fresh attempt without restarting the container.
    """
    url_map = {"gpu0": config.OLLAMA_0_URL, "gpu1": config.OLLAMA_1_URL}
    url = url_map.get(gpu)
    if not url:
        raise HTTPException(status_code=422, detail="gpu must be 'gpu0' or 'gpu1'")
    ok = gpu_router.reset_slot(url)
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

    # DEFAULT_MODEL change requires explicit confirmation. The FSM loads
    # models on demand (stage_pass mirrors current bucket-head demand onto
    # idle slots), so the change takes effect the next time work arrives —
    # no eager warm-load is scheduled here.
    new_model = body.get("default_model")
    if new_model and new_model != config.DEFAULT_MODEL:
        if not body.get("confirm_model_change"):
            return JSONResponse(
                status_code=409,
                content={
                    "ok": False,
                    "error": "Changing default_model will affect new dispatches. "
                             "Resend with confirm_model_change: true to proceed.",
                },
            )

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


@app.post("/api/batch/status-by-ids")
async def batch_status_by_ids(body: dict = Body(...)):
    """Return status for the given job IDs. Missing IDs are omitted from the
    response — the client distinguishes 'not yet written' from 'deleted' by
    consecutive-miss count on its side, not by HTTP error.

    Bypasses the LIMIT 200 window on GET /api/batch/status, so clients with
    many in-flight jobs don't lose visibility when newer jobs from other
    sources push theirs past the window.
    """
    ids = body.get("ids") or []
    if not isinstance(ids, list):
        raise HTTPException(status_code=400, detail="'ids' must be a list")
    return db.get_batch_jobs_by_ids([str(x) for x in ids])


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
    # Re-enqueue the async task: ``requeue_batch_job`` only updates DB status,
    # so without this the requeued job would sit in the queue forever.
    asyncio.ensure_future(queue_manager._run_batch_job_async(job_id))
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
async def batch_delete_completed(
    older_than_days: Optional[int] = None,
    include_failed: bool = False,
):
    """Delete terminal-state jobs.

    By default removes ``completed`` and ``cancelled`` only. Pass
    ``include_failed=true`` to also drop ``failed`` rows — useful when the
    caller is intentionally draining the queue (e.g. before a fresh ingest
    run) and doesn't need the failure history preserved.
    """
    n = db.delete_terminal_jobs(older_than_days, include_failed=include_failed)
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
    return {"ok": True, "routing": "fsm"}
