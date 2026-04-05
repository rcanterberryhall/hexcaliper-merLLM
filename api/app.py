"""
app.py — merLLM: centralized LLM traffic control for the Hexcaliper ecosystem.

Exposes a drop-in Ollama API proxy on :11400. Both LanceLLMot and Parsival
point their OLLAMA_BASE_URL here. merLLM routes requests to the appropriate
GPU-pinned Ollama instance and manages day/night mode transitions.

Additional endpoints:
  GET/POST  /api/merllm/status    — current mode, GPU layout, queue depth
  POST      /api/merllm/mode      — manual override (day/night/auto)
  GET/POST  /api/merllm/settings  — view/update configuration
  GET       /api/merllm/activity  — live per-instance request state + loaded models
  POST      /api/batch/submit     — queue a job for night-mode processing
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
import os
import subprocess
import time
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import Response, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

import config
import db
import geoip
import metrics
import mode_manager
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

    # Register mode-change callback to signal batch runner
    mode_manager.register_mode_change_callback(
        lambda m: queue_manager.signal_night_mode()
        if m == mode_manager.Mode.NIGHT else None
    )

    # Background tasks
    asyncio.create_task(metrics.collection_loop())
    asyncio.create_task(mode_manager.scheduler_loop())
    asyncio.create_task(queue_manager.worker_loop())
    asyncio.create_task(queue_manager.batch_runner_loop())

    print(f"[merllm] started — mode={mode_manager.current_mode().value}")
    yield
    print("[merllm] shutting down")


app = FastAPI(title="merLLM", version="1.0.0", lifespan=lifespan)

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


async def _stream_proxy(target: str, path: str, body: dict) -> StreamingResponse:
    """Stream an Ollama response, tracking activity per instance."""
    model = body.get("model", "")
    _activity_set(target, model, path)

    async def generate():
        # Line buffer: raw byte chunks from Ollama don't align with JSON lines.
        # We accumulate bytes and split on newlines to get complete NDJSON objects.
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
                        yield chunk
        finally:
            _activity_clear(target)

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
    """
    Proxy POST /api/generate to the appropriate Ollama instance.

    Routes based on requested model name in day mode. Interactive requests
    update the inactivity timer; if night mode is active and an interactive
    request arrives, triggers transition back to day mode.
    """
    body = await request.json()
    model = body.get("model", "")
    priority = _priority(request)
    ip = _client_ip(request)

    if priority == queue_manager.PRIORITY_INTERACTIVE:
        mode_manager.record_interactive(ip)
        if mode_manager.current_mode() == mode_manager.Mode.NIGHT:
            asyncio.create_task(mode_manager.transition_to_day("interactive_request"))

    target = mode_manager.get_target_url(model)
    stream = body.get("stream", True)

    async with mode_manager.InFlightGuard():
        if stream:
            return await _stream_proxy(target, "/api/generate", body)
        return await _proxy(target, "/api/generate", body)


@app.post("/api/chat")
async def proxy_chat(request: Request):
    """Proxy POST /api/chat."""
    body = await request.json()
    model = body.get("model", "")
    priority = _priority(request)
    ip = _client_ip(request)

    if priority == queue_manager.PRIORITY_INTERACTIVE:
        mode_manager.record_interactive(ip)
        if mode_manager.current_mode() == mode_manager.Mode.NIGHT:
            asyncio.create_task(mode_manager.transition_to_day("interactive_request"))

    target = mode_manager.get_target_url(model)
    stream = body.get("stream", True)

    async with mode_manager.InFlightGuard():
        if stream:
            return await _stream_proxy(target, "/api/chat", body)
        return await _proxy(target, "/api/chat", body)


@app.post("/api/embeddings")
async def proxy_embeddings(request: Request):
    """Proxy POST /api/embeddings to GPU 0."""
    body = await request.json()
    async with mode_manager.InFlightGuard():
        return await _proxy(config.OLLAMA_0_URL, "/api/embeddings", body)


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
    async with mode_manager.InFlightGuard():
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
        **mode_manager.status(),
        "queue":         queue_manager.queue_depth(),
        "ollama":        ollama_health,
        "gpus":          metrics.gpu_snapshot(),
        "batch_counts":  db.count_batch_jobs_by_status(),
        "last_transition": db.list_transitions(limit=1),
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


@app.post("/api/merllm/mode")
async def set_mode(request: Request):
    """
    Manual mode override.

    Body: {"mode": "day" | "night" | "auto"}
    "auto" clears the override and returns to schedule-driven behaviour.
    """
    body = await request.json()
    mode = body.get("mode")
    if mode not in ("day", "night", "auto", None):
        raise HTTPException(status_code=422, detail="mode must be 'day', 'night', or 'auto'")
    await mode_manager.set_override(None if mode == "auto" else mode)
    return {"ok": True, "mode": mode_manager.current_mode().value}


@app.get("/api/merllm/settings")
async def get_settings():
    saved = db.get_settings()
    return {
        "ollama_0_url":           config.OLLAMA_0_URL,
        "ollama_1_url":           config.OLLAMA_1_URL,
        "day_model_gpu0":         config.DAY_MODEL_GPU0,
        "day_model_gpu1":         config.DAY_MODEL_GPU1,
        "night_model":            config.NIGHT_MODEL,
        "night_num_ctx":          config.NIGHT_NUM_CTX,
        "inactivity_timeout_min": config.INACTIVITY_TIMEOUT_MIN,
        "base_day_end_local":     config.BASE_DAY_END_LOCAL,
        "geoip_offset_override":  config.GEOIP_OFFSET_OVERRIDE,
        "ollama_manage_via":      config.OLLAMA_MANAGE_VIA,
        "drain_timeout_sec":      config.DRAIN_TIMEOUT_SEC,
        "metrics_interval_sec":   config.METRICS_INTERVAL_SEC,
    }


@app.post("/api/merllm/settings")
async def save_settings(request: Request):
    body = await request.json()
    db.save_settings(body)
    config.apply_overrides(body)
    return {"ok": True}


# ── Batch API ─────────────────────────────────────────────────────────────────


@app.post("/api/batch/submit")
async def batch_submit(request: Request):
    """
    Queue a generation request for night-mode extended-context processing.

    Body: {
      "source_app": "parsival" | "lancellmot" | ...,
      "prompt": "...",
      "model": "qwen3:32b",       # optional, defaults to NIGHT_MODEL
      "options": {}               # optional Ollama options
    }
    """
    body = await request.json()
    prompt = body.get("prompt", "").strip()
    if not prompt:
        raise HTTPException(status_code=422, detail="prompt is required")

    job_id = queue_manager.submit_batch_job(
        source_app=body.get("source_app", "unknown"),
        model=body.get("model", config.NIGHT_MODEL),
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


# Services backed by Docker containers
_LOG_DOCKER = {
    "lancellmot-api":   "hexcaliper-lancellmot-api-1",
    "lancellmot-nginx": "hexcaliper-lancellmot-nginx-1",
    "parsival-api":     "hexcaliper-squire-parsival-api-1",
    "parsival-nginx":   "hexcaliper-squire-parsival-nginx-1",
    "merllm-api":       "merllm-api",
}

# Services backed by host systemd units
_LOG_SYSTEMD = {"ollama-gpu0", "ollama-gpu1", "ollama-night"}

_ALL_LOG_SERVICES = set(_LOG_DOCKER) | _LOG_SYSTEMD


@app.get("/api/merllm/logs/{service}")
async def service_logs(service: str, lines: int = 100):
    """Tail recent log lines — Docker containers via SDK, systemd units via journalctl."""
    if service not in _ALL_LOG_SERVICES:
        raise HTTPException(status_code=422,
                            detail=f"Unknown service. Valid: {sorted(_ALL_LOG_SERVICES)}")

    if service in _LOG_SYSTEMD:
        try:
            result = subprocess.run(
                ["journalctl", "--directory", "/var/log/journal",
                 "-u", f"{service}.service", "--no-pager", "-n", str(lines)],
                capture_output=True, text=True, timeout=10
            )
            return {"service": service, "lines": result.stdout.splitlines()}
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc))

    container_name = _LOG_DOCKER[service]
    try:
        import docker
        client = docker.from_env()
        container = client.containers.get(container_name)
        raw = container.logs(tail=lines, timestamps=False).decode("utf-8", errors="replace")
        return {"service": service, "container": container_name, "lines": raw.splitlines()}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/merllm/transitions")
async def transition_history():
    return db.list_transitions(limit=50)


@app.get("/api/merllm/geoip")
async def geoip_info(request: Request):
    ip = _client_ip(request)
    offset = geoip.get_utc_offset(ip)
    end_h, end_m = geoip.day_end_utc(config.BASE_DAY_END_LOCAL, offset)
    return {
        "client_ip":        ip,
        "utc_offset":       offset,
        "base_day_end":     config.BASE_DAY_END_LOCAL,
        "adjusted_day_end_utc": f"{end_h:02d}:{end_m:02d}",
        "geoip_db_present": os.path.exists(config.GEOIP_DB_PATH),
        "manual_override":  config.GEOIP_OFFSET_OVERRIDE or None,
    }


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
    return {"ok": True, "mode": mode_manager.current_mode().value}
