"""
queue_manager.py — Unified GPU queue with per-request tracking.

Every GPU request (interactive chat/generate, embeddings, batch jobs) is
registered in an in-memory tracker before acquiring a GPU slot.  This gives
the dashboard full visibility into what is queued, running, and recently
completed on each GPU.

Concurrency is still controlled by per-GPU asyncio.Semaphores (one slot per
GPU by default).  The tracker layers metadata on top so the UI can show
source app, model, request type, elapsed time, and target GPU for every
request in the system.

Batch jobs remain SQLite-persisted for retry/review.  Interactive requests
are transient — tracked while in-flight, removed on completion.
"""
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

import httpx

log = logging.getLogger(__name__)

import config
import db
import notifications

PRIORITY_INTERACTIVE = 0
PRIORITY_BATCH       = 1

# ── Per-GPU slot management ───────────────────────────────────────────────────

_gpu_semaphores: dict[str, asyncio.Semaphore] = {}


def _gpu_sem(target: str) -> asyncio.Semaphore:
    if target not in _gpu_semaphores:
        _gpu_semaphores[target] = asyncio.Semaphore(config.GPU_MAX_CONCURRENT)
    return _gpu_semaphores[target]


def gpu_slot_busy(target: str) -> bool:
    """Return True if all GPU_MAX_CONCURRENT slots for target are occupied."""
    return _gpu_sem(target)._value == 0


# ── Request tracking ─────────────────────────────────────────────────────────

@dataclass
class TrackedRequest:
    id: str
    source: str            # "lancellmot", "parsival", "direct", ...
    request_type: str      # "chat", "generate", "embeddings", "batch"
    model: str
    priority: int
    target: str            # GPU URL
    status: str            # "queued", "running", "completed", "failed"
    submitted_at: float
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    batch_job_id: Optional[str] = None


_tracked: dict[str, TrackedRequest] = {}

# Callback notified on every track/start/complete/fail so the SSE layer can
# push updates.  Set by app.py at startup.
_on_queue_change: Optional[callable] = None


def set_queue_change_callback(cb: callable) -> None:
    global _on_queue_change
    _on_queue_change = cb


def _notify_change() -> None:
    if _on_queue_change:
        try:
            _on_queue_change()
        except Exception:
            pass


def track_request(source: str, request_type: str, model: str,
                  priority: int, target: str,
                  batch_job_id: Optional[str] = None) -> str:
    """Register a new request in the tracker. Returns a tracking ID."""
    req_id = str(uuid.uuid4())
    _tracked[req_id] = TrackedRequest(
        id=req_id,
        source=source,
        request_type=request_type,
        model=model,
        priority=priority,
        target=target,
        status="queued",
        submitted_at=time.time(),
        batch_job_id=batch_job_id,
    )
    _notify_change()
    return req_id


async def acquire_gpu_slot(target: str, priority: int,
                           tracking_id: Optional[str] = None) -> bool:
    """
    Acquire a generation slot for `target`.

    If tracking_id is provided, the tracked entry transitions to "running"
    on success.

    Interactive priority: waits up to INTERACTIVE_QUEUE_TIMEOUT seconds.
    Returns False (caller should 503) if the slot is not available in time.

    Batch priority: waits indefinitely; always returns True.
    """
    sem = _gpu_sem(target)
    if priority == PRIORITY_INTERACTIVE:
        try:
            await asyncio.wait_for(sem.acquire(), timeout=config.INTERACTIVE_QUEUE_TIMEOUT)
        except asyncio.TimeoutError:
            if tracking_id and tracking_id in _tracked:
                _tracked[tracking_id].status = "failed"
                _tracked[tracking_id].error = "Timed out waiting for GPU slot"
                _tracked[tracking_id].completed_at = time.time()
                _notify_change()
                # Clean up failed interactive entries after a short delay
                asyncio.get_event_loop().call_later(10, _remove_tracked, tracking_id)
            return False
    else:
        await sem.acquire()

    if tracking_id and tracking_id in _tracked:
        _tracked[tracking_id].status = "running"
        _tracked[tracking_id].started_at = time.time()
        _notify_change()

    return True


def release_gpu_slot(target: str, tracking_id: Optional[str] = None) -> None:
    """Release a slot acquired by acquire_gpu_slot."""
    _gpu_semaphores[target].release()

    if tracking_id and tracking_id in _tracked:
        entry = _tracked[tracking_id]
        entry.status = "completed"
        entry.completed_at = time.time()
        _notify_change()
        # Interactive requests: remove from tracker shortly after completion.
        # Batch requests stay visible (persisted in SQLite anyway).
        if entry.priority == PRIORITY_INTERACTIVE:
            asyncio.get_event_loop().call_later(5, _remove_tracked, tracking_id)


def fail_request(tracking_id: str, error: str = "") -> None:
    """Mark a tracked request as failed (e.g. Ollama error during generation)."""
    if tracking_id in _tracked:
        entry = _tracked[tracking_id]
        entry.status = "failed"
        entry.error = error
        entry.completed_at = time.time()
        _notify_change()
        if entry.priority == PRIORITY_INTERACTIVE:
            asyncio.get_event_loop().call_later(10, _remove_tracked, tracking_id)


def _remove_tracked(tracking_id: str) -> None:
    _tracked.pop(tracking_id, None)
    _notify_change()


# ── Queue visibility ─────────────────────────────────────────────────────────


def active_queue() -> list[dict]:
    """
    Return all currently tracked requests (queued + running + recently
    completed), sorted by submitted_at.
    """
    now = time.time()
    result = []
    for entry in sorted(_tracked.values(), key=lambda e: e.submitted_at):
        d = {
            "id":            entry.id,
            "source":        entry.source,
            "request_type":  entry.request_type,
            "model":         entry.model,
            "priority":      "interactive" if entry.priority == PRIORITY_INTERACTIVE else "batch",
            "target":        entry.target,
            "status":        entry.status,
            "submitted_at":  entry.submitted_at,
            "started_at":    entry.started_at,
            "completed_at":  entry.completed_at,
            "error":         entry.error,
            "batch_job_id":  entry.batch_job_id,
        }
        if entry.status == "running" and entry.started_at:
            d["elapsed_sec"] = round(now - entry.started_at, 1)
        elif entry.status == "queued":
            d["waiting_sec"] = round(now - entry.submitted_at, 1)
        result.append(d)
    return result


def queue_depth() -> dict:
    """Summary counts for the status endpoint."""
    queued = sum(1 for e in _tracked.values() if e.status == "queued")
    running = sum(1 for e in _tracked.values() if e.status == "running")

    gpus: dict = {}
    for entry in _tracked.values():
        if entry.status in ("queued", "running"):
            gpu = gpus.setdefault(entry.target, {"queued": 0, "running": 0})
            gpu[entry.status] += 1

    return {"queued": queued, "running": running, "total": queued + running, "gpus": gpus}


# ── Batch job submission ──────────────────────────────────────────────────────


def submit_batch_job(source_app: str, model: str, prompt: str,
                     options: Optional[dict] = None) -> str:
    """
    Persist a batch job and enqueue it for processing.

    Returns the job ID. The job runs at batch priority whenever a GPU is free.
    """
    job_id = str(uuid.uuid4())
    db.insert_batch_job(job_id, source_app, model or config.DEFAULT_MODEL,
                        prompt, options or {})
    asyncio.ensure_future(_run_batch_job_async(job_id))
    return job_id


async def _run_batch_job_async(job_id: str) -> None:
    """Execute a batch job through the tracked GPU queue."""
    import gpu_router

    job = db.get_batch_job(job_id)
    if not job or job["status"] != "queued":
        return

    retries = job.get("retries", 0)
    db.update_batch_job(job_id, status="running", started_at=time.time())

    options: dict = {}
    try:
        options = json.loads(job.get("options", "{}"))
    except (json.JSONDecodeError, TypeError):
        pass

    body = {
        "model":   job["model"],
        "prompt":  job["prompt"],
        "stream":  False,
        "options": options,
    }

    target = gpu_router.get_target_url(job["model"])

    # Track this batch job in the unified queue
    tracking_id = track_request(
        source=job["source_app"],
        request_type="batch",
        model=job["model"],
        priority=PRIORITY_BATCH,
        target=target,
        batch_job_id=job_id,
    )

    try:
        acquired = await acquire_gpu_slot(target, PRIORITY_BATCH, tracking_id)
        if not acquired:
            raise RuntimeError("failed to acquire GPU slot")
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
                resp = await client.post(f"{target}/api/generate", json=body)
                resp.raise_for_status()
                result_text = resp.json().get("response", "")
            gpu_router.record_activity(target)
            db.update_batch_job(job_id, status="completed",
                                completed_at=time.time(), result=result_text)
            log.info("batch job %s completed", job_id[:8])
            completed_job = db.get_batch_job(job_id)
            if completed_job:
                asyncio.ensure_future(notifications.dispatch(
                    completed_job,
                    webhook_url=config.NOTIFICATION_WEBHOOK_URL or None,
                ))
        finally:
            release_gpu_slot(target, tracking_id)
    except Exception as exc:
        fail_request(tracking_id, str(exc))
        prev_error = (job.get("error") or "").strip()
        if retries < config.BATCH_MAX_RETRIES:
            backoff    = 30 * (4 ** retries)
            retry_after = time.time() + backoff
            new_error  = f"{prev_error} [attempt {retries + 1} failed: {exc}]".lstrip()
            db.update_batch_job(
                job_id,
                status="queued",
                retries=retries + 1,
                retry_after=retry_after,
                started_at=None,
                error=new_error,
            )
            log.warning(
                "batch job %s failed (attempt %d/%d), retrying in %ds: %s",
                job_id[:8], retries + 1, config.BATCH_MAX_RETRIES + 1, backoff, exc,
            )
            await asyncio.sleep(backoff)
            await _run_batch_job_async(job_id)
        else:
            final_error = f"{prev_error} [final failure after {retries + 1} attempts: {exc}]".lstrip()
            db.update_batch_job(job_id, status="failed",
                                completed_at=time.time(), error=final_error)
            log.error("batch job %s permanently failed after %d attempt(s): %s",
                      job_id[:8], retries + 1, exc)
            failed_job = db.get_batch_job(job_id)
            if failed_job:
                asyncio.ensure_future(notifications.dispatch(
                    failed_job,
                    webhook_url=config.NOTIFICATION_WEBHOOK_URL or None,
                ))


def get_queue_position(job_id: str) -> Optional[int]:
    """
    Return the zero-based queue position of a queued job (number of jobs ahead).

    Returns None if the job is not in queued status.
    """
    jobs = db.list_batch_jobs(status="queued")
    queued_ids = [j["id"] for j in sorted(jobs, key=lambda j: j.get("submitted_at", ""))]
    try:
        return queued_ids.index(job_id)
    except ValueError:
        return None


def get_job_status(job_id: str) -> Optional[dict]:
    job = db.get_batch_job(job_id)
    if not job:
        return None
    result: dict = {
        "id":           job["id"],
        "source_app":   job["source_app"],
        "model":        job["model"],
        "status":       job["status"],
        "submitted_at": job["submitted_at"],
        "started_at":   job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "error":        job.get("error"),
        "retries":      job.get("retries", 0),
        "retry_after":  job.get("retry_after"),
    }
    if job["status"] == "queued":
        result["queue_position"] = get_queue_position(job_id)
        result["estimated_start"] = None
    return result


def get_job_result(job_id: str) -> Optional[dict]:
    job = db.get_batch_job(job_id)
    if not job or job["status"] != "completed":
        return None
    return {
        "id":     job["id"],
        "result": job.get("result", ""),
    }
