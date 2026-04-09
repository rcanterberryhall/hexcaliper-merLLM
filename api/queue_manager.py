"""
queue_manager.py — Priority queue and GPU slot management.

Maintains an asyncio PriorityQueue with two levels:
  0 = interactive (high priority)
  1 = batch       (low priority)

Both levels run whenever a GPU slot is free. Interactive requests time out
after INTERACTIVE_QUEUE_TIMEOUT seconds; batch requests wait indefinitely.

Failed requests are retried on the other GPU when possible (failover).
"""
import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

log = logging.getLogger(__name__)

import config
import db
import notifications

PRIORITY_INTERACTIVE = 0
PRIORITY_BATCH       = 1

_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()

# ── Per-GPU slot management ───────────────────────────────────────────────────
# One asyncio.Semaphore per Ollama target URL, capped at GPU_MAX_CONCURRENT.
# Interactive requests wait up to INTERACTIVE_QUEUE_TIMEOUT seconds then give up.
# Batch requests wait indefinitely. Semaphore waiters are FIFO within priority.

_gpu_semaphores: dict[str, asyncio.Semaphore] = {}
_gpu_in_flight:  dict[str, int]               = {}


def _gpu_sem(target: str) -> asyncio.Semaphore:
    if target not in _gpu_semaphores:
        _gpu_semaphores[target] = asyncio.Semaphore(config.GPU_MAX_CONCURRENT)
        _gpu_in_flight[target]  = 0
    return _gpu_semaphores[target]


async def acquire_gpu_slot(target: str, priority: int) -> bool:
    """
    Acquire a generation slot for `target`.

    Interactive priority: waits up to INTERACTIVE_QUEUE_TIMEOUT seconds.
    Returns False (→ caller should 503) if the slot is not available in time.

    Batch priority: waits indefinitely; always returns True.
    """
    sem = _gpu_sem(target)
    if priority == PRIORITY_INTERACTIVE:
        try:
            await asyncio.wait_for(sem.acquire(), timeout=config.INTERACTIVE_QUEUE_TIMEOUT)
        except asyncio.TimeoutError:
            return False
    else:
        await sem.acquire()
    _gpu_in_flight[target] = _gpu_in_flight.get(target, 0) + 1
    return True


def release_gpu_slot(target: str) -> None:
    """Release a slot acquired by acquire_gpu_slot."""
    _gpu_semaphores[target].release()
    _gpu_in_flight[target] = max(0, _gpu_in_flight.get(target, 0) - 1)


def gpu_slot_busy(target: str) -> bool:
    """Return True if all GPU_MAX_CONCURRENT slots for target are occupied."""
    return _gpu_sem(target)._value == 0


@dataclass(order=True)
class QueuedRequest:
    priority: int
    ts:       float
    future:   Any = field(compare=False)
    target:   str = field(compare=False)
    body:     dict = field(compare=False)


# ── Public submission ─────────────────────────────────────────────────────────


async def enqueue(target: str, body: dict, priority: int = PRIORITY_INTERACTIVE) -> asyncio.Future:
    """
    Add an Ollama request to the priority queue.

    Returns a Future that resolves to (status_code, content, headers) when
    the request completes.
    """
    future: asyncio.Future = asyncio.get_event_loop().create_future()
    item = QueuedRequest(
        priority=priority,
        ts=time.time(),
        future=future,
        target=target,
        body=body,
    )
    await _queue.put(item)
    return future


# ── Worker ────────────────────────────────────────────────────────────────────


async def worker_loop() -> None:
    """Process queued requests (one Ollama call at a time per GPU)."""
    import gpu_router

    while True:
        item: QueuedRequest = await _queue.get()
        if item.future.cancelled():
            _queue.task_done()
            continue
        try:
            result = await _forward(item.target, item.body)
            gpu_router.record_activity(item.target)
            item.future.set_result(result)
        except Exception as exc:
            log.error("request to %s failed: %s", item.target, exc)
            # Failover: try the other GPU if healthy.
            other = _other_gpu(item.target)
            if other:
                try:
                    log.info("failing over to %s", other)
                    result = await _forward(other, item.body)
                    gpu_router.record_activity(other)
                    item.future.set_result(result)
                except Exception as exc2:
                    log.error("failover to %s also failed: %s", other, exc2)
                    gpu_router.mark_failed(other)
                    if not item.future.done():
                        item.future.set_exception(exc2)
            else:
                if not item.future.done():
                    item.future.set_exception(exc)
            gpu_router.mark_failed(item.target)
        finally:
            _queue.task_done()


def _other_gpu(target: str) -> Optional[str]:
    """Return the other GPU URL if it's healthy, else None."""
    import gpu_router
    urls = [config.OLLAMA_0_URL, config.OLLAMA_1_URL]
    for url in urls:
        if url != target:
            state = gpu_router._gpus.get(url)
            if state and state.health == gpu_router.GpuHealth.HEALTHY:
                return url
    return None


async def _forward(target: str, body: dict) -> tuple[int, bytes, dict]:
    """Make an Ollama HTTP request and return (status, body, headers)."""
    async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
        resp = await client.post(f"{target}/api/generate", json=body)
        return resp.status_code, resp.content, dict(resp.headers)


def queue_depth() -> dict:
    result: dict = {"total": _queue.qsize(), "gpus": {}}
    for target, in_flight in _gpu_in_flight.items():
        result["gpus"][target] = {"in_flight": in_flight}
    return result


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
    # Schedule async execution.
    asyncio.ensure_future(_run_batch_job_async(job_id))
    return job_id


async def _run_batch_job_async(job_id: str) -> None:
    """Execute a batch job through the normal priority queue."""
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

    try:
        acquired = await acquire_gpu_slot(target, PRIORITY_BATCH)
        if not acquired:
            # Should not happen for batch (waits indefinitely), but guard.
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
            release_gpu_slot(target)
    except Exception as exc:
        prev_error = (job.get("error") or "").strip()
        if retries < config.BATCH_MAX_RETRIES:
            backoff    = 30 * (4 ** retries)          # 30s, 120s, 480s …
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
