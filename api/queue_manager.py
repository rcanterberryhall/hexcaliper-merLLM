"""
queue_manager.py — Priority queue and batch job execution.

Maintains an asyncio PriorityQueue with two levels:
  0 = interactive (high priority)
  1 = batch       (low priority)

Batch jobs queued via POST /api/batch/submit are held in SQLite and executed
when night mode activates (or immediately if night mode is already active).
The night-mode batch runner processes jobs sequentially against the dual-GPU
Ollama instance with extended context.

Failed jobs are automatically retried up to BATCH_MAX_RETRIES times with
exponential backoff (30s, 120s, …). The accumulated error history is
preserved in the job's error field.
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
import mode_manager
import notifications

PRIORITY_INTERACTIVE = 0
PRIORITY_BATCH       = 1

_queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
_batch_event: asyncio.Event   = asyncio.Event()   # signalled when night mode starts

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
    """Process queued requests sequentially (one Ollama call at a time per GPU)."""
    while True:
        item: QueuedRequest = await _queue.get()
        if item.future.cancelled():
            _queue.task_done()
            continue
        try:
            async with mode_manager.InFlightGuard():
                result = await _forward(item.target, item.body)
            item.future.set_result(result)
        except Exception as exc:
            if not item.future.done():
                item.future.set_exception(exc)
        finally:
            _queue.task_done()


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
    Persist a batch job for night-mode execution.

    Returns the job ID. The job stays 'queued' until night mode activates.
    """
    job_id = str(uuid.uuid4())
    db.insert_batch_job(job_id, source_app, model or config.NIGHT_MODEL,
                        prompt, options or {})
    return job_id


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
        result["estimated_start"] = None   # no per-job duration data available
    return result


def get_job_result(job_id: str) -> Optional[dict]:
    job = db.get_batch_job(job_id)
    if not job or job["status"] != "completed":
        return None
    return {
        "id":     job["id"],
        "result": job.get("result", ""),
    }


def signal_night_mode() -> None:
    """Call when night mode activates to start draining the batch queue."""
    _batch_event.set()


# ── Night-mode batch runner ───────────────────────────────────────────────────


async def batch_runner_loop() -> None:
    """
    Execute queued batch jobs during night mode.

    Waits for the night-mode signal, then loops until either:
    - All jobs (including deferred retries) are processed, or
    - Night mode ends.

    Deferred retries (jobs with a future retry_after timestamp) are
    waited for in-place — the runner sleeps until the earliest one is
    ready rather than exiting and losing the night-mode window.
    """
    while True:
        await _batch_event.wait()
        _batch_event.clear()

        if mode_manager.current_mode() != mode_manager.Mode.NIGHT:
            continue

        log.info("batch runner activated")

        while mode_manager.current_mode() == mode_manager.Mode.NIGHT:
            ready = db.list_batch_jobs(status="queued", ready_only=True)
            if ready:
                log.info("batch: processing %d ready job(s)", len(ready))
                for job in ready:
                    if mode_manager.current_mode() != mode_manager.Mode.NIGHT:
                        log.info("batch: mode changed — pausing")
                        break
                    await _run_batch_job(job)
            else:
                next_ts = db.get_earliest_retry_after()
                if next_ts is None:
                    log.info("batch: queue empty, going idle")
                    break
                wait_sec = max(1.0, next_ts - time.time())
                log.info("batch: %.0fs until next retry", wait_sec)
                await asyncio.sleep(min(wait_sec, 60.0))


async def _run_batch_job(job: dict) -> None:
    """
    Execute one batch job against Ollama.

    On failure, automatically retries up to BATCH_MAX_RETRIES times with
    exponential backoff (30s × 4^attempt). Each retry increments the
    job's retry counter and sets retry_after. On final failure the job
    is marked 'failed' with accumulated error history.
    """
    job_id  = job["id"]
    retries = job.get("retries", 0)
    db.update_batch_job(job_id, status="running", started_at=time.time())

    options: dict = {}
    try:
        options = json.loads(job.get("options", "{}"))
    except (json.JSONDecodeError, TypeError):
        pass

    options.setdefault("num_ctx", config.NIGHT_NUM_CTX)

    body = {
        "model":   job["model"],
        "prompt":  job["prompt"],
        "stream":  False,
        "options": options,
    }

    target = config.OLLAMA_0_URL   # night mode: single dual-GPU instance

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(None)) as client:
            resp = await client.post(f"{target}/api/generate", json=body)
            resp.raise_for_status()
            result_text = resp.json().get("response", "")
        db.update_batch_job(job_id, status="completed",
                            completed_at=time.time(), result=result_text)
        log.info("batch job %s completed", job_id[:8])
        completed_job = db.get_batch_job(job_id)
        if completed_job:
            asyncio.ensure_future(notifications.dispatch(
                completed_job,
                webhook_url=config.NOTIFICATION_WEBHOOK_URL or None,
            ))
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
