"""
queue_manager.py — Priority queue and batch job execution.

Maintains an asyncio PriorityQueue with two levels:
  0 = interactive (high priority)
  1 = batch       (low priority)

Batch jobs queued via POST /api/batch/submit are held in SQLite and executed
when night mode activates (or immediately if night mode is already active).
The night-mode batch runner processes jobs sequentially against the dual-GPU
Ollama instance with extended context.
"""
import asyncio
import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

import httpx

import config
import db
import mode_manager

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


def get_job_status(job_id: str) -> Optional[dict]:
    job = db.get_batch_job(job_id)
    if not job:
        return None
    options = {}
    try:
        options = json.loads(job.get("options", "{}"))
    except (json.JSONDecodeError, TypeError):
        pass
    return {
        "id":           job["id"],
        "source_app":   job["source_app"],
        "model":        job["model"],
        "status":       job["status"],
        "submitted_at": job["submitted_at"],
        "started_at":   job.get("started_at"),
        "completed_at": job.get("completed_at"),
        "error":        job.get("error"),
    }


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

    Waits for the night-mode signal, then processes all queued jobs
    sequentially with extended context. Returns to waiting when the
    queue is empty or day mode resumes.
    """
    while True:
        await _batch_event.wait()
        _batch_event.clear()

        if mode_manager.current_mode() != mode_manager.Mode.NIGHT:
            continue

        queued = db.list_batch_jobs(status="queued")
        if not queued:
            continue

        print(f"[batch] night mode active — processing {len(queued)} job(s)")

        for job in queued:
            if mode_manager.current_mode() != mode_manager.Mode.NIGHT:
                print("[batch] mode changed — pausing batch processing")
                break
            await _run_batch_job(job)


async def _run_batch_job(job: dict) -> None:
    job_id = job["id"]
    db.update_batch_job(job_id, status="running", started_at=time.time())

    options = {}
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
        print(f"[batch] job {job_id[:8]} completed")
    except Exception as exc:
        db.update_batch_job(job_id, status="failed",
                            completed_at=time.time(), error=str(exc))
        print(f"[batch] job {job_id[:8]} failed: {exc}")
