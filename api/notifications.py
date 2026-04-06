"""
notifications.py — Batch job completion notification dispatch.

Supported channels:
  webhook     — HTTP POST a JSON payload to a configured URL.
                Works with Slack incoming webhooks, ntfy.sh, Gotify, etc.

  browser     — Server-Sent Event published to any connected dashboard
                clients.  Listeners register via GET /api/merllm/events.
                Requires the browser tab to be open; covers the common
                "I'm watching the queue" case.

Web Push (full push when the browser is closed) requires a service worker
and VAPID key infrastructure.  That is intentionally out of scope here.
The webhook channel covers the fully-async notification use case via
ntfy.sh or similar.

Webhook payload::

    {
        "job_id":         "...",
        "source_app":     "parsival",
        "status":         "completed",
        "submitted_at":   "2026-04-05T00:00:00Z",
        "completed_at":   "2026-04-05T01:00:00Z",
        "prompt_preview": "first 100 chars of prompt…",
        "error":          null
    }
"""
from __future__ import annotations

import asyncio
import json
import logging
import time

import httpx

log = logging.getLogger("merllm.notifications")

# In-memory queue of SSE events for connected browser clients.
# Each entry is a raw JSON string ready to send.
_sse_queue: asyncio.Queue | None = None
_sse_listeners: list[asyncio.Queue] = []
_sse_lock: asyncio.Lock | None = None


def _init_sse() -> None:
    global _sse_lock
    if _sse_lock is None:
        try:
            loop = asyncio.get_event_loop()
            _sse_lock = asyncio.Lock()
        except RuntimeError:
            pass


def add_sse_listener() -> asyncio.Queue:
    """Register a new SSE client. Returns a queue to read events from."""
    q: asyncio.Queue = asyncio.Queue()
    _sse_listeners.append(q)
    return q


def remove_sse_listener(q: asyncio.Queue) -> None:
    """Deregister an SSE client queue."""
    try:
        _sse_listeners.remove(q)
    except ValueError:
        pass


async def _broadcast_sse(event: dict) -> None:
    """Push an event to all connected SSE listeners."""
    data = json.dumps(event)
    for q in list(_sse_listeners):
        try:
            await q.put(data)
        except Exception:
            pass


async def dispatch(job: dict, webhook_url: str | None = None) -> None:
    """
    Fire all configured notification channels for a completed (or failed) batch job.

    :param job:         Full job record dict from db.
    :param webhook_url: Optional webhook URL from settings.
    """
    payload = {
        "job_id":         job.get("id") or job.get("job_id", ""),
        "source_app":     job.get("source_app", ""),
        "status":         job.get("status", "completed"),
        "submitted_at":   job.get("submitted_at"),
        "completed_at":   job.get("completed_at"),
        "prompt_preview": (job.get("prompt") or "")[:100],
        "error":          job.get("error"),
    }

    # 1. Browser SSE
    await _broadcast_sse({"type": "job_complete", **payload})

    # 2. Webhook
    if webhook_url:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                r = await client.post(webhook_url, json=payload)
                r.raise_for_status()
                log.info("webhook delivered for job %s → %s %d",
                         payload["job_id"][:8], webhook_url, r.status_code)
        except Exception as exc:
            log.warning("webhook delivery failed for job %s: %s",
                        payload["job_id"][:8], exc)
