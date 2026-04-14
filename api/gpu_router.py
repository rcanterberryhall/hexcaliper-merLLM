"""
gpu_router.py — Thermal monitoring and UNREACHABLE-slot recovery.

The Slot FSM in ``scheduler.py`` owns slot lifecycle and ``queue_manager``
owns the tick loop and effect application. This module is the thin I/O
bridge between hardware-level signals (GPU temperature, ollama
reachability) and the FSM's THERMAL_TRIP / THERMAL_CLEAR / PROBE_OK events.

Responsibilities:

  * ``update_thermal_state(url, temp_c)`` — called from the metrics
    collection loop. Applies hysteresis and feeds Event.THERMAL_TRIP /
    Event.THERMAL_CLEAR into the FSM.
  * ``recovery_loop()`` — background task that periodically probes
    UNREACHABLE slots and feeds Event.PROBE_OK on success.
  * ``record_activity(url)`` — bumps a per-slot last-active timestamp for
    the UI's ``idle_seconds`` display. Observability only; no control flow.
  * ``reset_slot(url)`` — operator action behind /api/merllm/gpu/{gpu}/reset.
  * ``status()`` — projects current slot state for the status endpoint.
"""
import asyncio
import logging
import time
from typing import Optional

import config
import queue_manager
from scheduler import Event, InvalidTransition, SlotState

log = logging.getLogger(__name__)


# ── Module state ─────────────────────────────────────────────────────────────
# Per-URL caches. Hysteresis state lives here so update_thermal_state can
# suppress duplicate trips/clears without re-querying the FSM.
_thermal_paused: dict[str, bool] = {}
_last_temp_c: dict[str, Optional[float]] = {}
_thermal_paused_since: dict[str, Optional[float]] = {}

# UI-only last-activity timestamp per URL, fed by record_activity().
_last_active: dict[str, float] = {}


def _slot_idx(url: str) -> Optional[int]:
    for i, s in enumerate(queue_manager._slots):
        if s.url == url:
            return i
    return None


def _feed(idx: int, event: Event) -> None:
    """Feed an event into the FSM and apply produced effects.

    Swallows :class:`InvalidTransition` — gpu_router fires events from
    coarse external signals (temperature crossings, periodic probes) and
    the FSM may legitimately be in a state where the event has no defined
    transition (e.g. THERMAL_TRIP from UNREACHABLE). Logging at DEBUG keeps
    those silent at rest while preserving forensics.
    """
    try:
        for e in queue_manager._feed_event(idx, event):
            queue_manager._apply_effect(e)
    except InvalidTransition as exc:
        log.debug("[gpu_router] skip %s: %s", event.value, exc)


# ── Activity tracking ────────────────────────────────────────────────────────

def record_activity(url: str) -> None:
    """Bump the per-URL last-active timestamp for the UI's idle counter."""
    _last_active[url] = time.time()


# ── Thermal management ───────────────────────────────────────────────────────

def update_thermal_state(target: str, temp_c: float) -> None:
    """Update a GPU's thermal state from a fresh temperature reading.

    Hysteresis:
      * ``temp >= GPU_TEMP_PAUSE_C``  → feed Event.THERMAL_TRIP into the FSM
      * ``temp <= GPU_TEMP_RESUME_C`` → feed Event.THERMAL_CLEAR into the FSM
      * in between → no transition (preserves current state)

    THERMAL_TRIP from BUSY/LOADING latches and only takes effect at the
    next WORK_END / LOAD_DONE; from READY it drives directly to COOLING.
    THERMAL_CLEAR drives COOLING → READY and wakes the tick loop so any
    backed-up bucket head gets a chance immediately.
    """
    _last_temp_c[target] = temp_c
    was_paused = _thermal_paused.get(target, False)

    if temp_c >= config.GPU_TEMP_PAUSE_C and not was_paused:
        _thermal_paused[target] = True
        _thermal_paused_since[target] = time.time()
        log.warning(
            "GPU %s thermally paused at %.1f°C (threshold %d°C) — "
            "blocking new dispatch until it cools below %d°C",
            target, temp_c, config.GPU_TEMP_PAUSE_C, config.GPU_TEMP_RESUME_C,
        )
        idx = _slot_idx(target)
        if idx is not None:
            _feed(idx, Event.THERMAL_TRIP)

    elif temp_c <= config.GPU_TEMP_RESUME_C and was_paused:
        _thermal_paused[target] = False
        since = _thermal_paused_since.get(target)
        paused_for = (time.time() - since) if since else 0
        _thermal_paused_since[target] = None
        log.info(
            "GPU %s thermal pause cleared at %.1f°C after %.0fs — resuming dispatch",
            target, temp_c, paused_for,
        )
        idx = _slot_idx(target)
        if idx is not None:
            _feed(idx, Event.THERMAL_CLEAR)
            queue_manager._wake_tick(reason="thermal_resume")


# ── UNREACHABLE recovery ─────────────────────────────────────────────────────

async def recovery_loop() -> None:
    """Periodically probe UNREACHABLE slots; feed PROBE_OK on recovery.

    Per-slot exponential backoff between HEALTH_BACKOFF_BASE and
    HEALTH_BACKOFF_CAP. After HEALTH_FAULT_TIMEOUT seconds of continuous
    failures the slot stays UNREACHABLE and only retries at the cap rate
    until an operator forces a reset via /api/merllm/gpu/{gpu}/reset.

    Slots that leave UNREACHABLE clear their backoff state, so a flap
    doesn't keep amplifying intervals.
    """
    next_probe: dict[str, float] = {}
    interval:   dict[str, float] = {}
    fail_since: dict[str, float] = {}

    while True:
        await asyncio.sleep(5)
        now = time.time()

        for i, s in enumerate(list(queue_manager._slots)):
            url = s.url
            if s.state is not SlotState.UNREACHABLE:
                next_probe.pop(url, None)
                interval.pop(url, None)
                fail_since.pop(url, None)
                continue

            if url not in fail_since:
                fail_since[url] = now
                interval[url]   = float(config.HEALTH_BACKOFF_BASE)
                next_probe[url] = now + interval[url]
                continue

            if now < next_probe[url]:
                continue

            if (now - fail_since[url]) >= config.HEALTH_FAULT_TIMEOUT:
                # Past the fault deadline — back off to the cap and wait
                # for an operator reset rather than spinning forever.
                interval[url]   = float(config.HEALTH_BACKOFF_CAP)
                next_probe[url] = now + interval[url]

            ok = await queue_manager._probe_url(url)
            if ok:
                log.info("GPU %s recovered — feeding PROBE_OK", url)
                _feed(i, Event.PROBE_OK)
                queue_manager._wake_tick(reason="probe_recovery")
                next_probe.pop(url, None)
                interval.pop(url, None)
                fail_since.pop(url, None)
            else:
                interval[url]   = min(interval[url] * 2,
                                      float(config.HEALTH_BACKOFF_CAP))
                next_probe[url] = now + interval[url]
                log.info("GPU %s probe failed (next in %.0fs)",
                         url, interval[url])


def reset_slot(target: str) -> bool:
    """Operator action: probe the slot and feed the result into the FSM.

    Used by /api/merllm/gpu/{gpu}/reset to coax a wedged UNREACHABLE slot
    back into rotation without restarting the container. Returns False if
    ``target`` is not one of the configured GPU URLs.
    """
    idx = _slot_idx(target)
    if idx is None:
        return False
    asyncio.ensure_future(_reset_and_probe(idx, target))
    return True


async def _reset_and_probe(idx: int, url: str) -> None:
    ok = await queue_manager._probe_url(url)
    _feed(idx, Event.PROBE_OK if ok else Event.PROBE_FAIL)
    queue_manager._wake_tick(reason="operator_reset")


# ── Status ───────────────────────────────────────────────────────────────────

# Map FSM SlotState onto the dashboard's older health vocabulary so the
# UI keeps colouring tiles correctly. Drop in a follow-up once the
# dashboard reads ``state`` directly.
_HEALTH_BY_STATE = {
    SlotState.UNKNOWN:     "unknown",
    SlotState.READY:       "healthy",
    SlotState.LOADING:     "healthy",
    SlotState.BUSY:        "healthy",
    SlotState.COOLING:     "healthy",
    SlotState.UNREACHABLE: "degraded",
    SlotState.DRAINING:    "draining",
}


def status() -> dict:
    """GPU state for the UI and /api/merllm/status endpoint.

    Reads slot lifecycle from queue_manager (FSM-owned) and merges in the
    thermal + activity metadata that lives in this module.
    """
    now = time.time()
    gpus: dict = {}
    for label, url in [("gpu0", config.OLLAMA_0_URL), ("gpu1", config.OLLAMA_1_URL)]:
        idx = _slot_idx(url)
        if idx is None:
            continue
        s = queue_manager._slots[idx]
        last_active = _last_active.get(url)
        gpus[label] = {
            "url":            url,
            "model":          s.model_loaded,
            "state":          s.state.value,
            "health":         _HEALTH_BY_STATE.get(s.state, "unknown"),
            "idle_seconds":   round(now - last_active, 1) if last_active else None,
            "in_flight":      s.state in (SlotState.BUSY, SlotState.LOADING),
            "thermal_paused": _thermal_paused.get(url, False),
            "last_temp_c":    _last_temp_c.get(url),
            "thermal_paused_since": _thermal_paused_since.get(url),
        }
    return {
        "routing":           "fsm",
        "default_model":     config.DEFAULT_MODEL,
        "gpu_temp_pause_c":  config.GPU_TEMP_PAUSE_C,
        "gpu_temp_resume_c": config.GPU_TEMP_RESUME_C,
        "gpus":              gpus,
    }
