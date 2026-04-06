"""
mode_manager.py — Day/night operating mode state machine.

Day mode:  Two Ollama instances, one per GPU. Interactive requests go to GPU 0
           (qwen3:32b). Batch/analysis requests go to GPU 1 (qwen3:30b-a3b).

Night mode: One Ollama instance spanning both GPUs. qwen3:32b with extended
            context (32K+). All requests serialised through this instance.

Transitions are triggered by inactivity timeout, schedule hint, or manual
override. The transition procedure drains in-flight requests before touching
Ollama.

Ollama lifecycle management (start/stop services) is performed via systemctl
when OLLAMA_MANAGE_VIA = "systemctl". Set OLLAMA_MANAGE_VIA = "none" to
operate in proxy-only mode (Ollama services managed externally).
"""
import asyncio
import logging
import subprocess
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Callable, Optional

import config
import db
import geoip

log = logging.getLogger(__name__)

# ── State ─────────────────────────────────────────────────────────────────────


class Mode(str, Enum):
    DAY          = "day"
    NIGHT        = "night"
    TRANSITIONING = "transitioning"


_mode: Mode = Mode.DAY
_override: Optional[Mode] = None        # None = auto
_last_interactive: float = time.time()
_transition_lock = asyncio.Lock()
_last_client_ip: str = "127.0.0.1"
_in_flight: int = 0
_in_flight_lock = asyncio.Lock()
_on_mode_change: list[Callable] = []    # callbacks for UI push


# ── Public accessors ──────────────────────────────────────────────────────────


def current_mode() -> Mode:
    return _mode


def current_override() -> Optional[str]:
    return _override.value if _override else None


def record_interactive(client_ip: str = "127.0.0.1") -> None:
    """Record that an interactive request just arrived."""
    global _last_interactive, _last_client_ip
    _last_interactive = time.time()
    _last_client_ip = client_ip


def get_target_url(model: str) -> str:
    """
    Return the Ollama instance URL that should serve this model.

    In day mode, routes by model name. In night mode, all traffic goes to
    the dual-GPU instance on OLLAMA_0_URL.
    """
    if _mode == Mode.NIGHT or _mode == Mode.TRANSITIONING:
        return config.OLLAMA_0_URL
    # Day mode: route by model name
    if model == config.DAY_MODEL_GPU1:
        return config.OLLAMA_1_URL
    return config.OLLAMA_0_URL


def status() -> dict:
    elapsed = time.time() - _last_interactive
    return {
        "mode":             _mode.value,
        "override":         _override.value if _override else None,
        "last_interactive": _last_interactive,
        "idle_seconds":     elapsed,
        "inactivity_timeout_min": config.INACTIVITY_TIMEOUT_MIN,
        "gpu0_url":         config.OLLAMA_0_URL,
        "gpu1_url":         config.OLLAMA_1_URL,
        "day_model_gpu0":   config.DAY_MODEL_GPU0,
        "day_model_gpu1":   config.DAY_MODEL_GPU1,
        "night_model":      config.NIGHT_MODEL,
        "night_num_ctx":    config.NIGHT_NUM_CTX,
        "in_flight":        _in_flight,
        "client_ip":        _last_client_ip,
    }


def register_mode_change_callback(fn: Callable) -> None:
    _on_mode_change.append(fn)


# ── In-flight tracking ────────────────────────────────────────────────────────


class InFlightGuard:
    """Context manager that tracks in-flight requests for drain logic."""
    async def __aenter__(self):
        global _in_flight
        async with _in_flight_lock:
            _in_flight += 1
        return self

    async def __aexit__(self, *_):
        global _in_flight
        async with _in_flight_lock:
            _in_flight -= 1


# ── Lifecycle management ──────────────────────────────────────────────────────


def _systemctl(action: str, service: str) -> bool:
    if config.OLLAMA_MANAGE_VIA != "systemctl":
        # Proxy-only mode: Ollama lifecycle is managed externally.
        # Log at INFO so operators can see the skipped step; always return True
        # because the routing change itself is the intended action.
        log.info("(proxy-only) skipping service management: systemctl %s %s", action, service)
        return True
    try:
        result = subprocess.run(
            ["systemctl", action, service],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode != 0:
            log.error("systemctl %s %s failed: %s", action, service, result.stderr.strip())
        return result.returncode == 0
    except Exception as exc:
        log.error("systemctl %s %s raised: %s", action, service, exc)
        return False


async def _drain(timeout_sec: int) -> None:
    """Wait for in-flight requests to complete."""
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        async with _in_flight_lock:
            if _in_flight == 0:
                return
        await asyncio.sleep(1)
    log.warning("drain timed out with %d requests still in flight", _in_flight)


# ── Transitions ───────────────────────────────────────────────────────────────


async def transition_to_night(trigger: str = "auto") -> bool:
    """
    Transition from day mode to night mode.

    1. Mark as transitioning (new requests queue).
    2. Drain in-flight requests.
    3. Stop GPU 1 service.
    4. Restart GPU 0 service with night config (CUDA_VISIBLE_DEVICES=0,1).
    5. Mark as night.
    """
    global _mode
    async with _transition_lock:
        if _mode == Mode.NIGHT:
            return True
        prev_mode = _mode
        start = time.time()
        _mode = Mode.TRANSITIONING
        _notify_mode_change()
        log.info("transitioning to night (trigger=%s)", trigger)

        await _drain(config.DRAIN_TIMEOUT_SEC)

        ok = True
        if config.OLLAMA_MANAGE_VIA == "systemctl":
            ok &= _systemctl("stop", config.GPU1_SERVICE)
            ok &= _systemctl("stop", config.GPU0_SERVICE)
            ok &= _systemctl("start", config.NIGHT_SERVICE)

        duration = time.time() - start
        if ok:
            _mode = Mode.NIGHT
            log.info("night mode active (%.1fs)", duration)
        else:
            _mode = prev_mode
            log.error("transition to night failed after %.1fs — rolled back to %s", duration, prev_mode.value)
        _notify_mode_change()
        db.insert_transition("day→night", trigger, duration, ok)
        return ok


async def transition_to_day(trigger: str = "interactive") -> bool:
    """
    Transition from night mode to day mode.

    1. Mark as transitioning.
    2. Wait for current batch request to finish (do not kill mid-generation).
    3. Stop night service. Restart GPU 0 (single GPU) and GPU 1.
    4. Mark as day.
    """
    global _mode
    async with _transition_lock:
        if _mode == Mode.DAY:
            return True
        prev_mode = _mode
        start = time.time()
        _mode = Mode.TRANSITIONING
        _notify_mode_change()
        log.info("transitioning to day (trigger=%s)", trigger)

        await _drain(config.DRAIN_TIMEOUT_SEC)

        ok = True
        if config.OLLAMA_MANAGE_VIA == "systemctl":
            ok &= _systemctl("stop",  config.NIGHT_SERVICE)
            ok &= _systemctl("start", config.GPU0_SERVICE)
            ok &= _systemctl("start", config.GPU1_SERVICE)

        duration = time.time() - start
        if ok:
            _mode = Mode.DAY
            log.info("day mode active (%.1fs)", duration)
        else:
            _mode = prev_mode
            log.error("transition to day failed after %.1fs — rolled back to %s", duration, prev_mode.value)
        _notify_mode_change()
        db.insert_transition("night→day", trigger, duration, ok)
        return ok


def _notify_mode_change() -> None:
    for fn in _on_mode_change:
        try:
            fn(_mode)
        except Exception:
            pass


# ── Override ──────────────────────────────────────────────────────────────────


async def set_override(mode: Optional[str]) -> bool:
    """
    Set or clear manual mode override.

    Returns True if the target mode was reached (or was already active),
    False if the transition failed. On failure the mode is rolled back by
    the transition function; the override is still recorded so the caller
    can inspect it.
    """
    global _override
    if mode is None:
        _override = None
        return True
    target = Mode(mode)
    _override = target
    ok = True
    if target == Mode.DAY and _mode != Mode.DAY:
        ok = await transition_to_day("manual_override")
    elif target == Mode.NIGHT and _mode != Mode.NIGHT:
        ok = await transition_to_night("manual_override")
    return ok


# ── Background scheduler ──────────────────────────────────────────────────────


async def scheduler_loop() -> None:
    """
    Periodically evaluate whether a mode transition is warranted.

    Checks every 60 seconds. Skips if a manual override is active or a
    transition is already in progress.
    """
    while True:
        await asyncio.sleep(60)
        if _override is not None or _mode == Mode.TRANSITIONING:
            continue
        try:
            await _evaluate_mode()
        except Exception as exc:
            print(f"[mode] scheduler error: {exc}")


async def _evaluate_mode() -> None:
    global _mode
    idle_sec = time.time() - _last_interactive
    idle_min = idle_sec / 60

    if _mode == Mode.DAY and idle_min >= config.INACTIVITY_TIMEOUT_MIN:
        if _past_schedule_hint():
            await transition_to_night("inactivity")

    elif _mode == Mode.NIGHT:
        # Night → day is triggered by interactive requests, not the scheduler.
        # The scheduler only checks whether we're past the inactivity window.
        pass


def _past_schedule_hint() -> bool:
    """Return True if the current UTC time is past the adjusted day-end hint."""
    offset = geoip.get_utc_offset(_last_client_ip)
    end_h, end_m = geoip.day_end_utc(config.BASE_DAY_END_LOCAL, offset)
    now = datetime.now(timezone.utc)
    end_minutes = end_h * 60 + end_m
    now_minutes = now.hour * 60 + now.minute
    return now_minutes >= end_minutes
