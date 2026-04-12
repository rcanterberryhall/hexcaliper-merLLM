"""
config.py — merLLM configuration.

All settings are read from environment variables with sensible defaults.
``apply_overrides`` hot-reloads settings saved via POST /api/merllm/settings
without restarting the container.
"""
import os
import sys


def _get(key: str, default: str = "") -> str:
    return os.environ.get(key, default).strip()


# ── Ollama instances ──────────────────────────────────────────────────────────

OLLAMA_0_URL  = _get("OLLAMA_0_URL",  "http://host.docker.internal:11434")
OLLAMA_1_URL  = _get("OLLAMA_1_URL",  "http://host.docker.internal:11435")

# ── Model assignment ──────────────────────────────────────────────────────────

DEFAULT_MODEL = _get("DEFAULT_MODEL", "qwen3:32b")

# ── GPU health ────────────────────────────────────────────────────────────────

# Seconds a GPU must be idle with a non-default model before reclaiming.
RECLAIM_TIMEOUT = int(_get("RECLAIM_TIMEOUT", "300"))

# Exponential backoff for health probes on degraded GPUs.
HEALTH_BACKOFF_BASE = int(_get("HEALTH_BACKOFF_BASE", "10"))     # initial probe interval
HEALTH_BACKOFF_CAP  = int(_get("HEALTH_BACKOFF_CAP",  "300"))    # max probe interval
HEALTH_FAULT_TIMEOUT = int(_get("HEALTH_FAULT_TIMEOUT", "1800")) # seconds until faulted

# Thermal throttling. A GPU that reaches GPU_TEMP_PAUSE_C is marked thermally
# paused — the dispatcher will not route new work to it until it cools back
# below GPU_TEMP_RESUME_C. In-flight work on a paused GPU is not interrupted;
# only new dispatches are blocked. The 25°C hysteresis prevents flapping.
GPU_TEMP_PAUSE_C  = int(_get("GPU_TEMP_PAUSE_C",  "85"))
GPU_TEMP_RESUME_C = int(_get("GPU_TEMP_RESUME_C", "60"))

# ── Storage ───────────────────────────────────────────────────────────────────

EXTRA_DISK_PATHS    = _get("EXTRA_DISK_PATHS",    "")   # e.g. "archive=/mnt/archive,data=/mnt/data"
DB_PATH             = _get("DB_PATH",             "/data/merllm.db")
METRICS_RETAIN_DAYS = int(_get("METRICS_RETAIN_DAYS", "7"))
METRICS_INTERVAL_SEC = int(_get("METRICS_INTERVAL_SEC", "10"))

# Directory where backup_db.sh and POST /api/merllm/backup write backup files.
BACKUP_DIR       = _get("BACKUP_DIR",       "/data/backups")
BACKUP_KEEP_DAYS = int(_get("BACKUP_KEEP_DAYS", "7"))

# ── Batch job execution ───────────────────────────────────────────────────────

# Max number of automatic retries for a failed batch job (default 2 → 3 total attempts).
BATCH_MAX_RETRIES = int(_get("BATCH_MAX_RETRIES", "2"))

# Maximum prompt length in characters accepted by POST /api/batch/submit.
BATCH_MAX_PROMPT_LEN = int(_get("BATCH_MAX_PROMPT_LEN", "100000"))

# ── Queue / GPU slot management ──────────────────────────────────────────────

# Seconds an interactive request will wait for a GPU slot before returning 503.
# Sized to roughly 2× MODEL_SWAP_COST_SECONDS so a single worst-case swap plus
# normal queue wait does not trip the deadline. Genuine overload still sheds.
INTERACTIVE_QUEUE_TIMEOUT = int(_get("INTERACTIVE_QUEUE_TIMEOUT", "40"))

# Max concurrent generation requests per GPU instance. 1 = fully serialised.
GPU_MAX_CONCURRENT = int(_get("GPU_MAX_CONCURRENT", "1"))

# Conservative upper bound (seconds) for loading the largest model this system
# serves. Used by the dispatcher as a static cost for the "swap vs wait"
# decision. Tune only when hardware or the largest model changes.
# Default 20s is calibrated for qwen3:32b (~19 GB Q4_K_M) on a Tesla P40.
MODEL_SWAP_COST_SECONDS = int(_get("MODEL_SWAP_COST_SECONDS", "20"))

# Maximum wall-clock seconds a single dispatched request may hold its GPU
# slot. After this, the watchdog force-fails the entry and frees the slot
# so the queue cannot deadlock on a hung downstream call. Sized for the
# slowest legitimate batch on a P40 (qwen3:32b reasoning at ~3 tok/s with
# the largest num_predict any caller currently sends — 1024 tokens — plus
# headroom). Tune up only if a legitimate caller hits the wall.
SLOT_MAX_WALL_SECONDS = int(_get("SLOT_MAX_WALL_SECONDS", "1800"))

# Watchdog poll interval. Cheap (linear scan over _tracked) so 30s is plenty.
WATCHDOG_INTERVAL_SECONDS = int(_get("WATCHDOG_INTERVAL_SECONDS", "30"))

# How often the streaming proxy emits a keepalive NDJSON chunk while a
# request is waiting for a GPU slot. Each chunk resets the caller's
# between-chunk read-gap timeout, so a client with a modest 60s timeout
# can ride out a queue wait that exceeds 60s without disconnecting. Must
# be strictly less than the tightest caller read-gap timeout on the
# stack — parsival's llm._ollama_local uses 60-90s, so 20s gives a
# comfortable 3x safety margin. Raising this risks client disconnects;
# lowering it just wastes a few bytes of traffic per wait.
QUEUE_HEARTBEAT_INTERVAL_SECONDS = int(
    _get("QUEUE_HEARTBEAT_INTERVAL_SECONDS", "20")
)

# Upper-bound httpx read timeout for proxied Ollama requests. Must be >=
# SLOT_MAX_WALL_SECONDS so the watchdog is the first to fire — the bound
# httpx timeout is defence-in-depth, not the primary limit. Setting this
# to None reproduces the original deadlock; do not.
PROXY_READ_TIMEOUT_SECONDS = int(_get("PROXY_READ_TIMEOUT_SECONDS", "1800"))

# ── API server ────────────────────────────────────────────────────────────────

PORT     = int(_get("PORT",     "11400"))
API_HOST = _get("API_HOST", "0.0.0.0")

# ── SSH proxy ─────────────────────────────────────────────────────────────────

SSH_HOST     = _get("SSH_HOST",     "host.docker.internal")
SSH_PORT     = int(_get("SSH_PORT", "22"))
SSH_USER     = _get("SSH_USER",     "")
SSH_KEY_PATH = _get("SSH_KEY_PATH", "/data/ssh_key")

# ── VNC proxy ─────────────────────────────────────────────────────────────────

VNC_HOST = _get("VNC_HOST", "host.docker.internal")
VNC_PORT = int(_get("VNC_PORT", "5900"))

# ── Fan controller ────────────────────────────────────────────────────────────

# URL of the iDRAC fan controller REST API (api_server.py on port 8080).
FAN_CONTROLLER_URL = _get("FAN_CONTROLLER_URL", "http://host.docker.internal:8080")

# ── Notifications ─────────────────────────────────────────────────────────────

# Webhook URL to POST batch job completion payloads to.  Leave empty to disable.
# Supports Slack incoming webhooks, ntfy.sh topics (e.g. https://ntfy.sh/my-topic),
# Gotify, and any HTTP endpoint that accepts JSON POST.
NOTIFICATION_WEBHOOK_URL = _get("NOTIFICATION_WEBHOOK_URL", "")

# ── Companion service URLs (for My Day panel) ────────────────────────────────

# URL of the Parsival (hexcaliper-squire) nginx, reachable from merLLM container.
PARSIVAL_URL = _get("PARSIVAL_URL", "http://host.docker.internal:8082")

# URL of the LanceLLMot (hexcaliper) nginx, reachable from merLLM container.
LANCELLMOT_URL = _get("LANCELLMOT_URL", "http://host.docker.internal:8080")

# ── Log container name mapping ────────────────────────────────────────────────
# Maps logical service names to Docker container names for the /api/merllm/logs
# endpoint. Override if your compose project name or container names differ.
LOG_CONTAINER_LANCELLMOT_API   = _get("LOG_CONTAINER_LANCELLMOT_API",   "hexcaliper-lancellmot-api-1")
LOG_CONTAINER_LANCELLMOT_NGINX = _get("LOG_CONTAINER_LANCELLMOT_NGINX", "hexcaliper-lancellmot-nginx-1")
LOG_CONTAINER_PARSIVAL_API     = _get("LOG_CONTAINER_PARSIVAL_API",     "hexcaliper-squire-parsival-api-1")
LOG_CONTAINER_PARSIVAL_NGINX   = _get("LOG_CONTAINER_PARSIVAL_NGINX",   "hexcaliper-squire-parsival-nginx-1")
LOG_CONTAINER_MERLLM_API       = _get("LOG_CONTAINER_MERLLM_API",       "merllm-api")


def apply_overrides(d: dict) -> None:
    """Hot-reload settings from the database into module-level variables."""
    mod = sys.modules[__name__]
    str_fields = {
        "ollama_0_url":              "OLLAMA_0_URL",
        "ollama_1_url":              "OLLAMA_1_URL",
        "default_model":             "DEFAULT_MODEL",
        "ssh_user":                  "SSH_USER",
        "ssh_key_path":              "SSH_KEY_PATH",
        "notification_webhook_url":  "NOTIFICATION_WEBHOOK_URL",
        "parsival_url":              "PARSIVAL_URL",
        "lancellmot_url":            "LANCELLMOT_URL",
    }
    int_fields = {
        "reclaim_timeout":         "RECLAIM_TIMEOUT",
        "health_backoff_base":     "HEALTH_BACKOFF_BASE",
        "health_backoff_cap":      "HEALTH_BACKOFF_CAP",
        "health_fault_timeout":    "HEALTH_FAULT_TIMEOUT",
        "metrics_interval_sec":    "METRICS_INTERVAL_SEC",
        "gpu_temp_pause_c":        "GPU_TEMP_PAUSE_C",
        "gpu_temp_resume_c":       "GPU_TEMP_RESUME_C",
    }
    for key, attr in str_fields.items():
        if key in d and d[key] is not None:
            setattr(mod, attr, str(d[key]))
    for key, attr in int_fields.items():
        if key in d and d[key] is not None:
            try:
                setattr(mod, attr, int(d[key]))
            except (ValueError, TypeError):
                pass
