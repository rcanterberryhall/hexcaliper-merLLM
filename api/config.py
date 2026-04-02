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

DAY_MODEL_GPU0 = _get("DAY_MODEL_GPU0", "qwen3:32b")
DAY_MODEL_GPU1 = _get("DAY_MODEL_GPU1", "qwen3:30b-a3b")
NIGHT_MODEL    = _get("NIGHT_MODEL",    "qwen3:32b")
NIGHT_NUM_CTX  = int(_get("NIGHT_NUM_CTX", "32768"))

# ── Mode scheduling ───────────────────────────────────────────────────────────

# Minutes of no interactive requests before switching to night mode.
INACTIVITY_TIMEOUT_MIN = int(_get("INACTIVITY_TIMEOUT_MIN", "90"))

# Base schedule hint: local time after which night mode may activate (HH:MM).
BASE_DAY_END_LOCAL = _get("BASE_DAY_END_LOCAL", "22:00")

# ── GeoIP ─────────────────────────────────────────────────────────────────────

GEOIP_DB_PATH      = _get("GEOIP_DB_PATH", "/data/GeoLite2-City.mmdb")
GEOIP_OFFSET_OVERRIDE = _get("GEOIP_OFFSET_OVERRIDE", "")   # e.g. "-5" for EST

# ── Ollama lifecycle management ───────────────────────────────────────────────

# "systemctl" — use systemctl on the host (requires appropriate permissions).
# "none"      — proxy only; do not manage Ollama services.
OLLAMA_MANAGE_VIA = _get("OLLAMA_MANAGE_VIA", "none")

GPU0_SERVICE  = _get("GPU0_SERVICE",  "ollama-gpu0")
GPU1_SERVICE  = _get("GPU1_SERVICE",  "ollama-gpu1")
NIGHT_SERVICE = _get("NIGHT_SERVICE", "ollama-night")

# Max seconds to wait for in-flight requests to drain before transitioning.
DRAIN_TIMEOUT_SEC = int(_get("DRAIN_TIMEOUT_SEC", "300"))

# ── Storage ───────────────────────────────────────────────────────────────────

DB_PATH             = _get("DB_PATH",             "/data/merllm.db")
METRICS_RETAIN_DAYS = int(_get("METRICS_RETAIN_DAYS", "7"))
METRICS_INTERVAL_SEC = int(_get("METRICS_INTERVAL_SEC", "10"))

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


def apply_overrides(d: dict) -> None:
    """Hot-reload settings from the database into module-level variables."""
    mod = sys.modules[__name__]
    str_fields = {
        "ollama_0_url":            "OLLAMA_0_URL",
        "ollama_1_url":            "OLLAMA_1_URL",
        "day_model_gpu0":          "DAY_MODEL_GPU0",
        "day_model_gpu1":          "DAY_MODEL_GPU1",
        "night_model":             "NIGHT_MODEL",
        "base_day_end_local":      "BASE_DAY_END_LOCAL",
        "geoip_offset_override":   "GEOIP_OFFSET_OVERRIDE",
        "ollama_manage_via":       "OLLAMA_MANAGE_VIA",
        "ssh_user":                "SSH_USER",
        "ssh_key_path":            "SSH_KEY_PATH",
    }
    int_fields = {
        "night_num_ctx":           "NIGHT_NUM_CTX",
        "inactivity_timeout_min":  "INACTIVITY_TIMEOUT_MIN",
        "drain_timeout_sec":       "DRAIN_TIMEOUT_SEC",
        "metrics_interval_sec":    "METRICS_INTERVAL_SEC",
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
