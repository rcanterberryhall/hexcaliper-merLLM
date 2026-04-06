# merLLM

Centralized LLM traffic control for the Hexcaliper ecosystem.

merLLM sits between LanceLLMot (hexcaliper) and Parsival (hexcaliper-squire) and both GPU-pinned Ollama instances. It routes requests to the right GPU via a priority queue, manages day/night mode transitions, queues batch jobs for overnight extended-context processing with automatic retry, sends completion notifications, and exposes a browser dashboard with a unified "My Day" attention panel, system metrics charts, logs, SSH terminal, and VNC viewer.

```
LanceLLMot  ──┐
              ├──► merLLM :11400 ──► ollama-gpu0 :11434  (qwen3:32b,   GPU 0)
Parsival    ──┘                 └──► ollama-gpu1 :11435  (qwen3:30b-a3b, GPU 1)
```

## Requirements

- Docker + Docker Compose
- Two GPU-pinned Ollama systemd services running on the host (`ollama-gpu0` on :11434, `ollama-gpu1` on :11435)
- *(Optional)* MaxMind GeoLite2-City.mmdb for timezone-aware scheduling
- *(Optional)* SSH key at `./data/ssh_key` for the browser SSH terminal

## Quick start

```bash
git clone https://github.com/rcanterberryhall/hexcaliper-merllm
cd hexcaliper-merllm
docker compose up -d
```

Dashboard: http://localhost:11400/web/

## Configuration

All settings are environment variables in `docker-compose.yml`. They can also be changed live via the Settings tab in the dashboard (persisted in SQLite, survive restarts).

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_0_URL` | `http://host.docker.internal:11434` | GPU 0 Ollama instance |
| `OLLAMA_1_URL` | `http://host.docker.internal:11435` | GPU 1 Ollama instance |
| `DAY_MODEL_GPU0` | `qwen3:32b` | Model routed to GPU 0 in day mode |
| `DAY_MODEL_GPU1` | `qwen3:30b-a3b` | Model routed to GPU 1 in day mode |
| `NIGHT_MODEL` | `qwen3:32b` | Model used for batch jobs (night mode) |
| `NIGHT_NUM_CTX` | `32768` | Context window for night-mode batch jobs |
| `INACTIVITY_TIMEOUT_MIN` | `90` | Minutes with no requests before night mode activates |
| `BASE_DAY_END_LOCAL` | `22:00` | Local time after which night mode may activate |
| `GEOIP_DB_PATH` | `/data/GeoLite2-City.mmdb` | Path to MaxMind DB (inside container) |
| `GEOIP_OFFSET_OVERRIDE` | *(empty)* | Force a UTC offset (e.g. `-5`) instead of GeoIP lookup |
| `OLLAMA_MANAGE_VIA` | `none` | `systemctl` to let merLLM start/stop Ollama services |
| `GPU0_SERVICE` | `ollama-gpu0` | systemd unit name for GPU 0 Ollama |
| `GPU1_SERVICE` | `ollama-gpu1` | systemd unit name for GPU 1 Ollama |
| `NIGHT_SERVICE` | `ollama-night` | systemd unit name for dual-GPU night instance |
| `DRAIN_TIMEOUT_SEC` | `300` | Max seconds to wait for in-flight requests before transitioning |
| `BATCH_MAX_RETRIES` | `2` | Max automatic retries for a failed batch job (2 = 3 total attempts) |
| `BATCH_MAX_PROMPT_LEN` | `100000` | Max prompt length in characters accepted by `POST /api/batch/submit` (422 if exceeded) |
| `INTERACTIVE_QUEUE_TIMEOUT` | `30` | Seconds an interactive request waits for a GPU slot before returning 503 |
| `GPU_MAX_CONCURRENT` | `1` | Max concurrent generation requests per GPU (1 = fully serialised) |
| `DB_PATH` | `/data/merllm.db` | SQLite database path (inside container) |
| `METRICS_INTERVAL_SEC` | `10` | How often system metrics are collected |
| `NOTIFICATION_WEBHOOK_URL` | *(empty)* | Webhook URL for batch job completion notifications (Slack, ntfy.sh, Gotify, or any JSON POST endpoint) |
| `PARSIVAL_URL` | `http://host.docker.internal:8082` | Parsival nginx URL (for My Day panel) |
| `LANCELLMOT_URL` | `http://host.docker.internal:8080` | LanceLLMot nginx URL (for My Day panel) |
| `LOG_CONTAINER_*` | *(see below)* | Docker container name mapping for the logs endpoint |
| `SSH_USER` | *(empty)* | Username for SSH terminal (leave blank to disable) |
| `SSH_KEY_PATH` | `/data/ssh_key` | Path to SSH private key (inside container) |
| `VNC_HOST` | `host.docker.internal` | VNC server host |
| `VNC_PORT` | `5900` | VNC server port |

## Day / Night mode

**Day mode** — two Ollama instances run simultaneously, each pinned to one GPU. Requests are routed by model name: `DAY_MODEL_GPU1` goes to GPU 1, everything else goes to GPU 0.

**Night mode** — activates automatically when both conditions are met:
1. No interactive requests for `INACTIVITY_TIMEOUT_MIN` minutes
2. Current time is past `BASE_DAY_END_LOCAL` (adjusted for the client's timezone via GeoIP)

In night mode, merLLM switches to the `ollama-night` service (both GPUs combined, larger context) and drains the batch job queue. Any new interactive request triggers an immediate return to day mode.

Mode transitions fail honestly — if a systemctl call fails, the mode is not changed and the failure is recorded in the transition history with an explicit error. The `POST /api/merllm/mode` endpoint returns the failure status so the UI can display it. When `OLLAMA_MANAGE_VIA=none`, transitions are skipped entirely.

Manual override is available via the dashboard Mode controls or `POST /api/merllm/mode`.

## Batch jobs

Jobs submitted via `POST /api/batch/submit` are held in SQLite until night mode activates, then processed sequentially with `NIGHT_NUM_CTX` context. LanceLLMot and Parsival use this for deep-analysis tasks that benefit from extended context.

Prompts exceeding `BATCH_MAX_PROMPT_LEN` characters are rejected with HTTP 422.

**Automatic retry** — failed jobs are retried up to `BATCH_MAX_RETRIES` times (default 2, giving 3 total attempts) with exponential backoff (30s, 120s). On final failure, the accumulated error messages are stored on the job record.

**Completion notifications** — when a batch job completes (success or final failure), merLLM can notify you via:
- **Webhook** — set `NOTIFICATION_WEBHOOK_URL` to a Slack incoming webhook, ntfy.sh topic, Gotify URL, or any endpoint accepting JSON POST. Payload includes `job_id`, `source_app`, `status`, timestamps, and a prompt preview.
- **SSE** — the dashboard's activity stream receives a completion event in real time.

```bash
curl -X POST http://localhost:11400/api/batch/submit \
  -H "Content-Type: application/json" \
  -d '{"source_app":"parsival","prompt":"Analyze...","model":"qwen3:32b"}'
# → {"ok":true,"id":"<uuid>"}

curl http://localhost:11400/api/batch/status/<uuid>
curl http://localhost:11400/api/batch/results/<uuid>
```

## Ollama proxy API

merLLM is a drop-in replacement for `OLLAMA_BASE_URL`. All standard endpoints are proxied.

**Priority queue** — all `/api/generate` and `/api/chat` requests pass through a GPU slot queue. Interactive requests proceed immediately if a GPU is available; when both GPUs are busy, the request waits up to `INTERACTIVE_QUEUE_TIMEOUT` seconds (default 30) before returning 503. Batch-priority requests yield to queued interactive requests. The queue tracks in-flight count per GPU for informed routing decisions.

**Transparent wait** — when a request is queued, a `queue_status` NDJSON line is emitted before generation starts:
```json
{"type": "queue_status", "position": 2, "reason": "transitioning to day mode", "estimated_wait_seconds": 35}
```
LanceLLMot and Parsival recognise this event and display a waiting indicator with the reason.

| Endpoint | Notes |
|---|---|
| `POST /api/generate` | Routed by model via priority queue; interactive requests update activity timer |
| `POST /api/chat` | Same routing as generate |
| `POST /api/embeddings` | Always GPU 0 |
| `GET /api/tags` | Model list from GPU 0 (shared model store) |
| `POST /api/show` | GPU 0 |
| `POST /api/pull` | GPU 0 (shared store) |
| `GET /api/ps` | Aggregates loaded models from both instances |

Set `X-Priority: batch` on a request to place it in the low-priority queue without submitting a batch job.

## Activity SSE stream

The overview dashboard subscribes to `GET /api/merllm/activity/stream` (Server-Sent Events) for real-time per-GPU activity. Each event carries a JSON snapshot:

```json
{
  "gpu0": {"model": "qwen3:32b", "endpoint": "/api/chat", "chunks": 47, "text": "...recent tokens..."},
  "gpu1": null
}
```

`null` means the GPU is idle. Events are pushed immediately on request start/end and rate-limited to 10/sec during streaming. A keepalive comment (`: keepalive`) is sent every 5 s when idle to keep the connection alive through proxies.

LanceLLMot and Parsival dashboards consume this stream to show live token output without polling.

## GeoIP setup

Download the free MaxMind GeoLite2-City database:

1. Register at https://www.maxmind.com/en/geolite2/signup
2. Download `GeoLite2-City.mmdb`
3. Place it at `./data/GeoLite2-City.mmdb` (mounted as `/data/` in the container)

Without the database, merLLM falls back to the `BASE_DAY_END_LOCAL` schedule with no timezone adjustment.

## Log container name mapping

The `/api/merllm/logs/{service}` endpoint maps logical service names to Docker container names. Override these if your compose project names or container names differ:

| Variable | Default |
|---|---|
| `LOG_CONTAINER_LANCELLMOT_API` | `hexcaliper-lancellmot-api-1` |
| `LOG_CONTAINER_LANCELLMOT_NGINX` | `hexcaliper-lancellmot-nginx-1` |
| `LOG_CONTAINER_PARSIVAL_API` | `hexcaliper-squire-parsival-api-1` |
| `LOG_CONTAINER_PARSIVAL_NGINX` | `hexcaliper-squire-parsival-nginx-1` |
| `LOG_CONTAINER_MERLLM_API` | `merllm-api` |

## Ollama service management

To allow merLLM to start and stop Ollama services on mode transitions, set `OLLAMA_MANAGE_VIA=systemctl` and ensure the container has access to the host's systemd socket. Without this, transitions are logged but services are not actually started/stopped — useful if you manage them separately.

The `ollama-night` service should be a systemd unit that starts Ollama with `CUDA_VISIBLE_DEVICES=0,1` for dual-GPU operation.

## My Day panel

The dashboard landing page includes a "My Day" panel that aggregates attention items from across the ecosystem:

- **From Parsival** — count of new/investigating situations, overdue follow-ups, unread high-attention items (via `GET {PARSIVAL_URL}/page/api/attention/summary`)
- **From LanceLLMot** — conversations with pending deep analysis results, escalation results awaiting review (via `GET {LANCELLMOT_URL}/api/status/pending`)
- **From merLLM** — completed batch jobs not yet retrieved, active warnings (GPU temp, Ollama down, transition failures)

Each card links directly to the relevant item in the relevant app (opens in a new tab). If a companion app is unreachable, the panel shows "offline" instead of failing entirely.

Configure companion URLs via `PARSIVAL_URL` and `LANCELLMOT_URL` env vars.

## Docker health checks

Both services include Docker `healthcheck` directives:
- `merllm-api`: `curl -f http://localhost:11400/api/merllm/status`
- `merllm-nginx`: `curl -f http://localhost:11400/web/`

Interval: 30s, timeout: 5s, retries: 3.

## Dashboard tabs

| Tab | Contents |
|---|---|
| Overview | My Day panel, mode status, Ollama health, per-GPU live activity (model, token stream, chunk count via SSE), batch counts, recent transitions |
| Batch Jobs | Submit jobs, view queue, cancel/requeue; failed jobs show retry count |
| Metrics | Time-series charts (Chart.js) for CPU, RAM, disk, GPU utilization, GPU VRAM, GPU temperature, network throughput; range selector: 1h / 6h / 24h / 7d |
| Logs | Live log tail from any Docker container (configurable via `LOG_CONTAINER_*` env vars) |
| SSH | Browser SSH terminal (xterm.js) |
| VNC | Browser VNC viewer (noVNC) |
| Settings | Live configuration editor (includes notification webhook, companion URLs) |
| Help | Quick reference for all features |

## Database backup

`POST /api/merllm/backup` triggers an online SQLite backup using Python's
`sqlite3.backup()` API (safe while the database is in use).  
Backup files are written to `BACKUP_DIR` (default `/data/backups`) as
`merllm-YYYYMMDD-HHMMSS.db` and rotated to keep the newest `BACKUP_KEEP_DAYS`
copies (default 7).

| Variable | Default | Purpose |
|---|---|---|
| `BACKUP_DIR` | `/data/backups` | Directory for backup files |
| `BACKUP_KEEP_DAYS` | `7` | Number of backups to retain |

**Scheduled backup via cron (run on the host):**

```bash
# Back up every night at 02:00
0 2 * * * curl -s -X POST http://localhost:11400/api/merllm/backup \
  >> /var/log/merllm-backup.log 2>&1
```

Or with the helper script (if present):

```bash
0 2 * * * /opt/hexcaliper/merllm/backup_db.sh >> /var/log/merllm-backup.log 2>&1
```

## Request logging

All HTTP requests are logged via middleware: timestamp, method, path, status code, duration (ms), and user email (from `CF-Access-Authenticated-User-Email` header, or `anonymous`). Log levels: INFO for 2xx/3xx, WARNING for 4xx, ERROR for 5xx.

## Development

```bash
cd api
pip install -r requirements.txt
DB_PATH=/tmp/dev.db uvicorn app:app --reload --port 11400
```

```bash
pytest tests/ -v
```

## Repository layout

```
api/
  app.py            — FastAPI application, all HTTP + WebSocket endpoints
  config.py         — Environment variable loading and hot-reload
  db.py             — SQLite WAL: batch jobs, metrics, settings, transitions
  geoip.py          — MaxMind GeoLite2 timezone lookup
  metrics.py        — psutil + pynvml collection loop
  mode_manager.py   — Day/Night state machine and scheduler
  queue_manager.py  — asyncio priority queue, batch runner with retry
  notifications.py  — Batch job completion dispatch (webhook, SSE)
  Dockerfile
  requirements.txt
nginx/
  default.conf      — Reverse proxy, WebSocket upgrade, static file serving
web/
  index.html        — Dashboard shell
  app.js            — Dashboard logic
  styles.css        — Dark theme
tests/
  test_config.py
  test_db.py
  test_queue_manager.py
  test_activity_sse.py   — SSE push rate limiting, queue fan-out, activity helpers
docker-compose.yml
```
