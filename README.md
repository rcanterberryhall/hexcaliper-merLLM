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
| `INTERACTIVE_QUEUE_TIMEOUT` | `30` | Seconds a `chat`-bucket request waits for a GPU slot before returning 503 (other buckets wait indefinitely) |
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

## GPU routing

Both Ollama instances run continuously. Requests are round-robined across healthy GPUs. When a request specifies a non-default model, merLLM waits for an idle GPU, swaps it to the requested model, and routes subsequent matching requests there. After `RECLAIM_TIMEOUT` seconds of idle with a non-default model, the GPU is reclaimed back to `DEFAULT_MODEL`.

**Health tracking** — if a GPU fails, it is marked *degraded* and excluded from routing. Health probes use exponential backoff. After `HEALTH_FAULT_TIMEOUT` seconds of continuous failure, the GPU is declared *faulted* and requires manual reset.

## Priority buckets

Every request that lands on merLLM is placed in one of five priority buckets. The dispatcher drains them **strictly top-down** — bucket *N* must be empty before bucket *N+1* gets a GPU slot. Within a bucket the order is FIFO. There is no preemption, no fairness, and no aging: a steady stream of chat would starve background forever (and that's the point — chat is cheap and background is resumable).

| # | Bucket       | `X-Priority` value | Intended for |
|---|--------------|--------------------|--------------|
| 1 | **chat**       | `chat` (alias: `interactive`) | Real-time human chat. Anything a user is actively waiting on. |
| 2 | **reserved**   | `reserved`         | Intentionally unused. Room to grow — don't route traffic here. |
| 3 | **short**      | `short`            | Short, bounded Parsival work: contacts parsing, extraction, situation summaries. |
| 4 | **feedback**   | `feedback`         | LLM work *spawned by* background work — e.g. a reanalyze job that needs a follow-up summary. Higher than bucket 5 so long-running batch work can unblock itself without jumping ahead of user-facing work. |
| 5 | **background** | `background` (alias: `batch`) | Batch analysis, bulk reanalyze, LanceLLMot document ingestion, overnight work. |

**Call-site assignment rule** — each call site picks its bucket once and declares it on every request via the `X-Priority` header. Callers do not get to promote themselves at runtime. If a request has no `X-Priority` header, merLLM currently defaults it to `chat` for back-compat during the rollout; unknown string values fall back to `background` so typos cannot escalate.

The dispatcher state is visible on the dashboard's Queue tab (one lane per bucket, with live depth) and machine-readably at `GET /api/merllm/queue`, which includes a `buckets` map keyed by bucket name.

## Batch jobs

Jobs submitted via `POST /api/batch/submit` are persisted in SQLite and run in the **background** bucket (bucket 5) whenever every higher bucket is empty. They survive restarts; directly-submitted `X-Priority: background` requests do not.

Prompts exceeding `BATCH_MAX_PROMPT_LEN` characters are rejected with HTTP 422.

**Automatic retry** — failed jobs are retried up to `BATCH_MAX_RETRIES` times (default 2, giving 3 total attempts) with exponential backoff (30s, 120s). On final failure, the accumulated error messages are stored on the job record.

**Options forwarding** — the `options` dict in the submission body is forwarded verbatim to Ollama. Callers submitting reasoning-model work (qwen3:*) **must** populate `options` with `think: false` and a bounded `num_predict`, otherwise the model will reason unbounded and wedge the queue. Example:
```json
{
  "source_app": "parsival",
  "prompt": "...",
  "model": "qwen3:32b",
  "options": {"think": false, "num_predict": 768, "num_ctx": 8192, "temperature": 0.1}
}
```

**Startup recovery** — when merllm-api restarts, any batch jobs left in `running` status are requeued, and all `queued` rows are re-launched via `asyncio.ensure_future(_run_batch_job_async(...))`. Prior to this fix, restarts would leave SQLite rows in `queued` status with no in-process task driving them.

**Slot watchdog** — a background `_watchdog_loop` task scans `_tracked` every `WATCHDOG_INTERVAL_SECONDS` (default 30s) and force-fails any slot whose `started_at` exceeds `SLOT_MAX_WALL_SECONDS` (default 1800s). This is the last line of defence against hung upstream Ollama calls deadlocking the strict-priority dispatcher — a reclaim log line (`watchdog: reclaiming wedged slot ...`) indicates a bug in either the caller (missing `num_predict`) or the proxy layer, **not** a knob to tune. Bounded `PROXY_READ_TIMEOUT_SECONDS` (default 1800s) on every httpx call to Ollama is the defence-in-depth layer behind it.

| Variable | Default | Description |
|---|---|---|
| `SLOT_MAX_WALL_SECONDS` | `1800` | Per-slot wall-clock budget before the watchdog reclaims it |
| `WATCHDOG_INTERVAL_SECONDS` | `30` | How often the watchdog scans for wedged slots |
| `PROXY_READ_TIMEOUT_SECONDS` | `1800` | httpx read timeout on every upstream Ollama call |
| `QUEUE_HEARTBEAT_INTERVAL_SECONDS` | `20` | Interval between queue-wait keepalive NDJSON chunks (see [Transparent wait](#ollama-proxy-api)) |

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

**Unified GPU queue** — every request is tracked in a single queue visible via `GET /api/merllm/queue`. The dashboard shows all active and waiting requests with source app, model, target GPU, priority bucket, status, and elapsed time. Client apps identify themselves with the `X-Source` header (e.g. `lancellmot`, `parsival`) and pick a bucket with `X-Priority` (see [Priority buckets](#priority-buckets)).

**Transparent wait** — when a request is queued, a `queue_status` NDJSON line is emitted before generation starts:
```json
{"type": "queue_status", "reason": "GPU slot occupied by another request", "estimated_wait_seconds": 30}
```
LanceLLMot and Parsival recognise this event and display a waiting indicator with the reason.

If the wait exceeds `QUEUE_HEARTBEAT_INTERVAL_SECONDS` (default `20`), the streaming proxy emits periodic keepalive chunks:
```json
{"type": "queue_status", "waiting": true, "elapsed_seconds": 20}
```
Each chunk resets the caller's between-chunk read-gap timer so a client using `requests.post(timeout=60, stream=True)` (parsival's `_ollama_local`) can ride out a multi-minute background-drain wait without disconnecting. Keep this interval strictly less than the tightest caller timeout on the stack — parsival uses 60–90 s, so the default 20 s gives a 3× safety margin. Ollama's stream consumer ignores unknown JSON fields, so these keepalive lines are transparent to downstream parsers.

| Endpoint | Notes |
|---|---|
| `POST /api/generate` | Round-robin routed by model; tracked in unified queue |
| `POST /api/chat` | Same routing as generate |
| `POST /api/embeddings` | Round-robin routed; tracked in unified queue |
| `GET /api/tags` | Model list from GPU 0 (shared model store) |
| `POST /api/show` | GPU 0 |
| `POST /api/pull` | GPU 0 (shared store) |
| `GET /api/ps` | Aggregates loaded models from both instances |

Set `X-Priority: <bucket>` on any proxied request to select a priority bucket (`chat`, `reserved`, `short`, `feedback`, `background`; the legacy values `interactive`/`batch` still work). See [Priority buckets](#priority-buckets) for the full semantics.

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
