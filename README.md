# merLLM

Centralized LLM traffic control for the Hexcaliper ecosystem.

merLLM sits between LanceLLMot (hexcaliper) and Parsival (hexcaliper-squire) and both GPU-pinned Ollama instances. It routes requests to the right GPU, manages day/night mode transitions, queues batch jobs for overnight extended-context processing, and exposes a browser dashboard with system metrics, logs, SSH terminal, and VNC viewer.

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
| `DB_PATH` | `/data/merllm.db` | SQLite database path (inside container) |
| `METRICS_INTERVAL_SEC` | `10` | How often system metrics are collected |
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

Manual override is available via the dashboard Mode controls or `POST /api/merllm/mode`.

## Batch jobs

Jobs submitted via `POST /api/batch/submit` are held in SQLite until night mode activates, then processed sequentially with `NIGHT_NUM_CTX` context. LanceLLMot and Parsival use this for deep-analysis tasks that benefit from extended context.

```bash
curl -X POST http://localhost:11400/api/batch/submit \
  -H "Content-Type: application/json" \
  -d '{"source_app":"parsival","prompt":"Analyze...","model":"qwen3:32b"}'
# → {"ok":true,"id":"<uuid>"}

curl http://localhost:11400/api/batch/status/<uuid>
curl http://localhost:11400/api/batch/results/<uuid>
```

## Ollama proxy API

merLLM is a drop-in replacement for `OLLAMA_BASE_URL`. All standard endpoints are proxied:

| Endpoint | Notes |
|---|---|
| `POST /api/generate` | Routed by model; interactive requests update activity timer |
| `POST /api/chat` | Same routing as generate |
| `POST /api/embeddings` | Always GPU 0 |
| `GET /api/tags` | Model list from GPU 0 (shared model store) |
| `POST /api/show` | GPU 0 |
| `POST /api/pull` | GPU 0 (shared store) |
| `GET /api/ps` | Aggregates loaded models from both instances |

Set `X-Priority: batch` on a request to place it in the low-priority queue without submitting a batch job.

## GeoIP setup

Download the free MaxMind GeoLite2-City database:

1. Register at https://www.maxmind.com/en/geolite2/signup
2. Download `GeoLite2-City.mmdb`
3. Place it at `./data/GeoLite2-City.mmdb` (mounted as `/data/` in the container)

Without the database, merLLM falls back to the `BASE_DAY_END_LOCAL` schedule with no timezone adjustment.

## Ollama service management

To allow merLLM to start and stop Ollama services on mode transitions, set `OLLAMA_MANAGE_VIA=systemctl` and ensure the container has access to the host's systemd socket. Without this, transitions are logged but services are not actually started/stopped — useful if you manage them separately.

The `ollama-night` service should be a systemd unit that starts Ollama with `CUDA_VISIBLE_DEVICES=0,1` for dual-GPU operation.

## Dashboard tabs

| Tab | Contents |
|---|---|
| Overview | Mode status, Ollama health, GPU stats, batch counts, recent transitions |
| Batch Jobs | Submit jobs, view queue, cancel/requeue |
| Metrics | CPU, RAM, disk, GPU utilization and VRAM, network throughput |
| Logs | Live log tail from any systemd service |
| SSH | Browser SSH terminal (xterm.js) |
| VNC | Browser VNC viewer (noVNC) |
| Settings | Live configuration editor |
| Help | Quick reference for all features |

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
  queue_manager.py  — asyncio priority queue and batch runner
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
docker-compose.yml
```
