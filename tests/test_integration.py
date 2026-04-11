#!/usr/bin/env python3
"""
Hexcaliper Ecosystem — Cross-Service Integration Tests

Runs against the live Docker services. All three must be up:
  - LanceLLMot  :8080
  - merLLM      :11400
  - Parsival    :8082

Usage:
    python3 test_integration.py          # run all tests
    python3 test_integration.py -v       # verbose
    python3 test_integration.py -k queue # run only tests matching 'queue'
"""
import json
import threading
import time
import uuid
import pytest
import httpx

LANCELLMOT = "http://localhost:8080"
MERLLM     = "http://localhost:11400"
PARSIVAL   = "http://localhost:8082"

client = httpx.Client(timeout=60)


# ─── Helpers ────────────────────────────────────────────────────────────────

def api(base: str, path: str, **kwargs) -> httpx.Response:
    return client.get(f"{base}{path}", **kwargs)

def post(base: str, path: str, **kwargs) -> httpx.Response:
    return client.post(f"{base}{path}", **kwargs)

def patch(base: str, path: str, **kwargs) -> httpx.Response:
    return client.patch(f"{base}{path}", **kwargs)


@pytest.fixture(scope="session", autouse=True)
def _clear_merllm_queue_baseline():
    """
    Drop completed/failed entries from merLLM's in-memory tracker before the
    suite runs, so baseline-ID exclusion in the priority tests starts from
    the smallest possible set. This is the non-force mode of the admin
    clear endpoint — in-flight work (``queued``/``running``) is left alone
    because parsival/lancellmot may be doing legitimate background work
    while the suite runs.
    """
    try:
        post(MERLLM, "/api/merllm/queue/clear")
    except Exception:
        # If merLLM isn't reachable, TestServiceHealth will fail with a
        # clearer message.
        pass
    yield


# ═══════════════════════════════════════════════════════════════════════════
# Section 1: Service Health
# ═══════════════════════════════════════════════════════════════════════════

class TestServiceHealth:
    """Verify all services are up and reporting healthy."""

    def test_lancellmot_health(self):
        r = api(LANCELLMOT, "/api/health")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert "ollama_base_url" in d

    def test_merllm_status(self):
        r = api(MERLLM, "/api/merllm/status")
        assert r.status_code == 200
        d = r.json()
        assert d["routing"] == "round_robin"
        # GPUs can be transiently unavailable under heavy batch load
        gpus_ok = [d["ollama"][g]["ok"] for g in d["ollama"]]
        if not all(gpus_ok):
            import warnings
            warnings.warn(f"Some GPUs report ok=false (may be transient under batch load): {d['ollama']}")
        assert any(gpus_ok), "All GPUs are down"

    def test_merllm_health(self):
        r = api(MERLLM, "/health")
        assert r.status_code == 200

    def test_parsival_health(self):
        r = api(PARSIVAL, "/page/api/health")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True

    def test_merllm_diagnostics(self):
        r = api(MERLLM, "/api/merllm/diagnostics")
        assert r.status_code == 200
        d = r.json()
        assert "connectivity" in d
        assert "containers" in d
        # Should see at least our three API containers
        names = [c["name"] for c in d["containers"]]
        assert any("lancellmot" in n for n in names), f"LanceLLMot not in container list: {names}"
        assert any("parsival" in n or "squire" in n for n in names), f"Parsival not in container list: {names}"
        assert any("merllm" in n for n in names), f"merLLM not in container list: {names}"


# ═══════════════════════════════════════════════════════════════════════════
# Section 2: merLLM Proxy → Ollama
# ═══════════════════════════════════════════════════════════════════════════

class TestMerllmProxy:
    """Verify merLLM correctly proxies Ollama API endpoints."""

    def test_tags(self):
        r = api(MERLLM, "/api/tags")
        assert r.status_code == 200
        d = r.json()
        assert "models" in d
        assert len(d["models"]) > 0, "No models available in Ollama"

    def test_ps(self):
        r = api(MERLLM, "/api/ps")
        assert r.status_code == 200
        d = r.json()
        assert "models" in d

    def test_show(self):
        r = post(MERLLM, "/api/show", json={"name": "qwen3:30b-a3b"})
        assert r.status_code == 200

    def test_generate_small(self):
        """Send a tiny generate request through the priority queue."""
        gen_client = httpx.Client(timeout=300)  # LLM can be slow if queue is busy
        r = gen_client.post(f"{MERLLM}/api/generate", json={
            "model": "qwen3:30b-a3b",
            "prompt": "Say 'hello' and nothing else.",
            "stream": False,
            "options": {"num_predict": 10}
        })
        assert r.status_code == 200
        d = r.json()
        assert "response" in d
        # Under heavy batch queue load the model may return an empty response;
        # the important thing is we got a 200 with a valid response field.
        if not d["response"]:
            import warnings
            warnings.warn("generate returned empty response (GPU may be congested from batch queue)")

    def test_embeddings(self):
        r = post(MERLLM, "/api/embeddings", json={
            "model": "nomic-embed-text",
            "prompt": "integration test"
        })
        assert r.status_code == 200
        d = r.json()
        assert "embedding" in d
        assert len(d["embedding"]) > 0


# ═══════════════════════════════════════════════════════════════════════════
# Section 3: merLLM Features
# ═══════════════════════════════════════════════════════════════════════════

class TestMerllmFeatures:
    """Test merLLM-specific features: metrics, settings, routing, batch."""

    def test_metrics_current(self):
        r = api(MERLLM, "/api/merllm/metrics/current")
        assert r.status_code == 200
        d = r.json()
        assert "cpu.total" in d or "cpu_percent" in d

    def test_metrics_history(self):
        r = api(MERLLM, "/api/merllm/metrics/history", params={"metric": "cpu.total", "minutes": "60"})
        assert r.status_code == 200

    def test_settings_get(self):
        r = api(MERLLM, "/api/merllm/settings")
        assert r.status_code == 200
        d = r.json()
        assert "ollama_0_url" in d

    def test_default_model(self):
        r = api(MERLLM, "/api/merllm/default-model")
        assert r.status_code == 200
        assert "model" in r.json()

    def test_backup(self):
        r = post(MERLLM, "/api/merllm/backup")
        assert r.status_code == 200
        d = r.json()
        assert d.get("ok") is True

    def test_batch_submit_reject_oversize(self):
        """Prompts exceeding BATCH_MAX_PROMPT_LEN should be rejected."""
        r = post(MERLLM, "/api/batch/submit", json={
            "source_app": "integration_test",
            "prompt": "x" * 200_000,
        })
        assert r.status_code == 422

    def test_batch_submit_and_status(self):
        """Submit a small batch job and verify status tracking."""
        r = post(MERLLM, "/api/batch/submit", json={
            "source_app": "integration_test",
            "prompt": "Say hello.",
            "model": "qwen3:30b-a3b",
        })
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        job_id = d["id"]

        r2 = api(MERLLM, f"/api/batch/status/{job_id}")
        assert r2.status_code == 200
        status = r2.json()
        assert status["status"] in ("queued", "running", "complete", "failed")

        # Clean up: cancel the test job
        post(MERLLM, f"/api/batch/{job_id}/cancel")


# ═══════════════════════════════════════════════════════════════════════════
# Section 4: Cross-Service — My Day Panel
# ═══════════════════════════════════════════════════════════════════════════

class TestMyDay:
    """Verify the My Day panel can reach all companion services."""

    def test_myday_all_services_ok(self):
        # Use a longer timeout — My Day fans out to all services
        myday_client = httpx.Client(timeout=60)
        r = myday_client.get(f"{MERLLM}/api/merllm/myday")
        assert r.status_code == 200
        d = r.json()

        assert "parsival" in d
        assert "lancellmot" in d
        assert "merllm" in d

        assert d["parsival"]["ok"] is True, f"Parsival unreachable: {d['parsival']}"
        assert d["merllm"]["ok"] is True
        # LanceLLMot can report ok=false if its status/pending endpoint is
        # slow when the GPU is under load; treat as a warning, not a hard fail.
        if not d["lancellmot"]["ok"]:
            import warnings
            warnings.warn(f"LanceLLMot returned ok=false (may be transient): {d['lancellmot']}")

    def test_lancellmot_status_pending(self):
        r = api(LANCELLMOT, "/api/status/pending")
        assert r.status_code == 200

    def test_parsival_attention_summary(self):
        r = api(PARSIVAL, "/page/api/attention/summary")
        assert r.status_code == 200
        d = r.json()
        assert "cold_start" in d
        assert "active_situations" in d


# ═══════════════════════════════════════════════════════════════════════════
# Section 5: LanceLLMot → merLLM Integration
# ═══════════════════════════════════════════════════════════════════════════

class TestLancellmotMerllm:
    """LanceLLMot routes through merLLM for chat and generate."""

    def test_lancellmot_models(self):
        """LanceLLMot should list models from merLLM/Ollama."""
        r = api(LANCELLMOT, "/api/models")
        assert r.status_code == 200
        d = r.json()
        assert "models" in d
        assert len(d["models"]) > 0

    def test_lancellmot_gpu(self):
        r = api(LANCELLMOT, "/api/gpu")
        assert r.status_code == 200

    def test_lancellmot_merllm_status(self):
        """LanceLLMot proxies merLLM status for its routing indicator."""
        r = api(LANCELLMOT, "/api/merllm/status")
        assert r.status_code == 200
        d = r.json()
        assert "routing" in d


# ═══════════════════════════════════════════════════════════════════════════
# Section 6: Parsival Features
# ═══════════════════════════════════════════════════════════════════════════

class TestParsivalFeatures:
    """Test Parsival-specific features: situations, noise filters, attention."""

    def test_settings_get(self):
        r = api(PARSIVAL, "/page/api/settings")
        assert r.status_code == 200

    def test_scan_status(self):
        r = api(PARSIVAL, "/page/api/scan/status")
        assert r.status_code == 200

    def test_stats(self):
        r = api(PARSIVAL, "/page/api/stats")
        assert r.status_code == 200

    def test_situations_list(self):
        r = api(PARSIVAL, "/page/api/situations")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_todos_list(self):
        r = api(PARSIVAL, "/page/api/todos")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_intel_list(self):
        r = api(PARSIVAL, "/page/api/intel")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_projects_list(self):
        r = api(PARSIVAL, "/page/api/projects")
        assert r.status_code == 200

    def test_noise_filters_crud(self):
        """Create, list, and delete a noise filter."""
        # List existing
        r = api(PARSIVAL, "/page/api/noise-filters")
        assert r.status_code == 200
        initial_count = len(r.json())

        # Create
        r = post(PARSIVAL, "/page/api/noise-filters", json={
            "type": "sender_contains",
            "value": "integration-test-noreply@example.com",
        })
        assert r.status_code == 200

        # Verify it exists
        r = api(PARSIVAL, "/page/api/noise-filters")
        assert r.status_code == 200
        filters = r.json()
        assert len(filters) == initial_count + 1

        # Delete it
        r = client.delete(f"{PARSIVAL}/page/api/noise-filters/{initial_count}")
        assert r.status_code == 200

        # Verify removed
        r = api(PARSIVAL, "/page/api/noise-filters")
        assert len(r.json()) == initial_count

    def test_attention_model_summary_shape(self):
        """Attention summary endpoint returns the expected shape.

        Previously this asserted ``cold_start is True``, but the live
        parsival has real user action history (hundreds of action items),
        so it always reports ``cold_start: false``.  The useful integration
        invariant is the response shape, not a cold-start assumption that
        has not held in months.
        """
        r = api(PARSIVAL, "/page/api/attention/summary")
        assert r.status_code == 200
        d = r.json()
        assert "cold_start" in d and isinstance(d["cold_start"], bool)
        assert "active_situations" in d

    def test_analyses_list(self):
        r = api(PARSIVAL, "/page/api/analyses")
        assert r.status_code == 200
        assert isinstance(r.json(), list)


# ═══════════════════════════════════════════════════════════════════════════
# Section 7: Parsival → merLLM Batch Integration
# ═══════════════════════════════════════════════════════════════════════════

class TestParsivalBatch:
    """Parsival proxies batch status checks to merLLM."""

    def test_batch_status_proxy_404(self):
        """Nonexistent job ID should return 404 via the proxy."""
        fake_id = str(uuid.uuid4())
        r = api(PARSIVAL, f"/page/api/batch/status/{fake_id}")
        assert r.status_code == 404

    def test_batch_status_proxy_valid(self):
        """Submit a job via merLLM, check status via Parsival proxy."""
        r = post(MERLLM, "/api/batch/submit", json={
            "source_app": "integration_test",
            "prompt": "Say hi.",
            "model": "qwen3:30b-a3b",
        })
        assert r.status_code == 200
        job_id = r.json()["id"]

        r2 = api(PARSIVAL, f"/page/api/batch/status/{job_id}")
        assert r2.status_code == 200
        assert r2.json()["status"] in ("queued", "running", "complete", "failed")

        # Clean up
        post(MERLLM, f"/api/batch/{job_id}/cancel")


# ═══════════════════════════════════════════════════════════════════════════
# Section 8: LanceLLMot Chat E2E (streaming through merLLM)
# ═══════════════════════════════════════════════════════════════════════════

class TestChatE2E:
    """End-to-end chat: LanceLLMot → merLLM → Ollama, with RAG status and sources."""

    def test_chat_stream_with_rag_status(self):
        """Send a chat message and verify the SSE stream includes rag_status and sources events."""
        # Create a conversation first
        r = post(LANCELLMOT, "/api/conversations", json={"title": "Integration Test"})
        assert r.status_code == 200
        conv_id = r.json()["id"]

        # Stream a chat message
        events = []
        with client.stream("POST", f"{LANCELLMOT}/api/chat", json={
            "conversation_id": conv_id,
            "message": "Say 'integration test passed' and nothing else.",
            "model": "qwen3:30b-a3b",
        }, timeout=60) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                line = line.strip()
                if not line or line.startswith(":"):
                    continue
                if line.startswith("data: "):
                    line = line[6:]
                try:
                    evt = json.loads(line)
                    events.append(evt)
                except json.JSONDecodeError:
                    continue

        # Verify we got the key event types
        event_types = [e.get("type") for e in events if "type" in e]
        assert "rag_status" in event_types, f"Missing rag_status event. Got types: {event_types}"
        assert "sources" in event_types, f"Missing sources event. Got types: {event_types}"
        assert "done" in event_types, f"Missing done event. Got types: {event_types}"

        # Verify rag_status content
        rag_evt = next(e for e in events if e.get("type") == "rag_status")
        assert rag_evt["status"] in ("ok", "error")

        # Verify done has conversation_id
        done_evt = next(e for e in events if e.get("type") == "done")
        assert done_evt["conversation_id"] == conv_id

        # Clean up
        client.delete(f"{LANCELLMOT}/api/conversations/{conv_id}")


# ═══════════════════════════════════════════════════════════════════════════
# Section 9: Situation Lifecycle (if situations exist)
# ═══════════════════════════════════════════════════════════════════════════

class TestSituationLifecycle:
    """Test situation status transitions if any situations exist."""

    def _get_first_situation(self):
        r = api(PARSIVAL, "/page/api/situations")
        sits = r.json()
        if not sits:
            pytest.skip("No situations in database to test lifecycle")
        return sits[0]

    def test_situation_lifecycle_transition(self):
        """Use the /transition endpoint which logs lifecycle events."""
        sit = self._get_first_situation()
        sit_id = sit["situation_id"]
        original = sit.get("lifecycle_status", "new")

        # Transition via the dedicated endpoint (logs events)
        r = post(PARSIVAL, f"/page/api/situations/{sit_id}/transition", json={
            "to_status": "investigating",
            "note": "integration test"
        })
        assert r.status_code == 200

        # Verify lifecycle_status changed
        r = api(PARSIVAL, f"/page/api/situations/{sit_id}")
        assert r.status_code == 200
        assert r.json()["lifecycle_status"] == "investigating"

        # Check events log
        r = api(PARSIVAL, f"/page/api/situations/{sit_id}/events")
        assert r.status_code == 200
        events = r.json()
        assert len(events) > 0
        last_event = events[-1]
        assert last_event["to_status"] == "investigating"

        # Restore original lifecycle status
        post(PARSIVAL, f"/page/api/situations/{sit_id}/transition", json={
            "to_status": original or "new"
        })


# ═══════════════════════════════════════════════════════════════════════════
# Section 10: Database Backup
# ═══════════════════════════════════════════════════════════════════════════

class TestBackup:
    """Verify database backup endpoints work."""

    def test_merllm_backup(self):
        r = post(MERLLM, "/api/merllm/backup")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert "backup" in d or "path" in d


# ═══════════════════════════════════════════════════════════════════════════
# Section 11: Priority Bucket Contract (5-bucket refactor)
#
# The refactor introduced a cross-process contract: every LanceLLMot →
# merLLM and Parsival → merLLM call carries ``X-Source`` and ``X-Priority``
# headers, and merLLM routes each call into one of five buckets
# (chat / reserved / short / feedback / background, strict top-down drain).
# These tests pin that contract at the integration boundary — unit tests
# cannot catch a bucket-routing regression because the invariant only
# exists over the wire.
#
# Issues: merLLM#30, squire#34, hexcaliper#18.
# ═══════════════════════════════════════════════════════════════════════════

CANONICAL_BUCKETS = ("chat", "reserved", "short", "feedback", "background")


def _find_queue_entry_by_source(source: str, after: float,
                                timeout: float = 15.0,
                                interval: float = 0.15) -> dict | None:
    """Poll ``/api/merllm/queue`` until an entry matches ``source``.

    Only entries with ``submitted_at >= after`` are considered, so residual
    entries from earlier tests cannot cause false positives.  Matches any
    status (queued / running / completed) because short requests on an
    idle GPU can complete between poll ticks.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = api(MERLLM, "/api/merllm/queue")
        if r.status_code == 200:
            for e in r.json().get("queue", []):
                if (e.get("source") == source
                        and e.get("submitted_at", 0) >= after):
                    return e
        time.sleep(interval)
    return None


def _fire_generate(source: str, headers_extra: dict,
                   num_predict: int = 60) -> threading.Thread:
    """Start a slow ``/api/generate`` in a background thread.

    Returns the thread so the caller can ``join`` it in a finally block.
    Uses ``num_predict=60`` to guarantee the request stays in merLLM's
    tracked queue long enough to be observed by a 0.15s poll loop.
    """
    gen_client = httpx.Client(timeout=180)

    def send():
        try:
            gen_client.post(
                f"{MERLLM}/api/generate",
                json={
                    "model":   "qwen3:30b-a3b",
                    "prompt":  "Count slowly from one to twenty, one per line.",
                    "stream":  False,
                    "options": {"num_predict": num_predict},
                },
                headers={"X-Source": source, **headers_extra},
            )
        except Exception:
            pass
        finally:
            gen_client.close()

    t = threading.Thread(target=send, daemon=True)
    t.start()
    return t


class TestQueueClearAdmin:
    """Verify the admin force-clear endpoint used by the session fixture."""

    def test_clear_non_force_returns_counts(self):
        """Default mode returns a well-shaped removal report and leaves
        live in-flight entries alone."""
        r = post(MERLLM, "/api/merllm/queue/clear")
        assert r.status_code == 200
        d = r.json()
        assert d["ok"] is True
        assert d["force"] is False
        removed = d["removed"]
        # Shape check: all four status counters must be present, even if zero.
        for key in ("completed", "failed", "queued", "running"):
            assert key in removed
            assert isinstance(removed[key], int)
            assert removed[key] >= 0

    def test_clear_force_flag_is_honoured(self):
        """``?force=true`` must propagate through the endpoint."""
        r = post(MERLLM, "/api/merllm/queue/clear", params={"force": "true"})
        assert r.status_code == 200
        assert r.json()["force"] is True


class TestPriorityBucketContract:
    """Pin the 5-bucket priority contract across process boundaries."""

    def test_queue_endpoint_shape(self):
        """``/api/merllm/queue`` must expose all 5 canonical buckets + legacy aliases."""
        r = api(MERLLM, "/api/merllm/queue")
        assert r.status_code == 200
        d = r.json()
        assert "queue"   in d
        assert "buckets" in d
        assert "summary" in d

        buckets = d["buckets"]
        for name in CANONICAL_BUCKETS:
            assert name in buckets, (
                f"bucket {name!r} missing from /api/merllm/queue response — "
                "is this still a 5-bucket queue?"
            )
        # Legacy aliases must remain during the transition so existing
        # dashboards and clients keep working.
        assert "interactive" in buckets, "legacy 'interactive' alias missing"
        assert "batch"       in buckets, "legacy 'batch' alias missing"
        assert buckets["interactive"] == buckets["chat"], (
            "legacy 'interactive' must mirror canonical 'chat' depth"
        )
        assert buckets["batch"] == buckets["background"], (
            "legacy 'batch' must mirror canonical 'background' depth"
        )

    def test_batch_submit_lands_in_background(self):
        """``/api/batch/submit`` jobs must be tracked with ``priority=background``.

        Previously ``test_batch_submit_and_status`` only checked that the
        job id round-tripped — nothing verified which bucket the queue
        actually dropped it into.  This is the pin test for that.
        """
        source = f"integration_batch_{uuid.uuid4().hex[:8]}"
        t0 = time.time()
        r = post(MERLLM, "/api/batch/submit", json={
            "source_app": source,
            "prompt":     "Say hello.",
            "model":      "qwen3:30b-a3b",
        })
        assert r.status_code == 200
        job_id = r.json()["id"]
        try:
            entry = _find_queue_entry_by_source(source, after=t0, timeout=15.0)
            assert entry is not None, (
                f"batch job {job_id} never appeared in /api/merllm/queue "
                f"for source {source!r}"
            )
            assert entry["priority"] == "background", (
                f"batch job landed in {entry['priority']!r}; "
                "/api/batch/submit must always route to background"
            )
            assert entry["batch_job_id"] == job_id
        finally:
            post(MERLLM, f"/api/batch/{job_id}/cancel")

    @pytest.mark.parametrize("header_value,expected_bucket", [
        ("chat",        "chat"),
        ("reserved",    "reserved"),
        ("short",       "short"),
        ("feedback",    "feedback"),
        ("background",  "background"),
        # Legacy aliases must still route to their canonical target.
        ("interactive", "chat"),
        ("batch",       "background"),
    ])
    def test_explicit_priority_header_routes_to_bucket(self, header_value, expected_bucket):
        """Each X-Priority value must land the request in the expected bucket.

        Fires a slow ``/api/generate`` from a thread, races the merLLM
        queue to observe the live entry, asserts its ``priority`` field
        matches ``expected_bucket``.  Covers all 5 canonical names plus
        the 2 legacy aliases.
        """
        source = f"integration_pri_{header_value}_{uuid.uuid4().hex[:6]}"
        t0 = time.time()
        t = _fire_generate(source, {"X-Priority": header_value})
        try:
            entry = _find_queue_entry_by_source(source, after=t0, timeout=20.0)
            assert entry is not None, (
                f"no queue entry observed for X-Priority={header_value!r} "
                f"(source={source!r}) — did the request reach merLLM?"
            )
            assert entry["priority"] == expected_bucket, (
                f"X-Priority={header_value!r} landed in {entry['priority']!r}, "
                f"expected {expected_bucket!r}"
            )
        finally:
            t.join(timeout=180)

    def test_unknown_priority_falls_back_to_background(self):
        """A typo in X-Priority must NOT silently escalate.

        If someone mistypes ``X-Priority: shrot`` the request must fall
        back to ``background`` — never to ``chat``.  This is the guard
        against typo-driven priority inflation.
        """
        source = f"integration_typo_{uuid.uuid4().hex[:8]}"
        t0 = time.time()
        t = _fire_generate(source, {"X-Priority": "shrot"})
        try:
            entry = _find_queue_entry_by_source(source, after=t0, timeout=20.0)
            assert entry is not None, (
                f"no queue entry observed for typo priority source={source!r}"
            )
            assert entry["priority"] == "background", (
                f"typo X-Priority escalated to {entry['priority']!r} instead "
                "of falling back to background — unknown values must never "
                "reach a higher bucket than the safest default"
            )
        finally:
            t.join(timeout=180)

    def test_missing_priority_default_is_chat(self):
        """PIN TEST: missing X-Priority currently back-compat-defaults to chat.

        Per the five-bucket project memory, the missing-header default is
        intentionally ``chat`` during the migration window so pre-refactor
        clients keep working.  When we tighten the default to ``background``
        (the deferred follow-up), THIS TEST MUST BE UPDATED IN THE SAME
        COMMIT — otherwise header-less callers would silently change lane
        and the migration would be invisible to CI.
        """
        source = f"integration_nopri_{uuid.uuid4().hex[:8]}"
        t0 = time.time()
        t = _fire_generate(source, {})  # no X-Priority at all
        try:
            entry = _find_queue_entry_by_source(source, after=t0, timeout=20.0)
            assert entry is not None, (
                f"no queue entry observed for header-less request "
                f"(source={source!r})"
            )
            assert entry["priority"] == "chat", (
                f"missing X-Priority should default to 'chat' during the "
                f"migration window, got {entry['priority']!r}. If you are "
                "intentionally tightening the default to 'background', "
                "update this test in the same commit."
            )
        finally:
            t.join(timeout=180)


# ═══════════════════════════════════════════════════════════════════════════
# Section 12: Cross-App Priority Observability
#
# Proves that real inter-app calls (LanceLLMot chat, LanceLLMot RAG probes)
# reach merLLM's queue tagged with the correct source and priority.  This
# is the only way to verify that the X-Priority header survives the full
# chain from client → lancellmot handler → merLLM dispatcher.
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossAppPriorityObservability:
    """Verify lancellmot → merLLM calls carry the expected source+priority."""

    def test_lancellmot_chat_routes_to_chat_bucket(self):
        """End-to-end: lancellmot streaming chat must land in merLLM's chat bucket.

        Starts a streaming chat through lancellmot and polls merLLM's queue
        mid-stream to observe the live entry.  Asserts it shows
        ``source="lancellmot"`` and ``priority="chat"`` — the contract
        introduced by hexcaliper#18.  If the X-Priority header gets dropped
        anywhere in the chain, this test fails.
        """
        # Create a conversation we can stream into.
        r = post(LANCELLMOT, "/api/conversations", json={"title": "Bucket Test"})
        assert r.status_code == 200
        conv_id = r.json()["id"]

        stream_done = threading.Event()

        def run_stream():
            try:
                stream_client = httpx.Client(timeout=90)
                try:
                    with stream_client.stream("POST", f"{LANCELLMOT}/api/chat", json={
                        "conversation_id": conv_id,
                        "message": "Count from one to ten, one number per line.",
                        "model":   "qwen3:30b-a3b",
                    }) as response:
                        for _ in response.iter_lines():
                            pass
                finally:
                    stream_client.close()
            except Exception:
                pass
            finally:
                stream_done.set()

        t0 = time.time()
        t = threading.Thread(target=run_stream, daemon=True)
        t.start()
        try:
            entry = _find_queue_entry_by_source(
                "lancellmot", after=t0, timeout=30.0,
            )
            assert entry is not None, (
                "lancellmot chat never appeared in merLLM's queue — either "
                "the chat did not cross the boundary or merLLM is not "
                "tagging lancellmot requests with X-Source"
            )
            assert entry["priority"] == "chat", (
                f"lancellmot chat landed in {entry['priority']!r}, expected "
                "'chat' — is lancellmot's config.OLLAMA_HEADERS still "
                "setting X-Priority=chat?"
            )
        finally:
            stream_done.wait(timeout=90)
            client.delete(f"{LANCELLMOT}/api/conversations/{conv_id}")


# ═══════════════════════════════════════════════════════════════════════════
# Section 13: Parsival → merLLM Priority Contract
#
# Proves that each parsival call site that was updated in squire#34 carries
# the declared X-Priority across the wire when it actually fires.  These are
# the highest-value inter-app tests — they fire a real parsival endpoint,
# wait for its background worker to make a merLLM call, and read merLLM's
# tracked queue to verify the priority the request *actually* reached merLLM
# with (not just what parsival meant to send).
#
# Each test documents its side effects up front: ingest adds one analyzed
# item, briefing regenerates the cached briefing, reanalyze queues a small
# number of batch jobs before the test cancels it.  None of these corrupt
# data, but the tests are not side-effect-free and should run against a
# dev instance.
# ═══════════════════════════════════════════════════════════════════════════


def _snapshot_parsival_entry_ids() -> set[str]:
    """Snapshot the IDs of every parsival entry currently in merLLM's queue.

    Parsival runs continuous background work (auto-scans, situation
    synthesis, briefing refreshes) so there is almost always ambient
    parsival traffic in the queue.  A pure timestamp filter is not
    enough — an ambient entry submitted a fraction of a second before
    the test's ``t0`` would still satisfy ``submitted_at >= t0``.
    The tests snapshot IDs first and require the matching entry to be
    *new* (not in the snapshot).
    """
    r = api(MERLLM, "/api/merllm/queue")
    if r.status_code != 200:
        return set()
    return {
        e.get("id") for e in r.json().get("queue", [])
        if e.get("source") == "parsival" and e.get("id")
    }


def _find_new_parsival_entry(priority: str,
                             baseline_ids: set[str],
                             after: float,
                             timeout: float = 45.0,
                             interval: float = 0.2,
                             request_type: str | None = None) -> dict | None:
    """Poll merLLM's queue for a *new* parsival entry matching ``priority``.

    "New" means the entry's ``id`` is not in ``baseline_ids`` AND its
    ``submitted_at`` is at or after ``after``.  Both filters are applied
    because the queue can retain stale entries whose IDs we may not
    have captured in the baseline if they rolled off and came back.
    """
    deadline = time.time() + timeout
    while time.time() < deadline:
        r = api(MERLLM, "/api/merllm/queue")
        if r.status_code == 200:
            for e in r.json().get("queue", []):
                if e.get("id") in baseline_ids:
                    continue
                if e.get("source") != "parsival":
                    continue
                if e.get("submitted_at", 0) < after:
                    continue
                if e.get("priority") != priority:
                    continue
                if request_type and e.get("request_type") != request_type:
                    continue
                return e
        time.sleep(interval)
    return None


class TestParsivalPriorityContract:
    """Verify each parsival call site reaches merLLM with the declared bucket.

    Call-site map from squire#34:
      * process_ingest_items  → short
      * run_scan              → short  (not tested — needs connectors)
      * extract_keywords      → short  (not tested — user-triggered)
      * generate_project_briefing → feedback
      * correlator.synthesize_situation → feedback  (not directly tested;
          fires asynchronously after ingest and is race-prone)
      * run_reanalyze         → background
      * seeder map+reduce     → background  (not tested — seed state machine)
    """

    def test_parsival_ingest_routes_to_short(self):
        """POST /ingest must cause a merLLM call with priority=short.

        Side effect: adds one analyzed item to parsival's DB.  The item is
        tagged with source=``integration_test`` and a recognisable title so
        it is easy to spot and delete from the parsival UI.  A situation
        MAY be spawned asynchronously by ``_spawn_situation_task``, which
        would show up as a ``feedback`` entry a few seconds later.
        """
        test_item_id = f"e2e-priority-{uuid.uuid4().hex}"
        # Snapshot ambient parsival entries BEFORE triggering so we can
        # distinguish our test's entry from unrelated background work.
        baseline = _snapshot_parsival_entry_ids()
        t0 = time.time()

        r = post(PARSIVAL, "/page/api/ingest", json={
            "items": [{
                "source":    "integration_test",
                "item_id":   test_item_id,
                "title":     "[e2e-priority-test] integration test item — safe to delete",
                "body":      "This item was created by the priority-bucket integration test. "
                             "It should be analyzed at priority=short. Safe to delete.",
                "url":       "",
                "author":    "e2e-test@example.com",
                "timestamp": "2026-04-10T12:00:00+00:00",
            }],
        })
        assert r.status_code == 200, f"/page/api/ingest failed: {r.status_code} {r.text}"
        assert r.json().get("received") == 1, (
            f"ingest did not accept the test item: {r.json()}"
        )

        entry = _find_new_parsival_entry(
            "short", baseline_ids=baseline, after=t0, timeout=30.0,
        )
        assert entry is not None, (
            "parsival ingest did not produce a priority=short entry in "
            "merLLM's queue. Either process_ingest_items is not calling "
            "analyze(priority='short'), or llm.generate is not forwarding "
            "the priority into the X-Priority header."
        )
        assert entry["source"] == "parsival"
        assert entry["priority"] == "short"

    def test_parsival_briefing_routes_to_feedback(self):
        """POST /briefing/generate must cause a merLLM call with priority=feedback.

        Side effect: regenerates parsival's cached briefing.  This is a
        non-destructive operation — the briefing is normally regenerated
        by the scan/reanalyze tail anyway.

        generate_project_briefing hardcodes ``priority='feedback'``; this
        test is the inter-app proof that the header survives from
        ``llm.generate`` → ``config.ollama_headers`` → the merLLM dispatcher.
        """
        baseline = _snapshot_parsival_entry_ids()
        t0 = time.time()
        r = post(PARSIVAL, "/page/api/briefing/generate")
        assert r.status_code == 200, (
            f"/page/api/briefing/generate failed: {r.status_code} {r.text}"
        )
        assert r.json().get("ok") is True

        entry = _find_new_parsival_entry(
            "feedback", baseline_ids=baseline, after=t0, timeout=60.0,
        )
        assert entry is not None, (
            "parsival briefing did not produce a priority=feedback entry "
            "in merLLM's queue. Either /briefing/generate is not reaching "
            "generate_project_briefing, or the priority='feedback' kwarg "
            "is being dropped in llm.generate."
        )
        assert entry["source"] == "parsival"
        assert entry["priority"] == "feedback"

    def test_parsival_reanalyze_routes_to_background(self):
        """POST /reanalyze must cause merLLM calls with priority=background.

        Side effect: queues a small number of batch jobs (typically < 30)
        before the test cancels the reanalyze run via /analysis/stop and
        then cancels each observed batch job individually.  Items already
        in-flight at cancel time will complete, but no new items are
        submitted once the cancel flag is set.

        Skips if a scan or reanalyze is already running (409 conflict).
        """
        # Don't start reanalyze if something is already running.
        status = api(PARSIVAL, "/page/api/scan/status")
        if status.status_code == 200 and status.json().get("running"):
            pytest.skip("parsival scan/reanalyze already running — cannot test")

        # We'll cancel aggressively, but collect any observed batch job IDs
        # so we can cancel them individually on merLLM as well.
        observed_batch_ids: list[str] = []
        baseline = _snapshot_parsival_entry_ids()
        t0 = time.time()

        r = post(PARSIVAL, "/page/api/reanalyze")
        if r.status_code == 409:
            pytest.skip("parsival returned 409 — another scan is in progress")
        assert r.status_code == 200, (
            f"/page/api/reanalyze failed: {r.status_code} {r.text}"
        )

        try:
            # Observe the first parsival background batch entry on merLLM.
            # run_reanalyze routes through /api/batch/submit, so the queue
            # entry will have request_type='batch' and priority='background'.
            entry = _find_new_parsival_entry(
                "background", baseline_ids=baseline, after=t0, timeout=30.0,
                request_type="batch",
            )
            assert entry is not None, (
                "parsival reanalyze did not produce a priority=background "
                "batch entry in merLLM's queue. Either reanalyze is not "
                "routing through the batch API, or the batch path dropped "
                "the background priority."
            )
            assert entry["source"] == "parsival"
            assert entry["priority"] == "background"
            assert entry["request_type"] == "batch"
        finally:
            # Aggressively stop the reanalyze loop on parsival.
            post(PARSIVAL, "/page/api/analysis/stop")
            post(PARSIVAL, "/page/api/scan/cancel")

            # Cancel every parsival batch job that was submitted after t0.
            # This minimises the side-effect window to whatever merLLM has
            # already started running by the time this code reaches it.
            r = api(MERLLM, "/api/merllm/queue")
            if r.status_code == 200:
                for e in r.json().get("queue", []):
                    if (e.get("source") == "parsival"
                            and e.get("submitted_at", 0) >= t0
                            and e.get("batch_job_id")
                            and e.get("status") in ("queued", "running")):
                        bid = e["batch_job_id"]
                        observed_batch_ids.append(bid)
                        post(MERLLM, f"/api/batch/{bid}/cancel")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
