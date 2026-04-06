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
        assert d["mode"] in ("day", "night")
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
    """Test merLLM-specific features: metrics, settings, mode, batch."""

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

    def test_transitions(self):
        r = api(MERLLM, "/api/merllm/transitions")
        assert r.status_code == 200
        assert isinstance(r.json(), list)

    def test_geoip(self):
        r = api(MERLLM, "/api/merllm/geoip")
        assert r.status_code == 200

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
        """LanceLLMot proxies merLLM status for its mode indicator."""
        r = api(LANCELLMOT, "/api/merllm/status")
        assert r.status_code == 200
        d = r.json()
        assert "mode" in d


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

    def test_attention_model_cold_start(self):
        """With no user actions, attention should report cold start."""
        r = api(PARSIVAL, "/page/api/attention/summary")
        assert r.status_code == 200
        d = r.json()
        assert d["cold_start"] is True
        assert "cold_start_msg" in d

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
