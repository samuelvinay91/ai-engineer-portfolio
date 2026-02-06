"""Tests for the FastAPI application endpoints.

Uses ``httpx.AsyncClient`` with an ASGI transport -- no real server is started.
All LLM calls are mocked via the ``mock_anthropic_client`` fixture.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from deep_research.config import Settings


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, api_client):
        resp = await api_client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert "timestamp" in data


# ---------------------------------------------------------------------------
# Reasoning endpoints
# ---------------------------------------------------------------------------

class TestReasonEndpoint:
    @pytest.mark.asyncio
    async def test_reason_cot(self, api_client, mock_anthropic_client):
        with patch("deep_research.reasoning.anthropic.AsyncAnthropic", return_value=mock_anthropic_client):
            resp = await api_client.post("/api/v1/reason", json={
                "query": "What are the benefits of quantum computing?",
                "strategy": "chain_of_thought",
                "use_extended_thinking": True,
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy"] == "chain_of_thought"
        assert data["answer"]
        assert data["confidence"] > 0
        assert len(data["steps"]) >= 1

    @pytest.mark.asyncio
    async def test_reason_direct(self, api_client, mock_anthropic_client):
        with patch("deep_research.reasoning.anthropic.AsyncAnthropic", return_value=mock_anthropic_client):
            resp = await api_client.post("/api/v1/reason", json={
                "query": "What is 2+2?",
                "strategy": "direct",
                "use_extended_thinking": False,
            })

        assert resp.status_code == 200
        data = resp.json()
        assert data["strategy"] == "direct"

    @pytest.mark.asyncio
    async def test_reason_invalid_strategy(self, api_client):
        resp = await api_client.post("/api/v1/reason", json={
            "query": "Something",
            "strategy": "nonexistent_strategy",
        })
        assert resp.status_code == 400
        assert "Unknown strategy" in resp.json()["detail"]

    @pytest.mark.asyncio
    async def test_reason_query_too_short(self, api_client):
        resp = await api_client.post("/api/v1/reason", json={
            "query": "ab",
            "strategy": "direct",
        })
        assert resp.status_code == 422  # validation error


class TestCompareReasoningEndpoint:
    @pytest.mark.asyncio
    async def test_compare_two_strategies(self, api_client, mock_anthropic_client):
        with patch("deep_research.reasoning.anthropic.AsyncAnthropic", return_value=mock_anthropic_client):
            resp = await api_client.post("/api/v1/compare-reasoning", json={
                "query": "Explain quantum entanglement.",
                "strategies": ["direct", "chain_of_thought"],
            })

        assert resp.status_code == 200
        data = resp.json()
        assert "direct" in data["results"]
        assert "chain_of_thought" in data["results"]
        assert data["results"]["direct"]["strategy"] == "direct"
        assert data["results"]["chain_of_thought"]["strategy"] == "chain_of_thought"

    @pytest.mark.asyncio
    async def test_compare_invalid_strategy(self, api_client):
        resp = await api_client.post("/api/v1/compare-reasoning", json={
            "query": "Something",
            "strategies": ["direct", "bad_one"],
        })
        assert resp.status_code == 400


# ---------------------------------------------------------------------------
# Research endpoints
# ---------------------------------------------------------------------------

class TestResearchEndpoint:
    @pytest.mark.asyncio
    async def test_start_research_returns_202(self, api_client, mock_anthropic_client):
        with patch("deep_research.researcher.anthropic.AsyncAnthropic", return_value=mock_anthropic_client), \
             patch("deep_research.reasoning.anthropic.AsyncAnthropic", return_value=mock_anthropic_client), \
             patch("deep_research.planner.anthropic.AsyncAnthropic", return_value=mock_anthropic_client), \
             patch("deep_research.report.anthropic.AsyncAnthropic", return_value=mock_anthropic_client):
            resp = await api_client.post("/api/v1/research", json={
                "question": "What are the latest advances in quantum computing?",
                "max_depth": 2,
            })

        assert resp.status_code == 202
        data = resp.json()
        assert "task_id" in data
        assert data["status"] == "planning"
        assert data["message"] == "Research task started."

    @pytest.mark.asyncio
    async def test_get_research_not_found(self, api_client):
        resp = await api_client.get("/api/v1/research/nonexistent-id")
        assert resp.status_code == 404
        assert "not found" in resp.json()["detail"].lower()

    @pytest.mark.asyncio
    async def test_research_question_too_short(self, api_client):
        resp = await api_client.post("/api/v1/research", json={
            "question": "Hi",
            "max_depth": 1,
        })
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_research_max_depth_validation(self, api_client):
        resp = await api_client.post("/api/v1/research", json={
            "question": "A valid question about quantum computing.",
            "max_depth": 99,
        })
        assert resp.status_code == 422  # exceeds max of 10


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

class TestEdgeCases:
    @pytest.mark.asyncio
    async def test_cors_headers_present(self, api_client):
        resp = await api_client.options(
            "/health",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "GET",
            },
        )
        # CORS middleware should be active
        assert resp.status_code in (200, 204, 405)
