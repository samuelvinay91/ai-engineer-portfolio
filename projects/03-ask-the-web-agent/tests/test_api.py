"""Tests for the FastAPI endpoints.

All external dependencies (LLM calls, search API) are mocked so these tests
can run without API keys or network access.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from ask_the_web.agents.fact_checker import FactCheckReport
from ask_the_web.agents.router import RouteType, RoutingDecision
from ask_the_web.agents.searcher import SearchResponse, SearchResult
from ask_the_web.agents.synthesizer import SynthesisResult
from ask_the_web.api import create_app
from ask_the_web.config import Settings
from ask_the_web.workflow import WorkflowState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def app(settings: Settings):
    """Create a FastAPI app with test settings."""
    with patch("ask_the_web.api.get_settings", return_value=settings):
        application = create_app()
        yield application


@pytest.fixture()
async def client(app) -> AsyncClient:
    """Async HTTP client bound to the test app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_200(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "ask-the-web-agent"
        assert "version" in data


# ---------------------------------------------------------------------------
# POST /api/v1/ask
# ---------------------------------------------------------------------------


class TestAskEndpoint:
    @pytest.mark.asyncio
    async def test_ask_returns_answer(
        self,
        client: AsyncClient,
        sample_synthesis_result: SynthesisResult,
        sample_fact_check_report: FactCheckReport,
    ) -> None:
        """Successful ask request returns a full answer response."""
        mock_state: WorkflowState = {
            "query": "What is Python?",
            "routing_decision": None,
            "search_response": None,
            "search_results": [],
            "synthesis_result": sample_synthesis_result,
            "fact_check_report": sample_fact_check_report,
            "answer": sample_synthesis_result.answer,
            "citations": [c.model_dump() for c in sample_synthesis_result.citations],
            "follow_up_questions": sample_synthesis_result.follow_up_questions,
            "error": None,
            "elapsed_ms": 1234.5,
            "route_used": "simple_search",
        }

        with patch("ask_the_web.api.run_ask_pipeline", new_callable=AsyncMock) as mock_pipeline:
            mock_pipeline.return_value = mock_state

            resp = await client.post(
                "/api/v1/ask",
                json={"query": "What is Python?"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "What is Python?"
        assert "population" in data["answer"].lower() or len(data["answer"]) > 0
        assert data["route_used"] == "simple_search"
        assert data["elapsed_ms"] == 1234.5
        assert "request_id" in data
        assert len(data["citations"]) == 3
        assert len(data["follow_up_questions"]) == 3

    @pytest.mark.asyncio
    async def test_ask_validates_empty_query(self, client: AsyncClient) -> None:
        """Empty query should return a 422 validation error."""
        resp = await client.post("/api/v1/ask", json={"query": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_ask_handles_pipeline_error(self, client: AsyncClient) -> None:
        """If the pipeline raises, the endpoint should return 500."""
        with patch("ask_the_web.api.run_ask_pipeline", new_callable=AsyncMock) as mock_pipeline:
            mock_pipeline.side_effect = RuntimeError("LLM unavailable")

            resp = await client.post(
                "/api/v1/ask",
                json={"query": "Will this fail?"},
            )

        assert resp.status_code == 500


# ---------------------------------------------------------------------------
# POST /api/v1/search
# ---------------------------------------------------------------------------


class TestSearchEndpoint:
    @pytest.mark.asyncio
    async def test_search_returns_results(
        self,
        client: AsyncClient,
        sample_search_response: SearchResponse,
    ) -> None:
        """Raw search endpoint returns search results."""
        with patch.object(
            type(client._transport._app.state.searcher),  # type: ignore[union-attr]
            "search",
            new_callable=AsyncMock,
            return_value=sample_search_response,
        ):
            resp = await client.post(
                "/api/v1/search",
                json={"query": "population of France", "strategy": "general"},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["query"] == "population of France 2024"
        assert data["total_results"] == 3

    @pytest.mark.asyncio
    async def test_search_validates_strategy(self, client: AsyncClient) -> None:
        """Invalid strategy should be rejected."""
        resp = await client.post(
            "/api/v1/search",
            json={"query": "test", "strategy": "invalid_strategy"},
        )
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# GET /api/v1/history
# ---------------------------------------------------------------------------


class TestHistoryEndpoint:
    @pytest.mark.asyncio
    async def test_history_empty_initially(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/history")
        assert resp.status_code == 200
        assert resp.json() == []

    @pytest.mark.asyncio
    async def test_history_populated_after_ask(
        self,
        client: AsyncClient,
        sample_synthesis_result: SynthesisResult,
        sample_fact_check_report: FactCheckReport,
    ) -> None:
        """After an /ask call, history should contain one entry."""
        mock_state: WorkflowState = {
            "query": "test question",
            "routing_decision": None,
            "search_response": None,
            "search_results": [],
            "synthesis_result": sample_synthesis_result,
            "fact_check_report": sample_fact_check_report,
            "answer": "Test answer",
            "citations": [],
            "follow_up_questions": [],
            "error": None,
            "elapsed_ms": 500.0,
            "route_used": "simple_search",
        }

        with patch("ask_the_web.api.run_ask_pipeline", new_callable=AsyncMock) as mock_pipeline:
            mock_pipeline.return_value = mock_state
            await client.post("/api/v1/ask", json={"query": "test question"})

        resp = await client.get("/api/v1/history")
        assert resp.status_code == 200
        history = resp.json()
        assert len(history) >= 1
        assert history[0]["query"] == "test question"
        assert history[0]["route_used"] == "simple_search"


# ---------------------------------------------------------------------------
# POST /api/v1/ask/stream
# ---------------------------------------------------------------------------


class TestStreamEndpoint:
    @pytest.mark.asyncio
    async def test_stream_returns_sse(self, client: AsyncClient) -> None:
        """Streaming endpoint should return an SSE response."""
        mock_decision = RoutingDecision(
            route=RouteType.SIMPLE_SEARCH,
            confidence=0.9,
            reasoning="Factual question",
            reformulated_query="test",
            search_queries=["test"],
        )
        mock_search_resp = SearchResponse(
            query="test",
            results=[
                SearchResult(
                    title="Test",
                    url="https://example.com",
                    snippet="Test snippet",
                    content="Test content",
                    score=0.9,
                    source_domain="example.com",
                    content_hash="abc123",
                ),
            ],
            total_results=1,
            search_queries_used=["test"],
            search_duration_ms=100.0,
        )

        async def mock_stream(*args: Any, **kwargs: Any):
            yield "Hello "
            yield "world!"

        with (
            patch(
                "ask_the_web.api.QueryRouterAgent"
            ) as MockRouter,
        ):
            MockRouter.return_value.route = AsyncMock(return_value=mock_decision)

            with patch.object(
                type(client._transport._app.state.searcher),  # type: ignore[union-attr]
                "search",
                new_callable=AsyncMock,
                return_value=mock_search_resp,
            ):
                with patch.object(
                    type(client._transport._app.state.synthesiser),  # type: ignore[union-attr]
                    "synthesise_stream",
                    side_effect=mock_stream,
                ):
                    resp = await client.post(
                        "/api/v1/ask/stream",
                        json={"query": "What is Python?"},
                    )

        # SSE responses return 200 with text/event-stream content type
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
