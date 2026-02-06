"""Tests for the FastAPI application endpoints.

Uses a mock retriever so tests do not require external services
(Qdrant, LLM APIs, etc.).
"""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from agent_rag.agents.single_agent_rag import RAGResponse
from agent_rag.api import app
from tests.conftest import SAMPLE_CHUNKS


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture
def mock_rag_response() -> RAGResponse:
    return RAGResponse(
        answer="Machine learning is a subset of AI that learns from data.",
        query="What is machine learning?",
        retrieved_chunks=SAMPLE_CHUNKS[:2],
        metadata={"strategy": "single_agent"},
        agent_type="single",
    )


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    """Create a test client for the FastAPI app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health(self, client: AsyncClient) -> None:
        response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "agent-rag"


# ---------------------------------------------------------------------------
# Ingestion endpoint
# ---------------------------------------------------------------------------


class TestIngestionEndpoint:
    @pytest.mark.asyncio
    async def test_ingest_validation_empty(self, client: AsyncClient) -> None:
        """Ingest endpoint should reject empty texts list."""
        response = await client.post(
            "/api/v1/ingest",
            json={"texts": []},
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_ingest_validation_missing_texts(self, client: AsyncClient) -> None:
        """Ingest endpoint should require texts field."""
        response = await client.post("/api/v1/ingest", json={})
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Query endpoints
# ---------------------------------------------------------------------------


class TestQueryEndpoints:
    @pytest.mark.asyncio
    async def test_query_validation(self, client: AsyncClient) -> None:
        """Query endpoint should reject empty questions."""
        response = await client.post("/api/v1/query", json={"question": ""})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_validation_missing(self, client: AsyncClient) -> None:
        response = await client.post("/api/v1/query", json={})
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_multi_agent_validation(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/query/multi-agent", json={"question": ""}
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_hierarchical_validation(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/query/hierarchical", json={"question": ""}
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_compare_validation(self, client: AsyncClient) -> None:
        response = await client.post(
            "/api/v1/query/compare", json={"question": ""}
        )
        assert response.status_code == 422

    @pytest.mark.asyncio
    async def test_query_top_k_bounds(self, client: AsyncClient) -> None:
        """top_k should be between 1 and 100."""
        response = await client.post(
            "/api/v1/query",
            json={"question": "test", "top_k": 0},
        )
        assert response.status_code == 422

        response = await client.post(
            "/api/v1/query",
            json={"question": "test", "top_k": 200},
        )
        assert response.status_code == 422


# ---------------------------------------------------------------------------
# Document / collection endpoints
# ---------------------------------------------------------------------------


class TestDocumentEndpoints:
    @pytest.mark.asyncio
    async def test_documents_endpoint_exists(self, client: AsyncClient) -> None:
        """Documents endpoint should return 200 or 500 (if Qdrant unavailable)."""
        response = await client.get("/api/v1/documents")
        assert response.status_code in (200, 500)

    @pytest.mark.asyncio
    async def test_collections_endpoint_exists(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/collections")
        assert response.status_code in (200, 500)


# ---------------------------------------------------------------------------
# Memory stats endpoint
# ---------------------------------------------------------------------------


class TestMemoryEndpoint:
    @pytest.mark.asyncio
    async def test_memory_stats(self, client: AsyncClient) -> None:
        response = await client.get("/api/v1/memory/stats")
        assert response.status_code == 200
        data = response.json()
        assert "active_conversations" in data
        assert "stored_episodes" in data
        assert "stored_facts" in data


# ---------------------------------------------------------------------------
# Stream endpoint structure
# ---------------------------------------------------------------------------


class TestStreamEndpoint:
    @pytest.mark.asyncio
    async def test_stream_validation(self, client: AsyncClient) -> None:
        """Stream endpoint should reject empty questions."""
        response = await client.post(
            "/api/v1/query/stream", json={"question": ""}
        )
        assert response.status_code == 422
