"""Tests for the FastAPI application endpoints.

Uses httpx.AsyncClient with the ASGI transport to test the API without
starting a real server or making external service calls.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from capstone.agents.base import AgentResult, TaskStatus
from capstone.api import TaskStore, create_app
from capstone.config import Settings
from capstone.memory import MemoryManager, ShortTermMemory
from capstone.orchestrator import MultiAgentOrchestrator, OrchestratorState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def app(settings: Settings):
    """Create a FastAPI app instance for testing."""
    return create_app(settings)


@pytest.fixture()
async def client(app) -> AsyncClient:
    """Create an httpx AsyncClient bound to the test app."""
    # We need to manage the lifespan context for the app
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    """Tests for the /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_check(self, settings: Settings):
        """Health endpoint returns 200 with service metadata."""
        import capstone.api as api_module

        # Mock the global state
        mock_orchestrator = MagicMock(spec=MultiAgentOrchestrator)
        mock_orchestrator.list_agents.return_value = [
            {"name": "researcher", "description": "test", "capabilities": []},
            {"name": "coder", "description": "test", "capabilities": []},
        ]

        original = api_module._orchestrator
        api_module._orchestrator = mock_orchestrator

        try:
            app = create_app(settings)
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                response = await client.get("/health")
        finally:
            api_module._orchestrator = original

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "capstone-multiagent"
        assert data["agents"] == 2


# ---------------------------------------------------------------------------
# Agents endpoint
# ---------------------------------------------------------------------------


class TestAgentsEndpoint:
    """Tests for the /api/v1/agents endpoint."""

    @pytest.mark.asyncio
    async def test_list_agents(self, settings: Settings):
        """GET /api/v1/agents returns registered agent metadata."""
        import capstone.api as api_module

        mock_orchestrator = MagicMock(spec=MultiAgentOrchestrator)
        mock_orchestrator.list_agents.return_value = [
            {"name": "researcher", "description": "Research agent", "capabilities": ["research"]},
            {"name": "coder", "description": "Coding agent", "capabilities": ["code"]},
        ]

        original = api_module._orchestrator
        api_module._orchestrator = mock_orchestrator

        try:
            app = create_app(settings)
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                response = await client.get("/api/v1/agents")
        finally:
            api_module._orchestrator = original

        assert response.status_code == 200
        agents = response.json()
        assert len(agents) == 2
        assert agents[0]["name"] == "researcher"
        assert "capabilities" in agents[0]


# ---------------------------------------------------------------------------
# Task endpoints
# ---------------------------------------------------------------------------


class TestTaskEndpoints:
    """Tests for task submission and retrieval endpoints."""

    @pytest.mark.asyncio
    async def test_submit_task(self, settings: Settings):
        """POST /api/v1/tasks submits and executes a task."""
        import capstone.api as api_module

        completed_state = OrchestratorState(
            task_id="test-123",
            user_request="Test request",
            plan="Execute directly.",
            final_output="Here is the answer.",
            status="completed",
            session_id="sess-1",
            agent_results=[
                AgentResult(
                    task_id="sub-1",
                    agent_name="researcher",
                    status=TaskStatus.COMPLETED,
                    output="Research done.",
                    confidence=0.9,
                )
            ],
        )

        mock_orchestrator = MagicMock(spec=MultiAgentOrchestrator)
        mock_orchestrator.run = AsyncMock(return_value=completed_state)

        mock_memory = MagicMock(spec=MemoryManager)
        mock_memory.add_message = MagicMock()

        mock_store = TaskStore()

        original_orch = api_module._orchestrator
        original_mem = api_module._memory
        original_store = api_module._task_store
        api_module._orchestrator = mock_orchestrator
        api_module._memory = mock_memory
        api_module._task_store = mock_store

        try:
            app = create_app(settings)
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/api/v1/tasks",
                    json={"request": "Test request", "session_id": "sess-1"},
                )
        finally:
            api_module._orchestrator = original_orch
            api_module._memory = original_mem
            api_module._task_store = original_store

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "test-123"
        assert data["status"] == "completed"
        assert data["final_output"] == "Here is the answer."
        assert len(data["agent_results"]) == 1

    @pytest.mark.asyncio
    async def test_get_task_not_found(self, settings: Settings):
        """GET /api/v1/tasks/{task_id} returns 404 for unknown task."""
        import capstone.api as api_module

        mock_store = TaskStore()
        original = api_module._task_store
        api_module._task_store = mock_store

        try:
            app = create_app(settings)
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                response = await client.get("/api/v1/tasks/nonexistent")
        finally:
            api_module._task_store = original

        assert response.status_code == 404

    @pytest.mark.asyncio
    async def test_get_task_found(self, settings: Settings):
        """GET /api/v1/tasks/{task_id} returns task data when found."""
        import capstone.api as api_module

        state = OrchestratorState(
            task_id="found-123",
            user_request="A request",
            status="completed",
            final_output="The answer.",
        )

        mock_store = TaskStore()
        mock_store.save(state)
        original = api_module._task_store
        api_module._task_store = mock_store

        try:
            app = create_app(settings)
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                response = await client.get("/api/v1/tasks/found-123")
        finally:
            api_module._task_store = original

        assert response.status_code == 200
        data = response.json()
        assert data["task_id"] == "found-123"
        assert data["final_output"] == "The answer."


# ---------------------------------------------------------------------------
# Agent direct execution
# ---------------------------------------------------------------------------


class TestAgentDirectExecution:
    """Tests for the /api/v1/agents/{agent_name}/execute endpoint."""

    @pytest.mark.asyncio
    async def test_execute_unknown_agent(self, settings: Settings):
        """Executing a non-existent agent returns 404."""
        import capstone.api as api_module

        mock_orchestrator = MagicMock(spec=MultiAgentOrchestrator)
        mock_orchestrator.list_agents.return_value = [
            {"name": "researcher", "description": "test", "capabilities": []},
        ]

        original = api_module._orchestrator
        api_module._orchestrator = mock_orchestrator

        try:
            app = create_app(settings)
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                response = await client.post(
                    "/api/v1/agents/nonexistent/execute",
                    json={"description": "Do something"},
                )
        finally:
            api_module._orchestrator = original

        assert response.status_code == 404


# ---------------------------------------------------------------------------
# Memory endpoint
# ---------------------------------------------------------------------------


class TestMemoryEndpoint:
    """Tests for the /api/v1/memory/{session_id} endpoint."""

    @pytest.mark.asyncio
    async def test_get_session_memory(self, settings: Settings):
        """GET /api/v1/memory/{session_id} returns memory summary."""
        import capstone.api as api_module

        mock_memory = MagicMock(spec=MemoryManager)
        mock_memory.get_session_summary = AsyncMock(
            return_value={
                "session_id": "sess-test",
                "short_term": {"message_count": 2, "messages": []},
                "long_term": {"entry_count": 0, "entries": []},
            }
        )

        original = api_module._memory
        api_module._memory = mock_memory

        try:
            app = create_app(settings)
            async with AsyncClient(
                transport=ASGITransport(app=app),
                base_url="http://testserver",
            ) as client:
                response = await client.get("/api/v1/memory/sess-test")
        finally:
            api_module._memory = original

        assert response.status_code == 200
        data = response.json()
        assert data["session_id"] == "sess-test"
        assert "short_term" in data
        assert "long_term" in data


# ---------------------------------------------------------------------------
# Task store
# ---------------------------------------------------------------------------


class TestTaskStore:
    """Tests for the in-memory task store."""

    def test_save_and_get(self):
        """Tasks can be saved and retrieved by ID."""
        store = TaskStore()
        state = OrchestratorState(
            task_id="store-test",
            user_request="test",
            status="completed",
        )
        store.save(state)
        retrieved = store.get("store-test")
        assert retrieved is not None
        assert retrieved.task_id == "store-test"

    def test_get_nonexistent_returns_none(self):
        """Retrieving a non-existent task returns None."""
        store = TaskStore()
        assert store.get("does-not-exist") is None
