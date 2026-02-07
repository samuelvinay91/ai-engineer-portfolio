"""Shared fixtures for the capstone multi-agent test suite.

All fixtures here are designed to work without external services (no LLM API
calls, no database, no Redis).  Tests that require live services should be
marked with ``@pytest.mark.integration``.
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from capstone.agents.analyst import AnalystAgent
from capstone.agents.base import AgentResult, AgentTask, BaseAgent, TaskStatus
from capstone.agents.coder import CoderAgent
from capstone.agents.researcher import ResearcherAgent
from capstone.agents.writer import WriterAgent
from capstone.config import Settings
from capstone.memory import MemoryManager, ShortTermMemory, WorkingMemory
from capstone.orchestrator import MultiAgentOrchestrator, OrchestratorState


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings() -> Settings:
    """Return test settings with dummy API keys."""
    return Settings(
        anthropic_api_key="test-key-not-real",
        openai_api_key="test-key-not-real",
        database_url="sqlite+aiosqlite:///",
        redis_url="redis://localhost:6379/0",
        environment="local",
        log_level="DEBUG",
    )


# ---------------------------------------------------------------------------
# Mock LLM responses
# ---------------------------------------------------------------------------


def _make_mock_llm_response(content: str) -> MagicMock:
    """Create a mock LLM response object."""
    response = MagicMock()
    response.content = content
    return response


@pytest.fixture()
def mock_llm_invoke():
    """Patch ChatAnthropic.ainvoke to return controlled responses."""
    with patch("langchain_anthropic.ChatAnthropic.ainvoke", new_callable=AsyncMock) as mock:
        yield mock


# ---------------------------------------------------------------------------
# Agent fixtures
# ---------------------------------------------------------------------------


class StubAgent(BaseAgent):
    """A deterministic test agent that returns pre-configured results."""

    name = "stub"
    description = "Stub agent for testing."
    capabilities = ["testing"]

    def __init__(
        self,
        settings: Settings,
        output: str = "Stub output.",
        confidence: float = 0.9,
        can_handle_score: float = 0.5,
    ) -> None:
        super().__init__(settings)
        self._output = output
        self._confidence = confidence
        self._can_handle_score = can_handle_score

    async def execute(self, task: AgentTask) -> AgentResult:
        return AgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            status=TaskStatus.COMPLETED,
            output=self._output,
            confidence=self._confidence,
        )

    async def can_handle(self, task: AgentTask) -> float:
        return self._can_handle_score


@pytest.fixture()
def stub_agent(settings: Settings) -> StubAgent:
    return StubAgent(settings)


@pytest.fixture()
def researcher_agent(settings: Settings) -> ResearcherAgent:
    return ResearcherAgent(settings)


@pytest.fixture()
def coder_agent(settings: Settings) -> CoderAgent:
    return CoderAgent(settings)


@pytest.fixture()
def analyst_agent(settings: Settings) -> AnalystAgent:
    return AnalystAgent(settings)


@pytest.fixture()
def writer_agent(settings: Settings) -> WriterAgent:
    return WriterAgent(settings)


# ---------------------------------------------------------------------------
# Orchestrator fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def orchestrator_with_stubs(settings: Settings) -> MultiAgentOrchestrator:
    """Create an orchestrator with stub agents (no real LLM calls)."""
    orchestrator = MultiAgentOrchestrator(settings)

    for name, desc, caps, score in [
        ("researcher", "Stub researcher", ["research"], 0.3),
        ("coder", "Stub coder", ["code"], 0.3),
        ("analyst", "Stub analyst", ["data"], 0.3),
        ("writer", "Stub writer", ["writing"], 0.3),
    ]:
        agent = StubAgent(
            settings,
            output=f"Output from {name}.",
            can_handle_score=score,
        )
        agent.name = name
        agent.description = desc
        agent.capabilities = caps
        orchestrator.register_agent(agent)

    return orchestrator


# ---------------------------------------------------------------------------
# Memory fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def short_term_memory() -> ShortTermMemory:
    return ShortTermMemory(session_id="test-session", max_messages=10)


@pytest.fixture()
def working_memory() -> WorkingMemory:
    return WorkingMemory(ttl_seconds=60)


@pytest.fixture()
def memory_manager(settings: Settings) -> MemoryManager:
    return MemoryManager(settings)


# ---------------------------------------------------------------------------
# Task fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_task() -> AgentTask:
    return AgentTask(
        task_id="test-task-001",
        description="Research the latest trends in AI agent architectures.",
        context={"source": "test"},
        constraints=["Be concise"],
    )


@pytest.fixture()
def sample_coding_task() -> AgentTask:
    return AgentTask(
        task_id="test-task-002",
        description="Write a Python function that implements binary search.",
        context={"language": "python"},
    )


@pytest.fixture()
def sample_analysis_task() -> AgentTask:
    return AgentTask(
        task_id="test-task-003",
        description="Analyze the following sales data and identify trends.",
        context={"data": [100, 120, 115, 140, 135, 160]},
    )


@pytest.fixture()
def sample_writing_task() -> AgentTask:
    return AgentTask(
        task_id="test-task-004",
        description="Write a blog post about the future of multi-agent AI systems.",
        constraints=["Professional tone", "1000 words"],
    )
