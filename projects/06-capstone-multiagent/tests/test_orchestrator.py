"""Tests for the multi-agent orchestrator.

These tests verify the orchestration logic, agent routing, state management,
and error handling without making any real LLM API calls.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from capstone.agents.base import AgentResult, AgentTask, BaseAgent, TaskStatus
from capstone.config import Settings
from capstone.orchestrator import (
    MultiAgentOrchestrator,
    OrchestratorState,
    create_orchestrator,
)


# ---------------------------------------------------------------------------
# Agent registration and listing
# ---------------------------------------------------------------------------


class TestAgentRegistration:
    """Tests for registering agents with the orchestrator."""

    def test_register_agent(self, settings: Settings, stub_agent):
        """Agents can be registered with the orchestrator."""
        orchestrator = MultiAgentOrchestrator(settings)
        orchestrator.register_agent(stub_agent)

        agents = orchestrator.list_agents()
        assert len(agents) == 1
        assert agents[0]["name"] == "stub"

    def test_register_multiple_agents(self, orchestrator_with_stubs):
        """Multiple agents can be registered and listed."""
        agents = orchestrator_with_stubs.list_agents()
        assert len(agents) == 4
        names = {a["name"] for a in agents}
        assert names == {"researcher", "coder", "analyst", "writer"}

    def test_list_agents_returns_metadata(self, orchestrator_with_stubs):
        """Listed agents include name, description, and capabilities."""
        agents = orchestrator_with_stubs.list_agents()
        for agent_info in agents:
            assert "name" in agent_info
            assert "description" in agent_info
            assert "capabilities" in agent_info
            assert isinstance(agent_info["capabilities"], list)


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------


class TestOrchestratorState:
    """Tests for the orchestrator state model."""

    def test_default_state(self):
        """Default state has sensible initial values."""
        state = OrchestratorState(user_request="Test request")
        assert state.user_request == "Test request"
        assert state.status == "pending"
        assert state.current_iteration == 0
        assert state.subtasks == []
        assert state.agent_results == []
        assert state.final_output == ""

    def test_state_with_all_fields(self):
        """State can be constructed with all fields populated."""
        state = OrchestratorState(
            task_id="custom-id",
            user_request="Complex request",
            session_id="session-001",
            plan="Step 1, Step 2",
            max_iterations=5,
        )
        assert state.task_id == "custom-id"
        assert state.session_id == "session-001"
        assert state.max_iterations == 5


# ---------------------------------------------------------------------------
# Agent task and result models
# ---------------------------------------------------------------------------


class TestAgentModels:
    """Tests for AgentTask and AgentResult data models."""

    def test_task_defaults(self):
        """AgentTask has sensible defaults."""
        task = AgentTask(description="Test task")
        assert task.task_id  # auto-generated
        assert task.status == TaskStatus.PENDING
        assert task.attempt == 0
        assert task.max_retries == 2

    def test_task_with_context(self, sample_task):
        """Tasks can carry structured context."""
        assert sample_task.context == {"source": "test"}
        assert sample_task.constraints == ["Be concise"]

    def test_result_defaults(self):
        """AgentResult has sensible defaults."""
        result = AgentResult(
            task_id="t1",
            agent_name="test",
            status=TaskStatus.COMPLETED,
            output="Done.",
        )
        assert result.confidence == 1.0
        assert result.duration_seconds == 0.0
        assert result.error is None
        assert result.structured_data == {}

    def test_result_with_structured_data(self):
        """Results can carry structured data alongside text output."""
        result = AgentResult(
            task_id="t1",
            agent_name="analyst",
            status=TaskStatus.COMPLETED,
            output="Analysis complete.",
            structured_data={"metrics": [1, 2, 3]},
            confidence=0.85,
        )
        assert result.structured_data["metrics"] == [1, 2, 3]
        assert result.confidence == 0.85


# ---------------------------------------------------------------------------
# Agent confidence scoring
# ---------------------------------------------------------------------------


class TestAgentConfidence:
    """Tests for agent can_handle scoring."""

    @pytest.mark.asyncio
    async def test_researcher_high_confidence_for_research(self, researcher_agent):
        """Researcher agent scores high for research-related tasks."""
        task = AgentTask(description="Research the history of artificial intelligence")
        score = await researcher_agent.can_handle(task)
        assert score > 0.2

    @pytest.mark.asyncio
    async def test_coder_high_confidence_for_code(self, coder_agent):
        """Coder agent scores high for coding tasks."""
        task = AgentTask(description="Write a Python function to sort a list")
        score = await coder_agent.can_handle(task)
        assert score > 0.2

    @pytest.mark.asyncio
    async def test_analyst_high_confidence_for_data(self, analyst_agent):
        """Analyst agent scores high for data analysis tasks."""
        task = AgentTask(description="Analyze this data and find trends in statistics")
        score = await analyst_agent.can_handle(task)
        assert score > 0.2

    @pytest.mark.asyncio
    async def test_writer_high_confidence_for_writing(self, writer_agent):
        """Writer agent scores high for content creation tasks."""
        task = AgentTask(description="Write a blog post about machine learning")
        score = await writer_agent.can_handle(task)
        assert score > 0.2

    @pytest.mark.asyncio
    async def test_preferred_agent_hint_boosts_confidence(self, researcher_agent):
        """Setting preferred_agent boosts the agent's confidence score."""
        task = AgentTask(
            description="Do something generic",
            preferred_agent="researcher",
        )
        score = await researcher_agent.can_handle(task)
        assert score >= 0.85

    @pytest.mark.asyncio
    async def test_confidence_bounded_to_1(self, coder_agent):
        """Confidence scores are capped at 1.0."""
        task = AgentTask(
            description="Write code to implement a function class debug fix error test",
            preferred_agent="coder",
        )
        score = await coder_agent.can_handle(task)
        assert score <= 1.0


# ---------------------------------------------------------------------------
# Orchestrator execution (with mocked LLM)
# ---------------------------------------------------------------------------


class TestOrchestratorExecution:
    """Tests for the full orchestration pipeline with mocked LLM calls."""

    @pytest.mark.asyncio
    async def test_run_produces_output(self, settings: Settings):
        """The orchestrator produces a final output from a mocked pipeline."""
        decomposition_response = json.dumps({
            "plan": "Research AI trends then summarize.",
            "subtasks": [
                {
                    "description": "Research AI trends",
                    "preferred_agent": "researcher",
                    "constraints": [],
                    "priority": 1,
                }
            ],
        })

        refinement_response = json.dumps({
            "is_complete": True,
            "reasoning": "All subtasks completed successfully.",
            "additional_subtasks": [],
        })

        mock_responses = iter([
            MagicMock(content=decomposition_response),
            MagicMock(content=refinement_response),
            MagicMock(content="Final aggregated output about AI trends."),
        ])

        with patch("langchain_anthropic.ChatAnthropic.ainvoke", new_callable=AsyncMock) as mock_llm:
            mock_llm.side_effect = lambda *args, **kwargs: next(mock_responses)

            orchestrator = MultiAgentOrchestrator(settings)

            # Register stub agents
            from tests.conftest import StubAgent

            for name in ["researcher", "coder", "analyst", "writer"]:
                agent = StubAgent(settings, output=f"Result from {name}.")
                agent.name = name
                orchestrator.register_agent(agent)

            state = await orchestrator.run("What are the latest AI trends?")

        assert state.status in ("completed", "partial")
        assert state.final_output != ""
        assert state.task_id != ""

    @pytest.mark.asyncio
    async def test_run_handles_empty_subtasks(self, settings: Settings):
        """Orchestrator handles the case where no subtasks are generated."""
        empty_response = json.dumps({
            "plan": "Direct answer.",
            "subtasks": [],
        })

        with patch("langchain_anthropic.ChatAnthropic.ainvoke", new_callable=AsyncMock) as mock_llm:
            mock_llm.return_value = MagicMock(content=empty_response)

            orchestrator = MultiAgentOrchestrator(settings)

            from tests.conftest import StubAgent

            agent = StubAgent(settings)
            agent.name = "researcher"
            orchestrator.register_agent(agent)

            state = await orchestrator.run("Hello")

        assert state.status in ("completed", "partial")

    @pytest.mark.asyncio
    async def test_find_best_agent(self, orchestrator_with_stubs):
        """The orchestrator can find the best agent for a task description."""
        best = await orchestrator_with_stubs._find_best_agent("Research some topic")
        assert best in {"researcher", "coder", "analyst", "writer"}

    @pytest.mark.asyncio
    async def test_find_best_agent_no_agents(self, settings: Settings):
        """Returns 'researcher' as fallback when no agents are registered."""
        orchestrator = MultiAgentOrchestrator(settings)
        best = await orchestrator._find_best_agent("anything")
        assert best == "researcher"


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


class TestCreateOrchestrator:
    """Tests for the create_orchestrator factory function."""

    def test_creates_orchestrator_with_all_agents(self, settings: Settings):
        """Factory creates an orchestrator with all four specialist agents."""
        orchestrator = create_orchestrator(settings)
        agents = orchestrator.list_agents()
        assert len(agents) == 4
        names = {a["name"] for a in agents}
        assert "researcher" in names
        assert "coder" in names
        assert "analyst" in names
        assert "writer" in names


# ---------------------------------------------------------------------------
# JSON parsing
# ---------------------------------------------------------------------------


class TestJSONParsing:
    """Tests for the supervisor's JSON parsing helper."""

    def test_parse_clean_json(self):
        """Clean JSON is parsed correctly."""
        raw = '{"plan": "Do something", "subtasks": []}'
        result = MultiAgentOrchestrator._parse_json_response(raw)
        assert result["plan"] == "Do something"

    def test_parse_json_with_markdown_fences(self):
        """JSON wrapped in markdown code fences is handled."""
        raw = '```json\n{"plan": "Fenced", "subtasks": []}\n```'
        result = MultiAgentOrchestrator._parse_json_response(raw)
        assert result["plan"] == "Fenced"

    def test_parse_invalid_json_returns_fallback(self):
        """Invalid JSON returns a fallback dict with the raw text as plan."""
        raw = "This is not JSON at all."
        result = MultiAgentOrchestrator._parse_json_response(raw)
        assert "plan" in result
        assert result["is_complete"] is True  # safe fallback


# ---------------------------------------------------------------------------
# Result formatting
# ---------------------------------------------------------------------------


class TestResultFormatting:
    """Tests for the result formatting helper."""

    def test_format_results_for_prompt(self):
        """Agent results are formatted into a readable string."""
        results = [
            AgentResult(
                task_id="t1",
                agent_name="researcher",
                status=TaskStatus.COMPLETED,
                output="Research findings here.",
                confidence=0.9,
            ),
            AgentResult(
                task_id="t2",
                agent_name="coder",
                status=TaskStatus.FAILED,
                output="",
                confidence=0.0,
                error="API timeout",
            ),
        ]
        formatted = MultiAgentOrchestrator._format_results_for_prompt(results)
        assert "researcher" in formatted
        assert "Research findings here." in formatted
        assert "FAILED" in formatted
        assert "API timeout" in formatted
