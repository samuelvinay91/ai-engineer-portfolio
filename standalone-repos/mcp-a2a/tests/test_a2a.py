"""Tests for A2A protocol, AgentCard, discovery, and orchestrator."""

from __future__ import annotations

import asyncio

import pytest

from mcp_a2a.a2a.agent_card import (
    AgentCard,
    AgentCardRegistry,
    AuthConfig,
    Capability,
    Skill,
    build_default_agent_card,
)
from mcp_a2a.a2a.discovery import AgentDiscoveryService
from mcp_a2a.a2a.orchestrator import A2AOrchestrator, TaskDecomposer
from mcp_a2a.a2a.protocol import (
    A2AClient,
    A2AServer,
    DataPart,
    FilePart,
    Message,
    Task,
    TaskSendParams,
    TaskState,
    TextPart,
)


# ---------------------------------------------------------------------------
# AgentCard tests
# ---------------------------------------------------------------------------


class TestAgentCard:
    """Tests for AgentCard model and registry."""

    def test_build_default_card(self) -> None:
        card = build_default_agent_card("http://localhost:9000")
        assert card.name == "AI Portfolio Agent"
        assert card.url == "http://localhost:9000"
        assert len(card.capabilities) > 0
        assert len(card.skills) > 0
        assert "text" in card.supported_modalities

    def test_card_serialization(self) -> None:
        card = build_default_agent_card()
        data = card.model_dump()
        assert data["name"] == "AI Portfolio Agent"
        assert "capabilities" in data
        assert "skills" in data

    def test_card_deserialization(self) -> None:
        card = build_default_agent_card()
        data = card.model_dump()
        restored = AgentCard.model_validate(data)
        assert restored.name == card.name
        assert len(restored.capabilities) == len(card.capabilities)

    def test_capability_model(self) -> None:
        cap = Capability(
            name="test",
            description="A test capability",
            input_schema={"type": "object"},
            output_schema={"type": "object"},
        )
        assert cap.name == "test"

    def test_skill_model(self) -> None:
        skill = Skill(
            id="test-skill",
            name="Test",
            description="A test skill",
            tags=["test", "demo"],
            examples=["Do something"],
        )
        assert skill.id == "test-skill"
        assert "test" in skill.tags

    def test_auth_config(self) -> None:
        auth = AuthConfig(type="oauth2", scopes=["read", "write"])
        assert auth.type == "oauth2"
        assert len(auth.scopes) == 2


class TestAgentCardRegistry:
    """Tests for AgentCard registry."""

    def test_register_and_list(self) -> None:
        registry = AgentCardRegistry()
        card = build_default_agent_card("http://agent1:8000")
        registry.register(card)
        agents = registry.list_agents()
        assert len(agents) == 1
        assert agents[0].name == card.name

    def test_unregister(self) -> None:
        registry = AgentCardRegistry()
        card = build_default_agent_card("http://agent1:8000")
        registry.register(card)
        removed = registry.unregister("http://agent1:8000")
        assert removed is True
        assert len(registry.list_agents()) == 0

    def test_unregister_nonexistent(self) -> None:
        registry = AgentCardRegistry()
        assert registry.unregister("http://nonexistent:8000") is False

    def test_find_by_capability(self) -> None:
        registry = AgentCardRegistry()
        card = build_default_agent_card("http://agent1:8000")
        registry.register(card)
        matches = registry.find_by_capability("math_computation")
        assert len(matches) == 1
        assert matches[0].url == "http://agent1:8000"

    def test_find_by_skill_tag(self) -> None:
        registry = AgentCardRegistry()
        card = build_default_agent_card("http://agent1:8000")
        registry.register(card)
        matches = registry.find_by_skill_tag("search")
        assert len(matches) == 1

    def test_find_by_modality(self) -> None:
        registry = AgentCardRegistry()
        card = build_default_agent_card("http://agent1:8000")
        registry.register(card)
        matches = registry.find_by_modality("text")
        assert len(matches) == 1
        matches_audio = registry.find_by_modality("audio")
        assert len(matches_audio) == 0

    def test_get_card(self) -> None:
        registry = AgentCardRegistry()
        card = build_default_agent_card("http://agent1:8000")
        registry.register(card)
        fetched = registry.get_card("http://agent1:8000")
        assert fetched is not None
        assert fetched.name == card.name
        assert registry.get_card("http://unknown") is None


# ---------------------------------------------------------------------------
# Protocol tests
# ---------------------------------------------------------------------------


class TestA2AProtocol:
    """Tests for A2A protocol models."""

    def test_text_part(self) -> None:
        part = TextPart(text="Hello world")
        assert part.type == "text"
        assert part.text == "Hello world"

    def test_file_part(self) -> None:
        part = FilePart(file_name="test.csv", mime_type="text/csv", data="base64data")
        assert part.type == "file"
        assert part.file_name == "test.csv"

    def test_data_part(self) -> None:
        part = DataPart(data={"key": "value"})
        assert part.type == "data"
        assert part.data["key"] == "value"

    def test_message_creation(self) -> None:
        msg = Message(role="user", parts=[TextPart(text="Hello")])
        assert msg.role == "user"
        assert len(msg.parts) == 1
        assert msg.timestamp > 0

    def test_task_default_state(self) -> None:
        task = Task()
        assert task.state == TaskState.SUBMITTED
        assert task.id != ""
        assert len(task.messages) == 0

    def test_task_state_values(self) -> None:
        assert TaskState.SUBMITTED.value == "submitted"
        assert TaskState.WORKING.value == "working"
        assert TaskState.INPUT_REQUIRED.value == "input-required"
        assert TaskState.COMPLETED.value == "completed"
        assert TaskState.FAILED.value == "failed"
        assert TaskState.CANCELED.value == "canceled"

    def test_task_send_params(self) -> None:
        params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text="Do something")])
        )
        assert params.id != ""
        assert params.message.role == "user"


class TestA2AServer:
    """Tests for A2A server-side protocol handling."""

    @pytest.mark.asyncio
    async def test_submit_task(self) -> None:
        server = A2AServer()
        params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text="Analyze this data")])
        )
        task = await server.handle_task_send(params)
        assert task.id == params.id
        assert task.state in (TaskState.WORKING, TaskState.COMPLETED)

    @pytest.mark.asyncio
    async def test_get_task(self) -> None:
        server = A2AServer()
        params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text="Test")])
        )
        created = await server.handle_task_send(params)
        fetched = await server.handle_task_get(created.id)
        assert fetched is not None
        assert fetched.id == created.id

    @pytest.mark.asyncio
    async def test_get_nonexistent_task(self) -> None:
        server = A2AServer()
        result = await server.handle_task_get("nonexistent-id")
        assert result is None

    @pytest.mark.asyncio
    async def test_task_completes(self) -> None:
        server = A2AServer()
        params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text="Process this request")])
        )
        task = await server.handle_task_send(params)
        # Wait for async processing
        await asyncio.sleep(0.3)
        updated = await server.handle_task_get(task.id)
        assert updated is not None
        assert updated.state == TaskState.COMPLETED
        # Should have at least 2 messages (user + agent response)
        assert len(updated.messages) >= 2

    @pytest.mark.asyncio
    async def test_task_produces_artifacts(self) -> None:
        server = A2AServer()
        params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text="Analyze something")])
        )
        task = await server.handle_task_send(params)
        await asyncio.sleep(0.3)
        updated = await server.handle_task_get(task.id)
        assert updated is not None
        assert len(updated.artifacts) > 0

    @pytest.mark.asyncio
    async def test_cancel_task(self) -> None:
        server = A2AServer()
        params = TaskSendParams(
            message=Message(role="user", parts=[TextPart(text="Cancel me")])
        )
        task = await server.handle_task_send(params)
        cancelled = await server.handle_task_cancel(task.id)
        assert cancelled is not None
        # Task may already be completed due to fast processing
        assert cancelled.state in (TaskState.CANCELED, TaskState.COMPLETED)

    @pytest.mark.asyncio
    async def test_list_tasks(self) -> None:
        server = A2AServer()
        for i in range(3):
            params = TaskSendParams(
                message=Message(role="user", parts=[TextPart(text=f"Task {i}")])
            )
            await server.handle_task_send(params)
        await asyncio.sleep(0.5)
        all_tasks = server.list_tasks()
        assert len(all_tasks) == 3


# ---------------------------------------------------------------------------
# Discovery tests
# ---------------------------------------------------------------------------


class TestDiscoveryService:
    """Tests for agent discovery."""

    def test_find_agents_for_task_keyword_matching(self) -> None:
        registry = AgentCardRegistry()
        card = build_default_agent_card("http://agent:8000")
        registry.register(card)
        svc = AgentDiscoveryService(registry=registry)

        # Should match "math" via computation skill tag, "search" via web_search cap/skill
        matches = svc.find_agents_for_task("I need math computation and search")
        assert len(matches) >= 1

    def test_find_agents_for_task_no_match(self) -> None:
        registry = AgentCardRegistry()
        card = build_default_agent_card("http://agent:8000")
        registry.register(card)
        svc = AgentDiscoveryService(registry=registry)

        matches = svc.find_agents_for_task("xyzzy foobar baz")
        assert len(matches) == 0

    def test_find_by_capability(self) -> None:
        registry = AgentCardRegistry()
        card = build_default_agent_card("http://agent:8000")
        registry.register(card)
        svc = AgentDiscoveryService(registry=registry)
        matches = svc.find_by_capability("weather_lookup")
        assert len(matches) == 1

    def test_get_status(self) -> None:
        registry = AgentCardRegistry()
        card = build_default_agent_card("http://agent:8000")
        registry.register(card)
        svc = AgentDiscoveryService(registry=registry)
        status = svc.get_status()
        assert status["total_agents"] == 1
        assert len(status["agents"]) == 1


# ---------------------------------------------------------------------------
# Orchestrator tests
# ---------------------------------------------------------------------------


class TestTaskDecomposer:
    """Tests for task decomposition."""

    def test_decompose_math_task(self) -> None:
        decomposer = TaskDecomposer()
        subtasks = decomposer.decompose("Calculate the sum of 10 and 20")
        assert len(subtasks) >= 1
        assert any("math" in s.description.lower() or "computation" in s.description.lower() for s in subtasks)

    def test_decompose_search_task(self) -> None:
        decomposer = TaskDecomposer()
        subtasks = decomposer.decompose("Search the web for AI news")
        assert len(subtasks) >= 1

    def test_decompose_complex_task(self) -> None:
        decomposer = TaskDecomposer()
        subtasks = decomposer.decompose(
            "Search for weather data, calculate averages, and analyze the results"
        )
        # Should match multiple patterns
        assert len(subtasks) >= 2

    def test_decompose_unrecognized_falls_back(self) -> None:
        decomposer = TaskDecomposer()
        subtasks = decomposer.decompose("xyzzy foobar baz")
        assert len(subtasks) == 1
        assert "Process" in subtasks[0].description


class TestOrchestrator:
    """Tests for the A2A orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrate_assigns_agents(self) -> None:
        registry = AgentCardRegistry()
        card = build_default_agent_card("http://agent:8000")
        registry.register(card)
        discovery = AgentDiscoveryService(registry=registry)
        orch = A2AOrchestrator(discovery=discovery)

        plan = await orch.orchestrate("Calculate something with math")
        assert plan.state == TaskState.COMPLETED
        assert len(plan.subtasks) >= 1
        assert plan.aggregated_result is not None

    @pytest.mark.asyncio
    async def test_orchestrate_no_agents(self) -> None:
        discovery = AgentDiscoveryService(registry=AgentCardRegistry())
        orch = A2AOrchestrator(discovery=discovery)
        plan = await orch.orchestrate("xyzzy foobar")
        assert plan.state == TaskState.COMPLETED
        # Subtask should fail because no agent was found
        assert any(s.error is not None for s in plan.subtasks)

    def test_plan_to_dict(self) -> None:
        discovery = AgentDiscoveryService()
        orch = A2AOrchestrator(discovery=discovery)
        from mcp_a2a.a2a.orchestrator import OrchestrationPlan
        plan = OrchestrationPlan(original_request="test")
        result = orch.plan_to_dict(plan)
        assert "task_id" in result
        assert result["original_request"] == "test"
