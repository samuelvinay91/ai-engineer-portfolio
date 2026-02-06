"""A2A Orchestrator — decomposes complex tasks and routes them to capable agents.

The orchestrator acts as a meta-agent that:
1. Receives a high-level task from a user or upstream agent.
2. Discovers capable downstream agents via the discovery service.
3. Decomposes the task into subtasks.
4. Routes each subtask to the most suitable agent.
5. Collects and aggregates results.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from mcp_a2a.a2a.agent_card import AgentCard
from mcp_a2a.a2a.discovery import AgentDiscoveryService
from mcp_a2a.a2a.protocol import (
    A2AClient,
    Message,
    Task,
    TaskState,
    TextPart,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SubtaskState(str, Enum):
    PENDING = "pending"
    DISPATCHED = "dispatched"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Subtask:
    """A decomposed piece of the original task."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    description: str = ""
    assigned_agent_url: str | None = None
    assigned_agent_name: str | None = None
    state: SubtaskState = SubtaskState.PENDING
    result: dict[str, Any] | None = None
    error: str | None = None
    started_at: float | None = None
    completed_at: float | None = None


@dataclass
class OrchestrationPlan:
    """Execution plan for a complex task."""

    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_request: str = ""
    subtasks: list[Subtask] = field(default_factory=list)
    state: TaskState = TaskState.SUBMITTED
    created_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    aggregated_result: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Task decomposer (rule-based; production version would use an LLM)
# ---------------------------------------------------------------------------


class TaskDecomposer:
    """Decomposes a high-level task into subtasks using keyword heuristics.

    In production this would delegate to an LLM for semantic decomposition.
    """

    # Keyword -> subtask description mapping
    _PATTERNS: list[tuple[list[str], str]] = [
        (["calculate", "compute", "math", "add", "subtract", "multiply"],
         "Perform the requested mathematical computation"),
        (["weather", "temperature", "forecast", "climate"],
         "Look up weather information for the specified location"),
        (["search", "find", "look up", "research", "web"],
         "Search the web for relevant information"),
        (["database", "query", "employees", "records", "data"],
         "Query the database for the requested records"),
        (["file", "read", "open", "contents"],
         "Read the specified file contents"),
        (["analyze", "analysis", "insights", "summarize", "summary"],
         "Analyze the provided content and produce insights"),
    ]

    def decompose(self, task_description: str) -> list[Subtask]:
        """Break a task description into subtasks based on keyword matching."""
        description_lower = task_description.lower()
        subtasks: list[Subtask] = []
        matched = False

        for keywords, subtask_desc in self._PATTERNS:
            if any(kw in description_lower for kw in keywords):
                subtasks.append(Subtask(description=subtask_desc))
                matched = True

        # If no specific pattern matched, create a generic subtask
        if not matched:
            subtasks.append(
                Subtask(description=f"Process the request: {task_description}")
            )

        return subtasks


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class A2AOrchestrator:
    """Orchestrates complex tasks across multiple A2A agents.

    Parameters
    ----------
    discovery:
        Agent discovery service for finding capable agents.
    client:
        A2A client for sending tasks to remote agents.
    decomposer:
        Task decomposer for breaking down complex requests.
    """

    def __init__(
        self,
        discovery: AgentDiscoveryService,
        client: A2AClient | None = None,
        decomposer: TaskDecomposer | None = None,
    ) -> None:
        self._discovery = discovery
        self._client = client or A2AClient()
        self._decomposer = decomposer or TaskDecomposer()
        self._plans: dict[str, OrchestrationPlan] = {}

    # -- Public API ----------------------------------------------------------

    async def orchestrate(self, request: str) -> OrchestrationPlan:
        """Handle a complex task end-to-end.

        1. Decompose the request into subtasks.
        2. Find agents for each subtask.
        3. Dispatch subtasks concurrently.
        4. Aggregate results.
        """
        plan = OrchestrationPlan(original_request=request)
        plan.subtasks = self._decomposer.decompose(request)
        plan.state = TaskState.WORKING
        self._plans[plan.task_id] = plan

        logger.info(
            "orchestrator.plan_created",
            task_id=plan.task_id,
            subtask_count=len(plan.subtasks),
        )

        # Assign agents to subtasks
        await self._assign_agents(plan)

        # Dispatch all subtasks concurrently
        await self._dispatch_subtasks(plan)

        # Aggregate
        plan.aggregated_result = self._aggregate_results(plan)

        # Determine final state
        failed = [s for s in plan.subtasks if s.state == SubtaskState.FAILED]
        if failed:
            plan.state = TaskState.COMPLETED  # Partial success
        else:
            plan.state = TaskState.COMPLETED
        plan.completed_at = time.time()

        logger.info(
            "orchestrator.plan_completed",
            task_id=plan.task_id,
            state=plan.state.value,
            failed_subtasks=len(failed),
        )

        return plan

    def get_plan(self, task_id: str) -> OrchestrationPlan | None:
        """Retrieve an orchestration plan by ID."""
        return self._plans.get(task_id)

    def list_plans(self, limit: int = 20) -> list[OrchestrationPlan]:
        """List recent orchestration plans."""
        plans = sorted(
            self._plans.values(), key=lambda p: p.created_at, reverse=True
        )
        return plans[:limit]

    # -- Internal helpers ----------------------------------------------------

    async def _assign_agents(self, plan: OrchestrationPlan) -> None:
        """Match each subtask to the best available agent."""
        for subtask in plan.subtasks:
            candidates = self._discovery.find_agents_for_task(subtask.description)
            if candidates:
                best = candidates[0]
                subtask.assigned_agent_url = best.url
                subtask.assigned_agent_name = best.name
                logger.info(
                    "orchestrator.agent_assigned",
                    subtask_id=subtask.id,
                    agent=best.name,
                )
            else:
                logger.warning(
                    "orchestrator.no_agent_found",
                    subtask_id=subtask.id,
                    description=subtask.description,
                )

    async def _dispatch_subtasks(self, plan: OrchestrationPlan) -> None:
        """Dispatch all subtasks to their assigned agents concurrently."""
        tasks = []
        for subtask in plan.subtasks:
            if subtask.assigned_agent_url:
                tasks.append(self._dispatch_single(subtask))
            else:
                subtask.state = SubtaskState.FAILED
                subtask.error = "No agent available for this subtask"

        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _dispatch_single(self, subtask: Subtask) -> None:
        """Send a single subtask to its assigned agent."""
        assert subtask.assigned_agent_url is not None

        subtask.state = SubtaskState.DISPATCHED
        subtask.started_at = time.time()

        try:
            result = await self._client.send_task(
                agent_url=subtask.assigned_agent_url,
                message=subtask.description,
                metadata={"subtask_id": subtask.id},
            )

            if "error" in result:
                subtask.state = SubtaskState.FAILED
                subtask.error = result["error"]
            else:
                subtask.state = SubtaskState.COMPLETED
                subtask.result = result

            subtask.completed_at = time.time()

        except Exception as exc:
            subtask.state = SubtaskState.FAILED
            subtask.error = str(exc)
            subtask.completed_at = time.time()
            logger.error(
                "orchestrator.dispatch_error",
                subtask_id=subtask.id,
                error=str(exc),
            )

    def _aggregate_results(self, plan: OrchestrationPlan) -> dict[str, Any]:
        """Combine subtask results into a unified response."""
        completed = [s for s in plan.subtasks if s.state == SubtaskState.COMPLETED]
        failed = [s for s in plan.subtasks if s.state == SubtaskState.FAILED]

        subtask_summaries = []
        for s in plan.subtasks:
            summary: dict[str, Any] = {
                "id": s.id,
                "description": s.description,
                "state": s.state.value,
                "assigned_agent": s.assigned_agent_name,
            }
            if s.result is not None:
                summary["result"] = s.result
            if s.error is not None:
                summary["error"] = s.error
            if s.started_at and s.completed_at:
                summary["duration_ms"] = round((s.completed_at - s.started_at) * 1000, 2)
            subtask_summaries.append(summary)

        return {
            "original_request": plan.original_request,
            "total_subtasks": len(plan.subtasks),
            "completed": len(completed),
            "failed": len(failed),
            "subtasks": subtask_summaries,
        }

    # -- Serialization -------------------------------------------------------

    def plan_to_dict(self, plan: OrchestrationPlan) -> dict[str, Any]:
        """Serialize an orchestration plan to a dictionary."""
        return {
            "task_id": plan.task_id,
            "original_request": plan.original_request,
            "state": plan.state.value,
            "created_at": plan.created_at,
            "completed_at": plan.completed_at,
            "subtasks": [
                {
                    "id": s.id,
                    "description": s.description,
                    "state": s.state.value,
                    "assigned_agent_url": s.assigned_agent_url,
                    "assigned_agent_name": s.assigned_agent_name,
                    "result": s.result,
                    "error": s.error,
                }
                for s in plan.subtasks
            ],
            "aggregated_result": plan.aggregated_result,
        }
