"""FastAPI application for the Capstone Multi-Agent Platform.

Exposes REST endpoints for task submission, agent management, streaming
execution, and memory inspection.  The API follows RESTful conventions with
``/api/v1`` versioned routes.

Endpoints
---------
* ``POST /api/v1/tasks`` -- Submit a complex task for orchestrated execution.
* ``POST /api/v1/tasks/stream`` -- Submit a task with SSE streaming output.
* ``GET  /api/v1/tasks/{task_id}`` -- Retrieve task status and results.
* ``GET  /api/v1/agents`` -- List all registered agents and their capabilities.
* ``POST /api/v1/agents/{agent_name}/execute`` -- Execute a specific agent directly.
* ``GET  /api/v1/memory/{session_id}`` -- Inspect session memory.
* ``GET  /health`` -- Health check.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from capstone.agents.base import AgentResult, AgentTask, TaskStatus
from capstone.config import Settings, get_settings
from capstone.memory import MemoryManager
from capstone.orchestrator import MultiAgentOrchestrator, OrchestratorState, create_orchestrator

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class TaskRequest(BaseModel):
    """Request body for submitting a task to the orchestrator."""

    request: str = Field(description="Natural-language description of the task.")
    session_id: str = Field(
        default_factory=lambda: uuid.uuid4().hex[:12],
        description="Session ID for memory continuity.",
    )
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional additional context for the task.",
    )


class TaskResponse(BaseModel):
    """Response body for a completed or in-progress task."""

    task_id: str
    status: str
    plan: str = ""
    final_output: str = ""
    agent_results: list[dict[str, Any]] = Field(default_factory=list)
    session_id: str = ""
    error: str | None = None


class AgentExecuteRequest(BaseModel):
    """Request body for direct agent execution."""

    description: str = Field(description="Task description for the agent.")
    context: dict[str, Any] = Field(default_factory=dict)
    constraints: list[str] = Field(default_factory=list)


class AgentInfo(BaseModel):
    """Public metadata about a registered agent."""

    name: str
    description: str
    capabilities: list[str]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str = "healthy"
    service: str = "capstone-multiagent"
    version: str = "0.1.0"
    agents: int = 0


# ---------------------------------------------------------------------------
# In-memory task store (production: use Redis or a database)
# ---------------------------------------------------------------------------


class TaskStore:
    """Simple in-memory store for task results, keyed by task_id.

    In production this would be backed by Redis or PostgreSQL for persistence
    and horizontal scaling.  The in-memory implementation keeps the project
    self-contained for local development.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, OrchestratorState] = {}

    def save(self, state: OrchestratorState) -> None:
        self._tasks[state.task_id] = state

    def get(self, task_id: str) -> OrchestratorState | None:
        return self._tasks.get(task_id)


# ---------------------------------------------------------------------------
# Application lifecycle
# ---------------------------------------------------------------------------

_orchestrator: MultiAgentOrchestrator | None = None
_memory: MemoryManager | None = None
_task_store: TaskStore | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Initialize and tear down shared resources."""
    global _orchestrator, _memory, _task_store
    settings = get_settings()

    # Memory
    _memory = MemoryManager(settings)
    await _memory.initialize()

    # Orchestrator
    _orchestrator = create_orchestrator(settings)

    # Task store
    _task_store = TaskStore()

    logger.info(
        "application_started",
        service=settings.app_name,
        environment=settings.environment,
        agents=[a["name"] for a in _orchestrator.list_agents()],
    )

    yield

    # Cleanup
    if _memory:
        await _memory.close()
    logger.info("application_shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Construct and return the FastAPI application."""
    settings = settings or get_settings()

    app = FastAPI(
        title="Capstone Multi-Agent AI Platform",
        description=(
            "Orchestrates specialized AI agents (researcher, coder, analyst, writer) "
            "to solve complex, multi-faceted tasks through decomposition, parallel "
            "execution, and intelligent aggregation."
        ),
        version=settings.app_version,
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(_build_router())

    return app


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

from fastapi import APIRouter

_router = APIRouter()


def _build_router() -> APIRouter:
    return _router


def _get_orchestrator() -> MultiAgentOrchestrator:
    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="Orchestrator not initialized.")
    return _orchestrator


def _get_memory() -> MemoryManager:
    if _memory is None:
        raise HTTPException(status_code=503, detail="Memory manager not initialized.")
    return _memory


def _get_task_store() -> TaskStore:
    if _task_store is None:
        raise HTTPException(status_code=503, detail="Task store not initialized.")
    return _task_store


# -- Health ----------------------------------------------------------------


@_router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check() -> HealthResponse:
    """Return service health status."""
    orchestrator = _get_orchestrator()
    return HealthResponse(
        agents=len(orchestrator.list_agents()),
    )


# -- Tasks -----------------------------------------------------------------


@_router.post("/api/v1/tasks", response_model=TaskResponse, tags=["tasks"])
async def submit_task(body: TaskRequest) -> TaskResponse:
    """Submit a complex task for multi-agent orchestrated execution.

    The orchestrator will decompose the task, assign subtasks to specialist
    agents, execute them (in parallel where possible), and aggregate the
    results into a coherent final output.
    """
    orchestrator = _get_orchestrator()
    memory = _get_memory()
    store = _get_task_store()

    # Record in conversation memory
    memory.add_message(body.session_id, "user", body.request)

    logger.info("task_submitted", session_id=body.session_id, request_length=len(body.request))

    try:
        state = await orchestrator.run(
            request=body.request,
            session_id=body.session_id,
        )
    except Exception as exc:
        logger.exception("task_execution_failed", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Task execution failed: {exc}") from exc

    # Persist state
    store.save(state)

    # Record assistant response
    if state.final_output:
        memory.add_message(body.session_id, "assistant", state.final_output)

    return _state_to_response(state)


@_router.post("/api/v1/tasks/stream", tags=["tasks"])
async def stream_task(body: TaskRequest) -> EventSourceResponse:
    """Submit a task with Server-Sent Events streaming for real-time updates.

    Events emitted:
    * ``plan`` -- The supervisor's execution plan.
    * ``agent_start`` -- An agent has begun processing a subtask.
    * ``agent_result`` -- An agent has completed its subtask.
    * ``final`` -- The aggregated final output.
    * ``error`` -- An error occurred during execution.
    """
    orchestrator = _get_orchestrator()
    memory = _get_memory()
    store = _get_task_store()

    memory.add_message(body.session_id, "user", body.request)

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        try:
            # We run the full pipeline and stream intermediate events.
            # For a truly streaming implementation, each graph node would
            # emit events via a channel; here we simulate by running the
            # pipeline and yielding key state transitions.
            yield {
                "event": "status",
                "data": json.dumps({"status": "decomposing", "message": "Analysing task..."}),
            }

            state = await orchestrator.run(
                request=body.request,
                session_id=body.session_id,
            )
            store.save(state)

            # Emit plan
            yield {
                "event": "plan",
                "data": json.dumps({"plan": state.plan}),
            }

            # Emit individual agent results
            for result in state.agent_results:
                yield {
                    "event": "agent_result",
                    "data": json.dumps({
                        "agent": result.agent_name,
                        "status": result.status,
                        "output_preview": result.output[:500],
                        "confidence": result.confidence,
                    }),
                }

            # Final output
            yield {
                "event": "final",
                "data": json.dumps({
                    "task_id": state.task_id,
                    "status": state.status,
                    "output": state.final_output,
                }),
            }

            if state.final_output:
                memory.add_message(body.session_id, "assistant", state.final_output)

        except Exception as exc:
            logger.exception("stream_error", error=str(exc))
            yield {
                "event": "error",
                "data": json.dumps({"error": str(exc)}),
            }

    return EventSourceResponse(event_generator())


@_router.get("/api/v1/tasks/{task_id}", response_model=TaskResponse, tags=["tasks"])
async def get_task(task_id: str) -> TaskResponse:
    """Retrieve the status and results of a previously submitted task."""
    store = _get_task_store()
    state = store.get(task_id)
    if state is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found.")
    return _state_to_response(state)


# -- Agents ----------------------------------------------------------------


@_router.get("/api/v1/agents", response_model=list[AgentInfo], tags=["agents"])
async def list_agents() -> list[AgentInfo]:
    """List all registered agents and their capabilities."""
    orchestrator = _get_orchestrator()
    return [AgentInfo(**a) for a in orchestrator.list_agents()]


@_router.post(
    "/api/v1/agents/{agent_name}/execute",
    response_model=dict[str, Any],
    tags=["agents"],
)
async def execute_agent(agent_name: str, body: AgentExecuteRequest) -> dict[str, Any]:
    """Execute a specific agent directly, bypassing the orchestrator.

    Useful for targeted tasks where the caller already knows which specialist
    to invoke.
    """
    orchestrator = _get_orchestrator()
    agents = {a["name"]: a for a in orchestrator.list_agents()}

    if agent_name not in agents:
        available = list(agents.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Agent '{agent_name}' not found. Available: {available}",
        )

    agent = orchestrator._agents[agent_name]
    task = AgentTask(
        description=body.description,
        context=body.context,
        constraints=body.constraints,
        preferred_agent=agent_name,
    )

    result = await agent._safe_execute(task)

    return {
        "task_id": result.task_id,
        "agent_name": result.agent_name,
        "status": result.status,
        "output": result.output,
        "structured_data": result.structured_data,
        "confidence": result.confidence,
        "duration_seconds": result.duration_seconds,
        "error": result.error,
    }


# -- Memory ----------------------------------------------------------------


@_router.get("/api/v1/memory/{session_id}", tags=["memory"])
async def get_session_memory(session_id: str) -> dict[str, Any]:
    """Inspect the memory state for a given session.

    Returns both short-term conversation context and long-term persistent
    memories associated with the session.
    """
    memory = _get_memory()
    return await memory.get_session_summary(session_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _state_to_response(state: OrchestratorState) -> TaskResponse:
    """Convert an orchestrator state to an API response."""
    return TaskResponse(
        task_id=state.task_id,
        status=state.status,
        plan=state.plan,
        final_output=state.final_output,
        agent_results=[
            {
                "agent_name": r.agent_name,
                "status": r.status,
                "output_preview": r.output[:500] if r.output else "",
                "confidence": r.confidence,
                "duration_seconds": r.duration_seconds,
                "error": r.error,
            }
            for r in state.agent_results
        ],
        session_id=state.session_id,
        error=state.error,
    )
