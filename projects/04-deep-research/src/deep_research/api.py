"""FastAPI application -- REST + SSE endpoints for Deep Research.

Endpoints
---------
POST /api/v1/research           Start a deep research task (async, returns task ID).
POST /api/v1/research/stream    Start research and stream progress via SSE.
GET  /api/v1/research/{task_id} Retrieve status / results of a research task.
POST /api/v1/reason             Single reasoning query with CoT.
POST /api/v1/compare-reasoning  Compare multiple reasoning strategies side-by-side.
GET  /health                    Liveness / readiness probe.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from dataclasses import asdict
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from deep_research.config import Settings, get_settings
from deep_research.reasoning import ReasoningEngine, ReasoningResult, ReasoningStrategy
from deep_research.researcher import DeepResearcher, ResearchStatus, ResearchTask

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# In-memory task store (production would use Redis / PostgreSQL)
# ---------------------------------------------------------------------------

_tasks: dict[str, ResearchTask] = {}


# ---------------------------------------------------------------------------
# Request / response schemas
# ---------------------------------------------------------------------------

class ResearchRequest(BaseModel):
    """Payload for starting a research task."""

    question: str = Field(..., min_length=5, max_length=2_000, description="The research question.")
    max_depth: int = Field(default=5, ge=1, le=10, description="Maximum research depth.")


class ResearchResponse(BaseModel):
    task_id: str
    status: str
    message: str


class ResearchStatusResponse(BaseModel):
    task_id: str
    status: str
    question: str
    depth: int = 0
    findings_count: int = 0
    report_markdown: str | None = None
    report_json: str | None = None
    error: str | None = None
    elapsed_seconds: float = 0.0


class ReasonRequest(BaseModel):
    """Payload for a single reasoning query."""

    query: str = Field(..., min_length=3, max_length=5_000)
    strategy: str = Field(default="chain_of_thought", description="Reasoning strategy.")
    use_extended_thinking: bool = Field(default=True)


class ReasonResponse(BaseModel):
    strategy: str
    query: str
    answer: str
    confidence: float
    steps: list[dict[str, Any]]
    thinking_content: str = ""
    duration_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0


class CompareRequest(BaseModel):
    """Payload for comparing reasoning strategies."""

    query: str = Field(..., min_length=3, max_length=5_000)
    strategies: list[str] = Field(
        default=["direct", "chain_of_thought", "tree_of_thought"],
    )


class CompareResponse(BaseModel):
    query: str
    results: dict[str, ReasonResponse]


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: float


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------

def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and return the FastAPI application."""
    settings = settings or get_settings()

    app = FastAPI(
        title="Deep Research API",
        description="Deep research with reasoning models, CoT, ToT, and inference-time scaling.",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store settings on app state for handler access
    app.state.settings = settings

    # --- Health ---------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["infra"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            version="0.1.0",
            timestamp=time.time(),
        )

    # --- Research -------------------------------------------------------------

    @app.post(
        "/api/v1/research",
        response_model=ResearchResponse,
        status_code=status.HTTP_202_ACCEPTED,
        tags=["research"],
    )
    async def start_research(req: ResearchRequest) -> ResearchResponse:
        """Kick off a deep research task in the background."""
        task_id = str(uuid.uuid4())
        task = ResearchTask(id=task_id, question=req.question, max_depth=req.max_depth)
        _tasks[task_id] = task

        researcher = DeepResearcher(settings=settings)
        asyncio.create_task(_run_research(researcher, task))

        return ResearchResponse(
            task_id=task_id,
            status=task.status.value,
            message="Research task started.",
        )

    @app.post("/api/v1/research/stream", tags=["research"])
    async def stream_research(req: ResearchRequest) -> EventSourceResponse:
        """Start research and stream progress events via SSE."""
        researcher = DeepResearcher(settings=settings)

        async def _event_generator():
            async for event in researcher.research_stream(
                req.question,
                max_depth=req.max_depth,
            ):
                yield {
                    "event": event.status.value,
                    "data": json.dumps({
                        "task_id": event.task_id,
                        "status": event.status.value,
                        "message": event.message,
                        "progress_pct": event.progress_pct,
                        "metadata": event.metadata,
                    }),
                }

        return EventSourceResponse(_event_generator())

    @app.get(
        "/api/v1/research/{task_id}",
        response_model=ResearchStatusResponse,
        tags=["research"],
    )
    async def get_research(task_id: str) -> ResearchStatusResponse:
        """Retrieve current status and results of a research task."""
        task = _tasks.get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail="Task not found.")

        elapsed = (task.completed_at or time.time()) - task.started_at
        return ResearchStatusResponse(
            task_id=task.id,
            status=task.status.value,
            question=task.question,
            depth=task.depth,
            findings_count=len(task.findings),
            report_markdown=task.report.to_markdown() if task.report else None,
            report_json=task.report.to_json() if task.report else None,
            error=task.error,
            elapsed_seconds=round(elapsed, 2),
        )

    # --- Reasoning ------------------------------------------------------------

    @app.post("/api/v1/reason", response_model=ReasonResponse, tags=["reasoning"])
    async def reason(req: ReasonRequest) -> ReasonResponse:
        """Execute a single reasoning query."""
        try:
            strategy = ReasoningStrategy(req.strategy)
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown strategy '{req.strategy}'. Choose from: {[s.value for s in ReasoningStrategy]}",
            )

        engine = ReasoningEngine(settings=settings)
        result = await engine.reason(
            req.query,
            strategy=strategy,
            use_extended_thinking=req.use_extended_thinking,
        )
        return _result_to_response(result)

    @app.post(
        "/api/v1/compare-reasoning",
        response_model=CompareResponse,
        tags=["reasoning"],
    )
    async def compare_reasoning(req: CompareRequest) -> CompareResponse:
        """Compare multiple reasoning strategies on the same query."""
        strategies: list[ReasoningStrategy] = []
        for s in req.strategies:
            try:
                strategies.append(ReasoningStrategy(s))
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Unknown strategy '{s}'.",
                )

        engine = ReasoningEngine(settings=settings)
        comparison = await engine.compare_strategies(req.query, strategies)
        return CompareResponse(
            query=req.query,
            results={name: _result_to_response(r) for name, r in comparison.items()},
        )

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _run_research(researcher: DeepResearcher, task: ResearchTask) -> None:
    """Background coroutine that drives a research task to completion."""
    try:
        completed = await researcher.research(task.question, max_depth=task.max_depth)
        # Copy results into the stored task reference
        task.plan = completed.plan
        task.findings = completed.findings
        task.search_results = completed.search_results
        task.reasoning_results = completed.reasoning_results
        task.report = completed.report
        task.depth = completed.depth
        task.status = ResearchStatus.COMPLETED
        task.completed_at = time.time()
    except Exception as exc:
        task.status = ResearchStatus.FAILED
        task.error = str(exc)
        task.completed_at = time.time()
        logger.error("background_research.failed", task_id=task.id, error=str(exc))


def _result_to_response(result: ReasoningResult) -> ReasonResponse:
    """Convert a ``ReasoningResult`` dataclass into a Pydantic response."""
    return ReasonResponse(
        strategy=result.strategy.value,
        query=result.query,
        answer=result.answer,
        confidence=result.confidence,
        steps=[
            {
                "step_number": s.step_number,
                "description": s.description,
                "content": s.content,
                "confidence": s.confidence,
                "duration_ms": s.duration_ms,
            }
            for s in result.steps
        ],
        thinking_content=result.thinking_content,
        duration_ms=result.total_duration_ms,
        input_tokens=result.total_input_tokens,
        output_tokens=result.total_output_tokens,
    )
