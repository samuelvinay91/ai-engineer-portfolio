"""FastAPI application for the Ask-the-Web agent.

Endpoints
---------
- ``POST /api/v1/ask``        -- Ask a question (returns full JSON answer).
- ``POST /api/v1/ask/stream`` -- Ask a question with Server-Sent Events streaming.
- ``POST /api/v1/search``     -- Execute a raw web search (no synthesis).
- ``GET  /api/v1/history``    -- Retrieve recent search history.
- ``GET  /health``            -- Health check.
"""

from __future__ import annotations

import json
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from common.models import ErrorResponse, HealthResponse

from ask_the_web.agents.searcher import WebSearchAgent
from ask_the_web.agents.synthesizer import SynthesisAgent
from ask_the_web.config import Settings, get_settings
from ask_the_web.workflow import WorkflowState, compile_workflow, run_ask_pipeline

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# In-memory history (would use Redis/DB in production)
# ---------------------------------------------------------------------------
_history: deque[dict[str, Any]] = deque(maxlen=50)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    """Payload for the /ask endpoints."""

    query: str = Field(..., min_length=1, max_length=2000, description="The user's question.")
    max_sources: int = Field(default=8, ge=1, le=20, description="Max sources to include.")
    fact_check: bool = Field(default=True, description="Whether to run the fact-checker.")


class SearchRequest(BaseModel):
    """Payload for the /search endpoint."""

    query: str = Field(..., min_length=1, max_length=500)
    strategy: str = Field(default="general", pattern="^(general|news|deep)$")
    max_results: int = Field(default=8, ge=1, le=20)


class AskResponse(BaseModel):
    """Full answer response from the ask pipeline."""

    query: str
    answer: str
    citations: list[dict[str, Any]] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    fact_check: dict[str, Any] | None = None
    route_used: str = ""
    elapsed_ms: float = 0.0
    request_id: str = ""


class HistoryEntry(BaseModel):
    """A single history entry."""

    request_id: str
    query: str
    route_used: str
    elapsed_ms: float
    timestamp: float


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Application lifespan: initialise and tear down resources."""
    settings = get_settings()
    app.state.settings = settings
    app.state.compiled_workflow = compile_workflow(settings)
    app.state.searcher = WebSearchAgent(settings)
    app.state.synthesiser = SynthesisAgent(settings)
    logger.info("app_started", app=settings.app_name, version=settings.app_version)
    yield
    logger.info("app_stopped")


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Build and return the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title="Ask-the-Web Agent API",
        description=(
            "Perplexity-like search-and-synthesis agent demonstrating agentic "
            "patterns: routing, prompt chaining, and LangGraph workflows."
        ),
        version=settings.app_version,
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Health ----------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["infra"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            service=settings.app_name,
            version=settings.app_version,
        )

    # -- Ask (full) ------------------------------------------------------------

    @app.post("/api/v1/ask", response_model=AskResponse, tags=["ask"])
    async def ask(body: AskRequest, request: Request) -> AskResponse:
        """Run the full ask-the-web pipeline and return a cited answer."""
        request_id = str(uuid.uuid4())
        logger.info("ask_request", request_id=request_id, query=body.query[:120])

        try:
            result: WorkflowState = await run_ask_pipeline(
                query=body.query,
                settings=request.app.state.settings,
            )
        except Exception as exc:
            logger.exception("ask_pipeline_error", request_id=request_id)
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        # Build response
        fact_check_data: dict[str, Any] | None = None
        if result.get("fact_check_report"):
            report = result["fact_check_report"]
            fact_check_data = report.model_dump() if hasattr(report, "model_dump") else None

        response = AskResponse(
            query=body.query,
            answer=result.get("answer", ""),
            citations=result.get("citations", []),
            follow_up_questions=result.get("follow_up_questions", []),
            fact_check=fact_check_data,
            route_used=result.get("route_used", ""),
            elapsed_ms=result.get("elapsed_ms", 0.0),
            request_id=request_id,
        )

        # Store in history
        _history.appendleft(
            {
                "request_id": request_id,
                "query": body.query,
                "route_used": response.route_used,
                "elapsed_ms": response.elapsed_ms,
                "timestamp": time.time(),
            }
        )

        return response

    # -- Ask (streaming) -------------------------------------------------------

    @app.post("/api/v1/ask/stream", tags=["ask"])
    async def ask_stream(body: AskRequest, request: Request) -> EventSourceResponse:
        """Stream an answer via Server-Sent Events.

        Events emitted:

        * ``route``    -- routing decision (JSON)
        * ``sources``  -- search results metadata (JSON)
        * ``token``    -- individual answer tokens
        * ``done``     -- final metadata (JSON)
        * ``error``    -- on failure (JSON)
        """
        request_id = str(uuid.uuid4())
        settings: Settings = request.app.state.settings

        async def event_generator() -> AsyncIterator[dict[str, str]]:
            try:
                # 1. Route
                from ask_the_web.agents.router import QueryRouterAgent
                router = QueryRouterAgent(settings)
                decision = await router.route(body.query)
                yield {
                    "event": "route",
                    "data": json.dumps({
                        "route": decision.route.value,
                        "confidence": decision.confidence,
                        "reasoning": decision.reasoning,
                    }),
                }

                # 2. Search (skip for creative)
                search_results = []
                if decision.route != "creative":
                    searcher: WebSearchAgent = request.app.state.searcher
                    queries = decision.search_queries or [decision.reformulated_query]
                    search_resp = await searcher.search(queries)
                    search_results = search_resp.results

                    yield {
                        "event": "sources",
                        "data": json.dumps({
                            "count": len(search_results),
                            "sources": [
                                {"title": r.title, "url": r.url, "domain": r.source_domain}
                                for r in search_results[:10]
                            ],
                        }),
                    }

                # 3. Stream synthesis
                synthesiser: SynthesisAgent = request.app.state.synthesiser
                async for token in synthesiser.synthesise_stream(body.query, search_results):
                    yield {"event": "token", "data": token}

                # 4. Done
                yield {
                    "event": "done",
                    "data": json.dumps({
                        "request_id": request_id,
                        "route_used": decision.route.value,
                    }),
                }

            except Exception as exc:
                logger.exception("stream_error", request_id=request_id)
                yield {
                    "event": "error",
                    "data": json.dumps({"error": str(exc)}),
                }

        return EventSourceResponse(event_generator())

    # -- Raw search ------------------------------------------------------------

    @app.post("/api/v1/search", tags=["search"])
    async def search(body: SearchRequest, request: Request) -> dict[str, Any]:
        """Execute a raw web search without synthesis."""
        searcher: WebSearchAgent = request.app.state.searcher
        try:
            result = await searcher.search(
                [body.query], strategy=body.strategy
            )
            return result.model_dump()
        except Exception as exc:
            logger.exception("search_error")
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    # -- History ---------------------------------------------------------------

    @app.get("/api/v1/history", tags=["history"])
    async def history(limit: int = 20) -> list[dict[str, Any]]:
        """Return recent search history entries."""
        return list(_history)[:limit]

    return app
