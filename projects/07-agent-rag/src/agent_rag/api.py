"""FastAPI application for the Agent & RAG system.

Endpoints:
- POST /api/v1/ingest          - Ingest documents
- POST /api/v1/query           - Single-agent RAG query
- POST /api/v1/query/multi-agent   - Multi-agent RAG query
- POST /api/v1/query/hierarchical  - Hierarchical RAG query
- POST /api/v1/query/stream    - Streaming single-agent query
- POST /api/v1/query/compare   - Compare all three RAG approaches
- GET  /api/v1/documents       - List ingested documents
- GET  /api/v1/collections     - List vector collections
- GET  /health                 - Health check
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from typing import Any, AsyncIterator

import structlog
from fastapi import FastAPI, HTTPException, UploadFile
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from agent_rag.agents.hierarchical_rag import HierarchicalRAG, HierarchicalResponse
from agent_rag.agents.multi_agent_rag import MultiAgentRAG, MultiAgentResponse
from agent_rag.agents.single_agent_rag import RAGResponse, SingleAgentRAG
from agent_rag.config import settings
from agent_rag.ingestion.pipeline import IngestionPipeline
from agent_rag.memory import MemoryManager
from agent_rag.retrieval.query_processor import RewriteStrategy

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

_state: dict[str, Any] = {}


def _pipeline() -> IngestionPipeline:
    return _state["pipeline"]


def _single_agent() -> SingleAgentRAG:
    return _state["single_agent"]


def _multi_agent() -> MultiAgentRAG:
    return _state["multi_agent"]


def _hierarchical() -> HierarchicalRAG:
    return _state["hierarchical"]


def _memory() -> MemoryManager:
    return _state["memory"]


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Initialise shared components on startup."""
    logger.info("starting_agent_rag", port=settings.port)

    memory = MemoryManager()
    pipeline = IngestionPipeline()

    _state["pipeline"] = pipeline
    _state["memory"] = memory
    _state["single_agent"] = SingleAgentRAG(memory_manager=memory)
    _state["multi_agent"] = MultiAgentRAG(memory_manager=memory)
    _state["hierarchical"] = HierarchicalRAG(memory_manager=memory)

    yield

    _state.clear()
    logger.info("shutdown_agent_rag")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Agent & RAG System",
    description=(
        "Advanced RAG with single-agent, multi-agent, and hierarchical patterns. "
        "Supports query decomposition, context engineering, and agent memory."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class IngestRequest(BaseModel):
    """Request to ingest text documents."""

    texts: list[str] = Field(
        ..., min_length=1, description="List of text strings to ingest."
    )
    source_prefix: str = Field(
        default="api_upload", description="Prefix for source identifiers."
    )
    metadata: dict[str, Any] | None = Field(
        default=None, description="Extra metadata to attach to all chunks."
    )


class IngestResponse(BaseModel):
    documents_processed: int
    documents_failed: int
    total_chunks: int
    total_vectors: int


class QueryRequest(BaseModel):
    """Request to query the RAG system."""

    question: str = Field(..., min_length=1, description="The user's question.")
    conversation_id: str | None = Field(
        default=None, description="Optional conversation ID for memory."
    )
    top_k: int = Field(default=10, ge=1, le=100, description="Number of chunks to retrieve.")
    rewrite_strategy: str = Field(
        default="none",
        description="Query rewrite strategy: none, hyde, step_back, multi_query.",
    )
    use_reranker: bool = Field(default=False, description="Apply reranking.")
    filters: dict[str, Any] | None = Field(
        default=None, description="Metadata filters for retrieval."
    )


class QueryResponse(BaseModel):
    answer: str
    agent_type: str
    query: str
    num_chunks_retrieved: int
    metadata: dict[str, Any] = {}


class MultiAgentQueryResponse(BaseModel):
    answer: str
    agent_type: str
    query: str
    analysis: str
    critique: str
    revision_count: int
    num_chunks_retrieved: int
    num_steps: int
    metadata: dict[str, Any] = {}


class HierarchicalQueryResponse(BaseModel):
    answer: str
    agent_type: str
    query: str
    plan_complexity: str
    plan_strategies: list[str]
    plan_granularity: str
    plan_reasoning: str
    validation: str
    num_chunks_retrieved: int
    num_tool_calls: int
    metadata: dict[str, Any] = {}


class CompareResponse(BaseModel):
    """Response comparing all three RAG approaches."""

    question: str
    single_agent: QueryResponse
    multi_agent: MultiAgentQueryResponse
    hierarchical: HierarchicalQueryResponse
    timing_ms: dict[str, float]


class StreamRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "healthy", "service": "agent-rag"}


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------

@app.post("/api/v1/ingest", response_model=IngestResponse)
async def ingest_documents(req: IngestRequest) -> IngestResponse:
    """Ingest text documents into the vector store."""
    try:
        stats = await _pipeline().ingest_texts(
            req.texts,
            source_prefix=req.source_prefix,
            extra_metadata=req.metadata,
        )
        return IngestResponse(
            documents_processed=stats.documents_processed,
            documents_failed=stats.documents_failed,
            total_chunks=stats.total_chunks,
            total_vectors=stats.total_vectors,
        )
    except Exception as exc:
        logger.error("ingest_failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/ingest/file")
async def ingest_file(file: UploadFile) -> dict[str, Any]:
    """Ingest a single uploaded file (PDF, DOCX, TXT, MD, HTML)."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    result = await _pipeline().ingest_document(
        file.filename,
        mime_type=file.content_type,
        stream=file.file,
    )
    return {
        "document_id": result.document_id,
        "source": result.source,
        "chunks_created": result.chunks_created,
        "vectors_stored": result.vectors_stored,
        "success": result.success,
        "errors": result.errors,
        "duration_ms": round(result.duration_ms, 1),
    }


# ---------------------------------------------------------------------------
# Query endpoints
# ---------------------------------------------------------------------------

@app.post("/api/v1/query", response_model=QueryResponse)
async def query_single_agent(req: QueryRequest) -> QueryResponse:
    """Query using single-agent RAG (simple retrieve-and-generate)."""
    try:
        strategy = RewriteStrategy(req.rewrite_strategy)
    except ValueError:
        strategy = RewriteStrategy.NONE

    try:
        result: RAGResponse = await _single_agent().query(
            req.question,
            conversation_id=req.conversation_id,
            rewrite_strategy=strategy,
            use_reranker=req.use_reranker,
            top_k=req.top_k,
            filters=req.filters,
        )
        return QueryResponse(
            answer=result.answer,
            agent_type=result.agent_type,
            query=result.query,
            num_chunks_retrieved=len(result.retrieved_chunks),
            metadata=result.metadata,
        )
    except Exception as exc:
        logger.error("single_agent_query_failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/query/multi-agent", response_model=MultiAgentQueryResponse)
async def query_multi_agent(req: QueryRequest) -> MultiAgentQueryResponse:
    """Query using multi-agent RAG (retriever, analyzer, generator, critic)."""
    try:
        result: MultiAgentResponse = await _multi_agent().query(
            req.question,
            conversation_id=req.conversation_id,
            top_k=req.top_k,
            filters=req.filters,
        )
        return MultiAgentQueryResponse(
            answer=result.answer,
            agent_type=result.agent_type,
            query=result.query,
            analysis=result.analysis,
            critique=result.critique,
            revision_count=result.revision_count,
            num_chunks_retrieved=len(result.retrieved_chunks),
            num_steps=len(result.steps),
            metadata=result.metadata,
        )
    except Exception as exc:
        logger.error("multi_agent_query_failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/query/hierarchical", response_model=HierarchicalQueryResponse)
async def query_hierarchical(req: QueryRequest) -> HierarchicalQueryResponse:
    """Query using hierarchical RAG (A-RAG pattern with adaptive retrieval)."""
    try:
        result: HierarchicalResponse = await _hierarchical().query(
            req.question,
            conversation_id=req.conversation_id,
            top_k=req.top_k,
            filters=req.filters,
        )
        return HierarchicalQueryResponse(
            answer=result.answer,
            agent_type=result.agent_type,
            query=result.query,
            plan_complexity=result.plan.complexity.value,
            plan_strategies=result.plan.strategies,
            plan_granularity=result.plan.granularity.value,
            plan_reasoning=result.plan.reasoning,
            validation=result.validation,
            num_chunks_retrieved=len(result.retrieved_chunks),
            num_tool_calls=len(result.tool_calls),
            metadata=result.metadata,
        )
    except Exception as exc:
        logger.error("hierarchical_query_failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/query/stream")
async def query_stream(req: StreamRequest) -> EventSourceResponse:
    """Stream a single-agent RAG response via SSE."""

    async def event_generator() -> AsyncIterator[dict[str, str]]:
        try:
            async for token in _single_agent().query_stream(
                req.question, top_k=req.top_k, filters=req.filters
            ):
                yield {"data": token}
        except Exception as exc:
            logger.error("stream_failed", exc_info=True)
            yield {"event": "error", "data": str(exc)}

    return EventSourceResponse(event_generator())


@app.post("/api/v1/query/compare", response_model=CompareResponse)
async def query_compare(req: QueryRequest) -> CompareResponse:
    """Compare all three RAG approaches on the same query.

    Runs single-agent, multi-agent, and hierarchical RAG in sequence
    and returns the results side by side with timing information.
    """
    timings: dict[str, float] = {}

    # Single agent
    t0 = time.time()
    try:
        single_result = await _single_agent().query(
            req.question, top_k=req.top_k, filters=req.filters
        )
    except Exception as exc:
        single_result = RAGResponse(
            answer=f"Error: {exc}", query=req.question, retrieved_chunks=[]
        )
    timings["single_agent_ms"] = (time.time() - t0) * 1000

    # Multi agent
    t0 = time.time()
    try:
        multi_result = await _multi_agent().query(
            req.question, top_k=req.top_k, filters=req.filters
        )
    except Exception as exc:
        multi_result = MultiAgentResponse(
            answer=f"Error: {exc}", query=req.question, retrieved_chunks=[]
        )
    timings["multi_agent_ms"] = (time.time() - t0) * 1000

    # Hierarchical
    t0 = time.time()
    try:
        hier_result = await _hierarchical().query(
            req.question, top_k=req.top_k, filters=req.filters
        )
    except Exception as exc:
        from agent_rag.agents.hierarchical_rag import (
            GranularityLevel,
            QueryComplexity,
            RetrievalPlan,
        )

        hier_result = HierarchicalResponse(
            answer=f"Error: {exc}",
            query=req.question,
            plan=RetrievalPlan(
                complexity=QueryComplexity.SIMPLE,
                strategies=[],
                granularity=GranularityLevel.PARAGRAPH,
                sub_queries=[],
                reasoning="Error",
            ),
            retrieved_chunks=[],
        )
    timings["hierarchical_ms"] = (time.time() - t0) * 1000

    return CompareResponse(
        question=req.question,
        single_agent=QueryResponse(
            answer=single_result.answer,
            agent_type=single_result.agent_type,
            query=single_result.query,
            num_chunks_retrieved=len(single_result.retrieved_chunks),
            metadata=single_result.metadata,
        ),
        multi_agent=MultiAgentQueryResponse(
            answer=multi_result.answer,
            agent_type=multi_result.agent_type,
            query=multi_result.query,
            analysis=multi_result.analysis,
            critique=multi_result.critique,
            revision_count=multi_result.revision_count,
            num_chunks_retrieved=len(multi_result.retrieved_chunks),
            num_steps=len(multi_result.steps),
            metadata=multi_result.metadata,
        ),
        hierarchical=HierarchicalQueryResponse(
            answer=hier_result.answer,
            agent_type=hier_result.agent_type,
            query=hier_result.query,
            plan_complexity=hier_result.plan.complexity.value,
            plan_strategies=hier_result.plan.strategies,
            plan_granularity=hier_result.plan.granularity.value,
            plan_reasoning=hier_result.plan.reasoning,
            validation=hier_result.validation,
            num_chunks_retrieved=len(hier_result.retrieved_chunks),
            num_tool_calls=len(hier_result.tool_calls),
            metadata=hier_result.metadata,
        ),
        timing_ms=timings,
    )


# ---------------------------------------------------------------------------
# Collection / document management
# ---------------------------------------------------------------------------

@app.get("/api/v1/documents")
async def list_documents() -> list[dict[str, Any]]:
    """List all ingested documents."""
    try:
        return await _pipeline().list_documents()
    except Exception as exc:
        logger.error("list_documents_failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/api/v1/collections")
async def list_collections() -> list[dict[str, Any]]:
    """List vector store collections."""
    try:
        return await _pipeline().get_collections()
    except Exception as exc:
        logger.error("list_collections_failed", exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# Memory stats
# ---------------------------------------------------------------------------

@app.get("/api/v1/memory/stats")
async def memory_stats() -> dict[str, int]:
    """Return current memory statistics."""
    return await _memory().get_stats()
