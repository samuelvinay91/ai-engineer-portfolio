"""FastAPI application for the MCP & A2A Integration service.

Exposes REST endpoints for:
- MCP tool discovery, execution, and security scanning
- MCP vs REST comparison
- A2A task submission, status polling, and SSE streaming
- Agent discovery and the well-known AgentCard endpoint
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from common import ErrorResponse, HealthResponse

from mcp_a2a.a2a.agent_card import AgentCardRegistry, build_default_agent_card
from mcp_a2a.a2a.discovery import AgentDiscoveryService
from mcp_a2a.a2a.orchestrator import A2AOrchestrator
from mcp_a2a.a2a.protocol import (
    A2AClient,
    A2AServer,
    Message,
    TaskSendParams,
    TextPart,
)
from mcp_a2a.config import MCPSettings, get_settings
from mcp_a2a.mcp.comparison import compare_approaches, generate_full_report
from mcp_a2a.mcp.security import (
    ATTACK_VECTORS,
    AuditAction,
    AuditLogger,
    SecurityGateway,
    SecurityScanResult,
    ToolRateLimiter,
    detect_prompt_injection,
)
from mcp_a2a.mcp.server import (
    execute_tool,
    get_prompt,
    list_prompts,
    list_resources,
    list_tools,
    read_resource,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class ToolExecuteRequest(BaseModel):
    arguments: dict[str, Any] = Field(default_factory=dict)
    actor: str = "anonymous"


class CompareRequest(BaseModel):
    operation: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class DiscoverRequest(BaseModel):
    urls: list[str]


class SecurityScanRequest(BaseModel):
    text: str


class PromptRequest(BaseModel):
    arguments: dict[str, str] = Field(default_factory=dict)


class ResourceRequest(BaseModel):
    uri: str


class TaskRequest(BaseModel):
    message: str
    task_id: str | None = None
    session_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class OrchestrateRequest(BaseModel):
    request: str


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(settings: MCPSettings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = settings or get_settings()

    app = FastAPI(
        title="MCP & A2A Integration",
        description="Model Context Protocol server/client and Agent-to-Agent interoperability layer",
        version=settings.service_version,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # -- Shared state --------------------------------------------------------
    security_gateway = SecurityGateway(
        rate_limiter=ToolRateLimiter(max_per_minute=settings.mcp_rate_limit_per_minute),
        audit_logger=AuditLogger(enabled=settings.mcp_audit_log_enabled),
        max_input_length=settings.mcp_max_input_length,
    )
    agent_card = build_default_agent_card(base_url=settings.a2a_base_url)
    registry = AgentCardRegistry(cache_ttl_seconds=settings.discovery_cache_ttl_seconds)
    registry.register(agent_card)

    discovery_service = AgentDiscoveryService(registry=registry)
    a2a_server = A2AServer()
    a2a_client = A2AClient()
    orchestrator = A2AOrchestrator(discovery=discovery_service, client=a2a_client)

    # Store on app state for access in tests
    app.state.settings = settings
    app.state.security_gateway = security_gateway
    app.state.registry = registry
    app.state.discovery_service = discovery_service
    app.state.a2a_server = a2a_server
    app.state.orchestrator = orchestrator

    # -----------------------------------------------------------------------
    # Health & well-known
    # -----------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            service=settings.service_name,
            version=settings.service_version,
        )

    @app.get("/.well-known/agent.json", tags=["a2a"])
    async def well_known_agent_card() -> dict[str, Any]:
        """Serve the AgentCard at the well-known URL per the A2A spec."""
        return agent_card.model_dump()

    # -----------------------------------------------------------------------
    # MCP Tool endpoints
    # -----------------------------------------------------------------------

    @app.post("/api/v1/mcp/tools", tags=["mcp"])
    async def mcp_list_tools() -> dict[str, Any]:
        """List all available MCP tools with their schemas."""
        security_gateway.audit_logger.log(AuditAction.TOOL_DISCOVERY, actor="api")
        return {
            "tools": list_tools(),
            "total": len(list_tools()),
        }

    @app.post("/api/v1/mcp/tools/{tool_name}/execute", tags=["mcp"])
    async def mcp_execute_tool(tool_name: str, req: ToolExecuteRequest) -> dict[str, Any]:
        """Execute an MCP tool with security checks."""
        # Security gate
        allowed, scan_result, rate_result = security_gateway.authorize_tool_call(
            tool_name, req.arguments, actor=req.actor
        )
        if not allowed:
            if not rate_result.allowed:
                raise HTTPException(
                    status_code=429,
                    detail=f"Rate limit exceeded. Resets at {rate_result.reset_at}",
                )
            raise HTTPException(
                status_code=400,
                detail={
                    "message": "Request blocked by security scan",
                    "findings": scan_result.findings,
                    "threat_level": scan_result.threat_level.value,
                },
            )

        result = execute_tool(tool_name, req.arguments)
        return result.model_dump()

    @app.get("/api/v1/mcp/resources", tags=["mcp"])
    async def mcp_list_resources() -> dict[str, Any]:
        """List all available MCP resources."""
        return {"resources": list_resources()}

    @app.post("/api/v1/mcp/resources/read", tags=["mcp"])
    async def mcp_read_resource(req: ResourceRequest) -> dict[str, Any]:
        """Read a specific MCP resource by URI."""
        security_gateway.audit_logger.log(
            AuditAction.RESOURCE_READ, actor="api", details={"uri": req.uri}
        )
        return read_resource(req.uri)

    @app.get("/api/v1/mcp/prompts", tags=["mcp"])
    async def mcp_list_prompts() -> dict[str, Any]:
        """List all available MCP prompt templates."""
        return {"prompts": list_prompts()}

    @app.post("/api/v1/mcp/prompts/{prompt_name}", tags=["mcp"])
    async def mcp_get_prompt(prompt_name: str, req: PromptRequest) -> dict[str, Any]:
        """Expand a prompt template with the given arguments."""
        security_gateway.audit_logger.log(
            AuditAction.PROMPT_GET,
            actor="api",
            details={"prompt": prompt_name},
        )
        return get_prompt(prompt_name, req.arguments)

    # -----------------------------------------------------------------------
    # MCP Comparison endpoints
    # -----------------------------------------------------------------------

    @app.post("/api/v1/mcp/compare", tags=["mcp"])
    async def mcp_compare(req: CompareRequest) -> dict[str, Any]:
        """Compare MCP vs traditional REST for the same operation."""
        result = await compare_approaches(req.operation, req.arguments)
        return result.model_dump()

    @app.get("/api/v1/mcp/compare/report", tags=["mcp"])
    async def mcp_compare_report() -> dict[str, Any]:
        """Generate a full comparison report across all tools."""
        return await generate_full_report()

    # -----------------------------------------------------------------------
    # MCP Security endpoints
    # -----------------------------------------------------------------------

    @app.get("/api/v1/mcp/security/scan", tags=["security"])
    async def security_scan_info() -> dict[str, Any]:
        """Return information about the security scanning capabilities."""
        return {
            "description": "MCP Security Scanner for prompt injection detection",
            "capabilities": [
                "Prompt injection detection",
                "Input validation",
                "Rate limiting",
                "Audit logging",
                "Path traversal prevention",
                "SQL injection detection",
            ],
            "attack_vectors": ATTACK_VECTORS,
        }

    @app.post("/api/v1/mcp/security/scan", tags=["security"])
    async def security_scan(req: SecurityScanRequest) -> dict[str, Any]:
        """Scan text for prompt injection and other security threats."""
        security_gateway.audit_logger.log(
            AuditAction.SECURITY_SCAN,
            actor="api",
            details={"text_length": len(req.text)},
        )
        result = detect_prompt_injection(req.text)
        return result.model_dump()

    @app.get("/api/v1/mcp/security/audit", tags=["security"])
    async def security_audit_log(limit: int = 50) -> dict[str, Any]:
        """Retrieve recent audit log entries."""
        entries = security_gateway.audit_logger.get_entries(limit=limit)
        return {
            "entries": [e.model_dump() for e in entries],
            "total": len(entries),
        }

    # -----------------------------------------------------------------------
    # A2A Task endpoints
    # -----------------------------------------------------------------------

    @app.post("/api/v1/a2a/tasks", tags=["a2a"])
    async def a2a_submit_task(req: TaskRequest) -> dict[str, Any]:
        """Submit a task via the A2A protocol."""
        params = TaskSendParams(
            id=req.task_id or "",
            session_id=req.session_id,
            message=Message(
                role="user",
                parts=[TextPart(text=req.message)],
            ),
            metadata=req.metadata,
        )
        if not params.id:
            params.id = str(__import__("uuid").uuid4())

        task = await a2a_server.handle_task_send(params)
        return task.model_dump()

    @app.get("/api/v1/a2a/tasks/{task_id}", tags=["a2a"])
    async def a2a_get_task(task_id: str) -> dict[str, Any]:
        """Get current status of an A2A task."""
        task = await a2a_server.handle_task_get(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        return task.model_dump()

    @app.post("/api/v1/a2a/tasks/{task_id}/cancel", tags=["a2a"])
    async def a2a_cancel_task(task_id: str) -> dict[str, Any]:
        """Cancel an in-progress A2A task."""
        task = await a2a_server.handle_task_cancel(task_id)
        if task is None:
            raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")
        return task.model_dump()

    @app.get("/api/v1/a2a/tasks/{task_id}/stream", tags=["a2a"])
    async def a2a_stream_task(task_id: str) -> EventSourceResponse:
        """Stream task status updates via SSE."""

        async def event_generator():  # type: ignore[no-untyped-def]
            async for update in a2a_server.subscribe(task_id):
                yield {
                    "event": "task_status",
                    "data": json.dumps(update.model_dump(), default=str),
                }
                if update.final:
                    break

        return EventSourceResponse(event_generator())

    # -----------------------------------------------------------------------
    # A2A Discovery endpoints
    # -----------------------------------------------------------------------

    @app.post("/api/v1/a2a/discover", tags=["a2a"])
    async def a2a_discover_agents(req: DiscoverRequest) -> dict[str, Any]:
        """Discover agents at the given URLs."""
        cards = await discovery_service.discover_agents(req.urls)
        return {
            "discovered": len(cards),
            "agents": [c.model_dump() for c in cards],
        }

    @app.get("/api/v1/a2a/agents", tags=["a2a"])
    async def a2a_list_agents() -> dict[str, Any]:
        """List all known agents in the registry."""
        return discovery_service.get_status()

    @app.post("/api/v1/a2a/orchestrate", tags=["a2a"])
    async def a2a_orchestrate(req: OrchestrateRequest) -> dict[str, Any]:
        """Submit a complex task for orchestration across agents."""
        plan = await orchestrator.orchestrate(req.request)
        return orchestrator.plan_to_dict(plan)

    # -----------------------------------------------------------------------
    # Error handlers
    # -----------------------------------------------------------------------

    @app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.error("unhandled_exception", error=str(exc), path=request.url.path)
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                detail=str(exc),
                status_code=500,
            ).model_dump(),
        )

    return app
