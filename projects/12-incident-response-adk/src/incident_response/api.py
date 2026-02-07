"""FastAPI application for the Incident Response Orchestrator.

Exposes REST endpoints for:
- Submitting alerts for automated incident response
- Monitoring incident status and streaming real-time progress
- Manual escalation, resolution, and human takeover
- Listing incidents, runbooks, and diagnostics
- Health checks
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from common import ErrorResponse, HealthResponse

from incident_response.config import Settings
from incident_response.mock_data.alerts import MOCK_ALERTS, get_alert_by_id
from incident_response.models import (
    Alert,
    EscalationLevel,
    IncidentSession,
    IncidentState,
    SeverityLevel,
)
from incident_response.streaming import (
    EVENT_ERROR,
    EVENT_ESCALATING,
    EVENT_HUMAN_TAKEOVER,
    EVENT_RESOLVED,
    IncidentEventStream,
)
from incident_response.tools.runbooks import list_runbooks as get_all_runbooks
from incident_response.workflow.pipeline import run_incident_pipeline

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class SubmitAlertRequest(BaseModel):
    """Request to submit a new alert for automated incident response."""

    alert_id: str | None = Field(
        default=None,
        description="ID of a mock alert to use. If not provided, use custom alert fields.",
    )
    source: str = Field(default="api", description="Alert source system")
    title: str = Field(default="", description="Alert title")
    description: str = Field(default="", description="Alert description")
    service: str = Field(default="", description="Affected service name")
    host: str = Field(default="", description="Affected host")
    raw_data: dict[str, Any] = Field(default_factory=dict)


class EscalateRequest(BaseModel):
    """Request to manually escalate an incident."""

    level: str = Field(
        default="",
        description="Target escalation level (L2_ONCALL, L3_SENIOR, L4_MANAGEMENT)",
    )
    reason: str = Field(default="", description="Reason for manual escalation")


class ResolveRequest(BaseModel):
    """Request to manually mark an incident as resolved."""

    resolution_summary: str = Field(default="Manually resolved by operator")


class TakeoverRequest(BaseModel):
    """Request for a human operator to take over an incident."""

    operator: str = Field(default="unknown", description="Operator name or ID")
    reason: str = Field(default="", description="Reason for human takeover")


# ---------------------------------------------------------------------------
# Session Manager (in-memory)
# ---------------------------------------------------------------------------


class SessionManager:
    """In-memory incident session store."""

    def __init__(self) -> None:
        self._sessions: dict[str, IncidentSession] = {}
        self._pipeline_tasks: dict[str, asyncio.Task[Any]] = {}

    async def create_session(self, alert: Alert) -> IncidentSession:
        """Create a new incident session from an alert."""
        session_id = str(uuid.uuid4())
        session = IncidentSession(
            id=session_id,
            state=IncidentState.RECEIVED,
            alert=alert,
            created_at=datetime.now(tz=timezone.utc),
            updated_at=datetime.now(tz=timezone.utc),
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> IncidentSession | None:
        """Retrieve a session by ID."""
        return self._sessions.get(session_id)

    def update_session(self, session_id: str, **kwargs: Any) -> IncidentSession | None:
        """Update session fields."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        for key, value in kwargs.items():
            if hasattr(session, key):
                setattr(session, key, value)
        session.updated_at = datetime.now(tz=timezone.utc)
        return session

    def list_sessions(
        self,
        state: IncidentState | None = None,
        severity: SeverityLevel | None = None,
    ) -> list[IncidentSession]:
        """Return all sessions with optional filters."""
        sessions = list(self._sessions.values())
        if state is not None:
            sessions = [s for s in sessions if s.state == state]
        if severity is not None:
            sessions = [s for s in sessions if s.severity == severity]
        return sorted(sessions, key=lambda s: s.created_at, reverse=True)


# ---------------------------------------------------------------------------
# Application State container
# ---------------------------------------------------------------------------


class AppState:
    """Shared application state accessible from route handlers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session_manager = SessionManager()
        self.event_stream = IncidentEventStream()


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = settings or Settings()

    app = FastAPI(
        title="Incident Response Orchestrator",
        description=(
            "Automated IT incident response using Google ADK workflow patterns. "
            "Triages alerts, runs parallel diagnostics, and loops through "
            "escalation/remediation until resolution."
        ),
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

    # Shared state
    state = AppState(settings)
    app.state.app_state = state
    app.state.settings = settings

    # -------------------------------------------------------------------
    # Health
    # -------------------------------------------------------------------

    @app.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        """Service health check."""
        return HealthResponse(
            status="healthy",
            service=settings.service_name,
            version=settings.service_version,
        )

    # -------------------------------------------------------------------
    # Incident endpoints
    # -------------------------------------------------------------------

    @app.post("/api/v1/incidents", tags=["incidents"])
    async def submit_incident(req: SubmitAlertRequest) -> dict[str, Any]:
        """Submit an alert for automated incident response.

        Creates an incident session and kicks off the full pipeline
        (triage -> diagnostics -> remediation) asynchronously.
        Use the ``/stream`` endpoint to follow real-time progress.
        """
        # Resolve alert
        if req.alert_id:
            alert = get_alert_by_id(req.alert_id)
            if alert is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"Mock alert '{req.alert_id}' not found. "
                        f"Available: {[a.id for a in MOCK_ALERTS]}"
                    ),
                )
        else:
            if not req.title or not req.service:
                raise HTTPException(
                    status_code=400,
                    detail="Either alert_id or both title and service are required.",
                )
            alert = Alert(
                id=f"ALT-{uuid.uuid4().hex[:6].upper()}",
                source=req.source,
                title=req.title,
                description=req.description,
                service=req.service,
                host=req.host,
                raw_data=req.raw_data,
            )

        # Create session
        session = await state.session_manager.create_session(alert)

        # Launch pipeline asynchronously
        async def _run_pipeline() -> None:
            try:
                result = await run_incident_pipeline(
                    alert=alert,
                    settings=settings,
                    event_stream=state.event_stream,
                    session_id=session.id,
                )
                # Update session with final state
                is_resolved = result.get("loop_complete", False)
                final_state = (
                    IncidentState.RESOLVED if is_resolved else IncidentState.HUMAN_TAKEOVER
                )
                state.session_manager.update_session(
                    session.id,
                    state=final_state,
                    severity=result.get("severity", SeverityLevel.P4),
                    context=result.get("incident_context"),
                    diagnostics=[
                        result.get(k)
                        for k in ["log_diagnostics", "metrics_diagnostics", "config_diagnostics"]
                        if result.get(k) is not None
                    ],
                    remediations=result.get("remediation_actions", []),
                    escalation_level=result.get("escalation_level", EscalationLevel.L1_AUTO),
                    report=result.get("report"),
                )
            except Exception as exc:
                logger.error("pipeline_task_error", session_id=session.id, error=str(exc))
                state.session_manager.update_session(
                    session.id,
                    state=IncidentState.FAILED,
                    error=str(exc),
                )
                await state.event_stream.emit(
                    session.id,
                    EVENT_ERROR,
                    data={"error": str(exc)},
                    message=f"Pipeline failed: {exc}",
                )

        task = asyncio.create_task(_run_pipeline())
        state.session_manager._pipeline_tasks[session.id] = task

        return {
            "session_id": session.id,
            "alert_id": alert.id,
            "status": session.state.value,
            "message": f"Incident response initiated for: {alert.title}",
            "stream_url": f"/api/v1/incidents/{session.id}/stream",
        }

    @app.get("/api/v1/incidents/{session_id}", tags=["incidents"])
    async def get_incident(session_id: str) -> dict[str, Any]:
        """Get the current state of an incident session."""
        session = state.session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")
        return session.model_dump(mode="json")

    @app.get("/api/v1/incidents/{session_id}/stream", tags=["incidents"])
    async def stream_incident(session_id: str) -> EventSourceResponse:
        """SSE stream of incident response workflow events."""
        session = state.session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        async def event_generator():  # type: ignore[no-untyped-def]
            async for event in state.event_stream.subscribe(session_id):
                yield {
                    "event": event.event_type,
                    "data": json.dumps(event.model_dump(), default=str),
                }

        return EventSourceResponse(event_generator())

    @app.post("/api/v1/incidents/{session_id}/escalate", tags=["incidents"])
    async def escalate_incident(
        session_id: str, req: EscalateRequest
    ) -> dict[str, Any]:
        """Manually escalate an incident to a higher tier."""
        session = state.session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        if session.state in (IncidentState.RESOLVED, IncidentState.FAILED):
            raise HTTPException(
                status_code=400,
                detail=f"Cannot escalate incident in state '{session.state.value}'",
            )

        # Determine target level
        target_level = EscalationLevel.L3_SENIOR
        if req.level:
            try:
                target_level = EscalationLevel(req.level)
            except ValueError:
                raise HTTPException(
                    status_code=400,
                    detail=f"Invalid escalation level: {req.level}. "
                    f"Valid: {[e.value for e in EscalationLevel]}",
                )

        state.session_manager.update_session(
            session_id,
            state=IncidentState.ESCALATING,
            escalation_level=target_level,
        )

        await state.event_stream.emit(
            session_id,
            EVENT_ESCALATING,
            data={
                "escalation_level": target_level.value,
                "reason": req.reason,
            },
            message=f"Manually escalated to {target_level.value}: {req.reason}",
        )

        return {
            "session_id": session_id,
            "status": "escalated",
            "escalation_level": target_level.value,
            "message": f"Incident escalated to {target_level.value}",
        }

    @app.post("/api/v1/incidents/{session_id}/resolve", tags=["incidents"])
    async def resolve_incident(
        session_id: str, req: ResolveRequest
    ) -> dict[str, Any]:
        """Manually mark an incident as resolved."""
        session = state.session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        if session.state == IncidentState.RESOLVED:
            raise HTTPException(status_code=400, detail="Incident is already resolved")

        # Cancel any running pipeline task
        task = state.session_manager._pipeline_tasks.get(session_id)
        if task and not task.done():
            task.cancel()

        state.session_manager.update_session(
            session_id,
            state=IncidentState.RESOLVED,
        )

        await state.event_stream.emit(
            session_id,
            EVENT_RESOLVED,
            data={"resolution_summary": req.resolution_summary},
            message=f"Incident manually resolved: {req.resolution_summary}",
        )

        return {
            "session_id": session_id,
            "status": "resolved",
            "message": req.resolution_summary,
        }

    @app.post("/api/v1/incidents/{session_id}/takeover", tags=["incidents"])
    async def takeover_incident(
        session_id: str, req: TakeoverRequest
    ) -> dict[str, Any]:
        """Human operator takes over the incident."""
        session = state.session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        if session.state == IncidentState.RESOLVED:
            raise HTTPException(status_code=400, detail="Incident is already resolved")

        # Cancel any running pipeline task
        task = state.session_manager._pipeline_tasks.get(session_id)
        if task and not task.done():
            task.cancel()

        state.session_manager.update_session(
            session_id,
            state=IncidentState.HUMAN_TAKEOVER,
        )

        await state.event_stream.emit(
            session_id,
            EVENT_HUMAN_TAKEOVER,
            data={
                "operator": req.operator,
                "reason": req.reason,
            },
            message=f"Human takeover by {req.operator}: {req.reason}",
        )

        return {
            "session_id": session_id,
            "status": "human_takeover",
            "operator": req.operator,
            "message": f"Incident taken over by {req.operator}",
        }

    @app.get("/api/v1/incidents", tags=["incidents"])
    async def list_incidents(
        state_filter: IncidentState | None = Query(default=None, alias="state"),
        severity_filter: SeverityLevel | None = Query(default=None, alias="severity"),
    ) -> dict[str, Any]:
        """List all incidents with optional filters."""
        sessions = state.session_manager.list_sessions(
            state=state_filter,
            severity=severity_filter,
        )
        return {
            "incidents": [s.model_dump(mode="json") for s in sessions],
            "total": len(sessions),
            "filters": {
                "state": state_filter.value if state_filter else None,
                "severity": severity_filter.value if severity_filter else None,
            },
        }

    # -------------------------------------------------------------------
    # Runbooks endpoint
    # -------------------------------------------------------------------

    @app.get("/api/v1/runbooks", tags=["runbooks"])
    async def list_runbooks_endpoint() -> dict[str, Any]:
        """List all available remediation runbooks."""
        runbooks = get_all_runbooks()
        return {
            "runbooks": runbooks,
            "total": len(runbooks),
        }

    # -------------------------------------------------------------------
    # Diagnostics endpoint
    # -------------------------------------------------------------------

    @app.get("/api/v1/diagnostics/{session_id}", tags=["diagnostics"])
    async def get_diagnostics(session_id: str) -> dict[str, Any]:
        """Get diagnostic results for an incident session."""
        session = state.session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(status_code=404, detail=f"Session {session_id} not found")

        diagnostics = [d.model_dump(mode="json") for d in session.diagnostics]
        return {
            "session_id": session_id,
            "diagnostics": diagnostics,
            "total": len(diagnostics),
        }

    # -------------------------------------------------------------------
    # Mock alerts endpoint (for discovery)
    # -------------------------------------------------------------------

    @app.get("/api/v1/alerts/mock", tags=["alerts"])
    async def list_mock_alerts() -> dict[str, Any]:
        """List all available mock alerts for testing."""
        return {
            "alerts": [a.model_dump(mode="json") for a in MOCK_ALERTS],
            "total": len(MOCK_ALERTS),
        }

    # -------------------------------------------------------------------
    # Error handlers
    # -------------------------------------------------------------------

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        """Handle unhandled exceptions."""
        logger.error(
            "unhandled_exception", error=str(exc), path=request.url.path
        )
        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error="Internal server error",
                detail=str(exc),
                status_code=500,
            ).model_dump(),
        )

    return app
