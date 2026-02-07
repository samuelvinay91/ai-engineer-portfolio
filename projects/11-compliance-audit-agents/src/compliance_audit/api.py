"""FastAPI application for the Compliance Audit Agents service.

Exposes REST endpoints for:
- Audit session management (create, status, approve, reject)
- SSE streaming of audit workflow progress
- Audit report retrieval
- Policy listing and quick single-transaction checks
"""

from __future__ import annotations

import asyncio
import json
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

from common import ErrorResponse, HealthResponse

from compliance_audit.config import Settings
from compliance_audit.mock_data.policies import MOCK_POLICIES
from compliance_audit.mock_data.transactions import MOCK_TRANSACTIONS
from compliance_audit.models import (
    AuditSession,
    AuditSessionState,
    TransactionLog,
)
from compliance_audit.streaming import (
    EVENT_APPROVED,
    EVENT_COMPLETED,
    EVENT_ERROR,
    EVENT_REJECTED,
    AuditEventStream,
)
from compliance_audit.workflow.graph import AuditWorkflowGraph, compile_audit_graph
from compliance_audit.workflow.state import AuditGraphState

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------


class AuditRequest(BaseModel):
    """Incoming audit request with optional transactions."""

    transactions: list[dict[str, Any]] = Field(default_factory=list)
    use_mock_data: bool = True


class ApprovalRequest(BaseModel):
    """Audit approval request with optional comments."""

    comments: str = ""


class QuickCheckRequest(BaseModel):
    """Single-transaction quick compliance check."""

    actor: str
    action: str
    resource: str
    department: str = "unknown"
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Session manager (in-memory)
# ---------------------------------------------------------------------------


class SessionManager:
    """In-memory audit session store."""

    def __init__(self) -> None:
        self._sessions: dict[str, AuditSession] = {}
        self._graph_tasks: dict[str, asyncio.Task[Any]] = {}

    async def create_session(
        self,
        transactions: list[TransactionLog],
    ) -> AuditSession:
        """Create a new audit session."""
        session_id = str(uuid.uuid4())
        session = AuditSession(
            id=session_id,
            state=AuditSessionState.INGESTING,
            transactions=transactions,
            created_at=datetime.now(tz=timezone.utc),
            updated_at=datetime.now(tz=timezone.utc),
        )
        self._sessions[session_id] = session
        return session

    def get_session(self, session_id: str) -> AuditSession | None:
        """Retrieve a session by ID."""
        return self._sessions.get(session_id)

    def update_session(
        self,
        session_id: str,
        **kwargs: Any,
    ) -> AuditSession | None:
        """Update session fields."""
        session = self._sessions.get(session_id)
        if session is None:
            return None
        for key, value in kwargs.items():
            if hasattr(session, key):
                setattr(session, key, value)
        session.updated_at = datetime.now(tz=timezone.utc)
        return session

    def list_sessions(self) -> list[AuditSession]:
        """Return all sessions."""
        return list(self._sessions.values())


# ---------------------------------------------------------------------------
# Application state container
# ---------------------------------------------------------------------------


class AppState:
    """Shared application state accessible from route handlers."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.session_manager = SessionManager()
        self.event_stream = AuditEventStream()
        self.approval_events: dict[str, asyncio.Event] = {}
        self.approval_results: dict[str, bool] = {}


# ---------------------------------------------------------------------------
# Application factory
# ---------------------------------------------------------------------------


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = settings or Settings()

    app = FastAPI(
        title="Compliance Audit Agents",
        description=(
            "Multi-agent compliance audit system that ingests policy "
            "documents and transaction logs, dispatches specialized agents "
            "to check SOX, GDPR, and SOC 2 compliance. Graph-based "
            "workflow routes findings through classification, domain "
            "checkers, risk scoring, and remediation, with human-in-the-loop "
            "approval."
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
        return HealthResponse(
            status="healthy",
            service=settings.service_name,
            version=settings.service_version,
        )

    # -------------------------------------------------------------------
    # Audit session endpoints
    # -------------------------------------------------------------------

    @app.post("/api/v1/audits", tags=["audits"])
    async def create_audit(req: AuditRequest) -> dict[str, Any]:
        """Submit transactions for compliance audit.

        Creates an audit session and kicks off the workflow graph
        asynchronously.  Use the ``/stream`` endpoint to follow progress.
        """
        # Build transaction list
        if req.use_mock_data and not req.transactions:
            transactions = MOCK_TRANSACTIONS
        else:
            transactions = []
            for tx_dict in req.transactions:
                transactions.append(TransactionLog(**tx_dict))

        if len(transactions) > settings.max_transactions_per_audit:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Too many transactions ({len(transactions)}). "
                    f"Maximum is {settings.max_transactions_per_audit}."
                ),
            )

        session = await state.session_manager.create_session(transactions)

        # Set up approval gate
        approval_event = asyncio.Event()
        state.approval_events[session.id] = approval_event

        async def _run_graph() -> None:
            """Execute the audit workflow graph."""
            graph = compile_audit_graph(state.event_stream)

            initial: AuditGraphState = {
                "session_id": session.id,
                "current_state": AuditSessionState.INGESTING,
                "transactions": transactions,
                "policies": MOCK_POLICIES,
                "classified": {},
                "regulation_types_found": [],
                "sox_findings": [],
                "gdpr_findings": [],
                "soc2_findings": [],
                "all_findings": [],
                "risk_scores": [],
                "remediations": [],
                "approved": None,
                "report": None,
                "error": None,
                "messages": [],
            }

            result = await graph.execute(initial)

            # Update session with results
            current_state = result.get(
                "current_state", AuditSessionState.FAILED
            )
            state.session_manager.update_session(
                session.id,
                state=current_state,
                findings=result.get("all_findings", []),
                error=result.get("error"),
            )

            # If awaiting approval, wait for human decision
            if current_state == AuditSessionState.AWAITING_APPROVAL:
                if settings.human_approval_required:
                    logger.info(
                        "awaiting_approval",
                        session_id=session.id,
                    )
                    await approval_event.wait()

                    approved = state.approval_results.get(session.id, False)
                    result["approved"] = approved

                    if approved:
                        result["current_state"] = AuditSessionState.APPROVED
                        state.session_manager.update_session(
                            session.id,
                            state=AuditSessionState.APPROVED,
                        )
                        await state.event_stream.emit(
                            session_id=session.id,
                            event_type=EVENT_APPROVED,
                            message="Audit findings approved.",
                        )
                    else:
                        result["current_state"] = AuditSessionState.REJECTED
                        state.session_manager.update_session(
                            session.id,
                            state=AuditSessionState.REJECTED,
                        )
                        await state.event_stream.emit(
                            session_id=session.id,
                            event_type=EVENT_REJECTED,
                            message="Audit findings rejected.",
                        )

                    # Finalize report
                    graph_instance = compile_audit_graph(state.event_stream)
                    result = await graph_instance._node_finalize(result)
                    state.session_manager.update_session(
                        session.id,
                        state=AuditSessionState.COMPLETED,
                        report=result.get("report"),
                    )
                else:
                    # Auto-approve
                    graph_instance = compile_audit_graph(state.event_stream)
                    result = await graph_instance._node_finalize(result)
                    state.session_manager.update_session(
                        session.id,
                        state=AuditSessionState.COMPLETED,
                        report=result.get("report"),
                    )

        task = asyncio.create_task(_run_graph())
        state.session_manager._graph_tasks[session.id] = task

        return {
            "session_id": session.id,
            "status": session.state.value,
            "transaction_count": len(transactions),
            "message": f"Audit session created with {len(transactions)} transactions.",
            "stream_url": f"/api/v1/audits/{session.id}/stream",
        }

    @app.get("/api/v1/audits/{session_id}", tags=["audits"])
    async def get_audit_session(session_id: str) -> dict[str, Any]:
        """Get the current state of an audit session."""
        session = state.session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found",
            )
        return session.model_dump(mode="json")

    @app.get("/api/v1/audits/{session_id}/stream", tags=["audits"])
    async def stream_audit_session(session_id: str) -> EventSourceResponse:
        """SSE stream of audit workflow events."""
        session = state.session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found",
            )

        async def event_generator():  # type: ignore[no-untyped-def]
            async for event in state.event_stream.subscribe(session_id):
                yield {
                    "event": event.event_type,
                    "data": json.dumps(
                        event.model_dump(mode="json"), default=str
                    ),
                }

        return EventSourceResponse(event_generator())

    @app.post("/api/v1/audits/{session_id}/approve", tags=["audits"])
    async def approve_audit(
        session_id: str, req: ApprovalRequest | None = None
    ) -> dict[str, Any]:
        """Approve audit findings and proceed to finalization."""
        session = state.session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found",
            )

        if session.state != AuditSessionState.AWAITING_APPROVAL:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Session is in state '{session.state.value}', "
                    "not awaiting approval."
                ),
            )

        state.approval_results[session_id] = True
        approval_event = state.approval_events.get(session_id)
        if approval_event:
            approval_event.set()

        return {
            "session_id": session_id,
            "status": "approved",
            "message": "Audit findings approved. Generating final report.",
        }

    @app.post("/api/v1/audits/{session_id}/reject", tags=["audits"])
    async def reject_audit(
        session_id: str, req: ApprovalRequest | None = None
    ) -> dict[str, Any]:
        """Reject audit findings."""
        session = state.session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found",
            )

        if session.state != AuditSessionState.AWAITING_APPROVAL:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Session is in state '{session.state.value}', "
                    "not awaiting approval."
                ),
            )

        state.approval_results[session_id] = False
        approval_event = state.approval_events.get(session_id)
        if approval_event:
            approval_event.set()

        return {
            "session_id": session_id,
            "status": "rejected",
            "message": "Audit findings rejected. Review required.",
        }

    # -------------------------------------------------------------------
    # Report endpoints
    # -------------------------------------------------------------------

    @app.get("/api/v1/reports/{session_id}", tags=["reports"])
    async def get_audit_report(session_id: str) -> dict[str, Any]:
        """Get the audit report for a completed session."""
        session = state.session_manager.get_session(session_id)
        if session is None:
            raise HTTPException(
                status_code=404,
                detail=f"Session {session_id} not found",
            )

        if session.report is None:
            raise HTTPException(
                status_code=400,
                detail=(
                    "No report available yet. Session is in state: "
                    f"{session.state.value}"
                ),
            )

        return session.report.model_dump(mode="json")

    # -------------------------------------------------------------------
    # Policy endpoints
    # -------------------------------------------------------------------

    @app.get("/api/v1/policies", tags=["policies"])
    async def list_policies() -> dict[str, Any]:
        """List all available compliance policies."""
        return {
            "policies": [p.model_dump(mode="json") for p in MOCK_POLICIES],
            "total": len(MOCK_POLICIES),
        }

    @app.post("/api/v1/policies/check", tags=["policies"])
    async def quick_check(req: QuickCheckRequest) -> dict[str, Any]:
        """Run a quick compliance check on a single transaction.

        Creates a temporary audit session with one transaction and
        returns findings synchronously.
        """
        transaction = TransactionLog(
            id=str(uuid.uuid4()),
            actor=req.actor,
            action=req.action,
            resource=req.resource,
            department=req.department,
            metadata=req.metadata,
        )

        # Run the full pipeline synchronously on one transaction
        graph = compile_audit_graph(state.event_stream)

        quick_state: AuditGraphState = {
            "session_id": f"quick-{uuid.uuid4()}",
            "current_state": AuditSessionState.INGESTING,
            "transactions": [transaction],
            "policies": MOCK_POLICIES,
            "classified": {},
            "regulation_types_found": [],
            "sox_findings": [],
            "gdpr_findings": [],
            "soc2_findings": [],
            "all_findings": [],
            "risk_scores": [],
            "remediations": [],
            "approved": None,
            "report": None,
            "error": None,
            "messages": [],
        }

        result = await graph.execute(quick_state)

        findings = result.get("all_findings", [])
        risk_scores = result.get("risk_scores", [])
        remediations = result.get("remediations", [])

        return {
            "transaction_id": transaction.id,
            "findings": [f.model_dump(mode="json") for f in findings],
            "risk_scores": [s.model_dump(mode="json") for s in risk_scores],
            "remediations": [r.model_dump(mode="json") for r in remediations],
            "finding_count": len(findings),
        }

    # -------------------------------------------------------------------
    # Error handlers
    # -------------------------------------------------------------------

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.error(
            "unhandled_exception",
            error=str(exc),
            path=request.url.path,
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
