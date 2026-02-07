"""Workflow state type definitions.

Defines the TypedDict that flows through the incident response pipeline,
carrying all accumulated data from enrichment through resolution.
"""

from __future__ import annotations

from typing import Any, TypedDict

from incident_response.models import (
    DiagnosticResult,
    EscalationLevel,
    IncidentContext,
    IncidentReport,
    RemediationAction,
    SeverityLevel,
)


class IncidentWorkflowState(TypedDict, total=False):
    """Full state passed through the incident response workflow.

    Each workflow stage reads from and writes to this shared state dict.
    Fields are optional (``total=False``) because they are populated
    progressively as the pipeline executes.
    """

    # --- Input ---
    alert: Any  # Alert model instance
    session_id: str

    # --- Enrichment (SequentialAgent stage 1) ---
    incident_context: IncidentContext
    service_info: dict[str, Any]
    owner_team: str

    # --- Triage (SequentialAgent stage 2) ---
    severity: SeverityLevel
    triage_reasoning: str

    # --- Responder assignment (SequentialAgent stage 3) ---
    assigned_responder: dict[str, Any]
    escalation_path: list[dict[str, Any]]
    escalation_level: EscalationLevel
    notification_channels: list[str]

    # --- Diagnostics (ParallelAgent) ---
    log_diagnostics: DiagnosticResult
    metrics_diagnostics: DiagnosticResult
    config_diagnostics: DiagnosticResult

    # --- Remediation loop (LoopAgent) ---
    remediation_actions: list[RemediationAction]
    last_remediation: RemediationAction
    verification_result: dict[str, Any]
    needs_escalation: bool

    # --- Loop control ---
    loop_iteration: int
    loop_complete: bool
    loop_exhausted: bool

    # --- Final ---
    report: IncidentReport
    errors: dict[str, str]
