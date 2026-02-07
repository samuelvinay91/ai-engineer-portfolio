"""Pydantic models for the Incident Response Orchestrator.

Covers alerts, incident lifecycle states, severity levels, escalation tiers,
diagnostic results, remediation actions, incident reports, sessions, and
SSE event payloads.
"""

from __future__ import annotations

import enum
from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class SeverityLevel(str, enum.Enum):
    """Incident severity classification following PagerDuty conventions."""

    P1 = "P1"  # Critical - service down, customer-facing impact
    P2 = "P2"  # High - significant degradation
    P3 = "P3"  # Medium - elevated errors, performance issues
    P4 = "P4"  # Low - minor anomaly, informational


class IncidentState(str, enum.Enum):
    """Lifecycle states of an incident response session."""

    RECEIVED = "received"
    ENRICHING = "enriching"
    TRIAGING = "triaging"
    DIAGNOSING = "diagnosing"
    REMEDIATING = "remediating"
    VERIFYING = "verifying"
    ESCALATING = "escalating"
    RESOLVED = "resolved"
    HUMAN_TAKEOVER = "human_takeover"
    FAILED = "failed"


class EscalationLevel(str, enum.Enum):
    """On-call escalation tiers."""

    L1_AUTO = "L1_AUTO"      # Automated response
    L2_ONCALL = "L2_ONCALL"  # Primary on-call engineer
    L3_SENIOR = "L3_SENIOR"  # Senior / Staff engineer
    L4_MANAGEMENT = "L4_MANAGEMENT"  # Engineering management


# ---------------------------------------------------------------------------
# Core domain models
# ---------------------------------------------------------------------------


class Alert(BaseModel):
    """Incoming alert from a monitoring system (e.g. PagerDuty, Datadog)."""

    id: str
    source: str  # e.g. "datadog", "cloudwatch", "prometheus"
    title: str
    description: str
    service: str
    host: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    raw_data: dict[str, Any] = Field(default_factory=dict)


class IncidentContext(BaseModel):
    """Enriched context assembled around an alert for triage."""

    alert: Alert
    service_info: dict[str, Any] = Field(default_factory=dict)
    recent_deploys: list[dict[str, Any]] = Field(default_factory=list)
    owner_team: str = ""
    related_incidents: list[dict[str, Any]] = Field(default_factory=list)


class DiagnosticResult(BaseModel):
    """Output from a diagnostic agent (log, metric, or config analysis)."""

    agent_name: str
    findings: list[str] = Field(default_factory=list)
    anomalies: list[dict[str, Any]] = Field(default_factory=list)
    severity_indicators: list[str] = Field(default_factory=list)
    raw_data: dict[str, Any] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class RemediationAction(BaseModel):
    """A single remediation step executed by the remediator agent."""

    action_type: str  # restart_service, scale_up, rollback_deploy, etc.
    description: str
    runbook_id: str = ""
    parameters: dict[str, Any] = Field(default_factory=dict)
    executed: bool = False
    success: bool = False
    output: str = ""


class IncidentReport(BaseModel):
    """Final incident report summarizing the full response lifecycle."""

    id: str
    session_id: str
    alert: Alert
    severity: SeverityLevel = SeverityLevel.P4
    diagnostics: list[DiagnosticResult] = Field(default_factory=list)
    remediations: list[RemediationAction] = Field(default_factory=list)
    escalation_history: list[dict[str, Any]] = Field(default_factory=list)
    resolution_summary: str = ""
    timeline: list[dict[str, Any]] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class IncidentSession(BaseModel):
    """Full state of an incident response session."""

    id: str
    state: IncidentState = IncidentState.RECEIVED
    alert: Alert
    severity: SeverityLevel = SeverityLevel.P4
    context: IncidentContext | None = None
    diagnostics: list[DiagnosticResult] = Field(default_factory=list)
    remediations: list[RemediationAction] = Field(default_factory=list)
    escalation_level: EscalationLevel = EscalationLevel.L1_AUTO
    report: IncidentReport | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    error: str | None = None


# ---------------------------------------------------------------------------
# SSE event model
# ---------------------------------------------------------------------------


class IncidentEvent(BaseModel):
    """Server-Sent Event pushed during incident response workflow execution."""

    event_type: str
    session_id: str
    data: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
