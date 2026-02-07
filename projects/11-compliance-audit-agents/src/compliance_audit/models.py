"""Pydantic models for the Compliance Audit Agents service.

Defines all domain objects: policy documents, transaction logs, audit
findings, risk scores, remediation actions, reports, sessions, and
streaming events.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class RegulationType(str, Enum):
    """Supported compliance regulation frameworks."""

    SOX = "SOX"
    GDPR = "GDPR"
    SOC2 = "SOC2"


class SeverityLevel(str, Enum):
    """Finding severity levels (highest to lowest)."""

    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class AuditSessionState(str, Enum):
    """Lifecycle states for an audit session."""

    INGESTING = "INGESTING"
    CLASSIFYING = "CLASSIFYING"
    CHECKING = "CHECKING"
    SCORING = "SCORING"
    REMEDIATING = "REMEDIATING"
    AWAITING_APPROVAL = "AWAITING_APPROVAL"
    APPROVED = "APPROVED"
    REJECTED = "REJECTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


# ---------------------------------------------------------------------------
# Domain models
# ---------------------------------------------------------------------------


class PolicyDocument(BaseModel):
    """A compliance policy document containing rules to check against."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    regulation_type: RegulationType
    content: str
    rules: list[str] = Field(default_factory=list)


class TransactionLog(BaseModel):
    """A single auditable transaction or activity record."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    actor: str
    action: str
    resource: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    department: str = "unknown"


class AuditFinding(BaseModel):
    """A compliance violation or observation discovered during an audit."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    regulation_type: RegulationType
    severity: SeverityLevel
    rule_violated: str
    evidence: str
    transaction_id: str
    description: str
    recommendation: str = ""


class RiskScore(BaseModel):
    """Quantified risk assessment for a single audit finding."""

    finding_id: str
    probability: float = Field(ge=0.0, le=1.0)
    impact: float = Field(ge=0.0, le=1.0)
    score: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class RemediationAction(BaseModel):
    """A recommended corrective action for a finding."""

    finding_id: str
    action: str
    priority: SeverityLevel = SeverityLevel.MEDIUM
    estimated_effort: str = "unknown"
    description: str = ""


class AuditReport(BaseModel):
    """Full audit report aggregating findings, scores, and remediations."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    findings: list[AuditFinding] = Field(default_factory=list)
    risk_scores: list[RiskScore] = Field(default_factory=list)
    remediations: list[RemediationAction] = Field(default_factory=list)
    overall_risk_level: SeverityLevel = SeverityLevel.INFO
    summary: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))


class AuditSession(BaseModel):
    """Top-level audit session tracking state and results."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    state: AuditSessionState = AuditSessionState.INGESTING
    transactions: list[TransactionLog] = Field(default_factory=list)
    findings: list[AuditFinding] = Field(default_factory=list)
    report: Optional[AuditReport] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    error: Optional[str] = None


class AuditEvent(BaseModel):
    """A real-time event emitted during audit workflow execution."""

    event_type: str
    session_id: str
    data: dict[str, Any] = Field(default_factory=dict)
    message: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
