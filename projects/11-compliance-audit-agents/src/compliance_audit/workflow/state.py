"""Workflow graph state definition.

Defines the typed state dictionary that flows through the audit
workflow graph nodes.
"""

from __future__ import annotations

from typing import Any, Optional, TypedDict

from compliance_audit.models import (
    AuditFinding,
    AuditReport,
    AuditSessionState,
    PolicyDocument,
    RegulationType,
    RemediationAction,
    RiskScore,
    TransactionLog,
)


class AuditGraphState(TypedDict, total=False):
    """State passed between audit workflow graph nodes.

    All fields are optional (``total=False``) so nodes can populate
    them incrementally.
    """

    # Session identity
    session_id: str

    # Current lifecycle phase
    current_state: AuditSessionState

    # Inputs
    transactions: list[TransactionLog]
    policies: list[PolicyDocument]

    # Classification results
    classified: dict[str, list[TransactionLog]]  # RegulationType.value -> transactions
    regulation_types_found: list[str]  # list of RegulationType.value strings

    # Checker results
    sox_findings: list[AuditFinding]
    gdpr_findings: list[AuditFinding]
    soc2_findings: list[AuditFinding]
    all_findings: list[AuditFinding]

    # Risk scoring
    risk_scores: list[RiskScore]

    # Remediation
    remediations: list[RemediationAction]

    # Approval gate
    approved: Optional[bool]

    # Final report
    report: Optional[AuditReport]

    # Error state
    error: Optional[str]

    # Metadata
    messages: list[dict[str, Any]]
