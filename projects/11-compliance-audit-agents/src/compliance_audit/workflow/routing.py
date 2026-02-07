"""Type-based routing functions for the audit workflow graph.

Selects which compliance checker agents should run based on the
regulation types identified during the classification step.
"""

from __future__ import annotations

from typing import Any

import structlog

from compliance_audit.models import RegulationType
from compliance_audit.workflow.state import AuditGraphState

logger = structlog.get_logger(__name__)


def determine_checkers(state: AuditGraphState) -> list[str]:
    """Return the list of checker node names to execute.

    Examines ``state["regulation_types_found"]`` and maps each
    regulation type to the corresponding graph node name.
    """
    reg_types = state.get("regulation_types_found", [])

    checkers: list[str] = []

    if RegulationType.SOX.value in reg_types:
        checkers.append("check_sox")
    if RegulationType.GDPR.value in reg_types:
        checkers.append("check_gdpr")
    if RegulationType.SOC2.value in reg_types:
        checkers.append("check_soc2")

    # If no specific types found, run all checkers for safety
    if not checkers:
        checkers = ["check_sox", "check_gdpr", "check_soc2"]

    logger.info(
        "routing_determined",
        regulation_types=reg_types,
        checkers=checkers,
    )
    return checkers


def should_await_approval(state: AuditGraphState) -> bool:
    """Decide whether human approval is required before finalization.

    Returns ``True`` if there are findings that need review.
    """
    findings = state.get("all_findings", [])
    return len(findings) > 0


def route_after_classification(state: AuditGraphState) -> str:
    """Route to the checker dispatch node or skip to finalization.

    If no transactions were classified, skip directly to finalization.
    """
    classified = state.get("classified", {})
    has_transactions = any(len(txs) > 0 for txs in classified.values())

    if has_transactions:
        return "route_to_checkers"

    logger.info("no_transactions_classified", action="skip_to_finalize")
    return "finalize"


def route_after_scoring(state: AuditGraphState) -> str:
    """Route after risk scoring -- either to remediation or finalize.

    If there are scored findings, route to remediation generation.
    """
    risk_scores = state.get("risk_scores", [])
    if risk_scores:
        return "generate_remediations"
    return "finalize"
