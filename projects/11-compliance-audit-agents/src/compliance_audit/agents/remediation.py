"""RemediationAgent -- generates corrective action recommendations.

Maps audit findings and risk scores to structured remediation actions
with priority, estimated effort, and actionable descriptions.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import structlog

from compliance_audit.agents.base import AgentResponse, ChatAgent
from compliance_audit.models import (
    AuditFinding,
    RegulationType,
    RemediationAction,
    RiskScore,
    SeverityLevel,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Remediation templates
# ---------------------------------------------------------------------------

_REMEDIATION_TEMPLATES: dict[str, dict[str, str]] = {
    "Segregation of Duties": {
        "action": "Implement dual-control authorization workflow",
        "description": (
            "Deploy a workflow system requiring separate individuals for "
            "initiation, approval, and execution of financial transactions. "
            "Configure role-based access controls to prevent single-person "
            "end-to-end processing."
        ),
        "estimated_effort": "2-4 weeks",
    },
    "Financial Authorization Controls": {
        "action": "Deploy tiered approval matrix",
        "description": (
            "Implement an automated approval routing system with thresholds: "
            "transactions >$10K require manager approval, >$50K require "
            "director approval, >$500K require VP/CFO approval."
        ),
        "estimated_effort": "1-2 weeks",
    },
    "Financial System Access Controls": {
        "action": "Implement role-based access controls for financial systems",
        "description": (
            "Conduct access review, remove unnecessary privileges, implement "
            "RBAC with quarterly recertification. Deploy privileged access "
            "management (PAM) for sensitive financial systems."
        ),
        "estimated_effort": "3-6 weeks",
    },
    "Audit Trail Requirements": {
        "action": "Deploy comprehensive audit logging",
        "description": (
            "Implement immutable audit logging for all financial transactions "
            "with timestamps, actor identification, and complete change "
            "history. Configure log retention and integrity monitoring."
        ),
        "estimated_effort": "2-3 weeks",
    },
    "Lawful Basis for Processing (Art. 6)": {
        "action": "Implement consent management platform",
        "description": (
            "Deploy a consent management platform to collect, store, and "
            "manage user consent. Ensure all data processing activities "
            "have documented lawful basis. Implement consent audit trails."
        ),
        "estimated_effort": "4-6 weeks",
    },
    "Data Retention Limits (Art. 5(1)(e))": {
        "action": "Implement automated data lifecycle management",
        "description": (
            "Deploy data classification and retention policies with "
            "automated purging. Implement data inventory and mapping to "
            "track retention periods across all systems."
        ),
        "estimated_effort": "3-5 weeks",
    },
    "International Data Transfer (Art. 44-49)": {
        "action": "Establish data transfer compliance framework",
        "description": (
            "Implement Standard Contractual Clauses for all international "
            "data transfers. Conduct Transfer Impact Assessments. Deploy "
            "data localization controls where required."
        ),
        "estimated_effort": "4-8 weeks",
    },
    "Right to Erasure (Art. 17)": {
        "action": "Build automated erasure request workflow",
        "description": (
            "Implement a data subject request portal with automated erasure "
            "processing across all data stores. Ensure 30-day SLA with "
            "verification and confirmation workflows."
        ),
        "estimated_effort": "3-5 weeks",
    },
    "Data Security (Art. 32)": {
        "action": "Deploy comprehensive data encryption",
        "description": (
            "Implement AES-256 encryption at rest and TLS 1.3 in transit. "
            "Deploy data masking for logs and non-production environments. "
            "Conduct regular encryption key rotation."
        ),
        "estimated_effort": "2-4 weeks",
    },
    "Access Control (CC6.1) - Multi-Factor Authentication": {
        "action": "Enforce organization-wide MFA",
        "description": (
            "Deploy MFA for all user accounts with phishing-resistant "
            "methods (FIDO2/WebAuthn). Configure conditional access "
            "policies requiring MFA for sensitive resources."
        ),
        "estimated_effort": "1-3 weeks",
    },
    "Data Protection (CC6.1) - Encryption": {
        "action": "Implement encryption-everywhere policy",
        "description": (
            "Enable encryption at rest for all data stores. Enforce TLS "
            "for all network communication. Deploy certificate management "
            "and monitor for encryption gaps."
        ),
        "estimated_effort": "2-4 weeks",
    },
    "Change Management (CC8.1)": {
        "action": "Implement formal change management process",
        "description": (
            "Deploy ITIL-aligned change management with mandatory change "
            "tickets, impact assessments, peer review, and rollback plans. "
            "Configure automated change detection and alerting."
        ),
        "estimated_effort": "3-5 weeks",
    },
    "Logical Access (CC6.2) - Unauthorized Access": {
        "action": "Deploy advanced access monitoring and response",
        "description": (
            "Implement SIEM-based access monitoring with automated "
            "alerting for unauthorized access attempts. Deploy account "
            "lockout policies and implement least-privilege access model."
        ),
        "estimated_effort": "2-4 weeks",
    },
    "Incident Response (CC7.3)": {
        "action": "Establish incident response program",
        "description": (
            "Develop and document incident response playbooks. Deploy "
            "incident tracking system with automated escalation. Conduct "
            "tabletop exercises quarterly."
        ),
        "estimated_effort": "4-6 weeks",
    },
}

# Default template for unknown finding types
_DEFAULT_TEMPLATE = {
    "action": "Investigate and remediate compliance gap",
    "description": (
        "Conduct detailed investigation of the identified compliance "
        "gap. Develop and implement corrective controls. Schedule "
        "follow-up audit to verify remediation effectiveness."
    ),
    "estimated_effort": "2-4 weeks",
}


class RemediationAgent(ChatAgent):
    """Recommends remediation actions for audit findings.

    Maps findings to standardized remediation templates and adjusts
    priority based on associated risk scores.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(
            name="RemediationAgent",
            instructions=(
                "You are a compliance remediation specialist. Given audit "
                "findings and their risk scores, recommend specific, "
                "actionable remediation steps. For each finding, provide: "
                "the corrective action, priority level, estimated effort, "
                "and a detailed description of the remediation steps. "
                "Return a JSON array of remediation objects."
            ),
            model=model,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def recommend(
        self,
        findings: list[AuditFinding],
        risk_scores: list[RiskScore],
    ) -> list[RemediationAction]:
        """Generate remediation recommendations for *findings*.

        Uses *risk_scores* to prioritize the recommended actions.
        """
        if not findings:
            return []

        score_lookup = {s.finding_id: s for s in risk_scores}

        findings_data = [
            {
                "id": f.id,
                "regulation_type": f.regulation_type.value,
                "severity": f.severity.value,
                "rule_violated": f.rule_violated,
                "description": f.description,
                "risk_score": score_lookup.get(f.id, RiskScore(
                    finding_id=f.id, probability=0.5, impact=0.5, score=0.5,
                )).score,
            }
            for f in findings
        ]

        messages = [
            {
                "role": "user",
                "content": (
                    "Recommend remediation actions for these findings:\n\n"
                    + json.dumps(findings_data, indent=2)
                ),
            },
        ]

        response = await self.run(
            messages,
            context={"findings": findings, "risk_scores": risk_scores},
        )

        if response.structured_output and isinstance(response.structured_output, list):
            return self._parse_remediations(response.structured_output)

        return self._heuristic_recommend(findings, risk_scores)

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    async def _heuristic_fallback(
        self,
        messages: list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Template-based remediation recommendations."""
        findings = (context or {}).get("findings", [])
        risk_scores = (context or {}).get("risk_scores", [])
        remediations = self._heuristic_recommend(findings, risk_scores)
        output = [r.model_dump(mode="json") for r in remediations]
        return AgentResponse(
            content=json.dumps(output, indent=2),
            structured_output=output,
        )

    def _heuristic_recommend(
        self,
        findings: list[AuditFinding],
        risk_scores: list[RiskScore],
    ) -> list[RemediationAction]:
        """Deterministic template-based remediation mapping."""
        score_lookup = {s.finding_id: s for s in risk_scores}
        remediations: list[RemediationAction] = []

        for finding in findings:
            template = _REMEDIATION_TEMPLATES.get(
                finding.rule_violated, _DEFAULT_TEMPLATE
            )

            # Priority based on risk score if available, else severity
            risk = score_lookup.get(finding.id)
            if risk and risk.score >= 0.8:
                priority = SeverityLevel.CRITICAL
            elif risk and risk.score >= 0.6:
                priority = SeverityLevel.HIGH
            elif risk and risk.score >= 0.4:
                priority = SeverityLevel.MEDIUM
            elif risk:
                priority = SeverityLevel.LOW
            else:
                priority = finding.severity

            remediations.append(
                RemediationAction(
                    finding_id=finding.id,
                    action=template["action"],
                    priority=priority,
                    estimated_effort=template["estimated_effort"],
                    description=template["description"],
                )
            )

        logger.info("remediation_complete", remediations=len(remediations))
        return remediations

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_remediations(raw: list[dict[str, Any]]) -> list[RemediationAction]:
        """Parse LLM-produced remediation dicts into models."""
        remediations: list[RemediationAction] = []
        for item in raw:
            try:
                remediations.append(
                    RemediationAction(
                        finding_id=item.get("finding_id", str(uuid.uuid4())),
                        action=item.get("action", "Unknown"),
                        priority=SeverityLevel(item.get("priority", "MEDIUM")),
                        estimated_effort=item.get("estimated_effort", "unknown"),
                        description=item.get("description", ""),
                    )
                )
            except (ValueError, KeyError) as exc:
                logger.warning("remediation_parse_error", error=str(exc))
        return remediations
