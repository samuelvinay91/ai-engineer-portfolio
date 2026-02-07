"""SOC2CheckerAgent -- SOC 2 Trust Services Criteria compliance checks.

Examines transactions for SOC 2 violations including access control
failures, encryption gaps, availability monitoring issues, change
management gaps, and incident response deficiencies.
"""

from __future__ import annotations

import json
import uuid
from typing import Any

import structlog

from compliance_audit.agents.base import AgentResponse, ChatAgent
from compliance_audit.models import (
    AuditFinding,
    PolicyDocument,
    RegulationType,
    SeverityLevel,
    TransactionLog,
)

logger = structlog.get_logger(__name__)


class SOC2CheckerAgent(ChatAgent):
    """Checks transactions for SOC 2 compliance violations.

    Looks for:
    - Access control failures (missing MFA, unauthorized access)
    - Encryption gaps (unencrypted data storage or transmission)
    - Availability monitoring issues (missing health checks, SLA breaches)
    - Change management gaps (unauthorized configuration changes)
    - Incident response deficiencies (unlogged incidents, missing playbooks)
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(
            name="SOC2CheckerAgent",
            instructions=(
                "You are a SOC 2 compliance auditor. Examine the provided "
                "transaction logs and policy rules. Identify violations of "
                "SOC 2 Trust Services Criteria including: access control "
                "(CC6), encryption and data protection (CC6.1), availability "
                "(A1), change management (CC8), and incident response "
                "(CC7). For each finding, provide severity, rule violated, "
                "evidence, and remediation. Return findings as a JSON array."
            ),
            model=model,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def check(
        self,
        transactions: list[TransactionLog],
        policies: list[PolicyDocument],
    ) -> list[AuditFinding]:
        """Run SOC 2 compliance checks on *transactions*."""
        if not transactions:
            return []

        tx_data = [
            {
                "id": tx.id,
                "actor": tx.actor,
                "action": tx.action,
                "resource": tx.resource,
                "department": tx.department,
                "metadata": tx.metadata,
                "timestamp": tx.timestamp.isoformat(),
            }
            for tx in transactions
        ]

        policy_rules = []
        for p in policies:
            if p.regulation_type == RegulationType.SOC2:
                policy_rules.extend(p.rules)

        messages = [
            {
                "role": "user",
                "content": (
                    "Check these transactions for SOC 2 compliance violations.\n\n"
                    f"Policy rules:\n{json.dumps(policy_rules, indent=2)}\n\n"
                    f"Transactions:\n{json.dumps(tx_data, indent=2)}"
                ),
            },
        ]

        response = await self.run(
            messages,
            context={"transactions": transactions, "policies": policies},
        )

        if response.structured_output and isinstance(response.structured_output, list):
            return self._parse_findings(response.structured_output)

        return self._heuristic_check(transactions, policies)

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    async def _heuristic_fallback(
        self,
        messages: list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Rule-based SOC 2 compliance checking."""
        transactions = (context or {}).get("transactions", [])
        policies = (context or {}).get("policies", [])
        findings = self._heuristic_check(transactions, policies)
        output = [f.model_dump(mode="json") for f in findings]
        return AgentResponse(
            content=json.dumps(output, indent=2),
            structured_output=output,
        )

    def _heuristic_check(
        self,
        transactions: list[TransactionLog],
        policies: list[PolicyDocument],
    ) -> list[AuditFinding]:
        """Deterministic rule-based SOC 2 checks."""
        findings: list[AuditFinding] = []

        for tx in transactions:
            text = f"{tx.action} {tx.resource} {json.dumps(tx.metadata)}".lower()

            # Check 1: Missing MFA / weak authentication
            mfa_enabled = tx.metadata.get("mfa", tx.metadata.get("mfa_enabled", True))
            auth_method = str(tx.metadata.get("auth_method", "")).lower()
            if (
                mfa_enabled is False
                or auth_method in ("password", "basic", "none")
                or "without mfa" in text
                or "no mfa" in text
            ):
                findings.append(
                    AuditFinding(
                        id=str(uuid.uuid4()),
                        regulation_type=RegulationType.SOC2,
                        severity=SeverityLevel.CRITICAL,
                        rule_violated="Access Control (CC6.1) - Multi-Factor Authentication",
                        evidence=(
                            f"Access by '{tx.actor}' to '{tx.resource}' "
                            f"without MFA (auth method: {auth_method or 'unknown'})."
                        ),
                        transaction_id=tx.id,
                        description=(
                            "System access occurred without multi-factor "
                            "authentication, violating SOC 2 access control "
                            "requirements."
                        ),
                        recommendation=(
                            "Enable and enforce MFA for all system access, "
                            "especially for privileged accounts and "
                            "sensitive resources."
                        ),
                    )
                )

            # Check 2: Unencrypted data storage or transmission
            encrypted = tx.metadata.get("encrypted", tx.metadata.get("encryption", True))
            if (
                encrypted is False
                or "unencrypted" in text
                or "plaintext" in text
                or "no encryption" in text
            ):
                findings.append(
                    AuditFinding(
                        id=str(uuid.uuid4()),
                        regulation_type=RegulationType.SOC2,
                        severity=SeverityLevel.HIGH,
                        rule_violated="Data Protection (CC6.1) - Encryption",
                        evidence=(
                            f"Unencrypted data detected: '{tx.action}' "
                            f"on '{tx.resource}' by '{tx.actor}'."
                        ),
                        transaction_id=tx.id,
                        description=(
                            "Data was stored or transmitted without "
                            "encryption, exposing it to unauthorized access."
                        ),
                        recommendation=(
                            "Implement AES-256 encryption at rest and TLS 1.3 "
                            "in transit for all sensitive data."
                        ),
                    )
                )

            # Check 3: Unauthorized configuration changes
            is_config_change = any(
                kw in text
                for kw in (
                    "config",
                    "configuration",
                    "setting",
                    "firewall",
                    "rule change",
                    "policy change",
                )
            )
            has_change_ticket = tx.metadata.get(
                "change_ticket", tx.metadata.get("ticket_id", "")
            )
            has_approval = tx.metadata.get(
                "approved_by", tx.metadata.get("approver", "")
            )
            if is_config_change and not has_change_ticket and not has_approval:
                findings.append(
                    AuditFinding(
                        id=str(uuid.uuid4()),
                        regulation_type=RegulationType.SOC2,
                        severity=SeverityLevel.HIGH,
                        rule_violated="Change Management (CC8.1)",
                        evidence=(
                            f"Configuration change by '{tx.actor}' on "
                            f"'{tx.resource}' without change ticket or approval."
                        ),
                        transaction_id=tx.id,
                        description=(
                            "A system configuration change was made without "
                            "following the change management process."
                        ),
                        recommendation=(
                            "Require change tickets and peer approval for "
                            "all configuration changes. Implement automated "
                            "change detection and alerting."
                        ),
                    )
                )

            # Check 4: Unauthorized access attempts
            if (
                "unauthorized" in text
                or "access denied" in text
                or "forbidden" in text
                or "privilege escalation" in text
            ):
                findings.append(
                    AuditFinding(
                        id=str(uuid.uuid4()),
                        regulation_type=RegulationType.SOC2,
                        severity=SeverityLevel.HIGH,
                        rule_violated="Logical Access (CC6.2) - Unauthorized Access",
                        evidence=(
                            f"Unauthorized access attempt by '{tx.actor}' "
                            f"to '{tx.resource}': {tx.action}."
                        ),
                        transaction_id=tx.id,
                        description=(
                            "An unauthorized access attempt or privilege "
                            "escalation was detected."
                        ),
                        recommendation=(
                            "Implement least-privilege access controls, "
                            "automated access reviews, and real-time "
                            "alerting for unauthorized access attempts."
                        ),
                    )
                )

            # Check 5: Missing incident logging
            is_incident = any(
                kw in text
                for kw in (
                    "incident",
                    "breach",
                    "outage",
                    "failure",
                    "vulnerability",
                    "exploit",
                )
            )
            has_incident_id = tx.metadata.get(
                "incident_id", tx.metadata.get("ticket", "")
            )
            if is_incident and not has_incident_id:
                findings.append(
                    AuditFinding(
                        id=str(uuid.uuid4()),
                        regulation_type=RegulationType.SOC2,
                        severity=SeverityLevel.MEDIUM,
                        rule_violated="Incident Response (CC7.3)",
                        evidence=(
                            f"Security incident involving '{tx.resource}' "
                            f"by '{tx.actor}' lacks incident tracking ID."
                        ),
                        transaction_id=tx.id,
                        description=(
                            "A security incident was detected without "
                            "proper incident response tracking."
                        ),
                        recommendation=(
                            "Log all security incidents in the incident "
                            "management system with unique tracking IDs "
                            "and follow established response playbooks."
                        ),
                    )
                )

        logger.info("soc2_check_complete", findings=len(findings))
        return findings

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_findings(raw: list[dict[str, Any]]) -> list[AuditFinding]:
        """Parse LLM-produced finding dicts into models."""
        findings: list[AuditFinding] = []
        for item in raw:
            try:
                findings.append(
                    AuditFinding(
                        regulation_type=RegulationType.SOC2,
                        severity=SeverityLevel(item.get("severity", "MEDIUM")),
                        rule_violated=item.get("rule_violated", "Unknown"),
                        evidence=item.get("evidence", ""),
                        transaction_id=item.get("transaction_id", ""),
                        description=item.get("description", ""),
                        recommendation=item.get("recommendation", ""),
                    )
                )
            except (ValueError, KeyError) as exc:
                logger.warning("soc2_finding_parse_error", error=str(exc))
        return findings
