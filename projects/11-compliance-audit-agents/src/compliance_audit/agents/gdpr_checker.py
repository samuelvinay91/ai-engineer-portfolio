"""GDPRCheckerAgent -- General Data Protection Regulation compliance checks.

Examines transactions for GDPR violations including consent issues,
data retention policy breaches, cross-border data transfers, right-to-
delete compliance, and PII exposure.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import datetime, timedelta, timezone
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

# ---------------------------------------------------------------------------
# GDPR heuristic patterns
# ---------------------------------------------------------------------------

_PII_PATTERNS = {
    "email": re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "phone": re.compile(r"\b\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "credit_card": re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"),
}

_NON_EU_COUNTRIES = {
    "us", "usa", "united states", "china", "russia", "india", "brazil",
    "japan", "australia", "canada", "south korea", "mexico",
}

_MAX_RETENTION_DAYS = 365


class GDPRCheckerAgent(ChatAgent):
    """Checks transactions for GDPR compliance violations.

    Looks for:
    - Consent violations (processing without documented consent)
    - Data retention breaches (keeping data beyond retention period)
    - Cross-border data transfers (transfers to non-EU/EEA countries)
    - Right-to-delete violations (failure to honor erasure requests)
    - PII exposure (personal data in logs, unencrypted channels)
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(
            name="GDPRCheckerAgent",
            instructions=(
                "You are a GDPR compliance auditor. Examine the provided "
                "transaction logs and policy rules. Identify violations of "
                "the General Data Protection Regulation including: consent "
                "violations, data retention breaches, cross-border transfers "
                "to non-adequate countries, right-to-delete failures, and "
                "PII exposure. For each finding, provide the severity, rule "
                "violated, evidence, and remediation recommendation. Return "
                "findings as a JSON array."
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
        """Run GDPR compliance checks on *transactions*."""
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
            if p.regulation_type == RegulationType.GDPR:
                policy_rules.extend(p.rules)

        messages = [
            {
                "role": "user",
                "content": (
                    "Check these transactions for GDPR compliance violations.\n\n"
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
        """Pattern-based GDPR compliance checking."""
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
        """Deterministic pattern-based GDPR checks."""
        findings: list[AuditFinding] = []

        for tx in transactions:
            text = f"{tx.action} {tx.resource} {json.dumps(tx.metadata)}".lower()

            # Check 1: Processing personal data without consent
            involves_personal = any(
                kw in text
                for kw in ("personal", "pii", "user data", "customer data", "profile")
            )
            has_consent = tx.metadata.get("consent", False) or "consent" in text
            if involves_personal and not has_consent:
                findings.append(
                    AuditFinding(
                        id=str(uuid.uuid4()),
                        regulation_type=RegulationType.GDPR,
                        severity=SeverityLevel.CRITICAL,
                        rule_violated="Lawful Basis for Processing (Art. 6)",
                        evidence=(
                            f"Personal data processing by '{tx.actor}' "
                            f"on '{tx.resource}' without documented consent."
                        ),
                        transaction_id=tx.id,
                        description=(
                            "Personal data is being processed without a "
                            "documented lawful basis or consent record."
                        ),
                        recommendation=(
                            "Implement consent management and ensure all "
                            "personal data processing has a documented "
                            "lawful basis under GDPR Article 6."
                        ),
                    )
                )

            # Check 2: Data retention violations
            retention_days = tx.metadata.get("retention_days")
            if retention_days is not None:
                try:
                    retention_days = int(retention_days)
                except (ValueError, TypeError):
                    retention_days = 0
                if retention_days > _MAX_RETENTION_DAYS:
                    findings.append(
                        AuditFinding(
                            id=str(uuid.uuid4()),
                            regulation_type=RegulationType.GDPR,
                            severity=SeverityLevel.HIGH,
                            rule_violated="Data Retention Limits (Art. 5(1)(e))",
                            evidence=(
                                f"Data retention set to {retention_days} days, "
                                f"exceeding maximum of {_MAX_RETENTION_DAYS} days."
                            ),
                            transaction_id=tx.id,
                            description=(
                                f"Data retention period ({retention_days} days) "
                                f"exceeds the configured maximum of "
                                f"{_MAX_RETENTION_DAYS} days."
                            ),
                            recommendation=(
                                "Review and reduce data retention periods to "
                                "comply with the principle of storage limitation."
                            ),
                        )
                    )

            # Check 3: Cross-border data transfers
            destination = tx.metadata.get(
                "destination_country",
                tx.metadata.get("transfer_destination", ""),
            )
            if isinstance(destination, str) and destination.lower() in _NON_EU_COUNTRIES:
                findings.append(
                    AuditFinding(
                        id=str(uuid.uuid4()),
                        regulation_type=RegulationType.GDPR,
                        severity=SeverityLevel.HIGH,
                        rule_violated="International Data Transfer (Art. 44-49)",
                        evidence=(
                            f"Data transfer to '{destination}' by "
                            f"'{tx.actor}' without adequacy decision "
                            "or appropriate safeguards."
                        ),
                        transaction_id=tx.id,
                        description=(
                            f"Personal data transferred to {destination}, "
                            "which may lack adequate data protection."
                        ),
                        recommendation=(
                            "Implement Standard Contractual Clauses (SCCs) "
                            "or obtain an adequacy decision before "
                            "transferring personal data internationally."
                        ),
                    )
                )

            # Check 4: Right-to-delete violations
            if "erasure" in text or "delete" in text or "forget" in text:
                completed = tx.metadata.get("completed", tx.metadata.get("status", ""))
                if str(completed).lower() not in ("true", "completed", "done"):
                    findings.append(
                        AuditFinding(
                            id=str(uuid.uuid4()),
                            regulation_type=RegulationType.GDPR,
                            severity=SeverityLevel.HIGH,
                            rule_violated="Right to Erasure (Art. 17)",
                            evidence=(
                                f"Erasure request for '{tx.resource}' "
                                f"by '{tx.actor}' not completed."
                            ),
                            transaction_id=tx.id,
                            description=(
                                "A data subject erasure request has not "
                                "been fulfilled within the required timeframe."
                            ),
                            recommendation=(
                                "Implement automated erasure workflows and "
                                "ensure requests are fulfilled within 30 days."
                            ),
                        )
                    )

            # Check 5: PII exposure in unencrypted channels
            metadata_str = json.dumps(tx.metadata)
            for pii_type, pattern in _PII_PATTERNS.items():
                if pattern.search(metadata_str):
                    encrypted = tx.metadata.get("encrypted", False)
                    if not encrypted:
                        findings.append(
                            AuditFinding(
                                id=str(uuid.uuid4()),
                                regulation_type=RegulationType.GDPR,
                                severity=SeverityLevel.CRITICAL,
                                rule_violated="Data Security (Art. 32)",
                                evidence=(
                                    f"Unencrypted {pii_type} detected in "
                                    f"transaction metadata for '{tx.resource}'."
                                ),
                                transaction_id=tx.id,
                                description=(
                                    f"PII ({pii_type}) found in transaction "
                                    "metadata without encryption."
                                ),
                                recommendation=(
                                    "Encrypt all personal data at rest and "
                                    "in transit. Implement data masking for "
                                    "logs and audit trails."
                                ),
                            )
                        )
                    break  # One PII finding per transaction is sufficient

        logger.info("gdpr_check_complete", findings=len(findings))
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
                        regulation_type=RegulationType.GDPR,
                        severity=SeverityLevel(item.get("severity", "MEDIUM")),
                        rule_violated=item.get("rule_violated", "Unknown"),
                        evidence=item.get("evidence", ""),
                        transaction_id=item.get("transaction_id", ""),
                        description=item.get("description", ""),
                        recommendation=item.get("recommendation", ""),
                    )
                )
            except (ValueError, KeyError) as exc:
                logger.warning("gdpr_finding_parse_error", error=str(exc))
        return findings
