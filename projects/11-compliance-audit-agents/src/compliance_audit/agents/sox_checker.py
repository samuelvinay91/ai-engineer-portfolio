"""SOXCheckerAgent -- Sarbanes-Oxley compliance checks.

Examines transactions for SOX violations including segregation of
duties breaches, unauthorized financial access, financial irregularities,
and audit trail gaps.
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

# ---------------------------------------------------------------------------
# SOX rule thresholds
# ---------------------------------------------------------------------------

_HIGH_VALUE_THRESHOLD = 50_000.0
_CRITICAL_VALUE_THRESHOLD = 500_000.0


class SOXCheckerAgent(ChatAgent):
    """Checks transactions for Sarbanes-Oxley compliance violations.

    Looks for:
    - Segregation of duties violations (same person approves and executes)
    - Unauthorized access to financial systems
    - Financial irregularities (unusual amounts, patterns)
    - Audit trail gaps (missing approvals, timestamps)
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(
            name="SOXCheckerAgent",
            instructions=(
                "You are a SOX compliance auditor. Examine the provided "
                "financial transactions and policy rules. Identify violations "
                "of Sarbanes-Oxley requirements including: segregation of "
                "duties, unauthorized financial access, financial "
                "irregularities, and audit trail gaps. For each finding, "
                "provide the severity (CRITICAL/HIGH/MEDIUM/LOW/INFO), the "
                "rule violated, evidence from the transaction, and a "
                "remediation recommendation. Return findings as a JSON array."
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
        """Run SOX compliance checks on *transactions* against *policies*.

        Returns a list of :class:`AuditFinding` objects for any detected
        violations.
        """
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
            if p.regulation_type == RegulationType.SOX:
                policy_rules.extend(p.rules)

        messages = [
            {
                "role": "user",
                "content": (
                    "Check these transactions for SOX compliance violations.\n\n"
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

        # Heuristic already returns findings via context
        return self._heuristic_check(transactions, policies)

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    async def _heuristic_fallback(
        self,
        messages: list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Rule-based SOX compliance checking."""
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
        """Deterministic rule-based SOX checks."""
        findings: list[AuditFinding] = []

        # Build actor-action pairs for segregation of duties checks
        actor_actions: dict[str, set[str]] = {}
        for tx in transactions:
            actor_actions.setdefault(tx.actor, set()).add(tx.action.lower())

        for tx in transactions:
            text = f"{tx.action} {tx.resource} {json.dumps(tx.metadata)}".lower()

            # Check 1: Segregation of duties -- same actor both approves
            # and executes financial transactions
            actor_ops = actor_actions.get(tx.actor, set())
            has_approve = any(
                op in actor_ops for op in ("approve", "authorize", "sign-off")
            )
            has_execute = any(
                op in actor_ops
                for op in ("execute", "process", "submit", "transfer", "disburse")
            )
            if has_approve and has_execute and tx.action.lower() in (
                "approve",
                "authorize",
                "sign-off",
                "execute",
                "process",
                "submit",
                "transfer",
                "disburse",
            ):
                findings.append(
                    AuditFinding(
                        id=str(uuid.uuid4()),
                        regulation_type=RegulationType.SOX,
                        severity=SeverityLevel.CRITICAL,
                        rule_violated="Segregation of Duties",
                        evidence=(
                            f"Actor '{tx.actor}' both approves and executes "
                            f"financial transactions (action: {tx.action})"
                        ),
                        transaction_id=tx.id,
                        description=(
                            "The same individual is performing both "
                            "approval and execution of financial "
                            "transactions, violating segregation of duties."
                        ),
                        recommendation=(
                            "Implement dual-control procedures requiring "
                            "separate individuals for approval and execution."
                        ),
                    )
                )

            # Check 2: High-value transaction without approval
            amount = tx.metadata.get("amount", 0)
            try:
                amount = float(amount)
            except (ValueError, TypeError):
                amount = 0

            approval = tx.metadata.get("approved_by", tx.metadata.get("approver", ""))
            if amount > _HIGH_VALUE_THRESHOLD and not approval:
                severity = (
                    SeverityLevel.CRITICAL
                    if amount > _CRITICAL_VALUE_THRESHOLD
                    else SeverityLevel.HIGH
                )
                findings.append(
                    AuditFinding(
                        id=str(uuid.uuid4()),
                        regulation_type=RegulationType.SOX,
                        severity=severity,
                        rule_violated="Financial Authorization Controls",
                        evidence=(
                            f"Transaction of ${amount:,.2f} by '{tx.actor}' "
                            f"has no approval record."
                        ),
                        transaction_id=tx.id,
                        description=(
                            f"High-value financial transaction (${amount:,.2f}) "
                            "processed without documented approval."
                        ),
                        recommendation=(
                            "Require managerial approval for all transactions "
                            f"exceeding ${_HIGH_VALUE_THRESHOLD:,.0f}."
                        ),
                    )
                )

            # Check 3: Unauthorized access to financial systems
            if "unauthorized" in text or "denied" in text:
                findings.append(
                    AuditFinding(
                        id=str(uuid.uuid4()),
                        regulation_type=RegulationType.SOX,
                        severity=SeverityLevel.HIGH,
                        rule_violated="Financial System Access Controls",
                        evidence=(
                            f"Unauthorized access attempt by '{tx.actor}' "
                            f"on resource '{tx.resource}'"
                        ),
                        transaction_id=tx.id,
                        description=(
                            "An unauthorized access attempt to financial "
                            "systems was detected."
                        ),
                        recommendation=(
                            "Review access permissions and implement role-based "
                            "access controls for financial systems."
                        ),
                    )
                )

            # Check 4: Missing audit trail
            if not tx.metadata.get("audit_trail") and "financial" in text:
                has_timestamp = bool(tx.timestamp)
                if not has_timestamp or "ledger" in text or "journal" in text:
                    findings.append(
                        AuditFinding(
                            id=str(uuid.uuid4()),
                            regulation_type=RegulationType.SOX,
                            severity=SeverityLevel.MEDIUM,
                            rule_violated="Audit Trail Requirements",
                            evidence=(
                                f"Transaction by '{tx.actor}' on "
                                f"'{tx.resource}' lacks complete audit trail."
                            ),
                            transaction_id=tx.id,
                            description=(
                                "Financial transaction is missing required "
                                "audit trail documentation."
                            ),
                            recommendation=(
                                "Ensure all financial transactions include "
                                "complete audit trail with timestamps, actors, "
                                "and approval chains."
                            ),
                        )
                    )

        logger.info("sox_check_complete", findings=len(findings))
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
                        regulation_type=RegulationType.SOX,
                        severity=SeverityLevel(item.get("severity", "MEDIUM")),
                        rule_violated=item.get("rule_violated", "Unknown"),
                        evidence=item.get("evidence", ""),
                        transaction_id=item.get("transaction_id", ""),
                        description=item.get("description", ""),
                        recommendation=item.get("recommendation", ""),
                    )
                )
            except (ValueError, KeyError) as exc:
                logger.warning("sox_finding_parse_error", error=str(exc))
        return findings
