"""ClassifierAgent -- routes transactions to the appropriate regulation checkers.

Uses keyword-based heuristics when no LLM is available, matching
transaction actions and resources against regulation-specific patterns.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from compliance_audit.agents.base import AgentResponse, ChatAgent
from compliance_audit.models import RegulationType, TransactionLog

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Keyword sets for heuristic classification
# ---------------------------------------------------------------------------

_SOX_KEYWORDS = {
    "financial",
    "audit",
    "controls",
    "accounting",
    "ledger",
    "revenue",
    "expense",
    "invoice",
    "payment",
    "approve",
    "authorize",
    "journal",
    "reconciliation",
    "fiscal",
    "treasury",
    "budget",
    "procurement",
    "disbursement",
    "segregation",
    "material",
}

_GDPR_KEYWORDS = {
    "data",
    "privacy",
    "consent",
    "personal",
    "pii",
    "retention",
    "erasure",
    "delete",
    "subject",
    "gdpr",
    "transfer",
    "cross-border",
    "processing",
    "controller",
    "processor",
    "opt-in",
    "opt-out",
    "cookie",
    "tracking",
    "profiling",
}

_SOC2_KEYWORDS = {
    "access",
    "security",
    "availability",
    "encryption",
    "mfa",
    "firewall",
    "vulnerability",
    "patch",
    "incident",
    "monitoring",
    "logging",
    "change",
    "configuration",
    "backup",
    "disaster",
    "recovery",
    "authentication",
    "authorization",
    "penetration",
    "scan",
}


class ClassifierAgent(ChatAgent):
    """Classifies transactions by applicable compliance regulation type.

    Each transaction may map to one or more regulation types.  The
    classifier dispatches transactions to the appropriate downstream
    checker agents.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(
            name="ClassifierAgent",
            instructions=(
                "You are an expert compliance analyst. Given a list of "
                "transaction logs, classify each transaction by the "
                "regulation types it should be checked against: SOX "
                "(financial controls), GDPR (data privacy), or SOC2 "
                "(security and availability). A transaction may belong to "
                "multiple categories. Return a JSON object mapping "
                "regulation type to a list of transaction IDs.\n\n"
                "Example output:\n"
                '{"SOX": ["tx-1","tx-3"], "GDPR": ["tx-2","tx-3"], '
                '"SOC2": ["tx-4"]}'
            ),
            model=model,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def classify(
        self,
        transactions: list[TransactionLog],
    ) -> dict[RegulationType, list[TransactionLog]]:
        """Classify *transactions* by regulation type.

        Returns a mapping of ``RegulationType`` to the transactions that
        should be audited under that regulation.
        """
        tx_summaries = [
            {
                "id": tx.id,
                "actor": tx.actor,
                "action": tx.action,
                "resource": tx.resource,
                "department": tx.department,
                "metadata": tx.metadata,
            }
            for tx in transactions
        ]

        messages = [
            {
                "role": "user",
                "content": (
                    "Classify these transactions by regulation type "
                    "(SOX, GDPR, SOC2):\n\n"
                    + json.dumps(tx_summaries, indent=2)
                ),
            },
        ]

        response = await self.run(
            messages,
            context={"transactions": transactions},
        )

        # Parse LLM response if structured output is available
        if response.structured_output:
            return self._parse_classification(response.structured_output, transactions)

        # Fall through to heuristic result (already computed in fallback)
        return self._heuristic_classify(transactions)

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    async def _heuristic_fallback(
        self,
        messages: list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Keyword-based classification fallback."""
        transactions = (context or {}).get("transactions", [])
        result = self._heuristic_classify(transactions)
        output = {
            rt.value: [tx.id for tx in txs]
            for rt, txs in result.items()
        }
        return AgentResponse(
            content=json.dumps(output, indent=2),
            structured_output=output,
        )

    def _heuristic_classify(
        self,
        transactions: list[TransactionLog],
    ) -> dict[RegulationType, list[TransactionLog]]:
        """Deterministic keyword-based classification."""
        result: dict[RegulationType, list[TransactionLog]] = {
            RegulationType.SOX: [],
            RegulationType.GDPR: [],
            RegulationType.SOC2: [],
        }

        for tx in transactions:
            text = (
                f"{tx.action} {tx.resource} {tx.department} "
                f"{json.dumps(tx.metadata)}"
            ).lower()

            matched = False

            if any(kw in text for kw in _SOX_KEYWORDS):
                result[RegulationType.SOX].append(tx)
                matched = True

            if any(kw in text for kw in _GDPR_KEYWORDS):
                result[RegulationType.GDPR].append(tx)
                matched = True

            if any(kw in text for kw in _SOC2_KEYWORDS):
                result[RegulationType.SOC2].append(tx)
                matched = True

            # Unmatched transactions go to all checkers for safety
            if not matched:
                result[RegulationType.SOX].append(tx)
                result[RegulationType.GDPR].append(tx)
                result[RegulationType.SOC2].append(tx)

        logger.info(
            "classification_complete",
            sox=len(result[RegulationType.SOX]),
            gdpr=len(result[RegulationType.GDPR]),
            soc2=len(result[RegulationType.SOC2]),
        )
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _parse_classification(
        self,
        output: dict[str, Any],
        transactions: list[TransactionLog],
    ) -> dict[RegulationType, list[TransactionLog]]:
        """Convert LLM-produced ID mapping back to transaction objects."""
        tx_lookup = {tx.id: tx for tx in transactions}
        result: dict[RegulationType, list[TransactionLog]] = {
            RegulationType.SOX: [],
            RegulationType.GDPR: [],
            RegulationType.SOC2: [],
        }

        for reg_key in ("SOX", "GDPR", "SOC2"):
            reg_type = RegulationType(reg_key)
            tx_ids = output.get(reg_key, [])
            for tx_id in tx_ids:
                if tx_id in tx_lookup:
                    result[reg_type].append(tx_lookup[tx_id])

        return result
