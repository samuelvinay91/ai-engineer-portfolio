"""RiskScorerAgent -- quantifies risk for audit findings.

Uses a severity-based probability/impact matrix with regulation weights
to produce a normalized risk score for each finding.
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
    RiskScore,
    SeverityLevel,
)

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Risk scoring matrices
# ---------------------------------------------------------------------------

_SEVERITY_PROBABILITY: dict[SeverityLevel, float] = {
    SeverityLevel.CRITICAL: 0.90,
    SeverityLevel.HIGH: 0.70,
    SeverityLevel.MEDIUM: 0.50,
    SeverityLevel.LOW: 0.30,
    SeverityLevel.INFO: 0.10,
}

_SEVERITY_IMPACT: dict[SeverityLevel, float] = {
    SeverityLevel.CRITICAL: 0.95,
    SeverityLevel.HIGH: 0.75,
    SeverityLevel.MEDIUM: 0.50,
    SeverityLevel.LOW: 0.25,
    SeverityLevel.INFO: 0.10,
}

_REGULATION_WEIGHT: dict[RegulationType, float] = {
    RegulationType.SOX: 1.0,
    RegulationType.GDPR: 0.95,
    RegulationType.SOC2: 0.85,
}


class RiskScorerAgent(ChatAgent):
    """Scores audit findings by probability, impact, and regulation weight.

    Produces a normalized risk score in [0, 1] for each finding to help
    prioritize remediation efforts.
    """

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        super().__init__(
            name="RiskScorerAgent",
            instructions=(
                "You are a risk assessment specialist. Given a list of "
                "audit findings with their severity levels and regulation "
                "types, calculate a risk score for each finding. Consider "
                "the probability of the issue causing harm, the potential "
                "impact, and the regulatory weight. Return a JSON array of "
                "risk score objects with: finding_id, probability (0-1), "
                "impact (0-1), score (0-1), and rationale."
            ),
            model=model,
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def score(
        self,
        findings: list[AuditFinding],
    ) -> list[RiskScore]:
        """Score *findings* and return a list of :class:`RiskScore`."""
        if not findings:
            return []

        findings_data = [
            {
                "id": f.id,
                "regulation_type": f.regulation_type.value,
                "severity": f.severity.value,
                "rule_violated": f.rule_violated,
                "description": f.description,
            }
            for f in findings
        ]

        messages = [
            {
                "role": "user",
                "content": (
                    "Score the risk for each audit finding:\n\n"
                    + json.dumps(findings_data, indent=2)
                ),
            },
        ]

        response = await self.run(
            messages,
            context={"findings": findings},
        )

        if response.structured_output and isinstance(response.structured_output, list):
            return self._parse_scores(response.structured_output)

        return self._heuristic_score(findings)

    # ------------------------------------------------------------------
    # Heuristic fallback
    # ------------------------------------------------------------------

    async def _heuristic_fallback(
        self,
        messages: list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Severity-based risk scoring matrix."""
        findings = (context or {}).get("findings", [])
        scores = self._heuristic_score(findings)
        output = [s.model_dump(mode="json") for s in scores]
        return AgentResponse(
            content=json.dumps(output, indent=2),
            structured_output=output,
        )

    def _heuristic_score(
        self,
        findings: list[AuditFinding],
    ) -> list[RiskScore]:
        """Deterministic severity-based risk scoring."""
        scores: list[RiskScore] = []

        for finding in findings:
            probability = _SEVERITY_PROBABILITY.get(finding.severity, 0.5)
            impact = _SEVERITY_IMPACT.get(finding.severity, 0.5)
            reg_weight = _REGULATION_WEIGHT.get(finding.regulation_type, 0.9)

            # Composite score: weighted geometric mean capped at 1.0
            raw_score = (probability * impact) ** 0.5 * reg_weight
            score = min(round(raw_score, 3), 1.0)

            rationale = (
                f"Severity {finding.severity.value} under "
                f"{finding.regulation_type.value} "
                f"(weight={reg_weight}): probability={probability}, "
                f"impact={impact}, composite={score}"
            )

            scores.append(
                RiskScore(
                    finding_id=finding.id,
                    probability=probability,
                    impact=impact,
                    score=score,
                    rationale=rationale,
                )
            )

        logger.info("risk_scoring_complete", scores=len(scores))
        return scores

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_scores(raw: list[dict[str, Any]]) -> list[RiskScore]:
        """Parse LLM-produced score dicts into models."""
        scores: list[RiskScore] = []
        for item in raw:
            try:
                scores.append(
                    RiskScore(
                        finding_id=item.get("finding_id", str(uuid.uuid4())),
                        probability=float(item.get("probability", 0.5)),
                        impact=float(item.get("impact", 0.5)),
                        score=float(item.get("score", 0.5)),
                        rationale=item.get("rationale", ""),
                    )
                )
            except (ValueError, KeyError) as exc:
                logger.warning("risk_score_parse_error", error=str(exc))
        return scores
