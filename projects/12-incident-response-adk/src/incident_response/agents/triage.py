"""Triage Agent.

Classifies incident severity (P1-P4) based on enriched context using
keyword-based heuristics as a fallback for the LLM-powered ADK triage.
"""

from __future__ import annotations

from typing import Any

import structlog

from incident_response.agents.base import LlmAgent
from incident_response.models import IncidentContext, SeverityLevel

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Keyword-based severity classification rules
# ---------------------------------------------------------------------------

P1_KEYWORDS = {
    "outage", "down", "critical", "crash", "crashloopbackoff",
    "oom", "out of memory", "data loss", "security breach",
    "complete failure", "service unavailable", "503", "502",
    "connection refused", "fatal", "catastrophic",
}

P2_KEYWORDS = {
    "degraded", "high error", "high-error", "spike", "surge",
    "elevated", "exhaustion", "exhausted", "timeout", "leak",
    "circuit breaker", "5xx", "connection pool", "expir",
}

P3_KEYWORDS = {
    "slow", "latency", "delayed", "warning", "warn",
    "increased", "above threshold", "drift", "unassigned",
    "yellow", "lag", "backlog",
}


class TriageAgent(LlmAgent):
    """Classifies incident severity and provides reasoning.

    In a production ADK deployment this would use Gemini to reason about
    the alert context. Here we use keyword-based heuristics as a fallback.
    """

    def __init__(self) -> None:
        super().__init__(
            name="triage_agent",
            instruction=(
                "You are an incident triage specialist. Analyze the enriched "
                "incident context and classify severity as P1 (critical), "
                "P2 (high), P3 (medium), or P4 (low). Consider: service tier, "
                "customer impact, blast radius, and recent changes. Provide "
                "clear reasoning for your classification."
            ),
            description="Classifies incident severity using context analysis",
        )

    async def _heuristic_run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Classify severity using keyword-based heuristics.

        Expects:
            context["incident_context"]: IncidentContext model instance

        Sets:
            context["severity"]: SeverityLevel enum value
            context["triage_reasoning"]: Explanation string
        """
        incident_ctx: IncidentContext = context["incident_context"]
        alert = incident_ctx.alert
        service_info = incident_ctx.service_info

        logger.info("triaging_incident", alert_id=alert.id, service=alert.service)

        # Build searchable text from alert content
        search_text = (
            f"{alert.title} {alert.description} "
            f"{' '.join(str(v) for v in alert.raw_data.values())}"
        ).lower()

        # Determine base severity from keywords
        severity, keyword_match = _classify_by_keywords(search_text)

        # Adjust severity based on service tier
        service_tier = service_info.get("tier", "medium")
        tier_adjustment = ""
        if service_tier == "critical" and severity in (SeverityLevel.P3, SeverityLevel.P4):
            severity = SeverityLevel(f"P{max(1, int(severity.value[1]) - 1)}")
            tier_adjustment = (
                f" Upgraded from {severity.value} due to critical service tier."
            )

        # Check for recent deployments correlation
        deploy_correlation = ""
        recent_deploys = incident_ctx.recent_deploys
        if recent_deploys:
            latest = recent_deploys[0]
            deploy_correlation = (
                f" Recent deploy detected: {latest.get('version', 'unknown')} "
                f"deployed {latest.get('deployed_at', 'recently')} by "
                f"{latest.get('deployed_by', 'unknown')}. "
                f"Changelog: {latest.get('changelog', 'N/A')}."
            )

        # Build reasoning
        reasoning_parts = [
            f"Alert '{alert.title}' on service '{alert.service}' "
            f"classified as {severity.value}.",
            f"Keyword match: '{keyword_match}'." if keyword_match else "",
            f"Service tier: {service_tier}.",
            tier_adjustment,
            deploy_correlation,
            f"Related incidents found: {len(incident_ctx.related_incidents)}.",
        ]
        reasoning = " ".join(part for part in reasoning_parts if part)

        context["severity"] = severity
        context["triage_reasoning"] = reasoning

        logger.info(
            "triage_complete",
            alert_id=alert.id,
            severity=severity.value,
            keyword_match=keyword_match,
            service_tier=service_tier,
        )

        return context


def _classify_by_keywords(text: str) -> tuple[SeverityLevel, str]:
    """Match text against severity keyword sets.

    Returns:
        Tuple of (SeverityLevel, matched_keyword). Falls back to P4
        if no keywords match.
    """
    for keyword in P1_KEYWORDS:
        if keyword in text:
            return SeverityLevel.P1, keyword

    for keyword in P2_KEYWORDS:
        if keyword in text:
            return SeverityLevel.P2, keyword

    for keyword in P3_KEYWORDS:
        if keyword in text:
            return SeverityLevel.P3, keyword

    return SeverityLevel.P4, ""
