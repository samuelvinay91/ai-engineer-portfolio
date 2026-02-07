"""Verification Agent.

Performs post-remediation health checks to determine whether the service
has recovered. Checks HTTP health endpoints, error rates, and latency
against SLA thresholds.
"""

from __future__ import annotations

from typing import Any

import structlog

from incident_response.agents.base import LlmAgent
from incident_response.models import EscalationLevel, RemediationAction
from incident_response.tools.health_checks import check_service_health

logger = structlog.get_logger(__name__)

# Escalation level progression
_ESCALATION_ORDER = [
    EscalationLevel.L1_AUTO,
    EscalationLevel.L2_ONCALL,
    EscalationLevel.L3_SENIOR,
    EscalationLevel.L4_MANAGEMENT,
]


class VerificationAgent(LlmAgent):
    """Verifies service health after remediation and determines next steps.

    Runs health checks against the service and decides whether the
    incident is resolved, needs another remediation attempt, or should
    be escalated to a human operator.
    """

    def __init__(self) -> None:
        super().__init__(
            name="verifier",
            instruction=(
                "You are a verification specialist. After a remediation action, "
                "check the service health: HTTP endpoint status, error rate, "
                "and latency against SLA. Determine if the incident is resolved, "
                "needs further remediation, or should be escalated."
            ),
            description="Verifies post-remediation health and decides next steps",
            tools=[check_service_health],
        )

    async def _heuristic_run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Verify service health after remediation.

        Expects:
            context["alert"]: Alert
            context["last_remediation"]: RemediationAction
            context["loop_iteration"]: int
            context["escalation_level"]: EscalationLevel

        Sets:
            context["verification_result"]: dict with health check results
            context["loop_complete"]: bool (True if resolved)
            context["escalation_level"]: EscalationLevel (upgraded if needed)
            context["needs_escalation"]: bool
        """
        alert = context["alert"]
        service = alert.service
        iteration = context.get("loop_iteration", 1)
        last_remediation: RemediationAction | None = context.get("last_remediation")

        logger.info(
            "verifying_health",
            service=service,
            iteration=iteration,
            last_action=last_remediation.action_type if last_remediation else "none",
        )

        # Run health check
        health_result = await check_service_health(service, iteration=iteration)

        is_healthy = health_result.get("healthy", False)
        verdict = health_result.get("verdict", "FAIL")

        # Store verification result
        context["verification_result"] = health_result

        if is_healthy:
            # Service recovered
            context["loop_complete"] = True
            context["needs_escalation"] = False

            logger.info(
                "verification_passed",
                service=service,
                iteration=iteration,
                verdict=verdict,
            )
        else:
            # Service still unhealthy - escalate
            current_level: EscalationLevel = context.get(
                "escalation_level", EscalationLevel.L1_AUTO
            )
            next_level = _next_escalation_level(current_level)
            context["escalation_level"] = next_level
            context["needs_escalation"] = True
            context["loop_complete"] = False

            logger.warning(
                "verification_failed",
                service=service,
                iteration=iteration,
                verdict=verdict,
                current_level=current_level.value,
                next_level=next_level.value,
                evidence=health_result.get("evidence", []),
            )

        return context


def _next_escalation_level(current: EscalationLevel) -> EscalationLevel:
    """Advance to the next escalation level.

    Returns the next level in the chain, or L4_MANAGEMENT if already
    at the highest level.
    """
    try:
        current_idx = _ESCALATION_ORDER.index(current)
        next_idx = min(current_idx + 1, len(_ESCALATION_ORDER) - 1)
        return _ESCALATION_ORDER[next_idx]
    except ValueError:
        return EscalationLevel.L2_ONCALL
