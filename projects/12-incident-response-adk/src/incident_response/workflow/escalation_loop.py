"""Escalation Loop Pipeline.

Demonstrates ADK's LoopAgent pattern by iterating through:
1. RemediationAgent - execute runbook-based fix
2. VerificationAgent - check if service recovered

The loop continues until verification passes (loop_complete=True)
or max_iterations is reached, at which point the incident escalates
to a human operator.
"""

from __future__ import annotations

from incident_response.agents.base import LoopAgent
from incident_response.agents.remediator import RemediationAgent
from incident_response.agents.verifier import VerificationAgent


def build_escalation_loop(max_iterations: int = 3) -> LoopAgent:
    """Construct the remediation/verification escalation loop.

    Args:
        max_iterations: Maximum remediation attempts before escalating
                       to human takeover.

    Returns:
        A LoopAgent that alternates between remediation and verification,
        escalating on each failed iteration.
    """
    remediator = RemediationAgent()
    verifier = VerificationAgent()

    return LoopAgent(
        name="escalation_loop",
        description=(
            "Remediation escalation loop that attempts automated fixes "
            "and verifies service recovery. Escalates to higher-tier "
            "responders on each failed iteration. Mirrors Google ADK "
            "LoopAgent pattern."
        ),
        sub_agents=[remediator, verifier],
        max_iterations=max_iterations,
    )
