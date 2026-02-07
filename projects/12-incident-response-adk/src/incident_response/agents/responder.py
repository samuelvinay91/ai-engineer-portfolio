"""Responder Assigner Agent.

Maps incident severity and service to the appropriate on-call team and
builds the escalation path for the incident response lifecycle.
"""

from __future__ import annotations

from typing import Any

import structlog

from incident_response.agents.base import LlmAgent
from incident_response.mock_data.infrastructure import ONCALL_ROTATION
from incident_response.models import EscalationLevel, IncidentContext, SeverityLevel

logger = structlog.get_logger(__name__)


class ResponderAssignerAgent(LlmAgent):
    """Assigns the initial responder and escalation path.

    Maps the incident severity and affected service to the on-call rotation,
    determining the first responder and building an escalation chain.
    """

    def __init__(self) -> None:
        super().__init__(
            name="responder_assigner",
            instruction=(
                "You are an incident responder assigner. Based on the severity "
                "and affected service, identify the on-call team, assign the "
                "initial responder, and build the escalation path."
            ),
            description="Assigns on-call responders and builds escalation paths",
        )

    async def _heuristic_run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Assign responder based on severity and service ownership.

        Expects:
            context["incident_context"]: IncidentContext
            context["severity"]: SeverityLevel
            context["owner_team"]: str

        Sets:
            context["assigned_responder"]: dict with responder info
            context["escalation_path"]: list of escalation steps
            context["escalation_level"]: EscalationLevel
            context["notification_channels"]: list of notification targets
        """
        incident_ctx: IncidentContext = context["incident_context"]
        severity: SeverityLevel = context["severity"]
        owner_team: str = context.get("owner_team", "platform")

        logger.info(
            "assigning_responder",
            alert_id=incident_ctx.alert.id,
            severity=severity.value,
            owner_team=owner_team,
        )

        # Look up on-call rotation
        oncall = ONCALL_ROTATION.get(owner_team, ONCALL_ROTATION.get("platform", {}))

        # Determine initial escalation level based on severity
        if severity == SeverityLevel.P1:
            # P1: Start at L2 (skip automation for critical)
            initial_level = EscalationLevel.L2_ONCALL
            assigned_to = oncall.get("l2_oncall", "unknown")
        elif severity == SeverityLevel.P2:
            # P2: Start with automation, but alert on-call
            initial_level = EscalationLevel.L1_AUTO
            assigned_to = oncall.get("l1_oncall", "unknown")
        else:
            # P3/P4: Full automation
            initial_level = EscalationLevel.L1_AUTO
            assigned_to = "automation"

        # Build escalation path
        escalation_path = _build_escalation_path(severity, oncall)

        # Determine notification channels
        notification_channels = [oncall.get("slack_channel", "#incidents")]
        if severity in (SeverityLevel.P1, SeverityLevel.P2):
            notification_channels.append(f"pagerduty:{oncall.get('pagerduty_service', 'default')}")
        if severity == SeverityLevel.P1:
            notification_channels.append("#incident-war-room")

        assigned_responder = {
            "name": assigned_to,
            "team": owner_team,
            "level": initial_level.value,
            "pagerduty_service": oncall.get("pagerduty_service", ""),
            "slack_channel": oncall.get("slack_channel", ""),
        }

        context["assigned_responder"] = assigned_responder
        context["escalation_path"] = escalation_path
        context["escalation_level"] = initial_level
        context["notification_channels"] = notification_channels

        logger.info(
            "responder_assigned",
            alert_id=incident_ctx.alert.id,
            assigned_to=assigned_to,
            escalation_level=initial_level.value,
            notifications=len(notification_channels),
        )

        return context


def _build_escalation_path(
    severity: SeverityLevel,
    oncall: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the escalation chain based on severity.

    Returns a list of escalation steps in order. P1 incidents start
    higher in the chain and have shorter timeouts.
    """
    base_timeout = {
        SeverityLevel.P1: 5,    # 5 min before escalation
        SeverityLevel.P2: 15,   # 15 min
        SeverityLevel.P3: 30,   # 30 min
        SeverityLevel.P4: 60,   # 60 min
    }
    timeout_minutes = base_timeout.get(severity, 30)

    path: list[dict[str, Any]] = [
        {
            "level": EscalationLevel.L1_AUTO.value,
            "action": "Automated remediation attempt",
            "timeout_minutes": timeout_minutes,
            "assignee": "automation",
        },
        {
            "level": EscalationLevel.L2_ONCALL.value,
            "action": "Page primary on-call engineer",
            "timeout_minutes": timeout_minutes * 2,
            "assignee": oncall.get("l2_oncall", "unknown"),
        },
        {
            "level": EscalationLevel.L3_SENIOR.value,
            "action": "Escalate to senior/staff engineer",
            "timeout_minutes": timeout_minutes * 3,
            "assignee": oncall.get("l3_senior", "unknown"),
        },
        {
            "level": EscalationLevel.L4_MANAGEMENT.value,
            "action": "Escalate to engineering management",
            "timeout_minutes": timeout_minutes * 4,
            "assignee": oncall.get("l4_manager", "unknown"),
        },
    ]

    return path
