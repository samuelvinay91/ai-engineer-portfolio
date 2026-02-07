"""Context Enricher Agent.

Enriches a raw alert with service metadata, recent deployments, owner team
information, and related incident history. This is the first step in the
sequential triage pipeline.
"""

from __future__ import annotations

from typing import Any

import structlog

from incident_response.agents.base import LlmAgent
from incident_response.mock_data.infrastructure import (
    DEPLOY_HISTORY,
    SERVICE_REGISTRY,
    get_oncall,
)
from incident_response.models import Alert, IncidentContext

logger = structlog.get_logger(__name__)


class ContextEnricherAgent(LlmAgent):
    """Enriches an alert with operational context from infrastructure sources.

    In a production ADK deployment this would use an LLM to decide which
    context sources to query. Here we use deterministic lookups against
    mock infrastructure data.
    """

    def __init__(self) -> None:
        super().__init__(
            name="context_enricher",
            instruction=(
                "You are an incident context enricher. Given an alert, gather "
                "all relevant context: service info, recent deploys, owner team, "
                "and related incidents. Produce a structured IncidentContext."
            ),
            description="Enriches alerts with service metadata and operational context",
        )

    async def _heuristic_run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Enrich the alert with operational context using mock data lookups.

        Expects:
            context["alert"]: Alert model instance

        Sets:
            context["incident_context"]: IncidentContext model instance
        """
        alert: Alert = context["alert"]
        service_name = alert.service

        logger.info(
            "enriching_context",
            alert_id=alert.id,
            service=service_name,
        )

        # 1. Service registry lookup
        service_info = SERVICE_REGISTRY.get(service_name, {})
        owner_team = service_info.get("owner_team", "unknown")

        # 2. Recent deployments
        recent_deploys = DEPLOY_HISTORY.get(service_name, [])[:5]

        # 3. On-call rotation
        oncall_info = get_oncall(owner_team)

        # 4. Related incidents (simulated - check for same service alerts)
        related_incidents = _find_related_incidents(service_name, alert.id)

        # Build enriched context
        enriched_service_info = {
            **service_info,
            "oncall": oncall_info,
        }

        incident_context = IncidentContext(
            alert=alert,
            service_info=enriched_service_info,
            recent_deploys=recent_deploys,
            owner_team=owner_team,
            related_incidents=related_incidents,
        )

        context["incident_context"] = incident_context
        context["owner_team"] = owner_team
        context["service_info"] = enriched_service_info

        logger.info(
            "context_enriched",
            alert_id=alert.id,
            service=service_name,
            owner_team=owner_team,
            recent_deploys=len(recent_deploys),
            related_incidents=len(related_incidents),
        )

        return context


def _find_related_incidents(service: str, exclude_alert_id: str) -> list[dict[str, Any]]:
    """Find related historical incidents for the same service.

    In production this would query an incident management system like
    PagerDuty or ServiceNow. Here we return mock related incidents.
    """
    # Service dependency mapping for blast radius analysis
    service_deps = SERVICE_REGISTRY.get(service, {}).get("dependencies", [])

    related: list[dict[str, Any]] = []

    # Add a mock related incident if the service has known issues
    known_recurring: dict[str, list[dict[str, Any]]] = {
        "payment-service": [
            {
                "incident_id": "INC-2024-0847",
                "title": "Payment service CPU spike during flash sale",
                "severity": "P2",
                "resolved_at": "2024-11-15T14:30:00Z",
                "root_cause": "Thread pool exhaustion under high concurrency",
                "resolution": "Increased thread pool size and added circuit breaker",
            },
        ],
        "order-processor": [
            {
                "incident_id": "INC-2024-0923",
                "title": "OOM kills after batch size increase",
                "severity": "P1",
                "resolved_at": "2024-12-01T08:15:00Z",
                "root_cause": "Memory limit too low for new batch processing config",
                "resolution": "Increased memory limit to 4Gi and reduced batch size",
            },
        ],
        "api-gateway": [
            {
                "incident_id": "INC-2024-0956",
                "title": "5xx cascade from payment-service outage",
                "severity": "P1",
                "resolved_at": "2024-12-10T16:45:00Z",
                "root_cause": "Missing circuit breaker for payment-service upstream",
                "resolution": "Added circuit breaker with 10-failure threshold",
            },
        ],
        "db-proxy": [
            {
                "incident_id": "INC-2024-0891",
                "title": "Connection pool exhaustion during peak",
                "severity": "P2",
                "resolved_at": "2024-11-28T11:20:00Z",
                "root_cause": "Leaked connections from long-running queries",
                "resolution": "Added connection timeout and query kill after 30s",
            },
        ],
    }

    related.extend(known_recurring.get(service, []))

    # Note any affected dependencies
    if service_deps:
        related.append({
            "type": "dependency_note",
            "message": f"Service depends on: {', '.join(service_deps)}",
            "impact": "Failure in dependencies may cascade to this service",
        })

    return related
