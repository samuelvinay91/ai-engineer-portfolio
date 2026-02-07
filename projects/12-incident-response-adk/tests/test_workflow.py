"""Workflow and agent tests for Incident Response Orchestrator."""

from __future__ import annotations

import asyncio

import pytest


@pytest.mark.asyncio
async def test_context_enricher():
    """ContextEnricherAgent enriches alert with service info."""
    from incident_response.agents.enricher import ContextEnricherAgent
    from incident_response.models import Alert

    agent = ContextEnricherAgent()
    alert = Alert(
        id="alert-test",
        source="prometheus",
        title="High CPU",
        description="CPU at 95%",
        service="payment-service",
        host="node-01",
        timestamp="2025-01-15T10:00:00Z",
        raw_data={"cpu_percent": 95},
    )

    context = await agent.run({"alert": alert})
    assert "context" in context or "incident_context" in context


@pytest.mark.asyncio
async def test_triage_agent():
    """TriageAgent classifies severity correctly."""
    from incident_response.agents.triage import TriageAgent
    from incident_response.models import Alert, IncidentContext, SeverityLevel

    agent = TriageAgent()
    # Critical alert should get P1 or P2
    alert = Alert(
        id="alert-critical",
        source="pagerduty",
        title="Service outage - payment-service DOWN",
        description="Complete service outage affecting all transactions",
        service="payment-service",
        host="node-01",
        timestamp="2025-01-15T10:00:00Z",
        raw_data={"status": "down"},
    )
    incident_context = IncidentContext(
        alert=alert,
        service_info={"tier": "critical"},
        recent_deploys=[],
        owner_team="payments",
        related_incidents=[],
    )

    result = await agent.run({"alert": alert, "incident_context": incident_context})
    assert "severity" in result
    assert result["severity"] in (SeverityLevel.P1, SeverityLevel.P2)


@pytest.mark.asyncio
async def test_log_analyzer():
    """LogAnalyzerAgent finds correlated errors."""
    from incident_response.agents.log_analyzer import LogAnalyzerAgent
    from incident_response.models import Alert

    agent = LogAnalyzerAgent()
    alert = Alert(
        id="alert-log-test",
        source="prometheus",
        title="High error rate",
        description="Error rate spike on payment-service",
        service="payment-service",
        host="node-01",
    )
    result = await agent.run({"alert": alert})
    assert "log_diagnostics" in result


@pytest.mark.asyncio
async def test_metrics_checker():
    """MetricsCheckerAgent detects anomalies."""
    from incident_response.agents.metrics_checker import MetricsCheckerAgent
    from incident_response.models import Alert

    agent = MetricsCheckerAgent()
    alert = Alert(
        id="alert-metrics-test",
        source="prometheus",
        title="CPU spike on payment-service",
        description="CPU at 95%",
        service="payment-service",
        host="node-01",
    )
    result = await agent.run({"alert": alert})
    assert "metrics_diagnostics" in result


@pytest.mark.asyncio
async def test_config_auditor():
    """ConfigAuditorAgent detects config drift."""
    from incident_response.agents.config_auditor import ConfigAuditorAgent
    from incident_response.models import Alert

    agent = ConfigAuditorAgent()
    alert = Alert(
        id="alert-config-test",
        source="prometheus",
        title="Config drift detected",
        description="Configuration mismatch on payment-service",
        service="payment-service",
        host="node-01",
    )
    result = await agent.run({"alert": alert})
    assert "config_diagnostics" in result


@pytest.mark.asyncio
async def test_mock_alerts():
    """Mock data provides realistic alerts."""
    from incident_response.mock_data.alerts import MOCK_ALERTS

    assert len(MOCK_ALERTS) >= 5
    services = {a.service for a in MOCK_ALERTS}
    assert len(services) >= 3


@pytest.mark.asyncio
async def test_mock_infrastructure():
    """Mock infrastructure data is available."""
    from incident_response.mock_data.infrastructure import SERVICE_REGISTRY

    assert len(SERVICE_REGISTRY) >= 5
