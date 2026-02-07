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
    from incident_response.models import Alert

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

    result = await agent.run({"alert": alert})
    assert "severity" in result
    assert result["severity"] in ("P1", "P2")


@pytest.mark.asyncio
async def test_log_analyzer():
    """LogAnalyzerAgent finds correlated errors."""
    from incident_response.agents.log_analyzer import LogAnalyzerAgent

    agent = LogAnalyzerAgent()
    result = await agent.run({
        "service": "payment-service",
        "timerange": "1h",
    })
    assert "diagnostic" in result or "findings" in result


@pytest.mark.asyncio
async def test_metrics_checker():
    """MetricsCheckerAgent detects anomalies."""
    from incident_response.agents.metrics_checker import MetricsCheckerAgent

    agent = MetricsCheckerAgent()
    result = await agent.run({
        "service": "payment-service",
        "timerange": "1h",
    })
    assert "diagnostic" in result or "findings" in result


@pytest.mark.asyncio
async def test_config_auditor():
    """ConfigAuditorAgent detects config drift."""
    from incident_response.agents.config_auditor import ConfigAuditorAgent

    agent = ConfigAuditorAgent()
    result = await agent.run({
        "service": "payment-service",
    })
    assert "diagnostic" in result or "findings" in result


@pytest.mark.asyncio
async def test_mock_alerts():
    """Mock data provides realistic alerts."""
    from incident_response.mock_data.alerts import get_mock_alerts

    alerts = get_mock_alerts()
    assert len(alerts) >= 5
    services = {a.service for a in alerts}
    assert len(services) >= 3


@pytest.mark.asyncio
async def test_mock_infrastructure():
    """Mock infrastructure data is available."""
    from incident_response.mock_data.infrastructure import get_service_registry

    registry = get_service_registry()
    assert len(registry) >= 5
