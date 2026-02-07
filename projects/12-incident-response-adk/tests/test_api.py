"""API endpoint tests for Incident Response Orchestrator."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_health(client):
    """Health endpoint returns service info."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "incident-response-adk"


@pytest.mark.asyncio
async def test_create_incident(client):
    """Submit an alert for automated incident response."""
    resp = await client.post(
        "/api/v1/incidents",
        json={
            "source": "prometheus",
            "title": "High CPU on payment-service",
            "description": "CPU usage exceeded 95% for 5 minutes",
            "service": "payment-service",
            "host": "k8s-node-01",
            "raw_data": {"cpu_percent": 97.5, "duration_minutes": 5},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "stream_url" in data


@pytest.mark.asyncio
async def test_get_incident(client):
    """Get incident status after creation."""
    create_resp = await client.post(
        "/api/v1/incidents",
        json={
            "source": "datadog",
            "title": "OOM on order-processor",
            "description": "Container killed due to memory limit",
            "service": "order-processor",
            "host": "k8s-node-02",
            "raw_data": {"memory_mb": 2048, "limit_mb": 2048},
        },
    )
    session_id = create_resp.json()["session_id"]

    import asyncio
    await asyncio.sleep(0.5)

    resp = await client.get(f"/api/v1/incidents/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == session_id


@pytest.mark.asyncio
async def test_get_nonexistent_incident(client):
    """404 for unknown incident."""
    resp = await client.get("/api/v1/incidents/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_list_incidents(client):
    """List all incidents."""
    resp = await client.get("/api/v1/incidents")
    assert resp.status_code == 200
    data = resp.json()
    assert "incidents" in data


@pytest.mark.asyncio
async def test_list_runbooks(client):
    """List available remediation runbooks."""
    resp = await client.get("/api/v1/runbooks")
    assert resp.status_code == 200
    data = resp.json()
    assert "runbooks" in data
    assert len(data["runbooks"]) > 0


@pytest.mark.asyncio
async def test_resolve_incident(client):
    """Resolve an incident."""
    create_resp = await client.post(
        "/api/v1/incidents",
        json={
            "source": "test",
            "title": "Test alert",
            "description": "Test",
            "service": "test-service",
            "host": "test-host",
            "raw_data": {},
        },
    )
    session_id = create_resp.json()["session_id"]

    import asyncio
    await asyncio.sleep(1.0)

    resp = await client.post(
        f"/api/v1/incidents/{session_id}/resolve",
        json={"resolution_summary": "Resolved by automated remediation"},
    )
    assert resp.status_code in (200, 400)


@pytest.mark.asyncio
async def test_human_takeover(client):
    """Human takes over from automated response."""
    create_resp = await client.post(
        "/api/v1/incidents",
        json={
            "source": "test",
            "title": "Complex issue",
            "description": "Requires human investigation",
            "service": "auth-service",
            "host": "test-host",
            "raw_data": {},
        },
    )
    session_id = create_resp.json()["session_id"]

    import asyncio
    await asyncio.sleep(0.5)

    resp = await client.post(
        f"/api/v1/incidents/{session_id}/takeover",
        json={"operator": "test-engineer", "reason": "Manual investigation needed"},
    )
    assert resp.status_code in (200, 400)
