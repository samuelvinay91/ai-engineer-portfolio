"""API endpoint tests for Compliance Audit Agents."""

from __future__ import annotations

import pytest


@pytest.mark.asyncio
async def test_health(client):
    """Health endpoint returns service info."""
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert data["service"] == "compliance-audit-agents"


@pytest.mark.asyncio
async def test_list_policies(client):
    """Policies endpoint returns loaded compliance policies."""
    resp = await client.get("/api/v1/policies")
    assert resp.status_code == 200
    data = resp.json()
    assert "policies" in data
    assert len(data["policies"]) > 0
    # Should have SOX, GDPR, SOC2
    types = {p["regulation_type"] for p in data["policies"]}
    assert "SOX" in types or "sox" in types.union({t.lower() for t in types})


@pytest.mark.asyncio
async def test_create_audit_session(client):
    """Submit transactions for compliance audit."""
    resp = await client.post(
        "/api/v1/audits",
        json={
            "transactions": [
                {
                    "id": "txn-001",
                    "timestamp": "2025-01-15T10:30:00Z",
                    "actor": "john.smith",
                    "action": "approve_payment",
                    "resource": "invoice-5001",
                    "metadata": {"amount": 50000, "department": "finance"},
                    "department": "finance",
                }
            ]
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "stream_url" in data


@pytest.mark.asyncio
async def test_get_audit_session(client):
    """Get audit session status."""
    # Create session first
    create_resp = await client.post(
        "/api/v1/audits",
        json={
            "transactions": [
                {
                    "id": "txn-002",
                    "timestamp": "2025-01-15T10:30:00Z",
                    "actor": "jane.doe",
                    "action": "access_pii",
                    "resource": "customer-database",
                    "metadata": {},
                    "department": "marketing",
                }
            ]
        },
    )
    session_id = create_resp.json()["session_id"]

    # Give the async workflow a moment
    import asyncio
    await asyncio.sleep(0.5)

    resp = await client.get(f"/api/v1/audits/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["id"] == session_id


@pytest.mark.asyncio
async def test_get_nonexistent_session(client):
    """404 for unknown session."""
    resp = await client.get("/api/v1/audits/nonexistent-id")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_quick_check(client):
    """Quick single-transaction compliance check."""
    resp = await client.post(
        "/api/v1/policies/check",
        json={
            "actor": "admin",
            "action": "delete_personal_data",
            "resource": "user-records",
            "department": "IT",
            "metadata": {"reason": "gdpr_request"},
        },
    )
    assert resp.status_code == 200
    data = resp.json()
    assert "findings" in data or "result" in data


@pytest.mark.asyncio
async def test_approve_audit(client):
    """Approve audit findings."""
    # Create a session
    create_resp = await client.post(
        "/api/v1/audits",
        json={
            "transactions": [
                {
                    "id": "txn-approve",
                    "timestamp": "2025-01-15T10:30:00Z",
                    "actor": "test",
                    "action": "test_action",
                    "resource": "test-resource",
                    "metadata": {},
                    "department": "test",
                }
            ]
        },
    )
    session_id = create_resp.json()["session_id"]

    # Wait for workflow to complete (uses heuristic, should be fast)
    import asyncio
    await asyncio.sleep(1.0)

    resp = await client.post(f"/api/v1/audits/{session_id}/approve")
    # Either 200 (if in awaiting_approval state) or 400 (if already completed)
    assert resp.status_code in (200, 400)
