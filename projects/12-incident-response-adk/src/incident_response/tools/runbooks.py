"""Predefined runbook definitions for automated remediation.

Each runbook defines a sequence of steps, required parameters, expected
duration, and success criteria for common incident remediation actions.
"""

from __future__ import annotations

from typing import Any


# ---------------------------------------------------------------------------
# Runbook Registry
# ---------------------------------------------------------------------------

RUNBOOKS: dict[str, dict[str, Any]] = {
    "restart_service": {
        "id": "RB-001",
        "name": "Service Rolling Restart",
        "description": (
            "Perform a rolling restart of the service deployment to clear "
            "transient state, reset connections, and reload configuration."
        ),
        "steps": [
            "Verify current pod status and replica count",
            "Initiate rolling restart via kubectl rollout restart",
            "Wait for all pods to reach Ready state (timeout: 120s)",
            "Verify health endpoint returns 200 OK",
            "Check error rate returns to baseline within 60s",
        ],
        "parameters": {
            "service": {"type": "string", "required": True},
            "namespace": {"type": "string", "required": True},
            "timeout_seconds": {"type": "int", "default": 120},
        },
        "expected_duration_seconds": 180,
        "risk_level": "low",
        "auto_approve": True,
        "applicable_symptoms": [
            "memory_leak",
            "connection_pool_exhaustion",
            "stale_cache",
            "thread_pool_exhaustion",
        ],
    },
    "scale_up": {
        "id": "RB-002",
        "name": "Horizontal Scale-Up",
        "description": (
            "Increase the replica count of a deployment to handle elevated "
            "traffic or reduce per-pod resource pressure."
        ),
        "steps": [
            "Check current replica count and HPA configuration",
            "Calculate target replicas based on current load",
            "Scale deployment to target replica count",
            "Wait for new pods to be scheduled and reach Ready state",
            "Verify load is distributed across new replicas",
            "Monitor error rate and latency for 60s after scaling",
        ],
        "parameters": {
            "service": {"type": "string", "required": True},
            "namespace": {"type": "string", "required": True},
            "target_replicas": {"type": "int", "default": 0},
            "scale_factor": {"type": "float", "default": 1.5},
        },
        "expected_duration_seconds": 300,
        "risk_level": "low",
        "auto_approve": True,
        "applicable_symptoms": [
            "high_cpu",
            "high_latency",
            "connection_pool_exhaustion",
            "traffic_spike",
        ],
    },
    "rollback_deploy": {
        "id": "RB-003",
        "name": "Deployment Rollback",
        "description": (
            "Roll back to the previous known-good deployment version. "
            "Triggered when a recent deploy correlates with the incident."
        ),
        "steps": [
            "Identify current and previous deployment versions",
            "Verify previous version is available in registry",
            "Initiate rollback via kubectl rollout undo",
            "Wait for rollback to complete (all pods running previous version)",
            "Verify health endpoint returns 200 OK",
            "Monitor error rate for 120s to confirm improvement",
            "Tag current version in deploy tracker as 'rolled-back'",
        ],
        "parameters": {
            "service": {"type": "string", "required": True},
            "namespace": {"type": "string", "required": True},
            "target_version": {"type": "string", "default": ""},
        },
        "expected_duration_seconds": 240,
        "risk_level": "medium",
        "auto_approve": False,
        "applicable_symptoms": [
            "regression_after_deploy",
            "new_error_patterns",
            "config_drift",
            "performance_degradation",
        ],
    },
    "clear_cache": {
        "id": "RB-004",
        "name": "Cache Invalidation",
        "description": (
            "Clear application caches (Redis, in-memory, CDN) to resolve "
            "issues caused by stale or corrupted cached data."
        ),
        "steps": [
            "Identify cache backends used by the service",
            "Flush Redis cache keys matching service prefix",
            "Trigger in-memory cache clear via admin endpoint",
            "Invalidate CDN cache for affected paths (if applicable)",
            "Monitor cache hit rate recovery over 60s",
            "Verify error rate decrease after cache rebuild",
        ],
        "parameters": {
            "service": {"type": "string", "required": True},
            "cache_type": {"type": "string", "default": "all"},
            "key_pattern": {"type": "string", "default": "*"},
        },
        "expected_duration_seconds": 60,
        "risk_level": "low",
        "auto_approve": True,
        "applicable_symptoms": [
            "stale_data",
            "inconsistent_responses",
            "serialization_errors",
        ],
    },
    "rotate_certs": {
        "id": "RB-005",
        "name": "Certificate Rotation",
        "description": (
            "Rotate TLS certificates for a service. Triggers cert-manager "
            "renewal or manual certificate replacement."
        ),
        "steps": [
            "Check current certificate expiry and issuer",
            "Trigger cert-manager certificate renewal",
            "Wait for new certificate to be issued (timeout: 300s)",
            "Verify new certificate is valid and not expired",
            "Restart pods to pick up new certificate",
            "Verify TLS handshake succeeds with new certificate",
            "Update certificate monitoring to track new expiry",
        ],
        "parameters": {
            "service": {"type": "string", "required": True},
            "namespace": {"type": "string", "required": True},
            "issuer": {"type": "string", "default": "letsencrypt-prod"},
            "force_renewal": {"type": "bool", "default": False},
        },
        "expected_duration_seconds": 420,
        "risk_level": "medium",
        "auto_approve": False,
        "applicable_symptoms": [
            "certificate_expiry",
            "tls_handshake_failure",
            "ssl_error",
        ],
    },
    "drain_connections": {
        "id": "RB-006",
        "name": "Connection Pool Drain",
        "description": (
            "Gracefully drain and reset connection pools to resolve "
            "connection exhaustion or stale connection issues."
        ),
        "steps": [
            "Reduce incoming traffic via load balancer weight adjustment",
            "Wait for in-flight requests to complete (timeout: 30s)",
            "Reset connection pool configuration",
            "Gradually restore traffic weight",
            "Monitor connection pool utilization for 60s",
        ],
        "parameters": {
            "service": {"type": "string", "required": True},
            "pool_type": {"type": "string", "default": "database"},
            "max_connections": {"type": "int", "default": 200},
        },
        "expected_duration_seconds": 120,
        "risk_level": "medium",
        "auto_approve": True,
        "applicable_symptoms": [
            "connection_pool_exhaustion",
            "connection_timeout",
            "stale_connections",
        ],
    },
}


def get_runbook(runbook_id: str) -> dict[str, Any] | None:
    """Look up a runbook by its action type key."""
    return RUNBOOKS.get(runbook_id)


def list_runbooks() -> list[dict[str, Any]]:
    """Return all available runbooks with summary info."""
    return [
        {
            "action_type": key,
            "id": rb["id"],
            "name": rb["name"],
            "description": rb["description"],
            "risk_level": rb["risk_level"],
            "auto_approve": rb["auto_approve"],
            "expected_duration_seconds": rb["expected_duration_seconds"],
        }
        for key, rb in RUNBOOKS.items()
    ]


def select_runbook_for_symptom(symptom: str) -> str | None:
    """Find the best runbook for a given symptom.

    Returns the action_type key of the matching runbook, or None.
    """
    for action_type, rb in RUNBOOKS.items():
        if symptom in rb.get("applicable_symptoms", []):
            return action_type
    return None
