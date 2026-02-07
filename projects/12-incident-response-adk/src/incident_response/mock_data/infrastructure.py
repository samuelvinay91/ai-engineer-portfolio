"""Mock infrastructure state for the Incident Response Orchestrator.

Provides realistic service registry, deployment history, on-call rotation,
and baseline metric data for a Kubernetes-based microservices platform.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(tz=timezone.utc)


# ---------------------------------------------------------------------------
# Service Registry
# ---------------------------------------------------------------------------

SERVICE_REGISTRY: dict[str, dict[str, Any]] = {
    "payment-service": {
        "name": "payment-service",
        "namespace": "payments",
        "owner_team": "payments",
        "tier": "critical",
        "replicas": 4,
        "cpu_limit": "2000m",
        "memory_limit": "4Gi",
        "sla_latency_ms": 200,
        "sla_error_rate": 0.001,
        "sla_availability": 99.99,
        "language": "java",
        "framework": "spring-boot",
        "dependencies": ["db-proxy", "auth-service", "notification-service"],
        "health_endpoint": "/actuator/health",
        "image": "registry.example.com/payment-service:v2.14.3",
    },
    "order-processor": {
        "name": "order-processor",
        "namespace": "fulfillment",
        "owner_team": "fulfillment",
        "tier": "critical",
        "replicas": 3,
        "cpu_limit": "1500m",
        "memory_limit": "2Gi",
        "sla_latency_ms": 500,
        "sla_error_rate": 0.005,
        "sla_availability": 99.95,
        "language": "python",
        "framework": "fastapi",
        "dependencies": ["db-proxy", "payment-service", "notification-service"],
        "health_endpoint": "/health",
        "image": "registry.example.com/order-processor:v1.8.2",
    },
    "api-gateway": {
        "name": "api-gateway",
        "namespace": "platform",
        "owner_team": "platform",
        "tier": "critical",
        "replicas": 6,
        "cpu_limit": "1000m",
        "memory_limit": "2Gi",
        "sla_latency_ms": 50,
        "sla_error_rate": 0.001,
        "sla_availability": 99.99,
        "language": "go",
        "framework": "gin",
        "dependencies": ["auth-service"],
        "health_endpoint": "/healthz",
        "image": "registry.example.com/api-gateway:v3.2.1",
    },
    "auth-service": {
        "name": "auth-service",
        "namespace": "identity",
        "owner_team": "identity",
        "tier": "critical",
        "replicas": 4,
        "cpu_limit": "1000m",
        "memory_limit": "2Gi",
        "sla_latency_ms": 100,
        "sla_error_rate": 0.001,
        "sla_availability": 99.99,
        "language": "go",
        "framework": "gin",
        "dependencies": ["db-proxy"],
        "health_endpoint": "/healthz",
        "image": "registry.example.com/auth-service:v2.6.0",
    },
    "log-aggregator": {
        "name": "log-aggregator",
        "namespace": "observability",
        "owner_team": "observability",
        "tier": "high",
        "replicas": 3,
        "cpu_limit": "2000m",
        "memory_limit": "8Gi",
        "sla_latency_ms": 1000,
        "sla_error_rate": 0.01,
        "sla_availability": 99.9,
        "language": "java",
        "framework": "custom",
        "dependencies": [],
        "health_endpoint": "/health",
        "image": "registry.example.com/log-aggregator:v1.4.0",
    },
    "recommendation-engine": {
        "name": "recommendation-engine",
        "namespace": "ml-platform",
        "owner_team": "ml-platform",
        "tier": "high",
        "replicas": 3,
        "cpu_limit": "4000m",
        "memory_limit": "4Gi",
        "sla_latency_ms": 300,
        "sla_error_rate": 0.01,
        "sla_availability": 99.9,
        "language": "python",
        "framework": "fastapi",
        "dependencies": ["db-proxy", "search-service"],
        "health_endpoint": "/health",
        "image": "registry.example.com/recommendation-engine:v2.1.0",
    },
    "db-proxy": {
        "name": "db-proxy",
        "namespace": "data",
        "owner_team": "data",
        "tier": "critical",
        "replicas": 2,
        "cpu_limit": "500m",
        "memory_limit": "1Gi",
        "sla_latency_ms": 10,
        "sla_error_rate": 0.0001,
        "sla_availability": 99.999,
        "language": "go",
        "framework": "pgbouncer",
        "dependencies": [],
        "health_endpoint": "/healthz",
        "image": "registry.example.com/db-proxy:v1.2.0",
    },
    "search-service": {
        "name": "search-service",
        "namespace": "search",
        "owner_team": "search",
        "tier": "high",
        "replicas": 4,
        "cpu_limit": "2000m",
        "memory_limit": "4Gi",
        "sla_latency_ms": 200,
        "sla_error_rate": 0.005,
        "sla_availability": 99.95,
        "language": "python",
        "framework": "fastapi",
        "dependencies": ["db-proxy"],
        "health_endpoint": "/health",
        "image": "registry.example.com/search-service:v3.0.1",
    },
    "notification-service": {
        "name": "notification-service",
        "namespace": "communications",
        "owner_team": "communications",
        "tier": "medium",
        "replicas": 3,
        "cpu_limit": "1000m",
        "memory_limit": "2Gi",
        "sla_latency_ms": 500,
        "sla_error_rate": 0.01,
        "sla_availability": 99.9,
        "language": "typescript",
        "framework": "nestjs",
        "dependencies": ["db-proxy"],
        "health_endpoint": "/health",
        "image": "registry.example.com/notification-service:v1.9.0",
    },
    "cdn-edge": {
        "name": "cdn-edge",
        "namespace": "edge",
        "owner_team": "platform",
        "tier": "critical",
        "replicas": 8,
        "cpu_limit": "500m",
        "memory_limit": "1Gi",
        "sla_latency_ms": 20,
        "sla_error_rate": 0.0001,
        "sla_availability": 99.999,
        "language": "rust",
        "framework": "actix-web",
        "dependencies": ["auth-service"],
        "health_endpoint": "/healthz",
        "image": "registry.example.com/cdn-edge:v4.1.0",
    },
}


# ---------------------------------------------------------------------------
# Deployment History (most recent 5 per service)
# ---------------------------------------------------------------------------

DEPLOY_HISTORY: dict[str, list[dict[str, Any]]] = {
    "payment-service": [
        {
            "version": "v2.14.3",
            "deployed_at": (_now() - timedelta(hours=2)).isoformat(),
            "deployed_by": "ci-bot",
            "commit": "a1b2c3d",
            "changelog": "Fix race condition in refund processing",
            "rollback_available": True,
        },
        {
            "version": "v2.14.2",
            "deployed_at": (_now() - timedelta(days=1)).isoformat(),
            "deployed_by": "engineer-alice",
            "commit": "e4f5g6h",
            "changelog": "Add retry logic for payment gateway timeouts",
            "rollback_available": True,
        },
        {
            "version": "v2.14.1",
            "deployed_at": (_now() - timedelta(days=3)).isoformat(),
            "deployed_by": "ci-bot",
            "commit": "i7j8k9l",
            "changelog": "Update payment SDK to v4.2",
            "rollback_available": True,
        },
        {
            "version": "v2.14.0",
            "deployed_at": (_now() - timedelta(days=7)).isoformat(),
            "deployed_by": "engineer-bob",
            "commit": "m0n1o2p",
            "changelog": "Add Apple Pay support",
            "rollback_available": True,
        },
        {
            "version": "v2.13.8",
            "deployed_at": (_now() - timedelta(days=14)).isoformat(),
            "deployed_by": "ci-bot",
            "commit": "q3r4s5t",
            "changelog": "Performance optimization for high-volume transactions",
            "rollback_available": False,
        },
    ],
    "order-processor": [
        {
            "version": "v1.8.2",
            "deployed_at": (_now() - timedelta(hours=6)).isoformat(),
            "deployed_by": "engineer-charlie",
            "commit": "u6v7w8x",
            "changelog": "Increase batch size for order aggregation",
            "rollback_available": True,
        },
        {
            "version": "v1.8.1",
            "deployed_at": (_now() - timedelta(days=2)).isoformat(),
            "deployed_by": "ci-bot",
            "commit": "y9z0a1b",
            "changelog": "Fix memory leak in order serialization",
            "rollback_available": True,
        },
        {
            "version": "v1.8.0",
            "deployed_at": (_now() - timedelta(days=5)).isoformat(),
            "deployed_by": "engineer-diana",
            "commit": "c2d3e4f",
            "changelog": "Add bulk order processing endpoint",
            "rollback_available": True,
        },
        {
            "version": "v1.7.9",
            "deployed_at": (_now() - timedelta(days=10)).isoformat(),
            "deployed_by": "ci-bot",
            "commit": "g5h6i7j",
            "changelog": "Upgrade FastAPI to 0.110",
            "rollback_available": True,
        },
        {
            "version": "v1.7.8",
            "deployed_at": (_now() - timedelta(days=15)).isoformat(),
            "deployed_by": "ci-bot",
            "commit": "k8l9m0n",
            "changelog": "Add distributed tracing spans",
            "rollback_available": False,
        },
    ],
    "api-gateway": [
        {
            "version": "v3.2.1",
            "deployed_at": (_now() - timedelta(days=1)).isoformat(),
            "deployed_by": "ci-bot",
            "commit": "o1p2q3r",
            "changelog": "Update rate limiting configuration",
            "rollback_available": True,
        },
        {
            "version": "v3.2.0",
            "deployed_at": (_now() - timedelta(days=4)).isoformat(),
            "deployed_by": "engineer-eve",
            "commit": "s4t5u6v",
            "changelog": "Add circuit breaker for downstream services",
            "rollback_available": True,
        },
    ],
    "auth-service": [
        {
            "version": "v2.6.0",
            "deployed_at": (_now() - timedelta(days=3)).isoformat(),
            "deployed_by": "ci-bot",
            "commit": "w7x8y9z",
            "changelog": "Add OIDC provider integration",
            "rollback_available": True,
        },
    ],
    "search-service": [
        {
            "version": "v3.0.1",
            "deployed_at": (_now() - timedelta(hours=4)).isoformat(),
            "deployed_by": "engineer-frank",
            "commit": "a0b1c2d",
            "changelog": "Fix Elasticsearch query timeout handling",
            "rollback_available": True,
        },
        {
            "version": "v3.0.0",
            "deployed_at": (_now() - timedelta(days=2)).isoformat(),
            "deployed_by": "ci-bot",
            "commit": "e3f4g5h",
            "changelog": "Major: Upgrade to Elasticsearch 8.x",
            "rollback_available": True,
        },
    ],
}


# ---------------------------------------------------------------------------
# On-Call Rotation
# ---------------------------------------------------------------------------

ONCALL_ROTATION: dict[str, dict[str, Any]] = {
    "payments": {
        "team": "payments",
        "l1_oncall": "engineer-alice",
        "l2_oncall": "engineer-bob",
        "l3_senior": "staff-eng-grace",
        "l4_manager": "em-henry",
        "slack_channel": "#payments-oncall",
        "escalation_policy": "payments-critical",
        "pagerduty_service": "PSVC001",
    },
    "fulfillment": {
        "team": "fulfillment",
        "l1_oncall": "engineer-charlie",
        "l2_oncall": "engineer-diana",
        "l3_senior": "staff-eng-ivan",
        "l4_manager": "em-julia",
        "slack_channel": "#fulfillment-oncall",
        "escalation_policy": "fulfillment-standard",
        "pagerduty_service": "PSVC002",
    },
    "platform": {
        "team": "platform",
        "l1_oncall": "engineer-eve",
        "l2_oncall": "engineer-frank",
        "l3_senior": "staff-eng-kate",
        "l4_manager": "em-leo",
        "slack_channel": "#platform-oncall",
        "escalation_policy": "platform-critical",
        "pagerduty_service": "PSVC003",
    },
    "identity": {
        "team": "identity",
        "l1_oncall": "engineer-mallory",
        "l2_oncall": "engineer-nick",
        "l3_senior": "staff-eng-olivia",
        "l4_manager": "em-peter",
        "slack_channel": "#identity-oncall",
        "escalation_policy": "identity-critical",
        "pagerduty_service": "PSVC004",
    },
    "observability": {
        "team": "observability",
        "l1_oncall": "engineer-quinn",
        "l2_oncall": "engineer-rachel",
        "l3_senior": "staff-eng-sam",
        "l4_manager": "em-tina",
        "slack_channel": "#observability-oncall",
        "escalation_policy": "observability-standard",
        "pagerduty_service": "PSVC005",
    },
    "ml-platform": {
        "team": "ml-platform",
        "l1_oncall": "engineer-uma",
        "l2_oncall": "engineer-victor",
        "l3_senior": "staff-eng-wendy",
        "l4_manager": "em-xavier",
        "slack_channel": "#ml-platform-oncall",
        "escalation_policy": "ml-platform-standard",
        "pagerduty_service": "PSVC006",
    },
    "data": {
        "team": "data",
        "l1_oncall": "engineer-yolanda",
        "l2_oncall": "engineer-zach",
        "l3_senior": "staff-eng-adam",
        "l4_manager": "em-beth",
        "slack_channel": "#data-oncall",
        "escalation_policy": "data-critical",
        "pagerduty_service": "PSVC007",
    },
    "search": {
        "team": "search",
        "l1_oncall": "engineer-frank",
        "l2_oncall": "engineer-gloria",
        "l3_senior": "staff-eng-hank",
        "l4_manager": "em-iris",
        "slack_channel": "#search-oncall",
        "escalation_policy": "search-standard",
        "pagerduty_service": "PSVC008",
    },
    "communications": {
        "team": "communications",
        "l1_oncall": "engineer-jack",
        "l2_oncall": "engineer-karen",
        "l3_senior": "staff-eng-larry",
        "l4_manager": "em-monica",
        "slack_channel": "#comms-oncall",
        "escalation_policy": "comms-standard",
        "pagerduty_service": "PSVC009",
    },
}


# ---------------------------------------------------------------------------
# Baseline Metrics (per service)
# ---------------------------------------------------------------------------

BASELINE_METRICS: dict[str, dict[str, Any]] = {
    "payment-service": {
        "cpu_pct": 45.0,
        "memory_pct": 60.0,
        "error_rate": 0.001,
        "p50_latency_ms": 45,
        "p99_latency_ms": 180,
        "requests_per_sec": 500,
        "active_connections": 120,
    },
    "order-processor": {
        "cpu_pct": 55.0,
        "memory_pct": 65.0,
        "error_rate": 0.003,
        "p50_latency_ms": 120,
        "p99_latency_ms": 450,
        "requests_per_sec": 200,
        "active_connections": 80,
    },
    "api-gateway": {
        "cpu_pct": 30.0,
        "memory_pct": 40.0,
        "error_rate": 0.0005,
        "p50_latency_ms": 8,
        "p99_latency_ms": 35,
        "requests_per_sec": 5000,
        "active_connections": 2000,
    },
    "auth-service": {
        "cpu_pct": 25.0,
        "memory_pct": 35.0,
        "error_rate": 0.0002,
        "p50_latency_ms": 15,
        "p99_latency_ms": 80,
        "requests_per_sec": 3000,
        "active_connections": 500,
    },
    "log-aggregator": {
        "cpu_pct": 60.0,
        "memory_pct": 70.0,
        "error_rate": 0.005,
        "p50_latency_ms": 200,
        "p99_latency_ms": 800,
        "requests_per_sec": 10000,
        "active_connections": 50,
    },
    "recommendation-engine": {
        "cpu_pct": 70.0,
        "memory_pct": 55.0,
        "error_rate": 0.008,
        "p50_latency_ms": 80,
        "p99_latency_ms": 250,
        "requests_per_sec": 400,
        "active_connections": 60,
    },
    "db-proxy": {
        "cpu_pct": 20.0,
        "memory_pct": 30.0,
        "error_rate": 0.0001,
        "p50_latency_ms": 2,
        "p99_latency_ms": 8,
        "requests_per_sec": 15000,
        "active_connections": 150,
    },
    "search-service": {
        "cpu_pct": 50.0,
        "memory_pct": 55.0,
        "error_rate": 0.003,
        "p50_latency_ms": 30,
        "p99_latency_ms": 120,
        "requests_per_sec": 1200,
        "active_connections": 200,
    },
    "notification-service": {
        "cpu_pct": 35.0,
        "memory_pct": 45.0,
        "error_rate": 0.005,
        "p50_latency_ms": 50,
        "p99_latency_ms": 200,
        "requests_per_sec": 5000,
        "active_connections": 100,
    },
    "cdn-edge": {
        "cpu_pct": 15.0,
        "memory_pct": 25.0,
        "error_rate": 0.0001,
        "p50_latency_ms": 3,
        "p99_latency_ms": 15,
        "requests_per_sec": 50000,
        "active_connections": 10000,
    },
}


def get_service_info(service: str) -> dict[str, Any]:
    """Look up service registry entry."""
    return SERVICE_REGISTRY.get(service, {})


def get_deploy_history(service: str) -> list[dict[str, Any]]:
    """Return recent deployment history for a service."""
    return DEPLOY_HISTORY.get(service, [])


def get_oncall(team: str) -> dict[str, Any]:
    """Look up on-call rotation for a team."""
    return ONCALL_ROTATION.get(team, {})


def get_baseline_metrics(service: str) -> dict[str, Any]:
    """Return baseline metric thresholds for a service."""
    return BASELINE_METRICS.get(service, {})
