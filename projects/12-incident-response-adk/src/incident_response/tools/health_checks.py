"""Mock health check tools for post-remediation verification.

Simulates checking service health endpoints, error rates, latency, and
uptime after a remediation action has been applied.
"""

from __future__ import annotations

import random
from datetime import datetime, timezone
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


async def check_service_health(
    service: str,
    iteration: int = 1,
) -> dict[str, Any]:
    """Check the health of a service after remediation.

    Simulates health endpoint checks, error rate monitoring, latency
    measurement, and uptime verification. Results improve with each
    iteration to simulate successful remediation taking effect.

    Args:
        service: Service name to health-check.
        iteration: Which remediation iteration this check follows
                   (higher = more likely to be healthy).

    Returns:
        Dictionary with health status, metrics, and pass/fail verdict.
    """
    logger.info("health_check", service=service, iteration=iteration)

    # Probability of passing increases with each iteration
    # Iteration 1: 30%, Iteration 2: 65%, Iteration 3: 90%
    pass_probability = min(0.3 + (iteration - 1) * 0.35, 0.95)
    is_healthy = random.random() < pass_probability

    if is_healthy:
        # Healthy state
        error_rate = round(random.uniform(0.0001, 0.005), 4)
        p99_latency_ms = random.randint(20, 200)
        http_status = 200
        uptime_seconds = random.randint(60, 7200)
        checks_passed = ["http_health", "error_rate", "latency_sla", "dependency_check"]
        checks_failed: list[str] = []
    else:
        # Still unhealthy
        error_rate = round(random.uniform(0.05, 0.25), 4)
        p99_latency_ms = random.randint(1000, 10000)
        http_status = random.choice([200, 503, 502])
        uptime_seconds = random.randint(5, 60)
        checks_passed = ["dependency_check"] if random.random() > 0.5 else []
        checks_failed = [
            check
            for check in ["http_health", "error_rate", "latency_sla", "dependency_check"]
            if check not in checks_passed
        ]

    result = {
        "service": service,
        "iteration": iteration,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "healthy": is_healthy,
        "http_status": http_status,
        "error_rate": error_rate,
        "p99_latency_ms": p99_latency_ms,
        "uptime_seconds": uptime_seconds,
        "checks": {
            "passed": checks_passed,
            "failed": checks_failed,
            "total": len(checks_passed) + len(checks_failed),
        },
        "verdict": "PASS" if is_healthy else "FAIL",
        "evidence": [],
    }

    # Add evidence details
    if is_healthy:
        result["evidence"] = [
            f"Health endpoint returned HTTP {http_status}",
            f"Error rate {error_rate:.4f} below threshold 0.01",
            f"P99 latency {p99_latency_ms}ms within SLA",
            f"Service uptime: {uptime_seconds}s since last restart",
        ]
    else:
        result["evidence"] = [
            f"Health endpoint returned HTTP {http_status}",
            f"Error rate {error_rate:.4f} exceeds threshold 0.01",
            f"P99 latency {p99_latency_ms}ms exceeds SLA",
            "Service has not stabilized after remediation",
        ]

    return result
