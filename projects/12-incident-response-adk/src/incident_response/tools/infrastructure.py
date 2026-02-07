"""Mock infrastructure query tools for diagnostic agents.

Simulates querying centralized logging, metrics platforms, configuration
management systems, and Kubernetes API calls. Returns realistic mock data
based on the service being investigated.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Any

import structlog

from incident_response.mock_data.infrastructure import (
    BASELINE_METRICS,
    SERVICE_REGISTRY,
)

logger = structlog.get_logger(__name__)


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(tz=timezone.utc)


async def query_logs(
    service: str,
    timerange_minutes: int = 30,
    severity: str = "error",
) -> dict[str, Any]:
    """Query centralized log index for correlated errors.

    Simulates an Elasticsearch / Loki / CloudWatch Logs query returning
    structured log entries matching the given service and severity filter.

    Args:
        service: Service name to search logs for.
        timerange_minutes: How far back to search.
        severity: Minimum log severity (debug, info, warn, error, fatal).

    Returns:
        Dictionary with log entries, patterns detected, and summary stats.
    """
    logger.info("query_logs", service=service, timerange=timerange_minutes)

    # Service-specific mock log patterns
    log_patterns: dict[str, list[dict[str, Any]]] = {
        "payment-service": [
            {
                "timestamp": (_now() - timedelta(minutes=1)).isoformat(),
                "level": "ERROR",
                "message": "Connection timeout to payment gateway after 30s",
                "logger": "com.example.payments.GatewayClient",
                "trace_id": "abc123def456",
                "count": 47,
            },
            {
                "timestamp": (_now() - timedelta(minutes=3)).isoformat(),
                "level": "ERROR",
                "message": "Thread pool exhausted: ActiveCount=200, MaxSize=200",
                "logger": "com.example.payments.ThreadPoolManager",
                "trace_id": "ghi789jkl012",
                "count": 12,
            },
            {
                "timestamp": (_now() - timedelta(minutes=5)).isoformat(),
                "level": "WARN",
                "message": "GC pause exceeded 500ms: type=Full, duration=1247ms",
                "logger": "com.example.payments.GCMonitor",
                "trace_id": "",
                "count": 8,
            },
        ],
        "order-processor": [
            {
                "timestamp": (_now() - timedelta(minutes=2)).isoformat(),
                "level": "FATAL",
                "message": "OutOfMemoryError: Java heap space",
                "logger": "order_processor.batch",
                "trace_id": "mno345pqr678",
                "count": 12,
                "stack_trace": (
                    "java.lang.OutOfMemoryError: Java heap space\n"
                    "  at order_processor.batch.OrderAggregator.aggregate(OrderAggregator.java:142)\n"
                    "  at order_processor.batch.BatchRunner.run(BatchRunner.java:87)"
                ),
            },
            {
                "timestamp": (_now() - timedelta(minutes=8)).isoformat(),
                "level": "ERROR",
                "message": "Failed to serialize order batch: payload exceeds 64MB limit",
                "logger": "order_processor.serializer",
                "trace_id": "stu901vwx234",
                "count": 5,
            },
        ],
        "api-gateway": [
            {
                "timestamp": (_now() - timedelta(minutes=1)).isoformat(),
                "level": "ERROR",
                "message": "Upstream service unavailable: payment-service returned 503",
                "logger": "api_gateway.proxy",
                "trace_id": "yza567bcd890",
                "count": 234,
            },
            {
                "timestamp": (_now() - timedelta(minutes=2)).isoformat(),
                "level": "ERROR",
                "message": "Circuit breaker OPEN for payment-service: 50 failures in 60s",
                "logger": "api_gateway.circuit_breaker",
                "trace_id": "",
                "count": 3,
            },
        ],
        "search-service": [
            {
                "timestamp": (_now() - timedelta(minutes=5)).isoformat(),
                "level": "ERROR",
                "message": "Elasticsearch query timeout after 10s on index 'products-v3'",
                "logger": "search_service.es_client",
                "trace_id": "efg123hij456",
                "count": 89,
            },
            {
                "timestamp": (_now() - timedelta(minutes=10)).isoformat(),
                "level": "WARN",
                "message": "Elasticsearch cluster health YELLOW: 1 unassigned replica shard",
                "logger": "search_service.health_monitor",
                "trace_id": "",
                "count": 1,
            },
        ],
        "db-proxy": [
            {
                "timestamp": (_now() - timedelta(minutes=1)).isoformat(),
                "level": "ERROR",
                "message": "Connection pool exhausted: 200/200 active, 47 waiting",
                "logger": "db_proxy.pool",
                "trace_id": "",
                "count": 156,
            },
            {
                "timestamp": (_now() - timedelta(minutes=3)).isoformat(),
                "level": "WARN",
                "message": "Slow query detected: SELECT * FROM orders WHERE ... took 12.4s",
                "logger": "db_proxy.query_analyzer",
                "trace_id": "klm789nop012",
                "count": 23,
            },
        ],
    }

    entries = log_patterns.get(service, [
        {
            "timestamp": (_now() - timedelta(minutes=5)).isoformat(),
            "level": "ERROR",
            "message": f"Unexpected error in {service}: internal processing failure",
            "logger": f"{service}.main",
            "trace_id": f"generic-{random.randint(1000, 9999)}",
            "count": random.randint(1, 20),
        },
    ])

    return {
        "service": service,
        "timerange_minutes": timerange_minutes,
        "severity_filter": severity,
        "total_entries": sum(e.get("count", 1) for e in entries),
        "unique_patterns": len(entries),
        "entries": entries,
        "query_time_ms": random.randint(50, 500),
    }


async def query_metrics(
    service: str,
    metric_name: str = "all",
    timerange_minutes: int = 30,
) -> dict[str, Any]:
    """Query metrics platform for service health indicators.

    Simulates a Datadog / Prometheus / CloudWatch Metrics query returning
    current and baseline values for key infrastructure metrics.

    Args:
        service: Service name to query metrics for.
        metric_name: Specific metric or "all" for full dashboard.
        timerange_minutes: How far back to query.

    Returns:
        Dictionary with current values, baselines, and anomaly flags.
    """
    logger.info("query_metrics", service=service, metric=metric_name)

    baseline = BASELINE_METRICS.get(service, {
        "cpu_pct": 40.0,
        "memory_pct": 50.0,
        "error_rate": 0.005,
        "p50_latency_ms": 50,
        "p99_latency_ms": 200,
        "requests_per_sec": 1000,
        "active_connections": 100,
    })

    # Generate "current" values with some anomalies for known problem services
    anomaly_multipliers: dict[str, dict[str, float]] = {
        "payment-service": {"cpu_pct": 2.2, "error_rate": 15.0, "p99_latency_ms": 3.0},
        "order-processor": {"memory_pct": 1.5, "error_rate": 8.0},
        "api-gateway": {"error_rate": 124.0, "p99_latency_ms": 5.0},
        "search-service": {"p99_latency_ms": 20.0, "p50_latency_ms": 8.0},
        "db-proxy": {"active_connections": 1.3, "p99_latency_ms": 100.0},
        "log-aggregator": {"cpu_pct": 1.4, "memory_pct": 1.3},
        "recommendation-engine": {"memory_pct": 1.45, "cpu_pct": 1.2},
    }

    multipliers = anomaly_multipliers.get(service, {})
    current: dict[str, float] = {}
    anomalies: list[dict[str, Any]] = []

    for metric, baseline_val in baseline.items():
        mult = multipliers.get(metric, 1.0 + random.uniform(-0.05, 0.05))
        current_val = round(baseline_val * mult, 4)
        current[metric] = current_val

        # Flag as anomaly if more than 50% above baseline
        if mult > 1.5:
            anomalies.append({
                "metric": metric,
                "current": current_val,
                "baseline": baseline_val,
                "deviation_pct": round((mult - 1.0) * 100, 1),
                "severity": "critical" if mult > 3.0 else "warning",
            })

    return {
        "service": service,
        "timerange_minutes": timerange_minutes,
        "current": current,
        "baseline": dict(baseline),
        "anomalies": anomalies,
        "anomaly_count": len(anomalies),
        "query_time_ms": random.randint(20, 200),
    }


async def check_config(service: str) -> dict[str, Any]:
    """Compare running configuration against expected configuration.

    Simulates a configuration drift detection system that compares
    the live Kubernetes deployment spec against the declared GitOps state.

    Args:
        service: Service name to audit configuration for.

    Returns:
        Dictionary with config comparison, drift items, and compliance status.
    """
    logger.info("check_config", service=service)

    svc_info = SERVICE_REGISTRY.get(service, {})
    if not svc_info:
        return {
            "service": service,
            "status": "unknown",
            "message": f"Service {service} not found in registry",
            "drifts": [],
        }

    # Simulate config drift scenarios per service
    drift_scenarios: dict[str, list[dict[str, Any]]] = {
        "payment-service": [
            {
                "field": "env.JAVA_OPTS",
                "expected": "-Xmx3g -Xms3g -XX:+UseG1GC",
                "actual": "-Xmx2g -Xms1g -XX:+UseParallelGC",
                "severity": "high",
                "impact": "Suboptimal GC configuration may cause long pause times",
            },
            {
                "field": "resources.limits.cpu",
                "expected": "2000m",
                "actual": "1500m",
                "severity": "medium",
                "impact": "CPU limit lower than declared, may cause throttling",
            },
        ],
        "order-processor": [
            {
                "field": "resources.limits.memory",
                "expected": "4Gi",
                "actual": "2Gi",
                "severity": "critical",
                "impact": "Memory limit set to half of recommended, causing OOM kills",
            },
            {
                "field": "env.BATCH_SIZE",
                "expected": "100",
                "actual": "1000",
                "severity": "high",
                "impact": "Batch size 10x higher than recommended, excessive memory usage",
            },
        ],
        "api-gateway": [
            {
                "field": "env.CIRCUIT_BREAKER_THRESHOLD",
                "expected": "10",
                "actual": "50",
                "severity": "medium",
                "impact": "Circuit breaker threshold too high, slow failure detection",
            },
        ],
        "search-service": [
            {
                "field": "env.ES_QUERY_TIMEOUT_MS",
                "expected": "5000",
                "actual": "30000",
                "severity": "medium",
                "impact": "Elasticsearch query timeout too high, holding connections",
            },
            {
                "field": "image",
                "expected": "registry.example.com/search-service:v3.0.1",
                "actual": "registry.example.com/search-service:v3.0.0",
                "severity": "high",
                "impact": "Running previous version, missing timeout fix",
            },
        ],
    }

    drifts = drift_scenarios.get(service, [])
    has_drift = len(drifts) > 0

    return {
        "service": service,
        "status": "drifted" if has_drift else "compliant",
        "expected_image": svc_info.get("image", "unknown"),
        "replicas": {
            "expected": svc_info.get("replicas", 1),
            "actual": svc_info.get("replicas", 1),
        },
        "drifts": drifts,
        "drift_count": len(drifts),
        "last_sync": (_now() - timedelta(minutes=random.randint(5, 60))).isoformat(),
        "gitops_repo": "github.com/example/k8s-manifests",
    }


async def kubectl_exec(command: str) -> dict[str, Any]:
    """Execute a simulated kubectl command.

    Provides mock output for common kubectl commands used in incident
    response: get pods, describe pod, top pods, logs, etc.

    Args:
        command: The kubectl command to simulate (without 'kubectl' prefix).

    Returns:
        Dictionary with command output and execution metadata.
    """
    logger.info("kubectl_exec", command=command)

    # Parse common commands
    parts = command.strip().split()
    if not parts:
        return {"success": False, "output": "Empty command", "exit_code": 1}

    verb = parts[0]

    if verb == "get" and "pods" in command:
        return {
            "success": True,
            "output": (
                "NAME                              READY   STATUS    RESTARTS   AGE\n"
                "payment-service-7f8d9-abc12       1/1     Running   0          2h\n"
                "payment-service-7f8d9-def34       1/1     Running   0          2h\n"
                "payment-service-7f8d9-ghi56       1/1     Running   3          2h\n"
                "payment-service-7f8d9-jkl78       0/1     CrashLoopBackOff   5   2h\n"
            ),
            "exit_code": 0,
        }

    if verb == "top" and "pods" in command:
        return {
            "success": True,
            "output": (
                "NAME                              CPU(cores)   MEMORY(bytes)\n"
                "payment-service-7f8d9-abc12       1847m        3421Mi\n"
                "payment-service-7f8d9-def34       1923m        3512Mi\n"
                "payment-service-7f8d9-ghi56       1756m        3287Mi\n"
                "payment-service-7f8d9-jkl78       45m          128Mi\n"
            ),
            "exit_code": 0,
        }

    if verb == "rollout":
        return {
            "success": True,
            "output": "deployment.apps/payment-service restarted\n",
            "exit_code": 0,
        }

    if verb == "scale":
        return {
            "success": True,
            "output": "deployment.apps/payment-service scaled\n",
            "exit_code": 0,
        }

    return {
        "success": True,
        "output": f"Simulated output for: kubectl {command}",
        "exit_code": 0,
    }
