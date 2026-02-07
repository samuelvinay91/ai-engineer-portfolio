"""Mock alert payloads simulating various infrastructure incidents.

Each alert represents a realistic monitoring system notification from
sources such as Datadog, CloudWatch, Prometheus, and PagerDuty.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from incident_response.models import Alert


def _now() -> datetime:
    """Return current UTC timestamp."""
    return datetime.now(tz=timezone.utc)


MOCK_ALERTS: list[Alert] = [
    Alert(
        id="ALT-001",
        source="datadog",
        title="CPU spike on payment-service",
        description=(
            "payment-service CPU usage exceeded 95% threshold for 5 minutes "
            "across 3/4 pods in us-east-1. Correlation with increased 5xx "
            "error rate observed."
        ),
        service="payment-service",
        host="payment-service-pod-7f8d9",
        timestamp=_now() - timedelta(minutes=2),
        raw_data={
            "monitor_id": "mon-12345",
            "metric": "system.cpu.user",
            "threshold": 95.0,
            "current_value": 98.3,
            "duration_seconds": 300,
            "tags": ["env:production", "region:us-east-1", "team:payments"],
        },
    ),
    Alert(
        id="ALT-002",
        source="cloudwatch",
        title="OOM kills on order-processor",
        description=(
            "order-processor has experienced 12 OOM kills in the last 15 minutes. "
            "Container memory limit of 2Gi consistently exceeded. "
            "Possible memory leak in batch processing module."
        ),
        service="order-processor",
        host="order-processor-pod-3a4b5",
        timestamp=_now() - timedelta(minutes=5),
        raw_data={
            "alarm_name": "order-processor-oom",
            "metric": "container.memory.usage",
            "limit_bytes": 2147483648,
            "peak_bytes": 2415919104,
            "oom_kill_count": 12,
            "restart_count": 12,
            "tags": ["env:production", "cluster:main", "team:fulfillment"],
        },
    ),
    Alert(
        id="ALT-003",
        source="prometheus",
        title="5xx error surge on api-gateway",
        description=(
            "api-gateway HTTP 5xx error rate jumped from 0.1% to 12.4% "
            "over the last 10 minutes. Upstream services payment-service "
            "and order-processor showing elevated latency."
        ),
        service="api-gateway",
        host="api-gateway-pod-9c8d7",
        timestamp=_now() - timedelta(minutes=1),
        raw_data={
            "alertname": "HighErrorRate",
            "metric": "http_requests_total{status=~'5..'}",
            "baseline_rate": 0.001,
            "current_rate": 0.124,
            "affected_endpoints": ["/api/v1/orders", "/api/v1/payments", "/api/v1/checkout"],
            "labels": {"severity": "critical", "team": "platform"},
        },
    ),
    Alert(
        id="ALT-004",
        source="pagerduty",
        title="TLS certificate expiry on auth-service",
        description=(
            "auth-service TLS certificate expires in 48 hours. "
            "Certificate CN=auth.example.com issued by Let's Encrypt. "
            "Auto-renewal may have failed due to DNS challenge timeout."
        ),
        service="auth-service",
        host="auth-service-pod-1e2f3",
        timestamp=_now() - timedelta(hours=1),
        raw_data={
            "cert_cn": "auth.example.com",
            "expiry_date": (_now() + timedelta(hours=48)).isoformat(),
            "issuer": "Let's Encrypt Authority X3",
            "auto_renew_status": "failed",
            "last_renewal_attempt": (_now() - timedelta(hours=6)).isoformat(),
            "error": "DNS challenge timeout after 120s",
        },
    ),
    Alert(
        id="ALT-005",
        source="datadog",
        title="Disk usage critical on log-aggregator",
        description=(
            "log-aggregator disk usage at 94% on /data volume. "
            "Growth rate: 2GB/hour. Estimated time to full: 3 hours. "
            "Log retention policy may need adjustment."
        ),
        service="log-aggregator",
        host="log-aggregator-pod-6g7h8",
        timestamp=_now() - timedelta(minutes=15),
        raw_data={
            "monitor_id": "mon-67890",
            "metric": "system.disk.in_use",
            "mount_point": "/data",
            "current_pct": 94.2,
            "total_bytes": 536870912000,
            "used_bytes": 505765068800,
            "growth_rate_bytes_per_hour": 2147483648,
            "tags": ["env:production", "team:observability"],
        },
    ),
    Alert(
        id="ALT-006",
        source="prometheus",
        title="Memory leak in recommendation-engine",
        description=(
            "recommendation-engine heap memory usage growing linearly at "
            "~100MB/hour with no corresponding increase in traffic. "
            "Currently at 3.2GB of 4GB limit. No GC activity detected."
        ),
        service="recommendation-engine",
        host="recommendation-engine-pod-4i5j6",
        timestamp=_now() - timedelta(minutes=30),
        raw_data={
            "alertname": "MemoryLeakSuspected",
            "metric": "jvm_heap_memory_used_bytes",
            "current_bytes": 3435973837,
            "limit_bytes": 4294967296,
            "growth_rate_mb_per_hour": 100,
            "gc_pause_count_last_hour": 0,
            "labels": {"severity": "warning", "team": "ml-platform"},
        },
    ),
    Alert(
        id="ALT-007",
        source="cloudwatch",
        title="Connection pool exhaustion on db-proxy",
        description=(
            "db-proxy connection pool utilization at 98%. "
            "Active connections: 196/200. Connection wait queue depth: 47. "
            "Average connection checkout time: 8.2s (SLA: 100ms)."
        ),
        service="db-proxy",
        host="db-proxy-pod-2k3l4",
        timestamp=_now() - timedelta(minutes=3),
        raw_data={
            "alarm_name": "db-proxy-pool-exhaustion",
            "metric": "connection_pool.active",
            "max_connections": 200,
            "active_connections": 196,
            "waiting_requests": 47,
            "avg_checkout_ms": 8200,
            "sla_checkout_ms": 100,
            "tags": ["env:production", "cluster:main", "team:data"],
        },
    ),
    Alert(
        id="ALT-008",
        source="prometheus",
        title="High latency on search-service",
        description=(
            "search-service p99 latency increased from 120ms to 2.4s "
            "in the last 20 minutes. Query throughput stable at 1.2k QPS. "
            "Elasticsearch cluster health yellow (1 unassigned shard)."
        ),
        service="search-service",
        host="search-service-pod-5m6n7",
        timestamp=_now() - timedelta(minutes=8),
        raw_data={
            "alertname": "HighLatency",
            "metric": "http_request_duration_seconds",
            "quantile": "0.99",
            "baseline_ms": 120,
            "current_ms": 2400,
            "throughput_qps": 1200,
            "es_cluster_health": "yellow",
            "unassigned_shards": 1,
            "labels": {"severity": "high", "team": "search"},
        },
    ),
    Alert(
        id="ALT-009",
        source="datadog",
        title="Kafka consumer lag spike on notification-service",
        description=(
            "notification-service Kafka consumer group 'notifications-v2' "
            "lag exceeded 50,000 messages on topic 'user-events'. "
            "Processing rate dropped from 5k/s to 200/s."
        ),
        service="notification-service",
        host="notification-service-pod-8o9p0",
        timestamp=_now() - timedelta(minutes=12),
        raw_data={
            "monitor_id": "mon-11111",
            "metric": "kafka.consumer.lag",
            "consumer_group": "notifications-v2",
            "topic": "user-events",
            "current_lag": 50342,
            "baseline_lag": 100,
            "processing_rate_per_sec": 200,
            "baseline_rate_per_sec": 5000,
            "tags": ["env:production", "team:communications"],
        },
    ),
    Alert(
        id="ALT-010",
        source="pagerduty",
        title="DNS resolution failures on cdn-edge",
        description=(
            "cdn-edge nodes in us-west-2 reporting intermittent DNS "
            "resolution failures for internal services. Failure rate: 8%. "
            "CoreDNS pods restarted 3 times in the last hour."
        ),
        service="cdn-edge",
        host="cdn-edge-node-us-west-2a",
        timestamp=_now() - timedelta(minutes=6),
        raw_data={
            "failure_rate_pct": 8.0,
            "affected_region": "us-west-2",
            "coredns_restarts": 3,
            "resolution_timeout_ms": 5000,
            "affected_domains": [
                "payment-service.internal",
                "auth-service.internal",
                "order-processor.internal",
            ],
        },
    ),
]


def get_alert_by_id(alert_id: str) -> Alert | None:
    """Look up a mock alert by its ID."""
    for alert in MOCK_ALERTS:
        if alert.id == alert_id:
            return alert
    return None


def get_alerts_by_service(service: str) -> list[Alert]:
    """Return all mock alerts for a given service."""
    return [a for a in MOCK_ALERTS if a.service == service]
