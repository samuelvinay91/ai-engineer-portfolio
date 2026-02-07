"""Metrics Checker Agent.

Queries infrastructure metrics (CPU, memory, error rate, latency) and
detects anomalies by comparing current values against baseline thresholds.
"""

from __future__ import annotations

from typing import Any

import structlog

from incident_response.agents.base import LlmAgent
from incident_response.models import DiagnosticResult
from incident_response.tools.infrastructure import query_metrics

logger = structlog.get_logger(__name__)


class MetricsCheckerAgent(LlmAgent):
    """Checks infrastructure metrics for anomalies and threshold violations.

    Compares current metric values against historical baselines to detect
    CPU spikes, memory pressure, error rate increases, and latency degradation.
    """

    def __init__(self) -> None:
        super().__init__(
            name="metrics_checker",
            instruction=(
                "You are a metrics analysis specialist. Query infrastructure "
                "metrics for the affected service and compare against baselines. "
                "Identify anomalies in CPU, memory, error rate, latency, and "
                "connection counts. Highlight any metrics that deviate "
                "significantly from normal operating ranges."
            ),
            description="Checks metrics for threshold violations and anomalies",
            tools=[query_metrics],
        )

    async def _heuristic_run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Check metrics for anomalies against baselines.

        Expects:
            context["alert"]: Alert instance

        Sets:
            context["metrics_diagnostics"]: DiagnosticResult instance
        """
        alert = context["alert"]
        service = alert.service

        logger.info("checking_metrics", service=service, alert_id=alert.id)

        # Query all metrics for the service
        metrics_data = await query_metrics(service, metric_name="all")

        findings: list[str] = []
        anomalies: list[dict[str, Any]] = []
        severity_indicators: list[str] = []

        current = metrics_data.get("current", {})
        baseline = metrics_data.get("baseline", {})
        detected_anomalies = metrics_data.get("anomalies", [])

        # Report on each detected anomaly
        for anomaly in detected_anomalies:
            metric_name = anomaly["metric"]
            current_val = anomaly["current"]
            baseline_val = anomaly["baseline"]
            deviation = anomaly["deviation_pct"]
            sev = anomaly.get("severity", "warning")

            finding = (
                f"Metric '{metric_name}' is {deviation}% above baseline: "
                f"current={current_val}, baseline={baseline_val}"
            )
            findings.append(finding)
            anomalies.append(anomaly)

            # Map metric anomalies to severity indicators
            if metric_name == "error_rate" and deviation > 500:
                severity_indicators.append("extreme_error_rate")
            elif metric_name == "cpu_pct" and current_val > 90:
                severity_indicators.append("cpu_critical")
            elif metric_name == "memory_pct" and current_val > 90:
                severity_indicators.append("memory_critical")
            elif "latency" in metric_name and deviation > 200:
                severity_indicators.append("latency_degradation")
            elif metric_name == "active_connections" and deviation > 80:
                severity_indicators.append("connection_pressure")

            if sev == "critical":
                severity_indicators.append(f"{metric_name}_critical")

        # Check for metrics within normal range (good to report as well)
        normal_metrics = []
        for metric_name, val in current.items():
            baseline_val = baseline.get(metric_name, val)
            if baseline_val > 0:
                ratio = val / baseline_val
                if 0.8 <= ratio <= 1.5:
                    normal_metrics.append(metric_name)

        if normal_metrics:
            findings.append(
                f"Metrics within normal range: {', '.join(normal_metrics)}"
            )

        # Summary finding
        if not detected_anomalies:
            findings.insert(0, f"No significant metric anomalies detected for {service}.")
        else:
            findings.insert(
                0,
                f"Detected {len(detected_anomalies)} metric anomalies for {service}.",
            )

        diagnostic = DiagnosticResult(
            agent_name=self.name,
            findings=findings,
            anomalies=anomalies,
            severity_indicators=list(set(severity_indicators)),
            raw_data=metrics_data,
        )

        context["metrics_diagnostics"] = diagnostic

        logger.info(
            "metrics_check_complete",
            service=service,
            anomaly_count=len(detected_anomalies),
            findings=len(findings),
        )

        return context
