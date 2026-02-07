"""Log Analyzer Agent.

Searches centralized log indices for correlated errors, stack traces,
error spikes, and timeout patterns related to the incident.
"""

from __future__ import annotations

from typing import Any

import structlog

from incident_response.agents.base import LlmAgent
from incident_response.models import DiagnosticResult
from incident_response.tools.infrastructure import query_logs

logger = structlog.get_logger(__name__)


class LogAnalyzerAgent(LlmAgent):
    """Analyzes application and infrastructure logs for incident clues.

    Searches log indices for error patterns, stack traces, correlation
    with upstream/downstream services, and temporal anomalies.
    """

    def __init__(self) -> None:
        super().__init__(
            name="log_analyzer",
            instruction=(
                "You are a log analysis specialist. Search centralized logs "
                "for error patterns, stack traces, timeout events, and "
                "anomalous log volumes related to the incident. Correlate "
                "findings across services and time windows."
            ),
            description="Analyzes logs for error patterns and anomalies",
            tools=[query_logs],
        )

    async def _heuristic_run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Analyze logs for the affected service.

        Expects:
            context["alert"]: Alert instance
            context["incident_context"]: IncidentContext (optional)

        Sets:
            context["log_diagnostics"]: DiagnosticResult instance
        """
        alert = context["alert"]
        service = alert.service

        logger.info("analyzing_logs", service=service, alert_id=alert.id)

        # Query logs for the affected service
        log_results = await query_logs(service, timerange_minutes=30, severity="error")

        # Extract findings
        findings: list[str] = []
        anomalies: list[dict[str, Any]] = []
        severity_indicators: list[str] = []

        entries = log_results.get("entries", [])
        total_errors = log_results.get("total_entries", 0)
        unique_patterns = log_results.get("unique_patterns", 0)

        findings.append(
            f"Found {total_errors} error log entries across "
            f"{unique_patterns} unique patterns in the last 30 minutes."
        )

        for entry in entries:
            level = entry.get("level", "ERROR")
            message = entry.get("message", "")
            count = entry.get("count", 1)

            # Detect stack traces
            if entry.get("stack_trace"):
                findings.append(
                    f"Stack trace detected: {message} (occurred {count} times)"
                )
                severity_indicators.append("stack_trace_present")
                anomalies.append({
                    "type": "stack_trace",
                    "message": message,
                    "count": count,
                    "stack_trace": entry["stack_trace"][:500],
                })

            # Detect error spikes (more than 10 of the same error)
            if count > 10:
                findings.append(
                    f"Error spike: '{message}' occurred {count} times "
                    f"(level: {level})"
                )
                severity_indicators.append("error_spike")
                anomalies.append({
                    "type": "error_spike",
                    "message": message,
                    "count": count,
                    "level": level,
                })

            # Detect timeout patterns
            if any(kw in message.lower() for kw in ["timeout", "timed out", "deadline exceeded"]):
                findings.append(f"Timeout pattern: {message} ({count} occurrences)")
                severity_indicators.append("timeout_pattern")
                anomalies.append({
                    "type": "timeout",
                    "message": message,
                    "count": count,
                })

            # Detect FATAL level logs
            if level == "FATAL":
                findings.append(f"FATAL log detected: {message}")
                severity_indicators.append("fatal_log")

        # Also check dependency service logs
        dependencies = context.get("service_info", {}).get("dependencies", [])
        if dependencies:
            findings.append(
                f"Service depends on: {', '.join(dependencies)}. "
                "Dependency logs should also be reviewed."
            )

        diagnostic = DiagnosticResult(
            agent_name=self.name,
            findings=findings,
            anomalies=anomalies,
            severity_indicators=list(set(severity_indicators)),
            raw_data=log_results,
        )

        context["log_diagnostics"] = diagnostic

        logger.info(
            "log_analysis_complete",
            service=service,
            findings=len(findings),
            anomalies=len(anomalies),
        )

        return context
