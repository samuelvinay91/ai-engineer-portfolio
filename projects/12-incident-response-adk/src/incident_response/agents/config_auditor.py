"""Config Auditor Agent.

Compares running service configuration against expected/declared config,
detecting drift in environment variables, resource limits, image versions,
and other Kubernetes deployment parameters.
"""

from __future__ import annotations

from typing import Any

import structlog

from incident_response.agents.base import LlmAgent
from incident_response.models import DiagnosticResult
from incident_response.tools.infrastructure import check_config

logger = structlog.get_logger(__name__)


class ConfigAuditorAgent(LlmAgent):
    """Audits service configuration for drift and misconfigurations.

    Compares the live running configuration against the declared GitOps
    state to detect environment variable changes, resource limit
    modifications, and image version mismatches.
    """

    def __init__(self) -> None:
        super().__init__(
            name="config_auditor",
            instruction=(
                "You are a configuration audit specialist. Compare the running "
                "configuration of the affected service against its expected "
                "state. Look for: changed environment variables, modified "
                "resource limits, different image versions, and any other "
                "configuration drift that could explain the incident."
            ),
            description="Detects configuration drift between live and declared state",
            tools=[check_config],
        )

    async def _heuristic_run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Audit configuration for drift.

        Expects:
            context["alert"]: Alert instance

        Sets:
            context["config_diagnostics"]: DiagnosticResult instance
        """
        alert = context["alert"]
        service = alert.service

        logger.info("auditing_config", service=service, alert_id=alert.id)

        # Check configuration drift
        config_data = await check_config(service)

        findings: list[str] = []
        anomalies: list[dict[str, Any]] = []
        severity_indicators: list[str] = []

        status = config_data.get("status", "unknown")
        drifts = config_data.get("drifts", [])

        if status == "compliant":
            findings.append(
                f"No configuration drift detected for {service}. "
                "Running config matches declared GitOps state."
            )
        elif status == "drifted":
            findings.append(
                f"Configuration drift detected for {service}: "
                f"{len(drifts)} drift item(s) found."
            )

            for drift in drifts:
                field = drift.get("field", "unknown")
                expected = drift.get("expected", "N/A")
                actual = drift.get("actual", "N/A")
                drift_severity = drift.get("severity", "medium")
                impact = drift.get("impact", "Unknown impact")

                finding = (
                    f"Config drift in '{field}': "
                    f"expected='{expected}', actual='{actual}'. "
                    f"Impact: {impact}"
                )
                findings.append(finding)

                anomalies.append({
                    "type": "config_drift",
                    "field": field,
                    "expected": expected,
                    "actual": actual,
                    "severity": drift_severity,
                    "impact": impact,
                })

                # Map drift types to severity indicators
                if "memory" in field.lower() or "cpu" in field.lower():
                    severity_indicators.append("resource_limit_drift")
                if "image" in field.lower():
                    severity_indicators.append("image_version_mismatch")
                if "env" in field.lower():
                    severity_indicators.append("env_var_drift")
                if drift_severity == "critical":
                    severity_indicators.append("critical_config_drift")

        else:
            findings.append(
                f"Unable to determine configuration status for {service}."
            )

        # Add GitOps sync info
        last_sync = config_data.get("last_sync", "unknown")
        findings.append(f"Last GitOps sync: {last_sync}")

        diagnostic = DiagnosticResult(
            agent_name=self.name,
            findings=findings,
            anomalies=anomalies,
            severity_indicators=list(set(severity_indicators)),
            raw_data=config_data,
        )

        context["config_diagnostics"] = diagnostic

        logger.info(
            "config_audit_complete",
            service=service,
            status=status,
            drift_count=len(drifts),
        )

        return context
