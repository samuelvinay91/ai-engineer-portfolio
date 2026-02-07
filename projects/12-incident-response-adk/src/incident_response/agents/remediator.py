"""Remediation Agent.

Executes runbook-driven remediation steps based on diagnostic findings.
Simulates actions like service restarts, scale-ups, deploy rollbacks,
cache clears, and certificate rotations with realistic delays and outcomes.
"""

from __future__ import annotations

import asyncio
import random
from typing import Any

import structlog

from incident_response.agents.base import LlmAgent
from incident_response.models import (
    DiagnosticResult,
    EscalationLevel,
    RemediationAction,
    SeverityLevel,
)
from incident_response.tools.runbooks import RUNBOOKS, select_runbook_for_symptom

logger = structlog.get_logger(__name__)

# Mapping from diagnostic severity indicators to remediation symptoms
INDICATOR_TO_SYMPTOM: dict[str, str] = {
    "cpu_critical": "high_cpu",
    "memory_critical": "memory_leak",
    "extreme_error_rate": "regression_after_deploy",
    "latency_degradation": "high_latency",
    "connection_pressure": "connection_pool_exhaustion",
    "stack_trace_present": "regression_after_deploy",
    "error_spike": "regression_after_deploy",
    "timeout_pattern": "connection_timeout",
    "fatal_log": "memory_leak",
    "resource_limit_drift": "memory_leak",
    "image_version_mismatch": "regression_after_deploy",
    "env_var_drift": "config_drift",
    "critical_config_drift": "config_drift",
}


class RemediationAgent(LlmAgent):
    """Executes automated remediation based on diagnostic findings.

    Selects the appropriate runbook based on observed symptoms, executes
    the remediation steps with simulated delays, and reports success or
    failure outcomes.
    """

    def __init__(self) -> None:
        super().__init__(
            name="remediator",
            instruction=(
                "You are a remediation specialist. Based on the diagnostic "
                "findings, select and execute the most appropriate runbook. "
                "Actions include: restart_service, scale_up, rollback_deploy, "
                "clear_cache, rotate_certs. Execute the action and report "
                "the outcome."
            ),
            description="Executes runbook-driven remediation actions",
        )

    async def _heuristic_run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Select and execute remediation based on diagnostics.

        Expects:
            context["alert"]: Alert
            context["severity"]: SeverityLevel
            context["log_diagnostics"]: DiagnosticResult (optional)
            context["metrics_diagnostics"]: DiagnosticResult (optional)
            context["config_diagnostics"]: DiagnosticResult (optional)
            context["loop_iteration"]: int (which remediation attempt)

        Sets:
            context["remediation_actions"]: list of RemediationAction
            context["last_remediation"]: RemediationAction (most recent)
        """
        alert = context["alert"]
        service = alert.service
        iteration = context.get("loop_iteration", 1)

        logger.info(
            "selecting_remediation",
            service=service,
            iteration=iteration,
        )

        # Collect all severity indicators from diagnostics
        all_indicators: list[str] = []
        for diag_key in ["log_diagnostics", "metrics_diagnostics", "config_diagnostics"]:
            diag: DiagnosticResult | None = context.get(diag_key)
            if diag and isinstance(diag, DiagnosticResult):
                all_indicators.extend(diag.severity_indicators)

        # Select remediation action based on iteration and indicators
        action = _select_action(service, all_indicators, iteration, context)

        # Execute the remediation
        executed_action = await _execute_remediation(action, service, iteration)

        # Store results
        existing_actions: list[RemediationAction] = context.get("remediation_actions", [])
        existing_actions.append(executed_action)
        context["remediation_actions"] = existing_actions
        context["last_remediation"] = executed_action

        logger.info(
            "remediation_executed",
            service=service,
            action=executed_action.action_type,
            success=executed_action.success,
            iteration=iteration,
        )

        return context


def _select_action(
    service: str,
    indicators: list[str],
    iteration: int,
    context: dict[str, Any],
) -> str:
    """Select the best remediation action based on symptoms and iteration.

    On each successive iteration, escalate to more aggressive actions:
    - Iteration 1: restart_service or clear_cache (low risk)
    - Iteration 2: scale_up or drain_connections (medium)
    - Iteration 3: rollback_deploy (higher risk)
    """
    # Map indicators to symptoms
    symptoms: set[str] = set()
    for indicator in indicators:
        symptom = INDICATOR_TO_SYMPTOM.get(indicator)
        if symptom:
            symptoms.add(symptom)

    # Escalation-based action selection
    action_priority_by_iteration: dict[int, list[str]] = {
        1: ["restart_service", "clear_cache", "drain_connections"],
        2: ["scale_up", "drain_connections", "restart_service"],
        3: ["rollback_deploy", "scale_up", "rotate_certs"],
    }

    candidates = action_priority_by_iteration.get(iteration, ["restart_service"])

    # Try to match a symptom to a runbook first
    for symptom in symptoms:
        matched_action = select_runbook_for_symptom(symptom)
        if matched_action and matched_action in candidates:
            return matched_action

    # Check for specific patterns
    if "certificate_expiry" in symptoms or "tls_handshake_failure" in symptoms:
        return "rotate_certs"

    if "config_drift" in symptoms and iteration >= 2:
        return "rollback_deploy"

    # Fall back to the first candidate for this iteration
    return candidates[0]


async def _execute_remediation(
    action_type: str,
    service: str,
    iteration: int,
) -> RemediationAction:
    """Simulate executing a remediation action with realistic timing.

    Success probability increases slightly with iteration to simulate
    that escalated actions are more likely to resolve the issue.
    """
    runbook = RUNBOOKS.get(action_type, {})
    runbook_name = runbook.get("name", action_type)
    runbook_id = runbook.get("id", "RB-000")

    logger.info(
        "executing_runbook",
        action=action_type,
        runbook=runbook_name,
        service=service,
        iteration=iteration,
    )

    # Simulate execution delay (shorter for low-risk actions)
    base_delay = {
        "restart_service": 2.0,
        "clear_cache": 1.0,
        "scale_up": 3.0,
        "rollback_deploy": 4.0,
        "rotate_certs": 5.0,
        "drain_connections": 2.5,
    }
    delay = base_delay.get(action_type, 2.0)
    # Scale down for demo purposes (real would be much longer)
    await asyncio.sleep(delay * 0.5)

    # Determine success (probability increases with iteration)
    # Iteration 1: 40%, 2: 60%, 3: 85%
    success_prob = min(0.4 + (iteration - 1) * 0.2, 0.85)
    success = random.random() < success_prob

    # Build output message
    if success:
        output = (
            f"Successfully executed {runbook_name} on {service}. "
            f"Steps completed: {len(runbook.get('steps', []))} of "
            f"{len(runbook.get('steps', []))}. "
            f"Service is responding to health checks."
        )
    else:
        failure_reasons = [
            f"Step 3 failed: health check returned non-200 status",
            f"Timeout waiting for pods to reach Ready state",
            f"Error rate did not decrease within expected timeframe",
            f"Partial success: some pods healthy but not all replicas",
        ]
        output = (
            f"Remediation {runbook_name} on {service} did not fully succeed. "
            f"Reason: {random.choice(failure_reasons)}"
        )

    # Build parameters used
    parameters = {
        "service": service,
        "namespace": "production",
    }
    if action_type == "scale_up":
        parameters["scale_factor"] = 1.5
        parameters["target_replicas"] = 6
    elif action_type == "rollback_deploy":
        parameters["target_version"] = "previous"

    return RemediationAction(
        action_type=action_type,
        description=f"Execute {runbook_name} for {service} (attempt {iteration})",
        runbook_id=runbook_id,
        parameters=parameters,
        executed=True,
        success=success,
        output=output,
    )
