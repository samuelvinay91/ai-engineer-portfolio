"""Full Incident Response Pipeline.

Composes the three ADK workflow patterns into a complete incident response
pipeline that processes an alert from receipt through resolution:

1. **SequentialAgent** -- enrich -> triage -> assign responder
2. **ParallelAgent** -- log_analyzer + metrics_checker + config_auditor
3. **LoopAgent** -- remediate -> verify (escalate until fixed, max 3 iterations)

The pipeline emits SSE events at each stage for real-time UI updates.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any

import structlog

from incident_response.config import Settings
from incident_response.models import (
    Alert,
    DiagnosticResult,
    EscalationLevel,
    IncidentReport,
    IncidentState,
    RemediationAction,
    SeverityLevel,
)
from incident_response.streaming import (
    EVENT_DIAGNOSING_CONFIG,
    EVENT_DIAGNOSING_LOGS,
    EVENT_DIAGNOSING_METRICS,
    EVENT_DIAGNOSTICS_COMPLETE,
    EVENT_ENRICHED,
    EVENT_ENRICHING,
    EVENT_ERROR,
    EVENT_ESCALATING,
    EVENT_HUMAN_TAKEOVER,
    EVENT_RECEIVED,
    EVENT_REMEDIATION_ATTEMPTED,
    EVENT_REMEDIATING,
    EVENT_RESOLVED,
    EVENT_TRIAGED,
    EVENT_TRIAGING,
    EVENT_VERIFICATION_RESULT,
    EVENT_VERIFYING,
    IncidentEventStream,
)
from incident_response.workflow.escalation_loop import build_escalation_loop
from incident_response.workflow.parallel_diagnostics import build_parallel_diagnostics
from incident_response.workflow.sequential_triage import build_sequential_triage

logger = structlog.get_logger(__name__)


async def run_incident_pipeline(
    alert: Alert,
    settings: Settings,
    event_stream: IncidentEventStream,
    session_id: str,
) -> dict[str, Any]:
    """Execute the full incident response pipeline.

    Orchestrates three ADK workflow agent types in sequence:

    1. **SequentialAgent** (Triage Pipeline):
       - ContextEnricherAgent: enrich alert with operational context
       - TriageAgent: classify severity P1-P4
       - ResponderAssignerAgent: assign on-call and escalation path

    2. **ParallelAgent** (Diagnostics):
       - LogAnalyzerAgent: search logs for error patterns
       - MetricsCheckerAgent: check metrics for anomalies
       - ConfigAuditorAgent: detect configuration drift

    3. **LoopAgent** (Escalation Loop):
       - RemediationAgent: execute runbook-based fix
       - VerificationAgent: check recovery, escalate if needed

    Args:
        alert: The incoming alert to process.
        settings: Application settings.
        event_stream: SSE event stream for real-time updates.
        session_id: Unique session identifier.

    Returns:
        Final workflow context dict with all accumulated data.
    """
    logger.info("pipeline_start", alert_id=alert.id, session_id=session_id)

    # Initialize shared context
    context: dict[str, Any] = {
        "alert": alert,
        "session_id": session_id,
        "timeline": [],
    }

    timeline: list[dict[str, Any]] = context["timeline"]

    try:
        # ------------------------------------------------------------------
        # Phase 0: Receive alert
        # ------------------------------------------------------------------
        await event_stream.emit(
            session_id,
            EVENT_RECEIVED,
            data={"alert_id": alert.id, "service": alert.service, "title": alert.title},
            message=f"Alert received: {alert.title}",
        )
        _add_timeline(timeline, "alert_received", f"Alert {alert.id} received from {alert.source}")

        # ------------------------------------------------------------------
        # Phase 1: Sequential Triage (enrich -> triage -> assign)
        # ------------------------------------------------------------------
        await event_stream.emit(
            session_id,
            EVENT_ENRICHING,
            data={"service": alert.service},
            message=f"Enriching context for {alert.service}...",
        )
        _add_timeline(timeline, "enrichment_started", "Context enrichment started")

        triage_pipeline = build_sequential_triage()
        context = await triage_pipeline.run(context)

        await event_stream.emit(
            session_id,
            EVENT_ENRICHED,
            data={
                "owner_team": context.get("owner_team", ""),
                "recent_deploys": len(
                    context.get("incident_context", {}).recent_deploys
                    if hasattr(context.get("incident_context", {}), "recent_deploys")
                    else []
                ),
            },
            message=f"Context enriched. Owner team: {context.get('owner_team', 'unknown')}",
        )
        _add_timeline(timeline, "enrichment_complete", "Context enrichment complete")

        await event_stream.emit(
            session_id,
            EVENT_TRIAGING,
            data={"service": alert.service},
            message="Classifying incident severity...",
        )

        severity: SeverityLevel = context.get("severity", SeverityLevel.P4)
        reasoning = context.get("triage_reasoning", "")

        await event_stream.emit(
            session_id,
            EVENT_TRIAGED,
            data={
                "severity": severity.value,
                "reasoning": reasoning,
                "assigned_responder": context.get("assigned_responder", {}),
                "escalation_level": context.get("escalation_level", EscalationLevel.L1_AUTO).value,
            },
            message=f"Severity classified as {severity.value}. {reasoning[:200]}",
        )
        _add_timeline(
            timeline,
            "triage_complete",
            f"Severity: {severity.value}, Responder: {context.get('assigned_responder', {}).get('name', 'N/A')}",
        )

        # ------------------------------------------------------------------
        # Phase 2: Parallel Diagnostics (logs + metrics + config)
        # ------------------------------------------------------------------
        await event_stream.emit(
            session_id,
            EVENT_DIAGNOSING_LOGS,
            data={"service": alert.service},
            message="Analyzing application logs...",
        )
        await event_stream.emit(
            session_id,
            EVENT_DIAGNOSING_METRICS,
            data={"service": alert.service},
            message="Checking infrastructure metrics...",
        )
        await event_stream.emit(
            session_id,
            EVENT_DIAGNOSING_CONFIG,
            data={"service": alert.service},
            message="Auditing service configuration...",
        )
        _add_timeline(timeline, "diagnostics_started", "Parallel diagnostics started")

        diagnostics_pipeline = build_parallel_diagnostics()
        context = await diagnostics_pipeline.run(context)

        # Collect diagnostic summaries
        diag_summary: dict[str, Any] = {}
        for key, label in [
            ("log_diagnostics", "logs"),
            ("metrics_diagnostics", "metrics"),
            ("config_diagnostics", "config"),
        ]:
            diag: DiagnosticResult | None = context.get(key)
            if diag and isinstance(diag, DiagnosticResult):
                diag_summary[label] = {
                    "findings": len(diag.findings),
                    "anomalies": len(diag.anomalies),
                    "severity_indicators": diag.severity_indicators,
                }

        await event_stream.emit(
            session_id,
            EVENT_DIAGNOSTICS_COMPLETE,
            data={"summary": diag_summary},
            message=f"Diagnostics complete. Found issues in: {', '.join(diag_summary.keys())}",
        )
        _add_timeline(
            timeline,
            "diagnostics_complete",
            f"Diagnostics: {sum(d.get('anomalies', 0) for d in diag_summary.values())} total anomalies",
        )

        # ------------------------------------------------------------------
        # Phase 3: Escalation Loop (remediate -> verify -> escalate)
        # ------------------------------------------------------------------
        await event_stream.emit(
            session_id,
            EVENT_REMEDIATING,
            data={"service": alert.service, "max_iterations": settings.max_escalation_levels},
            message="Starting automated remediation loop...",
        )
        _add_timeline(timeline, "remediation_started", "Remediation loop started")

        escalation_loop = build_escalation_loop(
            max_iterations=settings.max_escalation_levels,
        )
        context = await escalation_loop.run(context)

        # Emit events for each remediation action taken
        remediation_actions: list[RemediationAction] = context.get("remediation_actions", [])
        for action in remediation_actions:
            await event_stream.emit(
                session_id,
                EVENT_REMEDIATION_ATTEMPTED,
                data={
                    "action_type": action.action_type,
                    "success": action.success,
                    "output": action.output[:500],
                },
                message=f"Remediation: {action.action_type} - {'success' if action.success else 'failed'}",
            )

        # Emit verification result
        verification = context.get("verification_result", {})
        await event_stream.emit(
            session_id,
            EVENT_VERIFYING,
            data={"service": alert.service},
            message="Verifying service health...",
        )
        await event_stream.emit(
            session_id,
            EVENT_VERIFICATION_RESULT,
            data=verification,
            message=f"Verification: {verification.get('verdict', 'UNKNOWN')}",
        )

        # ------------------------------------------------------------------
        # Phase 4: Resolution or Escalation
        # ------------------------------------------------------------------
        is_resolved = context.get("loop_complete", False)
        loop_exhausted = context.get("loop_exhausted", False)

        if is_resolved:
            # Build incident report
            report = _build_report(context, session_id)
            context["report"] = report

            await event_stream.emit(
                session_id,
                EVENT_RESOLVED,
                data={
                    "severity": severity.value,
                    "resolution_summary": report.resolution_summary,
                    "total_actions": len(remediation_actions),
                    "iterations": context.get("loop_iteration", 0),
                },
                message=f"Incident resolved! {report.resolution_summary}",
            )
            _add_timeline(timeline, "resolved", report.resolution_summary)
            logger.info("pipeline_resolved", session_id=session_id, severity=severity.value)

        elif loop_exhausted:
            # Escalate to human
            escalation_level = context.get("escalation_level", EscalationLevel.L4_MANAGEMENT)

            await event_stream.emit(
                session_id,
                EVENT_ESCALATING,
                data={
                    "escalation_level": escalation_level.value,
                    "iterations_exhausted": settings.max_escalation_levels,
                },
                message=(
                    f"Automated remediation exhausted ({settings.max_escalation_levels} attempts). "
                    f"Escalating to {escalation_level.value}."
                ),
            )
            _add_timeline(
                timeline,
                "escalation",
                f"Escalated to {escalation_level.value} after {settings.max_escalation_levels} attempts",
            )

            # Build partial report
            report = _build_report(context, session_id, resolved=False)
            context["report"] = report

            await event_stream.emit(
                session_id,
                EVENT_HUMAN_TAKEOVER,
                data={
                    "severity": severity.value,
                    "escalation_level": escalation_level.value,
                    "assigned_responder": context.get("assigned_responder", {}),
                    "attempts": settings.max_escalation_levels,
                },
                message=(
                    f"Human takeover required. Severity: {severity.value}. "
                    f"Escalation level: {escalation_level.value}."
                ),
            )
            _add_timeline(timeline, "human_takeover", "Incident handed to human operator")
            logger.warning(
                "pipeline_human_takeover",
                session_id=session_id,
                severity=severity.value,
                escalation_level=escalation_level.value,
            )

    except Exception as exc:
        logger.error("pipeline_error", session_id=session_id, error=str(exc))
        _add_timeline(timeline, "error", str(exc))

        await event_stream.emit(
            session_id,
            EVENT_ERROR,
            data={"error": str(exc)},
            message=f"Pipeline error: {exc}",
        )
        context["error"] = str(exc)

    logger.info("pipeline_complete", session_id=session_id)
    return context


def _build_report(
    context: dict[str, Any],
    session_id: str,
    resolved: bool = True,
) -> IncidentReport:
    """Build the final incident report from accumulated workflow state."""
    alert = context["alert"]
    severity = context.get("severity", SeverityLevel.P4)
    remediation_actions: list[RemediationAction] = context.get("remediation_actions", [])
    diagnostics: list[DiagnosticResult] = []

    for key in ["log_diagnostics", "metrics_diagnostics", "config_diagnostics"]:
        diag = context.get(key)
        if diag and isinstance(diag, DiagnosticResult):
            diagnostics.append(diag)

    # Build escalation history
    escalation_history: list[dict[str, Any]] = []
    for i, action in enumerate(remediation_actions, 1):
        escalation_history.append({
            "iteration": i,
            "action": action.action_type,
            "success": action.success,
            "output": action.output[:200],
        })

    # Resolution summary
    if resolved:
        successful_actions = [a for a in remediation_actions if a.success]
        if successful_actions:
            last_success = successful_actions[-1]
            resolution_summary = (
                f"Incident resolved via {last_success.action_type} on attempt "
                f"{context.get('loop_iteration', '?')}. "
                f"Total actions taken: {len(remediation_actions)}."
            )
        else:
            resolution_summary = "Incident resolved after automated diagnostics and remediation."
    else:
        resolution_summary = (
            f"Automated remediation exhausted after {len(remediation_actions)} attempts. "
            f"Escalated to {context.get('escalation_level', EscalationLevel.L4_MANAGEMENT).value}."
        )

    return IncidentReport(
        id=f"RPT-{uuid.uuid4().hex[:8].upper()}",
        session_id=session_id,
        alert=alert,
        severity=severity,
        diagnostics=diagnostics,
        remediations=remediation_actions,
        escalation_history=escalation_history,
        resolution_summary=resolution_summary,
        timeline=context.get("timeline", []),
    )


def _add_timeline(
    timeline: list[dict[str, Any]],
    event: str,
    description: str,
) -> None:
    """Append a timestamped entry to the incident timeline."""
    timeline.append({
        "event": event,
        "description": description,
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    })
