"""Audit workflow graph -- orchestrates the full compliance audit pipeline.

Nodes:
  ingest -> classify -> route_to_checkers -> [check_sox, check_gdpr, check_soc2]
  -> aggregate -> score_risks -> generate_remediations -> present_for_approval
  -> finalize

Each node emits SSE events so the frontend can track progress in real
time.  The graph uses a simple sequential/conditional execution engine
that does *not* require LangGraph at runtime.
"""

from __future__ import annotations

import asyncio
import uuid
from datetime import datetime, timezone
from typing import Any, Callable, Awaitable

import structlog

from compliance_audit.agents.classifier import ClassifierAgent
from compliance_audit.agents.gdpr_checker import GDPRCheckerAgent
from compliance_audit.agents.remediation import RemediationAgent
from compliance_audit.agents.risk_scorer import RiskScorerAgent
from compliance_audit.agents.soc2_checker import SOC2CheckerAgent
from compliance_audit.agents.sox_checker import SOXCheckerAgent
from compliance_audit.models import (
    AuditFinding,
    AuditReport,
    AuditSessionState,
    RegulationType,
    SeverityLevel,
)
from compliance_audit.streaming import (
    EVENT_AWAITING_APPROVAL,
    EVENT_CHECK_COMPLETE,
    EVENT_CHECKING_GDPR,
    EVENT_CHECKING_SOC2,
    EVENT_CHECKING_SOX,
    EVENT_CLASSIFYING,
    EVENT_CLASSIFICATION_DONE,
    EVENT_COMPLETED,
    EVENT_ERROR,
    EVENT_INGESTING,
    EVENT_REMEDIATING,
    EVENT_REMEDIATION_DONE,
    EVENT_SCORING,
    EVENT_SCORING_DONE,
    AuditEventStream,
)
from compliance_audit.workflow.routing import (
    determine_checkers,
    route_after_classification,
    route_after_scoring,
    should_await_approval,
)
from compliance_audit.workflow.state import AuditGraphState

logger = structlog.get_logger(__name__)

# Type alias for node functions
NodeFn = Callable[[AuditGraphState], Awaitable[AuditGraphState]]


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


class AuditWorkflowGraph:
    """Simple graph execution engine for the compliance audit pipeline.

    Maintains an ordered list of node functions with conditional routing
    to simulate a LangGraph-style workflow without the dependency.
    """

    def __init__(self, event_stream: AuditEventStream) -> None:
        self.event_stream = event_stream
        self._nodes: dict[str, NodeFn] = {}
        self._build_nodes()

    def _build_nodes(self) -> None:
        """Register all graph node functions."""
        self._nodes = {
            "ingest": self._node_ingest,
            "classify": self._node_classify,
            "route_to_checkers": self._node_route_to_checkers,
            "check_sox": self._node_check_sox,
            "check_gdpr": self._node_check_gdpr,
            "check_soc2": self._node_check_soc2,
            "aggregate": self._node_aggregate,
            "score_risks": self._node_score_risks,
            "generate_remediations": self._node_generate_remediations,
            "present_for_approval": self._node_present_for_approval,
            "finalize": self._node_finalize,
        }

    # ------------------------------------------------------------------
    # Main execution
    # ------------------------------------------------------------------

    async def execute(self, initial_state: AuditGraphState) -> AuditGraphState:
        """Run the full audit workflow graph.

        Executes nodes in sequence with conditional routing at branch
        points.  Any exception in a node transitions to the ``FAILED``
        state and emits an error event.
        """
        state = dict(initial_state)  # shallow copy

        try:
            # Phase 1: Ingest
            state = await self._run_node("ingest", state)

            # Phase 2: Classify
            state = await self._run_node("classify", state)

            # Phase 3: Route and run checkers
            next_step = route_after_classification(state)
            if next_step == "route_to_checkers":
                state = await self._run_node("route_to_checkers", state)

                # Run selected checkers
                checkers = determine_checkers(state)
                checker_tasks = []
                for checker_name in checkers:
                    if checker_name in self._nodes:
                        checker_tasks.append(
                            self._run_node(checker_name, state)
                        )

                # Execute checkers concurrently
                if checker_tasks:
                    results = await asyncio.gather(*checker_tasks, return_exceptions=True)

                    # Merge findings from all checkers
                    for result in results:
                        if isinstance(result, Exception):
                            logger.error("checker_failed", error=str(result))
                            continue
                        # Merge findings
                        for key in ("sox_findings", "gdpr_findings", "soc2_findings"):
                            if key in result:
                                state[key] = result[key]

                # Aggregate
                state = await self._run_node("aggregate", state)

            # Phase 4: Score risks
            state = await self._run_node("score_risks", state)

            # Phase 5: Generate remediations
            next_step = route_after_scoring(state)
            if next_step == "generate_remediations":
                state = await self._run_node("generate_remediations", state)

            # Phase 6: Present for approval (if findings exist)
            if should_await_approval(state):
                state = await self._run_node("present_for_approval", state)
            else:
                # No findings: skip straight to finalize
                state = await self._run_node("finalize", state)

        except Exception as exc:
            logger.error(
                "workflow_error",
                session_id=state.get("session_id", "unknown"),
                error=str(exc),
            )
            state["current_state"] = AuditSessionState.FAILED
            state["error"] = str(exc)
            await self.event_stream.emit(
                session_id=state.get("session_id", ""),
                event_type=EVENT_ERROR,
                data={"error": str(exc)},
                message=f"Workflow failed: {exc}",
            )

        return state  # type: ignore[return-value]

    async def _run_node(
        self, name: str, state: AuditGraphState
    ) -> AuditGraphState:
        """Execute a single named node."""
        node_fn = self._nodes.get(name)
        if node_fn is None:
            raise ValueError(f"Unknown graph node: {name}")

        logger.info("node_start", node=name, session_id=state.get("session_id"))
        result = await node_fn(state)
        logger.info("node_complete", node=name, session_id=state.get("session_id"))
        return result

    # ------------------------------------------------------------------
    # Node implementations
    # ------------------------------------------------------------------

    async def _node_ingest(self, state: AuditGraphState) -> AuditGraphState:
        """Validate and ingest transactions and policies."""
        session_id = state.get("session_id", "")
        state["current_state"] = AuditSessionState.INGESTING

        await self.event_stream.emit(
            session_id=session_id,
            event_type=EVENT_INGESTING,
            data={"transaction_count": len(state.get("transactions", []))},
            message="Ingesting transactions and policies...",
        )

        transactions = state.get("transactions", [])
        policies = state.get("policies", [])

        logger.info(
            "ingest_complete",
            transactions=len(transactions),
            policies=len(policies),
        )

        return state  # type: ignore[return-value]

    async def _node_classify(self, state: AuditGraphState) -> AuditGraphState:
        """Classify transactions by regulation type."""
        session_id = state.get("session_id", "")
        state["current_state"] = AuditSessionState.CLASSIFYING

        await self.event_stream.emit(
            session_id=session_id,
            event_type=EVENT_CLASSIFYING,
            message="Classifying transactions by regulation type...",
        )

        classifier = ClassifierAgent()
        transactions = state.get("transactions", [])
        classified = await classifier.classify(transactions)

        # Convert to serializable dict
        classified_dict: dict[str, list] = {}
        reg_types_found: list[str] = []
        for reg_type, txs in classified.items():
            if txs:
                classified_dict[reg_type.value] = txs
                reg_types_found.append(reg_type.value)

        state["classified"] = classified_dict
        state["regulation_types_found"] = reg_types_found

        await self.event_stream.emit(
            session_id=session_id,
            event_type=EVENT_CLASSIFICATION_DONE,
            data={
                "regulation_types": reg_types_found,
                "counts": {rt: len(txs) for rt, txs in classified_dict.items()},
            },
            message=f"Classification complete. Found: {', '.join(reg_types_found)}",
        )

        return state  # type: ignore[return-value]

    async def _node_route_to_checkers(
        self, state: AuditGraphState
    ) -> AuditGraphState:
        """Prepare state for parallel checker execution."""
        state["current_state"] = AuditSessionState.CHECKING
        state.setdefault("sox_findings", [])
        state.setdefault("gdpr_findings", [])
        state.setdefault("soc2_findings", [])
        return state  # type: ignore[return-value]

    async def _node_check_sox(self, state: AuditGraphState) -> AuditGraphState:
        """Run SOX compliance checks."""
        session_id = state.get("session_id", "")

        await self.event_stream.emit(
            session_id=session_id,
            event_type=EVENT_CHECKING_SOX,
            message="Running SOX compliance checks...",
        )

        classified = state.get("classified", {})
        sox_transactions = classified.get(RegulationType.SOX.value, [])
        policies = state.get("policies", [])

        checker = SOXCheckerAgent()
        findings = await checker.check(sox_transactions, policies)
        state["sox_findings"] = findings

        await self.event_stream.emit(
            session_id=session_id,
            event_type=EVENT_CHECK_COMPLETE,
            data={"checker": "SOX", "findings": len(findings)},
            message=f"SOX check complete: {len(findings)} findings",
        )

        return state  # type: ignore[return-value]

    async def _node_check_gdpr(self, state: AuditGraphState) -> AuditGraphState:
        """Run GDPR compliance checks."""
        session_id = state.get("session_id", "")

        await self.event_stream.emit(
            session_id=session_id,
            event_type=EVENT_CHECKING_GDPR,
            message="Running GDPR compliance checks...",
        )

        classified = state.get("classified", {})
        gdpr_transactions = classified.get(RegulationType.GDPR.value, [])
        policies = state.get("policies", [])

        checker = GDPRCheckerAgent()
        findings = await checker.check(gdpr_transactions, policies)
        state["gdpr_findings"] = findings

        await self.event_stream.emit(
            session_id=session_id,
            event_type=EVENT_CHECK_COMPLETE,
            data={"checker": "GDPR", "findings": len(findings)},
            message=f"GDPR check complete: {len(findings)} findings",
        )

        return state  # type: ignore[return-value]

    async def _node_check_soc2(self, state: AuditGraphState) -> AuditGraphState:
        """Run SOC 2 compliance checks."""
        session_id = state.get("session_id", "")

        await self.event_stream.emit(
            session_id=session_id,
            event_type=EVENT_CHECKING_SOC2,
            message="Running SOC 2 compliance checks...",
        )

        classified = state.get("classified", {})
        soc2_transactions = classified.get(RegulationType.SOC2.value, [])
        policies = state.get("policies", [])

        checker = SOC2CheckerAgent()
        findings = await checker.check(soc2_transactions, policies)
        state["soc2_findings"] = findings

        await self.event_stream.emit(
            session_id=session_id,
            event_type=EVENT_CHECK_COMPLETE,
            data={"checker": "SOC2", "findings": len(findings)},
            message=f"SOC 2 check complete: {len(findings)} findings",
        )

        return state  # type: ignore[return-value]

    async def _node_aggregate(self, state: AuditGraphState) -> AuditGraphState:
        """Merge findings from all checkers into a single list."""
        all_findings: list[AuditFinding] = []
        all_findings.extend(state.get("sox_findings", []))
        all_findings.extend(state.get("gdpr_findings", []))
        all_findings.extend(state.get("soc2_findings", []))
        state["all_findings"] = all_findings

        logger.info(
            "findings_aggregated",
            total=len(all_findings),
            sox=len(state.get("sox_findings", [])),
            gdpr=len(state.get("gdpr_findings", [])),
            soc2=len(state.get("soc2_findings", [])),
        )

        return state  # type: ignore[return-value]

    async def _node_score_risks(self, state: AuditGraphState) -> AuditGraphState:
        """Score risk for all findings."""
        session_id = state.get("session_id", "")
        state["current_state"] = AuditSessionState.SCORING

        await self.event_stream.emit(
            session_id=session_id,
            event_type=EVENT_SCORING,
            data={"findings_count": len(state.get("all_findings", []))},
            message="Scoring risk for findings...",
        )

        scorer = RiskScorerAgent()
        findings = state.get("all_findings", [])
        risk_scores = await scorer.score(findings)
        state["risk_scores"] = risk_scores

        await self.event_stream.emit(
            session_id=session_id,
            event_type=EVENT_SCORING_DONE,
            data={"scores_count": len(risk_scores)},
            message=f"Risk scoring complete: {len(risk_scores)} scores",
        )

        return state  # type: ignore[return-value]

    async def _node_generate_remediations(
        self, state: AuditGraphState
    ) -> AuditGraphState:
        """Generate remediation recommendations."""
        session_id = state.get("session_id", "")
        state["current_state"] = AuditSessionState.REMEDIATING

        await self.event_stream.emit(
            session_id=session_id,
            event_type=EVENT_REMEDIATING,
            message="Generating remediation recommendations...",
        )

        agent = RemediationAgent()
        findings = state.get("all_findings", [])
        risk_scores = state.get("risk_scores", [])
        remediations = await agent.recommend(findings, risk_scores)
        state["remediations"] = remediations

        await self.event_stream.emit(
            session_id=session_id,
            event_type=EVENT_REMEDIATION_DONE,
            data={"remediations_count": len(remediations)},
            message=f"Remediation generation complete: {len(remediations)} actions",
        )

        return state  # type: ignore[return-value]

    async def _node_present_for_approval(
        self, state: AuditGraphState
    ) -> AuditGraphState:
        """Present findings for human approval.

        This node transitions the session to AWAITING_APPROVAL and emits
        an event.  The actual approval/rejection happens via the API
        endpoint and is handled by the session manager.
        """
        session_id = state.get("session_id", "")
        state["current_state"] = AuditSessionState.AWAITING_APPROVAL

        findings = state.get("all_findings", [])
        risk_scores = state.get("risk_scores", [])

        # Compute overall risk level
        overall_risk = self._compute_overall_risk(findings, risk_scores)

        await self.event_stream.emit(
            session_id=session_id,
            event_type=EVENT_AWAITING_APPROVAL,
            data={
                "findings_count": len(findings),
                "overall_risk_level": overall_risk.value,
                "summary": self._build_summary(findings, risk_scores),
            },
            message=(
                f"Audit complete with {len(findings)} findings. "
                "Awaiting human approval."
            ),
        )

        return state  # type: ignore[return-value]

    async def _node_finalize(self, state: AuditGraphState) -> AuditGraphState:
        """Build the final audit report."""
        session_id = state.get("session_id", "")
        findings = state.get("all_findings", [])
        risk_scores = state.get("risk_scores", [])
        remediations = state.get("remediations", [])

        overall_risk = self._compute_overall_risk(findings, risk_scores)
        summary = self._build_summary(findings, risk_scores)

        report = AuditReport(
            id=str(uuid.uuid4()),
            session_id=session_id,
            findings=findings,
            risk_scores=risk_scores,
            remediations=remediations,
            overall_risk_level=overall_risk,
            summary=summary,
            created_at=datetime.now(tz=timezone.utc),
        )

        state["report"] = report
        state["current_state"] = AuditSessionState.COMPLETED

        await self.event_stream.emit(
            session_id=session_id,
            event_type=EVENT_COMPLETED,
            data={
                "report_id": report.id,
                "findings_count": len(findings),
                "overall_risk_level": overall_risk.value,
            },
            message="Audit completed successfully.",
        )

        return state  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_overall_risk(
        findings: list[AuditFinding],
        risk_scores: list,
    ) -> SeverityLevel:
        """Determine the overall risk level from findings."""
        if not findings:
            return SeverityLevel.INFO

        severity_order = [
            SeverityLevel.CRITICAL,
            SeverityLevel.HIGH,
            SeverityLevel.MEDIUM,
            SeverityLevel.LOW,
            SeverityLevel.INFO,
        ]

        for severity in severity_order:
            if any(f.severity == severity for f in findings):
                return severity

        return SeverityLevel.INFO

    @staticmethod
    def _build_summary(
        findings: list[AuditFinding],
        risk_scores: list,
    ) -> str:
        """Build a human-readable audit summary."""
        if not findings:
            return "No compliance findings detected. All transactions passed audit checks."

        # Count by regulation type
        by_reg: dict[str, int] = {}
        by_severity: dict[str, int] = {}
        for f in findings:
            by_reg[f.regulation_type.value] = by_reg.get(f.regulation_type.value, 0) + 1
            by_severity[f.severity.value] = by_severity.get(f.severity.value, 0) + 1

        parts = [
            f"Audit identified {len(findings)} compliance finding(s).",
        ]

        reg_parts = [f"{reg}: {count}" for reg, count in sorted(by_reg.items())]
        parts.append(f"By regulation: {', '.join(reg_parts)}.")

        sev_parts = [f"{sev}: {count}" for sev, count in sorted(by_severity.items())]
        parts.append(f"By severity: {', '.join(sev_parts)}.")

        if risk_scores:
            avg_score = sum(s.score for s in risk_scores) / len(risk_scores)
            parts.append(f"Average risk score: {avg_score:.2f}.")

        return " ".join(parts)


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------


def compile_audit_graph(event_stream: AuditEventStream) -> AuditWorkflowGraph:
    """Create a configured audit workflow graph.

    This mirrors the ``compile_shopping_graph`` pattern from the
    reference project but uses a lightweight execution engine instead
    of LangGraph.
    """
    return AuditWorkflowGraph(event_stream=event_stream)
