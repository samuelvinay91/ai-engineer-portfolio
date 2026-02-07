"""Telemetry middleware -- collects timing and usage metrics per agent.

Provides an OpenTelemetry-compatible interface that works without the
SDK installed.  Tracks execution timing, token usage estimates, and
per-agent span information.
"""

from __future__ import annotations

import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class TelemetrySpan(BaseModel):
    """A single telemetry span recording an agent execution."""

    span_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    trace_id: str = ""
    agent_name: str
    operation: str = "invoke"
    start_time: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    end_time: datetime | None = None
    duration_ms: float = 0.0
    status: str = "ok"
    attributes: dict[str, Any] = Field(default_factory=dict)
    token_usage: dict[str, int] = Field(default_factory=dict)


class TelemetryMiddleware:
    """Collects timing metrics, token usage, and span data per agent.

    Provides an interface compatible with OpenTelemetry concepts
    (traces, spans) but runs entirely in-memory without requiring
    the OpenTelemetry SDK.

    Usage::

        telemetry = TelemetryMiddleware()

        async with telemetry.span("SOXCheckerAgent", "check") as s:
            result = await checker.check(transactions, policies)
            s.attributes["findings"] = len(result)

        metrics = telemetry.get_metrics()
    """

    def __init__(self, service_name: str = "compliance-audit-agents") -> None:
        self.service_name = service_name
        self._spans: list[TelemetrySpan] = []
        self._trace_id = str(uuid.uuid4())
        self._agent_metrics: dict[str, dict[str, Any]] = {}

    @asynccontextmanager
    async def span(
        self,
        agent_name: str,
        operation: str = "invoke",
    ) -> AsyncIterator[TelemetrySpan]:
        """Create a telemetry span for an agent operation.

        The span records timing information automatically and is
        stored for later retrieval via :meth:`get_spans`.
        """
        telemetry_span = TelemetrySpan(
            trace_id=self._trace_id,
            agent_name=agent_name,
            operation=operation,
            start_time=datetime.now(tz=timezone.utc),
        )

        start = time.perf_counter()

        try:
            yield telemetry_span
            telemetry_span.status = "ok"
        except Exception as exc:
            telemetry_span.status = "error"
            telemetry_span.attributes["error"] = str(exc)
            raise
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            telemetry_span.end_time = datetime.now(tz=timezone.utc)
            telemetry_span.duration_ms = round(elapsed_ms, 2)

            self._spans.append(telemetry_span)
            self._update_agent_metrics(telemetry_span)

            logger.debug(
                "telemetry_span_complete",
                agent=agent_name,
                operation=operation,
                duration_ms=telemetry_span.duration_ms,
                status=telemetry_span.status,
            )

    def record_token_usage(
        self,
        agent_name: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
    ) -> None:
        """Record token usage for an agent invocation."""
        metrics = self._agent_metrics.setdefault(agent_name, {
            "invocations": 0,
            "total_duration_ms": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "errors": 0,
        })
        metrics["total_prompt_tokens"] += prompt_tokens
        metrics["total_completion_tokens"] += completion_tokens

    def get_spans(
        self,
        agent_name: str | None = None,
    ) -> list[TelemetrySpan]:
        """Return recorded spans, optionally filtered by agent name."""
        if agent_name:
            return [s for s in self._spans if s.agent_name == agent_name]
        return list(self._spans)

    def get_metrics(self) -> dict[str, dict[str, Any]]:
        """Return aggregated metrics per agent.

        Returns a dict mapping agent names to their metrics:
        invocation count, total duration, token usage, error count.
        """
        return dict(self._agent_metrics)

    def get_summary(self) -> dict[str, Any]:
        """Return a high-level telemetry summary."""
        total_spans = len(self._spans)
        total_duration = sum(s.duration_ms for s in self._spans)
        error_count = sum(1 for s in self._spans if s.status == "error")

        return {
            "service_name": self.service_name,
            "trace_id": self._trace_id,
            "total_spans": total_spans,
            "total_duration_ms": round(total_duration, 2),
            "error_count": error_count,
            "agents": list(self._agent_metrics.keys()),
            "agent_metrics": self._agent_metrics,
        }

    def reset(self) -> None:
        """Clear all collected telemetry data."""
        self._spans.clear()
        self._agent_metrics.clear()
        self._trace_id = str(uuid.uuid4())

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _update_agent_metrics(self, span: TelemetrySpan) -> None:
        """Update aggregated metrics for the agent."""
        metrics = self._agent_metrics.setdefault(span.agent_name, {
            "invocations": 0,
            "total_duration_ms": 0.0,
            "total_prompt_tokens": 0,
            "total_completion_tokens": 0,
            "errors": 0,
        })
        metrics["invocations"] += 1
        metrics["total_duration_ms"] = round(
            metrics["total_duration_ms"] + span.duration_ms, 2
        )
        if span.status == "error":
            metrics["errors"] += 1

        # Merge token usage from span
        for key, value in span.token_usage.items():
            metric_key = f"total_{key}"
            if metric_key in metrics:
                metrics[metric_key] += value
