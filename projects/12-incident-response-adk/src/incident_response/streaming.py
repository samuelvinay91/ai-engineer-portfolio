"""SSE streaming manager for real-time incident response workflow updates.

Provides an event bus that workflow agents write to and that API endpoints
consume via ``async for`` iteration. Follows the ShoppingEventStream pattern.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any, AsyncIterator

import structlog

from incident_response.models import IncidentEvent

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Canonical event type constants
# ---------------------------------------------------------------------------

EVENT_RECEIVED = "received"
EVENT_ENRICHING = "enriching"
EVENT_ENRICHED = "enriched"
EVENT_TRIAGING = "triaging"
EVENT_TRIAGED = "triaged"
EVENT_DIAGNOSING_LOGS = "diagnosing_logs"
EVENT_DIAGNOSING_METRICS = "diagnosing_metrics"
EVENT_DIAGNOSING_CONFIG = "diagnosing_config"
EVENT_DIAGNOSTICS_COMPLETE = "diagnostics_complete"
EVENT_REMEDIATING = "remediating"
EVENT_REMEDIATION_ATTEMPTED = "remediation_attempted"
EVENT_VERIFYING = "verifying"
EVENT_VERIFICATION_RESULT = "verification_result"
EVENT_ESCALATING = "escalating"
EVENT_RESOLVED = "resolved"
EVENT_HUMAN_TAKEOVER = "human_takeover"
EVENT_ERROR = "error"

# Terminal event types that end the SSE stream
_TERMINAL_EVENTS = {EVENT_RESOLVED, EVENT_HUMAN_TAKEOVER, EVENT_ERROR}


class IncidentEventStream:
    """In-memory pub/sub for incident session SSE events.

    Each incident session gets its own ``asyncio.Queue`` so that multiple
    SSE subscribers can consume events independently.
    """

    def __init__(self, max_queue_size: int = 256) -> None:
        self._queues: dict[str, list[asyncio.Queue[IncidentEvent | None]]] = {}
        self._max_queue_size = max_queue_size
        self._history: dict[str, list[IncidentEvent]] = {}

    # ------------------------------------------------------------------
    # Publishing
    # ------------------------------------------------------------------

    async def emit(
        self,
        session_id: str,
        event_type: str,
        data: dict[str, Any] | None = None,
        message: str = "",
    ) -> IncidentEvent:
        """Push an event to all subscribers of *session_id*.

        Returns the constructed :class:`IncidentEvent` for convenience.
        """
        event = IncidentEvent(
            event_type=event_type,
            session_id=session_id,
            data=data or {},
            message=message,
            timestamp=datetime.now(tz=timezone.utc),
        )

        # Persist in history
        self._history.setdefault(session_id, []).append(event)

        # Fan-out to all live subscriber queues
        queues = self._queues.get(session_id, [])
        for queue in queues:
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning(
                    "event_queue_full",
                    session_id=session_id,
                    event_type=event_type,
                )

        logger.debug(
            "event_emitted",
            session_id=session_id,
            event_type=event_type,
            subscribers=len(queues),
        )
        return event

    # ------------------------------------------------------------------
    # Subscribing
    # ------------------------------------------------------------------

    async def subscribe(self, session_id: str) -> AsyncIterator[IncidentEvent]:
        """Yield events for *session_id* as they arrive.

        The iterator terminates when the session emits a terminal event
        (resolved, human_takeover, error), or when ``close(session_id)``
        is called (which pushes ``None`` as a sentinel).
        """
        queue: asyncio.Queue[IncidentEvent | None] = asyncio.Queue(
            maxsize=self._max_queue_size
        )
        self._queues.setdefault(session_id, []).append(queue)

        # Replay any historical events first so late joiners catch up
        for past_event in self._history.get(session_id, []):
            yield past_event

        try:
            while True:
                event = await queue.get()
                if event is None:
                    # Sentinel: stream closed
                    break
                yield event
                if event.event_type in _TERMINAL_EVENTS:
                    break
        finally:
            # Clean up this subscriber
            session_queues = self._queues.get(session_id, [])
            if queue in session_queues:
                session_queues.remove(queue)

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    def close(self, session_id: str) -> None:
        """Signal all subscribers of *session_id* to stop iterating."""
        for queue in self._queues.get(session_id, []):
            try:
                queue.put_nowait(None)
            except asyncio.QueueFull:
                pass
        self._queues.pop(session_id, None)

    def get_history(self, session_id: str) -> list[IncidentEvent]:
        """Return all events emitted for a given session."""
        return list(self._history.get(session_id, []))

    def clear(self, session_id: str) -> None:
        """Remove all state associated with a session."""
        self.close(session_id)
        self._history.pop(session_id, None)
