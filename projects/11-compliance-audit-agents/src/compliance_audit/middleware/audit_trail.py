"""Audit trail middleware -- logs every agent invocation.

Records timestamp, agent name, input/output hash, and duration for
each agent call.  Maintains an in-memory log that can be queried for
compliance reporting.
"""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from typing import Any

import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


class AuditTrailEntry(BaseModel):
    """A single audit trail log entry."""

    timestamp: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    agent_name: str
    input_hash: str
    output_hash: str
    duration_ms: float
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditTrailMiddleware:
    """Logs every agent invocation with timestamp, hashes, and timing.

    Usage::

        trail = AuditTrailMiddleware()
        trail.record_invocation(
            agent_name="SOXCheckerAgent",
            input_data={"transactions": [...]},
            output_data={"findings": [...]},
            duration_ms=123.4,
        )
        entries = trail.get_entries()
    """

    def __init__(self) -> None:
        self._entries: list[AuditTrailEntry] = []

    def record_invocation(
        self,
        agent_name: str,
        input_data: Any,
        output_data: Any,
        duration_ms: float,
        metadata: dict[str, Any] | None = None,
    ) -> AuditTrailEntry:
        """Record an agent invocation in the audit trail.

        Parameters
        ----------
        agent_name:
            Name of the agent that was invoked.
        input_data:
            Input data (will be hashed, not stored directly).
        output_data:
            Output data (will be hashed, not stored directly).
        duration_ms:
            Execution duration in milliseconds.
        metadata:
            Optional additional metadata to attach.
        """
        input_hash = self._hash_data(input_data)
        output_hash = self._hash_data(output_data)

        entry = AuditTrailEntry(
            agent_name=agent_name,
            input_hash=input_hash,
            output_hash=output_hash,
            duration_ms=round(duration_ms, 2),
            metadata=metadata or {},
        )

        self._entries.append(entry)

        logger.info(
            "audit_trail_recorded",
            agent=agent_name,
            input_hash=input_hash[:12],
            output_hash=output_hash[:12],
            duration_ms=entry.duration_ms,
        )

        return entry

    def get_entries(
        self,
        agent_name: str | None = None,
    ) -> list[AuditTrailEntry]:
        """Return audit trail entries, optionally filtered by agent name."""
        if agent_name:
            return [e for e in self._entries if e.agent_name == agent_name]
        return list(self._entries)

    def clear(self) -> None:
        """Clear all audit trail entries."""
        self._entries.clear()

    @property
    def entry_count(self) -> int:
        """Total number of recorded entries."""
        return len(self._entries)

    @staticmethod
    def _hash_data(data: Any) -> str:
        """Produce a SHA-256 hex digest of the serialized data."""
        try:
            serialized = json.dumps(data, sort_keys=True, default=str)
        except (TypeError, ValueError):
            serialized = str(data)
        return hashlib.sha256(serialized.encode()).hexdigest()
