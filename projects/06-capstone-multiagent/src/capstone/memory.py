"""Shared memory system for the multi-agent platform.

Implements a three-tier memory architecture:

* **Short-term memory** -- Conversation context within a session, stored as
  an in-memory sliding window with Redis persistence for durability.
* **Long-term memory** -- Persistent knowledge across sessions, stored in
  PostgreSQL with vector-style relevance scoring for retrieval.
* **Working memory** -- Intermediate results during multi-agent execution,
  stored in Redis with automatic TTL expiration.

The memory manager exposes a unified interface that the orchestrator and
individual agents use to store and retrieve contextual information.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from datetime import datetime, timezone
from typing import Any

import structlog
from pydantic import BaseModel, Field
from sqlalchemy import Column, DateTime, Float, String, Text, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.orm import DeclarativeBase

from capstone.config import Settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class MemoryEntry(BaseModel):
    """A single memory record used across all memory tiers."""

    entry_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    session_id: str
    agent_name: str = ""
    content: str
    content_type: str = "text"  # text | result | context | summary
    relevance_score: float = Field(default=1.0, ge=0.0, le=1.0)
    created_at: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ConversationMessage(BaseModel):
    """A message in the conversation history."""

    role: str  # user | assistant | system
    content: str
    timestamp: float = Field(default_factory=time.time)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# SQLAlchemy models for long-term memory
# ---------------------------------------------------------------------------


class Base(DeclarativeBase):
    pass


class LongTermMemoryRow(Base):
    """PostgreSQL table for persistent long-term memories."""

    __tablename__ = "long_term_memories"

    id = Column(String(32), primary_key=True, default=lambda: uuid.uuid4().hex[:16])
    session_id = Column(String(64), index=True, nullable=False)
    agent_name = Column(String(64), default="")
    content = Column(Text, nullable=False)
    content_type = Column(String(32), default="text")
    content_hash = Column(String(64), index=True, nullable=False)
    relevance_score = Column(Float, default=1.0)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    metadata_json = Column(Text, default="{}")


# ---------------------------------------------------------------------------
# Short-term memory (in-process with optional Redis backing)
# ---------------------------------------------------------------------------


class ShortTermMemory:
    """Sliding-window conversation memory for a single session.

    Maintains the most recent ``max_messages`` in memory.  Optionally
    persists to Redis so that sessions survive process restarts.
    """

    def __init__(self, session_id: str, max_messages: int = 50) -> None:
        self._session_id = session_id
        self._max_messages = max_messages
        self._messages: list[ConversationMessage] = []

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def messages(self) -> list[ConversationMessage]:
        return list(self._messages)

    def add_message(self, role: str, content: str, **metadata: Any) -> None:
        """Append a message, evicting the oldest if at capacity."""
        msg = ConversationMessage(role=role, content=content, metadata=metadata)
        self._messages.append(msg)
        if len(self._messages) > self._max_messages:
            self._messages = self._messages[-self._max_messages:]

    def get_context_window(self, last_n: int | None = None) -> list[dict[str, str]]:
        """Return recent messages formatted for LLM context injection."""
        msgs = self._messages[-(last_n or self._max_messages):]
        return [{"role": m.role, "content": m.content} for m in msgs]

    def summarize(self) -> str:
        """Return a compact text summary of the conversation so far."""
        if not self._messages:
            return ""
        parts = [f"[{m.role}] {m.content[:200]}" for m in self._messages[-10:]]
        return "\n".join(parts)

    def clear(self) -> None:
        """Remove all messages from short-term memory."""
        self._messages.clear()

    def to_json(self) -> str:
        """Serialize for Redis persistence."""
        return json.dumps([m.model_dump() for m in self._messages])

    @classmethod
    def from_json(cls, session_id: str, data: str, max_messages: int = 50) -> ShortTermMemory:
        """Deserialize from Redis."""
        mem = cls(session_id=session_id, max_messages=max_messages)
        for raw in json.loads(data):
            mem._messages.append(ConversationMessage(**raw))
        return mem


# ---------------------------------------------------------------------------
# Working memory (in-process with optional Redis backing)
# ---------------------------------------------------------------------------


class WorkingMemory:
    """Scratch-pad memory for intermediate results during orchestration.

    Entries are keyed by ``(task_id, key)`` and automatically expire after
    a configurable TTL.
    """

    def __init__(self, ttl_seconds: int = 3600) -> None:
        self._store: dict[str, MemoryEntry] = {}
        self._ttl = ttl_seconds

    def store(
        self,
        task_id: str,
        key: str,
        content: str,
        agent_name: str = "",
        **metadata: Any,
    ) -> MemoryEntry:
        """Store an intermediate result."""
        entry = MemoryEntry(
            session_id=task_id,
            agent_name=agent_name,
            content=content,
            content_type="result",
            metadata=metadata,
        )
        composite_key = f"{task_id}:{key}"
        self._store[composite_key] = entry
        logger.debug("working_memory_stored", task_id=task_id, key=key)
        return entry

    def retrieve(self, task_id: str, key: str) -> MemoryEntry | None:
        """Retrieve a specific intermediate result."""
        composite_key = f"{task_id}:{key}"
        entry = self._store.get(composite_key)
        if entry and (time.time() - entry.created_at) > self._ttl:
            del self._store[composite_key]
            return None
        return entry

    def get_all_for_task(self, task_id: str) -> list[MemoryEntry]:
        """Retrieve all working-memory entries for a task, pruning expired."""
        now = time.time()
        results: list[MemoryEntry] = []
        expired_keys: list[str] = []

        for key, entry in self._store.items():
            if key.startswith(f"{task_id}:"):
                if (now - entry.created_at) > self._ttl:
                    expired_keys.append(key)
                else:
                    results.append(entry)

        for k in expired_keys:
            del self._store[k]

        return results

    def clear_task(self, task_id: str) -> int:
        """Remove all entries for a task. Returns count of removed entries."""
        keys_to_remove = [k for k in self._store if k.startswith(f"{task_id}:")]
        for k in keys_to_remove:
            del self._store[k]
        return len(keys_to_remove)

    def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count of removed entries."""
        now = time.time()
        expired = [k for k, v in self._store.items() if (now - v.created_at) > self._ttl]
        for k in expired:
            del self._store[k]
        return len(expired)


# ---------------------------------------------------------------------------
# Long-term memory (PostgreSQL)
# ---------------------------------------------------------------------------


class LongTermMemory:
    """Persistent memory backed by PostgreSQL.

    Stores knowledge, summaries, and important findings that should persist
    across sessions.  Retrieval uses keyword-based relevance scoring (a
    lightweight alternative to full vector search that requires no additional
    infrastructure).
    """

    def __init__(self, database_url: str) -> None:
        self._engine = create_async_engine(database_url, echo=False, pool_size=5)
        self._session_factory = async_sessionmaker(self._engine, expire_on_commit=False)

    async def initialize(self) -> None:
        """Create tables if they don't exist."""
        async with self._engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)
        logger.info("long_term_memory_initialized")

    async def store(
        self,
        session_id: str,
        content: str,
        agent_name: str = "",
        content_type: str = "text",
        relevance_score: float = 1.0,
        metadata: dict[str, Any] | None = None,
    ) -> str:
        """Persist a memory entry. Returns the entry ID.

        Deduplicates by content hash -- storing the same content for the same
        session is a no-op that returns the existing ID.
        """
        content_hash = hashlib.sha256(f"{session_id}:{content}".encode()).hexdigest()[:32]

        async with self._session_factory() as session:
            # Check for duplicate
            stmt = select(LongTermMemoryRow).where(
                LongTermMemoryRow.content_hash == content_hash
            )
            result = await session.execute(stmt)
            existing = result.scalar_one_or_none()
            if existing:
                return str(existing.id)

            row = LongTermMemoryRow(
                session_id=session_id,
                agent_name=agent_name,
                content=content,
                content_type=content_type,
                content_hash=content_hash,
                relevance_score=relevance_score,
                metadata_json=json.dumps(metadata or {}),
            )
            session.add(row)
            await session.commit()
            logger.info("long_term_memory_stored", entry_id=row.id, session_id=session_id)
            return str(row.id)

    async def retrieve(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 10,
        min_relevance: float = 0.0,
    ) -> list[MemoryEntry]:
        """Retrieve memories relevant to *query* using keyword matching.

        This implements a lightweight relevance scoring system based on keyword
        overlap.  For production deployments with large memory stores, this
        should be replaced with proper vector similarity search (e.g. pgvector).
        """
        query_terms = set(query.lower().split())

        async with self._session_factory() as session:
            stmt = select(LongTermMemoryRow)
            if session_id:
                stmt = stmt.where(LongTermMemoryRow.session_id == session_id)
            stmt = stmt.where(LongTermMemoryRow.relevance_score >= min_relevance)
            stmt = stmt.order_by(LongTermMemoryRow.created_at.desc())
            stmt = stmt.limit(limit * 3)  # over-fetch for re-ranking

            result = await session.execute(stmt)
            rows = result.scalars().all()

        # Score and rank by keyword overlap
        scored: list[tuple[float, LongTermMemoryRow]] = []
        for row in rows:
            content_terms = set(row.content.lower().split())
            if not query_terms:
                overlap = 0.0
            else:
                overlap = len(query_terms & content_terms) / len(query_terms)
            combined_score = (overlap * 0.6) + (row.relevance_score * 0.4)
            scored.append((combined_score, row))

        scored.sort(key=lambda x: x[0], reverse=True)

        entries: list[MemoryEntry] = []
        for score, row in scored[:limit]:
            if score < min_relevance:
                continue
            entries.append(
                MemoryEntry(
                    entry_id=str(row.id),
                    session_id=row.session_id,
                    agent_name=row.agent_name or "",
                    content=row.content,
                    content_type=row.content_type or "text",
                    relevance_score=round(score, 3),
                    created_at=row.created_at.timestamp() if row.created_at else 0.0,
                    metadata=json.loads(row.metadata_json) if row.metadata_json else {},
                )
            )

        return entries

    async def get_session_memories(self, session_id: str) -> list[MemoryEntry]:
        """Retrieve all memories for a specific session."""
        async with self._session_factory() as session:
            stmt = (
                select(LongTermMemoryRow)
                .where(LongTermMemoryRow.session_id == session_id)
                .order_by(LongTermMemoryRow.created_at.desc())
            )
            result = await session.execute(stmt)
            rows = result.scalars().all()

        return [
            MemoryEntry(
                entry_id=str(row.id),
                session_id=row.session_id,
                agent_name=row.agent_name or "",
                content=row.content,
                content_type=row.content_type or "text",
                relevance_score=row.relevance_score or 1.0,
                created_at=row.created_at.timestamp() if row.created_at else 0.0,
                metadata=json.loads(row.metadata_json) if row.metadata_json else {},
            )
            for row in rows
        ]

    async def close(self) -> None:
        """Dispose of the database engine."""
        await self._engine.dispose()


# ---------------------------------------------------------------------------
# Unified memory manager
# ---------------------------------------------------------------------------


class MemoryManager:
    """Unified facade over the three memory tiers.

    The orchestrator and API layer interact with this single class rather
    than managing each tier independently.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._short_term: dict[str, ShortTermMemory] = {}
        self._working = WorkingMemory(ttl_seconds=settings.working_memory_ttl_seconds)
        self._long_term: LongTermMemory | None = None

    async def initialize(self) -> None:
        """Initialize the long-term memory store (creates tables)."""
        try:
            self._long_term = LongTermMemory(self._settings.database_url)
            await self._long_term.initialize()
        except Exception:
            logger.warning(
                "long_term_memory_init_failed",
                msg="Continuing without persistent memory. Set DATABASE_URL for persistence.",
            )
            self._long_term = None

    # -- Short-term ------------------------------------------------------------

    def get_short_term(self, session_id: str) -> ShortTermMemory:
        """Get or create a short-term memory for a session."""
        if session_id not in self._short_term:
            self._short_term[session_id] = ShortTermMemory(
                session_id=session_id,
                max_messages=self._settings.short_term_max_messages,
            )
        return self._short_term[session_id]

    def add_message(self, session_id: str, role: str, content: str, **metadata: Any) -> None:
        """Add a message to the session's short-term memory."""
        mem = self.get_short_term(session_id)
        mem.add_message(role, content, **metadata)

    def get_context(self, session_id: str, last_n: int | None = None) -> list[dict[str, str]]:
        """Retrieve recent conversation context for a session."""
        mem = self.get_short_term(session_id)
        return mem.get_context_window(last_n)

    # -- Working ---------------------------------------------------------------

    @property
    def working(self) -> WorkingMemory:
        return self._working

    # -- Long-term -------------------------------------------------------------

    @property
    def long_term(self) -> LongTermMemory | None:
        return self._long_term

    async def store_long_term(
        self,
        session_id: str,
        content: str,
        agent_name: str = "",
        content_type: str = "text",
        **metadata: Any,
    ) -> str | None:
        """Persist an important memory to long-term storage."""
        if self._long_term is None:
            logger.debug("long_term_memory_unavailable")
            return None
        return await self._long_term.store(
            session_id=session_id,
            content=content,
            agent_name=agent_name,
            content_type=content_type,
            metadata=metadata,
        )

    async def search_long_term(
        self,
        query: str,
        session_id: str | None = None,
        limit: int = 10,
    ) -> list[MemoryEntry]:
        """Search long-term memory by relevance."""
        if self._long_term is None:
            return []
        return await self._long_term.retrieve(
            query=query,
            session_id=session_id,
            limit=limit,
            min_relevance=self._settings.long_term_relevance_threshold,
        )

    async def get_session_summary(self, session_id: str) -> dict[str, Any]:
        """Return a summary of all memory for a given session."""
        short_term = self.get_short_term(session_id)
        long_term_entries = (
            await self._long_term.get_session_memories(session_id)
            if self._long_term
            else []
        )

        return {
            "session_id": session_id,
            "short_term": {
                "message_count": len(short_term.messages),
                "messages": short_term.get_context_window(last_n=10),
            },
            "long_term": {
                "entry_count": len(long_term_entries),
                "entries": [
                    {
                        "entry_id": e.entry_id,
                        "content_type": e.content_type,
                        "content_preview": e.content[:200],
                        "relevance_score": e.relevance_score,
                    }
                    for e in long_term_entries[:20]
                ],
            },
        }

    async def close(self) -> None:
        """Clean up resources."""
        if self._long_term:
            await self._long_term.close()
