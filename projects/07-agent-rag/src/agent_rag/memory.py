"""Agent memory: conversation, episodic, and semantic memory.

Three memory types serve different purposes:
- ConversationMemory: short-term chat history for multi-turn context
- EpisodicMemory: past query-answer pairs for similar query detection
- SemanticMemory: learned facts and user preferences

MemoryManager coordinates all three types and provides a unified interface.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

import structlog

from agent_rag.config import settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Conversation Memory
# ---------------------------------------------------------------------------

@dataclass
class ConversationTurn:
    """A single turn in a conversation."""

    role: str  # "user" or "assistant"
    content: str
    timestamp: float = field(default_factory=time.time)


class ConversationMemory:
    """Short-term conversation history for multi-turn interactions.

    Stores recent turns per conversation ID.  In production this would
    be backed by Redis with TTL; here we use an in-memory store with
    LRU eviction as a portable implementation.
    """

    def __init__(
        self,
        max_turns: int = 20,
        max_conversations: int = 1000,
        ttl_seconds: int | None = None,
    ) -> None:
        self._max_turns = max_turns
        self._max_conversations = max_conversations
        self._ttl = ttl_seconds or settings.conversation_ttl_seconds
        self._store: OrderedDict[str, list[ConversationTurn]] = OrderedDict()

    async def add_turn(
        self,
        conversation_id: str,
        role: str,
        content: str,
    ) -> None:
        """Add a turn to the conversation."""
        if conversation_id not in self._store:
            self._store[conversation_id] = []
        self._store.move_to_end(conversation_id)

        turns = self._store[conversation_id]
        turns.append(ConversationTurn(role=role, content=content))

        # Trim old turns
        if len(turns) > self._max_turns:
            self._store[conversation_id] = turns[-self._max_turns:]

        # Evict oldest conversations
        while len(self._store) > self._max_conversations:
            self._store.popitem(last=False)

    async def get_history(
        self,
        conversation_id: str,
        max_turns: int | None = None,
    ) -> list[dict[str, str]]:
        """Get conversation history as a list of role/content dicts.

        Returns messages formatted for LLM consumption.
        """
        turns = self._store.get(conversation_id, [])
        if not turns:
            return []

        # Filter expired turns
        now = time.time()
        valid = [t for t in turns if now - t.timestamp < self._ttl]
        self._store[conversation_id] = valid

        limit = max_turns or self._max_turns
        recent = valid[-limit:]
        return [{"role": t.role, "content": t.content} for t in recent]

    async def clear(self, conversation_id: str) -> None:
        """Clear a conversation's history."""
        self._store.pop(conversation_id, None)

    @property
    def active_conversations(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Episodic Memory
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    """A past query-answer pair stored for future retrieval."""

    query: str
    answer: str
    query_hash: str
    complexity: str = ""
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    """Store past query-answer pairs for similar query detection.

    When a new query arrives, the episodic memory is searched for similar
    past queries.  If a close match is found, the previous answer can be
    reused or used as a starting point, reducing latency and cost.

    Uses simple text hashing and keyword overlap for matching.
    In production, this would use embedding similarity.
    """

    def __init__(self, max_episodes: int | None = None) -> None:
        self._max = max_episodes or settings.episodic_memory_size
        self._episodes: OrderedDict[str, Episode] = OrderedDict()

    async def store(
        self,
        query: str,
        answer: str,
        complexity: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a query-answer episode."""
        query_hash = self._hash_query(query)
        episode = Episode(
            query=query,
            answer=answer,
            query_hash=query_hash,
            complexity=complexity,
            metadata=metadata or {},
        )
        self._episodes[query_hash] = episode
        self._episodes.move_to_end(query_hash)

        while len(self._episodes) > self._max:
            self._episodes.popitem(last=False)

    async def find_similar(
        self,
        query: str,
        threshold: float = 0.7,
        top_k: int = 3,
    ) -> list[Episode]:
        """Find similar past queries using keyword overlap.

        Args:
            query: The new query to match.
            threshold: Minimum similarity (0.0 - 1.0).
            top_k: Maximum number of similar episodes.

        Returns:
            List of similar episodes sorted by similarity.
        """
        query_terms = set(query.lower().split())
        if not query_terms:
            return []

        scored: list[tuple[float, Episode]] = []
        for episode in self._episodes.values():
            ep_terms = set(episode.query.lower().split())
            if not ep_terms:
                continue
            overlap = len(query_terms & ep_terms) / len(query_terms | ep_terms)
            if overlap >= threshold:
                scored.append((overlap, episode))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [ep for _, ep in scored[:top_k]]

        for ep in results:
            ep.access_count += 1

        return results

    async def get_exact(self, query: str) -> Episode | None:
        """Check for an exact (hash) match."""
        query_hash = self._hash_query(query)
        ep = self._episodes.get(query_hash)
        if ep:
            ep.access_count += 1
        return ep

    @staticmethod
    def _hash_query(query: str) -> str:
        normalized = " ".join(query.lower().split())
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    @property
    def size(self) -> int:
        return len(self._episodes)


# ---------------------------------------------------------------------------
# Semantic Memory
# ---------------------------------------------------------------------------

@dataclass
class Fact:
    """A learned fact or preference."""

    key: str
    value: str
    source: str = ""  # Where this fact was learned
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    access_count: int = 0


class SemanticMemory:
    """Long-term memory for learned facts and user preferences.

    Stores key-value facts extracted from conversations and documents.
    Can be queried to augment context with relevant background knowledge.

    Examples of stored facts:
    - User preference: "user prefers concise answers"
    - Domain fact: "The company uses Python 3.11"
    - Entity: "Project Alpha is a machine learning pipeline"
    """

    def __init__(self, max_facts: int | None = None) -> None:
        self._max = max_facts or settings.semantic_memory_size
        self._facts: OrderedDict[str, Fact] = OrderedDict()

    async def store(
        self,
        key: str,
        value: str,
        *,
        source: str = "",
        confidence: float = 1.0,
    ) -> None:
        """Store or update a fact."""
        normalized_key = key.lower().strip()
        existing = self._facts.get(normalized_key)

        if existing:
            # Update with higher confidence value, or refresh timestamp
            if confidence >= existing.confidence:
                existing.value = value
                existing.confidence = confidence
                existing.timestamp = time.time()
            self._facts.move_to_end(normalized_key)
        else:
            self._facts[normalized_key] = Fact(
                key=normalized_key,
                value=value,
                source=source,
                confidence=confidence,
            )

        while len(self._facts) > self._max:
            self._facts.popitem(last=False)

    async def recall(self, query: str, top_k: int = 5) -> list[Fact]:
        """Retrieve facts relevant to a query using keyword matching."""
        query_terms = set(query.lower().split())
        if not query_terms:
            return []

        scored: list[tuple[float, Fact]] = []
        for fact in self._facts.values():
            fact_terms = set(fact.key.split()) | set(fact.value.lower().split())
            overlap = len(query_terms & fact_terms)
            if overlap > 0:
                score = overlap / len(query_terms) * fact.confidence
                scored.append((score, fact))

        scored.sort(key=lambda x: x[0], reverse=True)
        results = [f for _, f in scored[:top_k]]

        for f in results:
            f.access_count += 1

        return results

    async def get(self, key: str) -> Fact | None:
        """Get a specific fact by key."""
        return self._facts.get(key.lower().strip())

    async def delete(self, key: str) -> bool:
        """Delete a fact."""
        normalized = key.lower().strip()
        if normalized in self._facts:
            del self._facts[normalized]
            return True
        return False

    @property
    def size(self) -> int:
        return len(self._facts)


# ---------------------------------------------------------------------------
# Memory Manager
# ---------------------------------------------------------------------------

class MemoryManager:
    """Coordinate all memory types into a unified interface.

    Provides access to:
    - conversation: short-term multi-turn history
    - episodic: past query-answer pairs
    - semantic: learned facts and preferences

    The manager can augment a query with relevant memories from all sources.
    """

    def __init__(
        self,
        *,
        conversation: ConversationMemory | None = None,
        episodic: EpisodicMemory | None = None,
        semantic: SemanticMemory | None = None,
    ) -> None:
        self.conversation = conversation or ConversationMemory()
        self.episodic = episodic or EpisodicMemory()
        self.semantic = semantic or SemanticMemory()

    async def augment_query(
        self,
        query: str,
        *,
        conversation_id: str | None = None,
    ) -> dict[str, Any]:
        """Gather relevant memories to augment a query.

        Returns a dict with:
        - history: recent conversation turns
        - similar_episodes: past similar queries and answers
        - relevant_facts: background facts
        """
        result: dict[str, Any] = {
            "history": [],
            "similar_episodes": [],
            "relevant_facts": [],
        }

        # Conversation history
        if conversation_id:
            result["history"] = await self.conversation.get_history(
                conversation_id, max_turns=6
            )

        # Similar past queries
        similar = await self.episodic.find_similar(query, threshold=0.5)
        result["similar_episodes"] = [
            {"query": ep.query, "answer": ep.answer[:200]}
            for ep in similar
        ]

        # Relevant facts
        facts = await self.semantic.recall(query)
        result["relevant_facts"] = [
            {"key": f.key, "value": f.value, "confidence": f.confidence}
            for f in facts
        ]

        return result

    async def get_stats(self) -> dict[str, int]:
        """Return memory statistics."""
        return {
            "active_conversations": self.conversation.active_conversations,
            "stored_episodes": self.episodic.size,
            "stored_facts": self.semantic.size,
        }
