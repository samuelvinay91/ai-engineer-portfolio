"""Conversation management with Redis-backed session storage.

This module owns the full lifecycle of a customer-support conversation:

* **Session persistence** -- every conversation is keyed by a ``session_id``
  and stored in Redis as a JSON blob.
* **Sliding-window history** -- only the most recent *N* messages are kept in
  the active context; older messages are summarised and prepended so the model
  never loses important context.
* **Sentiment tracking** -- a lightweight per-turn sentiment score is recorded
  so downstream logic (e.g. escalation rules) can react to deteriorating
  customer mood.
* **State machine** -- conversations transition through well-defined states
  (``greeting`` -> ``understanding`` -> ``resolving`` -> ``closing``) which
  influence prompt selection and escalation thresholds.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any, AsyncGenerator, Literal

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Conversation states
# ---------------------------------------------------------------------------

class ConversationState(StrEnum):
    """Finite-state-machine states for a support conversation."""

    GREETING = "greeting"
    UNDERSTANDING = "understanding"
    RESOLVING = "resolving"
    CLOSING = "closing"
    ESCALATED = "escalated"
    CLOSED = "closed"

    @staticmethod
    def valid_transitions() -> dict[str, list[str]]:
        """Return the allowed state-transition graph."""
        return {
            "greeting": ["understanding", "escalated", "closed"],
            "understanding": ["resolving", "escalated", "closed"],
            "resolving": ["understanding", "closing", "escalated", "closed"],
            "closing": ["understanding", "closed"],
            "escalated": ["closed"],
            "closed": [],
        }

    def can_transition_to(self, target: ConversationState) -> bool:
        return target.value in self.valid_transitions().get(self.value, [])


# ---------------------------------------------------------------------------
# Message model
# ---------------------------------------------------------------------------

class SentimentLevel(StrEnum):
    VERY_NEGATIVE = "very_negative"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"
    POSITIVE = "positive"
    VERY_POSITIVE = "very_positive"


@dataclass(slots=True)
class Message:
    """A single message in a conversation."""

    role: Literal["user", "assistant", "system"]
    content: str
    timestamp: float = field(default_factory=time.time)
    message_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    sentiment: SentimentLevel = SentimentLevel.NEUTRAL
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Message:
        data["sentiment"] = SentimentLevel(data.get("sentiment", "neutral"))
        return cls(**data)

    def to_llm_format(self) -> dict[str, str]:
        """Return the ``{role, content}`` dict expected by LLM APIs."""
        return {"role": self.role, "content": self.content}


# ---------------------------------------------------------------------------
# Conversation session
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ConversationSession:
    """Full state for a single customer conversation."""

    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    state: ConversationState = ConversationState.GREETING
    messages: list[Message] = field(default_factory=list)
    summary: str = ""
    customer_name: str | None = None
    intent: str | None = None
    sentiment_history: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- Serialisation --------------------------------------------------------

    def to_json(self) -> str:
        data = {
            "session_id": self.session_id,
            "state": self.state.value,
            "messages": [m.to_dict() for m in self.messages],
            "summary": self.summary,
            "customer_name": self.customer_name,
            "intent": self.intent,
            "sentiment_history": self.sentiment_history,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }
        return json.dumps(data)

    @classmethod
    def from_json(cls, raw: str) -> ConversationSession:
        data = json.loads(raw)
        messages = [Message.from_dict(m) for m in data.pop("messages", [])]
        data["state"] = ConversationState(data.get("state", "greeting"))
        data["sentiment_history"] = data.get("sentiment_history", [])
        return cls(messages=messages, **data)

    # -- Convenience ----------------------------------------------------------

    @property
    def turn_count(self) -> int:
        return sum(1 for m in self.messages if m.role == "user")

    @property
    def latest_sentiment(self) -> SentimentLevel:
        if self.sentiment_history:
            return SentimentLevel(self.sentiment_history[-1])
        return SentimentLevel.NEUTRAL

    def add_message(self, message: Message) -> None:
        self.messages.append(message)
        if message.role == "user":
            self.sentiment_history.append(message.sentiment.value)
        self.updated_at = time.time()


# ---------------------------------------------------------------------------
# ConversationManager -- Redis-backed session orchestration
# ---------------------------------------------------------------------------

class ConversationManager:
    """Manages conversation lifecycle, persistence, and context windowing.

    Parameters:
        redis_client: An ``redis.asyncio.Redis`` instance.
        max_history: Maximum messages retained in the sliding window.
        ttl_seconds: Redis key expiry for idle sessions.
        summary_threshold: Number of messages that triggers summarisation.
    """

    KEY_PREFIX = "chat:session:"

    def __init__(
        self,
        redis_client: Any,  # redis.asyncio.Redis
        *,
        max_history: int = 20,
        ttl_seconds: int = 86_400,
        summary_threshold: int = 15,
    ) -> None:
        self._redis = redis_client
        self._max_history = max_history
        self._ttl = ttl_seconds
        self._summary_threshold = summary_threshold

    # -- Key helpers ----------------------------------------------------------

    def _key(self, session_id: str) -> str:
        return f"{self.KEY_PREFIX}{session_id}"

    # -- CRUD -----------------------------------------------------------------

    async def create_session(
        self,
        *,
        session_id: str | None = None,
        customer_name: str | None = None,
    ) -> ConversationSession:
        session = ConversationSession(
            session_id=session_id or uuid.uuid4().hex,
            customer_name=customer_name,
        )
        await self._save(session)
        logger.info("session_created", session_id=session.session_id)
        return session

    async def get_session(self, session_id: str) -> ConversationSession | None:
        raw = await self._redis.get(self._key(session_id))
        if raw is None:
            return None
        return ConversationSession.from_json(raw)

    async def delete_session(self, session_id: str) -> bool:
        deleted = await self._redis.delete(self._key(session_id))
        if deleted:
            logger.info("session_deleted", session_id=session_id)
        return bool(deleted)

    async def _save(self, session: ConversationSession) -> None:
        await self._redis.set(
            self._key(session.session_id),
            session.to_json(),
            ex=self._ttl,
        )

    # -- Message handling -----------------------------------------------------

    async def add_message(
        self,
        session_id: str,
        role: Literal["user", "assistant", "system"],
        content: str,
        *,
        sentiment: SentimentLevel = SentimentLevel.NEUTRAL,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationSession:
        """Append a message to the conversation, applying the sliding window."""
        session = await self.get_session(session_id)
        if session is None:
            session = await self.create_session(session_id=session_id)

        msg = Message(
            role=role,
            content=content,
            sentiment=sentiment,
            metadata=metadata or {},
        )
        session.add_message(msg)

        # Sliding-window: summarise then trim
        if len(session.messages) > self._max_history:
            session = await self._apply_sliding_window(session)

        await self._save(session)
        return session

    # -- Sliding window & summarisation ---------------------------------------

    async def _apply_sliding_window(
        self,
        session: ConversationSession,
    ) -> ConversationSession:
        """Trim messages exceeding the window, prepending a summary."""
        overflow = len(session.messages) - self._max_history
        if overflow <= 0:
            return session

        old_messages = session.messages[:overflow]
        session.messages = session.messages[overflow:]

        # Build textual summary of trimmed messages
        summary_parts: list[str] = []
        if session.summary:
            summary_parts.append(session.summary)
        for m in old_messages:
            summary_parts.append(f"[{m.role}] {m.content[:200]}")

        session.summary = self._compress_summary("\n".join(summary_parts))
        logger.debug(
            "sliding_window_applied",
            session_id=session.session_id,
            trimmed=overflow,
            remaining=len(session.messages),
        )
        return session

    @staticmethod
    def _compress_summary(text: str, max_length: int = 1000) -> str:
        """Naively truncate the running summary to *max_length* characters.

        In production this would call an LLM to produce a proper abstractive
        summary; the truncation fallback keeps the system functional even when
        the LLM is unavailable.
        """
        if len(text) <= max_length:
            return text
        return text[:max_length].rsplit(" ", 1)[0] + " [truncated]"

    async def summarise_with_llm(
        self,
        session: ConversationSession,
        llm_callable: Any,
    ) -> str:
        """Use an LLM to produce an abstractive summary of the conversation.

        Args:
            session: The conversation to summarise.
            llm_callable: An async callable ``(system: str, messages: list) -> str``.

        Returns:
            The summary string.
        """
        system = (
            "You are a concise summariser.  Produce a brief summary of the "
            "following customer-support conversation, preserving all key facts, "
            "decisions, and unresolved questions.  Use no more than 4 sentences."
        )
        transcript = "\n".join(
            f"{m.role.upper()}: {m.content}" for m in session.messages
        )
        summary: str = await llm_callable(system, [{"role": "user", "content": transcript}])
        session.summary = summary
        await self._save(session)
        logger.info("llm_summary_generated", session_id=session.session_id, length=len(summary))
        return summary

    # -- State transitions ----------------------------------------------------

    async def transition_state(
        self,
        session_id: str,
        target: ConversationState,
    ) -> ConversationSession:
        """Advance the conversation state machine."""
        session = await self.get_session(session_id)
        if session is None:
            raise ValueError(f"Session {session_id!r} not found")

        if not session.state.can_transition_to(target):
            raise ValueError(
                f"Invalid transition {session.state.value!r} -> {target.value!r}"
            )

        old = session.state
        session.state = target
        await self._save(session)
        logger.info(
            "state_transition",
            session_id=session_id,
            old_state=old.value,
            new_state=target.value,
        )
        return session

    async def auto_transition(self, session: ConversationSession) -> ConversationState:
        """Heuristically determine the next state based on conversation signals.

        Rules:
        - greeting -> understanding: after the first user message.
        - understanding -> resolving: after the assistant has asked a clarifying
          question and received a reply (>=2 user turns).
        - resolving -> closing: after the assistant provides a solution and the
          user confirms satisfaction.
        """
        current = session.state
        turns = session.turn_count

        if current == ConversationState.GREETING and turns >= 1:
            target = ConversationState.UNDERSTANDING
        elif current == ConversationState.UNDERSTANDING and turns >= 2:
            target = ConversationState.RESOLVING
        elif current == ConversationState.RESOLVING and turns >= 4:
            # Check latest sentiment for satisfaction signal
            if session.latest_sentiment in (
                SentimentLevel.POSITIVE,
                SentimentLevel.VERY_POSITIVE,
            ):
                target = ConversationState.CLOSING
            else:
                target = current
        else:
            target = current

        if target != current and current.can_transition_to(target):
            session = await self.transition_state(session.session_id, target)

        return session.state

    # -- Context preparation for LLM -----------------------------------------

    def prepare_context(self, session: ConversationSession) -> list[dict[str, str]]:
        """Build the messages list to send to the LLM, including any summary.

        Returns a list of ``{"role": ..., "content": ...}`` dicts.
        """
        context: list[dict[str, str]] = []

        if session.summary:
            context.append({
                "role": "user",
                "content": (
                    f"[Context from earlier in the conversation]\n{session.summary}"
                ),
            })
            context.append({
                "role": "assistant",
                "content": (
                    "Thank you for the context. I have reviewed the earlier "
                    "conversation and am ready to continue helping you."
                ),
            })

        for msg in session.messages:
            if msg.role in ("user", "assistant"):
                context.append(msg.to_llm_format())

        return context

    # -- Sentiment helpers ----------------------------------------------------

    @staticmethod
    def compute_sentiment_trend(
        history: list[str],
        window: int = 5,
    ) -> Literal["improving", "stable", "deteriorating"]:
        """Analyse the sentiment trend over the last *window* entries.

        Uses a simple numeric mapping: very_negative=-2 ... very_positive=+2.
        """
        score_map: dict[str, int] = {
            "very_negative": -2,
            "negative": -1,
            "neutral": 0,
            "positive": 1,
            "very_positive": 2,
        }
        recent = history[-window:]
        if len(recent) < 2:
            return "stable"

        scores = [score_map.get(s, 0) for s in recent]
        delta = scores[-1] - scores[0]
        if delta > 0:
            return "improving"
        if delta < 0:
            return "deteriorating"
        return "stable"

    # -- Streaming helper -----------------------------------------------------

    async def stream_messages(
        self,
        session_id: str,
    ) -> AsyncGenerator[Message, None]:
        """Yield new messages as they arrive (polling-based placeholder).

        A production implementation would use Redis Pub/Sub or Streams.
        """
        last_count = 0
        while True:
            session = await self.get_session(session_id)
            if session is None:
                return
            if len(session.messages) > last_count:
                for msg in session.messages[last_count:]:
                    yield msg
                last_count = len(session.messages)
            if session.state == ConversationState.CLOSED:
                return
