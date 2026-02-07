"""Intent classification and routing module.

Classifies incoming customer messages into one or more support intents using
an LLM-based approach with structured JSON output.  The classifier produces:

* A **primary intent** from a fixed taxonomy.
* An optional set of **secondary intents** (multi-label support).
* A **confidence score** (0.0 -- 1.0) for each label.
* A brief **reasoning** string explaining the classification decision.

Supported intents:

    billing      -- invoices, charges, refunds, subscriptions, pricing
    technical    -- bugs, errors, performance, integrations, API
    account      -- login, password, profile, permissions, settings
    general      -- product info, hours, feedback, how-to questions
    escalation   -- angry customer, legal, security, repeated failures

The module is deliberately decoupled from any specific LLM provider; it
accepts an async callable that implements a simple request/response contract.
"""

from __future__ import annotations

import json
import textwrap
import time
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Protocol, runtime_checkable

import structlog

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Intent taxonomy
# ---------------------------------------------------------------------------

class Intent(StrEnum):
    """Supported customer-support intents."""

    BILLING = "billing"
    TECHNICAL = "technical"
    ACCOUNT = "account"
    GENERAL = "general"
    ESCALATION = "escalation"

    @classmethod
    def all_values(cls) -> list[str]:
        return [m.value for m in cls]

    @classmethod
    def from_str(cls, value: str) -> Intent:
        """Lenient parser that normalises casing and whitespace."""
        normalised = value.strip().lower()
        try:
            return cls(normalised)
        except ValueError:
            # Fuzzy fallback: check prefix match
            for member in cls:
                if normalised.startswith(member.value[:4]):
                    return member
            return cls.GENERAL


# ---------------------------------------------------------------------------
# Classification result
# ---------------------------------------------------------------------------

@dataclass(frozen=True, slots=True)
class IntentScore:
    """A single intent label with its confidence score."""

    intent: Intent
    confidence: float  # 0.0 -- 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            object.__setattr__(
                self, "confidence", max(0.0, min(1.0, self.confidence))
            )


@dataclass(frozen=True, slots=True)
class ClassificationResult:
    """Full classification output for a single customer message."""

    primary: IntentScore
    secondary: tuple[IntentScore, ...] = ()
    reasoning: str = ""
    raw_response: str = ""
    latency_ms: float = 0.0

    @property
    def primary_intent(self) -> Intent:
        return self.primary.intent

    @property
    def primary_confidence(self) -> float:
        return self.primary.confidence

    @property
    def all_intents(self) -> list[IntentScore]:
        """Return primary + secondary sorted by descending confidence."""
        combined = [self.primary, *self.secondary]
        return sorted(combined, key=lambda s: s.confidence, reverse=True)

    @property
    def needs_escalation(self) -> bool:
        """True if escalation appears in any label above threshold."""
        return any(
            s.intent == Intent.ESCALATION and s.confidence >= 0.5
            for s in self.all_intents
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary_intent": self.primary.intent.value,
            "primary_confidence": round(self.primary.confidence, 3),
            "secondary_intents": [
                {"intent": s.intent.value, "confidence": round(s.confidence, 3)}
                for s in self.secondary
            ],
            "reasoning": self.reasoning,
            "needs_escalation": self.needs_escalation,
            "latency_ms": round(self.latency_ms, 1),
        }


# ---------------------------------------------------------------------------
# LLM callable protocol
# ---------------------------------------------------------------------------

@runtime_checkable
class LLMCallable(Protocol):
    """Protocol for the async LLM function the classifier depends on."""

    async def __call__(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str: ...


# ---------------------------------------------------------------------------
# IntentClassifier
# ---------------------------------------------------------------------------

_CLASSIFICATION_SYSTEM_PROMPT = textwrap.dedent("""\
    You are an intent-classification engine for a customer-support system.

    Given a customer message, classify it into one or more of these intents:
    - billing:    invoices, charges, refunds, subscriptions, pricing, payment methods
    - technical:  bugs, errors, crashes, performance, integrations, API, connectivity
    - account:    login, password, profile, permissions, settings, account changes
    - general:    product information, business hours, feedback, how-to questions
    - escalation: customer is angry/threatening, legal issues, security breaches,
                  repeated unresolved issues, explicit request for human agent

    Respond with ONLY a valid JSON object (no markdown, no extra text) in this
    exact schema:

    {
        "primary_intent": "<intent>",
        "primary_confidence": <float 0-1>,
        "secondary_intents": [
            {"intent": "<intent>", "confidence": <float 0-1>}
        ],
        "reasoning": "<one sentence explaining why>"
    }

    Rules:
    - Always include exactly one primary_intent.
    - secondary_intents may be empty or contain 1-2 entries.
    - Confidence values must sum to <= 1.5 across all labels.
    - If the message expresses strong negative emotion, include escalation
      as a secondary intent even if the primary topic is something else.
    - If unclear, default to general with lower confidence.
""")


class IntentClassifier:
    """LLM-powered intent classifier with multi-label support.

    Args:
        llm: An async callable matching the :class:`LLMCallable` protocol.
        confidence_threshold: Minimum confidence to include a secondary intent.
    """

    def __init__(
        self,
        llm: LLMCallable,
        *,
        confidence_threshold: float = 0.3,
    ) -> None:
        self._llm = llm
        self._threshold = confidence_threshold

    async def classify(
        self,
        message: str,
        *,
        conversation_context: list[dict[str, str]] | None = None,
    ) -> ClassificationResult:
        """Classify a customer message into support intents.

        Args:
            message: The customer message to classify.
            conversation_context: Optional prior conversation messages to give
                the classifier additional context.

        Returns:
            A :class:`ClassificationResult` with primary and secondary intents.
        """
        messages: list[dict[str, str]] = []
        if conversation_context:
            messages.extend(conversation_context[-4:])  # last 4 messages for context
        messages.append({"role": "user", "content": message})

        t0 = time.monotonic()
        try:
            raw_response = await self._llm(
                _CLASSIFICATION_SYSTEM_PROMPT,
                messages,
                temperature=0.0,
                max_tokens=512,
            )
        except Exception:
            logger.exception("classification_llm_error", message=message[:100])
            return self._fallback_result(message)

        latency_ms = (time.monotonic() - t0) * 1000

        return self._parse_response(raw_response, latency_ms=latency_ms)

    # -- Response parsing -----------------------------------------------------

    def _parse_response(
        self,
        raw: str,
        *,
        latency_ms: float = 0.0,
    ) -> ClassificationResult:
        """Parse the JSON response from the LLM into a ClassificationResult."""
        # Strip markdown fences if the model wrapped its output
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[-1]
        if cleaned.endswith("```"):
            cleaned = cleaned.rsplit("```", 1)[0]
        cleaned = cleaned.strip()

        try:
            data = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("classification_json_parse_failed", raw=raw[:200])
            return self._fallback_result(raw, latency_ms=latency_ms)

        # Primary intent
        primary_intent = Intent.from_str(data.get("primary_intent", "general"))
        primary_confidence = float(data.get("primary_confidence", 0.5))
        primary = IntentScore(intent=primary_intent, confidence=primary_confidence)

        # Secondary intents
        secondary: list[IntentScore] = []
        for sec in data.get("secondary_intents", []):
            intent = Intent.from_str(sec.get("intent", "general"))
            conf = float(sec.get("confidence", 0.0))
            if conf >= self._threshold and intent != primary_intent:
                secondary.append(IntentScore(intent=intent, confidence=conf))

        reasoning = data.get("reasoning", "")

        result = ClassificationResult(
            primary=primary,
            secondary=tuple(secondary),
            reasoning=reasoning,
            raw_response=raw,
            latency_ms=latency_ms,
        )

        logger.info(
            "intent_classified",
            primary=result.primary_intent.value,
            confidence=result.primary_confidence,
            secondary=[s.intent.value for s in result.secondary],
            latency_ms=round(latency_ms, 1),
        )
        return result

    @staticmethod
    def _fallback_result(
        context: str = "",
        *,
        latency_ms: float = 0.0,
    ) -> ClassificationResult:
        """Return a safe default when classification fails."""
        return ClassificationResult(
            primary=IntentScore(intent=Intent.GENERAL, confidence=0.3),
            reasoning="Fallback: classification failed or response unparseable.",
            raw_response=context[:200],
            latency_ms=latency_ms,
        )

    # -- Batch classification -------------------------------------------------

    async def classify_batch(
        self,
        messages: list[str],
    ) -> list[ClassificationResult]:
        """Classify multiple messages (sequentially for now)."""
        import asyncio
        tasks = [self.classify(msg) for msg in messages]
        return await asyncio.gather(*tasks)


# ---------------------------------------------------------------------------
# Routing helper
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class RouteDecision:
    """The result of routing a classified message to a handler."""

    intent: Intent
    template_name: str
    confidence: float
    needs_escalation: bool
    metadata: dict[str, Any] = field(default_factory=dict)


_INTENT_TO_TEMPLATE: dict[Intent, str] = {
    Intent.BILLING: "billing_support",
    Intent.TECHNICAL: "technical_support",
    Intent.ACCOUNT: "general_support",
    Intent.GENERAL: "general_support",
    Intent.ESCALATION: "general_support",
}


def route_to_template(result: ClassificationResult) -> RouteDecision:
    """Map a classification result to the appropriate prompt template."""
    template = _INTENT_TO_TEMPLATE.get(result.primary_intent, "general_support")
    return RouteDecision(
        intent=result.primary_intent,
        template_name=template,
        confidence=result.primary_confidence,
        needs_escalation=result.needs_escalation,
        metadata={"reasoning": result.reasoning},
    )
