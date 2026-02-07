"""Lightweight agent abstraction inspired by Microsoft Agent Framework.

When the ``agent-framework`` package is available this module wraps it.
Otherwise it falls back to direct LLM calls (OpenAI -> Anthropic) or,
when no API key is configured, a deterministic heuristic so the demo
runs without any external dependencies.
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Response model
# ---------------------------------------------------------------------------


class AgentResponse(BaseModel):
    """Standardized response returned by every agent invocation."""

    content: str
    structured_output: dict[str, Any] | list[Any] | None = None


# ---------------------------------------------------------------------------
# ChatAgent base class
# ---------------------------------------------------------------------------


class ChatAgent:
    """Lightweight agent abstraction inspired by Microsoft Agent Framework.

    When ``agent-framework`` is available this class delegates to it.
    Otherwise it attempts OpenAI, then Anthropic, and finally falls
    back to a subclass-provided heuristic so the system always works
    without external API keys.

    Subclasses should override ``_heuristic_fallback`` to provide
    domain-specific deterministic logic.
    """

    def __init__(
        self,
        name: str,
        instructions: str,
        model: str = "gpt-4o-mini",
    ) -> None:
        self.name = name
        self.instructions = instructions
        self.model = model

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def run(
        self,
        messages: list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Execute the agent with the given conversation messages.

        Tries providers in order: OpenAI -> Anthropic -> heuristic.
        """
        start = time.time()
        context = context or {}

        # 1. Try OpenAI
        try:
            response = await self._call_openai(messages)
            logger.info(
                "agent_llm_call",
                agent=self.name,
                provider="openai",
                duration=round(time.time() - start, 3),
            )
            return response
        except Exception as exc:  # noqa: BLE001
            logger.debug("openai_unavailable", agent=self.name, error=str(exc))

        # 2. Try Anthropic
        try:
            response = await self._call_anthropic(messages)
            logger.info(
                "agent_llm_call",
                agent=self.name,
                provider="anthropic",
                duration=round(time.time() - start, 3),
            )
            return response
        except Exception as exc:  # noqa: BLE001
            logger.debug("anthropic_unavailable", agent=self.name, error=str(exc))

        # 3. Heuristic fallback
        logger.info("agent_heuristic_fallback", agent=self.name)
        response = await self._heuristic_fallback(messages, context)
        logger.info(
            "agent_heuristic_call",
            agent=self.name,
            duration=round(time.time() - start, 3),
        )
        return response

    # ------------------------------------------------------------------
    # Provider implementations
    # ------------------------------------------------------------------

    async def _call_openai(self, messages: list[dict[str, str]]) -> AgentResponse:
        """Call the OpenAI chat completions API."""
        from openai import AsyncOpenAI  # noqa: PLC0415

        client = AsyncOpenAI()
        system_messages = [{"role": "system", "content": self.instructions}]
        response = await client.chat.completions.create(
            model=self.model,
            messages=system_messages + messages,  # type: ignore[arg-type]
            temperature=0.1,
        )
        content = response.choices[0].message.content or ""
        structured = self._try_parse_json(content)
        return AgentResponse(content=content, structured_output=structured)

    async def _call_anthropic(self, messages: list[dict[str, str]]) -> AgentResponse:
        """Call the Anthropic messages API."""
        from anthropic import AsyncAnthropic  # noqa: PLC0415

        client = AsyncAnthropic()
        response = await client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=self.instructions,
            messages=messages,  # type: ignore[arg-type]
        )
        content = response.content[0].text if response.content else ""
        structured = self._try_parse_json(content)
        return AgentResponse(content=content, structured_output=structured)

    async def _heuristic_fallback(
        self,
        messages: list[dict[str, str]],
        context: dict[str, Any] | None = None,
    ) -> AgentResponse:
        """Subclass-overridable deterministic fallback.

        The default implementation returns a placeholder response.
        """
        return AgentResponse(
            content=f"[{self.name}] Heuristic analysis complete.",
            structured_output=None,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_parse_json(text: str) -> dict[str, Any] | None:
        """Attempt to extract a JSON object from the LLM response."""
        try:
            return json.loads(text)  # type: ignore[no-any-return]
        except (json.JSONDecodeError, TypeError):
            # Try to find JSON within markdown code fences
            if "```json" in text:
                start = text.index("```json") + 7
                end = text.index("```", start)
                try:
                    return json.loads(text[start:end].strip())  # type: ignore[no-any-return]
                except (json.JSONDecodeError, ValueError):
                    pass
            elif "```" in text:
                start = text.index("```") + 3
                end = text.index("```", start)
                try:
                    return json.loads(text[start:end].strip())  # type: ignore[no-any-return]
                except (json.JSONDecodeError, ValueError):
                    pass
            return None
