"""Synthesis Agent -- generates a comprehensive answer with citations.

This agent demonstrates the **prompt chaining** agentic pattern.  Rather than
making a single monolithic LLM call, it breaks answer generation into a
multi-step chain:

1. **Extract** -- pull key facts from each source.
2. **Organise** -- rank and group facts by relevance to the query.
3. **Generate** -- write a coherent answer with inline ``[n]`` citations.
4. **Follow-up** -- suggest related questions the user might ask next.

Each step's output feeds into the next, producing a higher-quality answer
than a single-shot prompt would yield.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ask_the_web.agents.searcher import SearchResult
from ask_the_web.config import Settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class Citation(BaseModel):
    """An inline citation linking a claim to a source."""

    index: int = Field(description="1-based citation number shown in the answer.")
    url: str
    title: str
    snippet: str = ""


class SynthesisResult(BaseModel):
    """The final synthesised answer produced by the agent."""

    answer: str = Field(description="Markdown-formatted answer with inline [n] citations.")
    citations: list[Citation] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    key_facts: list[str] = Field(default_factory=list)
    confidence: float = Field(
        ge=0.0, le=1.0, default=0.8, description="Overall confidence in the answer."
    )


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

EXTRACT_FACTS_PROMPT = """\
You are a research assistant.  Given the user's question and a set of web
search results, extract the key factual claims from the sources.

## User question
{query}

## Sources
{sources}

## Instructions
- Extract **only** concrete facts relevant to the question.
- For each fact, note which source(s) support it (by index number).
- Return a JSON array of objects: [{{"fact": "...", "source_indices": [1, 2]}}]
- Maximum 15 facts.
"""

ORGANISE_FACTS_PROMPT = """\
You are a research organiser.  Given extracted facts and the user's question,
rank and group them so the most relevant and important facts come first.

## User question
{query}

## Extracted facts
{facts_json}

## Instructions
- Remove duplicate or trivially redundant facts.
- Group related facts together under brief headings.
- Return a JSON object: {{"grouped_facts": [{{"heading": "...", "facts": ["...", ...]}}], "relevance_order": [<fact indices>]}}
"""

GENERATE_ANSWER_PROMPT = """\
You are an expert research writer (like Perplexity AI).  Using the organised
facts and original sources, write a comprehensive answer to the user's question.

## User question
{query}

## Organised facts
{organised_facts}

## Sources (for citation)
{sources_for_citation}

## Rules
1. Write in clear, professional Markdown.
2. Use inline citations like **[1]**, **[2]** etc., corresponding to the source
   numbers.
3. Every factual claim MUST have at least one citation.
4. Start with a direct, concise answer to the question in the first paragraph.
5. Then expand with supporting detail.
6. If sources conflict, note the disagreement and cite both sides.
7. End with a brief summary sentence.
8. Do NOT invent information not found in the sources.
"""

FOLLOW_UP_PROMPT = """\
Based on the user's original question and the answer provided, suggest 3
natural follow-up questions the user might want to ask next.

## Original question
{query}

## Answer summary
{answer_summary}

Return a JSON array of exactly 3 strings.
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class SynthesisAgent:
    """Multi-step synthesis agent using prompt chaining."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = ChatAnthropic(
            model=settings.synthesizer_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
        )
        self._streaming_llm = ChatAnthropic(
            model=settings.synthesizer_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=settings.temperature,
            max_tokens=settings.max_tokens,
            streaming=True,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def synthesise(
        self,
        query: str,
        search_results: list[SearchResult],
    ) -> SynthesisResult:
        """Run the full prompt chain and return a :class:`SynthesisResult`.

        Chain: extract facts -> organise -> generate answer -> follow-ups.
        """
        sources_text = self._format_sources(search_results)
        citations = self._build_citations(search_results)

        # Step 1 -- Extract key facts
        logger.info("synthesis_step", step="extract_facts", query=query[:80])
        facts_json = await self._extract_facts(query, sources_text)

        # Step 2 -- Organise facts
        logger.info("synthesis_step", step="organise_facts")
        organised_facts = await self._organise_facts(query, facts_json)

        # Step 3 -- Generate answer with citations
        logger.info("synthesis_step", step="generate_answer")
        sources_for_citation = self._format_sources_for_citation(search_results)
        answer = await self._generate_answer(query, organised_facts, sources_for_citation)

        # Step 4 -- Generate follow-up questions
        logger.info("synthesis_step", step="follow_up_questions")
        follow_ups = await self._generate_follow_ups(query, answer[:500])

        # Parse key facts from step 1
        key_facts = self._parse_key_facts(facts_json)

        return SynthesisResult(
            answer=answer,
            citations=citations,
            follow_up_questions=follow_ups,
            key_facts=key_facts,
            confidence=0.85 if len(search_results) >= 3 else 0.6,
        )

    async def synthesise_stream(
        self,
        query: str,
        search_results: list[SearchResult],
    ) -> AsyncIterator[str]:
        """Stream the final answer token-by-token.

        This is a simplified streaming path that skips the full chain and
        generates the answer in one shot for lower latency.
        """
        sources_text = self._format_sources(search_results)
        sources_for_citation = self._format_sources_for_citation(search_results)

        prompt = GENERATE_ANSWER_PROMPT.format(
            query=query,
            organised_facts="(Not pre-organised -- use sources directly.)",
            sources_for_citation=sources_for_citation,
        )

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=f"Question: {query}\n\nSources:\n{sources_text}"),
        ]

        async for chunk in self._streaming_llm.astream(messages):
            if chunk.content:
                yield chunk.content  # type: ignore[misc]

    # ------------------------------------------------------------------
    # Chain steps
    # ------------------------------------------------------------------

    async def _extract_facts(self, query: str, sources_text: str) -> str:
        """Step 1: Extract key facts from sources."""
        prompt = EXTRACT_FACTS_PROMPT.format(query=query, sources=sources_text)
        messages = [SystemMessage(content=prompt), HumanMessage(content="Extract the key facts.")]
        resp = await self._llm.ainvoke(messages)
        return resp.content  # type: ignore[return-value]

    async def _organise_facts(self, query: str, facts_json: str) -> str:
        """Step 2: Organise and rank extracted facts."""
        prompt = ORGANISE_FACTS_PROMPT.format(query=query, facts_json=facts_json)
        messages = [SystemMessage(content=prompt), HumanMessage(content="Organise the facts.")]
        resp = await self._llm.ainvoke(messages)
        return resp.content  # type: ignore[return-value]

    async def _generate_answer(
        self,
        query: str,
        organised_facts: str,
        sources_for_citation: str,
    ) -> str:
        """Step 3: Generate the final cited answer."""
        prompt = GENERATE_ANSWER_PROMPT.format(
            query=query,
            organised_facts=organised_facts,
            sources_for_citation=sources_for_citation,
        )
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="Write the answer now."),
        ]
        resp = await self._llm.ainvoke(messages)
        return resp.content  # type: ignore[return-value]

    async def _generate_follow_ups(self, query: str, answer_summary: str) -> list[str]:
        """Step 4: Suggest follow-up questions."""
        prompt = FOLLOW_UP_PROMPT.format(query=query, answer_summary=answer_summary)
        messages = [SystemMessage(content=prompt), HumanMessage(content="Suggest follow-ups.")]
        resp = await self._llm.ainvoke(messages)
        raw: str = resp.content  # type: ignore[assignment]
        try:
            # Strip markdown fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[: cleaned.rfind("```")]
            return json.loads(cleaned.strip())
        except (json.JSONDecodeError, TypeError):
            logger.warning("follow_up_parse_failed", raw=raw[:200])
            return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_sources(results: list[SearchResult]) -> str:
        """Format search results into a numbered text block for prompts."""
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[Source {i}] {r.title}\n"
                f"URL: {r.url}\n"
                f"Content: {r.content[:1500]}\n"
            )
        return "\n---\n".join(parts)

    @staticmethod
    def _format_sources_for_citation(results: list[SearchResult]) -> str:
        """Compact reference list for the generation step."""
        lines: list[str] = []
        for i, r in enumerate(results, 1):
            lines.append(f"[{i}] {r.title} -- {r.url}")
        return "\n".join(lines)

    @staticmethod
    def _build_citations(results: list[SearchResult]) -> list[Citation]:
        """Build Citation objects from search results."""
        return [
            Citation(
                index=i,
                url=r.url,
                title=r.title,
                snippet=r.snippet[:300],
            )
            for i, r in enumerate(results, 1)
        ]

    @staticmethod
    def _parse_key_facts(facts_json: str) -> list[str]:
        """Best-effort parse of key facts from step 1 output."""
        try:
            cleaned = facts_json.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
            if cleaned.endswith("```"):
                cleaned = cleaned[: cleaned.rfind("```")]
            data = json.loads(cleaned.strip())
            if isinstance(data, list):
                return [item.get("fact", str(item)) for item in data][:15]
        except (json.JSONDecodeError, TypeError, AttributeError):
            pass
        return []
