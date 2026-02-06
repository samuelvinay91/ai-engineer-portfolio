"""Query Router Agent -- classifies incoming queries and picks the best strategy.

This agent demonstrates the *routing* agentic pattern: an LLM inspects the
user's question and decides which downstream pipeline should handle it.

Supported routes
----------------
* **simple_search** -- A factual question that can be answered with a single
  round of web search (e.g. "What is the population of France?").
* **deep_research** -- A nuanced or multi-faceted question requiring several
  search rounds and cross-referencing (e.g. "Compare the economic policies of
  the last three US presidents").
* **calculation** -- A question that primarily requires mathematical reasoning
  or data analysis, optionally augmented with search.
* **creative** -- A request for creative content (poems, stories, brainstorming)
  that does not need web search at all.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ask_the_web.config import Settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class RouteType(str, Enum):
    """Available routing destinations."""

    SIMPLE_SEARCH = "simple_search"
    DEEP_RESEARCH = "deep_research"
    CALCULATION = "calculation"
    CREATIVE = "creative"


class RoutingDecision(BaseModel):
    """Structured output produced by the router."""

    route: RouteType = Field(description="The chosen processing route.")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Confidence in the routing decision (0-1)."
    )
    reasoning: str = Field(description="Brief explanation of why this route was chosen.")
    reformulated_query: str = Field(
        description="An optimised version of the original query for downstream use."
    )
    search_queries: list[str] = Field(
        default_factory=list,
        description="Suggested search queries if the route involves web search.",
    )


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

ROUTER_SYSTEM_PROMPT = """\
You are a query-routing classifier inside an Ask-the-Web agent (similar to
Perplexity AI).  Your job is to analyse the user's question and decide the
best processing strategy.

## Routes

| Route            | When to use |
|------------------|-------------|
| simple_search    | Factual / lookup questions answerable with 1-2 web searches. |
| deep_research    | Complex, multi-part, comparative, or opinion-heavy questions needing several searches and cross-referencing. |
| calculation      | Questions that primarily require math, unit conversion, or data analysis.  May need a supporting search. |
| creative         | Requests for creative content (writing, brainstorming, ideation) that do NOT need web search. |

## Instructions

1. Read the user's query carefully.
2. Pick **exactly one** route.
3. Provide a confidence score between 0 and 1.
4. Reformulate the query into a concise, search-engine-friendly version.
5. If the route involves web search, suggest 1-3 diverse search queries.

You MUST respond with valid JSON matching this schema:
{
  "route": "<simple_search|deep_research|calculation|creative>",
  "confidence": <float 0-1>,
  "reasoning": "<short explanation>",
  "reformulated_query": "<optimised query>",
  "search_queries": ["<query1>", ...]
}
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class QueryRouterAgent:
    """LLM-powered query router that classifies questions and picks a strategy.

    The router is intentionally lightweight -- it makes a single LLM call with
    a constrained JSON output to keep latency low.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = ChatAnthropic(
            model=settings.router_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0.0,  # deterministic routing
            max_tokens=1024,
        )

    async def route(self, query: str) -> RoutingDecision:
        """Classify *query* and return a :class:`RoutingDecision`.

        Parameters
        ----------
        query:
            The raw user question.

        Returns
        -------
        RoutingDecision
            Contains the chosen route, confidence, reasoning, and optional
            reformulated queries.
        """
        logger.info("routing_query", query=query[:120])

        messages = [
            SystemMessage(content=ROUTER_SYSTEM_PROMPT),
            HumanMessage(content=query),
        ]

        try:
            response = await self._llm.ainvoke(messages)
            raw: str = response.content  # type: ignore[assignment]
            decision = self._parse_response(raw, query)
        except Exception:
            logger.exception("router_llm_error", query=query[:120])
            decision = self._fallback_decision(query)

        logger.info(
            "routing_decision",
            route=decision.route.value,
            confidence=decision.confidence,
            search_queries=decision.search_queries,
        )
        return decision

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_response(raw: str, original_query: str) -> RoutingDecision:
        """Parse the LLM's JSON response into a :class:`RoutingDecision`."""
        # The model may wrap JSON in markdown fences -- strip them.
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
        cleaned = cleaned.strip()

        try:
            data: dict[str, Any] = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("router_json_parse_failed", raw=raw[:300])
            return QueryRouterAgent._fallback_decision(original_query)

        # Normalise route value
        route_value = data.get("route", "simple_search").lower().strip()
        try:
            route = RouteType(route_value)
        except ValueError:
            route = RouteType.SIMPLE_SEARCH

        return RoutingDecision(
            route=route,
            confidence=min(max(float(data.get("confidence", 0.7)), 0.0), 1.0),
            reasoning=data.get("reasoning", "No reasoning provided."),
            reformulated_query=data.get("reformulated_query", original_query),
            search_queries=data.get("search_queries", [original_query]),
        )

    @staticmethod
    def _fallback_decision(query: str) -> RoutingDecision:
        """Return a conservative fallback when routing fails."""
        return RoutingDecision(
            route=RouteType.SIMPLE_SEARCH,
            confidence=0.5,
            reasoning="Fallback: unable to classify query; defaulting to simple search.",
            reformulated_query=query,
            search_queries=[query],
        )
