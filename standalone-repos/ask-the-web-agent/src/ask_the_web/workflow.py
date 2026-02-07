"""LangGraph workflow orchestrating the Ask-the-Web agent pipeline.

The workflow is a **StateGraph** with the following stages:

    query --> route --> search --> synthesise --> fact_check --> respond

Conditional edges handle routing decisions:
* ``creative`` queries skip the search and go straight to synthesis (with no
  sources).
* ``calculation`` queries use a single focused search then synthesise.
* ``deep_research`` queries may execute multiple search rounds.
* ``simple_search`` queries follow the standard single-search path.

The graph also implements error handling: if any node raises, the error
is captured in state and the pipeline short-circuits to the response node.
"""

from __future__ import annotations

import time
from typing import Any, TypedDict

import structlog
from langgraph.graph import END, StateGraph

from ask_the_web.agents.fact_checker import FactCheckReport, FactCheckerAgent
from ask_the_web.agents.router import QueryRouterAgent, RouteType, RoutingDecision
from ask_the_web.agents.searcher import SearchResponse, SearchResult, WebSearchAgent
from ask_the_web.agents.synthesizer import SynthesisAgent, SynthesisResult
from ask_the_web.config import Settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Workflow state
# ---------------------------------------------------------------------------

class WorkflowState(TypedDict, total=False):
    """Typed dictionary describing the state flowing through the graph."""

    # Input
    query: str
    # Routing
    routing_decision: RoutingDecision | None
    # Search
    search_response: SearchResponse | None
    search_results: list[SearchResult]
    # Synthesis
    synthesis_result: SynthesisResult | None
    # Fact checking
    fact_check_report: FactCheckReport | None
    # Final output
    answer: str
    citations: list[dict[str, Any]]
    follow_up_questions: list[str]
    # Meta
    error: str | None
    elapsed_ms: float
    route_used: str


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def _make_route_node(router: QueryRouterAgent):
    """Create the routing node function."""

    async def route_node(state: WorkflowState) -> WorkflowState:
        """Classify the query and decide on a processing strategy."""
        try:
            decision = await router.route(state["query"])
            return {
                **state,
                "routing_decision": decision,
                "route_used": decision.route.value,
                "error": None,
            }
        except Exception as exc:
            logger.exception("route_node_error")
            return {
                **state,
                "routing_decision": None,
                "route_used": "simple_search",
                "error": f"Routing failed: {exc}",
            }

    return route_node


def _make_search_node(searcher: WebSearchAgent):
    """Create the search node function."""

    async def search_node(state: WorkflowState) -> WorkflowState:
        """Execute web searches based on the routing decision."""
        decision = state.get("routing_decision")
        if decision is None:
            # Fallback: search with the raw query
            queries = [state["query"]]
            strategy = "general"
        else:
            queries = decision.search_queries or [decision.reformulated_query]
            # Pick strategy based on route
            strategy = {
                RouteType.SIMPLE_SEARCH: "general",
                RouteType.DEEP_RESEARCH: "deep",
                RouteType.CALCULATION: "general",
                RouteType.CREATIVE: "general",
            }.get(decision.route, "general")

        try:
            response = await searcher.search(queries, strategy=strategy)
            return {
                **state,
                "search_response": response,
                "search_results": response.results,
                "error": None,
            }
        except Exception as exc:
            logger.exception("search_node_error")
            return {
                **state,
                "search_response": None,
                "search_results": [],
                "error": f"Search failed: {exc}",
            }

    return search_node


def _make_synthesise_node(synthesiser: SynthesisAgent):
    """Create the synthesis node function."""

    async def synthesise_node(state: WorkflowState) -> WorkflowState:
        """Synthesise an answer from search results."""
        search_results = state.get("search_results", [])
        query = state["query"]

        try:
            result = await synthesiser.synthesise(query, search_results)
            return {
                **state,
                "synthesis_result": result,
                "answer": result.answer,
                "citations": [c.model_dump() for c in result.citations],
                "follow_up_questions": result.follow_up_questions,
                "error": None,
            }
        except Exception as exc:
            logger.exception("synthesise_node_error")
            return {
                **state,
                "synthesis_result": None,
                "answer": "I'm sorry, I was unable to generate an answer. Please try again.",
                "citations": [],
                "follow_up_questions": [],
                "error": f"Synthesis failed: {exc}",
            }

    return synthesise_node


def _make_fact_check_node(fact_checker: FactCheckerAgent):
    """Create the fact-check node function."""

    async def fact_check_node(state: WorkflowState) -> WorkflowState:
        """Cross-reference the synthesised answer against sources."""
        answer = state.get("answer", "")
        search_results = state.get("search_results", [])

        try:
            report = await fact_checker.check(answer, search_results)
            return {**state, "fact_check_report": report, "error": None}
        except Exception as exc:
            logger.exception("fact_check_node_error")
            return {
                **state,
                "fact_check_report": None,
                "error": f"Fact-check failed: {exc}",
            }

    return fact_check_node


async def respond_node(state: WorkflowState) -> WorkflowState:
    """Terminal node -- compute elapsed time and finalise output."""
    return {
        **state,
        "elapsed_ms": state.get("elapsed_ms", 0.0),
    }


# ---------------------------------------------------------------------------
# Conditional routing
# ---------------------------------------------------------------------------

def should_search(state: WorkflowState) -> str:
    """Decide whether to search or skip straight to synthesis.

    Creative queries skip search entirely; everything else searches.
    """
    decision = state.get("routing_decision")
    if decision and decision.route == RouteType.CREATIVE:
        return "synthesise"
    return "search"


def should_fact_check(state: WorkflowState) -> str:
    """Decide whether to run the fact-checker after synthesis.

    Fact-checking is skipped for creative routes or when disabled.
    """
    route = state.get("route_used", "")
    if route == RouteType.CREATIVE.value:
        return "respond"
    # If there was an error or no answer, skip fact check
    if state.get("error") or not state.get("answer"):
        return "respond"
    return "fact_check"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_workflow(settings: Settings) -> StateGraph:
    """Construct and compile the LangGraph workflow.

    Returns a compiled :class:`StateGraph` ready to be invoked with
    ``await graph.ainvoke({"query": "..."})``.
    """
    # Instantiate agents
    router = QueryRouterAgent(settings)
    searcher = WebSearchAgent(settings)
    synthesiser = SynthesisAgent(settings)
    fact_checker = FactCheckerAgent(settings)

    # Build graph
    graph = StateGraph(WorkflowState)

    # Add nodes
    graph.add_node("route", _make_route_node(router))
    graph.add_node("search", _make_search_node(searcher))
    graph.add_node("synthesise", _make_synthesise_node(synthesiser))
    graph.add_node("fact_check", _make_fact_check_node(fact_checker))
    graph.add_node("respond", respond_node)

    # Set entry point
    graph.set_entry_point("route")

    # Conditional edge after routing: search or skip to synthesise
    graph.add_conditional_edges(
        "route",
        should_search,
        {
            "search": "search",
            "synthesise": "synthesise",
        },
    )

    # After search, always synthesise
    graph.add_edge("search", "synthesise")

    # Conditional edge after synthesis: fact-check or go to respond
    graph.add_conditional_edges(
        "synthesise",
        should_fact_check,
        {
            "fact_check": "fact_check",
            "respond": "respond",
        },
    )

    # After fact-check, respond
    graph.add_edge("fact_check", "respond")

    # Respond is the terminal node
    graph.add_edge("respond", END)

    return graph


def compile_workflow(settings: Settings):
    """Build and compile the graph, returning a runnable."""
    graph = build_workflow(settings)
    return graph.compile()


# ---------------------------------------------------------------------------
# Convenience runner
# ---------------------------------------------------------------------------

async def run_ask_pipeline(
    query: str,
    settings: Settings,
) -> WorkflowState:
    """Run the full ask-the-web pipeline for a single question.

    This is the primary entry point for non-streaming usage.

    Parameters
    ----------
    query:
        The user's natural-language question.
    settings:
        Application settings.

    Returns
    -------
    WorkflowState
        The final state containing the answer, citations, follow-up
        questions, fact-check report, and metadata.
    """
    start = time.perf_counter()
    compiled = compile_workflow(settings)

    initial_state: WorkflowState = {
        "query": query,
        "routing_decision": None,
        "search_response": None,
        "search_results": [],
        "synthesis_result": None,
        "fact_check_report": None,
        "answer": "",
        "citations": [],
        "follow_up_questions": [],
        "error": None,
        "elapsed_ms": 0.0,
        "route_used": "",
    }

    result = await compiled.ainvoke(initial_state)
    result["elapsed_ms"] = round((time.perf_counter() - start) * 1000, 1)

    logger.info(
        "pipeline_complete",
        query=query[:80],
        route=result.get("route_used"),
        elapsed_ms=result["elapsed_ms"],
        has_error=result.get("error") is not None,
    )

    return result  # type: ignore[return-value]
