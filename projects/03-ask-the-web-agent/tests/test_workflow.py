"""Tests for the LangGraph workflow with mocked agents.

These tests verify that the workflow graph correctly:
- Routes queries through the right nodes
- Handles the full pipeline (route -> search -> synthesise -> fact_check -> respond)
- Skips search for creative routes
- Handles errors gracefully
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from ask_the_web.agents.fact_checker import ClaimVerification, FactCheckReport
from ask_the_web.agents.router import RouteType, RoutingDecision
from ask_the_web.agents.searcher import SearchResponse, SearchResult
from ask_the_web.agents.synthesizer import Citation, SynthesisResult
from ask_the_web.config import Settings
from ask_the_web.workflow import (
    WorkflowState,
    build_workflow,
    compile_workflow,
    run_ask_pipeline,
    should_fact_check,
    should_search,
)


# ---------------------------------------------------------------------------
# Unit tests for conditional edge functions
# ---------------------------------------------------------------------------


class TestConditionalEdges:
    """Test the routing logic functions used by conditional edges."""

    def test_should_search_returns_search_for_simple(
        self, sample_routing_decision: RoutingDecision
    ) -> None:
        state: WorkflowState = {
            "query": "test",
            "routing_decision": sample_routing_decision,
            "route_used": "simple_search",
        }
        assert should_search(state) == "search"

    def test_should_search_returns_search_for_deep_research(self) -> None:
        decision = RoutingDecision(
            route=RouteType.DEEP_RESEARCH,
            confidence=0.9,
            reasoning="Complex question",
            reformulated_query="test",
            search_queries=["test"],
        )
        state: WorkflowState = {"query": "test", "routing_decision": decision}
        assert should_search(state) == "search"

    def test_should_search_skips_for_creative(self) -> None:
        decision = RoutingDecision(
            route=RouteType.CREATIVE,
            confidence=0.95,
            reasoning="Creative request",
            reformulated_query="write a poem",
            search_queries=[],
        )
        state: WorkflowState = {"query": "write a poem", "routing_decision": decision}
        assert should_search(state) == "synthesise"

    def test_should_search_defaults_to_search_when_no_decision(self) -> None:
        state: WorkflowState = {"query": "test", "routing_decision": None}
        assert should_search(state) == "search"

    def test_should_fact_check_returns_fact_check_for_simple(self) -> None:
        state: WorkflowState = {
            "query": "test",
            "route_used": "simple_search",
            "answer": "Some answer",
            "error": None,
        }
        assert should_fact_check(state) == "fact_check"

    def test_should_fact_check_skips_for_creative(self) -> None:
        state: WorkflowState = {
            "query": "test",
            "route_used": "creative",
            "answer": "A poem",
            "error": None,
        }
        assert should_fact_check(state) == "respond"

    def test_should_fact_check_skips_on_error(self) -> None:
        state: WorkflowState = {
            "query": "test",
            "route_used": "simple_search",
            "answer": "",
            "error": "Something broke",
        }
        assert should_fact_check(state) == "respond"

    def test_should_fact_check_skips_when_no_answer(self) -> None:
        state: WorkflowState = {
            "query": "test",
            "route_used": "simple_search",
            "answer": "",
            "error": None,
        }
        assert should_fact_check(state) == "respond"


# ---------------------------------------------------------------------------
# Integration test: full pipeline with mocked LLM/search
# ---------------------------------------------------------------------------


class TestFullWorkflow:
    """Test the complete workflow with all agents mocked."""

    @pytest.mark.asyncio
    async def test_full_pipeline_simple_search(
        self,
        settings: Settings,
        sample_routing_decision: RoutingDecision,
        sample_search_response: SearchResponse,
        sample_search_results: list[SearchResult],
        sample_synthesis_result: SynthesisResult,
        sample_fact_check_report: FactCheckReport,
    ) -> None:
        """The happy path: query -> route(simple) -> search -> synthesise -> fact_check -> respond."""
        with (
            patch(
                "ask_the_web.workflow.QueryRouterAgent"
            ) as MockRouter,
            patch(
                "ask_the_web.workflow.WebSearchAgent"
            ) as MockSearcher,
            patch(
                "ask_the_web.workflow.SynthesisAgent"
            ) as MockSynthesiser,
            patch(
                "ask_the_web.workflow.FactCheckerAgent"
            ) as MockFactChecker,
        ):
            # Configure mocks
            MockRouter.return_value.route = AsyncMock(return_value=sample_routing_decision)
            MockSearcher.return_value.search = AsyncMock(return_value=sample_search_response)
            MockSynthesiser.return_value.synthesise = AsyncMock(
                return_value=sample_synthesis_result
            )
            MockFactChecker.return_value.check = AsyncMock(
                return_value=sample_fact_check_report
            )

            result = await run_ask_pipeline("What is the population of France?", settings)

        # Verify the pipeline produced output
        assert result["query"] == "What is the population of France?"
        assert result["route_used"] == "simple_search"
        assert result["answer"] != ""
        assert len(result["citations"]) == 3
        assert len(result["follow_up_questions"]) == 3
        assert result["fact_check_report"] is not None
        assert result["fact_check_report"].overall_confidence == 0.9
        assert result["elapsed_ms"] > 0
        assert result["error"] is None

    @pytest.mark.asyncio
    async def test_creative_route_skips_search(
        self,
        settings: Settings,
        sample_synthesis_result: SynthesisResult,
    ) -> None:
        """Creative queries should skip search and go straight to synthesis."""
        creative_decision = RoutingDecision(
            route=RouteType.CREATIVE,
            confidence=0.95,
            reasoning="Creative writing request",
            reformulated_query="write a haiku about AI",
            search_queries=[],
        )

        with (
            patch("ask_the_web.workflow.QueryRouterAgent") as MockRouter,
            patch("ask_the_web.workflow.WebSearchAgent") as MockSearcher,
            patch("ask_the_web.workflow.SynthesisAgent") as MockSynthesiser,
            patch("ask_the_web.workflow.FactCheckerAgent") as MockFactChecker,
        ):
            MockRouter.return_value.route = AsyncMock(return_value=creative_decision)
            MockSearcher.return_value.search = AsyncMock()
            MockSynthesiser.return_value.synthesise = AsyncMock(
                return_value=sample_synthesis_result
            )
            MockFactChecker.return_value.check = AsyncMock()

            result = await run_ask_pipeline("Write a haiku about AI", settings)

        assert result["route_used"] == "creative"
        assert result["answer"] != ""
        # Search should NOT have been called
        MockSearcher.return_value.search.assert_not_called()
        # Fact checker should NOT have been called (creative route)
        MockFactChecker.return_value.check.assert_not_called()

    @pytest.mark.asyncio
    async def test_pipeline_handles_search_error(
        self,
        settings: Settings,
        sample_routing_decision: RoutingDecision,
        sample_synthesis_result: SynthesisResult,
    ) -> None:
        """If search fails, the pipeline should still produce an answer (with empty sources)."""
        with (
            patch("ask_the_web.workflow.QueryRouterAgent") as MockRouter,
            patch("ask_the_web.workflow.WebSearchAgent") as MockSearcher,
            patch("ask_the_web.workflow.SynthesisAgent") as MockSynthesiser,
            patch("ask_the_web.workflow.FactCheckerAgent") as MockFactChecker,
        ):
            MockRouter.return_value.route = AsyncMock(return_value=sample_routing_decision)
            MockSearcher.return_value.search = AsyncMock(side_effect=Exception("Tavily down"))
            MockSynthesiser.return_value.synthesise = AsyncMock(
                return_value=sample_synthesis_result
            )
            MockFactChecker.return_value.check = AsyncMock(
                return_value=FactCheckReport(overall_confidence=0.5)
            )

            result = await run_ask_pipeline("What is Python?", settings)

        # The pipeline should still complete (error captured in state)
        assert result["error"] is not None or result["answer"] != ""

    @pytest.mark.asyncio
    async def test_deep_research_uses_deep_strategy(
        self,
        settings: Settings,
        sample_search_response: SearchResponse,
        sample_synthesis_result: SynthesisResult,
        sample_fact_check_report: FactCheckReport,
    ) -> None:
        """Deep research routes should use the 'deep' search strategy."""
        deep_decision = RoutingDecision(
            route=RouteType.DEEP_RESEARCH,
            confidence=0.88,
            reasoning="Multi-faceted comparison question",
            reformulated_query="compare economic policies US presidents",
            search_queries=[
                "US presidential economic policy comparison",
                "Biden vs Trump economic record",
                "Obama economic legacy",
            ],
        )

        with (
            patch("ask_the_web.workflow.QueryRouterAgent") as MockRouter,
            patch("ask_the_web.workflow.WebSearchAgent") as MockSearcher,
            patch("ask_the_web.workflow.SynthesisAgent") as MockSynthesiser,
            patch("ask_the_web.workflow.FactCheckerAgent") as MockFactChecker,
        ):
            MockRouter.return_value.route = AsyncMock(return_value=deep_decision)
            MockSearcher.return_value.search = AsyncMock(return_value=sample_search_response)
            MockSynthesiser.return_value.synthesise = AsyncMock(
                return_value=sample_synthesis_result
            )
            MockFactChecker.return_value.check = AsyncMock(
                return_value=sample_fact_check_report
            )

            result = await run_ask_pipeline(
                "Compare the economic policies of the last three US presidents",
                settings,
            )

        assert result["route_used"] == "deep_research"
        # Verify search was called with strategy="deep"
        call_kwargs = MockSearcher.return_value.search.call_args
        assert call_kwargs[1]["strategy"] == "deep"


# ---------------------------------------------------------------------------
# Unit tests for graph structure
# ---------------------------------------------------------------------------


class TestGraphStructure:
    """Verify the graph has the expected nodes and edges."""

    def test_graph_has_all_nodes(self, settings: Settings) -> None:
        graph = build_workflow(settings)
        node_names = set(graph.nodes.keys())
        expected = {"route", "search", "synthesise", "fact_check", "respond"}
        assert expected.issubset(node_names), f"Missing nodes: {expected - node_names}"

    def test_graph_compiles(self, settings: Settings) -> None:
        compiled = compile_workflow(settings)
        assert compiled is not None
