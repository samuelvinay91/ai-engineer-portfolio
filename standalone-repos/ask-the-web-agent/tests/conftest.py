"""Shared test fixtures for the Ask-the-Web agent test suite."""

from __future__ import annotations

from typing import Any, AsyncIterator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio

from ask_the_web.agents.fact_checker import ClaimVerification, FactCheckReport
from ask_the_web.agents.router import RouteType, RoutingDecision
from ask_the_web.agents.searcher import SearchResponse, SearchResult
from ask_the_web.agents.synthesizer import Citation, SynthesisResult
from ask_the_web.config import Settings


# ---------------------------------------------------------------------------
# Settings fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def settings() -> Settings:
    """Return a Settings instance with dummy API keys for testing."""
    return Settings(
        anthropic_api_key="test-anthropic-key",
        tavily_api_key="test-tavily-key",
        openai_api_key="test-openai-key",
        default_model="claude-sonnet-4-20250514",
        fact_check_enabled=True,
        log_level="DEBUG",
        environment="test",
    )


# ---------------------------------------------------------------------------
# Mock data factories
# ---------------------------------------------------------------------------

@pytest.fixture()
def sample_routing_decision() -> RoutingDecision:
    """A typical routing decision for a factual question."""
    return RoutingDecision(
        route=RouteType.SIMPLE_SEARCH,
        confidence=0.92,
        reasoning="This is a factual question that can be answered with a web search.",
        reformulated_query="population of France 2024",
        search_queries=["population of France 2024", "France population latest"],
    )


@pytest.fixture()
def sample_search_results() -> list[SearchResult]:
    """A small set of realistic search results."""
    return [
        SearchResult(
            title="France Population (2024) - Worldometer",
            url="https://www.worldometers.info/world-population/france-population/",
            snippet="The current population of France is 68,170,228 as of 2024.",
            content=(
                "The current population of France is 68,170,228 based on projections of "
                "the latest United Nations data. France 2024 population is estimated at "
                "68,170,228 people at mid-year. France population is equivalent to 0.85% "
                "of the total world population."
            ),
            score=0.95,
            published_date="2024-01-15",
            source_domain="www.worldometers.info",
            content_hash="abc123def456",
        ),
        SearchResult(
            title="Demographics of France - Wikipedia",
            url="https://en.wikipedia.org/wiki/Demographics_of_France",
            snippet="France has a population of approximately 68 million people.",
            content=(
                "France has a population of approximately 68 million people as of January "
                "2024. The country is the second most populated in the European Union "
                "after Germany. Metropolitan France has a density of 122 inhabitants "
                "per square kilometre."
            ),
            score=0.88,
            published_date=None,
            source_domain="en.wikipedia.org",
            content_hash="ghi789jkl012",
        ),
        SearchResult(
            title="France - The World Factbook - CIA",
            url="https://www.cia.gov/the-world-factbook/countries/france/",
            snippet="Population: 68,521,974 (2024 est.)",
            content=(
                "Population: 68,521,974 (2024 est.). Country comparison to the world: 21. "
                "Nationality: noun: Frenchman, Frenchwoman. adjective: French. "
                "Ethnic groups: Celtic and Latin with Teutonic, Slavic, North African, "
                "Sub-Saharan African, Indochinese, Basque minorities."
            ),
            score=0.85,
            published_date="2024-06-01",
            source_domain="www.cia.gov",
            content_hash="mno345pqr678",
        ),
    ]


@pytest.fixture()
def sample_search_response(sample_search_results: list[SearchResult]) -> SearchResponse:
    """A complete search response wrapping sample results."""
    return SearchResponse(
        query="population of France 2024",
        results=sample_search_results,
        total_results=3,
        search_queries_used=["population of France 2024", "France population latest"],
        search_duration_ms=450.0,
    )


@pytest.fixture()
def sample_synthesis_result() -> SynthesisResult:
    """A typical synthesis result with citations."""
    return SynthesisResult(
        answer=(
            "The population of France in 2024 is approximately **68.2 million** people "
            "[1][2]. According to the CIA World Factbook, the more precise estimate is "
            "68,521,974 [3]. France is the **second most populated country** in the "
            "European Union, after Germany [2].\n\n"
            "Metropolitan France has a population density of about 122 inhabitants per "
            "square kilometre [2], and the country accounts for approximately 0.85% of "
            "the total world population [1]."
        ),
        citations=[
            Citation(
                index=1,
                url="https://www.worldometers.info/world-population/france-population/",
                title="France Population (2024) - Worldometer",
                snippet="The current population of France is 68,170,228 as of 2024.",
            ),
            Citation(
                index=2,
                url="https://en.wikipedia.org/wiki/Demographics_of_France",
                title="Demographics of France - Wikipedia",
                snippet="France has a population of approximately 68 million people.",
            ),
            Citation(
                index=3,
                url="https://www.cia.gov/the-world-factbook/countries/france/",
                title="France - The World Factbook - CIA",
                snippet="Population: 68,521,974 (2024 est.)",
            ),
        ],
        follow_up_questions=[
            "What is the population growth rate of France?",
            "How does France's population compare to other EU countries?",
            "What are the major demographic trends in France?",
        ],
        key_facts=[
            "France population is approximately 68.2 million in 2024",
            "France is the second most populated EU country after Germany",
            "Metropolitan France has a density of 122 inhabitants/km2",
        ],
        confidence=0.85,
    )


@pytest.fixture()
def sample_fact_check_report() -> FactCheckReport:
    """A typical fact-check report."""
    return FactCheckReport(
        claims=[
            ClaimVerification(
                claim="France population is approximately 68.2 million in 2024",
                supported_by=[1, 2, 3],
                contradicted_by=[],
                confidence=0.95,
                status="verified",
                note="Consistent across all three sources.",
            ),
            ClaimVerification(
                claim="France is the second most populated EU country",
                supported_by=[2],
                contradicted_by=[],
                confidence=0.8,
                status="verified",
                note="Stated in Wikipedia source.",
            ),
        ],
        overall_confidence=0.9,
        conflicts_found=0,
        flags=[],
    )
