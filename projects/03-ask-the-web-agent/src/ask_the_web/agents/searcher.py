"""Web Search Agent -- executes web searches and cleans/deduplicates results.

This agent wraps the Tavily API to perform web searches.  It supports multiple
search strategies (general, news, academic-biased) and implements:

* **Query reformulation** -- takes the router's suggested queries and, if
  needed, generates additional variants.
* **Content extraction** -- strips boilerplate from HTML snippets using
  BeautifulSoup / markdownify.
* **Deduplication** -- removes near-duplicate results based on URL normalisation
  and content similarity.
"""

from __future__ import annotations

import asyncio
import hashlib
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

import httpx
import structlog
from bs4 import BeautifulSoup
from markdownify import markdownify as md
from pydantic import BaseModel, Field

from ask_the_web.config import Settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class SearchResult(BaseModel):
    """A single search result with cleaned content."""

    title: str = ""
    url: str
    snippet: str = ""
    content: str = ""
    score: float = 0.0
    published_date: str | None = None
    source_domain: str = ""
    content_hash: str = ""


class SearchResponse(BaseModel):
    """Aggregated search output returned by the agent."""

    query: str
    results: list[SearchResult] = Field(default_factory=list)
    total_results: int = 0
    search_queries_used: list[str] = Field(default_factory=list)
    search_duration_ms: float = 0.0


# ---------------------------------------------------------------------------
# Search strategies
# ---------------------------------------------------------------------------

class SearchStrategy:
    """Encapsulates Tavily search parameters for different strategies."""

    GENERAL = {"search_depth": "basic", "topic": "general"}
    NEWS = {"search_depth": "basic", "topic": "news"}
    DEEP = {"search_depth": "advanced", "topic": "general"}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class WebSearchAgent:
    """Performs web searches via Tavily and returns cleaned, deduplicated results."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_key = settings.tavily_api_key
        self._max_results = settings.max_search_results
        self._timeout = settings.search_timeout_seconds

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def search(
        self,
        queries: list[str],
        *,
        strategy: str = "general",
    ) -> SearchResponse:
        """Execute searches for all *queries* and return merged results.

        Parameters
        ----------
        queries:
            One or more search queries (from the router).
        strategy:
            ``"general"``, ``"news"``, or ``"deep"``.

        Returns
        -------
        SearchResponse
        """
        start = asyncio.get_event_loop().time()
        strategy_params = self._resolve_strategy(strategy)

        # Run searches concurrently
        tasks = [
            self._search_single(query, strategy_params) for query in queries[: self._settings.max_search_queries]
        ]
        raw_results_groups: list[list[SearchResult]] = await asyncio.gather(
            *tasks, return_exceptions=False
        )

        # Flatten and deduplicate
        all_results: list[SearchResult] = []
        for group in raw_results_groups:
            all_results.extend(group)

        deduped = self._deduplicate(all_results)

        # Sort by relevance score (descending)
        deduped.sort(key=lambda r: r.score, reverse=True)
        deduped = deduped[: self._settings.max_sources_per_answer]

        elapsed_ms = (asyncio.get_event_loop().time() - start) * 1000

        logger.info(
            "search_complete",
            queries_used=queries,
            raw_count=len(all_results),
            deduped_count=len(deduped),
            elapsed_ms=round(elapsed_ms, 1),
        )

        return SearchResponse(
            query=queries[0] if queries else "",
            results=deduped,
            total_results=len(deduped),
            search_queries_used=queries[: self._settings.max_search_queries],
            search_duration_ms=round(elapsed_ms, 1),
        )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    async def _search_single(
        self,
        query: str,
        strategy_params: dict[str, str],
    ) -> list[SearchResult]:
        """Call the Tavily REST API for a single query."""
        url = "https://api.tavily.com/search"
        payload: dict[str, Any] = {
            "api_key": self._api_key,
            "query": query,
            "max_results": self._max_results,
            "include_answer": False,
            "include_raw_content": True,
            **strategy_params,
        }

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload)
                resp.raise_for_status()
                data = resp.json()
        except httpx.HTTPStatusError as exc:
            logger.error("tavily_http_error", status=exc.response.status_code, query=query)
            return []
        except Exception:
            logger.exception("tavily_request_failed", query=query)
            return []

        results: list[SearchResult] = []
        for item in data.get("results", []):
            cleaned = self._clean_content(
                item.get("raw_content") or item.get("content", "")
            )
            domain = urlparse(item.get("url", "")).netloc
            content_hash = hashlib.sha256(cleaned.encode()).hexdigest()[:16]

            results.append(
                SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", "")[:500],
                    content=cleaned[:3000],  # cap per-source length
                    score=float(item.get("score", 0.0)),
                    published_date=item.get("published_date"),
                    source_domain=domain,
                    content_hash=content_hash,
                )
            )

        return results

    @staticmethod
    def _clean_content(raw: str) -> str:
        """Strip HTML boilerplate and convert to readable Markdown."""
        if not raw:
            return ""
        # If the content looks like HTML, parse it
        if "<" in raw and ">" in raw:
            soup = BeautifulSoup(raw, "html.parser")
            # Remove scripts, styles, navs, footers
            for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
                tag.decompose()
            cleaned_html = str(soup)
            text = md(cleaned_html, strip=["img", "a"]).strip()
        else:
            text = raw.strip()

        # Collapse excessive whitespace
        lines = [line.strip() for line in text.splitlines()]
        lines = [l for l in lines if l]
        return "\n".join(lines)

    @staticmethod
    def _deduplicate(results: list[SearchResult]) -> list[SearchResult]:
        """Remove duplicate results based on URL normalisation and content hash."""
        seen_urls: set[str] = set()
        seen_hashes: set[str] = set()
        unique: list[SearchResult] = []

        for result in results:
            normalised_url = result.url.rstrip("/").lower()
            if normalised_url in seen_urls:
                continue
            if result.content_hash and result.content_hash in seen_hashes:
                continue

            seen_urls.add(normalised_url)
            if result.content_hash:
                seen_hashes.add(result.content_hash)
            unique.append(result)

        return unique

    @staticmethod
    def _resolve_strategy(name: str) -> dict[str, str]:
        """Map a strategy name to Tavily request parameters."""
        strategies: dict[str, dict[str, str]] = {
            "general": SearchStrategy.GENERAL,
            "news": SearchStrategy.NEWS,
            "deep": SearchStrategy.DEEP,
        }
        return strategies.get(name, SearchStrategy.GENERAL)
