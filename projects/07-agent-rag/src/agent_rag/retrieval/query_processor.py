"""Query processing: decomposition, rewriting, and expansion.

Uses an LLM for intelligent query transformation before retrieval:
- QueryDecomposer: breaks complex queries into sub-queries
- QueryRewriter: rewrites queries for better retrieval (HyDE, step-back)
- QueryExpander: adds synonyms and related terms
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum

import structlog

from agent_rag.config import LLMProvider, settings

logger = structlog.get_logger(__name__)


class RewriteStrategy(str, Enum):
    """Available query rewrite strategies."""

    HYDE = "hyde"               # Hypothetical Document Embeddings
    STEP_BACK = "step_back"    # Step-back prompting for broader context
    MULTI_QUERY = "multi_query" # Generate multiple query variants
    NONE = "none"


@dataclass
class ProcessedQuery:
    """Result of query processing."""

    original: str
    sub_queries: list[str] = field(default_factory=list)
    rewritten: str = ""
    hypothetical_answer: str = ""  # HyDE
    expanded_terms: list[str] = field(default_factory=list)
    strategy_used: RewriteStrategy = RewriteStrategy.NONE
    metadata: dict[str, str] = field(default_factory=dict)

    @property
    def effective_queries(self) -> list[str]:
        """Return the queries that should be sent to the retriever."""
        queries = list(self.sub_queries) if self.sub_queries else [self.original]
        if self.rewritten:
            queries.append(self.rewritten)
        if self.hypothetical_answer:
            queries.append(self.hypothetical_answer)
        return queries


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

async def _call_llm(system: str, user: str) -> str:
    """Call the configured LLM and return its text response."""
    if settings.llm_provider == LLMProvider.ANTHROPIC:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
        response = await client.messages.create(
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return response.content[0].text
    else:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        response = await client.chat.completions.create(
            model=settings.openai_model,
            max_tokens=settings.llm_max_tokens,
            temperature=settings.llm_temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return response.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# QueryDecomposer
# ---------------------------------------------------------------------------

_DECOMPOSE_SYSTEM = """You are a query decomposition assistant.
Given a complex user question, break it into simpler, independent sub-queries
that can each be answered by searching a document collection.

Rules:
- Output ONLY a JSON array of strings.
- Each sub-query should be self-contained and answerable independently.
- If the query is already simple, return a single-element array with the original.
- Maximum 5 sub-queries.
- Do NOT add any explanation.
"""


class QueryDecomposer:
    """Break complex queries into independent sub-queries.

    Complex questions like "Compare X and Y in terms of A, B, C" are split
    into focused sub-queries that can be answered via targeted retrieval.
    """

    async def decompose(self, query: str) -> list[str]:
        """Decompose *query* into sub-queries.

        Returns a list of 1-5 sub-queries.  For simple questions the
        original query is returned unchanged.
        """
        try:
            raw = await _call_llm(_DECOMPOSE_SYSTEM, query)
            # Parse JSON array
            raw = raw.strip()
            if raw.startswith("["):
                sub_queries: list[str] = json.loads(raw)
                if isinstance(sub_queries, list) and all(
                    isinstance(q, str) for q in sub_queries
                ):
                    return sub_queries[:5]
        except Exception:
            logger.warning("decomposition_failed", query=query, exc_info=True)

        return [query]


# ---------------------------------------------------------------------------
# QueryRewriter
# ---------------------------------------------------------------------------

_HYDE_SYSTEM = """You are a document retrieval assistant.
Given a question, write a short passage (3-5 sentences) that would appear
in a document containing the answer.  Do NOT answer the question directly;
instead, write text that a relevant document might contain.
Output ONLY the hypothetical passage, no explanation."""

_STEP_BACK_SYSTEM = """You are a query reformulation assistant.
Given a specific question, generate a broader, more general version of
the question that retrieves background context needed to answer the original.
Output ONLY the broader question, nothing else."""

_MULTI_QUERY_SYSTEM = """You are a query reformulation assistant.
Given a user question, generate 3 alternative phrasings that capture
the same information need but use different vocabulary and structure.
Output ONLY a JSON array of 3 strings. No explanation."""


class QueryRewriter:
    """Rewrite queries for better retrieval.

    Strategies:
    - HyDE: Generate a hypothetical answer, then embed it for retrieval.
      Works well for factual questions where the answer "shape" is predictable.
    - Step-back: Broaden the query to retrieve more background context.
      Useful for specific questions needing general knowledge.
    - Multi-query: Generate multiple phrasings to increase recall.
    """

    async def rewrite(
        self,
        query: str,
        strategy: RewriteStrategy = RewriteStrategy.MULTI_QUERY,
    ) -> ProcessedQuery:
        """Rewrite *query* using the specified strategy."""
        result = ProcessedQuery(original=query, strategy_used=strategy)

        if strategy == RewriteStrategy.NONE:
            return result

        try:
            if strategy == RewriteStrategy.HYDE:
                result.hypothetical_answer = await _call_llm(_HYDE_SYSTEM, query)

            elif strategy == RewriteStrategy.STEP_BACK:
                result.rewritten = await _call_llm(_STEP_BACK_SYSTEM, query)

            elif strategy == RewriteStrategy.MULTI_QUERY:
                raw = await _call_llm(_MULTI_QUERY_SYSTEM, query)
                raw = raw.strip()
                if raw.startswith("["):
                    variants = json.loads(raw)
                    if isinstance(variants, list):
                        result.sub_queries = [str(v) for v in variants[:3]]

        except Exception:
            logger.warning(
                "rewrite_failed",
                query=query,
                strategy=strategy.value,
                exc_info=True,
            )

        return result


# ---------------------------------------------------------------------------
# QueryExpander
# ---------------------------------------------------------------------------

_EXPAND_SYSTEM = """You are a search query expansion assistant.
Given a query, provide a JSON array of 3-5 related terms or synonyms that
should be included when searching documents.  These terms should capture
different ways the same concept might be expressed.
Output ONLY a JSON array of strings."""


class QueryExpander:
    """Add synonyms and related terms to improve keyword search recall.

    Generates alternate terms via an LLM so that keyword-based retrievers
    can match documents using different vocabulary.
    """

    async def expand(self, query: str) -> list[str]:
        """Return a list of expanded terms for the given query."""
        try:
            raw = await _call_llm(_EXPAND_SYSTEM, query)
            raw = raw.strip()
            if raw.startswith("["):
                terms = json.loads(raw)
                if isinstance(terms, list):
                    return [str(t) for t in terms[:5]]
        except Exception:
            logger.warning("expansion_failed", query=query, exc_info=True)
        return []


# ---------------------------------------------------------------------------
# Unified processor
# ---------------------------------------------------------------------------

class QueryProcessor:
    """Orchestrate query processing: decompose -> rewrite -> expand.

    Combines all query transformation techniques into a single pipeline
    that can be configured per request.
    """

    def __init__(self) -> None:
        self._decomposer = QueryDecomposer()
        self._rewriter = QueryRewriter()
        self._expander = QueryExpander()

    async def process(
        self,
        query: str,
        *,
        decompose: bool = True,
        rewrite_strategy: RewriteStrategy = RewriteStrategy.MULTI_QUERY,
        expand: bool = False,
    ) -> ProcessedQuery:
        """Process a query through the full transformation pipeline.

        Args:
            query: The user's original query.
            decompose: Whether to decompose complex queries.
            rewrite_strategy: Which rewrite strategy to apply.
            expand: Whether to expand with synonyms.

        Returns:
            A ProcessedQuery with all transformations applied.
        """
        logger.info(
            "processing_query",
            query=query,
            decompose=decompose,
            strategy=rewrite_strategy.value,
            expand=expand,
        )

        # Start with rewrite (includes original in effective_queries)
        result = await self._rewriter.rewrite(query, strategy=rewrite_strategy)

        # Decompose
        if decompose:
            sub_queries = await self._decomposer.decompose(query)
            if len(sub_queries) > 1:
                result.sub_queries = sub_queries

        # Expand
        if expand:
            terms = await self._expander.expand(query)
            result.expanded_terms = terms

        logger.info(
            "query_processed",
            original=query,
            sub_queries=len(result.sub_queries),
            has_rewrite=bool(result.rewritten),
            has_hyde=bool(result.hypothetical_answer),
            expanded_terms=len(result.expanded_terms),
        )

        return result
