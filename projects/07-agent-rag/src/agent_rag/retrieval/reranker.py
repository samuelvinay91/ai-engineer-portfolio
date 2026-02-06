"""Reranking service: Cohere reranker, LLM-based reranker, and cross-encoder concept.

Reranking is the second stage of retrieval: after an initial set of candidates
is retrieved (fast but approximate), a reranker scores each candidate more
carefully using cross-attention between query and document.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass

import structlog

from agent_rag.config import settings
from agent_rag.retrieval.retriever import RetrievedChunk

logger = structlog.get_logger(__name__)


@dataclass
class RerankResult:
    """A chunk with its rerank relevance score."""

    chunk: RetrievedChunk
    relevance_score: float
    rank: int


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseReranker(ABC):
    """Base class for reranking strategies."""

    @abstractmethod
    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        """Rerank chunks by relevance to the query."""


# ---------------------------------------------------------------------------
# Cohere reranker
# ---------------------------------------------------------------------------

class CohereReranker(BaseReranker):
    """Rerank using the Cohere Rerank API.

    Cohere's reranker is a cross-encoder model specifically trained for
    relevance scoring.  It reads query + document together, making it
    more accurate than bi-encoder (embedding) similarity.

    Tradeoff: slower than embedding similarity (cross-attention on each
    pair) but much better at distinguishing relevant from irrelevant
    passages.
    """

    def __init__(
        self,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        import cohere

        self._model = model or settings.reranker_model
        key = api_key or settings.cohere_api_key.get_secret_value()
        self._client = cohere.AsyncClientV2(api_key=key)

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        top_k = top_k or settings.reranker_top_k
        if not chunks:
            return []

        documents = [c.text for c in chunks]

        response = await self._client.rerank(
            model=self._model,
            query=query,
            documents=documents,
            top_n=min(top_k, len(chunks)),
        )

        results: list[RerankResult] = []
        for rank, item in enumerate(response.results):
            idx = item.index
            results.append(RerankResult(
                chunk=chunks[idx],
                relevance_score=item.relevance_score,
                rank=rank + 1,
            ))

        logger.info(
            "cohere_reranked",
            query_len=len(query),
            candidates=len(chunks),
            returned=len(results),
        )
        return results


# ---------------------------------------------------------------------------
# LLM-based reranker
# ---------------------------------------------------------------------------

_RERANK_SYSTEM = """You are a relevance scoring assistant.
Given a query and a list of text passages, score each passage's relevance
to the query on a scale of 0.0 to 1.0.

Output ONLY a JSON array of objects with "index" (0-based) and "score" fields.
Sort by score descending. Example:
[{"index": 2, "score": 0.95}, {"index": 0, "score": 0.72}, {"index": 1, "score": 0.3}]
"""


class LLMReranker(BaseReranker):
    """Rerank using an LLM for relevance scoring.

    Uses the configured LLM to evaluate query-passage relevance.
    More flexible than a dedicated reranker model -- can follow
    domain-specific relevance criteria -- but slower and costlier.

    Best for:
    - Small candidate sets (< 20 passages)
    - Custom relevance criteria
    - When no dedicated reranker API is available
    """

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        top_k = top_k or settings.reranker_top_k
        if not chunks:
            return []

        # Build passage list for the LLM
        passage_list = "\n\n".join(
            f"[Passage {i}]: {c.text[:500]}" for i, c in enumerate(chunks)
        )
        user_prompt = f"Query: {query}\n\nPassages:\n{passage_list}"

        try:
            raw = await self._call_llm(user_prompt)
            scored = json.loads(raw.strip())
            if not isinstance(scored, list):
                raise ValueError("Expected JSON array")

            results: list[RerankResult] = []
            for rank, item in enumerate(scored[:top_k]):
                idx = int(item["index"])
                score = float(item["score"])
                if 0 <= idx < len(chunks):
                    results.append(RerankResult(
                        chunk=chunks[idx],
                        relevance_score=score,
                        rank=rank + 1,
                    ))

            return results

        except Exception:
            logger.warning("llm_rerank_failed", exc_info=True)
            # Fallback: return chunks in original order with decaying scores
            return [
                RerankResult(chunk=c, relevance_score=1.0 / (i + 1), rank=i + 1)
                for i, c in enumerate(chunks[:top_k])
            ]

    @staticmethod
    async def _call_llm(user_prompt: str) -> str:
        from agent_rag.retrieval.query_processor import _call_llm

        return await _call_llm(_RERANK_SYSTEM, user_prompt)


# ---------------------------------------------------------------------------
# Cross-encoder concept (educational)
# ---------------------------------------------------------------------------

class CrossEncoderReranker(BaseReranker):
    """Conceptual cross-encoder reranker.

    In a production deployment this would load a cross-encoder model
    (e.g., ms-marco-MiniLM-L-12-v2) via sentence-transformers.  The key
    difference from bi-encoders:

    Bi-encoder (embedding similarity):
        score = cosine(embed(query), embed(passage))
        - Query and passage encoded *independently*
        - Very fast (pre-compute passage embeddings)
        - Less accurate for nuanced relevance

    Cross-encoder:
        score = model([query, passage])
        - Query and passage encoded *together* with full cross-attention
        - Slow (must run model for each query-passage pair)
        - More accurate, especially for subtle relevance distinctions

    This implementation falls back to simple keyword overlap scoring
    as a lightweight stand-in.
    """

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        top_k = top_k or settings.reranker_top_k
        if not chunks:
            return []

        query_terms = set(query.lower().split())
        scored: list[tuple[float, int]] = []

        for i, chunk in enumerate(chunks):
            chunk_terms = set(chunk.text.lower().split())
            if not query_terms:
                scored.append((0.0, i))
                continue
            overlap = len(query_terms & chunk_terms)
            score = overlap / len(query_terms)
            # Boost for exact phrase match
            if query.lower() in chunk.text.lower():
                score = min(score + 0.3, 1.0)
            scored.append((score, i))

        scored.sort(reverse=True)

        results: list[RerankResult] = []
        for rank, (score, idx) in enumerate(scored[:top_k]):
            results.append(RerankResult(
                chunk=chunks[idx],
                relevance_score=score,
                rank=rank + 1,
            ))

        return results


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_reranker(kind: str = "cohere") -> BaseReranker:
    """Create a reranker instance."""
    if kind == "cohere":
        return CohereReranker()
    elif kind == "llm":
        return LLMReranker()
    elif kind == "cross_encoder":
        return CrossEncoderReranker()
    else:
        raise ValueError(f"Unknown reranker: {kind}")
