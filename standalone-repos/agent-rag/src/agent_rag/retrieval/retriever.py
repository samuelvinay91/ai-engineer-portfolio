"""Multi-strategy retriever: keyword, semantic, and hybrid search.

Implements three retrieval strategies that can be used independently or
combined via Reciprocal Rank Fusion (RRF):
- KeywordRetriever:  BM25-style keyword matching
- SemanticRetriever: vector similarity search via Qdrant
- HybridRetriever:   combines keyword + semantic with RRF
"""

from __future__ import annotations

import math
import re
from abc import ABC, abstractmethod
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient, models

from agent_rag.config import settings
from agent_rag.ingestion.embedder import BaseEmbedder, get_embedder

logger = structlog.get_logger(__name__)


@dataclass
class RetrievedChunk:
    """A single retrieved chunk with its relevance score."""

    id: str
    text: str
    score: float
    document_id: str = ""
    document_source: str = ""
    level: str = "paragraph"
    section_title: str = ""
    parent_chunk_id: str | None = None
    chunk_index: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class RetrievalResult:
    """Result of a retrieval operation."""

    query: str
    chunks: list[RetrievedChunk]
    strategy: str
    total_candidates: int = 0

    @property
    def texts(self) -> list[str]:
        return [c.text for c in self.chunks]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseRetriever(ABC):
    """Base class for all retrieval strategies."""

    @abstractmethod
    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        """Retrieve chunks relevant to the query."""


# ---------------------------------------------------------------------------
# Keyword (BM25-style) retriever
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"\b\w+\b", text.lower())


class KeywordRetriever(BaseRetriever):
    """BM25-style keyword search over Qdrant payloads.

    Since Qdrant is primarily a vector DB, this implementation scrolls
    documents and computes BM25 scores in-memory.  For production with
    large corpora, consider a dedicated full-text index (Elasticsearch /
    Qdrant full-text).
    """

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient | None = None,
        collection: str | None = None,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        self._qdrant = qdrant_client or AsyncQdrantClient(url=settings.qdrant_url)
        self._collection = collection or settings.qdrant_collection
        self._k1 = k1
        self._b = b

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        top_k = top_k or settings.retrieval_top_k
        query_tokens = _tokenize(query)
        if not query_tokens:
            return RetrievalResult(query=query, chunks=[], strategy="keyword")

        # Scroll all points (for small-medium collections)
        scroll_filter = _build_qdrant_filter(filters) if filters else None
        records, _ = await self._qdrant.scroll(
            collection_name=self._collection,
            limit=10_000,
            with_payload=True,
            with_vectors=False,
            scroll_filter=scroll_filter,
        )

        if not records:
            return RetrievalResult(query=query, chunks=[], strategy="keyword")

        # Build corpus statistics
        doc_lengths: list[int] = []
        doc_tokens: list[list[str]] = []
        df: Counter[str] = Counter()

        for rec in records:
            text = (rec.payload or {}).get("text", "")
            tokens = _tokenize(text)
            doc_tokens.append(tokens)
            doc_lengths.append(len(tokens))
            for term in set(tokens):
                df[term] += 1

        n = len(records)
        avgdl = sum(doc_lengths) / n if n > 0 else 1

        # Score each document
        scored: list[tuple[float, int]] = []
        for i, tokens in enumerate(doc_tokens):
            score = 0.0
            tf_counts = Counter(tokens)
            dl = doc_lengths[i]
            for qt in query_tokens:
                if qt in tf_counts:
                    tf_val = tf_counts[qt]
                    idf = math.log((n - df[qt] + 0.5) / (df[qt] + 0.5) + 1.0)
                    numerator = tf_val * (self._k1 + 1)
                    denominator = tf_val + self._k1 * (1 - self._b + self._b * dl / avgdl)
                    score += idf * numerator / denominator
            if score > 0:
                scored.append((score, i))

        scored.sort(reverse=True)
        top_scored = scored[:top_k]

        chunks: list[RetrievedChunk] = []
        for score, idx in top_scored:
            rec = records[idx]
            payload = rec.payload or {}
            chunks.append(RetrievedChunk(
                id=str(rec.id),
                text=payload.get("text", ""),
                score=score,
                document_id=payload.get("document_id", ""),
                document_source=payload.get("document_source", ""),
                level=payload.get("level", "paragraph"),
                section_title=payload.get("section_title", ""),
                parent_chunk_id=payload.get("parent_chunk_id"),
                chunk_index=payload.get("chunk_index", 0),
                metadata=payload,
            ))

        return RetrievalResult(
            query=query,
            chunks=chunks,
            strategy="keyword",
            total_candidates=len(scored),
        )


# ---------------------------------------------------------------------------
# Semantic (vector similarity) retriever
# ---------------------------------------------------------------------------

class SemanticRetriever(BaseRetriever):
    """Vector similarity search via Qdrant.

    Embeds the query and performs approximate nearest-neighbor search
    against the stored vectors.
    """

    def __init__(
        self,
        qdrant_client: AsyncQdrantClient | None = None,
        embedder: BaseEmbedder | None = None,
        collection: str | None = None,
    ) -> None:
        self._qdrant = qdrant_client or AsyncQdrantClient(url=settings.qdrant_url)
        self._embedder = embedder or get_embedder()
        self._collection = collection or settings.qdrant_collection

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        top_k = top_k or settings.retrieval_top_k
        query_vector = await self._embedder.embed_text(query)

        search_params = models.SearchParams(hnsw_ef=128, exact=False)
        qdrant_filter = _build_qdrant_filter(filters) if filters else None

        results = await self._qdrant.search(
            collection_name=self._collection,
            query_vector=query_vector,
            limit=top_k,
            search_params=search_params,
            query_filter=qdrant_filter,
            with_payload=True,
        )

        chunks: list[RetrievedChunk] = []
        for hit in results:
            payload = hit.payload or {}
            chunks.append(RetrievedChunk(
                id=str(hit.id),
                text=payload.get("text", ""),
                score=hit.score,
                document_id=payload.get("document_id", ""),
                document_source=payload.get("document_source", ""),
                level=payload.get("level", "paragraph"),
                section_title=payload.get("section_title", ""),
                parent_chunk_id=payload.get("parent_chunk_id"),
                chunk_index=payload.get("chunk_index", 0),
                metadata=payload,
            ))

        return RetrievalResult(
            query=query,
            chunks=chunks,
            strategy="semantic",
            total_candidates=len(results),
        )


# ---------------------------------------------------------------------------
# Hybrid retriever (keyword + semantic with RRF)
# ---------------------------------------------------------------------------

class HybridRetriever(BaseRetriever):
    """Combine keyword and semantic retrieval with Reciprocal Rank Fusion.

    RRF merges two ranked lists by scoring each document as:
        score = sum( 1 / (k + rank_i) )
    where rank_i is the document's rank in list i and k is a constant
    (default 60) that controls how much lower ranks are penalised.

    The *alpha* parameter controls the weight balance:
    - alpha = 1.0: pure semantic
    - alpha = 0.0: pure keyword
    - alpha = 0.7: default balanced towards semantic
    """

    def __init__(
        self,
        keyword_retriever: KeywordRetriever | None = None,
        semantic_retriever: SemanticRetriever | None = None,
        alpha: float | None = None,
        rrf_k: int | None = None,
    ) -> None:
        self._keyword = keyword_retriever or KeywordRetriever()
        self._semantic = semantic_retriever or SemanticRetriever()
        self._alpha = alpha if alpha is not None else settings.hybrid_alpha
        self._rrf_k = rrf_k if rrf_k is not None else settings.rrf_k

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        top_k = top_k or settings.retrieval_top_k

        # Retrieve from both strategies (fetch more candidates for fusion)
        candidate_k = top_k * 3
        keyword_result = await self._keyword.retrieve(
            query, top_k=candidate_k, filters=filters
        )
        semantic_result = await self._semantic.retrieve(
            query, top_k=candidate_k, filters=filters
        )

        # Build rank maps
        keyword_ranks: dict[str, int] = {}
        for rank, chunk in enumerate(keyword_result.chunks):
            keyword_ranks[chunk.id] = rank + 1  # 1-indexed

        semantic_ranks: dict[str, int] = {}
        for rank, chunk in enumerate(semantic_result.chunks):
            semantic_ranks[chunk.id] = rank + 1

        # Combine all chunks, de-duplicated
        all_chunks: dict[str, RetrievedChunk] = {}
        for c in keyword_result.chunks:
            all_chunks[c.id] = c
        for c in semantic_result.chunks:
            all_chunks[c.id] = c

        # Compute RRF scores
        rrf_scores: dict[str, float] = defaultdict(float)
        for chunk_id in all_chunks:
            keyword_rank = keyword_ranks.get(chunk_id, candidate_k + 1)
            semantic_rank = semantic_ranks.get(chunk_id, candidate_k + 1)

            keyword_score = (1.0 - self._alpha) / (self._rrf_k + keyword_rank)
            semantic_score = self._alpha / (self._rrf_k + semantic_rank)
            rrf_scores[chunk_id] = keyword_score + semantic_score

        # Sort by RRF score
        sorted_ids = sorted(rrf_scores, key=lambda cid: rrf_scores[cid], reverse=True)
        ranked: list[RetrievedChunk] = []
        for cid in sorted_ids[:top_k]:
            chunk = all_chunks[cid]
            chunk.score = rrf_scores[cid]
            ranked.append(chunk)

        total_candidates = len(keyword_result.chunks) + len(semantic_result.chunks)

        return RetrievalResult(
            query=query,
            chunks=ranked,
            strategy="hybrid",
            total_candidates=total_candidates,
        )


# ---------------------------------------------------------------------------
# Filter helper
# ---------------------------------------------------------------------------

def _build_qdrant_filter(filters: dict[str, Any]) -> models.Filter:
    """Build a Qdrant filter from a simple dict of field=value pairs."""
    conditions: list[models.FieldCondition] = []
    for key, value in filters.items():
        if isinstance(value, list):
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchAny(any=value),
                )
            )
        else:
            conditions.append(
                models.FieldCondition(
                    key=key,
                    match=models.MatchValue(value=value),
                )
            )
    return models.Filter(must=conditions)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_retriever(
    strategy: str | None = None,
    **kwargs: Any,
) -> BaseRetriever:
    """Create a retriever for the configured strategy."""
    name = (strategy or settings.retrieval_strategy).lower()
    if name == "keyword":
        return KeywordRetriever(**kwargs)
    elif name == "semantic":
        return SemanticRetriever(**kwargs)
    elif name == "hybrid":
        return HybridRetriever(**kwargs)
    else:
        raise ValueError(f"Unknown retrieval strategy: {name}")
