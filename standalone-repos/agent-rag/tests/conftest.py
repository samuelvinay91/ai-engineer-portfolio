"""Shared test fixtures for agent-rag tests."""

from __future__ import annotations

from typing import Any, Sequence

import pytest
import pytest_asyncio

from agent_rag.context_engineering import ContextBuilder
from agent_rag.ingestion.chunker import (
    Chunk,
    ChunkMetadata,
    FixedSizeChunker,
    HierarchicalChunker,
    RecursiveChunker,
    SemanticChunker,
)
from agent_rag.ingestion.embedder import BaseEmbedder, LocalEmbedder, Vector
from agent_rag.memory import ConversationMemory, EpisodicMemory, MemoryManager, SemanticMemory
from agent_rag.retrieval.reranker import BaseReranker, RerankResult
from agent_rag.retrieval.retriever import BaseRetriever, RetrievalResult, RetrievedChunk


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """# Introduction to Machine Learning

Machine learning is a subset of artificial intelligence that focuses on building
systems that learn from data. Instead of being explicitly programmed, these systems
improve their performance through experience.

## Supervised Learning

Supervised learning uses labeled training data to learn a mapping from inputs to
outputs. Common algorithms include linear regression, decision trees, and neural
networks. The model is trained on known input-output pairs and then makes
predictions on new, unseen data.

### Classification

Classification is a type of supervised learning where the output is a discrete
category. Examples include email spam detection, image recognition, and medical
diagnosis. Popular classification algorithms include logistic regression, SVM,
and random forests.

### Regression

Regression predicts continuous numerical values. Examples include predicting
house prices, stock market trends, and temperature forecasting. Linear regression
and polynomial regression are common approaches.

## Unsupervised Learning

Unsupervised learning finds patterns in unlabeled data. Clustering algorithms
like K-means and DBSCAN group similar data points together. Dimensionality
reduction techniques like PCA and t-SNE help visualise high-dimensional data.

## Reinforcement Learning

Reinforcement learning trains agents to make decisions by maximizing cumulative
rewards. The agent interacts with an environment, receiving rewards or penalties
for its actions. Applications include game playing, robotics, and autonomous
vehicles.
"""

SAMPLE_CHUNKS = [
    RetrievedChunk(
        id="chunk_001",
        text=(
            "Machine learning is a subset of artificial intelligence that focuses "
            "on building systems that learn from data."
        ),
        score=0.95,
        document_id="doc_001",
        document_source="ml_intro.md",
        level="paragraph",
        section_title="Introduction to Machine Learning",
        chunk_index=0,
    ),
    RetrievedChunk(
        id="chunk_002",
        text=(
            "Supervised learning uses labeled training data to learn a mapping "
            "from inputs to outputs. Common algorithms include linear regression, "
            "decision trees, and neural networks."
        ),
        score=0.88,
        document_id="doc_001",
        document_source="ml_intro.md",
        level="paragraph",
        section_title="Supervised Learning",
        chunk_index=1,
    ),
    RetrievedChunk(
        id="chunk_003",
        text=(
            "Classification is a type of supervised learning where the output is "
            "a discrete category. Examples include email spam detection and image "
            "recognition."
        ),
        score=0.82,
        document_id="doc_001",
        document_source="ml_intro.md",
        level="paragraph",
        section_title="Classification",
        chunk_index=2,
    ),
    RetrievedChunk(
        id="chunk_004",
        text=(
            "Unsupervised learning finds patterns in unlabeled data. Clustering "
            "algorithms like K-means and DBSCAN group similar data points together."
        ),
        score=0.75,
        document_id="doc_001",
        document_source="ml_intro.md",
        level="paragraph",
        section_title="Unsupervised Learning",
        chunk_index=3,
    ),
    RetrievedChunk(
        id="chunk_005",
        text=(
            "Reinforcement learning trains agents to make decisions by maximizing "
            "cumulative rewards. Applications include game playing, robotics, and "
            "autonomous vehicles."
        ),
        score=0.70,
        document_id="doc_001",
        document_source="ml_intro.md",
        level="paragraph",
        section_title="Reinforcement Learning",
        chunk_index=4,
    ),
]


# ---------------------------------------------------------------------------
# Mock retriever
# ---------------------------------------------------------------------------

class MockRetriever(BaseRetriever):
    """In-memory retriever for testing."""

    def __init__(self, chunks: list[RetrievedChunk] | None = None) -> None:
        self._chunks = chunks or SAMPLE_CHUNKS

    async def retrieve(
        self,
        query: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RetrievalResult:
        top_k = top_k or 5
        # Simple keyword matching for test
        query_terms = set(query.lower().split())
        scored = []
        for chunk in self._chunks:
            chunk_terms = set(chunk.text.lower().split())
            overlap = len(query_terms & chunk_terms)
            scored.append((overlap + chunk.score, chunk))

        scored.sort(key=lambda x: x[0], reverse=True)
        result_chunks = [c for _, c in scored[:top_k]]

        return RetrievalResult(
            query=query,
            chunks=result_chunks,
            strategy="mock",
            total_candidates=len(scored),
        )


class MockReranker(BaseReranker):
    """Pass-through reranker for testing."""

    async def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        *,
        top_k: int | None = None,
    ) -> list[RerankResult]:
        top_k = top_k or 5
        return [
            RerankResult(chunk=c, relevance_score=1.0 / (i + 1), rank=i + 1)
            for i, c in enumerate(chunks[:top_k])
        ]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_text() -> str:
    return SAMPLE_TEXT


@pytest.fixture
def sample_chunks() -> list[RetrievedChunk]:
    return list(SAMPLE_CHUNKS)


@pytest.fixture
def mock_retriever() -> MockRetriever:
    return MockRetriever()


@pytest.fixture
def mock_reranker() -> MockReranker:
    return MockReranker()


@pytest.fixture
def local_embedder() -> LocalEmbedder:
    return LocalEmbedder(dimensions=128)


@pytest.fixture
def fixed_chunker() -> FixedSizeChunker:
    return FixedSizeChunker(chunk_size=100, chunk_overlap=10)


@pytest.fixture
def semantic_chunker() -> SemanticChunker:
    return SemanticChunker(max_tokens=100)


@pytest.fixture
def hierarchical_chunker() -> HierarchicalChunker:
    return HierarchicalChunker(max_tokens=100)


@pytest.fixture
def recursive_chunker() -> RecursiveChunker:
    return RecursiveChunker(chunk_size=100, chunk_overlap=10)


@pytest.fixture
def context_builder() -> ContextBuilder:
    return ContextBuilder(max_tokens=1000, enable_compression=True)


@pytest.fixture
def memory_manager() -> MemoryManager:
    return MemoryManager(
        conversation=ConversationMemory(max_turns=10, max_conversations=50),
        episodic=EpisodicMemory(max_episodes=50),
        semantic=SemanticMemory(max_facts=50),
    )
