"""Tests for retrieval components: chunking, embedding, context, memory, and reranking."""

from __future__ import annotations

import pytest

from agent_rag.context_engineering import (
    ContextBuilder,
    ContextCompressor,
    ContextRanker,
    estimate_tokens,
)
from agent_rag.ingestion.chunker import (
    Chunk,
    FixedSizeChunker,
    HierarchicalChunker,
    RecursiveChunker,
    SemanticChunker,
    get_chunker,
)
from agent_rag.ingestion.embedder import LocalEmbedder
from agent_rag.memory import ConversationMemory, EpisodicMemory, MemoryManager, SemanticMemory
from agent_rag.retrieval.retriever import RetrievedChunk


# =====================================================================
# Chunking tests
# =====================================================================


class TestFixedSizeChunker:
    def test_empty_text(self, fixed_chunker: FixedSizeChunker) -> None:
        chunks = fixed_chunker.chunk("")
        assert chunks == []

    def test_short_text(self, fixed_chunker: FixedSizeChunker) -> None:
        chunks = fixed_chunker.chunk("Hello world.")
        assert len(chunks) == 1
        assert chunks[0].text == "Hello world."

    def test_long_text_produces_multiple_chunks(
        self, fixed_chunker: FixedSizeChunker, sample_text: str
    ) -> None:
        chunks = fixed_chunker.chunk(
            sample_text, document_id="d1", document_source="test.md"
        )
        assert len(chunks) > 1
        for c in chunks:
            assert c.metadata.document_id == "d1"
            assert c.metadata.document_source == "test.md"
            assert c.metadata.total_chunks == len(chunks)

    def test_chunk_ids_are_unique(
        self, fixed_chunker: FixedSizeChunker, sample_text: str
    ) -> None:
        chunks = fixed_chunker.chunk(sample_text)
        ids = [c.id for c in chunks]
        assert len(ids) == len(set(ids))


class TestSemanticChunker:
    def test_empty_text(self, semantic_chunker: SemanticChunker) -> None:
        assert semantic_chunker.chunk("") == []

    def test_preserves_section_titles(
        self, semantic_chunker: SemanticChunker, sample_text: str
    ) -> None:
        chunks = semantic_chunker.chunk(sample_text, document_id="d1")
        sections = {c.metadata.section_title for c in chunks if c.metadata.section_title}
        # Should capture at least some section headings
        assert len(sections) > 0

    def test_chunks_have_content(
        self, semantic_chunker: SemanticChunker, sample_text: str
    ) -> None:
        chunks = semantic_chunker.chunk(sample_text)
        for c in chunks:
            assert len(c.text.strip()) > 0


class TestHierarchicalChunker:
    def test_creates_multiple_levels(
        self, hierarchical_chunker: HierarchicalChunker, sample_text: str
    ) -> None:
        chunks = hierarchical_chunker.chunk(sample_text, document_id="d1")
        levels = {c.metadata.level for c in chunks}
        # Should have at least document and paragraph levels
        assert "document" in levels
        assert "paragraph" in levels or "section" in levels

    def test_parent_references(
        self, hierarchical_chunker: HierarchicalChunker, sample_text: str
    ) -> None:
        chunks = hierarchical_chunker.chunk(sample_text, document_id="d1")
        paragraph_chunks = [c for c in chunks if c.metadata.level == "paragraph"]
        # Paragraph chunks should reference a parent
        for pc in paragraph_chunks:
            assert pc.metadata.parent_chunk_id is not None

    def test_document_level_chunk_exists(
        self, hierarchical_chunker: HierarchicalChunker, sample_text: str
    ) -> None:
        chunks = hierarchical_chunker.chunk(sample_text)
        doc_chunks = [c for c in chunks if c.metadata.level == "document"]
        assert len(doc_chunks) == 1


class TestRecursiveChunker:
    def test_empty_text(self, recursive_chunker: RecursiveChunker) -> None:
        assert recursive_chunker.chunk("") == []

    def test_produces_chunks(
        self, recursive_chunker: RecursiveChunker, sample_text: str
    ) -> None:
        chunks = recursive_chunker.chunk(sample_text, document_id="d1")
        assert len(chunks) > 1

    def test_chunk_indices_sequential(
        self, recursive_chunker: RecursiveChunker, sample_text: str
    ) -> None:
        chunks = recursive_chunker.chunk(sample_text)
        indices = [c.metadata.chunk_index for c in chunks]
        assert indices == list(range(len(chunks)))


class TestChunkerFactory:
    def test_get_fixed(self) -> None:
        c = get_chunker("fixed")
        assert isinstance(c, FixedSizeChunker)

    def test_get_semantic(self) -> None:
        c = get_chunker("semantic")
        assert isinstance(c, SemanticChunker)

    def test_get_hierarchical(self) -> None:
        c = get_chunker("hierarchical")
        assert isinstance(c, HierarchicalChunker)

    def test_get_recursive(self) -> None:
        c = get_chunker("recursive")
        assert isinstance(c, RecursiveChunker)

    def test_invalid_strategy(self) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            get_chunker("nonexistent")


# =====================================================================
# Embedding tests
# =====================================================================


class TestLocalEmbedder:
    @pytest.mark.asyncio
    async def test_embed_single(self, local_embedder: LocalEmbedder) -> None:
        vec = await local_embedder.embed_text("hello world")
        assert len(vec) == 128
        assert all(isinstance(v, float) for v in vec)

    @pytest.mark.asyncio
    async def test_embed_batch(self, local_embedder: LocalEmbedder) -> None:
        texts = ["hello", "world", "test"]
        vecs = await local_embedder.embed_texts(texts)
        assert len(vecs) == 3
        for v in vecs:
            assert len(v) == 128

    @pytest.mark.asyncio
    async def test_deterministic(self, local_embedder: LocalEmbedder) -> None:
        v1 = await local_embedder.embed_text("same text")
        v2 = await local_embedder.embed_text("same text")
        assert v1 == v2

    @pytest.mark.asyncio
    async def test_different_texts_different_vectors(
        self, local_embedder: LocalEmbedder
    ) -> None:
        v1 = await local_embedder.embed_text("text one")
        v2 = await local_embedder.embed_text("text two")
        assert v1 != v2


# =====================================================================
# Context engineering tests
# =====================================================================


class TestContextRanker:
    def test_rank_by_score(self, sample_chunks: list[RetrievedChunk]) -> None:
        ranker = ContextRanker()
        scored = ranker.rank(sample_chunks)
        assert len(scored) == len(sample_chunks)
        # Highest scoring chunk should be first
        assert scored[0].chunk.id == "chunk_001"

    def test_empty_input(self) -> None:
        ranker = ContextRanker()
        assert ranker.rank([]) == []


class TestContextCompressor:
    def test_deduplicate(self, sample_chunks: list[RetrievedChunk]) -> None:
        # Add a near-duplicate
        dup = RetrievedChunk(
            id="chunk_dup",
            text=sample_chunks[0].text,  # Exact same text
            score=0.5,
            document_id="doc_001",
        )
        compressor = ContextCompressor(similarity_threshold=0.8)
        result = compressor.compress([sample_chunks[0], dup])
        assert len(result) == 1

    def test_truncation(self) -> None:
        long_chunk = RetrievedChunk(
            id="long",
            text="word " * 500,
            score=1.0,
        )
        compressor = ContextCompressor(max_passage_tokens=50)
        result = compressor.compress([long_chunk])
        assert len(result) == 1
        assert estimate_tokens(result[0].text) <= 60  # Allow some tolerance

    def test_empty_input(self) -> None:
        compressor = ContextCompressor()
        assert compressor.compress([]) == []


class TestContextBuilder:
    def test_build_basic(
        self,
        context_builder: ContextBuilder,
        sample_chunks: list[RetrievedChunk],
    ) -> None:
        context = context_builder.build(sample_chunks)
        assert "[Passage 1]" in context
        assert len(context) > 0

    def test_build_empty(self, context_builder: ContextBuilder) -> None:
        context = context_builder.build([])
        assert "No relevant context" in context

    def test_token_budget(self, sample_chunks: list[RetrievedChunk]) -> None:
        builder = ContextBuilder(max_tokens=50, enable_compression=False)
        context = builder.build(sample_chunks)
        # Should fit roughly within budget
        tokens = estimate_tokens(context)
        assert tokens < 100  # Some tolerance for headers

    def test_build_detailed(
        self,
        context_builder: ContextBuilder,
        sample_chunks: list[RetrievedChunk],
    ) -> None:
        result = context_builder.build_detailed(sample_chunks)
        assert result.num_passages > 0
        assert result.total_tokens > 0
        assert len(result.text) > 0


class TestTokenEstimation:
    def test_basic(self) -> None:
        assert estimate_tokens("hello world") >= 1

    def test_empty(self) -> None:
        assert estimate_tokens("") == 1  # min 1


# =====================================================================
# Memory tests
# =====================================================================


class TestConversationMemory:
    @pytest.mark.asyncio
    async def test_add_and_get(self) -> None:
        mem = ConversationMemory()
        await mem.add_turn("conv1", "user", "Hello")
        await mem.add_turn("conv1", "assistant", "Hi there")
        history = await mem.get_history("conv1")
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[1]["role"] == "assistant"

    @pytest.mark.asyncio
    async def test_max_turns(self) -> None:
        mem = ConversationMemory(max_turns=3)
        for i in range(5):
            await mem.add_turn("conv1", "user", f"msg {i}")
        history = await mem.get_history("conv1")
        assert len(history) == 3

    @pytest.mark.asyncio
    async def test_clear(self) -> None:
        mem = ConversationMemory()
        await mem.add_turn("conv1", "user", "test")
        await mem.clear("conv1")
        history = await mem.get_history("conv1")
        assert len(history) == 0

    @pytest.mark.asyncio
    async def test_multiple_conversations(self) -> None:
        mem = ConversationMemory()
        await mem.add_turn("conv1", "user", "Hello from conv1")
        await mem.add_turn("conv2", "user", "Hello from conv2")
        h1 = await mem.get_history("conv1")
        h2 = await mem.get_history("conv2")
        assert len(h1) == 1
        assert len(h2) == 1
        assert h1[0]["content"] != h2[0]["content"]


class TestEpisodicMemory:
    @pytest.mark.asyncio
    async def test_store_and_find(self) -> None:
        mem = EpisodicMemory()
        await mem.store("What is machine learning?", "ML is a subset of AI.")
        results = await mem.find_similar("machine learning definition", threshold=0.3)
        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_exact_match(self) -> None:
        mem = EpisodicMemory()
        await mem.store("test query", "test answer")
        ep = await mem.get_exact("test query")
        assert ep is not None
        assert ep.answer == "test answer"

    @pytest.mark.asyncio
    async def test_max_episodes(self) -> None:
        mem = EpisodicMemory(max_episodes=3)
        for i in range(5):
            await mem.store(f"query {i}", f"answer {i}")
        assert mem.size == 3


class TestSemanticMemory:
    @pytest.mark.asyncio
    async def test_store_and_recall(self) -> None:
        mem = SemanticMemory()
        await mem.store("python version", "3.11")
        facts = await mem.recall("python")
        assert len(facts) >= 1
        assert facts[0].value == "3.11"

    @pytest.mark.asyncio
    async def test_update_higher_confidence(self) -> None:
        mem = SemanticMemory()
        await mem.store("lang", "Python 3.10", confidence=0.5)
        await mem.store("lang", "Python 3.11", confidence=0.9)
        fact = await mem.get("lang")
        assert fact is not None
        assert fact.value == "Python 3.11"

    @pytest.mark.asyncio
    async def test_delete(self) -> None:
        mem = SemanticMemory()
        await mem.store("key1", "val1")
        deleted = await mem.delete("key1")
        assert deleted is True
        assert await mem.get("key1") is None


class TestMemoryManager:
    @pytest.mark.asyncio
    async def test_augment_query(self, memory_manager: MemoryManager) -> None:
        await memory_manager.conversation.add_turn("c1", "user", "hello")
        await memory_manager.episodic.store("hello world", "greeting response")
        await memory_manager.semantic.store("language", "Python")

        result = await memory_manager.augment_query(
            "hello Python", conversation_id="c1"
        )
        assert "history" in result
        assert "similar_episodes" in result
        assert "relevant_facts" in result
        assert len(result["history"]) >= 1

    @pytest.mark.asyncio
    async def test_stats(self, memory_manager: MemoryManager) -> None:
        await memory_manager.conversation.add_turn("c1", "user", "hi")
        await memory_manager.episodic.store("q", "a")
        await memory_manager.semantic.store("k", "v")

        stats = await memory_manager.get_stats()
        assert stats["active_conversations"] >= 1
        assert stats["stored_episodes"] >= 1
        assert stats["stored_facts"] >= 1


# =====================================================================
# Mock retriever tests
# =====================================================================


class TestMockRetriever:
    @pytest.mark.asyncio
    async def test_retrieve(self, mock_retriever) -> None:
        result = await mock_retriever.retrieve("machine learning")
        assert len(result.chunks) > 0
        assert result.strategy == "mock"

    @pytest.mark.asyncio
    async def test_top_k(self, mock_retriever) -> None:
        result = await mock_retriever.retrieve("learning", top_k=2)
        assert len(result.chunks) <= 2


class TestMockReranker:
    @pytest.mark.asyncio
    async def test_rerank(self, mock_reranker, sample_chunks) -> None:
        results = await mock_reranker.rerank("test query", sample_chunks, top_k=3)
        assert len(results) == 3
        assert results[0].rank == 1
        assert results[0].relevance_score > results[1].relevance_score
