"""Single-Agent RAG: the simplest retrieve-and-generate pattern.

Architecture:
    User Query -> Query Processing -> Retrieve -> Rerank -> Generate Answer

This is the baseline RAG pattern.  A single agent handles the full pipeline:
1. Process the user query (optional rewriting)
2. Retrieve relevant chunks from the vector store
3. Optionally rerank for precision
4. Generate a grounded answer using the retrieved context

Tradeoffs vs multi-agent / hierarchical:
+ Simple, fast, easy to debug
+ Low latency (single LLM call for generation)
+ Sufficient for straightforward factual queries
- No iterative refinement or self-critique
- Cannot adaptively choose retrieval strategy
- Struggles with complex multi-part questions
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, AsyncIterator

import structlog

from agent_rag.config import LLMProvider, settings
from agent_rag.context_engineering import ContextBuilder
from agent_rag.memory import ConversationMemory, MemoryManager
from agent_rag.retrieval.query_processor import ProcessedQuery, QueryProcessor, RewriteStrategy
from agent_rag.retrieval.reranker import BaseReranker, RerankResult, get_reranker
from agent_rag.retrieval.retriever import BaseRetriever, RetrievedChunk, get_retriever

logger = structlog.get_logger(__name__)


_SYSTEM_PROMPT = """You are a knowledgeable assistant that answers questions based on the provided context.

Rules:
1. Base your answer ONLY on the provided context passages.
2. If the context does not contain enough information, say so explicitly.
3. Cite the relevant passage numbers in your answer using [1], [2], etc.
4. Be concise but thorough.
5. If multiple passages provide different information, synthesize them.
"""


@dataclass
class RAGResponse:
    """Response from a RAG agent."""

    answer: str
    query: str
    retrieved_chunks: list[RetrievedChunk]
    reranked: list[RerankResult] | None = None
    processed_query: ProcessedQuery | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_type: str = "single"


class SingleAgentRAG:
    """Single-agent RAG: retrieve-and-generate in one pass.

    The simplest and fastest RAG pattern.  Best for:
    - Simple factual queries
    - Low-latency requirements
    - Well-structured, homogeneous document collections
    """

    def __init__(
        self,
        *,
        retriever: BaseRetriever | None = None,
        reranker: BaseReranker | None = None,
        query_processor: QueryProcessor | None = None,
        context_builder: ContextBuilder | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        self._retriever = retriever or get_retriever()
        self._reranker = reranker
        self._query_processor = query_processor or QueryProcessor()
        self._context_builder = context_builder or ContextBuilder()
        self._memory = memory_manager

    async def query(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        rewrite_strategy: RewriteStrategy = RewriteStrategy.NONE,
        use_reranker: bool = False,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> RAGResponse:
        """Execute the single-agent RAG pipeline.

        Args:
            question: User's question.
            conversation_id: Optional ID for conversation memory.
            rewrite_strategy: Query rewriting strategy.
            use_reranker: Whether to apply reranking.
            top_k: Number of chunks to retrieve.
            filters: Metadata filters for retrieval.

        Returns:
            RAGResponse with the generated answer and supporting evidence.
        """
        top_k = top_k or settings.retrieval_top_k

        # 1. Load conversation history for context
        history: list[dict[str, str]] = []
        if self._memory and conversation_id:
            conv_mem = self._memory.conversation
            history = await conv_mem.get_history(conversation_id)

        # 2. Process query
        processed = await self._query_processor.process(
            question,
            decompose=False,
            rewrite_strategy=rewrite_strategy,
            expand=False,
        )

        # 3. Retrieve
        queries = processed.effective_queries
        all_chunks: list[RetrievedChunk] = []
        seen_ids: set[str] = set()

        for q in queries:
            result = await self._retriever.retrieve(q, top_k=top_k, filters=filters)
            for chunk in result.chunks:
                if chunk.id not in seen_ids:
                    seen_ids.add(chunk.id)
                    all_chunks.append(chunk)

        # 4. Rerank (optional)
        reranked: list[RerankResult] | None = None
        if use_reranker and self._reranker and all_chunks:
            reranked = await self._reranker.rerank(
                question, all_chunks, top_k=settings.reranker_top_k
            )
            # Use reranked order for context
            context_chunks = [r.chunk for r in reranked]
        else:
            context_chunks = all_chunks[: settings.reranker_top_k]

        # 5. Build context
        context_text = self._context_builder.build(
            chunks=context_chunks,
            max_tokens=settings.context_max_tokens,
        )

        # 6. Generate answer
        answer = await self._generate(question, context_text, history)

        # 7. Store in conversation memory
        if self._memory and conversation_id:
            await self._memory.conversation.add_turn(
                conversation_id, "user", question
            )
            await self._memory.conversation.add_turn(
                conversation_id, "assistant", answer
            )

        return RAGResponse(
            answer=answer,
            query=question,
            retrieved_chunks=all_chunks,
            reranked=reranked,
            processed_query=processed,
            metadata={"strategy": "single_agent"},
            agent_type="single",
        )

    async def query_stream(
        self,
        question: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> AsyncIterator[str]:
        """Stream the generated answer token by token."""
        top_k = top_k or settings.retrieval_top_k

        # Retrieve
        result = await self._retriever.retrieve(question, top_k=top_k, filters=filters)
        context_chunks = result.chunks[: settings.reranker_top_k]
        context_text = self._context_builder.build(
            chunks=context_chunks,
            max_tokens=settings.context_max_tokens,
        )

        # Stream generation
        async for token in self._generate_stream(question, context_text):
            yield token

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _generate(
        question: str,
        context: str,
        history: list[dict[str, str]] | None = None,
    ) -> str:
        """Generate an answer using the configured LLM."""
        user_content = f"Context:\n{context}\n\nQuestion: {question}"

        if settings.llm_provider == LLMProvider.ANTHROPIC:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(
                api_key=settings.anthropic_api_key.get_secret_value()
            )
            messages: list[dict[str, str]] = []
            if history:
                messages.extend(history[-6:])  # Last 3 turns
            messages.append({"role": "user", "content": user_content})

            response = await client.messages.create(
                model=settings.anthropic_model,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                system=_SYSTEM_PROMPT,
                messages=messages,
            )
            return response.content[0].text
        else:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=settings.openai_api_key.get_secret_value()
            )
            oai_messages: list[dict[str, str]] = [
                {"role": "system", "content": _SYSTEM_PROMPT}
            ]
            if history:
                oai_messages.extend(history[-6:])
            oai_messages.append({"role": "user", "content": user_content})

            response = await client.chat.completions.create(
                model=settings.openai_model,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                messages=oai_messages,
            )
            return response.choices[0].message.content or ""

    @staticmethod
    async def _generate_stream(
        question: str,
        context: str,
    ) -> AsyncIterator[str]:
        """Stream tokens from the LLM."""
        user_content = f"Context:\n{context}\n\nQuestion: {question}"

        if settings.llm_provider == LLMProvider.ANTHROPIC:
            from anthropic import AsyncAnthropic

            client = AsyncAnthropic(
                api_key=settings.anthropic_api_key.get_secret_value()
            )
            async with client.messages.stream(
                model=settings.anthropic_model,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_content}],
            ) as stream:
                async for text in stream.text_stream:
                    yield text
        else:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(
                api_key=settings.openai_api_key.get_secret_value()
            )
            stream = await client.chat.completions.create(
                model=settings.openai_model,
                max_tokens=settings.llm_max_tokens,
                temperature=settings.llm_temperature,
                stream=True,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
            )
            async for chunk in stream:
                delta = chunk.choices[0].delta
                if delta.content:
                    yield delta.content
