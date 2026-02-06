"""Hierarchical RAG: A-RAG pattern with adaptive retrieval and multi-granularity search.

Architecture:
    User Query -> Manager Agent (orchestrator)
                      |
          +-----------+-----------+
          |           |           |
    keyword_search  semantic_search  chunk_read
          |           |           |
          +-----------+-----------+
                      |
              Sub-Agent: Synthesizer
                      |
              Sub-Agent: Validator
                      |
                 Final Answer

This implements the A-RAG (Adaptive RAG) pattern from recent research:

1. Manager Agent:  Analyzes query complexity and selects retrieval strategy.
   Acts as a meta-agent / planner that decides WHICH tools to use and HOW.

2. Three retrieval tools:
   - keyword_search: BM25-style search for specific terms
   - semantic_search: vector similarity for conceptual matching
   - chunk_read: read a specific chunk or its parent for more context

3. Multi-granularity search: document, section, and paragraph levels.
   The manager can search at different levels depending on query type:
   - Broad questions -> section/document level
   - Specific questions -> paragraph level
   - Follow-up questions -> read parent chunk for context

4. Adaptive strategy: The manager decides based on query analysis:
   - Simple factual -> semantic search, paragraph level
   - Multi-part complex -> decompose + multiple searches
   - Keyword-heavy (names, codes) -> keyword search
   - Ambiguous -> hybrid search + section level

Tradeoffs vs single-agent:
+ Adaptive strategy selection based on query complexity
+ Multi-granularity retrieval for better context
+ More robust for diverse query types
- Higher latency and cost (planning + multiple searches)
- More complex system to maintain

Tradeoffs vs multi-agent:
+ Dynamic tool selection (not a fixed pipeline)
+ Can combine different retrieval strategies per query
+ Manager can read intermediate results and adapt
- Even higher latency (planning step + potential re-retrieval)
- Requires more sophisticated orchestration
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from agent_rag.config import LLMProvider, settings
from agent_rag.context_engineering import ContextBuilder
from agent_rag.memory import MemoryManager
from agent_rag.retrieval.retriever import (
    BaseRetriever,
    HybridRetriever,
    KeywordRetriever,
    RetrievedChunk,
    SemanticRetriever,
    get_retriever,
)

logger = structlog.get_logger(__name__)


class QueryComplexity(str, Enum):
    SIMPLE = "simple"           # Single factual question
    MODERATE = "moderate"       # Needs some reasoning
    COMPLEX = "complex"         # Multi-part, needs decomposition
    KEYWORD_HEAVY = "keyword"   # Names, codes, specific terms


class GranularityLevel(str, Enum):
    DOCUMENT = "document"
    SECTION = "section"
    PARAGRAPH = "paragraph"


@dataclass
class RetrievalPlan:
    """Plan created by the manager agent for retrieval."""

    complexity: QueryComplexity
    strategies: list[str]         # e.g., ["semantic", "keyword"]
    granularity: GranularityLevel
    sub_queries: list[str]
    reasoning: str


@dataclass
class ToolCall:
    """Record of a tool invocation by the manager agent."""

    tool: str
    input_args: dict[str, Any]
    output_summary: str
    chunks_returned: int


@dataclass
class HierarchicalResponse:
    """Response from the hierarchical RAG system."""

    answer: str
    query: str
    plan: RetrievalPlan
    retrieved_chunks: list[RetrievedChunk]
    tool_calls: list[ToolCall] = field(default_factory=list)
    validation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_type: str = "hierarchical"


# ---------------------------------------------------------------------------
# LLM helper
# ---------------------------------------------------------------------------

async def _call_llm(system: str, user: str, temperature: float = 0.1) -> str:
    if settings.llm_provider == LLMProvider.ANTHROPIC:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
        resp = await client.messages.create(
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
            temperature=temperature,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return resp.content[0].text
    else:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=settings.openai_api_key.get_secret_value())
        resp = await client.chat.completions.create(
            model=settings.openai_model,
            max_tokens=settings.llm_max_tokens,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Manager Agent (planner / orchestrator)
# ---------------------------------------------------------------------------

_PLANNER_SYSTEM = """You are a retrieval planning agent. Given a user question, analyze it and create a retrieval plan.

Available tools:
1. keyword_search - BM25-style keyword matching. Best for specific terms, names, codes, abbreviations.
2. semantic_search - Vector similarity search. Best for conceptual/meaning-based queries.
3. chunk_read - Read a specific chunk's parent section for broader context.

Available granularity levels:
- "document" - high-level document summaries
- "section" - section-level chunks
- "paragraph" - fine-grained paragraph chunks

Output a JSON object with:
{
  "complexity": "simple|moderate|complex|keyword",
  "strategies": ["semantic", "keyword"],
  "granularity": "paragraph|section|document",
  "sub_queries": ["sub query 1", "sub query 2"],
  "reasoning": "Why this plan was chosen"
}

Guidelines:
- Simple factual questions: ["semantic"], "paragraph"
- Questions with specific names/codes: ["keyword", "semantic"], "paragraph"
- Broad conceptual questions: ["semantic"], "section"
- Multi-part complex questions: decompose into sub_queries, ["semantic", "keyword"], "paragraph"
- If the query is already simple, sub_queries should contain just the original query.
"""

_SYNTHESIZER_SYSTEM = """You are a synthesis agent. Given a question and retrieved passages at multiple granularity levels, synthesize a comprehensive answer.

Rules:
1. Use ONLY information from the provided passages.
2. Cite passages using [1], [2], etc.
3. If passages from different granularity levels provide different detail, prefer the most specific.
4. Structure your answer logically.
5. Acknowledge any gaps in the available information.
"""

_VALIDATOR_SYSTEM = """You are a validation agent. Check the synthesized answer against the source passages.

Evaluate:
1. Are all claims in the answer supported by the passages?
2. Are there any hallucinated facts?
3. Is the answer complete for what the passages support?

Output:
VALID: [YES or NO]
ISSUES: [list any problems found]
CONFIDENCE: [HIGH, MEDIUM, or LOW]
"""


class ManagerAgent:
    """Meta-agent that plans the retrieval strategy.

    Analyzes query complexity and decides:
    - Which retrieval tools to use
    - At what granularity level to search
    - Whether to decompose into sub-queries
    """

    async def plan(self, question: str) -> RetrievalPlan:
        """Create a retrieval plan for the given question."""
        try:
            raw = await _call_llm(_PLANNER_SYSTEM, question)
            # Extract JSON from response
            raw = raw.strip()
            # Handle code blocks
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0].strip()
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0].strip()

            data = json.loads(raw)
            return RetrievalPlan(
                complexity=QueryComplexity(data.get("complexity", "simple")),
                strategies=data.get("strategies", ["semantic"]),
                granularity=GranularityLevel(data.get("granularity", "paragraph")),
                sub_queries=data.get("sub_queries", [question]),
                reasoning=data.get("reasoning", ""),
            )
        except Exception:
            logger.warning("planning_failed", exc_info=True)
            return RetrievalPlan(
                complexity=QueryComplexity.SIMPLE,
                strategies=["semantic"],
                granularity=GranularityLevel.PARAGRAPH,
                sub_queries=[question],
                reasoning="Fallback to default plan due to planning error",
            )


# ---------------------------------------------------------------------------
# Retrieval Tools
# ---------------------------------------------------------------------------

class RetrievalToolkit:
    """Collection of retrieval tools available to the manager agent.

    Three tools with different strengths:
    - keyword_search:  exact term matching (BM25)
    - semantic_search: meaning-based similarity
    - chunk_read:      read parent chunk for broader context
    """

    def __init__(
        self,
        keyword_retriever: KeywordRetriever | None = None,
        semantic_retriever: SemanticRetriever | None = None,
        hybrid_retriever: HybridRetriever | None = None,
    ) -> None:
        self._keyword = keyword_retriever or KeywordRetriever()
        self._semantic = semantic_retriever or SemanticRetriever()
        self._hybrid = hybrid_retriever or HybridRetriever()

    async def keyword_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        granularity: GranularityLevel = GranularityLevel.PARAGRAPH,
    ) -> list[RetrievedChunk]:
        """BM25-style keyword search."""
        filters: dict[str, Any] = {}
        if granularity != GranularityLevel.PARAGRAPH:
            filters["level"] = granularity.value

        result = await self._keyword.retrieve(query, top_k=top_k, filters=filters or None)
        return result.chunks

    async def semantic_search(
        self,
        query: str,
        *,
        top_k: int = 10,
        granularity: GranularityLevel = GranularityLevel.PARAGRAPH,
    ) -> list[RetrievedChunk]:
        """Vector similarity search."""
        filters: dict[str, Any] = {}
        if granularity != GranularityLevel.PARAGRAPH:
            filters["level"] = granularity.value

        result = await self._semantic.retrieve(query, top_k=top_k, filters=filters or None)
        return result.chunks

    async def chunk_read(
        self,
        chunk_id: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        """Read a chunk's parent to get broader context.

        Looks for sibling chunks (same parent_chunk_id) in the retrieved set.
        In a full implementation, this would query Qdrant by parent_chunk_id.
        """
        target = None
        for c in chunks:
            if c.id == chunk_id:
                target = c
                break

        if not target or not target.parent_chunk_id:
            return []

        # Find siblings
        parent_id = target.parent_chunk_id
        siblings = [
            c for c in chunks
            if c.parent_chunk_id == parent_id or c.id == parent_id
        ]
        return siblings


# ---------------------------------------------------------------------------
# Synthesizer Agent
# ---------------------------------------------------------------------------

class SynthesizerAgent:
    """Synthesize an answer from multi-granularity retrieved passages."""

    def __init__(self, context_builder: ContextBuilder | None = None) -> None:
        self._context_builder = context_builder or ContextBuilder()

    async def synthesize(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        plan: RetrievalPlan,
    ) -> str:
        """Generate a comprehensive answer from retrieved chunks."""
        # Group by granularity level for structured context
        by_level: dict[str, list[RetrievedChunk]] = {
            "document": [],
            "section": [],
            "paragraph": [],
        }
        for c in chunks:
            level = c.level if c.level in by_level else "paragraph"
            by_level[level].append(c)

        # Build structured context
        context_parts: list[str] = []
        passage_num = 1
        for level in ["document", "section", "paragraph"]:
            level_chunks = by_level[level]
            if level_chunks:
                context_parts.append(f"\n--- {level.upper()} LEVEL ---")
                for c in level_chunks[:5]:
                    context_parts.append(
                        f"[{passage_num}] ({level}, "
                        f"section: {c.section_title or 'N/A'}): "
                        f"{c.text[:800]}"
                    )
                    passage_num += 1

        context_text = "\n\n".join(context_parts)

        user_prompt = (
            f"Question: {question}\n\n"
            f"Retrieval plan: {plan.reasoning}\n\n"
            f"Retrieved passages:\n{context_text}"
        )

        return await _call_llm(_SYNTHESIZER_SYSTEM, user_prompt, temperature=0.3)


# ---------------------------------------------------------------------------
# Validator Agent
# ---------------------------------------------------------------------------

class ValidatorAgent:
    """Validate the synthesized answer against source passages."""

    async def validate(
        self,
        question: str,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        """Validate the answer for faithfulness."""
        passages = "\n\n".join(
            f"[{i + 1}]: {c.text[:400]}" for i, c in enumerate(chunks[:10])
        )
        user_prompt = (
            f"Question: {question}\n\n"
            f"Answer: {answer}\n\n"
            f"Source passages:\n{passages}"
        )
        return await _call_llm(_VALIDATOR_SYSTEM, user_prompt, temperature=0.1)


# ---------------------------------------------------------------------------
# Hierarchical RAG orchestrator
# ---------------------------------------------------------------------------

class HierarchicalRAG:
    """Hierarchical RAG with adaptive retrieval (A-RAG pattern).

    The manager agent analyzes query complexity, creates a retrieval plan,
    then executes it using the appropriate tools at the right granularity.
    Results are synthesized and validated before returning.

    This is the most sophisticated RAG pattern -- best for:
    - Diverse query types (factual, conceptual, multi-part)
    - Large, heterogeneous document collections
    - When retrieval quality matters more than latency
    """

    def __init__(
        self,
        *,
        manager: ManagerAgent | None = None,
        toolkit: RetrievalToolkit | None = None,
        synthesizer: SynthesizerAgent | None = None,
        validator: ValidatorAgent | None = None,
        memory_manager: MemoryManager | None = None,
    ) -> None:
        self._manager = manager or ManagerAgent()
        self._toolkit = toolkit or RetrievalToolkit()
        self._synthesizer = synthesizer or SynthesizerAgent()
        self._validator = validator or ValidatorAgent()
        self._memory = memory_manager

    async def query(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        skip_validation: bool = False,
    ) -> HierarchicalResponse:
        """Execute the hierarchical RAG pipeline.

        Args:
            question: User's question.
            conversation_id: Optional conversation ID for memory.
            top_k: Number of chunks per retrieval call.
            filters: Metadata filters.
            skip_validation: Skip the validation step.

        Returns:
            HierarchicalResponse with answer, plan, and full trace.
        """
        top_k = top_k or settings.retrieval_top_k
        tool_calls: list[ToolCall] = []

        # Step 1: Plan
        plan = await self._manager.plan(question)
        logger.info(
            "retrieval_planned",
            complexity=plan.complexity.value,
            strategies=plan.strategies,
            granularity=plan.granularity.value,
            sub_queries=len(plan.sub_queries),
        )

        # Step 2: Execute retrieval using planned strategy
        all_chunks: list[RetrievedChunk] = []
        seen_ids: set[str] = set()

        for sub_query in plan.sub_queries:
            for strategy in plan.strategies:
                chunks = await self._execute_search(
                    strategy=strategy,
                    query=sub_query,
                    top_k=top_k,
                    granularity=plan.granularity,
                )
                tool_calls.append(ToolCall(
                    tool=f"{strategy}_search",
                    input_args={
                        "query": sub_query,
                        "granularity": plan.granularity.value,
                    },
                    output_summary=f"Retrieved {len(chunks)} chunks",
                    chunks_returned=len(chunks),
                ))

                for c in chunks:
                    if c.id not in seen_ids:
                        seen_ids.add(c.id)
                        all_chunks.append(c)

        # Step 3: For complex queries, also search at section level
        if plan.complexity in (QueryComplexity.COMPLEX, QueryComplexity.MODERATE):
            if plan.granularity == GranularityLevel.PARAGRAPH:
                section_chunks = await self._execute_search(
                    strategy="semantic",
                    query=question,
                    top_k=5,
                    granularity=GranularityLevel.SECTION,
                )
                tool_calls.append(ToolCall(
                    tool="semantic_search",
                    input_args={
                        "query": question,
                        "granularity": "section",
                    },
                    output_summary=f"Retrieved {len(section_chunks)} section chunks",
                    chunks_returned=len(section_chunks),
                ))
                for c in section_chunks:
                    if c.id not in seen_ids:
                        seen_ids.add(c.id)
                        all_chunks.append(c)

        if not all_chunks:
            return HierarchicalResponse(
                answer="I could not find any relevant information to answer your question.",
                query=question,
                plan=plan,
                retrieved_chunks=[],
                tool_calls=tool_calls,
                agent_type="hierarchical",
            )

        # Step 4: Synthesize
        answer = await self._synthesizer.synthesize(question, all_chunks, plan)

        # Step 5: Validate
        validation = ""
        if not skip_validation:
            validation = await self._validator.validate(question, answer, all_chunks)

            # If validation fails, try re-synthesis with broader context
            if "VALID: NO" in validation.upper():
                logger.info("validation_failed_retrying", question=question)
                # Broaden: search at section level
                broader_chunks = await self._execute_search(
                    strategy="semantic",
                    query=question,
                    top_k=10,
                    granularity=GranularityLevel.SECTION,
                )
                for c in broader_chunks:
                    if c.id not in seen_ids:
                        seen_ids.add(c.id)
                        all_chunks.append(c)

                tool_calls.append(ToolCall(
                    tool="semantic_search",
                    input_args={"query": question, "granularity": "section"},
                    output_summary=f"Broader search: {len(broader_chunks)} chunks",
                    chunks_returned=len(broader_chunks),
                ))

                answer = await self._synthesizer.synthesize(question, all_chunks, plan)
                validation = await self._validator.validate(
                    question, answer, all_chunks
                )

        # Store in memory
        if self._memory and conversation_id:
            await self._memory.conversation.add_turn(
                conversation_id, "user", question
            )
            await self._memory.conversation.add_turn(
                conversation_id, "assistant", answer
            )
            # Store in episodic memory for future similar queries
            await self._memory.episodic.store(
                question, answer, plan.complexity.value
            )

        return HierarchicalResponse(
            answer=answer,
            query=question,
            plan=plan,
            retrieved_chunks=all_chunks,
            tool_calls=tool_calls,
            validation=validation,
            metadata={
                "strategy": "hierarchical",
                "complexity": plan.complexity.value,
                "tools_used": len(tool_calls),
                "total_chunks": len(all_chunks),
            },
            agent_type="hierarchical",
        )

    async def _execute_search(
        self,
        strategy: str,
        query: str,
        top_k: int,
        granularity: GranularityLevel,
    ) -> list[RetrievedChunk]:
        """Execute a search using the specified strategy and granularity."""
        if strategy == "keyword":
            return await self._toolkit.keyword_search(
                query, top_k=top_k, granularity=granularity
            )
        elif strategy == "semantic":
            return await self._toolkit.semantic_search(
                query, top_k=top_k, granularity=granularity
            )
        else:
            # Default to semantic
            return await self._toolkit.semantic_search(
                query, top_k=top_k, granularity=granularity
            )
