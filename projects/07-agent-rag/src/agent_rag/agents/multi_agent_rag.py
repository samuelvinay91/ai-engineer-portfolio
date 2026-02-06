"""Multi-Agent RAG: specialized agents collaborating in a pipeline.

Architecture:
    User Query -> Planner -> Retriever Agent -> Analyzer Agent
                                                    |
                         Critic Agent <-- Generator Agent
                              |
                        Final Answer (or feedback loop)

Four specialized agents:
1. Retriever Agent:  handles query processing and document retrieval
2. Analyzer Agent:   analyzes retrieved passages for relevance and coverage
3. Generator Agent:  synthesizes the answer from analyzed passages
4. Critic Agent:     evaluates the generated answer for quality/faithfulness

Tradeoffs vs single-agent:
+ Self-critique catches hallucinations and gaps
+ Each agent can be tuned independently (different prompts, temperatures)
+ Explicit feedback loop for iterative refinement
- Higher latency (multiple sequential LLM calls)
- More complex to debug and maintain
- Higher cost per query

Tradeoffs vs hierarchical:
+ Simpler agent topology (linear pipeline)
+ Easier to understand and modify
- Less adaptive retrieval strategy
- No dynamic tool selection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from agent_rag.config import LLMProvider, settings
from agent_rag.context_engineering import ContextBuilder
from agent_rag.memory import MemoryManager
from agent_rag.retrieval.query_processor import ProcessedQuery, QueryProcessor, RewriteStrategy
from agent_rag.retrieval.reranker import BaseReranker, RerankResult
from agent_rag.retrieval.retriever import BaseRetriever, RetrievedChunk, get_retriever

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Agent prompts
# ---------------------------------------------------------------------------

_ANALYZER_SYSTEM = """You are a passage analysis agent.
Given a user question and a set of retrieved passages, your job is to:
1. Identify which passages are relevant to the question.
2. Note any gaps -- information the question asks for that is NOT covered.
3. Highlight key facts and relationships from relevant passages.

Output a structured analysis in this format:
RELEVANT_PASSAGES: [list of passage numbers that are relevant]
KEY_FACTS:
- fact 1
- fact 2
GAPS: [what information is missing, if any]
SUFFICIENT: [YES or NO -- is there enough context to answer?]
"""

_GENERATOR_SYSTEM = """You are an answer generation agent.
Given a question, a passage analysis, and the original passages, generate
a comprehensive, well-cited answer.

Rules:
1. Base your answer ONLY on the provided passages.
2. Use citations like [1], [2] to reference passage numbers.
3. If the analysis indicates gaps, acknowledge what you cannot answer.
4. Structure your answer clearly with key points.
5. Be accurate -- never fabricate information beyond the passages.
"""

_CRITIC_SYSTEM = """You are a quality assurance agent reviewing a generated answer.
Evaluate the answer on these criteria:
1. FAITHFULNESS: Does the answer stay true to the source passages? (0-10)
2. COMPLETENESS: Does the answer address all parts of the question? (0-10)
3. CLARITY: Is the answer well-structured and easy to understand? (0-10)
4. CITATIONS: Are citations properly used? (0-10)

Output in this format:
FAITHFULNESS: [score] - [brief justification]
COMPLETENESS: [score] - [brief justification]
CLARITY: [score] - [brief justification]
CITATIONS: [score] - [brief justification]
OVERALL: [average score]
NEEDS_REVISION: [YES or NO]
FEEDBACK: [specific suggestions for improvement, if any]
"""


@dataclass
class AgentStep:
    """Record of a single agent's execution."""

    agent_name: str
    input_summary: str
    output_summary: str
    duration_ms: float = 0.0


@dataclass
class MultiAgentResponse:
    """Response from the multi-agent RAG pipeline."""

    answer: str
    query: str
    retrieved_chunks: list[RetrievedChunk]
    analysis: str = ""
    critique: str = ""
    revision_count: int = 0
    steps: list[AgentStep] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    agent_type: str = "multi"


# ---------------------------------------------------------------------------
# LLM call helper
# ---------------------------------------------------------------------------

async def _call_llm(system: str, user: str, temperature: float | None = None) -> str:
    temp = temperature if temperature is not None else settings.llm_temperature
    if settings.llm_provider == LLMProvider.ANTHROPIC:
        from anthropic import AsyncAnthropic

        client = AsyncAnthropic(api_key=settings.anthropic_api_key.get_secret_value())
        resp = await client.messages.create(
            model=settings.anthropic_model,
            max_tokens=settings.llm_max_tokens,
            temperature=temp,
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
            temperature=temp,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return resp.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# Individual agent implementations
# ---------------------------------------------------------------------------

class RetrieverAgent:
    """Agent 1: Handles query processing and document retrieval."""

    def __init__(
        self,
        retriever: BaseRetriever | None = None,
        query_processor: QueryProcessor | None = None,
        reranker: BaseReranker | None = None,
    ) -> None:
        self._retriever = retriever or get_retriever()
        self._processor = query_processor or QueryProcessor()
        self._reranker = reranker

    async def run(
        self,
        question: str,
        *,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> tuple[list[RetrievedChunk], ProcessedQuery]:
        """Process the query and retrieve relevant chunks."""
        top_k = top_k or settings.retrieval_top_k

        processed = await self._processor.process(
            question,
            decompose=True,
            rewrite_strategy=RewriteStrategy.MULTI_QUERY,
            expand=False,
        )

        all_chunks: list[RetrievedChunk] = []
        seen: set[str] = set()

        for q in processed.effective_queries:
            result = await self._retriever.retrieve(q, top_k=top_k, filters=filters)
            for chunk in result.chunks:
                if chunk.id not in seen:
                    seen.add(chunk.id)
                    all_chunks.append(chunk)

        # Rerank if available
        if self._reranker and all_chunks:
            reranked = await self._reranker.rerank(question, all_chunks)
            all_chunks = [r.chunk for r in reranked]

        logger.info("retriever_agent_done", chunks=len(all_chunks))
        return all_chunks, processed


class AnalyzerAgent:
    """Agent 2: Analyzes retrieved passages for relevance and coverage."""

    async def run(
        self,
        question: str,
        chunks: list[RetrievedChunk],
    ) -> str:
        """Analyze passage relevance and identify gaps."""
        passages_text = "\n\n".join(
            f"[Passage {i + 1}]: {c.text[:600]}" for i, c in enumerate(chunks[:10])
        )
        user_prompt = f"Question: {question}\n\nPassages:\n{passages_text}"
        analysis = await _call_llm(_ANALYZER_SYSTEM, user_prompt, temperature=0.1)
        logger.info("analyzer_agent_done", analysis_len=len(analysis))
        return analysis


class GeneratorAgent:
    """Agent 3: Generates the answer from analyzed passages."""

    def __init__(self, context_builder: ContextBuilder | None = None) -> None:
        self._context_builder = context_builder or ContextBuilder()

    async def run(
        self,
        question: str,
        chunks: list[RetrievedChunk],
        analysis: str,
        *,
        previous_feedback: str = "",
    ) -> str:
        """Generate the answer.  Optionally incorporates critic feedback."""
        context = self._context_builder.build(
            chunks=chunks, max_tokens=settings.context_max_tokens
        )

        user_prompt = f"Question: {question}\n\nAnalysis:\n{analysis}\n\nPassages:\n{context}"
        if previous_feedback:
            user_prompt += f"\n\nPrevious feedback (please address):\n{previous_feedback}"

        answer = await _call_llm(_GENERATOR_SYSTEM, user_prompt, temperature=0.3)
        logger.info("generator_agent_done", answer_len=len(answer))
        return answer


class CriticAgent:
    """Agent 4: Evaluates answer quality and provides feedback.

    The feedback loop: if the critic finds issues, the generator can
    revise its answer -- up to a configurable maximum number of revisions.
    """

    async def run(
        self,
        question: str,
        answer: str,
        chunks: list[RetrievedChunk],
    ) -> tuple[str, bool]:
        """Evaluate the answer.

        Returns:
            Tuple of (critique_text, needs_revision).
        """
        passages_text = "\n\n".join(
            f"[Passage {i + 1}]: {c.text[:400]}" for i, c in enumerate(chunks[:10])
        )
        user_prompt = (
            f"Question: {question}\n\n"
            f"Generated Answer:\n{answer}\n\n"
            f"Source Passages:\n{passages_text}"
        )

        critique = await _call_llm(_CRITIC_SYSTEM, user_prompt, temperature=0.1)
        needs_revision = "NEEDS_REVISION: YES" in critique.upper()
        logger.info("critic_agent_done", needs_revision=needs_revision)
        return critique, needs_revision


# ---------------------------------------------------------------------------
# Multi-Agent orchestrator
# ---------------------------------------------------------------------------

class MultiAgentRAG:
    """Multi-Agent RAG with specialized agents and feedback loops.

    Pipeline: Retriever -> Analyzer -> Generator -> Critic
    If the critic requests revision, the generator reruns with feedback.
    Maximum revisions controlled by *max_revisions*.
    """

    def __init__(
        self,
        *,
        retriever_agent: RetrieverAgent | None = None,
        analyzer_agent: AnalyzerAgent | None = None,
        generator_agent: GeneratorAgent | None = None,
        critic_agent: CriticAgent | None = None,
        memory_manager: MemoryManager | None = None,
        max_revisions: int = 2,
    ) -> None:
        self._retriever = retriever_agent or RetrieverAgent()
        self._analyzer = analyzer_agent or AnalyzerAgent()
        self._generator = generator_agent or GeneratorAgent()
        self._critic = critic_agent or CriticAgent()
        self._memory = memory_manager
        self._max_revisions = max_revisions

    async def query(
        self,
        question: str,
        *,
        conversation_id: str | None = None,
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
        skip_critique: bool = False,
    ) -> MultiAgentResponse:
        """Execute the multi-agent RAG pipeline.

        Args:
            question: User's question.
            conversation_id: Optional conversation ID for memory.
            top_k: Number of chunks to retrieve.
            filters: Metadata filters for retrieval.
            skip_critique: Skip the critic agent (faster but no quality check).

        Returns:
            MultiAgentResponse with answer, analysis, critique, and step log.
        """
        steps: list[AgentStep] = []

        # Step 1: Retrieve
        chunks, processed = await self._retriever.run(
            question, top_k=top_k, filters=filters
        )
        steps.append(AgentStep(
            agent_name="retriever",
            input_summary=question,
            output_summary=f"Retrieved {len(chunks)} chunks",
        ))

        if not chunks:
            return MultiAgentResponse(
                answer="I could not find any relevant information to answer your question.",
                query=question,
                retrieved_chunks=[],
                steps=steps,
                agent_type="multi",
            )

        # Step 2: Analyze
        analysis = await self._analyzer.run(question, chunks)
        steps.append(AgentStep(
            agent_name="analyzer",
            input_summary=f"{len(chunks)} chunks",
            output_summary=analysis[:200],
        ))

        # Step 3: Generate
        answer = await self._generator.run(question, chunks, analysis)
        steps.append(AgentStep(
            agent_name="generator",
            input_summary="analysis + passages",
            output_summary=answer[:200],
        ))

        # Step 4: Critique + feedback loop
        critique = ""
        revision_count = 0

        if not skip_critique:
            for revision in range(self._max_revisions + 1):
                critique, needs_revision = await self._critic.run(
                    question, answer, chunks
                )
                steps.append(AgentStep(
                    agent_name="critic",
                    input_summary=f"answer (revision {revision})",
                    output_summary=critique[:200],
                ))

                if not needs_revision or revision == self._max_revisions:
                    break

                # Revise
                revision_count += 1
                answer = await self._generator.run(
                    question, chunks, analysis, previous_feedback=critique
                )
                steps.append(AgentStep(
                    agent_name="generator",
                    input_summary=f"revision {revision_count} with feedback",
                    output_summary=answer[:200],
                ))

        # Store in memory
        if self._memory and conversation_id:
            await self._memory.conversation.add_turn(
                conversation_id, "user", question
            )
            await self._memory.conversation.add_turn(
                conversation_id, "assistant", answer
            )

        return MultiAgentResponse(
            answer=answer,
            query=question,
            retrieved_chunks=chunks,
            analysis=analysis,
            critique=critique,
            revision_count=revision_count,
            steps=steps,
            metadata={
                "strategy": "multi_agent",
                "revisions": revision_count,
                "agents_invoked": len(steps),
            },
            agent_type="multi",
        )
