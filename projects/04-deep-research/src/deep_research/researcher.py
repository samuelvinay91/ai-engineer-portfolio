"""Deep Researcher Agent -- multi-step research with inference-time scaling.

Implements the full research lifecycle:

1. **Plan** -- decompose the question into sub-questions.
2. **Search** -- execute parallel web searches via Tavily.
3. **Analyse** -- extract key findings from retrieved sources.
4. **Reason** -- synthesise findings using CoT / extended thinking.
5. **Verify** -- cross-check findings, identify gaps.
6. **Iterate** -- go deeper on unresolved sub-questions.
7. **Report** -- generate the final research report.

The researcher applies *inference-time scaling*: harder sub-questions are
automatically allocated more thinking budget.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, AsyncIterator

import anthropic
import structlog

from deep_research.config import Settings, get_settings
from deep_research.planner import ResearchPlan, ResearchPlanner, SubQuestion, SubQuestionStatus
from deep_research.reasoning import ReasoningEngine, ReasoningResult, ReasoningStrategy
from deep_research.report import ReportGenerator, ResearchReport

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ResearchStatus(str, Enum):
    PLANNING = "planning"
    SEARCHING = "searching"
    ANALYSING = "analysing"
    REASONING = "reasoning"
    VERIFYING = "verifying"
    ITERATING = "iterating"
    REPORTING = "reporting"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class SearchResult:
    """A single search result from a web-search provider."""

    title: str
    url: str
    snippet: str
    relevance_score: float = 0.0
    content: str = ""


@dataclass
class ResearchFinding:
    """A key finding extracted from one or more search results."""

    claim: str
    evidence: str
    sources: list[str] = field(default_factory=list)
    confidence: float = 0.0
    sub_question_id: str = ""


@dataclass
class ProgressEvent:
    """Streaming progress update sent to clients."""

    task_id: str
    status: ResearchStatus
    message: str
    progress_pct: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchTask:
    """Top-level container for an in-flight research task."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    status: ResearchStatus = ResearchStatus.PLANNING
    plan: ResearchPlan | None = None
    findings: list[ResearchFinding] = field(default_factory=list)
    search_results: list[SearchResult] = field(default_factory=list)
    reasoning_results: list[ReasoningResult] = field(default_factory=list)
    report: ResearchReport | None = None
    depth: int = 0
    max_depth: int = 5
    started_at: float = field(default_factory=time.time)
    completed_at: float | None = None
    error: str | None = None


# ---------------------------------------------------------------------------
# Web search adapter
# ---------------------------------------------------------------------------

class WebSearcher:
    """Thin wrapper around the Tavily search API."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._api_key = self.settings.tavily_api_key

    async def search(self, query: str, max_results: int = 5) -> list[SearchResult]:
        """Execute a web search and return structured results."""
        if not self._api_key:
            logger.warning("web_search.no_api_key", query=query[:80])
            return self._mock_results(query)

        try:
            from tavily import AsyncTavilyClient

            client = AsyncTavilyClient(api_key=self._api_key)
            response = await client.search(
                query=query,
                max_results=max_results,
                include_raw_content=False,
            )
            results: list[SearchResult] = []
            for item in response.get("results", []):
                results.append(SearchResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    snippet=item.get("content", ""),
                    relevance_score=item.get("score", 0.0),
                ))
            return results
        except Exception as exc:
            logger.error("web_search.error", error=str(exc), query=query[:80])
            return self._mock_results(query)

    @staticmethod
    def _mock_results(query: str) -> list[SearchResult]:
        """Return placeholder results when no API key is available."""
        return [
            SearchResult(
                title=f"Mock result for: {query[:60]}",
                url="https://example.com/mock",
                snippet=f"This is a placeholder result for the query: {query}",
                relevance_score=0.5,
            ),
        ]


# ---------------------------------------------------------------------------
# Deep Researcher
# ---------------------------------------------------------------------------

class DeepResearcher:
    """Orchestrates the full deep-research pipeline.

    Parameters
    ----------
    settings:
        Application settings.
    client:
        Anthropic async client, injected for testability.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        client: anthropic.AsyncAnthropic | None = None,
    ) -> None:
        self.settings = settings or get_settings()
        self._client = client or anthropic.AsyncAnthropic(
            api_key=self.settings.anthropic_api_key or None,
        )
        self.planner = ResearchPlanner(settings=self.settings, client=self._client)
        self.reasoner = ReasoningEngine(settings=self.settings, client=self._client)
        self.searcher = WebSearcher(settings=self.settings)
        self.report_gen = ReportGenerator(settings=self.settings, client=self._client)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    async def research(
        self,
        question: str,
        *,
        max_depth: int | None = None,
    ) -> ResearchTask:
        """Run a complete deep-research session (non-streaming)."""
        task = ResearchTask(
            question=question,
            max_depth=max_depth or self.settings.max_research_depth,
        )
        log = logger.bind(task_id=task.id, question=question[:120])
        log.info("researcher.start")

        try:
            await self._execute_pipeline(task)
        except Exception as exc:
            task.status = ResearchStatus.FAILED
            task.error = str(exc)
            log.error("researcher.failed", error=str(exc))
            raise

        task.completed_at = time.time()
        task.status = ResearchStatus.COMPLETED
        log.info("researcher.complete", depth=task.depth, findings=len(task.findings))
        return task

    async def research_stream(
        self,
        question: str,
        *,
        max_depth: int | None = None,
    ) -> AsyncIterator[ProgressEvent]:
        """Run deep research and yield streaming progress events."""
        task = ResearchTask(
            question=question,
            max_depth=max_depth or self.settings.max_research_depth,
        )
        log = logger.bind(task_id=task.id)

        try:
            async for event in self._execute_pipeline_streaming(task):
                yield event
        except Exception as exc:
            task.status = ResearchStatus.FAILED
            task.error = str(exc)
            log.error("researcher.stream.failed", error=str(exc))
            yield ProgressEvent(
                task_id=task.id,
                status=ResearchStatus.FAILED,
                message=f"Research failed: {exc}",
            )

    # ------------------------------------------------------------------
    # Pipeline implementation
    # ------------------------------------------------------------------

    async def _execute_pipeline(self, task: ResearchTask) -> None:
        """Execute the full pipeline without streaming."""
        # 1. Plan
        task.status = ResearchStatus.PLANNING
        task.plan = await self.planner.create_plan(task.question)

        while task.depth < task.max_depth:
            task.depth += 1
            ready = task.plan.get_ready_questions()
            if not ready:
                break

            # 2. Search
            task.status = ResearchStatus.SEARCHING
            await self._search_batch(task, ready)

            # 3. Analyse
            task.status = ResearchStatus.ANALYSING
            await self._analyse_batch(task, ready)

            # 4. Reason
            task.status = ResearchStatus.REASONING
            await self._reason_batch(task, ready)

            # 5. Verify
            task.status = ResearchStatus.VERIFYING
            gaps = await self._verify(task)

            # 6. Iterate?
            if not gaps or task.depth >= task.max_depth:
                break
            task.status = ResearchStatus.ITERATING
            task.plan = await self.planner.refine_plan(task.plan)

        # 7. Report
        task.status = ResearchStatus.REPORTING
        task.report = await self.report_gen.generate(task)

    async def _execute_pipeline_streaming(
        self,
        task: ResearchTask,
    ) -> AsyncIterator[ProgressEvent]:
        """Execute pipeline yielding progress events at each stage."""

        def _event(status: ResearchStatus, msg: str, pct: float = 0.0, **kw: Any) -> ProgressEvent:
            task.status = status
            return ProgressEvent(task_id=task.id, status=status, message=msg, progress_pct=pct, metadata=kw)

        yield _event(ResearchStatus.PLANNING, "Decomposing research question into sub-questions...")
        task.plan = await self.planner.create_plan(task.question)
        yield _event(
            ResearchStatus.PLANNING,
            f"Created plan with {task.plan.total_count} sub-questions.",
            progress_pct=5.0,
            sub_questions=[sq.question for sq in task.plan.sub_questions],
        )

        while task.depth < task.max_depth:
            task.depth += 1
            ready = task.plan.get_ready_questions()
            if not ready:
                break

            base_pct = 5.0 + (task.depth - 1) * (80.0 / task.max_depth)
            step_pct = 80.0 / task.max_depth / 4

            yield _event(ResearchStatus.SEARCHING, f"Searching for {len(ready)} sub-questions (depth {task.depth})...", base_pct)
            await self._search_batch(task, ready)

            yield _event(ResearchStatus.ANALYSING, "Analysing search results...", base_pct + step_pct)
            await self._analyse_batch(task, ready)

            yield _event(ResearchStatus.REASONING, "Reasoning over findings...", base_pct + 2 * step_pct)
            await self._reason_batch(task, ready)

            yield _event(ResearchStatus.VERIFYING, "Verifying and cross-checking...", base_pct + 3 * step_pct)
            gaps = await self._verify(task)

            if not gaps or task.depth >= task.max_depth:
                break
            yield _event(ResearchStatus.ITERATING, "Refining research plan for next iteration...", base_pct + 4 * step_pct)
            task.plan = await self.planner.refine_plan(task.plan)

        yield _event(ResearchStatus.REPORTING, "Generating research report...", 90.0)
        task.report = await self.report_gen.generate(task)

        task.completed_at = time.time()
        yield _event(ResearchStatus.COMPLETED, "Research complete.", 100.0)

    # ------------------------------------------------------------------
    # Stage implementations
    # ------------------------------------------------------------------

    async def _search_batch(
        self,
        task: ResearchTask,
        sub_questions: list[SubQuestion],
    ) -> None:
        """Run web searches for a batch of sub-questions in parallel."""
        sem = asyncio.Semaphore(self.settings.max_parallel_searches)

        async def _search_one(sq: SubQuestion) -> list[SearchResult]:
            async with sem:
                return await self.searcher.search(
                    sq.question,
                    max_results=self.settings.max_sources_per_query,
                )

        results = await asyncio.gather(*[_search_one(sq) for sq in sub_questions])
        for sq, res_list in zip(sub_questions, results):
            for r in res_list:
                task.search_results.append(r)
                sq.metadata.setdefault("search_results", []).append(r.url)

    async def _analyse_batch(
        self,
        task: ResearchTask,
        sub_questions: list[SubQuestion],
    ) -> None:
        """Extract key findings from search results for each sub-question."""
        for sq in sub_questions:
            sq.status = SubQuestionStatus.IN_PROGRESS
            urls = sq.metadata.get("search_results", [])
            relevant = [sr for sr in task.search_results if sr.url in urls]

            evidence_text = "\n\n".join(
                f"Source: {sr.title} ({sr.url})\n{sr.snippet}" for sr in relevant
            )
            if not evidence_text:
                evidence_text = "(no search results available)"

            response = await self._client.messages.create(
                model=self.settings.fast_model,
                max_tokens=2_048,
                system=(
                    "You are a research analyst. Extract the key findings from the "
                    "provided evidence that answer the question. Be factual and cite sources."
                ),
                messages=[{
                    "role": "user",
                    "content": (
                        f"Question: {sq.question}\n\nEvidence:\n{evidence_text}\n\n"
                        "Provide key findings as a bulleted list."
                    ),
                }],
            )
            findings_text = "\n".join(
                b.text for b in response.content if b.type == "text"
            )
            task.findings.append(ResearchFinding(
                claim=sq.question,
                evidence=findings_text,
                sources=[sr.url for sr in relevant],
                sub_question_id=sq.id,
            ))

    async def _reason_batch(
        self,
        task: ResearchTask,
        sub_questions: list[SubQuestion],
    ) -> None:
        """Apply reasoning to synthesise findings for each sub-question.

        Harder questions (those with low initial confidence) receive *more*
        inference-time compute via extended thinking -- this is the core of
        inference-time scaling.
        """
        for sq in sub_questions:
            related_findings = [
                f for f in task.findings if f.sub_question_id == sq.id
            ]
            context = "\n".join(f.evidence for f in related_findings)

            reasoning_query = (
                f"Based on the following research findings, provide a thorough "
                f"answer to: {sq.question}\n\nFindings:\n{context}"
            )
            result = await self.reasoner.reason(
                reasoning_query,
                strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
                use_extended_thinking=True,
            )
            task.reasoning_results.append(result)

            assert task.plan is not None
            task.plan.mark_complete(sq.id, result.answer, result.confidence)

    async def _verify(self, task: ResearchTask) -> list[str]:
        """Cross-check findings and identify knowledge gaps.

        Returns a list of gap descriptions.  If empty, no further iteration
        is needed.
        """
        if not task.findings:
            return []

        findings_text = "\n\n".join(
            f"Claim: {f.claim}\nEvidence: {f.evidence}\nConfidence: {f.confidence}"
            for f in task.findings
        )
        response = await self._client.messages.create(
            model=self.settings.fast_model,
            max_tokens=2_048,
            system=(
                "You are a research verifier. Identify contradictions, unsupported "
                "claims, and important gaps in the research findings. If everything "
                "is well-supported respond with VERIFIED."
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"Original question: {task.question}\n\n"
                    f"Findings:\n{findings_text}\n\n"
                    "List any gaps or issues, one per line. "
                    "If none, respond with a single line: VERIFIED"
                ),
            }],
        )
        text = "\n".join(b.text for b in response.content if b.type == "text")
        if "VERIFIED" in text.upper():
            return []

        gaps = [line.strip() for line in text.split("\n") if line.strip()]
        logger.info("researcher.verify.gaps", gap_count=len(gaps))
        return gaps
