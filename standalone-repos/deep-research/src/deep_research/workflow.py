"""LangGraph workflow -- full research pipeline as a compiled state graph.

The graph has the following nodes:

    plan --> search --> analyse --> reason --> verify
                                                 |
                                    (gaps?) ---+--> iterate (loops back to search)
                                               |
                                               +--> report --> END

Conditional edges decide whether to iterate (go deeper) or proceed to
report generation.  Independent sub-questions within a single iteration
are executed concurrently via ``asyncio.gather``.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal, TypedDict

import structlog
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages

from deep_research.config import Settings, get_settings
from deep_research.planner import ResearchPlan, ResearchPlanner, SubQuestionStatus
from deep_research.reasoning import ReasoningEngine, ReasoningResult, ReasoningStrategy
from deep_research.report import ReportGenerator, ResearchReport
from deep_research.researcher import (
    DeepResearcher,
    ProgressEvent,
    ResearchFinding,
    ResearchStatus,
    ResearchTask,
    SearchResult,
    WebSearcher,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------

class ResearchState(TypedDict, total=False):
    """Shared state passed between LangGraph nodes."""

    # Inputs
    question: str
    max_depth: int

    # Mutable research state
    task_id: str
    plan: ResearchPlan | None
    search_results: list[SearchResult]
    findings: list[ResearchFinding]
    reasoning_results: list[ReasoningResult]
    report: ResearchReport | None
    gaps: list[str]

    # Control flow
    current_depth: int
    status: str
    progress_events: Annotated[list[dict[str, Any]], add_messages]
    error: str | None


# ---------------------------------------------------------------------------
# Node functions
# ---------------------------------------------------------------------------

def _make_components(settings: Settings | None = None) -> tuple[
    ResearchPlanner, WebSearcher, ReasoningEngine, ReportGenerator, Settings,
]:
    """Instantiate shared components (one client for all)."""
    import anthropic

    settings = settings or get_settings()
    client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key or None)
    planner = ResearchPlanner(settings=settings, client=client)
    searcher = WebSearcher(settings=settings)
    reasoner = ReasoningEngine(settings=settings, client=client)
    report_gen = ReportGenerator(settings=settings, client=client)
    return planner, searcher, reasoner, report_gen, settings


async def plan_node(state: ResearchState) -> dict[str, Any]:
    """Decompose the research question into sub-questions."""
    planner, _, _, _, settings = _make_components()
    question = state["question"]
    logger.info("workflow.plan", question=question[:100])

    plan = await planner.create_plan(question)
    return {
        "plan": plan,
        "task_id": state.get("task_id", str(uuid.uuid4())),
        "current_depth": 0,
        "search_results": [],
        "findings": [],
        "reasoning_results": [],
        "gaps": [],
        "status": ResearchStatus.PLANNING.value,
        "progress_events": [
            {"role": "system", "content": f"Created plan with {plan.total_count} sub-questions."},
        ],
    }


async def search_node(state: ResearchState) -> dict[str, Any]:
    """Run web searches for all ready sub-questions."""
    _, searcher, _, _, settings = _make_components()
    plan: ResearchPlan | None = state.get("plan")
    if plan is None:
        return {"error": "No plan available for search."}

    ready = plan.get_ready_questions()
    if not ready:
        return {"search_results": state.get("search_results", [])}

    logger.info("workflow.search", ready_count=len(ready))
    sem = asyncio.Semaphore(settings.max_parallel_searches)
    all_results: list[SearchResult] = list(state.get("search_results", []))

    async def _search_one(q: str) -> list[SearchResult]:
        async with sem:
            return await searcher.search(q, max_results=settings.max_sources_per_query)

    batch = await asyncio.gather(*[_search_one(sq.question) for sq in ready])
    for sq, results in zip(ready, batch):
        for r in results:
            all_results.append(r)
            sq.metadata.setdefault("search_results", []).append(r.url)

    return {
        "search_results": all_results,
        "status": ResearchStatus.SEARCHING.value,
        "progress_events": [
            {"role": "system", "content": f"Searched {len(ready)} sub-questions, got {sum(len(b) for b in batch)} results."},
        ],
    }


async def analyse_node(state: ResearchState) -> dict[str, Any]:
    """Extract findings from search results."""
    import anthropic as _anthropic

    settings = get_settings()
    client = _anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key or None)
    plan: ResearchPlan | None = state.get("plan")
    if plan is None:
        return {"error": "No plan available for analysis."}

    ready = [sq for sq in plan.sub_questions if sq.status == SubQuestionStatus.PENDING]
    search_results = state.get("search_results", [])
    findings: list[ResearchFinding] = list(state.get("findings", []))

    logger.info("workflow.analyse", sub_questions=len(ready))

    for sq in ready:
        urls = sq.metadata.get("search_results", [])
        relevant = [sr for sr in search_results if sr.url in urls]
        evidence_text = "\n\n".join(
            f"Source: {sr.title} ({sr.url})\n{sr.snippet}" for sr in relevant
        ) or "(no evidence)"

        response = await client.messages.create(
            model=settings.fast_model,
            max_tokens=2_048,
            system="Extract key findings from the evidence. Be factual, cite sources.",
            messages=[{
                "role": "user",
                "content": f"Question: {sq.question}\n\nEvidence:\n{evidence_text}",
            }],
        )
        text = "\n".join(b.text for b in response.content if b.type == "text")
        findings.append(ResearchFinding(
            claim=sq.question,
            evidence=text,
            sources=[sr.url for sr in relevant],
            sub_question_id=sq.id,
        ))

    return {
        "findings": findings,
        "status": ResearchStatus.ANALYSING.value,
        "progress_events": [
            {"role": "system", "content": f"Extracted findings for {len(ready)} sub-questions."},
        ],
    }


async def reason_node(state: ResearchState) -> dict[str, Any]:
    """Synthesise findings with CoT reasoning."""
    _, _, reasoner, _, settings = _make_components()
    plan: ResearchPlan | None = state.get("plan")
    findings = state.get("findings", [])
    reasoning_results: list[ReasoningResult] = list(state.get("reasoning_results", []))

    if plan is None:
        return {"error": "No plan for reasoning."}

    pending = [sq for sq in plan.sub_questions if sq.status == SubQuestionStatus.PENDING]
    logger.info("workflow.reason", sub_questions=len(pending))

    for sq in pending:
        related = [f for f in findings if f.sub_question_id == sq.id]
        context = "\n".join(f.evidence for f in related)
        query = f"Answer: {sq.question}\n\nFindings:\n{context}"

        result = await reasoner.reason(
            query,
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            use_extended_thinking=True,
        )
        reasoning_results.append(result)
        plan.mark_complete(sq.id, result.answer, result.confidence)

    return {
        "reasoning_results": reasoning_results,
        "status": ResearchStatus.REASONING.value,
        "progress_events": [
            {"role": "system", "content": f"Reasoned over {len(pending)} sub-questions."},
        ],
    }


async def verify_node(state: ResearchState) -> dict[str, Any]:
    """Cross-check findings and identify gaps."""
    import anthropic as _anthropic

    settings = get_settings()
    client = _anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key or None)
    findings = state.get("findings", [])

    if not findings:
        return {"gaps": [], "status": ResearchStatus.VERIFYING.value}

    findings_text = "\n\n".join(
        f"Claim: {f.claim}\nEvidence: {f.evidence}" for f in findings
    )
    response = await client.messages.create(
        model=settings.fast_model,
        max_tokens=2_048,
        system="Identify contradictions, unsupported claims, and gaps. If sound, respond VERIFIED.",
        messages=[{
            "role": "user",
            "content": f"Question: {state['question']}\n\nFindings:\n{findings_text}",
        }],
    )
    text = "\n".join(b.text for b in response.content if b.type == "text")

    if "VERIFIED" in text.upper():
        gaps: list[str] = []
    else:
        gaps = [l.strip() for l in text.split("\n") if l.strip()]

    logger.info("workflow.verify", gaps=len(gaps))
    return {
        "gaps": gaps,
        "status": ResearchStatus.VERIFYING.value,
        "progress_events": [
            {"role": "system", "content": f"Verification complete. Gaps found: {len(gaps)}."},
        ],
    }


async def iterate_node(state: ResearchState) -> dict[str, Any]:
    """Refine the plan and increment depth for the next iteration."""
    planner, _, _, _, _ = _make_components()
    plan: ResearchPlan | None = state.get("plan")
    current_depth = state.get("current_depth", 0) + 1

    if plan is not None:
        plan = await planner.refine_plan(plan)

    logger.info("workflow.iterate", depth=current_depth)
    return {
        "plan": plan,
        "current_depth": current_depth,
        "status": ResearchStatus.ITERATING.value,
        "progress_events": [
            {"role": "system", "content": f"Iterating research (depth {current_depth})."},
        ],
    }


async def report_node(state: ResearchState) -> dict[str, Any]:
    """Generate the final research report."""
    _, _, _, report_gen, _ = _make_components()

    # Build a lightweight ResearchTask from state
    task = ResearchTask(
        id=state.get("task_id", str(uuid.uuid4())),
        question=state["question"],
        findings=state.get("findings", []),
        search_results=state.get("search_results", []),
        reasoning_results=state.get("reasoning_results", []),
    )
    report = await report_gen.generate(task)

    logger.info("workflow.report.done", sections=len(report.sections))
    return {
        "report": report,
        "status": ResearchStatus.COMPLETED.value,
        "progress_events": [
            {"role": "system", "content": "Research report generated."},
        ],
    }


# ---------------------------------------------------------------------------
# Conditional edge: iterate vs report
# ---------------------------------------------------------------------------

def should_iterate(state: ResearchState) -> Literal["iterate", "report"]:
    """Decide whether to do another research loop or generate the report."""
    gaps = state.get("gaps", [])
    current_depth = state.get("current_depth", 0)
    max_depth = state.get("max_depth", 5)

    if gaps and current_depth < max_depth:
        return "iterate"
    return "report"


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

def build_research_graph() -> StateGraph:
    """Construct and compile the LangGraph research workflow.

    Returns the compiled graph ready for ``ainvoke`` or ``astream``.
    """
    graph = StateGraph(ResearchState)

    # Add nodes
    graph.add_node("plan", plan_node)
    graph.add_node("search", search_node)
    graph.add_node("analyse", analyse_node)
    graph.add_node("reason", reason_node)
    graph.add_node("verify", verify_node)
    graph.add_node("iterate", iterate_node)
    graph.add_node("report", report_node)

    # Linear edges
    graph.add_edge("plan", "search")
    graph.add_edge("search", "analyse")
    graph.add_edge("analyse", "reason")
    graph.add_edge("reason", "verify")

    # Conditional: iterate or report
    graph.add_conditional_edges("verify", should_iterate, {
        "iterate": "iterate",
        "report": "report",
    })

    # Iterate loops back to search
    graph.add_edge("iterate", "search")

    # Report is terminal
    graph.add_edge("report", END)

    # Entry point
    graph.set_entry_point("plan")

    return graph.compile()
