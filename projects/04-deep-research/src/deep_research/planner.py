"""Research Planner -- decomposes complex questions into a dependency graph.

The planner is the first stage of the deep-research pipeline.  It takes a
high-level research question and:

1. Generates sub-questions using CoT decomposition.
2. Identifies dependencies between sub-questions.
3. Prioritises them by importance and topological order.
4. Supports iterative refinement as new information comes in.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import anthropic
import structlog

from deep_research.config import Settings, get_settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class SubQuestionStatus(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    BLOCKED = "blocked"
    SKIPPED = "skipped"


@dataclass
class SubQuestion:
    """A single sub-question within a research plan."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    question: str = ""
    rationale: str = ""
    priority: int = 0  # lower = higher priority
    status: SubQuestionStatus = SubQuestionStatus.PENDING
    depends_on: list[str] = field(default_factory=list)
    findings: str = ""
    confidence: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ResearchPlan:
    """A complete research plan with sub-questions and dependency edges."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    original_question: str = ""
    sub_questions: list[SubQuestion] = field(default_factory=list)
    iteration: int = 0
    created_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- convenience helpers -----------------------------------------------

    @property
    def completed_count(self) -> int:
        return sum(1 for sq in self.sub_questions if sq.status == SubQuestionStatus.COMPLETED)

    @property
    def total_count(self) -> int:
        return len(self.sub_questions)

    @property
    def progress_pct(self) -> float:
        return (self.completed_count / self.total_count * 100) if self.total_count else 0.0

    def get_ready_questions(self) -> list[SubQuestion]:
        """Return sub-questions whose dependencies have all been completed."""
        completed_ids = {
            sq.id for sq in self.sub_questions if sq.status == SubQuestionStatus.COMPLETED
        }
        return [
            sq
            for sq in self.sub_questions
            if sq.status == SubQuestionStatus.PENDING
            and all(dep in completed_ids for dep in sq.depends_on)
        ]

    def mark_complete(self, question_id: str, findings: str, confidence: float) -> None:
        for sq in self.sub_questions:
            if sq.id == question_id:
                sq.status = SubQuestionStatus.COMPLETED
                sq.findings = findings
                sq.confidence = confidence
                return
        raise KeyError(f"SubQuestion {question_id} not found in plan")


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = """\
You are an expert research planner.  Given a complex research question, break
it down into 3-7 focused sub-questions that, when answered together, fully
address the original question.

For each sub-question specify:
- The question itself
- A short rationale for why it matters
- Which other sub-question IDs it depends on (use SQ-1, SQ-2, etc.)
- A priority from 1 (highest) to 5 (lowest)

Use this EXACT format for each sub-question:

[SQ-<n>]
Question: <the sub-question>
Rationale: <why this matters>
Depends on: <comma-separated SQ IDs, or NONE>
Priority: <1-5>

Original question: {question}
"""

REFINE_PROMPT = """\
You are refining an existing research plan based on new findings.

Original question: {question}

Current sub-questions and their status:
{current_plan}

New findings so far:
{findings}

Identify:
1. Sub-questions that are now unnecessary (mark SKIP).
2. New sub-questions that should be added.
3. Priority adjustments.

Use the same format as before.  For existing questions you want to keep, just
write [KEEP SQ-<id>].  For skips write [SKIP SQ-<id>].  For new questions use
[SQ-NEW-<n>] with the same fields.
"""


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class ResearchPlanner:
    """Decomposes a research question into a plan of sub-questions.

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

    async def create_plan(self, question: str) -> ResearchPlan:
        """Generate an initial research plan from a high-level question."""
        log = logger.bind(question=question[:120])
        log.info("planner.create_plan.start")
        t0 = time.perf_counter()

        response = await self._client.messages.create(
            model=self.settings.fast_model,
            max_tokens=4_096,
            system="You are a meticulous research planner.",
            messages=[{"role": "user", "content": DECOMPOSE_PROMPT.format(question=question)}],
        )
        text = self._extract_text(response)
        sub_questions = self._parse_sub_questions(text)

        plan = ResearchPlan(
            original_question=question,
            sub_questions=sub_questions,
            iteration=0,
        )
        dur = (time.perf_counter() - t0) * 1_000
        log.info("planner.create_plan.done", sub_questions=len(sub_questions), duration_ms=round(dur, 1))
        return plan

    async def refine_plan(self, plan: ResearchPlan) -> ResearchPlan:
        """Refine the plan based on findings gathered so far.

        This is called between research iterations so the planner can add new
        sub-questions, skip irrelevant ones, or re-prioritise.
        """
        log = logger.bind(plan_id=plan.id, iteration=plan.iteration)
        log.info("planner.refine_plan.start")

        current_text = self._plan_to_text(plan)
        findings_text = self._findings_to_text(plan)

        response = await self._client.messages.create(
            model=self.settings.fast_model,
            max_tokens=4_096,
            system="You are a meticulous research planner refining an existing plan.",
            messages=[{
                "role": "user",
                "content": REFINE_PROMPT.format(
                    question=plan.original_question,
                    current_plan=current_text,
                    findings=findings_text,
                ),
            }],
        )
        text = self._extract_text(response)
        self._apply_refinements(plan, text)
        plan.iteration += 1

        log.info("planner.refine_plan.done", sub_questions=len(plan.sub_questions))
        return plan

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_text(response: anthropic.types.Message) -> str:
        return "\n".join(
            block.text for block in response.content if block.type == "text"
        )

    @staticmethod
    def _parse_sub_questions(text: str) -> list[SubQuestion]:
        """Parse ``[SQ-<n>]`` blocks from model output."""
        import re

        pattern = re.compile(
            r"\[SQ-(\d+)\]\s*\n"
            r"Question:\s*(.+)\n"
            r"Rationale:\s*(.+)\n"
            r"Depends on:\s*(.+)\n"
            r"Priority:\s*(\d+)",
            re.IGNORECASE,
        )
        questions: list[SubQuestion] = []
        id_map: dict[str, str] = {}

        for m in pattern.finditer(text):
            sq_label = f"SQ-{m.group(1)}"
            sq = SubQuestion(
                question=m.group(2).strip(),
                rationale=m.group(3).strip(),
                priority=int(m.group(5)),
            )
            id_map[sq_label] = sq.id
            dep_str = m.group(4).strip()
            if dep_str.upper() != "NONE":
                sq.metadata["raw_depends"] = [d.strip() for d in dep_str.split(",")]
            questions.append(sq)

        # Resolve dependency labels to real UUIDs
        for sq in questions:
            raw_deps = sq.metadata.pop("raw_depends", [])
            sq.depends_on = [id_map[d] for d in raw_deps if d in id_map]

        # Sort by priority then dependency order
        questions.sort(key=lambda q: q.priority)
        return questions

    def _apply_refinements(self, plan: ResearchPlan, text: str) -> None:
        """Apply KEEP / SKIP / NEW directives from the refinement response."""
        import re

        # SKIP existing
        for m in re.finditer(r"\[SKIP\s+SQ-(\S+)\]", text, re.IGNORECASE):
            label = m.group(1)
            for sq in plan.sub_questions:
                if sq.id.startswith(label) or sq.question:
                    # heuristic: skip by index
                    pass
            # Simple approach: mark matching by order
            idx = int(label) - 1 if label.isdigit() else -1
            if 0 <= idx < len(plan.sub_questions):
                plan.sub_questions[idx].status = SubQuestionStatus.SKIPPED

        # NEW sub-questions
        new_pattern = re.compile(
            r"\[SQ-NEW-(\d+)\]\s*\n"
            r"Question:\s*(.+)\n"
            r"Rationale:\s*(.+)\n"
            r"Depends on:\s*(.+)\n"
            r"Priority:\s*(\d+)",
            re.IGNORECASE,
        )
        for m in new_pattern.finditer(text):
            sq = SubQuestion(
                question=m.group(2).strip(),
                rationale=m.group(3).strip(),
                priority=int(m.group(5)),
            )
            plan.sub_questions.append(sq)

    @staticmethod
    def _plan_to_text(plan: ResearchPlan) -> str:
        lines: list[str] = []
        for i, sq in enumerate(plan.sub_questions, 1):
            lines.append(
                f"SQ-{i} [{sq.status.value}] (priority {sq.priority}): {sq.question}"
            )
        return "\n".join(lines)

    @staticmethod
    def _findings_to_text(plan: ResearchPlan) -> str:
        lines: list[str] = []
        for sq in plan.sub_questions:
            if sq.status == SubQuestionStatus.COMPLETED and sq.findings:
                lines.append(f"Q: {sq.question}\nA: {sq.findings}\n")
        return "\n".join(lines) if lines else "(no findings yet)"
