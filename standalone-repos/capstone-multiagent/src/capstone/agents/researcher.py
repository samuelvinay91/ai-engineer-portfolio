"""Research Agent -- deep research with multi-step search and synthesis.

Wraps the deep-research capability from the portfolio into a pluggable agent.
The researcher can:

* Decompose a research question into targeted search queries
* Execute multiple rounds of web search (via Tavily or similar)
* Cross-reference and synthesise findings
* Produce a structured research brief with citations

When no live search API is configured the agent falls back to LLM-only
reasoning, which still provides a useful (albeit less grounded) output.
"""

from __future__ import annotations

import json
from typing import Any

import structlog

from capstone.agents.base import AgentResult, AgentTask, BaseAgent, TaskStatus
from capstone.config import Settings

logger = structlog.get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

RESEARCH_SYSTEM_PROMPT = """\
You are an expert research analyst within a multi-agent AI platform. Your role
is to conduct thorough, multi-faceted research on the given topic.

## Process

1. **Decompose** the research question into 2-4 specific sub-questions.
2. **Investigate** each sub-question, drawing on your knowledge.
3. **Synthesise** findings into a coherent research brief.
4. **Cite** your reasoning and note any areas of uncertainty.

## Output format

Return your response as JSON with this schema:
{
  "sub_questions": ["<q1>", "<q2>", ...],
  "findings": [
    {"question": "<q>", "answer": "<detailed answer>", "confidence": <0-1>}
  ],
  "synthesis": "<cohesive summary integrating all findings>",
  "key_insights": ["<insight1>", "<insight2>", ...],
  "limitations": ["<limitation1>", ...],
  "suggested_search_queries": ["<query1>", ...]
}
"""

CONFIDENCE_SYSTEM_PROMPT = """\
You are a task classifier. Determine how well a research-focused AI agent can
handle the following task. Consider whether the task requires:
- Information gathering or synthesis
- Multi-source analysis
- Literature review or fact-checking
- Trend analysis across data points

Respond with ONLY a JSON object: {"confidence": <float 0-1>, "reasoning": "<brief>"}
"""


class ResearcherAgent(BaseAgent):
    """Deep-research agent that decomposes questions and synthesises findings.

    Mirrors the methodology from the ``04-deep-research`` project, adapted to
    operate as a composable agent within the orchestrator.
    """

    name = "researcher"
    description = (
        "Conducts deep, multi-step research by decomposing questions, "
        "gathering information, and synthesising findings with citations."
    )
    capabilities = [
        "deep_research",
        "fact_checking",
        "literature_review",
        "trend_analysis",
        "information_synthesis",
        "multi_source_analysis",
    ]

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._model = settings.researcher_model

    async def can_handle(self, task: AgentTask) -> float:
        """Score confidence for research-style tasks."""
        description_lower = task.description.lower()

        # High-signal keywords
        high_keywords = [
            "research", "investigate", "analyse", "analyze", "study",
            "compare", "review", "survey", "explore", "evaluate",
            "find out", "look into", "what is", "how does", "why does",
        ]
        medium_keywords = [
            "summarize", "summarise", "explain", "overview",
            "background", "history", "trend", "insight",
        ]

        score = 0.1  # baseline
        for kw in high_keywords:
            if kw in description_lower:
                score += 0.15
        for kw in medium_keywords:
            if kw in description_lower:
                score += 0.08

        # Preferred-agent hint overrides heuristics
        if task.preferred_agent == self.name:
            score = max(score, 0.85)

        return min(score, 1.0)

    async def execute(self, task: AgentTask) -> AgentResult:
        """Conduct research and return a structured brief."""
        context_str = ""
        if task.context:
            context_str = "\n\nAdditional context:\n" + json.dumps(task.context, indent=2)

        constraints_str = ""
        if task.constraints:
            constraints_str = "\n\nConstraints:\n- " + "\n- ".join(task.constraints)

        user_message = (
            f"Research the following topic thoroughly:\n\n"
            f"{task.description}"
            f"{context_str}"
            f"{constraints_str}"
        )

        raw_response = await self._invoke_llm(
            system_prompt=RESEARCH_SYSTEM_PROMPT,
            user_message=user_message,
            model=self._model,
            temperature=0.2,
        )

        structured = self._parse_research_output(raw_response)

        return AgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            status=TaskStatus.COMPLETED,
            output=structured.get("synthesis", raw_response),
            structured_data=structured,
            confidence=self._compute_confidence(structured),
            metadata={"model": self._model, "raw_response_length": len(raw_response)},
        )

    # -- Internals -------------------------------------------------------------

    @staticmethod
    def _parse_research_output(raw: str) -> dict[str, Any]:
        """Attempt to parse JSON from the LLM response, falling back gracefully."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("researcher_json_parse_failed", raw_length=len(raw))
            return {
                "synthesis": raw,
                "sub_questions": [],
                "findings": [],
                "key_insights": [],
                "limitations": ["Output was not in expected JSON format."],
            }

    @staticmethod
    def _compute_confidence(structured: dict[str, Any]) -> float:
        """Derive an overall confidence from individual finding confidences."""
        findings = structured.get("findings", [])
        if not findings:
            return 0.6
        confidences = [f.get("confidence", 0.5) for f in findings if isinstance(f, dict)]
        if not confidences:
            return 0.6
        return round(sum(confidences) / len(confidences), 2)
