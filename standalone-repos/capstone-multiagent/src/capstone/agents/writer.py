"""Content Writer Agent -- versatile content creation and editing.

The writer agent handles all content-production tasks:

* **Blog posts** and long-form articles
* **Technical documentation** and API references
* **Business reports** and executive summaries
* **Creative writing** with multiple style options
* **SEO optimization** for web-published content
* **Editing and proofreading** of existing text
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

WRITER_SYSTEM_PROMPT = """\
You are a professional content writer and editor within a multi-agent AI
platform. You produce polished, publication-ready content across multiple
formats and styles.

## Capabilities

1. **Blog Posts & Articles** -- Engaging, well-structured long-form content
   with compelling introductions, clear sections, and strong conclusions.
2. **Technical Documentation** -- Precise, developer-friendly docs with code
   examples, API references, and step-by-step guides.
3. **Business Reports** -- Data-driven reports with executive summaries,
   findings, and actionable recommendations.
4. **SEO Optimization** -- Content structured for search visibility with
   strategic keyword placement, meta descriptions, and heading hierarchy.
5. **Editing & Proofreading** -- Grammar, style, clarity, and consistency
   improvements while preserving the author's voice.

## Output format

Return your response as JSON:
{
  "content_type": "blog_post|documentation|report|creative|seo_content|edit",
  "title": "<suggested title>",
  "summary": "<1-2 sentence summary>",
  "content": "<the full content in markdown format>",
  "metadata": {
    "word_count": <int>,
    "reading_time_minutes": <int>,
    "tone": "<professional|casual|technical|persuasive|academic>",
    "target_audience": "<description>"
  },
  "seo": {
    "meta_description": "<155 chars max>",
    "keywords": ["<kw1>", "<kw2>", ...],
    "heading_structure": ["H1: ...", "H2: ...", ...]
  },
  "suggestions": ["<improvement1>", ...]
}
"""

CONFIDENCE_KEYWORDS_HIGH = [
    "write", "blog", "article", "post", "content", "copy",
    "documentation", "docs", "report", "essay", "letter",
    "email", "newsletter", "press release", "creative",
    "story", "narrative", "editing", "proofread", "rewrite",
]

CONFIDENCE_KEYWORDS_MEDIUM = [
    "seo", "keyword", "headline", "title", "summary",
    "draft", "outline", "tone", "style", "audience",
    "publish", "format", "template", "communicate",
]


class WriterAgent(BaseAgent):
    """Content creation agent for blogs, docs, reports, and creative writing."""

    name = "writer"
    description = (
        "Professional content writer and editor that produces blog posts, "
        "documentation, reports, and creative content across styles with "
        "SEO optimization and editorial polish."
    )
    capabilities = [
        "blog_writing",
        "technical_documentation",
        "report_writing",
        "creative_writing",
        "seo_optimization",
        "editing_proofreading",
        "copywriting",
    ]

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._model = settings.writer_model

    async def can_handle(self, task: AgentTask) -> float:
        """Score confidence for content-writing tasks."""
        description_lower = task.description.lower()

        score = 0.05
        for kw in CONFIDENCE_KEYWORDS_HIGH:
            if kw in description_lower:
                score += 0.12
        for kw in CONFIDENCE_KEYWORDS_MEDIUM:
            if kw in description_lower:
                score += 0.06

        # Check for writing-style indicators in context
        context_str = json.dumps(task.context) if task.context else ""
        if any(marker in context_str.lower() for marker in ["tone", "audience", "style", "draft"]):
            score += 0.1

        if task.preferred_agent == self.name:
            score = max(score, 0.85)

        return min(score, 1.0)

    async def execute(self, task: AgentTask) -> AgentResult:
        """Create or edit content and return structured output."""
        context_str = ""
        if task.context:
            context_str = "\n\nContext / source material:\n```\n"
            context_str += json.dumps(task.context, indent=2)
            context_str += "\n```"

        constraints_str = ""
        if task.constraints:
            constraints_str = "\n\nContent requirements:\n- " + "\n- ".join(task.constraints)

        user_message = (
            f"Content task:\n\n"
            f"{task.description}"
            f"{context_str}"
            f"{constraints_str}"
        )

        raw_response = await self._invoke_llm(
            system_prompt=WRITER_SYSTEM_PROMPT,
            user_message=user_message,
            model=self._model,
            temperature=0.6,  # higher creativity for writing
        )

        structured = self._parse_writer_output(raw_response)

        return AgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            status=TaskStatus.COMPLETED,
            output=structured.get("content", structured.get("summary", raw_response)),
            structured_data=structured,
            confidence=self._assess_content_quality(structured),
            metadata={
                "model": self._model,
                "content_type": structured.get("content_type", "unknown"),
                "word_count": structured.get("metadata", {}).get("word_count", 0),
            },
        )

    # -- Internals -------------------------------------------------------------

    @staticmethod
    def _parse_writer_output(raw: str) -> dict[str, Any]:
        """Parse JSON output from the writer LLM."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("writer_json_parse_failed", raw_length=len(raw))
            return {
                "content_type": "unknown",
                "title": "",
                "summary": "",
                "content": raw,
                "metadata": {},
                "seo": {},
                "suggestions": [],
            }

    @staticmethod
    def _assess_content_quality(structured: dict[str, Any]) -> float:
        """Heuristic confidence from output completeness."""
        score = 0.5
        if structured.get("content") and len(structured["content"]) > 100:
            score += 0.15
        if structured.get("title"):
            score += 0.1
        if structured.get("seo", {}).get("meta_description"):
            score += 0.1
        if structured.get("metadata", {}).get("word_count", 0) > 0:
            score += 0.05
        if structured.get("content_type") != "unknown":
            score += 0.1
        return min(score, 1.0)
