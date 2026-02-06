"""Coding Agent -- code generation, review, bug fixing, and explanation.

This agent handles all software-engineering tasks within the multi-agent
platform:

* **Code generation** from natural-language specifications
* **Code review** with actionable improvement suggestions
* **Bug fixing** given error descriptions or failing test output
* **Code explanation** for complex or unfamiliar codebases
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

CODER_SYSTEM_PROMPT = """\
You are an expert software engineer within a multi-agent AI platform. You
handle all coding tasks with precision and best-practice adherence.

## Capabilities

1. **Code Generation** -- Write clean, well-documented, production-quality code
   from natural-language descriptions. Include type hints, docstrings, and error
   handling.
2. **Code Review** -- Analyse code for bugs, security issues, performance
   problems, and style violations. Provide specific, actionable suggestions.
3. **Bug Fixing** -- Given a bug description and/or error trace, identify root
   causes and provide corrected code with explanations.
4. **Code Explanation** -- Break down complex code into understandable parts,
   explaining the logic, patterns, and design decisions.

## Output format

Return your response as JSON:
{
  "task_type": "generation|review|bugfix|explanation",
  "summary": "<concise summary of what was done>",
  "code_blocks": [
    {
      "filename": "<file.py>",
      "language": "<python|javascript|...>",
      "code": "<the code>",
      "description": "<what this code does>"
    }
  ],
  "suggestions": ["<suggestion1>", ...],
  "issues_found": [
    {"severity": "critical|warning|info", "description": "<issue>", "fix": "<how to fix>"}
  ],
  "explanation": "<detailed explanation if applicable>"
}
"""

CONFIDENCE_KEYWORDS_HIGH = [
    "code", "implement", "function", "class", "api", "endpoint",
    "debug", "fix", "bug", "error", "exception", "refactor",
    "review", "test", "unit test", "integration", "script",
    "program", "algorithm", "data structure", "sql", "query",
    "python", "javascript", "typescript", "java", "rust", "go",
]

CONFIDENCE_KEYWORDS_MEDIUM = [
    "build", "create", "develop", "design", "architecture",
    "optimize", "performance", "deploy", "configuration",
    "dockerfile", "yaml", "json", "schema",
]


class CoderAgent(BaseAgent):
    """Software engineering agent for code generation, review, and debugging."""

    name = "coder"
    description = (
        "Expert software engineer that generates, reviews, debugs, and "
        "explains code across multiple languages and frameworks."
    )
    capabilities = [
        "code_generation",
        "code_review",
        "bug_fixing",
        "code_explanation",
        "refactoring",
        "test_writing",
        "api_design",
    ]

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._model = settings.coder_model

    async def can_handle(self, task: AgentTask) -> float:
        """Score confidence for coding tasks."""
        description_lower = task.description.lower()

        score = 0.05
        for kw in CONFIDENCE_KEYWORDS_HIGH:
            if kw in description_lower:
                score += 0.12
        for kw in CONFIDENCE_KEYWORDS_MEDIUM:
            if kw in description_lower:
                score += 0.06

        # Check for code in context (strong signal)
        context_str = json.dumps(task.context) if task.context else ""
        if any(marker in context_str for marker in ["def ", "class ", "function ", "import "]):
            score += 0.2

        if task.preferred_agent == self.name:
            score = max(score, 0.85)

        return min(score, 1.0)

    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute a coding task and return structured output."""
        context_str = ""
        if task.context:
            context_str = "\n\nContext / existing code:\n```\n"
            context_str += json.dumps(task.context, indent=2)
            context_str += "\n```"

        constraints_str = ""
        if task.constraints:
            constraints_str = "\n\nRequirements:\n- " + "\n- ".join(task.constraints)

        user_message = (
            f"Task:\n{task.description}"
            f"{context_str}"
            f"{constraints_str}"
        )

        raw_response = await self._invoke_llm(
            system_prompt=CODER_SYSTEM_PROMPT,
            user_message=user_message,
            model=self._model,
            temperature=0.1,  # low temperature for precise code
        )

        structured = self._parse_coder_output(raw_response)

        return AgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            status=TaskStatus.COMPLETED,
            output=structured.get("summary", raw_response),
            structured_data=structured,
            confidence=self._assess_output_quality(structured),
            metadata={"model": self._model, "task_type": structured.get("task_type", "unknown")},
        )

    # -- Internals -------------------------------------------------------------

    @staticmethod
    def _parse_coder_output(raw: str) -> dict[str, Any]:
        """Parse JSON output from the coding LLM."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("coder_json_parse_failed", raw_length=len(raw))
            return {
                "task_type": "unknown",
                "summary": raw,
                "code_blocks": [],
                "suggestions": [],
                "issues_found": [],
            }

    @staticmethod
    def _assess_output_quality(structured: dict[str, Any]) -> float:
        """Heuristic confidence based on output completeness."""
        score = 0.6
        if structured.get("code_blocks"):
            score += 0.15
        if structured.get("summary") and len(structured["summary"]) > 20:
            score += 0.1
        if structured.get("suggestions"):
            score += 0.05
        if structured.get("task_type") != "unknown":
            score += 0.1
        return min(score, 1.0)
