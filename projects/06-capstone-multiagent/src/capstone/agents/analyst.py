"""Data Analysis Agent -- structured data analysis, insights, and trends.

The analyst agent specializes in:

* Analysing structured and semi-structured data
* Generating statistical summaries and insights
* Identifying trends, patterns, and anomalies
* Producing data-driven recommendations
* Creating analysis narratives from raw numbers
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

ANALYST_SYSTEM_PROMPT = """\
You are a senior data analyst within a multi-agent AI platform. You excel at
extracting meaning from data and communicating insights clearly.

## Capabilities

1. **Statistical Analysis** -- Compute descriptive statistics, distributions,
   correlations, and significance tests.
2. **Trend Identification** -- Detect temporal patterns, seasonality, and
   directional trends in time-series or sequential data.
3. **Insight Generation** -- Go beyond numbers to explain *why* patterns exist
   and *what* they mean for decision-making.
4. **Anomaly Detection** -- Flag outliers and unexpected values with context.
5. **Summary Reports** -- Produce executive-ready summaries with key takeaways.

## Output format

Return your response as JSON:
{
  "analysis_type": "statistical|trend|comparative|exploratory",
  "summary": "<executive summary of findings>",
  "key_metrics": [
    {"name": "<metric>", "value": "<value>", "interpretation": "<what it means>"}
  ],
  "trends": [
    {"description": "<trend>", "direction": "up|down|stable|cyclical", "significance": "high|medium|low"}
  ],
  "insights": [
    {"finding": "<what was found>", "implication": "<what it means>", "confidence": <0-1>}
  ],
  "anomalies": [
    {"description": "<anomaly>", "severity": "critical|warning|info"}
  ],
  "recommendations": ["<recommendation1>", ...],
  "methodology": "<brief description of analytical approach>"
}
"""

CONFIDENCE_KEYWORDS_HIGH = [
    "data", "analysis", "analyze", "analyse", "statistics",
    "metrics", "numbers", "trends", "patterns", "insights",
    "dashboard", "report", "kpi", "performance", "correlation",
    "distribution", "outlier", "anomaly", "forecast",
]

CONFIDENCE_KEYWORDS_MEDIUM = [
    "compare", "measure", "track", "monitor", "evaluate",
    "benchmark", "growth", "decline", "percentage", "ratio",
    "average", "median", "variance", "deviation",
]


class AnalystAgent(BaseAgent):
    """Data analysis agent for deriving insights from structured data."""

    name = "analyst"
    description = (
        "Senior data analyst that examines structured data, identifies trends "
        "and patterns, generates statistical summaries, and produces "
        "actionable insights with clear recommendations."
    )
    capabilities = [
        "statistical_analysis",
        "trend_identification",
        "anomaly_detection",
        "insight_generation",
        "data_summarization",
        "comparative_analysis",
        "forecasting",
    ]

    def __init__(self, settings: Settings) -> None:
        super().__init__(settings)
        self._model = settings.analyst_model

    async def can_handle(self, task: AgentTask) -> float:
        """Score confidence for data-analysis tasks."""
        description_lower = task.description.lower()

        score = 0.05
        for kw in CONFIDENCE_KEYWORDS_HIGH:
            if kw in description_lower:
                score += 0.12
        for kw in CONFIDENCE_KEYWORDS_MEDIUM:
            if kw in description_lower:
                score += 0.06

        # Strong signal: numeric data in context
        context_str = json.dumps(task.context) if task.context else ""
        numeric_chars = sum(1 for c in context_str if c.isdigit())
        if numeric_chars > 20:
            score += 0.15

        # Check for tabular markers
        if any(marker in context_str for marker in ["csv", "json", "table", "column", "row"]):
            score += 0.1

        if task.preferred_agent == self.name:
            score = max(score, 0.85)

        return min(score, 1.0)

    async def execute(self, task: AgentTask) -> AgentResult:
        """Analyse data and return structured insights."""
        context_str = ""
        if task.context:
            context_str = "\n\nData / context:\n```\n"
            context_str += json.dumps(task.context, indent=2)
            context_str += "\n```"

        constraints_str = ""
        if task.constraints:
            constraints_str = "\n\nAnalysis requirements:\n- " + "\n- ".join(task.constraints)

        user_message = (
            f"Analyse the following:\n\n"
            f"{task.description}"
            f"{context_str}"
            f"{constraints_str}"
        )

        raw_response = await self._invoke_llm(
            system_prompt=ANALYST_SYSTEM_PROMPT,
            user_message=user_message,
            model=self._model,
            temperature=0.2,
        )

        structured = self._parse_analyst_output(raw_response)

        return AgentResult(
            task_id=task.task_id,
            agent_name=self.name,
            status=TaskStatus.COMPLETED,
            output=structured.get("summary", raw_response),
            structured_data=structured,
            confidence=self._compute_analysis_confidence(structured),
            metadata={
                "model": self._model,
                "analysis_type": structured.get("analysis_type", "unknown"),
                "num_insights": len(structured.get("insights", [])),
            },
        )

    # -- Internals -------------------------------------------------------------

    @staticmethod
    def _parse_analyst_output(raw: str) -> dict[str, Any]:
        """Parse JSON output from the analyst LLM."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("analyst_json_parse_failed", raw_length=len(raw))
            return {
                "analysis_type": "unknown",
                "summary": raw,
                "key_metrics": [],
                "trends": [],
                "insights": [],
                "anomalies": [],
                "recommendations": [],
            }

    @staticmethod
    def _compute_analysis_confidence(structured: dict[str, Any]) -> float:
        """Derive confidence from analysis completeness."""
        score = 0.5
        if structured.get("key_metrics"):
            score += 0.1
        if structured.get("trends"):
            score += 0.1
        if structured.get("insights"):
            score += 0.1
        if structured.get("recommendations"):
            score += 0.1
        if structured.get("methodology"):
            score += 0.1
        return min(score, 1.0)
