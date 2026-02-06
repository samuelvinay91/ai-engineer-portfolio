"""Report Generator -- structured research reports with citations.

Produces a comprehensive research report from the findings, reasoning traces,
and source data collected during a deep-research session.

Output formats
--------------
* **Markdown** -- human-readable, suitable for rendering in a browser or PDF.
* **Structured JSON** -- machine-readable, for downstream programmatic use.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any

import anthropic
import structlog

from deep_research.config import Settings, get_settings

if TYPE_CHECKING:
    from deep_research.researcher import ResearchTask

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SourceCitation:
    """A single cited source with a relevance score."""

    title: str
    url: str
    relevance_score: float = 0.0
    snippet: str = ""


@dataclass
class ReportSection:
    """One section of the research report."""

    title: str
    content: str
    citations: list[SourceCitation] = field(default_factory=list)
    confidence: float = 0.0


@dataclass
class ResearchReport:
    """The final deliverable of a deep-research session."""

    title: str = ""
    executive_summary: str = ""
    key_findings: list[str] = field(default_factory=list)
    sections: list[ReportSection] = field(default_factory=list)
    all_citations: list[SourceCitation] = field(default_factory=list)
    confidence_assessment: str = ""
    further_research: list[str] = field(default_factory=list)
    generated_at: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    # -- Output helpers ----------------------------------------------------

    def to_markdown(self) -> str:
        """Render the report as a Markdown string."""
        parts: list[str] = []
        parts.append(f"# {self.title}\n")
        parts.append(f"## Executive Summary\n\n{self.executive_summary}\n")

        if self.key_findings:
            parts.append("## Key Findings\n")
            for i, finding in enumerate(self.key_findings, 1):
                parts.append(f"{i}. {finding}")
            parts.append("")

        for section in self.sections:
            parts.append(f"## {section.title}\n\n{section.content}\n")
            if section.citations:
                parts.append("**Sources:**\n")
                for cite in section.citations:
                    parts.append(
                        f"- [{cite.title}]({cite.url}) "
                        f"(relevance: {cite.relevance_score:.0%})"
                    )
                parts.append("")

        if self.confidence_assessment:
            parts.append(f"## Confidence Assessment\n\n{self.confidence_assessment}\n")

        if self.further_research:
            parts.append("## Areas for Further Research\n")
            for item in self.further_research:
                parts.append(f"- {item}")
            parts.append("")

        if self.all_citations:
            parts.append("## References\n")
            for i, cite in enumerate(self.all_citations, 1):
                parts.append(f"{i}. [{cite.title}]({cite.url})")
            parts.append("")

        return "\n".join(parts)

    def to_json(self) -> str:
        """Serialise the report as a JSON string."""
        return json.dumps(asdict(self), indent=2, default=str)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

REPORT_PROMPT = """\
You are a professional research report writer.  Using the research findings
below, produce a structured research report.

ORIGINAL QUESTION: {question}

FINDINGS:
{findings}

REASONING SUMMARIES:
{reasoning}

SOURCES:
{sources}

Output the report in the following EXACT format (preserve section headers):

[TITLE]
<a descriptive title for the report>

[EXECUTIVE SUMMARY]
<2-3 paragraph executive summary>

[KEY FINDINGS]
- <finding 1>
- <finding 2>
...

[SECTION: <section title>]
<detailed analysis>

(Repeat [SECTION: ...] as needed)

[CONFIDENCE ASSESSMENT]
<assessment of overall confidence, noting areas of high and low certainty>

[FURTHER RESEARCH]
- <area 1>
- <area 2>
...
"""


# ---------------------------------------------------------------------------
# Report Generator
# ---------------------------------------------------------------------------

class ReportGenerator:
    """Generates structured research reports from deep-research results.

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

    async def generate(self, task: ResearchTask) -> ResearchReport:
        """Generate a complete report from a ``ResearchTask``."""
        log = logger.bind(task_id=task.id)
        log.info("report.generate.start")
        t0 = time.perf_counter()

        findings_text = self._format_findings(task)
        reasoning_text = self._format_reasoning(task)
        sources_text = self._format_sources(task)

        response = await self._client.messages.create(
            model=self.settings.reasoning_model,
            max_tokens=8_192,
            temperature=1,
            thinking={"type": "enabled", "budget_tokens": self.settings.thinking_budget_tokens},
            system="You are an expert research report writer producing clear, well-cited reports.",
            messages=[{
                "role": "user",
                "content": REPORT_PROMPT.format(
                    question=task.question,
                    findings=findings_text,
                    reasoning=reasoning_text,
                    sources=sources_text,
                ),
            }],
        )
        text = "\n".join(b.text for b in response.content if b.type == "text")

        report = self._parse_report(text, task)
        dur = (time.perf_counter() - t0) * 1_000
        log.info("report.generate.done", sections=len(report.sections), duration_ms=round(dur, 1))
        return report

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _format_findings(task: ResearchTask) -> str:
        if not task.findings:
            return "(no findings)"
        return "\n\n".join(
            f"**{f.claim}**\n{f.evidence}" for f in task.findings
        )

    @staticmethod
    def _format_reasoning(task: ResearchTask) -> str:
        if not task.reasoning_results:
            return "(no reasoning traces)"
        parts: list[str] = []
        for r in task.reasoning_results:
            parts.append(f"Strategy: {r.strategy.value} | Confidence: {r.confidence:.2f}")
            parts.append(r.answer[:500])
            parts.append("---")
        return "\n".join(parts)

    @staticmethod
    def _format_sources(task: ResearchTask) -> str:
        if not task.search_results:
            return "(no sources)"
        seen: set[str] = set()
        lines: list[str] = []
        for sr in task.search_results:
            if sr.url not in seen:
                seen.add(sr.url)
                lines.append(f"- {sr.title}: {sr.url} (relevance: {sr.relevance_score:.2f})")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Parsing
    # ------------------------------------------------------------------

    def _parse_report(self, text: str, task: ResearchTask) -> ResearchReport:
        """Parse the structured report output from the model."""
        report = ResearchReport()

        # Title
        if "[TITLE]" in text:
            title_block = text.split("[TITLE]")[1].split("[")[0].strip()
            report.title = title_block
        else:
            report.title = f"Research Report: {task.question[:80]}"

        # Executive summary
        if "[EXECUTIVE SUMMARY]" in text:
            summary_block = text.split("[EXECUTIVE SUMMARY]")[1].split("[")[0].strip()
            report.executive_summary = summary_block

        # Key findings
        if "[KEY FINDINGS]" in text:
            findings_block = text.split("[KEY FINDINGS]")[1].split("[")[0].strip()
            report.key_findings = [
                line.lstrip("- ").strip()
                for line in findings_block.split("\n")
                if line.strip().startswith("-")
            ]

        # Sections
        import re
        section_pattern = re.compile(
            r"\[SECTION:\s*(.+?)\]\s*\n([\s\S]*?)(?=\[SECTION:|\[CONFIDENCE|\[FURTHER|\Z)"
        )
        for m in section_pattern.finditer(text):
            section_title = m.group(1).strip()
            section_content = m.group(2).strip()
            section_citations = self._extract_citations_for_section(
                section_content, task,
            )
            report.sections.append(ReportSection(
                title=section_title,
                content=section_content,
                citations=section_citations,
            ))

        # Confidence assessment
        if "[CONFIDENCE ASSESSMENT]" in text:
            conf_block = text.split("[CONFIDENCE ASSESSMENT]")[1].split("[")[0].strip()
            report.confidence_assessment = conf_block

        # Further research
        if "[FURTHER RESEARCH]" in text:
            further_block = text.split("[FURTHER RESEARCH]")[1].strip()
            report.further_research = [
                line.lstrip("- ").strip()
                for line in further_block.split("\n")
                if line.strip().startswith("-")
            ]

        # Collect all citations
        report.all_citations = self._deduplicate_citations(task)

        return report

    @staticmethod
    def _extract_citations_for_section(
        content: str,
        task: ResearchTask,
    ) -> list[SourceCitation]:
        """Match URLs mentioned in section content to known search results."""
        citations: list[SourceCitation] = []
        for sr in task.search_results:
            if sr.url in content or sr.title.lower() in content.lower():
                citations.append(SourceCitation(
                    title=sr.title,
                    url=sr.url,
                    relevance_score=sr.relevance_score,
                    snippet=sr.snippet[:200],
                ))
        return citations

    @staticmethod
    def _deduplicate_citations(task: ResearchTask) -> list[SourceCitation]:
        """Build a deduplicated list of all sources used in the research."""
        seen: set[str] = set()
        citations: list[SourceCitation] = []
        for sr in task.search_results:
            if sr.url not in seen:
                seen.add(sr.url)
                citations.append(SourceCitation(
                    title=sr.title,
                    url=sr.url,
                    relevance_score=sr.relevance_score,
                    snippet=sr.snippet[:200],
                ))
        citations.sort(key=lambda c: c.relevance_score, reverse=True)
        return citations
