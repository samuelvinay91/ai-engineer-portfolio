"""Fact Checker Agent -- cross-references claims across sources.

This agent adds a verification layer to the pipeline.  It takes the
synthesised answer together with the original search results and:

1. Identifies the key factual claims in the answer.
2. Cross-references each claim against the available sources.
3. Assigns a per-claim confidence score.
4. Flags conflicting information or potential misinformation.
5. Returns an overall verification report that can be surfaced to the user.
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from ask_the_web.agents.searcher import SearchResult
from ask_the_web.config import Settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ClaimVerification(BaseModel):
    """Verification result for a single factual claim."""

    claim: str = Field(description="The factual claim being verified.")
    supported_by: list[int] = Field(
        default_factory=list,
        description="1-based indices of sources that support the claim.",
    )
    contradicted_by: list[int] = Field(
        default_factory=list,
        description="1-based indices of sources that contradict the claim.",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence that the claim is accurate (0-1).",
    )
    status: str = Field(
        default="unverified",
        description="One of: verified, partially_verified, contradicted, unverified.",
    )
    note: str = Field(default="", description="Optional note about conflicts or caveats.")


class FactCheckReport(BaseModel):
    """Full fact-check report for a synthesised answer."""

    claims: list[ClaimVerification] = Field(default_factory=list)
    overall_confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.8,
        description="Aggregate confidence across all claims.",
    )
    conflicts_found: int = Field(
        default=0, description="Number of claims with conflicting sources."
    )
    flags: list[str] = Field(
        default_factory=list,
        description="Human-readable warnings or flags.",
    )


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

FACT_CHECK_PROMPT = """\
You are a meticulous fact-checker.  You are given an AI-generated answer and
the original web sources it was derived from.  Your job is to verify the claims
in the answer against the sources.

## Answer to verify
{answer}

## Sources
{sources}

## Instructions

1. Identify every distinct factual claim in the answer.
2. For each claim, check which source(s) support or contradict it.
3. Assign a confidence score (0-1):
   - 1.0 = explicitly stated in multiple independent sources
   - 0.8 = stated in one reliable source
   - 0.5 = partially supported / ambiguous
   - 0.2 = only weakly implied
   - 0.0 = contradicted by sources
4. Flag any claim that is contradicted or not supported by any source.

Return valid JSON matching this schema:
{{
  "claims": [
    {{
      "claim": "...",
      "supported_by": [1, 3],
      "contradicted_by": [],
      "confidence": 0.9,
      "status": "verified",
      "note": ""
    }}
  ],
  "overall_confidence": 0.85,
  "conflicts_found": 0,
  "flags": ["..."]
}}

If no claims can be extracted (e.g. creative content), return:
{{"claims": [], "overall_confidence": 1.0, "conflicts_found": 0, "flags": []}}
"""


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class FactCheckerAgent:
    """Cross-references synthesised answer claims against search sources."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm = ChatAnthropic(
            model=settings.fact_checker_model,
            anthropic_api_key=settings.anthropic_api_key,
            temperature=0.0,  # deterministic fact checking
            max_tokens=4096,
        )

    async def check(
        self,
        answer: str,
        search_results: list[SearchResult],
    ) -> FactCheckReport:
        """Verify *answer* against *search_results*.

        Parameters
        ----------
        answer:
            The synthesised markdown answer (with inline citations).
        search_results:
            The original search results used to generate the answer.

        Returns
        -------
        FactCheckReport
        """
        if not search_results:
            logger.info("fact_check_skipped", reason="no_sources")
            return FactCheckReport(
                overall_confidence=0.5,
                flags=["No sources available for verification."],
            )

        sources_text = self._format_sources(search_results)
        prompt = FACT_CHECK_PROMPT.format(answer=answer, sources=sources_text)

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content="Verify the claims now."),
        ]

        try:
            response = await self._llm.ainvoke(messages)
            raw: str = response.content  # type: ignore[assignment]
            report = self._parse_response(raw)
        except Exception:
            logger.exception("fact_check_llm_error")
            report = FactCheckReport(
                overall_confidence=0.5,
                flags=["Fact-checking failed; treat answer with caution."],
            )

        logger.info(
            "fact_check_complete",
            claims_count=len(report.claims),
            overall_confidence=report.overall_confidence,
            conflicts=report.conflicts_found,
        )
        return report

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _format_sources(results: list[SearchResult]) -> str:
        """Format sources for the fact-check prompt."""
        parts: list[str] = []
        for i, r in enumerate(results, 1):
            parts.append(
                f"[Source {i}] {r.title}\n"
                f"URL: {r.url}\n"
                f"Content: {r.content[:1200]}\n"
            )
        return "\n---\n".join(parts)

    @staticmethod
    def _parse_response(raw: str) -> FactCheckReport:
        """Parse LLM JSON response into a FactCheckReport."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
        cleaned = cleaned.strip()

        try:
            data: dict[str, Any] = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("fact_check_json_parse_failed", raw=raw[:300])
            return FactCheckReport(
                overall_confidence=0.5,
                flags=["Failed to parse fact-check results."],
            )

        claims: list[ClaimVerification] = []
        for item in data.get("claims", []):
            claims.append(
                ClaimVerification(
                    claim=item.get("claim", ""),
                    supported_by=item.get("supported_by", []),
                    contradicted_by=item.get("contradicted_by", []),
                    confidence=min(max(float(item.get("confidence", 0.5)), 0.0), 1.0),
                    status=item.get("status", "unverified"),
                    note=item.get("note", ""),
                )
            )

        return FactCheckReport(
            claims=claims,
            overall_confidence=min(
                max(float(data.get("overall_confidence", 0.5)), 0.0), 1.0
            ),
            conflicts_found=int(data.get("conflicts_found", 0)),
            flags=data.get("flags", []),
        )
