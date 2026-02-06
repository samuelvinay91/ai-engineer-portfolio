"""Shared test fixtures for the Deep Research test suite.

All tests run with **mock** Anthropic / Tavily clients so no real API calls
are made.  The fixtures in this module wire up realistic but deterministic
responses that exercise every code-path in the reasoning, planning, and
research modules.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio

from deep_research.config import Settings


# ---------------------------------------------------------------------------
# Settings fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def settings() -> Settings:
    """Return a ``Settings`` instance with safe test defaults."""
    return Settings(
        reasoning_model="test-model",
        fast_model="test-model-fast",
        anthropic_api_key="test-key",
        openai_api_key="test-key",
        tavily_api_key="test-tavily-key",
        thinking_budget_tokens=1_024,
        max_research_depth=2,
        max_parallel_searches=2,
        tot_branching_factor=2,
        tot_max_depth=2,
        confidence_threshold=0.7,
        environment="development",
        log_level="DEBUG",
    )


# ---------------------------------------------------------------------------
# Mock Anthropic response helpers
# ---------------------------------------------------------------------------

@dataclass
class _MockUsage:
    input_tokens: int = 100
    output_tokens: int = 200


@dataclass
class _MockTextBlock:
    type: str = "text"
    text: str = ""


@dataclass
class _MockThinkingBlock:
    type: str = "thinking"
    thinking: str = ""


@dataclass
class _MockMessage:
    content: list[Any] = field(default_factory=list)
    usage: _MockUsage = field(default_factory=_MockUsage)


def make_mock_response(
    text: str,
    thinking: str = "",
    input_tokens: int = 100,
    output_tokens: int = 200,
) -> _MockMessage:
    """Build a mock ``anthropic.types.Message``-like object."""
    blocks: list[Any] = []
    if thinking:
        blocks.append(_MockThinkingBlock(thinking=thinking))
    blocks.append(_MockTextBlock(text=text))
    return _MockMessage(
        content=blocks,
        usage=_MockUsage(input_tokens=input_tokens, output_tokens=output_tokens),
    )


# ---------------------------------------------------------------------------
# Mock Anthropic client fixture
# ---------------------------------------------------------------------------

COT_RESPONSE_TEXT = """\
[Step 1]
Description: Identify the core components of the question
Reasoning: The question asks about quantum computing advantages. We need to break this into hardware, software, and application dimensions.
Confidence: 0.85

[Step 2]
Description: Analyse hardware advantages
Reasoning: Quantum computers use qubits that can exist in superposition, enabling parallel computation on an exponential scale compared to classical bits.
Confidence: 0.9

[Final Answer]
Quantum computing offers exponential speedup for certain problem classes including factoring, simulation, and optimisation through superposition and entanglement.
Overall Confidence: 0.88
"""

TOT_GENERATION_TEXT = """\
[Approach 1]
Thought: Focus on the computational complexity advantages -- problems in BQP that are not in P.
Rationale: This is the most rigorous way to characterise quantum advantage.

[Approach 2]
Thought: Focus on practical near-term applications like quantum chemistry simulation.
Rationale: This grounds the discussion in real-world impact.
"""

TOT_EVALUATION_TEXT = """\
Approach 1: 0.85
Approach 2: 0.75
"""

DIRECT_RESPONSE_TEXT = "Quantum computing provides speedup for certain algorithms."

SELF_REFLECTION_TEXT = """\
[Critique]
The answer is solid but could mention quantum error correction as a current limitation.

[Revised Answer]
Quantum computing offers exponential speedup for specific problems (factoring, simulation) but practical advantage is currently limited by error rates and qubit coherence times.

[Confidence]
0.82
"""

PLAN_RESPONSE_TEXT = """\
[SQ-1]
Question: What are the theoretical foundations of quantum advantage?
Rationale: Understanding the theory is essential before discussing applications.
Depends on: NONE
Priority: 1

[SQ-2]
Question: What are the current practical applications of quantum computing?
Rationale: Grounds the research in real-world impact.
Depends on: SQ-1
Priority: 2

[SQ-3]
Question: What are the main challenges and limitations?
Rationale: A balanced report must address limitations.
Depends on: NONE
Priority: 2
"""

ANALYSIS_RESPONSE_TEXT = """\
- Quantum computers leverage superposition and entanglement for computation.
- Shor's algorithm provides exponential speedup for integer factoring.
- Current quantum computers have 50-1000+ qubits but high error rates.
"""

VERIFY_RESPONSE_TEXT = "VERIFIED"


@pytest.fixture()
def mock_anthropic_client() -> AsyncMock:
    """Return a mock ``anthropic.AsyncAnthropic`` whose ``.messages.create``
    returns appropriate responses based on the prompt content.
    """
    client = AsyncMock()

    async def _create(**kwargs: Any) -> _MockMessage:
        user_content = ""
        for msg in kwargs.get("messages", []):
            if msg.get("role") == "user":
                user_content = msg.get("content", "")

        # Route to the appropriate mock based on prompt content
        if "break it down" in (kwargs.get("system", "") or "").lower() or "sub-question" in user_content.lower():
            return make_mock_response(PLAN_RESPONSE_TEXT)
        if "extract key findings" in (kwargs.get("system", "") or "").lower() or "Extract key findings" in (kwargs.get("system", "") or ""):
            return make_mock_response(ANALYSIS_RESPONSE_TEXT)
        if "VERIFIED" in user_content or "contradictions" in (kwargs.get("system", "") or "").lower():
            return make_mock_response(VERIFY_RESPONSE_TEXT)
        if "critical reviewer" in (kwargs.get("system", "") or "").lower():
            return make_mock_response(SELF_REFLECTION_TEXT, thinking="Reflecting...")
        if "generate" in (kwargs.get("system", "") or "").lower() and "approach" in user_content.lower():
            return make_mock_response(TOT_GENERATION_TEXT)
        if "evaluat" in (kwargs.get("system", "") or "").lower():
            return make_mock_response(TOT_EVALUATION_TEXT)
        if "report" in (kwargs.get("system", "") or "").lower():
            return make_mock_response(
                "[TITLE]\nQuantum Computing Research Report\n\n"
                "[EXECUTIVE SUMMARY]\nThis report covers quantum computing.\n\n"
                "[KEY FINDINGS]\n- Quantum speedup is real for specific problems\n"
                "- Error correction remains a challenge\n\n"
                "[SECTION: Theoretical Foundations]\nQuantum advantage stems from superposition.\n\n"
                "[CONFIDENCE ASSESSMENT]\nHigh confidence in theoretical claims.\n\n"
                "[FURTHER RESEARCH]\n- Investigate fault-tolerant quantum computing\n",
                thinking="Synthesising report...",
            )
        if "step by step" in (kwargs.get("system", "") or "").lower():
            return make_mock_response(COT_RESPONSE_TEXT, thinking="Deep thinking here...")
        if "synthesise" in (kwargs.get("system", "") or "").lower() or "final answer" in user_content.lower():
            return make_mock_response(
                "Quantum computing provides exponential speedup for BQP problems.",
                thinking="Synthesising ToT result...",
            )
        # Default: direct response
        return make_mock_response(DIRECT_RESPONSE_TEXT)

    client.messages.create = AsyncMock(side_effect=_create)
    return client


# ---------------------------------------------------------------------------
# FastAPI test client fixture
# ---------------------------------------------------------------------------

@pytest_asyncio.fixture()
async def api_client(settings: Settings, mock_anthropic_client: AsyncMock):
    """Return an ``httpx.AsyncClient`` wired to the test app."""
    import httpx
    from deep_research.api import create_app

    app = create_app(settings=settings)
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client
