"""Tests for the reasoning engine -- CoT, ToT, direct, and self-reflection.

All tests use mock API clients so no real LLM calls are made.
"""

from __future__ import annotations

import pytest

from deep_research.config import Settings
from deep_research.reasoning import (
    ReasoningEngine,
    ReasoningResult,
    ReasoningStep,
    ReasoningStrategy,
    ThoughtBranch,
)


# ---------------------------------------------------------------------------
# Chain-of-Thought tests
# ---------------------------------------------------------------------------

class TestChainOfThought:
    """Tests for the CoT reasoning strategy."""

    @pytest.mark.asyncio
    async def test_cot_returns_result_with_steps(
        self,
        settings: Settings,
        mock_anthropic_client,
    ):
        engine = ReasoningEngine(settings=settings, client=mock_anthropic_client)
        result = await engine.reason(
            "What are the advantages of quantum computing?",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
        )

        assert isinstance(result, ReasoningResult)
        assert result.strategy == ReasoningStrategy.CHAIN_OF_THOUGHT
        assert result.answer  # non-empty
        assert result.confidence > 0.0
        assert len(result.steps) >= 1
        assert result.total_duration_ms > 0

    @pytest.mark.asyncio
    async def test_cot_parses_steps_correctly(
        self,
        settings: Settings,
        mock_anthropic_client,
    ):
        engine = ReasoningEngine(settings=settings, client=mock_anthropic_client)
        result = await engine.reason(
            "Explain quantum advantage.",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
        )

        # The mock returns two steps (Step 1 and Step 2)
        step_numbers = [s.step_number for s in result.steps]
        assert 1 in step_numbers
        # Confidence should be parsed from the formatted output
        for step in result.steps:
            assert 0.0 <= step.confidence <= 1.0

    @pytest.mark.asyncio
    async def test_cot_captures_thinking_content(
        self,
        settings: Settings,
        mock_anthropic_client,
    ):
        engine = ReasoningEngine(settings=settings, client=mock_anthropic_client)
        result = await engine.reason(
            "What is quantum computing?",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
        )

        # The mock returns a thinking block
        assert result.thinking_content  # non-empty

    @pytest.mark.asyncio
    async def test_cot_self_reflection_triggered_on_low_confidence(
        self,
        settings: Settings,
        mock_anthropic_client,
    ):
        """When confidence is below the threshold, self-reflection should fire."""
        # Set a very high threshold so reflection is always triggered
        settings.confidence_threshold = 0.99
        engine = ReasoningEngine(settings=settings, client=mock_anthropic_client)
        result = await engine.reason(
            "Hard question requiring reflection.",
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
        )

        # Reflection should add at least one more step
        reflection_steps = [
            s for s in result.steps if "reflection" in s.description.lower()
        ]
        assert len(reflection_steps) >= 1


# ---------------------------------------------------------------------------
# Tree-of-Thought tests
# ---------------------------------------------------------------------------

class TestTreeOfThought:
    """Tests for the ToT reasoning strategy."""

    @pytest.mark.asyncio
    async def test_tot_returns_result(
        self,
        settings: Settings,
        mock_anthropic_client,
    ):
        engine = ReasoningEngine(settings=settings, client=mock_anthropic_client)
        result = await engine.reason(
            "What is the best approach to quantum error correction?",
            strategy=ReasoningStrategy.TREE_OF_THOUGHT,
        )

        assert isinstance(result, ReasoningResult)
        assert result.strategy == ReasoningStrategy.TREE_OF_THOUGHT
        assert result.answer
        assert len(result.steps) >= 2  # generation + evaluation + synthesis

    @pytest.mark.asyncio
    async def test_tot_explores_multiple_branches(
        self,
        settings: Settings,
        mock_anthropic_client,
    ):
        engine = ReasoningEngine(settings=settings, client=mock_anthropic_client)
        result = await engine.reason(
            "Compare different quantum computing paradigms.",
            strategy=ReasoningStrategy.TREE_OF_THOUGHT,
        )

        # Check that generation steps created branches
        gen_steps = [
            s for s in result.steps if "generate" in s.description.lower()
        ]
        assert len(gen_steps) >= 1

        # Check evaluation steps scored branches
        eval_steps = [
            s for s in result.steps if "evaluate" in s.description.lower()
        ]
        assert len(eval_steps) >= 1

    @pytest.mark.asyncio
    async def test_tot_tracks_token_usage(
        self,
        settings: Settings,
        mock_anthropic_client,
    ):
        engine = ReasoningEngine(settings=settings, client=mock_anthropic_client)
        result = await engine.reason(
            "A question for ToT.",
            strategy=ReasoningStrategy.TREE_OF_THOUGHT,
        )

        assert result.total_input_tokens > 0
        assert result.total_output_tokens > 0


# ---------------------------------------------------------------------------
# Direct (baseline) tests
# ---------------------------------------------------------------------------

class TestDirectReasoning:
    """Tests for the direct (baseline) reasoning strategy."""

    @pytest.mark.asyncio
    async def test_direct_returns_result(
        self,
        settings: Settings,
        mock_anthropic_client,
    ):
        engine = ReasoningEngine(settings=settings, client=mock_anthropic_client)
        result = await engine.reason(
            "What is quantum computing?",
            strategy=ReasoningStrategy.DIRECT,
        )

        assert result.strategy == ReasoningStrategy.DIRECT
        assert result.answer
        assert result.confidence == 0.5  # default for direct
        assert len(result.steps) == 1

    @pytest.mark.asyncio
    async def test_direct_no_extended_thinking_by_default(
        self,
        settings: Settings,
        mock_anthropic_client,
    ):
        engine = ReasoningEngine(settings=settings, client=mock_anthropic_client)
        result = await engine.reason(
            "Simple question.",
            strategy=ReasoningStrategy.DIRECT,
            use_extended_thinking=False,
        )

        # Should still return a valid result
        assert result.answer


# ---------------------------------------------------------------------------
# Strategy comparison tests
# ---------------------------------------------------------------------------

class TestCompareStrategies:
    """Tests for the strategy comparison feature."""

    @pytest.mark.asyncio
    async def test_compare_returns_all_strategies(
        self,
        settings: Settings,
        mock_anthropic_client,
    ):
        engine = ReasoningEngine(settings=settings, client=mock_anthropic_client)
        comparison = await engine.compare_strategies(
            "Compare quantum vs classical computing.",
            strategies=[ReasoningStrategy.DIRECT, ReasoningStrategy.CHAIN_OF_THOUGHT],
        )

        assert "direct" in comparison
        assert "chain_of_thought" in comparison
        assert isinstance(comparison["direct"], ReasoningResult)
        assert isinstance(comparison["chain_of_thought"], ReasoningResult)

    @pytest.mark.asyncio
    async def test_compare_all_strategies_by_default(
        self,
        settings: Settings,
        mock_anthropic_client,
    ):
        engine = ReasoningEngine(settings=settings, client=mock_anthropic_client)
        comparison = await engine.compare_strategies("A broad question.")

        assert len(comparison) == len(ReasoningStrategy)


# ---------------------------------------------------------------------------
# Parsing and helper tests
# ---------------------------------------------------------------------------

class TestParsingHelpers:
    """Unit tests for internal parsing logic."""

    def test_parse_cot_steps_structured(self):
        text = (
            "[Step 1]\n"
            "Description: First step\n"
            "Reasoning: Some reasoning here\n"
            "Confidence: 0.85\n\n"
            "[Step 2]\n"
            "Description: Second step\n"
            "Reasoning: More reasoning\n"
            "Confidence: 0.90\n\n"
            "[Final Answer]\nThe answer.\nOverall Confidence: 0.88"
        )
        steps = ReasoningEngine._parse_cot_steps(text)
        assert len(steps) == 2
        assert steps[0].step_number == 1
        assert steps[0].confidence == 0.85
        assert steps[1].step_number == 2

    def test_parse_cot_steps_fallback(self):
        """When the model doesn't follow the format, a single step is created."""
        text = "Just some unstructured reasoning without step markers."
        steps = ReasoningEngine._parse_cot_steps(text)
        assert len(steps) == 1
        assert steps[0].description == "Reasoning (unstructured)"

    def test_parse_final_answer(self):
        text = "Some preamble\n[Final Answer]\nThe real answer.\nOverall Confidence: 0.92"
        answer, confidence = ReasoningEngine._parse_final_answer(text)
        assert answer == "The real answer."
        assert confidence == 0.92

    def test_parse_final_answer_missing(self):
        text = "Just an answer without a marker."
        answer, confidence = ReasoningEngine._parse_final_answer(text)
        assert answer == text.strip()
        assert confidence == 0.5

    def test_select_best_leaf(self):
        root = ThoughtBranch(branch_id="root", parent_id=None, depth=0, thought="root")
        child_a = ThoughtBranch(branch_id="a", parent_id="root", depth=1, thought="a", evaluation_score=0.6)
        child_b = ThoughtBranch(branch_id="b", parent_id="root", depth=1, thought="b", evaluation_score=0.9)
        root.children = [child_a, child_b]

        engine = ReasoningEngine.__new__(ReasoningEngine)
        best = engine._select_best_leaf(root)
        assert best.branch_id == "b"

    def test_apply_scores(self):
        branches = [
            ThoughtBranch(branch_id="1", parent_id=None, depth=0, thought="a"),
            ThoughtBranch(branch_id="2", parent_id=None, depth=0, thought="b"),
        ]
        eval_text = "Approach 1: 0.8\nApproach 2: 0.65"
        ReasoningEngine._apply_scores(branches, eval_text)
        assert branches[0].evaluation_score == 0.8
        assert branches[1].evaluation_score == 0.65
