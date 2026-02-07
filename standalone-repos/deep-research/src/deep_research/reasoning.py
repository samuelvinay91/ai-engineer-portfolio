"""Reasoning Engine -- Chain-of-Thought, Tree-of-Thought, and self-reflection.

This module implements three complementary reasoning strategies that exploit
*inference-time compute scaling*: the idea that spending more tokens at
inference time (rather than training time) yields better answers on complex
tasks.

Strategies
----------
1. **Chain-of-Thought (CoT)** -- sequential step-by-step reasoning.
2. **Tree-of-Thought (ToT)** -- explores multiple reasoning branches in
   parallel, scores them, and prunes weak paths.
3. **Self-Reflection** -- the model critiques its own answer and iterates
   until a confidence threshold is met.

All strategies expose a unified ``ReasoningResult`` that captures the full
reasoning trace so callers can inspect *why* the model reached a conclusion.
"""

from __future__ import annotations

import asyncio
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

class ReasoningStrategy(str, Enum):
    """Supported reasoning strategies."""

    DIRECT = "direct"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    TREE_OF_THOUGHT = "tree_of_thought"


@dataclass
class ReasoningStep:
    """A single step inside a reasoning trace."""

    step_number: int
    description: str
    content: str
    confidence: float = 0.0
    duration_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ThoughtBranch:
    """A branch in a Tree-of-Thought exploration."""

    branch_id: str
    parent_id: str | None
    depth: int
    thought: str
    evaluation_score: float = 0.0
    children: list[ThoughtBranch] = field(default_factory=list)
    is_terminal: bool = False
    is_pruned: bool = False


@dataclass
class ReasoningResult:
    """Complete result of a reasoning session."""

    strategy: ReasoningStrategy
    query: str
    answer: str
    confidence: float
    steps: list[ReasoningStep] = field(default_factory=list)
    thinking_content: str = ""
    total_duration_ms: float = 0.0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

COT_SYSTEM_PROMPT = """\
You are an expert analytical reasoner.  When presented with a question you MUST
think through it step by step.

For EACH step you produce, output it in the following format:

[Step N]
Description: <one-line summary of what this step does>
Reasoning: <detailed reasoning for this step>
Confidence: <a float between 0.0 and 1.0 representing how confident you are>

After all steps, output your final answer in this format:

[Final Answer]
<your concise, well-supported answer>
Overall Confidence: <float 0.0-1.0>
"""

TOT_GENERATION_PROMPT = """\
You are exploring multiple reasoning paths for a complex question.
Given the current reasoning context, generate {branching_factor} DISTINCT
approaches or hypotheses for the next step of reasoning.

Current context:
{context}

Question: {question}

For each approach, output:

[Approach N]
Thought: <the reasoning approach>
Rationale: <why this approach might work>
"""

TOT_EVALUATION_PROMPT = """\
You are evaluating reasoning approaches for quality and promise.
Score each approach from 0.0 to 1.0 based on:
- Logical soundness
- Relevance to the question
- Likelihood of leading to a correct answer

Approaches to evaluate:
{approaches}

Original question: {question}

For each approach output exactly one line:
Approach N: <score>
"""

SELF_REFLECTION_PROMPT = """\
You are a critical reviewer.  Examine the following answer and reasoning for
the given question.  Identify any:
1. Logical errors or gaps
2. Unsupported claims
3. Missing perspectives
4. Factual inaccuracies

Question: {question}

Answer: {answer}

Reasoning trace:
{reasoning_trace}

Provide your critique and then give a revised answer if needed.

[Critique]
<your critique>

[Revised Answer]
<revised answer or "NO REVISION NEEDED" if the original is sound>

[Confidence]
<float 0.0-1.0>
"""

DIRECT_PROMPT = """\
Answer the following question directly and concisely.

Question: {question}
"""


# ---------------------------------------------------------------------------
# Reasoning Engine
# ---------------------------------------------------------------------------

class ReasoningEngine:
    """Orchestrates different reasoning strategies using Claude models.

    The engine supports Claude's *extended thinking* mode which allocates a
    dedicated thinking budget before the model produces its visible answer.
    This is a concrete implementation of inference-time compute scaling --
    giving the model more compute at inference time to tackle harder problems.

    Parameters
    ----------
    settings:
        Application settings.  When ``None`` the global singleton is used.
    client:
        An ``anthropic.AsyncAnthropic`` client.  Injected for testability.
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def reason(
        self,
        query: str,
        strategy: ReasoningStrategy = ReasoningStrategy.CHAIN_OF_THOUGHT,
        *,
        use_extended_thinking: bool = True,
    ) -> ReasoningResult:
        """Run the chosen reasoning strategy on *query*.

        Parameters
        ----------
        query:
            The question or problem to reason about.
        strategy:
            Which reasoning strategy to use.
        use_extended_thinking:
            Whether to use Claude's extended thinking mode (extra inference-
            time compute).
        """
        dispatch = {
            ReasoningStrategy.DIRECT: self._reason_direct,
            ReasoningStrategy.CHAIN_OF_THOUGHT: self._reason_cot,
            ReasoningStrategy.TREE_OF_THOUGHT: self._reason_tot,
        }

        handler = dispatch[strategy]
        log = logger.bind(strategy=strategy.value, query=query[:120])
        log.info("reasoning.start")
        t0 = time.perf_counter()

        result = await handler(query, use_extended_thinking=use_extended_thinking)

        result.total_duration_ms = (time.perf_counter() - t0) * 1_000
        log.info(
            "reasoning.complete",
            confidence=result.confidence,
            steps=len(result.steps),
            duration_ms=round(result.total_duration_ms, 1),
        )
        return result

    async def compare_strategies(
        self,
        query: str,
        strategies: list[ReasoningStrategy] | None = None,
    ) -> dict[str, ReasoningResult]:
        """Run multiple strategies concurrently and return a comparison dict.

        This is useful for demonstrating how different inference-time
        techniques produce different reasoning traces and confidence levels.
        """
        strategies = strategies or list(ReasoningStrategy)
        tasks = {
            s.value: self.reason(query, strategy=s)
            for s in strategies
        }
        results: dict[str, ReasoningResult] = {}
        for name, coro in tasks.items():
            results[name] = await coro  # sequential to avoid rate limits
        return results

    # ------------------------------------------------------------------
    # Direct (baseline)
    # ------------------------------------------------------------------

    async def _reason_direct(
        self,
        query: str,
        *,
        use_extended_thinking: bool = False,
    ) -> ReasoningResult:
        """Baseline: ask the model directly without explicit CoT scaffolding."""
        t0 = time.perf_counter()
        response = await self._call_model(
            system="You are a helpful assistant. Answer directly and concisely.",
            user=DIRECT_PROMPT.format(question=query),
            use_thinking=use_extended_thinking,
        )
        duration = (time.perf_counter() - t0) * 1_000

        answer_text, thinking_text = self._extract_response(response)

        step = ReasoningStep(
            step_number=1,
            description="Direct answer",
            content=answer_text,
            confidence=0.5,  # no explicit reasoning -- lower default confidence
            duration_ms=duration,
        )
        return ReasoningResult(
            strategy=ReasoningStrategy.DIRECT,
            query=query,
            answer=answer_text,
            confidence=0.5,
            steps=[step],
            thinking_content=thinking_text,
            total_input_tokens=response.usage.input_tokens,
            total_output_tokens=response.usage.output_tokens,
        )

    # ------------------------------------------------------------------
    # Chain-of-Thought
    # ------------------------------------------------------------------

    async def _reason_cot(
        self,
        query: str,
        *,
        use_extended_thinking: bool = True,
    ) -> ReasoningResult:
        """Chain-of-Thought: step-by-step sequential reasoning.

        The model is instructed to decompose its reasoning into numbered steps
        with explicit confidence scores.  When extended thinking is enabled the
        model also gets a hidden "scratch-pad" budget (inference-time scaling).
        """
        t0 = time.perf_counter()
        response = await self._call_model(
            system=COT_SYSTEM_PROMPT,
            user=f"Question: {query}",
            use_thinking=use_extended_thinking,
        )
        duration = (time.perf_counter() - t0) * 1_000

        answer_text, thinking_text = self._extract_response(response)
        steps = self._parse_cot_steps(answer_text)
        final_answer, overall_confidence = self._parse_final_answer(answer_text)

        # --- Self-reflection loop (verify & iterate) ----------------------
        if overall_confidence < self.settings.confidence_threshold:
            logger.info(
                "cot.self_reflection_triggered",
                confidence=overall_confidence,
                threshold=self.settings.confidence_threshold,
            )
            reflection = await self._self_reflect(query, final_answer, steps)
            steps.extend(reflection.steps)
            final_answer = reflection.answer
            overall_confidence = reflection.confidence

        return ReasoningResult(
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            query=query,
            answer=final_answer,
            confidence=overall_confidence,
            steps=steps,
            thinking_content=thinking_text,
            total_input_tokens=response.usage.input_tokens,
            total_output_tokens=response.usage.output_tokens,
        )

    # ------------------------------------------------------------------
    # Tree-of-Thought
    # ------------------------------------------------------------------

    async def _reason_tot(
        self,
        query: str,
        *,
        use_extended_thinking: bool = True,
    ) -> ReasoningResult:
        """Tree-of-Thought: explore multiple reasoning branches in parallel.

        Algorithm
        ---------
        1. Generate *branching_factor* candidate thoughts.
        2. Evaluate / score each candidate.
        3. Keep the top candidates and expand them (recurse up to *max_depth*).
        4. Return the best leaf node's reasoning path.

        This is a concrete implementation of the ToT pattern from *Yao et al.*
        (2023) adapted for LLM API calls.
        """
        root = ThoughtBranch(
            branch_id=str(uuid.uuid4()),
            parent_id=None,
            depth=0,
            thought=f"Root question: {query}",
        )

        all_steps: list[ReasoningStep] = []
        step_counter = 0
        total_input = 0
        total_output = 0

        # BFS-style expansion
        frontier: list[ThoughtBranch] = [root]

        for depth in range(self.settings.tot_max_depth):
            next_frontier: list[ThoughtBranch] = []

            for node in frontier:
                if node.is_pruned:
                    continue

                # 1. Generate candidate branches --------------------------
                step_counter += 1
                gen_t0 = time.perf_counter()
                context = self._build_tot_context(node)
                gen_response = await self._call_model(
                    system="You are a creative problem solver exploring multiple approaches.",
                    user=TOT_GENERATION_PROMPT.format(
                        branching_factor=self.settings.tot_branching_factor,
                        context=context,
                        question=query,
                    ),
                    use_thinking=use_extended_thinking,
                )
                total_input += gen_response.usage.input_tokens
                total_output += gen_response.usage.output_tokens

                gen_text, _ = self._extract_response(gen_response)
                candidates = self._parse_tot_branches(gen_text, node)
                gen_dur = (time.perf_counter() - gen_t0) * 1_000

                all_steps.append(ReasoningStep(
                    step_number=step_counter,
                    description=f"ToT generate (depth={depth}, node={node.branch_id[:8]})",
                    content=gen_text,
                    confidence=0.0,
                    duration_ms=gen_dur,
                    metadata={"branches_generated": len(candidates)},
                ))

                # 2. Evaluate candidates -----------------------------------
                step_counter += 1
                eval_t0 = time.perf_counter()
                approaches_text = "\n".join(
                    f"Approach {i + 1}: {c.thought}" for i, c in enumerate(candidates)
                )
                eval_response = await self._call_model(
                    system="You are a rigorous evaluator of reasoning quality.",
                    user=TOT_EVALUATION_PROMPT.format(
                        approaches=approaches_text,
                        question=query,
                    ),
                    use_thinking=False,  # evaluation is fast
                )
                total_input += eval_response.usage.input_tokens
                total_output += eval_response.usage.output_tokens

                eval_text, _ = self._extract_response(eval_response)
                self._apply_scores(candidates, eval_text)
                eval_dur = (time.perf_counter() - eval_t0) * 1_000

                all_steps.append(ReasoningStep(
                    step_number=step_counter,
                    description=f"ToT evaluate (depth={depth})",
                    content=eval_text,
                    confidence=max((c.evaluation_score for c in candidates), default=0),
                    duration_ms=eval_dur,
                ))

                # 3. Prune weak branches and keep top candidates -----------
                candidates.sort(key=lambda c: c.evaluation_score, reverse=True)
                keep = max(1, self.settings.tot_branching_factor // 2)
                for c in candidates[keep:]:
                    c.is_pruned = True
                node.children = candidates
                next_frontier.extend(c for c in candidates if not c.is_pruned)

            frontier = next_frontier
            if not frontier:
                break

        # Select best leaf
        best = self._select_best_leaf(root)

        # Generate final answer from best path
        step_counter += 1
        path_text = self._trace_path(best)
        synthesis_response = await self._call_model(
            system="You are an expert synthesiser. Given the best reasoning path, produce a final answer.",
            user=(
                f"Question: {query}\n\n"
                f"Best reasoning path:\n{path_text}\n\n"
                "Provide a clear, well-supported final answer."
            ),
            use_thinking=use_extended_thinking,
        )
        total_input += synthesis_response.usage.input_tokens
        total_output += synthesis_response.usage.output_tokens

        answer_text, thinking_text = self._extract_response(synthesis_response)

        all_steps.append(ReasoningStep(
            step_number=step_counter,
            description="ToT synthesis -- final answer",
            content=answer_text,
            confidence=best.evaluation_score,
            duration_ms=0,
        ))

        return ReasoningResult(
            strategy=ReasoningStrategy.TREE_OF_THOUGHT,
            query=query,
            answer=answer_text,
            confidence=best.evaluation_score,
            steps=all_steps,
            thinking_content=thinking_text,
            total_input_tokens=total_input,
            total_output_tokens=total_output,
            metadata={"best_branch_id": best.branch_id, "tree_depth": best.depth},
        )

    # ------------------------------------------------------------------
    # Self-reflection
    # ------------------------------------------------------------------

    async def _self_reflect(
        self,
        query: str,
        current_answer: str,
        steps: list[ReasoningStep],
    ) -> ReasoningResult:
        """Run one round of self-reflection to critique and optionally revise."""
        trace_text = "\n".join(
            f"Step {s.step_number}: {s.description}\n{s.content}" for s in steps
        )
        response = await self._call_model(
            system="You are a meticulous critical reviewer.",
            user=SELF_REFLECTION_PROMPT.format(
                question=query,
                answer=current_answer,
                reasoning_trace=trace_text,
            ),
            use_thinking=True,
        )
        text, thinking = self._extract_response(response)

        revised_answer = current_answer
        confidence = 0.6
        if "[Revised Answer]" in text:
            revised_section = text.split("[Revised Answer]")[1]
            if "[Confidence]" in revised_section:
                parts = revised_section.split("[Confidence]")
                revised_body = parts[0].strip()
                if revised_body.upper() != "NO REVISION NEEDED":
                    revised_answer = revised_body
                try:
                    confidence = float(parts[1].strip().split()[0])
                except (ValueError, IndexError):
                    confidence = 0.6
            else:
                revised_body = revised_section.strip()
                if revised_body.upper() != "NO REVISION NEEDED":
                    revised_answer = revised_body

        reflection_step = ReasoningStep(
            step_number=len(steps) + 1,
            description="Self-reflection and verification",
            content=text,
            confidence=confidence,
        )
        return ReasoningResult(
            strategy=ReasoningStrategy.CHAIN_OF_THOUGHT,
            query=query,
            answer=revised_answer,
            confidence=confidence,
            steps=[reflection_step],
            thinking_content=thinking,
        )

    # ------------------------------------------------------------------
    # LLM helpers
    # ------------------------------------------------------------------

    async def _call_model(
        self,
        *,
        system: str,
        user: str,
        use_thinking: bool = False,
        model: str | None = None,
    ) -> anthropic.types.Message:
        """Call the Anthropic API with optional extended thinking."""
        model = model or self.settings.reasoning_model
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": 16_384,
            "system": system,
            "messages": [{"role": "user", "content": user}],
        }
        if use_thinking:
            kwargs["temperature"] = 1  # required by extended-thinking API
            kwargs["thinking"] = {
                "type": "enabled",
                "budget_tokens": self.settings.thinking_budget_tokens,
            }

        response = await self._client.messages.create(**kwargs)
        return response

    @staticmethod
    def _extract_response(
        response: anthropic.types.Message,
    ) -> tuple[str, str]:
        """Return ``(visible_text, thinking_text)`` from a Message."""
        visible_parts: list[str] = []
        thinking_parts: list[str] = []
        for block in response.content:
            if block.type == "thinking":
                thinking_parts.append(block.thinking)
            elif block.type == "text":
                visible_parts.append(block.text)
        return "\n".join(visible_parts), "\n".join(thinking_parts)

    # ------------------------------------------------------------------
    # Parsing helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_cot_steps(text: str) -> list[ReasoningStep]:
        """Parse numbered ``[Step N]`` blocks from CoT model output."""
        steps: list[ReasoningStep] = []
        import re

        pattern = re.compile(
            r"\[Step\s+(\d+)\]\s*\n"
            r"Description:\s*(.+)\n"
            r"Reasoning:\s*([\s\S]*?)\n"
            r"Confidence:\s*([\d.]+)",
            re.IGNORECASE,
        )
        for m in pattern.finditer(text):
            steps.append(ReasoningStep(
                step_number=int(m.group(1)),
                description=m.group(2).strip(),
                content=m.group(3).strip(),
                confidence=float(m.group(4)),
            ))

        # Fallback: if the model didn't follow the format exactly, create one
        # step from the entire text.
        if not steps:
            steps.append(ReasoningStep(
                step_number=1,
                description="Reasoning (unstructured)",
                content=text.split("[Final Answer]")[0].strip() if "[Final Answer]" in text else text,
                confidence=0.5,
            ))
        return steps

    @staticmethod
    def _parse_final_answer(text: str) -> tuple[str, float]:
        """Extract the ``[Final Answer]`` section and overall confidence."""
        if "[Final Answer]" not in text:
            return text.strip(), 0.5

        answer_section = text.split("[Final Answer]")[1].strip()
        confidence = 0.5
        if "Overall Confidence:" in answer_section:
            parts = answer_section.split("Overall Confidence:")
            answer_body = parts[0].strip()
            try:
                confidence = float(parts[1].strip().split()[0])
            except (ValueError, IndexError):
                pass
        else:
            answer_body = answer_section

        return answer_body, confidence

    def _parse_tot_branches(
        self,
        text: str,
        parent: ThoughtBranch,
    ) -> list[ThoughtBranch]:
        """Parse ``[Approach N]`` blocks from ToT generation output."""
        import re

        branches: list[ThoughtBranch] = []
        pattern = re.compile(
            r"\[Approach\s+(\d+)\]\s*\n"
            r"Thought:\s*([\s\S]*?)\n"
            r"Rationale:\s*([\s\S]*?)(?=\[Approach|\Z)",
            re.IGNORECASE,
        )
        for m in pattern.finditer(text):
            branches.append(ThoughtBranch(
                branch_id=str(uuid.uuid4()),
                parent_id=parent.branch_id,
                depth=parent.depth + 1,
                thought=m.group(2).strip(),
            ))

        # Fallback: produce at least one branch
        if not branches:
            branches.append(ThoughtBranch(
                branch_id=str(uuid.uuid4()),
                parent_id=parent.branch_id,
                depth=parent.depth + 1,
                thought=text.strip(),
            ))
        return branches

    @staticmethod
    def _apply_scores(candidates: list[ThoughtBranch], eval_text: str) -> None:
        """Apply scores from the evaluation model to candidate branches."""
        import re

        for i, candidate in enumerate(candidates):
            pattern = rf"Approach\s+{i + 1}\s*:\s*([\d.]+)"
            m = re.search(pattern, eval_text, re.IGNORECASE)
            if m:
                try:
                    candidate.evaluation_score = float(m.group(1))
                except ValueError:
                    candidate.evaluation_score = 0.5
            else:
                candidate.evaluation_score = 0.5

    def _build_tot_context(self, node: ThoughtBranch) -> str:
        """Build a textual context by tracing from *node* up to the root."""
        parts: list[str] = []
        current: ThoughtBranch | None = node
        while current is not None:
            parts.append(f"Depth {current.depth}: {current.thought}")
            # We don't store back-pointers, so the context is just this node.
            current = None
        return "\n".join(reversed(parts))

    def _select_best_leaf(self, root: ThoughtBranch) -> ThoughtBranch:
        """DFS to find the highest-scored leaf in the tree."""
        best: ThoughtBranch = root
        stack = [root]
        while stack:
            node = stack.pop()
            unpruned = [c for c in node.children if not c.is_pruned]
            if not unpruned:
                # leaf
                if node.evaluation_score >= best.evaluation_score:
                    best = node
            else:
                stack.extend(unpruned)
        return best

    @staticmethod
    def _trace_path(leaf: ThoughtBranch) -> str:
        """Return a textual description of the path to *leaf*."""
        return f"[Depth {leaf.depth}] {leaf.thought} (score: {leaf.evaluation_score:.2f})"
