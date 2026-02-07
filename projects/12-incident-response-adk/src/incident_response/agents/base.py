"""Abstract agent base classes mirroring Google ADK's agent API.

These classes provide the same interface as ``google.adk.agents`` but work
without the actual ``google-adk`` package installed. Each class demonstrates
a different ADK workflow pattern:

- :class:`BaseAgent` -- abstract root of the agent hierarchy
- :class:`LlmAgent` -- agent that uses an LLM for reasoning (with heuristic fallback)
- :class:`SequentialAgent` -- runs sub-agents in sequence, passing context through
- :class:`ParallelAgent` -- runs sub-agents concurrently, merging results
- :class:`LoopAgent` -- runs sub-agents in a loop until a condition is met
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from typing import Any, Callable

import structlog

logger = structlog.get_logger(__name__)


class BaseAgent(ABC):
    """Abstract base class for all ADK-style agents.

    Every agent receives a shared ``context`` dict, performs work, and
    returns an updated copy of that context.
    """

    def __init__(self, name: str, description: str = "") -> None:
        self.name = name
        self.description = description

    @abstractmethod
    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute the agent's logic and return updated context.

        Args:
            context: Shared state dictionary passed between agents.

        Returns:
            Updated context dictionary.
        """
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"


class LlmAgent(BaseAgent):
    """Agent that uses an LLM for reasoning, with heuristic fallback.

    In production with ``google-adk``, this would delegate to Gemini.
    Here we implement heuristic logic so the agent works without API keys.
    Subclasses override :meth:`_heuristic_run` to provide domain logic.
    """

    def __init__(
        self,
        name: str,
        instruction: str = "",
        description: str = "",
        model: str = "gemini-2.0-flash",
        tools: list[Callable[..., Any]] | None = None,
    ) -> None:
        super().__init__(name=name, description=description)
        self.instruction = instruction
        self.model = model
        self.tools = tools or []

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute via heuristic fallback (no LLM API required).

        In a real ADK deployment the ``instruction`` would be sent to the
        configured model. Here we call :meth:`_heuristic_run` instead.
        """
        logger.info(
            "agent_running",
            agent=self.name,
            model=self.model,
            mode="heuristic_fallback",
        )
        return await self._heuristic_run(context)

    async def _heuristic_run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Override in subclasses to provide domain-specific heuristic logic."""
        return context


class SequentialAgent(BaseAgent):
    """Runs sub-agents in sequence, passing context through each one.

    Mirrors ``google.adk.agents.SequentialAgent``. The output context of
    agent *N* becomes the input context of agent *N+1*.
    """

    def __init__(
        self,
        name: str,
        sub_agents: list[BaseAgent],
        description: str = "",
    ) -> None:
        super().__init__(name=name, description=description)
        self.sub_agents = sub_agents

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute sub-agents sequentially, threading context through."""
        logger.info(
            "sequential_agent_start",
            agent=self.name,
            sub_agents=[a.name for a in self.sub_agents],
        )
        current_context = context.copy()
        for agent in self.sub_agents:
            logger.debug("sequential_step", parent=self.name, child=agent.name)
            current_context = await agent.run(current_context)
        logger.info("sequential_agent_complete", agent=self.name)
        return current_context


class ParallelAgent(BaseAgent):
    """Runs sub-agents concurrently, merging results into context.

    Mirrors ``google.adk.agents.ParallelAgent``. All sub-agents receive
    the same input context snapshot. Results are merged via dict update
    in declaration order (later agents' keys win on conflict).
    """

    def __init__(
        self,
        name: str,
        sub_agents: list[BaseAgent],
        description: str = "",
    ) -> None:
        super().__init__(name=name, description=description)
        self.sub_agents = sub_agents

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute sub-agents concurrently with asyncio.gather."""
        logger.info(
            "parallel_agent_start",
            agent=self.name,
            sub_agents=[a.name for a in self.sub_agents],
        )
        # Each sub-agent gets the same snapshot
        tasks = [agent.run(context.copy()) for agent in self.sub_agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results into a combined context
        merged = context.copy()
        for agent, result in zip(self.sub_agents, results):
            if isinstance(result, Exception):
                logger.error(
                    "parallel_agent_error",
                    agent=agent.name,
                    error=str(result),
                )
                merged.setdefault("errors", {})[agent.name] = str(result)
            else:
                merged.update(result)

        logger.info("parallel_agent_complete", agent=self.name)
        return merged


class LoopAgent(BaseAgent):
    """Runs sub-agents in a loop until a condition is met or max iterations.

    Mirrors ``google.adk.agents.LoopAgent``. On each iteration the full
    sub-agent sequence runs. The loop exits when:
    - ``context["loop_complete"]`` is set to ``True``
    - ``max_iterations`` is reached
    """

    def __init__(
        self,
        name: str,
        sub_agents: list[BaseAgent],
        max_iterations: int = 3,
        description: str = "",
    ) -> None:
        super().__init__(name=name, description=description)
        self.sub_agents = sub_agents
        self.max_iterations = max_iterations

    async def run(self, context: dict[str, Any]) -> dict[str, Any]:
        """Execute sub-agents in a loop with escalation on each pass."""
        logger.info(
            "loop_agent_start",
            agent=self.name,
            max_iterations=self.max_iterations,
            sub_agents=[a.name for a in self.sub_agents],
        )

        current_context = context.copy()
        current_context.setdefault("loop_iteration", 0)
        current_context.setdefault("loop_complete", False)

        for iteration in range(self.max_iterations):
            current_context["loop_iteration"] = iteration + 1
            logger.info(
                "loop_iteration",
                agent=self.name,
                iteration=iteration + 1,
                max_iterations=self.max_iterations,
            )

            for agent in self.sub_agents:
                current_context = await agent.run(current_context)

            if current_context.get("loop_complete", False):
                logger.info(
                    "loop_condition_met",
                    agent=self.name,
                    iteration=iteration + 1,
                )
                break
        else:
            logger.warning(
                "loop_max_iterations_reached",
                agent=self.name,
                max_iterations=self.max_iterations,
            )
            current_context["loop_exhausted"] = True

        logger.info(
            "loop_agent_complete",
            agent=self.name,
            iterations=current_context["loop_iteration"],
        )
        return current_context
