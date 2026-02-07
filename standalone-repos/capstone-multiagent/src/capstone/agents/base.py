"""Base agent abstraction for the multi-agent platform.

Every specialized agent inherits from :class:`BaseAgent` and implements two
core methods:

* :meth:`execute` -- carry out a task and return a structured result.
* :meth:`can_handle` -- return a confidence score (0-1) indicating how well
  the agent can handle a given task, enabling the orchestrator to route work
  to the most capable specialist.
"""

from __future__ import annotations

import time
import uuid
from abc import ABC, abstractmethod
from enum import StrEnum
from typing import Any

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from pydantic import BaseModel, Field

from capstone.config import Settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class TaskStatus(StrEnum):
    """Lifecycle states for an agent task."""

    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"


class AgentTask(BaseModel):
    """A unit of work dispatched to an agent by the orchestrator."""

    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    parent_task_id: str | None = Field(
        default=None,
        description="ID of the top-level task this subtask belongs to.",
    )
    description: str = Field(description="Natural-language description of what to do.")
    context: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional context from previous agents or the user.",
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="Constraints or requirements the agent must respect.",
    )
    preferred_agent: str | None = Field(
        default=None,
        description="Hint from the supervisor about which agent should handle this.",
    )
    max_retries: int = Field(default=2, ge=0, le=5)
    timeout_seconds: int = Field(default=120, ge=10, le=600)

    # Mutable state
    status: TaskStatus = TaskStatus.PENDING
    attempt: int = 0


class AgentResult(BaseModel):
    """Structured output returned by an agent after executing a task."""

    task_id: str
    agent_name: str
    status: TaskStatus
    output: str = Field(description="Primary textual output from the agent.")
    structured_data: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional structured data (tables, metrics, code blocks, etc.).",
    )
    confidence: float = Field(
        default=1.0,
        ge=0.0,
        le=1.0,
        description="Agent's self-assessed confidence in its output.",
    )
    duration_seconds: float = 0.0
    error: str | None = None
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Tokens used, model version, and other telemetry.",
    )


# ---------------------------------------------------------------------------
# Base class
# ---------------------------------------------------------------------------


class BaseAgent(ABC):
    """Abstract base class for all specialized agents.

    Subclasses must implement :meth:`execute` and :meth:`can_handle`.
    The base class provides common LLM interaction helpers, error handling,
    and structured logging.
    """

    name: str = "base"
    description: str = "Abstract base agent."
    capabilities: list[str] = []

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._llm: ChatAnthropic | None = None

    # -- Lazy LLM initialization -----------------------------------------------

    def _get_llm(self, model: str | None = None, temperature: float | None = None) -> ChatAnthropic:
        """Return a :class:`ChatAnthropic` instance, creating one lazily."""
        model = model or self._settings.supervisor_model
        temperature = temperature if temperature is not None else self._settings.default_temperature
        if self._llm is None or self._llm.model != model:
            self._llm = ChatAnthropic(
                model=model,
                anthropic_api_key=self._settings.anthropic_api_key.get_secret_value(),
                temperature=temperature,
                max_tokens=self._settings.default_max_tokens,
            )
        return self._llm

    # -- Core interface --------------------------------------------------------

    @abstractmethod
    async def execute(self, task: AgentTask) -> AgentResult:
        """Execute *task* and return a structured result.

        Implementations should:
        1. Build an appropriate prompt from the task description and context.
        2. Call the LLM (possibly multiple rounds).
        3. Parse the response into an :class:`AgentResult`.
        """

    @abstractmethod
    async def can_handle(self, task: AgentTask) -> float:
        """Return a confidence score (0.0 - 1.0) for handling *task*.

        The orchestrator calls this on every registered agent and routes the
        task to the agent with the highest confidence.  Agents should inspect
        ``task.description`` and ``task.context`` to determine relevance.
        """

    # -- Helpers ---------------------------------------------------------------

    async def _invoke_llm(
        self,
        system_prompt: str,
        user_message: str,
        model: str | None = None,
        temperature: float | None = None,
    ) -> str:
        """Send a single-turn message to the LLM and return the text response."""
        llm = self._get_llm(model=model, temperature=temperature)
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_message),
        ]
        response: AIMessage = await llm.ainvoke(messages)
        return str(response.content)

    async def _safe_execute(self, task: AgentTask) -> AgentResult:
        """Wrap :meth:`execute` with timing, logging, and error handling."""
        task.status = TaskStatus.RUNNING
        task.attempt += 1
        start = time.monotonic()

        log = logger.bind(
            agent=self.name,
            task_id=task.task_id,
            attempt=task.attempt,
        )
        log.info("agent_task_started", description=task.description[:120])

        try:
            result = await self.execute(task)
            result.duration_seconds = round(time.monotonic() - start, 3)
            log.info(
                "agent_task_completed",
                status=result.status,
                confidence=result.confidence,
                duration=result.duration_seconds,
            )
            return result

        except Exception as exc:
            duration = round(time.monotonic() - start, 3)
            log.exception("agent_task_failed", error=str(exc), duration=duration)
            return AgentResult(
                task_id=task.task_id,
                agent_name=self.name,
                status=TaskStatus.FAILED,
                output="",
                error=str(exc),
                duration_seconds=duration,
            )

    def info(self) -> dict[str, Any]:
        """Return a JSON-serializable summary of this agent."""
        return {
            "name": self.name,
            "description": self.description,
            "capabilities": self.capabilities,
        }
