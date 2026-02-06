"""A2A (Agent-to-Agent) protocol implementation.

Follows the Google A2A specification for inter-agent communication:
- Task lifecycle: submitted -> working -> [input-required] -> completed | failed
- Message model with typed parts (text, file, data)
- Server that handles incoming task requests
- Client that submits tasks to remote agents
- Streaming support via SSE for real-time updates
"""

from __future__ import annotations

import asyncio
import time
import uuid
from enum import Enum
from typing import Any, AsyncIterator

import httpx
import structlog
from pydantic import BaseModel, Field

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Core protocol models
# ---------------------------------------------------------------------------


class TaskState(str, Enum):
    """Task lifecycle states per the A2A spec."""

    SUBMITTED = "submitted"
    WORKING = "working"
    INPUT_REQUIRED = "input-required"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class TextPart(BaseModel):
    """A text content part within a message."""

    type: str = "text"
    text: str


class FilePart(BaseModel):
    """A file content part within a message."""

    type: str = "file"
    file_name: str
    mime_type: str
    data: str  # base64-encoded content or URL


class DataPart(BaseModel):
    """A structured-data content part within a message."""

    type: str = "data"
    data: dict[str, Any]


# Union type for message parts
MessagePart = TextPart | FilePart | DataPart


class Message(BaseModel):
    """A single message exchanged between agents."""

    role: str  # "user" | "agent"
    parts: list[MessagePart]
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: float = Field(default_factory=time.time)


class TaskArtifact(BaseModel):
    """An output artifact produced by the agent for a task."""

    name: str
    description: str | None = None
    parts: list[MessagePart]
    index: int = 0


class Task(BaseModel):
    """A2A Task — the fundamental unit of work in the protocol."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str | None = None
    state: TaskState = TaskState.SUBMITTED
    messages: list[Message] = Field(default_factory=list)
    artifacts: list[TaskArtifact] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)


class TaskSendParams(BaseModel):
    """Parameters for submitting or updating a task."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str | None = None
    message: Message
    metadata: dict[str, Any] = Field(default_factory=dict)


class TaskStatusUpdate(BaseModel):
    """SSE event for task status changes."""

    task_id: str
    state: TaskState
    message: Message | None = None
    artifact: TaskArtifact | None = None
    final: bool = False
    timestamp: float = Field(default_factory=time.time)


class TaskQueryParams(BaseModel):
    """Parameters for querying task status."""

    task_id: str


# ---------------------------------------------------------------------------
# A2A Server — handles incoming task requests
# ---------------------------------------------------------------------------


class A2AServer:
    """Server-side A2A protocol handler.

    Receives tasks, manages their lifecycle, invokes a processing callback,
    and supports SSE streaming of status updates.
    """

    def __init__(self) -> None:
        self._tasks: dict[str, Task] = {}
        self._subscribers: dict[str, list[asyncio.Queue[TaskStatusUpdate]]] = {}

    # -- Task management -----------------------------------------------------

    async def handle_task_send(self, params: TaskSendParams) -> Task:
        """Handle a ``tasks/send`` request — create or update a task."""
        task = self._tasks.get(params.id)
        if task is None:
            task = Task(
                id=params.id,
                session_id=params.session_id,
                messages=[params.message],
                metadata=params.metadata,
            )
            self._tasks[task.id] = task
            logger.info("a2a.task.created", task_id=task.id)
        else:
            task.messages.append(params.message)
            task.updated_at = time.time()
            logger.info("a2a.task.updated", task_id=task.id)

        # Transition to working
        await self._update_state(task, TaskState.WORKING)

        # Process asynchronously
        asyncio.create_task(self._process_task(task))

        return task

    async def handle_task_get(self, task_id: str) -> Task | None:
        """Handle a ``tasks/get`` request — return current task state."""
        return self._tasks.get(task_id)

    async def handle_task_cancel(self, task_id: str) -> Task | None:
        """Handle a ``tasks/cancel`` request."""
        task = self._tasks.get(task_id)
        if task and task.state not in (TaskState.COMPLETED, TaskState.FAILED):
            await self._update_state(task, TaskState.CANCELED)
        return task

    # -- SSE streaming -------------------------------------------------------

    async def subscribe(self, task_id: str) -> AsyncIterator[TaskStatusUpdate]:
        """Subscribe to SSE updates for a task."""
        queue: asyncio.Queue[TaskStatusUpdate] = asyncio.Queue()
        if task_id not in self._subscribers:
            self._subscribers[task_id] = []
        self._subscribers[task_id].append(queue)

        try:
            while True:
                update = await queue.get()
                yield update
                if update.final:
                    break
        finally:
            self._subscribers[task_id].remove(queue)
            if not self._subscribers[task_id]:
                del self._subscribers[task_id]

    # -- Internal processing -------------------------------------------------

    async def _process_task(self, task: Task) -> None:
        """Process a task (demo: echoes input with analysis)."""
        try:
            # Extract user message text
            user_text = ""
            for msg in task.messages:
                if msg.role == "user":
                    for part in msg.parts:
                        if isinstance(part, TextPart):
                            user_text += part.text + " "

            user_text = user_text.strip()

            # Simulate processing delay
            await asyncio.sleep(0.1)

            # Create response
            response_message = Message(
                role="agent",
                parts=[
                    TextPart(
                        text=f"Processed your request: '{user_text}'. "
                        f"Analysis complete with {len(user_text.split())} words analyzed."
                    )
                ],
            )
            task.messages.append(response_message)

            # Create artifact
            artifact = TaskArtifact(
                name="analysis_result",
                description="Result of processing the submitted task.",
                parts=[
                    DataPart(
                        data={
                            "input_length": len(user_text),
                            "word_count": len(user_text.split()),
                            "processed_at": time.time(),
                            "status": "success",
                        }
                    )
                ],
            )
            task.artifacts.append(artifact)

            await self._update_state(
                task,
                TaskState.COMPLETED,
                message=response_message,
                artifact=artifact,
            )

        except Exception as exc:
            logger.error("a2a.task.processing_error", task_id=task.id, error=str(exc))
            error_message = Message(
                role="agent",
                parts=[TextPart(text=f"Task processing failed: {exc}")],
            )
            task.messages.append(error_message)
            await self._update_state(task, TaskState.FAILED, message=error_message)

    async def _update_state(
        self,
        task: Task,
        new_state: TaskState,
        message: Message | None = None,
        artifact: TaskArtifact | None = None,
    ) -> None:
        """Transition a task to a new state and notify subscribers."""
        task.state = new_state
        task.updated_at = time.time()
        logger.info("a2a.task.state_changed", task_id=task.id, state=new_state.value)

        update = TaskStatusUpdate(
            task_id=task.id,
            state=new_state,
            message=message,
            artifact=artifact,
            final=new_state in (TaskState.COMPLETED, TaskState.FAILED, TaskState.CANCELED),
        )

        # Notify SSE subscribers
        for queue in self._subscribers.get(task.id, []):
            await queue.put(update)

    # -- Introspection -------------------------------------------------------

    def list_tasks(
        self,
        state: TaskState | None = None,
        limit: int = 50,
    ) -> list[Task]:
        """List tasks, optionally filtered by state."""
        tasks = list(self._tasks.values())
        if state is not None:
            tasks = [t for t in tasks if t.state == state]
        return sorted(tasks, key=lambda t: t.updated_at, reverse=True)[:limit]


# ---------------------------------------------------------------------------
# A2A Client — sends tasks to remote agents
# ---------------------------------------------------------------------------


class A2AClient:
    """Client for submitting tasks to remote A2A-compliant agents."""

    def __init__(self, timeout: float = 30.0) -> None:
        self._timeout = timeout

    async def send_task(
        self,
        agent_url: str,
        message: str,
        task_id: str | None = None,
        session_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Submit a task to a remote agent via ``POST /api/v1/a2a/tasks``.

        Parameters
        ----------
        agent_url:
            Base URL of the target agent (e.g. ``http://agent:8008``).
        message:
            The user message text to send.
        task_id:
            Optional task ID (auto-generated if omitted).
        session_id:
            Optional session ID for multi-turn conversations.
        metadata:
            Arbitrary metadata to include with the task.
        """
        url = f"{agent_url.rstrip('/')}/api/v1/a2a/tasks"
        payload = TaskSendParams(
            id=task_id or str(uuid.uuid4()),
            session_id=session_id,
            message=Message(
                role="user",
                parts=[TextPart(text=message)],
            ),
            metadata=metadata or {},
        )

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url, json=payload.model_dump())
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            logger.error("a2a.client.send_failed", url=url, error=str(exc))
            return {"error": str(exc), "task_id": payload.id}

    async def get_task(self, agent_url: str, task_id: str) -> dict[str, Any]:
        """Poll task status via ``GET /api/v1/a2a/tasks/{task_id}``."""
        url = f"{agent_url.rstrip('/')}/api/v1/a2a/tasks/{task_id}"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.get(url)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            logger.error("a2a.client.get_failed", url=url, error=str(exc))
            return {"error": str(exc), "task_id": task_id}

    async def cancel_task(self, agent_url: str, task_id: str) -> dict[str, Any]:
        """Cancel a task via ``POST /api/v1/a2a/tasks/{task_id}/cancel``."""
        url = f"{agent_url.rstrip('/')}/api/v1/a2a/tasks/{task_id}/cancel"
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(url)
                resp.raise_for_status()
                return resp.json()
        except Exception as exc:
            logger.error("a2a.client.cancel_failed", url=url, error=str(exc))
            return {"error": str(exc), "task_id": task_id}

    async def send_and_wait(
        self,
        agent_url: str,
        message: str,
        poll_interval: float = 0.5,
        max_wait: float = 30.0,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Submit a task and poll until it reaches a terminal state.

        Convenience wrapper around :meth:`send_task` + :meth:`get_task`.
        """
        result = await self.send_task(agent_url, message, **kwargs)
        if "error" in result:
            return result

        task_id = result.get("id", result.get("task_id"))
        if not task_id:
            return result

        start = time.time()
        while time.time() - start < max_wait:
            status = await self.get_task(agent_url, task_id)
            state = status.get("state", "")
            if state in ("completed", "failed", "canceled"):
                return status
            await asyncio.sleep(poll_interval)

        return {"error": "Timed out waiting for task completion", "task_id": task_id}
