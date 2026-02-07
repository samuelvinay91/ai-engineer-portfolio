"""Multi-Agent Orchestrator -- LangGraph-based supervisor pattern.

This module implements the core orchestration logic using a LangGraph
:class:`StateGraph`.  The graph follows a **supervisor pattern** where a
meta-agent (the supervisor) receives complex tasks, decomposes them into
subtasks, routes each subtask to the most capable specialist agent,
coordinates parallel execution where safe, and aggregates results into a
coherent final output.

Graph topology
--------------

.. code-block:: text

    START
      |
      v
    [supervisor]  <---+
      |               |
      +--> [researcher]  --+
      +--> [coder]        -+
      +--> [analyst]      -+-> [aggregator] --> END
      +--> [writer]       -+
      |                    |
      +-----> (loop) ------+

* **supervisor** -- Decomposes the user task, creates subtasks, and decides
  which agents to invoke next (or whether to loop for refinement).
* **agent nodes** -- Each specialist processes its assigned subtask(s).
* **aggregator** -- Merges all agent results into a unified response.

The supervisor can route to multiple agents in a single step (parallel fan-out)
and can loop back for iterative refinement up to a configurable depth.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from typing import Annotated, Any, Literal

import structlog
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from capstone.agents.base import AgentResult, AgentTask, BaseAgent, TaskStatus
from capstone.config import Settings

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# State schema
# ---------------------------------------------------------------------------


def _merge_results(
    existing: list[AgentResult], new: list[AgentResult]
) -> list[AgentResult]:
    """Merge strategy for agent results -- append new, deduplicate by task_id."""
    seen = {r.task_id for r in existing}
    merged = list(existing)
    for r in new:
        if r.task_id not in seen:
            merged.append(r)
            seen.add(r.task_id)
    return merged


def _merge_subtasks(
    existing: list[AgentTask], new: list[AgentTask]
) -> list[AgentTask]:
    """Merge strategy for subtasks -- append new, deduplicate by task_id."""
    seen = {t.task_id for t in existing}
    merged = list(existing)
    for t in new:
        if t.task_id not in seen:
            merged.append(t)
            seen.add(t.task_id)
    return merged


class OrchestratorState(BaseModel):
    """Typed state flowing through the LangGraph orchestration graph.

    LangGraph uses this as the shared state object.  Each node reads from
    and writes to specific fields, enabling deterministic state transitions.
    """

    # -- Input -----------------------------------------------------------------
    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    user_request: str = ""
    session_id: str = ""

    # -- Planning --------------------------------------------------------------
    plan: str = ""
    subtasks: Annotated[list[AgentTask], _merge_subtasks] = Field(default_factory=list)
    pending_agents: list[str] = Field(default_factory=list)

    # -- Execution -------------------------------------------------------------
    agent_results: Annotated[list[AgentResult], _merge_results] = Field(default_factory=list)
    current_iteration: int = 0
    max_iterations: int = 3

    # -- Output ----------------------------------------------------------------
    final_output: str = ""
    status: str = "pending"
    error: str | None = None

    class Config:
        arbitrary_types_allowed = True


# ---------------------------------------------------------------------------
# Supervisor prompts
# ---------------------------------------------------------------------------

DECOMPOSITION_PROMPT = """\
You are the Supervisor in a multi-agent AI platform. You receive complex user
requests and decompose them into subtasks for specialist agents.

## Available Agents

| Agent       | Capabilities |
|-------------|-------------|
| researcher  | Deep research, fact-checking, literature review, information synthesis |
| coder       | Code generation, code review, bug fixing, code explanation, test writing |
| analyst     | Statistical analysis, trend identification, anomaly detection, data insights |
| writer      | Blog posts, documentation, reports, creative writing, SEO, editing |

## Instructions

1. Analyse the user's request.
2. Break it into 1-{max_subtasks} focused subtasks, each assignable to ONE agent.
3. Identify dependencies between subtasks (which must finish before others start).
4. If the task is simple enough for a single agent, create just one subtask.

Respond with JSON:
{{
  "plan": "<high-level execution plan in 1-3 sentences>",
  "subtasks": [
    {{
      "description": "<what the agent should do>",
      "preferred_agent": "<researcher|coder|analyst|writer>",
      "depends_on": [],
      "constraints": ["<constraint1>", ...],
      "priority": <1=highest>
    }}
  ]
}}
"""

AGGREGATION_PROMPT = """\
You are the Supervisor aggregating results from specialist agents into a
cohesive final response for the user.

## Original Request
{user_request}

## Plan
{plan}

## Agent Results
{agent_results}

## Instructions

Synthesise all agent outputs into a single, well-structured response that
directly addresses the user's original request.  Integrate insights
seamlessly -- do not simply concatenate outputs.  If any agent failed,
acknowledge the gap and work with what succeeded.

Be thorough but concise.  Use markdown formatting for readability.
"""

REFINEMENT_PROMPT = """\
You are the Supervisor reviewing intermediate results to decide next steps.

## Original Request
{user_request}

## Plan
{plan}

## Results So Far
{agent_results}

## Instructions

Evaluate whether the results adequately address the user's request.

Respond with JSON:
{{
  "is_complete": <true|false>,
  "reasoning": "<why complete or incomplete>",
  "additional_subtasks": [
    {{
      "description": "<what else is needed>",
      "preferred_agent": "<agent name>",
      "constraints": []
    }}
  ]
}}
"""


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class MultiAgentOrchestrator:
    """LangGraph-based orchestrator implementing the supervisor pattern.

    The orchestrator manages the full lifecycle of a complex task:

    1. **Decomposition** -- The supervisor analyses the request and creates
       a plan with subtasks assigned to specialist agents.
    2. **Execution** -- Subtasks are dispatched to agents, running in parallel
       when there are no dependencies between them.
    3. **Refinement** -- The supervisor reviews intermediate results and may
       create additional subtasks for another iteration.
    4. **Aggregation** -- All results are merged into a coherent final output.
    """

    def __init__(self, settings: Settings, agents: dict[str, BaseAgent] | None = None) -> None:
        self._settings = settings
        self._agents: dict[str, BaseAgent] = agents or {}
        self._supervisor_llm = ChatAnthropic(
            model=settings.supervisor_model,
            anthropic_api_key=settings.anthropic_api_key.get_secret_value(),
            temperature=0.2,
            max_tokens=settings.default_max_tokens,
        )
        self._graph = self._build_graph()

    def register_agent(self, agent: BaseAgent) -> None:
        """Register a specialist agent with the orchestrator."""
        self._agents[agent.name] = agent
        logger.info("agent_registered", agent=agent.name, capabilities=agent.capabilities)

    def list_agents(self) -> list[dict[str, Any]]:
        """Return metadata for all registered agents."""
        return [agent.info() for agent in self._agents.values()]

    # -- Graph construction ----------------------------------------------------

    def _build_graph(self) -> StateGraph:
        """Construct the LangGraph StateGraph with supervisor routing."""
        graph = StateGraph(OrchestratorState)

        # Add nodes
        graph.add_node("supervisor", self._supervisor_node)
        graph.add_node("execute_agents", self._execute_agents_node)
        graph.add_node("refine", self._refine_node)
        graph.add_node("aggregate", self._aggregate_node)

        # Entry point
        graph.set_entry_point("supervisor")

        # Edges from supervisor -> execute_agents
        graph.add_edge("supervisor", "execute_agents")

        # After execution, decide whether to refine or aggregate
        graph.add_conditional_edges(
            "execute_agents",
            self._should_refine_or_aggregate,
            {
                "refine": "refine",
                "aggregate": "aggregate",
            },
        )

        # After refinement, either go back to execute or aggregate
        graph.add_conditional_edges(
            "refine",
            self._should_continue_or_aggregate,
            {
                "execute_agents": "execute_agents",
                "aggregate": "aggregate",
            },
        )

        # Aggregation is terminal
        graph.add_edge("aggregate", END)

        return graph.compile()

    # -- Graph nodes -----------------------------------------------------------

    async def _supervisor_node(self, state: OrchestratorState) -> dict[str, Any]:
        """Decompose the user request into subtasks with agent assignments."""
        logger.info(
            "supervisor_decomposing",
            task_id=state.task_id,
            request_length=len(state.user_request),
        )

        prompt = DECOMPOSITION_PROMPT.format(max_subtasks=self._settings.max_subtasks)
        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=state.user_request),
        ]

        response = await self._supervisor_llm.ainvoke(messages)
        raw = str(response.content)

        parsed = self._parse_json_response(raw)
        plan = parsed.get("plan", "Execute the request directly.")
        raw_subtasks = parsed.get("subtasks", [])

        subtasks: list[AgentTask] = []
        pending_agents: list[str] = []

        for st in raw_subtasks:
            agent_name = st.get("preferred_agent", "researcher")
            if agent_name not in self._agents:
                # Fall back to best-matching agent
                agent_name = await self._find_best_agent(st.get("description", ""))

            task = AgentTask(
                parent_task_id=state.task_id,
                description=st.get("description", ""),
                constraints=st.get("constraints", []),
                preferred_agent=agent_name,
                max_retries=self._settings.max_agent_retries,
                timeout_seconds=self._settings.agent_timeout_seconds,
            )
            subtasks.append(task)
            if agent_name not in pending_agents:
                pending_agents.append(agent_name)

        logger.info(
            "supervisor_plan_created",
            task_id=state.task_id,
            num_subtasks=len(subtasks),
            agents=pending_agents,
        )

        return {
            "plan": plan,
            "subtasks": subtasks,
            "pending_agents": pending_agents,
            "status": "running",
        }

    async def _execute_agents_node(self, state: OrchestratorState) -> dict[str, Any]:
        """Dispatch pending subtasks to their assigned agents, running in parallel."""
        logger.info(
            "executing_agents",
            task_id=state.task_id,
            num_subtasks=len(state.subtasks),
            iteration=state.current_iteration,
        )

        # Find subtasks that haven't been executed yet
        completed_ids = {r.task_id for r in state.agent_results}
        pending = [t for t in state.subtasks if t.task_id not in completed_ids]

        if not pending:
            return {"agent_results": [], "current_iteration": state.current_iteration + 1}

        # Group by agent for parallel dispatch
        agent_tasks: dict[str, list[AgentTask]] = {}
        for task in pending:
            agent_name = task.preferred_agent or "researcher"
            agent_tasks.setdefault(agent_name, []).append(task)

        # Execute all agents concurrently (respecting max parallelism)
        semaphore = asyncio.Semaphore(self._settings.max_parallel_agents)

        async def _run_agent_task(agent_name: str, task: AgentTask) -> AgentResult:
            async with semaphore:
                agent = self._agents.get(agent_name)
                if agent is None:
                    return AgentResult(
                        task_id=task.task_id,
                        agent_name=agent_name,
                        status=TaskStatus.FAILED,
                        output="",
                        error=f"Agent '{agent_name}' not registered.",
                    )

                # Inject context from prior results into the task
                prior_outputs = {
                    r.agent_name: r.output
                    for r in state.agent_results
                    if r.status == TaskStatus.COMPLETED
                }
                if prior_outputs:
                    task.context["prior_agent_outputs"] = prior_outputs

                return await agent._safe_execute(task)

        coroutines = [
            _run_agent_task(agent_name, task)
            for agent_name, tasks in agent_tasks.items()
            for task in tasks
        ]
        results = await asyncio.gather(*coroutines, return_exceptions=True)

        # Convert exceptions to failed results
        agent_results: list[AgentResult] = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                agent_results.append(
                    AgentResult(
                        task_id=pending[i].task_id if i < len(pending) else "unknown",
                        agent_name="unknown",
                        status=TaskStatus.FAILED,
                        output="",
                        error=str(result),
                    )
                )
            else:
                agent_results.append(result)

        return {
            "agent_results": agent_results,
            "current_iteration": state.current_iteration + 1,
        }

    async def _refine_node(self, state: OrchestratorState) -> dict[str, Any]:
        """Review results and optionally create additional subtasks."""
        logger.info(
            "supervisor_refining",
            task_id=state.task_id,
            iteration=state.current_iteration,
            num_results=len(state.agent_results),
        )

        results_summary = self._format_results_for_prompt(state.agent_results)
        prompt = REFINEMENT_PROMPT.format(
            user_request=state.user_request,
            plan=state.plan,
            agent_results=results_summary,
        )

        messages = [
            SystemMessage(content="You are the Supervisor reviewing results."),
            HumanMessage(content=prompt),
        ]

        response = await self._supervisor_llm.ainvoke(messages)
        parsed = self._parse_json_response(str(response.content))

        additional_subtasks: list[AgentTask] = []
        for st in parsed.get("additional_subtasks", []):
            agent_name = st.get("preferred_agent", "researcher")
            if agent_name not in self._agents:
                agent_name = await self._find_best_agent(st.get("description", ""))

            additional_subtasks.append(
                AgentTask(
                    parent_task_id=state.task_id,
                    description=st.get("description", ""),
                    constraints=st.get("constraints", []),
                    preferred_agent=agent_name,
                    max_retries=self._settings.max_agent_retries,
                    timeout_seconds=self._settings.agent_timeout_seconds,
                )
            )

        return {"subtasks": additional_subtasks}

    async def _aggregate_node(self, state: OrchestratorState) -> dict[str, Any]:
        """Synthesise all agent results into a unified final output."""
        logger.info(
            "aggregating_results",
            task_id=state.task_id,
            num_results=len(state.agent_results),
        )

        # If there is only one successful result, use it directly
        successful = [r for r in state.agent_results if r.status == TaskStatus.COMPLETED]
        if len(successful) == 1:
            return {
                "final_output": successful[0].output,
                "status": "completed",
            }

        # Multiple results: use LLM to synthesise
        results_summary = self._format_results_for_prompt(state.agent_results)
        prompt = AGGREGATION_PROMPT.format(
            user_request=state.user_request,
            plan=state.plan,
            agent_results=results_summary,
        )

        messages = [
            SystemMessage(content="You are the Supervisor producing the final response."),
            HumanMessage(content=prompt),
        ]

        response = await self._supervisor_llm.ainvoke(messages)
        final_output = str(response.content)

        # Determine overall status
        failed = [r for r in state.agent_results if r.status == TaskStatus.FAILED]
        status = "completed" if not failed else "partial"

        return {
            "final_output": final_output,
            "status": status,
        }

    # -- Routing functions -----------------------------------------------------

    def _should_refine_or_aggregate(self, state: OrchestratorState) -> str:
        """Decide whether to refine (loop) or aggregate (finish)."""
        if state.current_iteration >= state.max_iterations:
            return "aggregate"

        # If any results failed and we have retries left, try refining
        failed = [r for r in state.agent_results if r.status == TaskStatus.FAILED]
        if failed and state.current_iteration < state.max_iterations:
            return "refine"

        # If all subtasks completed on first pass, aggregate directly
        completed_ids = {r.task_id for r in state.agent_results}
        all_done = all(t.task_id in completed_ids for t in state.subtasks)
        if all_done:
            return "aggregate"

        return "refine"

    def _should_continue_or_aggregate(self, state: OrchestratorState) -> str:
        """After refinement, decide whether to execute more or finalize."""
        completed_ids = {r.task_id for r in state.agent_results}
        pending = [t for t in state.subtasks if t.task_id not in completed_ids]

        if pending and state.current_iteration < state.max_iterations:
            return "execute_agents"
        return "aggregate"

    # -- Public execution API --------------------------------------------------

    async def run(self, request: str, session_id: str = "") -> OrchestratorState:
        """Execute the full orchestration pipeline for a user request.

        Parameters
        ----------
        request:
            The user's natural-language request.
        session_id:
            Optional session identifier for memory continuity.

        Returns
        -------
        OrchestratorState
            The final state containing the aggregated output and metadata.
        """
        initial_state = OrchestratorState(
            user_request=request,
            session_id=session_id,
            max_iterations=min(3, self._settings.max_agent_retries + 1),
        )

        logger.info(
            "orchestration_started",
            task_id=initial_state.task_id,
            request_length=len(request),
        )

        start = time.monotonic()
        final_state = await self._graph.ainvoke(initial_state)
        duration = round(time.monotonic() - start, 3)

        # Handle both dict and OrchestratorState returns from langgraph
        if isinstance(final_state, dict):
            result_state = OrchestratorState(**final_state)
        else:
            result_state = final_state

        logger.info(
            "orchestration_completed",
            task_id=result_state.task_id,
            status=result_state.status,
            num_results=len(result_state.agent_results),
            duration=duration,
        )

        return result_state

    # -- Helpers ---------------------------------------------------------------

    async def _find_best_agent(self, description: str) -> str:
        """Find the agent with the highest confidence for a task description."""
        if not self._agents:
            return "researcher"

        task = AgentTask(description=description)
        scores: list[tuple[str, float]] = []
        for name, agent in self._agents.items():
            try:
                score = await agent.can_handle(task)
                scores.append((name, score))
            except Exception:
                logger.warning("agent_scoring_failed", agent=name)
                scores.append((name, 0.0))

        if not scores:
            return "researcher"

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[0][0]

    @staticmethod
    def _format_results_for_prompt(results: list[AgentResult]) -> str:
        """Format agent results into a readable string for LLM prompts."""
        parts: list[str] = []
        for r in results:
            status_icon = "OK" if r.status == TaskStatus.COMPLETED else "FAILED"
            parts.append(
                f"### Agent: {r.agent_name} [{status_icon}] "
                f"(confidence: {r.confidence})\n\n{r.output}"
            )
            if r.error:
                parts.append(f"\n**Error:** {r.error}")
            parts.append("")
        return "\n".join(parts)

    @staticmethod
    def _parse_json_response(raw: str) -> dict[str, Any]:
        """Extract JSON from an LLM response that may include markdown fences."""
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = cleaned.split("\n", 1)[1] if "\n" in cleaned else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[: cleaned.rfind("```")]
        cleaned = cleaned.strip()

        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("supervisor_json_parse_failed", raw_length=len(raw))
            return {"plan": raw, "subtasks": [], "is_complete": True}


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_orchestrator(settings: Settings) -> MultiAgentOrchestrator:
    """Create a fully-configured orchestrator with all specialist agents.

    This is the main entry point for constructing the orchestration pipeline.
    It registers all available agents and returns a ready-to-use orchestrator.
    """
    from capstone.agents.analyst import AnalystAgent
    from capstone.agents.coder import CoderAgent
    from capstone.agents.researcher import ResearcherAgent
    from capstone.agents.writer import WriterAgent

    orchestrator = MultiAgentOrchestrator(settings)
    orchestrator.register_agent(ResearcherAgent(settings))
    orchestrator.register_agent(CoderAgent(settings))
    orchestrator.register_agent(AnalystAgent(settings))
    orchestrator.register_agent(WriterAgent(settings))

    return orchestrator
