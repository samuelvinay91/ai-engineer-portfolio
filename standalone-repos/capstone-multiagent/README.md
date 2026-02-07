# Project 6: Capstone Multi-Agent AI Platform

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white)
![LangGraph](https://img.shields.io/badge/LangGraph-Orchestration-orange)

A production-grade **multi-agent orchestration platform** that decomposes complex tasks, routes subtasks to specialized AI agents (Researcher, Coder, Analyst, Writer), executes them in parallel, and aggregates results into coherent outputs. Built with **LangGraph** using the **supervisor pattern** for reliable, observable multi-agent coordination.

---

## What You'll Learn

- **Multi-Agent Systems** -- How to design, build, and orchestrate multiple specialized AI agents that collaborate to solve complex tasks no single agent could handle alone
- **Supervisor Pattern** -- A meta-agent (supervisor) that decomposes tasks, assigns work, monitors progress, and synthesizes results -- the most production-ready multi-agent architecture
- **LangGraph State Machines** -- Building directed graphs with typed state, conditional edges, and fan-out/fan-in execution using LangGraph's `StateGraph`
- **Task Decomposition** -- How an LLM breaks complex requests into structured subtask plans with agent assignments, dependencies, and constraints
- **Parallel Agent Execution** -- Running multiple agents concurrently with `asyncio.Semaphore` for controlled parallelism, automatic error handling, and result aggregation
- **Iterative Refinement** -- The supervisor reviews intermediate results and creates additional subtasks if the output is incomplete, looping up to a configurable depth
- **Shared Memory Architecture** -- Three-tier memory system (short-term conversation, working scratch-pad, long-term persistent) enabling context sharing across agents and sessions
- **Specialized Agent Design** -- Building agents with distinct capabilities, confidence scoring for task routing, structured output parsing, and graceful error handling

---

## Architecture

```
                          User Request
                               |
                               v
                    +--------------------+
                    |   FastAPI Server   |
                    |    (Port 8000)     |
                    +--------+-----------+
                             |
                    +--------v-----------+
                    |    ORCHESTRATOR     |
                    |   (LangGraph)       |
                    +--------+-----------+
                             |
                    +--------v-----------+
                    |    SUPERVISOR       |  <--- Decomposes task into subtasks
                    |  (Claude Sonnet)    |       Assigns agents, creates plan
                    +--------+-----------+
                             |
              +--------------+--------------+
              |              |              |
     +--------v---+  +------v-----+  +-----v------+
     | Researcher |  |   Coder    |  |  Analyst   |  +--------+
     | - Research |  | - Generate |  | - Analyze  |  | Writer |
     | - Fact-chk |  | - Review   |  | - Trends   |  | - Blog |
     | - Synthesize| | - Debug    |  | - Anomaly  |  | - Docs |
     +--------+---+  +------+-----+  +-----+------+  +---+----+
              |              |              |              |
              +--------------+--------------+--------------+
                             |
                    +--------v-----------+
                    |  REFINE / LOOP?    |  <--- Reviews results, creates
                    |  (Supervisor)      |       more subtasks if needed
                    +--------+-----------+
                             |
                    +--------v-----------+
                    |    AGGREGATOR      |  <--- Synthesizes all agent
                    |  (Claude Sonnet)   |       outputs into final response
                    +--------+-----------+
                             |
                    +--------v-----------+
                    |    Memory Manager   |
                    | Short | Work | Long |
                    +--------------------+
```

### LangGraph State Machine

```
    START --> [supervisor] --> [execute_agents] --+--> [refine] --> [execute_agents]
                                                  |                       |
                                                  +--> [aggregate] <------+
                                                            |
                                                           END
```

- **supervisor** -- Decomposes request into subtasks with agent assignments
- **execute_agents** -- Dispatches subtasks to agents in parallel
- **refine** -- Reviews results; creates additional subtasks if incomplete
- **aggregate** -- Synthesizes all results into a unified final output

---

## Quick Start

### Docker (Recommended)

```bash
# Build the image
docker build -t capstone-multiagent -f Dockerfile .

# Run with API key
docker run -p 8000:8000 \
  -e CAPSTONE_ANTHROPIC_API_KEY=sk-ant-your-key-here \
  capstone-multiagent

# Verify it's running
curl http://localhost:8000/health
```

### Local Development

```bash
# Navigate to the project
# Already in project root

# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Configure environment
cat > .env << 'EOF'
CAPSTONE_ANTHROPIC_API_KEY=sk-ant-your-key-here
CAPSTONE_ENVIRONMENT=local
CAPSTONE_DEBUG=true
CAPSTONE_LOG_LEVEL=DEBUG
EOF

# Start the server
uvicorn capstone.main:app --host 0.0.0.0 --port 8000 --reload

# Open the API docs
open http://localhost:8000/docs
```

---

## API Reference

### Submit a Complex Task

```bash
curl -X POST http://localhost:8000/api/v1/tasks \
  -H "Content-Type: application/json" \
  -d '{
    "request": "Research the current state of quantum computing, write a technical blog post about it, and include code examples showing quantum circuit simulation in Python.",
    "session_id": "my-session-01"
  }'
```

**Response:**
```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "completed",
  "plan": "1) Research quantum computing state-of-the-art. 2) Generate Python quantum circuit code. 3) Write technical blog post integrating research and code.",
  "final_output": "# The State of Quantum Computing in 2025\n\n...",
  "agent_results": [
    {"agent_name": "researcher", "status": "completed", "confidence": 0.87},
    {"agent_name": "coder", "status": "completed", "confidence": 0.92},
    {"agent_name": "writer", "status": "completed", "confidence": 0.90}
  ],
  "session_id": "my-session-01"
}
```

### Stream Task Execution (SSE)

```bash
curl -N http://localhost:8000/api/v1/tasks/stream \
  -H "Content-Type: application/json" \
  -d '{
    "request": "Analyze the pros and cons of microservices vs monoliths and write a decision framework.",
    "session_id": "stream-demo"
  }'
```

Events emitted: `status`, `plan`, `agent_result`, `final`, `error`.

### Get Task Status

```bash
curl http://localhost:8000/api/v1/tasks/a1b2c3d4e5f6
```

### List All Agents

```bash
curl http://localhost:8000/api/v1/agents
```

**Response:**
```json
[
  {"name": "researcher", "description": "Conducts deep, multi-step research...", "capabilities": ["deep_research", "fact_checking", "literature_review", ...]},
  {"name": "coder", "description": "Generates, reviews, and debugs code...", "capabilities": ["code_generation", "code_review", "debugging", ...]},
  {"name": "analyst", "description": "Statistical analysis, trend identification...", "capabilities": ["statistical_analysis", "trend_analysis", ...]},
  {"name": "writer", "description": "Produces blog posts, documentation...", "capabilities": ["blog_writing", "documentation", "editing", ...]}
]
```

### Execute a Specific Agent Directly

```bash
curl -X POST http://localhost:8000/api/v1/agents/coder/execute \
  -H "Content-Type: application/json" \
  -d '{
    "description": "Write a Python decorator that retries failed async functions with exponential backoff.",
    "constraints": ["Use only standard library", "Include type hints"]
  }'
```

### Inspect Session Memory

```bash
curl http://localhost:8000/api/v1/memory/my-session-01
```

---

## Implementation Deep Dive

### 1. Supervisor Pattern with LangGraph

The orchestrator is built as a `StateGraph` where each node is an async function that reads and writes to a shared `OrchestratorState`:

```python
class OrchestratorState(BaseModel):
    task_id: str                    # Unique task identifier
    user_request: str               # Original user input
    plan: str                       # Supervisor's execution plan
    subtasks: list[AgentTask]       # Decomposed subtasks
    agent_results: list[AgentResult] # Results from agents
    current_iteration: int          # Refinement loop counter
    final_output: str               # Aggregated final response
    status: str                     # pending | running | completed | partial
```

The graph wires four nodes with conditional routing:

```python
graph = StateGraph(OrchestratorState)
graph.add_node("supervisor", self._supervisor_node)
graph.add_node("execute_agents", self._execute_agents_node)
graph.add_node("refine", self._refine_node)
graph.add_node("aggregate", self._aggregate_node)

graph.set_entry_point("supervisor")
graph.add_edge("supervisor", "execute_agents")
graph.add_conditional_edges("execute_agents", self._should_refine_or_aggregate,
    {"refine": "refine", "aggregate": "aggregate"})
graph.add_conditional_edges("refine", self._should_continue_or_aggregate,
    {"execute_agents": "execute_agents", "aggregate": "aggregate"})
graph.add_edge("aggregate", END)
```

### 2. Task Decomposition

The supervisor uses Claude Sonnet with a structured prompt that includes a table of available agents and their capabilities. It produces a JSON plan:

```json
{
  "plan": "Research AI trends, then write a blog post with code examples.",
  "subtasks": [
    {
      "description": "Research current AI industry trends for 2025",
      "preferred_agent": "researcher",
      "depends_on": [],
      "constraints": ["Include citations"],
      "priority": 1
    },
    {
      "description": "Generate Python code demonstrating key AI concepts",
      "preferred_agent": "coder",
      "depends_on": [],
      "constraints": ["Include type hints", "Add docstrings"],
      "priority": 2
    }
  ]
}
```

### 3. Specialized Agents

Each agent inherits from `BaseAgent` and implements two core methods:

- **`execute(task)`** -- Carries out the task using an LLM with a domain-specific system prompt, returning structured output with confidence scores
- **`can_handle(task)`** -- Returns a confidence score (0.0-1.0) based on keyword matching and task-agent affinity, enabling the orchestrator to route work to the best specialist

| Agent | Model | Capabilities | Confidence Keywords |
|-------|-------|-------------|-------------------|
| Researcher | Claude Sonnet | Deep research, fact-checking, literature review, trend analysis | research, investigate, analyze, compare, survey |
| Coder | Claude Sonnet | Code generation, review, debugging, test writing, explanation | code, implement, program, function, debug, test |
| Analyst | Claude Sonnet | Statistical analysis, trend identification, anomaly detection, data insights | analyze, statistics, data, trend, anomaly, metrics |
| Writer | Claude Sonnet | Blog posts, documentation, reports, creative writing, SEO | write, blog, article, document, report, edit |

The `_safe_execute` wrapper on the base class provides timing, structured logging, and automatic error conversion to `AgentResult` with `FAILED` status.

### 4. Parallel Execution with Controlled Concurrency

Agents run concurrently using `asyncio.gather` with a semaphore limiting parallelism:

```python
semaphore = asyncio.Semaphore(self._settings.max_parallel_agents)  # default: 4

async def _run_agent_task(agent_name, task):
    async with semaphore:
        # Inject context from prior results
        prior_outputs = {r.agent_name: r.output for r in state.agent_results if r.status == "completed"}
        task.context["prior_agent_outputs"] = prior_outputs
        return await agent._safe_execute(task)

results = await asyncio.gather(*coroutines, return_exceptions=True)
```

### 5. Three-Tier Memory System

**Short-Term Memory** (`ShortTermMemory`):
- Sliding window of recent conversation messages per session
- Configurable capacity (default: 50 messages)
- Serializable to/from JSON for Redis persistence
- Provides `get_context_window()` for LLM prompt injection

**Working Memory** (`WorkingMemory`):
- Keyed by `(task_id, key)` for intermediate agent results
- Automatic TTL expiration (default: 3600 seconds)
- Used during orchestration for sharing data between agents

**Long-Term Memory** (`LongTermMemory`):
- PostgreSQL-backed persistent storage using SQLAlchemy async
- Content-hash deduplication prevents duplicate entries
- Keyword-based relevance scoring (weighted: 60% keyword overlap, 40% stored relevance)
- Production-ready: swap to pgvector for embedding-based similarity

The `MemoryManager` facade unifies all three tiers behind a single interface.

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | FastAPI 0.115+ | Async REST API with SSE streaming |
| Orchestration | LangGraph 0.2+ | State machine for multi-agent coordination |
| LLM | Anthropic Claude Sonnet | Supervisor reasoning and agent execution |
| LLM Framework | LangChain Core, LangChain Anthropic | LLM abstraction and message handling |
| Streaming | SSE-Starlette | Server-Sent Events for real-time updates |
| Short-Term Memory | In-memory + Redis 5.0+ | Conversation context with persistence |
| Long-Term Memory | PostgreSQL + SQLAlchemy 2.0 | Persistent knowledge with keyword retrieval |
| Async DB | asyncpg 0.30+ | Async PostgreSQL driver |
| Validation | Pydantic 2.6+ | Typed state schemas and API models |
| Logging | structlog 24.1+ | Structured JSON logging with agent context |
| Runtime | Python 3.11+ | Async/await, StrEnum, type hints |

---

## Project Structure

```
06-capstone-multiagent/
├── Dockerfile                      # Multi-stage production build
├── pyproject.toml                  # Dependencies and build config
├── src/
│   └── capstone/
│       ├── __init__.py
│       ├── main.py                 # Uvicorn entry point
│       ├── config.py               # Settings (models, memory, orchestration limits)
│       ├── api.py                  # FastAPI endpoints (tasks, agents, memory, streaming)
│       ├── orchestrator.py         # LangGraph StateGraph: supervisor, execute, refine, aggregate
│       ├── memory.py               # ShortTermMemory, WorkingMemory, LongTermMemory, MemoryManager
│       └── agents/
│           ├── __init__.py
│           ├── base.py             # BaseAgent ABC, AgentTask, AgentResult, TaskStatus
│           ├── researcher.py       # Deep research with question decomposition and synthesis
│           ├── coder.py            # Code generation, review, debugging, and testing
│           ├── analyst.py          # Statistical analysis and trend identification
│           └── writer.py           # Blog posts, documentation, reports, and editing
└── tests/
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CAPSTONE_ANTHROPIC_API_KEY` | `""` | Anthropic API key (required) |
| `CAPSTONE_OPENAI_API_KEY` | `""` | OpenAI API key (optional) |
| `CAPSTONE_SUPERVISOR_MODEL` | `claude-sonnet-4-5-20250929` | Model for supervisor reasoning |
| `CAPSTONE_RESEARCHER_MODEL` | `claude-sonnet-4-5-20250929` | Model for researcher agent |
| `CAPSTONE_CODER_MODEL` | `claude-sonnet-4-5-20250929` | Model for coder agent |
| `CAPSTONE_MAX_PARALLEL_AGENTS` | `4` | Max concurrent agent executions |
| `CAPSTONE_MAX_SUBTASKS` | `8` | Max subtasks per decomposition |
| `CAPSTONE_MAX_AGENT_RETRIES` | `2` | Retry limit per agent |
| `CAPSTONE_AGENT_TIMEOUT_SECONDS` | `120` | Per-agent execution timeout |
| `CAPSTONE_REDIS_URL` | `redis://localhost:6379/0` | Redis for memory persistence |
| `CAPSTONE_DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL for long-term memory |
| `CAPSTONE_PORT` | `8000` | Server port |

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Write tests for new functionality
4. Ensure all tests pass (`pytest`)
5. Submit a pull request

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
