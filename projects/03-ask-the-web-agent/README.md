# 🌐 Ask the Web Agent

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](../../LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg?logo=docker)](Dockerfile)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)

A Perplexity-like search agent that takes a natural-language question, routes it through an intelligent pipeline, searches the web, synthesizes a cited answer, and fact-checks the result. Built with LangGraph to demonstrate core agentic patterns: routing, prompt chaining, tool use, and conditional workflows.

![Ask the Web Screenshot](https://via.placeholder.com/900x400?text=Ask+the+Web+Agent+%E2%80%93+Screenshot+Coming+Soon)

---

## 📚 What You'll Learn

| Concept | Description |
|---------|-------------|
| **Agents vs LLMs** | The difference between a plain LLM call and an agentic system with decision-making |
| **Agency Levels** | From simple prompt chaining to fully autonomous agent loops |
| **LangGraph Workflows** | State graphs, nodes, edges, conditional routing, compiled runnables |
| **Prompt Chaining** | Multi-step LLM pipelines: extract -> organize -> generate -> follow-up |
| **Query Routing** | LLM-powered classification to pick the optimal processing strategy |
| **Web Search + Synthesis** | Real-time information retrieval with Tavily, content cleaning, deduplication |
| **Fact Checking** | Cross-referencing claims against sources for verification |
| **Citation Generation** | Inline `[n]` citations linking every claim to its source |

---

## 🏗️ Architecture

### LangGraph Workflow

```
                              ┌───────────────────────────┐
                              │       User Query          │
                              │  "What caused the 2024    │
                              │   CrowdStrike outage?"    │
                              └─────────────┬─────────────┘
                                            │
                                   ┌────────▼────────┐
                                   │   ROUTE NODE    │
                                   │                 │
                                   │ Classifies into:│
                                   │ • simple_search │
                                   │ • deep_research │
                                   │ • calculation   │
                                   │ • creative      │
                                   └────────┬────────┘
                                            │
                           ┌────────────────┼─────────────────┐
                           │                │                  │
                    creative?        search-needed?      calculation?
                           │                │                  │
                           │       ┌────────▼────────┐         │
                           │       │  SEARCH NODE    │         │
                           │       │                 │         │
                           │       │ • Tavily API    │         │
                           │       │ • Parallel      │         │
                           │       │   queries       │         │
                           │       │ • Deduplication │         │
                           │       │ • Content       │         │
                           │       │   cleaning      │         │
                           │       └────────┬────────┘         │
                           │                │                  │
                           └────────┬───────┘──────────────────┘
                                    │
                           ┌────────▼────────┐
                           │ SYNTHESISE NODE │
                           │                 │
                           │ Prompt Chain:   │
                           │ 1. Extract facts│
                           │ 2. Organize     │
                           │ 3. Generate     │
                           │    answer with  │
                           │    citations    │
                           │ 4. Follow-up    │
                           │    questions    │
                           └────────┬────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
             creative route?   has error?    normal route
                    │               │               │
                    │               │      ┌────────▼────────┐
                    │               │      │ FACT CHECK NODE │
                    │               │      │                 │
                    │               │      │ • Extract claims│
                    │               │      │ • Cross-ref     │
                    │               │      │   sources       │
                    │               │      │ • Score each    │
                    │               │      │   claim         │
                    │               │      │ • Flag conflicts│
                    │               │      └────────┬────────┘
                    │               │               │
                    └───────┬───────┘───────────────┘
                            │
                   ┌────────▼────────┐
                   │  RESPOND NODE   │
                   │                 │
                   │ Final output:   │
                   │ • Answer (md)   │
                   │ • Citations     │
                   │ • Follow-ups    │
                   │ • Fact-check    │
                   │   report        │
                   │ • Elapsed time  │
                   └─────────────────┘
```

### Conditional Routing Logic

```python
# After ROUTE node: should we search or skip?
def should_search(state):
    if state["routing_decision"].route == "creative":
        return "synthesise"   # Skip search entirely
    return "search"           # All other routes search

# After SYNTHESISE node: should we fact-check?
def should_fact_check(state):
    if state["route_used"] == "creative":
        return "respond"      # No sources to check against
    if state.get("error"):
        return "respond"      # Don't fact-check errors
    return "fact_check"       # Normal path
```

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# From the repository root
docker build -f projects/03-ask-the-web-agent/Dockerfile \
  -t ask-the-web-agent .

docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=your-key \
  -e TAVILY_API_KEY=your-tavily-key \
  ask-the-web-agent
```

### Option 2: Local Development

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e libs/common
pip install -e "projects/03-ask-the-web-agent[dev]"

export ANTHROPIC_API_KEY=your-key
export TAVILY_API_KEY=your-tavily-key   # Get one free at tavily.com

cd projects/03-ask-the-web-agent
python -m ask_the_web.main
```

The API will be available at **http://localhost:8000**. Interactive docs at **http://localhost:8000/docs**.

> **Note:** You need a [Tavily API key](https://tavily.com) for web search. The free tier includes 1,000 searches/month.

---

## 📡 API Reference

### Health Check

```bash
curl http://localhost:8000/health
```

### Ask a Question (Full Pipeline)

```bash
curl -X POST http://localhost:8000/api/v1/ask \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the main differences between React and Vue.js in 2025?",
    "max_sources": 8,
    "fact_check": true
  }'
```

Response includes the answer with inline citations, source list, follow-up questions, fact-check report, route used, and timing.

### Ask with Streaming (SSE)

```bash
curl -N -X POST http://localhost:8000/api/v1/ask/stream \
  -H "Content-Type: application/json" \
  -d '{"query": "Explain the recent advances in nuclear fusion energy"}'
```

Events emitted:
- `route` -- routing decision (JSON)
- `sources` -- search results metadata (JSON)
- `token` -- individual answer tokens (text)
- `done` -- final metadata (JSON)
- `error` -- on failure (JSON)

### Raw Web Search (No Synthesis)

```bash
curl -X POST http://localhost:8000/api/v1/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "LangGraph state machine tutorial",
    "strategy": "general",
    "max_results": 5
  }'
```

Strategies: `general`, `news`, `deep`.

### Search History

```bash
curl "http://localhost:8000/api/v1/history?limit=10"
```

---

## 🔬 Implementation Deep Dive

### 1. What Are Agents?

The term "agent" is heavily overloaded in AI. Here is a practical spectrum:

```
Level 0: Simple LLM Call
  User → LLM → Answer
  (No decisions, no tools, no loops)

Level 1: Prompt Chain (this project's synthesis step)
  User → LLM₁ → LLM₂ → LLM₃ → Answer
  (Fixed sequence, each step feeds the next)

Level 2: Router + Chain (this project)
  User → Router LLM → [Branch A or Branch B] → Answer
  (LLM makes a decision about which path to take)

Level 3: Tool-Using Agent
  User → LLM → [Tool Call] → LLM → [Tool Call] → Answer
  (LLM decides WHEN and WHICH tools to use)

Level 4: Autonomous Agent Loop
  User → LLM → [Plan → Execute → Observe → Reflect]ⁿ → Answer
  (LLM drives an open-ended loop until the task is done)
```

This project implements **Level 2** (routing + conditional workflows) with elements of **Level 3** (web search as a tool). The LangGraph framework makes it easy to compose these patterns into a reliable, observable pipeline.

**Key distinction:** An *agentic system* is not the same as an *agent*. This project is an agentic system -- it uses LLM-powered decisions (routing, fact-checking) within a structured workflow. A fully autonomous agent would make all decisions dynamically without a predefined graph.

### 2. LangGraph Workflow: State, Nodes, and Edges

LangGraph models your pipeline as a **directed graph** where:

- **State** is a TypedDict flowing through the graph
- **Nodes** are async functions that transform state
- **Edges** connect nodes (unconditional or conditional)

```python
# Define the shared state
class WorkflowState(TypedDict, total=False):
    query: str                              # Input
    routing_decision: RoutingDecision       # After route node
    search_results: list[SearchResult]      # After search node
    answer: str                             # After synthesise node
    fact_check_report: FactCheckReport      # After fact_check node
    citations: list[dict]                   # Accumulated citations
    error: str | None                       # Error tracking

# Build the graph
graph = StateGraph(WorkflowState)

# Add nodes (each is an async function: state -> state)
graph.add_node("route", route_node)
graph.add_node("search", search_node)
graph.add_node("synthesise", synthesise_node)
graph.add_node("fact_check", fact_check_node)
graph.add_node("respond", respond_node)

# Conditional edge: route → search OR synthesise
graph.add_conditional_edges("route", should_search, {
    "search": "search",
    "synthesise": "synthesise",
})

# Linear edges
graph.add_edge("search", "synthesise")

# Conditional edge: synthesise → fact_check OR respond
graph.add_conditional_edges("synthesise", should_fact_check, {
    "fact_check": "fact_check",
    "respond": "respond",
})

graph.add_edge("fact_check", "respond")
graph.add_edge("respond", END)
```

The compiled graph is invoked with a single call:

```python
result = await compiled_graph.ainvoke({"query": "What is quantum computing?"})
```

### 3. Prompt Chaining: The Synthesis Pipeline

The `SynthesisAgent` demonstrates **prompt chaining** -- breaking a complex task into sequential LLM calls where each step's output feeds the next:

```
Step 1: EXTRACT FACTS
  Input:  query + raw source content
  Output: JSON array of {fact, source_indices} objects
  Goal:   Distill relevant information from noisy web pages

          ↓

Step 2: ORGANIZE FACTS
  Input:  query + extracted facts
  Output: grouped and ranked facts under headings
  Goal:   Remove redundancy, prioritize by relevance

          ↓

Step 3: GENERATE ANSWER
  Input:  query + organized facts + source reference list
  Output: Markdown answer with inline [n] citations
  Goal:   Write a coherent, well-cited response

          ↓

Step 4: FOLLOW-UP QUESTIONS
  Input:  query + answer summary
  Output: JSON array of 3 follow-up questions
  Goal:   Anticipate what the user wants to learn next
```

**Why chain instead of one big prompt?** Each step has a focused objective and can be debugged independently. The extract step alone reduces a 20-page source document to 15 key facts, making the generation step much more reliable.

### 4. Building a Search Agent

The search pipeline has three stages:

**Query Routing:** The `QueryRouterAgent` classifies the query into one of four routes. This is a single LLM call with JSON output and temperature=0 for determinism:

| Route | Example Query | Behavior |
|-------|--------------|----------|
| `simple_search` | "What is the capital of France?" | 1-2 searches, standard synthesis |
| `deep_research` | "Compare economic policies of recent US presidents" | Multiple search rounds, cross-referencing |
| `calculation` | "What is 15% of $3,450?" | Optional search, math-focused synthesis |
| `creative` | "Write a poem about autumn" | Skip search entirely, pure generation |

**Web Search:** The `WebSearchAgent` wraps the Tavily API with:
- Concurrent search execution across multiple reformulated queries
- HTML content cleaning via BeautifulSoup and markdownify
- URL normalization and content-hash deduplication
- Configurable strategies: `general`, `news`, `deep` (advanced depth)

**Fact Checking:** The `FactCheckerAgent` takes the final answer and cross-references it against the original sources:

```json
{
  "claims": [
    {
      "claim": "React was created by Facebook in 2013",
      "supported_by": [1, 3],
      "contradicted_by": [],
      "confidence": 0.95,
      "status": "verified"
    },
    {
      "claim": "Vue.js has a larger market share than React",
      "supported_by": [],
      "contradicted_by": [2, 4],
      "confidence": 0.1,
      "status": "contradicted",
      "note": "Multiple sources indicate React has larger market share"
    }
  ],
  "overall_confidence": 0.82,
  "conflicts_found": 1
}
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | FastAPI | Async REST API with SSE streaming |
| **Orchestration** | LangGraph | State graph workflow with conditional routing |
| **LLM Provider** | Anthropic Claude (via LangChain) | Routing, synthesis, fact-checking |
| **Web Search** | Tavily API | Real-time web search with content extraction |
| **Content Cleaning** | BeautifulSoup, markdownify | HTML to clean Markdown conversion |
| **HTTP Client** | httpx | Async HTTP requests to Tavily |
| **Streaming** | SSE-Starlette | Server-Sent Events for progressive answers |
| **Data Validation** | Pydantic v2 | Request/response models, structured agent output |
| **Config** | Pydantic Settings | Environment-based configuration |
| **Logging** | structlog | Structured JSON logging |
| **Containerization** | Docker | Multi-stage production builds |
| **Orchestration** | Kubernetes | Deployment manifests |

---

## 📁 Project Structure

```
03-ask-the-web-agent/
├── src/ask_the_web/
│   ├── __init__.py
│   ├── main.py              # Uvicorn entry point
│   ├── api.py               # FastAPI app: /ask, /ask/stream, /search, /history
│   ├── config.py            # Settings (API keys, model config, workflow tuning)
│   ├── workflow.py           # LangGraph StateGraph: nodes, edges, conditional routing
│   └── agents/
│       ├── __init__.py
│       ├── router.py         # QueryRouterAgent: classifies queries into routes
│       ├── searcher.py       # WebSearchAgent: Tavily search, cleaning, deduplication
│       ├── synthesizer.py    # SynthesisAgent: 4-step prompt chain with citations
│       └── fact_checker.py   # FactCheckerAgent: claim verification against sources
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   └── test_workflow.py
├── k8s/
│   └── deployment.yaml
├── Dockerfile
├── pyproject.toml
└── README.md
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

---

## 📄 License

This project is part of the AI Engineer Portfolio and is licensed under the [MIT License](../../LICENSE).
