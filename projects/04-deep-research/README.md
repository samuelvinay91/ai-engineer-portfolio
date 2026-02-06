# 🔬 Deep Research

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](../../LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg?logo=docker)](Dockerfile)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-orange.svg)](https://langchain-ai.github.io/langgraph/)
[![Claude](https://img.shields.io/badge/Claude-Extended_Thinking-cc785c.svg)](https://docs.anthropic.com/en/docs/build-with-claude/extended-thinking)

A deep research engine that tackles complex questions through iterative planning, multi-source search, chain-of-thought and tree-of-thought reasoning, self-reflection, and automatic report generation. Powered by Claude's extended thinking mode to demonstrate how inference-time compute scaling produces dramatically better answers on hard problems.

![Deep Research Screenshot](https://via.placeholder.com/900x400?text=Deep+Research+Engine+%E2%80%93+Screenshot+Coming+Soon)

---

## 📚 What You'll Learn

| Concept | Description |
|---------|-------------|
| **Reasoning LLMs** | How models like Claude with extended thinking and OpenAI o1/o3 allocate extra compute at inference time |
| **Chain-of-Thought (CoT)** | Step-by-step reasoning that makes the model "show its work" |
| **Tree-of-Thought (ToT)** | Exploring multiple reasoning branches in parallel, evaluating, and pruning |
| **Self-Reflection** | The model critiques and revises its own answers when confidence is low |
| **Inference-Time Scaling** | Why spending more tokens thinking leads to better answers on complex tasks |
| **Research Planning** | Decomposing complex questions into dependency-ordered sub-questions |
| **Iterative Deepening** | Verifying findings, identifying gaps, and refining the research plan |
| **Report Generation** | Structured output with executive summaries, sections, citations, and confidence assessments |

---

## 🏗️ Architecture

### Research Workflow (LangGraph)

```
                          ┌──────────────────────────┐
                          │     Research Question     │
                          │ "What are the long-term   │
                          │  economic impacts of AI   │
                          │  on the labor market?"    │
                          └────────────┬─────────────┘
                                       │
                              ┌────────▼────────┐
                              │   PLAN NODE     │
                              │                 │
                              │ Decompose into  │
                              │ 3-7 sub-questions│
                              │ with dependency │
                              │ graph + priority│
                              └────────┬────────┘
                                       │
  ┌────────────────────────────────────┐│┌────────────────────────────────────┐
  │  ITERATION LOOP (depth 1..N)       │││                                    │
  │                                    │││                                    │
  │  ┌────────────────────────────┐    │││                                    │
  │  │      SEARCH NODE          │◄───┘││                                    │
  │  │                           │     ││                                    │
  │  │ Parallel web searches     │     ││                                    │
  │  │ via Tavily for each       │     ││                                    │
  │  │ ready sub-question        │     ││                                    │
  │  └────────────┬──────────────┘     ││                                    │
  │               │                    ││                                    │
  │  ┌────────────▼──────────────┐     ││                                    │
  │  │     ANALYSE NODE          │     ││                                    │
  │  │                           │     ││                                    │
  │  │ Extract key findings      │     ││                                    │
  │  │ from sources per          │     ││                                    │
  │  │ sub-question              │     ││                                    │
  │  └────────────┬──────────────┘     ││                                    │
  │               │                    ││                                    │
  │  ┌────────────▼──────────────┐     ││                                    │
  │  │      REASON NODE          │     ││                                    │
  │  │                           │     ││                                    │
  │  │ CoT + Extended Thinking   │     ││                                    │
  │  │ Synthesize findings per   │     ││                                    │
  │  │ sub-question              │     ││                                    │
  │  │ Self-reflect if low conf  │     ││                                    │
  │  └────────────┬──────────────┘     ││                                    │
  │               │                    ││                                    │
  │  ┌────────────▼──────────────┐     ││                                    │
  │  │     VERIFY NODE           │     ││                                    │
  │  │                           │     ││                                    │
  │  │ Cross-check findings      │     ││                                    │
  │  │ Identify contradictions   │     ││                                    │
  │  │ Find knowledge gaps       │     ││                                    │
  │  └────────────┬──────────────┘     ││                                    │
  │               │                    ││                                    │
  │       ┌───────▼───────┐            ││                                    │
  │       │ Gaps found    │            ││                                    │
  │       │ AND depth <   │────YES────▶││  ┌──────────────────────────────┐   │
  │       │ max_depth?    │            ││  │      ITERATE NODE           │   │
  │       └───────┬───────┘            ││  │                             │   │
  │            NO │                    ││  │  Refine plan:               │   │
  │               │                    ││  │  • Skip resolved questions  │   │
  │               │                    ││  │  • Add new sub-questions    │   │
  │               │                    ││  │  • Re-prioritize            │   │
  │               │                    │└──│                             │   │
  │               │                    │   └─────────────────────────────┘   │
  └───────────────┼────────────────────┘                                    │
                  │                                                         │
         ┌────────▼────────┐                                                │
         │  REPORT NODE    │                                                │
         │                 │                                                │
         │ Extended thinking│                                               │
         │ report generation│                                               │
         │                 │                                                │
         │ Output:         │                                                │
         │ • Title         │                                                │
         │ • Exec Summary  │                                                │
         │ • Key Findings  │                                                │
         │ • Sections      │                                                │
         │ • Citations     │                                                │
         │ • Confidence    │                                                │
         │ • Further       │                                                │
         │   Research      │                                                │
         └────────┬────────┘                                                │
                  │                                                         │
                  ▼                                                         │
              END                                                           │
```

### Conditional Edge Logic

```python
def should_iterate(state: ResearchState) -> Literal["iterate", "report"]:
    """Decide whether to research deeper or generate the report."""
    gaps = state.get("gaps", [])
    current_depth = state.get("current_depth", 0)
    max_depth = state.get("max_depth", 5)

    if gaps and current_depth < max_depth:
        return "iterate"    # Go deeper: refine plan → search again
    return "report"         # Satisfied: generate final report
```

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# From the repository root
docker build -f projects/04-deep-research/Dockerfile \
  -t deep-research .

docker run -p 8000:8000 \
  -e DEEP_RESEARCH_ANTHROPIC_API_KEY=your-key \
  -e DEEP_RESEARCH_TAVILY_API_KEY=your-tavily-key \
  deep-research
```

### Option 2: Local Development

```bash
python -m venv .venv
source .venv/bin/activate

pip install -e libs/common
pip install -e "projects/04-deep-research[dev]"

export DEEP_RESEARCH_ANTHROPIC_API_KEY=your-key
export DEEP_RESEARCH_TAVILY_API_KEY=your-tavily-key

cd projects/04-deep-research
python -m deep_research.main
```

The API will be available at **http://localhost:8000**. Interactive docs at **http://localhost:8000/docs**.

---

## 📡 API Reference

### Health Check

```bash
curl http://localhost:8000/health
```

### Start a Deep Research Task (Async)

```bash
# Start research (returns immediately with a task ID)
curl -X POST http://localhost:8000/api/v1/research \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the long-term economic impacts of generative AI on the global labor market?",
    "max_depth": 5
  }'

# Poll for results
curl http://localhost:8000/api/v1/research/{task_id}
```

### Stream Research Progress (SSE)

```bash
curl -N -X POST http://localhost:8000/api/v1/research/stream \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Compare the effectiveness of mRNA vs protein subunit COVID vaccines",
    "max_depth": 3
  }'
```

Events emitted: `planning`, `searching`, `analysing`, `reasoning`, `verifying`, `iterating`, `reporting`, `completed`.

Each event includes `task_id`, `status`, `message`, `progress_pct`, and `metadata`.

### Single Reasoning Query (CoT)

```bash
curl -X POST http://localhost:8000/api/v1/reason \
  -H "Content-Type: application/json" \
  -d '{
    "query": "A farmer has 17 sheep. All but 9 die. How many sheep does the farmer have left?",
    "strategy": "chain_of_thought",
    "use_extended_thinking": true
  }'
```

Available strategies: `direct`, `chain_of_thought`, `tree_of_thought`.

### Compare Reasoning Strategies Side by Side

```bash
curl -X POST http://localhost:8000/api/v1/compare-reasoning \
  -H "Content-Type: application/json" \
  -d '{
    "query": "If a ball is placed on top of a hill and rolls down, will it end up at the bottom? Consider the shape of the terrain, obstacles, and wind.",
    "strategies": ["direct", "chain_of_thought", "tree_of_thought"]
  }'
```

This returns a side-by-side comparison of each strategy's answer, confidence, reasoning steps, and token usage -- letting you see firsthand how additional inference-time compute improves results.

---

## 🔬 Implementation Deep Dive

### 1. Chain-of-Thought Prompting

**Chain-of-Thought (CoT)** instructs the model to reason step by step rather than jumping to an answer. This is implemented in `reasoning.py` with a structured system prompt:

```
System Prompt:
  "For EACH step you produce, output it in this format:
   [Step N]
   Description: <one-line summary>
   Reasoning: <detailed reasoning>
   Confidence: <0.0 to 1.0>"

Input: "A farmer has 17 sheep. All but 9 die. How many are left?"

WITHOUT CoT (direct):                  WITH CoT:
┌─────────────────────┐               ┌─────────────────────────────────┐
│ Answer: 8           │               │ [Step 1]                        │
│ Confidence: 0.50    │               │ Description: Parse the question │
│                     │               │ Reasoning: "All but 9 die"      │
│ (WRONG)             │               │ means 9 survive.                │
│                     │               │ Confidence: 0.95                │
│                     │               │                                 │
│                     │               │ [Step 2]                        │
│                     │               │ Description: Calculate answer   │
│                     │               │ Reasoning: 9 sheep remain       │
│                     │               │ alive regardless of the 17.     │
│                     │               │ Confidence: 0.95                │
│                     │               │                                 │
│                     │               │ [Final Answer]                  │
│                     │               │ 9 sheep                         │
│                     │               │ Overall Confidence: 0.95        │
│                     │               │                                 │
│                     │               │ (CORRECT)                       │
└─────────────────────┘               └─────────────────────────────────┘
```

**Why does CoT work?** By generating intermediate steps, the model effectively allocates more compute to the problem. Each step conditions the next, reducing the chance of errors propagating silently. Research shows CoT improves accuracy on math, logic, and multi-step reasoning by 15-40%.

**Self-Reflection Loop:** When the CoT confidence falls below the `confidence_threshold` (default: 0.7), the engine automatically triggers a self-reflection round:

```
CoT answer (confidence: 0.55) → Self-Reflection
  ↓
  Critique: "Step 2 assumes X without justification..."
  Revised answer: "..."
  New confidence: 0.82
```

### 2. Tree-of-Thought

Where CoT follows a single reasoning path, **Tree-of-Thought (ToT)** explores multiple paths simultaneously and picks the best one:

```
                         Root Question
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         Approach 1      Approach 2      Approach 3
         (score: 0.4)    (score: 0.85)   (score: 0.6)
         [PRUNED]             │          [PRUNED]
                              │
                 ┌────────────┼────────────┐
                 │            │            │
            Sub-idea 1   Sub-idea 2   Sub-idea 3
            (score: 0.7) (score: 0.9) (score: 0.5)
                              │          [PRUNED]
                              │
                     ┌────────┼────────┐
                     │        │        │
                     ...     ...      ...
                              │
                         Best Leaf
                         (score: 0.9)
                              │
                    ┌─────────▼──────────┐
                    │   SYNTHESIZE       │
                    │   Final answer     │
                    │   from best path   │
                    └────────────────────┘
```

**The ToT Algorithm (from `reasoning.py`):**

1. **Generate** -- Ask the LLM to produce `branching_factor` (default: 3) distinct approaches
2. **Evaluate** -- A separate LLM call scores each approach (0.0-1.0) for logical soundness, relevance, and promise
3. **Prune** -- Keep the top half, discard the rest
4. **Recurse** -- Expand surviving branches up to `tot_max_depth` (default: 3)
5. **Select** -- DFS to find the highest-scored leaf node
6. **Synthesize** -- Generate the final answer using the best reasoning path

**When to use ToT vs CoT:**

| Factor | Chain-of-Thought | Tree-of-Thought |
|--------|-----------------|-----------------|
| Best for | Problems with a clear solution path | Ambiguous problems with multiple valid approaches |
| Token cost | ~1x | ~6-15x (multiple branches) |
| Latency | Low (1 LLM call) | High (many sequential calls) |
| Accuracy gain | +15-40% over direct | +5-15% over CoT on hard problems |

### 3. Inference-Time Scaling

Traditional ML scaling improves models by training longer on more data (training-time scaling). **Inference-time scaling** takes a different approach: give the model more compute at inference time.

```
Scaling Dimension          What Changes                  Examples
─────────────────────────────────────────────────────────────────────
Training-time scaling      Model size, dataset, epochs   GPT-3 → GPT-4
Inference-time scaling     Thinking tokens, search depth  Standard → Extended Thinking

                ┌──────────────────────────────────────────┐
                │       Accuracy vs Inference Compute       │
                │                                          │
  Accuracy      │                     ╱── ToT              │
     ↑          │                   ╱                      │
     │          │               ╱──── CoT + reflection     │
     │          │            ╱╱                             │
     │          │        ╱╱──── CoT                         │
     │          │      ╱╱                                   │
     │          │   ╱╱──────── Direct                       │
     │          │ ╱╱                                        │
     │          ╱╱                                          │
     └──────────────────────────────────────────────────────┘
                         Tokens Used →
```

**How this project implements inference-time scaling:**

1. **Extended Thinking Mode:** Claude's extended thinking API allocates a dedicated thinking budget (default: 10,000 tokens) before generating the visible answer:

```python
response = await client.messages.create(
    model="claude-opus-4-6",
    max_tokens=16_384,
    temperature=1,               # Required for extended thinking
    thinking={
        "type": "enabled",
        "budget_tokens": 10_000,  # Up to 128K for hard problems
    },
    messages=[...],
)

# Response contains both thinking and visible content
for block in response.content:
    if block.type == "thinking":
        print(f"Internal reasoning: {block.thinking}")
    elif block.type == "text":
        print(f"Final answer: {block.text}")
```

2. **Adaptive Compute Allocation:** The `DeepResearcher` automatically gives harder sub-questions (those with lower initial confidence) more thinking budget.

3. **Strategy Comparison:** The `/api/v1/compare-reasoning` endpoint lets you see how `direct` (low compute) vs `chain_of_thought` (medium) vs `tree_of_thought` (high) trade off between cost and quality on the same question.

### 4. Deep Research Workflow

The full pipeline orchestrates all the above into a **7-stage iterative workflow**:

```
Stage 1: PLAN
  Input:   "What are the long-term economic impacts of AI on labor?"
  Process: LLM decomposes into 3-7 sub-questions with dependencies
  Output:  ResearchPlan with prioritized SubQuestion graph

  Example sub-questions:
    SQ-1: Historical precedents of technology displacing labor (Priority: 1)
    SQ-2: Current AI adoption rates across industries (Priority: 1)
    SQ-3: Economic models for AI-driven productivity gains (Priority: 2, depends on SQ-2)
    SQ-4: Policy responses and retraining programs (Priority: 3, depends on SQ-1, SQ-3)

Stage 2: SEARCH
  Input:   Ready sub-questions (all dependencies met)
  Process: Parallel Tavily searches with semaphore-limited concurrency
  Output:  SearchResult list linked to sub-questions

Stage 3: ANALYSE
  Input:   Search results per sub-question
  Process: LLM extracts key findings from raw source content
  Output:  ResearchFinding list (claim + evidence + sources)

Stage 4: REASON
  Input:   Findings per sub-question
  Process: CoT + extended thinking synthesis
  Output:  ReasoningResult with confidence score
  Side effect: Marks sub-question as completed in the plan

Stage 5: VERIFY
  Input:   All findings so far
  Process: LLM cross-checks for contradictions and gaps
  Output:  Gap list (empty = all verified)

Stage 6: ITERATE (conditional)
  Condition: Gaps exist AND current_depth < max_depth
  Process:   Refine the plan -- add new sub-questions, skip resolved ones
  Then:      Loop back to Stage 2

Stage 7: REPORT
  Input:   All findings, reasoning results, and sources
  Process: Extended thinking report generation
  Output:  ResearchReport with sections, citations, confidence assessment
```

**Example output structure:**

```markdown
# Research Report: Economic Impacts of AI on the Global Labor Market

## Executive Summary
AI is projected to automate 25-40% of current work tasks by 2035...

## Key Findings
1. Historical technology transitions created more jobs than they displaced...
2. Current AI adoption is concentrated in knowledge work sectors...

## Section: Historical Precedents
[detailed analysis with inline citations]

## Section: Current AI Adoption Rates
[detailed analysis with inline citations]

## Confidence Assessment
High confidence in near-term projections (0.85), moderate confidence
in long-term economic models (0.65) due to limited historical precedent.

## Areas for Further Research
- Impact on developing economies
- Role of AI in creating new job categories
- Effectiveness of retraining programs

## References
1. [McKinsey Global Institute - AI and the Future of Work](https://...)
2. [OECD Employment Outlook 2024](https://...)
```

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | FastAPI | Async REST API with SSE streaming |
| **Orchestration** | LangGraph | State graph with iterative deepening loop |
| **Reasoning Model** | Claude Opus 4.6 | Complex reasoning with extended thinking |
| **Fast Model** | Claude Sonnet 4.5 | Quick extractions, analysis, verification |
| **Extended Thinking** | Anthropic API | Inference-time compute scaling (up to 128K thinking tokens) |
| **Web Search** | Tavily API | Real-time information retrieval |
| **Streaming** | SSE-Starlette | Real-time progress events |
| **Data Models** | Pydantic v2 + dataclasses | Type-safe configuration and structured output |
| **Config** | Pydantic Settings | Environment-based configuration with env prefix |
| **Logging** | structlog | Structured JSON logging |
| **Database** | PostgreSQL + asyncpg | (Optional) persistent research storage |
| **Cache** | Redis | (Optional) result caching |
| **Containerization** | Docker | Multi-stage production builds |

---

## 📁 Project Structure

```
04-deep-research/
├── src/deep_research/
│   ├── __init__.py
│   ├── main.py              # Uvicorn entry point
│   ├── api.py               # FastAPI app: research, reason, compare-reasoning
│   ├── config.py            # Settings (models, reasoning params, research params, ToT config)
│   ├── reasoning.py         # ReasoningEngine: Direct, CoT, ToT, self-reflection
│   ├── planner.py           # ResearchPlanner: question decomposition, plan refinement
│   ├── researcher.py        # DeepResearcher: full pipeline orchestration + WebSearcher
│   ├── report.py            # ReportGenerator: structured report with citations
│   └── workflow.py          # LangGraph StateGraph: plan→search→analyse→reason→verify→report
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   ├── test_reasoning.py
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

Reasoning strategies are modular -- adding a new strategy means implementing a `_reason_X` method and registering it in the `dispatch` dict.

---

## 📄 License

This project is part of the AI Engineer Portfolio and is licensed under the [MIT License](../../LICENSE).
