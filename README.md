# AI Engineer Portfolio

> Production-grade, cloud-native AI applications covering the full spectrum of modern AI engineering — from LLM fundamentals to multi-agent systems with MCP & A2A protocols.

[![CI Pipeline](https://github.com/samuelvinay91/ai-engineer-portfolio/actions/workflows/ci.yml/badge.svg)](https://github.com/samuelvinay91/ai-engineer-portfolio/actions)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Architecture

```
ai-engineer-portfolio/
├── libs/common/              # Shared utilities, config, models
├── projects/
│   ├── 01-llm-playground/          # LLM fundamentals & text generation
│   ├── 02-customer-support-chatbot/ # Fine-tuning, PEFT/LoRA, prompts
│   ├── 03-ask-the-web-agent/       # Perplexity-like search agent
│   ├── 04-deep-research/           # Reasoning & inference-time scaling
│   ├── 05-image-generation/        # Diffusion models & T2I service
│   ├── 06-capstone-multiagent/     # Multi-agent orchestration platform
│   ├── 07-agent-rag/              # Advanced RAG with agent patterns
│   └── 08-mcp-a2a/               # MCP servers & A2A protocol
├── k8s/                      # Kubernetes manifests
├── .github/workflows/        # CI/CD pipelines
└── docker-compose.yml        # Local development environment
```

## Projects

### Project 1: LLM Playground
**Topics**: LLM Foundations, Pre-Training, Data Collection, Tokenization (BPE), Transformer Architecture, Text Generation

Interactive playground for exploring LLM fundamentals. Compare tokenizers (tiktoken, HuggingFace), experiment with generation strategies (greedy, beam search, top-k, top-p), and explore transformer architectures.

**Tech**: FastAPI, tiktoken, HuggingFace Tokenizers, Anthropic SDK, OpenAI SDK, SSE streaming

### Project 2: Customer Support Chatbot
**Topics**: Adaptation Techniques, Fine-tuning, PEFT, LoRA, Adapters, Prompt Engineering

Production chatbot with advanced prompt engineering, intent classification, conversation management, and a complete LoRA fine-tuning pipeline. Features versioned prompt templates and conversation state machine.

**Tech**: FastAPI, LangChain, Redis, PostgreSQL, PEFT/LoRA, Jinja2 templates

### Project 3: Ask-the-Web Agent (Perplexity Clone)
**Topics**: Agents Overview, Agency Levels, Workflow, Prompt Chaining, Routing

Perplexity-like search agent built with LangGraph. Features query routing, web search, content synthesis with citations, and fact checking. Demonstrates agentic patterns: prompt chaining, conditional routing, parallel execution.

**Tech**: FastAPI, LangGraph, Tavily Search, Anthropic SDK, SSE streaming

### Project 4: Deep Research Capability
**Topics**: Reasoning LLMs, OpenAI o-series, DeepSeek-R1, Inference-time Techniques, CoT Prompting

Deep research engine with Chain-of-Thought and Tree-of-Thought reasoning. Decomposes complex questions, executes multi-step research with iterative deepening, and generates structured reports with confidence scoring.

**Tech**: FastAPI, LangGraph, Claude Extended Thinking, Inference-time Scaling

### Project 5: Image Generation Service
**Topics**: VAE, GANs, Auto-regressive Models, Diffusion Models, Text-to-Image (T2I)

Cloud-native image generation service with multi-provider support (DALL-E, Stable Diffusion, FLUX). Features prompt enhancement, batch generation, image-to-image, and educational modules explaining generative architectures.

**Tech**: FastAPI, OpenAI DALL-E, Replicate, Pillow, S3 Storage

### Project 6: Capstone Multi-Agent Platform
**Topics**: Multi-Agent Orchestration, Supervisor Pattern, Shared Memory, Task Decomposition

Capstone project combining all techniques. Multi-agent platform with specialized agents (researcher, coder, analyst, writer) orchestrated by a supervisor via LangGraph. Features shared memory, task decomposition, and parallel execution.

**Tech**: FastAPI, LangGraph, PostgreSQL, Redis, Multi-Agent Orchestration

### Module 7: Agent & RAG System
**Topics**: Query Decomposition, Context Engineering, Memory, Meta-agent vs Planner vs Orchestrator, Single/Multi/Hierarchical RAG

Comprehensive RAG system implementing three architectures: single-agent, multi-agent, and hierarchical (A-RAG). Features document ingestion, semantic chunking, hybrid retrieval, reranking, and context engineering.

**Tech**: FastAPI, LangGraph, Qdrant, Cohere, LangChain Text Splitters

### Module 8: MCP & A2A Integration
**Topics**: Model Context Protocol, MCP Servers/Clients, Tool Discovery, A2A Protocol, AgentCards

Full MCP server and client implementation with tool/resource/prompt support. A2A protocol with AgentCards for agent discovery and inter-agent communication. Includes security module for prompt injection defense.

**Tech**: FastAPI, MCP SDK, A2A Protocol, AgentCards, SSE

## Tech Stack

| Category | Technologies |
|----------|-------------|
| **Language** | Python 3.11+ |
| **API Framework** | FastAPI, SSE, WebSocket |
| **LLM Providers** | Anthropic (Claude 4.5/4.6), OpenAI (GPT-4o), Open-source (Llama, DeepSeek) |
| **Agent Frameworks** | LangGraph, LangChain |
| **Vector Database** | Qdrant |
| **Databases** | PostgreSQL, Redis |
| **Containerization** | Docker, Docker Compose |
| **Orchestration** | Kubernetes, Kustomize |
| **CI/CD** | GitHub Actions |
| **Package Management** | uv (workspace monorepo) |
| **Code Quality** | Ruff, mypy, pytest |
| **Protocols** | MCP (Model Context Protocol), A2A (Agent-to-Agent) |

## Quick Start

### Prerequisites
- Python 3.11+
- Docker & Docker Compose
- uv (Python package manager)

### 1. Clone & Setup
```bash
git clone https://github.com/samuelvinay91/ai-engineer-portfolio.git
cd ai-engineer-portfolio
cp .env.example .env
# Edit .env with your API keys
```

### 2. Run with Docker Compose (recommended)
```bash
docker compose up -d
```

Services will be available at:
| Service | URL |
|---------|-----|
| LLM Playground | http://localhost:8001 |
| Customer Support Chatbot | http://localhost:8002 |
| Ask-the-Web Agent | http://localhost:8003 |
| Deep Research | http://localhost:8004 |
| Image Generation | http://localhost:8005 |
| Capstone Multi-Agent | http://localhost:8006 |
| Agent RAG | http://localhost:8007 |
| MCP & A2A | http://localhost:8008 |

### 3. Run Locally (individual project)
```bash
# Install dependencies
uv sync --all-packages

# Run a specific project
uv run uvicorn projects.01-llm-playground.src.llm_playground.main:app --port 8001
```

### 4. Run Tests
```bash
# All tests
uv run pytest

# Specific project
uv run pytest projects/01-llm-playground/tests/ -v
```

### 5. Deploy to Kubernetes
```bash
kubectl apply -k k8s/base/
# Then apply individual project manifests
kubectl apply -f projects/01-llm-playground/k8s/
```

## Models Used

| Use Case | Recommended Model | Alternative |
|----------|------------------|-------------|
| General Chat | Claude Sonnet 4.5 | GPT-4o |
| Complex Reasoning | Claude Opus 4.6 | OpenAI o3 |
| Code Generation | Claude Sonnet 4.5 | DeepSeek R1 |
| Embeddings | Cohere embed-v4 | OpenAI text-embedding-3-large |
| Image Generation | DALL-E 3 / FLUX 1.1 | Stable Diffusion 3.5 |
| Reranking | Cohere Rerank | LLM-based reranking |

## License

MIT License - see [LICENSE](LICENSE) for details.
