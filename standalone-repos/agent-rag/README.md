# Project 7: Advanced Agent & RAG System

![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue?logo=python&logoColor=white)
![License: MIT](https://img.shields.io/badge/License-MIT-green)
![Docker](https://img.shields.io/badge/Docker-Ready-blue?logo=docker&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-Vector%20DB-red)

An advanced **Retrieval-Augmented Generation** system implementing three distinct RAG architecture patterns -- **Single-Agent**, **Multi-Agent**, and **Hierarchical (A-RAG)** -- with a full document ingestion pipeline, multiple chunking strategies, hybrid retrieval with reranking, context engineering, and agent memory. Compare all three patterns side-by-side on the same query.

---

## What You'll Learn

- **RAG Fundamentals** -- The complete pipeline from raw documents to grounded LLM answers: Load, Chunk, Embed, Store, Retrieve, Rerank, Generate
- **Three RAG Architecture Patterns** -- When to use single-agent (simple Q&A), multi-agent (collaborative retriever-analyzer-generator-critic), or hierarchical (adaptive planning with multi-granularity retrieval)
- **Document Ingestion Pipeline** -- Loading PDF, DOCX, TXT, MD, and HTML files; splitting them with multiple chunking strategies; embedding with OpenAI/Cohere; storing in Qdrant
- **Chunking Strategies** -- Fixed-size, semantic, hierarchical (document/section/paragraph), and recursive character splitting with configurable overlap
- **Hybrid Retrieval** -- Combining keyword search (BM25-style) and semantic search (dense embeddings) using Reciprocal Rank Fusion (RRF)
- **Reranking** -- Three reranker implementations: Cohere Rerank API (cross-encoder), LLM-based relevance scoring, and a lightweight cross-encoder with keyword overlap fallback
- **Context Engineering** -- Token budget management, passage ranking (retrieval score + position + diversity + level bonuses), deduplication via shingling, extractive compression, and sentence-boundary truncation
- **Agent Memory** -- Conversation memory (episodic), semantic memory (long-term knowledge), and working memory for multi-turn RAG sessions
- **Query Processing** -- Query rewriting strategies: HyDE (Hypothetical Document Embeddings), step-back prompting, and multi-query decomposition

---

## Architecture

```
                        +-------------------+
                        |   FastAPI Server   |
                        |    (Port 8007)     |
                        +--------+----------+
                                 |
              +------------------+-------------------+
              |                  |                   |
    +---------v--------+ +------v-------+ +---------v----------+
    |  Single-Agent    | | Multi-Agent  | |   Hierarchical     |
    |     RAG          | |    RAG       | |    RAG (A-RAG)     |
    | (Retrieve +      | | (Retriever + | | (Planner + Multi-  |
    |  Generate)       | |  Analyzer +  | |  Granularity       |
    +--------+---------+ |  Generator + | |  Retrieval +       |
             |           |  Critic)     | |  Validation)       |
             |           +------+-------+ +---------+----------+
             |                  |                   |
             +------------------+-------------------+
                                |
                    +-----------v-----------+
                    |    Context Builder     |
                    | (Rank, Compress, Build)|
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |   Hybrid Retriever    |
                    | Keyword + Semantic    |
                    |   + RRF Fusion        |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |     Reranker          |
                    | Cohere | LLM | Cross  |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |    Qdrant Vector DB    |
                    | (Embeddings Store)     |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |  Ingestion Pipeline    |
                    | Load -> Chunk -> Embed |
                    |        -> Store        |
                    +-----------------------+
```

---

## Quick Start

### Docker (Recommended)

```bash
# Start Qdrant vector database
docker run -d -p 6333:6333 qdrant/qdrant

# Build the application
docker build -t agent-rag -f Dockerfile .

# Run with API keys
docker run -p 8007:8007 \
  -e AGENT_RAG_ANTHROPIC_API_KEY=sk-ant-your-key-here \
  -e AGENT_RAG_OPENAI_API_KEY=sk-your-key-here \
  -e AGENT_RAG_COHERE_API_KEY=your-cohere-key \
  -e AGENT_RAG_QDRANT_URL=http://host.docker.internal:6333 \
  agent-rag

# Verify it's running
curl http://localhost:8007/health
```

### Local Development

```bash
# Navigate to the project
# Already in project root

# Create virtual environment
python -m venv .venv && source .venv/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Start Qdrant (Docker)
docker run -d -p 6333:6333 qdrant/qdrant

# Configure environment
cat > .env << 'EOF'
AGENT_RAG_ANTHROPIC_API_KEY=sk-ant-your-key-here
AGENT_RAG_OPENAI_API_KEY=sk-your-key-here
AGENT_RAG_COHERE_API_KEY=your-cohere-key
AGENT_RAG_QDRANT_URL=http://localhost:6333
AGENT_RAG_CHUNK_STRATEGY=recursive
AGENT_RAG_RETRIEVAL_STRATEGY=hybrid
AGENT_RAG_RERANKER_ENABLED=true
EOF

# Start the server
python -m agent_rag.main

# Open the API docs
open http://localhost:8007/docs
```

---

## API Reference

### Ingest Documents

```bash
# Ingest text strings
curl -X POST http://localhost:8007/api/v1/ingest \
  -H "Content-Type: application/json" \
  -d '{
    "texts": [
      "Retrieval-Augmented Generation (RAG) enhances LLM responses by grounding them in external knowledge...",
      "Vector databases store embeddings as high-dimensional points, enabling similarity search..."
    ],
    "source_prefix": "rag_tutorial"
  }'

# Upload a file (PDF, DOCX, TXT, MD, HTML)
curl -X POST http://localhost:8007/api/v1/ingest/file \
  -F "file=@research_paper.pdf"
```

### Single-Agent RAG Query

```bash
curl -X POST http://localhost:8007/api/v1/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How does RAG improve LLM accuracy?",
    "top_k": 10,
    "use_reranker": true,
    "rewrite_strategy": "hyde"
  }'
```

### Multi-Agent RAG Query

```bash
curl -X POST http://localhost:8007/api/v1/query/multi-agent \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Compare vector search vs keyword search for document retrieval. What are the tradeoffs?",
    "top_k": 15
  }'
```

### Hierarchical RAG Query (A-RAG)

```bash
curl -X POST http://localhost:8007/api/v1/query/hierarchical \
  -H "Content-Type: application/json" \
  -d '{
    "question": "Explain the complete RAG pipeline from document ingestion to answer generation, including all optimization techniques.",
    "top_k": 20
  }'
```

### Compare All Three RAG Approaches

```bash
curl -X POST http://localhost:8007/api/v1/query/compare \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the best practices for chunking documents in RAG systems?",
    "top_k": 10
  }'
```

**Response includes side-by-side results with timing:**
```json
{
  "question": "...",
  "single_agent": {"answer": "...", "num_chunks_retrieved": 10},
  "multi_agent": {"answer": "...", "analysis": "...", "critique": "...", "revision_count": 1},
  "hierarchical": {"answer": "...", "plan_complexity": "moderate", "plan_strategies": ["semantic", "keyword"]},
  "timing_ms": {"single_agent_ms": 1234, "multi_agent_ms": 3456, "hierarchical_ms": 5678}
}
```

### Streaming Query (SSE)

```bash
curl -N http://localhost:8007/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RAG?", "top_k": 5}'
```

### Manage Documents and Collections

```bash
# List ingested documents
curl http://localhost:8007/api/v1/documents

# List vector collections
curl http://localhost:8007/api/v1/collections

# Memory statistics
curl http://localhost:8007/api/v1/memory/stats
```

---

## Implementation Deep Dive

### 1. Three RAG Architecture Patterns

**Single-Agent RAG** (`SingleAgentRAG`):
The classic retrieve-and-generate pattern. Retrieves relevant chunks, builds a context window, and generates an answer in a single LLM call. Best for simple factual questions.

```
Query --> [Query Processor] --> [Retriever] --> [Reranker] --> [Context Builder] --> [LLM] --> Answer
```

**Multi-Agent RAG** (`MultiAgentRAG`):
A collaborative pipeline with specialized roles. The Retriever agent fetches relevant chunks. The Analyzer examines them for relevance and gaps. The Generator produces an answer. The Critic evaluates quality and may trigger revisions. Best for complex analytical questions.

```
Query --> [Retriever Agent] --> [Analyzer Agent] --> [Generator Agent] --> [Critic Agent] --+--> Answer
                                                          ^                                |
                                                          +-------- (revise if needed) ----+
```

**Hierarchical RAG / A-RAG** (`HierarchicalRAG`):
An adaptive pattern that plans its retrieval strategy based on query complexity. The Planner assesses complexity (simple/moderate/complex) and selects retrieval strategies, granularity levels (document/section/paragraph), and sub-queries. Multi-granularity retrieval fetches at different levels. A Validator checks the answer against the original question. Best for multi-faceted research questions.

```
Query --> [Planner] --> [Strategy Selection] --> [Multi-Granularity Retrieval] --> [Synthesis] --> [Validator] --> Answer
              |                                         |         |         |
              v                                    Document   Section   Paragraph
         Complexity                                 level      level     level
         Assessment
```

### 2. Document Ingestion Pipeline

The `IngestionPipeline` orchestrates the full document processing flow:

```
Raw Document --> [Loader] --> [Chunker] --> [Embedder] --> [Qdrant Store]
                  |             |              |
               PDF/DOCX      Fixed/         OpenAI/
               TXT/MD/HTML   Semantic/      Cohere
                             Hierarchical/   embeddings
                             Recursive
```

**Chunking Strategies** (configured via `AGENT_RAG_CHUNK_STRATEGY`):

| Strategy | Description | Best For |
|----------|-------------|----------|
| `fixed` | Split by character count with overlap | Uniform, predictable chunks |
| `semantic` | Split at sentence/paragraph boundaries using NLP heuristics | Preserving semantic units |
| `hierarchical` | Three-level splitting: document, section (by headings), paragraph | Multi-granularity retrieval |
| `recursive` | LangChain's recursive character splitter with configurable separators | General-purpose (default) |

### 3. Hybrid Retrieval with Reranking

The retriever combines two search modalities using **Reciprocal Rank Fusion**:

```python
# Hybrid retrieval with RRF
keyword_results = keyword_search(query)       # BM25-style term matching
semantic_results = semantic_search(query)     # Dense embedding similarity

# RRF fusion: score = sum(1 / (k + rank_i)) for each result list
# k=60 (configurable), alpha=0.7 controls semantic vs keyword weight
fused_results = reciprocal_rank_fusion(keyword_results, semantic_results, k=60)
```

**Three Reranker Implementations:**

| Reranker | Method | Latency | Quality |
|----------|--------|---------|---------|
| `CohereReranker` | Cohere Rerank API (cross-encoder model) | ~200ms | Highest |
| `LLMReranker` | LLM scores each chunk 0-10 for relevance | ~2-5s | High |
| `CrossEncoderReranker` | Keyword overlap + IDF weighting (educational) | ~10ms | Moderate |

### 4. Context Engineering

The `ContextBuilder` orchestrates three stages to produce optimal LLM context:

**Stage 1: Ranking** (`ContextRanker`)
Computes a composite score for each passage:
```
composite = normalized_retrieval_score + position_bonus(0.9^i) - diversity_penalty + level_bonus
```
- Position bonus: exponential decay rewards earlier results
- Diversity penalty: down-ranks near-duplicate passages (MD5-based)
- Level bonus: section-level chunks get +0.05

**Stage 2: Compression** (`ContextCompressor`)
- **Deduplication** via character k-shingling (Jaccard similarity, threshold 0.8)
- **Truncation** at sentence boundaries within token budget (max 300 tokens per passage)
- **Extractive compression**: keeps only sentences with query term overlap

**Stage 3: Budget Management** (`ContextBuilder.build()`)
- Token estimation: ~4 characters per token
- Iteratively adds ranked passages until budget is exhausted (default: 8192 tokens)
- Formats each passage with `[Passage N] | Section: ... | Source: ...` headers

### 5. Query Rewriting Strategies

| Strategy | Method | Description |
|----------|--------|-------------|
| `none` | Pass-through | Use query as-is |
| `hyde` | Hypothetical Document Embeddings | Generate a hypothetical answer, embed it, search for similar real documents |
| `step_back` | Step-Back Prompting | Rephrase as a broader question to capture more relevant context |
| `multi_query` | Multi-Query Decomposition | Break complex query into 2-4 sub-queries, retrieve for each, merge results |

### 6. Agent Memory System

The `MemoryManager` provides three types of memory for multi-turn RAG:

- **Conversation Memory** -- Tracks the current conversation thread with sliding window (configurable TTL, default 3600s)
- **Episodic Memory** -- Records notable query-answer pairs for pattern recognition (capacity: 100 entries)
- **Semantic Memory** -- Stores distilled knowledge and facts learned across sessions (capacity: 500 entries)

Memory context is injected into LLM prompts to enable follow-up questions and reference to prior answers within a session.

---

## Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Framework | FastAPI 0.115+ | Async REST API with SSE streaming |
| Vector DB | Qdrant 1.12+ | Dense + sparse vector storage and retrieval |
| Embeddings | OpenAI text-embedding-3-small | 1536-dim document embeddings |
| LLM | Anthropic Claude Sonnet | Answer generation and query analysis |
| Reranking | Cohere Rerank v3 | Cross-encoder reranking |
| Chunking | LangChain Text Splitters | Recursive, semantic, hierarchical splitting |
| Orchestration | LangGraph 0.2+ | Multi-agent RAG workflow |
| Document Parsing | pypdf, python-docx | PDF and DOCX loading |
| Database | PostgreSQL + SQLAlchemy 2.0 | Long-term memory persistence |
| Cache | Redis 5.0+ | Conversation memory and query caching |
| Streaming | SSE-Starlette | Real-time token streaming |
| Validation | Pydantic 2.6+ | Request/response schemas |
| Logging | structlog 24.1+ | Structured JSON logging |
| Runtime | Python 3.11+ | Async/await, type hints |

---

## Project Structure

```
07-agent-rag/
├── pyproject.toml                        # Dependencies and build config
├── src/
│   └── agent_rag/
│       ├── __init__.py
│       ├── main.py                       # Uvicorn entry point
│       ├── config.py                     # Settings (LLM, embeddings, chunking, retrieval, memory)
│       ├── api.py                        # FastAPI endpoints (ingest, query, compare, stream)
│       ├── context_engineering.py        # ContextRanker, ContextCompressor, ContextBuilder
│       ├── memory.py                     # MemoryManager (conversation, episodic, semantic)
│       ├── agents/
│       │   ├── single_agent_rag.py       # Classic retrieve-and-generate RAG
│       │   ├── multi_agent_rag.py        # Retriever + Analyzer + Generator + Critic
│       │   └── hierarchical_rag.py       # A-RAG with adaptive planning and multi-granularity retrieval
│       ├── ingestion/
│       │   ├── pipeline.py               # IngestionPipeline: Load -> Chunk -> Embed -> Store
│       │   ├── loader.py                 # Document loaders (PDF, DOCX, TXT, MD, HTML)
│       │   ├── chunker.py               # Chunking strategies (fixed, semantic, hierarchical, recursive)
│       │   └── embedder.py              # Embedding providers (OpenAI, Cohere)
│       └── retrieval/
│           ├── retriever.py              # HybridRetriever: keyword + semantic + RRF fusion
│           ├── reranker.py               # CohereReranker, LLMReranker, CrossEncoderReranker
│           └── query_processor.py        # Query rewriting (HyDE, step-back, multi-query)
└── tests/
```

---

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENT_RAG_ANTHROPIC_API_KEY` | `""` | Anthropic API key for LLM |
| `AGENT_RAG_OPENAI_API_KEY` | `""` | OpenAI API key for embeddings |
| `AGENT_RAG_COHERE_API_KEY` | `""` | Cohere API key for reranking |
| `AGENT_RAG_QDRANT_URL` | `http://localhost:6333` | Qdrant vector DB URL |
| `AGENT_RAG_CHUNK_STRATEGY` | `recursive` | Chunking: `fixed`, `semantic`, `hierarchical`, `recursive` |
| `AGENT_RAG_CHUNK_SIZE` | `512` | Target chunk size in characters |
| `AGENT_RAG_RETRIEVAL_STRATEGY` | `hybrid` | Retrieval: `keyword`, `semantic`, `hybrid` |
| `AGENT_RAG_HYBRID_ALPHA` | `0.7` | Semantic weight in hybrid (1.0 = pure semantic) |
| `AGENT_RAG_RERANKER_ENABLED` | `true` | Enable reranking stage |
| `AGENT_RAG_RERANKER_MODEL` | `rerank-english-v3.0` | Cohere reranker model |
| `AGENT_RAG_CONTEXT_MAX_TOKENS` | `8192` | Max tokens in context window |
| `AGENT_RAG_CONTEXT_COMPRESSION_ENABLED` | `true` | Enable context compression |
| `AGENT_RAG_PORT` | `8007` | Server port |

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
