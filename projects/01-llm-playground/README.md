# 🧪 LLM Playground

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](../../LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg?logo=docker)](Dockerfile)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)

An interactive API for exploring the building blocks of Large Language Models. Tokenize text with different encoding schemes, compare generation strategies side by side, and inspect transformer architectures -- all through a clean REST API with educational endpoints designed for hands-on learning.

![LLM Playground Screenshot](https://via.placeholder.com/900x400?text=LLM+Playground+API+%E2%80%93+Screenshot+Coming+Soon)

---

## 📚 What You'll Learn

| Concept | Description |
|---------|-------------|
| **LLM Fundamentals** | How large language models work end-to-end: data, tokenization, pre-training, alignment |
| **Tokenization (BPE)** | Byte Pair Encoding, WordPiece, Unigram -- how text becomes numbers |
| **Transformer Architecture** | Encoder-only, decoder-only, encoder-decoder; attention types; positional encoding |
| **Text Generation Strategies** | Greedy decoding, beam search, top-k sampling, nucleus (top-p) sampling, temperature scaling |
| **Sampling Simulation** | Visualize how each strategy reshapes token probability distributions |
| **Model Comparison** | Compare architectures across GPT, LLaMA, Claude, BERT, and T5 families |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                       │
│                        (api.py + main.py)                        │
├──────────────────┬──────────────────┬───────────────────────────┤
│                  │                  │                           │
│  Tokenizer       │  Generation      │  Transformer Explorer     │
│  Service         │  Service         │                           │
│  (tokenizer.py)  │  (generation.py) │  (transformer.py)         │
│                  │                  │                           │
│  - tiktoken      │  - Greedy        │  - Architecture registry  │
│  - HuggingFace   │  - Beam search   │  - Attention patterns     │
│  - BPE trainer   │  - Top-k / Top-p │  - Positional encoding    │
│  - Comparison    │  - Temperature   │  - Model families         │
│                  │  - Simulation    │  - Pre-training overview   │
├──────────────────┴──────────────────┴───────────────────────────┤
│                                                                 │
│  External Providers                                             │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Anthropic   │  │   OpenAI    │  │  HuggingFace Tokenizers │ │
│  │  (Claude)    │  │  (GPT-4o)   │  │  (LLaMA, Mistral, etc.) │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# From the repository root
docker build -f projects/01-llm-playground/Dockerfile -t llm-playground .

# Run with your API keys
docker run -p 8000:8000 \
  -e ANTHROPIC_API_KEY=your-key \
  -e OPENAI_API_KEY=your-key \
  llm-playground
```

### Option 2: Local Development

```bash
# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install the common library and this project
pip install -e libs/common
pip install -e "projects/01-llm-playground[dev]"

# Set environment variables
export ANTHROPIC_API_KEY=your-key
export OPENAI_API_KEY=your-key

# Run the server
python -m llm_playground.main
```

### Option 3: uv (Fast)

```bash
# Install dependencies with uv
uv venv && source .venv/bin/activate
uv pip install -e libs/common -e "projects/01-llm-playground[dev]"

# Set API keys and run
export ANTHROPIC_API_KEY=your-key
python -m llm_playground.main
```

The API will be available at **http://localhost:8000**. Interactive docs at **http://localhost:8000/docs**.

---

## 📡 API Reference

### Health Check

```bash
curl http://localhost:8000/health
```

### Tokenization

**Tokenize text:**

```bash
curl -X POST http://localhost:8000/api/v1/tokenize \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello, world!", "tokenizer_name": "cl100k_base"}'
```

**Decode tokens back to text:**

```bash
curl -X POST http://localhost:8000/api/v1/decode \
  -H "Content-Type: application/json" \
  -d '{"tokens": [9906, 11, 1917, 0], "tokenizer_name": "cl100k_base"}'
```

**Compare tokenizers side by side:**

```bash
curl -X POST http://localhost:8000/api/v1/compare-tokenizers \
  -H "Content-Type: application/json" \
  -d '{
    "text": "The quick brown fox jumps over the lazy dog",
    "tokenizer_names": ["cl100k_base", "o200k_base", "p50k_base"]
  }'
```

**Train a BPE tokenizer from scratch (educational):**

```bash
curl -X POST http://localhost:8000/api/v1/train-bpe \
  -H "Content-Type: application/json" \
  -d '{
    "corpus": ["The cat sat on the mat.", "The dog ran in the park.", "A cat and a dog played."],
    "vocab_size": 300
  }'
```

**List available tokenizers:**

```bash
curl http://localhost:8000/api/v1/tokenizers
```

**Get vocabulary stats:**

```bash
curl http://localhost:8000/api/v1/vocab-stats/cl100k_base
```

### Text Generation

**Generate text:**

```bash
curl -X POST http://localhost:8000/api/v1/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Explain quantum computing in simple terms",
    "strategy": "top_p",
    "temperature": 0.7,
    "top_p": 0.9,
    "max_tokens": 512
  }'
```

**Stream generation (SSE):**

```bash
curl -N -X POST http://localhost:8000/api/v1/generate/stream \
  -H "Content-Type: application/json" \
  -d '{"prompt": "Write a haiku about programming", "strategy": "temperature", "temperature": 0.9}'
```

**Simulate sampling strategies (visualize probability distributions):**

```bash
curl -X POST http://localhost:8000/api/v1/simulate-sampling \
  -H "Content-Type: application/json" \
  -d '{
    "vocab_labels": ["the", "a", "cat", "dog", "sat", "on", "mat", "ran", "big", "red"],
    "logits": [2.0, 1.5, 1.8, 1.2, 0.9, 0.5, 0.3, 0.7, 0.4, 0.6],
    "strategy": "top_k",
    "temperature": 0.8,
    "top_k": 5
  }'
```

**List models and strategies:**

```bash
curl http://localhost:8000/api/v1/models
curl http://localhost:8000/api/v1/strategies
```

### Architecture Exploration

```bash
# List all architectures
curl http://localhost:8000/api/v1/architectures

# Get a specific architecture
curl http://localhost:8000/api/v1/architectures/GPT-4

# Compare architecture types
curl http://localhost:8000/api/v1/architecture-comparison

# Model family timelines
curl http://localhost:8000/api/v1/model-families

# Attention pattern visualizations
curl "http://localhost:8000/api/v1/attention-patterns?seq_length=8"

# Positional encoding demonstrations
curl "http://localhost:8000/api/v1/positional-encoding?max_position=64&d_model=128"

# Pre-training pipeline overview
curl http://localhost:8000/api/v1/pretraining
```

---

## 🔬 Implementation Deep Dive

### 1. Tokenization: How BPE Works

**Byte Pair Encoding (BPE)** is the dominant tokenization algorithm used by GPT, LLaMA, and most modern LLMs. Here is how it works:

```
Step 1: Start with individual characters
  "lower" → ["l", "o", "w", "e", "r"]

Step 2: Count all adjacent pairs in the corpus
  ("l","o"): 5    ("o","w"): 7    ("w","e"): 3    ("e","r"): 9

Step 3: Merge the most frequent pair → "er"
  "lower" → ["l", "o", "w", "er"]

Step 4: Repeat until target vocabulary size is reached
  "lower" → ["low", "er"]
```

The `/api/v1/train-bpe` endpoint lets you watch this process unfold on your own corpus. The `/api/v1/compare-tokenizers` endpoint reveals how different tokenizers split the same text differently:

```
Input: "Tokenization is fascinating!"

cl100k_base (GPT-4):    ["Token", "ization", " is", " fascinating", "!"]  → 5 tokens
o200k_base  (GPT-4o):   ["Tokenization", " is", " fascinating", "!"]     → 4 tokens
p50k_base   (Codex):    ["Token", "iz", "ation", " is", " fasc", "inating", "!"] → 7 tokens
```

**Key insight:** Newer tokenizers have larger vocabularies and produce fewer tokens, which means lower cost and more room in the context window.

### 2. Text Generation Strategies

Every time an LLM generates the next token, it produces a probability distribution over the entire vocabulary. The **generation strategy** determines how we pick from that distribution.

```
Vocabulary:  [the]  [a]  [cat] [dog] [sat] [on]  [mat] [ran] [big] [red]
Raw logits:   2.0   1.5   1.8  1.2   0.9  0.5    0.3   0.7   0.4   0.6
Softmax:     0.24  0.15  0.20 0.11  0.08  0.05   0.04  0.07  0.05  0.06

GREEDY:      [the] ████████████████████████ 100%  ← Always picks highest prob
             Everything else: 0%

TOP-K (k=3): [the] ████████████████  41%  ← Re-normalize top 3
             [cat] █████████████    33%
             [a]   ██████████       26%

TOP-P (p=0.9): Keeps smallest set whose cumulative prob >= 0.9
             [the] ████████████████  ~27%
             [cat] █████████████    ~22%
             [a]   ██████████       ~17%
             [dog] ████████         ~12%
             [sat] ██████           ~9%
             [ran] █████            ~8%
             [red] ████             ~5%

TEMPERATURE (T=0.5): Sharpens distribution → more deterministic
TEMPERATURE (T=2.0): Flattens distribution → more random
```

The `/api/v1/simulate-sampling` endpoint returns the `original_probs` and `filtered_probs` arrays so you can visualize exactly how each strategy reshapes the distribution.

### 3. Transformer Architecture

The project provides structured data about three major architecture families:

**Attention Mechanism Types:**

```
CAUSAL (Decoder-only: GPT, LLaMA, Claude)
┌───┬───┬───┬───┐
│ 1 │   │   │   │    Token 1 sees only itself
│ 1 │ 1 │   │   │    Token 2 sees tokens 1-2
│ 1 │ 1 │ 1 │   │    Token 3 sees tokens 1-3
│ 1 │ 1 │ 1 │ 1 │    Token 4 sees tokens 1-4
└───┴───┴───┴───┘

BIDIRECTIONAL (Encoder-only: BERT)
┌───┬───┬───┬───┐
│ 1 │ 1 │ 1 │ 1 │    Every token sees
│ 1 │ 1 │ 1 │ 1 │    every other token
│ 1 │ 1 │ 1 │ 1 │    (full context)
│ 1 │ 1 │ 1 │ 1 │
└───┴───┴───┴───┘

SLIDING WINDOW (Mistral-style)
┌───┬───┬───┬───┐
│ 1 │   │   │   │    Each token attends
│ 1 │ 1 │   │   │    to a fixed-size local
│   │ 1 │ 1 │   │    window for efficiency
│   │   │ 1 │ 1 │
└───┴───┴───┴───┘
```

**Positional Encoding Methods:**

| Method | Used By | Key Property |
|--------|---------|-------------|
| Learned absolute | GPT-2, GPT-3 | Fixed maximum length |
| Sinusoidal | Original Transformer | Fixed, no parameters |
| RoPE (Rotary) | LLaMA, Mistral, Gemma | Relative position, length extrapolation |
| ALiBi | BLOOM, MPT | Linear bias on attention, excellent extrapolation |

The `/api/v1/positional-encoding` endpoint returns numerical values for each method so you can plot and compare them.

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | FastAPI | Async REST API with auto-generated OpenAPI docs |
| **LLM Providers** | Anthropic, OpenAI | Cloud-based text generation |
| **Tokenization** | tiktoken, HuggingFace tokenizers | BPE encoding/decoding, vocabulary analysis |
| **ML Runtime** | PyTorch, Transformers | HuggingFace model loading and tokenizer support |
| **Numerical** | NumPy | Sampling simulation, attention matrices |
| **Streaming** | SSE-Starlette | Server-Sent Events for token streaming |
| **Config** | Pydantic Settings | Type-safe configuration from environment variables |
| **Logging** | structlog | Structured JSON logging |
| **Containerization** | Docker (multi-stage) | Reproducible builds with non-root runtime |
| **Orchestration** | Kubernetes | Production deployment manifests |

---

## 📁 Project Structure

```
01-llm-playground/
├── src/llm_playground/
│   ├── __init__.py
│   ├── main.py              # Uvicorn entry point
│   ├── api.py               # FastAPI app with all endpoints
│   ├── config.py            # Settings, model registry
│   ├── tokenizer.py         # TokenizerService: encode, decode, compare, train BPE
│   ├── generation.py        # GenerationService: greedy, beam, top-k, top-p, temp
│   └── transformer.py       # TransformerExplorer: architectures, attention, positional encoding
├── tests/
│   ├── conftest.py           # Shared fixtures
│   ├── test_api.py           # API integration tests
│   └── test_tokenizer.py     # Tokenizer unit tests
├── k8s/
│   └── deployment.yaml       # Kubernetes deployment manifest
├── Dockerfile                # Multi-stage Docker build
├── pyproject.toml            # Dependencies and build config
└── README.md                 # This file
```

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

Please ensure all tests pass and follow the existing code style (Ruff for linting).

---

## 📄 License

This project is part of the AI Engineer Portfolio and is licensed under the [MIT License](../../LICENSE).
