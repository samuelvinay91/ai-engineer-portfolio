# 🤖 Customer Support Chatbot

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg?logo=docker)](Dockerfile)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)

A production-grade customer support chatbot that combines advanced prompt engineering, PEFT/LoRA fine-tuning, LLM-based intent classification, and Redis-backed conversation management. Built to demonstrate how real-world AI support systems are designed -- from the system prompt all the way to the training pipeline.

![Chatbot Screenshot](https://via.placeholder.com/900x400?text=Customer+Support+Chatbot+%E2%80%93+Screenshot+Coming+Soon)

---

## 📚 What You'll Learn

| Concept | Description |
|---------|-------------|
| **Prompt Engineering** | System prompts, few-shot examples, chain-of-thought reasoning, template versioning |
| **Fine-tuning with PEFT** | Parameter-Efficient Fine-Tuning -- train an LLM on your data without full model weights |
| **LoRA Adapters** | Low-Rank Adaptation -- how and why it works, hands-on training pipeline |
| **Conversation Management** | Finite state machines, sliding-window context, LLM-based summarization |
| **Intent Classification** | LLM-powered multi-label classification with structured JSON output |
| **Production Patterns** | Redis session storage, streaming responses, health checks, error handling |

---

## 🏗️ Architecture

```
                         ┌──────────────┐
                         │   Client     │
                         │  (REST/SSE)  │
                         └──────┬───────┘
                                │
                    ┌───────────▼──────────────┐
                    │     FastAPI Application   │
                    │         (api.py)          │
                    └───────────┬──────────────┘
                                │
           ┌────────────────────┼────────────────────┐
           │                    │                     │
    ┌──────▼──────┐    ┌───────▼────────┐    ┌──────▼──────────┐
    │   Intent    │    │    Prompt      │    │  Conversation   │
    │ Classifier  │    │   Registry     │    │    Manager      │
    │(classifier  │    │  (prompts.py)  │    │(conversation.py)│
    │    .py)     │    │                │    │                 │
    └──────┬──────┘    └───────┬────────┘    └──────┬──────────┘
           │                   │                     │
           │  ┌───────────┐    │  ┌──────────────┐   │  ┌─────────┐
           │  │ billing   │    │  │ YAML         │   │  │  Redis  │
           ├──│ technical │    ├──│ Templates    │   └──│ Sessions│
           │  │ account   │    │  │ + Versions   │      └─────────┘
           │  │ general   │    │  └──────────────┘
           │  │ escalation│    │
           │  └───────────┘    │  ┌──────────────┐
           │                   ├──│ Few-Shot Mgr │
           │                   │  └──────────────┘
           │                   │
           │                   │  ┌──────────────┐
           │                   └──│ CoT Template │
           │                      └──────────────┘
           │
    ┌──────▼──────────────────────────────────────────┐
    │                  LLM Provider                    │
    │            (Anthropic Claude API)                 │
    └──────────────────────────────────────────────────┘
           │
    ┌──────▼──────────────────────────────────────────┐
    │            Fine-tuning Pipeline                   │
    │            (finetuning.py)                        │
    │                                                  │
    │  DatasetPreparator → LoRA Config → SFTTrainer    │
    │       ↓                                          │
    │  Base Model → QLoRA (4-bit) → Train → Export     │
    └──────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### Option 1: Docker (Recommended)

```bash
# From the repository root
docker build -f Dockerfile \
  -t customer-support-chatbot .

# Run with your API key
docker run -p 8000:8000 \
  -e CHATBOT_ANTHROPIC_API_KEY=your-key \
  customer-support-chatbot
```

### Option 2: Local Development

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .
pip install -e "projects/02-customer-support-chatbot[dev]"

# Set environment variables
export CHATBOT_ANTHROPIC_API_KEY=your-key

# (Optional) Start Redis for conversation persistence
docker run -d -p 6379:6379 redis:7-alpine

# Run the server
cd projects/02-customer-support-chatbot
python -m customer_support.main
```

The API will be available at **http://localhost:8000**. Interactive docs at **http://localhost:8000/docs**.

---

## 📡 API Reference

### Health Check

```bash
curl http://localhost:8000/health
```

### Chat (Synchronous)

```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "I was charged twice for my subscription this month",
    "customer_name": "Alice",
    "use_chain_of_thought": true
  }'
```

Response includes the reply, detected intent, confidence, conversation state, and sentiment trend.

### Chat (Streaming via SSE)

```bash
curl -N -X POST http://localhost:8000/api/v1/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"message": "How do I reset my password?", "session_id": "abc123"}'
```

### Intent Classification

```bash
curl -X POST http://localhost:8000/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "message": "Your app keeps crashing whenever I try to upload photos",
    "context": [
      {"role": "user", "content": "I need help with the mobile app"},
      {"role": "assistant", "content": "Of course! What issue are you experiencing?"}
    ]
  }'
```

### Conversation History

```bash
curl http://localhost:8000/api/v1/conversations/abc123
```

### Prompt Templates

```bash
# List all registered templates
curl http://localhost:8000/api/v1/prompts

# Test-render a prompt template
curl -X POST http://localhost:8000/api/v1/prompts/test \
  -H "Content-Type: application/json" \
  -d '{
    "template_name": "billing_support",
    "knowledge_base": "Refund policy: full refunds within 30 days."
  }'
```

---

## 🔬 Implementation Deep Dive

### 1. Prompt Engineering

The chatbot uses a **composable, version-controlled prompt system** built on four layers:

**Layer 1 -- SystemPromptBuilder (Fluent API):**

```python
prompt = (
    SystemPromptBuilder()
    .with_role("Acme Corp Customer Support Agent")
    .with_knowledge_base(kb_text)
    .with_tone("professional", "empathetic", "concise")
    .with_escalation_rules(rules)
    .with_response_format(fmt)
    .with_guardrails(safety_policy)
    .with_chain_of_thought(visible=False)
    .with_few_shot_examples(manager, "billing")
    .build()
)
```

Each section is wrapped in XML tags (`<role>`, `<knowledge_base>`, `<tone_guidelines>`, etc.) for clear prompt structure. Sections are assembled in a deterministic priority order.

**Layer 2 -- YAML Templates with Versioning:**

```yaml
# data/templates/billing_support.yaml
name: billing_support
description: Handles billing, refunds, and subscription queries
version: "1.2.0"
chain_of_thought_enabled: true
system_prompt: |
  You are a billing support specialist for Acme Corp...
few_shot_examples:
  - user: "I was charged twice this month."
    assistant: "I'm sorry about the duplicate charge. Let me look into..."
    tags: [billing, refund, duplicate-charge]
```

Every template is snapshotted with a SHA-256 hash, so prompt changes are trackable and reversible.

**Layer 3 -- Few-Shot Examples:**

The `FewShotManager` organizes examples by category and renders them as XML blocks inside the system prompt:

```xml
<examples>
  <example>
    <user>I was charged twice this month.</user>
    <assistant>I'm sorry about the duplicate charge...</assistant>
  </example>
</examples>
```

**Layer 4 -- Chain-of-Thought:**

For complex issues, the `ChainOfThoughtTemplate` wraps the user message in a structured reasoning framework:

```
Step 1 -- Problem identification: What is the core issue?
Step 2 -- Context gathering: What additional info is relevant?
Step 3 -- Solution exploration: List 2-3 possible approaches
Step 4 -- Escalation check: Does this require a human?
Step 5 -- Response composition: Draft the final response
```

The model reasons internally and outputs only the final customer-facing message.

### 2. LoRA Fine-tuning

**What is LoRA?** Low-Rank Adaptation freezes the pre-trained model weights and injects small trainable matrices into specific layers:

```
Original weight matrix W (d x d):        4096 x 4096 = 16.7M params

LoRA decomposition:
  W' = W + (A x B)
  where A is (d x r) and B is (r x d)

With rank r=16:
  A: 4096 x 16 =   65K params  ┐
  B: 16 x 4096 =   65K params  ├── 0.78% of original!
                    130K params ┘
```

**The Training Pipeline:**

```
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌──────────┐
│ Load JSONL  │───▶│ Apply Chat   │───▶│  Tokenize   │───▶│  Train   │
│ Training    │    │ Template     │    │  (truncate  │    │  (SFT +  │
│ Data        │    │ (format for  │    │   to 2048)  │    │  QLoRA)  │
│             │    │  the model)  │    │             │    │          │
└─────────────┘    └──────────────┘    └─────────────┘    └────┬─────┘
                                                               │
                   ┌──────────────┐    ┌─────────────┐    ┌────▼─────┐
                   │ Export       │◀───│  Evaluate   │◀───│ Validate │
                   │ Adapter      │    │  (loss,     │    │ (held-   │
                   │ (~50MB)      │    │  perplexity)│    │  out set)│
                   └──────────────┘    └─────────────┘    └──────────┘
```

Key configuration from `LoRAConfig`:

| Parameter | Default | What it controls |
|-----------|---------|-----------------|
| `rank` | 16 | Size of low-rank matrices (higher = more capacity) |
| `alpha` | 32 | Scaling factor (rule of thumb: 2x rank) |
| `dropout` | 0.05 | Regularization to prevent overfitting |
| `target_modules` | q/k/v/o/gate/up/down_proj | Which layers get adapters |
| `quantization_bits` | 4 | QLoRA: 4-bit NormalFloat quantization |
| `learning_rate` | 2e-4 | Peak LR with cosine schedule |

Run the pipeline:

```bash
python -m customer_support.finetuning \
  --base-model mistralai/Mistral-7B-Instruct-v0.3 \
  --dataset data/training/support_conversations.jsonl \
  --output-dir ./training_output \
  --epochs 3 --lora-rank 16
```

### 3. Conversation Management

The conversation module implements a **finite state machine** with Redis-backed persistence:

```
┌──────────┐    1st user msg    ┌───────────────┐
│ GREETING │───────────────────▶│ UNDERSTANDING │
└──────────┘                    └───────┬───────┘
                                        │ 2+ turns
                                ┌───────▼───────┐
                                │   RESOLVING   │◄─────┐
                                └───────┬───────┘      │
                                        │              │ new issue
                                ┌───────▼───────┐      │
                  satisfaction  │    CLOSING    │──────┘
                   signal       └───────┬───────┘
                                        │
                                ┌───────▼───────┐
                                │    CLOSED     │
                                └───────────────┘

  Any state ───(anger/legal/security)───▶ ESCALATED ───▶ CLOSED
```

**Sliding-Window Context:** When conversations exceed `max_history` messages (default: 20), older messages are summarized and prepended to maintain context without exceeding token limits:

```
[Summary of turns 1-15] + [Full messages 16-20] → LLM
```

**Sentiment Tracking:** Each user message records a sentiment level (very_negative to very_positive). The system computes a trend (improving / stable / deteriorating) to trigger proactive escalation.

### 4. Intent Classification

The classifier uses an **LLM-based approach** with structured JSON output rather than a traditional ML model:

```
Input:  "Your app keeps crashing whenever I try to upload photos"

┌──────────────────────────────────────────────────────┐
│  LLM Classification (temperature=0.0 for determinism)│
│                                                      │
│  System Prompt: taxonomy definition + JSON schema    │
│  User Message:  the customer's text                  │
│  Context:       last 4 messages (optional)           │
└──────────────────────────────────┬───────────────────┘
                                   │
Output JSON:                       ▼
{
  "primary_intent": "technical",
  "primary_confidence": 0.92,
  "secondary_intents": [
    {"intent": "escalation", "confidence": 0.35}
  ],
  "reasoning": "User reports app crash during photo upload - technical issue"
}
```

**Why LLM-based instead of traditional ML?**

| Factor | LLM Classification | Traditional ML (e.g. BERT) |
|--------|-------------------|---------------------------|
| Training data needed | Zero (zero-shot) | Hundreds-thousands of labeled examples |
| New intent support | Update the prompt | Retrain the model |
| Explainability | Built-in reasoning field | Requires separate explanation model |
| Latency | ~200-500ms | ~10-50ms |
| Cost | API call per message | One-time training cost |

The classification result drives **routing** to the appropriate prompt template (billing_support, technical_support, or general_support).

---

## 🛠️ Tech Stack

| Layer | Technology | Purpose |
|-------|-----------|---------|
| **Framework** | FastAPI | Async REST API with OpenAPI docs |
| **LLM Provider** | Anthropic Claude | Chat completions and classification |
| **Prompt Management** | YAML + Jinja2 | Version-controlled prompt templates |
| **Session Storage** | Redis | Conversation persistence with TTL |
| **Fine-tuning** | PEFT, LoRA, bitsandbytes | Parameter-efficient model adaptation |
| **Training** | HuggingFace Transformers, TRL | SFTTrainer with QLoRA support |
| **Experiment Tracking** | Weights & Biases | Training metrics and model comparison |
| **Streaming** | SSE-Starlette | Real-time token streaming |
| **Database** | PostgreSQL + SQLAlchemy | (Optional) structured data storage |
| **Config** | Pydantic Settings | Type-safe environment configuration |
| **Logging** | structlog | Structured JSON logging |
| **Containerization** | Docker (multi-stage) | Secure, slim production image |

---

## 📁 Project Structure

```
02-customer-support-chatbot/
├── src/customer_support/
│   ├── __init__.py
│   ├── main.py              # Uvicorn entry point
│   ├── api.py               # FastAPI app: chat, classify, prompts, conversations
│   ├── config.py            # Settings (env vars, model config, Redis URL)
│   ├── prompts.py           # SystemPromptBuilder, PromptRegistry, FewShotManager, CoT
│   ├── classifier.py        # IntentClassifier with LLM-based multi-label classification
│   ├── conversation.py      # ConversationManager, state machine, sentiment tracking
│   └── finetuning.py        # LoRA training pipeline (DatasetPreparator, FineTuningPipeline)
├── data/templates/
│   ├── billing_support.yaml
│   ├── technical_support.yaml
│   └── general_support.yaml
├── tests/
│   ├── conftest.py
│   ├── test_api.py
│   └── test_prompts.py
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
5. For fine-tuning work, install training extras: `pip install -e ".[training]"`
6. Submit a pull request

---

## 📄 License

This project is part of the AI Engineer Portfolio and is licensed under the [MIT License](LICENSE).
