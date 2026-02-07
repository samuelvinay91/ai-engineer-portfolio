#!/usr/bin/env python3
"""Generate standalone GitHub repositories from the monorepo.

Each project becomes a fully self-contained repo with:
- Inlined common library
- Standalone pyproject.toml, Dockerfile, docker-compose.yml
- CI/CD pipeline, .env.example, .gitignore, LICENSE, K8s manifests
- Updated README with standalone setup instructions
"""

import os
import shutil
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STANDALONE_DIR = ROOT / "standalone-repos"

# ── Project definitions ──────────────────────────────────────────────────────

PROJECTS = [
    {
        "num": "01",
        "dir": "01-llm-playground",
        "name": "llm-playground",
        "pkg": "llm_playground",
        "display": "LLM Playground",
        "desc": "Interactive LLM Playground - Explore tokenization, text generation strategies, and transformer architectures",
        "port": 8001,
        "python": "3.12",
        "infra": [],  # no postgres/redis/qdrant
        "volumes": ["model-cache:/app/models"],
        "extra_copy": [],
        "env_vars": {
            "OPENAI_API_KEY": "sk-your-openai-key",
            "ANTHROPIC_API_KEY": "sk-ant-your-anthropic-key",
            "HUGGINGFACE_TOKEN": "hf_your-token",
        },
        "ci_secrets": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        "deps": [
            '"fastapi>=0.115.0"',
            '"uvicorn[standard]>=0.32.0"',
            '"pydantic>=2.6.0"',
            '"pydantic-settings>=2.2.0"',
            '"anthropic>=0.40.0"',
            '"openai>=1.50.0"',
            '"tiktoken>=0.8.0"',
            '"tokenizers>=0.20.0"',
            '"transformers>=4.46.0"',
            '"torch>=2.4.0"',
            '"numpy>=1.26.0"',
            '"structlog>=24.1.0"',
            '"httpx>=0.27.0"',
            '"sse-starlette>=2.0.0"',
        ],
        "optional_deps": {},
        "cmd": 'CMD ["python", "-m", "llm_playground.main"]',
        "topics": "llm tokenization bpe transformer generation-strategies fastapi docker cloud-native ai-engineer",
        "badges_extra": "",
        "emoji": "🧪",
    },
    {
        "num": "02",
        "dir": "02-customer-support-chatbot",
        "name": "customer-support-chatbot",
        "pkg": "customer_support",
        "display": "Customer Support Chatbot",
        "desc": "Production customer support chatbot with PEFT/LoRA fine-tuning and advanced prompt engineering",
        "port": 8002,
        "python": "3.12",
        "infra": ["postgres", "redis"],
        "volumes": [],
        "extra_copy": ["data"],
        "env_vars": {
            "OPENAI_API_KEY": "sk-your-openai-key",
            "ANTHROPIC_API_KEY": "sk-ant-your-anthropic-key",
            "POSTGRES_PASSWORD": "localdev",
            "DATABASE_URL": "postgresql://aiportfolio:localdev@postgres:5432/aiportfolio",
            "REDIS_URL": "redis://redis:6379/0",
        },
        "ci_secrets": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        "deps": [
            '"fastapi>=0.115.0"',
            '"uvicorn[standard]>=0.32.0"',
            '"pydantic>=2.6.0"',
            '"pydantic-settings>=2.2.0"',
            '"anthropic>=0.40.0"',
            '"openai>=1.50.0"',
            '"langchain-core>=0.3.0"',
            '"langchain-anthropic>=0.3.0"',
            '"langchain-openai>=0.2.0"',
            '"redis>=5.0.0"',
            '"sqlalchemy>=2.0.0"',
            '"asyncpg>=0.30.0"',
            '"alembic>=1.13.0"',
            '"jinja2>=3.1.0"',
            '"structlog>=24.1.0"',
            '"httpx>=0.27.0"',
            '"sse-starlette>=2.0.0"',
            '"pyyaml>=6.0.0"',
        ],
        "optional_deps": {
            "training": [
                '"transformers>=4.46.0"',
                '"peft>=0.13.0"',
                '"datasets>=3.0.0"',
                '"torch>=2.4.0"',
                '"bitsandbytes>=0.44.0"',
                '"accelerate>=1.0.0"',
                '"wandb>=0.18.0"',
            ],
        },
        "cmd": 'CMD ["uvicorn", "customer_support.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]',
        "topics": "chatbot customer-support lora peft fine-tuning prompt-engineering langchain fastapi docker ai-engineer",
        "badges_extra": "",
        "emoji": "💬",
    },
    {
        "num": "03",
        "dir": "03-ask-the-web-agent",
        "name": "ask-the-web-agent",
        "pkg": "ask_the_web",
        "display": "Ask-the-Web Agent",
        "desc": "Perplexity-like Ask-the-Web agent with search, synthesis, and citation - demonstrating agentic patterns",
        "port": 8003,
        "python": "3.11",
        "infra": ["redis"],
        "volumes": [],
        "extra_copy": [],
        "env_vars": {
            "OPENAI_API_KEY": "sk-your-openai-key",
            "ANTHROPIC_API_KEY": "sk-ant-your-anthropic-key",
            "TAVILY_API_KEY": "tvly-your-tavily-key",
            "REDIS_URL": "redis://redis:6379/0",
        },
        "ci_secrets": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        "deps": [
            '"fastapi>=0.115.0"',
            '"uvicorn[standard]>=0.32.0"',
            '"pydantic>=2.6.0"',
            '"pydantic-settings>=2.2.0"',
            '"anthropic>=0.40.0"',
            '"openai>=1.50.0"',
            '"langchain-core>=0.3.0"',
            '"langchain-anthropic>=0.3.0"',
            '"langgraph>=0.2.0"',
            '"tavily-python>=0.5.0"',
            '"httpx>=0.27.0"',
            '"beautifulsoup4>=4.12.0"',
            '"markdownify>=0.13.0"',
            '"redis>=5.0.0"',
            '"structlog>=24.1.0"',
            '"sse-starlette>=2.0.0"',
        ],
        "optional_deps": {},
        "cmd": 'CMD ["python", "-m", "ask_the_web.main"]',
        "topics": "langgraph agent perplexity web-search rag langchain fastapi docker cloud-native ai-engineer",
        "badges_extra": "",
        "emoji": "🔍",
    },
    {
        "num": "04",
        "dir": "04-deep-research",
        "name": "deep-research",
        "pkg": "deep_research",
        "display": "Deep Research",
        "desc": "Deep Research capability with reasoning models, CoT prompting, and inference-time scaling",
        "port": 8004,
        "python": "3.11",
        "infra": ["postgres", "redis"],
        "volumes": [],
        "extra_copy": [],
        "env_vars": {
            "OPENAI_API_KEY": "sk-your-openai-key",
            "ANTHROPIC_API_KEY": "sk-ant-your-anthropic-key",
            "TAVILY_API_KEY": "tvly-your-tavily-key",
            "POSTGRES_PASSWORD": "localdev",
            "DATABASE_URL": "postgresql://aiportfolio:localdev@postgres:5432/aiportfolio",
            "REDIS_URL": "redis://redis:6379/0",
        },
        "ci_secrets": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        "deps": [
            '"fastapi>=0.115.0"',
            '"uvicorn[standard]>=0.32.0"',
            '"pydantic>=2.6.0"',
            '"pydantic-settings>=2.2.0"',
            '"anthropic>=0.40.0"',
            '"openai>=1.50.0"',
            '"langchain-core>=0.3.0"',
            '"langchain-anthropic>=0.3.0"',
            '"langchain-openai>=0.2.0"',
            '"langgraph>=0.2.0"',
            '"tavily-python>=0.5.0"',
            '"httpx>=0.27.0"',
            '"redis>=5.0.0"',
            '"sqlalchemy>=2.0.0"',
            '"asyncpg>=0.30.0"',
            '"structlog>=24.1.0"',
            '"sse-starlette>=2.0.0"',
        ],
        "optional_deps": {},
        "cmd": 'CMD ["python", "-m", "deep_research.main"]',
        "topics": "deep-research chain-of-thought tree-of-thought reasoning langgraph inference-scaling fastapi docker ai-engineer",
        "badges_extra": "",
        "emoji": "🔬",
    },
    {
        "num": "05",
        "dir": "05-image-generation",
        "name": "image-generation",
        "pkg": "image_generation",
        "display": "Image Generation Service",
        "desc": "Cloud-native Image Generation Service with diffusion models, VAE, and multi-provider support",
        "port": 8005,
        "python": "3.11",
        "infra": [],
        "volumes": ["generated-images:/app/outputs"],
        "extra_copy": [],
        "env_vars": {
            "OPENAI_API_KEY": "sk-your-openai-key",
            "REPLICATE_API_TOKEN": "your-replicate-token",
            "STABILITY_API_KEY": "your-stability-key",
        },
        "ci_secrets": ["OPENAI_API_KEY"],
        "deps": [
            '"fastapi>=0.115.0"',
            '"uvicorn[standard]>=0.32.0"',
            '"pydantic>=2.6.0"',
            '"pydantic-settings>=2.2.0"',
            '"openai>=1.50.0"',
            '"httpx>=0.27.0"',
            '"Pillow>=10.4.0"',
            '"replicate>=0.34.0"',
            '"structlog>=24.1.0"',
            '"redis>=5.0.0"',
            '"boto3>=1.35.0"',
            '"aiofiles>=24.1.0"',
        ],
        "optional_deps": {
            "local": [
                '"torch>=2.4.0"',
                '"diffusers>=0.31.0"',
                '"transformers>=4.46.0"',
                '"accelerate>=1.0.0"',
                '"safetensors>=0.4.0"',
            ],
        },
        "cmd": 'CMD ["python", "-m", "image_generation.main"]',
        "topics": "image-generation diffusion dall-e stable-diffusion flux text-to-image fastapi docker ai-engineer",
        "badges_extra": "",
        "emoji": "🎨",
    },
    {
        "num": "06",
        "dir": "06-capstone-multiagent",
        "name": "capstone-multiagent",
        "pkg": "capstone",
        "display": "Capstone Multi-Agent Platform",
        "desc": "Capstone: Multi-Agent AI Platform orchestrating specialized agents for complex task solving",
        "port": 8006,
        "python": "3.11",
        "infra": ["postgres", "redis"],
        "volumes": [],
        "extra_copy": [],
        "env_vars": {
            "OPENAI_API_KEY": "sk-your-openai-key",
            "ANTHROPIC_API_KEY": "sk-ant-your-anthropic-key",
            "POSTGRES_PASSWORD": "localdev",
            "DATABASE_URL": "postgresql://aiportfolio:localdev@postgres:5432/aiportfolio",
            "REDIS_URL": "redis://redis:6379/0",
        },
        "ci_secrets": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        "deps": [
            '"fastapi>=0.115.0"',
            '"uvicorn[standard]>=0.32.0"',
            '"pydantic>=2.6.0"',
            '"pydantic-settings>=2.2.0"',
            '"anthropic>=0.40.0"',
            '"openai>=1.50.0"',
            '"langchain-core>=0.3.0"',
            '"langchain-anthropic>=0.3.0"',
            '"langchain-openai>=0.2.0"',
            '"langgraph>=0.2.0"',
            '"redis>=5.0.0"',
            '"sqlalchemy>=2.0.0"',
            '"asyncpg>=0.30.0"',
            '"httpx>=0.27.0"',
            '"structlog>=24.1.0"',
            '"sse-starlette>=2.0.0"',
        ],
        "optional_deps": {},
        "cmd": 'CMD ["python", "-m", "capstone.main"]',
        "topics": "multi-agent supervisor-pattern orchestration langgraph langchain fastapi docker cloud-native ai-engineer",
        "badges_extra": "",
        "emoji": "🤖",
    },
    {
        "num": "07",
        "dir": "07-agent-rag",
        "name": "agent-rag",
        "pkg": "agent_rag",
        "display": "Agent RAG System",
        "desc": "Advanced Agent & RAG System with hierarchical retrieval, query decomposition, and multi-agent patterns",
        "port": 8007,
        "python": "3.11",
        "infra": ["postgres", "qdrant"],
        "volumes": [],
        "extra_copy": [],
        "env_vars": {
            "OPENAI_API_KEY": "sk-your-openai-key",
            "ANTHROPIC_API_KEY": "sk-ant-your-anthropic-key",
            "COHERE_API_KEY": "your-cohere-key",
            "POSTGRES_PASSWORD": "localdev",
            "DATABASE_URL": "postgresql://aiportfolio:localdev@postgres:5432/aiportfolio",
            "QDRANT_URL": "http://qdrant:6333",
        },
        "ci_secrets": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        "deps": [
            '"fastapi>=0.115.0"',
            '"uvicorn[standard]>=0.32.0"',
            '"pydantic>=2.6.0"',
            '"pydantic-settings>=2.2.0"',
            '"anthropic>=0.40.0"',
            '"openai>=1.50.0"',
            '"langchain-core>=0.3.0"',
            '"langchain-anthropic>=0.3.0"',
            '"langchain-openai>=0.2.0"',
            '"langchain-text-splitters>=0.3.0"',
            '"langgraph>=0.2.0"',
            '"qdrant-client>=1.12.0"',
            '"cohere>=5.11.0"',
            '"sqlalchemy>=2.0.0"',
            '"asyncpg>=0.30.0"',
            '"httpx>=0.27.0"',
            '"redis>=5.0.0"',
            '"structlog>=24.1.0"',
            '"sse-starlette>=2.0.0"',
            '"pypdf>=5.0.0"',
            '"python-docx>=1.1.0"',
        ],
        "optional_deps": {},
        "cmd": 'CMD ["python", "-m", "agent_rag.main"]',
        "topics": "rag retrieval-augmented-generation vector-search qdrant langgraph langchain fastapi docker ai-engineer",
        "badges_extra": "",
        "emoji": "📚",
    },
    {
        "num": "08",
        "dir": "08-mcp-a2a",
        "name": "mcp-a2a",
        "pkg": "mcp_a2a",
        "display": "MCP & A2A Integration",
        "desc": "MCP Server/Client implementation and A2A protocol with AgentCards for agent interoperability",
        "port": 8008,
        "python": "3.11",
        "infra": [],
        "volumes": [],
        "extra_copy": [],
        "env_vars": {
            "OPENAI_API_KEY": "sk-your-openai-key",
            "ANTHROPIC_API_KEY": "sk-ant-your-anthropic-key",
        },
        "ci_secrets": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        "deps": [
            '"fastapi>=0.115.0"',
            '"uvicorn[standard]>=0.32.0"',
            '"pydantic>=2.6.0"',
            '"pydantic-settings>=2.2.0"',
            '"anthropic>=0.40.0"',
            '"openai>=1.50.0"',
            '"httpx>=0.27.0"',
            '"mcp>=1.0.0"',
            '"structlog>=24.1.0"',
            '"sse-starlette>=2.0.0"',
            '"redis>=5.0.0"',
        ],
        "optional_deps": {},
        "cmd": 'CMD ["python", "-m", "mcp_a2a.main"]',
        "topics": "mcp model-context-protocol a2a agent-to-agent agentcard interoperability fastapi docker ai-engineer",
        "badges_extra": "",
        "emoji": "🔗",
    },
    {
        "num": "09",
        "dir": "09-ucp-merchant-server",
        "name": "ucp-merchant-server",
        "pkg": "ucp_merchant",
        "display": "UCP Merchant Server",
        "desc": "UCP-compliant merchant platform with checkout state machine, AP2 payment verification, and MCP tool bindings",
        "port": 8011,
        "python": "3.11",
        "infra": [],
        "volumes": [],
        "extra_copy": [],
        "env_vars": {
            "MERCHANT_NAME": "TechVault Electronics",
            "ENVIRONMENT": "development",
        },
        "ci_secrets": [],
        "deps": [
            '"fastapi>=0.115.0"',
            '"uvicorn[standard]>=0.32.0"',
            '"pydantic>=2.6.0"',
            '"pydantic-settings>=2.2.0"',
            '"cryptography>=43.0.0"',
            '"httpx>=0.27.0"',
            '"structlog>=24.1.0"',
            '"sse-starlette>=2.0.0"',
        ],
        "optional_deps": {},
        "cmd": 'CMD ["python", "-m", "ucp_merchant.main"]',
        "topics": "ucp universal-commerce-protocol agentic-commerce ap2 checkout-state-machine ecdsa mcp fastapi docker ai-engineer",
        "badges_extra": "",
        "emoji": "🛒",
    },
    {
        "num": "10",
        "dir": "10-ucp-shopping-agent",
        "name": "ucp-shopping-agent",
        "pkg": "ucp_shopping",
        "display": "UCP Shopping Agent",
        "desc": "AI shopping agent that discovers UCP merchants, compares products, optimizes multi-vendor orders with LangGraph",
        "port": 8020,
        "python": "3.11",
        "infra": [],
        "volumes": [],
        "extra_copy": [],
        "env_vars": {
            "ANTHROPIC_API_KEY": "sk-ant-your-anthropic-key",
            "ENVIRONMENT": "development",
        },
        "ci_secrets": ["ANTHROPIC_API_KEY"],
        "deps": [
            '"fastapi>=0.115.0"',
            '"uvicorn[standard]>=0.32.0"',
            '"pydantic>=2.6.0"',
            '"pydantic-settings>=2.2.0"',
            '"langgraph>=0.2.0"',
            '"langchain-core>=0.3.0"',
            '"langchain-anthropic>=0.3.0"',
            '"httpx>=0.27.0"',
            '"structlog>=24.1.0"',
            '"sse-starlette>=2.0.0"',
        ],
        "optional_deps": {},
        "cmd": 'CMD ["python", "-m", "ucp_shopping.main"]',
        "topics": "ucp universal-commerce-protocol agentic-commerce shopping-agent langgraph multi-merchant comparison mcp fastapi docker ai-engineer",
        "badges_extra": "",
        "emoji": "🛍️",
    },
    {
        "num": "11",
        "dir": "11-compliance-audit-agents",
        "name": "compliance-audit-agents",
        "pkg": "compliance_audit",
        "display": "Compliance Audit Agents",
        "desc": "Multi-agent compliance audit system with graph workflows, middleware pipelines, and human-in-the-loop - Microsoft Agent Framework patterns",
        "port": 8012,
        "python": "3.11",
        "infra": [],
        "volumes": [],
        "extra_copy": [],
        "env_vars": {
            "OPENAI_API_KEY": "sk-your-openai-key",
            "ANTHROPIC_API_KEY": "sk-ant-your-anthropic-key",
            "ENVIRONMENT": "development",
        },
        "ci_secrets": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        "deps": [
            '"fastapi>=0.115.0"',
            '"uvicorn[standard]>=0.32.0"',
            '"pydantic>=2.6.0"',
            '"pydantic-settings>=2.2.0"',
            '"openai>=1.50.0"',
            '"anthropic>=0.40.0"',
            '"opentelemetry-api>=1.25.0"',
            '"opentelemetry-sdk>=1.25.0"',
            '"httpx>=0.27.0"',
            '"structlog>=24.1.0"',
            '"sse-starlette>=2.0.0"',
        ],
        "optional_deps": {},
        "cmd": 'CMD ["python", "-m", "compliance_audit.main"]',
        "topics": "microsoft-agent-framework compliance audit sox gdpr soc2 graph-workflow middleware human-in-the-loop fastapi docker ai-engineer",
        "badges_extra": "",
        "emoji": "🛡️",
    },
    {
        "num": "12",
        "dir": "12-incident-response-adk",
        "name": "incident-response-adk",
        "pkg": "incident_response",
        "display": "Incident Response Orchestrator",
        "desc": "Automated IT incident response with SequentialAgent, ParallelAgent, LoopAgent - Google ADK patterns",
        "port": 8013,
        "python": "3.11",
        "infra": [],
        "volumes": [],
        "extra_copy": [],
        "env_vars": {
            "OPENAI_API_KEY": "sk-your-openai-key",
            "ANTHROPIC_API_KEY": "sk-ant-your-anthropic-key",
            "ENVIRONMENT": "development",
        },
        "ci_secrets": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        "deps": [
            '"fastapi>=0.115.0"',
            '"uvicorn[standard]>=0.32.0"',
            '"pydantic>=2.6.0"',
            '"pydantic-settings>=2.2.0"',
            '"openai>=1.50.0"',
            '"anthropic>=0.40.0"',
            '"httpx>=0.27.0"',
            '"structlog>=24.1.0"',
            '"sse-starlette>=2.0.0"',
        ],
        "optional_deps": {},
        "cmd": 'CMD ["python", "-m", "incident_response.main"]',
        "topics": "google-adk agent-development-kit incident-response sre devops sequential-agent parallel-agent loop-agent fastapi docker ai-engineer",
        "badges_extra": "",
        "emoji": "🚨",
    },
    {
        "num": "13",
        "dir": "13-contract-lifecycle-crew",
        "name": "contract-lifecycle-crew",
        "pkg": "contract_lifecycle",
        "display": "Contract Lifecycle Crew",
        "desc": "End-to-end contract lifecycle management with role-based Crews and event-driven Flows - CrewAI patterns",
        "port": 8014,
        "python": "3.11",
        "infra": [],
        "volumes": [],
        "extra_copy": [],
        "env_vars": {
            "OPENAI_API_KEY": "sk-your-openai-key",
            "ANTHROPIC_API_KEY": "sk-ant-your-anthropic-key",
            "ENVIRONMENT": "development",
        },
        "ci_secrets": ["OPENAI_API_KEY", "ANTHROPIC_API_KEY"],
        "deps": [
            '"fastapi>=0.115.0"',
            '"uvicorn[standard]>=0.32.0"',
            '"pydantic>=2.6.0"',
            '"pydantic-settings>=2.2.0"',
            '"crewai>=1.9.0"',
            '"openai>=1.50.0"',
            '"anthropic>=0.40.0"',
            '"httpx>=0.27.0"',
            '"structlog>=24.1.0"',
            '"sse-starlette>=2.0.0"',
        ],
        "optional_deps": {},
        "cmd": 'CMD ["python", "-m", "contract_lifecycle.main"]',
        "topics": "crewai crew-based-agents contract-lifecycle legal-tech risk-assessment negotiation flow-orchestration fastapi docker ai-engineer",
        "badges_extra": "",
        "emoji": "📝",
    },
]

# ── Infrastructure service definitions ────────────────────────────────────────

INFRA_SERVICES = {
    "postgres": """\
  postgres:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: aiportfolio
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-localdev}
      POSTGRES_DB: aiportfolio
    ports:
      - "5432:5432"
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U aiportfolio"]
      interval: 5s
      timeout: 5s
      retries: 5""",
    "redis": """\
  redis:
    image: redis:7-alpine
    ports:
      - "6379:6379"
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 5s
      timeout: 5s
      retries: 5""",
    "qdrant": """\
  qdrant:
    image: qdrant/qdrant:latest
    ports:
      - "6333:6333"
      - "6334:6334"
    volumes:
      - qdrant-data:/qdrant/storage
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:6333/healthz"]
      interval: 5s
      timeout: 5s
      retries: 5""",
}

INFRA_VOLUMES = {
    "postgres": "  postgres-data:",
    "qdrant": "  qdrant-data:",
}


# ── Generator functions ───────────────────────────────────────────────────────


def generate_pyproject(proj: dict) -> str:
    deps = ",\n    ".join(proj["deps"])
    optional = ""
    for group, group_deps in proj.get("optional_deps", {}).items():
        optional += f'\n{group} = [\n    {", ".join(group_deps)}\n]'

    optional_section = ""
    if optional:
        optional_section = f"\n[project.optional-dependencies]\ndev = [\"pytest>=8.0.0\", \"pytest-asyncio>=0.24.0\", \"pytest-cov>=5.0.0\", \"httpx>=0.27.0\"]{optional}\n"
    else:
        optional_section = '\n[project.optional-dependencies]\ndev = ["pytest>=8.0.0", "pytest-asyncio>=0.24.0", "pytest-cov>=5.0.0", "httpx>=0.27.0"]\n'

    return f'''[project]
name = "{proj["name"]}"
version = "0.1.0"
description = "{proj["desc"]}"
readme = "README.md"
requires-python = ">=3.11"
license = {{ text = "MIT" }}
dependencies = [
    {deps},
]
{optional_section}
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/{proj["pkg"]}", "src/common"]

[tool.ruff]
target-version = "py311"
line-length = 100

[tool.ruff.lint]
select = ["E", "W", "F", "I", "B", "C4", "UP", "SIM", "TCH", "RUF"]
ignore = ["E501"]

[tool.ruff.lint.isort]
known-first-party = ["{proj["pkg"]}"]

[tool.mypy]
python_version = "3.11"
warn_return_any = true
warn_unused_configs = true
disallow_untyped_defs = true
check_untyped_defs = true

[tool.pytest.ini_options]
testpaths = ["tests"]
asyncio_mode = "auto"
addopts = "-v --tb=short"
markers = [
    "slow: marks tests as slow (deselect with '-m \\"not slow\\"')",
    "integration: marks integration tests requiring external services",
]
'''


def generate_dockerfile(proj: dict) -> str:
    py_ver = proj["python"]
    has_postgres = "postgres" in proj["infra"]

    # Build stage system deps
    sys_deps_build = ""
    if has_postgres:
        sys_deps_build = textwrap.dedent("""\

        # System deps for building Python packages
        RUN apt-get update && \\
            apt-get install -y --no-install-recommends gcc libpq-dev && \\
            rm -rf /var/lib/apt/lists/*
        """)

    # Runtime system deps
    sys_deps_runtime = ""
    if has_postgres:
        sys_deps_runtime = textwrap.dedent("""\

        # Runtime system deps (libpq for asyncpg)
        RUN apt-get update && \\
            apt-get install -y --no-install-recommends libpq5 curl && \\
            rm -rf /var/lib/apt/lists/*
        """)

    # Extra COPY for data dirs
    extra_copy_build = ""
    extra_copy_runtime = ""
    for d in proj.get("extra_copy", []):
        extra_copy_build += f"COPY {d}/ ./{d}/\n"
        extra_copy_runtime += f"COPY --from=builder /build/{d} ./{d}\n"

    return f"""# Stage 1: Build
FROM python:{py_ver}-slim AS builder

WORKDIR /build
{sys_deps_build}
# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies first (layer cache optimisation)
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \\
    pip install --no-cache-dir . || true

# Copy source and install the package
COPY src/ ./src/
{extra_copy_build}RUN pip install --no-cache-dir .

# Stage 2: Runtime
FROM python:{py_ver}-slim AS runtime

# Security: run as non-root
RUN groupadd --gid 1000 appuser && \\
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser
{sys_deps_runtime}
# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \\
    PYTHONDONTWRITEBYTECODE=1 \\
    PYTHONUNBUFFERED=1

WORKDIR /app

# Copy application code
COPY --from=builder /build/src ./src
{extra_copy_runtime}
# Switch to non-root user
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \\
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()"

{proj["cmd"]}
"""


def generate_docker_compose(proj: dict) -> str:
    # App service
    depends = ""
    if proj["infra"]:
        deps_list = "\n".join(
            [f"      {svc}:\n        condition: service_healthy" for svc in proj["infra"]]
        )
        depends = f"\n    depends_on:\n{deps_list}"

    volume_mounts = ""
    if proj["volumes"]:
        vol_list = "\n".join([f"      - {v}" for v in proj["volumes"]])
        volume_mounts = f"\n    volumes:\n{vol_list}"

    infra_blocks = "\n\n".join([INFRA_SERVICES[svc] for svc in proj["infra"]])
    if infra_blocks:
        infra_blocks = "\n\n" + infra_blocks

    # Volumes section
    volume_defs = []
    for svc in proj["infra"]:
        if svc in INFRA_VOLUMES:
            volume_defs.append(INFRA_VOLUMES[svc])
    for v in proj["volumes"]:
        vol_name = v.split(":")[0]
        volume_defs.append(f"  {vol_name}:")

    volumes_section = ""
    if volume_defs:
        volumes_section = "\n\nvolumes:\n" + "\n".join(volume_defs)

    return f"""services:
  app:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "{proj["port"]}:8000"
    env_file: .env{depends}{volume_mounts}
    healthcheck:
      test: ["CMD", "python", "-c", "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"]
      interval: 30s
      timeout: 10s
      retries: 3{infra_blocks}{volumes_section}
"""


def generate_env_example(proj: dict) -> str:
    lines = [
        f"# {'=' * 50}",
        f"# {proj['display']} - Environment Variables",
        f"# {'=' * 50}",
        "# Copy this to .env and fill in your values",
        "",
    ]
    for key, val in proj["env_vars"].items():
        lines.append(f"{key}={val}")

    lines.extend([
        "",
        "# --- Observability (optional) ---",
        "LANGCHAIN_TRACING_V2=false",
        "LANGCHAIN_API_KEY=your-langsmith-key",
        f"LANGCHAIN_PROJECT={proj['name']}",
        "",
        "# --- Application ---",
        "LOG_LEVEL=INFO",
        "ENVIRONMENT=development",
        "",
    ])
    return "\n".join(lines)


def generate_ci(proj: dict) -> str:
    secrets_env = "\n".join(
        [f"          {s}: ${{{{ secrets.{s} }}}}" for s in proj["ci_secrets"]]
    )

    return f"""name: CI

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

env:
  PYTHON_VERSION: "3.11"

jobs:
  lint:
    name: Lint & Format
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{{{ env.PYTHON_VERSION }}}}

      - name: Install dependencies
        run: |
          pip install ruff mypy
          pip install -e ".[dev]"

      - name: Run Ruff linter
        run: ruff check --output-format=github .

      - name: Run Ruff formatter check
        run: ruff format --check .

      - name: Run mypy
        run: mypy src/ --ignore-missing-imports

  test:
    name: Test
    runs-on: ubuntu-latest
    needs: lint
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: ${{{{ env.PYTHON_VERSION }}}}

      - name: Install dependencies
        run: pip install -e ".[dev]"

      - name: Run tests
        run: pytest tests/ -v --tb=short --cov=src/
        env:
{secrets_env}

  docker:
    name: Docker Build
    runs-on: ubuntu-latest
    needs: test
    steps:
      - uses: actions/checkout@v4

      - name: Set up Docker Buildx
        uses: docker/setup-buildx-action@v3

      - name: Build Docker image
        uses: docker/build-push-action@v6
        with:
          context: .
          push: false
          tags: {proj["name"]}:${{{{ github.sha }}}}
          cache-from: type=gha
          cache-to: type=gha,mode=max
"""


def generate_gitignore() -> str:
    return """# Python
__pycache__/
*.py[cod]
*$py.class
*.so
*.egg-info/
*.egg
dist/
build/
.eggs/

# Virtual environments
.venv/
venv/
env/

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Environment variables
.env
.env.local
.env.*.local

# OS
.DS_Store
Thumbs.db

# Testing
.coverage
htmlcov/
.pytest_cache/
.mypy_cache/

# Docker
*.log

# ML artifacts
*.pt
*.pth
*.onnx
*.safetensors
*.bin
models/
checkpoints/
data/raw/
data/processed/
wandb/
mlruns/

# Notebooks
.ipynb_checkpoints/

# uv
uv.lock
"""


def generate_readme(proj: dict) -> str:
    infra_note = ""
    if proj["infra"]:
        services = ", ".join([s.capitalize() for s in proj["infra"]])
        infra_note = f"\n> **Infrastructure**: This project uses {services}. The `docker-compose.yml` handles everything automatically.\n"

    docker_compose_note = ""
    if proj["infra"]:
        docker_compose_note = f"""
### Option 1: Docker Compose (Recommended)

```bash
# Clone the repository
git clone https://github.com/samuelvinay91/{proj["name"]}.git
cd {proj["name"]}

# Copy environment variables
cp .env.example .env
# Edit .env with your API keys

# Start everything (app + infrastructure)
docker compose up --build
```

The API will be available at **http://localhost:{proj["port"]}**. Docs at **http://localhost:{proj["port"]}/docs**.
"""
    else:
        docker_compose_note = f"""
### Option 1: Docker (Recommended)

```bash
# Clone the repository
git clone https://github.com/samuelvinay91/{proj["name"]}.git
cd {proj["name"]}

# Build and run
docker build -t {proj["name"]} .
docker run -p {proj["port"]}:8000 --env-file .env {proj["name"]}
```

The API will be available at **http://localhost:{proj["port"]}**. Docs at **http://localhost:{proj["port"]}/docs**.
"""

    env_setup = "\n".join([f"export {k}={v}" for k, v in list(proj["env_vars"].items())[:3]])

    return f"""# {proj["emoji"]} {proj["display"]}

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-ready-2496ED.svg?logo=docker)](Dockerfile)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![CI](https://github.com/samuelvinay91/{proj["name"]}/actions/workflows/ci.yml/badge.svg)](https://github.com/samuelvinay91/{proj["name"]}/actions)

{proj["desc"]}
{infra_note}
---

## Quick Start
{docker_compose_note}
### Option 2: Local Development

```bash
# Clone and setup
git clone https://github.com/samuelvinay91/{proj["name"]}.git
cd {proj["name"]}

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate

# Install dependencies
pip install -e ".[dev]"

# Set environment variables
cp .env.example .env
# Edit .env with your API keys
{env_setup}

# Run the server
python -m {proj["pkg"]}.main
```

### Option 3: uv (Fast)

```bash
uv venv && source .venv/bin/activate
uv pip install -e ".[dev]"
python -m {proj["pkg"]}.main
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Health check |
| GET | `/docs` | Interactive API documentation |

> See the full API docs at `http://localhost:{proj["port"]}/docs` after starting the server.

---

## Project Structure

```
{proj["name"]}/
├── src/{proj["pkg"]}/          # Application source code
│   ├── __init__.py
│   ├── main.py                 # Uvicorn entry point
│   ├── api.py                  # FastAPI routes
│   └── config.py               # Settings
├── tests/                      # Test suite
├── k8s/                        # Kubernetes manifests
│   └── deployment.yaml
├── .github/workflows/ci.yml    # CI/CD pipeline
├── Dockerfile                  # Multi-stage Docker build
├── docker-compose.yml          # Docker Compose with dependencies
├── pyproject.toml              # Dependencies & tool config
├── .env.example                # Environment variable template
└── README.md
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=src/{proj["pkg"]}

# Skip slow/integration tests
pytest tests/ -v -m "not slow and not integration"
```

---

## Docker

```bash
# Build
docker build -t {proj["name"]} .

# Run
docker run -p {proj["port"]}:8000 --env-file .env {proj["name"]}
```

## Kubernetes

```bash
# Apply manifests
kubectl apply -f k8s/

# Create secrets first
kubectl create secret generic {proj["name"]}-secrets \\
  --from-literal=openai-api-key=$OPENAI_API_KEY \\
  --from-literal=anthropic-api-key=$ANTHROPIC_API_KEY
```

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/my-feature`
3. Install dev dependencies: `pip install -e ".[dev]"`
4. Run tests: `pytest tests/ -v`
5. Submit a pull request

---

## License

MIT License - see [LICENSE](LICENSE) for details.
"""


def generate_k8s_kustomization(proj: dict) -> str:
    resources = ["  - deployment.yaml"]
    for svc in proj["infra"]:
        resources.append(f"  - {svc}.yaml")

    return f"""apiVersion: kustomize.config.k8s.io/v1beta1
kind: Kustomization

resources:
{chr(10).join(resources)}

commonLabels:
  app: {proj["name"]}
"""


def copy_k8s_infra(proj: dict, dest: Path) -> None:
    """Copy relevant K8s infrastructure manifests."""
    k8s_base = ROOT / "k8s" / "base"
    for svc in proj["infra"]:
        src = k8s_base / f"{svc}.yaml"
        if src.exists():
            shutil.copy2(src, dest / f"{svc}.yaml")


# ── Main generation logic ─────────────────────────────────────────────────────


def generate_repo(proj: dict) -> None:
    repo_dir = STANDALONE_DIR / proj["name"]

    # Clean if exists
    if repo_dir.exists():
        shutil.rmtree(repo_dir)

    repo_dir.mkdir(parents=True)

    # 1. Copy project source
    src_origin = ROOT / "projects" / proj["dir"] / "src"
    if src_origin.exists():
        shutil.copytree(src_origin, repo_dir / "src")

    # 1b. Inline the common library into src/common/
    common_origin = ROOT / "libs" / "common" / "src" / "common"
    if common_origin.exists():
        shutil.copytree(common_origin, repo_dir / "src" / "common")

    # 2. Copy tests
    tests_origin = ROOT / "projects" / proj["dir"] / "tests"
    if tests_origin.exists():
        shutil.copytree(tests_origin, repo_dir / "tests")

    # 3. Copy extra directories (e.g., data/)
    for d in proj.get("extra_copy", []):
        src_d = ROOT / "projects" / proj["dir"] / d
        if src_d.exists():
            shutil.copytree(src_d, repo_dir / d)

    # 4. Copy K8s deployment manifest
    k8s_origin = ROOT / "projects" / proj["dir"] / "k8s"
    if k8s_origin.exists():
        k8s_dest = repo_dir / "k8s"
        shutil.copytree(k8s_origin, k8s_dest)
        # Copy infra manifests and kustomization
        copy_k8s_infra(proj, k8s_dest)
        (k8s_dest / "kustomization.yaml").write_text(generate_k8s_kustomization(proj))

    # 5. Copy existing README from project (we'll keep the original detailed README)
    readme_origin = ROOT / "projects" / proj["dir"] / "README.md"
    if readme_origin.exists():
        original_readme = readme_origin.read_text()
        fixed_readme = original_readme

        # Fix license path
        fixed_readme = fixed_readme.replace("../../LICENSE", "LICENSE")

        # Fix all monorepo project path references
        fixed_readme = fixed_readme.replace(f"projects/{proj['dir']}/Dockerfile", "Dockerfile")
        fixed_readme = fixed_readme.replace(f"-f projects/{proj['dir']}/Dockerfile -t", "-t")
        fixed_readme = fixed_readme.replace(f"cd projects/{proj['dir']}", "# Already in project root")
        fixed_readme = fixed_readme.replace(f"projects/{proj['dir']}", ".")

        # Fix common library references
        fixed_readme = fixed_readme.replace("pip install -e libs/common\n", "")
        fixed_readme = fixed_readme.replace("pip install -e libs/common\r\n", "")
        fixed_readme = fixed_readme.replace("pip install -e . -e ", "pip install -e ")
        fixed_readme = fixed_readme.replace("uv pip install -e . -e ", "uv pip install -e ")
        fixed_readme = fixed_readme.replace("libs/common", ".")

        # Fix "From the repository root" comments
        fixed_readme = fixed_readme.replace("# From the repository root\n", "")
        fixed_readme = fixed_readme.replace("# From the repository root", "")

        # Fix Docker build command
        fixed_readme = fixed_readme.replace(
            f"docker build -f Dockerfile -t {proj['name']} .",
            f"docker build -t {proj['name']} .",
        )

        (repo_dir / "README.md").write_text(fixed_readme)
    else:
        (repo_dir / "README.md").write_text(generate_readme(proj))

    # 6. Generate standalone files
    (repo_dir / "pyproject.toml").write_text(generate_pyproject(proj))
    (repo_dir / "Dockerfile").write_text(generate_dockerfile(proj))
    (repo_dir / "docker-compose.yml").write_text(generate_docker_compose(proj))
    (repo_dir / ".env.example").write_text(generate_env_example(proj))
    (repo_dir / ".gitignore").write_text(generate_gitignore())
    (repo_dir / "LICENSE").write_text((ROOT / "LICENSE").read_text())

    # 7. CI/CD
    ci_dir = repo_dir / ".github" / "workflows"
    ci_dir.mkdir(parents=True)
    (ci_dir / "ci.yml").write_text(generate_ci(proj))

    print(f"  Generated: {proj['name']}/")


def generate_create_repos_script() -> str:
    """Generate a helper script to create GitHub repos and push."""
    project_lines = []
    for p in PROJECTS:
        project_lines.append(f'  "{p["name"]}"')

    projects_array = "\n".join(project_lines)
    topics_map = "\n".join(
        [f'  ["{p["name"]}"]="{p["topics"]}"' for p in PROJECTS]
    )
    desc_map = "\n".join(
        [f'  ["{p["name"]}"]="{p["desc"]}"' for p in PROJECTS]
    )

    return f'''#!/usr/bin/env bash
# =============================================================================
# Create separate GitHub repos for each AI Engineer Portfolio project
# =============================================================================
# Usage:
#   ./create-github-repos.sh                    # Create all repos
#   ./create-github-repos.sh llm-playground     # Create one repo
#   DRY_RUN=1 ./create-github-repos.sh          # Preview without creating
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
STANDALONE_DIR="${{SCRIPT_DIR}}/standalone-repos"
GITHUB_USER="${{GITHUB_USER:-$(gh api user -q .login 2>/dev/null || echo "samuelvinay91")}}"
DRY_RUN="${{DRY_RUN:-0}}"

# Projects
PROJECTS=(
{projects_array}
)

# Topics for each repo
declare -A TOPICS=(
{topics_map}
)

# Descriptions
declare -A DESCRIPTIONS=(
{desc_map}
)

create_repo() {{
  local name="$1"
  local desc="${{DESCRIPTIONS[$name]}}"
  local topics="${{TOPICS[$name]}}"
  local repo_dir="${{STANDALONE_DIR}}/${{name}}"

  if [[ ! -d "$repo_dir" ]]; then
    echo "ERROR: Directory $repo_dir not found. Run generate_standalone_repos.py first."
    return 1
  fi

  echo ""
  echo "================================================================"
  echo "  Creating repo: ${{GITHUB_USER}}/${{name}}"
  echo "================================================================"

  if [[ "$DRY_RUN" == "1" ]]; then
    echo "  [DRY RUN] Would create: ${{GITHUB_USER}}/${{name}}"
    echo "  [DRY RUN] Description: ${{desc}}"
    echo "  [DRY RUN] Topics: ${{topics}}"
    return 0
  fi

  cd "$repo_dir"

  # Initialize git repo
  git init -b main
  git add .
  git commit -m "Initial commit: ${{name}} - AI Engineer Portfolio Project"

  # Create GitHub repo
  gh repo create "${{name}}" \\
    --public \\
    --description "${{desc}}" \\
    --source . \\
    --remote origin \\
    --push

  # Set topics
  for topic in $topics; do
    gh repo edit "${{GITHUB_USER}}/${{name}}" --add-topic "$topic" 2>/dev/null || true
  done

  echo "  Created: https://github.com/${{GITHUB_USER}}/${{name}}"
  cd - > /dev/null
}}

# Main
echo "GitHub User: ${{GITHUB_USER}}"
echo "Standalone Dir: ${{STANDALONE_DIR}}"
echo ""

if [[ $# -gt 0 ]]; then
  # Create specific repo
  create_repo "$1"
else
  # Create all repos
  for name in "${{PROJECTS[@]}}"; do
    create_repo "$name"
  done
fi

echo ""
echo "Done! All repos created at https://github.com/${{GITHUB_USER}}"
'''


def main() -> None:
    print(f"Generating standalone repos in: {STANDALONE_DIR}")
    print()

    STANDALONE_DIR.mkdir(exist_ok=True)

    for proj in PROJECTS:
        generate_repo(proj)

    # Generate helper script
    script_path = ROOT / "create-github-repos.sh"
    script_path.write_text(generate_create_repos_script())
    script_path.chmod(0o755)
    print(f"\n  Generated: create-github-repos.sh")

    print(f"\nAll {len(PROJECTS)} standalone repos generated successfully!")
    print(f"\nNext steps:")
    print(f"  1. Review repos in: {STANDALONE_DIR}")
    print(f"  2. Update samuelvinay91 in READMEs and badges")
    print(f"  3. Run: ./create-github-repos.sh")
    print(f"     Or for a dry run: DRY_RUN=1 ./create-github-repos.sh")


if __name__ == "__main__":
    main()
