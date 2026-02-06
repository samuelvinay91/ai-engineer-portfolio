"""Shared pytest fixtures for the customer-support-chatbot test suite."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, AsyncGenerator

import pytest
import pytest_asyncio

# ---------------------------------------------------------------------------
# Path fixtures
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TEMPLATE_DIR = PROJECT_ROOT / "data" / "templates"


@pytest.fixture(scope="session")
def project_root() -> Path:
    return PROJECT_ROOT


@pytest.fixture(scope="session")
def template_dir() -> Path:
    return TEMPLATE_DIR


# ---------------------------------------------------------------------------
# Settings fixture (overrides for testing)
# ---------------------------------------------------------------------------

@pytest.fixture()
def test_settings():
    """Return a Settings instance configured for testing."""
    from customer_support.config import Settings

    return Settings(
        environment="local",
        debug=True,
        anthropic_api_key="test-key-not-real",
        openai_api_key="test-key-not-real",
        redis_url="redis://localhost:6379/15",  # dedicated test DB
        database_url="sqlite+aiosqlite:///test.db",
        prompt_template_dir=TEMPLATE_DIR,
        max_conversation_history=10,
        conversation_ttl_seconds=300,
    )


# ---------------------------------------------------------------------------
# Prompt registry fixture
# ---------------------------------------------------------------------------

@pytest.fixture()
def prompt_registry(template_dir: Path):
    """Return a fresh PromptRegistry loaded from the test template dir."""
    from customer_support.prompts import PromptRegistry

    registry = PromptRegistry()
    registry.load_from_directory(template_dir)
    return registry


# ---------------------------------------------------------------------------
# Mock LLM callable
# ---------------------------------------------------------------------------

class MockLLM:
    """A deterministic mock LLM for testing.

    Records calls and returns pre-configured responses.
    """

    def __init__(self, default_response: str = "Mock assistant response.") -> None:
        self.default_response = default_response
        self.calls: list[dict[str, Any]] = []
        self._responses: dict[str, str] = {}

    def set_response(self, key: str, response: str) -> None:
        """Register a canned response for messages containing *key*."""
        self._responses[key] = response

    async def __call__(
        self,
        system: str,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> str:
        self.calls.append({
            "system": system,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        })
        # Check for canned responses
        last_user = ""
        for m in reversed(messages):
            if m["role"] == "user":
                last_user = m["content"]
                break
        for key, resp in self._responses.items():
            if key.lower() in last_user.lower():
                return resp
        return self.default_response


@pytest.fixture()
def mock_llm() -> MockLLM:
    return MockLLM()


# ---------------------------------------------------------------------------
# Mock LLM that returns classification JSON
# ---------------------------------------------------------------------------

@pytest.fixture()
def classification_llm() -> MockLLM:
    """An LLM mock pre-configured to return classification JSON."""
    llm = MockLLM()
    llm.default_response = json.dumps({
        "primary_intent": "general",
        "primary_confidence": 0.85,
        "secondary_intents": [],
        "reasoning": "General inquiry about the product.",
    })
    llm.set_response(
        "charged twice",
        json.dumps({
            "primary_intent": "billing",
            "primary_confidence": 0.95,
            "secondary_intents": [
                {"intent": "escalation", "confidence": 0.3},
            ],
            "reasoning": "Customer reports a duplicate charge.",
        }),
    )
    llm.set_response(
        "crash",
        json.dumps({
            "primary_intent": "technical",
            "primary_confidence": 0.92,
            "secondary_intents": [],
            "reasoning": "Customer reports application crash.",
        }),
    )
    llm.set_response(
        "speak to a manager",
        json.dumps({
            "primary_intent": "escalation",
            "primary_confidence": 0.97,
            "secondary_intents": [
                {"intent": "general", "confidence": 0.4},
            ],
            "reasoning": "Customer explicitly requests human agent.",
        }),
    )
    return llm


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------

@pytest.fixture()
def app(test_settings):
    """Create a FastAPI app for testing (no Redis, no real LLM)."""
    from customer_support.api import create_app

    return create_app(settings=test_settings)


@pytest.fixture()
def client(app):
    """Synchronous test client via httpx."""
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://testserver")
