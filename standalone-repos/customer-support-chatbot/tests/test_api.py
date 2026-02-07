"""Tests for the FastAPI application.

Covers:
- Health endpoint
- Prompt listing and testing endpoints
- Chat endpoint (with mocked LLM)
- Classification endpoint (with mocked LLM)
- Error handling for missing resources
"""

from __future__ import annotations

import json
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from customer_support.api import create_app
from customer_support.config import Settings

# All tests in this module are async
pytestmark = pytest.mark.asyncio


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def test_settings(template_dir):
    """Settings tuned for testing -- no real external services."""
    return Settings(
        environment="local",
        debug=True,
        anthropic_api_key="test-key-not-real",
        openai_api_key="test-key-not-real",
        redis_url="redis://localhost:6379/15",
        database_url="sqlite+aiosqlite:///test.db",
        prompt_template_dir=template_dir,
        max_conversation_history=10,
    )


@pytest.fixture()
def app(test_settings):
    return create_app(settings=test_settings)


@pytest_asyncio.fixture()
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as c:
        yield c


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealthEndpoint:
    async def test_health_returns_200(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200

    async def test_health_body(self, client: AsyncClient) -> None:
        data = (await client.get("/health")).json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert "uptime_seconds" in data


# ---------------------------------------------------------------------------
# Prompt endpoints
# ---------------------------------------------------------------------------


class TestPromptEndpoints:
    async def test_list_prompts(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/prompts")
        assert resp.status_code == 200
        data = resp.json()
        assert "templates" in data
        assert data["total"] >= 3
        names = [t["name"] for t in data["templates"]]
        assert "general_support" in names
        assert "technical_support" in names
        assert "billing_support" in names

    async def test_list_prompts_structure(self, client: AsyncClient) -> None:
        data = (await client.get("/api/v1/prompts")).json()
        tpl = data["templates"][0]
        assert "name" in tpl
        assert "description" in tpl
        assert "current_version" in tpl
        assert "chain_of_thought_enabled" in tpl
        assert "few_shot_example_count" in tpl

    async def test_test_prompt_general(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/prompts/test",
            json={"template_name": "general_support"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["template_name"] == "general_support"
        assert "<role>" in data["rendered_prompt"]
        assert data["character_count"] > 0

    async def test_test_prompt_with_knowledge_base(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/prompts/test",
            json={
                "template_name": "billing_support",
                "knowledge_base": "All plans include a 30-day trial.",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "30-day trial" in data["rendered_prompt"]

    async def test_test_prompt_with_extra_sections(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/prompts/test",
            json={
                "template_name": "general_support",
                "extra_sections": {"sla_info": "99.9% uptime guarantee"},
            },
        )
        assert resp.status_code == 200
        assert "99.9% uptime" in resp.json()["rendered_prompt"]

    async def test_test_prompt_unknown_template(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/prompts/test",
            json={"template_name": "nonexistent_template"},
        )
        assert resp.status_code == 404

    async def test_test_prompt_returns_version(self, client: AsyncClient) -> None:
        data = (
            await client.post(
                "/api/v1/prompts/test",
                json={"template_name": "technical_support"},
            )
        ).json()
        assert data["version"] == "1.0.0"


# ---------------------------------------------------------------------------
# Chat endpoint (requires mocking the LLM)
# ---------------------------------------------------------------------------


class TestChatEndpoint:
    async def test_chat_returns_reply(self, client: AsyncClient) -> None:
        """Patch the Anthropic client to avoid real API calls."""
        mock_response = AsyncMock()
        mock_response.content = [AsyncMock(text="Hello! How can I help you today?")]

        with patch("customer_support.api.anthropic.AsyncAnthropic") as mock_cls:
            instance = AsyncMock()
            instance.messages.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = instance

            resp = await client.post(
                "/api/v1/chat",
                json={"message": "Hi, I need help with my account."},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert "reply" in data
        assert data["reply"] == "Hello! How can I help you today?"
        assert "session_id" in data
        assert "conversation_state" in data

    async def test_chat_with_template_override(self, client: AsyncClient) -> None:
        mock_response = AsyncMock()
        mock_response.content = [AsyncMock(text="Technical support here.")]

        with patch("customer_support.api.anthropic.AsyncAnthropic") as mock_cls:
            instance = AsyncMock()
            instance.messages.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = instance

            resp = await client.post(
                "/api/v1/chat",
                json={
                    "message": "My app crashes.",
                    "template_override": "technical_support",
                },
            )

        assert resp.status_code == 200
        assert resp.json()["reply"] == "Technical support here."

    async def test_chat_with_session_id(self, client: AsyncClient) -> None:
        mock_response = AsyncMock()
        mock_response.content = [AsyncMock(text="Continuing conversation.")]

        with patch("customer_support.api.anthropic.AsyncAnthropic") as mock_cls:
            instance = AsyncMock()
            instance.messages.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = instance

            resp = await client.post(
                "/api/v1/chat",
                json={
                    "message": "Follow-up question.",
                    "session_id": "test-session-123",
                },
            )

        assert resp.status_code == 200
        assert resp.json()["session_id"] == "test-session-123"

    async def test_chat_empty_message_rejected(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/chat", json={"message": ""})
        assert resp.status_code == 422  # validation error

    async def test_chat_missing_message_rejected(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/chat", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Classification endpoint
# ---------------------------------------------------------------------------


class TestClassifyEndpoint:
    async def test_classify_returns_result(self, client: AsyncClient) -> None:
        classification_json = json.dumps({
            "primary_intent": "billing",
            "primary_confidence": 0.93,
            "secondary_intents": [],
            "reasoning": "Customer asks about charges.",
        })

        mock_response = AsyncMock()
        mock_response.content = [AsyncMock(text=classification_json)]

        with patch("customer_support.api.anthropic.AsyncAnthropic") as mock_cls:
            instance = AsyncMock()
            instance.messages.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = instance

            resp = await client.post(
                "/api/v1/classify",
                json={"message": "I was charged twice."},
            )

        assert resp.status_code == 200
        data = resp.json()
        assert data["primary_intent"] == "billing"
        assert data["primary_confidence"] == pytest.approx(0.93, abs=0.01)
        assert "reasoning" in data

    async def test_classify_with_context(self, client: AsyncClient) -> None:
        classification_json = json.dumps({
            "primary_intent": "technical",
            "primary_confidence": 0.88,
            "secondary_intents": [],
            "reasoning": "Ongoing technical issue.",
        })

        mock_response = AsyncMock()
        mock_response.content = [AsyncMock(text=classification_json)]

        with patch("customer_support.api.anthropic.AsyncAnthropic") as mock_cls:
            instance = AsyncMock()
            instance.messages.create = AsyncMock(return_value=mock_response)
            mock_cls.return_value = instance

            resp = await client.post(
                "/api/v1/classify",
                json={
                    "message": "It's still not working.",
                    "context": [
                        {"role": "user", "content": "The app crashes on startup."},
                        {"role": "assistant", "content": "Try clearing your cache."},
                    ],
                },
            )

        assert resp.status_code == 200
        assert resp.json()["primary_intent"] == "technical"


# ---------------------------------------------------------------------------
# Conversation history (Redis-dependent -- graceful degradation)
# ---------------------------------------------------------------------------


class TestConversationEndpoint:
    async def test_conversation_not_found_without_redis(
        self, client: AsyncClient
    ) -> None:
        """Without Redis the endpoint returns 503 (service unavailable)."""
        resp = await client.get("/api/v1/conversations/nonexistent-session")
        # Should be 503 (no Redis) or 404 (not found)
        assert resp.status_code in (404, 503)


# ---------------------------------------------------------------------------
# 404 catch-all
# ---------------------------------------------------------------------------


class TestNotFound:
    async def test_unknown_route(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/nonexistent")
        assert resp.status_code == 404
