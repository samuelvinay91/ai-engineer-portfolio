"""Shared pytest fixtures for the LLM Playground test suite."""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from llm_playground.api import create_app
from llm_playground.config import Settings
from llm_playground.tokenizer import TokenizerService


@pytest.fixture(scope="session")
def settings() -> Settings:
    return Settings(
        environment="test",
        log_level="DEBUG",
        anthropic_api_key="test-key",
        openai_api_key="test-key",
    )


@pytest.fixture(scope="session")
def tokenizer_service() -> TokenizerService:
    return TokenizerService()


@pytest.fixture
async def client() -> AsyncIterator[AsyncClient]:
    app = create_app()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
