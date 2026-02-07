"""Shared test fixtures for MCP & A2A tests."""

from __future__ import annotations

from typing import AsyncIterator

import pytest
from httpx import ASGITransport, AsyncClient

from mcp_a2a.api import create_app
from mcp_a2a.config import MCPSettings


@pytest.fixture()
def settings() -> MCPSettings:
    """Return test-specific settings."""
    return MCPSettings(
        environment="test",
        log_level="DEBUG",
        mcp_rate_limit_per_minute=120,
        mcp_audit_log_enabled=True,
    )


@pytest.fixture()
def app(settings: MCPSettings):  # type: ignore[no-untyped-def]
    """Create a FastAPI app instance for testing."""
    return create_app(settings)


@pytest.fixture()
async def client(app) -> AsyncIterator[AsyncClient]:  # type: ignore[no-untyped-def]
    """Async HTTP client wired to the test app."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
