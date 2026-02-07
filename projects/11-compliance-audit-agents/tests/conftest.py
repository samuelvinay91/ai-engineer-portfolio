"""Test fixtures for Compliance Audit Agents."""

from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def _test_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Set up test environment variables."""
    monkeypatch.setenv("ENVIRONMENT", "testing")
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")
    monkeypatch.setenv("HUMAN_APPROVAL_REQUIRED", "false")
    # Clear any real API keys so tests use heuristic fallback
    for key in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture()
def settings():
    """Create test settings."""
    os.environ.setdefault("ENVIRONMENT", "testing")
    from compliance_audit.config import Settings

    return Settings(
        environment="testing",
        log_level="DEBUG",
        human_approval_required=False,
    )


@pytest.fixture()
def app(settings):
    """Create a test FastAPI application."""
    from compliance_audit.api import create_app

    return create_app(settings)


@pytest.fixture()
def client(app):
    """Create an async test client."""
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")
