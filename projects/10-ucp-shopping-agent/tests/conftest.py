"""Shared test fixtures for UCP Shopping Agent."""

import pytest

from ucp_shopping.config import ShoppingSettings
from ucp_shopping.main import create_app


@pytest.fixture
def settings():
    """Create test settings."""
    return ShoppingSettings(
        environment="testing",
        openai_api_key="test-key",
        human_confirmation_required=False,
    )


@pytest.fixture
def app(settings):
    """Create FastAPI app for testing."""
    return create_app(settings)
