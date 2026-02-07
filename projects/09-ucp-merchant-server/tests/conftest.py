"""Shared test fixtures for UCP Merchant Server."""

import pytest

from ucp_merchant.config import MerchantSettings
from ucp_merchant.main import create_app


@pytest.fixture
def settings():
    """Create test settings."""
    return MerchantSettings(
        environment="testing",
        merchant_name="Test Store",
        merchant_domain="test.example.com",
        ap2_enabled=True,
        mcp_enabled=True,
    )


@pytest.fixture
def app(settings):
    """Create FastAPI app for testing."""
    return create_app(settings)
