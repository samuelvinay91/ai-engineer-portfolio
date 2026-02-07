"""Configuration management for the Compliance Audit Agents service."""

from __future__ import annotations

from common.config import Settings as BaseSettings


class Settings(BaseSettings):
    """Compliance Audit Agents configuration.

    Inherits common provider keys and infrastructure settings from
    ``common.config.Settings`` and adds compliance-audit-specific options.
    """

    # Service identity
    service_name: str = "compliance-audit-agents"
    service_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8012

    # LLM configuration
    default_model: str = "gpt-4o-mini"

    # Human-in-the-loop
    human_approval_required: bool = True

    # Audit limits
    max_transactions_per_audit: int = 100

    # Session management
    session_ttl_seconds: int = 3600


def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
