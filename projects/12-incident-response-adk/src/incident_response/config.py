"""Configuration management for the Incident Response Orchestrator.

Extends the common Settings base class with incident-response-specific
settings including escalation limits, remediation timeouts, and session TTL.
"""

from __future__ import annotations

from common.config import Settings as BaseSettings


class Settings(BaseSettings):
    """Incident Response Orchestrator configuration.

    Inherits common provider keys and infrastructure settings from
    ``common.config.Settings`` and adds incident-response-specific options.
    """

    # Service identity
    service_name: str = "incident-response-adk"
    service_version: str = "0.1.0"
    host: str = "0.0.0.0"
    port: int = 8013

    # LLM configuration (Gemini model to match ADK defaults)
    default_model: str = "gemini-2.0-flash"

    # Escalation
    max_escalation_levels: int = 3

    # Remediation
    remediation_timeout: int = 120

    # Session management
    session_ttl_seconds: int = 3600


def get_settings() -> Settings:
    """Return a cached settings instance."""
    return Settings()
