"""Configuration for the Ask-the-Web agent."""

from __future__ import annotations

from common.config import Settings as BaseSettings


class Settings(BaseSettings):
    """Ask-the-Web agent settings.

    Inherits shared settings (API keys, infra URLs) from common.Settings
    and adds project-specific configuration for search, synthesis, and routing.
    """

    # --- Tavily search ---------------------------------------------------------
    tavily_api_key: str = ""
    max_search_results: int = 8
    search_timeout_seconds: int = 15

    # --- LLM defaults ----------------------------------------------------------
    default_model: str = "claude-sonnet-4-20250514"
    router_model: str = "claude-sonnet-4-20250514"
    synthesizer_model: str = "claude-sonnet-4-20250514"
    fact_checker_model: str = "claude-sonnet-4-20250514"
    temperature: float = 0.2
    max_tokens: int = 4096

    # --- Workflow tuning -------------------------------------------------------
    max_search_queries: int = 3
    max_sources_per_answer: int = 10
    fact_check_enabled: bool = True
    confidence_threshold: float = 0.6

    # --- Application -----------------------------------------------------------
    app_name: str = "ask-the-web-agent"
    app_version: str = "0.1.0"
    cors_origins: list[str] = ["*"]
    history_max_items: int = 50


def get_settings() -> Settings:
    """Return a cached Settings instance (reads .env once)."""
    return Settings()
