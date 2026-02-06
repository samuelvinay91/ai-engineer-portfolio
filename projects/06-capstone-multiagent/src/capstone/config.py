"""Application configuration for the Capstone Multi-Agent Platform.

Loads settings from environment variables with sensible defaults for local
development. Every secret or deployment-specific value is configurable via
env vars so the same image works across dev / staging / prod without code
changes.
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Deployment environment identifier."""

    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class Settings(BaseSettings):
    """Central application settings for the multi-agent platform.

    Values are loaded from environment variables. The ``model_config``
    directive tells pydantic-settings to look for a ``.env`` file as well.
    """

    model_config = SettingsConfigDict(
        env_prefix="CAPSTONE_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Identity / environment ------------------------------------------------
    app_name: str = "capstone-multiagent"
    app_version: str = "0.1.0"
    environment: Environment = Environment.LOCAL
    debug: bool = False
    log_level: str = "INFO"

    # -- LLM providers ---------------------------------------------------------
    anthropic_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")

    # -- Model selection -------------------------------------------------------
    supervisor_model: str = "claude-sonnet-4-5-20250929"
    researcher_model: str = "claude-sonnet-4-5-20250929"
    coder_model: str = "claude-sonnet-4-5-20250929"
    analyst_model: str = "claude-sonnet-4-5-20250929"
    writer_model: str = "claude-sonnet-4-5-20250929"

    # -- Model parameters ------------------------------------------------------
    default_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    default_max_tokens: int = Field(default=4096, ge=1, le=32768)

    # -- Infrastructure --------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/capstone"

    # -- Orchestration ---------------------------------------------------------
    max_agent_retries: int = Field(default=2, ge=0, le=5)
    agent_timeout_seconds: int = Field(default=120, ge=10, le=600)
    max_parallel_agents: int = Field(default=4, ge=1, le=10)
    max_subtasks: int = Field(default=8, ge=1, le=20)

    # -- Memory ----------------------------------------------------------------
    short_term_max_messages: int = Field(
        default=50,
        ge=5,
        le=200,
        description="Maximum messages in short-term conversation memory.",
    )
    working_memory_ttl_seconds: int = Field(
        default=3600,
        ge=60,
        description="TTL for working memory entries in Redis (seconds).",
    )
    long_term_relevance_threshold: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Minimum relevance score for long-term memory retrieval.",
    )

    # -- Server ----------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1
    cors_origins: list[str] = ["*"]

    # -- Search (for researcher agent) -----------------------------------------
    tavily_api_key: str = ""
    max_search_results: int = 8

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of application settings."""
    return Settings()
