"""Application configuration using pydantic-settings.

Loads settings from environment variables with sensible defaults for local
development.  Every secret or deployment-specific value is configurable via
env vars so the same image works across dev / staging / prod without code
changes.
"""

from __future__ import annotations

import os
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Self

from pydantic import Field, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(StrEnum):
    """Deployment environment identifier."""

    LOCAL = "local"
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class LogLevel(StrEnum):
    """Supported structured-log levels."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class Settings(BaseSettings):
    """Central application settings.

    Values are loaded from environment variables.  The ``model_config``
    directive tells pydantic-settings to look for a ``.env`` file as well.
    """

    model_config = SettingsConfigDict(
        env_prefix="CHATBOT_",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Identity / environment ------------------------------------------------
    app_name: str = "customer-support-chatbot"
    environment: Environment = Environment.LOCAL
    debug: bool = False
    log_level: LogLevel = LogLevel.INFO

    # -- LLM provider ----------------------------------------------------------
    default_model: str = "claude-sonnet-4-5-20250929"
    anthropic_api_key: SecretStr = SecretStr("")
    openai_api_key: SecretStr = SecretStr("")
    model_temperature: float = Field(default=0.3, ge=0.0, le=2.0)
    model_max_tokens: int = Field(default=2048, ge=1, le=8192)

    # -- Infrastructure --------------------------------------------------------
    redis_url: str = "redis://localhost:6379/0"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/chatbot"

    # -- Prompt templates ------------------------------------------------------
    prompt_template_dir: Path = Path(__file__).resolve().parent.parent.parent / "data" / "templates"

    # -- Conversation ----------------------------------------------------------
    max_conversation_history: int = Field(
        default=20,
        ge=1,
        le=200,
        description="Maximum number of messages retained in the sliding window.",
    )
    conversation_ttl_seconds: int = Field(
        default=86400,
        ge=60,
        description="Time-to-live for idle conversation sessions in Redis (seconds).",
    )
    context_summary_threshold: int = Field(
        default=15,
        ge=5,
        description="Number of messages that triggers automatic context summarization.",
    )

    # -- Fine-tuning defaults --------------------------------------------------
    training_output_dir: Path = Path("./training_output")
    wandb_project: str = "customer-support-chatbot"
    lora_rank: int = Field(default=16, ge=1, le=256)
    lora_alpha: int = Field(default=32, ge=1, le=512)
    lora_dropout: float = Field(default=0.05, ge=0.0, le=0.5)

    # -- Server ----------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 1

    # -- Validation ------------------------------------------------------------
    @model_validator(mode="after")
    def _resolve_template_dir(self) -> Self:
        """Make prompt_template_dir absolute regardless of how it was provided."""
        if not self.prompt_template_dir.is_absolute():
            self.prompt_template_dir = Path(os.getcwd()) / self.prompt_template_dir
        return self

    @property
    def is_production(self) -> bool:
        return self.environment == Environment.PRODUCTION


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of application settings."""
    return Settings()
