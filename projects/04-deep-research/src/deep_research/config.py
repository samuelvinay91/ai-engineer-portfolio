"""Configuration for the Deep Research service.

Centralises every tuneable knob behind a single ``Settings`` instance that is
read from environment variables (with sensible defaults for local development).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application-wide settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="DEEP_RESEARCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- LLM Models -----------------------------------------------------------
    reasoning_model: str = Field(
        default="claude-opus-4-6",
        description="Model used for complex, multi-step reasoning tasks.",
    )
    fast_model: str = Field(
        default="claude-sonnet-4-5-20250929",
        description="Model used for quick, low-latency tasks (summaries, extractions).",
    )

    # --- Reasoning Parameters -------------------------------------------------
    thinking_budget_tokens: int = Field(
        default=10_000,
        ge=1_000,
        le=128_000,
        description="Maximum tokens allocated for the model's extended thinking.",
    )
    max_reasoning_steps: int = Field(
        default=15,
        ge=1,
        le=50,
        description="Hard cap on chain-of-thought reasoning steps.",
    )
    confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum confidence score to accept a reasoning conclusion.",
    )

    # --- Research Parameters --------------------------------------------------
    max_research_depth: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Maximum number of research-iteration loops.",
    )
    max_parallel_searches: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Maximum concurrent web-search requests.",
    )
    max_sources_per_query: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Maximum search results to retain per sub-question.",
    )

    # --- Tree-of-Thought Parameters -------------------------------------------
    tot_branching_factor: int = Field(
        default=3,
        ge=2,
        le=6,
        description="Number of branches to explore at each Tree-of-Thought node.",
    )
    tot_max_depth: int = Field(
        default=3,
        ge=1,
        le=5,
        description="Maximum depth of the Tree-of-Thought search.",
    )

    # --- API keys (loaded from env / secrets) ---------------------------------
    anthropic_api_key: str = Field(default="", description="Anthropic API key.")
    openai_api_key: str = Field(default="", description="OpenAI API key.")
    tavily_api_key: str = Field(default="", description="Tavily search API key.")

    # --- Infrastructure -------------------------------------------------------
    redis_url: str = Field(default="redis://localhost:6379/0")
    database_url: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:5432/deep_research",
    )
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    environment: Literal["development", "staging", "production"] = "development"

    # --- Server ---------------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached ``Settings`` singleton."""
    return Settings()
