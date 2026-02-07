"""Configuration for the Agent & RAG system."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbeddingProvider(str, Enum):
    OPENAI = "openai"
    COHERE = "cohere"
    LOCAL = "local"


class LLMProvider(str, Enum):
    ANTHROPIC = "anthropic"
    OPENAI = "openai"


class ChunkStrategy(str, Enum):
    FIXED = "fixed"
    SEMANTIC = "semantic"
    HIERARCHICAL = "hierarchical"
    RECURSIVE = "recursive"


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_RAG_",
        env_file=".env",
        extra="ignore",
    )

    # --- API ---
    host: str = "0.0.0.0"
    port: int = 8007
    debug: bool = False
    log_level: str = "info"

    # --- LLM ---
    llm_provider: LLMProvider = LLMProvider.ANTHROPIC
    anthropic_api_key: SecretStr = SecretStr("")
    anthropic_model: str = "claude-sonnet-4-20250514"
    openai_api_key: SecretStr = SecretStr("")
    openai_model: str = "gpt-4o"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 4096

    # --- Embeddings ---
    embedding_provider: EmbeddingProvider = EmbeddingProvider.OPENAI
    embedding_model: str = "text-embedding-3-small"
    embedding_dimensions: int = 1536
    embedding_batch_size: int = 128
    cohere_api_key: SecretStr = SecretStr("")

    # --- Chunking ---
    chunk_strategy: ChunkStrategy = ChunkStrategy.RECURSIVE
    chunk_size: int = 512
    chunk_overlap: int = 64
    min_chunk_size: int = 50

    # --- Qdrant ---
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: SecretStr | None = None
    qdrant_collection: str = "documents"
    qdrant_on_disk: bool = False

    # --- Reranker ---
    reranker_model: str = "rerank-english-v3.0"
    reranker_top_k: int = 5
    reranker_enabled: bool = True

    # --- Retrieval ---
    retrieval_top_k: int = 20
    retrieval_strategy: Literal["keyword", "semantic", "hybrid"] = "hybrid"
    hybrid_alpha: float = 0.7  # Weight for semantic vs keyword (1.0 = pure semantic)
    rrf_k: int = 60  # RRF constant

    # --- Context Engineering ---
    context_max_tokens: int = 8192
    context_compression_enabled: bool = True
    context_overlap_window: int = 128

    # --- Memory ---
    redis_url: str = "redis://localhost:6379/0"
    conversation_ttl_seconds: int = 3600
    episodic_memory_size: int = 100
    semantic_memory_size: int = 500

    # --- Database ---
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/agent_rag"


settings = Settings()
