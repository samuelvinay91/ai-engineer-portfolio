"""Project-specific configuration for the LLM Playground."""

from functools import lru_cache

from common import Settings as BaseSettings


class ModelInfo:
    """Registry of supported model identifiers grouped by provider."""

    ANTHROPIC = [
        "claude-sonnet-4-5-20250929",
        "claude-haiku-4-5-20250929",
        "claude-opus-4-0-20250514",
    ]
    OPENAI = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-4-turbo",
        "gpt-3.5-turbo",
    ]
    OPEN_SOURCE = [
        "meta-llama/Llama-3.1-8B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.3",
        "google/gemma-2-9b-it",
    ]

    @classmethod
    def all_models(cls) -> list[str]:
        return cls.ANTHROPIC + cls.OPENAI + cls.OPEN_SOURCE


class Settings(BaseSettings):
    """LLM Playground settings extending the shared base."""

    app_name: str = "llm-playground"
    app_version: str = "0.1.0"

    default_model: str = "claude-sonnet-4-5-20250929"
    max_tokens: int = 4096
    supported_models: list[str] = ModelInfo.all_models()

    # Generation defaults
    default_temperature: float = 0.7
    default_top_p: float = 0.9
    default_top_k: int = 50

    # Tokenizer defaults
    default_tokenizer: str = "cl100k_base"

    # Rate limiting
    requests_per_minute: int = 60


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
