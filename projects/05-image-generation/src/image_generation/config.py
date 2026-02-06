"""Configuration management for the Image Generation Service."""

from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ProviderName(str, Enum):
    """Supported image generation providers."""

    OPENAI = "openai"
    REPLICATE = "replicate"
    LOCAL = "local"


class ImageSize(str, Enum):
    """Supported output image sizes."""

    SQUARE_SM = "512x512"
    SQUARE_MD = "768x768"
    SQUARE_LG = "1024x1024"
    LANDSCAPE_MD = "1344x768"
    LANDSCAPE_LG = "1536x1024"
    PORTRAIT_MD = "768x1344"
    PORTRAIT_LG = "1024x1536"

    @property
    def width(self) -> int:
        return int(self.value.split("x")[0])

    @property
    def height(self) -> int:
        return int(self.value.split("x")[1])


class Settings(BaseSettings):
    """Application settings for the Image Generation Service.

    All settings can be overridden via environment variables or a ``.env`` file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        env_prefix="IMG_GEN_",
    )

    # --- Provider selection ---
    default_provider: ProviderName = ProviderName.OPENAI

    # --- API keys ---
    openai_api_key: str = ""
    replicate_api_token: str = ""

    # --- Model identifiers ---
    openai_model: str = "dall-e-3"
    replicate_model: str = "stability-ai/stable-diffusion-3.5-large"
    local_model: str = "stabilityai/stable-diffusion-xl-base-1.0"

    # --- Image defaults ---
    default_size: ImageSize = ImageSize.SQUARE_LG
    default_steps: int = Field(default=30, ge=1, le=150)
    default_guidance_scale: float = Field(default=7.5, ge=1.0, le=30.0)
    max_batch_size: int = Field(default=8, ge=1, le=16)
    supported_formats: list[str] = ["png", "jpeg", "webp"]

    # --- Output / storage ---
    output_dir: Path = Path("generated_images")
    storage_backend: str = "local"  # "local" | "s3"

    # --- S3 configuration ---
    s3_bucket: str = ""
    s3_prefix: str = "image-generation/"
    s3_region: str = "us-east-1"
    s3_endpoint_url: str = ""
    aws_access_key_id: str = ""
    aws_secret_access_key: str = ""

    # --- Redis / caching ---
    redis_url: str = "redis://localhost:6379/1"

    # --- Prompt enhancement ---
    prompt_enhance_model: str = "gpt-4o-mini"
    enable_prompt_enhancement: bool = True

    # --- Server ---
    host: str = "0.0.0.0"
    port: int = 8005
    log_level: str = "INFO"
    environment: str = "development"

    # --- Rate limiting ---
    rate_limit_rpm: int = 30

    @property
    def is_production(self) -> bool:
        return self.environment == "production"


def get_settings() -> Settings:
    """Return a cached Settings instance."""
    return Settings()
