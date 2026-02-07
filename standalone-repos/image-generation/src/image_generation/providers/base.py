"""Abstract base class and shared data models for image generation providers."""

from __future__ import annotations

import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------


class GenerationStatus(str, Enum):
    """Lifecycle status of an image generation request."""

    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class GenerationResult(BaseModel):
    """Output of a single image generation call."""

    image_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    provider: str
    model: str
    prompt: str
    negative_prompt: str = ""
    width: int
    height: int
    steps: int | None = None
    guidance_scale: float | None = None
    seed: int | None = None
    status: GenerationStatus = GenerationStatus.COMPLETED
    image_url: str | None = None
    image_bytes: bytes | None = None
    image_format: str = "png"
    duration_seconds: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    error: str | None = None

    model_config = {"arbitrary_types_allowed": True}


class Img2ImgInput(BaseModel):
    """Input for image-to-image generation."""

    image_bytes: bytes
    image_format: str = "png"

    model_config = {"arbitrary_types_allowed": True}


class InpaintInput(BaseModel):
    """Input for inpainting generation."""

    image_bytes: bytes
    mask_bytes: bytes
    image_format: str = "png"

    model_config = {"arbitrary_types_allowed": True}


# ---------------------------------------------------------------------------
# Abstract provider
# ---------------------------------------------------------------------------


class ImageProvider(ABC):
    """Abstract interface that every image generation provider must implement.

    Subclasses wrap a specific backend (OpenAI DALL-E, Replicate, local
    diffusers) and expose a uniform ``generate`` / ``img2img`` / ``inpaint``
    surface.
    """

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Human-readable identifier for this provider."""

    @property
    @abstractmethod
    def supported_models(self) -> list[str]:
        """List of model identifiers this provider supports."""

    @abstractmethod
    async def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate an image from a text prompt.

        Args:
            prompt: Text description of the desired image.
            negative_prompt: Concepts to avoid in the output.
            width: Output width in pixels.
            height: Output height in pixels.
            steps: Number of diffusion / sampling steps (provider-dependent).
            guidance_scale: Classifier-free guidance strength.
            seed: Random seed for reproducibility.
            model: Override the default model for this provider.
            **kwargs: Provider-specific extra parameters.

        Returns:
            A :class:`GenerationResult` containing the generated image or a URL.
        """

    async def img2img(
        self,
        image: Img2ImgInput,
        prompt: str,
        *,
        strength: float = 0.75,
        negative_prompt: str = "",
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Transform an existing image guided by a text prompt.

        Not all providers support img2img. The default implementation raises
        :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            f"{self.provider_name} does not support image-to-image generation."
        )

    async def inpaint(
        self,
        image: InpaintInput,
        prompt: str,
        *,
        negative_prompt: str = "",
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Fill masked regions of an image guided by a text prompt.

        Not all providers support inpainting. The default implementation raises
        :class:`NotImplementedError`.
        """
        raise NotImplementedError(
            f"{self.provider_name} does not support inpainting."
        )

    async def health_check(self) -> dict[str, Any]:
        """Return provider health status. Override to add custom checks."""
        return {
            "provider": self.provider_name,
            "status": "available",
            "models": self.supported_models,
        }
