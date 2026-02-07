"""OpenAI DALL-E provider implementation."""

from __future__ import annotations

import base64
import time
from typing import Any

import httpx
import structlog

from image_generation.config import Settings
from image_generation.providers.base import (
    GenerationResult,
    GenerationStatus,
    ImageProvider,
    Img2ImgInput,
    InpaintInput,
)

logger = structlog.get_logger(__name__)

# DALL-E 3 only supports specific sizes.
_DALLE3_SIZES: dict[str, str] = {
    "1024x1024": "1024x1024",
    "1024x1792": "1024x1792",
    "1792x1024": "1792x1024",
}

_DALLE2_SIZES: dict[str, str] = {
    "256x256": "256x256",
    "512x512": "512x512",
    "1024x1024": "1024x1024",
}


def _snap_size(width: int, height: int, model: str) -> str:
    """Map arbitrary dimensions to the nearest supported DALL-E size."""
    target = f"{width}x{height}"
    sizes = _DALLE3_SIZES if "dall-e-3" in model else _DALLE2_SIZES
    if target in sizes:
        return target

    # Determine orientation and return best match.
    if width > height:
        return "1792x1024" if "dall-e-3" in model else "1024x1024"
    elif height > width:
        return "1024x1792" if "dall-e-3" in model else "1024x1024"
    return "1024x1024"


class OpenAIProvider(ImageProvider):
    """Image generation via OpenAI DALL-E 2 / 3 API.

    DALL-E 3 supports text-to-image only. DALL-E 2 additionally supports
    image-to-image variations and inpainting via the edits endpoint.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_key = settings.openai_api_key
        self._default_model = settings.openai_model

        if not self._api_key:
            logger.warning("openai_provider.no_api_key", hint="Set IMG_GEN_OPENAI_API_KEY")

    # -- ABC properties -------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "openai"

    @property
    def supported_models(self) -> list[str]:
        return ["dall-e-3", "dall-e-2"]

    # -- Helpers ---------------------------------------------------------------

    def _client(self) -> Any:
        """Lazy-import and instantiate the async OpenAI client."""
        from openai import AsyncOpenAI

        return AsyncOpenAI(api_key=self._api_key)

    # -- generate --------------------------------------------------------------

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
        model = model or self._default_model
        size_str = _snap_size(width, height, model)
        actual_w, actual_h = (int(d) for d in size_str.split("x"))

        logger.info(
            "openai_provider.generate",
            model=model,
            size=size_str,
            prompt_length=len(prompt),
        )

        start = time.monotonic()
        try:
            client = self._client()

            # Build request kwargs.
            req: dict[str, Any] = {
                "model": model,
                "prompt": prompt,
                "n": 1,
                "size": size_str,
                "response_format": "b64_json",
            }
            if "dall-e-3" in model:
                req["quality"] = kwargs.get("quality", "standard")
                req["style"] = kwargs.get("style", "vivid")

            response = await client.images.generate(**req)
            elapsed = time.monotonic() - start

            image_data = response.data[0]
            image_bytes = base64.b64decode(image_data.b64_json) if image_data.b64_json else None
            revised_prompt = getattr(image_data, "revised_prompt", None)

            result = GenerationResult(
                provider=self.provider_name,
                model=model,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=actual_w,
                height=actual_h,
                status=GenerationStatus.COMPLETED,
                image_bytes=image_bytes,
                image_url=image_data.url,
                image_format="png",
                duration_seconds=round(elapsed, 3),
                metadata={
                    "revised_prompt": revised_prompt,
                    "quality": req.get("quality"),
                    "style": req.get("style"),
                },
            )
            logger.info("openai_provider.generate.success", image_id=result.image_id, elapsed=elapsed)
            return result

        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error("openai_provider.generate.error", error=str(exc), elapsed=elapsed)
            return GenerationResult(
                provider=self.provider_name,
                model=model,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=actual_w,
                height=actual_h,
                status=GenerationStatus.FAILED,
                duration_seconds=round(elapsed, 3),
                error=str(exc),
            )

    # -- img2img (DALL-E 2 only) -----------------------------------------------

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
        """DALL-E 2 image variations (prompt is ignored by the API for variations)."""
        model = model or "dall-e-2"
        if "dall-e-3" in model:
            raise NotImplementedError("DALL-E 3 does not support image-to-image generation.")

        size_str = _snap_size(width or 1024, height or 1024, model)
        actual_w, actual_h = (int(d) for d in size_str.split("x"))

        start = time.monotonic()
        try:
            client = self._client()
            response = await client.images.create_variation(
                image=image.image_bytes,
                n=1,
                size=size_str,
                response_format="b64_json",
            )
            elapsed = time.monotonic() - start

            image_data = response.data[0]
            image_bytes_out = base64.b64decode(image_data.b64_json) if image_data.b64_json else None

            return GenerationResult(
                provider=self.provider_name,
                model=model,
                prompt=prompt,
                width=actual_w,
                height=actual_h,
                status=GenerationStatus.COMPLETED,
                image_bytes=image_bytes_out,
                image_url=image_data.url,
                image_format="png",
                duration_seconds=round(elapsed, 3),
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error("openai_provider.img2img.error", error=str(exc))
            return GenerationResult(
                provider=self.provider_name,
                model=model,
                prompt=prompt,
                width=actual_w,
                height=actual_h,
                status=GenerationStatus.FAILED,
                duration_seconds=round(elapsed, 3),
                error=str(exc),
            )

    # -- inpaint (DALL-E 2 only) -----------------------------------------------

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
        """DALL-E 2 inpainting via the edits endpoint."""
        model = model or "dall-e-2"
        if "dall-e-3" in model:
            raise NotImplementedError("DALL-E 3 does not support inpainting.")

        size_str = _snap_size(width or 1024, height or 1024, model)
        actual_w, actual_h = (int(d) for d in size_str.split("x"))

        start = time.monotonic()
        try:
            client = self._client()
            response = await client.images.edit(
                image=image.image_bytes,
                mask=image.mask_bytes,
                prompt=prompt,
                n=1,
                size=size_str,
                response_format="b64_json",
            )
            elapsed = time.monotonic() - start

            image_data = response.data[0]
            image_bytes_out = base64.b64decode(image_data.b64_json) if image_data.b64_json else None

            return GenerationResult(
                provider=self.provider_name,
                model=model,
                prompt=prompt,
                width=actual_w,
                height=actual_h,
                status=GenerationStatus.COMPLETED,
                image_bytes=image_bytes_out,
                image_url=image_data.url,
                image_format="png",
                duration_seconds=round(elapsed, 3),
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error("openai_provider.inpaint.error", error=str(exc))
            return GenerationResult(
                provider=self.provider_name,
                model=model,
                prompt=prompt,
                width=actual_w,
                height=actual_h,
                status=GenerationStatus.FAILED,
                duration_seconds=round(elapsed, 3),
                error=str(exc),
            )

    # -- health ----------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        status = "available" if self._api_key else "unavailable (no API key)"
        return {
            "provider": self.provider_name,
            "status": status,
            "models": self.supported_models,
            "default_model": self._default_model,
        }
