"""Replicate provider for Stable Diffusion 3.5 and FLUX models."""

from __future__ import annotations

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

# Well-known Replicate model identifiers.
REPLICATE_MODELS: dict[str, str] = {
    "sd-3.5-large": "stability-ai/stable-diffusion-3.5-large",
    "sd-3.5-medium": "stability-ai/stable-diffusion-3.5-medium",
    "flux-1.1-pro": "black-forest-labs/flux-1.1-pro",
    "flux-dev": "black-forest-labs/flux-dev",
    "flux-schnell": "black-forest-labs/flux-schnell",
}


class ReplicateProvider(ImageProvider):
    """Image generation via the Replicate API.

    Supports Stable Diffusion 3.5 (all sizes) and FLUX models.  Most
    models expose ``prompt``, ``negative_prompt``, ``width``, ``height``,
    ``num_inference_steps``, ``guidance_scale``, and ``seed``.
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._api_token = settings.replicate_api_token
        self._default_model = settings.replicate_model

        if not self._api_token:
            logger.warning(
                "replicate_provider.no_api_token",
                hint="Set IMG_GEN_REPLICATE_API_TOKEN",
            )

    # -- ABC properties --------------------------------------------------------

    @property
    def provider_name(self) -> str:
        return "replicate"

    @property
    def supported_models(self) -> list[str]:
        return list(REPLICATE_MODELS.values())

    # -- Helpers ---------------------------------------------------------------

    def _client(self) -> Any:
        """Lazy-import and instantiate the Replicate client."""
        import replicate

        return replicate.Client(api_token=self._api_token)

    def _resolve_model(self, model: str | None) -> str:
        """Resolve a short alias or return the full model ref."""
        if model is None:
            return self._default_model
        return REPLICATE_MODELS.get(model, model)

    @staticmethod
    def _build_input(
        prompt: str,
        *,
        negative_prompt: str = "",
        width: int = 1024,
        height: int = 1024,
        steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """Build the input dict for Replicate's prediction API."""
        inp: dict[str, Any] = {
            "prompt": prompt,
            "width": width,
            "height": height,
        }
        if negative_prompt:
            inp["negative_prompt"] = negative_prompt
        if steps is not None:
            inp["num_inference_steps"] = steps
        if guidance_scale is not None:
            inp["guidance_scale"] = guidance_scale
        if seed is not None:
            inp["seed"] = seed
        inp.update(extra)
        return inp

    @staticmethod
    async def _download_image(url: str) -> bytes:
        """Download image bytes from a Replicate output URL."""
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.get(url)
            resp.raise_for_status()
            return resp.content

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
        model_ref = self._resolve_model(model)
        logger.info(
            "replicate_provider.generate",
            model=model_ref,
            size=f"{width}x{height}",
            prompt_length=len(prompt),
        )

        inp = self._build_input(
            prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            **kwargs,
        )

        start = time.monotonic()
        try:
            client = self._client()

            # replicate.run is synchronous but returns quickly with polling.
            # We run it in the default executor to avoid blocking the loop.
            import asyncio

            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(
                None,
                lambda: client.run(model_ref, input=inp),
            )
            elapsed = time.monotonic() - start

            # Output can be a list of URLs or a single FileOutput / URL string.
            image_url: str | None = None
            image_bytes: bytes | None = None

            if isinstance(output, list) and output:
                image_url = str(output[0])
            elif isinstance(output, str):
                image_url = output
            else:
                # FileOutput or iterator — convert to string URL.
                image_url = str(output)

            if image_url:
                try:
                    image_bytes = await self._download_image(image_url)
                except Exception as dl_exc:
                    logger.warning("replicate_provider.download_failed", error=str(dl_exc))

            result = GenerationResult(
                provider=self.provider_name,
                model=model_ref,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=seed,
                status=GenerationStatus.COMPLETED,
                image_url=image_url,
                image_bytes=image_bytes,
                image_format="png",
                duration_seconds=round(elapsed, 3),
                metadata={"replicate_input": inp},
            )
            logger.info(
                "replicate_provider.generate.success",
                image_id=result.image_id,
                elapsed=elapsed,
            )
            return result

        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error("replicate_provider.generate.error", error=str(exc), elapsed=elapsed)
            return GenerationResult(
                provider=self.provider_name,
                model=model_ref,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                status=GenerationStatus.FAILED,
                duration_seconds=round(elapsed, 3),
                error=str(exc),
            )

    # -- img2img ---------------------------------------------------------------

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
        """Image-to-image via Replicate.

        Encodes the input image as a data URI and passes ``prompt_strength``
        (inverse of ``strength``) to the model.
        """
        import base64

        model_ref = self._resolve_model(model)
        data_uri = (
            f"data:image/{image.image_format};base64,"
            + base64.b64encode(image.image_bytes).decode()
        )

        inp = self._build_input(
            prompt,
            negative_prompt=negative_prompt,
            width=width or 1024,
            height=height or 1024,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            image=data_uri,
            prompt_strength=1.0 - strength,
            **kwargs,
        )

        start = time.monotonic()
        try:
            import asyncio

            client = self._client()
            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(
                None,
                lambda: client.run(model_ref, input=inp),
            )
            elapsed = time.monotonic() - start

            image_url = str(output[0]) if isinstance(output, list) and output else str(output)
            image_bytes_out: bytes | None = None
            try:
                image_bytes_out = await self._download_image(image_url)
            except Exception:
                pass

            return GenerationResult(
                provider=self.provider_name,
                model=model_ref,
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width or 1024,
                height=height or 1024,
                status=GenerationStatus.COMPLETED,
                image_url=image_url,
                image_bytes=image_bytes_out,
                image_format="png",
                duration_seconds=round(elapsed, 3),
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error("replicate_provider.img2img.error", error=str(exc))
            return GenerationResult(
                provider=self.provider_name,
                model=model_ref,
                prompt=prompt,
                width=width or 1024,
                height=height or 1024,
                status=GenerationStatus.FAILED,
                duration_seconds=round(elapsed, 3),
                error=str(exc),
            )

    # -- inpaint ---------------------------------------------------------------

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
        """Inpainting via Replicate (SD 3.5 models support inpaint inputs)."""
        import base64

        model_ref = self._resolve_model(model)
        img_uri = (
            f"data:image/{image.image_format};base64,"
            + base64.b64encode(image.image_bytes).decode()
        )
        mask_uri = (
            f"data:image/{image.image_format};base64,"
            + base64.b64encode(image.mask_bytes).decode()
        )

        inp = self._build_input(
            prompt,
            negative_prompt=negative_prompt,
            width=width or 1024,
            height=height or 1024,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            image=img_uri,
            mask=mask_uri,
            **kwargs,
        )

        start = time.monotonic()
        try:
            import asyncio

            client = self._client()
            loop = asyncio.get_running_loop()
            output = await loop.run_in_executor(
                None,
                lambda: client.run(model_ref, input=inp),
            )
            elapsed = time.monotonic() - start

            image_url = str(output[0]) if isinstance(output, list) and output else str(output)
            image_bytes_out: bytes | None = None
            try:
                image_bytes_out = await self._download_image(image_url)
            except Exception:
                pass

            return GenerationResult(
                provider=self.provider_name,
                model=model_ref,
                prompt=prompt,
                width=width or 1024,
                height=height or 1024,
                status=GenerationStatus.COMPLETED,
                image_url=image_url,
                image_bytes=image_bytes_out,
                image_format="png",
                duration_seconds=round(elapsed, 3),
            )
        except Exception as exc:
            elapsed = time.monotonic() - start
            logger.error("replicate_provider.inpaint.error", error=str(exc))
            return GenerationResult(
                provider=self.provider_name,
                model=model_ref,
                prompt=prompt,
                width=width or 1024,
                height=height or 1024,
                status=GenerationStatus.FAILED,
                duration_seconds=round(elapsed, 3),
                error=str(exc),
            )

    # -- health ----------------------------------------------------------------

    async def health_check(self) -> dict[str, Any]:
        status = "available" if self._api_token else "unavailable (no API token)"
        return {
            "provider": self.provider_name,
            "status": status,
            "models": self.supported_models,
            "default_model": self._default_model,
        }
