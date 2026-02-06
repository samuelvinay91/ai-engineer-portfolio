"""Image generation pipeline with prompt enhancement and batch support.

The pipeline orchestrates:
  1. **Prompt enhancement** -- optionally rewrite user prompts via an LLM for
     improved generation quality.
  2. **Provider dispatch** -- forward the (enhanced) prompt to the configured
     image generation provider.
  3. **Post-processing** -- persist results via the storage service and
     return structured output.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

from image_generation.config import Settings, get_settings
from image_generation.providers import (
    GenerationResult,
    GenerationStatus,
    ImageProvider,
    get_provider,
)

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Prompt enhancement
# ---------------------------------------------------------------------------


class PromptEnhancer:
    """Uses an LLM to expand terse user prompts into rich image descriptions.

    The enhancer rewrites a short prompt into a detailed description that
    typically yields better results from diffusion models by including
    composition, lighting, style, and quality keywords.
    """

    SYSTEM_PROMPT = (
        "You are an expert prompt engineer for AI image generation models. "
        "Given a short image description, rewrite it as a detailed, vivid prompt "
        "that will produce a high-quality image. Include details about composition, "
        "lighting, color palette, style, mood, and camera angle where appropriate. "
        "Keep the enhanced prompt under 200 words. Do NOT add quotation marks. "
        "Return ONLY the enhanced prompt text, nothing else."
    )

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    async def enhance(self, prompt: str) -> str:
        """Return an LLM-enhanced version of *prompt*.

        Falls back to the original prompt if the API call fails.
        """
        if not self._settings.openai_api_key:
            logger.debug("prompt_enhancer.skipped", reason="no API key")
            return prompt

        try:
            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=self._settings.openai_api_key)
            response = await client.chat.completions.create(
                model=self._settings.prompt_enhance_model,
                messages=[
                    {"role": "system", "content": self.SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.7,
                max_tokens=300,
            )
            enhanced = response.choices[0].message.content or prompt
            enhanced = enhanced.strip().strip('"').strip("'")
            logger.info(
                "prompt_enhancer.success",
                original_length=len(prompt),
                enhanced_length=len(enhanced),
            )
            return enhanced
        except Exception as exc:
            logger.warning("prompt_enhancer.failed", error=str(exc))
            return prompt


# ---------------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------------


class StylePreset(str, Enum):
    """Named parameter presets for common generation styles."""

    PHOTOREALISTIC = "photorealistic"
    ARTISTIC = "artistic"
    ANIME = "anime"
    CINEMATIC = "cinematic"
    DIGITAL_ART = "digital_art"
    WATERCOLOR = "watercolor"
    OIL_PAINTING = "oil_painting"
    PIXEL_ART = "pixel_art"
    SKETCH = "sketch"
    DEFAULT = "default"


@dataclass(frozen=True)
class PresetConfig:
    """Generation parameters associated with a style preset."""

    suffix: str = ""
    negative_prompt: str = ""
    guidance_scale: float = 7.5
    steps: int = 30


_PRESET_CONFIGS: dict[StylePreset, PresetConfig] = {
    StylePreset.PHOTOREALISTIC: PresetConfig(
        suffix=(
            ", ultra-realistic photography, 8K UHD, sharp focus, natural lighting, "
            "shot on Canon EOS R5, professional color grading"
        ),
        negative_prompt="cartoon, illustration, painting, drawing, anime, blurry, low quality",
        guidance_scale=8.0,
        steps=35,
    ),
    StylePreset.ARTISTIC: PresetConfig(
        suffix=", artistic masterpiece, expressive brushstrokes, vibrant palette, gallery quality",
        negative_prompt="photorealistic, photograph, blurry, low quality",
        guidance_scale=9.0,
        steps=35,
    ),
    StylePreset.ANIME: PresetConfig(
        suffix=(
            ", anime style, studio quality, detailed cel shading, vibrant colors, "
            "clean linework, trending on pixiv"
        ),
        negative_prompt=(
            "photorealistic, photograph, western cartoon, 3D render, blurry, "
            "bad anatomy, low quality"
        ),
        guidance_scale=8.5,
        steps=30,
    ),
    StylePreset.CINEMATIC: PresetConfig(
        suffix=(
            ", cinematic shot, dramatic lighting, anamorphic lens flare, "
            "depth of field, color graded, 35mm film"
        ),
        negative_prompt="cartoon, anime, flat, low quality, blurry",
        guidance_scale=8.0,
        steps=35,
    ),
    StylePreset.DIGITAL_ART: PresetConfig(
        suffix=", digital art, detailed illustration, trending on ArtStation, concept art",
        negative_prompt="photograph, photorealistic, blurry, low quality",
        guidance_scale=7.5,
        steps=30,
    ),
    StylePreset.WATERCOLOR: PresetConfig(
        suffix=(
            ", watercolor painting, soft washes, delicate brushwork, "
            "paper texture, transparent layers, wet-on-wet technique"
        ),
        negative_prompt="photograph, digital, sharp edges, low quality",
        guidance_scale=7.0,
        steps=30,
    ),
    StylePreset.OIL_PAINTING: PresetConfig(
        suffix=(
            ", oil painting on canvas, impasto technique, rich textures, "
            "classical composition, warm tones, museum quality"
        ),
        negative_prompt="photograph, digital, anime, cartoon, low quality",
        guidance_scale=7.5,
        steps=35,
    ),
    StylePreset.PIXEL_ART: PresetConfig(
        suffix=", pixel art, 16-bit, retro game style, clean pixels, limited color palette",
        negative_prompt="photorealistic, smooth, blurry, 3D, high resolution photo",
        guidance_scale=8.0,
        steps=25,
    ),
    StylePreset.SKETCH: PresetConfig(
        suffix=(
            ", pencil sketch, detailed line drawing, cross-hatching, "
            "graphite on paper, professional illustration"
        ),
        negative_prompt="color, photograph, painting, digital, blurry",
        guidance_scale=7.0,
        steps=25,
    ),
    StylePreset.DEFAULT: PresetConfig(),
}


def get_preset(style: StylePreset | str) -> PresetConfig:
    """Retrieve the :class:`PresetConfig` for a given style name."""
    if isinstance(style, str):
        try:
            style = StylePreset(style.lower())
        except ValueError:
            logger.warning("pipeline.unknown_preset", style=style)
            return _PRESET_CONFIGS[StylePreset.DEFAULT]
    return _PRESET_CONFIGS.get(style, _PRESET_CONFIGS[StylePreset.DEFAULT])


# ---------------------------------------------------------------------------
# Batch generation result
# ---------------------------------------------------------------------------


@dataclass
class BatchResult:
    """Container for the results of a batch generation request."""

    results: list[GenerationResult] = field(default_factory=list)
    total_requested: int = 0
    total_completed: int = 0
    total_failed: int = 0
    total_duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_requested": self.total_requested,
            "total_completed": self.total_completed,
            "total_failed": self.total_failed,
            "total_duration_seconds": round(self.total_duration_seconds, 3),
            "results": [
                {
                    "image_id": r.image_id,
                    "status": r.status.value,
                    "seed": r.seed,
                    "image_url": r.image_url,
                    "duration_seconds": r.duration_seconds,
                    "error": r.error,
                }
                for r in self.results
            ],
        }


# ---------------------------------------------------------------------------
# ImagePipeline
# ---------------------------------------------------------------------------


class ImagePipeline:
    """High-level orchestrator for text-to-image generation.

    The pipeline wires together prompt enhancement, provider dispatch,
    style presets, and batch generation into a single coherent interface.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        provider: ImageProvider | None = None,
        enhancer: PromptEnhancer | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._provider = provider or get_provider(settings=self._settings)
        self._enhancer = enhancer or PromptEnhancer(self._settings)

    @property
    def provider(self) -> ImageProvider:
        return self._provider

    # -- Single generation -----------------------------------------------------

    async def generate(
        self,
        prompt: str,
        *,
        negative_prompt: str = "",
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        seed: int | None = None,
        style: StylePreset | str = StylePreset.DEFAULT,
        enhance_prompt: bool | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """Generate a single image.

        1. Optionally enhance the prompt via LLM.
        2. Apply style-preset suffixes and defaults.
        3. Dispatch to the configured provider.
        """
        start = time.monotonic()

        # Resolve style preset.
        preset = get_preset(style)

        # Enhance prompt if requested.
        should_enhance = (
            enhance_prompt
            if enhance_prompt is not None
            else self._settings.enable_prompt_enhancement
        )
        working_prompt = prompt
        if should_enhance:
            working_prompt = await self._enhancer.enhance(prompt)

        # Apply preset suffix and negative prompt.
        final_prompt = working_prompt + preset.suffix
        final_negative = negative_prompt or preset.negative_prompt

        # Resolve defaults.
        gen_width = width or self._settings.default_size.width
        gen_height = height or self._settings.default_size.height
        gen_steps = steps or preset.steps or self._settings.default_steps
        gen_guidance = guidance_scale if guidance_scale is not None else preset.guidance_scale

        logger.info(
            "pipeline.generate",
            provider=self._provider.provider_name,
            style=style if isinstance(style, str) else style.value,
            enhanced=should_enhance,
            size=f"{gen_width}x{gen_height}",
        )

        result = await self._provider.generate(
            prompt=final_prompt,
            negative_prompt=final_negative,
            width=gen_width,
            height=gen_height,
            steps=gen_steps,
            guidance_scale=gen_guidance,
            seed=seed,
            model=model,
            **kwargs,
        )

        # Attach the original and enhanced prompts to metadata.
        result.metadata["original_prompt"] = prompt
        result.metadata["enhanced_prompt"] = working_prompt
        result.metadata["style"] = style if isinstance(style, str) else style.value
        result.duration_seconds = round(time.monotonic() - start, 3)

        return result

    # -- Batch generation ------------------------------------------------------

    async def generate_batch(
        self,
        prompt: str,
        *,
        count: int = 4,
        seeds: list[int] | None = None,
        negative_prompt: str = "",
        width: int | None = None,
        height: int | None = None,
        steps: int | None = None,
        guidance_scale: float | None = None,
        style: StylePreset | str = StylePreset.DEFAULT,
        enhance_prompt: bool | None = None,
        model: str | None = None,
        **kwargs: Any,
    ) -> BatchResult:
        """Generate multiple images with different seeds.

        Images are generated concurrently (up to ``max_batch_size``).
        """
        count = min(count, self._settings.max_batch_size)

        if seeds is None:
            seeds = [random.randint(0, 2**32 - 1) for _ in range(count)]
        else:
            seeds = seeds[:count]
            while len(seeds) < count:
                seeds.append(random.randint(0, 2**32 - 1))

        logger.info("pipeline.generate_batch", count=count, seeds=seeds)
        start = time.monotonic()

        tasks = [
            self.generate(
                prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                steps=steps,
                guidance_scale=guidance_scale,
                seed=s,
                style=style,
                enhance_prompt=enhance_prompt,
                model=model,
                **kwargs,
            )
            for s in seeds
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        elapsed = time.monotonic() - start

        generation_results: list[GenerationResult] = []
        for idx, r in enumerate(results):
            if isinstance(r, Exception):
                logger.error("pipeline.batch_item_failed", index=idx, error=str(r))
                generation_results.append(
                    GenerationResult(
                        provider=self._provider.provider_name,
                        model=model or "",
                        prompt=prompt,
                        width=width or self._settings.default_size.width,
                        height=height or self._settings.default_size.height,
                        seed=seeds[idx],
                        status=GenerationStatus.FAILED,
                        error=str(r),
                    )
                )
            else:
                generation_results.append(r)

        completed = sum(1 for r in generation_results if r.status == GenerationStatus.COMPLETED)
        failed = sum(1 for r in generation_results if r.status == GenerationStatus.FAILED)

        return BatchResult(
            results=generation_results,
            total_requested=count,
            total_completed=completed,
            total_failed=failed,
            total_duration_seconds=round(elapsed, 3),
        )

    # -- Prompt enhancement (standalone) ---------------------------------------

    async def enhance_prompt(self, prompt: str) -> dict[str, str]:
        """Enhance a prompt without generating an image.

        Returns both the original and enhanced prompts.
        """
        enhanced = await self._enhancer.enhance(prompt)
        return {
            "original": prompt,
            "enhanced": enhanced,
        }
