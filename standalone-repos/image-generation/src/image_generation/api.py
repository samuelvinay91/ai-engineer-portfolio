"""FastAPI application for the Image Generation Service.

Endpoints
---------
POST /api/v1/generate            -- generate an image from a text prompt
POST /api/v1/generate/batch      -- batch generation with multiple seeds
POST /api/v1/img2img             -- image-to-image transformation
POST /api/v1/enhance-prompt      -- enhance a prompt via LLM
GET  /api/v1/gallery             -- list generated images
GET  /api/v1/gallery/{image_id}  -- get metadata for a specific image
GET  /api/v1/concepts/{model_type} -- educational architecture explanation
GET  /api/v1/providers           -- list available providers
GET  /health                     -- health check
"""

from __future__ import annotations

import base64
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Query, UploadFile, File, Form
from pydantic import BaseModel, Field

from image_generation import __version__
from image_generation.config import ImageSize, ProviderName, Settings, get_settings
from image_generation.models.concepts import ModelType, get_explanation, list_model_types
from image_generation.pipeline import ImagePipeline, StylePreset
from image_generation.providers import (
    GenerationStatus,
    Img2ImgInput,
    get_provider,
    list_providers,
)
from image_generation.storage import ImageStorageService

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------


class GenerateRequest(BaseModel):
    """Request body for text-to-image generation."""

    prompt: str = Field(..., min_length=1, max_length=4000, description="Text prompt describing the desired image")
    negative_prompt: str = Field("", max_length=2000, description="Concepts to avoid in the output")
    width: int | None = Field(None, ge=256, le=2048, description="Output width in pixels")
    height: int | None = Field(None, ge=256, le=2048, description="Output height in pixels")
    steps: int | None = Field(None, ge=1, le=150, description="Number of generation steps")
    guidance_scale: float | None = Field(None, ge=1.0, le=30.0, description="Guidance strength")
    seed: int | None = Field(None, description="Random seed for reproducibility")
    style: str = Field("default", description="Style preset name")
    enhance_prompt: bool | None = Field(None, description="Whether to enhance the prompt via LLM")
    provider: str | None = Field(None, description="Override default provider (openai/replicate)")
    model: str | None = Field(None, description="Override default model for the provider")


class BatchGenerateRequest(BaseModel):
    """Request body for batch image generation."""

    prompt: str = Field(..., min_length=1, max_length=4000)
    negative_prompt: str = ""
    count: int = Field(4, ge=1, le=16)
    seeds: list[int] | None = None
    width: int | None = Field(None, ge=256, le=2048)
    height: int | None = Field(None, ge=256, le=2048)
    steps: int | None = Field(None, ge=1, le=150)
    guidance_scale: float | None = Field(None, ge=1.0, le=30.0)
    style: str = "default"
    enhance_prompt: bool | None = None
    provider: str | None = None
    model: str | None = None


class EnhancePromptRequest(BaseModel):
    """Request body for prompt enhancement."""

    prompt: str = Field(..., min_length=1, max_length=4000)


class GenerateResponse(BaseModel):
    """Response for a single image generation."""

    image_id: str
    status: str
    provider: str
    model: str
    prompt: str
    width: int
    height: int
    seed: int | None = None
    duration_seconds: float | None = None
    image_url: str | None = None
    image_base64: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class GalleryItemResponse(BaseModel):
    """Single gallery item."""

    image_id: str
    filename: str
    prompt: str
    provider: str
    model: str
    width: int
    height: int
    image_format: str
    seed: int | None = None
    style: str | None = None
    duration_seconds: float | None = None
    created_at: str


class GalleryListResponse(BaseModel):
    """Paginated gallery list."""

    total: int
    items: list[GalleryItemResponse]


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class AppState:
    """Lazily initialised shared application state."""

    def __init__(self) -> None:
        self._settings: Settings | None = None
        self._pipeline: ImagePipeline | None = None
        self._storage: ImageStorageService | None = None

    @property
    def settings(self) -> Settings:
        if self._settings is None:
            self._settings = get_settings()
        return self._settings

    @settings.setter
    def settings(self, value: Settings) -> None:
        self._settings = value

    @property
    def pipeline(self) -> ImagePipeline:
        if self._pipeline is None:
            self._pipeline = ImagePipeline(settings=self.settings)
        return self._pipeline

    @pipeline.setter
    def pipeline(self, value: ImagePipeline) -> None:
        self._pipeline = value

    @property
    def storage(self) -> ImageStorageService:
        if self._storage is None:
            self._storage = ImageStorageService(settings=self.settings)
        return self._storage

    @storage.setter
    def storage(self, value: ImageStorageService) -> None:
        self._storage = value


state = AppState()


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------


app = FastAPI(
    title="Image Generation Service",
    description=(
        "Cloud-native image generation service with multi-provider support "
        "(OpenAI DALL-E, Replicate Stable Diffusion / FLUX), style presets, "
        "prompt enhancement, and educational content on generative architectures."
    ),
    version=__version__,
    docs_url="/docs",
    redoc_url="/redoc",
)


# ---------------------------------------------------------------------------
# Health check
# ---------------------------------------------------------------------------


@app.get("/health", tags=["health"])
async def health_check() -> dict[str, str]:
    return {
        "status": "healthy",
        "service": "image-generation",
        "version": __version__,
    }


# ---------------------------------------------------------------------------
# Generate
# ---------------------------------------------------------------------------


@app.post("/api/v1/generate", response_model=GenerateResponse, tags=["generation"])
async def generate_image(request: GenerateRequest) -> GenerateResponse:
    """Generate a single image from a text prompt."""
    logger.info(
        "api.generate",
        prompt_length=len(request.prompt),
        style=request.style,
        provider=request.provider,
    )

    # Build a pipeline with the requested provider if overridden.
    pipeline = state.pipeline
    if request.provider:
        provider = get_provider(request.provider, state.settings)
        pipeline = ImagePipeline(
            settings=state.settings,
            provider=provider,
        )

    result = await pipeline.generate(
        prompt=request.prompt,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        steps=request.steps,
        guidance_scale=request.guidance_scale,
        seed=request.seed,
        style=request.style,
        enhance_prompt=request.enhance_prompt,
        model=request.model,
    )

    if result.status == GenerationStatus.FAILED:
        raise HTTPException(status_code=502, detail=result.error or "Image generation failed")

    # Store the image.
    await state.storage.store(result)

    # Encode bytes as base64 for the response.
    image_b64: str | None = None
    if result.image_bytes:
        image_b64 = base64.b64encode(result.image_bytes).decode()

    return GenerateResponse(
        image_id=result.image_id,
        status=result.status.value,
        provider=result.provider,
        model=result.model,
        prompt=result.prompt,
        width=result.width,
        height=result.height,
        seed=result.seed,
        duration_seconds=result.duration_seconds,
        image_url=result.image_url,
        image_base64=image_b64,
        metadata=result.metadata,
    )


# ---------------------------------------------------------------------------
# Batch generate
# ---------------------------------------------------------------------------


@app.post("/api/v1/generate/batch", tags=["generation"])
async def generate_batch(request: BatchGenerateRequest) -> dict[str, Any]:
    """Generate multiple images with different seeds."""
    logger.info("api.generate_batch", prompt_length=len(request.prompt), count=request.count)

    pipeline = state.pipeline
    if request.provider:
        provider = get_provider(request.provider, state.settings)
        pipeline = ImagePipeline(settings=state.settings, provider=provider)

    batch = await pipeline.generate_batch(
        prompt=request.prompt,
        count=request.count,
        seeds=request.seeds,
        negative_prompt=request.negative_prompt,
        width=request.width,
        height=request.height,
        steps=request.steps,
        guidance_scale=request.guidance_scale,
        style=request.style,
        enhance_prompt=request.enhance_prompt,
        model=request.model,
    )

    # Store all completed images.
    for r in batch.results:
        if r.status == GenerationStatus.COMPLETED:
            await state.storage.store(r)

    return batch.to_dict()


# ---------------------------------------------------------------------------
# Image-to-image
# ---------------------------------------------------------------------------


@app.post("/api/v1/img2img", response_model=GenerateResponse, tags=["generation"])
async def img2img(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    strength: float = Form(0.75),
    negative_prompt: str = Form(""),
    provider: str | None = Form(None),
    model: str | None = Form(None),
) -> GenerateResponse:
    """Transform an existing image guided by a text prompt."""
    image_bytes = await image.read()

    target_provider = get_provider(provider or state.settings.default_provider.value, state.settings)

    img_input = Img2ImgInput(image_bytes=image_bytes)
    result = await target_provider.img2img(
        image=img_input,
        prompt=prompt,
        strength=strength,
        negative_prompt=negative_prompt,
        model=model,
    )

    if result.status == GenerationStatus.FAILED:
        raise HTTPException(status_code=502, detail=result.error or "img2img generation failed")

    await state.storage.store(result)

    image_b64: str | None = None
    if result.image_bytes:
        image_b64 = base64.b64encode(result.image_bytes).decode()

    return GenerateResponse(
        image_id=result.image_id,
        status=result.status.value,
        provider=result.provider,
        model=result.model,
        prompt=result.prompt,
        width=result.width,
        height=result.height,
        seed=result.seed,
        duration_seconds=result.duration_seconds,
        image_url=result.image_url,
        image_base64=image_b64,
        metadata=result.metadata,
    )


# ---------------------------------------------------------------------------
# Prompt enhancement
# ---------------------------------------------------------------------------


@app.post("/api/v1/enhance-prompt", tags=["tools"])
async def enhance_prompt(request: EnhancePromptRequest) -> dict[str, str]:
    """Enhance a text prompt using an LLM for improved generation quality."""
    return await state.pipeline.enhance_prompt(request.prompt)


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------


@app.get("/api/v1/gallery", response_model=GalleryListResponse, tags=["gallery"])
async def list_gallery(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    provider: str | None = Query(None),
    style: str | None = Query(None),
) -> GalleryListResponse:
    """List generated images in the gallery."""
    items = state.storage.gallery.list(
        limit=limit,
        offset=offset,
        provider=provider,
        style=style,
    )
    return GalleryListResponse(
        total=state.storage.gallery.count,
        items=[
            GalleryItemResponse(
                image_id=m.image_id,
                filename=m.filename,
                prompt=m.prompt,
                provider=m.provider,
                model=m.model,
                width=m.width,
                height=m.height,
                image_format=m.image_format,
                seed=m.seed,
                style=m.style,
                duration_seconds=m.duration_seconds,
                created_at=m.created_at.isoformat(),
            )
            for m in items
        ],
    )


@app.get("/api/v1/gallery/{image_id}", tags=["gallery"])
async def get_gallery_image(image_id: str) -> dict[str, Any]:
    """Get metadata and base64-encoded image for a specific gallery entry."""
    meta = state.storage.gallery.get(image_id)
    if meta is None:
        raise HTTPException(status_code=404, detail=f"Image '{image_id}' not found")

    result: dict[str, Any] = meta.to_dict()

    # Optionally include raw bytes as base64.
    image_bytes = await state.storage.load_image(image_id)
    if image_bytes:
        result["image_base64"] = base64.b64encode(image_bytes).decode()

    return result


# ---------------------------------------------------------------------------
# Concepts (educational)
# ---------------------------------------------------------------------------


@app.get("/api/v1/concepts", tags=["concepts"])
async def list_concepts() -> list[dict[str, str]]:
    """List all available generative model architecture explanations."""
    return list_model_types()


@app.get("/api/v1/concepts/{model_type}", tags=["concepts"])
async def get_concept(model_type: str) -> dict[str, Any]:
    """Get a detailed educational explanation of a generative model architecture.

    Valid values: ``vae``, ``gan``, ``diffusion``, ``autoregressive``.
    """
    try:
        explanation = get_explanation(model_type)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return explanation.to_dict()


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


@app.get("/api/v1/providers", tags=["providers"])
async def get_providers() -> list[dict[str, Any]]:
    """List all registered image generation providers and their status."""
    return await list_providers(state.settings)


# ---------------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------------


@app.get("/api/v1/styles", tags=["tools"])
async def list_styles() -> list[dict[str, str]]:
    """List available style presets."""
    return [{"name": s.value, "description": s.value.replace("_", " ").title()} for s in StylePreset]
