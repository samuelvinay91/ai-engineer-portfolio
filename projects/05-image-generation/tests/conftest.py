"""Shared fixtures for the Image Generation Service test suite."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest
import pytest_asyncio
from httpx import ASGITransport, AsyncClient

from image_generation.config import ImageSize, ProviderName, Settings
from image_generation.pipeline import ImagePipeline, PromptEnhancer
from image_generation.providers.base import (
    GenerationResult,
    GenerationStatus,
    ImageProvider,
    Img2ImgInput,
    InpaintInput,
)
from image_generation.storage import ImageStorageService


# ---------------------------------------------------------------------------
# Settings fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def settings(tmp_path: Path) -> Settings:
    """Provide test settings with a temporary output directory."""
    return Settings(
        _env_file=None,
        default_provider=ProviderName.OPENAI,
        openai_api_key="test-key-not-real",
        replicate_api_token="test-token-not-real",
        output_dir=tmp_path / "generated_images",
        storage_backend="local",
        enable_prompt_enhancement=False,
        log_level="DEBUG",
        environment="test",
    )


# ---------------------------------------------------------------------------
# Fake / mock provider
# ---------------------------------------------------------------------------


# A tiny 1x1 red PNG for tests.
_FAKE_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
    b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
    b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
    b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
)


class FakeProvider(ImageProvider):
    """Deterministic provider for unit tests."""

    def __init__(self, *, fail: bool = False) -> None:
        self._fail = fail

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def supported_models(self) -> list[str]:
        return ["fake-model-v1"]

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
        if self._fail:
            return GenerationResult(
                provider=self.provider_name,
                model=model or "fake-model-v1",
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                seed=seed,
                status=GenerationStatus.FAILED,
                error="Simulated failure",
            )
        return GenerationResult(
            provider=self.provider_name,
            model=model or "fake-model-v1",
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            status=GenerationStatus.COMPLETED,
            image_bytes=_FAKE_PNG,
            image_format="png",
            duration_seconds=0.123,
        )

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
        return await self.generate(
            prompt,
            negative_prompt=negative_prompt,
            width=width or 1024,
            height=height or 1024,
            steps=steps,
            guidance_scale=guidance_scale,
            seed=seed,
            model=model,
        )


@pytest.fixture()
def fake_provider() -> FakeProvider:
    return FakeProvider()


@pytest.fixture()
def failing_provider() -> FakeProvider:
    return FakeProvider(fail=True)


# ---------------------------------------------------------------------------
# Pipeline fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def pipeline(settings: Settings, fake_provider: FakeProvider) -> ImagePipeline:
    return ImagePipeline(
        settings=settings,
        provider=fake_provider,
        enhancer=PromptEnhancer(settings),
    )


# ---------------------------------------------------------------------------
# Storage fixture
# ---------------------------------------------------------------------------


@pytest.fixture()
def storage_service(settings: Settings) -> ImageStorageService:
    return ImageStorageService(settings=settings)


# ---------------------------------------------------------------------------
# FastAPI test client
# ---------------------------------------------------------------------------


@pytest_asyncio.fixture()
async def client(settings: Settings, fake_provider: FakeProvider) -> AsyncGenerator[AsyncClient, None]:
    """Provide an async HTTPX client wired to the FastAPI app with a fake provider."""
    from image_generation.api import app, state

    # Inject test dependencies.
    state.settings = settings
    state.pipeline = ImagePipeline(
        settings=settings,
        provider=fake_provider,
        enhancer=PromptEnhancer(settings),
    )
    state.storage = ImageStorageService(settings=settings)

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac
