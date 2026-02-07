"""Tests for the image generation pipeline."""

from __future__ import annotations

import pytest

from image_generation.config import Settings
from image_generation.models.concepts import (
    ModelType,
    get_explanation,
    list_model_types,
)
from image_generation.pipeline import (
    BatchResult,
    ImagePipeline,
    PresetConfig,
    PromptEnhancer,
    StylePreset,
    get_preset,
)
from image_generation.providers.base import GenerationResult, GenerationStatus
from image_generation.storage import (
    Gallery,
    ImageMetadata,
    ImageStorageService,
    LocalStorageBackend,
)

from .conftest import FakeProvider


# ---------------------------------------------------------------------------
# Concepts / explainers
# ---------------------------------------------------------------------------


class TestConcepts:
    """Tests for the educational model architecture explainers."""

    @pytest.mark.parametrize("model_type", list(ModelType))
    def test_get_explanation_returns_all_fields(self, model_type: ModelType) -> None:
        explanation = get_explanation(model_type)
        assert explanation.model_type == model_type.value
        assert explanation.title
        assert explanation.summary
        assert explanation.architecture
        assert explanation.training
        assert explanation.strengths
        assert explanation.weaknesses
        assert explanation.key_models
        assert explanation.ascii_diagram
        assert explanation.data_preparation

    @pytest.mark.parametrize("model_type", list(ModelType))
    def test_explanation_to_dict_is_serialisable(self, model_type: ModelType) -> None:
        data = get_explanation(model_type).to_dict()
        assert isinstance(data, dict)
        assert data["model_type"] == model_type.value

    def test_get_explanation_string_input(self) -> None:
        explanation = get_explanation("vae")
        assert explanation.model_type == "vae"

    def test_get_explanation_invalid_type(self) -> None:
        with pytest.raises(ValueError, match="Unknown model type"):
            get_explanation("transformers")

    def test_list_model_types(self) -> None:
        types = list_model_types()
        assert len(types) == len(ModelType)
        for item in types:
            assert "type" in item
            assert "title" in item
            assert "summary" in item


# ---------------------------------------------------------------------------
# Style presets
# ---------------------------------------------------------------------------


class TestStylePresets:

    def test_get_known_preset(self) -> None:
        preset = get_preset(StylePreset.PHOTOREALISTIC)
        assert preset.suffix
        assert preset.negative_prompt
        assert preset.guidance_scale > 0

    def test_get_preset_by_string(self) -> None:
        preset = get_preset("anime")
        assert "anime" in preset.suffix.lower()

    def test_get_unknown_preset_returns_default(self) -> None:
        preset = get_preset("nonexistent_style")
        default = get_preset(StylePreset.DEFAULT)
        assert preset == default


# ---------------------------------------------------------------------------
# PromptEnhancer
# ---------------------------------------------------------------------------


class TestPromptEnhancer:

    @pytest.mark.asyncio
    async def test_enhance_without_api_key_returns_original(self, settings: Settings) -> None:
        settings.openai_api_key = ""
        enhancer = PromptEnhancer(settings)
        result = await enhancer.enhance("a cat")
        assert result == "a cat"


# ---------------------------------------------------------------------------
# ImagePipeline
# ---------------------------------------------------------------------------


class TestImagePipeline:

    @pytest.mark.asyncio
    async def test_generate_success(self, pipeline: ImagePipeline) -> None:
        result = await pipeline.generate("a sunset over the ocean")
        assert result.status == GenerationStatus.COMPLETED
        assert result.image_bytes is not None
        assert result.provider == "fake"

    @pytest.mark.asyncio
    async def test_generate_with_style(self, pipeline: ImagePipeline) -> None:
        result = await pipeline.generate("mountains", style="cinematic")
        assert result.status == GenerationStatus.COMPLETED
        assert result.metadata.get("style") == "cinematic"

    @pytest.mark.asyncio
    async def test_generate_with_seed(self, pipeline: ImagePipeline) -> None:
        result = await pipeline.generate("flowers", seed=42)
        assert result.status == GenerationStatus.COMPLETED
        assert result.seed == 42

    @pytest.mark.asyncio
    async def test_generate_failure(self, settings: Settings) -> None:
        failing = FakeProvider(fail=True)
        pipeline = ImagePipeline(settings=settings, provider=failing)
        result = await pipeline.generate("anything")
        assert result.status == GenerationStatus.FAILED
        assert result.error

    @pytest.mark.asyncio
    async def test_generate_batch(self, pipeline: ImagePipeline) -> None:
        batch = await pipeline.generate_batch("roses", count=3)
        assert isinstance(batch, BatchResult)
        assert batch.total_requested == 3
        assert batch.total_completed == 3
        assert batch.total_failed == 0
        assert len(batch.results) == 3

    @pytest.mark.asyncio
    async def test_generate_batch_with_seeds(self, pipeline: ImagePipeline) -> None:
        seeds = [1, 2, 3]
        batch = await pipeline.generate_batch("roses", count=3, seeds=seeds)
        result_seeds = [r.seed for r in batch.results]
        assert result_seeds == seeds

    @pytest.mark.asyncio
    async def test_generate_batch_clamps_count(self, pipeline: ImagePipeline) -> None:
        batch = await pipeline.generate_batch("test", count=100)
        assert batch.total_requested <= pipeline._settings.max_batch_size

    @pytest.mark.asyncio
    async def test_batch_to_dict(self, pipeline: ImagePipeline) -> None:
        batch = await pipeline.generate_batch("test", count=2)
        data = batch.to_dict()
        assert "total_requested" in data
        assert "results" in data
        assert len(data["results"]) == 2

    @pytest.mark.asyncio
    async def test_enhance_prompt_standalone(self, pipeline: ImagePipeline) -> None:
        result = await pipeline.enhance_prompt("a dog")
        assert "original" in result
        assert "enhanced" in result
        # Enhancement is disabled in test settings, so enhanced == original.
        assert result["original"] == "a dog"


# ---------------------------------------------------------------------------
# Storage
# ---------------------------------------------------------------------------


class TestGallery:

    def test_add_and_get(self) -> None:
        gallery = Gallery()
        meta = ImageMetadata(
            image_id="abc123",
            filename="abc123.png",
            storage_path="/tmp/abc123.png",
            prompt="test",
            provider="fake",
            model="fake-v1",
            width=512,
            height=512,
        )
        gallery.add(meta)
        assert gallery.get("abc123") is meta
        assert gallery.count == 1

    def test_list_pagination(self) -> None:
        gallery = Gallery()
        for i in range(10):
            gallery.add(
                ImageMetadata(
                    image_id=f"img_{i:03d}",
                    filename=f"img_{i:03d}.png",
                    storage_path=f"/tmp/img_{i:03d}.png",
                    prompt=f"prompt {i}",
                    provider="fake",
                    model="fake-v1",
                    width=512,
                    height=512,
                )
            )
        page = gallery.list(limit=3, offset=2)
        assert len(page) == 3

    def test_list_filter_by_provider(self) -> None:
        gallery = Gallery()
        gallery.add(ImageMetadata(
            image_id="a", filename="a.png", storage_path="/a", prompt="p",
            provider="openai", model="m", width=512, height=512,
        ))
        gallery.add(ImageMetadata(
            image_id="b", filename="b.png", storage_path="/b", prompt="p",
            provider="replicate", model="m", width=512, height=512,
        ))
        assert len(gallery.list(provider="openai")) == 1

    def test_delete(self) -> None:
        gallery = Gallery()
        gallery.add(ImageMetadata(
            image_id="x", filename="x.png", storage_path="/x", prompt="p",
            provider="f", model="m", width=512, height=512,
        ))
        assert gallery.delete("x") is True
        assert gallery.get("x") is None
        assert gallery.delete("x") is False


class TestLocalStorage:

    @pytest.mark.asyncio
    async def test_save_and_load(self, settings: Settings) -> None:
        backend = LocalStorageBackend(settings.output_dir)
        path = await backend.save("test001", b"image-data", "png")
        assert await backend.exists(path)

        loaded = await backend.load(path)
        assert loaded == b"image-data"

    @pytest.mark.asyncio
    async def test_delete(self, settings: Settings) -> None:
        backend = LocalStorageBackend(settings.output_dir)
        path = await backend.save("test002", b"data", "png")
        await backend.delete(path)
        assert not await backend.exists(path)


class TestImageStorageService:

    @pytest.mark.asyncio
    async def test_store_and_retrieve(self, storage_service: ImageStorageService) -> None:
        result = GenerationResult(
            provider="fake",
            model="fake-v1",
            prompt="hello",
            width=512,
            height=512,
            status=GenerationStatus.COMPLETED,
            image_bytes=b"fake-image-data",
            image_format="png",
        )
        meta = await storage_service.store(result)
        assert meta is not None
        assert meta.image_id == result.image_id

        loaded = await storage_service.load_image(result.image_id)
        assert loaded == b"fake-image-data"

    @pytest.mark.asyncio
    async def test_store_no_bytes_returns_none(self, storage_service: ImageStorageService) -> None:
        result = GenerationResult(
            provider="fake",
            model="fake-v1",
            prompt="hello",
            width=512,
            height=512,
            status=GenerationStatus.COMPLETED,
            image_bytes=None,
        )
        meta = await storage_service.store(result)
        assert meta is None

    @pytest.mark.asyncio
    async def test_delete_image(self, storage_service: ImageStorageService) -> None:
        result = GenerationResult(
            provider="fake",
            model="fake-v1",
            prompt="hello",
            width=512,
            height=512,
            status=GenerationStatus.COMPLETED,
            image_bytes=b"data",
            image_format="png",
        )
        await storage_service.store(result)
        deleted = await storage_service.delete_image(result.image_id)
        assert deleted is True
        assert await storage_service.load_image(result.image_id) is None
