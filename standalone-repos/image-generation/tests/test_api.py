"""Tests for the Image Generation Service FastAPI application."""

from __future__ import annotations

import pytest
from httpx import AsyncClient


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


class TestHealth:

    @pytest.mark.asyncio
    async def test_health_check(self, client: AsyncClient) -> None:
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["service"] == "image-generation"


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


class TestGenerate:

    @pytest.mark.asyncio
    async def test_generate_success(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/generate",
            json={"prompt": "a beautiful sunset over the ocean"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"
        assert data["image_id"]
        assert data["provider"] == "fake"
        assert data["image_base64"]

    @pytest.mark.asyncio
    async def test_generate_with_style(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/generate",
            json={"prompt": "mountain landscape", "style": "cinematic"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "completed"

    @pytest.mark.asyncio
    async def test_generate_with_seed(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/generate",
            json={"prompt": "a cat", "seed": 42},
        )
        assert resp.status_code == 200
        assert resp.json()["seed"] == 42

    @pytest.mark.asyncio
    async def test_generate_with_dimensions(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/generate",
            json={"prompt": "landscape", "width": 1024, "height": 768},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["width"] == 1024
        assert data["height"] == 768

    @pytest.mark.asyncio
    async def test_generate_empty_prompt_rejected(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/generate", json={"prompt": ""})
        assert resp.status_code == 422

    @pytest.mark.asyncio
    async def test_generate_missing_prompt_rejected(self, client: AsyncClient) -> None:
        resp = await client.post("/api/v1/generate", json={})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# Batch generation
# ---------------------------------------------------------------------------


class TestBatchGenerate:

    @pytest.mark.asyncio
    async def test_batch_generate(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/generate/batch",
            json={"prompt": "roses in a garden", "count": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_requested"] == 3
        assert data["total_completed"] == 3
        assert len(data["results"]) == 3

    @pytest.mark.asyncio
    async def test_batch_generate_with_seeds(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/generate/batch",
            json={"prompt": "sky", "count": 2, "seeds": [10, 20]},
        )
        assert resp.status_code == 200
        seeds = [r["seed"] for r in resp.json()["results"]]
        assert seeds == [10, 20]


# ---------------------------------------------------------------------------
# Prompt enhancement
# ---------------------------------------------------------------------------


class TestEnhancePrompt:

    @pytest.mark.asyncio
    async def test_enhance_prompt(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/enhance-prompt",
            json={"prompt": "a dog"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "original" in data
        assert "enhanced" in data
        assert data["original"] == "a dog"


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------


class TestGallery:

    @pytest.mark.asyncio
    async def test_gallery_empty(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/gallery")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] == 0
        assert data["items"] == []

    @pytest.mark.asyncio
    async def test_gallery_after_generation(self, client: AsyncClient) -> None:
        # Generate an image first.
        gen_resp = await client.post(
            "/api/v1/generate",
            json={"prompt": "test image for gallery"},
        )
        assert gen_resp.status_code == 200
        image_id = gen_resp.json()["image_id"]

        # Check gallery.
        gallery_resp = await client.get("/api/v1/gallery")
        assert gallery_resp.status_code == 200
        data = gallery_resp.json()
        assert data["total"] >= 1
        ids = [item["image_id"] for item in data["items"]]
        assert image_id in ids

    @pytest.mark.asyncio
    async def test_gallery_get_specific_image(self, client: AsyncClient) -> None:
        gen_resp = await client.post(
            "/api/v1/generate",
            json={"prompt": "specific image test"},
        )
        image_id = gen_resp.json()["image_id"]

        resp = await client.get(f"/api/v1/gallery/{image_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["image_id"] == image_id
        assert "image_base64" in data

    @pytest.mark.asyncio
    async def test_gallery_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/gallery/nonexistent_id_12345")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------


class TestConcepts:

    @pytest.mark.asyncio
    async def test_list_concepts(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/concepts")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) == 4
        types = {item["type"] for item in data}
        assert types == {"vae", "gan", "diffusion", "autoregressive"}

    @pytest.mark.asyncio
    @pytest.mark.parametrize("model_type", ["vae", "gan", "diffusion", "autoregressive"])
    async def test_get_concept(self, client: AsyncClient, model_type: str) -> None:
        resp = await client.get(f"/api/v1/concepts/{model_type}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["model_type"] == model_type
        assert data["title"]
        assert data["ascii_diagram"]
        assert data["architecture"]
        assert data["training"]
        assert data["strengths"]
        assert data["weaknesses"]
        assert data["key_models"]
        assert data["data_preparation"]

    @pytest.mark.asyncio
    async def test_get_concept_invalid(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/concepts/transformer")
        assert resp.status_code == 404


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------


class TestProviders:

    @pytest.mark.asyncio
    async def test_list_providers(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/providers")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        provider_names = {p["provider"] for p in data}
        assert "openai" in provider_names
        assert "replicate" in provider_names


# ---------------------------------------------------------------------------
# Styles
# ---------------------------------------------------------------------------


class TestStyles:

    @pytest.mark.asyncio
    async def test_list_styles(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/styles")
        assert resp.status_code == 200
        data = resp.json()
        assert isinstance(data, list)
        names = {s["name"] for s in data}
        assert "photorealistic" in names
        assert "anime" in names
        assert "default" in names
