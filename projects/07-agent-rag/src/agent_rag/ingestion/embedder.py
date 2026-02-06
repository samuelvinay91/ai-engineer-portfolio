"""Embedding service with multi-provider support, batching, and caching.

Supports:
- OpenAI (text-embedding-3-small/large with optional dimension reduction via MRL)
- Cohere (embed-english-v3.0)
- Local / mock embeddings for testing
"""

from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Sequence

import structlog

from agent_rag.config import EmbeddingProvider, settings

logger = structlog.get_logger(__name__)

Vector = list[float]


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    """Base class for embedding providers."""

    @property
    @abstractmethod
    def dimensions(self) -> int:
        """Return the output vector dimensionality."""

    @abstractmethod
    async def embed_texts(self, texts: Sequence[str]) -> list[Vector]:
        """Embed a batch of texts and return vectors."""

    async def embed_text(self, text: str) -> Vector:
        """Embed a single text string."""
        results = await self.embed_texts([text])
        return results[0]


# ---------------------------------------------------------------------------
# LRU embedding cache
# ---------------------------------------------------------------------------

class _EmbeddingCache:
    """Simple in-memory LRU cache for embeddings."""

    def __init__(self, max_size: int = 10_000) -> None:
        self._max_size = max_size
        self._store: OrderedDict[str, Vector] = OrderedDict()

    def _key(self, text: str, model: str) -> str:
        return hashlib.sha256(f"{model}:{text}".encode()).hexdigest()

    def get(self, text: str, model: str) -> Vector | None:
        k = self._key(text, model)
        if k in self._store:
            self._store.move_to_end(k)
            return self._store[k]
        return None

    def put(self, text: str, model: str, vector: Vector) -> None:
        k = self._key(text, model)
        self._store[k] = vector
        self._store.move_to_end(k)
        if len(self._store) > self._max_size:
            self._store.popitem(last=False)

    @property
    def size(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# OpenAI embedder
# ---------------------------------------------------------------------------

class OpenAIEmbedder(BaseEmbedder):
    """OpenAI embedding with Matryoshka Representation Learning (MRL) support."""

    def __init__(
        self,
        model: str | None = None,
        dimensions: int | None = None,
        api_key: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        from openai import AsyncOpenAI

        self._model = model or settings.embedding_model
        self._dims = dimensions or settings.embedding_dimensions
        self._batch_size = batch_size or settings.embedding_batch_size
        key = api_key or settings.openai_api_key.get_secret_value()
        self._client = AsyncOpenAI(api_key=key)
        self._cache = _EmbeddingCache()

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed_texts(self, texts: Sequence[str]) -> list[Vector]:
        results: list[Vector | None] = [None] * len(texts)
        uncached_indices: list[int] = []

        # Check cache first
        for i, text in enumerate(texts):
            cached = self._cache.get(text, self._model)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)

        if not uncached_indices:
            return [v for v in results if v is not None]

        # Batch embed uncached texts
        uncached_texts = [texts[i] for i in uncached_indices]
        for batch_start in range(0, len(uncached_texts), self._batch_size):
            batch = uncached_texts[batch_start: batch_start + self._batch_size]
            batch_indices = uncached_indices[batch_start: batch_start + self._batch_size]

            kwargs: dict = {"model": self._model, "input": batch}
            # MRL dimension reduction -- only text-embedding-3-* supports this
            if "text-embedding-3" in self._model:
                kwargs["dimensions"] = self._dims

            response = await self._client.embeddings.create(**kwargs)

            for j, emb_obj in enumerate(response.data):
                vec = emb_obj.embedding
                original_idx = batch_indices[j]
                results[original_idx] = vec
                self._cache.put(texts[original_idx], self._model, vec)

            # Simple rate-limit backoff between batches
            if batch_start + self._batch_size < len(uncached_texts):
                time.sleep(0.05)

        return [v for v in results if v is not None]


# ---------------------------------------------------------------------------
# Cohere embedder
# ---------------------------------------------------------------------------

class CohereEmbedder(BaseEmbedder):
    """Cohere embedding service."""

    def __init__(
        self,
        model: str = "embed-english-v3.0",
        api_key: str | None = None,
        batch_size: int | None = None,
    ) -> None:
        import cohere

        self._model = model
        self._batch_size = batch_size or settings.embedding_batch_size
        key = api_key or settings.cohere_api_key.get_secret_value()
        self._client = cohere.AsyncClientV2(api_key=key)
        self._cache = _EmbeddingCache()
        self._dims = 1024  # embed-english-v3.0 default

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed_texts(self, texts: Sequence[str]) -> list[Vector]:
        results: list[Vector | None] = [None] * len(texts)
        uncached_indices: list[int] = []

        for i, text in enumerate(texts):
            cached = self._cache.get(text, self._model)
            if cached is not None:
                results[i] = cached
            else:
                uncached_indices.append(i)

        if not uncached_indices:
            return [v for v in results if v is not None]

        uncached_texts = [texts[i] for i in uncached_indices]
        for batch_start in range(0, len(uncached_texts), self._batch_size):
            batch = list(uncached_texts[batch_start: batch_start + self._batch_size])
            batch_indices = uncached_indices[batch_start: batch_start + self._batch_size]

            response = await self._client.embed(
                texts=batch,
                model=self._model,
                input_type="search_document",
                embedding_types=["float"],
            )

            embeddings = response.embeddings.float_ or []
            for j, vec in enumerate(embeddings):
                original_idx = batch_indices[j]
                results[original_idx] = list(vec)
                self._cache.put(texts[original_idx], self._model, list(vec))

        return [v for v in results if v is not None]


# ---------------------------------------------------------------------------
# Local / mock embedder (for testing)
# ---------------------------------------------------------------------------

class LocalEmbedder(BaseEmbedder):
    """Deterministic local embedder for testing and development.

    Produces consistent embeddings by hashing text content -- not meaningful
    for similarity but useful for pipeline testing without API keys.
    """

    def __init__(self, dimensions: int | None = None) -> None:
        self._dims = dimensions or settings.embedding_dimensions

    @property
    def dimensions(self) -> int:
        return self._dims

    async def embed_texts(self, texts: Sequence[str]) -> list[Vector]:
        vectors: list[Vector] = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            # Extend hash bytes to fill dimensions
            raw = list(digest) * ((self._dims // len(digest)) + 1)
            # Normalize to [-1, 1] range
            vec = [(b / 127.5 - 1.0) for b in raw[: self._dims]]
            vectors.append(vec)
        return vectors


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedder(provider: EmbeddingProvider | None = None) -> BaseEmbedder:
    """Create an embedder for the configured provider."""
    prov = provider or settings.embedding_provider
    if prov == EmbeddingProvider.OPENAI:
        return OpenAIEmbedder()
    elif prov == EmbeddingProvider.COHERE:
        return CohereEmbedder()
    elif prov == EmbeddingProvider.LOCAL:
        return LocalEmbedder()
    else:
        raise ValueError(f"Unknown embedding provider: {prov}")
