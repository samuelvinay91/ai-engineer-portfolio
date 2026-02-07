"""Image storage service with local filesystem and S3-compatible backends.

Provides:
- Persisting generated images (bytes -> file / S3 object).
- Image metadata tracking in a lightweight in-memory gallery.
- Gallery management: list, get, delete.
"""

from __future__ import annotations

import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import structlog

from image_generation.config import Settings, get_settings
from image_generation.providers.base import GenerationResult

logger = structlog.get_logger(__name__)


# ---------------------------------------------------------------------------
# Image metadata (gallery entry)
# ---------------------------------------------------------------------------


class ImageMetadata:
    """Metadata record for a single stored image."""

    def __init__(
        self,
        image_id: str,
        filename: str,
        storage_path: str,
        prompt: str,
        provider: str,
        model: str,
        width: int,
        height: int,
        image_format: str = "png",
        seed: int | None = None,
        style: str | None = None,
        duration_seconds: float | None = None,
        extra: dict[str, Any] | None = None,
        created_at: datetime | None = None,
    ) -> None:
        self.image_id = image_id
        self.filename = filename
        self.storage_path = storage_path
        self.prompt = prompt
        self.provider = provider
        self.model = model
        self.width = width
        self.height = height
        self.image_format = image_format
        self.seed = seed
        self.style = style
        self.duration_seconds = duration_seconds
        self.extra = extra or {}
        self.created_at = created_at or datetime.now(timezone.utc)

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "filename": self.filename,
            "storage_path": self.storage_path,
            "prompt": self.prompt,
            "provider": self.provider,
            "model": self.model,
            "width": self.width,
            "height": self.height,
            "image_format": self.image_format,
            "seed": self.seed,
            "style": self.style,
            "duration_seconds": self.duration_seconds,
            "extra": self.extra,
            "created_at": self.created_at.isoformat(),
        }


# ---------------------------------------------------------------------------
# Abstract storage backend
# ---------------------------------------------------------------------------


class StorageBackend(ABC):
    """Interface for persisting image bytes."""

    @abstractmethod
    async def save(self, image_id: str, data: bytes, image_format: str) -> str:
        """Store *data* and return the storage path / URL."""

    @abstractmethod
    async def load(self, storage_path: str) -> bytes:
        """Load image bytes from *storage_path*."""

    @abstractmethod
    async def delete(self, storage_path: str) -> None:
        """Remove the stored image at *storage_path*."""

    @abstractmethod
    async def exists(self, storage_path: str) -> bool:
        """Check if a stored image exists at *storage_path*."""


# ---------------------------------------------------------------------------
# Local filesystem backend
# ---------------------------------------------------------------------------


class LocalStorageBackend(StorageBackend):
    """Persists images to a local directory."""

    def __init__(self, output_dir: Path) -> None:
        self._output_dir = output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("storage.local.init", output_dir=str(output_dir))

    async def save(self, image_id: str, data: bytes, image_format: str) -> str:
        filename = f"{image_id}.{image_format}"
        path = self._output_dir / filename
        path.write_bytes(data)
        logger.debug("storage.local.saved", path=str(path), size=len(data))
        return str(path)

    async def load(self, storage_path: str) -> bytes:
        path = Path(storage_path)
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {storage_path}")
        return path.read_bytes()

    async def delete(self, storage_path: str) -> None:
        path = Path(storage_path)
        if path.exists():
            path.unlink()
            logger.debug("storage.local.deleted", path=str(path))

    async def exists(self, storage_path: str) -> bool:
        return Path(storage_path).exists()


# ---------------------------------------------------------------------------
# S3-compatible storage backend
# ---------------------------------------------------------------------------


class S3StorageBackend(StorageBackend):
    """Persists images to an S3-compatible object store (AWS S3, MinIO, etc.)."""

    def __init__(self, settings: Settings) -> None:
        self._bucket = settings.s3_bucket
        self._prefix = settings.s3_prefix
        self._region = settings.s3_region
        self._endpoint_url = settings.s3_endpoint_url or None

        import boto3

        client_kwargs: dict[str, Any] = {"region_name": self._region}
        if self._endpoint_url:
            client_kwargs["endpoint_url"] = self._endpoint_url
        if settings.aws_access_key_id:
            client_kwargs["aws_access_key_id"] = settings.aws_access_key_id
            client_kwargs["aws_secret_access_key"] = settings.aws_secret_access_key

        self._s3 = boto3.client("s3", **client_kwargs)
        logger.info(
            "storage.s3.init",
            bucket=self._bucket,
            prefix=self._prefix,
            region=self._region,
        )

    def _key(self, image_id: str, image_format: str) -> str:
        return f"{self._prefix}{image_id}.{image_format}"

    async def save(self, image_id: str, data: bytes, image_format: str) -> str:
        import asyncio

        key = self._key(image_id, image_format)
        content_type = f"image/{image_format}"

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._s3.put_object(
                Bucket=self._bucket,
                Key=key,
                Body=data,
                ContentType=content_type,
            ),
        )
        storage_path = f"s3://{self._bucket}/{key}"
        logger.debug("storage.s3.saved", key=key, size=len(data))
        return storage_path

    async def load(self, storage_path: str) -> bytes:
        import asyncio

        # Parse key from s3://bucket/key or plain key.
        key = storage_path
        if storage_path.startswith("s3://"):
            key = storage_path.split("/", 3)[-1]

        loop = asyncio.get_running_loop()
        response = await loop.run_in_executor(
            None,
            lambda: self._s3.get_object(Bucket=self._bucket, Key=key),
        )
        return response["Body"].read()

    async def delete(self, storage_path: str) -> None:
        import asyncio

        key = storage_path
        if storage_path.startswith("s3://"):
            key = storage_path.split("/", 3)[-1]

        loop = asyncio.get_running_loop()
        await loop.run_in_executor(
            None,
            lambda: self._s3.delete_object(Bucket=self._bucket, Key=key),
        )
        logger.debug("storage.s3.deleted", key=key)

    async def exists(self, storage_path: str) -> bool:
        import asyncio

        key = storage_path
        if storage_path.startswith("s3://"):
            key = storage_path.split("/", 3)[-1]

        loop = asyncio.get_running_loop()
        try:
            await loop.run_in_executor(
                None,
                lambda: self._s3.head_object(Bucket=self._bucket, Key=key),
            )
            return True
        except Exception:
            return False


# ---------------------------------------------------------------------------
# Gallery (image metadata tracker)
# ---------------------------------------------------------------------------


class Gallery:
    """In-memory gallery that tracks image metadata.

    In production this would be backed by a database.  The in-memory
    implementation is suitable for development and small deployments.
    """

    def __init__(self) -> None:
        self._items: dict[str, ImageMetadata] = {}

    def add(self, meta: ImageMetadata) -> None:
        self._items[meta.image_id] = meta

    def get(self, image_id: str) -> ImageMetadata | None:
        return self._items.get(image_id)

    def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        provider: str | None = None,
        style: str | None = None,
    ) -> list[ImageMetadata]:
        """Return a paginated, optionally filtered list of gallery items."""
        items = list(self._items.values())

        if provider:
            items = [i for i in items if i.provider == provider]
        if style:
            items = [i for i in items if i.style == style]

        # Sort newest first.
        items.sort(key=lambda i: i.created_at, reverse=True)
        return items[offset : offset + limit]

    def delete(self, image_id: str) -> bool:
        if image_id in self._items:
            del self._items[image_id]
            return True
        return False

    @property
    def count(self) -> int:
        return len(self._items)


# ---------------------------------------------------------------------------
# ImageStorageService (composite)
# ---------------------------------------------------------------------------


class ImageStorageService:
    """Orchestrates storing image bytes and tracking metadata."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._backend = self._build_backend()
        self._gallery = Gallery()

    def _build_backend(self) -> StorageBackend:
        if self._settings.storage_backend == "s3" and self._settings.s3_bucket:
            return S3StorageBackend(self._settings)
        return LocalStorageBackend(self._settings.output_dir)

    @property
    def gallery(self) -> Gallery:
        return self._gallery

    async def store(self, result: GenerationResult) -> ImageMetadata | None:
        """Persist a generation result and register it in the gallery.

        Returns the :class:`ImageMetadata` on success, or ``None`` if
        there are no image bytes to store.
        """
        if not result.image_bytes:
            logger.debug("storage.store.no_bytes", image_id=result.image_id)
            return None

        storage_path = await self._backend.save(
            result.image_id,
            result.image_bytes,
            result.image_format,
        )

        meta = ImageMetadata(
            image_id=result.image_id,
            filename=f"{result.image_id}.{result.image_format}",
            storage_path=storage_path,
            prompt=result.prompt,
            provider=result.provider,
            model=result.model,
            width=result.width,
            height=result.height,
            image_format=result.image_format,
            seed=result.seed,
            style=result.metadata.get("style"),
            duration_seconds=result.duration_seconds,
            extra=result.metadata,
            created_at=result.created_at,
        )
        self._gallery.add(meta)
        logger.info(
            "storage.store.success",
            image_id=result.image_id,
            path=storage_path,
        )
        return meta

    async def load_image(self, image_id: str) -> bytes | None:
        """Load raw image bytes for a given image ID."""
        meta = self._gallery.get(image_id)
        if meta is None:
            return None
        try:
            return await self._backend.load(meta.storage_path)
        except Exception as exc:
            logger.error("storage.load.error", image_id=image_id, error=str(exc))
            return None

    async def delete_image(self, image_id: str) -> bool:
        """Delete an image from storage and gallery."""
        meta = self._gallery.get(image_id)
        if meta is None:
            return False
        try:
            await self._backend.delete(meta.storage_path)
        except Exception as exc:
            logger.warning("storage.delete.backend_error", error=str(exc))
        self._gallery.delete(image_id)
        return True
