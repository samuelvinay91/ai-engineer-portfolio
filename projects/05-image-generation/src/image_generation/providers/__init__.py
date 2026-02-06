"""Provider registry and factory for image generation backends."""

from __future__ import annotations

from typing import Any

import structlog

from image_generation.config import ProviderName, Settings
from image_generation.providers.base import (
    GenerationResult,
    GenerationStatus,
    ImageProvider,
    Img2ImgInput,
    InpaintInput,
)
from image_generation.providers.openai_provider import OpenAIProvider
from image_generation.providers.replicate_provider import ReplicateProvider

logger = structlog.get_logger(__name__)

__all__ = [
    "GenerationResult",
    "GenerationStatus",
    "ImageProvider",
    "Img2ImgInput",
    "InpaintInput",
    "OpenAIProvider",
    "ReplicateProvider",
    "get_provider",
    "list_providers",
]

# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------

_PROVIDER_MAP: dict[ProviderName, type[ImageProvider]] = {
    ProviderName.OPENAI: OpenAIProvider,
    ProviderName.REPLICATE: ReplicateProvider,
}

# Cache instantiated providers so expensive init only happens once.
_provider_cache: dict[ProviderName, ImageProvider] = {}


def get_provider(
    provider_name: ProviderName | str | None = None,
    settings: Settings | None = None,
) -> ImageProvider:
    """Return a (cached) provider instance.

    Args:
        provider_name: Which provider to use.  Defaults to
            ``settings.default_provider``.
        settings: Application settings.  Defaults to :func:`get_settings`.

    Returns:
        An initialised :class:`ImageProvider`.

    Raises:
        ValueError: If the requested provider is unknown.
    """
    if settings is None:
        from image_generation.config import get_settings

        settings = get_settings()

    if provider_name is None:
        name = settings.default_provider
    elif isinstance(provider_name, str):
        try:
            name = ProviderName(provider_name)
        except ValueError:
            valid = [p.value for p in ProviderName]
            raise ValueError(
                f"Unknown provider '{provider_name}'. Must be one of {valid}"
            ) from None
    else:
        name = provider_name

    if name in _provider_cache:
        return _provider_cache[name]

    provider_cls = _PROVIDER_MAP.get(name)
    if provider_cls is None:
        if name == ProviderName.LOCAL:
            raise ValueError(
                "Local provider requires 'torch' and 'diffusers'. "
                "Install with: pip install image-generation[local]"
            )
        raise ValueError(f"No provider implementation for '{name.value}'")

    provider = provider_cls(settings)
    _provider_cache[name] = provider
    logger.info("provider.initialised", provider=name.value)
    return provider


def clear_provider_cache() -> None:
    """Clear the cached provider instances (useful in tests)."""
    _provider_cache.clear()


async def list_providers(settings: Settings | None = None) -> list[dict[str, Any]]:
    """Return health/status info for all registered providers."""
    if settings is None:
        from image_generation.config import get_settings

        settings = get_settings()

    results: list[dict[str, Any]] = []
    for name, cls in _PROVIDER_MAP.items():
        try:
            provider = get_provider(name, settings)
            info = await provider.health_check()
        except Exception as exc:
            info = {
                "provider": name.value,
                "status": f"error: {exc}",
                "models": [],
            }
        results.append(info)
    return results
