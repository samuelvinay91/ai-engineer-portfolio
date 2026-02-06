"""Entry point for the Image Generation Service.

Run with::

    uvicorn image_generation.main:app --reload
    # or
    python -m image_generation.main
"""

from __future__ import annotations

import uvicorn
from common.logging import setup_logging

from image_generation.api import app  # noqa: F401 -- re-exported for uvicorn
from image_generation.config import get_settings


def main() -> None:
    """Bootstrap logging and start the Uvicorn server."""
    settings = get_settings()
    setup_logging(settings.log_level)

    uvicorn.run(
        "image_generation.api:app",
        host=settings.host,
        port=settings.port,
        reload=not settings.is_production,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
