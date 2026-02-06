"""Entry point for the Capstone Multi-Agent AI Platform.

Run with::

    uvicorn capstone.main:app --reload
    # or
    python -m capstone.main
"""

from __future__ import annotations

import uvicorn
from common import setup_logging

from capstone.api import create_app
from capstone.config import get_settings

settings = get_settings()
setup_logging(settings.log_level)

app = create_app(settings)


def main() -> None:
    """Launch the Uvicorn server programmatically."""
    uvicorn.run(
        "capstone.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=not settings.is_production,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
