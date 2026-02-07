"""Entry point for the Ask-the-Web agent service."""

from __future__ import annotations

import uvicorn
from common.logging import setup_logging

from ask_the_web.api import create_app
from ask_the_web.config import get_settings


def main() -> None:
    """Start the Uvicorn server."""
    settings = get_settings()
    setup_logging(settings.log_level)

    app = create_app()

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level=settings.log_level.lower(),
        access_log=True,
    )


if __name__ == "__main__":
    main()
