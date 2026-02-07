"""Entry point for the MCP & A2A Integration service."""

from __future__ import annotations

import uvicorn

from common import setup_logging

from mcp_a2a.api import create_app
from mcp_a2a.config import get_settings


def main() -> None:
    """Launch the FastAPI server."""
    settings = get_settings()
    setup_logging(settings.log_level)

    app = create_app(settings)

    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
