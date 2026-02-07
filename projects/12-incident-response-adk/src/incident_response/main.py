"""Entry point for the Incident Response Orchestrator service.

Creates the FastAPI application, configures logging, and starts the
uvicorn server.
"""

from __future__ import annotations

import structlog
import uvicorn

from common import setup_logging

from incident_response.api import create_app
from incident_response.config import Settings, get_settings

logger = structlog.get_logger(__name__)


def build_app(settings: Settings | None = None) -> object:
    """Construct the fully-configured application.

    Returns the FastAPI application ready to serve requests.
    """
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    app = create_app(settings)

    logger.info(
        "application_ready",
        service=settings.service_name,
        version=settings.service_version,
        port=settings.port,
        docs_url=f"http://localhost:{settings.port}/docs",
    )

    return app


def main() -> None:
    """Launch the Incident Response Orchestrator server."""
    settings = get_settings()
    setup_logging(settings.log_level)

    app = build_app(settings)

    uvicorn.run(
        app,  # type: ignore[arg-type]
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
