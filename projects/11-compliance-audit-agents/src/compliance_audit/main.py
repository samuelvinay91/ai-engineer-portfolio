"""Entry point for the Compliance Audit Agents service.

Creates the main FastAPI application, configures logging, and starts
the uvicorn server.
"""

from __future__ import annotations

import structlog
import uvicorn

from common import setup_logging

from compliance_audit.api import create_app
from compliance_audit.config import Settings, get_settings

logger = structlog.get_logger(__name__)


def build_app(settings: Settings | None = None) -> object:
    """Construct the fully-configured application.

    Returns the FastAPI application ready to serve requests.
    """
    settings = settings or get_settings()
    setup_logging(settings.log_level)

    app = create_app(settings)

    base_url = f"http://{settings.host}:{settings.port}"
    if settings.host == "0.0.0.0":
        base_url = f"http://localhost:{settings.port}"

    logger.info(
        "application_ready",
        service=settings.service_name,
        version=settings.service_version,
        docs_url=f"{base_url}/docs",
    )

    return app


# Module-level app for uvicorn CLI usage:
#   uvicorn compliance_audit.main:app --host 0.0.0.0 --port 8000
app = build_app()


def main() -> None:
    """Launch the Compliance Audit Agents server."""
    settings = get_settings()
    setup_logging(settings.log_level)

    application = build_app(settings)

    uvicorn.run(
        application,  # type: ignore[arg-type]
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
