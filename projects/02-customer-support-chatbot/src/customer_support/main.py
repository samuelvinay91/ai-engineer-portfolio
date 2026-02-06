"""Application entry-point.

Run the server with::

    python -m customer_support.main

Or via uvicorn directly::

    uvicorn customer_support.main:app --reload
"""

from __future__ import annotations

import structlog

from customer_support.api import create_app
from customer_support.config import get_settings

# Configure structlog once at import time
structlog.configure(
    processors=[
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer() if not get_settings().is_production
        else structlog.processors.JSONRenderer(),
    ],
    wrapper_class=structlog.make_filtering_bound_logger(
        getattr(structlog, get_settings().log_level.value, 20),
    ),
    context_class=dict,
    logger_factory=structlog.PrintLoggerFactory(),
    cache_logger_on_first_use=True,
)

# The ASGI app object (importable by uvicorn)
app = create_app()


def main() -> None:
    """Start the uvicorn server programmatically."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "customer_support.main:app",
        host=settings.host,
        port=settings.port,
        workers=settings.workers,
        reload=not settings.is_production,
        log_level=settings.log_level.value.lower(),
    )


if __name__ == "__main__":
    main()
