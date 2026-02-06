"""Entry point for the LLM Playground service."""

import uvicorn

from common import setup_logging

from llm_playground.config import get_settings


def main() -> None:
    settings = get_settings()
    setup_logging(settings.log_level)

    uvicorn.run(
        "llm_playground.api:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.environment == "development",
        log_level=settings.log_level.lower(),
    )


if __name__ == "__main__":
    main()
