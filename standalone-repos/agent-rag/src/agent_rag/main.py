"""Entry point for the Agent & RAG system."""

from __future__ import annotations

import uvicorn

from agent_rag.config import settings


def main() -> None:
    """Run the Agent & RAG API server."""
    uvicorn.run(
        "agent_rag.api:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level=settings.log_level,
    )


if __name__ == "__main__":
    main()
