# Stage 1: Build with uv
FROM ghcr.io/astral-sh/uv:0.5-python3.12-bookworm-slim AS builder

WORKDIR /app

# Copy workspace root files needed for dependency resolution
COPY pyproject.toml /app/pyproject.toml
COPY libs /app/libs

# Copy project source
COPY projects/01-llm-playground /app/projects/01-llm-playground

# Install dependencies into a virtual environment
RUN uv venv /app/.venv && \
    uv pip install --python /app/.venv/bin/python \
        -e /app/libs/common \
        -e /app/projects/01-llm-playground

# Stage 2: Runtime
FROM python:3.12-slim-bookworm AS runtime

RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy virtual environment and source from builder
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/libs /app/libs
COPY --from=builder /app/projects/01-llm-playground /app/projects/01-llm-playground

# Ensure venv binaries are on PATH
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Switch to non-root user
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()"

CMD ["python", "-m", "llm_playground.main"]
