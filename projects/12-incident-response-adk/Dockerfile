# ---------------------------------------------------------------------------
# Incident Response Orchestrator -- Production Dockerfile
# ---------------------------------------------------------------------------
# Multi-stage build for a minimal, secure runtime image.
#
# Build:
#   docker build -t incident-response-adk .
#
# Run:
#   docker run -p 8013:8000 incident-response-adk
# ---------------------------------------------------------------------------

# --- Stage 1: build --------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build tooling
RUN pip install --no-cache-dir hatchling uv

# Copy dependency manifests first for layer caching
COPY pyproject.toml ./
COPY src/ ./src/

# Copy the common library (workspace dependency)
COPY ../../libs/common /build/libs/common

# Install dependencies using uv for speed
RUN uv pip install --system --no-cache . && \
    uv pip install --system --no-cache /build/libs/common

# --- Stage 2: runtime ------------------------------------------------------
FROM python:3.11-slim AS runtime

# Security: non-root user
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY src/ ./src/

# Runtime configuration
ENV PYTHONPATH="/app/src" \
    PYTHONUNBUFFERED="1" \
    HOST="0.0.0.0" \
    PORT="8000" \
    ENVIRONMENT="production" \
    LOG_LEVEL="INFO"

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()"

# Switch to non-root user
USER appuser

# Start the application
CMD ["uvicorn", "incident_response.main:app", "--host", "0.0.0.0", "--port", "8000"]
