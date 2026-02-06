# ============================================================================
# Multi-stage Dockerfile for the Deep Research service
# ============================================================================

# ---- Stage 1: build dependencies -------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# Install build tooling
RUN pip install --no-cache-dir hatchling

# Copy only dependency manifests first (layer caching)
COPY pyproject.toml ./
COPY src/ src/

# Build the wheel
RUN pip wheel --no-cache-dir --wheel-dir /wheels .

# ---- Stage 2: runtime image ------------------------------------------------
FROM python:3.12-slim AS runtime

# Security: create a non-root user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home appuser

WORKDIR /app

# Install runtime dependencies from pre-built wheels
COPY --from=builder /wheels /wheels
RUN pip install --no-cache-dir /wheels/*.whl \
    && rm -rf /wheels

# Copy application source
COPY src/ src/

# Environment defaults
ENV DEEP_RESEARCH_HOST="0.0.0.0" \
    DEEP_RESEARCH_PORT="8000" \
    DEEP_RESEARCH_LOG_LEVEL="INFO" \
    DEEP_RESEARCH_ENVIRONMENT="production"

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()" || exit 1

# Drop to non-root
USER appuser

# Entry-point
CMD ["python", "-m", "deep_research.main"]
