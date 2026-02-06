# =============================================================================
# Multi-stage Dockerfile for Customer Support Chatbot
# =============================================================================
# Stage 1: build    -- install dependencies into a virtual-env
# Stage 2: runtime  -- copy only what is needed into a slim image
# =============================================================================

# ---------------------------------------------------------------------------
# Stage 1 -- Builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

WORKDIR /build

# System deps for building Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libpq-dev && \
    rm -rf /var/lib/apt/lists/*

# Create virtual-env so we can copy it cleanly into the runtime stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install dependencies first (layer cache optimisation)
COPY pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir ".[dev]" || \
    pip install --no-cache-dir -e .

# Copy source and install the package itself
COPY src/ ./src/
COPY data/ ./data/
RUN pip install --no-cache-dir .

# ---------------------------------------------------------------------------
# Stage 2 -- Runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

# Security: run as non-root
RUN groupadd --gid 1001 appgroup && \
    useradd --uid 1001 --gid appgroup --shell /bin/false --create-home appuser

# Runtime system deps (libpq for asyncpg)
RUN apt-get update && \
    apt-get install -y --no-install-recommends libpq5 curl && \
    rm -rf /var/lib/apt/lists/*

# Copy virtual-env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Copy application code and data
WORKDIR /app
COPY --from=builder /build/src ./src
COPY --from=builder /build/data ./data

# Ensure the template dir env var points to the correct location
ENV CHATBOT_PROMPT_TEMPLATE_DIR=/app/data/templates

# Switch to non-root user
USER appuser

EXPOSE 8000

# Health check -- hit the /health endpoint every 30s
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application via uvicorn
CMD ["uvicorn", "customer_support.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
