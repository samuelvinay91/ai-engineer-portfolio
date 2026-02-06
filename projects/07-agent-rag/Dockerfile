# ---- Build stage ----
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml ./
COPY src/ ./src/

RUN pip install --no-cache-dir --prefix=/install .


# ---- Runtime stage ----
FROM python:3.11-slim AS runtime

# Non-root user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid 1000 --create-home appuser

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application source
WORKDIR /app
COPY --chown=appuser:appuser src/ ./src/

USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    AGENT_RAG_HOST=0.0.0.0 \
    AGENT_RAG_PORT=8007

EXPOSE 8007

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import httpx; r = httpx.get('http://localhost:8007/health'); r.raise_for_status()"

ENTRYPOINT ["python", "-m", "agent_rag.main"]
