# ---------------------------------------------------------------------------
# Stage 1: Build
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps
RUN pip install --no-cache-dir hatchling

# Copy common lib first (for the "common" dependency)
COPY libs/common /build/libs/common

# Copy project
COPY projects/03-ask-the-web-agent /build/project

# Build common wheel
RUN cd /build/libs/common && pip wheel --no-deps --wheel-dir /build/wheels .

# Build project wheel
RUN cd /build/project && pip wheel --no-deps --wheel-dir /build/wheels .

# Install all wheels + runtime deps into a virtual env
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --no-cache-dir /build/wheels/*.whl \
    && pip install --no-cache-dir \
        fastapi uvicorn[standard] pydantic pydantic-settings \
        anthropic openai langchain-core langchain-anthropic langgraph \
        tavily-python httpx beautifulsoup4 markdownify redis \
        structlog sse-starlette

# ---------------------------------------------------------------------------
# Stage 2: Runtime
# ---------------------------------------------------------------------------
FROM python:3.11-slim AS runtime

# Create non-root user
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser

# Copy virtual env from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy source (for module resolution at runtime)
COPY projects/03-ask-the-web-agent/src /app/src

# Switch to non-root
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import httpx; r = httpx.get('http://localhost:8000/health'); r.raise_for_status()"]

CMD ["python", "-m", "ask_the_web.main"]
