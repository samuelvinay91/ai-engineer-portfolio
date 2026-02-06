# ---------- Stage 1: Build ----------
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build dependencies
RUN pip install --no-cache-dir uv

# Copy workspace root for dependency resolution
COPY pyproject.toml ./
COPY libs/common ./libs/common

# Copy project sources
COPY projects/08-mcp-a2a/pyproject.toml ./projects/08-mcp-a2a/
COPY projects/08-mcp-a2a/src ./projects/08-mcp-a2a/src

# Install into a virtual environment
RUN uv venv /opt/venv && \
    . /opt/venv/bin/activate && \
    uv pip install ./libs/common && \
    uv pip install ./projects/08-mcp-a2a

# ---------- Stage 2: Runtime ----------
FROM python:3.11-slim AS runtime

# Security: run as non-root
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid 1000 --shell /bin/bash --create-home appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Copy application source
COPY projects/08-mcp-a2a/src ./src

# Switch to non-root user
USER appuser

EXPOSE 8008

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD ["python", "-c", "import httpx; httpx.get('http://localhost:8008/health').raise_for_status()"]

CMD ["python", "-m", "mcp_a2a.main"]
