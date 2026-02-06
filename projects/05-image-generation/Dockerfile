FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output.
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install minimal system dependencies.
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl && \
    rm -rf /var/lib/apt/lists/*

# Create a non-root user.
RUN groupadd --gid 1000 appuser && \
    useradd --uid 1000 --gid appuser --create-home appuser

WORKDIR /app

# ---------------------------------------------------------------------------
# Builder stage: install Python dependencies
# ---------------------------------------------------------------------------
FROM base AS builder

COPY libs/common /app/libs/common
COPY projects/05-image-generation/pyproject.toml /app/projects/05-image-generation/

# Install the common library first, then the project dependencies.
RUN pip install --no-cache-dir /app/libs/common && \
    pip install --no-cache-dir /app/projects/05-image-generation/

# Copy the project source.
COPY projects/05-image-generation/src /app/projects/05-image-generation/src

# Re-install in editable mode so the source is on the path.
RUN pip install --no-cache-dir --no-deps /app/projects/05-image-generation/

# ---------------------------------------------------------------------------
# Runtime stage
# ---------------------------------------------------------------------------
FROM base AS runtime

# Copy installed packages from builder.
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
COPY --from=builder /app /app

WORKDIR /app/projects/05-image-generation

# Create output directory and give ownership to appuser.
RUN mkdir -p /app/generated_images && chown -R appuser:appuser /app

USER appuser

EXPOSE 8005

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8005/health || exit 1

ENTRYPOINT ["python", "-m", "image_generation.main"]
