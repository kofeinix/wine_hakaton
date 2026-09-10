# syntax=docker/dockerfile:1

# ---------- Build stage ----------
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS builder

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy

WORKDIR /app

# Install dependencies first for better layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# ---------- Runtime stage ----------
FROM python:3.13-slim-bookworm AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

# Run as non-root user
RUN useradd --create-home --uid 1000 appuser

# Copy the virtual environment from the builder
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Copy application source
COPY --chown=appuser:appuser main.py ./
COPY --chown=appuser:appuser src ./src
COPY --chown=appuser:appuser scripts ./scripts

USER appuser

EXPOSE 8000

CMD ["python", "main.py"]
