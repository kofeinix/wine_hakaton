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

# YOLO_AUTOINSTALL=false: Ultralytics не пытается ставить пакеты pip'ом во время запроса (pip в образе нет,
# попытка стоила ~2 с на каждом файле, который не открылся как картинка); pi-heif для HEIC — в зависимостях.
# YOLO_CONFIG_DIR=/tmp — настройки Ultralytics в /tmp/Ultralytics (домашняя папка appuser для него не пишется)
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/app/.venv/bin:$PATH" \
    YOLO_AUTOINSTALL=false \
    YOLO_CONFIG_DIR=/tmp

WORKDIR /app

# Runtime shared libraries required by opencv-python, imported by ultralytics.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
        libxrender1 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

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
