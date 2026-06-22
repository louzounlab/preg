
FROM python:3.11-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    MPLBACKEND=Agg \
    PORT=8000

# System packages needed by numpy/scipy/matplotlib/lightgbm/torch at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        libstdc++6 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first for better layer caching.
# Use the PyTorch CPU wheel index so we don't pull the CUDA build (~2GB).
COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -r requirements.txt

# Application code. Everything lives under ./app in the repo and is copied into
# the image WORKDIR (/app), so the flat imports — `import config`,
# `from api ...` — and the `app:app` gunicorn target keep working unchanged.
COPY app/ ./

# Create non-root user and make sure the static dir is writable
# (twin_fwe adapter writes per-request output under /app/static/<timestamp>/)
RUN useradd --create-home --uid 1000 appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT}/Home" || exit 1

# Gunicorn: single worker by default because torch + lightgbm models are loaded
# per-process (lru_cache) and each replica costs RAM. Bump WEB_CONCURRENCY for
# more parallelism, and GUNICORN_TIMEOUT for slow predictions.
ENV WEB_CONCURRENCY=2 \
    GUNICORN_TIMEOUT=120

CMD exec gunicorn \
    --bind "0.0.0.0:${PORT}" \
    --workers "${WEB_CONCURRENCY}" \
    --threads 4 \
    --timeout "${GUNICORN_TIMEOUT}" \
    --access-logfile - \
    --error-logfile - \
    app:app
