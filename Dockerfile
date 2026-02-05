# Optional Dockerfile for reproducibility (offline-friendly)
ARG PY_BASE=python:3.12-slim
FROM ${PY_BASE}

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
  && apt-get install -y --no-install-recommends bash ffmpeg ca-certificates \
  && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy minimal project files
COPY pyproject.toml README.md ./
COPY src ./src
COPY scripts ./scripts
COPY fixtures ./fixtures
COPY docs ./docs

RUN python -m pip install -U pip setuptools \
  && python -m pip install -e . \
  && python -m pip install requests

CMD ["bash", "scripts/verify_all.sh"]
