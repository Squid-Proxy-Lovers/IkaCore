# syntax=docker/dockerfile:1.7

FROM python:3.12-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    file \
    git \
    ripgrep \
    tree \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

COPY --from=ikacore . /workspace/IkaCore
COPY --from=v8src . /workspace/v8

RUN python -m pip install --no-cache-dir --upgrade pip && \
    python -m pip install --no-cache-dir httpx

WORKDIR /workspace/IkaCore

CMD ["sleep", "infinity"]
