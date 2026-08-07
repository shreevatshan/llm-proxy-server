# syntax=docker/dockerfile:1

# Use Python 3.11 slim image for efficiency
FROM python:3.11-slim

# Build-time proxy support. buildx automatically forwards the predefined
# http_proxy/https_proxy build args to each RUN, so declaring the ARGs is enough
# for network access during build. They are intentionally NOT re-exported as ENV
# so the proxy URL never persists into the image config / `docker history`.
ARG http_proxy
ARG https_proxy

# Set by buildx per target platform (e.g. linux/arm64); used to keep the pip
# download cache below separate per architecture.
ARG TARGETPLATFORM

# Rotated weekly by docker-image-push.sh (ISO year+week). It is part of the pip
# cache mount id, and therefore of the pip layer's cache key, so bumping it both
# forces a fresh dependency install and starts a fresh download cache. See the
# comment block in docker-image-push.sh for how it is rotated and cleaned up.
ARG PIP_CACHE_EPOCH=0

# Set working directory
WORKDIR /llm-proxy-server

# No system build toolchain is installed on purpose. Every dependency in
# requirements.txt (and its transitive tree) publishes prebuilt manylinux
# wheels for both amd64 and arm64, so nothing has to be compiled. Installing
# gcc here used to cost ~12 minutes per arm64 build under QEMU emulation.

# Copy requirements first for better layer caching
COPY requirements.txt .

# --only-binary=:all: keeps the "no compiler needed" invariant enforced: if a
# dependency ever ships without a wheel for one of the target arches, the build
# fails here with a clear message instead of silently trying (and failing) to
# compile it. The one exception is cuid (pulled in by traceloop-sdk), which is
# sdist-only but pure Python, so it needs no toolchain to build.
# The cache mount keeps downloaded wheels across builds without adding a layer.
RUN --mount=type=cache,target=/root/.cache/pip,id=pip-${TARGETPLATFORM}-${PIP_CACHE_EPOCH},sharing=locked \
    pip install --only-binary=:all: --no-binary=cuid -r requirements.txt

# Copy application files
COPY app/ ./app/
COPY run.py .

# Create a non-root user and the runtime data/logs directories, owned by that
# user so the app can write them (and bind mounts land as appuser, not root).
RUN useradd -r -u 1000 -d /llm-proxy-server appuser \
    && mkdir -p data logs \
    && chown -R appuser:appuser /llm-proxy-server

# Drop privileges: run the application as the non-root user
USER appuser

# Run the application
CMD ["python", "run.py"]
