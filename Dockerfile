# Use Python 3.11 slim image for efficiency
FROM python:3.11-slim

# Build-time proxy support. buildx automatically forwards the predefined
# http_proxy/https_proxy build args to each RUN, so declaring the ARGs is enough
# for network access during build. They are intentionally NOT re-exported as ENV
# so the proxy URL never persists into the image config / `docker history`.
ARG http_proxy
ARG https_proxy

# Set working directory
WORKDIR /llm-proxy-server

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better layer caching
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

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
