# Use Python 3.11 slim image for efficiency
FROM python:3.11-slim

ARG http_proxy
ARG https_proxy
ENV http_proxy=$http_proxy
ENV https_proxy=$https_proxy

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


# Create data directory - Python code will handle permissions automatically
RUN mkdir -p data

# Unset proxy for application runtime
ENV http_proxy=""
ENV https_proxy=""

# Run the application
CMD ["python", "run.py"]
