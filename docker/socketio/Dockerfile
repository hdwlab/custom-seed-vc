FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-devel

# Set working directory
WORKDIR /app

# Set environment variables for non-interactive installation
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Tokyo

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    portaudio19-dev \
    libasound2-dev \
    libportaudio2 \
    libpulse-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:0.10.6 /uv /uvx /usr/local/bin/
ENV UV_LINK_MODE=copy

# Copy uv configuration files
COPY pyproject.toml uv.lock ./
# Copy README.md (required for package metadata)
COPY README.md ./

# Install dependencies (main + client groups, without dev)
RUN uv sync --locked --no-dev --no-install-project
ENV PATH="/app/.venv/bin:$PATH"

# Copy source code
COPY seed_vc/ ./seed_vc/
COPY configs/ ./configs/
COPY assets/ ./assets/

# Create cache directory for HuggingFace models
RUN mkdir -p /root/.cache/huggingface

# Expose port
EXPOSE 5000

# Set environment variables with defaults
ENV LOG_LEVEL=INFO

# Command to run the server
CMD python -m seed_vc.socketio.server --host 0.0.0.0 --port 5000 --log-level $LOG_LEVEL
