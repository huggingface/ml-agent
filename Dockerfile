# Stage 1: Build frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json frontend/package-lock.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Production
FROM python:3.12-slim

# Install uv directly from official image
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Create user with UID 1000 (required for HF Spaces)
RUN useradd -m -u 1000 user

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install dependencies into /app/.venv
# Use --frozen to ensure exact versions from uv.lock
RUN uv sync --extra agent --no-dev --frozen

# Copy application code
COPY agent/ ./agent/
COPY backend/ ./backend/
COPY configs/ ./configs/

# Copy built frontend
COPY --from=frontend-builder /app/frontend/dist ./static/

# Create directories and set ownership
RUN mkdir -p /app/session_logs && \
    chown -R user:user /app

# Switch to non-root user
USER user

# Set environment
ENV HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/app/.venv/bin:$PATH"

# Expose port
EXPOSE 7860

# Run the application from backend directory
WORKDIR /app/backend
CMD ["bash", "start.sh"]
