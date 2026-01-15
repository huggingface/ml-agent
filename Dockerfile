# Stage 1: Build frontend
FROM node:20-alpine AS frontend-builder
WORKDIR /app/frontend
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

# Stage 2: Production
FROM python:3.12-slim

# Create user with UID 1000 (required for HF Spaces)
RUN useradd -m -u 1000 user

WORKDIR /app

# Install system dependencies and uv
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && curl -LsSf https://astral.sh/uv/install.sh | sh \
    && rm -rf /var/lib/apt/lists/*

ENV PATH="/root/.local/bin:$PATH"

# Copy pyproject.toml and install dependencies with uv
COPY pyproject.toml .
RUN uv sync --extra agent --no-dev

# Copy application code with proper ownership
COPY --chown=user agent/ ./agent/
COPY --chown=user backend/ ./backend/
COPY --chown=user configs/ ./configs/

# Copy built frontend
COPY --from=frontend-builder --chown=user /app/frontend/dist ./static/

# Create directories for session logs
RUN mkdir -p /app/session_logs && chown user:user /app/session_logs

# Switch to non-root user
USER user

# Set environment
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

# Expose port
EXPOSE 7860

# Run the application from backend directory
WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
