# Stage 1: Build frontend
FROM node:20-alpine AS frontend-builder

WORKDIR /frontend

# Copy frontend package files first (better caching)
COPY frontend/package*.json ./
RUN npm install

# Copy frontend source and build
COPY frontend/ ./
RUN npm run build

# Stage 2: Python backend with static frontend
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies (including OCR for scanned PDFs)
RUN apt-get update && apt-get install -y \
    curl \
    ocrmypdf \
    tesseract-ocr \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml for install (deps defined there)
COPY backend/pyproject.toml backend/README.md* ./
RUN mkdir -p ragbrain && touch ragbrain/__init__.py

# Install package with all dependencies in one step
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -e ".[redis]"

# NOW copy application code (this layer rebuilds on code changes, but deps are cached)
COPY backend/ragbrain ./ragbrain/

# Copy built frontend from stage 1
COPY --from=frontend-builder /frontend/dist /app/static

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run application
CMD ["uvicorn", "ragbrain.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
