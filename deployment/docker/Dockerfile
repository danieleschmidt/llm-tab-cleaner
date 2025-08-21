# Production Dockerfile for LLM Tab Cleaner
FROM python:3.9-alpine AS builder

# Install build dependencies
RUN apk add --no-cache gcc musl-dev postgresql-dev

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Production stage
FROM python:3.9-alpine

# Create non-root user
RUN addgroup -g 1001 appgroup && \
    adduser -D -u 1001 -G appgroup appuser

# Install runtime dependencies
RUN apk add --no-cache postgresql-libs libffi

# Copy virtual environment
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appgroup src/ /app/src/
COPY --chown=appuser:appgroup pyproject.toml /app/

WORKDIR /app
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Start application
CMD ["python", "-m", "llm_tab_cleaner.api"]
