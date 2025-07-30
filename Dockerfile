# Multi-stage build for LLM Tab Cleaner
FROM python:3.11-slim as builder

# Set build arguments
ARG BUILD_DATE
ARG VERSION=0.1.0
ARG VCS_REF

# Add labels for better container management
LABEL org.opencontainers.image.title="LLM Tab Cleaner"
LABEL org.opencontainers.image.description="Production data cleaning with language models"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${VCS_REF}"
LABEL org.opencontainers.image.source="https://github.com/terragonlabs/llm-tab-cleaner"
LABEL org.opencontainers.image.licenses="MIT"

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    POETRY_VENV_IN_PROJECT=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd --gid 1000 llmuser && \
    useradd --uid 1000 --gid llmuser --shell /bin/bash --create-home llmuser

# Set work directory
WORKDIR /app

# Copy dependency files
COPY pyproject.toml README.md ./
COPY src/ ./src/

# Install Python dependencies
RUN pip install --no-cache-dir -e ".[all]"

# Production stage
FROM python:3.11-slim as production

# Copy non-root user from builder
COPY --from=builder /etc/passwd /etc/passwd
COPY --from=builder /etc/group /etc/group

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/home/llmuser/.local/bin:$PATH"

# Set work directory
WORKDIR /app

# Copy installed packages and application
COPY --from=builder --chown=llmuser:llmuser /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder --chown=llmuser:llmuser /usr/local/bin /usr/local/bin
COPY --from=builder --chown=llmuser:llmuser /app /app

# Switch to non-root user
USER llmuser

# Create directories for data and logs
RUN mkdir -p /app/data /app/logs /app/temp

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from llm_tab_cleaner import TableCleaner; print('OK')" || exit 1

# Default command
CMD ["llm-clean", "--help"]

# Development stage
FROM builder as development

# Install development dependencies
RUN pip install --no-cache-dir -e ".[dev,test]"

# Install additional development tools
RUN apt-get update && apt-get install -y \
    vim \
    htop \
    && rm -rf /var/lib/apt/lists/*

# Switch to non-root user
USER llmuser

# Set default command for development
CMD ["/bin/bash"]

# Spark-enabled stage  
FROM production as spark

# Switch back to root for Spark installation
USER root

# Install Java for Spark
RUN apt-get update && apt-get install -y \
    openjdk-11-jre-headless \
    && rm -rf /var/lib/apt/lists/*

# Install Spark dependencies
RUN pip install --no-cache-dir pyspark delta-spark

# Set Spark environment variables
ENV JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64
ENV SPARK_HOME=/usr/local/lib/python3.11/site-packages/pyspark
ENV PATH="$SPARK_HOME/bin:$PATH"

# Switch back to non-root user
USER llmuser

# Default command for Spark variant
CMD ["llm-clean", "--help"]