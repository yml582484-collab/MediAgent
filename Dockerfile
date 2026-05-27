# ============================================================
# Stage 1: Builder - Install Python dependencies
# ============================================================
FROM python:3.11-slim AS builder

LABEL maintainer="MediAgent Team"
LABEL stage="builder"

WORKDIR /build

# Install system dependencies required for building Python packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Create and activate virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python build tools first (for packages that need compilation)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir -r requirements.txt

# ============================================================
# Stage 2: Runtime - Minimal production image
# ============================================================
FROM python:3.11-slim AS runtime

LABEL maintainer="MediAgent Team"
LABEL description="MediAgent - AI Medical Assistant Service"
LABEL version="2.1.0"
LABEL org.opencontainers.image.source="https://github.com/mediagent/mediagent"
LABEL org.opencontainers.image.title="MediAgent"
LABEL org.opencontainers.image.description="Production container for MediAgent medical AI assistant"

WORKDIR /app

# Install only runtime system dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        curl \
        nodejs \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r mediagent && \
    useradd -r -g mediagent -d /app -s /sbin/nologin mediagent

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY src/ ./src/
COPY configs/ ./configs/
COPY pyproject.toml .

# Create required directories with proper permissions
RUN mkdir -p /app/data/chromadb /app/logs /app/workspace && \
    chown -R mediagent:mediagent /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app \
    DEEPSEEK_API_BASE=https://api.deepseek.com/v1 \
    LOG_LEVEL=INFO

# Switch to non-root user
USER mediagent

# Expose application port
EXPOSE 8000

# Health check using curl
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/api/health || exit 1

# Start the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
