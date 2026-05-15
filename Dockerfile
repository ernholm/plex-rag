FROM ubuntu:22.04

WORKDIR /app

# Disable NumPy CPU optimization requirements
ENV NPY_DISABLE_CPU_FEATURES=AVX2,AVX512F
ENV OPENBLAS_CORETYPE=NEHALEM

# Install Python and dependencies
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3-pip \
    python3-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install NumPy first with specific version
RUN pip install --upgrade pip && \
    pip install --upgrade "numpy>=1.21.0,<1.24.0" --only-binary :all:

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY main.py .
COPY indexer.py .

# Create static directory and copy UI
RUN mkdir -p static
COPY static/ static/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python3 -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["python3", "main.py"]
