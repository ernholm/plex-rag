FROM python:3.11-slim

WORKDIR /app

# Set environment variable to disable NumPy CPU optimization requirements
ENV OPENBLAS=0
ENV MKL_THREADING_LAYER=GNU

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

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

# Create data directory for ChromaDB
RUN mkdir -p chroma_db

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the application
CMD ["python", "main.py"]
