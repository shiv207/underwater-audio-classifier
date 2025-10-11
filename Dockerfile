# Dockerfile for Underwater Acoustic Classification System
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsndfile1 \
    ffmpeg \
    libfftw3-dev \
    pkg-config \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ ./src/
COPY main.py .
COPY models/ ./models/

# Create data and results directories
RUN mkdir -p data results

# Set environment variables
ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

# Default command
CMD ["python", "main.py", "--help"]

# Usage examples:
# Build: docker build -t uda-classifier .
# Run inference: docker run -v /path/to/data:/app/data -v /path/to/results:/app/results uda-classifier python main.py --input data/audio.wav --output results/results.json
# Run evaluation: docker run -v /path/to/data:/app/data uda-classifier python main.py --evaluate --ground-truth data/gt.json --predictions data/pred.json
