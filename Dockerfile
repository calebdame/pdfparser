FROM python:3.10-slim

# Install system dependencies for PDF to image conversion and OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr libtesseract-dev poppler-utils \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Default command uses PORT env var if provided
CMD ["sh", "-c", "uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}"]
